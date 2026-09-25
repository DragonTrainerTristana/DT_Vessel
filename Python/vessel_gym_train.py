"""
vessel_gym_train.py — GPU 배치 PPO 학습 (vessel_gym + networks.py 재사용).

networks.py의 CNNPolicy 서브모듈(RadarEncoder·ControlActor·Critic)을 그대로 써서 학습하므로
체크포인트가 Unity 쪽 CNNPolicy와 shape 호환 → 학습된 정책을 VESSEL_LOAD_MODEL=1로 Unity에 이식 가능.

arm:
  OFF    : others_msg ≡ 0 (통신 없음)
  ORACLE : others_msg[..., :GOAL_SIZE] = 파트너의 정규화 goal obs(d/(d+150), angle/180) nearest-4 평균
           (Unity networks.py oracle과 동일 의미·스케일. 원좌표 주입 금지 — 스케일 폭파로 붕괴함)
둘 다 others_msg가 정책 파라미터와 무관한 상수 입력 → PPO 업데이트에서 그대로 재사용(미러 불필요).
Stage 2(학습된 통신 ON)의 배치 집계는 별도 작업(networks.py 통신경로가 per-env dict라 배치화 필요).

사용:
  python vessel_gym_train.py --arm OFF --steps 1000000 --envs 1024 --seed 42
  python vessel_gym_train.py --arm ORACLE --steps 1000000 --envs 1024 --seed 42
"""
import os, sys, time, argparse, hashlib
import torch
import torch.nn as nn

import config as cfg
import vessel_gym as vg
import networks as net_mod          # 텔레메트리가 모듈 전역(_MSG_TOKEN_GAIN)을 읽는다
from networks import CNNPolicy

MSG_RANDOM_SD = cfg.MSG_RANDOM_SD   # ★2026-09-10 RANDOM 팔 난수 sd. ckpt_io.restore_policy 가 스냅샷으로 덮어씀

GOAL_SIZE = cfg.GOAL_SIZE
FRAMES = cfg.FRAMES
STATE = cfg.STATE_SIZE       # 360
MSG_DIM = cfg.MSG_DIM
DEG_RAD = 3.141592653589793 / 180.0


def parse_obs(obs):
    """env obs [E,N,369] → radar[E,N,360], goal[E,N,2], self[E,N,4], situation[E,N] (long)."""
    radar = obs[..., 0:360]
    goal = obs[..., 360:362]
    self_s = obs[..., 362:366]
    situation = obs[..., 368].round().long().clamp(0, cfg.NUM_COLREGS_SITUATIONS - 1)
    return radar, goal, self_s, situation


class FrameStack:
    """radar 3프레임 스택 [E,N,FRAMES*360]. done 에이전트는 현재 프레임으로 초기화."""
    def __init__(self, E, N, device):
        self.buf = torch.zeros(E, N, FRAMES, STATE, device=device)
    def reset_all(self, radar):
        self.buf[:] = radar.unsqueeze(2)          # 3프레임 모두 현재로
    def push(self, radar, done):
        self.buf = torch.roll(self.buf, shifts=-1, dims=2)
        self.buf[:, :, -1, :] = radar
        # ★2026-09-26 `if done.any():` 가드 제거 — 매 스텝 GPU→CPU sync 1회였다. where 는 산술 없는 원소 선택이라
        #   done 이 전부 False 면 buf 값 그대로, 하나라도 True 면 예전과 같은 코드가 돈다 = 비트동일.
        #   (버퍼 객체는 roll 이 새로 만들고 push 뒤엔 안 건드리므로 rollout 버퍼의 x 뷰도 무관)
        d = done.unsqueeze(-1).unsqueeze(-1)       # done 에이전트는 3프레임 리셋
        self.buf = torch.where(d, radar.unsqueeze(2).expand_as(self.buf), self.buf)
    def get(self):
        E, N = self.buf.shape[0], self.buf.shape[1]
        return self.buf.reshape(E, N, FRAMES * STATE)


def make_others_msg(env, arm, E, N, device):
    """arm별 others_msg [E,N,MSG_DIM] (상수 입력 — 정책 파라미터와 무관해 update 에서 그대로 재사용).

    ★RANDOM (난수 메시지 대조군, 2026-09-05 구현):
      메시지 자리에 *내용 없는 난수*를 넣고 학습시키는 통제군. `ABLATION_PLAN.md` §3-b ③ 이
      "심사자가 먼저 물어볼 대조군"으로 지목했으나 미구현이던 항목이다.
      가르는 것: 통신 ON 이 OFF 를 이겼을 때 그 이득이 **메시지에 담긴 정보** 때문인지,
      아니면 **메시지 경로가 붙으며 늘어난 파라미터·gradient 경로**(정규화 효과) 때문인지.
        난수 ≈ OFF  < 진짜메시지  → 이득은 정보에서 옴 (통신 주장 성립)
        난수 ≈ 진짜메시지          → 이득은 구조에서 옴 (통신이라 부를 수 없음)
      ⚠️PPO ratio 안전성: 난수는 파트너 obs 의 함수가 아니므로 sender→receiver gradient 가 없다.
        그래서 OFF·ORACLE 과 같은 *상수 입력* 경로로 넣는다. rollout 이 만든 난수를 버퍼('om')에
        저장하고 update 가 그 값을 그대로 쓰므로 rollout=update 가 **구성적으로** 보장된다
        (별도 미러 검증이 필요 없다). 수신측 fc2 메시지 슬라이스 파라미터와 그 gradient 경로는
        진짜 메시지 팔과 동일하게 살아 있다 = 파라미터 효과만 남긴 통제.
      ⚠️분포 정합: 난수의 스케일은 비교 대상 팔의 실측 others_msg 표준편차에 맞춰야 공정하다.
        `_diag_msg_channel.py(→_archive, 현행 diag_ckpt.py)` 가 찍는 `om_sd` 를 읽어 `VESSEL_MSG_RANDOM_SD` 로 주입할 것.
        기본 0.20 은 m2 C1~C6 실측 om_sd 0.14~0.23 의 중앙 근처값이다(runs/m2_ablation/diag/channel.jsonl).
    """
    om = torch.zeros(E, N, MSG_DIM, device=device)
    if arm == 'ORACLE':
        pg = env.partner_goals_oracle()            # [E,N,2]
        om[..., :GOAL_SIZE] = pg
    elif arm == 'RANDOM':
        sd = MSG_RANDOM_SD          # 모듈 전역 (config 기본값; ckpt_io 가 스냅샷으로 덮어씀)
        # 균등분포로 tanh 와 같은 유계 지지집합을 유지하되 표준편차를 sd 에 맞춘다: U(-a,a), a = sd*sqrt(3)
        a = sd * 1.7320508075688772
        om = (torch.rand(E, N, MSG_DIM, device=device) * 2.0 - 1.0) * a
    elif arm not in ('OFF', 'ON'):
        # ★2026-09-07: 예전엔 else 가 없어 모르는 arm 이 *에러 없이* zeros(=OFF)로 학습됐다.
        #   argparse choices 가 유일한 방어선이었는데 diag_timeout.py(→_archive) 는 choices 조차 없다.
        raise ValueError(f"make_others_msg: 모르는 arm '{arm}' — OFF/ON/ORACLE/RANDOM 중 하나여야 함")
    return om


def batched_gae(rewards, values, dones, truncs, last_value, gamma, lam):
    """rewards/values/dones/truncs [T,E,N], last_value [E,N] → returns, adv [T,E,N].
    ★dones = 모든 종료(goal/collision/timeout, =에피소드 경계). truncs = 그중 timeout 절단만.
      절단은 자기 value로 bootstrap(미래가치 유지 — timeout이 60~80%라 이 구분이 value target 지배),
      진짜 종료(goal/collision)는 bootstrap 0. 둘 다 경계에서 GAE 역전파 절단."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    next_v = last_value
    for t in reversed(range(T)):
        term = dones[t] * (1.0 - truncs[t])                       # 진짜 종료만 1 (절단 제외)
        boot = truncs[t] * values[t] + (1.0 - truncs[t]) * next_v * (1.0 - term)
        cont = 1.0 - dones[t]                                     # 경계(종료·절단)면 역전파 절단
        delta = rewards[t] + gamma * boot - values[t]
        gae = delta + gamma * lam * cont * gae
        adv[t] = gae
        next_v = values[t]
    returns = adv + values
    return returns, adv


class ValueNorm:
    """리턴 running 정규화 (MAPPO value normalization, 편향보정 EMA).

    ★왜 필요한가 (2026-08-30 실측): 리턴이 평균 153·표준편차 17 스케일이라 value_loss 가 422 까지 커지고,
      critic gradient 혼자 norm 10,980 을 만든다. clip_grad_norm_ 은 policy.parameters() *전체*를 한 덩어리로
      자르므로(MAX_GRAD_NORM=0.5) actor 도 같이 1/22,500 로 눌린다. 더 나쁜 건 방향이다 — 통신 ON 이면
      critic 이 others_msg 를 통해 MessageActor 로도 gradient 를 보내는데, 그 크기가 policy_loss 대비
      1775.6 : 0.058 = 30,600 : 1 이었다. 즉 메시지가 '받는 배가 잘 행동하게'가 아니라 '가치를 잘 맞히게'
      학습됐다. 정규화 후 같은 비가 105 : 1 로 떨어진다(실측).
    ★누적(Welford) 대신 EMA 인 이유: 학습이 진행되면 리턴 분포가 이동한다(정책이 좋아지므로). 무한누적이면
      count 가 수백만이 돼 후반부에 통계가 사실상 동결되고, 목표값이 정규화 공간에서 다시 발산한다.
      beta=0.98 → 유효창 ~50 update = 표류는 따라가되 목표는 안정.
    ⚠️'50 update 가 학습의 몇 %인가'는 --envs 에 의존한다. 총 update 수 = steps / (envs·vessels·rollout).
      --envs 128(RUNS.md 표준): 16M / 65,536 = 244 update → 유효창 20%  ← beta=0.98 은 이 값 기준으로 잡았다
      --envs  32            : 16M / 16,384 = 977 update → 유효창  5%  (통계가 4배 반응적 = 불안정)
      envs 를 줄여 쓸 거면 beta 를 함께 올려야 같은 유효창이 된다(32 라면 0.995 ≈ 200 update ≈ 20%).
      VESSEL_VALNORM_BETA 로 조정 가능.
    critic 은 *정규화 공간*의 값을 출력한다 → GAE·bootstrap 은 denormalize 해서 원 스케일로 쓰고,
      value_loss 만 정규화 공간에서 계산한다. 체크포인트에 통계를 함께 저장(이어학습·분석용).
    """
    def __init__(self, device, beta=None, eps=1e-6):
        self.beta = cfg.VALNORM_BETA if beta is None else beta
        self.eps = eps
        self.m1 = torch.zeros((), device=device)     # EMA of E[x]   (편향 있음)
        self.m2 = torch.zeros((), device=device)     # EMA of E[x^2] (편향 있음)
        self.debias = torch.zeros((), device=device) # 편향보정 분모

    def update(self, x):
        b = self.beta
        xm = x.mean().detach()
        xs = (x * x).mean().detach()
        self.m1 = self.m1 * b + xm * (1.0 - b)
        self.m2 = self.m2 * b + xs * (1.0 - b)
        self.debias = self.debias * b + (1.0 - b)

    def _stats(self):
        d = self.debias.clamp(min=self.eps)
        mean = self.m1 / d
        var = (self.m2 / d - mean * mean).clamp(min=self.eps)
        return mean, var.sqrt()

    def normalize(self, x):
        m, s = self._stats()
        return (x - m) / s

    def denormalize(self, x):
        m, s = self._stats()
        return x * s + m

    def state(self):
        m, s = self._stats()
        return {'m1': float(self.m1), 'm2': float(self.m2), 'debias': float(self.debias),
                'beta': self.beta, 'mean': float(m), 'std': float(s)}

    def load(self, d):
        if not d:
            return
        for k in ('m1', 'm2', 'debias'):
            setattr(self, k, torch.as_tensor(d[k], device=self.m1.device, dtype=self.m1.dtype))
        self.beta = d.get('beta', self.beta)


def compute_own_future(pos, head_deg, done, intent_k, horizon, pos_scale):
    """★intent 미래라벨 (memory.py:220~ 벡터화 이식). 각 스텝 t 의 미래 K 지점 변위를
    t 시점 body frame 으로 회전. done 경계를 넘으면 mask=0 (respawn 텔레포트 누설 방지).

    pos [T,E,N,2] · head_deg [T,E,N] · done [T,E,N] (bool, 그 스텝에서 종료)
    Returns fut, fut_mask 둘 다 [T,E,N,K*3] = K 개 [국소 우현변위, 국소 전방변위, Δ침로]

    ⚠️memory.py 와 동일 규약: 회전은 ch=cos(h), sh=sin(h) 로
      loc_star = dx*ch - dz*sh,  loc_fwd = dx*sh + dz*ch  (Unity atan2(x,z) 규약)
      Δ침로는 wrap[-180,180] 후 /180, 변위는 /INTENT_POS_SCALE.
    ⚠️버퍼 끝을 넘는 미래(j>=T)는 mask=0. memory.py 의 `break` 와 동치.
    """
    T = pos.shape[0]
    hr = head_deg * DEG_RAD
    ch, sh = torch.cos(hr), torch.sin(hr)                       # [T,E,N]
    # dn_cum[t] = t 까지의 누적 종료 수 → 구간 (t, j) 에 종료가 있었는지를 차이로 판정
    dn_cum = torch.cumsum(done.to(pos.dtype), dim=0)            # [T,E,N]
    outs, masks = [], []
    for k in range(intent_k):
        h = (k + 1) * horizon
        f = torch.zeros_like(pos[..., 0])                       # [T,E,N]
        star = torch.zeros_like(f); fwd = torch.zeros_like(f); dh = torch.zeros_like(f)
        m = torch.zeros_like(f)
        if h < T:
            src = slice(0, T - h); dst = slice(h, T)
            d = pos[dst] - pos[src]                             # [T-h,E,N,2]
            dx, dz = d[..., 0], d[..., 1]
            star[src] = dx * ch[src] - dz * sh[src]
            fwd[src] = dx * sh[src] + dz * ch[src]
            _dh = head_deg[dst] - head_deg[src]
            dh[src] = torch.remainder(_dh + 180.0, 360.0) - 180.0
            # 구간 [t, t+h) 에 종료가 하나도 없어야 유효 (memory.py 의 dn[t:j].any() 와 동치)
            # sum(done[t .. j-1]) = dn_cum[j] - done[j] - dn_cum[t] + done[t]
            #   (memory.py 의 dn[t:j].any() 와 동치 — 시작점 t 를 *포함*한다)
            crossed = (dn_cum[dst] - done[dst].to(pos.dtype)
                       - dn_cum[src] + done[src].to(pos.dtype))
            m[src] = (crossed <= 0).to(pos.dtype)
        outs.append(torch.stack([star / pos_scale, fwd / pos_scale, dh / 180.0], dim=-1) * m.unsqueeze(-1))
        masks.append(m.unsqueeze(-1).expand(-1, -1, -1, 3))
    return torch.cat(outs, dim=-1), torch.cat(masks, dim=-1)


_THREAT_ANG = None
def build_global_feat(env):
    """★중앙 critic(CTDE) 전역 상태: 배 i에게 모든 배 j의 (상대위치/COMM_R, sin침로, cos침로, 속도비, 거리/COMM_R).
    [E,N,N,6]. j=i 행은 상대위치 0 + 자기 운동학 = 무해. critic 전용(학습때만) — actor 는 국소 유지."""
    pos, hd = env.pos, env.heading * DEG_RAD
    rel = (pos.unsqueeze(1) - pos.unsqueeze(2)) / cfg.COMM_RANGE          # [E,i,j,2] = pos_j - pos_i
    d = rel.norm(dim=-1, keepdim=True)                                    # [E,i,j,1]
    sh, ch = torch.sin(hd), torch.cos(hd)
    spd = env.speed / torch.clamp(env.max_speed, min=1e-6)
    kin = torch.stack([sh, ch, spd], dim=-1)                              # [E,j,3]
    E, N = pos.shape[0], pos.shape[1]
    kin = kin.unsqueeze(1).expand(E, N, N, 3)
    return torch.cat([rel, kin, d], dim=-1)                               # [E,N,N,6]


def compute_own_threat(x, threat_k, device):
    """★threat-relay 라벨(GPU 이식, memory.py L172~ 미러): 각 배의 ego-radar(frame-stack 최신 프레임)에서
    top-K 최근접 위협 기하 [sin방위,cos방위,거리,closing] 추출. → 메시지가 '내가 본 위협'을 인코딩하도록
    self-supervised(THREAT_COEF). occlusion으로 가린 위협은 안 잡힘=정직. 미감지 슬롯은 mask=0.
    x [M, FRAMES*STATE]. Returns own_threat [M, K*4], mask [M, K*4]."""
    global _THREAT_ANG
    M = x.shape[0]
    stacked = x.reshape(M, FRAMES, STATE)
    cur = stacked[:, -1, :] + 0.5              # [M, 360] 현재 정규화 거리 ∈[0,1]
    prev = stacked[:, -2, :] + 0.5             # 직전(closing 계산)
    if _THREAT_ANG is None or _THREAT_ANG[0].shape[0] != STATE:
        ang = torch.deg2rad(torch.arange(STATE, dtype=torch.float32, device=device))
        _THREAT_ANG = (torch.sin(ang), torch.cos(ang))
    sin_a, cos_a = _THREAT_ANG
    dist_k, idx = torch.topk(cur, threat_k, dim=-1, largest=False)   # [M,K] 최근접 K
    prev_k = torch.gather(prev, 1, idx)
    closing = prev_k - dist_k                                        # 양수=접근
    sin_k = sin_a[idx]; cos_k = cos_a[idx]
    detect = (dist_k < 0.999).float().unsqueeze(-1)                  # 미감지(≈1.0) 제외
    thr = torch.stack([sin_k, cos_k, dist_k, closing], dim=-1) * detect   # [M,K,4]
    mask = detect.expand(-1, -1, 4)
    return thr.reshape(M, threat_k * 4), mask.reshape(M, threat_k * 4)


def comm_gather(policy, env, x, goal, self_s, sit, K, send_mask=None, recv_mask=None,
                msg_override=None, return_dist=False, groups=None, ext_shuffle_gen=None):
    """★배치 학습형 comm (2026-08): 각 배의 COMM_RANGE 내 nearest-K 파트너 메시지를 pos_ground 집계.
    evaluate_actions(update)의 pos_ground 분기와 *동일 함수형* → PPO ratio 유효 (mirror 검증 대상).
    Returns:
      others_msg [E,N,MSG_DIM]  — acting(ctr_actor/critic)용
      partner tensors (px,pg,ps,pmask,prelpos,psit) [E,N,K,·] — update evaluate_actions용(sender→receiver grad)
    x/goal/self_s [E,N,dim], sit [E,N] long. env.pos [E,N,2], env.heading [E,N] deg.
    send_mask/recv_mask [N] 또는 [E,N] bool — 혼합 함대 평가 전용
      (기본 None=전원 통신, 학습 동작 불변). [E,N]을 쓰면 환경마다 다른 비율을 동시에 평가할 수 있다.
      send_mask=False인 배는 파트너 후보에서 빠져 아무도 그 배의 메시지를 못 받는다.
      recv_mask=False인 배는 받은 메시지가 0이 된다(통신 OFF 팔과 동일 입력).
    msg_override [E,N,MSG_DIM] — 텔레메트리 전용. 주면 msg_actor 를 부르지 않고 이 메시지로 집계한다
      (메시지를 0/섞기 로 바꿔 수신자 민감도를 재는 데 씀). 기본 None = 학습 동작 불변.
    return_dist — True 면 (others_msg, partners, topd) 3-튜플. 기본 False = 반환 형태 불변.
    ★COMM_EXT (2026-09-25): policy.relpos_dim > 3 이면 prelpos 뒤에 파트너 확장필드 20차원(vg.comm_pair_features)을 붙인다.
      켜는 그룹은 net_mod.COMM_GROUPS(스냅샷이 덮어씀). groups 인자 = 평가·텔레메트리 절제 전용 override.
      ext_shuffle_gen(torch.Generator) = 평가 전용 field-shuffle(같은 env 안 다른 수신자의 필드로 바꿔 정보만 끊음).
      파트너 선택 반경 = net_mod.PARTNER_RANGE(None 이면 COMM_RANGE). relpos 거리 정규화는 항상 COMM_RANGE."""
    E, N = x.shape[0], x.shape[1]
    dev = x.device
    pos = env.pos                                              # [E,N,2]
    hdg = env.heading                                          # [E,N] deg
    COMM_R = cfg.COMM_RANGE
    _pr = net_mod.PARTNER_RANGE                                # ★2026-09-25: 파트너 반경(ARPA@56 = 56). 보상 반경과 분리
    PART_R = COMM_R if _pr is None else float(_pr)
    d = torch.cdist(pos, pos)                                  # [E,N,N]
    BIG = 1e9
    d = d + torch.eye(N, device=dev).unsqueeze(0) * BIG        # 자기 제외
    d = torch.where(d <= PART_R, d, torch.full_like(d, BIG))   # 범위 밖 제외
    if send_mask is not None:                                  # 송신 불가 선박은 파트너 후보에서 제외
        _sm = send_mask.view(1, 1, N) if send_mask.dim() == 1 else send_mask.view(E, 1, N)
        d = torch.where(_sm, d, torch.full_like(d, BIG))
    Kc = min(K, N - 1)
    topd, topi = torch.topk(d, Kc, dim=-1, largest=False)      # [E,N,Kc]
    pmask = (topd < BIG).float().unsqueeze(-1)                 # [E,N,Kc,1] 유효 파트너
    if recv_mask is not None:
        # 수신 불가 선박은 파트너를 하나도 못 가진 것으로 처리 → others_msg가 0이 된다.
        # ★others_msg를 나중에 0으로 곱하지 않고 여기서 pmask를 지우는 이유:
        #   pmask는 rollout 버퍼에 저장돼 PPO 업데이트의 evaluate_actions가 그대로 다시 쓴다.
        #   여기서 지워야 rollout과 update의 others_msg가 똑같아지고 PPO ratio가 유효하다.
        _rm = recv_mask.view(1, N, 1, 1) if recv_mask.dim() == 1 else recv_mask.view(E, N, 1, 1)
        pmask = pmask * _rm.to(pmask.dtype)
    b = torch.arange(E, device=dev)[:, None, None]            # [E,1,1]
    px = x[b, topi]                                            # [E,N,Kc,F*S]
    pg = goal[b, topi]                                         # [E,N,Kc,2]
    ps = self_s[b, topi]                                       # [E,N,Kc,4]
    psit = sit[b, topi]                                        # [E,N,Kc]
    ppos = pos[b, topi]                                        # [E,N,Kc,2]
    # relpos (수신자 body frame): [sin(bearing), cos(bearing), dist/COMM_R] — main._compute_relpos 규약
    dx = ppos[..., 0] - pos[..., 0:1]                          # [E,N,Kc]
    dz = ppos[..., 1] - pos[..., 1:2]
    bearing = torch.atan2(dx, dz) - hdg[..., None] * vg.DEG
    prelpos = torch.stack([torch.sin(bearing), torch.cos(bearing),
                           (topd / COMM_R).clamp(max=1.0)], dim=-1) * pmask   # [E,N,Kc,3], padding=0
    if policy.relpos_dim > 3:
        # ★의도·역할 통신 확장필드 (2026-09-25). ⚠️미러 규칙: 이 prelpos 텐서 *하나*를 아래 집계와 반환(버퍼 저장)에 같이 쓴다.
        #   update(evaluate_actions)는 저장값을 재사용하고 다시 계산하지 않으므로 rollout=update 가 구조적으로 성립한다.
        ext = vg.comm_pair_features(env, topi, PART_R)                        # [E,N,Kc,20]
        if ext_shuffle_gen is not None:
            # field-shuffle(평가 전용): 같은 env 안의 *유효* (수신자,슬롯) 항목끼리만 값을 돌린다 → 유효 슬롯 값의 모음(분포)은
            #   그대로, 짝(누가 누구에 대해)만 끊음. 무작위 순서로 늘어놓고 한 칸씩 당겨 받으므로 유효 항목이 2개 이상인 env 에서는
            #   자기 값으로 돌아오는 슬롯이 없다(derangement). 유효 1개뿐인 env 는 자기 값 유지(끊을 상대가 없음).
            #   ★2026-09-25 리뷰 M1/E1 수정: 예전엔 수신자 행을 통째로 섞어 패딩 슬롯의 반경 밖·자기 쌍 값이 유효 슬롯에 들어갔다.
            _vm = (pmask.squeeze(-1) > 0).reshape(-1)                          # [E*N*Kc]
            _F = _vm.nonzero(as_tuple=False).squeeze(-1)
            _M = int(_F.numel())
            if _M > 1:
                _env = torch.div(_F, N * Kc, rounding_mode='floor')
                _key = torch.rand(_M, generator=ext_shuffle_gen, dtype=torch.float64).to(dev)
                _ord = torch.argsort(_env.to(torch.float64) * 2.0 + _key)      # env 별로 묶고 그 안에서 무작위 순서
                _se = _env[_ord]
                _pos = torch.arange(_M, device=dev)
                _is_st = torch.ones(_M, dtype=torch.bool, device=dev); _is_st[1:] = _se[1:] != _se[:-1]
                _is_en = torch.ones(_M, dtype=torch.bool, device=dev); _is_en[:-1] = _se[:-1] != _se[1:]
                _st = torch.cummax(torch.where(_is_st, _pos, torch.zeros_like(_pos)), 0).values
                _nxt = torch.where(_is_en, _st, _pos + 1)                      # 같은 env 안에서 다음 항목(끝이면 처음으로)
                _flat = ext.reshape(E * N * Kc, -1)
                _src = _flat.clone()
                _flat[_F[_ord]] = _src[_F[_ord[_nxt]]]
                ext = _flat.reshape(ext.shape)
        _gs = net_mod.COMM_GROUPS if groups is None else tuple(groups)
        gmask = torch.zeros(ext.shape[-1], device=dev, dtype=ext.dtype)
        for _g in _gs:
            _a, _b = cfg.COMM_EXT_GROUPS[_g]
            gmask[_a:_b] = 1.0
        ext = torch.where(pmask > 0, ext * gmask, torch.zeros_like(ext))      # 패딩은 곱이 아니라 where(NaN×0 방지)
        prelpos = torch.cat([prelpos, ext], dim=-1)                            # [E,N,Kc,3+20]
    # others_msg: msg_actor(파트너) → msg_encoder(pos_ground) → masked mean (evaluate_actions 미러)
    prel_f = prelpos.reshape(E * N, Kc, -1); pmask_f = pmask.reshape(E * N, Kc, 1)
    # ★2026-08-31 중복 제거: 기존엔 파트너 obs 를 gather 한 뒤 msg_actor 를 E*N*Kc 개에 돌렸다.
    #   그런데 배 j 의 메시지는 j 의 obs 에만 의존하므로 수신자와 무관하다 → K(=4)배 중복 계산이었다.
    #   메시지를 배당 1회(E*N)만 만들고 gather 한다. networks._get_others_msg(Unity 경로)와 동일한 방식이며
    #   msg_actor 가 sample 단위 함수(MoE 라우팅도 자기 situation)라 결과는 수학적으로 동일하다.
    #   ⚠️px/pg/ps/psit 는 update(evaluate_actions)가 sender→receiver gradient 용으로 쓰므로 그대로 반환한다.
    msg_all = policy.msg_actor(x, goal, self_s, sit) if msg_override is None else msg_override
    msg_part = msg_all[b, topi].reshape(E * N, Kc, -1)                        # [E*N,Kc,MSG_DIM] gather
    # ★집계 3분기 — networks.evaluate_actions 의 분기와 *같은 순서·같은 함수형*이어야 PPO ratio 가 유효하다.
    #   우선순위: attention > pos_ground > sum/mean/scale. 마지막에 msg_gain 까지 동일하게 건다.
    #   ⚠️2026-09-05 fix (blocker): 이전에는 "attention 아니면 무조건 pos_ground(msg_encoder)" 였다.
    #     그래서 VESSEL_POS_GROUND=0 (sum/mean 대조군)으로 돌리면
    #        rollout = pos_ground(msg_encoder) / update = sum   으로 갈려 comm-ON 팔만 ratio 가 조용히 깨졌다.
    #     에러 없이 학습이 망가지므로 그 설정으로 돌린 gym 실행은 무효다. attention 분기 누락(09-04)과 같은 계열의 버그.
    #   ⚠️VESSEL_MSG_GAIN 도 update(networks.py)에만 걸려 있어 rollout 에 누락돼 있었다 — 여기서 같이 건다.
    Kcount = pmask_f.sum(dim=1, keepdim=True).clamp(min=1.0)                  # [E*N,1,1]
    # ★2026-09-10: update(networks.evaluate_actions)와 **같은 모듈 전역**을 읽는다(미러). 예전엔 각자 env 를 읽었다.
    agg_mode, nearest_scale, msg_gain = net_mod.AGG_MODE, net_mod.NEAREST_SCALE, net_mod.MSG_GAIN
    if nearest_scale > 0:
        agg_mode = 'scale'
    if getattr(policy, 'use_attention', False):
        # ★attention 집계 (2026-09-04): evaluate_actions의 attention 분기와 *같은 모듈·같은 함수형*
        #   (q=[self,goal], 토큰=[relpos⊕msg], aggregate_batch) → PPO ratio 유효.
        q_in = torch.cat([self_s, goal], dim=-1).reshape(E * N, 1, -1)        # [E*N,1,6]
        others_msg = policy.attn.aggregate_batch(q_in, prel_f, msg_part, pmask_f).reshape(E, N, -1)
    elif getattr(policy, 'pos_ground', False):
        localized = policy.msg_encoder(torch.cat([prel_f, msg_part], dim=-1))     # [E*N,Kc,MSG_DIM]
        others_msg = ((localized * pmask_f).sum(dim=1, keepdim=True) / Kcount).reshape(E, N, -1)
    else:
        s = (msg_part * pmask_f).sum(dim=1, keepdim=True)                     # [E*N,1,MSG_DIM]
        if agg_mode == 'mean':
            s = s / Kcount
        elif agg_mode == 'scale':
            s = s * (nearest_scale / Kcount)
        others_msg = s.reshape(E, N, -1)
    if msg_gain != 1.0:
        others_msg = others_msg * msg_gain
    if return_dist:
        return others_msg, (px, pg, ps, pmask, prelpos, psit), topd
    return others_msg, (px, pg, ps, pmask, prelpos, psit)


# ★통신 텔레메트리 (2026-09-08): 지금까지 매번 체크포인트를 다시 굴려서 재던 값들을 학습 로그에 남긴다.
#   학습 로그에 있던 것은 성능지표·보상·상태복원 5그룹뿐이라, 아래는 전부 사후에 재실행해야 했다.
#   VESSEL_COMM_TELEMETRY=1 일 때만 동작(기본 0 = 호출 자체가 없음 = 비트동일).
#   ⚠️학습 RNG 를 절대 건드리지 않는다: 전용 CPU Generator 로만 섞고, 표본이 아니라 평균행동(_route)을 쓴다.
COMM_TELE_COLS = ('msg_sd', 'msg_eff_dim', 'msg_axes90', 'msg_sat', 'msg_corr',
                  'gate_ctr', 'gate_cri',
                  'alpha_unif', 'alpha_dmsg', 'alpha_dpos', 'alpha_near',
                  'act_zero', 'act_shuf', 'read_ratio',
                  'part_n', 'part_med', 'part_out', 'enc_alive',
                  # ★2026-09-10 추가 (뒤에만 붙임): 구 _diag_msg_channel.py(→_archive, 현행 diag_ckpt.py) 의 지표 흡수 + 진단 게이트용
                  'msg_dc', 'threat_r2', 'om_erank', 'label_erank', 'sit_rate')
# ★COMM_EXT 런 전용 열 (2026-09-25, 뒤에만 붙임 — EXT=0 런은 헤더·값 불변). 수신자가 확장필드 그룹을 읽나:
#   alpha_dext = 확장필드만 파트너끼리 섞었을 때 α 총변동 / act_zero_<g> = 그 그룹만 0 으로 했을 때 조타 변화(자기 sd 대비)
COMM_TELE_EXT_COLS = ('alpha_dext', 'act_zero_ext', 'act_zero_state', 'act_zero_role', 'act_zero_intent')


def comm_tele_cols(policy):
    """이 정책의 텔레메트리 열. EXT 정책이면 기존 열 뒤에 COMM_TELE_EXT_COLS 를 붙인다."""
    return COMM_TELE_COLS + (COMM_TELE_EXT_COLS if getattr(policy, 'relpos_dim', 3) > 3 else ())


def msg_stats(mf):
    """메시지 [M,D] → 산포·유효차원·90%축수·포화·축간상관·DC비중.
    comm_telemetry 와 diag_ckpt(조우/비조우 분리)가 **같은 정의**를 쓰기 위한 유일한 구현 (2026-09-10).
      sd      : 차원별 std 의 평균
      eff_dim : 참여비 (Σλ)²/Σλ² — 분산이 몇 방향에 실렸나
      axes90  : 분산 90% 를 설명하는 축 수
      sat     : |msg|>0.99 비율 (tanh 포화)
      corr    : 축간 절대상관 평균
      dc      : |E[msg]|² / E[|msg|²] — 2차 모멘트 중 상수(평균 벡터) 몫. 1 에 가까우면 '거의 상수'
    """
    out = {}
    out['sd'] = float(mf.std(0).mean())
    out['sat'] = float((mf.abs() > 0.99).float().mean())
    mc = (mf - mf.mean(0, keepdim=True)).double()
    C = (mc.T @ mc) / max(1, mc.shape[0] - 1)
    ev = torch.linalg.eigvalsh(C).clamp(min=0).flip(0)
    out['eff_dim'] = float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-30))
    cum = torch.cumsum(ev, 0) / ev.sum().clamp(min=1e-30)
    out['axes90'] = float((cum < 0.90).sum() + 1)
    sdv = torch.sqrt(torch.diag(C)).clamp(min=1e-12)
    R = C / (sdv[:, None] * sdv[None, :])
    d_ = R.shape[0]
    out['corr'] = float(R[~torch.eye(d_, dtype=torch.bool, device=R.device)].abs().mean())
    md = mf.double()
    mu2 = float(md.mean(0).pow(2).sum()); e2 = float(md.pow(2).mean(0).sum())
    out['dc'] = (mu2 / e2) if e2 > 0 else float('nan')
    return out


def comm_telemetry(policy, env, x, goal, self_s, sit, K, gen, radar_range):
    """통신 경로를 한 번에 진단한다. 모두 no_grad·평균행동 → 학습에 영향 없음.

    msg_*      메시지 자체: 산포 / 유효차원(참여비) / 분산 90% 축수 / tanh 포화 / 축간 상관
    gate_*     ControlActor·Critic 의 sigmoid(msg_gate) 평균. 0=차단 1=그대로 통과
    alpha_*    어텐션이 파트너를 무엇으로 고르나
                 unif  1=완전균일(구분 안 함)
                 dmsg  메시지만 섞었을 때 α 총변동  (내용 민감도)
                 dpos  relpos 만 섞었을 때 α 총변동 (위치 민감도)
                 near  α 최대가 최근접 파트너인 비율. 무작위면 1/K
               dmsg << dpos 이면 "위치로만 고르고 내용은 안 봄"
    act_*      수신자가 실제로 쓰나. 조타명령 변화 / 자기 표준편차
                 zero  메시지를 0 으로   shuf  남의 메시지로 바꿔치기
                 read_ratio = shuf/zero. 0 에 가까우면 '있냐 없냐'만 보고 내용은 안 읽음
    part_*     통신 상대 기하: 유효 파트너 수 / 거리 중앙값 / 레이더 밖 비율
    enc_alive  메시지망 레이더 인코더 출력 중 살아있는 유닛 비율 (dying ReLU 감시)
    """
    E, N = x.shape[0], x.shape[1]
    M = E * N
    out = {}
    with torch.no_grad():
        om0, parts0, topd = comm_gather(policy, env, x, goal, self_s, sit, K, return_dist=True)
        pmask = parts0[3]                                             # [E,N,Kc,1]
        prel = parts0[4]
        Kc = pmask.shape[2]
        msg = policy.msg_actor(x, goal, self_s, sit)                  # [E,N,MSG_DIM]
        mf = msg.reshape(M, -1)

        # ── 메시지 자체 (정의는 msg_stats 하나 — diag_ckpt 의 조우/비조우 분리와 공유) ──
        _ms = msg_stats(mf)
        out['msg_sd'] = _ms['sd']; out['msg_sat'] = _ms['sat']; out['msg_eff_dim'] = _ms['eff_dim']
        out['msg_axes90'] = _ms['axes90']; out['msg_corr'] = _ms['corr']; out['msg_dc'] = _ms['dc']
        # ── 메시지에 실린 위협 정보 (구 _diag_msg_channel.py(→_archive, 현행 diag_ckpt.py) 흡수, 2026-09-10) ──
        #   threat_r2  : threat_decoder 가 메시지에서 송신자의 top-K 위협 기하를 얼마나 복원하나 (1=완벽, ≤0=평균만 못함)
        #   om_erank   : 수신측 집계 메시지(others_msg)의 유효차원
        #   label_erank: 위협 라벨 자체의 유효차원 — 복원 천장 참고용
        #   sit_rate   : 이 배치에서 COLREGs 조우 중(sit≠0)인 배 비율 — 진단 창이 조우 없는 구간이면 위 지표가 무의미
        thr, tmask = compute_own_threat(x.reshape(M, -1), cfg.THREAT_K, x.device)
        pred = policy.threat_decoder(mf)
        _mse = float(((pred - thr).pow(2) * tmask).sum() / tmask.sum().clamp(min=1))
        _base = thr.sum(0) / tmask.sum(0).clamp(min=1)
        _mse0 = float(((_base.unsqueeze(0) - thr).pow(2) * tmask).sum() / tmask.sum().clamp(min=1))
        out['threat_r2'] = (1.0 - _mse / _mse0) if _mse0 > 0 else float('nan')
        out['label_erank'] = msg_stats(thr * tmask)['eff_dim']
        out['om_erank'] = msg_stats(om0.reshape(M, -1))['eff_dim']
        out['sit_rate'] = float((sit.reshape(-1) != 0).float().mean())

        # ── 게이트 ──
        for tag, mod in (('gate_ctr', policy.ctr_actor), ('gate_cri', policy.critic)):
            gs = [torch.sigmoid(p).mean() for n_, p in mod.named_parameters() if 'msg_gate' in n_]
            out[tag] = float(torch.stack(gs).mean()) if gs else float('nan')

        # ── 파트너 기하 ──
        pm_b = pmask.reshape(-1).bool()
        dv = topd.reshape(-1)[pm_b]
        out['part_n'] = float(pmask.sum(dim=2).mean())
        out['part_med'] = float(dv.median()) if dv.numel() else float('nan')
        out['part_out'] = float((dv > radar_range).float().mean()) if dv.numel() else float('nan')

        # ── 어텐션: 위치로 고르나 내용으로 고르나 ──
        if getattr(policy, 'use_attention', False) and Kc >= 2:
            at = policy.attn
            q_in = torch.cat([self_s, goal], dim=-1).reshape(M, 1, -1)
            prel_f = prel.reshape(M, Kc, -1)
            pm_f = pmask.reshape(M, Kc, 1)
            b_ = torch.arange(E, device=x.device)[:, None, None]
            # comm_gather 와 동일하게 파트너 인덱스를 다시 만든다(거리로 topk). ★2026-09-25: 파트너 반경도 comm_gather 와 같게
            _pr = net_mod.PARTNER_RANGE
            dmat = torch.cdist(env.pos, env.pos) + torch.eye(N, device=x.device).unsqueeze(0) * 1e9
            dmat = torch.where(dmat <= (cfg.COMM_RANGE if _pr is None else float(_pr)), dmat, torch.full_like(dmat, 1e9))
            _, topi = torch.topk(dmat, Kc, dim=-1, largest=False)
            msg_f = (msg[b_, topi] * pmask).reshape(M, Kc, -1)

            def _alpha(rel, mp):
                # aggregate_batch 와 같은 토큰 (COMM_LATENT 1.0 이면 곱하지 않음 = 기존과 비트동일)
                _mt = mp * net_mod._MSG_TOKEN_GAIN if net_mod.COMM_LATENT == 1.0 else mp * net_mod._MSG_TOKEN_GAIN * net_mod.COMM_LATENT
                tok = torch.cat([rel, _mt], dim=-1)
                sc = (at.k_proj(tok) * at.q_proj(q_in)).sum(-1) / at.scale
                sc = sc.masked_fill(pm_f.squeeze(-1) <= 0, -1e9)
                return torch.softmax(sc, dim=1)

            perm = torch.randperm(Kc, generator=gen).to(x.device)      # ★전용 CPU generator
            a0 = _alpha(prel_f, msg_f)
            nval = pm_f.squeeze(-1).sum(1)
            ok = nval >= 2
            if ok.any():
                p = a0.clamp(min=1e-12)
                out['alpha_unif'] = float((-(p * p.log()).sum(1) / nval.clamp(min=2).log())[ok].mean())
                out['alpha_dmsg'] = float(((_alpha(prel_f, msg_f[:, perm]) - a0).abs().sum(1) / 2)[ok].mean())
                # alpha_dpos 는 relpos 3열만 섞는다(정의 유지 — EXT=0 이면 prel_f 가 3열이라 기존과 같은 값)
                _rp = torch.cat([prel_f[:, perm, :3], prel_f[:, :, 3:]], dim=-1)
                out['alpha_dpos'] = float(((_alpha(_rp, msg_f) - a0).abs().sum(1) / 2)[ok].mean())
                out['alpha_near'] = float((a0.argmax(1) == topd.reshape(M, Kc).argmin(1))[ok].float().mean())
                if prel_f.shape[-1] > 3:
                    _re = torch.cat([prel_f[:, :, :3], prel_f[:, perm, 3:]], dim=-1)
                    out['alpha_dext'] = float(((_alpha(_re, msg_f) - a0).abs().sum(1) / 2)[ok].mean())
        for kk in ('alpha_unif', 'alpha_dmsg', 'alpha_dpos', 'alpha_near'):
            out.setdefault(kk, float('nan'))

        # ── 수신자가 내용을 읽나 (평균행동 기준 → RNG 소비 없음) ──
        perm_n = torch.randperm(N, generator=gen).to(x.device)
        _, a_base, _, _, _ = policy.ctr_actor._route(x, goal, self_s, om0, sit)
        om_z, _ = comm_gather(policy, env, x, goal, self_s, sit, K,
                              msg_override=torch.zeros_like(msg))
        om_s, _ = comm_gather(policy, env, x, goal, self_s, sit, K,
                              msg_override=msg[:, perm_n])
        _, a_z, _, _, _ = policy.ctr_actor._route(x, goal, self_s, om_z, sit)
        _, a_s, _, _, _ = policy.ctr_actor._route(x, goal, self_s, om_s, sit)
        asd = a_base.reshape(M, -1).std(0).clamp(min=1e-8)
        out['act_zero'] = float(((a_z - a_base).reshape(M, -1).abs().mean(0) / asd)[0])
        out['act_shuf'] = float(((a_s - a_base).reshape(M, -1).abs().mean(0) / asd)[0])
        out['read_ratio'] = out['act_shuf'] / out['act_zero'] if out['act_zero'] > 1e-8 else float('nan')
        # ★COMM_EXT (2026-09-25): 확장필드 그룹을 하나씩 끄면 조타가 얼마나 바뀌나 = 수신자가 그 그룹을 읽나
        #   (act_zero 는 정의 유지 — latent 만 0, 확장필드는 그대로). 켜진 그룹이 아니면 nan.
        if getattr(policy, 'relpos_dim', 3) > 3:
            _on = tuple(net_mod.COMM_GROUPS)
            for _col, _drop in (('act_zero_ext', _on), ('act_zero_state', ('state',)),
                                ('act_zero_role', ('role',)), ('act_zero_intent', ('intent',))):
                if not set(_drop) & set(_on):
                    out[_col] = float('nan')
                    continue
                om_g, _ = comm_gather(policy, env, x, goal, self_s, sit, K,
                                      groups=tuple(g for g in _on if g not in _drop))
                _, a_g, _, _, _ = policy.ctr_actor._route(x, goal, self_s, om_g, sit)
                out[_col] = float(((a_g - a_base).reshape(M, -1).abs().mean(0) / asd)[0])
            out.setdefault('alpha_dext', float('nan'))

        # ── 인코더 건강 (dying ReLU): 배치 산포가 죽은 유닛 비율의 여집합 ──
        core = policy.msg_actor.cores()[0] if hasattr(policy.msg_actor, 'cores') else None
        if core is not None and hasattr(core, 'radar_encoder'):
            feat = core.radar_encoder(x.reshape(M, -1))
            out['enc_alive'] = float((feat.std(0) > 1e-6).float().mean())
        else:
            out['enc_alive'] = float('nan')
    return out


def main():
    # (ASCII 대시만 — U+2014 는 cp949 콘솔(Windows 리다이렉트)에서 UnicodeEncodeError 로 즉사함, 2026-09-10 실측)
    print(f"[version] {getattr(cfg, 'CODE_VERSION', '?')} - env 로 안 준 키는 config 기본값(YUGIOH)", flush=True)
    print(f"[sim] dyn_profile={cfg.DYN_PROFILE} obstacles={cfg.OBSTACLES_MODE} formula={cfg.DYN['formula']} "
          f"rudder_rate={cfg.DYN['rudder_rate']} r_full={cfg.DYN['r_full']} goal_reached={cfg.DYN['goal_reached']:.2f}", flush=True)
    ap = argparse.ArgumentParser()
    # OFF=통신 없음 / ORACLE=참 파트너 goal 주입(정보 상한) / ON=학습형 comm / RANDOM=난수 메시지 대조군
    ap.add_argument('--arm', default='OFF', choices=['OFF', 'ORACLE', 'ON', 'RANDOM'])
    ap.add_argument('--comm_on_at', type=int, default=0)   # ON arm: comm 켜는 decision 임계(curriculum). 0=처음부터
    ap.add_argument('--max_partners', type=int, default=cfg.MAX_COMM_PARTNERS)  # msg 처리: 4=aggregation, 1=nearest-1
    ap.add_argument('--ckpt_every', type=float, default=0.0)  # M단위 중간 체크포인트(0=끄기)
    ap.add_argument('--steps', type=int, default=1_000_000)   # 총 env-decision (환경당)
    ap.add_argument('--envs', type=int, default=None)
    ap.add_argument('--vessels', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rollout', type=int, default=64)        # rollout 길이(결정)
    ap.add_argument('--ring', type=float, default=1.0)   # ★2026-08-27: 0.7→1.0 (씬 원본). 0.7 은 goal 4개가
                                                         #   장애물 표면에 얹혀 도달률 0% 였음. C# 기본도 1.0.
    ap.add_argument('--crossing', type=int, default=0)   # 2=대척(구 기본) / 그 외=최소거리 랜덤(씬 기본)
    # ★재개(2026-08-31): 저장된 체크포인트에서 이어 학습. --resume_at 은 '이미 끝낸 결정 수'로,
    #   total_decisions 초기값이자 CSV step 오프셋이 된다(학습곡선이 끊기지 않게).
    #   ⚠️Adam 모멘트가 체크포인트에 있으면 함께 복원한다. 없으면(구 체크포인트) 0에서 다시 쌓이는데,
    #     lr 3e-4·beta(0.9,0.999) 기준 수백 step 이면 회복되므로 12M 이어달리기에선 무시할 수준이다.
    ap.add_argument('--resume', default=None, help='이어 학습할 체크포인트 경로')
    ap.add_argument('--resume_at', type=int, default=0, help='이미 완료한 결정 수(로그·종료조건 기준)')
    # ★재개 워밍업 (2026-09-03): --resume 은 가중치·Adam·ValueNorm 만 복원하고 환경은 새로 reset()
    #   한다. 그러면 전 선박이 *같은 위상*으로 동시에 출발해, 한 에피소드 주기 동안 보상이 크게
    #   출렁인다(실측: 출발 직후 2.49 → 에피소드 절반 1.02 → 1주기 뒤 1.70 으로 복귀).
    #   평상시엔 배들이 서로 다른 단계에 흩어져 있어 평균이 안정적인데 리셋이 그걸 맞춰버린 것.
    #   그 구간이 분기 학습의 34%(36/107 업데이트)를 차지해 *그림뿐 아니라 학습도* 오염된다.
    #   → 학습·기록 없이 환경만 굴려 위상을 흩은 뒤 시작한다. 에이전트당 결정 수로 지정.
    #   ⚠️워밍업은 arm 과 무관하게 항상 통신 OFF(others_msg=0)로 굴린다. 그래야 두 팔이
    #     *같은 환경 상태*에서 출발해 분기 비교가 성립한다.
    ap.add_argument('--resume_warmup', type=int, default=0,
                    help='재개 직후 학습·기록 없이 굴릴 에이전트당 결정 수 (권장: 수렴 에피소드 길이 ~1200)')
    # ★분기 규약 우회 (2026-09-15): 커리큘럼 ON(--comm_on_at>0)은 trunk 에서 분기한 런만 허용한다(아래 [branch]).
    #   이 플래그로 돌린 런은 OFF 와 9M 까지 같은 모델이 아니므로 ON/OFF 짝 비교에 쓰지 않는다.
    ap.add_argument('--allow_unbranched', action='store_true',
                    help='trunk 분기 없이 --comm_on_at>0 ON 을 처음부터 돌림 (탐색 전용, 짝 비교 금지)')
    ap.add_argument('--save', default=None)
    ap.add_argument('--csv', default=None)   # 조밀 학습곡선 CSV 경로(None이면 save 기반 자동)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    E = args.envs or (1024 if device == 'cuda' else 64)
    N = args.vessels
    torch.manual_seed(args.seed)

    # commgate 수요 보상 (ORACLE·OFF 동일)
    # ★2026-08-30 보상 반경 통일 (사용자 결정): 레이더 밴드(0~56m)와 통신 밴드(56~200m)를 *같은 항*으로 비용화.
    #   reward_range=COMM_RANGE(200) → colcourse/perpair 가 200m 까지 연속으로 걸린다(56m 절벽 제거).
    #   farfield PBRS 는 같은 구간을 telescoping 으로 또 건드리므로 기본 OFF(중복). VESSEL_FARFIELD_COEF 로 부활 가능.
    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device=device, seed=args.seed,
                            ring_scale=args.ring, crossing=args.crossing,
                            risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE,
                            # ★reward#1 fix 2026-08: per-pair 벌점을 risk³로 집중(exp 1.6→3.0)+계수↓(-0.3→-0.15)
                            #   → 중간위험 다중선박 통과 허용, 고위험만 강함. (colcourse/proximity 집중과 짝)
                            farfield_coef=cfg.FARFIELD_COEF,
                            perpair_coef=cfg.PERPAIR_COEF,
                            perpair_exp=3.0)
    policy = CNNPolicy(MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, FRAMES).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.LEARNING_RATE)
    vnorm = ValueNorm(device)   # ★리턴 정규화 (critic 출력 = 정규화 공간)

    # ★재개: 가중치 + ValueNorm 통계 + (있으면) Adam 모멘트 복원
    # ★분기 규약 (2026-09-15, 사용자 지시): ON/OFF 는 통신 켜는 지점까지 *같은 체크포인트 파일*에서 출발한다.
    #   09-10 배치는 OFF·ON 을 같은 시드로 따로 처음부터 돌렸는데, 결정론 설정이 없어 2번째 update(131,072)부터
    #   갈라졌고 9M 체크포인트가 이미 다른 모델이었다(s43 도착 89.1% vs 98.2%). 시드가 아니라 파일로 보장한다.
    #   분기점 = resume_at == comm_on_at > 0 인 재개. 그 파일(trunk)의 SHA256 을 스냅샷에 남긴다.
    #   분기 *뒤*의 크래시 재개는 이전 스냅샷의 분기 키를 물려받는다(짝 관계 유지).
    _branch = None
    if args.resume:
        _ck = torch.load(args.resume, map_location=device)
        _prev_snap = _ck.get('cfg_snapshot') or {}
        # ★2026-09-21 sim 설정 일치 — 재개·분기는 같은 동역학·시나리오여야 함(다르면 다른 실험을 이어 붙이는 것). 우회 없음.
        _prev_dp = str(_prev_snap.get('dyn_profile') or 'agile').lower()
        _prev_ob = str(_prev_snap.get('obstacles') or 'grid3x3').lower()
        if _prev_dp != cfg.DYN_PROFILE or _prev_ob != cfg.OBSTACLES_MODE:
            raise SystemExit(f"[resume] 거부: 체크포인트 sim 설정 dyn_profile={_prev_dp} obstacles={_prev_ob} != "
                             f"현재 {cfg.DYN_PROFILE}/{cfg.OBSTACLES_MODE} - VESSEL_DYN_PROFILE/VESSEL_OBSTACLES 를 맞출 것")
        # ★이름이 같아도 프로필 *정의*(숫자)가 바뀐 코드면 다른 실험이다. 스냅샷의 dyn 숫자 dict 가 유일한 근거.
        if isinstance(_prev_snap.get('dyn'), dict):
            from ckpt_io import dyn_constants_mismatch
            _cur_dyn = vg.current_dyn_constants()
            _bad_dyn = dyn_constants_mismatch(_prev_snap['dyn'], _cur_dyn)
            if _bad_dyn:
                _k = _bad_dyn[0]
                raise SystemExit(f"[resume] 거부: 체크포인트 dyn 상수 {_k} = {_prev_snap['dyn'].get(_k)} != "
                                 f"현재 {_cur_dyn.get(_k)} — 프로필 정의가 바뀐 코드로 재개 불가(스냅샷이 유일 근거)"
                                 + (f" / 다른 키도 다름: {_bad_dyn[1:]}" if len(_bad_dyn) > 1 else ""))
        # ★2026-09-23 sim 상수(보상 계수·게이트·COLREGS_MODE·에피소드 길이)도 같아야 한다 — 다르면 보상이 다른
        #   실험을 이어 붙이는 것. 체크포인트에 있는 키만 본다(구 체크포인트 호환). 우회 없음.
        if isinstance(_prev_snap.get('sim'), dict):
            from ckpt_io import dyn_constants_mismatch
            _ck_sim, _cur_sim = _prev_snap['sim'], cfg.sim_constants()
            _bad_sim = dyn_constants_mismatch(_ck_sim, {k: v for k, v in _cur_sim.items() if k in _ck_sim})
            if _bad_sim:
                _d = ', '.join(f"{k} ckpt={_ck_sim[k]!r} 현재={_cur_sim.get(k, '<없음>')!r}" for k in _bad_sim)
                raise SystemExit(f"[resume] 거부: 체크포인트 sim 상수 불일치 {_bad_sim} — 보상·게이트가 다른 "
                                 f"코드/env 로 재개 불가(스냅샷이 유일 근거): {_d}")
        # ★2026-09-25 의도·역할 통신 설정 일치. 가중치에 흔적이 없는 값(필드·latent 배율·파트너 반경·aux 배율)은 스냅샷이
        #   유일 근거라, 크래시 재개에서 env 하나를 빼먹으면 다른 팔이 조용히 이어 붙는다(예: intent 런이 latent 로).
        #   분기점(trunk=OFF → 갈래)에서는 팔마다 필드가 다른 게 정상이므로 comm_ext(구조)만 본다. 우회 없음.
        _cur_comm = {'comm_ext': int(cfg.COMM_EXT), 'comm_fields': cfg.COMM_FIELDS, 'comm_latent': float(cfg.COMM_LATENT),
                     'partner_range': (None if cfg.PARTNER_RANGE is None else float(cfg.PARTNER_RANGE)),
                     'aux_loss_scale': float(cfg.AUX_LOSS_SCALE)}
        _legacy_comm = {'comm_ext': 0, 'comm_fields': 'latent', 'comm_latent': 1.0, 'partner_range': None, 'aux_loss_scale': 1.0}
        _prev_comm = {k: _prev_snap.get(k, _legacy_comm[k]) for k in _cur_comm}
        _is_branch_pt = args.comm_on_at > 0 and args.resume_at == args.comm_on_at
        _keys_chk = ('comm_ext',) if _is_branch_pt else tuple(_cur_comm)
        _bad_comm = [k for k in _keys_chk if _prev_comm[k] != _cur_comm[k]]
        if _bad_comm:
            raise SystemExit("[resume] 거부: 통신 설정 불일치 " + ', '.join(
                f"{k} ckpt={_prev_comm[k]!r} 현재={_cur_comm[k]!r}" for k in _bad_comm)
                + " - VESSEL_COMM_EXT/COMM_FIELDS/COMM_LATENT/PARTNER_RANGE/AUX_LOSS_SCALE 를 체크포인트와 맞출 것")
        if args.comm_on_at > 0 and args.resume_at == args.comm_on_at:
            if _ck.get('comm_active'):
                raise SystemExit('[branch] 거부: trunk 가 통신이 켜진 뒤의 모델임(comm_active=True). '
                                 '분기점 trunk 는 통신 OFF 로 학습한 모델이어야 함')
            if _ck.get('steps') is not None and int(_ck['steps']) != args.resume_at:
                raise SystemExit(f"[branch] 거부: trunk steps={_ck['steps']} != --resume_at {args.resume_at}")
            if _ck.get('seed') is not None and int(_ck['seed']) != args.seed:
                raise SystemExit(f"[branch] 거부: trunk seed={_ck['seed']} != --seed {args.seed} "
                                 f"- 갈래들이 같은 RNG 로 출발해야 함")
            _h = hashlib.sha256()
            with open(args.resume, 'rb') as _fh:
                for _chunk in iter(lambda: _fh.read(1 << 20), b''):
                    _h.update(_chunk)
            _branch = {'branch_from': os.path.basename(args.resume), 'branch_from_sha256': _h.hexdigest(),
                       'branch_at': int(args.resume_at)}
        elif _prev_snap.get('branch_from_sha256'):
            _branch = {k: _prev_snap.get(k) for k in ('branch_from', 'branch_from_sha256', 'branch_at')}
        policy.load_state_dict(_ck['model_state_dict'] if 'model_state_dict' in _ck else _ck)
        # ★2026-09-05 fix: ValueNorm.load 는 통계가 없으면 조용히 return 한다.
        #   그러면 debias=0 이라 _stats() 가 mean=0, std=sqrt(1e-6)=1e-3 을 돌려주고,
        #   복원된 critic 은 정규화 공간(std≈17) 값을 내는데 denormalize 가 ×1e-3 해
        #   values≈0 → 첫 update 의 GAE 가 통째로 오염된다. 조용한 실패를 막는다.
        _vn = _ck.get('value_norm')
        if not _vn:
            print('[resume] 경고: 체크포인트에 value_norm 없음 - 첫 update 의 value/GAE 가 스케일 붕괴함. '
                  '--resume_warmup 을 주거나 통계가 있는 ckpt 를 쓸 것', flush=True)
        vnorm.load(_vn)
        if 'optimizer_state_dict' in _ck:
            opt.load_state_dict(_ck['optimizer_state_dict'])
            _os_msg = 'Adam 복원'
        else:
            _os_msg = 'Adam 없음(재축적)'
        print(f"[resume] {os.path.basename(args.resume)} 에서 이어감 | {args.resume_at/1e6:.2f}M 완료분 | "
              f"ValueNorm {_ck.get('value_norm')} | {_os_msg}", flush=True)

    if _branch:
        print(f"[branch] trunk={_branch['branch_from']} sha256={_branch['branch_from_sha256'][:12]} "
              f"branch_at={_branch['branch_at']} arm={args.arm}", flush=True)
    # ★분기 규약 강제 (2026-09-15): 커리큘럼 ON 은 trunk 분기 런만. 따로 처음부터 돌린 ON 은 OFF 와 짝이 아님.
    if args.arm == 'ON' and args.comm_on_at > 0 and not (_branch and int(_branch['branch_at']) == args.comm_on_at):
        _bmsg = (f"[branch] --arm ON --comm_on_at {args.comm_on_at} 인데 trunk 분기가 아님. "
                 f"OFF 로 {args.comm_on_at} 결정까지 학습한 trunk 에서 --resume <trunk> --resume_at {args.comm_on_at} "
                 f"로 분기할 것 (run_repro.sh train 이 자동으로 함).")
        if not args.allow_unbranched:
            raise SystemExit(_bmsg + ' 탐색 목적이면 --allow_unbranched.')
        print(_bmsg + ' --allow_unbranched 로 진행 - 이 런은 ON/OFF 짝 비교에 쓰지 말 것', flush=True)

    # ★arm ON 인데 통신이 꺼져 있으면 rollout(comm_gather 는 USE_COMMUNICATION 을 안 봄)과
    #   update(networks.evaluate_actions 는 0으로 만듦)의 others_msg 가 달라져 PPO ratio 가 조용히 깨진다.
    assert not (args.arm == 'ON' and not cfg.USE_COMMUNICATION), \
        "--arm ON 인데 VESSEL_USE_COMM=0 임. rollout != update 로 PPO 가 깨짐 — 둘 중 하나를 맞출 것"
    # ★2026-09-05 fix: ORACLE 은 rollout 을 make_others_msg(참 goal 주입)로 만드는데, --arm ON 이면
    #   rollout 은 학습채널(comm_gather)을 쓰고 update(networks.evaluate_actions)는 USE_ORACLE 분기를 타
    #   *참 goal* 을 주입한다 → rollout != update 로 ratio 가 조용히 깨진다. 조합 자체를 막는다.
    assert not (args.arm == 'ON' and cfg.USE_ORACLE), \
        "--arm ON 과 VESSEL_ORACLE=1 은 같이 못 씀. update 가 oracle 분기를 타서 rollout 과 어긋남 " \
        "(oracle 통제군은 --arm ORACLE 로 돌릴 것)"
    # ★난수 대조군은 상수 입력 경로(OFF/ORACLE와 동일)로 흐르므로 통신 채널 학습이 없어야 정상이다.
    if args.arm == 'RANDOM':
        print(f"[arm RANDOM] 난수 메시지 대조군 - others_msg ~ U(-a,a), sd="
              f"{MSG_RANDOM_SD:.3f} "
              f"(비교 팔의 실측 om_sd 에 맞출 것: _diag_msg_channel.py(→_archive, 현행 diag_ckpt.py))", flush=True)

    # ★2026-09-05 fix(opt-in): timeout 절단을 GAE 에서 '절단'으로 취급할지.
    #   기본 0 = 기존 동작(상수 trunc=0, timeout 을 진짜 종료로 취급) — 비트동일.
    _trunc_boot = cfg.TIMEOUT_BOOTSTRAP
    _grad_tele = cfg.GRAD_TELEMETRY   # 모듈별 grad norm·clip 계수 (진단 전용)
    _gacc = {}
    # ★2026-09-26 구간 시간 측정 (VESSEL_TIMING=1, 기본 0 = stdout 한 글자도 안 바뀜). rollout / update 벽시계를 따로 재서
    #   5-update 로그 줄 끝에 붙인다. cuda.synchronize 는 계산·RNG 를 건드리지 않는다(측정 전용). 기본 0 이면 호출 자체가 없음.
    _timing = cfg.TIMING
    _tsync = (lambda: torch.cuda.synchronize()) if (_timing and device == 'cuda') else (lambda: None)
    _t_roll = _t_upd = 0.0
    _clip_per_module = cfg.CLIP_PER_MODULE
    if _clip_per_module:
        print('[clip] VESSEL_CLIP_PER_MODULE=1 - msg_actor/ctr_actor/critic/나머지를 각각 '
              f'{cfg.MAX_GRAD_NORM} 로 자름 (기존은 전체 한 덩어리). 두 팔에 동일 적용할 것.', flush=True)
    if _trunc_boot:
        print('[gae] VESSEL_TIMEOUT_BOOTSTRAP=1 - timeout 은 절단으로 처리(V(s_t) bootstrap). '
              '기본(0)과 학습 결과가 다름 ― 과거 run 과 직접 비교 금지', flush=True)
    # ★2026-09-05 fix: MSG_GATE_COEF(VESSEL_MSG_GATE_L2)가 이 학습기에선 loss 에 한 번도
    #   안 더해졌다(gate_open_sum 참조처는 main.py 뿐). 값을 주고 돌려도 cfg 스냅샷에만
    #   남아 문서와 실행이 어긋난다. 계수를 살리는 건 설계 변경이므로 기본은 그대로 두고
    #   ① 조용한 무효화를 경고로 드러내고 ② VESSEL_MSG_GATE_APPLY=1 로만 실제 적용한다.
    _gate_apply = cfg.MSG_GATE_APPLY
    if args.arm == 'ON' and cfg.MSG_GATE_COEF > 0.0 and not _gate_apply:
        print(f'[msg-gate] 경고: MSG_GATE_COEF={cfg.MSG_GATE_COEF} 이지만 이 학습기에선 loss 에 미적용 '
              f'(죽은 knob). 적용하려면 VESSEL_MSG_GATE_APPLY=1', flush=True)
    elif _gate_apply:
        print(f'[msg-gate] 게이트 개방 페널티 적용 (coef={cfg.MSG_GATE_COEF}) - '
              f'기본(미적용) run 과 학습 결과가 다름', flush=True)

    fs = FrameStack(E, N, device)
    obs = env.reset()
    radar, goal, self_s, sit = parse_obs(obs)
    fs.reset_all(radar)

    # ── 재개 워밍업: 학습·기록 없이 환경만 굴려 에이전트 위상을 흩는다 ──
    if args.resume and args.resume_warmup > 0:
        _t0w = time.time()
        for _ in range(args.resume_warmup):
            with torch.no_grad():
                _x = fs.get()
                _om = torch.zeros(E, N, MSG_DIM, device=device)   # 항상 통신 OFF (두 팔 동일 상태)
                _a, _, _, _ = policy.ctr_actor(_x, goal, self_s, _om, sit)
            obs, _, _done, _ = env.step(_a)
            radar, goal, self_s, sit = parse_obs(obs)
            fs.push(radar, _done)
        print(f'[warmup] 재개 워밍업 {args.resume_warmup} 결정/에이전트 완료 ({time.time()-_t0w:.0f}s) - 학습·기록 없음', flush=True)

    # ── 혼합 함대 학습(선택) ──────────────────────────────────────────────
    # 함대의 일부 선박이 통신 장비 없이 *학습 단계부터* 항해한다. 평가할 때만 통신을
    # 끊으면 그 배는 불리해지는 게 아니라 겪어본 적 없는 상황에 놓일 뿐이라, 통신의
    # 값어치를 재려면 없는 채로 키워야 한다. 공유 신경망 하나가 두 처지를 다 배운다.
    #   VESSEL_NOCOMM_SWEEP="2,4,6,8,10,12,14"  환경마다 다른 비율을 심는다(한 번의 학습으로 전 비율 커버)
    #   VESSEL_NOCOMM_MODE=radar|rx|mix         못 보내고 못 받음 / 듣기만 함 / 환경마다 번갈아
    send_mask = recv_mask = None
    _sweep = cfg.NOCOMM_SWEEP
    if _sweep:
        _ks = [int(s) for s in _sweep.split(',') if s.strip()]
        _mode = cfg.NOCOMM_MODE
        _nocomm = torch.zeros(E, N, dtype=torch.bool, device=device)
        _rxonly = torch.zeros(E, dtype=torch.bool, device=device)
        for e in range(E):
            k = _ks[e % len(_ks)]
            step = N / k if k > 0 else 0
            idx = sorted({int(round(i * step)) % N for i in range(k)}) if k > 0 else []
            _nocomm[e, idx] = True
            _rxonly[e] = (_mode == 'rx') or (_mode == 'mix' and (e // len(_ks)) % 2 == 1)
        send_mask = ~_nocomm                       # 장비 없는 배는 아무도 그 메시지를 못 받는다
        recv_mask = ~_nocomm | _rxonly.view(E, 1)  # 듣기만 하는 배는 받기는 한다
        print(f"[mixed-fleet] 비율 {_ks} 모드={_mode} | 통신불가 평균 "
              f"{_nocomm.float().sum(1).mean():.1f}/{N}척, 듣기만 환경 "
              f"{int(_rxonly.sum())}/{E}", flush=True)

    # ★2026-09-05: 체크포인트에 설정 스냅샷을 함께 저장한다.
    #   2026-09-10: 정의를 ckpt_io.snapshot_config 로 승격 — eval_ckpt·eval_mixed·diag_ckpt 가 같은 키를 읽는다.
    #   (키 추가: farfield_coef·perpair_coef·perpair_exp·radar_range·trainer. 기존 키 값은 불변.)
    def _cfg_snapshot():
        from ckpt_io import snapshot_config
        _snap = snapshot_config(arm=args.arm, msg_dim=MSG_DIM, seed=args.seed, n_envs=E, n_vessels=N,
                                max_partners=args.max_partners, trunc_boot=_trunc_boot,
                                comm_on_at=args.comm_on_at, ring=args.ring, crossing=args.crossing,
                                rollout=args.rollout, trainer='gym')
        if _branch:   # ★2026-09-15 분기 출처 — 분기(재개) 런에만 붙는 키. verify/check_branch.py 가 읽음
            _snap.update(_branch)
        return _snap

    # ★2026-09-05: intent/state-recon 라벨 정렬 버그를 고쳤다(pos/hdg 를 env.step *앞*에서 기록).
    #   고치기 전에는 라벨이 한 스텝 밀려 있었고 respawn 텔레포트가 마스크를 관통했다.
    #   INTENT_COEF·STATE_RECON_COEF 가 둘 다 0 이면 이 경로를 안 타므로 기본 설정은 비트동일이지만,
    #   둘 중 하나라도 켜면 *결과가 달라진다*. 조용히 달라지면 안 되므로 시작할 때 남긴다.
    if cfg.INTENT_COEF > 0.0 or cfg.STATE_RECON_COEF > 0.0:
        print(f"[label-fix] intent/state-recon 라벨 정렬 수정본임 (INTENT_COEF={cfg.INTENT_COEF} "
              f"STATE_RECON_COEF={cfg.STATE_RECON_COEF}). 2026-09-05 이전 run 과 직접 비교하지 말 것.",
              flush=True)

    T = args.rollout
    total_decisions = args.resume_at      # ★재개 시 이어서 카운트 (--steps 는 '총' 결정 수)
    outcome_counts = torch.zeros(5, device=device)
    t_start = time.time()
    update_i = 0
    _comm_was_active = False   # [comm] 전환 로그용 (comm_on_at 커리큘럼에서 9M 에 켜졌는지 로그로 확인)
    # ★재개 시 이미 저장된 마크에서 시작 (2026-08-31 fix). 0 으로 두면 재개 직후 첫 업데이트에서
    #   mark > 0 이 성립해 *방금 재개한 그 체크포인트*를 덮어쓴다. 9M 경계에서 재개하면
    #   comm_on_at 을 이미 넘은 상태라, 통신 OFF 모델이어야 할 .step9M.pt 가
    #   통신 ON 1업데이트분으로 조용히 바뀜다(ABLATION_PLAN §4 가 이 파일을 OFF 모델로 지정).
    # ★2026-09-05 fix: int(ckpt_every*1e6) 이 0 이면(0<ckpt_every<1e-6) ZeroDivisionError → 최소 1 로 클램프.
    _ck_div = max(int(args.ckpt_every * 1e6), 1) if args.ckpt_every > 0 else 0
    ckpt_mark = int(args.resume_at // _ck_div) if _ck_div else 0
    # ★2026-09-05 fix: total_decisions 는 rollout 당 E*N*T 만큼 한 번에 뛰고 저장 판정은 rollout 끝에서만
    #   한다. 간격보다 rollout 이 크면 중간 마크가 조용히 건너뛰어져 학습곡선에 구멍이 난 줄 모른다.
    #   건너뛴 이름으로 같은 가중치를 복제 저장하면 'step0.5M.pt' 가 실제로는 1M 모델이 돼 파일 라벨이
    #   거짓이 되므로(이 저장소 제1원칙 위반) 복제 대신 *경고*로 드러낸다.
    if _ck_div and E * N * T > _ck_div:
        print(f"[ckpt] 경고: rollout 1회 = {E*N*T:,} 결정 > ckpt 간격 {_ck_div:,} - 중간 마크가 건너뛰어짐. "
              f"--ckpt_every 를 {E*N*T/1e6:.3g} 이상으로 줄 것", flush=True)
    # ★조밀 로깅(2026-08): 매 update마다 (step, raw_reward, ema) CSV 기록 → 깨끗한 학습곡선용(참고 figure 스타일).
    csv_path = args.csv or (os.path.splitext(args.save)[0] + '_curve.csv' if args.save else None)
    _csv_mode = 'a' if (args.resume and csv_path and os.path.exists(csv_path)) else 'w'
    csv_f = open(csv_path, _csv_mode, encoding='utf-8') if csv_path else None
    if csv_f and _csv_mode == 'w':
        csv_f.write('step,raw_reward,ema_reward\n')
    ema_r = None
    # ★2026-09-15: 분기 갈래는 trunk 곡선을 이어 쓰므로 EMA 도 마지막 값에서 잇는다.
    #   안 그러면 EMA 가 분기점(= 통신 켜는 지점)에서 raw 값으로 튀어 통신 효과처럼 보이는 꺾임이 생긴다.
    if csv_f and _csv_mode == 'a':
        try:
            with open(csv_path, encoding='utf-8') as _cf:
                _rows = [ln for ln in _cf.read().splitlines() if ln and not ln.startswith('step')]
            if _rows:
                ema_r = float(_rows[-1].split(',')[2])
        except (OSError, ValueError, IndexError):
            ema_r = None
    # 붕괴 검출기 상태 (아래 [blind] 참조)
    _BLIND_WARN = cfg.BLIND_WARN_AFTER
    # ★2026-09-26 검출기 상태를 GPU 텐서로. 예전엔 미니배치마다 ctr_actor 70개 + 인코더 5×10개 텐서를 float() 로 하나씩
    #   호스트로 내렸다(120 sync/미니배치 = 30,720 sync/update — update 시간의 절반 이상, explore_train_loop.md 실측).
    #   이제 판정(연속 0 카운터)은 GPU 에서 매 미니배치 그대로 하고, 호스트 동기화는 update 당 1회(.tolist())만.
    #   MOE_SHARED=1 이면 전문가 5벌이 RadarEncoder *한 객체*를 공유해 같은 10개 텐서를 5번 세던 것도 id() 로 뗀다
    #   (합이 0 인지는 중복·덧셈 순서와 무관 — 음수 아닌 항의 합이라 상쇄가 없음). 정책은 여기서 확정된 상태
    #   (load_state_dict 는 in-place 복사라 파라미터 객체 동일성 유지).
    _blind_dev = next(policy.parameters()).device
    _enc_params = list({id(p): p for _c in policy.ctr_actor.cores() for p in _c.radar_encoder.parameters()}.values())
    _ctr_params = list(policy.ctr_actor.parameters())          # nn.Module.parameters() 는 이미 중복 제거
    _blind_run_t = torch.zeros((), dtype=torch.int64, device=_blind_dev)   # 연속 0 미니배치 수 (구 _blind_run, update 경계 넘어 이어짐)
    _blind_hit_t = torch.zeros((), dtype=torch.bool, device=_blind_dev)    # 이번 update 안에 임계 도달했나
    _blind_rg_t = torch.zeros((), dtype=torch.float64, device=_blind_dev)  # 도달한 미니배치의 ctr_actor 전체 grad (경고문용)
    _i64_zero = torch.zeros((), dtype=torch.int64, device=_blind_dev)
    _foreach_l1 = getattr(torch, '_foreach_norm', None)          # 비공개 API — 없으면 텐서별 abs().sum() (sync 는 여전히 0)

    def _l1_sum(gs, dtype):
        """grad 텐서 목록의 L1 합(0-d, dtype). 텐서별 L1 은 float32 로 재고 dtype 으로 올린 뒤 합친다."""
        if not gs:
            return torch.zeros((), dtype=dtype, device=_blind_dev)
        _n = _foreach_l1(gs, 1) if _foreach_l1 is not None else [g.abs().sum() for g in gs]
        return torch.stack(_n).to(dtype).sum()

    # ★2026-09-26 모듈별 clip 그룹을 루프 밖에서 한 번만 만든다(예전엔 미니배치마다 id() 집합을 다시 만들었다).
    #   순서·구성원 동일(공유 인코더 그룹 먼저 — 2026-09-10 메모 — 그다음 msg/ctr/critic/나머지) → clip 결과 비트동일.
    #   clip_grad_norm_ 은 호출마다 grad=None 파라미터를 걸러내므로 grad 유무는 예전과 같이 호출 시점에 반영된다.
    _clip_groups = []
    if _clip_per_module:
        _seen = set()
        if getattr(policy, 'shared_encoder', '0') != '0':
            # ★2026-09-10 공유 인코더는 세 망에 걸쳐 있으므로 자기 그룹으로 먼저 뗀다
            #   (안 그러면 순서상 msg_actor 그룹이 가져가 그 그룹 norm 을 인코더가 지배함)
            _enc = [p for _c in policy.ctr_actor.cores() for p in _c.radar_encoder.parameters()
                    if id(p) not in _seen]
            _seen.update(id(p) for p in _enc)
            _clip_groups.append(_enc)
        for _m in (policy.msg_actor, policy.ctr_actor, policy.critic):
            _ps = [p for p in _m.parameters() if id(p) not in _seen]
            _seen.update(id(p) for p in _ps)
            _clip_groups.append(_ps)
        _rest = [p for p in policy.parameters() if id(p) not in _seen]
        if _rest:
            _clip_groups.append(_rest)
        del _seen
    # ★상태복원 그룹별 손실 감시 CSV (2026-09-04): gradient 쏠림을 학습 '도중에' 본다
    #   2026-09-26: 그룹별 원손실을 python float dict 대신 float64 [5] 텐서로 *같은 순서로* 누적한다 — networks.StateReconDecoder
    #   .loss 가 raw[g] 를 float() 대신 0-d 텐서로 주면(미니배치당 sync 5회 제거) 여기서 한 번(update 당 .tolist() 1회)만 내린다.
    #   float() 는 float32→float64 정확 변환이고 .to(float64) 도 같다. 예전엔 python float64 로 0.0+v1+v2+… 순차 덧셈이었고
    #   지금은 device 에서 같은 순차 float64 덧셈(IEEE-754 double add 는 CPU/GPU 동일) → aux CSV 바이트 동일.
    #   raw 값이 python float 여도 as_tensor(dtype=float64) 가 받으므로 두 버전의 networks.py 와 다 맞는다.
    _SR_GROUPS = ('goal', 'self', 'sit', 'threat', 'future')   # aux CSV 열 순서 = StateReconDecoder.GROUPS
    _sr_acc = None
    _sr_n = 0
    aux_f = None
    if cfg.STATE_RECON_COEF > 0.0 and csv_path:
        _aux_path = os.path.splitext(csv_path)[0] + '_aux.csv'
        # ★2026-09-05 fix: 이전엔 curve CSV 의 모드(_csv_mode)를 그대로 썼다. 돌던 run 을 --resume
        #   하면서 그때 처음 STATE_RECON_COEF>0 을 켜면 curve 는 있으니 'a' 인데 _aux.csv 는 새로
        #   생기므로 헤더 없는 파일이 됐다(pandas 가 첫 데이터 행을 헤더로 먹음). 자기 경로로 판정한다.
        _aux_mode = 'a' if (args.resume and os.path.exists(_aux_path)) else 'w'
        aux_f = open(_aux_path, _aux_mode, encoding='utf-8')
        if _aux_mode == 'w':
            aux_f.write('step,goal,self,sit,threat,future\n')

    # ★통신 텔레메트리 CSV (2026-09-08): VESSEL_COMM_TELEMETRY=1 일 때만. 기본 0 = 비트동일.
    #   ON 팔에서만 의미 있음(OFF 는 others_msg≡0). 전용 CPU generator 로 학습 RNG 와 분리한다.
    _tele_f = None
    _tele_gen = None
    _tele_every = cfg.COMM_TELEMETRY_EVERY   # update 단위
    if cfg.COMM_TELEMETRY and csv_path and args.arm == 'ON':
        _tele_path = os.path.splitext(csv_path)[0] + '_comm.csv'
        _tele_mode = 'a' if (args.resume and os.path.exists(_tele_path)) else 'w'
        _tele_f = open(_tele_path, _tele_mode, encoding='utf-8')
        if _tele_mode == 'w':
            _tele_f.write('step,' + ','.join(comm_tele_cols(policy)) + '\n')
        _tele_gen = torch.Generator().manual_seed(args.seed + 100003)       # 학습 RNG 와 완전 분리
        print(f"[telemetry] 통신 텔레메트리 켬 → {os.path.basename(_tele_path)} "
              f"({_tele_every} update 마다, {len(comm_tele_cols(policy))}개 열)", flush=True)

    # ★2026-09-05 fix: 예외(KeyboardInterrupt·CUDA OOM·env 오류)로 죽으면 csv_f·aux_f 가 닫히지
    #   않아 마지막 flush 이후 최대 19 update 분 기록이 통째로 날아갔다(둘 다 20 update 마다만 flush).
    #   특히 aux_f 는 정상 종료 경로에도 close 가 아예 없었다. try/finally 로 두 핸들을 확실히 닫는다.
    try:
        while total_decisions < args.steps:
            # ─── rollout ───
            # ★comm curriculum: ON arm이고 comm_on_at 넘으면 학습형 comm 활성(이 rollout 내내 일관 → PPO 정합)
            comm_active = (args.arm == 'ON' and total_decisions >= args.comm_on_at)
            if comm_active and not _comm_was_active:
                print(f"[comm] ON at dec={total_decisions/1e6:.3f}M (comm_on_at={args.comm_on_at}) - 이 rollout 부터 학습형 통신 + 텔레메트리", flush=True)
            _comm_was_active = comm_active
            keys = ['x', 'goal', 'self', 'sit', 'om', 'act', 'logp', 'val', 'rew', 'done', 'trunc']
            if cfg.CENTRAL_CRITIC:
                keys += ['gf']
            # ★2026-09-25 AUX_LOSS_SCALE=0 이면 ON 전용 보조손실을 통째로 끈다 → 라벨 버퍼·state_recon 계산도 생략
            #   (evaluate_actions 는 own_future/own_threat 가 None 이면 state_recon 을 안 돈다). 1.0(기본)은 조건 불변 = 비트동일.
            use_intent = comm_active and (cfg.INTENT_COEF > 0.0 or cfg.STATE_RECON_COEF > 0.0) and cfg.AUX_LOSS_SCALE != 0.0
            if use_intent:
                keys += ['pos', 'hdg']
            if comm_active:
                keys += ['px', 'pg', 'ps', 'pmask', 'prel', 'psit']
            use_threat = comm_active and (cfg.THREAT_COEF > 0.0 or cfg.STATE_RECON_COEF > 0.0) and cfg.AUX_LOSS_SCALE != 0.0
            if use_threat:
                keys += ['othr', 'othrm']
            buf = {k: [] for k in keys}
            _tsync(); _t0 = time.perf_counter()
            for _ in range(T):
                x = fs.get()
                with torch.no_grad():
                    if comm_active:
                        om, (px, pg, ps, pmask, prel, psit) = comm_gather(
                            policy, env, x, goal, self_s, sit, args.max_partners,
                            send_mask=send_mask, recv_mask=recv_mask)
                    else:
                        om = make_others_msg(env, 'OFF' if args.arm == 'ON' else args.arm, E, N, device)
                    action, logp, _, action_raw = policy.ctr_actor(x, goal, self_s, om, sit)
                    _gf = build_global_feat(env) if cfg.CENTRAL_CRITIC else None
                    value = vnorm.denormalize(policy.critic(x, goal, self_s, om, sit,
                                                            global_feat=_gf).squeeze(-1))
                # ★2026-09-05 fix: pos/hdg 를 env.step *앞*에서 저장함. 뒤에서 찍으면 s_{t+1}
                #   (종료한 배는 재스폰 좌표)이라 buf['x']=s_t 와 한 칸 어긋나 둘이 깨졌음:
                #     ① 계통지연 — 라벨이 s_{t+1} 기준 미래변위인데 s_t 의 obs 로 맞히게 학습(1/INTENT_HORIZON 편향).
                #     ② 텔레포트 누설 — compute_own_future 마스크가 sum(done[t..t+h-1]) 라
                #        pos 가 밀리면 끝점 재스폰을 못 거름. 재스폰 변위는 정상 라벨보다 두 자릿수 커서
                #        StateReconDecoder 의 run_var 를 부풀려 나머지 라벨 gradient 를 눌렀다.
                #   이제 pos[t]=s_t 라 memory.py 의 dn[t:j].any() 마스크와 정확히 일치함.
                #   ⚠영향 범위: use_intent(=INTENT_COEF>0 또는 STATE_RECON_COEF>0, 둘 다 기본 0) run 만.
                #     기본 설정 run 은 비트동일.
                if use_intent:
                    buf['pos'].append(env.pos.clone()); buf['hdg'].append(env.heading.clone())
                obs, reward, done, outcome = env.step(action)                # env엔 tanh action 적용
                # ★2026-09-26 5회 루프(커널 ~15개) → bincount 1회. outcome 은 long 0..4(vessel_gym OUT_*)라 bincount 가 정확히
                #   (outcome==oc).sum() 과 같은 정수. float32 로 옮겨 더해도 5-update 창 합이 2^24 미만이라 예전처럼 정확한 정수.
                outcome_counts += torch.bincount(outcome.reshape(-1), minlength=5)[:5].to(outcome_counts.dtype)
                buf['x'].append(x); buf['goal'].append(goal); buf['self'].append(self_s)
                # ★act = pre-tanh raw 저장(update가 그대로 재사용 → PPO ratio 정합)
                buf['sit'].append(sit); buf['om'].append(om); buf['act'].append(action_raw)
                if cfg.CENTRAL_CRITIC:
                    buf['gf'].append(_gf)
                buf['logp'].append(logp.squeeze(-1)); buf['val'].append(value)
                buf['rew'].append(reward); buf['done'].append(done.float())
                # ★2026-09-05 fix(opt-in): env 는 timeout 을 outcome=OUT_TIMEOUT 으로 구분해 주는데
                #   학습기는 trunc 를 상수 0 으로 채워, batched_gae 의 term=dones*(1-truncs) 에서
                #   timeout 이 *진짜 종료*로 처리됨(bootstrap 절단). 목표 코앞에서 절단된 배의
                #   value target 이 r_T(TIMEOUT_PENALTY 포함)로 주저앉아 장거리 여정을 회피하게 밀림.
                #   ⚠다만 이 수정은 모든 팔의 학습 결과를 바꿔 과거 보고 숫자와 비교가 끊긴다.
                #     기본은 기존 동작 유지, VESSEL_TIMEOUT_BOOTSTRAP=1 일 때만 켜진다(저자 결정 사항).
                buf['trunc'].append((outcome == vg.OUT_TIMEOUT).float() if _trunc_boot
                                    else torch.zeros_like(done.float()))
                if comm_active:
                    buf['px'].append(px); buf['pg'].append(pg); buf['ps'].append(ps)
                    buf['pmask'].append(pmask); buf['prel'].append(prel); buf['psit'].append(psit)
                if use_threat:
                    othr, othrm = compute_own_threat(x.reshape(E * N, -1), cfg.THREAT_K, device)
                    buf['othr'].append(othr.reshape(E, N, -1)); buf['othrm'].append(othrm.reshape(E, N, -1))
                radar, goal, self_s, sit = parse_obs(obs)
                fs.push(radar, done)
                total_decisions += E * N

            # last value
            # ★버그fix(2026-08-30): arm='ON' 일 때 make_others_msg 가 zeros 를 돌려줘, rollout 의 T 스텝은
            #   실제 통신 입력으로 value 를 재고 마지막 bootstrap 만 통신 없는 입력으로 쟀다(GAE 계통오차).
            with torch.no_grad():
                if comm_active:
                    om, _ = comm_gather(policy, env, fs.get(), goal, self_s, sit, args.max_partners,
                                        send_mask=send_mask, recv_mask=recv_mask)
                else:
                    om = make_others_msg(env, 'OFF' if args.arm == 'ON' else args.arm, E, N, device)
                last_v = vnorm.denormalize(policy.critic(
                    fs.get(), goal, self_s, om, sit,
                    global_feat=build_global_feat(env) if cfg.CENTRAL_CRITIC else None).squeeze(-1))
            _tsync(); _t_roll += time.perf_counter() - _t0; _t1 = time.perf_counter()

            # stack [T,E,N,...]
            S = {k: torch.stack(v) for k, v in buf.items()}
            del buf     # ★2026-09-26 stack 뒤엔 안 씀 — update 내내 잡고 있던 rollout 리스트(~1.5 GB, E=128 ON)를 놓는다. 메모리만.
            returns, adv = batched_gae(S['rew'], S['val'], S['done'], S['trunc'], last_v,
                                       cfg.DISCOUNT_FACTOR, cfg.GAE_LAMBDA)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # ★리턴 정규화: 통계 갱신 후 value target 을 정규화 공간으로 (critic 출력과 같은 공간)
            vnorm.update(returns)
            returns = vnorm.normalize(returns)

            # flatten [T*E*N, ...]
            def flat(t): return t.reshape(-1, *t.shape[3:]) if t.dim() > 3 else t.reshape(-1)
            fx = flat(S['x']); fg = flat(S['goal']); fsf = flat(S['self']); fsit = flat(S['sit'])
            fom = flat(S['om']); fact = flat(S['act']); flogp = flat(S['logp'])
            fret = flat(returns); fadv = flat(adv)
            fgf = flat(S['gf']) if cfg.CENTRAL_CRITIC else None      # [M, N_ships, 6]
            M = fx.shape[0]
            if comm_active:
                fpx = flat(S['px']); fpg = flat(S['pg']); fps = flat(S['ps'])
                fpmask = flat(S['pmask']); fprel = flat(S['prel']); fpsit = flat(S['psit'])
            if use_threat:
                fothr = flat(S['othr']); fothrm = flat(S['othrm'])
            if use_intent:
                _fut, _futm = compute_own_future(S['pos'], S['hdg'], S['done'].bool(),
                                                 cfg.INTENT_K, cfg.INTENT_HORIZON, cfg.INTENT_POS_SCALE)
                ffut = flat(_fut); ffutm = flat(_futm)

            # ─── PPO update ───
            # ★2026-09-05 fix: randperm 을 epoch 루프 *안*으로. 밖에 있으면 N_EPOCH 회가 전부 같은
            #   미니배치 분할을 반복해 epoch 간 표본 상관이 생긴다(PPO 표준은 epoch 마다 재셔플).
            mb = cfg.MINIBATCH_SIZE
            for _ in range(cfg.N_EPOCH):
                idx_all = torch.randperm(M, device=device)
                for s in range(0, M, mb):
                    mi = idx_all[s:s + mb]
                    aux = 0.0
                    if comm_active:
                        # ★학습형 comm: evaluate_actions가 파트너 obs로 others_msg 재계산(sender→receiver grad) + aux손실
                        # ★threat-relay: own_threat 라벨 제공 → 메시지가 '내 레이더가 본 위협' 인코딩(THREAT_COEF).
                        othr_b = fothr[mi].unsqueeze(1) if use_threat else None
                        othrm_b = fothrm[mi].unsqueeze(1) if use_threat else None
                        fut_b = ffut[mi].unsqueeze(1) if use_intent else None
                        futm_b = ffutm[mi].unsqueeze(1) if use_intent else None
                        val_u, logp_u, entropy, msg_reg, it_l, tt_l, gl_l, rl_l, cl_l = policy.evaluate_actions(
                            fx[mi], fg[mi], fsf[mi], fpx[mi], fpg[mi], fps[mi], fpmask[mi], fprel[mi],
                            fact[mi], own_future=fut_b, own_future_mask=futm_b,
                            own_threat=othr_b, own_threat_mask=othrm_b,
                            situation=fsit[mi], partner_situations=fpsit[mi],
                            global_feat=fgf[mi] if fgf is not None else None)
                        logp_new = logp_u.squeeze(1).squeeze(-1)
                        value_new = val_u.squeeze(1).squeeze(-1)
                        # ★버그fix(2026-08-30): cl_l(consumer_loss)을 언팩만 하고 aux 에 안 더해
                        #   VESSEL_COMM_CONSUMER_COEF 가 이 학습기에서 항상 무효였다. 기본값 0 이라 기존 run 과는 비트동일.
                        aux = (cfg.MSG_L2_COEF * msg_reg + cfg.GOAL_COMM_COEF * gl_l
                               + cfg.INTENT_COEF * it_l + cfg.THREAT_COEF * tt_l + cfg.ROLE_COMM_COEF * rl_l
                               + cfg.COMM_CONSUMER_COEF * cl_l)
                        # ★2026-09-05 fix(opt-in): 게이트 개방 페널티(H1a value-of-information≥0 안전장치)는
                        #   지금까지 main.py 에만 있고 이 GPU 학습기엔 없었다 = 죽은 knob. 기본은 미적용(비트동일),
                        #   VESSEL_MSG_GATE_APPLY=1 일 때만 더한다. comm-OFF 팔은 others_msg≡0 이라 무관(anti-rigging).
                        if _gate_apply and cfg.MSG_GATE_COEF != 0.0:
                            aux = aux + cfg.MSG_GATE_COEF * (policy.ctr_actor.gate_open_sum()
                                                             + policy.critic.gate_open_sum())
                        # ★통합 상태복원 (2026-09-04): 반환 시그니처 불변 — 속성으로 수령
                        if cfg.STATE_RECON_COEF > 0.0:
                            _sr_l, _sr_raw = policy._last_state_recon
                            aux = aux + cfg.STATE_RECON_COEF * _sr_l
                            if _sr_raw:   # {} = state_recon 을 안 돈 미니배치(라벨 없음) → 예전처럼 집계 제외
                                # float 또는 0-d 텐서 둘 다 float64 로 정확 변환. 순차 덧셈(update 당 1회 .tolist(), 위 _sr_acc 메모)
                                _v = torch.stack([torch.as_tensor(_sr_raw.get(_g, 0.0), dtype=torch.float64, device=_blind_dev)
                                                  for _g in _SR_GROUPS])
                                _sr_acc = _v if _sr_acc is None else _sr_acc + _v
                                _sr_n += 1
                        # ★2026-09-25 ON 전용 보조손실 배율(VESSEL_AUX_LOSS_SCALE). 0 = 목적함수를 OFF 와 대칭으로(H1a 전제).
                        #   1.0(기본)이면 곱하지 않음 = 비트동일.
                        if cfg.AUX_LOSS_SCALE != 1.0:
                            aux = aux * cfg.AUX_LOSS_SCALE
                    else:
                        # ctr_actor/critic는 [batch, n_agent, dim] 기대 → n_agent=1로 unsqueeze
                        x_b = fx[mi].unsqueeze(1); g_b = fg[mi].unsqueeze(1); s_b = fsf[mi].unsqueeze(1)
                        om_b = fom[mi].unsqueeze(1); sit_b = fsit[mi].unsqueeze(1); act_b = fact[mi].unsqueeze(1)
                        logp_new, entropy, _, _ = policy.ctr_actor.get_logprob_entropy(
                            x_b, g_b, s_b, om_b, act_b, sit_b)
                        logp_new = logp_new.squeeze(1).squeeze(-1)
                        value_new = policy.critic(x_b, g_b, s_b, om_b, sit_b,
                                                  global_feat=fgf[mi] if fgf is not None else None
                                                  ).squeeze(1).squeeze(-1)
                    ratio = torch.exp(logp_new - flogp[mi])
                    a_mb = fadv[mi]
                    pg1 = ratio * a_mb
                    pg2 = torch.clamp(ratio, 1 - cfg.EPSILON, 1 + cfg.EPSILON) * a_mb
                    policy_loss = -torch.min(pg1, pg2).mean()
                    value_loss = ((value_new - fret[mi]) ** 2).mean()
                    loss = policy_loss + cfg.CRITIC_LOSS_WEIGHT * value_loss - cfg.ENTROPY_BONUS * entropy + aux
                    opt.zero_grad()
                    loss.backward()
                    # ★2026-09-05 붕괴 검출기: ControlActor 의 레이더 인코더에 gradient 가 흐르는가.
                    #   2026-09-04 off_s45 붕괴의 실측 원인 = 이 인코더 출력 ReLU 가 전멸(dying ReLU)해
                    #   **정책이 레이더를 물리적으로 못 보게 된 것**. 근거: step2M -> 6M 가중치 변화가
                    #   정확히 0.000000 (같은 런의 critic 레이더는 2105, 정상 시드는 1655~2154).
                    #   장애물을 알려주는 채널은 레이더가 유일하므로 장애물 충돌로 터진다(oColl 57~74%).
                    #   같은 지문을 3건 찾음(m2_S_off_s45 / m2_F1base_s42 / tb_s46) = 재현되는 실패 모드.
                    #   조용히 16M 을 태우고 나중에 '학습 실패 시드'로 버려지던 것을 *학습 중에* 잡는다.
                    # ★2026-09-26 GPU 상주 판정(sync 0, 커널 ~11개/미니배치). 의미 변화는 정확히 두 가지 —
                    #   ① 경고 출력 시점: 임계 도달 미니배치 즉시 → 그 update 가 끝난 뒤(지연 ≤ 미니배치 수 = 1 update 이내).
                    #      한 update 안에서 두 번 도달하면 한 줄만 찍힌다(연속 0 이 update 경계를 넘어 이어져 초반에 W 도달 →
                    #      리셋 → 같은 update 안에서 다시 W 도달하는 경우; 옛 코드는 두 줄. 리뷰 2026-09-26 지적. 가중치·CSV 무관).
                    #   ② 경고문의 'ctr_actor 전체 grad': 텐서별 float32 L1 을 float64 로 합친 값. 예전엔 python float64 순차합.
                    #      .3e 표기에선 병적인 경우 빼곤 같은 숫자. 카운터 규칙(연속 0, 0 아니면 리셋, update 경계 넘어 이어짐)은 그대로.
                    #   grad 는 읽기만 한다 → 가중치·Adam·곡선(골든)에 영향 없음.
                    _g_enc = [_p.grad for _p in _enc_params if _p.grad is not None]
                    _g_ctr = [_p.grad for _p in _ctr_params if _p.grad is not None]
                    _radar_g = _l1_sum(_g_enc, torch.float32)
                    _rg = _l1_sum(_g_ctr, torch.float64)
                    _blind_run_t = torch.where(_radar_g == 0.0, _blind_run_t + 1, _i64_zero)
                    _now_hit = (_blind_run_t == _BLIND_WARN)
                    _blind_rg_t = torch.where(_now_hit, _rg, _blind_rg_t)     # 도달 미니배치의 값을 스냅샷
                    _blind_hit_t = _blind_hit_t | _now_hit
                    # ★2026-09-07 진단 텔레메트리(opt-in, VESSEL_GRAD_TELEMETRY=1 · 기본 off = 비트동일):
                    #   clip_grad_norm_ 은 policy.parameters() *전체*를 한 벡터로 자른다. 통신 ON 팔에만 있는
                    #   보조 손실(MSG_L2·StateRecon 등) 기울기가 그 벡터에 얹히면 clip 계수가 작아져
                    #   ControlActor 의 실효 학습률까지 같이 눌린다 — OFF 팔엔 없는 비대칭. 모듈별 norm 과
                    #   clip 계수를 업데이트마다 찍어 그 크기를 잰다.
                    if _grad_tele:
                        def _gn(ps):
                            _s = [(_p.grad.detach().float() ** 2).sum() for _p in ps if _p.grad is not None]
                            return float(torch.stack(_s).sum().sqrt()) if _s else 0.0   # OFF 팔은 msg grad 없음
                        _gt = {'ctr': _gn(policy.ctr_actor.parameters()),
                               'cri': _gn(policy.critic.parameters()),
                               'msg': _gn(policy.msg_actor.parameters()),
                               'comm': _gn(list(policy.msg_encoder.parameters()) + list(policy.attn.parameters())),
                               'aux': _gn([_p for _n, _m in policy.named_children()
                                           if _n in ('intent_decoder', 'threat_decoder', 'goal_decoder',
                                                     'role_decoder', 'state_recon')
                                           for _p in _m.parameters()]),
                               'total': _gn(policy.parameters())}
                        _gt['clip'] = min(1.0, cfg.MAX_GRAD_NORM / max(_gt['total'], 1e-12))
                        for _k, _v in _gt.items():
                            _gacc[_k] = _gacc.get(_k, 0.0) + _v
                        _gacc['n'] = _gacc.get('n', 0) + 1
                    # ★2026-09-07 모듈별 clip (VESSEL_CLIP_PER_MODULE=1, 기본 0 = 기존 동작 비트동일).
                    #   왜 — clip_grad_norm_(policy.parameters()) 는 *전체를 한 벡터*로 보고 자른다.
                    #     통신 ON 팔에만 있는 보조 손실(StateRecon 등)이 msg_actor 기울기를 키우면
                    #     clip 계수가 작아져 ControlActor 의 실효 학습률까지 같이 눌린다 — OFF 엔 없는 비대칭.
                    #     실측(0.4M, 시드43, 6번째 업데이트): ON msg=31.25 total=31.42 clip=0.018 → ctr 실효 0.005
                    #                                        OFF msg=0     total=2.04  clip=0.329 → ctr 실효 0.088  (17배)
                    #     StateRecon 만 끄면 ON 도 0.115 로 회복 → 원인 확정.
                    #   해법 — 세 망을 각각 MAX_GRAD_NORM 으로 자른다. 한 망이 커도 다른 망의 스텝이 안 줄어든다.
                    #     나머지(통신 집계·보조 디코더)는 한 덩어리로 묶어 같은 한도를 건다.
                    #   (2026-09-26: 그룹 목록 _clip_groups 는 루프 밖에서 한 번 만듦 — 위 메모. 순서·구성원 동일)
                    if _clip_per_module:
                        for _ps in _clip_groups:
                            if _ps:
                                nn.utils.clip_grad_norm_(_ps, cfg.MAX_GRAD_NORM)
                    else:
                        nn.utils.clip_grad_norm_(policy.parameters(), cfg.MAX_GRAD_NORM)
                    opt.step()
            _tsync(); _t_upd += time.perf_counter() - _t1

            # [blind] update 당 1회 호스트 동기화 — 이번 update 안에 임계 도달했으면 경고 (문구 불변, 위 메모 ①②)
            _hit, _rg_hit = torch.stack([_blind_hit_t.double(), _blind_rg_t]).tolist()
            if _hit:
                print(f'[blind] ControlActor 레이더 인코더 gradient 가 {_BLIND_WARN} 미니배치 연속 0 임. '
                      f'정책이 레이더를 못 보는 상태(dying ReLU)로 굳는 중일 수 있음 — '
                      f'2026-09-04 off_s45 붕괴와 같은 지문. ctr_actor 전체 grad={_rg_hit:.3e}. '
                      f'장애물 충돌률(oColl)을 확인할 것.', flush=True)
                _blind_hit_t.zero_()

            if _grad_tele and _gacc.get('n'):
                _n = _gacc['n']
                print('[grad] upd=%d  norm  ctr=%.3f  cri=%.3f  msg=%.3f  comm=%.4f  aux=%.3f  total=%.3f'
                      '  | clip=%.4f  → ctr 실효 스텝 = ctr×clip = %.4f' % (
                          update_i, _gacc['ctr'] / _n, _gacc['cri'] / _n, _gacc['msg'] / _n, _gacc['comm'] / _n,
                          _gacc['aux'] / _n, _gacc['total'] / _n, _gacc['clip'] / _n,
                          _gacc['ctr'] / _n * _gacc['clip'] / _n), flush=True)
                _gacc = {}
            update_i += 1
            # ★중간 체크포인트(2026-08-10): --ckpt_every M마다 저장 → 각 지점을 frozen eval로 찍어
            #   '신뢰할 수 있는 학습곡선'(에피소드 return) 생성. 학습 창 aliasing 우회.
            if _ck_div and args.save:
                mark = int(total_decisions // _ck_div)
                if mark > ckpt_mark:
                    if mark > ckpt_mark + 1:   # ★2026-09-05 fix: 건너뛴 마크를 로그로 남김(조용한 구멍 방지)
                        print(f"[ckpt] 경고: 마크 {ckpt_mark+1}~{mark-1} 건너뜀(rollout 이 간격보다 큼) "
                              f"― 해당 step 파일 없음", flush=True)
                    ckpt_mark = mark
                    cp = f"{os.path.splitext(args.save)[0]}.step{mark * args.ckpt_every:g}M.pt"
                    torch.save({'model_state_dict': policy.state_dict(), 'arm': args.arm,
                                'comm_active': bool(comm_active),   # ★저장 시점 실효 통신 (ON 팔의 .step9M.pt = OFF 모델 판별용)
                                'seed': args.seed, 'steps': total_decisions,
                                'value_norm': vnorm.state(), 'cfg_snapshot': _cfg_snapshot(),
                                'optimizer_state_dict': opt.state_dict()}, cp)
            # ★매 update 조밀 로깅: raw reward + EMA(깨끗한 곡선). ~2400 point/16M run.
            if csv_f:
                raw_r = float(S['rew'].mean().item())
                ema_r = raw_r if ema_r is None else 0.02 * raw_r + 0.98 * ema_r
                csv_f.write(f"{total_decisions},{raw_r:.5f},{ema_r:.5f}\n")
                if update_i % 20 == 0:
                    csv_f.flush()
            if aux_f and _sr_n > 0:
                _vals = _sr_acc.tolist()          # update 당 1회 sync. 나눗셈·.6f 는 예전과 같이 python 에서
                aux_f.write(f"{total_decisions}," + ",".join(f"{_v / _sr_n:.6f}" for _v in _vals) + "\n")
                _sr_acc = None
                _sr_n = 0
                if update_i % 20 == 0:
                    aux_f.flush()
            # ★통신 텔레메트리: comm 이 실제로 켜져 있을 때만. 실패해도 학습은 계속한다.
            if _tele_f is not None and comm_active and update_i % _tele_every == 0:
                try:
                    _tv = comm_telemetry(policy, env, fs.get(), goal, self_s, sit,
                                         args.max_partners, _tele_gen, float(vg.RADAR_RANGE))
                    _tele_f.write(f"{total_decisions}," +
                                  ",".join(f"{_tv.get(c, float('nan')):.6g}" for c in comm_tele_cols(policy)) + "\n")
                    if update_i % 20 == 0:
                        _tele_f.flush()
                except Exception as _e:
                    print(f"[telemetry] 실패(무시하고 계속): {type(_e).__name__}: {_e}", flush=True)
                    _tele_f.close(); _tele_f = None
            if update_i % 5 == 0:
                # ★종료 에피소드 기준 % (이전의 전-agent-step 분모는 running이 지배해 커브가 안 읽혔음)
                term = outcome_counts[1:].sum().clamp(min=1)
                pct = (outcome_counts[1:] / term * 100).tolist()      # [goal, vColl, oColl, TO]
                eps = int(term.item())
                mean_len = (5 * T * E * N) / max(eps, 1)              # 윈도우 agent-결정 / 종료 수
                # ★재개 보정(2026-08-31): t_start 는 재시작하는데 total_decisions 는 이어받아
                #   그대로 나누면 dec/s 가 수십배로 부풀려 진행이 정상인 것처럼 보임.
                sps = (total_decisions - args.resume_at) / max(time.time() - t_start, 1e-6)
                # ★2026-09-26 VESSEL_TIMING=1 일 때만 창(5 update) 평균 rollout/update 초를 뒤에 붙임. 0 이면 빈 문자열 = 줄 불변.
                _tsfx = f" | roll={_t_roll/5:.1f}s upd={_t_upd/5:.1f}s" if _timing else ""
                print(f"[{args.arm}] dec={total_decisions/1e6:.2f}M | ep={eps} len~{mean_len:.0f} | "
                      f"goal={pct[0]:.1f}% vColl={pct[1]:.1f}% oColl={pct[2]:.1f}% TO={pct[3]:.1f}% | "
                      f"R={S['rew'].mean().item():.3f} | {sps:.0f} dec/s" + _tsfx)
                _t_roll = _t_upd = 0.0
                outcome_counts.zero_()

    finally:
        if csv_f:
            csv_f.flush(); csv_f.close()
        if aux_f:
            aux_f.flush(); aux_f.close()
        if _tele_f:
            _tele_f.flush(); _tele_f.close()
    # save (Unity CNNPolicy 호환 state_dict)
    save = args.save or f"vessel_gym_{args.arm}_s{args.seed}.pt"
    torch.save({'model_state_dict': policy.state_dict(), 'arm': args.arm, 'seed': args.seed,
                'comm_active': bool(args.arm == 'ON' and total_decisions >= args.comm_on_at),
                'steps': total_decisions, 'value_norm': vnorm.state(), 'cfg_snapshot': _cfg_snapshot(),
                'optimizer_state_dict': opt.state_dict()}, save)
    print(f"saved -> {save} ({total_decisions/1e6:.2f}M decisions, {(time.time()-t_start)/60:.1f}min)")


if __name__ == '__main__':
    main()
