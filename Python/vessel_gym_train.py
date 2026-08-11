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
import os, sys, time, argparse
import torch
import torch.nn as nn

import config as cfg
import vessel_gym as vg
from networks import CNNPolicy

GOAL_SIZE = cfg.GOAL_SIZE
FRAMES = cfg.FRAMES
STATE = cfg.STATE_SIZE       # 360
MSG_DIM = cfg.MSG_DIM


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
        if done.any():                             # done 에이전트는 3프레임 리셋
            d = done.unsqueeze(-1).unsqueeze(-1)
            self.buf = torch.where(d, radar.unsqueeze(2).expand_as(self.buf), self.buf)
    def get(self):
        E, N = self.buf.shape[0], self.buf.shape[1]
        return self.buf.reshape(E, N, FRAMES * STATE)


def make_others_msg(env, arm, E, N, device):
    """arm별 others_msg [E,N,MSG_DIM] (상수 입력)."""
    om = torch.zeros(E, N, MSG_DIM, device=device)
    if arm == 'ORACLE':
        pg = env.partner_goals_oracle()            # [E,N,2]
        om[..., :GOAL_SIZE] = pg
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


_THREAT_ANG = None
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


def comm_gather(policy, env, x, goal, self_s, sit, K, send_mask=None, recv_mask=None):
    """★배치 학습형 comm (2026-08): 각 배의 COMM_RANGE 내 nearest-K 파트너 메시지를 pos_ground 집계.
    evaluate_actions(update)의 pos_ground 분기와 *동일 함수형* → PPO ratio 유효 (mirror 검증 대상).
    Returns:
      others_msg [E,N,MSG_DIM]  — acting(ctr_actor/critic)용
      partner tensors (px,pg,ps,pmask,prelpos,psit) [E,N,K,·] — update evaluate_actions용(sender→receiver grad)
    x/goal/self_s [E,N,dim], sit [E,N] long. env.pos [E,N,2], env.heading [E,N] deg.
    send_mask/recv_mask [N] 또는 [E,N] bool — 혼합 함대 평가 전용
      (기본 None=전원 통신, 학습 동작 불변). [E,N]을 쓰면 환경마다 다른 비율을 동시에 평가할 수 있다.
      send_mask=False인 배는 파트너 후보에서 빠져 아무도 그 배의 메시지를 못 받는다.
      recv_mask=False인 배는 받은 메시지가 0이 된다(통신 OFF 팔과 동일 입력)."""
    E, N = x.shape[0], x.shape[1]
    dev = x.device
    pos = env.pos                                              # [E,N,2]
    hdg = env.heading                                          # [E,N] deg
    COMM_R = cfg.COMM_RANGE
    d = torch.cdist(pos, pos)                                  # [E,N,N]
    BIG = 1e9
    d = d + torch.eye(N, device=dev).unsqueeze(0) * BIG        # 자기 제외
    d = torch.where(d <= COMM_R, d, torch.full_like(d, BIG))   # 범위 밖 제외
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
    # others_msg: msg_actor(파트너) → msg_encoder(pos_ground) → masked mean (evaluate_actions 미러)
    px_f = px.reshape(E * N, Kc, -1); pg_f = pg.reshape(E * N, Kc, -1)
    ps_f = ps.reshape(E * N, Kc, -1); psit_f = psit.reshape(E * N, Kc)
    prel_f = prelpos.reshape(E * N, Kc, -1); pmask_f = pmask.reshape(E * N, Kc, 1)
    msg_part = policy.msg_actor(px_f, pg_f, ps_f, psit_f)                     # [E*N,Kc,MSG_DIM]
    localized = policy.msg_encoder(torch.cat([prel_f, msg_part], dim=-1))     # [E*N,Kc,MSG_DIM]
    Kcount = pmask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
    others_msg = ((localized * pmask_f).sum(dim=1, keepdim=True) / Kcount).reshape(E, N, -1)
    return others_msg, (px, pg, ps, pmask, prelpos, psit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='OFF', choices=['OFF', 'ORACLE', 'ON'])  # ON=학습형 comm
    ap.add_argument('--comm_on_at', type=int, default=0)   # ON arm: comm 켜는 decision 임계(curriculum). 0=처음부터
    ap.add_argument('--max_partners', type=int, default=cfg.MAX_COMM_PARTNERS)  # msg 처리: 4=aggregation, 1=nearest-1
    ap.add_argument('--ckpt_every', type=float, default=0.0)  # M단위 중간 체크포인트(0=끄기)
    ap.add_argument('--steps', type=int, default=1_000_000)   # 총 env-decision (환경당)
    ap.add_argument('--envs', type=int, default=None)
    ap.add_argument('--vessels', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rollout', type=int, default=64)        # rollout 길이(결정)
    ap.add_argument('--ring', type=float, default=0.7)
    ap.add_argument('--save', default=None)
    ap.add_argument('--csv', default=None)   # 조밀 학습곡선 CSV 경로(None이면 save 기반 자동)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    E = args.envs or (1024 if device == 'cuda' else 64)
    N = args.vessels
    torch.manual_seed(args.seed)

    # commgate 수요 보상 (ORACLE·OFF 동일)
    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device=device, seed=args.seed,
                            ring_scale=args.ring, crossing=2, risk_range=420.0,
                            # ★reward#1 fix 2026-08: per-pair 벌점을 risk³로 집중(exp 1.6→3.0)+계수↓(-0.3→-0.15)
                            #   → 중간위험 다중선박 통과 허용, 고위험만 강함. (colcourse/proximity 집중과 짝)
                            farfield_coef=0.5, perpair_coef=-0.15, perpair_exp=3.0)
    policy = CNNPolicy(MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, FRAMES).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.LEARNING_RATE)

    fs = FrameStack(E, N, device)
    obs = env.reset()
    radar, goal, self_s, sit = parse_obs(obs)
    fs.reset_all(radar)

    # ── 혼합 함대 학습(선택) ──────────────────────────────────────────────
    # 함대의 일부 선박이 통신 장비 없이 *학습 단계부터* 항해한다. 평가할 때만 통신을
    # 끊으면 그 배는 불리해지는 게 아니라 겪어본 적 없는 상황에 놓일 뿐이라, 통신의
    # 값어치를 재려면 없는 채로 키워야 한다. 공유 신경망 하나가 두 처지를 다 배운다.
    #   VESSEL_NOCOMM_SWEEP="2,4,6,8,10,12,14"  환경마다 다른 비율을 심는다(한 번의 학습으로 전 비율 커버)
    #   VESSEL_NOCOMM_MODE=radar|rx|mix         못 보내고 못 받음 / 듣기만 함 / 환경마다 번갈아
    send_mask = recv_mask = None
    _sweep = os.environ.get('VESSEL_NOCOMM_SWEEP', '').strip()
    if _sweep:
        _ks = [int(s) for s in _sweep.split(',') if s.strip()]
        _mode = os.environ.get('VESSEL_NOCOMM_MODE', 'mix').lower()
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

    T = args.rollout
    total_decisions = 0
    outcome_counts = torch.zeros(5, device=device)
    t_start = time.time()
    update_i = 0
    ckpt_mark = 0
    # ★조밀 로깅(2026-08): 매 update마다 (step, raw_reward, ema) CSV 기록 → 깨끗한 학습곡선용(참고 figure 스타일).
    csv_path = args.csv or (os.path.splitext(args.save)[0] + '_curve.csv' if args.save else None)
    csv_f = open(csv_path, 'w', encoding='utf-8') if csv_path else None
    if csv_f:
        csv_f.write('step,raw_reward,ema_reward\n')
    ema_r = None

    while total_decisions < args.steps:
        # ─── rollout ───
        # ★comm curriculum: ON arm이고 comm_on_at 넘으면 학습형 comm 활성(이 rollout 내내 일관 → PPO 정합)
        comm_active = (args.arm == 'ON' and total_decisions >= args.comm_on_at)
        keys = ['x', 'goal', 'self', 'sit', 'om', 'act', 'logp', 'val', 'rew', 'done', 'trunc']
        if comm_active:
            keys += ['px', 'pg', 'ps', 'pmask', 'prel', 'psit']
        use_threat = comm_active and cfg.THREAT_COEF > 0.0
        if use_threat:
            keys += ['othr', 'othrm']
        buf = {k: [] for k in keys}
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
                value = policy.critic(x, goal, self_s, om, sit).squeeze(-1)
            obs, reward, done, outcome = env.step(action)                # env엔 tanh action 적용
            for oc in range(5):
                outcome_counts[oc] += (outcome == oc).sum()
            buf['x'].append(x); buf['goal'].append(goal); buf['self'].append(self_s)
            # ★act = pre-tanh raw 저장(update가 그대로 재사용 → PPO ratio 정합)
            buf['sit'].append(sit); buf['om'].append(om); buf['act'].append(action_raw)
            buf['logp'].append(logp.squeeze(-1)); buf['val'].append(value)
            buf['rew'].append(reward); buf['done'].append(done.float())
            buf['trunc'].append(torch.zeros_like(done.float()))
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
        with torch.no_grad():
            om = make_others_msg(env, args.arm, E, N, device)
            last_v = policy.critic(fs.get(), goal, self_s, om, sit).squeeze(-1)

        # stack [T,E,N,...]
        S = {k: torch.stack(v) for k, v in buf.items()}
        returns, adv = batched_gae(S['rew'], S['val'], S['done'], S['trunc'], last_v,
                                   cfg.DISCOUNT_FACTOR, cfg.GAE_LAMBDA)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # flatten [T*E*N, ...]
        def flat(t): return t.reshape(-1, *t.shape[3:]) if t.dim() > 3 else t.reshape(-1)
        fx = flat(S['x']); fg = flat(S['goal']); fsf = flat(S['self']); fsit = flat(S['sit'])
        fom = flat(S['om']); fact = flat(S['act']); flogp = flat(S['logp'])
        fret = flat(returns); fadv = flat(adv)
        M = fx.shape[0]
        if comm_active:
            fpx = flat(S['px']); fpg = flat(S['pg']); fps = flat(S['ps'])
            fpmask = flat(S['pmask']); fprel = flat(S['prel']); fpsit = flat(S['psit'])
        if use_threat:
            fothr = flat(S['othr']); fothrm = flat(S['othrm'])

        # ─── PPO update ───
        idx_all = torch.randperm(M, device=device)
        mb = cfg.MINIBATCH_SIZE
        for _ in range(cfg.N_EPOCH):
            for s in range(0, M, mb):
                mi = idx_all[s:s + mb]
                aux = 0.0
                if comm_active:
                    # ★학습형 comm: evaluate_actions가 파트너 obs로 others_msg 재계산(sender→receiver grad) + aux손실
                    # ★threat-relay: own_threat 라벨 제공 → 메시지가 '내 레이더가 본 위협' 인코딩(THREAT_COEF).
                    othr_b = fothr[mi].unsqueeze(1) if use_threat else None
                    othrm_b = fothrm[mi].unsqueeze(1) if use_threat else None
                    val_u, logp_u, entropy, msg_reg, it_l, tt_l, gl_l, rl_l, cl_l = policy.evaluate_actions(
                        fx[mi], fg[mi], fsf[mi], fpx[mi], fpg[mi], fps[mi], fpmask[mi], fprel[mi],
                        fact[mi], own_threat=othr_b, own_threat_mask=othrm_b,
                        situation=fsit[mi], partner_situations=fpsit[mi])
                    logp_new = logp_u.squeeze(1).squeeze(-1)
                    value_new = val_u.squeeze(1).squeeze(-1)
                    aux = (cfg.MSG_L2_COEF * msg_reg + cfg.GOAL_COMM_COEF * gl_l
                           + cfg.INTENT_COEF * it_l + cfg.THREAT_COEF * tt_l + cfg.ROLE_COMM_COEF * rl_l)
                else:
                    # ctr_actor/critic는 [batch, n_agent, dim] 기대 → n_agent=1로 unsqueeze
                    x_b = fx[mi].unsqueeze(1); g_b = fg[mi].unsqueeze(1); s_b = fsf[mi].unsqueeze(1)
                    om_b = fom[mi].unsqueeze(1); sit_b = fsit[mi].unsqueeze(1); act_b = fact[mi].unsqueeze(1)
                    logp_new, entropy, _, _ = policy.ctr_actor.get_logprob_entropy(
                        x_b, g_b, s_b, om_b, act_b, sit_b)
                    logp_new = logp_new.squeeze(1).squeeze(-1)
                    value_new = policy.critic(x_b, g_b, s_b, om_b, sit_b).squeeze(1).squeeze(-1)
                ratio = torch.exp(logp_new - flogp[mi])
                a_mb = fadv[mi]
                pg1 = ratio * a_mb
                pg2 = torch.clamp(ratio, 1 - cfg.EPSILON, 1 + cfg.EPSILON) * a_mb
                policy_loss = -torch.min(pg1, pg2).mean()
                value_loss = ((value_new - fret[mi]) ** 2).mean()
                loss = policy_loss + cfg.CRITIC_LOSS_WEIGHT * value_loss - cfg.ENTROPY_BONUS * entropy + aux
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.MAX_GRAD_NORM)
                opt.step()

        update_i += 1
        # ★중간 체크포인트(2026-08-10): --ckpt_every M마다 저장 → 각 지점을 frozen eval로 찍어
        #   '신뢰할 수 있는 학습곡선'(에피소드 return) 생성. 학습 창 aliasing 우회.
        if args.ckpt_every > 0 and args.save:
            mark = int(total_decisions // int(args.ckpt_every * 1e6))
            if mark > ckpt_mark:
                ckpt_mark = mark
                cp = f"{os.path.splitext(args.save)[0]}.step{mark * args.ckpt_every:g}M.pt"
                torch.save({'model_state_dict': policy.state_dict(), 'arm': args.arm,
                            'seed': args.seed, 'steps': total_decisions}, cp)
        # ★매 update 조밀 로깅: raw reward + EMA(깨끗한 곡선). ~2400 point/16M run.
        if csv_f:
            raw_r = float(S['rew'].mean().item())
            ema_r = raw_r if ema_r is None else 0.02 * raw_r + 0.98 * ema_r
            csv_f.write(f"{total_decisions},{raw_r:.5f},{ema_r:.5f}\n")
            if update_i % 20 == 0:
                csv_f.flush()
        if update_i % 5 == 0:
            # ★종료 에피소드 기준 % (이전의 전-agent-step 분모는 running이 지배해 커브가 안 읽혔음)
            term = outcome_counts[1:].sum().clamp(min=1)
            pct = (outcome_counts[1:] / term * 100).tolist()      # [goal, vColl, oColl, TO]
            eps = int(term.item())
            mean_len = (5 * T * E * N) / max(eps, 1)              # 윈도우 agent-결정 / 종료 수
            sps = total_decisions / (time.time() - t_start)
            print(f"[{args.arm}] dec={total_decisions/1e6:.2f}M | ep={eps} len~{mean_len:.0f} | "
                  f"goal={pct[0]:.1f}% vColl={pct[1]:.1f}% oColl={pct[2]:.1f}% TO={pct[3]:.1f}% | "
                  f"R={S['rew'].mean().item():.3f} | {sps:.0f} dec/s")
            outcome_counts.zero_()

    if csv_f:
        csv_f.close()
    # save (Unity CNNPolicy 호환 state_dict)
    save = args.save or f"vessel_gym_{args.arm}_s{args.seed}.pt"
    torch.save({'model_state_dict': policy.state_dict(), 'arm': args.arm, 'seed': args.seed}, save)
    print(f"saved → {save} ({total_decisions/1e6:.2f}M decisions, {(time.time()-t_start)/60:.1f}min)")


if __name__ == '__main__':
    main()
