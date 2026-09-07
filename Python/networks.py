"""
Vessel Navigation Policy Network
- MessageActor: observation → 6D message
- ControlActor: observation + gate·others_msg → action
- Critic: observation + gate·others_msg → value

PPO 구현은 CleanRL 방식을 따름 (검증된 구현)

obs 계약(369D 중 네트워크 입력 366D): radar(360 raw ray, frame-stack ×3 → RadarEncoder Conv1D 압축) + goal(2) + self(4).
ARPA 제거(2026-06-05): 충돌 기하는 360 raw ray + frame-stack(RadarEncoder Conv1D가 bearing-rate 학습)이 대체. COLREGs one-hot도 제거됨(vessel-label leak).
★radar: C# min-pool(360→30) 제거 → 360 raw ray를 RadarEncoder(Conv1D 원형패딩)가 학습형 압축(RADAR_FEAT_DIM, 기본 30D = 옛 섹터수와 동일 차원이되 학습형).

★ 통신 credit assignment 수정 (이전 버그):
이전 evaluate_actions는 PPO 배치가 n_agent=1이라 straight-through self-loop만 타서
sender→receiver gradient가 0이었음. 이제 파트너 obs를 저장해두고 update 때 MessageActor를
파트너 obs로 재실행(rollout과 동일 집계: sum/mean/scale/attention/pos_ground 미러)하여
"내 메시지가 옆 배 회피를 도왔나" gradient가 흐르게 함.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from config import (STATE_SIZE, RADAR_FEAT_DIM, USE_COMMUNICATION,
                    SELF_STATE_SIZE, GOAL_SIZE, USE_ATTENTION, ATTN_DIM,
                    INTENT_COEF, INTENT_K, THREAT_COEF, THREAT_K, GOAL_COMM_COEF,
                    ROLE_COMM_COEF, USE_MOE, NUM_COLREGS_SITUATIONS, MAX_COMM_PARTNERS,
                    COMM_CONSUMER_COEF, COMM_CONSUMER_K, COMM_CONSUMER_COUPLING, USE_ORACLE,
                    SITUATION_INPUT, MOE_WIDTH, MOE_SHARED, POS_GROUND, STATE_RECON_COEF,
                    CENTRAL_CRITIC)


_RADAR_LEAKY = os.environ.get('VESSEL_RADAR_ACT', 'relu').lower() == 'leaky'

# ★2026-09-07 레이더 인코더 head 스위치: VESSEL_RADAR_HEAD=flat(기본, 비트동일) | bottleneck
#   왜 — fc(2880→30) 붕괴의 원인이 *fan-in* 임을 측정으로 확정함.
#     Adam t=1 은 모든 가중치를 정확히 lr 만큼 움직이므로 유닛 합의 변화량은 |Δz| = lr·‖x‖₁ (정확).
#     초기화 직후 conv3 출력(2880, 전부 ≥0)의 ‖x‖₁≈70 → |Δz|≈0.021 인데 |z| 중앙값은 0.015 →
#     S/|z| = 1.37 (시드57) / 1.59 (시드44). conv1~3 은 0.01~0.06. fc 만 한 걸음에 부호가 뒤집힘.
#     학습이 진행돼 conv3 가 희소해지면 S/|z|=0.02 로 안전 → "첫 몇 걸음에만 죽는다"와 일치.
#   해법 — 더하는 수 자체를 줄인다. conv3 뒤 1×1 conv 로 채널 64→8 (방위 45칸 보존) → fc 360→30.
#     S/|z| 0.05 (conv 와 같은 수준). 파라미터 86,430→11,350. 출력 30 은 그대로라 fc2·MoE·통신 불변.
#     기각: 중간층 2880→256→30 (앞층 fan-in 그대로 1.37), fc 앞 LayerNorm (‖x‖₁ 2,300 으로 악화),
#           초기값 축소 (|z|↓ 라 비율 악화), global pool (방위 소실).
#   활성함수 스위치(VESSEL_RADAR_ACT) 와 독립. 체크포인트 shape 이 바뀌므로 from-scratch.
_RADAR_HEAD = os.environ.get('VESSEL_RADAR_HEAD', 'flat').lower()
_RADAR_BOTTLENECK_CH = int(os.environ.get('VESSEL_RADAR_BOTTLENECK_CH', 8))

# ★2026-09-07 attention 토큰 안 메시지 게인: VESSEL_MSG_TOKEN_GAIN (기본 1.0 = 비트동일)
#   왜 — 토큰 = [relpos(3) ‖ msg(6)] 인데 relpos 는 O(1), msg 는 실측 std 0.118 로 8배 작다.
#     k_proj·v_proj 의 기울기는 입력 크기에 비례하므로 relpos 열이 msg 열보다 빠르게 자란다
#     (유용성과 무관한 스케일 경주). 실측 결과:
#       · v 토큰 분산 relpos 몫 0.28 vs msg 몫 0.002 (약 100배) → others_msg 가 사실상 '주소 채널'
#       · α 가 메시지 내용에 무관 — msg 를 난수로 바꿔도 α 가 소수점 3자리까지 동일
#       · 송신자 속도 복원 R²: raw 0.99 → v 토큰 0.59 → others_msg 0.39 (34% 만 도착)
#   해법 — 토큰을 만들 때만 msg 를 상수배해 relpos 와 크기를 맞춘다. k_proj 도 같은 스케일을 보므로
#     "누구 말을 들을지"까지 내용 기반이 된다(v 만 고치는 VESSEL_MSG_GAIN 과 다른 지점).
#   ⚠️상수배여야 한다. 배치 통계 정규화는 rollout(E*N)과 update(미니배치)의 통계가 달라
#     others_msg 가 갈리고 PPO ratio 가 조용히 깨진다.
#   미러 안전: rollout(vessel_gym_train.comm_gather)·update(evaluate_actions) 둘 다
#     같은 aggregate_batch 를 호출하므로 여기 한 곳만 고치면 양쪽에 동일 적용된다.
_MSG_TOKEN_GAIN = float(os.environ.get('VESSEL_MSG_TOKEN_GAIN', 1.0))

# ★2026-09-07 MoE 라우팅 배치화: VESSEL_MOE_FAST=1 (기본 0 = 기존 루프, 비트동일)
#   왜 — 지금 라우팅은 전문가마다 `if mask.any(): core(x[mask])` 를 돈다. 문제는 연산량이 아니라
#     (a) mask.any() 가 GPU→CPU 동기화, (b) x[mask]/z[mask]= 가 데이터 의존 크기라 또 동기화,
#     (c) 부분배치가 작아 커널 런치가 계산보다 비쌈 — 미니배치 1개당 3망×5전문가 = 15회 sync.
#     실측(조타망 forward, batch 2048, 유휴 GPU): 루프 11.78ms vs 배치화 1.62ms vs MoE off 1.54ms.
#     상황 분포가 98.7% 가 상황0 이라 "5등분"이 아니라 "큰 덩어리 1 + 부스러기 4" 였고,
#     그래서 인코더만 루프 밖으로 빼는 것으로는 7.36ms(1.6배) 밖에 안 준다 — 루프 자체를 없애야 한다.
#   방법 — 각 샘플은 어차피 전문가 1개만 통과하므로, 전문가 가중치를 stack 해서 샘플별로 gather 한 뒤
#     baddbmm 한 방에 돌린다. 수학적으로 동일(Linear 는 샘플 독립 연산).
#   ⚠️비트동일은 아니다 — 배치 크기가 달라지면 cuBLAS 가 다른 리덕션 순서를 쓴다. 실측 오차 ~1e-7.
#     rollout·update 가 같은 전역 스위치를 보므로 두 경로는 항상 같은 함수형이다(PPO ratio 안전).
#   ⚠️MOE_SHARED=1 (전문가들이 radar_encoder 를 한 객체로 공유) 일 때만 켜진다. 안 그러면 인코더를
#     배치 전체에 한 번 돌릴 수 없다.
_MOE_FAST = os.environ.get('VESSEL_MOE_FAST', '0') == '1'


def _bmm_linear(cores, sit, h, attr):
    """전문가별 nn.Linear 을 샘플 단위로 적용. cores[k].<attr> 가 Linear. h [M,in] → [M,out].
    루프·동기화 없음. stack 은 매 forward 새로 만들어 state_dict 키를 안 바꾼다(체크포인트 호환)."""
    W = torch.stack([getattr(c, attr).weight for c in cores])        # [K,out,in]
    b = torch.stack([getattr(c, attr).bias for c in cores])          # [K,out]
    return torch.baddbmm(b[sit].unsqueeze(1), h.unsqueeze(1), W[sit].transpose(1, 2)).squeeze(1)


def _bmm_seq2(cores, sit, h, attr):
    """전문가별 nn.Sequential(Linear, ReLU, Linear) 을 샘플 단위로 적용 (Critic.glob_enc 용).
    h [M,S,in] → [M,S,out]. per-ship 축 S 는 그대로 둔 채 전문가만 gather."""
    m0 = [getattr(c, attr)[0] for c in cores]
    m2 = [getattr(c, attr)[2] for c in cores]
    W0 = torch.stack([m.weight for m in m0])[sit]                    # [M,h,in]
    b0 = torch.stack([m.bias for m in m0])[sit].unsqueeze(1)         # [M,1,h]
    W2 = torch.stack([m.weight for m in m2])[sit]                    # [M,out,h]
    b2 = torch.stack([m.bias for m in m2])[sit].unsqueeze(1)         # [M,1,out]
    a = F.relu(torch.baddbmm(b0, h, W0.transpose(1, 2)))
    return torch.baddbmm(b2, a, W2.transpose(1, 2))


def _bmm_layernorm(cores, sit, h, attr):
    """전문가별 nn.LayerNorm 을 샘플 단위로 적용. 정규화는 파라미터 없는 연산이라 한 번에,
    affine(weight·bias)만 전문가별로 gather."""
    ln0 = getattr(cores[0], attr)
    y = F.layer_norm(h, ln0.normalized_shape, None, None, ln0.eps)
    w = torch.stack([getattr(c, attr).weight for c in cores])[sit]
    b = torch.stack([getattr(c, attr).bias for c in cores])[sit]
    return y * w + b


def _moe_fast_on(wrapper):
    """이 wrapper 에서 배치화 경로를 쓸 수 있나 — MoE 켜짐 + 인코더가 한 객체(MOE_SHARED)."""
    if not (_MOE_FAST and getattr(wrapper, 'use_moe', False)):
        return False
    cs = wrapper.cores()
    return len({id(c.radar_encoder) for c in cs}) == 1


def _share_radar_encoder(experts):
    """★공유지각 MoE(MOE_SHARED=1): 전문가들의 radar_encoder를 experts[0] 것으로 통일(모듈 공유).
    지각(코어 파라미터 ~82.5%)은 전체 데이터로 학습, 상황별 특화는 결정부(fc2~head)만.
    parameters()/optimizer는 공유 텐서를 자동 dedup, state_dict는 5벌 동일 사본 저장(load 호환)."""
    for _k in range(1, len(experts)):
        experts[_k].radar_encoder = experts[0].radar_encoder


def _w(n, width, floor=4):
    """iso-parameter MoE용 폭 스케일: 내부 차원 n에 배수 width 적용 (하한 floor).
    width=1.0이면 정확히 n 반환 → 기존 구조와 비트동일. 외부 인터페이스(msg 6D, 행동 2D,
    goal/self/situation 입력)는 스케일 대상이 아님 — 코어 *내부* 차원만."""
    return max(floor, int(round(n * width)))

# ★COLREGs situation 정책 입력 (2026-07-02, VESSEL_SITUATION_INPUT=1): obs[368] 상황(0~4)을 one-hot 5D로
#   세 네트워크 fc2에 직접 입력. 판정은 자기 센서 반경(56m=레이더) 내 기하로 산출 + 실선 ARPA/AIS 동등 정보.
#   통신 ON/OFF 양 arm 동일 적용 = anti-rigging 안전. 기본 OFF = 기존과 비트동일(체크포인트 호환).
#   ⚠️ concat 순서는 항상 [radar, goal, self, (sit_onehot), msg] — msg가 *마지막*이어야
#   fc2 메시지 슬라이스의 [:, -msg_dim:] 인덱싱(×0.1 init·grad 텔레메트리)이 유효.
SIT_INPUT_DIM = NUM_COLREGS_SITUATIONS if SITUATION_INPUT else 0


def _situation_onehot(situation, M, ref):
    """situation [b,n]|[b,n,1]|None → one-hot [M, NUM_COLREGS_SITUATIONS] float.
    None이면 상황 0(None) 취급 — MoE 라우팅의 'None→코어 0' 규약과 일치.
    rollout·update가 같은 저장값(memory situations/p_sits)으로 호출 → PPO ratio 정합."""
    if situation is not None:
        sit = situation.reshape(M).long().clamp(0, NUM_COLREGS_SITUATIONS - 1)
    else:
        sit = torch.zeros(M, dtype=torch.long, device=ref.device)
    return F.one_hot(sit, NUM_COLREGS_SITUATIONS).to(dtype=ref.dtype)


class RadarEncoder(nn.Module):
    """360 raw ray(원형 각도 거리 프로파일)를 *학습형 신경망*으로 RADAR_FEAT_DIM(기본 30) 압축.

    ★C# min-pool(360 ray→30 섹터, 고정·정보손실) 제거 → 모든 ray가 신경망에 들어와
      "무엇을 남길지"를 학습으로 결정(고정 min-pool과 같은 30D로 압축하되 학습형이라 손실 최소화).
    구조 = Conv1D(원형 padding) × 3 + FC:
      - 입력 [M, frames, n_rays]: **frames=시간축을 입력 채널로** → 같은 ray bin의 프레임간
        변화(=bearing-rate, COLREGs 핵심)를 conv가 직접 본다.
      - padding_mode='circular': ray 359 ↔ ray 0 인접(각도 wrap) 존중 → 정면 가로지르는 물체 보존.
      - stride-2 ×3로 360→180→90→45 다운샘플(학습형 pooling, min-pool 아님).
    ★출력 out_dim(=RADAR_FEAT_DIM) = 다운스트림 fc2 입력. 각 네트워크가 독립 인스턴스 보유.
      차원 변경 시 fc2 weight shape 변경 → from-scratch 필요. VESSEL_RADAR_FEAT_DIM로 튜닝.
    """
    def __init__(self, frames, n_rays, out_dim=RADAR_FEAT_DIM, width=1.0):
        super(RadarEncoder, self).__init__()
        self.frames = frames
        self.n_rays = n_rays
        # width: iso-parameter MoE용 채널 배수 (기본 1.0 = 32/64/64 그대로)
        c1, c2, c3 = _w(32, width), _w(64, width), _w(64, width)
        self.conv1 = nn.Conv1d(frames, c1, kernel_size=5, stride=2, padding=2, padding_mode='circular')
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2, padding_mode='circular')
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=3, stride=2, padding=1, padding_mode='circular')
        # ★병목 head(VESSEL_RADAR_HEAD=bottleneck): 1×1 conv 로 채널만 64→8 줄여 fc fan-in 2880→360.
        #   방위 45칸은 그대로. flat(기본)이면 None → 기존과 비트동일(state_dict 키도 동일).
        if _RADAR_HEAD == 'bottleneck':
            self.reduce = nn.Conv1d(c3, _w(_RADAR_BOTTLENECK_CH, width, floor=2), kernel_size=1)
        else:
            self.reduce = None
        # conv flatten 크기는 n_rays에 의존 → 더미 forward로 산출(차원 하드코딩 방지).
        with torch.no_grad():
            _flat = self._conv(torch.zeros(1, frames, n_rays)).shape[1]
        self.fc = nn.Linear(_flat, out_dim)

    # ★2026-09-05 진단·완화 스위치: VESSEL_RADAR_ACT=leaky 면 LeakyReLU(0.01), 기본 relu = 비트동일.
    #   왜 필요한가 — 실측된 붕괴 원인이 이 인코더의 dying ReLU 임:
    #     2026-09-04 off_s45 의 ctr_actor 레이더 인코더가 step2M->6M 사이 정확히 0.000000 변함
    #     (같은 런 critic 레이더는 2105, 정상 시드는 1656/2154). 레이더를 무장애로 바꿔도 |Δaction|=0.
    #     장애물을 알려주는 채널은 레이더뿐이라 장애물 충돌로 터짐(oColl 57~74%).
    #   64런 실측 붕괴율 4/64 = 6.3%. GAE trunc·진행보상 계수·StateRecon 통계와 무관(전부 A/B 로 기각).
    #   ReLU 가 4겹(conv1~3 + fc)이고 fc 뒤가 죽으면 30 유닛 전멸 -> gradient 영구 0 = 회복 불가.
    #   LeakyReLU 는 음수쪽 기울기를 남겨 그 흡수상태를 없앤다. 활성함수 교체는 신경망 설계 변경이라
    #   기본값은 바꾸지 않고 스위치로만 둠(저자 승인 사항).
    def _act(self, t):
        return F.leaky_relu(t, 0.01) if _RADAR_LEAKY else F.relu(t)

    def _conv(self, x):
        a = self._act(self.conv1(x))
        a = self._act(self.conv2(a))
        a = self._act(self.conv3(a))
        if self.reduce is not None:
            a = self._act(self.reduce(a))       # [M, 8, 45] — 채널 병목, 방위 보존
        return a.reshape(a.shape[0], -1)

    def forward(self, x):
        """x: [batch, n_agent, frames*n_rays] 또는 [M, frames*n_rays] → [M, out_dim] (post-ReLU)."""
        x = x.reshape(-1, self.frames, self.n_rays)
        return self._act(self.fc(self._conv(x)))


class IntentDecoder(nn.Module):
    """
    메시지 latent → sender의 미래 K-step 의도(상대변위+heading변화) 예측 (Phase 2, self-supervised).

    msg(msg_dim) → MLP → [K×3] = K개 미래시점의 [국소 starboard변위, 국소 forward변위, Δheading].
    ★정책/가치와 무관한 별도 head → 손실이 MessageActor로만 흘러 "메시지가 미래를 인코딩"하도록 강제.
      receiver의 ControlActor/Critic은 안 건드림 → 게이트로 메시지 무시 가능(H1a 보존).
    표준 init(zero-init 불요): 예측이 0에서 자라야 하나 정책경로 무관이라 무해.
    """
    def __init__(self, msg_dim, intent_k):
        super(IntentDecoder, self).__init__()
        self.intent_k = intent_k
        self.dec = nn.Sequential(
            nn.Linear(msg_dim, 64), nn.ReLU(), nn.Linear(64, intent_k * 3)
        )

    def forward(self, msg):
        return self.dec(msg)   # [..., K*3]


class StateReconDecoder(nn.Module):
    """통합 상태복원 디코더 (2026-09-04, COMM_PLAN.md 4-1) — 라벨을 사람이 고르지 않는다.

    메시지 하나로 sender 상태 전부를 복원하도록 강제:
      그룹 = goal(2) / self_state(4) / situation one-hot(5) / threat(K_t*4, mask) / future(K_i*3, mask)
    ★성분별 z-score: 러닝 mean/var(모멘텀 0.01)로 표준화 — 의도 라벨처럼 수치가 작은 성분이
      MSE에서 자동으로 굶는 것을 막는다(실측: 우현변위 std 0.013 → gradient 0.8%).
    ★그룹별 EMA 정규화 + 동등 평균: 손실 원값이 그룹 간 500배 차이나므로(ROLE 4e-5 vs CONSUMER 0.02)
      각 그룹 손실을 자기 EMA로 나눠 gradient 기여를 지속적으로 균등화.
    STATE_RECON_COEF=0이면 인스턴스 자체가 안 만들어짐(구 체크포인트 strict 로드 호환)."""
    GROUPS = ('goal', 'self', 'sit', 'threat', 'future')

    def __init__(self, msg_dim, threat_k, intent_k, num_sit=5):
        super(StateReconDecoder, self).__init__()
        self.dims = {'goal': GOAL_SIZE, 'self': SELF_STATE_SIZE, 'sit': num_sit,
                     'threat': threat_k * 4, 'future': intent_k * 3}
        self.num_sit = num_sit
        D = sum(self.dims.values())
        self.out_dim = D
        self.net = nn.Sequential(nn.Linear(msg_dim, 128), nn.ReLU(), nn.Linear(128, D))
        # 성분별 러닝 표준화 통계 + 그룹별 손실 EMA
        self.register_buffer('run_mean', torch.zeros(D))
        self.register_buffer('run_var', torch.ones(D))
        self.register_buffer('stat_inited', torch.zeros(1))
        self.register_buffer('loss_ema', torch.ones(len(self.GROUPS)))
        self.momentum = 0.01
        # ★2026-09-05 fix(opt-in): 그룹 EMA 정규화가 *갱신 후* 값으로 나눠 자기억제적이었음
        #   (분모에 현재 배치 손실이 momentum만큼 섞임 → gl 급등 시 비율이 1/momentum=100에서 잘림).
        #   정석은 *갱신 전* 값으로 정규화. 과거 학습 재현을 조용히 깨지 않도록 기본은 옛 동작 유지,
        #   VESSEL_RECON_EMA_PRE=1 로만 교정 동작. state_dict 불변 → 구 ckpt 로드 영향 없음.
        #   ※감사가 함께 지적한 '초기 1.0 프라이어 왜곡'은 해당 없음 — 타깃이 z-score라 init 시
        #     모든 그룹 손실이 E[z^2]≈1에서 출발함(라벨 원단위 500배 차이는 z-score가 이미 제거).
        #     즉 프라이어 1.0은 오히려 정확한 스케일 → ema_inited 버퍼 추가(=구 ckpt 키 불일치) 불요.
        import os as _os_ema
        self.ema_pre = _os_ema.environ.get('VESSEL_RECON_EMA_PRE', '0') == '1'

    def _slices(self):
        out, s0 = {}, 0
        for g in self.GROUPS:
            out[g] = (s0, s0 + self.dims[g]); s0 += self.dims[g]
        return out

    def loss(self, msg, goal, self_state, situation, own_threat, own_threat_mask,
             own_future, own_future_mask):
        """msg [N,1,msg_dim] → (총손실 스칼라, 그룹별 원손실 dict[float] — 로깅용)."""
        N = msg.shape[0]
        sit_oh = F.one_hot(situation.reshape(N).long().clamp(0, self.num_sit - 1),
                           self.num_sit).float().unsqueeze(1)               # [N,1,5]
        tm = torch.ones_like(own_threat) if own_threat_mask is None else own_threat_mask
        fm = torch.ones_like(own_future) if own_future_mask is None else own_future_mask
        tgt = torch.cat([goal, self_state, sit_oh, own_threat, own_future], dim=-1)  # [N,1,D]
        msk = torch.cat([torch.ones_like(goal), torch.ones_like(self_state),
                         torch.ones_like(sit_oh), tm, fm], dim=-1)
        tf = tgt.reshape(N, -1); mf = msk.reshape(N, -1)
        # 러닝 z-score 갱신 (mask 유효 표본 기준)
        if self.training:
            with torch.no_grad():
                cnt = mf.sum(0).clamp(min=1.0)
                bm = (tf * mf).sum(0) / cnt
                bv = (((tf - bm).pow(2)) * mf).sum(0) / cnt
                # ★2026-09-05 fix: 유효 표본이 0인 성분은 통계를 갱신하지 않는다.
                #   기존엔 mask 가 전부 0인 성분도 bm=0·bv=0 으로 EMA 에 섞여 run_mean→0, run_var→1e-8 로
                #   지수 감쇠했다. 그 성분의 mask 가 다시 살아나는 첫 배치에서 z=(tf-0)/1e-3 로 폭발한다.
                #   future(intent) 라벨은 에피소드 경계·버퍼 끝에서 통째로 마스크되므로 실제로 발생하는 경로다.
                valid = (mf.sum(0) > 0)
                # ★2026-09-05 진단 스위치: VESSEL_RECON_LEGACY_STAT=1 이면 수정 전(버그) 동작으로 되돌린다.
                #   버그 = 유효표본 0 인 성분도 bm=0·bv=0 으로 EMA 에 섞여 run_var 가 1e-8 로 지수감쇠 →
                #          그 성분이 다시 유효해지는 첫 배치에서 z=(tf-0)/1e-3 폭발.
                #   future(intent) 라벨이 에피소드 경계에서 통째로 마스크되므로 실제로 발생하는 경로다.
                #   붕괴(2026-09-04 off_s45)의 원인인지 가르기 위한 대조군. 기본 0 = 수정본.
                # (2026-09-07 fix) _os_ln 은 MessageActor.__init__ 의 지역 import 라 여기선 NameError.
                #   통신 ON + STATE_RECON_COEF>0 경로에서만 실행돼 OFF 스모크·OFF 진단배치는 안 걸렸음.
                if os.environ.get('VESSEL_RECON_LEGACY_STAT', '0') == '1':
                    valid = torch.ones_like(valid)
                if float(self.stat_inited) == 0.0:
                    self.run_mean.copy_(torch.where(valid, bm, self.run_mean))
                    self.run_var.copy_(torch.where(valid, bv.clamp(min=1e-8), self.run_var))
                    self.stat_inited.fill_(1.0)
                else:
                    new_m = self.run_mean * (1 - self.momentum) + self.momentum * bm
                    new_v = self.run_var * (1 - self.momentum) + self.momentum * bv.clamp(min=1e-8)
                    self.run_mean.copy_(torch.where(valid, new_m, self.run_mean))
                    self.run_var.copy_(torch.where(valid, new_v, self.run_var))
        sd = self.run_var.clamp(min=1e-6).sqrt()
        z = ((tf - self.run_mean) / sd) * mf                                 # 표준화 타깃
        pred = self.net(msg).reshape(N, -1)
        se = (pred - z).pow(2) * mf
        raw, total = {}, 0.0
        for gi, (g, (a, b)) in enumerate(self._slices().items()):
            gl = se[:, a:b].sum() / mf[:, a:b].sum().clamp(min=1.0)
            raw[g] = float(gl.detach())
            w_pre = self.loss_ema[gi].detach().clone()          # 갱신 *전* 값(교정 경로용)
            if self.training:
                with torch.no_grad():
                    self.loss_ema[gi] = (1 - self.momentum) * self.loss_ema[gi] + self.momentum * gl.detach()
            # ★2026-09-05 fix: ema_pre=1이면 갱신 전 값으로 정규화(자기억제 제거). 기본(0)은 옛 동작.
            w = w_pre if self.ema_pre else self.loss_ema[gi].detach()
            total = total + gl / (w + 1e-4)
        return total / len(self.GROUPS), raw


class ThreatDecoder(nn.Module):
    """
    메시지 latent → sender가 본 top-K 위협 기하 복원 (L1, Phase 2, self-supervised).

    msg(msg_dim) → MLP → [K×4] = K개 최근접 위협의 [sin방위, cos방위, 거리/maxrange, closing].
    ★IntentDecoder 동형 별도 head → 손실이 MessageActor로만 흘러 "메시지가 sender가 본 제3 위협을
      인코딩"하도록 강제. occlusion으로 receiver가 못 보는 위협을 메시지로 복원 → 통신 필수화.
      receiver의 ControlActor/Critic은 안 건드림 → 게이트로 메시지 무시 가능(H1a 보존).
    표준 init(zero-init 불요): 정책경로 무관이라 무해.
    """
    def __init__(self, msg_dim, threat_k):
        super(ThreatDecoder, self).__init__()
        self.threat_k = threat_k
        self.dec = nn.Sequential(
            nn.Linear(msg_dim, 64), nn.ReLU(), nn.Linear(64, threat_k * 4)
        )

    def forward(self, msg):
        return self.dec(msg)   # [..., K*4]


class GoalDecoder(nn.Module):
    """
    메시지 latent → sender의 목적지(goal: 거리·방위) 복원 (L4, Phase 2, self-supervised).

    msg(msg_dim) → MLP → [GOAL_SIZE=2] = sender의 goal(거리 d/(d+k), 방위/180).
    ★IntentDecoder/ThreatDecoder 동형 별도 head → 손실이 MessageActor로만 흘러 "메시지가 sender의
      *어디로 갈지(목적지)*를 인코딩"하도록 강제. 통신거리(420m)>레이더(56m)라 먼 배의 목적지를
      통신으로만 알 수 있음 → receiver가 미리 협응(COLREGs·연료·궤적).
      receiver의 ControlActor/Critic은 안 건드림 → 게이트로 메시지 무시 가능(H1a 보존).
    ★라벨 = sender 자기 goal(이미 obs/evaluate_actions 인자) → 별도 라벨 텐서 불요(self-prediction).
    """
    def __init__(self, msg_dim, goal_size):
        super(GoalDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Linear(msg_dim, 64), nn.ReLU(), nn.Linear(64, goal_size)
        )

    def forward(self, msg):
        return self.dec(msg)   # [..., GOAL_SIZE]


class RoleDecoder(nn.Module):
    """
    메시지 latent → sender의 COLREGs 상황/역할(situation: 0~4) 분류 (Phase 2, self-supervised).

    msg(msg_dim) → MLP → [num_situations] logits = sender의 COLREGs 상황 분류(cross-entropy).
      0=None 1=HeadOn 2=CrossingStandOn 3=CrossingGiveWay 4=Overtaking (COLREGsHandler enum 일치).
    ★IntentDecoder/ThreatDecoder/GoalDecoder 동형 별도 head → 손실이 MessageActor로만 흘러 "메시지가
      sender의 *COLREGs 역할(누가 give-way·누가 stand-on)*"을 인코딩하도록 강제. receiver가 attention으로
      sender 역할을 알면 "쟤가 give-way(3)면 난 stand-on 유지" 식으로 미리 협응 → compliance↑.
      receiver의 ControlActor/Critic은 안 건드림 → 게이트로 메시지 무시 가능(H1a 보존).
    ★라벨 = sender 자기 situation(이미 obs/메모리·evaluate_actions 인자) → 별도 라벨 텐서 불요(self-prediction).
    표준 init(zero-init 불요): 분류가 0에서 자라야 하나 정책경로 무관이라 무해.
    """
    def __init__(self, msg_dim, num_situations):
        super(RoleDecoder, self).__init__()
        self.num_situations = num_situations
        self.dec = nn.Sequential(
            nn.Linear(msg_dim, 64), nn.ReLU(), nn.Linear(64, num_situations)
        )

    def forward(self, msg):
        return self.dec(msg)   # [..., num_situations] logits


class GroundedAttention(nn.Module):
    """
    위치 grounding + single-head attention 집계 (sum/mean 대체).

    - query : receiver의 [self_state ⊕ goal] (= 내 상황) → q_proj
    - key/value : 각 partner의 [relpos(sin,cos,거리) ⊕ msg] (= 어디서 온 어떤 의도) → k_proj/v_proj
    - context = Σ_j softmax(q·k/√d)_j · v_j   (sum의 무차별 합 대신 가중선택)

    출력차원 = msg_dim → ControlActor/Critic 게이트·fc2 메시지슬롯 *불변*.
    ★v_proj 소진폭 init (2026-06-12 채널동결 fix): zero-init은 fc2 메시지슬라이스 zero와 *직렬 곱
      새들*을 만들어 채널 양단 grad가 항등 0 → 1M step 비트동결(체크포인트 포렌식 실측). 소진폭(×0.1)은
      grad를 step1부터 살리면서 초기 영향은 작게 유지. comm 공정성은 구조 강제가 아니라
      comm-OFF arm과의 ground-truth 비교로 검증한다.
    ★aggregate_single(rollout)·aggregate_batch(update)는 *같은 함수형* → 같은 partner 입력에
      같은 결과 → PPO ratio(old_logprob) 유효. batch만 padding을 -inf 마스킹으로 제외.
    """
    def __init__(self, msg_dim, relpos_dim, query_in_dim, d_attn=32):
        super(GroundedAttention, self).__init__()
        self.msg_dim = msg_dim
        self.scale = float(d_attn) ** 0.5
        token_dim = relpos_dim + msg_dim
        self.q_proj = nn.Linear(query_in_dim, d_attn)
        self.k_proj = nn.Linear(token_dim, d_attn)
        self.v_proj = nn.Linear(token_dim, msg_dim)
        with torch.no_grad():
            self.v_proj.weight.mul_(0.1)
            self.v_proj.bias.zero_()

    def aggregate_single(self, q_in, relpos, msg_p):
        """rollout 1-receiver 집계. q_in [Q], relpos [K,R], msg_p [K,M] → context [M].
        실제 파트너만 들어옴(padding 없음) → 마스킹 불필요."""
        query = self.q_proj(q_in)                              # [d]
        token = torch.cat([relpos, msg_p * _MSG_TOKEN_GAIN], dim=-1)   # [K, R+M]
        keys = self.k_proj(token)                              # [K, d]
        vals = self.v_proj(token)                              # [K, M]
        scores = (keys @ query) / self.scale                   # [K]
        alpha = torch.softmax(scores, dim=0)                   # [K]
        return (alpha.unsqueeze(-1) * vals).sum(dim=0)         # [M]

    def aggregate_batch(self, q_in, relpos, msg_part, mask):
        """update 배치 집계. q_in [N,1,Q], relpos [N,K,R], msg_part [N,K,M], mask [N,K,1]
        → context [N,1,M]. padding(mask=0)은 -inf 마스킹으로 softmax 제외, 전무파트너는 0."""
        query = self.q_proj(q_in)                              # [N,1,d]
        token = torch.cat([relpos, msg_part * _MSG_TOKEN_GAIN], dim=-1)   # [N,K,R+M]
        keys = self.k_proj(token)                              # [N,K,d]
        vals = self.v_proj(token)                              # [N,K,M]
        scores = (keys * query).sum(dim=-1) / self.scale       # [N,K]  (query [N,1,d] broadcast)
        scores = scores.masked_fill(mask.squeeze(-1) <= 0, -1e9)  # padding 제외
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)     # [N,K,1]
        context = (alpha * vals).sum(dim=1, keepdim=True)      # [N,1,M]
        has = (mask.sum(dim=1, keepdim=True) > 0).float()      # [N,1,1] 파트너 0이면 context=0
        return context * has


class _MessageActorCore(nn.Module):
    """단일 MessageActor 본체: radar(360 raw ray → RadarEncoder Conv1D 압축) → MLP → FC tanh → msg_dim 메시지.

    ★완전분리 MoE(2026-06-26): MessageActor/ControlActor/Critic 각각이 USE_MOE=1이면 COLREGs 5상황별로
      이 *코어를 통째 5벌*(radar_encoder 포함) 보유 → 상황별 완전 독립망. USE_MOE=0이면 코어 1벌(단일망).
    flat 입력 [M, ...]으로 동작(상위 wrapper가 reshape·상황 라우팅 담당)."""
    def __init__(self, frames, msg_dim, width=1.0):
        super(_MessageActorCore, self).__init__()
        self.msg_dim = msg_dim
        # width<1.0 = iso-parameter MoE 코어 (내부 차원만 축소, 입출력 인터페이스 불변)
        f_dim = _w(RADAR_FEAT_DIM, width)
        self.hidden = _w(128, width, floor=8)
        # Radar feature extraction: 360 raw ray → 학습형 Conv1D 압축(원형 padding). min-pool 제거(2026-06-04).
        self.radar_encoder = RadarEncoder(frames, STATE_SIZE, f_dim, width)
        # f_dim + goal(2) + self_state(4) + situation one-hot(SIT_INPUT_DIM, 기본 0)
        self.fc2 = nn.Linear(f_dim + 2 + 4 + SIT_INPUT_DIM, self.hidden)
        # ★msg_out 앞 LayerNorm (2026-08-31 사용자 승인 — tanh 포화 방지):
        #   실측: 기존 구조에서 시드 절반(s43·s44 등)의 메시지가 mean|tanh|=1.000 으로 전 원소 포화 =
        #   gradient 0 = 학습 영구 정지. 원인은 fc2 은닉(post-ReLU, 비유계)이 msg_out 을 거치며 pre-tanh 가
        #   수십까지 자라는 것. LayerNorm 이 pre-tanh 입력 스케일을 O(1) 로 유지 → tanh 선형구간 유지.
        #   VESSEL_MSG_LN=0 으로 옛 구조(포화 위험) 재현 가능. 평가 스크립트는 ckpt 키에 'msg_ln' 유무를
        #   스니핑해 자동 설정(옛 체크포인트 strict 로드 호환).
        import os as _os_ln
        self.msg_ln = nn.LayerNorm(self.hidden) if _os_ln.environ.get('VESSEL_MSG_LN', '1') == '1' else None
        self.msg_out = nn.Linear(self.hidden, msg_dim)
        # ★생산측 소진폭 init (2026-06-12 채널동결 fix): weight+bias zero-init은 msg≡tanh(0)=0을 만들어
        #   소비측 zero-init과 직렬 곱 새들 형성 → 채널 전체 grad 항등 0, 1M step 비트동결(실측).
        #   소진폭(×0.1)이면 메시지가 step1부터 상태의존적이되 작음(tanh 선형구간) → 학습 신호 생존.
        with torch.no_grad():
            self.msg_out.weight.mul_(0.1)
            self.msg_out.bias.zero_()

    def forward(self, x_flat, goal_flat, self_state_flat, sit_oh=None):
        """x_flat [M, frames*STATE_SIZE], goal_flat [M,2], self_state_flat [M,4],
        sit_oh [M, SIT_INPUT_DIM] (SITUATION_INPUT=1일 때만) → msg [M, msg_dim]."""
        a = self.radar_encoder(x_flat)   # [M, RADAR_FEAT_DIM] (post-ReLU)
        if SITUATION_INPUT:
            a = torch.cat((a, goal_flat, self_state_flat, sit_oh), dim=-1)
        else:
            a = torch.cat((a, goal_flat, self_state_flat), dim=-1)
        a = F.relu(self.fc2(a))
        if self.msg_ln is not None:
            a = self.msg_ln(a)              # pre-tanh 스케일 O(1) 고정 → 포화 방지
        return torch.tanh(self.msg_out(a))  # bounded [-1, 1]


class MessageActor(nn.Module):
    """
    각 에이전트의 observation을 msg_dim 메시지로 압축.
    ★완전분리 MoE: USE_MOE=1이면 sender의 COLREGs 상황(0~4)으로 *코어를 통째 라우팅*(radar_encoder 포함
      5벌 독립). 메시지 = "그 상황의 전용 인코더가 만든 신호". USE_MOE=0이면 단일 코어.
    ⚠️ rollout(CNNPolicy.forward)·update(evaluate_actions) 모두 *동일 sender situation*으로 라우팅해야
      PPO ratio 유효 → update는 partner_situations(rollout서 저장)로 파트너 메시지를 재생성한다.
    """
    def __init__(self, frames, msg_dim):
        super(MessageActor, self).__init__()
        self.frames = frames
        self.msg_dim = msg_dim
        self.use_moe = USE_MOE
        self.num_experts = NUM_COLREGS_SITUATIONS
        if self.use_moe:
            # MOE_WIDTH<1.0 = iso-parameter MoE (5코어 합계 ≈ 단일망). 단일망은 항상 폭 1.0.
            self.experts = nn.ModuleList(
                [_MessageActorCore(frames, msg_dim, MOE_WIDTH) for _ in range(self.num_experts)])
            if MOE_SHARED:
                _share_radar_encoder(self.experts)
        else:
            self.core = _MessageActorCore(frames, msg_dim)

    def cores(self):
        """텔레메트리/옵티마이저 그룹용: 활성 코어 리스트(단일=[core], MoE=experts)."""
        return list(self.experts) if self.use_moe else [self.core]

    def mean_msg_out_norm(self):
        cs = self.cores()
        return float(sum(float(c.msg_out.weight.norm()) for c in cs) / len(cs))

    def forward(self, x, goal, self_state, situation=None):
        """
        Args:
            x: [batch, n_agent, frames * STATE_SIZE]
            goal: [batch, n_agent, 2]; self_state: [batch, n_agent, 4]
            situation: [batch, n_agent] or [batch, n_agent, 1] (sender COLREGs 0~4). USE_MOE=1 라우팅.
        Returns: msg [batch, n_agent, msg_dim]
        """
        batch_size, n_agent, _ = x.shape
        M = batch_size * n_agent
        x_f = x.reshape(M, -1)
        goal_f = goal.reshape(M, -1)
        self_f = self_state.reshape(M, -1)

        sit_oh = _situation_onehot(situation, M, x_f) if SITUATION_INPUT else None
        if not self.use_moe:
            msg = self.core(x_f, goal_f, self_f, sit_oh)
        else:
            # 상황별 hard-route: 각 sample은 자기 상황 코어만 통과 → 그 코어만 gradient.
            if situation is not None:
                sit = situation.reshape(M).long().clamp(0, self.num_experts - 1)
            else:
                sit = torch.zeros(M, dtype=torch.long, device=x.device)   # 상황 없으면 None(0) 코어
            if _moe_fast_on(self):
                # ★배치화 경로: 루프·동기화 0. 코어 forward 와 같은 순서
                #   (radar → concat → fc2 → relu → msg_ln → msg_out → tanh)
                cs = self.cores()
                a = cs[0].radar_encoder(x_f)                       # 공유 인코더 1회
                a = (torch.cat((a, goal_f, self_f, sit_oh), dim=-1) if SITUATION_INPUT
                     else torch.cat((a, goal_f, self_f), dim=-1))
                a = F.relu(_bmm_linear(cs, sit, a, 'fc2'))
                if cs[0].msg_ln is not None:
                    a = _bmm_layernorm(cs, sit, a, 'msg_ln')
                return torch.tanh(_bmm_linear(cs, sit, a, 'msg_out')).reshape(batch_size, n_agent, -1)
            msg = x_f.new_zeros(M, self.msg_dim)
            for k in range(self.num_experts):
                mask = (sit == k)
                if mask.any():
                    msg[mask] = self.experts[k](x_f[mask], goal_f[mask], self_f[mask],
                                                sit_oh[mask] if sit_oh is not None else None)
        return msg.view(batch_size, n_agent, self.msg_dim)


class _ControlActorCore(nn.Module):
    """단일 ControlActor 본체(radar_encoder + fc2 + msg_gate + consumer_decoder + fc3 + action head).

    ★완전분리 MoE(2026-06-26): radar_encoder를 *코어 안*에 둠 → USE_MOE=1이면 상황별 5벌이 perception까지
      완전 독립(과거의 공유 backbone 폐기). backbone(z)·head(mean,logstd) flat 입력으로 동작.
    """
    def __init__(self, frames, msg_dim, action_size, width=1.0):
        super(_ControlActorCore, self).__init__()
        self.msg_dim = msg_dim
        self.action_size = action_size
        # width<1.0 = iso-parameter MoE 코어 (내부 차원만 축소, 입출력 인터페이스 불변)
        f_dim = _w(RADAR_FEAT_DIM, width)
        self.hidden = _w(128, width, floor=8)
        _fc3_h = _w(64, width)
        _cons_h = _w(64, width)
        self.radar_encoder = RadarEncoder(frames, STATE_SIZE, f_dim, width)  # 360 raw ray → 학습형 Conv1D 압축
        # f_dim + goal(2) + self_state(4) + situation one-hot(SIT_INPUT_DIM, 기본 0) + others_msg(msg_dim, ★항상 마지막)
        self.fc2 = nn.Linear(f_dim + 2 + 4 + SIT_INPUT_DIM + msg_dim, self.hidden)
        # ★메시지 슬라이스 소진폭 init (2026-06-12 채널동결 fix): zero-init은 ∂L/∂msg≡0을 만들어 생산측
        #   zero와 직렬 곱 새들 형성 → 채널 영구 동결. 소진폭(×0.1)은 grad를 살리되 초기 메시지 영향을 작게.
        with torch.no_grad():
            self.fc2.weight[:, -msg_dim:].mul_(0.1)
        # ★메시지 게이트(learnable 다이얼, 초기 0=sigmoid 0.5 중립): others_msg * sigmoid(msg_gate).
        self.msg_gate = nn.Parameter(torch.tensor(0.0))
        # ★C5c 수신측 decode-in-policy: backbone z로 파트너 의도(goal) 복원 head (코어별 독립).
        self.consumer_k = max(1, min(COMM_CONSUMER_K, MAX_COMM_PARTNERS))
        self.consumer_decoder = nn.Sequential(
            nn.Linear(self.hidden, _cons_h), nn.ReLU(), nn.Linear(_cons_h, self.consumer_k * GOAL_SIZE)
        )
        # ★H2 coupling: decode(재구성)를 fc3 입력에 concat. OFF면 fc3 입력=hidden.
        self.consumer_coupling = COMM_CONSUMER_COUPLING
        _fc3_in = self.hidden + (self.consumer_k * GOAL_SIZE if self.consumer_coupling else 0)
        self.fc3 = nn.Linear(_fc3_in, _fc3_h)
        self.action_mean = nn.Linear(_fc3_h, action_size)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.zero_()
        # Learnable log std — per-dim: rudder(dim0) -1.0(std≈0.37), thrust(dim1) -0.5(std≈0.61).
        if action_size == 2:
            self.action_logstd = nn.Parameter(torch.tensor([[-1.0, -0.5]]))
        else:
            self.action_logstd = nn.Parameter(torch.full((1, action_size), -0.5))

    def backbone(self, x_f, goal_f, self_f, others_msg_f, sit_oh=None):
        """flat 입력 → z[M,128]. 게이트가 forward·update 양쪽에 동일 적용 → PPO ratio 정합.
        sit_oh [M, SIT_INPUT_DIM]: SITUATION_INPUT=1일 때만 concat(msg 앞 = msg 슬라이스 인덱싱 보존)."""
        gated = others_msg_f * torch.sigmoid(self.msg_gate)
        radar_feat = self.radar_encoder(x_f)   # [M, RADAR_FEAT_DIM] (post-ReLU)
        if SITUATION_INPUT:
            return torch.tanh(self.fc2(torch.cat((radar_feat, goal_f, self_f, sit_oh, gated), dim=-1)))
        return torch.tanh(self.fc2(torch.cat((radar_feat, goal_f, self_f, gated), dim=-1)))

    def head(self, z):
        """z[M,hidden] → mean[M,act], logstd[M,act], dec(coupling이면 복원값, 아니면 None).
        ★2026-09-05 fix: coupling일 때 계산한 consumer_decoder 출력을 *함께 반환*. 기존엔 같은 z로
          evaluate_actions가 consumer_decode를 다시 돌려 미니배치마다 두 번 forward했음(MoE면 5코어
          마스킹 루프까지 재실행). 같은 파라미터·같은 입력이라 값·gradient는 동일 → 결과 비트동일,
          비용만 절감. 또한 앞으로 decoder에 dropout 등 확률요소가 들어가도 '정책이 쓴 복원값'과
          '손실이 벌한 복원값'이 갈라지지 않음."""
        dec = self.consumer_decoder(z) if self.consumer_coupling else None
        zc = torch.cat([z, dec], dim=-1) if dec is not None else z
        a = torch.tanh(self.fc3(zc))
        mean = self.action_mean(a)
        logstd = self.action_logstd.expand(z.shape[0], -1)
        return mean, logstd, dec


class ControlActor(nn.Module):
    """
    자기 observation + 타 에이전트 메시지로 행동 결정 (CleanRL Normal + tanh squashing).
    ★완전분리 MoE(2026-06-26): USE_MOE=1이면 COLREGs 상황(0~4)으로 *코어를 통째 라우팅*(radar_encoder 포함
      5벌 독립). USE_MOE=0이면 단일 코어(=단일망 baseline). 각 sample은 자기 상황 코어만 통과 → 그 코어만 gradient.
      rollout(forward)·update(get_logprob_entropy)가 동일 situation으로 재라우팅 → old_logprob 정합(PPO ratio 유효).
    """
    def __init__(self, frames, msg_dim, action_size):
        super(ControlActor, self).__init__()
        self.frames = frames
        self.msg_dim = msg_dim
        self.action_size = action_size
        self.use_moe = USE_MOE
        self.num_experts = NUM_COLREGS_SITUATIONS      # 5 (None/HeadOn/StandOn/GiveWay/Overtaking)
        self.comm_consumer_coef = COMM_CONSUMER_COEF   # evaluate_actions C5c 게이트
        self.consumer_k = max(1, min(COMM_CONSUMER_K, MAX_COMM_PARTNERS))
        if self.use_moe:
            # MOE_WIDTH<1.0 = iso-parameter MoE (5코어 합계 ≈ 단일망). 단일망은 항상 폭 1.0.
            self.experts = nn.ModuleList(
                [_ControlActorCore(frames, msg_dim, action_size, MOE_WIDTH) for _ in range(self.num_experts)])
            if MOE_SHARED:
                _share_radar_encoder(self.experts)
        else:
            self.core = _ControlActorCore(frames, msg_dim, action_size)
        self.core_hidden = (self.experts[0] if self.use_moe else self.core).hidden
        # ★2026-09-05 fix: coupling에서 head가 이미 만든 복원값의 1회용 캐시 (z 객체, dec).
        #   _route가 채우고 pop_consumer_dec가 꺼내 비움 → 중복 forward 제거. 미스면 재계산(기존 경로).
        self._dec_cache = None

    def cores(self):
        return list(self.experts) if self.use_moe else [self.core]

    def mean_gate_sigmoid(self):
        cs = self.cores()
        return float(sum(float(torch.sigmoid(c.msg_gate)) for c in cs) / len(cs))

    def gate_open_sum(self):
        """게이트 개방 페널티용: 활성 코어들의 sigmoid(gate) 합(미분가능 → 모든 게이트로 grad)."""
        return sum(torch.sigmoid(c.msg_gate) for c in self.cores())

    def fc2_msg_slice_norm(self):
        md = self.msg_dim; cs = self.cores()
        return float(sum(float(c.fc2.weight[:, -md:].norm()) for c in cs) / len(cs))

    def _route(self, x, goal, self_state, others_msg, situation):
        """flat 라우팅: situation(0~4)별 코어로 backbone+head. → z[M,128], mean[M,act], logstd[M,act], B, N."""
        batch_size, n_agent, _ = x.shape
        M = batch_size * n_agent
        x_f = x.reshape(M, -1)
        goal_f = goal.reshape(M, -1)
        self_f = self_state.reshape(M, -1)
        om_f = others_msg.reshape(M, -1)
        sit_oh = _situation_onehot(situation, M, x_f) if SITUATION_INPUT else None
        if not self.use_moe:
            z = self.core.backbone(x_f, goal_f, self_f, om_f, sit_oh)
            mean, logstd, dec = self.core.head(z)
            self._cache_dec(z, dec)
            return z, mean, logstd, batch_size, n_agent
        # 상황별 hard-route (코어 통째). 각 sample은 자기 상황 코어만 통과 → 그 코어만 gradient.
        if situation is not None:
            sit = situation.reshape(M).long().clamp(0, self.num_experts - 1)
        else:
            sit = torch.zeros(M, dtype=torch.long, device=x.device)
        if _moe_fast_on(self):
            # ★배치화 경로: backbone(gate→radar→concat→fc2→tanh) + head(fc3→tanh→action_mean, logstd)
            cs = self.cores()
            gate = torch.sigmoid(torch.stack([c.msg_gate for c in cs]))[sit].unsqueeze(-1)
            radar_feat = cs[0].radar_encoder(x_f)                  # 공유 인코더 1회
            h = (torch.cat((radar_feat, goal_f, self_f, sit_oh, om_f * gate), dim=-1) if SITUATION_INPUT
                 else torch.cat((radar_feat, goal_f, self_f, om_f * gate), dim=-1))
            z = torch.tanh(_bmm_linear(cs, sit, h, 'fc2'))
            dec = None
            if cs[0].consumer_coupling:
                d0 = F.relu(_bmm_linear([c.consumer_decoder for c in cs], sit, z, '0'))
                dec = _bmm_linear([c.consumer_decoder for c in cs], sit, d0, '2')
                zc = torch.cat([z, dec], dim=-1)
            else:
                zc = z
            a = torch.tanh(_bmm_linear(cs, sit, zc, 'fc3'))
            mean = _bmm_linear(cs, sit, a, 'action_mean')
            logstd = torch.stack([c.action_logstd for c in cs]).squeeze(1)[sit].expand(M, self.action_size)
            self._cache_dec(z, dec)
            return z, mean, logstd, batch_size, n_agent
        z = x_f.new_zeros(M, self.core_hidden)
        mean = x_f.new_zeros(M, self.action_size)
        logstd = x_f.new_zeros(M, self.action_size)
        dec_full = None
        for k in range(self.num_experts):
            mask = (sit == k)
            if mask.any():
                zk = self.experts[k].backbone(x_f[mask], goal_f[mask], self_f[mask], om_f[mask],
                                              sit_oh[mask] if sit_oh is not None else None)
                mk, lk, dk = self.experts[k].head(zk)
                z[mask] = zk
                mean[mask] = mk
                logstd[mask] = lk
                if dk is not None:   # coupling: 코어별 복원값을 전체 배치로 재조립(모든 행이 정확히 한 코어 소속)
                    if dec_full is None:
                        dec_full = x_f.new_zeros(M, dk.shape[-1])
                    dec_full[mask] = dk
        self._cache_dec(z, dec_full)
        return z, mean, logstd, batch_size, n_agent

    def _cache_dec(self, z, dec):
        """★2026-09-05 fix: head가 계산한 복원값을 (z 객체, dec)로 캐시. dec가 None(coupling OFF)이면 미캐시."""
        self._dec_cache = (z, dec) if dec is not None else None

    def pop_consumer_dec(self, z, situation=None):
        """캐시된 복원값을 *1회용*으로 반환. 캐시가 없거나 z 객체가 다르면 consumer_decode 재계산(기존 경로).
        객체 동일성으로만 재사용 → 스테일 캐시 사용 불가(PPO 정합 안전)."""
        c = self._dec_cache
        self._dec_cache = None
        if c is not None and c[0] is z:
            return c[1]
        return self.consumer_decode(z, situation)

    def consumer_decode(self, z, situation=None):
        """C5c: z[M,128] → 파트너 의도 복원 [M, consumer_k*GOAL_SIZE]. 라우팅된 코어의 consumer_decoder 사용."""
        M = z.shape[0]
        out_dim = self.consumer_k * GOAL_SIZE
        if not self.use_moe:
            return self.core.consumer_decoder(z)
        if situation is not None:
            sit = situation.reshape(-1).long().clamp(0, self.num_experts - 1)
        else:
            sit = torch.zeros(M, dtype=torch.long, device=z.device)
        out = z.new_zeros(M, out_dim)
        for k in range(self.num_experts):
            mask = (sit == k)
            if mask.any():
                out[mask] = self.experts[k].consumer_decoder(z[mask])
        return out

    def forward(self, x, goal, self_state, others_msg, situation=None):
        """Returns: action [b,n,act], logprob [b,n,1], mean [b,n,act], action_raw [b,n,act].
        ★action_raw(=pre-tanh 샘플)를 함께 반환 → rollout이 memory에 raw를 저장하고 update가 그대로
          재사용(get_logprob_entropy) → tanh 보정이 비트 일치, PPO ratio가 epoch0에서 1.0(포화 샘플 포함).
        situation: [b,n] or [b,n,1] (0~4)."""
        z, action_mean, action_logstd, batch_size, n_agent = self._route(
            x, goal, self_state, others_msg, situation)

        action_mean = torch.clamp(action_mean, -3.0, 3.0)            # tanh(3) ≈ 0.995
        action_logstd = torch.clamp(action_logstd, -2.3, 0.0)       # std 0.1 ~ 1.0
        action_std = torch.exp(action_logstd)

        dist = Normal(action_mean, action_std)
        action_raw = dist.sample()

        # Squashed Gaussian: tanh로 [-1, 1] + log_prob 보정
        action = torch.tanh(action_raw)
        logprob = dist.log_prob(action_raw) - torch.log(1 - action.pow(2) + 1e-6)
        logprob = logprob.sum(dim=-1, keepdim=True)

        action = action.view(batch_size, n_agent, -1)
        logprob = logprob.view(batch_size, n_agent, -1)
        action_mean = action_mean.view(batch_size, n_agent, -1)
        action_raw = action_raw.view(batch_size, n_agent, -1)
        self._dec_cache = None   # ★2026-09-05 fix: rollout 경로는 캐시 소비자가 없음 → 그래프 참조 즉시 해제
        return action, logprob, action_mean, action_raw

    def get_logprob_entropy(self, x, goal, self_state, others_msg, action_raw, situation=None):
        """PPO 업데이트용: rollout이 저장한 pre-tanh raw로 log_prob·entropy 재계산.
        ★situation은 rollout 저장값 → forward와 *동일* 코어 라우팅 → old_logprob 정합(PPO ratio 유효).
        ★action_raw는 rollout forward가 sample한 pre-tanh 값(memory 저장). 기존엔 tanh 저장값을
          atanh(clamp(·,±0.999))로 역변환했으나 |raw|>3.8 포화 샘플에서 old≠new logprob → ratio가
          policy 변화와 무관하게 epoch0부터 clip 밴드를 벗어났음(전타/전속 commit 기동에서 gradient 오염).
          raw를 직접 받아 action=tanh(raw)를 재계산 → 보정항이 forward와 정확히 동일."""
        z, action_mean, action_logstd, batch_size, n_agent = self._route(
            x, goal, self_state, others_msg, situation)
        action_raw_flat = action_raw.reshape(batch_size * n_agent, -1)

        action_mean = torch.clamp(action_mean, -3.0, 3.0)
        action_logstd = torch.clamp(action_logstd, -2.3, 0.0)
        action_std = torch.exp(action_logstd)

        # ★저장된 raw로 tanh를 재계산 → forward의 squash 보정과 비트 일치(무손실).
        action = torch.tanh(action_raw_flat)

        dist = Normal(action_mean, action_std)
        logprob = dist.log_prob(action_raw_flat) - torch.log(1 - action.pow(2) + 1e-6)
        logprob = logprob.sum(dim=-1, keepdim=True)

        # Squashed Gaussian entropy 보정 (동일 tanh 값 사용)
        gaussian_entropy = dist.entropy().sum(dim=-1)
        squash_correction = torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        entropy = (gaussian_entropy + squash_correction).mean()

        logprob = logprob.view(batch_size, n_agent, -1)
        action_mean = action_mean.view(batch_size, n_agent, -1)
        # ★z [B*N,128] 반환(C5c): evaluate_actions가 consumer_decode(z, situation)로 파트너 의도 복원 손실 계산.
        return logprob, entropy, action_mean, z


class _CriticCore(nn.Module):
    """단일 Critic 본체(radar_encoder + fc2 + msg_gate + value_out). 완전분리 MoE에서 상황별 5벌 독립.
    ★과거의 situation one-hot 조건화 폐기 — 상황별로 코어를 통째 라우팅하므로 가치망도 perception까지 독립."""
    def __init__(self, frames, msg_dim, width=1.0):
        super(_CriticCore, self).__init__()
        # width<1.0 = iso-parameter MoE 코어 (내부 차원만 축소, 입출력 인터페이스 불변)
        f_dim = _w(RADAR_FEAT_DIM, width)
        self.hidden = _w(128, width, floor=8)
        self.radar_encoder = RadarEncoder(frames, STATE_SIZE, f_dim, width)  # 360 raw ray → 학습형 Conv1D 압축
        # f_dim + goal(2) + self_state(4) + situation one-hot(SIT_INPUT_DIM, 기본 0) + others_msg(msg_dim, ★항상 마지막)
        # ★중앙 critic(CTDE, 2026-09-04): 전 선박 (상대위치·침로·속도·거리) 6D → per-ship 인코딩 → mean pool 64D.
        #   CENTRAL_CRITIC=0이면 브랜치 미생성(구 ckpt strict 로드 호환·비트동일).
        self.glob_enc = (nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64))
                         if CENTRAL_CRITIC else None)
        _glob_dim = 64 if CENTRAL_CRITIC else 0
        self.fc2 = nn.Linear(f_dim + 2 + 4 + SIT_INPUT_DIM + _glob_dim + msg_dim, self.hidden)
        with torch.no_grad():
            self.fc2.weight[:, -msg_dim:].mul_(0.1)   # 메시지 슬라이스 소진폭 init (채널동결 fix)
        self.msg_gate = nn.Parameter(torch.tensor(0.0))
        self.value_out = nn.Linear(self.hidden, 1)

    def forward(self, x_f, goal_f, self_f, others_msg_f, sit_oh=None, glob_f=None):
        gated = others_msg_f * torch.sigmoid(self.msg_gate)
        v = self.radar_encoder(x_f)   # [M, RADAR_FEAT_DIM] (post-ReLU)
        parts = [v, goal_f, self_f]
        if SITUATION_INPUT:
            parts.append(sit_oh)
        if self.glob_enc is not None:
            # glob_f [M, N_ships, 6] → per-ship 인코딩 후 mean pool (순열 불변)
            # ★glob_f=None 방어: 평가·미러 스크립트가 전역 상태 없이 부를 수 있음 → 0 대체(결정론적)
            if glob_f is None:
                glob_f = x_f.new_zeros(x_f.shape[0], 1, 6)
            parts.append(self.glob_enc(glob_f).mean(dim=1))
        parts.append(gated)                      # 메시지 슬라이스는 항상 마지막(소진폭 init 계약)
        v = F.relu(self.fc2(torch.cat(parts, dim=-1)))
        return self.value_out(v)   # [M, 1]


class Critic(nn.Module):
    """Value function estimator (others_msg 포함).
    ★완전분리 MoE(2026-06-26): USE_MOE=1이면 COLREGs 상황(0~4)으로 *코어 통째 라우팅*(radar_encoder 포함 5벌).
      rollout(forward, CNNPolicy.forward)·update(evaluate_actions) 동일 situation → value 추정 일관."""
    def __init__(self, frames, msg_dim):
        super(Critic, self).__init__()
        self.frames = frames
        self.msg_dim = msg_dim
        self.use_moe = USE_MOE
        self.num_experts = NUM_COLREGS_SITUATIONS
        if self.use_moe:
            # MOE_WIDTH<1.0 = iso-parameter MoE (5코어 합계 ≈ 단일망). 단일망은 항상 폭 1.0.
            self.experts = nn.ModuleList(
                [_CriticCore(frames, msg_dim, MOE_WIDTH) for _ in range(self.num_experts)])
            if MOE_SHARED:
                _share_radar_encoder(self.experts)
        else:
            self.core = _CriticCore(frames, msg_dim)

    def cores(self):
        return list(self.experts) if self.use_moe else [self.core]

    def mean_gate_sigmoid(self):
        cs = self.cores()
        return float(sum(float(torch.sigmoid(c.msg_gate)) for c in cs) / len(cs))

    def gate_open_sum(self):
        return sum(torch.sigmoid(c.msg_gate) for c in self.cores())

    def fc2_msg_slice_norm(self):
        md = self.msg_dim; cs = self.cores()
        return float(sum(float(c.fc2.weight[:, -md:].norm()) for c in cs) / len(cs))

    def fc2_rest_norm(self):
        md = self.msg_dim; cs = self.cores()
        return float(sum(float(c.fc2.weight[:, :-md].norm()) for c in cs) / len(cs))

    def forward(self, x, goal, self_state, others_msg, situation=None, global_feat=None):
        batch_size, n_agent, _ = x.shape
        M = batch_size * n_agent
        x_f = x.reshape(M, -1)
        goal_f = goal.reshape(M, -1)
        self_f = self_state.reshape(M, -1)
        om_f = others_msg.reshape(M, -1)
        # ★중앙 critic: global_feat [..., N_ships, 6] → [M, N_ships, 6]
        gf_f = global_feat.reshape(M, global_feat.shape[-2], 6) if global_feat is not None else None
        sit_oh = _situation_onehot(situation, M, x_f) if SITUATION_INPUT else None
        if not self.use_moe:
            v = self.core(x_f, goal_f, self_f, om_f, sit_oh, gf_f)
        else:
            if situation is not None:
                sit = situation.reshape(M).long().clamp(0, self.num_experts - 1)
            else:
                sit = torch.zeros(M, dtype=torch.long, device=x.device)
            if _moe_fast_on(self):
                # ★배치화 경로: 코어 forward 와 같은 순서
                #   (gate → radar → [goal,self,sit,(glob),msg] concat → fc2 → relu → value_out)
                cs = self.cores()
                gate = torch.sigmoid(torch.stack([c.msg_gate for c in cs]))[sit].unsqueeze(-1)
                parts = [cs[0].radar_encoder(x_f), goal_f, self_f]      # 공유 인코더 1회
                if SITUATION_INPUT:
                    parts.append(sit_oh)
                if cs[0].glob_enc is not None:
                    _gf = gf_f if gf_f is not None else x_f.new_zeros(M, 1, 6)   # 코어와 같은 방어
                    parts.append(_bmm_seq2(cs, sit, _gf, 'glob_enc').mean(dim=1))
                parts.append(om_f * gate)                                # 메시지 슬라이스는 항상 마지막
                h = F.relu(_bmm_linear(cs, sit, torch.cat(parts, dim=-1), 'fc2'))
                return _bmm_linear(cs, sit, h, 'value_out').reshape(batch_size, n_agent, 1)
            v = x_f.new_zeros(M, 1)
            for k in range(self.num_experts):
                mask = (sit == k)
                if mask.any():
                    v[mask] = self.experts[k](x_f[mask], goal_f[mask], self_f[mask], om_f[mask],
                                              sit_oh[mask] if sit_oh is not None else None,
                                              gf_f[mask] if gf_f is not None else None)
        return v.view(batch_size, n_agent, 1)


class CNNPolicy(nn.Module):
    """
    전체 정책 네트워크 (메시지 교환 기반)

    흐름:
    1. 모든 에이전트의 obs → MessageActor → 각자의 msg_dim 메시지
    2. 통신 파트너(범위 내 nearest-K) 메시지 집계(sum/mean/scale/attention, VESSEL_AGG_MODE·VESSEL_USE_ATTENTION·VESSEL_POS_GROUND) = others_msg
    3. 자기 obs + others_msg → ControlActor → 행동
    4. Critic → 가치 추정
    """
    def __init__(self, msg_dim, action_size, frames):
        super(CNNPolicy, self).__init__()
        # ★C5c/oracle는 others_msg[..., :GOAL_SIZE]에 파트너 goal을 주입/복원 → msg_dim≥GOAL_SIZE 필요.
        #   MSG_DIM=6,GOAL_SIZE=2라 현재 성립. H2에서 MSG_DIM<GOAL_SIZE로 줄이면 명시 실패(silent corruption 방지).
        assert msg_dim >= GOAL_SIZE, f"msg_dim({msg_dim}) < GOAL_SIZE({GOAL_SIZE}): C5c/oracle goal 주입 불가"
        self.frames = frames
        self.msg_dim = msg_dim
        self.action_size = action_size

        self.msg_actor = MessageActor(frames, msg_dim)
        self.ctr_actor = ControlActor(frames, msg_dim, action_size)
        self.critic = Critic(frames, msg_dim)

        # ★ 위치 grounding (AIS-style, 2026-07-03 기본 ON): 파트너의 [상대방위(sin,cos)+거리] 3D를 메시지에 결합 →
        #   receiver가 "어느 방위에서 온 메시지"인지 알게 됨. VESSEL_POS_GROUND=0으로 sum 대조군.
        #   relpos는 "주소"(어디서), 학습 6D latent는 "내용"(의도) → 학습메시지 thesis 유지.
        self.pos_ground = POS_GROUND
        self.relpos_dim = 3
        self.msg_encoder = nn.Sequential(
            nn.Linear(self.relpos_dim + msg_dim, 32), nn.ReLU(), nn.Linear(32, msg_dim)
        )

        # ★ 위치 grounding + attention 집계 (sum의 상위호환; VESSEL_USE_ATTENTION=1일 때만 사용).
        #   query=receiver[self,goal], key/value=[relpos⊕msg] → softmax 가중선택.
        #   출력차원 msg_dim → 게이트/fc2 불변. v_proj zero-init → context=0 at init(H1a).
        self.use_attention = USE_ATTENTION
        _query_in = SELF_STATE_SIZE + GOAL_SIZE   # 4+2 = 6
        self.attn = GroundedAttention(msg_dim, self.relpos_dim, _query_in, ATTN_DIM)

        # ★ intent self-supervised 디코더 (Phase 2): 메시지가 sender 미래의도를 담게 강제.
        #   INTENT_COEF=0(default)이면 evaluate_actions에서 미호출 → 기존과 비트동일.
        self.intent_coef = INTENT_COEF
        self.intent_k = INTENT_K
        self.intent_decoder = IntentDecoder(msg_dim, INTENT_K)

        # ★ threat-relay self-supervised 디코더 (L1, Phase 2): 메시지가 sender가 본 top-K 위협 기하를
        #   담게 강제 → occlusion으로 receiver가 못 보는 제3 위협을 메시지로 복원(통신 필수화).
        #   THREAT_COEF=0(default)이면 evaluate_actions에서 미호출 → 기존과 비트동일.
        self.threat_coef = THREAT_COEF
        self.threat_k = THREAT_K
        self.threat_decoder = ThreatDecoder(msg_dim, THREAT_K)

        # ★ L4 goal-broadcast 디코더: 메시지가 sender 목적지(goal)를 담게 강제. 통신거리>레이더라
        #   먼 배의 목적지를 통신으로만 알 수 있음 → 미리 협응. GOAL_COMM_COEF=0이면 미호출=비트동일.
        self.goal_comm_coef = GOAL_COMM_COEF
        self.goal_decoder = GoalDecoder(msg_dim, GOAL_SIZE)

        # ★ Role-broadcast 디코더 (C, 사용자 비전): 메시지가 sender COLREGs 상황/역할(0~4)을 담게 강제.
        #   receiver가 sender 역할(give-way/stand-on)을 알면 미리 협응 → compliance↑. ROLE_COMM_COEF=0이면 미호출=비트동일.
        #   라벨=sender 자기 situation(evaluate_actions 인자, self-prediction). 정책/가치 무오염(별도 head).
        self.role_comm_coef = ROLE_COMM_COEF
        self.role_decoder = RoleDecoder(msg_dim, NUM_COLREGS_SITUATIONS)

        # ★통합 상태복원 (2026-09-04): STATE_RECON_COEF>0일 때만 생성(구 ckpt strict 로드 호환).
        self.state_recon_coef = STATE_RECON_COEF
        self.state_recon = (StateReconDecoder(msg_dim, THREAT_K, INTENT_K, NUM_COLREGS_SITUATIONS)
                            if STATE_RECON_COEF > 0.0 else None)
        self._last_state_recon = None   # evaluate_actions가 (loss, 그룹dict) 저장 — 반환 시그니처 불변 유지
        self._central_warned = False    # 중앙 critic 무력화 경고 1회용 플래그(아래 forward)

    def _get_others_msg(self, msg, comm_partners=None, agent_id_list=None, comm_relpos=None,
                        self_state=None, goal=None):
        """메시지 교환 로직 (rollout, annealing 없음 - 즉시 100%)

        env override:
          VESSEL_USE_ATTENTION: 1이면 위치 grounding+attention 집계 (최우선)
          VESSEL_POS_GROUND: 1이면 위치 grounding+mean 집계
          VESSEL_AGG_MODE: 'sum' | 'mean' | 'scale' (default 'sum')
          VESSEL_NEAREST_SCALE: float, default 0 (>0이면 'scale' 자동 활성)
          VESSEL_MSG_GAIN: float, default 1.0 (최종 결과 gating 계수)
        ⚠️ evaluate_actions의 update-time 집계와 동일해야 PPO ratio 유효 (attention/pos_ground/sum 각각 미러).
        self_state/goal은 attention query용(receiver 상황). rollout forward가 전달.
        """
        import os as _os
        agg_mode = _os.environ.get('VESSEL_AGG_MODE', 'sum').lower()
        nearest_scale = float(_os.environ.get('VESSEL_NEAREST_SCALE', 0))
        msg_gain = float(_os.environ.get('VESSEL_MSG_GAIN', 1.0))
        if nearest_scale > 0:
            agg_mode = 'scale'

        batch_size, n_agent, _ = msg.shape

        # ★oracle-OFF 통제(rollout): 학습채널 대신 *참 파트너 goal* 주입(채널 우회). update(evaluate_actions)와
        #   동일 함수형(파트너 goal 평균을 others_msg 앞 GOAL_SIZE 차원에 주입, 나머지 0) → PPO ratio 유효.
        #   ★USE_COMMUNICATION 가드: oracle은 comm-path 변종 → COMM=0(OFF baseline)이면 미발화(others_msg≡0 보존).
        if USE_ORACLE and USE_COMMUNICATION:
            # ★파트너 정보 누락 시 zeros 반환 — 아래 mean-field 폴백으로 새는 것 차단. update(evaluate_actions)는
            #   무조건 oracle 주입이므로 여기서 폴백에 떨어지면 rollout≠update = PPO ratio 파손.
            if comm_partners is None or agent_id_list is None or goal is None:
                return torch.zeros_like(msg)
            id_to_idx = {aid: idx for idx, aid in enumerate(agent_id_list)}
            others_msg = torch.zeros_like(msg)
            for i, agent_id in enumerate(agent_id_list):
                partners = comm_partners.get(agent_id, [])
                # [:MAX_COMM_PARTNERS] 절단: update의 K-slot masked-mean과 도메인을 구조적으로 일치
                #   (obs_utils가 원천 절단하므로 현행 무영향 — 방어적 불변식).
                pidx = [id_to_idx[p] for p in partners if p in id_to_idx][:MAX_COMM_PARTNERS]
                if not pidx:
                    continue
                others_msg[0, i, :GOAL_SIZE] = goal[0, pidx, :].mean(dim=0)   # 참 파트너 goal 평균
            return others_msg

        if not USE_COMMUNICATION:
            return torch.zeros_like(msg)

        if comm_partners is not None and agent_id_list is not None:
            id_to_idx = {aid: idx for idx, aid in enumerate(agent_id_list)}

            # ★ attention 벡터화 경로 (per-agent 파이썬 루프 제거 → GPU 커널 런치 급감, 6-way 병렬 회복).
            #   update의 evaluate_actions와 *동일한 aggregate_batch* 사용 → PPO mirror 구조적 보장.
            #   파트너 인덱스 행렬을 CPU에서 1회 구성(가벼움) → gather + aggregate_batch 1회(GPU 벡터연산).
            #   per-agent aggregate_single 루프와 수치 동일(softmax가 padding을 -inf 마스킹).
            if self.use_attention and comm_relpos is not None and self_state is not None:
                Kmax = MAX_COMM_PARTNERS
                idx_mat = np.zeros((n_agent, Kmax), dtype=np.int64)
                mask_mat = np.zeros((n_agent, Kmax), dtype=np.float32)
                relpos_mat = np.zeros((n_agent, Kmax, self.relpos_dim), dtype=np.float32)
                for i, agent_id in enumerate(agent_id_list):
                    partners = comm_partners.get(agent_id, [])
                    if not partners or agent_id not in comm_relpos:
                        continue
                    rp = comm_relpos[agent_id]   # [K_actual, relpos_dim]
                    kept_pos = [k for k, p in enumerate(partners) if p in id_to_idx][:Kmax]
                    for j, k in enumerate(kept_pos):
                        idx_mat[i, j] = id_to_idx[partners[k]]
                        mask_mat[i, j] = 1.0
                        if k < len(rp):
                            relpos_mat[i, j] = rp[k]
                idx_t = torch.as_tensor(idx_mat, device=msg.device)                              # [N,Kmax]
                mask_t = torch.as_tensor(mask_mat, device=msg.device).unsqueeze(-1)              # [N,Kmax,1]
                relpos_t = torch.as_tensor(relpos_mat, dtype=torch.float32, device=msg.device)  # [N,Kmax,R]
                msg_part = msg[0][idx_t] * mask_t                                                # [N,Kmax,M] (padding=0)
                q_in = torch.cat([self_state[0], goal[0]], dim=-1).unsqueeze(1)                  # [N,1,Q]
                others = self.attn.aggregate_batch(q_in, relpos_t, msg_part, mask_t)            # [N,1,M]
                others_msg = others.transpose(0, 1).contiguous()                                # [1,N,M]
                if msg_gain != 1.0:
                    others_msg = others_msg * msg_gain
                return others_msg

            # ★ pos_ground 벡터화 경로 (2026-07-03 기본 ON 승격과 함께 추가): attention 벡터화와 동일한
            #   행렬 구성 → msg_encoder 1회 배치 호출 + masked mean. per-agent 루프의 커널 런치 병목 방지
            #   (attention이 겪은 6-way 병렬 처리량 붕괴의 재발 방지). update의 masked-mean(★아래
            #   evaluate_actions 분기)과 동일 함수형: 인코더 출력에 mask 적용 후 실파트너 수 Kc로 나눔
            #   → 패딩 슬롯 기여 0, per-agent mean(dim=0)과 수치 동일 → PPO ratio 유효.
            if self.pos_ground and comm_relpos is not None:
                Kmax = MAX_COMM_PARTNERS
                # 순수 파이썬 리스트로 행렬 구성 (numpy 미경유): torch↔numpy 버전 비호환에 무관하게 동작.
                # N·Kmax ≤ 수십 원소라 비용 무시 가능.
                idx_rows, mask_rows, rel_rows = [], [], []
                for i, agent_id in enumerate(agent_id_list):
                    idx_r = [0] * Kmax
                    mask_r = [0.0] * Kmax
                    rel_r = [[0.0] * self.relpos_dim for _ in range(Kmax)]
                    partners = comm_partners.get(agent_id, [])
                    if partners and agent_id in comm_relpos:
                        rp = comm_relpos[agent_id]   # [K_actual, relpos_dim]
                        kept_pos = [k for k, p in enumerate(partners) if p in id_to_idx][:Kmax]
                        for j, k in enumerate(kept_pos):
                            idx_r[j] = id_to_idx[partners[k]]
                            mask_r[j] = 1.0
                            if k < len(rp):
                                rel_r[j] = [float(v) for v in rp[k]]
                    idx_rows.append(idx_r)
                    mask_rows.append(mask_r)
                    rel_rows.append(rel_r)
                idx_t = torch.tensor(idx_rows, dtype=torch.long, device=msg.device)              # [N,Kmax]
                mask_t = torch.tensor(mask_rows, dtype=torch.float32, device=msg.device).unsqueeze(-1)  # [N,Kmax,1]
                relpos_t = torch.tensor(rel_rows, dtype=torch.float32, device=msg.device)        # [N,Kmax,R]
                msg_part = msg[0][idx_t] * mask_t                                                # [N,Kmax,M]
                localized = self.msg_encoder(torch.cat([relpos_t, msg_part], dim=-1))            # [N,Kmax,M]
                Kc = mask_t.sum(dim=1, keepdim=True).clamp(min=1.0)                              # [N,1,1]
                others = (localized * mask_t).sum(dim=1, keepdim=True) / Kc                      # [N,1,M]
                others_msg = others.transpose(0, 1).contiguous()                                # [1,N,M]
                if msg_gain != 1.0:
                    others_msg = others_msg * msg_gain
                return others_msg

            others_msg = torch.zeros_like(msg)
            for i, agent_id in enumerate(agent_id_list):
                partners = comm_partners.get(agent_id, [])
                if not partners:
                    continue
                # ★ kept_pos: partners 중 id_to_idx 생존 *위치* — partner_indices와 relpos를 같은
                #   위치로 추출해 정렬을 *구성적으로* 보장(중간 파트너가 필터돼도 relpos[j]↔msg[j] 불변).
                #   현재 partners ⊆ agent_id_list라 필터는 no-op(kept_pos=전체) → 기존 동작과 비트 동일.
                kept_pos = [k for k, p in enumerate(partners) if p in id_to_idx]
                partner_indices = [id_to_idx[partners[k]] for k in kept_pos]
                K = len(partner_indices)
                if K == 0:
                    continue
                if (self.use_attention and comm_relpos is not None and agent_id in comm_relpos
                        and self_state is not None):
                    # ★ 위치 grounding + attention: query=receiver[self,goal], kv=[relpos⊕msg]
                    rp = torch.as_tensor(comm_relpos[agent_id][kept_pos], dtype=torch.float32, device=msg.device)  # [K,3]
                    msg_p = msg[0, partner_indices, :]                                                       # [K,6]
                    q_in = torch.cat([self_state[0, i], goal[0, i]], dim=-1)                                 # [6]
                    s = self.attn.aggregate_single(q_in, rp, msg_p)                                          # [6]
                # ⚠️도달 불가(죽은 분기): 동일 조건은 위 벡터화 pos_ground 분기가 항상 먼저 return.
                #   내부의 numpy fancy-indexing→torch.as_tensor 경로는 torch/numpy 버전 비호환 환경에서
                #   실행 불가하므로, 벡터화 분기 조건을 바꿀 경우 이 폴백을 살리지 말고 벡터화 쪽을 수정할 것.
                elif self.pos_ground and comm_relpos is not None and agent_id in comm_relpos:
                    # 위치 grounding: [상대방위·거리 + 메시지] → encoder → mean
                    rp = torch.as_tensor(comm_relpos[agent_id][kept_pos], dtype=torch.float32, device=msg.device)  # [K,3]
                    msg_p = msg[0, partner_indices, :]                                                       # [K,6]
                    s = self.msg_encoder(torch.cat([rp, msg_p], dim=-1)).mean(dim=0)                          # [6]
                else:
                    s = msg[0, partner_indices, :].sum(dim=0)
                    if agg_mode == 'mean':
                        s = s / K
                    elif agg_mode == 'scale':
                        s = s * (nearest_scale / K)
                if msg_gain != 1.0:
                    s = s * msg_gain
                others_msg[0, i, :] = s
            return others_msg

        # Mean-field fallback (파트너 정보 없을 때)
        msg_sum = msg.sum(dim=1, keepdim=True).repeat(1, n_agent, 1)
        others_msg = msg_sum - msg
        if msg_gain != 1.0:
            others_msg = others_msg * msg_gain
        return others_msg

    def _central_critic_active(self):
        """중앙 critic 브랜치(glob_enc)가 *실제로* 만들어졌는지 — 모듈 상수 대신 인스턴스로 판정.
        (eval_ckpt가 networks.CENTRAL_CRITIC를 ckpt 스니핑 값으로 덮어쓴 뒤 생성하므로 상수 독해는 부정확)."""
        try:
            return getattr(self.critic.cores()[0], 'glob_enc', None) is not None
        except Exception:
            return False

    def forward(self, x, goal, self_state,
                return_msg=False, comm_partners=None, agent_id_list=None, comm_relpos=None,
                situation=None, return_raw=False, global_feat=None):
        """
        Rollout forward. Returns value, action, logprob, mean [, msg, others_msg] [, action_raw]
        ★return_raw=True면 pre-tanh action_raw를 튜플 끝에 추가 반환(학습 rollout이 memory에 저장 →
          update가 그대로 재사용, PPO ratio 정합). 기본 False → 기존 caller(arity) 불변(anti-regression).
        situation: [b,n] COLREGs 상황(0~4) — USE_MOE=1이면 head/critic 라우팅. None/OFF면 무시(단일 head).
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            goal = goal.unsqueeze(1)
            self_state = self_state.unsqueeze(1)

        # 1. 메시지 생성
        msg = self.msg_actor(x, goal, self_state, situation)
        # 2. 메시지 교환 (grounding 시 comm_relpos, attention 시 self_state/goal을 query로)
        others_msg = self._get_others_msg(msg, comm_partners, agent_id_list, comm_relpos,
                                          self_state=self_state, goal=goal)
        # 3. 행동 (situation으로 상황별 head 라우팅)
        action, logprob, mean, action_raw = self.ctr_actor(x, goal, self_state, others_msg, situation)
        # 4. 가치 (situation 라우팅 + 중앙 critic 전역상태)
        # ★2026-09-05 fix: global_feat를 critic까지 전달. 기존엔 CNNPolicy.forward가 인자를 아예 안 받아
        #   CENTRAL_CRITIC=1로 Unity(main.py) 경로를 돌리면 rollout·update 둘 다 glob_enc가 zeros만 받아
        #   64D 상수 bias로 퇴화 → '중앙 critic 썼다'는 기록과 달리 critic이 전역상태를 한 번도 못 봤음(조용한 무력화).
        #   default None = 기존 호출부 전부 비트동일. 실제 CTDE는 호출부(main.py)가 전역상태를 넘겨야 성립
        #   → 넘기지 않은 채 CENTRAL_CRITIC=1이면 stderr로 1회 경고(조용한 무력화 차단).
        if global_feat is None and not self._central_warned and self._central_critic_active():
            self._central_warned = True
            import sys as _sys_cc
            print('[networks] WARNING: CENTRAL_CRITIC=1 인데 CNNPolicy.forward가 global_feat 없이 호출됨 '
                  '→ critic 전역입력이 zeros(중앙 critic 무력화). 호출부에서 global_feat 전달 필요.',
                  file=_sys_cc.stderr)
        value = self.critic(x, goal, self_state, others_msg, situation, global_feat=global_feat)

        if return_msg and return_raw:
            return value, action, logprob, mean, msg, others_msg, action_raw
        if return_msg:
            return value, action, logprob, mean, msg, others_msg
        if return_raw:
            return value, action, logprob, mean, action_raw
        return value, action, logprob, mean

    def evaluate_actions(self, x, goal, self_state,
                         partner_x, partner_goal, partner_self, partner_mask,
                         partner_relpos, action_raw, own_future=None, own_future_mask=None,
                         own_threat=None, own_threat_mask=None, situation=None,
                         partner_situations=None, global_feat=None):
        """
        PPO 업데이트용. ★통신 sender→receiver gradient 수정★
        통신 ON이면 파트너 obs로 MessageActor를 재실행(미분가능)하여 others_msg를 재구성.
        집계(sum/mean/scale/attention/pos_ground)는 rollout _get_others_msg와 동일 함수형으로
        미러링 → PPO ratio(old_logprob) 유효. (아래 분기는 _get_others_msg와 1:1 대응)
        MessageActor는 공유 가중치 → 파트너 메시지의 gradient가 sender 학습으로 흐름.

        ★Phase2 intent: own_future(=내 미래 K-step 변위/heading, self-supervised 라벨)가 주어지고
          INTENT_COEF>0이면, 내 obs로 재생성한 own msg를 IntentDecoder로 통과시켜 미래를 예측,
          MSE 손실을 반환(MessageActor로만 gradient → "메시지가 미래의도 인코딩" 강제). 정책/가치 무오염.

        ★L1 threat: own_threat(=내 ego-radar가 본 top-K 위협 기하, self-supervised 라벨)가 주어지고
          THREAT_COEF>0이면, 내 obs로 재생성한 own msg를 ThreatDecoder로 통과시켜 위협 기하를 예측,
          MSE 손실을 반환(MessageActor로만 gradient → "메시지가 제3 위협 인코딩" 강제). 정책/가치 무오염.

        x: [N,1,F*S], partner_x: [N,K,F*S], partner_mask: [N,K,1], own_future: [N,1,K*3], own_threat: [N,1,K*4]
        ★Role(C): situation>0이고 ROLE_COMM_COEF>0이면, 내 obs로 재생성한 own msg를 RoleDecoder로 통과시켜
          내 COLREGs 상황(0~4)을 분류, cross-entropy 손실을 반환(MessageActor로만 gradient → "메시지가
          sender 역할 인코딩" 강제 → receiver가 attention으로 역할 알아 협응). 정책/가치 무오염.

        Returns: value, logprob, entropy, msg_reg, intent_loss, threat_loss, goal_loss, role_loss, consumer_loss
        ★consumer_loss(C5c): 수신측 backbone z로 파트너 의도(goal) 복원 MSE → fc2 메시지슬라이스에 gradient.
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            goal = goal.unsqueeze(1)
            self_state = self_state.unsqueeze(1)
            action_raw = action_raw.unsqueeze(1)

        if USE_ORACLE and USE_COMMUNICATION:
            # ★oracle-OFF 통제(update): 학습채널 대신 *참 파트너 goal* 주입(채널 우회). rollout _get_others_msg의
            #   oracle 분기와 동일 함수형(파트너 goal masked-mean을 앞 GOAL_SIZE에 주입, 나머지 0) → PPO ratio 유효.
            #   ★USE_COMMUNICATION 가드: COMM=0이면 else(zeros)로 → comm-OFF 불변(oracle 누수 방지).
            _Kc = partner_mask.sum(dim=1, keepdim=True).clamp(min=1.0)             # [N,1,1]
            _tgt = (partner_goal * partner_mask).sum(dim=1, keepdim=True) / _Kc    # [N,1,GOAL_SIZE]
            _has = (partner_mask.sum(dim=1, keepdim=True) > 0).float()             # [N,1,1]
            others_msg = torch.zeros(x.shape[0], 1, self.msg_dim, device=x.device)
            others_msg[..., :GOAL_SIZE] = _tgt * _has
            msg_reg = torch.zeros((), device=x.device)
        elif USE_COMMUNICATION:
            # 파트너들의 메시지를 그들의 obs로부터 재생성 → 집계. ★ rollout _get_others_msg와 동일 집계여야
            #   PPO ratio(old_logprob)가 유효함 → agg_mode/msg_gain을 여기서 그대로 미러링 ★
            import os as _os
            agg_mode = _os.environ.get('VESSEL_AGG_MODE', 'sum').lower()
            nearest_scale = float(_os.environ.get('VESSEL_NEAREST_SCALE', 0))
            msg_gain = float(_os.environ.get('VESSEL_MSG_GAIN', 1.0))
            if nearest_scale > 0:
                agg_mode = 'scale'

            # ★완전분리 MoE: 파트너 메시지를 *각 파트너의 situation*으로 라우팅(sender 코어). rollout forward의
            #   msg=msg_actor(...,situation)와 동일 함수형 → PPO ratio 유효. partner_situations[N,K]는 rollout서 저장.
            msg_part = self.msg_actor(partner_x, partner_goal, partner_self, partner_situations)  # [N,K,msg_dim]
            Kc = partner_mask.sum(dim=1, keepdim=True).clamp(min=1.0)                       # [N,1,1] 실제 파트너 수
            if self.use_attention and partner_relpos is not None:
                # ★ 위치 grounding + attention (rollout aggregate_single과 동일 함수형 → PPO ratio 유효)
                q_in = torch.cat([self_state, goal], dim=-1)                                  # [N,1,6]
                others_msg = self.attn.aggregate_batch(q_in, partner_relpos, msg_part, partner_mask)  # [N,1,6]
            elif self.pos_ground and partner_relpos is not None:
                # 위치 grounding: [상대방위·거리 + 메시지] → encoder → masked mean
                localized = self.msg_encoder(torch.cat([partner_relpos, msg_part], dim=-1))  # [N,K,6]
                others_msg = (localized * partner_mask).sum(dim=1, keepdim=True) / Kc         # [N,1,6]
            else:
                s = (msg_part * partner_mask).sum(dim=1, keepdim=True)                        # [N,1,6]
                if agg_mode == 'mean':
                    s = s / Kc
                elif agg_mode == 'scale':
                    s = s * (nearest_scale / Kc)
                others_msg = s
            if msg_gain != 1.0:
                others_msg = others_msg * msg_gain
            # 메시지 L2 정규화 항: 유효 파트너 메시지의 평균 제곱(원소당). loss에 더해져 메시지를 0쪽으로 압박.
            msg_reg = (msg_part.pow(2) * partner_mask).sum() / (partner_mask.sum().clamp(min=1.0) * self.msg_dim)
        else:
            others_msg = torch.zeros(x.shape[0], 1, self.msg_dim, device=x.device)
            msg_reg = torch.zeros((), device=x.device)

        # ★ intent self-supervised 손실: 내 obs로 own msg 재생성 → 미래의도 예측 → MSE(라벨=실제 미래변위).
        #   USE_COMMUNICATION·INTENT_COEF>0·own_future 제공 시에만. 정책/가치 경로와 독립(별도 head).
        if USE_COMMUNICATION and self.intent_coef > 0.0 and own_future is not None:
            own_msg = self.msg_actor(x, goal, self_state, situation)              # [N,1,msg_dim]
            pred = self.intent_decoder(own_msg)                        # [N,1,K*3]
            if own_future_mask is None:
                own_future_mask = torch.ones_like(own_future)
            denom = own_future_mask.sum().clamp(min=1.0)
            intent_loss = ((pred - own_future).pow(2) * own_future_mask).sum() / denom
        else:
            intent_loss = torch.zeros((), device=x.device)

        # ★ L1 threat-relay 손실: 내 obs로 own msg 재생성 → top-K 위협 기하 예측 → MSE(라벨=ego-radar 관측).
        #   USE_COMMUNICATION·THREAT_COEF>0·own_threat 제공 시에만. 정책/가치 경로와 독립(별도 head).
        #   own_threat_mask로 "ego가 못 본 위협 슬롯"(radar=maxrange) 제외 → sender가 실제 본 것만 학습(정직).
        if USE_COMMUNICATION and self.threat_coef > 0.0 and own_threat is not None:
            own_msg_t = self.msg_actor(x, goal, self_state, situation)           # [N,1,msg_dim]
            pred_t = self.threat_decoder(own_msg_t)                   # [N,1,K*4]
            if own_threat_mask is None:
                own_threat_mask = torch.ones_like(own_threat)
            denom_t = own_threat_mask.sum().clamp(min=1.0)
            threat_loss = ((pred_t - own_threat).pow(2) * own_threat_mask).sum() / denom_t
        else:
            threat_loss = torch.zeros((), device=x.device)

        # ★ L4 goal-broadcast 손실: 내 obs로 own msg 재생성 → 목적지 예측 → MSE(라벨=내 goal, self-prediction).
        #   USE_COMMUNICATION·GOAL_COMM_COEF>0 시에만. 라벨=goal(이미 인자) → 별도 텐서 불요. 정책/가치 무오염(별도 head).
        if USE_COMMUNICATION and self.goal_comm_coef > 0.0:
            own_msg_g = self.msg_actor(x, goal, self_state, situation)           # [N,1,msg_dim]
            pred_g = self.goal_decoder(own_msg_g)                     # [N,1,GOAL_SIZE]
            goal_loss = (pred_g - goal).pow(2).mean()
        else:
            goal_loss = torch.zeros((), device=x.device)

        # ★ Role-broadcast 손실 (C): 내 obs로 own msg 재생성 → COLREGs 상황(0~4) 분류 → cross-entropy.
        #   라벨 = 내 situation(이미 인자, self-prediction → 별도 텐서 불요). 정책/가치 무오염(별도 head).
        #   USE_COMMUNICATION·ROLE_COMM_COEF>0·situation 제공 시에만. comm-OFF/COEF=0이면 role_loss=0(비트동일).
        if USE_COMMUNICATION and self.role_comm_coef > 0.0 and situation is not None:
            own_msg_r = self.msg_actor(x, goal, self_state, situation)           # [N,1,msg_dim]
            logits_r = self.role_decoder(own_msg_r)                   # [N,1,num_situations]
            logits_flat = logits_r.reshape(-1, self.role_decoder.num_situations)  # [N, S]
            label_flat = situation.reshape(-1).long().clamp(0, self.role_decoder.num_situations - 1)  # [N]
            role_loss = F.cross_entropy(logits_flat, label_flat)
        else:
            role_loss = torch.zeros((), device=x.device)

        logprob, entropy, _, z_ctrl = self.ctr_actor.get_logprob_entropy(
            x, goal, self_state, others_msg, action_raw, situation
        )
        # ★통합 상태복원 손실 (2026-09-04): own msg 하나로 상태 전부 복원. 기존 5-손실 대체
        #   (사용 시 기존 계수는 0으로 → 이중계상 없음). 반환 9-튜플 불변 — _last_state_recon 속성으로 전달.
        if (USE_COMMUNICATION and self.state_recon is not None and self.state_recon_coef > 0.0
                and own_threat is not None and own_future is not None and situation is not None):
            own_msg_sr = self.msg_actor(x, goal, self_state, situation)
            _sr_loss, _sr_raw = self.state_recon.loss(
                own_msg_sr, goal, self_state, situation, own_threat, own_threat_mask,
                own_future, own_future_mask)
            self._last_state_recon = (_sr_loss, _sr_raw)
        else:
            self._last_state_recon = (torch.zeros((), device=x.device), {})

        # ★C5c consumer 손실: 수신측 backbone z로 파트너 의도(goal masked-mean) 복원 → MSE가 fc2 메시지슬라이스로
        #   gradient(수신 정책이 파트너 의도를 표상하게 강제). USE_COMMUNICATION·COMM_CONSUMER_COEF>0·非oracle만.
        #   comm-OFF/COEF=0이면 0(미가산=수신 무영향, 비트동일). oracle은 정보 직접이라 불요(skip).
        if USE_COMMUNICATION and self.ctr_actor.comm_consumer_coef > 0.0 and not USE_ORACLE:
            # ★H2 per-slot: nearest-K 파트너 각각의 goal을 슬롯별 복원(mean 아님). partner_goal은 거리순 정렬(obs_utils).
            #   K 의도가 MSG_DIM others_msg 하나를 공유 → MSG_DIM이 rate-distortion 병목 → 차원↑ = 더 많은 이웃 전달.
            Ksl = self.ctr_actor.consumer_k
            tgt_c = partner_goal[:, :Ksl, :]                                          # [N,Ksl,GOAL_SIZE] 거리순 nearest-K
            msk_c = partner_mask[:, :Ksl, :]                                          # [N,Ksl,1]
            # ★2026-09-05 fix: coupling이면 head가 이미 만든 복원값 재사용(중복 forward 제거). 아니면 재계산.
            pred_c = self.ctr_actor.pop_consumer_dec(z_ctrl, situation).view(x.shape[0], Ksl, GOAL_SIZE)  # [N,Ksl,GOAL_SIZE] (코어 라우팅)
            denom_c = msk_c.sum().clamp(min=1.0) * GOAL_SIZE
            consumer_loss = ((pred_c - tgt_c).pow(2) * msk_c).sum() / denom_c
        else:
            consumer_loss = torch.zeros((), device=x.device)
        value = self.critic(x, goal, self_state, others_msg, situation, global_feat=global_feat)
        return value, logprob, entropy, msg_reg, intent_loss, threat_loss, goal_loss, role_loss, consumer_loss
