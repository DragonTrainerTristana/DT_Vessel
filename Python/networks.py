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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from config import (STATE_SIZE, RADAR_FEAT_DIM, USE_COMMUNICATION,
                    SELF_STATE_SIZE, GOAL_SIZE, USE_ATTENTION, ATTN_DIM,
                    INTENT_COEF, INTENT_K, THREAT_COEF, THREAT_K, GOAL_COMM_COEF,
                    ROLE_COMM_COEF, USE_MOE, NUM_COLREGS_SITUATIONS, MAX_COMM_PARTNERS,
                    COMM_CONSUMER_COEF, COMM_CONSUMER_K, COMM_CONSUMER_COUPLING, USE_ORACLE)


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
    def __init__(self, frames, n_rays, out_dim=RADAR_FEAT_DIM):
        super(RadarEncoder, self).__init__()
        self.frames = frames
        self.n_rays = n_rays
        self.conv1 = nn.Conv1d(frames, 32, kernel_size=5, stride=2, padding=2, padding_mode='circular')
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, padding_mode='circular')
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, padding_mode='circular')
        # conv flatten 크기는 n_rays에 의존 → 더미 forward로 산출(차원 하드코딩 방지).
        with torch.no_grad():
            _flat = self._conv(torch.zeros(1, frames, n_rays)).shape[1]
        self.fc = nn.Linear(_flat, out_dim)

    def _conv(self, x):
        a = F.relu(self.conv1(x))
        a = F.relu(self.conv2(a))
        a = F.relu(self.conv3(a))
        return a.reshape(a.shape[0], -1)

    def forward(self, x):
        """x: [batch, n_agent, frames*n_rays] 또는 [M, frames*n_rays] → [M, out_dim] (post-ReLU)."""
        x = x.reshape(-1, self.frames, self.n_rays)
        return F.relu(self.fc(self._conv(x)))


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
        token = torch.cat([relpos, msg_p], dim=-1)             # [K, R+M]
        keys = self.k_proj(token)                              # [K, d]
        vals = self.v_proj(token)                              # [K, M]
        scores = (keys @ query) / self.scale                   # [K]
        alpha = torch.softmax(scores, dim=0)                   # [K]
        return (alpha.unsqueeze(-1) * vals).sum(dim=0)         # [M]

    def aggregate_batch(self, q_in, relpos, msg_part, mask):
        """update 배치 집계. q_in [N,1,Q], relpos [N,K,R], msg_part [N,K,M], mask [N,K,1]
        → context [N,1,M]. padding(mask=0)은 -inf 마스킹으로 softmax 제외, 전무파트너는 0."""
        query = self.q_proj(q_in)                              # [N,1,d]
        token = torch.cat([relpos, msg_part], dim=-1)          # [N,K,R+M]
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
    def __init__(self, frames, msg_dim):
        super(_MessageActorCore, self).__init__()
        self.msg_dim = msg_dim
        # Radar feature extraction: 360 raw ray → 학습형 Conv1D 압축(원형 padding). min-pool 제거(2026-06-04).
        self.radar_encoder = RadarEncoder(frames, STATE_SIZE, RADAR_FEAT_DIM)
        # RADAR_FEAT_DIM + goal(2) + self_state(4)
        self.fc2 = nn.Linear(RADAR_FEAT_DIM + 2 + 4, 128)
        self.msg_out = nn.Linear(128, msg_dim)
        # ★생산측 소진폭 init (2026-06-12 채널동결 fix): weight+bias zero-init은 msg≡tanh(0)=0을 만들어
        #   소비측 zero-init과 직렬 곱 새들 형성 → 채널 전체 grad 항등 0, 1M step 비트동결(실측).
        #   소진폭(×0.1)이면 메시지가 step1부터 상태의존적이되 작음(tanh 선형구간) → 학습 신호 생존.
        with torch.no_grad():
            self.msg_out.weight.mul_(0.1)
            self.msg_out.bias.zero_()

    def forward(self, x_flat, goal_flat, self_state_flat):
        """x_flat [M, frames*STATE_SIZE], goal_flat [M,2], self_state_flat [M,4] → msg [M, msg_dim]."""
        a = self.radar_encoder(x_flat)   # [M, RADAR_FEAT_DIM] (post-ReLU)
        a = torch.cat((a, goal_flat, self_state_flat), dim=-1)
        a = F.relu(self.fc2(a))
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
            self.experts = nn.ModuleList([_MessageActorCore(frames, msg_dim) for _ in range(self.num_experts)])
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

        if not self.use_moe:
            msg = self.core(x_f, goal_f, self_f)
        else:
            # 상황별 hard-route: 각 sample은 자기 상황 코어만 통과 → 그 코어만 gradient.
            if situation is not None:
                sit = situation.reshape(M).long().clamp(0, self.num_experts - 1)
            else:
                sit = torch.zeros(M, dtype=torch.long, device=x.device)   # 상황 없으면 None(0) 코어
            msg = x_f.new_zeros(M, self.msg_dim)
            for k in range(self.num_experts):
                mask = (sit == k)
                if mask.any():
                    msg[mask] = self.experts[k](x_f[mask], goal_f[mask], self_f[mask])
        return msg.view(batch_size, n_agent, self.msg_dim)


class _ControlActorCore(nn.Module):
    """단일 ControlActor 본체(radar_encoder + fc2 + msg_gate + consumer_decoder + fc3 + action head).

    ★완전분리 MoE(2026-06-26): radar_encoder를 *코어 안*에 둠 → USE_MOE=1이면 상황별 5벌이 perception까지
      완전 독립(과거의 공유 backbone 폐기). backbone(z)·head(mean,logstd) flat 입력으로 동작.
    """
    def __init__(self, frames, msg_dim, action_size):
        super(_ControlActorCore, self).__init__()
        self.msg_dim = msg_dim
        self.action_size = action_size
        self.radar_encoder = RadarEncoder(frames, STATE_SIZE, RADAR_FEAT_DIM)  # 360 raw ray → 학습형 Conv1D 압축
        # RADAR_FEAT_DIM + goal(2) + self_state(4) + others_msg(msg_dim)
        self.fc2 = nn.Linear(RADAR_FEAT_DIM + 2 + 4 + msg_dim, 128)
        # ★메시지 슬라이스 소진폭 init (2026-06-12 채널동결 fix): zero-init은 ∂L/∂msg≡0을 만들어 생산측
        #   zero와 직렬 곱 새들 형성 → 채널 영구 동결. 소진폭(×0.1)은 grad를 살리되 초기 메시지 영향을 작게.
        with torch.no_grad():
            self.fc2.weight[:, -msg_dim:].mul_(0.1)
        # ★메시지 게이트(learnable 다이얼, 초기 0=sigmoid 0.5 중립): others_msg * sigmoid(msg_gate).
        self.msg_gate = nn.Parameter(torch.tensor(0.0))
        # ★C5c 수신측 decode-in-policy: backbone z로 파트너 의도(goal) 복원 head (코어별 독립).
        self.consumer_k = max(1, min(COMM_CONSUMER_K, MAX_COMM_PARTNERS))
        self.consumer_decoder = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, self.consumer_k * GOAL_SIZE)
        )
        # ★H2 coupling: decode(재구성)를 fc3 입력에 concat. OFF면 fc3 입력=128.
        self.consumer_coupling = COMM_CONSUMER_COUPLING
        _fc3_in = 128 + (self.consumer_k * GOAL_SIZE if self.consumer_coupling else 0)
        self.fc3 = nn.Linear(_fc3_in, 64)
        self.action_mean = nn.Linear(64, action_size)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.zero_()
        # Learnable log std — per-dim: rudder(dim0) -1.0(std≈0.37), thrust(dim1) -0.5(std≈0.61).
        if action_size == 2:
            self.action_logstd = nn.Parameter(torch.tensor([[-1.0, -0.5]]))
        else:
            self.action_logstd = nn.Parameter(torch.full((1, action_size), -0.5))

    def backbone(self, x_f, goal_f, self_f, others_msg_f):
        """flat 입력 → z[M,128]. 게이트가 forward·update 양쪽에 동일 적용 → PPO ratio 정합."""
        gated = others_msg_f * torch.sigmoid(self.msg_gate)
        radar_feat = self.radar_encoder(x_f)   # [M, RADAR_FEAT_DIM] (post-ReLU)
        return torch.tanh(self.fc2(torch.cat((radar_feat, goal_f, self_f, gated), dim=-1)))

    def head(self, z):
        """z[M,128] → mean[M,act], logstd[M,act]. consumer_coupling이면 decode를 z에 concat."""
        zc = torch.cat([z, self.consumer_decoder(z)], dim=-1) if self.consumer_coupling else z
        a = torch.tanh(self.fc3(zc))
        mean = self.action_mean(a)
        logstd = self.action_logstd.expand(z.shape[0], -1)
        return mean, logstd


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
            self.experts = nn.ModuleList(
                [_ControlActorCore(frames, msg_dim, action_size) for _ in range(self.num_experts)])
        else:
            self.core = _ControlActorCore(frames, msg_dim, action_size)

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
        if not self.use_moe:
            z = self.core.backbone(x_f, goal_f, self_f, om_f)
            mean, logstd = self.core.head(z)
            return z, mean, logstd, batch_size, n_agent
        # 상황별 hard-route (코어 통째). 각 sample은 자기 상황 코어만 통과 → 그 코어만 gradient.
        if situation is not None:
            sit = situation.reshape(M).long().clamp(0, self.num_experts - 1)
        else:
            sit = torch.zeros(M, dtype=torch.long, device=x.device)
        z = x_f.new_zeros(M, 128)
        mean = x_f.new_zeros(M, self.action_size)
        logstd = x_f.new_zeros(M, self.action_size)
        for k in range(self.num_experts):
            mask = (sit == k)
            if mask.any():
                zk = self.experts[k].backbone(x_f[mask], goal_f[mask], self_f[mask], om_f[mask])
                mk, lk = self.experts[k].head(zk)
                z[mask] = zk
                mean[mask] = mk
                logstd[mask] = lk
        return z, mean, logstd, batch_size, n_agent

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
        """Returns: action [b,n,act], logprob [b,n,1], mean [b,n,act]. situation: [b,n] or [b,n,1] (0~4)."""
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
        return action, logprob, action_mean

    def get_logprob_entropy(self, x, goal, self_state, others_msg, action, situation=None):
        """PPO 업데이트용: 주어진 action의 log_prob과 entropy 계산.
        ★situation은 rollout 저장값 → forward와 *동일* 코어 라우팅 → old_logprob 정합(PPO ratio 유효)."""
        z, action_mean, action_logstd, batch_size, n_agent = self._route(
            x, goal, self_state, others_msg, situation)
        action_flat = action.reshape(batch_size * n_agent, -1)

        action_mean = torch.clamp(action_mean, -3.0, 3.0)
        action_logstd = torch.clamp(action_logstd, -2.3, 0.0)
        action_std = torch.exp(action_logstd)

        # action은 이미 tanh 적용값 → arctanh로 역변환
        action_clamped = torch.clamp(action_flat, -0.999, 0.999)
        action_raw = 0.5 * torch.log((1 + action_clamped) / (1 - action_clamped))

        dist = Normal(action_mean, action_std)
        logprob = dist.log_prob(action_raw) - torch.log(1 - action_clamped.pow(2) + 1e-6)
        logprob = logprob.sum(dim=-1, keepdim=True)

        # Squashed Gaussian entropy 보정
        gaussian_entropy = dist.entropy().sum(dim=-1)
        squash_correction = torch.log(1 - action_clamped.pow(2) + 1e-6).sum(dim=-1)
        entropy = (gaussian_entropy + squash_correction).mean()

        logprob = logprob.view(batch_size, n_agent, -1)
        action_mean = action_mean.view(batch_size, n_agent, -1)
        # ★z [B*N,128] 반환(C5c): evaluate_actions가 consumer_decode(z, situation)로 파트너 의도 복원 손실 계산.
        return logprob, entropy, action_mean, z


class _CriticCore(nn.Module):
    """단일 Critic 본체(radar_encoder + fc2 + msg_gate + value_out). 완전분리 MoE에서 상황별 5벌 독립.
    ★과거의 situation one-hot 조건화 폐기 — 상황별로 코어를 통째 라우팅하므로 가치망도 perception까지 독립."""
    def __init__(self, frames, msg_dim):
        super(_CriticCore, self).__init__()
        self.radar_encoder = RadarEncoder(frames, STATE_SIZE, RADAR_FEAT_DIM)  # 360 raw ray → 학습형 Conv1D 압축
        # RADAR_FEAT_DIM + goal(2) + self_state(4) + others_msg(msg_dim)
        self.fc2 = nn.Linear(RADAR_FEAT_DIM + 2 + 4 + msg_dim, 128)
        with torch.no_grad():
            self.fc2.weight[:, -msg_dim:].mul_(0.1)   # 메시지 슬라이스 소진폭 init (채널동결 fix)
        self.msg_gate = nn.Parameter(torch.tensor(0.0))
        self.value_out = nn.Linear(128, 1)

    def forward(self, x_f, goal_f, self_f, others_msg_f):
        gated = others_msg_f * torch.sigmoid(self.msg_gate)
        v = self.radar_encoder(x_f)   # [M, RADAR_FEAT_DIM] (post-ReLU)
        v = F.relu(self.fc2(torch.cat((v, goal_f, self_f, gated), dim=-1)))
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
            self.experts = nn.ModuleList([_CriticCore(frames, msg_dim) for _ in range(self.num_experts)])
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

    def forward(self, x, goal, self_state, others_msg, situation=None):
        batch_size, n_agent, _ = x.shape
        M = batch_size * n_agent
        x_f = x.reshape(M, -1)
        goal_f = goal.reshape(M, -1)
        self_f = self_state.reshape(M, -1)
        om_f = others_msg.reshape(M, -1)
        if not self.use_moe:
            v = self.core(x_f, goal_f, self_f, om_f)
        else:
            if situation is not None:
                sit = situation.reshape(M).long().clamp(0, self.num_experts - 1)
            else:
                sit = torch.zeros(M, dtype=torch.long, device=x.device)
            v = x_f.new_zeros(M, 1)
            for k in range(self.num_experts):
                mask = (sit == k)
                if mask.any():
                    v[mask] = self.experts[k](x_f[mask], goal_f[mask], self_f[mask], om_f[mask])
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

        # ★ 위치 grounding (AIS-style): 파트너의 [상대방위(sin,cos)+거리] 3D를 메시지에 결합 →
        #   receiver가 "어느 방위에서 온 메시지"인지 알게 됨. VESSEL_POS_GROUND=1일 때만 사용.
        #   relpos는 "주소"(어디서), 학습 6D latent는 "내용"(의도) → 학습메시지 thesis 유지.
        import os as _os
        self.pos_ground = _os.environ.get('VESSEL_POS_GROUND', '0') == '1'
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
        if (USE_ORACLE and USE_COMMUNICATION and comm_partners is not None
                and agent_id_list is not None and goal is not None):
            id_to_idx = {aid: idx for idx, aid in enumerate(agent_id_list)}
            others_msg = torch.zeros_like(msg)
            for i, agent_id in enumerate(agent_id_list):
                partners = comm_partners.get(agent_id, [])
                pidx = [id_to_idx[p] for p in partners if p in id_to_idx]
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

    def forward(self, x, goal, self_state,
                return_msg=False, comm_partners=None, agent_id_list=None, comm_relpos=None,
                situation=None):
        """
        Rollout forward. Returns value, action, logprob, mean [, msg, others_msg]
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
        action, logprob, mean = self.ctr_actor(x, goal, self_state, others_msg, situation)
        # 4. 가치 (situation one-hot 조건화)
        value = self.critic(x, goal, self_state, others_msg, situation)

        if return_msg:
            return value, action, logprob, mean, msg, others_msg
        return value, action, logprob, mean

    def evaluate_actions(self, x, goal, self_state,
                         partner_x, partner_goal, partner_self, partner_mask,
                         partner_relpos, action, own_future=None, own_future_mask=None,
                         own_threat=None, own_threat_mask=None, situation=None,
                         partner_situations=None):
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
            action = action.unsqueeze(1)

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
            x, goal, self_state, others_msg, action, situation
        )
        # ★C5c consumer 손실: 수신측 backbone z로 파트너 의도(goal masked-mean) 복원 → MSE가 fc2 메시지슬라이스로
        #   gradient(수신 정책이 파트너 의도를 표상하게 강제). USE_COMMUNICATION·COMM_CONSUMER_COEF>0·非oracle만.
        #   comm-OFF/COEF=0이면 0(미가산=수신 무영향, 비트동일). oracle은 정보 직접이라 불요(skip).
        if USE_COMMUNICATION and self.ctr_actor.comm_consumer_coef > 0.0 and not USE_ORACLE:
            # ★H2 per-slot: nearest-K 파트너 각각의 goal을 슬롯별 복원(mean 아님). partner_goal은 거리순 정렬(obs_utils).
            #   K 의도가 MSG_DIM others_msg 하나를 공유 → MSG_DIM이 rate-distortion 병목 → 차원↑ = 더 많은 이웃 전달.
            Ksl = self.ctr_actor.consumer_k
            tgt_c = partner_goal[:, :Ksl, :]                                          # [N,Ksl,GOAL_SIZE] 거리순 nearest-K
            msk_c = partner_mask[:, :Ksl, :]                                          # [N,Ksl,1]
            pred_c = self.ctr_actor.consumer_decode(z_ctrl, situation).view(x.shape[0], Ksl, GOAL_SIZE)  # [N,Ksl,GOAL_SIZE] (코어 라우팅)
            denom_c = msk_c.sum().clamp(min=1.0) * GOAL_SIZE
            consumer_loss = ((pred_c - tgt_c).pow(2) * msk_c).sum() / denom_c
        else:
            consumer_loss = torch.zeros((), device=x.device)
        value = self.critic(x, goal, self_state, others_msg, situation)
        return value, logprob, entropy, msg_reg, intent_loss, threat_loss, goal_loss, role_loss, consumer_loss
