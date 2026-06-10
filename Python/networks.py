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
                    INTENT_COEF, INTENT_K, USE_MOE, NUM_COLREGS_SITUATIONS, MAX_COMM_PARTNERS)


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


class GroundedAttention(nn.Module):
    """
    위치 grounding + single-head attention 집계 (sum/mean 대체).

    - query : receiver의 [self_state ⊕ goal] (= 내 상황) → q_proj
    - key/value : 각 partner의 [relpos(sin,cos,거리) ⊕ msg] (= 어디서 온 어떤 의도) → k_proj/v_proj
    - context = Σ_j softmax(q·k/√d)_j · v_j   (sum의 무차별 합 대신 가중선택)

    출력차원 = msg_dim → ControlActor/Critic 게이트·fc2 메시지슬롯 *불변*.
    ★v_proj zero-init → init에서 context=0 → comm-ON이 comm-OFF와 정확히 같은 출발선(H1a).
      도움될 때만 v_proj 가중치가 0에서 자람. (msg_out·fc2 zero-init 철학과 동일.)
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
        # ★ context=0 at init (H1a). q/k는 random이어도 v=0이라 출력 0 → 무영향.
        with torch.no_grad():
            self.v_proj.weight.zero_()
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


class MessageActor(nn.Module):
    """
    각 에이전트의 observation을 msg_dim 메시지로 압축
    radar(360 raw ray → RadarEncoder Conv1D 압축) → MLP → FC 후 tanh로 메시지 생성
    """
    def __init__(self, frames, msg_dim):
        super(MessageActor, self).__init__()
        self.frames = frames
        self.msg_dim = msg_dim

        # Radar feature extraction: 360 raw ray → 학습형 Conv1D 압축(원형 padding). min-pool 제거(2026-06-04).
        self.radar_encoder = RadarEncoder(frames, STATE_SIZE, RADAR_FEAT_DIM)
        # RADAR_FEAT_DIM + goal(2) + self_state(4)
        self.fc2 = nn.Linear(RADAR_FEAT_DIM + 2 + 4, 128)
        self.msg_out = nn.Linear(128, msg_dim)
        # ★생산측 zero-init: 메시지가 step1부터 0에서 학습되게(소비측 fc2 zero-init과 짝).
        #   → comm-ON이 init에서 메시지=0 → comm-OFF와 *정확히* 동일 출발선(H1a 진짜 동등).
        with torch.no_grad():
            self.msg_out.weight.zero_()
            self.msg_out.bias.zero_()

    def forward(self, x, goal, self_state):
        """
        Args:
            x: [batch, n_agent, frames * STATE_SIZE]
            goal: [batch, n_agent, 2]
            self_state: [batch, n_agent, 4]
        Returns:
            msg: [batch, n_agent, msg_dim]
        """
        batch_size, n_agent, _ = x.shape

        goal_flat = goal.reshape(batch_size * n_agent, -1)
        self_state_flat = self_state.reshape(batch_size * n_agent, -1)

        a = self.radar_encoder(x)   # [B*N, RADAR_FEAT_DIM] (post-ReLU)
        a = torch.cat((a, goal_flat, self_state_flat), dim=-1)
        a = F.relu(self.fc2(a))

        msg = torch.tanh(self.msg_out(a))  # bounded [-1, 1]
        return msg.view(batch_size, n_agent, self.msg_dim)


class ControlActor(nn.Module):
    """
    자기 observation + 타 에이전트 메시지로 행동 결정
    CleanRL 방식: Normal distribution 사용, tanh squashing
    """
    def __init__(self, frames, msg_dim, action_size):
        super(ControlActor, self).__init__()
        self.frames = frames
        self.msg_dim = msg_dim
        self.action_size = action_size
        self.use_moe = USE_MOE                         # COLREGs 상황별 정책 head 라우팅
        self.num_experts = NUM_COLREGS_SITUATIONS      # 5 (None/HeadOn/StandOn/GiveWay/Overtaking)

        self.radar_encoder = RadarEncoder(frames, STATE_SIZE, RADAR_FEAT_DIM)  # 360 raw ray → 학습형 Conv1D 압축
        # RADAR_FEAT_DIM + goal(2) + self_state(4) + others_msg(msg_dim)
        self.fc2 = nn.Linear(RADAR_FEAT_DIM + 2 + 4 + msg_dim, 128)
        # ★메시지 슬라이스 zero-init: comm-ON을 comm-OFF와 정확히 같은 출발선에(value-of-information≥0 보장).
        #   도움될 때만 가중치가 0에서 자람. comm-OFF 경로는 zeros 입력이라 불변(anti-rigging 100% 안전).
        with torch.no_grad():
            self.fc2.weight[:, -msg_dim:].zero_()
        # ★메시지 게이트(learnable, 초기 닫힘): others_msg * sigmoid(msg_gate). sigmoid(-3)≈0.047.
        #   "메시지 무시"를 init-time 성질이 아니라 *학습된 안정 평형*으로 만든다(H1a 위배 근본수정).
        #   메시지가 advantage를 유의하게 줄일 때만 gate가 열림(loss에 개방 페널티 MSG_GATE_COEF).
        #   comm-OFF는 others_msg≡0이라 gate가 무의미·무영향(anti-rigging 안전).
        self.msg_gate = nn.Parameter(torch.tensor(-3.0))
        self.fc3 = nn.Linear(128, 64)

        self.action_mean = nn.Linear(64, action_size)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.zero_()

        # Learnable log std — per-dim: rudder(dim0) 차분(-1.0→std≈0.37), thrust(dim1) -0.5(std≈0.61).
        # 차원공유 스칼라였을 때 rudder가 thrust와 같은 큰 std 강요받아 깔작 노이즈 유발 → 분리.
        if action_size == 2:
            self.action_logstd = nn.Parameter(torch.tensor([[-1.0, -0.5]]))
        else:
            self.action_logstd = nn.Parameter(torch.full((1, action_size), -0.5))

        # ── ★COLREGs MoE: 상황별 maneuvering head 5개 (USE_MOE=1일 때만 생성·사용) ──
        #   공유 backbone(radar_fc+fc2 → z 128D) 위에 fc3→mean→logstd만 상황별로 분리.
        #   ⚠️ USE_MOE=0이면 미생성 → state_dict 키가 단일망과 동일 = 기존 체크포인트 strict 로드 가능.
        #   각 head는 단일 head와 동일 init(mean weight×0.1, bias0, per-dim logstd) → 공정 출발선.
        if self.use_moe:
            self.head_fc3 = nn.ModuleList([nn.Linear(128, 64) for _ in range(self.num_experts)])
            self.head_mean = nn.ModuleList([nn.Linear(64, action_size) for _ in range(self.num_experts)])
            for h in self.head_mean:
                h.weight.data.mul_(0.1); h.bias.data.zero_()
            if action_size == 2:
                self.head_logstd = nn.ParameterList(
                    [nn.Parameter(torch.tensor([[-1.0, -0.5]])) for _ in range(self.num_experts)])
            else:
                self.head_logstd = nn.ParameterList(
                    [nn.Parameter(torch.full((1, action_size), -0.5)) for _ in range(self.num_experts)])

    def _backbone(self, x, goal, self_state, others_msg):
        """공유 backbone: obs+통신 → z(128D). USE_MOE 무관 공통(인지·통신 융합)."""
        batch_size, n_agent, _ = x.shape
        goal_flat = goal.reshape(batch_size * n_agent, -1)
        self_state_flat = self_state.reshape(batch_size * n_agent, -1)
        # ★게이트 적용: 도움될 때만 메시지가 흐름. rollout(forward)·update(get_logprob_entropy) 모두
        #   이 backbone을 거치므로 게이트가 양쪽에 동일 적용 → PPO ratio 정합성 구조적 보장.
        others_msg_flat = (others_msg * torch.sigmoid(self.msg_gate)).reshape(batch_size * n_agent, -1)

        a = self.radar_encoder(x)   # [B*N, RADAR_FEAT_DIM] (post-ReLU)
        a = torch.cat((a, goal_flat, self_state_flat, others_msg_flat), dim=-1)
        z = torch.tanh(self.fc2(a))   # [B*N, 128] 공유 embedding
        return z, batch_size, n_agent

    def _head(self, z, situation_flat):
        """z[B*N,128] → action_mean[B*N,act], action_logstd[B*N,act].
        ★USE_MOE=0 또는 situation 없음 → 단일 head(기존 fc3/action_mean/logstd) = 비트동일.
        ★USE_MOE=1 → situation(0~4)으로 head hard-route. rollout==update 동일 situation → PPO 유효.
        """
        if not self.use_moe or situation_flat is None:
            a = torch.tanh(self.fc3(z))
            mean = self.action_mean(a)
            logstd = self.action_logstd.expand(z.shape[0], -1)
            return mean, logstd
        # MoE: 상황별 head 라우팅 (hard switch). 각 sample은 자기 상황 head만 통과 → 그 head만 gradient.
        mean = z.new_zeros(z.shape[0], self.action_size)
        logstd = z.new_zeros(z.shape[0], self.action_size)
        sit = situation_flat.long().clamp(0, self.num_experts - 1)
        for k in range(self.num_experts):
            mask = (sit == k)
            if mask.any():
                a_k = torch.tanh(self.head_fc3[k](z[mask]))
                mean[mask] = self.head_mean[k](a_k)
                logstd[mask] = self.head_logstd[k].expand(int(mask.sum()), -1)
        return mean, logstd

    def forward(self, x, goal, self_state, others_msg, situation=None):
        """
        Returns: action [b,n,act], logprob [b,n,1], mean [b,n,act]
        situation: [b,n] or [b,n,1] (COLREGs 0~4). USE_MOE=1일 때 head 라우팅. None/USE_MOE=0이면 단일 head.
        """
        z, batch_size, n_agent = self._backbone(x, goal, self_state, others_msg)
        sit_flat = situation.reshape(-1) if situation is not None else None
        action_mean, action_logstd = self._head(z, sit_flat)

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
        ★situation은 rollout에서 저장된 값 → forward와 *동일* head 라우팅 → old_logprob 정합(PPO ratio 유효)."""
        z, batch_size, n_agent = self._backbone(x, goal, self_state, others_msg)
        sit_flat = situation.reshape(-1) if situation is not None else None
        action_mean, action_logstd = self._head(z, sit_flat)
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
        return logprob, entropy, action_mean


class Critic(nn.Module):
    """Value function estimator (others_msg 포함)"""
    def __init__(self, frames, msg_dim):
        super(Critic, self).__init__()
        self.frames = frames
        self.use_moe = USE_MOE
        self.num_experts = NUM_COLREGS_SITUATIONS

        self.radar_encoder = RadarEncoder(frames, STATE_SIZE, RADAR_FEAT_DIM)  # 360 raw ray → 학습형 Conv1D 압축
        # RADAR_FEAT_DIM + goal(2) + self_state(4) + [situation one-hot(num_experts) if MoE] + others_msg(msg_dim)
        # ⚠️ situation one-hot은 *메시지 앞*에 삽입 → 메시지가 항상 cat 마지막 → zero-init [:, -msg_dim:] 불변.
        _extra = self.num_experts if self.use_moe else 0
        self.fc2 = nn.Linear(RADAR_FEAT_DIM + 2 + 4 + _extra + msg_dim, 128)
        # ★메시지 슬라이스 zero-init (ControlActor와 동일 이유): critic이 노이즈 메시지에 조건화되는 것 방지.
        with torch.no_grad():
            self.fc2.weight[:, -msg_dim:].zero_()
        # ★메시지 게이트 (ControlActor와 동일): critic도 메시지를 기본 무시, 도움될 때만 조건화.
        self.msg_gate = nn.Parameter(torch.tensor(-3.0))
        self.value_out = nn.Linear(128, 1)

    def forward(self, x, goal, self_state, others_msg, situation=None):
        batch_size, n_agent, _ = x.shape
        goal_flat = goal.reshape(batch_size * n_agent, -1)
        self_state_flat = self_state.reshape(batch_size * n_agent, -1)
        others_msg_flat = (others_msg * torch.sigmoid(self.msg_gate)).reshape(batch_size * n_agent, -1)

        v = self.radar_encoder(x)   # [B*N, RADAR_FEAT_DIM] (post-ReLU)
        if self.use_moe:
            # 상황 one-hot 조건화(가치도 상황 인지). situation 없으면 zeros(robust). msg 앞에 삽입.
            if situation is not None:
                sit = situation.reshape(-1).long().clamp(0, self.num_experts - 1)
                sit_onehot = F.one_hot(sit, self.num_experts).float()
            else:
                sit_onehot = v.new_zeros(batch_size * n_agent, self.num_experts)
            v = torch.cat((v, goal_flat, self_state_flat, sit_onehot, others_msg_flat), dim=-1)
        else:
            v = torch.cat((v, goal_flat, self_state_flat, others_msg_flat), dim=-1)
        v = F.relu(self.fc2(v))
        v = self.value_out(v)
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
        msg = self.msg_actor(x, goal, self_state)
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
                         partner_relpos, action, own_future=None, own_future_mask=None, situation=None):
        """
        PPO 업데이트용. ★통신 sender→receiver gradient 수정★
        통신 ON이면 파트너 obs로 MessageActor를 재실행(미분가능)하여 others_msg를 재구성.
        집계(sum/mean/scale/attention/pos_ground)는 rollout _get_others_msg와 동일 함수형으로
        미러링 → PPO ratio(old_logprob) 유효. (아래 분기는 _get_others_msg와 1:1 대응)
        MessageActor는 공유 가중치 → 파트너 메시지의 gradient가 sender 학습으로 흐름.

        ★Phase2 intent: own_future(=내 미래 K-step 변위/heading, self-supervised 라벨)가 주어지고
          INTENT_COEF>0이면, 내 obs로 재생성한 own msg를 IntentDecoder로 통과시켜 미래를 예측,
          MSE 손실을 반환(MessageActor로만 gradient → "메시지가 미래의도 인코딩" 강제). 정책/가치 무오염.

        x: [N,1,F*S], partner_x: [N,K,F*S], partner_mask: [N,K,1], own_future: [N,1,K*3]
        Returns: value [N,1,1], logprob [N,1,1], entropy (scalar), msg_reg, intent_loss
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            goal = goal.unsqueeze(1)
            self_state = self_state.unsqueeze(1)
            action = action.unsqueeze(1)

        if USE_COMMUNICATION:
            # 파트너들의 메시지를 그들의 obs로부터 재생성 → 집계. ★ rollout _get_others_msg와 동일 집계여야
            #   PPO ratio(old_logprob)가 유효함 → agg_mode/msg_gain을 여기서 그대로 미러링 ★
            import os as _os
            agg_mode = _os.environ.get('VESSEL_AGG_MODE', 'sum').lower()
            nearest_scale = float(_os.environ.get('VESSEL_NEAREST_SCALE', 0))
            msg_gain = float(_os.environ.get('VESSEL_MSG_GAIN', 1.0))
            if nearest_scale > 0:
                agg_mode = 'scale'

            msg_part = self.msg_actor(partner_x, partner_goal, partner_self)  # [N,K,6]
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
            own_msg = self.msg_actor(x, goal, self_state)              # [N,1,msg_dim]
            pred = self.intent_decoder(own_msg)                        # [N,1,K*3]
            if own_future_mask is None:
                own_future_mask = torch.ones_like(own_future)
            denom = own_future_mask.sum().clamp(min=1.0)
            intent_loss = ((pred - own_future).pow(2) * own_future_mask).sum() / denom
        else:
            intent_loss = torch.zeros((), device=x.device)

        logprob, entropy, _ = self.ctr_actor.get_logprob_entropy(
            x, goal, self_state, others_msg, action, situation
        )
        value = self.critic(x, goal, self_state, others_msg, situation)
        return value, logprob, entropy, msg_reg, intent_loss
