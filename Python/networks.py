"""
Vessel Navigation Policy Network
- MessageActor: observation → 6D message
- ControlActor: observation + others_msg → action
- Critic: observation → value

PPO 구현은 CleanRL 방식을 따름 (검증된 구현)

obs 계약(59D 중 네트워크 입력 57D): radar(30, frame-stack ×3) + goal(2) + self(4) + ARPA(21).
COLREGs one-hot/auxiliary classifier는 제거됨(vessel-label leak). 충돌 기하는 label-blind ARPA(21D)로 들어옴.

★ 통신 credit assignment 수정 (이전 버그):
이전 evaluate_actions는 PPO 배치가 n_agent=1이라 straight-through self-loop만 타서
sender→receiver gradient가 0이었음. 이제 파트너 obs를 저장해두고 update 때 MessageActor를
파트너 obs로 재실행(masked-sum)하여 "내 메시지가 옆 배 회피를 도왔나" gradient가 흐르게 함.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from config import STATE_SIZE, ARPA_SIZE, USE_COMMUNICATION


class MessageActor(nn.Module):
    """
    각 에이전트의 observation을 msg_dim 메시지로 압축
    radar(섹터압축) → MLP → FC 후 tanh로 메시지 생성
    """
    def __init__(self, frames, msg_dim):
        super(MessageActor, self).__init__()
        self.frames = frames
        self.msg_dim = msg_dim

        # Radar feature extraction (Conv1D 제거 → MLP; 섹터 압축으로 입력이 작아 Conv 불필요)
        self.radar_fc = nn.Linear(STATE_SIZE * frames, 256)
        # 256 + goal(2) + self_state(4) + arpa(21) = 283
        self.fc2 = nn.Linear(256 + 2 + 4 + ARPA_SIZE, 128)
        self.msg_out = nn.Linear(128, msg_dim)
        # ★생산측 zero-init: 메시지가 step1부터 0에서 학습되게(소비측 fc2 zero-init과 짝).
        #   → comm-ON이 init에서 메시지=0 → comm-OFF와 *정확히* 동일 출발선(H1a 진짜 동등).
        with torch.no_grad():
            self.msg_out.weight.zero_()
            self.msg_out.bias.zero_()

    def forward(self, x, goal, self_state, arpa):
        """
        Args:
            x: [batch, n_agent, frames * STATE_SIZE]
            goal: [batch, n_agent, 2]
            self_state: [batch, n_agent, 4]
            arpa: [batch, n_agent, 21]
        Returns:
            msg: [batch, n_agent, msg_dim]
        """
        batch_size, n_agent, _ = x.shape

        x_flat = x.view(batch_size * n_agent, self.frames * STATE_SIZE)
        goal_flat = goal.reshape(batch_size * n_agent, -1)
        self_state_flat = self_state.reshape(batch_size * n_agent, -1)
        arpa_flat = arpa.reshape(batch_size * n_agent, -1)

        a = F.relu(self.radar_fc(x_flat))
        a = torch.cat((a, goal_flat, self_state_flat, arpa_flat), dim=-1)
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

        self.radar_fc = nn.Linear(STATE_SIZE * frames, 256)
        # 256 + goal(2) + self_state(4) + arpa(21) + others_msg(msg_dim)
        self.fc2 = nn.Linear(256 + 2 + 4 + ARPA_SIZE + msg_dim, 128)
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

    def _features(self, x, goal, self_state, arpa, others_msg):
        batch_size, n_agent, _ = x.shape
        x_flat = x.view(batch_size * n_agent, self.frames * STATE_SIZE)
        goal_flat = goal.reshape(batch_size * n_agent, -1)
        self_state_flat = self_state.reshape(batch_size * n_agent, -1)
        arpa_flat = arpa.reshape(batch_size * n_agent, -1)
        # ★게이트 적용: 도움될 때만 메시지가 흐름. rollout(forward)·update(get_logprob_entropy) 모두
        #   이 _features를 거치므로 게이트가 양쪽에 동일 적용 → PPO ratio 정합성 구조적 보장.
        others_msg_flat = (others_msg * torch.sigmoid(self.msg_gate)).reshape(batch_size * n_agent, -1)

        a = F.relu(self.radar_fc(x_flat))
        a = torch.cat((a, goal_flat, self_state_flat, arpa_flat, others_msg_flat), dim=-1)
        a = torch.tanh(self.fc2(a))
        a = torch.tanh(self.fc3(a))
        return a, batch_size, n_agent

    def forward(self, x, goal, self_state, arpa, others_msg):
        """
        Returns: action [b,n,act], logprob [b,n,1], mean [b,n,act]
        """
        a, batch_size, n_agent = self._features(x, goal, self_state, arpa, others_msg)

        action_mean = torch.clamp(self.action_mean(a), -3.0, 3.0)  # tanh(3) ≈ 0.995
        action_logstd = torch.clamp(self.action_logstd, -2.3, 0.0)  # std 0.1 ~ 1.0
        action_logstd = action_logstd.expand(batch_size * n_agent, -1)
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

    def get_logprob_entropy(self, x, goal, self_state, arpa, others_msg, action):
        """PPO 업데이트용: 주어진 action의 log_prob과 entropy 계산"""
        a, batch_size, n_agent = self._features(x, goal, self_state, arpa, others_msg)
        action_flat = action.reshape(batch_size * n_agent, -1)

        action_mean = torch.clamp(self.action_mean(a), -3.0, 3.0)
        action_logstd = torch.clamp(self.action_logstd, -2.3, 0.0)
        action_logstd = action_logstd.expand(batch_size * n_agent, -1)
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

        self.radar_fc = nn.Linear(STATE_SIZE * frames, 256)
        # 256 + goal(2) + self_state(4) + arpa(21) + others_msg(msg_dim)
        self.fc2 = nn.Linear(256 + 2 + 4 + ARPA_SIZE + msg_dim, 128)
        # ★메시지 슬라이스 zero-init (ControlActor와 동일 이유): critic이 노이즈 메시지에 조건화되는 것 방지.
        with torch.no_grad():
            self.fc2.weight[:, -msg_dim:].zero_()
        # ★메시지 게이트 (ControlActor와 동일): critic도 메시지를 기본 무시, 도움될 때만 조건화.
        self.msg_gate = nn.Parameter(torch.tensor(-3.0))
        self.value_out = nn.Linear(128, 1)

    def forward(self, x, goal, self_state, arpa, others_msg):
        batch_size, n_agent, _ = x.shape
        x_flat = x.view(batch_size * n_agent, self.frames * STATE_SIZE)
        goal_flat = goal.reshape(batch_size * n_agent, -1)
        self_state_flat = self_state.reshape(batch_size * n_agent, -1)
        arpa_flat = arpa.reshape(batch_size * n_agent, -1)
        others_msg_flat = (others_msg * torch.sigmoid(self.msg_gate)).reshape(batch_size * n_agent, -1)

        v = F.relu(self.radar_fc(x_flat))
        v = torch.cat((v, goal_flat, self_state_flat, arpa_flat, others_msg_flat), dim=-1)
        v = F.relu(self.fc2(v))
        v = self.value_out(v)
        return v.view(batch_size, n_agent, 1)


class CNNPolicy(nn.Module):
    """
    전체 정책 네트워크 (메시지 교환 기반)

    흐름:
    1. 모든 에이전트의 obs → MessageActor → 각자의 msg_dim 메시지
    2. 통신 파트너(범위 내 nearest-K) 메시지 합 = others_msg
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

    def _get_others_msg(self, msg, comm_partners=None, agent_id_list=None, comm_relpos=None):
        """메시지 교환 로직 (rollout, annealing 없음 - 즉시 100%)

        env override:
          VESSEL_AGG_MODE: 'sum' | 'mean' | 'scale' (default 'sum')
          VESSEL_NEAREST_SCALE: float, default 0 (>0이면 'scale' 자동 활성)
          VESSEL_MSG_GAIN: float, default 1.0 (최종 결과 gating 계수)
        ⚠️ evaluate_actions의 update-time 집계(masked sum)와 동일해야 PPO ratio 유효 (default 'sum').
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
            others_msg = torch.zeros_like(msg)
            id_to_idx = {aid: idx for idx, aid in enumerate(agent_id_list)}
            for i, agent_id in enumerate(agent_id_list):
                partners = comm_partners.get(agent_id, [])
                if not partners:
                    continue
                partner_indices = [id_to_idx[p] for p in partners if p in id_to_idx]
                K = len(partner_indices)
                if K == 0:
                    continue
                if self.pos_ground and comm_relpos is not None and agent_id in comm_relpos:
                    # 위치 grounding: [상대방위·거리 + 메시지] → encoder → mean
                    rp = torch.as_tensor(comm_relpos[agent_id][:K], dtype=torch.float32, device=msg.device)  # [K,3]
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

    def forward(self, x, goal, self_state, arpa,
                return_msg=False, comm_partners=None, agent_id_list=None, comm_relpos=None):
        """
        Rollout forward. Returns value, action, logprob, mean [, msg, others_msg]
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            goal = goal.unsqueeze(1)
            self_state = self_state.unsqueeze(1)
            arpa = arpa.unsqueeze(1)

        # 1. 메시지 생성
        msg = self.msg_actor(x, goal, self_state, arpa)
        # 2. 메시지 교환 (위치 grounding 시 comm_relpos 사용)
        others_msg = self._get_others_msg(msg, comm_partners, agent_id_list, comm_relpos)
        # 3. 행동
        action, logprob, mean = self.ctr_actor(x, goal, self_state, arpa, others_msg)
        # 4. 가치
        value = self.critic(x, goal, self_state, arpa, others_msg)

        if return_msg:
            return value, action, logprob, mean, msg, others_msg
        return value, action, logprob, mean

    def evaluate_actions(self, x, goal, self_state, arpa,
                         partner_x, partner_goal, partner_self, partner_arpa, partner_mask,
                         partner_relpos, action):
        """
        PPO 업데이트용. ★통신 sender→receiver gradient 수정★
        통신 ON이면 파트너 obs로 MessageActor를 재실행(미분가능)하여 others_msg를 재구성.
        masked-sum 집계 = rollout _get_others_msg의 'sum'과 동일 → PPO ratio(old_logprob) 유효.
        MessageActor는 공유 가중치 → 파트너 메시지의 gradient가 sender 학습으로 흐름.

        x: [N,1,F*S], partner_x: [N,K,F*S], partner_mask: [N,K,1]
        Returns: value [N,1,1], logprob [N,1,1], entropy (scalar)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            goal = goal.unsqueeze(1)
            self_state = self_state.unsqueeze(1)
            arpa = arpa.unsqueeze(1)
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

            msg_part = self.msg_actor(partner_x, partner_goal, partner_self, partner_arpa)  # [N,K,6]
            Kc = partner_mask.sum(dim=1, keepdim=True).clamp(min=1.0)                       # [N,1,1] 실제 파트너 수
            if self.pos_ground and partner_relpos is not None:
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

        logprob, entropy, _ = self.ctr_actor.get_logprob_entropy(
            x, goal, self_state, arpa, others_msg, action
        )
        value = self.critic(x, goal, self_state, arpa, others_msg)
        return value, logprob, entropy, msg_reg
