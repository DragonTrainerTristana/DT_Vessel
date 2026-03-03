"""
Vessel Navigation Policy Network
- MessageActor: observation → 6D message
- ControlActor: observation + others_msg → action
- Critic: observation → value
- COLREGs Classifier: radar → situation prediction (auxiliary task)

PPO 구현은 CleanRL 방식을 따름 (검증된 구현)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from config import STATE_SIZE, USE_COMMUNICATION


class MessageActor(nn.Module):
    """
    각 에이전트의 observation을 6D 메시지로 압축
    Conv1D로 레이더 처리 후 tanh로 메시지 생성
    """
    def __init__(self, frames, msg_dim):
        super(MessageActor, self).__init__()
        self.frames = frames
        self.msg_dim = msg_dim

        # Conv1D layers for radar feature extraction
        self.conv1 = nn.Conv1d(in_channels=frames, out_channels=32,
                               kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=2, padding=1)

        # Conv output: (360 -> 179 -> 90) * 32 = 2880
        self.fc1 = nn.Linear(90 * 32, 256)
        # 256 + goal(2) + self_state(4) + colregs(5) = 267
        self.fc2 = nn.Linear(256 + 2 + 4 + 5, 128)
        self.msg_out = nn.Linear(128, msg_dim)

    def forward(self, x, goal, self_state, colregs):
        """
        Args:
            x: [batch, n_agent, frames * STATE_SIZE]
            goal: [batch, n_agent, 2]
            self_state: [batch, n_agent, 4]
            colregs: [batch, n_agent, 5]
        Returns:
            msg: [batch, n_agent, msg_dim]
        """
        batch_size, n_agent, _ = x.shape

        # Flatten for processing
        x_flat = x.view(batch_size * n_agent, self.frames, STATE_SIZE)
        goal_flat = goal.view(batch_size * n_agent, -1)
        self_state_flat = self_state.view(batch_size * n_agent, -1)
        colregs_flat = colregs.view(batch_size * n_agent, -1)

        # Conv1D feature extraction
        a = F.relu(self.conv1(x_flat))
        a = F.relu(self.conv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.fc1(a))

        # Concatenate all features
        a = torch.cat((a, goal_flat, self_state_flat, colregs_flat), dim=-1)
        a = F.relu(self.fc2(a))

        # 6D message (tanh for bounded output)
        msg = torch.tanh(self.msg_out(a))

        return msg.view(batch_size, n_agent, self.msg_dim)


class ControlActor(nn.Module):
    """
    자기 observation + 타 에이전트 메시지로 행동 결정
    CleanRL 방식: Normal distribution 사용, clamp 없음
    """
    def __init__(self, frames, msg_dim, action_size):
        super(ControlActor, self).__init__()
        self.frames = frames
        self.msg_dim = msg_dim
        self.action_size = action_size

        # Conv1D for radar feature extraction
        self.conv1 = nn.Conv1d(in_channels=frames, out_channels=32,
                               kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=2, padding=1)

        # Conv output: 90 * 32 = 2880
        self.fc1 = nn.Linear(90 * 32, 256)
        # 256 + goal(2) + self_state(4) + colregs(5) + others_msg(6) = 273
        self.fc2 = nn.Linear(256 + 2 + 4 + 5 + msg_dim, 128)
        self.fc3 = nn.Linear(128, 64)

        # Action mean output
        self.action_mean = nn.Linear(64, action_size)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.zero_()

        # Learnable log std (초기값 -0.5 → std ≈ 0.6, 적당한 탐색)
        self.action_logstd = nn.Parameter(torch.full((1, action_size), -0.5))

    def forward(self, x, goal, self_state, colregs, others_msg):
        """
        Args:
            x: [batch, n_agent, frames * STATE_SIZE]
            goal: [batch, n_agent, 2]
            self_state: [batch, n_agent, 4]
            colregs: [batch, n_agent, 5]
            others_msg: [batch, n_agent, msg_dim]
        Returns:
            action: [batch, n_agent, action_size]
            logprob: [batch, n_agent, 1]
            mean: [batch, n_agent, action_size]
        """
        batch_size, n_agent, _ = x.shape

        # Flatten for processing
        x_flat = x.view(batch_size * n_agent, self.frames, STATE_SIZE)
        goal_flat = goal.view(batch_size * n_agent, -1)
        self_state_flat = self_state.view(batch_size * n_agent, -1)
        colregs_flat = colregs.view(batch_size * n_agent, -1)
        others_msg_flat = others_msg.view(batch_size * n_agent, -1)

        # Conv1D feature extraction
        a = F.relu(self.conv1(x_flat))
        a = F.relu(self.conv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.fc1(a))

        # Concatenate all features
        a = torch.cat((a, goal_flat, self_state_flat, colregs_flat, others_msg_flat), dim=-1)
        a = torch.tanh(self.fc2(a))
        a = torch.tanh(self.fc3(a))

        # Action mean (clamp으로 안정성 확보, tanh는 sample에 적용)
        action_mean = torch.clamp(self.action_mean(a), -3.0, 3.0)  # tanh(3) ≈ 0.995

        # Action std (clamp으로 안정성 확보: 0.1 ~ 1.0)
        action_logstd = torch.clamp(self.action_logstd, -2.3, 0.0)  # exp(-2.3)≈0.1, exp(0)=1
        action_logstd = action_logstd.expand(batch_size * n_agent, -1)
        action_std = torch.exp(action_logstd)

        # Sample action using Normal distribution
        dist = Normal(action_mean, action_std)
        action_raw = dist.sample()

        # Squashed Gaussian: tanh로 [-1, 1] 범위로 제한 + log_prob 보정
        action = torch.tanh(action_raw)
        # tanh 변환의 log-det-jacobian 보정
        logprob = dist.log_prob(action_raw) - torch.log(1 - action.pow(2) + 1e-6)
        logprob = logprob.sum(dim=-1, keepdim=True)

        # Reshape back
        action = action.view(batch_size, n_agent, -1)
        logprob = logprob.view(batch_size, n_agent, -1)
        action_mean = action_mean.view(batch_size, n_agent, -1)

        return action, logprob, action_mean

    def get_logprob_entropy(self, x, goal, self_state, colregs, others_msg, action):
        """
        PPO 업데이트용: 주어진 action의 log_prob과 entropy 계산
        """
        batch_size, n_agent, _ = x.shape

        # Flatten for processing
        x_flat = x.view(batch_size * n_agent, self.frames, STATE_SIZE)
        goal_flat = goal.view(batch_size * n_agent, -1)
        self_state_flat = self_state.view(batch_size * n_agent, -1)
        colregs_flat = colregs.view(batch_size * n_agent, -1)
        others_msg_flat = others_msg.view(batch_size * n_agent, -1)
        action_flat = action.view(batch_size * n_agent, -1)

        # Conv1D feature extraction
        a = F.relu(self.conv1(x_flat))
        a = F.relu(self.conv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.fc1(a))

        # Concatenate all features
        a = torch.cat((a, goal_flat, self_state_flat, colregs_flat, others_msg_flat), dim=-1)
        a = torch.tanh(self.fc2(a))
        a = torch.tanh(self.fc3(a))

        # Action mean (clamp으로 안정성 확보)
        action_mean = torch.clamp(self.action_mean(a), -3.0, 3.0)

        # Action std (clamp으로 안정성 확보)
        action_logstd = torch.clamp(self.action_logstd, -2.3, 0.0)
        action_logstd = action_logstd.expand(batch_size * n_agent, -1)
        action_std = torch.exp(action_logstd)

        # Squashed Gaussian: action은 이미 tanh 적용된 값이므로 역변환 필요
        # action_raw = arctanh(action) = 0.5 * log((1+action)/(1-action))
        action_clamped = torch.clamp(action_flat, -0.999, 0.999)  # arctanh 안정성
        action_raw = 0.5 * torch.log((1 + action_clamped) / (1 - action_clamped))

        # Calculate log_prob with tanh correction
        dist = Normal(action_mean, action_std)
        logprob = dist.log_prob(action_raw) - torch.log(1 - action_clamped.pow(2) + 1e-6)
        logprob = logprob.sum(dim=-1, keepdim=True)

        # Squashed Gaussian entropy: H(Y) = H(X) + E[log|det(dg/dx)|]
        # tanh 변환의 Jacobian 보정 포함
        gaussian_entropy = dist.entropy().sum(dim=-1)
        # action_raw 기반 보정 (log(1 - tanh(x)^2) 항)
        squash_correction = torch.log(1 - action_clamped.pow(2) + 1e-6).sum(dim=-1)
        entropy = (gaussian_entropy + squash_correction).mean()

        # Reshape
        logprob = logprob.view(batch_size, n_agent, -1)
        action_mean = action_mean.view(batch_size, n_agent, -1)

        return logprob, entropy, action_mean


class Critic(nn.Module):
    """
    Value function estimator
    """
    def __init__(self, frames):
        super(Critic, self).__init__()
        self.frames = frames

        # Conv1D for radar feature extraction
        self.conv1 = nn.Conv1d(in_channels=frames, out_channels=32,
                               kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(90 * 32, 256)
        # 256 + goal(2) + self_state(4) + colregs(5) = 267
        self.fc2 = nn.Linear(256 + 2 + 4 + 5, 128)
        self.value_out = nn.Linear(128, 1)

    def forward(self, x, goal, self_state, colregs):
        """
        Args:
            x: [batch, n_agent, frames * STATE_SIZE]
            goal: [batch, n_agent, 2]
            self_state: [batch, n_agent, 4]
            colregs: [batch, n_agent, 5]
        Returns:
            value: [batch, n_agent, 1]
        """
        batch_size, n_agent, _ = x.shape

        # Flatten
        x_flat = x.view(batch_size * n_agent, self.frames, STATE_SIZE)
        goal_flat = goal.view(batch_size * n_agent, -1)
        self_state_flat = self_state.view(batch_size * n_agent, -1)
        colregs_flat = colregs.view(batch_size * n_agent, -1)

        # Conv
        v = F.relu(self.conv1(x_flat))
        v = F.relu(self.conv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.fc1(v))

        # Concat and output
        v = torch.cat((v, goal_flat, self_state_flat, colregs_flat), dim=-1)
        v = F.relu(self.fc2(v))
        v = self.value_out(v)

        return v.view(batch_size, n_agent, 1)


class COLREGsClassifier(nn.Module):
    """
    Auxiliary task: radar → COLREGs situation prediction
    """
    def __init__(self, frames):
        super(COLREGsClassifier, self).__init__()
        self.frames = frames

        self.conv1 = nn.Conv1d(in_channels=frames, out_channels=32,
                               kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(90 * 32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, 5)

    def forward(self, x):
        """
        Args:
            x: [batch, n_agent, frames * STATE_SIZE]
        Returns:
            pred: [batch, n_agent, 5] - raw logits (F.cross_entropy가 내부적으로 softmax 적용)
        """
        batch_size, n_agent, _ = x.shape

        x_flat = x.view(batch_size * n_agent, self.frames, STATE_SIZE)

        c = F.relu(self.conv1(x_flat))
        c = F.relu(self.conv2(c))
        c = c.view(c.shape[0], -1)
        c = F.relu(self.fc1(c))
        c = F.relu(self.fc2(c))
        c = self.classifier(c)

        return c.view(batch_size, n_agent, 5)


class CNNPolicy(nn.Module):
    """
    전체 정책 네트워크 (메시지 교환 기반)

    흐름:
    1. 모든 에이전트의 obs → MessageActor → 각자의 6D 메시지
    2. 메시지 합계 계산 → 자신 제외 = 이웃 메시지 합
    3. 자기 obs 전체 + others_msg → ControlActor → 행동
    4. Critic → 가치 추정
    5. COLREGsClassifier → 상황 예측 (auxiliary task)
    """
    def __init__(self, msg_dim, action_size, frames):
        super(CNNPolicy, self).__init__()
        self.frames = frames
        self.msg_dim = msg_dim
        self.action_size = action_size

        # Sub-networks
        self.msg_actor = MessageActor(frames, msg_dim)
        self.ctr_actor = ControlActor(frames, msg_dim, action_size)
        self.critic = Critic(frames)
        self.colregs_classifier = COLREGsClassifier(frames)

    def _get_others_msg(self, msg, comm_partners=None, agent_id_list=None):
        """메시지 교환 로직"""
        batch_size, n_agent, _ = msg.shape

        if not USE_COMMUNICATION:
            # Phase 1: 통신 비활성화
            return torch.zeros_like(msg)

        if comm_partners is not None and agent_id_list is not None:
            # 통신 범위 기반 메시지 교환
            others_msg = torch.zeros_like(msg)
            id_to_idx = {aid: idx for idx, aid in enumerate(agent_id_list)}

            for i, agent_id in enumerate(agent_id_list):
                partners = comm_partners.get(agent_id, [])
                if partners:
                    partner_indices = [id_to_idx[p] for p in partners if p in id_to_idx]
                    if partner_indices:
                        others_msg[0, i, :] = msg[0, partner_indices, :].sum(dim=0)
            return others_msg

        # 기본: 모든 에이전트와 통신
        msg_sum = msg.sum(dim=1, keepdim=True).repeat(1, n_agent, 1)
        return msg_sum - msg

    def forward(self, x, goal, self_state, colregs,
                return_msg=False, comm_partners=None, agent_id_list=None):
        """
        Forward pass

        Args:
            x: [batch, n_agent, frames * STATE_SIZE]
            goal: [batch, n_agent, 2]
            self_state: [batch, n_agent, 4]
            colregs: [batch, n_agent, 5]

        Returns:
            value, action, logprob, mean, colregs_pred
        """
        # Handle 2D input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            goal = goal.unsqueeze(1)
            self_state = self_state.unsqueeze(1)
            colregs = colregs.unsqueeze(1)

        # 1. Generate messages
        msg = self.msg_actor(x, goal, self_state, colregs)

        # 2. Message exchange
        others_msg = self._get_others_msg(msg, comm_partners, agent_id_list)

        # 3. Get action
        action, logprob, mean = self.ctr_actor(x, goal, self_state, colregs, others_msg)

        # 4. Get value
        value = self.critic(x, goal, self_state, colregs)

        # 5. COLREGs prediction (auxiliary)
        colregs_pred = self.colregs_classifier(x)

        if return_msg:
            return value, action, logprob, mean, colregs_pred, msg, others_msg

        return value, action, logprob, mean, colregs_pred

    def evaluate_actions(self, x, goal, self_state, colregs, others_msg, action):
        """
        PPO 업데이트용: 주어진 action의 가치와 확률 평가
        ★ others_msg를 직접 받아서 사용 (forward와 일치) ★
        """
        # Handle 2D input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            goal = goal.unsqueeze(1)
            self_state = self_state.unsqueeze(1)
            colregs = colregs.unsqueeze(1)
            others_msg = others_msg.unsqueeze(1)
            action = action.unsqueeze(1)

        # 1. Get logprob and entropy (저장된 others_msg 직접 사용)
        logprob, entropy, mean = self.ctr_actor.get_logprob_entropy(
            x, goal, self_state, colregs, others_msg, action
        )

        # 2. Get value
        value = self.critic(x, goal, self_state, colregs)

        # 3. COLREGs prediction
        colregs_pred = self.colregs_classifier(x)

        return value, logprob, entropy, colregs_pred
