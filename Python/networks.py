import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *
from functions import log_normal_density

class MessageActor(nn.Module):
    """GitHub 방식: Conv1D + sigmoid + tanh로 12D 메시지 생성"""
    def __init__(self, frames, msg_action_space, max_neighbors):
        super(MessageActor, self).__init__()
        self.frames = frames
        self.max_neighbors = max_neighbors
        self.logstd = nn.Parameter(torch.zeros(2 * msg_action_space))  # 12D

        # Conv1D layers for radar feature extraction (GitHub 방식)
        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32,
                                      kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32,
                                      kernel_size=3, stride=2, padding=1)

        # Conv output size 계산: (360 - 5 + 2) / 2 + 1 = 179.5 ≈ 179
        # 다시: (179 - 3 + 2) / 2 + 1 = 90
        self.act_fc1 = nn.Linear(90*32, 256)  # 2880 → 256

        # 256 + goal(2) + speed(2) + colregs(5)
        self.act_fc2 = nn.Linear(256+2+2+5, 128)

        # GitHub 방식: 2개 출력 (sigmoid, tanh)
        self.actor1 = nn.Linear(128, msg_action_space)  # 6D (sigmoid)
        self.actor2 = nn.Linear(128, msg_action_space)  # 6D (tanh)

    def forward(self, x, goal, speed, colregs_situations):
        """순전파 함수 (COLREGs 정보 포함)"""
        # Reshape to [batch, frames, STATE_SIZE] for Conv1D
        x = x.view(-1, self.frames, STATE_SIZE)
        batch_size = x.shape[0]

        # Conv1D feature extraction
        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a))

        # Flatten
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))
        a = a.view(batch_size, -1, 256)

        # goal과 speed 차원 맞추기: [batch, 2] -> [batch, 1, 2]
        if len(goal.shape) == 2:
            goal = goal.unsqueeze(1)
        if len(speed.shape) == 2:
            speed = speed.unsqueeze(1)

        # COLREGs 차원 맞추기: [batch, 5] -> [batch, 1, 5]
        if len(colregs_situations.shape) == 2:
            colregs_situations = colregs_situations.unsqueeze(1)

        # 배치 크기 맞추기
        if goal.shape[0] != batch_size:
            goal = goal.expand(batch_size, -1, -1)
        if speed.shape[0] != batch_size:
            speed = speed.expand(batch_size, -1, -1)
        if colregs_situations.shape[0] != batch_size:
            colregs_situations = colregs_situations.expand(batch_size, -1, -1)

        # COLREGs 정보 포함하여 결합
        a = torch.cat((a, goal, speed, colregs_situations), dim=-1)
        a = F.relu(self.act_fc2(a))

        # GitHub 방식: sigmoid + tanh
        mean1 = torch.sigmoid(self.actor1(a))  # (batch, 1, 6)
        mean2 = torch.tanh(self.actor2(a))     # (batch, 1, 6)
        mean = torch.cat((mean1, mean2), dim=-1)  # (batch, 1, 12)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        msg = torch.normal(mean, std)

        logprob = log_normal_density(msg, mean, std=std, log_std=logstd)
        return msg, logprob, mean

class ControlActor(nn.Module):
    """행동을 결정하는 네트워크 (GitHub 방식: 24D 입력)"""
    def __init__(self, frames, msg_action_space, ctr_action_space, n_agent):
        super(ControlActor, self).__init__()
        self.frames = frames
        self.n_agent = n_agent

        # GitHub 방식: 12D (self) + 12D (neighbors) = 24D
        self.act_fc1 = nn.Linear(2 * msg_action_space * 2, 64)  # 24D input
        # 64 + goal(2) + speed(2) + colregs(5)
        self.act_fc2 = nn.Linear(64+2+2+5, 128)
        self.mu = nn.Linear(128, ctr_action_space)
        self.mu.weight.data.mul_(0.1)
        self.logstd = nn.Parameter(torch.zeros(ctr_action_space))

    def forward(self, x, goal, speed, colregs_situations):
        """순전파 함수 (COLREGs 정보 포함)
        Args:
            x: 결합된 메시지 (self_msg + neighbor_sum) [batch, 1, msg_action_space*2]
            goal: 목표 정보 [batch, 2] or [batch, 1, 2]
            speed: 속도 정보 [batch, 2] or [batch, 1, 2]
            colregs_situations: COLREGs 상황 [batch, 5] or [batch, 1, 5]
        """
        # 메시지 처리
        act = self.act_fc1(x)
        act = act.view(-1, 1, 64)
        batch_size = act.shape[0]

        # goal과 speed 차원 맞추기: [batch, 2] -> [batch, 1, 2]
        if len(goal.shape) == 2:
            goal = goal.unsqueeze(1)
        if len(speed.shape) == 2:
            speed = speed.unsqueeze(1)

        # COLREGs 차원 맞추기: [batch, 5] -> [batch, 1, 5]
        if len(colregs_situations.shape) == 2:
            colregs_situations = colregs_situations.unsqueeze(1)

        # 배치 크기 맞추기
        if goal.shape[0] != batch_size:
            goal = goal.expand(batch_size, -1, -1)
        if speed.shape[0] != batch_size:
            speed = speed.expand(batch_size, -1, -1)
        if colregs_situations.shape[0] != batch_size:
            colregs_situations = colregs_situations.expand(batch_size, -1, -1)

        # Goal, speed, COLREGs 결합
        act = torch.cat((act, goal, speed, colregs_situations), dim=-1)
        act = torch.tanh(act)
        act = self.act_fc2(act)
        act = torch.tanh(act)
        mean = self.mu(act)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return action, logprob, mean

class CNNPolicy(nn.Module):
    """전체 정책을 관리하는 네트워크 (COLREGs 직접 학습)"""
    def __init__(self, msg_action_space, ctr_action_space, frames, max_neighbors):
        super(CNNPolicy, self).__init__()
        self.frames = frames
        self.max_neighbors = max_neighbors

        # 메시지 액터와 컨트롤 액터 초기화
        self.msg_actor = MessageActor(frames, msg_action_space, max_neighbors)
        self.ctr_actor = ControlActor(frames, msg_action_space, ctr_action_space, max_neighbors)
        self.logstd = nn.Parameter(torch.zeros(ctr_action_space))

        # 크리틱(가치 평가) 네트워크 (Conv1D 사용)
        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32,
                                      kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32,
                                      kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(90*32, 256)  # 2880 → 256
        self.crt_fc2 = nn.Linear(256+2+2+5, 128)  # 256 + goal(2) + speed(2) + colregs(5)
        self.critic = nn.Linear(128, 1)

        # COLREGs 상황 분류기 (Auxiliary Task) - Conv1D 사용
        self.colregs_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32,
                                         kernel_size=5, stride=2, padding=1)
        self.colregs_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32,
                                         kernel_size=3, stride=2, padding=1)
        self.colregs_fc1 = nn.Linear(90*32, 128)
        self.colregs_fc2 = nn.Linear(128, 64)
        self.colregs_classifier = nn.Linear(64, 5)  # 5가지 COLREGs 상황 (None, HeadOn, CrossingStandOn, CrossingGiveWay, Overtaking)

    def forward(self, x, goal, speed, colregs_situations=None, neighbor_obs=None, neighbor_mask=None):
        """순전파 함수 (COLREGs 정보 포함)"""
        if len(x.shape) < 3:
            x = x.unsqueeze(0)

        # COLREGs 정보가 없으면 기본값 생성
        if colregs_situations is None:
            colregs_situations = torch.zeros(x.shape[0], 5, device=x.device)

        # COLREGs 상황 예측 (Auxiliary Task with Conv1D)
        x_colregs = x.view(-1, self.frames, STATE_SIZE)
        c = F.relu(self.colregs_fea_cv1(x_colregs))
        c = F.relu(self.colregs_fea_cv2(c))
        c = c.view(c.shape[0], -1)
        colregs_pred = F.relu(self.colregs_fc1(c))
        colregs_pred = F.relu(self.colregs_fc2(colregs_pred))
        colregs_pred = torch.softmax(self.colregs_classifier(colregs_pred), dim=-1)

        # 자신의 메시지 생성 (COLREGs 정보 포함)
        self_msg, _, _ = self.msg_actor(x, goal, speed, colregs_situations)
        
        # 이웃이 없는 경우 처리
        if neighbor_obs is None or neighbor_mask is None:
            # 빈 이웃 메시지 생성 (0으로 채움)
            batch_size = x.shape[0]
            neighbor_msgs = torch.zeros(batch_size, self.max_neighbors, 
                                        self_msg.shape[-1], device=x.device)
            # 모든 이웃이 없음을 표시하는 마스크
            if neighbor_mask is None:
                neighbor_mask = torch.zeros(batch_size, self.max_neighbors, 
                                           device=x.device, dtype=torch.bool)
        else:
            # 이웃 메시지 생성
            batch_size = neighbor_obs.shape[0]
            neighbor_msgs = torch.zeros(batch_size, self.max_neighbors, 
                                        self_msg.shape[-1], device=x.device)
            
            # 각 이웃의 observation으로부터 메시지 생성
            for i in range(self.max_neighbors):
                # 마스크가 True인 이웃만 처리
                valid_neighbors = neighbor_mask[:, i]
                if valid_neighbors.any():
                    valid_indices = torch.where(valid_neighbors)[0]
                    valid_obs = neighbor_obs[valid_indices, i]  # [N, 371]

                    # neighbor_obs 분해: [360 radar + 2 goal + 2 speed + 5 colregs + 1 heading + 1 rudder] = 371D
                    neighbor_radar = valid_obs[:, :360]  # [N, 360]
                    neighbor_goal = valid_obs[:, 360:362]  # [N, 2]
                    neighbor_speed = valid_obs[:, 362:364]  # [N, 2]
                    neighbor_colregs = valid_obs[:, 364:369]  # [N, 5]
                    # heading과 rudder는 메시지 생성에 직접 사용 안 함 (이미 speed/colregs에 정보 포함)

                    # Frame stacking: radar를 FRAMES번 반복
                    neighbor_radar_stacked = neighbor_radar.repeat(1, self.frames)  # [N, 360×3=1080]

                    # 이웃 메시지 생성 (neighbor의 radar, goal, speed, colregs 사용)
                    neighbor_msg, _, _ = self.msg_actor(
                        neighbor_radar_stacked,
                        neighbor_goal,
                        neighbor_speed,
                        neighbor_colregs
                    )

                    # 생성된 메시지 저장
                    for j, idx in enumerate(valid_indices):
                        neighbor_msgs[idx, i] = neighbor_msg[j, 0]
        
        # 이웃 메시지 집계 (마스크 적용)
        masked_msgs = neighbor_msgs * neighbor_mask.unsqueeze(-1).float()
        neighbor_sum = masked_msgs.sum(dim=1, keepdim=True)
        
        # 자신과 이웃 메시지 결합
        ctr_input = torch.cat((self_msg, neighbor_sum), 2)

        # 행동 생성 (COLREGs 정보 포함)
        action, logprob, mean = self.ctr_actor(ctr_input, goal, speed, colregs_situations)

        # 가치 평가 (Conv1D 사용, COLREGs 정보 포함)
        x_critic = x.view(-1, self.frames, STATE_SIZE)
        v = F.relu(self.crt_fea_cv1(x_critic))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v = v.view(-1, 1, 256)
        batch_size_v = v.shape[0]

        # goal과 speed 차원 맞추기: [batch, 2] -> [batch, 1, 2]
        goal_v = goal.unsqueeze(1) if len(goal.shape) == 2 else goal
        speed_v = speed.unsqueeze(1) if len(speed.shape) == 2 else speed

        # COLREGs 차원 맞추기: [batch, 5] -> [batch, 1, 5]
        colregs_v = colregs_situations.unsqueeze(1) if len(colregs_situations.shape) == 2 else colregs_situations

        # 배치 크기 맞추기
        if goal_v.shape[0] != batch_size_v:
            goal_v = goal_v.expand(batch_size_v, -1, -1)
        if speed_v.shape[0] != batch_size_v:
            speed_v = speed_v.expand(batch_size_v, -1, -1)
        if colregs_v.shape[0] != batch_size_v:
            colregs_v = colregs_v.expand(batch_size_v, -1, -1)

        v = torch.cat((v, goal_v, speed_v, colregs_v), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        # COLREGs 예측도 함께 반환
        return v, action, logprob, mean, colregs_pred

    def critic_forward(self, x, goal, speed, colregs_situations):
        """Critic 전용 forward (value estimation)"""
        x = x.view(-1, self.frames, STATE_SIZE)
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v = v.view(-1, 1, 256)
        batch_size = v.shape[0]

        # Goal, speed, COLREGs 차원 맞추기
        if len(goal.shape) == 2:
            goal = goal.unsqueeze(1)
        if len(speed.shape) == 2:
            speed = speed.unsqueeze(1)
        if len(colregs_situations.shape) == 2:
            colregs_situations = colregs_situations.unsqueeze(1)

        # 배치 크기 맞추기
        if goal.shape[0] != batch_size:
            goal = goal.expand(batch_size, -1, -1)
        if speed.shape[0] != batch_size:
            speed = speed.expand(batch_size, -1, -1)
        if colregs_situations.shape[0] != batch_size:
            colregs_situations = colregs_situations.expand(batch_size, -1, -1)

        v = torch.cat((v, goal, speed, colregs_situations), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)
        return v

    def colregs_forward(self, x):
        """COLREGs classifier 전용 forward (Auxiliary task)"""
        x = x.view(-1, self.frames, STATE_SIZE)
        c = F.relu(self.colregs_fea_cv1(x))
        c = F.relu(self.colregs_fea_cv2(c))
        c = c.view(c.shape[0], -1)
        c = F.relu(self.colregs_fc1(c))
        c = F.relu(self.colregs_fc2(c))
        c = torch.softmax(self.colregs_classifier(c), dim=-1)
        return c

    def evaluate_actions(self, x, goal, speed, action, colregs_situations=None):
        """주어진 행동의 가치와 확률을 평가하는 함수 (COLREGs 정보 포함)"""
        v, _, _, mean, colregs_pred = self.forward(x, goal, speed, colregs_situations)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)

        logprob = log_normal_density(action, mean, log_std=logstd, std=std)

        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()

        return v, logprob, dist_entropy, colregs_pred