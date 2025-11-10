import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *
from functions import log_normal_density

class MessageActor(nn.Module):
    """에이전트간 통신을 위한 메시지를 생성하는 네트워크 (COLREGs 인식)"""
    def __init__(self, frames, msg_action_space, max_neighbors):
        super(MessageActor, self).__init__()
        self.frames = frames
        self.max_neighbors = max_neighbors
        self.logstd = nn.Parameter(torch.zeros(msg_action_space))
        # 상태를 처리하는 완전연결 레이어 (STATE_SIZE * FRAMES 차원 입력)
        self.act_fc1 = nn.Linear(STATE_SIZE * FRAMES, 256)
        # 256 + goal(2) + speed(2) + colregs(4)
        self.act_fc2 = nn.Linear(256+2+2+4, 128)

        # 메시지 생성을 위한 출력 레이어
        self.actor = nn.Linear(128, msg_action_space)  # 6차원 메시지

    def forward(self, x, goal, speed, colregs_situations):
        """순전파 함수 (COLREGs 정보 포함)"""
        # STATE_SIZE * FRAMES 입력 처리
        x = x.view(-1, STATE_SIZE * FRAMES)
        batch_size = x.shape[0]  # 배치 크기는 view 이후 첫 번째 차원

        a = F.relu(self.act_fc1(x))
        a = a.view(batch_size, -1, 256)

        # goal과 speed 차원 맞추기: [batch, 2] -> [batch, 1, 2]
        if len(goal.shape) == 2:
            goal = goal.unsqueeze(1)
        if len(speed.shape) == 2:
            speed = speed.unsqueeze(1)

        # COLREGs 차원 맞추기: [batch, 4] -> [batch, 1, 4]
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

        msg = torch.tanh(self.actor(a))  # [-1,1] 범위

        logstd = self.logstd.expand_as(msg)
        std = torch.exp(logstd)
        msg = torch.normal(msg, std)

        logprob = log_normal_density(msg, msg, std=std, log_std=logstd)
        return msg, logprob, msg

class ControlActor(nn.Module):
    """행동을 결정하는 네트워크 (COLREGs 인식)"""
    def __init__(self, frames, msg_action_space, ctr_action_space, n_agent):
        super(ControlActor, self).__init__()
        self.frames = frames
        self.n_agent = n_agent

        # 메시지를 받아서 행동 생성
        self.act_fc1 = nn.Linear(msg_action_space+msg_action_space, 64)  # 자신 + 이웃 메시지
        # 64 + goal(2) + speed(2) + colregs(4)
        self.act_fc2 = nn.Linear(64+2+2+4, 128)
        self.mu = nn.Linear(128, ctr_action_space)
        self.mu.weight.data.mul_(0.1)
        self.logstd = nn.Parameter(torch.zeros(ctr_action_space))

    def forward(self, x, goal, speed, colregs_situations):
        """순전파 함수 (COLREGs 정보 포함)
        Args:
            x: 결합된 메시지 (self_msg + neighbor_sum) [batch, 1, msg_action_space*2]
            goal: 목표 정보 [batch, 2] or [batch, 1, 2]
            speed: 속도 정보 [batch, 2] or [batch, 1, 2]
            colregs_situations: COLREGs 상황 [batch, 4] or [batch, 1, 4]
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

        # COLREGs 차원 맞추기: [batch, 4] -> [batch, 1, 4]
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

        # 크리틱(가치 평가) 네트워크 (STATE_SIZE * FRAMES 입력)
        self.crt_fc1 = nn.Linear(STATE_SIZE * FRAMES, 256)
        self.crt_fc2 = nn.Linear(256+2+2+4, 128)  # 256 + goal(2) + speed(2) + colregs(4)
        self.critic = nn.Linear(128, 1)

        # COLREGs 상황 분류기 (Auxiliary Task)
        self.colregs_fc1 = nn.Linear(STATE_SIZE * FRAMES, 128)
        self.colregs_fc2 = nn.Linear(128, 64)
        self.colregs_classifier = nn.Linear(64, 4)  # 4가지 COLREGs 상황

    def forward(self, x, goal, speed, colregs_situations=None, neighbor_obs=None, neighbor_mask=None):
        """순전파 함수 (COLREGs 정보 포함)"""
        if len(x.shape) < 3:
            x = x.unsqueeze(0)

        # COLREGs 정보가 없으면 기본값 생성
        if colregs_situations is None:
            colregs_situations = torch.zeros(x.shape[0], 4, device=x.device)

        # COLREGs 상황 예측 (Auxiliary Task)
        colregs_pred = F.relu(self.colregs_fc1(x.view(-1, STATE_SIZE * FRAMES)))
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
                    valid_obs = neighbor_obs[valid_indices, i]
                    # 프레임 스택을 사용하므로 이웃 관측도 간단히 동일 프레임 반복으로 스택
                    if valid_obs.dim() == 2:
                        valid_obs = valid_obs.repeat(1, FRAMES)

                    # Goal과 speed는 항상 배치의 유효한 인덱스에서 추출
                    valid_goal = goal[valid_indices] if goal.shape[0] > 1 else goal
                    valid_speed = speed[valid_indices] if speed.shape[0] > 1 else speed
                    valid_colregs = colregs_situations[valid_indices] if colregs_situations.shape[0] > 1 else colregs_situations

                    # 이웃 메시지 생성 (COLREGs 정보 포함)
                    neighbor_msg, _, _ = self.msg_actor(valid_obs, valid_goal, valid_speed, valid_colregs)

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

        # 가치 평가 (STATE_SIZE * FRAMES 입력, COLREGs 정보 포함)
        x = x.view(-1, STATE_SIZE * FRAMES)
        v = F.relu(self.crt_fc1(x))
        v = v.view(-1, 1, 256)
        batch_size_v = v.shape[0]

        # goal과 speed 차원 맞추기: [batch, 2] -> [batch, 1, 2]
        goal_v = goal.unsqueeze(1) if len(goal.shape) == 2 else goal
        speed_v = speed.unsqueeze(1) if len(speed.shape) == 2 else speed

        # COLREGs 차원 맞추기: [batch, 4] -> [batch, 1, 4]
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

    def evaluate_actions(self, x, goal, speed, action, colregs_situations=None):
        """주어진 행동의 가치와 확률을 평가하는 함수 (COLREGs 정보 포함)"""
        v, _, _, mean, colregs_pred = self.forward(x, goal, speed, colregs_situations)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)

        logprob = log_normal_density(action, mean, log_std=logstd, std=std)

        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()

        return v, logprob, dist_entropy, colregs_pred