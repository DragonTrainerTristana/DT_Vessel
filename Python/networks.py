import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *
from functions import log_normal_density

class MessageActor(nn.Module):
    """에이전트간 통신을 위한 메시지를 생성하는 네트워크"""
    def __init__(self, frames, msg_action_space, max_neighbors):
        super(MessageActor, self).__init__()
        self.frames = frames
        self.max_neighbors = max_neighbors
        self.logstd = nn.Parameter(torch.zeros(msg_action_space))

        # 상태를 처리하는 완전연결 레이어 (46차원 입력)
        self.act_fc1 = nn.Linear(46, 256)  # 46차원 입력
        self.act_fc2 = nn.Linear(256+2+2, 128)  # 256 + goal(2) + speed(2)
        
        # 메시지 생성을 위한 출력 레이어
        self.actor = nn.Linear(128, msg_action_space)  # 6차원 메시지

    def forward(self, x, goal, speed):
        """순전파 함수"""
        batch_size = x.shape[0] if len(x.shape) > 2 else 1
        
        # 46차원 입력 처리
        x = x.view(-1, 46)  # 46차원으로 변경
        a = F.relu(self.act_fc1(x))
        
        if batch_size > 1:
            a = a.view(batch_size, -1, 256)
        else:
            a = a.view(1, -1, 256)

        if len(goal.shape) < 3:
            goal = goal.unsqueeze(0)
        if len(speed.shape) < 3:
            speed = speed.unsqueeze(0)

        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))

        msg = torch.tanh(self.actor(a))  # [-1,1] 범위

        logstd = self.logstd.expand_as(msg)
        std = torch.exp(logstd)
        msg = torch.normal(msg, std)

        logprob = log_normal_density(msg, msg, std=std, log_std=logstd)
        return msg, logprob, msg

class ControlActor(nn.Module):
    """행동을 결정하는 네트워크"""
    def __init__(self, frames, msg_action_space, ctr_action_space, n_agent):
        super(ControlActor, self).__init__()
        self.frames = frames
        self.n_agent = n_agent
        
        # 관측 상태 처리를 위한 레이어 (46차원 입력)
        self.act_obs_fc1 = nn.Linear(46, 256)  # 46차원 입력
        self.act_obs_fc2 = nn.Linear(256+2+2, 128)  # 256 + goal(2) + speed(2)
        self.act_obs_fc3 = nn.Linear(128, msg_action_space)
        
        # 메시지와 상태를 결합하여 행동 생성
        self.act_fc1 = nn.Linear(msg_action_space+msg_action_space, 64)  # 자신 + 이웃 메시지
        self.act_fc2 = nn.Linear(64+2+2, 128)  # 64 + goal(2) + speed(2)
        self.mu = nn.Linear(128, ctr_action_space)
        self.mu.weight.data.mul_(0.1)
        self.logstd = nn.Parameter(torch.zeros(ctr_action_space))

    def forward(self, x, goal, speed, y):
        """순전파 함수"""
        # 관측 상태 처리 (46차원)
        y = y.view(-1, 46)  # 46차원으로 변경
        a = F.relu(self.act_obs_fc1(y))
        a = a.view(-1, 1, 256)  # 단일 에이전트

        if len(goal.shape) < 3:
            goal = goal.unsqueeze(0)
        if len(speed.shape) < 3:
            speed = speed.unsqueeze(0)

        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_obs_fc2(a))
        a = F.relu(self.act_obs_fc3(a))  # msg_action_space 차원

        # 메시지 결합 (자신의 메시지 + 이웃 메시지)
        x = torch.cat((a, x), dim=-1)  # msg_action_space + msg_action_space
        act = self.act_fc1(x)
        act = act.view(-1, 1, 64)  # 단일 에이전트

        act = torch.cat((act, goal, speed), dim=-1)
        act = F.tanh(act)
        act = self.act_fc2(act)
        act = F.tanh(act)
        mean = self.mu(act)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return action, logprob, mean

class CNNPolicy(nn.Module):
    """전체 정책을 관리하는 네트워크"""
    def __init__(self, msg_action_space, ctr_action_space, frames, max_neighbors):
        super(CNNPolicy, self).__init__()
        self.frames = frames
        self.max_neighbors = max_neighbors
        
        # 메시지 액터와 컨트롤 액터 초기화
        self.msg_actor = MessageActor(frames, msg_action_space, max_neighbors)
        self.ctr_actor = ControlActor(frames, msg_action_space, ctr_action_space, max_neighbors)
        self.logstd = nn.Parameter(torch.zeros(ctr_action_space))

        # 크리틱(가치 평가) 네트워크 (46차원 입력)
        self.crt_fc1 = nn.Linear(46, 256)  # 46차원 입력
        self.crt_fc2 = nn.Linear(256+2+2, 128)  # 256 + goal(2) + speed(2)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, goal, speed, neighbor_obs=None, neighbor_mask=None):
        """순전파 함수"""
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        
        # 자신의 메시지 생성
        self_msg, _, _ = self.msg_actor(x, goal, speed)
        
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
                    valid_goal = goal[valid_indices] if len(goal.shape) > 2 else goal
                    valid_speed = speed[valid_indices] if len(speed.shape) > 2 else speed
                    
                    # 이웃 메시지 생성
                    with torch.no_grad():  # 학습 시 메모리 효율성을 위해
                        neighbor_msg, _, _ = self.msg_actor(valid_obs, valid_goal, valid_speed)
                    
                    # 생성된 메시지 저장
                    for j, idx in enumerate(valid_indices):
                        neighbor_msgs[idx, i] = neighbor_msg[j, 0]
        
        # 이웃 메시지 집계 (마스크 적용)
        masked_msgs = neighbor_msgs * neighbor_mask.unsqueeze(-1).float()
        neighbor_sum = masked_msgs.sum(dim=1, keepdim=True)
        
        # 자신과 이웃 메시지 결합
        ctr_input = torch.cat((self_msg, neighbor_sum), 2)
        
        # 행동 생성
        action, logprob, mean = self.ctr_actor(ctr_input, goal, speed, x)
        
        # 가치 평가 (46차원 입력)
        x = x.view(-1, 46)  # 46차원으로 변경
        v = F.relu(self.crt_fc1(x))
        v = v.view(-1, 1, 256)  # 단일 에이전트

        if len(goal.shape) < 3:
            goal = goal.unsqueeze(0)
        if len(speed.shape) < 3:
            speed = speed.unsqueeze(0)

        v = torch.cat((v, goal, speed), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action):
        """주어진 행동의 가치와 확률을 평가하는 함수"""
        v, _, _, mean = self.forward(x, goal, speed)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        
        return v, logprob, dist_entropy