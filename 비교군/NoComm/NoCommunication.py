import os
import torch
import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from Python.config import *
import matplotlib.pyplot as plt
import torch.onnx
from datetime import datetime

class PPOMemory:
    """PPO 학습을 위한 메모리 버퍼"""
    def __init__(self):
        self.states = []
        self.state_history = np.zeros((FRAMES, STATE_SIZE))  # 상태 히스토리 추가
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.dones = []
        
    def add(self, state, action, reward, value, logprob, done):
        # 상태 히스토리 업데이트
        self.state_history = np.roll(self.state_history, -1, axis=0)
        self.state_history[-1] = state
        
        # 스택된 상태 저장
        stacked_state = self.state_history.flatten()
        self.states.append(stacked_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.logprobs.append(logprob)
        self.dones.append(done)
        
    def clear(self):
        self.__init__()
        
    def get_batch(self):
        states = torch.FloatTensor(np.array(self.states)).to(DEVICE)
        actions = torch.FloatTensor(np.array(self.actions)).to(DEVICE)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(DEVICE)
        values = torch.FloatTensor(np.array(self.values)).to(DEVICE)
        logprobs = torch.FloatTensor(np.array(self.logprobs)).to(DEVICE)
        dones = torch.FloatTensor(np.array(self.dones)).to(DEVICE)
        return states, actions, rewards, values, logprobs, dones

class PPOPolicy(torch.nn.Module):
    """통신 없는 기본 PPO 정책 네트워크"""
    def __init__(self, obs_size, action_size):
        super(PPOPolicy, self).__init__()
        
        # 관측값 크기를 프레임 수만큼 곱해서 확장
        self.stacked_obs_size = obs_size * FRAMES
        
        # 관측값을 처리하는 신경망
        self.obs_net = torch.nn.Sequential(
            torch.nn.Linear(self.stacked_obs_size, 512),  # 입력 크기 변경
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU()
        )
        
        # 액터 (정책) 네트워크
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_size)
        )
        
        # 크리틱 (가치) 네트워크
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        
        # 행동의 표준편차 (학습 가능한 파라미터)
        self.logstd = torch.nn.Parameter(torch.zeros(action_size))

    def forward(self, obs):
        features = self.obs_net(obs)
        
        # 액터: 행동 평균과 로그 확률 계산
        action_mean = self.actor(features)
        action_std = torch.exp(self.logstd)
        
        # 크리틱: 상태 가치 계산
        value = self.critic(features)
        
        # 정규분포에서 행동 샘플링
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        return value, action, action_logprob, action_mean

    def evaluate_actions(self, obs, actions):
        features = self.obs_net(obs)
        
        action_mean = self.actor(features)
        action_std = torch.exp(self.logstd)
        value = self.critic(features)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(actions).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        return value, action_logprobs, dist_entropy

def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """GAE(Generalized Advantage Estimation) 계산"""
    advantages = []
    gae = 0
    
    for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_value * (1 - done) - v
        gae = delta + gamma * lam * (1 - done) * gae
        advantages.insert(0, gae)
        next_value = v
        
    advantages = torch.FloatTensor(advantages).to(DEVICE)
    returns = advantages + torch.FloatTensor(values).to(DEVICE)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return returns, advantages

def update_policy(policy, optimizer, memory, clip_param=0.2, c1=1, c2=0.01):
    """PPO 정책 업데이트"""
    states, actions, rewards, values, old_logprobs, dones = memory.get_batch()
    
    # GAE 계산
    with torch.no_grad():
        next_value = policy.critic(policy.obs_net(states[-1:])).squeeze()
    returns, advantages = compute_gae(rewards, values, dones, next_value, 
                                    DISCOUNT_FACTOR, 0.95)
    
    # 미니배치로 여러 번 업데이트
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    
    for _ in range(N_EPOCH):
        # 현재 정책으로 행동 재평가
        new_values, new_logprobs, dist_entropy = policy.evaluate_actions(states, actions)
        
        # 정책 비율 계산
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # PPO 클리핑
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 가치 함수 손실
        value_loss = F.mse_loss(new_values.squeeze(), returns)
        
        # 전체 손실 계산
        loss = policy_loss + c1 * value_loss - c2 * dist_entropy.mean()
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += dist_entropy.mean().item()
    
    return total_policy_loss / N_EPOCH, total_value_loss / N_EPOCH, total_entropy / N_EPOCH

def save_model_and_metrics(policy, episode, metrics, save_path, writer):
    """모델과 학습 메트릭을 저장하는 함수"""
    # 1. PyTorch 모델 저장
    torch.save(policy.state_dict(), 
              os.path.join(save_path, f'policy_episode_{episode}.pth'))
    
    # 2. ONNX 변환 및 저장
    dummy_input = torch.randn(1, policy.stacked_obs_size).to(DEVICE)
    torch.onnx.export(policy,
                     dummy_input,
                     os.path.join(save_path, f'policy_episode_{episode}.onnx'),
                     verbose=True,
                     input_names=['input'],
                     output_names=['value', 'action', 'action_logprob', 'action_mean'])
    
    # 3. 추가 메트릭 로깅
    writer.add_scalar('Training/Episode_Length', metrics['episode_length'], episode)
    writer.add_scalar('Training/Average_Value', metrics['avg_value'], episode)
    writer.add_scalar('Training/Policy_Gradient_Norm', metrics['policy_grad_norm'], episode)
    writer.add_scalar('Training/Average_Action', metrics['avg_action'], episode)
    writer.add_scalar('Training/Action_STD', metrics['action_std'], episode)
    
    # 4. 학습 커브 플로팅
    if hasattr(save_model_and_metrics, 'rewards'):
        save_model_and_metrics.rewards.append(metrics['episode_reward'])
        plt.figure(figsize=(10, 5))
        plt.plot(save_model_and_metrics.rewards)
        plt.title('Training Rewards over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(save_path, 'reward_curve.png'))
        plt.close()

def main():
    # 환경 설정
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name=None,  # 에디터에서 실행
        side_channels=[channel],
        worker_id=1,
        base_port=5004
    )
    channel.set_configuration_parameters(time_scale=TIME_SCALE)
    
    # 환경 정보 얻기
    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    
    # VesselAgent의 관측/행동 공간 크기 계산
    single_obs_size = 372  # 360(레이더) + 4(상태) + 1(타각) + 3(목표) + 4(COLREGs) + 1(위험도)
    obs_size = single_obs_size * FRAMES  # 프레임 수만큼 곱해서 확장
    action_size = 2  # 타각, 추진력
    
    # 정책 네트워크와 옵티마이저 초기화
    policy = PPOPolicy(obs_size, action_size).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    memory = PPOMemory()
    
    # 텐서보드 설정
    writer = SummaryWriter(log_dir=os.path.join(SAVE_PATH, 'logs'))
    
    # 보상 히스토리 초기화
    save_model_and_metrics.rewards = []
    
    # 학습 시작 시간 기록
    start_time = datetime.now()
    
    for episode in range(NUM_EPISODES):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        episode_rewards = {}
        episode_actions = []
        episode_values = []
        
        # 에피소드 실행
        for step in range(MAX_STEPS):
            # 활성 에이전트들의 행동 결정
            active_agents = decision_steps.agent_id
            if len(active_agents) == 0:
                break
                
            # 배치로 상태 처리
            batch_obs = torch.FloatTensor(decision_steps.obs[0]).to(DEVICE)
            
            # 정책 네트워크로 행동 선택
            with torch.no_grad():
                values, actions, logprobs, _ = policy(batch_obs)
            
            # 행동을 환경에 전달
            action_tuple = ActionTuple(continuous=actions.cpu().numpy())
            env.set_actions(behavior_name, action_tuple)
            env.step()
            
            # 다음 상태와 보상 받기
            next_decision_steps, next_terminal_steps = env.get_steps(behavior_name)
            
            # 보상 처리 및 기록
            for i, agent_id in enumerate(active_agents):
                if agent_id not in episode_rewards:
                    episode_rewards[agent_id] = 0
                
                # 에이전트가 여전히 활성 상태인지 확인
                if agent_id in next_decision_steps.agent_id:
                    reward = next_decision_steps.reward[agent_id]
                    done = False
                elif agent_id in next_terminal_steps.agent_id:
                    reward = next_terminal_steps.reward[agent_id]
                    done = True
                else:
                    continue
                
                episode_rewards[agent_id] += reward
                
                # 경험 저장
                memory.add(
                    decision_steps.obs[0][i],
                    actions[i].cpu().numpy(),
                    reward,
                    values[i].item(),
                    logprobs[i].item(),
                    done
                )
            
            decision_steps = next_decision_steps
            
            # 행동과 가치 저장
            episode_actions.extend(actions.cpu().numpy())
            episode_values.extend(values.cpu().numpy())
        
        # 에피소드 종료 후 메트릭 계산
        avg_reward = sum(episode_rewards.values()) / max(len(episode_rewards), 1)
        metrics = {
            'episode_reward': avg_reward,
            'episode_length': step,
            'avg_value': np.mean(episode_values),
            'policy_grad_norm': torch.nn.utils.clip_grad_norm_(
                policy.parameters(), MAX_GRAD_NORM).item(),
            'avg_action': np.mean(episode_actions),
            'action_std': np.std(episode_actions)
        }
        
        # 텐서보드에 기본 메트릭 기록
        writer.add_scalar('Reward/Episode', avg_reward, episode)
        writer.add_scalar('Training/Steps_Per_Episode', step, episode)
        
        # 주기적으로 모델과 메트릭 저장
        if episode % SAVE_INTERVAL == 0:
            save_model_and_metrics(policy, episode, metrics, SAVE_PATH, writer)
            
            # 학습 진행 상황 출력
            elapsed_time = datetime.now() - start_time
            print(f"Episode {episode}/{NUM_EPISODES}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Steps: {step}, "
                  f"Time: {elapsed_time}")
    
    # 학습 종료 후 최종 모델 저장
    save_model_and_metrics(policy, NUM_EPISODES, metrics, SAVE_PATH, writer)
    env.close()
    writer.close()

if __name__ == "__main__":
    main() 