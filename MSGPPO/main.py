import os   
import torch
import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from config import *
from networks import CNNPolicy
from memory import Memory
from functions import calculate_returns

def main():
    # 환경 설정
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name="AirCombatRL", 
        side_channels=[channel],
        worker_id=1,
        base_port=5006
    )
    channel.set_configuration_parameters(time_scale=1.0)

    # 정책 네트워크와 옵티마이저 초기화
    # N_AGENT는 이제 최대 이웃 수를 의미
    policy = CNNPolicy(MSG_ACTION_SPACE, CONTINUOUS_ACTION_SIZE, FRAMES, N_AGENT).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    # 텐서보드 로깅 설정
    writer = SummaryWriter(log_dir=os.path.join(SAVE_PATH, 'logs'))

    # 학습 루프
    total_steps = 0
    memory = Memory()  # 새로운 메모리 구조 사용
    
    for episode in range(NUM_EPISODES):
        # 환경 초기화
        env.reset()
        memory.reset_for_new_episode()  # 새 에피소드 시작 시 메모리 초기화
        behavior_name = list(env.behavior_specs)[0]
        
        step = 0
        episode_rewards = {}  # 에이전트별 보상 추적
        
        while step < MAX_STEPS:
            # 환경에서 상태 정보 얻기
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            # 현재 활성화된 에이전트 처리
            for agent_id in decision_steps.agent_id:
                # 에이전트의 상태 정보
                state = decision_steps.obs[0][agent_id]  # 기본 관찰
                
                # 목표와 속도 정보 (obs의 일부로 포함되었다고 가정)
                if len(decision_steps.obs) > 1:
                    goal = decision_steps.obs[1][agent_id]
                    speed = decision_steps.obs[2][agent_id]
                else:
                    # 더 좋은 방법은 Unity에서 obs[0]에 모든 정보를 포함시키는 것
                    goal = np.zeros(2)
                    speed = np.zeros(2)
                
                # 이웃 정보 (Unity에서 제공)
                if len(decision_steps.obs) > 3:
                    neighbor_ids = decision_steps.obs[3][agent_id]  # 이웃 ID 배열
                    neighbor_obs = decision_steps.obs[4][agent_id]  # 이웃 관찰 배열
                else:
                    # 이웃 정보가 없는 경우
                    neighbor_ids = np.zeros(N_AGENT, dtype=np.int32) - 1  # -1은 이웃 없음
                    neighbor_obs = np.zeros((N_AGENT, FRAMES, 512))
                
                # 이웃 마스크 생성 (유효한 이웃만 True)
                neighbor_mask = torch.tensor(neighbor_ids >= 0, dtype=torch.bool).to(DEVICE)
                
                # 텐서 변환
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(DEVICE)
                speed_tensor = torch.FloatTensor(speed).unsqueeze(0).to(DEVICE)
                neighbor_obs_tensor = torch.FloatTensor(neighbor_obs).unsqueeze(0).to(DEVICE)
                
                # 행동 선택
                with torch.no_grad():
                    value, action, logprob, _ = policy(
                        state_tensor, goal_tensor, speed_tensor, 
                        neighbor_obs_tensor, neighbor_mask.unsqueeze(0)
                    )
                
                # 행동 저장
                if agent_id not in episode_rewards:
                    episode_rewards[agent_id] = 0
                
                # 보상 추적
                reward = decision_steps.reward[agent_id]
                episode_rewards[agent_id] += reward
                
                # 경험 저장
                memory.add_agent_experience(
                    agent_id,
                    state, goal, speed,
                    {'obs': neighbor_obs, 'mask': neighbor_mask.cpu().numpy()},
                    action.cpu().numpy()[0, 0],
                    reward,
                    False,  # 아직 종료 안됨
                    value.cpu().numpy()[0, 0],
                    logprob.cpu().numpy()[0, 0]
                )
            
            # 종료된 에이전트 처리
            for agent_id in terminal_steps.agent_id:
                # 이미 처리된 에이전트 건너뛰기
                if agent_id in decision_steps.agent_id:
                    continue
                    
                # 종료된 에이전트의 최종 보상
                reward = terminal_steps.reward[agent_id]
                
                # 에피소드 보상 업데이트
                if agent_id in episode_rewards:
                    episode_rewards[agent_id] += reward
                else:
                    episode_rewards[agent_id] = reward
                
                # 종료 상태 저장
                memory.add_agent_experience(
                    agent_id,
                    np.zeros_like(state),  # 종료 상태는 중요하지 않음
                    np.zeros(2),
                    np.zeros(2),
                    {'obs': np.zeros((N_AGENT, FRAMES, 512)), 
                     'mask': np.zeros(N_AGENT, dtype=bool)},
                    np.zeros(CONTINUOUS_ACTION_SIZE),
                    reward,
                    True,  # 종료됨
                    0.0,   # 종료 상태의 가치는 0
                    0.0    # 종료 상태의 로그 확률은 0
                )
            # 통합된 행동 생성
            all_actions = np.zeros((decision_steps.agent_id.size, CONTINUOUS_ACTION_SIZE))
            for i, agent_id in enumerate(decision_steps.agent_id):
                all_actions[i] = action.cpu().numpy()[0, 0]
            
            # 환경에 행동 적용
            action_tuple = ActionTuple(continuous=all_actions)
            env.set_actions(behavior_name, action_tuple)
            env.step()
            
            total_steps += 1
            step += 1
            
            # 업데이트 수행
            if total_steps % UPDATE_INTERVAL == 0:
                # 모든 에이전트의 경험 수집
                experiences = memory.get_all_experiences()
                
                if len(experiences['states']) > 0:
                    # PPO 업데이트 수행
                    pass  # 임시 패스 문 추가

            # 모든 에이전트가 종료되었는지 확인
            if len(memory.get_active_agents()) == 0:
                break
        
        # 에피소드 로깅
        avg_reward = sum(episode_rewards.values()) / max(len(episode_rewards), 1)
        writer.add_scalar('Reward/Episode', avg_reward, episode)
        print(f"Episode {episode}, Active Agents: {len(episode_rewards)}, Avg Reward: {avg_reward:.2f}")
        
        # 모델 저장
        if episode % SAVE_INTERVAL == 0:
            torch.save(policy.state_dict(), 
                      os.path.join(SAVE_PATH, f'policy_episode_{episode}.pth'))

    # 환경 종료
    env.close()

if __name__ == "__main__":
    main()
