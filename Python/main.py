import os   
import torch
import numpy as np
import csv
import datetime
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from config import *
from networks import CNNPolicy
from memory import Memory
from functions import calculate_returns

def save_onnx_model(policy, save_path):
    """ONNX 모델 저장"""
    policy.eval()
    
    # 더미 입력 생성
    dummy_state = torch.randn(1, 46).to(DEVICE)
    dummy_goal = torch.randn(1, 2).to(DEVICE)
    dummy_speed = torch.randn(1, 2).to(DEVICE)
    dummy_neighbor_obs = torch.randn(1, 4, 46).to(DEVICE)
    dummy_neighbor_mask = torch.ones(1, 4, dtype=torch.bool).to(DEVICE)
    
    # ONNX 변환
    torch.onnx.export(
        policy, 
        (dummy_state, dummy_goal, dummy_speed, dummy_neighbor_obs, dummy_neighbor_mask),
        save_path,
        input_names=['state', 'goal', 'speed', 'neighbor_obs', 'neighbor_mask'],
        output_names=['action'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'goal': {0: 'batch_size'},
            'speed': {0: 'batch_size'},
            'neighbor_obs': {0: 'batch_size'},
            'neighbor_mask': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        },
        opset_version=11
    )
    print(f"ONNX 모델 저장됨: {save_path}")

def setup_logging():
    """로깅 설정"""
    # CSV 파일 경로 설정
    csv_dir = os.path.join(SAVE_PATH, 'csv_logs')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    # CSV 파일들 초기화
    episode_log_file = os.path.join(csv_dir, 'episode_logs.csv')
    step_log_file = os.path.join(csv_dir, 'step_logs.csv')
    reward_log_file = os.path.join(csv_dir, 'reward_logs.csv')
    training_log_file = os.path.join(csv_dir, 'training_logs.csv')
    policy_log_file = os.path.join(csv_dir, 'policy_logs.csv')
    
    # Episode 로그 헤더
    with open(episode_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'avg_reward', 'total_reward', 
            'collision_count', 'collision_rate', 'success_count', 'success_rate',
            'episode_length', 'active_agents', 'learning_rate'
        ])
    
    # Step 로그 헤더
    with open(step_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'step', 'total_steps', 'reward', 'action_0', 'action_1',
            'value', 'logprob', 'agent_id', 'done'
        ])
    
    # Reward 로그 헤더
    with open(reward_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'step', 'total_reward', 'arrival_reward', 'collision_penalty',
            'movement_reward', 'goal_progress_reward', 'rotation_penalty', 'stalemate_penalty'
        ])
    
    # Training 로그 헤더 (강화학습 표준 그래프용)
    with open(training_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'policy_loss', 'value_loss', 'entropy_loss',
            'total_loss', 'learning_rate', 'gradient_norm', 'clip_fraction',
            'value_mean', 'value_std', 'policy_entropy', 'approx_kl_div'
        ])
    
    # Policy 로그 헤더 (정책 분석용)
    with open(policy_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'action_mean_0', 'action_mean_1',
            'action_std_0', 'action_std_1', 'value_mean', 'value_std',
            'logprob_mean', 'logprob_std', 'entropy_mean'
        ])
    
    return episode_log_file, step_log_file, reward_log_file, training_log_file, policy_log_file

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


    # 연동 테스트

    # 정책 네트워크와 옵티마이저 초기화
    # N_AGENT는 이제 최대 이웃 수를 의미
    policy = CNNPolicy(MSG_ACTION_SPACE, CONTINUOUS_ACTION_SIZE, FRAMES, N_AGENT).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    # 텐서보드 로깅 설정
    writer = SummaryWriter(log_dir=os.path.join(SAVE_PATH, 'logs'))
    
    # CSV 로깅 설정
    episode_log_file, step_log_file, reward_log_file, training_log_file, policy_log_file = setup_logging()

    # 학습 루프
    total_steps = 0
    memory = Memory()  # 새로운 메모리 구조 사용
    
    # 통계 변수들
    episode_stats = {
        'collision_count': 0,
        'success_count': 0,
        'total_reward': 0,
        'step_count': 0
    }
    
    for episode in range(NUM_EPISODES):
        # 환경 초기화
        env.reset()
        memory.reset_for_new_episode()  # 새 에피소드 시작 시 메모리 초기화
        behavior_name = list(env.behavior_specs)[0]
        
        step = 0
        episode_rewards = {}  # 에이전트별 보상 추적
        
        # 에피소드 통계 초기화
        episode_stats = {
            'collision_count': 0,
            'success_count': 0,
            'total_reward': 0,
            'step_count': 0,
            'arrival_reward': 0,
            'collision_penalty': 0,
            'movement_reward': 0,
            'goal_progress_reward': 0,
            'rotation_penalty': 0,
            'stalemate_penalty': 0
        }
        
        while step < MAX_STEPS:
            # 환경에서 상태 정보 얻기
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            # 각 에이전트별 행동 저장
            agent_actions = {}
            
            # 현재 활성화된 에이전트 처리
            for agent_id in decision_steps.agent_id:
                # 에이전트의 상태 정보 (46차원)
                state = decision_steps.obs[0][agent_id]  # 46차원 기본 관찰
                
                # 목표 정보 (2차원)
                if len(decision_steps.obs) > 1:
                    goal = decision_steps.obs[1][agent_id]  # 2차원 목표
                else:
                    goal = np.zeros(2)
                
                # 속도 정보 (2차원)
                if len(decision_steps.obs) > 2:
                    speed = decision_steps.obs[2][agent_id]  # 2차원 속도
                else:
                    speed = np.zeros(2)
                
                # 이웃 정보 처리 (184차원 = 4 × 46)
                if len(decision_steps.obs) > 3:
                    # 이웃 관찰 배열 (184차원)
                    neighbor_obs_raw = decision_steps.obs[3][agent_id]
                    
                    # 184차원을 4개 이웃 × 46차원으로 재구성
                    neighbor_obs = neighbor_obs_raw.reshape(N_AGENT, -1)  # (4, 46)
                    
                    # 유효한 이웃 마스크 (0이 아닌 값이 있으면 유효)
                    neighbor_mask = torch.tensor(
                        [np.any(neighbor_obs[i] != 0) for i in range(N_AGENT)], 
                        dtype=torch.bool
                    ).to(DEVICE)
                else:
                    # 이웃 정보가 없는 경우
                    neighbor_obs = np.zeros((N_AGENT, 46))
                    neighbor_mask = torch.zeros(N_AGENT, dtype=torch.bool).to(DEVICE)
                
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
                
                # 각 에이전트별 행동 저장
                agent_actions[agent_id] = action.cpu().numpy()[0, 0]
                
                # 행동 저장
                if agent_id not in episode_rewards:
                    episode_rewards[agent_id] = 0
                
                # 보상 추적
                reward = decision_steps.reward[agent_id]
                episode_rewards[agent_id] += reward
                episode_stats['total_reward'] += reward
                
                # Step 로그 기록
                action_np = action.cpu().numpy()[0, 0]
                with open(step_log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode, step, total_steps, reward, 
                        action_np[0], action_np[1],  # action_0, action_1
                        value.cpu().numpy()[0, 0], logprob.cpu().numpy()[0, 0],
                        agent_id, False  # done
                    ])
                
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
                
                episode_stats['total_reward'] += reward
                episode_stats['collision_count'] += 1  # 종료 = 충돌로 가정
                
                # 종료 Step 로그 기록
                with open(step_log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode, step, total_steps, reward, 
                        0.0, 0.0,  # action_0, action_1 (종료 상태)
                        0.0, 0.0,  # value, logprob (종료 상태)
                        agent_id, True  # done
                    ])
                
                # 종료 상태 저장
                memory.add_agent_experience(
                    agent_id,
                    np.zeros_like(state),  # 종료 상태는 중요하지 않음
                    np.zeros(2),
                    np.zeros(2),
                    {'obs': np.zeros((N_AGENT, 46)), 
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
                all_actions[i] = agent_actions[agent_id]
            
            # 환경에 행동 적용
            action_tuple = ActionTuple(continuous=all_actions)
            env.set_actions(behavior_name, action_tuple)
            env.step()
            
            total_steps += 1
            step += 1
            episode_stats['step_count'] += 1
            
            # Reward 로그 기록 (매 스텝)
            with open(reward_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode, step, episode_stats['total_reward'],
                    episode_stats['arrival_reward'], episode_stats['collision_penalty'],
                    episode_stats['movement_reward'], episode_stats['goal_progress_reward'],
                    episode_stats['rotation_penalty'], episode_stats['stalemate_penalty']
                ])
            
            # 업데이트 수행
            if total_steps % UPDATE_INTERVAL == 0:
                # 모든 에이전트의 경험 수집
                experiences = memory.get_all_experiences()
                
                if len(experiences['states']) > 0:
                    # PPO 업데이트 수행
                    # 그래디언트 클리핑 및 학습률 스케줄링 추가
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                    
                    # 학습률 감소 (선택적)
                    if total_steps > 1000000:  # 100만 스텝 후
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = LEARNING_RATE * 0.5
                    
                    # Training 로그 기록 (업데이트 시)
                    with open(training_log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            episode, total_steps, 0.0, 0.0, 0.0,  # 손실값들 (PPO 구현 시 추가)
                            0.0, LEARNING_RATE, 0.0, 0.0,  # 학습 관련
                            0.0, 0.0, 0.0, 0.0  # 정책 통계
                        ])

            # 모든 에이전트가 종료되었는지 확인
            if len(memory.get_active_agents()) == 0:
                break
        
        # 에피소드 통계 계산
        avg_reward = sum(episode_rewards.values()) / max(len(episode_rewards), 1)
        collision_rate = episode_stats['collision_count'] / max(episode_stats['step_count'], 1)
        success_rate = episode_stats['success_count'] / max(len(episode_rewards), 1)
        
        # Episode 로그 기록
        with open(episode_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, total_steps, avg_reward, episode_stats['total_reward'],
                episode_stats['collision_count'], collision_rate,
                episode_stats['success_count'], success_rate,
                episode_stats['step_count'], len(episode_rewards), LEARNING_RATE
            ])
        
        # 텐서보드 로깅
        writer.add_scalar('Reward/Episode', avg_reward, episode)
        writer.add_scalar('Collision/Rate', collision_rate, episode)
        writer.add_scalar('Success/Rate', success_rate, episode)
        writer.add_scalar('Episode/Length', episode_stats['step_count'], episode)
        
        print(f"Episode {episode}, Active Agents: {len(episode_rewards)}, Avg Reward: {avg_reward:.2f}, Collision Rate: {collision_rate:.3f}")
        
        # Policy 로그 기록 (에피소드별 정책 통계)
        with open(policy_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, total_steps, 0.0, 0.0,  # action_mean_0, action_mean_1
                0.0, 0.0,  # action_std_0, action_std_1
                0.0, 0.0,  # value_mean, value_std
                0.0, 0.0, 0.0  # logprob_mean, logprob_std, entropy_mean
            ])
        
        # 모델 저장
        if episode % SAVE_INTERVAL == 0:
            # PyTorch 모델 저장
            torch.save(policy.state_dict(), 
                      os.path.join(SAVE_PATH, f'policy_episode_{episode}.pth'))
            
            # ONNX 모델 저장
            onnx_path = os.path.join(SAVE_PATH, f'policy_episode_{episode}.onnx')
            save_onnx_model(policy, onnx_path)

    # 환경 종료
    env.close()

if __name__ == "__main__":
    main()