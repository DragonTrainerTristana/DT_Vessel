import os
import torch
import numpy as np
import csv
import datetime
import math
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from config import *
from networks import CNNPolicy
from memory import Memory
from functions import calculate_returns, log_normal_density
from frame_stack import MultiAgentFrameStack

def save_onnx_model(policy, save_path):
    """
    Export policy network to ONNX format for Unity Barracuda inference.

    Unity ML-Agents Barracuda requires ONNX format to run models without Python.
    This exports the full forward pass including message generation.
    """
    try:
        import torch.onnx

        policy.eval()  # Evaluation mode

        # Create dummy inputs matching observation space
        dummy_state = torch.randn(1, STATE_SIZE * FRAMES).to(DEVICE)  # [1, 360*3=1080]
        dummy_goal = torch.randn(1, 2).to(DEVICE)                      # [1, 2]
        dummy_speed = torch.randn(1, 2).to(DEVICE)                     # [1, 2]
        dummy_colregs = torch.randn(1, 5).to(DEVICE)                   # [1, 5]
        dummy_neighbor_obs = torch.randn(1, MAX_NEIGHBORS, NEIGHBOR_OBS_SIZE).to(DEVICE)  # [1, 4, 371]
        dummy_neighbor_mask = torch.ones(1, MAX_NEIGHBORS, dtype=torch.bool).to(DEVICE)   # [1, 4]

        # Export to ONNX
        torch.onnx.export(
            policy,
            (dummy_state, dummy_goal, dummy_speed, dummy_colregs, dummy_neighbor_obs, dummy_neighbor_mask),
            save_path,
            input_names=[
                'state',           # [1, 1080]
                'goal',            # [1, 2]
                'speed',           # [1, 2]
                'colregs',         # [1, 5]
                'neighbor_obs',    # [1, 4, 371]
                'neighbor_mask'    # [1, 4]
            ],
            output_names=[
                'value',           # [1, 1, 1]
                'action',          # [1, 1, 2]
                'logprob',         # [1, 1]
                'action_mean',     # [1, 1, 2]
                'colregs_pred'     # [1, 5]
            ],
            dynamic_axes={
                'state': {0: 'batch'},
                'goal': {0: 'batch'},
                'speed': {0: 'batch'},
                'colregs': {0: 'batch'},
                'neighbor_obs': {0: 'batch'},
                'neighbor_mask': {0: 'batch'},
                'value': {0: 'batch'},
                'action': {0: 'batch'},
                'logprob': {0: 'batch'},
                'action_mean': {0: 'batch'},
                'colregs_pred': {0: 'batch'}
            },
            opset_version=11,  # Unity Barracuda supports opset 11
            do_constant_folding=True
        )

        print(f"  [OK] ONNX model exported: {save_path}")
        print(f"  [INFO] Use this .onnx file in Unity Inspector → Model")

        policy.train()  # Back to training mode

    except Exception as e:
        print(f"  [ERROR] ONNX export failed: {e}")
        print(f"  [INFO] Continuing with PyTorch .pth only")

def setup_logging():
    csv_dir = os.path.join(SAVE_PATH, 'csv_logs')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    episode_log_file = os.path.join(csv_dir, 'episode_logs.csv')
    step_log_file = os.path.join(csv_dir, 'step_logs.csv')
    reward_log_file = os.path.join(csv_dir, 'reward_logs.csv')
    training_log_file = os.path.join(csv_dir, 'training_logs.csv')
    policy_log_file = os.path.join(csv_dir, 'policy_logs.csv')

    with open(episode_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'avg_reward', 'total_reward',
            'collision_count', 'collision_rate', 'success_count', 'success_rate',
            'episode_length', 'active_agents', 'learning_rate'
        ])

    with open(step_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'step', 'total_steps', 'reward', 'action_0', 'action_1',
            'value', 'logprob', 'agent_id', 'done'
        ])

    with open(reward_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'step', 'total_reward', 'arrival_reward', 'collision_penalty',
            'goal_progress_reward', 'rotation_penalty', 'colregs_reward'
        ])

    with open(training_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'policy_loss', 'value_loss', 'entropy_loss',
            'total_loss', 'learning_rate', 'gradient_norm', 'clip_fraction',
            'value_mean', 'value_std', 'policy_entropy', 'approx_kl_div'
        ])

    with open(policy_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'action_mean_0', 'action_mean_1',
            'action_std_0', 'action_std_1', 'value_mean', 'value_std',
            'logprob_mean', 'logprob_std', 'entropy_mean'
        ])

    return episode_log_file, step_log_file, reward_log_file, training_log_file, policy_log_file

def parse_observation(obs_raw):
    """
    Parse 1855D observation (GitHub + COLREGs + Full Neighbor Obs):
    [0:360]      Self radar state (360 rays)
    [360:362]    goal (distance, angle)
    [362:364]    velocity (speed, yaw_rate)
    [364]        heading
    [365]        rudder
    [366:1850]   neighbor observations (4 × 371D each):
                   - Each neighbor: [360 radar + 2 goal + 2 speed + 5 colregs + 1 heading + 1 rudder] = 371D
    [1850:1855]  Self COLREGs situation (one-hot 5D)
    """
    # Self Radar
    state = obs_raw[:STATE_SIZE]  # [0:360]

    # Self state
    goal_distance = obs_raw[360]
    goal_angle = obs_raw[361]
    speed = obs_raw[362]
    yaw_rate = obs_raw[363]
    heading = obs_raw[364]
    rudder = obs_raw[365]

    goal = np.array([goal_distance, goal_angle])
    self_speed = np.array([speed, yaw_rate])

    # Neighbor observations (4 × 371D = 1484D)
    neighbor_obs = np.zeros((MAX_NEIGHBORS, NEIGHBOR_OBS_SIZE))
    neighbor_mask = np.zeros(MAX_NEIGHBORS, dtype=bool)

    start_idx = 366
    for i in range(MAX_NEIGHBORS):
        neighbor_start = start_idx + i * NEIGHBOR_OBS_SIZE
        neighbor_end = neighbor_start + NEIGHBOR_OBS_SIZE
        neighbor_data = obs_raw[neighbor_start:neighbor_end]

        # Check if this neighbor slot is valid (non-zero observation)
        if np.any(neighbor_data != 0):
            neighbor_obs[i] = neighbor_data
            neighbor_mask[i] = True

    # Self COLREGs situation (one-hot 5D)
    colregs_situation = obs_raw[1850:1855]

    return state, goal, self_speed, neighbor_obs, neighbor_mask, colregs_situation

def ppo_update(policy, optimizer, memory, writer, total_steps, training_log_file):
    """PPO 학습 업데이트 (COLREGs 학습 포함)"""
    experiences = memory.get_all_experiences()

    if len(experiences['states']) == 0:
        return

    # Returns와 Advantages 계산
    states = experiences['states']
    rewards = experiences['rewards']
    dones = experiences['dones']
    values = experiences['values']

    returns = calculate_returns(rewards, dones, 0, values, DISCOUNT_FACTOR, GAE_LAMBDA)
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Tensor로 변환 (GitHub 방식 + Neighbor Obs)
    states_tensor = torch.FloatTensor(states).to(DEVICE)
    goals_tensor = torch.FloatTensor(experiences['goals']).to(DEVICE)
    speeds_tensor = torch.FloatTensor(experiences['speeds']).to(DEVICE)
    colregs_tensor = torch.FloatTensor(experiences['colregs_situations']).to(DEVICE)  # COLREGs 추가
    neighbor_obs_tensor = torch.FloatTensor(experiences['neighbor_obs']).to(DEVICE)  # Neighbor observations
    neighbor_mask_tensor = torch.BoolTensor(experiences['neighbor_mask']).to(DEVICE)  # Neighbor mask
    actions_tensor = torch.FloatTensor(experiences['actions']).to(DEVICE)
    old_logprobs_tensor = torch.FloatTensor(experiences['logprobs']).to(DEVICE)
    returns_tensor = torch.FloatTensor(returns).to(DEVICE)
    advantages_tensor = torch.FloatTensor(advantages).to(DEVICE)

    # PPO 에폭 루프
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy_loss = 0
    total_colregs_loss = 0  # COLREGs loss 추가
    total_loss = 0
    clip_fraction = 0
    approx_kl = 0
    num_updates = 0

    for epoch in range(N_EPOCH):
        # Mini-batch 샘플링
        indices = np.random.permutation(len(states))

        for start in range(0, len(indices), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(indices))
            batch_indices = indices[start:end]

            # Batch 데이터 준비 (GitHub 방식 + Neighbor Obs)
            batch_states = states_tensor[batch_indices]
            batch_goals = goals_tensor[batch_indices]
            batch_speeds = speeds_tensor[batch_indices]
            batch_colregs = colregs_tensor[batch_indices]  # COLREGs 추가
            batch_neighbor_obs = neighbor_obs_tensor[batch_indices]  # Neighbor observations
            batch_neighbor_mask = neighbor_mask_tensor[batch_indices]  # Neighbor mask
            batch_actions = actions_tensor[batch_indices]
            batch_old_logprobs = old_logprobs_tensor[batch_indices]
            batch_returns = returns_tensor[batch_indices]
            batch_advantages = advantages_tensor[batch_indices]

            # Evaluate actions (COLREGs + Neighbor 정보 포함)
            # Note: evaluate_actions는 내부적으로 forward()를 호출하므로
            # neighbor_obs와 neighbor_mask를 전달해야 함
            # 하지만 현재 evaluate_actions는 neighbor 파라미터를 받지 않음
            # 따라서 forward()를 직접 호출하고 logprob를 재계산
            _, _, _, mean, colregs_pred = policy.forward(
                batch_states, batch_goals, batch_speeds, batch_colregs,
                batch_neighbor_obs, batch_neighbor_mask
            )

            # logprob 재계산
            logstd = policy.logstd.expand_as(mean)
            std = torch.exp(logstd)
            new_logprobs = log_normal_density(batch_actions, mean, std=std, log_std=logstd)

            # Critic forward로 value 계산
            values = policy.critic_forward(batch_states, batch_goals, batch_speeds, batch_colregs)

            # Entropy 계산
            dist_entropy = 0.5 + 0.5 * torch.log(2 * torch.tensor(math.pi)) + logstd
            dist_entropy = dist_entropy.sum(-1).mean()

            # PPO Loss 계산
            ratio = torch.exp(new_logprobs - batch_old_logprobs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(), batch_returns)

            # Use dist_entropy from evaluate_actions
            entropy_loss = -ENTROPY_BONUS * dist_entropy

            # COLREGs Classification Loss (Auxiliary Task)
            colregs_loss = F.cross_entropy(colregs_pred, batch_colregs.argmax(dim=-1))

            # Total loss (COLREGs loss 추가)
            loss = policy_loss + VALUE_LOSS_COEF * value_loss + entropy_loss + 0.1 * colregs_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            # 통계 수집
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_colregs_loss += colregs_loss.item()  # COLREGs loss 추가
            total_loss += loss.item()

            with torch.no_grad():
                clip_fraction += ((ratio - 1.0).abs() > EPSILON).float().mean().item()
                approx_kl += ((ratio - 1.0) - torch.log(ratio)).mean().item()

            num_updates += 1

    # 평균 계산
    if num_updates > 0:
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        avg_colregs_loss = total_colregs_loss / num_updates  # COLREGs loss 평균
        avg_total_loss = total_loss / num_updates
        avg_clip_fraction = clip_fraction / num_updates
        avg_approx_kl = approx_kl / num_updates

        # 로그 기록
        writer.add_scalar('Loss/Policy', avg_policy_loss, total_steps)
        writer.add_scalar('Loss/Value', avg_value_loss, total_steps)
        writer.add_scalar('Loss/Entropy', avg_entropy_loss, total_steps)
        writer.add_scalar('Loss/COLREGs', avg_colregs_loss, total_steps)  # COLREGs loss 로그
        writer.add_scalar('Loss/Total', avg_total_loss, total_steps)
        writer.add_scalar('PPO/ClipFraction', avg_clip_fraction, total_steps)
        writer.add_scalar('PPO/ApproxKL', avg_approx_kl, total_steps)

        with open(training_log_file, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                0, total_steps, avg_policy_loss, avg_value_loss, avg_entropy_loss,
                avg_total_loss, LEARNING_RATE, grad_norm, avg_clip_fraction,
                values.mean().item(), values.std().item(), dist_entropy.item(), avg_approx_kl
            ])

        print(f"  PPO Update: policy_loss={avg_policy_loss:.4f}, value_loss={avg_value_loss:.4f}, "
              f"colregs_loss={avg_colregs_loss:.4f}, entropy={avg_entropy_loss:.4f}, kl={avg_approx_kl:.4f}")

def main():
    print("="*80)
    print("[START] Vessel ML-Agent Training Start")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"PPO Epochs: {N_EPOCH}")
    print(f"Update Interval: {UPDATE_INTERVAL} steps")
    print(f"Max Episodes: {NUM_EPISODES}")
    print(f"Max Steps per Episode: {MAX_STEPS}")
    print("="*80)

    channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name=None,
        side_channels=[channel],
        worker_id=WORKER_ID,
        base_port=BASE_PORT
    )
    channel.set_configuration_parameters(time_scale=TIME_SCALE)

    policy = CNNPolicy(MSG_ACTION_SPACE, CONTINUOUS_ACTION_SIZE, FRAMES, N_AGENT).to(DEVICE)

    # 모델 로드
    if LOAD_MODEL:
        if MODEL_PATH is None:
            print("[ERROR] LOAD_MODEL=True but MODEL_PATH is None!")
            print("[ERROR] Please set MODEL_PATH in config.py")
            exit(1)

        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model file not found: {MODEL_PATH}")
            exit(1)

        try:
            policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"\n[OK] Model loaded from: {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            exit(1)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    print(f"[OK] Policy Network Ready: {sum(p.numel() for p in policy.parameters())} parameters")
    print(f"[INFO] Training Mode: {TRAIN_MODE}")
    print(f"[INFO] Model Load: {LOAD_MODEL}")

    writer = SummaryWriter(log_dir=os.path.join(SAVE_PATH, 'logs'))
    episode_log_file, step_log_file, reward_log_file, training_log_file, policy_log_file = setup_logging()

    total_steps = 0
    memory = Memory()
    frame_stack = MultiAgentFrameStack(FRAMES, STATE_SIZE)

    # 학습 모니터링을 위한 변수
    recent_rewards = []
    recent_collision_rates = []
    recent_success_rates = []
    window_size = 10  # 최근 10 에피소드 평균

    for episode in range(NUM_EPISODES):
        print(f"\n{'='*80}")
        print(f"[EPISODE] Episode {episode}/{NUM_EPISODES} Start (Total Steps: {total_steps})")
        print(f"{'='*80}")

        env.reset()
        memory.reset_for_new_episode()
        frame_stack.clear_all()
        behavior_name = list(env.behavior_specs)[0]

        step = 0
        episode_rewards = {}
        episode_stats = {
            'collision_count': 0,
            'success_count': 0,
            'total_reward': 0,
            'step_count': 0
        }

        last_print_step = 0
        print_interval = max(MAX_STEPS // 10, 1)  # 10번 진행 상황 출력

        while step < MAX_STEPS:
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            agent_actions = {}

            # ========== GitHub 방식 + Full Neighbor Obs: Message Exchange ==========
            # 1단계: 모든 에이전트의 observation 수집 및 파싱
            agent_data = {}  # {agent_id: {state_stack, goal, speed, neighbor_obs, neighbor_mask, colregs}}

            for agent_id in decision_steps.agent_id:
                obs_raw = decision_steps.obs[0][decision_steps.agent_id.tolist().index(agent_id)]
                state, goal, speed, neighbor_obs, neighbor_mask, colregs_situation = parse_observation(obs_raw)

                # Frame stack 적용
                state_stack = frame_stack.update(agent_id, state)

                agent_data[agent_id] = {
                    'state_stack': state_stack,
                    'goal': goal,
                    'speed': speed,
                    'neighbor_obs': neighbor_obs,
                    'neighbor_mask': neighbor_mask,
                    'colregs': colregs_situation
                }

                if agent_id not in episode_rewards:
                    episode_rewards[agent_id] = 0

            # 2단계: 행동 생성 (GitHub 방식 + Neighbor Obs)
            # policy.forward()를 사용하여 neighbor_obs와 neighbor_mask를 함께 전달
            for agent_id, data in agent_data.items():
                state_tensor = torch.FloatTensor(data['state_stack']).unsqueeze(0).to(DEVICE)
                goal_tensor = torch.FloatTensor(data['goal']).unsqueeze(0).to(DEVICE)
                speed_tensor = torch.FloatTensor(data['speed']).unsqueeze(0).to(DEVICE)
                colregs_tensor = torch.FloatTensor(data['colregs']).unsqueeze(0).to(DEVICE)
                neighbor_obs_tensor = torch.FloatTensor(data['neighbor_obs']).unsqueeze(0).to(DEVICE)
                neighbor_mask_tensor = torch.BoolTensor(data['neighbor_mask']).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    # policy.forward()가 neighbor_obs와 neighbor_mask를 받아서
                    # 내부적으로 메시지를 생성하고 행동을 결정
                    value, action, logprob, mean, colregs_pred = policy.forward(
                        state_tensor, goal_tensor, speed_tensor, colregs_tensor,
                        neighbor_obs_tensor, neighbor_mask_tensor
                    )

                agent_actions[agent_id] = action.cpu().numpy()[0, 0]

                # Reward 처리
                reward = decision_steps.reward[decision_steps.agent_id.tolist().index(agent_id)]
                episode_rewards[agent_id] += reward
                episode_stats['total_reward'] += reward

                # 경험 저장 (GitHub 방식 + Neighbor Obs)
                memory.add_agent_experience(
                    agent_id,
                    data['state_stack'],  # Frame stacked state
                    data['goal'],
                    data['speed'],
                    data['colregs'],  # COLREGs 상황
                    data['neighbor_obs'],  # Neighbor observations [4, 371]
                    data['neighbor_mask'],  # Neighbor mask [4]
                    action.cpu().numpy()[0, 0],
                    reward,
                    False,
                    value.cpu().numpy()[0, 0, 0],
                    logprob.cpu().numpy()[0, 0]
                )

                # Step 로그
                action_np = action.cpu().numpy()[0, 0]
                with open(step_log_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([
                        episode, step, total_steps, reward,
                        action_np[0], action_np[1],
                        value.cpu().numpy()[0, 0, 0], logprob.cpu().numpy()[0, 0],
                        agent_id, False
                    ])

            # Terminal steps 처리
            for idx, agent_id in enumerate(terminal_steps.agent_id):
                if agent_id in decision_steps.agent_id:
                    continue

                reward = terminal_steps.reward[idx]

                if agent_id in episode_rewards:
                    episode_rewards[agent_id] += reward
                else:
                    episode_rewards[agent_id] = reward

                episode_stats['total_reward'] += reward

                # 종료 타입 판정 (reward가 음수면 충돌, 양수면 성공)
                if reward < 0:
                    episode_stats['collision_count'] += 1
                else:
                    episode_stats['success_count'] += 1

                # Frame stack 제거
                frame_stack.remove_agent(agent_id)

                # 종료 경험 저장 (GitHub 방식 + Neighbor Obs)
                memory.add_agent_experience(
                    agent_id,
                    np.zeros(STATE_SIZE * FRAMES),
                    np.zeros(2),
                    np.zeros(2),
                    np.zeros(5),  # COLREGs 상황 (5D zeros)
                    np.zeros((MAX_NEIGHBORS, NEIGHBOR_OBS_SIZE)),  # Neighbor obs (zeros)
                    np.zeros(MAX_NEIGHBORS, dtype=bool),  # Neighbor mask (zeros)
                    np.zeros(CONTINUOUS_ACTION_SIZE),
                    reward,
                    True,
                    0.0,
                    0.0
                )

                with open(step_log_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([
                        episode, step, total_steps, reward,
                        0.0, 0.0, 0.0, 0.0, agent_id, True
                    ])

            # 액션 적용
            all_actions = np.zeros((decision_steps.agent_id.size, CONTINUOUS_ACTION_SIZE))
            for i, agent_id in enumerate(decision_steps.agent_id):
                all_actions[i] = agent_actions[agent_id]

            action_tuple = ActionTuple(continuous=all_actions)

            try:
                env.set_actions(behavior_name, action_tuple)
                env.step()
            except Exception as e:
                print(f"  [WARNING] Unity connection lost: {e}")
                print(f"  [WARNING] Attempting to save model and exit gracefully...")

                # 모델 저장
                if episode > 0:
                    torch.save(policy.state_dict(),
                              os.path.join(SAVE_PATH, f'policy_emergency_{episode}_{step}.pth'))
                    print(f"  Emergency model saved at episode {episode}, step {step}")
                break

            total_steps += 1
            step += 1
            episode_stats['step_count'] += 1

            # 진행 상황 출력
            if step - last_print_step >= print_interval:
                active_agents = len(memory.get_active_agents())
                avg_reward_so_far = episode_stats['total_reward'] / max(step, 1)
                print(f"  Step {step}/{MAX_STEPS} | Active Agents: {active_agents} | "
                      f"Avg Reward: {avg_reward_so_far:.3f} | "
                      f"Collisions: {episode_stats['collision_count']} | "
                      f"Success: {episode_stats['success_count']}")
                last_print_step = step

            # PPO 업데이트 (학습 모드에서만)
            if TRAIN_MODE and total_steps % UPDATE_INTERVAL == 0 and total_steps > 0:
                print(f"\n  [UPDATE] Running PPO update at step {total_steps}...")
                ppo_update(policy, optimizer, memory, writer, total_steps, training_log_file)
                memory.clear()  # 업데이트 후 메모리 클리어

            # 모든 에이전트가 종료되면 에피소드 종료
            if len(memory.get_active_agents()) == 0:
                print(f"  [WARNING] All agents terminated at step {step}")
                break

        # 에피소드 종료
        avg_reward = sum(episode_rewards.values()) / max(len(episode_rewards), 1)
        collision_rate = episode_stats['collision_count'] / max(episode_stats['step_count'], 1)
        success_rate = episode_stats['success_count'] / max(len(episode_rewards), 1)

        # 최근 에피소드 통계 업데이트
        recent_rewards.append(avg_reward)
        recent_collision_rates.append(collision_rate)
        recent_success_rates.append(success_rate)
        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)
            recent_collision_rates.pop(0)
            recent_success_rates.pop(0)

        with open(episode_log_file, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                episode, total_steps, avg_reward, episode_stats['total_reward'],
                episode_stats['collision_count'], collision_rate,
                episode_stats['success_count'], success_rate,
                episode_stats['step_count'], len(episode_rewards), LEARNING_RATE
            ])

        writer.add_scalar('Reward/Episode', avg_reward, episode)
        writer.add_scalar('Collision/Rate', collision_rate, episode)
        writer.add_scalar('Success/Rate', success_rate, episode)
        writer.add_scalar('Episode/Length', episode_stats['step_count'], episode)

        # 에피소드 결과 출력
        print(f"\n{'─'*80}")
        print(f"[SUMMARY] Episode {episode} Summary:")
        print(f"{'─'*80}")
        print(f"  Agents: {len(episode_rewards)}")
        print(f"  Total Steps: {episode_stats['step_count']}")
        print(f"  Average Reward: {avg_reward:.3f}")
        print(f"  Total Reward: {episode_stats['total_reward']:.3f}")
        print(f"  Collisions: {episode_stats['collision_count']} (Rate: {collision_rate:.2%})")
        print(f"  Success: {episode_stats['success_count']} (Rate: {success_rate:.2%})")

        # 최근 N 에피소드 평균
        if len(recent_rewards) >= 3:
            print(f"\n  [STATS] Last {len(recent_rewards)} Episodes Average:")
            print(f"    Reward: {np.mean(recent_rewards):.3f} (±{np.std(recent_rewards):.3f})")
            print(f"    Collision Rate: {np.mean(recent_collision_rates):.2%}")
            print(f"    Success Rate: {np.mean(recent_success_rates):.2%}")

            # 학습 추세 판정
            if len(recent_rewards) >= 5:
                recent_trend = np.mean(recent_rewards[-3:]) - np.mean(recent_rewards[-6:-3]) if len(recent_rewards) >= 6 else 0
                if recent_trend > 0.1:
                    print(f"    [GOOD] Trend: Improving (+{recent_trend:.3f})")
                elif recent_trend < -0.1:
                    print(f"    [WARNING] Trend: Declining ({recent_trend:.3f})")
                else:
                    print(f"    [STABLE] Trend: Stable ({recent_trend:+.3f})")

        print(f"{'─'*80}")

        # 모델 저장 (학습 모드에서만)
        if TRAIN_MODE and episode % SAVE_INTERVAL == 0 and episode > 0:
            torch.save(policy.state_dict(),
                      os.path.join(SAVE_PATH, f'policy_episode_{episode}.pth'))

            onnx_path = os.path.join(SAVE_PATH, f'policy_episode_{episode}.onnx')
            save_onnx_model(policy, onnx_path)
            print(f"  Model saved at episode {episode}")

    env.close()
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()
