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
from frame_stack import MultiAgentFrameStack

def save_onnx_model(policy, save_path):
    policy.eval()

    dummy_state = torch.randn(1, STATE_SIZE * FRAMES).to(DEVICE)
    dummy_goal = torch.randn(1, 2).to(DEVICE)
    dummy_speed = torch.randn(1, 2).to(DEVICE)
    dummy_neighbor_obs = torch.randn(1, N_AGENT, NEIGHBOR_STATE_SIZE).to(DEVICE)
    dummy_neighbor_mask = torch.ones(1, N_AGENT, dtype=torch.bool).to(DEVICE)

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
    Parse 324D observation into components (MDPI 2024):
    [0:180] self state (30 regions × 6)
    [180:184] message passing (goal=2, speed=2)
    [184:324] neighbors (4 × 35D)

    Neighbor 35D = compressed_radar(24) + vessel(4) + goal(3) + fuzzy_colregs(4)
    """
    state = obs_raw[:STATE_SIZE]  # [0:180]
    goal = obs_raw[STATE_SIZE:STATE_SIZE+2]  # [180:182]
    speed = obs_raw[STATE_SIZE+2:STATE_SIZE+4]  # [182:184]
    neighbor_obs_raw = obs_raw[STATE_SIZE+4:]  # [184:324] = 140D

    neighbor_obs = neighbor_obs_raw.reshape(N_AGENT, NEIGHBOR_STATE_SIZE)  # (4, 35)
    neighbor_mask = torch.tensor(
        [bool(np.any(neighbor_obs[i] != 0)) for i in range(N_AGENT)],
        dtype=torch.bool
    ).to(DEVICE)

    # COLREGs 정보 추출 (각 이웃의 마지막 4D: fuzzy weights for 4 situations)
    # [None, HeadOn, CrossingGiveWay, CrossingStandOn, Overtaking]
    colregs_situations = np.zeros(4)  # 집계된 COLREGs 상황

    for i in range(N_AGENT):
        if neighbor_mask[i]:  # 유효한 이웃만
            # 각 이웃의 fuzzy COLREGs는 마지막 4차원
            fuzzy_colregs = neighbor_obs[i, -4:]
            colregs_situations += fuzzy_colregs  # 모든 이웃의 COLREGs 상황 집계

    # 정규화 (0-1 범위)
    if np.sum(colregs_situations) > 0:
        colregs_situations = colregs_situations / np.sum(colregs_situations)

    return state, goal, speed, neighbor_obs, neighbor_mask, colregs_situations

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

    returns = calculate_returns(rewards, dones, 0, values, DISCOUNT_FACTOR)
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Tensor로 변환
    states_tensor = torch.FloatTensor(states).to(DEVICE)
    goals_tensor = torch.FloatTensor(experiences['goals']).to(DEVICE)
    speeds_tensor = torch.FloatTensor(experiences['speeds']).to(DEVICE)
    colregs_tensor = torch.FloatTensor(experiences['colregs_situations']).to(DEVICE)  # COLREGs 추가
    actions_tensor = torch.FloatTensor(experiences['actions']).to(DEVICE)
    old_logprobs_tensor = torch.FloatTensor(experiences['logprobs']).to(DEVICE)
    returns_tensor = torch.FloatTensor(returns).to(DEVICE)
    advantages_tensor = torch.FloatTensor(advantages).to(DEVICE)

    # Neighbor 정보 처리
    neighbor_obs_list = []
    neighbor_mask_list = []
    for neighbor_info in experiences['neighbor_infos']:
        neighbor_obs_list.append(neighbor_info['obs'])
        neighbor_mask_list.append(neighbor_info['mask'])
    neighbor_obs_tensor = torch.FloatTensor(np.array(neighbor_obs_list)).to(DEVICE)
    neighbor_mask_tensor = torch.BoolTensor(np.array(neighbor_mask_list)).to(DEVICE)

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

            # Batch 데이터 준비
            batch_states = states_tensor[batch_indices]
            batch_goals = goals_tensor[batch_indices]
            batch_speeds = speeds_tensor[batch_indices]
            batch_colregs = colregs_tensor[batch_indices]  # COLREGs 추가
            batch_actions = actions_tensor[batch_indices]
            batch_old_logprobs = old_logprobs_tensor[batch_indices]
            batch_returns = returns_tensor[batch_indices]
            batch_advantages = advantages_tensor[batch_indices]
            batch_neighbor_obs = neighbor_obs_tensor[batch_indices]
            batch_neighbor_mask = neighbor_mask_tensor[batch_indices]

            # Forward pass (COLREGs 정보 포함)
            values, _, new_logprobs, _, colregs_pred = policy(
                batch_states, batch_goals, batch_speeds,
                batch_colregs,
                batch_neighbor_obs, batch_neighbor_mask
            )

            # Evaluate actions for correct logprobs
            values, new_logprobs, dist_entropy, colregs_pred = policy.evaluate_actions(
                batch_states, batch_goals, batch_speeds,
                batch_actions, batch_colregs
            )

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
    print("🚢 Vessel ML-Agent Training Start")
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
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    print(f"\n✅ Policy Network Loaded: {sum(p.numel() for p in policy.parameters())} parameters")

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
        print(f"📋 Episode {episode}/{NUM_EPISODES} Start (Total Steps: {total_steps})")
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

            # Decision steps 처리
            for agent_id in decision_steps.agent_id:
                obs_raw = decision_steps.obs[0][agent_id]
                state, goal, speed, neighbor_obs, neighbor_mask, colregs_situations = parse_observation(obs_raw)

                # Frame stack 적용
                state_stack = frame_stack.update(agent_id, state)

                state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(DEVICE)
                goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(DEVICE)
                speed_tensor = torch.FloatTensor(speed).unsqueeze(0).to(DEVICE)
                neighbor_obs_tensor = torch.FloatTensor(neighbor_obs).unsqueeze(0).to(DEVICE)
                colregs_tensor = torch.FloatTensor(colregs_situations).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    value, action, logprob, _, colregs_pred = policy(
                        state_tensor, goal_tensor, speed_tensor,
                        colregs_tensor,
                        neighbor_obs_tensor, neighbor_mask.unsqueeze(0)
                    )

                agent_actions[agent_id] = action.cpu().numpy()[0, 0]

                if agent_id not in episode_rewards:
                    episode_rewards[agent_id] = 0

                reward = decision_steps.reward[agent_id]
                episode_rewards[agent_id] += reward
                episode_stats['total_reward'] += reward

                # 경험 저장 (COLREGs 정보 포함)
                memory.add_agent_experience(
                    agent_id,
                    state_stack,  # Frame stacked state
                    goal,
                    speed,
                    colregs_situations,  # COLREGs 상황 추가
                    {'obs': neighbor_obs, 'mask': neighbor_mask.cpu().numpy()},
                    action.cpu().numpy()[0, 0],
                    reward,
                    False,
                    value.cpu().numpy()[0, 0],
                    logprob.cpu().numpy()[0, 0]
                )

                # Step 로그
                action_np = action.cpu().numpy()[0, 0]
                with open(step_log_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([
                        episode, step, total_steps, reward,
                        action_np[0], action_np[1],
                        value.cpu().numpy()[0, 0], logprob.cpu().numpy()[0, 0],
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

                # 종료 경험 저장 (COLREGs 정보 포함)
                memory.add_agent_experience(
                    agent_id,
                    np.zeros(STATE_SIZE * FRAMES),
                    np.zeros(2),
                    np.zeros(2),
                    np.zeros(4),  # COLREGs 상황 (zeros)
                    {'obs': np.zeros((N_AGENT, NEIGHBOR_STATE_SIZE)),
                     'mask': np.zeros(N_AGENT, dtype=bool)},
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
                print(f"  ⚠️  Unity connection lost: {e}")
                print(f"  ⚠️  Attempting to save model and exit gracefully...")

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

            # PPO 업데이트
            if total_steps % UPDATE_INTERVAL == 0 and total_steps > 0:
                print(f"\n  🔄 Running PPO update at step {total_steps}...")
                ppo_update(policy, optimizer, memory, writer, total_steps, training_log_file)
                memory.clear()  # 업데이트 후 메모리 클리어

            # 모든 에이전트가 종료되면 에피소드 종료
            if len(memory.get_active_agents()) == 0:
                print(f"  ⚠️  All agents terminated at step {step}")
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
        print(f"📊 Episode {episode} Summary:")
        print(f"{'─'*80}")
        print(f"  Agents: {len(episode_rewards)}")
        print(f"  Total Steps: {episode_stats['step_count']}")
        print(f"  Average Reward: {avg_reward:.3f}")
        print(f"  Total Reward: {episode_stats['total_reward']:.3f}")
        print(f"  Collisions: {episode_stats['collision_count']} (Rate: {collision_rate:.2%})")
        print(f"  Success: {episode_stats['success_count']} (Rate: {success_rate:.2%})")

        # 최근 N 에피소드 평균
        if len(recent_rewards) >= 3:
            print(f"\n  📈 Last {len(recent_rewards)} Episodes Average:")
            print(f"    Reward: {np.mean(recent_rewards):.3f} (±{np.std(recent_rewards):.3f})")
            print(f"    Collision Rate: {np.mean(recent_collision_rates):.2%}")
            print(f"    Success Rate: {np.mean(recent_success_rates):.2%}")

            # 학습 추세 판정
            if len(recent_rewards) >= 5:
                recent_trend = np.mean(recent_rewards[-3:]) - np.mean(recent_rewards[-6:-3]) if len(recent_rewards) >= 6 else 0
                if recent_trend > 0.1:
                    print(f"    ✅ Trend: Improving (+{recent_trend:.3f})")
                elif recent_trend < -0.1:
                    print(f"    ⚠️  Trend: Declining ({recent_trend:.3f})")
                else:
                    print(f"    ➡️  Trend: Stable ({recent_trend:+.3f})")

        print(f"{'─'*80}")

        # 모델 저장
        if episode % SAVE_INTERVAL == 0 and episode > 0:
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
