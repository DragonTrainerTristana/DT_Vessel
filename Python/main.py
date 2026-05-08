import os
# CUDA allocator 최적화 — 반드시 torch import 전에 설정 (fragmentation 50%↓)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                       'max_split_size_mb:128,expandable_segments:True')

import random
import torch
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import csv
import datetime
import math
import time
import threading
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from config import *
from networks import CNNPolicy
from memory import Memory
from functions import calculate_returns, RunningMeanStd
from frame_stack import MultiAgentFrameStack
from obs_utils import parse_observation, get_comm_partners

# Conv1D 성능 부스트 (kernel autotune + TF32)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except AttributeError:
    pass

def setup_logging():
    """로깅 설정"""
    csv_dir = os.path.join(SAVE_PATH, 'csv_logs')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    episode_log_file = os.path.join(csv_dir, 'episode_logs.csv')
    training_log_file = os.path.join(csv_dir, 'training_logs.csv')
    message_log_file = os.path.join(csv_dir, 'message_logs.csv')

    with open(episode_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'avg_reward', 'total_reward',
            'collision_count', 'collision_rate', 'success_count', 'success_rate',
            'episode_length', 'active_agents', 'learning_rate'
        ])

    with open(training_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'policy_loss', 'value_loss', 'entropy_loss',
            'colregs_loss', 'total_loss', 'learning_rate', 'gradient_norm'
        ])

    # 메시지 로그 (6D 메시지 + 메타데이터)
    with open(message_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'total_steps', 'agent_id',
            'msg_0', 'msg_1', 'msg_2', 'msg_3', 'msg_4', 'msg_5',
            'others_msg_0', 'others_msg_1', 'others_msg_2', 'others_msg_3', 'others_msg_4', 'others_msg_5',
            'n_agents', 'colregs_situation'
        ])

    return episode_log_file, training_log_file, message_log_file



# parse_observation, get_comm_partners → obs_utils.py로 이동


def ppo_update(policy, optimizer, memory, writer, total_steps, training_log_file):
    """PPO 학습 업데이트"""
    experiences = memory.get_all_experiences()

    if len(experiences['states']) == 0:
        return

    n_samples = len(experiences['states'])

    # ✅ Returns는 이미 에이전트별로 계산되어 있음 (memory.get_all_experiences에서)
    states = experiences['states']
    values = experiences['values']
    returns = experiences['returns']  # 에이전트별 GAE 계산 완료

    # Advantages 계산
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Tensor로 변환
    states_tensor = torch.as_tensor(states, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    goals_tensor = torch.as_tensor(experiences['goals'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    self_states_tensor = torch.as_tensor(experiences['self_states'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    colregs_tensor = torch.as_tensor(experiences['colregs'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    others_msgs_tensor = torch.as_tensor(experiences['others_msgs'], dtype=torch.float32, device=DEVICE).unsqueeze(1)  # ★ others_msg 추가 ★
    actions_tensor = torch.as_tensor(experiences['actions'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    old_logprobs_tensor = torch.as_tensor(experiences['logprobs'], dtype=torch.float32, device=DEVICE)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)
    advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy_loss = 0
    total_colregs_loss = 0
    total_loss = 0
    num_updates = 0

    for epoch in range(N_EPOCH):
        indices = np.random.permutation(n_samples)

        # PPO mini-batch (rollout BATCH_SIZE보다 작게 → GPU peak memory 1/4)
        for start in range(0, n_samples, MINIBATCH_SIZE):
            end = min(start + MINIBATCH_SIZE, n_samples)
            batch_indices = indices[start:end]

            batch_states = states_tensor[batch_indices]
            batch_goals = goals_tensor[batch_indices]
            batch_self_states = self_states_tensor[batch_indices]
            batch_colregs = colregs_tensor[batch_indices]
            batch_others_msgs = others_msgs_tensor[batch_indices]  # ★ others_msg 추가 ★
            batch_actions = actions_tensor[batch_indices]
            batch_old_logprobs = old_logprobs_tensor[batch_indices]
            batch_returns = returns_tensor[batch_indices]
            batch_advantages = advantages_tensor[batch_indices]

            values, logprobs, dist_entropy, colregs_pred = policy.evaluate_actions(
                batch_states, batch_goals, batch_self_states, batch_colregs, batch_others_msgs, batch_actions
            )

            values = values.squeeze(-1).squeeze(-1)
            logprobs = logprobs.squeeze(-1).squeeze(-1)

            ratio = torch.exp(logprobs - batch_old_logprobs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, batch_returns)
            entropy_loss = -ENTROPY_BONUS * dist_entropy

            colregs_pred_flat = colregs_pred.squeeze(1)
            colregs_target = batch_colregs.squeeze(1)
            colregs_loss = F.cross_entropy(colregs_pred_flat, colregs_target.argmax(dim=-1))

            loss = policy_loss + VALUE_LOSS_COEF * value_loss + entropy_loss + COLREGS_LOSS_COEF * colregs_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_colregs_loss += colregs_loss.item()
            total_loss += loss.item()
            num_updates += 1

    if num_updates > 0:
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        avg_colregs_loss = total_colregs_loss / num_updates
        avg_total_loss = total_loss / num_updates

        writer.add_scalar('Loss/Policy', avg_policy_loss, total_steps)
        writer.add_scalar('Loss/Value', avg_value_loss, total_steps)
        writer.add_scalar('Loss/Entropy', avg_entropy_loss, total_steps)
        writer.add_scalar('Loss/COLREGs', avg_colregs_loss, total_steps)
        writer.add_scalar('Loss/Total', avg_total_loss, total_steps)

        with open(training_log_file, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                0, total_steps, avg_policy_loss, avg_value_loss, avg_entropy_loss,
                avg_colregs_loss, avg_total_loss, LEARNING_RATE, grad_norm
            ])

    # GPU peak memory 해제 (5 process 동시 학습 시 fragmentation 방지)
    del states_tensor, goals_tensor, self_states_tensor, colregs_tensor
    del others_msgs_tensor, actions_tensor, old_logprobs_tensor
    del returns_tensor, advantages_tensor
    del experiences   # numpy dict 사본도 즉시 해제 (RAM 7-10GB 누적 방지)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_observations(env, env_idx, behavior_name, frame_stack):
    """
    Phase 1: Unity에서 observation 수집
    Returns: env_data dict or None
    """
    try:
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        n_agents = len(agent_ids)
        if n_agents == 0:
            return {
                'env_idx': env_idx,
                'n_agents': 0,
                'decision_steps': decision_steps,
                'terminal_steps': terminal_steps,
                'empty': True
            }

        batch_states = []
        batch_goals = []
        batch_self_states = []
        batch_colregs = []
        batch_positions = {}
        agent_id_list = []
        global_id_list = []
        rewards = []

        # enumerate로 직접 인덱스 사용 (O(n) -> O(1))
        for idx, agent_id in enumerate(agent_ids):
            obs_raw = decision_steps.obs[0][idx]
            state, goal, self_state, colregs, _, position = parse_observation(obs_raw)

            global_id = f"env{env_idx}_{agent_id}"
            state_stacked = frame_stack.update(agent_id, state)

            batch_states.append(state_stacked)
            batch_goals.append(goal)
            batch_self_states.append(self_state)
            batch_colregs.append(colregs)
            batch_positions[agent_id] = position
            agent_id_list.append(agent_id)
            global_id_list.append(global_id)
            rewards.append(decision_steps.reward[idx])

        # 통신 파트너 계산
        comm_partners = {}
        for agent_id in agent_id_list:
            partners = get_comm_partners(agent_id, batch_positions[agent_id], batch_positions)
            comm_partners[agent_id] = partners

        return {
            'env_idx': env_idx,
            'n_agents': n_agents,
            'batch_states': batch_states,
            'batch_goals': batch_goals,
            'batch_self_states': batch_self_states,
            'batch_colregs': batch_colregs,
            'agent_id_list': agent_id_list,
            'global_id_list': global_id_list,
            'comm_partners': comm_partners,
            'rewards': rewards,
            'decision_steps': decision_steps,
            'terminal_steps': terminal_steps,
            'empty': False
        }
    except Exception as e:
        print(f"  [ERROR] collect_observations env {env_idx}: {e}")
        return None


def send_actions(env, env_idx, behavior_name, decision_steps, actions_for_env):
    """
    Phase 3: Unity에 action 전송 (병렬 실행)
    """
    try:
        action_tuple = ActionTuple(continuous=actions_for_env)
        env.set_actions(behavior_name, action_tuple)
        env.step()
        return True
    except Exception as e:
        print(f"  [ERROR] send_actions env {env_idx}: {e}")
        return False


def setup_environments():
    """Unity 환경 생성 및 연결"""
    envs = []
    channels = []
    behavior_names = []

    print(f"\n[INFO] Creating {NUM_ENVS} Unity environments (headless mode)...", flush=True)
    for i in range(NUM_ENVS):
        channel = EngineConfigurationChannel()
        env = UnityEnvironment(
            file_name=ENV_PATH,  # 빌드된 exe 사용 (병렬 학습)
            side_channels=[channel],
            worker_id=i,
            base_port=BASE_PORT + i,
            timeout_wait=60,
            additional_args=["-batchmode", "-nographics"]  # Headless 모드
        )
        channel.set_configuration_parameters(time_scale=TIME_SCALE)
        env.reset()
        behavior_name = list(env.behavior_specs)[0]

        envs.append(env)
        channels.append(channel)
        behavior_names.append(behavior_name)
        print(f"  Environment {i}: port {BASE_PORT + i} - OK", flush=True)
    
    return envs, channels, behavior_names


def setup_training(policy):
    """학습에 필요한 optimizer, writer, log 파일 설정"""
    # Phase 2: MessageActor에 높은 학습률
    if USE_COMMUNICATION:
        msg_params = list(policy.msg_actor.parameters())
        msg_ids = set(id(p) for p in msg_params)
        other_params = [p for p in policy.parameters() if id(p) not in msg_ids]
        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': LEARNING_RATE},
            {'params': msg_params, 'lr': LEARNING_RATE * MSG_LR_SCALE}
        ])
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    print(f"[OK] Policy Network: {sum(p.numel() for p in policy.parameters())} parameters")

    writer = SummaryWriter(log_dir=os.path.join(SAVE_PATH, 'logs'))
    episode_log_file, training_log_file, message_log_file = setup_logging()

    return optimizer, writer, episode_log_file, training_log_file, message_log_file


def training_step(step, envs, behavior_names, frame_stacks, policy, memory,
                  reward_rms, reward_buffer, stats, interval_stats,
                  agent_rewards, agent_episode_steps):
    """한 스텝의 obs 수집 + 추론 + action 전송 + terminal 처리 (환경 I/O 병렬)"""
    total_agents = 0
    last_env_actions = None

    # ========== Phase 1: 환경에서 observation 수집 (병렬) ==========
    env_results = [None] * NUM_ENVS

    def _collect(env_idx):
        env_results[env_idx] = collect_observations(
            envs[env_idx], env_idx, behavior_names[env_idx],
            frame_stacks[env_idx]
        )

    threads = [threading.Thread(target=_collect, args=(i,)) for i in range(NUM_ENVS)]
    for t in threads: t.start()
    for t in threads: t.join()

    env_data_list = [r for r in env_results if r is not None]

    # ========== Phase 2: 환경별 분리 추론 (통신 파트너 적용) ==========
    actions_per_env = {}

    for env_data in env_data_list:
        if env_data['empty']:
            continue

        env_idx = env_data['env_idx']
        n = env_data['n_agents']
        total_agents += n

        # 환경별로 텐서 생성
        states_tensor = torch.as_tensor(np.array(env_data['batch_states']), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        goals_tensor = torch.as_tensor(np.array(env_data['batch_goals']), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        self_states_tensor = torch.as_tensor(np.array(env_data['batch_self_states']), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        colregs_tensor = torch.as_tensor(np.array(env_data['batch_colregs']), dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # ★ 핵심: comm_partners와 agent_id_list를 forward에 전달, others_msg 반환 ★
        with torch.no_grad():
            values, actions, logprobs, means, _, msg, others_msg = policy.forward(
                states_tensor, goals_tensor, self_states_tensor, colregs_tensor,
                return_msg=True,
                comm_partners=env_data['comm_partners'],
                agent_id_list=env_data['agent_id_list']
            )

        env_actions = np.asarray(actions.squeeze(0).cpu().detach())
        env_values = np.asarray(values.squeeze(0).cpu().detach())
        env_logprobs = np.asarray(logprobs.squeeze(0).cpu().detach())
        env_others_msgs = np.asarray(others_msg.squeeze(0).cpu().detach())
        last_env_actions = env_actions  # 로깅용

        # agent_id -> index 매핑 (O(1) 검색용)
        agent_id_to_idx = {aid: i for i, aid in enumerate(env_data['agent_id_list'])}

        # 경험 저장 (others_msg 포함)
        for i in range(n):
            global_id = env_data['global_id_list'][i]
            raw_reward = env_data['rewards'][i]

            reward_buffer.append(raw_reward)
            normalized_reward = raw_reward / (np.sqrt(reward_rms.var) + 1e-8)

            if global_id not in agent_rewards:
                agent_rewards[global_id] = 0
                agent_episode_steps[global_id] = 0
            agent_rewards[global_id] += raw_reward
            agent_episode_steps[global_id] += 1
            stats['total_reward'] += raw_reward
            interval_stats['reward_sum'] += raw_reward
            interval_stats['reward_count'] += 1

            memory.add_agent_experience(
                global_id,
                env_data['batch_states'][i],
                env_data['batch_goals'][i],
                env_data['batch_self_states'][i],
                env_data['batch_colregs'][i],
                env_others_msgs[i],
                env_actions[i],
                normalized_reward,
                False,
                env_values[i, 0],
                env_logprobs[i, 0]
            )

        # Unity로 보낼 action 준비 (dict 사용으로 O(1) 검색)
        decision_steps = env_data['decision_steps']
        full_actions = np.zeros((len(decision_steps.agent_id), CONTINUOUS_ACTION_SIZE))
        for i, agent_id in enumerate(decision_steps.agent_id):
            if agent_id in agent_id_to_idx:
                full_actions[i] = env_actions[agent_id_to_idx[agent_id]]
        actions_per_env[env_idx] = full_actions

        # Terminal steps 처리 (최종 보상 저장 + done=True 설정)
        for t_idx, agent_id in enumerate(env_data['terminal_steps'].agent_id):
            raw_reward = env_data['terminal_steps'].reward[t_idx]
            global_id = f"env{env_idx}_{agent_id}"

            reward_buffer.append(raw_reward)
            normalized_reward = raw_reward / (np.sqrt(reward_rms.var) + 1e-8)

            if global_id in memory.agent_memories:
                memory.agent_memories[global_id].mark_done(normalized_reward)

            if global_id in agent_rewards:
                agent_rewards[global_id] += raw_reward
            else:
                agent_rewards[global_id] = raw_reward

            stats['total_reward'] += raw_reward
            interval_stats['reward_sum'] += raw_reward
            interval_stats['reward_count'] += 1

            interval_stats['terminal_count'] += 1

            if raw_reward > 0:
                stats['success_count'] += 1
                interval_stats['success_count'] += 1
            elif raw_reward < COLLISION_REWARD_THRESHOLD:
                stats['collision_count'] += 1
                interval_stats['collision_count'] += 1
            elif raw_reward < SPINNING_REWARD_THRESHOLD:
                stats['spinning_count'] += 1
                interval_stats['spinning_count'] += 1

            # 에피소드 카운터 리셋
            if global_id in agent_rewards:
                del agent_rewards[global_id]
            if global_id in agent_episode_steps:
                del agent_episode_steps[global_id]

            frame_stacks[env_idx].remove_agent(agent_id)

    # ========== Phase 3: 환경에 action 전송 (병렬) ==========
    def _send(env_data):
        env_idx = env_data['env_idx']
        if env_data['empty']:
            envs[env_idx].step()
        elif env_idx in actions_per_env:
            send_actions(
                envs[env_idx], env_idx, behavior_names[env_idx],
                env_data['decision_steps'], actions_per_env[env_idx]
            )
        else:
            envs[env_idx].step()

    threads = [threading.Thread(target=_send, args=(ed,)) for ed in env_data_list]
    for t in threads: t.start()
    for t in threads: t.join()

    return total_agents, last_env_actions


def log_and_save(step, start_step, total_agents, last_env_actions, policy, optimizer,
                 memory, writer, reward_rms, reward_buffer, stats, interval_stats,
                 episode_log_file, training_log_file, training_start_time=None):
    """통계 출력 + 모델 저장 + tensorboard 로깅"""

    # Reward RMS 업데이트
    if len(reward_buffer) >= 100:
        reward_rms.update(np.array(reward_buffer))
        reward_buffer.clear()

    # Message annealing
    if USE_COMMUNICATION:
        policy.msg_anneal_step = step - start_step

    # PPO 업데이트
    if TRAIN_MODE and step > 0 and step % UPDATE_INTERVAL == 0:
        update_start = time.time()
        print(f"\n[UPDATE] PPO update at step {step}")
        if USE_COMMUNICATION:
            anneal_pct = min(100.0, (step - start_step) / max(MSG_ANNEAL_STEPS, 1) * 100)
            print(f"  Message annealing: {anneal_pct:.1f}%")
        ppo_update(policy, optimizer, memory, writer, step, training_log_file)
        memory.clear()
        elapsed = time.time() - training_start_time
        update_time = time.time() - update_start
        steps_per_sec = step / max(elapsed, 1)
        eta_hours = (RUN_STEP - step) / max(steps_per_sec, 1) / 3600
        print(f"  Update: {update_time:.1f}s | Total: {elapsed/60:.1f}min | {steps_per_sec:.0f} steps/s | ETA: {eta_hours:.1f}h")

    # 간단한 상태 출력 (500 스텝마다)
    if step % 500 == 0:
        if last_env_actions is not None and len(last_env_actions) > 0:
            # action 분포 확인 (rudder, thrust)
            rudder_mean = last_env_actions[:, 0].mean()
            thrust_mean = last_env_actions[:, 1].mean()
            print(f"[STEP {step}] Agents: {total_agents}, Action: rudder={rudder_mean:.3f}, thrust={thrust_mean:.3f}", flush=True)
        else:
            print(f"[STEP {step}] Agents: {total_agents}", flush=True)

    # 진행 상황 출력
    if step > 0 and step % 3000 == 0:
        avg_reward = interval_stats['reward_sum'] / max(interval_stats['reward_count'], 1)
        print(f"[STEP {step:,}/{MAX_STEPS:,}] Envs: {NUM_ENVS}, Agents: {total_agents}, "
              f"Collisions: {interval_stats['collision_count']} (total: {stats['collision_count']}), "
              f"Spinning: {interval_stats['spinning_count']} (total: {stats['spinning_count']}), "
              f"Success: {interval_stats['success_count']} (total: {stats['success_count']}), "
              f"Avg Reward: {avg_reward:.4f}")

        avg_normalized = avg_reward / (np.sqrt(reward_rms.var) + 1e-8)
        terminal_count = interval_stats['terminal_count']
        collision_rate = interval_stats['collision_count'] / max(terminal_count, 1)
        success_rate = interval_stats['success_count'] / max(terminal_count, 1)

        writer.add_scalar('Reward/Step_Raw', avg_reward, step)
        writer.add_scalar('Reward/Step_Normalized', float(avg_normalized), step)
        writer.add_scalar('Reward/RMS_Std', float(np.sqrt(reward_rms.var)), step)
        writer.add_scalar('Collision/Interval', interval_stats['collision_count'], step)
        writer.add_scalar('Collision/Total', stats['collision_count'], step)
        writer.add_scalar('Collision/Rate', collision_rate, step)
        writer.add_scalar('Success/Interval', interval_stats['success_count'], step)
        writer.add_scalar('Success/Total', stats['success_count'], step)
        writer.add_scalar('Success/Rate', success_rate, step)
        writer.add_scalar('Agents/Total', total_agents, step)

        # Episode CSV 로깅
        with open(episode_log_file, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                terminal_count, step, avg_reward, interval_stats['reward_sum'],
                interval_stats['collision_count'], collision_rate,
                interval_stats['success_count'], success_rate,
                interval_stats['reward_count'], total_agents, LEARNING_RATE
            ])

        interval_stats['reward_sum'] = 0
        interval_stats['reward_count'] = 0
        interval_stats['collision_count'] = 0
        interval_stats['spinning_count'] = 0
        interval_stats['success_count'] = 0
        interval_stats['terminal_count'] = 0

    # 모델 저장 (reward_rms 포함)
    if TRAIN_MODE and step > 0 and step % 10000 == 0:
        save_path = os.path.join(SAVE_PATH, f'policy_step_{step}.pth')
        checkpoint = {
            'model_state_dict': policy.state_dict(),
            'msg_anneal_step': getattr(policy, 'msg_anneal_step', 0),
        }
        torch.save(checkpoint, save_path)
        rms_save = save_path.replace('.pth', '_reward_rms.npz')
        np.savez(rms_save, mean=reward_rms.mean, var=reward_rms.var, count=reward_rms.count)
        print(f"  Model saved: {save_path}")
        writer.flush()


def main():
    # 재현성을 위한 random seed 고정
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("=" * 80)
    print("[START] Vessel ML-Agent Training (Multi-Instance)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Number of Environments: {NUM_ENVS}")
    print(f"Observation Size: {OBSERVATION_SIZE}D (per agent)")
    print(f"Message Dimension: {MSG_DIM}D")
    print(f"Communication Range: {COMM_RANGE}m")
    print(f"Max Communication Partners: {MAX_COMM_PARTNERS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Steps: {MAX_STEPS}")
    print("=" * 80)

    # 여러 Unity 환경 생성
    envs, channels, behavior_names = setup_environments()

    # Reward normalization
    reward_rms = RunningMeanStd()
    reward_buffer = []

    # 정책 네트워크
    policy = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES).to(DEVICE)

    if LOAD_MODEL and MODEL_PATH and os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            saved_state = checkpoint['model_state_dict']
            if 'msg_anneal_step' in checkpoint:
                policy.msg_anneal_step = checkpoint['msg_anneal_step']
                print(f"  [OK] Loaded msg_anneal_step = {policy.msg_anneal_step}")
        else:
            saved_state = checkpoint  # 이전 형식 호환
        model_state = policy.state_dict()
        filtered = {k: v for k, v in saved_state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in saved_state if k not in filtered]
        model_state.update(filtered)
        policy.load_state_dict(model_state)
        if skipped:
            print(f"[WARN] Skipped (shape mismatch): {skipped}")
        print(f"[OK] Model loaded: {MODEL_PATH} ({len(filtered)}/{len(saved_state)} layers)")

        # reward_rms 복원
        rms_path = MODEL_PATH.replace('.pth', '_reward_rms.npz')
        if os.path.exists(rms_path):
            rms_data = np.load(rms_path)
            reward_rms.mean = rms_data['mean']
            reward_rms.var = rms_data['var']
            reward_rms.count = float(rms_data['count'])
            print(f"[OK] Reward RMS loaded: std={np.sqrt(reward_rms.var):.4f}")

    optimizer, writer, episode_log_file, training_log_file, message_log_file = setup_training(policy)

    # 학습 설정 기록
    try:
        from config import get_config_dict
        config_dict = get_config_dict()
        config_path = os.path.join(SAVE_PATH, 'config_snapshot.txt')
        with open(config_path, 'w') as f:
            for k, v in config_dict.items():
                f.write(f"{k}: {v}\n")
        print(f"  [OK] Config saved to {config_path}")
    except Exception as e:
        print(f"  [WARN] Config dump failed: {e}")

    memory = Memory()
    frame_stacks = [MultiAgentFrameStack(FRAMES, STATE_SIZE) for _ in range(NUM_ENVS)]

    # 통계
    stats = {
        'collision_count': 0,
        'spinning_count': 0,
        'success_count': 0,
        'total_reward': 0,
    }
    agent_rewards = {}
    agent_episode_steps = {}

    interval_stats = {
        'reward_sum': 0,
        'reward_count': 0,
        'collision_count': 0,
        'spinning_count': 0,
        'success_count': 0,
        'terminal_count': 0,
    }

    # 시작 스텝 설정 (이어서 학습할 때 이전 스텝 누적)
    start_step = START_STEP if LOAD_MODEL else 0
    print(f"\n[TRAINING] Running for {MAX_STEPS} steps (starting from step {start_step:,})...")
    print(f"[INFO] Threaded parallel loop with {NUM_ENVS} environment(s)")

    # 타이밍 측정용 (rolling average, 무한 증가 방지)
    step_times = deque(maxlen=1000)
    training_start_time = time.time()

    for step in range(start_step, start_step + MAX_STEPS):
        step_start = time.time()

        total_agents, last_env_actions = training_step(
            step, envs, behavior_names, frame_stacks, policy, memory,
            reward_rms, reward_buffer, stats, interval_stats,
            agent_rewards, agent_episode_steps
        )

        step_time = time.time() - step_start
        step_times.append(step_time)

        log_and_save(
            step, start_step, total_agents, last_env_actions, policy, optimizer,
            memory, writer, reward_rms, reward_buffer, stats, interval_stats,
            episode_log_file, training_log_file, training_start_time
        )

    # 최종 통계
    print(f"\n{'=' * 80}")
    print(f"[DONE] Training completed!")
    print(f"  Total Steps: {MAX_STEPS}")
    print(f"  Total Environments: {NUM_ENVS}")
    print(f"  Total Collisions: {stats['collision_count']}")
    print(f"  Total Success: {stats['success_count']}")
    print(f"  Avg Reward: {stats['total_reward'] / max(len(agent_rewards), 1):.2f}")
    print(f"{'=' * 80}")

    # 환경 종료
    for env in envs:
        env.close()
    writer.close()


if __name__ == "__main__":
    main()
