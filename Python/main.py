import os
import torch
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import csv
import datetime
import math
import time
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from config import *
from networks import CNNPolicy
from memory import Memory
from functions import calculate_returns
from frame_stack import MultiAgentFrameStack

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


def parse_observation(obs_raw):
    """
    373D observation 파싱 (STATE_SIZE=360 기준):
    [0:360]      Radar (360 rays, 1도 간격)
    [360:362]    Goal (distance, angle)
    [362:364]    Speed (linear, angular)
    [364]        Heading
    [365]        Rudder
    [366:371]    COLREGs (5D one-hot)
    [371:373]    Position (x, z) - 통신용, 학습 제외
    """
    idx = STATE_SIZE  # 360
    state = obs_raw[:idx]

    goal_distance = obs_raw[idx]
    goal_angle = obs_raw[idx + 1]
    speed = obs_raw[idx + 2]
    yaw_rate = obs_raw[idx + 3]
    heading = obs_raw[idx + 4]
    rudder = obs_raw[idx + 5]

    goal = np.array([goal_distance, goal_angle])
    self_state = np.array([speed, yaw_rate, heading, rudder])  # 4D: speed, yaw_rate, heading, rudder
    colregs = obs_raw[idx + 6:idx + 11]
    position = obs_raw[idx + 11:idx + 13]  # x, z (통신 범위 계산용)

    return state, goal, self_state, colregs, position


def get_comm_partners(my_id, my_pos, all_positions):
    """
    통신 범위 내 가장 가까운 파트너들 반환 (같은 환경 내에서만)
    """
    distances = []
    for other_id, other_pos in all_positions.items():
        if other_id == my_id:
            continue
        dist = np.sqrt((my_pos[0] - other_pos[0])**2 + (my_pos[1] - other_pos[1])**2)
        if dist <= COMM_RANGE:
            distances.append((other_id, dist))

    distances.sort(key=lambda x: x[1])
    partners = [d[0] for d in distances[:MAX_COMM_PARTNERS]]

    return partners


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
    states_tensor = torch.FloatTensor(states).unsqueeze(1).to(DEVICE)
    goals_tensor = torch.FloatTensor(experiences['goals']).unsqueeze(1).to(DEVICE)
    self_states_tensor = torch.FloatTensor(experiences['self_states']).unsqueeze(1).to(DEVICE)
    colregs_tensor = torch.FloatTensor(experiences['colregs']).unsqueeze(1).to(DEVICE)
    others_msgs_tensor = torch.FloatTensor(experiences['others_msgs']).unsqueeze(1).to(DEVICE)  # ★ others_msg 추가 ★
    actions_tensor = torch.FloatTensor(experiences['actions']).unsqueeze(1).to(DEVICE)
    old_logprobs_tensor = torch.FloatTensor(experiences['logprobs']).to(DEVICE)
    returns_tensor = torch.FloatTensor(returns).to(DEVICE)
    advantages_tensor = torch.FloatTensor(advantages).to(DEVICE)

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy_loss = 0
    total_colregs_loss = 0
    total_loss = 0
    num_updates = 0

    for epoch in range(N_EPOCH):
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_samples)
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

            loss = policy_loss + VALUE_LOSS_COEF * value_loss + entropy_loss + 0.1 * colregs_loss

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
            state, goal, self_state, colregs, position = parse_observation(obs_raw)

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


def main():
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
    envs = []
    channels = []
    behavior_names = []

    print(f"\n[INFO] Creating {NUM_ENVS} Unity environments (headless mode)...", flush=True)
    for i in range(NUM_ENVS):
        print(f"  [DEBUG] Creating environment {i}...", flush=True)
        channel = EngineConfigurationChannel()
        print(f"  [DEBUG] Connecting to port {BASE_PORT + i}...", flush=True)
        env = UnityEnvironment(
            file_name=None,  # None = Unity 에디터 연결 (테스트용)
            # file_name=ENV_PATH,  # 빌드된 exe 사용시
            side_channels=[channel],
            worker_id=i,
            base_port=BASE_PORT + i,
            timeout_wait=30,  # 30초 timeout (디버그용)
            # additional_args=["-batchmode", "-nographics"]  # Headless 모드 (임시 비활성화)
        )
        print(f"  [DEBUG] Connected! Resetting...", flush=True)
        channel.set_configuration_parameters(time_scale=TIME_SCALE)
        env.reset()
        print(f"  [DEBUG] Reset done!", flush=True)
        behavior_name = list(env.behavior_specs)[0]

        envs.append(env)
        channels.append(channel)
        behavior_names.append(behavior_name)
        print(f"  Environment {i}: port {BASE_PORT + i} - OK", flush=True)

    # 정책 네트워크
    policy = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES).to(DEVICE)

    if LOAD_MODEL and MODEL_PATH and os.path.exists(MODEL_PATH):
        policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"[OK] Model loaded: {MODEL_PATH}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    print(f"[OK] Policy Network: {sum(p.numel() for p in policy.parameters())} parameters")

    writer = SummaryWriter(log_dir=os.path.join(SAVE_PATH, 'logs'))
    episode_log_file, training_log_file, message_log_file = setup_logging()

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
    agent_episode_steps = {}  # 에피소드별 스텝 카운트

    interval_stats = {
        'reward_sum': 0,
        'reward_count': 0,
        'collision_count': 0,
        'spinning_count': 0,
        'success_count': 0,
    }

    # 시작 스텝 설정 (이어서 학습할 때 이전 스텝 누적)
    start_step = START_STEP if LOAD_MODEL else 0
    print(f"\n[TRAINING] Running for {MAX_STEPS} steps (starting from step {start_step:,})...")
    print(f"[INFO] Single-threaded loop with {NUM_ENVS} environment(s)")

    # 타이밍 측정용
    step_times = []

    for step in range(start_step, start_step + MAX_STEPS):
        step_start = time.time()
        total_agents = 0

        # ========== Phase 1: 환경에서 observation 수집 ==========
        t1 = time.time()
        env_data_list = []
        for env_idx in range(NUM_ENVS):
            result = collect_observations(
                envs[env_idx], env_idx, behavior_names[env_idx],
                frame_stacks[env_idx]
            )
            if result is not None:
                env_data_list.append(result)
        t1_end = time.time()

        # ========== Phase 2: 환경별 분리 추론 (통신 파트너 적용) ==========
        t2 = time.time()
        actions_per_env = {}
        last_env_actions = None  # 로깅용

        for env_data in env_data_list:
            if env_data['empty']:
                continue

            env_idx = env_data['env_idx']
            n = env_data['n_agents']
            total_agents += n

            # 환경별로 텐서 생성
            states_tensor = torch.FloatTensor(np.array(env_data['batch_states'])).unsqueeze(0).to(DEVICE)
            goals_tensor = torch.FloatTensor(np.array(env_data['batch_goals'])).unsqueeze(0).to(DEVICE)
            self_states_tensor = torch.FloatTensor(np.array(env_data['batch_self_states'])).unsqueeze(0).to(DEVICE)
            colregs_tensor = torch.FloatTensor(np.array(env_data['batch_colregs'])).unsqueeze(0).to(DEVICE)

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
            env_msgs = np.asarray(msg.squeeze(0).cpu().detach())
            last_env_actions = env_actions  # 로깅용

            # 메시지 로그 기록 (10000 스텝마다)
            if step % 10000 == 0:
                with open(message_log_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    for i in range(n):
                        colregs_idx = int(np.argmax(env_data['batch_colregs'][i]))
                        csv_writer.writerow([
                            step, env_data['global_id_list'][i],
                            *env_msgs[i].tolist(),
                            *env_others_msgs[i].tolist(),
                            n, colregs_idx
                        ])

            # agent_id -> index 매핑 (O(1) 검색용)
            agent_id_to_idx = {aid: i for i, aid in enumerate(env_data['agent_id_list'])}

            # 경험 저장 (others_msg 포함)
            for i in range(n):
                global_id = env_data['global_id_list'][i]
                reward = env_data['rewards'][i]

                if global_id not in agent_rewards:
                    agent_rewards[global_id] = 0
                    agent_episode_steps[global_id] = 0
                agent_rewards[global_id] += reward
                agent_episode_steps[global_id] += 1
                stats['total_reward'] += reward
                interval_stats['reward_sum'] += reward
                interval_stats['reward_count'] += 1

                memory.add_agent_experience(
                    global_id,
                    env_data['batch_states'][i],
                    env_data['batch_goals'][i],
                    env_data['batch_self_states'][i],
                    env_data['batch_colregs'][i],
                    env_others_msgs[i],  # ★ others_msg 저장 ★
                    env_actions[i],
                    reward,
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
                reward = env_data['terminal_steps'].reward[t_idx]
                global_id = f"env{env_idx}_{agent_id}"

                # 마지막 경험에 최종 보상 추가 + done=True
                if global_id in memory.agent_memories:
                    memory.agent_memories[global_id].mark_done(reward)

                if global_id in agent_rewards:
                    agent_rewards[global_id] += reward
                else:
                    agent_rewards[global_id] = reward

                stats['total_reward'] += reward
                interval_stats['reward_sum'] += reward
                interval_stats['reward_count'] += 1

                # 성공/충돌/스피닝 분류 (spinningPenalty=-80, collisionPenalty=-100)
                if reward > 0:
                    stats['success_count'] += 1
                    interval_stats['success_count'] += 1
                elif reward < -90:  # collisionPenalty = -100
                    stats['collision_count'] += 1
                    interval_stats['collision_count'] += 1
                elif reward < -50:  # spinningPenalty = -80
                    stats['spinning_count'] += 1
                    interval_stats['spinning_count'] += 1

                # Episode 로그 기록
                ep_steps = agent_episode_steps.get(global_id, 0)
                ep_reward = agent_rewards.get(global_id, 0)
                with open(episode_log_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([
                        0, step, ep_reward / max(ep_steps, 1), ep_reward,
                        1 if reward < -90 else 0,
                        1.0 if reward < -90 else 0.0,
                        1 if reward > 0 else 0,
                        1.0 if reward > 0 else 0.0,
                        ep_steps, n, LEARNING_RATE
                    ])

                # 에피소드 종료된 에이전트 카운터 리셋
                if global_id in agent_rewards:
                    del agent_rewards[global_id]
                if global_id in agent_episode_steps:
                    del agent_episode_steps[global_id]

                frame_stacks[env_idx].remove_agent(agent_id)

        t2_end = time.time()

        # ========== Phase 3: 환경에 action 전송 ==========
        t3 = time.time()
        for env_data in env_data_list:
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
        t3_end = time.time()

        step_time = time.time() - step_start
        step_times.append(step_time)

        # PPO 업데이트
        if TRAIN_MODE and step > 0 and step % UPDATE_INTERVAL == 0:
            print(f"\n[UPDATE] PPO update at step {step}")
            ppo_update(policy, optimizer, memory, writer, step, training_log_file)
            memory.clear()

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

            writer.add_scalar('Reward/Step', avg_reward, step)
            writer.add_scalar('Collision/Interval', interval_stats['collision_count'], step)
            writer.add_scalar('Collision/Total', stats['collision_count'], step)
            writer.add_scalar('Spinning/Interval', interval_stats['spinning_count'], step)
            writer.add_scalar('Spinning/Total', stats['spinning_count'], step)
            writer.add_scalar('Success/Interval', interval_stats['success_count'], step)
            writer.add_scalar('Success/Total', stats['success_count'], step)
            writer.add_scalar('Agents/Total', total_agents, step)

            interval_stats['reward_sum'] = 0
            interval_stats['reward_count'] = 0
            interval_stats['collision_count'] = 0
            interval_stats['spinning_count'] = 0
            interval_stats['success_count'] = 0

        # 모델 저장
        if TRAIN_MODE and step > 0 and step % 10000 == 0:
            save_path = os.path.join(SAVE_PATH, f'policy_step_{step}.pth')
            torch.save(policy.state_dict(), save_path)
            print(f"  Model saved: {save_path}")

    # 최종 통계
    print(f"\n{'=' * 80}")
    print(f"[DONE] Training completed!")
    print(f"  Total Steps: {MAX_STEPS}")
    print(f"  Total Environments: {NUM_ENVS}")
    print(f"  Total Collisions: {stats['collision_count']}")
    print(f"  Total Spinning: {stats['spinning_count']}")
    print(f"  Total Success: {stats['success_count']}")
    print(f"  Avg Reward: {stats['total_reward'] / max(len(agent_rewards), 1):.2f}")
    print(f"{'=' * 80}")

    # 환경 종료ㅇ
    for env in envs:
        env.close()
    writer.close()


if __name__ == "__main__":
    main()
