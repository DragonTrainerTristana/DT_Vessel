"""
Vessel Navigation Model Test (Multi-Run Evaluation)
50회 반복 테스트 + Observation vs Message t-SNE 분석
+ 간단한 trajectory 수집 모드 추가 (2025-01-13)
"""

import os
import torch
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import argparse
import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from config import *
from networks import CNNPolicy
from frame_stack import MultiAgentFrameStack
from obs_utils import parse_observation, get_comm_partners

# ============================================================================
# Test Configuration
# ============================================================================
TEST_STEPS = 2000               # 테스트할 총 스텝 수 (per run)
NUM_RUNS = 10                   # 반복 횟수
TEST_TIME_SCALE = 20.0          # 테스트 시 시뮬레이션 속도 (main.py와 동일)
COLLECT_INTERVAL = 10           # 메시지 수집 간격 (매 N step마다)

# Trajectory 수집용 설정
TRAJECTORY_STEPS = 20000        # trajectory 수집 스텝 수
TRAJECTORY_TIME_SCALE = 20.0    # trajectory 수집 시 시뮬레이션 속도 (20배속)

# Observation 크기 (frame stacking 전)
OBS_SIZE = STATE_SIZE + GOAL_SIZE + SELF_STATE_SIZE + COLREGS_SIZE  # 360 + 2 + 4 + 5 = 371D

# 모델 경로 설정
MODEL_PATH_COMM_OFF = os.path.join(PROJECT_ROOT, "models", "COMM_NON", "VesselNavigation_20260419_194205", "policy_step_3220000.pth")
# 통신 모델: Phase 2 v2 16.67M 스텝 (narrow/coastal 실험 메인 모델)
MODEL_PATH_COMM_ON = os.path.join(PROJECT_ROOT, "models", "COMM_YES_PHASE2_v2", "VesselNavigation_20260310_171041", "policy_step_16670000.pth")

# USE_COMMUNICATION에 따라 자동 선택
TEST_MODEL_PATH = MODEL_PATH_COMM_ON if USE_COMMUNICATION else MODEL_PATH_COMM_OFF



# parse_observation → obs_utils.py로 이동


def analyze_tsne_with_observation(csv_path, save_dir=None):
    """
    t-SNE 분석: Observation vs Message 비교 + Action 관계
    """
    print("\n" + "=" * 80)
    print("[t-SNE ANALYSIS] Observation vs Message Comparison")
    print("=" * 80)

    # CSV 로드
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    # 컬럼 추출
    obs_cols = [f'obs_{i}' for i in range(OBS_SIZE)]
    self_msg_cols = [f'self_msg_{i}' for i in range(MSG_DIM)]
    others_msg_cols = [f'others_msg_{i}' for i in range(MSG_DIM)]

    obs_data = df[obs_cols].values  # [N, OBS_SIZE]
    self_msg = df[self_msg_cols].values  # [N, 6]
    others_msg = df[others_msg_cols].values  # [N, 6]
    action_0 = df['action_0'].values  # Throttle
    action_1 = df['action_1'].values  # Rudder
    colregs = df['colregs'].values  # COLREGs 상황

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # COLREGs 라벨
    colregs_labels = ['None', 'HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']
    colregs_colors = ['gray', 'red', 'blue', 'green', 'orange']

    # =========================================================================
    # Figure 1: Observation vs Self Message t-SNE (COLREGs 색상)
    # =========================================================================
    print(f"\n[INFO] Running t-SNE on Observation ({OBS_SIZE}D -> 2D)...")
    tsne_obs = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    obs_2d = tsne_obs.fit_transform(obs_data)

    print(f"[INFO] Running t-SNE on Self Message ({MSG_DIM}D -> 2D)...")
    tsne_msg = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    msg_2d = tsne_msg.fit_transform(self_msg)

    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle(f'Observation ({OBS_SIZE}D) vs Self Message ({MSG_DIM}D) - COLREGs Colored', fontsize=16, fontweight='bold')

    for i, label in enumerate(colregs_labels):
        mask = colregs == i
        if mask.sum() > 0:
            axes1[0].scatter(obs_2d[mask, 0], obs_2d[mask, 1],
                           c=colregs_colors[i], label=label, alpha=0.6, s=15)
            axes1[1].scatter(msg_2d[mask, 0], msg_2d[mask, 1],
                           c=colregs_colors[i], label=label, alpha=0.6, s=15)

    axes1[0].set_title(f'Observation t-SNE ({OBS_SIZE}D → 2D)')
    axes1[0].set_xlabel('t-SNE 1')
    axes1[0].set_ylabel('t-SNE 2')
    axes1[0].legend()

    axes1[1].set_title(f'Self Message t-SNE ({MSG_DIM}D → 2D)')
    axes1[1].set_xlabel('t-SNE 1')
    axes1[1].set_ylabel('t-SNE 2')
    axes1[1].legend()

    plt.tight_layout()
    fig1_path = os.path.join(save_dir, f'tsne_obs_vs_msg_colregs_{timestamp}.png')
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {fig1_path}")

    # =========================================================================
    # Figure 2: Observation vs Message colored by Action
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle('Observation vs Message → Action Relationship', fontsize=16, fontweight='bold')

    sc1 = axes2[0, 0].scatter(obs_2d[:, 0], obs_2d[:, 1], c=action_0, cmap='RdYlBu_r', alpha=0.6, s=15)
    axes2[0, 0].set_title('Observation → Throttle')
    axes2[0, 0].set_xlabel('t-SNE 1')
    axes2[0, 0].set_ylabel('t-SNE 2')
    plt.colorbar(sc1, ax=axes2[0, 0], label='Throttle')

    sc2 = axes2[0, 1].scatter(obs_2d[:, 0], obs_2d[:, 1], c=action_1, cmap='RdYlBu_r', alpha=0.6, s=15)
    axes2[0, 1].set_title('Observation → Rudder')
    axes2[0, 1].set_xlabel('t-SNE 1')
    axes2[0, 1].set_ylabel('t-SNE 2')
    plt.colorbar(sc2, ax=axes2[0, 1], label='Rudder')

    sc3 = axes2[1, 0].scatter(msg_2d[:, 0], msg_2d[:, 1], c=action_0, cmap='RdYlBu_r', alpha=0.6, s=15)
    axes2[1, 0].set_title('Self Message → Throttle')
    axes2[1, 0].set_xlabel('t-SNE 1')
    axes2[1, 0].set_ylabel('t-SNE 2')
    plt.colorbar(sc3, ax=axes2[1, 0], label='Throttle')

    sc4 = axes2[1, 1].scatter(msg_2d[:, 0], msg_2d[:, 1], c=action_1, cmap='RdYlBu_r', alpha=0.6, s=15)
    axes2[1, 1].set_title('Self Message → Rudder')
    axes2[1, 1].set_xlabel('t-SNE 1')
    axes2[1, 1].set_ylabel('t-SNE 2')
    plt.colorbar(sc4, ax=axes2[1, 1], label='Rudder')

    plt.tight_layout()
    fig2_path = os.path.join(save_dir, f'tsne_obs_msg_action_{timestamp}.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {fig2_path}")

    # =========================================================================
    # Figure 3: Others Message t-SNE
    # =========================================================================
    print(f"[INFO] Running t-SNE on Others Message ({MSG_DIM}D -> 2D)...")
    tsne_others = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    others_msg_2d = tsne_others.fit_transform(others_msg)

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle('Others Message (6D) t-SNE → My Action Relationship', fontsize=16, fontweight='bold')

    sc5 = axes3[0].scatter(others_msg_2d[:, 0], others_msg_2d[:, 1], c=action_0, cmap='RdYlBu_r', alpha=0.6, s=15)
    axes3[0].set_title('Others Message → My Throttle')
    axes3[0].set_xlabel('t-SNE 1')
    axes3[0].set_ylabel('t-SNE 2')
    plt.colorbar(sc5, ax=axes3[0], label='Throttle')

    sc6 = axes3[1].scatter(others_msg_2d[:, 0], others_msg_2d[:, 1], c=action_1, cmap='RdYlBu_r', alpha=0.6, s=15)
    axes3[1].set_title('Others Message → My Rudder')
    axes3[1].set_xlabel('t-SNE 1')
    axes3[1].set_ylabel('t-SNE 2')
    plt.colorbar(sc6, ax=axes3[1], label='Rudder')

    plt.tight_layout()
    fig3_path = os.path.join(save_dir, f'tsne_others_msg_action_{timestamp}.png')
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {fig3_path}")

    # =========================================================================
    # Figure 4: 각 메시지 차원별 t-SNE (Self Message)
    # =========================================================================
    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
    fig4.suptitle('Self Message: Each Dimension Visualization (t-SNE)', fontsize=16, fontweight='bold')

    for i in range(MSG_DIM):
        ax = axes4[i // 3, i % 3]
        sc = ax.scatter(msg_2d[:, 0], msg_2d[:, 1], c=self_msg[:, i], cmap='viridis', alpha=0.6, s=15)
        ax.set_title(f'self_msg_{i}')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.colorbar(sc, ax=ax, label=f'dim {i}')

    plt.tight_layout()
    fig4_path = os.path.join(save_dir, f'tsne_self_msg_dims_{timestamp}.png')
    plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {fig4_path}")

    # =========================================================================
    # Figure 5: 상관관계 히트맵 (Observation, Self Msg, Others Msg vs Action)
    # =========================================================================
    fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5))
    fig5.suptitle('Correlation: Features vs Actions', fontsize=16, fontweight='bold')

    # Self Message 상관관계
    self_corr = np.zeros((MSG_DIM, 2))
    for i in range(MSG_DIM):
        self_corr[i, 0] = np.corrcoef(self_msg[:, i], action_0)[0, 1]
        self_corr[i, 1] = np.corrcoef(self_msg[:, i], action_1)[0, 1]

    im1 = axes5[0].imshow(self_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes5[0].set_title('Self Message vs Action')
    axes5[0].set_xlabel('Action')
    axes5[0].set_ylabel('Message Dimension')
    axes5[0].set_xticks([0, 1])
    axes5[0].set_xticklabels(['Throttle', 'Rudder'])
    axes5[0].set_yticks(range(MSG_DIM))
    axes5[0].set_yticklabels([f'dim_{i}' for i in range(MSG_DIM)])
    for i in range(MSG_DIM):
        for j in range(2):
            axes5[0].text(j, i, f'{self_corr[i, j]:.2f}', ha='center', va='center', fontsize=10)
    plt.colorbar(im1, ax=axes5[0], label='Correlation')

    # Others Message 상관관계
    others_corr = np.zeros((MSG_DIM, 2))
    for i in range(MSG_DIM):
        others_corr[i, 0] = np.corrcoef(others_msg[:, i], action_0)[0, 1]
        others_corr[i, 1] = np.corrcoef(others_msg[:, i], action_1)[0, 1]

    im2 = axes5[1].imshow(others_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes5[1].set_title('Others Message vs My Action')
    axes5[1].set_xlabel('Action')
    axes5[1].set_ylabel('Message Dimension')
    axes5[1].set_xticks([0, 1])
    axes5[1].set_xticklabels(['Throttle', 'Rudder'])
    axes5[1].set_yticks(range(MSG_DIM))
    axes5[1].set_yticklabels([f'dim_{i}' for i in range(MSG_DIM)])
    for i in range(MSG_DIM):
        for j in range(2):
            axes5[1].text(j, i, f'{others_corr[i, j]:.2f}', ha='center', va='center', fontsize=10)
    plt.colorbar(im2, ax=axes5[1], label='Correlation')

    # Combined Message 상관관계 (self + others)
    combined_msg = np.concatenate([self_msg, others_msg], axis=1)  # [N, 12]
    combined_corr = np.zeros((12, 2))
    for i in range(12):
        combined_corr[i, 0] = np.corrcoef(combined_msg[:, i], action_0)[0, 1]
        combined_corr[i, 1] = np.corrcoef(combined_msg[:, i], action_1)[0, 1]

    im3 = axes5[2].imshow(combined_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes5[2].set_title('Combined (Self+Others) vs Action')
    axes5[2].set_xlabel('Action')
    axes5[2].set_ylabel('Message Dimension')
    axes5[2].set_xticks([0, 1])
    axes5[2].set_xticklabels(['Throttle', 'Rudder'])
    axes5[2].set_yticks(range(12))
    labels = [f'self_{i}' for i in range(6)] + [f'others_{i}' for i in range(6)]
    axes5[2].set_yticklabels(labels, fontsize=8)
    for i in range(12):
        for j in range(2):
            axes5[2].text(j, i, f'{combined_corr[i, j]:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im3, ax=axes5[2], label='Correlation')

    plt.tight_layout()
    fig5_path = os.path.join(save_dir, f'correlation_all_{timestamp}.png')
    plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {fig5_path}")

    # =========================================================================
    # Figure 6: Observation 정보 압축 분석 (Obs 유사도 vs Msg 유사도)
    # =========================================================================
    print("[INFO] Computing pairwise similarity analysis...")

    # 샘플링 (전체 하면 너무 오래 걸림)
    n_samples = min(1000, len(obs_data))
    indices = np.random.choice(len(obs_data), n_samples, replace=False)

    obs_sample = obs_data[indices]
    msg_sample = self_msg[indices]

    # 코사인 유사도 계산
    from sklearn.metrics.pairwise import cosine_similarity
    obs_sim = cosine_similarity(obs_sample)
    msg_sim = cosine_similarity(msg_sample)

    # 상삼각 행렬만 추출 (자기 자신 제외)
    triu_indices = np.triu_indices(n_samples, k=1)
    obs_sim_flat = obs_sim[triu_indices]
    msg_sim_flat = msg_sim[triu_indices]

    fig6, ax6 = plt.subplots(1, 1, figsize=(8, 8))
    ax6.scatter(obs_sim_flat, msg_sim_flat, alpha=0.3, s=5)
    ax6.set_xlabel('Observation Similarity (cosine)')
    ax6.set_ylabel('Message Similarity (cosine)')
    ax6.set_title(f'Information Preservation: Obs vs Msg\n(Correlation: {np.corrcoef(obs_sim_flat, msg_sim_flat)[0,1]:.3f})')
    ax6.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect preservation')
    ax6.legend()

    plt.tight_layout()
    fig6_path = os.path.join(save_dir, f'info_preservation_{timestamp}.png')
    plt.savefig(fig6_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {fig6_path}")

    plt.show()

    print("\n[t-SNE ANALYSIS COMPLETE]")
    print(f"  - Obs vs Msg (COLREGs): {fig1_path}")
    print(f"  - Obs vs Msg (Action): {fig2_path}")
    print(f"  - Others Msg (Action): {fig3_path}")
    print(f"  - Self Msg Dims: {fig4_path}")
    print(f"  - Correlation Heatmap: {fig5_path}")
    print(f"  - Info Preservation: {fig6_path}")

    return [fig1_path, fig2_path, fig3_path, fig4_path, fig5_path, fig6_path]



# get_comm_partners → obs_utils.py로 이동


def run_single_test(env, policy, frame_stack, behavior_name, test_steps, collect_interval, use_comm=False):
    """단일 테스트 실행 (1 run)"""

    colregs_labels = ['None', 'HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']

    stats = {
        'collision_count': 0,
        'success_count': 0,
        'total_reward': 0,
    }
    agent_rewards = {}
    collected_data = []
    trajectory_data = []

    # 환경 리셋
    env.reset()

    for step in range(test_steps):
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        agent_ids = list(decision_steps.agent_id)
        n_agents = len(agent_ids)

        if n_agents == 0:
            env.step()
            continue

        # 배치 데이터 준비
        batch_states = []
        batch_goals = []
        batch_self_states = []
        batch_colregs = []
        batch_obs_full = []
        batch_positions = {}
        agent_id_list = []

        for idx, agent_id in enumerate(agent_ids):
            obs_raw = decision_steps.obs[0][idx]
            state, goal, self_state, colregs, obs_full, position = parse_observation(obs_raw)

            state_stacked = frame_stack.update(agent_id, state)

            batch_states.append(state_stacked)
            batch_goals.append(goal)
            batch_self_states.append(self_state)
            batch_colregs.append(colregs)
            batch_obs_full.append(obs_full)
            batch_positions[agent_id] = position
            agent_id_list.append(agent_id)

            if agent_id not in agent_rewards:
                agent_rewards[agent_id] = 0

        # 통신 파트너 계산
        comm_partners = {}
        if use_comm:
            for agent_id in agent_id_list:
                comm_partners[agent_id] = get_comm_partners(agent_id, batch_positions[agent_id], batch_positions)

        # Tensor 변환
        states_tensor = torch.FloatTensor(np.array(batch_states)).unsqueeze(0).to(DEVICE)
        goals_tensor = torch.FloatTensor(np.array(batch_goals)).unsqueeze(0).to(DEVICE)
        self_states_tensor = torch.FloatTensor(np.array(batch_self_states)).unsqueeze(0).to(DEVICE)
        colregs_tensor = torch.FloatTensor(np.array(batch_colregs)).unsqueeze(0).to(DEVICE)

        # 행동 결정 + 메시지 수집
        collect_msg = (step % collect_interval == 0)

        with torch.no_grad():
            if collect_msg and use_comm:
                values, actions, logprobs, means, _, msg, others_msg = policy.forward(
                    states_tensor, goals_tensor, self_states_tensor, colregs_tensor,
                    return_msg=True,
                    comm_partners=comm_partners,
                    agent_id_list=agent_id_list
                )
                msg_np = msg.squeeze(0).cpu().numpy()
                others_msg_np = others_msg.squeeze(0).cpu().numpy()

                for i, agent_id in enumerate(agent_id_list):
                    colregs_idx = np.argmax(batch_colregs[i])
                    collected_data.append({
                        'step': step,
                        'agent_id': agent_id,
                        'obs_full': batch_obs_full[i].tolist(),
                        'self_msg': msg_np[i].tolist(),
                        'others_msg': others_msg_np[i].tolist(),
                        'colregs': int(colregs_idx),
                        'action': actions.squeeze(0)[i].cpu().numpy().tolist(),
                        'n_agents': n_agents
                    })
            elif use_comm:
                values, actions, logprobs, means, _ = policy.forward(
                    states_tensor, goals_tensor, self_states_tensor, colregs_tensor,
                    comm_partners=comm_partners,
                    agent_id_list=agent_id_list
                )
            else:
                values, actions, logprobs, means, _ = policy.forward(
                    states_tensor, goals_tensor, self_states_tensor, colregs_tensor
                )

        actions_np = actions.squeeze(0).cpu().numpy()

        # Trajectory 데이터 수집 (매 스텝)
        for i, agent_id in enumerate(agent_id_list):
            colregs_idx = int(np.argmax(batch_colregs[i]))
            pos = batch_positions[agent_id]
            trajectory_data.append({
                'step': step,
                'agent_id': agent_id,
                'colregs': colregs_idx,
                'colregs_name': colregs_labels[colregs_idx],
                'x': float(pos[0]),
                'z': float(pos[1]),
                'speed': float(batch_self_states[i][0]),
                'heading': float(batch_self_states[i][2]),
                'rudder': float(batch_self_states[i][3]),
                'goal_dist': float(batch_goals[i][0]),
                'goal_angle': float(batch_goals[i][1]),
            })

        # reward 누적
        for i, agent_id in enumerate(agent_id_list):
            idx = decision_steps.agent_id.tolist().index(agent_id)
            reward = decision_steps.reward[idx]
            agent_rewards[agent_id] += reward
            stats['total_reward'] += reward

        # Terminal steps 처리
        for idx, agent_id in enumerate(terminal_steps.agent_id):
            reward = terminal_steps.reward[idx]

            if agent_id in agent_rewards:
                agent_rewards[agent_id] += reward
            else:
                agent_rewards[agent_id] = reward

            stats['total_reward'] += reward

            if reward > 0:
                stats['success_count'] += 1
            elif reward < -50:
                stats['collision_count'] += 1

            frame_stack.remove_agent(agent_id)

        # Unity에 행동 전송
        all_actions = np.zeros((len(decision_steps.agent_id), CONTINUOUS_ACTION_SIZE))
        for i, agent_id in enumerate(decision_steps.agent_id):
            if agent_id in agent_id_list:
                agent_idx = agent_id_list.index(agent_id)
                all_actions[i] = actions_np[agent_idx]

        action_tuple = ActionTuple(continuous=all_actions)

        try:
            env.set_actions(behavior_name, action_tuple)
            env.step()
        except Exception as e:
            print(f"  [WARNING] Unity error: {e}")
            break

    # 평균 리워드 계산
    avg_reward = stats['total_reward'] / max(len(agent_rewards), 1)
    stats['avg_reward'] = avg_reward

    return stats, collected_data, trajectory_data


def run_multi_test(model_path=None, test_steps=TEST_STEPS, num_runs=NUM_RUNS, time_scale=TEST_TIME_SCALE):
    """다중 테스트 실행 (50회 반복 + 평균)"""
    print("=" * 80)
    print(f"[MULTI-RUN TEST] {num_runs} runs × {test_steps} steps = {num_runs * test_steps} total steps")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Message Dimension: {MSG_DIM}D")
    print(f"Observation Size: {OBS_SIZE}D")
    print("=" * 80)

    # 모델 경로 확인
    if model_path is None:
        model_path = TEST_MODEL_PATH

    if model_path is None or not os.path.exists(model_path):
        print("\n[ERROR] Model path not set or file not found!")
        print("Please set TEST_MODEL_PATH or pass --model argument.")
        return None

    # Unity 환경 연결
    print("\n[INFO] Connecting to Unity environment...")
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name=None,  # Unity 에디터 사용
        side_channels=[channel],
        base_port=BASE_PORT
    )
    channel.set_configuration_parameters(time_scale=time_scale)
    print("[OK] Unity environment connected")

    # 모델 로드
    print(f"\n[INFO] Loading model from: {model_path}")
    policy = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES).to(DEVICE)
    policy.load_state_dict(torch.load(model_path, map_location=DEVICE))
    policy.eval()
    print(f"[OK] Model loaded ({sum(p.numel() for p in policy.parameters())} parameters)")

    env.reset()
    behavior_name = list(env.behavior_specs)[0]

    # 결과 저장
    all_stats = []
    all_collected_data = []

    # 저장 경로 미리 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PROJECT_ROOT, "latent_data")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"multirun_{num_runs}x{test_steps}_{timestamp}.csv")

    print(f"\n[TESTING] Running {num_runs} tests...")
    print(f"[INFO] Data will be saved to: {csv_path}")

    try:
        for run in range(num_runs):
            frame_stack = MultiAgentFrameStack(FRAMES, STATE_SIZE)

            stats, collected_data, _ = run_single_test(
                env, policy, frame_stack, behavior_name, test_steps, COLLECT_INTERVAL
            )

            all_stats.append(stats)
            all_collected_data.extend(collected_data)

            print(f"  [Run {run + 1:2d}/{num_runs}] Collisions: {stats['collision_count']:3d}, "
                  f"Success: {stats['success_count']:3d}, Avg Reward: {stats['avg_reward']:.2f}, "
                  f"Samples: {len(collected_data)}")

    except Exception as e:
        print(f"\n[WARNING] Test interrupted: {e}")
        print(f"[INFO] Saving {len(all_stats)} completed runs...")

    env.close()

    # 완료된 run 수 확인
    completed_runs = len(all_stats)
    if completed_runs == 0:
        print("[ERROR] No runs completed. No data to save.")
        return None

    # 통계 계산
    collisions = [s['collision_count'] for s in all_stats]
    successes = [s['success_count'] for s in all_stats]
    rewards = [s['avg_reward'] for s in all_stats]

    print(f"\n{'=' * 80}")
    print(f"[RESULTS] {completed_runs}/{num_runs} Runs × {test_steps} Steps")
    print(f"{'=' * 80}")
    print(f"  Collision Rate: {np.mean(collisions):.2f} ± {np.std(collisions):.2f} per run")
    print(f"  Success Rate:   {np.mean(successes):.2f} ± {np.std(successes):.2f} per run")
    print(f"  Avg Reward:     {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Total Samples:  {len(all_collected_data)}")
    print(f"{'=' * 80}")

    # 데이터 저장 (csv_path, save_dir는 위에서 이미 생성됨)
    if len(all_collected_data) > 0:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 헤더
            header = ['step', 'agent_id', 'colregs', 'n_agents', 'action_0', 'action_1']
            header += [f'obs_{i}' for i in range(OBS_SIZE)]
            header += [f'self_msg_{i}' for i in range(MSG_DIM)]
            header += [f'others_msg_{i}' for i in range(MSG_DIM)]
            writer.writerow(header)

            # 데이터
            for d in all_collected_data:
                row = [d['step'], d['agent_id'], d['colregs'], d['n_agents'], d['action'][0], d['action'][1]]
                row += d['obs_full']
                row += d['self_msg']
                row += d['others_msg']
                writer.writerow(row)

        print(f"\n[SAVED] Data saved to: {csv_path}")

        # 결과 요약 저장
        summary_path = os.path.join(save_dir, f"summary_{completed_runs}x{test_steps}_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"=== Multi-Run Test Results ===\n")
            f.write(f"Runs: {completed_runs}/{num_runs}\n")
            f.write(f"Steps per run: {test_steps}\n")
            f.write(f"Total steps: {num_runs * test_steps}\n")
            f.write(f"Model: {model_path}\n\n")
            f.write(f"Collision Rate: {np.mean(collisions):.2f} ± {np.std(collisions):.2f}\n")
            f.write(f"Success Rate: {np.mean(successes):.2f} ± {np.std(successes):.2f}\n")
            f.write(f"Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n\n")
            f.write(f"Per-run details:\n")
            for i, s in enumerate(all_stats):
                f.write(f"  Run {i+1}: Collision={s['collision_count']}, Success={s['success_count']}, Reward={s['avg_reward']:.2f}\n")
        print(f"[SAVED] Summary saved to: {summary_path}")

        return csv_path

    return None


def collect_trajectory(model_path=None, test_steps=TRAJECTORY_STEPS, time_scale=TRAJECTORY_TIME_SCALE):
    """
    간단한 trajectory 수집: step, agent_id, COLREGs, x, z 만 수집
    통신 ON/OFF 비교용 데이터 수집
    """
    print("=" * 80)
    print(f"[TRAJECTORY COLLECTION] {test_steps} steps")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Communication: {'ON' if USE_COMMUNICATION else 'OFF'}")
    print(f"Time Scale: {time_scale}x")
    print("=" * 80)

    # 모델 경로 확인
    if model_path is None:
        model_path = TEST_MODEL_PATH

    if model_path is None or not os.path.exists(model_path):
        print(f"\n[ERROR] Model path not found: {model_path}")
        return None

    # Unity 환경 연결 (main.py와 완전히 동일하게)
    print("\n[INFO] Connecting to Unity environment...")
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name=None,
        side_channels=[channel],
        worker_id=0,
        base_port=BASE_PORT,
        timeout_wait=30
    )
    print("[OK] Unity environment connected")

    # time_scale 설정 후 바로 reset (main.py와 동일 순서)
    channel.set_configuration_parameters(time_scale=20.0)
    env.reset()
    print(f"[OK] Time scale: 20x, Reset done")

    behavior_name = list(env.behavior_specs)[0]

    # 모델 로드
    print(f"\n[INFO] Loading model from: {model_path}")
    policy = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()
    print(f"[OK] Model loaded")
    frame_stack = MultiAgentFrameStack(FRAMES, STATE_SIZE)

    # 데이터 수집 리스트
    trajectory_data = []
    colregs_labels = ['None', 'HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']

    print(f"\n[COLLECTING] Running {test_steps} steps...")

    try:
        for step in range(test_steps):
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            agent_ids = list(decision_steps.agent_id)
            n_agents = len(agent_ids)

            if n_agents == 0:
                env.step()
                continue

            # 데이터 수집 + 행동 결정 준비
            batch_states = []
            batch_goals = []
            batch_self_states = []
            batch_colregs = []
            batch_positions = []  # x, z 위치
            batch_colregs_idx = []  # COLREGs index

            for idx, agent_id in enumerate(agent_ids):
                obs_raw = decision_steps.obs[0][idx]
                state, goal, self_state, colregs, obs_full, position = parse_observation(obs_raw)

                x_pos, z_pos = position[0], position[1]

                # COLREGs 상황 (one-hot → index)
                colregs_idx = np.argmax(colregs)

                batch_positions.append((x_pos, z_pos))
                batch_colregs_idx.append(colregs_idx)

                # 행동 결정용
                state_stacked = frame_stack.update(agent_id, state)
                batch_states.append(state_stacked)
                batch_goals.append(goal)
                batch_self_states.append(self_state)
                batch_colregs.append(colregs)

            # 통신 파트너 계산
            positions_dict = {agent_ids[i]: batch_positions[i] for i in range(n_agents)}
            comm_partners = {}
            if USE_COMMUNICATION:
                for agent_id in agent_ids:
                    comm_partners[agent_id] = get_comm_partners(agent_id, positions_dict[agent_id], positions_dict)

            # 행동 결정
            states_tensor = torch.FloatTensor(np.array(batch_states)).unsqueeze(0).to(DEVICE)
            goals_tensor = torch.FloatTensor(np.array(batch_goals)).unsqueeze(0).to(DEVICE)
            self_states_tensor = torch.FloatTensor(np.array(batch_self_states)).unsqueeze(0).to(DEVICE)
            colregs_tensor = torch.FloatTensor(np.array(batch_colregs)).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                if USE_COMMUNICATION:
                    values, actions, logprobs, means, _, self_msg, others_msg = policy.forward(
                        states_tensor, goals_tensor, self_states_tensor, colregs_tensor,
                        return_msg=True,
                        comm_partners=comm_partners,
                        agent_id_list=list(agent_ids)
                    )
                    self_msg_np = self_msg.squeeze(0).cpu().numpy()
                    others_msg_np = others_msg.squeeze(0).cpu().numpy()
                else:
                    values, actions, logprobs, means, _ = policy.forward(
                        states_tensor, goals_tensor, self_states_tensor, colregs_tensor
                    )
                    self_msg_np = None
                    others_msg_np = None

            actions_np = actions.squeeze(0).cpu().numpy()

            # 데이터 저장 (메시지 포함)
            for idx, agent_id in enumerate(agent_ids):
                x_pos, z_pos = batch_positions[idx]
                colregs_idx = batch_colregs_idx[idx]
                colregs_name = colregs_labels[colregs_idx]

                data_entry = {
                    'step': step,
                    'agent_id': agent_id,
                    'colregs': colregs_idx,
                    'colregs_name': colregs_name,
                    'x': x_pos,
                    'z': z_pos,
                    'speed': float(batch_self_states[idx][0]),
                    'heading': float(batch_self_states[idx][2]),
                    'rudder': float(batch_self_states[idx][3]),
                    'goal_dist': float(batch_goals[idx][0]),
                    'goal_angle': float(batch_goals[idx][1]),
                }

                # 통신 모드면 메시지 추가
                if USE_COMMUNICATION and self_msg_np is not None:
                    for i in range(MSG_DIM):
                        data_entry[f'self_msg_{i}'] = self_msg_np[idx, i]
                        data_entry[f'others_msg_{i}'] = others_msg_np[idx, i]

                trajectory_data.append(data_entry)

            # Terminal steps 처리
            for t_idx, agent_id in enumerate(terminal_steps.agent_id):
                frame_stack.remove_agent(agent_id)

            # Unity에 행동 전송
            all_actions = np.zeros((len(decision_steps.agent_id), CONTINUOUS_ACTION_SIZE))
            for i, agent_id in enumerate(decision_steps.agent_id):
                if i < len(actions_np):
                    all_actions[i] = actions_np[i]

            action_tuple = ActionTuple(continuous=all_actions)
            env.set_actions(behavior_name, action_tuple)
            env.step()

            # 진행 상황 출력
            if step % 1000 == 0:
                print(f"  [Step {step:5d}/{test_steps}] Agents: {n_agents}, Samples: {len(trajectory_data)}")

    except Exception as e:
        print(f"\n[WARNING] Collection interrupted: {e}")

    env.close()

    # CSV 저장
    if len(trajectory_data) > 0:
        comm_status = "commON" if USE_COMMUNICATION else "commOFF"
        save_dir = os.path.join(PROJECT_ROOT, "trajectory_data")
        os.makedirs(save_dir, exist_ok=True)

        csv_path = os.path.join(save_dir, f"{comm_status}.csv")

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # 헤더 (통신 모드면 메시지 컬럼 추가)
            header = ['step', 'agent_id', 'colregs', 'colregs_name', 'x', 'z',
                      'speed', 'heading', 'rudder', 'goal_dist', 'goal_angle']
            if USE_COMMUNICATION:
                header += [f'self_msg_{i}' for i in range(MSG_DIM)]
                header += [f'others_msg_{i}' for i in range(MSG_DIM)]
            writer.writerow(header)

            # 데이터
            for d in trajectory_data:
                row = [d['step'], d['agent_id'], d['colregs'], d['colregs_name'],
                       d['x'], d['z'], d['speed'], d['heading'], d['rudder'],
                       d['goal_dist'], d['goal_angle']]
                if USE_COMMUNICATION:
                    row += [d.get(f'self_msg_{i}', 0) for i in range(MSG_DIM)]
                    row += [d.get(f'others_msg_{i}', 0) for i in range(MSG_DIM)]
                writer.writerow(row)

        print(f"\n{'=' * 80}")
        print(f"[SAVED] {csv_path}")
        print(f"  Samples: {len(trajectory_data)}")
        print(f"{'=' * 80}")

        # COLREGs 준수도 자동 평가
        evaluate_colregs_compliance(csv_path)

        return csv_path

    return None


# ============================================================================
# COLREGs Compliance Evaluation (논문 메트릭)
# ============================================================================

def detect_encounters(df, min_steps=5):
    """
    에이전트별 COLREGs 조우(encounter) 시퀀스 감지.
    연속된 스텝에서 COLREGs 상황이 유지되면 하나의 encounter로 묶음.
    run_id로 run 경계 구분.
    """
    df = _add_run_id(df)
    encounters = []
    colregs_names = {0: 'None', 1: 'HeadOn', 2: 'CrossStandOn', 3: 'CrossGiveWay', 4: 'Overtaking'}

    for run_id in df['run_id'].unique():
        run_df = df[df['run_id'] == run_id]
        for agent_id in run_df['agent_id'].unique():
            agent_data = run_df[run_df['agent_id'] == agent_id].sort_values('step').reset_index(drop=True)

            current_enc = None
            prev_step = -999

            for _, row in agent_data.iterrows():
                step = int(row['step'])
                colregs = int(row['colregs'])

                # 에피소드 경계 감지 (step 갭 > 5면 새 에피소드)
                if prev_step >= 0 and step - prev_step > 5:
                    if current_enc is not None:
                        encounters.append(current_enc)
                        current_enc = None
                prev_step = step

                if colregs != 0:
                    if current_enc is None or current_enc['colregs'] != colregs:
                        if current_enc is not None:
                            encounters.append(current_enc)
                        current_enc = {
                            'agent_id': agent_id,
                            'colregs': colregs,
                            'colregs_name': colregs_names.get(colregs, f'Unknown_{colregs}'),
                            'start_step': step,
                            'rudders': [],
                            'speeds': [],
                            'positions': [],
                        }
                    current_enc['rudders'].append(float(row['rudder']))
                    current_enc['speeds'].append(float(row['speed']))
                    current_enc['positions'].append((float(row['x']), float(row['z'])))
                else:
                    if current_enc is not None:
                        encounters.append(current_enc)
                        current_enc = None

            if current_enc is not None:
                encounters.append(current_enc)

    # 짧은 조우 필터링 (노이즈 제거)
    encounters = [e for e in encounters if len(e['rudders']) >= min_steps]
    return encounters


def evaluate_encounters(encounters):
    """
    COLREGs 준수 평가 (encounter 전체 의도 기반).

    평가 기준:
    - HeadOn (Rule 14): mean(rudder) > 0 + 강한 좌현 < 25%
    - CrossGiveWay (Rule 15/16): mean(rudder) > 0 + 강한 좌현 < 25%
    - CrossStandOn (Rule 17): |mean(rudder)| < 0.2
    - Overtaking (Rule 13): mean(rudder) > -0.05
    """
    results = {
        'HeadOn': {'total': 0, 'compliant': 0, 'details': []},
        'CrossStandOn': {'total': 0, 'compliant': 0, 'details': []},
        'CrossGiveWay': {'total': 0, 'compliant': 0, 'details': []},
        'Overtaking': {'total': 0, 'compliant': 0, 'details': []},
    }

    for enc in encounters:
        name = enc['colregs_name']
        if name not in results:
            continue

        rudders = np.array(enc['rudders'])
        speeds = np.array(enc['speeds'])
        n_steps = len(rudders)
        mean_rudder = float(np.mean(rudders))

        compliant = False
        detail = {
            'n_steps': n_steps,
            'agent_id': enc['agent_id'],
            'avg_rudder': mean_rudder,
        }

        if name == 'HeadOn':
            strong_port_rate = float(np.mean(rudders < -0.2))
            compliant = mean_rudder > 0 and strong_port_rate < 0.25
            detail['strong_port_rate'] = strong_port_rate

        elif name == 'CrossGiveWay':
            strong_port_rate = float(np.mean(rudders < -0.2))
            speed_reduction = float(speeds[0] - np.min(speeds)) if len(speeds) > 1 else 0
            compliant = mean_rudder > 0 and strong_port_rate < 0.25
            detail['strong_port_rate'] = strong_port_rate
            detail['speed_reduction'] = speed_reduction

        elif name == 'CrossStandOn':
            course_keeping_rate = float(np.mean(np.abs(rudders) < 0.15))
            speed_keeping_rate = float(np.mean(speeds > 0.5))
            compliant = abs(mean_rudder) < 0.2
            detail['course_keeping_rate'] = course_keeping_rate
            detail['speed_keeping_rate'] = speed_keeping_rate

        elif name == 'Overtaking':
            compliant = mean_rudder > -0.05
            detail['mean_rudder'] = mean_rudder

        detail['compliant'] = compliant
        results[name]['total'] += 1
        results[name]['compliant'] += int(compliant)
        results[name]['details'].append(detail)

    return results


def _add_run_id(df):
    """step 번호가 감소하는 지점을 run 경계로 감지하여 run_id 추가"""
    if 'run_id' in df.columns:
        return df
    run_id = 0
    run_ids = []
    prev_step = -1
    for step in df['step'].values:
        if step < prev_step:
            run_id += 1
        run_ids.append(run_id)
        prev_step = step
    df = df.copy()
    df['run_id'] = run_ids
    return df


def compute_encounter_dcpa(df, proximity_range=60.0, gap_tolerance=5):
    """에이전트 쌍별 encounter를 추적하여 각 encounter의 DCPA(최소 거리)를 계산.

    encounter 정의: 특정 쌍이 proximity_range 이내로 진입한 시점부터
    gap_tolerance step 연속으로 범위 밖이 될 때까지를 하나의 encounter로 봄.

    Returns: dict with avg_min_distance, min_distances (list), n_pairs
    """
    df = _add_run_id(df)
    all_dcpa = []

    for run_id, run_df in df.groupby('run_id'):
        steps_sorted = sorted(run_df['step'].unique())

        # 매 step 쌍별 거리 계산
        pair_steps = {}  # {(a1,a2): [(step, dist), ...]}
        for step in steps_sorted:
            step_data = run_df[run_df['step'] == step][['agent_id', 'x', 'z']].values
            n = len(step_data)
            for i in range(n):
                for j in range(i + 1, n):
                    a1, a2 = int(step_data[i][0]), int(step_data[j][0])
                    pair_key = tuple(sorted([a1, a2]))
                    dist = np.sqrt((step_data[i][1] - step_data[j][1]) ** 2 +
                                   (step_data[i][2] - step_data[j][2]) ** 2)
                    if pair_key not in pair_steps:
                        pair_steps[pair_key] = []
                    pair_steps[pair_key].append((step, dist))

        # 쌍별로 encounter 분리 후 DCPA 추출
        for pair_key, step_dists in pair_steps.items():
            step_dists.sort(key=lambda x: x[0])
            in_encounter = False
            encounter_min = float('inf')
            gap_count = 0

            for step, dist in step_dists:
                if dist <= proximity_range:
                    if not in_encounter:
                        in_encounter = True
                        encounter_min = dist
                        gap_count = 0
                    else:
                        encounter_min = min(encounter_min, dist)
                        gap_count = 0
                else:
                    if in_encounter:
                        gap_count += 1
                        if gap_count >= gap_tolerance:
                            all_dcpa.append(encounter_min)
                            in_encounter = False
                            encounter_min = float('inf')
                            gap_count = 0

            # 마지막 encounter 처리
            if in_encounter:
                all_dcpa.append(encounter_min)

    return {
        'avg_min_distance': float(np.mean(all_dcpa)) if all_dcpa else 0,
        'min_distances': all_dcpa,
        'n_pairs': len(all_dcpa)
    }


def evaluate_colregs_compliance(csv_path):
    """
    COLREGs 준수도 평가 (논문 메트릭).

    Required CSV columns: step, agent_id, colregs, x, z, rudder, speed
    Optional: heading, goal_dist, goal_angle

    Returns: dict with compliance rates, DCPA metrics
    """
    df = pd.read_csv(csv_path)

    required = ['step', 'agent_id', 'colregs', 'x', 'z', 'rudder', 'speed']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        print(f"[INFO] Available: {list(df.columns)}")
        print(f"[INFO] Trajectory CSV must include speed/heading/rudder fields.")
        print(f"[INFO] Re-run collect_trajectory to generate compatible CSV.")
        return None

    # 1. Encounter 감지
    encounters = detect_encounters(df)
    print(f"\n[INFO] Detected {len(encounters)} encounters (min 5 steps)")

    # 2. Per-encounter 준수도 평가
    compliance = evaluate_encounters(encounters)

    # 3. Pairwise DCPA 계산
    dcpa = compute_encounter_dcpa(df)

    # 4. 결과 출력
    total_enc = sum(r['total'] for r in compliance.values())
    total_comp = sum(r['compliant'] for r in compliance.values())

    print(f"\n{'=' * 60}")
    print(f"COLREGs Compliance Report")
    print(f"{'=' * 60}")
    print(f"Source: {os.path.basename(csv_path)}")
    print(f"Total encounters: {total_enc}")
    print(f"Overall compliance: {total_comp}/{total_enc} = {total_comp / max(total_enc, 1) * 100:.1f}%")

    print(f"\n{'Situation':<20} {'Compliant':>10} {'Total':>8} {'Rate':>8}")
    print(f"{'-' * 48}")
    for situation, data in compliance.items():
        if data['total'] > 0:
            rate = data['compliant'] / data['total'] * 100
            print(f"{situation:<20} {data['compliant']:>10} {data['total']:>8} {rate:>7.1f}%")
        else:
            print(f"{situation:<20} {'N/A':>10} {0:>8} {'N/A':>8}")

    # 상세 통계 (starboard rate, course keeping rate 평균)
    print(f"\nDetailed Metrics:")
    for situation, data in compliance.items():
        if data['total'] == 0:
            continue
        details = data['details']
        avg_rudder = np.mean([d['avg_rudder'] for d in details])
        if situation in ('HeadOn', 'CrossGiveWay'):
            avg_port = np.mean([d.get('strong_port_rate', 0) for d in details])
            print(f"  {situation}: Avg Rudder={avg_rudder:.3f}, Avg Strong Port Rate={avg_port:.2f}")
        elif situation == 'CrossStandOn':
            avg_ck = np.mean([d['course_keeping_rate'] for d in details])
            avg_sk = np.mean([d['speed_keeping_rate'] for d in details])
            print(f"  {situation}: Avg Rudder={avg_rudder:.3f}, Course Keeping={avg_ck:.2f}, Speed Keeping={avg_sk:.2f}")
        elif situation == 'Overtaking':
            print(f"  {situation}: Avg Rudder={avg_rudder:.3f}")

    print(f"\nSafety Metrics:")
    print(f"  Avg Min Passing Distance: {dcpa['avg_min_distance']:.2f}m")
    if dcpa['min_distances']:
        print(f"  Min Distance: {min(dcpa['min_distances']):.2f}m")
        print(f"  Max Distance: {max(dcpa['min_distances']):.2f}m")
    print(f"  Agent Pairs Tracked: {dcpa['n_pairs']}")
    print(f"{'=' * 60}")

    return {
        'compliance': compliance,
        'dcpa': dcpa,
        'total_encounters': total_enc,
        'overall_compliance_rate': total_comp / max(total_enc, 1),
    }


def load_model_with_padding(policy, model_path):
    """Phase 1 모델 로드 (msg_dim 차이 자동 패딩, others_msg=0이면 영향 없음)"""
    saved_state = torch.load(model_path, map_location=DEVICE)
    model_state = policy.state_dict()
    padded = []

    for key, saved_param in saved_state.items():
        if key not in model_state:
            continue
        model_param = model_state[key]
        if saved_param.shape == model_param.shape:
            model_state[key] = saved_param
        elif saved_param.dim() == 2 and model_param.dim() == 2:
            model_state[key] = torch.zeros_like(model_param)
            r = min(saved_param.shape[0], model_param.shape[0])
            c = min(saved_param.shape[1], model_param.shape[1])
            model_state[key][:r, :c] = saved_param[:r, :c]
            padded.append(f"{key}: {list(saved_param.shape)} -> {list(model_param.shape)}")

    policy.load_state_dict(model_state)
    if padded:
        print(f"[INFO] Padded layers (zero-filled): {padded}")


def run_comparison_test(test_steps=TEST_STEPS, num_runs=NUM_RUNS, time_scale=TEST_TIME_SCALE, tag=None):
    """Comm OFF vs Comm ON 비교 테스트 (Unity 한 번 연결)"""
    import networks

    print("=" * 80)
    print(f"[COMPARISON TEST] Comm OFF vs Comm ON")
    print(f"  {num_runs} runs x {test_steps} steps per model")
    print("=" * 80)

    # 모델 경로 확인
    for label, path in [("COMM_OFF", MODEL_PATH_COMM_OFF), ("COMM_ON", MODEL_PATH_COMM_ON)]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} model not found: {path}")
            return
        print(f"  {label}: {path}")

    # Unity 환경 연결 (한 번만)
    print("\n[INFO] Connecting to Unity environment...")
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name=None,
        side_channels=[channel],
        base_port=BASE_PORT
    )
    channel.set_configuration_parameters(time_scale=time_scale)
    print("[OK] Unity environment connected")

    env.reset()
    behavior_name = list(env.behavior_specs)[0]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PROJECT_ROOT, "latent_data")
    os.makedirs(save_dir, exist_ok=True)

    # tag가 있으면 파일명에 접두사 추가
    prefix = f"{tag}_" if tag else ""

    results = {}

    # === Test 1: Comm OFF ===
    print(f"\n{'=' * 60}")
    print("[TEST 1/2] Communication OFF")
    print(f"{'=' * 60}")
    networks.USE_COMMUNICATION = False

    policy_off = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES).to(DEVICE)
    load_model_with_padding(policy_off, MODEL_PATH_COMM_OFF)
    policy_off.eval()
    print(f"[OK] COMM_OFF model loaded")

    off_stats = []
    off_data = []
    off_trajectory = []
    for run in range(num_runs):
        frame_stack = MultiAgentFrameStack(FRAMES, STATE_SIZE)
        stats, collected, traj = run_single_test(env, policy_off, frame_stack, behavior_name, test_steps, COLLECT_INTERVAL, use_comm=False)
        off_stats.append(stats)
        off_data.extend(collected)
        off_trajectory.extend(traj)
        print(f"  [Run {run+1:2d}/{num_runs}] Collision: {stats['collision_count']:3d}, "
              f"Success: {stats['success_count']:3d}, Reward: {stats['avg_reward']:.2f}")

    results['off'] = off_stats
    del policy_off

    # === Test 2: Comm ON ===
    print(f"\n{'=' * 60}")
    print("[TEST 2/2] Communication ON")
    print(f"{'=' * 60}")
    networks.USE_COMMUNICATION = True

    policy_on = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES).to(DEVICE)
    policy_on.load_state_dict(torch.load(MODEL_PATH_COMM_ON, map_location=DEVICE))
    policy_on.eval()
    print(f"[OK] COMM_ON model loaded")

    on_stats = []
    on_data = []
    on_trajectory = []
    for run in range(num_runs):
        frame_stack = MultiAgentFrameStack(FRAMES, STATE_SIZE)
        stats, collected, traj = run_single_test(env, policy_on, frame_stack, behavior_name, test_steps, COLLECT_INTERVAL, use_comm=True)
        on_stats.append(stats)
        on_data.extend(collected)
        on_trajectory.extend(traj)
        print(f"  [Run {run+1:2d}/{num_runs}] Collision: {stats['collision_count']:3d}, "
              f"Success: {stats['success_count']:3d}, Reward: {stats['avg_reward']:.2f}")

    results['on'] = on_stats
    del policy_on

    env.close()

    # === Trajectory CSV 저장 ===
    traj_dir = os.path.join(PROJECT_ROOT, "trajectory_data")
    os.makedirs(traj_dir, exist_ok=True)

    traj_header = ['step', 'agent_id', 'colregs', 'colregs_name', 'x', 'z',
                   'speed', 'heading', 'rudder', 'goal_dist', 'goal_angle']

    off_traj_path = os.path.join(traj_dir, f"{prefix}compare_commOFF_{num_runs}x{test_steps}_{timestamp}.csv")
    on_traj_path = os.path.join(traj_dir, f"{prefix}compare_commON_{num_runs}x{test_steps}_{timestamp}.csv")

    for traj_list, traj_path in [(off_trajectory, off_traj_path), (on_trajectory, on_traj_path)]:
        if len(traj_list) > 0:
            with open(traj_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(traj_header)
                for d in traj_list:
                    w.writerow([d['step'], d['agent_id'], d['colregs'], d['colregs_name'],
                                d['x'], d['z'], d['speed'], d['heading'], d['rudder'],
                                d['goal_dist'], d['goal_angle']])
            print(f"[SAVED] Trajectory: {traj_path}")

    # === 결과 비교 (콘솔 출력) ===
    off_c = [s['collision_count'] for s in off_stats]
    off_s = [s['success_count'] for s in off_stats]
    off_r = [s['avg_reward'] for s in off_stats]
    on_c = [s['collision_count'] for s in on_stats]
    on_s = [s['success_count'] for s in on_stats]
    on_r = [s['avg_reward'] for s in on_stats]

    print(f"\n{'=' * 80}")
    print(f"[COMPARISON RESULTS] {num_runs} runs x {test_steps} steps")
    print(f"{'=' * 80}")
    print(f"{'Metric':<20} {'Comm OFF':>20} {'Comm ON':>20}")
    print(f"{'-'*60}")
    print(f"{'Collision':<20} {np.mean(off_c):>8.2f} +/- {np.std(off_c):<8.2f} {np.mean(on_c):>8.2f} +/- {np.std(on_c):<8.2f}")
    print(f"{'Success':<20} {np.mean(off_s):>8.2f} +/- {np.std(off_s):<8.2f} {np.mean(on_s):>8.2f} +/- {np.std(on_s):<8.2f}")
    print(f"{'Avg Reward':<20} {np.mean(off_r):>8.2f} +/- {np.std(off_r):<8.2f} {np.mean(on_r):>8.2f} +/- {np.std(on_r):<8.2f}")
    print(f"{'=' * 80}")

    # === COLREGs Compliance 평가 ===
    compliance_off = None
    compliance_on = None

    if len(off_trajectory) > 0:
        print(f"\n{'=' * 60}")
        print("[EVAL] COLREGs Compliance - Communication OFF")
        print(f"{'=' * 60}")
        compliance_off = evaluate_colregs_compliance(off_traj_path)

    if len(on_trajectory) > 0:
        print(f"\n{'=' * 60}")
        print("[EVAL] COLREGs Compliance - Communication ON")
        print(f"{'=' * 60}")
        compliance_on = evaluate_colregs_compliance(on_traj_path)

    # === 비교 그래프 생성 ===
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    if compliance_off is not None and compliance_on is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        C_OFF = '#4878A8'
        C_ON = '#D4652F'

        # --- Figure 1: COLREGs Compliance 비교 ---
        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))

        # (a) Per-situation compliance rate
        ax = axes1[0]
        situations = ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']
        labels = ['Head-on\n(Rule 14)', 'Stand-on\n(Rule 17)', 'Give-way\n(Rule 15/16)', 'Overtaking\n(Rule 13)']
        off_rates = []
        on_rates = []
        for sit in situations:
            off_d = compliance_off['compliance'].get(sit, {'total': 0, 'compliant': 0})
            on_d = compliance_on['compliance'].get(sit, {'total': 0, 'compliant': 0})
            off_rates.append(off_d['compliant'] / max(off_d['total'], 1) * 100)
            on_rates.append(on_d['compliant'] / max(on_d['total'], 1) * 100)

        x = np.arange(len(situations))
        w = 0.35
        bars1 = ax.bar(x - w/2, off_rates, w, color=C_OFF, edgecolor='black', linewidth=0.5, label='Without Comm.')
        bars2 = ax.bar(x + w/2, on_rates, w, color=C_ON, edgecolor='black', linewidth=0.5, label='With Comm.')
        ax.set_ylabel('Compliance Rate (%)')
        ax.set_title('(a) COLREGs Compliance by Situation', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim([0, 105])
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        for bar in bars1:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)

        # (b) Overall compliance + DCPA
        ax = axes1[1]
        off_overall = compliance_off['overall_compliance_rate'] * 100
        on_overall = compliance_on['overall_compliance_rate'] * 100
        off_dcpa = compliance_off['dcpa']['avg_min_distance']
        on_dcpa = compliance_on['dcpa']['avg_min_distance']

        metrics = ['Overall\nCompliance (%)', 'Avg DCPA (m)']
        off_vals = [off_overall, off_dcpa]
        on_vals = [on_overall, on_dcpa]

        x2 = np.arange(len(metrics))
        bars1 = ax.bar(x2 - w/2, off_vals, w, color=C_OFF, edgecolor='black', linewidth=0.5, label='Without Comm.')
        bars2 = ax.bar(x2 + w/2, on_vals, w, color=C_ON, edgecolor='black', linewidth=0.5, label='With Comm.')
        ax.set_title('(b) Overall Compliance & Safety Distance', fontweight='bold')
        ax.set_xticks(x2)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        env_label = tag.replace('_', ' ').title() if tag else 'Test'
        plt.suptitle(f'COLREGs Compliance Comparison: {env_label} Environment\n({num_runs} runs x {test_steps} steps)',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig1_path = os.path.join(figures_dir, f'{prefix}colregs_comparison_{timestamp}.png')
        plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
        print(f"[SAVED] {fig1_path}")

        # --- Figure 2: DCPA 분포 + Collision 비교 ---
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

        # (a) DCPA distribution
        ax = axes2[0]
        off_dists = compliance_off['dcpa']['min_distances']
        on_dists = compliance_on['dcpa']['min_distances']
        if off_dists and on_dists:
            bp = ax.boxplot([off_dists, on_dists], tick_labels=['Without\nComm.', 'With\nComm.'],
                            patch_artist=True,
                            boxprops=dict(linewidth=1.5),
                            whiskerprops=dict(linewidth=1.5),
                            medianprops=dict(linewidth=2, color='black'),
                            capprops=dict(linewidth=1.5))
            bp['boxes'][0].set_facecolor(C_OFF)
            bp['boxes'][1].set_facecolor(C_ON)
            ax.scatter([1, 2], [np.mean(off_dists), np.mean(on_dists)],
                       marker='D', color='white', edgecolors='black', s=60, zorder=5, label='Mean')
            ax.legend(fontsize=9)
        ax.set_ylabel('Min Passing Distance (m)')
        ax.set_title('(a) DCPA Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # (b) Collision per run
        ax = axes2[1]
        bp2 = ax.boxplot([off_c, on_c], tick_labels=['Without\nComm.', 'With\nComm.'],
                         patch_artist=True,
                         boxprops=dict(linewidth=1.5),
                         whiskerprops=dict(linewidth=1.5),
                         medianprops=dict(linewidth=2, color='black'),
                         capprops=dict(linewidth=1.5))
        bp2['boxes'][0].set_facecolor(C_OFF)
        bp2['boxes'][1].set_facecolor(C_ON)
        ax.scatter([1, 2], [np.mean(off_c), np.mean(on_c)],
                   marker='D', color='white', edgecolors='black', s=60, zorder=5, label='Mean')
        ax.set_ylabel('Collisions per Run')
        ax.set_title('(b) Collision Count Distribution', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.suptitle(f'Safety Comparison: {env_label} Environment\n({num_runs} runs x {test_steps} steps)',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig2_path = os.path.join(figures_dir, f'{prefix}safety_comparison_{timestamp}.png')
        plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
        print(f"[SAVED] {fig2_path}")
        plt.close('all')

    # === Summary 텍스트 저장 ===
    summary_path = os.path.join(save_dir, f"{prefix}comparison_{num_runs}x{test_steps}_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"=== Comparison Test Results ===\n")
        f.write(f"Runs: {num_runs}, Steps per run: {test_steps}\n")
        f.write(f"COMM_OFF model: {MODEL_PATH_COMM_OFF}\n")
        f.write(f"COMM_ON model: {MODEL_PATH_COMM_ON}\n\n")
        f.write(f"--- Comm OFF ---\n")
        f.write(f"Collision: {np.mean(off_c):.2f} +/- {np.std(off_c):.2f}\n")
        f.write(f"Success: {np.mean(off_s):.2f} +/- {np.std(off_s):.2f}\n")
        f.write(f"Avg Reward: {np.mean(off_r):.2f} +/- {np.std(off_r):.2f}\n\n")
        f.write(f"--- Comm ON ---\n")
        f.write(f"Collision: {np.mean(on_c):.2f} +/- {np.std(on_c):.2f}\n")
        f.write(f"Success: {np.mean(on_s):.2f} +/- {np.std(on_s):.2f}\n")
        f.write(f"Avg Reward: {np.mean(on_r):.2f} +/- {np.std(on_r):.2f}\n\n")
        if compliance_off:
            f.write(f"--- COLREGs Compliance ---\n")
            f.write(f"Overall OFF: {compliance_off['overall_compliance_rate']*100:.1f}%\n")
            f.write(f"Overall ON:  {compliance_on['overall_compliance_rate']*100:.1f}%\n")
            f.write(f"DCPA OFF: {compliance_off['dcpa']['avg_min_distance']:.2f}m\n")
            f.write(f"DCPA ON:  {compliance_on['dcpa']['avg_min_distance']:.2f}m\n\n")
        f.write(f"Per-run details (OFF):\n")
        for i, s in enumerate(off_stats):
            f.write(f"  Run {i+1}: Collision={s['collision_count']}, Success={s['success_count']}, Reward={s['avg_reward']:.2f}\n")
        f.write(f"\nPer-run details (ON):\n")
        for i, s in enumerate(on_stats):
            f.write(f"  Run {i+1}: Collision={s['collision_count']}, Success={s['success_count']}, Reward={s['avg_reward']:.2f}\n")

    print(f"\n[SAVED] {summary_path}")

    results['compliance_off'] = compliance_off
    results['compliance_on'] = compliance_on
    return results


# ============================================================================
# Mixed Model Test (Radar-only + Communication 혼합)
# ============================================================================

def run_mixed_test(env, behavior_name, policy_off, policy_on, n_radar, n_comm,
                   num_runs=10, test_steps=2000):
    """
    혼합 모델 테스트: radar-only 에이전트는 Phase 1 모델, comm 에이전트는 Phase 2 모델 사용.

    Args:
        env: Unity 환경
        behavior_name: behavior spec 이름
        policy_off: Phase 1 모델 (radar-only용)
        policy_on: Phase 2 모델 (comm용)
        n_radar: radar-only 에이전트 수 (agent_id 0 ~ n_radar-1)
        n_comm: communication 에이전트 수 (agent_id n_radar ~ n_radar+n_comm-1)
        num_runs: 반복 횟수
        test_steps: run당 스텝 수
    """
    import networks

    colregs_labels = ['None', 'HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']

    all_stats = []
    all_trajectory = []

    for run in range(num_runs):
        frame_stack = MultiAgentFrameStack(FRAMES, STATE_SIZE)

        stats = {
            'collision_count': 0,
            'success_count': 0,
            'total_reward': 0,
        }
        agent_rewards = {}
        trajectory_data = []

        # 환경 리셋
        env.reset()

        for step in range(test_steps):
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            agent_ids = list(decision_steps.agent_id)
            n_agents = len(agent_ids)

            if n_agents == 0:
                env.step()
                continue

            # 전체 에이전트 관측 파싱
            batch_states = []
            batch_goals = []
            batch_self_states = []
            batch_colregs = []
            batch_positions = {}
            agent_id_list = []

            for idx, agent_id in enumerate(agent_ids):
                obs_raw = decision_steps.obs[0][idx]
                state, goal, self_state, colregs_vec, obs_full, position = parse_observation(obs_raw)

                state_stacked = frame_stack.update(agent_id, state)

                batch_states.append(state_stacked)
                batch_goals.append(goal)
                batch_self_states.append(self_state)
                batch_colregs.append(colregs_vec)
                batch_positions[agent_id] = position
                agent_id_list.append(agent_id)

                if agent_id not in agent_rewards:
                    agent_rewards[agent_id] = 0

            # 에이전트를 radar / comm 그룹으로 분류
            radar_indices = [i for i, aid in enumerate(agent_id_list) if aid < n_radar]
            comm_indices = [i for i, aid in enumerate(agent_id_list) if aid >= n_radar]

            # 전체 행동 배열 (나중에 merge)
            merged_actions = np.zeros((n_agents, CONTINUOUS_ACTION_SIZE))

            # --- Radar-only 그룹 추론 (Phase 1 모델, 통신 없음) ---
            if radar_indices:
                r_states = np.array([batch_states[i] for i in radar_indices])
                r_goals = np.array([batch_goals[i] for i in radar_indices])
                r_self_states = np.array([batch_self_states[i] for i in radar_indices])
                r_colregs = np.array([batch_colregs[i] for i in radar_indices])

                r_states_t = torch.FloatTensor(r_states).unsqueeze(0).to(DEVICE)
                r_goals_t = torch.FloatTensor(r_goals).unsqueeze(0).to(DEVICE)
                r_self_states_t = torch.FloatTensor(r_self_states).unsqueeze(0).to(DEVICE)
                r_colregs_t = torch.FloatTensor(r_colregs).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    # Phase 1 모델: 통신 OFF
                    networks.USE_COMMUNICATION = False
                    _, r_actions, _, _, _ = policy_off.forward(
                        r_states_t, r_goals_t, r_self_states_t, r_colregs_t
                    )

                r_actions_np = r_actions.squeeze(0).cpu().numpy()
                for local_i, global_i in enumerate(radar_indices):
                    merged_actions[global_i] = r_actions_np[local_i]

            # --- Comm 그룹 추론 (Phase 2 모델, comm 에이전트끼리만 통신) ---
            if comm_indices:
                c_states = np.array([batch_states[i] for i in comm_indices])
                c_goals = np.array([batch_goals[i] for i in comm_indices])
                c_self_states = np.array([batch_self_states[i] for i in comm_indices])
                c_colregs = np.array([batch_colregs[i] for i in comm_indices])

                c_states_t = torch.FloatTensor(c_states).unsqueeze(0).to(DEVICE)
                c_goals_t = torch.FloatTensor(c_goals).unsqueeze(0).to(DEVICE)
                c_self_states_t = torch.FloatTensor(c_self_states).unsqueeze(0).to(DEVICE)
                c_colregs_t = torch.FloatTensor(c_colregs).unsqueeze(0).to(DEVICE)

                # comm 에이전트끼리만 통신 파트너 계산
                comm_agent_ids = [agent_id_list[i] for i in comm_indices]
                comm_positions = {aid: batch_positions[aid] for aid in comm_agent_ids}
                comm_partners = {}
                for aid in comm_agent_ids:
                    comm_partners[aid] = get_comm_partners(aid, comm_positions[aid], comm_positions)

                with torch.no_grad():
                    # Phase 2 모델: 통신 ON
                    networks.USE_COMMUNICATION = True
                    _, c_actions, _, _, _ = policy_on.forward(
                        c_states_t, c_goals_t, c_self_states_t, c_colregs_t,
                        comm_partners=comm_partners,
                        agent_id_list=comm_agent_ids
                    )

                c_actions_np = c_actions.squeeze(0).cpu().numpy()
                for local_i, global_i in enumerate(comm_indices):
                    merged_actions[global_i] = c_actions_np[local_i]

            # Trajectory 데이터 수집 (매 스텝)
            for i, agent_id in enumerate(agent_id_list):
                colregs_idx = int(np.argmax(batch_colregs[i]))
                pos = batch_positions[agent_id]
                agent_type = "radar" if agent_id < n_radar else "comm"
                trajectory_data.append({
                    'step': step,
                    'agent_id': agent_id,
                    'agent_type': agent_type,
                    'colregs': colregs_idx,
                    'colregs_name': colregs_labels[colregs_idx],
                    'x': float(pos[0]),
                    'z': float(pos[1]),
                    'speed': float(batch_self_states[i][0]),
                    'heading': float(batch_self_states[i][2]),
                    'rudder': float(batch_self_states[i][3]),
                    'goal_dist': float(batch_goals[i][0]),
                    'goal_angle': float(batch_goals[i][1]),
                    'action_0': float(merged_actions[i][0]),
                    'action_1': float(merged_actions[i][1]),
                })

            # Reward 누적
            for i, agent_id in enumerate(agent_id_list):
                idx = decision_steps.agent_id.tolist().index(agent_id)
                reward = decision_steps.reward[idx]
                agent_rewards[agent_id] += reward
                stats['total_reward'] += reward

            # Terminal steps 처리
            for t_idx, agent_id in enumerate(terminal_steps.agent_id):
                reward = terminal_steps.reward[t_idx]

                if agent_id in agent_rewards:
                    agent_rewards[agent_id] += reward
                else:
                    agent_rewards[agent_id] = reward

                stats['total_reward'] += reward

                if reward > 0:
                    stats['success_count'] += 1
                elif reward < -50:
                    stats['collision_count'] += 1

                frame_stack.remove_agent(agent_id)

            # Unity에 행동 전송
            all_actions = np.zeros((len(decision_steps.agent_id), CONTINUOUS_ACTION_SIZE))
            for i, agent_id in enumerate(decision_steps.agent_id):
                if agent_id in agent_id_list:
                    agent_idx = agent_id_list.index(agent_id)
                    all_actions[i] = merged_actions[agent_idx]

            action_tuple = ActionTuple(continuous=all_actions)

            try:
                env.set_actions(behavior_name, action_tuple)
                env.step()
            except Exception as e:
                print(f"  [WARNING] Unity error: {e}")
                break

        # 평균 리워드 계산
        avg_reward = stats['total_reward'] / max(len(agent_rewards), 1)
        stats['avg_reward'] = avg_reward

        # Radar / Comm 그룹별 통계
        radar_rewards = [v for k, v in agent_rewards.items() if k < n_radar]
        comm_rewards = [v for k, v in agent_rewards.items() if k >= n_radar]
        stats['radar_avg_reward'] = np.mean(radar_rewards) if radar_rewards else 0
        stats['comm_avg_reward'] = np.mean(comm_rewards) if comm_rewards else 0

        all_stats.append(stats)
        all_trajectory.extend(trajectory_data)

        print(f"  [Run {run+1:2d}/{num_runs}] Collision: {stats['collision_count']:3d}, "
              f"Success: {stats['success_count']:3d}, Reward: {stats['avg_reward']:.2f} "
              f"(Radar: {stats['radar_avg_reward']:.2f}, Comm: {stats['comm_avg_reward']:.2f})")

    return all_stats, all_trajectory


def run_mixed_experiment(num_runs=10, test_steps=2000, time_scale=20.0, tag=None):
    """
    혼합 모델 실험: 다양한 radar/comm 비율로 테스트 실행.

    설정 목록: (2,14), (4,12), (6,10), (8,8)
    """
    import networks

    configs = [
        (2, 14),   # 2 radar + 14 comm
        (4, 12),   # 4 radar + 12 comm
        (6, 10),   # 6 radar + 10 comm
        (8, 8),    # 8 radar + 8 comm
    ]

    print("=" * 80)
    print(f"[MIXED EXPERIMENT] {len(configs)} configs x {num_runs} runs x {test_steps} steps")
    print("=" * 80)

    # 모델 경로 확인
    for label, path in [("COMM_OFF (Phase 1)", MODEL_PATH_COMM_OFF), ("COMM_ON (Phase 2)", MODEL_PATH_COMM_ON)]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} model not found: {path}")
            return
        print(f"  {label}: {path}")

    # 두 모델 로드
    print("\n[INFO] Loading Phase 1 model (radar-only)...")
    policy_off = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES).to(DEVICE)
    load_model_with_padding(policy_off, MODEL_PATH_COMM_OFF)
    policy_off.eval()
    print(f"[OK] Phase 1 model loaded")

    print("[INFO] Loading Phase 2 model (communication)...")
    policy_on = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH_COMM_ON, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy_on.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy_on.load_state_dict(checkpoint)
    policy_on.eval()
    print(f"[OK] Phase 2 model loaded")

    # Unity 환경 연결 (한 번만)
    print("\n[INFO] Connecting to Unity environment...")
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name=None,
        side_channels=[channel],
        base_port=BASE_PORT
    )
    channel.set_configuration_parameters(time_scale=time_scale)
    print("[OK] Unity environment connected")

    env.reset()
    behavior_name = list(env.behavior_specs)[0]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    traj_dir = os.path.join(PROJECT_ROOT, "trajectory_data")
    os.makedirs(traj_dir, exist_ok=True)
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    prefix = f"{tag}_" if tag else ""
    all_results = {}

    for n_radar, n_comm in configs:
        config_label = f"R{n_radar}_C{n_comm}"
        print(f"\n{'=' * 60}")
        print(f"[CONFIG] {n_radar} radar + {n_comm} comm")
        print(f"{'=' * 60}")

        stats_list, trajectory = run_mixed_test(
            env, behavior_name, policy_off, policy_on,
            n_radar, n_comm, num_runs=num_runs, test_steps=test_steps
        )

        all_results[config_label] = {
            'stats': stats_list,
            'n_radar': n_radar,
            'n_comm': n_comm,
        }

        # Trajectory CSV 저장
        if len(trajectory) > 0:
            traj_header = ['step', 'agent_id', 'agent_type', 'colregs', 'colregs_name',
                           'x', 'z', 'speed', 'heading', 'rudder',
                           'goal_dist', 'goal_angle', 'action_0', 'action_1']
            traj_path = os.path.join(traj_dir, f"{prefix}mixed_{config_label}_{num_runs}x{test_steps}_{timestamp}.csv")

            with open(traj_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(traj_header)
                for d in trajectory:
                    w.writerow([d['step'], d['agent_id'], d['agent_type'],
                                d['colregs'], d['colregs_name'],
                                d['x'], d['z'], d['speed'], d['heading'], d['rudder'],
                                d['goal_dist'], d['goal_angle'],
                                d['action_0'], d['action_1']])

            print(f"[SAVED] Trajectory: {traj_path}")
            all_results[config_label]['traj_path'] = traj_path

            # COLREGs 준수도 평가
            compliance = evaluate_colregs_compliance(traj_path)
            all_results[config_label]['compliance'] = compliance

    env.close()
    del policy_off, policy_on

    # === 비교 결과 출력 ===
    print(f"\n{'=' * 80}")
    print(f"[MIXED EXPERIMENT RESULTS] {num_runs} runs x {test_steps} steps")
    print(f"{'=' * 80}")
    print(f"{'Config':<12} {'Collision':>12} {'Success':>12} {'Avg Reward':>14} {'Radar Rwd':>12} {'Comm Rwd':>12}")
    print(f"{'-' * 74}")

    for config_label, data in all_results.items():
        stats_list = data['stats']
        collisions = [s['collision_count'] for s in stats_list]
        successes = [s['success_count'] for s in stats_list]
        rewards = [s['avg_reward'] for s in stats_list]
        radar_rwds = [s['radar_avg_reward'] for s in stats_list]
        comm_rwds = [s['comm_avg_reward'] for s in stats_list]

        print(f"{config_label:<12} "
              f"{np.mean(collisions):>5.1f}+/-{np.std(collisions):<4.1f} "
              f"{np.mean(successes):>5.1f}+/-{np.std(successes):<4.1f} "
              f"{np.mean(rewards):>7.2f}+/-{np.std(rewards):<5.2f} "
              f"{np.mean(radar_rwds):>5.1f}+/-{np.std(radar_rwds):<4.1f} "
              f"{np.mean(comm_rwds):>5.1f}+/-{np.std(comm_rwds):<4.1f}")

    print(f"{'=' * 80}")

    # === 비교 그래프 생성 ===
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        config_labels = list(all_results.keys())
        x_labels = [f"{d['n_radar']}R+{d['n_comm']}C" for d in all_results.values()]
        x = np.arange(len(config_labels))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) Collision 비교
        ax = axes[0, 0]
        col_means = [np.mean([s['collision_count'] for s in all_results[cl]['stats']]) for cl in config_labels]
        col_stds = [np.std([s['collision_count'] for s in all_results[cl]['stats']]) for cl in config_labels]
        ax.bar(x, col_means, yerr=col_stds, color='#D4652F', edgecolor='black', linewidth=0.5, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('Collisions per Run')
        ax.set_title('(a) Collision Count', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # (b) Success 비교
        ax = axes[0, 1]
        suc_means = [np.mean([s['success_count'] for s in all_results[cl]['stats']]) for cl in config_labels]
        suc_stds = [np.std([s['success_count'] for s in all_results[cl]['stats']]) for cl in config_labels]
        ax.bar(x, suc_means, yerr=suc_stds, color='#4878A8', edgecolor='black', linewidth=0.5, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('Successes per Run')
        ax.set_title('(b) Goal Arrivals', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # (c) Radar vs Comm 그룹별 평균 보상
        ax = axes[1, 0]
        radar_means = [np.mean([s['radar_avg_reward'] for s in all_results[cl]['stats']]) for cl in config_labels]
        comm_means = [np.mean([s['comm_avg_reward'] for s in all_results[cl]['stats']]) for cl in config_labels]
        w = 0.35
        ax.bar(x - w/2, radar_means, w, color='#888888', edgecolor='black', linewidth=0.5, label='Radar-only')
        ax.bar(x + w/2, comm_means, w, color='#D4652F', edgecolor='black', linewidth=0.5, label='Comm')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('Avg Reward')
        ax.set_title('(c) Per-group Reward', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # (d) COLREGs 준수율
        ax = axes[1, 1]
        compliance_rates = []
        for cl in config_labels:
            comp = all_results[cl].get('compliance')
            if comp:
                compliance_rates.append(comp['overall_compliance_rate'] * 100)
            else:
                compliance_rates.append(0)
        ax.bar(x, compliance_rates, color='#2E8B57', edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('Compliance Rate (%)')
        ax.set_title('(d) COLREGs Compliance', fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)
        for i, rate in enumerate(compliance_rates):
            ax.text(i, rate + 1, f'{rate:.0f}%', ha='center', va='bottom', fontsize=10)

        env_label = tag.replace('_', ' ').title() if tag else 'Mixed Model'
        plt.suptitle(f'Mixed Model Experiment: {env_label}\n({num_runs} runs x {test_steps} steps)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f'{prefix}mixed_comparison_{timestamp}.png')
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"[SAVED] {fig_path}")
        plt.close('all')
    except Exception as e:
        print(f"[WARNING] 그래프 생성 실패: {e}")

    # === Summary 텍스트 저장 ===
    save_dir = os.path.join(PROJECT_ROOT, "latent_data")
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, f"{prefix}mixed_experiment_{num_runs}x{test_steps}_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"=== Mixed Model Experiment Results ===\n")
        f.write(f"Runs: {num_runs}, Steps per run: {test_steps}\n")
        f.write(f"COMM_OFF model: {MODEL_PATH_COMM_OFF}\n")
        f.write(f"COMM_ON model: {MODEL_PATH_COMM_ON}\n\n")
        for config_label, data in all_results.items():
            stats_list = data['stats']
            collisions = [s['collision_count'] for s in stats_list]
            successes = [s['success_count'] for s in stats_list]
            rewards = [s['avg_reward'] for s in stats_list]
            f.write(f"--- {config_label} ({data['n_radar']} radar + {data['n_comm']} comm) ---\n")
            f.write(f"Collision: {np.mean(collisions):.2f} +/- {np.std(collisions):.2f}\n")
            f.write(f"Success: {np.mean(successes):.2f} +/- {np.std(successes):.2f}\n")
            f.write(f"Avg Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}\n")
            comp = data.get('compliance')
            if comp:
                f.write(f"COLREGs Compliance: {comp['overall_compliance_rate']*100:.1f}%\n")
                f.write(f"Avg DCPA: {comp['dcpa']['avg_min_distance']:.2f}m\n")
            f.write(f"Per-run:\n")
            for i, s in enumerate(stats_list):
                f.write(f"  Run {i+1}: Col={s['collision_count']}, Suc={s['success_count']}, "
                        f"Rwd={s['avg_reward']:.2f}, R_rwd={s['radar_avg_reward']:.2f}, C_rwd={s['comm_avg_reward']:.2f}\n")
            f.write("\n")

    print(f"[SAVED] {summary_path}")
    return all_results


def run_simulation(model_path=None, max_steps=10_000_000, time_scale=5.0, log_interval_sec=10.0):
    """
    세계지도 ship traffic 시뮬레이션 모드.
    - 동적 vessel count (Unity 측 Progressive Spawn과 연동)
    - 데이터 수집 없음 (가벼운 inference only)
    - 무한에 가까운 step (기본 1천만)
    - 배가 추가/제거되어도 MultiAgentFrameStack이 dict 기반이라 자동 대응
    """
    import time

    print("=" * 80)
    print(f"[SIMULATION MODE] World map ship traffic simulation")
    print(f"max_steps={max_steps}, time_scale={time_scale}x")
    print("=" * 80)

    if model_path is None:
        model_path = TEST_MODEL_PATH
    if model_path is None or not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    # Unity 연결
    print("\n[INFO] Connecting to Unity...")
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=None, side_channels=[channel], worker_id=0,
                           base_port=BASE_PORT, timeout_wait=60)
    channel.set_configuration_parameters(time_scale=time_scale)
    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    print(f"[OK] Connected. Behavior: {behavior_name}")

    # 모델 로드
    print(f"[INFO] Loading: {model_path}")
    policy = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()
    frame_stack = MultiAgentFrameStack(FRAMES, STATE_SIZE)
    print(f"[OK] Model loaded. Communication: {'ON' if USE_COMMUNICATION else 'OFF'}")
    print(f"[OK] Starting simulation loop. Ctrl+C to stop.\n")

    last_log_time = time.time()
    start_time = last_log_time
    total_terminal = 0

    try:
        for step in range(max_steps):
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            agent_ids = list(decision_steps.agent_id)
            n_agents = len(agent_ids)

            # Terminal agents: frame_stack 정리
            for t_agent_id in terminal_steps.agent_id:
                frame_stack.remove_agent(t_agent_id)
                total_terminal += 1

            if n_agents == 0:
                env.step()
                continue

            # Batch state 구성
            batch_states, batch_goals, batch_self, batch_colregs, batch_pos = [], [], [], [], []
            for idx, agent_id in enumerate(agent_ids):
                obs_raw = decision_steps.obs[0][idx]
                state, goal, self_state, colregs, _, position = parse_observation(obs_raw)
                state_stacked = frame_stack.update(agent_id, state)
                batch_states.append(state_stacked)
                batch_goals.append(goal)
                batch_self.append(self_state)
                batch_colregs.append(colregs)
                batch_pos.append((position[0], position[1]))

            # 통신 파트너 (거리 기반)
            positions_dict = {agent_ids[i]: batch_pos[i] for i in range(n_agents)}
            comm_partners = {}
            if USE_COMMUNICATION:
                for agent_id in agent_ids:
                    comm_partners[agent_id] = get_comm_partners(
                        agent_id, positions_dict[agent_id], positions_dict)

            # Inference
            st_t = torch.FloatTensor(np.array(batch_states)).unsqueeze(0).to(DEVICE)
            gl_t = torch.FloatTensor(np.array(batch_goals)).unsqueeze(0).to(DEVICE)
            sf_t = torch.FloatTensor(np.array(batch_self)).unsqueeze(0).to(DEVICE)
            cr_t = torch.FloatTensor(np.array(batch_colregs)).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                if USE_COMMUNICATION:
                    _, actions, _, _, _ = policy.forward(
                        st_t, gl_t, sf_t, cr_t,
                        comm_partners=comm_partners,
                        agent_id_list=list(agent_ids))
                else:
                    _, actions, _, _, _ = policy.forward(st_t, gl_t, sf_t, cr_t)

            actions_np = actions.squeeze(0).cpu().numpy()
            action_tuple = ActionTuple(continuous=actions_np)
            env.set_actions(behavior_name, action_tuple)
            env.step()

            # 주기적 로그
            now = time.time()
            if now - last_log_time >= log_interval_sec:
                elapsed = now - start_time
                sps = (step + 1) / elapsed
                print(f"[step={step:>8d}] active={n_agents:>4d}  terminated_total={total_terminal:>6d}  "
                      f"sps={sps:5.1f}  elapsed={elapsed:6.1f}s")
                last_log_time = now

    except KeyboardInterrupt:
        print("\n[STOP] Simulation interrupted by user.")
    finally:
        env.close()
        print("[CLOSED] Unity environment closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Vessel ML-Agent')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model (.pth)')
    parser.add_argument('--steps', type=int, default=TRAJECTORY_STEPS, help='Steps for trajectory collection')
    parser.add_argument('--time_scale', type=float, default=20.0, help='Simulation speed')
    parser.add_argument('--analyze', type=str, default=None, help='Path to existing CSV for analysis only')
    parser.add_argument('--multitest', action='store_true', help='Run multi-test mode instead of trajectory')
    parser.add_argument('--compare', action='store_true', help='Run Comm OFF vs ON comparison test')
    parser.add_argument('--mixed', action='store_true', help='Run mixed model experiment (radar + comm)')
    parser.add_argument('--simulation', action='store_true', help='World-map ship traffic simulation (infinite loop, dynamic vessels)')
    parser.add_argument('--runs', type=int, default=NUM_RUNS, help='Number of runs (multi-test mode)')
    parser.add_argument('--tag', type=str, default=None, help='Tag prefix for output files (e.g., narrow, open)')
    parser.add_argument('--compliance', type=str, default=None, help='Path to trajectory CSV for COLREGs compliance evaluation')

    args = parser.parse_args()

    if args.compliance:
        evaluate_colregs_compliance(args.compliance)
    elif args.analyze:
        analyze_tsne_with_observation(args.analyze)
    elif args.simulation:
        run_simulation(
            model_path=args.model,
            max_steps=args.steps if args.steps != TRAJECTORY_STEPS else 10_000_000,
            time_scale=args.time_scale,
        )
    elif args.mixed:
        run_mixed_experiment(
            num_runs=args.runs,
            test_steps=args.steps if args.steps != TRAJECTORY_STEPS else TEST_STEPS,
            time_scale=args.time_scale,
            tag=args.tag
        )
    elif args.compare:
        run_comparison_test(
            test_steps=args.steps if args.steps != TRAJECTORY_STEPS else TEST_STEPS,
            num_runs=args.runs,
            time_scale=args.time_scale,
            tag=args.tag
        )
    elif args.multitest:
        run_multi_test(
            model_path=args.model,
            test_steps=args.steps,
            num_runs=args.runs,
            time_scale=args.time_scale
        )
    else:
        collect_trajectory(
            model_path=args.model,
            test_steps=args.steps,
            time_scale=args.time_scale
        )
