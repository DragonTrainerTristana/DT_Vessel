"""
논문용 그래프 전체 생성 스크립트
Wang & Zhao (2025) 스타일 참조

생성 그래프:
  Fig 1: Training Reward Curve (COMM_NON vs COMM_YES)
  Fig 2: Loss Curves (policy, value, entropy, colregs)
  Fig 3: Trajectory Comparison (COMM_OFF vs COMM_ON)
  Fig 4: COLREGs Encounter Detail (HeadOn, CrossStandOn, CrossGiveWay, Overtaking)
  Fig 5: Communication Message Analysis
  Fig 6: Distance / COLREGs Comparison Bar Charts
  Fig 7: Communication Spatial Heatmap
  Fig 8: Communication Vector Regression
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy import stats
from scipy.ndimage import gaussian_filter

# ============================================================================
# 경로 설정
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# 모델 경로
COMM_NON_DIR = os.path.join(PROJECT_ROOT, "models", "COMM_NON", "VesselNavigation_20260114_183130")
COMM_YES_DIR = os.path.join(PROJECT_ROOT, "models", "COMM_YES", "VesselNavigation_20260119_151615")

# 트레이닝 로그
TRAIN_LOG_NON = os.path.join(COMM_NON_DIR, "csv_logs", "training_logs.csv")
TRAIN_LOG_YES = os.path.join(COMM_YES_DIR, "csv_logs", "training_logs.csv")

# Tensorboard 로그
TB_LOG_NON = os.path.join(COMM_NON_DIR, "logs")
TB_LOG_YES = os.path.join(COMM_YES_DIR, "logs")

# 궤적 데이터
TRAJ_OFF = os.path.join(PROJECT_ROOT, "trajectory_data", "commOFF.csv")
TRAJ_ON = os.path.join(PROJECT_ROOT, "trajectory_data", "commON.csv")

# ============================================================================
# 스타일 설정
# ============================================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
})

# 색상 팔레트
COLOR_NON = '#2E86AB'    # 파란색 (No Communication)
COLOR_YES = '#E94F37'    # 빨간색 (With Communication)
COLREGS_COLORS = {
    'None': '#999999',
    'HeadOn': '#E94F37',
    'CrossStandOn': '#2E86AB',
    'CrossGiveWay': '#F6AE2D',
    'Overtaking': '#33A852',
}
AGENT_COLORS = plt.cm.Set2(np.linspace(0, 1, 8))


# ============================================================================
# 유틸리티 함수
# ============================================================================
def smooth_curve(values, weight=0.95):
    """지수 이동 평균 스무딩"""
    smoothed = []
    last = values[0]
    for v in values:
        s = last * weight + (1 - weight) * v
        smoothed.append(s)
        last = s
    return np.array(smoothed)


def rolling_std(values, window=50):
    """이동 표준편차 (shaded area 용)"""
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1, center=True).std().values


def parse_tensor_value(val):
    """'tensor(4.4230)' 형태 파싱"""
    if isinstance(val, str) and val.startswith('tensor('):
        return float(val.replace('tensor(', '').replace(')', ''))
    return float(val)


def load_training_logs():
    """트레이닝 로그 로드"""
    df_non = pd.read_csv(TRAIN_LOG_NON)
    df_yes = pd.read_csv(TRAIN_LOG_YES)
    df_non['gradient_norm'] = df_non['gradient_norm'].apply(parse_tensor_value)
    df_yes['gradient_norm'] = df_yes['gradient_norm'].apply(parse_tensor_value)
    return df_non, df_yes


def load_trajectory_data():
    """궤적 데이터 로드"""
    df_off = pd.read_csv(TRAJ_OFF)
    df_on = pd.read_csv(TRAJ_ON)
    return df_off, df_on


def save_figure(fig, name):
    """PNG + PDF 저장"""
    png_path = os.path.join(FIGURE_DIR, f"{name}.png")
    pdf_path = os.path.join(FIGURE_DIR, f"{name}.pdf")
    fig.savefig(png_path, facecolor='white')
    fig.savefig(pdf_path)
    print(f"  Saved: {png_path}")
    plt.close(fig)


# ============================================================================
# Fig 1: Training Reward Curve (Tensorboard)
# ============================================================================
def plot_reward_curve():
    """학습 보상 곡선 - COMM_NON vs COMM_YES"""
    print("[Fig 1] Training Reward Curve...")

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("  tensorboard 미설치 - 스킵")
        return

    def load_tb(log_dir, tag='Reward/Step'):
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        available = ea.Tags().get('scalars', [])
        if tag not in available:
            # 유사 태그 검색
            reward_tags = [t for t in available if 'reward' in t.lower() or 'Reward' in t]
            if reward_tags:
                tag = reward_tags[0]
            else:
                print(f"  Available tags: {available[:10]}")
                return None, None
        data = ea.Scalars(tag)
        steps = np.array([d.step for d in data])
        values = np.array([d.value for d in data])
        return steps, values

    steps_non, rewards_non = load_tb(TB_LOG_NON)
    steps_yes, rewards_yes = load_tb(TB_LOG_YES)

    if steps_non is None or steps_yes is None:
        print("  데이터 로드 실패")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # 단위: millions
    s_non = steps_non / 1e6
    s_yes = steps_yes / 1e6

    # 스무딩
    r_non_smooth = smooth_curve(rewards_non, 0.95)
    r_yes_smooth = smooth_curve(rewards_yes, 0.95)

    # Shaded std
    std_non = rolling_std(rewards_non, window=50)
    std_yes = rolling_std(rewards_yes, window=50)

    ax.fill_between(s_non, r_non_smooth - std_non, r_non_smooth + std_non,
                     alpha=0.15, color=COLOR_NON)
    ax.fill_between(s_yes, r_yes_smooth - std_yes, r_yes_smooth + std_yes,
                     alpha=0.15, color=COLOR_YES)

    ax.plot(s_non, r_non_smooth, label='Without Communication', color=COLOR_NON, linewidth=2)
    ax.plot(s_yes, r_yes_smooth, label='With Communication', color=COLOR_YES, linewidth=2)

    # Phase 전환선
    transition = steps_yes[0] / 1e6
    ax.axvline(x=transition, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax.annotate('Phase 2 Start', xy=(transition, ax.get_ylim()[1] * 0.9),
                fontsize=9, ha='left', color='gray',
                xytext=(transition + 0.3, ax.get_ylim()[1] * 0.9))

    ax.set_xlabel('Training Steps (×$10^6$)')
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Reward Curve')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    save_figure(fig, "fig1_reward_curve")


# ============================================================================
# Fig 2: Loss Curves
# ============================================================================
def plot_loss_curves():
    """손실 함수 곡선 (policy, value, entropy, colregs)"""
    print("[Fig 2] Loss Curves...")

    df_non, df_yes = load_training_logs()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    loss_configs = [
        ('policy_loss', 'Policy Loss', axes[0, 0]),
        ('value_loss', 'Value Loss', axes[0, 1]),
        ('entropy_loss', 'Entropy Loss', axes[1, 0]),
        ('colregs_loss', 'COLREGs Classification Loss', axes[1, 1]),
    ]

    for col, title, ax in loss_configs:
        s_non = df_non['total_steps'].values / 1e6
        s_yes = df_yes['total_steps'].values / 1e6
        v_non = smooth_curve(df_non[col].values, 0.9)
        v_yes = smooth_curve(df_yes[col].values, 0.9)

        ax.plot(s_non, v_non, label='Without Comm', color=COLOR_NON, linewidth=1.5)
        ax.plot(s_yes, v_yes, label='With Comm', color=COLOR_YES, linewidth=1.5)

        # Phase 전환선
        transition = df_yes['total_steps'].values[0] / 1e6
        ax.axvline(x=transition, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        ax.set_xlabel('Training Steps (×$10^6$)')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Training Loss Curves', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig2_loss_curves")


# ============================================================================
# Fig 3: Trajectory Comparison
# ============================================================================
def plot_trajectory_comparison():
    """궤적 비교: COMM_OFF vs COMM_ON"""
    print("[Fig 3] Trajectory Comparison...")

    df_off, df_on = load_trajectory_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, df, title in [(ax1, df_off, 'Without Communication'),
                           (ax2, df_on, 'With Communication')]:
        for aid in range(8):
            agent = df[df.agent_id == aid]
            x = agent['x'].values
            z = agent['z'].values
            color = AGENT_COLORS[aid]

            ax.plot(x, z, color=color, linewidth=1.0, alpha=0.8, label=f'Vessel {aid}')
            # 시작점
            ax.scatter(x[0], z[0], color=color, marker='o', s=60, zorder=5, edgecolors='black', linewidths=0.5)
            # 끝점
            ax.scatter(x[-1], z[-1], color=color, marker='s', s=60, zorder=5, edgecolors='black', linewidths=0.5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(title, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right', ncol=2)

    fig.suptitle('Vessel Trajectories Comparison', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig3_trajectory_comparison")


# ============================================================================
# Fig 4: COLREGs Encounter Detail
# ============================================================================
def plot_colregs_encounters():
    """COLREGs 상황별 궤적 비교"""
    print("[Fig 4] COLREGs Encounter Details...")

    df_off, df_on = load_trajectory_data()

    encounter_types = ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for col_idx, enc_type in enumerate(encounter_types):
        for row_idx, (df, mode_label) in enumerate([(df_off, 'No Comm'), (df_on, 'With Comm')]):
            ax = axes[row_idx, col_idx]

            # 해당 COLREGs 상황이 발생한 에이전트-스텝 찾기
            enc_mask = df['colregs_name'] == enc_type
            enc_agents = df.loc[enc_mask, 'agent_id'].unique()

            if len(enc_agents) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{enc_type}\n({mode_label})', fontsize=10)
                continue

            # 관련 에이전트의 전체 궤적 그리기
            for aid in range(8):
                agent = df[df.agent_id == aid]
                x, z = agent['x'].values, agent['z'].values

                if aid in enc_agents:
                    # 관련 에이전트: 굵게
                    ax.plot(x, z, color=AGENT_COLORS[aid], linewidth=1.5, alpha=0.9,
                            label=f'V{aid}')

                    # COLREGs 발생 구간 하이라이트
                    enc_steps = df.loc[enc_mask & (df.agent_id == aid), 'step'].values
                    if len(enc_steps) > 0:
                        enc_data = agent[agent['step'].isin(enc_steps)]
                        ax.scatter(enc_data['x'], enc_data['z'],
                                   color=COLREGS_COLORS[enc_type], s=8, alpha=0.5, zorder=4)
                else:
                    # 비관련 에이전트: 얇게
                    ax.plot(x, z, color='lightgray', linewidth=0.5, alpha=0.4)

                # 시작/끝점
                ax.scatter(x[0], z[0], color=AGENT_COLORS[aid], marker='o', s=30,
                           zorder=5, edgecolors='black', linewidths=0.3)

            ax.set_title(f'{enc_type}\n({mode_label})', fontsize=10, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            if col_idx == 0:
                ax.set_ylabel('Z (m)')
            if row_idx == 1:
                ax.set_xlabel('X (m)')
            if len(enc_agents) <= 6:
                ax.legend(fontsize=7, loc='best')

    fig.suptitle('Trajectories by COLREGs Encounter Type', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig4_colregs_encounters")


# ============================================================================
# Fig 5: Communication Message Analysis
# ============================================================================
def plot_communication_analysis():
    """통신 메시지 분석 (magnitude, per-colregs, per-agent)"""
    print("[Fig 5] Communication Message Analysis...")

    df_on = pd.read_csv(TRAJ_ON)

    msg_cols = [f'self_msg_{i}' for i in range(6)]
    others_cols = [f'others_msg_{i}' for i in range(6)]

    df_on['self_msg_norm'] = np.sqrt((df_on[msg_cols] ** 2).sum(axis=1))
    df_on['others_msg_norm'] = np.sqrt((df_on[others_cols] ** 2).sum(axis=1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- (a) Message Norm over Steps ---
    ax = axes[0]
    for aid in range(8):
        agent = df_on[df_on.agent_id == aid]
        ax.plot(agent['step'], agent['self_msg_norm'],
                color=AGENT_COLORS[aid], linewidth=0.8, alpha=0.7, label=f'V{aid}')

    ax.set_xlabel('Step')
    ax.set_ylabel('Message Norm ($||m||_2$)')
    ax.set_title('(a) Self Message Magnitude')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- (b) Message Norm by COLREGs Situation ---
    ax = axes[1]
    colregs_names = ['None', 'HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']

    box_data = []
    box_labels = []
    box_colors = []
    for name in colregs_names:
        if name == 'None':
            mask = df_on['colregs_name'].isna()
        else:
            mask = df_on['colregs_name'] == name
        data = df_on.loc[mask, 'self_msg_norm'].values
        if len(data) > 0:
            box_data.append(data)
            box_labels.append(name)
            box_colors.append(COLREGS_COLORS[name])

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Message Norm ($||m||_2$)')
    ax.set_title('(b) Message Norm by COLREGs')
    ax.grid(True, alpha=0.3, axis='y')

    # --- (c) Received Message Norm vs Self Message Norm ---
    ax = axes[2]
    sample = df_on.sample(min(5000, len(df_on)), random_state=42)
    sc = ax.scatter(sample['self_msg_norm'], sample['others_msg_norm'],
                    c=sample['step'], cmap='viridis', s=5, alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Step')
    ax.set_xlabel('Self Message Norm')
    ax.set_ylabel('Received Message Norm')
    ax.set_title('(c) Self vs Received Message')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Communication Vector Analysis', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig5_communication_analysis")


# ============================================================================
# Fig 6: Distance & COLREGs Comparison Bar Charts
# ============================================================================
def plot_bar_comparisons():
    """거리 비교 및 COLREGs 발생률 바 차트"""
    print("[Fig 6] Bar Chart Comparisons...")

    df_off, df_on = load_trajectory_data()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- (a) Total Distance per Agent ---
    ax = axes[0]
    dist_off = []
    dist_on = []
    for aid in range(8):
        agent_off = df_off[df_off.agent_id == aid]
        agent_on = df_on[df_on.agent_id == aid]
        d_off = np.sqrt(np.diff(agent_off.x.values)**2 + np.diff(agent_off.z.values)**2).sum()
        d_on = np.sqrt(np.diff(agent_on.x.values)**2 + np.diff(agent_on.z.values)**2).sum()
        dist_off.append(d_off)
        dist_on.append(d_on)

    x_pos = np.arange(8)
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, dist_off, width, label='No Comm', color=COLOR_NON, alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, dist_on, width, label='With Comm', color=COLOR_YES, alpha=0.8)

    # T-test
    t_stat, p_val = stats.ttest_ind(dist_off, dist_on)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    ax.text(0.5, 0.95, f't={t_stat:.2f}, p={p_val:.4f} ({sig})',
            transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Vessel ID')
    ax.set_ylabel('Total Distance (m)')
    ax.set_title('(a) Total Distance Traveled')
    ax.set_xticks(x_pos)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # --- (b) Average Distance Comparison ---
    ax = axes[1]
    means = [np.mean(dist_off), np.mean(dist_on)]
    stds = [np.std(dist_off), np.std(dist_on)]
    bars = ax.bar(['No Comm', 'With Comm'], means, yerr=stds,
                   color=[COLOR_NON, COLOR_YES], alpha=0.8, capsize=8, edgecolor='black', linewidth=0.5)

    ax.text(0.5, 0.95, f'p={p_val:.4f} ({sig})',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_ylabel('Avg Total Distance (m)')
    ax.set_title('(b) Average Distance (T-test)')
    ax.grid(True, alpha=0.3, axis='y')

    # --- (c) COLREGs Encounter Counts ---
    ax = axes[2]
    encounter_types = ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']
    counts_off = []
    counts_on = []
    for enc in encounter_types:
        counts_off.append(len(df_off[df_off.colregs_name == enc]))
        counts_on.append(len(df_on[df_on.colregs_name == enc]))

    x_pos = np.arange(len(encounter_types))
    bars1 = ax.bar(x_pos - width/2, counts_off, width, label='No Comm', color=COLOR_NON, alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, counts_on, width, label='With Comm', color=COLOR_YES, alpha=0.8)

    ax.set_xlabel('COLREGs Situation')
    ax.set_ylabel('Count (steps)')
    ax.set_title('(c) COLREGs Encounter Frequency')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(encounter_types, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig6_bar_comparisons")


# ============================================================================
# Fig 7: Communication Spatial Heatmap
# ============================================================================
def plot_communication_heatmap():
    """통신 벡터 크기의 공간 분포 히트맵"""
    print("[Fig 7] Communication Spatial Heatmap...")

    df_on = pd.read_csv(TRAJ_ON)
    msg_cols = [f'self_msg_{i}' for i in range(6)]
    df_on['msg_norm'] = np.sqrt((df_on[msg_cols] ** 2).sum(axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- (a) Self Message Norm Heatmap ---
    ax = axes[0]
    x = df_on['x'].values
    z = df_on['z'].values
    norms = df_on['msg_norm'].values

    # 2D 히스토그램으로 평균 메시지 크기 계산
    x_bins = np.linspace(x.min() - 10, x.max() + 10, 80)
    z_bins = np.linspace(z.min() - 10, z.max() + 10, 80)

    sum_grid, _, _ = np.histogram2d(x, z, bins=[x_bins, z_bins], weights=norms)
    count_grid, _, _ = np.histogram2d(x, z, bins=[x_bins, z_bins])
    count_grid[count_grid == 0] = 1  # division by zero 방지
    mean_grid = sum_grid / count_grid

    # 가우시안 스무딩
    mean_grid_smooth = gaussian_filter(mean_grid.T, sigma=1.5)

    im = ax.imshow(mean_grid_smooth, extent=[x_bins[0], x_bins[-1], z_bins[0], z_bins[-1]],
                    origin='lower', cmap='hot', aspect='equal', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Avg Message Norm')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('(a) Self Message Magnitude')

    # --- (b) Received Message Norm Heatmap ---
    ax = axes[1]
    others_cols = [f'others_msg_{i}' for i in range(6)]
    df_on['others_norm'] = np.sqrt((df_on[others_cols] ** 2).sum(axis=1))
    others_norms = df_on['others_norm'].values

    sum_grid2, _, _ = np.histogram2d(x, z, bins=[x_bins, z_bins], weights=others_norms)
    mean_grid2 = sum_grid2 / count_grid
    mean_grid2_smooth = gaussian_filter(mean_grid2.T, sigma=1.5)

    im2 = ax.imshow(mean_grid2_smooth, extent=[x_bins[0], x_bins[-1], z_bins[0], z_bins[-1]],
                     origin='lower', cmap='hot', aspect='equal', interpolation='bilinear')
    plt.colorbar(im2, ax=ax, label='Avg Received Msg Norm')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('(b) Received Message Magnitude')

    fig.suptitle('Spatial Distribution of Communication Messages',
                  fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig7_communication_heatmap")


# ============================================================================
# Fig 8: Communication Vector Regression
# ============================================================================
def plot_communication_regression():
    """통신 벡터 크기 vs 에이전트 간 거리 회귀 분석"""
    print("[Fig 8] Communication Vector Regression...")

    df_on = pd.read_csv(TRAJ_ON)
    msg_cols = [f'self_msg_{i}' for i in range(6)]
    others_cols = [f'others_msg_{i}' for i in range(6)]
    df_on['self_norm'] = np.sqrt((df_on[msg_cols] ** 2).sum(axis=1))
    df_on['others_norm'] = np.sqrt((df_on[others_cols] ** 2).sum(axis=1))

    # 각 스텝에서 에이전트 간 최소 거리 계산
    steps = df_on['step'].unique()
    min_distances = []
    self_norms = []
    others_norms = []

    # 샘플링 (전체 스텝은 너무 많음)
    sample_steps = np.random.RandomState(42).choice(steps, min(2000, len(steps)), replace=False)
    sample_steps.sort()

    for step in sample_steps:
        step_data = df_on[df_on.step == step]
        positions = step_data[['x', 'z']].values
        n = len(positions)
        if n < 2:
            continue

        # 모든 에이전트 쌍 거리 계산
        for i in range(n):
            dists = np.sqrt(((positions[i] - positions) ** 2).sum(axis=1))
            dists[i] = np.inf
            min_d = dists.min()
            min_distances.append(min_d)
            self_norms.append(step_data.iloc[i]['self_norm'])
            others_norms.append(step_data.iloc[i]['others_norm'])

    min_distances = np.array(min_distances)
    self_norms = np.array(self_norms)
    others_norms = np.array(others_norms)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- (a) Self Message Norm vs Min Distance ---
    ax = ax1
    ax.scatter(min_distances, self_norms, s=3, alpha=0.2, color=COLOR_YES)

    # 선형 회귀
    valid = np.isfinite(min_distances) & np.isfinite(self_norms)
    slope, intercept, r_value, p_value, _ = stats.linregress(min_distances[valid], self_norms[valid])
    x_line = np.linspace(min_distances[valid].min(), min_distances[valid].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2,
            label=f'y={slope:.5f}x+{intercept:.4f}\n$R^2$={r_value**2:.4f}')

    ax.set_xlabel('Min Distance to Nearest Vessel (m)')
    ax.set_ylabel('Self Message Norm')
    ax.set_title('(a) Self Message vs Distance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (b) Received Message Norm vs Min Distance ---
    ax = ax2
    ax.scatter(min_distances, others_norms, s=3, alpha=0.2, color=COLOR_NON)

    slope2, intercept2, r2, p2, _ = stats.linregress(min_distances[valid], others_norms[valid])
    ax.plot(x_line, slope2 * x_line + intercept2, 'k--', linewidth=2,
            label=f'y={slope2:.5f}x+{intercept2:.4f}\n$R^2$={r2**2:.4f}')

    ax.set_xlabel('Min Distance to Nearest Vessel (m)')
    ax.set_ylabel('Received Message Norm')
    ax.set_title('(b) Received Message vs Distance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Communication Message Regression Analysis',
                  fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig8_communication_regression")


# ============================================================================
# Fig 9: Gradient Norm & Learning Rate
# ============================================================================
def plot_training_diagnostics():
    """학습 진단 - Gradient Norm, Learning Rate"""
    print("[Fig 9] Training Diagnostics...")

    df_non, df_yes = load_training_logs()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- (a) Gradient Norm ---
    s_non = df_non['total_steps'].values / 1e6
    s_yes = df_yes['total_steps'].values / 1e6

    # 클리핑해서 보기 좋게 (극단 outlier 제거)
    gn_non = np.clip(df_non['gradient_norm'].values, 0, np.percentile(df_non['gradient_norm'].values, 99))
    gn_yes = np.clip(df_yes['gradient_norm'].values, 0, np.percentile(df_yes['gradient_norm'].values, 99))

    ax1.plot(s_non, smooth_curve(gn_non, 0.9), color=COLOR_NON, linewidth=1, label='No Comm', alpha=0.8)
    ax1.plot(s_yes, smooth_curve(gn_yes, 0.9), color=COLOR_YES, linewidth=1, label='With Comm', alpha=0.8)
    ax1.axvline(x=s_yes[0], color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.set_xlabel('Training Steps (×$10^6$)')
    ax1.set_ylabel('Gradient Norm')
    ax1.set_title('(a) Gradient Norm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- (b) Total Loss ---
    tl_non = smooth_curve(df_non['total_loss'].values, 0.9)
    tl_yes = smooth_curve(df_yes['total_loss'].values, 0.9)
    ax2.plot(s_non, tl_non, color=COLOR_NON, linewidth=1.5, label='No Comm')
    ax2.plot(s_yes, tl_yes, color=COLOR_YES, linewidth=1.5, label='With Comm')
    ax2.axvline(x=s_yes[0], color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.set_xlabel('Training Steps (×$10^6$)')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('(b) Total Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Training Diagnostics', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig9_training_diagnostics")


# ============================================================================
# Fig 10: Communication Message Dimension Analysis
# ============================================================================
def plot_message_dimensions():
    """6D 메시지 벡터 각 차원 분석"""
    print("[Fig 10] Message Dimension Analysis...")

    df_on = pd.read_csv(TRAJ_ON)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for dim in range(6):
        ax = axes[dim // 3, dim % 3]
        col = f'self_msg_{dim}'

        # 시간에 따른 변화 (에이전트별 평균)
        step_means = df_on.groupby('step')[col].mean()
        step_stds = df_on.groupby('step')[col].std()

        ax.fill_between(step_means.index,
                         step_means.values - step_stds.values,
                         step_means.values + step_stds.values,
                         alpha=0.2, color=COLOR_YES)
        ax.plot(step_means.index, step_means.values, color=COLOR_YES, linewidth=1)

        # COLREGs별 분포 (inset boxplot)
        colregs_data = {}
        for name in ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']:
            mask = df_on['colregs_name'] == name
            if mask.sum() > 0:
                colregs_data[name] = df_on.loc[mask, col].values

        ax.set_xlabel('Step')
        ax.set_ylabel(f'$m_{dim}$')
        ax.set_title(f'Message Dim {dim}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    fig.suptitle('6D Message Vector: Per-Dimension Analysis',
                  fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig10_message_dimensions")


# ============================================================================
# 메인 실행
# ============================================================================
def main():
    print("=" * 60)
    print("논문용 그래프 생성 시작")
    print(f"저장 위치: {FIGURE_DIR}")
    print("=" * 60)

    # 순서대로 생성
    plot_reward_curve()       # Fig 1
    plot_loss_curves()        # Fig 2
    plot_trajectory_comparison()  # Fig 3
    plot_colregs_encounters()     # Fig 4
    plot_communication_analysis() # Fig 5
    plot_bar_comparisons()        # Fig 6
    plot_communication_heatmap()  # Fig 7
    plot_communication_regression()  # Fig 8
    plot_training_diagnostics()      # Fig 9
    plot_message_dimensions()        # Fig 10

    print("=" * 60)
    print("전체 그래프 생성 완료!")
    print(f"저장 위치: {FIGURE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
