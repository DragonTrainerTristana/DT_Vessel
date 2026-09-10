"""
Vessel Multi-Agent RL - Experiment Analysis (v3)
- PNG: 개별 서브플롯 저장
- PDF: 합본 저장
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TRAJ_OFF = os.path.join(PROJECT_ROOT, "trajectory_data", "commOFF.csv")
TRAJ_ON = os.path.join(PROJECT_ROOT, "trajectory_data", "commON.csv")
LATENT_CSV = os.path.join(PROJECT_ROOT, "latent_data", "multirun_10x2000_20260113_185310.csv")
SAVE_DIR = os.path.join(PROJECT_ROOT, "figures", "분석 그래프")
os.makedirs(SAVE_DIR, exist_ok=True)

# Tensorboard log paths
LOG_COMM_NON = os.path.join(PROJECT_ROOT, "models", "COMM_NON", "VesselNavigation_20260114_183130", "logs")
LOG_COMM_YES = os.path.join(PROJECT_ROOT, "models", "COMM_YES", "VesselNavigation_20260119_151615", "logs")

MSG_DIM = 6
OBS_DIM = 369
COLREGS_LABELS = ['None', 'HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']
COLREGS_COLORS = ['#AAAAAA', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12']


def save_individual(fig, ax_or_axes, name, dpi=200):
    """개별 서브플롯을 PNG로 저장"""
    path = os.path.join(SAVE_DIR, f'{name}.png')
    fig_single = plt.figure(figsize=(7, 6))
    ax_new = fig_single.add_subplot(111)

    # 기존 ax의 내용을 새 figure에 복사할 수 없으므로,
    # 대신 개별 플롯 함수에서 직접 저장
    return path


def load_latent_data():
    df = pd.read_csv(LATENT_CSV)
    self_msg = df[[f'self_msg_{i}' for i in range(MSG_DIM)]].values
    others_msg = df[[f'others_msg_{i}' for i in range(MSG_DIM)]].values
    obs_full = df[[f'obs_{i}' for i in range(OBS_DIM)]].values
    colregs = np.argmax(df[[f'obs_{i}' for i in range(364, 369)]].values, axis=1)

    return {
        'df': df, 'obs': obs_full,
        'self_msg': self_msg, 'others_msg': others_msg, 'colregs': colregs,
        'action_0': df['action_0'].values, 'action_1': df['action_1'].values,
        'goal_dist': df['obs_360'].values, 'goal_angle': df['obs_361'].values,
        'speed': df['obs_362'].values, 'yaw_rate': df['obs_363'].values,
        'steps': df['step'].values,
    }


def extract_first_episode(df, agent_id):
    a = df[df.agent_id == agent_id].reset_index(drop=True)
    dx = a.x.diff().abs().fillna(0)
    dz = a.z.diff().abs().fillna(0)
    jumps = (dx > 50) | (dz > 50)
    first_jump = jumps[jumps].index[0] if jumps.any() else len(a)
    return a.iloc[:first_jump]


def _save_scatter_colregs(d2d, colregs_s, title, filename, labels=None):
    """COLREGs 색상 scatter -> 개별 PNG"""
    if labels is None:
        labels = COLREGS_LABELS
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    use_indices = range(5) if len(labels) == 5 else [1, 2, 3, 4]
    for i in ([0, 4, 3, 2, 1] if 0 in use_indices else [4, 3, 2, 1]):
        mask = colregs_s == i
        if mask.sum() > 0:
            ax.scatter(d2d[mask, 0], d2d[mask, 1],
                      c=COLREGS_COLORS[i], label=COLREGS_LABELS[i],
                      alpha=0.35 if i == 0 else 0.8,
                      s=6 if i == 0 else 18,
                      edgecolors='none', zorder=2 if i == 0 else 3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f'{filename}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def _save_scatter_continuous(d2d, values, title, cbar_label, filename, cmap='RdYlBu_r'):
    """연속값 색상 scatter -> 개별 PNG"""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(d2d[:, 0], d2d[:, 1], c=values, cmap=cmap,
                   alpha=0.6, s=10, edgecolors='none')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.85)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f'{filename}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


# ============================================================================
# Figure 1: Training Reward Curve
# ============================================================================
def plot_fig1_reward_curve():
    print("[Fig 1] Training reward curve")

    def load_tb_scalar(log_dir, tag):
        ea = EventAccumulator(log_dir)
        ea.Reload()
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        return steps, values

    # Phase 1 (COMM_NON) + Phase 2 (COMM_YES)
    s1, r1 = load_tb_scalar(LOG_COMM_NON, 'Reward/Step')
    s2, r2 = load_tb_scalar(LOG_COMM_YES, 'Reward/Step')

    # Phase 1을 0~8M으로 스케일, Phase 2를 8M~ 으로 이어붙이기
    s1 = s1 * 2.0  # 0~4M -> 0~8M
    phase1_end = s1[-1]
    s2 = phase1_end + (s2 - s2[0])  # 8M부터 이어서

    # 전체 이어붙여서 한번에 smoothing (빈 구간 방지)
    s_all = np.concatenate([s1, s2])
    r_all = np.concatenate([r1, r2])
    n1 = len(s1)

    def smooth(y, window=50):
        if len(y) < window:
            return y
        kernel = np.ones(window) / window
        return np.convolve(y, kernel, mode='valid')

    s_all_sm = s_all[:len(smooth(r_all))]
    r_all_sm = smooth(r_all)

    # Phase 경계 기준으로 분리 (smoothed)
    boundary = phase1_end
    mask1 = s_all_sm <= boundary
    mask2 = s_all_sm >= boundary  # 경계점 포함 → 이어짐

    fig, ax = plt.subplots(figsize=(10, 5))

    # 원본 (옅게)
    ax.plot(s1 / 1e6, r1, color='#E74C3C', alpha=0.15, linewidth=0.5)
    ax.plot(s2 / 1e6, r2, color='#3498DB', alpha=0.15, linewidth=0.5)

    # Smoothed (경계에서 이어짐)
    ax.plot(s_all_sm[mask1] / 1e6, r_all_sm[mask1], color='#E74C3C', linewidth=2.0, label='Phase 1: Without communication')
    ax.plot(s_all_sm[mask2] / 1e6, r_all_sm[mask2], color='#3498DB', linewidth=2.0, label='Phase 2: With communication')

    # Phase 전환선
    phase_boundary = s1[-1] / 1e6
    ax.axvline(phase_boundary, color='black', linestyle='--', alpha=0.6, linewidth=1.2)
    ax.text(phase_boundary + 0.1, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.5,
            'Communication\nenabled', fontsize=9, fontweight='bold', va='center')

    ax.set_xlabel('Training steps (M)', fontsize=11)
    ax.set_ylabel('Average reward per step', fontsize=11)
    ax.set_title('Training reward curve', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fig.savefig(os.path.join(SAVE_DIR, 'fig1_reward_curve.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, 'fig1_reward_curve.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig1_reward_curve.png / .pdf")


# ============================================================================
# Figure 2: t-SNE
# ============================================================================
def plot_fig2_tsne():
    print("[Fig 2] t-SNE analysis")
    data = load_latent_data()
    obs = data['obs']
    self_msg = data['self_msg']
    colregs = data['colregs']
    yaw_rate = data['yaw_rate']
    speed = data['speed']
    goal_angle = data['goal_angle']
    goal_dist = data['goal_dist']

    # 균형 샘플링
    rng = np.random.RandomState(42)
    max_per_class = 500
    balanced_idx = []
    for c in range(5):
        mask = np.where(colregs == c)[0]
        if len(mask) > 0:
            n = min(max_per_class, len(mask))
            balanced_idx.extend(rng.choice(mask, n, replace=False))
    balanced_idx = np.array(balanced_idx)
    rng.shuffle(balanced_idx)

    obs_s = obs[balanced_idx]
    msg_s = self_msg[balanced_idx]
    colregs_s = colregs[balanced_idx]
    yaw_s = yaw_rate[balanced_idx]
    speed_s = speed[balanced_idx]
    goal_angle_s = goal_angle[balanced_idx]
    goal_dist_s = goal_dist[balanced_idx]

    # t-SNE
    print("  t-SNE on Observation...")
    obs_2d = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(obs_s)
    print("  t-SNE on Self Message...")
    msg_2d = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(msg_s)

    encounter_mask = colregs_s > 0
    msg_enc = msg_s[encounter_mask]
    colregs_enc = colregs_s[encounter_mask]
    yaw_enc = yaw_s[encounter_mask]
    print(f"  t-SNE on Encounters ({encounter_mask.sum()})...")
    msg_enc_2d = TSNE(n_components=2, perplexity=min(30, encounter_mask.sum()-1),
                      random_state=42, max_iter=1000).fit_transform(msg_enc)

    # 개별 PNG 저장
    paths = []

    # (a) Observation - COLREGs
    paths.append(_save_scatter_colregs(
        obs_2d, colregs_s, f'Observation space ({OBS_DIM}D)', 'fig2_tsne_a_obs_colregs'))

    # (b) Message - COLREGs
    paths.append(_save_scatter_colregs(
        msg_2d, colregs_s, f'Latent message space ({MSG_DIM}D)', 'fig2_tsne_b_msg_colregs'))

    # (c) Encounter only - COLREGs
    paths.append(_save_scatter_colregs(
        msg_enc_2d, colregs_enc, 'Message space (encounters only)', 'fig2_tsne_c_encounter_colregs'))

    # (d) Message - Yaw Rate
    paths.append(_save_scatter_continuous(
        msg_2d, yaw_s, 'Message space colored by yaw rate', 'Yaw Rate', 'fig2_tsne_d_msg_yawrate', cmap='coolwarm'))

    # (e) Message - Goal Distance
    paths.append(_save_scatter_continuous(
        msg_2d, goal_dist_s, 'Message space colored by goal distance', 'Goal Dist', 'fig2_tsne_e_msg_goaldist', cmap='viridis'))

    # (f) Message - Speed
    paths.append(_save_scatter_continuous(
        msg_2d, speed_s, 'Message space colored by speed', 'Speed', 'fig2_tsne_f_msg_speed', cmap='viridis'))

    # (g) Message - Goal Angle
    paths.append(_save_scatter_continuous(
        msg_2d, goal_angle_s, 'Message space colored by goal angle', 'Goal Angle', 'fig2_tsne_g_msg_goalangle', cmap='coolwarm'))

    # (h) Encounter - Yaw Rate
    paths.append(_save_scatter_continuous(
        msg_enc_2d, yaw_enc, 'Encounters colored by yaw rate', 'Yaw Rate', 'fig2_tsne_h_encounter_yawrate', cmap='coolwarm'))

    # 합본 PDF
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    plot_specs = [
        (axes[0,0], obs_2d, colregs_s, 'colregs', f'(a) Observation space ({OBS_DIM}D)'),
        (axes[0,1], msg_2d, colregs_s, 'colregs', f'(b) Latent message space ({MSG_DIM}D)'),
        (axes[0,2], msg_enc_2d, colregs_enc, 'colregs_enc', '(c) Message space (encounters only)'),
        (axes[0,3], msg_2d, yaw_s, 'cont_coolwarm', '(d) Colored by yaw rate'),
        (axes[1,0], msg_2d, goal_dist_s, 'cont_viridis', '(e) Colored by goal distance'),
        (axes[1,1], msg_2d, speed_s, 'cont_viridis', '(f) Colored by speed'),
        (axes[1,2], msg_2d, goal_angle_s, 'cont_coolwarm', '(g) Colored by goal angle'),
        (axes[1,3], msg_enc_2d, yaw_enc, 'cont_coolwarm', '(h) Encounters by yaw rate'),
    ]

    for ax, d2d, vals, mode, title in plot_specs:
        if mode == 'colregs':
            for i in [0, 4, 3, 2, 1]:
                mask = vals == i
                if mask.sum() > 0:
                    ax.scatter(d2d[mask, 0], d2d[mask, 1], c=COLREGS_COLORS[i],
                              label=COLREGS_LABELS[i], alpha=0.35 if i==0 else 0.8,
                              s=4 if i==0 else 12, edgecolors='none', zorder=2 if i==0 else 3)
            ax.legend(fontsize=6, markerscale=2)
        elif mode == 'colregs_enc':
            for i in [4, 3, 2, 1]:
                mask = vals == i
                if mask.sum() > 0:
                    ax.scatter(d2d[mask, 0], d2d[mask, 1], c=COLREGS_COLORS[i],
                              label=COLREGS_LABELS[i], alpha=0.8, s=14, edgecolors='none')
            ax.legend(fontsize=6, markerscale=2)
        else:
            cmap = 'viridis' if 'viridis' in mode else ('coolwarm' if 'coolwarm' in mode else 'RdYlBu_r')
            sc = ax.scatter(d2d[:, 0], d2d[:, 1], c=vals, cmap=cmap, alpha=0.6, s=6, edgecolors='none')
            plt.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('t-SNE dim 1', fontsize=8)
        ax.set_ylabel('t-SNE dim 2', fontsize=8)
        ax.grid(alpha=0.15)

    plt.tight_layout()
    pdf_path = os.path.join(SAVE_DIR, 'fig2_tsne_combined.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    for p in paths:
        print(f"  Saved: {p}")
    print(f"  Saved: {pdf_path}")


# ============================================================================
# Figure 2: Trajectory
# ============================================================================
def plot_fig3_trajectory():
    print("[Fig 3] Trajectory comparison")
    df_off = pd.read_csv(TRAJ_OFF)
    df_on = pd.read_csv(TRAJ_ON)

    target_agents = [0, 3]
    colors_off = ['#E74C3C', '#C0392B']
    colors_on = ['#3498DB', '#2980B9']

    for idx, aid in enumerate(target_agents):
        fig, ax = plt.subplots(figsize=(7, 6))

        ep_off = extract_first_episode(df_off, aid)
        ep_on = extract_first_episode(df_on, aid)

        ax.plot(ep_off.x.values, ep_off.z.values, '-', color=colors_off[idx],
                linewidth=2.0, alpha=0.9, label=f'Comm OFF (len={len(ep_off)})')
        ax.plot(ep_off.x.iloc[0], ep_off.z.iloc[0], 'o', color=colors_off[idx],
                markersize=12, zorder=5, markeredgecolor='black', markeredgewidth=1)
        ax.plot(ep_off.x.iloc[-1], ep_off.z.iloc[-1], 's', color=colors_off[idx],
                markersize=12, zorder=5, markeredgecolor='black', markeredgewidth=1)

        ax.plot(ep_on.x.values, ep_on.z.values, '--', color=colors_on[idx],
                linewidth=2.0, alpha=0.9, label=f'Comm ON (len={len(ep_on)})')
        ax.plot(ep_on.x.iloc[0], ep_on.z.iloc[0], 'o', color=colors_on[idx],
                markersize=12, zorder=5, markeredgecolor='black', markeredgewidth=1)
        ax.plot(ep_on.x.iloc[-1], ep_on.z.iloc[-1], 's', color=colors_on[idx],
                markersize=12, zorder=5, markeredgecolor='black', markeredgewidth=1)

        for c in [1, 2, 3, 4]:
            pts_off = ep_off[ep_off.colregs == c]
            pts_on = ep_on[ep_on.colregs == c]
            if len(pts_off) > 0:
                ax.scatter(pts_off.x.values, pts_off.z.values, c=COLREGS_COLORS[c],
                          s=30, alpha=0.5, zorder=4, edgecolors='none',
                          label=f'{COLREGS_LABELS[c]}')
            if len(pts_on) > 0:
                ax.scatter(pts_on.x.values, pts_on.z.values, c=COLREGS_COLORS[c],
                          s=30, alpha=0.5, zorder=4, marker='D', edgecolors='none')

        start = (ep_off.x.iloc[0], ep_off.z.iloc[0])
        end_off = (ep_off.x.iloc[-1], ep_off.z.iloc[-1])
        end_on = (ep_on.x.iloc[-1], ep_on.z.iloc[-1])

        ax.set_title(f'Agent {aid} — Start: ({start[0]:.0f}, {start[1]:.0f})', fontsize=11)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.annotate('START', xy=start, fontsize=8, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points')

        plt.tight_layout()
        path = os.path.join(SAVE_DIR, f'fig3_trajectory_agent{aid}.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

    # 합본 PDF
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Trajectory comparison with identical initial conditions', fontsize=13, fontweight='bold')
    for idx, aid in enumerate(target_agents):
        ax = axes[idx]
        ep_off = extract_first_episode(df_off, aid)
        ep_on = extract_first_episode(df_on, aid)
        ax.plot(ep_off.x.values, ep_off.z.values, '-', color=colors_off[idx], linewidth=2, alpha=0.9, label='Comm OFF')
        ax.plot(ep_on.x.values, ep_on.z.values, '--', color=colors_on[idx], linewidth=2, alpha=0.9, label='Comm ON')
        ax.plot(ep_off.x.iloc[0], ep_off.z.iloc[0], 'o', color='black', markersize=10, zorder=5)
        ax.plot(ep_off.x.iloc[-1], ep_off.z.iloc[-1], 's', color=colors_off[idx], markersize=10, zorder=5, markeredgecolor='black')
        ax.plot(ep_on.x.iloc[-1], ep_on.z.iloc[-1], 's', color=colors_on[idx], markersize=10, zorder=5, markeredgecolor='black')
        for c in [1, 2, 3, 4]:
            for ep, mk in [(ep_off, 'o'), (ep_on, 'D')]:
                pts = ep[ep.colregs == c]
                if len(pts) > 0:
                    ax.scatter(pts.x.values, pts.z.values, c=COLREGS_COLORS[c], s=20, alpha=0.4, marker=mk, edgecolors='none')
        ax.set_title(f'Agent {aid}', fontsize=11)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig3_trajectory_combined.pdf'), bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Figure 3b: Trajectory + Neighbors + Comm Range
# ============================================================================
def _draw_traj_nb(ax, ep_main, df_all, aid, mode, is_pdf=False):
    """fig3b 공용 그리기: 궤적 + 이웃 + 통신/레이더 범위"""
    COMM_R = 90   # 통신 반경 (m)
    RADAR_R = 60  # 레이더 범위 (m)
    SNAP = 100    # 스냅샷 간격 (step)

    is_on = (mode == 'ON')
    main_color = '#3498DB' if is_on else '#E74C3C'
    lw = 1.8 if is_pdf else 2.2
    ms_main = 10 if is_pdf else 12
    ms_nb = 6 if is_pdf else 8
    fs = 9 if is_pdf else 11

    # 메인 궤적
    ax.plot(ep_main.x.values, ep_main.z.values, '-', color=main_color,
            linewidth=lw, alpha=0.9, zorder=5,
            label=f'Comm {mode} (len={len(ep_main)})')
    # START / GOAL 마커
    ax.plot(ep_main.x.iloc[0], ep_main.z.iloc[0], 'o', color=main_color,
            markersize=ms_main, zorder=10, markeredgecolor='black', markeredgewidth=1.2)
    ax.plot(ep_main.x.iloc[-1], ep_main.z.iloc[-1], 's', color=main_color,
            markersize=ms_main, zorder=10, markeredgecolor='black', markeredgewidth=1.2)
    ax.annotate('START', xy=(ep_main.x.iloc[0], ep_main.z.iloc[0]),
                fontsize=8, fontweight='bold', xytext=(6, 6), textcoords='offset points')
    ax.annotate('GOAL', xy=(ep_main.x.iloc[-1], ep_main.z.iloc[-1]),
                fontsize=8, fontweight='bold', xytext=(6, -10), textcoords='offset points')

    # COLREGs 이벤트 (궤적 위에 색상 점)
    colregs_added = set()
    for c in [1, 2, 3, 4]:
        pts = ep_main[ep_main.colregs == c]
        if len(pts) > 0:
            ax.scatter(pts.x.values, pts.z.values, c=COLREGS_COLORS[c],
                      s=18 if is_pdf else 25, alpha=0.5, zorder=7,
                      edgecolors='none', label=COLREGS_LABELS[c])
            colregs_added.add(c)

    # 스냅샷 step 목록
    steps_in_ep = ep_main.step.values
    snap_steps = steps_in_ep[::SNAP]
    if len(snap_steps) > 0 and snap_steps[-1] != steps_in_ep[-1]:
        snap_steps = np.append(snap_steps, steps_in_ep[-1])

    nb_legend_added = False
    range_legend_added = False

    for t in snap_steps:
        row_m = ep_main[ep_main.step == t]
        if len(row_m) == 0:
            continue
        mx, mz = row_m.x.iloc[0], row_m.z.iloc[0]

        others_at_t = df_all[(df_all.step == t) & (df_all.agent_id != aid)]

        if is_on:
            # === Comm ON: 범위 안 이웃이 있을 때만 통신 원 그리기 ===
            has_neighbor_in_range = False
            for _, row in others_at_t.iterrows():
                ox, oz = row.x, row.z
                dist = np.sqrt((mx - ox)**2 + (mz - oz)**2)
                if dist <= COMM_R:
                    has_neighbor_in_range = True
                    # 이웃 위치
                    ax.plot(ox, oz, 'o', color='#3498DB', markersize=ms_nb,
                            alpha=0.8, markeredgecolor='black',
                            markeredgewidth=0.6, zorder=6)
                    # 이웃 통신 반경 (파란색)
                    ax.add_patch(plt.Circle((ox, oz), COMM_R, fill=False,
                                            color='#3498DB', alpha=0.3,
                                            linewidth=1.0, linestyle='--', zorder=2))
                    # 연결선
                    ax.plot([mx, ox], [mz, oz], '-', color='#2C3E50',
                            alpha=0.35, linewidth=0.8, zorder=3)

                    if not nb_legend_added:
                        ax.plot([], [], 'o', color='#3498DB', markersize=6,
                                markeredgecolor='black', label='Neighbor vessel')
                        nb_legend_added = True

            if has_neighbor_in_range:
                # 내 통신 반경 (빨간색)
                ax.add_patch(plt.Circle((mx, mz), COMM_R, fill=False,
                                        color='#E74C3C', alpha=0.4,
                                        linewidth=1.5, linestyle='-', zorder=2))
                if not range_legend_added:
                    ax.plot([], [], '-', color='#E74C3C', alpha=0.6, lw=1.5,
                            label=f'Own comm range ({COMM_R}m)')
                    ax.plot([], [], '--', color='#3498DB', alpha=0.5, lw=1.0,
                            label=f'Neighbor comm range ({COMM_R}m)')
                    range_legend_added = True
        else:
            # === Comm OFF: 레이더 범위만 표시 ===
            has_neighbor_nearby = False
            for _, row in others_at_t.iterrows():
                ox, oz = row.x, row.z
                dist = np.sqrt((mx - ox)**2 + (mz - oz)**2)
                if dist <= RADAR_R * 2:  # 레이더 근처면 표시
                    ax.plot(ox, oz, 'o', color='#95A5A6', markersize=ms_nb,
                            alpha=0.6, markeredgecolor='black',
                            markeredgewidth=0.5, zorder=6)
                    if not nb_legend_added:
                        ax.plot([], [], 'o', color='#95A5A6', markersize=6,
                                markeredgecolor='black', label='Neighbor vessel')
                        nb_legend_added = True
                    if dist <= RADAR_R:
                        has_neighbor_nearby = True

            if has_neighbor_nearby:
                ax.add_patch(plt.Circle((mx, mz), RADAR_R, fill=False,
                                        color='#E74C3C', alpha=0.25,
                                        linewidth=1.0, linestyle=':', zorder=2))
                if not range_legend_added:
                    ax.plot([], [], ':', color='#E74C3C', alpha=0.5, lw=1.0,
                            label=f'Radar range ({RADAR_R}m)')
                    range_legend_added = True

    # 축 범위: 궤적 bounding box + 여유
    all_x = ep_main.x.values
    all_z = ep_main.z.values
    pad = COMM_R + 20
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_z.min() - pad, all_z.max() + pad)

    ax.set_title(f'Agent {aid} — Comm {mode}', fontsize=fs)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7 if is_pdf else 8, loc='best')


def plot_fig3b_trajectory_with_neighbors():
    print("[Fig 3b] Trajectory + neighbors + comm range")
    df_off = pd.read_csv(TRAJ_OFF)
    df_on = pd.read_csv(TRAJ_ON)
    target_agents = [0, 3]

    # 개별 PNG
    for aid in target_agents:
        ep_off = extract_first_episode(df_off, aid)
        ep_on = extract_first_episode(df_on, aid)
        for mode, ep, df_all, suffix in [
            ('OFF', ep_off, df_off, 'commOFF'),
            ('ON', ep_on, df_on, 'commON'),
        ]:
            fig, ax = plt.subplots(figsize=(8, 7))
            _draw_traj_nb(ax, ep, df_all, aid, mode, is_pdf=False)
            plt.tight_layout()
            fname = f'fig3b_trajectory_agent{aid}_{suffix}_addNeighbor'
            fig.savefig(os.path.join(SAVE_DIR, f'{fname}.png'), dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {fname}.png")

    # 합본 PDF (2x2: row=agent, col=OFF/ON)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Trajectory with neighboring vessels and range visualization',
                 fontsize=13, fontweight='bold')
    for row, aid in enumerate(target_agents):
        ep_off = extract_first_episode(df_off, aid)
        ep_on = extract_first_episode(df_on, aid)
        for col, (mode, ep, df_all) in enumerate([
            ('OFF', ep_off, df_off),
            ('ON', ep_on, df_on),
        ]):
            _draw_traj_nb(axes[row, col], ep, df_all, aid, mode, is_pdf=True)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig3b_trajectory_addNeighbor_combined.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig3b_trajectory_addNeighbor_combined.pdf")


# ============================================================================
# Figure 4: Message dimensions
# ============================================================================
def plot_fig4_message_dims():
    print("[Fig 4] Message dimension analysis")
    data = load_latent_data()
    self_msg = data['self_msg']
    others_msg = data['others_msg']
    colregs = data['colregs']
    action_0 = data['action_0']
    action_1 = data['action_1']

    features = {'Goal Angle': data['goal_angle'], 'Speed': data['speed'],
                'Goal Dist': data['goal_dist'], 'Yaw Rate': data['yaw_rate']}
    all_features = {**features, 'Rudder': action_0, 'Thrust': action_1}
    feat_names = list(all_features.keys())

    corr_self = np.zeros((MSG_DIM, len(feat_names)))
    corr_others = np.zeros((MSG_DIM, len(feat_names)))
    for i in range(MSG_DIM):
        for j, fn in enumerate(feat_names):
            corr_self[i,j] = np.corrcoef(self_msg[:,i], all_features[fn])[0,1]
            corr_others[i,j] = np.corrcoef(others_msg[:,i], all_features[fn])[0,1]

    unique_colregs = sorted([c for c in range(5) if (colregs==c).sum() > 10])

    # (a) Self msg correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr_self, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_yticks(range(MSG_DIM)); ax.set_yticklabels([f'Dim {i}' for i in range(MSG_DIM)])
    ax.set_xticks(range(len(feat_names))); ax.set_xticklabels(feat_names, fontsize=9, rotation=30, ha='right')
    ax.set_title('Self message correlation', fontsize=11)
    for i in range(MSG_DIM):
        for j in range(len(feat_names)):
            v = corr_self[i,j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                   fontweight='bold' if abs(v)>0.3 else 'normal', color='white' if abs(v)>0.4 else 'black')
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig4_msg_a_self_correlation.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # (b) Others msg correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr_others, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_yticks(range(MSG_DIM)); ax.set_yticklabels([f'Dim {i}' for i in range(MSG_DIM)])
    ax.set_xticks(range(len(feat_names))); ax.set_xticklabels(feat_names, fontsize=9, rotation=30, ha='right')
    ax.set_title('Received message correlation', fontsize=11)
    for i in range(MSG_DIM):
        for j in range(len(feat_names)):
            v = corr_others[i,j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                   fontweight='bold' if abs(v)>0.3 else 'normal', color='white' if abs(v)>0.4 else 'black')
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig4_msg_b_others_correlation.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # (c) Per-dim boxplot by COLREGs
    for dim in range(MSG_DIM):
        fig, ax = plt.subplots(figsize=(5, 4))
        box_data = [self_msg[colregs==ci, dim] for ci in unique_colregs]
        box_labels = [COLREGS_LABELS[ci] for ci in unique_colregs]
        box_colors = [COLREGS_COLORS[ci] for ci in unique_colregs]
        bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xticklabels(box_labels, fontsize=8, rotation=30)
        ax.set_ylabel('Message value'); ax.set_title(f'Dim {dim} by COLREGs', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(SAVE_DIR, f'fig4_msg_c_dim{dim}_boxplot.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # (d) Per-dim scatter vs best feature
    for dim in range(MSG_DIM):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        best_idx = np.argmax(np.abs(corr_self[dim, :4]))
        fn = list(features.keys())[best_idx]
        fv = list(features.values())[best_idx]
        r = corr_self[dim, best_idx]
        for ci in [0, 4, 3, 2, 1]:
            mask = colregs == ci
            if mask.sum() > 0:
                ax.scatter(fv[mask], self_msg[mask, dim], c=COLREGS_COLORS[ci], s=3, alpha=0.3, edgecolors='none')
        z = np.polyfit(fv, self_msg[:, dim], 1)
        xline = np.linspace(fv.min(), fv.max(), 100)
        ax.plot(xline, np.poly1d(z)(xline), 'k--', linewidth=1.5, alpha=0.8)
        ax.set_xlabel(fn); ax.set_ylabel('Message value')
        ax.set_title(f'Dim {dim} vs {fn} (r={r:.3f})', fontsize=10)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        fig.savefig(os.path.join(SAVE_DIR, f'fig4_msg_d_dim{dim}_scatter.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # 합본 PDF
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle('Latent message dimension analysis', fontsize=14, fontweight='bold')
    gs = GridSpec(3, 6, figure=fig, hspace=0.45, wspace=0.4)
    ax_h1 = fig.add_subplot(gs[0, :3])
    im1 = ax_h1.imshow(corr_self, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax_h1.set_yticks(range(MSG_DIM)); ax_h1.set_yticklabels([f'Dim {i}' for i in range(MSG_DIM)])
    ax_h1.set_xticks(range(len(feat_names))); ax_h1.set_xticklabels(feat_names, fontsize=8, rotation=30, ha='right')
    ax_h1.set_title('(a) Self message'); plt.colorbar(im1, ax=ax_h1, shrink=0.7)
    for i in range(MSG_DIM):
        for j in range(len(feat_names)):
            ax_h1.text(j,i,f'{corr_self[i,j]:.2f}',ha='center',va='center',fontsize=8,
                      color='white' if abs(corr_self[i,j])>0.4 else 'black')
    ax_h2 = fig.add_subplot(gs[0, 3:])
    im2 = ax_h2.imshow(corr_others, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax_h2.set_yticks(range(MSG_DIM)); ax_h2.set_yticklabels([f'Dim {i}' for i in range(MSG_DIM)])
    ax_h2.set_xticks(range(len(feat_names))); ax_h2.set_xticklabels(feat_names, fontsize=8, rotation=30, ha='right')
    ax_h2.set_title('(b) Received message'); plt.colorbar(im2, ax=ax_h2, shrink=0.7)
    for i in range(MSG_DIM):
        for j in range(len(feat_names)):
            ax_h2.text(j,i,f'{corr_others[i,j]:.2f}',ha='center',va='center',fontsize=8,
                      color='white' if abs(corr_others[i,j])>0.4 else 'black')
    for dim in range(MSG_DIM):
        ax = fig.add_subplot(gs[1, dim])
        bd = [self_msg[colregs==ci, dim] for ci in unique_colregs]
        bp = ax.boxplot(bd, patch_artist=True, showfliers=False, medianprops=dict(color='black'))
        for p, col in zip(bp['boxes'], [COLREGS_COLORS[ci] for ci in unique_colregs]):
            p.set_facecolor(col); p.set_alpha(0.7)
        ax.set_xticklabels([COLREGS_LABELS[ci][:5] for ci in unique_colregs], fontsize=5, rotation=45)
        ax.set_title(f'Dim {dim}', fontsize=9); ax.grid(axis='y', alpha=0.3)
    for dim in range(MSG_DIM):
        ax = fig.add_subplot(gs[2, dim])
        bi = np.argmax(np.abs(corr_self[dim, :4]))
        fn = list(features.keys())[bi]; fv = list(features.values())[bi]
        for ci in [0,4,3,2,1]:
            m = colregs==ci
            if m.sum()>0: ax.scatter(fv[m], self_msg[m,dim], c=COLREGS_COLORS[ci], s=2, alpha=0.3, edgecolors='none')
        z = np.polyfit(fv, self_msg[:,dim], 1)
        ax.plot(np.linspace(fv.min(),fv.max(),50), np.poly1d(z)(np.linspace(fv.min(),fv.max(),50)), 'k--', lw=1.5)
        ax.set_title(f'Dim {dim} vs {fn} (r={corr_self[dim,bi]:.2f})', fontsize=8); ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig4_msg_combined.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: msg_a/b + dim0-5 boxplot/scatter + combined.pdf")


# ============================================================================
# Figure 5: Test Performance Comparison (from comparison test)
# ============================================================================
def plot_fig5_test_performance():
    print("[Fig 5] Test performance comparison")

    # comparison_10x2000_20260209_154752.txt 결과
    off_collisions = np.array([2, 2, 3, 1, 3, 1, 0, 1, 1, 0])
    off_successes  = np.array([8, 10, 9, 12, 8, 5, 2, 8, 6, 8])
    off_rewards    = np.array([4045.24, 4190.23, 4137.72, 4224.22, 4094.80,
                               4148.03, 4025.58, 4130.29, 4126.16, 4164.10])

    on_collisions  = np.array([1, 2, 0, 0, 2, 1, 0, 0, 2, 0])
    on_successes   = np.array([7, 9, 5, 5, 11, 14, 6, 9, 9, 9])
    on_rewards     = np.array([4330.64, 4267.53, 4371.89, 4376.03, 4284.36,
                               4389.31, 4276.69, 4529.15, 4223.86, 4346.44])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = ['#E74C3C', '#3498DB']
    x = np.arange(2)
    w = 0.5

    data_sets = [
        ('Collision rate', off_collisions, on_collisions, 'Collisions per run'),
        ('Success rate', off_successes, on_successes, 'Successes per run'),
        ('Average reward', off_rewards, on_rewards, 'Cumulative reward'),
    ]

    for ax, (title, off_d, on_d, ylabel) in zip(axes, data_sets):
        means = [off_d.mean(), on_d.mean()]
        bars = ax.bar(x, means, width=w, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=0.8)
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(means) * 0.02,
                    f'{means[i]:.1f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Comm OFF', 'Comm ON'], fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig5_test_performance.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, 'fig5_test_performance.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig5_test_performance.png / .pdf")


# ============================================================================
# Figure 6: t-SNE on Received (Others) Messages
# ============================================================================
def plot_fig6_tsne_others():
    print("[Fig 6] t-SNE analysis (received messages)")
    data = load_latent_data()
    others_msg = data['others_msg']
    colregs = data['colregs']
    yaw_rate = data['yaw_rate']
    speed = data['speed']
    goal_angle = data['goal_angle']
    goal_dist = data['goal_dist']

    # others_msg가 전부 0인 샘플 제외 (통신 없는 timestep)
    has_msg = np.any(others_msg != 0, axis=1)
    print(f"  Non-zero received messages: {has_msg.sum()}/{len(has_msg)}")

    if has_msg.sum() < 100:
        print("  WARNING: Too few non-zero received messages, skipping Fig 6")
        return

    # 균형 샘플링 (non-zero 중에서)
    rng = np.random.RandomState(42)
    max_per_class = 500
    balanced_idx = []
    for c in range(5):
        mask = np.where((colregs == c) & has_msg)[0]
        if len(mask) > 0:
            n = min(max_per_class, len(mask))
            balanced_idx.extend(rng.choice(mask, n, replace=False))
    balanced_idx = np.array(balanced_idx)
    rng.shuffle(balanced_idx)

    omsg_s = others_msg[balanced_idx]
    colregs_s = colregs[balanced_idx]
    yaw_s = yaw_rate[balanced_idx]
    speed_s = speed[balanced_idx]
    goal_angle_s = goal_angle[balanced_idx]
    goal_dist_s = goal_dist[balanced_idx]

    # t-SNE on received messages
    print(f"  t-SNE on Received Messages ({len(balanced_idx)} samples)...")
    omsg_2d = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(omsg_s)

    # encounter only
    encounter_mask = colregs_s > 0
    omsg_enc = omsg_s[encounter_mask]
    colregs_enc = colregs_s[encounter_mask]
    yaw_enc = yaw_s[encounter_mask]
    print(f"  t-SNE on Encounters ({encounter_mask.sum()})...")
    omsg_enc_2d = TSNE(n_components=2, perplexity=min(30, encounter_mask.sum()-1),
                       random_state=42, max_iter=1000).fit_transform(omsg_enc)

    paths = []

    # (a) Received msg - COLREGs
    paths.append(_save_scatter_colregs(
        omsg_2d, colregs_s, f'Received message space ({MSG_DIM}D)', 'fig6_tsne_others_a_msg_colregs'))

    # (b) Received msg - encounters only
    paths.append(_save_scatter_colregs(
        omsg_enc_2d, colregs_enc, 'Received messages (encounters only)', 'fig6_tsne_others_b_encounter_colregs'))

    # (c) Received msg - Yaw Rate
    paths.append(_save_scatter_continuous(
        omsg_2d, yaw_s, 'Received messages colored by yaw rate', 'Yaw Rate', 'fig6_tsne_others_c_msg_yawrate', cmap='coolwarm'))

    # (d) Received msg - Goal Distance
    paths.append(_save_scatter_continuous(
        omsg_2d, goal_dist_s, 'Received messages colored by goal distance', 'Goal Dist', 'fig6_tsne_others_d_msg_goaldist', cmap='viridis'))

    # (e) Received msg - Speed
    paths.append(_save_scatter_continuous(
        omsg_2d, speed_s, 'Received messages colored by speed', 'Speed', 'fig6_tsne_others_e_msg_speed', cmap='viridis'))

    # (f) Received msg - Goal Angle
    paths.append(_save_scatter_continuous(
        omsg_2d, goal_angle_s, 'Received messages colored by goal angle', 'Goal Angle', 'fig6_tsne_others_f_msg_goalangle', cmap='coolwarm'))

    # (g) Encounter - Yaw Rate
    paths.append(_save_scatter_continuous(
        omsg_enc_2d, yaw_enc, 'Received encounters colored by yaw rate', 'Yaw Rate', 'fig6_tsne_others_g_encounter_yawrate', cmap='coolwarm'))

    # 합본 PDF (2x4)
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig.suptitle('t-SNE analysis of received (others) messages', fontsize=14, fontweight='bold')

    plot_specs = [
        (axes[0,0], omsg_2d, colregs_s, 'colregs', f'(a) Received msg space ({MSG_DIM}D)'),
        (axes[0,1], omsg_enc_2d, colregs_enc, 'colregs_enc', '(b) Encounters only'),
        (axes[0,2], omsg_2d, yaw_s, 'cont_coolwarm', '(c) Colored by yaw rate'),
        (axes[0,3], omsg_2d, goal_dist_s, 'cont_viridis', '(d) Colored by goal distance'),
        (axes[1,0], omsg_2d, speed_s, 'cont_viridis', '(e) Colored by speed'),
        (axes[1,1], omsg_2d, goal_angle_s, 'cont_coolwarm', '(f) Colored by goal angle'),
        (axes[1,2], omsg_enc_2d, yaw_enc, 'cont_coolwarm', '(g) Encounters by yaw rate'),
    ]

    for ax, d2d, vals, mode, title in plot_specs:
        if mode == 'colregs':
            for i in [0, 4, 3, 2, 1]:
                mask = vals == i
                if mask.sum() > 0:
                    ax.scatter(d2d[mask, 0], d2d[mask, 1], c=COLREGS_COLORS[i],
                              label=COLREGS_LABELS[i], alpha=0.35 if i==0 else 0.8,
                              s=4 if i==0 else 12, edgecolors='none', zorder=2 if i==0 else 3)
            ax.legend(fontsize=6, markerscale=2)
        elif mode == 'colregs_enc':
            for i in [4, 3, 2, 1]:
                mask = vals == i
                if mask.sum() > 0:
                    ax.scatter(d2d[mask, 0], d2d[mask, 1], c=COLREGS_COLORS[i],
                              label=COLREGS_LABELS[i], alpha=0.8, s=14, edgecolors='none')
            ax.legend(fontsize=6, markerscale=2)
        else:
            cmap = 'viridis' if 'viridis' in mode else ('coolwarm' if 'coolwarm' in mode else 'RdYlBu_r')
            sc = ax.scatter(d2d[:, 0], d2d[:, 1], c=vals, cmap=cmap, alpha=0.6, s=6, edgecolors='none')
            plt.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('t-SNE dim 1', fontsize=8)
        ax.set_ylabel('t-SNE dim 2', fontsize=8)
        ax.grid(alpha=0.15)

    # 마지막 빈 칸 숨기기
    axes[1, 3].set_visible(False)

    plt.tight_layout()
    pdf_path = os.path.join(SAVE_DIR, 'fig6_tsne_others_combined.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    for p in paths:
        print(f"  Saved: {p}")
    print(f"  Saved: {pdf_path}")


# ============================================================================
# Figure 7: Message Cross-Analysis (self_msg vs others_msg vs receiver obs)
# ============================================================================
def plot_fig7_message_cross_analysis():
    print("[Fig 7] Message cross-analysis")
    data = load_latent_data()
    self_msg = data['self_msg']
    others_msg = data['others_msg']
    colregs = data['colregs']

    radar = data['obs'][:, :360]  # 360 radar rays
    radar_min = radar.min(axis=1)
    radar_mean = radar.mean(axis=1)
    radar_std = radar.std(axis=1)
    radar_close_count = (radar < -0.2).sum(axis=1).astype(float)

    features_extended = {
        'Goal Dist': data['goal_dist'],
        'Goal Angle': data['goal_angle'],
        'Speed': data['speed'],
        'Yaw Rate': data['yaw_rate'],
        'Rudder': data['action_0'],
        'Thrust': data['action_1'],
        'Radar Min': radar_min,
        'Radar Mean': radar_mean,
        'Radar Std': radar_std,
        'Close Objects': radar_close_count,
    }
    feat_names = list(features_extended.keys())

    # (a) self_msg vs own features
    corr_self = np.zeros((MSG_DIM, len(feat_names)))
    for i in range(MSG_DIM):
        for j, fn in enumerate(feat_names):
            corr_self[i, j] = np.corrcoef(self_msg[:, i], features_extended[fn])[0, 1]

    # (b) others_msg vs receiver's features
    corr_others = np.zeros((MSG_DIM, len(feat_names)))
    for i in range(MSG_DIM):
        for j, fn in enumerate(feat_names):
            corr_others[i, j] = np.corrcoef(others_msg[:, i], features_extended[fn])[0, 1]

    # (c) cross-correlation: self_msg dim × others_msg dim
    cross_corr = np.zeros((MSG_DIM, MSG_DIM))
    for i in range(MSG_DIM):
        for j in range(MSG_DIM):
            cross_corr[i, j] = np.corrcoef(self_msg[:, i], others_msg[:, j])[0, 1]

    # (d) others_msg magnitude by COLREGs
    omsg_mag = np.linalg.norm(others_msg, axis=1)
    smsg_mag = np.linalg.norm(self_msg, axis=1)
    unique_colregs = sorted([c for c in range(5) if (colregs == c).sum() > 10])

    # ---- Individual PNGs ----

    # (a) self_msg correlation heatmap (extended features)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(corr_self, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_yticks(range(MSG_DIM)); ax.set_yticklabels([f'Dim {i}' for i in range(MSG_DIM)])
    ax.set_xticks(range(len(feat_names))); ax.set_xticklabels(feat_names, fontsize=8, rotation=40, ha='right')
    ax.set_title('Self message vs own features', fontsize=11)
    for i in range(MSG_DIM):
        for j in range(len(feat_names)):
            v = corr_self[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                    fontweight='bold' if abs(v) > 0.3 else 'normal',
                    color='white' if abs(v) > 0.4 else 'black')
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig7_cross_a_self_corr.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # (b) others_msg correlation heatmap (extended features)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(corr_others, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_yticks(range(MSG_DIM)); ax.set_yticklabels([f'Dim {i}' for i in range(MSG_DIM)])
    ax.set_xticks(range(len(feat_names))); ax.set_xticklabels(feat_names, fontsize=8, rotation=40, ha='right')
    ax.set_title('Received message vs receiver\'s own features', fontsize=11)
    for i in range(MSG_DIM):
        for j in range(len(feat_names)):
            v = corr_others[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                    fontweight='bold' if abs(v) > 0.3 else 'normal',
                    color='white' if abs(v) > 0.4 else 'black')
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig7_cross_b_others_corr.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # (c) cross-correlation matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cross_corr, cmap='RdBu_r', aspect='equal', vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(MSG_DIM)); ax.set_xticklabels([f'Others {i}' for i in range(MSG_DIM)], fontsize=9)
    ax.set_yticks(range(MSG_DIM)); ax.set_yticklabels([f'Self {i}' for i in range(MSG_DIM)], fontsize=9)
    ax.set_title('Cross-correlation: self msg vs received msg', fontsize=11)
    for i in range(MSG_DIM):
        for j in range(MSG_DIM):
            v = cross_corr[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=9,
                    fontweight='bold' if abs(v) > 0.1 else 'normal',
                    color='white' if abs(v) > 0.2 else 'black')
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig7_cross_c_self_vs_others.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # (d) message magnitude by COLREGs
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(unique_colregs))
    w = 0.35
    self_means = [smsg_mag[colregs == c].mean() for c in unique_colregs]
    self_stds = [smsg_mag[colregs == c].std() for c in unique_colregs]
    others_means = [omsg_mag[colregs == c].mean() for c in unique_colregs]
    others_stds = [omsg_mag[colregs == c].std() for c in unique_colregs]
    bars1 = ax.bar(x - w/2, self_means, w, yerr=self_stds, label='Self msg',
                   color='#E74C3C', alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + w/2, others_means, w, yerr=others_stds, label='Received msg',
                   color='#3498DB', alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([COLREGS_LABELS[c] for c in unique_colregs], fontsize=9)
    ax.set_ylabel('L2 magnitude')
    ax.set_title('Message magnitude by COLREGs situation', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig7_cross_d_magnitude_colregs.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # (e) radar proximity (min distance) vs others_msg dims scatter
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Received message dimensions vs radar closest distance', fontsize=12)
    for dim, ax in enumerate(axes.flat):
        for ci in [0, 4, 3, 2, 1]:
            mask = colregs == ci
            if mask.sum() > 0:
                ax.scatter(radar_min[mask], others_msg[mask, dim],
                          c=COLREGS_COLORS[ci], s=3, alpha=0.2, edgecolors='none')
        r = np.corrcoef(radar_min, others_msg[:, dim])[0, 1]
        z = np.polyfit(radar_min, others_msg[:, dim], 1)
        xline = np.linspace(radar_min.min(), radar_min.max(), 50)
        ax.plot(xline, np.poly1d(z)(xline), 'k--', linewidth=1.5, alpha=0.8)
        ax.set_title(f'Dim {dim} (r={r:.3f})', fontsize=10)
        ax.set_xlabel('Radar min distance')
        ax.set_ylabel('Received msg value')
        ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig7_cross_e_radar_vs_others.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ---- Combined PDF ----
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle('Message cross-analysis: self message vs received message vs receiver observation',
                 fontsize=14, fontweight='bold')
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.4)

    # Row 0: (a) self corr, (b) others corr
    ax_a = fig.add_subplot(gs[0, :2])
    im = ax_a.imshow(corr_self, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax_a.set_yticks(range(MSG_DIM)); ax_a.set_yticklabels([f'D{i}' for i in range(MSG_DIM)], fontsize=8)
    ax_a.set_xticks(range(len(feat_names))); ax_a.set_xticklabels(feat_names, fontsize=7, rotation=40, ha='right')
    ax_a.set_title('(a) Self msg vs own features', fontsize=10)
    for i in range(MSG_DIM):
        for j in range(len(feat_names)):
            v = corr_self[i, j]
            ax_a.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                      color='white' if abs(v) > 0.4 else 'black')
    plt.colorbar(im, ax=ax_a, shrink=0.7)

    ax_b = fig.add_subplot(gs[0, 2:])
    im = ax_b.imshow(corr_others, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax_b.set_yticks(range(MSG_DIM)); ax_b.set_yticklabels([f'D{i}' for i in range(MSG_DIM)], fontsize=8)
    ax_b.set_xticks(range(len(feat_names))); ax_b.set_xticklabels(feat_names, fontsize=7, rotation=40, ha='right')
    ax_b.set_title('(b) Received msg vs receiver\'s features', fontsize=10)
    for i in range(MSG_DIM):
        for j in range(len(feat_names)):
            v = corr_others[i, j]
            ax_b.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                      color='white' if abs(v) > 0.4 else 'black')
    plt.colorbar(im, ax=ax_b, shrink=0.7)

    # Row 1: (c) cross-corr, (d) magnitude
    ax_c = fig.add_subplot(gs[1, :2])
    im = ax_c.imshow(cross_corr, cmap='RdBu_r', aspect='equal', vmin=-0.3, vmax=0.3)
    ax_c.set_xticks(range(MSG_DIM)); ax_c.set_xticklabels([f'Others {i}' for i in range(MSG_DIM)], fontsize=8)
    ax_c.set_yticks(range(MSG_DIM)); ax_c.set_yticklabels([f'Self {i}' for i in range(MSG_DIM)], fontsize=8)
    ax_c.set_title('(c) Self msg vs received msg', fontsize=10)
    for i in range(MSG_DIM):
        for j in range(MSG_DIM):
            v = cross_corr[i, j]
            ax_c.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8,
                      color='white' if abs(v) > 0.2 else 'black')
    plt.colorbar(im, ax=ax_c, shrink=0.7)

    ax_d = fig.add_subplot(gs[1, 2:])
    x_bar = np.arange(len(unique_colregs))
    bars1 = ax_d.bar(x_bar - w/2, self_means, w, yerr=self_stds, label='Self msg',
                     color='#E74C3C', alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)
    bars2 = ax_d.bar(x_bar + w/2, others_means, w, yerr=others_stds, label='Received msg',
                     color='#3498DB', alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)
    ax_d.set_xticks(x_bar)
    ax_d.set_xticklabels([COLREGS_LABELS[c] for c in unique_colregs], fontsize=8)
    ax_d.set_ylabel('L2 magnitude'); ax_d.set_title('(d) Msg magnitude by COLREGs', fontsize=10)
    ax_d.legend(fontsize=8); ax_d.grid(axis='y', alpha=0.3)

    # Row 2: (e) radar vs others_msg dim0~5
    for dim in range(MSG_DIM):
        ax = fig.add_subplot(gs[2, dim]) if dim < 4 else None
        if ax is None:
            break
        for ci in [0, 4, 3, 2, 1]:
            mask = colregs == ci
            if mask.sum() > 0:
                ax.scatter(radar_min[mask], others_msg[mask, dim],
                          c=COLREGS_COLORS[ci], s=2, alpha=0.2, edgecolors='none')
        r = np.corrcoef(radar_min, others_msg[:, dim])[0, 1]
        z = np.polyfit(radar_min, others_msg[:, dim], 1)
        xline = np.linspace(radar_min.min(), radar_min.max(), 50)
        ax.plot(xline, np.poly1d(z)(xline), 'k--', lw=1.5)
        ax.set_title(f'(e{dim}) Dim {dim} vs radar min (r={r:.3f})', fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig7_cross_combined.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig7_cross_a~e + combined.pdf")


# ============================================================================
# Figure 8: COLREGs Fuzzy Compliance Analysis
# ============================================================================
# Comparison test CSV 경로
COMPARE_OFF_CSV = os.path.join(PROJECT_ROOT, "latent_data", "compare_commOFF_10x2000_20260209_154752.csv")
COMPARE_ON_CSV = os.path.join(PROJECT_ROOT, "latent_data", "compare_commON_10x2000_20260209_154752.csv")


def _load_colregs_data(csv_path):
    """비교 테스트 CSV에서 COLREGs 관련 데이터 추출 (레이더 포함)"""
    df = pd.read_csv(csv_path)
    colregs = np.argmax(df[[f'obs_{i}' for i in range(364, 369)]].values, axis=1)
    yaw_rate = df['obs_363'].values
    speed = df['obs_362'].values
    goal_dist = df['obs_360'].values
    # 레이더 360 rays: obs_0 ~ obs_359 (값 범위 [-0.5, 0.5], 높을수록 멀리)
    radar = df[[f'obs_{i}' for i in range(360)]].values
    radar_min = radar.min(axis=1)  # 가장 가까운 장애물
    return colregs, yaw_rate, speed, goal_dist, radar_min


def _compute_composite_compliance(colregs, rmin, collision_rate, overall_clearance, danger_thresh=-0.3):
    """
    복합 COLREGs 준수율 계산.

    3요소 결합:
      (1) 상황별 안전율: 해당 COLREGs 상황에서 위험 구간(radar_min < threshold) 비율
      (2) 충돌 회피율: 전체 테스트 결과 기반 (collisions / events)
      (3) 전체 레이더 안전 여유: 전 구간 평균 radar_min

    compliance = 0.2 * per_sit_safety + 0.4 * collision_avoidance + 0.4 * overall_clearance
    """
    situations = [1, 2, 3, 4]
    scores = {}
    collision_avoidance = 1.0 - collision_rate  # 충돌 회피율

    for sit in situations:
        mask = colregs == sit
        if mask.sum() > 0:
            # 상황별 안전율: 위험 timestep 비율의 역
            danger_frac = (rmin[mask] < danger_thresh).sum() / mask.sum()
            per_sit_safety = 1.0 - danger_frac
        else:
            per_sit_safety = 0.5

        scores[sit] = (0.2 * per_sit_safety
                        + 0.4 * collision_avoidance
                        + 0.4 * overall_clearance)
    return scores


def plot_fig8_colregs_compliance():
    print("[Fig 8] COLREGs fuzzy compliance analysis")

    colregs_off, yaw_off, speed_off, gdist_off, rmin_off = _load_colregs_data(COMPARE_OFF_CSV)
    colregs_on, yaw_on, speed_on, gdist_on, rmin_on = _load_colregs_data(COMPARE_ON_CSV)

    # --- 전체 테스트 결과 (comparison test) ---
    # OFF: 14 collisions, 76 successes / ON: 8 collisions, 84 successes
    total_off = 14 + 76  # collisions + successes
    total_on = 8 + 84
    collision_rate_off = 14 / total_off   # 0.156
    collision_rate_on = 8 / total_on      # 0.087

    # 전체 레이더 안전 여유 ([-0.5,0.5] → [0,1])
    clearance_off = np.clip(rmin_off.mean() + 0.5, 0, 1)  # ~0.535
    clearance_on = np.clip(rmin_on.mean() + 0.5, 0, 1)    # ~0.553

    situations = [1, 2, 3, 4]
    sit_names = ['Head-on', 'Stand-on', 'Give-way', 'Overtaking']
    sit_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

    # 복합 준수율 계산
    scores_off = _compute_composite_compliance(colregs_off, rmin_off, collision_rate_off, clearance_off)
    scores_on = _compute_composite_compliance(colregs_on, rmin_on, collision_rate_on, clearance_on)

    x = np.arange(len(situations))
    w = 0.35
    means_off = [scores_off[s] for s in situations]
    means_on = [scores_on[s] for s in situations]

    # --- (a) Compliance bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, means_off, w, color='#E74C3C', alpha=0.85,
                   edgecolor='black', linewidth=0.8, label='Comm OFF')
    bars2 = ax.bar(x + w/2, means_on, w, color='#3498DB', alpha=0.85,
                   edgecolor='black', linewidth=0.8, label='Comm ON')
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sit_names, fontsize=11)
    ax.set_ylabel('Composite compliance score', fontsize=11)
    ax.set_title('COLREGs compliance by situation', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig8_colregs_a_compliance_bar.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- (b) Yaw rate distribution per COLREGs (violin plot) ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle('Yaw rate distribution by COLREGs situation', fontsize=12, fontweight='bold')
    for idx, (sit, name) in enumerate(zip(situations, sit_names)):
        ax = axes[idx]
        mask_off = colregs_off == sit; mask_on = colregs_on == sit
        yaw_data, labels, colors_v = [], [], []
        if mask_off.sum() > 0:
            yaw_data.append(yaw_off[mask_off])
            labels.append(f'OFF\n(n={mask_off.sum()})'); colors_v.append('#E74C3C')
        if mask_on.sum() > 0:
            yaw_data.append(yaw_on[mask_on])
            labels.append(f'ON\n(n={mask_on.sum()})'); colors_v.append('#3498DB')
        if len(yaw_data) > 0:
            vp = ax.violinplot(yaw_data, showmeans=True, showmedians=True)
            for i, body in enumerate(vp['bodies']):
                body.set_facecolor(colors_v[i]); body.set_alpha(0.7)
            if 'cmeans' in vp: vp['cmeans'].set_color('black')
            if 'cmedians' in vp:
                vp['cmedians'].set_color('gray'); vp['cmedians'].set_linestyle('--')
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.set_title(name, fontsize=11, color=sit_colors[idx], fontweight='bold')
        ax.set_ylabel('Yaw rate' if idx == 0 else '')
        ax.grid(axis='y', alpha=0.3)
        for i, (mask, lp) in enumerate([(mask_off, 'OFF'), (mask_on, 'ON')]):
            if mask.sum() > 0:
                yv = yaw_off[mask] if lp == 'OFF' else yaw_on[mask]
                rpct = (yv > 0).sum() / len(yv) * 100
                ax.text(i + 1, ax.get_ylim()[1] * 0.85, f'R:{rpct:.0f}%',
                        ha='center', fontsize=9, fontweight='bold',
                        color='#E74C3C' if lp == 'OFF' else '#3498DB')
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig8_colregs_b_yaw_violin.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- (c) Radar chart (spider/polar) ---
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(situations), endpoint=False).tolist()
    angles += angles[:1]
    vals_off = means_off + means_off[:1]
    vals_on = means_on + means_on[:1]
    ax.plot(angles, vals_off, 'o-', color='#E74C3C', linewidth=2, label='Comm OFF', markersize=8)
    ax.fill(angles, vals_off, color='#E74C3C', alpha=0.15)
    ax.plot(angles, vals_on, 's-', color='#3498DB', linewidth=2, label='Comm ON', markersize=8)
    ax.fill(angles, vals_on, color='#3498DB', alpha=0.15)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(sit_names, fontsize=11)
    ax.set_ylim(0, 1.0); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('COLREGs composite compliance', fontsize=12, pad=20)
    ax.legend(fontsize=10, loc='lower right', bbox_to_anchor=(1.15, -0.05))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig8_colregs_c_radar.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- (d) Component breakdown: stacked bar ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    component_names = ['Situation safety', 'Collision avoidance', 'Radar clearance']
    # 각 component 값 계산
    comp_off = np.zeros((len(situations), 3))
    comp_on = np.zeros((len(situations), 3))
    ca_off = 1.0 - collision_rate_off
    ca_on = 1.0 - collision_rate_on
    for i, sit in enumerate(situations):
        mask_off = colregs_off == sit; mask_on = colregs_on == sit
        ps_off = 1.0 - ((rmin_off[mask_off] < -0.3).sum() / max(mask_off.sum(), 1)) if mask_off.sum() > 0 else 0.5
        ps_on = 1.0 - ((rmin_on[mask_on] < -0.3).sum() / max(mask_on.sum(), 1)) if mask_on.sum() > 0 else 0.5
        comp_off[i] = [0.2 * ps_off, 0.4 * ca_off, 0.4 * clearance_off]
        comp_on[i] = [0.2 * ps_on, 0.4 * ca_on, 0.4 * clearance_on]

    bar_w = 0.35
    x_pos = np.arange(len(situations))
    colors_comp = ['#F39C12', '#E74C3C', '#2ECC71']

    for mode, comp, offset, hatch in [('OFF', comp_off, -bar_w/2, ''), ('ON', comp_on, bar_w/2, '//')]:
        bottom = np.zeros(len(situations))
        for j in range(3):
            label = f'{component_names[j]} ({mode})' if j == 0 else None
            bars = ax.bar(x_pos + offset, comp[:, j], bar_w, bottom=bottom,
                         color=colors_comp[j], alpha=0.7 if mode == 'OFF' else 0.9,
                         edgecolor='black', linewidth=0.5, hatch=hatch)
            bottom += comp[:, j]
        # 총합 텍스트
        for i in range(len(situations)):
            total = comp[i].sum()
            ax.text(x_pos[i] + offset, total + 0.01, f'{total:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 범례
    from matplotlib.patches import Patch
    legend_items = []
    for j in range(3):
        legend_items.append(Patch(facecolor=colors_comp[j], alpha=0.8, label=component_names[j]))
    legend_items.append(Patch(facecolor='white', edgecolor='black', label='OFF (solid)'))
    legend_items.append(Patch(facecolor='white', edgecolor='black', hatch='//', label='ON (hatched)'))
    ax.legend(handles=legend_items, fontsize=9, loc='upper right')

    ax.set_xticks(x_pos); ax.set_xticklabels(sit_names, fontsize=11)
    ax.set_ylabel('Composite compliance score', fontsize=11)
    ax.set_title('Compliance score breakdown by component', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig8_colregs_d_breakdown.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- Combined PDF (2x2) ---
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('COLREGs composite compliance analysis — Comm OFF vs Comm ON',
                 fontsize=14, fontweight='bold')
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # (a) Bar
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.bar(x - w/2, means_off, w, color='#E74C3C', alpha=0.85,
             edgecolor='black', linewidth=0.8, label='Comm OFF')
    ax_a.bar(x + w/2, means_on, w, color='#3498DB', alpha=0.85,
             edgecolor='black', linewidth=0.8, label='Comm ON')
    for mode_means in [means_off, means_on]:
        off_x = -w/2 if mode_means is means_off else w/2
        for i, m in enumerate(mode_means):
            ax_a.text(i + off_x, m + 0.01, f'{m:.2f}', ha='center', va='bottom',
                      fontsize=9, fontweight='bold')
    ax_a.set_xticks(x); ax_a.set_xticklabels(sit_names, fontsize=10)
    ax_a.set_ylabel('Compliance score'); ax_a.set_title('(a) Composite compliance', fontsize=11)
    ax_a.set_ylim(0, 1.05); ax_a.legend(fontsize=9); ax_a.grid(axis='y', alpha=0.3)

    # (b) Radar
    ax_b = fig.add_subplot(gs[0, 1], polar=True)
    ax_b.plot(angles, vals_off, 'o-', color='#E74C3C', linewidth=2, label='Comm OFF', markersize=7)
    ax_b.fill(angles, vals_off, color='#E74C3C', alpha=0.15)
    ax_b.plot(angles, vals_on, 's-', color='#3498DB', linewidth=2, label='Comm ON', markersize=7)
    ax_b.fill(angles, vals_on, color='#3498DB', alpha=0.15)
    ax_b.set_xticks(angles[:-1]); ax_b.set_xticklabels(sit_names, fontsize=10)
    ax_b.set_ylim(0, 1.0); ax_b.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.set_title('(b) Compliance radar', fontsize=11, pad=15)
    ax_b.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1.15, -0.05))

    # (c) Violin
    ax_v = fig.add_subplot(gs[1, 0])
    all_yaw, all_labels, all_colors, pos_list = [], [], [], []
    p = 1
    for si, sn in zip(situations, sit_names):
        m_off = colregs_off == si; m_on = colregs_on == si
        if m_off.sum() > 0:
            all_yaw.append(yaw_off[m_off]); all_labels.append(f'{sn[:6]}\nOFF')
            all_colors.append('#E74C3C'); pos_list.append(p); p += 1
        if m_on.sum() > 0:
            all_yaw.append(yaw_on[m_on]); all_labels.append(f'{sn[:6]}\nON')
            all_colors.append('#3498DB'); pos_list.append(p); p += 1
        p += 0.5
    vp = ax_v.violinplot(all_yaw, positions=pos_list, showmeans=True, showmedians=True)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(all_colors[i]); body.set_alpha(0.7)
    if 'cmeans' in vp: vp['cmeans'].set_color('black')
    if 'cmedians' in vp: vp['cmedians'].set_color('gray'); vp['cmedians'].set_linestyle('--')
    ax_v.set_xticks(pos_list); ax_v.set_xticklabels(all_labels, fontsize=7)
    ax_v.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax_v.set_ylabel('Yaw rate'); ax_v.set_title('(c) Yaw rate distribution', fontsize=11)
    ax_v.grid(axis='y', alpha=0.3)

    # (d) Stacked breakdown
    ax_d = fig.add_subplot(gs[1, 1])
    for mode, comp, offset, hatch in [('OFF', comp_off, -bar_w/2, ''), ('ON', comp_on, bar_w/2, '//')]:
        bottom = np.zeros(len(situations))
        for j in range(3):
            ax_d.bar(x_pos + offset, comp[:, j], bar_w, bottom=bottom,
                     color=colors_comp[j], alpha=0.7 if mode == 'OFF' else 0.9,
                     edgecolor='black', linewidth=0.5, hatch=hatch)
            bottom += comp[:, j]
        for i in range(len(situations)):
            ax_d.text(x_pos[i] + offset, comp[i].sum() + 0.01, f'{comp[i].sum():.2f}',
                      ha='center', va='bottom', fontsize=8, fontweight='bold')
    legend_items_d = [Patch(facecolor=colors_comp[j], alpha=0.8, label=component_names[j]) for j in range(3)]
    legend_items_d += [Patch(facecolor='white', edgecolor='black', label='OFF'),
                       Patch(facecolor='white', edgecolor='black', hatch='//', label='ON')]
    ax_d.legend(handles=legend_items_d, fontsize=7, loc='upper right')
    ax_d.set_xticks(x_pos); ax_d.set_xticklabels(sit_names, fontsize=9)
    ax_d.set_ylabel('Score'); ax_d.set_title('(d) Component breakdown', fontsize=11)
    ax_d.set_ylim(0, 1.05); ax_d.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'fig8_colregs_combined.pdf'), bbox_inches='tight')
    plt.close(fig)

    # 결과 출력
    print("  Composite compliance scores:")
    print(f"    Global: collision_avoidance OFF={1-collision_rate_off:.3f} ON={1-collision_rate_on:.3f}")
    print(f"    Global: radar_clearance    OFF={clearance_off:.3f} ON={clearance_on:.3f}")
    for sit, name in zip(situations, sit_names):
        print(f"    {name:12s}: OFF={scores_off[sit]:.3f}, ON={scores_on[sit]:.3f} (delta={scores_on[sit]-scores_off[sit]:+.3f})")
    print("  Saved: fig8_colregs_a~d + combined.pdf")


if __name__ == '__main__':
    print("=" * 60)
    print(f"Output: {SAVE_DIR}")
    print("=" * 60)
    plot_fig1_reward_curve()
    plot_fig2_tsne()
    plot_fig3_trajectory()
    plot_fig3b_trajectory_with_neighbors()
    plot_fig4_message_dims()
    plot_fig5_test_performance()
    plot_fig6_tsne_others()
    plot_fig7_message_cross_analysis()
    plot_fig8_colregs_compliance()
    print("\nDone!")
