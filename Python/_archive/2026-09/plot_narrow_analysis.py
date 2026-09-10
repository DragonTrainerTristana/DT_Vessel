"""
Narrow Channel Vessel Test - Analysis Graphs
좁은 해협 환경 실험 분석 그래프 생성

생성 그래프:
  Fig 5 (narrow): Test performance bar chart (collision, success, reward)
  Fig 8 (narrow): COLREGs composite compliance analysis (bar + violin + radar + breakdown)
  Fig env_comparison: Open vs Narrow environment comparison
  Fig 2 (narrow): t-SNE of self messages (COLREGs, yaw, goal_dist, speed)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE

# ============================================================================
# 경로 설정
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

SAVE_DIR = os.path.join(PROJECT_ROOT, "figures", "분석 그래프 좁은 해협")
os.makedirs(SAVE_DIR, exist_ok=True)

# Narrow channel data
NARROW_OFF_CSV = os.path.join(PROJECT_ROOT, "latent_data", "narrow_compare_commOFF_10x2000_20260210_173617.csv")
NARROW_ON_CSV = os.path.join(PROJECT_ROOT, "latent_data", "narrow_compare_commON_10x2000_20260210_173617.csv")

# Open (original) data
OPEN_OFF_CSV = os.path.join(PROJECT_ROOT, "latent_data", "compare_commOFF_10x2000_20260209_154752.csv")
OPEN_ON_CSV = os.path.join(PROJECT_ROOT, "latent_data", "compare_commON_10x2000_20260209_154752.csv")

MSG_DIM = 6
OBS_DIM = 369
COLREGS_LABELS = ['None', 'HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']
COLREGS_COLORS = ['#AAAAAA', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

# Colors
COLOR_OFF = '#E74C3C'
COLOR_ON = '#3498DB'

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


# ============================================================================
# 유틸리티 함수
# ============================================================================
def _load_colregs_data(csv_path):
    """비교 테스트 CSV에서 COLREGs 관련 데이터 추출"""
    df = pd.read_csv(csv_path)
    colregs = np.argmax(df[[f'obs_{i}' for i in range(364, 369)]].values, axis=1)
    yaw_rate = df['obs_363'].values
    speed = df['obs_362'].values
    goal_dist = df['obs_360'].values
    radar = df[[f'obs_{i}' for i in range(360)]].values
    radar_min = radar.min(axis=1)
    return colregs, yaw_rate, speed, goal_dist, radar_min


def _compute_composite_compliance(colregs, rmin, collision_rate, overall_clearance, danger_thresh=-0.3):
    """
    복합 COLREGs 준수율 계산.
    compliance = 0.2 * per_sit_safety + 0.4 * collision_avoidance + 0.4 * overall_clearance
    """
    situations = [1, 2, 3, 4]
    scores = {}
    collision_avoidance = 1.0 - collision_rate

    for sit in situations:
        mask = colregs == sit
        if mask.sum() > 0:
            danger_frac = (rmin[mask] < danger_thresh).sum() / mask.sum()
            per_sit_safety = 1.0 - danger_frac
        else:
            per_sit_safety = 0.5

        scores[sit] = (0.2 * per_sit_safety
                        + 0.4 * collision_avoidance
                        + 0.4 * overall_clearance)
    return scores


def load_latent_data_narrow():
    """Narrow channel commON CSV에서 latent 데이터 로드"""
    df = pd.read_csv(NARROW_ON_CSV)
    self_msg = df[[f'self_msg_{i}' for i in range(MSG_DIM)]].values
    others_msg = df[[f'others_msg_{i}' for i in range(MSG_DIM)]].values
    colregs = np.argmax(df[[f'obs_{i}' for i in range(364, 369)]].values, axis=1)

    return {
        'df': df,
        'self_msg': self_msg, 'others_msg': others_msg, 'colregs': colregs,
        'goal_dist': df['obs_360'].values, 'goal_angle': df['obs_361'].values,
        'speed': df['obs_362'].values, 'yaw_rate': df['obs_363'].values,
        'steps': df['step'].values,
    }


def _save_scatter_colregs(d2d, colregs_s, title, filename):
    """COLREGs 색상 scatter -> 개별 PNG"""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i in [0, 4, 3, 2, 1]:
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
# Figure 5 (Narrow): Test Performance Comparison
# ============================================================================
def plot_narrow_fig5_test_performance():
    print("[Narrow Fig 5] Test performance comparison")

    # Narrow channel test results
    off_collisions = np.array([2, 1, 0, 1, 3, 4, 1, 2, 0, 3])
    off_successes  = np.array([8, 2, 5, 2, 3, 7, 15, 2, 7, 4])
    off_rewards    = np.array([4084.27, 4042.94, 4178.47, 3853.16, 4072.30,
                               3935.54, 4200.89, 3936.84, 4025.05, 3986.93])

    on_collisions  = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 2])
    on_successes   = np.array([3, 6, 6, 3, 6, 7, 5, 9, 12, 5])
    on_rewards     = np.array([4499.75, 4446.32, 4323.60, 4326.37, 4370.23,
                               4531.71, 4438.77, 4570.02, 4676.10, 4324.58])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = [COLOR_OFF, COLOR_ON]
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
                    f'{means[i]:.2f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Comm OFF', 'Comm ON'], fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Narrow Channel Test Performance', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'narrow_fig5_test_performance.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, 'narrow_fig5_test_performance.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: narrow_fig5_test_performance.png / .pdf")


# ============================================================================
# Figure 8 (Narrow): COLREGs Composite Compliance Analysis
# ============================================================================
def plot_narrow_fig8_colregs_compliance():
    print("[Narrow Fig 8] COLREGs compliance analysis")

    colregs_off, yaw_off, speed_off, gdist_off, rmin_off = _load_colregs_data(NARROW_OFF_CSV)
    colregs_on, yaw_on, speed_on, gdist_on, rmin_on = _load_colregs_data(NARROW_ON_CSV)

    # Narrow test collision rates
    # OFF: 17 collisions, 55 successes -> total events = 72
    # ON:  3 collisions, 62 successes  -> total events = 65
    total_off = 17 + 55
    total_on = 3 + 62
    collision_rate_off = 17 / total_off   # 0.236
    collision_rate_on = 3 / total_on      # 0.046

    clearance_off = np.clip(rmin_off.mean() + 0.5, 0, 1)
    clearance_on = np.clip(rmin_on.mean() + 0.5, 0, 1)

    situations = [1, 2, 3, 4]
    sit_names = ['Head-on', 'Stand-on', 'Give-way', 'Overtaking']
    sit_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

    scores_off = _compute_composite_compliance(colregs_off, rmin_off, collision_rate_off, clearance_off)
    scores_on = _compute_composite_compliance(colregs_on, rmin_on, collision_rate_on, clearance_on)

    x = np.arange(len(situations))
    w = 0.35
    means_off = [scores_off[s] for s in situations]
    means_on = [scores_on[s] for s in situations]

    # --- (a) Compliance bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, means_off, w, color=COLOR_OFF, alpha=0.85,
                   edgecolor='black', linewidth=0.8, label='Comm OFF')
    bars2 = ax.bar(x + w/2, means_on, w, color=COLOR_ON, alpha=0.85,
                   edgecolor='black', linewidth=0.8, label='Comm ON')
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sit_names, fontsize=11)
    ax.set_ylabel('Composite compliance score', fontsize=11)
    ax.set_title('COLREGs compliance by situation (Narrow Channel)', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'narrow_fig8_colregs_a_compliance_bar.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- (b) Yaw rate distribution per COLREGs (violin plot) ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle('Yaw rate distribution by COLREGs situation (Narrow Channel)', fontsize=12, fontweight='bold')
    for idx, (sit, name) in enumerate(zip(situations, sit_names)):
        ax = axes[idx]
        mask_off = colregs_off == sit
        mask_on = colregs_on == sit
        yaw_data, labels, colors_v = [], [], []
        if mask_off.sum() > 0:
            yaw_data.append(yaw_off[mask_off])
            labels.append(f'OFF\n(n={mask_off.sum()})')
            colors_v.append(COLOR_OFF)
        if mask_on.sum() > 0:
            yaw_data.append(yaw_on[mask_on])
            labels.append(f'ON\n(n={mask_on.sum()})')
            colors_v.append(COLOR_ON)
        if len(yaw_data) > 0:
            vp = ax.violinplot(yaw_data, showmeans=True, showmedians=True)
            for i, body in enumerate(vp['bodies']):
                body.set_facecolor(colors_v[i])
                body.set_alpha(0.7)
            if 'cmeans' in vp:
                vp['cmeans'].set_color('black')
            if 'cmedians' in vp:
                vp['cmedians'].set_color('gray')
                vp['cmedians'].set_linestyle('--')
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
                        color=COLOR_OFF if lp == 'OFF' else COLOR_ON)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'narrow_fig8_colregs_b_yaw_violin.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- (c) Radar chart (spider/polar) ---
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(situations), endpoint=False).tolist()
    angles += angles[:1]
    vals_off = means_off + means_off[:1]
    vals_on = means_on + means_on[:1]
    ax.plot(angles, vals_off, 'o-', color=COLOR_OFF, linewidth=2, label='Comm OFF', markersize=8)
    ax.fill(angles, vals_off, color=COLOR_OFF, alpha=0.15)
    ax.plot(angles, vals_on, 's-', color=COLOR_ON, linewidth=2, label='Comm ON', markersize=8)
    ax.fill(angles, vals_on, color=COLOR_ON, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(sit_names, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('COLREGs composite compliance (Narrow Channel)', fontsize=12, pad=20)
    ax.legend(fontsize=10, loc='lower right', bbox_to_anchor=(1.15, -0.05))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'narrow_fig8_colregs_c_radar.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- (d) Component breakdown: stacked bar ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    component_names = ['Situation safety', 'Collision avoidance', 'Radar clearance']
    comp_off = np.zeros((len(situations), 3))
    comp_on = np.zeros((len(situations), 3))
    ca_off = 1.0 - collision_rate_off
    ca_on = 1.0 - collision_rate_on
    for i, sit in enumerate(situations):
        m_off = colregs_off == sit
        m_on = colregs_on == sit
        ps_off = 1.0 - ((rmin_off[m_off] < -0.3).sum() / max(m_off.sum(), 1)) if m_off.sum() > 0 else 0.5
        ps_on = 1.0 - ((rmin_on[m_on] < -0.3).sum() / max(m_on.sum(), 1)) if m_on.sum() > 0 else 0.5
        comp_off[i] = [0.2 * ps_off, 0.4 * ca_off, 0.4 * clearance_off]
        comp_on[i] = [0.2 * ps_on, 0.4 * ca_on, 0.4 * clearance_on]

    bar_w = 0.35
    x_pos = np.arange(len(situations))
    colors_comp = ['#F39C12', '#E74C3C', '#2ECC71']

    for mode, comp, offset, hatch in [('OFF', comp_off, -bar_w/2, ''), ('ON', comp_on, bar_w/2, '//')]:
        bottom = np.zeros(len(situations))
        for j in range(3):
            ax.bar(x_pos + offset, comp[:, j], bar_w, bottom=bottom,
                   color=colors_comp[j], alpha=0.7 if mode == 'OFF' else 0.9,
                   edgecolor='black', linewidth=0.5, hatch=hatch)
            bottom += comp[:, j]
        for i in range(len(situations)):
            total = comp[i].sum()
            ax.text(x_pos[i] + offset, total + 0.01, f'{total:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    legend_items = []
    for j in range(3):
        legend_items.append(Patch(facecolor=colors_comp[j], alpha=0.8, label=component_names[j]))
    legend_items.append(Patch(facecolor='white', edgecolor='black', label='OFF (solid)'))
    legend_items.append(Patch(facecolor='white', edgecolor='black', hatch='//', label='ON (hatched)'))
    ax.legend(handles=legend_items, fontsize=9, loc='upper right')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(sit_names, fontsize=11)
    ax.set_ylabel('Composite compliance score', fontsize=11)
    ax.set_title('Compliance score breakdown by component (Narrow Channel)', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'narrow_fig8_colregs_d_breakdown.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- Combined PDF (2x2) ---
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('COLREGs composite compliance analysis (Narrow Channel) -- Comm OFF vs Comm ON',
                 fontsize=14, fontweight='bold')
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # (a) Bar
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.bar(x - w/2, means_off, w, color=COLOR_OFF, alpha=0.85,
             edgecolor='black', linewidth=0.8, label='Comm OFF')
    ax_a.bar(x + w/2, means_on, w, color=COLOR_ON, alpha=0.85,
             edgecolor='black', linewidth=0.8, label='Comm ON')
    for mode_means in [means_off, means_on]:
        off_x = -w/2 if mode_means is means_off else w/2
        for i, m in enumerate(mode_means):
            ax_a.text(i + off_x, m + 0.01, f'{m:.2f}', ha='center', va='bottom',
                      fontsize=9, fontweight='bold')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(sit_names, fontsize=10)
    ax_a.set_ylabel('Compliance score')
    ax_a.set_title('(a) Composite compliance', fontsize=11)
    ax_a.set_ylim(0, 1.05)
    ax_a.legend(fontsize=9)
    ax_a.grid(axis='y', alpha=0.3)

    # (b) Radar
    ax_b = fig.add_subplot(gs[0, 1], polar=True)
    ax_b.plot(angles, vals_off, 'o-', color=COLOR_OFF, linewidth=2, label='Comm OFF', markersize=7)
    ax_b.fill(angles, vals_off, color=COLOR_OFF, alpha=0.15)
    ax_b.plot(angles, vals_on, 's-', color=COLOR_ON, linewidth=2, label='Comm ON', markersize=7)
    ax_b.fill(angles, vals_on, color=COLOR_ON, alpha=0.15)
    ax_b.set_xticks(angles[:-1])
    ax_b.set_xticklabels(sit_names, fontsize=10)
    ax_b.set_ylim(0, 1.0)
    ax_b.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.set_title('(b) Compliance radar', fontsize=11, pad=15)
    ax_b.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1.15, -0.05))

    # (c) Violin
    ax_v = fig.add_subplot(gs[1, 0])
    all_yaw, all_labels, all_colors, pos_list = [], [], [], []
    p = 1
    for si, sn in zip(situations, sit_names):
        m_off = colregs_off == si
        m_on = colregs_on == si
        if m_off.sum() > 0:
            all_yaw.append(yaw_off[m_off])
            all_labels.append(f'{sn[:6]}\nOFF')
            all_colors.append(COLOR_OFF)
            pos_list.append(p)
            p += 1
        if m_on.sum() > 0:
            all_yaw.append(yaw_on[m_on])
            all_labels.append(f'{sn[:6]}\nON')
            all_colors.append(COLOR_ON)
            pos_list.append(p)
            p += 1
        p += 0.5
    vp = ax_v.violinplot(all_yaw, positions=pos_list, showmeans=True, showmedians=True)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(all_colors[i])
        body.set_alpha(0.7)
    if 'cmeans' in vp:
        vp['cmeans'].set_color('black')
    if 'cmedians' in vp:
        vp['cmedians'].set_color('gray')
        vp['cmedians'].set_linestyle('--')
    ax_v.set_xticks(pos_list)
    ax_v.set_xticklabels(all_labels, fontsize=7)
    ax_v.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax_v.set_ylabel('Yaw rate')
    ax_v.set_title('(c) Yaw rate distribution', fontsize=11)
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
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(sit_names, fontsize=9)
    ax_d.set_ylabel('Score')
    ax_d.set_title('(d) Component breakdown', fontsize=11)
    ax_d.set_ylim(0, 1.05)
    ax_d.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'narrow_fig8_colregs_combined.pdf'), bbox_inches='tight')
    plt.close(fig)

    # 결과 출력
    print("  Composite compliance scores (Narrow Channel):")
    print(f"    Global: collision_avoidance OFF={1-collision_rate_off:.3f} ON={1-collision_rate_on:.3f}")
    print(f"    Global: radar_clearance    OFF={clearance_off:.3f} ON={clearance_on:.3f}")
    for sit, name in zip(situations, sit_names):
        print(f"    {name:12s}: OFF={scores_off[sit]:.3f}, ON={scores_on[sit]:.3f} (delta={scores_on[sit]-scores_off[sit]:+.3f})")
    print("  Saved: narrow_fig8_colregs_a~d + combined.pdf")


# ============================================================================
# Figure env_comparison: Open vs Narrow Environment Comparison
# ============================================================================
def plot_narrow_fig_env_comparison():
    print("[Narrow Fig Env] Open vs Narrow environment comparison")

    # Open test results
    open_off_collision_mean = 1.40
    open_on_collision_mean = 0.80
    open_off_success_mean = 7.60
    open_on_success_mean = 8.40
    open_off_reward_mean = 4128.64
    open_on_reward_mean = 4339.59

    # Narrow test results
    narrow_off_collision_mean = 1.70
    narrow_on_collision_mean = 0.30
    narrow_off_success_mean = 5.50
    narrow_on_success_mean = 6.20
    narrow_off_reward_mean = 4031.64
    narrow_on_reward_mean = 4450.74

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    # Color scheme for 4 bars: [Open OFF, Open ON, Narrow OFF, Narrow ON]
    bar_colors = ['#E74C3C', '#3498DB', '#C0392B', '#2980B9']
    bar_labels = ['Open OFF', 'Open ON', 'Narrow OFF', 'Narrow ON']
    x = np.arange(4)
    w = 0.55

    # --- (a) Collision Rate ---
    ax = axes[0]
    vals = [open_off_collision_mean, open_on_collision_mean,
            narrow_off_collision_mean, narrow_on_collision_mean]
    bars = ax.bar(x, vals, width=w, color=bar_colors, alpha=0.85,
                  edgecolor='black', linewidth=0.8)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals) * 0.02,
                f'{vals[i]:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9, rotation=15, ha='right')
    ax.set_title('(a) Collision rate', fontsize=11)
    ax.set_ylabel('Collisions per run')
    ax.grid(axis='y', alpha=0.3)

    # --- (b) Success Rate ---
    ax = axes[1]
    vals = [open_off_success_mean, open_on_success_mean,
            narrow_off_success_mean, narrow_on_success_mean]
    bars = ax.bar(x, vals, width=w, color=bar_colors, alpha=0.85,
                  edgecolor='black', linewidth=0.8)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals) * 0.02,
                f'{vals[i]:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9, rotation=15, ha='right')
    ax.set_title('(b) Success rate', fontsize=11)
    ax.set_ylabel('Successes per run')
    ax.grid(axis='y', alpha=0.3)

    # --- (c) Average Reward ---
    ax = axes[2]
    vals = [open_off_reward_mean, open_on_reward_mean,
            narrow_off_reward_mean, narrow_on_reward_mean]
    bars = ax.bar(x, vals, width=w, color=bar_colors, alpha=0.85,
                  edgecolor='black', linewidth=0.8)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals) * 0.005,
                f'{vals[i]:.0f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9, rotation=15, ha='right')
    ax.set_title('(c) Average reward', fontsize=11)
    ax.set_ylabel('Cumulative reward')
    ax.grid(axis='y', alpha=0.3)
    # Start y-axis from a reasonable baseline
    ax.set_ylim(3800, max(vals) * 1.05)

    # --- (d) Communication Benefit Comparison ---
    ax = axes[3]
    # Collision reduction %
    open_coll_reduction = (open_off_collision_mean - open_on_collision_mean) / open_off_collision_mean * 100
    narrow_coll_reduction = (narrow_off_collision_mean - narrow_on_collision_mean) / narrow_off_collision_mean * 100
    # Success increase %
    open_succ_increase = (open_on_success_mean - open_off_success_mean) / open_off_success_mean * 100
    narrow_succ_increase = (narrow_on_success_mean - narrow_off_success_mean) / narrow_off_success_mean * 100
    # Reward increase %
    open_reward_increase = (open_on_reward_mean - open_off_reward_mean) / open_off_reward_mean * 100
    narrow_reward_increase = (narrow_on_reward_mean - narrow_off_reward_mean) / narrow_off_reward_mean * 100

    metrics = ['Collision\nReduction', 'Success\nIncrease', 'Reward\nIncrease']
    open_benefits = [open_coll_reduction, open_succ_increase, open_reward_increase]
    narrow_benefits = [narrow_coll_reduction, narrow_succ_increase, narrow_reward_increase]

    x_ben = np.arange(len(metrics))
    w_ben = 0.35
    bars1 = ax.bar(x_ben - w_ben/2, open_benefits, w_ben, color='#E74C3C', alpha=0.75,
                   edgecolor='black', linewidth=0.8, label='Open Sea')
    bars2 = ax.bar(x_ben + w_ben/2, narrow_benefits, w_ben, color='#3498DB', alpha=0.75,
                   edgecolor='black', linewidth=0.8, label='Narrow Channel')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + max(max(open_benefits), max(narrow_benefits)) * 0.02,
                    f'{h:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xticks(x_ben)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title('(d) Communication benefit (%)', fontsize=11)
    ax.set_ylabel('Improvement (%)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Open Sea vs Narrow Channel: Environment Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'narrow_fig_env_comparison.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, 'narrow_fig_env_comparison.pdf'), bbox_inches='tight')
    plt.close(fig)

    print(f"  Communication benefit comparison:")
    print(f"    Collision reduction: Open={open_coll_reduction:.1f}%, Narrow={narrow_coll_reduction:.1f}%")
    print(f"    Success increase:    Open={open_succ_increase:.1f}%, Narrow={narrow_succ_increase:.1f}%")
    print(f"    Reward increase:     Open={open_reward_increase:.1f}%, Narrow={narrow_reward_increase:.1f}%")
    print(f"  Saved: narrow_fig_env_comparison.png / .pdf")


# ============================================================================
# Figure 2 (Narrow): t-SNE of Self Messages
# ============================================================================
def plot_narrow_fig2_tsne():
    print("[Narrow Fig 2] t-SNE analysis (self messages)")

    data = load_latent_data_narrow()
    self_msg = data['self_msg']
    colregs = data['colregs']
    yaw_rate = data['yaw_rate']
    speed = data['speed']
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

    msg_s = self_msg[balanced_idx]
    colregs_s = colregs[balanced_idx]
    yaw_s = yaw_rate[balanced_idx]
    speed_s = speed[balanced_idx]
    goal_dist_s = goal_dist[balanced_idx]

    print(f"  Balanced sample: {len(balanced_idx)} total")
    for c in range(5):
        print(f"    Class {COLREGS_LABELS[c]}: {(colregs_s == c).sum()}")

    # t-SNE
    print("  Running t-SNE on self messages...")
    msg_2d = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(msg_s)

    # 개별 PNG 저장
    paths = []

    # (a) Self message - COLREGs
    paths.append(_save_scatter_colregs(
        msg_2d, colregs_s,
        f'Self message space ({MSG_DIM}D) - Narrow Channel',
        'narrow_fig2_tsne_a_msg_colregs'))

    # (b) Self message - Yaw Rate
    paths.append(_save_scatter_continuous(
        msg_2d, yaw_s,
        'Self message colored by yaw rate - Narrow',
        'Yaw Rate', 'narrow_fig2_tsne_b_msg_yawrate', cmap='coolwarm'))

    # (c) Self message - Goal Distance
    paths.append(_save_scatter_continuous(
        msg_2d, goal_dist_s,
        'Self message colored by goal distance - Narrow',
        'Goal Dist', 'narrow_fig2_tsne_c_msg_goaldist', cmap='viridis'))

    # (d) Self message - Speed
    paths.append(_save_scatter_continuous(
        msg_2d, speed_s,
        'Self message colored by speed - Narrow',
        'Speed', 'narrow_fig2_tsne_d_msg_speed', cmap='viridis'))

    # 합본 PDF (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle('t-SNE analysis of self messages (Narrow Channel)', fontsize=14, fontweight='bold')

    plot_specs = [
        (axes[0, 0], msg_2d, colregs_s, 'colregs', f'(a) Colored by COLREGs ({MSG_DIM}D)'),
        (axes[0, 1], msg_2d, yaw_s, 'cont_coolwarm', '(b) Colored by yaw rate'),
        (axes[1, 0], msg_2d, goal_dist_s, 'cont_viridis', '(c) Colored by goal distance'),
        (axes[1, 1], msg_2d, speed_s, 'cont_viridis', '(d) Colored by speed'),
    ]

    for ax, d2d, vals, mode, title in plot_specs:
        if mode == 'colregs':
            for i in [0, 4, 3, 2, 1]:
                mask = vals == i
                if mask.sum() > 0:
                    ax.scatter(d2d[mask, 0], d2d[mask, 1], c=COLREGS_COLORS[i],
                              label=COLREGS_LABELS[i], alpha=0.35 if i == 0 else 0.8,
                              s=4 if i == 0 else 12, edgecolors='none',
                              zorder=2 if i == 0 else 3)
            ax.legend(fontsize=7, markerscale=2)
        else:
            cmap = 'viridis' if 'viridis' in mode else 'coolwarm'
            sc = ax.scatter(d2d[:, 0], d2d[:, 1], c=vals, cmap=cmap,
                           alpha=0.6, s=6, edgecolors='none')
            plt.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('t-SNE dim 1', fontsize=9)
        ax.set_ylabel('t-SNE dim 2', fontsize=9)
        ax.grid(alpha=0.15)

    plt.tight_layout()
    pdf_path = os.path.join(SAVE_DIR, 'narrow_fig2_tsne_combined.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    for p in paths:
        print(f"  Saved: {p}")
    print(f"  Saved: {pdf_path}")


# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Narrow Channel Analysis Graphs")
    print(f"Output: {SAVE_DIR}")
    print("=" * 60)

    plot_narrow_fig5_test_performance()
    plot_narrow_fig8_colregs_compliance()
    plot_narrow_fig_env_comparison()
    plot_narrow_fig2_tsne()

    print("\n" + "=" * 60)
    print("All narrow channel analysis graphs generated!")
    print(f"Output: {SAVE_DIR}")
    print("=" * 60)
