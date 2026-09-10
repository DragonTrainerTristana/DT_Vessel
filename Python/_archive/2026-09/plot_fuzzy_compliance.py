"""
COLREGs Fuzzy Composite Compliance - Ocean Engineering Journal Style
- 실제 비교 테스트 CSV에서 radar 기반 composite compliance 계산
- (a) Bar chart: 상황별 composite score
- (b) Radar chart: spider/polar 비교
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SAVE_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================================
# Journal style
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.linewidth': 0.8,
    'legend.fontsize': 8.5,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '0.6',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

C_OFF = '#4878A8'
C_ON  = '#D4652F'

# ============================================================================
# Data paths
# ============================================================================
COMPARE_OFF_CSV = os.path.join(PROJECT_ROOT, "latent_data",
                               "compare_commOFF_10x2000_20260209_154752.csv")
COMPARE_ON_CSV = os.path.join(PROJECT_ROOT, "latent_data",
                              "compare_commON_10x2000_20260209_154752.csv")

SITUATIONS = [1, 2, 3, 4]
SIT_NAMES = ['Head-on', 'Stand-on', 'Give-way', 'Overtaking']
DANGER_THRESH = -0.3


def load_colregs_data(csv_path):
    df = pd.read_csv(csv_path)
    colregs = np.argmax(df[[f'obs_{i}' for i in range(364, 369)]].values, axis=1)
    radar = df[[f'obs_{i}' for i in range(360)]].values
    radar_min = radar.min(axis=1)
    return colregs, radar_min


def compute_composite_compliance(colregs, rmin, collision_rate, overall_clearance):
    """
    복합 COLREGs 준수율:
      0.2 * per_situation_safety + 0.4 * collision_avoidance + 0.4 * overall_clearance
    """
    collision_avoidance = 1.0 - collision_rate
    scores = {}
    components = {}

    for sit in SITUATIONS:
        mask = colregs == sit
        if mask.sum() > 0:
            danger_frac = (rmin[mask] < DANGER_THRESH).sum() / mask.sum()
            per_sit_safety = 1.0 - danger_frac
        else:
            per_sit_safety = 0.5

        s1 = 0.2 * per_sit_safety
        s2 = 0.4 * collision_avoidance
        s3 = 0.4 * overall_clearance
        scores[sit] = s1 + s2 + s3
        components[sit] = (s1, s2, s3)

    return scores, components


def plot_fuzzy_compliance():
    print("Loading comparison data...")
    colregs_off, rmin_off = load_colregs_data(COMPARE_OFF_CSV)
    colregs_on, rmin_on = load_colregs_data(COMPARE_ON_CSV)

    # 전체 테스트 결과
    total_off = 14 + 76
    total_on = 8 + 84
    collision_rate_off = 14 / total_off
    collision_rate_on = 8 / total_on

    clearance_off = np.clip(rmin_off.mean() + 0.5, 0, 1)
    clearance_on = np.clip(rmin_on.mean() + 0.5, 0, 1)

    scores_off, comp_off = compute_composite_compliance(
        colregs_off, rmin_off, collision_rate_off, clearance_off)
    scores_on, comp_on = compute_composite_compliance(
        colregs_on, rmin_on, collision_rate_on, clearance_on)

    means_off = [scores_off[s] for s in SITUATIONS]
    means_on = [scores_on[s] for s in SITUATIONS]

    print(f"  Clearance OFF: {clearance_off:.3f}, ON: {clearance_on:.3f}")
    print(f"  Collision rate OFF: {collision_rate_off:.3f}, ON: {collision_rate_on:.3f}")
    for sit, name in zip(SITUATIONS, SIT_NAMES):
        print(f"  {name}: OFF={scores_off[sit]:.3f}, ON={scores_on[sit]:.3f}")

    # ========================================================================
    # Figure: (a) bar chart + (b) radar chart + (c) component breakdown
    # ========================================================================
    fig = plt.figure(figsize=(10, 3.8))

    # Layout: bar(left), radar(center), breakdown(right)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.3], wspace=0.35)

    # --- (a) Composite compliance bar chart ---
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(SITUATIONS))
    w = 0.30

    err_kw = dict(capsize=3, capthick=0.7, elinewidth=0.7, ecolor='0.3')
    b1 = ax_a.bar(x - w / 2, means_off, w, color=C_OFF, alpha=0.88,
                  edgecolor='0.25', linewidth=0.6,
                  label='Without Communication', error_kw=err_kw)
    b2 = ax_a.bar(x + w / 2, means_on, w, color=C_ON, alpha=0.88,
                  hatch='///', edgecolor='0.25', linewidth=0.6,
                  label='With Communication', error_kw=err_kw)

    for bar, val in zip(b1, means_off):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, color='0.15')
    for bar, val in zip(b2, means_on):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, color='0.15')

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(SIT_NAMES, fontsize=8)
    ax_a.set_ylabel('Composite compliance score')
    ax_a.set_title('(a) Compliance by encounter', loc='left', pad=6)
    ax_a.set_ylim(0, 1.08)
    ax_a.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax_a.grid(axis='y', linewidth=0.4, alpha=0.5, color='0.75')
    ax_a.set_axisbelow(True)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.legend(loc='lower left', fontsize=7)

    # --- (b) Radar / Spider chart ---
    ax_b = fig.add_subplot(gs[0, 1], polar=True)

    angles = np.linspace(0, 2 * np.pi, len(SITUATIONS), endpoint=False).tolist()
    angles += angles[:1]
    vals_off = means_off + means_off[:1]
    vals_on = means_on + means_on[:1]

    ax_b.plot(angles, vals_off, 'o-', color=C_OFF, linewidth=1.5, markersize=5,
              label='w/o Comm', markeredgecolor='0.2', markeredgewidth=0.5)
    ax_b.fill(angles, vals_off, color=C_OFF, alpha=0.12)
    ax_b.plot(angles, vals_on, 's--', color=C_ON, linewidth=1.5, markersize=5,
              label='w/ Comm', markeredgecolor='0.2', markeredgewidth=0.5)
    ax_b.fill(angles, vals_on, color=C_ON, alpha=0.12)

    ax_b.set_xticks(angles[:-1])
    ax_b.set_xticklabels(SIT_NAMES, fontsize=7.5)
    ax_b.set_ylim(0, 1.0)
    ax_b.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax_b.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=7, color='0.4')
    ax_b.set_title('(b) Radar view', loc='center', pad=15, fontsize=11, fontweight='bold')
    ax_b.legend(loc='lower right', bbox_to_anchor=(1.25, -0.08), fontsize=7)
    ax_b.grid(True, linewidth=0.4, alpha=0.5, color='0.7')
    ax_b.spines['polar'].set_linewidth(0.5)

    # --- (c) Component breakdown: stacked bar ---
    ax_c = fig.add_subplot(gs[0, 2])
    component_names = ['Situation safety', 'Collision avoidance', 'Radar clearance']
    comp_colors = ['#6BAED6', '#FC8D59', '#78C679']

    bar_w = 0.30
    x_pos = np.arange(len(SITUATIONS))

    for mode, comp_dict, offset, hatch in [
        ('OFF', comp_off, -bar_w / 2, ''),
        ('ON', comp_on, bar_w / 2, '///')
    ]:
        bottom = np.zeros(len(SITUATIONS))
        for j in range(3):
            vals = [comp_dict[sit][j] for sit in SITUATIONS]
            ax_c.bar(x_pos + offset, vals, bar_w, bottom=bottom,
                     color=comp_colors[j], alpha=0.75 if mode == 'OFF' else 0.9,
                     edgecolor='0.3', linewidth=0.5, hatch=hatch)
            bottom += vals

        for i in range(len(SITUATIONS)):
            total = sum(comp_dict[SITUATIONS[i]])
            ax_c.text(x_pos[i] + offset, total + 0.01, f'{total:.2f}',
                      ha='center', va='bottom', fontsize=7, color='0.15')

    legend_items = [Patch(facecolor=comp_colors[j], alpha=0.8, edgecolor='0.3',
                          label=component_names[j]) for j in range(3)]
    legend_items.append(Patch(facecolor='white', edgecolor='0.3', label='w/o Comm'))
    legend_items.append(Patch(facecolor='white', edgecolor='0.3', hatch='///', label='w/ Comm'))
    ax_c.legend(handles=legend_items, fontsize=6.5, loc='lower left')

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(SIT_NAMES, fontsize=8)
    ax_c.set_ylabel('Score contribution')
    ax_c.set_title('(c) Component breakdown', loc='left', pad=6)
    ax_c.set_ylim(0, 1.08)
    ax_c.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax_c.grid(axis='y', linewidth=0.4, alpha=0.5, color='0.75')
    ax_c.set_axisbelow(True)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    fig.suptitle('COLREGs composite compliance analysis (16 vessels)',
                 fontsize=10.5, y=1.02)

    for ext in ['png', 'pdf']:
        path = os.path.join(SAVE_DIR, f'fuzzy_compliance.{ext}')
        fig.savefig(path, facecolor='white')
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("  COLREGs Fuzzy Compliance - Journal Style")
    print("=" * 60)
    plot_fuzzy_compliance()
    print("\nDone!")
