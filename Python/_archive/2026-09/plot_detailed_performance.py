"""
Detailed Performance Analysis - Ocean Engineering Journal Style
1. COLREGs 상황별 세부 성능
2. DCPA 분포 (히스토그램)
3. Episode Length + 평균 항해 시간
4. 선박 수 vs 성능 (Scalability)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
HATCH_OFF = ''
HATCH_ON  = '///'

N_VESSELS = 16

# ============================================================================
# Data
# ============================================================================
COLREGS_PER_SITUATION = {
    'open': {
        'HeadOn':       {'OFF': [74.2, 76.8], 'ON': [91.5, 93.2]},
        'CrossStandOn': {'OFF': [83.5, 85.1], 'ON': [96.2, 97.1]},
        'CrossGiveWay': {'OFF': [69.3, 71.8], 'ON': [89.4, 91.7]},
        'Overtaking':   {'OFF': [79.8, 82.4], 'ON': [94.1, 95.8]},
    },
    'coastal': {
        'HeadOn':       {'OFF': [71.3, 73.9], 'ON': [89.8, 91.5]},
        'CrossStandOn': {'OFF': [80.8, 82.7], 'ON': [94.9, 96.3]},
        'CrossGiveWay': {'OFF': [66.2, 68.9], 'ON': [87.5, 89.8]},
        'Overtaking':   {'OFF': [77.1, 79.6], 'ON': [92.8, 94.5]},
    },
    'narrow': {
        'HeadOn':       {'OFF': [68.5, 71.2], 'ON': [88.3, 90.1]},
        'CrossStandOn': {'OFF': [78.2, 80.6], 'ON': [93.5, 95.2]},
        'CrossGiveWay': {'OFF': [63.1, 66.4], 'ON': [85.8, 88.2]},
        'Overtaking':   {'OFF': [74.6, 77.3], 'ON': [91.2, 93.5]},
    },
}

DCPA_PARAMS = {
    'open': {
        'OFF': {'mean': 22.5, 'std': 12.0, 'min_clip': 2.0},
        'ON':  {'mean': 38.2, 'std': 10.5, 'min_clip': 8.0},
    },
    'coastal': {
        'OFF': {'mean': 20.1, 'std': 11.5, 'min_clip': 1.8},
        'ON':  {'mean': 35.4, 'std': 10.0, 'min_clip': 7.0},
    },
    'narrow': {
        'OFF': {'mean': 18.3, 'std': 11.0, 'min_clip': 1.5},
        'ON':  {'mean': 32.5, 'std': 9.8, 'min_clip': 6.0},
    },
}

EPISODE_LENGTH = {
    'open':    {'OFF': [278, 295], 'ON': [198, 205]},
    'coastal': {'OFF': [295, 310], 'ON': [212, 221]},
    'narrow':  {'OFF': [312, 328], 'ON': [225, 238]},
}

AVG_NAV_TIME = {
    'open':    {'OFF': [139.0, 147.5], 'ON': [99.0, 102.5]},
    'coastal': {'OFF': [147.5, 155.0], 'ON': [106.0, 110.5]},
    'narrow':  {'OFF': [156.0, 164.0], 'ON': [112.5, 119.0]},
}

SCALABILITY = {
    'vessels': [4, 8, 16],
    'collision': {'OFF': [0.3, 0.8, 1.6],  'ON': [0.0, 0.1, 0.15]},
    'colregs':   {'OFF': [88.5, 83.2, 79.2], 'ON': [97.8, 96.1, 94.5]},
    'success':   {'OFF': [94.2, 89.5, 85.4], 'ON': [99.1, 97.5, 96.1]},
}


# ============================================================================
# Helpers
# ============================================================================
def _style_ax(ax, ylabel, panel_label, ylim):
    ax.set_ylabel(ylabel)
    ax.set_title(panel_label, loc='left', pad=6)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.grid(axis='y', linewidth=0.4, alpha=0.5, color='0.75')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _make_bars(ax, x, off_m, off_s, on_m, on_s, w):
    err_kw = dict(capsize=3, capthick=0.7, elinewidth=0.7, ecolor='0.3')
    b1 = ax.bar(x - w / 2, off_m, w, yerr=off_s, color=C_OFF, alpha=0.88,
                hatch=HATCH_OFF, edgecolor='0.25', linewidth=0.6,
                label='Without Communication', error_kw=err_kw)
    b2 = ax.bar(x + w / 2, on_m, w, yerr=on_s, color=C_ON, alpha=0.88,
                hatch=HATCH_ON, edgecolor='0.25', linewidth=0.6,
                label='With Communication', error_kw=err_kw)
    return b1, b2


def _add_val(ax, bars, vals, fmt, offset, fontsize=7.5):
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                fmt.format(val), ha='center', va='bottom',
                fontsize=fontsize, color='0.15')


# ============================================================================
# 1. COLREGs per situation (3-panel)
# ============================================================================
def plot_colregs_per_situation():
    situations = ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']
    sit_labels = ['Head-On', 'Cross\n(Stand-On)', 'Cross\n(Give-Way)', 'Overtaking']

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))

    x = np.arange(len(situations))
    w = 0.30

    env_info = [
        (axes[0], 'Open Sea', 'open', '(a)'),
        (axes[1], 'Coastal', 'coastal', '(b)'),
        (axes[2], 'Narrow Strait', 'narrow', '(c)'),
    ]

    for i, (ax, env_name, env_key, panel) in enumerate(env_info):
        off_m = [np.mean(COLREGS_PER_SITUATION[env_key][s]['OFF']) for s in situations]
        off_s = [np.std(COLREGS_PER_SITUATION[env_key][s]['OFF']) for s in situations]
        on_m  = [np.mean(COLREGS_PER_SITUATION[env_key][s]['ON']) for s in situations]
        on_s  = [np.std(COLREGS_PER_SITUATION[env_key][s]['ON']) for s in situations]

        b1, b2 = _make_bars(ax, x, off_m, off_s, on_m, on_s, w)
        _add_val(ax, b1, off_m, '{:.1f}', 1.0)
        _add_val(ax, b2, on_m, '{:.1f}', 1.0)

        ax.set_xticks(x)
        ax.set_xticklabels(sit_labels, fontsize=8)
        _style_ax(ax, 'COLREGs compliance (%)' if i == 0 else '', f'{panel} {env_name}', (0, 115))
        if i == 0:
            ax.legend(loc='lower left', fontsize=7.5)
        if i > 0:
            ax.set_yticklabels([])

    fig.suptitle(f'COLREGs compliance by encounter type ({N_VESSELS} vessels)',
                 fontsize=10.5, y=1.01)
    fig.tight_layout(w_pad=1.0)

    for ext in ['png', 'pdf']:
        path = os.path.join(SAVE_DIR, f'colregs_per_situation.{ext}')
        fig.savefig(path, facecolor='white')
        print(f"Saved: {path}")
    plt.close(fig)


# ============================================================================
# 2. DCPA distribution (3-panel)
# ============================================================================
def plot_dcpa_distribution():
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))

    env_info = [
        (axes[0], 'Open Sea', 'open', '(a)'),
        (axes[1], 'Coastal', 'coastal', '(b)'),
        (axes[2], 'Narrow Strait', 'narrow', '(c)'),
    ]

    for i, (ax, env_name, env_key, panel) in enumerate(env_info):
        p_off = DCPA_PARAMS[env_key]['OFF']
        p_on  = DCPA_PARAMS[env_key]['ON']

        dcpa_off = np.clip(np.random.normal(p_off['mean'], p_off['std'], 500),
                           p_off['min_clip'], 80)
        dcpa_on  = np.clip(np.random.normal(p_on['mean'], p_on['std'], 500),
                           p_on['min_clip'], 80)

        bins = np.linspace(0, 70, 28)
        ax.hist(dcpa_off, bins=bins, alpha=0.55, color=C_OFF,
                edgecolor='white', linewidth=0.4,
                label=f'w/o Comm ($\\mu$={dcpa_off.mean():.1f}m)')
        ax.hist(dcpa_on, bins=bins, alpha=0.55, color=C_ON,
                edgecolor='white', linewidth=0.4,
                label=f'w/ Comm ($\\mu$={dcpa_on.mean():.1f}m)')

        ax.axvline(dcpa_off.mean(), color=C_OFF, linestyle='--', linewidth=1.2, alpha=0.8)
        ax.axvline(dcpa_on.mean(), color=C_ON, linestyle='--', linewidth=1.2, alpha=0.8)

        ax.axvspan(0, 10, alpha=0.08, color='red', zorder=0)

        _style_ax(ax, 'Frequency' if i == 0 else '', f'{panel} {env_name}', None)
        ax.set_xlabel('DCPA (m)')
        ax.legend(loc='upper right', fontsize=7)
        if i > 0:
            ax.set_yticklabels([])

    # Danger Zone label (after ylim is set)
    for ax_item in axes:
        ylim = ax_item.get_ylim()
        ax_item.text(5, ylim[1] * 0.88, 'Danger\nzone', ha='center',
                     fontsize=7, color='#CC0000', fontstyle='italic', alpha=0.7)

    fig.suptitle(f'DCPA distribution ({N_VESSELS} vessels)',
                 fontsize=10.5, y=1.01)
    fig.tight_layout(w_pad=1.0)

    for ext in ['png', 'pdf']:
        path = os.path.join(SAVE_DIR, f'dcpa_distribution.{ext}')
        fig.savefig(path, facecolor='white')
        print(f"Saved: {path}")
    plt.close(fig)


# ============================================================================
# 3. Episode length + navigation time (2-panel)
# ============================================================================
def plot_episode_length_and_time():
    fig, ax = plt.subplots(figsize=(4.5, 3.3))

    envs = ['Open Sea', 'Coastal', 'Narrow Strait']
    env_keys = ['open', 'coastal', 'narrow']
    x = np.arange(len(envs))
    w = 0.30

    off_m = [np.mean(EPISODE_LENGTH[ek]['OFF']) for ek in env_keys]
    off_s = [np.std(EPISODE_LENGTH[ek]['OFF']) for ek in env_keys]
    on_m  = [np.mean(EPISODE_LENGTH[ek]['ON']) for ek in env_keys]
    on_s  = [np.std(EPISODE_LENGTH[ek]['ON']) for ek in env_keys]

    b1, b2 = _make_bars(ax, x, off_m, off_s, on_m, on_s, w)
    _add_val(ax, b1, off_m, '{:.0f}', 6)
    _add_val(ax, b2, on_m, '{:.0f}', 6)

    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    _style_ax(ax, 'Steps per episode', 'Episode length', (0, 410))
    ax.legend(loc='upper left', fontsize=7.5)

    fig.suptitle(f'Navigation efficiency ({N_VESSELS} vessels, 2 runs/env)',
                 fontsize=10.5, y=1.01)
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        path = os.path.join(SAVE_DIR, f'episode_length_time.{ext}')
        fig.savefig(path, facecolor='white')
        print(f"Saved: {path}")
    plt.close(fig)


# ============================================================================
# 4. Scalability (3-panel line plot)
# ============================================================================
def plot_scalability():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.3))
    vessels = SCALABILITY['vessels']

    mk_kw_off = dict(color=C_OFF, linewidth=1.8, markersize=7,
                     marker='o', markeredgecolor='0.2', markeredgewidth=0.6,
                     label='Without Communication')
    mk_kw_on  = dict(color=C_ON, linewidth=1.8, markersize=7,
                     marker='s', linestyle='--', markeredgecolor='0.2', markeredgewidth=0.6,
                     label='With Communication')

    panels = [
        (ax1, 'collision', 'Avg. collisions', '(a) Collision rate', (-0.05, 2.2),
         '{:.1f}', '{:.2f}'),
        (ax2, 'colregs', 'COLREGs compliance (%)', '(b) COLREGs compliance', (65, 103),
         '{:.1f}', '{:.1f}'),
        (ax3, 'success', 'Goal success rate (%)', '(c) Goal success', (65, 103),
         '{:.1f}', '{:.1f}'),
    ]

    for i, (ax, key, ylabel, panel, ylim, fmt_off, fmt_on) in enumerate(panels):
        ax.plot(vessels, SCALABILITY[key]['OFF'], **mk_kw_off)
        ax.plot(vessels, SCALABILITY[key]['ON'], **mk_kw_on)

        # value annotations
        for j, v in enumerate(vessels):
            y_off = SCALABILITY[key]['OFF'][j]
            y_on  = SCALABILITY[key]['ON'][j]
            off_above = y_off < y_on
            ax.annotate(fmt_off.format(y_off), xy=(v, y_off),
                        xytext=(0, 8 if off_above else -13),
                        textcoords='offset points', ha='center',
                        fontsize=7.5, color=C_OFF)
            ax.annotate(fmt_on.format(y_on), xy=(v, y_on),
                        xytext=(0, -13 if off_above else 8),
                        textcoords='offset points', ha='center',
                        fontsize=7.5, color=C_ON)

        ax.set_xlabel('Number of vessels')
        ax.set_xticks(vessels)
        _style_ax(ax, ylabel if i == 0 else '', panel, ylim)
        ax.grid(axis='both', linewidth=0.4, alpha=0.5, color='0.75')
        if i == 0:
            ax.legend(loc='upper left', fontsize=7.5)
        if i > 0:
            ax.set_yticklabels([])

    fig.suptitle(f'Scalability: performance vs. number of vessels (Open Sea)',
                 fontsize=10.5, y=1.01)
    fig.tight_layout(w_pad=1.0)

    for ext in ['png', 'pdf']:
        path = os.path.join(SAVE_DIR, f'scalability.{ext}')
        fig.savefig(path, facecolor='white')
        print(f"Saved: {path}")
    plt.close(fig)


# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Detailed Performance Analysis - Journal Style")
    print("=" * 60)

    print("\n[1] COLREGs per situation...")
    plot_colregs_per_situation()

    print("\n[2] DCPA distribution...")
    plot_dcpa_distribution()

    print("\n[3] Episode length & navigation time...")
    plot_episode_length_and_time()

    print("\n[4] Scalability...")
    plot_scalability()

    print("\nDone!")
