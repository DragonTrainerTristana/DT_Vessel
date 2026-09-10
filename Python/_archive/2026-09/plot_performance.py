"""
Performance Comparison: Comm OFF vs Comm ON
- 16 vessels, Open Sea / Coastal / Narrow Strait (2 runs each)
- Ocean Engineering journal style
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
# Journal style settings
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
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.6',
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ============================================================================
# 실험 결과 (16 vessels, 2 runs per environment)
# ============================================================================
N_VESSELS = 16
N_RUNS = 2
N_EVAL_STEPS = '30K'

COLLISION = {
    'open':    {'OFF': [1.4, 1.8], 'ON': [0.2, 0.1]},
    'coastal': {'OFF': [1.9, 2.2], 'ON': [0.3, 0.2]},
    'narrow':  {'OFF': [2.3, 2.7], 'ON': [0.4, 0.3]},
}

COLREGS = {
    'open':    {'OFF': [78.3, 80.1], 'ON': [93.8, 95.2]},
    'coastal': {'OFF': [75.1, 77.5], 'ON': [92.0, 93.8]},
    'narrow':  {'OFF': [72.4, 74.8], 'ON': [90.5, 92.1]},
}

SUCCESS = {
    'open':    {'OFF': [84.5, 86.2], 'ON': [95.3, 96.8]},
    'coastal': {'OFF': [79.2, 81.5], 'ON': [93.6, 95.1]},
    'narrow':  {'OFF': [73.8, 76.2], 'ON': [91.4, 93.0]},
}

# 학술 컬러 (색맹 친화 + 흑백 인쇄 hatching 대비)
C_OFF = '#4878A8'   # steel blue
C_ON  = '#D4652F'   # burnt orange
HATCH_OFF = ''
HATCH_ON  = '///'


def mean_std(data):
    return np.mean(data), np.std(data)


def _add_val(ax, bars, vals, fmt, color, offset, fontsize=8):
    """바 위에 값 표시 (절제된 스타일)"""
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                fmt.format(val), ha='center', va='bottom',
                fontsize=fontsize, color=color)


def _style_ax(ax, ylabel, panel_label, ylim, envs, x, legend=False):
    """공통 축 스타일"""
    ax.set_ylabel(ylabel)
    ax.set_title(panel_label, loc='left', pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.grid(axis='y', linewidth=0.4, alpha=0.5, color='0.75')
    ax.set_axisbelow(True)
    # 상단/우측 축 제거
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend:
        ax.legend(loc='upper left', frameon=True)


def _make_bars(ax, x, off_m, off_s, on_m, on_s, w):
    """바 생성 (hatching + error bar)"""
    err_kw = dict(capsize=4, capthick=0.8, elinewidth=0.8, ecolor='0.3')
    b1 = ax.bar(x - w / 2, off_m, w, yerr=off_s, color=C_OFF, alpha=0.88,
                hatch=HATCH_OFF, edgecolor='white', linewidth=0.6,
                label='Without Communication', error_kw=err_kw)
    b2 = ax.bar(x + w / 2, on_m, w, yerr=on_s, color=C_ON, alpha=0.88,
                hatch=HATCH_ON, edgecolor='white', linewidth=0.6,
                label='With Communication', error_kw=err_kw)
    # 바 테두리
    for b in b1:
        b.set_edgecolor('0.25')
        b.set_linewidth(0.6)
    for b in b2:
        b.set_edgecolor('0.25')
        b.set_linewidth(0.6)
    return b1, b2


# ============================================================================
# Main 4-panel
# ============================================================================
def plot_main_performance():
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.5))

    envs = ['Open Sea', 'Coastal', 'Narrow Strait']
    env_keys = ['open', 'coastal', 'narrow']
    x = np.arange(len(envs))
    w = 0.30

    panels = [
        (axes[0, 0], '(a)', COLLISION, 'Avg. collisions per episode', (0, 4.2), '{:.1f}', 0.25),
        (axes[0, 1], '(b)', COLREGS,   'COLREGs compliance (%)',      (0, 115), '{:.1f}', 1.2),
        (axes[1, 0], '(c)', SUCCESS,   'Goal success rate (%)',        (0, 115), '{:.1f}', 1.2),
    ]

    for i, (ax, label, data, ylabel, ylim, fmt, offset) in enumerate(panels):
        off_m = [np.mean(data[ek]['OFF']) for ek in env_keys]
        off_s = [np.std(data[ek]['OFF']) for ek in env_keys]
        on_m  = [np.mean(data[ek]['ON']) for ek in env_keys]
        on_s  = [np.std(data[ek]['ON']) for ek in env_keys]

        b1, b2 = _make_bars(ax, x, off_m, off_s, on_m, on_s, w)
        _add_val(ax, b1, off_m, fmt, '0.15', offset)
        _add_val(ax, b2, on_m, fmt, '0.15', offset)
        _style_ax(ax, ylabel, label, ylim, envs, x, legend=(i == 0))

    # ============ (d) Normalized Summary ============
    ax = axes[1, 1]
    metrics = ['Collision\n(lower = better)', 'COLREGs\ncompliance', 'Goal\nsuccess']

    off_raw, on_raw = [], []
    for ek in env_keys:
        off_raw.append([
            1 - np.mean(COLLISION[ek]['OFF']) / 10,
            np.mean(COLREGS[ek]['OFF']) / 100,
            np.mean(SUCCESS[ek]['OFF']) / 100,
        ])
        on_raw.append([
            1 - np.mean(COLLISION[ek]['ON']) / 10,
            np.mean(COLREGS[ek]['ON']) / 100,
            np.mean(SUCCESS[ek]['ON']) / 100,
        ])

    off_avg = np.mean(off_raw, axis=0)
    on_avg = np.mean(on_raw, axis=0)
    x_m = np.arange(len(metrics))

    err_kw = dict(capsize=4, capthick=0.8, elinewidth=0.8, ecolor='0.3')
    b1 = ax.bar(x_m - w / 2, off_avg, w, color=C_OFF, alpha=0.88,
                hatch=HATCH_OFF, edgecolor='0.25', linewidth=0.6,
                label='Without Communication', error_kw=err_kw)
    b2 = ax.bar(x_m + w / 2, on_avg, w, color=C_ON, alpha=0.88,
                hatch=HATCH_ON, edgecolor='0.25', linewidth=0.6,
                label='With Communication', error_kw=err_kw)
    _add_val(ax, b1, off_avg, '{:.2f}', '0.15', 0.015)
    _add_val(ax, b2, on_avg, '{:.2f}', '0.15', 0.015)

    ax.axhline(1.0, color='0.6', linestyle=':', linewidth=0.7)
    _style_ax(ax, 'Normalized score', '(d)', (0, 1.18), metrics, x_m)

    fig.suptitle(
        f'Performance comparison ({N_VESSELS} vessels, {N_RUNS} runs/env, averaged over {N_EVAL_STEPS} steps)',
        fontsize=10.5, y=0.995)
    fig.tight_layout(h_pad=2.2, w_pad=2.0)

    for ext in ['png', 'pdf']:
        path = os.path.join(SAVE_DIR, f'performance_comparison.{ext}')
        fig.savefig(path, facecolor='white')
        print(f"Saved: {path}")
    plt.close(fig)


# ============================================================================
# Focused 2-panel (Collision + COLREGs)
# ============================================================================
def plot_collision_colregs_focused():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.3))

    envs = ['Open Sea', 'Coastal', 'Narrow Strait']
    env_keys = ['open', 'coastal', 'narrow']
    x = np.arange(len(envs))
    w = 0.30

    # Collision
    off_m = [np.mean(COLLISION[ek]['OFF']) for ek in env_keys]
    off_s = [np.std(COLLISION[ek]['OFF']) for ek in env_keys]
    on_m  = [np.mean(COLLISION[ek]['ON']) for ek in env_keys]
    on_s  = [np.std(COLLISION[ek]['ON']) for ek in env_keys]

    b1, b2 = _make_bars(ax1, x, off_m, off_s, on_m, on_s, w)
    _add_val(ax1, b1, off_m, '{:.1f}', '0.15', 0.22)
    _add_val(ax1, b2, on_m, '{:.1f}', '0.15', 0.22)
    _style_ax(ax1, 'Avg. collisions per episode', '(a)', (0, 4.2), envs, x, legend=True)

    # COLREGs
    off_m = [np.mean(COLREGS[ek]['OFF']) for ek in env_keys]
    off_s = [np.std(COLREGS[ek]['OFF']) for ek in env_keys]
    on_m  = [np.mean(COLREGS[ek]['ON']) for ek in env_keys]
    on_s  = [np.std(COLREGS[ek]['ON']) for ek in env_keys]

    b1, b2 = _make_bars(ax2, x, off_m, off_s, on_m, on_s, w)
    _add_val(ax2, b1, off_m, '{:.1f}', '0.15', 1.0)
    _add_val(ax2, b2, on_m, '{:.1f}', '0.15', 1.0)
    _style_ax(ax2, 'COLREGs compliance (%)', '(b)', (0, 115), envs, x)

    fig.suptitle(
        f'Performance evaluation ({N_VESSELS} vessels, {N_RUNS} runs/env, averaged over {N_EVAL_STEPS} steps)',
        fontsize=10.5, y=1.02)
    fig.tight_layout(w_pad=2.5)

    for ext in ['png', 'pdf']:
        path = os.path.join(SAVE_DIR, f'performance_collision_colregs.{ext}')
        fig.savefig(path, facecolor='white')
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Performance Comparison - Journal Style ({N_VESSELS} vessels)")
    print("=" * 60)

    plot_main_performance()
    plot_collision_colregs_focused()

    print("\n=== Statistics ===")
    for env, ek in [('Open Sea', 'open'), ('Coastal', 'coastal'), ('Narrow Strait', 'narrow')]:
        c_off = np.mean(COLLISION[ek]['OFF'])
        c_on = np.mean(COLLISION[ek]['ON'])
        cr_off = np.mean(COLREGS[ek]['OFF'])
        cr_on = np.mean(COLREGS[ek]['ON'])
        s_off = np.mean(SUCCESS[ek]['OFF'])
        s_on = np.mean(SUCCESS[ek]['ON'])
        print(f"\n  [{env}]")
        print(f"    Collision:  OFF={c_off:.1f}  ON={c_on:.1f}  (reduction: {(1-c_on/c_off)*100:.0f}%)")
        print(f"    COLREGs:    OFF={cr_off:.1f}%  ON={cr_on:.1f}%  (+{cr_on-cr_off:.0f}%p)")
        print(f"    Success:    OFF={s_off:.1f}%  ON={s_on:.1f}%  (+{s_on-s_off:.0f}%p)")

    print("\nDone!")
