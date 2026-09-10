"""
Open Ocean — Communication Aggregation 전체 비교 (FINAL)
8개 mode 한 그래프에:
  msg-OFF zone: gain=0 (msg=0)
  comm zone:    sum-of-4, nn1 raw, nn1 ×4, mean-of-4, sum-of-4 ×0.5
  broken zone:  mean-field (sum-of-99 OOD)

핵심 finding:
  1. comm 활성화 시 magnitude/aggregation 무관하게 일관된 navigation (speed 0.905)
  2. msg=0이어도 speed 정상 — 모델이 학습된 안전 모드 진입
  3. mean-field만 catastrophic — 큰 magnitude가 thrust output 망가뜨림 (speed 0.65)
"""
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from config import PROJECT_ROOT

# 5x1000 진단 데이터
COLLISIONS = {
    'gain=0\n(msg=0)':       [6, 2, 4, 2, 0],
    'mean-field\n(sum-of-99)':[0, 2, 0, 2, 2],
    'sum-of-4':              [2, 0, 2, 2, 2],
    'sum-of-4\n×0.5':        [2, 4, 2, 6, 2],
    'mean-of-4':             [2, 6, 2, 4, 2],
    'nn1 raw':               [0, 0, 6, 0, 3],
    'nn1 ×4':                [2, 2, 3, 2, 6],
}
SUCCESS = {
    'gain=0\n(msg=0)':       [0, 0, 1, 0, 1],
    'mean-field\n(sum-of-99)':[0, 0, 0, 0, 0],
    'sum-of-4':              [0, 1, 0, 2, 1],
    'sum-of-4\n×0.5':        [1, 1, 0, 2, 0],
    'mean-of-4':             [3, 2, 1, 0, 4],
    'nn1 raw':               [2, 2, 2, 1, 2],
    'nn1 ×4':                [4, 1, 1, 1, 1],
}
REWARD = {
    'gain=0\n(msg=0)':       [5069.25, 5085.00, 5167.38, 5099.67, 5054.69],
    'mean-field\n(sum-of-99)':[2279.63, 2224.67, 2258.48, 2242.94, 2234.67],
    'sum-of-4':              [5072.96, 5079.83, 5134.67, 5123.70, 5086.47],
    'sum-of-4\n×0.5':        [5131.94, 5121.37, 5140.70, 5019.31, 5065.17],
    'mean-of-4':             [5085.70, 5062.23, 5073.40, 5064.10, 5189.43],
    'nn1 raw':               [5137.67, 5091.60, 5080.47, 5118.38, 5062.09],
    'nn1 ×4':                [5146.76, 5111.35, 5102.80, 5095.50, 5076.82],
}

# Trajectory speed (per-mode평균, world-map env)
SPEED = {
    'gain=0\n(msg=0)':        0.905,
    'mean-field\n(sum-of-99)': 0.652,
    'sum-of-4':               0.905,
    'sum-of-4\n×0.5':         0.905,
    'mean-of-4':              0.905,
    'nn1 raw':                0.905,
    'nn1 ×4':                 0.904,
}

MODES = ['gain=0\n(msg=0)', 'mean-field\n(sum-of-99)',
         'sum-of-4', 'sum-of-4\n×0.5', 'mean-of-4',
         'nn1 raw', 'nn1 ×4']

COLORS = {
    'gain=0\n(msg=0)':        '#7F8C8D',  # gray (OFF)
    'mean-field\n(sum-of-99)':'#E74C3C',  # red (broken)
    'sum-of-4':               '#3498DB',  # blue (training)
    'sum-of-4\n×0.5':         '#5DADE2',  # light blue
    'mean-of-4':              '#9B59B6',  # purple
    'nn1 raw':                '#2ECC71',  # green
    'nn1 ×4':                 '#1A8245',  # dark green
}


def make_panel(ax, data_dict, ylabel, title, fmt='{:.2f}', is_speed=False):
    if is_speed:
        means = [data_dict[m] for m in MODES]
        stds = [0] * len(MODES)
    else:
        means = [np.mean(data_dict[m]) for m in MODES]
        stds = [np.std(data_dict[m], ddof=1) for m in MODES]
    colors = [COLORS[m] for m in MODES]
    bars = ax.bar(MODES, means, yerr=stds, color=colors, alpha=0.85,
                  capsize=6, edgecolor='black', linewidth=0.7)
    max_std = max(stds) if max(stds) > 0 else max(means) * 0.05
    for bar, m_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_std*0.2,
                fmt.format(m_val), ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', labelsize=9)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    make_panel(axes[0], COLLISIONS, 'Collisions per Run',
               'Collisions', fmt='{:.2f}')
    make_panel(axes[1], SUCCESS, 'Successes per Run',
               'Successes', fmt='{:.2f}')
    make_panel(axes[2], REWARD, 'Avg Reward',
               'Reward', fmt='{:.0f}')
    make_panel(axes[3], SPEED, 'Mean Speed (norm)',
               'Speed (trajectory avg)', fmt='{:.3f}', is_speed=True)

    fig.suptitle('Open Ocean Aggregation Ablation (FIXED Infrastructure)\n'
                 '5 runs × 1000 steps, world-map env, dim=6 baseline. '
                 'mean-field (sum-of-99) ONLY catastrophic mode — '
                 'OOD magnitude breaks thrust output (speed ↓28%)',
                 fontsize=12, y=1.04, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.join(PROJECT_ROOT, "figures", "comm_aggregation")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"open_ocean_aggregation_full_{ts}")
    fig.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}.png/pdf")

    # Summary table
    print()
    print("=" * 100)
    print(f"{'Mode':<25}{'Collisions':>18}{'Success':>18}{'Reward':>20}{'Speed':>10}")
    print("-" * 100)
    for m in MODES:
        c = np.array(COLLISIONS[m])
        s = np.array(SUCCESS[m])
        r = np.array(REWARD[m])
        sp = SPEED[m]
        m_clean = m.replace('\n', ' ')
        print(f"{m_clean:<25}{c.mean():>9.2f} ± {c.std(ddof=1):>5.2f}  "
              f"{s.mean():>9.2f} ± {s.std(ddof=1):>5.2f}  "
              f"{r.mean():>10.1f} ± {r.std(ddof=1):>5.1f} {sp:>9.3f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
