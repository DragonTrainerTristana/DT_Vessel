"""
Open Ocean — Communication Aggregation 비교
OFF vs sum-of-4 vs nearest-1 (학습은 sum-of-4, inference 시 partner 수만 변경)

데이터 소스:
- OFF, sum-of-4: experiment_data_20260318.md 기록
- nearest-1:    오늘 (2026-05-06) 측정값
"""
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from config import PROJECT_ROOT

# Open Ocean 결과 (10 runs × 2000 steps, dim=6 baseline)
COLLISIONS = {
    'OFF':         [2, 2, 3, 1, 3, 1, 0, 1, 1, 0],
    'sum-of-4':    [1, 2, 0, 0, 2, 1, 0, 0, 2, 0],
    'nearest-1×1': [5, 1, 5, 4, 7, 3, 4, 6, 1, 1],
    'nearest-1×2': [4, 2, 2, 0, 3, 6, 0, 2, 3, 1],
    'nearest-1×4': [5, 2, 4, 2, 2, 2, 4, 6, 0, 3],
}
SUCCESS = {
    'OFF':         [8, 10, 9, 12, 8, 5, 2, 8, 6, 8],
    'sum-of-4':    [7, 9, 5, 5, 11, 14, 6, 9, 9, 9],
    'nearest-1×1': [0]*10,
    'nearest-1×2': [0]*10,
    'nearest-1×4': [0]*10,
}
REWARD = {
    'OFF':         [4045.24, 4190.23, 4137.72, 4224.22, 4094.80, 4148.03, 4025.58, 4130.29, 4126.16, 4164.10],
    'sum-of-4':    [4330.64, 4267.53, 4371.89, 4376.03, 4284.36, 4389.31, 4276.69, 4529.15, 4223.86, 4346.44],
    'nearest-1×1': [4402.73, 4428.25, 4403.98, 4373.50, 4344.87, 4362.73, 4394.68, 4447.06, 4378.96, 4388.25],
    'nearest-1×2': [4435.73, 4415.70, 4430.40, 4425.50, 4398.64, 4450.52, 4340.44, 4450.61, 4450.19, 4376.49],
    'nearest-1×4': [4413.12, 4390.54, 4454.72, 4405.83, 4438.12, 4373.45, 4386.86, 4417.63, 4428.79, 4373.22],
}

MODES = ['OFF', 'sum-of-4', 'nearest-1×1', 'nearest-1×2', 'nearest-1×4']
COLORS = {
    'OFF':         '#E74C3C',
    'sum-of-4':    '#3498DB',
    'nearest-1×1': '#95E89F',
    'nearest-1×2': '#2ECC71',
    'nearest-1×4': '#1A8245',
}


def make_panel(ax, data_dict, ylabel, title, fmt='{:.2f}'):
    means = [np.mean(data_dict[m]) for m in MODES]
    stds = [np.std(data_dict[m], ddof=1) for m in MODES]
    colors = [COLORS[m] for m in MODES]
    bars = ax.bar(MODES, means, yerr=stds, color=colors, alpha=0.85,
                  capsize=8, edgecolor='black', linewidth=0.7)
    max_std = max(stds) if max(stds) > 0 else 0.1
    for bar, m_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_std*0.15,
                fmt.format(m_val), ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    make_panel(axes[0], COLLISIONS, 'Collisions per Run',
               'Collisions', fmt='{:.2f}')
    make_panel(axes[1], SUCCESS, 'Successes per Run',
               'Successes', fmt='{:.2f}')
    make_panel(axes[2], REWARD, 'Avg Reward',
               'Reward', fmt='{:.0f}')

    fig.suptitle('Open Ocean — Communication Aggregation Comparison\n'
                 'dim=6 baseline (Phase 2 v2 step 16.67M), inference-only mode change',
                 fontsize=13, y=1.02, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.join(PROJECT_ROOT, "figures", "comm_aggregation")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"open_ocean_aggregation_{ts}")
    fig.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}.png/pdf")

    # 요약 표
    print()
    print("=" * 80)
    print(f"{'Mode':<12}{'Collisions':>20}{'Success':>20}{'Reward':>22}")
    print("-" * 80)
    for m in MODES:
        c = np.array(COLLISIONS[m])
        s = np.array(SUCCESS[m])
        r = np.array(REWARD[m])
        print(f"{m:<12}{c.mean():>10.2f} ± {c.std(ddof=1):>5.2f}  "
              f"{s.mean():>10.2f} ± {s.std(ddof=1):>5.2f}  "
              f"{r.mean():>10.1f} ± {r.std(ddof=1):>5.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
