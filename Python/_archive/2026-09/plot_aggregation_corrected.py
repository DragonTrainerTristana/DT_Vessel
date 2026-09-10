"""
Open Ocean — Communication Aggregation 비교 (CORRECTED)
이전 plot_aggregation_open_ocean.py는 use_comm=False 버그로 잘못된 결과 보고.
수정된 인프라로 측정한 진짜 결과:

데이터: 5 runs × 1000 steps (world-map env, 100 vessels per step)
- sum-of-4: 학습 분포 (training match)
- nn1 raw: nearest 1, no scaling
- nn1 ×4: nearest 1 × 4 (magnitude scale)
- mean-of-4: 4명 평균 (magnitude /4)
- sum-of-4 ×0.5: 학습 분포 절반 magnitude (gating)

핵심 finding: aggregation 방법 차이는 navigation 행동에 거의 영향 X.
"""
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from config import PROJECT_ROOT

# 5x1000 step 진단 결과 (run-level: 5개 run의 raw 값)
COLLISIONS = {
    'sum-of-4':       [2, 0, 2, 2, 2],
    'nn1 raw':        [0, 0, 6, 0, 3],
    'nn1 ×4':         [2, 2, 3, 2, 6],
    'mean-of-4':      [2, 6, 2, 4, 2],
    'sum-of-4 ×0.5':  [2, 4, 2, 6, 2],
}
SUCCESS = {
    'sum-of-4':       [0, 1, 0, 2, 1],
    'nn1 raw':        [2, 2, 2, 1, 2],
    'nn1 ×4':         [4, 1, 1, 1, 1],
    'mean-of-4':      [3, 2, 1, 0, 4],
    'sum-of-4 ×0.5':  [1, 1, 0, 2, 0],
}
REWARD = {
    'sum-of-4':       [5072.96, 5079.83, 5134.67, 5123.70, 5086.47],
    'nn1 raw':        [5137.67, 5091.60, 5080.47, 5118.38, 5062.09],
    'nn1 ×4':         [5146.76, 5111.35, 5102.80, 5095.50, 5076.82],
    'mean-of-4':      [5085.70, 5062.23, 5073.40, 5064.10, 5189.43],
    'sum-of-4 ×0.5':  [5131.94, 5121.37, 5140.70, 5019.31, 5065.17],
}

MODES = ['sum-of-4', 'nn1 raw', 'nn1 ×4', 'mean-of-4', 'sum-of-4 ×0.5']
COLORS = {
    'sum-of-4':       '#3498DB',
    'nn1 raw':        '#2ECC71',
    'nn1 ×4':         '#1A8245',
    'mean-of-4':      '#9B59B6',
    'sum-of-4 ×0.5':  '#F39C12',
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
    ax.tick_params(axis='x', rotation=15)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    make_panel(axes[0], COLLISIONS, 'Collisions per Run',
               'Collisions', fmt='{:.2f}')
    make_panel(axes[1], SUCCESS, 'Successes per Run',
               'Successes', fmt='{:.2f}')
    make_panel(axes[2], REWARD, 'Avg Reward',
               'Reward', fmt='{:.0f}')

    fig.suptitle('Open Ocean — Communication Aggregation (CORRECTED)\n'
                 '5 runs × 1000 steps, world-map env, dim=6 baseline (Phase 2 v2 step 16.67M), '
                 'inference-only mode change',
                 fontsize=12, y=1.02, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.join(PROJECT_ROOT, "figures", "comm_aggregation")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"open_ocean_aggregation_corrected_{ts}")
    fig.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}.png/pdf")

    # 요약 표
    print()
    print("=" * 90)
    print(f"{'Mode':<18}{'Collisions':>20}{'Success':>20}{'Reward':>22}")
    print("-" * 90)
    for m in MODES:
        c = np.array(COLLISIONS[m])
        s = np.array(SUCCESS[m])
        r = np.array(REWARD[m])
        print(f"{m:<18}{c.mean():>10.2f} ± {c.std(ddof=1):>5.2f}  "
              f"{s.mean():>10.2f} ± {s.std(ddof=1):>5.2f}  "
              f"{r.mean():>10.1f} ± {r.std(ddof=1):>5.1f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
