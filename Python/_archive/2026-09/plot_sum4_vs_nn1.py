"""
sum-of-4 vs nearest-1 — 2 mode 비교 (4 metrics)

Metrics:
  1. Collisions per run
  2. Successes per run
  3. Episode time (distance-normalized): active_steps / goal_dist_progress
  4. Fuel consumption (distance-normalized): Σspeed² / goal_dist_progress

데이터: 5 runs × 1000 steps, world-map env, dim=6 baseline (Phase 2 v2 16.67M)
"""
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import PROJECT_ROOT


TRAJ_DIR = os.path.join(PROJECT_ROOT, "trajectory_data")

MODE_FILES = [
    ('sum-of-4',  'diag_sum4_5x1000_20260506_150622.csv',     '#3498DB'),
    ('nearest-1', 'diag_nn1_raw_5x1000_20260506_151128.csv',  '#2ECC71'),
]

COLLISIONS_RAW = {
    'sum-of-4':  [2, 0, 2, 2, 2],
    'nearest-1': [0, 0, 6, 0, 3],
}
SUCCESS_RAW = {
    'sum-of-4':  [0, 1, 0, 2, 1],
    'nearest-1': [2, 2, 2, 1, 2],
}

GD_THRESHOLD = 0.99
MIN_PROGRESS = 0.05
MIN_STEPS = 5


def compute_efficiency(df):
    times, fuels = [], []
    runs = df['run_id'].unique() if 'run_id' in df.columns else [0]
    for run_id in runs:
        rdf = df[df['run_id'] == run_id] if 'run_id' in df.columns else df
        for aid in rdf['agent_id'].unique():
            v = rdf[rdf['agent_id'] == aid].sort_values('step')
            if len(v) < MIN_STEPS:
                continue
            active = v[v['goal_dist'] < GD_THRESHOLD]
            if len(active) < MIN_STEPS:
                continue
            start_gd = active['goal_dist'].iloc[0]
            end_gd = active['goal_dist'].iloc[-1]
            progress = start_gd - end_gd
            if progress < MIN_PROGRESS:
                continue
            n_steps = len(active)
            fuel = (active['speed'] ** 2).sum()
            times.append(n_steps / progress)
            fuels.append(fuel / progress)
    return np.array(times), np.array(fuels)


def make_panel(ax, mode_means, mode_stds, mode_labels, mode_colors,
               ylabel, title, fmt='{:.2f}'):
    bars = ax.bar(mode_labels, mode_means, yerr=mode_stds, color=mode_colors,
                  alpha=0.85, capsize=10, edgecolor='black', linewidth=0.8, width=0.55)
    max_h = max(mode_means) if max(mode_means) > 0 else 1
    for bar, val in zip(bars, mode_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_h * 0.02,
                fmt.format(val), ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', labelsize=12)


def main():
    eff = {}
    for label, fname, color in MODE_FILES:
        df = pd.read_csv(os.path.join(TRAJ_DIR, fname))
        times, fuels = compute_efficiency(df)
        eff[label] = (times, fuels, color)
        print(f"[{label}] {len(times)} active vessels  "
              f"time {times.mean():.1f} ± {times.std():.1f}  "
              f"fuel {fuels.mean():.1f} ± {fuels.std():.1f}")

    fig, axes = plt.subplots(1, 4, figsize=(15, 5.5))
    labels = [l for l, _, _ in MODE_FILES]
    colors = [c for _, _, c in MODE_FILES]

    coll_m = [np.mean(COLLISIONS_RAW[l]) for l in labels]
    coll_s = [np.std(COLLISIONS_RAW[l], ddof=1) for l in labels]
    make_panel(axes[0], coll_m, coll_s, labels, colors,
               'Collisions per Run', 'Collisions', fmt='{:.2f}')

    succ_m = [np.mean(SUCCESS_RAW[l]) for l in labels]
    succ_s = [np.std(SUCCESS_RAW[l], ddof=1) for l in labels]
    make_panel(axes[1], succ_m, succ_s, labels, colors,
               'Successes per Run', 'Successes', fmt='{:.2f}')

    time_m = [eff[l][0].mean() for l in labels]
    time_s = [eff[l][0].std() for l in labels]
    make_panel(axes[2], time_m, time_s, labels, colors,
               'Steps / unit goal-dist progress',
               'Episode Time\n(distance-normalized)', fmt='{:.0f}')

    fuel_m = [eff[l][1].mean() for l in labels]
    fuel_s = [eff[l][1].std() for l in labels]
    make_panel(axes[3], fuel_m, fuel_s, labels, colors,
               'Σspeed² / unit goal-dist progress',
               'Fuel Consumption\n(distance-normalized)', fmt='{:.0f}')

    fig.suptitle('Sum-of-4 vs Nearest-1 — Open Ocean (5 runs × 1000 steps, world-map env)',
                 fontsize=14, y=1.02, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.join(PROJECT_ROOT, "figures", "comm_aggregation")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"sum4_vs_nn1_{ts}")
    fig.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SAVED] {out_path}.png/pdf")

    print()
    print("=" * 90)
    print(f"{'Mode':<14}{'Collisions':>16}{'Successes':>16}"
          f"{'Time (steps/d)':>20}{'Fuel (sp²/d)':>20}")
    print("-" * 90)
    for l in labels:
        c = np.array(COLLISIONS_RAW[l])
        s = np.array(SUCCESS_RAW[l])
        t = eff[l][0]
        f = eff[l][1]
        print(f"{l:<14}"
              f"{c.mean():>7.2f}±{c.std(ddof=1):>5.2f}  "
              f"{s.mean():>7.2f}±{s.std(ddof=1):>5.2f}  "
              f"{t.mean():>9.0f}±{t.std():>7.0f}  "
              f"{f.mean():>9.0f}±{f.std():>7.0f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
