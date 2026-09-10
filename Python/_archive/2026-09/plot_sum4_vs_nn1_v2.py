"""
sum-of-4 vs nearest-1 — 7 metrics 비교

Metrics:
  1. Collisions per run (from log summary)
  2. Successes per run (from log summary)
  3. COLREGs compliance rate (overall %)
  4. DCPA (avg min passing distance)
  5. Episode time (distance-normalized): active_steps / progress
  6. Fuel consumption (distance-normalized): Σspeed² / progress
  7. Vessels in radar range (200m) per step

데이터: 5 runs × 1000 steps, world-map env, dim=6 baseline (Phase 2 v2 16.67M)
"""
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import PROJECT_ROOT
from test import evaluate_colregs_compliance


TRAJ_DIR = os.path.join(PROJECT_ROOT, "trajectory_data")

MODES = [
    ('sum-of-6',  'v3_nn1_R200_5x1000_20260507_152320.csv',   '#3498DB'),
    ('nearest-1', 'v3_sum4_R200_5x1000_20260507_152203.csv',  '#2ECC71'),
]

COLLISIONS_RAW = {
    'sum-of-6':  [0, 0, 0, 0, 0],
    'nearest-1': [0, 1, 0, 1, 2],
}
SUCCESS_RAW = {
    'sum-of-6':  [1, 2, 4, 0, 2],
    'nearest-1': [3, 3, 2, 3, 0],
}

# 환경 상수 (small-scale, ±250m)
RADAR_RANGE = 200.0       # 미터
GD_THRESHOLD = 0.99       # active vessel 기준
MIN_PROGRESS = 0.05
MIN_STEPS = 5


def compute_efficiency(df):
    """per-vessel episode_time, fuel (distance-normalized)"""
    times, fuels = [], []
    for run_id in df['run_id'].unique():
        rdf = df[df['run_id'] == run_id]
        for aid in rdf['agent_id'].unique():
            v = rdf[rdf['agent_id'] == aid].sort_values('step')
            if len(v) < MIN_STEPS:
                continue
            active = v[v['goal_dist'] < GD_THRESHOLD]
            if len(active) < MIN_STEPS:
                continue
            progress = active['goal_dist'].iloc[0] - active['goal_dist'].iloc[-1]
            if progress < MIN_PROGRESS:
                continue
            times.append(len(active) / progress)
            fuels.append((active['speed'] ** 2).sum() / progress)
    return np.array(times), np.array(fuels)


def compute_radar_count(df, max_steps_sample=200):
    """per step 평균: 각 vessel의 radar(200m) 안에 들어오는 다른 vessel 수"""
    counts = []
    for run_id in df['run_id'].unique():
        rdf = df[df['run_id'] == run_id]
        steps = sorted(rdf['step'].unique())[::5]  # subsample for speed
        for step in steps[:max_steps_sample]:
            s = rdf[rdf['step'] == step]
            if len(s) < 2:
                continue
            positions = s[['x', 'z']].values
            diff = positions[:, None] - positions[None, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dist, np.inf)
            in_range = (dist < RADAR_RANGE).sum(axis=1)
            counts.extend(in_range.tolist())
    return np.array(counts)


def make_panel(ax, mode_means, mode_stds, mode_labels, mode_colors,
               ylabel, title, fmt='{:.2f}'):
    bars = ax.bar(mode_labels, mode_means, color=mode_colors,
                  alpha=0.85, edgecolor='black', linewidth=0.8, width=0.55)
    max_h = max(mode_means) if max(mode_means) > 0 else 1
    for bar, val in zip(bars, mode_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_h * 0.02,
                fmt.format(val), ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', labelsize=11)


def main():
    metrics = {}
    for label, fname, color in MODES:
        path = os.path.join(TRAJ_DIR, fname)
        print(f"\n[{label}] Loading {fname}")
        df = pd.read_csv(path)
        print(f"  rows={len(df):,}")

        # 1. Episode time / Fuel
        times, fuels = compute_efficiency(df)
        print(f"  active vessels={len(times)}, time={times.mean():.0f}, fuel={fuels.mean():.0f}")

        # 2. Radar count
        radar_n = compute_radar_count(df)
        print(f"  radar count (200m): mean={radar_n.mean():.2f}, n samples={len(radar_n)}")

        # 3. COLREGs + DCPA (use existing evaluate)
        print(f"  Running evaluate_colregs_compliance...")
        colregs_result = evaluate_colregs_compliance(path)
        if colregs_result:
            comp_rate = colregs_result['overall_compliance_rate'] * 100
            dcpa_avg = colregs_result['dcpa']['avg_min_distance']
            print(f"  COLREGs compliance: {comp_rate:.1f}%, DCPA avg: {dcpa_avg:.1f}m")
        else:
            comp_rate = 0
            dcpa_avg = 0

        metrics[label] = {
            'times': times, 'fuels': fuels, 'radar_n': radar_n,
            'colregs_rate': comp_rate, 'dcpa_avg': dcpa_avg,
            'color': color,
        }

    # Plot 7 panels (2 rows x 4 cols, last is empty)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    labels = [l for l, _, _ in MODES]
    colors = [c for _, _, c in MODES]

    # 1. Collisions
    coll_m = [np.mean(COLLISIONS_RAW[l]) for l in labels]
    coll_s = [np.std(COLLISIONS_RAW[l], ddof=1) for l in labels]
    make_panel(axes[0], coll_m, coll_s, labels, colors,
               'Collisions per Run', 'Collisions', fmt='{:.2f}')

    # 2. Successes
    succ_m = [np.mean(SUCCESS_RAW[l]) for l in labels]
    succ_s = [np.std(SUCCESS_RAW[l], ddof=1) for l in labels]
    make_panel(axes[1], succ_m, succ_s, labels, colors,
               'Successes per Run', 'Successes', fmt='{:.2f}')

    # 3. COLREGs compliance
    colregs_m = [metrics[l]['colregs_rate'] for l in labels]
    colregs_s = [0] * len(labels)
    make_panel(axes[2], colregs_m, colregs_s, labels, colors,
               'Compliance Rate (%)', 'COLREGs Compliance', fmt='{:.1f}')

    # 4. DCPA
    dcpa_m = [metrics[l]['dcpa_avg'] for l in labels]
    dcpa_s = [0] * len(labels)
    make_panel(axes[3], dcpa_m, dcpa_s, labels, colors,
               'Avg Min Passing Dist (m)', 'DCPA', fmt='{:.1f}')

    # 5. Episode time (distance-norm)
    time_m = [metrics[l]['times'].mean() for l in labels]
    time_s = [metrics[l]['times'].std() for l in labels]
    make_panel(axes[4], time_m, time_s, labels, colors,
               'Steps / unit goal-dist progress',
               'Episode Time (distance-norm)', fmt='{:.0f}')

    # 6. Fuel consumption (distance-norm)
    fuel_m = [metrics[l]['fuels'].mean() for l in labels]
    fuel_s = [metrics[l]['fuels'].std() for l in labels]
    make_panel(axes[5], fuel_m, fuel_s, labels, colors,
               'Σspeed² / unit goal-dist progress',
               'Fuel Consumption (distance-norm)', fmt='{:.0f}')

    # 7. Radar count
    rad_m = [metrics[l]['radar_n'].mean() for l in labels]
    rad_s = [metrics[l]['radar_n'].std() for l in labels]
    make_panel(axes[6], rad_m, rad_s, labels, colors,
               'Vessels within 200m radar range',
               'Vessels in Radar Range', fmt='{:.2f}')

    # 8. Empty
    axes[7].axis('off')

    fig.suptitle('Sum-of-6 vs Nearest-1 — Open Ocean (5 runs × 1000 steps)\n'
                 'dim=6 baseline (Phase 2 v2 16.67M)',
                 fontsize=14, y=1.02, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.join(PROJECT_ROOT, "figures", "comm_aggregation")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"sum4_vs_nn1_v2_{ts}")
    fig.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SAVED] {out_path}.png/pdf")

    # Summary table
    print()
    print("=" * 100)
    print(f"{'Metric':<35}{'sum-of-6':>30}{'nearest-1':>30}")
    print("-" * 100)
    print(f"{'Collisions per run':<35}"
          f"{f'{coll_m[0]:.2f} ± {coll_s[0]:.2f}':>30}"
          f"{f'{coll_m[1]:.2f} ± {coll_s[1]:.2f}':>30}")
    print(f"{'Successes per run':<35}"
          f"{f'{succ_m[0]:.2f} ± {succ_s[0]:.2f}':>30}"
          f"{f'{succ_m[1]:.2f} ± {succ_s[1]:.2f}':>30}")
    print(f"{'COLREGs compliance (%)':<35}"
          f"{f'{colregs_m[0]:.1f}':>30}"
          f"{f'{colregs_m[1]:.1f}':>30}")
    print(f"{'DCPA avg (m)':<35}"
          f"{f'{dcpa_m[0]:.1f}':>30}"
          f"{f'{dcpa_m[1]:.1f}':>30}")
    print(f"{'Episode time (steps/dist)':<35}"
          f"{f'{time_m[0]:.0f} ± {time_s[0]:.0f}':>30}"
          f"{f'{time_m[1]:.0f} ± {time_s[1]:.0f}':>30}")
    print(f"{'Fuel (Σsp²/dist)':<35}"
          f"{f'{fuel_m[0]:.0f} ± {fuel_s[0]:.0f}':>30}"
          f"{f'{fuel_m[1]:.0f} ± {fuel_s[1]:.0f}':>30}")
    print(f"{'Vessels in radar range (200m)':<35}"
          f"{f'{rad_m[0]:.2f} ± {rad_s[0]:.2f}':>30}"
          f"{f'{rad_m[1]:.2f} ± {rad_s[1]:.2f}':>30}")
    print("=" * 100)


if __name__ == "__main__":
    main()
