"""
Aggregation Ablation — Efficiency metrics
Reward 빼고: Collisions, Successes, Episode Time (distance-norm), Fuel (distance-norm)

Episode time:
    per vessel: active_steps / goal_dist_progress (lower = faster navigation)
    "active" = goal_dist < 0.99 (within maxMapDistance)

Fuel consumption:
    per vessel: Σ(speed²) / goal_dist_progress (lower = more efficient)
    speed² 는 drag energy proxy

데이터: 7 mode × 5 runs × 1000 steps (world-map env)
"""
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import PROJECT_ROOT


TRAJ_DIR = os.path.join(PROJECT_ROOT, "trajectory_data")

MODE_FILES = [
    ('gain=0\n(msg=0)',        'diag_gain0_5x1000_20260506_153915.csv',        '#7F8C8D'),
    ('mean-field\n(sum-of-99)','diag_meanfield_5x1000_20260506_154431.csv',    '#E74C3C'),
    ('sum-of-4',               'diag_sum4_5x1000_20260506_150622.csv',         '#3498DB'),
    ('sum-of-4\n×0.5',         'diag_sum4_gain05_5x1000_20260506_152707.csv',  '#5DADE2'),
    ('mean-of-4',              'diag_mean4_5x1000_20260506_152153.csv',        '#9B59B6'),
    ('nn1 raw',                'diag_nn1_raw_5x1000_20260506_151128.csv',      '#2ECC71'),
    ('nn1 ×4',                 'diag_nn1_x4_5x1000_20260506_151639.csv',       '#1A8245'),
]

# 5 run × 1000 step 진단 결과 (run-level: 5개 raw 값)
COLLISIONS_RAW = {
    'gain=0\n(msg=0)':        [6, 2, 4, 2, 0],
    'mean-field\n(sum-of-99)':[0, 2, 0, 2, 2],
    'sum-of-4':               [2, 0, 2, 2, 2],
    'sum-of-4\n×0.5':         [2, 4, 2, 6, 2],
    'mean-of-4':              [2, 6, 2, 4, 2],
    'nn1 raw':                [0, 0, 6, 0, 3],
    'nn1 ×4':                 [2, 2, 3, 2, 6],
}
SUCCESS_RAW = {
    'gain=0\n(msg=0)':        [0, 0, 1, 0, 1],
    'mean-field\n(sum-of-99)':[0, 0, 0, 0, 0],
    'sum-of-4':               [0, 1, 0, 2, 1],
    'sum-of-4\n×0.5':         [1, 1, 0, 2, 0],
    'mean-of-4':              [3, 2, 1, 0, 4],
    'nn1 raw':                [2, 2, 2, 1, 2],
    'nn1 ×4':                 [4, 1, 1, 1, 1],
}

GD_THRESHOLD = 0.99       # goal_dist saturation (벽 너머 vessel)
MIN_PROGRESS = 0.05       # 최소 진척도 (이하면 censored)
MIN_STEPS = 5             # 최소 active steps


def compute_efficiency(df):
    """trajectory df → per-vessel episode_time_norm, fuel_norm"""
    times, fuels = [], []
    runs = df['run_id'].unique() if 'run_id' in df.columns else [0]
    for run_id in runs:
        rdf = df[df['run_id'] == run_id] if 'run_id' in df.columns else df
        for aid in rdf['agent_id'].unique():
            v = rdf[rdf['agent_id'] == aid].sort_values('step')
            if len(v) < MIN_STEPS:
                continue
            # active: goal_dist saturated 미만 구간만
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
                  alpha=0.85, capsize=6, edgecolor='black', linewidth=0.7)
    max_h = max(mode_means) if max(mode_means) > 0 else 1
    for bar, val in zip(bars, mode_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_h * 0.02,
                fmt.format(val), ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', labelsize=9)


def main():
    # 효율성 metric 계산
    eff_results = {}
    for label, fname, color in MODE_FILES:
        path = os.path.join(TRAJ_DIR, fname)
        if not os.path.exists(path):
            print(f"[MISS] {label}: {fname}")
            eff_results[label] = (np.array([]), np.array([]), color)
            continue
        df = pd.read_csv(path)
        times, fuels = compute_efficiency(df)
        eff_results[label] = (times, fuels, color)
        print(f"[OK] {label}: {len(times)} valid vessels  "
              f"time {times.mean():.1f} ± {times.std():.1f}  "
              f"fuel {fuels.mean():.2f} ± {fuels.std():.2f}")

    # Plot 4 panels
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.8))

    labels = [l for l, _, _ in MODE_FILES]
    colors = [c for _, _, c in MODE_FILES]

    # 1. Collisions
    coll_means = [np.mean(COLLISIONS_RAW[l]) for l in labels]
    coll_stds = [np.std(COLLISIONS_RAW[l], ddof=1) for l in labels]
    make_panel(axes[0], coll_means, coll_stds, labels, colors,
               'Collisions per Run', 'Collisions', fmt='{:.2f}')

    # 2. Successes
    succ_means = [np.mean(SUCCESS_RAW[l]) for l in labels]
    succ_stds = [np.std(SUCCESS_RAW[l], ddof=1) for l in labels]
    make_panel(axes[1], succ_means, succ_stds, labels, colors,
               'Successes per Run', 'Successes', fmt='{:.2f}')

    # 3. Episode time (distance-norm)
    time_means = [eff_results[l][0].mean() if len(eff_results[l][0]) > 0 else 0 for l in labels]
    time_stds = [eff_results[l][0].std() if len(eff_results[l][0]) > 0 else 0 for l in labels]
    make_panel(axes[2], time_means, time_stds, labels, colors,
               'Steps / unit goal-dist progress',
               'Episode Time (distance-norm)\nlower = faster', fmt='{:.0f}')

    # 4. Fuel consumption (distance-norm)
    fuel_means = [eff_results[l][1].mean() if len(eff_results[l][1]) > 0 else 0 for l in labels]
    fuel_stds = [eff_results[l][1].std() if len(eff_results[l][1]) > 0 else 0 for l in labels]
    make_panel(axes[3], fuel_means, fuel_stds, labels, colors,
               'Σspeed² / unit goal-dist progress',
               'Fuel Consumption (distance-norm)\nlower = more efficient', fmt='{:.0f}')

    fig.suptitle('Open Ocean Aggregation Ablation — Efficiency metrics\n'
                 '5 runs × 1000 steps, world-map env, dim=6 baseline (Phase 2 v2 16.67M)\n'
                 'Episode time and fuel are per-vessel, normalized by goal_dist progress made',
                 fontsize=12, y=1.04, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.join(PROJECT_ROOT, "figures", "comm_aggregation")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"open_ocean_aggregation_efficiency_{ts}")
    fig.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SAVED] {out_path}.png/pdf")

    # Summary table
    print()
    print("=" * 110)
    print(f"{'Mode':<25}{'Collision':>15}{'Success':>15}"
          f"{'Time (steps/dist)':>22}{'Fuel (sp²/dist)':>22}")
    print("-" * 110)
    for label in labels:
        c = np.array(COLLISIONS_RAW[label])
        s = np.array(SUCCESS_RAW[label])
        t = eff_results[label][0]
        f = eff_results[label][1]
        m_clean = label.replace('\n', ' ')
        print(f"{m_clean:<25}"
              f"{c.mean():>7.2f}±{c.std(ddof=1):>5.2f}  "
              f"{s.mean():>7.2f}±{s.std(ddof=1):>5.2f}  "
              f"{t.mean():>10.1f}±{t.std():>8.1f}  "
              f"{f.mean():>10.2f}±{f.std():>8.2f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
