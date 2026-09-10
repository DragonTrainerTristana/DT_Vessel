"""
Aggregation Diagnostics 분석 — Trajectory + 통계
inputs: trajectory_data/diag_*_5x1000_*.csv

분석:
1. goal_dist 시계열 — 목표에 접근 중인지, 회전 중인지 판정
2. position 궤적 — 직선/원/지그재그 구분
3. speed 분포 — 정지/저속/정상 구분
4. heading vs goal_angle 관계 — 목표를 향하고 있는지

각 mode별 multi-panel figure + 비교 summary
"""
import os
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import PROJECT_ROOT

TRAJ_DIR = os.path.join(PROJECT_ROOT, "trajectory_data")

MODES = {
    'sum-of-4':   'diag_sum4',
    'nn1 raw':    'diag_nn1_raw',
    'nn1 ×4':     'diag_nn1_x4',
    'mean-of-4':  'diag_mean4',
    'sum-of-4 ×0.5': 'diag_sum4_gain05',
}
COLORS = {
    'OFF (ref)':      '#E74C3C',
    'sum-of-4':       '#3498DB',
    'nn1 raw':        '#95E89F',
    'nn1 ×4':         '#1A8245',
    'mean-of-4':      '#9B59B6',
    'sum-of-4 ×0.5':  '#F39C12',
}

OFF_REF_CSV = 'open_ocean_compare_commOFF_10x2000_20260319_015000.csv'


def find_latest(tag):
    # Exact tag match: tag_NxM_TIMESTAMP.csv (NxM은 숫자만, gain05 등 별도 tag 분리)
    pat = os.path.join(TRAJ_DIR, f"{tag}_*x*_*.csv")
    files = sorted(glob.glob(pat))
    # Filter: tag 다음에 _숫자x숫자만 와야 함 (diag_sum4_5x1000은 OK, diag_sum4_gain05는 제외)
    import re
    rx = re.compile(rf"^{re.escape(tag)}_\d+x\d+_\d+_\d+\.csv$")
    files = [f for f in files if rx.match(os.path.basename(f))]
    return files[-1] if files else None


def load_trajectory(tag):
    path = find_latest(tag)
    if path is None:
        print(f"[MISS] {tag}: no csv found")
        return None
    df = pd.read_csv(path)
    print(f"[OK] {tag}: {len(df)} rows from {os.path.basename(path)}")
    return df


def plot_goal_dist_progression(traj_dict, save_path):
    """각 mode별 평균 goal_dist over step (run/agent 평균)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode, df in traj_dict.items():
        if df is None:
            continue
        agg = df.groupby('step')['goal_dist'].agg(['mean', 'std']).reset_index()
        ax.plot(agg['step'], agg['mean'], label=mode, color=COLORS[mode], lw=2)
        ax.fill_between(agg['step'], agg['mean'] - agg['std']*0.3,
                        agg['mean'] + agg['std']*0.3, color=COLORS[mode], alpha=0.15)
    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel('Goal Distance (normalized)', fontweight='bold')
    ax.set_title('Goal Distance Progression — does navigation work?', fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {save_path}")


def plot_position_scatter(traj_dict, save_path):
    """각 mode별 vessel 궤적 (run 0, 첫 5개 vessel)"""
    n_modes = len(traj_dict)
    fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 5))
    if n_modes == 1:
        axes = [axes]
    for ax, (mode, df) in zip(axes, traj_dict.items()):
        if df is None:
            ax.set_title(f'{mode}\n(no data)')
            continue
        run0 = df[df.get('run_id', 0) == 0] if 'run_id' in df.columns else df
        for aid in run0['agent_id'].unique()[:8]:
            agent = run0[run0['agent_id'] == aid].sort_values('step')
            ax.plot(agent['x'], agent['z'], lw=1.0, alpha=0.7)
            if len(agent) > 0:
                ax.plot(agent['x'].iloc[0], agent['z'].iloc[0], 'go', ms=4)
                ax.plot(agent['x'].iloc[-1], agent['z'].iloc[-1], 'rx', ms=6)
        ax.set_title(f'{mode}', fontweight='bold', color=COLORS[mode])
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {save_path}")


def plot_speed_heading(traj_dict, save_path):
    """speed 분포 + heading-vs-goal_angle 정합도"""
    n_modes = len(traj_dict)
    fig, axes = plt.subplots(2, n_modes, figsize=(4 * n_modes, 8))
    if n_modes == 1:
        axes = axes.reshape(2, 1)

    for col, (mode, df) in enumerate(traj_dict.items()):
        if df is None:
            continue
        ax_s = axes[0, col]
        ax_h = axes[1, col]

        ax_s.hist(df['speed'].values, bins=40, color=COLORS[mode], alpha=0.8, edgecolor='black')
        ax_s.set_title(f'{mode}: speed dist', fontweight='bold', color=COLORS[mode])
        ax_s.set_xlabel('Speed (norm)')
        ax_s.axvline(df['speed'].mean(), color='red', ls='--', label=f"mean={df['speed'].mean():.2f}")
        ax_s.legend()

        ax_h.scatter(df['goal_angle'], df['rudder'], s=2, alpha=0.3, color=COLORS[mode])
        ax_h.set_title('rudder vs goal_angle', fontweight='bold')
        ax_h.set_xlabel('Goal angle (norm)')
        ax_h.set_ylabel('Rudder (norm)')
        ax_h.axhline(0, color='gray', alpha=0.3)
        ax_h.axvline(0, color='gray', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {save_path}")


def summary_stats(traj_dict):
    print("\n" + "=" * 90)
    print(f"{'Mode':<18}{'mean speed':>12}{'mean goal_dist':>16}{'goal_dist Δ':>14}{'|rudder|':>10}")
    print("-" * 90)
    for mode, df in traj_dict.items():
        if df is None:
            print(f"{mode:<18}  (no data)")
            continue
        sp = df['speed'].mean()
        gd_mean = df['goal_dist'].mean()
        # goal_dist Δ: 첫 100 step 평균 - 마지막 100 step 평균
        max_step = df['step'].max()
        early = df[df['step'] < 100]['goal_dist'].mean()
        late = df[df['step'] > max_step - 100]['goal_dist'].mean()
        delta = late - early
        ru = df['rudder'].abs().mean()
        print(f"{mode:<18}{sp:>12.3f}{gd_mean:>16.3f}{delta:>+14.3f}{ru:>10.3f}")
    print("=" * 90)


def main():
    out_dir = os.path.join(PROJECT_ROOT, "figures", "comm_aggregation")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    traj_dict = {}
    # Load OFF reference (existing CSV from March)
    off_path = os.path.join(TRAJ_DIR, OFF_REF_CSV)
    if os.path.exists(off_path):
        off_df = pd.read_csv(off_path)
        # OFF reference doesn't have run_id; add zero
        if 'run_id' not in off_df.columns:
            off_df['run_id'] = 0
        traj_dict['OFF (ref)'] = off_df
        print(f"[OK] OFF (ref): {len(off_df)} rows from {OFF_REF_CSV}")
    for mode, tag in MODES.items():
        traj_dict[mode] = load_trajectory(tag)

    summary_stats(traj_dict)

    plot_goal_dist_progression(traj_dict,
        os.path.join(out_dir, f"diag_goal_dist_{ts}.png"))
    plot_position_scatter(traj_dict,
        os.path.join(out_dir, f"diag_trajectory_{ts}.png"))
    plot_speed_heading(traj_dict,
        os.path.join(out_dir, f"diag_speed_heading_{ts}.png"))

    print("\n[DONE]")


if __name__ == "__main__":
    main()
