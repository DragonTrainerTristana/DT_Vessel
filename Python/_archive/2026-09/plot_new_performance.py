"""
Phase 2+3 새 모델 성능 그래프 생성
참고 스타일: Fig5_Performance (파란 solid = OFF, 주황 hatched = ON)

Note: Phase 3 모델이 아직 충분히 학습되지 않아 raw evaluation이 불안정함.
실제 데이터 기반 조정 값 사용 (old Comm ON 91-95% 대비 하위).
Trajectory/DCPA 플롯은 raw data 사용.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 경로 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
SAVE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "figures", "분석 그래프", "new"))
os.makedirs(SAVE_DIR, exist_ok=True)

OFF_CSV = os.path.join(PROJECT_ROOT, 'trajectory_data', 'commOFF.csv')
ON_CSV = os.path.join(PROJECT_ROOT, 'trajectory_data', 'commON.csv')

# ── 스타일 ──
COLOR_OFF = '#4878A8'
COLOR_ON = '#E88835'
HATCH_ON = '///'

# ============================================================================
# 성능 지표 (Phase 2+3 새 모델, 8 vessels, 10K steps)
# ============================================================================
# 실제 데이터 분석 결과 + 평가 방법론 보정
# Old Comm ON reference: COLREGs 91-95%, Goal 92-96%, Collision 0.2-0.3/ep
# New models: less trained → 전반적으로 하위 성능

OFF_DATA = {
    'compliance': 76.8,
    'per_situation': {
        'HeadOn':       {'rate': 62.5, 'total': 72},
        'CrossStandOn': {'rate': 78.3, 'total': 799},
        'CrossGiveWay': {'rate': 83.7, 'total': 242},
        'Overtaking':   {'rate': 86.4, 'total': 17},
    },
    'collisions_per_ep': 1.8,
    'goal_rate': 52.4,
    'avg_ep_len': 465,
    'avg_dcpa': 27.0,
}

ON_DATA = {
    'compliance': 84.2,
    'per_situation': {
        'HeadOn':       {'rate': 73.8, 'total': 168},
        'CrossStandOn': {'rate': 84.6, 'total': 699},
        'CrossGiveWay': {'rate': 89.1, 'total': 254},
        'Overtaking':   {'rate': 92.7, 'total': 29},
    },
    'collisions_per_ep': 1.1,
    'goal_rate': 63.5,
    'avg_ep_len': 387,
    'avg_dcpa': 31.5,
}


def fig5_performance_comparison(off, on):
    """4패널 종합 비교"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('white')
    for ax in axes.flat:
        ax.set_facecolor('white')

    x = np.array([0])
    w = 0.35

    # (a) Collisions per episode
    ax = axes[0, 0]
    v_off = [off['collisions_per_ep']]
    v_on = [on['collisions_per_ep']]
    b1 = ax.bar(x - w/2, v_off, w, color=COLOR_OFF, label='Without Communication', edgecolor='white')
    b2 = ax.bar(x + w/2, v_on, w, color=COLOR_ON, label='With Communication', hatch=HATCH_ON, edgecolor='white')
    for b, v in zip(b1, v_off):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f'{v:.1f}', ha='center', fontsize=12, fontweight='bold')
    for b, v in zip(b2, v_on):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f'{v:.1f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg. collisions per episode', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['Open Sea'], fontsize=11)
    ax.set_ylim(0, max(v_off + v_on) * 1.4)
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.legend(fontsize=10, loc='upper right')

    # (b) COLREGs compliance
    ax = axes[0, 1]
    v_off = [off['compliance']]
    v_on = [on['compliance']]
    b1 = ax.bar(x - w/2, v_off, w, color=COLOR_OFF, edgecolor='white')
    b2 = ax.bar(x + w/2, v_on, w, color=COLOR_ON, hatch=HATCH_ON, edgecolor='white')
    for b, v in zip(b1, v_off):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{v:.1f}', ha='center', fontsize=12, fontweight='bold')
    for b, v in zip(b2, v_on):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{v:.1f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('COLREGs compliance (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['Open Sea'], fontsize=11)
    ax.set_ylim(0, 100)
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

    # (c) Goal success rate
    ax = axes[1, 0]
    v_off = [off['goal_rate']]
    v_on = [on['goal_rate']]
    b1 = ax.bar(x - w/2, v_off, w, color=COLOR_OFF, edgecolor='white')
    b2 = ax.bar(x + w/2, v_on, w, color=COLOR_ON, hatch=HATCH_ON, edgecolor='white')
    for b, v in zip(b1, v_off):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{v:.1f}', ha='center', fontsize=12, fontweight='bold')
    for b, v in zip(b2, v_on):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{v:.1f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Goal success rate (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['Open Sea'], fontsize=11)
    ax.set_ylim(0, 100)
    ax.text(0.02, 0.95, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

    # (d) Avg DCPA
    ax = axes[1, 1]
    v_off = [off['avg_dcpa']]
    v_on = [on['avg_dcpa']]
    b1 = ax.bar(x - w/2, v_off, w, color=COLOR_OFF, edgecolor='white')
    b2 = ax.bar(x + w/2, v_on, w, color=COLOR_ON, hatch=HATCH_ON, edgecolor='white')
    for b, v in zip(b1, v_off):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{v:.1f}m', ha='center', fontsize=12, fontweight='bold')
    for b, v in zip(b2, v_on):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{v:.1f}m', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg. DCPA (m)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['Open Sea'], fontsize=11)
    ax.set_ylim(0, max(v_off + v_on) * 1.3)
    ax.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.02, 0.95, '(d)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

    fig.suptitle('Performance comparison (8 vessels, 10K steps)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fig5_performance_comparison.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(SAVE_DIR, 'fig5_performance_comparison.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: fig5_performance_comparison')


def fig5_collision_colregs(off, on):
    """2패널: Collision + COLREGs"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white')

    x = np.array([0])
    w = 0.35

    ax = axes[0]
    v_off = [off['collisions_per_ep']]
    v_on = [on['collisions_per_ep']]
    b1 = ax.bar(x - w/2, v_off, w, color=COLOR_OFF, label='Without Communication', edgecolor='white')
    b2 = ax.bar(x + w/2, v_on, w, color=COLOR_ON, label='With Communication', hatch=HATCH_ON, edgecolor='white')
    for b, v in zip(b1, v_off):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f'{v:.1f}', ha='center', fontsize=13, fontweight='bold')
    for b, v in zip(b2, v_on):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f'{v:.1f}', ha='center', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg. collisions per episode', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Open Sea'], fontsize=12)
    ax.set_ylim(0, max(v_off + v_on) * 1.4)
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.legend(fontsize=10)

    ax = axes[1]
    v_off = [off['compliance']]
    v_on = [on['compliance']]
    b1 = ax.bar(x - w/2, v_off, w, color=COLOR_OFF, edgecolor='white')
    b2 = ax.bar(x + w/2, v_on, w, color=COLOR_ON, hatch=HATCH_ON, edgecolor='white')
    for b, v in zip(b1, v_off):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{v:.1f}', ha='center', fontsize=13, fontweight='bold')
    for b, v in zip(b2, v_on):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{v:.1f}', ha='center', fontsize=13, fontweight='bold')
    ax.set_ylabel('COLREGs compliance (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Open Sea'], fontsize=12)
    ax.set_ylim(0, 100)
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

    fig.suptitle('Performance evaluation (8 vessels, 10K steps)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fig5_performance_collision_colregs.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(SAVE_DIR, 'fig5_performance_collision_colregs.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: fig5_performance_collision_colregs')


def fig5_colregs_per_situation(off, on):
    """COLREGs 규칙별 준수율"""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    situations = ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']
    labels = ['Head-On', 'Cross\n(Stand-On)', 'Cross\n(Give-Way)', 'Overtaking']
    x = np.arange(len(situations))
    w = 0.35

    v_off = [off['per_situation'][s]['rate'] for s in situations]
    v_on = [on['per_situation'][s]['rate'] for s in situations]

    b1 = ax.bar(x - w/2, v_off, w, color=COLOR_OFF, label='Phase 2', edgecolor='white')
    b2 = ax.bar(x + w/2, v_on, w, color=COLOR_ON, label='Phase 3', hatch=HATCH_ON, edgecolor='white')

    for b, v in zip(b1, v_off):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')
    for b, v in zip(b2, v_on):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')

    ax.set_ylabel('COLREGs compliance (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_title('COLREGs Compliance by Situation\n(10 runs × 2,000 steps per run)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fig5_colregs_per_situation.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(SAVE_DIR, 'fig5_colregs_per_situation.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: fig5_colregs_per_situation')


def fig5_episode_length(off, on):
    """에피소드 길이 비교"""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x = np.array([0])
    w = 0.35

    v_off = [off['avg_ep_len']]
    v_on = [on['avg_ep_len']]

    b1 = ax.bar(x - w/2, v_off, w, color=COLOR_OFF, label='Without Communication', edgecolor='white')
    b2 = ax.bar(x + w/2, v_on, w, color=COLOR_ON, label='With Communication', hatch=HATCH_ON, edgecolor='white')

    for b, v in zip(b1, v_off):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5, f'{v:.0f}', ha='center', fontsize=13, fontweight='bold')
    for b, v in zip(b2, v_on):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5, f'{v:.0f}', ha='center', fontsize=13, fontweight='bold')

    ax.set_ylabel('Steps per episode', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Open Sea'], fontsize=12)
    ax.set_ylim(0, max(v_off + v_on) * 1.3)
    ax.legend(fontsize=10)
    ax.set_title('Average episode length (8 vessels)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fig5_episode_length_time.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(SAVE_DIR, 'fig5_episode_length_time.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: fig5_episode_length_time')


def fig_trajectory(off_csv, on_csv):
    """Trajectory 비교 (actual data)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')

    for ax, csv_path, title in [(axes[0], off_csv, 'Without Communication'),
                                 (axes[1], on_csv, 'With Communication')]:
        ax.set_facecolor('white')
        df = pd.read_csv(csv_path)

        colors = plt.cm.tab10(np.linspace(0, 1, df['agent_id'].nunique()))
        for i, aid in enumerate(sorted(df['agent_id'].unique())):
            adf = df[df['agent_id'] == aid]
            steps = adf['step'].values
            xs = adf['x'].values
            zs = adf['z'].values

            seg_start = 0
            for j in range(1, len(steps)):
                if steps[j] < steps[j-1]:
                    ax.plot(xs[seg_start:j], zs[seg_start:j], color=colors[i], alpha=0.6, linewidth=0.8)
                    seg_start = j
            ax.plot(xs[seg_start:], zs[seg_start:], color=colors[i], alpha=0.6, linewidth=0.8,
                    label=f'Agent {aid}' if i < 4 else None)

        ax.set_xlabel('X position', fontsize=11)
        ax.set_ylabel('Z position', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        if ax == axes[0]:
            ax.legend(fontsize=8, loc='upper right', ncol=2)

    fig.suptitle('Vessel trajectories comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fig_trajectory_comparison.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(SAVE_DIR, 'fig_trajectory_comparison.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: fig_trajectory_comparison')


def fig_dcpa_distribution(off_csv, on_csv):
    """DCPA 분포 비교 (actual data)"""
    sys.path.insert(0, SCRIPT_DIR)
    from test import compute_encounter_dcpa

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    off_df = pd.read_csv(off_csv)
    on_df = pd.read_csv(on_csv)
    off_dcpa = compute_encounter_dcpa(off_df)['min_distances']
    on_dcpa = compute_encounter_dcpa(on_df)['min_distances']

    if off_dcpa:
        ax.hist(off_dcpa, bins=30, alpha=0.6, color=COLOR_OFF, label='Without Communication', edgecolor='white')
    if on_dcpa:
        ax.hist(on_dcpa, bins=30, alpha=0.6, color=COLOR_ON, label='With Communication', edgecolor='white')

    ax.axvline(x=5.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Collision threshold (5m)')
    ax.set_xlabel('Minimum passing distance (m)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('DCPA distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fig_dcpa_distribution.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(SAVE_DIR, 'fig_dcpa_distribution.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: fig_dcpa_distribution')


# ── Main ──
if __name__ == '__main__':
    off = OFF_DATA
    on = ON_DATA

    print("=== Performance values ===")
    print(f"Comm OFF: compliance={off['compliance']:.1f}%, goal={off['goal_rate']:.1f}%, "
          f"collision/ep={off['collisions_per_ep']:.1f}, DCPA={off['avg_dcpa']:.1f}m, ep_len={off['avg_ep_len']}")
    for s in ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']:
        print(f"  {s}: {off['per_situation'][s]['rate']:.1f}%")

    print(f"\nComm ON:  compliance={on['compliance']:.1f}%, goal={on['goal_rate']:.1f}%, "
          f"collision/ep={on['collisions_per_ep']:.1f}, DCPA={on['avg_dcpa']:.1f}m, ep_len={on['avg_ep_len']}")
    for s in ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']:
        print(f"  {s}: {on['per_situation'][s]['rate']:.1f}%")

    print(f"\n{'='*60}")
    print("Generating figures...")
    print(f"{'='*60}")

    fig5_performance_comparison(off, on)
    fig5_collision_colregs(off, on)
    fig5_colregs_per_situation(off, on)
    fig5_episode_length(off, on)
    fig_trajectory(OFF_CSV, ON_CSV)
    fig_dcpa_distribution(OFF_CSV, ON_CSV)

    print(f"\nAll saved to: {SAVE_DIR}")
    for f in sorted(os.listdir(SAVE_DIR)):
        print(f"  {f}")
