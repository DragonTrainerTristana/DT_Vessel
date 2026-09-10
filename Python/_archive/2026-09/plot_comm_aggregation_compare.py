"""
Communication Aggregation 비교: sum-of-4 vs nearest-1 (vs OFF baseline)

데이터 소스:
- 기존 OFF/ON(sum-4):  trajectory_data/{env}_compare_comm{OFF,ON}_10x2000_*.csv
- 새 nearest-1:        trajectory_data/{env}_..._nearest1_*.csv (사용자 측정 후)

환경: Open Ocean, Narrow Channel, Coastal
메트릭: collision count, success count (per-run), Avg Reward

논문 스타일 (gen_5graphs_final.py와 동일 색/포맷)
"""
import os
import re
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PROJECT_ROOT

# 논문 스타일 색상
COLOR_OFF = '#E74C3C'   # 빨강 — Comm OFF
COLOR_SUM = '#3498DB'   # 파랑 — Sum-of-4 (기존 ON)
COLOR_NN1 = '#2ECC71'   # 녹색 — Nearest-1
ENV_ORDER = ['Open Ocean', 'Narrow Channel', 'Coastal']
TRAJ_DIR = os.path.join(PROJECT_ROOT, "trajectory_data")


def find_latest(pattern):
    files = sorted(glob.glob(os.path.join(TRAJ_DIR, pattern)))
    return files[-1] if files else None


def load_runs(csv_path):
    """csv 로드 후 run별 (collision, success, avg_reward) 추출"""
    if csv_path is None or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if 'run_id' not in df.columns:
        # _add_run_id 없는 옛 csv는 step 단조성 기반 분리
        return None
    n_runs = df['run_id'].nunique()
    cols, sucs, rews = [], [], []
    for r in range(n_runs):
        d = df[df['run_id'] == r]
        # collision: goal_dist 변화로 추정 어려움. 대신 별도 통계 가정
        # 여기선 approximation: 'speed', 'goal_dist' 분석 어려우니 단순 row count로 episode 추정
        cols.append(0)
        sucs.append(0)
        rews.append(d['speed'].mean() if 'speed' in d.columns else 0)
    return {'collision': cols, 'success': sucs, 'reward': rews}


# 기존 raw collision/success 데이터 (memory 기록 기반 — fallback)
RAW_OFF_COLLISIONS = {
    'Open Ocean':    [2, 2, 3, 1, 3, 1, 0, 1, 1, 0],
    'Narrow Channel':[8, 15, 21, 13, 17, 15, 4, 11, 10, 9],
    'Coastal':       [33, 49, 36, 48, 35, 35, 37, 37, 41, 28],
}
RAW_ON_COLLISIONS = {
    'Open Ocean':    [1, 2, 0, 0, 2, 1, 0, 0, 2, 0],
    'Narrow Channel':[8, 6, 9, 5, 10, 7, 8, 7, 10, 11],
    'Coastal':       [18, 20, 22, 24, 19, 21, 16, 29, 18, 17],
}
RAW_OFF_SUCCESS = {
    'Open Ocean':    [8, 10, 9, 12, 8, 5, 2, 8, 6, 8],
    'Narrow Channel':[9, 11, 9, 6, 15, 14, 6, 8, 10, 8],
    'Coastal':       [4, 1, 3, 1, 3, 2, 3, 5, 4, 1],
}
RAW_ON_SUCCESS = {
    'Open Ocean':    [7, 9, 5, 5, 11, 14, 6, 9, 9, 9],
    'Narrow Channel':[11, 7, 9, 10, 8, 16, 10, 9, 13, 16],
    'Coastal':       [9, 10, 8, 10, 18, 10, 11, 9, 12, 6],
}


def parse_summary_csv(env_key, mode):
    """nearest1 평가 후 summary csv에서 collision/success 추출"""
    env_tag = {
        'Open Ocean': 'open_ocean',
        'Narrow Channel': 'narrow',
        'Coastal': 'coastal',
    }[env_key]
    pat = f"{env_tag}*nearest1*summary*.csv"
    path = find_latest(pat)
    if path is None:
        return None
    df = pd.read_csv(path)
    return df


def plot_bar_3mode(metric_dict, ylabel, title, filename, fmt='{:.1f}'):
    """3 환경 × 3 모드 (OFF/sum-4/nearest-1) 비교 bar"""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(ENV_ORDER))
    width = 0.27

    means_off = [np.mean(metric_dict['OFF'][e]) for e in ENV_ORDER]
    stds_off  = [np.std(metric_dict['OFF'][e])  for e in ENV_ORDER]
    means_sum = [np.mean(metric_dict['SUM'][e]) for e in ENV_ORDER]
    stds_sum  = [np.std(metric_dict['SUM'][e])  for e in ENV_ORDER]
    means_nn1 = [np.mean(metric_dict['NN1'][e]) if metric_dict['NN1'][e] else 0 for e in ENV_ORDER]
    stds_nn1  = [np.std(metric_dict['NN1'][e])  if metric_dict['NN1'][e] else 0 for e in ENV_ORDER]

    b1 = ax.bar(x - width, means_off, width, yerr=stds_off, color=COLOR_OFF,
                alpha=0.85, capsize=5, edgecolor='black', linewidth=0.5, label='COMM OFF')
    b2 = ax.bar(x,         means_sum, width, yerr=stds_sum, color=COLOR_SUM,
                alpha=0.85, capsize=5, edgecolor='black', linewidth=0.5, label='COMM ON (sum-of-4)')
    b3 = ax.bar(x + width, means_nn1, width, yerr=stds_nn1, color=COLOR_NN1,
                alpha=0.85, capsize=5, edgecolor='black', linewidth=0.5, label='COMM ON (nearest-1)')

    max_std = max(max(stds_off), max(stds_sum), max(stds_nn1)) or 0.1
    for bars, means in [(b1, means_off), (b2, means_sum), (b3, means_nn1)]:
        for bar, m in zip(bars, means):
            if m > 0 or m == 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_std*0.15,
                        fmt.format(m), ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

    ax.set_xlabel('Environment', fontweight='bold', fontsize=12)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
    ax.set_title(f'{title}\n(10 runs × 2,000 steps per run)',
                 fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(ENV_ORDER, fontsize=11)
    ax.legend(fontsize=10, loc='best')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_dir = os.path.join(PROJECT_ROOT, "figures", "comm_aggregation")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{filename}_{ts}")
    fig.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}.png/pdf")
    return out_path


def main():
    print("=" * 70)
    print("Communication Aggregation Comparison: OFF vs sum-of-4 vs nearest-1")
    print("=" * 70)

    # nearest-1 데이터 로드 (있으면)
    nn1_collisions = {}
    nn1_success = {}
    for env in ENV_ORDER:
        df = parse_summary_csv(env, 'nearest1')
        if df is not None:
            # arrival_step에서 collision 여부 추정 어려움. 단순화: arrived flag 또는 별도 처리
            # multitest summary는 collision_count, success_count 컬럼 가정
            if 'collision_count' in df.columns:
                nn1_collisions[env] = df['collision_count'].tolist()
            else:
                nn1_collisions[env] = []
            if 'success_count' in df.columns:
                nn1_success[env] = df['success_count'].tolist()
            else:
                nn1_success[env] = []
            print(f"[OK] nearest-1 {env}: {len(nn1_collisions[env])} runs")
        else:
            nn1_collisions[env] = []
            nn1_success[env] = []
            print(f"[MISSING] nearest-1 {env}: csv 없음")

    # Plot Collision
    plot_bar_3mode(
        {'OFF': RAW_OFF_COLLISIONS, 'SUM': RAW_ON_COLLISIONS, 'NN1': nn1_collisions},
        ylabel='Collisions per Run',
        title='Collisions: OFF vs Sum-of-4 vs Nearest-1',
        filename='aggregation_collisions',
        fmt='{:.1f}',
    )

    # Plot Success
    plot_bar_3mode(
        {'OFF': RAW_OFF_SUCCESS, 'SUM': RAW_ON_SUCCESS, 'NN1': nn1_success},
        ylabel='Successes per Run',
        title='Successes: OFF vs Sum-of-4 vs Nearest-1',
        filename='aggregation_successes',
        fmt='{:.1f}',
    )


if __name__ == "__main__":
    main()
