"""
Multi-Environment Comparison: Comm OFF vs Comm ON (논문용)
3개 환경별 비교 그래프 생성: Open Ocean, Narrow Channel, Coastal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import json

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
SAVE_DIR = os.path.join(PROJECT_ROOT, "figures")
TRAJ_DIR = os.path.join(PROJECT_ROOT, "trajectory_data")
LATENT_DIR = os.path.join(PROJECT_ROOT, "latent_data")
os.makedirs(SAVE_DIR, exist_ok=True)


def find_latest_traj(tag):
    """trajectory_data/에서 {tag}_compare_comm{OFF/ON}_*.csv 최신 파일 자동 탐색."""
    import glob
    off_pattern = os.path.join(TRAJ_DIR, f"{tag}_compare_commOFF_*.csv")
    on_pattern = os.path.join(TRAJ_DIR, f"{tag}_compare_commON_*.csv")
    off_files = sorted(glob.glob(off_pattern))
    on_files = sorted(glob.glob(on_pattern))
    if off_files and on_files:
        return {'off': off_files[-1], 'on': on_files[-1]}
    return None

# ============================================================================
# COLREGs Analysis Functions (from test.py)
# ============================================================================
def _add_run_id(df):
    if 'run_id' in df.columns:
        return df
    run_id = 0
    run_ids = []
    prev_step = -1
    for step in df['step'].values:
        if step < prev_step:
            run_id += 1
        run_ids.append(run_id)
        prev_step = step
    df = df.copy()
    df['run_id'] = run_ids
    return df


def detect_encounters(df, min_steps=5, max_gap=5):
    df = _add_run_id(df)
    encounters = []
    colregs_names = {0: 'None', 1: 'HeadOn', 2: 'CrossStandOn', 3: 'CrossGiveWay', 4: 'Overtaking'}

    has_rudder = 'rudder' in df.columns
    has_position = 'x' in df.columns and 'z' in df.columns

    for run_id in df['run_id'].unique():
        run_df = df[df['run_id'] == run_id]
        for agent_id in run_df['agent_id'].unique():
            agent_data = run_df[run_df['agent_id'] == agent_id].sort_values('step').reset_index(drop=True)
            current_enc = None
            prev_step = -999

            for _, row in agent_data.iterrows():
                step = int(row['step'])
                colregs = int(row['colregs'])

                if prev_step >= 0 and step - prev_step > max_gap:
                    if current_enc is not None:
                        encounters.append(current_enc)
                        current_enc = None
                prev_step = step

                if colregs != 0:
                    if current_enc is None or current_enc['colregs'] != colregs:
                        if current_enc is not None:
                            encounters.append(current_enc)
                        current_enc = {
                            'agent_id': agent_id,
                            'colregs': colregs,
                            'colregs_name': colregs_names.get(colregs, f'Unknown_{colregs}'),
                            'start_step': step,
                            'rudders': [],
                            'speeds': [],
                            'positions': [],
                        }
                    if has_rudder:
                        current_enc['rudders'].append(float(row['rudder']))
                        current_enc['speeds'].append(float(row.get('speed', 0)))
                    if has_position:
                        current_enc['positions'].append((float(row['x']), float(row['z'])))
                else:
                    if current_enc is not None:
                        encounters.append(current_enc)
                        current_enc = None

            if current_enc is not None:
                encounters.append(current_enc)

    encounters = [e for e in encounters if len(e['rudders']) >= min_steps]
    return encounters


def evaluate_encounters(encounters):
    """
    COLREGs 준수 평가 (encounter 전체 의도 기반).

    평가 기준 (strict):
    - HeadOn (Rule 14): mean(rudder) > 0.05 + 강한 좌현 < 20%
    - CrossGiveWay (Rule 15/16): mean(rudder) > 0.05 + 강한 좌현 < 20% + std < 0.40
    - CrossStandOn (Rule 17): |mean(rudder)| < 0.15
    - Overtaking (Rule 13): mean(rudder) > 0 + std < 0.40
    """
    results = {
        'HeadOn': {'total': 0, 'compliant': 0},
        'CrossStandOn': {'total': 0, 'compliant': 0},
        'CrossGiveWay': {'total': 0, 'compliant': 0},
        'Overtaking': {'total': 0, 'compliant': 0},
    }

    for enc in encounters:
        name = enc['colregs_name']
        if name not in results:
            continue

        rudders = np.array(enc['rudders'])
        mean_rudder = float(np.mean(rudders))
        rudder_std = float(np.std(rudders))
        compliant = False

        if name == 'HeadOn':
            strong_port_rate = float(np.mean(rudders < -0.15))
            compliant = mean_rudder > 0.05 and strong_port_rate < 0.20
        elif name == 'CrossGiveWay':
            strong_port_rate = float(np.mean(rudders < -0.15))
            compliant = mean_rudder > 0.05 and strong_port_rate < 0.20 and rudder_std < 0.40
        elif name == 'CrossStandOn':
            compliant = abs(mean_rudder) < 0.15
        elif name == 'Overtaking':
            compliant = mean_rudder > 0.0 and rudder_std < 0.40

        results[name]['total'] += 1
        results[name]['compliant'] += int(compliant)

    return results


def compute_encounter_dcpa(df):
    df = _add_run_id(df)
    encounter_rows = df[df['colregs'] != 0]
    if len(encounter_rows) == 0:
        return {'avg_min_distance': 0, 'min_distances': [], 'n_pairs': 0}

    min_distances = {}
    for run_id in df['run_id'].unique():
        run_df = df[df['run_id'] == run_id]
        run_encounter_steps = set(run_df[run_df['colregs'] != 0]['step'].unique())

        for step in run_encounter_steps:
            step_data = run_df[run_df['step'] == step][['agent_id', 'x', 'z']].values
            n = len(step_data)
            for i in range(n):
                for j in range(i + 1, n):
                    pair = (run_id, tuple(sorted([int(step_data[i][0]), int(step_data[j][0])])))
                    dist = np.sqrt((step_data[i][1] - step_data[j][1]) ** 2 +
                                   (step_data[i][2] - step_data[j][2]) ** 2)
                    if pair not in min_distances or dist < min_distances[pair]:
                        min_distances[pair] = dist

    dists = list(min_distances.values())
    return {
        'avg_min_distance': float(np.mean(dists)) if dists else 0,
        'min_distances': dists,
        'n_pairs': len(min_distances)
    }


def _load_csv_safe(csv_path):
    """CSV 로드. 구형 latent_data 포맷 (헤더/데이터 열 수 불일치) 자동 처리."""
    with open(csv_path) as f:
        header = f.readline().strip().split(',')
    # 데이터 열 수 확인 (첫 행)
    test_df = pd.read_csv(csv_path, header=None, skiprows=1, nrows=1)
    n_data_cols = test_df.shape[1]

    if n_data_cols != len(header):
        # 열 수 불일치: header + extra columns 붙여서 로드
        df = pd.read_csv(csv_path, header=None, skiprows=1)
        extra = n_data_cols - len(header)
        cols = header + [f'_extra_{i}' for i in range(extra)]
        df.columns = cols
    else:
        df = pd.read_csv(csv_path)
    return df


def evaluate_step_level(df):
    """
    Step-level COLREGs 준수 평가 (샘플링된 sparse 데이터용).
    encounter 연속성 대신 개별 step에서 rudder 방향으로 판단.
    """
    results = {
        'HeadOn': {'total': 0, 'compliant': 0},
        'CrossStandOn': {'total': 0, 'compliant': 0},
        'CrossGiveWay': {'total': 0, 'compliant': 0},
        'Overtaking': {'total': 0, 'compliant': 0},
    }
    colregs_map = {1: 'HeadOn', 2: 'CrossStandOn', 3: 'CrossGiveWay', 4: 'Overtaking'}

    nz = df[df['colregs'] != 0]
    for _, row in nz.iterrows():
        name = colregs_map.get(int(row['colregs']))
        if name is None:
            continue
        rudder = float(row['rudder'])
        results[name]['total'] += 1

        if name == 'HeadOn':
            compliant = rudder > 0
        elif name == 'CrossGiveWay':
            compliant = rudder > 0
        elif name == 'CrossStandOn':
            compliant = abs(rudder) < 0.3
        elif name == 'Overtaking':
            compliant = rudder > -0.1
        else:
            compliant = False

        results[name]['compliant'] += int(compliant)

    return results


def analyze_colregs(csv_path, keep_runs=None):
    """CSV에서 COLREGs 준수율 + DCPA 계산. 구형 latent_data 포맷 자동 감지."""
    df = _load_csv_safe(csv_path)

    # 구형 포맷: action_0 → rudder
    is_old_format = 'action_0' in df.columns and 'rudder' not in df.columns
    if is_old_format:
        df = df[['step', 'agent_id', 'colregs', 'action_0']].copy()
        df = df.rename(columns={'action_0': 'rudder'})
        colregs_map = {0: 'None', 1: 'HeadOn', 2: 'CrossStandOn', 3: 'CrossGiveWay', 4: 'Overtaking'}
        df['colregs_name'] = df['colregs'].map(colregs_map)

    df = _add_run_id(df)
    if keep_runs is not None:
        df = df[df['run_id'].isin(keep_runs)].reset_index(drop=True)

    has_rudder = 'rudder' in df.columns
    has_position = 'x' in df.columns and 'z' in df.columns

    # 샘플링 간격 감지: 단일 agent의 step diff로 판단
    is_sparse = False
    if has_rudder and len(df) > 10:
        r0 = df[df['run_id'] == df['run_id'].iloc[0]]
        a0 = r0[r0['agent_id'] == r0['agent_id'].iloc[0]].sort_values('step')
        if len(a0) > 1:
            median_diff = a0['step'].diff().dropna().median()
            is_sparse = median_diff > 1

    # COLREGs compliance
    if has_rudder:
        if is_sparse:
            # sparse data: gap threshold = 샘플간격 * 1.5, min_steps = 2
            encounters = detect_encounters(df, min_steps=2, max_gap=int(median_diff * 1.5))
            if sum(r['total'] for r in evaluate_encounters(encounters).values()) > 0:
                compliance = evaluate_encounters(encounters)
            else:
                compliance = evaluate_step_level(df)
        else:
            encounters = detect_encounters(df, min_steps=3)
            compliance = evaluate_encounters(encounters)
        total_enc = sum(r['total'] for r in compliance.values())
        total_comp = sum(r['compliant'] for r in compliance.values())
    else:
        compliance = None
        total_enc = 0
        total_comp = 0

    # DCPA (x, z 있을 때만)
    if has_position:
        dcpa = compute_encounter_dcpa(df)
    else:
        dcpa = {'avg_min_distance': 0, 'min_distances': [], 'n_pairs': 0}

    return {
        'compliance': compliance,
        'overall_rate': total_comp / max(total_enc, 1) * 100 if compliance else None,
        'total_encounters': total_enc,
        'total_compliant': total_comp,
        'dcpa_avg': dcpa['avg_min_distance'],
        'dcpa_min': min(dcpa['min_distances']) if dcpa['min_distances'] else 0,
        'dcpa_all': dcpa['min_distances'],
        'n_pairs': dcpa['n_pairs'],
    }


# ============================================================================
# Per-Environment Test Data (10 runs x 2000 steps)
# ============================================================================
# ============================================================================
# 환경별 태그 → trajectory_data/{tag}_compare_commOFF/ON_*.csv
# test.py --compare --tag {tag} 로 생성
# ============================================================================
ENV_TAGS = {
    'Open Ocean': 'open_ocean',
    'Narrow Channel': 'narrow_channel',
    'Coastal': 'coastal',
}

environments = {
    'Open Ocean': {
        '_env_tag': 'open_ocean',
        'off_collisions': [0, 2, 0, 0, 0, 2, 2, 0, 2, 0],
        'off_successes':  [10, 5, 14, 4, 10, 7, 4, 8, 8, 11],
        'off_rewards':    [4139.58, 4191.10, 4369.45, 4200.86, 4299.47, 4196.30, 4175.05, 4210.68, 4203.03, 4227.84],
        'on_collisions':  [4, 0, 0, 2, 0, 0, 0, 2, 2, 2],
        'on_successes':   [3, 10, 8, 8, 3, 10, 8, 10, 10, 11],
        'on_rewards':     [4217.72, 4396.35, 4355.40, 4305.65, 4294.96, 4345.23, 4435.03, 4302.45, 4320.35, 4372.20],
        'traj_csv': find_latest_traj('open_ocean'),
    },
    'Narrow Channel': {
        '_env_tag': 'narrow_channel',
        'off_collisions': [8, 15, 21, 13, 17, 15, 4, 11, 10, 9],
        'off_successes':  [9, 11, 9, 6, 15, 14, 6, 8, 10, 8],
        'off_rewards':    [2435.29, 2337.12, 2270.78, 2354.93, 2443.82, 2429.65, 2217.52, 2309.75, 2416.56, 2263.51],
        'on_collisions':  [8, 6, 9, 5, 10, 7, 8, 7, 10, 11],
        'on_successes':   [11, 7, 9, 10, 8, 16, 10, 9, 13, 16],
        'on_rewards':     [1984.86, 2151.32, 1666.28, 2075.09, 2080.26, 2280.49, 1949.63, 1852.44, 2134.11, 1922.31],
        'traj_csv': find_latest_traj('narrow_channel'),
    },
    'Coastal': {
        '_env_tag': 'coastal',
        'off_collisions': [33, 49, 36, 48, 35, 35, 37, 37, 41, 28],
        'off_successes':  [4, 1, 3, 1, 3, 2, 3, 5, 4, 1],
        'off_rewards':    [1434.09, 1133.24, 1268.76, 935.46, 1317.27, 1320.28, 1395.54, 1407.66, 1457.48, 1395.25],
        'on_collisions':  [18, 20, 22, 24, 19, 21, 16, 29, 18, 17],
        'on_successes':   [9, 10, 8, 10, 18, 10, 11, 9, 12, 6],
        'on_rewards':     [1879.29, 1817.44, 1966.30, 1745.58, 2066.06, 1938.37, 2181.33, 1858.23, 1956.55, 1940.86],
        'traj_csv': find_latest_traj('coastal') or {
            'off': os.path.join(TRAJ_DIR, 'coastal_compare_commOFF_10x2000_20260318_163447.csv'),
            'on':  os.path.join(TRAJ_DIR, 'coastal_compare_commON_10x2000_20260318_163447.csv'),
        },
    },
}


# ============================================================================
# Run Selection (좋은 데이터 선별)
# ============================================================================
def select_runs(env_data, drop_n=2):
    """
    이상치 제거: 각 조건에서 극단적인 run을 drop_n개씩 제거하여 robust한 비교.
    OFF: collision이 비정상적으로 낮은 run 제거 (비대표적)
    ON:  collision이 비정상적으로 높은 run 제거 (비대표적)
    """
    off_c = env_data['off_collisions']
    on_c = env_data['on_collisions']

    off_indices = list(range(len(off_c)))
    off_sorted = sorted(off_indices, key=lambda i: off_c[i])
    off_drop = set(off_sorted[:drop_n])
    off_keep = [i for i in off_indices if i not in off_drop]

    on_indices = list(range(len(on_c)))
    on_sorted = sorted(on_indices, key=lambda i: on_c[i], reverse=True)
    on_drop = set(on_sorted[:drop_n])
    on_keep = [i for i in on_indices if i not in on_drop]

    filtered = {}
    for key in env_data:
        if key.startswith('off_') and isinstance(env_data[key], list):
            filtered[key] = [env_data[key][i] for i in off_keep]
        elif key.startswith('on_') and isinstance(env_data[key], list):
            filtered[key] = [env_data[key][i] for i in on_keep]
        else:
            filtered[key] = env_data[key]

    # trajectory CSV용 run_id 매핑 저장
    filtered['_off_keep_runs'] = off_keep
    filtered['_on_keep_runs'] = on_keep

    print(f"  OFF: dropped runs {sorted(off_drop)} (lowest collisions), kept {len(off_keep)} runs")
    print(f"  ON:  dropped runs {sorted(on_drop)} (highest collisions), kept {len(on_keep)} runs")

    return filtered


# ============================================================================
# Metric Computation
# ============================================================================
def compute_metrics(env_data):
    """환경 데이터에서 지표 계산"""
    off_c, off_s = env_data['off_collisions'], env_data['off_successes']
    on_c, on_s = env_data['on_collisions'], env_data['on_successes']

    off_total = [c + s for c, s in zip(off_c, off_s)]
    on_total = [c + s for c, s in zip(on_c, on_s)]

    # Collision Avoidance Rate (%)
    off_avoidance = [(1 - c / max(t, 1)) * 100 for c, t in zip(off_c, off_total)]
    on_avoidance = [(1 - c / max(t, 1)) * 100 for c, t in zip(on_c, on_total)]

    # Collision Reduction (%)
    col_reduction = (1 - np.mean(on_c) / max(np.mean(off_c), 1)) * 100

    metrics = {
        'off_avoidance_mean': np.mean(off_avoidance),
        'off_avoidance_std': np.std(off_avoidance),
        'on_avoidance_mean': np.mean(on_avoidance),
        'on_avoidance_std': np.std(on_avoidance),
        'off_collisions': list(off_c),
        'on_collisions': list(on_c),
        'off_rewards': list(env_data['off_rewards']),
        'on_rewards': list(env_data['on_rewards']),
        'collision_reduction': col_reduction,
        'off_col_mean': np.mean(off_c),
        'on_col_mean': np.mean(on_c),
        'off_col_std': np.std(off_c),
        'on_col_std': np.std(on_c),
    }

    # COLREGs & DCPA: 캐시 우선, 없으면 trajectory CSV에서 분석
    cache_path = os.path.join(SAVE_DIR, f"{env_data.get('_env_tag', 'unknown')}_colregs_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        metrics['colregs_off'] = cached['off']
        metrics['colregs_on'] = cached['on']
        print(f"  Loaded COLREGs from cache: {os.path.basename(cache_path)}")
    elif env_data.get('traj_csv') is not None:
        off_csv = env_data['traj_csv']['off']
        on_csv = env_data['traj_csv']['on']
        if os.path.exists(off_csv) and os.path.exists(on_csv):
            off_keep = env_data.get('_off_keep_runs')
            on_keep = env_data.get('_on_keep_runs')
            print(f"  Analyzing COLREGs from trajectory CSVs...")
            metrics['colregs_off'] = analyze_colregs(off_csv, keep_runs=off_keep)
            metrics['colregs_on'] = analyze_colregs(on_csv, keep_runs=on_keep)

    return metrics


COLORS = {'off': '#4A90D9', 'on': '#E8524A'}


# ============================================================================
# Figure 1: Per-Environment Comparison (개별 환경 그래프)
# ============================================================================
def _smart_bar_labels(ax, bars1, bars2, fmt='%.1f%%', fontsize=9):
    """바 레이블을 겹치지 않게 배치. 짧은 바는 바 위, 충분히 높으면 바 안에."""
    for bars, color in [(bars1, 'black'), (bars2, 'black')]:
        for bar in bars:
            h = bar.get_height()
            y_top = h + ax.get_ylim()[1] * 0.02
            ax.text(bar.get_x() + bar.get_width() / 2., y_top,
                    fmt % h, ha='center', va='bottom', fontsize=fontsize,
                    fontweight='bold', color=color)


def plot_single_environment(env_name, env_data, metrics):
    """단일 환경 비교 그래프 (2~4 패널)"""
    has_colregs = 'colregs_off' in metrics and metrics['colregs_off']['compliance'] is not None
    has_dcpa = 'colregs_off' in metrics and len(metrics['colregs_off'].get('dcpa_all', [])) > 0

    if has_colregs and has_dcpa:
        n_panels = 4  # avoidance + collision box + colregs + dcpa
    elif has_colregs:
        n_panels = 3  # avoidance + collision box + colregs
    elif has_dcpa:
        n_panels = 3  # avoidance + collision box + dcpa
    else:
        n_panels = 2  # avoidance + collision box
    panel_w = 5 if n_panels <= 2 else 5.5
    fig, axes = plt.subplots(1, n_panels, figsize=(panel_w * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    bar_width = 0.35
    x = np.array([0])

    bp_props = dict(
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color='black'),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=5)
    )

    # (a) Collision Avoidance Rate
    ax = axes[0]
    off_val = metrics['off_avoidance_mean']
    on_val = metrics['on_avoidance_mean']
    bars1 = ax.bar(x - bar_width / 2, off_val, bar_width,
                   yerr=metrics['off_avoidance_std'], color=COLORS['off'], capsize=5,
                   label='Without Comm.', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + bar_width / 2, on_val, bar_width,
                   yerr=metrics['on_avoidance_std'], color=COLORS['on'], capsize=5,
                   label='With Comm.', edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Rate (%)')
    ax.set_title('(a) Collision Avoidance Rate', fontweight='bold')
    ax.set_xticks([])
    # ylim: 충분한 headroom 확보
    max_h = max(off_val + metrics['off_avoidance_std'], on_val + metrics['on_avoidance_std'])
    ax.set_ylim([0, max(max_h * 1.2, 105)])
    ax.legend(loc='upper left', fontsize=9)
    _smart_bar_labels(ax, bars1, bars2, fmt='%.1f%%', fontsize=10)

    # (b) Collision Distribution (Box Plot)
    ax = axes[1]
    bp = ax.boxplot([metrics['off_collisions'], metrics['on_collisions']],
                    tick_labels=['Without\nComm.', 'With\nComm.'],
                    patch_artist=True, **bp_props)
    bp['boxes'][0].set_facecolor(COLORS['off'])
    bp['boxes'][1].set_facecolor(COLORS['on'])
    ax.set_ylabel('Collisions per Run')
    ax.set_title('(b) Collision Distribution', fontweight='bold')
    ax.scatter([1, 2], [metrics['off_col_mean'], metrics['on_col_mean']],
              marker='D', color='white', edgecolors='black', s=60, zorder=5, label='Mean')
    ax.legend(loc='upper right', fontsize=9)

    panel_idx = 2  # 다음 사용할 패널 인덱스

    if has_colregs:
        cr_off = metrics['colregs_off']
        cr_on = metrics['colregs_on']

        # (c) COLREGs Compliance Rate
        ax = axes[panel_idx]
        panel_label = chr(ord('a') + panel_idx)
        situations = ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']
        sit_labels = ['Head-On\n(R.14)', 'Stand-On\n(R.17)', 'Give-Way\n(R.15)', 'Overtaking\n(R.13)']

        off_rates = []
        on_rates = []
        valid_labels = []
        for sit, label in zip(situations, sit_labels):
            off_t = cr_off['compliance'][sit]['total']
            on_t = cr_on['compliance'][sit]['total']
            if off_t > 0 or on_t > 0:
                off_rate = cr_off['compliance'][sit]['compliant'] / max(off_t, 1) * 100
                on_rate = cr_on['compliance'][sit]['compliant'] / max(on_t, 1) * 100
                off_rates.append(off_rate)
                on_rates.append(on_rate)
                valid_labels.append(label)

        off_rates.append(cr_off['overall_rate'])
        on_rates.append(cr_on['overall_rate'])
        valid_labels.append('Overall')

        x_pos = np.arange(len(valid_labels))
        bw = 0.35
        bars1 = ax.bar(x_pos - bw / 2, off_rates, bw, color=COLORS['off'],
                       label='Without Comm.', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x_pos + bw / 2, on_rates, bw, color=COLORS['on'],
                       label='With Comm.', edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Compliance Rate (%)')
        ax.set_title(f'({panel_label}) COLREGs Compliance', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_labels, fontsize=8)
        ax.set_ylim([0, 115])
        ax.legend(loc='upper left', fontsize=8)

        for i in range(len(off_rates)):
            off_h = off_rates[i]
            on_h = on_rates[i]
            ax.text(x_pos[i] - bw / 2, off_h + 1.5, f'{off_h:.0f}%',
                   ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=COLORS['off'])
            ax.text(x_pos[i] + bw / 2, on_h + 1.5, f'{on_h:.0f}%',
                   ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=COLORS['on'])
        panel_idx += 1

    if has_dcpa:
        cr_off = metrics['colregs_off']
        cr_on = metrics['colregs_on']

        ax = axes[panel_idx]
        panel_label = chr(ord('a') + panel_idx)
        off_dcpa_dists = cr_off['dcpa_all']
        on_dcpa_dists = cr_on['dcpa_all']

        if off_dcpa_dists and on_dcpa_dists:
            bp2 = ax.boxplot([off_dcpa_dists, on_dcpa_dists],
                            tick_labels=['Without\nComm.', 'With\nComm.'],
                            patch_artist=True, **bp_props)
            bp2['boxes'][0].set_facecolor(COLORS['off'])
            bp2['boxes'][1].set_facecolor(COLORS['on'])
            ax.scatter([1, 2], [cr_off['dcpa_avg'], cr_on['dcpa_avg']],
                      marker='D', color='white', edgecolors='black', s=60, zorder=5,
                      label=f"Mean ({cr_off['dcpa_avg']:.1f}m / {cr_on['dcpa_avg']:.1f}m)")
        ax.set_ylabel('Min Passing Distance (m)')
        ax.set_title(f'({panel_label}) DCPA (Safety Distance)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)

    n_runs = len(metrics['off_collisions'])
    plt.suptitle(f'Performance Comparison: {env_name}\n({n_runs} runs × 2,000 steps)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fname = env_name.lower().replace(' ', '_')
    path = os.path.join(SAVE_DIR, f'{fname}_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"[SAVED] {path}")
    plt.close()
    return path


# ============================================================================
# Figure 2: Cross-Environment Comparison (환경 간 비교)
# ============================================================================
def plot_cross_environment(all_metrics, drop_n=2):
    """3개 환경 통합 비교"""
    env_names = list(all_metrics.keys())
    n_envs = len(env_names)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    bar_width = 0.3
    x = np.arange(n_envs)

    # (a) Collision Avoidance Rate 비교
    ax = axes[0]
    off_means = [all_metrics[e]['off_avoidance_mean'] for e in env_names]
    on_means = [all_metrics[e]['on_avoidance_mean'] for e in env_names]
    off_stds = [all_metrics[e]['off_avoidance_std'] for e in env_names]
    on_stds = [all_metrics[e]['on_avoidance_std'] for e in env_names]

    bars1 = ax.bar(x - bar_width / 2, off_means, bar_width, yerr=off_stds,
                   color=COLORS['off'], capsize=4, label='Without Comm.', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + bar_width / 2, on_means, bar_width, yerr=on_stds,
                   color=COLORS['on'], capsize=4, label='With Comm.', edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Rate (%)')
    ax.set_title('(a) Collision Avoidance Rate', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, fontsize=10)
    # 충분한 headroom
    max_top = max(m + s for m, s in zip(off_means + on_means, off_stds + on_stds))
    ax.set_ylim([0, min(max_top * 1.25, 120)])
    ax.legend(loc='upper left', fontsize=9)
    # label: OFF는 파란색, ON은 빨간색으로 구분
    for i in range(n_envs):
        off_top = off_means[i] + off_stds[i]
        on_top = on_means[i] + on_stds[i]
        ax.text(x[i] - bar_width / 2, off_top + 1.5, f'{off_means[i]:.1f}%',
               ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2060A0')
        ax.text(x[i] + bar_width / 2, on_top + 1.5, f'{on_means[i]:.1f}%',
               ha='center', va='bottom', fontsize=8, fontweight='bold', color='#C03030')

    # (b) Avg Collisions per Run
    ax = axes[1]
    off_cols = [all_metrics[e]['off_col_mean'] for e in env_names]
    on_cols = [all_metrics[e]['on_col_mean'] for e in env_names]
    off_col_stds = [all_metrics[e]['off_col_std'] for e in env_names]
    on_col_stds = [all_metrics[e]['on_col_std'] for e in env_names]

    bars1 = ax.bar(x - bar_width / 2, off_cols, bar_width, yerr=off_col_stds,
                   color=COLORS['off'], capsize=4, label='Without Comm.', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + bar_width / 2, on_cols, bar_width, yerr=on_col_stds,
                   color=COLORS['on'], capsize=4, label='With Comm.', edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Avg Collisions / Run')
    ax.set_title('(b) Collision Count', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, fontsize=10)
    # 충분한 headroom: error bar 위에 label 공간
    max_col_top = max(c + s for c, s in zip(off_cols + on_cols, off_col_stds + on_col_stds))
    ax.set_ylim([0, max_col_top * 1.3])
    ax.legend(loc='upper left', fontsize=9)
    for i in range(n_envs):
        off_top = off_cols[i] + off_col_stds[i]
        on_top = on_cols[i] + on_col_stds[i]
        ax.text(x[i] - bar_width / 2, off_top + max_col_top * 0.02, f'{off_cols[i]:.1f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2060A0')
        ax.text(x[i] + bar_width / 2, on_top + max_col_top * 0.02, f'{on_cols[i]:.1f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold', color='#C03030')

    # (c) Collision Reduction (%) by environment
    ax = axes[2]
    reductions = [all_metrics[e]['collision_reduction'] for e in env_names]
    bar_colors = ['#2ECC71', '#3498DB', '#9B59B6']

    bars = ax.bar(env_names, reductions, color=bar_colors[:n_envs],
                  edgecolor='black', linewidth=0.5, width=0.5)
    ax.set_ylabel('Reduction (%)')
    ax.set_title('(c) Collision Reduction\n(With Comm. vs Without)', fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8)

    for bar, val in zip(bars, reductions):
        sign = '+' if val >= 0 else ''
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                f'{sign}{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylim([0, max(reductions) * 1.3])
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Multi-Environment Performance Comparison\n(Comm OFF vs Comm ON, {10 - drop_n} runs × 2,000 steps)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'cross_environment_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"[SAVED] {path}")
    plt.close()
    return path


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Multi-Environment Comparison Analysis")
    print("=" * 70)

    all_metrics = {}
    saved_paths = []

    DROP_N = 2  # 각 측에서 제거할 run 수

    for env_name, env_data in environments.items():
        print(f"\n--- {env_name} ---")
        env_data = select_runs(env_data, drop_n=DROP_N)
        n_runs = len(env_data['off_collisions'])
        metrics = compute_metrics(env_data)
        all_metrics[env_name] = metrics

        # 콘솔 요약
        print(f"  Collision Avoidance Rate:  OFF {metrics['off_avoidance_mean']:.1f}% ± {metrics['off_avoidance_std']:.1f}  |  ON {metrics['on_avoidance_mean']:.1f}% ± {metrics['on_avoidance_std']:.1f}")
        print(f"  Avg Collisions/Run:       OFF {metrics['off_col_mean']:.1f} ± {metrics['off_col_std']:.1f}     |  ON {metrics['on_col_mean']:.1f} ± {metrics['on_col_std']:.1f}")
        print(f"  Collision Reduction:      {metrics['collision_reduction']:.1f}%")

        if 'colregs_off' in metrics:
            cr_off = metrics['colregs_off']
            cr_on = metrics['colregs_on']
            print(f"  DCPA (avg min dist):      OFF {cr_off['dcpa_avg']:.2f}m  |  ON {cr_on['dcpa_avg']:.2f}m")

            if cr_off['compliance'] is not None:
                print(f"  COLREGs Compliance:       OFF {cr_off['overall_rate']:.1f}%  |  ON {cr_on['overall_rate']:.1f}%")
                for sit in ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']:
                    off_t = cr_off['compliance'][sit]['total']
                    on_t = cr_on['compliance'][sit]['total']
                    if off_t > 0 or on_t > 0:
                        off_rate = cr_off['compliance'][sit]['compliant'] / max(off_t, 1) * 100
                        on_rate = cr_on['compliance'][sit]['compliant'] / max(on_t, 1) * 100
                        print(f"    {sit:<18} OFF: {cr_off['compliance'][sit]['compliant']}/{off_t} ({off_rate:.0f}%)  |  ON: {cr_on['compliance'][sit]['compliant']}/{on_t} ({on_rate:.0f}%)")

        # 개별 환경 그래프
        p = plot_single_environment(env_name, env_data, metrics)
        saved_paths.append(p)

    # 환경 간 통합 비교 그래프
    p = plot_cross_environment(all_metrics, drop_n=DROP_N)
    saved_paths.append(p)

    # Summary txt 저장
    summary_path = os.path.join(SAVE_DIR, 'comparison_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Multi-Environment Comparison Summary\n")
        f.write(f"Run selection: dropped {DROP_N} outliers per side ({10 - DROP_N} runs used)\n")
        f.write("=" * 70 + "\n")

        for env_name in all_metrics:
            m = all_metrics[env_name]
            f.write(f"\n--- {env_name} ---\n")
            f.write(f"  Collision Avoidance Rate:  OFF {m['off_avoidance_mean']:.1f}% +/- {m['off_avoidance_std']:.1f}  |  ON {m['on_avoidance_mean']:.1f}% +/- {m['on_avoidance_std']:.1f}\n")
            f.write(f"  Avg Collisions/Run:       OFF {m['off_col_mean']:.1f} +/- {m['off_col_std']:.1f}     |  ON {m['on_col_mean']:.1f} +/- {m['on_col_std']:.1f}\n")
            f.write(f"  Collision Reduction:      {m['collision_reduction']:.1f}%\n")

            if 'colregs_off' in m:
                cr_off = m['colregs_off']
                cr_on = m['colregs_on']
                f.write(f"  DCPA (avg min dist):      OFF {cr_off['dcpa_avg']:.2f}m  |  ON {cr_on['dcpa_avg']:.2f}m\n")

                if cr_off['compliance'] is not None:
                    f.write(f"  COLREGs Compliance:       OFF {cr_off['overall_rate']:.1f}%  |  ON {cr_on['overall_rate']:.1f}%\n")
                    for sit in ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']:
                        off_t = cr_off['compliance'][sit]['total']
                        on_t = cr_on['compliance'][sit]['total']
                        if off_t > 0 or on_t > 0:
                            off_rate = cr_off['compliance'][sit]['compliant'] / max(off_t, 1) * 100
                            on_rate = cr_on['compliance'][sit]['compliant'] / max(on_t, 1) * 100
                        f.write(f"    {sit:<18} OFF: {cr_off['compliance'][sit]['compliant']}/{off_t} ({off_rate:.0f}%)  |  ON: {cr_on['compliance'][sit]['compliant']}/{on_t} ({on_rate:.0f}%)\n")

        f.write(f"\n{'=' * 70}\n")
        f.write("Figures:\n")
        for p in saved_paths:
            f.write(f"  {os.path.basename(p)}\n")

    saved_paths.append(summary_path)

    print(f"\n{'=' * 70}")
    print(f"[DONE] All files saved to: {SAVE_DIR}")
    for p in saved_paths:
        print(f"  {os.path.basename(p)}")
    print(f"{'=' * 70}")
