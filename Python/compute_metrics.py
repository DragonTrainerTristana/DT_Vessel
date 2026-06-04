"""
Vessel Navigation - 효율성 메트릭 종합 계산
6개 핵심 메트릭: Success Rate, Collision Rate, TTA, DCPA, Near-miss Rate, Smoothness
Open Ocean + Coastal: CSV에서 실측값 계산
Narrow Channel: 보간 추정값 (CSV 없음)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import pi

# ============================================================================
# 프로젝트 경로 설정
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
TRAJECTORY_DIR = os.path.join(PROJECT_ROOT, "trajectory_data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# 물리 상수 (Unity 환경과 동일)
# ============================================================================
MAX_SPEED = 5.0           # VesselDynamics.maxSpeed
MAX_MAP_DISTANCE = 1000.0 # VesselAgent.maxMapDistance
GOAL_REACHED_DIST = 15.0  # VesselAgent.goalReachedDistance (meters)
FIXED_DELTA_TIME = 0.02   # Unity FixedUpdate interval (seconds)
STEPS_PER_SECOND = int(1.0 / FIXED_DELTA_TIME)  # 50 steps/second
NUM_AGENTS = 8            # 에이전트 수
NEAR_MISS_THRESHOLD = 15.0  # 근접 조우 임계값 (meters) — 선박 길이 10m 고려

# ============================================================================
# 궤적 CSV 파일 경로
# ============================================================================
CSV_FILES = {
    'open_ocean': {
        'OFF': os.path.join(TRAJECTORY_DIR, "open_ocean_compare_commOFF_10x2000_20260319_015000.csv"),
        'ON':  os.path.join(TRAJECTORY_DIR, "open_ocean_compare_commON_10x2000_20260319_015000.csv"),
    },
    'coastal': {
        # 실제 파일명은 narrow_compare이지만 Coastal 데이터임
        'OFF': os.path.join(TRAJECTORY_DIR, "narrow_compare_commOFF_10x2000_20260318_163447.csv"),
        'ON':  os.path.join(TRAJECTORY_DIR, "narrow_compare_commON_10x2000_20260318_163447.csv"),
    },
}

# ============================================================================
# Raw 실험 통계 (test.py에서 수집, 10 runs x 2000 steps)
# ============================================================================
RAW_STATS = {
    'open_ocean': {
        'OFF': {
            'collisions': [2, 2, 3, 1, 3, 1, 0, 1, 1, 0],
            'successes':  [8, 10, 9, 12, 8, 5, 2, 8, 6, 8],
        },
        'ON': {
            'collisions': [1, 2, 0, 0, 2, 1, 0, 0, 2, 0],
            'successes':  [7, 9, 5, 5, 11, 14, 6, 9, 9, 9],
        },
    },
    'narrow': {
        'OFF': {
            'collisions': [8, 15, 21, 13, 17, 15, 4, 11, 10, 9],
            'successes':  [9, 11, 9, 6, 15, 14, 6, 8, 10, 8],
        },
        'ON': {
            'collisions': [8, 6, 9, 5, 10, 7, 8, 7, 10, 11],
            'successes':  [11, 7, 9, 10, 8, 16, 10, 9, 13, 16],
        },
    },
    'coastal': {
        'OFF': {
            'collisions': [33, 49, 36, 48, 35, 35, 37, 37, 41, 28],
            'successes':  [4, 1, 3, 1, 3, 2, 3, 5, 4, 1],
        },
        'ON': {
            'collisions': [18, 20, 22, 24, 19, 21, 16, 29, 18, 17],
            'successes':  [9, 10, 8, 10, 18, 10, 11, 9, 12, 6],
        },
    },
}

NUM_RUNS = 10
TEST_STEPS = 2000


# ============================================================================
# 데이터 로딩 및 전처리
# ============================================================================

def load_and_split_runs(csv_path):
    """CSV 로드 후 run별로 분리. 반환: list of DataFrames (per run, all agents)"""
    df = pd.read_csv(csv_path)

    # run_id 할당: agent 0의 step이 감소하는 지점이 run 경계
    agent0_mask = df['agent_id'] == 0
    agent0_idx = df.index[agent0_mask]
    agent0_steps = df.loc[agent0_idx, 'step'].values

    # run 경계 찾기 (step이 감소하는 지점)
    run_breaks = np.where(np.diff(agent0_steps) < 0)[0] + 1
    # agent0의 row index에서 실제 df index로 변환
    break_rows = agent0_idx[run_breaks].values

    # 모든 row에 run_id 할당
    run_id = np.zeros(len(df), dtype=int)
    for i, brk in enumerate(break_rows):
        run_id[brk:] = i + 1
    df['run_id'] = run_id

    runs = []
    for rid in range(NUM_RUNS):
        run_df = df[df['run_id'] == rid].copy()
        runs.append(run_df)

    return runs


def get_episodes_for_agent(run_df, agent_id):
    """한 run 내에서 특정 agent의 에피소드들을 분리.
    반환: list of DataFrames (per episode)"""
    agent_data = run_df[run_df['agent_id'] == agent_id].reset_index(drop=True)
    if len(agent_data) == 0:
        return []

    step_vals = agent_data['step'].values
    # step 간격이 1보다 큰 곳 = 에피소드 경계 (terminal 후 respawn)
    gaps = np.where(np.diff(step_vals) > 1)[0]

    episodes = []
    start = 0
    for gap_end in gaps:
        ep = agent_data.iloc[start:gap_end + 1].copy()
        if len(ep) >= 2:  # 최소 2 step 이상
            episodes.append(ep)
        start = gap_end + 1

    # 마지막 에피소드
    if start < len(agent_data):
        ep = agent_data.iloc[start:].copy()
        if len(ep) >= 2:
            episodes.append(ep)

    return episodes


# ============================================================================
# 메트릭 1, 2: Success Rate, Collision Rate (raw stats 사용)
# ============================================================================

def compute_rate_metrics(env_name):
    """raw stats에서 Success Rate, Collision Rate 계산"""
    results = {}
    for comm in ['OFF', 'ON']:
        stats = RAW_STATS[env_name][comm]
        successes = np.array(stats['successes'], dtype=float)
        collisions = np.array(stats['collisions'], dtype=float)

        results[comm] = {
            'success_rate_mean': np.mean(successes),
            'success_rate_std': np.std(successes),
            'collision_rate_mean': np.mean(collisions),
            'collision_rate_std': np.std(collisions),
        }
    return results


# ============================================================================
# 메트릭 3: Path Efficiency (경로 효율)
# ============================================================================

def compute_path_efficiency(episodes):
    """에피소드들의 경로 효율(Path Efficiency) 계산.
    성공 에피소드(마지막 goal_dist < threshold)만 대상.
    PE = straight_line_distance / actual_path_length
    PE = 1.0 이면 최적 직선 경로, 낮을수록 우회가 많음
    """
    pe_values = []
    success_threshold = GOAL_REACHED_DIST / MAX_MAP_DISTANCE + 0.06  # ~0.075 (관측 지연 고려)

    for ep in episodes:
        final_goal_dist = ep['goal_dist'].iloc[-1]
        if final_goal_dist > success_threshold:
            continue  # 성공 에피소드가 아님

        if len(ep) < 3:
            continue

        # 실제 이동 경로 길이
        x = ep['x'].values
        z = ep['z'].values
        dx = np.diff(x)
        dz = np.diff(z)
        segment_lengths = np.sqrt(dx**2 + dz**2)
        actual_path = np.sum(segment_lengths)

        # 직선 거리 (시작점 → 끝점)
        straight_dist = np.sqrt((x[-1] - x[0])**2 + (z[-1] - z[0])**2)

        if actual_path < 1.0:  # 거의 이동 안 한 경우
            continue

        pe = straight_dist / actual_path
        pe = min(pe, 1.0)  # 1.0 초과 방지
        pe_values.append(pe)

    return pe_values


# ============================================================================
# 메트릭 4, 5: DCPA, Near-miss Rate
# ============================================================================

def compute_dcpa_and_nearmiss(run_df, sample_interval=5):
    """한 run에서 COLREGs 조우가 있었던 에이전트 쌍의 DCPA와 Near-miss 계산.
    COLREGs 상호작용이 없는 먼 쌍은 제외 → 실질적 위험 조우만 평가.
    sample_interval: 성능을 위해 N step마다 샘플링
    """
    agents = sorted(run_df['agent_id'].unique())
    steps = sorted(run_df['step'].unique())
    sampled_steps = steps[::sample_interval]

    # step별 agent 위치 + colregs 상태 dict
    pos_by_step = {}
    colregs_by_step = {}
    for step in sampled_steps:
        step_data = run_df[run_df['step'] == step]
        pos = {}
        colregs = {}
        for _, row in step_data.iterrows():
            aid = int(row['agent_id'])
            pos[aid] = np.array([row['x'], row['z']])
            colregs[aid] = str(row.get('colregs_name', 'None'))
        pos_by_step[step] = pos
        colregs_by_step[step] = colregs

    # 에이전트 쌍별 최소 거리 추적 (COLREGs 조우가 있었던 쌍만)
    pair_min_dists = []

    agent_pairs = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent_pairs.append((agents[i], agents[j]))

    for a1, a2 in agent_pairs:
        encounter_dists = []
        had_colregs_interaction = False
        in_encounter = False

        for step in sampled_steps:
            pos = pos_by_step[step]
            cr = colregs_by_step[step]
            if a1 in pos and a2 in pos:
                dist = np.linalg.norm(pos[a1] - pos[a2])
                # 접근 중인 쌍만 (100m 이내 — 실질적 위험 조우)
                if dist < 100.0:
                    encounter_dists.append(dist)
                    in_encounter = True
                    # COLREGs 상호작용 확인 (둘 중 하나라도 None이 아니면)
                    if (a1 in cr and cr[a1] != 'None') or (a2 in cr and cr[a2] != 'None'):
                        had_colregs_interaction = True
            else:
                if in_encounter and len(encounter_dists) > 0:
                    # COLREGs 조우가 있었거나 가까이 접근한 쌍만 기록
                    if had_colregs_interaction or min(encounter_dists) < 50.0:
                        pair_min_dists.append(min(encounter_dists))
                    encounter_dists = []
                    had_colregs_interaction = False
                in_encounter = False

        # 마지막 encounter
        if len(encounter_dists) > 0:
            if had_colregs_interaction or min(encounter_dists) < 50.0:
                pair_min_dists.append(min(encounter_dists))

    if len(pair_min_dists) == 0:
        return 50.0, 0.0, 0  # 조우 없으면 안전 거리 반환

    avg_dcpa = np.mean(pair_min_dists)
    near_miss_count = sum(1 for d in pair_min_dists if d < NEAR_MISS_THRESHOLD)
    near_miss_rate = near_miss_count / len(pair_min_dists) * 100.0

    return avg_dcpa, near_miss_rate, len(pair_min_dists)


# ============================================================================
# 메트릭 6: Smoothness (heading 변화의 std)
# ============================================================================

def compute_smoothness_for_episodes(episodes):
    """에피소드별 heading 변화의 표준편차 계산.
    heading은 [-1, 1] (정규화된 값, heading/180).
    delta_heading = heading[t+1] - heading[t]
    smoothness = std(delta_heading) per episode
    """
    smoothness_values = []

    for ep in episodes:
        if len(ep) < 3:
            continue
        headings = ep['heading'].values
        delta_h = np.diff(headings)
        # heading wrap-around 처리 ([-1,1] 범위)
        delta_h = np.where(delta_h > 1.0, delta_h - 2.0, delta_h)
        delta_h = np.where(delta_h < -1.0, delta_h + 2.0, delta_h)
        smoothness_values.append(np.std(delta_h))

    return smoothness_values


# ============================================================================
# CSV 기반 메트릭 계산 (Open Ocean, Coastal)
# ============================================================================

def compute_csv_metrics(env_name, csv_key):
    """CSV에서 Path Efficiency, DCPA, Near-miss, Smoothness 계산"""
    results = {}
    raw_key = 'open_ocean' if csv_key == 'open_ocean' else 'coastal'

    for comm in ['OFF', 'ON']:
        csv_path = CSV_FILES[csv_key][comm]
        if not os.path.exists(csv_path):
            print(f"[경고] CSV 파일 없음: {csv_path}")
            results[comm] = None
            continue

        print(f"  [{env_name} {comm}] CSV 로딩: {os.path.basename(csv_path)}")
        runs = load_and_split_runs(csv_path)
        collision_per_run = RAW_STATS[raw_key][comm]['collisions']

        all_pe = []
        all_smoothness = []
        run_dcpa_means = []
        run_nearmiss_rates = []

        for rid, run_df in enumerate(runs):
            # 모든 에이전트의 에피소드 수집
            run_episodes = []
            for aid in run_df['agent_id'].unique():
                eps = get_episodes_for_agent(run_df, aid)
                run_episodes.extend(eps)

            # Path Efficiency
            pe_vals = compute_path_efficiency(run_episodes)
            all_pe.extend(pe_vals)

            # Smoothness
            smooth_vals = compute_smoothness_for_episodes(run_episodes)
            all_smoothness.extend(smooth_vals)

            # DCPA, Near-miss (run 단위)
            dcpa, nearmiss, n_encounters = compute_dcpa_and_nearmiss(run_df)

            # 충돌 = DCPA 0 포함 (물리적으로 충돌은 최소 접근 거리 0)
            n_collisions = collision_per_run[rid] if rid < len(collision_per_run) else 0
            if n_collisions > 0 and n_encounters > 0:
                total = n_encounters + n_collisions
                dcpa = (dcpa * n_encounters) / total
                nearmiss_count = (nearmiss / 100.0) * n_encounters + n_collisions
                nearmiss = nearmiss_count / total * 100.0

            run_dcpa_means.append(dcpa)
            run_nearmiss_rates.append(nearmiss)

        results[comm] = {
            'pe_mean': np.mean(all_pe) if all_pe else float('nan'),
            'pe_std': np.std(all_pe) if all_pe else float('nan'),
            'pe_n': len(all_pe),
            'dcpa_mean': np.mean(run_dcpa_means),
            'dcpa_std': np.std(run_dcpa_means),
            'nearmiss_mean': np.mean(run_nearmiss_rates),
            'nearmiss_std': np.std(run_nearmiss_rates),
            'smoothness_mean': np.mean(all_smoothness) if all_smoothness else float('nan'),
            'smoothness_std': np.std(all_smoothness) if all_smoothness else float('nan'),
        }

    return results


# ============================================================================
# Narrow Channel 추정 (보간법)
# ============================================================================

def estimate_narrow_metrics(open_ocean_metrics, coastal_metrics):
    """Narrow Channel 메트릭 추정.
    Narrow는 Open Ocean(쉬움)과 Coastal(어려움) 사이의 난이도.
    가중 보간으로 추정.
    """
    # 보간 가중치 (Narrow는 Coastal에 더 가까움)
    weights = {
        'pe':         (0.4, 0.6),  # (open_ocean, coastal)
        'dcpa':       (0.3, 0.7),
        'nearmiss':   (0.3, 0.7),
        'smoothness': (0.4, 0.6),
    }

    results = {}
    for comm in ['OFF', 'ON']:
        oo = open_ocean_metrics[comm]
        co = coastal_metrics[comm]

        if oo is None or co is None:
            results[comm] = None
            continue

        w_oo_pe, w_co_pe = weights['pe']
        w_oo_dcpa, w_co_dcpa = weights['dcpa']
        w_oo_nm, w_co_nm = weights['nearmiss']
        w_oo_sm, w_co_sm = weights['smoothness']

        results[comm] = {
            'pe_mean': w_oo_pe * oo['pe_mean'] + w_co_pe * co['pe_mean'],
            'pe_std': w_oo_pe * oo['pe_std'] + w_co_pe * co['pe_std'],
            'pe_n': 0,  # 추정값
            'dcpa_mean': w_oo_dcpa * oo['dcpa_mean'] + w_co_dcpa * co['dcpa_mean'],
            'dcpa_std': w_oo_dcpa * oo['dcpa_std'] + w_co_dcpa * co['dcpa_std'],
            'nearmiss_mean': w_oo_nm * oo['nearmiss_mean'] + w_co_nm * co['nearmiss_mean'],
            'nearmiss_std': w_oo_nm * oo['nearmiss_std'] + w_co_nm * co['nearmiss_std'],
            'smoothness_mean': w_oo_sm * oo['smoothness_mean'] + w_co_sm * co['smoothness_mean'],
            'smoothness_std': w_oo_sm * oo['smoothness_std'] + w_co_sm * co['smoothness_std'],
        }

    return results


# ============================================================================
# 결과 테이블 출력
# ============================================================================

def print_metrics_table(all_metrics):
    """종합 메트릭 테이블 출력"""
    envs = ['open_ocean', 'narrow', 'coastal']
    env_labels = {
        'open_ocean': 'Open Ocean',
        'narrow':     'Narrow Channel*',
        'coastal':    'Coastal',
    }

    print("\n" + "=" * 95)
    print("                    종합 효율성 메트릭 비교 (COMM OFF vs ON)")
    print("=" * 95)

    # 헤더
    header = f"{'Metric':<22}"
    for env in envs:
        label = env_labels[env]
        header += f"  {label:^26}"
    print(header)

    sub_header = f"{'':22}"
    for _ in envs:
        sub_header += f"  {'OFF':>11}  {'ON':>11}  "
    print(sub_header)
    print("-" * 95)

    # 메트릭 행
    metric_rows = [
        ('Success Rate',      'rate', 'success_rate_mean', 'success_rate_std', '{:.1f}', '+-{:.1f}'),
        ('Collision Rate',    'rate', 'collision_rate_mean', 'collision_rate_std', '{:.1f}', '+-{:.1f}'),
        ('Path Efficiency',   'csv', 'pe_mean', 'pe_std', '{:.3f}', '+-{:.3f}'),
        ('Avg DCPA (m)',      'csv', 'dcpa_mean', 'dcpa_std', '{:.1f}', '+-{:.1f}'),
        ('Near-miss (%)',     'csv', 'nearmiss_mean', 'nearmiss_std', '{:.1f}', '+-{:.1f}'),
        ('Smoothness',        'csv', 'smoothness_mean', 'smoothness_std', '{:.4f}', '+-{:.4f}'),
    ]

    for name, src, mean_key, std_key, fmt_m, fmt_s in metric_rows:
        row = f"{name:<22}"
        for env in envs:
            for comm in ['OFF', 'ON']:
                if src == 'rate':
                    data = all_metrics[env]['rate'][comm]
                else:
                    data = all_metrics[env]['csv'][comm]

                if data is None:
                    row += f"  {'N/A':>11}"
                else:
                    val = fmt_m.format(data[mean_key])
                    std = fmt_s.format(data[std_key])
                    row += f"  {val:>6}{std:>7}"
        print(row)

    print("-" * 95)
    print("* Narrow Channel: CSV not available, estimated via weighted interpolation")
    print(f"  Near-miss threshold: {NEAR_MISS_THRESHOLD}m")
    print("  Path Efficiency = straight_line_dist / actual_path_length (1.0 = optimal)")
    print()

    # 개선율 요약
    print("=" * 95)
    print("                    Communication 효과 요약 (OFF → ON 변화)")
    print("=" * 95)

    improvement_rows = [
        ('Success Rate',   'rate', 'success_rate_mean', True,  '{:+.1f}', '(higher=better)'),
        ('Collision Rate', 'rate', 'collision_rate_mean', False, '{:+.1f}', '(lower=better) '),
        ('Path Efficiency','csv',  'pe_mean', True, '{:+.3f}', '(higher=better)'),
        ('Avg DCPA',       'csv',  'dcpa_mean', True,  '{:+.1f}', '(higher=better)'),
        ('Near-miss',      'csv',  'nearmiss_mean', False, '{:+.1f}', '(lower=better) '),
        ('Smoothness',     'csv',  'smoothness_mean', False, '{:+.4f}', '(lower=better) '),
    ]

    for name, src, key, higher_better, fmt, note in improvement_rows:
        row = f"  {name:<20} {note:<14}"
        for env in envs:
            if src == 'rate':
                off_data = all_metrics[env]['rate']['OFF']
                on_data = all_metrics[env]['rate']['ON']
            else:
                off_data = all_metrics[env]['csv']['OFF']
                on_data = all_metrics[env]['csv']['ON']

            if off_data is None or on_data is None:
                row += f"  {'N/A':>12}"
            else:
                diff = on_data[key] - off_data[key]
                improved = (diff > 0) if higher_better else (diff < 0)
                marker = " [+]" if improved else " [-]"
                row += f"  {fmt.format(diff):>9}{marker}"
        print(row)

    print()


# ============================================================================
# 레이더 차트 생성
# ============================================================================

def create_radar_chart(all_metrics):
    """6개 메트릭의 레이더 차트 생성 (3환경 × 2조건)"""

    categories = ['Success\nRate', 'Safety\n(1/Collision)', 'Path\nEfficiency',
                  'DCPA', 'Safety\n(1/Near-miss)', 'Smoothness\n(1/std)']
    N = len(categories)

    # 각 메트릭을 [0, 1]로 정규화 (높을수록 좋게 변환)
    envs = ['open_ocean', 'narrow', 'coastal']
    comms = ['OFF', 'ON']

    # raw 값 수집
    raw_values = {}
    for env in envs:
        for comm in comms:
            rate = all_metrics[env]['rate'][comm]
            csv_data = all_metrics[env]['csv'][comm]

            success = rate['success_rate_mean']
            collision = rate['collision_rate_mean']
            pe = csv_data['pe_mean'] if csv_data else 0.5
            dcpa = csv_data['dcpa_mean'] if csv_data else 50.0
            nearmiss = csv_data['nearmiss_mean'] if csv_data else 50.0
            smoothness = csv_data['smoothness_mean'] if csv_data else 0.1

            # 높을수록 좋게 변환
            raw_values[(env, comm)] = [
                success,                                    # Success Rate (높을수록 좋음)
                1.0 / (collision + 1),                      # Safety (낮은 collision = 좋음)
                pe,                                         # Path Efficiency (높을수록 좋음)
                dcpa,                                       # DCPA (높을수록 안전)
                1.0 / max(nearmiss + 1, 1),                 # Safety (낮은 near-miss = 좋음)
                1.0 / max(smoothness, 0.0001),              # Smoothness (낮은 std = 좋음)
            ]

    # 각 카테고리에서 min-max 정규화
    all_vals_by_cat = [[] for _ in range(N)]
    for key, vals in raw_values.items():
        for i, v in enumerate(vals):
            all_vals_by_cat[i].append(v)

    normalized = {}
    for key, vals in raw_values.items():
        norm = []
        for i, v in enumerate(vals):
            cat_min = min(all_vals_by_cat[i])
            cat_max = max(all_vals_by_cat[i])
            if cat_max - cat_min > 1e-10:
                norm.append(0.1 + 0.9 * (v - cat_min) / (cat_max - cat_min))
            else:
                norm.append(0.5)
        normalized[key] = norm

    # 차트 생성
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # 닫기

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    fig.suptitle('Efficiency Metrics Radar Chart: COMM OFF vs ON', fontsize=16, fontweight='bold', y=1.02)

    env_labels = {'open_ocean': 'Open Ocean', 'narrow': 'Narrow Channel*', 'coastal': 'Coastal'}
    colors = {'OFF': '#2196F3', 'ON': '#FF5722'}

    for idx, env in enumerate(envs):
        ax = axes[idx]

        for comm in comms:
            vals = normalized[(env, comm)] + normalized[(env, comm)][:1]
            ax.plot(angles, vals, 'o-', linewidth=2, label=f'COMM {comm}',
                    color=colors[comm], markersize=5)
            ax.fill(angles, vals, alpha=0.15, color=colors[comm])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7)
        ax.set_title(env_labels[env], fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "efficiency_radar_chart.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[저장] 레이더 차트: {save_path}")

    save_path_pdf = os.path.join(FIGURES_DIR, "efficiency_radar_chart.pdf")
    fig.savefig(save_path_pdf, bbox_inches='tight')
    print(f"[저장] 레이더 차트 PDF: {save_path_pdf}")

    plt.close(fig)


# ============================================================================
# 추가: 바 차트 (메트릭별 비교)
# ============================================================================

def create_bar_charts(all_metrics):
    """메트릭별 바 차트 (OFF vs ON, 3환경)"""
    envs = ['open_ocean', 'narrow', 'coastal']
    env_labels = ['Open Ocean', 'Narrow*', 'Coastal']
    x = np.arange(len(envs))
    width = 0.35

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Efficiency Metrics: COMM OFF vs ON across Environments', fontsize=14, fontweight='bold')

    metrics_info = [
        ('Success Rate (per run)', 'rate', 'success_rate_mean', 'success_rate_std', True),
        ('Collision Rate (per run)', 'rate', 'collision_rate_mean', 'collision_rate_std', False),
        ('Path Efficiency', 'csv', 'pe_mean', 'pe_std', True),
        ('Avg DCPA (m)', 'csv', 'dcpa_mean', 'dcpa_std', True),
        ('Near-miss Rate (%)', 'csv', 'nearmiss_mean', 'nearmiss_std', False),
        ('Smoothness (heading std)', 'csv', 'smoothness_mean', 'smoothness_std', False),
    ]

    colors_off = '#4A90D9'
    colors_on = '#E8694A'

    for idx, (title, src, mean_key, std_key, higher_better) in enumerate(metrics_info):
        ax = axes[idx // 3, idx % 3]

        off_vals, off_errs = [], []
        on_vals, on_errs = [], []

        for env in envs:
            if src == 'rate':
                off_data = all_metrics[env]['rate']['OFF']
                on_data = all_metrics[env]['rate']['ON']
            else:
                off_data = all_metrics[env]['csv']['OFF']
                on_data = all_metrics[env]['csv']['ON']

            off_vals.append(off_data[mean_key] if off_data else 0)
            off_errs.append(off_data[std_key] if off_data else 0)
            on_vals.append(on_data[mean_key] if on_data else 0)
            on_errs.append(on_data[std_key] if on_data else 0)

        bars_off = ax.bar(x - width / 2, off_vals, width, yerr=off_errs,
                         label='COMM OFF', color=colors_off, alpha=0.85, capsize=3)
        bars_on = ax.bar(x + width / 2, on_vals, width, yerr=on_errs,
                        label='COMM ON', color=colors_on, alpha=0.85, capsize=3)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(env_labels, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Narrow에 '*' 표시
        # (이미 env_labels에 포함)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "efficiency_bar_charts.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[저장] 바 차트: {save_path}")

    save_path_pdf = os.path.join(FIGURES_DIR, "efficiency_bar_charts.pdf")
    fig.savefig(save_path_pdf, bbox_inches='tight')
    print(f"[저장] 바 차트 PDF: {save_path_pdf}")

    plt.close(fig)


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("=" * 70)
    print("  Vessel Navigation - 효율성 메트릭 종합 계산")
    print("=" * 70)

    all_metrics = {}

    # ------------------------------------------------------------------
    # Part 1: Open Ocean (실측)
    # ------------------------------------------------------------------
    print("\n[1/3] Open Ocean 메트릭 계산...")
    all_metrics['open_ocean'] = {
        'rate': compute_rate_metrics('open_ocean'),
        'csv': compute_csv_metrics('Open Ocean', 'open_ocean'),
    }

    # ------------------------------------------------------------------
    # Part 2: Coastal (실측)
    # ------------------------------------------------------------------
    print("\n[2/3] Coastal 메트릭 계산...")
    all_metrics['coastal'] = {
        'rate': compute_rate_metrics('coastal'),
        'csv': compute_csv_metrics('Coastal', 'coastal'),
    }

    # ------------------------------------------------------------------
    # Part 3: Narrow Channel (추정)
    # ------------------------------------------------------------------
    print("\n[3/3] Narrow Channel 메트릭 추정 (보간)...")
    all_metrics['narrow'] = {
        'rate': compute_rate_metrics('narrow'),
        'csv': estimate_narrow_metrics(
            all_metrics['open_ocean']['csv'],
            all_metrics['coastal']['csv'],
        ),
    }

    # ------------------------------------------------------------------
    # Part 4: 결과 출력
    # ------------------------------------------------------------------
    print_metrics_table(all_metrics)

    # ------------------------------------------------------------------
    # Part 5: 그래프 생성
    # ------------------------------------------------------------------
    print("\n[그래프 생성]")
    create_radar_chart(all_metrics)
    create_bar_charts(all_metrics)

    # ------------------------------------------------------------------
    # Part 6: 논문용 LaTeX 테이블 출력
    # ------------------------------------------------------------------
    print_latex_table(all_metrics)

    print("\n[완료] 모든 메트릭 계산 및 그래프 생성 완료")


def print_latex_table(all_metrics):
    """논문용 LaTeX 테이블 출력"""
    print("\n" + "=" * 70)
    print("  LaTeX Table (논문용)")
    print("=" * 70)
    print()
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Efficiency metrics across three environments (COMM OFF vs ON)}")
    print(r"\label{tab:efficiency_metrics}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{l|cc|cc|cc}")
    print(r"\hline")
    print(r" & \multicolumn{2}{c|}{Open Ocean} & \multicolumn{2}{c|}{Narrow Channel$^\dagger$} & \multicolumn{2}{c}{Coastal} \\")
    print(r"Metric & OFF & ON & OFF & ON & OFF & ON \\")
    print(r"\hline")

    envs = ['open_ocean', 'narrow', 'coastal']
    rows = [
        ('Success Rate',    'rate', 'success_rate_mean',  '{:.1f}'),
        ('Collision Rate',  'rate', 'collision_rate_mean', '{:.1f}'),
        ('Path Eff.',       'csv',  'pe_mean',            '{:.3f}'),
        ('Avg DCPA (m)',    'csv',  'dcpa_mean',          '{:.1f}'),
        ('Near-miss (\\%)', 'csv',  'nearmiss_mean',      '{:.1f}'),
        ('Smoothness',      'csv',  'smoothness_mean',    '{:.4f}'),
    ]

    for name, src, key, fmt in rows:
        parts = [name]
        for env in envs:
            for comm in ['OFF', 'ON']:
                if src == 'rate':
                    data = all_metrics[env]['rate'][comm]
                else:
                    data = all_metrics[env]['csv'][comm]
                if data:
                    parts.append(fmt.format(data[key]))
                else:
                    parts.append('--')
        print(" & ".join(parts) + r" \\")

    print(r"\hline")
    print(r"\end{tabular}}")
    print(r"\begin{tablenotes}")
    print(r"\small")
    print(r"\item[$\dagger$] Narrow Channel metrics (Path Eff., DCPA, Near-miss, Smoothness) are estimated via weighted interpolation.")
    print(r"\end{tablenotes}")
    print(r"\end{table}")
    print()


if __name__ == "__main__":
    main()
