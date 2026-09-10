"""
Vessel Navigation Efficiency & Safety Analysis
3환경(Open Ocean, Narrow Channel, Coastal) × COMM ON/OFF 비교 분석

Metrics:
  A. Safety: COLREGs compliance, DCPA, Average vessels in radar range
  B. Efficiency: Fuel consumption, Episode time, Smoothness
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from itertools import combinations

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 경로 설정
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
TRAJ_DIR = os.path.join(PROJECT_ROOT, "trajectory_data")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# 상수
# ============================================================================
RADAR_RANGE = 200.0       # 레이더 범위 (미터)
NUM_RUNS = 10             # 실험 반복 횟수
STEPS_PER_RUN = 2000      # 런 당 스텝 수
SUCCESS_GOAL_DIST = 0.075 # 성공 판정 (goal_dist < 15m / 200m maxMapDistance)
NUM_AGENTS = 8            # 에이전트 수 (환경당)

# ============================================================================
# 데이터 파일 매핑
# ============================================================================
CSV_FILES = {
    'Open Ocean': {
        'OFF': os.path.join(TRAJ_DIR, 'open_ocean_compare_commOFF_10x2000_20260319_015000.csv'),
        'ON':  os.path.join(TRAJ_DIR, 'open_ocean_compare_commON_10x2000_20260319_015000.csv'),
    },
    'Coastal': {
        # narrow_compare 파일이 실제로는 Coastal 데이터 (파일명 오류)
        'OFF': os.path.join(TRAJ_DIR, 'narrow_compare_commOFF_10x2000_20260318_163447.csv'),
        'ON':  os.path.join(TRAJ_DIR, 'narrow_compare_commON_10x2000_20260318_163447.csv'),
    },
}

# Narrow Channel: CSV 없음, raw 통계만 존재
NARROW_RAW = {
    'OFF': {
        'collisions': [8, 15, 21, 13, 17, 15, 4, 11, 10, 9],
        'successes':  [9, 11, 9, 6, 15, 14, 6, 8, 10, 8],
    },
    'ON': {
        'collisions': [8, 6, 9, 5, 10, 7, 8, 7, 10, 11],
        'successes':  [11, 7, 9, 10, 8, 16, 10, 9, 13, 16],
    },
}


# ============================================================================
# CSV 로딩 및 런 분할
# ============================================================================
def load_csv(filepath):
    """CSV 파일 로드 후 DataFrame 반환"""
    if not os.path.exists(filepath):
        print(f"[경고] 파일 없음: {filepath}")
        return None
    df = pd.read_csv(filepath)
    return df


def split_into_runs(df, num_runs=NUM_RUNS):
    """step 기준으로 런 분할. step이 0으로 리셋되는 지점을 기준으로 분할."""
    # step이 감소하는 지점 = 새로운 런 시작
    step_diffs = df['step'].diff()
    run_starts = df.index[step_diffs < 0].tolist()
    run_starts = [0] + run_starts

    runs = []
    for i in range(len(run_starts)):
        start = run_starts[i]
        end = run_starts[i + 1] if i + 1 < len(run_starts) else len(df)
        run_df = df.iloc[start:end].copy()
        run_df = run_df.reset_index(drop=True)
        runs.append(run_df)

    # 정확히 num_runs개가 아닐 수 있으므로 조정
    if len(runs) > num_runs:
        runs = runs[:num_runs]
    return runs


# ============================================================================
# A. Safety Metrics
# ============================================================================

def compute_colregs_compliance(run_df):
    """
    COLREGs 상황별 준수율 계산
    - HeadOn: rudder > 0 (우현 회피) = 준수
    - GiveWay (CrossGiveWay): rudder > 0 = 준수
    - StandOn (CrossStandOn): |rudder| < 0.2 (침로 유지) = 준수
    - Overtaking: |rudder| > 0.05 (회피 기동) = 준수
    """
    results = {}
    situations = {
        'HeadOn': lambda r: r > 0,
        'CrossStandOn': lambda r: abs(r) < 0.2,
        'CrossGiveWay': lambda r: r > 0,
        'Overtaking': lambda r: abs(r) > 0.05,
    }

    for sit_name, compliance_fn in situations.items():
        mask = run_df['colregs_name'] == sit_name
        sit_data = run_df.loc[mask]
        if len(sit_data) == 0:
            results[sit_name] = np.nan
            continue
        compliant = sit_data['rudder'].apply(compliance_fn).sum()
        results[sit_name] = compliant / len(sit_data)

    # 전체 COLREGs 준수율 (None 제외)
    colregs_mask = run_df['colregs_name'] != 'None'
    colregs_data = run_df.loc[colregs_mask]
    if len(colregs_data) == 0:
        results['Overall'] = np.nan
    else:
        total_compliant = 0
        total_count = 0
        for sit_name, compliance_fn in situations.items():
            mask = colregs_data['colregs_name'] == sit_name
            sit_subset = colregs_data.loc[mask]
            if len(sit_subset) > 0:
                total_compliant += sit_subset['rudder'].apply(compliance_fn).sum()
                total_count += len(sit_subset)
        results['Overall'] = total_compliant / total_count if total_count > 0 else np.nan

    return results


def compute_dcpa(run_df):
    """
    DCPA (Distance at Closest Point of Approach) 계산 - 벡터화 버전
    각 step에서 모든 에이전트 쌍의 거리를 계산하고,
    레이더 범위 내 조우(encounter)의 최소 거리를 추적
    """
    # step별 에이전트 위치를 피벗 테이블로 변환
    pivot_x = run_df.pivot_table(index='step', columns='agent_id', values='x', aggfunc='first')
    pivot_z = run_df.pivot_table(index='step', columns='agent_id', values='z', aggfunc='first')
    agents = sorted(pivot_x.columns)

    if len(agents) < 2:
        return np.nan, 0

    # 모든 에이전트 쌍에 대해 거리 시계열 계산
    encounter_min_dists = []
    for i_idx in range(len(agents)):
        for j_idx in range(i_idx + 1, len(agents)):
            ai, aj = agents[i_idx], agents[j_idx]
            dx = pivot_x[ai].values - pivot_x[aj].values
            dz = pivot_z[ai].values - pivot_z[aj].values
            dists = np.sqrt(dx**2 + dz**2)

            # NaN 제거 (한쪽 에이전트가 해당 step에 없는 경우)
            valid = ~np.isnan(dists)
            dists = dists[valid]
            if len(dists) == 0:
                continue

            # 레이더 범위 내 구간(encounter) 식별
            in_range = dists < RADAR_RANGE
            # 구간 경계 찾기: diff로 0→1(진입), 1→0(이탈) 감지
            changes = np.diff(in_range.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            # 처음부터 범위 내인 경우
            if in_range[0]:
                starts = np.concatenate([[0], starts])
            # 끝까지 범위 내인 경우
            if in_range[-1]:
                ends = np.concatenate([ends, [len(dists)]])

            for s, e in zip(starts, ends):
                encounter_min_dists.append(np.min(dists[s:e]))

    if len(encounter_min_dists) == 0:
        return np.nan, 0
    return np.mean(encounter_min_dists), len(encounter_min_dists)


def compute_avg_vessels_in_radar(run_df):
    """각 에이전트의 각 step에서 레이더 범위 내 다른 에이전트 수의 평균 - 벡터화 버전"""
    pivot_x = run_df.pivot_table(index='step', columns='agent_id', values='x', aggfunc='first')
    pivot_z = run_df.pivot_table(index='step', columns='agent_id', values='z', aggfunc='first')
    agents = sorted(pivot_x.columns)

    if len(agents) < 2:
        return 0.0

    # 각 에이전트 쌍의 거리 행렬을 step별로 계산
    # (n_steps, n_agents) 행렬
    x_vals = pivot_x[agents].values  # (n_steps, n_agents)
    z_vals = pivot_z[agents].values
    n_steps, n_agents = x_vals.shape

    # 각 에이전트별 레이더 범위 내 이웃 수 합산
    total_counts = 0
    total_valid = 0

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            dx = x_vals[:, i] - x_vals[:, j]
            dz = z_vals[:, i] - z_vals[:, j]
            dists = np.sqrt(dx**2 + dz**2)
            valid = ~np.isnan(dists)
            in_range = (dists < RADAR_RANGE) & valid
            # 양쪽 에이전트 모두에게 이웃 1씩 카운트
            total_counts += in_range.sum() * 2
            total_valid += valid.sum() * 2

    # 에이전트당 step당 평균 이웃 수
    return total_counts / (n_steps * n_agents) if (n_steps * n_agents) > 0 else 0.0


# ============================================================================
# B. Efficiency Metrics
# ============================================================================

def identify_episodes(run_df):
    """
    에이전트별 에피소드를 식별.
    goal_dist가 SUCCESS_GOAL_DIST 이하로 떨어지면 성공 에피소드로 판정.
    에이전트의 goal_dist가 급격히 증가하면 새 에피소드 시작으로 간주.
    """
    episodes = []
    for aid in run_df['agent_id'].unique():
        agent_data = run_df[run_df['agent_id'] == aid].sort_values('step').reset_index(drop=True)
        if len(agent_data) == 0:
            continue

        # goal_dist가 크게 증가하는 지점 = 새 에피소드 시작
        goal_diffs = agent_data['goal_dist'].diff()
        ep_starts = [0] + agent_data.index[goal_diffs > 0.3].tolist()

        for i in range(len(ep_starts)):
            start = ep_starts[i]
            end = ep_starts[i + 1] if i + 1 < len(ep_starts) else len(agent_data)
            ep_data = agent_data.iloc[start:end].copy()

            # 성공 판정
            success = (ep_data['goal_dist'].min() < SUCCESS_GOAL_DIST)
            episodes.append({
                'agent_id': aid,
                'data': ep_data,
                'success': success,
                'steps': len(ep_data),
            })

    return episodes


def compute_fuel(ep_data):
    """
    연료 소비 프록시: fuel = Σ(|speed| + 0.5 * |Δrudder|)
    speed, rudder는 정규화된 값 [-1, 1]
    """
    speed_cost = np.abs(ep_data['speed'].values).sum()
    rudder_changes = np.abs(np.diff(ep_data['rudder'].values))
    rudder_cost = 0.5 * rudder_changes.sum()
    return speed_cost + rudder_cost


def compute_smoothness(ep_data):
    """
    경로 부드러움: std(Δheading)
    heading은 정규화된 값 [-1, 1]
    """
    heading_changes = np.diff(ep_data['heading'].values)
    # heading wrapping 처리 (-1 ~ 1 범위)
    heading_changes = np.where(heading_changes > 1, heading_changes - 2, heading_changes)
    heading_changes = np.where(heading_changes < -1, heading_changes + 2, heading_changes)
    return np.std(heading_changes) if len(heading_changes) > 0 else np.nan


# ============================================================================
# 전체 분석 실행
# ============================================================================

def analyze_environment(env_name, comm_label, csv_path):
    """한 환경-통신 조건에 대한 전체 메트릭 계산"""
    df = load_csv(csv_path)
    if df is None:
        return None

    runs = split_into_runs(df)
    print(f"  [{env_name} COMM {comm_label}] {len(runs)} runs 로드 완료")

    # 런별 메트릭 수집
    all_colregs = []
    all_dcpa = []
    all_dcpa_count = []
    all_radar_count = []
    all_fuel = []
    all_episode_time = []
    all_smoothness = []

    for run_idx, run_df in enumerate(runs):
        # A. Safety
        colregs = compute_colregs_compliance(run_df)
        all_colregs.append(colregs)

        dcpa_mean, dcpa_n = compute_dcpa(run_df)
        all_dcpa.append(dcpa_mean)
        all_dcpa_count.append(dcpa_n)

        radar_count = compute_avg_vessels_in_radar(run_df)
        all_radar_count.append(radar_count)

        # B. Efficiency
        episodes = identify_episodes(run_df)
        success_episodes = [ep for ep in episodes if ep['success']]

        run_fuels = []
        run_times = []
        run_smoothness = []
        for ep in success_episodes:
            run_fuels.append(compute_fuel(ep['data']))
            run_times.append(ep['steps'])
            run_smoothness.append(compute_smoothness(ep['data']))

        all_fuel.append(np.mean(run_fuels) if run_fuels else np.nan)
        all_episode_time.append(np.mean(run_times) if run_times else np.nan)
        all_smoothness.append(np.mean(run_smoothness) if run_smoothness else np.nan)

        print(f"    Run {run_idx+1}: COLREGs={colregs.get('Overall', 0):.3f}, "
              f"DCPA={dcpa_mean:.1f}m, Radar={radar_count:.2f}, "
              f"Fuel={all_fuel[-1]:.1f}, Steps={all_episode_time[-1]:.0f}, "
              f"Smooth={all_smoothness[-1]:.4f}, "
              f"Episodes={len(episodes)}(success={len(success_episodes)})")

    # 평균/표준편차 집계
    result = {
        'colregs_overall': (np.nanmean([c['Overall'] for c in all_colregs]),
                            np.nanstd([c['Overall'] for c in all_colregs])),
        'colregs_headon': (np.nanmean([c.get('HeadOn', np.nan) for c in all_colregs]),
                           np.nanstd([c.get('HeadOn', np.nan) for c in all_colregs])),
        'colregs_standon': (np.nanmean([c.get('CrossStandOn', np.nan) for c in all_colregs]),
                            np.nanstd([c.get('CrossStandOn', np.nan) for c in all_colregs])),
        'colregs_giveway': (np.nanmean([c.get('CrossGiveWay', np.nan) for c in all_colregs]),
                            np.nanstd([c.get('CrossGiveWay', np.nan) for c in all_colregs])),
        'colregs_overtaking': (np.nanmean([c.get('Overtaking', np.nan) for c in all_colregs]),
                               np.nanstd([c.get('Overtaking', np.nan) for c in all_colregs])),
        'dcpa': (np.nanmean(all_dcpa), np.nanstd(all_dcpa)),
        'dcpa_encounters': (np.mean(all_dcpa_count), np.std(all_dcpa_count)),
        'radar_count': (np.mean(all_radar_count), np.std(all_radar_count)),
        'fuel': (np.nanmean(all_fuel), np.nanstd(all_fuel)),
        'episode_time': (np.nanmean(all_episode_time), np.nanstd(all_episode_time)),
        'smoothness': (np.nanmean(all_smoothness), np.nanstd(all_smoothness)),
        'success_rate': None,   # CSV 기반 환경에서는 에피소드 분석으로 계산
        'collision_rate': None,
    }

    return result


def analyze_narrow_channel():
    """Narrow Channel: raw 통계만으로 분석"""
    results = {}
    for comm_label in ['OFF', 'ON']:
        raw = NARROW_RAW[comm_label]
        collisions = np.array(raw['collisions'])
        successes = np.array(raw['successes'])
        # 총 에피소드 추정 (collision + success + timeout 등)
        # 에이전트 8개 × 런당 여러 에피소드. 정확한 총수 모르므로 collision+success 기준
        total_episodes = collisions + successes + 5  # timeout 에피소드 추정치

        collision_rate = collisions / total_episodes
        success_rate = successes / total_episodes

        results[comm_label] = {
            'collision_rate': (np.mean(collision_rate), np.std(collision_rate)),
            'success_rate': (np.mean(success_rate), np.std(success_rate)),
            'collisions': (np.mean(collisions), np.std(collisions)),
            'successes': (np.mean(successes), np.std(successes)),
        }

    return results


# ============================================================================
# 출력: 콘솔 테이블
# ============================================================================

def print_results_table(all_results, narrow_results):
    """분석 결과를 포맷된 테이블로 출력"""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("=" * 100)

    # A. Safety Metrics
    print("\n--- A. SAFETY METRICS ---")
    print(f"{'Metric':<25} {'Open Ocean OFF':>18} {'Open Ocean ON':>18} {'Coastal OFF':>18} {'Coastal ON':>18}")
    print("-" * 100)

    for metric_key, label in [
        ('colregs_overall', 'COLREGs Overall'),
        ('colregs_headon', '  HeadOn'),
        ('colregs_standon', '  StandOn'),
        ('colregs_giveway', '  GiveWay'),
        ('colregs_overtaking', '  Overtaking'),
        ('dcpa', 'DCPA (m)'),
        ('radar_count', 'Avg Vessels in Radar'),
    ]:
        vals = []
        for env in ['Open Ocean', 'Coastal']:
            for comm in ['OFF', 'ON']:
                key = f"{env}_{comm}"
                if key in all_results and all_results[key] is not None:
                    m, s = all_results[key][metric_key]
                    if 'colregs' in metric_key:
                        vals.append(f"{m*100:.1f}±{s*100:.1f}%")
                    elif metric_key == 'dcpa':
                        vals.append(f"{m:.1f}±{s:.1f}")
                    else:
                        vals.append(f"{m:.2f}±{s:.2f}")
                else:
                    vals.append("N/A")
        print(f"{label:<25} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18} {vals[3]:>18}")

    # Narrow Channel
    print(f"\n{'Narrow Channel':<25} {'COMM OFF':>18} {'COMM ON':>18}")
    print("-" * 65)
    for metric_key, label in [
        ('collision_rate', 'Collision Rate'),
        ('success_rate', 'Success Rate'),
        ('collisions', 'Collisions (raw)'),
        ('successes', 'Successes (raw)'),
    ]:
        vals = []
        for comm in ['OFF', 'ON']:
            m, s = narrow_results[comm][metric_key]
            if 'rate' in metric_key:
                vals.append(f"{m*100:.1f}±{s*100:.1f}%")
            else:
                vals.append(f"{m:.1f}±{s:.1f}")
        print(f"{label:<25} {vals[0]:>18} {vals[1]:>18}")

    # B. Efficiency Metrics
    print("\n--- B. EFFICIENCY METRICS ---")
    print(f"{'Metric':<25} {'Open Ocean OFF':>18} {'Open Ocean ON':>18} {'Coastal OFF':>18} {'Coastal ON':>18}")
    print("-" * 100)

    for metric_key, label, fmt in [
        ('fuel', 'Fuel Consumption', '.1f'),
        ('episode_time', 'Episode Time (steps)', '.0f'),
        ('smoothness', 'Smoothness (σ Δhdg)', '.4f'),
    ]:
        vals = []
        for env in ['Open Ocean', 'Coastal']:
            for comm in ['OFF', 'ON']:
                key = f"{env}_{comm}"
                if key in all_results and all_results[key] is not None:
                    m, s = all_results[key][metric_key]
                    vals.append(f"{m:{fmt}}±{s:{fmt}}")
                else:
                    vals.append("N/A")
        print(f"{label:<25} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18} {vals[3]:>18}")


# ============================================================================
# 출력: LaTeX 테이블
# ============================================================================

def save_latex_table(all_results, narrow_results):
    """LaTeX 형식 테이블 저장"""
    filepath = os.path.join(FIG_DIR, "efficiency_analysis_table.tex")

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comprehensive Performance Analysis: COMM OFF vs COMM ON}")
    lines.append(r"\label{tab:efficiency}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l|cc|cc|cc}")
    lines.append(r"\hline")
    lines.append(r"\multirow{2}{*}{\textbf{Metric}} & \multicolumn{2}{c|}{\textbf{Open Ocean}} & \multicolumn{2}{c|}{\textbf{Coastal}} & \multicolumn{2}{c}{\textbf{Narrow Channel}} \\")
    lines.append(r" & OFF & ON & OFF & ON & OFF & ON \\")
    lines.append(r"\hline\hline")

    def fmt_val(result_dict, metric_key, fmt='.1f', pct=False):
        if result_dict is None:
            return "---"
        m, s = result_dict[metric_key]
        if np.isnan(m):
            return "---"
        if pct:
            return f"${m*100:{fmt}}\\pm{s*100:{fmt}}$\\%"
        return f"${m:{fmt}}\\pm{s:{fmt}}$"

    # Safety metrics
    lines.append(r"\multicolumn{7}{l}{\textbf{A. Safety Metrics}} \\")
    lines.append(r"\hline")

    safety_metrics = [
        ('colregs_overall', 'COLREGs Compliance (\\%)', '.1f', True),
        ('dcpa', 'DCPA (m)', '.1f', False),
        ('radar_count', 'Avg. Vessels in Radar', '.2f', False),
    ]

    for key, label, fmt, pct in safety_metrics:
        oo_off = fmt_val(all_results.get('Open Ocean_OFF'), key, fmt, pct)
        oo_on = fmt_val(all_results.get('Open Ocean_ON'), key, fmt, pct)
        co_off = fmt_val(all_results.get('Coastal_OFF'), key, fmt, pct)
        co_on = fmt_val(all_results.get('Coastal_ON'), key, fmt, pct)
        # Narrow: 일부 메트릭만 있음
        if key == 'colregs_overall':
            nc_off, nc_on = "---", "---"
        else:
            nc_off, nc_on = "---", "---"
        lines.append(f"{label} & {oo_off} & {oo_on} & {co_off} & {co_on} & {nc_off} & {nc_on} \\\\")

    # Narrow collision/success
    nc_off_col = fmt_val(narrow_results.get('OFF'), 'collision_rate', '.1f', True)
    nc_on_col = fmt_val(narrow_results.get('ON'), 'collision_rate', '.1f', True)
    lines.append(f"Collision Rate (\\%) & --- & --- & --- & --- & {nc_off_col} & {nc_on_col} \\\\")

    nc_off_suc = fmt_val(narrow_results.get('OFF'), 'success_rate', '.1f', True)
    nc_on_suc = fmt_val(narrow_results.get('ON'), 'success_rate', '.1f', True)
    lines.append(f"Success Rate (\\%) & --- & --- & --- & --- & {nc_off_suc} & {nc_on_suc} \\\\")

    lines.append(r"\hline")

    # Efficiency metrics
    lines.append(r"\multicolumn{7}{l}{\textbf{B. Efficiency Metrics}} \\")
    lines.append(r"\hline")

    eff_metrics = [
        ('fuel', 'Fuel Consumption', '.1f', False),
        ('episode_time', 'Episode Time (steps)', '.0f', False),
        ('smoothness', 'Smoothness ($\\sigma_{\\Delta hdg}$)', '.4f', False),
    ]

    for key, label, fmt, pct in eff_metrics:
        oo_off = fmt_val(all_results.get('Open Ocean_OFF'), key, fmt, pct)
        oo_on = fmt_val(all_results.get('Open Ocean_ON'), key, fmt, pct)
        co_off = fmt_val(all_results.get('Coastal_OFF'), key, fmt, pct)
        co_on = fmt_val(all_results.get('Coastal_ON'), key, fmt, pct)
        lines.append(f"{label} & {oo_off} & {oo_on} & {co_off} & {co_on} & --- & --- \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nLaTeX 테이블 저장: {filepath}")


# ============================================================================
# 그래프: 6-subplot 바 차트
# ============================================================================

def plot_bar_charts(all_results, narrow_results):
    """6개 서브플롯 바 차트 생성"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Efficiency & Safety Analysis: COMM OFF vs ON', fontsize=16, fontweight='bold')

    envs = ['Open Ocean', 'Coastal', 'Narrow']
    colors_off = '#E74C3C'   # 빨강
    colors_on = '#3498DB'    # 파랑
    x = np.arange(len(envs))
    width = 0.35

    def get_metric(env, comm, metric_key):
        """메트릭 값 추출 (mean, std)"""
        if env == 'Narrow':
            if metric_key in ['collision_rate', 'success_rate', 'collisions', 'successes']:
                return narrow_results[comm][metric_key]
            else:
                return (np.nan, np.nan)
        key = f"{env}_{comm}"
        if key in all_results and all_results[key] is not None:
            return all_results[key].get(metric_key, (np.nan, np.nan))
        return (np.nan, np.nan)

    # 1. COLREGs Compliance
    ax = axes[0, 0]
    off_vals = [get_metric(e, 'OFF', 'colregs_overall')[0] * 100 for e in envs]
    off_errs = [get_metric(e, 'OFF', 'colregs_overall')[1] * 100 for e in envs]
    on_vals = [get_metric(e, 'ON', 'colregs_overall')[0] * 100 for e in envs]
    on_errs = [get_metric(e, 'ON', 'colregs_overall')[1] * 100 for e in envs]
    # Narrow에 NaN이면 0으로 표시 안함
    bars1 = ax.bar(x - width/2, off_vals, width, yerr=off_errs, label='COMM OFF',
                   color=colors_off, capsize=4, alpha=0.85)
    bars2 = ax.bar(x + width/2, on_vals, width, yerr=on_errs, label='COMM ON',
                   color=colors_on, capsize=4, alpha=0.85)
    ax.set_ylabel('Compliance (%)')
    ax.set_title('COLREGs Compliance Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.legend()
    ax.set_ylim(0, 100)

    # 2. DCPA
    ax = axes[0, 1]
    off_vals = [get_metric(e, 'OFF', 'dcpa')[0] for e in envs]
    off_errs = [get_metric(e, 'OFF', 'dcpa')[1] for e in envs]
    on_vals = [get_metric(e, 'ON', 'dcpa')[0] for e in envs]
    on_errs = [get_metric(e, 'ON', 'dcpa')[1] for e in envs]
    ax.bar(x - width/2, off_vals, width, yerr=off_errs, label='COMM OFF',
           color=colors_off, capsize=4, alpha=0.85)
    ax.bar(x + width/2, on_vals, width, yerr=on_errs, label='COMM ON',
           color=colors_on, capsize=4, alpha=0.85)
    ax.set_ylabel('Distance (m)')
    ax.set_title('Avg. DCPA per Encounter')
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.legend()

    # 3. Avg Vessels in Radar Range
    ax = axes[0, 2]
    off_vals = [get_metric(e, 'OFF', 'radar_count')[0] for e in envs]
    off_errs = [get_metric(e, 'OFF', 'radar_count')[1] for e in envs]
    on_vals = [get_metric(e, 'ON', 'radar_count')[0] for e in envs]
    on_errs = [get_metric(e, 'ON', 'radar_count')[1] for e in envs]
    ax.bar(x - width/2, off_vals, width, yerr=off_errs, label='COMM OFF',
           color=colors_off, capsize=4, alpha=0.85)
    ax.bar(x + width/2, on_vals, width, yerr=on_errs, label='COMM ON',
           color=colors_on, capsize=4, alpha=0.85)
    ax.set_ylabel('Count')
    ax.set_title('Avg. Vessels in Radar Range')
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.legend()

    # 4. Fuel Consumption
    ax = axes[1, 0]
    off_vals = [get_metric(e, 'OFF', 'fuel')[0] for e in envs]
    off_errs = [get_metric(e, 'OFF', 'fuel')[1] for e in envs]
    on_vals = [get_metric(e, 'ON', 'fuel')[0] for e in envs]
    on_errs = [get_metric(e, 'ON', 'fuel')[1] for e in envs]
    ax.bar(x - width/2, off_vals, width, yerr=off_errs, label='COMM OFF',
           color=colors_off, capsize=4, alpha=0.85)
    ax.bar(x + width/2, on_vals, width, yerr=on_errs, label='COMM ON',
           color=colors_on, capsize=4, alpha=0.85)
    ax.set_ylabel('Fuel (proxy)')
    ax.set_title('Fuel Consumption per Episode')
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.legend()

    # 5. Episode Time
    ax = axes[1, 1]
    off_vals = [get_metric(e, 'OFF', 'episode_time')[0] for e in envs]
    off_errs = [get_metric(e, 'OFF', 'episode_time')[1] for e in envs]
    on_vals = [get_metric(e, 'ON', 'episode_time')[0] for e in envs]
    on_errs = [get_metric(e, 'ON', 'episode_time')[1] for e in envs]
    ax.bar(x - width/2, off_vals, width, yerr=off_errs, label='COMM OFF',
           color=colors_off, capsize=4, alpha=0.85)
    ax.bar(x + width/2, on_vals, width, yerr=on_errs, label='COMM ON',
           color=colors_on, capsize=4, alpha=0.85)
    ax.set_ylabel('Steps')
    ax.set_title('Episode Time (steps to success)')
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.legend()

    # 6. Smoothness
    ax = axes[1, 2]
    off_vals = [get_metric(e, 'OFF', 'smoothness')[0] for e in envs]
    off_errs = [get_metric(e, 'OFF', 'smoothness')[1] for e in envs]
    on_vals = [get_metric(e, 'ON', 'smoothness')[0] for e in envs]
    on_errs = [get_metric(e, 'ON', 'smoothness')[1] for e in envs]
    ax.bar(x - width/2, off_vals, width, yerr=off_errs, label='COMM OFF',
           color=colors_off, capsize=4, alpha=0.85)
    ax.bar(x + width/2, on_vals, width, yerr=on_errs, label='COMM ON',
           color=colors_on, capsize=4, alpha=0.85)
    ax.set_ylabel(r'$\sigma(\Delta heading)$')
    ax.set_title('Path Smoothness (lower = smoother)')
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # NaN 바 숨기기 (Narrow에 데이터 없는 메트릭)
    for ax_row in axes:
        for ax in ax_row:
            for bar in ax.patches:
                if np.isnan(bar.get_height()):
                    bar.set_height(0)
                    bar.set_visible(False)

    png_path = os.path.join(FIG_DIR, "efficiency_analysis.png")
    pdf_path = os.path.join(FIG_DIR, "efficiency_analysis.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\n바 차트 저장: {png_path}")
    print(f"바 차트 저장: {pdf_path}")


# ============================================================================
# 그래프: 궤적 비교
# ============================================================================

def plot_trajectory_comparison(all_dfs):
    """Open Ocean, Coastal 궤적 비교 (COMM OFF vs ON)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Trajectory Comparison: COMM OFF (red) vs COMM ON (blue)', fontsize=14, fontweight='bold')

    for idx, env_name in enumerate(['Open Ocean', 'Coastal']):
        ax = axes[idx]
        ax.set_title(env_name, fontsize=13)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        for comm_label, color, alpha in [('OFF', '#E74C3C', 0.3), ('ON', '#3498DB', 0.3)]:
            key = f"{env_name}_{comm_label}"
            if key not in all_dfs or all_dfs[key] is None:
                continue
            df = all_dfs[key]
            # 첫 번째 런만 사용 (가독성)
            runs = split_into_runs(df, 1)
            if not runs:
                continue
            run_df = runs[0]

            for aid in run_df['agent_id'].unique():
                agent_data = run_df[run_df['agent_id'] == aid].sort_values('step')
                ax.plot(agent_data['x'].values, agent_data['z'].values,
                        color=color, alpha=alpha, linewidth=0.8)
                # 시작점
                ax.scatter(agent_data['x'].iloc[0], agent_data['z'].iloc[0],
                           color=color, s=20, zorder=5, marker='o')
                # 끝점
                ax.scatter(agent_data['x'].iloc[-1], agent_data['z'].iloc[-1],
                           color=color, s=20, zorder=5, marker='x')

        # 범례
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#E74C3C', lw=2, label='COMM OFF'),
            Line2D([0], [0], color='#3498DB', lw=2, label='COMM ON'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = os.path.join(FIG_DIR, "trajectory_comparison.png")
    pdf_path = os.path.join(FIG_DIR, "trajectory_comparison.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"궤적 비교 저장: {png_path}")
    print(f"궤적 비교 저장: {pdf_path}")


# ============================================================================
# COLREGs 상황별 상세 바 차트
# ============================================================================

def plot_colregs_detail(all_results):
    """COLREGs 상황별 준수율 상세 바 차트"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('COLREGs Compliance by Situation Type', fontsize=14, fontweight='bold')

    situations = ['HeadOn', 'StandOn', 'GiveWay', 'Overtaking']
    metric_keys = ['colregs_headon', 'colregs_standon', 'colregs_giveway', 'colregs_overtaking']
    colors_off = '#E74C3C'
    colors_on = '#3498DB'
    x = np.arange(len(situations))
    width = 0.35

    for idx, env_name in enumerate(['Open Ocean', 'Coastal']):
        ax = axes[idx]
        ax.set_title(env_name, fontsize=13)

        off_vals, off_errs = [], []
        on_vals, on_errs = [], []
        for mk in metric_keys:
            key_off = f"{env_name}_OFF"
            key_on = f"{env_name}_ON"
            if key_off in all_results and all_results[key_off] is not None:
                m, s = all_results[key_off][mk]
                off_vals.append(m * 100 if not np.isnan(m) else 0)
                off_errs.append(s * 100 if not np.isnan(s) else 0)
            else:
                off_vals.append(0)
                off_errs.append(0)
            if key_on in all_results and all_results[key_on] is not None:
                m, s = all_results[key_on][mk]
                on_vals.append(m * 100 if not np.isnan(m) else 0)
                on_errs.append(s * 100 if not np.isnan(s) else 0)
            else:
                on_vals.append(0)
                on_errs.append(0)

        ax.bar(x - width/2, off_vals, width, yerr=off_errs, label='COMM OFF',
               color=colors_off, capsize=4, alpha=0.85)
        ax.bar(x + width/2, on_vals, width, yerr=on_errs, label='COMM ON',
               color=colors_on, capsize=4, alpha=0.85)
        ax.set_ylabel('Compliance (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(situations)
        ax.legend()
        ax.set_ylim(0, 100)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = os.path.join(FIG_DIR, "colregs_detail.png")
    pdf_path = os.path.join(FIG_DIR, "colregs_detail.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"COLREGs 상세 저장: {png_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Vessel Navigation Efficiency & Safety Analysis")
    print("=" * 60)

    all_results = {}
    all_dfs = {}

    # CSV 기반 환경 분석
    for env_name, files in CSV_FILES.items():
        print(f"\n--- {env_name} ---")
        for comm_label, filepath in files.items():
            key = f"{env_name}_{comm_label}"
            result = analyze_environment(env_name, comm_label, filepath)
            all_results[key] = result

            # 궤적 비교용 DataFrame 보관
            df = load_csv(filepath)
            all_dfs[key] = df

    # Narrow Channel 분석
    print("\n--- Narrow Channel ---")
    narrow_results = analyze_narrow_channel()
    print(f"  [Narrow COMM OFF] Collisions={np.mean(NARROW_RAW['OFF']['collisions']):.1f}, "
          f"Successes={np.mean(NARROW_RAW['OFF']['successes']):.1f}")
    print(f"  [Narrow COMM ON]  Collisions={np.mean(NARROW_RAW['ON']['collisions']):.1f}, "
          f"Successes={np.mean(NARROW_RAW['ON']['successes']):.1f}")

    # 결과 출력
    print_results_table(all_results, narrow_results)

    # LaTeX 테이블 저장
    save_latex_table(all_results, narrow_results)

    # 그래프 생성
    print("\n--- 그래프 생성 ---")
    plot_bar_charts(all_results, narrow_results)
    plot_trajectory_comparison(all_dfs)
    plot_colregs_detail(all_results)

    print("\n분석 완료!")


if __name__ == '__main__':
    main()
