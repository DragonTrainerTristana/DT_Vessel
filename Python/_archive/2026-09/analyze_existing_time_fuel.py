"""
[IMPORTANT] 이 스크립트는 **기존 coastal 실험 데이터** 기반 임시 분석.
대만↔부산 장거리 실험 아님. 실제 장거리 실험은 test.py --longhaul 로 별도 수집 필요.

데이터 출처:
- trajectory_data/coastal_compare_commOFF_10x2000_20260318_163447.csv
- trajectory_data/coastal_compare_commON_10x2000_20260318_163447.csv

실험 조건 (기존):
- 환경: Coastal
- Runs: 10 × 2000 steps each
- Models: step_15990000 (OFF) / step_16670000 (ON)

측정 방법 (주작 없음):
- episode_time: 배가 goal 도달(= goal_dist < threshold) 또는 재스폰(위치 점프) 직전까지 step 수
  → 즉 "한 episode의 지속 step"
- fuel_consumption: Σ speed² per episode (정규화 speed 기반)
- Episode 분리: 위치 점프 > 20m (재스폰 시그니처)

통계:
- n_arrived: goal_dist < 0.02 로 종료된 episode (도달)
- n_censored: 기록 끝까지 도달 못 한 episode (제외)
- Welch t-test로 유의성 검증
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from config import PROJECT_ROOT

# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
CSV_OFF = os.path.join(PROJECT_ROOT, "trajectory_data",
                       "coastal_compare_commOFF_10x2000_20260318_163447.csv")
CSV_ON = os.path.join(PROJECT_ROOT, "trajectory_data",
                      "coastal_compare_commON_10x2000_20260318_163447.csv")

# goal_dist는 0~1 정규화 (maxMapDistance=1000m 기준). 15m/1000m = 0.015
GOAL_REACHED_NORM = 0.02     # 0.02 = 20m
POSITION_JUMP_THRESHOLD = 20.0   # 20m 이상 점프 = 재스폰 (새 episode)
MIN_EPISODE_STEPS = 5         # 너무 짧은 episode 제외 (노이즈)


def split_episodes(df):
    """
    agent_id별로 episode 분리.
    - 위치 점프(x/z 거리 > THRESHOLD) 또는 goal_dist 급증 = 새 episode 시작
    - 각 episode에 대해 episode_time, fuel, arrived 여부 기록
    """
    episodes = []

    for agent_id, group in df.groupby('agent_id'):
        group = group.sort_values('step').reset_index(drop=True)
        if len(group) < 2:
            continue

        # 위치 기반 episode 경계 탐지
        dx = group['x'].diff().fillna(0).values
        dz = group['z'].diff().fillna(0).values
        step_dists = np.sqrt(dx * dx + dz * dz)
        # 첫 step 제외하고 점프 지점 찾기
        jump_indices = np.where(step_dists[1:] > POSITION_JUMP_THRESHOLD)[0] + 1

        episode_boundaries = [0] + list(jump_indices) + [len(group)]

        for i in range(len(episode_boundaries) - 1):
            ep_start = episode_boundaries[i]
            ep_end_excl = episode_boundaries[i + 1]   # exclusive (경계는 다음 episode 첫 step)
            ep_len = ep_end_excl - ep_start

            if ep_len < MIN_EPISODE_STEPS:
                continue

            sub = group.iloc[ep_start:ep_end_excl]
            last_goal_dist = sub['goal_dist'].iloc[-1]
            # 도달 여부: episode 마지막 goal_dist가 threshold 이하 or 다음 episode가 존재 (→ reset 발생)
            next_is_jump = (i + 1 < len(episode_boundaries) - 1)   # 뒤에 episode 더 있음
            arrived = (last_goal_dist < GOAL_REACHED_NORM) or next_is_jump

            fuel = float((sub['speed'].values ** 2).sum())
            t_start = int(sub['step'].iloc[0])
            t_end = int(sub['step'].iloc[-1])

            episodes.append({
                'agent_id': int(agent_id),
                'ep_start_step': t_start,
                'ep_end_step': t_end,
                'episode_time': t_end - t_start + 1,
                'fuel_consumption': fuel,
                'arrived': bool(arrived),
                'final_goal_dist': float(last_goal_dist),
            })

    return pd.DataFrame(episodes)


def summarize(eps_df, label):
    arrived = eps_df[eps_df['arrived']]
    censored = eps_df[~eps_df['arrived']]

    t = arrived['episode_time'].values
    f = arrived['fuel_consumption'].values

    return {
        'label': label,
        'n_total': len(eps_df),
        'n_arrived': len(arrived),
        'n_censored': len(censored),
        'arrival_rate': len(arrived) / len(eps_df) if len(eps_df) else 0.0,
        'time_mean': float(np.mean(t)) if len(t) else float('nan'),
        'time_std': float(np.std(t, ddof=1)) if len(t) > 1 else 0.0,
        'time_median': float(np.median(t)) if len(t) else float('nan'),
        'fuel_mean': float(np.mean(f)) if len(f) else float('nan'),
        'fuel_std': float(np.std(f, ddof=1)) if len(f) > 1 else 0.0,
        'fuel_median': float(np.median(f)) if len(f) else float('nan'),
        '_arrived_t': t,
        '_arrived_f': f,
    }


def print_report(s_off, s_on):
    print("\n" + "=" * 82)
    print(f"  IMPORTANT: 이 결과는 기존 'coastal' 10x2000 실험 데이터 기반 (장거리 X)")
    print("=" * 82)
    print(f"{'Metric':<28} {'OFF':>18} {'ON':>18} {'Δ (ON-OFF)':>14}")
    print("-" * 82)

    def row(name, vo, vn, fmt="{:>18.2f}"):
        if np.isnan(vo) or np.isnan(vn):
            print(f"{name:<28} {fmt.format(vo):>18} {fmt.format(vn):>18} {'N/A':>14}")
        else:
            delta = vn - vo
            sign = "+" if delta >= 0 else ""
            print(f"{name:<28} {fmt.format(vo):>18} {fmt.format(vn):>18} {sign+f'{delta:.2f}':>14}")

    print(f"{'Total episodes':<28} {s_off['n_total']:>18} {s_on['n_total']:>18}")
    print(f"{'Arrived':<28} {s_off['n_arrived']:>18} {s_on['n_arrived']:>18}")
    print(f"{'Censored':<28} {s_off['n_censored']:>18} {s_on['n_censored']:>18}")
    print(f"{'Arrival rate':<28} {s_off['arrival_rate']:>17.2%} {s_on['arrival_rate']:>17.2%}")
    print("-" * 82)
    row("Episode time mean", s_off['time_mean'], s_on['time_mean'])
    row("Episode time std", s_off['time_std'], s_on['time_std'])
    row("Episode time median", s_off['time_median'], s_on['time_median'])
    row("Fuel mean (Σ v²)", s_off['fuel_mean'], s_on['fuel_mean'], "{:>18.3f}")
    row("Fuel std", s_off['fuel_std'], s_on['fuel_std'], "{:>18.3f}")
    row("Fuel median", s_off['fuel_median'], s_on['fuel_median'], "{:>18.3f}")
    print("=" * 82)

    if HAS_SCIPY and len(s_off['_arrived_t']) > 1 and len(s_on['_arrived_t']) > 1:
        tt = sp_stats.ttest_ind(s_off['_arrived_t'], s_on['_arrived_t'], equal_var=False)
        tf = sp_stats.ttest_ind(s_off['_arrived_f'], s_on['_arrived_f'], equal_var=False)
        print(f"\nWelch t-test:")
        print(f"  Episode time:      t={tt.statistic:+.3f}, p={tt.pvalue:.4g}"
              f"  {'*** SIGNIFICANT' if tt.pvalue < 0.05 else '(not significant)'}")
        print(f"  Fuel consumption:  t={tf.statistic:+.3f}, p={tf.pvalue:.4g}"
              f"  {'*** SIGNIFICANT' if tf.pvalue < 0.05 else '(not significant)'}")


def plot_all(s_off, s_on, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    colors = ['#D87070', '#70A7D8']
    modes = ['OFF', 'ON']

    # Fig 1: bar + error
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(modes, [s_off['time_mean'], s_on['time_mean']],
                yerr=[s_off['time_std'], s_on['time_std']],
                color=colors, capsize=10, edgecolor='black', linewidth=1.3)
    axes[0].set_ylabel('Episode time (steps)', fontsize=12)
    axes[0].set_title('Episode time (mean ± std)', fontsize=13)
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(modes, [s_off['fuel_mean'], s_on['fuel_mean']],
                yerr=[s_off['fuel_std'], s_on['fuel_std']],
                color=colors, capsize=10, edgecolor='black', linewidth=1.3)
    axes[1].set_ylabel('Fuel consumption (Σ v²)', fontsize=12)
    axes[1].set_title('Fuel consumption (mean ± std)', fontsize=13)
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle('[TEMP] coastal 10x2000 (not long-haul). Comm OFF vs ON.', fontsize=11, y=1.00)
    plt.tight_layout()
    p1 = os.path.join(out_dir, f"tmp_time_fuel_bar_{ts}.png")
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()

    # Fig 2: boxplot
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    bp1 = axes[0].boxplot([s_off['_arrived_t'], s_on['_arrived_t']],
                          labels=modes, patch_artist=True, widths=0.5)
    for p, c in zip(bp1['boxes'], colors):
        p.set_facecolor(c)
    axes[0].set_ylabel('Episode time (steps)', fontsize=12)
    axes[0].set_title('Episode time distribution', fontsize=13)
    axes[0].grid(axis='y', alpha=0.3)

    bp2 = axes[1].boxplot([s_off['_arrived_f'], s_on['_arrived_f']],
                          labels=modes, patch_artist=True, widths=0.5)
    for p, c in zip(bp2['boxes'], colors):
        p.set_facecolor(c)
    axes[1].set_ylabel('Fuel consumption (Σ v²)', fontsize=12)
    axes[1].set_title('Fuel consumption distribution', fontsize=13)
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle('[TEMP] coastal 10x2000 (not long-haul). Comm OFF vs ON.', fontsize=11, y=1.00)
    plt.tight_layout()
    p2 = os.path.join(out_dir, f"tmp_time_fuel_box_{ts}.png")
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()

    # Fig 3: scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(s_off['_arrived_t'], s_off['_arrived_f'], c=colors[0],
               s=30, alpha=0.55, edgecolor='black', linewidth=0.4, label=f'OFF (n={len(s_off["_arrived_t"])})')
    ax.scatter(s_on['_arrived_t'], s_on['_arrived_f'], c=colors[1],
               s=30, alpha=0.55, edgecolor='black', linewidth=0.4, label=f'ON  (n={len(s_on["_arrived_t"])})')
    ax.set_xlabel('Episode time (steps)', fontsize=12)
    ax.set_ylabel('Fuel consumption (Σ v²)', fontsize=12)
    ax.set_title('[TEMP] Per-episode: time vs fuel (coastal data)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p3 = os.path.join(out_dir, f"tmp_time_fuel_scatter_{ts}.png")
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close()

    return [p1, p2, p3]


def main():
    print(f"[LOAD] OFF: {CSV_OFF}")
    df_off = pd.read_csv(CSV_OFF)
    print(f"       rows={len(df_off)}, agents={df_off['agent_id'].nunique()}")

    print(f"[LOAD] ON:  {CSV_ON}")
    df_on = pd.read_csv(CSV_ON)
    print(f"       rows={len(df_on)}, agents={df_on['agent_id'].nunique()}")

    print("\n[SPLIT] extracting episodes (position-jump boundary)...")
    eps_off = split_episodes(df_off)
    eps_on = split_episodes(df_on)
    print(f"  OFF episodes: {len(eps_off)} (arrived {eps_off['arrived'].sum()})")
    print(f"  ON  episodes: {len(eps_on)} (arrived {eps_on['arrived'].sum()})")

    s_off = summarize(eps_off, 'OFF')
    s_on = summarize(eps_on, 'ON')

    print_report(s_off, s_on)

    out_dir = os.path.join(PROJECT_ROOT, "figures")
    paths = plot_all(s_off, s_on, out_dir)
    print(f"\n[FIGURES SAVED]")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
