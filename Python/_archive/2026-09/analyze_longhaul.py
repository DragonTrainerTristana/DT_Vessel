"""
Long-haul (e.g., Taiwan↔Busan) Comm ON/OFF 비교 분석 및 플롯

입력:
- {tag}_commOFF_{n}runs_*_summary.csv
- {tag}_commON_{n}runs_*_summary.csv
  컬럼: run_id, agent_id, arrival_step, fuel_consumption, comm_mode

측정 지표:
- episode_time: arrival_step (첫 도달까지 걸린 step 수, max_steps인 경우 censored)
- fuel_consumption: Σ speed² over unarrived steps (정규화 speed 사용)

주작 방지:
- 실제 Unity 시뮬 결과 CSV만 사용
- 통계 요약 (mean, std, median, IQR)
- censored (도달 못 한 경우) 별도 표기
- Welch t-test로 유의성 검증

사용법:
    python analyze_longhaul.py <summary_commOFF.csv> <summary_commON.csv>
또는 기본값 자동 탐색:
    python analyze_longhaul.py
    → PROJECT_ROOT/trajectory_data/ 에서 가장 최근 longhaul_commOFF/ON summary 찾기
"""
import os
import sys
import glob
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from config import PROJECT_ROOT


def find_latest_summary(tag="longhaul", comm_mode="OFF"):
    pattern = os.path.join(PROJECT_ROOT, "trajectory_data",
                           f"{tag}_comm{comm_mode}_*runs_*_summary.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[-1]


def load_summaries(off_path=None, on_path=None, tag="longhaul"):
    off_path = off_path or find_latest_summary(tag, "OFF")
    on_path = on_path or find_latest_summary(tag, "ON")
    if off_path is None or on_path is None:
        raise FileNotFoundError(
            f"Summary CSV not found. Expected pattern: "
            f"{tag}_commOFF/ON_*runs_*_summary.csv in trajectory_data/"
        )
    df_off = pd.read_csv(off_path)
    df_on = pd.read_csv(on_path)
    print(f"[LOAD] OFF: {off_path} ({len(df_off)} rows)")
    print(f"[LOAD] ON:  {on_path} ({len(df_on)} rows)")
    return df_off, df_on, off_path, on_path


def summarize(df, label, max_steps=None):
    """per-mode 요약 통계.
    censored (arrival_step == max_steps) 배는 별도 집계.
    """
    if max_steps is None:
        max_steps = df['arrival_step'].max()
    arrived = df[df['arrival_step'] < max_steps]
    censored = df[df['arrival_step'] >= max_steps]

    n_total = len(df)
    n_arrived = len(arrived)
    n_censored = len(censored)

    t = arrived['arrival_step'].values
    f = arrived['fuel_consumption'].values

    s = {
        'label': label,
        'n_total': n_total,
        'n_arrived': n_arrived,
        'n_censored': n_censored,
        'arrival_rate': n_arrived / n_total if n_total > 0 else 0,
        'time_mean': np.mean(t) if len(t) else float('nan'),
        'time_std': np.std(t, ddof=1) if len(t) > 1 else float('nan'),
        'time_median': np.median(t) if len(t) else float('nan'),
        'fuel_mean': np.mean(f) if len(f) else float('nan'),
        'fuel_std': np.std(f, ddof=1) if len(f) > 1 else float('nan'),
        'fuel_median': np.median(f) if len(f) else float('nan'),
        '_arrived_t': t,
        '_arrived_f': f,
    }
    return s


def print_summary_table(s_off, s_on):
    print("\n" + "=" * 80)
    print(f"{'Metric':<25} {'OFF':>20} {'ON':>20} {'Δ (ON-OFF)':>12}")
    print("-" * 80)
    def row(name, vo, vn, fmt="{:>20.2f}"):
        delta = vn - vo
        print(f"{name:<25} {fmt.format(vo):>20} {fmt.format(vn):>20} {delta:+.2f}")

    print(f"{'Total agents':<25} {s_off['n_total']:>20} {s_on['n_total']:>20}")
    print(f"{'Arrived':<25} {s_off['n_arrived']:>20} {s_on['n_arrived']:>20}")
    print(f"{'Censored':<25} {s_off['n_censored']:>20} {s_on['n_censored']:>20}")
    print(f"{'Arrival rate':<25} {s_off['arrival_rate']:>20.2%} {s_on['arrival_rate']:>20.2%}")
    print("-" * 80)
    row("Episode time mean", s_off['time_mean'], s_on['time_mean'])
    row("Episode time std", s_off['time_std'], s_on['time_std'])
    row("Episode time median", s_off['time_median'], s_on['time_median'])
    row("Fuel mean (Σ v²)", s_off['fuel_mean'], s_on['fuel_mean'], "{:>20.3f}")
    row("Fuel std", s_off['fuel_std'], s_on['fuel_std'], "{:>20.3f}")
    row("Fuel median", s_off['fuel_median'], s_on['fuel_median'], "{:>20.3f}")
    print("=" * 80)

    if HAS_SCIPY and len(s_off['_arrived_t']) > 1 and len(s_on['_arrived_t']) > 1:
        tt = sp_stats.ttest_ind(s_off['_arrived_t'], s_on['_arrived_t'], equal_var=False)
        tf = sp_stats.ttest_ind(s_off['_arrived_f'], s_on['_arrived_f'], equal_var=False)
        print(f"\n[Welch t-test]")
        print(f"  Episode time:      t={tt.statistic:+.3f}, p={tt.pvalue:.4f}"
              f"  {'(significant)' if tt.pvalue < 0.05 else '(not significant)'}")
        print(f"  Fuel consumption:  t={tf.statistic:+.3f}, p={tf.pvalue:.4f}"
              f"  {'(significant)' if tf.pvalue < 0.05 else '(not significant)'}")


def plot_comparison(s_off, s_on, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Figure 1: bar + error (episode time, fuel)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    modes = ['OFF', 'ON']
    time_means = [s_off['time_mean'], s_on['time_mean']]
    time_stds = [s_off['time_std'], s_on['time_std']]
    fuel_means = [s_off['fuel_mean'], s_on['fuel_mean']]
    fuel_stds = [s_off['fuel_std'], s_on['fuel_std']]

    bar_colors = ['#D87070', '#70A7D8']

    axes[0].bar(modes, time_means, yerr=time_stds, color=bar_colors, capsize=8,
                edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Episode time (steps)')
    axes[0].set_title('Episode time: OFF vs ON')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(modes, fuel_means, yerr=fuel_stds, color=bar_colors, capsize=8,
                edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Fuel consumption (Σ v²)')
    axes[1].set_title('Fuel consumption: OFF vs ON')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(out_dir, f"longhaul_bar_{ts}.png")
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[FIG] {p1}")

    # Figure 2: box plot (arrival time and fuel distribution)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    t_data = [s_off['_arrived_t'], s_on['_arrived_t']]
    f_data = [s_off['_arrived_f'], s_on['_arrived_f']]

    bp1 = axes[0].boxplot(t_data, labels=modes, patch_artist=True, widths=0.5)
    for patch, c in zip(bp1['boxes'], bar_colors):
        patch.set_facecolor(c)
    axes[0].set_ylabel('Episode time (steps)')
    axes[0].set_title('Episode time distribution')
    axes[0].grid(axis='y', alpha=0.3)

    bp2 = axes[1].boxplot(f_data, labels=modes, patch_artist=True, widths=0.5)
    for patch, c in zip(bp2['boxes'], bar_colors):
        patch.set_facecolor(c)
    axes[1].set_ylabel('Fuel consumption (Σ v²)')
    axes[1].set_title('Fuel consumption distribution')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p2 = os.path.join(out_dir, f"longhaul_box_{ts}.png")
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[FIG] {p2}")

    # Figure 3: scatter (time vs fuel per episode)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(s_off['_arrived_t'], s_off['_arrived_f'], c=bar_colors[0], s=40,
               alpha=0.7, edgecolor='black', linewidth=0.5, label='OFF')
    ax.scatter(s_on['_arrived_t'], s_on['_arrived_f'], c=bar_colors[1], s=40,
               alpha=0.7, edgecolor='black', linewidth=0.5, label='ON')
    ax.set_xlabel('Episode time (steps)')
    ax.set_ylabel('Fuel consumption (Σ v²)')
    ax.set_title('Per-episode: time vs fuel')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p3 = os.path.join(out_dir, f"longhaul_scatter_{ts}.png")
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[FIG] {p3}")

    return [p1, p2, p3]


def main():
    args = sys.argv[1:]
    off_path = args[0] if len(args) > 0 else None
    on_path = args[1] if len(args) > 1 else None

    df_off, df_on, _, _ = load_summaries(off_path, on_path)

    # max_steps 추정 (censored 경계)
    max_step = max(df_off['arrival_step'].max(), df_on['arrival_step'].max())

    s_off = summarize(df_off, 'OFF', max_steps=max_step)
    s_on = summarize(df_on, 'ON', max_steps=max_step)

    print_summary_table(s_off, s_on)

    out_dir = os.path.join(PROJECT_ROOT, "figures")
    plot_comparison(s_off, s_on, out_dir)


if __name__ == "__main__":
    main()
