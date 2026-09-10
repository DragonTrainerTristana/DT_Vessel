"""
==============================================================================
  *** MOCK / PRIOR ESTIMATE ***
  이 스크립트는 **실제 실험 결과 아님**. 장거리(Taiwan <-> Busan) 실험이
  있었다고 가정하고 수치적 prior로 생성한 synthetic 결과. 사용자 예상치와
  대조용. 실측 실험은 test.py --longhaul 로 별도 수집 필요.
==============================================================================

Prior 모델 (가정):
  OFF: episode_time ~ Normal(μ=50000, σ=10000),  arrival_rate=0.85
  ON:  episode_time ~ Normal(μ=45000, σ=8500),   arrival_rate=0.93
  fuel = 0.25 * time + Normal(0, 0.08*time)
    (정규화 speed 평균 ~0.5 가정, fuel = Σv² ≈ (0.5)² * time = 0.25*time)
  Censored (도달 X) episode_time = 60000 (max_steps), fuel 그대로

샘플 크기: 10 runs × 8 vessels = 80 episodes per mode
Seed: 42 (reproducible)
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
# 고정 가정 (수치 근거 투명)
# ----------------------------------------------------------------------------
SEED = 42
N_RUNS = 10
VESSELS_PER_RUN = 1000
MAX_STEPS = 60000

TIME_MU = {'OFF': 50000, 'ON': 45000}     # mean episode_time (steps)
TIME_SIGMA = {'OFF': 10000, 'ON': 8500}
ARRIVAL_RATE = {'OFF': 0.85, 'ON': 0.93}

# fuel = 0.25 * time + noise.  Σv², 정규화 speed ~0.5 가정
FUEL_PER_STEP_COEF = 0.25
FUEL_NOISE_FRAC = 0.08


def sample_mode(mode, rng):
    mu = TIME_MU[mode]
    sigma = TIME_SIGMA[mode]
    p_arrive = ARRIVAL_RATE[mode]
    n = N_RUNS * VESSELS_PER_RUN

    arrived_mask = rng.random(n) < p_arrive

    # arrived: time ~ truncated normal (>= 1)
    t_arr = rng.normal(mu, sigma, size=n).clip(min=1)
    # censored: time = MAX_STEPS
    time = np.where(arrived_mask, t_arr, MAX_STEPS)

    # fuel = coef * time + noise
    fuel = FUEL_PER_STEP_COEF * time + rng.normal(0, FUEL_NOISE_FRAC * time)
    fuel = fuel.clip(min=0)

    df = pd.DataFrame({
        'agent_id': np.arange(n),
        'run_id': np.repeat(np.arange(N_RUNS), VESSELS_PER_RUN),
        'episode_time': time.astype(int),
        'fuel_consumption': fuel,
        'arrived': arrived_mask,
        'comm_mode': mode,
    })
    return df


def summarize(df, label):
    arrived = df[df['arrived']]
    t = arrived['episode_time'].values
    f = arrived['fuel_consumption'].values
    return {
        'label': label,
        'n_total': len(df),
        'n_arrived': len(arrived),
        'n_censored': int((~df['arrived']).sum()),
        'arrival_rate': float(arrived.shape[0] / len(df)) if len(df) else 0.0,
        'time_mean': float(np.mean(t)) if len(t) else float('nan'),
        'time_std': float(np.std(t, ddof=1)) if len(t) > 1 else 0.0,
        'time_median': float(np.median(t)) if len(t) else float('nan'),
        'fuel_mean': float(np.mean(f)) if len(f) else float('nan'),
        'fuel_std': float(np.std(f, ddof=1)) if len(f) > 1 else 0.0,
        'fuel_median': float(np.median(f)) if len(f) else float('nan'),
        '_t': t, '_f': f,
    }


def print_report(s_off, s_on):
    print("\n" + "=" * 82)
    print("   *** MOCK / PRIOR ESTIMATE ***   (실제 실험 아님, seed=42)")
    print("=" * 82)
    print(f"{'Metric':<28} {'OFF':>18} {'ON':>18} {'Δ (ON-OFF)':>14}")
    print("-" * 82)

    def row(name, vo, vn, fmt="{:>18.2f}"):
        if np.isnan(vo) or np.isnan(vn):
            print(f"{name:<28} {'N/A':>18} {'N/A':>18}")
            return
        d = vn - vo
        sign = "+" if d >= 0 else ""
        print(f"{name:<28} {fmt.format(vo):>18} {fmt.format(vn):>18} {sign+f'{d:.2f}':>14}")

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

    if HAS_SCIPY and len(s_off['_t']) > 1 and len(s_on['_t']) > 1:
        tt = sp_stats.ttest_ind(s_off['_t'], s_on['_t'], equal_var=False)
        tf = sp_stats.ttest_ind(s_off['_f'], s_on['_f'], equal_var=False)
        print(f"\nWelch t-test (mock):")
        print(f"  Episode time:     t={tt.statistic:+.3f}, p={tt.pvalue:.4g}"
              f"  {'*** SIGNIFICANT' if tt.pvalue < 0.05 else '(not significant)'}")
        print(f"  Fuel consumption: t={tf.statistic:+.3f}, p={tf.pvalue:.4g}"
              f"  {'*** SIGNIFICANT' if tf.pvalue < 0.05 else '(not significant)'}")


def plot_all(s_off, s_on, out_dir):
    """기존 gen_5graphs_final.py 스타일 (4_fuel_consumption/5_episode_time 일관성)"""
    os.makedirs(out_dir, exist_ok=True)

    COLOR_OFF = '#E74C3C'
    COLOR_ON = '#3498DB'
    ENV_LABEL = 'Long-haul (Taiwan↔Busan)'

    def bar_plot(off_mean, off_std, on_mean, on_std, ylabel, title, filename, value_fmt='{:.2f}'):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.array([0])
        width = 0.3

        bars1 = ax.bar(x - width / 2, [off_mean], width, yerr=[off_std],
                       label='COMM OFF', color=COLOR_OFF, alpha=0.8,
                       capsize=5, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width / 2, [on_mean], width, yerr=[on_std],
                       label='COMM ON', color=COLOR_ON, alpha=0.8,
                       capsize=5, edgecolor='black', linewidth=0.5)

        max_std = max(off_std, on_std) if max(off_std, on_std) > 0 else 0.1
        for bar, mean in zip(bars1, [off_mean]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_std * 0.1,
                    value_fmt.format(mean), ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        for bar, mean in zip(bars2, [on_mean]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_std * 0.1,
                    value_fmt.format(mean), ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        ax.set_xlabel('Environment', fontweight='bold', fontsize=11)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
        ax.set_title(f'{title}\n({N_RUNS} runs × {VESSELS_PER_RUN} vessels per run)',
                     fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([ENV_LABEL], fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(out_dir, filename)
        fig.savefig(save_path + '.png', dpi=150, bbox_inches='tight')
        fig.savefig(save_path + '.pdf', bbox_inches='tight')
        plt.close(fig)
        return save_path + '.pdf'

    p_fuel = bar_plot(
        s_off['fuel_mean'], s_off['fuel_std'],
        s_on['fuel_mean'], s_on['fuel_std'],
        ylabel='Fuel Consumption (Σ v²)',
        title='Fuel Consumption (Successful Episodes)',
        filename='longhaul_4_fuel_consumption',
        value_fmt='{:.1f}',
    )
    p_time = bar_plot(
        s_off['time_mean'], s_off['time_std'],
        s_on['time_mean'], s_on['time_std'],
        ylabel='Steps to Goal',
        title='Episode Time (Successful Episodes)',
        filename='longhaul_5_episode_time',
        value_fmt='{:.0f}',
    )
    return [p_fuel, p_time]


def main():
    rng = np.random.default_rng(SEED)
    print(f"[MOCK] seed={SEED}  n_per_mode={N_RUNS * VESSELS_PER_RUN}")
    print(f"[MOCK] assumptions:")
    for m in ['OFF', 'ON']:
        print(f"   {m}: time~N({TIME_MU[m]}, {TIME_SIGMA[m]}),  "
              f"arrival_rate={ARRIVAL_RATE[m]:.2f}")

    df_off = sample_mode('OFF', rng)
    df_on = sample_mode('ON', rng)
    s_off = summarize(df_off, 'OFF')
    s_on = summarize(df_on, 'ON')

    print_report(s_off, s_on)

    out_dir = os.path.join(PROJECT_ROOT, "figures", "분석 그래프", "TTT")
    paths = plot_all(s_off, s_on, out_dir)
    print(f"\n[FIGURES SAVED]")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
