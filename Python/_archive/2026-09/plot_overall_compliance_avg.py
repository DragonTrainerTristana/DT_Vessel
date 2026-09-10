"""
16척 전체 평균 COLREGs Compliance 계산 + 막대그래프.
- analyze_mixed.py의 agent_type별 분리(R/C) 대신, 16척 통합 평균
- ensure_comm_better 같은 인위적 보정 없이 raw 값
- 10 runs × 2000 steps, 4 configs (R2_C14, R4_C12, R6_C10, R8_C8)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
TRAJ_DIR = os.path.join(PROJECT_ROOT, "trajectory_data")
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Build", "MOE", "Various Agent"))

CONFIGS = ['R2_C14', 'R4_C12', 'R6_C10', 'R8_C8']
CONFIG_LABELS = ['2O/14C', '4O/12C', '6O/10C', '8O/8C']
NUM_RUNS = 10

# analyze_mixed.py와 동일 기준
SITUATIONS = {
    'HeadOn':       lambda r: r > 0,
    'CrossGiveWay': lambda r: r > 0,
    'CrossStandOn': lambda r: abs(r) < 0.2,
    'Overtaking':   lambda r: abs(r) > 0.01,
}


def split_into_runs(df, num_runs=NUM_RUNS):
    step_diffs = df['step'].diff()
    run_starts = df.index[step_diffs < 0].tolist()
    run_starts = [0] + run_starts
    runs = []
    for i in range(len(run_starts)):
        s = run_starts[i]
        e = run_starts[i + 1] if i + 1 < len(run_starts) else len(df)
        runs.append(df.iloc[s:e].copy().reset_index(drop=True))
    if len(runs) > num_runs:
        runs = runs[:num_runs]
    return runs


def compute_overall_compliance(run_df):
    """16척 전체 통합 compliance (R/C 구분 없음)"""
    total_compliant, total_count = 0, 0
    per_situation = {}
    for sit_name, fn in SITUATIONS.items():
        mask = run_df['colregs_name'] == sit_name
        sit = run_df.loc[mask]
        if len(sit) == 0:
            per_situation[sit_name] = (0, 0)
            continue
        compliant = int(sit['rudder'].apply(fn).sum())
        per_situation[sit_name] = (compliant, len(sit))
        total_compliant += compliant
        total_count += len(sit)
    overall = (total_compliant / total_count) if total_count > 0 else np.nan
    return overall, per_situation


def main():
    print("=" * 70)
    print("16-vessel Overall COLREGs Compliance")
    print("=" * 70)

    results = {}
    per_sit_aggr = {}
    for cfg in CONFIGS:
        path = os.path.join(TRAJ_DIR, f'open_ocean_mixed_{cfg}_10x2000_20260401_181757.csv')
        if not os.path.exists(path):
            print(f"[ERROR] not found: {path}")
            continue
        print(f"\n[{cfg}] loading {os.path.basename(path)} ...")
        df = pd.read_csv(path)
        runs = split_into_runs(df)
        per_run = []
        sit_totals = {k: [0, 0] for k in SITUATIONS}
        for i, run in enumerate(runs):
            overall, per_sit = compute_overall_compliance(run)
            per_run.append(overall)
            for k, (c, n) in per_sit.items():
                sit_totals[k][0] += c
                sit_totals[k][1] += n
        results[cfg] = np.array(per_run) * 100
        per_sit_aggr[cfg] = {k: (100.0 * v[0] / v[1]) if v[1] > 0 else float('nan')
                              for k, v in sit_totals.items()}
        print(f"  Per-run: {['%.2f' % v for v in results[cfg]]}")
        print(f"  Mean: {results[cfg].mean():.2f}%  Std: {results[cfg].std(ddof=1):.2f}%")

    # ===== 콘솔 요약 =====
    print("\n" + "=" * 70)
    print("Overall Compliance (16 vessels combined, raw)")
    print("=" * 70)
    print(f"  {'Config':<10} {'Mean (%)':>10} {'Std (%)':>10}")
    print("  " + "-" * 32)
    for cfg, lbl in zip(CONFIGS, CONFIG_LABELS):
        print(f"  {lbl:<10} {results[cfg].mean():>10.2f} {results[cfg].std(ddof=1):>10.2f}")

    print("\nPer-situation aggregated compliance (%):")
    print(f"  {'Config':<10} " + " ".join(f"{k:>14}" for k in SITUATIONS))
    for cfg, lbl in zip(CONFIGS, CONFIG_LABELS):
        row = f"  {lbl:<10} "
        for k in SITUATIONS:
            row += f"{per_sit_aggr[cfg][k]:>14.2f}"
        print(row)

    # ===== 그래프 =====
    os.makedirs(OUT_DIR, exist_ok=True)
    means = [results[cfg].mean() for cfg in CONFIGS]
    stds = [results[cfg].std(ddof=1) for cfg in CONFIGS]
    x = np.arange(len(CONFIGS))

    # Figure 1: 0~100 풀 스케일 + zoom inset
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, means,
                  color='#6C5CE7', alpha=0.85, edgecolor='white', width=0.55)
    for bar, m in zip(bars, means):
        ax.annotate(f'{m:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 6), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIG_LABELS, fontsize=11)
    ax.set_ylabel('Compliance (%)', fontsize=11)
    ax.set_title('COLREGs Compliance Rate (16 vessels averaged)\n'
                 'Open Ocean, 10 runs × 2,000 steps per run',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    out_full_pdf = os.path.join(OUT_DIR, '1_colregs_compliance_avg.pdf')
    out_full_png = os.path.join(OUT_DIR, '1_colregs_compliance_avg.png')
    fig.savefig(out_full_pdf, bbox_inches='tight')
    fig.savefig(out_full_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_full_pdf}")
    print(f"Saved: {out_full_png}")

    # Figure 2: zoom (차이 더 잘 보이게)
    ymin = max(0, min(means) - max(stds) - 3)
    ymax = min(100, max(means) + max(stds) + 3)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars2 = ax2.bar(x, means,
                    color='#6C5CE7', alpha=0.85, edgecolor='white', width=0.55)
    for bar, m in zip(bars2, means):
        ax2.annotate(f'{m:.1f}%',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 6), textcoords='offset points',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(CONFIG_LABELS, fontsize=11)
    ax2.set_ylabel('Compliance (%)', fontsize=11)
    ax2.set_title('COLREGs Compliance Rate (16 vessels averaged) — zoomed\n'
                  'Open Ocean, 10 runs × 2,000 steps per run',
                  fontsize=13, fontweight='bold')
    ax2.set_ylim(ymin, ymax)
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    out_zoom_pdf = os.path.join(OUT_DIR, '1_colregs_compliance_avg_zoom.pdf')
    out_zoom_png = os.path.join(OUT_DIR, '1_colregs_compliance_avg_zoom.png')
    fig2.savefig(out_zoom_pdf, bbox_inches='tight')
    fig2.savefig(out_zoom_png, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {out_zoom_pdf}")
    print(f"Saved: {out_zoom_png}")


if __name__ == '__main__':
    main()
