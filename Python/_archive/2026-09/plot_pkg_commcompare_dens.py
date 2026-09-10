# -*- coding: utf-8 -*-
"""pkg commON vs commOFF 5-패널 비교 figure (3-seed paired, metric.csv 17열 ground-truth).

⚠️ 어떤 보정/합성도 없음 — metric.csv 원본 수렴구간(last-25%) seed별 집계 후
   seed 간 mean±95%CI (t-기반). seed 페어링: ON_s42↔OFF_s42, s43, s44.

metric.csv 17열 (0-index):
  0 id,1 ep,2 outcome,3 steps,4 fuel,5 rudderVar,6 compliance,7 occlRate,8 commandVar,
  9 minVesselDist,10 nearMissSteps,11 straightness,12 headingTravel,
  13 minDCPA,14 dcpaBelowSteps,15 fuelThrust,16 fuelTurn
  outcome ∈ {goal, collision_vessel, collision_obstacle, timeout}

5 패널:
  (1) episode_time = goal_steps (goal-only) × STEP_SEC
  (2) compliance (전체)
  (3) fuelTurn (goal-only)
  (4) minDCPA (approaching pairs, >=0)
  (5) goal% / vColl% (전체)
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_OUTCOME, C_STEPS, C_FUEL, C_COMP = 2, 3, 4, 6
C_MINDCPA, C_FUELTURN = 13, 16
STEP_SEC = 0.04  # 1 Academy step = 0.04s
WINDOW = 0.25    # 수렴구간 = 마지막 25% 에피소드

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results"))
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "figures", "dens_v1"))

RUNS = {
    'commON': {
        42: '20260621_190026_dens_commON_s42',
        43: '20260621_190031_dens_commON_s43',
        44: '20260621_190036_dens_commON_s44',
    },
    'commOFF': {
        42: '20260621_190041_dens_commOFF_s42',
        43: '20260621_190047_dens_commOFF_s43',
        44: '20260621_190052_dens_commOFF_s44',
    },
}
SEEDS = [42, 43, 44]
COLORS = {'commON': '#3498DB', 'commOFF': '#E74C3C'}
# t_{0.975, df=2} for n=3 seeds (df = n-1 = 2)
T95_DF2 = 4.302653


def read_rows(run_dir):
    rows = []
    path = os.path.join(RESULTS_DIR, run_dir, 'metric.csv')
    with open(path, encoding='utf-8', errors='ignore') as f:
        for r in csv.reader(f):
            if len(r) >= 17:
                rows.append(r)
    if not rows:
        raise RuntimeError(f"empty/short metric.csv: {path}")
    rows = rows[int(len(rows) * (1.0 - WINDOW)):]
    return rows


def seed_mean(rows, idx, pred=None):
    """수렴구간 에피소드들에서 idx 열의 평균(조건 pred 만족분)."""
    vals = []
    for r in rows:
        try:
            v = float(r[idx])
        except (ValueError, IndexError):
            continue
        if pred is None or pred(r, v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def seed_rate(rows, outcome):
    n = len(rows)
    if n == 0:
        return np.nan
    c = sum(1 for r in rows if r[C_OUTCOME] == outcome)
    return 100.0 * c / n


def agg_seeds(per_seed_vals):
    """seed별 스칼라 리스트 → (mean, 95%CI half-width). nan 무시."""
    a = np.array([v for v in per_seed_vals if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return np.nan, 0.0
    m = float(np.mean(a))
    if len(a) < 2:
        return m, 0.0
    sem = float(np.std(a, ddof=1) / np.sqrt(len(a)))
    ci = T95_DF2 * sem if len(a) == 3 else 1.96 * sem
    return m, ci


def collect_metric(cond_rows, idx, pred=None, scale=1.0):
    """조건별(commON/commOFF) seed mean → (mean, ci, [seed vals])."""
    per_seed = [seed_mean(cond_rows[s], idx, pred) * scale for s in SEEDS]
    m, ci = agg_seeds(per_seed)
    return m, ci, per_seed


def collect_rate(cond_rows, outcome):
    per_seed = [seed_rate(cond_rows[s], outcome) for s in SEEDS]
    m, ci = agg_seeds(per_seed)
    return m, ci, per_seed


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 조건별 seed별 rows 로드
    rows = {}
    for cond, seedmap in RUNS.items():
        rows[cond] = {}
        for s, d in seedmap.items():
            rows[cond][s] = read_rows(d)

    goal_only = lambda r, v: r[C_OUTCOME] == 'goal'
    approaching = lambda r, v: v >= 0  # minDCPA 유효(조우 있음)

    # 각 패널: (title, ylabel, kind, args)
    panels = []

    # (1) episode time (goal-only)
    on = collect_metric(rows['commON'], C_STEPS, goal_only, scale=STEP_SEC)
    off = collect_metric(rows['commOFF'], C_STEPS, goal_only, scale=STEP_SEC)
    panels.append(('Episode Time (goal episodes)', 'time to goal (s)', on, off))

    # (2) compliance (전체)
    on = collect_metric(rows['commON'], C_COMP)
    off = collect_metric(rows['commOFF'], C_COMP)
    panels.append(('COLREGs Compliance', 'compliance (0-1)', on, off))

    # (3) fuelTurn (goal-only)
    on = collect_metric(rows['commON'], C_FUELTURN, goal_only)
    off = collect_metric(rows['commOFF'], C_FUELTURN, goal_only)
    panels.append(('Turn Fuel Cost (goal episodes)', 'fuelTurn (sum 0.5 turn^2)', on, off))

    # (4) minDCPA (approaching pairs)
    on = collect_metric(rows['commON'], C_MINDCPA, approaching)
    off = collect_metric(rows['commOFF'], C_MINDCPA, approaching)
    panels.append(('Min DCPA (approaching pairs)', 'min DCPA (m)', on, off))

    # (5) goal% / vColl% (전체) — 듀얼 metric 패널
    on_goal = collect_rate(rows['commON'], 'goal')
    off_goal = collect_rate(rows['commOFF'], 'goal')
    on_vc = collect_rate(rows['commON'], 'collision_vessel')
    off_vc = collect_rate(rows['commOFF'], 'collision_vessel')

    # ---- 단일 figure, 5 subplot ----
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    conds = ['commON', 'commOFF']

    def draw_bar(ax, title, ylabel, on, off):
        means = [on[0], off[0]]
        cis = [on[1], off[1]]
        x = np.arange(2)
        bars = ax.bar(x, means, 0.55, yerr=cis, capsize=5,
                      color=[COLORS['commON'], COLORS['commOFF']],
                      edgecolor='black', linewidth=0.8)
        # seed별 점 오버레이
        for i, cond_vals in enumerate([on[2], off[2]]):
            xs = np.full(len([v for v in cond_vals if np.isfinite(v)]), x[i])
            ys = [v for v in cond_vals if np.isfinite(v)]
            ax.scatter(xs, ys, color='black', s=22, zorder=5, alpha=0.7)
        for i, m in enumerate(means):
            if np.isfinite(m):
                top = m + (cis[i] if np.isfinite(cis[i]) else 0)
                ax.text(i, top, f"{m:.3g}", ha='center', va='bottom', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(conds)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    for ax, (title, ylabel, on, off) in zip(axes[:4], panels):
        draw_bar(ax, title, ylabel, on, off)

    # 패널 5: goal% / vColl% grouped
    ax = axes[4]
    x = np.arange(2)  # 0=commON, 1=commOFF
    w = 0.35
    goal_m = [on_goal[0], off_goal[0]]
    goal_c = [on_goal[1], off_goal[1]]
    vc_m = [on_vc[0], off_vc[0]]
    vc_c = [on_vc[1], off_vc[1]]
    ax.bar(x - w / 2, goal_m, w, yerr=goal_c, capsize=4, color='#2ECC71',
           edgecolor='black', linewidth=0.8, label='goal %')
    ax.bar(x + w / 2, vc_m, w, yerr=vc_c, capsize=4, color='#C0392B',
           edgecolor='black', linewidth=0.8, label='vessel collision %')
    for i in range(2):
        ax.text(i - w / 2, goal_m[i] + goal_c[i], f"{goal_m[i]:.1f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(i + w / 2, vc_m[i] + vc_c[i], f"{vc_m[i]:.2f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel('%')
    ax.set_title('Outcome: goal% / vColl%', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('1M PROVISIONAL  dens: commON vs commOFF  (3-seed paired, convergence = last 25% '
                 'episodes, mean +- 95%CI across seeds, no correction)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png = os.path.join(OUT_DIR, 'dens_commcompare_5panels.png')
    fig.savefig(png, dpi=150)
    plt.close(fig)

    print("[saved]", png)
    print("[summary] (mean +- 95%CI across 3 seeds, last-25% window)")
    labels = ['episode_time(s)', 'compliance', 'fuelTurn', 'minDCPA(m)']
    for (title, ylabel, on, off), lab in zip(panels, labels):
        print(f"  {lab:16s} ON={on[0]:.4g}+-{on[1]:.3g}  OFF={off[0]:.4g}+-{off[1]:.3g}")
    print(f"  goal%            ON={on_goal[0]:.3g}+-{on_goal[1]:.3g}  OFF={off_goal[0]:.3g}+-{off_goal[1]:.3g}")
    print(f"  vColl%           ON={on_vc[0]:.3g}+-{on_vc[1]:.3g}  OFF={off_vc[0]:.3g}+-{off_vc[1]:.3g}")


if __name__ == '__main__':
    main()
