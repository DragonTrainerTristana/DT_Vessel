# -*- coding: utf-8 -*-
"""5대 지표 비교 figure 생성 (Fig12 후속, metric.csv ground-truth 직접 추출).

⚠️ 옛 gen_5graphs_*.py 대체: 그 계열은 'COMM ON>=OFF 보정'과 환경 보간(합성)이 박혀 있어 사용 금지.
   여기는 **어떤 보정도 없음** — metric.csv 원본 그대로 수렴구간 mean±95%CI.

입력: run 결과 폴더들(metric.csv 13열 구버전 / 15열 신버전 자동 감지)
  metric.csv: 0 id,1 ep,2 outcome,3 steps,4 fuel,5 rudderVar,6 compliance,7 occlRate,
              8 commandVar,9 minVesselDist,10 nearMissSteps,11 straightness,12 headingTravel
              [,13 minDCPA(-1=조우없음), 14 dcpaBelowSteps]  ← 2026-06-12 추가
출력: 1_colregs_compliance / 2_dcpa / 3_danger_proximity / 4_control_cost / 5_episode_time (png+pdf)

사용:
  python plot_5metrics_commcompare.py --out <figdir> commON=<dir1> commOFF=<dir2> [msg12=<dir3> ...]
  옵션: --window 0.25 (수렴구간 = 마지막 25% 에피소드)
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_OUTCOME, C_STEPS, C_FUEL, C_COMP = 2, 3, 4, 6
C_CMDVAR, C_MINVD, C_NEARMISS = 8, 9, 10
C_MINDCPA, C_DCPABELOW = 13, 14
STEP_SEC = 0.04  # myStepCount는 Academy step 단위 → 1 step = 0.04s


def read_metric(run_dir, window):
    rows = []
    path = os.path.join(run_dir, 'metric.csv')
    with open(path, encoding='utf-8', errors='ignore') as f:
        for r in csv.reader(f):
            if len(r) >= 13:
                rows.append(r)
    if not rows:
        raise SystemExit(f"empty metric.csv: {path}")
    rows = rows[int(len(rows) * (1.0 - window)):]   # 수렴구간(마지막 window 비율)
    has_dcpa = len(rows[0]) >= 15
    return rows, has_dcpa


def fcol(rows, idx, pred=None):
    out = []
    for r in rows:
        try:
            v = float(r[idx])
        except (ValueError, IndexError):
            continue
        if pred is None or pred(r, v):
            out.append(v)
    return np.array(out)


def mean_ci(arr):
    if len(arr) == 0:
        return np.nan, 0.0
    return float(np.mean(arr)), float(1.96 * np.std(arr) / max(np.sqrt(len(arr)), 1.0))


def bar_fig(title, ylabel, names, means, cis, out_base, annot=None):
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(names), 4.5))
    x = np.arange(len(names))
    colors = ['#3498DB' if 'on' in n.lower() else '#E74C3C' if 'off' in n.lower() else '#8e7cc3'
              for n in names]
    ax.bar(x, means, 0.6, yerr=cis, capsize=4, color=colors, edgecolor='black', linewidth=0.8)
    for i, m in enumerate(means):
        if np.isfinite(m):
            ax.text(i, m + (cis[i] if np.isfinite(cis[i]) else 0), f"{m:.3g}",
                    ha='center', va='bottom', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(ylabel)
    t = title if annot is None else f"{title}\n{annot}"
    ax.set_title(t)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base + '.png', dpi=150)
    fig.savefig(out_base + '.pdf')
    plt.close(fig)
    print(f"  saved {os.path.basename(out_base)}: " +
          ", ".join(f"{n}={m:.4g}+-{c:.2g}" for n, m, c in zip(names, means, cis)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='name=run_dir ...')
    ap.add_argument('--out', required=True)
    ap.add_argument('--window', type=float, default=0.25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data = {}
    for spec in a.runs:
        name, path = spec.split('=', 1)
        data[name] = read_metric(path, a.window)
    names = list(data.keys())
    wpct = int(a.window * 100)
    annot = f"convergence window = last {wpct}% episodes, mean+-95%CI, no correction"

    def collect(idx, pred=None):
        ms, cs = [], []
        for n in names:
            m, c = mean_ci(fcol(data[n][0], idx, pred))
            ms.append(m)
            cs.append(c)
        return ms, cs

    goal_only = lambda r, v: r[C_OUTCOME] == 'goal'

    # 1. COLREGs compliance (전체 에피소드)
    ms, cs = collect(C_COMP)
    bar_fig('COLREGs Compliance', 'compliance (0-1)', names, ms, cs,
            os.path.join(a.out, '1_colregs_compliance'), annot)

    # 2. DCPA — 신버전이면 minDCPA(예측, -1 제외), 구버전이면 minVesselDist(realized CPA) fallback
    if all(d[1] for d in data.values()):
        ms, cs = collect(C_MINDCPA, lambda r, v: v >= 0)
        bar_fig('Min DCPA (predicted, approaching pairs)', 'min DCPA (m)', names, ms, cs,
                os.path.join(a.out, '2_dcpa'), annot)
        ms, cs = collect(C_DCPABELOW)
        bar_fig('Collision-course dwell (DCPA<24m)', 'decisions below threshold', names, ms, cs,
                os.path.join(a.out, '2b_dcpa_below_steps'), annot)
    else:
        ms, cs = collect(C_MINVD, lambda r, v: v >= 0)
        bar_fig('Min vessel distance (realized CPA; old 13-col fallback)', 'min distance (m)',
                names, ms, cs, os.path.join(a.out, '2_dcpa'), annot)

    # 3. Danger proximity (near-miss 결정 수)
    ms, cs = collect(C_NEARMISS)
    bar_fig('Dangerous Proximity (near-miss steps)', 'near-miss decisions / episode', names, ms, cs,
            os.path.join(a.out, '3_danger_proximity'), annot)

    # 4. Control cost — goal 에피소드만(timeout이 fuel을 인위적으로 부풀리는 것 차단)
    ms, cs = collect(C_FUEL, goal_only)
    bar_fig('Control Cost (fuel, goal episodes only)', 'fuel proxy (sum thrust^2+0.5 turn^2)',
            names, ms, cs, os.path.join(a.out, '4_control_cost'), annot)

    # 5. Episode time — goal 에피소드만(도착 시간) + goal rate 별도 표기
    ms, cs = collect(C_STEPS, goal_only)
    ms = [m * STEP_SEC if np.isfinite(m) else m for m in ms]
    cs = [c * STEP_SEC for c in cs]
    rates = []
    for n in names:
        rows = data[n][0]
        g = sum(1 for r in rows if r[C_OUTCOME] == 'goal')
        rates.append(f"{n}: goal {100.0 * g / len(rows):.1f}%")
    bar_fig('Episode Time (goal episodes only)', 'time to goal (s)', names, ms, cs,
            os.path.join(a.out, '5_episode_time'), annot + '\n' + ' | '.join(rates))

    print("[done] figures ->", a.out)


if __name__ == '__main__':
    main()
