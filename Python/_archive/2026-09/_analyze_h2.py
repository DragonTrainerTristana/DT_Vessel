# H2 MSG_DIM 2~12 수렴분석 (2026-06-24). 각 dim 결과폴더의 metric.csv 마지막 N% 수렴창에서 ground-truth 집계.
# metric.csv 17열(0-idx): 0 id,1 ep,2 outcome,3 steps,4 fuel,5 rudderVar,6 compliance,7 occlRate,
#   8 commandVar,9 minVesselDist,10 nearMissSteps,11 straightness,12 headingTravel,13 minDCPA,14 dcpaBelowSteps,
#   15 fuelThrust,16 fuelTurn. outcome ∈ {goal, collision_vessel, collision_obstacle, timeout}.
import os, glob, csv, sys
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results"))
FRAC = 0.30   # 수렴창 = 마지막 30% 에피소드

def dim_of(name):
    import re
    m = re.search(r'msgc5c(\d+)_', name)
    return int(m.group(1)) if m else -1

folders = [d for d in glob.glob(os.path.join(BASE, "*msgc5c*_s42")) if os.path.isdir(d)]
rows = []
for d in sorted(folders, key=lambda p: dim_of(os.path.basename(p))):
    dim = dim_of(os.path.basename(d))
    mpath = os.path.join(d, "metric.csv")
    if not os.path.exists(mpath):
        continue
    recs = []
    with open(mpath, newline='') as f:
        for r in csv.reader(f):
            if len(r) < 17:
                continue
            recs.append(r)
    if not recs:
        continue
    n = len(recs)
    k = max(1, int(n * FRAC))
    win = recs[-k:]
    outc = [r[2] for r in win]
    def rate(o): return 100.0 * sum(1 for x in outc if x == o) / len(win)
    def meanf(col, cond=None):
        vals = [float(r[col]) for r in win if (cond is None or cond(r))]
        return float(np.mean(vals)) if vals else float('nan')
    goal_rows = [r for r in win if r[2] == 'goal']
    rows.append({
        'MSG_DIM': dim, 'eps_total': n, 'win_eps': k,
        'goal%': round(rate('goal'), 1),
        'vColl%': round(rate('collision_vessel'), 1),
        'oColl%': round(rate('collision_obstacle'), 1),
        'timeout%': round(rate('timeout'), 1),
        'minDCPA': round(meanf(13), 2),
        'dcpaBelow': round(meanf(14), 0),
        'minVesDist': round(meanf(9), 2),
        'compliance': round(meanf(6), 3),
        # 연료/궤적은 goal 달성분만(공정 비교): 도달 못하면 fuel 의미 다름
        'fuel_goal': round(meanf(4, lambda r: r[2] == 'goal'), 0),
        'steps_goal': round(meanf(3, lambda r: r[2] == 'goal'), 0),
    })

cols = ['MSG_DIM','eps_total','win_eps','goal%','vColl%','oColl%','timeout%',
        'minDCPA','dcpaBelow','minVesDist','compliance','fuel_goal','steps_goal']
print("=== H2 MSG_DIM sweep — 수렴창(마지막 {:.0%}) ground-truth, single seed s42 ===".format(FRAC))
print("  ".join(f"{c:>10}" for c in cols))
for r in rows:
    print("  ".join(f"{str(r[c]):>10}" for c in cols))
print("\n⚠️ single seed(n=1) — 곡선 *모양* 참고용, 유의성 아님. OFF 바닥 없음(run_sweep_c5c 필요). anti-Schelling 미통제.")
