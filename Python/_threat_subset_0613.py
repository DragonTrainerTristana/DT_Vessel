# 위협 조우 에피소드만 떼어서 ON vs OFF 비교 (전체평균은 위협無 에피소드에 희석됨).
# 통신 가치는 "먼 위협을 만난" 에피소드에서만 나타날 수 있음. dcpaBelow(충돌코스 체류 step)로 위협강도 분층.
import csv
import glob
import os
import numpy as np

RES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results"))
C_OUT, C_STEPS, C_FUEL, C_COMP, C_MINVD, C_MINDCPA, C_DCPABELOW = 2, 3, 4, 6, 9, 13, 14

def load(d, frac=0.25):
    rows = []
    with open(os.path.join(d, 'metric.csv'), encoding='utf-8', errors='ignore') as f:
        for r in csv.reader(f):
            if len(r) >= 15:
                rows.append(r)
    return rows[int(len(rows)*(1-frac)):]

def arm_rows(prefix):
    out = []
    for s in (42, 43, 44):
        d = glob.glob(os.path.join(RES, f'*{prefix}_s{s}'))[0]
        out += load(d)
    return out

on, off = arm_rows('commON'), arm_rows('commOFF')

def f(rows, i):
    return np.array([float(r[i]) for r in rows])

# 위협강도 = dcpaBelow (전 에피소드 통합 분위수로 컷)
allb = np.concatenate([f(on, C_DCPABELOW), f(off, C_DCPABELOW)])
q = {p: np.percentile(allb, p) for p in (50, 75, 90)}
print(f"dcpaBelow 분위: median={q[50]:.0f}, p75={q[75]:.0f}, p90={q[90]:.0f}  (전체 {len(allb)}ep)")
print()

def subset_stats(rows, lo, hi):
    sel = [r for r in rows if lo <= float(r[C_DCPABELOW]) < hi]
    if not sel:
        return None
    g = [r for r in sel if r[C_OUT] == 'goal']
    return {
        'n': len(sel),
        'goal%': 100*len(g)/len(sel),
        'fuel(goal)': np.mean([float(r[C_FUEL]) for r in g]) if g else float('nan'),
        'time(goal)s': np.mean([float(r[C_STEPS]) for r in g])*0.04 if g else float('nan'),
        'minDCPA': np.mean([float(r[C_MINDCPA]) for r in sel if float(r[C_MINDCPA])>=0]),
        'vColl%': 100*sum(1 for r in sel if r[C_OUT]=='collision_vessel')/len(sel),
    }

bands = [('위협無/약 (dcpaBelow<median)', 0, q[50]),
         ('위협中 (median~p75)', q[50], q[75]),
         ('위협高 (p75~p90)', q[75], q[90]),
         ('위협極 (>p90)', q[90], 1e18)]

for label, lo, hi in bands:
    so, sf = subset_stats(on, lo, hi), subset_stats(off, lo, hi)
    if not so or not sf:
        continue
    print(f"== {label}  [ON n={so['n']}, OFF n={sf['n']}] ==")
    for k in ('goal%', 'fuel(goal)', 'time(goal)s', 'minDCPA', 'vColl%'):
        d = so[k] - sf[k]
        print(f"   {k:14} ON={so[k]:9.2f}  OFF={sf[k]:9.2f}  diff={d:+8.2f}")
    print()
