# 중간점검(1회성): 진행 중인 6 run의 last-25% window 지표 비교 (mid-training — 채택 금지, 경향만)
import csv
import glob
import os

import numpy as np

RES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results"))

def load(run_dir, frac=0.25):
    rows = []
    with open(os.path.join(run_dir, 'metric.csv'), encoding='utf-8', errors='ignore') as f:
        for r in csv.reader(f):
            if len(r) >= 15:
                rows.append(r)
    return rows[int(len(rows) * (1 - frac)):], len(rows)

def stats(rows):
    n = len(rows)
    out = {}
    oc = [r[2] for r in rows]
    out['n'] = n
    out['goal%'] = 100 * oc.count('goal') / n
    out['vColl%'] = 100 * oc.count('collision_vessel') / n
    out['timeout%'] = 100 * oc.count('timeout') / n
    f = lambda i, pred=None: np.array([float(r[i]) for r in rows if (pred is None or pred(r))])
    g = lambda r: r[2] == 'goal'
    out['fuel(goal)'] = f(4, g).mean() if oc.count('goal') else float('nan')
    out['time_s(goal)'] = f(3, g).mean() * 0.04 if oc.count('goal') else float('nan')
    out['compliance'] = f(6).mean()
    out['minVD'] = f(9, lambda r: float(r[9]) >= 0).mean()
    mind = f(13, lambda r: float(r[13]) >= 0)
    out['minDCPA'] = mind.mean() if len(mind) else float('nan')
    out['dcpaBelow'] = f(14).mean()
    return out

hdr = None
table = {}
for d in sorted(glob.glob(os.path.join(RES, '20260612_18*'))):
    name = os.path.basename(d).split('_', 2)[2]
    rows, total = load(d)
    s = stats(rows)
    s['total_ep'] = total
    table[name] = s
    if hdr is None:
        hdr = list(s.keys())

cols = ['n', 'total_ep', 'goal%', 'vColl%', 'timeout%', 'fuel(goal)', 'time_s(goal)',
        'compliance', 'minVD', 'minDCPA', 'dcpaBelow']
print('run        ' + ''.join(f'{c:>13}' for c in cols))
for name, s in table.items():
    print(f'{name:<11}' + ''.join(f'{s[c]:>13.2f}' if isinstance(s[c], float) else f'{s[c]:>13}' for c in cols))

print()
for met in ['goal%', 'vColl%', 'timeout%', 'fuel(goal)', 'time_s(goal)', 'compliance', 'minDCPA', 'dcpaBelow']:
    on = np.mean([table[f'commON_s{s}'][met] for s in (42, 43, 44)])
    off = np.mean([table[f'commOFF_s{s}'][met] for s in (42, 43, 44)])
    print(f'{met:<14} ON={on:9.2f}  OFF={off:9.2f}  diff(ON-OFF)={on - off:+9.2f}')
