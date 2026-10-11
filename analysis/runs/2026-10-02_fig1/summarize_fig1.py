"""summarize_fig1.py — Fig1 table (spec docs/superpowers/specs/2026-10-02-fig1-latent-design.md §3).

Reads lr_<PRE><arm>_s<seed>.txt (eval_scripted.py --policy learned: eval_ckpt metrics + Woerner + rudder lines) for
arms off · comm · offb. Rule (2026-09-30 spec §7): comm wins a metric if it is better on all same-seed pairs (n/n) AND
|mean diff| > N, N = max(off seed range, |offb − off| mean, fixed floor). Per-seed values are printed next to every mean.
Best-OFF pool (spec §3, as 2026-09-30 §7): for goal · vColl · DCPA every comm seed must also beat the best pool mean,
pool = this batch's off and offb + earlier learned t_off (lr_t_off_s*.txt in runs/2026-10-01_scripted/out), healthy only (goal >= 50).

Usage: python summarize_fig1.py <dir> [prefix f_]
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '2026-10-01_scripted'))
from summarize import parse_all  # noqa: E402

MAIN = [('vColl', '배끼리 충돌 %', -1, 12.1), ('goal', '도착 %', +1, 0.0), ('dcpa', 'DCPA(통과거리) m', +1, 1.5),
        ('w_score', 'COLREGs(Woerner) %', +1, 0.0), ('fuel', '연료(도착 ep)', -1, 0.0),
        ('fleet_fuel', '도착 1회당 함대 연료', -1, 0.0)]
POOL_KEYS = ('goal', 'vColl', 'dcpa')
AUX = [('head', '방향 바꾼 총량 °', -1, 0.0), ('rud_trav', '타 이동량 °/ep', -1, 0.0), ('C', '옛 COLREGs %', +1, 0.0),
       ('TO', '시간초과 %', -1, 0.0), ('oColl', '벽 충돌 %', -1, 0.0), ('epReward', 'epReward', +1, 0.0)]


def fmt(v, nd=1):
    return 'n/a' if v is None else f"{v:.{nd}f}"


def load(d, pre, arm):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, f'lr_{pre}{arm}_s*.txt'))):
        m = re.search(r'_s(\d+)\.txt$', f)
        r = parse_all(f)
        if m and 'goal' in r:
            out[int(m.group(1))] = r
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out')
    pre = sys.argv[2] if len(sys.argv) > 2 else 'f_'
    off, comm, offb = load(d, pre, 'off'), load(d, pre, 'comm'), load(d, pre, 'offb')
    seeds = sorted(set(off) & set(comm))
    print(f"# Fig1 — 통신(comm) vs OFF (같은 trunk 짝 {len(seeds)}개: {seeds})\n")
    print("판정(결과 전 고정): 같은 시드 짝 전부 승 + |평균차| > N, N = max(OFF 시드 범위, |offb − off| 평균, 고정 하한)\n")
    pool = {'off': off, 'offb': offb, 't_off': load(os.path.join(HERE, '..', '2026-10-01_scripted', 'out'), 't_', 'off')}
    pool_mean = {}
    for nm, runs in pool.items():
        g = [r.get('goal') for r in runs.values() if r.get('goal') is not None]
        if runs and g and sum(g) / len(g) >= 50.0:
            pool_mean[nm] = {k: sum(r[k] for r in runs.values() if r.get(k) is not None)
                                / max(1, sum(1 for r in runs.values() if r.get(k) is not None)) for k in POOL_KEYS}
    print(f"최선 OFF 풀(goal ≥ 50 만): {sorted(pool_mean) or '없음'}\n")
    win_main = 0
    for title, rows in (('주 지표', MAIN), ('보조 지표', AUX)):
        print(f"## {title}\n")
        print('| 지표 | off 평균 (시드별) | comm 평균 (시드별) | 승/짝 | 평균차 | N | 통과 |')
        print('|---|---|---|---|---|---|---|')
        for k, lab, sg, floor in rows:
            pa = [(off[s].get(k), comm[s].get(k)) for s in seeds]
            pa = [(x, y) for x, y in pa if x is not None and y is not None]
            if not pa:
                print(f'| {lab} | n/a | n/a | n/a | n/a | n/a | n/a |')
                continue
            w = sum(1 for x, y in pa if (y - x) * sg > 0)
            mo = sum(x for x, _ in pa) / len(pa)
            mc = sum(y for _, y in pa) / len(pa)
            rng = max(x for x, _ in pa) - min(x for x, _ in pa)
            nb = [abs(offb[s].get(k) - off[s].get(k)) for s in seeds if s in offb and offb[s].get(k) is not None]
            N = max(rng, (sum(nb) / len(nb)) if nb else 0.0, floor)
            ok = w == len(pa) and abs(mc - mo) > N
            pool_note = ''
            if k in POOL_KEYS and pool_mean:
                who, best = max(((nm, pm[k]) for nm, pm in pool_mean.items()), key=lambda t: sg * t[1])
                pok = all(sg * (comm[s][k] - best) > 0 for s in seeds if comm[s].get(k) is not None)
                ok = ok and pok
                pool_note = f" · 풀 최선 {who} {fmt(best, 1)} {'넘음' if pok else '못 넘음'}"
            win_main += int(ok and title == '주 지표')
            nd = 0 if k in ('fuel', 'fleet_fuel', 'head', 'epReward') else 1
            print(f"| {lab} | {fmt(mo, nd)} ({'·'.join(fmt(x, nd) for x, _ in pa)}) | {fmt(mc, nd)} ({'·'.join(fmt(y, nd) for _, y in pa)}) "
                  f"| {w}/{len(pa)} | {fmt(mc - mo, 2)} | {fmt(N, 2)} | {'**통과**' if ok else '불통과'}{pool_note} |")
        print()
    print(f"주 지표 통과: {win_main}/{len(MAIN)}")


if __name__ == '__main__':
    main()
