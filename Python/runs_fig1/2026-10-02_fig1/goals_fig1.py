"""goals_fig1.py — author success criteria, 3-seed verdict (spec §11, was 5 seeds; spec docs/superpowers/specs/2026-11-02-author-goals-criteria.md, fixed before results).

Reads, per batch, lrm_<prefix>{off,comm}_s<seed>.txt (mean-action main eval, end model) and _curve.md (curve M).
Goal metrics a) Woerner COLREGs, b) DCPA, c) mean |rudder|: success = comm wins >= 3/3 paired seeds AND the 95 % t-CI
(df n-1) of the paired difference excludes 0 on comm's side. Goal 1 (reward curve rises): comm arm passes curve M in >= 3/3.
Guard: mean paired difference arrival >= -3 pt and ship collision <= +3 pt; if it fails, no a-c superiority claim.
Report only (no verdict): optional seed subsets, e.g. healthy trunks by qd_ spec §2 (--subset 'trunk 건강=43,45').

Usage: python goals_fig1.py <out_dir>:<prefix> [...] [--subset NAME=s1,s2 ...] [--kind lrm16]   e.g. _repro_out_qr:qr_ _repro_out_qr2:qr2_ _repro_out_qr3:qr3_
       --kind lrm16 = 16M (original endpoint) files kept by `_run_f.sh extend`; default lrm = current end model (24M after extend)
       python goals_fig1.py --selftest
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_fig1 import load, fmt  # noqa: E402  (same eval-file parser as the Fig1 table)

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
GOALS = [('bin', 'a) COLREGs 이진 준수율 %', +1), ('w_score', 'a2) COLREGs Woerner %', +1),   # ★2026-11-04 §9: a) = 이진, Woerner 는 같이 판정·보고
         ('dcpa', 'b) DCPA(통과거리) m', +1), ('rud_abs', 'c) 평균 타각 크기 °', -1)]
GUARD = [('goal', '도착 %', +1), ('vColl', '배끼리 충돌 %', -1)]
MARGIN, NEED_WIN, N_SEEDS = 3.0, 3, 3   # ★2026-11-05 스펙 §11: 시드 43–45(Windows DESKTOP) 3개 · 승 3/3 (전 = 5 시드 · 4/5)


def paired(diffs):
    """diffs = comm - off per seed → (mean, lo, hi) with 95 % t-CI; CI None if n < 2."""
    n = len(diffs)
    m = sum(diffs) / n
    if n < 2:
        return m, None, None
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    h = T95[n - 1] * sd / math.sqrt(n)
    return m, m - h, m + h


def curve_m(d):
    """_curve.md → {(arm, seed): True/False}; '우상향' = M pass (curve_fig1.py)."""
    out, p = {}, os.path.join(d, '_curve.md')
    if os.path.exists(p):
        for ln in open(p, encoding='utf-8', errors='replace'):
            m = re.match(r'\|\s*(off|comm)\s*\|\s*(\d+)\s*\|.*\|\s*([^|]+?)\s*\|\s*$', ln)
            if m:
                out[(m.group(1), int(m.group(2)))] = m.group(3) == '우상향'
    return out


def main(pairs, subsets=(), kind='lrm'):
    off, comm, curve = {}, {}, {}
    for dp in pairs:
        d, pre = dp.rsplit(':', 1)
        off.update(load(d, pre, 'off', kind))
        comm.update(load(d, pre, 'comm', kind))
        for arm, dst in (('off', off), ('comm', comm)):   # ★2026-11-04 이진 준수율 = 같은 끝 모델 재평가 파일 <kind>b_*(lrmb_) 에서만 가져옴
            for s, r in load(d, pre, arm, kind + 'b').items():
                if s in dst and r.get('bin') is not None:
                    dst[s]['bin'] = r['bin']
        curve.update(curve_m(d))
    seeds = sorted(set(off) & set(comm))
    full = len(seeds) == N_SEEDS
    print(f"# 저자 성공 지표 — 통신 vs OFF, 주 평가 {kind}(lrm = 평균 행동) (짝 {len(seeds)}개: {seeds})\n")
    print(f"판정(결과 전 고정, 스펙 2026-11-02): 지표마다 승 ≥ {NEED_WIN}/{N_SEEDS} + 95 % t-CI 가 0 을 안 걸침 · "
          f"가드 = 짝 차 평균 도착 ≥ −{MARGIN:g} pt · 배 충돌 ≤ +{MARGIN:g} pt"
          + ('' if full else f" — ★시드 {len(seeds)}개라 판정 안 함({N_SEEDS} 시드 필요), 값만 적음") + "\n")
    print('| 지표 | off (시드별) | comm (시드별) | 짝 차 (시드별) | 승/짝 | 평균차 [95 % CI] | 판정 |')
    print('|---|---|---|---|---|---|---|')
    verdict = {}
    for rows, kind in ((GOALS, 'goal'), (GUARD, 'guard')):
        for k, lab, sg in rows:
            ss = [s for s in seeds if off[s].get(k) is not None and comm[s].get(k) is not None]
            if not ss:
                print(f'| {lab} | n/a | n/a | n/a | n/a | n/a | n/a |')
                continue
            dd = [comm[s][k] - off[s][k] for s in ss]
            w = sum(1 for x in dd if sg * x > 0)
            m, lo, hi = paired(dd)
            if kind == 'goal':
                ok = full and len(ss) == N_SEEDS and w >= NEED_WIN and lo is not None and (lo > 0 if sg > 0 else hi < 0)
            else:
                ok = full and len(ss) == N_SEEDS and sg * m >= -MARGIN
            verdict[k] = ok
            ci = f"[{fmt(lo, 2)}, {fmt(hi, 2)}]" if lo is not None else '[n/a]'
            tag = ('**성공**' if ok else '불성공') if kind == 'goal' else ('가드 통과' if ok else '★가드 불통과')   # noqa: kind = 행 종류(goal/guard)
            print(f"| {lab} | {fmt(sum(off[s][k] for s in ss) / len(ss))} ({'·'.join(fmt(off[s][k]) for s in ss)}) "
                  f"| {fmt(sum(comm[s][k] for s in ss) / len(ss))} ({'·'.join(fmt(comm[s][k]) for s in ss)}) "
                  f"| {'·'.join(f'{x:+.1f}' for x in dd)} | {w}/{len(ss)} | {fmt(m, 2)} {ci} | {tag if full else '–'} |")
    cm = [s for s in seeds if curve.get(('comm', s)) is not None]
    nc = sum(1 for s in cm if curve[('comm', s)])
    no = sum(1 for s in seeds if curve.get(('off', s)))
    ok1 = full and len(cm) == N_SEEDS and nc >= NEED_WIN
    print(f"| 1) 곡선 우상향(M) | {no}/{len(seeds)} 시드 | {nc}/{len(cm)} 시드 | – | – | – | "
          f"{('**성공**' if ok1 else '불성공') if full else '–'} |\n")
    guard = all(verdict.get(k, False) for k, _, _ in GUARD)
    if full:
        print(f"가드: {'통과' if guard else '★불통과 — a–c 를 통과해도 통신 우월 주장 안 함'} · "
              f"성공 지표: 1) {'성공' if ok1 else '불성공'} · "
              + ' · '.join(f"{lab.split()[0]} {'성공' if verdict.get(k) else '불성공'}" for k, lab, _ in GOALS) + "\n")
    for nm, sub in subsets:   # ★2026-11-02 qr_ 재실행: 부분 집합은 인자로(판정 아님)
        ss = [s for s in seeds if s in sub]
        if not ss:
            continue
        parts = []
        for k, lab, sg in GOALS + GUARD:
            dd = [comm[s][k] - off[s][k] for s in ss if off[s].get(k) is not None and comm[s].get(k) is not None]
            if dd:
                short = re.sub(r'^[a-d]\) ', '', lab)
                parts.append(f"{short} {sum(dd) / len(dd):+.2f} (승 {sum(1 for x in dd if sg * x > 0)}/{len(dd)})")
        print(f"- 같이 적음(판정 아님) {nm} {ss}: " + ' · '.join(parts))
    print("\n시드 1–2개로 단독 주장 안 함. 표본 행동(lr) 표·10-02 N 규칙 표(_fig1.md)는 배치별로 같이 적음")


def selftest():
    m, lo, hi = paired([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m == 3.0 and abs((hi - lo) / 2 - 2.776 * math.sqrt(2.5) / math.sqrt(5)) < 1e-9
    assert paired([2.0])[1] is None
    print('selftest OK')


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    elif len(sys.argv) > 1:
        a = sys.argv[1:]
        opt = {'--subset', '--kind', '--interim'}
        if '--interim' in a:   # ★2026-11-05 스펙 §10: 서버 시드(46·47) 접근 불가 동안 시드 3개 중간 판정 = 승 3/3 + 95 % t-CI(df 2)
            N_SEEDS = NEED_WIN = int(a[a.index('--interim') + 1])
            print(f"> ★ 시드 {N_SEEDS}개 · 승 {NEED_WIN}/{N_SEEDS} + 95 % t-CI (스펙 §10·§11, §11 부터 기본값과 같음)\n")
        subs = [(v.split('=')[0], tuple(int(x) for x in v.split('=')[1].split(','))) for k, v in zip(a, a[1:]) if k == '--subset']
        kind = next((v for k, v in zip(a, a[1:]) if k == '--kind'), 'lrm')
        main([x for i, x in enumerate(a) if x not in opt and (i == 0 or a[i - 1] not in opt)], subs, kind)
    else:
        print(__doc__)
