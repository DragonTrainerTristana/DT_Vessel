"""summarize.py — scripted-ship eval table (2026-10-01). Parses with runs/2026-09-30_t/report_parse.parse_eval
(the parser behind the t_ report tables), so scripted and learned numbers share one definition.

Usage: python summarize.py <dir with sc_<policy>_e<seed>.txt>   (Mac or Windows, parsing only)
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '2026-09-30_t'))
from report_parse import parse_eval  # noqa: E402

# report section 2 metrics + 2026-10-01 COLREGs·rudder check: (key, label, +1 higher is better / -1 lower is better)
METRICS = [('vColl', '배끼리 충돌 %', -1), ('dcpa', '통과거리(DCPA) m', +1), ('minPass', '조우 최소통과 m', +1),
           ('goal', '도착 %', +1),
           ('w_score', 'COLREGs(Woerner) %', +1), ('w_giveway', 'W 양보선 %', +1), ('w_headon', 'W 정면 %', +1),
           ('w_overtaking', 'W 추월 %', +1), ('w_standon', 'W 유지선 %', +1),
           ('C', 'COLREGs 준수(옛 지표) %', +1), ('C_headon', 'HeadOn 준수 %', +1), ('C_standon', 'StandOn 준수 %', +1),
           ('C_giveway', 'GiveWay 준수 %', +1), ('C_overtake', 'Overtake 준수 %', +1),
           ('rks_v2', '역할 판정기 %', +1), ('rks_v2x', '역할 판정기(해소 제외) %', +1),
           ('fuel', '연료(도착 ep)', -1), ('fleet_fuel', '도착 1회당 함대 연료', -1), ('fuel_per_m', '진행 1 m 당 연료', -1),
           ('head', '방향 바꾼 총량 °', -1), ('rud_trav', '타 이동량 °/ep(도착)', -1), ('rud_abs', '평균 |타각| °(도착)', -1),
           ('TO', '시간초과 %', -1), ('oColl', '벽 충돌 %', -1), ('epReward', 'epReward', +1)]
PAIRS = [('col300ih', 'col56h', '★Fig1 상한: 의도 공유 + 규칙 v2 300 m vs 규칙 v2 56 m'),
         ('col300i', 'col56', '의도 공유 300 m vs 56 m (규칙 v1)'),
         ('col300h', 'col56h', '규칙 v2: 보는 거리 효과 (의도 없음)'),
         ('col300ih', 'col300h', '규칙 v2 300 m: 의도 공유 효과'),
         ('col300i', 'col300', '규칙 v1 300 m: 의도 공유 효과'),
         ('vo300i', 'vo300', 'VO 300 m: 의도 공유 효과'),
         ('col300', 'col56', 'COLREGs 규칙 배: 통신 정보 효과 (규칙·H 같음, 보는 거리만 다름)'),
         ('vo300', 'vo56h150', 'VO 규칙 배: 통신 정보 효과 (H 같음, 거리만 다름)'),
         ('vo300', 'vo56', 'VO: 거리 + H 합산'),
         ('vo56h150', 'vo56', 'VO: 내다보는 시간 효과 (거리 같음)'),
         ('col56', 'vo56h150', '56 m: COLREGs 규칙 추가 효과'),
         ('col300', 'vo300', '300 m: COLREGs 규칙 추가 효과')]
NDEC = {'fuel': 0, 'head': 0, 'epReward': 0, 'fleet_fuel': 0, 'fuel_per_m': 3, 'rud_abs': 2}
RUD_RE = re.compile(r"\[scripted-rudder\] goal-ep n=\d+ rudderTravel=\s*([-\d.na]+)deg/ep meanAbsRudder=\s*([-\d.na]+)deg")
W_RE = re.compile(r"\[woerner-COLREGs\] score=\s*([\d.na]+)%")
W_ROLE_RE = {k: re.compile(k + r"=\s*([\d.na]+)%\(") for k in ('giveway', 'headon', 'overtaking', 'standon')}
MODE_RE = re.compile(r"\[scripted-mode\] (.*?) \(ship-decisions")


def parse_all(f):
    r = parse_eval(f)
    s = open(f, encoding='utf-8', errors='replace').read()
    m = RUD_RE.search(s)
    if m:
        try:
            r['rud_trav'], r['rud_abs'] = float(m.group(1)), float(m.group(2))
        except ValueError:
            pass
    m = W_RE.search(s)
    if m:
        try:
            r['w_score'] = float(m.group(1))
        except ValueError:
            pass
    for k, rx in W_ROLE_RE.items():
        m = rx.search(s)
        if m:
            try:
                r['w_' + k] = float(m.group(1))
            except ValueError:
                pass
    m = MODE_RE.search(s)
    r['mode'] = m.group(1) if m else None
    return r


def fmt(v, nd=1):
    return 'n/a' if v is None else f"{v:.{nd}f}"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out')
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, 'sc_*_e*.txt'))):
        m = re.match(r'sc_(\w+?)_e(\d+)\.txt$', os.path.basename(f))
        r = parse_all(f)
        if m and 'goal' in r:
            runs.setdefault(m.group(1), {})[int(m.group(2))] = r
    learned = {}
    tdir = os.path.join(HERE, '..', '2026-09-30_t', 'out')
    for arm in ('off', 'a6'):
        # ★2026-10-02 lr_t_<arm>_s4?.txt = 같은 체크포인트를 감싸기 평가로 다시 잰 것(Woerner·타 지표 포함). 있으면 그것을 씀
        fs = sorted(glob.glob(os.path.join(d, f'lr_t_{arm}_s4?.txt'))) or sorted(glob.glob(os.path.join(tdir, f'eval_t_{arm}_s4?.txt')))
        for f in fs:
            r = parse_all(f)
            if 'goal' in r:
                learned.setdefault(f't_{arm}', {})[os.path.basename(f)] = r

    print('# 규칙 배 평가 — 보고서 지표 전부 (eval_ckpt.py 같은 코드, t_ 주 평가와 같은 조건)\n')
    print('평균(시드별 값). 학습 배 t_off·t_a6 = 기존 주 평가(eval seed 999, 학습 시드 43·46·47) 참고용.\n')
    cols = ['정책'] + [m[1] for m in METRICS]
    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * len(cols))
    for name, rs in list(runs.items()) + list(learned.items()):
        cells = [name]
        for k, _, _ in METRICS:
            vals = [r.get(k) for r in rs.values()]
            vs = [v for v in vals if v is not None]
            nd = NDEC.get(k, 1)
            cells.append(fmt(sum(vs) / len(vs), nd) + ' (' + '·'.join(fmt(v, nd) for v in vals) + ')' if vs else 'n/a')
        print('| ' + ' | '.join(cells) + ' |')

    for a, b, title in PAIRS:
        if a not in runs or b not in runs:
            print(f'\n## {a} vs {b} — {title}: 데이터 없음')
            continue
        seeds = sorted(set(runs[a]) & set(runs[b]))
        print(f'\n## {a} vs {b} — {title} (같은 평가 시드 짝 {len(seeds)}개)\n')
        print(f'| 지표 | {a} 평균 | {b} 평균 | {a} 승/짝 |')
        print('|---|---|---|---|')
        for k, lab, sgn in METRICS:
            pa = [runs[a][s].get(k) for s in seeds]
            pb = [runs[b][s].get(k) for s in seeds]
            ok = [(x, y) for x, y in zip(pa, pb) if x is not None and y is not None]
            if not ok:
                print(f'| {lab} | n/a | n/a | n/a |')
                continue
            win = sum(1 for x, y in ok if (x - y) * sgn > 0)
            nd = NDEC.get(k, 1)
            ma = sum(x for x, _ in ok) / len(ok)
            mb = sum(y for _, y in ok) / len(ok)
            print(f'| {lab} | {fmt(ma, nd)} | {fmt(mb, nd)} | {win}/{len(ok)} |')
    modes = [(nm, rs) for nm, rs in runs.items() if any(r.get('mode') for r in rs.values())]
    if modes:
        print('\n## COLREGs 규칙 배 행동 비율 (burn-in 뒤 배-결정)\n')
        for nm, rs in modes:
            for sd, r in sorted(rs.items()):
                print(f"- {nm} e{sd}: {r.get('mode')}")
    print('\n주의: 규칙 배(VO)는 보이는 배의 위치·속도 참값을 씀(레이더 ray 아님). 시드 3개라 승패 수와 함께만 읽을 것.')


if __name__ == '__main__':
    main()
