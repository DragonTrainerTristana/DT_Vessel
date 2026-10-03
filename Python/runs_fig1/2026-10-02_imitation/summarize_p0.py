"""summarize_p0.py — P0 table for the imitation start (spec 2026-10-02 §5).

Student = DAgger checkpoint i_dagger_d6_sS.pt before any PPO, scored by the main eval (eval_ckpt, seed 999).
P0 per seed: goal >= 80 % and vessel collision <= 15 %. Pass if >= 3 of the seeds pass.
References (same eval code): rule ships goal / vo56 (runs/2026-10-01_scripted/out) and learned t_off (runs/2026-09-30_t/out).

Usage: python summarize_p0.py [dir with eval_i_dagger_d6_s*.txt]   (default: ./out)
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(RUNS, '2026-09-30_t'))
from report_parse import parse_eval  # noqa: E402

P0_GOAL, P0_VCOLL, P0_MIN_SEEDS = 80.0, 15.0, 3
COLS = [('goal', '도착 %'), ('vColl', '배끼리 충돌 %'), ('oColl', '벽 %'), ('TO', '시간초과 %'), ('dcpa', '통과거리 m'),
        ('C', 'COLREGs %'), ('fuel', '연료'), ('head', '방향 바꾼 총량 °')]


def fmt(v, nd=1):
    return 'n/a' if v is None else f"{v:.{nd}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def row(name, rs):
    cells = []
    for k, _ in COLS:
        nd = 0 if k in ('fuel', 'head') else 1
        cells.append(fmt(mean([r.get(k) for r in rs]), nd))
    return f"| {name} | " + ' | '.join(cells) + ' |'


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out')
    pre = sys.argv[2] if len(sys.argv) > 2 else 'i_'          # ★2026-10-02 접두어 인자(c_ = 목표 침로 배치). 기본 i_ = 그대로
    stu = {}
    for f in sorted(glob.glob(os.path.join(d, f'eval_{pre}dagger_d6_s*.txt'))):
        m = re.search(r'_s(\d+)\.txt$', f)
        r = parse_eval(f)
        if m and 'goal' in r:
            stu[int(m.group(1))] = r
    print('# P0 — 흉내만 낸 학생(PPO 전) 주 평가\n')
    print(f'기준(결과 전 고정): 시드마다 도착 ≥ {P0_GOAL:g} % 그리고 배끼리 충돌 ≤ {P0_VCOLL:g} %. {P0_MIN_SEEDS}시드 이상 통과 → phase1\n')
    print('| 시드 | ' + ' | '.join(lab for _, lab in COLS) + ' | P0 |')
    print('|' + '---|' * (len(COLS) + 2))
    npass = 0
    for s, r in sorted(stu.items()):
        ok = r.get('goal') is not None and r.get('vColl') is not None and r['goal'] >= P0_GOAL and r['vColl'] <= P0_VCOLL
        npass += ok
        print(f"| s{s} | " + ' | '.join(fmt(r.get(k), 0 if k in ('fuel', 'head') else 1) for k, _ in COLS)
              + f" | {'통과' if ok else '불통과'} |")
        side = os.path.join(d, f'{pre}dagger_d6_s{s}.imitation.json')
        if os.path.exists(side):
            info = json.load(open(side, encoding='utf-8'))
            dl = info.get('dagger_last') or {}
            print(f"|  ↳ DAgger 마지막 창 | 학생 운전 도착 {fmt(dl.get('student_goal'))} · 충돌 {fmt(dl.get('student_vColl'))} · "
                  f"타 일치 {fmt(dl.get('agree'))} % · 손실 {fmt(dl.get('loss'), 4)} | | | | | | | |")

    print('\n## 참고 (같은 평가 코드)\n')
    print('| 정책 | ' + ' | '.join(lab for _, lab in COLS) + ' |')
    print('|' + '---|' * (len(COLS) + 1))
    if stu:
        print(row('학생 평균(이번)', list(stu.values())))
    sc = os.path.join(RUNS, '2026-10-01_scripted', 'out')
    for pol in ('vo56', 'goal'):
        rs = [parse_eval(f) for f in sorted(glob.glob(os.path.join(sc, f'sc_{pol}_e*.txt')))]
        if rs:
            print(row(f'규칙 배 {pol}', rs))
    tt = [parse_eval(f) for f in sorted(glob.glob(os.path.join(RUNS, '2026-09-30_t', 'out', 'eval_t_off_s4?.txt')))]
    if tt:
        print(row('학습 t_off', tt))

    verdict = '통과' if npass >= P0_MIN_SEEDS else '불통과'
    print(f"\nP0 판정: {verdict} ({npass}/{len(stu)} 시드, 필요 {P0_MIN_SEEDS})")


if __name__ == '__main__':
    main()
