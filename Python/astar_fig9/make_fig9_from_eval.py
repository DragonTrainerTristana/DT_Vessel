"""eval_astar_global.py 출력 → Fig9 그림. 전부 실측이라 환산·추정이 0.

입력: eval_astar_global.py 가 찍은 결과 줄들을 모아둔 텍스트 파일.
      (한 줄 = 한 런. 시드·팔마다 한 줄씩)

읽는 값 (전부 그 런에서 측정된 것):
  len=   도착 에피소드 평균 결정수      → 항해시간 = len × 0.4 s
  fuel=  도착 에피소드 평균 Σ(v²+0.5δ²)
  plan=  배정된 전역경로 평균 길이(m)   → 항해거리. **이게 실제 거리라 곱셈이 필요 없음**
  goal=  도착률

곱하는 곳이 한 군데도 없음. 거리는 plan 에서, 시간은 len 에서 그대로 나옴.

사용:
    # 1) 런 결과 모으기 (시드 × 팔)
    for S in 42 43 44; do for A in OFF ON; do
        python eval_astar_global.py --ckpt ql_SE_START_s$S.pt --arm $A --path astar \\
            --envs 96 --vessels 16 --tag s$S >> runs_astar.txt
    done; done

    # 2) 그림
    python make_fig9_from_eval.py runs_astar.txt --title "Korea - Taiwan corridor"
"""
import argparse
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style  # noqa: E402

C_OFF, C_ON = '#C0504D', '#4472C4'
DECS = 0.4          # s per decision (DT 0.04 × SUBSTEPS 10)

LINE = re.compile(
    r'arm=(?P<arm>\w+)\s+path=(?P<path>\w+).*?'
    r'goal=\s*(?P<goal>[\d.]+)%.*?'
    r'vColl=\s*(?P<vcoll>[\d.]+)%.*?'
    r'fuel=\s*(?P<fuel>[\d.]+).*?'
    r'len=\s*(?P<len>\d+)')
PLAN = re.compile(r'plan=\s*([\d.]+)m')


def parse(paths):
    rows = []
    for p in paths:
        with open(p, encoding='utf-8', errors='replace') as fh:
            for ln in fh:
                m = LINE.search(ln)
                if not m:
                    continue
                d = m.groupdict()
                pl = PLAN.search(ln)
                rows.append(dict(arm=d['arm'], path=d['path'],
                                 goal=float(d['goal']), vcoll=float(d['vcoll']),
                                 fuel=float(d['fuel']), length=float(d['len']),
                                 plan=float(pl.group(1)) if pl else float('nan')))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+', help='eval_astar_global.py 결과 줄이 든 텍스트 파일')
    ap.add_argument('--group', default='arm', choices=['arm', 'path'],
                    help='arm=통신 OFF/ON 대조 (기본), path=전역경로 direct/astar 대조')
    ap.add_argument('--only', default='', help='한쪽 축 고정 (예: --group arm --only astar)')
    ap.add_argument('--title', default='', help='그림 제목. 비우면 조건에서 자동 생성')
    ap.add_argument('--subtitle', default='')
    ap.add_argument('--time_unit', default='auto', choices=['auto', 'days', 'hours', 'dec'])
    ap.add_argument('--out', default=os.path.join(HERE, '_fig9_candidate', 'fig9_measured'))
    args = ap.parse_args()

    rows = parse(args.files)
    if not rows:
        raise SystemExit('결과 줄을 못 찾음 — eval_astar_global.py 출력이 맞는지 확인할 것')

    if args.group == 'arm':
        keys, labels, colors = ('OFF', 'ON'), ('Comm OFF', 'Comm ON'), (C_OFF, C_ON)
        sel = lambda r, k: r['arm'] == k and (not args.only or r['path'] == args.only)
    else:
        keys, labels, colors = ('direct', 'astar'), ('No global path', 'A* global path'), (C_OFF, C_ON)
        sel = lambda r, k: r['path'] == k and (not args.only or r['arm'] == args.only)

    grp = {k: [r for r in rows if sel(r, k)] for k in keys}
    for k in keys:
        if not grp[k]:
            raise SystemExit(f'{k} 조건의 런이 없음 — 두 팔 다 돌렸는지 확인할 것')
    n_seed = min(len(grp[k]) for k in keys)

    F = {k: np.array([r['fuel'] for r in grp[k]], float) for k in keys}
    L = {k: np.array([r['length'] for r in grp[k]], float) for k in keys}
    G = {k: np.array([r['goal'] for r in grp[k]], float) for k in keys}
    P = np.array([r['plan'] for k in keys for r in grp[k]], float)
    dist_m = float(np.nanmean(P)) if np.isfinite(P).any() else float('nan')

    # 시간 단위 — 실제 크기에 맞춰 고름
    sec = {k: L[k] * DECS for k in keys}
    unit = args.time_unit
    if unit == 'auto':
        mx = max(sec[k].mean() for k in keys)
        unit = 'days' if mx >= 86400 else ('hours' if mx >= 3600 else 'dec')
    div, ylab, tfmt = {'days': (86400.0, 'Voyage time (days)', '{:.2f}'),
                       'hours': (3600.0, 'Voyage time (hours)', '{:.1f}'),
                       'dec': (1.0, 'Decisions to goal', '{:,.0f}')}[unit]
    T = {k: (sec[k] / div if unit != 'dec' else L[k]) for k in keys}

    paper_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    xs = np.array([0.0])
    w = 0.42
    for ax, D, yl, tt, fmt in ((axes[0], F, r'Fuel  $\Sigma(v^2+0.5\delta^2)$',
                                '(a) Fuel per voyage', '{:,.0f}'),
                               (axes[1], T, ylab, '(b) Voyage time', tfmt)):
        a, b = D[keys[0]].mean(), D[keys[1]].mean()
        ax.bar(xs - w / 2, a, w, color=colors[0], alpha=.85, label=labels[0], zorder=2)
        ax.bar(xs + w / 2, b, w, color=colors[1], alpha=.85, label=labels[1], zorder=2)
        ax.text(xs[0] - w / 2, a * .5, fmt.format(a), ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')
        ax.text(xs[0] + w / 2, b * .5, fmt.format(b), ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')
        top = max(a, b)
        ax.text(xs[0], top * 1.13, f'{100*(b-a)/a:+.1f}%', ha='center', va='bottom',
                fontsize=11, color=colors[1])
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{dist_m/1000:,.0f} km' if dist_m >= 1000
                            else f'{dist_m:,.0f} m'], fontsize=10)
        ax.set_ylabel(yl)
        ax.set_title(tt, loc='left', pad=8)
        ax.margins(x=.45)
        ax.set_ylim(0, top * 1.32)
    axes[0].legend(loc='upper left', fontsize=9)

    title = args.title or f'Voyage cost — {dist_m:,.0f} m route'
    fig.suptitle(title, fontsize=12, y=1.04)
    sub = args.subtitle or f'n = {n_seed} seeds'
    fig.text(.5, -.035, sub, ha='center', fontsize=8, color='0.55')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for e in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{e}', bbox_inches='tight')

    print(f'런 {len(rows)}개 → {args.group} 대조, 시드 {n_seed}')
    print(f'항해거리(plan 실측) {dist_m:,.1f} m')
    for k, lab in zip(keys, labels):
        print(f'  {lab:16s} fuel {F[k].mean():10,.1f}  time {T[k].mean():10.3f}  '
              f'goal {G[k].mean():5.1f}%   (n={len(grp[k])})')
    wf = int((F[keys[1]][:n_seed] < F[keys[0]][:n_seed]).sum())
    wt = int((T[keys[1]][:n_seed] < T[keys[0]][:n_seed]).sum())
    print(f'  시드별 승: 연료 {wf}/{n_seed}, 시간 {wt}/{n_seed}')
    print('저장:', args.out + '.png')


if __name__ == '__main__':
    main()
