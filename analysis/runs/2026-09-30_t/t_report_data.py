"""t_report_data.py — t_ 배치 보고서용 표·그림(읽기만). base anaconda python(matplotlib) 으로 실행.

  python t_report_data.py --out <dir>
만드는 것:
  goal2_train.md  — 목표 2 사전등록 판정(학습 곡선): 분기 뒤 통신 이동평균 >= off 인 update 가 90 % 이상
                    그리고 끝 차이 > |offb-off| 끝 차이 (시드별, 결정당 reward·에피소드 반환 두 곡선)
  evalcurve.md/png — 고정 평가 곡선(저자 선택 주 그림, 사후 추가): 점별 평균 에피소드 점수, 시드별 off 대비 승수(기술 통계, 판정 아님)
  oldckpt.md      — 옛 체크포인트(s_off·x_off·h_a6) v2 판정 재채점 값
숫자는 파일에 있는 값만 씀. 없으면 '없음'.
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_parse as rp  # noqa: E402

O = r'C:\work\DT_Vessel_v3\Python\_repro_out_t'
BR, TOTAL, UPD = 9043968, 16056320, 65536
SEEDS = (43, 46, 47)
ARMS = ('a6', 'a8', 'a2', 'a4')
COL = {'trunk': '#777777', 'off': '#111111', 'offb': '#888888', 'a6': '#1f77b4', 'a8': '#d62728', 'a2': '#2ca02c', 'a4': '#ff7f0e'}
LS = {'offb': '--'}


def up(m):
    """m M 결정을 처음 넘은 update 경계."""
    return -(-int(m * 1_000_000) // UPD) * UPD


def load_curve(fn, col):
    xs, ys = [], []
    with open(fn, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            try:
                xs.append(int(r['step']))
                ys.append(float(r[col]))
            except (KeyError, ValueError):
                pass
    return np.array(xs), np.array(ys)


def cma(y, w=20):
    h = w // 2
    return np.array([y[max(0, i - h):i + h].mean() for i in range(len(y))])


def goal2_train():
    lines = ['## 목표 2 — 학습 곡선 사전등록 판정 (분기 뒤, 20 update 중심 이동평균)', '',
             '통과 = 분기 뒤 통신이 off 이상인 update 비율 90 % 이상 **그리고** 끝 차이(통신 − off) > |offb − off| 끝 차이. '
             '시드 3개 모두, 두 곡선 모두 통과해야 목표 2 달성.', '',
             '| 팔 | 곡선 | s43: 위 비율 · 끝 차이 (잡음) | s46 | s47 | 통과 시드 |', '|---|---|---|---|---|---|']
    for arm in ARMS:
        for kind, suf, col in (('결정당 reward', '.csv', 'raw_reward'), ('에피소드 반환', '_ep.csv', 'ep_return_mean')):
            cells, npass = [], 0
            for s in SEEDS:
                try:
                    d = {}
                    for a in ('off', 'offb', arm):
                        x, y = load_curve(os.path.join(O, f't_{a}_s{s}{suf}'), col)
                        m = cma(y)
                        d[a] = m[x > BR]
                    n = min(len(v) for v in d.values())
                    mo, ma, mb = d['off'][:n], d[arm][:n], d['offb'][:n]
                    frac = 100 * (ma >= mo).mean()
                    endd = ma[-1] - mo[-1]
                    nz = abs(mb[-1] - mo[-1])
                    ok = bool(frac >= 90 and endd > nz)
                    npass += ok
                    if kind == '결정당 reward':
                        txt = f'{frac:.0f}% · {endd:+.3f} ({nz:.3f})'
                    else:
                        txt = f'{frac:.0f}% · {endd:+.0f} ({nz:.0f})'
                    cells.append(txt + (' **통과**' if ok else ''))
                except FileNotFoundError:
                    cells.append('없음')
            lines.append(f'| t_{arm} | {kind} | ' + ' | '.join(cells) + f' | {npass}/3 |')
    return '\n'.join(lines) + '\n'


def ec_step(base):
    m = re.search(r'\.step(\d+)M$', base)
    if m:
        return up(int(m.group(1)))
    if '_init_' in base:
        return 0
    if '_trunk_' in base:
        return BR
    return TOTAL


def read_ec():
    out = {}
    for f in glob.glob(os.path.join(O, 'evalcurve', 'ec_t_*.txt')):
        base = os.path.basename(f)[3:-4]
        d = rp.parse_eval(f)
        if d.get('epReward') is None or d.get('goal') is None:
            continue
        mm = re.match(r't_(init_d6|trunk_d6|[a-z0-9]+)_s(4\d)', base)
        if not mm:
            continue
        arm = 'trunk' if mm.group(1) in ('init_d6', 'trunk_d6') else mm.group(1)
        out.setdefault((arm, int(mm.group(2))), {})[ec_step(base)] = d
    return out


def evalcurve(outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    ec = read_ec()
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    keys = (('epReward', '에피소드 점수(평균)'), ('goal', '도착 %'), ('vColl', '선박 충돌 %'))
    for arm in ('trunk', 'off', 'offb', 'a6', 'a8', 'a2', 'a4'):
        per = []
        for s in SEEDS:
            tr = ec.get(('trunk', s), {})
            if arm == 'trunk':
                pts = dict(tr)
            else:
                pts = {BR: tr[BR]} if BR in tr else {}
                pts.update(ec.get((arm, s), {}))
                if len(pts) <= 1:
                    continue
            if not pts:
                continue
            per.append(pts)
            xs = sorted(pts)
            for ax, (k, _) in zip(axs, keys):
                ax.plot(np.array(xs) / 1e6, [pts[x][k] for x in xs], color=COL[arm], ls=LS.get(arm, '-'), lw=0.7, alpha=0.3)
        if len(per) < 3:
            continue
        common = sorted(set.intersection(*[set(p) for p in per]))
        if not common:
            continue
        for ax, (k, _) in zip(axs, keys):
            ax.plot(np.array(common) / 1e6, [np.mean([p[x][k] for p in per]) for x in common], color=COL[arm],
                    ls=LS.get(arm, '-'), lw=2.4, marker='o', ms=4,
                    label=('뿌리(통신 없음)' if arm == 'trunk' else arm) + ' 평균(시드 3)')
    for ax, (k, lab) in zip(axs, keys):
        ax.axvline(BR / 1e6, color='k', ls=':', lw=0.8)
        ax.set_xlabel('결정 수 (M)')
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axs[0].set_title('고정 평가 학습곡선 — 같은 평가 환경(seed 999, 첫 에피소드 제외). 0 = 학습 전', fontsize=10)
    axs[0].legend(fontsize=8, loc='lower right')
    axs[1].set_title('고정 평가 — 도착 %', fontsize=10)
    axs[2].set_title('고정 평가 — 선박 충돌 %', fontsize=10)
    fig.tight_layout()
    png = os.path.join(outdir, 'evalcurve.png')
    fig.savefig(png, dpi=130)
    plt.close(fig)
    cols = [(up(m), f'{m}M') for m in (10, 12, 14, 16)] + [(TOTAL, '끝')]
    lines = ['## 고정 평가 학습곡선 (Claude 권장안·저자 사후 승인 — 사후 추가, 기술 통계만, 창 2000 편향 값)', '',
             '점마다 3 시드 평균 에피소드 점수. 괄호 = 같은 점에서 off 보다 높은 시드 수. 뿌리 끝(9.04M)은 모든 팔 공통.', '',
             '| 팔 | ' + ' | '.join(c[1] for c in cols) + ' |', '|---|' + '---|' * len(cols)]
    for arm in ('off', 'offb', 'a6', 'a8', 'a2', 'a4'):
        row = []
        for st, _ in cols:
            vals = [ec.get((arm, s), {}).get(st, {}).get('epReward') for s in SEEDS]
            offs = [ec.get(('off', s), {}).get(st, {}).get('epReward') for s in SEEDS]
            if any(v is None for v in vals):
                row.append('없음')
                continue
            mv = float(np.mean(vals))
            if arm == 'off':
                row.append(f'{mv:.0f}')
            else:
                w = sum(1 for v, o in zip(vals, offs) if o is not None and v > o)
                row.append(f'{mv:.0f} ({w}/3)')
        lines.append(f'| t_{arm} | ' + ' | '.join(row) + ' |')
    tr = [ec.get(('trunk', s), {}) for s in SEEDS]
    tline = []
    for st, lab in ((0, '학습 전'), (up(2), '2M'), (up(4), '4M'), (up(6), '6M'), (up(8), '8M'), (BR, '뿌리 끝')):
        v = [t.get(st, {}).get('epReward') for t in tr]
        tline.append(f'{lab} {np.mean(v):.0f}' if all(x is not None for x in v) else f'{lab} 없음')
    lines += ['', '뿌리(통신 없음, 3 시드 평균 에피소드 점수): ' + ' · '.join(tline)]
    with open(os.path.join(outdir, 'evalcurve.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return png


def oldckpt():
    keys = ('goal', 'vColl', 'oColl', 'TO', 'dcpa', 'C', 'rks_end', 'rks_v2', 'fuel', 'head')
    lines = ['## 옛 체크포인트 재채점 (보상·뿌리가 달라 짝 비교 아님 — 참고)', '',
             '| 런 | ' + ' | '.join(keys) + ' |', '|---|' + '---|' * len(keys)]
    for pre, arm in (('s_', 'off'), ('x_', 'off'), ('h_', 'a6')):
        runs = rp.load_runs(O, pre, arm, (43, 44, 45))
        for s in (43, 44, 45):
            d = runs.get(s)
            if not d:
                lines.append(f'| {pre}{arm}_s{s} | ' + ' | '.join('없음' for _ in keys) + ' |')
                continue
            lines.append(f'| {pre}{arm}_s{s} | ' + ' | '.join('없음' if d.get(k) is None else f'{d[k]:.1f}' for k in keys) + ' |')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'goal2_train.md'), 'w', encoding='utf-8') as fh:
        fh.write(goal2_train())
    with open(os.path.join(a.out, 'oldckpt.md'), 'w', encoding='utf-8') as fh:
        fh.write(oldckpt())
    print('png', evalcurve(a.out))
    print('ok')
