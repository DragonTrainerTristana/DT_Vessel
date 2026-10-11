# f2_ 학습 곡선(학습 중 기록): 결정당 보상(csv raw_reward)과 에피소드 수익(_ep.csv ep_return_mean, n_ep 가중). 로그에서 읽기만 함.
import csv, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')
P = 'C:/work/DT_Vessel_v3/Python/_repro_out_f2/'
out = sys.argv[1]
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 8.5, 'axes.grid': True, 'grid.color': '#e4e3df',
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})
BR = 9.043968
COL = {'off': '#d9822b', 'comm': '#1b6fd6', 'offb': '#9c9a94'}


def rd(name, ep):
    fp = P + name + ('_ep.csv' if ep else '.csv')
    if not os.path.exists(fp):
        return []
    rows = []
    for r in csv.DictReader(open(fp, encoding='utf-8')):
        try:
            if ep:
                rows.append((float(r['step']) / 1e6, float(r['ep_return_mean']), float(r['n_ep']), float(r['goal']), float(r['vColl'])))
            else:
                rows.append((float(r['step']) / 1e6, float(r['raw_reward']), 1.0))
        except (ValueError, KeyError):
            pass
    return rows


def smooth(rows, w=8):
    xs, ys = [], []
    for i in range(len(rows)):
        seg = rows[max(0, i - w + 1): i + 1]
        den = sum(r[2] for r in seg)
        if den > 0:
            xs.append(rows[i][0]); ys.append(sum(r[1] * r[2] for r in seg) / den)
    return xs, ys


def seg_mean(rows, a, b):
    seg = [r for r in rows if a < r[0] <= b]
    den = sum(r[2] for r in seg)
    return sum(r[1] * r[2] for r in seg) / den if den > 0 else float('nan')


SEEDS = (43, 44, 45)
fig, axs = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
for j, s in enumerate(SEEDS):
    for i, ep in enumerate((False, True)):
        ax = axs[i][j]
        for arm in ('offb', 'off', 'comm'):
            rows = rd(f'f2_{arm}_s{s}', ep)
            if rows:
                xs, ys = smooth(rows)
                ax.plot(xs, ys, color=COL[arm], lw=1.4 if arm != 'offb' else 1.0, label=arm)
        ax.axvline(BR, color='#52514e', lw=0.8, ls=':')
        ax.axvspan(0, 2.3, color='#f3f3f3', zorder=0)
        if i == 0:
            ax.set_title(f'시드 {s}')
        if j == 0:
            ax.set_ylabel('결정당 보상 (8 update 이동평균)' if i == 0 else '에피소드 수익 (끝난 ep 가중 이동평균)')
        if i == 1:
            ax.set_xlabel('결정 수 (M)')
axs[0][0].legend(frameon=False, fontsize=8, loc='lower right')
fig.suptitle('f2_ 학습 곡선 — 0–9.04M = 공통 trunk(세 팔이 같은 줄), 점선 = 분기점. 회색 띠(0–2.3M) = 도착 ep 이 아직 안 끝나 낮게 잡히는 시작 구간', fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out)
print('saved', out)
SEGS = [(0, 1.0), (1.0, 2.3), (2.3, 4.5), (4.5, 6.8), (6.8, 9.044), (9.044, 11.4), (11.4, 13.7), (13.7, 16.1)]
for ep, nm in ((True, '에피소드 수익'), (False, '결정당 보상')):
    print(f'\n== {nm} 구간 평균')
    print('run          ' + ''.join(f'{a:>5.1f}–{b:<5.1f}' for a, b in SEGS))
    for s in SEEDS:
        for arm in ('off', 'comm', 'offb'):
            rows = rd(f'f2_{arm}_s{s}', ep)
            if rows:
                print(f'{arm}_s{s:<8d}' + ''.join(f'{seg_mean(rows, a, b):>10.1f} ' if ep else f'{seg_mean(rows, a, b):>10.3f} ' for a, b in SEGS))
