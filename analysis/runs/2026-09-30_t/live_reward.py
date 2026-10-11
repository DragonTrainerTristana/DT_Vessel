"""live_reward.py — 학습 중 reward·결과 실시간 감시 그림 (2026-09-30). 5분마다 _live_loop.sh 가 부름. 판정용 아님(끝 쪽은 이동평균이 덜 참).

읽는 것(있는 것만):
  t_ 배치   C:\\work\\DT_Vessel_v3\\Python\\_repro_out_t\\t_*.csv · t_*_ep.csv · t_*.log   (이 서버)
  Stage A   Dropbox runs\\2026-09-30_t\\other_pc\\back\\live\\s_*.csv · s_*.log            (3060 PC 가 _sync_3060.sh 로 5분마다 보냄)
            + 기준선 s_off: C:\\work\\DT_Vessel\\Python\\_repro_out_s\\s_off_s*.csv
쓰는 것:    Dropbox runs\\2026-09-30_t\\live\\reward_live.pdf · .png · status.txt
실행:       C:/Users/OSH/anaconda3/python.exe live_reward.py      (matplotlib 있는 base python)
"""
import glob
import os
import re
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
T_DIR = r'C:\work\DT_Vessel_v3\Python\_repro_out_t'
A_DIR = os.path.join(HERE, 'other_pc', 'back', 'live')
S_DIR = r'C:\work\DT_Vessel\Python\_repro_out_s'
OUT = os.path.join(HERE, 'live')
BRANCH_AT = 9_043_968
MA = 20                    # update 수(= 65,536 결정 × 20 ≈ 1.3M 결정) 뒤로 이동평균
ARM_STYLE = {'off': ('#222222', '-'), 'offb': ('#8c8c8c', '--'), 'a6': ('#1f77b4', '-'), 'a8': ('#ff7f0e', '-'),
             'a2': ('#17becf', '-'), 'a4': ('#9467bd', '-'), 'c6': ('#d62728', '-'), 'trunk': ('#bbbbbb', '-')}
SEED_COL = {43: '#e41a1c', 44: '#4daf4a', 45: '#984ea3', 46: '#ff7f00', 47: '#a65628'}   # 뿌리(분기 전) 시드 색
WIN_RE = re.compile(r"\[(\w+)\] dec=([\d.]+)M \| ep=(\d+) len~(\d+) \| goal=([\d.]+)% vColl=([\d.]+)% oColl=([\d.]+)% TO=([\d.]+)% \| "
                    r"R=(-?[\d.]+) \| (\d+) dec/s(?:.*?epRet=(-?[\d.]+|nan))?")


def read_csv(path, cols):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        head = f.readline().strip().split(',')
        idx = [head.index(c) if c in head else None for c in cols]
        for line in f:
            p = line.strip().split(',')
            if len(p) < len(head):
                continue
            try:
                rows.append([float(p[i]) if (i is not None and p[i] not in ('', 'nan')) else np.nan for i in idx])
            except ValueError:
                continue
    if not rows:
        return None
    a = np.array(rows, dtype=float)
    _, keep = np.unique(a[:, 0], return_index=True)          # 같은 step 중복(재개) → 첫 값
    return a[np.sort(keep)]


def trail_ma(y, w=MA):
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - w + 1):i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def last_window(log):
    if not os.path.exists(log):
        return None
    last, saved = None, False
    with open(log, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = WIN_RE.search(line)
            if m:
                last = m
            if 'saved ->' in line:
                saved = True
    if last is None:
        return {'saved': saved, 'age': (time.time() - os.path.getmtime(log)) / 60}
    g = last.groups()
    return {'arm': g[0], 'dec': float(g[1]), 'goal': float(g[4]), 'vColl': float(g[5]), 'oColl': float(g[6]),
            'TO': float(g[7]), 'R': float(g[8]), 'dps': int(g[9]), 'epRet': (float(g[10]) if g[10] not in (None, 'nan') else np.nan),
            'saved': saved, 'age': (time.time() - os.path.getmtime(log)) / 60}


EC_DIR = os.path.join(T_DIR, 'evalcurve')
EC_RE = re.compile(r"goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%\s+oColl=\s*([\d.]+)%\s+TO=\s*([\d.]+)%\s+\|\s*(\d+) eps.*?epReward=\s*(-?[\d.]+)")
UPDATE_DEC = 65_536
TOTAL = 16_056_320


def ec_step(base):
    """ec_<name> → 결정 수. init=0, stepNM = 그 N M 을 처음 넘은 update 경계, trunk 끝 = 분기점, 갈래 끝 = TOTAL."""
    m = re.search(r'\.step(\d+)M$', base)
    if m:
        k = int(m.group(1)) * 1_000_000
        return -(-k // UPDATE_DEC) * UPDATE_DEC
    if '_init_' in base:
        return 0
    if '_trunk_' in base:
        return BRANCH_AT
    return TOTAL


def read_evalcurve():
    """{(arm, seed): [(step, epReward, goal, vColl, oColl, eps), ...]} — arm 'trunk' 은 init·뿌리 점."""
    out = {}
    for f in glob.glob(os.path.join(EC_DIR, 'ec_t_*.txt')):
        base = os.path.basename(f)[3:-4]
        txt = open(f, encoding='utf-8', errors='replace').read()
        m = EC_RE.search(txt)
        if not m:
            continue
        mm = re.match(r't_(init_d6|trunk_d6|[a-z0-9]+)_s(4\d)', base)
        if not mm:
            continue
        arm = 'trunk' if mm.group(1) in ('init_d6', 'trunk_d6') else mm.group(1)
        seed = int(mm.group(2))
        g, v, o, to, eps, er = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5)), float(m.group(6))
        out.setdefault((arm, seed), []).append((ec_step(base), er, g, v, o, eps))
    for k in out:
        out[k].sort()
    return out


def plot_evalcurve(axR, axG):
    ec = read_evalcurve()
    if not ec:
        for ax in (axR, axG):
            ax.text(0.5, 0.5, '고정 평가 곡선 — 아직 평가 결과 없음(evalcurve 가 도는 중)', ha='center', va='center',
                    transform=ax.transAxes, color='#666666')
        return
    seeds = sorted({k[1] for k in ec})
    arms = [a for a in ('off', 'a6', 'offb', 'a8', 'a2', 'a4') if any(k[0] == a for k in ec)]
    for arm in ['trunk'] + arms:
        c, ls = ARM_STYLE.get(arm, ('#444444', '-'))
        if arm == 'trunk':
            c = '#555555'
        per = []
        for sd in seeds:
            pts = list(ec.get(('trunk', sd), [])) if arm != 'trunk' else list(ec.get(('trunk', sd), []))
            if arm != 'trunk':
                pts = [p for p in pts if p[0] == BRANCH_AT] + ec.get((arm, sd), [])   # 분기점에서 이어 그림
            if not pts:
                continue
            x = np.array([p[0] for p in pts]) / 1e6
            axR.plot(x, [p[1] for p in pts], color=c, ls=ls, lw=0.8, alpha=0.35, marker='.', ms=3)
            axG.plot(x, [p[2] for p in pts], color=c, ls='-', lw=0.8, alpha=0.35)
            axG.plot(x, [p[3] for p in pts], color=c, ls=':', lw=0.8, alpha=0.35)
            per.append({p[0]: p for p in pts})
        common = sorted(set.intersection(*[set(d) for d in per])) if per else []
        if common:
            xs = np.array(common) / 1e6
            axR.plot(xs, [np.mean([d[k][1] for d in per]) for k in common], color=c, ls=ls, lw=2.4, marker='o', ms=4,
                     label=('뿌리(통신 없음)' if arm == 'trunk' else arm) + f' 평균(시드 {len(per)})')
            axG.plot(xs, [np.mean([d[k][2] for d in per]) for k in common], color=c, ls='-', lw=2.2)
            axG.plot(xs, [np.mean([d[k][3] for d in per]) for k in common], color=c, ls=':', lw=2.2)
    for ax in (axR, axG):
        ax.axvline(BRANCH_AT / 1e6, color='k', ls=':', lw=0.8)
        ax.set_xlim(-0.3, TOTAL / 1e6 + 0.3)
        ax.grid(alpha=0.3)
        ax.set_xlabel('결정 수 (M)')
    axR.set_title('★고정 평가 학습곡선 — 에피소드 점수(같은 평가 환경·seed 999·첫 에피소드 제외). 0 = 학습 전', fontsize=10)
    axR.set_ylabel('에피소드 점수(평균)')
    axR.legend(fontsize=7, loc='lower right')
    axG.set_title('고정 평가 — 도착 %(실선) · 선박 충돌 %(점선)', fontsize=10)
    axG.set_ylabel('%')


def runs_in(d, prefix):
    out = {}
    for p in glob.glob(os.path.join(d, f'{prefix}*_s4?.csv')):
        name = os.path.basename(p)[:-4]
        if name.endswith(('_aux', '_ep', '_comm')):
            continue
        m = re.match(rf'{re.escape(prefix)}(.+)_s(4\d)$', name)
        if m:
            arm = 'trunk' if m.group(1).startswith('trunk') else m.group(1)
            out[name] = (arm, int(m.group(2)))
    return out


def plot_curves(ax, d, runs, col, title, ylab, csv_suffix='', ycols=('raw_reward',), ma=True, branch_only_for_arms=True):
    means = {}
    for name, (arm, seed) in sorted(runs.items()):
        a = read_csv(os.path.join(d, name + csv_suffix + '.csv'), ('step',) + ycols)
        if a is None or len(a) == 0:
            continue
        x, y = a[:, 0], a[:, 1]
        if arm != 'trunk' and branch_only_for_arms:
            sel = x > BRANCH_AT
            x, y = x[sel], y[sel]
            if len(x) == 0:
                continue
        yy = trail_ma(y) if ma else y
        c, ls = ARM_STYLE.get(arm, ('#444444', '-'))
        if arm == 'trunk':                                  # 뿌리는 시드별 색(분기 전엔 시드 구분이 중요)
            c = SEED_COL.get(seed, '#999999')
        ax.plot(x / 1e6, yy, color=c, ls=ls, lw=1.1 if arm == 'trunk' else 1.3, alpha=0.6 if arm == 'trunk' else 0.9,
                label=(f'뿌리 s{seed}' if arm == 'trunk' else None))
        if arm != 'trunk':
            means.setdefault(arm, []).append((x, yy))
    for arm in means:                                    # 범례용 한 줄씩
        c, ls = ARM_STYLE.get(arm, ('#444444', '-'))
        ax.plot([], [], color=c, ls=ls, lw=2, label=f"{arm} (시드 {len(means[arm])})")
    ax.axvline(BRANCH_AT / 1e6, color='k', ls=':', lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('결정 수 (M)')
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc='best', ncol=2)


def main():
    os.makedirs(OUT, exist_ok=True)
    now = time.strftime('%Y-%m-%d %H:%M')
    t_runs = runs_in(T_DIR, 't_')
    a_runs = runs_in(A_DIR, 's_') if os.path.isdir(A_DIR) else {}
    s_ref = {k: v for k, v in runs_in(S_DIR, 's_').items() if v[0] == 'off'}

    fig = plt.figure(figsize=(16.5, 16.0))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1, 1, 0.75], hspace=0.42, wspace=0.18)
    axE, axEG = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax1, ax2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax3, ax4 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    axT = fig.add_subplot(gs[3, :]); axT.axis('off')
    plot_evalcurve(axE, axEG)

    plot_curves(ax1, T_DIR, t_runs, None, f'(학습 로그) 결정당 reward (뒤로 {MA} update 이동평균)', '결정당 reward')
    plot_curves(ax2, T_DIR, t_runs, None, f'(학습 로그) 에피소드 반환 (뒤로 {MA} update 이동평균, 시계 잘린 첫 에피소드 포함)', '에피소드 반환',
                csv_suffix='_ep', ycols=('ep_return_mean',))
    # goal / vColl (ep.csv) — 팔별 선, 실선 goal · 점선 vColl
    for name, (arm, seed) in sorted(t_runs.items()):
        a = read_csv(os.path.join(T_DIR, name + '_ep.csv'), ('step', 'goal', 'vColl'))
        if a is None:
            continue
        x = a[:, 0]
        sel = (x > BRANCH_AT) if arm != 'trunk' else np.ones_like(x, dtype=bool)
        c, _ = ARM_STYLE.get(arm, ('#444444', '-'))
        if arm == 'trunk':
            c = SEED_COL.get(seed, '#999999')
        ax3.plot(x[sel] / 1e6, trail_ma(a[sel, 1]), color=c, lw=1.1, alpha=0.5 if arm == 'trunk' else 0.9)
        ax3.plot(x[sel] / 1e6, trail_ma(a[sel, 2]), color=c, lw=1.1, ls=':', alpha=0.5 if arm == 'trunk' else 0.9)
    ax3.plot([], [], color='k', lw=1.2, label='도착 %'); ax3.plot([], [], color='k', lw=1.2, ls=':', label='선박 충돌 %')
    ax3.axvline(BRANCH_AT / 1e6, color='k', ls=':', lw=0.8)
    ax3.set_title(f't_ 배치 — 도착 %(실선) · 선박 충돌 %(점선), 색 = 팔', fontsize=10)
    ax3.set_xlabel('결정 수 (M)'); ax3.set_ylabel('%'); ax3.grid(alpha=0.3); ax3.legend(fontsize=7)
    # Stage A (3060) — s_off 기준선(서버 s_ 배치) + s_a6·s_c6
    stage = dict(s_ref); stage.update({k: v for k, v in a_runs.items() if v[0] in ('a6', 'c6')})
    if a_runs:
        for name, (arm, seed) in sorted(stage.items()):
            d = S_DIR if arm == 'off' else A_DIR
            a = read_csv(os.path.join(d, name + '.csv'), ('step', 'raw_reward'))
            if a is None:
                continue
            x, y = a[:, 0], a[:, 1]
            sel = x > BRANCH_AT
            c, ls = ARM_STYLE.get(arm, ('#444444', '-'))
            ax4.plot(x[sel] / 1e6, trail_ma(y[sel]), color=c, ls=ls, lw=1.2)
        for arm in ('off', 'a6', 'c6'):
            c, ls = ARM_STYLE[arm]; ax4.plot([], [], color=c, ls=ls, lw=2, label=f's_{arm}')
        ax4.legend(fontsize=7)
    else:
        ax4.text(0.5, 0.5, 'Stage A(3060) 데이터 아직 없음\n(3060 에서 _sync_3060.sh 를 켜면 5분마다 들어옴)', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=10, color='#666666')
    ax4.set_title('Stage A (3060 PC) — 결정당 reward, 분기 뒤', fontsize=10)
    ax4.set_xlabel('결정 수 (M)'); ax4.grid(alpha=0.3)

    # 상태표
    rows = []
    for d, rr in ((T_DIR, t_runs), (A_DIR, a_runs)):
        for name, (arm, seed) in sorted(rr.items()):
            w = last_window(os.path.join(d, name + '.log'))
            if not w:
                continue
            flag = '끝' if w.get('saved') else ('!정지?' if w['age'] >= 15 else '진행')
            if 'dec' in w and w['dec'] >= 7.5 and w['oColl'] >= 20:
                flag += ' !벽'
            if 'dec' in w:
                rows.append(f"{name:18s} {w['dec']:6.2f}M  goal {w['goal']:5.1f}  vColl {w['vColl']:5.1f}  oColl {w['oColl']:5.1f}  TO {w['TO']:5.1f}  "
                            f"R {w['R']:7.3f}  epRet {w['epRet']:8.1f}  {w['dps']:5d} dec/s  로그 {w['age']:4.0f}분 전  {flag}")
            else:
                rows.append(f"{name:18s} (창 줄 아직 없음)  로그 {w['age']:4.0f}분 전  {flag}")
    txt = '\n'.join(rows) if rows else '(로그 없음)'
    axT.text(0.0, 1.0, '최근 창(5 update) 상태\n' + txt, family=['Malgun Gothic'], fontsize=7.2, va='top', ha='left', transform=axT.transAxes)
    tl = os.path.join(T_DIR, '_timeline.txt')
    last_tl = open(tl, encoding='utf-8', errors='replace').read().strip().splitlines()[-1] if os.path.exists(tl) else ''
    fig.suptitle(f'실시간 학습 감시 — {now}   (판정용 아님 · 끝 쪽 이동평균은 덜 참)\n드라이버: {last_tl[:150]}', fontsize=11)
    fig.savefig(os.path.join(OUT, 'reward_live.png'), dpi=110, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'reward_live.pdf'), bbox_inches='tight')
    plt.close(fig)
    with open(os.path.join(OUT, 'status.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{now}\n{last_tl}\n\n{txt}\n')
    print(now, len(t_runs), 't_ runs,', len(a_runs), 'stageA runs')


if __name__ == '__main__':
    main()
