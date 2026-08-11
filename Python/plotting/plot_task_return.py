"""
★Task Return 학습곡선 (논문 스타일) — x축 step, y축 '종료보상 기준 return'.
  task_return = 1.50*goal% - 3.00*collision% - 0.50*timeout%  (dense shaping 제외)
근거: per-decision shaped reward는 실제 성능과 무상관(r=+0.02) → 종료보상만이 과제 성공과 정합.
출력: Figures/00_MAIN/task_return_{basis,dimension}.png(+pdf)
"""
import os, re
import numpy as np


def _sd(v):
    # sample stdev; returns 0 when only one run (avoids NaN)
    v = np.asarray(v, dtype=float)
    return float(np.std(v, ddof=1)) if v.size > 1 else 0.0
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import paper_style as ps

FIG = os.path.dirname(os.path.abspath(__file__))
# 데이터 경로: 프로젝트 내 _data(고정 사본)를 우선 사용 → 임시폴더가 지워져도 재생성 가능
_LOCAL = os.path.join(FIG, '_data')
SCR = os.environ.get('VESSEL_LOG_DIR', _LOCAL if os.path.isdir(_LOCAL) else
    os.path.join(FIG, 'logs'))
OUT = os.path.join(FIG, '00_MAIN'); os.makedirs(OUT, exist_ok=True)
ps.apply()

LINE = re.compile(r'dec=([\d.]+)M \| ep=(\d+) len~\d+ \| goal=([\d.]+)% vColl=([\d.]+)%'
                  r' oColl=([\d.]+)% TO=([\d.]+)%')
EVPAT = re.compile(r'\[(\w+)\].*?goal=\s*([\d.]+)%.*?vColl=\s*([\d.]+)%.*?oColl=\s*([\d.]+)%'
                   r'.*?TO=\s*([\d.]+)%')


def eval_tr(prefixes, cw):
    """frozen eval(metrics_v2)에서 라벨 prefix_s42/43/44의 return 평균±std."""
    d = {}
    p = os.path.join(SCR, 'metrics_v2.txt')
    if os.path.exists(p):
        for ln in open(p, encoding='utf-8', errors='ignore'):
            m = EVPAT.search(ln)
            if m:
                g, v, o, t = (float(m.group(i)) for i in (2, 3, 4, 5))
                d[m.group(1)] = 1.5 * g - cw * (v + o) - 0.5 * t
    vals = [d[f'{prefixes}_{s}'] for s in ('s42', 's43', 's44') if f'{prefixes}_{s}' in d]
    if not vals:
        return None
    return float(np.mean(vals)), float(_sd(vals))
MIN_EP = 40      # 표본 적은 창 제외(파도 aliasing 방지)
EMA_A = 0.10     # 지터 보존 스무딩


def load(fn, cw=3.0):
    """로그 → (step[M], raw return). cw=충돌 가중(안전가중 곡선은 6.0)."""
    p = os.path.join(SCR, fn)
    if not os.path.exists(p):
        return None
    xs, ys = [], []
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = LINE.search(ln)
        if not m:
            continue
        dec, ep = float(m.group(1)), int(m.group(2))
        g, v, o, t = (float(m.group(i)) for i in (3, 4, 5, 6))
        if ep < MIN_EP:
            continue
        xs.append(dec * 1e6); ys.append(1.50 * g - cw * (v + o) - 0.50 * t)  # x=raw step
    if len(ys) < 5:
        return None
    return np.array(xs), np.array(ys)


ROLL_K = 6   # trailing rolling mean 창(로그포인트 6개 ≈ 2M steps) — EMA init-lag 없음


def basis(off_files=('base_off.log', 'base_off_s43.log', 'base_off_s44.log'),
          on_files=('base_comm.log', 'base_comm_s43.log', 'base_comm_s44.log'),
          outname='task_return_basis',
          subtitle='Baseline (16 vessels, MoE-5x, COLREGs 1.5×, msg dim 6)', cw=3.0,
          ev_prefixes=None):
    off = [r for r in (load(f, cw) for f in off_files) if r]
    on = [r for r in (load(f, cw) for f in on_files) if r]
    if not off or not on:
        print(f'  {outname}: 데이터 부족'); return
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(11.2, 4.3),
                                  gridspec_kw={'width_ratios': [1.25, 1]})
    # (a) 절대 곡선 — rolling mean (실제 최종값 그대로)
    finals = []
    curves = []
    for runs, c, lab in ((off, '#555555', f'No Communication (n={len(off)})'),
                         (on, '#2a8f3a', f'Communication ON @9M (n={len(on)})')):
        x = runs[0][0]
        xs, m, _ = ps.seed_band(ax, x, [r[1] for r in runs], c, label=lab,
                                smooth=('roll', ROLL_K))
        finals.append((x[len(m) - 1], m[-1], c))
        curves.append((xs, m))
    # ★통신 이득 영역 음영 + 수렴 가속 주석 (comm 평균이 OFF 평균 위인 구간)
    La = min(len(curves[0][1]), len(curves[1][1]))
    xc, mo, mc = curves[0][0][:La], curves[0][1][:La], curves[1][1][:La]
    ax.fill_between(xc, mo, mc, where=(mc > mo), color='#2a8f3a', alpha=0.13,
                    lw=0, zorder=1, interpolate=True)
    co = np.argmax((xc > 9e6) & (mo > 0)) if ((xc > 9e6) & (mo > 0)).any() else 0
    cc = np.argmax((xc > 9e6) & (mc > 0)) if ((xc > 9e6) & (mc > 0)).any() else 0
    if co and cc and xc[co] > xc[cc]:
        ax.annotate('', xy=(xc[co], 0), xytext=(xc[cc], 0),
                    arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.1))
        ax.annotate(f'{(xc[co] - xc[cc]) / 1e6:.1f}M steps faster', fontsize=8.5,
                    xy=((xc[co] + xc[cc]) / 2, 0), xytext=(0, 7),
                    textcoords='offset points', ha='center', color='#333333')
    for i, (fx, fy, c) in enumerate(sorted(finals, key=lambda z: z[1])):
        ax.annotate(f'{fy:.0f}', xy=(fx, fy), xytext=(fx + 2e5, fy), color=c, fontsize=9,
                    va='top' if i == 0 else 'bottom')
    ax.axvline(9e6, color='#999999', ls='--', lw=0.9, zorder=1)
    lo, hi = ax.get_ylim()
    ax.text(9.15e6, lo + (hi - lo) * 0.04, 'communication ON', color='#888888', fontsize=8.5)
    ax.set_xlabel('Training steps')
    ax.set_ylabel(f'Task return  (+1.5 goal, −{cw:.0f} coll, −0.5 TO, %)')
    ax.set_title('(a) Task return')
    ax.legend(loc='upper left', fontsize=8.5)
    ax.set_xlim(0, 17.9e6)
    # (b) ★같은 seed 짝 차이 Δ = comm − no-comm: 9M 이전엔 동일설정(Δ≈0) → 이후 통신효과만 분리
    L = min(min(len(r[0]) for r in off), min(len(r[0]) for r in on))
    x = off[0][0][:L]
    deltas = []
    for o, c in zip(off, on):
        d = ps.roll(c[1][:L], ROLL_K) - ps.roll(o[1][:L], ROLL_K)
        deltas.append(d)
        axd.plot(x, d, color='#2a8f3a', lw=0.7, alpha=0.30, zorder=2)
    dm = np.stack(deltas).mean(0)
    axd.plot(x, dm, color='#2a8f3a', lw=1.7, zorder=3, label='mean of seed pairs')
    axd.axhline(0, color='#333333', lw=0.8)
    axd.axvline(9e6, color='#999999', ls='--', lw=0.9)
    axd.annotate(f'{dm[-1]:+.0f}', xy=(x[-1], dm[-1]), xytext=(x[-1] + 1.5e5, dm[-1]),
                 color='#2a8f3a', fontsize=9, va='center')
    axd.set_xlabel('Training steps')
    axd.set_ylabel('Δ Task return  (comm − no comm)')
    axd.set_title('(b) Seed-paired difference')
    axd.legend(loc='upper left', fontsize=8.5)
    axd.set_xlim(0, 17.4e6)
    fig.suptitle(f'{subtitle} — 3 seeds', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{outname}.{e}'), bbox_inches='tight')
    plt.close(fig); print(f'saved 00_MAIN/{outname} (rolling + paired Δ)')


def dimension():
    dims = [(2, 'v2_DIM2.log'), (4, 'v2_DIM4.log'), (6, 'base_comm.log'),
            (8, 'v2_DIM8.log'), (10, 'v2_DIM10.log'), (12, 'v2_DIM12.log')]
    cols = cm.viridis(np.linspace(0, 0.9, len(dims)))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    finals = []
    for (d, fn), c in zip(dims, cols):
        r = load(fn)
        if r is None:
            continue
        ps.run_curve(ax, r[0], r[1], c, label=f'dim {d}', a=EMA_A)
        finals.append((d, r[0][-1], ps.ema(r[1], EMA_A)[-1], c))
    for d, x, y, c in sorted(finals, key=lambda z: -z[2]):
        ax.annotate(f'{y:.0f}', xy=(x, y), xytext=(x + 1.8e5, y), color=c, fontsize=8.5, va='center')
    ax.axvline(9e6, color='#999999', ls='--', lw=0.9)
    lo, hi = ax.get_ylim()
    ax.text(9.15e6, lo + (hi - lo) * 0.04, 'communication ON', color='#888888', fontsize=8.5)
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Task return (%)')
    ax.set_title('Task return by message dimension (seed 42)')
    ax.legend(loc='upper left', ncol=3)
    ax.set_xlim(0, 17.4e6)
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'task_return_dimension.{e}'), bbox_inches='tight')
    plt.close(fig); print('saved 00_MAIN/task_return_dimension (paper style)')


if __name__ == '__main__':
    basis(ev_prefixes=('OFF', 'COMM'))
    # ★안전가중 return (충돌 −6 = goal의 4배 벌점): 해운 실비용 구조(충돌≫지연) 반영. 표준(−3)과 병기 공개.
    dimension()
