"""
★H4 MoE 축 — 4-arm 비교 (frozen eval, metrics_v2.txt):
  Single(단일망)      : MOE_SINGLE(s42) + MOE_SINGLE_s43/s44   (364k params)
  MoE-5x(basis)       : COMM_s42/43/44 (basis = MoE5x)          (1.82M)
  MoE-iso(naive 분리) : MOE_ISO                                  (358k)
출력: Figures/00_MAIN/moe_axis.png(+pdf)
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
import paper_style as ps

FIG = os.path.dirname(os.path.abspath(__file__))
# 데이터 경로: 프로젝트 내 _data(고정 사본)를 우선 사용 → 임시폴더가 지워져도 재생성 가능
_LOCAL = os.path.join(FIG, '_data')
SCR = os.environ.get('VESSEL_LOG_DIR', _LOCAL if os.path.isdir(_LOCAL) else
    os.path.join(FIG, 'logs'))
OUT = os.path.join(FIG, '00_MAIN'); os.makedirs(OUT, exist_ok=True)
ps.apply()

PAT = re.compile(r'\[(\w+)\].*?goal=\s*([\d.]+)%.*?vColl=\s*([\d.]+)%.*?oColl=\s*([\d.]+)%'
                 r'.*?fuel=\s*([\d.]+)\s+headTravel=\s*([\d.]+)deg.*?epReward=\s*(-?[\d.]+)'
                 r'\s+colregs=\s*([\d.]+)\s+colregsOK=\s*([\d.]+)%')

ARMS = [  # (표시명, 라벨목록, 색) — naive/shared 구분은 캡션에서 설명
    ('Single\nnetwork',  ['MOE_SINGLE', 'MOE_SINGLE_s43', 'MOE_SINGLE_s44'], '#7f7f7f'),
    ('MoE\n(isolated)',  ['MOE_ISO'],                                        '#d62728'),
    ('MoE\n(proposed)',  ['MOE_SE_s42', 'MOE_SE_s43', 'MOE_SE_s44'],         '#2ca02c'),
]
SPECS = [('Task return\n(1.5·goal −3·coll −0.5·TO)', 'tr', True),
         ('Goal reached (%)', 'goal', True), ('Collision (%)', 'coll', False),
         ('COLREGs compliance (%)', 'colregs', True), ('Fuel consumption', 'fuel', False)]


def load():
    d = {}
    p = os.path.join(SCR, 'metrics_v2.txt')
    if not os.path.exists(p):
        return d
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = PAT.search(ln)
        if m:
            g = float(m.group(2)); c = float(m.group(3)) + float(m.group(4))
            to = max(0.0, 100.0 - g - c)
            d[m.group(1)] = dict(goal=g, coll=c, tr=1.5 * g - 3.0 * c - 0.5 * to,
                                 fuel=float(m.group(5)), head=float(m.group(6)),
                                 reward=float(m.group(7)), colregs=float(m.group(9)))
    return d


def main():
    d = load()
    arms = []
    for name, labels, c in ARMS:
        vals = [d[l] for l in labels if l in d]
        if vals:
            arms.append((name, vals, c))
    if len(arms) < 2:
        print('  moe_axis: 데이터 부족'); return
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, (title, f, hb) in zip(axes, SPECS):
        xs = np.arange(len(arms))
        ms = [np.mean([v[f] for v in vals]) for _, vals, _ in arms]
        ss = [_sd([v[f] for v in vals]) for _, vals, _ in arms]
        ns = [len(vals) for _, vals, _ in arms]
        cols = [c for _, _, c in arms]
        ax.bar(xs, ms, yerr=ss, color=cols, alpha=0.75, width=0.62, capsize=3,
               error_kw=dict(lw=0.9, ecolor='#333333'))
        # per-seed 점 오버레이 (논문식: 분산을 점으로 정직하게 노출)
        for x, (_, vals, c) in zip(xs, arms):
            ax.plot([x] * len(vals), [v[f] for v in vals], 'o', ms=2.6, mfc='white',
                    mec='#222222', mew=0.6, zorder=4)
        for x, m, s, n in zip(xs, ms, ss, ns):
            ax.annotate(f'{m:.1f}', xy=(x, m), xytext=(0, 8 + (6 if s else 0)),
                        textcoords='offset points', ha='center', fontsize=8.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([a[0] for a in arms], fontsize=7.5)
        ax.set_title(title, fontsize=10)
        lo = min(min(v[f] for v in vals) for _, vals, _ in arms)
        hi = max(max(v[f] for v in vals) for _, vals, _ in arms)
        rng = (hi - lo) or 1
        ax.set_ylim(lo - rng * 0.18, hi + rng * 0.3)
        ax.text(0.97, 0.96, '↑' if hb else '↓', transform=ax.transAxes,
                ha='right', va='top', fontsize=10, color='#555555')
    fig.suptitle('MoE architecture ablation (16 vessels, msg dim 6, frozen eval; dots = seeds)',
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'moe_axis.{e}'), bbox_inches='tight')
    plt.close(fig)
    print('saved 00_MAIN/moe_axis  |  arms:', ', '.join(f'{a[0].splitlines()[0]}(n={len(a[1])})' for a in arms))


if __name__ == '__main__':
    main()
