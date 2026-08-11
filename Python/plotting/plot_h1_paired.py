"""
★H1 헤드라인 — seed-paired slope 차트 (OFF→ON, 같은 seed끼리 연결).
훈련 reward 곡선은 과제성능과 무상관(r≈0.02)이라 H1 근거로 부적합 → ground-truth 지표의
시드쌍 비교가 정직한 헤드라인: 연료·조타는 3/3 시드 일관 개선.
출력: Figures/00_MAIN/h1_paired.png(+pdf)
"""
import os, re
import numpy as np
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
SEEDS = ['s42', 's43', 's44']
SEED_C = {'s42': '#4c72b0', 's43': '#dd8452', 's44': '#55a868'}
SPECS = [('Fuel consumption', 'fuel', False, '{:.0f}'),
         ('Heading travel (deg)', 'head', False, '{:.0f}'),
         ('COLREGs compliance (%)', 'colregs', True, '{:.1f}'),
         ('Goal reached (%)', 'goal', True, '{:.1f}'),
         ('Collision (%)', 'coll', False, '{:.1f}'),
         ('Task return', 'tr', True, '{:.1f}')]


def load():
    d = {}
    p = os.path.join(SCR, 'metrics_v2.txt')
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = PAT.search(ln)
        if m:
            g = float(m.group(2)); c = float(m.group(3)) + float(m.group(4))
            to = max(0.0, 100.0 - g - c)
            d[m.group(1)] = dict(goal=g, coll=c, tr=1.5 * g - 3.0 * c - 0.5 * to,
                                 fuel=float(m.group(5)), head=float(m.group(6)),
                                 colregs=float(m.group(9)))
    return d


def main(off_pre='OFF', on_pre='COMM', outname='h1_paired', subtitle='baseline (MoE-5x)'):
    d = load()
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2))
    for ax, (title, f, hb, fmt) in zip(axes.ravel(), SPECS):
        for s in SEEDS:
            o, c = d.get(f'{off_pre}_{s}'), d.get(f'{on_pre}_{s}')
            if not o or not c:
                continue
            better = (c[f] > o[f]) if hb else (c[f] < o[f])
            ax.plot([0, 1], [o[f], c[f]], '-', color=SEED_C[s], lw=1.2,
                    alpha=0.9 if better else 0.45, zorder=3)
            ax.plot([0, 1], [o[f], c[f]], 'o', color=SEED_C[s], ms=4, zorder=4,
                    label=f'seed {s[1:]}' if f == 'fuel' else None)
        mo = np.mean([d[f'{off_pre}_{s}'][f] for s in SEEDS])
        mc = np.mean([d[f'{on_pre}_{s}'][f] for s in SEEDS])
        ax.plot([0, 1], [mo, mc], '-', color='#222222', lw=2.0, zorder=5)
        ax.plot([0, 1], [mo, mc], 's', color='#222222', ms=5, zorder=6)
        pct = 100.0 * (mc - mo) / abs(mo) if mo else 0.0
        ax.set_title(f'{title}\nmean {fmt.format(mo)} → {fmt.format(mc)}  ({pct:+.1f}%)',
                     fontsize=9.5)
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['no comm', 'comm'])
        ax.text(0.97, 0.96, '↑' if hb else '↓', transform=ax.transAxes,
                ha='right', va='top', fontsize=10, color='#555555')
    axes[0, 0].legend(loc='lower left', fontsize=8)
    fig.suptitle(f'Communication effect, seed-paired — {subtitle}  (16 vessels; black = mean)',
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{outname}.{e}'), bbox_inches='tight')
    plt.close(fig)
    print(f'saved 00_MAIN/{outname}')


if __name__ == '__main__':
    main('SE_OFF', 'MOE_SE', 'h1_paired', 'proposed architecture')
