"""
★H2 (최종판) — 메시지 차원의 효과를 '통신이 기여하는 지표'로 판정:
  1) COLREGs 준수율: dim과 함께 상승 (통신 질 ∝ 채널 용량)
  2) fuel / heading: 6~8 스위트스팟
  goal%는 timeout(혼잡) 노이즈 지배라 H2 판정 지표에서 제외 (부록행).
데이터: 각 dim 최대 3-seed (metrics_v2.txt).
출력: Figures/00_MAIN/h2_dimension.png(+pdf)
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

DIMS = {2:  ['DIM2', 'DIM2_s43', 'DIM2_s44'],
        4:  ['DIM4', 'DIM4_s43', 'DIM4_s44'],
        6:  ['COMM_s42', 'COMM_s43', 'COMM_s44'],
        8:  ['DIM8', 'DIM8_s43', 'DIM8_s44'],
        10: ['DIM10', 'DIM10_s43', 'DIM10_s44'],
        12: ['DIM12', 'DIM12_s43', 'DIM12_s44']}
SPECS = [('COLREGs compliance (%)', 'colregs', True),
         ('Fuel consumption', 'fuel', False),
         ('Heading travel (deg)', 'head', False)]


def load():
    d = {}
    for ln in open(os.path.join(SCR, 'metrics_v2.txt'), encoding='utf-8', errors='ignore'):
        m = PAT.search(ln)
        if m:
            d[m.group(1)] = dict(goal=float(m.group(2)),
                                 coll=float(m.group(3)) + float(m.group(4)),
                                 fuel=float(m.group(5)), head=float(m.group(6)),
                                 colregs=float(m.group(9)))
    return d


def main():
    d = load()
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
    for ax, (title, f, hb) in zip(axes, SPECS):
        xs, ms, ss, ns = [], [], [], []
        for k in sorted(DIMS):
            vals = [d[l][f] for l in DIMS[k] if l in d]
            if not vals:
                continue
            xs.append(k); ms.append(np.mean(vals)); ss.append(_sd(vals)); ns.append(len(vals))
            # per-seed 점 (정직한 분산 노출)
            ax.plot([k] * len(vals), vals, 'o', ms=2.8, mfc='white', mec='#555555',
                    mew=0.6, zorder=4)
        ax.errorbar(xs, ms, yerr=ss, color='#2a8f3a', lw=1.6, marker='o', ms=5,
                    capsize=3, elinewidth=0.9, zorder=3)
        for x, m, n in zip(xs, ms, ns):
            ax.annotate(f'{m:.1f}' if f == 'colregs' else f'{m:.0f}', xy=(x, m),
                        xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
        ax.set_xticks(sorted(DIMS))
        ax.set_xlabel('Message dimension (MSG_DIM)')
        ax.set_title(title, fontsize=10)
        ax.text(0.97, 0.05 if hb else 0.96, '↑ better' if hb else '↓ better',
                transform=ax.transAxes, ha='right', va='bottom' if hb else 'top',
                fontsize=8.5, color='#555555')
    fig.suptitle('H2: Communication-quality metrics vs message dimension'
                 '  (16 vessels, frozen eval; dots = seeds)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'h2_dimension.{e}'), bbox_inches='tight')
    plt.close(fig)
    ninfo = {k: sum(1 for l in DIMS[k] if l in d) for k in sorted(DIMS)}
    print('saved 00_MAIN/h2_dimension  | n per dim:', ninfo)


if __name__ == '__main__':
    main()
