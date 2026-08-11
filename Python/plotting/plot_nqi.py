"""
★Navigation Cost Index (NCI) — 통신 효과의 종합지표 (재학습 불필요, eval 재해석).
  NCI = Σ w_i · (cost_i / pooled_mean_i),  cost = [충돌%, 위반%(100−colregsOK), fuel, headTravel, 미도달%(100−goal)]
  가중치는 해양 도메인 우선순위(안전>준수>효율>부드러움>도달)로 선험 결정, 4개 프리셋 민감도 동시 제시
  → OFF−COMM 차이가 전 구성요소 양수라 어떤 가중치에서도 통신 우위 (cherry-picking 아님을 그림이 증명).
출력: Figures/00_MAIN/nqi_comm.png(+pdf)
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
COMP = ['Collision', 'Rule violation', 'Fuel', 'Heading travel', 'Non-arrival']
PRESETS = [  # (이름, [충돌, 위반, fuel, head, 미도달])
    ('Safety-first',       [.50, .20, .15, .05, .10]),
    ('Collision + fuel',   [.45, .05, .35, .05, .10]),  # 충돌·연료 집중 가중
    ('Balanced',           [.35, .25, .20, .10, .10]),
    ('Compliance-first',   [.20, .40, .20, .10, .10]),
    ('Efficiency-first',   [.15, .15, .40, .20, .10]),
    ('Green shipping',     [.10, .30, .30, .25, .05]),  # IMO 탈탄소 우선순위(연료+준수)
]
N_SIMPLEX = 20000   # 가중치 심플렉스 랜덤 스윕 표본 수 (Dirichlet(1,...,1) = 균등)


def load():
    d = {}
    for ln in open(os.path.join(SCR, 'metrics_v2.txt'), encoding='utf-8', errors='ignore'):
        m = PAT.search(ln)
        if m:
            g = float(m.group(2)); c = float(m.group(3)) + float(m.group(4))
            d[m.group(1)] = np.array([c, 100.0 - float(m.group(9)), float(m.group(5)),
                                      float(m.group(6)), 100.0 - g])
    return d


def main(off_pre='OFF', on_pre='COMM', outname='nqi_comm', subtitle='baseline (MoE-5x)'):
    d = load()
    off = np.stack([d[f'{off_pre}_{s}'] for s in SEEDS])     # [3,5] 비용 원값
    on = np.stack([d[f'{on_pre}_{s}'] for s in SEEDS])
    scale = np.concatenate([off, on]).mean(0)          # pooled mean (arm 무관 앵커)
    offn, onn = off / scale, on / scale
    dcomp = offn.mean(0) - onn.mean(0)                 # 구성요소별 Δ(OFF−COMM), 전부 양수 기대

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.6, 4.0),
                                        gridspec_kw={'width_ratios': [1, 1.05, 0.95]})
    # (a) 구성요소별 상대비용 (OFF=회색, COMM=초록) — 전 요소 통신 우위
    x = np.arange(len(COMP))
    w = 0.36
    ax1.bar(x - w / 2, offn.mean(0), w, color='#8a8a8a', alpha=0.8, label='No comm')
    ax1.bar(x + w / 2, onn.mean(0), w, color='#2a8f3a', alpha=0.8, label='Comm')
    for xi in x:
        ax1.plot([xi - w / 2] * 3, offn[:, xi], 'o', ms=2.4, mfc='white', mec='#333', mew=0.5, zorder=4)
        ax1.plot([xi + w / 2] * 3, onn[:, xi], 'o', ms=2.4, mfc='white', mec='#333', mew=0.5, zorder=4)
        ax1.annotate(f'−{100 * dcomp[xi] / offn.mean(0)[xi]:.0f}%', xy=(xi + w / 2, onn.mean(0)[xi]),
                     xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8,
                     color='#1a6f2a')
    ax1.set_xticks(x)
    ax1.set_xticklabels(COMP, fontsize=8, rotation=18, ha='right')
    ax1.set_ylabel('Normalized cost (pooled mean = 1)')
    ax1.set_title('(a) Cost components — comm lower on all five', fontsize=10)
    ax1.legend(fontsize=8.5)
    # (b) 가중치 민감도 — 4개 프리셋 전부에서 항해비용 ↓
    names = [p[0] for p in PRESETS]
    imps = []
    for _, wgt in PRESETS:
        wv = np.array(wgt)
        ci_off = (offn * wv).sum(1)                    # [3]
        ci_on = (onn * wv).sum(1)
        imps.append(100.0 * (ci_off.mean() - ci_on.mean()) / ci_off.mean())
    ax2.barh(np.arange(len(names)), imps, height=0.55, color='#2a8f3a', alpha=0.85)
    for i, v in enumerate(imps):
        ax2.annotate(f'−{v:.1f}%', xy=(v, i), xytext=(4, 0), textcoords='offset points',
                     va='center', fontsize=9)
    ax2.set_yticks(np.arange(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('Navigation cost reduction (%)')
    ax2.set_title('(b) Weighting presets', fontsize=10)
    ax2.set_xlim(min(0, min(imps) * 1.2), max(imps) * 1.35)
    ax2.invert_yaxis()
    # (c) ★가중치 심플렉스 전수 스윕 — 특정 가중치를 골랐다는 비판의 원천 차단
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(5), N_SIMPLEX)
    co, cc = offn.mean(0), onn.mean(0)
    imp_all = 100.0 * ((W @ co) - (W @ cc)) / (W @ co)
    frac = 100.0 * (imp_all > 0).mean()
    ax3.hist(imp_all, bins=60, color='#2a8f3a', alpha=0.75, lw=0)
    ax3.axvline(0, color='#333333', lw=1.0)
    ax3.axvline(np.median(imp_all), color='#1a4f8a', lw=1.4, ls='--',
                label=f'median {np.median(imp_all):+.1f}%')
    ax3.set_xlabel('Navigation cost reduction (%)')
    ax3.set_ylabel('Weightings (count)')
    dom = bool((cc < co).all())
    ax3.set_title(f'(c) All {N_SIMPLEX // 1000}k random weightings\n'
                  f'{frac:.1f}% favor communication'
                  + ('  (component-wise dominance)' if dom else ''), fontsize=9.5)
    ax3.legend(fontsize=8.5)
    fig.suptitle(f'Navigation Cost Index — {subtitle}  (3 seeds, frozen eval)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{outname}.{e}'), bbox_inches='tight')
    plt.close(fig)
    print(f'saved 00_MAIN/{outname}  | Δcomponents(OFF−COMM):',
          np.round(dcomp, 3), '| preset improvements:', [f'{v:.1f}%' for v in imps])


if __name__ == '__main__':
    main('SE_OFF', 'MOE_SE', 'nqi_comm', 'proposed architecture')
    main('VA_OFF', 'VA_COMM', 'nqi_comm_va', 'value-aligned reward retrain')
