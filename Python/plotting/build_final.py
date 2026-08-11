"""
99_FINAL/ — 논문 제출용 최종 그림 (주제별 폴더).
각 폴더 = reward 학습곡선 1장 + 지표별 막대 그림 여러 장 (참조 폴더 구성 방식).
사용: python regenerate_all.py  →  python build_final.py
"""
import os, re, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = os.path.dirname(os.path.abspath(__file__))
SCR = os.environ.get('VESSEL_LOG_DIR', os.path.join(FIG, '_data'))
DST = os.path.join(FIG, '99_FINAL')
BAR = '#7b7be0'

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'axes.labelsize': 13, 'axes.labelweight': 'bold',
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'axes.linewidth': 1.2,
    'axes.spines.top': True, 'axes.spines.right': True,
    'figure.dpi': 110, 'savefig.dpi': 240,
})

PAT = re.compile(r'\[(\w+)\].*?goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%\s+oColl=\s*([\d.]+)%'
                 r'\s+TO=\s*([\d.]+)%.*?fuel=\s*([\d.]+)\s+headTravel=\s*([\d.]+)deg'
                 r'\s+minSep=\s*([\d.]+)m\s+len=\s*([\d.]+).*?colregsOK=\s*([\d.]+)%')


SITPAT = re.compile(r'\[(\w+)\].*?sit1=\s*([\d.]+)%.*?sit2=\s*([\d.]+)%'
                    r'.*?sit3=\s*([\d.]+)%.*?sit4=\s*([\d.]+)%')
SIT_NAMES = {1: 'Head-on', 2: 'Crossing\n(stand-on)',
             3: 'Crossing\n(give-way)', 4: 'Overtaking'}


def load_sit():
    """상황별 COLREGs 준수율 (metrics_sit.txt). 없으면 빈 dict."""
    d = {}
    p = os.path.join(SCR, 'metrics_sit.txt')
    if not os.path.exists(p):
        return d
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = SITPAT.search(ln)
        if m:
            d[m.group(1)] = {k: float(m.group(k + 1)) for k in (1, 2, 3, 4)}
    return d


def sit_figure(path, labels, arms_vals):
    """상황별 준수율 그룹 막대. arms_vals: [(label, {sit:mean}, {sit:sd})]"""
    sits = [1, 3, 4, 2]      # 우현양보 3종 먼저, 침로유지 마지막
    x = np.arange(len(sits)); w = 0.8 / max(len(arms_vals), 1)
    cols = ['#7b7be0', '#e08b7b', '#7be0a0', '#e0d17b']
    fig, ax = plt.subplots(figsize=(max(8.0, 2.2 * len(sits)), 5.4))
    for i, (lab, mv, sv) in enumerate(arms_vals):
        off = (i - (len(arms_vals) - 1) / 2) * w
        ms = [mv[k] for k in sits]; es = [sv[k] for k in sits]
        ax.bar(x + off, ms, w * 0.92, yerr=es, label=lab, color=cols[i % len(cols)],
               edgecolor='#222222', linewidth=1.0, capsize=3,
               error_kw=dict(lw=1.0, ecolor='#222222'))
        for xi, m, e in zip(x + off, ms, es):
            ax.annotate(f'{m:.0f}', xy=(xi, m + e), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([SIT_NAMES[k] for k in sits], fontweight='bold')
    ax.set_ylabel('Compliance Rate (%)')
    ax.set_title('COLREGs Compliance by Encounter Situation  (16 vessels)', pad=14)
    ax.legend(fontsize=11)
    hi = max(m + e for _, mv, sv in arms_vals for m, e in zip(mv.values(), sv.values()))
    ax.set_ylim(0, hi * 1.22)
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(f'{path}.{e}', bbox_inches='tight')
    plt.close(fig)


def load():
    d = {}
    for ln in open(os.path.join(SCR, 'metrics_v2.txt'), encoding='utf-8', errors='ignore'):
        m = PAT.search(ln)
        if m:
            g, vc, oc, to, fu, hd, ms, ln_, ok = (float(m.group(i)) for i in range(2, 11))
            d[m.group(1)] = dict(goal=g, coll=vc + oc, to=to, fuel=fu, head=hd,
                                 minsep=ms, length=ln_, ok=ok)
    return d


def sd(v):
    v = np.asarray(v, dtype=float)
    return float(np.std(v, ddof=1)) if v.size > 1 else 0.0


# (파일명, 제목, key, y라벨, 값 서식)
METRICS = [
    ('1_colregs_compliance', 'COLREGs Compliance Rate', 'ok',     'Compliance Rate (%)',       '{:.1f}%'),
    ('3_collision_rate',     'Collision Rate',          'coll',   'Collision Rate (%)',        '{:.2f}%'),
    ('4_fuel_consumption',   'Fuel Consumption',        'fuel',   'Fuel Consumption',          '{:.0f}'),
    ('5_heading_travel',     'Heading Travel',          'head',   'Heading Travel (deg)',      '{:.0f}'),
    ('6_min_separation',     'Minimum Separation',      'minsep', 'Minimum Separation (m)',    '{:.1f}m'),
    ('7_episode_length',     'Episode Length',          'length', 'Episode Length (decisions)', '{:.0f}'),
]

S = ('s42', 's43', 's44')
TOPICS = {
    'Fig1_Communication': (
        '20_REWARD/ablation_reward_1_communication',
        [('No Communication', ['SE_OFF_' + s for s in S]),
         ('Communication',    ['MOE_SE_' + s for s in S])]),
    'Fig2_MoE_Architecture': (
        '20_REWARD/ablation_reward_2_moe',
        [('Single network',  ['MOE_SINGLE', 'MOE_SINGLE_s43', 'MOE_SINGLE_s44']),
         ('MoE (isolated)',  ['MOE_ISO']),
         ('MoE (proposed)',  ['MOE_SE_' + s for s in S])]),
    'Fig3_Message_Aggregation': (
        '20_REWARD/ablation_reward_3_aggregation',
        [('Nearest-1',      ['SE_NEAR1_' + s for s in S]),
         ('Aggregation-4',  ['MOE_SE_' + s for s in S])]),
    'Fig4_Message_Dimension': (
        '20_REWARD/ablation_reward_5_dimension',
        [('MSG_DIM=2',  ['DIM2', 'DIM2_s43', 'DIM2_s44']),
         ('MSG_DIM=4',  ['DIM4', 'DIM4_s43', 'DIM4_s44']),
         ('MSG_DIM=6',  ['COMM_s42', 'COMM_s43', 'COMM_s44']),
         ('MSG_DIM=8',  ['DIM8', 'DIM8_s43', 'DIM8_s44']),
         ('MSG_DIM=10', ['DIM10', 'DIM10_s43', 'DIM10_s44']),
         ('MSG_DIM=12', ['DIM12', 'DIM12_s43', 'DIM12_s44'])]),
    'Fig5_COLREGs_Term': (
        '20_REWARD/ablation_reward_6_colregs',
        [('Without COLREGs term', ['COLREGSOFF']),
         ('With COLREGs term',    ['COMM_s42', 'COMM_s43', 'COMM_s44'])]),
}


def bar_figure(path, title, ylabel, labels, means, errs, fmt):
    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * len(labels) + 3.0), 5.4))
    x = np.arange(len(labels))
    ax.bar(x, means, 0.62, yerr=errs, color=BAR, edgecolor='#222222', linewidth=1.0,
           capsize=4, error_kw=dict(lw=1.2, ecolor='#222222'))
    for xi, m, e in zip(x, means, errs):
        ax.annotate(fmt.format(m), xy=(xi, m + e), xytext=(0, 7),
                    textcoords='offset points', ha='center',
                    fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}  (16 vessels)', pad=14)
    lo = min(np.array(means) - np.array(errs))
    hi = max(np.array(means) + np.array(errs))
    rng = (hi - lo) or max(abs(hi), 1.0)
    ax.set_ylim(max(0.0, lo - rng * 0.35), hi + rng * 0.30)
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(f'{path}.{e}', bbox_inches='tight')
    plt.close(fig)


def main():
    d = load()
    ds = load_sit()
    # 폴더가 탐색기에 열려 있어도 안전하게: 폴더는 두고 파일만 지운다
    if os.path.isdir(DST):
        for root, _, files in os.walk(DST):
            for fn in files:
                if fn.lower().endswith(('.png', '.pdf')):
                    try:
                        os.remove(os.path.join(root, fn))
                    except OSError:
                        pass
    n_fig = 0
    for folder, (curve_src, arms) in TOPICS.items():
        out = os.path.join(DST, folder)
        os.makedirs(out, exist_ok=True)
        # reward 학습곡선
        for e in ('png', 'pdf'):
            p = os.path.join(FIG, f'{curve_src}.{e}')
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(out, f'0_reward_curve.{e}')); n_fig += 1
        # 지표별 막대
        for fname, title, key, ylab, fmt in METRICS:
            labels, means, errs = [], [], []
            for lab, keys in arms:
                vals = [d[k][key] for k in keys if k in d]
                if not vals:
                    continue
                labels.append(lab); means.append(float(np.mean(vals))); errs.append(sd(vals))
            if len(labels) < 2:
                continue
            bar_figure(os.path.join(out, fname), title, ylab, labels, means, errs, fmt)
            n_fig += 1
        # 상황별 COLREGs 준수율
        av = []
        for lab, keys in arms:
            vs = [ds[k] for k in keys if k in ds]
            if not vs:
                continue
            av.append((lab, {k: float(np.mean([v[k] for v in vs])) for k in (1, 2, 3, 4)},
                       {k: sd([v[k] for v in vs]) for k in (1, 2, 3, 4)}))
        if len(av) >= 2:
            sit_figure(os.path.join(out, '2_colregs_by_situation'), None, av)
            n_fig += 1
        print(f'  {folder}: 곡선 1 + 지표 {len([1 for f,_,k,_,_ in METRICS])}')
    print(f'99_FINAL 생성 완료 — 주제 {len(TOPICS)}개, 파일 {n_fig}쌍(png/pdf)')


if __name__ == '__main__':
    main()
