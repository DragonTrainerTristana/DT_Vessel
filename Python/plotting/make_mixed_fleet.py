"""
혼합 함대 그림 — 통신 장비를 갖춘 배와 그렇지 않은 배가 같은 바다에 섞여 있을 때.

eval_mixed.py 가 남긴 mixed_fleet.csv 를 읽어 참조 폴더와 같은 형식의 묶음 막대 그림을
만든다. 값은 로그를 그대로 읽어 평균낼 뿐, 어떤 보정도 하지 않는다.

두 가지 상황을 따로 그린다.
  radar : 통신 장비가 아예 없는 배 (보내지도 받지도 못함)
  rx    : 듣기만 하는 배 (받지만 보내지 못함)

같은 그림 안의 두 막대는 *같은 환경에서 같은 시각에 나란히 항해한* 두 무리다.
바다도 교통 상황도 동일하므로, 차이는 통신 능력에서만 온다.

사용: python make_mixed_fleet.py
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = os.path.dirname(os.path.abspath(__file__))
SCR = os.environ.get('VESSEL_LOG_DIR', os.path.join(FIG, '_data'))
DST = os.path.join(FIG, '99_FINAL')

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'axes.labelsize': 13, 'axes.labelweight': 'bold',
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'axes.linewidth': 1.2,
    'axes.spines.top': True, 'axes.spines.right': True,
    'figure.dpi': 110, 'savefig.dpi': 240,
})

C_NO, C_YES = '#e08b7b', '#5b8fd6'      # 통신 불가 / 통신 가능

# 번호 주의: Fig6은 통신 시작 시점(build_final.py) 차지. 혼합 함대는 7·8로 민다.
MODES = {
    'radar': ('Fig7_Various_Agent', 'Radar-only', 'R',
              'Mixed Fleet: Vessels Without Communication Equipment'),
    'rx':    ('Fig8_Small_Boat', 'Receive-only', 'Rx',
              'Mixed Fleet: Vessels That Can Only Listen'),
}

# (파일명, 제목, csv열, y라벨, 값 서식, 낮을수록 좋은가)
METRICS = [
    ('1_colregs_compliance', 'COLREGs Compliance Rate', 'colregs', 'Compliance Rate (%)', '{:.1f}', False),
    ('2_min_separation',     'Minimum Separation',      'minsep',  'Minimum Separation (m)', '{:.1f}', False),
    ('3_collision_rate',     'Collision Rate',          'coll',    'Collision Rate (%)', '{:.2f}', True),
    ('4_fuel_consumption',   'Fuel Consumption',        'fuel',    'Fuel Consumption', '{:.0f}', True),
    ('5_heading_travel',     'Heading Travel',          'head',    'Heading Travel (deg)', '{:.0f}', True),
    ('6_episode_time',       'Time to Reach Goal',      'length',  'Steps to Goal', '{:.0f}', True),
]


def sd(v):
    v = np.asarray(v, dtype=float)
    return float(np.std(v, ddof=1)) if v.size > 1 else 0.0


# 두 무리(통신 가능/불가)를 함대 하나로 합칠 때 쓰는 가중치.
# 무리마다 배 수가 다르므로(예: 2척 vs 14척) 단순평균은 소수 무리를 과대평가한다.
# 지표마다 실제 표본 단위가 다르다: 결과 비율은 에피소드 수, 준수율은 판정된 조우 수,
# 궤적 지표(연료·조타·시간)는 '도착한 에피소드' 수가 분모다.
WEIGHT = {'goal': 'eps', 'coll': 'eps', 'to': 'eps', 'minsep': 'eps',
          'colregs': 'colregs_n', 'fuel': 'goal_eps', 'head': 'goal_eps',
          'length': 'goal_eps', 'sit1': 'colregs_n', 'sit2': 'colregs_n',
          'sit3': 'colregs_n', 'sit4': 'colregs_n'}


def load(path):
    """(mode, nocomm, tag) -> {group: {열: 값}}   — 시드(tag)별로 두 무리를 짝지어 둔다."""
    d = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row['mode'], int(row['nocomm']), row['tag'])
            r = {k: float(v) for k, v in row.items()
                 if k not in ('tag', 'mode', 'nocomm', 'group')}
            r['goal_eps'] = r['eps'] * r['goal'] / 100.0     # 도착 에피소드 수(궤적 지표 분모)
            d.setdefault(key, {})[row['group']] = r
    return d


def fleet(groups, col):
    """한 시드의 두 무리를 표본 수 가중으로 합쳐 함대 전체 값 하나로."""
    wcol = WEIGHT.get(col, 'eps')
    num = sum(g[col] * g[wcol] for g in groups.values() if g.get(wcol, 0) > 0)
    den = sum(g[wcol] for g in groups.values() if g.get(wcol, 0) > 0)
    return num / den if den > 0 else float('nan')


def bar_figure(path, title, ylabel, xlabels, series, fmt):
    """series: [(라벨, 색, 평균목록, 편차목록)]"""
    x = np.arange(len(xlabels))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.5 * len(xlabels) + 3.0), 5.4))
    for i, (lab, col, ms, es) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * w
        ax.bar(x + off, ms, w * 0.9, yerr=es, label=lab, color=col,
               edgecolor='#222222', linewidth=1.0, capsize=3,
               error_kw=dict(lw=1.0, ecolor='#222222'))
        for xi, m, e in zip(x + off, ms, es):
            ax.annotate(fmt.format(m), xy=(xi, m + e), xytext=(0, 5),
                        textcoords='offset points', ha='center',
                        fontsize=9.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Configuration (no communication / communication)')
    ax.set_title(f'{title}  (16 vessels)', pad=14)
    if len(series) > 1:          # 함대 전체를 한 막대로 그릴 땐 범례가 의미 없다
        ax.legend(fontsize=11, loc='lower right')
    lo = min(m - e for _, _, ms, es in series for m, e in zip(ms, es))
    hi = max(m + e for _, _, ms, es in series for m, e in zip(ms, es))
    rng = (hi - lo) or max(abs(hi), 1.0)
    ax.set_ylim(max(0.0, lo - rng * 0.45), hi + rng * 0.28)
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(f'{path}.{e}', bbox_inches='tight')
    plt.close(fig)


def main():
    src = os.path.join(SCR, 'mixed_fleet.csv')
    if not os.path.exists(src):
        print(f'없음: {src} — eval_mixed.py 를 먼저 돌려야 한다')
        return
    d = load(src)
    n_fig = 0
    for mode, (folder, nolab, suffix, headline) in MODES.items():
        ks = sorted({k for (m, k, _t) in d if m == mode})
        if not ks:
            continue
        seeds = sorted({t for (m, _k, t) in d if m == mode})
        out = os.path.join(DST, folder)
        os.makedirs(out, exist_ok=True)
        xlabels = [f'{k}{suffix}/{16 - k}C' for k in ks]

        def pooled(k, col):
            """구성 k에서 시드별 함대 전체 값 목록."""
            return [fleet(d[(mode, k, t)], col) for t in seeds if (mode, k, t) in d]

        for fname, title, col, ylab, fmt, _lo in METRICS:
            ms = [float(np.mean(pooled(k, col))) for k in ks]
            es = [sd(pooled(k, col)) for k in ks]      # 시드 간 편차. 시드 1개면 0
            bar_figure(os.path.join(out, fname), title, ylab, xlabels,
                       [('Fleet average', C_YES, ms, es)], fmt)
            n_fig += 1
        # 요약 출력 — 함대 전체 값(시드 평균 ± 시드 편차)
        print(f'[{folder}] 시드 {len(seeds)}개 {seeds}, 설정 {len(ks)}개')
        for k in ks:
            v = {c: pooled(k, c) for c in ('goal', 'coll', 'colregs', 'fuel', 'head')}
            print(f'   {k:2d}{suffix}/{16-k:2d}C  도착 {np.mean(v["goal"]):5.1f}%  '
                  f'충돌 {np.mean(v["coll"]):4.2f}%  COLREGs {np.mean(v["colregs"]):5.1f}'
                  f'±{sd(v["colregs"]):.1f}  연료 {np.mean(v["fuel"]):6.1f}  '
                  f'조타 {np.mean(v["head"]):6.0f}')
    print(f'혼합 함대 그림 {n_fig}쌍 저장')


if __name__ == '__main__':
    main()
