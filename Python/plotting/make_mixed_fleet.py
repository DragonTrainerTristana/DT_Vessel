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

MODES = {
    'radar': ('Fig6_Various_Agent', 'Radar-only', 'R',
              'Mixed Fleet: Vessels Without Communication Equipment'),
    'rx':    ('Fig7_Small_Boat', 'Receive-only', 'Rx',
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


def load(path):
    """(mode, nocomm, group) -> {열: [시드별 값]}"""
    d = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row['mode'], int(row['nocomm']), row['group'])
            slot = d.setdefault(key, {})
            for k, v in row.items():
                if k in ('tag', 'mode', 'nocomm', 'group'):
                    continue
                slot.setdefault(k, []).append(float(v))
    return d


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
        ks = sorted({k for (m, k, _g) in d if m == mode})
        if not ks:
            continue
        out = os.path.join(DST, folder)
        os.makedirs(out, exist_ok=True)
        xlabels = [f'{k}{suffix}/{16 - k}C' for k in ks]
        n_seed = max(len(d[(mode, k, g)]['colregs']) for k in ks for g in ('nocomm', 'comm')
                     if (mode, k, g) in d)
        for fname, title, col, ylab, fmt, _lo in METRICS:
            series = []
            for g, lab, c in (('nocomm', nolab, C_NO), ('comm', 'Communication', C_YES)):
                ms = [float(np.mean(d[(mode, k, g)][col])) if (mode, k, g) in d else 0.0 for k in ks]
                es = [sd(d[(mode, k, g)][col]) if (mode, k, g) in d else 0.0 for k in ks]
                series.append((lab, c, ms, es))
            bar_figure(os.path.join(out, fname), title, ylab, xlabels, series, fmt)
            n_fig += 1
        # 요약 출력 (표로도 확인 가능하게)
        print(f'[{folder}] 시드 {n_seed}개, 설정 {len(ks)}개')
        for k in ks:
            a = d.get((mode, k, 'nocomm')); b = d.get((mode, k, 'comm'))
            if a and b:
                print(f'   {k:2d}{suffix}/{16-k:2d}C  COLREGs {np.mean(a["colregs"]):5.1f} vs '
                      f'{np.mean(b["colregs"]):5.1f}   연료 {np.mean(a["fuel"]):6.1f} vs '
                      f'{np.mean(b["fuel"]):6.1f}   조타 {np.mean(a["head"]):6.0f} vs '
                      f'{np.mean(b["head"]):6.0f}   시간 {np.mean(a["length"]):5.0f} vs '
                      f'{np.mean(b["length"]):5.0f}')
    print(f'혼합 함대 그림 {n_fig}쌍 저장')


if __name__ == '__main__':
    main()
