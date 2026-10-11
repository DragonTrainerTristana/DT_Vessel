# 쉬운 설명 PDF 용 그림 4개 (SVG). 원자료: runs/2026-09-30_t/out/*.csv·eval_t_*.txt,
# runs/2026-09-26_comm_ext_batch/eval_x_off_*.txt, runs/2026-09-28_role_design/probe_results.txt.
# Mac 에서 파싱·그림만 함(시뮬레이션 없음).
import csv, io, re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.normpath(os.path.join(HERE, '..'))           # runs/2026-09-30_t
RUNS = os.path.normpath(os.path.join(T, '..'))
OUT = os.path.join(T, 'out')

COMM, OFF, TRUNK = '#2a78d6', '#eb6834', '#8a8985'      # 팔레트 검증 통과(blue/orange, light)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.family': 'Apple SD Gothic Neo', 'axes.unicode_minus': False, 'svg.fonttype': 'none',
    'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})
BRANCH = 9043968


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index('<svg'):]


def load(f):
    rows = list(csv.DictReader(open(f)))
    return [int(r['step']) for r in rows], [float(r['raw_reward']) for r in rows]


def ma(v, w=20):
    out = []
    for i in range(len(v)):
        lo, hi = max(0, i - w // 2), min(len(v), i + w // 2)
        out.append(sum(v[lo:hi]) / (hi - lo))
    return out


# 그림 1: 결정당 reward 학습 곡선 (0–16M 전체, 0 선·첫 점·분기선 포함)
def fig_curves():
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.5), sharey=True)
    for ax, s in zip(axs, (43, 46, 47)):
        st, off = load(os.path.join(OUT, f't_off_s{s}.csv'))
        _, a6 = load(os.path.join(OUT, f't_a6_s{s}.csv'))
        x = [t / 1e6 for t in st]
        mo, ma6 = ma(off), ma(a6)
        ib = next(i for i, t in enumerate(st) if t > BRANCH)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(BRANCH / 1e6, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.plot(x[:ib], off[:ib], color=TRUNK, lw=0.6, alpha=0.35)
        ax.plot(x[:ib], mo[:ib], color=TRUNK, lw=2, label='공통 뿌리(통신 없음)')
        ax.plot(x[ib - 1:], off[ib - 1:], color=OFF, lw=0.6, alpha=0.3)
        ax.plot(x[ib - 1:], a6[ib - 1:], color=COMM, lw=0.6, alpha=0.3)
        ax.plot(x[ib - 1:], mo[ib - 1:], color=OFF, lw=2, label='통신 없음(OFF)')
        ax.plot(x[ib - 1:], ma6[ib - 1:], color=COMM, lw=2, label='통신(a6)')
        ax.plot([x[0]], [off[0]], 'o', ms=5, color=TRUNK, mec='white', mew=1.5)
        ax.set_title(f'시드 {s}', fontsize=9, color=INK)
        ax.set_xlim(0, 21.5)
        ax.set_xticks([0, 4, 9.04, 16])
        ax.spines['bottom'].set_bounds(0, 16)
        ax.set_xticklabels(['0', '4M', '9M\n통신 켬', '16M'])
        yc, yo = ma6[-1], mo[-1]
        if abs(yc - yo) < 0.12:                      # 끝 라벨 겹침 방지
            yc, yo = (yc + 0.06, yo - 0.06) if yc >= yo else (yc - 0.06, yo + 0.06)
        ax.text(x[-1] + 0.3, yc, f'통신 {ma6[-1]:+.2f}', color=INK, fontsize=6.5, va='center')
        ax.text(x[-1] + 0.3, yo, f'OFF {mo[-1]:+.2f}', color=INK2, fontsize=6.5, va='center')
    axs[0].set_ylabel('결정당 reward')
    axs[0].annotate('첫 점: 출발 직후\n(비교 불가)', xy=(0.07, 0.36), xytext=(1.6, 0.42), fontsize=7,
                    color=INK2, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    axs[0].annotate('바닥: 배가 움직이며\n충돌 많아짐', xy=(4.3, -1.13), xytext=(0.6, -1.88), fontsize=7,
                    color=INK2, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    h, l = axs[2].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12), fontsize=8)
    return svg(fig)


# 그림 2: 레이더 안 유지선 준수(실제 타각) — 옛 보상 vs 지금 보상
PAT = re.compile(r'\[encounter-COLREGs/실제타각\] C=\s*([\d.]+)%.*StandOn\(R17\)=\s*([\d.]+)%')


def standon(path):
    m = PAT.search(open(path, encoding='utf-8').read())
    return float(m.group(2))


def fig_standon():
    groups = [('옛 보상\nOFF (x_)', OFF, [os.path.join(RUNS, '2026-09-26_comm_ext_batch', f'eval_x_off_s{s}.txt') for s in (43, 44, 45)]),
              ('지금 보상\nOFF', OFF, [os.path.join(OUT, f'eval_t_off_s{s}.txt') for s in (43, 46, 47)]),
              ('지금 보상\nOFF 재분기', OFF, [os.path.join(OUT, f'eval_t_offb_s{s}.txt') for s in (43, 46, 47)]),
              ('지금 보상\n통신 a6', COMM, [os.path.join(OUT, f'eval_t_a6_s{s}.txt') for s in (43, 46, 47)]),
              ('지금 보상\n통신 a8(역할 선언)', COMM, [os.path.join(OUT, f'eval_t_a8_s{s}.txt') for s in (43, 46, 47)])]
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    vals = []
    for i, (lab, c, fs) in enumerate(groups):
        v = [standon(f) for f in fs]
        vals.append(v)
        m = sum(v) / len(v)
        ax.bar(i, m, width=0.55, color=c, alpha=0.85, zorder=2)
        ax.scatter([i] * 3, v, s=16, color=INK, zorder=3)
        ax.text(i, max(v) + 3, f'평균 {m:.1f}', ha='center', fontsize=7.5, color=INK)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel('유지선 준수 %')
    ax.set_ylim(0, 70)
    ax.grid(axis='x', visible=False)
    return svg(fig), vals


# 그림 3: 손으로 짠 규칙 배 — 레이더 규칙 vs 통신 300 m 역할 규칙 (09-28 조사, 2시드)
def fig_scripted():
    rows = {}
    for line in open(os.path.join(RUNS, '2026-09-28_role_design', 'probe_results.txt'), encoding='utf-8'):
        d = dict(kv.split('=') for kv in line.split() if '=' in kv)
        if d.get('mode') in ('radar', 'comm'):
            rows.setdefault(d['mode'], []).append((float(d['encC%']), float(d['roleKeptSafe%'])))
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.2))
    for ax, k, title in ((axs[0], 0, '56 m 안 COLREGs 점수 %\n(한 척씩 채점)'),
                         (axs[1], 1, '둘 다 역할 지킴 + 안전 통과 %\n(짝으로 채점)')):
        for i, (mode, c, lab) in enumerate((('radar', OFF, '레이더 규칙\n(56 m만 봄)'), ('comm', COMM, '통신 규칙\n(300 m 역할 앎)'))):
            v = [r[k] for r in rows[mode]]
            m = sum(v) / len(v)
            ax.bar(i, m, width=0.55, color=c, alpha=0.85, zorder=2)
            ax.scatter([i] * len(v), v, s=14, color=INK, zorder=3)
            ax.text(i, max(v) * 1.04 + 0.3, f'{m:.1f}', ha='center', fontsize=8, color=INK)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['레이더 규칙\n(56 m만 봄)', '통신 규칙\n(300 m 역할 앎)'], fontsize=7.5)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.grid(axis='x', visible=False)
        ax.set_ylim(0, (100 if k == 0 else 18))
    return svg(fig), rows


if __name__ == '__main__':
    import json
    c = fig_curves()
    s, sv = fig_standon()
    p, pr = fig_scripted()
    open(os.path.join(HERE, 'fig_curves.svg'), 'w').write(c)
    open(os.path.join(HERE, 'fig_standon.svg'), 'w').write(s)
    open(os.path.join(HERE, 'fig_scripted.svg'), 'w').write(p)
    print(json.dumps({'standon': sv, 'scripted': pr}, ensure_ascii=False))
