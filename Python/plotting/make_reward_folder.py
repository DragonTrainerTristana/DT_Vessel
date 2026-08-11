"""
20_REWARD/ — reward 그래프 전용 폴더. 모든 그림 동일 처리:
  · y = task return = W_GOAL·도착% − W_COLL·충돌% − W_TO·타임아웃%  (양 arm/모든 조건 동일 채점, ×3)
  · 전 구간(0~16M) 표준 학습곡선, 시드 통합(그림 내 시드 표기 없음), 창 잡음 완화(rolling 20)
  · 16M 정밀 평가(frozen eval) 종점을 다이아몬드로 병기
※ 가중치 정의는 논문 캡션/방법론에 반드시 명시할 것 (W_COLL=6: 해운 실비용 구조 반영).
출력: reward_learning_curve[_baseline|_dimension|_reward_design].png, reward_episode_return.png
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
OUT = os.path.join(FIG, '20_REWARD'); os.makedirs(OUT, exist_ok=True)
ps.apply()
SCALE, COMM_ON, ROLL, MIN_EP = 3.0, 9e6, 10, 0   # 카운트 가중이라 표본필터 불필요
BIN = 3.3e5   # step 구간 — run마다 길이가 다를 수 있어 인덱스가 아닌 step으로 정렬
X0 = 4.0e6   # 학습 초기 폭락(미학습 정책) 구간 제외 — 비교 구간의 y축 해상도 확보
X1 = float(os.environ.get('VESSEL_FIG_XMAX', '1e12'))  # 표시 상한(기본=전체)
W_GOAL, W_COLL, W_TO = 1.5, 6.0, 0.5
GREY, GREEN = '#555555', '#2a8f3a'
YLAB = 'Reward'
LINE = re.compile(r'dec=([\d.]+)M \| ep=(\d+) len~\d+ \| goal=([\d.]+)% vColl=([\d.]+)%'
                  r' oColl=([\d.]+)% TO=([\d.]+)%')
EVPAT = re.compile(r'\[(\w+)\].*?goal=\s*([\d.]+)%.*?vColl=\s*([\d.]+)%.*?oColl=\s*([\d.]+)%'
                   r'.*?TO=\s*([\d.]+)%')

SETS = {
    'baseline':   (['base_off', 'base_off_s43', 'base_off_s44'],
                   ['base_comm', 'base_comm_s43', 'base_comm_s44'],
                   ('OFF', 'COMM')),
    'va':         (['qi_VA_OFF_s42', 'qi_VA_OFF_s43', 'qi_VA_OFF_s44'],
                   ['qi_VA_COMM_s42', 'qi_VA_COMM_s43', 'qi_VA_COMM_s44'],
                   ('VA_OFF', 'VA_COMM')),
    'far':        (['qk_FAR_OFF_s42', 'qk_FAR_OFF_s43', 'qk_FAR_OFF_s44'],
                   ['qk_FAR_COMM_s42', 'qk_FAR_COMM_s43', 'qk_FAR_COMM_s44'],
                   ('FAR_OFF', 'FAR_COMM')),
}
DIMS = {2: ['v2_DIM2', 'q_DIM2_s43', 'q_DIM2_s44'],
        4: ['v2_DIM4', 'qe_DIM4_s43', 'qe_DIM4_s44'],
        6: ['base_comm', 'base_comm_s43', 'base_comm_s44'],
        8: ['v2_DIM8', 'qe_DIM8_s43', 'qe_DIM8_s44'],
        10: ['v2_DIM10', 'qh_DIM10_s43', 'qh_DIM10_s44'],
        12: ['v2_DIM12', 'q_DIM12_s43', 'q_DIM12_s44']}


def load(stem):
    """로그 → (step, 종료수, 도착수, 충돌수, 타임아웃수). 비율이 아니라 *카운트*를 반환한다:
    창마다 종료 표본 수가 크게 다르므로(파도), 비율 평균 대신 카운트를 합쳐(pooling) 비율을 내야
    통계적으로 올바르다. 표본 적은 창의 튐이 자동으로 완화된다."""
    p = os.path.join(SCR, stem + '.log')
    if not os.path.exists(p):
        return None
    x, n, g, c, t = [], [], [], [], []
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = LINE.search(ln)
        if not m:
            continue
        e = int(m.group(2))
        if e < MIN_EP:
            continue
        x.append(float(m.group(1)) * 1e6); n.append(e)
        g.append(float(m.group(3)) * e / 100.0)
        c.append((float(m.group(4)) + float(m.group(5))) * e / 100.0)
        t.append(float(m.group(6)) * e / 100.0)
    if len(x) <= 20:
        return None
    return tuple(np.array(v) for v in (x, n, g, c, t))


def agg(stems):
    """여러 run을 *step 구간별로* 카운트 합산 (인덱스 정렬 금지: run 길이가 다르면 어긋난다)."""
    runs = [r for r in (load(s) for s in stems) if r is not None]
    if not runs:
        return None
    nb = int(np.ceil(16.4e6 / BIN))
    n, g, c, t = (np.zeros(nb) for _ in range(4))
    for r in runs:
        idx = np.clip(np.round(r[0] / BIN).astype(int), 0, nb - 1)
        for arr, dst in zip(r[1:], (n, g, c, t)):
            np.add.at(dst, idx, arr)
    keep = n > 0
    x = (np.arange(nb)) * BIN
    return x[keep], n[keep], g[keep], c[keep], t[keep]


def pooled(a, k=None):
    """k창 pooling으로 task return 곡선 산출 (카운트 합산 후 비율화)."""
    k = k or ROLL
    _, n, g, c, t = a
    out = np.empty(len(n))
    for i in range(len(n)):
        s = slice(max(0, i - k + 1), i + 1)
        nn = max(n[s].sum(), 1e-9)
        out[i] = (W_GOAL * g[s].sum() / nn - W_COLL * c[s].sum() / nn
                  - W_TO * t[s].sum() / nn) * SCALE
    return out


def evals():
    d = {}
    p = os.path.join(SCR, 'metrics_v2.txt')
    if not os.path.exists(p):
        return d
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = EVPAT.search(ln)
        if m:
            g, v, o, t = (float(m.group(i)) for i in (2, 3, 4, 5))
            d[m.group(1)] = (W_GOAL * g - W_COLL * (v + o) - W_TO * t) * SCALE
    return d


def eval_end(prefix, ev):
    vals = [ev[f'{prefix}_{s}'] for s in ('s42', 's43', 's44') if f'{prefix}_{s}' in ev]
    return (float(np.mean(vals)), float(_sd(vals))) if vals else None


def pair_curve(key, title, fname, ax=None, show_legend=True):
    offs, ons, evp = SETS[key]
    a, b = agg(offs), agg(ons)
    if a is None or b is None:
        print(f'  {fname}: 데이터 없음'); return False
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8.2, 4.7))
    xs = np.intersect1d(a[0], b[0]); xs = xs[xs <= X1]
    ia = np.searchsorted(a[0], xs); ib = np.searchsorted(b[0], xs)
    x = xs
    ea, eb = pooled(a)[ia], pooled(b)[ib]
    ra, rb = pooled(a, 1)[ia], pooled(b, 1)[ib]
    ax.plot(x, ea, color=GREY, lw=1.9, label='No communication', zorder=3)
    ax.plot(x, eb, color=GREEN, lw=1.9, label='Communication', zorder=3)
    ax.fill_between(x, ea, eb, where=(eb > ea), color=GREEN, alpha=0.17, lw=0,
                    zorder=1, interpolate=True)
    xr = x[-1]
    ax.axvline(COMM_ON, color='#999999', ls='--', lw=0.9, zorder=1)
    from matplotlib.transforms import blended_transform_factory as _blend
    ax.text(COMM_ON + 1.3e5, 0.03, 'communication ON', color='#888888', fontsize=8.5,
            transform=_blend(ax.transData, ax.transAxes))
    sel = x >= COMM_ON
    mo, mc = ra[sel].mean(), rb[sel].mean()
    frac = 100 * (eb[sel] > ea[sel]).mean()
    ax.set_xlabel('Training steps'); ax.set_ylabel(YLAB)
    ax.set_title(f'{title}\nafter communication is enabled: {mo:.2f} → {mc:.2f}'
                 f'   (ahead over {frac:.0f}% of the phase)', fontsize=10.5)
    if show_legend:
        ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(X0, min(xr, X1) + 0.2e6)
    vis = x >= X0
    v0, v1 = min(ea[vis].min(), eb[vis].min()), max(ea[vis].max(), eb[vis].max())
    pad = (v1 - v0) * 0.12
    ax.set_ylim(v0 - pad, v1 + pad)
    if own:
        fig.tight_layout()
        for e in ('png', 'pdf'):
            fig.savefig(os.path.join(OUT, f'{fname}.{e}'), bbox_inches='tight')
        plt.close(fig)
        print(f'saved 20_REWARD/{fname}  | {mo:.2f} → {mc:.2f}, ahead {frac:.0f}%')
    return True


def dimension():
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    cols = cm.viridis(np.linspace(0, 0.88, len(DIMS)))
    for (d, stems), c in zip(sorted(DIMS.items()), cols):
        r = agg(stems)
        if r is None:
            continue
        _k = r[0] <= X1
        ax.plot(r[0][_k], pooled(r)[_k], color=c, lw=1.5, label=f'dim {d}')
    ax.axvline(COMM_ON, color='#999999', ls='--', lw=0.9)
    ax.set_xlabel('Training steps'); ax.set_ylabel(YLAB)
    ax.set_title('Learning curve by message dimension', fontsize=10.5)
    ax.legend(fontsize=8.5, ncol=3, loc='lower right')
    ax.set_xlim(X0, min(16.4e6, X1 + 0.2e6))
    ys = [pooled(agg(st))[(agg(st)[0] >= X0) & (agg(st)[0] <= X1)] for st in DIMS.values() if agg(st) is not None]
    v0 = min(v.min() for v in ys); v1 = max(v.max() for v in ys); pad = (v1 - v0) * 0.12
    ax.set_ylim(v0 - pad, v1 + pad)
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'reward_learning_curve_dimension.{e}'),
                    bbox_inches='tight')
    plt.close(fig); print('saved 20_REWARD/reward_learning_curve_dimension')


def reward_design():
    keys = [('baseline', '(a) Standard reward'), ('va', '(b) Value-aligned reward')]
    keys = [(k, t) for k, t in keys if agg(SETS[k][0]) is not None]
    if agg(SETS['far'][0]) is not None and agg(SETS['far'][1]) is not None:
        keys.append(('far', '(c) Far-field risk in reward'))
    fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (k, t) in zip(axes, keys):
        pair_curve(k, t, None, ax=ax, show_legend=(ax is axes[0]))
    fig.suptitle('Reward design comparison', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'reward_learning_curve_reward_design.{e}'),
                    bbox_inches='tight')
    plt.close(fig); print('saved 20_REWARD/reward_learning_curve_reward_design')


def episode_return():
    d = {}
    for ln in open(os.path.join(SCR, 'metrics_v2.txt'), encoding='utf-8', errors='ignore'):
        m = re.search(r'\[(\w+)\].*?epReward=\s*(-?[\d.]+)', ln)
        if m:
            d[m.group(1)] = float(m.group(2)) * SCALE
    S = ('s42', 's43', 's44')
    eo = [d[f'OFF_{s}'] for s in S if f'OFF_{s}' in d]
    ec = [d[f'COMM_{s}'] for s in S if f'COMM_{s}' in d]
    if not eo or not ec:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.bar(0, np.mean(eo), 0.5, yerr=_sd(eo), color=GREY, alpha=0.85, capsize=3,
           error_kw=dict(lw=0.9, ecolor='#333333'))
    ax.bar(1, np.mean(ec), 0.5, yerr=_sd(ec), color=GREEN, alpha=0.85, capsize=3,
           error_kw=dict(lw=0.9, ecolor='#333333'))
    for xx, v in ((0, np.mean(eo)), (1, np.mean(ec))):
        ax.annotate(f'{v:.0f}', xy=(xx, v), xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=10)
    rel = 100 * (np.mean(ec) - np.mean(eo)) / abs(np.mean(eo))
    ax.annotate(f'{rel:+.0f}%', xy=(0.5, max(np.mean(eo), np.mean(ec))), xytext=(0, 34),
                textcoords='offset points', ha='center', fontsize=13, color=GREEN)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['No\ncommunication', 'Communication'])
    ax.set_ylabel('Episode return')
    ax.set_title('Reward accumulated per voyage', fontsize=10.5)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, max(np.mean(eo), np.mean(ec)) * 1.45)
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'reward_episode_return.{e}'), bbox_inches='tight')
    plt.close(fig); print('saved 20_REWARD/reward_episode_return')


if __name__ == '__main__':
    episode_return()
