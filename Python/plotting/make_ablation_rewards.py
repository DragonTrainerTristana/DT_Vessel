"""
20_REWARD/ — ablation 축별 reward 학습곡선 (Dropbox 참조 폴더 스타일 재현).
  스타일: x축 Training Steps (Millions) · y축 Average Reward · 굵은 평활선 ·
          굵은 제목 · 범례 우하단 · 통신 전환 시점 점선
  지표: 결과 기반 가중 보상 = W_GOAL·도착 − W_COLL·충돌 − W_TO·타임아웃 (모든 조건 동일 채점)
        창별 종료 에피소드 수로 가중 합산(pooling) → 표본 적은 창의 잡음 제거
  시드는 통합(평균), 그림 내 시드 표기 없음.
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


def dense(x, y, n=700):
    """측정점을 그대로 지나가는 부드러운 곡선으로 촘촘히 그린다.
    PCHIP은 형상보존 보간이라 없는 봉우리를 만들지 않는다. 값 자체는 바뀌지 않는다."""
    if len(x) < 3:
        return x, y
    xi = np.linspace(x[0], x[-1], n)
    return xi, PchipInterpolator(x, y)(xi)

FIG = os.path.dirname(os.path.abspath(__file__))
# 데이터 경로: 프로젝트 내 _data(고정 사본)를 우선 사용 → 임시폴더가 지워져도 재생성 가능
_LOCAL = os.path.join(FIG, '_data')
SCR = os.environ.get('VESSEL_LOG_DIR', _LOCAL if os.path.isdir(_LOCAL) else
    r"C:/Users/sengh/AppData/Local/Temp/claude/C--Users-sengh-Dropbox-Private-Paper-Project-0702-NewVessel/e8fe723f-83ff-4c13-af4f-54455574c53b/scratchpad")
OUT = os.path.join(FIG, '20_REWARD'); os.makedirs(OUT, exist_ok=True)

W_GOAL, W_COLL, W_TO = 1.5, 6.0, 0.5
SCALE, ROLL, MIN_EP = 3.0, 4, 0   # 카운트 가중이라 표본필터 불필요(오히려 run간 정렬을 깨뜨렸음)
X0 = 4.0                       # 표시 시작 (Millions)
X1 = float(os.environ.get('VESSEL_FIG_XMAX', '99.0'))   # 표시 상한(기본=전체)
if X1 > 1000:      # 이 스크립트의 x축 단위는 '백만 step'. raw step으로 들어오면 환산.
    X1 /= 1e6
OUT = os.environ.get('VESSEL_FIG_OUT', OUT)             # 출력 폴더 override
YLIM = None                   # 전 그림 공통 y범위 (2-pass로 계산해 주입)
COMM_ON = 9.0
C = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'axes.labelsize': 13, 'legend.fontsize': 11, 'legend.frameon': True,
    'legend.framealpha': 0.9, 'axes.grid': True, 'grid.alpha': 0.25,
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'axes.linewidth': 1.0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 110, 'savefig.dpi': 260,
})
LINE = re.compile(r'dec=([\d.]+)M \| ep=(\d+) len~\d+ \| goal=([\d.]+)% vColl=([\d.]+)%'
                  r' oColl=([\d.]+)% TO=([\d.]+)%')


def load(stem):
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
        x.append(float(m.group(1))); n.append(e)
        g.append(float(m.group(3)) * e / 100.0)
        c.append((float(m.group(4)) + float(m.group(5))) * e / 100.0)
        t.append(float(m.group(6)) * e / 100.0)
    if len(x) <= 20:
        return None
    return [np.array(v) for v in (x, n, g, c, t)]


BIN = 0.33   # step 구간(M) — run마다 로그 창 수가 달라 인덱스 합산은 어긋난다(실측 확인). step으로 묶는다.


def series(stems):
    """여러 run을 *step 구간별로* 카운트 합산 → (step[M], raw 보상, pooled 보상)."""
    runs = [r for r in (load(s) for s in stems) if r is not None]
    if not runs:
        return None
    nb = int(np.ceil(16.2 / BIN))
    n, g, c, t = (np.zeros(nb) for _ in range(4))
    for r in runs:
        idx = np.clip(np.round(r[0] / BIN).astype(int), 0, nb - 1)
        for arr, dst in zip(r[1:], (n, g, c, t)):
            np.add.at(dst, idx, arr)
    keep = n > 0
    x = np.arange(nb) * BIN
    x, n, g, c, t = x[keep], n[keep], g[keep], c[keep], t[keep]
    L = len(x)

    def val(sl):
        nn = max(n[sl].sum(), 1e-9)
        return (W_GOAL * g[sl].sum() / nn - W_COLL * c[sl].sum() / nn
                - W_TO * t[sl].sum() / nn) * SCALE
    raw = np.array([val(slice(i, i + 1)) for i in range(L)])
    sm = np.array([val(slice(max(0, i - ROLL + 1), i + 1)) for i in range(L)])
    return x, raw, sm


def _render(fname, title, entries, note_comm=True, x0=None):
    """entries: [(label, [stems]), ...]"""
    x0 = X0 if x0 is None else x0
    data = [(lab, series(st)) for lab, st in entries]
    data = [(lab, d) for lab, d in data if d is not None]
    if len(data) < 2:
        print(f'  {fname}: 데이터 부족'); return
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    allv = []
    for i, (lab, (x, raw, sm)) in enumerate(data):
        k = (x >= x0) & (x <= X1)
        col = C[i % len(C)]
        xd, yd = dense(x[k], sm[k])
        ax.plot(xd, yd, color=col, lw=2.0, label=lab, zorder=3,
                solid_joinstyle='round', solid_capstyle='round')
        allv.append(sm[k])
    # 로버스트 y범위: 한 곡선의 초반 저점이 전체를 압축하지 않도록 하위 백분위 사용
    v = np.concatenate(allv)
    lo, hi = float(v.min()), float(v.max())
    if note_comm:
        ax.axvline(COMM_ON, color='#888888', ls='--', lw=1.2, zorder=1)
        from matplotlib.transforms import blended_transform_factory as _bl
        ax.text(COMM_ON + 0.1, 0.02, 'Communication ON', color='#777777', fontsize=10,
                transform=_bl(ax.transData, ax.transAxes))
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    else:
        pad = (hi - lo) * 0.12
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(x0, X1)
    ax.set_xlabel('Training Steps (Millions)')
    ax.set_ylabel('Average Reward')
    ax.set_title(title, pad=12)
    ax.legend(loc='lower right')
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{fname}.{e}'), bbox_inches='tight')
    plt.close(fig)
    finals = ', '.join(f'{lab} {d[2][(d[0] <= X1)][-1]:.2f}' for lab, d in data)
    print(f'saved {fname}  | {finals}')


OFF3 = ['qf_SE_OFF_s42', 'qf_SE_OFF_s43', 'qf_SE_OFF_s44']   # 공유지각 구조 (저분산)
ON3 = ['qd_MOE_SE_s42', 'qd_MOE_SE_s43', 'qd_MOE_SE_s44']
BOFF = ['base_off', 'base_off_s43', 'base_off_s44']
BON = ['base_comm', 'base_comm_s43', 'base_comm_s44']

FIGS = []


def figure(*a, **k):          # 등록만 하고 실제 렌더는 아래에서
    FIGS.append((a, k))


if __name__ == '__main__':
    # 기준 아키텍처 = 공유지각 MoE (제안 구조). 데이터가 있는 축은 전부 이 구조로 통일.
    figure('ablation_reward_1_communication', 'Effect of Communication',
           [('No Communication', OFF3), ('Communication', ON3)])
    figure('ablation_reward_2_moe', 'MoE Architecture',
           [('Single network', ['q_MOE_SINGLE', 'qd_MOE_SINGLE_s43', 'qd_MOE_SINGLE_s44']),
            ('MoE (isolated experts)', ['q_MOE_ISO']),
            ('MoE (proposed)', ON3)], x0=5.0)
    figure('ablation_reward_3_aggregation', 'Message Aggregation',
           [('Nearest-1', ['qf_SE_NEAR1_s42', 'qf_SE_NEAR1_s43', 'qf_SE_NEAR1_s44']),
            ('Aggregation of 4 neighbours', ON3)])
    figure('ablation_reward_4_reward_design', 'Reward Design',
           [('Value-aligned reward', ['qi_VA_COMM_s42', 'qi_VA_COMM_s43', 'qi_VA_COMM_s44']),
            ('Standard reward', ON3)])
    figure('ablation_reward_5_dimension', 'Message Dimension',
           [('Dimension 2', ['v2_DIM2', 'q_DIM2_s43', 'q_DIM2_s44']),
            ('Dimension 4', ['v2_DIM4', 'qe_DIM4_s43', 'qe_DIM4_s44']),
            ('Dimension 6', BON),
            ('Dimension 8', ['v2_DIM8', 'qe_DIM8_s43', 'qe_DIM8_s44']),
            ('Dimension 10', ['v2_DIM10', 'qh_DIM10_s43', 'qh_DIM10_s44']),
            ('Dimension 12', ['v2_DIM12', 'q_DIM12_s43', 'q_DIM12_s44'])])
    figure('ablation_reward_6_colregs', 'COLREGs Reinforcement',
           [('Without COLREGs term', ['q_COLREGSOFF']),
            ('With COLREGs term', BON)])

    # ── 1-pass: 전 그림의 곡선 범위를 모아 공통 y범위 확정 ──
    _v = []
    for a, k in FIGS:
        x0 = k.get('x0', X0)
        for lab, st in a[2]:
            d = series(st)
            if d is None:
                continue
            x, _, sm = d
            m = (x >= x0) & (x <= X1)
            if m.any():
                _v.append(sm[m])
    _v = np.concatenate(_v)
    _lo, _hi = float(np.percentile(_v, 1)), float(np.percentile(_v, 100))
    _pad = (_hi - _lo) * 0.08
    YLIM = (_lo - _pad, _hi + _pad)
    print(f'공통 y범위: {YLIM[0]:.2f} ~ {YLIM[1]:.2f}')

    # ── 2-pass: 렌더 ──
    _reg = figure
    figure = _render
    for a, k in FIGS:
        _render(*a, **k)
