"""논문 스타일 공통 세팅 (2026-08-08 사용자 피드백 반영).
원칙: 얇은 선(1.2~1.5pt) + raw 지터를 옅게 노출(과도한 스무딩 금지) + 볼드 절제 + top/right spine 제거.
모든 Figures/*.py가 rcParams 대신 apply()를 호출한다.
"""
import numpy as np
import matplotlib.pyplot as plt


def apply():
    plt.rcParams.update({
        'font.size': 10.5, 'axes.titlesize': 11.5, 'axes.titleweight': 'normal',
        'axes.labelsize': 10.5, 'axes.labelweight': 'normal',
        'legend.fontsize': 9.5, 'legend.frameon': False,
        'axes.grid': True, 'grid.alpha': 0.15, 'grid.linewidth': 0.5,
        'axes.axisbelow': True, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'figure.dpi': 110, 'savefig.dpi': 220,
    })


def ema(y, a):
    y = np.asarray(y, dtype=float)
    o = np.empty_like(y)
    acc = y[0]
    for i, v in enumerate(y):
        acc = a * v + (1 - a) * acc
        o[i] = acc
    return o


def roll(y, k):
    """trailing rolling mean (초기 lag 없음 — EMA의 init-편향 대체용)."""
    y = np.asarray(y, dtype=float)
    o = np.empty_like(y)
    for i in range(len(y)):
        o[i] = y[max(0, i - k + 1):i + 1].mean()
    return o


def run_curve(ax, x, y, color, label=None, lw=1.4, a=0.12, raw_alpha=0.18, zorder=3):
    """단일 run 커브: raw(지터) 옅게 + EMA 본선 — RL 논문 표준 질감."""
    ax.plot(x, y, color=color, lw=0.6, alpha=raw_alpha, zorder=zorder - 1)
    ax.plot(x, ema(y, a), color=color, lw=lw, label=label, zorder=zorder)


def seed_band(ax, x, runs, color, label=None, lw=1.5, a=0.10, zorder=3, smooth=None):
    """다중 seed: 각 seed raw를 아주 옅게 + 평활 평균 본선 + ±std 밴드.
    smooth: None=EMA(a), ('roll', k)=trailing rolling mean(초기 lag 없음)."""
    L = min(len(r) for r in runs)
    if len(set(len(r) for r in runs)) > 1:
        raise ValueError('seed_band: run 길이가 다릅니다 — step 정렬이 필요합니다 '
                         f'(lengths={[len(r) for r in runs]})')
    Y_raw = np.stack([np.asarray(r[:L], dtype=float) for r in runs])
    for r in Y_raw:
        ax.plot(x[:L], r, color=color, lw=0.5, alpha=0.10, zorder=zorder - 2)
    if smooth and smooth[0] == 'roll':
        Y = np.stack([roll(r, smooth[1]) for r in Y_raw])
    else:
        Y = np.stack([ema(r, a) for r in Y_raw])
    m, s = Y.mean(0), Y.std(0)
    ax.plot(x[:L], m, color=color, lw=lw, label=label, zorder=zorder)
    ax.fill_between(x[:L], m - s, m + s, color=color, alpha=0.12, lw=0, zorder=zorder - 1)
    return x[:L], m, s
