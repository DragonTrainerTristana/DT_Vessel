#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Episode-return curves of a trunk/branch batch — spec 2026-09-30 §7 '목표 2(그림)' ② and ③.

Input (``--dir``, then ``--also_dir`` fallbacks): ``<prefix><arm>_s<seed>_ep.csv`` with columns
``step,n_ep,ep_return_mean,goal,vColl,oColl,TO`` (one row per PPO update).  Rows whose ``ep_return_mean``
is nan or whose ``n_ep`` is 0 (no episode ended in that update) are skipped, so the moving average runs over
the valid rows only.  The trunk file ``<prefix>trunk_d<dim>_s<seed>_ep.csv`` is used exactly as in
plot_reward_v3 (cross-check of the shared prefix, fill-in when a file starts after the branch point).
Rates are drawn in percent — the trainer writes them in percent (``--rate_unit pct``, default); ``auto`` /
``frac`` cover files that store fractions (the choice is printed).

Layout per row (seed, or seed mean): episode return 0..total (raw faint + centred MA) | post-branch zoom |
goal (solid) and vColl (dotted) MA in %.  stdout = the plot_reward_v3 table on the episode-return MA plus
the post-branch mean of the goal/vColl MA.  When the directory holds no ``*_ep.csv`` at all (batches that
predate the writer, e.g. s_) the script says so and exits 0.

matplotlib lives in the base anaconda python, not mltest::

    C:/Users/OSH/anaconda3/python.exe plot_ep_return.py --dir <out dir> --prefix t_ \\
        --arms off,a6,a8,a2,a4,offb --seeds 43,44,45 --out <file.png|file.pdf>   (both are written)
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from plot_reward_v3 import (BRANCH_AT, TOTAL_DEC, MA_WIN, INK, arm_styles, build_items_mean,   # noqa: E402
                            build_items_seed, centred_ma, check_trunk_match, collect, compute_table,
                            draw_curves, first_point, full_arm_for, full_ylim, print_table, save_fig,
                            setup_style, split_list, zoom_ylim)

EP_SUFFIX = '_ep.csv'
RATE_COLS = ('goal', 'vColl', 'oColl', 'TO')


def load_ep(path, note):
    """``step,n_ep,ep_return_mean,goal,vColl,oColl,TO`` -> dict of arrays sorted by step ('y' = ep_return_mean).
    Rows with nan ep_return_mean or n_ep == 0 are dropped; a repeated step keeps its last row."""
    rows, n_all, n_ok, n_nan = {}, 0, 0, 0
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            n_all += 1
            try:
                step = int(float(r['step']))
                y = float(r['ep_return_mean']) if (r.get('ep_return_mean') or '').strip() else float('nan')   # 빈 칸 = nan
                n_ep = int(float(r['n_ep'])) if (r.get('n_ep') or '').strip() else -1                        # 열 없음/빈 칸 = 모름
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(y) or n_ep == 0:
                n_nan += 1
                continue
            rates = []
            for k in RATE_COLS:
                try:
                    rates.append(float(r[k]))
                except (KeyError, TypeError, ValueError):
                    rates.append(float('nan'))
            rows[step] = (y, n_ep, rates)
            n_ok += 1
    n_dup, n_bad = n_ok - len(rows), n_all - n_ok - n_nan
    if n_nan or n_dup or n_bad:
        note(f'{os.path.basename(path)}: 행 {n_all} 중 유효 {len(rows)} (nan/사건 0 행 {n_nan} 건너뜀, '
             f'깨진 행 {n_bad} 버림, 중복 step {n_dup} 은 마지막 행)')
    steps = np.array(sorted(rows), dtype=np.int64)
    out = {'step': steps,
           'y': np.array([rows[s][0] for s in steps], dtype=float),
           'n_ep': np.array([rows[s][1] for s in steps], dtype=np.int64)}
    for i, k in enumerate(RATE_COLS):
        out[k] = np.array([rows[s][2][i] for s in steps], dtype=float)
    return out


def detect_rate_unit(series):
    """'frac' when every rate value in every run is <= 1.0 (stored as fractions), else 'pct'."""
    vals = np.concatenate([s[k][np.isfinite(s[k])] for s in series.values() for k in RATE_COLS] or [np.zeros(0)])
    if not len(vals):
        return 'pct'
    return 'frac' if float(vals.max()) <= 1.0 else 'pct'


def rate_ylim(series, mas_goal, mas_vcoll):
    vals = np.concatenate([np.concatenate([mas_goal[k], mas_vcoll[k]]) for k in series])
    vals = vals[np.isfinite(vals)]
    hi = float(vals.max()) if len(vals) else 100.0
    lo = min(0.0, float(vals.min())) if len(vals) else 0.0
    pad = 0.06 * ((hi - lo) or 1.0)
    return lo - pad, hi + pad


def make_figure(series, mas, mas_goal, mas_vcoll, seeds, arms, prefix, layout, branch, total, zoom_from, title,
                cut_at=None, full_arm=None):
    styles = arm_styles(arms)
    rows = list(seeds) if layout == 'seed' else ['mean']
    fig, axes = plt.subplots(len(rows), 3, figsize=(15.6, 3.15 * len(rows) + 0.7), squeeze=False,
                             gridspec_kw={'width_ratios': [1.35, 1, 1.1]})
    xmax = max(total, max(int(s['step'][-1]) for s in series.values())) / 1e6
    yl_full = full_ylim(series)
    yl_zoom = zoom_ylim(series, mas, zoom_from * 1e6)
    yl_rate = rate_ylim(series, mas_goal, mas_vcoll)
    cf = dict(cut_at=cut_at, full_arm=full_arm)
    dotted = (0, (1, 1.5))
    for i, row in enumerate(rows):
        if layout == 'seed':
            items = build_items_seed(series, mas, row, arms, prefix, styles, **cf)
            r_goal = build_items_seed(series, mas_goal, row, arms, prefix, styles, raw=False, **cf)
            r_vc = build_items_seed(series, mas_vcoll, row, arms, prefix, styles, raw=False, ls=dotted, label=False, **cf)
            fp = first_point(series, row)
        else:
            items = build_items_mean(series, mas, seeds, arms, prefix, styles, **cf)
            r_goal = build_items_mean(series, mas_goal, seeds, arms, prefix, styles, raw=False, **cf)
            r_vc = build_items_mean(series, mas_vcoll, seeds, arms, prefix, styles, raw=False, ls=dotted, label=False, **cf)
            fp = first_point(series)
        ax_f, ax_z, ax_r = axes[i]
        draw_curves(ax_f, items, (-0.15, xmax + 0.15), yl_full, branch, first_pt=fp, legend=(i == 0), end_fmt='{:.1f}')
        draw_curves(ax_z, items, (zoom_from, xmax + 0.7), yl_zoom, branch, end_labels=True, branch_label=False,
                    end_fmt='{:.1f}')
        draw_curves(ax_r, r_goal + r_vc, (-0.15, xmax + 0.15), yl_rate, branch, branch_label=False)
        if i == 0:
            ax_r.legend(handles=[Line2D([], [], color=INK, lw=1.2, label='goal (실선)'),
                                 Line2D([], [], color=INK, lw=1.2, ls=(0, (1, 1.5)), label='vColl (점선)')],
                        loc='best', fontsize=8)
        ax_f.set_title(f'seed {row}' if layout == 'seed' else f'시드 평균 (n={len(seeds)}; 얇은 선 = 시드별)')
        ax_z.set_title(f'분기 뒤 확대 ({zoom_from:.1f}M–{xmax:.2f}M)')
        ax_r.set_title('도착률 goal · 선박충돌 vColl (MA, %)')
        ax_f.set_ylabel('에피소드 반환 평균')
        ax_r.set_ylabel('%')
        if i == len(rows) - 1:
            for ax in axes[i]:
                ax.set_xlabel('결정 (M)')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='run CSV 폴더(예: Python/_repro_out_t)')
    ap.add_argument('--also_dir', action='append', default=[], help='없는 run 을 찾아볼 추가 폴더(반복 가능)')
    ap.add_argument('--prefix', default='t_')
    ap.add_argument('--arms', default='off,a6,a8,a2,a4,offb')
    ap.add_argument('--seeds', default='43,44,45')
    ap.add_argument('--dim', type=int, default=6, help='trunk 파일 이름의 d<dim>')
    ap.add_argument('--out', required=True, help='그림 경로(.png 또는 .pdf; 둘 다 저장)')
    ap.add_argument('--win', type=int, default=MA_WIN, help='중심 이동평균 창(유효 update 수)')
    ap.add_argument('--branch', type=int, default=BRANCH_AT, help='분기점(결정)')
    ap.add_argument('--total', type=int, default=TOTAL_DEC, help='x 축 끝(결정)')
    ap.add_argument('--zoom_from', type=float, default=9.0, help='확대 패널 시작(M 결정)')
    ap.add_argument('--layout', choices=('seed', 'mean'), default='seed')
    ap.add_argument('--ref', default='off', help='표의 기준 팔')
    ap.add_argument('--noise', default='offb', help='재분기 잡음 기준 팔(끝 차이 N)')
    ap.add_argument('--rate_unit', choices=('pct', 'auto', 'frac'), default='pct',
                    help='goal/vColl/oColl/TO 열의 단위. 학습기(vessel_gym_train.py _ep.csv)는 % 로 씀 = 기본; '
                         'auto = 전부 <= 1.0 이면 frac 으로 보고 ×100')
    args = ap.parse_args(argv)
    tag = 'plot_ep_return'
    note = lambda m: print(f'[{tag}] {m}')  # noqa: E731

    setup_style()
    dirs = [args.dir] + list(args.also_dir)
    arms, seeds = split_list(args.arms), [int(s) for s in split_list(args.seeds)]
    note(f'dir={dirs} prefix={args.prefix} arms={arms} seeds={seeds} win={args.win}(중심, 유효 행 기준, 양 끝 창 축소) '
         f'분기={args.branch:,} 기준={args.ref} 잡음={args.noise}')
    series, trunks, missing = collect(dirs, args.prefix, arms, seeds, load_ep, EP_SUFFIX, args.dim, note)
    if not series:
        note(f'{args.dir}: {args.prefix}*{EP_SUFFIX} 없음 — 에피소드 반환 그림 생략 '
             f'(이 파일을 쓰기 전 배치이거나 아직 학습 전). rc 0')
        return 0
    for m in missing:
        note(f'없음: {m}')
    unit = detect_rate_unit(series) if args.rate_unit == 'auto' else args.rate_unit
    if unit == 'frac':
        for s in series.values():
            for k in RATE_COLS:
                s[k] = s[k] * 100.0
    note(f'goal/vColl/oColl/TO 단위 = {unit}{" (자동 판정)" if args.rate_unit == "auto" else ""} → % 로 그림')
    n_bad = check_trunk_match(series, trunks, seeds, arms, args.branch, note)
    full_arm = full_arm_for(series, seeds, arms, args.ref)      # trunk 는 기준 팔이 한 번만 그림(plot_reward_v3 와 같음)
    cut_at = None if n_bad else args.branch
    if n_bad:
        note('분기 전 구간 불일치 → 모든 팔을 0 부터 전체 구간으로 그림')
    mas = {k: centred_ma(s['y'], args.win) for k, s in series.items()}
    mas_goal = {k: centred_ma(s['goal'], args.win) for k, s in series.items()}
    mas_vcoll = {k: centred_ma(s['vColl'], args.win) for k, s in series.items()}
    rows = compute_table(series, mas, seeds, arms, args.ref, args.branch)
    print_table(rows, seeds, arms, args.ref, args.noise, 2, tag, '에피소드 반환 MA')
    print(f'[{tag}] 분기 뒤 goal / vColl MA 평균(%):')
    for seed in seeds:
        parts = []
        for arm in arms:
            k = (arm, seed)
            if k not in series:
                continue
            post = series[k]['step'] > args.branch
            if post.any():
                parts.append(f'{arm} {np.nanmean(mas_goal[k][post]):.1f}/{np.nanmean(mas_vcoll[k][post]):.1f}')
        print(f'{seed:>5}  ' + '  '.join(parts))
    title = (f'{args.prefix} 배치 에피소드 반환 — raw(옅음) + 중심 이동평균 {args.win} update · '
             f'점선 = 분기 {args.branch / 1e6:.2f}M')
    fig = make_figure(series, mas, mas_goal, mas_vcoll, seeds, arms, args.prefix, args.layout, args.branch,
                      args.total, args.zoom_from, title, cut_at=cut_at, full_arm=full_arm)
    save_fig(fig, args.out, note)
    return 0


if __name__ == '__main__':
    sys.exit(main())
