#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-decision reward curves of a trunk/branch batch — spec 2026-09-30 §7 '목표 2(그림)' ① and ③.

Input (``--dir``, then ``--also_dir`` fallbacks): the curve CSVs written by ``vessel_gym_train.py`` —
``<prefix><arm>_s<seed>.csv`` with columns ``step,raw_reward,ema_reward``, one row per PPO update
(65,536 decisions).  ``raw_reward`` is the mean per-decision reward of that update's rollout.  A branch
run's file carries the trunk rows (1..138, up to 9,043,968) followed by its own rows; the trunk file
``<prefix>trunk_d<dim>_s<seed>.csv`` cross-checks that prefix and fills it in when a file starts after
the branch point.

Curve: raw values (faint) + a CENTRED moving average over ``--win`` updates (default 20; the window
shrinks at both ends = pandas ``rolling(win, center=True, min_periods=1)``).  The ``ema_reward`` column is
not used (alpha 0.02 lags ~50 updates; spec §7 keeps it out of the verdict).

Layout: one row per seed (``--layout seed``) or one row with per-seed thin lines + seed mean
(``--layout mean``).  Left column = 0..total decisions with one shared y-range that always contains 0 and
the first-update value; right column = zoom of the post-branch window.

stdout table (per arm x seed, plus a seed mean): post-branch mean of the MA, share of post-branch updates
whose arm MA >= reference-arm MA (common steps only), end gap arm - ref at the last common update, and
the ``offb - off`` end gap as the re-branch noise reference N when that arm exists.  The §7 rule
(share >= 90 % and end gap > |N|, 3/3) is tallied for information only — the verdict is the author's.

matplotlib lives in the base anaconda python, not mltest::

    C:/Users/OSH/anaconda3/python.exe plot_reward_v3.py --dir <out dir> --prefix t_ \\
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
from matplotlib import font_manager                  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import paper_style as ps                             # noqa: E402

BRANCH_AT = 9_043_968       # §8-1 분기점(결정) = trunk 마지막 update(행 138)
TOTAL_DEC = 16_056_320      # 학습 끝(결정) = config.YUGIOH_ARGS --steps
UPDATE_DEC = 65_536         # update 당 결정 수(128 env × 16 척 × rollout 32) = 곡선 CSV 한 행
MA_WIN = 20                 # 중심 이동평균 창(update 수, 스펙 §7 목표 2)
FIRST_NOTE = '첫 update: 32결정, 사건 0'   # 첫 행 주석 — 스펙 §1: +0.16 은 첫 update(32결정, 사건 0) 값

INK = '#0b0b0b'             # 글자색(값·라벨은 항상 잉크색, 팔 색은 선에만)
MUTED = '#898781'
# 팔 색은 실행 이름(entity)에 고정 — 팔 목록·순서가 바뀌어도 같은 팔은 같은 색(dataviz 규약).
# 처치 팔 4색은 dataviz validate_palette.js 통과(흰 바탕, 인접·전쌍 모두 PASS; aqua 는 대비 경고 → 끝값 직접 라벨로 보완).
ARM_COLOR = {
    'off': '#3d3d3d',                 # 기준선(통신 OFF) = 진회색
    'offb': '#9a9a9a',                # OFF 재분기(잡음 기준) = 연회색 + 점선(색 외 2차 부호)
    'a6': '#2a78d6', 'a8': '#eb6834', 'a2': '#1baf7a', 'a4': '#4a3aa7',   # t_ 처치 팔
    'z6': '#2a78d6', 'z12': '#eb6834',                                     # s_ 팔(다른 배치라 색 재사용)
}
ARM_LS = {'offb': (0, (4, 2))}
_SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#e34948', '#eda100']   # 모르는 팔: 남는 색을 --arms 순서로


# ----------------------------------------------------------------------------- 공용(plot_ep_return 도 씀)
def split_list(s):
    """'a,b, c' -> ['a', 'b', 'c']."""
    return [t.strip() for t in s.split(',') if t.strip()]


def setup_style():
    """paper_style + Korean font (Malgun Gothic when installed, else a fallback) + ASCII minus."""
    ps.apply()
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ('Malgun Gothic', 'AppleGothic', 'NanumGothic'):
        if cand in names:
            plt.rcParams['font.family'] = [cand, 'DejaVu Sans']
            break
    plt.rcParams['axes.unicode_minus'] = False       # Malgun 에 유니코드 마이너스 글리프 없음
    plt.rcParams['pdf.fonttype'] = 42                # PDF 에 TrueType 그대로(한글 텍스트 검색·복사 가능)


def centred_ma(y, win):
    """Centred moving average over ``win`` samples, window shrunk at both ends
    (pandas ``rolling(win, center=True, min_periods=1)``); an even ``win`` spans [i-win//2, i+win-win//2)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy()
    half = win // 2
    cs = np.concatenate([[0.0], np.cumsum(y)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n, idx + (win - half))
    return (cs[hi] - cs[lo]) / (hi - lo)


def find_run_file(name, dirs):
    """First existing ``<dir>/<name>`` over ``dirs`` (search order = --dir, then --also_dir), else None."""
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def load_curve(path, note):
    """``step,raw_reward,ema_reward`` -> {'step': int64[], 'y': float64[]} sorted by step.
    A repeated step (crash + resume) keeps its last row; the EMA column is ignored."""
    rows, n_all, n_ok = {}, 0, 0
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            n_all += 1
            try:
                rows[int(float(r['step']))] = float(r['raw_reward'])
                n_ok += 1
            except (KeyError, TypeError, ValueError):
                continue
    n_dup, n_bad = n_ok - len(rows), n_all - n_ok
    if n_dup or n_bad:
        note(f'{os.path.basename(path)}: 행 {n_all} 중 유효 {len(rows)} (깨진 행 {n_bad} 버림, 중복 step {n_dup} 은 마지막 행)')
    steps = np.array(sorted(rows), dtype=np.int64)
    return {'step': steps, 'y': np.array([rows[s] for s in steps], dtype=float)}


def with_trunk_prefix(s, trunk, name, note):
    """Prepend the trunk rows that precede the run's first row (a branch file that was not seeded with the
    trunk curve).  Every array in ``s`` that is aligned with 'step' gets the trunk's matching array or nan."""
    if trunk is None or len(s['step']) == 0 or s['step'][0] <= UPDATE_DEC:
        return s
    m = trunk['step'] < s['step'][0]
    if not m.any():
        return s
    note(f'{name}: 첫 행 step {s["step"][0]:,} — trunk 행 {int(m.sum())}개를 앞에 붙임')
    out = {}
    for k, v in s.items():
        if k in trunk and len(trunk[k]) == len(trunk['step']):
            out[k] = np.concatenate([trunk[k][m], v])
        else:
            out[k] = np.concatenate([np.full(int(m.sum()), np.nan), v])
    return out


def collect(dirs, prefix, arms, seeds, loader, suffix, dim, note):
    """Load ``<prefix><arm>_s<seed><suffix>`` for every arm x seed and the trunk ``<prefix>trunk_d<dim>_s<seed><suffix>``.
    Returns (series {(arm, seed): dict}, trunks {seed: dict}, missing [file names])."""
    series, trunks, missing = {}, {}, []
    for seed in seeds:
        tp = find_run_file(f'{prefix}trunk_d{dim}_s{seed}{suffix}', dirs)
        if tp is not None:
            t = loader(tp, note)
            if len(t['step']):
                trunks[seed] = t
        for arm in arms:
            name = f'{prefix}{arm}_s{seed}{suffix}'
            p = find_run_file(name, dirs)
            if p is None:
                missing.append(name)
                continue
            s = loader(p, note)
            if not len(s['step']):
                note(f'{name}: 빈 파일 — 건너뜀')
                missing.append(name)
                continue
            series[(arm, seed)] = with_trunk_prefix(s, trunks.get(seed), name, note)
    return series, trunks, missing


def check_trunk_match(series, trunks, seeds, arms, branch, note):
    """§8-1: every arm of a seed must carry the same curve up to the branch point (trunk file, else the first
    arm present).  A mismatch is reported, not repaired — the figure is still drawn.  Returns the mismatch count."""
    n_bad = 0
    for seed in seeds:
        base, base_name = trunks.get(seed), 'trunk'
        if base is None:
            for arm in arms:
                if (arm, seed) in series:
                    base, base_name = series[(arm, seed)], arm
                    break
        if base is None:
            continue
        bm = base['step'] <= branch
        bmap = dict(zip(base['step'][bm].tolist(), base['y'][bm].tolist()))
        for arm in arms:
            s = series.get((arm, seed))
            if s is None or arm == base_name:
                continue
            m = s['step'] <= branch
            bad = sum(1 for x, v in zip(s['step'][m].tolist(), s['y'][m].tolist()) if bmap.get(x) != v)
            if int(m.sum()) != len(bmap) or bad:
                n_bad += 1
                note(f'★s{seed} {arm}: 분기 전 구간이 {base_name} 와 다름(행 {int(m.sum())} vs {len(bmap)}, '
                     f'값 불일치 {bad}) — §8-1 분기 규약 확인')
    return n_bad


def arm_styles(arms):
    """{arm: (color, linestyle)} — known arms keep their fixed color; unknown arms take the unused slots in --arms order."""
    used = {ARM_COLOR[a] for a in arms if a in ARM_COLOR}
    free = [c for c in _SLOTS if c not in used]
    out = {}
    for a in arms:
        if a in ARM_COLOR:
            out[a] = (ARM_COLOR[a], ARM_LS.get(a, '-'))
        else:
            out[a] = (free.pop(0) if free else '#777777', ARM_LS.get(a, '-'))
    return out


def compute_table(series, mas, seeds, arms, ref, branch):
    """One row per (arm, seed): post-branch MA mean; share of post-branch updates with arm MA >= ref MA and the
    end gap arm - ref, both on the steps the two runs have in common (in-progress runs end early)."""
    rows = []
    for seed in seeds:
        r = series.get((ref, seed))
        ref_map = dict(zip(r['step'].tolist(), mas[(ref, seed)].tolist())) if r is not None else None
        for arm in arms:
            s = series.get((arm, seed))
            if s is None:
                rows.append(dict(seed=seed, arm=arm, missing=True))
                continue
            st, ma = s['step'], mas[(arm, seed)]
            post = st > branch
            row = dict(seed=seed, arm=arm, missing=False, n_post=int(post.sum()),
                       post_mean=float(ma[post].mean()) if post.any() else float('nan'),
                       last_step=int(st[-1]), end_val=float(ma[-1]),
                       share=float('nan'), n_common=0, gap=float('nan'), gap_step=None)
            if ref_map is not None and arm != ref:
                cs = [(int(x), float(v)) for x, v in zip(st[post], ma[post]) if int(x) in ref_map]
                if cs:
                    d = np.array([v - ref_map[x] for x, v in cs])
                    row.update(n_common=len(cs), share=float((d >= 0).mean()), gap=float(d[-1]), gap_step=cs[-1][0])
            rows.append(row)
    return rows


def _fmt(v, prec, sign=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '-'
    return f'{v:+.{prec}f}' if sign else f'{v:.{prec}f}'


def print_table(rows, seeds, arms, ref, noise_arm, prec, tag, what):
    """Print the per-seed table, the seed mean, the offb-off noise gaps and the §7 tally."""
    print(f'[{tag}] 표: {what} — 분기 뒤 = step > 분기점, share/gap 은 {ref} 와 겹치는 update 에서')
    print(f'{"seed":>5} {"arm":<6} {"n_post":>6} {"post_mean_MA":>13} {"share>=" + ref:>11} {"n_cmp":>5} '
          f'{"end_gap-" + ref:>12} {"gap@M":>7} {"end_MA":>10} {"last_M":>7}')
    by = {(r['seed'], r['arm']): r for r in rows}
    for seed in seeds:
        for arm in arms:
            r = by[(seed, arm)]
            if r['missing']:
                print(f'{seed:>5} {arm:<6} {"없음":>6}')
                continue
            gap_m = f'{r["gap_step"] / 1e6:.2f}' if r['gap_step'] else '-'
            print(f'{seed:>5} {arm:<6} {r["n_post"]:>6} {_fmt(r["post_mean"], prec):>13} '
                  f'{_fmt(r["share"], 2, False):>11} {r["n_common"]:>5} {_fmt(r["gap"], prec):>12} '
                  f'{gap_m:>7} {_fmt(r["end_val"], prec):>10} {r["last_step"] / 1e6:>7.2f}')
    for arm in arms:
        rs = [by[(sd, arm)] for sd in seeds if not by[(sd, arm)]['missing']]
        if not rs:
            continue
        mean = lambda k: float(np.nanmean([r[k] for r in rs])) if any(not math.isnan(r[k]) for r in rs) else float('nan')  # noqa: E731
        print(f'{"mean":>5} {arm:<6} {"n=" + str(len(rs)):>6} {_fmt(mean("post_mean"), prec):>13} '
              f'{_fmt(mean("share"), 2, False):>11} {"":>5} {_fmt(mean("gap"), prec):>12}')
    noise = {}
    for seed in seeds:
        r = by.get((seed, noise_arm))
        if r is not None and not r['missing'] and not math.isnan(r['gap']):
            noise[seed] = r['gap']
            print(f'[{tag}] N s{seed}: ({noise_arm}-{ref}) 끝 차이 {_fmt(r["gap"], prec)} → |N| {abs(r["gap"]):.{prec}f}')
    if not noise:
        print(f'[{tag}] {noise_arm} 없음 → 잡음 N(재분기 차이) 미산출, 끝 차이 > N 은 판정 못 함')
    for arm in arms:
        if arm in (ref, noise_arm):
            continue
        sh, gp = [], []
        for seed in seeds:
            r = by[(seed, arm)]
            if r['missing'] or math.isnan(r['share']):
                continue
            sh.append((seed, r['share'] >= 0.9, r['share']))
            if seed in noise:
                gp.append((seed, r['gap'] > abs(noise[seed]), r['gap']))
        if not sh:
            continue
        s_txt = ' '.join(f's{sd}:{"O" if ok else "X"}({v:.2f})' for sd, ok, v in sh)
        g_txt = ' '.join(f's{sd}:{"O" if ok else "X"}({_fmt(v, prec)})' for sd, ok, v in gp) if gp else 'N 없음'
        print(f'[{tag}] §7 목표 2 규칙({what}) {arm}: MA>={ref} 90%↑ {sum(ok for _, ok, _ in sh)}/{len(sh)} [{s_txt}] · '
              f'끝 차이>|N| {sum(ok for _, ok, _ in gp)}/{len(gp)} [{g_txt}]  (판정은 저자)')


def full_arm_for(series, seeds, arms, ref):
    """The arm that draws the shared trunk (rows up to the branch) in full: ``ref`` when present, else the first
    arm present.  All other arms start at the branch point, so the identical trunk rows are drawn once."""
    if any((ref, sd) in series for sd in seeds):
        return ref
    for arm in arms:
        if any((arm, sd) in series for sd in seeds):
            return arm
    return None


def _mask(s, cut):
    return np.ones(len(s['step']), dtype=bool) if cut is None else (s['step'] >= cut)


def build_items_seed(series, mas, seed, arms, prefix, styles, raw=True, ls=None, label=True, cut_at=None,
                     full_arm=None):
    """Draw list for one seed row: per arm raw (faint) + MA (labelled).  With ``cut_at`` every arm except
    ``full_arm`` is drawn from that step on (the MA itself is still computed on the whole series)."""
    items = []
    for arm in arms:
        s = series.get((arm, seed))
        if s is None:
            continue
        c, arm_ls = styles[arm]
        m = _mask(s, None if arm == full_arm else cut_at)
        if not m.any():
            continue
        items.append(dict(x=s['step'][m] / 1e6, raw=s['y'][m] if raw else None, ma=mas[(arm, seed)][m], color=c,
                          ls=ls or arm_ls, label=f'{prefix}{arm}' if label else None, lw=1.4,
                          raw_alpha=0.15, ma_alpha=1.0, endlabel=True))
    return items


def build_items_mean(series, mas, seeds, arms, prefix, styles, raw=True, ls=None, label=True, cut_at=None,
                     full_arm=None):
    """Draw list for the seed-mean row: per seed thin raw + thin MA, plus the mean MA on the common steps.
    ``cut_at``/``full_arm`` as in build_items_seed."""
    items = []
    for arm in arms:
        c, arm_ls = styles[arm]
        have = [(sd, series[(arm, sd)], mas[(arm, sd)]) for sd in seeds if (arm, sd) in series]
        if not have:
            continue
        cut = None if arm == full_arm else cut_at
        for _, s, ma in have:
            m = _mask(s, cut)
            if m.any():
                items.append(dict(x=s['step'][m] / 1e6, raw=s['y'][m] if raw else None, ma=ma[m], color=c,
                                  ls=ls or arm_ls, label=None, lw=0.7, raw_alpha=0.08, ma_alpha=0.45,
                                  endlabel=False))
        common = have[0][1]['step']
        for _, s, _ in have[1:]:
            common = np.intersect1d(common, s['step'])
        if cut is not None:
            common = common[common >= cut]
        if len(common):
            stack = []
            for _, s, ma in have:
                m = dict(zip(s['step'].tolist(), ma.tolist()))
                stack.append([m[x] for x in common.tolist()])
            items.append(dict(x=common / 1e6, raw=None, ma=np.mean(stack, axis=0), color=c, ls=ls or arm_ls,
                              label=f'{prefix}{arm} (n={len(have)})' if label else None, lw=1.8,
                              raw_alpha=0.0, ma_alpha=1.0, endlabel=True))
    return items


def full_ylim(series, key='y'):
    """Shared y-range of the full-run column: contains 0, every run's first value and all raw values (+5 % pad)."""
    vals = [s[key][np.isfinite(s[key])] for s in series.values()]
    vals = [v for v in vals if len(v)]
    lo = min([0.0] + [float(v[0]) for v in vals] + [float(v.min()) for v in vals])
    hi = max([0.0] + [float(v[0]) for v in vals] + [float(v.max()) for v in vals])
    pad = 0.05 * ((hi - lo) or 1.0)
    return lo - pad, hi + pad


def zoom_ylim(series, mas, x0):
    """y-range of the zoom column from the MA values at step >= x0 (all runs), padded by 30 % of their spread."""
    vals = []
    for k, s in series.items():
        m = s['step'] >= x0
        if m.any():
            vals.append(mas[k][m])
    if not vals:
        return -1.0, 1.0
    v = np.concatenate(vals)
    lo, hi = float(v.min()), float(v.max())
    pad = 0.3 * ((hi - lo) or 1.0)
    return lo - pad, hi + pad


def _end_labels(ax, items, xlim, ylim, fmt):
    """Direct labels of the last MA value at the right edge, pushed apart so they do not overlap."""
    labs = sorted(((float(it['ma'][-1]), float(it['x'][-1]), it['color']) for it in items if len(it['ma'])),
                  key=lambda t: t[0])
    if not labs:
        return
    yr = ylim[1] - ylim[0]
    gap = 0.06 * yr
    ys = []
    for y, _, _ in labs:
        ys.append(y if not ys else max(y, ys[-1] + gap))
    over = ys[-1] - (ylim[1] - 0.03 * yr)
    if over > 0:                                     # 위로 밀린 만큼 되돌림
        ys = [v - over for v in ys]
    for (y, x, c), yy in zip(labs, ys):
        ax.annotate(fmt.format(y), xy=(x, y), xytext=(xlim[1] - 0.01 * (xlim[1] - xlim[0]), yy),
                    textcoords='data', ha='right', va='center', fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle='-', color=c, lw=0.6, alpha=0.8, shrinkA=0, shrinkB=1.5), zorder=5)


def draw_curves(ax, items, xlim, ylim, branch, first_pt=None, end_labels=False, legend=False,
                branch_label=True, end_fmt='{:.3f}'):
    """Raw (faint) under MA lines, dotted branch line, optional first-point note / end labels / legend."""
    for it in items:
        if it['raw'] is not None and it['raw_alpha'] > 0:
            ax.plot(it['x'], it['raw'], color=it['color'], lw=0.5, alpha=it['raw_alpha'], zorder=1)
    for it in items:
        ax.plot(it['x'], it['ma'], color=it['color'], ls=it['ls'], lw=it['lw'], alpha=it['ma_alpha'],
                label=it['label'], zorder=3)
    ax.axvline(branch / 1e6, color='#555555', ls=(0, (1, 2.5)), lw=0.9, zorder=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    xr, yr = xlim[1] - xlim[0], ylim[1] - ylim[0]
    if branch_label:
        ax.text(branch / 1e6 + 0.012 * xr, ylim[0] + 0.03 * yr, f'분기 {branch / 1e6:.2f}M', color=MUTED,
                fontsize=8, ha='left', va='bottom', zorder=4)
    if first_pt is not None:
        ax.annotate(FIRST_NOTE, xy=first_pt, xytext=(xlim[0] + 0.11 * xr, first_pt[1]), fontsize=8,
                    color=INK, ha='left', va='center', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85),
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7, shrinkA=0, shrinkB=2))
    if end_labels:
        _end_labels(ax, [it for it in items if it['endlabel']], xlim, ylim, end_fmt)
    if legend:
        ax.legend(loc='best', fontsize=8)


def first_point(series, seed=None):
    """(x[M], y) of the earliest row of the seed (or of all runs) for the '첫 update' note — None unless that row
    really is update 1 (step 65,536; an _ep.csv whose first valid row comes later gets no note)."""
    cands = [s for (a, sd), s in series.items() if (seed is None or sd == seed) and len(s['step'])]
    if not cands:
        return None
    s = min(cands, key=lambda s: int(s['step'][0]))
    if int(s['step'][0]) != UPDATE_DEC:
        return None
    return float(s['step'][0]) / 1e6, float(s['y'][0])


def save_fig(fig, out, note):
    """Write <out>.png and <out>.pdf (whatever extension --out carries; the author opens PDFs)."""
    base, ext = os.path.splitext(out)
    if ext.lower() not in ('.png', '.pdf'):
        base = out
    d = os.path.dirname(os.path.abspath(base))
    os.makedirs(d, exist_ok=True)
    for e in ('png', 'pdf'):
        fig.savefig(f'{base}.{e}', bbox_inches='tight')
        note(f'저장 {base}.{e}')
    plt.close(fig)


# ----------------------------------------------------------------------------- 이 스크립트 고유
def make_figure(series, mas, seeds, arms, prefix, layout, branch, total, win, zoom_from, ylabel, title,
                cut_at=None, full_arm=None):
    styles = arm_styles(arms)
    rows = list(seeds) if layout == 'seed' else ['mean']
    fig, axes = plt.subplots(len(rows), 2, figsize=(11.6, 3.15 * len(rows) + 0.7), squeeze=False,
                             gridspec_kw={'width_ratios': [1.35, 1]})
    xmax = max(total, max(int(s['step'][-1]) for s in series.values())) / 1e6
    yl_full = full_ylim(series)
    yl_zoom = zoom_ylim(series, mas, zoom_from * 1e6)
    for i, row in enumerate(rows):
        if layout == 'seed':
            items = build_items_seed(series, mas, row, arms, prefix, styles, cut_at=cut_at, full_arm=full_arm)
            fp = first_point(series, row)
        else:
            items = build_items_mean(series, mas, seeds, arms, prefix, styles, cut_at=cut_at, full_arm=full_arm)
            fp = first_point(series)
        ax_f, ax_z = axes[i][0], axes[i][1]
        draw_curves(ax_f, items, (-0.15, xmax + 0.15), yl_full, branch, first_pt=fp, legend=(i == 0))
        draw_curves(ax_z, items, (zoom_from, xmax + 0.7), yl_zoom, branch, end_labels=True, branch_label=False)
        ax_f.set_title(f'seed {row}' if layout == 'seed' else f'시드 평균 (n={len(seeds)}; 얇은 선 = 시드별)')
        ax_z.set_title(f'분기 뒤 확대 ({zoom_from:.1f}M–{xmax:.2f}M)')
        ax_f.set_ylabel(ylabel)
        if i == len(rows) - 1:
            ax_f.set_xlabel('결정 (M)')
            ax_z.set_xlabel('결정 (M)')
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
    ap.add_argument('--win', type=int, default=MA_WIN, help='중심 이동평균 창(update 수)')
    ap.add_argument('--branch', type=int, default=BRANCH_AT, help='분기점(결정)')
    ap.add_argument('--total', type=int, default=TOTAL_DEC, help='x 축 끝(결정)')
    ap.add_argument('--zoom_from', type=float, default=9.0, help='확대 패널 시작(M 결정)')
    ap.add_argument('--layout', choices=('seed', 'mean'), default='seed')
    ap.add_argument('--ref', default='off', help='표의 기준 팔')
    ap.add_argument('--noise', default='offb', help='재분기 잡음 기준 팔(끝 차이 N)')
    args = ap.parse_args(argv)
    tag = 'plot_reward_v3'
    note = lambda m: print(f'[{tag}] {m}')  # noqa: E731

    setup_style()
    dirs = [args.dir] + list(args.also_dir)
    arms, seeds = split_list(args.arms), [int(s) for s in split_list(args.seeds)]
    note(f'dir={dirs} prefix={args.prefix} arms={arms} seeds={seeds} win={args.win}(중심, 양 끝 창 축소) '
         f'분기={args.branch:,} 기준={args.ref} 잡음={args.noise}')
    series, trunks, missing = collect(dirs, args.prefix, arms, seeds, load_curve, '.csv', args.dim, note)
    for m in missing:
        note(f'없음: {m}')
    if not series:
        note('★데이터 없음 — 그림·표 생략')
        return 1
    n_bad = check_trunk_match(series, trunks, seeds, arms, args.branch, note)
    # 분기 전 구간이 같으면 trunk 는 기준 팔이 한 번만 그리고 나머지 팔은 분기점부터; 다르면 전부 전체 구간으로 그려 드러냄
    full_arm = full_arm_for(series, seeds, arms, args.ref)
    cut_at = None if n_bad else args.branch
    if n_bad:
        note('분기 전 구간 불일치 → 모든 팔을 0 부터 전체 구간으로 그림')
    mas = {k: centred_ma(s['y'], args.win) for k, s in series.items()}
    rows = compute_table(series, mas, seeds, arms, args.ref, args.branch)
    print_table(rows, seeds, arms, args.ref, args.noise, 4, tag, '결정당 보상 MA')
    title = (f'{args.prefix} 배치 결정당 보상 — raw(옅음) + 중심 이동평균 {args.win} update · '
             f'점선 = 분기 {args.branch / 1e6:.2f}M (EMA 미사용)')
    fig = make_figure(series, mas, seeds, arms, args.prefix, args.layout, args.branch, args.total, args.win,
                      args.zoom_from, '결정당 보상', title, cut_at=cut_at, full_arm=full_arm)
    save_fig(fig, args.out, note)
    return 0


if __name__ == '__main__':
    sys.exit(main())
