"""curve_fig1.py — fixed-scene checkpoint reward curve + V-shape check M (spec docs/superpowers/specs/2026-10-07-pure-rl-fig1-design.md).

Reads curve_<PRE><run>_<x>M.txt (eval/eval_ckpt.py on one checkpoint: 256 env, burn-in 2400, 3000 decisions, drain 3000,
eval seed 999) written by _run_f.sh curve_all. <run> = trunk_d6_s<seed> | <arm>_s<seed>; <x> = decisions in millions.
Series per (arm, seed) = that seed's trunk points + the arm's branch points, sorted by x. Value = epReward (mean reward
of all episodes that ended in the window).
M (fixed before results): drop = max_t (max_{s<=t} c_s - c_t). Pass ('rising') if drop <= 0.1 * (max c - first c) and
max c > first c. Reported per seed with pass counts; never stops the batch. Curves are drawn in full (no cropping).

Usage: python curve_fig1.py <dir> [prefix n_]   -> <dir>/<prefix>curve.csv, <prefix>curve.pdf (if matplotlib), stdout table
"""
import glob
import os
import re
import sys

ARMS = ('off', 'comm', 'offb')
M_FRAC = 0.1


def m_check(c):
    """(drop, pass) for one series c (list of floats in x order)."""
    run_max, drop = c[0], 0.0
    for v in c:
        run_max = max(run_max, v)
        drop = max(drop, run_max - v)
    rise = max(c) - c[0]
    return drop, rise > 0 and drop <= M_FRAC * rise


def load(d, pre):
    pts = {}                     # (run, x) -> epReward
    for f in glob.glob(os.path.join(d, f'curve_{pre}*_*M.txt')):
        m = re.match(rf'curve_{re.escape(pre)}(.+)_([0-9.]+)M\.txt$', os.path.basename(f))
        e = re.search(r'epReward=\s*(-?[0-9.]+)', open(f, encoding='utf-8', errors='replace').read())
        if m and e:
            pts[(m.group(1), float(m.group(2)))] = float(e.group(1))
    return pts


def series(pts):
    seeds = sorted({int(r.split('_s')[-1]) for r, _ in pts})
    out = {}
    for arm in ARMS:
        for s in seeds:
            xs = sorted([(x, v) for (r, x), v in pts.items() if r in (f'trunk_d6_s{s}', f'{arm}_s{s}')])
            if any(r == f'{arm}_s{s}' for r, _ in pts):
                out[(arm, s)] = xs
    return out


def main():
    d, pre = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'n_')
    ser = series(load(d, pre))
    if not ser:
        print(f"★FAIL 곡선 점 없음 ({d}/curve_{pre}*)")
        return 1
    with open(os.path.join(d, f'{pre}curve.csv'), 'w', encoding='utf-8') as f:
        f.write('arm,seed,decisions_M,epReward\n')
        for (arm, s), xs in sorted(ser.items()):
            for x, v in xs:
                f.write(f'{arm},{s},{x:g},{v:.4f}\n')
    print(f"# 고정 장면 체크포인트 곡선 ({pre}) — epReward, M = 최대 하락 ≤ {M_FRAC:g} × (최고 − 첫 값)")
    print('| 팔 | 시드 | 점 수 | 첫 값 | 최고 | 끝 | 최대 하락 | M |')
    print('|---|---|---|---|---|---|---|---|')
    npass = {a: [0, 0] for a in ARMS}
    for (arm, s), xs in sorted(ser.items()):
        c = [v for _, v in xs]
        drop, ok = m_check(c)
        npass[arm][0] += ok
        npass[arm][1] += 1
        print(f"| {arm} | {s} | {len(c)} | {c[0]:.1f} | {max(c):.1f} | {c[-1]:.1f} | {drop:.1f} | {'우상향' if ok else '★불통과'} |")
    print('M 통과: ' + ' · '.join(f"{a} {p}/{n}" for a, (p, n) in npass.items() if n))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm, col in zip(ARMS, ('#4c72b0', '#c44e52', '#8c8c8c')):
            for k, ((a, s), xs) in enumerate(sorted((k, v) for k, v in ser.items() if k[0] == arm)):
                ax.plot([x for x, _ in xs], [v for _, v in xs], '-o', ms=3, color=col, alpha=0.8,
                        label=arm if k == 0 else None)
        ax.set_xlabel('decisions (M)')
        ax.set_ylabel('epReward (fixed scenes)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(d, f'{pre}curve.pdf'))
    except ImportError:
        print('(matplotlib 없음 — 그림 건너뜀, CSV 는 씀)')
    return 0


if __name__ == '__main__':
    assert m_check([1, 2, 3])[1] and not m_check([1, 5, 2, 6])[1] and not m_check([3, 2, 1])[1]
    sys.exit(main())
