"""Fig. 9 — 전역 A* 항로와 국지 회피 거동 (2단 축척).

구성
  (a) 전역: Natural Earth 1:10m 해안선 위에서 실제로 탐색한 A* 항로.
      격자 0.06°, 해안 여유 12 km. 경로장·우회율은 계산된 값.
  (b) 국지: 학습된 정책의 실측 항적. 자체 축척막대를 붙여 (a)와 축척이 다름을 명시.

두 패널의 축척이 다르다는 점이 이 그림의 핵심 — 전역 계획은 1,000 km 규모,
회피 기동은 100 m 규모에서 일어난다. 축척막대를 각각 두는 이유가 그것.
"""
import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, ConnectionPatch
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '_fig9_candidate')
DATA = os.path.join(OUT, 'data')

SEA, SEA2 = '#0b1b2b', '#12283d'
LAND, LAND_E = '#e8e6df', '#b9b5a6'
ROUTE = '#e03131'
WPT = '#0b3d91'
TRK_A, TRK_B = '#0b6e4f', '#b3541e'
INK, MUTED = '#12181f', '#5a6b7a'


def km_per_deg(lat):
    return 111.32, 111.32 * math.cos(math.radians(lat))


def proj(pts, lat0):
    k = math.cos(math.radians(lat0))
    a = np.asarray(pts, dtype=float)
    return np.column_stack([a[:, 0] * k, a[:, 1]])


def load_tracks(path, max_n, min_len=40):
    """실측 항적 로드. corridor_run.py(.pt) 또는 traj_capture.py(.npz) 둘 다 지원."""
    if path.endswith('.pt'):
        import torch
        d = torch.load(path, map_location='cpu')
        P, O = d['P'], d['OUT']
        T, E, N, _ = P.shape
        segs = []
        for e in range(E):
            for n in range(N):
                oc = O[:, e, n]
                term = torch.nonzero(oc != 0).flatten().tolist()
                s0 = 0
                for t in term:
                    if t - s0 >= min_len:
                        segs.append(np.asarray(P[s0:t + 1, e, n].tolist(), dtype=float))
                    s0 = t + 1
        meta = dict(kind='corridor', length=float(d['length']), width=float(d['width']))
    else:
        d = np.load(path)
        P, eps = d['P'], d['eps']
        segs = []
        for e, n, t0, t1, oc in eps:
            if t1 - t0 >= min_len:
                segs.append(np.asarray(P[t0:t1 + 1, e, n], dtype=float))
        meta = dict(kind='arena', length=None, width=None)
    if len(segs) > max_n:
        idx = np.linspace(0, len(segs) - 1, max_n).astype(int)
        segs = [segs[i] for i in idx]
    return segs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', default=os.path.join(DATA, 'route_astar.json'))
    ap.add_argument('--coast', default=os.path.join(DATA, 'coast_ea.json'))
    ap.add_argument('--tracks', default='')
    ap.add_argument('--max_tracks', type=int, default=400)
    ap.add_argument('--out', default=os.path.join(OUT, 'fig9_paper'))
    args = ap.parse_args()

    R = json.load(open(args.route))
    C = json.load(open(args.coast))['polys']
    path = np.asarray(R['path'], dtype=float)
    lat0 = float(path[:, 1].mean())

    tracks_path = args.tracks
    if not tracks_path:
        for cand in (os.path.join(DATA, 'corridor_traj.pt'),
                     os.path.join(os.path.dirname(HERE), 'runs', 'traj.npz')):
            if os.path.exists(cand):
                tracks_path = cand
                break
    segs, tmeta = load_tracks(tracks_path, args.max_tracks) if tracks_path else ([], {})

    plt.rcParams.update({'font.size': 9.5, 'axes.linewidth': .7,
                         'font.family': 'DejaVu Sans'})
    fig = plt.figure(figsize=(11.0, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=.14)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ───────────────── (a) 전역 A* 항로 ─────────────────
    axA.set_facecolor(SEA)
    for poly in C:
        if len(poly) < 8:
            continue
        axA.add_patch(Polygon(proj(poly, lat0), closed=True, fc=LAND, ec=LAND_E,
                              lw=.5, zorder=2))
    # ── 격자 A* 경로 (계단) ──
    pp = proj(path, lat0)
    # ── vessel route 가 지나는 격자 셀을 사각형으로 ──
    cell = float(R['step_deg'])
    seen, cells = set(), []
    for a_, b_ in zip(path, path[1:]):
        n_ = max(2, int(max(abs(b_[0] - a_[0]), abs(b_[1] - a_[1])) / cell * 4))
        for t_ in range(n_ + 1):
            x_ = a_[0] + (b_[0] - a_[0]) * t_ / n_
            y_ = a_[1] + (b_[1] - a_[1]) * t_ / n_
            k_ = (round(x_ / cell), round(y_ / cell))
            if k_ not in seen:
                seen.add(k_)
                cells.append((k_[0] * cell, k_[1] * cell))
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle as _Rect
    kx = math.cos(math.radians(lat0)) * cell
    cc = proj(np.asarray(cells, dtype=float), lat0)
    axA.add_collection(PatchCollection(
        [_Rect((x - kx / 2, y - cell / 2), kx, cell) for x, y in cc],
        facecolor='#7fd4ff', edgecolor='#cdeeff', linewidths=.35,
        alpha=.85, zorder=4))
    axA.plot([], [], 's', ms=7, mfc='#7fd4ff', mec='#cdeeff',
             label=f'waypoint ({len(cells)})')
    axA.plot(pp[:, 0], pp[:, 1], color=ROUTE, lw=0.9, zorder=6,
             solid_capstyle='round', label='vessel route')
    for p, nm, dx, dy, ha in ((pp[0], 'Busan', 10, -14, 'left'),
                              (pp[-1], 'Kaohsiung', 12, 4, 'left')):
        axA.plot(p[0], p[1], 's', ms=6.5, mfc=WPT, mec='white', mew=1.2, zorder=7)
        axA.annotate(nm, (p[0], p[1]), xytext=(dx, dy), textcoords='offset points',
                     fontsize=10.5, color='white', fontweight='bold', ha=ha, zorder=9,
                     path_effects=[pe.withStroke(linewidth=2.6, foreground='#0b1b2b')])

    klat, klon = km_per_deg(lat0)
    sb = 300.0
    x0 = pp[:, 0].max() - sb / klat - 0.35
    y0 = float(path[:, 1].min()) - 0.85
    axA.plot([x0, x0 + sb / klat], [y0, y0], color='white', lw=2.0, zorder=8)
    for xx in (x0, x0 + sb / klat):
        axA.plot([xx, xx], [y0 - .18, y0 + .18], color='white', lw=2.0, zorder=8)
    axA.text(x0 + sb / klat / 2, y0 + .30, f'{sb:.0f} km', ha='center', va='bottom',
             fontsize=8.5, color='white', zorder=8)

    m = 1.4
    axA.set_xlim(pp[:, 0].min() - m * 1.6, pp[:, 0].max() + m * 1.9)
    axA.set_ylim(path[:, 1].min() - 1.9, path[:, 1].max() + 1.1)
    axA.set_aspect('equal')
    axA.set_xticks([]); axA.set_yticks([])
    axA.legend(loc='lower left', frameon=True, framealpha=.94, fontsize=7.8,
               edgecolor=LAND_E, borderpad=.6)
    axA.set_title('(a)  Global route from A* over the coastline',
                  loc='left', fontsize=11, pad=7)
    # ───────────────── (b) 국지 회피 항적 ─────────────────
    axB.set_facecolor('white')
    if segs:
        north = [s for s in segs if s[-1, 0] < s[0, 0]]
        south = [s for s in segs if s[-1, 0] >= s[0, 0]]
        for grp, col in ((north, TRK_A), (south, TRK_B)):
            if grp:
                axB.add_collection(LineCollection(grp, colors=col, linewidths=.55,
                                                  alpha=.28, zorder=3))
        allp = np.vstack(segs)
        pad = .06 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]))
        axB.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
        axB.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
        # 축척막대 (m)
        span = float(np.ptp(allp[:, 0]))
        sb2 = 10 ** math.floor(math.log10(span / 3))
        if span / sb2 > 6:
            sb2 *= 2
        xL = axB.get_xlim(); yL = axB.get_ylim()
        bx = xL[0] + .06 * (xL[1] - xL[0]); by = yL[0] + .11 * (yL[1] - yL[0])
        axB.plot([bx, bx + sb2], [by, by], color=INK, lw=2.0, zorder=8)
        for xx in (bx, bx + sb2):
            axB.plot([xx, xx], [by - .012 * (yL[1] - yL[0]), by + .012 * (yL[1] - yL[0])],
                     color=INK, lw=2.0, zorder=8)
        axB.text(bx + sb2 / 2, by + .022 * (yL[1] - yL[0]), f'{sb2:,.0f} m',
                 ha='center', va='bottom', fontsize=8.5, color=INK, zorder=8)
        axB.text(.985, .02, f'n = {len(segs):,} tracks', transform=axB.transAxes,
                 ha='right', va='bottom', fontsize=8.5, color=MUTED)
        from matplotlib.lines import Line2D
        axB.legend(handles=[Line2D([], [], color=TRK_A, lw=2, label='northbound'),
                            Line2D([], [], color=TRK_B, lw=2, label='southbound')],
                   loc='upper right', frameon=True, framealpha=.94, fontsize=8.5,
                   edgecolor=LAND_E, borderpad=.6)
    else:
        axB.text(.5, .5, 'no track data', ha='center', va='center',
                 transform=axB.transAxes, color=MUTED)
    axB.set_aspect('equal')
    axB.set_xticks([]); axB.set_yticks([])
    for s in axB.spines.values():
        s.set_color(LAND_E)
    if tmeta.get('kind') == 'corridor':
        ttl = '(b)  Simulated tracks in the traffic corridor'
        sub = (f"corridor {tmeta['length']:.0f} x {tmeta['width']:.0f} m · "
               f"head-on encounters")
    else:
        ttl = '(b)  Simulated tracks in the training environment'
        sub = 'square arena with nine circular obstacles'
    axB.set_title(ttl, loc='left', fontsize=11, pad=7)
    axB.text(.015, .022, sub, transform=axB.transAxes, fontsize=8,
             color=MUTED, va='bottom')

    fig.savefig(f'{args.out}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{args.out}.pdf', bbox_inches='tight', facecolor='white')
    print(f"(a) A* {R['length_km']:,.0f} km / 직선 {R['straight_km']:,.0f} km / "
          f"우회 {R['length_km']/R['straight_km']:.3f}, waypoint {len(path)}")
    print(f"(b) 항적 {len(segs):,}개  출처 {os.path.basename(tracks_path) if tracks_path else '-'}")
    print('저장:', args.out + '.png')


if __name__ == '__main__':
    main()
