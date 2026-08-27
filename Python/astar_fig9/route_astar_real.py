"""실제 해안선 위 A* 항로 계산 — 부산 ↔ 대만.

육지 폴리곤(Natural Earth 1:10m)을 점유격자로 굽고 8방향 A* 를 돌린다.
합성·근사 경로가 아니라 **이 격자 위에서 실제로 탐색한 결과**임.

  - 격자: 경위도 등간격. 거리 비용은 위도 보정한 실제 km 로 계산(등거리 근사).
  - 통항 여유(clearance): 해안에서 최소 `--clear` km 를 띄운다(팽창 = 격자 dilation).
  - 후처리: 가시선 string-pull 로 격자 계단을 직선 구간으로 편다.
"""
import argparse
import heapq
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '_fig9_candidate')
COAST = os.path.join(OUT, 'data', 'coast_ea.json')

R_EARTH = 6371.0
BUSAN = (129.05, 35.10)
KAOHSIUNG = (120.30, 22.62)
TAIPEI = (121.75, 25.05)


def km_per_deg(lat):
    return (111.32, 111.32 * math.cos(math.radians(lat)))


def build_grid(polys, win, step):
    """육지=1 격자. matplotlib Path.contains_points 로 벡터 래스터화(파이썬 루프 없음)."""
    from matplotlib.path import Path as MPath
    lon0, lon1, lat0, lat1 = win
    nx = int((lon1 - lon0) / step) + 1
    ny = int((lat1 - lat0) / step) + 1
    xs = lon0 + np.arange(nx) * step
    ys = lat0 + np.arange(ny) * step
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    land = np.zeros(pts.shape[0], dtype=bool)
    for poly in polys:
        P = np.asarray(poly, dtype=float)
        if P[:, 0].max() < lon0 or P[:, 0].min() > lon1 or \
           P[:, 1].max() < lat0 or P[:, 1].min() > lat1:
            continue
        todo = ~land
        if not todo.any():
            break
        hit = MPath(P).contains_points(pts[todo])
        idx = np.flatnonzero(todo)[hit]
        land[idx] = True
    return land.reshape(ny, nx), xs, ys


def dilate(mask, k):
    """k 셀만큼 팽창 (통항 여유)."""
    out = mask.copy()
    for _ in range(k):
        m = out
        nb = np.zeros_like(m)
        nb[1:, :] |= m[:-1, :]
        nb[:-1, :] |= m[1:, :]
        nb[:, 1:] |= m[:, :-1]
        nb[:, :-1] |= m[:, 1:]
        out = m | nb
    return out


def astar(block, xs, ys, start, goal):
    ny, nx = block.shape
    step_lon = xs[1] - xs[0]
    step_lat = ys[1] - ys[0]

    def idx(lon, lat):
        return (int(round((lon - xs[0]) / step_lon)),
                int(round((lat - ys[0]) / step_lat)))

    def free_near(i, j):
        if 0 <= i < nx and 0 <= j < ny and not block[j, i]:
            return i, j
        for r in range(1, 80):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    a, b = i + di, j + dj
                    if 0 <= a < nx and 0 <= b < ny and not block[b, a]:
                        return a, b
        return i, j

    si, sj = free_near(*idx(*start))
    gi, gj = free_near(*idx(*goal))
    kl_lat, _ = km_per_deg(0)

    def cost(i0, j0, i1, j1):
        latm = ys[(j0 + j1) // 2]
        klat, klon = km_per_deg(latm)
        dx = (i1 - i0) * step_lon * klon
        dy = (j1 - j0) * step_lat * klat
        return math.hypot(dx, dy)

    def h(i, j):
        return cost(i, j, gi, gj)

    s = sj * nx + si
    g = gj * nx + gi
    gs = {s: 0.0}
    came = {}
    pq = [(h(si, sj), s)]
    closed = np.zeros(ny * nx, dtype=bool)
    NB = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    while pq:
        _, k = heapq.heappop(pq)
        if closed[k]:
            continue
        if k == g:
            break
        closed[k] = True
        ki, kj = k % nx, k // nx
        base = gs[k]
        for di, dj in NB:
            a, b = ki + di, kj + dj
            if not (0 <= a < nx and 0 <= b < ny) or block[b, a]:
                continue
            if di and dj and (block[kj, a] or block[b, ki]):
                continue
            nk = b * nx + a
            ng = base + cost(ki, kj, a, b)
            if ng < gs.get(nk, 1e18) - 1e-9:
                gs[nk] = ng
                came[nk] = k
                heapq.heappush(pq, (ng + h(a, b), nk))
    if g not in gs:
        return None, None
    path, k = [], g
    while k != s:
        path.append((xs[k % nx], ys[k // nx]))
        k = came[k]
    path.append((xs[si], ys[sj]))
    path.reverse()
    return path, gs[g]


def line_free(block, xs, ys, p, q, n=None):
    step_lon = xs[1] - xs[0]
    nx = len(xs)
    ny = len(ys)
    n = n or max(4, int(max(abs(q[0] - p[0]), abs(q[1] - p[1])) / step_lon) * 2)
    for t in range(n + 1):
        x = p[0] + (q[0] - p[0]) * t / n
        y = p[1] + (q[1] - p[1]) * t / n
        i = int(round((x - xs[0]) / step_lon))
        j = int(round((y - ys[0]) / (ys[1] - ys[0])))
        if not (0 <= i < nx and 0 <= j < ny) or block[j, i]:
            return False
    return True


def string_pull(block, xs, ys, path):
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_free(block, xs, ys, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def path_km(path):
    tot = 0.0
    for a, b in zip(path, path[1:]):
        latm = (a[1] + b[1]) / 2
        klat, klon = km_per_deg(latm)
        tot += math.hypot((b[0] - a[0]) * klon, (b[1] - a[1]) * klat)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=float, default=0.05, help='격자 간격(도)')
    ap.add_argument('--clear', type=float, default=12.0, help='해안 통항 여유(km)')
    ap.add_argument('--dest', default='kaohsiung', choices=['kaohsiung', 'taipei'])
    ap.add_argument('--out', default=os.path.join(OUT, 'data', 'route_astar.json'))
    args = ap.parse_args()

    d = json.load(open(COAST))
    win = (117.0, 133.0, 20.0, 37.5)
    print(f'격자 굽는 중 (step {args.step}°, 창 {win}) …')
    land, xs, ys = build_grid(d['polys'], win, args.step)
    klat, _ = km_per_deg(28)
    k_cells = max(1, int(round(args.clear / (args.step * klat))))
    block = dilate(land, k_cells)
    print(f'  격자 {land.shape}, 육지 {land.mean():.1%}, 여유 {args.clear}km = {k_cells}셀 팽창 → 통항불가 {block.mean():.1%}')

    goal = KAOHSIUNG if args.dest == 'kaohsiung' else TAIPEI
    raw, dist = astar(block, xs, ys, BUSAN, goal)
    if raw is None:
        raise SystemExit('경로 없음 — 여유를 줄이거나 격자를 촘촘히 할 것')
    sp = string_pull(block, xs, ys, raw)
    L = path_km(sp)
    gc_klat, gc_klon = km_per_deg((BUSAN[1] + goal[1]) / 2)
    straight = math.hypot((goal[0] - BUSAN[0]) * gc_klon, (goal[1] - BUSAN[1]) * gc_klat)
    print(f'  A* 격자경로 {len(raw)}점 → string-pull {len(sp)}점')
    print(f'  항로장 {L:,.0f} km   (직선 {straight:,.0f} km, 우회율 {L/straight:.3f})')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # 격자 경로(raw)와 통항불가 셀 좌표도 저장 — 그림에서 A* 가 격자 탐색임을 보이기 위함
    ys_i, xs_i = np.nonzero(block)
    blk = np.column_stack([xs[xs_i], ys[ys_i]])
    json.dump(dict(path=sp, raw=[list(map(float, q)) for q in raw],
                   raw_n=len(raw), length_km=L, straight_km=straight,
                   raw_length_km=path_km(raw),
                   clear_km=args.clear, step_deg=args.step,
                   blocked=[[round(float(a), 3), round(float(b), 3)] for a, b in blk],
                   start=BUSAN, goal=list(goal), dest=args.dest),
              open(args.out, 'w'), ensure_ascii=False)
    print('저장:', args.out)


if __name__ == '__main__':
    main()
