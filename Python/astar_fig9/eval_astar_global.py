"""전역경로(A*) + 국소 회피 평가 — Fig9(Global) 실측용.

구 Fig9_Global 은 mock_longhaul_estimate.py 가 만든 합성 그림이었음. 이 스크립트는
그 실험을 **실제로** 돌리는 하네스임. Unity NavMesh A*(WaypointPathFinder.cs) 가 하던 일을
vessel_gym 배치 심 위에서 재현함 — 그래야 Fig1~Fig8 과 같은 정책·같은 지표로 비교됨.

구조
----
1. A* 가 장애물을 피하는 전역경로를 뽑음 (격자 A* + string-pull + 최소간격 병합).
   Unity `WaypointPathFinder.CalculatePath` 와 같은 산출물: 시작점 제외, 목표 포함.
2. 그 waypoint 를 `env.goal` 에 하나씩 꽂아 넣음. 정책은 waypoint 를 목표로 인식함
   (Unity `VesselAutoPilot.SetWaypoints` → `goalPosition = waypoints[i]` 미러).
   정책·네트워크는 손대지 않음 — 목표 좌표만 바뀜.
3. 최종 waypoint = 진짜 목표. 여기 도달해야 OUT_GOAL. 중간 waypoint 는 종료 안 시킴.
4. 지표는 eval_ckpt.py 와 **같은 정의·같은 출력 형식** → metrics_*.txt 에 그대로 이어붙일 수 있음.

팔 (2×2)
-------
  --path direct : waypoint 1개(=목표). 현행 eval_ckpt.py 와 비트 동일한 조건.
  --path astar  : A* 전역경로.
  --arm OFF/ON  : 통신 (eval_ckpt.py 와 동일)

  `--path direct --arm OFF` 는 eval_ckpt.py 재현이어야 함 — 하네스 검증용 sanity check.

주의 (정직성)
------------
- 정책은 waypoint 없이(직행 목표로) 학습됨. A* arm 은 **zero-shot 일반화 평가**임.
  waypoint 는 목표보다 가까운 좌표라 학습 분포 안(에피소드 종반)이지만, 그래도 재학습이 아님.
  A* arm 이 지면 "A* 가 나쁘다"가 아니라 "이 정책이 waypoint 추종을 못 배웠다"일 수 있음.
- epReward 는 arm 간 비교 금지. waypoint 전환 때 progress 항의 기준거리(prev_dist)를
  다시 잡으므로 보상 스케일이 direct arm 과 다름. **ground-truth 지표(goal%/vColl%/fuel/len)만 씀.**
- 장거리(Taiwan↔Busan) 로 키우려면 --arena_scale 을 올릴 것. 비용은 README 참고.
"""
import argparse
import heapq
import math
import os
import time

import torch

import config as cfg
import vessel_gym as vg
from networks import CNNPolicy
from vessel_gym_train import comm_gather, parse_obs, FrameStack, make_others_msg


# ─────────────────────────── A* (격자, CPU) ───────────────────────────

class GridAStar:
    """아레나 위 8방향 격자 A*.

    점유 = 장애물 원(반지름 r)을 `inflate` 만큼 팽창시킨 영역, + 벽 안쪽 여유.
    팽창량 기본값 = 선체 반길이(SHIP_HALF_LEN). 선박을 점으로 보고 계획하기 위함.
    """

    DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1))

    def __init__(self, arena_inner, obstacles_xz, obstacle_r, cell, inflate, wall_margin):
        self.inner = float(arena_inner)
        self.cell = float(cell)
        self.margin = float(wall_margin)
        self.lim = self.inner - self.margin            # 중심이 들어갈 수 있는 최대 |x|,|z|
        self.n = int(math.floor(2.0 * self.lim / self.cell)) + 1
        self.obs = [(float(x), float(z)) for x, z in obstacles_xz]
        self.rad = float(obstacle_r) + float(inflate)
        self.r2 = self.rad * self.rad
        self.blocked = bytearray(self.n * self.n)
        for j in range(self.n):
            z = self._coord(j)
            for i in range(self.n):
                x = self._coord(i)
                for (ox, oz) in self.obs:
                    dx, dz = x - ox, z - oz
                    if dx * dx + dz * dz < self.r2:
                        self.blocked[j * self.n + i] = 1
                        break

    # ── 좌표 <-> 격자 ──
    def _coord(self, i):
        return -self.lim + i * self.cell

    def _idx(self, v):
        i = int(round((v + self.lim) / self.cell))
        return min(max(i, 0), self.n - 1)

    def cell_of(self, x, z):
        return self._idx(x), self._idx(z)

    def is_free(self, i, j):
        return not self.blocked[j * self.n + i]

    def nearest_free(self, i, j, max_ring=40):
        """점유 칸이면 가장 가까운 자유 칸으로 밀어냄 (스폰/목표가 팽창영역에 걸린 경우)."""
        if self.is_free(i, j):
            return i, j
        for r in range(1, max_ring + 1):
            best, bd = None, None
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    a, b = i + di, j + dj
                    if 0 <= a < self.n and 0 <= b < self.n and self.is_free(a, b):
                        d = di * di + dj * dj
                        if bd is None or d < bd:
                            best, bd = (a, b), d
            if best is not None:
                return best
        return i, j

    # ── 탐색 ──
    def search(self, start_xz, goal_xz):
        """격자 경로 [(x,z), ...] (시작칸 포함). 실패 시 None."""
        si, sj = self.nearest_free(*self.cell_of(*start_xz))
        gi, gj = self.nearest_free(*self.cell_of(*goal_xz))
        if (si, sj) == (gi, gj):
            return [(self._coord(si), self._coord(sj))]
        n = self.n
        s, g = sj * n + si, gj * n + gi
        SQ2 = math.sqrt(2.0)

        def h(k):
            di = abs(k % n - gi)
            dj = abs(k // n - gj)
            lo, hi = (di, dj) if di < dj else (dj, di)
            return (hi - lo) + SQ2 * lo          # octile

        gscore = {s: 0.0}
        came = {}
        pq = [(h(s), s)]
        closed = bytearray(n * n)
        while pq:
            _, k = heapq.heappop(pq)
            if closed[k]:
                continue
            if k == g:
                break
            closed[k] = 1
            ki, kj = k % n, k // n
            gk = gscore[k]
            for di, dj in self.DIRS:
                a, b = ki + di, kj + dj
                if not (0 <= a < n and 0 <= b < n):
                    continue
                if self.blocked[b * n + a]:
                    continue
                if di and dj:                     # 대각: 모서리 관통 금지
                    if self.blocked[kj * n + a] or self.blocked[b * n + ki]:
                        continue
                    step = SQ2
                else:
                    step = 1.0
                nk = b * n + a
                ng = gk + step
                if ng < gscore.get(nk, 1e18) - 1e-12:
                    gscore[nk] = ng
                    came[nk] = k
                    heapq.heappush(pq, (ng + h(nk), nk))
        if g not in gscore:
            return None
        path, k = [], g
        while k != s:
            path.append((self._coord(k % n), self._coord(k // n)))
            k = came[k]
        path.append((self._coord(si), self._coord(sj)))
        path.reverse()
        return path

    # ── 후처리 ──
    def point_free(self, p):
        """임의 좌표(격자 아님)가 팽창영역·벽 밖인지."""
        if abs(p[0]) > self.lim or abs(p[1]) > self.lim:
            return False
        for (ox, oz) in self.obs:
            dx, dz = p[0] - ox, p[1] - oz
            if dx * dx + dz * dz < self.r2:
                return False
        return True

    def line_free(self, p, q, step=None):
        """p→q 선분이 팽창 장애물을 안 지나는지 (샘플링)."""
        step = step or min(self.cell * 0.5, 1.0)
        dx, dz = q[0] - p[0], q[1] - p[1]
        L = math.hypot(dx, dz)
        m = max(2, int(L / step) + 1)
        for t in range(m + 1):
            x = p[0] + dx * t / m
            z = p[1] + dz * t / m
            if abs(x) > self.lim or abs(z) > self.lim:
                return False
            for (ox, oz) in self.obs:
                ddx, ddz = x - ox, z - oz
                if ddx * ddx + ddz * ddz < self.r2:
                    return False
        return True

    def string_pull(self, path):
        """가시선이 닿는 한 중간점 제거 (격자 계단 → 직선 구간)."""
        if len(path) <= 2:
            return list(path)
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.line_free(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out


def build_waypoints(planner, start_xz, goal_xz, min_wpt_dist):
    """Unity WaypointPathFinder.CalculatePath 미러:
    시작점 제외 · 목표 항상 포함 · 최소간격 미만 중간점 병합 · 실패 시 [goal] 폴백."""
    raw = planner.search(start_xz, goal_xz)
    if not raw:
        return [tuple(goal_xz)]
    # ★격자 스냅점이 아니라 *실제* 시작·목표로 양 끝을 되돌린 뒤 string-pull 한다.
    #   스냅점으로 가시선을 판정하고 마지막만 진짜 목표로 바꿔치기하면, 검사한 선분과
    #   실제 주행 선분이 최대 cell/√2 만큼 어긋나 팽창영역을 스치는 경로가 나온다(실측 32/320).
    raw = list(raw)
    if planner.point_free(start_xz):
        raw[0] = tuple(start_xz)
    if planner.point_free(goal_xz):
        raw[-1] = tuple(goal_xz)
    pts = planner.string_pull(raw)[1:]              # 시작점 제거
    if len(pts) <= 1:
        # 가시선이 목표까지 곧장 닿음 → 중간 waypoint 불필요.
        # (Unity SimplifyPath 의 `if (points.Count <= 1) return points;` 조기반환에 해당.
        #  이 가드가 없으면 마지막 점을 두 번 넣어 목표가 중복된다.)
        return [tuple(goal_xz)]
    simp = [pts[0]]                                 # SimplifyPath 미러
    for p in pts[1:-1]:
        if math.dist(simp[-1], p) >= min_wpt_dist:
            simp.append(p)
    simp.append(pts[-1])
    simp[-1] = tuple(goal_xz)                       # 마지막은 진짜 목표로 교체
    if len(simp) >= 2 and math.dist(simp[-2], simp[-1]) < 1e-6:
        simp.pop(-2)                                # 목표와 겹친 직전 점 제거
    return simp


# ─────────────────────────── waypoint 추종 ───────────────────────────

class WaypointRunner:
    """env.goal 을 현재 waypoint 로 덮어써 정책이 waypoint 를 좇게 만듦.

    - (spawn_idx, goal_idx) 조합이 유한(20×16)하므로 경로를 **전부 미리 계산**해 표로 들고 있음.
      → 리스폰 때 A* 를 다시 돌지 않음(런타임 비용 0).
    - 중간 waypoint 는 종료를 유발하면 안 됨. env 의 goal 판정 반경은 GOAL_REACHED(3.0m)이고
      결정당 최대 이동거리는 max_speed(1.8)×0.4s = 0.72m 이므로, step *전에* `wpt_reach`(기본 8m)
      에서 앞당겨 전환하면 3.0m 안으로 들어가기 전에 목표가 바뀜. assert 로 강제함.
    """

    def __init__(self, env, table_xz, table_len, wpt_reach):
        self.env = env
        self.wp = table_xz                    # [S,G,Lmax,2]
        self.wn = table_len                   # [S,G]  (>=1)
        self.reach = float(wpt_reach)
        E, N = env.E, env.N
        dev = env.device
        self.gi = torch.zeros(E, N, dtype=torch.long, device=dev)   # goal 인덱스
        self.cur = torch.zeros(E, N, dtype=torch.long, device=dev)  # 현재 waypoint 번호
        self.final = env.goal.clone()                               # 진짜 목표
        self.legs = torch.zeros(E, N, device=dev)                   # 완주 시 총 경로장(진단)
        step_m = float(vg.SPEED_MULT_MAX) * vg.MAX_SPEED_BASE * vg.DT * vg.SUBSTEPS
        assert self.reach > vg.GOAL_REACHED + step_m, (
            f'wpt_reach({self.reach}) 는 GOAL_REACHED({vg.GOAL_REACHED}) + '
            f'결정당 최대이동({step_m:.2f}) 보다 커야 중간 waypoint 오종료가 없음')

    def _goal_index(self, goal_xz):
        """env.goal 좌표 → goal_pts 인덱스 (스케일 적용 좌표와 정확히 일치)."""
        gp = self.env.goal_pts * self.env.ring_scale                # [G,2]
        d = torch.linalg.norm(goal_xz.unsqueeze(-2) - gp, dim=-1)   # [E,N,G]
        return d.argmin(dim=-1)

    def _apply(self, mask=None):
        """현재 waypoint 를 env.goal 에 반영하고 progress 기준거리를 다시 잡음."""
        E, N = self.env.E, self.env.N
        si = self.env.spawn_idx
        idx = self.cur.clamp(max=self.wp.shape[2] - 1)
        wp = self.wp[si, self.gi]                                   # [E,N,Lmax,2]
        tgt = torch.gather(wp, 2, idx[..., None, None].expand(E, N, 1, 2)).squeeze(2)
        if mask is None:
            self.env.goal = tgt
        else:
            self.env.goal = torch.where(mask.unsqueeze(-1), tgt, self.env.goal)
        d = torch.linalg.norm(self.env.goal - self.env.pos, dim=-1)
        # waypoint 전환 지점에서 progress 항이 튀지 않도록 기준거리 재설정
        self.env.prev_dist = torch.where(mask, d, self.env.prev_dist) if mask is not None else d

    def on_respawn(self, mask):
        """리스폰된 배의 경로를 새로 잡음. env.goal 에는 아직 '진짜 목표'가 들어있는 시점에 호출."""
        gi = self._goal_index(self.env.goal)
        self.gi = torch.where(mask, gi, self.gi)
        self.final = torch.where(mask.unsqueeze(-1), self.env.goal, self.final)
        self.cur = torch.where(mask, torch.zeros_like(self.cur), self.cur)
        self._apply(mask)

    def advance(self):
        """step 전에 호출. 도달한 중간 waypoint 를 넘김. 마지막 waypoint 는 넘기지 않음.
        반환: 이번에 전환이 일어났는지(bool). True면 호출측이 obs 의 goal 항을 다시 만들어야 함."""
        d = torch.linalg.norm(self.env.goal - self.env.pos, dim=-1)
        last = self.wn[self.env.spawn_idx, self.gi] - 1              # [E,N]
        hop = (d < self.reach) & (self.cur < last)
        if not bool(hop.any()):
            return False
        self.cur = torch.where(hop, self.cur + 1, self.cur)
        self._apply(hop)
        return True

    def path_length(self):
        """현재 배정된 전역경로의 총 길이 [E,N] (스폰점→...→목표). 진단용."""
        E, N = self.env.E, self.env.N
        si = self.env.spawn_idx
        wp = self.wp[si, self.gi]                                    # [E,N,Lmax,2]
        n = self.wn[si, self.gi]                                     # [E,N]
        start = (self.env.spawn_pts[si] * self.env.ring_scale).unsqueeze(2)   # [E,N,1,2]
        seq = torch.cat([start, wp], dim=2)                          # [E,N,Lmax+1,2]
        seg = torch.linalg.norm(seq[:, :, 1:] - seq[:, :, :-1], dim=-1)       # [E,N,Lmax]
        ar = torch.arange(seg.shape[2], device=seg.device)
        return (seg * (ar[None, None] < n.unsqueeze(-1)).to(seg.dtype)).sum(-1)


def build_path_table(env, mode, cell, inflate, wall_margin, min_wpt_dist, verbose=True):
    """(spawn, goal) 전 조합의 waypoint 표. 반환 [S,G,Lmax,2], [S,G]."""
    dev = env.device
    S = env.spawn_pts.shape[0]
    G = env.goal_pts.shape[0]
    sp = (env.spawn_pts * env.ring_scale).tolist()
    gp = (env.goal_pts * env.ring_scale).tolist()
    if mode == 'direct':
        wp = torch.tensor(gp, device=dev, dtype=env.dtype)[None, :, None, :].expand(S, G, 1, 2)
        return wp.contiguous(), torch.ones(S, G, dtype=torch.long, device=dev)

    t0 = time.time()
    planner = GridAStar(vg.ARENA_INNER, env.obstacles.tolist(), env.obstacle_r,
                        cell, inflate, wall_margin)
    paths = {}
    Lmax = 1
    for si in range(S):
        for gi in range(G):
            w = build_waypoints(planner, sp[si], gp[gi], min_wpt_dist)
            paths[(si, gi)] = w
            Lmax = max(Lmax, len(w))
    wp = torch.zeros(S, G, Lmax, 2, device=dev, dtype=env.dtype)
    wn = torch.zeros(S, G, dtype=torch.long, device=dev)
    for (si, gi), w in paths.items():
        for k, (x, z) in enumerate(w):
            wp[si, gi, k, 0] = x
            wp[si, gi, k, 1] = z
        for k in range(len(w), Lmax):                    # 꼬리는 목표로 패딩(안전)
            wp[si, gi, k] = wp[si, gi, len(w) - 1]
        wn[si, gi] = len(w)
    if verbose:
        nw = wn.float()
        # 직선 대비 경로 늘어난 비율 — A* 가 실제로 우회했는지 확인
        det = []
        for si in range(S):
            for gi in range(G):
                w = paths[(si, gi)]
                straight = math.dist(sp[si], gp[gi])
                pl = math.dist(sp[si], w[0]) + sum(math.dist(w[k], w[k + 1])
                                                   for k in range(len(w) - 1))
                det.append(pl / max(straight, 1e-6))
        det.sort()
        print(f'[A*] 격자 {planner.n}x{planner.n} (cell {cell}m, 팽창 {inflate:.2f}m) | '
              f'{S*G} 조합 {time.time()-t0:.1f}s | waypoint 수 평균 {nw.mean():.2f} 최대 {int(wn.max())} | '
              f'경로/직선 중앙 {det[len(det)//2]:.3f} 최대 {det[-1]:.3f}', flush=True)
    return wp, wn


# ─────────────────────────── 평가 ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arm', default='OFF', choices=['OFF', 'ORACLE', 'ON'])
    ap.add_argument('--path', default='astar', choices=['astar', 'direct'],
                    help='astar=전역경로 / direct=목표 직행(eval_ckpt.py 재현)')
    ap.add_argument('--envs', type=int, default=96)
    ap.add_argument('--vessels', type=int, default=16)
    ap.add_argument('--max_partners', type=int, default=cfg.MAX_COMM_PARTNERS)
    ap.add_argument('--eval_decisions', type=int, default=10000)
    ap.add_argument('--burnin', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--ring', type=float, default=1.0)
    ap.add_argument('--crossing', type=int, default=0)
    # A* 파라미터 (Unity WaypointPathFinder / GlobalScale 대응값이 기본)
    ap.add_argument('--cell', type=float, default=5.0, help='격자 한 칸(m)')
    ap.add_argument('--inflate', type=float, default=vg.SHIP_HALF_LEN,
                    help='장애물 팽창(m). 기본=선체 반길이')
    ap.add_argument('--wall_margin', type=float, default=vg.SHIP_HALF_LEN,
                    help='벽에서 띄울 여유(m)')
    ap.add_argument('--min_wpt_dist', type=float, default=1.5,
                    help='waypoint 최소간격(m). GlobalScale.MIN_WAYPOINT_DIST')
    ap.add_argument('--wpt_reach', type=float, default=8.0,
                    help='중간 waypoint 도달 반경(m). GOAL_REACHED+결정당이동 보다 커야 함')
    ap.add_argument('--tag', default='', help='출력 줄 앞에 붙일 라벨')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    E, N = args.envs, args.vessels
    torch.manual_seed(args.seed)
    scr = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.environ.get('VESSEL_CKPT_DIR', os.path.join(scr, 'checkpoints'))
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ckpt_dir, args.ckpt)

    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device=dev, seed=args.seed,
                            ring_scale=args.ring, crossing=args.crossing, risk_range=420.0,
                            farfield_coef=0.5, perpair_coef=-0.15, perpair_exp=3.0)
    policy = CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES).to(dev)
    sd = torch.load(ckpt_path, map_location=dev)
    policy.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)
    policy.eval()

    wp_tab, wn_tab = build_path_table(env, args.path, args.cell, args.inflate,
                                      args.wall_margin, args.min_wpt_dist)

    fs = FrameStack(E, N, dev)
    obs = env.reset()
    runner = WaypointRunner(env, wp_tab, wn_tab, args.wpt_reach)
    runner.on_respawn(torch.ones(E, N, dtype=torch.bool, device=dev))
    obs = env._build_obs()                       # goal 을 waypoint 로 바꿨으니 obs 재생성
    radar, goal, self_s, sit = parse_obs(obs)
    fs.reset_all(radar)

    counts = torch.zeros(5, device=dev)
    total = 0

    def act(x, goal, self_s, sit):
        with torch.no_grad():
            if args.arm == 'ON':
                om, _ = comm_gather(policy, env, x, goal, self_s, sit, args.max_partners)
            else:
                om = make_others_msg(env, args.arm, E, N, dev)
            action, _, _, _ = policy.ctr_actor(x, goal, self_s, om, sit)
        return action

    t0 = time.time()
    BIG = 1e9
    eye = torch.eye(N, device=dev).unsqueeze(0) * BIG

    def wrap180(x):
        return (x + 180.0) % 360.0 - 180.0

    def nearest_sep():
        d = torch.cdist(env.pos, env.pos) + eye
        return d.min(dim=-1).values

    def refresh_goal():
        """waypoint 가 바뀌면 obs 의 goal 항만 다시 만든다.
        radar 는 목표와 무관하므로 FrameStack 은 건드리지 않는다(프레임 중복 push 금지)."""
        _o = env._build_obs()
        _r, g2, s2, t2 = parse_obs(_o)
        goal_r[0], self_r[0], sit_r[0] = g2, s2, t2
        return _o

    def one_step():
        """waypoint 전환 → 행동 → step → 리스폰된 배 경로 재배정."""
        if runner.advance():
            refresh_goal()
        a = act(fs.get(), goal_r[0], self_r[0], sit_r[0])
        o, r, d, oc = env.step(a)
        if bool(d.any()):
            runner.on_respawn(d)                 # env.goal(진짜 목표) → waypoint[0]
            o = env._build_obs()
        return a, o, r, d, oc

    goal_r, self_r, sit_r = [goal], [self_s], [sit]

    # ── burn-in: 집계 없이 위상 dephase ──
    for i in range(args.burnin):
        _, obs, _, done, _ = one_step()
        radar, g2, s2, t2 = parse_obs(obs)
        goal_r[0], self_r[0], sit_r[0] = g2, s2, t2
        fs.push(radar, done)
        if i == 200:
            dps = 201 * E * N / (time.time() - t0)
            eta = (args.burnin + args.eval_decisions) * E * N / dps
            print(f'  [{os.path.basename(ckpt_path)}] {dps:.0f} dec/s, ETA ~{eta:.0f}s', flush=True)

    # ── 누적기 (eval_ckpt.py 와 동일 규약) ──
    ep_fuel = torch.zeros(E, N, device=dev)
    ep_head = torch.zeros(E, N, device=dev)
    ep_len = torch.zeros(E, N, device=dev)
    ep_minsep = torch.full((E, N), BIG, device=dev)
    ep_reward = torch.zeros(E, N, device=dev)
    ep_travel = torch.zeros(E, N, device=dev)          # ★실제 이동거리(경로효율 진단)
    ep_plan = runner.path_length()                      # ★배정된 전역경로 길이
    prev_head = env.heading.clone()
    counted = torch.zeros(E, N, dtype=torch.bool, device=dev)
    msum = {oc: dict(fuel=0.0, head=0.0, minsep=0.0, length=0.0, reward=0.0,
                     travel=0.0, plan=0.0, n=0) for oc in range(1, 5)}
    all_minsep_sum = 0.0
    all_minsep_n = 0
    comp_sum = 0.0
    comp_n = 0
    comp_bin = 0.0
    sit_ok = {k: 0.0 for k in (1, 2, 3, 4)}
    sit_n = {k: 0 for k in (1, 2, 3, 4)}

    for _ in range(args.eval_decisions):
        prev_pos = env.pos.clone()
        prev_spd = env.speed.clone()          # ★종료 스텝 이동거리 대체용(아래 ep_travel)
        sr = env.speed / torch.clamp(env.max_speed, min=1e-6)
        if runner.advance():
            refresh_goal()
        a = act(fs.get(), goal_r[0], self_r[0], sit_r[0])
        turn01 = a[..., 0].abs().clamp(0, 1)
        ep_fuel += sr ** 2 + 0.5 * turn01 ** 2
        dh = wrap180(env.heading - prev_head).abs()
        ep_head += dh
        prev_head = env.heading.clone()
        ep_minsep = torch.minimum(ep_minsep, nearest_sep())
        ep_len += 1.0
        # COLREGs 준수도 (eval_ckpt.py 와 동일 정의)
        _pw = env._last_pw
        if _pw is not None:
            _mrisk = _pw['risk'].max(dim=-1).values
            _sit = env.situation
            _rud = a[..., 0]
            _gate = (_mrisk > 0.3) & (_sit > 0)
            if bool(_gate.any()):
                _star = ((_sit == 1) | (_sit == 3) | (_sit == 4)).to(ep_fuel.dtype)
                _hold = (_sit == 2).to(ep_fuel.dtype)
                _viol = _star * torch.clamp(-_rud, min=0.0) + _hold * _rud.abs()
                _comp = (1.0 - _viol).clamp(0.0, 1.0)
                comp_sum += float(_comp[_gate].sum())
                comp_bin += float((_viol[_gate] < 0.1).float().sum())
                comp_n += int(_gate.sum())
                for _k in (1, 2, 3, 4):
                    _m = _gate & (_sit == _k)
                    if bool(_m.any()):
                        sit_ok[_k] += float((_viol[_m] < 0.1).float().sum())
                        sit_n[_k] += int(_m.sum())

        obs, rew_step, done, outcome = env.step(a)
        ep_reward += rew_step
        # ★2026-08-27 fix — env.step() 이 내부에서 _respawn(done) 을 부르므로(vessel_gym.py:741)
        #   종료한 배의 env.pos 는 이미 새 스폰 좌표다. 그대로 빼면 '종료지점→스폰점' 텔레포트
        #   거리(평균 340m, 최대 636m)가 더해져 travel/excess 가 통째로 오염됐다.
        #   종료 스텝만 이동 상한(speed×0.4s, ≤0.72m)으로 대체한다.
        #   eval_ckpt.py:83 의 '★step 전 유효 상태로 누적' 규약을 travel 에도 적용.
        ep_travel += torch.where(done, prev_spd * vg.DT * vg.SUBSTEPS,
                                 torch.linalg.norm(env.pos - prev_pos, dim=-1))

        for oc in range(1, 5):
            mask = (outcome == oc) & counted
            c = int(mask.sum())
            if c:
                counts[oc] += c
                total += c
                msum[oc]['fuel'] += float(ep_fuel[mask].sum())
                msum[oc]['head'] += float(ep_head[mask].sum())
                msum[oc]['minsep'] += float(ep_minsep[mask].clamp(max=vg.RADAR_RANGE * 4).sum())
                msum[oc]['length'] += float(ep_len[mask].sum())
                msum[oc]['reward'] += float(ep_reward[mask].sum())
                msum[oc]['travel'] += float(ep_travel[mask].sum())
                msum[oc]['plan'] += float(ep_plan[mask].sum())
                msum[oc]['n'] += c

        term = outcome != 0
        if int(term.sum()):
            _rec = term & counted
            if int(_rec.sum()):
                ms = ep_minsep[_rec].clamp(max=vg.RADAR_RANGE * 4)
                all_minsep_sum += float(ms.sum())
                all_minsep_n += int(_rec.sum())
            z = torch.zeros_like(ep_fuel)
            ep_fuel = torch.where(term, z, ep_fuel)
            ep_head = torch.where(term, z, ep_head)
            ep_len = torch.where(term, z, ep_len)
            ep_reward = torch.where(term, z, ep_reward)
            ep_travel = torch.where(term, z, ep_travel)
            ep_minsep = torch.where(term, torch.full_like(ep_minsep, BIG), ep_minsep)
            counted = counted | term
            runner.on_respawn(term)               # 새 목표 → 새 전역경로
            obs = env._build_obs()
            ep_plan = torch.where(term, runner.path_length(), ep_plan)

        radar, g2, s2, t2 = parse_obs(obs)
        goal_r[0], self_r[0], sit_r[0] = g2, s2, t2
        fs.push(radar, done)
        prev_head = torch.where(done, env.heading, prev_head)

    if total == 0:
        print(f'{os.path.basename(ckpt_path):26s} | no terminations')
        return
    g, v, o, t = (float(counts[i]) / total * 100 for i in (1, 2, 3, 4))
    gm = msum[1]
    if gm['n'] > 0:
        gf = gm['fuel'] / gm['n']
        gh = gm['head'] / gm['n']
        gs = gm['minsep'] / gm['n']
        gl = gm['length'] / gm['n']
        gt = gm['travel'] / gm['n']
        gp = gm['plan'] / gm['n']
    else:
        gf = gh = gs = gl = gt = gp = float('nan')
    amin = all_minsep_sum / max(all_minsep_n, 1)
    r_all = sum(msum[oc]['reward'] for oc in range(1, 5)) / max(total, 1)
    comp = comp_sum / max(comp_n, 1)
    compb = comp_bin / max(comp_n, 1) * 100
    tag = f'[{args.tag}] ' if args.tag else ''
    print(f'{tag}{os.path.basename(ckpt_path):26s} | arm={args.arm:6s} path={args.path:6s} | '
          f'goal={g:5.1f}%  vColl={v:5.1f}%  oColl={o:5.1f}%  TO={t:5.1f}%  | {total} eps || '
          f'[goal-ep] fuel={gf:6.1f}  headTravel={gh:6.0f}deg  minSep={gs:5.1f}m  len={gl:5.0f}  '
          f'| allMinSep={amin:5.1f}m || travel={gt:6.1f}m  plan={gp:6.1f}m  '
          f'excess={gt/max(gp,1e-6):5.3f} || epReward={r_all:8.2f}  colregs={comp:.4f}  '
          f'colregsOK={compb:5.1f}%  (n={comp_n})'
          + ' || ' + ' '.join(f'sit{k}={100*sit_ok[k]/max(sit_n[k],1):5.1f}%(n={sit_n[k]})'
                              for k in (1, 2, 3, 4)))


if __name__ == '__main__':
    main()
