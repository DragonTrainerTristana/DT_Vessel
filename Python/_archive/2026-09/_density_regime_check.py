"""
Density regime pre-training validation (geometry Monte-Carlo).

학습 전(=환경 빌드만으로) "다물체 액션 모호성" 빈도를 추정하여, density 레버
(VESSEL_SPAWN_RING_SCALE / VESSEL_VESSEL_COUNT / VESSEL_CROSSING=2)가 실제로
통신 niche(3척+ 동시 56m내 한점수렴)를 만드는지 *학습 전에* 검증한다.

방법은 조사단계 몬테카를로와 동일:
  - spawn = 정사각 둘레 링(±RING_HALF), goal = 둘레 링(±GOAL_HALF), 둘 다 원점 동심.
  - ring scale s: 좌표를 중심 기준 ×s.
  - crossing 모드:
      'farthest'(=VESSEL_CROSSING=1): 가장 먼 goal(코너 fan-out).
      'antipodal'(=VESSEL_CROSSING=2): 중심대칭점에 가장 가까운 goal(중심 관통).
      'random'(=off): 랜덤 goal.
  - 각 trial: N척을 spawn에 무작위 배치, 각자 등속 직선 진행(위상 무작위=async respawn 반영),
    무작위 시각 스냅샷에서 "어떤 배가 radar(56m)내에 2+ 동시 충돌코스 위협(0<TCPA<TCPA_MAX,
    DCPA<SAFE_PASSING)을 마주침" 비율 = ambiguity(통신 niche)를 센다.

이 지표가 baseline(~4.5%)에서 충분히(>~35%) 올라가야 density regime이 의미. 학습은 그 뒤.

⚠️ 이건 *기하 근사*(보상/정책 무관). 실제 학습 중 진짜 검증은 VESSEL_METRIC_LOG의
occlRate/minVesselDist + per-step threat 카운트(런타임 로깅)로 한다. 여기선 빌드 전 sanity check.
"""
import math
import random
import argparse

# GlobalScale와 동기 (VESSEL_SCALE 0.2)
RADAR = 56.0          # GlobalScale.RADAR_RANGE = 280 * 0.2 (절대 불변)
SAFE_PASSING = 12.0   # GlobalScale.SAFE_PASSING = 60 * 0.2 (DCPA 충돌코스 임계)
BASE_MAX_SPEED = 1.0  # GlobalScale.MAX_SPEED = 5 * 0.2
TCPA_MAX = 30.0       # 0<TCPA<30s 윈도우 (조사 기준)


def square_ring_points(half, n):
    """정사각 둘레에 균등 n점 (조사단계 spawn 링 = 둘레 균등)."""
    per = 4.0 * 2.0 * half
    pts = []
    for i in range(n):
        d = (i / n) * per
        if d < 2 * half:            # 아래변
            x, z = -half + d, -half
        elif d < 4 * half:          # 오른변
            x, z = half, -half + (d - 2 * half)
        elif d < 6 * half:          # 윗변
            x, z = half - (d - 4 * half), half
        else:                       # 왼변
            x, z = -half, half - (d - 6 * half)
        pts.append((x, z))
    return pts


def pick_goal(spawn, goals, mode):
    sx, sz = spawn
    if mode == 'antipodal':
        ax, az = -sx, -sz       # 중심(원점) 대칭점
        return min(goals, key=lambda g: (g[0] - ax) ** 2 + (g[1] - az) ** 2)
    if mode == 'farthest':
        return max(goals, key=lambda g: (g[0] - sx) ** 2 + (g[1] - sz) ** 2)
    return random.choice(goals)


def dcpa_tcpa(p_rel, v_rel):
    """상대위치/상대속도 → (dcpa, tcpa). v_rel≈0이면 tcpa=inf."""
    vv = v_rel[0] ** 2 + v_rel[1] ** 2
    if vv < 1e-9:
        return math.hypot(*p_rel), float('inf')
    tcpa = -(p_rel[0] * v_rel[0] + p_rel[1] * v_rel[1]) / vv
    cx = p_rel[0] + v_rel[0] * tcpa
    cz = p_rel[1] + v_rel[1] * tcpa
    return math.hypot(cx, cz), tcpa


def run(ring_half, goal_half, n, mode, ring_scale, trials, snaps_per_trial, seed):
    random.seed(seed)
    spawn_pts = [(x * ring_scale, z * ring_scale) for (x, z) in square_ring_points(ring_half, 20)]
    goal_pts = [(x * ring_scale, z * ring_scale) for (x, z) in square_ring_points(goal_half, 16)]

    any_pair = 0          # 스냅샷에 1+ pair conflict(1위협) 존재
    ambiguity = 0         # 스냅샷에 어떤 배가 2+ 동시위협(=joint maneuver 모호)
    triple = 0            # 3+ 동시한점(어떤 배가 3+ 동시위협)
    total_snaps = 0

    for _ in range(trials):
        # N척 배치: spawn에서 무복원 추출 (점모드 1점1척)
        idxs = random.sample(range(len(spawn_pts)), min(n, len(spawn_pts)))
        ships = []
        for si in idxs:
            sp = spawn_pts[si]
            gp = pick_goal(sp, goal_pts, mode)
            d = math.hypot(gp[0] - sp[0], gp[1] - sp[1])
            if d < 1e-6:
                continue
            spd = BASE_MAX_SPEED * random.uniform(0.8, 1.8)  # VesselAgent speedMultiplier
            vx = (gp[0] - sp[0]) / d * spd
            vz = (gp[1] - sp[1]) / d * spd
            travel_time = d / spd
            ships.append((sp, (vx, vz), travel_time))

        for _ in range(snaps_per_trial):
            total_snaps += 1
            # async 위상: 각 배 독립 진행률(0~1) → 동기리셋 없음 반영
            pos = []
            for (sp, v, tt) in ships:
                phase = random.random()
                t = phase * tt
                pos.append((sp[0] + v[0] * t, sp[1] + v[1] * t, v))

            snap_pair = False
            snap_ambig = False
            snap_triple = False
            for i in range(len(pos)):
                xi, zi, vi = pos[i]
                threats = 0
                for j in range(len(pos)):
                    if i == j:
                        continue
                    xj, zj, vj = pos[j]
                    rng = math.hypot(xj - xi, zj - zi)
                    if rng > RADAR:
                        continue
                    p_rel = (xj - xi, zj - zi)
                    v_rel = (vj[0] - vi[0], vj[1] - vi[1])
                    dcpa, tcpa = dcpa_tcpa(p_rel, v_rel)
                    if 0.0 < tcpa < TCPA_MAX and dcpa < SAFE_PASSING:
                        threats += 1
                if threats >= 1:
                    snap_pair = True
                if threats >= 2:
                    snap_ambig = True
                if threats >= 3:
                    snap_triple = True
            any_pair += snap_pair
            ambiguity += snap_ambig
            triple += snap_triple

    return {
        'pair_conflict_pct': 100.0 * any_pair / total_snaps,
        'ambiguity_pct': 100.0 * ambiguity / total_snaps,    # ★핵심: 2+ 동시위협 = 통신 niche
        'triple_pct': 100.0 * triple / total_snaps,          # 3+ 동시위협
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ring-half', type=float, default=250.0)   # 500m 정사각의 반변(±250)
    ap.add_argument('--goal-half', type=float, default=200.0)   # 400m goal 링
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--mode', choices=['random', 'farthest', 'antipodal'], default='farthest')
    ap.add_argument('--ring-scale', type=float, default=1.0)
    ap.add_argument('--trials', type=int, default=2000)
    ap.add_argument('--snaps', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sweep', action='store_true', help='대표 조합 일괄 출력')
    a = ap.parse_args()

    if a.sweep:
        combos = [
            ('baseline  ring1.0 n16 farthest ', 1.0, 16, 'farthest'),
            ('ring0.5    n16 farthest         ', 0.5, 16, 'farthest'),
            ('ring0.5    n16 antipodal        ', 0.5, 16, 'antipodal'),
            ('ring0.5    n24 antipodal        ', 0.5, 24, 'antipodal'),
            ('ring0.35   n20 antipodal        ', 0.35, 20, 'antipodal'),
            ('ring0.35   n20 farthest         ', 0.35, 20, 'farthest'),
        ]
        print(f"{'regime':34s}  pair%   ambig%  triple%   (ambig=통신 niche, baseline~4.5%)")
        for label, s, n, mode in combos:
            r = run(a.ring_half, a.goal_half, n, mode, s, a.trials, a.snaps, a.seed)
            print(f"{label}  {r['pair_conflict_pct']:5.1f}  {r['ambiguity_pct']:6.1f}  {r['triple_pct']:6.1f}")
    else:
        r = run(a.ring_half, a.goal_half, a.n, a.mode, a.ring_scale, a.trials, a.snaps, a.seed)
        print(f"mode={a.mode} ring_scale={a.ring_scale} n={a.n}")
        print(f"  pair_conflict (1+위협)        : {r['pair_conflict_pct']:.1f}%")
        print(f"  ambiguity     (2+동시위협,niche): {r['ambiguity_pct']:.1f}%")
        print(f"  triple        (3+동시위협)      : {r['triple_pct']:.1f}%")
