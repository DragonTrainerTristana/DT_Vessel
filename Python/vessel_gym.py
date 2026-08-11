"""
vessel_gym.py — Unity 선박 RL 환경의 GPU 배치 시뮬레이터 (sim2sim, v1: 물리 코어).

목적: Unity↔Python 동기 왕복(초당 ~25스텝) 병목을 우회해 GPU에서 수천 환경을 동시에 굴려
      탐색을 100배 가속. 학습된 정책은 networks.py·체크포인트 그대로 Unity로 이식(같은 obs 369D/행동 2D).

★이 파일은 C# 원본(VesselDynamics/VesselRadar/COLREGsHandler/VesselManager)의 수식을 그대로 옮긴 것이다.
  수식 근거는 vessel-sim-spec-extract 워크플로(2026-07-03) 추출 명세. 추측 없음.

v1 범위(이 파일): 동역학 · 레이더(360 ray) · 충돌 · 스폰/에피소드(비동기) · COLREGs 상황판정 · 369D obs.
v2 예정: 보상 12항(reward), 통신 집계는 기존 networks.py가 담당(정책 쪽).

좌표계: Unity 좌표를 원점 중심 로컬 프레임으로 평행이동(물리 불변). 아레나 600×600 → [-300,300].
        heading = Unity eulerAngles.y (0=+Z 북, 시계방향 +). forward=(sin(h),cos(h)) in (x,z).
        장애물 9개 = 반지름 20 원(capsule가 sphere로 collapse), 3×3 격자 step 120, 원점 중심.

레이아웃: 텐서 [E, N] — E=병렬 환경 수, N=환경당 선박 수. 선박 상호작용은 환경 내에서만([E,N,N]).
단위: world units (VESSEL_SCALE=0.2 이미 반영된 값들). dt=0.04s, 결정당 10 서브스텝.
"""
import math
import os
import torch

# ─────────────────────────── 상수 (GlobalScale × VESSEL_SCALE=0.2 반영) ───────────────────────────
DT = 0.04                    # fixedDeltaTime
SUBSTEPS = 10                # DecisionPeriod — 결정(0.4s)당 물리 서브스텝
MAX_SPEED_BASE = 1.0         # GlobalScale.MAX_SPEED
ACCEL = 0.1                  # 가속률 (/s 아님, MoveTowards maxDelta = ACCEL*dt)
DECEL = 0.04
RUDDER_RATE = 12.0           # deg/s 타각 슬루
MAX_TURN_RATE = 30.0         # deg (명령 타각 상한)
TURN_FACTOR = 1.5            # rudderEff(1.5)·(10/length 2.0)·(beam 0.4/2) = 1.5
MAX_YAW_RATE = 45.0          # MAX_TURN_RATE * TURN_FACTOR (speedRatio=1, full rudder)
DRAG_COEF = 0.1
DRAG_THRUST_MULT = 0.3       # targetSpeed>=0.1이면 drag ×0.3
DRAG_THRUST_THRESH = 0.1     # 절대속도 단위

RADAR_RANGE = float(os.environ.get('VESSEL_RADAR_RANGE', '56.0'))  # ★제한시계(안개) regime: *지각(obs)만* 축소
RADAR_RANGE_BASE = 56.0      # 보상 기준 원값 — VESSEL_RADAR_RANGE와 무관하게 보상 불변(CTDE privileged). <20m 금지(THR 19.6 클립)
# ★센서고장(radar dropout) regime: 배별 간헐 블랙아웃 — obs만 마스킹(전방위 미감지 +0.5), 보상 불변.
#   블랙아웃 중 유일한 정보원 = 통신(파트너 threat-relay/위치) → 통신 가치가 필수가 되는 공정 시나리오.
RADAR_DROPOUT_P = float(os.environ.get('VESSEL_RADAR_DROPOUT_P', '0'))     # 결정당 블랙아웃 진입확률
RADAR_DROPOUT_LEN = int(os.environ.get('VESSEL_RADAR_DROPOUT_LEN', '50'))  # 블랙아웃 지속(결정 수) ≈ 20s
RADAR_RAYS = 360
RAY_HEIGHT = 0.2             # obs 평면 판정엔 무영향(수평 ray) — 기록용
GOAL_NORM_K = 150.0
GOAL_REACHED = 3.0
COMM_RANGE = 420.0           # (정책 통신 파트너용, sim은 위치만 제공)

DETECTION_RANGE = 56.0       # COLREGS_DETECTION
TCPA_RISK_DENOM = 30.0
DCPA_RISK = 24.0
HEAD_ON_ANGLE = 15.0
CROSSING_ANGLE = 112.5

# 선박 box collider (world, ×VESSEL_SCALE 0.2 반영): 길이(Z,bow) 14.18316, 폭(X,beam) 1.92741
SHIP_HALF_LEN = 14.18316 / 2.0     # 7.0916 (bow 방향 반길이)
SHIP_HALF_BEAM = 1.92741 / 2.0     # 0.9637 (beam 방향 반폭)

OBSTACLE_RADIUS = 20.0
OBSTACLE_GRID_STEP = 120.0   # 3×3 격자 간격
ARENA_HALF = 300.0           # 벽 box 중심 (600×600) — 씬 실측(2026-07-04): 벽=두께 1.0 box, 중심 ±300
WALL_HALF_THICK = 0.5        # 벽 두께/2 → 레이더·충돌이 만나는 면은 내면 ±299.5
ARENA_INNER = ARENA_HALF - WALL_HALF_THICK   # Unity 레이더 실측 대조로 확정 (평면 ±300 가정은 평균 0.64m 계통오차)

SPEED_MULT_MIN = 0.8
SPEED_MULT_MAX = 1.8

# ── 씬 실측 좌표 (Simulation.unity '대양' 아레나, 2026-07-04 파싱·검산 완료) ──
# spawn 20점: ±250 사각 둘레, 100m 간격. goal 16점: ±200 사각 둘레(스폰 안쪽 중첩), 100m 간격.
# ⚠️순서 = 씬 VesselManager 리스트 순서 그대로 — C# goal 배정의 strict-< 타이브레이크가 순서 의존.
SPAWN_PTS_SCENE = (
    (250.0, 250.0), (150.0, 250.0), (50.0, 250.0), (-50.0, 250.0), (-150.0, 250.0),
    (-250.0, 250.0), (-250.0, 150.0), (-250.0, 50.0), (-250.0, -50.0), (-250.0, -150.0),
    (-250.0, -250.0), (-150.0, -250.0), (-50.0, -250.0), (50.0, -250.0), (150.0, -250.0),
    (250.0, -250.0), (250.0, -150.0), (250.0, -50.0), (250.0, 50.0), (250.0, 150.0),
)
GOAL_PTS_SCENE = (
    (200.0, 200.0), (100.0, 200.0), (0.0, 200.0), (-100.0, 200.0), (-200.0, 200.0),
    (-200.0, 100.0), (-200.0, 0.0), (-200.0, -100.0), (-200.0, -200.0), (-100.0, -200.0),
    (0.0, -200.0), (100.0, -200.0), (200.0, -200.0), (200.0, -100.0), (200.0, 0.0),
    (200.0, 100.0),
)
# ★2026-08 진단 기반 상향(16000→30000, 1600→3000 결정): timeout 시점 목표까지 중앙값 68.9m(초기 373m)
#   = 목표 근처 4%뿐 → 반경 문제 아니고 *시간부족*. 여정 373m에 직선만도 ~900결정 필요한데 회피
#   detour(headTravel 1491°≈4회전)로 1600을 초과. 밀도·척수·기하는 그대로 두고 시간만 부여(난이도 유지).
MAX_EPISODE_STEPS = int(os.environ.get('VESSEL_MAX_EP_STEPS', '30000'))   # → 3000 결정
COLLISION_PENALTY = float(os.environ.get('VESSEL_COLLISION_PENALTY', '-300.0'))
# ★가치정렬 reward 실험용(2026-08-09): 안전·효율 가중을 학습목표에 반영 — 양 arm 동일 적용이라 공정
FUEL_COEF = float(os.environ.get('VESSEL_FUEL_COEF', '0.02'))
ARRIVAL_REWARD = 150.0        # ★reward#1 iter2 2026-08: 100→150 (도달 가치↑, 영구회피보다 도달 유리하게)
TIMEOUT_PENALTY = -50.0       # ★2026-08: timeout=실패 → 완만 페널티(충돌 -300보다 훨씬 약해 충돌 유발 안 함, 배회 억제)
# ★ablation 토글: COLREGs 준수 페널티 계수 (기본 0.45=1.5×강화). VESSEL_SIM_COLREGS_COEF=0 → COLREGs 강화 OFF arm.
COLREGS_SIM_COEF = float(os.environ.get('VESSEL_SIM_COLREGS_COEF', '0.45'))

# 종료 코드
OUT_RUNNING = 0
OUT_GOAL = 1
OUT_COLLISION_VESSEL = 2
OUT_COLLISION_OBSTACLE = 3
OUT_TIMEOUT = 4

# 상황 코드 (COLREGsHandler enum)
SIT_NONE, SIT_HEADON, SIT_STANDON, SIT_GIVEWAY, SIT_OVERTAKING = 0, 1, 2, 3, 4

DEG = math.pi / 180.0


def _move_toward(a, b, max_delta):
    """Unity Mathf.MoveTowards: a를 b쪽으로 최대 max_delta만큼 이동."""
    return a + torch.clamp(b - a, -max_delta, max_delta)


def _wrap180(deg):
    """[-180,180]로 wrap."""
    return (deg + 180.0) % 360.0 - 180.0


class VesselBatchEnv:
    """GPU 배치 선박 환경. 모든 상태는 [E,N] 텐서.

    사용:
        env = VesselBatchEnv(num_envs=1024, n_vessels=16, device='cuda')
        obs = env.reset()                       # [E, N, 369]
        obs, done, outcome = env.step(actions)  # actions [E, N, 2] ∈ [-1,1]
    """

    def __init__(self, num_envs=256, n_vessels=16, device='cpu', seed=0,
                 ring_scale=1.0, crossing=2, risk_range=56.0, dtype=torch.float32,
                 farfield_coef=0.0, perpair_coef=0.0, perpair_exp=1.6,
                 farpair_coef=None, farpair_exp=None):
        self.E, self.N = num_envs, n_vessels
        self.device = torch.device(device)
        self.dtype = dtype
        self.ring_scale = ring_scale
        self.crossing = crossing
        self.risk_range = risk_range
        # 보상 env 계수 (commgate: farfield=0.5, perpair=-0.3)
        self.farfield_coef = farfield_coef
        # ★far-field 직접 비용 (env override 가능, 기본 0 = 기존 동작 유지)
        self.farpair_coef = (float(os.environ.get('VESSEL_FARPAIR_COEF', '0'))
                             if farpair_coef is None else farpair_coef)
        self.farpair_exp = (float(os.environ.get('VESSEL_FARPAIR_EXP', '2.0'))
                            if farpair_exp is None else farpair_exp)
        self.perpair_coef = perpair_coef
        self.perpair_exp = perpair_exp
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        E, N = self.E, self.N
        z = lambda *s: torch.zeros(*s, device=self.device, dtype=self.dtype)
        # 상태
        self.pos = z(E, N, 2)          # (x, z) world-local
        self.heading = z(E, N)         # deg, Unity 규약
        self.speed = z(E, N)           # 현재 속도
        self.rudder = z(E, N)          # 실제(슬루된) 타각 deg
        self.cmd_rudder = z(E, N)      # 명령 타각 deg (결정당 고정)
        self.target_speed = z(E, N)    # 목표 속도 (결정당 고정)
        self.max_speed = z(E, N)       # 에피소드별 per-vessel (0.8~1.8)
        self.goal = z(E, N, 2)
        self.prev_dist = z(E, N)       # progress 보상용
        self.prev_far_risk = z(E, N)   # far-field PBRS용 (리셋 시 -1)
        self.prev_rudder = z(E, N)     # smoothness용
        self.step_count = torch.zeros(E, N, device=self.device, dtype=torch.long)
        self.situation = torch.zeros(E, N, device=self.device, dtype=torch.long)  # obs[368], 1-step stale
        self.dropout_left = torch.zeros(E, N, device=self.device, dtype=torch.long)  # 센서고장 잔여(결정 수)
        self.spawn_idx = torch.zeros(E, N, device=self.device, dtype=torch.long)

        # 정적 장애물 9개 (원점 중심 3×3, 반지름 20) — [9,2]
        gx = torch.tensor([-OBSTACLE_GRID_STEP, 0.0, OBSTACLE_GRID_STEP], device=self.device, dtype=self.dtype)
        ox, oz = torch.meshgrid(gx, gx)   # torch<1.10 기본 'ij' indexing
        self.obstacles = torch.stack([ox.reshape(-1), oz.reshape(-1)], dim=-1)  # [9,2]
        self.obstacle_r = OBSTACLE_RADIUS

        # 레이더 ray local 방향 (ray0=bow +Z, 시계방향) — [360,2] (x,z)
        ang = torch.arange(RADAR_RAYS, device=self.device, dtype=self.dtype) * (2 * math.pi / RADAR_RAYS)
        self.ray_local = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)  # [360,2]

        # 스폰/goal = 씬 실측 좌표 (2026-07-04, 링 근사 대체). 스폰 centroid = 아레나 원점 = 장애물 그리드 중심.
        assert self.N <= len(SPAWN_PTS_SCENE), f"n_vessels {self.N} > 씬 스폰 포인트 {len(SPAWN_PTS_SCENE)}"
        self.spawn_pts = torch.tensor(SPAWN_PTS_SCENE, device=self.device, dtype=self.dtype)   # [20,2]
        self.n_spawn = self.spawn_pts.shape[0]
        self.goal_pts = torch.tensor(GOAL_PTS_SCENE, device=self.device, dtype=self.dtype)     # [16,2]
        # goal 배정 (C# AssignGoalForVessel, CROSSING=2): antipode = -spawn(원좌표) 에 가장 가까운
        # goalPoint를 리스트 순서 strict-< 스캔으로 선택(동률 = 앞선 인덱스 승리 — argmin 첫 최소와 동일).
        # ⚠️CROSSING=1(최원거리)은 일부 스폰에서 다른 goal이 나옴(검산됨) — 이 sim은 대척(=2)만 구현.
        #   commgate Stage 1/2는 전부 CROSSING=2라 무영향. crossing!=2로 쓸 거면 최원거리 스캔 추가할 것.
        d2 = ((-self.spawn_pts[:, None, :]) - self.goal_pts[None, :, :]).pow(2).sum(-1)        # [20,16]
        self.goal_map = d2.argmin(dim=1)                                                        # [20]

    # ─────────────────────────── 스폰 / 리셋 ───────────────────────────
    def reset(self):
        mask = torch.ones(self.E, self.N, dtype=torch.bool, device=self.device)
        self._respawn(mask, initial=True)
        self._update_situation()
        return self._build_obs()

    def _respawn(self, mask, initial=False):
        """mask=True인 (env,vessel)만 리스폰 (비동기). initial=True면 전 선박 고유 spawn point 배정."""
        E, N = self.E, self.N
        n_reset = int(mask.sum().item())
        if n_reset == 0:
            return
        # spawn point 배정: 각 env에서 N개 선박에 서로 다른 point (근사 — Unity는 미사용 랜덤/최원거리)
        # 배치 근사: env마다 spawn_pts를 셔플해 앞 N개 사용. 리셋된 배만 갱신.
        if initial:
            for e in range(E):
                perm = torch.randperm(self.n_spawn, generator=self.gen, device=self.device)[:N]
                self.spawn_idx[e] = perm
        else:
            # 개별 리셋: Unity GetRandomUnusedSpawnIndex 미러 — env별 '생존 배가 점유한 포인트' 제외 랜덤.
            #   ⚠️이전의 순수 randint는 timeout 동기화 후 같은 포인트에 중복 스폰 → 즉사 vColl 연쇄
            #   (2026-07-05 eval 실측: 종료의 97%가 이 아티팩트 = Stage1 판정지표 오염) → 점유 제외 필수.
            #   같은 스텝 다중 리셋은 척당 순차 배정으로 상호 중복도 방지 (N<=n_spawn 전제, __init__ assert).
            arangeE = torch.arange(E, device=self.device)
            used = torch.zeros(E, self.n_spawn, dtype=torch.bool, device=self.device)
            keep = ~mask
            for i in range(N):
                k = keep[:, i]
                if k.any():
                    used[arangeE[k], self.spawn_idx[k, i]] = True
            for i in range(N):
                m = mask[:, i]
                if not m.any():
                    continue
                r = torch.rand(E, self.n_spawn, generator=self.gen, device=self.device, dtype=self.dtype)
                r = torch.where(used, torch.full_like(r, -1.0), r)   # 점유 인덱스 제외
                pick = r.argmax(dim=1)                                # free 중 랜덤 (전부 점유는 N<=20이라 불발)
                self.spawn_idx[:, i] = torch.where(m, pick, self.spawn_idx[:, i])
                used[arangeE[m], pick[m]] = True

        base = self.spawn_pts[self.spawn_idx]                  # [E,N,2]
        base = base * self.ring_scale                          # ring homothety (중심=원점, C# ApplyRingScale)

        # 목표 배정: 씬 goalPoint 16개 중 antipode 최근접 (원좌표로 매핑 후 ring scale — C# 순서 동일)
        goal = self.goal_pts[self.goal_map[self.spawn_idx]] * self.ring_scale   # [E,N,2]

        # 초기 heading = goal 방위 + U(-10,10)°
        to_goal = goal - base
        base_hdg = torch.atan2(to_goal[..., 0], to_goal[..., 1]) / DEG   # Unity: atan2(x,z)
        offset = (torch.rand(E, N, generator=self.gen, device=self.device, dtype=self.dtype) * 20 - 10)
        new_hdg = base_hdg + offset

        # per-vessel maxSpeed (Unity: Initialize 1회 고정. sim: 리셋마다 재추첨 — gotcha, v2에서 고정옵션)
        new_maxspeed = MAX_SPEED_BASE * (SPEED_MULT_MIN + (SPEED_MULT_MAX - SPEED_MULT_MIN) *
                                         torch.rand(E, N, generator=self.gen, device=self.device, dtype=self.dtype))
        # 초기 target speed = U(0.2,0.5) * maxSpeed
        init_ts = (0.2 + 0.3 * torch.rand(E, N, generator=self.gen, device=self.device, dtype=self.dtype)) * new_maxspeed

        m = mask
        self.pos = torch.where(m.unsqueeze(-1), base, self.pos)
        self.goal = torch.where(m.unsqueeze(-1), goal, self.goal)
        self.heading = torch.where(m, new_hdg, self.heading)
        self.max_speed = torch.where(m, new_maxspeed, self.max_speed)
        self.speed = torch.where(m, init_ts, self.speed)
        self.target_speed = torch.where(m, init_ts, self.target_speed)
        self.rudder = torch.where(m, torch.zeros_like(self.rudder), self.rudder)
        self.cmd_rudder = torch.where(m, torch.zeros_like(self.cmd_rudder), self.cmd_rudder)
        # ★위상 분산(2026-08): initial 리셋에서만 step_count를 U(0, MAX_EPISODE_STEPS) 랜덤 시작 →
        #   모든 배가 동시에 timeout하는 '종료 파도' 제거. 파도는 (a) reward 곡선에 3.28M 주기 폭락,
        #   (b) per-window goal% aliasing, (c) rollout이 한 outcome으로만 채워져 gradient 과보정을 유발했음.
        #   respawn(initial=False)은 0에서 시작(정상) — 한 번 흩뜨리면 위상이 계속 유지됨.
        if initial:
            phase = torch.randint(0, MAX_EPISODE_STEPS, (E, N), generator=self.gen,
                                  device=self.device, dtype=torch.long)
            self.step_count = torch.where(m, phase, self.step_count)
        else:
            self.step_count = torch.where(m, torch.zeros_like(self.step_count), self.step_count)
        self.prev_dist = torch.where(m, torch.linalg.norm(goal - base, dim=-1), self.prev_dist)
        self.prev_far_risk = torch.where(m, torch.full_like(self.prev_far_risk, -1.0), self.prev_far_risk)  # 첫 스텝 PBRS 스킵
        self.prev_rudder = torch.where(m, torch.zeros_like(self.prev_rudder), self.prev_rudder)

    # ─────────────────────────── 동역학 (10 서브스텝) ───────────────────────────
    def _apply_action(self, actions):
        """actions [E,N,2] ∈[-1,1] → 명령 타각·목표속도 세팅 (결정당 1회)."""
        a0 = torch.clamp(actions[..., 0], -1, 1)
        a1 = torch.clamp(actions[..., 1], -1, 1)
        self.cmd_rudder = a0 * MAX_TURN_RATE
        self.target_speed = torch.clamp((a1 + 1) * 0.5 * self.max_speed, torch.zeros_like(self.max_speed), self.max_speed)

    def _substep(self):
        """단일 물리 서브스텝 (dt=0.04). C# UpdateDynamics 순서 그대로."""
        # 1. 속도 1차 가감속
        accel_delta = torch.where(self.target_speed > self.speed,
                                  torch.full_like(self.speed, ACCEL * DT),
                                  torch.full_like(self.speed, DECEL * DT))
        self.speed = _move_toward(self.speed, self.target_speed, accel_delta)
        # 2. clamp [0, maxSpeed]
        self.speed = torch.clamp(self.speed, torch.zeros_like(self.speed), self.max_speed)
        # 3. 타각 슬루
        self.rudder = _move_toward(self.rudder, self.cmd_rudder, RUDDER_RATE * DT)
        # 4. 유효타각 = 실제타각 × speedRatio
        speed_ratio = self.speed / torch.clamp(self.max_speed, min=1e-6)
        eff_rudder = self.rudder * speed_ratio
        # 6. yawRate [deg/s]
        yaw_rate = eff_rudder * TURN_FACTOR
        # 7. 위치 적분 (회전 前 heading·현재속도) → heading 회전
        h_rad = self.heading * DEG
        fwd = torch.stack([torch.sin(h_rad), torch.cos(h_rad)], dim=-1)   # [E,N,2]
        self.pos = self.pos + fwd * (self.speed.unsqueeze(-1) * DT)
        self.heading = self.heading + yaw_rate * DT
        # 8. drag (속도 세팅 後, 다음 서브스텝 반영)
        drag_effect = torch.full_like(self.speed, DRAG_COEF * DT)
        drag_effect = torch.where(self.target_speed >= DRAG_THRUST_THRESH,
                                  drag_effect * DRAG_THRUST_MULT, drag_effect)
        self.speed = self.speed * (1 - drag_effect)

    # ─────────────────────────── 레이더 (360 ray, batched) ───────────────────────────
    def _radar(self):
        """[E,N,360] 정규화 거리 (dist/56-0.5, 미감지 +0.5). ray vs 원(장애물)+OBB(타선)+벽."""
        E, N, R = self.E, self.N, RADAR_RAYS
        h_rad = self.heading * DEG                                  # [E,N]
        cos_h, sin_h = torch.cos(h_rad), torch.sin(h_rad)
        # world ray dir: rotate local by heading. Unity: worldDir = R(h)*localDir.
        # local (lx,lz), world (wx,wz): wx = lx*cos + lz*sin ; wz = -lx*sin + lz*cos  (Unity Y-up 시계회전)
        lx = self.ray_local[:, 0]                                    # [R]
        lz = self.ray_local[:, 1]
        wx = lx[None, None] * cos_h[..., None] + lz[None, None] * sin_h[..., None]   # [E,N,R]
        wz = -lx[None, None] * sin_h[..., None] + lz[None, None] * cos_h[..., None]
        origin = self.pos                                            # [E,N,2]

        best = torch.full((E, N, R), RADAR_RANGE, device=self.device, dtype=self.dtype)

        # (a) ray vs 장애물 원 9개 — 각 원 [9,2]
        # 원점 o=origin, 방향 d=(wx,wz) 단위. 원 중심 c, 반지름 r. t = closest hit.
        for ci in range(self.obstacles.shape[0]):
            c = self.obstacles[ci]                                   # [2]
            best = self._ray_circle(origin, wx, wz, c, self.obstacle_r, best)

        # (b) ray vs 타선 OBB (선박 box world 14.18(len,Z)×1.93(beam,X), heading 회전)
        for j in range(N):
            best = self._ray_obb_batch(origin, wx, wz, j, best)

        # (c) ray vs 벽 (아레나 경계 4개 축평행 선분) — 축평행 box라 간단히 경계 평면 교차
        best = self._ray_walls(origin, wx, wz, best)

        return best / RADAR_RANGE - 0.5

    def _ray_circle(self, origin, wx, wz, c, r, best):
        """origin[E,N,2], dir(wx,wz)[E,N,R], 원 c[2] r → best[E,N,R] 갱신 (static 원)."""
        ox = origin[..., 0:1] - c[0]                                 # [E,N,1]
        oz = origin[..., 1:2] - c[1]
        # |o + t d|^2 = r^2 → t^2 + 2(o·d)t + (|o|^2-r^2)=0, d 단위
        b = ox * wx + oz * wz                                        # [E,N,R]
        cc = ox * ox + oz * oz - r * r
        disc = b * b - cc
        hit = disc >= 0
        sq = torch.sqrt(torch.clamp(disc, min=0))
        t = -b - sq                                                  # 가까운 근
        valid = hit & (t > 1e-4) & (t < best)
        return torch.where(valid, t, best)

    def _ray_obb_batch(self, origin, wx, wz, j, best):
        """모든 caster ray가 타선 j의 OBB(box)에 부딪히는 거리. slab method (box-local frame).
        origin[E,N,2], dir(wx,wz)[E,N,R], target j: center pos[:,j] heading[:,j]. 자기자신(N=j) 스킵."""
        hx, hz = SHIP_HALF_BEAM, SHIP_HALF_LEN                      # box 반폭·반길이
        cj = self.pos[:, j, :]                                      # [E,2]
        hj = self.heading[:, j] * DEG                               # [E]
        cos_h = torch.cos(hj)[:, None, None]                        # [E,1,1]
        sin_h = torch.sin(hj)[:, None, None]
        # ray origin을 box-local로 (rel 회전 -h)
        rx = origin[..., 0:1] - cj[:, None, 0:1]                    # [E,N,1]
        rz = origin[..., 1:2] - cj[:, None, 1:2]
        ox_l = rx * cos_h - rz * sin_h                              # local x(beam) [E,N,1]
        oz_l = rx * sin_h + rz * cos_h                              # local z(len)
        dx_l = wx * cos_h - wz * sin_h                              # [E,N,R]
        dz_l = wx * sin_h + wz * cos_h
        eps = 1e-9
        # slab X (반폭 hx)
        invx = 1.0 / torch.where(dx_l.abs() < eps, torch.full_like(dx_l, eps), dx_l)
        tx1 = (-hx - ox_l) * invx; tx2 = (hx - ox_l) * invx
        txmin = torch.minimum(tx1, tx2); txmax = torch.maximum(tx1, tx2)
        # slab Z (반길이 hz)
        invz = 1.0 / torch.where(dz_l.abs() < eps, torch.full_like(dz_l, eps), dz_l)
        tz1 = (-hz - oz_l) * invz; tz2 = (hz - oz_l) * invz
        tzmin = torch.minimum(tz1, tz2); tzmax = torch.maximum(tz1, tz2)
        tmin = torch.maximum(txmin, tzmin)
        tmax = torch.minimum(txmax, tzmax)
        hit = tmax >= torch.clamp(tmin, min=0)                      # 교차 존재
        t_entry = torch.where(tmin > 1e-4, tmin, tmax)              # 내부면 tmax
        valid = hit & (t_entry > 1e-4) & (t_entry < best)
        # 자기 자신 vessel(N축 인덱스 j) 스킵
        self_mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        self_mask[j] = True
        valid = valid & (~self_mask[None, :, None])
        return torch.where(valid, t_entry, best)

    def _ray_walls(self, origin, wx, wz, best):
        """축평행 아레나 경계(벽 내면 [-A,A]×[-A,A])와 ray 교차 (내부→경계 거리)."""
        A = ARENA_INNER
        ox, oz = origin[..., 0:1], origin[..., 1:2]
        for (axis_o, axis_d, lim) in [(ox, wx, A), (ox, wx, -A), (oz, wz, A), (oz, wz, -A)]:
            with torch.no_grad():
                denom = torch.where(axis_d.abs() < 1e-9, torch.full_like(axis_d, 1e-9), axis_d)
            t = (lim - axis_o) / denom
            valid = (t > 1e-4) & (t < best)
            best = torch.where(valid, t, best)
        return best

    # ─────────────────────────── COLREGs 상황판정 + risk (batched [E,N,N]) ───────────────────────────
    def _pairwise(self):
        """모든 배 쌍 (i,j)의 bearing·otherBearing·dist·rawTCPA·dcpa·risk·situation."""
        E, N = self.E, self.N
        pos_i = self.pos[:, :, None, :]                              # [E,N,1,2]
        pos_j = self.pos[:, None, :, :]                              # [E,1,N,2]
        to_other = pos_j - pos_i                                     # [E,N,N,2]
        dist = torch.linalg.norm(to_other, dim=-1)                  # [E,N,N]
        h = self.heading * DEG
        fwd = torch.stack([torch.sin(h), torch.cos(h)], dim=-1)     # [E,N,2]
        fi = fwd[:, :, None, :]                                      # [E,N,1,2]
        fj = fwd[:, None, :, :]                                      # [E,1,N,2]
        dx, dz = to_other[..., 0], to_other[..., 1]
        fx, fz = fi[..., 0], fi[..., 1]
        gx, gz = fj[..., 0], fj[..., 1]
        # bearing = SignedAngle(myFwd, toOther, up)
        bearing = torch.atan2(fz * dx - fx * dz, fx * dx + fz * dz) / DEG
        other_bearing = torch.atan2(gx * dz - gz * dx, -(gx * dx + gz * dz)) / DEG
        # 속도
        vel = fwd * self.speed.unsqueeze(-1)                        # [E,N,2]
        rel_vel = vel[:, None, :, :] - vel[:, :, None, :]           # [E,N,N,2]
        rel_speed = torch.linalg.norm(rel_vel, dim=-1)
        rel_pos = to_other
        dot = (rel_pos * rel_vel).sum(-1)
        raw_tcpa = torch.where(rel_speed < 0.01,
                               torch.full_like(dot, float('inf')),
                               -dot / torch.clamp(rel_speed ** 2, min=1e-9))
        tcpa = torch.clamp(raw_tcpa, min=0)
        pos_at = rel_pos + rel_vel * tcpa.unsqueeze(-1)
        dcpa = torch.where(rel_speed < 0.01, torch.linalg.norm(rel_pos, dim=-1),
                           torch.linalg.norm(pos_at, dim=-1))

        # risk (근거리, dist<=56 & rawTCPA>=0)
        distance_risk = 1.0 - dist / DETECTION_RANGE
        tcpa_risk = 1.0 / (1.0 + tcpa / TCPA_RISK_DENOM)
        dcpa_risk = 1.0 - torch.clamp(dcpa / DCPA_RISK, 0, 1)
        base_risk = distance_risk * 0.3 + tcpa_risk * 0.4 + dcpa_risk * 0.3

        # situation cascade
        absB, absOB = bearing.abs(), other_bearing.abs()
        sit = torch.full_like(dist, SIT_NONE, dtype=torch.long)
        # 유효 조우: dist<=56, rawTCPA>=0, not clear
        valid = (dist <= DETECTION_RANGE) & (raw_tcpa >= 0) & (absB <= 100.0)
        clear_pp = (bearing < -10.0) & (other_bearing < -10.0)
        clear_ss = (bearing > 10.0) & (other_bearing > 10.0)
        valid = valid & (~clear_pp) & (~clear_ss)
        # HeadOn: |b|<15 & |ob|<15
        headon = valid & (absB < HEAD_ON_ANGLE) & (absOB < HEAD_ON_ANGLE)
        # Overtaking: |ob|>112.5 (선미섹터)
        overtake = valid & (~headon) & (absOB > CROSSING_ANGLE)
        # Crossing (사각지대 fix: 5° 하한 제거): |b|<112.5
        crossing_zone = valid & (~headon) & (~overtake) & (absB < CROSSING_ANGLE)
        # 5° 초과: 내 bearing 부호. 5° 이하: 상대 bearing 부호
        cross_giveway = crossing_zone & torch.where(absB > 5.0, bearing > 0, other_bearing < 0)
        cross_standon = crossing_zone & (~cross_giveway)
        sit = torch.where(headon, torch.full_like(sit, SIT_HEADON), sit)
        sit = torch.where(overtake, torch.full_like(sit, SIT_OVERTAKING), sit)
        sit = torch.where(cross_giveway, torch.full_like(sit, SIT_GIVEWAY), sit)
        sit = torch.where(cross_standon, torch.full_like(sit, SIT_STANDON), sit)

        # 상황곱 risk (HeadOn×2.0/GiveWay×1.5/StandOn×1.3/Overtaking×1.2)
        sit_mult = torch.ones_like(base_risk)
        sit_mult = torch.where(sit == SIT_HEADON, torch.full_like(sit_mult, 2.0), sit_mult)
        sit_mult = torch.where(sit == SIT_GIVEWAY, torch.full_like(sit_mult, 1.5), sit_mult)
        sit_mult = torch.where(sit == SIT_STANDON, torch.full_like(sit_mult, 1.3), sit_mult)
        sit_mult = torch.where(sit == SIT_OVERTAKING, torch.full_like(sit_mult, 1.2), sit_mult)
        risk = torch.clamp(base_risk * sit_mult, 0, 1)
        # 무효 쌍(자기자신·미조우) risk=0
        eye = torch.eye(N, device=self.device, dtype=torch.bool)[None]
        invalid = eye | (dist > DETECTION_RANGE) | (raw_tcpa < 0)
        risk = torch.where(invalid, torch.zeros_like(risk), risk)
        sit = torch.where(invalid, torch.full_like(sit, SIT_NONE), sit)

        # far-field risk (56m~riskRange 띠, 상황곱 없음) — commgate far-field 보상용
        far_dist_risk = 1.0 - torch.clamp(dist / max(self.risk_range, 1e-6), 0, 1)
        far_risk = torch.clamp(far_dist_risk * 0.3 + tcpa_risk * 0.4 + dcpa_risk * 0.3, 0, 1)
        far_invalid = eye | (dist <= DETECTION_RANGE) | (dist > self.risk_range) | (raw_tcpa < 0)
        far_risk = torch.where(far_invalid, torch.zeros_like(far_risk), far_risk)

        return {'dist': dist, 'risk': risk, 'sit': sit, 'tcpa': tcpa, 'dcpa': dcpa, 'far_risk': far_risk}

    def _update_situation(self):
        """obs[368]용: argmax-risk 상대의 situation을 캐시 (Unity: 1-step stale)."""
        pw = self._pairwise()
        risk = pw['risk']                                            # [E,N,N]
        max_risk, arg = risk.max(dim=-1)                             # [E,N]
        sit = pw['sit']                                              # [E,N,N]
        chosen = torch.gather(sit, -1, arg.unsqueeze(-1)).squeeze(-1)
        # 위험 없으면 None
        self.situation = torch.where(max_risk > 0, chosen, torch.zeros_like(chosen))
        self._last_pw = pw

    # ─────────────────────────── obs 369D ───────────────────────────
    def _build_obs(self):
        E, N = self.E, self.N
        radar = self._radar()                                        # [E,N,360]
        if RADAR_DROPOUT_P > 0:
            radar = torch.where((self.dropout_left > 0).unsqueeze(-1),
                                torch.full_like(radar, 0.5), radar)  # 블랙아웃: 전방위 미감지(+0.5). 보상은 진짜 radar 사용
        to_goal = self.goal - self.pos                               # [E,N,2]
        d = torch.linalg.norm(to_goal, dim=-1)                       # [E,N]
        goal_dist = d / (d + GOAL_NORM_K)
        h = self.heading * DEG
        fwd = torch.stack([torch.sin(h), torch.cos(h)], dim=-1)
        # SignedAngle(fwd, toGoal, up)
        gx, gz = to_goal[..., 0], to_goal[..., 1]
        fx, fz = fwd[..., 0], fwd[..., 1]
        goal_angle = torch.atan2(fz * gx - fx * gz, fx * gx + fz * gz) / DEG
        goal_angle_n = goal_angle / 180.0
        speed_ratio = self.speed / torch.clamp(self.max_speed, min=1e-6)
        yaw_rate = self.rudder * speed_ratio * TURN_FACTOR
        yaw_n = yaw_rate / MAX_YAW_RATE
        heading_n = _wrap180(self.heading) / 180.0
        rudder_n = self.rudder / MAX_TURN_RATE
        situation_f = self.situation.to(self.dtype)
        # 조립: [radar360, goalDist, goalAngle, speed, yaw, heading, rudder, posX, posZ, situation] = 369
        obs = torch.cat([
            radar,
            goal_dist.unsqueeze(-1), goal_angle_n.unsqueeze(-1),
            speed_ratio.unsqueeze(-1), yaw_n.unsqueeze(-1),
            heading_n.unsqueeze(-1), rudder_n.unsqueeze(-1),
            self.pos[..., 0:1], self.pos[..., 1:2],
            situation_f.unsqueeze(-1),
        ], dim=-1)
        return obs

    # ─────────────────────────── 보상 (12항, 결정 단위) ───────────────────────────
    def _reward(self, actions, radar):
        """[E,N] shaping 보상. Unity CalculateReward를 결정 단위로 옮김(per-step 항은 ×SUBSTEPS).
        ★근사: Unity는 매 물리스텝(10회) 보상 누적 → 여기선 결정당 1회 계산 후 ×10(항 상수 가정).
        collision/arrival 종료보상은 step()에서 outcome 기반으로 별도 가산."""
        E, N = self.E, self.N
        pw = self._last_pw
        speed_ratio = self.speed / torch.clamp(self.max_speed, min=1e-6)
        r = torch.zeros(E, N, device=self.device, dtype=self.dtype)
        max_risk = pw['risk'].max(dim=-1).values   # [E,N] 최고 충돌위험 — 저속게이트·colcourse 공용

        # 1. time penalty -0.07
        r = r - 0.07
        # 2. forward bonus 0.1×speedRatio
        r = r + 0.1 * speed_ratio
        # 3. low speed penalty -0.15 (★reward#1 fix 2026-08: 위험 없을 때만 → 충돌코스 감속회피 Rule-8 허용)
        r = r + torch.where((speed_ratio < 0.2) & (max_risk < 0.1), torch.full_like(r, -0.15), torch.zeros_like(r))
        # 4. fuel -0.02×(speedRatio² + 0.5×turn01²), turn01=|명령타각|/maxTurn=|a0|
        turn01 = torch.clamp(actions[..., 0].abs(), 0, 1)
        r = r - FUEL_COEF * (speed_ratio ** 2 + 0.5 * turn01 ** 2)
        # 5. proximity -2.0×(1-d/19.6), ±135° 전방 섹터 최소거리 <19.6m
        THR = RADAR_RANGE_BASE * 0.35  # 19.6 — 안개(radar 축소)에서도 보상 불변
        front = torch.cat([radar[..., 0:135], radar[..., 225:360]], dim=-1)  # ±135°
        front_dist = (front + 0.5) * RADAR_RANGE  # 정규화 복원(m)
        fmin = front_dist.min(dim=-1).values
        # ★reward#1 fix: 근접벌점 계수↓(-2.0→-1.0)+제곱 → 중간거리 통과 허용, 접촉부근만 강함(fmin=0서 -10).
        _pnorm = torch.clamp(1 - fmin / THR, min=0.0)
        prox = torch.where(fmin < THR, -1.0 * _pnorm * _pnorm, torch.zeros_like(fmin))
        r = r + prox
        # 6. dense collision-course (★reward#1 fix: -0.8×risk³ → 중간위험(risk~0.3) 미미, 고위험만 강함)
        r = r + torch.where(max_risk > 0.05, -0.8 * max_risk ** 3, torch.zeros_like(max_risk))
        # 7. per-pair (commgate): perpair_coef × Σ min(risk,1)^exp (risk>0.05)
        if self.perpair_coef != 0.0:
            rj = pw['risk']
            contrib = torch.where(rj > 0.05, torch.clamp(rj, max=1.0) ** self.perpair_exp, torch.zeros_like(rj))
            perpair_cost = contrib.sum(dim=-1)
            r = r + self.perpair_coef * perpair_cost
        # 7-b. ★far-field 직접 비용 (2026-08-10, CTDE privileged reward):
        #   레이더 밖(56m~risk_range=420m, 즉 *통신으로만 알 수 있는* 구간) 충돌위험에 매 스텝 벌점.
        #   PBRS(#8)는 telescoping이라 최적정책 불변 → 먼 위협 회피 유인이 사실상 0이었음.
        #   이 항은 직접 비용이라 "먼 거리에서 미리 피한다"가 실제로 이득이 된다.
        #   ⚠️보상=privileged(전역), 관측=국소 유지 → 통신 arm만 이 비용을 줄일 수 있음(정보의 가치).
        #   양 arm 동일 적용. 계수 0이면 기존과 비트동일.
        if self.farpair_coef != 0.0:
            fr = pw['far_risk']
            fcost = torch.where(fr > 0.05, torch.clamp(fr, max=1.0) ** self.farpair_exp,
                                torch.zeros_like(fr)).sum(dim=-1)
            r = r + self.farpair_coef * fcost
        # 8. far-field PBRS (commgate): coef×(prevFar - curFar), 첫스텝(prev<0) 스킵
        cur_far = pw['far_risk'].sum(dim=-1)
        if self.farfield_coef > 0.0:
            pbrs = torch.where(self.prev_far_risk >= 0,
                               self.farfield_coef * (self.prev_far_risk - cur_far),
                               torch.zeros_like(cur_far))
            r = r + pbrs
        self.prev_far_risk = cur_far
        # 9. ★COLREGs 준수 페널티 (2026-08 Unity CalculateColregsReward 포팅 = "좌현 패널티"):
        #    max_risk>0.3에서 상황별 비준수 변침을 벌함 → 우현양보/침로유지 관례를 학습 → 조율(deadlock 해소).
        #    HeadOn(1)/GiveWay(3)/Overtaking(4)=우현 권장 → 좌현(rudder<0) 변침만 페널티(우현은 free).
        #    CrossingStandOn(2)=침로유지 권장 → |rudder| 페널티. situation은 방금 _update_situation이 갱신.
        _sit = self.situation                                   # [E,N] 0~4 (가장 위험한 조우 상황)
        _rud = actions[..., 0]                                  # [-1,1], >0=우현, <0=좌현
        _cgate = (max_risk > 0.3).to(r.dtype)
        _riskw = 1.0 + max_risk                                 # 1~2 (Unity riskWeight)
        _starboard = ((_sit == 1) | (_sit == 3) | (_sit == 4)).to(r.dtype)
        _port_pen = _starboard * torch.clamp(-_rud, min=0.0)    # 좌현 변침량(>0), 우현이면 0
        _hold_pen = (_sit == 2).to(r.dtype) * _rud.abs()        # StandOn: 어떤 변침도 억제
        r = r - COLREGS_SIM_COEF * _riskw * _cgate * (_port_pen + _hold_pen)
        # (per-step 항 소계 ×SUBSTEPS: Unity는 물리스텝마다 누적)
        r = r * SUBSTEPS
        # 10. navigation progress: (prevDist - d)×3.0  [★reward#1 iter2: ×1.0→×3.0 전진유인 강화 —
        #     중심부 건너기가 영구회피보다 확실히 이득이게. telescoping이라 정지=0(farming 없음)]
        d = torch.linalg.norm(self.goal - self.pos, dim=-1)
        r = r + (self.prev_dist - d) * 3.0
        self.prev_dist = d
        # angle: cos(goalAngle)×0.15×speedRatio ×SUBSTEPS
        to_goal = self.goal - self.pos
        h = self.heading * DEG
        fwd = torch.stack([torch.sin(h), torch.cos(h)], dim=-1)
        cos_ang = (fwd * to_goal).sum(-1) / torch.clamp(torch.linalg.norm(to_goal, dim=-1), min=1e-6)
        r = r + cos_ang * 0.15 * speed_ratio * SUBSTEPS
        # 12. smoothness -0.02×|Δ실제타각|/maxTurn ×SUBSTEPS
        rudder_change = (self.rudder - self.prev_rudder).abs() / MAX_TURN_RATE
        r = r - 0.02 * rudder_change * SUBSTEPS
        self.prev_rudder = self.rudder.clone()
        return r

    # ─────────────────────────── OBB 충돌 (SAT / box-원) ───────────────────────────
    def _obb_axes(self):
        """각 선박 box의 local 축(forward,starboard)·중심 반환."""
        h = self.heading * DEG
        fwd = torch.stack([torch.sin(h), torch.cos(h)], dim=-1)     # z-axis(bow) [E,N,2]
        stb = torch.stack([torch.cos(h), -torch.sin(h)], dim=-1)    # x-axis(beam)
        return fwd, stb

    def _obb_circle_hit(self, circles, r):
        """선박 OBB vs 원(장애물). circles[K,2] r 스칼라 → [E,N] hit."""
        fwd, stb = self._obb_axes()                                 # [E,N,2]
        rel = circles[None, None] - self.pos[:, :, None, :]         # [E,N,K,2]
        # box-local 좌표
        lx = (rel * stb[:, :, None, :]).sum(-1)                     # beam축 [E,N,K]
        lz = (rel * fwd[:, :, None, :]).sum(-1)                     # len축
        # 최근접점까지 거리 = clamp된 좌표와의 차
        cx = torch.clamp(lx, -SHIP_HALF_BEAM, SHIP_HALF_BEAM)
        cz = torch.clamp(lz, -SHIP_HALF_LEN, SHIP_HALF_LEN)
        closest_d = torch.sqrt((lx - cx) ** 2 + (lz - cz) ** 2)     # [E,N,K]
        return (closest_d < r).any(dim=-1)

    def _obb_obb_hit(self):
        """모든 선박 쌍 OBB-OBB SAT → [E,N] (자기 제외, 하나라도 겹치면 True)."""
        E, N = self.E, self.N
        fwd, stb = self._obb_axes()                                 # [E,N,2]
        c = self.pos                                                # [E,N,2]
        # 축 4개(각 박스 forward,stb). i=caster, j=target
        ci = c[:, :, None, :]; cj = c[:, None, :, :]                # [E,N,N,2]
        d = cj - ci                                                 # 중심 차 [E,N,N,2]
        axes = [fwd[:, :, None, :], stb[:, :, None, :],             # i축
                fwd[:, None, :, :], stb[:, None, :, :]]             # j축
        hi = [SHIP_HALF_LEN, SHIP_HALF_BEAM]                        # i 반extent (fwd,stb 순)
        hj = [SHIP_HALF_LEN, SHIP_HALF_BEAM]
        fi, si = fwd[:, :, None, :], stb[:, :, None, :]
        fj, sj = fwd[:, None, :, :], stb[:, None, :, :]
        overlap = torch.ones(E, N, N, dtype=torch.bool, device=self.device)
        for L, ax in enumerate(axes):
            # 두 박스를 축 ax에 투영한 반경
            proj_i = SHIP_HALF_LEN * (fi * ax).sum(-1).abs() + SHIP_HALF_BEAM * (si * ax).sum(-1).abs()
            proj_j = SHIP_HALF_LEN * (fj * ax).sum(-1).abs() + SHIP_HALF_BEAM * (sj * ax).sum(-1).abs()
            dist_ax = (d * ax).sum(-1).abs()
            sep = dist_ax > (proj_i + proj_j)                       # 이 축에서 분리
            overlap = overlap & (~sep)
        eye = torch.eye(N, device=self.device, dtype=torch.bool)[None]
        overlap = overlap & (~eye)
        return overlap.any(dim=-1)

    # ─────────────────────────── 충돌 / 종료 ───────────────────────────
    def _check_termination(self):
        """[E,N] outcome 코드. goal/collision(vessel·obstacle)/timeout."""
        E, N = self.E, self.N
        outcome = torch.full((E, N), OUT_RUNNING, device=self.device, dtype=torch.long)
        # goal 도달
        d = torch.linalg.norm(self.goal - self.pos, dim=-1)
        goal_hit = d < GOAL_REACHED
        # 장애물 충돌 (선박 OBB vs 원): 선박 중심~원중심 최근접거리(box-local) < obstacle_r
        obs_hit = self._obb_circle_hit(self.obstacles, self.obstacle_r)   # [E,N]
        # 벽 충돌: 선박 OBB의 최대 반경(꼭짓점)이 경계(벽 내면) 넘으면. 보수적으로 중심+대각반경 사용
        diag = math.hypot(SHIP_HALF_LEN, SHIP_HALF_BEAM)
        wall_hit = (self.pos.abs() > (ARENA_INNER - diag)).any(dim=-1)
        # 선박끼리 충돌: OBB-OBB SAT
        vessel_hit = self._obb_obb_hit()                            # [E,N]
        # timeout
        timeout = self.step_count >= MAX_EPISODE_STEPS
        # 우선순위: goal > collision_vessel > collision_obstacle(벽 포함) > timeout
        outcome = torch.where(timeout, torch.full_like(outcome, OUT_TIMEOUT), outcome)
        outcome = torch.where(obs_hit | wall_hit, torch.full_like(outcome, OUT_COLLISION_OBSTACLE), outcome)
        outcome = torch.where(vessel_hit, torch.full_like(outcome, OUT_COLLISION_VESSEL), outcome)
        outcome = torch.where(goal_hit, torch.full_like(outcome, OUT_GOAL), outcome)
        return outcome

    def step(self, actions):
        """actions [E,N,2] → obs[E,N,369], reward[E,N], done[E,N], outcome[E,N]."""
        self._apply_action(actions)
        for _ in range(SUBSTEPS):
            self._substep()
        self.step_count = self.step_count + SUBSTEPS
        self._update_situation()                       # _last_pw 갱신 (보상·situation)
        radar = self._radar()
        reward = self._reward(actions, radar)          # shaping (종료보상 전)
        outcome = self._check_termination()
        # 종료 보상: goal +100, collision -300 (Unity 종료 이벤트)
        reward = reward + torch.where(outcome == OUT_GOAL, torch.full_like(reward, ARRIVAL_REWARD), torch.zeros_like(reward))
        coll = (outcome == OUT_COLLISION_VESSEL) | (outcome == OUT_COLLISION_OBSTACLE)
        reward = reward + torch.where(coll, torch.full_like(reward, COLLISION_PENALTY), torch.zeros_like(reward))
        # ★2026-08: timeout 완만 페널티(실패 신호). 충돌(-300)보다 훨씬 약해 충돌 유발 안 함.
        reward = reward + torch.where(outcome == OUT_TIMEOUT, torch.full_like(reward, TIMEOUT_PENALTY), torch.zeros_like(reward))
        done = outcome != OUT_RUNNING
        if done.any():
            self._respawn(done, initial=False)
        if RADAR_DROPOUT_P > 0:
            # 센서고장 상태 전이(결정당 1회): 재스폰 초기화 → 잔여 감소 → 신규 진입 추첨
            self.dropout_left = torch.where(done, torch.zeros_like(self.dropout_left), self.dropout_left)
            self.dropout_left = torch.clamp(self.dropout_left - 1, min=0)
            enter = (torch.rand(self.E, self.N, generator=self.gen, device=self.device) < RADAR_DROPOUT_P) \
                    & (self.dropout_left == 0)
            self.dropout_left = torch.where(enter, torch.full_like(self.dropout_left, RADAR_DROPOUT_LEN),
                                            self.dropout_left)
        self._update_situation()                       # 리셋 후 obs용 상황 재계산
        obs = self._build_obs()
        return obs, reward, done, outcome

    def partner_goals_oracle(self):
        """ORACLE arm용: Unity oracle(networks.py msg 집계) 미러 — 파트너의 *정규화 goal obs*
        (d/(d+GOAL_NORM_K), signedAngle/180 — 파트너 자기중심) nearest-K 평균 [E,N,2].
        ⚠️이전 구현(goal 원좌표 ±140m 평균)은 msg 슬롯(O(1) 기대)을 스케일 폭파시켜 ORACLE arm이
        전-timeout 퇴행으로 붕괴(2026-07-05 16M 실측) — Unity 의미·스케일과 반드시 일치시킬 것."""
        E, N = self.E, self.N
        # 각 배 자신의 정규화 goal obs (\_build_obs와 동일 수식)
        to_goal = self.goal - self.pos
        d_own = torch.linalg.norm(to_goal, dim=-1)
        gd = d_own / (d_own + GOAL_NORM_K)
        h = self.heading * DEG
        fwd = torch.stack([torch.sin(h), torch.cos(h)], dim=-1)
        gx, gz = to_goal[..., 0], to_goal[..., 1]
        fx, fz = fwd[..., 0], fwd[..., 1]
        ga = torch.atan2(fz * gx - fx * gz, fx * gx + fz * gz) / DEG / 180.0
        gobs = torch.stack([gd, ga], dim=-1)                             # [E,N,2]
        # nearest-K 파트너 (COMM_RANGE 내) 평균
        dist = torch.linalg.norm(self.pos[:, :, None, :] - self.pos[:, None, :, :], dim=-1)  # [E,N,N]
        eye = torch.eye(N, device=self.device, dtype=torch.bool)[None]
        dist = torch.where(eye, torch.full_like(dist, 1e9), dist)
        in_range = dist < COMM_RANGE
        K = min(4, N - 1)
        _, idx = dist.topk(K, dim=-1, largest=False)                     # [E,N,K]
        gobs_j = gobs[:, None, :, :].expand(E, N, N, 2)
        sel = torch.gather(gobs_j, 2, idx.unsqueeze(-1).expand(E, N, K, 2))       # [E,N,K,2]
        val = torch.gather(in_range, 2, idx).unsqueeze(-1).to(self.dtype)         # [E,N,K,1]
        cnt = val.sum(dim=2).clamp(min=1.0)
        mean_g = (sel * val).sum(dim=2) / cnt                            # [E,N,2]
        has = (val.sum(dim=2) > 0).to(self.dtype)
        return mean_g * has
