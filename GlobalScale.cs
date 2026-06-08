/// <summary>
/// 모든 거리/속도 스케일을 중앙에서 관리.
/// Awake()에서 이 값들로 Prefab Inspector 값을 강제 덮어쓰므로
/// Unity에서 Prefab을 직접 수정할 필요가 없음.
///
/// 스케일 변경 방법: VESSEL_SCALE 한 줄만 수정하면 전체 재스케일.
/// - 1.0f: 원본 (length 10m)
/// - 0.1f: 1/10 (length 1m)  ← 현재
/// - 0.05f: 1/20 (length 0.5m) - Unity 물리 경계, 비추천
/// </summary>
public static class GlobalScale
{
    // ========================================================================
    // 두 개의 독립 스케일
    // VESSEL_SCALE: 배 자체 크기 (길이/속도/센서/물리)
    // MAP_SCALE:    월드맵 내 활동 영역 (spawn zone, goal distance 등)
    //
    // 두 스케일을 분리한 이유: 월드맵이 큰데 spawn zone까지 배 스케일로 줄이면
    // 배들이 한 점에 몰려 학습 분포 벗어남.
    // ========================================================================
    public const float VESSEL_SCALE = 0.2f;   // 0.3→0.2 (0.3은 진짜충돌률 ~78%로 학습불가였음. 0.1=충돌0/0.3=78% 사이 학습가능 regime 탐색). Python 동기화 필수
    public const float MAP_SCALE = 0.1f;   // VESSEL_SCALE과 통일 (센서↔맵 스케일 정합: dual-scale 불일치 제거)

    // Simulation Mode: 세계지도 ship traffic 시뮬레이션 전용
    // true면 maxEpisodeSteps 무한 (배가 goal/collision 외엔 안 끝남), Debug Ray off 등 최적화
    public const bool SIMULATION_MODE = true;

    // Debug Ray 렌더링: Editor 성능 크게 잡아먹음. SIMULATION_MODE에선 강제 off
    // (static readonly: const 대신 사용해야 CS0162 unreachable 경고 방지)
    public static readonly bool SHOW_DEBUG_RAYS = false;

    // 런타임 시각화 전역 off: goal line, detection sphere, trajectory line 등 모두 꺼짐
    public static readonly bool SHOW_RUNTIME_GIZMOS = false;

    // 통신/레이더 "범위" 원만 표시 (배당 wire circle 2개 → 가벼움, 렉 없음).
    // 무거운 360 감지선(SHOW_DEBUG_RAYS)·trajectory(SHOW_RUNTIME_GIZMOS)와 독립 토글.
    // 에디터에서 학습 관찰 시 ON. Scene 뷰(또는 Game 뷰 상단 Gizmos 버튼)에서 보임.
    public static readonly bool SHOW_RANGE_GIZMOS = true;

    // ===== Base values (원본 스케일) =====
    public const float BASE_LENGTH = 10f;
    public const float BASE_BEAM = 2f;
    public const float BASE_MAX_SPEED = 5f;
    public const float BASE_ACCEL = 0.5f;
    public const float BASE_DECEL = 0.2f;
    public const float BASE_BRAKE = 1f;

    public const float BASE_RADAR_RANGE = 280f;    // ×VESSEL_SCALE 0.2=56m. 부분관측 생성→통신이 먼 배 정보 메우게. dist/range 정규화 자동→from-scratch
    public const float BASE_GOAL_REACHED = 15f;
    public const float BASE_WAYPOINT_REACHED = 20f;
    public const float BASE_RAY_HEIGHT = 1f;

    // 맵/spawn 관련 (MAP_SCALE 대상)
    public const float BASE_MAP_DISTANCE = 200f;          // 1000→200 (×0.1=20m, goal obs 정규화 분모: 검증2에서 실측 확정)
    // goal distance obs 비선형 정규화 d/(d+k) 상수. 선형 분모(MAP_DISTANCE=20)는 실제 목표거리(중앙값 ~160 world units)의
    // 1/8이라 63% saturate였음 → 비선형으로 교체. k=로컬 항해 특성거리. ×MAP_SCALE=150 world units(측정 중앙값 부근).
    // d/(d+k)는 hard saturate가 없어 맵 스케일이 커져도 안 깨짐(원거리는 1.0에 점근, 근거리는 gradient 유지).
    public const float BASE_GOAL_NORM_K = 1500f;          // ×0.1=150 world units
    public const float BASE_MIN_GOAL_DISTANCE = 25f;      // 50→25 (×0.1=2.5m, γ horizon 내 도달)
    public const float BASE_NAVMESH_SNAP_RADIUS = 100f;
    public const float BASE_SPAWNZONE_RADIUS = 100f;      // 300→100 (×0.1=10m, radar 8m의 ~1.25배)
    public const float BASE_SPAWNZONE_SAMPLE = 50f;
    public const float BASE_WAYPOINT_SAMPLE = 5f;
    public const float BASE_MIN_WAYPOINT_DIST = 15f;

    // 배들 간 거리 (VESSEL_SCALE 대상, 배 길이 기준 5배)
    public const float BASE_MIN_SPAWN_DISTANCE = 50f;     // 15 → 50 (배 길이의 1.5배 → 5배)
    public const float BASE_SPAWN_OFFSET_MAX = 10f;
    public const float BASE_SPAWN_RANDOM_OFFSET = 20f;

    public const float BASE_AUTOPILOT_RADAR = 40f;
    public const float BASE_AUTOPILOT_COMM = 100f;
    public const float BASE_AUTOPILOT_GOAL = 10f;

    public const float BASE_COMM_RANGE = 2100f;           // 통신 파트너/시각화 범위 (×VESSEL_SCALE 0.2=420m). Python COMM_RANGE와 매칭. 아래 COLREGS_DETECTION(보상)과 분리
    public const float BASE_COLREGS_DETECTION = BASE_RADAR_RANGE;  // 레이더 범위와 동일 (×VESSEL_SCALE 0.2=56m). 윈도우 ~31s → 시간상수(21s) 재산정 검토 필요
    public const float BASE_RULE_17B_DIST = 90f;          // 30→90 (×VESSEL_SCALE 0.2=18m, COLREGS_DETECTION 확대 비율)
    public const float BASE_RULE_17C_DIST = 45f;          // 15→45 (×VESSEL_SCALE 0.2=9m)
    public const float BASE_SAFE_PASSING = 60f;           // 20→60 (×VESSEL_SCALE 0.2=12m)
    public const float BASE_CRITICAL_CPA = 30f;           // 10→30 (×VESSEL_SCALE 0.2=6m)
    public const float BASE_EFFECTIVE_SPEED_MIN = 3.5f;
    public const float BASE_MIN_SPEED_REDUCTION = 0.5f;
    public const float BASE_DCPA_RISK = 120f;             // 50→120 (×VESSEL_SCALE 0.2=24m, COLREGS의 ~43%)

    // COLREGs 시간상수 (초). 거리·속도가 둘 다 VESSEL_SCALE → 상쇄되어 스케일 불변.
    // COLREGS_DETECTION 40m / closing speed ~1.3m/s ≈ 31s 감지윈도우 기준 재산정 (28m→40m, ×1.43)
    //   closing 1.3 = maxSpeed 0.5 × speedMultiplier 평균 1.3 → 두 배 정면 합산 ≈ 1.3
    //   (시간상수 < 윈도우(31s) 유지해야 단계 게이팅 작동; 넘으면 감지 즉시 전단계 발동 → 붕괴)
    public const float BASE_EARLY_ACTION_TIME = 21.5f;        // 15 × 40/28
    public const float BASE_SUBSTANTIAL_ACTION_TIME = 11.5f;  // 8 × 40/28
    public const float BASE_RULE_17B_TIME = 7f;               // 5 × 40/28
    public const float BASE_RULE_17C_TIME = 3.5f;             // 2.5 × 40/28
    public const float BASE_TCPA_RISK_DENOM = 30f;            // 21 × 40/28 (≈ 40/1.3 윈도우)

    // ===== Scaled values (Awake에서 사용) =====
    public const float LENGTH = BASE_LENGTH * VESSEL_SCALE;
    public const float BEAM = BASE_BEAM * VESSEL_SCALE;
    public const float MAX_SPEED = BASE_MAX_SPEED * VESSEL_SCALE;
    public const float ACCEL = BASE_ACCEL * VESSEL_SCALE;
    public const float DECEL = BASE_DECEL * VESSEL_SCALE;
    public const float BRAKE = BASE_BRAKE * VESSEL_SCALE;

    // 배 센서/물리 (VESSEL_SCALE)
    public const float RADAR_RANGE = BASE_RADAR_RANGE * VESSEL_SCALE;
    public const int RADAR_RAYS    = 360;  // ★radar raw ray 수 = obs로 송신하는 radar 차원 (2026-06-04: min-pool 제거, Python RadarEncoder가 Conv1D 압축). config.py STATE_SIZE와 일치
    public const float GOAL_REACHED = BASE_GOAL_REACHED * VESSEL_SCALE;
    public const float WAYPOINT_REACHED = BASE_WAYPOINT_REACHED * VESSEL_SCALE;
    public const float RAY_HEIGHT = BASE_RAY_HEIGHT * VESSEL_SCALE;

    // 배들 간 거리 (VESSEL_SCALE)
    public const float MIN_SPAWN_DISTANCE = BASE_MIN_SPAWN_DISTANCE * VESSEL_SCALE;
    public const float SPAWN_OFFSET_MAX = BASE_SPAWN_OFFSET_MAX * VESSEL_SCALE;
    public const float SPAWN_RANDOM_OFFSET = BASE_SPAWN_RANDOM_OFFSET * VESSEL_SCALE;

    // 맵/spawn 영역 (MAP_SCALE) ← 월드맵 scale에 따라 조정
    public const float MAP_DISTANCE = BASE_MAP_DISTANCE * MAP_SCALE;
    public const float GOAL_NORM_K = BASE_GOAL_NORM_K * MAP_SCALE;   // 비선형 goal dist 정규화 d/(d+k)
    public const float MIN_GOAL_DISTANCE = BASE_MIN_GOAL_DISTANCE * MAP_SCALE;
    public const float NAVMESH_SNAP_RADIUS = BASE_NAVMESH_SNAP_RADIUS * MAP_SCALE;
    public const float SPAWNZONE_RADIUS = BASE_SPAWNZONE_RADIUS * MAP_SCALE;
    public const float SPAWNZONE_SAMPLE = BASE_SPAWNZONE_SAMPLE * MAP_SCALE;
    public const float WAYPOINT_SAMPLE = BASE_WAYPOINT_SAMPLE * MAP_SCALE;
    public const float MIN_WAYPOINT_DIST = BASE_MIN_WAYPOINT_DIST * MAP_SCALE;

    public const float AUTOPILOT_RADAR = BASE_AUTOPILOT_RADAR * VESSEL_SCALE;
    public const float AUTOPILOT_COMM = BASE_AUTOPILOT_COMM * VESSEL_SCALE;
    public const float AUTOPILOT_GOAL = BASE_AUTOPILOT_GOAL * VESSEL_SCALE;

    public const float COMM_RANGE = BASE_COMM_RANGE * VESSEL_SCALE;
    public const float COLREGS_DETECTION = BASE_COLREGS_DETECTION * VESSEL_SCALE;
    public const float RULE_17B_DIST = BASE_RULE_17B_DIST * VESSEL_SCALE;
    public const float RULE_17C_DIST = BASE_RULE_17C_DIST * VESSEL_SCALE;
    public const float SAFE_PASSING = BASE_SAFE_PASSING * VESSEL_SCALE;
    public const float CRITICAL_CPA = BASE_CRITICAL_CPA * VESSEL_SCALE;
    // near-miss 진단 임계(로그 전용): 이 거리 미만으로 타선 접근한 결정 수를 셈 → 충돌 직전 위험노출 측정.
    public const float NEAR_MISS_DIST = SAFE_PASSING;   // = 12m (BASE 60×VESSEL_SCALE 0.2). ★보상 절대 비연결(제약3)
    public const float EFFECTIVE_SPEED_MIN = BASE_EFFECTIVE_SPEED_MIN * VESSEL_SCALE;
    public const float MIN_SPEED_REDUCTION = BASE_MIN_SPEED_REDUCTION * VESSEL_SCALE;
    public const float DCPA_RISK = BASE_DCPA_RISK * VESSEL_SCALE;

    // COLREGs 시간상수: 스케일 불변 (거리/속도 동시 ×VESSEL_SCALE 상쇄) → BASE 그대로
    public const float EARLY_ACTION_TIME = BASE_EARLY_ACTION_TIME;
    public const float SUBSTANTIAL_ACTION_TIME = BASE_SUBSTANTIAL_ACTION_TIME;
    public const float RULE_17B_TIME = BASE_RULE_17B_TIME;
    public const float RULE_17C_TIME = BASE_RULE_17C_TIME;
    public const float TCPA_RISK_DENOM = BASE_TCPA_RISK_DENOM;

    // 타속(steering-gear rudder rate, °/s). 각도/초 차원 → 스케일 불변(시간상수 그룹과 동일).
    // full hard-over(30°→-30°=60°) 소요 = 60/RUDDER_RATE초. decision(0.4s)당 RUDDER_RATE×0.4°만 이동.
    // 권장 12 (허용 8~18). VESSEL_RUDDER_RATE env로 재빌드 없이 override(VesselAgent.Initialize 주입).
    public const float RUDDER_RATE = 12.0f;

    // Rigidbody mass 스케일 (volume, s³)
    public const float MASS_SCALE = VESSEL_SCALE * VESSEL_SCALE * VESSEL_SCALE;

    // Max episode steps: 맵이 배 대비 크면 도달에 더 많은 step 필요
    // 실효 비율 = MAP_SCALE / VESSEL_SCALE (맵이 배 크기의 몇 배인지)
    // SIMULATION_MODE: 0 = ML-Agents 무한 (goal/collision 외엔 안 끝남)
    public const int BASE_MAX_EPISODE_STEPS = 15000;
    // 학습 안전망: SIMULATION_MODE(무한)에서도 rollout 오염·배회 정체 방지용 유한 상한.
    // ⚠️ 실측: 목표거리 실제 ~100-225m (옛 "~20m" 가정 틀림). step이동=속도1×0.04s=0.04m/step.
    //   → 225m 직선=5625step. 12000→16000으로 상향: crossing 모드(목표=대각선 최대거리, ~500m)는 곡선 회피경로까지 여유 필요.
    //   16000×0.04×1m/s = 640m 예산. VESSEL_MAX_STEP env로 더 올림 가능(재빌드 없이; corner 목표가 멀면 20000).
    public const int TRAINING_MAX_STEPS = 16000;
    public const int MAX_EPISODE_STEPS = SIMULATION_MODE
        ? 0
        : (int)(BASE_MAX_EPISODE_STEPS * (MAP_SCALE / VESSEL_SCALE));

    // Transform localScale용 Vector 값
    public static UnityEngine.Vector3 TRANSFORM_SCALE =>
        new UnityEngine.Vector3(VESSEL_SCALE, VESSEL_SCALE, VESSEL_SCALE);
}
