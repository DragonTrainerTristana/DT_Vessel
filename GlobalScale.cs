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
    public const float VESSEL_SCALE = 0.1f;
    public const float MAP_SCALE = 1.0f;   // 학습 시와 동일 (월드맵 내 활동 영역은 300m 반경 유지)

    // Simulation Mode: 세계지도 ship traffic 시뮬레이션 전용
    // true면 maxEpisodeSteps 무한 (배가 goal/collision 외엔 안 끝남), Debug Ray off 등 최적화
    public const bool SIMULATION_MODE = true;

    // Debug Ray 렌더링: Editor 성능 크게 잡아먹음. SIMULATION_MODE에선 강제 off
    public const bool SHOW_DEBUG_RAYS = false;

    // ===== Base values (원본 스케일) =====
    public const float BASE_LENGTH = 10f;
    public const float BASE_BEAM = 2f;
    public const float BASE_MAX_SPEED = 5f;
    public const float BASE_ACCEL = 0.5f;
    public const float BASE_DECEL = 0.2f;
    public const float BASE_BRAKE = 1f;

    public const float BASE_RADAR_RANGE = 200f;
    public const float BASE_GOAL_REACHED = 15f;
    public const float BASE_WAYPOINT_REACHED = 20f;
    public const float BASE_PROXIMITY_THRESHOLD = 50f;
    public const float BASE_RAY_HEIGHT = 1f;

    // 맵/spawn 관련 (MAP_SCALE 대상)
    public const float BASE_MAP_DISTANCE = 1000f;
    public const float BASE_MIN_GOAL_DISTANCE = 50f;
    public const float BASE_NAVMESH_SNAP_RADIUS = 100f;
    public const float BASE_SPAWNZONE_RADIUS = 300f;
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

    public const float BASE_COLREGS_DETECTION = 200f;
    public const float BASE_RULE_17B_DIST = 30f;
    public const float BASE_RULE_17C_DIST = 15f;
    public const float BASE_SAFE_PASSING = 20f;
    public const float BASE_CRITICAL_CPA = 10f;
    public const float BASE_EFFECTIVE_SPEED_MIN = 3.5f;
    public const float BASE_MIN_SPEED_REDUCTION = 0.5f;
    public const float BASE_DCPA_RISK = 50f;

    // ===== Scaled values (Awake에서 사용) =====
    public const float LENGTH = BASE_LENGTH * VESSEL_SCALE;
    public const float BEAM = BASE_BEAM * VESSEL_SCALE;
    public const float MAX_SPEED = BASE_MAX_SPEED * VESSEL_SCALE;
    public const float ACCEL = BASE_ACCEL * VESSEL_SCALE;
    public const float DECEL = BASE_DECEL * VESSEL_SCALE;
    public const float BRAKE = BASE_BRAKE * VESSEL_SCALE;

    // 배 센서/물리 (VESSEL_SCALE)
    public const float RADAR_RANGE = BASE_RADAR_RANGE * VESSEL_SCALE;
    public const float GOAL_REACHED = BASE_GOAL_REACHED * VESSEL_SCALE;
    public const float WAYPOINT_REACHED = BASE_WAYPOINT_REACHED * VESSEL_SCALE;
    public const float PROXIMITY_THRESHOLD = BASE_PROXIMITY_THRESHOLD * VESSEL_SCALE;
    public const float RAY_HEIGHT = BASE_RAY_HEIGHT * VESSEL_SCALE;

    // 배들 간 거리 (VESSEL_SCALE)
    public const float MIN_SPAWN_DISTANCE = BASE_MIN_SPAWN_DISTANCE * VESSEL_SCALE;
    public const float SPAWN_OFFSET_MAX = BASE_SPAWN_OFFSET_MAX * VESSEL_SCALE;
    public const float SPAWN_RANDOM_OFFSET = BASE_SPAWN_RANDOM_OFFSET * VESSEL_SCALE;

    // 맵/spawn 영역 (MAP_SCALE) ← 월드맵 scale에 따라 조정
    public const float MAP_DISTANCE = BASE_MAP_DISTANCE * MAP_SCALE;
    public const float MIN_GOAL_DISTANCE = BASE_MIN_GOAL_DISTANCE * MAP_SCALE;
    public const float NAVMESH_SNAP_RADIUS = BASE_NAVMESH_SNAP_RADIUS * MAP_SCALE;
    public const float SPAWNZONE_RADIUS = BASE_SPAWNZONE_RADIUS * MAP_SCALE;
    public const float SPAWNZONE_SAMPLE = BASE_SPAWNZONE_SAMPLE * MAP_SCALE;
    public const float WAYPOINT_SAMPLE = BASE_WAYPOINT_SAMPLE * MAP_SCALE;
    public const float MIN_WAYPOINT_DIST = BASE_MIN_WAYPOINT_DIST * MAP_SCALE;

    public const float AUTOPILOT_RADAR = BASE_AUTOPILOT_RADAR * VESSEL_SCALE;
    public const float AUTOPILOT_COMM = BASE_AUTOPILOT_COMM * VESSEL_SCALE;
    public const float AUTOPILOT_GOAL = BASE_AUTOPILOT_GOAL * VESSEL_SCALE;

    public const float COLREGS_DETECTION = BASE_COLREGS_DETECTION * VESSEL_SCALE;
    public const float RULE_17B_DIST = BASE_RULE_17B_DIST * VESSEL_SCALE;
    public const float RULE_17C_DIST = BASE_RULE_17C_DIST * VESSEL_SCALE;
    public const float SAFE_PASSING = BASE_SAFE_PASSING * VESSEL_SCALE;
    public const float CRITICAL_CPA = BASE_CRITICAL_CPA * VESSEL_SCALE;
    public const float EFFECTIVE_SPEED_MIN = BASE_EFFECTIVE_SPEED_MIN * VESSEL_SCALE;
    public const float MIN_SPEED_REDUCTION = BASE_MIN_SPEED_REDUCTION * VESSEL_SCALE;
    public const float DCPA_RISK = BASE_DCPA_RISK * VESSEL_SCALE;

    // Rigidbody mass 스케일 (volume, s³)
    public const float MASS_SCALE = VESSEL_SCALE * VESSEL_SCALE * VESSEL_SCALE;

    // Max episode steps: 맵이 배 대비 크면 도달에 더 많은 step 필요
    // 실효 비율 = MAP_SCALE / VESSEL_SCALE (맵이 배 크기의 몇 배인지)
    // SIMULATION_MODE: 0 = ML-Agents 무한 (goal/collision 외엔 안 끝남)
    public const int BASE_MAX_EPISODE_STEPS = 15000;
    public const int MAX_EPISODE_STEPS = SIMULATION_MODE
        ? 0
        : (int)(BASE_MAX_EPISODE_STEPS * (MAP_SCALE / VESSEL_SCALE));

    // Transform localScale용 Vector 값
    public static UnityEngine.Vector3 TRANSFORM_SCALE =>
        new UnityEngine.Vector3(VESSEL_SCALE, VESSEL_SCALE, VESSEL_SCALE);
}
