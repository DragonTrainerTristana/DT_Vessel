using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;

public class VesselAgent : Agent
{
    public VesselDynamics vesselDynamics;

    [Header("Episode Settings")]
    public int maxEpisodeSteps = 15000;           // 학습: 15000, 시연: 50000

    public float arrivalReward = 100.0f;          // 도착 보상 강화
    public float goalDistanceCoef = 1.0f;         // 0.5 → 1.0 (progress 보상 강화)
    public float collisionPenalty = -300.0f;       // -100→-300 (충돌 종료가 reward 분류에서 절대 안 묻히게 + 회피 압력↑). threshold -150과 짝
    public float colregsRewardCoef = 0.45f;       // COLREGs 보상 계수 (1x=0.30, 1.5x=0.45). Initialize에서 VESSEL_COLREGS_COEF로 override
    public bool enableColregsReward = true;       // COLREGs 활성화 (우현 보상 제거됨, 좌현 패널티만)
    public float angleRewardCoef = 0.15f;         // 목적지 방향 보상 계수. ★실제 기본값은 Initialize에서 0.15로 강제(rush>avoid 수정: 0.5→0.15)
    public float forwardSpeedBonus = 0.1f;        // 전진 보상 (강화: 0.05 → 0.1)

    public float timePenalty = -0.07f;            // 매 스텝마다 패널티. ★실제 기본값은 Initialize에서 -0.07로 강제(회피 detour 완화: -0.1→-0.07)

    // Low speed penalty - 제자리 정지 방지
    public float lowSpeedThreshold = 0.2f;        // 20% maxSpeed 미만이면 패널티 (COLREGs 감속과 호환)
    public float lowSpeedPenalty = -0.15f;        // 저속 패널티 (강화: -0.1 → -0.15)

    public float goalReachedDistance = 1.5f;    // 1/10 스케일 (원본 15m)
    public float goalNormK = 150f;               // goal dist 비선형 정규화 d/(d+k) 상수 (Initialize에서 GlobalScale로 override)

    public Vector3 goalPosition;
    public bool hasGoal = false;
    private float previousDistanceToGoal;

    [Header("Waypoint Navigation")]
    private List<Vector3> waypoints;
    private int currentWaypointIndex = 0;
    public float waypointReachedDistance = 2f;      // 중간 웨이포인트 도달 거리 (1/10 스케일, 원본 20m)
    public float intermediateWaypointReward = 15f;   // 중간 도착 보상 (보상값 - 스케일 무관)
    private bool useWaypoints = false;

    private bool isCollided = false;

    // ── 에피소드 종료(outcome) 직접 기록: reward 추론 버리고 ground truth. per-run 파일(VESSEL_OUTCOME_LOG) ──
    private bool outcomeLogged = false;   // 이번 에피소드가 goal/collision으로 기록됐나 (false인 채 다음 에피소드 시작 → timeout)
    private int episodeIndex = 0;         // per-agent 에피소드 인덱스
    private int myStepCount = 0;          // 에피소드 길이(결정 수)
    private static string _outcomeLogPath = null;
    private static bool _outcomePathInit = false;

    private VesselManager vesselManager;
    private List<VesselAgent> cachedVessels;
    private Rigidbody rb;
    private Vector3 initialPosition;
    private Quaternion initialRotation;

    [Header("Radar Settings")]
    public VesselRadar radar;
    public float radarRange = 20f;    // 1/10 스케일 (원본 200m)
    public int radarObsSize = 360;    // radar obs 차원 = raw ray 360 (2026-06-04: min-pool 제거, Python RadarEncoder가 압축). Initialize에서 GlobalScale로 override
    public LayerMask radarDetectionLayers;

    // COLREGs Rule 17 추적: 3개 Dictionary → struct 1개 Dictionary (해시조회 75%↓, 메모리 3배↓)
    private struct PrevVesselState
    {
        public Vector3 position;
        public Vector3 forward;
        public float speed;
    }
    [Header("COLREGs Tracking (Rule 17)")]
    private Dictionary<GameObject, PrevVesselState> prevVesselStates;
    private float lastTrackingTime;

    // Per-frame 캐시: 가장 위험한 선박 계산 결과 (CollectObservations + CalculateReward 공유)
    private int dangerCacheFrame = -1;
    private VesselAgent cachedDangerousVessel;
    private float cachedDangerRisk;
    private COLREGsHandler.CollisionSituation cachedDangerSituation;

    [Header("Proximity Penalty (전방 장애물 회피)")]
    public float proximityThreshold = 5f;           // Initialize에서 radarRange×0.35로 override (관측 줄여도 회피 반응거리 보존)
    public float proximityPenaltyCoef = -2.0f;      // -1.0→-2.0 (근접 시 돌진 보상 ~+1.3 확실히 이김). Initialize 강제
    public int proximitySectorAngle = 135;           // ±45→±135 (충돌 45%가 측면. 전방+측면 270° 커버, 후방 제외). Initialize 강제

    [Header("Smoothness Reward")]
    // ⚠️ comm ON/OFF 비교 공정성 위해 항상 ON (보상은 두 조건 동일, 통신은 정보만 차이).
    public bool enableSmoothnessReward = true;      // 부드러운 회피 유도(급변침 패널티) - 연료 논제
    public float smoothnessCoef = -0.02f;           // 실제 타각 변화 패널티 (슬루가 이미 평탄화 → -0.05에서 완화)
    public float commandMismatchCoef = -0.03f;      // 명령-실제 타각 불일치(타속 한계에 박는 정도) 패널티 — 슬루 후 새로 필요
    private float previousRudderAngle = 0f;         // 이전 프레임의 실제 타각
    private float previousCommandedRudder = 0f;     // 이전 프레임의 명령 타각 (논제 메트릭용)
    private float episodeCommandVar = 0f;           // 명령 타각 변화 누적(통신 ON/OFF 비교 — 슬루에도 안 죽는 의도 메트릭)

    [Header("Fuel/Early-Avoid Reward (연료 논제)")]
    public float fuelCoef = -0.02f;                 // 연료 프록시 패널티(thrust²+0.5·turn²/step). 멀리서 부드러운 회피 유도
    public float earlyAvoidCoef = 0.3f;             // 조기 회피 보상: 멀리서 DCPA를 벌릴 때(+). 회피 EV를 +로 전환
    // ★ Dense 충돌코스 페널티: U자 회귀(수렴하며 회피를 돌진에 양보) 방지. true risk 기반, 매 스텝.
    //   VESSEL_COLCOURSE_COEF env로 재빌드 없이 튜닝(기본 -0.8). risk(0~1)에 비례 → 충돌코스 지속 비용.
    public float collisionCourseCoef = -0.8f;
    // ★ dense-steepening: risk^exp 비선형화(포화 제거+거리 gradient 복원). exp=1이면 옛 선형 동작 = 안전 기본.
    //   VESSEL_COLCOURSE_EXP env로 튜닝(기본 1.0=불변, 시험권장 1.6~2.0). gate를 risk>exp 곡선에 맞춰 분리.
    private float colcourseExp  = 1.0f;   // risk 지수 (1=선형/옛동작, 2=risk² 스티프닝)
    private float colcourseGate = 0.05f;  // dense 발화 risk 게이트 (VESSEL_COLCOURSE_GATE)
    // ★ Terminal-proximity ramp (terminal-magnitude fix): 희소 -300을 확실·저분산 거리×접근율 그라디언트로 변환.
    //   VESSEL_PROX_RAMP_COEF env(기본 0=off, 안전). 시험권장 -2.5 (p_eq ~6%→~2%, EV math로 산정).
    //   VESSEL_PROX_RAMP_DIST env(기본 24m=DCPA_RISK). 이 거리부터 ramp 시작(0) → 0거리에서 최대.
    private float proxRampCoef = 0f;      // 0=off. <0이면 발화. proximity²·closing01에 곱.
    private float proxRampDist = 24f;     // ramp 시작 거리(m). Initialize에서 GlobalScale.DCPA_RISK로 강제.
    private float prevDcpa = -1f;                   // 직전 프레임 가장 위험한 위협 DCPA (early-avoid 보상용, COLREGsHandler 기반)
    // ── per-episode 메트릭 누적 (VESSEL_METRIC_LOG ground-truth, comm ON/OFF 연료 비교) ──
    private float episodeFuel = 0f;
    private float episodeRudderVar = 0f;
    private float colregsComplianceSum = 0f;
    private int   colregsComplianceCount = 0;
    private static string _metricLogPath = null;
    private static bool _metricPathInit = false;
    // ── occlusion(LOS) 게이트 + 측정 (가림막 실험용) ──
    // VESSEL_LOS_GATE=1: 원통 등에 *가려진* 위협은 보상 risk에서 제외 → 못 보는 걸로 안 벌줌(anti-rigging).
    //   통신은 가려진 위협을 메시지로 전하므로, comm-ON은 진짜 충돌(-300)·연료로 이득.
    private bool losGate = false;
    private int threatSteps = 0;          // comm 범위 내 위협 횟수 (decision×vessel)
    private int occludedThreatCount = 0;  // 그중 LOS 차단(가려진) 수 → occlusion 비율 측정

    // ── 보상 재설계 토글 (env, 재빌드 없이 ablation) ──
    private float earlyRiskGate = 0.1f;        // earlyAvoid 발화 risk 게이트(0.3→0.1: 근접 회피도 DCPA-증가 보상)
    private bool  relaxEarlyTcpa = true;       // earlyAvoid의 tcpa>11.5s 게이트 제거(any tcpa)
    private bool  enableStraightBonus = false; // 직진보너스(rudder≈0 보상=회피 변침과 충돌) 기본 off
    // ── ★ Rule-8 속도회피 언락 (env, 기본 OFF=옛 동작 불변) ──
    //   진단: reward가 감속회피를 strictly dominated로 만듦 — (1) lowSpeedPenalty가 충돌코스 감속도 벌하고,
    //   (2) earlyAvoid가 ×speedRatio라 감속에 의한 DCPA-증가 보상을 반토막. → 정책이 "절대 감속 안 함 = 돌진" 학습.
    //   언락1: 실제 충돌코스(접근중 risk>게이트)에선 lowSpeedPenalty 면제.
    //   언락2: 충돌코스에선 earlyAvoid의 ×speedRatio 제거(감속회피로 얻은 DCPA-증가에 full 보상).
    //   ★freeze-proof: cachedDangerRisk는 rawTCPA<0(후퇴/통과)면 0으로 하드컷 → 자유수면/정지·후퇴엔 risk=0 →
    //   면제·full보상 발화 안 함. "실제 접근중 위협의 DCPA가 커질 때"만 작동 = loiter/freeze farming 불가.
    //   comm ON/OFF 동일(risk=privileged ground-truth, 메시지 무관) = anti-rigging 안전.
    private bool  enableSpeedAvoidUnlock = false;   // VESSEL_SPEED_AVOID_UNLOCK=1
    private float speedUnlockRiskGate   = 0.3f;     // 언락 발화 risk 게이트(VESSEL_SPEED_UNLOCK_GATE). 실질충돌코스만
    // ── ★ Far-field 조기회피 보상 (CTDE-lite, env, 기본 OFF=현재와 비트동일) ──
    //   목적: 레이더 안 줄이고 통신을 가치있게. 보상이 56m(=레이더)만 보던 걸, 56m~riskRange 먼 배의
    //   충돌코스 risk *합*이 줄어들 때(=조기회피) carrot로 보상. comm-ON은 옆 배가 전한 먼 위협에 미리
    //   행동→far risk↓→보상; comm-OFF는 못 봐서 보너스만 적게(★벌점 X=baseline 안 무너짐, carrot).
    //   "줄어듦만 보상"이라 정지/farming 무의미. LOS 가림 제외. comm ON/OFF 동일 coef = anti-rigging.
    private float farFieldCoef = 0f;     // VESSEL_FARFIELD_COEF (기본 0=off). >0이면 carrot 발화
    private float riskRange = 0f;        // VESSEL_RISK_RANGE (Initialize에서 COLREGS_DETECTION으로 기본=띠없음)
    private float cachedFarRiskSum = 0f; // 이번 프레임 56m~riskRange 먼 배 risk 합(UpdateDangerCache가 채움)
    private float prevFarRiskSum = -1f;  // 직전 프레임 far risk 합 (carrot용, 에피소드 -1 sentinel)
    // ── circling/near-miss 진단 instrumentation (★로그 전용, 보상 절대 비연결 — 제약3 spinning 탐지킬 금지) ──
    private Vector3 episodeSpawnPos;
    private Vector3 lastPathPos;
    private float episodePathLength = 0f;      // 누적 경로 길이
    private float episodeHeadingTravel = 0f;   // 누적 |Δheading| (도)
    private float prevHeadingForTravel = 0f;
    private float episodeMinVesselDist = float.MaxValue;  // 에피소드 최근접 타선 거리
    private int   nearMissSteps = 0;                       // minDist<NEAR_MISS_DIST 결정 수
    private float currentMinVesselDist = float.MaxValue;   // 이번 프레임 최근접(UpdateDangerCache가 채움)

    public override void Initialize()
    {
        // Prefab Inspector 값 무시하고 GlobalScale로 강제 덮어쓰기
        // SIMULATION_MODE 무한 에피소드는 학습 rollout 오염·배회 정체 유발 → 유한 안전망 적용
        // VESSEL_MAX_STEP env로 MaxStep override (재빌드 없이 튜닝). 목표거리 vs 이동예산 정합용.
        int trainMax = GlobalScale.TRAINING_MAX_STEPS;
        string envMaxStep = System.Environment.GetEnvironmentVariable("VESSEL_MAX_STEP");
        if (!string.IsNullOrEmpty(envMaxStep) && int.TryParse(envMaxStep, out int parsedMax) && parsedMax > 0)
            trainMax = parsedMax;
        maxEpisodeSteps = GlobalScale.SIMULATION_MODE ? trainMax : GlobalScale.MAX_EPISODE_STEPS;
        MaxStep = maxEpisodeSteps;

        radarRange = GlobalScale.RADAR_RANGE;
        // VESSEL_RADAR_RANGE env: 에이전트 레이더 범위만 축소(재빌드 없이) — "인지 부족" 통신 실험용.
        //   보상(true risk, COLREGS_DETECTION)은 그대로라 reduced radar면 못 보는 충돌을 통신이 메워야 함.
        string envRadar = System.Environment.GetEnvironmentVariable("VESSEL_RADAR_RANGE");
        if (!string.IsNullOrEmpty(envRadar) && float.TryParse(envRadar, out float parsedRadar) && parsedRadar > 0f)
            radarRange = parsedRadar;
        // VESSEL_COLCOURSE_COEF env: dense 충돌코스 페널티 계수 튜닝(재빌드 없이).
        string envCC = System.Environment.GetEnvironmentVariable("VESSEL_COLCOURSE_COEF");
        if (!string.IsNullOrEmpty(envCC) && float.TryParse(envCC, out float parsedCC))
            collisionCourseCoef = parsedCC;
        // VESSEL_COLCOURSE_EXP env: dense risk 지수(포화 제거 스티프닝). 1=옛 선형 동작(안전 기본).
        string envCE = System.Environment.GetEnvironmentVariable("VESSEL_COLCOURSE_EXP");
        if (!string.IsNullOrEmpty(envCE) && float.TryParse(envCE, out float parsedCE) && parsedCE > 0f)
            colcourseExp = parsedCE;
        // VESSEL_COLCOURSE_GATE env: dense 발화 게이트(스티프닝 시 저-risk 잡음 컷용, 기본 0.05).
        string envCG = System.Environment.GetEnvironmentVariable("VESSEL_COLCOURSE_GATE");
        if (!string.IsNullOrEmpty(envCG) && float.TryParse(envCG, out float parsedCG) && parsedCG >= 0f)
            colcourseGate = parsedCG;
        // ★ VESSEL_SPEED_AVOID_UNLOCK env: Rule-8 속도회피 언락(1=ON). 충돌코스 감속 lowSpeedPenalty 면제 +
        //   감속회피 earlyAvoid ×speedRatio floor. comm ON/OFF 동일 적용(anti-rigging). 기본 OFF=옛 동작 불변.
        enableSpeedAvoidUnlock = System.Environment.GetEnvironmentVariable("VESSEL_SPEED_AVOID_UNLOCK") == "1";
        // VESSEL_SPEED_UNLOCK_GATE env: 언락 발화 risk 게이트(기본 0.3 = 실질 충돌코스만). 낮추면 더 자주 면제.
        string envSUG = System.Environment.GetEnvironmentVariable("VESSEL_SPEED_UNLOCK_GATE");
        if (!string.IsNullOrEmpty(envSUG) && float.TryParse(envSUG, out float parsedSUG) && parsedSUG >= 0f)
            speedUnlockRiskGate = parsedSUG;
        // VESSEL_LOS_GATE env: 가림막 실험 시 1 (가려진 위협은 보상 제외 = anti-rigging).
        losGate = System.Environment.GetEnvironmentVariable("VESSEL_LOS_GATE") == "1";

        // ★ VESSEL_FARFIELD_COEF / VESSEL_RISK_RANGE env: far-field 조기회피 carrot(CTDE-lite).
        //   기본 coef 0 = off(현재와 비트동일). riskRange 기본 = COLREGS_DETECTION(56m) = 띠 없음 → far합 항상 0.
        //   riskRange를 통신범위(420m 등)로 키우고 coef>0면 56m 밖 먼 배 조기회피가 보상됨(통신 가치 창출).
        riskRange = GlobalScale.COLREGS_DETECTION;
        string envFFC = System.Environment.GetEnvironmentVariable("VESSEL_FARFIELD_COEF");
        if (!string.IsNullOrEmpty(envFFC) && float.TryParse(envFFC, out float parsedFFC) && parsedFFC >= 0f)
            farFieldCoef = parsedFFC;
        string envRR = System.Environment.GetEnvironmentVariable("VESSEL_RISK_RANGE");
        if (!string.IsNullOrEmpty(envRR) && float.TryParse(envRR, out float parsedRiskR) && parsedRiskR > 0f)
            riskRange = parsedRiskR;

        // VESSEL_PROX_RAMP_COEF / _DIST env: terminal-proximity ramp (terminal-magnitude fix).
        //   기본 0=off(안전). <0이면 발화. 시험권장 -2.5 (p_eq ~6%→~2%, EV math 산정). dist 기본=DCPA_RISK(24m).
        proxRampDist = GlobalScale.DCPA_RISK;
        string envPRC = System.Environment.GetEnvironmentVariable("VESSEL_PROX_RAMP_COEF");
        if (!string.IsNullOrEmpty(envPRC) && float.TryParse(envPRC, out float parsedPRC))
            proxRampCoef = parsedPRC;
        string envPRD = System.Environment.GetEnvironmentVariable("VESSEL_PROX_RAMP_DIST");
        if (!string.IsNullOrEmpty(envPRD) && float.TryParse(envPRD, out float parsedPRD) && parsedPRD > 0f)
            proxRampDist = parsedPRD;

        // ── 보상 재설계 env 토글 (재빌드 없이 ablation; 기본값=새 설계) ──
        //   진단으로 밝혀진 'rush>avoid' 원인은 path-dependent 항(angleReward·직진보너스·timePenalty)이
        //   회피 변침/detour를 직접 벌하는 것 → 그 항들을 줄이고 회피 EV(earlyAvoid)를 키움.
        //   progress(거리차)는 이미 telescoping(γ=1 PBS)이라 farming 없음 → 건드리지 않음(γ=0.99 PBS는 정지 farming 버그).
        string envAngle = System.Environment.GetEnvironmentVariable("VESSEL_ANGLE_COEF");
        angleRewardCoef = (!string.IsNullOrEmpty(envAngle) && float.TryParse(envAngle, out float pAngle)) ? pAngle : 0.15f;  // 0.5→0.15
        string envTime = System.Environment.GetEnvironmentVariable("VESSEL_TIME_PENALTY");
        timePenalty = (!string.IsNullOrEmpty(envTime) && float.TryParse(envTime, out float pTime)) ? pTime : -0.07f;        // -0.1→-0.07
        string envGate = System.Environment.GetEnvironmentVariable("VESSEL_EARLY_RISK_GATE");
        if (!string.IsNullOrEmpty(envGate) && float.TryParse(envGate, out float pGate)) earlyRiskGate = pGate;
        relaxEarlyTcpa = System.Environment.GetEnvironmentVariable("VESSEL_EARLY_RELAX_TCPA") != "0";
        enableStraightBonus = System.Environment.GetEnvironmentVariable("VESSEL_STRAIGHT_BONUS") == "1";

        // ── detour-cost 축소 토글 (재빌드 없이 ablation) ──
        //   진단: 회피 detour_cost는 arrival-discounting 100·γ^t_rem·(1−γ^D)이 지배(근접목표서 ~90%).
        //   이 peaky 종단보상이 "늦지만 안전한 도착"을 비싸게 만들어 rush>avoid EV를 형성(p_eq≈dc/300).
        //   수정: arrivalReward를 낮춰 종단 detour_cost를 *비례* 축소하되,
        //   잃은 goal-pull은 telescoping progress(거리차×goalDistanceCoef)로 이전.
        //   progress는 path-independent(총합=dist_start×coef, 정지/circling farming 불가)이라
        //   detour를 *벌하지 않으면서* navigation 견인을 유지 → timeout 회귀 방지.
        //   ⚠️ comm ON/OFF 동일 적용(보상 동일, 통신은 정보만) = anti-rigging.
        string envArrival = System.Environment.GetEnvironmentVariable("VESSEL_ARRIVAL_REWARD");
        if (!string.IsNullOrEmpty(envArrival) && float.TryParse(envArrival, out float pArrival) && pArrival > 0f)
            arrivalReward = pArrival;     // 기본 prefab 100 → 권장 45 (detour_cost ×0.45)
        string envGoalCoef = System.Environment.GetEnvironmentVariable("VESSEL_GOAL_COEF");
        if (!string.IsNullOrEmpty(envGoalCoef) && float.TryParse(envGoalCoef, out float pGoalCoef) && pGoalCoef > 0f)
            goalDistanceCoef = pGoalCoef; // 기본 1.0 → 권장 1.35 = (260−arrival)/160 (exact-isolation: 1.5는 과보전→레버 상쇄).

        // VESSEL_COLREGS_COEF env: COLREGs 준수 보상 계수 override (재빌드 후 적용). 브랜치별 실험 토글.
        //   baseline(1x)=0.30, reinforcement(1.5x)=0.45. comm ON/OFF 무관하게 동일 빌드로 토글(same-build anti-rigging).
        //   미설정 시 prefab/필드 기본값(0.45) 유지 — 브랜치 EXPERIMENT.md가 값을 명시.
        string envColregsCoef = System.Environment.GetEnvironmentVariable("VESSEL_COLREGS_COEF");
        if (!string.IsNullOrEmpty(envColregsCoef) && float.TryParse(envColregsCoef, out float pColregsCoef) && pColregsCoef >= 0f)
            colregsRewardCoef = pColregsCoef;

        radarObsSize = GlobalScale.RADAR_RAYS;   // 360 raw ray (min-pool 제거)
        goalNormK = GlobalScale.GOAL_NORM_K;
        goalReachedDistance = GlobalScale.GOAL_REACHED;
        waypointReachedDistance = GlobalScale.WAYPOINT_REACHED;
        proximityThreshold = radarRange * 0.35f;   // ~30m 유지 (radar 84m로 줄여도 회피 반응거리 보존: 0.25→0.35). 관측만 줄이고 회피는 고정
        proximityPenaltyCoef = -2.0f;   // prefab 무시 강제. 근접 시 goal-rush(+1.3) 확실히 이기게
        proximitySectorAngle = 135;     // prefab 무시 강제. 측면 위협(충돌 45%) 커버
        collisionPenalty = -300.0f;     // prefab(-100) 무시 강제. 충돌 분류 신뢰화(≪-150) + 회피 압력↑

        // 에디터 학습 시 BehaviorParameters Inspector 수동 세팅 불필요:
        // obs 차원을 GlobalScale 기준으로 강제. radarObsSize(360) + 9 (= goal2 + self4 + position2 + situation1) = 369D.
        // ARPA 제거(2026-06-05): 충돌기하는 360 raw ray + frame-stack(RadarEncoder)이 학습. situation1 = MoE 라우터(0~4). config.py OBSERVATION_SIZE와 일치.
        // 프레임 스태킹은 Python(frame_stack.py)이 담당하므로 Unity 스택은 1로 고정.
        var behaviorParams = GetComponent<BehaviorParameters>();
        if (behaviorParams != null)
        {
            behaviorParams.BrainParameters.VectorObservationSize = radarObsSize + 2 + 4 + 2 + 1;
            behaviorParams.BrainParameters.NumStackedVectorObservations = 1;
        }

        rb = GetComponent<Rigidbody>();
        if (rb == null) rb = gameObject.AddComponent<Rigidbody>();

        rb.useGravity = false;
        rb.linearDamping = 0.0f;
        rb.angularDamping = 0.0f;
        rb.constraints = RigidbodyConstraints.FreezePositionY |
                        RigidbodyConstraints.FreezeRotationX |
                        RigidbodyConstraints.FreezeRotationZ;

        initialPosition = transform.position;
        initialRotation = transform.rotation;

        if (vesselDynamics == null)
        {
            vesselDynamics = gameObject.GetComponent<VesselDynamics>();
            if (vesselDynamics == null) vesselDynamics = gameObject.AddComponent<VesselDynamics>();
        }
        vesselDynamics.Initialize(rb);

        // VESSEL_RUDDER_RATE env: 타속(°/s) override (재빌드 없이 깔작 vs 회피지연 튜닝). Awake의 GlobalScale 기본값 덮어씀.
        string envRudderRate = System.Environment.GetEnvironmentVariable("VESSEL_RUDDER_RATE");
        if (!string.IsNullOrEmpty(envRudderRate) && float.TryParse(envRudderRate, out float parsedRR) && parsedRR > 0f)
            vesselDynamics.rudderRate = parsedRR;

        // 계층 무관하게 VesselManager 찾기 (Unity 6의 non-obsolete API 사용)
        vesselManager = GetComponentInParent<VesselManager>();
        if (vesselManager == null)
            vesselManager = FindAnyObjectByType<VesselManager>();

        if (vesselManager != null)
        {
            cachedVessels = vesselManager.GetAllVesselAgents();
        }

        // 각 배는 자신의 속도가 랜덤이어야 한다.
        float speedMultiplier = Random.Range(0.8f, 1.8f);
        vesselDynamics.maxSpeed *= speedMultiplier;

        radar = gameObject.GetComponent<VesselRadar>();
        if (radar == null) radar = gameObject.AddComponent<VesselRadar>();

        radar.radarRange = radarRange;

        // detect layer는 전부 장애물임. collidor가 있는 경우에는 전부.
        // ★ 레이더 장님 버그 수정: prefab의 radarDetectionLayers가 비어(0) 있으면 VesselRadar의
        //   기본값 ~0(전체 레이어)을 덮어써서 레이더가 아무것도 감지 못 했음. 비어 있으면 전체 레이어로 fallback.
        radar.detectionLayers = (radarDetectionLayers.value != 0) ? radarDetectionLayers : (LayerMask)(~0);

        // Rule 17 추적을 위한 딕셔너리 초기화
        prevVesselStates = new Dictionary<GameObject, PrevVesselState>();
        lastTrackingTime = Time.time;

        // danger 캐시 초기화
        dangerCacheFrame = -1;
        cachedDangerousVessel = null;
        cachedDangerRisk = 0f;
        cachedDangerSituation = COLREGsHandler.CollisionSituation.None;
    }

    public override void OnEpisodeBegin()
    {
        // 직전 에피소드가 goal/collision 기록 없이 끝남 → MaxStep timeout (절단)
        if (episodeIndex > 0 && !outcomeLogged) LogOutcome("timeout");
        outcomeLogged = false;
        myStepCount = 0;
        episodeIndex++;

        vesselDynamics.ResetState();
        isCollided = false;

        // 웨이포인트 초기화
        currentWaypointIndex = 0;
        useWaypoints = false;
        waypoints = null;

        // cachedVessels 갱신 (에피소드 시작 시 stale 참조 방지)
        if (vesselManager != null)
        {
            cachedVessels = vesselManager.GetAllVesselAgents();
        }

        if (vesselManager != null)
        {
            vesselManager.RespawnVessel(gameObject);
        }
        else
        {
            Debug.LogWarning($"===== [RESET] {gameObject.name}: vesselManager NULL! rotation={initialRotation.eulerAngles} =====");
            transform.position = initialPosition;
            transform.rotation = initialRotation;
        }

        // RespawnVessel 이후에 초기 속도 설정 (ResetState가 targetSpeed=0으로 덮어쓰는 문제 방지)
        float initialSpeed = Random.Range(0.2f, 0.5f) * vesselDynamics.maxSpeed;
        vesselDynamics.SetTargetSpeed(initialSpeed);

        if (hasGoal) previousDistanceToGoal = Vector3.Distance(transform.position, goalPosition);

        // Rule 17 추적 초기화 (struct Dict 통합)
        prevVesselStates.Clear();
        lastTrackingTime = Time.time;

        // Smoothness reward 초기화
        previousRudderAngle = 0f;
        previousCommandedRudder = 0f;

        // danger 캐시 초기화 (이전 에피소드 데이터 무효화)
        dangerCacheFrame = -1;
        cachedDangerousVessel = null;
        cachedDangerRisk = 0f;
        cachedDangerSituation = COLREGsHandler.CollisionSituation.None;
        cachedFarRiskSum = 0f;
        prevFarRiskSum = -1f;   // far-field carrot: 첫 프레임 스킵 sentinel (prevDcpa 패턴)

        // 메트릭/early-avoid 초기화 (이전 에피소드 잔존값 제거)
        prevDcpa = -1f;
        episodeFuel = 0f;
        episodeRudderVar = 0f;
        episodeCommandVar = 0f;
        colregsComplianceSum = 0f;
        colregsComplianceCount = 0;
        threatSteps = 0;
        occludedThreatCount = 0;

        // ── 진단 instrumentation 리셋 (respawn 후 위치/heading 기준) ──
        episodeSpawnPos = transform.position;
        lastPathPos = transform.position;
        prevHeadingForTravel = transform.eulerAngles.y;
        episodePathLength = 0f;
        episodeHeadingTravel = 0f;
        episodeMinVesselDist = float.MaxValue;
        currentMinVesselDist = float.MaxValue;
        nearMissSteps = 0;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        myStepCount++;
        try
        {
            float targetRudderAngle = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f) * vesselDynamics.maxTurnRate;
            float targetThrust = (Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f) + 1f) / 2f * vesselDynamics.maxSpeed;

            vesselDynamics.SetRudderAngle(targetRudderAngle);
            vesselDynamics.SetTargetSpeed(targetThrust);
            vesselDynamics.SetBraking(false);

            CalculateReward();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[OnAction] {gameObject.name}: {e.Message}\n{e.StackTrace}");
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;

        // 학습 모드에서는 사용되지 않음
        // Python과 연결되지 않을 때만 호출됨
        continuousActionsOut[0] = 0f;  // 타각
        continuousActionsOut[1] = 0f;  // 속도
    }

    private void CalculateReward()
    {
        // 0. Time penalty (빨리 끝내도록 유도)
        AddReward(timePenalty);

        // 0-1. Forward speed bonus + Low speed penalty
        float speedRatio = vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed;
        AddReward(forwardSpeedBonus * speedRatio);

        // ★ Rule-8 속도회피 언락(언락1): 실제 충돌코스(접근중 risk>게이트)면 lowSpeedPenalty 면제 →
        //   감속을 회피 선택지로 허용(돌진 강제 해소). UpdateDangerCache는 frameCount 캐시라 중복비용 0.
        //   risk는 rawTCPA<0(후퇴/통과)면 0 하드컷 → 자유수면/정지·후퇴엔 면제 안 됨 = loiter/freeze farming 불가.
        UpdateDangerCache();
        bool onClosingCourse = enableSpeedAvoidUnlock && cachedDangerRisk > speedUnlockRiskGate;
        if (speedRatio < lowSpeedThreshold)
        {
            // ★언락 시 충돌코스 감속은 페널티 ×0.3로 *완화*(완전 면제 X) → Rule-8 감속을 회피 선택지로
            //   허용하되, parked-to-zero 자기지속 freeze 평형은 차단(qa 검증: 완전면제면 정지가 게이트 안에서
            //   risk>0 유지→면제 영속=timid 재발). 언락 OFF면 onClosingCourse=false → 옛 동작과 100% 동일.
            AddReward(onClosingCourse ? lowSpeedPenalty * 0.3f : lowSpeedPenalty);
        }

        // 0-1b. Fuel proxy penalty (연료 논제: thrust²+0.5·turn² → 멀리서 부드럽게 회피해야 이득)
        // ⚠️ comm ON/OFF 동일 적용(보상은 같고, 통신은 정보만 차이) → 연료 차이가 통신 효과의 증거
        // ⚠️ 슬루 도입 후: 보상-shaping은 *명령* 타각 기반(실제 타각은 슬루로 평탄화돼 gradient 소실).
        float turn01 = Mathf.Abs(vesselDynamics.CommandedRudderAngle / vesselDynamics.maxTurnRate);
        float fuel = speedRatio * speedRatio + 0.5f * turn01 * turn01;
        AddReward(fuelCoef * fuel);
        episodeFuel += fuel;

        // 0-2. Proximity penalty (전방 장애물 회피 유도)
        CalculateProximityReward();

        // 0-3. ★ Dense 충돌코스 페널티 (true risk 기반, 매 스텝, ungated) — U자 회귀 방지.
        //   충돌코스에 지속 비용 부과 → 회피를 수렴까지 유지(돌진 보상에 안 밀림).
        //   risk는 접근중(rawTCPA>0)일 때만 >0 → 정지/후퇴엔 페널티 거의 없음(freeze 유인 아님).
        //   계수는 VESSEL_COLCOURSE_COEF로 튜닝.
        UpdateDangerCache();
        if (cachedDangerRisk > colcourseGate)
        {
            // ★ dense-steepening (saturation fix): risk가 충돌코스에서 56m부터 1.0으로 포화 →
            //   per-step 페널티가 거리-무관 flat -0.8 → PPO가 "조기 변침 > 늦은 변침" 신호를 못 받음.
            //   risk^p (p>1)로 비선형화 → 진짜 고risk(임박 충돌)만 강하게, 중간 DCPA는 부드럽게 →
            //   거리 gradient 복원 + 관측된 sudden-tail(moderate-DCPA) 충돌의 적분 비용을 충돌 전에 키움.
            //   risk는 rawTCPA<0(후퇴)·큰 DCPA에서 0 → 정상 DCPA 궤도(orbit)는 페널티≈0 = circling 자기제한.
            float shapedRisk = Mathf.Pow(cachedDangerRisk, colcourseExp);
            AddReward(collisionCourseCoef * shapedRisk);
        }

        // 0-3c. ★ Far-field 조기회피 carrot (CTDE-lite, 기본 off). 56m~riskRange 먼 배 risk *합*이
        //   줄어들 때(=먼 위협을 미리 회피)만 보상 → comm-ON이 옆 배가 전한 먼 위협에 조기행동하게.
        //   ★×speedRatio (near earlyAvoid와 동일): 감속으로 far-risk 줄여 따먹는 걸 차단 → 전진하며
        //   *돌려서* 회피해야 보상 = 먼 배 방위를 알아야 함 = 메시지(통신) 필요 → 게이트 개방 유도.
        //   carrot이라 baseline 벌점 X. UpdateDangerCache(477)가 cachedFarRiskSum 채움. 첫 프레임(prev=-1) 스킵.
        if (farFieldCoef > 0f && prevFarRiskSum >= 0f)
        {
            float farGain = prevFarRiskSum - cachedFarRiskSum;   // + = 먼 위협 줄임(조기회피)
            if (farGain > 0f) AddReward(farFieldCoef * farGain * speedRatio);
        }
        prevFarRiskSum = cachedFarRiskSum;

        // 0-3b. ★ Terminal-proximity ramp (terminal-magnitude fix, 기본 off — env 토글).
        //   목적: 희소 -300 충돌 종료를 *확실·저분산*의 거리×접근속도 그라디언트로 변환.
        //   dense risk(0-3)는 TCPA/DCPA 기반이라 56m부터 포화(flat) → "늦게 꺾어도 같은 비용" → EV상 돌진 허용.
        //   이 항은 *물리적 근접거리×접근율*만 사용(포화 없는 거리 gradient) → 충돌코스를 마지막 접근에서
        //   매 스텝 확실히 벌해 회피 detour_cost(~19)를 넘기는 marginal 비용을 부과 → p_eq를 ~6%→~2%로 이동.
        //   anti-rigging: comm ON/OFF 동일(상대 vessel 실제거리 기반, 메시지 무관). value-of-info≥0 불변.
        //   freeze 방지: closing>0(접근중)일 때만 발화 → 정지·후퇴·orbit(DCPA 증가)엔 0 → timid/circling 유인 없음.
        // ★다선 정합성 fix: 거리·근접·접근율을 모두 *동일 선박*(cachedDangerousVessel=최고위험)으로 통일.
        //   (옛 버그: 게이트·proximity01은 currentMinVesselDist=전역최근접, closing은 위험선박 → 서로 다른 배의
        //   기하를 곱해 무의미. 또 currentMinVesselDist는 LOS 가림 무시 → VESSEL_LOS_GATE=1서 안 보이는 위협을
        //   벌해 anti-rigging 위반. cachedDangerousVessel은 1053행에서 occluded면 continue → LOS-게이트됨.)
        if (proxRampCoef < 0f && cachedDangerousVessel != null)
        {
            // 상대거리의 접근율(closing rate): 접근중(+)일 때만 비용.
            Vector3 toOther = cachedDangerousVessel.transform.position - transform.position;
            float sep = toOther.magnitude;
            if (sep < proxRampDist && sep > 1e-3f)
            {
                Vector3 relVel = (cachedDangerousVessel.transform.forward * cachedDangerousVessel.vesselDynamics.CurrentSpeed)
                               - (transform.forward * vesselDynamics.CurrentSpeed);
                float closing = -Vector3.Dot(relVel, toOther / sep);   // +면 접근, -면 이탈
                if (closing > 0f)
                {
                    // proximity01: proxRampDist에서 0 → 0거리에서 1 (선형, 포화 없음 → 거리 gradient 유지)
                    //   ★sep(위험선박 실거리) 사용 — closing과 동일 선박이라 곱이 일관(다선 정합성).
                    float proximity01 = 1f - (sep / proxRampDist);
                    // closing01: 정면 합산 최대 접근율(2·maxSpeed)로 정규화
                    float closing01 = Mathf.Clamp01(closing / (2f * vesselDynamics.maxSpeed));
                    // 비용 = coef · proximity² · closing  (proximity² → 마지막 접근에서 가파르게, 먼 거리엔 약하게)
                    AddReward(proxRampCoef * proximity01 * proximity01 * closing01);
                }
            }
        }

        // ── 진단 instrumentation 갱신 (★로그 전용, 보상 비연결) ──
        episodePathLength += Vector3.Distance(transform.position, lastPathPos);
        lastPathPos = transform.position;
        float curHeading = transform.eulerAngles.y;
        episodeHeadingTravel += Mathf.Abs(Mathf.DeltaAngle(prevHeadingForTravel, curHeading));
        prevHeadingForTravel = curHeading;
        if (currentMinVesselDist < episodeMinVesselDist) episodeMinVesselDist = currentMinVesselDist;
        if (currentMinVesselDist < GlobalScale.NEAR_MISS_DIST) nearMissSteps++;

        // 1. Navigation reward (목표 도달 + 진행 + 방향)
        if (CalculateNavigationReward(speedRatio)) return;  // 에피소드 종료됨

        // 2. COLREGs compliance (위험도 기반)
        CalculateColregsReward();

        // 3. Smoothness reward (comm ON/OFF 항상 동일 적용 — 공정성)
        CalculateSmoothnessReward();

        // 추적 시간 업데이트
        lastTrackingTime = Time.time;
    }

    /// <summary>
    /// 목표 도달, 진행도, 방향 보상 계산
    /// </summary>
    /// <returns>에피소드가 종료되었으면 true</returns>
    private bool CalculateNavigationReward(float speedRatio)
    {
        if (!hasGoal) return false;

        float currentDistanceToGoal = Vector3.Distance(transform.position, goalPosition);

        // 도착 판정: 웨이포인트 모드에서는 중간/최종 구분
        bool isLastWaypoint = !useWaypoints || currentWaypointIndex >= waypoints.Count - 1;
        float reachDistance = isLastWaypoint ? goalReachedDistance : waypointReachedDistance;

        if (currentDistanceToGoal < reachDistance)
        {
            if (isLastWaypoint)
            {
                // 최종 목표 도착 → 에피소드 종료
                AddReward(arrivalReward);
                LogOutcome("goal");
                outcomeLogged = true;
                EndEpisode();
                return true;  // 에피소드 종료
            }
            else
            {
                // 중간 웨이포인트 도착 → 다음 웨이포인트로 전환
                AddReward(intermediateWaypointReward);
                AdvanceToNextWaypoint();
            }
        }

        float distanceChange = previousDistanceToGoal - currentDistanceToGoal;
        float goalProgressReward = distanceChange * goalDistanceCoef;
        AddReward(goalProgressReward);

        // 방향 보상: 목적지를 향하면서 전진할 때만 보상 (제자리 회전 방지)
        Vector3 toGoal = goalPosition - transform.position;
        float goalAngle = Vector3.SignedAngle(transform.forward, toGoal, Vector3.up);
        float cosAngle = Mathf.Cos(goalAngle * Mathf.Deg2Rad);
        float angleReward = cosAngle * angleRewardCoef * speedRatio;
        AddReward(angleReward);

        // 직진 보너스: 목표 향하며 직진할 때만 (빙빙 도는 것 방지)
        float rudderRatio = Mathf.Abs(vesselDynamics.RudderAngle / vesselDynamics.maxTurnRate);
        if (enableStraightBonus && rudderRatio < 0.1f && cosAngle > 0.5f)
        {
            AddReward(forwardSpeedBonus);  // 직진 보너스 (기본 off: rudder≈0 보상이 회피 변침과 충돌)
        }

        previousDistanceToGoal = currentDistanceToGoal;
        return false;  // 계속 진행
    }

    /// <summary>
    /// COLREGs 준수도 보상 계산 (가장 위험한 선박 하나만 처리)
    /// </summary>
    private void CalculateColregsReward()
    {
        if (!enableColregsReward || cachedVessels == null) return;

        // Per-frame 캐시 사용 (CollectObservations와 동일한 결과 공유)
        UpdateDangerCache();
        VesselAgent mostDangerousVessel = cachedDangerousVessel;
        float maxRisk = cachedDangerRisk;

        // ★게이트 완화: earlyRiskGate(기본 0.1)로 낮춰 근접 위협에서도 DCPA-증가 회피보상이 흐르게.
        //   (COLREGs 준수보상은 아래에서 별도로 maxRisk>0.3 게이트 유지 — 기존 동작 보존)
        if (mostDangerousVessel == null || maxRisk <= earlyRiskGate) { prevDcpa = -1f; return; }

        var situation = cachedDangerSituation;
        if (situation == COLREGsHandler.CollisionSituation.None) { prevDcpa = -1f; return; }

        float speedRatio = vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed;

        // TCPA/DCPA 계산
        Vector3 myVelocity = transform.forward * vesselDynamics.CurrentSpeed;
        Vector3 otherVelocity = mostDangerousVessel.transform.forward * mostDangerousVessel.vesselDynamics.CurrentSpeed;
        float tcpa = COLREGsHandler.CalculateTCPA(
            transform.position, myVelocity,
            mostDangerousVessel.transform.position, otherVelocity
        );
        float dcpa = COLREGsHandler.CalculateDCPA(
            transform.position, myVelocity,
            mostDangerousVessel.transform.position, otherVelocity
        );

        float riskWeight = 1.0f + maxRisk;  // 1.0 ~ 2.0

        // ── COLREGs 준수 보상: 위험도 높을 때만(maxRisk>0.3 — 기존 동작 유지) ──
        if (maxRisk > 0.3f)
        {
            // 상대 선박의 회피 행동 감지 (Rule 17을 위해)
            bool otherVesselTakingAction = false;
            GameObject otherVesselObj = mostDangerousVessel.gameObject;
            float deltaTime = Time.time - lastTrackingTime;

            // 해시 1회 조회로 3개 필드 모두 획득 (기존 ContainsKey + 3x indexer = 4회)
            if (deltaTime > 0.1f && prevVesselStates.TryGetValue(otherVesselObj, out PrevVesselState prev))
            {
                otherVesselTakingAction = COLREGsHandler.IsVesselTakingAvoidanceAction(
                    prev.position,
                    mostDangerousVessel.transform.position,
                    prev.forward,
                    mostDangerousVessel.transform.forward,
                    prev.speed,
                    mostDangerousVessel.vesselDynamics.CurrentSpeed,
                    deltaTime
                );
            }
            else if (mostDangerousVessel.vesselDynamics != null)
            {
                otherVesselTakingAction =
                    Mathf.Abs(mostDangerousVessel.vesselDynamics.RudderAngle) > 0.3f ||
                    mostDangerousVessel.vesselDynamics.CurrentSpeed < mostDangerousVessel.vesselDynamics.maxSpeed * 0.7f;
            }

            // 현재 상태 저장 (struct 1회 대입)
            prevVesselStates[otherVesselObj] = new PrevVesselState
            {
                position = mostDangerousVessel.transform.position,
                forward = mostDangerousVessel.transform.forward,
                speed = mostDangerousVessel.vesselDynamics.CurrentSpeed
            };

            // 권장 행동 계산
            var (recommendedRudder, recommendedSpeed) = COLREGsHandler.GetRecommendedAction(
                situation,
                vesselDynamics.CurrentSpeed,
                mostDangerousVessel.transform.position - transform.position,
                tcpa,
                dcpa,
                otherVesselTakingAction
            );

            // COLREGs 준수도 평가
            float compliance = COLREGsHandler.EvaluateCompliance(
                situation,
                vesselDynamics.RudderAngle,
                recommendedRudder,
                vesselDynamics.maxTurnRate,
                vesselDynamics.CurrentSpeed,
                recommendedSpeed,
                tcpa,
                dcpa
            );

            // 위험도에 비례한 보상 (위험할수록 COLREGs 준수가 더 중요)
            AddReward(compliance * colregsRewardCoef * riskWeight);

            // 메트릭 누적 (ground-truth COLREGs 준수도, comm ON/OFF 비교용)
            colregsComplianceSum += compliance;
            colregsComplianceCount++;
        }

        // ── 조기/근접 회피 보상: DCPA를 벌릴 때(+)만 → 회피 EV를 +로, orbit(DCPA정체)=보상0.
        //   게이트 완화(relaxEarlyTcpa면 any tcpa) + ×speedRatio(정지로 게임 못 함, 전진 회피만 보상).
        //   → '거리유지 매스텝 지급'이 아닌 '통과 완료 이벤트'로 회피를 바꿔 돌진편향·orbit 동시 완화. ──
        if (prevDcpa >= 0f && (relaxEarlyTcpa || tcpa > GlobalScale.SUBSTANTIAL_ACTION_TIME))
        {
            float dcpaGain = dcpa - prevDcpa;   // + = 최근접거리 벌어짐(안전해짐)
            if (dcpaGain > 0f)
            {
                // ★언락2: 충돌코스(maxRisk>게이트)에선 ×speedRatio 가중을 floor(lowSpeedThreshold)로 끌어올려
                //   감속회피(타가 아닌 속도로 DCPA를 키움)도 거의 full 보상 → "감속=손해"를 제거.
                //   발화 전제가 이미 maxRisk>earlyRiskGate(접근중 위협)+dcpaGain>0(실제 안전화)라 loiter farming 불가.
                float avoidSpeedWeight = (enableSpeedAvoidUnlock && maxRisk > speedUnlockRiskGate)
                    ? Mathf.Max(speedRatio, lowSpeedThreshold)
                    : speedRatio;
                AddReward(earlyAvoidCoef * Mathf.Clamp01(dcpaGain / GlobalScale.DCPA_RISK) * riskWeight * avoidSpeedWeight);
            }
        }
        prevDcpa = dcpa;
    }

    /// <summary>
    /// 타각 변화에 대한 부드러움 패널티 (comm ON/OFF 항상 동일 — 보상 공정성, 통신은 정보만 차이)
    /// </summary>
    private void CalculateSmoothnessReward()
    {
        // 실제 타각 변화(슬루로 ≤rudderRate·dt 제한됨) — 약한 calibration 패널티 + 메트릭
        float rudderChange = Mathf.Abs(vesselDynamics.RudderAngle - previousRudderAngle);
        float normalizedChange = rudderChange / vesselDynamics.maxTurnRate;
        episodeRudderVar += normalizedChange;

        // 명령 타각 변화 — 슬루에도 안 죽는 "의도" 메트릭(통신 ON/OFF 부드러움 비교의 진짜 신호)
        float cmdChange = Mathf.Abs(vesselDynamics.CommandedRudderAngle - previousCommandedRudder);
        episodeCommandVar += cmdChange / vesselDynamics.maxTurnRate;

        // 타속 한계 포화: 명령이 실제보다 과도(타를 못 따라올 만큼)하면 비효율 → 패널티(슬루 후 새로 필요)
        float saturation = Mathf.Clamp01(
            Mathf.Abs(vesselDynamics.CommandedRudderAngle - vesselDynamics.RudderAngle) / vesselDynamics.maxTurnRate);

        if (enableSmoothnessReward)
        {
            AddReward(smoothnessCoef * normalizedChange);
            AddReward(commandMismatchCoef * saturation);
        }
        previousRudderAngle = vesselDynamics.RudderAngle;
        previousCommandedRudder = vesselDynamics.CommandedRudderAngle;
    }

    /// <summary>
    /// 전방+측면 장애물 근접 패널티 (±proximitySectorAngle° 섹터, Initialize에서 135로 강제 → ±135° 270° 커버)
    /// </summary>
    private void CalculateProximityReward()
    {
        if (radar == null) return;

        float frontMinDist = radar.GetMinFrontDistance(proximitySectorAngle);

        // 전방 장애물이 threshold 이내면 선형 패널티
        if (frontMinDist < proximityThreshold)
        {
            float normalizedProximity = 1.0f - (frontMinDist / proximityThreshold); // 0~1
            AddReward(proximityPenaltyCoef * normalizedProximity);
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        HandleCollision(collision.gameObject);
    }

    void OnTriggerEnter(Collider other)
    {
        HandleCollision(other.gameObject);
    }

    /// <summary>
    /// 에피소드 종료 결과를 per-run 파일에 직접 기록 (reward 추론 대체, ground truth).
    /// 형식: agentId,episodeIndex,outcome,stepCount  | outcome ∈ {goal,collision_vessel,collision_obstacle,timeout}
    /// 파일 경로 = env VESSEL_OUTCOME_LOG (run별 분리), 없으면 빌드폴더/outcomes.csv.
    /// </summary>
    private void LogOutcome(string outcome)
    {
        try
        {
            if (!_outcomePathInit)
            {
                _outcomeLogPath = System.Environment.GetEnvironmentVariable("VESSEL_OUTCOME_LOG");
                if (string.IsNullOrEmpty(_outcomeLogPath))
                    _outcomeLogPath = System.IO.Path.Combine(Application.dataPath, "..", "outcomes.csv");
                _outcomePathInit = true;
            }
            System.IO.File.AppendAllText(_outcomeLogPath,
                $"{GetInstanceID()},{episodeIndex},{outcome},{myStepCount}\n");

            // ── per-episode 메트릭 ground-truth 로깅 (comm ON/OFF 비교) ──
            // 형식(13열): agentId,episodeIndex,outcome,steps,fuel,rudderVar,complianceMean,occlRate,commandVar,
            //            minVesselDist,nearMissSteps,straightness,headingTravel
            //   ★뒤 4열=진단 전용(near-miss/circling), 보상에 절대 비연결(제약3). straightness=순변위/경로길이(낮을수록 빙빙).
            if (!_metricPathInit)
            {
                _metricLogPath = System.Environment.GetEnvironmentVariable("VESSEL_METRIC_LOG");
                if (string.IsNullOrEmpty(_metricLogPath))
                    _metricLogPath = System.IO.Path.Combine(Application.dataPath, "..", "metrics.csv");
                _metricPathInit = true;
            }
            float compMean = colregsComplianceCount > 0 ? colregsComplianceSum / colregsComplianceCount : 0f;
            float occlRate = threatSteps > 0 ? (float)occludedThreatCount / threatSteps : 0f;
            float netDisp = Vector3.Distance(transform.position, episodeSpawnPos);
            float straightness = episodePathLength > 0.01f ? Mathf.Clamp01(netDisp / episodePathLength) : 1f;
            float minVD = episodeMinVesselDist < float.MaxValue ? episodeMinVesselDist : -1f;
            System.IO.File.AppendAllText(_metricLogPath,
                $"{GetInstanceID()},{episodeIndex},{outcome},{myStepCount},{episodeFuel:F3},{episodeRudderVar:F3},{compMean:F3},{occlRate:F3},{episodeCommandVar:F3},{minVD:F3},{nearMissSteps},{straightness:F4},{episodeHeadingTravel:F1}\n");

            // ── 종료 이벤트 위치 로깅(VESSEL_EVENT_LOG, 선택적) — 충돌/도착의 *공간 분포*(궤적 분석) ──
            // 형식: agentId,ep,outcome,step,startX,startZ,endX,endZ,heading,speed
            //   충돌이 원통 근처/교차 중심에 몰리는지, 어떤 heading·속도에서 나는지 사후 분석.
            string evPath = System.Environment.GetEnvironmentVariable("VESSEL_EVENT_LOG");
            if (!string.IsNullOrEmpty(evPath))
            {
                Vector3 p = transform.position;
                System.IO.File.AppendAllText(evPath,
                    $"{GetInstanceID()},{episodeIndex},{outcome},{myStepCount},{episodeSpawnPos.x:F2},{episodeSpawnPos.z:F2},{p.x:F2},{p.z:F2},{transform.eulerAngles.y:F1},{vesselDynamics.CurrentSpeed:F3}\n");
            }
        }
        catch { }
    }

    private void HandleCollision(GameObject collidedObject)
    {
        if (isCollided) return;

        bool isVessel = collidedObject.GetComponent<VesselAgent>() != null;
        bool isObstacle = collidedObject.CompareTag("Obstacle");

        if (isVessel || isObstacle)
        {
            LogOutcome(isVessel ? "collision_vessel" : "collision_obstacle");
            outcomeLogged = true;

            isCollided = true;
            AddReward(collisionPenalty);
            EndEpisode();
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        radar.ScanRadar();

        if (!hasGoal)
        {
            // 369D = radar(360 raw ray) + goal(2) + self(4) + position(2) + situation(1)
            // ⚠️ zero-fill 개수는 VectorObservationSize와 정확히 일치해야 함(불일치 시 ML-Agents 연결 shape 에러)
            for (int i = 0; i < radarObsSize + 2 + 4 + 2 + 1; i++) sensor.AddObservation(0f);
            return;
        }

        // ========== 1. Radar (360 raw ray) ==========
        // ★min-pool(360→30 섹터) 제거(2026-06-04): 360 raw ray를 그대로 송신 → Python RadarEncoder(Conv1D)가
        //   학습형 압축. 정보손실 없이 "무엇을 남길지"를 신경망이 결정. 정규화 = dist/range-0.5(미감지 0.5).
        float[] rayDistances = radar.GetAllRayDistances();
        sensor.AddObservation(rayDistances);

        // ========== 2. Self State (6D) ==========
        Vector3 toGoal = goalPosition - transform.position;
        float goalDistance = toGoal.magnitude;
        float goalAngle = Vector3.SignedAngle(transform.forward, toGoal, Vector3.up);

        sensor.AddObservation(goalDistance / (goalDistance + goalNormK));               // Goal distance (비선형 d/(d+k), [0,1) saturate 없음)
        sensor.AddObservation(goalAngle / 180f);                                       // Goal angle
        sensor.AddObservation(vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed);  // Linear velocity
        sensor.AddObservation(vesselDynamics.YawRate / vesselDynamics.MaxYawRate);    // Angular velocity (실제 최대 yawRate로 정규화)

        // Heading을 -180~180 범위로 변환 후 정규화 (0°와 360°가 같은 값이 되도록)
        float heading = transform.eulerAngles.y;
        float headingNormalized = heading > 180f ? heading - 360f : heading;  // -180 ~ 180
        sensor.AddObservation(headingNormalized / 180f);                               // Heading (-1 ~ 1)
        sensor.AddObservation(vesselDynamics.RudderAngle / vesselDynamics.maxTurnRate); // Rudder angle

        // ========== 3. Position (2D) - 통신 범위 계산용, 학습에서 제외 ==========
        sensor.AddObservation(transform.position.x);
        sensor.AddObservation(transform.position.z);

        // ========== 4. Situation (1D) - COLREGs MoE 라우터, 학습 feature 제외 ==========
        // cachedDangerSituation = 최고위험 위협 1척의 COLREGs 상황(0=None,1=HeadOn,2=StandOn,3=GiveWay,4=Overtaking).
        // ⚠️ 보상이 채우는 캐시라 여기선 1스텝 stale(UpdateDangerCache는 보상경로 전용, leak 방지 — obs path 미호출).
        //   COLREGs 상황은 초 단위 유지라 0.4s 지연 무시 가능. Python이 MoE head 라우팅에만 사용(네트워크 feature 아님).
        sensor.AddObservation((float)(int)cachedDangerSituation);

        // 총 관측 차원: 360 (radar raw ray) + 2 (goal) + 4 (self) + 2 (position) + 1 (situation) = 369D
        // position·situation은 Python 라우팅/통신 계산용으로만 사용 (네트워크 입력 제외)
    }

    public void SetGoal(Vector3 position)
    {
        goalPosition = position;
        hasGoal = true;
        previousDistanceToGoal = Vector3.Distance(transform.position, goalPosition);
    }

    /// <summary>
    /// 웨이포인트 리스트 설정. 첫 웨이포인트를 현재 목표로 설정.
    /// </summary>
    public void SetWaypoints(List<Vector3> newWaypoints)
    {
        if (newWaypoints == null || newWaypoints.Count == 0)
        {
            useWaypoints = false;
            return;
        }

        waypoints = new List<Vector3>(newWaypoints);
        currentWaypointIndex = 0;
        useWaypoints = true;

        // 첫 웨이포인트를 목표로 설정
        SetGoal(waypoints[0]);
    }

    /// <summary>
    /// 다음 웨이포인트로 전환. goalPosition을 갱신하고 previousDistanceToGoal 재계산.
    /// </summary>
    private void AdvanceToNextWaypoint()
    {
        currentWaypointIndex++;
        if (currentWaypointIndex < waypoints.Count)
        {
            goalPosition = waypoints[currentWaypointIndex];
            previousDistanceToGoal = Vector3.Distance(transform.position, goalPosition);
        }
    }

    private void FixedUpdate()
    {
        try
        {
            vesselDynamics.UpdateDynamics(Time.fixedDeltaTime);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[FixedUpdate] {gameObject.name}: {e.Message}\n{e.StackTrace}");
        }
    }

    /// <summary>
    /// 가장 위험한 선박과 상황을 per-frame 캐시로 계산 (CollectObservations + CalculateReward 공유)
    /// CalculateRiskWithSituation을 사용하여 AnalyzeSituation 이중 호출 방지
    /// </summary>
    private void UpdateDangerCache()
    {
        if (Time.frameCount == dangerCacheFrame) return;
        dangerCacheFrame = Time.frameCount;

        cachedDangerousVessel = null;
        cachedDangerRisk = 0f;
        cachedDangerSituation = COLREGsHandler.CollisionSituation.None;
        currentMinVesselDist = float.MaxValue;   // 진단: 이번 프레임 최근접 타선거리(near-miss 측정)
        cachedFarRiskSum = 0f;                    // far-field 조기회피: 이번 프레임 먼 배 risk 합 리셋

        if (cachedVessels == null) return;

        float commRange = GlobalScale.COMM_RANGE;
        float rayH = radar != null ? radar.rayHeight : GlobalScale.RAY_HEIGHT;
        Vector3 origin = transform.position + Vector3.up * rayH;

        foreach (var otherVessel in cachedVessels)
        {
            if (otherVessel == this) continue;

            // ── LOS 검사: 자신→상대 사이에 장애물(원통 등)이 있으면 가려짐(occluded) ──
            Vector3 toOther = otherVessel.transform.position - transform.position;
            float dist = toOther.magnitude;
            if (dist < currentMinVesselDist) currentMinVesselDist = dist;   // 진단: 최근접 타선거리
            bool occluded = false;
            if (dist > 0.01f &&
                Physics.Raycast(origin, toOther / dist, out RaycastHit losHit, dist, ~0))
            {
                // 첫 충돌이 상대 선박이 아니면 = 사이에 다른 게 막음 = 가려짐
                if (losHit.collider.gameObject != otherVessel.gameObject)
                    occluded = true;
            }

            // occlusion 비율 측정 (comm 범위 내 위협 중 가려진 비율 → 원통 크기 튜닝용)
            if (dist <= commRange)
            {
                threatSteps++;
                if (occluded) occludedThreatCount++;
            }

            // ★ LOS 게이트: 가려진 위협은 보상 risk에서 제외 (레이더로 못 보는 걸로 안 벌줌)
            if (losGate && occluded) continue;

            // ★ Far-field 조기회피: 56m~riskRange 먼 배의 충돌코스 risk 합산(근거리 max와 분리, 띠 disjoint).
            //   coef 0(기본)이면 스킵 → 비트동일·비용0. occluded는 위에서 이미 continue됨(LOS 일관).
            if (farFieldCoef > 0f)
            {
                cachedFarRiskSum += COLREGsHandler.CalculateFarFieldRisk(
                    transform.position, transform.forward, vesselDynamics.CurrentSpeed,
                    otherVessel.transform.position, otherVessel.transform.forward,
                    otherVessel.vesselDynamics.CurrentSpeed,
                    GlobalScale.COLREGS_DETECTION, riskRange);
            }

            var (risk, situation) = COLREGsHandler.CalculateRiskWithSituation(
                transform.position, transform.forward, vesselDynamics.CurrentSpeed,
                otherVessel.transform.position, otherVessel.transform.forward,
                otherVessel.vesselDynamics.CurrentSpeed
            );

            if (risk > cachedDangerRisk)
            {
                cachedDangerRisk = risk;
                cachedDangerousVessel = otherVessel;
                cachedDangerSituation = situation;
            }
        }
    }
}
