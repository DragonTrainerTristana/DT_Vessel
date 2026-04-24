using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class VesselAgent : Agent
{
    // Collider → VesselAgent O(1) 조회. Radar의 GetComponent 360회/vessel 제거용.
    private static readonly Dictionary<Collider, VesselAgent> _colliderToAgent = new Dictionary<Collider, VesselAgent>();
    public static bool IsVesselCollider(Collider c) => c != null && _colliderToAgent.ContainsKey(c);

    public VesselDynamics vesselDynamics;

    [Header("Episode Settings")]
    public int maxEpisodeSteps = 15000;           // 학습: 15000, 시연: 50000

    public float arrivalReward = 100.0f;          // 도착 보상 강화
    public float goalDistanceCoef = 1.0f;         // 0.5 → 1.0 (progress 보상 강화)
    public float collisionPenalty = -100.0f;       // 충돌 패널티 유지
    public float colregsRewardCoef = 0.45f;       // COLREGs 보상 계수 (1.5x = Phase 3)
    public bool enableColregsReward = true;       // COLREGs 활성화 (우현 보상 제거됨, 좌현 패널티만)
    public float angleRewardCoef = 0.5f;          // 목적지 방향 보상 계수 (강화: 0.2 → 0.5)
    public float forwardSpeedBonus = 0.1f;        // 전진 보상 (강화: 0.05 → 0.1)

    public float timePenalty = -0.1f;             // 매 스텝마다 패널티 (강화: -0.03 → -0.1, 빙빙 방지)

    // Low speed penalty - 제자리 정지 방지
    public float lowSpeedThreshold = 0.2f;        // 20% maxSpeed 미만이면 패널티 (COLREGs 감속과 호환)
    public float lowSpeedPenalty = -0.15f;        // 저속 패널티 (강화: -0.1 → -0.15)

    public float goalReachedDistance = 1.5f;    // 1/10 스케일 (원본 15m)
    public float maxMapDistance = 100f;          // 맵 최대 거리 (goal distance 정규화용, 1/10 스케일)

    public Vector3 goalPosition;
    public bool hasGoal = false;
    public string goalPointName;
    public int goalPointIndex;
    private float previousDistanceToGoal;

    [Header("Waypoint Navigation")]
    private List<Vector3> waypoints;
    private int currentWaypointIndex = 0;
    public float waypointReachedDistance = 2f;      // 중간 웨이포인트 도달 거리 (1/10 스케일, 원본 20m)
    public float intermediateWaypointReward = 15f;   // 중간 도착 보상 (보상값 - 스케일 무관)
    private bool useWaypoints = false;
    private Vector3 finalGoalPosition;               // 최종 목적지 (웨이포인트 리스트의 마지막)

    private bool isCollided = false;

    private VesselManager vesselManager;
    private List<VesselAgent> cachedVessels;
    private Rigidbody rb;
    private Vector3 initialPosition;
    private Quaternion initialRotation;

    [Header("Radar Settings")]
    public VesselRadar radar;
    public float radarRange = 20f;    // 1/10 스케일 (원본 200m)
    public LayerMask radarDetectionLayers;

    [Header("Communication Settings")]
    public int maxCommunicationPartners = 4;

    [Header("Coordinate Settings")]
    public Vector3 originReference = Vector3.zero;

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

    [Header("COLREGs Caching")]
    private Dictionary<GameObject, COLREGsHandler.CollisionSituation> cachedSituations;
    private int cacheFrame = -1;

    // Per-frame 캐시: 가장 위험한 선박 계산 결과 (CollectObservations + CalculateReward 공유)
    private int dangerCacheFrame = -1;
    private VesselAgent cachedDangerousVessel;
    private float cachedDangerRisk;
    private COLREGsHandler.CollisionSituation cachedDangerSituation;

    [Header("Proximity Penalty (전방 장애물 회피)")]
    public float proximityThreshold = 5f;           // 전방 5m 이내 장애물에 반응 (1/10 스케일, 원본 50m)
    public float proximityPenaltyCoef = -0.15f;     // 최대 -0.15/step (선형)
    public int proximitySectorAngle = 45;            // 전방 ±45° 섹터
    private float previousFrontMinDist = float.MaxValue;

    [Header("Smoothness Reward (Phase 2 전용)")]
    public bool enableSmoothnessReward = false;     // Phase 2에서만 활성화 (통신 있을 때 부드러운 회피 유도)
    public float smoothnessCoef = -0.05f;           // 타각 변화량에 대한 패널티 계수
    private float previousRudderAngle = 0f;         // 이전 프레임의 타각

    public override void Initialize()
    {
        // Prefab Inspector 값 무시하고 GlobalScale로 강제 덮어쓰기
        maxEpisodeSteps = GlobalScale.MAX_EPISODE_STEPS;
        MaxStep = maxEpisodeSteps;

        radarRange = GlobalScale.RADAR_RANGE;
        maxMapDistance = GlobalScale.MAP_DISTANCE;
        goalReachedDistance = GlobalScale.GOAL_REACHED;
        waypointReachedDistance = GlobalScale.WAYPOINT_REACHED;
        proximityThreshold = GlobalScale.PROXIMITY_THRESHOLD;

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

        vesselManager = transform.root.GetComponentInChildren<VesselManager>();

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

        // detect layer는 전부 장애물임. collidor가 있는 경우에는 전부
        radar.detectionLayers = radarDetectionLayers;

        // Rule 17 추적을 위한 딕셔너리 초기화
        prevVesselStates = new Dictionary<GameObject, PrevVesselState>();
        lastTrackingTime = Time.time;

        // COLREGs 캐싱 초기화
        cachedSituations = new Dictionary<GameObject, COLREGsHandler.CollisionSituation>();
        dangerCacheFrame = -1;
        cachedDangerousVessel = null;
        cachedDangerRisk = 0f;
        cachedDangerSituation = COLREGsHandler.CollisionSituation.None;

        // Radar의 GetComponent 제거용 static dict 등록
        var myCollider = GetComponent<Collider>();
        if (myCollider != null) _colliderToAgent[myCollider] = this;
    }

    void OnDestroy()
    {
        var myCollider = GetComponent<Collider>();
        if (myCollider != null) _colliderToAgent.Remove(myCollider);
    }

    public override void OnEpisodeBegin()
    {
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

        // Proximity penalty 초기화
        previousFrontMinDist = float.MaxValue;

        // Smoothness reward 초기화
        previousRudderAngle = 0f;

        // danger 캐시 초기화 (이전 에피소드 데이터 무효화)
        dangerCacheFrame = -1;
        cachedDangerousVessel = null;
        cachedDangerRisk = 0f;
        cachedDangerSituation = COLREGsHandler.CollisionSituation.None;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float targetRudderAngle = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f) * vesselDynamics.maxTurnRate;
        // 네트워크 출력(-1~1)을 0~1로 변환: (-1+1)/2=0, (0+1)/2=0.5, (1+1)/2=1
        float targetThrust = (Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f) + 1f) / 2f * vesselDynamics.maxSpeed;

        vesselDynamics.SetRudderAngle(targetRudderAngle);
        vesselDynamics.SetTargetSpeed(targetThrust);
        vesselDynamics.SetBraking(false);

        CalculateReward();
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

        if (speedRatio < lowSpeedThreshold)
        {
            AddReward(lowSpeedPenalty);
        }

        // 0-2. Proximity penalty (전방 장애물 회피 유도)
        CalculateProximityReward();

        // 1. Navigation reward (목표 도달 + 진행 + 방향)
        if (CalculateNavigationReward(speedRatio)) return;  // 에피소드 종료됨

        // 2. COLREGs compliance (위험도 기반)
        CalculateColregsReward();

        // 3. Smoothness reward (Phase 2 전용)
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
        if (rudderRatio < 0.1f && cosAngle > 0.5f)
        {
            AddReward(forwardSpeedBonus);  // 직진 보너스
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

        if (mostDangerousVessel == null || maxRisk <= 0.3f) return;

        var situation = cachedDangerSituation;
        if (situation == COLREGsHandler.CollisionSituation.None) return;

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
        float riskWeight = 1.0f + maxRisk;  // 1.0 ~ 2.0
        AddReward(compliance * colregsRewardCoef * riskWeight);
    }

    /// <summary>
    /// Phase 2 전용: 타각 변화에 대한 부드러움 패널티
    /// </summary>
    private void CalculateSmoothnessReward()
    {
        if (enableSmoothnessReward)
        {
            float rudderChange = Mathf.Abs(vesselDynamics.RudderAngle - previousRudderAngle);
            float normalizedChange = rudderChange / vesselDynamics.maxTurnRate;  // 0~2 → 정규화
            AddReward(smoothnessCoef * normalizedChange);
        }
        previousRudderAngle = vesselDynamics.RudderAngle;
    }

    /// <summary>
    /// 전방 장애물 근접 패널티 (±45° 섹터 기반)
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

        previousFrontMinDist = frontMinDist;
    }

    void OnCollisionEnter(Collision collision)
    {
        HandleCollision(collision.gameObject);
    }

    void OnTriggerEnter(Collider other)
    {
        HandleCollision(other.gameObject);
    }

    private void HandleCollision(GameObject collidedObject)
    {
        if (isCollided) return;

        bool isVessel = collidedObject.GetComponent<VesselAgent>() != null;
        bool isObstacle = collidedObject.CompareTag("Obstacle");

        if (isVessel || isObstacle)
        {
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
            // 373D = radar(360) + self_state(6) + colregs(5) + position(2)
            for (int i = 0; i < radar.rayCount + 13; i++) sensor.AddObservation(0f);
            return;
        }

        // ========== 1. Radar (360D) ==========
        // IList<float> overload 1회 호출 (기존 360회 호출 → 1회, dispatch 오버헤드 제거)
        float[] rayDistances = radar.GetAllRayDistances();
        sensor.AddObservation(rayDistances);

        // ========== 2. Self State (6D) ==========
        Vector3 toGoal = goalPosition - transform.position;
        float goalDistance = toGoal.magnitude;
        float goalAngle = Vector3.SignedAngle(transform.forward, toGoal, Vector3.up);

        sensor.AddObservation(Mathf.Clamp(goalDistance / maxMapDistance, 0f, 1f));       // Goal distance (0~1 정규화)
        sensor.AddObservation(goalAngle / 180f);                                       // Goal angle
        sensor.AddObservation(vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed);  // Linear velocity
        sensor.AddObservation(vesselDynamics.YawRate / vesselDynamics.MaxYawRate);    // Angular velocity (실제 최대 yawRate로 정규화)

        // Heading을 -180~180 범위로 변환 후 정규화 (0°와 360°가 같은 값이 되도록)
        float heading = transform.eulerAngles.y;
        float headingNormalized = heading > 180f ? heading - 360f : heading;  // -180 ~ 180
        sensor.AddObservation(headingNormalized / 180f);                               // Heading (-1 ~ 1)
        sensor.AddObservation(vesselDynamics.RudderAngle / vesselDynamics.maxTurnRate); // Rudder angle

        // ========== 3. COLREGs Situation (5D) - One-hot encoding ==========
        // Per-frame 캐시 사용 (CalculateReward와 동일한 결과 공유)
        UpdateDangerCache();
        COLREGsHandler.CollisionSituation currentSituation = cachedDangerSituation;

        sensor.AddObservation(currentSituation == COLREGsHandler.CollisionSituation.None ? 1f : 0f);
        sensor.AddObservation(currentSituation == COLREGsHandler.CollisionSituation.HeadOn ? 1f : 0f);
        sensor.AddObservation(currentSituation == COLREGsHandler.CollisionSituation.CrossingStandOn ? 1f : 0f);
        sensor.AddObservation(currentSituation == COLREGsHandler.CollisionSituation.CrossingGiveWay ? 1f : 0f);
        sensor.AddObservation(currentSituation == COLREGsHandler.CollisionSituation.Overtaking ? 1f : 0f);

        // ========== 4. Position (2D) - 통신 범위 계산용, 학습에서 제외 ==========
        sensor.AddObservation(transform.position.x);
        sensor.AddObservation(transform.position.z);

        // 총 관측 차원: 360 (radar) + 6 (self state) + 5 (colregs) + 2 (position) = 373D
        // position은 Python에서 통신 파트너 계산용으로만 사용 (네트워크 입력 제외)
    }

    public Vector3 WorldToLocalPosition(Vector3 worldPos)
    {
        return worldPos - originReference;
    }

    public Vector3 LocalToWorldPosition(Vector3 localPos)
    {
        return localPos + originReference;
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
        finalGoalPosition = waypoints[waypoints.Count - 1];

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

            // proximity 리셋 (새 웨이포인트 향해 방향 전환)
            previousFrontMinDist = float.MaxValue;
        }
    }

    private void FixedUpdate()
    {
        vesselDynamics.UpdateDynamics(Time.fixedDeltaTime);
    }

    void OnDrawGizmos()
    {
        // Gizmos 비활성화 (Scene 뷰에서 Gizmos 토글로 제어)
    }

    public void SetRadarRange(float range)
    {
        radarRange = range;
        if (radar != null) radar.radarRange = range;
    }

    public void SetRadarDetectionLayers(LayerMask layers)
    {
        radarDetectionLayers = layers;
        if (radar != null) radar.detectionLayers = layers;
    }

    private COLREGsHandler.CollisionSituation GetCachedSituation(VesselAgent otherVessel)
    {
        // 새 프레임이면 캐시 초기화
        if (Time.frameCount != cacheFrame)
        {
            cachedSituations.Clear();
            cacheFrame = Time.frameCount;
        }

        GameObject otherVesselObj = otherVessel.gameObject;

        // 캐시에 없으면 계산 후 저장
        if (!cachedSituations.ContainsKey(otherVesselObj))
        {
            var situation = COLREGsHandler.AnalyzeSituation(
                transform.position, transform.forward, vesselDynamics.CurrentSpeed,
                otherVessel.transform.position, otherVessel.transform.forward,
                otherVessel.vesselDynamics.CurrentSpeed
            );
            cachedSituations[otherVesselObj] = situation;
        }

        return cachedSituations[otherVesselObj];
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

        if (cachedVessels == null) return;

        foreach (var otherVessel in cachedVessels)
        {
            if (otherVessel == this) continue;

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
