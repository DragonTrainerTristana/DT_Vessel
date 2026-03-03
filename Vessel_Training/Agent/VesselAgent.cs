using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class VesselAgent : Agent
{
    public VesselDynamics vesselDynamics;

    public float arrivalReward = 100.0f;          // 도착 보상 강화
    public float goalDistanceCoef = 1.0f;         // 0.5 → 1.0 (progress 보상 강화)
    public float collisionPenalty = -100.0f;       // 충돌 패널티 유지
    public float colregsRewardCoef = 0.45f;       // 0.3 → 0.45 (1.5배 강화, COMM_YES_COLREGS15 실험)
    public bool enableColregsReward = true;       // COLREGs 활성화 (우현 보상 제거됨, 좌현 패널티만)
    public float angleRewardCoef = 0.5f;          // 목적지 방향 보상 계수 (강화: 0.2 → 0.5)
    public float forwardSpeedBonus = 0.1f;        // 전진 보상 (강화: 0.05 → 0.1)

    public float timePenalty = -0.1f;             // 매 스텝마다 패널티 (강화: -0.03 → -0.1, 빙빙 방지)

    // Low speed penalty - 제자리 정지 방지
    public float lowSpeedThreshold = 0.2f;        // 20% maxSpeed 미만이면 패널티 (COLREGs 감속과 호환)
    public float lowSpeedPenalty = -0.15f;        // 저속 패널티 (강화: -0.1 → -0.15)

    public float goalReachedDistance = 15.0f;    // 10 → 15 (2025-01-05)
    public float maxMapDistance = 200f;           // 맵 최대 거리 (goal distance 정규화용)

    public Vector3 goalPosition;
    public bool hasGoal = false;
    public string goalPointName;
    public int goalPointIndex;
    private float previousDistanceToGoal;

    private bool isCollided = false;

    private VesselManager vesselManager;
    private List<VesselAgent> cachedVessels;
    private Rigidbody rb;
    private Vector3 initialPosition;
    private Quaternion initialRotation;

    [Header("Radar Settings")]
    public VesselRadar radar;
    public float radarRange = 60f;
    public LayerMask radarDetectionLayers;

    [Header("Communication Settings")]
    public int maxCommunicationPartners = 4;

    [Header("Coordinate Settings")]
    public Vector3 originReference = Vector3.zero;

    [Header("COLREGs Tracking (Rule 17)")]
    private Dictionary<GameObject, Vector3> previousVesselPositions;
    private Dictionary<GameObject, Vector3> previousVesselForwards;
    private Dictionary<GameObject, float> previousVesselSpeeds;
    private float lastTrackingTime;

    [Header("COLREGs Caching")]
    private Dictionary<GameObject, COLREGsHandler.CollisionSituation> cachedSituations;
    private int cacheFrame = -1;

    // Per-frame 캐시: 가장 위험한 선박 계산 결과 (CollectObservations + CalculateReward 공유)
    private int dangerCacheFrame = -1;
    private VesselAgent cachedDangerousVessel;
    private float cachedDangerRisk;
    private COLREGsHandler.CollisionSituation cachedDangerSituation;

    [Header("Spinning Detection")]
    private float netRotation = 0f;                 // 순회전량 (방향 포함, 우회전+/좌회전-)
    private float previousHeading = 0f;             // 이전 heading
    public float spinningThreshold = 180f;          // 한 방향으로 180도 이상 회전하면 패널티 (반바퀴)
    public float spinningPenalty = -80f;             // 빙글빙글 패널티 (충돌과 구분: -80)

    public override void Initialize()
    {
        MaxStep = 10000;  // 에피소드당 최대 스텝 (10000 스텝 후 자동 종료)

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
        previousVesselPositions = new Dictionary<GameObject, Vector3>();
        previousVesselForwards = new Dictionary<GameObject, Vector3>();
        previousVesselSpeeds = new Dictionary<GameObject, float>();
        lastTrackingTime = Time.time;

        // COLREGs 캐싱 초기화
        cachedSituations = new Dictionary<GameObject, COLREGsHandler.CollisionSituation>();
        dangerCacheFrame = -1;
        cachedDangerousVessel = null;
        cachedDangerRisk = 0f;
        cachedDangerSituation = COLREGsHandler.CollisionSituation.None;
    }

    public override void OnEpisodeBegin()
    {
        vesselDynamics.ResetState();
        isCollided = false;

        float initialSpeed = Random.Range(0.2f, 0.5f) * vesselDynamics.maxSpeed;
        vesselDynamics.SetTargetSpeed(initialSpeed);

        if (vesselManager != null)
        {
            vesselManager.RespawnVessel(gameObject);
        }
        else
        {
            transform.position = initialPosition;
            transform.rotation = initialRotation;
        }

        if (hasGoal) previousDistanceToGoal = Vector3.Distance(transform.position, goalPosition);

        // Rule 17 추적 초기화
        previousVesselPositions.Clear();
        previousVesselForwards.Clear();
        previousVesselSpeeds.Clear();
        lastTrackingTime = Time.time;

        // Spinning detection 초기화
        netRotation = 0f;
        previousHeading = transform.eulerAngles.y;
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
        // 0. Spinning detection (빙글빙글 도는 것 감지 및 패널티)
        // 순회전량 추적: 우회전 +, 좌회전 -
        // 정상 회피: 우회전 후 좌회전 → 상쇄됨
        // 빙글빙글: 계속 한 방향 → 누적됨
        float currentHeading = transform.eulerAngles.y;
        float headingDelta = Mathf.DeltaAngle(previousHeading, currentHeading);  // 부호 있음
        netRotation += headingDelta;
        previousHeading = currentHeading;

        // 한 방향으로 spinningThreshold(180)도 이상 회전하면 패널티
        if (Mathf.Abs(netRotation) >= spinningThreshold)
        {
            AddReward(spinningPenalty);
            EndEpisode();
            return;
        }

        // 0-1. Time penalty (빨리 끝내도록 유도)
        AddReward(timePenalty);

        // 0-2. Forward speed bonus (전진 장려)
        float speedRatio = vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed;
        AddReward(forwardSpeedBonus * speedRatio);

        // 0-2. Low speed penalty (저속 패널티 - 제자리 정지 방지)
        if (speedRatio < lowSpeedThreshold)
        {
            AddReward(lowSpeedPenalty);
        }

        // 1. Goal reward (arrival + progress)
        if (hasGoal)
        {
            float currentDistanceToGoal = Vector3.Distance(transform.position, goalPosition);

            if (currentDistanceToGoal < goalReachedDistance)
            {
                AddReward(arrivalReward);
                EndEpisode();
                return;
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
                AddReward(0.1f);  // 직진 보너스
            }

            previousDistanceToGoal = currentDistanceToGoal;
        }

        // 2. COLREGs compliance (위험도 기반 - 가장 위험한 선박 하나만 처리)
        // Phase 1에서는 비활성화하여 기본 네비게이션 먼저 학습
        if (enableColregsReward && cachedVessels != null)
        {
            // Per-frame 캐시 사용 (CollectObservations와 동일한 결과 공유)
            UpdateDangerCache();
            VesselAgent mostDangerousVessel = cachedDangerousVessel;
            float maxRisk = cachedDangerRisk;

            if (mostDangerousVessel != null && maxRisk > 0.3f)
            {
                var situation = cachedDangerSituation;

                if (situation != COLREGsHandler.CollisionSituation.None)
                {
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

                    if (deltaTime > 0.1f && previousVesselPositions.ContainsKey(otherVesselObj))
                    {
                        otherVesselTakingAction = COLREGsHandler.IsVesselTakingAvoidanceAction(
                            previousVesselPositions[otherVesselObj],
                            mostDangerousVessel.transform.position,
                            previousVesselForwards[otherVesselObj],
                            mostDangerousVessel.transform.forward,
                            previousVesselSpeeds[otherVesselObj],
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

                    // 현재 상태 저장 (다음 프레임용)
                    previousVesselPositions[otherVesselObj] = mostDangerousVessel.transform.position;
                    previousVesselForwards[otherVesselObj] = mostDangerousVessel.transform.forward;
                    previousVesselSpeeds[otherVesselObj] = mostDangerousVessel.vesselDynamics.CurrentSpeed;

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
                        vesselDynamics.CurrentSpeed,
                        recommendedSpeed,
                        tcpa,
                        dcpa
                    );

                    // 위험도에 비례한 보상 (위험할수록 COLREGs 준수가 더 중요)
                    float riskWeight = 1.0f + maxRisk;  // 1.0 ~ 2.0
                    AddReward(compliance * colregsRewardCoef * riskWeight);
                }
            }
        }

        // 추적 시간 업데이트
        lastTrackingTime = Time.time;
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

        if (collidedObject.CompareTag("Obstacle") || collidedObject.GetComponent<VesselAgent>() != null)
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
        float[] rayDistances = radar.GetAllRayDistances();
        for (int i = 0; i < radar.rayCount; i++)
        {
            sensor.AddObservation(rayDistances[i]);
        }

        // ========== 2. Self State (6D) ==========
        Vector3 toGoal = goalPosition - transform.position;
        float goalDistance = toGoal.magnitude;
        float goalAngle = Vector3.SignedAngle(transform.forward, toGoal, Vector3.up);

        sensor.AddObservation(Mathf.Clamp(goalDistance / maxMapDistance, 0f, 1f));       // Goal distance (0~1 정규화)
        sensor.AddObservation(goalAngle / 180f);                                       // Goal angle
        sensor.AddObservation(vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed);  // Linear velocity
        sensor.AddObservation(vesselDynamics.YawRate / vesselDynamics.maxTurnRate);    // Angular velocity

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

    private void FixedUpdate()
    {
        vesselDynamics.UpdateDynamics(Time.fixedDeltaTime);
    }

    void OnDrawGizmos()
    {
        if (hasGoal)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawLine(transform.position, goalPosition);
            Gizmos.DrawSphere(goalPosition, 1f);
        }
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

            float risk = COLREGsHandler.CalculateRisk(
                transform.position, transform.forward, vesselDynamics.CurrentSpeed,
                otherVessel.transform.position, otherVessel.transform.forward,
                otherVessel.vesselDynamics.CurrentSpeed
            );

            if (risk > cachedDangerRisk)
            {
                cachedDangerRisk = risk;
                cachedDangerousVessel = otherVessel;
            }
        }

        if (cachedDangerousVessel != null && cachedDangerRisk > 0.3f)
        {
            cachedDangerSituation = COLREGsHandler.AnalyzeSituation(
                transform.position, transform.forward, vesselDynamics.CurrentSpeed,
                cachedDangerousVessel.transform.position, cachedDangerousVessel.transform.forward,
                cachedDangerousVessel.vesselDynamics.CurrentSpeed
            );
        }
    }
}
