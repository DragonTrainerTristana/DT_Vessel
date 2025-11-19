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

    // ===== 수정된 보상 파라미터 (2025-11-13) =====
    public float arrivalReward = 50.0f;           // 15 → 50 (도착 보상 강화)
    public float goalDistanceCoef = 0.5f;         // 2.5 → 0.5 (progress 보상 약화)
    public float collisionPenalty = -50.0f;       // -15 → -50 (충돌 패널티 강화)
    public float colregsRewardCoef = 2.0f;        // 0.3 → 2.0 (COLREGs 준수 보상 강화)

    // Time penalty 추가 (빨리 끝내도록)
    public float timePenalty = -0.01f;            // 매 스텝마다 작은 패널티

    // Rotation penalty (조건부)
    public float rotationPenalty = -0.05f;        // -0.1 → -0.05 (약화)
    public float maxAngularVelocity = 20.0f;      // 10 → 20 (기준 완화)
    public bool enableRotationPenalty = false;    // 기본적으로 비활성화

    public float goalReachedDistance = 5.0f;

    public Vector3 goalPosition;
    public bool hasGoal = false;
    public string goalPointName;
    public int goalPointIndex;
    private float previousDistanceToGoal;

    private bool isCollided = false;
    private float collisionTimer = 0f;
    private float collisionCooldown = 1.0f;

    private VesselManager vesselManager;
    private List<VesselAgent> cachedVessels;
    private Rigidbody rb;
    private Vector3 initialPosition;
    private Quaternion initialRotation;

    [Header("Radar Settings")]
    public VesselRadar radar;
    public float radarRange = 100f;
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

    public override void Initialize()
    {
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

        vesselManager = FindFirstObjectByType<VesselManager>();
        if (vesselManager != null)
        {
            cachedVessels = vesselManager.GetAllVesselAgents();
        }

        float speedMultiplier = Random.Range(0.8f, 1.8f);
        vesselDynamics.maxSpeed *= speedMultiplier;

        radar = gameObject.GetComponent<VesselRadar>();
        if (radar == null) radar = gameObject.AddComponent<VesselRadar>();

        radar.radarRange = radarRange;
        radar.detectionLayers = radarDetectionLayers;

        AudioListener audioListener = GetComponent<AudioListener>();
        if (audioListener != null) Destroy(audioListener);

        // Rule 17 추적을 위한 딕셔너리 초기화
        previousVesselPositions = new Dictionary<GameObject, Vector3>();
        previousVesselForwards = new Dictionary<GameObject, Vector3>();
        previousVesselSpeeds = new Dictionary<GameObject, float>();
        lastTrackingTime = Time.time;

        // COLREGs 캐싱 초기화
        cachedSituations = new Dictionary<GameObject, COLREGsHandler.CollisionSituation>();
    }

    public override void OnEpisodeBegin()
    {
        vesselDynamics.ResetState();
        isCollided = false;
        collisionTimer = 0f;

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
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float targetRudderAngle = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f) * vesselDynamics.maxTurnRate;
        float targetThrust = Mathf.Clamp(actions.ContinuousActions[1], 0f, 1f) * vesselDynamics.maxSpeed;

        vesselDynamics.SetRudderAngle(targetRudderAngle);
        vesselDynamics.SetTargetSpeed(targetThrust);
        vesselDynamics.SetBraking(false);

        CalculateReward();
    }

    private void CalculateReward()
    {
        // 0. Time penalty (빨리 끝내도록 유도)
        AddReward(timePenalty);

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

            previousDistanceToGoal = currentDistanceToGoal;
        }

        // 2. Rotation penalty (조건부 - COLREGs 상황이 아닐 때만)
        if (enableRotationPenalty)
        {
            // COLREGs 상황이 있는지 체크
            bool hasCollisionRisk = false;
            if (cachedVessels != null)
            {
                foreach (var otherVessel in cachedVessels)
                {
                    if (otherVessel == this) continue;
                    var situation = GetCachedSituation(otherVessel);
                    if (situation != COLREGsHandler.CollisionSituation.None)
                    {
                        hasCollisionRisk = true;
                        break;
                    }
                }
            }

            // 충돌 위험이 없을 때만 과도한 회전 패널티
            if (!hasCollisionRisk)
            {
                float angularVelocity = Mathf.Abs(vesselDynamics.YawRate);
                if (angularVelocity > maxAngularVelocity)
                {
                    AddReward(rotationPenalty * angularVelocity);
                }
            }
        }

        // 3. COLREGs compliance (Rule 13-17 완전 구현)
        float totalCompliance = 0f;
        int complianceCount = 0;

        if (cachedVessels != null)
        {
            // 선박별 회피 행동 추적을 위한 딕셔너리
            Dictionary<GameObject, bool> vesselsTakingAction = new Dictionary<GameObject, bool>();

            foreach (var otherVessel in cachedVessels)
            {
                if (otherVessel == this) continue;

                // 캐싱된 상황 사용 또는 새로 계산
                var situation = GetCachedSituation(otherVessel);

                if (situation != COLREGsHandler.CollisionSituation.None)
                {
                    // TCPA/DCPA 계산
                    Vector3 myVelocity = transform.forward * vesselDynamics.CurrentSpeed;
                    Vector3 otherVelocity = otherVessel.transform.forward * otherVessel.vesselDynamics.CurrentSpeed;
                    float tcpa = COLREGsHandler.CalculateTCPA(
                        transform.position, myVelocity,
                        otherVessel.transform.position, otherVelocity
                    );
                    float dcpa = COLREGsHandler.CalculateDCPA(
                        transform.position, myVelocity,
                        otherVessel.transform.position, otherVelocity
                    );

                    // 상대 선박의 회피 행동 감지 (Rule 17을 위해)
                    bool otherVesselTakingAction = false;
                    GameObject otherVesselObj = otherVessel.gameObject;
                    float deltaTime = Time.time - lastTrackingTime;

                    if (deltaTime > 0.1f && previousVesselPositions.ContainsKey(otherVesselObj))
                    {
                        // IsVesselTakingAvoidanceAction 함수 사용
                        otherVesselTakingAction = COLREGsHandler.IsVesselTakingAvoidanceAction(
                            previousVesselPositions[otherVesselObj],
                            otherVessel.transform.position,
                            previousVesselForwards[otherVesselObj],
                            otherVessel.transform.forward,
                            previousVesselSpeeds[otherVesselObj],
                            otherVessel.vesselDynamics.CurrentSpeed,
                            deltaTime
                        );
                    }
                    else if (otherVessel.vesselDynamics != null)
                    {
                        // 초기 상태나 데이터 부족 시 간단한 감지
                        otherVesselTakingAction =
                            Mathf.Abs(otherVessel.vesselDynamics.RudderAngle) > 0.3f ||
                            otherVessel.vesselDynamics.CurrentSpeed < otherVessel.vesselDynamics.maxSpeed * 0.7f;
                    }

                    // 현재 상태 저장 (다음 프레임용)
                    previousVesselPositions[otherVesselObj] = otherVessel.transform.position;
                    previousVesselForwards[otherVesselObj] = otherVessel.transform.forward;
                    previousVesselSpeeds[otherVesselObj] = otherVessel.vesselDynamics.CurrentSpeed;

                    vesselsTakingAction[otherVesselObj] = otherVesselTakingAction;

                    // 권장 행동 계산 (TCPA/DCPA 포함)
                    var (recommendedRudder, recommendedSpeed) = COLREGsHandler.GetRecommendedAction(
                        situation,
                        vesselDynamics.CurrentSpeed,
                        otherVessel.transform.position - transform.position,
                        tcpa,
                        dcpa,
                        otherVesselTakingAction
                    );

                    // COLREGs 준수도 평가 (개선된 버전)
                    float compliance = COLREGsHandler.EvaluateCompliance(
                        situation,
                        vesselDynamics.RudderAngle,
                        recommendedRudder,
                        vesselDynamics.CurrentSpeed,
                        recommendedSpeed,
                        tcpa,
                        dcpa
                    );

                    totalCompliance += compliance;
                    complianceCount++;
                }
            }
        }

        if (complianceCount > 0)
        {
            float avgCompliance = totalCompliance / complianceCount;
            AddReward(avgCompliance * colregsRewardCoef);
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
            // Self: 360 radar + 6 self state = 366D
            // Neighbors: 4 × 371D = 1484D
            // COLREGs: 5D
            // Total: 366 + 1484 + 5 = 1855D
            for (int i = 0; i < 1855; i++) sensor.AddObservation(0f);
            return;
        }

        // ========== Radar Observation (360D): 360 rays × 1 param (거리만) ==========
        float[] rayDistances = radar.GetAllRayDistances();
        for (int i = 0; i < 360; i++)
        {
            sensor.AddObservation(rayDistances[i]);
            // ray 인덱스 = 각도 (0=0°, 90=90°, 180=180°, 270=270°)
        }

        // ========== Self State (6D) - GitHub 방식 ==========
        Vector3 toGoal = goalPosition - transform.position;
        float goalDistance = toGoal.magnitude;
        float goalAngle = Vector3.SignedAngle(transform.forward, toGoal, Vector3.up);

        sensor.AddObservation(goalDistance / radarRange);                              // Goal distance
        sensor.AddObservation(goalAngle / 180f);                                       // Goal angle
        sensor.AddObservation(vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed);  // Linear velocity
        sensor.AddObservation(vesselDynamics.YawRate / vesselDynamics.maxTurnRate);    // Angular velocity

        float heading = transform.eulerAngles.y;
        sensor.AddObservation(heading / 180f - 1f);                                    // Heading
        sensor.AddObservation(vesselDynamics.RudderAngle / vesselDynamics.maxTurnRate); // Rudder angle

        // ========== Neighbor Observations (GitHub 방식) ==========
        // 각 이웃의 완전한 observation 전달: radar(360) + goal(2) + speed(2) + colregs(5) + heading(1) + rudder(1) = 371D per neighbor
        // 최대 4 neighbors × 371D = 1484D
        List<VesselAgent> neighbors = new List<VesselAgent>();
        if (cachedVessels != null)
        {
            foreach (var otherVessel in cachedVessels)
            {
                if (otherVessel == this) continue;

                float distance = Vector3.Distance(transform.position, otherVessel.transform.position);
                if (distance <= radarRange && neighbors.Count < maxCommunicationPartners)
                {
                    neighbors.Add(otherVessel);
                }
            }
        }

        for (int i = 0; i < maxCommunicationPartners; i++)
        {
            if (i < neighbors.Count)
            {
                VesselAgent neighbor = neighbors[i];

                // Neighbor radar data (360D)
                float[] neighborRadar = neighbor.radar.GetAllRayDistances();
                for (int j = 0; j < 360; j++)
                {
                    sensor.AddObservation(neighborRadar[j]);
                }

                // Neighbor goal (2D)
                Vector3 neighborToGoal = neighbor.goalPosition - neighbor.transform.position;
                float neighborGoalDistance = neighborToGoal.magnitude;
                float neighborGoalAngle = Vector3.SignedAngle(neighbor.transform.forward, neighborToGoal, Vector3.up);
                sensor.AddObservation(neighborGoalDistance / radarRange);
                sensor.AddObservation(neighborGoalAngle / 180f);

                // Neighbor speed (2D)
                sensor.AddObservation(neighbor.vesselDynamics.CurrentSpeed / neighbor.vesselDynamics.maxSpeed);
                sensor.AddObservation(neighbor.vesselDynamics.YawRate / neighbor.vesselDynamics.maxTurnRate);

                // Neighbor COLREGs situation (4D) - from neighbor's perspective
                COLREGsHandler.CollisionSituation neighborSituation = COLREGsHandler.CollisionSituation.None;
                VesselAgent neighborMostDangerous = null;
                float neighborMinDist = float.MaxValue;

                foreach (var otherVessel in cachedVessels)
                {
                    if (otherVessel == neighbor) continue;
                    float dist = Vector3.Distance(neighbor.transform.position, otherVessel.transform.position);
                    if (dist < neighborMinDist)
                    {
                        neighborMinDist = dist;
                        neighborMostDangerous = otherVessel;
                    }
                }

                if (neighborMostDangerous != null)
                {
                    neighborSituation = COLREGsHandler.AnalyzeSituation(
                        neighbor.transform.position, neighbor.transform.forward, neighbor.vesselDynamics.CurrentSpeed,
                        neighborMostDangerous.transform.position, neighborMostDangerous.transform.forward,
                        neighborMostDangerous.vesselDynamics.CurrentSpeed
                    );
                }

                sensor.AddObservation(neighborSituation == COLREGsHandler.CollisionSituation.None ? 1f : 0f);
                sensor.AddObservation(neighborSituation == COLREGsHandler.CollisionSituation.HeadOn ? 1f : 0f);
                sensor.AddObservation(neighborSituation == COLREGsHandler.CollisionSituation.CrossingStandOn ? 1f : 0f);
                sensor.AddObservation(neighborSituation == COLREGsHandler.CollisionSituation.CrossingGiveWay ? 1f : 0f);
                sensor.AddObservation(neighborSituation == COLREGsHandler.CollisionSituation.Overtaking ? 1f : 0f);

                // Neighbor heading (1D)
                float neighborHeading = neighbor.transform.eulerAngles.y;
                sensor.AddObservation(neighborHeading / 180f - 1f);

                // Neighbor rudder (1D)
                sensor.AddObservation(neighbor.vesselDynamics.RudderAngle / neighbor.vesselDynamics.maxTurnRate);
            }
            else
            {
                // No neighbor: 371D 전부 0으로 채움 (radar 360 + goal 2 + speed 2 + colregs 5 + heading 1 + rudder 1)
                for (int j = 0; j < 371; j++)
                {
                    sensor.AddObservation(0f);
                }
            }
        }

        // ========== COLREGs Situation (4D) - One-hot encoding ==========
        COLREGsHandler.CollisionSituation currentSituation = COLREGsHandler.CollisionSituation.None;

        if (cachedVessels != null)
        {
            // 가장 가까운 선박 찾기
            VesselAgent mostDangerous = null;
            float minDistance = float.MaxValue;

            foreach (var otherVessel in cachedVessels)
            {
                if (otherVessel == this) continue;

                float distance = Vector3.Distance(transform.position, otherVessel.transform.position);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    mostDangerous = otherVessel;
                }
            }

            // 가장 위험한 선박에 대한 COLREGs 상황 분석
            if (mostDangerous != null)
            {
                currentSituation = COLREGsHandler.AnalyzeSituation(
                    transform.position, transform.forward, vesselDynamics.CurrentSpeed,
                    mostDangerous.transform.position, mostDangerous.transform.forward,
                    mostDangerous.vesselDynamics.CurrentSpeed
                );
            }
        }

        // One-hot encoding (5D)
        sensor.AddObservation(currentSituation == COLREGsHandler.CollisionSituation.None ? 1f : 0f);
        sensor.AddObservation(currentSituation == COLREGsHandler.CollisionSituation.HeadOn ? 1f : 0f);
        sensor.AddObservation(currentSituation == COLREGsHandler.CollisionSituation.CrossingStandOn ? 1f : 0f);
        sensor.AddObservation(currentSituation == COLREGsHandler.CollisionSituation.CrossingGiveWay ? 1f : 0f);
        sensor.AddObservation(currentSituation == COLREGsHandler.CollisionSituation.Overtaking ? 1f : 0f);

        // 총 관측 차원:
        // Self: 360D (radar) + 6D (self state) = 366D
        // Neighbors: 4 × [360D (radar) + 2D (goal) + 2D (speed) + 5D (colregs) + 1D (heading) + 1D (rudder)] = 4 × 371D = 1484D
        // COLREGs: 5D (self situation)
        // Total: 366 + 1484 + 5 = 1855D
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

        if (isCollided)
        {
            collisionTimer += Time.fixedDeltaTime;
            if (collisionTimer >= collisionCooldown) isCollided = false;
        }
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
}
