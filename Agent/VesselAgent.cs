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

    public float arrivalReward = 15.0f;
    public float goalDistanceCoef = 2.5f;
    public float collisionPenalty = -15.0f;
    public float rotationPenalty = -0.1f;
    public float maxAngularVelocity = 10.0f;
    public float goalReachedDistance = 5.0f;
    public float colregsRewardCoef = 0.3f;

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

        vesselManager = FindObjectOfType<VesselManager>();
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

        // 2. Rotation penalty (only if exceeds threshold)
        float angularVelocity = Mathf.Abs(vesselDynamics.YawRate);
        if (angularVelocity > maxAngularVelocity)
        {
            AddReward(rotationPenalty * angularVelocity);
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
            // 324D (180 + 4 + 140) 전부 0으로 채움
            for (int i = 0; i < 324; i++) sensor.AddObservation(0f);
            return;
        }

        // ========== Self Observation (180D): 30 regions × 6 params ==========
        for (int region = 0; region < 30; region++)
        {
            var regionData = radar.GetRegionData(region);
            sensor.AddObservation(regionData.closestDistance);
            sensor.AddObservation(regionData.relativeBearing);
            sensor.AddObservation(regionData.speedRatio);
            sensor.AddObservation(regionData.tcpa);
            sensor.AddObservation(regionData.dcpa);
            sensor.AddObservation(regionData.phase);
        }

        // ========== Message Passing Info (4D) ==========
        Vector3 directionToGoal = (goalPosition - transform.position).normalized;
        sensor.AddObservation(directionToGoal.x);
        sensor.AddObservation(directionToGoal.z);
        sensor.AddObservation(vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed);
        sensor.AddObservation(vesselDynamics.RudderAngle / vesselDynamics.maxTurnRate);

        // ========== Neighbor Observations (140D = 4 neighbors × 35D each) ==========
        var communication = GetComponent<VesselCommunication>();
        if (communication != null)
        {
            var commData = communication.GetCommunicationData();

            for (int i = 0; i < maxCommunicationPartners; i++)
            {
                if (i < commData.Count)
                {
                    var data = commData.Values.ElementAt(i);

                    // Compressed neighbor data (35D)
                    // 8 regions × 3 params (24D) + vessel (4D) + goal (3D) + fuzzy COLREGs (4D)
                    if (data.compressedRadarData != null && data.compressedRadarData.Length == 24)
                    {
                        for (int j = 0; j < 24; j++) sensor.AddObservation(data.compressedRadarData[j]);
                        for (int j = 0; j < 4; j++) sensor.AddObservation(data.vesselState[j]);
                        for (int j = 0; j < 3; j++) sensor.AddObservation(data.goalInfo[j]);
                        for (int j = 0; j < 4; j++) sensor.AddObservation(data.fuzzyCOLREGs[j]);
                    }
                    else
                    {
                        for (int j = 0; j < 35; j++) sensor.AddObservation(0f);
                    }
                }
                else
                {
                    for (int j = 0; j < 35; j++) sensor.AddObservation(0f);
                }
            }
        }
        else
        {
            for (int i = 0; i < maxCommunicationPartners * 35; i++) sensor.AddObservation(0f);
        }
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
