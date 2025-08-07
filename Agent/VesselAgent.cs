using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class VesselAgent : Agent
{
    // 동역학 참조
    [Header("동역학 모듈")]
    public VesselDynamics vesselDynamics;
    
    [Header("보상 파라미터")]
    public float arrivalReward = 15.0f;      // 목표 도달 보상
    public float goalDistanceCoef = 5.0f;     // 목표 거리 감소에 대한 보상 계수 (증가됨)
    public float collisionPenalty = -15.0f;   // 충돌 패널티
    public float angularVelocityPenalty = -0.05f; // 회전 속도 패널티 (완화됨)
    public float maxAngularVelocity = 0.7f;   // 패널티를 받기 시작하는 회전 속도 임계값
    public float goalReachedDistance = 5.0f;  // 목표 도달로 간주하는 거리
    public float collisionRadius = 2.0f;      // 충돌 감지 거리
    public float movementRewardCoef = 0.01f;  // 기본 이동 보상 계수 (새로 추가)
    public float stalematePenalty = -2.0f;    // 정지 상태 패널티 (강화됨)
    
    // 목표 위치 변수
    [Header("목표 위치 정보")]
    public Vector3 goalPosition;
    public bool hasGoal = false;
    public string goalPointName;  // 목표 지점 이름 추가
    public int goalPointIndex;    // 목표 지점 인덱스 추가
    private Vector3 previousPosition;
    private float previousDistanceToGoal;
    
    // 충돌 감지용 변수
    private bool isCollided = false;
    private float collisionTimer = 0f;
    private float collisionCooldown = 1.0f;  // 충돌 패널티 간격
    
    // 매니저 참조 (재스폰용)
    private VesselManager vesselManager;
    
    // 초기 위치 및 환경 설정
    private Rigidbody rb;
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    
    [Header("레이더 설정")]
    public VesselRadar radar;
    public float radarRange = 100f;           // 레이더 감지 거리 설정
    public LayerMask radarDetectionLayers;    // 레이더가 감지할 레이어 설정
    
    [Header("통신 설정")]
    public int maxCommunicationPartners = 4;  // 최대 통신 가능 선박 수
    
    [Header("좌표계 설정")]
    public Vector3 originReference = Vector3.zero;  // 원점 기준 좌표
    
    /// <summary>
    /// 에이전트 초기화 함수
    /// </summary>
    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
        }
        
        // 물리 설정
        rb.useGravity = false;
        rb.drag = 0.0f;
        rb.angularDrag = 0.0f;
        rb.constraints = RigidbodyConstraints.FreezePositionY | RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        
        // 초기 상태 저장
        initialPosition = transform.position;
        initialRotation = transform.rotation;
        
        // 선박 동역학 초기화
        if (vesselDynamics == null)
        {
            vesselDynamics = gameObject.GetComponent<VesselDynamics>();
            if (vesselDynamics == null)
            {
                vesselDynamics = gameObject.AddComponent<VesselDynamics>();
            }
        }
        
        vesselDynamics.Initialize(rb);
        
        // 매니저 찾기
        vesselManager = FindObjectOfType<VesselManager>();
        if (vesselManager == null)
        {
            Debug.LogWarning("VesselManager를 찾을 수 없습니다. 자동 재스폰이 작동하지 않을 수 있습니다.");
        }
        
        // 랜덤 속도 계수 설정 범위 확대
        float speedMultiplier = Random.Range(0.8f, 1.8f);
        vesselDynamics.maxSpeed *= speedMultiplier;
        
        // 레이더 컴포넌트 추가 및 설정
        radar = gameObject.GetComponent<VesselRadar>();
        if (radar == null)
        {
            radar = gameObject.AddComponent<VesselRadar>();
        }
        
        // 레이더 설정 적용
        radar.radarRange = radarRange;
        radar.detectionLayers = radarDetectionLayers;
    }
    
    /// <summary>
    /// 에피소드 시작 시 호출되는 함수
    /// </summary>
    public override void OnEpisodeBegin()
    {
        // 상태 재설정
        vesselDynamics.ResetState();
        isCollided = false;
        collisionTimer = 0f;
        
        // 초기 속도에 랜덤성 추가
        float initialSpeed = Random.Range(0.2f, 0.5f) * vesselDynamics.maxSpeed;
        vesselDynamics.SetTargetSpeed(initialSpeed);
        
        if (vesselManager != null)
        {
            vesselManager.RespawnVessel(gameObject);
        }
        else
        {
            // 매니저가 없는 경우 기본 리셋
            transform.position = initialPosition;
            transform.rotation = initialRotation;
        }
        
        previousPosition = transform.position;
        if (hasGoal)
        {
            previousDistanceToGoal = Vector3.Distance(transform.position, goalPosition);
        }
    }
    
    /// <summary>
    /// 에이전트 행동 수행 함수
    /// </summary>
    public override void OnActionReceived(ActionBuffers actions)
    {
        // 연속적인 행동 공간에서 타각과 추진력 받기
        float targetRudderAngle = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f) * vesselDynamics.maxTurnRate;
        float targetThrust = Mathf.Clamp(actions.ContinuousActions[1], 0f, 1f) * vesselDynamics.maxSpeed;
        
        // 동역학 모듈에 제어 명령 전달
        vesselDynamics.SetRudderAngle(targetRudderAngle);
        vesselDynamics.SetTargetSpeed(targetThrust);
        vesselDynamics.SetBraking(false);
        
        // 현재 프레임에서의 보상 계산
        CalculateReward();
    }

    /// <summary>
    /// 보상 계산 함수
    /// </summary>
    private void CalculateReward()
    {
        // 1. 기본 이동 보상 (정규화)
        float currentSpeed = vesselDynamics.CurrentSpeed;
        float movementReward = (currentSpeed / vesselDynamics.maxSpeed) * 0.1f;  // 스케일 조정
        AddReward(movementReward);
        
        // 2. 목표 도달 보상
        if (hasGoal)
        {
            float currentDistanceToGoal = Vector3.Distance(transform.position, goalPosition);
            
            if (currentDistanceToGoal < goalReachedDistance)
            {
                AddReward(10.0f);  // 목표 도달 보상 감소
                EndEpisode();
                return;
            }
            
            // 목표 방향으로의 진행 보상 계산 (정규화)
            float distanceChange = previousDistanceToGoal - currentDistanceToGoal;
            float goalProgressReward = Mathf.Clamp(distanceChange * 2.0f, -1.0f, 1.0f);  // 스케일 조정
            AddReward(goalProgressReward);
            
            // 목표를 향한 방향 정렬 보상 (정규화)
            Vector3 directionToGoal = (goalPosition - transform.position).normalized;
            float alignment = Vector3.Dot(transform.forward, directionToGoal);
            AddReward(alignment * 0.1f);  // 스케일 조정
            
            previousDistanceToGoal = currentDistanceToGoal;
        }
        
        // 3. 회전 패널티 (완화됨)
        float angularVelocity = Mathf.Abs(vesselDynamics.YawRate);
        if (angularVelocity > maxAngularVelocity)
        {
            float rotationPenalty = -0.1f * (angularVelocity / maxAngularVelocity);  // 정규화된 패널티
            AddReward(rotationPenalty);
        }
        
        // 4. COLREGs 규칙 준수 보상 (정규화)
        foreach (var otherVessel in FindObjectsOfType<VesselAgent>())
        {
            if (otherVessel == this) continue;

            var situation = COLREGsHandler.AnalyzeSituation(
                transform.position, transform.forward, vesselDynamics.CurrentSpeed,
                otherVessel.transform.position, otherVessel.transform.forward, otherVessel.vesselDynamics.CurrentSpeed
            );

            if (situation != COLREGsHandler.CollisionSituation.None)
            {
                var (recommendedRudder, recommendedSpeed) = COLREGsHandler.GetRecommendedAction(
                    situation,
                    vesselDynamics.CurrentSpeed,
                    otherVessel.transform.position - transform.position
                );

                float complianceReward = COLREGsHandler.EvaluateCompliance(
                    situation,
                    vesselDynamics.RudderAngle,
                    recommendedRudder
                ) * 0.5f;  // COLREGs 준수 보상 정규화

                AddReward(complianceReward);
            }
        }
    }
    
    /// <summary>
    /// 충돌 감지 함수
    /// </summary>
    void OnCollisionEnter(Collision collision)
    {
        HandleCollision(collision.gameObject);
    }

    /// <summary>
    /// 트리거 충돌 감지 함수
    /// </summary>
    void OnTriggerEnter(Collider other)
    {
        HandleCollision(other.gameObject);
    }
    
    /// <summary>
    /// 충돌 처리 함수
    /// </summary>
    private void HandleCollision(GameObject collidedObject)
    {
        // 충돌 쿨다운 시간 확인
        if (isCollided)
            return;
            
        // 장애물 또는 다른 선박과의 충돌 확인
        if (collidedObject.CompareTag("Obstacle") || collidedObject.GetComponent<VesselAgent>() != null)
        {
            // 충돌 패널티 적용
            AddReward(collisionPenalty);
            Debug.Log($"{gameObject.name}이(가) {collidedObject.name}와(과) 충돌! 패널티: {collisionPenalty}");
            
            // 에피소드 즉시 종료
            EndEpisode();
        }
    }
    
    /// <summary>
    /// 관측 데이터 수집 함수
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        if (!hasGoal)
        {
            Debug.LogWarning($"{gameObject.name}: 목표가 설정되지 않았습니다.");
            return;
        }

        // 레이더 스캔 실행
        radar.ScanRadar();

        // 1. 섹터별 레이더 관측값 (8섹터 × 3정보 = 24차원)
        for (int sector = 0; sector < 8; sector++) // 8개 섹터 (45도씩)
        {
            float minDist = float.MaxValue;
            float maxDist = 0f;
            float sumDist = 0f;
            
            // 각 섹터 내 45개 각도 분석
            for (int angle = 0; angle < 45; angle++)
            {
                float distance = radar.GetDistanceAtAngle(sector * 45 + angle);
                minDist = Mathf.Min(minDist, distance);
                maxDist = Mathf.Max(maxDist, distance);
                sumDist += distance;
            }
            
            sensor.AddObservation(minDist / radar.radarRange);        // 섹터 내 최근접 거리
            sensor.AddObservation((sumDist / 45) / radar.radarRange); // 섹터 내 평균 거리
            sensor.AddObservation(maxDist / radar.radarRange);        // 섹터 내 최원거리
        }

        // 2. 선박 상태
        sensor.AddObservation(vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed);  // 정규화된 속도
        sensor.AddObservation(transform.forward.x);  // 선수 방향 x
        sensor.AddObservation(transform.forward.z);  // 선수 방향 z
        sensor.AddObservation(vesselDynamics.YawRate / vesselDynamics.maxTurnRate);  // 정규화된 회전 속도

        // 3. 목표 관련
        Vector3 directionToGoal = (goalPosition - transform.position).normalized;
        sensor.AddObservation(directionToGoal.x);  // 목표 방향 x
        sensor.AddObservation(directionToGoal.z);  // 목표 방향 z
        sensor.AddObservation(Vector3.Distance(transform.position, goalPosition) / radarRange);  // 정규화된 목표 거리

        // COLREGs 상황 및 위험도 관측
        var (mostDangerousSituation, maxRisk, dangerousVessel) = 
            COLREGsHandler.AnalyzeMostDangerousVessel(this, radar.GetDetectedVessels());

        // COLREGs 상황을 강조하기 위해 3번 반복 (4차원 → 12차원)
        for (int repeat = 0; repeat < 3; repeat++)
        {
            sensor.AddObservation(mostDangerousSituation == COLREGsHandler.CollisionSituation.HeadOn ? 1.0f : 0.0f);
            sensor.AddObservation(mostDangerousSituation == COLREGsHandler.CollisionSituation.CrossingStandOn ? 1.0f : 0.0f);
            sensor.AddObservation(mostDangerousSituation == COLREGsHandler.CollisionSituation.CrossingGiveWay ? 1.0f : 0.0f);
            sensor.AddObservation(mostDangerousSituation == COLREGsHandler.CollisionSituation.Overtaking ? 1.0f : 0.0f);
        }
        
        // 위험도를 강조하기 위해 3번 반복 (1차원 → 3차원)
        for (int repeat = 0; repeat < 3; repeat++)
        {
            sensor.AddObservation(maxRisk);
        }
    }

    /// <summary>
    /// 목표 정보 수집 (Python에서 obs[1]로 접근)
    /// </summary>
    public void CollectGoalObservations(VectorSensor sensor)
    {
        if (hasGoal)
        {
            Vector3 directionToGoal = (goalPosition - transform.position).normalized;
            sensor.AddObservation(directionToGoal.x);  // 목표 방향 x
            sensor.AddObservation(directionToGoal.z);  // 목표 방향 z
        }
        else
        {
            sensor.AddObservation(0f);  // 목표 방향 x
            sensor.AddObservation(0f);  // 목표 방향 z
        }
    }

    /// <summary>
    /// 속도 정보 수집 (Python에서 obs[2]로 접근)
    /// </summary>
    public void CollectSpeedObservations(VectorSensor sensor)
    {
        sensor.AddObservation(vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed);  // 정규화된 속도
        sensor.AddObservation(vesselDynamics.RudderAngle / vesselDynamics.maxTurnRate);  // 정규화된 타각
    }

    /// <summary>
    /// 이웃 정보 수집 (Python에서 obs[3]로 접근)
    /// </summary>
    public void CollectNeighborObservations(VectorSensor sensor)
    {
        var communication = GetComponent<VesselCommunication>();
        if (communication != null)
        {
            var commData = communication.GetCommunicationData();
            
            // 이웃 관찰 배열 - 각 이웃의 46차원 상태
            for (int i = 0; i < maxCommunicationPartners; i++)
            {
                if (i < commData.Count)
                {
                    var data = commData.Values.ElementAt(i);
                    
                    // 이웃의 46차원 상태 전송
                    if (data.radarData != null)
                    {
                        // 섹터별 레이더 데이터 (24차원)
                        for (int j = 0; j < 24; j++)
                        {
                            sensor.AddObservation(data.radarData[j]);
                        }
                        
                        // 선박 상태 (4차원)
                        for (int j = 0; j < 4; j++)
                        {
                            sensor.AddObservation(data.vesselState[j]);
                        }
                        
                        // 목표 정보 (3차원)
                        for (int j = 0; j < 3; j++)
                        {
                            sensor.AddObservation(data.goalInfo[j]);
                        }
                        
                        // COLREGs 상황 (12차원)
                        for (int j = 0; j < 12; j++)
                        {
                            sensor.AddObservation(data.colregsSituation[j]);
                        }
                        
                        // 위험도 (3차원)
                        for (int j = 0; j < 3; j++)
                        {
                            sensor.AddObservation(data.dangerLevel[j]);
                        }
                    }
                    else
                    {
                        // 더미 데이터 (46차원)
                        for (int j = 0; j < 46; j++)
                        {
                            sensor.AddObservation(0f);
                        }
                    }
                }
                else
                {
                    // 더미 데이터 (46차원)
                    for (int j = 0; j < 46; j++)
                    {
                        sensor.AddObservation(0f);
                    }
                }
            }
        }
        else
        {
            // 통신 컴포넌트가 없는 경우 더미 데이터
            for (int i = 0; i < maxCommunicationPartners * 46; i++)
            {
                sensor.AddObservation(0f);  // 이웃 관찰
            }
        }
    }
    
    /// <summary>
    /// 월드 좌표를 로컬 좌표로 변환
    /// </summary>
    public Vector3 WorldToLocalPosition(Vector3 worldPos)
    {
        return worldPos - originReference;
    }

    /// <summary>
    /// 로컬 좌표를 월드 좌표로 변환
    /// </summary>
    public Vector3 LocalToWorldPosition(Vector3 localPos)
    {
        return localPos + originReference;
    }

    /// <summary>
    /// 목표 위치 설정 메서드
    /// </summary>
    public void SetGoal(Vector3 position)
    {
        goalPosition = position;
        hasGoal = true;
        previousDistanceToGoal = Vector3.Distance(transform.position, goalPosition);
        Debug.Log($"새로운 목표 설정 - 월드 좌표: {position}, 로컬 좌표: {WorldToLocalPosition(position)}");
    }
    
    /// <summary>
    /// FixedUpdate에서 동역학 업데이트 호출
    /// </summary>
    private void FixedUpdate()
    {
        vesselDynamics.UpdateDynamics(Time.fixedDeltaTime);
        
        // 충돌 쿨다운 업데이트
        if (isCollided)
        {
            collisionTimer += Time.fixedDeltaTime;
            if (collisionTimer >= collisionCooldown)
            {
                isCollided = false;
            }
        }
        
        // 정지된 상태 감지 (선택적)
        DetectStalemate();
    }
    
    /// <summary>
    /// 교착 상태 감지 (오랫동안 제자리에 있으면 패널티)
    /// </summary>
    private float stalemateTimer = 0f;
    private float stalemateThreshold = 20f; // 정지 상태 임계값 감소
    private Vector3 lastPosition;
    private float positionThreshold = 1.0f;
    private float lastStaleCheckTime = 0f;
    private float staleCheckInterval = 5f;  // 더 자주 체크
    
    private void DetectStalemate()
    {
        if (!hasGoal) return;
        
        if (Time.time - lastStaleCheckTime >= staleCheckInterval)
        {
            float movedDistance = Vector3.Distance(transform.position, lastPosition);
            
            if (movedDistance < positionThreshold)
            {
                AddReward(stalematePenalty);
                Debug.Log($"{gameObject.name}이(가) 정지 상태입니다. 패널티: {stalematePenalty}");
                
                if (Time.time - stalemateTimer > stalemateThreshold)
                {
                    Debug.Log($"{gameObject.name}이(가) 너무 오래 정지해 있습니다. 에피소드 종료");
                    EndEpisode();
                }
            }
            else
            {
                stalemateTimer = Time.time;
            }
            
            lastStaleCheckTime = Time.time;
            lastPosition = transform.position;
        }
    }
    
    /// <summary>
    /// 목표 위치를 시각적으로 표시
    /// </summary>
    void OnDrawGizmos()
    {
        if (hasGoal)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawLine(transform.position, goalPosition);
            Gizmos.DrawSphere(goalPosition, 1f);
        }
    }

    /// <summary>
    /// 레이더 범위 설정
    /// </summary>
    public void SetRadarRange(float range)
    {
        radarRange = range;
        if (radar != null)
        {
            radar.radarRange = range;
        }
    }

    /// <summary>
    /// 레이더 감지 레이어 설정
    /// </summary>
    public void SetRadarDetectionLayers(LayerMask layers)
    {
        radarDetectionLayers = layers;
        if (radar != null)
        {
            radar.detectionLayers = layers;
        }
    }

    void Start()
    {
        // 기존 코드...
        
        // Audio Listener 제거
        AudioListener audioListener = GetComponent<AudioListener>();
        if (audioListener != null)
        {
            Destroy(audioListener);
        }
        
        // 기존 코드...
    }
}
