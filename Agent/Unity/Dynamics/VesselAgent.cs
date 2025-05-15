using System.Collections;
using System.Collections.Generic;
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
    public float goalDistanceCoef = 2.5f;     // 목표 거리 감소에 대한 보상 계수
    public float collisionPenalty = -15.0f;   // 충돌 패널티
    public float angularVelocityPenalty = -0.1f; // 회전 속도 패널티
    public float maxAngularVelocity = 0.7f;   // 패널티를 받기 시작하는 회전 속도 임계값
    public float goalReachedDistance = 5.0f;  // 목표 도달로 간주하는 거리
    public float collisionRadius = 2.0f;      // 충돌 감지 거리
    
    // 목표 위치 변수
    private Vector3 goalPosition;
    private bool hasGoal = false;
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
        
        // 매니저가 있으면 새 목표 요청
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
        
        // 현재 위치 저장 (거리 변화 계산용)
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
        // 1. 목표 도달 보상 (g_r)
        if (hasGoal)
        {
            float currentDistanceToGoal = Vector3.Distance(transform.position, goalPosition);
            
            // 목표에 도달했는지 확인
            if (currentDistanceToGoal < goalReachedDistance)
            {
                // 목표 도달 보상
                AddReward(arrivalReward);
                Debug.Log($"{gameObject.name}이(가) 목표에 도달했습니다! 보상: {arrivalReward}");
                EndEpisode();
                return;
            }
            
            // 목표에 가까워지는 것에 대한 보상
            float distanceChange = previousDistanceToGoal - currentDistanceToGoal;
            float goalProgressReward = goalDistanceCoef * distanceChange;
            AddReward(goalProgressReward);
            
            // 이동 거리 저장
            previousDistanceToGoal = currentDistanceToGoal;
        }
        
        // 2. 회전 패널티 (w_r)
        float angularVelocity = Mathf.Abs(vesselDynamics.YawRate);
        if (angularVelocity > maxAngularVelocity)
        {
            float rotationPenalty = angularVelocityPenalty * angularVelocity;
            AddReward(rotationPenalty);
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
            
        // 장애물인지 확인
        if (collidedObject.CompareTag("Obstacle") || collidedObject.CompareTag("Vessel"))
        {
            // 충돌 패널티 적용
            AddReward(collisionPenalty);
            Debug.Log($"{gameObject.name}이(가) {collidedObject.name}와(과) 충돌! 패널티: {collisionPenalty}");
            
            // 충돌 상태 설정
            isCollided = true;
            collisionTimer = 0f;
            
            // 심각한 충돌이면 에피소드 종료 (선택 사항)
            if (Random.value < 0.3f) // 30% 확률로 에피소드 종료
            {
                EndEpisode();
            }
        }
    }
    
    /// <summary>
    /// 관측 데이터 수집 함수
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        // 속도 관련 정보
        sensor.AddObservation(vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed);
        
        // 방향 및 회전 관련 정보
        sensor.AddObservation(transform.forward.x);
        sensor.AddObservation(transform.forward.z);
        sensor.AddObservation(vesselDynamics.YawRate / vesselDynamics.maxTurnRate);
        
        // 제어 입력 상태
        sensor.AddObservation(vesselDynamics.RudderAngle / vesselDynamics.maxTurnRate);
        
        // 목표 관련 관측 추가 (목표가 있는 경우)
        if (hasGoal)
        {
            // 목표까지의 방향 (로컬 좌표계)
            Vector3 directionToGoal = transform.InverseTransformDirection(goalPosition - transform.position);
            sensor.AddObservation(directionToGoal.normalized.x);
            sensor.AddObservation(directionToGoal.normalized.z);
            
            // 목표까지의 거리 (정규화)
            float distanceToGoal = Vector3.Distance(transform.position, goalPosition);
            sensor.AddObservation(Mathf.Clamp01(distanceToGoal / 500f)); // 500m 기준으로 정규화
        }
        else
        {
            // 목표가 없는 경우 0으로 채움
            sensor.AddObservation(0);
            sensor.AddObservation(0);
            sensor.AddObservation(0);
        }
        
        // 레이캐스트로 장애물 감지 (선택적)
        AddRaycastObservations(sensor);
    }
    
    /// <summary>
    /// 레이캐스트 관측 추가 (장애물 감지용)
    /// </summary>
    private void AddRaycastObservations(VectorSensor sensor)
    {
        // 8방향 레이캐스트 (전방 및 45도 간격)
        float rayLength = 50f;
        int rayCount = 8;
        
        for (int i = 0; i < rayCount; i++)
        {
            float angle = i * 360f / rayCount;
            Vector3 direction = Quaternion.Euler(0, angle, 0) * transform.forward;
            
            Ray ray = new Ray(transform.position, direction);
            RaycastHit hit;
            
            if (Physics.Raycast(ray, out hit, rayLength))
            {
                // 장애물까지의 정규화된 거리
                float normalizedDistance = hit.distance / rayLength;
                sensor.AddObservation(normalizedDistance);
                
                // 디버깅용 레이
                Debug.DrawRay(ray.origin, ray.direction * hit.distance, Color.red);
            }
            else
            {
                // 장애물이 없으면 1.0 (최대 거리)
                sensor.AddObservation(1.0f);
                
                // 디버깅용 레이
                Debug.DrawRay(ray.origin, ray.direction * rayLength, Color.green);
            }
        }
    }
    
    /// <summary>
    /// 목표 위치 설정 메서드
    /// </summary>
    public void SetGoal(Vector3 position)
    {
        goalPosition = position;
        hasGoal = true;
        previousDistanceToGoal = Vector3.Distance(transform.position, goalPosition);
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
    private float stalemateThreshold = 30f; // 30초 정지 상태 임계값
    private Vector3 lastPosition;
    private float positionThreshold = 1.0f; // 1m 이상 이동해야 정지 상태가 아님
    
    private void DetectStalemate()
    {
        if (!hasGoal) return;
        
        // 10초마다 위치 확인
        if (Time.time - stalemateTimer > 10f)
        {
            float movedDistance = Vector3.Distance(transform.position, lastPosition);
            
            if (movedDistance < positionThreshold)
            {
                // 이동이 거의 없으면 패널티
                AddReward(-1.0f);
                Debug.Log($"{gameObject.name}이(가) 정지 상태입니다. 패널티 적용");
                
                // 너무 오래 정지해 있으면 에피소드 종료
                if (Time.time - stalemateTimer > stalemateThreshold)
                {
                    Debug.Log($"{gameObject.name}이(가) 너무 오래 정지해 있습니다. 에피소드 종료");
                    EndEpisode();
                }
            }
            else
            {
                // 충분히 이동했으면 타이머 리셋
                stalemateTimer = Time.time;
                lastPosition = transform.position;
            }
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
}
