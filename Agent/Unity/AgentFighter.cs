using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// ML-Agents를 이용한 선박 에이전트의 주요 기능을 담당하는 클래스입니다.
/// 강화학습 관련 핵심 로직과 다른 컴포넌트(COLREGs, 통신)의 결과를 통합 관리합니다.
/// </summary>
[RequireComponent(typeof(Rigidbody))]
[RequireComponent(typeof(COLREGsHandler))]
[RequireComponent(typeof(VesselCommunicationSystem))]
public class AgentFighter : Agent
{
    public float maxPower = 2f;
    public float maxSpeed = 3.0f;
    public float maxRudderSpeed = 0.9f;

    private Rigidbody rb;
    private float rbVelocity;

    public int stepCount;
    private int stepInterval;

    private float initialGoalDistance;
    public float goalDistance;
    private float preDistance;
    public float power;                // 현재 적용중인 추진력
    public float rudderSpeed;          // 현재 방향타 속도

    public Transform myInitialPos;     // 초기 위치
    public GameObject[] myGoals;       // 가능한 목표점 배열
    private GameObject myGoal;         // 현재 에피소드의 목표점
    public int randomIndex;            // 목표점 선택을 위한 랜덤 인덱스

    public int numberOfRays = 120;     // 레이캐스트 개수
    public float rayLength = 10;       // 레이캐스트 길이
    private float fieldOfView = 360f;  // 시야각
    private float[] currentRayHits;    // 현재 레이캐스트 결과
    private float[] previousRayHits;   // 이전 레이캐스트 결과

    // 추가된 컴포넌트 참조
    private COLREGsHandler colregsHandler;
    private VesselCommunicationSystem communicationSystem;

    /// <summary>
    /// 에이전트 초기화 함수입니다.
    /// </summary>
    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        colregsHandler = GetComponent<COLREGsHandler>();
        communicationSystem = GetComponent<VesselCommunicationSystem>();
        
        previousRayHits = new float[numberOfRays];
        currentRayHits = new float[numberOfRays];
        
        // 초기값 설정
        for (int i = 0; i < numberOfRays; i++)
        {
            previousRayHits[i] = rayLength;
            currentRayHits[i] = rayLength;
        }
    }

    /// <summary>
    /// 각 에피소드 시작 시 호출되는 함수입니다.
    /// 에이전트와 환경을 초기 상태로 리셋합니다.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        randomIndex = UnityEngine.Random.Range(0, myGoals.Length);
        myGoal = myGoals[randomIndex];

        this.gameObject.transform.position = myInitialPos.position;
        this.gameObject.transform.rotation = myInitialPos.rotation;
        
        initialGoalDistance = Vector3.Distance(myGoal.transform.position, this.gameObject.transform.position);
        goalDistance = UnityEngine.Vector3.Distance(myGoal.transform.position, this.gameObject.transform.position);
        preDistance = goalDistance;

        stepCount = 0;
        stepInterval = 0;

        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
    }

    /// <summary>
    /// 에이전트가 결정한 행동을 실행하고 보상을 계산하는 함수입니다.
    /// </summary>
    /// <param name="actions">에이전트가 선택한 행동 (추진력, 방향타)</param>
    public override void OnActionReceived(ActionBuffers actions)
    {
        stepCount++;
        stepInterval++;
        if (stepCount > 3000)
        {
            AddReward(-0.1f * goalDistance);
            EndEpisode();
        }
        
        goalDistance = UnityEngine.Vector3.Distance(myGoal.transform.position, this.gameObject.transform.position);
    
        float powerAction = actions.ContinuousActions[0];  // 추진력 액션 (-1 ~ 1)
        float rudderAction = actions.ContinuousActions[1]; // 방향타 액션 (-1 ~ 1)

        AgentMovement(powerAction, rudderAction);

        Vector3 targetDirection = (myGoal.transform.position - this.gameObject.transform.position).normalized;
        float angleDifference = Vector3.Angle(transform.forward, targetDirection);

        if (rb.velocity.magnitude == 0)
        {
            AddReward(-0.01f); 
        }

        if (goalDistance <= 1.0f)
        {
            AddReward(10.0f);
            EndEpisode();
        }
        else
        {
            if (stepInterval >= 10)
            {
                stepInterval = 0;

                if (angleDifference < 10f)
                {
                    AddReward(0.05f);
                }
                else if (angleDifference > 90f)
                {
                    AddReward(-0.05f);
                }

                if (goalDistance <= preDistance)
                {
                    AddReward(0.1f * (preDistance - goalDistance));
                    preDistance = goalDistance;
                }
                else
                {
                    AddReward(0.1f * (preDistance - goalDistance));
                }
            }
        }

        // COLREGs 상황에 따른 보상 적용 - COLREGsHandler 컴포넌트의 결과 활용
        if (colregsHandler.currentSituation != COLREGsHandler.COLREGsSituation.None)
        {
            float colregsReward = colregsHandler.CalculateCOLREGsReward(rudderSpeed);
            AddReward(colregsReward);
        }
    }
    
    /// <summary>
    /// 에이전트의 움직임을 처리하는 함수입니다.
    /// </summary>
    /// <param name="powerAction">추진력 액션 (-1 ~ 1)</param>
    /// <param name="rudderAction">방향타 액션 (-1 ~ 1)</param>
    /// <returns>없음</returns>
    public void AgentMovement(float powerAction, float rudderAction)
    {
        power = maxPower * ((powerAction + 1f) / 2f);

        float desiredRudderSpeed = maxRudderSpeed * rudderAction;
        rudderSpeed = Mathf.MoveTowards(rudderSpeed, desiredRudderSpeed, Time.deltaTime * 1.0f);

        if (rb.velocity.magnitude > maxSpeed)
        {
            rb.velocity *= 0.98f;
        }
            
        Quaternion turnRotation = Quaternion.Euler(0f, rudderSpeed, 0f);
        rb.MoveRotation(rb.rotation * turnRotation);

        Vector3 force = transform.forward * power;
        rbVelocity = force.z;
        rb.AddForce(force, ForceMode.VelocityChange);
    }
    
    /// <summary>
    /// 에이전트의 관측 정보를 수집하는 함수입니다.
    /// </summary>
    /// <param name="sensor">관측 정보를 저장할 센서 객체</param>
    public override void CollectObservations(VectorSensor sensor)
    {
        float angleStep = fieldOfView / numberOfRays;
        Vector3 forward = transform.forward; 
        Vector3 startDirection = Quaternion.Euler(0, -fieldOfView / 2, 0) * forward;
        
        bool vesselDetected = false;
        float closestVesselDistance = rayLength;
        Vector3 closestVesselDirection = Vector3.zero;
        GameObject closestVessel = null;

        // 레이캐스트를 통한 주변 감지
        for (int i = 0; i < numberOfRays; i++)
        {
            Vector3 rayDirection = Quaternion.Euler(0, angleStep * i, 0) * startDirection;
            RaycastHit hit;

            float distance = rayLength;
            if (Physics.Raycast(transform.position, rayDirection, out hit, rayLength))
            {
                distance = hit.distance;
                currentRayHits[i] = distance;
                
                // 다른 선박을 감지했을 때
                if (hit.collider.CompareTag("Vessel"))
                {
                    if (distance < closestVesselDistance)
                    {
                        vesselDetected = true;
                        closestVesselDistance = distance;
                        closestVesselDirection = rayDirection;
                        closestVessel = hit.collider.gameObject;
                    }
                }
            }
            else
            {
                currentRayHits[i] = rayLength;
            }

            sensor.AddObservation(distance / rayLength); 
        }
        
        // 가장 가까운 선박에 대한 COLREGs 상황 판단 - COLREGsHandler 컴포넌트에 위임
        if (vesselDetected && closestVessel != null)
        {
            colregsHandler.DetermineCOLREGsSituation(closestVesselDirection, closestVessel, closestVesselDistance);
            
            // 현재 COLREGs 상황을 관측에 추가 (원-핫 인코딩)
            sensor.AddOneHot((int)colregsHandler.currentSituation, 5); // None 포함 5가지 상태
        }
        else
        {
            colregsHandler.currentSituation = COLREGsHandler.COLREGsSituation.None;
            // 상황이 없을 때 원-핫 인코딩
            sensor.AddOneHot(0, 5);
        }

        sensor.AddObservation(transform.forward); 
        sensor.AddObservation(transform.rotation.y);
        UnityEngine.Vector3 relativePosition = myGoal.transform.position - this.gameObject.transform.position;
        sensor.AddObservation(relativePosition.magnitude);
        sensor.AddObservation(rb.velocity.magnitude);

        // 통신으로 받은 데이터를 관측에 추가 - VesselCommunicationSystem 컴포넌트의 결과 활용
        int communicationTargetCount = communicationSystem.GetCommunicationTargetCount();
        
        for (int i = 0; i < maxCommunicationTargets; i++)
        {
            if (i < communicationTargetCount)
            {
                VesselCommunicationSystem.CommunicationData data = communicationSystem.GetReceivedData(i);
                
                // 각 통신 대상에 대한 상대적 위치 (3)
                Vector3 relativePos = data.position - transform.position;
                sensor.AddObservation(relativePos.normalized);
                
                // 상대적 속도 (3)
                Vector3 relativeVel = data.velocity - rb.velocity;
                sensor.AddObservation(relativeVel.normalized);
                
                // 상대 선박의 방향 (3)
                sensor.AddObservation(data.forward);
                
                // 상대 선박의 제어 상태 (2)
                sensor.AddObservation(data.rudderSpeed);
                sensor.AddObservation(data.power);
                
                // 상대 선박의 COLREGs 상황 (원-핫 인코딩) (5)
                sensor.AddOneHot((int)data.situation, 5);
            }
            else
            {
                // 통신 대상이 없는 경우 0으로 채움
                sensor.AddObservation(Vector3.zero); // 상대적 위치 (3)
                sensor.AddObservation(Vector3.zero); // 상대적 속도 (3)
                sensor.AddObservation(Vector3.zero); // 방향 (3)
                sensor.AddObservation(0f); // 러더 속도 (1)
                sensor.AddObservation(0f); // 파워 (1)
                sensor.AddOneHot(0, 5); // COLREGs 상황 (5)
            }
        }
    }

    /// <summary>
    /// 레이캐스트 결과를 시각적으로 표현합니다.
    /// </summary>
    private void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;

        float angleStep = fieldOfView / numberOfRays;
        Vector3 forward = transform.forward;
        Vector3 startDirection = Quaternion.Euler(0, -fieldOfView / 2, 0) * forward;

        for (int i = 0; i < numberOfRays; i++)
        {
            Vector3 rayDirection = Quaternion.Euler(0, angleStep * i, 0) * startDirection;
            
            // 레이 색상 - 충돌 감지된 레이는 빨간색, 아니면 녹색
            if (currentRayHits != null && i < currentRayHits.Length && currentRayHits[i] < rayLength)
            {
                Gizmos.color = Color.red;
                Gizmos.DrawRay(transform.position, rayDirection * currentRayHits[i]);
                Gizmos.color = Color.yellow;
                Gizmos.DrawRay(transform.position + rayDirection * currentRayHits[i], 
                               rayDirection * (rayLength - currentRayHits[i]));
            }
            else
            {
                Gizmos.color = Color.green;
                Gizmos.DrawRay(transform.position, rayDirection * rayLength);
            }
        }
    }

    /// <summary>
    /// 충돌 처리 함수입니다.
    /// </summary>
    /// <param name="col">충돌 정보</param>
    public void OnCollisionEnter(Collision col)
    {
        if (col.collider.CompareTag("Obstacle"))
        {
            AddReward(-10.0f);
            EndEpisode();
        }
    }
}
