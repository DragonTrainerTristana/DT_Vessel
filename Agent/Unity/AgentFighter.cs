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

    // 통신을 위한 원본 액션값 저장
    public float lastRudderAction;     // 마지막 방향타 액션 (-1~1)
    public float lastPowerAction;      // 마지막 추진력 액션 (-1~1)

    public Transform myInitialPos;     // 초기 위치
    public GameObject[] myGoals;       // 가능한 목표점 배열
    private GameObject myGoal;         // 현재 에피소드의 목표점
    public int randomIndex;            // 목표점 선택을 위한 랜덤 인덱스

    public int numberOfRays = 120;     // 레이캐스트 개수
    public float rayLength = 10;       // 레이캐스트 길이
    private float fieldOfView = 360f;  // 시야각
    private float[] currentRayHits;    // 현재 레이캐스트 결과
    private float[] previousRayHits;   // 이전 레이캐스트 결과

    // 통신 관련 변수 (거리 가장 가까운 애들 순으로 해야함) 
    public int maxCommunicationTargets = 4; // 최대 통신 대상 수

    // 추가된 컴포넌트 참조
    private COLREGsHandler colregsHandler;
    private VesselCommunicationSystem communicationSystem;

    // 추가된 필드: 움직임 관련 메트릭
    private int stationaryFrames = 0;  // 움직이지 않은 프레임 수
    private float minMovementThreshold = 0.1f;  // 최소 움직임 임계값
    private Vector3 lastPosition;  // 마지막 위치
    private float stationaryPenalty = -0.005f;  // 정지 패널티
    private float progressRewardMultiplier = 0.3f;  // 진행 보상 증가

    [Header("Trajectory Settings")]
    public bool showTrajectory = true;
    public Color trajectoryColor = Color.cyan;
    public float trajectoryWidth = 0.1f;
    public int maxTrajectoryPoints = 100;
    private List<Vector3> trajectoryPoints = new List<Vector3>();
    private LineRenderer trajectoryLine;

    [Header("COLREGs Settings")]
    public float colregsRewardMultiplier = 1.0f;
    public float collisionPenalty = -10.0f;
    public float rightTurnReward = 0.5f;
    public float wrongTurnPenalty = -0.3f;
    public float speedReductionReward = 0.3f;
    public float overtakingReward = 0.4f;

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

        // 충돌 설정 디버깅
        Debug.Log("Agent Collider is " + (GetComponent<Collider>() ? "present" : "missing"));
        Debug.Log("Agent Rigidbody is " + (GetComponent<Rigidbody>() ? "present" : "missing"));
        if (GetComponent<Rigidbody>())
            Debug.Log("Agent Rigidbody is Kinematic: " + GetComponent<Rigidbody>().isKinematic);

        // 궤적 시각화를 위한 LineRenderer 추가
        trajectoryLine = gameObject.AddComponent<LineRenderer>();
        trajectoryLine.material = new Material(Shader.Find("Sprites/Default"));
        trajectoryLine.startColor = trajectoryColor;
        trajectoryLine.endColor = trajectoryColor;
        trajectoryLine.startWidth = trajectoryWidth;
        trajectoryLine.endWidth = trajectoryWidth;
        trajectoryLine.positionCount = 0;
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
        stationaryFrames = 0;

        lastPosition = transform.position;

        lastRudderAction = 0f;
        lastPowerAction = 0f;

        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // 시작 시 작은 힘을 줘서 움직이도록 유도
        rb.AddForce(transform.forward * 0.5f, ForceMode.VelocityChange);

        // 궤적 초기화
        trajectoryPoints.Clear();
        trajectoryPoints.Add(transform.position);
        UpdateTrajectoryLine();
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
            Debug.Log("Episode ended due to timeout");
            AddReward(-0.1f * goalDistance);
            EndEpisode();
        }

        goalDistance = UnityEngine.Vector3.Distance(myGoal.transform.position, this.gameObject.transform.position);

        float powerAction = actions.ContinuousActions[0];  // 추진력 액션 (-1 ~ 1)
        float rudderAction = actions.ContinuousActions[1]; // 방향타 액션 (-1 ~ 1)

        // 움직임을 강제하는 로직 - 너무 작은 액션일 경우 최소값 적용
        if (Mathf.Abs(powerAction) < 0.2f)
        {
            powerAction = powerAction < 0 ? -0.2f : 0.2f;
        }

        // 원본 액션 값 저장 (통신 시스템에서 사용)
        lastPowerAction = powerAction;
        lastRudderAction = rudderAction;

        AgentMovement(powerAction, rudderAction);

        Vector3 targetDirection = (myGoal.transform.position - this.gameObject.transform.position).normalized;
        float angleDifference = Vector3.Angle(transform.forward, targetDirection);

        // 정지 상태 검사 및 페널티 적용
        float moveDistance = Vector3.Distance(transform.position, lastPosition);
        if (moveDistance < minMovementThreshold)
        {
            stationaryFrames++;
            if (stationaryFrames > 10)  // 10프레임 이상 멈춰있으면
            {
                AddReward(stationaryPenalty * stationaryFrames);  // 누적 페널티
                // 디버그 로그 추가
                if (stationaryFrames % 50 == 0)
                {
                    Debug.Log($"Agent stationary for {stationaryFrames} frames, applying penalty");
                }
            }
        }
        else
        {
            stationaryFrames = 0;
            // 목표를 향해 움직일 때 작은 보상
            if (goalDistance < preDistance)
            {
                float progressReward = progressRewardMultiplier * (preDistance - goalDistance);
                AddReward(progressReward);
            }
        }
        lastPosition = transform.position;

        if (rb.velocity.magnitude == 0)
        {
            AddReward(-0.03f);  // 움직이지 않는 페널티 증가
        }

        if (goalDistance <= 1.0f)
        {
            Debug.Log("Goal reached! Adding reward and ending episode");
            AddReward(20.0f);  // 목표 도달 보상 증가
            EndEpisode();
        }
        else
        {
            if (stepInterval >= 10)
            {
                stepInterval = 0;

                if (angleDifference < 10f)
                {
                    AddReward(0.1f);  // 목표 방향을 향한 보상 증가
                }
                else if (angleDifference > 90f)
                {
                    AddReward(-0.05f);
                }

                if (goalDistance <= preDistance)
                {
                    AddReward(0.2f * (preDistance - goalDistance));  // 진행 보상 증가
                    preDistance = goalDistance;
                }
                else
                {
                    AddReward(0.1f * (preDistance - goalDistance));
                }
            }
        }

        // COLREGs 규칙 준수 보상 강화
        float colregsReward = colregsHandler.CalculateCOLREGsReward(rudderSpeed);
        AddReward(colregsReward * colregsRewardMultiplier);

        // 우현 회피 보상/페널티
        if (colregsHandler.currentSituation == COLREGsHandler.COLREGsSituation.HeadOn ||
            colregsHandler.currentSituation == COLREGsHandler.COLREGsSituation.CrossingGiveWay)
        {
            if (rudderSpeed < 0) // 우현으로 회피
            {
                AddReward(rightTurnReward);
            }
            else if (rudderSpeed > 0) // 좌현으로 회피 (잘못된 방향)
            {
                AddReward(wrongTurnPenalty);
            }
        }

        // 속도 감소 보상 (Crossing 상황에서)
        if (colregsHandler.currentSituation == COLREGsHandler.COLREGsSituation.CrossingGiveWay)
        {
            if (power < maxPower * 0.5f) // 속도 감소
            {
                AddReward(speedReductionReward);
            }
        }

        // 추월 상황 보상
        if (colregsHandler.currentSituation == COLREGsHandler.COLREGsSituation.Overtaking)
        {
            if (rudderSpeed < 0 && power > maxPower * 0.7f) // 우현으로 추월
            {
                AddReward(overtakingReward);
            }
        }

        // 충돌 페널티 강화
        if (Vector3.Distance(transform.position, lastPosition) < 0.1f)
        {
            AddReward(collisionPenalty);
        }

        // 궤적 업데이트
        UpdateTrajectory();
    }

    /// <summary>
    /// 에이전트의 움직임을 처리하는 함수입니다.
    /// 선박의 동역학을 고려하여 추진력과 방향타의 상호작용을 구현합니다.
    /// </summary>
    /// <param name="powerAction">추진력 액션 (-1 ~ 1)</param>
    /// <param name="rudderAction">방향타 액션 (-1 ~ 1)</param>
    public void AgentMovement(float powerAction, float rudderAction)
    {
        // COLREGs 상황에 따른 강제 회피
        if (colregsHandler.currentSituation == COLREGsHandler.COLREGsSituation.HeadOn ||
            colregsHandler.currentSituation == COLREGsHandler.COLREGsSituation.CrossingGiveWay)
        {
            // 우현 회피 강제
            rudderAction = Mathf.Min(rudderAction, -0.2f);
        }
        else if (colregsHandler.currentSituation == COLREGsHandler.COLREGsSituation.Overtaking)
        {
            // 추월 시 우현 회피 강제
            rudderAction = Mathf.Min(rudderAction, -0.1f);
        }

        // 추진력 계산 (0 ~ maxPower)
        power = maxPower * ((powerAction + 1f) / 2f);

        // 방향타 각도 계산 (라디안)
        float rudderAngle = Mathf.Clamp(rudderAction * Mathf.PI / 4, -Mathf.PI / 4, Mathf.PI / 4);

        // 선박의 길이 (실제 선박의 특성에 맞게 조정 필요)
        float vesselLength = 10f;

        // 관성 모멘트 (실제 선박의 특성에 맞게 조정 필요)
        float momentOfInertia = 1000f;

        // 회전 모멘트 계산 (추진력에 비례)
        float rotationMoment = power * vesselLength * Mathf.Sin(rudderAngle);

        // 회전 각가속도 계산
        float angularAcceleration = rotationMoment / momentOfInertia;

        // 실제 회전 속도 계산 (추진력에 비례)
        float effectiveRudderSpeed = maxRudderSpeed * (power / maxPower) * rudderAction;

        // 부드러운 회전을 위한 보간
        rudderSpeed = Mathf.MoveTowards(rudderSpeed, effectiveRudderSpeed, Time.deltaTime * 1.0f);

        // 최대 속도 제한
        if (rb.velocity.magnitude > maxSpeed)
        {
            rb.velocity *= 0.98f;
        }

        // 회전 적용 (추진력이 있을 때만 회전)
        if (power > 0.1f)
        {
            Quaternion turnRotation = Quaternion.Euler(0f, rudderSpeed, 0f);
            rb.MoveRotation(rb.rotation * turnRotation);
        }

        // 추진력 적용
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

                // 다른 선박을 감지했을 때 - 태그 대신 컴포넌트로 확인
                if (hit.collider.GetComponent<AgentFighter>() != null && hit.collider.gameObject != this.gameObject)
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

        // COLREGs 상황 판단 - 가장 가까운 선박 기준
        if (vesselDetected && closestVessel != null)
        {
            colregsHandler.DetermineCOLREGsSituation(closestVesselDirection, closestVessel, closestVesselDistance);
        }

        // 나의 현재 COLREGs 상황 정보 (원-핫 인코딩)
        int situationIndex = (int)colregsHandler.currentSituation;
        for (int i = 0; i < 5; i++) // None 포함 5가지 상태
        {
            sensor.AddObservation(i == situationIndex ? 1.0f : 0.0f);
        }

        // 내 위험도 (가장 위험한 상황 기준)
        float myRisk = colregsHandler.trackedVessels.Count > 0 ? colregsHandler.trackedVessels[0].risk : 0f;
        sensor.AddObservation(myRisk);

        // 내 기본 정보
        sensor.AddObservation(transform.forward);    // 방향 (3)
        sensor.AddObservation(transform.rotation.y); // 회전 (1)
        sensor.AddObservation(rb.velocity.magnitude); // 속도 크기 (1)
        sensor.AddObservation(power);                // 현재 추진력 (1)
        sensor.AddObservation(rudderSpeed);          // 현재 방향타 속도 (1)

        // 목표 관련 정보
        Vector3 relativePosition = myGoal.transform.position - transform.position;
        sensor.AddObservation(relativePosition.normalized); // 목표 방향 (3)
        sensor.AddObservation(relativePosition.magnitude);  // 목표 거리 (1)

        // 통신으로 받은 데이터를 관측에 추가
        int communicationTargetCount = communicationSystem.GetCommunicationTargetCount();

        for (int i = 0; i < maxCommunicationTargets; i++)
        {
            if (i < communicationTargetCount)
            {
                VesselCommunicationSystem.CommunicationData data = communicationSystem.GetReceivedData(i);

                // A. 다른 선박의 Observation
                // 각 통신 대상에 대한 상대적 위치 (3)
                Vector3 relativePos = data.position - transform.position;
                sensor.AddObservation(relativePos.normalized);

                // 상대적 속도 (3)
                Vector3 relativeVel = data.velocity - rb.velocity;
                sensor.AddObservation(relativeVel.normalized);
                sensor.AddObservation(relativeVel.magnitude); // 상대속도 크기 (1)

                // 상대 선박의 방향 (3)
                sensor.AddObservation(data.forward);

                // B. 다른 선박의 Action
                // 상대 선박의 제어 상태 (4)
                sensor.AddObservation(data.rudderSpeed);  // 현재 적용 중인 방향타 속도
                sensor.AddObservation(data.power);        // 현재 적용 중인 추진력
                sensor.AddObservation(data.originalRudderAction); // 원본 방향타 액션 값
                sensor.AddObservation(data.originalPowerAction);  // 원본 추진력 액션 값

                // C. 다른 선박의 COLREGs 상태와 위험 정보
                // 상대 선박의 COLREGs 상황 (원-핫 인코딩) (5)
                int sitIndex = (int)data.situation;
                for (int j = 0; j < 5; j++)
                {
                    sensor.AddObservation(j == sitIndex ? 1.0f : 0.0f);
                }

                // 위험도 (1)
                sensor.AddObservation(data.risk);

                // 추적 중인 선박 수 (1) - 복잡성 지표
                sensor.AddObservation(data.trackedVesselsCount);
            }
            else
            {
                // 통신 대상이 없는 경우 0으로 채움
                // A. Observation (7)
                sensor.AddObservation(Vector3.zero); // 상대적 위치 (3)
                sensor.AddObservation(Vector3.zero); // 상대적 속도 방향 (3)
                sensor.AddObservation(0f);          // 상대속도 크기 (1)

                // 선박 방향 (3)
                sensor.AddObservation(Vector3.zero);

                // B. Action (4)
                sensor.AddObservation(0f); // 현재 러더 속도 (1)
                sensor.AddObservation(0f); // 현재 파워 (1)
                sensor.AddObservation(0f); // 원본 러더 액션 (1)
                sensor.AddObservation(0f); // 원본 파워 액션 (1)

                // C. COLREGs 상황 (5) - None 상태로 원-핫 인코딩
                sensor.AddObservation(1.0f); // None에 1
                for (int j = 1; j < 5; j++)
                {
                    sensor.AddObservation(0.0f);
                }

                // 위험도와 추적 선박 수 (2)
                sensor.AddObservation(0f);
                sensor.AddObservation(0);
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

        // 궤적 시각화 (에디터에서)
        if (showTrajectory && trajectoryPoints.Count > 1)
        {
            Gizmos.color = trajectoryColor;
            for (int i = 1; i < trajectoryPoints.Count; i++)
            {
                Gizmos.DrawLine(trajectoryPoints[i-1], trajectoryPoints[i]);
            }
        }
    }

    /// <summary>
    /// 충돌 처리 함수입니다.
    /// </summary>
    /// <param name="col">충돌 정보</param>
    public void OnCollisionEnter(Collision col)
    {
       // Debug.Log("Collision detected with: " + col.gameObject.name + ", Tag: " + col.gameObject.tag);

        // 선박인지 장애물인지 컴포넌트로 구분
        if (col.collider.CompareTag("Obstacle"))
        {
            // 선박과 충돌한 경우 (다른 에이전트)
            if (col.collider.GetComponent<AgentFighter>() != null)
            {
               // Debug.Log("Collision with another vessel");
                AddReward(-5.0f); // 선박 충돌은 페널티가 적음
            }
            // 장애물과 충돌한 경우
            else
            {
                //Debug.Log("Collision with obstacle");
                AddReward(-10.0f); // 장애물 충돌은 더 큰 페널티
            }
            EndEpisode();
            //Debug.Log("Episode ended due to collision");
        }
    }

    /// <summary>
    /// Trigger 충돌 처리 함수입니다.
    /// </summary>
    public void OnTriggerEnter(Collider col)
    {
        //Debug.Log("Trigger detected with: " + col.gameObject.name + ", Tag: " + col.gameObject.tag);

        if (col.CompareTag("Obstacle"))
        {
            //Debug.Log("Trigger with obstacle");
            AddReward(-10.0f);
            EndEpisode();
            //Debug.Log("Episode ended due to trigger collision");
        }
    }

    private void UpdateTrajectory()
    {
        if (!showTrajectory) return;

        // 현재 위치를 궤적에 추가
        trajectoryPoints.Add(transform.position);

        // 최대 포인트 수 제한
        if (trajectoryPoints.Count > maxTrajectoryPoints)
        {
            trajectoryPoints.RemoveAt(0);
        }

        UpdateTrajectoryLine();
    }

    private void UpdateTrajectoryLine()
    {
        if (!showTrajectory) return;

        trajectoryLine.positionCount = trajectoryPoints.Count;
        trajectoryLine.SetPositions(trajectoryPoints.ToArray());
    }
}
