using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class AgentFighter : Agent
{
    public float maxPower = 2f;
    public float maxSpeed = 3.0f;
    public float maxRudderSpeed = 0.9f;

    Rigidbody rd;
    float rdVelocity;

    public int stepCount;
    private int stepInterval;

    private float initialGoalDistance;
    public float goalDistance;
    float preDistance;
    public float power;
    public float rudderSpeed;

    public Transform myInitialPos;
    public GameObject[] myGoals;
    private GameObject myGoal;
    public int randomIndex;

    public int numberOfRays = 120;
    public float rayLength = 10;
    private float fieldOfView = 360f;

    // COLREGs 상황 식별을 위한 변수
    public enum COLREGsSituation
    {
        None,
        HeadOn,
        CrossingGiveWay,
        CrossingStandOn,
        Overtaking
    }
    
    public COLREGsSituation currentSituation = COLREGsSituation.None;
    private GameObject detectedVessel;
    private Vector3 relativeVesselVelocity;

    // 통신 관련 변수
    public float communicationRange = 15f; // 통신 범위
    public int maxCommunicationTargets = 4; // 최대 통신 대상 수
    public bool visualizeCommunication = true; // 통신 범위 시각화 여부
    public Color communicationRangeColor = new Color(0f, 0.5f, 1f, 0.2f); // 반투명 파란색
    
    private List<AgentFighter> communicationTargets = new List<AgentFighter>(); // 현재 통신 중인 대상
    private List<float> communicationDistances = new List<float>(); // 통신 대상과의 거리
    
    // 다른 에이전트와 공유할 정보 구조체
    [System.Serializable]
    public struct CommunicationData
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 forward;
        public float rudderSpeed;
        public float power;
        public COLREGsSituation situation;
    }
    
    private CommunicationData myData;
    private CommunicationData[] receivedData;

    public override void Initialize()
    {
        rd = GetComponent<Rigidbody>();
        previousRayHits = new float[numberOfRays];
        currentRayHits = new float[numberOfRays];
        
        // 초기값 설정
        for (int i = 0; i < numberOfRays; i++)
        {
            previousRayHits[i] = rayLength;
            currentRayHits[i] = rayLength;
        }
        
        // 통신 데이터 초기화
        receivedData = new CommunicationData[maxCommunicationTargets];
    }

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

        rd.velocity = Vector3.zero;
        rd.angularVelocity = Vector3.zero;
    }

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
    
        float powerAction = actions.ContinuousActions[0];
        float rudderAction = actions.ContinuousActions[1];

        AgentMovement(powerAction, rudderAction);

        Vector3 targetDirection = (myGoal.transform.position - this.gameObject.transform.position).normalized;
        float angleDifference = Vector3.Angle(transform.forward, targetDirection);

        if (rd.velocity.magnitude == 0)
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

        // COLREGs 상황에 따른 보상 적용
        if (currentSituation != COLREGsSituation.None)
        {
            ApplyCOLREGsReward();
        }
    }
    public void AgentMovement(float powerAction, float rudderAction)
    {
        power = maxPower * ((powerAction + 1f) / 2f);

        float desiredRudderSpeed = maxRudderSpeed * rudderAction;
        rudderSpeed = Mathf.MoveTowards(rudderSpeed, desiredRudderSpeed, Time.deltaTime * 1.0f);

        if (rd.velocity.magnitude > maxSpeed)
        {
            rd.velocity *= 0.98f;
        }
            
        Quaternion turnRotation = Quaternion.Euler(0f, rudderSpeed, 0f);
        rd.MoveRotation(rd.rotation * turnRotation);

    
        Vector3 force = transform.forward * power;
        rdVelocity = force.z;
        rd.AddForce(force, ForceMode.VelocityChange);
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        float angleStep = fieldOfView / numberOfRays;
        Vector3 forward = transform.forward; 
        Vector3 startDirection = Quaternion.Euler(0, -fieldOfView / 2, 0) * forward;
        
        bool vesselDetected = false;
        float closestVesselDistance = rayLength;
        Vector3 closestVesselDirection = Vector3.zero;
        GameObject closestVessel = null;

        for (int i = 0; i < numberOfRays; i++)
        {
            Vector3 rayDirection = Quaternion.Euler(0, angleStep * i, 0) * startDirection;
            RaycastHit hit;

            float distance = rayLength;
            if (Physics.Raycast(transform.position, rayDirection, out hit, rayLength))
            {
                distance = hit.distance;
                
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

            sensor.AddObservation(distance / rayLength); 
        }
        
        // 가장 가까운 선박에 대한 COLREGs 상황 판단
        if (vesselDetected && closestVessel != null)
        {
            detectedVessel = closestVessel;
            DetermineCOLREGsSituation(closestVesselDirection, closestVessel);
            
            // 현재 COLREGs 상황을 관측에 추가 (원-핫 인코딩)
            sensor.AddOneHot((int)currentSituation, 5); // None 포함 5가지 상태
        }
        else
        {
            currentSituation = COLREGsSituation.None;
            // 상황이 없을 때 원-핫 인코딩
            sensor.AddOneHot(0, 5);
        }

        sensor.AddObservation(transform.forward); 
        sensor.AddObservation(transform.rotation.y);
        UnityEngine.Vector3 relativePosition = myGoal.transform.position - this.gameObject.transform.position;
        sensor.AddObservation(relativePosition.magnitude);
        sensor.AddObservation(rd.velocity.magnitude);

        // 통신으로 받은 데이터를 관측에 추가
        for (int i = 0; i < maxCommunicationTargets; i++)
        {
            if (i < communicationTargets.Count)
            {
                // 각 통신 대상에 대한 상대적 위치 (3)
                Vector3 relativePos = communicationTargets[i].transform.position - transform.position;
                sensor.AddObservation(relativePos.normalized);
                
                // 상대적 속도 (3)
                Vector3 relativeVel = receivedData[i].velocity - rd.velocity;
                sensor.AddObservation(relativeVel.normalized);
                
                // 상대 선박의 방향 (3)
                sensor.AddObservation(receivedData[i].forward);
                
                // 상대 선박의 제어 상태 (2)
                sensor.AddObservation(receivedData[i].rudderSpeed);
                sensor.AddObservation(receivedData[i].power);
                
                // 상대 선박의 COLREGs 상황 (원-핫 인코딩) (5)
                sensor.AddOneHot((int)receivedData[i].situation, 5);
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
        
        // COLREGs 상황 시각화
        if (currentSituation != COLREGsSituation.None && obstacleDirection != Vector3.zero)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawSphere(transform.position + obstacleDirection * obstacleDistance, 0.3f);
        }
        
        // 통신 범위 시각화
        if (visualizeCommunication)
        {
            Gizmos.color = communicationRangeColor;
            Gizmos.DrawSphere(transform.position, communicationRange);
            
            // 통신 대상과의 연결선
            Gizmos.color = Color.blue;
            foreach (AgentFighter target in communicationTargets)
            {
                if (target != null)
                {
                    Gizmos.DrawLine(transform.position, target.transform.position);
                }
            }
        }
    }

    public void OnCollisionEnter(Collision col)
    {
        if (col.collider.CompareTag("Obstacle"))
        {
            AddReward(-10.0f);
            EndEpisode();
        }
    }

    private void DetermineCOLREGsSituation(Vector3 targetDirection, GameObject vessel)
    {
        // 상대 선박과의 각도 계산
        float angle = Vector3.SignedAngle(transform.forward, targetDirection, Vector3.up);
        
        // 상대 선박의 속도 (가능하다면)
        Rigidbody vesselRb = vessel.GetComponent<Rigidbody>();
        Vector3 relativeVelocity = Vector3.zero;
        
        if (vesselRb != null)
        {
            relativeVelocity = vesselRb.velocity - rd.velocity;
            relativeVesselVelocity = relativeVelocity;
        }
        
        // COLREGs 상황 판단
        // 1. Head-On (정면 조우): 선박이 서로 마주보며 접근
        if (Mathf.Abs(angle) < 15f)
        {
            currentSituation = COLREGsSituation.HeadOn;
        }
        // 2. Crossing-Give Way: 다른 선박이 우현(오른쪽)에서 접근
        else if (angle > 15f && angle < 120f)
        {
            currentSituation = COLREGsSituation.CrossingGiveWay;
        }
        // 3. Crossing-Stand On: 다른 선박이 좌현(왼쪽)에서 접근
        else if (angle < -15f && angle > -120f)
        {
            currentSituation = COLREGsSituation.CrossingStandOn;
        }
        // 4. Overtaking: 다른 선박을 추월
        else if (Mathf.Abs(angle) > 120f)
        {
            currentSituation = COLREGsSituation.Overtaking;
        }
    }
    
    // COLREGs 상황에 따른 보상 함수
    private void ApplyCOLREGsReward()
    {
        switch (currentSituation)
        {
            case COLREGsSituation.HeadOn:
                // 정면 조우 시 우현 변침 보상
                if (rudderSpeed < 0) // 우현으로 변침
                    AddReward(0.05f);
                break;
                
            case COLREGsSituation.CrossingGiveWay:
                // 횡단 시 피항 보상
                if (rudderSpeed < 0) // 우현으로 변침하여 피항
                    AddReward(0.05f);
                break;
                
            case COLREGsSituation.CrossingStandOn:
                // 횡단 시 진로 유지 보상
                if (Mathf.Abs(rudderSpeed) < 0.1f) // 직진 유지
                    AddReward(0.03f);
                break;
                
            case COLREGsSituation.Overtaking:
                // 추월 시 적절한 행동 보상
                // 실제 규정에 맞게 구체적인 보상 정책 수립 필요
                AddReward(0.02f);
                break;
        }
    }

    // 매 프레임마다 실행
    private void Update()
    {
        // 통신 대상 찾기 및 데이터 공유 (매 프레임마다 할 필요는 없을 수 있음)
        if (Time.frameCount % 5 == 0) // 5프레임마다 실행하여 성능 최적화
        {
            FindCommunicationTargets();
            ShareData();
        }
    }
    
    // 통신 범위 내 가장 가까운 선박들 찾기
    private void FindCommunicationTargets()
    {
        communicationTargets.Clear();
        communicationDistances.Clear();
        
        // 모든 AgentFighter 객체 찾기
        AgentFighter[] allAgents = FindObjectsOfType<AgentFighter>();
        
        foreach (AgentFighter agent in allAgents)
        {
            // 자기 자신은 제외
            if (agent == this) continue;
            
            float distance = Vector3.Distance(transform.position, agent.transform.position);
            
            // 통신 범위 내에 있는 경우
            if (distance <= communicationRange)
            {
                // 정렬된 위치에 삽입
                int insertIndex = 0;
                while (insertIndex < communicationDistances.Count && 
                       distance > communicationDistances[insertIndex])
                {
                    insertIndex++;
                }
                
                communicationDistances.Insert(insertIndex, distance);
                communicationTargets.Insert(insertIndex, agent);
                
                // 최대 통신 대상 수 유지
                if (communicationTargets.Count > maxCommunicationTargets)
                {
                    communicationTargets.RemoveAt(maxCommunicationTargets);
                    communicationDistances.RemoveAt(maxCommunicationTargets);
                }
            }
        }
    }
    
    // 내 데이터 업데이트 및 다른 선박과 공유
    private void ShareData()
    {
        // 내 데이터 업데이트
        myData = new CommunicationData
        {
            position = transform.position,
            velocity = rd.velocity,
            forward = transform.forward,
            rudderSpeed = rudderSpeed,
            power = power,
            situation = currentSituation
        };
        
        // 다른 선박과 데이터 공유
        for (int i = 0; i < communicationTargets.Count; i++)
        {
            // 실제 멀티에이전트 환경에서는 이 부분이 네트워크 메시지가 될 수 있음
            communicationTargets[i].ReceiveData(i, myData, this);
            
            // 데이터 수신 (양방향 통신)
            if (i < receivedData.Length)
            {
                receivedData[i] = communicationTargets[i].GetCommunicationData();
            }
        }
    }
    
    // 다른 선박으로부터 데이터 수신
    public void ReceiveData(int index, CommunicationData data, AgentFighter sender)
    {
        if (index < receivedData.Length)
        {
            receivedData[index] = data;
        }
    }
    
    // 통신용 데이터 반환
    public CommunicationData GetCommunicationData()
    {
        return myData;
    }
}
