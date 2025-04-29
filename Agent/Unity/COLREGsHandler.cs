using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// COLREGs(국제해상충돌예방규칙) 상황 식별 및 처리를 담당하는 컴포넌트
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class COLREGsHandler : MonoBehaviour
{
    // COLREGs 상황 식별을 위한 열거형
    public enum COLREGsSituation
    {
        None,        // 특별한 상황 없음
        HeadOn,      // 정면 마주침 상황
        CrossingGiveWay, // 횡단 시 양보선
        CrossingStandOn, // 횡단 시 진로선
        Overtaking   // 추월 상황
    }

    // 최대 처리할 선박 수
    public int maxVesselsToTrack = 5;
    
    // 각 선박에 대한 COLREGs 상황 정보를 저장하는 구조체
    [System.Serializable]
    public struct VesselCOLREGsInfo
    {
        public GameObject vessel;              // 대상 선박
        public COLREGsSituation situation;     // COLREGs 상황
        public float risk;                     // 위험도 (0-1)
        public Vector3 direction;              // 대상 방향
        public float distance;                 // 거리
        public Vector3 relativeVelocity;       // 상대 속도
    }

    // 현재 추적 중인 모든 선박 정보 리스트
    public List<VesselCOLREGsInfo> trackedVessels = new List<VesselCOLREGsInfo>();
    
    // 레거시 코드 호환성을 위한 현재 상황 (가장 위험한 상황)
    public COLREGsSituation currentSituation = COLREGsSituation.None;
    
    private Rigidbody rb;
    private AgentFighter agentFighter;

    // 위험도 계산 파라미터
    [Header("Risk Assessment Parameters")]
    public float minRiskDistance = 10f;
    public float maxRiskDistance = 30f;
    public float dangerDistance = 20f;  // 위험 거리 추가
    public float relativeSpeedFactor = 0.5f;
    public float distanceFactor = 1.0f;
    public float headOnRiskMultiplier = 1.2f;
    public float crossingRiskMultiplier = 1.0f;
    public float overtakingRiskMultiplier = 0.7f;
    public float angleThreshold = 15f;

    private void Awake()
    {
        rb = GetComponent<Rigidbody>();
        agentFighter = GetComponent<AgentFighter>();
    }

    /// <summary>
    /// 다른 선박과의 상대 방향 및 위치에 따라 COLREGs 상황을 판단합니다.
    /// </summary>
    /// <param name="targetDirection">다른 선박의 방향 벡터</param>
    /// <param name="vessel">감지된 선박 게임오브젝트</param>
    /// <param name="distance">선박까지의 거리</param>
    /// <returns>없음 - 상황은 trackedVessels 리스트에 저장됨</returns>
    public void DetermineCOLREGsSituation(Vector3 targetDirection, GameObject vessel, float distance)
    {
        if (vessel == null) return;

        // 이미 추적 중인 선박인지 확인
        int existingIndex = -1;
        for (int i = 0; i < trackedVessels.Count; i++)
        {
            if (trackedVessels[i].vessel == vessel)
            {
                existingIndex = i;
                break;
            }
        }

        // 상대 각도 계산
        float angle = Vector3.SignedAngle(transform.forward, targetDirection, Vector3.up);

        // 상대 속도 계산
        Rigidbody vesselRb = vessel.GetComponent<Rigidbody>();
        Vector3 relativeVelocity = Vector3.zero;
        if (vesselRb != null)
        {
            relativeVelocity = vesselRb.velocity - rb.velocity;
        }

        float relativeSpeed = relativeVelocity.magnitude;
        bool isApproaching = Vector3.Dot(relativeVelocity, targetDirection) > 0;

        // COLREGs 상황 판단
        COLREGsSituation situation = COLREGsSituation.None;  // 초기화 추가

        // 상황 판단 개선
        if (isApproaching || distance < dangerDistance)
        {
            // 정면 마주침 (Head-on)
            if (Mathf.Abs(angle) < angleThreshold)
            {
                situation = COLREGsSituation.HeadOn;
            }
            // 교차 상황 (Crossing)
            else if (angle > angleThreshold && angle < 120f)
            {
                situation = COLREGsSituation.CrossingGiveWay;
            }
            else if (angle < -angleThreshold && angle > -120f)
            {
                situation = COLREGsSituation.CrossingStandOn;
            }
            // 추월 상황 (Overtaking)
            else if (Mathf.Abs(angle) > 120f)
            {
                situation = COLREGsSituation.Overtaking;
            }
        }

        // 위험도 계산 개선
        float risk = CalculateRisk(distance, relativeSpeed, angle, situation);

        // 새 정보 생성
        VesselCOLREGsInfo info = new VesselCOLREGsInfo
        {
            vessel = vessel,
            situation = situation,
            risk = risk,
            direction = targetDirection,
            distance = distance,
            relativeVelocity = relativeVelocity
        };

        // 리스트에 추가 또는 업데이트
        if (existingIndex >= 0)
        {
            trackedVessels[existingIndex] = info;
        }
        else
        {
            trackedVessels.Add(info);
        }

        // 리스트가 최대 크기를 초과하면 위험도가 낮은 선박부터 제거
        if (trackedVessels.Count > maxVesselsToTrack)
        {
            SortVesselsByRisk();
            trackedVessels.RemoveAt(trackedVessels.Count - 1);
        }

        // 가장 위험한 상황을 현재 상황으로 설정 (레거시 코드 호환성)
        UpdateCurrentSituation();
    }

    /// <summary>
    /// 위험도에 따라 선박 리스트를 정렬합니다.
    /// </summary>
    private void SortVesselsByRisk()
    {
        trackedVessels.Sort((a, b) => b.risk.CompareTo(a.risk));
    }

    /// <summary>
    /// 가장 위험한 상황을 현재 상황으로 업데이트합니다.
    /// </summary>
    private void UpdateCurrentSituation()
    {
        if (trackedVessels.Count == 0)
        {
            currentSituation = COLREGsSituation.None;
            return;
        }

        SortVesselsByRisk();
        currentSituation = trackedVessels[0].situation;
    }

    /// <summary>
    /// 거리, 상대 속도, 각도를 고려하여 위험도를 계산합니다.
    /// </summary>
    /// <param name="distance">대상까지의 거리</param>
    /// <param name="relativeSpeed">상대 속도 크기</param>
    /// <param name="angle">상대 각도</param>
    /// <param name="situation">COLREGs 상황</param>
    /// <returns>0-1 사이의 위험도 값</returns>
    private float CalculateRisk(float distance, float relativeSpeed, float angle, COLREGsSituation situation)
    {
        // 거리 기반 위험도
        float distanceRisk = 0;
        if (distance <= minRiskDistance)
        {
            distanceRisk = 1.0f;
        }
        else if (distance >= maxRiskDistance)
        {
            distanceRisk = 0.0f;
        }
        else
        {
            distanceRisk = 1.0f - ((distance - minRiskDistance) / (maxRiskDistance - minRiskDistance));
        }

        // 속도 기반 위험도
        float speedRisk = Mathf.Clamp01(relativeSpeed / 5.0f);

        // 상황별 가중치 적용
        float situationMultiplier = 1.0f;
        switch (situation)
        {
            case COLREGsSituation.HeadOn:
                situationMultiplier = headOnRiskMultiplier;
                break;
            case COLREGsSituation.CrossingGiveWay:
            case COLREGsSituation.CrossingStandOn:
                situationMultiplier = crossingRiskMultiplier;
                break;
            case COLREGsSituation.Overtaking:
                situationMultiplier = overtakingRiskMultiplier;
                break;
        }

        // 최종 위험도 계산 (가중치 적용)
        float risk = ((distanceRisk * distanceFactor) + (speedRisk * relativeSpeedFactor)) / (distanceFactor + relativeSpeedFactor);
        risk *= situationMultiplier;

        return Mathf.Clamp01(risk);
    }

    /// <summary>
    /// 주어진 방향타 속도와 COLREGs 상황을 고려하여 규정 준수 정도에 따른 보상을 계산합니다.
    /// </summary>
    /// <param name="rudderSpeed">현재 방향타 속도</param>
    /// <returns>계산된 보상 값</returns>
    public float CalculateCOLREGsReward(float rudderSpeed)
    {
        if (trackedVessels.Count == 0)
            return 0f;

        float totalReward = 0f;
        float totalWeight = 0f;

        foreach (VesselCOLREGsInfo info in trackedVessels)
        {
            if (info.risk <= 0.01f) continue;

            float reward = 0f;
            switch (info.situation)
            {
                case COLREGsSituation.HeadOn:
                    // 정면 마주침 시 우현쪽 피침 강화
                    if (rudderSpeed < -0.2f) // 우현으로 강하게 피함
                        reward = 0.1f;
                    else if (rudderSpeed > 0.2f) // 좌현으로 피하면 큰 패널티
                        reward = -0.1f;
                    break;

                case COLREGsSituation.CrossingGiveWay:
                    // 횡단 시 우현 피침 및 속도 감소
                    if (rudderSpeed < -0.2f) // 우현으로 피함
                        reward = 0.08f;
                    else if (rudderSpeed > 0.2f) // 좌현으로 가로지르면 큰 패널티
                        reward = -0.1f;
                    break;

                case COLREGsSituation.CrossingStandOn:
                    // 횡단 시 진로 유지
                    if (Mathf.Abs(rudderSpeed) < 0.1f) // 현재 진로 유지
                        reward = 0.05f;
                    else if (rudderSpeed > 0.3f) // 급격한 방향 전환 패널티
                        reward = -0.05f;
                    break;

                case COLREGsSituation.Overtaking:
                    // 추월 시 우현으로 추월
                    if (rudderSpeed < -0.1f) // 우현으로 추월
                        reward = 0.06f;
                    else if (rudderSpeed > 0.1f) // 좌현으로 추월 시 패널티
                        reward = -0.05f;
                    break;
            }

            // 위험도를 가중치로 사용
            totalReward += reward * info.risk;
            totalWeight += info.risk;
        }

        return totalWeight > 0 ? totalReward / totalWeight : 0f;
    }

    /// <summary>
    /// 특정 인덱스의 선박 COLREGs 정보를 반환합니다.
    /// </summary>
    /// <param name="index">선박 인덱스</param>
    /// <returns>해당 인덱스의 선박 정보</returns>
    public VesselCOLREGsInfo GetVesselInfo(int index)
    {
        if (index >= 0 && index < trackedVessels.Count)
            return trackedVessels[index];

        return new VesselCOLREGsInfo();
    }

    /// <summary>
    /// 추적 중인 선박 수를 반환합니다.
    /// </summary>
    /// <returns>추적 중인 선박 수</returns>
    public int GetTrackedVesselsCount()
    {
        return trackedVessels.Count;
    }

    /// <summary>
    /// COLREGs 상황을 시각적으로 표현합니다.
    /// </summary>
    public void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;

        // 모든 추적 중인 선박에 대한 시각화
        foreach (VesselCOLREGsInfo info in trackedVessels)
        {
            if (info.vessel == null) continue;

            // 위험도에 따른 색상 설정 (위험도가 높을수록 빨간색)
            Color vesselColor = Color.Lerp(Color.blue, Color.red, info.risk);
            vesselColor.a = 0.8f;
            Gizmos.color = vesselColor;
            
            // 선박 위치에 구체 표시
            Gizmos.DrawSphere(info.vessel.transform.position, 0.3f);
            
            // 이 선박과의 연결선 표시
            Gizmos.DrawLine(transform.position, info.vessel.transform.position);
            
            // 상황에 따른 추가 시각화
            switch(info.situation)
            {
                case COLREGsSituation.HeadOn:
                    // 정면 마주침 시 화살표 표시
                    DrawArrow(transform.position, info.vessel.transform.position, Color.red, 0.5f);
                    break;
                case COLREGsSituation.CrossingGiveWay:
                    // 양보해야 하는 경우 점선 화살표
                    DrawArrow(transform.position, info.vessel.transform.position, Color.yellow, 0.5f);
                    break;
                default:
                    break;
            }
        }
    }
    
    /// <summary>
    /// 디버깅을 위한 화살표 그리기 함수
    /// </summary>
    private void DrawArrow(Vector3 start, Vector3 end, Color color, float arrowHeadLength = 0.25f)
    {
        Gizmos.color = color;
        Gizmos.DrawLine(start, end);
        
        Vector3 direction = (end - start).normalized;
        Vector3 right = Quaternion.Euler(0, 30, 0) * -direction;
        Vector3 left = Quaternion.Euler(0, -30, 0) * -direction;
        
        Gizmos.DrawLine(end, end + right * arrowHeadLength);
        Gizmos.DrawLine(end, end + left * arrowHeadLength);
    }
}
