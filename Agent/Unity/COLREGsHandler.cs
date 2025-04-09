using UnityEngine;

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
        HeadOn,      // 정면 조우 상황
        CrossingGiveWay, // 횡단 시 피항선
        CrossingStandOn, // 횡단 시 유지선
        Overtaking   // 추월 상황
    }
    
    public COLREGsSituation currentSituation = COLREGsSituation.None;
    private GameObject detectedVessel;        // 감지된 가장 가까운 선박
    private Vector3 relativeVesselVelocity;   // 감지된 선박의 상대 속도
    private Vector3 obstacleDirection;        // 장애물 방향
    private float obstacleDistance;           // 장애물까지의 거리
    
    private Rigidbody rb;
    private AgentFighter agentFighter;
    
    private void Awake()
    {
        rb = GetComponent<Rigidbody>();
        agentFighter = GetComponent<AgentFighter>();
    }
    
    /// <summary>
    /// 다른 선박과의 상대 방향 및 위치에 따라 COLREGs 상황을 판단합니다.
    /// </summary>
    /// <param name="targetDirection">다른 선박의 방향 벡터</param>
    /// <param name="vessel">감지된 선박 게임 오브젝트</param>
    /// <param name="distance">선박까지의 거리</param>
    /// <returns>없음 - 상황은 currentSituation 필드에 저장됨</returns>
    public void DetermineCOLREGsSituation(Vector3 targetDirection, GameObject vessel, float distance)
    {
        obstacleDirection = targetDirection;
        obstacleDistance = distance;
        
        // 상대 선박과의 각도 계산
        float angle = Vector3.SignedAngle(transform.forward, targetDirection, Vector3.up);
        
        // 상대 선박의 속도 (가능하다면)
        Rigidbody vesselRb = vessel.GetComponent<Rigidbody>();
        Vector3 relativeVelocity = Vector3.zero;
        
        if (vesselRb != null)
        {
            relativeVelocity = vesselRb.velocity - rb.velocity;
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
    
    /// <summary>
    /// COLREGs 상황에 따른 보상값을 계산합니다.
    /// </summary>
    /// <param name="rudderSpeed">현재 선박의 방향타 속도</param>
    /// <returns>COLREGs 규정 준수에 따른 보상값</returns>
    public float CalculateCOLREGsReward(float rudderSpeed)
    {
        float reward = 0f;
        
        switch (currentSituation)
        {
            case COLREGsSituation.HeadOn:
                // 정면 조우 시 우현 변침 보상
                if (rudderSpeed < 0) // 우현으로 변침
                    reward += 0.05f;
                break;
                
            case COLREGsSituation.CrossingGiveWay:
                // 횡단 시 피항 보상
                if (rudderSpeed < 0) // 우현으로 변침하여 피항
                    reward += 0.05f;
                break;
                
            case COLREGsSituation.CrossingStandOn:
                // 횡단 시 진로 유지 보상
                if (Mathf.Abs(rudderSpeed) < 0.1f) // 직진 유지
                    reward += 0.03f;
                break;
                
            case COLREGsSituation.Overtaking:
                // 추월 시 적절한 행동 보상
                reward += 0.02f;
                break;
        }
        
        return reward;
    }
    
    /// <summary>
    /// COLREGs 상황을 시각적으로 표현합니다.
    /// </summary>
    public void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;
        
        // COLREGs 상황 시각화
        if (currentSituation != COLREGsSituation.None && obstacleDirection != Vector3.zero)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawSphere(transform.position + obstacleDirection * obstacleDistance, 0.3f);
        }
    }
}
