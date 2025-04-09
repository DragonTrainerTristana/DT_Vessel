using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 선박 간 통신 시스템을 관리하는 컴포넌트입니다.
/// 주변 선박 감지 및 정보 공유 기능을 담당합니다.
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class VesselCommunicationSystem : MonoBehaviour
{
    public float communicationRange = 15f; // 통신 범위
    public int maxCommunicationTargets = 4; // 최대 통신 대상 수
    public bool visualizeCommunication = true; // 통신 범위 시각화 여부
    public Color communicationRangeColor = new Color(0f, 0.5f, 1f, 0.2f); // 반투명 파란색
    
    private List<AgentFighter> communicationTargets = new List<AgentFighter>(); // 현재 통신 중인 대상
    private List<float> communicationDistances = new List<float>(); // 통신 대상과의 거리
    
    private AgentFighter agentFighter;
    private Rigidbody rb;
    private COLREGsHandler colregsHandler;
    
    // 다른 에이전트와 공유할 정보 구조체
    [System.Serializable]
    public struct CommunicationData
    {
        public Vector3 position;      // 위치
        public Vector3 velocity;      // 속도
        public Vector3 forward;       // 전방 방향
        public float rudderSpeed;     // 방향타 속도
        public float power;           // 추진력
        public COLREGsHandler.COLREGsSituation situation; // COLREGs 상황
    }
    
    private CommunicationData myData;
    private CommunicationData[] receivedData;
    
    private void Awake()
    {
        agentFighter = GetComponent<AgentFighter>();
        rb = GetComponent<Rigidbody>();
        colregsHandler = GetComponent<COLREGsHandler>();
        
        // 통신 데이터 초기화
        receivedData = new CommunicationData[maxCommunicationTargets];
    }
    
    private void Update()
    {
        // 통신 대상 찾기 및 데이터 공유 (매 프레임마다 할 필요는 없을 수 있음)
        if (Time.frameCount % 5 == 0) // 5프레임마다 실행하여 성능 최적화
        {
            FindCommunicationTargets();
            ShareData();
        }
    }
    
    /// <summary>
    /// 통신 범위 내 가장 가까운 선박들을 찾아 목록에 저장합니다.
    /// </summary>
    /// <returns>없음 - 결과는 communicationTargets 목록에 저장됨</returns>
    public void FindCommunicationTargets()
    {
        communicationTargets.Clear();
        communicationDistances.Clear();
        
        // 모든 AgentFighter 객체 찾기
        AgentFighter[] allAgents = FindObjectsOfType<AgentFighter>();
        
        foreach (AgentFighter agent in allAgents)
        {
            // 자기 자신은 제외
            if (agent == this.agentFighter) continue;
            
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
    
    /// <summary>
    /// 내 데이터를 업데이트하고 다른 선박과 공유합니다.
    /// </summary>
    /// <returns>없음</returns>
    public void ShareData()
    {
        // 내 데이터 업데이트
        myData = new CommunicationData
        {
            position = transform.position,
            velocity = rb.velocity,
            forward = transform.forward,
            rudderSpeed = agentFighter.rudderSpeed,
            power = agentFighter.power,
            situation = colregsHandler.currentSituation
        };
        
        // 다른 선박과 데이터 공유
        for (int i = 0; i < communicationTargets.Count; i++)
        {
            // 통신 시스템 컴포넌트 가져오기
            VesselCommunicationSystem targetComSystem = 
                communicationTargets[i].GetComponent<VesselCommunicationSystem>();
                
            if (targetComSystem != null)
            {
                // 데이터 전송
                targetComSystem.ReceiveData(i, myData);
            }
            
            // 데이터 수신 (양방향 통신)
            if (i < receivedData.Length && targetComSystem != null)
            {
                receivedData[i] = targetComSystem.GetCommunicationData();
            }
        }
    }
    
    /// <summary>
    /// 다른 선박으로부터 데이터를 수신합니다.
    /// </summary>
    /// <param name="index">통신 대상 인덱스</param>
    /// <param name="data">수신한 통신 데이터</param>
    /// <returns>없음</returns>
    public void ReceiveData(int index, CommunicationData data)
    {
        if (index < receivedData.Length)
        {
            receivedData[index] = data;
        }
    }
    
    /// <summary>
    /// 현재 선박의 통신 데이터를 반환합니다.
    /// </summary>
    /// <returns>선박의 현재 상태 정보</returns>
    public CommunicationData GetCommunicationData()
    {
        return myData;
    }
    
    /// <summary>
    /// 현재 통신 중인 대상 수를 반환합니다.
    /// </summary>
    /// <returns>통신 중인 선박 수</returns>
    public int GetCommunicationTargetCount()
    {
        return communicationTargets.Count;
    }
    
    /// <summary>
    /// 특정 인덱스의 통신 데이터를 반환합니다.
    /// </summary>
    /// <param name="index">통신 대상 인덱스</param>
    /// <returns>해당 인덱스의 통신 데이터</returns>
    public CommunicationData GetReceivedData(int index)
    {
        if (index < receivedData.Length)
            return receivedData[index];
            
        return new CommunicationData();
    }
    
    /// <summary>
    /// 통신 범위 및 연결을 시각적으로 표현합니다.
    /// </summary>
    public void OnDrawGizmos()
    {
        if (!Application.isPlaying || !visualizeCommunication) return;
        
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
