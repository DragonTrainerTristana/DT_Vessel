using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class VesselManager : MonoBehaviour
{
    [Header("선박 설정")]
    public GameObject vesselPrefab;        // 생성할 선박 프리팹
    public int vesselCount = 8;            // 생성할 선박 수
    public float minGoalDistance = 100f;   // 목표 지점까지의 최소 거리
    public GameObject vesselParent;
    
    [Header("스폰 포인트")]
    public List<Transform> spawnPoints = new List<Transform>();  // 가능한 모든 스폰 포인트
    
    // 생성된 선박 목록
    private List<GameObject> vessels = new List<GameObject>();
    // 사용된 스폰 포인트와 목표 포인트 추적
    private HashSet<int> usedSpawnIndices = new HashSet<int>();
    private Dictionary<GameObject, Transform> vesselGoals = new Dictionary<GameObject, Transform>();
    
    // 시작시 실행
    void Start()
    {
        InitializeVessels();
    }
    
    /// <summary>
    /// 선박들을 초기화하고 스폰
    /// </summary>
    public void InitializeVessels()
    {
        // 기존 선박들 제거
        foreach (var vessel in vessels)
        {
            if (vessel != null)
                Destroy(vessel);
        }
        
        vessels.Clear();
        usedSpawnIndices.Clear();
        vesselGoals.Clear();
        
        // 스폰 포인트가 충분한지 확인
        if (spawnPoints.Count < vesselCount)
        {
            //Debug.LogError($"스폰 포인트({spawnPoints.Count}개)가 필요한 선박 수({vesselCount}개)보다 적습니다!");
            return;
        }
        
        // 랜덤 위치에 선박 생성
        for (int i = 0; i < vesselCount; i++)
        {
            SpawnVessel();
        }
    }
    
    /// <summary>
    /// 선박을 랜덤 위치에 스폰하고 목표 위치 할당
    /// </summary>
    private GameObject SpawnVessel()
    {
        // 사용되지 않은 랜덤 스폰 포인트 선택
        int spawnIndex = GetRandomUnusedSpawnIndex();
        if (spawnIndex == -1)
            return null;
        
        Transform spawnPoint = spawnPoints[spawnIndex];
        usedSpawnIndices.Add(spawnIndex);
        
        // vesselParent를 부모로 사용하여 선박 생성
        GameObject vessel = Instantiate(vesselPrefab, spawnPoint.position, spawnPoint.rotation, vesselParent.transform);
        vessels.Add(vessel);
        
        // 이름 부여
        vessel.name = $"Vessel_{vessels.Count}";
        
        // VesselDynamics 컴포넌트 확인/추가
        VesselDynamics dynamics = vessel.GetComponent<VesselDynamics>();
        if (dynamics == null)
        {
            dynamics = vessel.AddComponent<VesselDynamics>();
        }
        
        // VesselAgent가 이미 있는지 확인하고 없을 때만 추가
        VesselAgent agent = vessel.GetComponent<VesselAgent>();
        if (agent == null)
        {
            agent = vessel.AddComponent<VesselAgent>();
            agent.vesselDynamics = dynamics;
        }
        
        // 목표 위치 할당
        AssignGoalForVessel(vessel, spawnPoint);
        
        return vessel;
    }
    
    /// <summary>
    /// 사용되지 않은 랜덤 스폰 인덱스 가져오기
    /// </summary>
    private int GetRandomUnusedSpawnIndex()
    {
        // 모든 스폰 포인트가 사용 중인 경우
        if (usedSpawnIndices.Count >= spawnPoints.Count)
            return -1;
        
        // 사용되지 않은 인덱스 중에서 랜덤 선택
        List<int> availableIndices = new List<int>();
        for (int i = 0; i < spawnPoints.Count; i++)
        {
            if (!usedSpawnIndices.Contains(i))
                availableIndices.Add(i);
        }
        
        int randomIndex = Random.Range(0, availableIndices.Count);
        return availableIndices[randomIndex];
    }
    
    /// <summary>
    /// 선박에 목표 위치 할당
    /// </summary>
    private void AssignGoalForVessel(GameObject vessel, Transform spawnPoint)
    {
        // 스폰 위치로부터 최소 거리(minGoalDistance) 이상 떨어진 포인트들 찾기
        List<Transform> validGoalPoints = new List<Transform>();
        
        foreach (Transform point in spawnPoints)
        {
            float distance = Vector3.Distance(spawnPoint.position, point.position);
            if (distance >= minGoalDistance)
                validGoalPoints.Add(point);
        }
        
        // 유효한 목표 포인트가 없는 경우
        if (validGoalPoints.Count == 0)
        {
            //Debug.LogWarning($"{vessel.name}의 유효한 목표 포인트가 없습니다. 최소 거리를 줄여보세요.");
            // 최소한 아무 포인트나 선택
            validGoalPoints = new List<Transform>(spawnPoints);
        }
        
        // 랜덤 목표 포인트 선택
        int goalIndex = Random.Range(0, validGoalPoints.Count);
        Transform goalPoint = validGoalPoints[goalIndex];
        
        // 선박에 목표 할당
        vesselGoals[vessel] = goalPoint;
        
        VesselAgent agent = vessel.GetComponent<VesselAgent>();
        if (agent != null)
        {
            agent.SetGoal(goalPoint.position);
            agent.goalPointName = goalPoint.name;
            agent.goalPointIndex = goalIndex;
            Debug.Log($"{vessel.name}의 목표 위치: {goalPoint.name} (인덱스: {goalIndex})");
        }
    }
    
    /// <summary>
    /// 모든 선박의 목표 위치 시각화
    /// </summary>
    void OnDrawGizmos()
    {
        // 게임 실행 중에만 라인 그리기
        if (!Application.isPlaying)
            return;
            
        foreach (var vessel in vessels)
        {
            if (vessel != null && vesselGoals.ContainsKey(vessel))
            {
                // 선박과 목표 지점 사이에 라인 그리기
                Gizmos.color = Color.yellow;
                Gizmos.DrawLine(vessel.transform.position, vesselGoals[vessel].position);
                
                // 목표 지점 표시
                Gizmos.color = Color.red;
                Gizmos.DrawSphere(vesselGoals[vessel].position, 1.5f);
            }
        }
    }
    
    /// <summary>
    /// 콘솔에서 선박 상태 출력
    /// </summary>
    public void LogVesselStatus()
    {
        //Debug.Log($"총 선박 수: {vessels.Count}");
        foreach (var vessel in vessels)
        {
            if (vessel != null && vesselGoals.ContainsKey(vessel))
            {
                float distance = Vector3.Distance(vessel.transform.position, vesselGoals[vessel].position);
                //Debug.Log($"{vessel.name} → {vesselGoals[vessel].name}, 남은 거리: {distance:F1}m");
            }
        }
    }
    
    /// <summary>
    /// 선박 수 변경 (에디터에서 조정 가능)
    /// </summary>
    public void ChangeVesselCount(int newCount)
    {
        vesselCount = Mathf.Clamp(newCount, 1, spawnPoints.Count);
        InitializeVessels();
    }
    
    /// <summary>
    /// 선박 재스폰 및 새 목표 설정
    /// </summary>
    public void RespawnVessel(GameObject vessel)
    {
        if (vessel == null || !vessels.Contains(vessel))
            return;
            
        // 기존에 사용된 스폰 위치와 목표 위치 정리
        if (vesselGoals.ContainsKey(vessel))
        {
            vesselGoals.Remove(vessel);
        }
        
        // 새로운 스폰 포인트 선택
        int spawnIndex = GetRandomUnusedSpawnIndex();
        if (spawnIndex == -1)
        {
            // 모든 스폰 포인트가 사용 중이면 초기화
            usedSpawnIndices.Clear();
            spawnIndex = GetRandomUnusedSpawnIndex();
        }
        
        Transform spawnPoint = spawnPoints[spawnIndex];
        usedSpawnIndices.Add(spawnIndex);
        
        // 선박 위치 및 회전 초기화
        vessel.transform.position = spawnPoint.position;
        vessel.transform.rotation = spawnPoint.rotation;
        
        // 물리 상태 초기화
        Rigidbody rb = vessel.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
        
        // 동역학 상태 초기화
        VesselDynamics dynamics = vessel.GetComponent<VesselDynamics>();
        if (dynamics != null)
        {
            dynamics.ResetState();
        }
        
        // 새 목표 할당
        AssignGoalForVessel(vessel, spawnPoint);
    }
} 