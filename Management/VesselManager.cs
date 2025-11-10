using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class VesselManager : MonoBehaviour
{
    [Header("Spawn Settings")]
    public List<Transform> spawnPoints = new List<Transform>();

    [Header("Goal Settings")]
    public List<Transform> goalPoints = new List<Transform>();  // 별도의 목표 지점들

    [Header("Vessel Settings")]
    public GameObject vesselPrefab;
    public int vesselCount = 4;
    public float minGoalDistance = 50f;
    public GameObject vesselParent;

    private List<GameObject> vessels = new List<GameObject>();
    private List<VesselAgent> vesselAgents = new List<VesselAgent>();
    private HashSet<int> usedSpawnIndices = new HashSet<int>();
    private Dictionary<GameObject, Transform> vesselGoals = new Dictionary<GameObject, Transform>();
    private Dictionary<GameObject, int> vesselSpawnIndices = new Dictionary<GameObject, int>();

    public List<VesselAgent> GetAllVesselAgents()
    {
        return vesselAgents;
    }

    void Start()
    {
        InitializeVessels();
    }

    public void InitializeVessels()
    {
        foreach (var vessel in vessels)
        {
            if (vessel != null) Destroy(vessel);
        }

        vessels.Clear();
        vesselAgents.Clear();
        usedSpawnIndices.Clear();
        vesselGoals.Clear();
        vesselSpawnIndices.Clear();

        if (spawnPoints.Count < vesselCount)
        {
            Debug.LogWarning($"Not enough spawn points! Need {vesselCount}, have {spawnPoints.Count}");
            return;
        }

        if (goalPoints.Count == 0)
        {
            Debug.LogWarning("No goal points set! Will use spawn points as goals (not recommended)");
        }

        for (int i = 0; i < vesselCount; i++)
        {
            SpawnVessel();
        }
    }

    private GameObject SpawnVessel()
    {
        // 수정


        int spawnIndex = GetRandomUnusedSpawnIndex();
        if (spawnIndex == -1) return null;

        Transform spawnPoint = spawnPoints[spawnIndex];
        usedSpawnIndices.Add(spawnIndex);

        // 일단 기본 위치에 생성
        GameObject vessel = Instantiate(vesselPrefab, spawnPoint.position, spawnPoint.rotation, vesselParent.transform);
        vessels.Add(vessel);
        vessel.name = $"Vessel_{vessels.Count}";

        VesselDynamics dynamics = vessel.GetComponent<VesselDynamics>();
        if (dynamics == null) dynamics = vessel.AddComponent<VesselDynamics>();

        VesselAgent agent = vessel.GetComponent<VesselAgent>();
        if (agent != null) vesselAgents.Add(agent);

        vesselSpawnIndices[vessel] = spawnIndex;

        // 목표 할당 후 목표를 향해 회전
        AssignGoalForVessel(vessel, spawnPoint);
        RotateVesselTowardsGoal(vessel);

        return vessel;
    }

    private int GetRandomUnusedSpawnIndex()
    {
        if (usedSpawnIndices.Count >= spawnPoints.Count) return -1;

        List<int> availableIndices = new List<int>();
        for (int i = 0; i < spawnPoints.Count; i++)
        {
            if (!usedSpawnIndices.Contains(i)) availableIndices.Add(i);
        }

        int randomIndex = Random.Range(0, availableIndices.Count);
        return availableIndices[randomIndex];
    }

    private void AssignGoalForVessel(GameObject vessel, Transform spawnPoint)
    {
        // goalPoints가 설정되어 있으면 사용, 없으면 spawnPoints 사용 (호환성)
        List<Transform> availableGoals = goalPoints.Count > 0 ? goalPoints : spawnPoints;
        List<Transform> validGoalPoints = new List<Transform>();

        // 스폰 위치에서 충분히 먼 목표점들만 선택
        foreach (Transform point in availableGoals)
        {
            float distance = Vector3.Distance(spawnPoint.position, point.position);
            if (distance >= minGoalDistance) validGoalPoints.Add(point);
        }

        // 거리 조건을 만족하는 목표점이 없으면 모든 목표점 사용
        if (validGoalPoints.Count == 0)
        {
            validGoalPoints = new List<Transform>(availableGoals);
        }

        // 랜덤하게 목표점 선택
        int goalIndex = Random.Range(0, validGoalPoints.Count);
        Transform goalPoint = validGoalPoints[goalIndex];

        vesselGoals[vessel] = goalPoint;

        VesselAgent agent = vessel.GetComponent<VesselAgent>();
        if (agent != null)
        {
            agent.SetGoal(goalPoint.position);
            agent.goalPointName = goalPoint.name;
            agent.goalPointIndex = goalIndex;
        }
    }

    private void RotateVesselTowardsGoal(GameObject vessel)
    {
        if (!vesselGoals.ContainsKey(vessel)) return;

        Transform goal = vesselGoals[vessel];
        Vector3 direction = (goal.position - vessel.transform.position).normalized;

        // Y축 회전만 적용 (선박은 수평면에서만 회전)
        direction.y = 0;

        if (direction != Vector3.zero)
        {
            Quaternion targetRotation = Quaternion.LookRotation(direction);

            // 랜덤 각도 오차 추가 (-10도 ~ +10도)
            float randomAngleOffset = Random.Range(-10f, 10f);
            Quaternion randomRotation = Quaternion.Euler(0, randomAngleOffset, 0);

            // 목표 방향에 랜덤 오차를 적용
            vessel.transform.rotation = targetRotation * randomRotation;
        }
    }

    void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;

        foreach (var vessel in vessels)
        {
            if (vessel != null && vesselGoals.ContainsKey(vessel))
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawLine(vessel.transform.position, vesselGoals[vessel].position);

                Gizmos.color = Color.red;
                Gizmos.DrawSphere(vesselGoals[vessel].position, 1.5f);
            }
        }
    }

    public void ChangeVesselCount(int newCount)
    {
        vesselCount = Mathf.Clamp(newCount, 1, spawnPoints.Count);
        InitializeVessels();
    }

    public void RespawnVessel(GameObject vessel)
    {
        if (vessel == null || !vessels.Contains(vessel)) return;

        if (vesselSpawnIndices.ContainsKey(vessel))
        {
            int oldIndex = vesselSpawnIndices[vessel];
            usedSpawnIndices.Remove(oldIndex);
        }

        if (vesselGoals.ContainsKey(vessel))
        {
            vesselGoals.Remove(vessel);
        }

        int spawnIndex = GetRandomUnusedSpawnIndex();
        if (spawnIndex == -1)
        {
            spawnIndex = Random.Range(0, spawnPoints.Count);
        }

        Transform spawnPoint = spawnPoints[spawnIndex];
        usedSpawnIndices.Add(spawnIndex);
        vesselSpawnIndices[vessel] = spawnIndex;

        vessel.transform.position = spawnPoint.position;
        vessel.transform.rotation = spawnPoint.rotation;

        Rigidbody rb = vessel.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        VesselDynamics dynamics = vessel.GetComponent<VesselDynamics>();
        if (dynamics != null) dynamics.ResetState();

        // 목표 할당 후 목표를 향해 회전
        AssignGoalForVessel(vessel, spawnPoint);
        RotateVesselTowardsGoal(vessel);
    }
}
