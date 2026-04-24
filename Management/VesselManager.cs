using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
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
    public float minGoalDistance = 5f;    // 1/10 스케일 (원본 50m)
    public GameObject vesselParent;

    [Header("NavMesh Pathfinding")]
    public WaypointPathFinder pathFinder;       // Inspector에서 할당
    public bool useNavMeshPathfinding = true;   // NavMesh 경로탐색 사용 여부

    [Header("Spawn Safety")]
    public float minSpawnDistance = 1.5f;         // 다른 선박과의 최소 스폰 거리 (1/10 스케일, 원본 15m)

    [Header("NavMesh Snapping")]
    [Tooltip("스폰/목표 좌표를 가장 가까운 NavMesh로 스냅할 때 검색 반경. Cube가 land 근처여도 자동 보정됨.")]
    public float navMeshSnapRadius = 10f;         // 1/10 스케일 (원본 100m)

    [Header("Progressive Spawn (Simulation Mode)")]
    [Tooltip("체크하면 초기 N대 스폰 후 일정 간격으로 M대씩 추가 스폰 (세계지도 트래픽 시뮬)")]
    public bool progressiveSpawnEnabled = false;
    [Tooltip("초기에 스폰할 배 수")]
    public int initialSpawnCount = 100;
    [Tooltip("추가 스폰 배치 크기")]
    public int spawnBatchSize = 100;
    [Tooltip("추가 스폰 간격 (초)")]
    public float spawnIntervalSec = 60f;
    [Tooltip("최대 배 수 (도달 후 추가 스폰 중단)")]
    public int maxVesselCount = 1000;
    [Tooltip("배치 내부에서 프레임당 스폰 개수 (한 프레임에 100대 몰아 스폰하면 hitch)")]
    public int spawnPerFrame = 5;

    private List<GameObject> vessels = new List<GameObject>();
    private List<VesselAgent> vesselAgents = new List<VesselAgent>();
    private HashSet<int> usedSpawnIndices = new HashSet<int>();
    private Dictionary<GameObject, Transform> vesselGoals = new Dictionary<GameObject, Transform>();
    private Dictionary<GameObject, int> vesselSpawnIndices = new Dictionary<GameObject, int>();

    // 재사용 버퍼 (1000대 스폰 시 GC spike 방지)
    private readonly List<Vector3> _existingPosBuffer = new List<Vector3>(1024);
    private readonly List<int> _availableIndicesBuffer = new List<int>(64);
    private readonly List<Transform> _validGoalBuffer = new List<Transform>(64);

    public List<VesselAgent> GetAllVesselAgents()
    {
        return vesselAgents;
    }

    void Awake()
    {
        // Prefab Inspector 값 무시하고 GlobalScale로 강제 덮어쓰기
        minGoalDistance = GlobalScale.MIN_GOAL_DISTANCE;
        minSpawnDistance = GlobalScale.MIN_SPAWN_DISTANCE;
        navMeshSnapRadius = GlobalScale.NAVMESH_SNAP_RADIUS;
    }

    void Start()
    {
        if (progressiveSpawnEnabled)
        {
            StartCoroutine(ProgressiveSpawnRoutine());
        }
        else
        {
            InitializeVessels();
        }
    }

    /// <summary>
    /// 점진적 스폰 (세계지도 시뮬레이션 전용)
    /// 초기 initialSpawnCount → 간격마다 spawnBatchSize씩 → 최대 maxVesselCount
    /// 각 배치는 프레임당 spawnPerFrame개씩 분산하여 hitch 방지
    /// </summary>
    private System.Collections.IEnumerator ProgressiveSpawnRoutine()
    {
        // 기존 정리
        foreach (var vessel in vessels) if (vessel != null) Destroy(vessel);
        vessels.Clear();
        vesselAgents.Clear();
        usedSpawnIndices.Clear();
        vesselGoals.Clear();
        vesselSpawnIndices.Clear();

        if (spawnPoints.Count == 0)
        {
            Debug.LogWarning("[VesselManager] No spawn points configured!");
            yield break;
        }

        // 초기 배치
        vesselCount = maxVesselCount;    // 내부 상한 갱신
        yield return StartCoroutine(SpawnBatchCoroutine(initialSpawnCount));

        // 배치 단위 추가
        while (vessels.Count < maxVesselCount)
        {
            yield return new WaitForSeconds(spawnIntervalSec);
            int toSpawn = Mathf.Min(spawnBatchSize, maxVesselCount - vessels.Count);
            if (toSpawn <= 0) break;
            yield return StartCoroutine(SpawnBatchCoroutine(toSpawn));
        }
    }

    /// <summary>
    /// 지정 개수 배를 프레임당 spawnPerFrame 개씩 분산 스폰 (hitch 방지)
    /// </summary>
    private System.Collections.IEnumerator SpawnBatchCoroutine(int count)
    {
        int spawnedThisFrame = 0;
        for (int i = 0; i < count; i++)
        {
            if (vessels.Count >= maxVesselCount) yield break;
            SpawnVessel();
            spawnedThisFrame++;
            if (spawnedThisFrame >= spawnPerFrame)
            {
                spawnedThisFrame = 0;
                yield return null;   // 다음 프레임으로
            }
        }
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

        if (spawnPoints.Count == 0)
        {
            Debug.LogWarning("[VesselManager] No spawn points configured!");
            return;
        }

        bool zoneMode = AnySpawnPointIsZone();

        if (!zoneMode && spawnPoints.Count < vesselCount)
        {
            Debug.LogWarning($"[VesselManager] Not enough spawn points! Need {vesselCount}, have {spawnPoints.Count}. " +
                             $"Hint: add SpawnZone components to spawn points to allow multi-vessel per point.");
            return;
        }

        if (goalPoints.Count == 0)
        {
            Debug.LogWarning("[VesselManager] No goal points set! Will use spawn points as goals (not recommended)");
        }

        for (int i = 0; i < vesselCount; i++)
        {
            SpawnVessel();
        }
    }

    private GameObject SpawnVessel()
    {
        // SpawnZone 컴포넌트가 붙어있는 spawnPoint가 하나라도 있으면 zone 모드
        int spawnIndex = AnySpawnPointIsZone()
            ? Random.Range(0, spawnPoints.Count)   // zone 모드: 점 재사용 허용
            : GetRandomUnusedSpawnIndex();          // 점 모드: 1점당 1척

        if (spawnIndex == -1) return null;

        Transform spawnPoint = spawnPoints[spawnIndex];
        SpawnZone zone = spawnPoint.GetComponent<SpawnZone>();

        Vector3 spawnPos;
        if (zone != null)
        {
            // Zone 모드: 영역 안 random point + NavMesh 검증 + 다른 vessel 거리 체크
            // 재사용 버퍼로 GC spike 방지 (1000대 × 1000 other = 매 스폰 1000 Vector3 alloc 제거)
            _existingPosBuffer.Clear();
            for (int i = 0; i < vessels.Count; i++)
            {
                if (vessels[i] != null) _existingPosBuffer.Add(vessels[i].transform.position);
            }

            if (!zone.TryGetRandomPoint(out spawnPos, minSpawnDistance, _existingPosBuffer))
            {
                // 혼잡 시 string 보간 GC 회피 위해 간소화
                Debug.LogWarning("[VesselManager] Zone spawn failed after max attempts. Skipping vessel.");
                return null;
            }
        }
        else
        {
            // 점 모드: NavMesh 있으면 스냅, 없으면 원래 위치 사용
            if (useNavMeshPathfinding && TrySnapToNavMesh(spawnPoint.position, out spawnPos))
            {
                // NavMesh 스냅 성공
            }
            else
            {
                spawnPos = spawnPoint.position;
            }
            usedSpawnIndices.Add(spawnIndex);
        }

        // 스냅된 위치에 생성 (rotation은 spawnPoint 그대로)
        GameObject vessel = Instantiate(vesselPrefab, spawnPos, spawnPoint.rotation, vesselParent.transform);
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

        // 재사용 버퍼 (per-spawn List alloc 제거)
        _availableIndicesBuffer.Clear();
        for (int i = 0; i < spawnPoints.Count; i++)
        {
            if (!usedSpawnIndices.Contains(i)) _availableIndicesBuffer.Add(i);
        }
        return _availableIndicesBuffer[Random.Range(0, _availableIndicesBuffer.Count)];
    }

    /// <summary>
    /// 모든 스폰 포인트가 사용 중일 때, 다른 활성 선박과 가장 먼 스폰 포인트를 반환
    /// </summary>
    private int GetSafestSpawnIndex(GameObject excludeVessel)
    {
        int bestIndex = 0;
        float bestMinDist = -1f;

        for (int i = 0; i < spawnPoints.Count; i++)
        {
            Vector3 candidatePos = spawnPoints[i].position;
            float minDistToOther = float.MaxValue;

            // 다른 모든 활성 선박과의 최소 거리 계산
            foreach (var vessel in vessels)
            {
                if (vessel == null || vessel == excludeVessel) continue;
                float dist = Vector3.Distance(candidatePos, vessel.transform.position);
                if (dist < minDistToOther) minDistToOther = dist;
            }

            // 가장 가까운 선박과의 거리가 가장 큰 스폰 포인트 선택
            if (minDistToOther > bestMinDist)
            {
                bestMinDist = minDistToOther;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    private void AssignGoalForVessel(GameObject vessel, Transform spawnPoint)
    {
        // goalPoints가 설정되어 있으면 사용, 없으면 spawnPoints 사용 (호환성)
        List<Transform> availableGoals = goalPoints.Count > 0 ? goalPoints : spawnPoints;

        // 재사용 버퍼 (1000대 스폰 × 매번 List alloc 제거)
        _validGoalBuffer.Clear();

        // 스폰 위치에서 충분히 먼 목표점들만 선택 (자기 자신 제외)
        for (int i = 0; i < availableGoals.Count; i++)
        {
            Transform point = availableGoals[i];
            if (point == spawnPoint) continue;
            float distance = Vector3.Distance(spawnPoint.position, point.position);
            if (distance >= minGoalDistance) _validGoalBuffer.Add(point);
        }

        // 거리 조건을 만족하는 목표점이 없으면 자기 자신 제외한 모든 목표점 사용
        if (_validGoalBuffer.Count == 0)
        {
            for (int i = 0; i < availableGoals.Count; i++)
            {
                Transform point = availableGoals[i];
                if (point != spawnPoint) _validGoalBuffer.Add(point);
            }
            if (_validGoalBuffer.Count == 0) _validGoalBuffer.Add(availableGoals[0]);
        }

        // 랜덤하게 목표점 선택
        int goalIndex = Random.Range(0, _validGoalBuffer.Count);
        Transform goalPoint = _validGoalBuffer[goalIndex];

        vesselGoals[vessel] = goalPoint;

        VesselAgent agent = vessel.GetComponent<VesselAgent>();
        if (agent != null)
        {
            agent.goalPointName = goalPoint.name;
            agent.goalPointIndex = goalIndex;

            // goal 좌표 결정 — zone이면 영역 안 random, 아니면 NavMesh 스냅
            Vector3 goalPos;
            SpawnZone goalZone = goalPoint.GetComponent<SpawnZone>();
            if (goalZone != null)
            {
                if (!goalZone.TryGetRandomGoalPoint(out goalPos))
                {
                    Debug.LogWarning($"[VesselManager] Goal zone '{goalZone.name}' couldn't find valid point. " +
                                     $"Using zone center.");
                    goalPos = goalPoint.position;
                }
            }
            else if (useNavMeshPathfinding && TrySnapToNavMesh(goalPoint.position, out goalPos))
            {
                // NavMesh 스냅 성공
            }
            else
            {
                goalPos = goalPoint.position;
            }

            if (useNavMeshPathfinding && pathFinder != null)
            {
                // NavMesh 경로 계산 → 웨이포인트 순차 목표
                List<Vector3> waypoints = pathFinder.CalculatePath(vessel.transform.position, goalPos);
                agent.SetWaypoints(waypoints);
            }
            else
            {
                // 폴백: 단일 목표 (기존 방식)
                agent.SetGoal(goalPos);
            }
        }
    }

    private void RotateVesselTowardsGoal(GameObject vessel)
    {
        if (!vesselGoals.ContainsKey(vessel)) return;

        // 최종 destination(goal Cube/zone 중심)을 향해 회전
        // 첫 waypoint(우회 corner)가 아닌 직선 방향으로 — 양방향 traffic 시각 일관성
        Vector3 targetPos = vesselGoals[vessel].position;
        Vector3 direction = (targetPos - vessel.transform.position).normalized;

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
        // Gizmos 비활성화
    }

    public void ChangeVesselCount(int newCount)
    {
        vesselCount = Mathf.Clamp(newCount, 1, spawnPoints.Count);
        InitializeVessels();
    }

    public void RespawnVessel(GameObject vessel)
    {
        if (vessel == null || !vessels.Contains(vessel)) return;

        bool zoneMode = AnySpawnPointIsZone();

        if (vesselSpawnIndices.ContainsKey(vessel) && !zoneMode)
        {
            int oldIndex = vesselSpawnIndices[vessel];
            usedSpawnIndices.Remove(oldIndex);
        }

        if (vesselGoals.ContainsKey(vessel))
        {
            vesselGoals.Remove(vessel);
        }

        int spawnIndex;
        if (zoneMode)
        {
            // Zone 모드: 점 재사용 허용 → 그냥 random
            spawnIndex = Random.Range(0, spawnPoints.Count);
        }
        else
        {
            spawnIndex = GetRandomUnusedSpawnIndex();
            if (spawnIndex == -1)
            {
                // 모든 스폰 포인트 사용 중 → 가장 안전한 스폰 포인트 선택
                spawnIndex = GetSafestSpawnIndex(vessel);
            }
        }

        Transform spawnPoint = spawnPoints[spawnIndex];
        SpawnZone zone = spawnPoint.GetComponent<SpawnZone>();

        Vector3 spawnPos;
        if (zone != null)
        {
            // Zone 모드: 재사용 버퍼 (1000대 respawn 시 GC spike 제거)
            _existingPosBuffer.Clear();
            for (int i = 0; i < vessels.Count; i++)
            {
                var v = vessels[i];
                if (v != null && v != vessel) _existingPosBuffer.Add(v.transform.position);
            }

            if (!zone.TryGetRandomPoint(out spawnPos, minSpawnDistance, _existingPosBuffer))
            {
                Debug.LogWarning("[VesselManager] Respawn zone couldn't find valid point. Using zone center.");
                spawnPos = spawnPoint.position;
            }
        }
        else
        {
            // 점 모드: NavMesh 있으면 스냅, 없으면 원래 위치 사용
            if (useNavMeshPathfinding && TrySnapToNavMesh(spawnPoint.position, out spawnPos))
            {
                // NavMesh 스냅 성공
            }
            else
            {
                spawnPos = spawnPoint.position;
            }
            usedSpawnIndices.Add(spawnIndex);
        }

        vesselSpawnIndices[vessel] = spawnIndex;

        // 점 모드일 때만 다른 선박과 거리 체크 (zone 모드는 이미 zone 내부에서 처리)
        if (zone == null && IsTooCloseToOtherVessels(spawnPos, vessel))
        {
            spawnPos = AddSafeOffset(spawnPos, vessel);
            if (TrySnapToNavMesh(spawnPos, out Vector3 reSnapped))
            {
                spawnPos = reSnapped;
            }
        }

        vessel.transform.position = spawnPos;
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

    /// <summary>
    /// spawnPoints 중 하나라도 SpawnZone 컴포넌트를 가지고 있는지 확인.
    /// 하나라도 있으면 zone 모드로 동작 (점 재사용 허용 → vesselCount > spawnPoints.Count 가능).
    /// </summary>
    private bool AnySpawnPointIsZone()
    {
        foreach (var point in spawnPoints)
        {
            if (point != null && point.GetComponent<SpawnZone>() != null) return true;
        }
        return false;
    }

    /// <summary>
    /// 주어진 좌표를 가장 가까운 NavMesh 위로 스냅. 검색 반경 안에 NavMesh가 없으면 false.
    /// Cube가 land 안쪽에 박혀 있거나 Y값이 NavMesh와 어긋나도 자동 보정됨.
    /// </summary>
    private bool TrySnapToNavMesh(Vector3 worldPos, out Vector3 snapped)
    {
        if (NavMesh.SamplePosition(worldPos, out NavMeshHit hit, navMeshSnapRadius, NavMesh.AllAreas))
        {
            snapped = hit.position;
            return true;
        }
        snapped = worldPos;
        return false;
    }

    /// <summary>
    /// 지정 위치가 다른 활성 선박과 너무 가까운지 확인
    /// </summary>
    private bool IsTooCloseToOtherVessels(Vector3 position, GameObject excludeVessel)
    {
        foreach (var vessel in vessels)
        {
            if (vessel == null || vessel == excludeVessel) continue;
            if (Vector3.Distance(position, vessel.transform.position) < minSpawnDistance)
                return true;
        }
        return false;
    }

    /// <summary>
    /// 다른 선박과 겹치지 않도록 랜덤 오프셋 추가
    /// </summary>
    private Vector3 AddSafeOffset(Vector3 basePosition, GameObject excludeVessel)
    {
        // 여러 방향으로 시도하여 안전한 위치 탐색
        for (int attempt = 0; attempt < 8; attempt++)
        {
            float angle = attempt * 45f;
            float offsetDist = minSpawnDistance + Random.Range(0f, GlobalScale.SPAWN_OFFSET_MAX);
            Vector3 offset = Quaternion.Euler(0, angle, 0) * Vector3.forward * offsetDist;
            Vector3 candidate = basePosition + offset;

            if (!IsTooCloseToOtherVessels(candidate, excludeVessel))
                return candidate;
        }

        // 모든 시도 실패 시 랜덤 오프셋 적용
        float offsetRange = GlobalScale.SPAWN_RANDOM_OFFSET;
        Vector3 randomOffset = new Vector3(Random.Range(-offsetRange, offsetRange), 0f, Random.Range(-offsetRange, offsetRange));
        return basePosition + randomOffset;
    }
}
