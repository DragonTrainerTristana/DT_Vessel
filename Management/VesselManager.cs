using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using System.Linq;
using Unity.MLAgents.Policies;

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

    [Header("Control Mode")]
    [Tooltip("true: rule-based (VesselAutoPilot, Python 불필요, A*+local COLREGs). false: RL (VesselAgent).")]
    public bool useRuleBasedMode = false;

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

    // ──────────────────────────────────────────────────────────────────────
    // ★Density regime (2026-06-21): 의도통신-필수 regime을 "밀도 상향"으로 생성.
    //   동기는 "통신 가치 = 다물체 동시조우의 액션 모호성"(3척+이 56m 안에서 한 점 수렴 →
    //   shared-policy 예측가능성·짝 COLREGs 규약이 깨짐 → 단일 정답 액션 없음 → 의도공유로 joint maneuver 합의).
    //   현 regime(ring500·n16·crossing fan-out)은 ambiguity(2+동시충돌코스) ~4.5%로 통신 잉여 진단됨.
    //   레버: ① ring 축소(중심집중) ② 척수↑ ③ 진짜 대척 goal(fan-out 제거)로 빈도를 ~40-80%까지 상향.
    //   ⚠️ per-agent 비동기 respawn 구조라 "매 에피소드 100% 보장"은 아님(확률적 상향) — 검증지표로 확인.
    //
    //   anti-rigging: 전부 env 토글, default(scale 1.0 / count 미설정 / crossing≠2)면 좌표·로직 비트동일.
    //   comm ON/OFF 동일 환경(밀도는 spawn 기하라 ON/OFF에 동일 적용) → baseline 불구화 아님.
    //   도달성: ring 축소는 이동거리를 *줄여* 도달성을 높임(MaxStep/goalDist 미증가 = 제약3 준수).
    // ──────────────────────────────────────────────────────────────────────
    private static float _spawnRingScale = float.NaN;     // VESSEL_SPAWN_RING_SCALE (기본 1.0)
    private static int   _vesselCountOverride = -2;        // VESSEL_VESSEL_COUNT (-2=미파싱, -1=미설정)
    private static int   _antipodalGoalMode = -1;          // VESSEL_CROSSING==2 → 대척 goal
    private Vector3 _spawnCentroid;                        // spawn/goal 링의 중심(축소 기준점)
    private bool _spawnCentroidCached = false;

    /// <summary>VESSEL_SPAWN_RING_SCALE: spawn/goal 좌표를 중심 기준 곱수(기본 1.0=비트동일). &lt;1이면 링 축소→밀도↑.</summary>
    private float SpawnRingScale()
    {
        if (float.IsNaN(_spawnRingScale))
        {
            string v = System.Environment.GetEnvironmentVariable("VESSEL_SPAWN_RING_SCALE");
            _spawnRingScale = (!string.IsNullOrEmpty(v) && float.TryParse(v, out float s) && s > 0f) ? s : 1.0f;
        }
        return _spawnRingScale;
    }

    /// <summary>VESSEL_VESSEL_COUNT: 스폰할 배 수 override(미설정=scene 직렬화값=비트동일). 점모드는 spawnPoints.Count 상한.</summary>
    private int ResolvedVesselCount()
    {
        if (_vesselCountOverride == -2)
        {
            string v = System.Environment.GetEnvironmentVariable("VESSEL_VESSEL_COUNT");
            _vesselCountOverride = (!string.IsNullOrEmpty(v) && int.TryParse(v, out int n) && n > 0) ? n : -1;
        }
        return _vesselCountOverride > 0 ? _vesselCountOverride : vesselCount;
    }

    /// <summary>VESSEL_CROSSING==2 → 대척(중심관통) goal 배정. fan-out(가장먼=코너) 대신 경로가 정확히 중심 교차.</summary>
    private bool UseAntipodalGoals()
    {
        if (_antipodalGoalMode < 0)
            _antipodalGoalMode = (System.Environment.GetEnvironmentVariable("VESSEL_CROSSING") == "2") ? 1 : 0;
        return _antipodalGoalMode == 1;
    }

    /// <summary>spawn 링의 중심(모든 spawnPoint 평균). 1회 캐시. ring scale의 축소 기준점.</summary>
    private Vector3 SpawnCentroid()
    {
        if (!_spawnCentroidCached)
        {
            Vector3 sum = Vector3.zero;
            int cnt = 0;
            foreach (var p in spawnPoints) { if (p != null) { sum += p.position; cnt++; } }
            _spawnCentroid = (cnt > 0) ? sum / cnt : Vector3.zero;
            _spawnCentroidCached = true;
        }
        return _spawnCentroid;
    }

    /// <summary>좌표를 중심 기준으로 ring scale 적용. scale==1.0이면 입력 그대로(비트동일).</summary>
    private Vector3 ApplyRingScale(Vector3 worldPos)
    {
        float s = SpawnRingScale();
        if (s == 1.0f) return worldPos;
        Vector3 c = SpawnCentroid();
        return c + (worldPos - c) * s;
    }

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

        // 배치 완료 후 autoPilot 상호 인식 갱신
        if (useRuleBasedMode) InjectAutoPilotList();
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
        int spawnTarget = ResolvedVesselCount();   // VESSEL_VESSEL_COUNT override (기본=scene값=비트동일)

        if (!zoneMode && spawnPoints.Count < spawnTarget)
        {
            // ★빈 환경 방지: 과거엔 여기서 return → headless 학습이 배 0척으로 조용히 진행되는 무효 실험 벡터.
            //   스폰 가능한 수로 클램프하고 경고만 남긴다 (VESSEL_VESSEL_COUNT docstring의 '점모드 상한'과 일치).
            Debug.LogWarning($"[VesselManager] Not enough spawn points! Need {spawnTarget}, have {spawnPoints.Count} — clamping to available. " +
                             $"Hint: add SpawnZone components to spawn points to allow multi-vessel per point.");
            spawnTarget = spawnPoints.Count;
        }

        if (goalPoints.Count == 0)
        {
            Debug.LogWarning("[VesselManager] No goal points set! Will use spawn points as goals (not recommended)");
        }

        for (int i = 0; i < spawnTarget; i++)
        {
            SpawnVessel();
        }

        // Rule-based 모드: 모든 autoPilot끼리 서로 인식하도록 리스트 주입
        if (useRuleBasedMode) InjectAutoPilotList();
    }

    /// <summary>
    /// 모든 VesselAutoPilot에 전체 autoPilot 리스트 전달 (서로 감지용).
    /// Progressive Spawn에도 배치 추가마다 재호출.
    /// </summary>
    private void InjectAutoPilotList()
    {
        var autoPilots = new List<VesselAutoPilot>();
        foreach (var v in vessels)
        {
            if (v == null) continue;
            var ap = v.GetComponent<VesselAutoPilot>();
            if (ap != null && ap.enabled) autoPilots.Add(ap);
        }
        foreach (var ap in autoPilots) ap.Initialize(autoPilots);
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
            // 점 모드: ring scale 적용(기본 1.0=원좌표) → NavMesh 스냅
            Vector3 basePos = ApplyRingScale(spawnPoint.position);
            if (useNavMeshPathfinding && TrySnapToNavMesh(basePos, out spawnPos))
            {
                // NavMesh 스냅 성공
            }
            else
            {
                spawnPos = basePos;
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
        VesselAutoPilot autoPilot = vessel.GetComponent<VesselAutoPilot>();

        if (useRuleBasedMode)
        {
            // Rule-based 모드: VesselAgent 비활성, VesselAutoPilot 활성
            if (agent != null) agent.enabled = false;

            // ML-Agents Python trainer 연결 시도 차단 (5초 timeout 제거, "Registered Communicator" 로그 방지)
            var bp = vessel.GetComponent<BehaviorParameters>();
            if (bp != null) bp.BehaviorType = BehaviorType.HeuristicOnly;

            if (autoPilot == null) autoPilot = vessel.AddComponent<VesselAutoPilot>();
            autoPilot.enabled = true;
            autoPilot.EnsureInitialized();
        }
        else
        {
            // RL 모드: VesselAgent 활성, VesselAutoPilot 비활성
            if (agent != null) agent.enabled = true;
            if (autoPilot != null) autoPilot.enabled = false;
            if (agent != null) vesselAgents.Add(agent);
        }

        vesselSpawnIndices[vessel] = spawnIndex;

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

    private static int _crossingGoalMode = -1;
    /// <summary>VESSEL_CROSSING=1 → 스폰에서 "가장 먼" 목표 배정. 경로가 중심에서 교차(4-way) → Type B 협응충돌 유도 → 통신 검증용. 기본 OFF.</summary>
    private bool UseCrossingGoals()
    {
        if (_crossingGoalMode < 0)
            _crossingGoalMode = (System.Environment.GetEnvironmentVariable("VESSEL_CROSSING") == "1") ? 1 : 0;
        return _crossingGoalMode == 1;
    }

    private void AssignGoalForVessel(GameObject vessel, Transform spawnPoint)
    {
        // goalPoints가 설정되어 있으면 사용, 없으면 spawnPoints 사용 (호환성)
        List<Transform> availableGoals = goalPoints.Count > 0 ? goalPoints : spawnPoints;

        Transform goalPoint = null;
        int goalIndex = 0;

        // ── 대척 교차 모드(VESSEL_CROSSING=2): spawn의 중심대칭점에 가장 가까운 goal 배정 ──
        //   경로가 정확히 중심(centroid) 관통 → fan-out(코너 산개, centerMiss 평균 ~45m) 제거 →
        //   같은 ring·n에서 다물체 한점수렴 빈도 대폭↑. ring scale과 결합 시 시너지(density 검증).
        if (UseAntipodalGoals())
        {
            Vector3 c = SpawnCentroid();
            Vector3 antipode = c + (c - spawnPoint.position);   // spawn의 중심대칭점(반대편)
            float minD = float.MaxValue;
            string sName = spawnPoint.name;
            for (int i = 0; i < availableGoals.Count; i++)
            {
                Transform p = availableGoals[i];
                if (p == spawnPoint || p.name == sName) continue;
                float d = Vector3.Distance(antipode, p.position);
                if (d < minD) { minD = d; goalPoint = p; goalIndex = i; }   // 대척점에 가장 가까운 goal
            }
        }

        // ── 4-way 교차 모드(VESSEL_CROSSING=1): 스폰에서 거리가 가장 먼 목표 배정 → 경로가 중심 교차 ──
        // (초기/respawn 모두 이 함수를 거치므로 매 에피소드 적용됨)
        if (goalPoint == null && UseCrossingGoals())
        {
            float maxD = -1f;
            string sName = spawnPoint.name;
            for (int i = 0; i < availableGoals.Count; i++)
            {
                Transform p = availableGoals[i];
                if (p == spawnPoint || p.name == sName) continue;
                float d = Vector3.Distance(spawnPoint.position, p.position);
                if (d > maxD) { maxD = d; goalPoint = p; goalIndex = i; }   // 가장 먼 = 반대편
            }
        }

        // 인덱스 매칭: spawnPoints[i] → goalPoints[i] (Inspector에서 크로스 페어링)
        if (goalPoint == null && goalPoints.Count > 0 && goalPoints.Count == spawnPoints.Count)
        {
            int spawnIdx = spawnPoints.IndexOf(spawnPoint);
            if (spawnIdx >= 0)
            {
                goalPoint = goalPoints[spawnIdx];
                goalIndex = spawnIdx;
            }
        }

        // 인덱스 매칭 실패 시 기존 랜덤 로직 폴백
        if (goalPoint == null)
        {
            _validGoalBuffer.Clear();
            string spawnName = spawnPoint.name;
            for (int i = 0; i < availableGoals.Count; i++)
            {
                Transform point = availableGoals[i];
                if (point == spawnPoint || point.name == spawnName) continue;
                float distance = Vector3.Distance(spawnPoint.position, point.position);
                if (distance >= minGoalDistance) _validGoalBuffer.Add(point);
            }

            if (_validGoalBuffer.Count == 0)
            {
                for (int i = 0; i < availableGoals.Count; i++)
                {
                    Transform point = availableGoals[i];
                    if (point != spawnPoint && point.name != spawnName) _validGoalBuffer.Add(point);
                }
                if (_validGoalBuffer.Count == 0) _validGoalBuffer.Add(availableGoals[0]);
            }

            goalIndex = Random.Range(0, _validGoalBuffer.Count);
            goalPoint = _validGoalBuffer[goalIndex];
        }

        vesselGoals[vessel] = goalPoint;

        // goal 좌표 결정 (공통)
        Vector3 goalPos;
        SpawnZone goalZone = goalPoint.GetComponent<SpawnZone>();
        if (goalZone != null)
        {
            if (!goalZone.TryGetRandomGoalPoint(out goalPos))
            {
                Debug.LogWarning("[VesselManager] Goal zone couldn't find valid point. Using zone center.");
                goalPos = goalPoint.position;
            }
        }
        else
        {
            // goal도 spawn과 동일 ring scale 적용(중심 동심 유지) → NavMesh 스냅. 기본 1.0=원좌표(비트동일).
            Vector3 goalBase = ApplyRingScale(goalPoint.position);
            if (useNavMeshPathfinding && TrySnapToNavMesh(goalBase, out goalPos))
            {
                // NavMesh 스냅 성공
            }
            else
            {
                goalPos = goalBase;
            }
        }

        // 경로 계산 (공통)
        List<Vector3> waypoints = null;
        if (useNavMeshPathfinding && pathFinder != null)
        {
            waypoints = pathFinder.CalculatePath(vessel.transform.position, goalPos);
        }

        if (useRuleBasedMode)
        {
            // Rule-based: VesselAutoPilot에 주입
            VesselAutoPilot autoPilot = vessel.GetComponent<VesselAutoPilot>();
            if (autoPilot != null)
            {
                if (waypoints != null && waypoints.Count > 0) autoPilot.SetWaypoints(waypoints);
                else autoPilot.SetGoal(goalPos);
            }
        }
        else
        {
            // RL: VesselAgent에 주입
            VesselAgent agent = vessel.GetComponent<VesselAgent>();
            if (agent != null)
            {
                if (waypoints != null && waypoints.Count > 0) agent.SetWaypoints(waypoints);
                else agent.SetGoal(goalPos);
            }
        }
    }

    private void RotateVesselTowardsGoal(GameObject vessel)
    {
        // 최종 목적지 방향으로 회전 (waypoint가 아닌 최종 destination)
        if (!vesselGoals.ContainsKey(vessel)) return;

        // ★ring scale 정합: 에이전트가 실제 항해하는 goalPosition(스케일 적용된 좌표)을 기준으로 회전.
        //   vesselGoals[vessel].position(원 Transform)을 쓰면 scale<1.0일 때 초기 침로가 어긋남(homothety로 방향은 보존되나
        //   원점=spawn이 scaled라 벡터가 다름). AssignGoalForVessel→SetGoal이 이미 scaled goal을 주입했으므로 그걸 사용.
        Vector3 targetPos = vesselGoals[vessel].position;
        VesselAgent ag = vessel.GetComponent<VesselAgent>();
        if (ag != null && ag.hasGoal) targetPos = ag.goalPosition;

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
            // 점 모드: ring scale 적용(기본 1.0=원좌표) → NavMesh 스냅
            Vector3 basePos = ApplyRingScale(spawnPoint.position);
            if (useNavMeshPathfinding && TrySnapToNavMesh(basePos, out spawnPos))
            {
                // NavMesh 스냅 성공
            }
            else
            {
                spawnPos = basePos;
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
