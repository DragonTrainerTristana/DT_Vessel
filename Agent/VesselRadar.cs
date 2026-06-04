using UnityEngine;
using System.Collections.Generic;

public class VesselRadar : MonoBehaviour
{

    public float radarRange = 6f;           // 레이더 Range (1/10 스케일, 원본 60m - VesselAgent가 override)
    public int rayCount = 360;                // 360개 ray (1도 간격)
    public float rayHeight = 0.1f;            // 레이 높이 (수면 위, 1/10 스케일)


    public bool showDebugRays = true;         // 디버그 레이 표시 여부

    [Header("레이어 설정")]
    public LayerMask detectionLayers = ~0;    // 감지할 레이어 (기본값: 모든 레이어)

    // 레이더 감지 결과 저장 (Dictionary → 고정 배열)
    private RaycastHit[] radarHits;
    private bool[] rayHitFlags;

    // GetAllRayDistances 캐시 (매 호출 할당 방지)
    private float[] cachedDistances;

    // GetSectorMinDistances 캐시 (numSectors 가변 lazy alloc)
    private float[] sectorCache;

    // 사전 계산된 local direction (Awake 시 1회, Scan 시 Quaternion.Euler 360회 제거)
    private Vector3[] localDirections;

    private HashSet<GameObject> detectedVesselSet = new HashSet<GameObject>();
    private List<GameObject> detectedVessels = new List<GameObject>();

    void Awake()
    {
        // Prefab Inspector 값 무시하고 GlobalScale로 강제 덮어쓰기 (rayHeight만 - radarRange는 VesselAgent에서 덮어씀)
        rayHeight = GlobalScale.RAY_HEIGHT;
        showDebugRays = GlobalScale.SHOW_DEBUG_RAYS;   // 성능 최적화: Editor Gizmo 렌더링 부하 제거

        radarHits = new RaycastHit[rayCount];
        rayHitFlags = new bool[rayCount];
        cachedDistances = new float[rayCount];

        // forward 기준 local direction 선계산 (0° = +Z, 시계방향)
        localDirections = new Vector3[rayCount];
        float step = 2f * Mathf.PI / rayCount;
        for (int i = 0; i < rayCount; i++)
        {
            float rad = i * step;
            localDirections[i] = new Vector3(Mathf.Sin(rad), 0f, Mathf.Cos(rad));
        }
    }

    /// <summary>
    /// 레이더 스캔 실행
    /// </summary>
    public void ScanRadar()
    {
        detectedVesselSet.Clear();
        detectedVessels.Clear();

        // 루프 외부로 invariant 끌어올림
        Quaternion shipRotation = transform.rotation;
        Vector3 rayOrigin = transform.position + Vector3.up * rayHeight;

        for (int i = 0; i < rayCount; i++)
        {
            // 사전 계산된 local dir에 rotation만 적용 (Quaternion.Euler 생성 제거)
            Vector3 direction = shipRotation * localDirections[i];

            if (Physics.Raycast(rayOrigin, direction, out RaycastHit hit, radarRange, detectionLayers))
            {
                radarHits[i] = hit;
                rayHitFlags[i] = true;

                // static Dictionary O(1) 조회 (기존 GetComponent<VesselAgent> 대체). Vessel만 detectedVessels에 추가.
                if (VesselAgent.IsVesselCollider(hit.collider) && detectedVesselSet.Add(hit.collider.gameObject))
                {
                    detectedVessels.Add(hit.collider.gameObject);
                }
            }
            else
            {
                rayHitFlags[i] = false;
            }
        }
    }

    /// <summary>
    /// 360개 ray의 거리 배열 반환 (정규화: -0.5~0.5, GitHub 방식)
    /// </summary>
    public float[] GetAllRayDistances()
    {
        for (int i = 0; i < rayCount; i++)
        {
            if (rayHitFlags[i])
            {
                // GitHub 방식 정규화: distance / radarRange - 0.5
                // 범위: -0.5 (거리 0) ~ 0.5 (radarRange)
                cachedDistances[i] = (radarHits[i].distance / radarRange) - 0.5f;
            }
            else
            {
                // 감지 안 됨 = 최대 거리
                // 1.0 / radarRange - 0.5 = 0.5
                cachedDistances[i] = 0.5f;
            }
        }

        return cachedDistances;
    }

    /// <summary>
    /// 감지된 선박 목록 반환
    /// </summary>
    public List<GameObject> GetDetectedVessels()
    {
        return detectedVessels;
    }

    /// <summary>
    /// 특정 각도의 장애물 거리 반환
    /// </summary>
    public float GetDistanceAtAngle(float angle)
    {
        int index = Mathf.RoundToInt(angle * (rayCount / 360f)) % rayCount;
        if (rayHitFlags[index])
        {
            return radarHits[index].distance;
        }
        return radarRange;
    }

    /// <summary>
    /// 전체 레이 중 최소 거리 반환 (미터 단위)
    /// </summary>
    public float GetMinDistance()
    {
        float minDist = radarRange;
        for (int i = 0; i < rayCount; i++)
        {
            if (rayHitFlags[i] && radarHits[i].distance < minDist)
                minDist = radarHits[i].distance;
        }
        return minDist;
    }

    /// <summary>
    /// 전방 ±halfAngle 범위의 최소 거리 반환 (미터 단위)
    /// </summary>
    public float GetMinFrontDistance(float halfAngle = 30f)
    {
        float minDist = radarRange;
        int halfRays = Mathf.CeilToInt(halfAngle * rayCount / 360f);

        // 우측: index 0 ~ halfRays-1
        for (int i = 0; i < halfRays; i++)
        {
            if (rayHitFlags[i] && radarHits[i].distance < minDist)
                minDist = radarHits[i].distance;
        }
        // 좌측: index (rayCount - halfRays) ~ (rayCount - 1)
        for (int i = rayCount - halfRays; i < rayCount; i++)
        {
            if (rayHitFlags[i] && radarHits[i].distance < minDist)
                minDist = radarHits[i].distance;
        }
        return minDist;
    }

    /// <summary>
    /// 360 ray를 numSectors개 섹터로 압축. 각 섹터의 최소거리(가장 가까운 장애물)를
    /// GetAllRayDistances와 동일 정규화(dist/radarRange - 0.5, 미감지 0.5)로 반환.
    /// 충돌 회피엔 섹터 내 최근접 장애물이 핵심이므로 min pooling 사용.
    /// </summary>
    public float[] GetSectorMinDistances(int numSectors)
    {
        if (sectorCache == null || sectorCache.Length != numSectors)
            sectorCache = new float[numSectors];

        int raysPerSector = rayCount / numSectors;   // 360/36 = 10

        for (int s = 0; s < numSectors; s++)
        {
            float minNorm = 0.5f;   // 미감지 = 최대거리 (정규화값 0.5)
            int start = s * raysPerSector;
            int end = (s == numSectors - 1) ? rayCount : start + raysPerSector;  // 나머지 ray는 마지막 섹터에 흡수

            for (int i = start; i < end; i++)
            {
                if (rayHitFlags[i])
                {
                    float norm = (radarHits[i].distance / radarRange) - 0.5f;
                    if (norm < minNorm) minNorm = norm;
                }
            }
            sectorCache[s] = minNorm;
        }

        return sectorCache;
    }

    // ============================================================================
    // ARPA (Automatic Radar Plotting Aid) — label-blind 접점 추적
    // raw 360 ray hit(거리/hit.point)만으로 접점을 시간 추적 → 상대속도/TCPA/DCPA 추정.
    // vessel/obstacle 구분 안 함(라벨 비노출). 정지 장애물은 추정속도≈0 → 자연히 무시됨.
    // 절대 IsVesselCollider/detectedVessels/상대 vesselDynamics를 쓰지 않음(치트/라벨 누출 금지).
    // ============================================================================
    private class Track
    {
        public Vector3 worldPos;        // 마지막 관측 접점 위치 (world)
        public Vector3 contactWorldVel; // 추정 절대 속도 (world, EMA)
        public float range, tcpa, dcpa;
        public int age, miss;
        public bool used;               // association 임시 플래그
    }
    private struct RawContact { public Vector3 worldPos; public float rangeMin; public int kRay; }

    private readonly List<Track> _tracks = new List<Track>();
    private readonly List<RawContact> _contacts = new List<RawContact>();
    private float[] _arpaObs;

    public Vector3 RayOrigin => transform.position + Vector3.up * rayHeight;
    public Vector3 RayDirection(int i) => transform.rotation * localDirections[i];

    /// <summary>에피소드 시작 시 호출: 이전 에피소드 track 잔존 속도 제거.</summary>
    public void ResetTracks() { _tracks.Clear(); }

    /// <summary>
    /// 매 decision 1회(ScanRadar 직후) 호출. 21D ARPA feature 반환.
    /// dt = sim-time 결정 간격(초). own* = 자기 상태(자기가 아는 값).
    /// </summary>
    public float[] UpdateTracks(Vector3 ownPos, Vector3 ownForward, float ownSpeed, float dt)
    {
        if (_arpaObs == null || _arpaObs.Length != GlobalScale.ARPA_OBS_SIZE)
            _arpaObs = new float[GlobalScale.ARPA_OBS_SIZE];

        BuildContacts();                 // raw hit → _contacts (label-blind)
        Associate(dt);                   // _contacts → _tracks (gating, 속도추정)
        Vector3 ownVel = ownForward.normalized * ownSpeed;
        SelectAndEmit(ownPos, ownForward, ownVel);   // tcpa/dcpa/risk → top-K 21D
        return _arpaObs;
    }

    /// <summary>인접 hit ray를 거리연속성으로 묶어 접점(contact) 생성. 라벨 비참조.</summary>
    private void BuildContacts()
    {
        _contacts.Clear();
        float gapThresh = GlobalScale.ARPA_RANGE_GAP_FRAC * radarRange;

        // 모두 miss면 종료
        bool allMiss = true;
        int start = 0;
        for (int i = 0; i < rayCount; i++)
        {
            if (!rayHitFlags[i]) { start = i; allMiss = false; break; }   // 첫 gap을 시작점으로(wraparound run 분리)
            if (rayHitFlags[i]) allMiss = false;
        }
        // 위 루프는 첫 non-hit를 못 찾으면 start=0 유지(전부 hit). allMiss 재확인:
        allMiss = true;
        for (int i = 0; i < rayCount; i++) { if (rayHitFlags[i]) { allMiss = false; break; } }
        if (allMiss) return;

        int runStart = -1, kMin = -1, count = 0;
        float minD = float.MaxValue, prevDist = 0f;

        for (int s = 0; s <= rayCount; s++)
        {
            int i = (start + s) % rayCount;
            bool end = (s == rayCount);
            bool hit = !end && rayHitFlags[i];
            float d = hit ? radarHits[i].distance : 0f;
            bool contiguous = hit && runStart >= 0 && Mathf.Abs(d - prevDist) <= gapThresh;

            if (hit && (runStart < 0 || contiguous))
            {
                if (runStart < 0) { runStart = i; kMin = i; minD = d; count = 0; }
                if (d < minD) { minD = d; kMin = i; }
                count++; prevDist = d;
            }
            else
            {
                if (runStart >= 0 && count >= GlobalScale.ARPA_MIN_RAYS) FlushRun(kMin, minD);
                if (hit) { runStart = i; kMin = i; minD = d; count = 1; prevDist = d; }
                else { runStart = -1; count = 0; }
            }
        }
    }

    private void FlushRun(int kMin, float minD)
    {
        Vector3 wp = (kMin >= 0 && IsFinite(radarHits[kMin].point))
            ? radarHits[kMin].point                         // 정밀 hull 접점(서브-도, 양자화 완화)
            : RayOrigin + RayDirection(kMin) * minD;
        _contacts.Add(new RawContact { worldPos = wp, rangeMin = minD, kRay = kMin });
    }

    /// <summary>이번 scan 접점을 직전 track에 nearest-gating 매칭 → 상대속도 EMA 추정.</summary>
    private void Associate(float dt)
    {
        float gate = Mathf.Max(2f * GlobalScale.MAX_SPEED * dt,
                               0.5f * GlobalScale.LENGTH,
                               GlobalScale.ARPA_GATE_FRAC * radarRange);
        float gate2 = gate * gate;

        for (int i = 0; i < _tracks.Count; i++) _tracks[i].used = false;

        foreach (var c in _contacts)
        {
            Track best = null; float bestD2 = gate2;
            for (int i = 0; i < _tracks.Count; i++)
            {
                var t = _tracks[i];
                if (t.used) continue;
                Vector3 pred = t.worldPos + t.contactWorldVel * dt;
                float d2 = (c.worldPos - pred).sqrMagnitude;
                if (d2 <= bestD2) { bestD2 = d2; best = t; }
            }
            if (best != null)
            {
                Vector3 vMeas = (c.worldPos - best.worldPos) / Mathf.Max(dt, 1e-3f);
                best.contactWorldVel = (best.age <= 1)
                    ? vMeas
                    : Vector3.Lerp(best.contactWorldVel, vMeas, GlobalScale.ARPA_VEL_EMA);
                best.worldPos = c.worldPos;
                best.range = c.rangeMin;
                best.age++; best.miss = 0; best.used = true;
            }
            else
            {
                _tracks.Add(new Track {
                    worldPos = c.worldPos, contactWorldVel = Vector3.zero,
                    range = c.rangeMin, age = 1, miss = 0, used = true,
                    tcpa = float.MaxValue, dcpa = c.rangeMin
                });
            }
        }

        for (int i = _tracks.Count - 1; i >= 0; i--)
        {
            if (!_tracks[i].used)
            {
                _tracks[i].miss++;
                if (_tracks[i].miss > GlobalScale.ARPA_MAX_MISS) _tracks.RemoveAt(i);
            }
        }

        if (_tracks.Count > GlobalScale.ARPA_MAX_TRACKS)
        {
            _tracks.Sort((a, b) => a.range.CompareTo(b.range));
            _tracks.RemoveRange(GlobalScale.ARPA_MAX_TRACKS, _tracks.Count - GlobalScale.ARPA_MAX_TRACKS);
        }
    }

    /// <summary>각 track의 tcpa/dcpa 갱신, risk 정렬, top-K를 21D로 emit.</summary>
    private void SelectAndEmit(Vector3 ownPos, Vector3 ownForward, Vector3 ownVel)
    {
        for (int i = 0; i < _tracks.Count; i++)
        {
            var t = _tracks[i];
            float rawTcpa = COLREGsHandler.CalculateTCPA(ownPos, ownVel, t.worldPos, t.contactWorldVel);
            float dcpa = COLREGsHandler.CalculateDCPA(ownPos, ownVel, t.worldPos, t.contactWorldVel);
            t.tcpa = (t.age <= 1) ? rawTcpa : Mathf.Lerp(t.tcpa, rawTcpa, GlobalScale.ARPA_CPA_EMA);
            t.dcpa = (t.age <= 1) ? dcpa : Mathf.Lerp(t.dcpa, dcpa, GlobalScale.ARPA_CPA_EMA);
        }

        _tracks.Sort((a, b) => Risk(b, ownPos, ownVel).CompareTo(Risk(a, ownPos, ownVel)));

        System.Array.Clear(_arpaObs, 0, _arpaObs.Length);
        int n = Mathf.Min(GlobalScale.ARPA_K, _tracks.Count);
        for (int idx = 0; idx < n; idx++)
        {
            var t = _tracks[idx];
            Vector3 toC = t.worldPos - ownPos;
            float bearing = Vector3.SignedAngle(ownForward, toC, Vector3.up) * Mathf.Deg2Rad;
            Vector3 los = toC.sqrMagnitude > 1e-6f ? toC.normalized : ownForward.normalized;
            float closing = -Vector3.Dot(los, t.contactWorldVel - ownVel);   // + = 접근 중
            int b = idx * GlobalScale.ARPA_FEATURES_PER;
            _arpaObs[b + 0] = Mathf.Sin(bearing);
            _arpaObs[b + 1] = Mathf.Cos(bearing);
            _arpaObs[b + 2] = Mathf.Clamp01(t.range / radarRange);
            _arpaObs[b + 3] = Mathf.Clamp(closing / GlobalScale.ARPA_SPEED_NORM, -1f, 1f);
            _arpaObs[b + 4] = Mathf.Clamp(t.dcpa / radarRange, 0f, 1.5f) / 1.5f;
            _arpaObs[b + 5] = Mathf.Clamp01(Mathf.Max(t.tcpa, 0f) / GlobalScale.ARPA_TCPA_CAP);
            _arpaObs[b + 6] = (t.age >= GlobalScale.ARPA_MIN_AGE) ? 1f : 0f;
        }
    }

    /// <summary>충돌 위험도(0~1). 정지/멀어지는 접점(rock 등)은 0.1배로 down → top-K에서 밀림.</summary>
    private float Risk(Track t, Vector3 ownPos, Vector3 ownVel)
    {
        float dcpaRisk = 1f - Mathf.Clamp01(t.dcpa / GlobalScale.ARPA_DCPA_DENOM);
        float tcpaRisk = 1f / (1f + Mathf.Max(t.tcpa, 0f) / GlobalScale.ARPA_TCPA_CAP);
        float rangeRisk = 1f - Mathf.Clamp01(t.range / radarRange);
        float risk = 0.5f * dcpaRisk + 0.3f * tcpaRisk + 0.2f * rangeRisk;
        Vector3 toC = t.worldPos - ownPos;
        Vector3 los = toC.sqrMagnitude > 1e-6f ? toC.normalized : Vector3.forward;
        float closing = -Vector3.Dot(los, t.contactWorldVel - ownVel);
        if (t.tcpa <= 0f || closing <= 0f) risk *= 0.1f;
        return risk;
    }

    private static bool IsFinite(Vector3 v) =>
        !(float.IsNaN(v.x) || float.IsInfinity(v.x) ||
          float.IsNaN(v.y) || float.IsInfinity(v.y) ||
          float.IsNaN(v.z) || float.IsInfinity(v.z));

#if UNITY_EDITOR
    /// <summary>
    /// 레이더 시각화: 초록 원(범위) + 빨간 선(감지)
    /// Build에서는 완전히 제거됨 (1000대 × per-frame 호출 오버헤드 차단)
    /// </summary>
    private void OnDrawGizmos()
    {
        Vector3 origin = transform.position + Vector3.up * rayHeight;

        // ── 범위 원 (가벼움: 배당 2개 wire circle) — 학습 관찰용, 기본 ON ──────
        if (GlobalScale.SHOW_RANGE_GIZMOS)
        {
            // 초록 원: 레이더 감지 범위 (radarRange = 8m)
            Gizmos.color = Color.green;
            DrawGizmoCircle(origin, radarRange, 60);

            // 반투명 시안 원: 통신 범위 (= COMM_RANGE = 140m)
            Gizmos.color = new Color(0f, 0.8f, 1f, 0.5f);
            DrawGizmoCircle(origin, GlobalScale.COMM_RANGE, 72);
        }

        // ── 360 감지선 (무거움: 배당 최대 360선) — 기본 OFF(SHOW_DEBUG_RAYS) ────
        if (!showDebugRays || !Application.isPlaying || rayHitFlags == null) return;

        // 빨간 선: 감지된 ray만 표시
        Gizmos.color = Color.red;
        for (int i = 0; i < rayCount; i++)
        {
            if (!rayHitFlags[i]) continue;
            float angle = i * (360f / rayCount);
            Vector3 direction = Quaternion.Euler(0, angle, 0) * transform.forward;
            Gizmos.DrawLine(origin, origin + direction * radarHits[i].distance);
        }
    }
#endif

    private void DrawGizmoCircle(Vector3 center, float radius, int segments)
    {
        float angleStep = 360f / segments;
        Vector3 prevPoint = center + new Vector3(radius, 0, 0);
        for (int i = 1; i <= segments; i++)
        {
            float angle = i * angleStep * Mathf.Deg2Rad;
            Vector3 nextPoint = center + new Vector3(Mathf.Cos(angle) * radius, 0, Mathf.Sin(angle) * radius);
            Gizmos.DrawLine(prevPoint, nextPoint);
            prevPoint = nextPoint;
        }
    }

}
