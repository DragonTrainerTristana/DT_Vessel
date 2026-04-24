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

#if UNITY_EDITOR
    /// <summary>
    /// 레이더 시각화: 초록 원(범위) + 빨간 선(감지)
    /// Build에서는 완전히 제거됨 (1000대 × per-frame 호출 오버헤드 차단)
    /// </summary>
    private void OnDrawGizmos()
    {
        if (!showDebugRays) return;

        Vector3 origin = transform.position + Vector3.up * rayHeight;

        // 초록 원: 레이더 범위 (항상 표시)
        Gizmos.color = Color.green;
        DrawGizmoCircle(origin, radarRange, 60);

        if (!Application.isPlaying || rayHitFlags == null) return;

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
