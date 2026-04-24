using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

/// <summary>
/// 점이 아닌 영역으로 vessel을 스폰. zone 안 random point + NavMesh 검증.
/// 부산/대만 Cube 같은 GameObject에 붙여서 사용. VesselManager가 자동 인식.
/// </summary>
public class SpawnZone : MonoBehaviour
{
    [Header("Zone Settings")]
    [Tooltip("zone 반경 (이 영역 안에서 random spawn)")]
    public float radius = 30f;    // 1/10 스케일 (원본 300m)

    [Tooltip("NavMesh 위로 스냅할 때 검색 반경 (radius보다 작게)")]
    public float navMeshSampleRadius = 5f;    // 1/10 스케일 (원본 50m)

    [Tooltip("random point 시도 횟수")]
    public int maxAttempts = 50;

    [Header("Visualization")]
    public Color gizmoColor = Color.green;

    void Awake()
    {
        // Prefab Inspector 값 무시하고 GlobalScale로 강제 덮어쓰기
        radius = GlobalScale.SPAWNZONE_RADIUS;
        navMeshSampleRadius = GlobalScale.SPAWNZONE_SAMPLE;
    }

    /// <summary>
    /// zone 안에서 NavMesh 검증된 random point 반환.
    /// existingPoints와 minDistance 만족하는 위치만 사용.
    /// </summary>
    public bool TryGetRandomPoint(out Vector3 point, float minDistanceFromOthers,
                                  List<Vector3> existingPoints)
    {
        for (int i = 0; i < maxAttempts; i++)
        {
            // zone 안 random 2D point
            Vector2 disk = Random.insideUnitCircle * radius;
            Vector3 candidate = transform.position + new Vector3(disk.x, 0f, disk.y);

            // NavMesh 위로 스냅 — land 위면 실패
            if (!NavMesh.SamplePosition(candidate, out NavMeshHit hit, navMeshSampleRadius, NavMesh.AllAreas))
                continue;

            // 기존 vessel과 거리 체크
            bool tooClose = false;
            if (existingPoints != null)
            {
                foreach (var existing in existingPoints)
                {
                    if (Vector3.Distance(hit.position, existing) < minDistanceFromOthers)
                    {
                        tooClose = true;
                        break;
                    }
                }
            }
            if (tooClose) continue;

            point = hit.position;
            return true;
        }

        // 시도 모두 실패 → zone 중심으로 폴백
        point = transform.position;
        return false;
    }

    /// <summary>
    /// goal용 — 거리 체크 없이 단순 random point.
    /// </summary>
    public bool TryGetRandomGoalPoint(out Vector3 point)
    {
        return TryGetRandomPoint(out point, 0f, null);
    }

    void OnDrawGizmos()
    {
        Gizmos.color = gizmoColor;
        const int segments = 48;
        Vector3 center = transform.position;
        for (int i = 0; i < segments; i++)
        {
            float a1 = (i / (float)segments) * Mathf.PI * 2f;
            float a2 = ((i + 1) / (float)segments) * Mathf.PI * 2f;
            Vector3 p1 = center + new Vector3(Mathf.Cos(a1), 0f, Mathf.Sin(a1)) * radius;
            Vector3 p2 = center + new Vector3(Mathf.Cos(a2), 0f, Mathf.Sin(a2)) * radius;
            Gizmos.DrawLine(p1, p2);
        }
        // 중심 표시
        Gizmos.DrawSphere(center, radius * 0.02f);
    }
}
