using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class WaypointPathFinder : MonoBehaviour
{
    [Header("Path Settings")]
    public float navMeshSampleRadius = 0.5f;   // NavMesh 스냅 반경 (1/10 스케일, 원본 5m)
    public float minWaypointDistance = 1.5f;   // 웨이포인트 최소 간격 (1/10 스케일, 원본 15m)
    public float pathHeightY = 0f;             // Y 고정값 (수면 높이)

    void Awake()
    {
        // Prefab Inspector 값 무시하고 GlobalScale로 강제 덮어쓰기
        navMeshSampleRadius = GlobalScale.WAYPOINT_SAMPLE;
        minWaypointDistance = GlobalScale.MIN_WAYPOINT_DIST;
    }

    /// <summary>
    /// NavMesh 기반 경로 계산 후 웨이포인트 리스트 반환.
    /// 시작점은 제외, 마지막 점(목표)은 항상 포함.
    /// 실패 시 {end} 단일 리스트 반환 (폴백).
    /// </summary>
    public List<Vector3> CalculatePath(Vector3 start, Vector3 end)
    {
        // NavMesh 위로 스냅
        NavMeshHit startHit, endHit;
        if (!NavMesh.SamplePosition(start, out startHit, navMeshSampleRadius, NavMesh.AllAreas))
        {
            // 스냅 실패 → 폴백
            return new List<Vector3> { FlattenY(end) };
        }
        if (!NavMesh.SamplePosition(end, out endHit, navMeshSampleRadius, NavMesh.AllAreas))
        {
            return new List<Vector3> { FlattenY(end) };
        }

        // 경로 계산
        NavMeshPath path = new NavMeshPath();
        if (!NavMesh.CalculatePath(startHit.position, endHit.position, NavMesh.AllAreas, path))
        {
            return new List<Vector3> { FlattenY(end) };
        }

        // partial path도 사용 가능 (완전 실패만 폴백)
        if (path.status == NavMeshPathStatus.PathInvalid)
        {
            return new List<Vector3> { FlattenY(end) };
        }

        // corners 추출 → Y 고정 → 첫 점(시작 위치) 제거
        List<Vector3> waypoints = new List<Vector3>();
        for (int i = 1; i < path.corners.Length; i++) // i=1: 시작점 제거
        {
            waypoints.Add(FlattenY(path.corners[i]));
        }

        if (waypoints.Count == 0)
        {
            return new List<Vector3> { FlattenY(end) };
        }

        // 가까운 웨이포인트 병합
        waypoints = SimplifyPath(waypoints);

        // 마지막 점이 실제 목표와 다를 수 있으므로 교체
        waypoints[waypoints.Count - 1] = FlattenY(end);

        return waypoints;
    }

    /// <summary>
    /// minWaypointDistance 미만 간격인 중간 점들을 병합.
    /// 마지막 점(목표)은 항상 유지.
    /// </summary>
    private List<Vector3> SimplifyPath(List<Vector3> points)
    {
        if (points.Count <= 1) return points;

        List<Vector3> simplified = new List<Vector3>();
        simplified.Add(points[0]);

        for (int i = 1; i < points.Count - 1; i++)
        {
            float dist = Vector3.Distance(simplified[simplified.Count - 1], points[i]);
            if (dist >= minWaypointDistance)
            {
                simplified.Add(points[i]);
            }
        }

        // 마지막 점은 항상 포함
        simplified.Add(points[points.Count - 1]);

        return simplified;
    }

    private Vector3 FlattenY(Vector3 pos)
    {
        return new Vector3(pos.x, pathHeightY, pos.z);
    }
}
