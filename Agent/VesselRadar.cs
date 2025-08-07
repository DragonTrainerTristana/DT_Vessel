using UnityEngine;
using System.Collections.Generic;

public class VesselRadar : MonoBehaviour
{
    [Header("레이더 설정")]
    public float radarRange = 100f;           // 레이더 감지 거리
    public int rayCount = 360;                // 레이 개수 (1도 간격)
    public float rayHeight = 1f;              // 레이 높이 (수면 위)
    public bool showDebugRays = true;         // 디버그 레이 표시 여부

    [Header("레이어 설정")]
    public LayerMask detectionLayers;         // 감지할 레이어

    // 레이더 감지 결과 저장
    private Dictionary<int, RaycastHit> radarHits = new Dictionary<int, RaycastHit>();
    private List<GameObject> detectedVessels = new List<GameObject>();

    /// <summary>
    /// 레이더 스캔 실행
    /// </summary>
    public void ScanRadar()
    {
        radarHits.Clear();
        detectedVessels.Clear();

        for (int i = 0; i < rayCount; i++)
        {
            float angle = i * (360f / rayCount);
            Vector3 direction = Quaternion.Euler(0, angle, 0) * transform.forward;

            Ray ray = new Ray(transform.position + Vector3.up * rayHeight, direction);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, radarRange, detectionLayers))
            {
                radarHits[i] = hit;

                // 선박 감지 시 목록에 추가
                if (hit.collider.CompareTag("Vessel") && !detectedVessels.Contains(hit.collider.gameObject))
                {
                    detectedVessels.Add(hit.collider.gameObject);
                }
            }
        }
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
        if (radarHits.ContainsKey(index))
        {
            return radarHits[index].distance;
        }
        return radarRange;
    }

    /// <summary>
    /// 레이더 시각화
    /// </summary>
    private void OnDrawGizmos()
    {
        if (!showDebugRays || !Application.isPlaying)
            return;

        // 레이더 범위 원 표시
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireSphere(transform.position, radarRange);

        // 감지된 레이 표시
        foreach (var hit in radarHits)
        {
            float angle = hit.Key * (360f / rayCount);
            Vector3 direction = Quaternion.Euler(0, angle, 0) * transform.forward;

            // 감지된 물체의 종류에 따라 색상 변경
            if (hit.Value.collider.CompareTag("Vessel"))
            {
                Gizmos.color = Color.red;  // 다른 선박
            }
            else if (hit.Value.collider.CompareTag("Obstacle"))
            {
                Gizmos.color = Color.yellow;  // 장애물
            }
            else
            {
                Gizmos.color = Color.green;  // 기타 물체
            }

            // 레이 그리기
            Gizmos.DrawLine(transform.position + Vector3.up * rayHeight,
                          transform.position + Vector3.up * rayHeight + direction * hit.Value.distance);

            // 감지 지점 표시
            Gizmos.DrawSphere(hit.Value.point, 0.5f);
        }
    }

    /// <summary>
    /// 에디터에서 레이더 범위 시각화
    /// </summary>
    private void OnDrawGizmosSelected()
    {
        if (Application.isPlaying)
            return;

        // 레이더 최대 범위 표시
        Gizmos.color = new Color(0, 1, 1, 0.2f);
        Gizmos.DrawWireSphere(transform.position, radarRange);
    }
}