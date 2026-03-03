using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class VesselRadar : MonoBehaviour
{

    public float radarRange = 60f;           // 레이더 Range
    public int rayCount = 360;                // 360개 ray (1도 간격)
    public float rayHeight = 1f;              // 레이 높이 (수면 위)


    public bool showDebugRays = true;         // 디버그 레이 표시 여부

    [Header("레이어 설정")]
    public LayerMask detectionLayers = ~0;    // 감지할 레이어 (기본값: 모든 레이어)

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

                // 선박 감지 시 목록에 추가 (VesselAgent 컴포넌트로 확인)
                if (hit.collider.GetComponent<VesselAgent>() != null && !detectedVessels.Contains(hit.collider.gameObject))
                {
                    detectedVessels.Add(hit.collider.gameObject);
                }
            }
        }
    }

    /// <summary>
    /// 360개 ray의 거리 배열 반환 (정규화: -0.5~0.5, GitHub 방식)
    /// </summary>
    public float[] GetAllRayDistances()
    {
        float[] distances = new float[rayCount];

        for (int i = 0; i < rayCount; i++)
        {
            if (radarHits.ContainsKey(i))
            {
                // GitHub 방식 정규화: distance / radarRange - 0.5
                // 범위: -0.5 (거리 0) ~ 0.5 (radarRange)
                distances[i] = (radarHits[i].distance / radarRange) - 0.5f;
            }
            else
            {
                // 감지 안 됨 = 최대 거리
                // 1.0 / radarRange - 0.5 = 0.5
                distances[i] = 0.5f;
            }
        }

        return distances;
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

        // 감지된 레이만 표시 (원 제거)
        foreach (var hit in radarHits)
        {
            float angle = hit.Key * (360f / rayCount);
            Vector3 direction = Quaternion.Euler(0, angle, 0) * transform.forward;

            Gizmos.color = Color.red;

            // 레이 그리기
            Gizmos.DrawLine(transform.position + Vector3.up * rayHeight,
                          transform.position + Vector3.up * rayHeight + direction * hit.Value.distance);

            // 감지 지점 표시
            Gizmos.DrawSphere(hit.Value.point, 0.5f);
        }
    }

}