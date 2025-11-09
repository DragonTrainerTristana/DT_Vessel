using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class VesselRadar : MonoBehaviour
{
    [Header("레이더 설정")]
    public float radarRange = 100f;           // 레이더 감지 거리
    public int rayCount = 360;                // 레이 개수 (1도 간격)
    public int regionCount = 30;              // Region 개수 (12도 간격)
    public float rayHeight = 1f;              // 레이 높이 (수면 위)
    public bool showDebugRays = true;         // 디버그 레이 표시 여부

    [Header("레이어 설정")]
    public LayerMask detectionLayers;         // 감지할 레이어

    // 레이더 감지 결과 저장
    private Dictionary<int, RaycastHit> radarHits = new Dictionary<int, RaycastHit>();
    private List<GameObject> detectedVessels = new List<GameObject>();

    // Region 데이터 구조
    public struct RegionData
    {
        public float closestDistance;     // 가장 가까운 물체 거리 (정규화: 0~1)
        public float relativeBearing;     // 상대 방위각 (정규화: -1~1)
        public float speedRatio;          // 속도 비율 (정규화: 0~2)
        public float tcpa;                // Time to CPA (정규화: 0~1)
        public float dcpa;                // Distance at CPA (정규화: 0~1)
        public float phase;               // Navigation phase (0=normal, 1=avoidance)
    }

    private VesselAgent myAgent;  // 자신의 VesselAgent 참조

    private void Start()
    {
        myAgent = GetComponent<VesselAgent>();
    }

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
    /// 특정 region의 데이터 반환 (MDPI 2024 방식)
    /// </summary>
    public RegionData GetRegionData(int regionIndex)
    {
        if (myAgent == null) myAgent = GetComponent<VesselAgent>();

        RegionData data = new RegionData();

        // Region 각도 범위 계산 (12도씩)
        float regionAngle = 360f / regionCount;
        float startAngle = regionIndex * regionAngle;
        float endAngle = (regionIndex + 1) * regionAngle;

        // 해당 region의 ray들 수집
        List<float> distances = new List<float>();
        GameObject closestVessel = null;
        float minDistance = radarRange;

        int raysPerRegion = rayCount / regionCount;
        int startRay = regionIndex * raysPerRegion;
        int endRay = (regionIndex + 1) * raysPerRegion;

        for (int i = startRay; i < endRay; i++)
        {
            if (radarHits.ContainsKey(i))
            {
                float dist = radarHits[i].distance;
                distances.Add(dist);

                if (dist < minDistance && radarHits[i].collider.CompareTag("Vessel"))
                {
                    minDistance = dist;
                    closestVessel = radarHits[i].collider.gameObject;
                }
            }
        }

        // 1. Closest Distance (정규화)
        data.closestDistance = minDistance / radarRange;

        // 2-6: 가장 가까운 선박이 있으면 계산
        if (closestVessel != null && myAgent != null)
        {
            VesselAgent otherAgent = closestVessel.GetComponent<VesselAgent>();
            if (otherAgent != null)
            {
                Vector3 toOther = otherAgent.transform.position - transform.position;

                // 2. Relative Bearing (정규화: -1~1)
                float bearing = Vector3.SignedAngle(transform.forward, toOther, Vector3.up);
                data.relativeBearing = bearing / 180f;

                // 3. Speed Ratio (정규화: 0~2)
                float mySpeed = myAgent.vesselDynamics.CurrentSpeed;
                float otherSpeed = otherAgent.vesselDynamics.CurrentSpeed;
                float maxSpeed = Mathf.Max(myAgent.vesselDynamics.maxSpeed, otherAgent.vesselDynamics.maxSpeed);

                if (maxSpeed > 0.01f)
                    data.speedRatio = Mathf.Clamp01((mySpeed + otherSpeed) / (2f * maxSpeed));
                else
                    data.speedRatio = 0f;

                // 속도 벡터 계산
                Vector3 myVelocity = transform.forward * mySpeed;
                Vector3 otherVelocity = otherAgent.transform.forward * otherSpeed;

                // 4. TCPA (정규화: 0~1, 100초 기준)
                float tcpa = COLREGsHandler.CalculateTCPA(
                    transform.position, myVelocity,
                    otherAgent.transform.position, otherVelocity
                );
                data.tcpa = Mathf.Clamp01(1.0f - (tcpa / 100f));  // 가까울수록 1

                // 5. DCPA (정규화: 0~1, 50m 기준)
                float dcpa = COLREGsHandler.CalculateDCPA(
                    transform.position, myVelocity,
                    otherAgent.transform.position, otherVelocity
                );
                data.dcpa = Mathf.Clamp01(1.0f - (dcpa / 50f));  // 가까울수록 1

                // 6. Phase (0=normal, 1=collision avoidance)
                float risk = COLREGsHandler.CalculateRisk(
                    transform.position, transform.forward, mySpeed,
                    otherAgent.transform.position, otherAgent.transform.forward, otherSpeed
                );
                data.phase = risk > 0.3f ? 1.0f : 0.0f;  // 위험도 30% 이상이면 회피 모드
            }
        }
        else
        {
            // 선박이 없으면 기본값
            data.relativeBearing = 0f;
            data.speedRatio = 0f;
            data.tcpa = 0f;
            data.dcpa = 0f;
            data.phase = 0f;
        }

        return data;
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