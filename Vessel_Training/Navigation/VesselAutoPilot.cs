using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 자율 항해 선박 컨트롤러
/// Radar 모드: 레이더 감지 시 급격하게 피함 (큰 rudder)
/// Communication 모드: 미리 감지해서 여유롭게 피함 (작은 rudder)
/// </summary>
public class VesselAutoPilot : MonoBehaviour
{
    public enum NavigationMode
    {
        Radar,          // 레이더 거리에서 반응, 급격하게 피함
        Communication   // 통신 거리에서 미리 반응, 여유롭게 피함
    }

    [Header("Navigation Mode")]
    public NavigationMode mode = NavigationMode.Radar;

    [Header("Detection Range")]
    public float radarRange = 40f;          // Radar 모드 감지 거리
    public float communicationRange = 100f;  // Communication 모드 감지 거리

    [Header("Avoidance Parameters")]
    [Range(0.5f, 1.5f)]
    public float radarRudderMultiplier = 1.2f;      // Radar: 급격한 회피
    [Range(0.2f, 0.6f)]
    public float commRudderMultiplier = 0.4f;       // Communication: 여유로운 회피

    [Header("Goal Settings")]
    public Vector3 goalPosition;
    public bool hasGoal = false;
    public float goalReachedDistance = 10f;

    [Header("References")]
    public VesselDynamics dynamics;
    public VesselRadar radar;

    [Header("Status")]
    public bool hasArrived = false;
    public bool hasCollided = false;

    // 내부 변수
    private List<VesselAutoPilot> allVessels = new List<VesselAutoPilot>();
    private float baseSpeed;

    // Trajectory
    private List<Vector3> trajectoryPoints = new List<Vector3>();
    private float trajectoryInterval = 0.5f;
    private float lastTrajectoryTime;

    public List<Vector3> TrajectoryPoints => trajectoryPoints;

    void Start()
    {
        if (dynamics == null)
            dynamics = GetComponent<VesselDynamics>();

        if (radar == null)
            radar = GetComponent<VesselRadar>();

        lastTrajectoryTime = Time.time;
    }

    public void Initialize(List<VesselAutoPilot> vessels)
    {
        allVessels = vessels;

        // baseSpeed 초기화
        if (dynamics != null)
            baseSpeed = dynamics.maxSpeed * 0.7f;


    }

    public void SetGoal(Vector3 position)
    {
        goalPosition = position;
        hasGoal = true;
        hasArrived = false;
    }

    void FixedUpdate()
    {
        if (!hasGoal || hasArrived || hasCollided) return;

        // Trajectory 기록
        RecordTrajectory();

        // 목적지 도착 체크
        float distanceToGoal = Vector3.Distance(transform.position, goalPosition);
        if (distanceToGoal < goalReachedDistance)
        {
            hasArrived = true;
            dynamics.SetTargetSpeed(0);
            dynamics.SetBraking(true);

            return;
        }

        // 레이더 스캔
        if (radar != null)
            radar.ScanRadar();

        // 자율 항해 로직
        NavigateToGoal();

        // 동역학 업데이트
        dynamics.UpdateDynamics(Time.fixedDeltaTime);
    }

    private void RecordTrajectory()
    {
        if (Time.time - lastTrajectoryTime >= trajectoryInterval)
        {
            trajectoryPoints.Add(transform.position);
            lastTrajectoryTime = Time.time;
        }
    }

    private void NavigateToGoal()
    {
        // 기본: 목적지 방향으로 향함
        Vector3 toGoal = goalPosition - transform.position;
        float goalAngle = Vector3.SignedAngle(transform.forward, toGoal, Vector3.up);

        float targetRudder = 0f;
        float targetSpeed = baseSpeed;

        // COLREGs 회피 계산
        var (avoidRudder, avoidSpeed, needsAvoidance) = CalculateAvoidance();

        if (needsAvoidance)
        {
            // 회피 필요 시
            targetRudder = avoidRudder;
            targetSpeed = avoidSpeed;

        }
        else
        {
            // 회피 불필요 시 목적지로 향함
            targetRudder = Mathf.Clamp(goalAngle / 45f, -1f, 1f) * dynamics.maxTurnRate * 0.5f;
            targetSpeed = baseSpeed;
        }

        // 적용
        dynamics.SetRudderAngle(targetRudder);
        dynamics.SetTargetSpeed(targetSpeed);
        dynamics.SetBraking(false);
    }

    private (float rudder, float speed, bool needsAvoidance) CalculateAvoidance()
    {
        float detectionRange = (mode == NavigationMode.Radar) ? radarRange : communicationRange;

        float maxRudder = 0f;
        float minSpeed = baseSpeed;
        bool needsAvoidance = false;

        foreach (var other in allVessels)
        {
            if (other == this || other == null) continue;
            if (other.hasArrived || other.hasCollided) continue;

            Vector3 toOther = other.transform.position - transform.position;
            float distance = toOther.magnitude;

            // 감지 범위 체크
            if (distance > detectionRange) continue;

            // 상대방이 내 앞에 있는지 확인 (±90° 이내)
            float bearingAngle = Vector3.SignedAngle(transform.forward, toOther, Vector3.up);
            float absBearing = Mathf.Abs(bearingAngle);

            // 뒤에 있는 선박은 무시 (내가 피할 필요 없음)
            if (absBearing > 90f) continue;

            needsAvoidance = true;

            // ========== 간단한 COLREGs 규칙 ==========
            // 1. 항상 오른쪽(우현)으로 돈다
            // 2. 상대가 내 오른쪽에 있으면 (bearingAngle > 0) 내가 Give-way → 더 많이 감속
            // 3. 상대가 내 왼쪽에 있으면 (bearingAngle < 0) 내가 Stand-on → 속도 유지

            bool isGiveWay = bearingAngle > 0;  // 상대가 내 오른쪽 → 내가 양보

            // 거리에 따른 긴급도
            float urgency = 1f - (distance / detectionRange);

            // Rudder 계산: 항상 오른쪽으로, 가까울수록 더 강하게
            float rudderStrength;
            float speedFactor;

            if (mode == NavigationMode.Radar)
            {
                // Radar: 급격하게 회전
                rudderStrength = 0.5f + urgency * 0.5f;  // 0.5 ~ 1.0
                speedFactor = isGiveWay ? 0.7f : 1.0f;
            }
            else  // Communication
            {
                // Communication: 부드럽게 회전, Give-way는 많이 감속
                rudderStrength = 0.2f + urgency * 0.3f;  // 0.2 ~ 0.5
                speedFactor = isGiveWay ? 0.3f : 1.0f;   // Give-way는 70% 감속
            }

            float adjustedRudder = rudderStrength * dynamics.maxTurnRate;  // 항상 양수 (우현)
            float adjustedSpeed = baseSpeed * speedFactor;


            // 가장 큰 회피 각도 선택
            if (adjustedRudder > maxRudder)
            {
                maxRudder = adjustedRudder;
            }

            // 가장 낮은 속도 선택
            minSpeed = Mathf.Min(minSpeed, adjustedSpeed);
        }

        // 최소 속도 보장
        minSpeed = Mathf.Max(minSpeed, baseSpeed * 0.2f);

        return (maxRudder, minSpeed, needsAvoidance);
    }

    void OnCollisionEnter(Collision collision)
    {
        HandleCollision(collision.gameObject);
    }

    void OnTriggerEnter(Collider other)
    {
        HandleCollision(other.gameObject);
    }

    private void HandleCollision(GameObject collidedObject)
    {
        if (hasCollided) return;

        if (collidedObject.CompareTag("Obstacle") || collidedObject.GetComponent<VesselAutoPilot>() != null)
        {
            hasCollided = true;
            dynamics.SetTargetSpeed(0);
            dynamics.SetBraking(true);

        }
    }

    void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;

        // 감지 범위 표시
        float range = (mode == NavigationMode.Radar) ? radarRange : communicationRange;
        Gizmos.color = (mode == NavigationMode.Radar) ? new Color(1, 0, 0, 0.2f) : new Color(0, 1, 0, 0.2f);
        Gizmos.DrawWireSphere(transform.position, range);

        // 목적지 표시
        if (hasGoal)
        {
            Gizmos.color = hasArrived ? Color.green : Color.yellow;
            Gizmos.DrawLine(transform.position, goalPosition);
            Gizmos.DrawSphere(goalPosition, 2f);
        }
    }
}
