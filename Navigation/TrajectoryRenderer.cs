using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 선박의 궤적(Trajectory)을 LineRenderer로 그림
/// 사라지지 않고 계속 유지됨
/// </summary>
[RequireComponent(typeof(LineRenderer))]
public class TrajectoryRenderer : MonoBehaviour
{
    [Header("References")]
    public VesselAutoPilot autoPilot;

    [Header("Line Settings")]
    public Color lineColor = Color.white;
    public float lineWidth = 0.3f;
    public Material lineMaterial;

    [Header("Recording Settings")]
    public float recordInterval = 0.3f;
    public float heightOffset = 0.5f;  // 수면 위로 약간 올림

    private LineRenderer lineRenderer;
    private List<Vector3> points = new List<Vector3>();
    private float lastRecordTime;

    void Awake()
    {
        SetupLineRenderer();
    }

    void Start()
    {
        if (autoPilot == null)
            autoPilot = GetComponent<VesselAutoPilot>();

        // 시작 위치 기록
        RecordPoint();
        lastRecordTime = Time.time;
    }

    private void SetupLineRenderer()
    {
        lineRenderer = GetComponent<LineRenderer>();
        if (lineRenderer == null)
            lineRenderer = gameObject.AddComponent<LineRenderer>();

        // 기본 머티리얼 설정
        if (lineMaterial != null)
        {
            lineRenderer.material = lineMaterial;
        }
        else
        {
            // 기본 Unlit Color 머티리얼 사용
            lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        }

        lineRenderer.startColor = lineColor;
        lineRenderer.endColor = lineColor;
        lineRenderer.startWidth = lineWidth;
        lineRenderer.endWidth = lineWidth;
        lineRenderer.useWorldSpace = true;
        lineRenderer.positionCount = 0;

        // 그림자 끄기
        lineRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        lineRenderer.receiveShadows = false;
    }

    void Update()
    {
        // 일정 간격으로 위치 기록
        if (Time.time - lastRecordTime >= recordInterval)
        {
            RecordPoint();
            lastRecordTime = Time.time;
        }

        // LineRenderer 업데이트
        UpdateLineRenderer();
    }

    private void RecordPoint()
    {
        Vector3 point = transform.position;
        point.y += heightOffset;  // 수면 위로
        points.Add(point);
    }

    private void UpdateLineRenderer()
    {
        if (points.Count < 2) return;

        lineRenderer.positionCount = points.Count;
        lineRenderer.SetPositions(points.ToArray());
    }

    /// <summary>
    /// 궤적 색상 변경
    /// </summary>
    public void SetColor(Color color)
    {
        lineColor = color;
        if (lineRenderer != null)
        {
            lineRenderer.startColor = color;
            lineRenderer.endColor = color;
        }
    }

    /// <summary>
    /// 궤적 초기화
    /// </summary>
    public void ClearTrajectory()
    {
        points.Clear();
        if (lineRenderer != null)
        {
            lineRenderer.positionCount = 0;
        }
    }

    /// <summary>
    /// 궤적 포인트 리스트 반환
    /// </summary>
    public List<Vector3> GetTrajectoryPoints()
    {
        return new List<Vector3>(points);
    }

    /// <summary>
    /// 총 이동 거리 계산
    /// </summary>
    public float GetTotalDistance()
    {
        float distance = 0f;
        for (int i = 1; i < points.Count; i++)
        {
            distance += Vector3.Distance(points[i - 1], points[i]);
        }
        return distance;
    }

    void OnDestroy()
    {
        // LineRenderer는 컴포넌트이므로 별도 정리 불필요
        // 궤적 데이터는 자동으로 정리됨
    }

    // Gizmos로도 표시 (에디터에서 확인용)
    void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;
        if (points.Count < 2) return;

        Gizmos.color = lineColor;
        for (int i = 1; i < points.Count; i++)
        {
            Gizmos.DrawLine(points[i - 1], points[i]);
        }
    }
}
