using UnityEngine;

// 관찰용 오버헤드 카메라.
// - 빌드(exe)를 그래픽으로 띄우면 "No cameras rendering" 회색 대신 맵 전체를 위에서 내려다봄.
// - 학습(-batchmode/-nographics)에선 생성 안 함 → 학습 속도/결과 영향 0.
// - 구 Input API 미사용 → Input System 충돌 없음.
public class CameraController : MonoBehaviour
{
    // 빈 MonoBehaviour 유지 (기존 씬에 붙어 있어도 무해).

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void SetupOverheadCamera()
    {
        // headless 학습 모드면 카메라 불필요
        if (Application.isBatchMode) return;

        // 기존 메인 카메라 재사용, 없으면 생성
        Camera cam = Camera.main;
        if (cam == null)
        {
            var go = new GameObject("AutoOverheadCamera");
            go.tag = "MainCamera";
            cam = go.AddComponent<Camera>();
            go.AddComponent<AudioListener>();
        }
        if (cam.GetComponent<OverheadCameraFollow>() == null)
            cam.gameObject.AddComponent<OverheadCameraFollow>();
    }
}

// 매 프레임 모든 선박 위치로 맵 전체를 프레이밍하는 top-down 직교 카메라.
public class OverheadCameraFollow : MonoBehaviour
{
    private Camera cam;
    private VesselAgent[] agents;

    void Awake()
    {
        cam = GetComponent<Camera>();
        cam.depth = 100; // 다른 카메라 위에 풀스크린으로 렌더
        cam.orthographic = true;
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.05f, 0.10f, 0.18f); // 바다색 (회색 방지 확인용)
        cam.nearClipPlane = 0.1f;
    }

    void LateUpdate()
    {
        // 선박 목록 캐시 (스폰 후 1회 확보)
        if (agents == null || agents.Length == 0)
            agents = Object.FindObjectsOfType<VesselAgent>();
        if (agents == null || agents.Length == 0) return;

        float minX = float.MaxValue, minZ = float.MaxValue;
        float maxX = float.MinValue, maxZ = float.MinValue;
        int n = 0;
        foreach (var a in agents)
        {
            if (a == null) continue;
            Vector3 p = a.transform.position;
            if (p.x < minX) minX = p.x;
            if (p.z < minZ) minZ = p.z;
            if (p.x > maxX) maxX = p.x;
            if (p.z > maxZ) maxZ = p.z;
            n++;
        }
        if (n == 0) return;

        Vector3 center = new Vector3((minX + maxX) * 0.5f, 0f, (minZ + maxZ) * 0.5f);
        float extX = Mathf.Max(maxX - minX, 1f);
        float extZ = Mathf.Max(maxZ - minZ, 1f);
        float aspect = cam.aspect > 0.01f ? cam.aspect : (16f / 9f);

        // 가로/세로 둘 다 들어오게 + 30% 여백
        float size = Mathf.Max(extZ * 0.5f, extX * 0.5f / aspect) * 1.3f;
        cam.orthographicSize = Mathf.Max(size, 3f);

        float camHeight = Mathf.Max(size * 2f, 100f);
        cam.transform.position = center + Vector3.up * camHeight;
        cam.transform.rotation = Quaternion.Euler(90f, 0f, 0f); // 정수직 내려다보기
        cam.farClipPlane = camHeight + 2000f;
    }
}
