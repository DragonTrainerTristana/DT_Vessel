using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShipController : MonoBehaviour
{
    // 선박 동역학 모듈 참조
    [Header("동역학 모듈")]
    public VesselDynamics vesselDynamics;
    
    // 플레이어 입력 설정
    [Header("조작 설정")]
    public KeyCode forwardKey = KeyCode.W;         // 전진
    public KeyCode brakeKey = KeyCode.S;           // 감속/브레이크
    public KeyCode turnLeftKey = KeyCode.A;        // 좌회전
    public KeyCode turnRightKey = KeyCode.D;       // 우회전
    public KeyCode resetKey = KeyCode.R;           // 리셋
    public float rudderSensitivity = 1.0f;         // 타각 민감도
    public float throttleSensitivity = 0.5f;       // 추진력 민감도

    [Header("디버그 정보")]
    public float displaySpeed = 0f;
    public float displayRudderAngle = 0f;
    public float displayYawRate = 0f;
    public bool displayBraking = false;
    
    private Rigidbody rb;
    
    /// <summary>
    /// 초기화 함수
    /// </summary>
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
        }
        
        // 물리 설정
        rb.useGravity = false;
        rb.drag = 0.0f;
        rb.angularDrag = 0.0f;
        rb.constraints = RigidbodyConstraints.FreezePositionY | RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        
        // 선박 동역학 초기화
        if (vesselDynamics == null)
        {
            // 동적으로 동역학 컴포넌트 추가 (선택적)
            vesselDynamics = gameObject.GetComponent<VesselDynamics>();
            if (vesselDynamics == null)
            {
                vesselDynamics = gameObject.AddComponent<VesselDynamics>();
            }
        }
        
        vesselDynamics.Initialize(rb);
    }

    /// <summary>
    /// 매 프레임마다 실행되는 업데이트 함수
    /// </summary>
    void Update()
    {
        // 플레이어 입력 처리
        HandlePlayerInput();
        
        // 리셋 키 처리
        if (Input.GetKeyDown(resetKey))
        {
            vesselDynamics.ResetState();
        }
        
        // 디버그 정보 업데이트
        displaySpeed = vesselDynamics.CurrentSpeed;
        displayRudderAngle = vesselDynamics.RudderAngle;
        displayYawRate = vesselDynamics.YawRate;
        displayBraking = vesselDynamics.IsBraking;
    }
    
    /// <summary>
    /// 물리 계산 업데이트 함수
    /// </summary>
    void FixedUpdate()
    {
        // 동역학 모듈에 업데이트 위임
        vesselDynamics.UpdateDynamics(Time.fixedDeltaTime);
    }
    
    /// <summary>
    /// 플레이어 입력을 처리하는 함수
    /// </summary>
    private void HandlePlayerInput()
    {
        // 타각 입력 처리
        float rudderInput = 0f;
        if (Input.GetKey(turnLeftKey)) rudderInput -= rudderSensitivity;
        if (Input.GetKey(turnRightKey)) rudderInput += rudderSensitivity;
        vesselDynamics.SetRudderAngle(rudderInput * vesselDynamics.maxTurnRate);
        
        // 브레이크 상태 처리
        bool isBraking = Input.GetKey(brakeKey);
        vesselDynamics.SetBraking(isBraking);
        
        // 추진력 입력 처리
        if (Input.GetKey(forwardKey) && !isBraking) // 브레이크 중에는 가속 비활성화
        {
            vesselDynamics.SetTargetSpeed(vesselDynamics.maxSpeed * throttleSensitivity);
        }
        else if (!isBraking)
        {
            vesselDynamics.SetTargetSpeed(0f); // 입력이 없으면 목표 속도는 0
        }
    }
    
    /// <summary>
    /// GUI에 디버그 정보를 표시하는 함수
    /// </summary>
    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 10, 300, 150));
        GUILayout.Label($"속도: {displaySpeed:F2} m/s ({(displaySpeed/vesselDynamics.maxSpeed*100):F0}%)");
        GUILayout.Label($"타각: {displayRudderAngle:F2}° (유효: {vesselDynamics.RudderAngle * (vesselDynamics.CurrentSpeed/vesselDynamics.maxSpeed):F2}°)");
        GUILayout.Label($"회전율: {displayYawRate:F2}°/s");
        GUILayout.Label($"브레이크: {(displayBraking ? "활성" : "비활성")}");
        GUILayout.EndArea();
    }
    
    /// <summary>
    /// 선박의 방향과 속도를 시각화
    /// </summary>
    void OnDrawGizmos()
    {
        // 선박 전방 방향 표시 (파란색)
        Gizmos.color = Color.blue;
        Gizmos.DrawRay(transform.position, transform.forward * 3f);
        
        // 현재 속도 방향 표시 (빨간색)
        if (Application.isPlaying && rb != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawRay(transform.position, rb.velocity.normalized * 2f);
        }
        
        // 타 방향 표시 (녹색)
        if (Application.isPlaying && vesselDynamics != null)
        {
            Gizmos.color = Color.green;
            float effectiveAngle = vesselDynamics.RudderAngle * (vesselDynamics.CurrentSpeed/vesselDynamics.maxSpeed);
            Vector3 rudderDir = Quaternion.Euler(0, effectiveAngle, 0) * transform.forward;
            Gizmos.DrawRay(transform.position - transform.forward * (vesselDynamics.length/2), rudderDir * 1.5f);
        }
    }
}