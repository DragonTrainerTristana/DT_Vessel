using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 선박 동역학을 처리하는 클래스
/// ShipController와 VesselAgent 모두에서 사용할 수 있습니다.
/// </summary>
public class VesselDynamics : MonoBehaviour
{
    // 선박 파라미터
    [Header("선박 기본 파라미터")]
    public float mass = 1.0f;           // 질량
    public float length = 10.0f;        // 선박 길이
    public float beam = 2.0f;           // 선박 폭
    public float maxSpeed = 5.0f;       // 최대 속도
    public float maxTurnRate = 30.0f;   // 최대 회전 속도
    public float dragCoefficient = 0.1f; // 물의 저항 계수
    
    // 선박 제어 관련 파라미터
    [Header("선박 제어 파라미터")]
    public float accelerationRate = 0.5f;  // 가속률
    public float decelerationRate = 0.2f;  // 감속률
    public float brakeRate = 1.0f;         // 브레이크 감속률
    public float rudderEffectiveness = 1.5f; // 타 효과 계수
    
    // 동역학 상태
    private float currentSpeed = 0f;       // 현재 속도
    private float targetSpeed = 0f;        // 목표 속도
    private float rudderAngle = 0f;        // 타각
    private float effectiveRudderAngle = 0f; // 유효 타각 (속도에 비례)
    private float yawRate = 0f;            // 회전 속도
    private bool isBraking = false;        // 브레이크 상태
    
    // 리지드바디 컴포넌트
    private Rigidbody rb;
    
    // 초기 상태 저장
    private Vector3 initialPosition;
    private Quaternion initialRotation;

    // 상태 프로퍼티 - 외부에서 접근 가능
    public float CurrentSpeed { get { return currentSpeed; } }
    public float RudderAngle { get { return rudderAngle; } }
    public float YawRate { get { return yawRate; } }
    public bool IsBraking { get { return isBraking; } }

    /// <summary>
    /// 초기화
    /// </summary>
    public void Initialize(Rigidbody rigidbody)
    {
        rb = rigidbody;
        
        // 초기 상태 저장
        if (rb != null && rb.gameObject != null)
        {
            initialPosition = rb.gameObject.transform.position;
            initialRotation = rb.gameObject.transform.rotation;
        }
        
        // 상태 변수 초기화
        ResetState();
    }
    
    /// <summary>
    /// 상태 리셋
    /// </summary>
    public void ResetState()
    {
        currentSpeed = 0f;
        targetSpeed = 0f;
        rudderAngle = 0f;
        effectiveRudderAngle = 0f;
        yawRate = 0f;
        isBraking = false;
        
        if (rb != null)
        {
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.gameObject.transform.position = initialPosition;
            rb.gameObject.transform.rotation = initialRotation;
        }
    }

    /// <summary>
    /// 물리 업데이트 함수 - 외부에서 FixedUpdate에 호출
    /// </summary>
    public void UpdateDynamics(float deltaTime)
    {
        if (rb == null) return;
        
        // 1. 속도 업데이트
        if (isBraking)
        {
            // 브레이크 중이면 빠르게 감속
            currentSpeed = Mathf.MoveTowards(currentSpeed, 0f, brakeRate * deltaTime);
        }
        else if (targetSpeed > currentSpeed)
        {
            // 가속
            currentSpeed = Mathf.MoveTowards(currentSpeed, targetSpeed, accelerationRate * deltaTime);
        }
        else
        {
            // 일반 감속
            currentSpeed = Mathf.MoveTowards(currentSpeed, targetSpeed, decelerationRate * deltaTime);
        }
        
        // 2. 속도 제한
        currentSpeed = Mathf.Clamp(currentSpeed, 0f, maxSpeed);
        
        // 3. 타(rudder) 효과 계산 - 속도에 비례
        float speedRatio = currentSpeed / maxSpeed;
        effectiveRudderAngle = rudderAngle * speedRatio;
        
        // 4. 회전 효과 계산
        float lengthFactor = 10.0f / length; // 길이에 반비례
        float beamFactor = beam / 2.0f;      // 폭에 비례
        float turnFactor = rudderEffectiveness * speedRatio * lengthFactor * beamFactor;
        
        // 5. 회전 속도(yaw rate) 업데이트
        yawRate = effectiveRudderAngle * turnFactor;
        
        // 6. 물리 업데이트
        if (!float.IsNaN(currentSpeed) && !float.IsInfinity(currentSpeed))
        {
            // 전방 방향으로만 이동
            Vector3 forwardVelocity = rb.gameObject.transform.forward * currentSpeed;
            rb.velocity = forwardVelocity;
            
            // 회전 적용
            if (!float.IsNaN(yawRate) && !float.IsInfinity(yawRate))
            {
                rb.gameObject.transform.Rotate(0, yawRate * deltaTime, 0);
            }
        }
        
        // 7. 항력(drag) 적용 - 자동 감속
        if (targetSpeed < 0.1f && !isBraking) // 입력이 없을 때 (브레이크 중이 아닐 때)
        {
            // 물의 저항에 의한 자연 감속
            currentSpeed *= (1.0f - dragCoefficient * deltaTime);
        }
    }
    
    /// <summary>
    /// 타각 설정
    /// </summary>
    public void SetRudderAngle(float angle)
    {
        rudderAngle = Mathf.Clamp(angle, -maxTurnRate, maxTurnRate);
    }
    
    /// <summary>
    /// 목표 속도 설정
    /// </summary>
    public void SetTargetSpeed(float speed)
    {
        targetSpeed = Mathf.Clamp(speed, 0f, maxSpeed);
    }
    
    /// <summary>
    /// 브레이크 상태 설정
    /// </summary>
    public void SetBraking(bool brake)
    {
        isBraking = brake;
    }
} 