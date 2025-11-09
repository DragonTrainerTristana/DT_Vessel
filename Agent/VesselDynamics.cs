using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VesselDynamics : MonoBehaviour
{
    // 선박 종류마다 달라질 수 있음. (이거는 어떻게 러닝해야 하는가? 그냥 통일해도 될까?)
    public float mass = 1.0f;   
    public float length = 10.0f;     
    public float beam = 2.0f;           // 선박 폭
    public float maxSpeed = 5.0f;      
    public float maxTurnRate = 30.0f;   
    public float dragCoefficient = 0.1f; // 물의 저항 계수암
    
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

    private Rigidbody rb;

    private Vector3 initialPosition;
    private Quaternion initialRotation;

    public float CurrentSpeed { get { return currentSpeed; } }
    public float RudderAngle { get { return rudderAngle; } }
    public float YawRate { get { return yawRate; } }
    public bool IsBraking { get { return isBraking; } }
    public Vector3 Velocity { get { return rb != null ? rb.linearVelocity : Vector3.zero; } }

    public void Initialize(Rigidbody rigidbody)
    {
        rb = rigidbody;
        
        if (rb != null && rb.gameObject != null)
        {
            initialPosition = rb.gameObject.transform.position;
            initialRotation = rb.gameObject.transform.rotation;
        }
        
        ResetState();
    }

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
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.gameObject.transform.position = initialPosition;
            rb.gameObject.transform.rotation = initialRotation;
        }
    }

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
            rb.linearVelocity = forwardVelocity;
            
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
    public void SetRudderAngle(float angle)
    {
        rudderAngle = Mathf.Clamp(angle, -maxTurnRate, maxTurnRate);
    }
    

    public void SetTargetSpeed(float speed)
    {
        targetSpeed = Mathf.Clamp(speed, 0f, maxSpeed);
    }

    public void SetBraking(bool brake)
    {
        isBraking = brake;
    }
} 