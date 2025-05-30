using UnityEngine;
using System.Collections.Generic;

public class COLREGsHandler
{
    // 충돌 회피 상황 열거형
    public enum CollisionSituation
    {
        None,
        HeadOn,
        CrossingStandOn,
        CrossingGiveWay,
        Overtaking
    }

    // 상수 정의
    private const float HEAD_ON_ANGLE = 15f;        // 정면 조우 판정 각도
    private const float CROSSING_ANGLE = 112.5f;    // 횡단 상황@ 판정 각도
    private const float OVERTAKING_ANGLE = 112.5f;  // 추월 판정 각도
    private const float DETECTION_RANGE = 100f;     // 충돌 위험 감지 거리

    /// <summary>
    /// 두 선박 간의 상황을 판단
    /// </summary>
    public static CollisionSituation AnalyzeSituation(
        Vector3 myPosition, Vector3 myForward, float mySpeed,
        Vector3 otherPosition, Vector3 otherForward, float otherSpeed)
    {
        Vector3 toOther = otherPosition - myPosition;
        float distance = toOther.magnitude;
        
        // 감지 범위 밖이면 무시
        if (distance > DETECTION_RANGE) return CollisionSituation.None;

        // 상대 방향 각도 계산
        float bearingAngle = Vector3.SignedAngle(myForward, toOther, Vector3.up);
        float otherBearingAngle = Vector3.SignedAngle(otherForward, -toOther, Vector3.up);

        // Head-on 상황 체크
        if (Mathf.Abs(bearingAngle) < HEAD_ON_ANGLE && Mathf.Abs(otherBearingAngle) < HEAD_ON_ANGLE)
        {
            return CollisionSituation.HeadOn;
        }

        // Overtaking 상황 체크
        if (Mathf.Abs(otherBearingAngle) < OVERTAKING_ANGLE && mySpeed > otherSpeed)
        {
            return CollisionSituation.Overtaking;
        }

        // Crossing 상황 체크
        if (Mathf.Abs(bearingAngle) < CROSSING_ANGLE && Mathf.Abs(otherBearingAngle) < CROSSING_ANGLE)
        {
            // 우측에서 접근하는 선박이 Stand-on
            return bearingAngle > 0 ? CollisionSituation.CrossingGiveWay : CollisionSituation.CrossingStandOn;
        }

        return CollisionSituation.None;
    }

    /// <summary>
    /// 상황에 따른 권장 행동 계산
    /// </summary>
    public static (float suggestedRudder, float suggestedSpeed) GetRecommendedAction(
        CollisionSituation situation, 
        float currentSpeed,
        Vector3 toOther)
    {
        float suggestedRudder = 0f;
        float suggestedSpeed = currentSpeed;

        switch (situation)
        {
            case CollisionSituation.HeadOn:
                suggestedRudder = 0.5f;  // 우현 변침
                break;

            case CollisionSituation.CrossingStandOn:
                suggestedSpeed = currentSpeed;  // 속도 유지
                break;

            case CollisionSituation.CrossingGiveWay:
                suggestedRudder = 0.5f;  // 우현 변침
                suggestedSpeed = currentSpeed * 0.7f;  // 감속
                break;

            case CollisionSituation.Overtaking:
                suggestedRudder = 0.3f;  // 우현 변침
                break;
        }

        return (suggestedRudder, suggestedSpeed);
    }

    /// <summary>
    /// COLREGs 규칙 준수 여부 평가
    /// </summary>
    public static float EvaluateCompliance(
        CollisionSituation situation,
        float actualRudder,
        float recommendedRudder)
    {
        if (situation == CollisionSituation.None) return 0f;

        float reward = 0f;

        // 우현 변침 규칙 준수 평가
        if (situation != CollisionSituation.CrossingStandOn)
        {
            if (actualRudder > 0) // 우현 변침
                reward += 1.0f;
            else if (actualRudder < 0) // 좌현 변침 (패널티)
                reward -= 2.0f;
        }

        return reward;
    }

    /// <summary>
    /// COLREGs 위험도 계산
    /// </summary>
    public static float CalculateRisk(
        Vector3 myPosition, Vector3 myForward, float mySpeed,
        Vector3 otherPosition, Vector3 otherForward, float otherSpeed)
    {
        Vector3 toOther = otherPosition - myPosition;
        float distance = toOther.magnitude;
        
        // 거리가 멀수록 위험도 감소
        if (distance > DETECTION_RANGE) return 0f;
        
        float risk = 1.0f - (distance / DETECTION_RANGE);  // 기본 거리 기반 위험도
        
        // 상대 방향 각도 계산
        float bearingAngle = Vector3.SignedAngle(myForward, toOther, Vector3.up);
        float otherBearingAngle = Vector3.SignedAngle(otherForward, -toOther, Vector3.up);
        
        // 상황별 위험도 가중치
        var situation = AnalyzeSituation(myPosition, myForward, mySpeed, otherPosition, otherForward, otherSpeed);
        switch (situation)
        {
            case CollisionSituation.HeadOn:
                risk *= 2.0f;  // Head-on은 가장 위험
                break;
            case CollisionSituation.CrossingGiveWay:
                risk *= 1.5f;  // Give-way는 두 번째로 위험
                break;
            case CollisionSituation.Overtaking:
                risk *= 1.2f;  // Overtaking은 상대적으로 덜 위험
                break;
            case CollisionSituation.CrossingStandOn:
                risk *= 1.3f;  // Stand-on도 주의 필요
                break;
        }
        
        // 상대 속도에 따른 위험도 조정
        float relativeSpeed = mySpeed + otherSpeed;
        risk *= (1.0f + relativeSpeed / 20f);  // 속도가 빠를수록 위험도 증가
        
        return Mathf.Clamp01(risk);  // 0~1 사이로 정규화
    }

    /// <summary>
    /// 가장 위험한 상황 분석
    /// </summary>
    public static (CollisionSituation situation, float maxRisk, GameObject vessel) 
    AnalyzeMostDangerousVessel(VesselAgent agent, List<GameObject> detectedVessels)
    {
        float maxRisk = 0f;
        CollisionSituation mostDangerousSituation = CollisionSituation.None;
        GameObject mostDangerousVessel = null;

        foreach (var vesselObj in detectedVessels)
        {
            VesselAgent otherAgent = vesselObj.GetComponent<VesselAgent>();
            if (otherAgent == null) continue;

            float risk = CalculateRisk(
                agent.transform.position, agent.transform.forward, agent.vesselDynamics.CurrentSpeed,
                otherAgent.transform.position, otherAgent.transform.forward, otherAgent.vesselDynamics.CurrentSpeed
            );

            if (risk > maxRisk)
            {
                maxRisk = risk;
                mostDangerousSituation = AnalyzeSituation(
                    agent.transform.position, agent.transform.forward, agent.vesselDynamics.CurrentSpeed,
                    otherAgent.transform.position, otherAgent.transform.forward, otherAgent.vesselDynamics.CurrentSpeed
                );
                mostDangerousVessel = vesselObj;
            }
        }

        return (mostDangerousSituation, maxRisk, mostDangerousVessel);
    }
} 
