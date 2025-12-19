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

    // Rule 16: Early and Substantial Action
    private const float EARLY_ACTION_TIME = 60f;    // Early action threshold (seconds)
    private const float SUBSTANTIAL_ACTION_TIME = 30f; // Substantial action threshold (seconds)

    // Rule 17: Stand-on Vessel Action Thresholds
    private const float RULE_17B_TIME = 20f;        // May take action threshold (seconds)
    private const float RULE_17B_DISTANCE = 30f;    // May take action distance (meters)
    private const float RULE_17C_TIME = 10f;        // Shall take action threshold (seconds)
    private const float RULE_17C_DISTANCE = 15f;    // Shall take action distance (meters)

    // Safe passing distances
    private const float SAFE_PASSING_DISTANCE = 20f; // Rule 8: Safe passing distance (meters)
    private const float CRITICAL_CPA = 10f;         // Critical CPA for emergency action (meters)

    /// <summary>
    /// 두 선박 간의 상황을 판단 (수정됨 2025-11-13)
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
        float absBearingAngle = Mathf.Abs(bearingAngle);

        // 1. Head-on 상황 체크 (±15° 이내)
        if (absBearingAngle < HEAD_ON_ANGLE && Mathf.Abs(otherBearingAngle) < HEAD_ON_ANGLE)
        {
            return CollisionSituation.HeadOn;
        }

        // 2. Overtaking 상황 체크 (수정됨)
        // 상대방이 내 정후방 112.5~247.5° 위치 AND 내가 더 빠름
        if (absBearingAngle > 112.5f && absBearingAngle < 247.5f && mySpeed > otherSpeed * 1.1f)
        {
            return CollisionSituation.Overtaking;
        }

        // 3. Crossing 상황 체크 (수정됨: 5~112.5° 범위만)
        if (absBearingAngle > 5f && absBearingAngle < 112.5f)
        {
            // 우측(+)에서 접근하는 선박 = Give-way
            // 좌측(-)에서 접근하는 선박 = Stand-on
            return bearingAngle > 0 ? CollisionSituation.CrossingGiveWay : CollisionSituation.CrossingStandOn;
        }

        return CollisionSituation.None;
    }

    /// <summary>
    /// 상황에 따른 권장 행동 계산 (Rule 16, 17 완전 구현)
    /// </summary>
    public static (float suggestedRudder, float suggestedSpeed) GetRecommendedAction(
        CollisionSituation situation,
        float currentSpeed,
        Vector3 toOther,
        float tcpa = float.MaxValue,
        float dcpa = float.MaxValue,
        bool otherVesselTakingAction = false)
    {
        float suggestedRudder = 0f;
        // currentSpeed가 0이면 기본 속도 사용 (5.0f는 일반적인 선박 속도)
        float effectiveSpeed = Mathf.Max(currentSpeed, 3.5f);  // 최소 3.5m/s
        float suggestedSpeed = effectiveSpeed;
        float distance = toOther.magnitude;

        switch (situation)
        {
            case CollisionSituation.HeadOn:
                // Rule 14: Head-on situation
                suggestedRudder = 0.5f;  // 우현 변침

                // Rule 16: Early and substantial action
                if (tcpa < SUBSTANTIAL_ACTION_TIME)
                {
                    suggestedRudder = 1.0f;  // Substantial action
                    suggestedSpeed = effectiveSpeed * 0.5f;  // Significant speed reduction
                }
                else if (tcpa < EARLY_ACTION_TIME)
                {
                    suggestedRudder = 0.7f;  // Early action
                }
                break;

            case CollisionSituation.CrossingStandOn:
                // Rule 17(a): Stand-on vessel shall keep course and speed
                suggestedSpeed = effectiveSpeed;
                suggestedRudder = 0f;

                // Rule 17(b): May take action if give-way vessel is not taking appropriate action
                if (!otherVesselTakingAction &&
                    (tcpa < RULE_17B_TIME || distance < RULE_17B_DISTANCE))
                {
                    // May take action to avoid collision by her manoeuvre alone
                    suggestedRudder = -0.5f;  // 좌현 변침 (give-way가 우현으로 가야 하므로 반대)

                    // Rule 17(c): Shall take action when collision cannot be avoided by give-way alone
                    if (tcpa < RULE_17C_TIME || dcpa < RULE_17C_DISTANCE)
                    {
                        // Shall take such action as will best aid to avoid collision
                        suggestedRudder = -1.0f;  // 최대 회피
                        suggestedSpeed = 0f;  // 긴급 정지
                    }
                }
                break;

            case CollisionSituation.CrossingGiveWay:
                // Rule 15 & 16: Give-way vessel shall take early and substantial action
                suggestedRudder = 0.5f;  // 우현 변침
                suggestedSpeed = effectiveSpeed * 0.7f;  // 감속

                // Rule 16: Early action
                if (tcpa < EARLY_ACTION_TIME)
                {
                    suggestedRudder = 0.7f;  // 더 강한 변침
                    suggestedSpeed = effectiveSpeed * 0.5f;  // 더 강한 감속
                }

                // Rule 16: Substantial action
                if (tcpa < SUBSTANTIAL_ACTION_TIME || dcpa < SAFE_PASSING_DISTANCE)
                {
                    suggestedRudder = 1.0f;  // 최대 우현 변침
                    suggestedSpeed = effectiveSpeed * 0.3f;  // 강력한 감속
                }

                // Critical situation
                if (dcpa < CRITICAL_CPA)
                {
                    suggestedRudder = 1.0f;
                    suggestedSpeed = 0f;  // 긴급 정지
                }
                break;

            case CollisionSituation.Overtaking:
                // Rule 13: Overtaking vessel shall keep out of the way
                suggestedRudder = 0.3f;  // 우현 변침

                // Rule 16: Early and substantial action for overtaking
                if (tcpa < EARLY_ACTION_TIME)
                {
                    suggestedRudder = 0.5f;
                }

                if (tcpa < SUBSTANTIAL_ACTION_TIME || dcpa < SAFE_PASSING_DISTANCE)
                {
                    suggestedRudder = 0.8f;
                    suggestedSpeed = effectiveSpeed * 0.8f;  // 추월 중 감속
                }
                break;
        }

        return (suggestedRudder, suggestedSpeed);
    }

    /// <summary>
    /// Give-way vessel이 회피 행동을 하고 있는지 감지 (Rule 17을 위해)
    /// </summary>
    public static bool IsVesselTakingAvoidanceAction(
        Vector3 previousPosition, Vector3 currentPosition,
        Vector3 previousForward, Vector3 currentForward,
        float previousSpeed, float currentSpeed,
        float deltaTime)
    {
        if (deltaTime <= 0) return false;

        // 진행 방향 변화 감지 (도/초)
        float headingChange = Vector3.Angle(previousForward, currentForward) / deltaTime;

        // 속도 변화 감지 (감속)
        float speedChangeRate = (previousSpeed - currentSpeed) / deltaTime;

        // 경로 변화 감지
        Vector3 expectedPosition = previousPosition + previousForward * previousSpeed * deltaTime;
        float pathDeviation = Vector3.Distance(currentPosition, expectedPosition);

        // 회피 행동 판정 기준
        const float MIN_HEADING_CHANGE_RATE = 2.0f;  // 초당 2도 이상 변침
        const float MIN_SPEED_REDUCTION_RATE = 0.5f;  // 초당 0.5m/s 이상 감속
        const float MIN_PATH_DEVIATION = 2.0f;  // 예상 경로에서 2m 이상 이탈

        return (headingChange > MIN_HEADING_CHANGE_RATE) ||
               (speedChangeRate > MIN_SPEED_REDUCTION_RATE) ||
               (pathDeviation > MIN_PATH_DEVIATION);
    }

    /// <summary>
    /// COLREGs 규칙 준수 여부 평가 (개선된 버전)
    /// </summary>
    public static float EvaluateCompliance(
        CollisionSituation situation,
        float actualRudder,
        float recommendedRudder,
        float actualSpeed = -1f,
        float recommendedSpeed = -1f,
        float tcpa = float.MaxValue,
        float dcpa = float.MaxValue)
    {
        if (situation == CollisionSituation.None) return 0f;

        float reward = 0f;

        // Rule 16: Early and Substantial Action 평가
        if (situation == CollisionSituation.CrossingGiveWay ||
            situation == CollisionSituation.HeadOn ||
            situation == CollisionSituation.Overtaking)
        {
            // 우현 변침 준수
            if (actualRudder > 0)
            {
                reward += 1.0f;

                // Early action bonus
                if (tcpa > EARLY_ACTION_TIME)
                    reward += 0.5f;

                // Substantial action bonus
                if (Mathf.Abs(actualRudder) > 0.7f && tcpa < SUBSTANTIAL_ACTION_TIME)
                    reward += 0.5f;
            }
            else if (actualRudder < 0)
            {
                reward -= 2.0f;  // 좌현 변침 패널티
            }

            // 속도 감소 평가 (Give-way의 경우)
            if (actualSpeed >= 0 && recommendedSpeed >= 0)
            {
                if (actualSpeed < recommendedSpeed)
                    reward += 0.3f;  // 감속 보상
            }
        }

        // Rule 17: Stand-on vessel 평가
        if (situation == CollisionSituation.CrossingStandOn)
        {
            // Rule 17(a): 초기에는 침로/속도 유지
            if (tcpa > RULE_17B_TIME)
            {
                if (Mathf.Abs(actualRudder) < 0.1f)
                    reward += 1.0f;  // 침로 유지 보상
            }
            // Rule 17(b,c): 필요시 회피
            else
            {
                if (Mathf.Abs(actualRudder) > 0.3f)
                    reward += 0.5f;  // 적절한 회피 보상
            }
        }

        // Safe passing distance 보너스
        if (dcpa > SAFE_PASSING_DISTANCE)
            reward += 0.5f;

        return reward;
    }

    /// <summary>
    /// TCPA (Time to Closest Point of Approach) 계산 (수정됨 2025-11-13)
    /// </summary>
    public static float CalculateTCPA(
        Vector3 myPosition, Vector3 myVelocity,
        Vector3 otherPosition, Vector3 otherVelocity)
    {
        Vector3 relativePosition = otherPosition - myPosition;
        Vector3 relativeVelocity = otherVelocity - myVelocity;

        float relativeSpeed = relativeVelocity.magnitude;

        // 상대 속도가 거의 0 (평행 이동 또는 정지)
        if (relativeSpeed < 0.01f)
        {
            // 현재 거리를 0.5m/s로 나눈 값으로 추정
            // (float.MaxValue는 정규화 시 문제 발생)
            float currentDistance = relativePosition.magnitude;
            return currentDistance / 0.5f;
        }

        // TCPA = -(relative_position · relative_velocity) / |relative_velocity|^2
        float tcpa = -Vector3.Dot(relativePosition, relativeVelocity) / (relativeSpeed * relativeSpeed);

        // TCPA가 음수면 이미 최근접점을 지났음
        return Mathf.Max(0f, tcpa);
    }

    /// <summary>
    /// DCPA (Distance at Closest Point of Approach) 계산
    /// </summary>
    public static float CalculateDCPA(
        Vector3 myPosition, Vector3 myVelocity,
        Vector3 otherPosition, Vector3 otherVelocity)
    {
        Vector3 relativePosition = otherPosition - myPosition;
        Vector3 relativeVelocity = otherVelocity - myVelocity;

        float relativeSpeed = relativeVelocity.magnitude;

        // 상대 속도가 0이면 현재 거리가 DCPA
        if (relativeSpeed < 0.01f) return relativePosition.magnitude;

        float tcpa = CalculateTCPA(myPosition, myVelocity, otherPosition, otherVelocity);

        // DCPA = |relative_position + relative_velocity * TCPA|
        Vector3 positionAtCPA = relativePosition + relativeVelocity * tcpa;
        return positionAtCPA.magnitude;
    }

    /// <summary>
    /// COLREGs 위험도 계산 (TCPA, DCPA 기반 개선)
    /// </summary>
    public static float CalculateRisk(
        Vector3 myPosition, Vector3 myForward, float mySpeed,
        Vector3 otherPosition, Vector3 otherForward, float otherSpeed)
    {
        Vector3 toOther = otherPosition - myPosition;
        float distance = toOther.magnitude;

        // 거리가 멀수록 위험도 감소
        if (distance > DETECTION_RANGE) return 0f;

        // 속도 벡터 계산
        Vector3 myVelocity = myForward.normalized * mySpeed;
        Vector3 otherVelocity = otherForward.normalized * otherSpeed;

        // TCPA와 DCPA 계산
        float tcpa = CalculateTCPA(myPosition, myVelocity, otherPosition, otherVelocity);
        float dcpa = CalculateDCPA(myPosition, myVelocity, otherPosition, otherVelocity);

        // 거리 기반 위험도
        float distanceRisk = 1.0f - (distance / DETECTION_RANGE);

        // TCPA 기반 위험도 (가까운 미래일수록 위험)
        float tcpaRisk = 1.0f / (1.0f + tcpa / 60f); // 60초 기준

        // DCPA 기반 위험도 (가까워질수록 위험)
        float dcpaRisk = 1.0f - Mathf.Clamp01(dcpa / 50f); // 50m 기준

        // 종합 위험도 (가중 평균)
        float risk = (distanceRisk * 0.3f + tcpaRisk * 0.4f + dcpaRisk * 0.3f);

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

        return Mathf.Clamp01(risk);  // 0~1 사이로 정규화
    }

} 
