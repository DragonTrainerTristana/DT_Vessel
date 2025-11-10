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

    // COLREGs 우선순위 (높을수록 우선)
    private static readonly Dictionary<CollisionSituation, int> COLREGs_PRIORITY = new Dictionary<CollisionSituation, int>
    {
        { CollisionSituation.HeadOn, 5 },           // 가장 높은 우선순위
        { CollisionSituation.CrossingGiveWay, 4 },  // Give-way는 높은 우선순위
        { CollisionSituation.Overtaking, 3 },       // 추월은 중간 우선순위
        { CollisionSituation.CrossingStandOn, 2 },  // Stand-on은 낮은 우선순위
        { CollisionSituation.None, 0 }              // 우선순위 없음
    };

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
        float suggestedSpeed = currentSpeed;
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
                    suggestedSpeed = currentSpeed * 0.5f;  // Significant speed reduction
                }
                else if (tcpa < EARLY_ACTION_TIME)
                {
                    suggestedRudder = 0.7f;  // Early action
                }
                break;

            case CollisionSituation.CrossingStandOn:
                // Rule 17(a): Stand-on vessel shall keep course and speed
                suggestedSpeed = currentSpeed;
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
                suggestedSpeed = currentSpeed * 0.7f;  // 감속

                // Rule 16: Early action
                if (tcpa < EARLY_ACTION_TIME)
                {
                    suggestedRudder = 0.7f;  // 더 강한 변침
                    suggestedSpeed = currentSpeed * 0.5f;  // 더 강한 감속
                }

                // Rule 16: Substantial action
                if (tcpa < SUBSTANTIAL_ACTION_TIME || dcpa < SAFE_PASSING_DISTANCE)
                {
                    suggestedRudder = 1.0f;  // 최대 우현 변침
                    suggestedSpeed = currentSpeed * 0.3f;  // 강력한 감속
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
                    suggestedSpeed = currentSpeed * 0.8f;  // 추월 중 감속
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
    /// TCPA (Time to Closest Point of Approach) 계산
    /// </summary>
    public static float CalculateTCPA(
        Vector3 myPosition, Vector3 myVelocity,
        Vector3 otherPosition, Vector3 otherVelocity)
    {
        Vector3 relativePosition = otherPosition - myPosition;
        Vector3 relativeVelocity = otherVelocity - myVelocity;

        float relativeSpeed = relativeVelocity.magnitude;

        // 상대 속도가 0이면 TCPA 계산 불가 (평행 이동)
        if (relativeSpeed < 0.01f) return float.MaxValue;

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

    /// <summary>
    /// 모든 COLREGs 상황을 우선순위별로 분석
    /// </summary>
    public static List<(CollisionSituation situation, float risk, GameObject vessel, int priority)> 
    AnalyzeAllCOLREGsSituations(VesselAgent agent, List<GameObject> detectedVessels)
    {
        var allSituations = new List<(CollisionSituation, float, GameObject, int)>();
        
        foreach (var vesselObj in detectedVessels)
        {
            VesselAgent otherAgent = vesselObj.GetComponent<VesselAgent>();
            if (otherAgent == null) continue;

            // 위험도 계산
            float risk = CalculateRisk(
                agent.transform.position, agent.transform.forward, agent.vesselDynamics.CurrentSpeed,
                otherAgent.transform.position, otherAgent.transform.forward, otherAgent.vesselDynamics.CurrentSpeed
            );

            // COLREGs 상황 분석
            var situation = AnalyzeSituation(
                agent.transform.position, agent.transform.forward, agent.vesselDynamics.CurrentSpeed,
                otherAgent.transform.position, otherAgent.transform.forward, otherAgent.vesselDynamics.CurrentSpeed
            );

            // 우선순위 가져오기
            int priority = COLREGs_PRIORITY[situation];

            // 위험도가 임계값 이상인 경우만 추가
            if (risk > 0.1f) // 10% 이상 위험한 경우만
            {
                allSituations.Add((situation, risk, vesselObj, priority));
            }
        }

        // 우선순위와 위험도로 정렬 (우선순위 높은 순, 같은 우선순위면 위험도 높은 순)
        allSituations.Sort((a, b) => 
        {
            if (a.Item4 != b.Item4) // priority
                return b.Item4.CompareTo(a.Item4); // 우선순위 높은 순
            return b.Item2.CompareTo(a.Item2); // 위험도 높은 순
        });

        return allSituations;
    }

    /// <summary>
    /// 우선순위 기반 통합 행동 결정 (Rule 16, 17 완전 구현)
    /// </summary>
    public static (float suggestedRudder, float suggestedSpeed, string reasoning)
    GetPriorityBasedAction(VesselAgent agent, List<GameObject> detectedVessels,
        Dictionary<GameObject, bool> vesselsTakingAction = null)
    {
        var allSituations = AnalyzeAllCOLREGsSituations(agent, detectedVessels);

        if (allSituations.Count == 0)
        {
            return (0f, agent.vesselDynamics.CurrentSpeed, "No COLREGs situations");
        }

        float totalRudder = 0f;
        float totalSpeed = 0f;
        float totalWeight = 0f;
        string reasoning = "";

        // 상위 3개 상황만 고려 (너무 많은 상황을 동시에 처리하면 혼란)
        int maxSituations = Mathf.Min(3, allSituations.Count);

        for (int i = 0; i < maxSituations; i++)
        {
            var situation = allSituations[i].Item1;
            var risk = allSituations[i].Item2;
            var vesselObj = allSituations[i].Item3;
            var priority = allSituations[i].Item4;

            VesselAgent otherAgent = vesselObj.GetComponent<VesselAgent>();
            if (otherAgent == null) continue;

            // TCPA/DCPA 계산
            Vector3 myVelocity = agent.transform.forward * agent.vesselDynamics.CurrentSpeed;
            Vector3 otherVelocity = otherAgent.transform.forward * otherAgent.vesselDynamics.CurrentSpeed;
            float tcpa = CalculateTCPA(agent.transform.position, myVelocity,
                otherAgent.transform.position, otherVelocity);
            float dcpa = CalculateDCPA(agent.transform.position, myVelocity,
                otherAgent.transform.position, otherVelocity);

            // 상대 선박의 회피 행동 여부 확인
            bool otherVesselTakingAction = false;
            if (vesselsTakingAction != null && vesselsTakingAction.ContainsKey(vesselObj))
            {
                otherVesselTakingAction = vesselsTakingAction[vesselObj];
            }

            // 위험도에 따른 가중치 계산
            float weight = risk * (priority / 5.0f); // 우선순위와 위험도를 모두 고려

            // 개별 행동 계산 (TCPA/DCPA 포함)
            var (rudder, speed) = GetRecommendedAction(
                situation,
                agent.vesselDynamics.CurrentSpeed,
                vesselObj.transform.position - agent.transform.position,
                tcpa,
                dcpa,
                otherVesselTakingAction);

            // 가중 평균으로 통합
            totalRudder += rudder * weight;
            totalSpeed += speed * weight;
            totalWeight += weight;

            reasoning += $"{situation}(R:{risk:F2},T:{tcpa:F1},D:{dcpa:F1}) ";
        }

        // 가중 평균 계산
        float finalRudder = totalWeight > 0 ? totalRudder / totalWeight : 0f;
        float finalSpeed = totalWeight > 0 ? totalSpeed / totalWeight : agent.vesselDynamics.CurrentSpeed;

        // 타각 제한
        finalRudder = Mathf.Clamp(finalRudder, -1f, 1f);
        finalSpeed = Mathf.Clamp(finalSpeed, 0f, agent.vesselDynamics.maxSpeed);

        return (finalRudder, finalSpeed, reasoning.Trim());
    }
} 