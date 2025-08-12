using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public struct VesselCommunicationData
{
    public Vector3 position;                 // 위치
    public Vector3 velocity;                 // 속도 벡터 (방향 포함)
    public float speed;                      // 속도 크기
    public COLREGsHandler.CollisionSituation currentCOLREGsSituation;  // 현재 COLREGs 상황
    public float riskLevel;                  // 위험도
    public float timeStamp;                  // 데이터 시간 기록
    
    // 전체 상태 정보 추가 (46차원)
    public float[] radarData;                // 섹터별 레이더 데이터 (24차원)
    public float[] vesselState;              // 선박 상태 (4차원)
    public float[] goalInfo;                 // 목표 정보 (3차원)
    public float[] colregsSituation;         // COLREGs 상황 (4차원)
    public float[] dangerLevel;              // 위험도 (1차원)
}

public class VesselCommunication : MonoBehaviour
{
    [Header("통신 설정")]
    public int maxCommunicationPartners = 4;     // 최대 통신 가능 선박 수
    public float communicationRange = 100f;       // 통신 가능 거리
    public float communicationInterval = 0.1f;    // 통신 주기

    private VesselAgent myVesselAgent;
    private Dictionary<int, VesselCommunicationData> receivedData;
    private float lastCommunicationTime;

    private void Start()
    {
        myVesselAgent = GetComponent<VesselAgent>();
        receivedData = new Dictionary<int, VesselCommunicationData>();
        lastCommunicationTime = 0f;
    }

    private void Update()
    {
        if (Time.time - lastCommunicationTime >= communicationInterval)
        {
            CommunicateWithNearbyVessels();
            lastCommunicationTime = Time.time;
        }
    }

    private void CommunicateWithNearbyVessels()
    {
        // 주변 선박 찾기
        var nearbyVessels = FindNearbyVessels();
        
        // 기존 데이터 초기화
        receivedData.Clear();

        // 가장 가까운 4대의 선박과만 통신
        foreach (var vessel in nearbyVessels.Take(maxCommunicationPartners))
        {
            var vesselComm = vessel.GetComponent<VesselCommunication>();
            if (vesselComm != null)
            {
                // 통신 데이터 교환
                ExchangeData(vesselComm);
            }
        }
    }

    private List<GameObject> FindNearbyVessels()
    {
        var vessels = new List<GameObject>();
        var colliders = Physics.OverlapSphere(transform.position, communicationRange);
        
        foreach (var collider in colliders)
        {
            if (collider.gameObject != gameObject && collider.GetComponent<VesselAgent>() != null)
            {
                vessels.Add(collider.gameObject);
            }
        }

        // 거리에 따라 정렬
        vessels.Sort((a, b) => 
            Vector3.Distance(transform.position, a.transform.position)
            .CompareTo(Vector3.Distance(transform.position, b.transform.position)));

        return vessels;
    }

    private void ExchangeData(VesselCommunication otherVessel)
    {
        var myData = CreateCommunicationData();
        var otherId = otherVessel.gameObject.GetInstanceID();

        // 상대방의 데이터 받기
        var otherData = otherVessel.GetVesselData();
        
        // 상대적 거리 계산
        float relativeDistance = Vector3.Distance(transform.position, otherVessel.transform.position);
        
        // 데이터 저장 (ID는 여전히 필요하지만, 통신 데이터에는 포함하지 않음)
        receivedData[otherId] = otherData;
    }

    private VesselCommunicationData CreateCommunicationData()
    {
        // 레이더 스캔 실행
        myVesselAgent.radar.ScanRadar();
        
        // 섹터별 레이더 데이터 수집 (24차원: 섹터당 [min, median(p50), hit_ratio])
        float[] radarData = new float[24];
        int radarIndex = 0;
        for (int sector = 0; sector < 8; sector++)
        {
            const int samplesPerSector = 45;
            float[] distancesNorm = new float[samplesPerSector];
            int hitCount = 0;

            for (int angle = 0; angle < samplesPerSector; angle++)
            {
                float distance = myVesselAgent.radar.GetDistanceAtAngle(sector * 45 + angle);
                float dNorm = Mathf.Clamp01(distance / myVesselAgent.radar.radarRange);
                distancesNorm[angle] = dNorm;
                if (dNorm < 1.0f) hitCount++;
            }

            float minNorm = 1.0f;
            for (int i = 0; i < samplesPerSector; i++)
            {
                float v = distancesNorm[i];
                if (v < minNorm) minNorm = v;
            }

            System.Array.Sort(distancesNorm);
            float medianNorm = (samplesPerSector % 2 == 1)
                ? distancesNorm[samplesPerSector / 2]
                : 0.5f * (distancesNorm[samplesPerSector / 2 - 1] + distancesNorm[samplesPerSector / 2]);

            float hitRatio = (float)hitCount / samplesPerSector;

            radarData[radarIndex++] = minNorm;
            radarData[radarIndex++] = medianNorm;
            radarData[radarIndex++] = hitRatio;
        }
        
        // 선박 상태 수집 (4차원)
        float[] vesselState = new float[4];
        vesselState[0] = myVesselAgent.vesselDynamics.CurrentSpeed / myVesselAgent.vesselDynamics.maxSpeed;  // 정규화된 속도
        vesselState[1] = myVesselAgent.transform.forward.x;  // 선수 방향 x
        vesselState[2] = myVesselAgent.transform.forward.z;  // 선수 방향 z
        vesselState[3] = myVesselAgent.vesselDynamics.YawRate / myVesselAgent.vesselDynamics.maxTurnRate;  // 정규화된 회전 속도
        

        
        // 목표 정보 수집 (3차원)
        float[] goalInfo = new float[3];
        if (myVesselAgent.hasGoal)
        {
            Vector3 directionToGoal = (myVesselAgent.goalPosition - myVesselAgent.transform.position).normalized;
            goalInfo[0] = directionToGoal.x;  // 목표 방향 x
            goalInfo[1] = directionToGoal.z;  // 목표 방향 z
            goalInfo[2] = Vector3.Distance(myVesselAgent.transform.position, myVesselAgent.goalPosition) / myVesselAgent.radarRange;  // 정규화된 목표 거리
        }
        else
        {
            goalInfo[0] = 0f;
            goalInfo[1] = 0f;
            goalInfo[2] = 0f;
        }
        
        // COLREGs 상황 및 위험도 관측 (12차원 + 3차원)
        var (mostDangerousSituation, maxRisk, dangerousVessel) = 
            COLREGsHandler.AnalyzeMostDangerousVessel(myVesselAgent, myVesselAgent.radar.GetDetectedVessels());
        
        // COLREGs 상황 one-hot (4차원)
        float[] colregsSituation = new float[4];
        colregsSituation[0] = mostDangerousSituation == COLREGsHandler.CollisionSituation.HeadOn ? 1.0f : 0.0f;
        colregsSituation[1] = mostDangerousSituation == COLREGsHandler.CollisionSituation.CrossingStandOn ? 1.0f : 0.0f;
        colregsSituation[2] = mostDangerousSituation == COLREGsHandler.CollisionSituation.CrossingGiveWay ? 1.0f : 0.0f;
        colregsSituation[3] = mostDangerousSituation == COLREGsHandler.CollisionSituation.Overtaking ? 1.0f : 0.0f;

        // 위험도 스칼라 (1차원)
        float[] dangerLevel = new float[1];
        dangerLevel[0] = maxRisk;
        
        return new VesselCommunicationData
        {
            position = transform.position,
            velocity = myVesselAgent.vesselDynamics.GetComponent<Rigidbody>().velocity,
            speed = myVesselAgent.vesselDynamics.CurrentSpeed,
            currentCOLREGsSituation = GetCurrentCOLREGsSituation(),
            riskLevel = CalculateCurrentRiskLevel(),
            timeStamp = Time.time,
            
            // 전체 상태 정보 추가
            radarData = radarData,                    // 24차원
            vesselState = vesselState,                // 4차원
            goalInfo = goalInfo,                      // 3차원
            colregsSituation = colregsSituation,      // 4차원
            dangerLevel = dangerLevel                 // 1차원
        };
    }

    public VesselCommunicationData GetVesselData()
    {
        return CreateCommunicationData();
    }

    // 현재 통신 중인 선박들의 데이터 가져오기
    public Dictionary<int, VesselCommunicationData> GetCommunicationData()
    {
        return new Dictionary<int, VesselCommunicationData>(receivedData);
    }

    private COLREGsHandler.CollisionSituation GetCurrentCOLREGsSituation()
    {
        // COLREGs 상황 분석 로직
        // 현재 가장 위험한 상황을 반환
        var (situation, _, _) = COLREGsHandler.AnalyzeMostDangerousVessel(
            myVesselAgent, 
            myVesselAgent.radar.GetDetectedVessels()
        );
        return situation;
    }

    private float CalculateCurrentRiskLevel()
    {
        // 현재 위험도 계산 로직
        var (_, risk, _) = COLREGsHandler.AnalyzeMostDangerousVessel(
            myVesselAgent, 
            myVesselAgent.radar.GetDetectedVessels()
        );
        return risk;
    }
} 