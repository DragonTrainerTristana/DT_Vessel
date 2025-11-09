using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public struct VesselCommunicationData
{
    public Vector3 position;
    public Vector3 velocity;
    public float speed;
    public COLREGsHandler.CollisionSituation currentCOLREGsSituation;
    public float riskLevel;
    public float timeStamp;

    // Legacy (for backwards compatibility)
    public float[] radarData;              // Old 8 sectors × 3 = 24D
    public float[] vesselState;            // 4D
    public float[] goalInfo;               // 3D
    public float[] colregsSituation;       // Old one-hot 4D
    public float[] dangerLevel;            // 1D

    // New compressed data for neighbors
    public float[] compressedRadarData;    // 8 regions × 3 (min, tcpa, dcpa) = 24D
    public float[] fuzzyCOLREGs;           // Fuzzy weights 4D
}

public class VesselCommunication : MonoBehaviour
{
    [Header("Communication Settings")]
    public int maxCommunicationPartners = 4;
    public float communicationRange = 100f;
    public float communicationInterval = 0.1f;

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
        var nearbyVessels = FindNearbyVessels();
        receivedData.Clear();

        foreach (var vessel in nearbyVessels.Take(maxCommunicationPartners))
        {
            var vesselComm = vessel.GetComponent<VesselCommunication>();
            if (vesselComm != null) ExchangeData(vesselComm);
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

        vessels.Sort((a, b) =>
            Vector3.Distance(transform.position, a.transform.position)
            .CompareTo(Vector3.Distance(transform.position, b.transform.position)));

        return vessels;
    }

    private void ExchangeData(VesselCommunication otherVessel)
    {
        var myData = CreateCommunicationData();
        var otherId = otherVessel.gameObject.GetInstanceID();
        var otherData = otherVessel.GetVesselData();

        receivedData[otherId] = otherData;
    }

    private VesselCommunicationData CreateCommunicationData()
    {
        // ========== Compressed Radar Data (8 regions × 3 = 24D) ==========
        float[] compressedRadar = new float[24];
        int radarIndex = 0;

        // 8 regions (45도씩) with min, tcpa, dcpa
        for (int region = 0; region < 8; region++)
        {
            // Map 8 regions to 30 regions (약 3.75배)
            // region 0 → radar regions 0-3
            // region 1 → radar regions 4-7, etc.
            int startRadarRegion = region * 4;
            int endRadarRegion = (region + 1) * 4;

            float minDist = 1.0f;
            float avgTCPA = 0f;
            float avgDCPA = 0f;
            int validCount = 0;

            for (int r = startRadarRegion; r < endRadarRegion && r < 30; r++)
            {
                var regionData = myVesselAgent.radar.GetRegionData(r);
                if (regionData.closestDistance < minDist)
                    minDist = regionData.closestDistance;

                avgTCPA += regionData.tcpa;
                avgDCPA += regionData.dcpa;
                validCount++;
            }

            if (validCount > 0)
            {
                avgTCPA /= validCount;
                avgDCPA /= validCount;
            }

            compressedRadar[radarIndex++] = minDist;
            compressedRadar[radarIndex++] = avgTCPA;
            compressedRadar[radarIndex++] = avgDCPA;
        }

        // ========== Vessel State (4D) ==========
        float[] vesselState = new float[4];
        vesselState[0] = myVesselAgent.vesselDynamics.CurrentSpeed / myVesselAgent.vesselDynamics.maxSpeed;
        vesselState[1] = myVesselAgent.transform.forward.x;
        vesselState[2] = myVesselAgent.transform.forward.z;
        vesselState[3] = myVesselAgent.vesselDynamics.YawRate / myVesselAgent.vesselDynamics.maxTurnRate;

        // ========== Goal Info (3D) ==========
        float[] goalInfo = new float[3];
        if (myVesselAgent.hasGoal)
        {
            Vector3 directionToGoal = (myVesselAgent.goalPosition - myVesselAgent.transform.position).normalized;
            goalInfo[0] = directionToGoal.x;
            goalInfo[1] = directionToGoal.z;
            goalInfo[2] = Vector3.Distance(myVesselAgent.transform.position, myVesselAgent.goalPosition) / myVesselAgent.radarRange;
        }

        // ========== Fuzzy COLREGs (4D) ==========
        float[] fuzzyCOLREGs = new float[4];
        var allSituations = COLREGsHandler.AnalyzeAllCOLREGsSituations(myVesselAgent, myVesselAgent.radar.GetDetectedVessels());

        if (allSituations.Count > 0)
        {
            float totalRisk = 0f;

            // 모든 상황의 위험도 합산
            foreach (var (situation, risk, vessel, priority) in allSituations)
            {
                fuzzyCOLREGs[(int)situation] += risk;
                totalRisk += risk;
            }

            // 정규화 (fuzzy weights)
            if (totalRisk > 0f)
            {
                for (int i = 0; i < 4; i++)
                    fuzzyCOLREGs[i] /= totalRisk;
            }
        }

        // ========== Legacy Data (for compatibility) ==========
        float[] radarData = null;
        float[] colregsSituation = null;
        float[] dangerLevel = new float[1];
        var (mostDangerousSituation, maxRisk, _) =
            COLREGsHandler.AnalyzeMostDangerousVessel(myVesselAgent, myVesselAgent.radar.GetDetectedVessels());
        dangerLevel[0] = maxRisk;

        return new VesselCommunicationData
        {
            position = transform.position,
            velocity = myVesselAgent.vesselDynamics.Velocity,
            speed = myVesselAgent.vesselDynamics.CurrentSpeed,
            currentCOLREGsSituation = mostDangerousSituation,
            riskLevel = maxRisk,
            timeStamp = Time.time,

            // Legacy (null for now)
            radarData = radarData,
            colregsSituation = colregsSituation,
            dangerLevel = dangerLevel,

            // New data
            vesselState = vesselState,
            goalInfo = goalInfo,
            compressedRadarData = compressedRadar,
            fuzzyCOLREGs = fuzzyCOLREGs
        };
    }

    public VesselCommunicationData GetVesselData()
    {
        return CreateCommunicationData();
    }

    public Dictionary<int, VesselCommunicationData> GetCommunicationData()
    {
        return new Dictionary<int, VesselCommunicationData>(receivedData);
    }
}
