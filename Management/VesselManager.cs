using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class VesselManager : MonoBehaviour
{
    public List<Transform> spawnPoints = new List<Transform>();
    public GameObject vesselPrefab;
    public int vesselCount = 4;
    public float minGoalDistance = 50f;
    public GameObject vesselParent;

    private List<GameObject> vessels = new List<GameObject>();
    private List<VesselAgent> vesselAgents = new List<VesselAgent>();
    private HashSet<int> usedSpawnIndices = new HashSet<int>();
    private Dictionary<GameObject, Transform> vesselGoals = new Dictionary<GameObject, Transform>();
    private Dictionary<GameObject, int> vesselSpawnIndices = new Dictionary<GameObject, int>();

    public List<VesselAgent> GetAllVesselAgents()
    {
        return vesselAgents;
    }

    void Start()
    {
        InitializeVessels();
    }

    public void InitializeVessels()
    {
        foreach (var vessel in vessels)
        {
            if (vessel != null) Destroy(vessel);
        }

        vessels.Clear();
        vesselAgents.Clear();
        usedSpawnIndices.Clear();
        vesselGoals.Clear();
        vesselSpawnIndices.Clear();

        if (spawnPoints.Count < vesselCount) return;

        for (int i = 0; i < vesselCount; i++)
        {
            SpawnVessel();
        }
    }

    private GameObject SpawnVessel()
    {
        int spawnIndex = GetRandomUnusedSpawnIndex();
        if (spawnIndex == -1) return null;

        Transform spawnPoint = spawnPoints[spawnIndex];
        usedSpawnIndices.Add(spawnIndex);

        GameObject vessel = Instantiate(vesselPrefab, spawnPoint.position, spawnPoint.rotation, vesselParent.transform);
        vessels.Add(vessel);
        vessel.name = $"Vessel_{vessels.Count}";

        VesselDynamics dynamics = vessel.GetComponent<VesselDynamics>();
        if (dynamics == null) dynamics = vessel.AddComponent<VesselDynamics>();

        VesselAgent agent = vessel.GetComponent<VesselAgent>();
        if (agent != null) vesselAgents.Add(agent);

        vesselSpawnIndices[vessel] = spawnIndex;
        AssignGoalForVessel(vessel, spawnPoint);

        return vessel;
    }

    private int GetRandomUnusedSpawnIndex()
    {
        if (usedSpawnIndices.Count >= spawnPoints.Count) return -1;

        List<int> availableIndices = new List<int>();
        for (int i = 0; i < spawnPoints.Count; i++)
        {
            if (!usedSpawnIndices.Contains(i)) availableIndices.Add(i);
        }

        int randomIndex = Random.Range(0, availableIndices.Count);
        return availableIndices[randomIndex];
    }

    private void AssignGoalForVessel(GameObject vessel, Transform spawnPoint)
    {
        List<Transform> validGoalPoints = new List<Transform>();

        foreach (Transform point in spawnPoints)
        {
            float distance = Vector3.Distance(spawnPoint.position, point.position);
            if (distance >= minGoalDistance) validGoalPoints.Add(point);
        }

        if (validGoalPoints.Count == 0)
        {
            validGoalPoints = new List<Transform>(spawnPoints);
        }

        int goalIndex = Random.Range(0, validGoalPoints.Count);
        Transform goalPoint = validGoalPoints[goalIndex];

        vesselGoals[vessel] = goalPoint;

        VesselAgent agent = vessel.GetComponent<VesselAgent>();
        if (agent != null)
        {
            agent.SetGoal(goalPoint.position);
            agent.goalPointName = goalPoint.name;
            agent.goalPointIndex = goalIndex;
        }
    }

    void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;

        foreach (var vessel in vessels)
        {
            if (vessel != null && vesselGoals.ContainsKey(vessel))
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawLine(vessel.transform.position, vesselGoals[vessel].position);

                Gizmos.color = Color.red;
                Gizmos.DrawSphere(vesselGoals[vessel].position, 1.5f);
            }
        }
    }

    public void ChangeVesselCount(int newCount)
    {
        vesselCount = Mathf.Clamp(newCount, 1, spawnPoints.Count);
        InitializeVessels();
    }

    public void RespawnVessel(GameObject vessel)
    {
        if (vessel == null || !vessels.Contains(vessel)) return;

        if (vesselSpawnIndices.ContainsKey(vessel))
        {
            int oldIndex = vesselSpawnIndices[vessel];
            usedSpawnIndices.Remove(oldIndex);
        }

        if (vesselGoals.ContainsKey(vessel))
        {
            vesselGoals.Remove(vessel);
        }

        int spawnIndex = GetRandomUnusedSpawnIndex();
        if (spawnIndex == -1)
        {
            spawnIndex = Random.Range(0, spawnPoints.Count);
        }

        Transform spawnPoint = spawnPoints[spawnIndex];
        usedSpawnIndices.Add(spawnIndex);
        vesselSpawnIndices[vessel] = spawnIndex;

        vessel.transform.position = spawnPoint.position;
        vessel.transform.rotation = spawnPoint.rotation;

        Rigidbody rb = vessel.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        VesselDynamics dynamics = vessel.GetComponent<VesselDynamics>();
        if (dynamics != null) dynamics.ResetState();

        AssignGoalForVessel(vessel, spawnPoint);
    }
}
