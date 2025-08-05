using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;
using System.Collections;

public class AgentStatusUI : MonoBehaviour
{
    public GameObject vesselParent;  // Inspector에서 VesselParent 할당

    [System.Serializable]
    public class AgentUIElements
    {
        public TextMeshProUGUI agentLabel;
        public TextMeshProUGUI goalDistance;
        public TextMeshProUGUI thrustValue;
        public TextMeshProUGUI rudderValue;
    }
    
    private List<AgentUIElements> agentUIList = new List<AgentUIElements>();
    private VesselAgent[] agents;

    void Awake()
    {
        // TMP Essentials가 없으면 자동으로 추가
        if (TMP_Settings.instance == null)
        {
            //Debug.LogWarning("TextMesh Pro Essentials가 자동으로 추가됩니다.");
            TMP_Settings.LoadDefaultSettings();
        }
    }

    void Start()
    {
        //Debug.Log("AgentStatusUI Start 시작");

        // 1. Canvas 및 기본 컴포넌트 설정
        if (!TryGetComponent<Canvas>(out var canvas))
        {
            canvas = gameObject.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 100;  // UI가 항상 위에 보이도록
            //Debug.Log("Canvas 생성됨");
        }

        if (!TryGetComponent<CanvasScaler>(out var scaler))
        {
            scaler = gameObject.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
        }

        if (!TryGetComponent<GraphicRaycaster>(out var raycaster))
        {
            raycaster = gameObject.AddComponent<GraphicRaycaster>();
        }

        // 2. 패널 생성
        GameObject panelObj = new GameObject("StatusPanel");
        RectTransform panelRect = panelObj.AddComponent<RectTransform>();
        panelObj.transform.SetParent(transform, false);
        
        panelRect.anchorMin = new Vector2(1, 1);
        panelRect.anchorMax = new Vector2(1, 1);
        panelRect.pivot = new Vector2(1, 1);
        panelRect.anchoredPosition = new Vector2(-10, -10);

        StartCoroutine(InitializeUIWithDelay(panelRect));
    }

    private IEnumerator InitializeUIWithDelay(RectTransform panelRect)
    {
        yield return new WaitForSeconds(1f);  // 1초 대기

        if (vesselParent != null)
        {
            // false = 활성화된 오브젝트만 찾기
            agents = vesselParent.GetComponentsInChildren<VesselAgent>(false);
            //Debug.Log($"vesselParent에서 찾은 활성 에이전트 수: {agents.Length}");

            foreach (var agent in agents)
            {
                //Debug.Log($"Found agent: {agent.gameObject.name}");
            }

            if (agents.Length > 0)
            {
                CreateUIElements(panelRect, agents.Length);
            }
        }
        else
        {
            //Debug.LogError("vesselParent가 할당되지 않았습니다!");
        }
    }

    private void CreateUIElements(RectTransform parent, int agentCount)
    {
        for (int i = 0; i < agentCount; i++)
        {
            // 에이전트별 컨테이너
            GameObject container = new GameObject($"AgentStatus_{i + 1}");
            RectTransform containerRect = container.AddComponent<RectTransform>();
            container.transform.SetParent(parent, false);
            
            containerRect.sizeDelta = new Vector2(200, 100);
            containerRect.anchorMin = new Vector2(1, 1);
            containerRect.anchorMax = new Vector2(1, 1);
            containerRect.pivot = new Vector2(1, 1);
            containerRect.anchoredPosition = new Vector2(0, -i * 110);

            // UI 요소들 생성
            AgentUIElements elements = new AgentUIElements
            {
                agentLabel = CreateText(container, $"Agent {i + 1}", 0),
                goalDistance = CreateText(container, "Distance: 0m", -25),
                thrustValue = CreateText(container, "Thrust: 0", -50),
                rudderValue = CreateText(container, "Rudder: 0°", -75)
            };

            agentUIList.Add(elements);
        }
    }

    private TextMeshProUGUI CreateText(GameObject parent, string defaultText, float yOffset)
    {
        GameObject textObj = new GameObject("Text");
        RectTransform rect = textObj.AddComponent<RectTransform>();
        textObj.transform.SetParent(parent.transform, false);
        
        TextMeshProUGUI tmp = textObj.AddComponent<TextMeshProUGUI>();
        
        // TMP 기본 설정
        tmp.font = TMP_Settings.defaultFontAsset;
        tmp.text = defaultText;
        tmp.fontSize = 16;
        tmp.alignment = TextAlignmentOptions.Left;
        tmp.color = Color.white;
        
        // 위치 설정
        rect.anchorMin = new Vector2(0, 1);
        rect.anchorMax = new Vector2(1, 1);
        rect.pivot = new Vector2(0.5f, 1);
        rect.anchoredPosition = new Vector2(0, yOffset);
        rect.sizeDelta = new Vector2(0, 20);
        
        return tmp;
    }

    void Update()
    {
        if (agents == null || agents.Length == 0)
        {
            //Debug.LogWarning("에이전트가 없습니다");
            return;
        }

        for (int i = 0; i < agents.Length; i++)
        {
            if (agents[i] != null)
            {
                float distance = Vector3.Distance(agents[i].transform.position, agents[i].goalPosition);
                
                if (agentUIList[i] == null)
                {
                    //Debug.LogError($"Agent {i}의 UI 요소가 없습니다");
                    continue;
                }

                agentUIList[i].goalDistance.text = $"Goal Distance: {distance:F1}m";
                agentUIList[i].thrustValue.text = $"Thrust: {agents[i].vesselDynamics.CurrentSpeed:F2}";
                agentUIList[i].rudderValue.text = $"Rudder: {agents[i].vesselDynamics.RudderAngle:F1}°";
            }
        }
    }
} 