# Unity → Python 데이터 흐름 문서

## 목차
1. [전체 아키텍처](#전체-아키텍처)
2. [Unity C# 데이터 수집](#unity-c-데이터-수집)
3. [ML-Agents 통신 계층](#ml-agents-통신-계층)
4. [Python 데이터 수신](#python-데이터-수신)
5. [Neural Network 처리](#neural-network-처리)
6. [전체 데이터 흐름 다이어그램](#전체-데이터-흐름-다이어그램)
7. [차원 요약 표](#차원-요약-표)

---

## 전체 아키텍처

```
[Unity C#]
   VesselAgent.cs
      ├─ CollectObservations()        → obs[0]: 자신의 상태 (36차원)
      ├─ CollectGoalObservations()    → obs[1]: 목표 정보 (2차원)
      ├─ CollectSpeedObservations()   → obs[2]: 속도 정보 (2차원)
      └─ CollectNeighborObservations()→ obs[3]: 이웃 정보 (144차원 = 4×36)
           ↓
[ML-Agents Communicator]
   gRPC/Socket 통신 (port 5005)
           ↓
[Python main.py]
   decision_steps.obs[0~3] 파싱
      ├─ state: [36]
      ├─ goal: [2]
      ├─ speed: [2]
      └─ neighbor_obs: [144] → reshape → [4, 36]
           ↓
[networks.py]
   CNNPolicy → MessageActor + ControlActor
      ├─ MessageActor: 36×3 → 6차원 latent message
      ├─ Neighbor processing: 각 이웃 36 → 6차원
      ├─ Message aggregation: 자신(6) + 이웃 합(6) = 12차원
      └─ ControlActor: 12 → 2차원 action (rudder, thrust)
           ↓
[Unity C#]
   OnActionReceived(action)
      ├─ action[0]: rudder angle
      └─ action[1]: thrust
```

---

## Unity C# 데이터 수집

### 1. obs[0]: 자신의 상태 (36차원)
**파일**: `Assets/Scripts/Agent/VesselAgent.cs:274-352`

```csharp
public override void CollectObservations(VectorSensor sensor)
{
    // 레이더 스캔 실행
    radar.ScanRadar();

    // 1. 섹터별 레이더 관측값 (8섹터 × 3정보 = 24차원)
    for (int sector = 0; sector < 8; sector++)
    {
        const int samplesPerSector = 45;
        float[] distancesNorm = new float[samplesPerSector];
        int hitCount = 0;

        // 각 섹터의 45도 범위를 1도씩 스캔
        for (int angle = 0; angle < samplesPerSector; angle++)
        {
            float distance = radar.GetDistanceAtAngle(sector * 45 + angle);
            float dNorm = Mathf.Clamp01(distance / radar.radarRange);
            distancesNorm[angle] = dNorm;
            if (dNorm < 1.0f) hitCount++;
        }

        // min 계산
        float minNorm = 1.0f;
        for (int i = 0; i < samplesPerSector; i++)
        {
            if (distancesNorm[i] < minNorm) minNorm = distancesNorm[i];
        }

        // median 계산
        System.Array.Sort(distancesNorm);
        float medianNorm = (samplesPerSector % 2 == 1)
            ? distancesNorm[samplesPerSector / 2]
            : 0.5f * (distancesNorm[samplesPerSector / 2 - 1] + distancesNorm[samplesPerSector / 2]);

        // hit ratio 계산
        float hitRatio = (float)hitCount / samplesPerSector;

        sensor.AddObservation(minNorm);      // 차원 1
        sensor.AddObservation(medianNorm);   // 차원 2
        sensor.AddObservation(hitRatio);     // 차원 3
    }

    // 2. 선박 상태 (4차원)
    sensor.AddObservation(vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed);  // 정규화된 속도
    sensor.AddObservation(transform.forward.x);  // 선수 방향 x
    sensor.AddObservation(transform.forward.z);  // 선수 방향 z
    sensor.AddObservation(vesselDynamics.YawRate / vesselDynamics.maxTurnRate);  // 정규화된 회전 속도

    // 3. 목표 관련 (3차원)
    Vector3 directionToGoal = (goalPosition - transform.position).normalized;
    sensor.AddObservation(directionToGoal.x);  // 목표 방향 x
    sensor.AddObservation(directionToGoal.z);  // 목표 방향 z
    sensor.AddObservation(Vector3.Distance(transform.position, goalPosition) / radarRange);  // 정규화된 거리

    // 4. COLREGs 상황 및 위험도 관측 (5차원)
    var (mostDangerousSituation, maxRisk, dangerousVessel) =
        COLREGsHandler.AnalyzeMostDangerousVessel(this, radar.GetDetectedVessels());

    // COLREGs 상황 one-hot (4차원)
    sensor.AddObservation(mostDangerousSituation == COLREGsHandler.CollisionSituation.HeadOn ? 1.0f : 0.0f);
    sensor.AddObservation(mostDangerousSituation == COLREGsHandler.CollisionSituation.CrossingStandOn ? 1.0f : 0.0f);
    sensor.AddObservation(mostDangerousSituation == COLREGsHandler.CollisionSituation.CrossingGiveWay ? 1.0f : 0.0f);
    sensor.AddObservation(mostDangerousSituation == COLREGsHandler.CollisionSituation.Overtaking ? 1.0f : 0.0f);

    // 위험도 스칼라 (1차원)
    sensor.AddObservation(maxRisk);
}
```

**차원 분해:**
- 레이더 데이터: 8 sectors × 3 (min, median, hit_ratio) = **24차원**
- 선박 상태: speed(1) + forward(2) + yaw_rate(1) = **4차원**
- 목표 정보: direction(2) + distance(1) = **3차원**
- COLREGs 상황: one-hot encoding = **4차원**
- 위험도: scalar = **1차원**
- **총합: 36차원**

---

### 2. obs[1]: 목표 정보 (2차원)
**파일**: `Assets/Scripts/Agent/VesselAgent.cs:357-370`

```csharp
public void CollectGoalObservations(VectorSensor sensor)
{
    if (hasGoal)
    {
        Vector3 directionToGoal = (goalPosition - transform.position).normalized;
        sensor.AddObservation(directionToGoal.x);  // 목표 방향 x
        sensor.AddObservation(directionToGoal.z);  // 목표 방향 z
    }
    else
    {
        sensor.AddObservation(0f);
        sensor.AddObservation(0f);
    }
}
```

**차원 분해:**
- 목표 방향 벡터 (x, z): **2차원**

---

### 3. obs[2]: 속도 정보 (2차원)
**파일**: `Assets/Scripts/Agent/VesselAgent.cs:375-379`

```csharp
public void CollectSpeedObservations(VectorSensor sensor)
{
    sensor.AddObservation(vesselDynamics.CurrentSpeed / vesselDynamics.maxSpeed);  // 정규화된 속도
    sensor.AddObservation(vesselDynamics.RudderAngle / vesselDynamics.maxTurnRate);  // 정규화된 타각
}
```

**차원 분해:**
- 현재 속도 (정규화): **1차원**
- 현재 타각 (정규화): **1차원**
- **총합: 2차원**

---

### 4. obs[3]: 이웃 정보 (144차원)
**파일**: `Assets/Scripts/Agent/VesselAgent.cs:384-458`

```csharp
public void CollectNeighborObservations(VectorSensor sensor)
{
    var communication = GetComponent<VesselCommunication>();
    if (communication != null)
    {
        var commData = communication.GetCommunicationData();

        // 최대 4명의 이웃 순회
        for (int i = 0; i < maxCommunicationPartners; i++)  // maxCommunicationPartners = 4
        {
            if (i < commData.Count)
            {
                var data = commData.Values.ElementAt(i);

                // 이웃의 36차원 상태 전송
                if (data.radarData != null)
                {
                    // 1. 레이더 데이터 (24차원)
                    for (int j = 0; j < 24; j++)
                        sensor.AddObservation(data.radarData[j]);

                    // 2. 선박 상태 (4차원)
                    for (int j = 0; j < 4; j++)
                        sensor.AddObservation(data.vesselState[j]);

                    // 3. 목표 정보 (3차원)
                    for (int j = 0; j < 3; j++)
                        sensor.AddObservation(data.goalInfo[j]);

                    // 4. COLREGs 상황 (4차원)
                    for (int j = 0; j < 4; j++)
                        sensor.AddObservation(data.colregsSituation[j]);

                    // 5. 위험도 (1차원)
                    sensor.AddObservation(data.dangerLevel[0]);
                }
                else
                {
                    // 더미 데이터 (36차원)
                    for (int j = 0; j < 36; j++)
                        sensor.AddObservation(0f);
                }
            }
            else
            {
                // 이웃이 없는 경우 0으로 패딩 (36차원)
                for (int j = 0; j < 36; j++)
                    sensor.AddObservation(0f);
            }
        }
    }
    else
    {
        // 통신 컴포넌트가 없는 경우 전체 더미 데이터
        for (int i = 0; i < maxCommunicationPartners * 36; i++)
            sensor.AddObservation(0f);
    }
}
```

**데이터 구조:**
```
[이웃1의 36차원][이웃2의 36차원][이웃3의 36차원][이웃4의 36차원]
```

**차원 분해:**
- 각 이웃: 36차원 (레이더 24 + 상태 4 + 목표 3 + COLREGs 4 + 위험도 1)
- 최대 이웃 수: 4명
- **총합: 4 × 36 = 144차원**

---

### 이웃 데이터 생성 과정
**파일**: `Assets/Scripts/Scripts/Communication/VesselCommunication.cs:105-202`

```csharp
private VesselCommunicationData CreateCommunicationData()
{
    // 레이더 스캔 실행
    myVesselAgent.radar.ScanRadar();

    // 섹터별 레이더 데이터 수집 (24차원)
    float[] radarData = new float[24];
    int radarIndex = 0;
    for (int sector = 0; sector < 8; sector++)
    {
        // ... min, median, hit_ratio 계산 ...
        radarData[radarIndex++] = minNorm;
        radarData[radarIndex++] = medianNorm;
        radarData[radarIndex++] = hitRatio;
    }

    // 선박 상태 수집 (4차원)
    float[] vesselState = new float[4];
    vesselState[0] = myVesselAgent.vesselDynamics.CurrentSpeed / myVesselAgent.vesselDynamics.maxSpeed;
    vesselState[1] = myVesselAgent.transform.forward.x;
    vesselState[2] = myVesselAgent.transform.forward.z;
    vesselState[3] = myVesselAgent.vesselDynamics.YawRate / myVesselAgent.vesselDynamics.maxTurnRate;

    // 목표 정보 수집 (3차원)
    float[] goalInfo = new float[3];
    if (myVesselAgent.hasGoal)
    {
        Vector3 directionToGoal = (myVesselAgent.goalPosition - myVesselAgent.transform.position).normalized;
        goalInfo[0] = directionToGoal.x;
        goalInfo[1] = directionToGoal.z;
        goalInfo[2] = Vector3.Distance(myVesselAgent.transform.position, myVesselAgent.goalPosition) / myVesselAgent.radarRange;
    }

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
        radarData = radarData,          // 24차원
        vesselState = vesselState,      // 4차원
        goalInfo = goalInfo,            // 3차원
        colregsSituation = colregsSituation,  // 4차원
        dangerLevel = dangerLevel       // 1차원
    };
}
```

---

## ML-Agents 통신 계층

### Unity → Python 전송
ML-Agents Communicator가 gRPC/Socket 통신을 통해 데이터 전송:
- **포트**: 5005 (config.py의 BASE_PORT)
- **프로토콜**: gRPC (Protocol Buffers)
- **전송 주기**: Unity의 `DecisionRequester` 컴포넌트 설정에 따름

### 데이터 구조
```protobuf
ObservationProto {
  obs[0]: float[] (36차원 - 자신의 상태)
  obs[1]: float[] (2차원 - 목표 정보)
  obs[2]: float[] (2차원 - 속도 정보)
  obs[3]: float[] (144차원 - 이웃 정보)
}
```

---

## Python 데이터 수신

### main.py에서 데이터 파싱
**파일**: `Assets/Scripts/Python/main.py:188-230`

```python
# 각 에이전트별 처리
for agent_id in decision_steps.agent_id:
    # obs[0]: 자신의 상태 (36차원)
    state = decision_steps.obs[0][agent_id]

    # obs[1]: 목표 정보 (2차원)
    if len(decision_steps.obs) > 1:
        goal = decision_steps.obs[1][agent_id]
    else:
        goal = np.zeros(2)

    # obs[2]: 속도 정보 (2차원)
    if len(decision_steps.obs) > 2:
        speed = decision_steps.obs[2][agent_id]
    else:
        speed = np.zeros(2)

    # obs[3]: 이웃 정보 (144차원 → 재구성)
    if len(decision_steps.obs) > 3:
        # Unity에서 받은 144차원 평탄 벡터
        neighbor_obs_raw = decision_steps.obs[3][agent_id]  # [144]

        # (4, 36) 형태로 재구성
        neighbor_obs = neighbor_obs_raw.reshape(N_AGENT, -1)  # [4, 36]

        # 유효한 이웃 판별 (0이 아닌 값이 있으면 유효)
        neighbor_mask = torch.tensor(
            [np.any(neighbor_obs[i] != 0) for i in range(N_AGENT)],
            dtype=torch.bool
        ).to(DEVICE)
    else:
        neighbor_obs = np.zeros((N_AGENT, STATE_SIZE))
        neighbor_mask = torch.zeros(N_AGENT, dtype=torch.bool).to(DEVICE)
```

### 프레임 스택 적용
**파일**: `Assets/Scripts/Python/main.py:223-230`

```python
# 프레임 스택: 최근 FRAMES개 관측을 concat
# 현재 구현: 동일 프레임 반복 (간단 버전)
# TODO: 실제로는 에이전트별 버퍼에서 과거 프레임을 가져와야 함
state_stack = np.tile(state, FRAMES)  # [36] → [108] (36 × 3)
state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(DEVICE)
goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(DEVICE)
speed_tensor = torch.FloatTensor(speed).unsqueeze(0).to(DEVICE)
neighbor_obs_tensor = torch.FloatTensor(neighbor_obs).unsqueeze(0).to(DEVICE)
```

**프레임 스택 설명:**
- FRAMES = 3 (config.py)
- state (36차원) × 3 = **108차원**
- 시간적 정보를 포함하여 속도/가속도 학습 가능

---

## Neural Network 처리

### 1. MessageActor: 자신과 이웃의 메시지 생성
**파일**: `Assets/Scripts/Python/networks.py:8-50`

```python
class MessageActor(nn.Module):
    def __init__(self, frames, msg_action_space, max_neighbors):
        super(MessageActor, self).__init__()
        self.frames = frames
        self.max_neighbors = max_neighbors
        self.logstd = nn.Parameter(torch.zeros(msg_action_space))

        # 입력: STATE_SIZE * FRAMES = 36 * 3 = 108차원
        self.act_fc1 = nn.Linear(STATE_SIZE * FRAMES, 256)
        self.act_fc2 = nn.Linear(256+2+2, 128)  # 256 + goal(2) + speed(2)
        self.actor = nn.Linear(128, msg_action_space)  # 128 → 6차원 메시지

    def forward(self, x, goal, speed):
        batch_size = x.shape[0] if len(x.shape) > 2 else 1

        # 108차원 → 256차원
        x = x.view(-1, STATE_SIZE * FRAMES)
        a = F.relu(self.act_fc1(x))

        if batch_size > 1:
            a = a.view(batch_size, -1, 256)
        else:
            a = a.view(1, -1, 256)

        # goal(2) + speed(2) 추가
        a = torch.cat((a, goal, speed), dim=-1)  # 256+2+2=260
        a = F.relu(self.act_fc2(a))  # 260 → 128

        msg = torch.tanh(self.actor(a))  # 128 → 6차원 latent message

        # 가우시안 노이즈 추가
        logstd = self.logstd.expand_as(msg)
        std = torch.exp(logstd)
        msg = torch.normal(msg, std)

        logprob = log_normal_density(msg, msg, std=std, log_std=logstd)
        return msg, logprob, msg
```

**압축 과정:**
```
입력 108차원 (36 × 3 frames)
  ↓ FC1
256차원
  ↓ concat goal(2) + speed(2)
260차원
  ↓ FC2
128차원
  ↓ actor layer
6차원 latent message
```

---

### 2. CNNPolicy: 메시지 집계 및 행동 생성
**파일**: `Assets/Scripts/Python/networks.py:122-194`

```python
def forward(self, x, goal, speed, neighbor_obs=None, neighbor_mask=None):
    # 1. 자신의 메시지 생성
    self_msg, _, _ = self.msg_actor(x, goal, speed)  # [batch, 1, 6]

    # 2. 이웃 메시지 생성
    if neighbor_obs is None or neighbor_mask is None:
        # 빈 이웃 메시지
        batch_size = x.shape[0]
        neighbor_msgs = torch.zeros(batch_size, self.max_neighbors,
                                    self_msg.shape[-1], device=x.device)
        neighbor_mask = torch.zeros(batch_size, self.max_neighbors,
                                   device=x.device, dtype=torch.bool)
    else:
        batch_size = neighbor_obs.shape[0]
        neighbor_msgs = torch.zeros(batch_size, self.max_neighbors,
                                    self_msg.shape[-1], device=x.device)

        # 각 이웃의 observation으로부터 메시지 생성
        for i in range(self.max_neighbors):  # 최대 4명
            valid_neighbors = neighbor_mask[:, i]
            if valid_neighbors.any():
                valid_indices = torch.where(valid_neighbors)[0]
                valid_obs = neighbor_obs[valid_indices, i]  # 이웃 i의 36차원

                # 프레임 스택 (간단 버전: 동일 프레임 반복)
                if valid_obs.dim() == 2:
                    valid_obs = valid_obs.repeat(1, FRAMES)  # 36 → 108

                valid_goal = goal[valid_indices] if len(goal.shape) > 2 else goal
                valid_speed = speed[valid_indices] if len(speed.shape) > 2 else speed

                # 이웃 메시지 생성: 36×3=108 → 6차원
                with torch.no_grad():
                    neighbor_msg, _, _ = self.msg_actor(valid_obs, valid_goal, valid_speed)

                # 생성된 메시지 저장
                for j, idx in enumerate(valid_indices):
                    neighbor_msgs[idx, i] = neighbor_msg[j, 0]

    # 3. 이웃 메시지 집계 (마스크 적용)
    masked_msgs = neighbor_msgs * neighbor_mask.unsqueeze(-1).float()
    neighbor_sum = masked_msgs.sum(dim=1, keepdim=True)  # [batch, 1, 6]

    # 4. 자신과 이웃 메시지 결합
    ctr_input = torch.cat((self_msg, neighbor_sum), 2)  # [batch, 1, 12]

    # 5. 행동 생성
    action, logprob, mean = self.ctr_actor(ctr_input, goal, speed, x)

    # 6. 가치 평가
    x = x.view(-1, STATE_SIZE * FRAMES)
    v = F.relu(self.crt_fc1(x))
    v = v.view(-1, 1, 256)
    v = torch.cat((v, goal, speed), dim=-1)
    v = F.relu(self.crt_fc2(v))
    v = self.critic(v)

    return v, action, logprob, mean
```

**메시지 처리 흐름:**
```
자신의 108차원 → MessageActor → 6차원 self_msg

이웃1의 36차원 → repeat(3) → 108차원 → MessageActor → 6차원
이웃2의 36차원 → repeat(3) → 108차원 → MessageActor → 6차원
이웃3의 36차원 → repeat(3) → 108차원 → MessageActor → 6차원
이웃4의 36차원 → repeat(3) → 108차원 → MessageActor → 6차원
                                          ↓
                                    neighbor_sum (6차원)
                                          ↓
                    self_msg(6) + neighbor_sum(6) = 12차원
                                          ↓
                                    ControlActor
                                          ↓
                                    2차원 action
```

---

### 3. ControlActor: 행동 결정
**파일**: `Assets/Scripts/Python/networks.py:52-103`

```python
class ControlActor(nn.Module):
    def __init__(self, frames, msg_action_space, ctr_action_space, n_agent):
        super(ControlActor, self).__init__()

        # 관측 상태 처리
        self.act_obs_fc1 = nn.Linear(STATE_SIZE * FRAMES, 256)  # 108 → 256
        self.act_obs_fc2 = nn.Linear(256+2+2, 128)  # 260 → 128
        self.act_obs_fc3 = nn.Linear(128, msg_action_space)  # 128 → 6

        # 메시지와 상태 결합하여 행동 생성
        self.act_fc1 = nn.Linear(msg_action_space+msg_action_space, 64)  # 12 → 64
        self.act_fc2 = nn.Linear(64+2+2, 128)  # 68 → 128
        self.mu = nn.Linear(128, ctr_action_space)  # 128 → 2
        self.logstd = nn.Parameter(torch.zeros(ctr_action_space))

    def forward(self, x, goal, speed, y):
        # 관측 상태 처리 (108차원)
        y = y.view(-1, STATE_SIZE * FRAMES)
        a = F.relu(self.act_obs_fc1(y))  # 108 → 256
        a = a.view(-1, 1, 256)
        a = torch.cat((a, goal, speed), dim=-1)  # 260
        a = F.relu(self.act_obs_fc2(a))  # 128
        a = F.relu(self.act_obs_fc3(a))  # 6

        # 메시지 결합 (자신 6 + 이웃 합 6 = 12)
        x = torch.cat((a, x), dim=-1)  # 12
        act = self.act_fc1(x)  # 64
        act = act.view(-1, 1, 64)
        act = torch.cat((act, goal, speed), dim=-1)  # 68
        act = F.tanh(act)
        act = self.act_fc2(act)  # 128
        act = F.tanh(act)
        mean = self.mu(act)  # 2차원 action

        # 가우시안 노이즈 추가
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return action, logprob, mean
```

**행동 생성 과정:**
```
메시지 결합 12차원
  ↓ FC1
64차원
  ↓ concat goal(2) + speed(2)
68차원
  ↓ FC2
128차원
  ↓ mu layer
2차원 action
  ├─ action[0]: rudder angle [-1, 1]
  └─ action[1]: thrust [0, 1]
```

---

## 전체 데이터 흐름 다이어그램

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                  UNITY C#                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  VesselAgent.cs                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ CollectObservations()                                          │         ║
║  │   ├─ Radar scan (360도)                                        │         ║
║  │   ├─ Sector aggregation (8 sectors × 3 metrics)               │         ║
║  │   ├─ Vessel state (speed, heading, yaw rate)                  │         ║
║  │   ├─ Goal info (direction, distance)                          │         ║
║  │   └─ COLREGs analysis (situation, risk)                       │         ║
║  │   → obs[0]: [36 dimensions]                                   │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ CollectGoalObservations()                                      │         ║
║  │   └─ Goal direction vector (x, z)                             │         ║
║  │   → obs[1]: [2 dimensions]                                    │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ CollectSpeedObservations()                                     │         ║
║  │   ├─ Current speed (normalized)                               │         ║
║  │   └─ Rudder angle (normalized)                                │         ║
║  │   → obs[2]: [2 dimensions]                                    │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ CollectNeighborObservations()                                 │         ║
║  │   └─ VesselCommunication.GetCommunicationData()               │         ║
║  │       ├─ Neighbor 1: [36 dimensions]                          │         ║
║  │       ├─ Neighbor 2: [36 dimensions]                          │         ║
║  │       ├─ Neighbor 3: [36 dimensions] or [0×36]                │         ║
║  │       └─ Neighbor 4: [36 dimensions] or [0×36]                │         ║
║  │   → obs[3]: [144 dimensions]                                  │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          ML-Agents Communicator                              ║
║                          gRPC/Socket (port 5005)                             ║
║                                  ↓↓↓                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                               PYTHON main.py                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  decision_steps, terminal_steps = env.get_steps(behavior_name)              ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ state = decision_steps.obs[0][agent_id]      # [36]           │         ║
║  │ goal = decision_steps.obs[1][agent_id]       # [2]            │         ║
║  │ speed = decision_steps.obs[2][agent_id]      # [2]            │         ║
║  │ neighbor_obs_raw = decision_steps.obs[3][agent_id]  # [144]   │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ # 이웃 데이터 재구성                                           │         ║
║  │ neighbor_obs = neighbor_obs_raw.reshape(4, 36)                │         ║
║  │                                                                │         ║
║  │ # 유효한 이웃 판별                                             │         ║
║  │ neighbor_mask = [                                             │         ║
║  │   np.any(neighbor_obs[0] != 0),  # True or False             │         ║
║  │   np.any(neighbor_obs[1] != 0),  # True or False             │         ║
║  │   np.any(neighbor_obs[2] != 0),  # True or False             │         ║
║  │   np.any(neighbor_obs[3] != 0)   # True or False             │         ║
║  │ ]                                                             │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ # 프레임 스택 적용 (시간적 정보)                              │         ║
║  │ state_stack = np.tile(state, FRAMES)  # [36] → [108]         │         ║
║  │ state_tensor = torch.FloatTensor(state_stack)                │         ║
║  │ goal_tensor = torch.FloatTensor(goal)                        │         ║
║  │ speed_tensor = torch.FloatTensor(speed)                      │         ║
║  │ neighbor_obs_tensor = torch.FloatTensor(neighbor_obs)        │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ # 신경망 실행                                                  │         ║
║  │ value, action, logprob, _ = policy(                           │         ║
║  │     state_tensor,          # [1, 108]                         │         ║
║  │     goal_tensor,           # [1, 2]                           │         ║
║  │     speed_tensor,          # [1, 2]                           │         ║
║  │     neighbor_obs_tensor,   # [1, 4, 36]                       │         ║
║  │     neighbor_mask          # [1, 4] (bool)                    │         ║
║  │ )                                                             │         ║
║  │   → action: [1, 1, 2]                                         │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ # Unity로 전송                                                 │         ║
║  │ action_tuple = ActionTuple(continuous=all_actions)            │         ║
║  │ env.set_actions(behavior_name, action_tuple)                  │         ║
║  │ env.step()                                                    │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            PYTHON networks.py                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CNNPolicy.forward()                                                         ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ 1. 자신의 메시지 생성                                          │         ║
║  │    state_tensor [1, 108] ────┐                                │         ║
║  │    goal_tensor [1, 2]  ──────┤                                │         ║
║  │    speed_tensor [1, 2] ──────┴─→ MessageActor                 │         ║
║  │                                      ↓                         │         ║
║  │                                  self_msg [1, 1, 6]            │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ 2. 이웃 메시지 생성 (각 이웃마다)                              │         ║
║  │    for i in range(4):                                         │         ║
║  │        if neighbor_mask[i] == True:                           │         ║
║  │            neighbor_obs[i] [36] ─→ repeat(3) ─→ [108]         │         ║
║  │                 ↓                                             │         ║
║  │            MessageActor                                       │         ║
║  │                 ↓                                             │         ║
║  │            neighbor_msg[i] [6]                                │         ║
║  │                                                               │         ║
║  │    neighbor_msgs: [1, 4, 6]                                   │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ 3. 메시지 집계 (마스크 적용)                                   │         ║
║  │    masked_msgs = neighbor_msgs × neighbor_mask                │         ║
║  │    neighbor_sum = Σ(masked_msgs)  # [1, 1, 6]                │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ 4. 메시지 결합                                                 │         ║
║  │    ctr_input = concat(self_msg, neighbor_sum)                 │         ║
║  │              = [1, 1, 12]                                     │         ║
║  │              = [자신의 6차원 + 이웃 합 6차원]                  │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ 5. 행동 생성                                                   │         ║
║  │    ctr_input [1, 1, 12] ──┐                                   │         ║
║  │    goal [1, 2]  ──────────┤                                   │         ║
║  │    speed [1, 2] ──────────┴─→ ControlActor                    │         ║
║  │                                    ↓                           │         ║
║  │                              action [1, 1, 2]                  │         ║
║  │                                ├─ [0]: rudder angle            │         ║
║  │                                └─ [1]: thrust                  │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ 6. 가치 평가                                                   │         ║
║  │    state_tensor [1, 108] ──┐                                  │         ║
║  │    goal [1, 2]  ────────────┤                                 │         ║
║  │    speed [1, 2] ────────────┴─→ Critic Network                │         ║
║  │                                    ↓                           │         ║
║  │                                value [1, 1, 1]                 │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          ML-Agents Communicator                              ║
║                          gRPC/Socket (port 5005)                             ║
║                                  ↑↑↑                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                  UNITY C#                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  VesselAgent.cs                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ OnActionReceived(ActionBuffers actions)                        │         ║
║  │   ├─ rudder = actions.ContinuousActions[0]  # [-1, 1]         │         ║
║  │   └─ thrust = actions.ContinuousActions[1]  # [0, 1]          │         ║
║  │                                                                │         ║
║  │ VesselDynamics.SetRudderAngle(rudder × maxTurnRate)           │         ║
║  │ VesselDynamics.SetTargetSpeed(thrust × maxSpeed)              │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
║  VesselDynamics.cs                                                           ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │ UpdateDynamics()                                               │         ║
║  │   ├─ 속도 업데이트 (가속/감속)                                 │         ║
║  │   ├─ 타각 효과 계산 (속도에 비례)                              │         ║
║  │   ├─ 회전 적용                                                 │         ║
║  │   └─ 물리 시뮬레이션 (Rigidbody)                               │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 차원 요약 표

### Unity → Python 전송

| Observation | 이름 | 차원 | 설명 | 범위 |
|------------|------|------|------|------|
| **obs[0]** | **자신의 상태** | **36** | **전체 센서 데이터** | **[0, 1] or [-1, 1]** |
| - | 레이더 (8 sectors × 3) | 24 | min, median, hit_ratio | [0, 1] |
| - | 선박 상태 | 4 | speed, heading(x,z), yaw_rate | [0, 1] or [-1, 1] |
| - | 목표 정보 | 3 | direction(x,z), distance | [-1, 1] or [0, 1] |
| - | COLREGs 상황 | 4 | one-hot encoding | {0, 1} |
| - | 위험도 | 1 | risk level | [0, 1] |
| **obs[1]** | **목표 정보** | **2** | **목표 방향 벡터** | **[-1, 1]** |
| **obs[2]** | **속도 정보** | **2** | **현재 속도 + 타각** | **[0, 1]** |
| **obs[3]** | **이웃 정보** | **144** | **4 neighbors × 36차원** | **[0, 1] or [-1, 1]** |
| - | 이웃 1 | 36 | 이웃의 전체 상태 | [0, 1] or [-1, 1] |
| - | 이웃 2 | 36 | 이웃의 전체 상태 | [0, 1] or [-1, 1] |
| - | 이웃 3 | 36 | 이웃의 전체 상태 (또는 0 패딩) | [0, 1] or [-1, 1] |
| - | 이웃 4 | 36 | 이웃의 전체 상태 (또는 0 패딩) | [0, 1] or [-1, 1] |
| **총합** | | **184** | **36 + 2 + 2 + 144** | |

### Python 신경망 처리

| Layer | Input | Output | 설명 |
|-------|-------|--------|------|
| **Frame Stack** | 36 | 108 | state × 3 frames |
| **MessageActor** | 108 + 2 + 2 | 6 | latent message 생성 |
| **Neighbor Processing** | 4 × (36×3) | 4 × 6 | 각 이웃 메시지 생성 |
| **Message Aggregation** | 4 × 6 | 6 | 이웃 메시지 합계 |
| **Message Concatenation** | 6 + 6 | 12 | 자신 + 이웃 합 |
| **ControlActor** | 12 + 2 + 2 | 2 | action 생성 |
| **Critic** | 108 + 2 + 2 | 1 | value 평가 |

### Python → Unity 전송

| Action | 차원 | 설명 | 범위 |
|--------|------|------|------|
| **action[0]** | 1 | Rudder angle | [-1, 1] |
| **action[1]** | 1 | Thrust | [0, 1] |
| **총합** | **2** | **연속 행동 공간** | |

---

## 핵심 요약

### 1. Unity에서 데이터 수집
- **4개의 VectorSensor**를 사용하여 구조화된 관측 데이터 수집
- **총 184차원** (자신 36 + 목표 2 + 속도 2 + 이웃 144)
- **VesselCommunication**을 통해 이웃 데이터 교환

### 2. ML-Agents 통신
- **gRPC/Socket** 통신 (port 5005)
- **Protocol Buffers** 직렬화
- Unity → Python: observations, rewards, done flags
- Python → Unity: actions

### 3. Python에서 데이터 파싱
- `decision_steps.obs[0~3]`로 각 관측 타입 분리
- 이웃 데이터는 **평탄 벡터 (144차원) → 재구성 (4, 36)**
- **프레임 스택** 적용으로 시간적 정보 추가 (36 → 108)
- **Mask 생성**으로 유효한 이웃만 처리

### 4. 신경망 처리
- **MessageActor**: 각 선박의 36×3=108차원 → 6차원 latent message로 압축
- **Message Aggregation**: 자신의 메시지 + 이웃 메시지 합 = 12차원
- **ControlActor**: 12차원 메시지 → 2차원 action (rudder, thrust)
- **Critic**: 상태 평가 → 1차원 value

### 5. Unity로 행동 반환
- **2차원 연속 행동**: rudder angle, thrust
- **VesselDynamics**가 물리 시뮬레이션 수행

---

## 참고 사항

### 프레임 스택의 중요성
현재 구현은 간단히 동일 프레임을 3번 반복하지만, 실제로는 **과거 3개 프레임**을 저장하여 사용하는 것이 권장됩니다:
```python
# 현재 구현 (간단 버전)
state_stack = np.tile(state, FRAMES)  # [36, 36, 36]

# 권장 구현
state_stack = np.concatenate([state_t-2, state_t-1, state_t])  # [36, 36, 36]
```

프레임 스택을 통해 신경망은:
- **속도 추정**: 위치 변화 감지
- **가속도 추정**: 속도 변화 감지
- **궤적 예측**: 미래 경로 예측

### 통신 범위와 이웃 수
- **통신 범위**: 100m (VesselCommunication.communicationRange)
- **최대 이웃 수**: 4명 (maxCommunicationPartners)
- **통신 주기**: 0.1초 (communicationInterval)
- 이웃이 4명 미만일 경우 자동으로 0 패딩

### 차원 확장 가능성
현재 시스템은 쉽게 확장 가능:
- **레이더 섹터 증가**: 8 → 16 섹터
- **이웃 수 증가**: 4 → 8명
- **추가 센서**: 속도계, GPS, AIS 등
- **추가 상태**: 연료, 손상도, 기상 조건 등

---

**작성일**: 2025-11-08
**버전**: 1.0
**프로젝트**: Vessel_MLAgent
**작성자**: Claude Code
