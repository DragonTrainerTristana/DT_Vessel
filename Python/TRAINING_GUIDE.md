# 🚢 Vessel ML-Agent 학습 가이드

## ✅ Unity Editor에서 학습 가능 여부

**YES!** Unity Editor에서 Build 없이 바로 학습 가능합니다.

`main.py`가 `UnityEnvironment(file_name=None)`으로 설정되어 있어서 Editor 연결 모드입니다.

---

## 📋 학습 전 체크리스트

### 1. Unity Inspector 설정 확인

**각 VesselAgent GameObject에서:**

#### Behavior Parameters 컴포넌트:
- ✅ **Behavior Name**: "VesselAgent" (또는 일관된 이름)
- ✅ **Vector Observation Space Size**: **184**
- ✅ **Vector Observation Space Type**: Default
- ✅ **Actions**:
  - Continuous Actions: **2** (rudder, thrust)
  - Discrete Branches: 0
- ✅ **Behavior Type**: **Default** (Python과 통신)
- ✅ **Model**: None (학습 중이므로 비워둠)

#### VesselAgent 컴포넌트:
- ✅ **Arrival Reward**: 15.0
- ✅ **Goal Distance Coef**: 2.5
- ✅ **Collision Penalty**: -15.0
- ✅ **Rotation Penalty**: -0.1
- ✅ **Max Angular Velocity**: 10.0
- ✅ **Goal Reached Distance**: 5.0
- ✅ **Colregs Reward Coef**: 0.3
- ✅ **Radar Range**: 100
- ✅ **Max Communication Partners**: 4

### 2. Python 환경 확인

```bash
cd Assets/Scripts/Python

# 필요한 패키지 확인
pip3 list | grep mlagents
pip3 list | grep torch
pip3 list | grep tensorboard
```

필요한 패키지:
- `mlagents==1.0.0` (또는 최신 버전)
- `torch>=1.8.0`
- `tensorboard`
- `numpy`

---

## 🚀 학습 시작 방법

### Step 1: Python 학습 스크립트 실행

터미널에서:
```bash
cd Assets/Scripts/Python
python3 main.py
```

출력 예시:
```
================================================================================
🚢 Vessel ML-Agent Training Start
================================================================================
Device: cpu (또는 cuda)
Learning Rate: 0.0001
Batch Size: 2048
PPO Epochs: 4
Update Interval: 500 steps
Max Episodes: 10000
Max Steps per Episode: 500
================================================================================

Waiting for Unity environment connection...
```

### Step 2: Unity Editor Play 버튼 클릭

Python이 대기 중인 상태에서 Unity Editor의 **▶ Play** 버튼을 클릭합니다.

연결되면 터미널에 다음과 같이 표시됩니다:
```
✅ Policy Network Loaded: 133459 parameters

================================================================================
📋 Episode 0/10000 Start (Total Steps: 0)
================================================================================
```

---

## 📊 학습이 잘 되는지 파악하는 방법

### 1. 실시간 콘솔 출력

#### 에피소드 시작:
```
================================================================================
📋 Episode 5/10000 Start (Total Steps: 2500)
================================================================================
```

#### 진행 상황 (자동으로 10회 출력):
```
  Step 50/500 | Active Agents: 4 | Avg Reward: -0.032 | Collisions: 1 | Success: 0
  Step 100/500 | Active Agents: 3 | Avg Reward: 0.015 | Collisions: 2 | Success: 0
  ...
```

**좋은 신호:**
- ✅ Active Agents가 오래 유지 (빨리 죽지 않음)
- ✅ Avg Reward가 점차 증가
- ✅ Collisions가 감소

**나쁜 신호:**
- ❌ Active Agents가 매우 빨리 0이 됨 (모두 충돌)
- ❌ Avg Reward가 계속 음수
- ❌ Collisions가 증가

#### PPO 업데이트:
```
  🔄 Running PPO update at step 500...
  PPO Update: policy_loss=0.3845, value_loss=0.0652, entropy=1.4163, kl=0.0012
```

**좋은 신호:**
- ✅ policy_loss가 점차 감소
- ✅ value_loss가 점차 감소
- ✅ kl이 0.01 미만 (너무 크면 학습 불안정)
- ✅ entropy가 천천히 감소 (탐험→활용 전환)

**나쁜 신호:**
- ❌ loss가 발산 (NaN, inf)
- ❌ kl이 0.1 이상 (정책 변화가 너무 큼)
- ❌ entropy가 너무 빨리 0에 근접 (조기 수렴)

#### 에피소드 종료:
```
────────────────────────────────────────────────────────────────────────────────
📊 Episode 5 Summary:
────────────────────────────────────────────────────────────────────────────────
  Agents: 4
  Total Steps: 345
  Average Reward: 0.234
  Total Reward: 0.936
  Collisions: 2 (Rate: 0.58%)
  Success: 2 (Rate: 50.00%)

  📈 Last 5 Episodes Average:
    Reward: 0.187 (±0.045)
    Collision Rate: 1.23%
    Success Rate: 40.00%
    ✅ Trend: Improving (+0.089)
────────────────────────────────────────────────────────────────────────────────
```

**학습 추세 판정:**
- ✅ **Improving**: Reward가 증가 중 (좋음!)
- ➡️  **Stable**: Reward가 안정화 (수렴 중)
- ⚠️  **Declining**: Reward가 감소 중 (문제 발생)

### 2. TensorBoard로 시각화

별도 터미널에서:
```bash
cd Assets/Scripts/Python
tensorboard --logdir=models/logs
```

브라우저에서 `http://localhost:6006` 접속

**확인할 그래프:**
- **Reward/Episode**: 상승 추세여야 함
- **Collision/Rate**: 하락 추세여야 함
- **Success/Rate**: 상승 추세여야 함
- **Loss/Policy**: 하락 후 안정화
- **Loss/Value**: 하락 후 안정화
- **PPO/ApproxKL**: 0.01 미만 유지

### 3. CSV 로그 분석

`Assets/Scripts/Python/models/csv_logs/` 폴더에 저장됩니다:
- `episode_logs.csv`: 에피소드별 통계
- `training_logs.csv`: PPO 학습 통계
- `step_logs.csv`: 스텝별 상세 로그 (디버깅용)

Excel이나 Python pandas로 분석 가능합니다.

---

## 🎯 학습 성공 지표

### 초기 단계 (Episode 0-100):
- Reward: -5 ~ 0
- Collision Rate: 50-80%
- Success Rate: 0-5%
- **목표**: 충돌을 피우는 법 학습

### 중기 단계 (Episode 100-1000):
- Reward: 0 ~ 5
- Collision Rate: 20-50%
- Success Rate: 5-20%
- **목표**: 목표 지점으로 이동하는 법 학습

### 후기 단계 (Episode 1000+):
- Reward: 5 ~ 10
- Collision Rate: 5-20%
- Success Rate: 20-50%
- **목표**: COLREGs 준수 및 효율적 항해

### 수렴 (잘 학습된 상태):
- Reward: 10+
- Collision Rate: <5%
- Success Rate: >50%
- **목표**: 안정적인 성능 유지

---

## ⚠️ 문제 해결

### 문제 1: Unity가 Python과 연결되지 않음

**증상:**
```
Waiting for Unity environment connection...
(계속 대기)
```

**해결:**
1. Unity Editor에서 Play 버튼을 눌렀는지 확인
2. Unity Console에 에러가 있는지 확인
3. `config.py`의 `BASE_PORT` 변경 (5004 → 5005)
4. Python 재시작 후 Unity Play

### 문제 2: Observation 차원 오류

**증상:**
```
RuntimeError: Expecting 184 observations but got 36
```

**해결:**
1. Unity Inspector → Behavior Parameters → Space Size = **184** 확인
2. VesselCommunication 컴포넌트가 VesselAgent에 붙어있는지 확인

### 문제 3: 학습이 전혀 안 됨 (Reward가 계속 음수)

**증상:**
- Reward가 -10 이하로 고정
- 모든 에이전트가 즉시 충돌

**해결:**
1. Reward 파라미터 조정:
   - `collisionPenalty`를 -15 → -10으로 완화
   - `arrivalReward`를 15 → 20으로 증가
2. Learning Rate 조정:
   - `config.py`의 `LEARNING_RATE`를 0.0001 → 0.0003으로 증가
3. 초기 속도 확인:
   - VesselAgent.cs의 `OnEpisodeBegin()`에서 초기 속도가 너무 빠른지 확인

### 문제 4: 학습 중 Unity가 느려짐

**증상:**
- Unity가 매우 느리게 실행됨
- FPS가 10 이하

**해결:**
1. Time Scale 조정:
   - `config.py`의 `TIME_SCALE`을 20으로 설정 (이미 설정됨)
2. 카메라 비활성화:
   - Scene View 카메라를 끄기 (Game View만 사용)
3. VSync 비활성화:
   - Edit → Project Settings → Quality → VSync Count = Don't Sync

### 문제 5: GPU 메모리 부족 (CUDA out of memory)

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결:**
1. `config.py`에서 `DEVICE = 'cpu'`로 변경
2. 또는 `BATCH_SIZE`를 2048 → 1024로 감소

---

## 💡 학습 팁

### 1. Time Scale 활용
- 학습 속도를 높이려면 `TIME_SCALE = 20` (현재 설정)
- 시각적으로 확인하려면 `TIME_SCALE = 1` (실시간)

### 2. 체크포인트 활용
- 모델은 `SAVE_INTERVAL`마다 자동 저장됩니다 (기본 100 에피소드)
- 저장 위치: `Assets/Scripts/Python/models/`
- 학습 중단 후 재개하려면 `main.py`에서 모델 로드 코드 추가 필요

### 3. 보상 함수 튜닝
- 학습이 잘 안 되면 `VesselAgent.cs`의 Reward Parameters 조정
- 가장 중요한 행동에 높은 보상 부여

### 4. 병렬 학습
- 여러 Unity 인스턴스로 병렬 학습 가능
- `WORKER_ID`를 다르게 설정 (0, 1, 2, ...)

---

## 📁 파일 구조

```
Assets/Scripts/Python/
├── main.py              # 메인 학습 스크립트
├── networks.py          # 신경망 아키텍처
├── config.py            # 하이퍼파라미터 설정
├── memory.py            # 경험 리플레이 버퍼
├── functions.py         # 유틸리티 함수
├── frame_stack.py       # 프레임 스태킹
├── models/              # 학습된 모델 저장
│   ├── logs/            # TensorBoard 로그
│   └── csv_logs/        # CSV 로그
└── TRAINING_GUIDE.md    # 이 파일
```

---

## 🎓 다음 단계

1. **초기 학습 (100 에피소드)**:
   - 기본 설정으로 학습 시작
   - TensorBoard와 콘솔로 모니터링
   - 학습 추세 확인

2. **하이퍼파라미터 튜닝**:
   - 학습이 잘 안 되면 `config.py` 조정
   - Learning Rate, Batch Size, PPO Epochs 변경

3. **보상 함수 개선**:
   - 원하는 행동을 유도하도록 보상 조정
   - COLREGs 준수 보상 강화

4. **모델 평가**:
   - 학습된 모델을 Unity에 배포
   - Behavior Type을 "Inference Only"로 변경
   - ONNX 모델을 Behavior Parameters의 Model에 할당

---

**행운을 빕니다! 🚢**
