# Vessel Multi-Agent RL — 통신 협력 항해 연구

## 프로젝트 (간단)
다중 선박이 강화학습으로 목표까지 항해하면서 **COLREGs(해양 충돌규정)**를 지키고 서로 충돌을 피한다.
핵심 연구 질문: **에이전트 간 통신(latent 메시지 교환)이 협력적 충돌 회피를 개선하는가?**
- **환경**: Unity ML-Agents (C#) — `Assets/Scripts/Agent`, `Navigation`, `Management`
- **학습**: PyTorch PPO (Python) — `Assets/Scripts/Python` (단일 source of truth: `config.py`)
- shared policy, 에이전트들이 `MSG_DIM`차원 latent 메시지를 범위 내(COMM_RANGE) 이웃과 교환

---

## 🎯 연구 목표 (Hypotheses — *전제 아님, 정직하게 입증할 가설*)

> ⚠️ 통신은 **자동으로 좋아지지 않는다**(학습돼야 하고 실패할 수 있음). 아래는 *증명 대상*이며,
> "통신이 이기도록 강제"하는 게 아니라 **공정하게 입증**하는 것이 목표다.

### H1. 통신 ON > 통신 OFF (reward · fuel_consumption ↓ · trajectory 간소화)
통신 ON이 OFF보다 **(a) 보상, (b) 연료 소비↓, (c) 궤적 간소화(멀리서 부드러운 조기 회피)**에서 우월함을 보인다.

**두 단계로 나눠서 (둘은 다른 주장):**
- **H1a (불변식·반드시 성립): comm-ON ≥ comm-OFF — 통신은 *절대 더 나빠지면 안 된다.*** 근거: 정보가치 비음수 정리 — 에이전트는 메시지를 무시할 수 있다(ControlActor fc2의 메시지 가중치=0 → comm-OFF와 동일). **comm-ON이 comm-OFF보다 *나쁘면 = 구현/최적화 버그* (반드시 고칠 것).** 현재 의심 원인: SUM 집계(others_msg ±4 큰 노이즈→무시 어려움), 약한 메시지 정규화(MSG_L2=0.001), 메시지 LR 3배, critic이 노이즈 메시지 조건화. → 수정: 강한 정규화/mean·정규화 집계/LR 1배로 comm-ON이 최소 comm-OFF는 따라잡게.
- **H1b (목표·조건부): comm-ON > comm-OFF — 통신이 *더 우월.*** 이건 *자동 아님* — task가 통신을 *필요*로 할 때만(국소 인지 부족/협응 모호). 완전관측에선 comm-ON = comm-OFF(같음)가 최선.
- **순서: H1a 먼저(안 나빠지게) → H1b(도움되게 regime 부여).**

**정직하게 입증하는 법 (rigging 방지 — 필수):**
1. **Ground-truth로 평가**: shaped reward가 아니라 **실제 충돌률·연료·궤적·COLREGs 준수** (`VESSEL_METRIC_LOG`, `VESSEL_OUTCOME_LOG`).
2. **공정한 baseline**: comm-ON은 *불구화 안 된 최선의 comm-OFF*를 이겨야 함. comm-OFF를 "못 보는 위협"으로 페널티 줘 인위적으로 나쁘게 만들면 = **rigging(무효)**. → 3-way 비교: `baseline-OFF` vs `extended-ON`, extended-ON이 *baseline-OFF*를 ground-truth로 이겨야 진짜.
3. **통신이 필요한 조건에서만**: 완전관측 COLREGs는 *기하로 최적행동이 결정*돼 통신 잉여. 통신은 **국소 인지 부족**(레이더 밖/가림) 또는 **협응 모호**(다물체)에서만 가치 → 그 regime에서 시험.

### H2. latent 차원 ↑ → 통신 질 ↑
메시지 차원 `MSG_DIM`이 클수록 정보량·협응 질이 올라간다.
- ⚠️ **H1이 성립한 *후에만* 의미.** 통신이 실제로 안 쓰이면 차원 수는 무의미 (실제로 MSG_DIM 2~12가 동일 수렴한 전례 있음 — 통신이 안 쓰여서). 통신이 가치를 가질 때 차원 스케일링 측정 (수확체감·과적합·불안정 주의).

---

## 현재 상태 & 핵심 발견 (2026-06-02)
- **★★H1a 통신 게이트 fix = 검증됨 (3-seed×1M, ground-truth)**: naive comm(fix前)은 수렴 vColl이 OFF의 ~2배(H1a 위배). 진단 결과 근본원인 = **"메시지 무시"가 init-time 성질일 뿐 학습 평형이 아님**(zero-init은 출발선만 맞추고 약한 L2로는 메시지가 0에서 자람). 수정: **학습형 게이트** `others_msg×sigmoid(msg_gate)` (ControlActor·Critic, gate 초기 −3=닫힘) + 게이트 개방 페널티(`MSG_GATE_COEF`/`VESSEL_MSG_GATE_L2`=0.02) + 생산측(`MessageActor.msg_out`) zero-init + `VESSEL_MSG_L2=0.01`. 전부 ON만(OFF는 others_msg≡0이라 불변=anti-rigging). **결과: comm-ON(fixed)이 OFF를 seed-paired Pareto로 이김 — goal 88.2 vs 86.0%, vColl 3.4 vs 5.7%, LATE 0/3 vs 3/3.** H1a 회복 + H1b 양성(단 3-seed, comm은 붕괴 *완화*지 제거 아님, 더 많은 seed 필요).
- **★★OFF baseline 자체가 깊은 수렴(1M)에서 rush>avoid 붕괴**: 보상재설계(0601)가 mid-training(~7천ep)엔 vColl ~1%로 좋아 보였으나, 1M 수렴서 **3/3 LATE, vColl 5.7%로 회귀**(goal 86%로 최대화·timeout 7%로 바닥나며 직선 돌진, straight 0.89). **보상재설계는 깊은 수렴서 충돌회귀를 못 막음 = 다음 디버그 1순위(통신 아닌 보상 문제, ON·OFF 공통).** "mid-training 좋아도 채택 금지" 재확인.
- **★레이더 장님 버그가 충돌의 진짜 원인이었음** (`VesselAgent.cs`의 빈 LayerMask가 radar 기본값 `~0`을 덮어씀 → 감지 0). 수정(`!=0 ? : ~0`) 후 **배충돌 34% → 4%.** 회피는 **egocentric ARPA 인지**로 해결됨. → "충돌 30% 구조적"이라던 과거 결론들은 *장님 데이터*라 무효.
- **통신은 아직 미입증**: 완전관측(레이더 56m)에선 잉여, 레이더 줄여도 도움 안 됨. 막힘 3개: (1) 메시지에 **위치 없음**(sum 집계 → 받는 배가 위협 방위 모름), (2) 보상 risk가 통신범위 미반영(56m까지만 → 먼 위협 조기회피 보상 0), (3) **decentralized critic**(협력 credit 노이즈).
- **다음 작업(통신을 학습 가능하게)**: ① 메시지에 sender **상대위치 grounding + attention**(sum 대신), ② 보상 risk를 통신범위로 확장(**privileged/CTDE** — 보상≠관측), ③ **중앙 critic(MAPPO)**. 그 후 H1/H2 시험. (참고: 보상이 "먼 위협"을 알아도 obs는 국소 유지 → 통신이 *필수*가 되는 게 CTDE 핵심.)

---

## obs 계약 (59D) — ⚠️ 바꾸면 4파일 동시 수정
`VesselAgent.CollectObservations`가 59 floats 송신:

| index | dim | 내용 | frame-stack |
|---|---|---|---|
| `[0:30]` | 30 | Radar 30섹터 min-distance (`GetSectorMinDistances`) | ✅ ×3 |
| `[30:32]` | 2 | Goal (dist 비선형 `d/(d+k)`, angle/180) | ✗ |
| `[32:36]` | 4 | Self (speed, yawRate, heading, rudder) | ✗ |
| `[36:57]` | 21 | **ARPA** top-3 접점 × 7 (sin,cos,range,closing,dcpa,tcpa,valid) — label-blind 충돌기하 | ✗ |
| `[57:59]` | 2 | Position (x,z) — 통신 파트너 계산용, **네트워크 입력 제외** |

네트워크 입력 57D. `config.py`: STATE_SIZE=30, GOAL_SIZE=2, SELF_STATE_SIZE=4, ARPA_SIZE=21, COLREGS_SIZE=0, POSITION_SIZE=2, OBSERVATION_SIZE=59.
networks fc2 입력: MessageActor `256+2+4+21=283`, ControlActor/Critic `256+2+4+21+MSG_DIM`.
**obs 차원/순서 변경 시 동시 수정**: `VesselAgent.cs:CollectObservations`(+ `Initialize`의 `VectorObservationSize`) → `obs_utils.py:parse_observation` → `config.py` → `networks.py` fc2. (`test.py`/`export_onnx.py`도 stale — 학습엔 무관, 분석 전 갱신 필요.)

## 네트워크 (요약)
- **MessageActor**: obs → MLP → 6D 메시지 (tanh, **msg_out zero-init**). **ControlActor**: obs + **gate·**others_msg → action(squashed gaussian). **Critic**: obs + **gate·**others_msg → value. (Conv1D는 MLP로 교체됨, COLREGs classifier 제거됨.)
- **★메시지 게이트 (H1a fix, 2026-06-02, 검증됨)**: ControlActor·Critic이 `others_msg × sigmoid(msg_gate)` 사용, `msg_gate` 초기 −3(거의 닫힘). loss에 게이트 개방 페널티 `MSG_GATE_COEF·sigmoid(gate)` (ON만). **"메시지 무시"를 init-time이 아닌 *학습된 안정 평형*으로** 만들어 value-of-information≥0를 수렴까지 보장. fc2 메시지슬라이스 zero-init(기존)만으론 메시지가 0에서 자라 H1a 위배됐던 게 근본원인. comm-OFF는 others_msg≡0이라 게이트 무영향(anti-rigging 안전).
- **통신 gradient 수정**: PPO update의 `evaluate_actions`가 **파트너 obs로 MessageActor 재실행**(masked-sum) → sender→receiver gradient (이전 straight-through self-loop 버그 수정, grad≠0 실측). 메시지 L2 정규화(`MSG_L2_COEF`)로 안정화.

## Action (2D continuous)
`[0]` rudder ∈[-1,1]→×maxTurnRate, `[1]` thrust ∈[-1,1]→`(x+1)/2`×maxSpeed. 변경 시 `VesselAgent.OnActionReceived` + `config.CONTINUOUS_ACTION_SIZE`.

## 보상 (C#, `VesselAgent.CalculateReward`)
time penalty / forward bonus / lowSpeed penalty / **fuel proxy**(thrust²+0.5turn²) / proximity(±135°,≤19.6m) / **dense 충돌코스 페널티**(true risk, 매스텝, U자 회귀 방지) / nav(arrival+100, progress, angle) / COLREGs compliance + early-DCPA / smoothness / collision −300.

---

## env 토글 (재빌드 없이 튜닝 — 대부분 Python/런타임)
| env | 의미 |
|---|---|
| `VESSEL_USE_COMM` | 통신 ON(1)/OFF(0). **통신은 Python 전담 → 같은 빌드로 ON/OFF 비교** |
| `VESSEL_MSG_DIM` | 메시지 차원 (H2용) |
| `VESSEL_CROSSING` | 1=4-way 교차(가장 먼 목표 배정, coordination-hard) |
| `VESSEL_RADAR_RANGE` | 에이전트 레이더 범위(m) override (인지부족 실험). 보상 risk는 불변 |
| `VESSEL_COLCOURSE_COEF` | dense 충돌코스 페널티 계수 (기본 -0.8) |
| `VESSEL_MSG_L2` | 메시지 L2 정규화 (통신 안정화, 기본 0.001; **H1a 시험 시 0.01 권장**) |
| `VESSEL_MSG_GATE_L2` | 메시지 게이트 개방 페널티 (기본 **0.02**). 게이트가 "메시지 무시"로 수렴하도록 압박 → H1a value-of-info≥0 보장. 너무 크면 H1b regime서 통신 죽임 |
| `VESSEL_MSG_LR` | MessageActor LR 배수 (기본 1.0; 3.0→1.0, zero-init 구조에선 빠른 LR=노이즈) |
| `VESSEL_AGG_MODE` | 메시지 집계 'sum'(기본)/'mean'/'scale'. ⚠️ rollout=update 동일해야 PPO ratio 유효. **H1a 시험 시 mean 권장** |
| **`VESSEL_ANGLE_COEF`** | angleReward 계수 (기본 **0.15**, 0.5→0.15). path-dependent 회피 변침 페널티 완화(rush>avoid 수정) |
| **`VESSEL_TIME_PENALTY`** | time penalty (기본 **-0.07**, -0.1→-0.07). 회피 detour 시간비용 완화. **-0.03 밑 금지(loiter)** |
| **`VESSEL_EARLY_RISK_GATE`** | earlyAvoid 발화 risk 게이트 (기본 **0.1**, 0.3→0.1). 근접에서도 DCPA-증가 회피보상 |
| **`VESSEL_EARLY_RELAX_TCPA`** | 1(기본)=earlyAvoid의 tcpa>11.5s 게이트 제거(any tcpa). 0이면 옛 동작 |
| **`VESSEL_STRAIGHT_BONUS`** | 1=직진보너스 재활성. 기본 0(off): rudder≈0 보상이 회피 변침과 충돌 |
| `VESSEL_RUDDER_RATE` | 타속(steering-gear, °/s). 기본 12(GlobalScale.RUDDER_RATE, 허용 8~18). 타가 명령으로 슬루 → 깔작 방지. **C#이라 재빌드 필요? 아니오 — VesselAgent.Initialize에서 env 읽어 주입(재빌드 없이 튜닝)** |
| `VESSEL_LOAD_MODEL` | 1이면 `VESSEL_MODEL_PATH` 로드(관찰/이어학습). 기본 0=from-scratch |
| `VESSEL_GRAPHICS` | 1이면 빌드 창 띄워 관찰(headless 끔). 오버헤드 카메라 자동 생성(CameraController, batchmode면 미생성) |
| `VESSEL_MAX_STEP` `VESSEL_RUN_STEP` `VESSEL_SEED` `VESSEL_NUM_ENVS` `VESSEL_BASE_PORT` `VESSEL_USE_EDITOR` `VESSEL_ENV_PATH` `VESSEL_MODEL_PATH` `VESSEL_COMM_FOLDER` `VESSEL_OUTCOME_LOG` `VESSEL_METRIC_LOG` `VESSEL_TIME_SCALE` | 학습/환경/로그 |

**성능**: latency-bound, GPU 놀음. 무위험 속도개선=**독립 프로세스 병렬(이 머신 ~6개)**. 단일 run은 못 빠르게(동기 왕복). NUM_ENVS>1(한 프로세스)은 2× 느림. 레이더 ray↓/척수↑는 *결과를 바꿈*(버그 아님). C# 바꾸면 **재빌드** 필수, Editor는 자동 반영.

**평가는 항상 ground-truth 로그로**: `VESSEL_OUTCOME_LOG`(goal/collision_vessel/collision_obstacle/timeout), `VESSEL_METRIC_LOG`(**13열**: id,ep,outcome,steps,fuel,rudderVar,compliance,occlRate,commandVar,**minVesselDist,nearMissSteps,straightness,headingTravel**). 뒤 4열=near-miss/circling **진단 전용, 보상 절대 비연결**(제약3). `convergence_gate.py`(수렴게이트+LATE_COLLAPSE), `analyze_circling_safety.py`(circling/near-miss)로 분석. reward 임계값 추정 금지.
- **★보상 재설계 + 통신 zero-init (2026-06-01)**: 진단 결과 'rush>avoid'의 원인은 progress가 아니라(이미 γ=1 telescoping) **path-dependent 항**(angleReward·직진보너스·timePenalty)이 회피 변침을 직접 벌함. 수정: angleReward 0.5→0.15, 직진보너스 off, timePenalty -0.1→-0.07, **earlyAvoid 게이트 0.3→0.1 + any-tcpa + ×speedRatio**(회피=DCPA증가 이벤트, orbit=보상0). progress는 불변(γ=0.99 PBS는 정지 farming 버그라 기각). **통신 H1a 진짜 버그=ControlActor/Critic fc2 메시지 슬라이스 random-init**(zero-init 아니라 comm-ON이 OFF와 다른 출발선)→`networks.py` zero-init으로 value-of-information≥0 구조 보장. **from-scratch 재학습 필수**.
- **★타속 슬루(2026-06-01)**: 타가 명령으로 *즉시* 안 가고 `RUDDER_RATE`(°/s)로 슬루(`VesselDynamics`) → "배 깔작대기" 물리버그 수정. 영향: ① **fuel·smoothness 보상은 *commanded* 타각 기반**(실제 타각은 슬루로 평탄화돼 gradient 소실 — RL연구원 진단). ② `commandMismatchCoef`(-0.03) 타속포화 패널티 신규. ③ rudderVar(실제) 메트릭은 슬루로 절대값↓ → **commandVar(명령)이 진짜 부드러움 신호**. ④ per-dim logstd(rudder std↓). **동역학 변경 → from-scratch 재학습 필수**(옛 모델 호환X).

---

## 코드 규칙
- **C#**: PascalCase(클래스/메서드), camelCase(지역). 주석 한국어. `[Header]` public 필드. `Debug.Log` 금지(`Debug.LogWarning`만, setup 에러).
- **Python**: snake_case. 주석 한국어/docstring 영어. **모든 상수·경로·차원은 `config.py`.** production 코드에 bare `print()` 금지(학습 진행/에러 출력만).
- **데이터 위치**: `models/`,`trajectory_data/`,`figures/`는 `Assets/` *밖*(Unity 무한 import 방지).
- **GitHub**: git root=`Assets/Scripts/`. C# 파일 복사 금지(Unity 중복 컴파일). 원본 직접 `git add`.

## 과학적 정직성 (이 프로젝트의 제1원칙)
통신이 도우면 ground-truth로 입증, **안 도우면 정직하게 "안 도움"이 결론.** baseline을 불구화해 통신을 이기게 만들지 않는다. H1/H2는 *달성할 목표*지만 *조작으로 만들 결과*가 아니다.
