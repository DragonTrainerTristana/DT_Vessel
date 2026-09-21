# VESSEL_DYN_PROFILE=imo — 조종성 현실화 토글 설계

2026-09-19 작성, 2026-09-21 보강(§2 open-sea 토글 · §4 7항목 지표 · §5 게이트 보류 · §8 차원 스윕). 브랜치 `feat/dyn-profile-imo` (기준점 태그 `v2026-09-15-branch-protocol`, main `6de6816`).
**구현 전 설계 문서.** 근거 조사 = 워크플로 17 에이전트(조사 6 · 반박 10 · 비평 1), 수치 = `2026-09-19-dyn-profile-imo/dyn_profiles.py`(독립 재구현 `indep_check.py`·`imo_check.py` 로 재현 확인).
구현 계획 = `docs/superpowers/plans/2026-09-21-dyn-profile-imo-python.md`(Python·실험) + `2026-09-21-dyn-profile-imo-unity.md`(C#, Windows).

---

## 0. 한 줄 + 결정

통신이 필요한 regime 은 레이더를 줄여서(센서 불구화) 만들지 않고, **배 동역학을 IMO 봉투로 현실화**해서 만듦. 양 팔 동일 조건.

저자 결정(2026-09-19):

| # | 결정 | 내용 |
|---|---|---|
| 1 | 선회 식 | 절대속도 식 `yaw = (rudder/30°)·speed/R_FULL`, **R_FULL = 2 L 전 선박 고정**(TD 4 L). agile 은 옛 식 그대로(비트동일) |
| 2 | 정지 | imo 프로필에 `DECEL/DRAG/ACCEL` 포함. 정지거리 ≈ 5 L (v 1.0) |
| 3 | 보상 상수 | 물리 파생 상수를 **같은 규칙으로 재산정**, 가중치 불변. 규칙 표(§5)를 결과 전에 고정. 양 팔 동일 |
| 4 | 파일럿 팔 | 기존 `OFF / ON / RANDOM` 만. AIS·POS-only 팔은 별도 스펙 |
| 5 | 버전 | 브랜치 + 스냅샷 키 `dyn_profile`. `CODE_VERSION` 유지(기본 agile 비트동일) |
| 6 (09-21) | 시나리오 | `VESSEL_OBSTACLES=grid3x3|none` 토글 추가. 파일럿 = **open-sea(`none`)** 주, coastal(`grid3x3`) 보조 |

---

## 1. 배경

### 1-1. 왜 레이더가 아니라 동역학인가

- 현재 sim: 전타 yaw 45 °/s, 정상 선회직경 **0.18 L**(v 1.0; 함대 0.14–0.32 L), 30° 변침 1.81 s, 정지 5 m / 12 s. L = 14.18 m 충돌 박스.
- IMO MSC.137(76): 선회직경 ≤ 5 L, advance ≤ 4.5 L, 초기선회 ≤ 2.5 L. 실선 TD 3–5 L. sim 은 규정 상한의 ~28배 민첩.
- 레이더 56 m = 선회직경 22개. 보이고 나서 피할 시간이 남아 통신이 낄 자리 없음. COMM_PLAN §3-2 실측 "충돌 상대 12 s 전 100 % 레이더 안" 과 정합.
- 실제 레이더는 range 가 sim 보다 훨씬 김(수 nmi). 56 m 는 부분관측 유도 설계값. 더 줄이면 현실에서 멀어짐.
- 격차 비 W/T = r_d/(2cR): **속도 무관, r_d/R 만 결정.** 레이더 고정이면 R 이 유일한 물리 지렛대.

### 1-2. 원인 — 길이가 소거되는 공식

- C# `VesselDynamics.cs:44,131-136`: `turnFactor = rudderEff 1.5 × (10/length) × (beam/2)`, length 2.0 · beam 0.4 (`GlobalScale.LENGTH/BEAM` = 10·2 × 0.2). B/L 고정이라 **모든 스케일에서 1.5** — 스케일을 바꿔도 민첩성 불변, 콜라이더만 커짐.
- 충돌 박스 14.18 × 1.93 m = 프리팹 BoxCollider 70.92 × 9.64 × 런타임 localScale 0.2 (`VesselDynamics.cs:71`). `GlobalScale.LENGTH` 2.0 은 선회 공식에서만 쓰이고 콜라이더와 코드상 무관. 두 값을 묶는 코드가 처음부터 없었음.
- Python `vessel_gym.py:34-37` 은 접힌 상수 `TURN_FACTOR 1.5 / MAX_YAW_RATE 45 / RUDDER_RATE 12` 리터럴. config 밖·스냅샷 밖.

### 1-3. 왜 FOG 는 rigging 이고 imo 는 아닌가 (COMM_PLAN §5 기각 번복 근거)

- FOG(레이더 축소) = **센서 불구화**. OFF 는 현실에서 가졌을 정보를 잃고, ON 의 relpos 채널(300 m)은 그대로 → 비대칭. 현실에서 멀어짐.
- imo = **플랜트 현실화**. 센서 불변, 양 팔 정보집합이 매 스텝 동일(메시지 채널만 차이). 현실에 가까워짐. 규정(MSC.137, SOLAS II-1/29) 근거.
- 09-04 기각 문구 "동일 조건에서 이겨야 함" 은 imo 가 충족.

### 1-4. 선례

- `VesselAgent.cs:362-365` (커밋 `662a3ff`, 2026-06-13): physicist 권고 최대타 10°, "R_min 이 radar 에 붙어야 OFF 가 56 m 서 급기동 강제, ON 은 미리 완만회피. comm ON/OFF 동일 적용 = anti-rigging". `VESSEL_MAX_TURN_RATE` env 만 C# 에 있고 Python 짝 없음. 실행 흔적 없음. 같은 서사, 미실행.
- 08-08 `qd_FOG28` (레이더 28 m, s42 1시드, ring 0.7 시절): OFF goal 59.9 / COMM 58.5 %. 근거로 못 씀(시드 1·구 기하·체크포인트 소실).

### 1-5. 왜 open-sea 를 주 시나리오로 (09-21, 저자 지시 7항)

- 장애물 3×3(r 20 m, 간격 120 m)은 통신이 못 돕는 정적 위험. 초기 학습 oColl 스파이크 76 %·dying-ReLU 붕괴(`COLLAPSE_ROOTCAUSE.md`)의 원인. R 28 m 면 통과폭 80 m 대비 빠듯.
- 통신 가설은 **선박 간** 협응. open-sea(장애물 0, 벽만)면 조우 기하(head-on/crossing/overtaking)만 남아 효과가 섞이지 않음. 문헌 표준 설정(Meyer 2020, Wang & Zhao 2024).
- coastal 은 보조: open-sea 에서 효과가 보이면 장애물 있는 환경에서도 유지되는지 확인.

---

## 2. 범위

**포함**
- `VESSEL_DYN_PROFILE=agile|imo` 토글. 기본 `agile` = 현행과 비트동일.
- `VESSEL_OBSTACLES=grid3x3|none` 토글. 기본 `grid3x3` = 비트동일.
- imo 프로필 = 선회 식 + R 고정 + 타속 + 정지(감속·저항·가속) + §5 파생 상수.
- 스냅샷 기록·복원·불일치 중단·분기 검사(§6). 기존 구멍(radar_range·dropout·los_gate 미복원) 같은 자리에서 수리.
- Python(GPU 배치) + C#(Unity 판정관) 미러.
- 파일럿 실험 계획 + 사전등록 기준(§4·§8).

**제외(별도 스펙)**
- AIS / POS-only 팔.
- earlyAvoid·저속 게이트의 300 m 확장(§5 보류 사유 참고).
- yaw 관성(Nomoto T). 1차 대수식 유지. zig-zag 오버슈트 1–8° 로 실선보다 얌전함은 한계로 명시.
- 선회 중 감속(실선 30–50 %). 미모델 명시.
- 센서 현실화(잡음·스캔주기·미탐) = 패키지 A. 파일럿 뒤.
- 스폰·목표 기하 변경.
- Fig9 A* 기준선 재평가(R 28 m 면 웨이포인트 추종 불가 — imo 에서 무효, 논문에 명시).
- 조우 프로필 지표(첫 변침 거리·최대 타각 분포) 확장 — 기존 `eval_ckpt` 의 첫 변침 시간·R_stb·DCPA 층화로 1차 판정, 부족하면 후속.

---

## 3. 프로필 정의

무차원 약속: 길이 L = 14.18 m(콜라이더), 시간 L/U (U 1 m/s → 14.2 s). 논문 표현 = "14 m 선체에 100 m 급 상선의 무차원 동역학(L/V 8–18 s)". MSC.137 적용범위(LOA ≥ 100 m)는 무차원으로 차용.

### 3-1. 식

| | agile (현행, 비트동일) | imo |
|---|---|---|
| yaw [°/s] | `rudder × (speed/max_speed_i) × TURN_FACTOR` | `(rudder/MAX_TURN_RATE) × speed / R_FULL / DEG` |
| 정상 선회반경 | `max_speed_i/(30·1.5·DEG)` = 1.27·max_speed_i m → **배별로 다름** | **R_FULL = 2 L = 28.37 m 전 선박 동일**, 속도 무관 (Nomoto r = K'(V/L)δ 와 동형) |
| 타 슬루 | `RUDDER_RATE 12` °/s | 3 °/s (SOLAS II-1/29: 35°→−30° 28 s ≈ 2.3 °/s 최소. 상선 전형 2.3–3.5) |
| 최대 타각 | 30° | 30° (불변 — obs[365]·보상 정규화 상수) |
| 감속 DECEL | 0.04 | 0.004 |
| 저항 DRAG_COEF | 0.1 | 0.005 |
| 가속 ACCEL | 0.1 | 0.01 |
| obs[363] 분모 `MAX_YAW_RATE` | 45 (= 30 × 1.5) | `1.8 / R_FULL / DEG` = 3.635 °/s (함대 최고속 1.8 기준 전타 yaw). **파생값으로 정의**, 리터럴 금지 |

- 식 선택은 프로필 dict 의 `formula` 키(`'ratio'` = agile / `'abs'` = imo). `vessel_gym.yaw_rate_deg()` 하나가 `_substep`·`_build_obs` 양쪽에 쓰임(agile 분기는 기존 연산 순서 그대로 = 비트동일).
- `MAX_YAW_RATE` 가 파생값이면 agile 에서 obs[363] = rudder_n × speed_ratio 로 값·의미 비트동일(반박 검증 confirm). imo 에서는 obs[363] = yaw/3.635 → 분포 바뀜 = from-scratch(어차피 전이함수 변화로 from-scratch).
- 정지거리(감속·저항 포함 적분, `stop_lat6.py`): v 0.8 → 49 m (3.5 L), v 1.0 → **70 m (4.95 L, 161 s)**, v 1.8 → 171 m (12.1 L). IMO 정지 한계 15 L 안. 레이더 56 m 안에서 정지 불가 = 감속 우회 차단.
- ACCEL 0.01: 0→1 m/s 100 s. 스폰 초기 target U(0.2,0.5)·vmax 라 초반 100 s 저속. 실선(분 단위)보다 빠름 — 설계값.

### 3-2. 검증된 수치 (TD 4 L, RR 3, v 1.0)

| 항목 | 값 | IMO 한계 |
|---|---|---|
| 정상 선회직경 | 4.00 L | ≤ 5 L ✓ |
| tactical diameter(180°) | 4.01 L | ≤ 5 L ✓ |
| advance(90°) | 2.35 L (v 1.8: 2.63) | ≤ 4.5 L ✓ |
| 초기선회(10° 타 → 10° 변침) | 1.17 L | ≤ 2.5 L ✓ |
| zig-zag 10/10 · 20/20 오버슈트 | 0.9–2.0° · 3.6–8.1° | ≤ 10–13.9° · 25° ✓ (관성 없어 얌전) |
| 30° 변침 | 19.8 s | (agile 1.81 s) |
| 전타 후 횡변위 t = 7/14/28/56 s | 0.2 / 1.6 / 9.0 / 34.9 m | |
| T_lat6 / T_lat12 / T_lat14 | 23.6 / 32.0 / 34.6 s | (agile T_lat12 14.1 s) |

TD 3 L 은 head-on 격차 ~1.0(약함), TD 5 L 은 tactical diameter 5.00–5.03 L 로 규정 경계 → **4 L 채택.** RR 3/5/12 는 2차 효과(T_lat12 31.9/30.1/28.4 s).

### 3-3. 격차 — 정직한 수치

W = 탐지지평 도달 후 접촉까지 시간, T = 회피 기동 소요.

| 조우 (r_d 56 m) | W | T | W/T imo | W/T agile |
|---|---|---|---|---|
| head-on 한 척 전담 12 m | 28 s | 32.0 s | **0.88** | 1.99 |
| head-on 양측 6 m 씩 | 28 s | 23.6 s | 1.19 | — |
| crossing give-way 단독 12 m (closing √2·v) | 39.6 s | 32.0 s | 1.24 | 2.81 |
| overtaking (closing 1.0) | 56 s | 32 s | 1.75 | 3.98 |
| head-on, r_d 84 / 112 / 168 m | | | 1.32 / 1.75 / 2.63 | |

- **격차는 얇음.** agile 대비 여유 ~2배 줄지만 대부분 레이더 안 급기동으로 해결 가능(W/T ≈ 1). IMO 상한 R ≤ 2.5 L 이면 r_d/R ≥ 1.6 → 규정 안에서 W/T 를 0.5 로 못 내림.
- 예상 1차 효과 = **충돌률이 아니라 효율·여유**(fuel · headTravel · 타각 사용 · 궤적 직진도 · 조우 시 DCPA 여유). COMM_PLAN §3-3 "perpair 벌점 54–58 % 가 56–200 m 밴드" 와 정합 — 둔한 배는 그 밴드 정보 가치가 큼.
- 통신 300 m 에서 W = 150 s: 5° 변침만으로 횡 26 m > SAFE_PASSING 12 m ("머리서 살짝"). 56 m 에서는 ~45° 전타 32 s ("급격"). 이 대비가 저자 지시 2항의 물리적 근거.
- 정지 우회 차단(§3-1) 없이는 격차가 더 얇아짐(정지 5 m).
- 이 수치는 단일 쌍 기하. 16척 open-sea 에서는 제3선 제약으로 필요 runway 가 더 큼(정량 미확인 → 파일럿).

### 3-4. 시나리오

| | coastal (`grid3x3`, 현행) | open-sea (`none`, 파일럿 주) |
|---|---|---|
| 장애물 | 9개 r 20 m 3×3 격자 120 m | 0 |
| 벽 | ±299.5 m | 동일 |
| 스폰·목표 | 씬 20/16 점, ring 1.0, crossing 2(대척) | 동일 — 대척 교통이 중앙에서 교차 = head-on/crossing 다발 |
| 밀도 | 16척 | 16척(파일럿), 밀도 스윕은 후속 |
| LOS 게이트 | `LOS_GATE` | 무의미(장애물 0) |
| Unity 판정관 | 현 씬 | 장애물 없는 씬 변형 필요(Plan B) |

---

## 4. 예상 효과·성공 기준 (사전등록) — 저자 지시 7항목 매핑

파일럿(§8) 결과 보기 전에 고정. 전부 ground-truth(`eval_ckpt.py`), 3시드, 시드별 승패 병기.

| 저자 항목 | 지표 (`eval_ckpt.py` 출력) | ON vs OFF 판정 | 물리적 기대 |
|---|---|---|---|
| 1 COLREGs 준수 | `colregs`(준수 점수) · `colregsOK %` · 역할별 R_stb/R_ck · Rule 8 @10° · 진입 DCPA 층화 | ON > OFF | give-way 가 56 m 밖에서 조기·충분 행동(Rule 16), stand-on 이 침로 유지(Rule 17) 가능해짐 |
| 2 궤적 여유 | `headTravel`(총 변침) · 첫 변침까지 시간 중앙값 · 무변침 조우 % · R_stb(타각 비율) | ON: headTravel↓, 첫 변침 이르고 작음 | 300 m 에서 5° vs 56 m 에서 45° |
| 3 연료 | `fuel`(goal-ep) | ON < OFF | 급기동·재가속 감소 |
| 4 보상 곡선 | `*.csv` EMA reward, `epReward` | ON > OFF (분기 뒤) | colcourse/perpair 300 m 벌점 감소 |
| 5 차원 | dim {2, 6, 12} × 3시드 | 단조 증가 **또는 포화** — 있는 그대로 | agile 에선 불지지(F 0.39). imo 에서 재검정 |
| 6 안전 | `vColl` · `minSep` · `allMinSep` · DCPA 층화 | ON ≤ OFF 충돌(H1a 필수), minSep ↑ | |
| 7 환경 | open-sea 주 / coastal 보조 | open-sea 에서 1–6 판정, coastal 은 유지 여부 | §1-5 |

- 판정 규칙: 3시드 중 2승 이상. 시드 제외는 통신에 불리한 방향으로만, 코드 먼저 의심(`COLLAPSE_ROOTCAUSE.md`).
- `RANDOM`: ON > RANDOM 이어야 "정보" 효과. RANDOM ≈ ON 이면 구조 효과 = 통신이라 부르지 않음.
- ON 이 OFF 보다 충돌·준수에서 **나쁘면 구현 버그로 취급**(H1a).
- 결과 나온 뒤 지표·기준 변경 금지(rigging). 변경하면 문서에 이력 남김.
- "압도적" 여부는 결과가 정함. 약속 아님.

---

## 5. 보상 상수 재산정 규칙 (imo 만, 양 팔 동일)

원칙: **가중치(계수) 불변. 물리 스케일에서 파생된 기준 시간·거리만 같은 파생 규칙으로 재계산.** 규칙 = 명시적 비율. 결과 보기 전에 이 표로 고정. 보상 정의 변경이라 루트 CLAUDE.md §1 저자 승인 사항(2026-09-19 결정 3 으로 승인).

시간 배율 k_t = T_lat12(imo)/T_lat12(agile) = 32.0/14.1 = **2.27**. 값은 `config.dyn_profile_constants('imo')` 한 곳에서 계산(스냅샷에 숫자로 기록).

| 상수 | 위치 | agile | 파생 규칙 | imo | 왜 |
|---|---|---|---|---|---|
| `TCPA_RISK_DENOM` | `vessel_gym.py:119` | 30 s | × k_t | 68.1 s | 위험 0.5 시점이 회피 가능 시점보다 앞서야 shaping 기울기가 살아 있음 |
| `RULE_17B_TIME / 17C_TIME` | `:84-85` | 7 / 3.5 s | × k_t | 15.9 / 7.95 s | 7 s 는 타 슬루(10 s)도 못 끝냄 |
| `RULE_17B_DIST / 17C_DIST` | `:86-87` | 18 / 9 m | 시간 × closing 파생 → × k_t | 40.9 / 20.4 m | 18 m < R 28 m 물리 무의미 |
| `EARLY/SUBSTANTIAL_ACTION_TIME` | `:82-83` | 21.5 / 11.5 s | × k_t (기본 비활성, 정합용) | 48.8 / 26.1 s | Rule 16 단계 게이팅 |
| `CMD_MISMATCH` 정규화 | `:925-926` | `|cmd−rudder|/30` | 달성 가능 슬루 공제: `max(0, |cmd−rudder| − SLACK)/30`, SLACK = RR × 0.4 s | SLACK 1.2° (agile 0 = 비트동일) | 타속 느림을 벌하면 물리를 벌하는 것 |
| `GOAL_REACHED` | `:62` 3.0 m | 0.21 L | → L/2 | **7.09 m** | 3 m 원 오버슛 시 178 m 루프 = 446 결정 손실 |
| 근접 THR | `:739` 19.6 m | RADAR_BASE×0.35 | **유지** | 19.6 m | 센서 유래. γ 0.99 지평(≈40 s) 안에 T_lat12 32 s 있어 credit 전달됨 |
| `DCPA_RISK` / `SAFE_PASSING` | `:120`, `:88` | 24 / 12 m | 센서·규정 유래 → **유지** | 동일 | 레이더 불변 |
| earlyAvoid 게이트 · 저속 게이트 | `:884`, `:734` | near_risk(56 m) | **파일럿 보류** | 유지 | 게이트가 `_sit>0`·`danger_idx`(56 m 판정)에 결합돼 있어 300 m 확장은 위험 대상 선택 로직 신설 = 보상 구조 변경. colcourse/perpair(risk³, 300 m 연속)가 이미 조기 회피를 벌점 감소로 유도하므로 파일럿은 유지, 결과 보고 2단계 |
| `MAX_EPISODE_STEPS` | `config.py:473` 4500 결정 | k 몬테카를로(agile) | 파일럿 trunk 로 k 재측정 후 결정(`--drain`) | 유지(4500) | 추정 상향 금지(§2 원칙). timeout 비율을 파일럿 지표로 기록 |
| smoothness · progress · angle · time · fuel · forward · 종료보상 | | | **불변** | | 가중치 |

- 상황판정 반경 `DETECTION_RANGE` 56 (`:115`, `:589`) · MoE 라우팅 · obs[368]: **불변**(C# `COLREGS_DETECTION` 동기 필수, 단독 변경 금지). imo 에서 "조우 판정 시점에 회피 여지 없음" 은 해석 단서로 논문에 명시.
- C# 미러: 같은 상수가 `GlobalScale.cs:82-93` const → `COLREGsHandler.cs` 인라인. imo 값을 C# 도 받아야 sim2sim 판정 정합(Plan B).

---

## 6. 구현 터치포인트 (요약 — 상세는 plans/)

### 6-1. Python 정본·sim

| 파일 | 할 것 |
|---|---|
| `config.py` | `DYN_PROFILE`·`OBSTACLES_MODE` env + `dyn_profile_constants(profile)` dict 함수 + `DYN` + `YUGIOH` 키 2개 |
| `vessel_gym.py` | 동역학·§5 상수를 `_cfg.DYN` 에서 import(모듈 속성 이름 유지) · `yaw_rate_deg()` 헬퍼 · `_substep`/`_build_obs` 가 헬퍼 사용 · cmd_mismatch SLACK 분기 · `apply_dyn_constants()` · 장애물 `none` 분기 |

### 6-2. 스냅샷·복원·검사 (기존 구멍 수리 포함)

| 파일 | 할 것 |
|---|---|
| `ckpt_io.snapshot_config` | `dyn_profile`·`dyn`(숫자 dict)·`obstacles`·`radar_dropout_p/len`·`los_gate`·`max_episode_steps` |
| `ckpt_io.restore_policy` | 스냅샷 `dyn_profile`(없으면 `'agile'`)·`obstacles`(없으면 `'grid3x3'`)·`radar_range` vs 현재 config 대조 → 불일치 **중단**, `allow_sim_mismatch`(comm_range 플래그와 **별도**)로만 우회. 그 뒤 `vg.apply_dyn_constants(snap['dyn'])`·`vg.OBSTACLES_MODE` 적용. `effective`·`header()` 에 기록 |
| `ckpt_io.make_env_from_snapshot` | env 생성 **전** 스냅샷 dyn·obstacles 적용(멱등) + 로그 |
| `ckpt_io.describe / _SNAP_TO_ENV` | 키 2개 |
| `vessel_gym_train.py` | 재개(trunk 분기 포함) 시 `_prev_snap` 의 dyn_profile·obstacles == 현재 아니면 거부. 시작 로그 1줄 |
| `verify/check_branch.py` | 같은 trunk 묶음 안 `dyn_profile`·`obstacles` 일치 |
| `eval/diag_ckpt.py` | config_match note 갱신 |
| `eval/eval_mixed.py`, `measure_regimes.py`, `corridor_run.py` | restore → env 순서로 |
| `run_repro.sh`, `smoke_mac.sh` | common_env 가 `${VESSEL_DYN_PROFILE:-agile}`·`${VESSEL_OBSTACLES:-grid3x3}` 를 export(바깥 override 보존). preflight 헤더에 프로필 출력. `arm_spec` 에 `on2/off2` |
| `verify/test_golden.py` | `_YUGIOH_CONSTS` 키 2개. 케이스 5개는 agile → 비트동일 자동 검증. **imo 골든은 `--regen` 승인 별건** |
| `verify/test_vessel_gym_fidelity.py` | 리터럴 45 / 0.004 → 프로필 파생값 기준 |
| `verify/test_dyn_profile.py` (신규) | agile dict == 옛 리터럴, imo 정상 선회 R 28.37±1 %, 정지거리 70 m±5 %, obstacles none, 스냅샷 왕복·불일치 중단 |
| `.claude/CLAUDE.md`, `WINDOWS_RUN.md` | 토글 행·실행 절차 |

### 6-3. C# (Plan B, Windows 재빌드)

`VesselDynamics`(imo 선회 분기·MaxYawRate)·`VesselAgent.Initialize`(env 읽기, 슬루·가감속·저항·도착반경·mismatch slack)·`COLREGsHandler`(const → static, k_t 적용)·장애물 없는 씬 변형. 미러 3곳·`_verify_comm_mirror.py` 무관(obs 계약·집계 불변).

---

## 7. 검증 계획 (구현 단계 게이트)

1. `verify/test_golden.py --check` 5케이스 ALL PASS(agile 비트동일) — 학습기 파일 건드리는 태스크마다.
2. `verify/test_dyn_profile.py` 전부 PASS + `test_vessel_gym_fidelity.py` PASS(agile), `VESSEL_DYN_PROFILE=imo` 로도 PASS.
3. `verify/_verify_comm_mirror.py` ALL PASS(Mac OK). `_verify_ppo_mirror.py` 는 Windows preflight.
4. 스냅샷 왕복: imo 로 2 update 학습 → `ckpt_io.py X.pt` 에 `dyn_profile=imo` → agile config 에서 restore → **중단** 확인 → `allow_sim_mismatch` 로만 진행.
5. trunk agile → 갈래 imo 시도 → 학습기 거부 확인. `check_branch.py` 불일치 FAIL 확인.
6. `smoke_mac.sh` agile + imo/none. Windows `run_repro.sh smoke` imo/none.
7. C# 재빌드 후 Unity 1 에피소드 궤적 vs Python 동일 명령열 → 위치 오차 ≤ 기존 fidelity 허용(Plan B).

---

## 8. 실험 계획

1. **파일럿(open-sea, imo)**: trunk OFF 3시드 9,043,968 결정 → 갈래 OFF / ON(dim 6) / RANDOM(sd = trunk `om_sd` 실측) → `eval_ckpt` §4 판정. 별도 `VESSEL_CKPT_DIR`·`VESSEL_OUT_DIR`(agile 배치와 섞지 않음).
2. **k 재측정**: trunk 로 `--drain` 평가 → timeout 비율·에피소드 길이 → `MAX_EPISODE_STEPS` 결정.
3. **차원 스윕(항목 5)**: 파일럿 통과 시 dim {2, 12} 추가(`on2/off2`, `on12/off12`) × 3시드. 보고 = 있는 그대로(단조/포화/무관).
4. **coastal 보조**: 같은 프로필로 `grid3x3` 1배치(OFF/ON) — 효과 유지 여부.
5. **dose-response**: x = r_d/R. `RADAR_RANGE` ∈ {56, 84, 112, 168} 위로 스윕(레벨마다 trunk 별도) → 이득 → 0 확인 = anti-rigging 그림.
6. 붕괴 검출기(`COLLAPSE_ROOTCAUSE.md` 기준 A 700·leaky 0/80)·`diag_ckpt` 조우율 5 % 게이트 = agile 실측 → imo 첫 배치는 검출기 OFF, 실측 후 재보정.
7. 텔레메트리 추가 1개(후속): 56–300 m 밴드에서의 회피 행동량 — "통신이 그 밴드에서 일하나" 직접 지표.

기존 12런·그림 전부 = agile regime. imo 는 **대체가 아니라 별도 regime**. 비교 불가 명시.

---

## 9. 위험

| 위험 | 근거 | 대응 |
|---|---|---|
| OFF 학습 실패 | `vessel_gym.py:905-912` "직진하다 박기 70배 유리", off_s45, oColl 스파이크 | open-sea 가 1차 완충. 그래도 안 배우면 TD 3 L 완화(IMO 안) → 문서화 |
| "0.3 스케일 78 % 학습불가" 전례 | 주석 한 줄, 같은 커밋에 timeout 78 %·오분류 80 % 혼재 → 측정 방식 불명 | 근거로 안 씀 |
| 기하 함정 | 스폰~벽 42.4 m = 1.5 R; 목표 3 m 루프; 스폰 간격 100 m vs TD 57 m | §5 GOAL 7.09 m. 파일럿 outcome 분해로 확인 |
| 시간초과 재발 | k 몬테카를로 agile 기준 | §8-2 재측정 |
| 격차가 얇아 ON ≈ OFF | §3-3 W/T 0.9–1.2 | 정직하게 보고. 효율 지표가 주 가설. r_d/R 곡선으로 "0 이득" 끝점 포함 |
| 프로토콜 학습 실패 | tanh 포화·게이트 흡수 전례 | 기존 `msg_ln`·×0.1 init 유지. 텔레메트리로 확인 |
| 스냅샷·복원 구멍 | §6-2 | 같이 수리. `allow_sim_mismatch` 별도 |
| Dropbox `.git` 공유 충돌 · 실행 중 학습 오염 | 09-11 사고 | Mac 개발 = Dropbox-ignore 된 `_dev_dyn/` clone. Dropbox 트리는 main 고정. Windows 는 GitHub clone 에서 `feat/dyn-profile-imo` pull |

---

## 10. 미결 (구현 중 저자 확인)

- `MAX_EPISODE_STEPS` — 파일럿 측정 후.
- 근접 THR 19.6 유지 여부 — 파일럿 분포 보고 후.
- earlyAvoid/저속 게이트 300 m 확장 — 2단계.
- 분기점 9,043,968 vs 고원 기준.
- imo 골든 케이스 추가(`--regen` 승인).
- Unity open-sea 씬 변형 방식(장애물 비활성 env 토글 vs 별도 씬).

---

## 11. 이력·버전

- `v2026-09-15-branch-protocol` (main `6de6816`) = agile 최종판 + §8-1 분기 규약. GitHub 반영.
- `feat/dyn-profile-imo` = 이 스펙 + 계획 + 구현. main 머지 조건: §7 게이트 전부 + 파일럿 1회.
- `CODE_VERSION 'YUGIOH-2026-09-10'` 유지. 체크포인트 구분 = 스냅샷 `dyn_profile`·`obstacles`.
- 09-04 COMM_PLAN §5 "레이더 축소 regime 기각" → §1-3 근거로 **동역학 현실화로 대체**(레이더 축소 자체는 여전히 안 함).

## 부록 — 근거 스크립트 (`2026-09-19-dyn-profile-imo/`)

| 파일 | 내용 |
|---|---|
| `dyn_profiles.py` + `.out` | `_substep` 식 그대로 옮긴 순수 파이썬. 프로필 × max_speed × RR 전 조합: 선회·advance·초기선회·횡변위·W/T |
| `indep_check.py` | 독립 재구현(dt 0.001) — 0.18 L·8.97 m·T_lat12 31.93 s 재현 |
| `imo_check.py` | zig-zag 오버슈트·정지거리 검산 |
| `stop_lat6.py` | 정지거리 후보(DECEL/DRAG) 표 · T_lat6/12/14 (절대속도 식) |

실행: `python3 <파일>` (numpy·torch 불요, Mac OK).
