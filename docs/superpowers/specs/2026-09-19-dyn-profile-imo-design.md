# VESSEL_DYN_PROFILE=imo — 조종성 현실화 토글 설계

2026-09-19. 브랜치 `feat/dyn-profile-imo` (기준점 태그 `v2026-09-15-branch-protocol`, main `6de6816`).
**구현 전 설계 문서.** 근거 조사 = 워크플로 17 에이전트(조사 6 · 반박 10 · 비평 1), 수치 = `2026-09-19-dyn-profile-imo/dyn_profiles.py`(독립 재구현 `indep_check.py`·`imo_check.py` 로 재현 확인).

---

## 0. 한 줄 + 결정

통신이 필요한 regime 은 레이더를 줄여서(센서 불구화) 만들지 않고, **배 동역학을 IMO 봉투로 현실화**해서 만듦. 양 팔 동일 조건.

저자 결정(2026-09-19):

| # | 결정 | 내용 |
|---|---|---|
| 1 | 선회 식 | 절대속도 식 `yaw = rudder·speed/(30°·R_full)`, **R_full = 2 L 전 선박 고정**(TD 4 L). agile 은 옛 식 그대로(비트동일) |
| 2 | 정지 | imo 프로필에 `DECEL/DRAG/ACCEL` 포함. 정지거리 ≈ 5 L (v 1.0) |
| 3 | 보상 상수 | 물리 파생 상수 8개를 **같은 규칙으로 재산정**, 가중치 불변. 규칙 표(§5)를 결과 전에 고정. 양 팔 동일 |
| 4 | 파일럿 팔 | 기존 `OFF / ON / RANDOM` 만. AIS·POS-only 팔은 별도 스펙 |
| 5 | 버전 | 브랜치 + 스냅샷 키 `dyn_profile`. `CODE_VERSION` 유지(기본 agile 비트동일) |

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

- `VesselAgent.cs:362-365` (커밋 `662a3ff`, 2026-06-13): physicist 권고 최대타 10°, "R_min 이 radar 에 붙어야 OFF 가 56 m 서 급기동 강제, ON 은 미리 완만회피. comm ON/OFF 동일 적용 = anti-rigging". `VESSEL_MAX_TURN_RATE` env 만 C# 에 있고 Python 짝 없음. 실행 흔적 없음(runs/·queue_scripts 에 없음). 같은 서사, 미실행.
- 08-08 `qd_FOG28` (레이더 28 m, s42 1시드, ring 0.7 시절): OFF goal 59.9 / COMM 58.5 %. 근거로 못 씀(시드 1·구 기하·체크포인트 소실).

---

## 2. 범위

**포함**
- `VESSEL_DYN_PROFILE=agile|imo` 토글. 기본 `agile` = 현행과 비트동일.
- imo 프로필 = 선회 식 + R 고정 + 타속 + 정지(감속·저항·가속).
- 보상 물리 파생 상수 재산정 규칙(§5) — imo 일 때만 적용.
- 스냅샷 기록·복원·불일치 중단·분기 검사(§6). 기존 구멍(radar_range 미복원) 같은 자리에서 수리.
- Python(GPU 배치) + C#(Unity 판정관) 미러.
- 파일럿 실험 계획 + 사전등록 기준(§8).

**제외(별도 스펙)**
- AIS / POS-only 팔.
- yaw 관성(Nomoto T). 1차 대수식 유지. zig-zag 오버슈트 1–8° 로 실선보다 얌전함은 한계로 명시.
- 선회 중 감속(실선 30–50 %). 미모델 명시.
- 센서 현실화(잡음·스캔주기·미탐) = 패키지 A. 파일럿 뒤.
- 장애물·스폰 기하 변경. 기존 유지, 위험은 §9.
- Fig9 A* 기준선 재평가(R 28 m 면 웨이포인트 추종 불가 — imo 에서 무효, 논문에 명시).

---

## 3. 프로필 정의

무차원 약속: 길이 L = 14.18 m(콜라이더), 시간 L/U (U 1 m/s → 14.2 s). 논문 표현 = "14 m 선체에 100 m 급 상선의 무차원 동역학(L/V 8–18 s)". MSC.137 적용범위(LOA ≥ 100 m)는 무차원으로 차용.

### 3-1. 식

| | agile (현행, 비트동일) | imo |
|---|---|---|
| yaw [°/s] | `rudder × (speed/max_speed_i) × TURN_FACTOR` | `rudder × speed / (MAX_TURN_RATE·DEG·R_FULL)` |
| 정상 선회반경 | `max_speed_i/(30·1.5·DEG)` = 1.27·max_speed_i m → **배별로 다름** | **R_FULL = 2 L = 28.37 m 전 선박 동일**, 속도 무관 (Nomoto r = K'(V/L)δ 와 동형) |
| 타 슬루 | `RUDDER_RATE 12` °/s | 3 °/s (SOLAS II-1/29: 35°→−30° 28 s ≈ 2.3 °/s 최소. 상선 전형 2.3–3.5) |
| 최대 타각 | 30° | 30° (불변 — obs[365]·보상 정규화 상수) |
| 감속 DECEL | 0.04 | 0.004 |
| 저항 DRAG_COEF | 0.1 | 0.005 |
| 가속 ACCEL | 0.1 | 0.01 |
| obs[363] 분모 `MAX_YAW_RATE` | 45 (= 30 × 1.5) | `30 × 1.8 /(30·DEG·R_FULL)` = 3.64 °/s (함대 최고속 1.8 기준 전타 yaw). **파생값으로 정의**, 리터럴 금지 |

- imo 식은 `_substep` 분기 1개(`vessel_gym.py:434-437`) + C# `UpdateDynamics`(`VesselDynamics.cs:127-136`) 분기 1개. agile 분기는 옛 코드 그대로.
- `MAX_YAW_RATE` 가 파생값이면 agile 에서 obs[363] = rudder_n × speed_ratio 로 값·의미 비트동일(반박 검증 confirm). imo 에서는 obs[363] = yaw/3.64 → 분포 바뀜 = from-scratch(어차피 전이함수 변화로 from-scratch).
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
- 예상 1차 효과 = **충돌률이 아니라 효율**(fuel · headTravel · 타각 사용 · 궤적 직진도). COMM_PLAN §3-3 "perpair 벌점 54–58 % 가 56–200 m 밴드" 와 정합 — 둔한 배는 그 밴드 정보 가치가 큼.
- 정지 우회 차단(§3-1) 없이는 격차가 더 얇아짐(정지 5 m).
- 이 수치는 단일 쌍 기하. 16척 + 장애물 3×3 에서는 제3선·장애물 제약으로 필요 runway 가 더 큼(정량 미확인 → 파일럿).

---

## 4. 예상 효과·성공 기준 (사전등록)

파일럿(§8) 결과 보기 전에 고정:

| 지표 (ground-truth, `eval_ckpt.py`) | ON vs OFF 판정 |
|---|---|
| vColl + oColl | ON ≤ OFF (H1a 필수). ON > OFF 면 구현 버그로 취급 |
| goal % | ON ≥ OFF − 시드 산포 |
| fuel · headTravel · rudderVar | ON < OFF 가 **주 가설**(H1b) |
| 시드 승패 | 3시드 중 2승 이상, 시드별 표 병기 |
| RANDOM | ON > RANDOM 이어야 "정보" 효과. RANDOM ≈ ON 이면 구조 효과 = 통신이라 부르지 않음 |

- 결과 나온 뒤 지표·기준 변경 금지(rigging). 변경하면 문서에 이력 남김.
- 시드 제외는 통신에 불리한 방향으로만, 코드 먼저 의심(`COLLAPSE_ROOTCAUSE.md`).

---

## 5. 보상 상수 재산정 규칙 (imo 만, 양 팔 동일)

원칙: **가중치(계수) 불변. 물리 스케일에서 파생된 기준 시간·거리만 같은 파생 규칙으로 재계산.** 규칙 = "agile 값 × (imo 회피소요 / agile 회피소요)" 등 명시적 비율. 결과 보기 전에 이 표로 고정. 보상 정의 변경이라 루트 CLAUDE.md §1 저자 승인 사항(2026-09-19 결정 3 으로 승인, 값은 구현 시 재확인).

시간 배율 k_t = T_lat12(imo)/T_lat12(agile) = 32.0/14.1 = **2.27**.

| 상수 | 위치 | agile | 파생 규칙 | imo | 왜 |
|---|---|---|---|---|---|
| `TCPA_RISK_DENOM` | `vessel_gym.py:119` | 30 s | × k_t | 68 s | 위험 0.5 시점이 회피 가능 시점보다 앞서야 shaping 기울기가 살아 있음. C# 주석 근거(감지창)는 "기동 ≪ 창" 전제 |
| `RULE_17B_TIME / 17C_TIME` | `:84-85` | 7 / 3.5 s | × k_t | 16 / 8 s | 7 s 는 타 슬루(10 s)도 못 끝냄 |
| `RULE_17B_DIST / 17C_DIST` | `:86-87` | 18 / 9 m | 시간 × closing(2.6 m/s) 파생 → × k_t | 41 / 20 m | 18 m < R 28 m 물리 무의미 |
| `EARLY/SUBSTANTIAL_ACTION_TIME` | `:82-83` | 21.5 / 11.5 s | × k_t (기본 비활성, 문서 정합용) | 49 / 26 s | Rule 16 단계 게이팅 |
| earlyAvoid 게이트 | `:884` `max_risk_near>0.1` (56 m) | near_risk | → `risk`(reward_range 300 m) 기반 | 게이트 반경 확장 | imo 회피는 56 m 밖에서 일어나야 하는데 거기서 보상 0 = 설계 목적과 충돌 |
| 저속 게이트 | `:734` `near_risk<0.1` | 56 m | → `risk` 기반 | 동일 | 56 m 밖 감속이 정당 |
| `CMD_MISMATCH_COEF` 적용 | `config.py:466`, `:925-926` | 0.03 | 계수 유지, 미스매치를 **달성 가능 슬루로 정규화**: `max(0, |cmd−rudder| − RR·0.4)/30` | 식 변경 | 타속 느림을 벌하면 물리를 벌하는 것. 전타당 벌점 −0.94 → −3.75 로 4배 |
| 근접 THR | `:739` 19.6 m | RADAR_BASE×0.35 | **유지** | 19.6 m | 센서 유래. R 28 m 보다 작아 조우마다 벌점 가능하나 γ 0.99 지평(≈40 s) 안에 T_lat12 32 s 있어 credit 전달됨. 유지하고 파일럿에서 분포 확인 |
| `DCPA_RISK` / `SAFE_PASSING` | `:120`, `:88` | 24 / 12 m | 센서·규정 유래 → **유지** | 동일 | 레이더 불변 |
| `GOAL_REACHED` | `:62` 3.0 m | 0.21 L | → L/2 | **7.1 m** (저자 재확인) | 3 m 원 오버슛 시 178 m 루프 = 446 결정 손실. 도착 판정 변경이라 §1 승인 |
| `MAX_EPISODE_STEPS` | `config.py:473` 4500 결정 | k 몬테카를로(agile) | **imo OFF trunk 로 k 재측정 후 결정**(`--drain`) | 미정 | 추정 상향 금지(§2 원칙). 측정 스크립트 저장소에 없음 → 새로 |
| smoothness · progress · angle · time · fuel · forward · 종료보상 | | | **불변** | | 가중치 |

- 상황판정 반경 `DETECTION_RANGE` 56 (`:115`, `:589`) · MoE 라우팅 · obs[368]: **불변**(C# `COLREGS_DETECTION` 동기 필수, 단독 변경 금지). imo 에서 "조우 판정 시점에 회피 여지 없음" 은 해석 단서로 논문에 명시.
- C# 미러: 같은 상수가 `GlobalScale.cs:82-93` const → `COLREGsHandler.cs` 인라인. imo 값을 C# 도 받아야 sim2sim 판정 정합(§6-3).

---

## 6. 구현 터치포인트

### 6-1. Python 정본·sim

| 파일 | 위치 | 할 것 | 패턴 근거 |
|---|---|---|---|
| `config.py` | :454-480 vessel_gym 상수 절 | `DYN_PROFILE = _env_str('VESSEL_DYN_PROFILE','agile').lower()` + assert. 프로필 표 dict `DYN_PROFILES = {'agile': {...}, 'imo': {...}}` → `TURN_FACTOR / R_FULL / MAX_YAW_RATE(파생) / RUDDER_RATE / ACCEL / DECEL / DRAG_COEF` + §5 보상 상수 파생. **한 함수 `dyn_constants(profile)`** 로 묶어 ckpt_io 가 재사용 | :456-469, :492-493 |
| `config.py` | :504-514 `YUGIOH` | `'VESSEL_DYN_PROFILE': 'agile'` | :510 |
| `config.py` | :503 `CODE_VERSION` | 유지. 스냅샷 `dyn_profile` 로 구분 | 결정 5 |
| `vessel_gym.py` | :29-40 | 리터럴 → `_cfg.*` import(모듈 속성 이름 유지 — `corridor_run.py:68-76` 덮어쓰기 선례 호환). `MAX_YAW_RATE` 여기서 재계산 금지 | :26 주석, :42 |
| `vessel_gym.py` | :434-437 `_substep` | 프로필 dict 의 `formula` 키(`'ratio'` = agile 옛 식 / `'abs'` = imo)로 분기: `yaw = rudder·speed/(30·DEG·R_FULL)`. agile 분기 코드는 글자 그대로 유지 | — |
| `vessel_gym.py` | :82-92, :119-120, :62 | §5 상수도 `_cfg` 경유(프로필별) | |
| `vessel_gym.py` | :884, :734 | 게이트 위험 선택을 상수화(`GATE_RISK = 'near'|'reward'`) — agile 'near' 비트동일 | |
| `vessel_gym.py` | :925-926 | cmd_mismatch 정규화 식 분기(agile 옛 식) | |

### 6-2. 스냅샷·복원·검사 (기존 구멍 수리 포함)

현행: `ckpt_io` 는 sim 상수를 **하나도 복원·대조 안 함**(`radar_range` 기록만 :73, DROPOUT·LOS_GATE 기록 없음). `diag_ckpt.py:153-154` config_match 게이트 `pass=True` 하드코딩. 학습기 trunk 분기 검사(`vessel_gym_train.py:575-585`)·`check_branch.py:92-99` 는 seed·steps·dim 만. 평가 3곳(`eval_mixed:123`, `measure_regimes:72`, `corridor_run:87`)은 env 를 restore 보다 먼저 생성.

| 파일 | 할 것 |
|---|---|
| `ckpt_io.snapshot_config` :34-82 | 키 추가: `dyn_profile` + 파생 숫자(`turn_factor / r_full / max_yaw_rate / rudder_rate / accel / decel / drag_coef`) + §5 상수 값 + `radar_dropout_p / los_gate`(기존 구멍). 삭제·의미 변경 없음 |
| `ckpt_io.restore_policy` :113-333 | 스냅샷 `dyn_profile`(없으면 리터럴 `'agile'` 폴백 — YUGIOH 기본이 나중에 바뀌어도 구 ckpt 오염 방지) vs `cfg.DYN_PROFILE` 대조 → 불일치 **중단**, `allow_dyn_mismatch` 로만 우회(`allow_comm_range_mismatch` 와 **별도 플래그** — 평가 4곳이 `VESSEL_ALLOW_COMM_RANGE_MISMATCH=1` 공유 중). 같은 자리에서 `radar_range` 도 대조(기존 구멍) |
| `ckpt_io.make_env_from_snapshot` :339-385 | env 생성 **전** `vg.*` 동역학·보상 상수를 스냅샷 값으로 덮어씀(`dyn_constants()` 재사용) + `used` 로그에 `dyn=` | 
| `ckpt_io.describe / _SNAP_TO_ENV` :388-418 | `dyn_profile` 행 |
| `vessel_gym_train.py` :575-585 | trunk 검사에 `_prev_snap['dyn_profile']`(없으면 'agile') == `cfg.DYN_PROFILE` 추가. 시작 로그 1줄 |
| `verify/check_branch.py` :41-49, :90-99 | 같은 trunk 묶음 안 `dyn_profile` 일치 |
| `eval/diag_ckpt.py` :153-154 | config_match note 를 실제 대조 결과로 |
| `eval/eval_mixed.py` :123, `measure_regimes.py` :72, `corridor_run.py` :87 | env 생성을 restore 뒤로 또는 `make_env_from_snapshot` 경유 |
| `run_repro.sh` :88-120 common_env / :137-141 preflight names / `smoke_mac.sh` :37-69 | `VESSEL_DYN_PROFILE=agile` export + 드리프트 names |
| `verify/test_golden.py` :189-194 `_YUGIOH_CONSTS` | `'DYN_PROFILE'`. 케이스 5개는 agile → 비트동일 자동 검증. **imo 골든은 `--regen` 승인 별건** |
| `verify/test_vessel_gym_fidelity.py` :100, :112 | 리터럴 45 / 0.004 → `vg.MAX_YAW_RATE` / `vg.ACCEL*vg.DT`. imo 케이스 추가 |
| `.claude/CLAUDE.md` §5·§7·§8 | 토글 행, "스냅샷이 유일 근거", 스냅샷 키 목록 |

### 6-3. C# (sim2sim 판정관, 재빌드 필수)

| 파일 | 할 것 |
|---|---|
| `Agent/VesselAgent.cs` :357-368 옆 | `VESSEL_DYN_PROFILE` env 읽기 → `vesselDynamics.profile` 세팅 + `rudderRate / accelerationRate / decelerationRate / dragCoefficient` 덮어쓰기. 기존 `VESSEL_RUDDER_RATE / VESSEL_MAX_TURN_RATE` 와 동시 지정 시 **프로필 우선, 경고**. GlobalScale const 무수정 |
| `Agent/VesselDynamics.cs` :127-136 | imo 분기: `yawRate = rudder × speed / (maxTurnRate·DEG·R_FULL)`. `MaxYawRate` 프로퍼티(:44)도 프로필 분기 → obs[363] 자동 추종 |
| `GlobalScale.cs` :82-93, :150 | §5 상수 imo 값은 const 로 못 바꿈 → `COLREGsHandler` 가 읽는 값을 인스턴스/정적 필드로 옮기고 Initialize 에서 프로필 적용. 주석 "권장 12(허용 8~18)" 갱신 |
| 프리팹·씬 | 무수정(DecisionPeriod 10 유지) |
| 검증 | `_verify_ppo_mirror.py`(Windows) · fidelity 비교(Python vs Unity 궤적, `test_vessel_gym_fidelity`) imo 로 1회 |

- 미러 3곳(`comm_gather / _get_others_msg / evaluate_actions`)·`_verify_comm_mirror.py`: **무관**(obs 계약·집계 불변).
- Unity C# `COMM_RANGE` 420 vs Python 300 기존 불일치 — 판정관 정합 점검 시 같이(별건).

---

## 7. 검증 계획 (구현 단계 게이트)

1. `verify/test_golden.py --check` 5케이스 ALL PASS(agile 비트동일) — env 없이.
2. `VESSEL_DYN_PROFILE=imo` 로 `test_vessel_gym_fidelity`(파라미터화 후) + `dyn_profiles.py` 표와 sim 실측(정상 선회직경 4.00 L, T_lat12 32 s, 정지 70 m) 일치.
3. `run_repro.sh preflight` imo — 드리프트·미러 ALL PASS.
4. 스냅샷 왕복: imo 로 2 update 학습 → `ckpt_io.py X.pt` describe 에 `dyn_profile=imo` → agile env 에서 restore → **중단** 확인 → `allow_dyn_mismatch` 로만 진행.
5. trunk agile → 갈래 imo 시도 → 학습기 거부 확인. `check_branch.py` 불일치 FAIL 확인.
6. C# 재빌드 후 Unity 1 에피소드 궤적 vs Python 동일 명령열 → 위치 오차 ≤ 기존 fidelity 허용.
7. smoke(`smoke_mac.sh` / `run_repro.sh smoke`) imo.

---

## 8. 실험 계획

1. **trunk**: imo · OFF · 3시드 · 9,043,968 결정(138 update). 분기점은 우선 agile 과 같은 스텝 — 수렴 늦으면 OFF 도착률 고원 기준으로 재정의(문서화).
2. **k 재측정**: trunk 로 `--drain` 평가 → 시간계수 k → `MAX_EPISODE_STEPS` 확정(§5). 필요하면 trunk 재학습.
3. **갈래**: OFF / ON / RANDOM(sd = imo trunk `om_sd` 실측으로 재보정) × 3시드. §8-1 분기 규약·`check_branch` 그대로.
4. **평가**: `eval_ckpt.py` ground-truth, §4 기준. Unity 판정관 1회(재빌드 뒤).
5. **통과 시 dose-response**: x = r_d/R. 고정 imo 에서 `RADAR_RANGE` ∈ {56, 84, 112, 168} **위로** 스윕 → r_d/R = 2, 3, 4, 6. 예상: 이득 → 0. "통신 필요 없을 땐 이득 0" 곡선 = anti-rigging 그림. (`RADAR_RANGE` > 56 상한 가드 없음, 보상 THR 는 56 고정이라 불변 — 단 obs 정규화 바뀜 = 레벨마다 trunk 별도.)
6. 붕괴 검출기(`COLLAPSE_ROOTCAUSE.md` 기준 A 700·leaky 0/80)·`diag_ckpt` 조우율 5 % 게이트 = agile 실측 → imo 첫 3시드는 검출기 OFF, 실측 후 재보정.
7. 텔레메트리 추가 1개: 56–300 m 밴드에서의 회피 행동량(Δheading·Δthrust) — "통신이 그 밴드에서 일하나" 직접 지표(구현 시 `comm_telemetry` 에 추가, 지표 정의 승인).

기존 12런·그림 전부 = agile regime. imo 는 **대체가 아니라 별도 regime**. 비교 불가 명시.

---

## 9. 위험

| 위험 | 근거 | 대응 |
|---|---|---|
| OFF 학습 실패(장애물 충돌 국소최적) | `vessel_gym.py:905-912` "직진하다 박기 70배 유리", off_s45, 0.33M oColl 76 % 스파이크 | R 28 m 로 스파이크 길어질 수 있음. 파일럿 OFF 가 안 배우면 TD 3 L 완화(IMO 안) → 문서화. 보상 재산정(§5) 이 완충 |
| "0.3 스케일 78 % 학습불가" 전례 | `GlobalScale.cs:21` 주석 한 줄. 같은 커밋 `ae0e315` 에 timeout 78 %·오분류 80 % 혼재 → **측정 방식 불명** | 인용 시 단서 필수. 근거로 안 씀 |
| 기하 함정 | 스폰~벽 42.4 m(선체 끝 기준) = 1.5 R; 장애물 통과폭 80 m vs 2R 57 m; 목표 3 m 루프; 스폰 간격 100 m vs TD 57 m | §5 GOAL 7.1 m. 나머지 파일럿에서 outcome 분해로 확인. 기하 변경은 범위 밖 |
| 시간초과 재발 | k 몬테카를로 agile 기준 | §8-2 재측정 후 설정 |
| 격차가 얇아 ON ≈ OFF | §3-3 W/T 0.9–1.2 | 정직하게 보고. 효율 지표가 주 가설. r_d/R 곡선으로 "0 이득" 끝점 포함 |
| 프로토콜 학습 실패 | tanh 포화·게이트 흡수 전례 | 기존 `msg_ln`·×0.1 init 유지. 텔레메트리로 확인 |
| 스냅샷·복원 구멍 | §6-2 | 같이 수리. `allow_*` 플래그 분리 |
| Dropbox `.git` 공유 충돌 | 09-11 사고 | Windows 는 GitHub clone 에서 `feat/dyn-profile-imo` pull |

---

## 10. 미결 (구현 중 저자 확인)

- `GOAL_REACHED` 7.1 m 값(§5).
- `MAX_EPISODE_STEPS` — 측정 후.
- 근접 THR 19.6 유지 여부 — 파일럿 분포 보고 후.
- 분기점 스텝 vs 고원 기준.
- 밴드 회피 텔레메트리 정의.
- imo 골든 케이스 추가(`--regen` 승인).

---

## 11. 이력·버전

- `v2026-09-15-branch-protocol` (main `6de6816`) = agile 최종판 + §8-1 분기 규약. GitHub 반영.
- `feat/dyn-profile-imo` = 이 스펙 + 구현. main 머지 조건: §7 게이트 전부 + 파일럿 1회.
- `CODE_VERSION 'YUGIOH-2026-09-10'` 유지. 체크포인트 구분 = 스냅샷 `dyn_profile`.
- 09-04 COMM_PLAN §5 "레이더 축소 regime 기각" → 이 스펙 §1-3 근거로 **동역학 현실화로 대체**(레이더 축소 자체는 여전히 안 함).

## 부록 — 근거 스크립트 (`2026-09-19-dyn-profile-imo/`)

| 파일 | 내용 |
|---|---|
| `dyn_profiles.py` + `.out` | `_substep` 식 그대로 옮긴 순수 파이썬. 프로필 × max_speed × RR 전 조합: 선회·advance·초기선회·횡변위·W/T |
| `indep_check.py` | 독립 재구현(dt 0.001) — 0.18 L·8.97 m·T_lat12 31.93 s 재현 |
| `imo_check.py` | zig-zag 오버슈트·정지거리 검산 |
| `stop_lat6.py` | 정지거리 후보(DECEL/DRAG) 표 · T_lat6/12/14 (절대속도 식) |

실행: `python3 <파일>` (numpy·torch 불요, Mac OK).
