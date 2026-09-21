# VESSEL_DYN_PROFILE=imo — Unity C# 미러 구현 계획 (Windows 전용)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. **컴파일·실행 검증은 Unity 가 있는 Windows 에서만 가능** — Mac 에서는 코드 작성·리뷰까지.

**Goal:** Unity 판정관(sim2sim ground-truth)이 Python `config.dyn_profile_constants('imo')` 와 같은 동역학·보상 상수로 돌고, `VESSEL_OBSTACLES=none` 이면 장애물을 비활성화한다. 기본(env 미설정) = 현행과 동일.

**Architecture:** env 읽기는 기존 규약대로 `VesselAgent.Initialize`(34개 env 와 같은 자리)에서 1회. 선회식 분기는 `VesselDynamics.UpdateDynamics`/`MaxYawRate` 한 곳. COLREGs 시간·거리 상수는 `COLREGsHandler` 의 `const` 를 `static` 필드로 바꾸고 `ApplyDynProfile(kT)` 로 1회 스케일. `GlobalScale` const 는 손대지 않는다(재빌드 범위 최소화 — 단 C# 수정 자체가 재빌드 필수).

**Tech Stack:** Unity 6, C#, ML-Agents. Python 미러 기준 = `Python/config.py dyn_profile_constants`, `Python/vessel_gym.py yaw_rate_deg`.

**Spec:** `docs/superpowers/specs/2026-09-19-dyn-profile-imo-design.md` §3, §5, §6-3

## Global Constraints

- C# PascalCase(클래스/메서드)·camelCase(지역), 주석 한국어, `Debug.Log` 금지(`Debug.LogWarning` 만).
- C# 파일은 git root(`Assets/Scripts`) 원본을 직접 편집·`git add`. 복사본 금지(Unity 중복 컴파일).
- 기본 동작(env 없음) 비트동일: 새 필드 기본값이 옛 식과 같은 값을 내야 함.
- Python 과 숫자 일치: R_FULL 28.36632, RR 3, ACCEL 0.01, DECEL 0.004, DRAG 0.005, k_t 2.27, goal 7.09158, slack 1.2°.
- 커밋 메시지 끝 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

### Task 1: VesselDynamics — imo 선회식·MaxYawRate

**Files:**
- Modify: `Agent/VesselDynamics.cs` :18-21(필드), :44(`MaxYawRate`), :127-136(`UpdateDynamics` 3-5)

- [ ] **Step 1: 필드 추가** (`public float rudderRate = 12.0f;` 뒤)

```csharp
    // ★2026-09-21 동역학 프로필 (VESSEL_DYN_PROFILE=imo, VesselAgent.Initialize 가 세팅). Python vessel_gym.yaw_rate_deg 미러.
    //   imoTurn=false(기본) = 옛 식 rudder·speedRatio·turnFactor 그대로(비트동일).
    //   imoTurn=true  = 절대속도 식 yaw[°/s] = (rudder/maxTurnRate)·speed/rFull·Rad2Deg → 정상 선회반경 rFull 고정(속도 무관).
    public bool imoTurn = false;
    public float rFull = 0f;            // [m] 전타 정상 선회반경 (imo: 2 L = 28.36632)
    public float fleetMaxSpeed = 1.8f;  // obs yaw 정규화 분모용 함대 최고속 (GlobalScale.MAX_SPEED × speedMultMax)
```

- [ ] **Step 2: MaxYawRate**

```csharp
    // 실제 최대 yaw rate (°/s): agile = speedRatio=1 전타 / imo = 함대 최고속 전타 (Python MAX_YAW_RATE 와 동일 정의)
    public float MaxYawRate
    {
        get
        {
            if (imoTurn) return fleetMaxSpeed / rFull * Mathf.Rad2Deg;
            return maxTurnRate * rudderEffectiveness * (10.0f / length) * (beam / 2.0f);
        }
    }
```

- [ ] **Step 3: UpdateDynamics 5단계**

`yawRate = effectiveRudderAngle * turnFactor;` →
```csharp
        // 5. 회전 속도(yaw rate) 업데이트 — 프로필별 식 (Python yaw_rate_deg 미러)
        yawRate = imoTurn
            ? (rudderAngle / maxTurnRate) * currentSpeed / rFull * Mathf.Rad2Deg
            : effectiveRudderAngle * turnFactor;
```

- [ ] **Step 4: 검증(Windows)** — Unity 콘솔 컴파일 에러 0. 에디터 플레이(env 없음)에서 전타 선회율 45°/s 유지(기존 `VesselAgent` 메트릭 로그 또는 `YawRate` 관찰).

- [ ] **Step 5: Commit** `git add Agent/VesselDynamics.cs` / `feat(dyn,unity): VesselDynamics imo 절대속도 선회식·MaxYawRate 분기`

---

### Task 2: COLREGsHandler — const → static + ApplyDynProfile

**Files:**
- Modify: `Navigation/COLREGsHandler.cs` :23-34(상수), :419-420, :469-470(`GlobalScale.TCPA_RISK_DENOM/DCPA_RISK` 직접 참조)
- Modify: `Agent/VesselAgent.cs` :812(`GlobalScale.SUBSTANTIAL_ACTION_TIME`)

- [ ] **Step 1: 상수를 static 필드로**

```csharp
    // ★2026-09-21 동역학 프로필: 물리 파생 시간·거리 상수는 프로세스당 1회 ApplyDynProfile(kT) 로 스케일된다(Python config.dyn_profile_constants 미러).
    //   const → static: 값은 GlobalScale 그대로(기본 비트동일). switch-case 등 const 요구 문맥이 없음을 컴파일로 확인.
    private static float EARLY_ACTION_TIME = GlobalScale.EARLY_ACTION_TIME;
    private static float SUBSTANTIAL_ACTION_TIME = GlobalScale.SUBSTANTIAL_ACTION_TIME;
    private static float RULE_17B_TIME = GlobalScale.RULE_17B_TIME;
    private static float RULE_17B_DISTANCE = GlobalScale.RULE_17B_DIST;
    private static float RULE_17C_TIME = GlobalScale.RULE_17C_TIME;
    private static float RULE_17C_DISTANCE = GlobalScale.RULE_17C_DIST;
    private static float TCPA_RISK_DENOM = GlobalScale.TCPA_RISK_DENOM;
    private static bool dynProfileApplied = false;

    public static float SubstantialActionTime => SUBSTANTIAL_ACTION_TIME;   // VesselAgent earlyAvoid 게이트가 읽음

    /// <summary>VESSEL_DYN_PROFILE=imo: 시간·거리 상수 × kT (2.27). 여러 Agent 가 불러도 1회만.</summary>
    public static void ApplyDynProfile(float kT)
    {
        if (dynProfileApplied) return;
        dynProfileApplied = true;
        EARLY_ACTION_TIME = GlobalScale.EARLY_ACTION_TIME * kT;
        SUBSTANTIAL_ACTION_TIME = GlobalScale.SUBSTANTIAL_ACTION_TIME * kT;
        RULE_17B_TIME = GlobalScale.RULE_17B_TIME * kT;
        RULE_17C_TIME = GlobalScale.RULE_17C_TIME * kT;
        RULE_17B_DISTANCE = GlobalScale.RULE_17B_DIST * kT;
        RULE_17C_DISTANCE = GlobalScale.RULE_17C_DIST * kT;
        TCPA_RISK_DENOM = GlobalScale.TCPA_RISK_DENOM * kT;
    }
```
(`SAFE_PASSING_DISTANCE`·`CRITICAL_CPA`·`DCPA_RISK` 는 스펙 §5 대로 불변 — 그대로 둔다.)

- [ ] **Step 2: 참조 교체** — :419, :469 의 `GlobalScale.TCPA_RISK_DENOM` → `TCPA_RISK_DENOM`. `VesselAgent.cs:812` 의 `GlobalScale.SUBSTANTIAL_ACTION_TIME` → `COLREGsHandler.SubstantialActionTime`.

- [ ] **Step 3: 검증(Windows)** — 컴파일 0 에러. env 없음 플레이에서 보상·상황판정 값이 이전 빌드와 동일(같은 시드 1 에피소드 `VESSEL_METRIC_LOG` 대조).

- [ ] **Step 4: Commit** `git add Navigation/COLREGsHandler.cs Agent/VesselAgent.cs` / `feat(dyn,unity): COLREGsHandler 시간·거리 상수 static + ApplyDynProfile(kT)`

---

### Task 3: VesselAgent.Initialize — env 읽기·프로필 적용·mismatch slack·도착반경

**Files:**
- Modify: `Agent/VesselAgent.cs` :99 근처(필드), :357-368 뒤(env), :834-850(cmd mismatch)

- [ ] **Step 1: 필드** (`public float commandMismatchCoef = -0.03f;` 뒤)

```csharp
    public float cmdMismatchSlackDeg = 0f;   // ★2026-09-21 imo: 결정당 달성 가능 슬루(RR×0.4°)는 벌하지 않음. 0 = 옛 식 그대로
```

- [ ] **Step 2: env 블록** (`VESSEL_MAX_TURN_RATE` 블록 :366-368 바로 뒤)

```csharp
        // ★2026-09-21 VESSEL_DYN_PROFILE=imo — Python config.dyn_profile_constants('imo') 미러 (스펙 §3·§5).
        //   기존 VESSEL_RUDDER_RATE/VESSEL_MAX_TURN_RATE 보다 나중에 적용 = 프로필 우선(동시 지정 시 경고).
        string dynProfile = (System.Environment.GetEnvironmentVariable("VESSEL_DYN_PROFILE") ?? "agile").Trim().ToLowerInvariant();
        if (dynProfile == "imo")
        {
            const float shipLen = 14.18316f;                    // 콜라이더 길이 L (Python SHIP_LEN_M)
            const float kT = 2.27f;                             // 시간 배율 (Python DYN_K_T)
            vesselDynamics.imoTurn = true;
            vesselDynamics.rFull = 2f * shipLen;                // 28.36632
            vesselDynamics.fleetMaxSpeed = GlobalScale.MAX_SPEED * speedMultMax;
            vesselDynamics.rudderRate = 3f;
            vesselDynamics.accelerationRate = 0.01f;
            vesselDynamics.decelerationRate = 0.004f;
            vesselDynamics.dragCoefficient = 0.005f;
            goalReachedDistance = shipLen / 2f;                 // 7.09158
            cmdMismatchSlackDeg = 3f * 0.4f;                    // 1.2°
            COLREGsHandler.ApplyDynProfile(kT);
            if (!string.IsNullOrEmpty(envRudderRate) || !string.IsNullOrEmpty(envMaxTurn))
                Debug.LogWarning("[dyn] VESSEL_DYN_PROFILE=imo 가 VESSEL_RUDDER_RATE / VESSEL_MAX_TURN_RATE 를 덮어씀");
        }
        else if (dynProfile != "agile")
        {
            Debug.LogWarning($"[dyn] VESSEL_DYN_PROFILE={dynProfile} 는 모름 (agile|imo) - agile 로 진행");
        }
```
(`speedMultMax` 가 이 지점보다 앞에서 env 로 읽히는지 확인 — `grep -n "speedMultMax" Agent/VesselAgent.cs`. 뒤라면 이 블록을 그 뒤로 옮긴다.)

- [ ] **Step 3: cmd mismatch slack** — :834-850 을 열어 `saturation` 계산을 찾는다. 형태가 `float saturation = Mathf.Clamp01(Mathf.Abs(commanded - actual) / maxTurnRate);` 류이면:
```csharp
            float mismatchDeg = Mathf.Abs(vesselDynamics.CommandedRudderAngle - vesselDynamics.RudderAngle);
            if (cmdMismatchSlackDeg > 0f) mismatchDeg = Mathf.Max(0f, mismatchDeg - cmdMismatchSlackDeg);   // ★imo slack (Python 12-b 미러)
            float saturation = Mathf.Clamp01(mismatchDeg / vesselDynamics.maxTurnRate);
```
로 바꾼다(변수 이름은 기존 코드에 맞춤, slack 0 이면 수치 동일).

- [ ] **Step 4: 검증(Windows)** — 컴파일 0 에러. `VESSEL_DYN_PROFILE=imo` 로 에디터 플레이: 전타 선회 직경 ≈ 56.7 m(기즈모/위치 로그), obs[363] ∈ [−1, 1]. env 없음: 이전과 동일.

- [ ] **Step 5: Commit** `feat(dyn,unity): VesselAgent VESSEL_DYN_PROFILE=imo 적용(선회·슬루·가감속·저항·도착반경·mismatch slack)`

---

### Task 4: open-sea — VESSEL_OBSTACLES=none 씬 장애물 비활성

**Files:**
- Modify: `Management/VesselManager.cs` (env 읽기 :85-106 패턴 옆)
- 씬: `Assets/Scenes/Simulation.unity` — 장애물 오브젝트의 태그/이름 확인 필요(Windows 에서 Hierarchy 검사)

- [ ] **Step 1: 장애물 식별 방법 확정** — Hierarchy 에서 장애물(캡슐) 9개의 공통 부모 또는 태그를 확인. 태그가 없으면 `Obstacle` 태그를 만들어 9개에 붙인다(씬 변경 = 커밋 대상).

- [ ] **Step 2: VesselManager.Awake (또는 기존 env 읽는 위치)**

```csharp
        // ★2026-09-21 VESSEL_OBSTACLES=none — open-sea: 장애물 9개 비활성 (Python vessel_gym OBSTACLES_MODE 미러). 기본 grid3x3 = 그대로.
        string obstMode = (System.Environment.GetEnvironmentVariable("VESSEL_OBSTACLES") ?? "grid3x3").Trim().ToLowerInvariant();
        if (obstMode == "none")
        {
            var obstacles = GameObject.FindGameObjectsWithTag("Obstacle");
            foreach (var o in obstacles) o.SetActive(false);
            Debug.LogWarning($"[obstacles] VESSEL_OBSTACLES=none - 장애물 {obstacles.Length}개 비활성");
        }
```

- [ ] **Step 3: 검증(Windows)** — `VESSEL_OBSTACLES=none` 플레이에서 레이더 기즈모에 장애물 반응 없음, `VESSEL_OUTCOME_LOG` 에 collision_obstacle 이 벽 외엔 없음. env 없음: 9개 활성.

- [ ] **Step 4: Commit** `feat(dyn,unity): VESSEL_OBSTACLES=none 장애물 비활성(open-sea)`

---

### Task 5: 재빌드 + Python 대조

- [ ] Unity Build → `Build/Vessel_MLAgent.exe`.
- [ ] `VESSEL_DYN_PROFILE=imo` 로 Python `verify/test_vessel_gym_fidelity.py` 의 스칼라 참조와 Unity 1척 전타 궤적(같은 명령열, `VESSEL_LOAD_MODEL=1 VESSEL_TRAIN=0` 경로 또는 `SIM2SIM_HANDOFF.md` 절차) 대조: 60 s 후 위치 오차 ≤ 기존 fidelity 허용치. 결과를 `Python/SIM2SIM_HANDOFF.md` 에 한 줄 기록.
- [ ] `Python/_archive/unity_path_2026-09/main.py` 로 Unity 경로 스모크 1회(임의 짧은 스텝) — obs 369D 트랩 통과.
- [ ] Commit + push.
