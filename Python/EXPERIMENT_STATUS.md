# 실험 현황 핸드오프 — 2026-06-02 갱신

> **상태: ONE 빌드 6-run sweep 완료 → 게이트 fix 검증됨 + rush>avoid 붕괴 근본원인 확정(EV 평형).**
> **rush-fix 레버를 C#에 구현(전부 default-OFF/identity, anti-rigging 대칭) → 재빌드 1회 → `run_sweep_rushfix.ps1` 대기.**
> 이 문서 하나로 clear 후 바로 이어갈 수 있게 정리. 분석 도구·런처·다음 단계 전부 포함.

---

## 0. 한 줄 요약 (2026-06-02)
ONE 빌드 수렴결과: **comm-OFF가 깊은 수렴에서 rush>avoid로 붕괴(vColl 5.1~6.2%, 3/3 LATE_COLLAPSE)**, comm1fix(게이트)는 이를 *완화*(vColl 2.8~4.6%, 0/3 LATE, OFF를 Pareto로 이김=H1a 회복). **남은 진짜 병목은 통신 아닌 보상**: 충돌 ~6%는 soft-penalty의 **EV 평형** `p_eq=detour_cost/(300·γ^D)≈6%`(버그 아닌 수학). 0601 재설계는 path-dependent 소액만 깎아 평형을 못 움직임. → **rush-fix 레버 구현(arrival-debulk·dense-steepening·prox-ramp·speed-unlock, 전부 env 토글, ON/OFF 동일=anti-rigging). 재빌드+검증sweep이 다음.**

### ⚠️ 정직한 한계 (워크플로우 18-agent 합의, 반드시 인지)
- **어떤 레버도 EV floor를 0으로 못 만든다.** p_eq를 *확실히* 낮추는 유일 레버는 −300 확대(예 −900)인데 그건 **기각된 timid-policy 레버**(freeze/timeout 회귀). 채택 레버는 detour_cost↓/runway↑로 **6%→~2~4%로 깎되 ~2~3% floor 잔존.** 게이트 효과와 비슷한 규모의 개선.
- **관측 충돌 class = moderate-DCPA sudden-tail**(minVD~40m, nearMiss/step~0.001): risk가 마지막 2~5 결정에서만 급상승 → **perception/timing(runway) 문제라 reward 레버로 완전 해결 불가.** 강한 param에도 vColl 불변이면 = 병목은 runway, reward 튜닝 중단이 정직한 결론.
- **이건 ON/OFF 공통 보상 문제**(통신 결과 아님). fix 후에도 comm-ON이 *강화된* OFF를 ground-truth로 이겨야 통신 가치 주장 가능.

---

## 1. 어디까지 왔나 (검증된 사실, ground-truth 로그 기준)
- **"충돌 4%→60%"의 정체**: ① "4%"는 200k의 *under-trained 소심정책* artifact(수렴값 아님) — 학습할수록 timeout↓·충돌↑(돌진 학습)로 회귀(문서화됨: 200k 3%→1M 16%). ② 0601 빌드에 추가한 **가운데 원통**이 oColl ~34% 새로 추가. → 60% ≈ 배충돌 27% + 원통 34%.
- **통신**: 수렴서 ON(63%)≈OFF(59%), vColl +4%p로 오히려 약간 유해. egocentric ARPA로 충분, 통신=노이즈. **통신은 처음부터 충돌의 원인이 아니었다.**
- **왜 길찾기인데 충돌이 나나(근원적 질문 답)**: 이건 단일 길찾기가 아니라 **멀티에이전트 + soft-penalty + 연속동역학** 게임.
  1. **soft penalty의 EV(제일 근원)**: 충돌은 *금지*가 아니라 −300 감점. RL은 기대보상 최적화라 `돌진이득(확정) > 충돌(−300)×확률(p<1)`이면 *충돌이 최적*. A\*는 hard constraint라 0, RL은 soft라 trade-off.
  2. **멀티에이전트**: 상대도 학습 중→비정상성. shared policy 대칭→머리 맞대면 같은 방향 틀어 서로 박음(COLREGs가 대칭깨기인 이유).
  3. 연속제어+관성+반응지연(늦으면 물리적으로 못 피함), 탐색 노이즈(std>0), credit assignment 모호, 밀도/기하(0.3스케일=78% 데이터).
  → **충돌은 버그가 아니라 보상 설계의 수학적 귀결.** 목표는 0이 아니라 "숙련 COLREGs 항해사 수준, 수렴에서도 유지".

---

## 2. 이번에 한 보상 재설계 (한 빌드, 전부 env 토글)

### ★진단 정정 (워크플로우 권고를 코드로 까서 기각한 것)
- 워크플로우: "progress가 non-PBS라 circling farming → γ=0.99 PBS로 교체".
- **코드 확인 결과 기각**: 현재 progress(`prevDist−currDist`)는 **이미 γ=1 potential-based(telescoping)** → 우회 net=0, farming 없음. γ=0.99 넣으면 정지 시 `0.01×거리`(≈+1.5/step) 양수 → **멀리서 가만히 떠 있는 게 최적**이 되는 farming 버그. → progress 안 건드림.
- **진짜 'rush>avoid' 원인** = path-dependent 항(angleReward·직진보너스·timePenalty)이 회피 변침/detour를 직접 벌함. (progress total은 path-independent=dist_start라 무관.)

### 변경 (VesselAgent.cs, 기본값=새 설계, env override 가능)
| 항목 | old→new | env | 왜 |
|---|---|---|---|
| angleRewardCoef | 0.5→**0.15** | `VESSEL_ANGLE_COEF` | 회피 변침(헤딩 돌림) 페널티 완화 |
| 직진보너스 | on→**off** | `VESSEL_STRAIGHT_BONUS` | rudder≈0 보상이 회피 변침과 직접 충돌 |
| timePenalty | -0.1→**-0.07** | `VESSEL_TIME_PENALTY` | detour 시간비용 완화(−0.03 밑 금지=loiter) |
| earlyAvoid 게이트 | 0.3→**0.1** | `VESSEL_EARLY_RISK_GATE` | 근접에서도 DCPA-증가 회피보상 |
| earlyAvoid tcpa게이트 | 11.5s→**제거** | `VESSEL_EARLY_RELAX_TCPA` | any tcpa에서 회피 보상 |
| earlyAvoid ×speedRatio | 신규 | — | 정지로 게임 못 함(전진 회피만 보상), orbit=보상0 |
| progress / proximity / collisionCourse | **불변** | (`VESSEL_COLCOURSE_COEF`) | telescoping이라 OK / 접근중만 발화 |

**인과**: 회피 detour 비용이 "몇 step 시간페널티"로 작아짐 → `EV(회피)>EV(돌진)` → 충돌↓ 수렴 유지. earlyAvoid가 DCPA증가에만+×speedRatio → orbit·freeze 억제.

### 통신 H1a 진짜 버그 수정 (networks.py)
- **버그**: ControlActor·Critic의 `fc2` 메시지 슬라이스가 **random-init**(zero-init 아님) → comm-ON이 comm-OFF와 *다른 출발선*에서 시작(value-of-information≥0 전제 위배).
- **수정**: `fc2.weight[:, -msg_dim:].zero_()` → comm-ON을 OFF와 같은 출발선에. 도움될 때만 0에서 자람. comm-OFF 완전불변(anti-rigging 100% 안전).
- 보조: `MSG_LR_SCALE 3.0→1.0`(`VESSEL_MSG_LR`). H1a 시험 시 `VESSEL_AGG_MODE=mean`+`VESSEL_MSG_L2=0.01` 권장.

---

## 3. 파일 변경 목록 (이번 세션)
- `Agent/VesselAgent.cs`: 보상 토글 필드+Initialize env, CalculateColregsReward 재구성(게이트 0.3→0.1, COLREGs준수는 maxRisk>0.3 블록 유지, earlyAvoid 완화+speedRatio), 직진보너스 게이트, **진단 instrumentation**(near-miss/circling), METRIC 9→**13열**, **events 로그**(VESSEL_EVENT_LOG).
- `Python/networks.py`: ControlActor·Critic fc2 zero-init.
- `Python/config.py`: MSG_LR_SCALE env 1.0.
- `GlobalScale.cs`: `NEAR_MISS_DIST = SAFE_PASSING`(6m).
- 신규 분석/런처: `convergence_gate.py`, `analyze_circling_safety.py`, `analyze_run.py`, `run_experiment.ps1`, `run_sweep.ps1`.
- `.claude/CLAUDE.md`: env 표 + 13열 메트릭 + 보상재설계 노트 갱신.

---

## 4. 로그 (전부 build\results\<timestamp>_<run>\ 에 자동 정리)
| 파일 | 내용 |
|---|---|
| `outcome.csv` | agentId,ep,outcome,steps |
| `metric.csv` (13열) | …,fuel,rudderVar,compliance,occlRate,commandVar,**minVesselDist,nearMissSteps,straightness,headingTravel** |
| `events.csv` | id,ep,outcome,step,**startX,startZ,endX,endZ**,heading,speed (충돌/도착 공간분포=궤적) |
| `run_meta.txt` | 그 run의 정확한 VESSEL_* 설정 스냅샷 |

⚠️ **straightness/minVesselDist/nearMiss/headingTravel는 진단 전용 — 보상에 절대 비연결**(제약3: spinning 탐지-킬 금지).

---

## 5. 돌리는 법
```powershell
# 1) Unity 재빌드 필수 (C# 변경). 예: Build\0601_reward\Vessel_MLAgent.exe
# 2) 6병렬 from-scratch sweep (OFF/sum × 3seed + ON/mean × 3seed)
cd <repo>\Assets\Scripts\Python
.\run_sweep.ps1 -Exe "C:\Users\sengh\Dropbox\Private_Paper_Project\0702_NewVessel\Build\0601_reward\Vessel_MLAgent.exe"
# (단일 run: .\run_experiment.ps1 -Exe <exe> -Seed 42 -Comm 0 -Port 5042)
```
성능: latency-bound. 이 머신 최적 = **6병렬 × NUM_ENVS=1**(NUM_ENVS=2는 2배 느림).

## 6. 분석하는 법 (학습 후 '바로')
```powershell
python analyze_run.py "<build>\results"          # ★통합 리포트(수렴 outcome+fuel+circling+near-miss+충돌공간분포+LATE_COLLAPSE)
python convergence_gate.py <metric.csv ...>      # 수렴게이트 단독(5등분 추세+LATE_COLLAPSE)
python analyze_circling_safety.py                # circling/near-miss 상세(condition별)
```
**판정 규율**: 마지막 30%(수렴)만 본다. `LATE` 플래그면 아직 악화 중 → 더 학습. mid-training 좋아도 채택 금지(과거 4번 속음).
**통신 판정(H1a)**: ON이 vColl·oColl·fuel/s·nearMiss/s에서 OFF를 *초과(나쁨)*면 = 정보가치≥0 위배 = 버그.

---

## 7. 다음 단계 — rush-fix 검증 (2026-06-02)

### 즉시: 재빌드 1회 → wave-1 sweep
```powershell
# 1) Unity 재빌드 필수 — arrival은 Initialize에 baked, 새 env(ARRIVAL/GOAL_COEF/COLCOURSE_EXP/PROX_RAMP/SPEED_AVOID_UNLOCK)는 재빌드 전엔 무시됨.
#    새 폴더 권장: Build\0602_rushfix\Vessel_MLAgent.exe (기존 ONE 결과와 분리)
# 2) wave-1: arrival-debulk + dense-steepening, OFF+ON × 3seed (6병렬)
cd <repo>\Assets\Scripts\Python
.\run_sweep_rushfix.ps1 -Exe "C:\...\Build\0602_rushfix\Vessel_MLAgent.exe"
# 3) 수렴 후: python analyze_run.py "<build>\results"
```
**판정(수렴=마지막30%, 3seed만)**: 성공=vColl 6%→3~4%, LATE 0/3, goal≥83~86%, TO≤10~11%, straightness 0.85~0.87.
**falsify**: 강param에도 vColl 불변 → 병목=perception runway → reward 튜닝 중단(정직한 결론). circling(straight<0.80/headTrv급증) → 그 param cell kill.

### rush-fix 레버 (전부 C# 구현완료, default-OFF/identity, env 토글, ON/OFF 동일=anti-rigging)
| 레버 | env | wave-1 값 | 표적 | 비고 |
|---|---|---|---|---|
| arrival-debulk | `VESSEL_ARRIVAL_REWARD` / `VESSEL_GOAL_COEF` | 45 / 1.35 | near-goal rush(EV 지배항) | goalCoef=(260-arrival)/160. **1.5는 과보전→레버 상쇄(주의)** |
| dense-steepening | `VESSEL_COLCOURSE_EXP` / `VESSEL_COLCOURSE_COEF` | 1.6 / -1.2 | moderate-DCPA sudden-tail | risk^p로 56m 포화 제거(거리 gradient 복원). head-on(risk=1포화)엔 EXP 무효→COEF가 head-on 레버 |
| prox-ramp (wave-2) | `VESSEL_PROX_RAMP_COEF` / `_DIST` | -2.5 / 24 | sudden-tail 최종접근 | **버그수정완료**(다선 정합성: sep=위험선박 실거리로 통일, LOS-게이트=occluded 미발화). 저분산 거리×접근율 gradient |
| speed-unlock (wave-2) | `VESSEL_SPEED_AVOID_UNLOCK` / `_GATE` | 1 / 0.3~0.4 | Rule-8 감속회피 | lowSpeedPenalty ×0.3 완화(완전면제X=freeze 차단). turn-only 강제 해소 |

### 후속
1. wave-1 성공 시 → 통신 H1a 재확인(comm-ON이 강화된 OFF를 이기나) → H1b(occlusion regime, `VESSEL_LOS_GATE=1`).
2. wave-1 불충분 시 → wave-2 prox-ramp(-2.5) 또는 speed-unlock 단독 ablation(synthesis ablation order 참조).
3. circling 시: COLCOURSE_COEF/EXP 약화. **동역학으로 풀지 말 것**(평형점만 이동). oColl 몰리면 원통 빼고 순수 배-배부터.
4. ⚠️ 모든 레버는 **상호 confound** — wave-1은 둘 묶음(엔지니어링 우선). 과학적 per-lever credit 필요시 단독 cell.

### 기각된 접근 (다시 시도 금지)
- γ=0.99 PBS(정지 farming 버그), 대칭 DCPA 페널티(freeze 재발), starboard dense 비대칭(반대방향 orbit), MAPPO(zero-init 후 한계효용↓, 보류), spinning 탐지-킬(과거 25% 오판 실패).

---

## 8. 하드 제약 (제1원칙 — 절대 어기지 말 것)
- **과학적 정직성**: 통신이 이기게 baseline 불구화 금지. comm-OFF를 '못 보는 위협'으로 페널티 금지. **안 도우면 정직하게 "안 도움"이 결론.**
- **anti-rigging**: comm-ON은 *불구화 안 된 최선의* comm-OFF를 ground-truth로 이겨야 진짜.
- **value-of-information≥0**: comm-ON은 절대 OFF보다 나쁘면 안 됨(나쁘면 버그).
- **goalReachedDistance/MaxStep 늘리기 = 치팅**(물리 필요시만).
- **수렴에서만 평가**(마지막 30%). mid-training snapshot 신뢰 금지.
- **spinning 탐지-킬 절대 재도입 금지**(과거 25% 오판). circling은 보상 구조로만.
- **동역학 변경(슬루)→from-scratch 재학습 필수**(옛 모델 호환 X).
