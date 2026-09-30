# 보상 v3 + 복원형 latent 사다리 배치(t_) 설계·사전등록 (2026-09-30)

- 상태: **저자 승인**(2026-09-30 plan mode, 결정 4건: ① 판정기 v2 전체 ② 보상 (b)+(c') ③ 그림·지표 정의 고정 ④ Stage A + Stage B 전체).
- 결과가 나오기 전에 고정한 문서임. 결과를 보고 지표·팔·기준·그림 정의를 바꾸지 않음. 불통과도 그대로 보고함.
- 선행 주기: `2026-09-29-latent-sweep-design.md`(s_), `2026-09-29-role-promise-design.md`(r_). 코드: 브랜치 `feat/reward-v3`(9cc8f6d 위, worktree `C:\work\DT_Vessel_v3`).
- 저자 고정 목표(09-29): ① 통신(latent msg)이 COLREGs 준수·DCPA·연료·궤적·goal·충돌 각각에서 명확히 우월 ② reward 그래프 우상향 + 통신이 압도. 방법을 고쳐 도달하며, 지표·시드·그림을 결과에 맞추지 않음.

## 1. 왜 (근거 = 09-26~29 배치 eval·log·코드)
- 이긴 통신은 전부 **복원형**(h_a6 goal 68.0 / vColl 28.9 / 통과거리 28.6 vs x_off 53.6 / 44.7 / 24.0, 3/3). 직접 읽기(c6·onl6·z6·z12)는 4번 다 최선 OFF 에 짐. 단 복원형도 fuel 0/3·headTravel 1/3·레이더 안 C 1/3 — h_a6 절제(msgzero)에서 goal↓ vColl↑ 이면서 fuel↓ head↓ C↑: 메시지는 "더 돌아서라도 피함"에 쓰임.
- 결정당 보상 예산(`vessel_gym.py:1249-1506`): 쉐이핑·위험 항이 종료 항의 ~30배. **안전 통과(d 60 m·tcpa 25 s·dcpa 30 m)도 위험 비용 −1.2/결정**(`_pairwise` 의 dist·tcpa 항이 dcpa 와 무관) → 통신 배가 빙 도는 엔진.
- 역할 약속 −20 은 조우 끝(CPA 3결정 뒤)에 옴(`RolePromiseTracker.update`): 위반과 수백 결정 차이(γ^300≈0.05) → 학습 불가. 엄격 전쌍 기준은 동시 조우 ~3.5개(13.7 조우/배·에피소드)에서 구조적으로 못 지킴 → 전 팔 12–20 %.
- "결정당 reward +0.16 이 천장" 진단(09-29 보고서)은 오류: +0.16 = 첫 update(32결정, 사건 0) 값(`s_trunk_d6_s43.csv:2`). 정상 상태: 학습 OFF −0.85/결정, 규칙 배 vo56 +0.11~0.15 → 여유 +1.0/결정. 곡선은 1.4M(−1.37: 이동 시작 → 충돌 88 %)부터 오름.
- p50 코덱 decode 불가(`comm_codec.py:161-162`) → 복원형은 p6·p12 뿐. 3M 파일럿은 판별력 0(s_ trunk 5개 모두 3M 에서 goal 1.5–3 %·oColl 42–57 %, 갈라짐은 6.2M 뒤).

## 2. Stage A — 탐색 배치(2026-09-30 즉시, 코드 변경 0, 확정 근거 아님)
- `s_a6`·`s_c6` 를 기존 s_ trunk(43·44·45)에서 분기. s_ 스펙은 팔을 off/z6/z12 로 고정했으므로 **사후 추가 팔**임을 명시. 워밍업 1200(s_off 와 동일). 드라이버 `runs/2026-09-30_t/_run_stageA.sh`.
- **예측(런 전 고정)**: s_a6 가 s_off 를 goal·vColl·통과최소거리 중앙 3/3 이김, fuel·headTravel(도착 에피소드 평균)은 짐. s_c6 는 goal·vColl 3/3 못 이김. 판정 = s_ 스펙 §7 규칙(3/3 + 평균차 > N). 절제 = msgzero·shuffle(·decl0), EVAL_DEC 5000(빠른 절제).
- 용도: 같은 trunk·같은 보상에서 복원 vs 직접 확정(복제 확인). t_ 배치의 팔 선택은 이 결과와 무관하게 아래 §5 로 이미 고정함.

## 3. 보상 v3 (모든 팔 동일, config 토글, 기본값 = 옛 동작 비트동일)
| 항목 | 새 키(기본) | 배치 값 | 코드 |
|---|---|---|---|
| 전진 보너스 | `VESSEL_FORWARD_COEF`(0.1) | 0 | `_reward` #2 |
| 시간 벌점 | `VESSEL_TIME_PENALTY`(0.07) | 0.035 | `_reward` #1 |
| 위험 비용 DCPA 게이트 | `VESSEL_RISK_DCPA_GATE_M`(0 = 끔) | 48 | `_pairwise`: `risk_rw = risk · clamp((48 − dcpa)/24, 0, 1)` — `_reward` #6 충돌코스·#7 per-pair 만 `risk_rw` 를 읽음. `near_risk`·situation·danger_idx·eval 지표 불변 |
| 판정기 | `VESSEL_ROLE_JUDGE`('end') | v2 | `RolePromiseTracker` |

**판정기 v2 규칙**(시작 조건은 그대로: ≤300 m·접근·dcpa < 24 m·양쪽 역할 > 0, 보완 규칙 포함)
1. F1 양보·정면 배 좌현 Δψ < −5° → 그 결정에 **위반자만** −20.
2. F2 유지 배 hold 창에서 |Δψ| > 10° → 위반자만 −20. hold 창 = tcpa ≤ EARLY_ACTION_TIME(imo 48.8 s)부터 17(b)(15.9 s)까지, 기준 침로 = 창 시작 침로.
3. F4 최소거리 < 24 m → 그 결정에 **둘 다** −20. F5 두 배 충돌 → 둘 다 −20(+충돌 −300).
4. 침로 기준(F1·F2·끝 판정의 "양보 max Δψ ≥ 10°")은 그 배의 **주 상대**(300 m 보상 risk argmax)인 결정에서만 누적·판정. 안전(F4·F5)은 모든 쌍.
5. 해소: tcpa > 17(b) 구간에서 dcpa ≥ 30 m 가 5결정 연속 → **성공 종료**(보너스 없음).
6. 늦은 시작: 시작 tcpa < 28 s 인 쌍은 안전(F4·F5)만 판정.
7. 쌍당 실패 1회. 실패 뒤엔 종료 조건(CPA 통과 3·300 m 밖 5·종료)까지 active 유지, 추가 벌점·재시작 없음. 끝에 판정 = 실패.
8. 조우 중 벽·제3선 충돌·시간초과로 끝난 배의 crash 벌점(s_ 규칙) 유지. 끝 판정에서 아직 실패하지 않은 쌍: 양보 max Δψ ≥ 10°(해소된 쌍은 면제) & 안전 → 성공, 아니면 그 결정에 −20(끝 판정 실패는 둘 다).
9. **넣지 않는 것**: Rule 16 기한(tcpa ≤ 17(b) 시점 10° 미달 → 실패) — OFF 가 56 m 정면 조우에서 물리적으로 못 지킴(타속 3°/s, 발견 tcpa ≈ 18 s) → 통신 쪽 조작 소지. 조기회피 300 m 보상(통신만 딸 수 있는 양의 항·farming 가능). 선회량 비용(연료 타각항과 중복, 이상 행동). γ 변경(미검증). 3M 파일럿.
- eval 은 옛 판정(`[role-promise]`, end·엄격 전쌍, 숫자 연속성)과 v2(`[role-promise/v2]`)를 **둘 다** 출력. `eval_ckpt.py --role_judge {snapshot,end,v2}` 로 옛 체크포인트(s_off·x_off·h_a6·s_a6)도 v2 로 재채점.
- 새 키 4개는 `SIM_SNAPSHOT_KEYS` 에 넣음(재개·분기·eval 이 보상 차이를 감지). 기본값에서 골든 비트동일.

## 4. 순위 게이트 v2 (`verify/check_reward_rank.py`, 학습 전, 결과 무관)
- 정책 13: goal·stbd·radar·vo56·vo56s·seeker·wall·wallesc(기존) + drift(초기 정책 모사: a0 = tanh(N(0, e^−1)), a1 = tanh(N(0, e^−0.5)), 전용 Generator) · idle(목표 조향, 속도 명령 0.25) · turnless(침로 변화 ≤ 2°/결정) · weaver(vo56 + 조우 중 ±10° 교대) · vo300/vo300s(같은 VO, R = 300 m, H ≥ 150 s = 통신 정보 상한 참조).
- 시드 1·2·3, E 32, T 1500. 설정: old(PEN 0) · s_(PEN 20, end) · **v3** · v3−게이트(DCPA 게이트만 끔). 지표: 할인 G · 결정당 r · 에피소드 반환(done 기준) · roleKeptSafe v2.
- v3 통과 조건(전 시드): (i) G: min(vo56, vo56s, vo300, vo300s) − max(나머지 9) > 0.05·|max| (ii) r/dec: vo56 − max(goal, stbd, radar) ≥ 0.05 그리고 drift·idle < goal − 0.2 (iii) 에피소드 반환 순서 = (i) (iv) 정보가치: vo300 ≥ vo56 (G·r/dec) (v) 판정기 보정: roleKeptSafe v2 순서 vo300s > radar > goal.
- 결정 규칙: v3 실패 & v3−게이트 통과 → DCPA 게이트 없이 진행. 둘 다 실패 → 멈추고 보고(저자 결정). 그 밖의 계수 조정 없음.

## 5. 팔·trunk·분기 (§8-1 유지: 분기 9,043,968, 분기 후 7M → 16,056,320)
- trunk `t_trunk_d6_s43–47`(보상 v3, 통신 OFF). 감시(드라이버, 로그 파싱): ≥3M 창에서 TO ≤ 40 % & len ≤ 2000; 6.2–6.5M 창에서 ≥3 trunk 가 goal ≥ 20 %·oColl ≤ 10 %·TO ≤ 10 % — 아니면 멈추고 보고(자동 보상 전환 없음). 9.04M 관문 = 마지막 4창 goal ≥ 30 %·oColl ≤ 5 %, 시드 순 앞 3개(< 3 이면 멈춤).
- 갈래(관문 시드 3): `t_off` · **`t_a6`(주 처치)** · `t_a8`(보조, p12_k8 역할 선언 복원) · `t_a2`·`t_a4`(latent 사다리 전용, 확정 주장 없음: p6 합성 레시피 동일, k 만 다름, SHA 고정) · `t_offb`(OFF 재분기, 워밍업 2401 — 같은 trunk 재분기 잡음 N 측정). 워밍업 2400(offb 2401). 1차 wave off·a6·a8 → 2차 a2·a4·offb.
- 절제: a6·a8·a2·a4 × {msgzero, shuffle, decl0} × 3. 궤적: off·a6 s43. 재채점(evalref, v2 판정): s_off·x_off·h_a6·s_a6.
- 운영: train `VESSEL_JOBS=8 VESSEL_GPU_CAP=2 VESSEL_LAUNCH_GAP=90`(GPU 당 ON 2개), eval·ablate·traj `JOBS=12 CAP=3`. 학습 GPU 에 테스트 금지. 정지 감지(로그 15분 무진행)·kill 절차 = s_.

## 6. 학습 전 검사(하나라도 FAIL 이면 학습 안 함)
- `test_golden.py --check`(.win32, 기본값 비트동일) · `test_role_promise.py`(rp1–13 기본값 PASS + rp14–20 v2) · `test_reward_v3.py`(토글 기본값 → 30결정 롤아웃 torch.equal, 항 값, 게이트 형태) · `test_sim_snapshot.py`(키 31) · 미러 2종(a2·a4 케이스 추가) · `test_eval_diff.py`(새 줄 태그만 허용) · 순위 게이트 v2 · smoke(off a6 a2) · 코덱 p6 k2/k4 충실도 표(보고용).

## 7. 사전등록 판정 (결과 전 고정)
- 지표·방향: goal ↑ · vColl ↓(oColl 병기, N 넘게 오르면 표시) · DCPA = `[pair-detail]` 통과최소거리 중앙 ↑ · COLREGs = 레이더 안 `[encounter-COLREGs/실제타각] C` ↑(주, 역할별 병기) + `roleKeptSafe v2` ↑(300 m 정보격차 지표; 옛 판정값 병기) · fuel ↓ · headTravel ↓(둘 다 `[goal-ep]` 도착 에피소드 평균 = 저자 정의). 진단만(판정 아님): fuel/진행거리, headTravel/len, 함대 단위 fuel/도착 수, len, TO, epReward.
- 통과 = 같은 시드 짝 3/3 **그리고** 평균차 > N. N = max(t_offb−t_off 재분기 차이(지표별), t_off 시드 범위, 09-28 잡음 N: vColl 12.1 pp·통과거리 1.5 m).
- 최선 OFF: goal·vColl·DCPA 는 건강한 OFF(goal ≥ 50 %) 풀 {t_off, s_off, x_off} 지표별 최대도 넘어야 통과. fuel·headTravel·COLREGs 는 같은 보상 t_off 짝 비교만 + OFF 유능성 게이트(t_off goal ≥ min(s_off, x_off) − N_goal, vColl ≤ max(s_off, x_off) + N_vColl; 미달이면 "보상이 기준선을 바꿈 — 그 지표 우월 주장 없음").
- 문장 규칙: "COLREGs 우월" = 레이더 안 C **와** roleKeptSafe v2 둘 다 통과. v2 만 통과 → "원거리 공동 역할 준수 우월, 레이더 안 C 는 (결과대로)"로 씀. "latent 덕분" = msgzero·shuffle 모두 vColl 3/3 악화. 주 처치 = t_a6; a8 은 보조, a2·a4 는 사다리 보고만.
- 목표 2(그림): ① 결정당 raw + 중심 이동평균(20 update), 0–16M 전체, y 축에 0·첫 값 포함, 분기선·첫 update 주석 ② 에피소드 반환(`<run>_ep.csv`) 0–16M ③ 분기 뒤 확대. 통과 = 분기 뒤 t_a6 이동평균 ≥ t_off 가 update 의 90 % 이상 & 끝 차이 > (t_offb−t_off 차이), 3/3, 두 곡선 모두. EMA(α 0.02, ~50 update 지연)는 판정에 안 씀.
- H1a: 어떤 통신 팔이 vColl 3/3 나쁘고 > N 이면 FAIL 보고, 버그 점검 먼저. 불통과·불리한 결과 전부 그대로 보고.

## 8. 한계(결과 전)
- 판정기 v2 와 옛 판정은 숫자가 다름(v2 는 주 상대·해소·늦은 시작 규칙) → 옛 배치와는 v2 재채점값으로만 비교.
- 보상 v3 는 모든 팔의 속도·선회를 바꿈 → 연료·선회·COLREGs 는 옛 배치와 짝 비교 불가(§7 규칙).
- 여전히 관측에 없는 조우 상태에 보상이 의존(마르코프 아님). γ 0.99. 시드 3.
- Stage A 는 사후 추가 팔(탐색). p6 k=2·4 코덱은 합성 데이터(p6_k6 과 같은 레시피).
