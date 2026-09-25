# 의도·역할 통신(COMM_EXT) 설계 + 사전등록 — 2026-09-25

브랜치 `feat/comm-intent` (base `feat/dyn-profile-imo` @ 522ab5b). 이 문서는 **결과 전에** 고정함. 결과 뒤 변경은 §9 이력에만 추가.

## 0. 배경·결정 이력

- 1차 imo 파일럿(r56, 09-22~24): OFF 도착 49–52 %·선박충돌 45–48 %. ON(latent 6D + relpos) 0-3 패, msgzero(메시지 0) 로 개선 → H1a 위반 신호. RANDOM 도 OFF 에 도착 0-3
- 09-25 저자 결정: 레이더 56 m 고정(r112/r168 분기 폐기). r112 도 OFF 도착 34–61 %(저자 보고)
- 09-25 보상 검토(워크플로 11 에이전트): 현행 보상은 이미 레이더 밖(56–300 m) 충돌코스에 벌점을 주는 통신 편 구조였고 그래도 ON 이 짐 → **보상 기울이기 기각**. 병목 = ON 전용 aux 비대칭, 관측 빈약(상대 속도 없음·절대속도 없음), 역할이 56 m 안·1척 기준, 조율
- 09-25 저자 제안(원문 요지): "통신 = 서로 좌표를 안다. 상대 입장에서 내가 타를 어떻게 줄지, 속도를 어떻게 줄지, COLREGs 역할(나는 stand-on 너는 give-way)을 알게 되면 상대는 보완 행동을 한다 → 통신 효과가 드러난다." 역할을 통신 반경 안에서 상대별로 판정해 주고받는 것을 지금 구현하라
- 09-25 검증(워크플로 3): 각본 16척 sim 충돌 OFF 41 % → 상태 공유(300 m 위치·속도) 13–26 % → +의도 12.5 %. 레이더가 상대 기동을 알아채는 시점 3–6 s(이상 관측)~5–18 s vs 의도 0.4 s. 효과 대부분은 레이더 밖 상태 공유, 의도는 늦은 조우(≲40 m) 조율에서 추가(함대 ±2 pp, 각본 기준). 정직성 검토 = conditional(아래 통제 필수)
- 09-25 저자 승인: 5팔 사다리 + 통신 팔 aux 0 방향으로 구현 진행. "내가 의도하는 모든 게 잘 드러나게"

## 1. 저자 의도 → 코드 → 측정 대응

| 저자 의도 | 코드(필드·구조) | 드러나는 곳(측정) |
|---|---|---|
| 서로 좌표를 안다 | relpos(기존 3) + 상대 침로·속력·ROT(state) | ON-state vs ARPA@56 (레이더 밖 상태 공유 가치) |
| 내가 타를 어떻게 줄지 | 파트너 직전 명령 타각 `cmd_rudder_j/30` (intent) | ON-intent vs ON-state, eval 절제 intent-zero, 텔레메트리 act_zero_intent |
| 속도를 어떻게 줄지 | 파트너 직전 명령 속력 `target_speed_j/1.8` (intent) | 위와 같음 |
| COLREGs 역할: 나는 X 너는 Y | 내 역할(i→j) one-hot 5 + 상대가 선언한 역할(j→i) one-hot 5, 통신 반경·상대별, 충돌위험 게이트 | eval 절제 role-zero, 조율 진단(역할 일치율) |
| 상대는 보완 행동 | 상대별 비선형 토큰 인코딩(k/v MLP) → attention → 조타 | 조율 진단: 양보선 우현 + 유지선 침로유지 비율, 충돌 쌍 역할 분해, 궤적 덤프(F5) |
| 통신 효과가 드러난다 | 같은 trunk 5팔 사다리, 필드만 다름 | ground-truth 승패(§6) |

## 2. 확장 필드 (COMM_EXT_LAYOUT = 'v1', 20차원, 파트너 j 를 수신자 i 가 받는 값)

| idx | 이름 | 식 | 그룹 |
|---|---|---|---|
| 0:2 | rel_heading | sin(ψj−ψi), cos(ψj−ψi) | state |
| 2 | sog | speed_j / 1.8 (함대 최고속 = MAX_SPEED_BASE 1.0 × SPEED_MULT_MAX 1.8) | state |
| 3 | rot | yaw_rate_deg(rudder_j, speed_j, max_speed_j) / MAX_YAW_RATE (프로필 상수) | state |
| 4:6 | relvel | (v_j − v_i) 수신자 선체좌표 [우현, 전방] / 3.6 | state |
| 6 | dcpa_risk | 1 − clamp(dcpa/24, 0, 1) (접근 중일 때만, 멀어지면 0) | state |
| 7 | tcpa_risk | 1/(1 + tcpa/TCPA_RISK_DENOM) (접근 중일 때만, 멀어지면 0) | state |
| 8:13 | my_role | i 의 j 에 대한 조우 역할 one-hot [None, HeadOn, StandOn, GiveWay, Overtaking] | role |
| 13:18 | their_role | j 가 i 에 대해 판정·선언하는 역할 one-hot | role |
| 18 | cmd_rudder | env.cmd_rudder_j / 30 (결정 t 에서 읽으면 t−1 명령 = 슬루 목표) | intent |
| 19 | cmd_speed | env.target_speed_j / 1.8 | intent |

- 좌표 규약: 우현 = dx·cos h − dz·sin h, 전방 = dx·sin h + dz·cos h (compute_own_future 와 같음)
- 역할 판정: `_pairwise` 의 situation cascade 를 그대로 옮긴 순수 함수. 유효 = 거리 ≤ 파트너 반경 · raw_tcpa ≥ 0 · |방위| ≤ 100° · 양현 clear 아님 **+ 충돌위험 게이트 dcpa < 24 m**(Rule 7·14·15 'so as to involve risk of collision'). 게이트 밖은 None. 래칭 없음(순간 판정)
  - 게이트 근거(검증): 300 m 로 그냥 넓히면 역할이 붙는 쌍의 72–83 % 가 DCPA ≥ 24 m(무위험)
  - 게이트를 끄고 반경 56 m 로 부르면 `_pairwise()['sit']` 와 전 쌍 일치해야 함(단위 테스트)
- 명칭: 논문에서 'COLREGs 역할' 로 부를지 '기하 조우 분류' 로 부를지는 저자 결정(§8)
- 패딩·파트너 없음: torch.where 로 0 (NaN×0 방지). 모든 값 유한
- 재스폰 직후 파트너: cmd_rudder=0, target_speed=U(0.2,0.5)×max (정책 출력 아님, 표시 안 함)

그룹 → 팔: latent = {} / state = {state, role} / intent = {state, role, intent}. 꺼진 그룹은 0.

## 3. 구조

- `VESSEL_COMM_EXT=1` 이면 `CNNPolicy.relpos_dim` 3 → 23. GroundedAttention k/v 를 선형 → MLP(token→64→·, ReLU). v 마지막 층 ×0.1·bias 0(소진폭 규약 유지). msg_encoder 입력도 23+msg_dim(미사용이지만 항상 생성)
  - MLP 근거: '상대가 give-way 이고 우현으로 틀고 있다 → 나는 유지' 는 역할×의도 상호작용. 선형 k/v 는 집계 전에 이걸 못 만듦. CPA 도 선형으로 못 만들어 명시 필드로 줌
  - 키 변화: `attn.k_proj.{0,2}.*`, `attn.v_proj.{0,2}.*` (EXT=0 은 `attn.k_proj.weight` 그대로 → 기본 비트동일)
  - `USE_ATTENTION=1` 필수(아니면 raise)
- `VESSEL_COMM_FIELDS` ∈ latent|state|intent (그룹 마스크). `VESSEL_COMM_LATENT` 1|0 (0 이면 attention 토큰의 latent 메시지 항 ×0 — rollout·update 가 같은 aggregate_batch 를 타서 미러 보장)
- `VESSEL_PARTNER_RANGE` (기본 = COMM_RANGE 300): 파트너 선택 반경. **보상 반경(reward_range = COMM_RANGE)과 분리** — ARPA@56 이 보상을 바꾸지 않게
- `VESSEL_AUX_LOSS_SCALE` (기본 1.0): ON 전용 보조손실(MSG_L2·state_recon 등) 배율. 0 이면 state_recon 라벨·계산 생략, 모듈 키 유지(strict 로드 호환)
- Unity 경로 `_get_others_msg`: relpos_dim ≠ 3 이면 명시 RuntimeError (gym 전용). C#·obs(369D) 무변경
- 미러: 확장 필드는 comm_gather 에서 env 상태로 계산해 prelpos 에 붙이고 그 텐서 하나를 집계·반환(버퍼 저장)에 같이 씀 → update 는 저장값 재사용. 확장 필드 재계산 경로가 update 에 없음 = 구조적 일치. 그룹 마스크·latent 배율·파트너 반경은 networks 모듈 전역에서만 읽음(ckpt_io 가 스냅샷으로 덮어씀)

## 4. 팔 (전부 같은 trunk 에서 분기, §8-1 규약)

| 팔 | run_repro 이름 | --arm | fields | latent | partner_range | 의미 |
|---|---|---|---|---|---|---|
| OFF | off | OFF | – | – | – | 레이더만 |
| ARPA@56 | arpa6 | ON | state | 0 | 56 | 통신 없이 레이더 추적으로 가질 수 있는 최선(표적별 속도·위험·역할) |
| ON-latent | onl6 | ON | latent | 1 | 300 | 학습 latent + 위치 (1차 ON 을 aux 0·EXT 구조로) |
| ON-state | ons6 | ON | state | 1 | 300 | + 레이더 밖 상태·역할 공유 |
| ON-intent | oni6 | ON | intent | 1 | 300 | + 타·속력 계획 공유 |

- 배치 공통: `VESSEL_COMM_EXT=1`, dyn=imo, obstacles=none, 레이더 56, msg_dim 6, 통신 팔 전부 `VESSEL_AUX_LOSS_SCALE=0`
- trunk: 새 OFF trunk(EXT=1) 9,043,968 결정. trunk 파일 접두어 분리(EXT=0 trunk 재사용 사고 방지)
- 병행(선택, G6 2단계): 1차 trunk_d6 에서 aux 0 ON 갈래 3개 → msgzero 해로움 원인이 aux 인지 확인. EXT 코드 불필요(`VESSEL_AUX_LOSS_SCALE` 만)
- RANDOM 학습 팔은 두지 않음. 구조화 필드의 영가설 대조는 eval field-shuffle 로 함

## 5. 평가·절제·기전 진단

- 절제(ON형 팔마다): msgzero(others_msg 전부 0) / latent-zero / state-zero / role-zero / intent-zero / field-shuffle
  - field-shuffle = 같은 env 안의 *유효* (수신자,슬롯) 항목끼리만 값을 돌림(무작위 순서 + 한 칸 회전 = 자기 값으로 안 돌아옴). 유효 슬롯 값의 모음은 그대로, 짝만 끊음. **구조화 필드의 주 영가설 대조**
  - 그룹 0(state0·role0·intent0)은 '0 값 입력'(role 전부 0 one-hot 등 학습 때 없던 값) → 보조 절제로만 보고
- 조율 진단(모든 팔, 기하 판정이라 OFF 에도 정의됨). 행동 = **실제 타각** δ̄=rudder/30, 쌍 지표(tex)와 같은 문턱(우현 δ̄>0.05 · 좌현 δ̄<−0.10 · 유지 |δ̄|<0.15). 역할 = §2 게이트된 기하 역할(≤COMM_RANGE)
  - 결정 단위(레이더 안 / 밖): 양보-유지 쌍의 보완(양보 우현 + 유지 유지)·양보 우현·유지 유지·유지 좌현·둘 다 유지, HeadOn 둘 다 우현, 역할 맞물림(GW↔SO, HO↔HO, OT↔None)
  - 충돌 분해: 선박 충돌로 끝난 배마다 직전 결정의 가장 가까운 상대와의 역할·행동 → 다선(위험 상대≥2) / 양보선: 미행동·행동+상대 유지·행동+상대도 회피 / 유지선: 회피·고수+상대 미행동·고수+상대 행동 / HeadOn 둘 다 우현·아님 / 추월 / 역할 없음
- 궤적 덤프 `--traj_out`(.pt, 위치·침로·속력·타각·명령·상황·목표·최대속력·종료). F5(같은 조우 ON vs OFF)는 `run_repro.sh traj` = 팔마다 같은 시드·burn-in 0 → reset 장면 동일, 첫 재스폰 전 조우만 짝지음
- 텔레메트리(EXT 런만, 열은 뒤에만 추가): act_zero_state / act_zero_role / act_zero_intent. 기존 열(act_zero 등)은 latent 기준 정의 유지

## 6. 사전등록 판정

- 주 지표: 선박충돌률 vColl (OBB 사건 판정 = 보상과 독립). 부 지표: 도착·timeout·allMinSep
- 보조(보상 결합 표시): fuel·headTravel·Rule8@10·colregs 계열. 순환: epReward·학습곡선(근거로 안 씀)
- 주 비교
  - H1a: ON형 팔 각각 ≥ OFF (3/3 시드에서 나쁘고 평균차 > OFF 범위면 FAIL → 버그 점검)
  - 상태 공유 가치: ON-state vs max{OFF, ARPA@56}
  - 의도 공유 가치: ON-intent vs ON-state
- 판정: 같은 trunk 짝 시드별 승패 + 평균차 + OFF 시드 범위. 주장은 3/3 승이고 평균차 > OFF 범위일 때만(시드 5 면 ≥4/5)
- 최선 OFF = 주 지표 기준 max{OFF, ARPA@56} (결과 전 고정)
- 검출력: 3시드 짝차 MDE vColl ≈ 6 pp, goal ≈ 11 pp. 이보다 작은 차이(의도 가치 예상 ±2 pp)는 '판정 불가' 로 보고하거나 시드 확대(§8)
- G1: r56 imo 는 G1 FAIL(1차 OFF 49–52 %) 그대로 보고. 대체 G1'(저자 승인 대기): ① OFF 곡선 고원(마지막 2M 결정 기울기) ② 최선 OFF 규칙. 주장 범위 = 'radar-only 학습 정책이 무너지는 regime(imo + 56 m)에서 공유 X 가 ground-truth 를 Δ 만큼 바꿈'. 금지 = '유능한 기준선 대비 통신 이득', '현실 선박 일반화'
- 중단 규칙: ON-intent·ON-state 가 최선 OFF 를 못 이기면 결론 '안 도움'. 이번 주기에 채널·보상 변형 추가 없음. 이긴 변형·지표만 골라 보고 금지 — 전부 보고
- 1차 파일럿(G6 FAIL) 결과는 그대로 보고, 대체하지 않음

## 7. 알려진 한계 (결과 전 기록)

- 이상화: 필드는 시뮬레이터 참값(지연 0, 잡음 = 탐색 잡음뿐). 실제 AIS(2–10 s 간격)·VHF(지연·오해) 보다 좋음 → '이상화된 공유의 상한' 으로만 주장
- 보상: 유지선 침로유지 보너스·Rule 17(b) 전환 15.9 s(imo 에선 늦음)·양보선 좌현 벌점 → 의도 정보가 있어도 유지선이 안 움직이게 학습될 수 있음(각본 sim: INT-strict ≈ STATE-strict). 이번 주기 보상 불변
- 파트너 선택: 거리순 nearest-4 유지(밴드 위험쌍의 28–43 % 만 포함). 결과 뒤 위험순으로 바꾸지 않음
- gym 전용: Unity 판정관(C#) 미러 없음
- ON-latent(EXT) 는 1차 on6 와 다른 구조(MLP k/v, aux 0) → 1차 결과와 섞지 않음

## 8. 저자 결정 대기

1. 시드: 3(43–45) 또는 5(43–47 사전 선언)
2. 연구 질문 틀: 두 부분(1차 latent 결과 + 구조화 공유) / 주제 전환 / latent 유지. ON-state·intent 는 시뮬레이터 참값 주입이라 현 규칙상 ORACLE 계열 → 처치로 쓰려면 재정의
3. 역할 필드 명칭('COLREGs 역할' vs '기하 조우 분류')
4. G1' 채택
5. 새 지표(조율 진단)는 기전 설명용. 사전등록 H1 판정에는 안 씀 — 동의 여부
6. '최선 OFF' 선택 단위(결과 전 고정 필요): (a) 팔 단위 — 시드 평균 vColl 이 낮은 팔 하나를 최선 OFF 로 두고 짝 승패·평균차·시드 범위 모두 그 팔 기준 [권고] / (b) 시드별 min vColl, 범위는 두 팔 합집합

## 9. 변경 이력

- 2026-09-25 최초 작성(결과 전)
- 2026-09-25 적대적 검토(5관점·반박 검증) 반영, 결과 전: field-shuffle 을 유효 항목끼리·자기값 제외로 수정 / 조율 진단 행동을 샘플 명령 → 실제 타각(tex 문턱)으로 / 충돌 분해 구현 / F5 용 traj 모드(burn-in 0) / test_comm_ext 가 배치 env 를 물려받아 preflight 가 막히던 것 수정 / env_lines partner_range None → unset / §8-6 최선 OFF 선택 단위 추가
