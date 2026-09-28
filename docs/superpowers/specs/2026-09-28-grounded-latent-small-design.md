# grounded latent 통신 + 원거리 COLREGs 보상 (작은 수정판) — 설계·사전등록 2026-09-28

> **결과 전 작성.** 저자 승인 2026-09-28("ㅇㅇ 수정하고, 지금 바로 윈도우에다가 진행"). 판정 규칙은 결과를 보기 전에 고정함. 결과가 나온 뒤 바꾼 것은 §7 이력에만 적음.
> 대체 관계: `2026-09-27-grounded-codec-design.md`(C 중심 9팔, Mac 6–7일)는 참고용으로만 보관함. 이 문서가 이번 주기의 정본임.

## 0. 왜 이렇게 하나

- 저자 논문 포인트: "적은 차원 latent 메시지로 통신하면 이득" — 안전(충돌·DCPA), COLREGs 준수, 일찍 조금 움직여서 빠르게
- 배치 X(09-26) 결과
  - 상태·역할 참값 공유(ons6·oni6)는 충돌·통과거리에서 3/3 이김
  - 학습 latent(onl6)는 OFF 보다 나쁨(H1a FAIL). ons6 에서도 latent0 절제가 성능을 3/3 개선함 → 창발 latent 는 뜻 없는 암호가 됨
  - 통신 팔은 COLREGs·연료·경로가 좋아지지 않음
- 원인(코드 확인): COLREGs 준수보상은 56 m 레이더 안 상황에서만 채점됨(`vessel_gym._reward` 의 `_cgate = max_risk_near > COLREGS_RISK_GATE`, 상황 판정 `dist <= DETECTION_RANGE`)
  - 56 m 밖에서는 규칙대로 해도 보상이 없고, 두 배가 다 틀어도 벌이 없음
  - 그래서 통신으로 멀리서 알아도 "양보선만 일찍 조금, 유지선은 유지"를 배울 이유가 없었음
- 저자 결정(09-28): 작은 수정 2개로 진행

## 1. 수정 1 — 원거리 COLREGs 준수보상 (`VESSEL_COLREGS_FAR_RANGE`, 기본 0 = 끔·비트동일)

- 값 300 이면: 56 m 안 상황이 없을 때, 300 m 안에서 충돌위험(dcpa < 24 m)이 있는 상대 가운데 보상 risk 최대인 배를 고름. 그 배와의 기하 역할(`encounter_role`, situation cascade 의 순수 함수판)로 **같은 준수항을 같은 계수로** 채점함
  - 양보선: 우현 +0.5 / 좌현 −0.5
  - 유지선(Rule 17a): 침로유지 +0.5, 속도유지
- 근거: COLREGs Rule 8·16(양보선의 조기·충분한 회피)·17(유지선의 침로·속력 유지)은 레이더 56 m 반경에 묶이지 않음
- **모든 팔 동일**, 9,043,968 결정(분기점)부터 켬. trunk 는 옛 보상 그대로
  - 학습기는 분기점에서만 이 키가 trunk 와 달라도 허용하고, 크래시 재개에서는 일치를 강제함
  - check_branch 는 갈래끼리 sim 일치를 강제함
- 관측(obs 369D)은 불변임. 레이더 56 m 도 불변임
- 09-25 "300 m 게이트" 기각과 다른 점: 그때는 통신에 유리하게 기울이는 안이었음. 이번에는 ① 규칙에 맞춤 ② 옛 보상 OFF 를 최선 OFF 후보에 포함함(§4) ③ 모든 팔 같은 보상

## 2. 수정 2 — grounded latent 메시지 (`VESSEL_COMM_CODEC*`, 기본 끔)

- 송신 페이로드 'p6' = 송신자가 결정 시점에 가진 자기 값 6개
  - [sinψ, cosψ, SOG/1.8, ROT/MAX_YAW_RATE, 직전 명령 타각/30, 직전 명령 속력/1.8]
  - `vessel_gym.own_payload`
- 동결 코덱 `Python/comm_codecs/p6_k6_s0.pt`
  - 구조: 인코더 6→64→64→6(tanh) + 8bit 균일 양자화 = **6차원 latent 메시지(48 bit/파트너/결정)**
  - 학습: 오프라인, 물리 일관 합성 데이터(imo 프로필), 정책 데이터·RL gradient 없음
  - 내용 SHA `fbe4c71a6bf4af3d2eab2626b929bc29d78a2c85d91ba3a0d54fc3091e0027f2` 고정(불일치면 중단)
  - 정책 파라미터·옵티마이저 밖, 항상 no_grad → z 의 뜻(송신자의 침로·속력·선회·명령)이 RL 로 바뀌지 않음
  - holdout 충실도: heading p50 0.24° / p99 0.98°, SOG p99 0.031 m/s. 실제 env 상태 기준 heading p99 1.03°, 역할 일치 0.992(`verify/test_grounded_latent.py`)
- 수신 두 방식. 둘 다 토큰 폭 불변(29) → 배치 X trunk 재사용 가능
  - **c6 (주 처치, direct)**: 확장필드 자리에 [z_j 6, 수신자 자기 상태 4(같은 식), 0×10] 를 넣음. attention k/v 가 z 를 **직접 읽고** 뜻을 학습함
  - **a6 (보험, decode)**: 수신측 동결 디코더로 복원한 뒤 `comm_pair_features(sender=복원값)` 로 20 필드를 계산함(oni6 형식, 송신자 값만 복원값)
- 창발 latent 끔(`COMM_LATENT=0`), 통신 팔 보조손실 0(`AUX_LOSS_SCALE=0`), 필드 그룹 마스크 전부(`COMM_FIELDS=intent`)
- 미러: 코덱 계산은 rollout `comm_gather` 에서만 하고 prelpos 로 버퍼에 저장함 → update 는 재사용함(구조적). `_verify_comm_mirror` 코덱 5케이스 ALL PASS

## 3. 팔 (새 배치, 접두어 `g_`, 시드 43·44·45, 배치 X trunk 복사본 `g_trunk_d6_s*` 에서 분기)

| 팔 | 보상 | 통신 | 역할 |
|---|---|---|---|
| g_off | 원거리 COLREGs 300 | 없음 | 새 보상 OFF |
| g_c6 | 같음 | 6D latent, 수신자가 직접 읽음 | **주 처치(latent 통신)** |
| g_a6 | 같음 | 같은 6D latent, 복원 후 쌍 필드 | 보험·읽기 진단 |
| (재사용) x_off | 옛 보상 | 없음 | 최선 OFF 후보 |
| (보고만) x_ons6·x_oni6·x_onl6 | 옛 보상 | — | 참고 |

- 학습 인자는 배치 X 와 같음(envs 128·vessels 16·rollout 32·crossing 0·16,056,320 결정·워밍업 1200)
- eval 도 배치 X 와 같음(envs 256·burn-in 2400·10,000 결정·seed 999)

## 4. 사전등록 판정 (결과 전 고정)

- **최선 OFF** = {x_off(옛 보상), g_off(새 보상)} 가운데 시드 평균 vColl 이 낮은 팔. 모든 지표에 이 한 팔을 씀
- **"우월" 지표 8개** (방향 고정. eval 출력 줄 이름 그대로):
  1. vColl ↓ (**주 지표**)
  2. `[pair-detail] 통과최소거리 중앙` ↑ (DCPA)
  3. `[조율/레이더 밖 56-300m] 양보 우현` ↑ (일찍 규칙대로)
  4. `[조율/레이더 밖 56-300m] 유지 유지` ↑ (유지선 규칙)
  5. `[encounter-COLREGs/실제타각] C` ↑ (전체 준수율)
  6. goal-ep headTravel ↓ (조금 움직임)
  7. goal-ep fuel ↓
  8. goal-ep len ↓ (빠르게)
- **지표별 판정**: 최선 OFF 와 시드별 짝 비교 3/3 승 **그리고** 평균차 > 최선 OFF 의 그 지표 시드 범위. vColl 은 여기에 평균차 > 6 pp(MDE)를 더함. 동률(소수 1자리 같음) = 승 아님
- **주 비교**: g_c6 vs 최선 OFF. **보조**: g_a6 vs 최선 OFF. g_c6 vs g_a6 는 서술만 함
- **"latent 메시지 덕분" 문구 조건**: g_c6 가 주 지표를 통과하고, 같은 체크포인트 절제 `msgzero`·`shuffle` 에서 vColl 이 3/3 나빠질 때만 씀. 아니면 "통신 채널 이득, 메시지 내용 기여 미확인"으로 씀
- **H1a**: g_c6·g_a6 가 최선 OFF 보다 vColl 3/3 나쁘고 평균차 > OFF 범위면 FAIL 로 보고함(버그 점검 반나절, 재튜닝 없음)
- 보고: 8지표 × 팔 × 시드 전부 보고함. 이긴 지표만 골라 쓰지 않음. 재학습 잡음(같은 trunk 재학습 vColl 최대 12.4 pp) 병기
- 금지: 결과를 본 뒤 k·코덱·반경·계수 재튜닝, 시드 추가(다음 주기 사전등록으로만)

## 5. 한계 (결과 전 기록)

- p6 는 6차원이고 z 도 6차원 → 압축이 아님. 정직한 명칭 = "송신자 상태에 grounding 된 6차원 latent 메시지(48 bit)". "스스로 배운(emergent) 통신" 아님
- 원거리 COLREGs 채점은 56 m 밖 역할을 쓰는데, 통신 없는 배는 그걸 관측하지 못함 → 준수 지표 3·4·5 는 통신 쪽이 유리할 수 있음. 이건 정보 차이(조작 아님)로 논문에 명시함. 옛 보상 OFF 를 최선 OFF 후보에 넣어 기준선이 불리해지는 것은 막음
- 절제 shuffle 은 유효 슬롯끼리 섞음. 한 송신자를 여러 수신자가 받으면 자기 z 로 돌아올 수 있음 → 정보 제거가 덜 됨 = 통신에 불리한 쪽(보수적)
- 코덱은 합성 데이터로 학습함. 실제 분포에서의 충실도는 §2 수치로 확인함(평가 창 안에서 재측정 권장)
- gym 전용임(Unity 판정관 미러 없음). 보상 변경도 gym 전용임

## 6. 검증 (Mac, torch 2.4 venv, 7013a3a 기준)

- 골든: 기본 설정 state_dict·Adam·ValueNorm·곡선 5케이스 비트동일(원본 7013a3a 로 만든 기준과 대조). 차이는 스냅샷 sim 키 1개 추가뿐. Windows 골든은 sim 키가 없는 09-10 판이라 영향 없음
- 통신 미러 36/36 ALL PASS(코덱 decode·direct·far300·혼합함대·K=1 포함)
- `verify/test_grounded_latent.py` 13/13: sender 대입 비트동일, SHA·프로필 가드, 양자화 결정론, 충실도, 원거리 보상은 상태 불변·대상 배만 다름, restore 누출 없음
- preflight 테스트(배치 env): fidelity·dyn_profile·sim_snapshot(키 25개로 갱신)·comm_ext ALL PASS
- 미니 파이프라인: trunk → off·a6·c6 분기(원거리 보상 전환 허용) → eval(코덱 스냅샷 복원) → check_branch ALL PASS(a6/c6 구분 표시). 재개 가드: 코덱 모드 불일치·원거리 반경 불일치 거부, 같은 설정 재개 OK. 텔레메트리 두 모드 정상

## 7. 변경 이력

- 2026-09-28 최초 작성(결과 전)
