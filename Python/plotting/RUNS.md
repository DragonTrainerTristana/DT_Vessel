# 그림 ↔ 실행 ↔ 설정 대조표

각 그림이 어떤 학습 실행에서 나왔고 그 실행이 어떤 설정이었는지 정리한 문서다.
표의 "구조" 열은 추측이 아니라 저장된 체크포인트를 열어 확인한 값이다.
(전문가들의 레이더 인식부 텐서가 같은 값이면 공유, 다르면 분리, 폭은 conv 채널 수로 판정)

---

## 1. 모든 실행에 공통인 설정

아래는 그림에 쓰인 모든 학습이 똑같이 쓴 값이다. 실행마다 다른 것은 2절의 표에만 적는다.

```
--envs 128 --vessels 16 --rollout 32 --steps 16000000 --ring 0.7
VESSEL_POS_GROUND=1        메시지에 송신자 상대위치를 실어 보냄
VESSEL_USE_ATTENTION=0     집계는 어텐션이 아니라 마스크 평균
VESSEL_THREAT_COEF=0.5     먼 거리 위협 보상 계수
VESSEL_RADAR_RANGE=56      레이더 탐지 거리(m)
--comm_on_at 9000000       통신을 켜는 시점 (전체 16M 중 9M 지점)
--max_partners 4           메시지를 받는 이웃 수
VESSEL_SIM_COLREGS_COEF=0.45   규정 준수 보상 계수
```

시드는 42, 43, 44 세 개다. 실행 이름 끝의 `_s43`, `_s44`가 시드이고, 접미사가
없는 것은 시드 42다.

## 신경망 구조 네 가지

| 이름 | 환경변수 | 레이더 인식부 | 판단부 | 파라미터 |
|---|---|---|---|---|
| 단일망 | `VESSEL_USE_MOE=0` | 하나 | 하나 | 369,131 |
| 분리·얇게 | `VESSEL_USE_MOE=1 VESSEL_MOE_WIDTH=0.44` | 5벌(얇게) | 5벌 | 363,004 |
| 분리·두껍게 | `VESSEL_USE_MOE=1 VESSEL_MOE_WIDTH=1.0` | 5벌 | 5벌 | 1,826,719 |
| **공유 (제안)** | 위에 더해 `VESSEL_MOE_SHARED=1` | **하나** | 5벌 | 511,543 |

5벌인 이유는 COLREGs 조우 상황이 다섯 가지(조우없음·정면·교차유지·교차양보·추월)이고,
상황을 기하로 판정해 그 값으로 곧바로 담당 전문가에게 보내기 때문이다. 게이트를
학습하지 않으므로 전문가 하나로 몰리는 붕괴가 없다.

---

## 2. 그림별 대조표

### Fig1_Communication — 통신을 쓸 때와 쓰지 않을 때

| 그림의 선 | 실행 이름 | 구조 | 다른 점 |
|---|---|---|---|
| No Communication | `qf_SE_OFF_s42/s43/s44` | 공유(제안) | `--arm OFF` (메시지 입력이 항상 0) |
| Communication | `qd_MOE_SE_s42/s43/s44` | 공유(제안) | `--arm ON` |

두 조건의 차이는 통신 하나뿐이다.

### Fig2_MoE_Architecture — 신경망을 어떻게 나눌 것인가

| 그림의 선 | 실행 이름 | 구조 | 시드 |
|---|---|---|---|
| Single network (369K) | `q_MOE_SINGLE`, `qd_MOE_SINGLE_s43/s44` | 단일망 | 3 |
| Separate experts, thin (363K) | `q_MOE_ISO` | 분리·얇게 | **1** |
| Separate experts, full (1.83M) | `base_comm`, `base_comm_s43/s44` | 분리·두껍게 | 3 |
| Shared perception (512K) | `qd_MOE_SE_s42/s43/s44` | 공유(제안) | 3 |

네 조건 모두 메시지 6차원, 통신 9M 시작, 이웃 4척으로 같고 구조만 다르다.
`base_comm`과 `qd_MOE_SE`는 레이더 인식부 공유 여부 하나만 다른 완전 대응쌍이다.

주의: thin 조건은 시드가 하나뿐이고 도착률 33.8%로 혼자 크게 무너져 있다.

### Fig3_Message_Aggregation — 몇 척에게서 메시지를 받을 것인가

| 그림의 선 | 실행 이름 | 구조 | 다른 점 |
|---|---|---|---|
| Nearest-1 | `qf_SE_NEAR1_s42/s43/s44` | 공유(제안) | `--max_partners 1` |
| Aggregation of 4 | `qd_MOE_SE_s42/s43/s44` | 공유(제안) | `--max_partners 4` |

### Fig4_Message_Dimension — 메시지 벡터의 길이

| 그림의 선 | 실행 이름 | 구조 |
|---|---|---|
| Dimension 2 | `v2_DIM2`, `q_DIM2_s43/s44` | 분리·두껍게 |
| Dimension 4 | `v2_DIM4`, `qe_DIM4_s43/s44` | 분리·두껍게 |
| Dimension 6 | `base_comm`, `base_comm_s43/s44` | 분리·두껍게 |
| Dimension 8 | `v2_DIM8`, `qe_DIM8_s43/s44` | 분리·두껍게 |
| Dimension 10 | `v2_DIM10`, `qh_DIM10_s43/s44` | 분리·두껍게 |
| Dimension 12 | `v2_DIM12`, `q_DIM12_s43/s44` | 분리·두껍게 |

`VESSEL_MSG_DIM`만 다르다. **이 축만 제안 구조가 아니라 분리·두껍게 위에서 쟀다.**
여섯 조건이 서로 같은 구조이므로 축 안의 비교는 유효하지만, 논문에 "모든 ablation을
제안 구조에서 수행했다"고 쓸 수는 없다. 옮기려면 15회를 다시 학습해야 한다.

### Fig5_COLREGs_Term — 규정 준수 보상 항의 유무

| 그림의 선 | 실행 이름 | 구조 | 다른 점 |
|---|---|---|---|
| Without COLREGs term | `q_COLREGSOFF` | 분리·두껍게 | `VESSEL_SIM_COLREGS_COEF=0` |
| With COLREGs term | `base_comm`, `base_comm_s43/s44` | 분리·두껍게 | `=0.45` |

제안 구조 위에서 다시 재는 학습(`qo_SE_COLREGSOFF_s42/s43/s44`)이 돌고 있다.
끝나면 "있음" 쪽을 `qd_MOE_SE`로 바꿔 이 축도 제안 구조로 통일한다.

### (진행 중) 통신 시작 시점

| 조건 | 실행 이름 | 구조 | 다른 점 |
|---|---|---|---|
| 처음부터 통신 | `ql_SE_START_s42/s43/s44` | 공유(제안) | `--comm_on_at 0` |
| 9M부터 통신 | `qd_MOE_SE_s42/s43/s44` | 공유(제안) | `--comm_on_at 9000000` |

---

## 3. 성능 값은 어떻게 쟀나

막대 그림의 값은 학습 로그가 아니라 **학습이 끝난 뒤 정책을 고정하고 다시 항해시켜**
얻은 것이다. 학습 중의 기록은 에피소드가 몰려서 끝나는 시점에 따라 창마다 크게
흔들리므로 최종 성능 판단에 쓰지 않았다.

```
python eval_ckpt.py --ckpt <체크포인트> --arm ON --max_partners 4 \
       --envs 96 --burnin 1500 --eval_decisions 3500
```

`--burnin 1500`은 집계 없이 먼저 돌리는 구간이다. 모든 배가 같은 시점에 출발하면
종료도 같이 몰려서 통계가 왜곡되므로, 위상을 흩뜨린 뒤부터 센다.

결과는 `_data/metrics_v2.txt`(전체 지표)와 `_data/metrics_sit.txt`(상황별 준수율)에
한 줄씩 쌓인다.

## 4. 그림을 만드는 코드

| 코드 | 만드는 것 |
|---|---|
| `make_ablation_rewards.py` | `20_REWARD/` 학습곡선 (로그의 창별 결과를 step으로 묶어 집계) |
| `build_final.py` | `99_FINAL/` 주제별 폴더 — 곡선 1장 + 지표 막대 여러 장 + 설명 |
| `make_mixed_fleet.py` | 혼합 함대 묶음 막대 |
| `regenerate_all.py` | 위를 순서대로 실행 (자료 경로와 가로축 상한을 고정) |

전부 `Figures/_data/`에 얼려 둔 로그만 읽는다. 값을 손대는 코드는 없다.

```
cd Figures
python regenerate_all.py     # 학습곡선
python build_final.py        # 최종 폴더
```

## 5. 학습곡선의 세로축이 무엇인가

로그에 찍히는 `R` 값이 아니라, 같은 로그의 결과 분포로 다시 계산한 값이다.

```
보상 = 1.5 x 도착 - 6.0 x 충돌 - 0.5 x 시간초과      (창별 종료 에피소드 수로 가중)
```

로그의 `R`은 매 결정마다 주는 유도 보상의 평균이라 도착·충돌 같은 최종 결과와
상관이 낮다(실측 r 약 0.02). 그래서 성능 그림과 곡선이 서로 다른 이야기를 하게 된다.
위 식은 모든 조건에 똑같이 적용되며, 충돌 가중치를 3배에서 15배까지 바꿔도 순서는
변하지 않는다.
