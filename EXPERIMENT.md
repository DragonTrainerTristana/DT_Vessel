# 실험 5 — Comm_MoE (COLREGs 상황별 분리 신경망)

> 통신 ON 상태에서 정책망을 **COLREGs 상황별 hard-routed Mixture-of-Experts**로 구성.
> 가설: 단일망이 4상황(head-on/stand-on/give-way/overtake)의 상충 회피규칙을 *평균내며 간섭* →
> 상황별 head로 분리하면 각 상황 전문화 → 충돌↓(특히 rush>avoid 붕괴 완화).

## 구조
- 공유 backbone + **5개 head**(None / HeadOn / CrossingStandOn / CrossingGiveWay / Overtaking).
- Unity가 판정한 상황(`obs[389]` = situation 라우터, 0~4)으로 head를 hard-route.
- **통신부 불변**(공유 MessageActor) — MoE는 *정책 head만* 분리. grad isolation 검증됨.
- `VESSEL_USE_MOE=0` = 단일망과 **비트동일**(공정 baseline, 검증 Δ=0) = anti-rigging.

## 토글 (390D 빌드 공통; USE_MOE만 토글)
| env | 값 | 의미 |
|---|---|---|
| `VESSEL_USE_COMM` | `1` | 통신 ON |
| `VESSEL_AGG_MODE` | `mean` | 〃 |
| **`VESSEL_USE_MOE`** | **`1`** | **COLREGs 5-head MoE (이 브랜치의 핵심)** |
| `VESSEL_MSG_DIM` | `6` | 〃 |
| `VESSEL_MSG_L2` | `0.01` | 〃 |
| `VESSEL_COLREGS_COEF` | `0.30` | COLREGs 1x |
| `VESSEL_RUN_STEP` | `1000000` | ★필수 명시 |

## 빌드
MoE 라우터는 `obs[389]`의 situation 슬롯을 사용 → **390D 빌드 필수**
(situation이 빠진 구 빌드면 head0로만 라우팅 = MoE 무의미). MoE 로직은 Python(networks.py).

## 실행
```powershell
# single(USE_MOE=0) × 3 + moe(USE_MOE=1) × 3, 통신 ON, 같은 빌드·seed
.\Python\run_sweep_moe.ps1 -Exe "<...>\Build\<390D>\Vessel_MLAgent.exe"
```

## 판정 (ground-truth)
수렴 마지막 30%, 3-seed: **moe arm의 vColl·oColl·LATE < single arm**이면 상황분리 효과.
moe ≈ single 이면 = 상황 간섭이 병목 아니었음(또는 라우팅 1-step stale / 희귀상황 head data-starve).
`analyze_run.py` + `convergence_gate.py`.
