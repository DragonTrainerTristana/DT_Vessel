# 실험 1 — NonComm_BaseLine (통신 없는 베이스라인)

> 모든 통신 실험의 **공정한 기준선**. 통신 OFF + COLREGs 1x.
> egocentric ARPA 인지만으로 항해·회피 — comm-ON 브랜치들이 *이 수치를 ground-truth로*
> 이겨야 통신의 가치가 입증된다(불구화된 baseline을 이기는 건 무효 = anti-rigging).

## 토글 (5개 브랜치 공통 390D 빌드, 코드 동일 — env만 다름)
| env | 값 | 의미 |
|---|---|---|
| `VESSEL_USE_COMM` | `0` | 통신 OFF (`others_msg ≡ 0`) |
| `VESSEL_COLREGS_COEF` | `0.30` | COLREGs 보상 1x (기준 강도) |
| `VESSEL_RUN_STEP` | `1000000` | ★필수 명시 (미설정 시 config 기본 30M = 며칠) |
| `VESSEL_MSG_DIM` | (무관) | 통신 OFF라 미사용 |

## 실행
```powershell
# run_sweep.ps1 의 comm=0 arm (seed 42/43/44, 독립 프로세스)
.\Python\run_sweep.ps1 -Exe "<...>\Build\<390D>\Vessel_MLAgent.exe"
```

## 빌드
`VectorObservationSize = 390` (radar 360 + goal2 + self4 + ARPA21 + position2 + situation1).
동역학/obs 변경 누적 → **from-scratch 학습 필수**(옛 모델 호환 X).

## 판정 (항상 ground-truth 로그)
- `VESSEL_OUTCOME_LOG`: goal / collision_vessel / collision_obstacle / timeout
- `VESSEL_METRIC_LOG`(13열): 연료 · 궤적(straightness/headingTravel) · compliance
- `convergence_gate.py` 로 수렴 + LATE_COLLAPSE 점검.

이 브랜치의 vColl/goal/연료가 **다른 모든 comm 브랜치의 비교 기준**이다.
