# 실험 2 — Comm_BaseLine (통신 ON, 기본)

> latent 메시지 교환을 켠 기본 통신 정책. **NonComm_BaseLine과 *같은 빌드·seed·보상*,
> 오직 `VESSEL_USE_COMM`만 토글** (공정 비교). MSG_DIM=6, mean 집계 + H1a 게이트.

## 가설
- **H1a (불변식, 반드시 성립)**: comm-ON ≥ comm-OFF. 통신은 *절대 더 나빠지면 안 됨*
  (메시지 무시 가능 = value-of-information ≥ 0). 위배 시 = 구현/최적화 버그.
- **H1b (목표, 조건부)**: comm-ON > comm-OFF. task가 통신을 *필요*로 할 때만(국소 인지 부족/협응 모호).
  완전관측에선 comm-ON = comm-OFF가 최선(통신 잉여).

## 토글 (390D 빌드 공통, env만)
| env | 값 | 의미 |
|---|---|---|
| `VESSEL_USE_COMM` | `1` | 통신 ON (공유 MessageActor) |
| `VESSEL_AGG_MODE` | `mean` | 메시지 집계 (검증된 comm1fix; sum은 노이즈) |
| `VESSEL_MSG_DIM` | `6` | 메시지 차원 (기본) |
| `VESSEL_MSG_L2` | `0.01` | 메시지 L2 정규화 (H1a 안정화) |
| `VESSEL_MSG_GATE_L2` | `0.02` | 게이트 개방 페널티 (메시지 무시로 수렴 압박) |
| `VESSEL_COLREGS_COEF` | `0.30` | COLREGs 1x (Reinforcement 브랜치와 대비) |
| `VESSEL_RUN_STEP` | `1000000` | ★필수 명시 |

## 실행
```powershell
# run_sweep.ps1 의 comm=1 arm (seed 42/43/44)
.\Python\run_sweep.ps1 -Exe "<...>\Build\<390D>\Vessel_MLAgent.exe"
```

## 빌드
`VectorObservationSize = 390`. attention/게이트는 Python 전용(재빌드 불필요). from-scratch.

## 판정 (ground-truth)
`VESSEL_OUTCOME_LOG` + `VESSEL_METRIC_LOG`, `convergence_gate.py`.
**NonComm_BaseLine 대비 seed-paired Pareto**(goal↑ · vColl↓ · 연료↓)면 H1b 양성.
comm-ON이 OFF보다 *나쁘면* H1a 위배 = 게이트/zero-init 디버그 1순위.
