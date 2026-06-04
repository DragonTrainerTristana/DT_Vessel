# 실험 3 — Comm_COLREGsReinforcement (통신 ON + COLREGs 1.5x)

> Comm_BaseLine과 **모든 것이 동일하되 COLREGs 준수 보상만 1.5x로 강화**
> (`VESSEL_COLREGS_COEF` 0.30 → 0.45). 규정 준수 shaping을 키우면
> 충돌↓·compliance↑가 되는지, 아니면 goal-pull과 상충해 timeout/우회만 늘리는지 시험.

## 토글 (390D 빌드 공통, Comm_BaseLine 대비 COLREGS_COEF만 변경)
| env | 값 | 의미 |
|---|---|---|
| `VESSEL_USE_COMM` | `1` | 통신 ON |
| `VESSEL_AGG_MODE` | `mean` | 〃 |
| `VESSEL_MSG_DIM` | `6` | 〃 |
| `VESSEL_MSG_L2` | `0.01` | 〃 |
| `VESSEL_MSG_GATE_L2` | `0.02` | 〃 |
| **`VESSEL_COLREGS_COEF`** | **`0.45`** | **COLREGs 1.5x (이 브랜치의 핵심 변수)** |
| `VESSEL_RUN_STEP` | `1000000` | ★필수 명시 |

## ⚠️ 빌드 주의
`VESSEL_COLREGS_COEF`는 **C# 신규 토글**(VesselAgent.Initialize) — 통합 베이스에 추가됨.
이 env가 효력을 가지려면 **그 코드가 포함된 390D 빌드로 재빌드**해야 한다.
(구 빌드는 필드 기본 0.45가 하드코딩 → 우연히 1.5x로 돌 수 있으니 빌드 버전 확인.)

## 실행
```powershell
$env:VESSEL_USE_COMM="1"; $env:VESSEL_AGG_MODE="mean"; $env:VESSEL_COLREGS_COEF="0.45"
$env:VESSEL_MSG_L2="0.01"; $env:VESSEL_MSG_GATE_L2="0.02"; $env:VESSEL_RUN_STEP="1000000"
# seed 42/43/44 독립 프로세스 — run_experiment.ps1 직접 호출 또는 run_sweep.ps1 변형
.\Python\run_experiment.ps1 -Exe "<...>\Build\<390D>\Vessel_MLAgent.exe" -Seed 42 -Comm 1 -Agg mean -Port 5242 -Tag colregs15_s42
```

## 판정 (ground-truth)
**Comm_BaseLine(0.30) 대비**: compliance(`VESSEL_METRIC_LOG`)↑ + vColl↓ 면 강화 효과.
compliance만 오르고 goal↓/timeout↑면 = 규정 shaping이 navigation과 상충(과강화).
`convergence_gate.py` + `analyze_circling_safety.py`.
