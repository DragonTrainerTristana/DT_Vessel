# 실험 4 — Comm_msgComparison (latent 차원 2~12 비교, H2)

> 통신 ON 상태에서 메시지 차원 `VESSEL_MSG_DIM`을 **{2, 4, 6, 8, 10, 12}**로 스윕,
> 차원↑ → 정보량·협응 질↑ 인지 측정(H2). 네트워크는 동일, MSG_DIM만 변동.

## ⚠️ 전제: H1이 먼저 성립해야 의미 있음
통신이 *실제로 쓰이지 않으면* 차원 수는 무의미하다(과거 MSG_DIM 2~12가 **동일 수렴**한 전례 =
통신 미사용). 그러니 **Comm_BaseLine에서 H1b(comm>OFF)가 확인된 뒤** 이 스윕을 해석할 것.
H1 미성립 상태의 "차원 무차별"은 H2 반증이 아니라 통신 비활성의 증거.

## 토글 (390D 빌드 공통; MSG_DIM만 arm별로 변동)
| env | 값 | 의미 |
|---|---|---|
| `VESSEL_USE_COMM` | `1` | 통신 ON |
| `VESSEL_AGG_MODE` | `mean` | 〃 |
| **`VESSEL_MSG_DIM`** | **2 / 4 / 6 / 8 / 10 / 12** | **이 브랜치의 스윕 변수** |
| `VESSEL_MSG_L2` | `0.01` | 〃 |
| `VESSEL_MSG_GATE_L2` | `0.02` | 〃 |
| `VESSEL_COLREGS_COEF` | `0.30` | COLREGs 1x |
| `VESSEL_RUN_STEP` | `1000000` | ★필수 명시 |

## 실행 (dim × seed 그리드)
```powershell
foreach ($dim in 2,4,6,8,10,12) {
  foreach ($seed in 42,43,44) {
    $env:VESSEL_USE_COMM="1"; $env:VESSEL_AGG_MODE="mean"; $env:VESSEL_MSG_DIM="$dim"
    $env:VESSEL_MSG_L2="0.01"; $env:VESSEL_RUN_STEP="1000000"
    .\Python\run_experiment.ps1 -Exe "<...>\Build\<390D>\Vessel_MLAgent.exe" `
      -Seed $seed -Comm 1 -Agg mean -Port (5300+$dim*10+$seed%10) -Tag ("dim{0}_s{1}" -f $dim,$seed)
  }
}
# 동시 18개는 과부하 → dim 2~3개씩 나눠 실행. 참고: launch_latentNEW.sh / launch_editor_dim6.ps1
```

## 판정 (ground-truth)
dim별 `VESSEL_OUTCOME_LOG`/`VESSEL_METRIC_LOG` 곡선. 차원↑에 따라 vColl↓·협응↑가
**단조 개선**하다 수확체감하는지, 아니면 평탄(통신 미사용)/과적합·불안정인지.
`analyze_run.py` 로 dim 간 비교.
