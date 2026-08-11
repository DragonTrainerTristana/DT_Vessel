<#
통신 가치 게이트 sweep — 계단식 2단계 (2026-07-03 아키텍처 기준).

전제 아키텍처(기본값): situation one-hot 입력(fc2 41/47) · pos_ground 집계 · 파트너 4척 · COLREGs 판정 사각지대 fix 포함 빌드.
⚠️ 빌드 주의: 사각지대 fix는 C# 변경 — 2026-07-03 이후 소스로 빌드한 exe 필수.
   main.py 빌드함정은 obs 크기(369)만 검사하므로 fix 前 빌드도 조용히 통과함. exe 날짜를 직접 확인할 것.

설계 논리 (정보 격차 → 수요 → 공급 → 소비):
  - 정보 격차: dense antipodal regime(CROSSING=2, ring 0.5, n16) + radar 56m 불변. 위협이 56~420m 대역에서
    발달하고, 밀집 씬에서는 레이더 가림도 실재 → 발신자만 아는 정보가 구조적으로 존재.
  - 수요(demand, 전 arm 동일 = anti-rigging):
      per-pair(음수 페널티): 모든 동시접근쌍을 가격 → 다물체 협응이 보상이 됨.
      far-field(양수 carrot, 대칭 PBRS): 56m~RISK_RANGE 띠의 risk 합 감소를 보상 → 먼 위협 조기회피가 보상이 됨.
      두 보상 모두 message-independent → OFF도 같은 보상을 받지만 벌 정보가 없음. 이것이 유일한 정당 비대칭.
  - 공급(supply, ON arm만): GoalComm + Intent 디코더(계수 0.05) — 메시지에 목적지·미래의도 인코딩 압력.
  - 소비(consumption, ONc5c arm만): Consumer 0.05 + Coupling=1 — 수신 정책이 메시지를 표상하고 행동에 사용.

단계:
  -Stage 1  : OFF vs ORACLE (정보가치 천장 측정 — 가장 싼 관문)
              oracle ≈ OFF → 학습 채널 무의미. regime을 더 조인다(-VesselCount ↑, -RingScale ↓) 후 재측정.
              oracle > OFF → 천장 존재 확정 → Stage 2 진행.
  -Stage 2  : ONsevered vs ONc5c (천장 확인 후에만)
              판정(수렴 꼬리 25~30%, ≥3 seed, seed-paired):
                ONc5c → oracle 근접           : 통신 정당 승리 (완승)
                ONsevered < ONc5c < oracle    : C5c 부분 기여 → 공급·소비 계수/attention 상향 여지
                ONc5c ≈ ONsevered ≈ OFF       : 채널 학습 실패 → 텔레메트리 진단(GateMean·Comm/Grad_*·메시지-위험 상관)
                ONc5c < OFF                   : C5c 간섭 → coef ↓ 또는 기각

사용 (Windows 학습 머신):
  .\run_sweep_commgate.ps1 -Stage 1                       # OFF+ORACLE × 3 seed = 6 run
  .\run_sweep_commgate.ps1 -Stage 2                       # ONsevered+ONc5c × 3 seed = 6 run
  .\run_sweep_commgate.ps1 -Stage 1 -RunStep 2000000      # 확정 판정용 2M
#>
param(
  [int]$Stage = 1,                  # 1: OFF+ORACLE (천장 게이트) / 2: ONsevered+ONc5c
  [string]$Exe = "",                # 비우면 프로젝트 루트 Build\Vessel_MLAgent.exe. ★2026-07-03 이후 소스 빌드 필수
  [int]$RunStep = 1000000,          # 1M 잠정, 확정 판정은 2M
  [int[]]$Seeds = @(42,43,44),
  [string]$RingScale = "0.5",       # dense antipodal
  [int]$VesselCount = 16,
  # ── 수요측 보상 (전 arm 동일) ──
  [string]$PerPairCoef = "-0.3",    # VESSEL_PERPAIR_COEF: 음수 = 다물체 동시접근 페널티 (0=off)
  [string]$PerPairExp  = "1.6",
  [string]$FarFieldCoef = "0.5",    # VESSEL_FARFIELD_COEF: ★양수만 유효(carrot, 대칭 PBRS — C# 파싱이 음수 무시).
                                    #   대칭화로 유효 강도 ~2×라 0.5~1.0 권장(코드 주석 기준)
  [string]$RiskRange = "420",       # VESSEL_RISK_RANGE: far-field 띠 상한 = 통신 범위와 일치
  # ── 공급측 (ON arm) ──
  [string]$GoalComm = "0.05",
  [string]$IntentCoef = "0.05",
  [string]$MsgL2 = "0.01"           # 통신 안정화 (OFF arm은 msg_reg≡0이라 무영향)
)
if (-not $Exe) { $Exe = Join-Path $PSScriptRoot '..\..\..\Build\Vessel_MLAgent.exe' }
if (-not (Test-Path $Exe)) { Write-Host "[ERROR] exe 없음: $Exe"; exit 1 }
$exeDate = (Get-Item $Exe).LastWriteTime
Write-Host ("빌드 날짜: {0} — 2026-07-03 이후(사각지대 fix 포함)인지 확인하세요." -f $exeDate)

# arm 구성. 집계는 기본 pos_ground(코드 기본값) — attention/sum 비교는 천장·승리 확인 후 별도 ablation.
$armsByStage = @{
  1 = @(
    @{Name="OFF";       Comm=0; Oracle=0; GoalComm="0.0";      Intent="0.0";        Consumer="0.0";  Coupling="0"},
    @{Name="ORACLE";    Comm=1; Oracle=1; GoalComm="0.0";      Intent="0.0";        Consumer="0.0";  Coupling="0"}
  );
  2 = @(
    @{Name="ONsevered"; Comm=1; Oracle=0; GoalComm=$GoalComm;  Intent=$IntentCoef;  Consumer="0.0";  Coupling="0"},
    @{Name="ONc5c";     Comm=1; Oracle=0; GoalComm=$GoalComm;  Intent=$IntentCoef;  Consumer="0.05"; Coupling="1"}
  )
}
$arms = $armsByStage[$Stage]
if (-not $arms) { Write-Host "[ERROR] -Stage 1 또는 2"; exit 1 }

# ── 전 arm 공통 env (dense regime + 수요측 보상). radar 56m 불변(VESSEL_RADAR_RANGE 미설정). ──
$env:VESSEL_CROSSING = "2"
$env:VESSEL_SPAWN_RING_SCALE = $RingScale
$env:VESSEL_VESSEL_COUNT = "$VesselCount"
$env:VESSEL_RUN_STEP = "$RunStep"          # ★명시 필수(config 기본 30M)
$env:VESSEL_PERPAIR_COEF = $PerPairCoef
$env:VESSEL_PERPAIR_EXP  = $PerPairExp
$env:VESSEL_FARFIELD_COEF = $FarFieldCoef
$env:VESSEL_RISK_RANGE = $RiskRange
$env:VESSEL_MSG_L2 = $MsgL2
Remove-Item Env:\VESSEL_RADAR_RANGE -ErrorAction SilentlyContinue
Remove-Item Env:\VESSEL_USE_ATTENTION -ErrorAction SilentlyContinue   # 집계 = 기본 pos_ground
Remove-Item Env:\VESSEL_POS_GROUND -ErrorAction SilentlyContinue

$port = 5600 + ($Stage - 1) * 10
foreach ($arm in $arms) {
  foreach ($seed in $Seeds) {
    $tag = "gate{0}_{1}_s{2}" -f $Stage, $arm.Name, $seed
    $env:VESSEL_USE_COMM = "$($arm.Comm)"
    $env:VESSEL_ORACLE = "$($arm.Oracle)"
    $env:VESSEL_GOAL_COMM_COEF = "$($arm.GoalComm)"
    $env:VESSEL_INTENT_COEF = "$($arm.Intent)"
    $env:VESSEL_COMM_CONSUMER_COEF = "$($arm.Consumer)"
    $env:VESSEL_COMM_CONSUMER_COUPLING = "$($arm.Coupling)"
    # -Agg mean은 pos_ground 기본 ON에서는 미사용(폴백 전용) — run_meta의 AGG_MODE=mean은 무시하고 읽을 것.
    $a = @("-NoExit","-ExecutionPolicy","Bypass","-File","`"$PSScriptRoot\run_experiment.ps1`"",
           "-Exe","`"$Exe`"","-Seed",$seed,"-Comm",$arm.Comm,"-Port",$port,"-Agg","mean","-Tag",$tag)
    Start-Process powershell -ArgumentList $a
    Write-Host ("launched {0} port{1} comm{2} oracle{3} goal{4} intent{5} consumer{6}/{7} (ring{8} n{9} perpair{10} farfield{11}@{12}m)" -f `
      $tag,$port,$arm.Comm,$arm.Oracle,$arm.GoalComm,$arm.Intent,$arm.Consumer,$arm.Coupling,$RingScale,$VesselCount,$PerPairCoef,$FarFieldCoef,$RiskRange)
    $port++
    Start-Sleep -Seconds 4
  }
}
Remove-Item Env:\VESSEL_USE_COMM,Env:\VESSEL_ORACLE,Env:\VESSEL_GOAL_COMM_COEF,Env:\VESSEL_INTENT_COEF,Env:\VESSEL_COMM_CONSUMER_COEF,Env:\VESSEL_COMM_CONSUMER_COUPLING -ErrorAction SilentlyContinue
Remove-Item Env:\VESSEL_CROSSING,Env:\VESSEL_SPAWN_RING_SCALE,Env:\VESSEL_VESSEL_COUNT,Env:\VESSEL_RUN_STEP -ErrorAction SilentlyContinue
Remove-Item Env:\VESSEL_PERPAIR_COEF,Env:\VESSEL_PERPAIR_EXP,Env:\VESSEL_FARFIELD_COEF,Env:\VESSEL_RISK_RANGE,Env:\VESSEL_MSG_L2 -ErrorAction SilentlyContinue

Write-Host "`nStage $Stage : $($arms.Count) arm x $($Seeds.Count) seed = $($arms.Count * $Seeds.Count) run (동시 6개 한계 내)."
if ($Stage -eq 1) {
  Write-Host "판정: oracle > OFF (수렴 꼬리, seed-paired) 확인 후 Stage 2. oracle ≈ OFF면 regime 강화 후 재측정."
} else {
  Write-Host "판정: ONc5c가 ONsevered를 이기고 Stage 1의 oracle에 근접하는가. 텔레메트리(GateMean·Comm/Grad_*) 병행 확인."
}
Write-Host "분석: python analyze_run.py <results> ; python convergence_gate.py <metric.csv...>"
