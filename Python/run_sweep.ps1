<#
6병렬 from-scratch sweep: baseline-OFF(sum) × 3seed + extended-ON(mean) × 3seed.
각 run은 독립 프로세스(이 머신 최적 = 6병렬 × NUM_ENVS=1). 결과는 <build>\results\ 에 자동 정리.
anti-rigging: OFF/ON 같은 빌드·같은 seed·같은 env, 통신만 토글(공정 baseline).

사용:
  .\run_sweep.ps1 -Exe "C:\...\Build\0601_reward\Vessel_MLAgent.exe"
#>
param(
  [string]$Exe = "C:\Users\sengh\Dropbox\Private_Paper_Project\Vessel\Vessel_MLAgent\Build\0601_reward\Vessel_MLAgent.exe"
)
if (-not (Test-Path $Exe)) { Write-Host "[ERROR] exe 없음: $Exe (먼저 Unity 재빌드)"; exit 1 }

# H1a 시험: ON은 mean 집계 권장(zero-init과 함께 value-of-info≥0). OFF는 통신 미사용이라 agg 무관.
$combos = @(
  @{Comm=0; Seed=42; Port=5042; Agg="sum"},
  @{Comm=0; Seed=43; Port=5043; Agg="sum"},
  @{Comm=0; Seed=44; Port=5044; Agg="sum"},
  @{Comm=1; Seed=42; Port=5142; Agg="mean"},
  @{Comm=1; Seed=43; Port=5143; Agg="mean"},
  @{Comm=1; Seed=44; Port=5144; Agg="mean"}
)
# ★RUN_STEP 명시 필수: 미설정 시 config.py 기본 30,000,000(30M)으로 돌아 며칠간 안 멈춤(2026-06-03 실측 버그).
#   rushfix/moe 런처와 동일 1M 깊이 비교 + 6 combo 전부 대칭 적용(anti-rigging: OFF/ON 같은 학습 길이).
#   (run_experiment.ps1은 -RunStep 미전달 시 env 안 건드림 → 여기서 자식 프로세스에 상속시킴.)
$env:VESSEL_RUN_STEP = "1000000"

foreach ($c in $combos) {
  $a = @("-NoExit","-ExecutionPolicy","Bypass","-File","$PSScriptRoot\run_experiment.ps1",
         "-Exe","`"$Exe`"","-Seed",$c.Seed,"-Comm",$c.Comm,"-Port",$c.Port,"-Agg",$c.Agg)
  Start-Process powershell -ArgumentList $a
  Write-Host ("launched comm{0} seed{1} port{2} agg{3}" -f $c.Comm, $c.Seed, $c.Port, $c.Agg)
  Start-Sleep -Seconds 4   # 포트/Unity 연결 충돌 회피
}
# env 정리: 같은 PowerShell 세션에서 다른 sweep을 이어 돌릴 때 RUN_STEP 상속 방지.
#   (이미 Start-Process된 자식들은 spawn 시점 env를 받았으므로 영향 없음 — 이후 sweep 격리용.)
Remove-Item "Env:\VESSEL_RUN_STEP" -ErrorAction SilentlyContinue
Write-Host "`n6 runs launched (독립 프로세스). 결과 → <build>\results\<timestamp>_*\"
Write-Host "분석: python analyze_run.py `"$(Split-Path $Exe -Parent)\results`""
