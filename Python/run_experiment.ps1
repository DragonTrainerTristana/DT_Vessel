<#
단일 학습 run 실행 + 모든 결과를 build 폴더의 results\<timestamp>_<name>\ 에 자동 정리.
  outcome.csv  : 에피소드 outcome (goal/collision_vessel/collision_obstacle/timeout)
  metric.csv   : 13열 per-episode 메트릭 (fuel/smoothness/commandVar/circling/near-miss)
  events.csv   : 충돌/도착의 공간 위치 (start→end, heading, speed) — 궤적/충돌분포 분석
  run_meta.txt : 정확한 설정 스냅샷 (후에 '무슨 설정이 이 결과를 냈나' 추적용)

사용:
  .\run_experiment.ps1 -Exe "C:\...\Build\0601_reward\Vessel_MLAgent.exe" -Seed 42 -Comm 0 -Port 5042
  (Comm 0=OFF/1=ON, Agg sum/mean — H1a 시험 시 ON은 mean 권장)
#>
param(
  [string]$Exe = "C:\Users\sengh\Dropbox\Private_Paper_Project\Vessel\Vessel_MLAgent\Build\0601_reward\Vessel_MLAgent.exe",
  [int]$Seed = 42,
  [int]$Comm = 0,
  [string]$Agg = "sum",
  [int]$Port = 5042,
  [string]$Tag = "",
  [string]$RunStep = ""
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Exe)) { Write-Host "[ERROR] exe 없음: $Exe (먼저 Unity 재빌드)"; exit 1 }
$buildDir = Split-Path $Exe -Parent
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
if ($Tag -ne "") { $name = $Tag } else { $name = "comm$Comm" + "_s$Seed" }
$runDir = Join-Path $buildDir ("results\{0}_{1}" -f $ts, $name)
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$env:VESSEL_USE_EDITOR  = "0"
$env:VESSEL_ENV_PATH    = $Exe
$env:VESSEL_LOAD_MODEL  = "0"
$env:VESSEL_NUM_ENVS    = "1"
$env:VESSEL_SEED        = "$Seed"
$env:VESSEL_BASE_PORT   = "$Port"
$env:VESSEL_USE_COMM    = "$Comm"
$env:VESSEL_AGG_MODE    = $Agg
$env:VESSEL_OUTCOME_LOG = Join-Path $runDir "outcome.csv"
$env:VESSEL_METRIC_LOG  = Join-Path $runDir "metric.csv"
$env:VESSEL_EVENT_LOG   = Join-Path $runDir "events.csv"
if ($RunStep -ne "") { $env:VESSEL_RUN_STEP = $RunStep }

# 설정 스냅샷 (보상 기본값은 빌드에 baked; 여기엔 실제 적용된 VESSEL_* env 전부 기록)
$lines = @()
$lines += "run        = $name"
$lines += "timestamp  = $ts"
$lines += "exe        = $Exe"
$lines += "reward baked default = angle0.15 / time-0.07 / earlyGate0.1 / relaxTcpa on / straightBonus off / msgLR1.0"
$lines += "(위 기본값은 해당 VESSEL_* env로 override 안 했을 때 적용. 아래는 실제 env 스냅샷)"
$lines += "--- VESSEL_* env ---"
Get-ChildItem env: | Where-Object { $_.Name -like "VESSEL_*" } | Sort-Object Name | ForEach-Object { $lines += ("  {0} = {1}" -f $_.Name, $_.Value) }
$lines | Out-File -Encoding utf8 (Join-Path $runDir "run_meta.txt")

Write-Host "[run] $name  →  $runDir"
Push-Location $PSScriptRoot
try { python main.py } finally { Pop-Location }
