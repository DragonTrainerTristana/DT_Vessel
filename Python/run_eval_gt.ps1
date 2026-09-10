<#
vessel_gym P3 체크포인트의 Unity ground-truth 평가 (판정관 단계, 2026-07-06).

6 run 병렬: {OFF, ORACLE} × seed {42,43,44}. 각 run = VESSEL_TRAIN=0(PPO 업데이트 스킵) +
VESSEL_LOAD_MODEL=1(vgP3 체크포인트). regime은 학습과 동일(CROSSING=2, ring 0.7, 16척).
결과: results\<ts>_gteval_{arm}_s{seed}\ (outcome/metric/events.csv + run_meta.txt)
판정: 꼬리 아님(전 구간 = 이미 수렴 정책의 평가) — collision_vessel + goal, seed-paired.

사용:  .\run_eval_gt.ps1                     # 기본: Build\0703 exe, results\vg_stage1 모델
       .\run_eval_gt.ps1 -RunStep 200000     # 더 긴 평가 (표본 ↑)
#>
param(
  [string]$Exe = "",
  [string]$ModelDir = "",
  [int]$RunStep = 100000,
  [int[]]$Seeds = @(42, 43, 44),
  [string]$RingScale = "0.7"
)
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
if (-not $Exe)      { $Exe      = Join-Path $root 'Build\0703\Vessel_MLAgent.exe' }
if (-not $ModelDir) { $ModelDir = Join-Path $root 'results\vg_stage1' }
if (-not (Test-Path $Exe)) { Write-Host "[ERROR] exe 없음: $Exe"; exit 1 }
Write-Host ("빌드 날짜: {0}" -f (Get-Item $Exe).LastWriteTime)

# 공통 env (학습 regime과 동일 + eval 오버라이드)
$env:VESSEL_USE_EDITOR = "0"
$env:VESSEL_ENV_PATH   = $Exe
$env:VESSEL_NUM_ENVS   = "1"
$env:VESSEL_TRAIN      = "0"          # ★eval: PPO 업데이트 스킵
$env:VESSEL_LOAD_MODEL = "1"
$env:VESSEL_RUN_STEP   = "$RunStep"
$env:VESSEL_CROSSING   = "2"
$env:VESSEL_SPAWN_RING_SCALE = $RingScale
$env:VESSEL_VESSEL_COUNT = "16"
Remove-Item Env:\VESSEL_RADAR_RANGE -ErrorAction SilentlyContinue

$port = 5700
foreach ($arm in @("OFF", "ORACLE")) {
  foreach ($seed in $Seeds) {
    $model = Join-Path $ModelDir ("vgP3_{0}_s{1}.pt" -f $arm, $seed)
    if (-not (Test-Path $model)) { Write-Host "[ERROR] 모델 없음: $model"; continue }
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $runDir = Join-Path $root ("results\{0}_gteval_{1}_s{2}" -f $ts, $arm, $seed)
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $env:VESSEL_MODEL_PATH  = $model
    $env:VESSEL_SEED        = "$seed"
    $env:VESSEL_BASE_PORT   = "$port"
    $env:VESSEL_USE_COMM    = $(if ($arm -eq "ORACLE") { "1" } else { "0" })
    $env:VESSEL_ORACLE      = $(if ($arm -eq "ORACLE") { "1" } else { "0" })
    $env:VESSEL_OUTCOME_LOG = Join-Path $runDir "outcome.csv"
    $env:VESSEL_METRIC_LOG  = Join-Path $runDir "metric.csv"
    $env:VESSEL_EVENT_LOG   = Join-Path $runDir "events.csv"
    $lines = @("run = gteval_{0}_s{1}" -f $arm, $seed; "model = $model"; "--- VESSEL_* env ---")
    Get-ChildItem env: | Where-Object { $_.Name -like "VESSEL_*" } | Sort-Object Name |
      ForEach-Object { $lines += ("  {0} = {1}" -f $_.Name, $_.Value) }
    $lines | Out-File -Encoding utf8 (Join-Path $runDir "run_meta.txt")
    Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command",
      "cd '$PSScriptRoot'; `$env:PYTHONUTF8='1'; python main.py")
    Write-Host ("launched gteval {0} s{1} port{2} ← {3}" -f $arm, $seed, $port, (Split-Path $model -Leaf))
    $port++
    Start-Sleep -Seconds 4
  }
}
Remove-Item Env:\VESSEL_TRAIN, Env:\VESSEL_LOAD_MODEL, Env:\VESSEL_MODEL_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\VESSEL_USE_COMM, Env:\VESSEL_ORACLE, Env:\VESSEL_CROSSING, Env:\VESSEL_SPAWN_RING_SCALE -ErrorAction SilentlyContinue
Remove-Item Env:\VESSEL_VESSEL_COUNT, Env:\VESSEL_RUN_STEP -ErrorAction SilentlyContinue
Write-Host "`n6 run 병렬 eval — 완료 후 outcome.csv를 seed-paired로 분석 (collision_vessel + goal)."
