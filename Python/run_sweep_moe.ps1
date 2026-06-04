<#
COLREGs Mixture-of-Experts 검증 sweep.
가설: 단일 정책망이 4상황(head-on/stand-on/give-way/overtake)의 상충 회피규칙을 평균내며 간섭 →
  상황별 head로 분리(hard-route)하면 각 상황 전문화 → 충돌↓(특히 rush>avoid 붕괴 완화).

★ MoE는 Python 전용(networks.py)이나, situation 라우터 obs(59→60D)가 C#에 추가됨 → *60D 재빌드 필요*.
   (구 59D 빌드면 obs_utils가 situation=0 폴백 → MoE가 head0로만 라우팅 = 의미없음. 반드시 새 빌드.)

anti-rigging (CLAUDE.md H1 honesty):
  - 두 arm 같은 빌드·seed·env, *오직 VESSEL_USE_MOE만 토글*. USE_MOE=0 = 기존 단일망 비트동일(검증 Δ=0).
  - 통신은 두 arm 모두 ON(공유 MessageActor) — MoE는 정책 head만 분리, 통신부 불변.
  - 진짜 승리 = moe arm이 single arm을 ground-truth(충돌·연료·궤적, analyze_run)로 이겨야.
  - 라우터=privileged 상황(보상과 동일 ground-truth)=CTDE 일관. PPO 정합=situation 저장→update 동일 재라우팅(검증완).

판정(수렴 마지막 30%, 3-seed): moe의 vColl·oColl·LATE가 single보다 낮으면 = 상황분리 효과.
falsify: moe ≈ single → 상황 간섭이 병목 아니었음(또는 라우팅 1-step stale / 희귀상황 head data-starve).

옵션:
  -RushFix : rush-fix wave-1 레버(arrival-debulk+dense-steepening) ON으로 floor 정리 후 MoE delta 측정(깨끗).
             기본 OFF = plain 보상(단일망이 붕괴하는 조건 → MoE가 그 붕괴를 막는지 직접 시험).

사용:
  .\run_sweep_moe.ps1 -Exe "C:\...\Build\FOUR\Vessel_MLAgent.exe"
  .\run_sweep_moe.ps1 -Exe "...\Build\FOUR\Vessel_MLAgent.exe" -RushFix   # floor 정리 버전
#>
param(
  [string]$Exe = "C:\Users\sengh\Dropbox\Private_Paper_Project\Vessel\Vessel_MLAgent\Build\FOUR\Vessel_MLAgent.exe",
  [switch]$RushFix
)
if (-not (Test-Path $Exe)) { Write-Host "[ERROR] exe 없음: $Exe (60D situation obs 재빌드 필요)"; exit 1 }

# ── 공통 env (두 arm 동일; 통신 ON = 검증된 comm1fix 설정) ──
$env:VESSEL_USE_COMM = "1"        # 통신 포함(user 요청). 공유 MessageActor — MoE는 정책 head만 분리.
$env:VESSEL_AGG_MODE = "mean"     # comm1fix(검증) 동일
$env:VESSEL_MSG_L2   = "0.01"     # comm1fix 동일 (H1a 게이트 안정화)
$env:VESSEL_RUN_STEP = "1000000"  # ★필수 명시 — 미설정 시 config 기본 30M(며칠 안 멈춤, 2026-06-03 실측버그)

if ($RushFix) {
  # rush-fix wave-1 레버 ON (floor 정리 → MoE delta 깨끗이 측정). 두 arm 동일 적용(anti-rigging).
  $env:VESSEL_ARRIVAL_REWARD = "45";  $env:VESSEL_GOAL_COEF = "1.35"
  $env:VESSEL_COLCOURSE_EXP  = "1.6"; $env:VESSEL_COLCOURSE_COEF = "-1.2"
  Write-Host "[mode] rush-fix 레버 ON (floor 정리 버전)"
} else {
  Write-Host "[mode] plain 보상 (단일망 붕괴 조건 — MoE가 막는지 직접 시험)"
}

# arm: VESSEL_USE_MOE만 토글 (0=단일망 비트동일 baseline, 1=COLREGs 5-head MoE)
$arms = @(
  @{Name="single"; Moe=0},   # 기존 단일망 (공정 baseline)
  @{Name="moe";    Moe=1}    # COLREGs 상황별 5-head MoE
)
$seeds = @(42, 43, 44)

$port = 5500
foreach ($arm in $arms) {
  foreach ($seed in $seeds) {
    $env:VESSEL_USE_MOE = "$($arm.Moe)"   # 자식 프로세스로 상속 (run_meta.txt에 자동 스냅샷)
    $tag = "moe_{0}_s{1}" -f $arm.Name, $seed
    if ($RushFix) { $tag = "rfmoe_{0}_s{1}" -f $arm.Name, $seed }
    $a = @("-NoExit","-ExecutionPolicy","Bypass","-File","$PSScriptRoot\run_experiment.ps1",
           "-Exe","`"$Exe`"","-Seed",$seed,"-Comm",1,"-Port",$port,"-Agg","mean","-Tag",$tag)
    Start-Process powershell -ArgumentList $a
    Write-Host ("launched {0} port{1} use_moe{2}" -f $tag, $port, $arm.Moe)
    $port++
    Start-Sleep -Seconds 4   # 포트/Unity 연결 충돌 회피
  }
}

# env 정리 (같은 세션 다음 sweep에 누설 방지)
foreach ($v in 'VESSEL_USE_MOE','VESSEL_USE_COMM','VESSEL_AGG_MODE','VESSEL_MSG_L2','VESSEL_RUN_STEP',
               'VESSEL_ARRIVAL_REWARD','VESSEL_GOAL_COEF','VESSEL_COLCOURSE_EXP','VESSEL_COLCOURSE_COEF') {
  Remove-Item "Env:\$v" -ErrorAction SilentlyContinue
}
Write-Host "`n6 runs (single×3 + moe×3, 통신 ON). 결과 → <build>\results\<ts>_*moe_*\"
Write-Host "분석(수렴 후): python analyze_run.py `"$(Split-Path $Exe -Parent)\results`""
Write-Host "핵심 비교: moe arm의 vColl/oColl/LATE < single arm = 상황분리 효과. ≈면 간섭이 병목 아니었음(falsify)."
Write-Host "★재빌드 확인: Assembly-CSharp.dll에 VESSEL_USE_MOE는 없어도 됨(Python전용). 단 obs 60D(situation)는 빌드에 있어야 — VectorObservationSize=60 확인."
