<#
MSG_DIM 2~12 sweep (H2: 메시지 latent 차원 ↑ → 통신 질 ↑?). comm ON, from-scratch, 독립 프로세스 병렬.

★현 아키텍처(ARPA 제거 obs 369D + radar 30D, branch msgComparision)는 *새 빌드 필요*.
   기존 Build\MOE(390D)·0601 등은 obs 차원 불일치로 연결 실패. Unity에서 현재 C#로 재빌드 후 -Exe로 지정.
   (Editor 모드 VESSEL_USE_EDITOR=1는 자동 컴파일되나 인스턴스 1개 → 병렬 sweep 불가, 1개씩 순차만.)

⚠️ H2 주의(CLAUDE.md): 통신이 실제로 안 쓰이면(완전관측 잉여) MSG_DIM 2~12가 동일 수렴한 전례.
   차원 효과를 보려면 통신이 가치를 갖는 regime(레이더 축소 등)이 필요. 우선 동작/곡선 확인용으로 사용.

사용:
  .\run_sweep_msgdim.ps1 -Exe "C:\...\Build\<369D빌드>\Vessel_MLAgent.exe"
  .\run_sweep_msgdim.ps1 -Exe "..." -Dims 2,4,6,8,10,12 -Seed 42 -Agg sum -Attn 0 -RunStep 1000000
  (레이더 축소 regime 시험: 추가로 $env:VESSEL_RADAR_RANGE 설정 후 실행)
#>
param(
  [Parameter(Mandatory=$true)][string]$Exe,        # ★369D 새 빌드 경로 (필수)
  [int[]]$Dims    = @(2,4,6,8,10,12),              # 6개 = 이 머신 병렬 최적
  [int]$Seed      = 42,
  [string]$Agg    = "sum",                          # sum/mean (통신 집계)
  [int]$Attn      = 0,                              # 1=위치 grounding+attention 집계
  [string]$RunStep = "1000000"                      # run당 step (수렴 깊이). ★필수 명시(미설정 시 config 30M 폭주)
)
if (-not (Test-Path $Exe)) {
  Write-Host "[ERROR] exe 없음: $Exe"
  Write-Host "  → 현재 코드(369D)로 Unity 재빌드 필요. 기존 MOE/0601 빌드(390D 이하)는 obs 불일치로 연결 실패."
  exit 1
}

$env:VESSEL_USE_ATTENTION = "$Attn"   # 자식 프로세스 상속 (run_experiment.ps1은 MSG_DIM/ATTENTION 안 건드림)

$port = 5200
foreach ($d in $Dims) {
  $env:VESSEL_MSG_DIM = "$d"          # ★per-dim: spawn 시점 env를 자식이 상속
  $tag = "msg{0}_s{1}_agg{2}{3}" -f $d, $Seed, $Agg, $(if ($Attn -eq 1) {"_attn"} else {""})
  $a = @("-NoExit","-ExecutionPolicy","Bypass","-File","$PSScriptRoot\run_experiment.ps1",
         "-Exe","`"$Exe`"","-Seed",$Seed,"-Comm",1,"-Port",$port,"-Agg",$Agg,"-Tag",$tag,"-RunStep",$RunStep)
  Start-Process powershell -ArgumentList $a
  Write-Host ("launched {0}  port{1}  (MSG_DIM={2})" -f $tag, $port, $d)
  $port++
  Start-Sleep -Seconds 4   # 포트/Unity 연결 충돌 회피
}
# env 정리 (같은 세션에서 다른 sweep 이어 돌릴 때 상속 방지; 이미 spawn된 자식은 영향 없음)
Remove-Item Env:\VESSEL_MSG_DIM -ErrorAction SilentlyContinue
Remove-Item Env:\VESSEL_USE_ATTENTION -ErrorAction SilentlyContinue

Write-Host ("`n{0} runs launched (MSG_DIM {1}). 결과 → <build>\results\<timestamp>_msg*\" -f $Dims.Count, ($Dims -join ","))
Write-Host "분석(수렴 후): python analyze_run.py `"$(Split-Path $Exe -Parent)\results`"  (MSG_DIM별 goal/vColl/fuel 비교)"
Write-Host "★H2 판정: MSG_DIM↑에 따라 ground-truth(vColl·fuel)가 개선되는가. 동일 수렴이면 = 통신 미사용(차원 무의미)."
