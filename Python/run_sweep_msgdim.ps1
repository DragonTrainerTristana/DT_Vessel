<#
MSG_DIM 2~12 sweep (H2: 메시지 latent 차원 ↑ → 통신 질 ↑?). C5c-aware (2026-06-22 갱신).

★★H2는 H1(C5c) 위에서만 의미가 있다 (CLAUDE.md + 4차 논검):
  - 통신이 *실제로 쓰여야* 차원이 의미. 수신측이 메시지를 안 쓰면(C5c 단절) MSG_DIM 2~12가 동일 수렴
    (실측 전례 있음). 그래서 이 sweep은 *반드시 C5c ON*(GoalComm 생산 + Consumer 수신)으로 돈다.
  - 순서: 먼저 run_sweep_c5c.ps1로 (a) oracle>OFF(통신 가치 존재) (b) ONc5c가 ONsevered/oracle 대비 개선
    을 확인 → *그 다음에* 이 MSG_DIM sweep이 의미. H1 미확인 상태로 돌리면 flat 위험(낭비).

⚠️ 정직한 예측(중요): 이 task의 사적 정보 = 파트너 *goal*(거리·방위) ≈ 2D. 즉 MSG_DIM=2면 이미 용량 충분.
  → C5c가 작동해도 H2 곡선이 *MSG_DIM=2에서 포화(flat)*일 수 있다. 그건 "통신 무용"이 아니라
    "이 task의 의도는 2D면 충분"이라는 *정직한 결과*. H2 스케일링(차원↑=개선)은 *더 고차원 사적 의도*가
    있어야 보임 → 곡선이 오르면 H2 양성, 평평하면 "intent가 저차원"으로 정직 보고(fix-until-win 금지).

★빌드: C5c/per-slot/coupling/intent는 Python 전용(재빌드 불필요)이나, ★T4 per-pair 보상(PerPairCoef)은 C#이라
   *새 빌드(0622, T4 포함)가 필요*. 0621 빌드로 돌리면 PerPairCoef가 무시돼 ground-truth가 평평할 수 있음(aux만 스케일).
   → ground-truth H2 스케일을 보려면 T4 포함 0622 빌드로. density 레버는 빌드(C#)가 제공. from-scratch 필수(MSG_DIM별 아키텍처 다름).
   ★assert msg_dim>=GOAL_SIZE(=2): C5c/oracle가 others_msg[:,:2]에 goal 주입 → 최소 dim 2 (2,4,6,8,10,12 모두 OK).

사용:
  .\run_sweep_msgdim.ps1 -Exe "...\Build\0621\Vessel_MLAgent.exe"
  .\run_sweep_msgdim.ps1 -Exe "..." -Dims 2,4,6,8,10,12 -Seeds 42,43,44 -RunStep 2000000
#>
param(
  [Parameter(Mandatory=$true)][string]$Exe,           # density 레버 포함 0621 빌드
  [int[]]$Dims    = @(2,4,6,8,10,12),                 # 6개 = 이 머신 1-wave 병렬 최적
  [int[]]$Seeds   = @(42),                            # 곡선 모양 먼저 1seed; 유의성은 43,44 추가
  [string]$RunStep = "2000000",                        # ★판정 깊이 2M(1M은 smoke). 필수 명시(미설정 시 config 30M 폭주)
  [string]$GoalComm = "0.05",                          # producer: 메시지에 자기 goal 인코딩 (C5c와 짝)
  [string]$Consumer = "0.05",                          # ★C5c: 수신측 decode-in-policy (이게 없으면 차원 flat 보장)
  [int]$ConsumerK  = 6,                                # ★H2 per-slot: nearest-K 이웃 *각각* 의도 복원(mean 아님). K 의도가 MSG_DIM 공유=차원 병목→스케일. 12 목표면 K≥6
  [int]$Coupling   = 1,                                # ★H2 행동-head coupling: 재구성을 fc3에 주입("재구성↓=행동↑" forward 항등) → ground-truth가 aux 추종
  [string]$PerPairCoef = "-0.3",                       # ★T4 다물체 페널티(C#, ★새 빌드 필요). 0이면 단일위협만→ground-truth 평평. 음수=다물체 회피 가격
  # ★H2 스케일 enabler (2026-06-22): 고차원 사적 의도 = K-step 미래 궤적(IntentDecoder 3K-D).
  #   goal(2D)만이면 MSG_DIM=2서 포화 → 차원 무의미. 미래궤적(IntentK=6→18D)이면 MSG_DIM이 진짜 병목 → 스케일 가능.
  #   IntentCoef=0이면 H2 곡선 평평 보장(전달정보 2D뿐). ★이건 정당한 enabler지 *스케일 강제*가 아님(곡선은 데이터가 결정).
  [string]$IntentCoef = "0.05",                        # 0이면 고차원 의도 없음(flat). >0=미래궤적 인코딩 ON
  [int]$IntentK    = 6,                                # 미래시점 개수 → 의도 차원 = 3*K (6→18D). 클수록 고차원
  [int]$Attn       = 1,                                # grounding+attention 집계
  [string]$RingScale = "0.5",                          # dense antipodal regime (통신이 가치 갖는 다물체 조정)
  [int]$VesselCount = 16
)
if (-not (Test-Path $Exe)) {
  Write-Host "[ERROR] exe 없음: $Exe (density 레버 포함 0621 빌드 필요)"
  exit 1
}

# ── 전 run 공통: C5c ON + dense regime + radar 56m(미설정). 자식 python→Unity 상속. ──
$env:VESSEL_USE_COMM = "1"
$env:VESSEL_USE_ATTENTION = "$Attn"
$env:VESSEL_GOAL_COMM_COEF = "$GoalComm"
$env:VESSEL_COMM_CONSUMER_COEF = "$Consumer"
$env:VESSEL_COMM_CONSUMER_K = "$ConsumerK"  # ★per-slot: K 이웃 각각 복원 → MSG_DIM rate-distortion 병목
$env:VESSEL_COMM_CONSUMER_COUPLING = "$Coupling"   # ★재구성→행동 head (ground-truth가 aux 추종)
$env:VESSEL_PERPAIR_COEF = "$PerPairCoef"   # ★T4 다물체 보상(C#, 새 빌드서만 발화) — ground-truth 스케일 수요측
$env:VESSEL_INTENT_COEF = "$IntentCoef"     # ★고차원 의도(미래궤적) — H2 스케일 enabler
$env:VESSEL_INTENT_K = "$IntentK"           # 의도 차원 = 3*K
$env:VESSEL_CROSSING = "2"
$env:VESSEL_SPAWN_RING_SCALE = "$RingScale"
$env:VESSEL_VESSEL_COUNT = "$VesselCount"
$env:VESSEL_RUN_STEP = "$RunStep"
Remove-Item Env:\VESSEL_RADAR_RANGE -ErrorAction SilentlyContinue   # 레이더 56m 보장
Remove-Item Env:\VESSEL_ORACLE -ErrorAction SilentlyContinue        # oracle 아님(실제 채널의 차원 효과 측정)

$port = 5200
foreach ($seed in $Seeds) {
  foreach ($d in $Dims) {
    $env:VESSEL_MSG_DIM = "$d"          # ★per-dim: spawn 시점 env를 자식이 상속
    $tag = "msgc5c{0}_s{1}" -f $d, $seed
    $a = @("-NoExit","-ExecutionPolicy","Bypass","-File","$PSScriptRoot\run_experiment.ps1",
           "-Exe","`"$Exe`"","-Seed",$seed,"-Comm",1,"-Port",$port,"-Agg","mean","-Tag",$tag,"-RunStep",$RunStep)
    Start-Process powershell -ArgumentList $a
    Write-Host ("launched {0}  port{1}  (MSG_DIM={2} C5c consumer={3} dense ring{4} n{5} radar56)" -f $tag,$port,$d,$Consumer,$RingScale,$VesselCount)
    $port++
    Start-Sleep -Seconds 4
  }
}
Remove-Item Env:\VESSEL_MSG_DIM,Env:\VESSEL_USE_COMM,Env:\VESSEL_USE_ATTENTION,Env:\VESSEL_GOAL_COMM_COEF,Env:\VESSEL_COMM_CONSUMER_COEF,Env:\VESSEL_COMM_CONSUMER_K -ErrorAction SilentlyContinue
Remove-Item Env:\VESSEL_COMM_CONSUMER_COUPLING,Env:\VESSEL_PERPAIR_COEF,Env:\VESSEL_INTENT_COEF,Env:\VESSEL_INTENT_K,Env:\VESSEL_CROSSING,Env:\VESSEL_SPAWN_RING_SCALE,Env:\VESSEL_VESSEL_COUNT,Env:\VESSEL_RUN_STEP -ErrorAction SilentlyContinue

$resRoot = Join-Path $PSScriptRoot '..\..\..\results'
Write-Host ("`n{0} runs ({1} dim x {2} seed). ★동시 ~6개 한계 → seed별로 나눠(한 wave=6 dim) 돌리세요." -f ($Dims.Count*$Seeds.Count), $Dims.Count, $Seeds.Count)
Write-Host "★전제: run_sweep_c5c.ps1로 oracle>OFF & ONc5c 개선을 *먼저* 확인. H1 미확인 시 이 sweep은 flat 위험."
Write-Host "분석(수렴 last 25~30%, seed-paired): python analyze_run.py `"$resRoot`"  (MSG_DIM별 vColl/fuel/DCPA/COLREGs 곡선)"
Write-Host "★H2 판정: MSG_DIM↑에 ground-truth 개선이 *단조 스케일*하면 H2 양성. MSG_DIM=2서 포화(flat)면 = intent가 저차원이라 충분 = 정직한 결과(통신 무용 아님). tensorboard Loss/CommConsumer로 차원별 디코드 학습 확인."
Write-Host "★★정직 의존성(H2-scaling-by-design): 이 sweep(per-slot consumer)은 **aux 재구성 MSE의 MSG_DIM 스케일**(채널용량 곡선)을 바로 보여줌. 단 **ground-truth(충돌/연료/DCPA) 스케일엔 (a)행동head coupling (b)per-pair 보상 T4(C# 재빌드)가 필요** — 없으면 정책이 2·3번째 이웃 의도를 쓸 gradient가 없어 aux만 스케일/ground-truth 평평. → 먼저 Loss/CommConsumer 차원곡선 확인 → 오르면 coupling+T4 wave로 ground-truth 전환."
Write-Host "★anti-Schelling: relpos-only 통제 arm 권장(consumer가 relpos 주소만으로 재구성하는지). 채널이 *추가로* 줄인 residual MSE만 H2로 인정. density는 OFF goal%>80% solvability gate 통과분만(impossibility-rigging 금지)."
