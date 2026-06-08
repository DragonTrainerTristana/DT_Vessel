<#
위치 grounding + Attention 통신 검증 sweep.
가설(H1b): sum은 "어느 방위 위협인지" 못 실어 인지부족 regime에서도 통신이 잉여였다.
  grounding+attention은 [상대위치 ⊕ msg]를 softmax 가중선택 → "상대가 어디로 가는지"를 전달 →
  레이더 축소(국소 관측) regime에서 통신이 *비잉여*가 되어 comm-ON이 회복.

★ attention은 Python 전용(networks.py) → *재빌드 불필요*, 기존 59D 빌드(positions obs[57:59])에서 바로 동작.
   (rush-fix 레버와 결합하려면 0602_rushfix 빌드로, 통신만 격리 시험하려면 0601_reward로.)

anti-rigging 정직 설계 (CLAUDE.md H1 honesty):
  - regime(레이더 축소)은 *모든 arm에 동일* 적용 → comm-OFF를 *추가* 페널티로 불구화하지 않음.
  - 보상 risk는 불변(privileged/CTDE): obs는 국소(VESSEL_RADAR_RANGE), 보상은 먼 위협도 앎 →
    통신이 "obs로 못 본 위협"을 복원하는 *유일한* 채널. OFF는 그 채널이 없을 뿐(대칭).
  - 진짜 승리 = redATTN이 (a) redOFF를 이기고(in-regime) (b) ref-OFF(완전관측 최선 baseline)에 근접/초과
    — 전부 ground-truth(VESSEL_OUTCOME_LOG/METRIC_LOG: 충돌·연료·궤적)로.
  - attention이 sum보다 나은지: redATTN vs redSUM(같은 regime, 집계만 다름).

판정(수렴=마지막 30%, 3-seed): redATTN의 vColl·fuel·near-miss가 redOFF보다 유의 낮고 ref-OFF에 근접.
  H1a 점검: redATTN이 redOFF보다 *나쁘면* = value-of-info<0 = 버그(게이트/zero-init 점검).
falsify: redATTN ≈ redOFF(regime서도 통신 무용) → grounding으로도 부족, 메시지 내용/보상범위 재설계로.

사용:
  .\run_sweep_attention.ps1 -Exe "C:\...\Build\0601_reward\Vessel_MLAgent.exe"
#>
param(
  [string]$Exe = "C:\Users\sengh\Dropbox\Private_Paper_Project\Vessel\Vessel_MLAgent\Build\0601_reward\Vessel_MLAgent.exe",
  [int]$RadarReduced = 28   # 국소 관측 regime (기본 레이더 56m → 28m 절반). COMM_RANGE 420m은 불변 → 통신여지.
)
if (-not (Test-Path $Exe)) { Write-Host "[ERROR] exe 없음: $Exe (59D positions 빌드 필요)"; exit 1 }

# arm 정의: tag, comm, agg, attention, intent(Phase2 self-supervised coef), radar(빈문자=완전관측 baseline)
# ★검증됨(코드): VESSEL_RADAR_RANGE는 obs(레이더)만 축소, 보상은 전체선박 privileged 56m → radar<56이면
#   (radar,56] 띠 위협이 "보상은 알지만 obs는 못 봄" = CTDE 격차가 *이미* 생김(재빌드 불필요).
$arms = @(
  @{Name="refOFF";    Comm=0; Agg="sum";  Attn=0; Intent=0;    Radar=""},            # 완전관측 최선 baseline (anti-rigging 기준)
  @{Name="redOFF";    Comm=0; Agg="sum";  Attn=0; Intent=0;    Radar="$RadarReduced"}, # 인지부족, 통신 없음
  @{Name="redSUM";    Comm=1; Agg="mean"; Attn=0; Intent=0;    Radar="$RadarReduced"}, # 인지부족, 통신 sum(grounding 없음)
  @{Name="redATTN";   Comm=1; Agg="mean"; Attn=1; Intent=0;    Radar="$RadarReduced"}, # 인지부족, grounding+attention ★
  @{Name="redINTENT"; Comm=1; Agg="mean"; Attn=1; Intent=0.05; Radar="$RadarReduced"}  # +intent(미래의도, 시간축) ★Phase2
)
$seeds = @(42, 43, 44)

# ★RUN_STEP 명시 필수: 미설정 시 config.py 기본 30,000,000(30M)으로 돌아 며칠간 안 멈춤(2026-06-03 실측버그).
#   rushfix/moe/sweep 런처와 동일 1M 깊이. 자식 프로세스(run_experiment.ps1)가 spawn 시점 env 상속.
$env:VESSEL_RUN_STEP = "1000000"

$port = 5400
foreach ($arm in $arms) {
  foreach ($seed in $seeds) {
    $tag = "attn_{0}_s{1}" -f $arm.Name, $seed
    # arm별 env (자식 프로세스로 상속; run_meta.txt에 자동 스냅샷)
    $env:VESSEL_USE_ATTENTION = "$($arm.Attn)"
    $env:VESSEL_INTENT_COEF = "$($arm.Intent)"   # 0=OFF(비트동일), >0=intent self-supervised ON
    if ($arm.Radar -ne "") { $env:VESSEL_RADAR_RANGE = $arm.Radar } else { Remove-Item Env:\VESSEL_RADAR_RANGE -ErrorAction SilentlyContinue }
    $a = @("-NoExit","-ExecutionPolicy","Bypass","-File","$PSScriptRoot\run_experiment.ps1",
           "-Exe","`"$Exe`"","-Seed",$seed,"-Comm",$arm.Comm,"-Port",$port,"-Agg",$arm.Agg,"-Tag",$tag)
    Start-Process powershell -ArgumentList $a
    Write-Host ("launched {0} port{1} comm{2} agg{3} attn{4} intent{5} radar{6}" -f $tag,$port,$arm.Comm,$arm.Agg,$arm.Attn,$arm.Intent,$(if($arm.Radar){$arm.Radar}else{"full"}))
    $port++
    Start-Sleep -Seconds 4   # 포트/Unity 연결 충돌 회피
  }
}
Remove-Item Env:\VESSEL_USE_ATTENTION -ErrorAction SilentlyContinue
Remove-Item Env:\VESSEL_INTENT_COEF -ErrorAction SilentlyContinue
Remove-Item Env:\VESSEL_RADAR_RANGE -ErrorAction SilentlyContinue
Remove-Item Env:\VESSEL_RUN_STEP -ErrorAction SilentlyContinue
Write-Host "`n15 runs (5 arm x 3 seed). ★동시 15개는 이 머신 과부하 → arm 2~3개씩 나눠 돌리세요(refOFF+redOFF 먼저, 그다음 redATTN/redINTENT)."
Write-Host "★재빌드 불필요(intent·attention 전부 Python; CTDE 격차도 radar env만으로 생김). 기존 0601 빌드로 즉시."
Write-Host "분석(수렴 후): python analyze_run.py `"$(Split-Path $Exe -Parent)\results`"  (arm별 vColl/fuel/near-miss 비교)"
Write-Host "핵심 비교: redATTN/redINTENT < redOFF(통신 가치) / <= refOFF(완전관측 회복) / redATTN < redSUM(attn>sum) / redINTENT <= redATTN(intent 시간축 보강)"
Write-Host "진단: tensorboard Loss/Intent 가 내려가면 메시지가 미래의도 인코딩(falsify: 내려가는데 ground-truth 무변=미래의도가 ARPA로 이미 충분)."
