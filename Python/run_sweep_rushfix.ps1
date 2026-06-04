<#
rush>avoid 붕괴 fix — 검증 sweep (wave-1).
진단(워크플로우 18-agent, ground-truth): 수렴 vColl≈6%는 soft-penalty EV 평형 p_eq=detour_cost/(300·γ^D).
0601 재설계는 path-dependent 소액만 깎아 평형을 못 움직임 → OFF 3/3 LATE_COLLAPSE 회귀.

wave-1 레버 (둘 다 anti-rigging 대칭=ON/OFF 동일 적용, freeze-위험 낮음, 상호보완):
  1) arrival-debulk : arrival 100→45 + goalCoef 1.0→1.35(=(260-45)/160, exact-isolation).
       종단 +100의 peaky goal-pull(근접목표 detour_cost의 ~90%)을 축소 → near-goal rush 제거.
       잃은 견인은 telescoping progress로 이전(path-independent=timeout 회귀 방지, farming 불가).
  2) dense-steepening: COLCOURSE_EXP 1.0→1.6 + COEF -0.8→-1.2.
       dense risk가 56m부터 1.0 포화→flat -0.8(거리 gradient 0) 문제를 risk^1.6로 비선형화 →
       관측된 moderate-DCPA sudden-tail(minVD~40m) 충돌의 적분비용을 충돌 전에 키움 + 조기변침>늦은변침 신호 복원.

⚠️ 이 빌드(working-tree)를 *반드시 재빌드* 후 실행(arrival은 Initialize에 baked, 새 env는 빌드 전엔 무시됨).
⚠️ 판정은 수렴(~1M step, 마지막 30%, 3-seed)에서만. mid-training 좋아도 채택 금지(과거 4번 속음).
   성공 = vColl 6%→3~4%대, LATE 0/3, goal≥83~86%, TO≤10~11%, straightness 0.85~0.87(과하락=circling).
   H1a 점검 = comm-ON이 *강화된* OFF를 ground-truth로 이겨야 진짜(못 이기고 더 나쁘면=버그).
실패(falsify) = 강한 param에도 vColl 불변 → 병목=perception runway(risk 늦게 상승), reward 아님 → wave-2 prox-ramp로.

사용:
  .\run_sweep_rushfix.ps1 -Exe "C:\...\Build\0602_rushfix\Vessel_MLAgent.exe"
#>
param(
  [string]$Exe = "C:\Users\sengh\Dropbox\Private_Paper_Project\Vessel\Vessel_MLAgent\Build\0602_rushfix\Vessel_MLAgent.exe"
)
if (-not (Test-Path $Exe)) { Write-Host "[ERROR] exe 없음: $Exe (먼저 Unity 재빌드 — 새 env는 재빌드 전엔 무시됨)"; exit 1 }

# ── wave-1 rush-fix 레버 (자식 프로세스로 상속됨; run_meta.txt에 자동 스냅샷) ──
$env:VESSEL_ARRIVAL_REWARD = "45"     # 100→45 : 종단 detour_cost ×0.45
$env:VESSEL_GOAL_COEF      = "1.35"   # (260-45)/160 : 낮춘 종단 goal-pull을 telescoping progress로 보전
$env:VESSEL_COLCOURSE_EXP  = "1.6"    # risk^1.6 : 포화 제거(거리 gradient 복원), moderate-DCPA 스티프닝
$env:VESSEL_COLCOURSE_COEF = "-1.2"   # -0.8→-1.2 : head-on(포화 risk=1)에선 EXP 무효라 COEF가 head-on 레버
# ON arm을 *검증된* comm1fix 설정과 동일하게: 메시지 게이트 fix(Python, networks.py)는 항상 ON이지만
#   MSG_L2는 env. 0601 comm1fix(H1a 검증 Pareto승)가 0.01로 돌았으므로 ON arm의 H1a 점검이 유효하려면 동일해야 함.
#   OFF arm은 others_msg≡0이라 MSG_L2 무영향(anti-rigging 안전).
$env:VESSEL_MSG_L2 = "0.01"           # comm1fix(검증됨)와 동일 — ON arm H1a 점검 유효성 보장
# ★RUN_STEP 명시 필수: 미설정 시 config.py 기본 30,000,000(30M)으로 돌아 며칠간 안 멈춤(2026-06-03 실측 버그).
#   ONE sweep(1M)과 동일 깊이 비교하려면 1M 고정. (run_experiment.ps1은 -RunStep 미전달 시 env 안 건드림)
$env:VESSEL_RUN_STEP = "1000000"

# anti-rigging: OFF/ON 같은 빌드·seed·env, 통신만 토글. ON은 mean 집계(zero-init+gate와 함께 value-of-info≥0).
$combos = @(
  @{Comm=0; Seed=42; Port=5242; Agg="sum"},
  @{Comm=0; Seed=43; Port=5243; Agg="sum"},
  @{Comm=0; Seed=44; Port=5244; Agg="sum"},
  @{Comm=1; Seed=42; Port=5342; Agg="mean"},
  @{Comm=1; Seed=43; Port=5343; Agg="mean"},
  @{Comm=1; Seed=44; Port=5344; Agg="mean"}
)
foreach ($c in $combos) {
  $tag = "rushfix_comm{0}_s{1}" -f $c.Comm, $c.Seed
  $a = @("-NoExit","-ExecutionPolicy","Bypass","-File","$PSScriptRoot\run_experiment.ps1",
         "-Exe","`"$Exe`"","-Seed",$c.Seed,"-Comm",$c.Comm,"-Port",$c.Port,"-Agg",$c.Agg,"-Tag",$tag)
  Start-Process powershell -ArgumentList $a
  Write-Host ("launched {0} port{1} agg{2}" -f $tag, $c.Port, $c.Agg)
  Start-Sleep -Seconds 4   # 포트/Unity 연결 충돌 회피
}
# env 정리: 같은 PowerShell 세션에서 *다른* sweep을 이어 돌릴 때 rush-fix 레버가 조용히 상속되는 것 방지.
#   (이미 Start-Process된 자식들은 spawn 시점 env를 받았으므로 영향 없음 — 이후 sweep 격리용.)
foreach ($v in 'VESSEL_ARRIVAL_REWARD','VESSEL_GOAL_COEF','VESSEL_COLCOURSE_EXP','VESSEL_COLCOURSE_COEF','VESSEL_MSG_L2','VESSEL_RUN_STEP') {
  Remove-Item "Env:\$v" -ErrorAction SilentlyContinue
}
Write-Host "`n6 runs launched (wave-1 rush-fix: arrival-debulk + dense-steepening). 결과 → <build>\results\<ts>_rushfix_*\"
Write-Host "분석(수렴 후): python analyze_run.py `"$(Split-Path $Exe -Parent)\results`""
Write-Host "wave-2 후보(필요시): VESSEL_PROX_RAMP_COEF=-2.5(prox-ramp, 버그수정완료) / VESSEL_SPEED_AVOID_UNLOCK=1(Rule-8 감속, ×0.3 freeze가드)"
