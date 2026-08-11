<#
★superseded (2026-07-03): run_sweep_commgate.ps1 사용 — 현 아키텍처(pos_ground 기본·situation 입력·
  파트너 4척·사각지대 fix 빌드) + far-field 수요 보상 반영. 이 파일은 06-22 아키텍처 기준 기록용.

C5c(수신측 decode-in-policy) + oracle-OFF 통제 sweep (2026-06-22, 4차 논검 "단 하나의 행동").

진단(4차 논검, 코드 실증): comm 무승부의 binding 원인 = C5c 단절.
  기존 aux decoder(intent/goal/role)는 own_msg 재실행→*생산자(MessageActor)*만 gradient.
  수신자 정책(ControlActor)이 others_msg를 *쓰도록* 강제하는 항이 없음 → 채널 LIVE여도 정책이 무시 → 무승부.
fix(Python 전용, 재빌드 불필요):
  ① C5c: ControlActor.consumer_decoder가 backbone z로 파트너 의도(goal) 복원 → MSE가 *수신측* fc2 메시지슬라이스로 gradient.
  ② oracle-OFF 통제: others_msg를 참 파트너 goal로 직접 주입(채널 우회) → "정보 가치" 천장 측정.

★레이더 56m 불변(사용자 하드제약, 2026-06-28 확정). perception nerf 없음. 56m에선 perception 가치=0이라
  통신 가치는 오직 (a) *의도(goal) 공유*(레이더는 현재위치만, 목적지 못 봄) + (b) 다물체 *협응*(equilibrium 선택)에서.
★supply(C5c 수신측 사용)+demand(per-pair 보상) 동시 투입 (2026-06-28):
  - demand = VESSEL_PERPAIR_COEF(전 arm 동일=anti-rigging): argmax 단일위협 대신 모든 동시접근쌍 가격 →
    "이웃 의도를 더 알수록 ground-truth 비용↓"이 성립 → 통신이 *벌* 게 생김(message-independent라 보상으론 못 이김).
  - supply = ONc5c arm의 Consumer+Coupling: 수신 정책이 메시지로 *행동*(represent→act). 둘 없으면 demand 있어도 무승부.
  ⚠️ per-pair는 C# 코드 → *그 코드가 든 빌드* 필요(이 프로젝트 소스로 재빌드). 옛 빌드면 per-pair 무발화(빌드 함정).

anti-rigging:
  - dense regime(CROSSING=2·ring0.5·n16)·radar 56m·per-pair = 전 arm 동일. OFF 추가페널티 없음(others_msg≡0이라 consumer/goalComm/oracle 미가산, per-pair는 message-independent라 ON·OFF 동일 부담).
  - C5c는 *수신 정책*을 형성(의도적 — 단절이 무승부 원인) → comm-ON은 여전히 ground-truth로 OFF를 이겨야 진짜.
    (C5c가 정책을 망치면 ON<OFF로 정직히 드러남 = 리깅 아님. ON-severed arm이 "C5c 없는 ON"의 통제.)
  - oracle는 진단/통제(comm-ON 주장 아님). 판정은 전부 ground-truth(VESSEL_OUTCOME_LOG/METRIC_LOG).

판정(수렴=마지막 25~30%, ≥3 seed, seed-paired, mid-training 금지):
  - oracle ≈ OFF        → task가 파트너 의도 불필요(C1/C4 거짓) → 정직한 종료(comm 무용 확정).
  - oracle > OFF, ON-severed ≈ OFF, ON-C5c → oracle  → ★C5c가 fix, comm 정당 승리.
  - oracle > OFF, ON-C5c > ON-severed(<oracle)        → C5c 부분 기여.
  - oracle > OFF, ON-C5c ≈ ON-severed ≈ OFF           → C5c로도 부족 → 더 깊은 채널/보상 재설계.
falsify: ON-C5c < OFF → C5c 보조과제가 정책 간섭(coef↓ 또는 기각).

사용:
  .\run_sweep_c5c.ps1                                  # -Exe 생략 시 프로젝트 루트 Build\Vessel_MLAgent.exe 사용
  .\run_sweep_c5c.ps1 -Exe "..\..\..\Build\0621\Vessel_MLAgent.exe"   # 특정 빌드 명시
#>
param(
  [string]$Exe = "",  # 비우면 프로젝트 루트 Build\Vessel_MLAgent.exe 사용. ★per-pair 코드 포함 빌드여야 함(06-22 18:04 라이브 소스). per-pair 前 빌드는 금지.
  [int]$RunStep = 1000000,          # 1M 잠정. 확정은 2000000(2M).
  [int[]]$Seeds = @(42,43,44),
  [string]$RingScale = "0.5",       # dense antipodal (ambiguity ~49%, goal-reach 유지=non-impossibility)
  [int]$VesselCount = 16,
  # ★per-pair de-argmax 보상 (2026-06-28, 56m+per-pair 전략): demand-side. 전 arm 동일 적용(message-independent
  #   privileged) = anti-rigging. radar 56m이라 perception 가치는 0이지만 *다물체 협응*을 벌 보상이 생김 →
  #   통신(C5c)이 그 협응을 메시지로 만족시키면 ON>OFF. ⚠️ per-pair는 C# 코드라 *그 코드가 든 빌드* 필요(빌드 함정).
  [string]$PerPairCoef = "-0.3",    # VESSEL_PERPAIR_COEF (0=off). 음수=다물체 동시접근 페널티
  [string]$PerPairExp  = "1.6"      # VESSEL_PERPAIR_EXP (쌍당 risk 지수, >1=센 쌍 우선)
)
# 상대경로화: 이 스크립트 위치(Assets\Scripts\Python) 기준 프로젝트 루트 = $PSScriptRoot\..\..\..
if (-not $Exe) { $Exe = Join-Path $PSScriptRoot '..\..\..\Build\Vessel_MLAgent.exe' }
if (-not (Test-Path $Exe)) { Write-Host "[ERROR] exe 없음: $Exe (per-pair 코드 포함 현재 소스 빌드 필요)"; exit 1 }

# arm: tag, comm, oracle, goalComm(생산자 인코딩), consumer(C5c 수신측), coupling(decode→fc3 행동 배선). attention 전 arm 동일(1).
#  - OFF        : 통신 없음 baseline
#  - ON_severed : 통신 ON, 생산자 goal 인코딩 ON, C5c OFF = "수신 단절" 재현(무승부 통제)
#  - ON_c5c     : + C5c 수신측 decode-in-policy ON ★fix. ★Coupling=1 *필수 동반*(2026-06-28 진단):
#                 Consumer만 켜면 수신측이 의도를 z에 *표상*만 하고 행동(fc3)엔 안 씀 → "represent≠act".
#                 Coupling=1이 decode(z)를 fc3 입력에 concat → "재구성→행동" forward 항등 = 정책이 메시지로 *행동*.
#  - ORACLE     : 참 파트너 goal 직접 주입(채널 우회) = 정보가치 천장 통제
$arms = @(
  @{Name="OFF";        Comm=0; Oracle=0; GoalComm="0.0";  Consumer="0.0";  Coupling="0"},
  @{Name="ONsevered";  Comm=1; Oracle=0; GoalComm="0.05"; Consumer="0.0";  Coupling="0"},
  @{Name="ONc5c";      Comm=1; Oracle=0; GoalComm="0.05"; Consumer="0.05"; Coupling="1"},
  @{Name="ORACLE";     Comm=1; Oracle=1; GoalComm="0.0";  Consumer="0.0";  Coupling="0"}
)

# ── 전 arm 공통 dense regime (C# 빌드가 env로 읽음; 자식 python→Unity 상속). radar 56m 불변(VESSEL_RADAR_RANGE 미설정). ──
$env:VESSEL_CROSSING = "2"                 # antipodal 중심관통
$env:VESSEL_SPAWN_RING_SCALE = $RingScale  # 링 축소 → 다물체 동시조우 ↑
$env:VESSEL_VESSEL_COUNT = "$VesselCount"
$env:VESSEL_USE_ATTENTION = "1"            # grounding+attention 집계(전 arm 동일)
$env:VESSEL_RUN_STEP = "$RunStep"          # ★명시 필수(config 기본 30M → 미설정 시 안 멈춤)
# ★per-pair de-argmax 보상 (demand-side, 전 arm 동일=anti-rigging). 단일위협 argmax 대신 모든 동시접근쌍을
#   가격 → "더 많은 이웃 의도를 알수록 ground-truth 비용↓"이 성립 → 통신이 *벌* 게 생김. message-independent라
#   통신은 보상으로 못 이기고 *메시지를 써서 협응*해야만 이김. coef=0이면 비트동일 baseline.
$env:VESSEL_PERPAIR_COEF = "$PerPairCoef"
$env:VESSEL_PERPAIR_EXP  = "$PerPairExp"
Remove-Item Env:\VESSEL_RADAR_RANGE -ErrorAction SilentlyContinue   # 레이더 56m(기본·사용자 하드제약) 보장

$port = 5500
foreach ($arm in $arms) {
  foreach ($seed in $Seeds) {
    $tag = "c5c_{0}_s{1}" -f $arm.Name, $seed
    $env:VESSEL_USE_COMM = "$($arm.Comm)"
    $env:VESSEL_ORACLE = "$($arm.Oracle)"
    $env:VESSEL_GOAL_COMM_COEF = "$($arm.GoalComm)"
    $env:VESSEL_COMM_CONSUMER_COEF = "$($arm.Consumer)"
    $env:VESSEL_COMM_CONSUMER_COUPLING = "$($arm.Coupling)"   # ★decode→fc3 행동 배선(ONc5c만 1)
    $a = @("-NoExit","-ExecutionPolicy","Bypass","-File","$PSScriptRoot\run_experiment.ps1",
           "-Exe","`"$Exe`"","-Seed",$seed,"-Comm",$arm.Comm,"-Port",$port,"-Agg","mean","-Tag",$tag)
    Start-Process powershell -ArgumentList $a
    Write-Host ("launched {0} port{1} comm{2} oracle{3} goalComm{4} consumer{5} coupling{6} (CROSSING2 ring{7} n{8} radar56 perpair{9}/{10})" -f `
      $tag,$port,$arm.Comm,$arm.Oracle,$arm.GoalComm,$arm.Consumer,$arm.Coupling,$RingScale,$VesselCount,$PerPairCoef,$PerPairExp)
    $port++
    Start-Sleep -Seconds 4
  }
}
Remove-Item Env:\VESSEL_USE_COMM,Env:\VESSEL_ORACLE,Env:\VESSEL_GOAL_COMM_COEF,Env:\VESSEL_COMM_CONSUMER_COEF,Env:\VESSEL_COMM_CONSUMER_COUPLING -ErrorAction SilentlyContinue
Remove-Item Env:\VESSEL_CROSSING,Env:\VESSEL_SPAWN_RING_SCALE,Env:\VESSEL_VESSEL_COUNT,Env:\VESSEL_USE_ATTENTION,Env:\VESSEL_RUN_STEP -ErrorAction SilentlyContinue
Remove-Item Env:\VESSEL_PERPAIR_COEF,Env:\VESSEL_PERPAIR_EXP -ErrorAction SilentlyContinue

Write-Host "`n$($arms.Count * $Seeds.Count) runs ($($arms.Count) arm x $($Seeds.Count) seed). ★이 머신 동시 ~6개 한계 → arm 2개씩 나눠 돌리세요:"
Write-Host "  1차: OFF + ORACLE (정보가치 천장부터: oracle>OFF여야 의미 있음. oracle≈OFF면 task가 comm 불필요 → 즉시 정직 종료)."
Write-Host "  2차: ONsevered + ONc5c (oracle>OFF 확인 후에만. C5c가 severed를 이기고 oracle에 근접하는가)."
Write-Host "★Python 전용 = 재빌드 불필요(0621 빌드 그대로). 동역학/아키텍처 변경(consumer head) → from-scratch."
Write-Host "분석(수렴 후, seed-paired): python analyze_run.py `"<results>`" ; python convergence_gate.py <metric.csv...> (LATE_COLLAPSE)."
Write-Host "tensorboard Loss/CommConsumer 하강 = 수신 정책이 파트너 의도 디코딩 학습 중(C5c 작동 신호)."
