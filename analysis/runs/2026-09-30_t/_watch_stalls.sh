#!/bin/bash
# 학습 런 정지 감시(드라이버 내장 감시 대체, 2026-09-30 16:15).
#   내장 감시의 버그 2개: ① 런 이름을 명령줄 첫 't_…_s4N' 으로 잡아 갈래(--resume t_trunk…)를 뿌리 로그로 판정
#   ② Git Bash 가 taskkill 의 /F /T /PID 를 경로로 바꿔 실제로는 아무것도 못 죽임(16:09 기록된 kill 은 실행 안 됨).
#   여기선 이름 = --save 파일 이름, 죽이기 = PowerShell Stop-Process.
# 규칙: 학습 프로세스가 15분 넘게 떠 있고 그 로그가 LIM 분(기본 30) 넘게 안 바뀌면(끝난 런 제외) 그 프로세스만 죽이고 _kill_log.txt 에 기록.
#   드라이버의 train 단계가 재시도(최대 3회)에서 그 갈래를 trunk 부터 다시 돌림. 드라이버가 끝나면 이 감시도 끝.
O=/c/work/DT_Vessel_v3/Python/_repro_out_t
LIM=${VESSEL_STALL_MIN:-45}   # 2026-09-30 17:52 30→45분(t_a6_s43 창 17.5분까지 늘어남, GPU1 메모리 여유 1.2 GB). 17:10 15→30분: GPU1 을 다른 계정 작업과 나눠 쓰면서 t_a6_s43 로그 창(0.33M 결정)이 3.8분→13분으로 늘어남(진행 중). 15분이면 살아 있는 런을 오탐 kill → trunk 부터 재시작
TL=$O/_timeline.txt
while true; do
  sleep 120
  tail -3 "$TL" 2>/dev/null | grep -q "전 단계 완료\|★멈춤" && exit 0
  now=$(date +%s)
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'vessel_gym_train' -and \$_.CommandLine -match '--save \S*/t_' } | ForEach-Object { \$m=[regex]::Match(\$_.CommandLine,'--save \S*/(t_[A-Za-z0-9_]+_s4\d)\.pt'); '{0} {1} {2}' -f \$_.ProcessId, \$m.Groups[1].Value, [int64](New-TimeSpan -Start \$_.CreationDate -End (Get-Date)).TotalMinutes }" 2>/dev/null | tr -d '\r' | while read -r pid nm pmin; do
    [ -n "$nm" ] && [ -f "$O/$nm.log" ] || continue
    grep -q "saved ->" "$O/$nm.log" && continue
    age=$(( (now - $(date -r "$O/$nm.log" +%s)) / 60 ))
    if [ "${pmin:-0}" -ge 15 ] && [ "$age" -ge "$LIM" ]; then
      echo "$(date '+%F %T') $nm (PID $pid) kill — 로그 ${age}분 무진행, 프로세스 ${pmin}분 (GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')) [_watch_stalls]" >> "$O/_kill_log.txt"
      powershell -NoProfile -Command "Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
    fi
  done
done
