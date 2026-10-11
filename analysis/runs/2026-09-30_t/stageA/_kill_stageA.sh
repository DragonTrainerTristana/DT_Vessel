#!/bin/bash
# Stage A 정지 런 kill (2026-09-30 13:2x 외부 사용자 GPU 점유 4–8 GB 가 겹쳐 GPU 0–2 가 15–16 GB → WDDM 페이징 → 6런 전부 로그 무진행 ≥15분,
#   CPU 는 100 % 스핀(90 s 에 CPU 194 s 증가) — s_/r_ 배치의 정지와 같은 모양)
# 절차(메모리 규약): 로그 15분 무진행 확인 → 그 런 kill → 사망 확인 → _kill_log.txt 기록. GPU 리셋·재부팅 금지.
O=/c/work/DT_Vessel/Python/_repro_out_s_stageA
now=$(date +%s)
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 's_(a6|c6)_' } | ForEach-Object { \$m=[regex]::Match(\$_.CommandLine,'s_(a6|c6)_s4\d'); '{0} {1}' -f \$_.ProcessId, \$m.Value }" | tr -d '\r' | while read -r pid nm; do
  [ -n "$nm" ] || continue
  age=$(( (now - $(date -r "$O/$nm.log" +%s)) / 60 ))
  if [ "$age" -ge 15 ]; then
    echo "$(date '+%F %T') $nm (PID $pid) kill — 로그 ${age}분 무진행, GPU mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')" | tee -a "$O/_kill_log.txt"
    taskkill /F /T /PID "$pid" >/dev/null 2>&1
  else
    echo "$nm 은 ${age}분 — 15분 미만, 안 죽임"
  fi
done
sleep 8
left=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine" 2>/dev/null | grep -c "s_a6_\|s_c6_")
echo "남은 Stage A 학습 프로세스: ${left:-0}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
mkdir -p /c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t/stageA
cp "$O/_kill_log.txt" /c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t/stageA/ 2>/dev/null; true
