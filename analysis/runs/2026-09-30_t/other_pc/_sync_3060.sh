#!/bin/bash
# 3060 PC 학습 진행을 서버의 실시간 감시 그림(runs/2026-09-30_t/live/reward_live.pdf)에 보이게 하는 보조 스크립트.
# 5분마다 Stage A 곡선 CSV·로그를 이 폴더 back/live/ 로 복사함(작은 텍스트 파일만). 학습에는 영향 없음.
# 실행: Stage A 스크립트와 *다른* Git Bash 창에서 한 줄
#   bash "<Dropbox>/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t/other_pc/_sync_3060.sh"
# 멈춤: back/_DONE.txt 가 생기면(학습 끝) 마지막으로 한 번 복사하고 끝. 24시간이 지나도 끝.
R=${VESSEL_REPO:-/c/work/DT_Vessel}
B="$(cd "$(dirname "$0")" && pwd)"
O="$R/Python/_repro_out_s_stageA"
mkdir -p "$B/back/live"
end=$(( $(date +%s) + 24*3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  for f in "$O"/s_a6_s4?.csv "$O"/s_c6_s4?.csv "$O"/s_a6_s4?.log "$O"/s_c6_s4?.log "$O"/_timeline.txt "$O"/_status_train.txt; do
    [ -f "$f" ] && cp "$f" "$B/back/live/" 2>/dev/null
  done
  echo "$(date '+%F %T') 복사함 ($(ls "$B/back/live" | wc -l) 파일)"
  [ -f "$B/back/_DONE.txt" ] && break
  sleep 300
done
