#!/bin/bash
# 10분 보고용 상태 요약(읽기만). 시각 · 드라이버 단계 · 런별 진행/결과/남은 시간 · 정지 기록 · 평가 곡선 · 3060 · GPU
O=/c/work/DT_Vessel_v3/Python/_repro_out_t
D=/c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t
now=$(date +%s)
echo "== $(date '+%F %T') =="
echo "-- 드라이버 마지막 3줄"; tail -3 "$O/_timeline.txt" | cut -c1-160
echo "-- 런 (결정·도착·선박충돌·벽·시간초과·R·epRet·속도·남은분·로그나이)"
for f in "$O"/t_*_s4?.log; do
  [ -f "$f" ] || continue
  nm=$(basename "$f" .log); case $nm in t_trunk*|t_init*) continue;; esac
  age=$(( (now - $(date -r "$f" +%s)) / 60 ))
  if grep -q "saved ->" "$f"; then st="끝"; else st="진행"; fi
  l=$(grep -h 'dec=' "$f" | tail -1)
  if [ -z "$l" ]; then echo "  $nm  (워밍업/시작, 로그 ${age}분 전) $st"; continue; fi
  echo "$l" | awk -v nm="$nm" -v age="$age" -v st="$st" -F'[= |~%]+' '{for(i=1;i<=NF;i++){if($i=="dec")d=$(i+1); if($i=="goal")g=$(i+1); if($i=="vColl")v=$(i+1); if($i=="oColl")o=$(i+1); if($i=="TO")t=$(i+1); if($i=="R")r=$(i+1); if($i=="epRet")e=$(i+1); if($(i+1)=="dec/s")s=$i}; d+=0; rem=(s>0)?(16.056-d)*1e6/s/60:-1; printf "  %-12s %5.2fM 도착%5.1f 선박%5.1f 벽%5.1f 초과%5.1f R%7.3f epRet%8.1f %5d/s 남은%4.0f분 로그%3d분 %s\n", nm, d, g, v, o, t, r, e, s, (st=="끝")?0:rem, age, st}'
done
echo "-- 정지·kill 기록(최근 3)"; tail -3 "$O/_kill_log.txt" 2>/dev/null | cut -c1-150
n_ec=$(ls "$O"/evalcurve/ec_*.txt 2>/dev/null | wc -l); n_ecd=$(grep -l "eps ||" "$O"/evalcurve/ec_*.txt 2>/dev/null | wc -l)
echo "-- 고정 평가 곡선: 끝 ${n_ecd} / 시작 ${n_ec}"
if [ -d "$D/other_pc/back/live" ]; then echo "-- 3060 Stage A: $(ls "$D/other_pc/back/live" | wc -l) 파일, 최근 $(ls -t "$D/other_pc/back/live" | head -1)"; else echo "-- 3060 Stage A: 동기화 데이터 없음(시작 여부 모름)"; fi
[ -f "$D/other_pc/back/_DONE.txt" ] && echo "   3060 끝: $(cat "$D/other_pc/back/_DONE.txt")"
echo "-- GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
