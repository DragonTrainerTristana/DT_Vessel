#!/bin/bash
# 5분마다 live_reward.py → Dropbox runs/2026-09-30_t/live/reward_live.pdf·png·status.txt 갱신.
# 멈춤: t_ 드라이버가 '전 단계 완료' 또는 '★멈춤' 을 찍고 Stage A 도 끝났거나(back/_DONE.txt), 30시간 지남. 마지막에 한 번 더 그림.
D=/c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t
TL=/c/work/DT_Vessel_v3/Python/_repro_out_t/_timeline.txt
PYB=/c/Users/OSH/anaconda3/python.exe
end=$(( $(date +%s) + 30*3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  "$PYB" "$D/live_reward.py" > "$D/live/_live_last.log" 2>&1
  t_done=0; tail -3 "$TL" 2>/dev/null | grep -q "전 단계 완료\|★멈춤" && t_done=1
  a_done=0; [ -f "$D/other_pc/back/_DONE.txt" ] && a_done=1
  [ $t_done = 1 ] && [ $a_done = 1 ] && break
  sleep 300
done
"$PYB" "$D/live_reward.py" > "$D/live/_live_last.log" 2>&1
