# 이벤트 하나 나면 끝남: ★멈춤/★FAIL · 로그 15분 무진행 · 단계 완료(trunk 모드 완료 · stage2 완료 · 절제 '끝 →')
cd /c/work/DT_Vessel_v3/Python
SEEN="$1"   # 이미 처리한 이벤트 키(공백 구분)
while true; do
  ev=""
  for t in qs qs2 qd3 qd4 qd5; do
    f=_repro_out_$t/_timeline.txt; [ -f $f ] || continue
    grep -q "★" $f && ev="$ev star:$t"
    grep -q "trunk 모드 완료" $f && ev="$ev done:$t"
    grep -q "stage2 완료" $f && ev="$ev done:$t"
  done
  for c in far56 decl0; do grep -q "끝 →" /c/work/DT_Vessel_abl/abl44_$c.log 2>/dev/null && ev="$ev done:abl_$c"; grep -q "★" /c/work/DT_Vessel_abl/abl44_$c.log 2>/dev/null && ev="$ev star:abl_$c"; done
  for p in qd3 qd4 qd5 qs qs2; do n=$(ls /c/work/DT_Vessel_v3/${p}_eval_*.log /c/work/DT_Vessel_v3/${p}eval_*.log 2>/dev/null | wc -l); [ $n -ge 4 ] && [ $(grep -l "끝 →" /c/work/DT_Vessel_v3/${p}_eval_*.log /c/work/DT_Vessel_v3/${p}eval_*.log 2>/dev/null | wc -l) -eq $n ] && ev="$ev evaldone:$p"; grep -Elq "★|Traceback" /c/work/DT_Vessel_v3/${p}_eval_*.log /c/work/DT_Vessel_v3/${p}eval_*.log 2>/dev/null && ev="$ev evalerr:$p"; done
  now=$(date +%s)
  for f in _repro_out_qs/qs_off_s43.log _repro_out_qs/qs_comm_s43.log _repro_out_qs2/qs2_off_s44.log _repro_out_qs2/qs2_comm_s44.log _repro_out_qd3/qd3_trunk_d6_s45.log _repro_out_qd4/qd4_trunk_d6_s46.log _repro_out_qd5/qd5_trunk_d6_s47.log; do
    [ -f $f ] || continue; d=$(dirname $f); grep -q "끝 branches\|끝 trunk" $d/_timeline.txt && { case $f in *trunk*) grep -q "끝 trunk" $d/_timeline.txt && continue;; *) grep -q "끝 branches" $d/_timeline.txt && continue;; esac; }
    age=$(( now - $(stat -c %Y $f) )); [ $age -gt 900 ] && ev="$ev stall:$(basename $f .log)"
  done
  new=""; for e in $ev; do case " $SEEN " in *" $e "*) ;; *) new="$new $e";; esac; done
  if [ -n "$new" ]; then date '+%F %T'; echo "NEW:$new"; echo "ALL:$ev"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo; exit 0; fi
  sleep 60
done
