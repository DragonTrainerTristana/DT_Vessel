#!/bin/bash
# r0_extra.sh <prefix> <run> <k:x>... — R0 속도용 추가 프로세스(2026-10-27, 저자 승인 '진행해줘요')
#   meaneval_ckpts.sh 의 env 블록(20–47행)을 그대로 eval 하고, 같은 명령(eval_ckpt.py / eval_mean.py, 같은 EV_ARGS)으로
#   지정 체크포인트만 평가해 같은 파일 이름에 씀. 본 스크립트는 'epReward=' 가 있는 파일을 건너뛰므로 결과 동일.
#   k = s(표본) | m(평균), x = 6 · 8 · 9.04397(trunk 끝 모델) 처럼 meaneval 파일 이름의 M 앞 값
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
PRE=$1; run=$2; shift 2
R=/c/work/DT_Vessel_v3; HERE=$R/Python/runs_fig1/2026-10-02_fig1
cd "$R/Python" || exit 1
eval "$(sed -n '20,47p' "$HERE/meaneval_ckpts.sh")"
env_run "$run" || exit 1
base="$CK/${PRE}${run}_s$SEED"
echo "$(date '+%F %T') 추가 $PRE $run [$*] ARPA=$ARPA_R GPU=${CUDA_VISIBLE_DEVICES:-auto} $(git -C "$R" log -1 --oneline | cut -c1-7)" | tee -a "$D/_extra_log.txt"
for job in "$@"; do
  k=${job%%:*}; x=${job#*:}
  if [ "$run" = trunk_d6 ] && [ "$x" = 9.04397 ]; then f="$base.pt"; else f="$base.step${x}M.pt"; fi
  [ -f "$f" ] || { echo "★멈춤: 체크포인트 없음 $f"; exit 1; }
  if [ "$k" = s ]; then
    out="$D/sample_${PRE}${run}_${x}M.txt"
    grep -q 'epReward=' "$out" 2>/dev/null || "$PY" -u eval/eval_ckpt.py --ckpt "$f" $EV_ARGS > "$out" 2>&1
  else
    out="$D/mean_${PRE}${run}_${x}M.txt"
    grep -q 'epReward=' "$out" 2>/dev/null || "$PY" -u "$HERE/eval_mean.py" "$R/Python" -- --ckpt "$f" $EV_ARGS > "$out" 2>&1
  fi
  grep -q 'epReward=' "$out" || { echo "★FAIL 평가 실패 $f ($out)"; exit 1; }
  echo "$(date '+%F %T') $run ${x}M $k 끝 (추가 프로세스)" | tee -a "$D/_extra_log.txt"
done
