#!/bin/bash
# 2026-09-30 t_ 배치 순서 조정(저자 17:4x '빨리'): 주 비교 평가(eval1)·핵심 표(table1)·옛 체크포인트 재채점(evalref)·궤적(traj)을 train2 와 동시에 돌림.
#   근거: 남의 GPU 점유(ljb llama-server 약 21 GB·jyh 학습 약 9 GB·데스크톱)로 GPU1·GPU3 엔 통신 런(4.8+1.5 GB)이 못 들어감 → 평가는 그 남는 자리에서 돌면 train2 를 안 늦춤.
#   실험 내용·설정·평가 프로토콜 변경 없음 — 명령은 _run_t.sh 의 evalref_old·traj 단계와 1:1, 순서만 바꿈.
#   드라이버는 타임라인의 '끝 eval1'·'끝 table1'·'끝 evalref'·'끝 traj' 표시(적용할 때 따로 씀)로 네 단계를 건너뜀.
#   고정 평가 곡선 루프는 적용 때 멈추고, 핵심 표(table1)가 나온 뒤 이 스크립트가 다시 켬(주 비교 평가가 GPU 자리를 먼저 쓰게).
#   train2 의 통신 런 배정이 먼저 끝나야 함(평가가 GPU 메모리를 먼저 잡으면 train2 가 한 바퀴 더 돎) → '시작 train2' 뒤 배정이 멎을 때까지 대기.
#   옛 체크포인트 재채점은 env 가 달라 사전 검사(14종)를 새로 돎 → 그 검사가 학습 GPU 를 안 건드리게 CUDA_VISIBLE_DEVICES="" (평가 작업은 run_repro 가 GPU 를 따로 지정).
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
R=/c/work/DT_Vessel_v3
DROP=/c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t
PRE=t_
cd $R/Python || exit 1
export PYTHONIOENCODING=utf-8 VESSEL_PY=/c/Users/OSH/anaconda3/envs/mltest/python.exe
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0
export VESSEL_COMM_TELEMETRY=1
export VESSEL_CKPT_DIR=$HOME/VESSEL_checkpoints/comm_intent
export VESSEL_BRANCH_WARMUP=2400
unset VESSEL_REQUIRE_TRUNK VESSEL_REUSE_ARMS VESSEL_SKIP_GOLDEN VESSEL_SEEDS VESSEL_NGPU
export VESSEL_RUN_PREFIX=$PRE VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035
export VESSEL_OUT_DIR=$PWD/_repro_out_${PRE%_}
O=$VESSEL_OUT_DIR; TL=$O/_timeline.txt
export VESSEL_RISK_DCPA_GATE_M=$(cat "$O/_gate_m.txt")
EVAL_GPU="VESSEL_JOBS=12 VESSEL_GPU_CAP=3 VESSEL_LAUNCH_GAP=10"
SEEDS=$(cat "$O/_seeds.txt")
log() { echo "$(date '+%F %T') $*" | tee -a "$TL"; }
sync_out() { cp "$O"/*.txt "$O"/*.log "$O"/*.csv "$O"/*.json "$DROP/out/" 2>/dev/null; cp "$O"/traj_*.pt "$O"/*.png "$O"/*.pdf "$DROP/out/" 2>/dev/null; true; }
# 1) train2 시작 대기 → 배정이 멎을 때까지(마지막 '→ GPU' 줄 뒤 3분 조용 & 시작 뒤 12분 이상, 최대 25분)
until grep -q " 시작 train2\$" "$TL"; do
  tail -3 "$TL" | grep -q "★멈춤\|전 단계 완료" && { echo "드라이버 멈춤/끝 — 동시 단계 안 돌림"; exit 1; }
  sleep 60
done
t0=$(date +%s)
while :; do
  now=$(date +%s); el=$(( now - t0 ))
  last=$(date -r "$O/_train2.log" +%s 2>/dev/null || echo $now)
  quiet=$(( now - $(stat -c %Y "$O/_train2.log" 2>/dev/null || echo $now) ))
  n=$(grep -c " → GPU " "$O/_train2.log" 2>/dev/null)
  { [ "$el" -ge 720 ] && [ "$quiet" -ge 180 ]; } && break
  [ "${n:-0}" -ge 9 ] && [ "$el" -ge 300 ] && break
  [ "$el" -ge 1500 ] && break
  sleep 30
done
log "동시: train2 배정 $(grep -c ' → GPU ' "$O/_train2.log" 2>/dev/null)개 뒤 시작 — eval1·table1·evalref·traj 를 train2 와 동시에"
fail=0
run_side() {   # 이름 명령… — 드라이버 step() 과 같은 판정(rc·로그의 FAIL/실패), 멈춤 대신 '★동시 실패' 기록
  local nm=$1; shift
  log "동시: 시작 $nm"
  "$@" > "$O/_$nm.log" 2>&1; local rc=$?
  if [ $rc -ne 0 ] || grep -q "★FAIL\|FAIL \|실패" "$O/_$nm.log"; then log "★동시 실패 $nm rc=$rc — $O/_$nm.log 확인"; fail=1
  else log "동시: $nm 완료(rc=0)"; fi
  sync_out
}
quick_table() {   # _run_t.sh 의 quick_table 과 1:1
  "$VESSEL_PY" "$DROP/report_parse.py" --dir "$O" --prefix "$PRE" --off off --arms a6 --noise offb --seeds "$(echo $SEEDS | tr ' ' ,)" \
      --extra_off "s_off=$(cygpath -m "$O"),x_off=$(cygpath -m "$O")" --json "$O/_primary_table.json" > "$O/_primary_table.md" 2>&1
  cat "$O/_primary_table.md"
}
run_side eval1 env VESSEL_NGPU=4 VESSEL_SEEDS="$SEEDS" $EVAL_GPU bash run_repro.sh eval
run_side table1 quick_table
MSYS_NO_PATHCONV=1 schtasks /run /tn vessel_evalcurve >/dev/null 2>&1 && log "동시: 고정 평가 곡선 루프 다시 켬"
run_side evalref env CUDA_VISIBLE_DEVICES= VESSEL_NGPU=4 VESSEL_SEEDS="43 44 45" VESSEL_RUN_PREFIX= VESSEL_FORWARD_COEF=0.1 VESSEL_TIME_PENALTY=0.07 VESSEL_RISK_DCPA_GATE_M=0 \
      VESSEL_ROLE_JUDGE=end VESSEL_EVAL_EXTRA="--role_judge v2" VESSEL_EVAL_NAMES="s_off x_off h_a6" $EVAL_GPU bash run_repro.sh evalref
run_side traj env VESSEL_NGPU=4 VESSEL_SEEDS="43" VESSEL_TRAJ_ARMS="off a6" $EVAL_GPU bash run_repro.sh traj
log "동시: 끝(실패 $fail)"
