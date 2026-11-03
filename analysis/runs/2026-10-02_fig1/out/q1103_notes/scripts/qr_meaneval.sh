#!/bin/bash
# 2026-11-03 author command 6): trunk meaneval for one seed after its extend ends.
#   = CUDA_VISIBLE_DEVICES=<GPU> VESSEL_PY=... VESSEL_F_REWARD_RANGE=56 VESSEL_F_SEED=<seed> bash .../meaneval_ckpts.sh <prefix> trunk_d6
# Split per checkpoint with VESSEL_F_ONLY (documented in meaneval_ckpts.sh, author-approved 10-27 for speed): same command,
# same args, same output files; then the table only (--table). Usage: qr_meaneval.sh <prefix> <seed> <gpuA> <gpuB>
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
PRE=$1 SEED=$2 GA=$3 GB=$4
cd /c/work/DT_Vessel_v3 || exit 1
LOG=/c/work/DT_Vessel_v3/run_meaneval_${PRE%_}.log
export VESSEL_PY=/c/Users/OSH/anaconda3/envs/mltest/python.exe VESSEL_F_REWARD_RANGE=56 VESSEL_F_SEED=$SEED
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3
echo "===== $(date '+%F %T') meaneval $PRE s$SEED GPU $GA/$GB (HEAD $(git rev-parse --short HEAD)) =====" >> "$LOG"
i=0; pids=()
for x in 2 4 6 8 9.04397; do
  g=$GA; [ $((i % 2)) -eq 1 ] && g=$GB
  ( CUDA_VISIBLE_DEVICES=$g VESSEL_F_ONLY="$x" bash Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh "$PRE" trunk_d6 > "/c/work/DT_Vessel_v3/run_meaneval_${PRE%_}_$x.log" 2>&1; echo "$(date '+%F %T') ckpt ${x}M GPU $g rc=$?" >> "$LOG" ) &
  pids+=($!); i=$((i + 1)); sleep 25
done
for p in "${pids[@]}"; do wait "$p"; done
bash Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh "$PRE" --table >> "$LOG" 2>&1
echo "===== $(date '+%F %T') meaneval $PRE done rc=$? =====" >> "$LOG"
