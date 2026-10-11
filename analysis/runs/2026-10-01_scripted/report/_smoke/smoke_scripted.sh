#!/bin/bash
# smoke: same env as _run_scripted.sh batch_env, tiny window, output to scratchpad only
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1 PYTHONIOENCODING=utf-8
R=/c/work/DT_Vessel_v3/Python
PY=/c/Users/OSH/anaconda3/envs/mltest/python.exe
DROP=/c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-01_scripted
CK=$HOME/VESSEL_checkpoints/comm_intent
SP=$(dirname "$0")
cd "$R" || exit 1
source "$R/common_env.sh"
common_env
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0 VESSEL_CKPT_DIR=$CK
export VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035 VESSEL_RISK_DCPA_GATE_M=48
export VESSEL_MSG_DIM=6 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${1:-2}
for p in vo300 goal; do
  t0=$(date +%s)
  "$PY" -u "$DROP/eval_scripted.py" --py_root "$R" --policy $p -- \
    --ckpt "$CK/t_off_s43.pt" --arm OFF --envs 8 --eval_decisions 40 --burnin 10 --seed 999 > "$SP/smoke_$p.txt" 2>&1
  echo "$p rc=$? $(( $(date +%s) - t0 ))s"
done
