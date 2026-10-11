#!/bin/bash
# 평균 행동 주 평가: f2_ 학생 5시드(목표 침로·ARPA@56, --arm ON) + i_ 학생 5시드(타각·레이더만, --arm OFF)
R=/c/work/DT_Vessel_v3; cd $R/Python
PY=/c/Users/OSH/anaconda3/envs/mltest/python.exe; W=$(cygpath -m "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/meaneval"); CK=C:/Users/OSH/VESSEL_checkpoints/comm_intent
export PYTHONIOENCODING=utf-8 FOR_DISABLE_CONSOLE_CTRL_HANDLER=1 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3 OMP_NUM_THREADS=2
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0 VESSEL_CKPT_DIR=$HOME/VESSEL_checkpoints/comm_intent
export VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035 VESSEL_RISK_DCPA_GATE_M=48
source $R/Python/common_env.sh; common_env
i=0
for s in 43 44 45 46 47; do
  ( export CUDA_VISIBLE_DEVICES=$((i % 4)) VESSEL_ACTION_MODE=course VESSEL_USE_COMM=1 VESSEL_MSG_DIM=6 VESSEL_COMM_LATENT=0.0 VESSEL_AUX_LOSS_SCALE=0.0 VESSEL_COMM_FIELDS=state VESSEL_PARTNER_RANGE=56
    $PY -u "$W/eval_mean.py" $R/Python -- --ckpt $CK/f2_dagger_d6_s$s.pt --arm ON --envs 256 --eval_decisions 10000 --burnin 2400 --seed 999 > "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/meaneval/meaneval_f2_dagger_d6_s$s.txt" 2>&1 ) &
  i=$((i + 1)); sleep 15
  ( export CUDA_VISIBLE_DEVICES=$((i % 4)) VESSEL_MSG_DIM=6 VESSEL_USE_COMM=0
    $PY -u "$W/eval_mean.py" $R/Python -- --ckpt $CK/i_dagger_d6_s$s.pt --arm OFF --envs 256 --eval_decisions 10000 --burnin 2400 --seed 999 > "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/meaneval/meaneval_i_dagger_d6_s$s.txt" 2>&1 ) &
  i=$((i + 1)); sleep 15
done
wait
for f in "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad"/meaneval/meaneval_*.txt; do echo "$(basename $f .txt): $(grep -o 'goal=.*eps' $f | head -1) $(grep -c 'eval_mean\] mean-action' $f)"; done
