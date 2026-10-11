#!/bin/bash
# 정면 조우 진단 실행(평가만). env 는 f2_preeval.sh / _run_f.sh eval 과 같음. 인자: <런 이름> <GPU> [결정 수]
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
HERE="$(cd "$(dirname "$0")" && pwd)"
R=/c/work/DT_Vessel_v3
cd $R/Python || exit 1
PRE=f2_
PY=$HOME/anaconda3/envs/mltest/python.exe
SDROP=$R/Python/runs_fig1/2026-10-01_scripted
export PYTHONIOENCODING=utf-8 VESSEL_PY=$PY PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0
export VESSEL_CKPT_DIR=$HOME/VESSEL_checkpoints/comm_intent
export VESSEL_BRANCH_WARMUP=2400
unset VESSEL_REQUIRE_TRUNK VESSEL_REUSE_ARMS VESSEL_SKIP_GOLDEN VESSEL_SEEDS VESSEL_NGPU
unset VESSEL_USE_COMM VESSEL_COMM_FIELDS VESSEL_COMM_LATENT VESSEL_PARTNER_RANGE VESSEL_AUX_LOSS_SCALE
unset VESSEL_COMM_CODEC VESSEL_COMM_CODEC_SHA VESSEL_COMM_CODEC_MODE VESSEL_BC_TEACHER VESSEL_BC_COEF VESSEL_BC_DECAY_DEC
export VESSEL_RUN_PREFIX=$PRE VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035 VESSEL_RISK_DCPA_GATE_M=48
export VESSEL_ACTION_MODE=course
export VESSEL_OUT_DIR="$HERE/_out"      # 배치 출력 폴더는 안 건드림
mkdir -p "$VESSEL_OUT_DIR"
CK=$VESSEL_CKPT_DIR
O=$R/Python/_repro_out_f2
source "$R/Python/common_env.sh"
r=$1; g=$2; nd=${3:-3000}; pol=${4:-learned}
a=${r%_s*}; s=${r#*_s}
if [ "$a" = comm ]; then
  cf=$(sed -n "s/^CODEC_s$s=\([^ ]*\) .*/\1/p" "$O/_codec_choice.txt"); sh=$(sed -n "s/^CODEC_s$s=.* SHA=\([0-9a-f]*\)$/\1/p" "$O/_codec_choice.txt")
  [ -n "$cf" ] && [ -n "$sh" ] || { echo "코덱 정보 없음"; exit 1; }
  envset="export VESSEL_USE_COMM=1 VESSEL_MSG_DIM=6 VESSEL_COMM_LATENT=0.0 VESSEL_AUX_LOSS_SCALE=0.0 VESSEL_COMM_FIELDS=intent VESSEL_PARTNER_RANGE=300 VESSEL_COMM_CODEC='$cf' VESSEL_COMM_CODEC_SHA=$sh VESSEL_COMM_CODEC_MODE=decode"
else
  envset="export VESSEL_USE_COMM=1 VESSEL_MSG_DIM=6 VESSEL_COMM_LATENT=0.0 VESSEL_AUX_LOSS_SCALE=0.0 VESSEL_COMM_FIELDS=state VESSEL_PARTNER_RANGE=56"
fi
arm=ON; [ "$pol" = learned ] || arm=OFF
common_env; export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2
bash -c "$envset
  '$PY' -u '$HERE/headon_diag.py' '$SDROP' -- --py_root '$R/Python' --policy $pol -- \
    --ckpt '$CK/${PRE}$r.pt' --arm $arm --envs 256 --eval_decisions $nd --burnin 2400 --seed 999" > "$HERE/diag_${pol}_$r.txt" 2>&1
echo "rc=$? $pol $r" >> "$HERE/_done.txt"
