#!/bin/bash
# ab_gpu_bitcheck.sh — 옛 코드 vs 새 코드 GPU 비트동일 A/B (2026-09-26)
#   결정론 커널(verify/_det_train.py) 아래 같은 명령을 옛 코드로 2회(대조군: old==old 아니면 판정 불가), 새 코드로 1회 돌려
#   state_dict·Adam·value_norm·steps·곡선 CSV 를 비트 비교한다. CPU 골든(test_golden)이 못 보는 GPU 경로·EXT·분기 경로 보강.
#   쓰는 법: OLD=/c/work/DT_Vessel/Python NEW=/c/work/DT_Vessel_fast/Python GPU=1 bash verify/ab_gpu_bitcheck.sh [case]
#     case = off (기본, 2 update from scratch) | on (실제 trunk 에서 oni6 팔로 1 update 분기; VESSEL_CKPT_DIR 필요)
set -u
OLD="${OLD:-/c/work/DT_Vessel/Python}"
NEW="${NEW:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${VESSEL_PY:-/c/Users/OSH/anaconda3/envs/mltest/python.exe}"
CASE="${1:-off}"
OUTROOT="${AB_OUT:-$NEW/_ab_out}/$CASE"
mkdir -p "$OUTROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-1}" OMP_NUM_THREADS=1 PYTHONIOENCODING=utf-8
source "$NEW/common_env.sh"; common_env
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0 VESSEL_COMM_TELEMETRY=0 VESSEL_MSG_DIM=6
if [ "$CASE" = off ]; then
  ARGS="--arm OFF --envs 128 --vessels 16 --rollout 32 --ring 1.0 --crossing 0 --max_partners 4 --seed 43 --steps 131072 --ckpt_every 0 --comm_on_at 0"
else
  CK="${VESSEL_CKPT_DIR:?VESSEL_CKPT_DIR 필요}"
  export VESSEL_COMM_FIELDS=intent VESSEL_COMM_LATENT=1.0 VESSEL_AUX_LOSS_SCALE=0.0; unset VESSEL_PARTNER_RANGE
  ARGS="--arm ON --envs 128 --vessels 16 --rollout 32 --ring 1.0 --crossing 0 --max_partners 4 --seed 43 --ckpt_every 0 \
        --resume $CK/x_trunk_d6_s43.pt --resume_at 9043968 --comm_on_at 9043968 --resume_warmup 8 --steps 9109504"
fi
run() {  # root outdir
  mkdir -p "$2"; rm -f "$2/m.pt" "$2/m_curve.csv"
  "$PY" -u "$NEW/verify/_det_train.py" "$1" $ARGS --save "$2/m.pt" --csv "$2/m_curve.csv" > "$2/log.txt" 2>&1
  echo "  $(basename "$2") rc=$? $(grep -c . "$2/m_curve.csv" 2>/dev/null) csv rows"
}
echo "[ab $CASE] GPU=$CUDA_VISIBLE_DEVICES old=$OLD new=$NEW"
run "$OLD" "$OUTROOT/old_a"
run "$OLD" "$OUTROOT/old_b"
run "$NEW" "$OUTROOT/new_a"
"$PY" "$NEW/verify/_ab_compare.py" "$OUTROOT/old_a" "$OUTROOT/old_b" "$OUTROOT/new_a"
rc=$?
diff <(grep -v -e 'dec/s' -e 'min)' -e 'roll=' "$OUTROOT/old_a/log.txt") <(grep -v -e 'dec/s' -e 'min)' -e 'roll=' "$OUTROOT/new_a/log.txt") > "$OUTROOT/log_diff.txt" && echo "  stdout(타이밍 제외) 동일" || echo "  stdout 차이 → $OUTROOT/log_diff.txt"
exit $rc
