#!/bin/bash
# lrm_eval.sh <prefix> [off comm | --table] — ★2026-11-02 끝난 1단계 갈래(phase1, 예 qd_·qd2_)에 평균 행동 주 평가(lrm)만 더함. 학습 없음
#   10-30 전 phase1 은 표본 행동(lr)만 돌렸음 → 저자 성공 지표(스펙 2026-11-02-author-goals-criteria.md) 주 추정량 lrm 이 비어 있음
#   평가 명령 = _run_f.sh eval_all 의 lrm 과 같음(eval_scripted.py --policy learned --mean_action, envs 256 · 10000 결정 · burnin 2400 · seed 999), 끝 모델만
#   env = meaneval_ckpts.sh 와 같음(그 배치 env + comm = p46 코덱) — restore_policy 가 스냅샷과 대조해 틀리면 멈춤
# 실행(Git Bash, 클론 루트, 팔마다 GPU 하나):
#   CUDA_VISIBLE_DEVICES=0 VESSEL_F_REWARD_RANGE=56 VESSEL_F_SEED=43 bash Python/runs_fig1/2026-10-02_fig1/lrm_eval.sh qd_ off
#   CUDA_VISIBLE_DEVICES=1 VESSEL_F_REWARD_RANGE=56 VESSEL_F_SEED=43 bash Python/runs_fig1/2026-10-02_fig1/lrm_eval.sh qd_ comm
#   표만: VESSEL_F_SEED=43 bash .../lrm_eval.sh qd_ --table
# 결과 = Python/_repro_out_<prefix>/lrm_<prefix><arm>_s<seed>.txt · _fig1_mean.md
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
PRE=${1:?사용: lrm_eval.sh <prefix> [off comm|--table]}; shift
RUNS=${*:-off comm}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$(git -C "$HERE" rev-parse --show-toplevel) || exit 1
cd "$R/Python" || exit 1
PY=${VESSEL_PY:-$HOME/anaconda3/envs/mltest/python.exe}
[ -x "$PY" ] || PY=$(command -v python3)
export PYTHONIOENCODING=utf-8
CK=${VESSEL_CKPT_DIR:-$HOME/VESSEL_checkpoints/comm_intent}
SEED=${VESSEL_F_SEED:-43}
O="$PWD/_repro_out_${PRE%_}"
SDROP="$R/Python/runs_fig1/2026-10-01_scripted"
# ── env = meaneval_ckpts.sh 와 같음 ──
source "$R/Python/common_env.sh"; common_env
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0 VESSEL_CKPT_DIR=$CK
export VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035 VESSEL_RISK_DCPA_GATE_M=48 VESSEL_ACTION_MODE=course
export VESSEL_ROLE_V2_PRIMARY=cum VESSEL_ROLE_V2_RES_F6=1
unset VESSEL_BC_TEACHER VESSEL_BC_COEF VESSEL_BC_DECAY_DEC VESSEL_REWARD_RANGE
unset VESSEL_LOGSTD_INIT0 VESSEL_LR_FINAL VESSEL_LR_DECAY_DEC VESSEL_ENTROPY_BONUS VESSEL_LOGSTD_MIN VESSEL_NOISE_HOLD
[ -z "${VESSEL_F_REWARD_RANGE:-}" ] || export VESSEL_REWARD_RANGE=$VESSEL_F_REWARD_RANGE
if [ "$RUNS" != "--table" ]; then
  echo "$(date '+%F %T') lrm $PRE [$RUNS] s$SEED $(git -C "$R" log -1 --oneline) GPU=${CUDA_VISIBLE_DEVICES:-auto}" | tee -a "$O/_lrm_log.txt"
  for a in $RUNS; do
    unset VESSEL_COMM_CODEC VESSEL_COMM_CODEC_SHA VESSEL_COMM_CODEC_MODE
    export VESSEL_USE_COMM=1 VESSEL_MSG_DIM=6 VESSEL_COMM_LATENT=0.0 VESSEL_AUX_LOSS_SCALE=0.0
    if [ "$a" = comm ]; then
      cf=$(sed -n "s/^CODEC_s$SEED=\([^ ]*\) .*/\1/p" "$O/_codec_choice.txt"); sh=$(sed -n "s/^CODEC_s$SEED=.* SHA=\([0-9a-f]*\)$/\1/p" "$O/_codec_choice.txt")
      [ -n "$cf" ] && [ -n "$sh" ] || { echo "★멈춤: 코덱 경로·SHA 없음($O/_codec_choice.txt s$SEED)"; exit 1; }
      export VESSEL_COMM_FIELDS=intent VESSEL_PARTNER_RANGE=300 VESSEL_COMM_CODEC="$cf" VESSEL_COMM_CODEC_SHA=$sh VESSEL_COMM_CODEC_MODE=decode
    else
      export VESSEL_COMM_FIELDS=state VESSEL_PARTNER_RANGE=56
    fi
    out="$O/lrm_${PRE}${a}_s$SEED.txt"
    grep -q "R8=" "$out" 2>/dev/null && { echo "건너뜀 $out (이미 있음)"; continue; }
    f="$CK/${PRE}${a}_s$SEED.pt"; [ -f "$f" ] || { echo "★멈춤: 체크포인트 없음 $f"; exit 1; }
    "$PY" -u "$SDROP/eval_scripted.py" --py_root "$R/Python" --policy learned --mean_action -- \
      --ckpt "$f" --arm ON --envs 256 --eval_decisions 10000 --burnin 2400 --seed 999 > "$out" 2>&1
    grep -q "R8=" "$out" || { echo "★FAIL lrm 평가 실패 $out"; exit 1; }
    echo "$(date '+%F %T') $a 끝 → $out" | tee -a "$O/_lrm_log.txt"
  done
fi
"$PY" "$HERE/summarize_fig1.py" "$O" "$PRE" lrm | tee "$O/_fig1_mean.md"
