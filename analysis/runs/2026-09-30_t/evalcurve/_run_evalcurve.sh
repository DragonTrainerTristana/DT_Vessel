#!/bin/bash
# 고정 평가 학습곡선(저자 선택 2026-09-30 15:55: 목표 2 의 reward 그림 = 체크포인트를 같은 조건으로 평가한 에피소드 점수 곡선)
#   점: 0(학습 전 = 같은 시드·같은 코드로 초기화만 한 정책) · 뿌리 2·4·6·8M · 뿌리 끝 9.04M · 갈래 10·12·14·16M · 갈래 끝 16.06M
#   평가: eval_ckpt.py --envs 256 --eval_decisions 2000 --burnin 2400 --seed 999 (모든 점 같은 평가 환경, 첫 에피소드 제외 = eval 규칙)
#   학습에는 손대지 않음(체크포인트 읽기만). 이미 평가한 점은 건너뜀 → 여러 번 돌려도 됨. --loop 면 10분마다 새 체크포인트를 찾아 평가,
#   갈래 끝(16.06M) 체크포인트가 전부 평가되면 끝.
# 실행: bash /c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t/evalcurve/_run_evalcurve.sh [--loop]
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1 PYTHONIOENCODING=utf-8
R=/c/work/DT_Vessel_v3/Python
PY=/c/Users/OSH/anaconda3/envs/mltest/python.exe
DROP=/c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t/evalcurve
O=$R/_repro_out_t/evalcurve
CK=$HOME/VESSEL_checkpoints/comm_intent
PRE=t_
ARMS="off a6 offb a8 a2 a4"
JOBS=${VESSEL_EC_JOBS:-3}
mkdir -p "$O" "$DROP/out"
SEEDS=$(cat "$R/_repro_out_t/_seeds.txt" 2>/dev/null) || { echo "★_seeds.txt 없음(관문 전)"; exit 1; }
cd "$R" || exit 1
source "$R/common_env.sh"
batch_env() {   # t_ 배치와 같은 env (_run_t.sh 와 1:1). eval_ckpt 가 스냅샷과 대조함
  common_env
  export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0 VESSEL_CKPT_DIR=$CK
  export VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
  export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035 VESSEL_RISK_DCPA_GATE_M=48
  export VESSEL_MSG_DIM=6 OMP_NUM_THREADS=1
}
pick_gpu() {    # 남은 메모리 가장 큰 GPU, 단 평가 1.2 GB + 여유 2.5 GB = 3.7 GB 이상 남아야(없으면 빈 문자열)
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -nr | awk -F, '$2>=3700{print $1; exit}'
}
settled() {     # ★2026-09-30 16:05 GPU 경합 방지: 새 학습 런이 막 뜬 뒤 10분(워밍업·첫 통신 rollout 까지 메모리 최대치 전)엔 평가를 안 띄움
  local f now; now=$(date +%s)
  for f in "$R"/_repro_out_t/t_*_s4?.log; do
    [ -f "$f" ] || continue
    grep -q "saved ->" "$f" && continue
    [ $(( now - $(stat -c %W "$f" 2>/dev/null || date -r "$f" +%s) )) -lt 600 ] && return 1
  done
  return 0
}
running_eval() {  # 같은 체크포인트를 평가 중인 eval_ckpt 가 있나(루프 재시작 때 중복 방지)
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine" 2>/dev/null | grep -F "eval_ckpt" | grep -qF "$(basename "$1")"
}
# 0 점: 학습 전 정책 — trunk 와 같은 인자·같은 시드로 update 0 번(--steps 0) → 초기화만 한 가중치 + 스냅샷 저장
make_init() {
  local s=$1 f="$CK/${PRE}init_d6_s$1.pt"
  [ -f "$f" ] && return 0
  ( batch_env; export VESSEL_USE_COMM=0 CUDA_VISIBLE_DEVICES=$(pick_gpu)
    "$PY" -u vessel_gym_train.py --arm OFF --steps 0 --comm_on_at 0 --envs 128 --vessels 16 --rollout 32 --ring 1.0 \
      --crossing 0 --max_partners 4 --seed "$s" --save "$f" > "$O/_init_s$s.log" 2>&1 )
  [ -f "$f" ] && echo "init s$s → $f" || echo "★init s$s 실패 — $O/_init_s$s.log"
}
N=0
ev() {          # ev <체크포인트 파일> <OFF|ON>
  local f=$1 arm=$2 b; b=$(basename "$f" .pt)
  local out="$O/ec_$b.txt"
  grep -q "eps ||" "$out" 2>/dev/null && return 0          # 이미 끝남(요약 줄 있음)
  running_eval "$f" && return 0                             # 다른 루프가 평가 중
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  until settled; do sleep 30; done
  local g; g=$(pick_gpu)
  while [ -z "$g" ]; do sleep 60; g=$(pick_gpu); done
  ( batch_env; export CUDA_VISIBLE_DEVICES=$g
    "$PY" -u eval/eval_ckpt.py --ckpt "$f" --arm "$arm" --envs 256 --eval_decisions 2000 --burnin 2400 --seed 999 > "$out" 2>&1
    cp "$out" "$DROP/out/" 2>/dev/null ) &
  N=$((N+1)); echo "  평가 $b ($arm) → GPU $g"
  sleep 20
}
one_pass() {
  N=0
  for s in $SEEDS; do make_init "$s"; done
  for s in $SEEDS; do
    ev "$CK/${PRE}init_d6_s$s.pt" OFF
    for m in 2 4 6 8; do [ -f "$CK/${PRE}trunk_d6_s$s.step${m}M.pt" ] && ev "$CK/${PRE}trunk_d6_s$s.step${m}M.pt" OFF; done
    [ -f "$CK/${PRE}trunk_d6_s$s.pt" ] && ev "$CK/${PRE}trunk_d6_s$s.pt" OFF
  done
  # ★2026-09-30 18:05 순서만 바꿈(점·평가 조건 그대로): 팔별 → 점별. 주 비교 팔(off·a6·offb)의 끝 점부터 → 통신 대 OFF 비교가 먼저 보이게
  local grp m a f arm
  for grp in "off a6 offb" "a8 a2 a4"; do
    for m in end 12 16 10 14; do
      for s in $SEEDS; do
        for a in $grp; do
          case $a in off|offb) arm=OFF ;; *) arm=ON ;; esac
          if [ $m = end ]; then f="$CK/${PRE}${a}_s$s.pt"; else f="$CK/${PRE}${a}_s$s.step${m}M.pt"; fi
          [ -f "$f" ] && ev "$f" $arm
        done
      done
    done
  done
  wait
  echo "$(date '+%F %T') 평가 $N건 새로 함"
}
if [ "${1:-}" = "--loop" ]; then
  end=$(( $(date +%s) + 20*3600 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    one_pass
    done_all=1
    for a in off a6; do for s in $SEEDS; do grep -q "eps ||" "$O/ec_${PRE}${a}_s$s.txt" 2>/dev/null || done_all=0; done; done
    [ $done_all = 1 ] && [ -n "$(ls "$CK/${PRE}"a4_s*.pt 2>/dev/null)" ] && { one_pass; break; }
    sleep 600
  done
else
  one_pass
fi
