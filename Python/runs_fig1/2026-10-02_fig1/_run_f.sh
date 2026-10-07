#!/bin/bash
# 2026-10-02 배치 f_ — Fig1 본 배치. 스펙 docs/superpowers/specs/2026-10-02-fig1-latent-design.md (feat/fig1-latent)
#   저자 승인 "제일 성공확률이 높은 방안" + "ㅇㅇ 진행해봐": 공통 뿌리 = ARPA@56·메시지 없음(목표 침로 행동) → 선생님 흉내(DAgger)
#   → trunk 9.04M(흉내 보조손실) → 시드마다 p46 코덱(레이더 임베딩 30 + 핵심 16 → k) → 갈래 off · comm · offb → 주 평가 + Woerner
#
# OFF 선생님 = vo56h150. 스펙 §5 규칙(주 지표에서 상대를 3/3 이긴 지표 수, 같으면 vo56)을 기존 측정(runs/2026-10-01_scripted, 평가 시드 3개)에 적용:
#   vo56h150 3/3 승 = 배끼리 충돌(8.3·8.9·8.6 vs 9.5·9.0·9.3)·도착(91.0·90.6·90.6 vs 89.8·90.2·90.1)·DCPA(23.1·23.2·23.3 vs 23.0·23.1·22.9)
#   vo56 3/3 승 = 연료(도착 ep). 함대 연료/도착 = 2/3 대 1/3(둘 다 3/3 아님). Woerner 는 vo56h150 미측정이지만 어느 쪽이 이겨도 3 > 2 → vo56h150.
#   vo56h150 Woerner 재측정은 기록용 — Windows 작업 vessel_sc_vo56h150(10-02 16:36)이 따로 돌림(선택에 영향 없음).
#
# ★2026-10-03 GitHub 만으로 도는 묶음: 이 파일·표 스크립트·평가 래퍼·참고 결과가 전부 저장소 Python/runs_fig1/ 안에 있음(Dropbox 불필요).
#   Dropbox runs/ 의 같은 이름 파일과 내용 같음(이 실행의 정본 = 저장소 사본). 기계마다 다른 값은 자동 감지하거나 env 로 줌:
#     저장소 = 이 파일 위치에서 git 루트 · python = VESSEL_PY(없으면 $HOME/anaconda3/envs/mltest/python.exe)
#     체크포인트 = VESSEL_CKPT_DIR(없으면 $HOME/VESSEL_checkpoints/comm_intent) · 결과 사본 = $HOME/Dropbox/.../runs/2026-10-02_fig1/out 이 있으면 거기, 없으면 _repro_out_f/results
# 준비(한 번, 아무 Windows 기계): git clone https://github.com/DragonTrainerTristana/DT_Vessel.git && cd DT_Vessel && git checkout feat/fig1-latent
#   (이미 클론이 있으면 git fetch origin && git checkout feat/fig1-latent && git pull). 얕은 클론 금지(preflight p6 차분이 옛 커밋 ae58b93 을 씀)
# 실행(Git Bash, 클론 루트에서):
#   VESSEL_F_AUTO1=1 bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh phase0   # P0 통과 시 phase1 까지 이어서
#   bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh phase1                     # phase1 만 다시
#   ★2026-10-06 P0 불통과여도 저자 결정으로 phase1 진행: VESSEL_F_P0_OVERRIDE="<저자 결정 문구>" 를 앞에 붙임
#     → _p0_override.txt 에 문구·시각·P0 판정 줄을 남기고 _fig1.md 맨 위에 그대로 적음(_p0.md 는 안 건드림). trunk 관문 이후는 그대로
# ★2026-10-07 배치 n_ (스펙 docs/superpowers/specs/2026-10-07-pure-rl-fig1-design.md, 저자 승인): 흉내 없는 순수 PPO
#   VESSEL_F_IMIT=0 VESSEL_F_PREFIX=n_ bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh phase1      # 시드 1개(43)
#   성공 뒤 확증(시드 3개): VESSEL_F_IMIT=0 VESSEL_F_PREFIX=n3_ VESSEL_F_SEEDS="43 44 45 46 47" VESSEL_F_NGATE=3 bash ... phase1
#   → preflight(smoke) · 순위 관문 v4 · trunk 처음부터(DAgger·흉내 보조손실 없음) · 관문 · 코덱 · 갈래(흉내 없음) · 평가 · 곡선(M) · 표.
#   보상 = v3 + VESSEL_ROLE_V2_PRIMARY=cum(누적 주 상대) + VESSEL_ROLE_V2_RES_F6=1(해소 종료에도 F6). phase0 없음
# 규칙: 단계 실패면 멈춤. 끝난 단계는 다시 돌릴 때 건너뜀(_timeline.txt). 배치 도중 HEAD 가 바뀌면 멈춤(→ VESSEL_F_PREFIX=f2_).
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
MODE=${1:-}
case "$MODE" in phase0|phase1) ;; *) echo "사용: $0 phase0|phase1"; exit 1 ;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)           # .../Python/runs_fig1/2026-10-02_fig1
BUNDLE=$(dirname "$HERE")                                     # .../Python/runs_fig1
R=${VESSEL_F_REPO:-$(git -C "$HERE" rev-parse --show-toplevel)} || { echo "★멈춤: git 저장소 안에서 실행할 것"; exit 1; }
IDROP=$BUNDLE/2026-10-02_imitation                            # summarize_p0.py
SDROP=$BUNDLE/2026-10-01_scripted                             # eval_scripted.py (+ colregs_*.py)
PRE=${VESSEL_F_PREFIX:-f_}
IMIT=${VESSEL_F_IMIT:-1}                # ★2026-10-07 0 = 흉내 없음(n_ 배치)
case "$IMIT" in 0|1) ;; *) echo "★멈춤: VESSEL_F_IMIT=$IMIT — 0 | 1"; exit 1 ;; esac
if [ "$IMIT" = 0 ]; then
  case "$PRE" in f_|f2_) echo "★멈춤: VESSEL_F_IMIT=0 은 새 접두어로(VESSEL_F_PREFIX=n_) — f_·f2_ 결과와 섞지 않음"; exit 1 ;; esac
  [ "$MODE" = phase1 ] || { echo "★멈춤: VESSEL_F_IMIT=0 은 phase1 만(phase0 = DAgger·P0)"; exit 1; }
fi
OFF_T=vo56h150                          # 위 주석(스펙 §5 규칙 적용 결과)
COMM_T=vo300i
SEEDS_ALL="43 44 45 46 47"
NGATE=3                                 # trunk 관문 통과 시드 수(갈래 시드 수)
# ★2026-10-07 저자 지시: n_ 는 시드 1개(43)로 먼저. 시뮬이 성공하면 같은 코드로 시드 3개(확증) = VESSEL_F_SEEDS="43 44 45 46 47" VESSEL_F_NGATE=3
if [ "$IMIT" = 0 ]; then SEEDS_ALL="43"; NGATE=1; fi
ARMS="off comm offb"
[ "$IMIT" = 0 ] && ARMS="off comm"      # ★2026-10-07 저자 결정: n_ 은 잡음 자 offb 없음 — 갈래 off · comm 둘만
SEEDS_ALL=${VESSEL_F_SEEDS:-$SEEDS_ALL}; NGATE=${VESSEL_F_NGATE:-$NGATE}
[ "$NGATE" -ge 1 ] && [ "$NGATE" -le $(echo $SEEDS_ALL | wc -w) ] || { echo "★멈춤: VESSEL_F_NGATE=$NGATE · 시드 [$SEEDS_ALL]"; exit 1; }
K=8                                     # 스펙 §4-1 파트너 수
BRANCH_AT=9043968
TOTAL=16056320
BC_DECAY=3000000                        # 스펙 §4-1 흉내 보조손실 1.0 → 0, 3M 결정
JOBS=${VESSEL_F_JOBS:-8}
GPU_CAP=${VESSEL_F_GPU_CAP:-2}          # GPU 한 장에 이 배치 런 2개까지(09-29 '한 장에 ON 3개 = 정지' 기록)
_busy=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine" 2>/dev/null | grep -c "vessel_gym_train\|dagger_init\|eval_ckpt\|comm_codec")   # 규칙 배 평가(eval_scripted)는 같이 돌아도 됨
[ "${_busy:-0}" -eq 0 ] || { echo "★멈춤: 학습·평가 python 프로세스 ${_busy}개가 돌고 있음 — 끝난 뒤 실행"; exit 1; }
cd "$R" || exit 1
[ -z "$(git status --short --untracked-files=no)" ] || { echo "★멈춤: 추적 파일 변경 있음 — git status 확인"; exit 1; }
[ "$(git rev-parse --is-shallow-repository)" = false ] || { echo "★멈춤: 얕은 클론 — git fetch --unshallow (preflight p6 차분에 ae58b93 필요)"; exit 1; }
git fetch -q origin feat/fig1-latent || { echo "★멈춤: git fetch 실패"; exit 1; }
git merge-base --is-ancestor HEAD FETCH_HEAD || { echo "★멈춤: HEAD $(git rev-parse --short HEAD) 가 origin/feat/fig1-latent 에 없음 — push 안 된 코드로 안 돌림"; exit 1; }
[ -z "${VESSEL_F_HEAD:-}" ] || [ "$(git rev-parse --short HEAD)" = "$(git rev-parse --short "$VESSEL_F_HEAD" 2>/dev/null)" ] || {
  echo "★멈춤: HEAD $(git rev-parse --short HEAD) ≠ VESSEL_F_HEAD=$VESSEL_F_HEAD"; exit 1; }
cd "$R/Python"
PY=${VESSEL_PY:-$HOME/anaconda3/envs/mltest/python.exe}
[ -x "$PY" ] || { echo "★멈춤: python 없음($PY) — VESSEL_PY=<python.exe 경로> 로 줄 것"; exit 1; }
"$PY" -c "import torch; print('[env] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())" || { echo "★멈춤: torch import 실패"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "★멈춤: nvidia-smi 없음"; exit 1; }
export PYTHONIOENCODING=utf-8 VESSEL_PY=$PY
# ── 배치 env = t_ 와 같음 + 목표 침로 행동. 통신 구조 값(USE_COMM·FIELDS·LATENT…)은 런마다 넣음(smoke·preflight 에 새지 않게) ──
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0
export VESSEL_CKPT_DIR=${VESSEL_CKPT_DIR:-$HOME/VESSEL_checkpoints/comm_intent}
export VESSEL_BRANCH_WARMUP=2400
unset VESSEL_REQUIRE_TRUNK VESSEL_REUSE_ARMS VESSEL_SKIP_GOLDEN VESSEL_SEEDS VESSEL_NGPU
unset VESSEL_USE_COMM VESSEL_COMM_FIELDS VESSEL_COMM_LATENT VESSEL_PARTNER_RANGE VESSEL_AUX_LOSS_SCALE
unset VESSEL_COMM_CODEC VESSEL_COMM_CODEC_SHA VESSEL_COMM_CODEC_MODE VESSEL_BC_TEACHER VESSEL_BC_COEF VESSEL_BC_DECAY_DEC
unset VESSEL_ROLE_V2_PRIMARY VESSEL_ROLE_V2_RES_F6
export VESSEL_RUN_PREFIX=$PRE VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035 VESSEL_RISK_DCPA_GATE_M=48
export VESSEL_ACTION_MODE=course
[ "$IMIT" = 0 ] && export VESSEL_ROLE_V2_PRIMARY=cum VESSEL_ROLE_V2_RES_F6=1   # ★2026-10-07 n_ 보상 = v3 + 판정기 토글 2
export VESSEL_OUT_DIR=$PWD/_repro_out_${PRE%_}
CK=$VESSEL_CKPT_DIR
O=$VESSEL_OUT_DIR; TL=$O/_timeline.txt; mkdir -p "$O" "$O/codec" "$CK"
DROPF=$HOME/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-02_fig1
if [ -d "$DROPF" ]; then RES=$DROPF/out; else RES=$O/results; fi   # 결과 사본(Dropbox 있으면 Mac 에서 바로 봄)
mkdir -p "$RES"
HEAD_NOW=$(git -C "$R" rev-parse HEAD)
if [ -f "$O/_commit.txt" ]; then
  _c0=$(head -1 "$O/_commit.txt")
  if [ "$_c0" != "$HEAD_NOW" ]; then
    # ★2026-10-06 실행 묶음(Python/runs_fig1)만 바뀐 커밋이면 계속(학습·평가 코드 동일). 그 밖의 변경이면 멈춤
    if git -C "$R" diff --quiet "$_c0" HEAD -- . ':(exclude)Python/runs_fig1'; then
      grep -q "$HEAD_NOW" "$O/_commit.txt" || echo "$HEAD_NOW runs_fig1-only $(date '+%F %T')" >> "$O/_commit.txt"
    else
      echo "★멈춤: 이 배치는 $_c0 로 시작 — 지금 HEAD $HEAD_NOW 는 학습·평가 코드가 다름 → 새 접두어(VESSEL_F_PREFIX)로"; exit 1
    fi
  fi
else
  echo "$HEAD_NOW" > "$O/_commit.txt"
fi
rm -f "$O"/.gpu*_*
source "$R/Python/common_env.sh"
TRAIN_GPU="VESSEL_JOBS=8 VESSEL_GPU_CAP=2 VESSEL_LAUNCH_GAP=20 VESSEL_VRAM_MARGIN=1500"
# 런 종류별 통신 구조 env (스펙 §2·§4-1). ARPA = 56 m 안 참값 상대 필드, 메시지 없음(latent 0·코덱 없음)
ENV_ARPA="export VESSEL_USE_COMM=1 VESSEL_MSG_DIM=6 VESSEL_COMM_LATENT=0.0 VESSEL_AUX_LOSS_SCALE=0.0 VESSEL_COMM_FIELDS=state VESSEL_PARTNER_RANGE=56"
env_comm() {   # env_comm <seed> : 300 m 안 메시지(p46 코덱 복원값, 56 m 안은 센서 참값) + 의도 필드
  local cf sh; cf=$(sed -n "s/^CODEC_s$1=\([^ ]*\) .*/\1/p" "$O/_codec_choice.txt"); sh=$(sed -n "s/^CODEC_s$1=.* SHA=\([0-9a-f]*\)$/\1/p" "$O/_codec_choice.txt")
  [ -n "$cf" ] && [ -n "$sh" ] || { echo "★FAIL 코덱 경로·SHA 없음 s$1" >&2; return 1; }
  echo "export VESSEL_USE_COMM=1 VESSEL_MSG_DIM=6 VESSEL_COMM_LATENT=0.0 VESSEL_AUX_LOSS_SCALE=0.0 VESSEL_COMM_FIELDS=intent VESSEL_PARTNER_RANGE=300 VESSEL_COMM_CODEC='$cf' VESSEL_COMM_CODEC_SHA=$sh VESSEL_COMM_CODEC_MODE=decode"
}
TARGS="--envs 128 --vessels 16 --rollout 32 --ring 1.0 --crossing 0 --max_partners $K"

log() { echo "$(date '+%F %T') $*" | tee -a "$TL"; }
sync_out() { cp "$O"/*.txt "$O"/*.log "$O"/*.csv "$O"/*.json "$O"/*.md "$RES/" 2>/dev/null; cp "$O"/codec/*.json "$RES/" 2>/dev/null; true; }
bad() {
  grep -q "★FAIL\|FAIL \|실패" "$1" 2>/dev/null && return 0
  for f in "$O"/_status_*.txt; do [ -f "$f" ] && grep -qv 'rc=0' "$f" && return 0; done
  return 1
}
step() {
  local nm=$1; shift
  if grep -q " 끝 $nm\$" "$TL" 2>/dev/null; then log "건너뜀 $nm (이미 끝남)"; return 0; fi
  log "시작 $nm"
  "$@" > "$O/_$nm.log" 2>&1; local rc=$?
  if [ $rc -ne 0 ] || bad "$O/_$nm.log"; then log "★멈춤 $nm rc=$rc — $O/_$nm.log 확인"; sync_out; exit 1; fi
  log "끝 $nm"; sync_out
}
# ── 실행기: 동시 JOBS 개, GPU 마다 이 배치 런 GPU_CAP 개까지(표식 파일 $O/.gpu<번호>_<로그>) ──
PIDS=()
running() { local n=0 p; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo $n; }
pick_gpu() {   # 남은 메모리 need + 1.5 GB 이상이고 표식 < GPU_CAP 인 GPU 중 여유 가장 큰 것
  local g
  for g in $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' \r' | sort -t, -k2 -nr \
             | awk -F, -v need="$1" '$2>=need+1500{print $1}'); do
    [ "$(ls "$O"/.gpu${g}_* 2>/dev/null | wc -l)" -lt "$GPU_CAP" ] && { echo "$g"; return; }
  done
}
launch() {     # launch <VRAM MB> <로그> <명령...> — 끝나면 '<로그> rc=N' 을 _status_f.txt 에
  local need=$1 lg=$2; shift 2
  while [ "$(running)" -ge "$JOBS" ]; do sleep 5; done
  local g; g=$(pick_gpu "$need"); while [ -z "$g" ]; do sleep 30; g=$(pick_gpu "$need"); done
  local mk="$O/.gpu${g}_$(basename "$lg")"; : > "$mk"
  ( common_env; export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2
    "$@" > "$lg" 2>&1; echo "$(basename "$lg") rc=$?" >> "$O/_status_f.txt"; rm -f "$mk" ) &
  PIDS+=($!)
  echo "  $(basename "$lg") → GPU $g"; sleep 20
}
wait_all() { local p; for p in "${PIDS[@]}"; do wait "$p" 2>/dev/null; done; PIDS=(); }
watch_stalls() {   # 학습 런 로그가 15분 무진행이면 그 런만 kill(_kill_log.txt). 런 이름 = --save 파일 이름. 이 배치 학습이 없으면 끝남
  sleep 180
  while true; do
    local pids; pids=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'vessel_gym_train' } | ForEach-Object { \$m=[regex]::Match(\$_.CommandLine,'--save\s+\S*?(${PRE}[a-z0-9_]+_s4\d)\.pt'); if (\$m.Success) { '{0} {1}' -f \$_.ProcessId, \$m.Groups[1].Value } }" 2>/dev/null | tr -d '\r')
    [ -n "$pids" ] || return 0
    echo "$pids" | while read -r pid nm; do
      [ -n "$nm" ] && [ -f "$O/$nm.log" ] || continue
      local age=$(( ($(date +%s) - $(date -r "$O/$nm.log" +%s)) / 60 ))
      if [ "$age" -ge 15 ]; then
        echo "$(date '+%F %T') $nm (PID $pid) kill — 로그 ${age}분 무진행" >> "$O/_kill_log.txt"
        MSYS2_ARG_CONV_EXCL='*' taskkill /F /T /PID "$pid" >/dev/null 2>&1
      fi
    done
    sleep 120
  done
}

# ── phase0: 초기 모델(--steps 0) → DAgger(ARPA 경로, OFF 선생님) → 학생 주 평가 → P0 표 ──
dagger_all() {
  : > "$O/_status_f.txt"; local s
  for s in $SEEDS_ALL; do
    [ -f "$CK/${PRE}dagger_d6_s$s.pt" ] && { echo "  재사용 ${PRE}dagger_d6_s$s.pt"; continue; }
    launch 2500 "$O/${PRE}dagger_d6_s$s.log" bash -c "$ENV_ARPA
      '$PY' -u vessel_gym_train.py --arm ON --steps 0 --comm_on_at 0 $TARGS --seed $s --save '$CK/${PRE}init_d6_s$s.pt' &&
      '$PY' -u imitation/dagger_init.py --init '$CK/${PRE}init_d6_s$s.pt' --seed $s --teacher $OFF_T --save '$CK/${PRE}dagger_d6_s$s.pt'"
  done
  wait_all
  for s in $SEEDS_ALL; do
    [ -f "$CK/${PRE}dagger_d6_s$s.pt" ] || { echo "★FAIL DAgger s$s — $O/${PRE}dagger_d6_s$s.log"; return 1; }
    cp "$CK/${PRE}dagger_d6_s$s.imitation.json" "$O/" 2>/dev/null
  done
}
evalp0_all() {
  : > "$O/_status_f.txt"; local s
  for s in $SEEDS_ALL; do
    grep -q "eps ||" "$O/eval_${PRE}dagger_d6_s$s.txt" 2>/dev/null && continue
    launch 2500 "$O/eval_${PRE}dagger_d6_s$s.txt" bash -c "$ENV_ARPA
      '$PY' -u eval/eval_ckpt.py --ckpt '$CK/${PRE}dagger_d6_s$s.pt' --arm ON --envs 256 --eval_decisions 10000 --burnin 2400 --seed 999"
  done
  wait_all
}
p0_table() {
  sync_out
  "$PY" "$IDROP/summarize_p0.py" "$RES" "$PRE" > "$O/_p0.md" 2>&1; local rc=$?
  cat "$O/_p0.md"; cp "$O/_p0.md" "$RES/"
  return $rc
}

# ── phase1 ──
trunk_all() {
  local s try
  for try in 1 2; do       # 멈춤 감시가 죽인 런은 처음부터 한 번 더
    : > "$O/_status_f.txt"
    watch_stalls & local wp=$!
    for s in $SEEDS_ALL; do
      [ -f "$CK/${PRE}trunk_d6_s$s.pt" ] && continue
      rm -f "$O/${PRE}trunk_d6_s$s.csv" "$O/${PRE}trunk_d6_s$s"_aux.csv "$O/${PRE}trunk_d6_s$s"_ep.csv
      if [ "$IMIT" = 0 ]; then   # ★2026-10-07 처음부터 순수 PPO(초기 모델 = 학습기 기본 초기화, 선생님 없음)
        launch 3000 "$O/${PRE}trunk_d6_s$s.log" bash -c "$ENV_ARPA
          '$PY' -u vessel_gym_train.py --arm ON --steps $BRANCH_AT --comm_on_at 0 \
            $TARGS --seed $s --ckpt_every 2 --save '$CK/${PRE}trunk_d6_s$s.pt' --csv '$O/${PRE}trunk_d6_s$s.csv'"
      else
      launch 3000 "$O/${PRE}trunk_d6_s$s.log" bash -c "$ENV_ARPA
        export VESSEL_BC_TEACHER=$OFF_T VESSEL_BC_COEF=1.0 VESSEL_BC_DECAY_DEC=$BC_DECAY
        '$PY' -u vessel_gym_train.py --arm ON --resume '$CK/${PRE}dagger_d6_s$s.pt' --resume_at 0 --steps $BRANCH_AT --comm_on_at 0 \
          $TARGS --seed $s --ckpt_every 2 --save '$CK/${PRE}trunk_d6_s$s.pt' --csv '$O/${PRE}trunk_d6_s$s.csv'"
      fi
    done
    wait_all; kill $wp 2>/dev/null
    local miss=0; for s in $SEEDS_ALL; do [ -f "$CK/${PRE}trunk_d6_s$s.pt" ] || miss=1; done
    [ $miss -eq 0 ] && { : > "$O/_status_f.txt"; return 0; }
    echo "trunk 시도 $try: 빠진 시드 있음 — $(tr '\n' ' ' < "$O/_status_f.txt")"
  done
  for s in $SEEDS_ALL; do [ -f "$CK/${PRE}trunk_d6_s$s.pt" ] || echo "★FAIL trunk s$s — $O/${PRE}trunk_d6_s$s.log"; done
  return 1
}
gate9() {   # t_ 와 같은 관문(결과 전 고정): 마지막 4창 평균 goal ≥ 30 · oColl ≤ 5. 시드 순서로 앞 NGATE 개(f_ 3 · n_ 1). G1 은 기록만
  local s L st pass="" g1
  for s in $SEEDS_ALL; do
    L="$O/${PRE}trunk_d6_s$s.log"
    st=$(grep '^\[ON\] dec=' "$L" 2>/dev/null | tail -4 | sed -E 's/.*goal=([0-9.]+)%.*oColl=([0-9.]+)%.*/\1 \2/' \
         | awk '{g+=$1; o+=$2; n++} END {if (n==4 && g/n>=30 && o/n<=5) v="통과"; else v="불통과"; printf "goal=%.1f oColl=%.1f 창=%d %s", (n?g/n:0), (n?o/n:0), n, v}')
    g1=$(grep '^\[ON\] dec=' "$L" 2>/dev/null | tail -1 | sed -E 's/.*goal=([0-9.]+)%.*vColl=([0-9.]+)%.*oColl=([0-9.]+)%.*/\1 \2 \3/' \
         | awk '{printf "G1(마지막 창 goal %.1f · vColl %.1f · oColl %.1f) %s", $1, $2, $3, ($1>=90 && $2<=5 && $3<=2) ? "충족" : "미충족"}')
    echo "s$s $st | $g1"
    case "$st" in *" 통과") [ $(echo $pass | wc -w) -lt $NGATE ] && pass="$pass $s" ;; esac
  done
  pass=$(echo $pass)
  if [ $(echo $pass | wc -w) -lt $NGATE ]; then echo "trunk 관문: 통과 시드 ${NGATE}개 미만($pass) — 멈춤(기준 안 바꿈, 그대로 보고)"; return 1; fi
  echo "$pass" > "$O/_seeds.txt"; echo "trunk 관문 통과 시드: $pass"
}
codec_all() {   # 스펙 §4-1: 시드마다 그 trunk 레이더 인코더 + vo300i 함대로 수집 → k=4·6·8·12 → 관문 → 공통 k = 모든 시드 통과하는 가장 작은 k(6→8→12)
  : > "$O/_status_f.txt"; local s
  for s in $SEEDS; do
    [ -f "$O/codec/rep_s$s.json" ] && continue
    rm -rf "$O/codec/s$s"
    launch 2500 "$O/${PRE}codec_s$s.log" bash -c "$ENV_ARPA
      '$PY' -u comm_codec.py collect46 --ckpt '$CK/${PRE}trunk_d6_s$s.pt' --out '$O/codec/data_s$s.pt' --envs 64 --burn 300 --T 1500 --seed $s --teacher $COMM_T &&
      '$PY' -u comm_codec.py sweep46 --data '$O/codec/data_s$s.pt' --out_dir '$O/codec/s$s' --seed 0 --report '$O/codec/rep_s$s.json'"
  done
  wait_all
  "$PY" - "$(cygpath -m "$O/codec")" $SEEDS 2>&1 <<'PYEOF' | tr -d '\r' > "$O/_codec_choice.txt"
import json, os, sys
d, seeds = sys.argv[1], sys.argv[2:]
reps = {s: json.load(open(os.path.join(d, f'rep_s{s}.json'), encoding='utf-8')) for s in seeds}
for s, r in reps.items():
    print(f"s{s}: " + ' '.join(f"k{k}={'통과' if r[k]['pass'] else '불통과'}" for k in ('4', '6', '8', '12')))
    for k in ('4', '6', '8', '12'):
        print(f"   k{k} {r[k]['fidelity_holdout']}")
ch = next((k for k in ('6', '8', '12') if all(reps[s][k]['pass'] for s in seeds)), None)
print(f"CHOSEN_K={ch if ch else 'NONE'}")
if ch:
    for s in seeds:
        print(f"CODEC_s{s}={reps[s][ch]['path'].replace(chr(92), '/')} SHA={reps[s][ch]['sha'][:16]}")
PYEOF
  cat "$O/_codec_choice.txt"
  grep -q "^CHOSEN_K=[0-9]" "$O/_codec_choice.txt" || { echo "코덱 관문: 공통 k 없음 — 멈춤(스펙 §4, 그대로 보고)"; return 1; }
}
branch_one() {   # branch_one <off|comm|offb> <seed>
  local arm=$1 s=$2 run="${PRE}$1_s$2" wu=2400 envset teach
  [ -f "$CK/$run.pt" ] && return 0
  [ "$arm" = offb ] && wu=2401             # offb = off 재분기(워밍업 +1 결정) — 잡음 자 N
  if [ "$arm" = comm ]; then envset=$(env_comm "$s") || return 1; teach=$COMM_T; else envset=$ENV_ARPA; teach=$OFF_T; fi
  rm -f "$O/$run.csv" "$O/${run}_aux.csv" "$O/${run}_ep.csv"
  cp "$O/${PRE}trunk_d6_s$s.csv" "$O/$run.csv"
  [ -f "$O/${PRE}trunk_d6_s${s}_aux.csv" ] && cp "$O/${PRE}trunk_d6_s${s}_aux.csv" "$O/${run}_aux.csv"
  [ -f "$O/${PRE}trunk_d6_s${s}_ep.csv" ] && cp "$O/${PRE}trunk_d6_s${s}_ep.csv" "$O/${run}_ep.csv"
  local bcset="export VESSEL_BC_TEACHER=$teach VESSEL_BC_COEF=1.0 VESSEL_BC_DECAY_DEC=$BC_DECAY"
  [ "$IMIT" = 0 ] && bcset="unset VESSEL_BC_TEACHER VESSEL_BC_COEF VESSEL_BC_DECAY_DEC"   # ★2026-10-07 갈래도 흉내 없음
  launch 3000 "$O/$run.log" bash -c "$envset
    $bcset
    '$PY' -u vessel_gym_train.py --arm ON --resume '$CK/${PRE}trunk_d6_s$s.pt' --resume_at $BRANCH_AT --comm_on_at $BRANCH_AT \
      --resume_warmup $wu --steps $TOTAL $TARGS --seed $s --ckpt_every 2 --save '$CK/$run.pt' --csv '$O/$run.csv'"
}
branches_all() {
  local s a try files=""
  for try in 1 2; do
    : > "$O/_status_f.txt"
    watch_stalls & local wp=$!
    for a in $ARMS; do for s in $SEEDS; do branch_one $a $s || { kill $wp 2>/dev/null; return 1; }; done; done   # offb 마지막(슬롯 모자라면 잡음 자가 늦게)
    wait_all; kill $wp 2>/dev/null
    local miss=0; for s in $SEEDS; do for a in $ARMS; do [ -f "$CK/${PRE}${a}_s$s.pt" ] || miss=1; done; done
    [ $miss -eq 0 ] && break
    echo "갈래 시도 $try: 빠진 런 있음 — $(tr '\n' ' ' < "$O/_status_f.txt")"
    [ $try -eq 2 ] && return 1
  done
  : > "$O/_status_f.txt"
  for s in $SEEDS; do for a in $ARMS; do files="$files $CK/${PRE}${a}_s$s.pt"; done; done
  "$PY" -u verify/check_branch.py --trunk_dir "$CK" --csv_dir "$O" $files > "$O/_branch_check.txt" 2>&1
  cat "$O/_branch_check.txt"
  grep -q "ALL PASS" "$O/_branch_check.txt" || { echo "★FAIL 분기 검사"; return 1; }
}
eval_all() {   # 주 평가(t_ 와 같은 조건) + Woerner·타 줄 = eval_scripted.py learned 모드(신경망 그대로, 지표 줄만 덧붙임)
  : > "$O/_status_f.txt"; local s a envset
  for s in $SEEDS; do for a in $ARMS; do
    grep -q "R8=" "$O/lr_${PRE}${a}_s$s.txt" 2>/dev/null && continue
    if [ "$a" = comm ]; then envset=$(env_comm "$s") || return 1; else envset=$ENV_ARPA; fi
    launch 3000 "$O/lr_${PRE}${a}_s$s.txt" bash -c "$envset
      '$PY' -u '$SDROP/eval_scripted.py' --py_root '$R/Python' --policy learned -- \
        --ckpt '$CK/${PRE}${a}_s$s.pt' --arm ON --envs 256 --eval_decisions 10000 --burnin 2400 --seed 999"
  done; done
  wait_all
}
rank_v4() {   # ★2026-10-07 관문 2(스펙 2026-10-07): 학습 전 보상 순위 관문, 설정 v4 하나로 판정. 불통과 = 멈추고 보고
  "$PY" -u verify/check_reward_rank.py --settings v4 --jobs 1 --out "$O/_rank_v4.json"; local rc=$?
  [ $rc -eq 0 ] || echo "★FAIL 순위 관문 v4 rc=$rc"
  return $rc
}
curve_all() {   # ★2026-10-07 고정 장면 체크포인트 곡선(스펙 2026-10-07 §곡선): step 체크포인트 + 끝 모델, 평가 시드 999
  : > "$O/_status_f.txt"; local s a f x run envset
  for s in $SEEDS; do for run in trunk_d6 $ARMS; do
    if [ "$run" = comm ]; then envset=$(env_comm "$s") || return 1; else envset=$ENV_ARPA; fi
    for f in "$CK/${PRE}${run}_s$s".step*M.pt "$CK/${PRE}${run}_s$s.pt"; do
      [ -f "$f" ] || continue
      case "$f" in *.step*M.pt) x=$(echo "$f" | sed -E 's/.*\.step([0-9.]+)M\.pt$/\1/') ;;
                   *) [ "$run" = trunk_d6 ] && x=$(awk "BEGIN{print $BRANCH_AT/1e6}") || x=$(awk "BEGIN{print $TOTAL/1e6}") ;; esac
      grep -q "epReward=" "$O/curve_${PRE}${run}_s${s}_${x}M.txt" 2>/dev/null && continue
      launch 2500 "$O/curve_${PRE}${run}_s${s}_${x}M.txt" bash -c "$envset
        '$PY' -u eval/eval_ckpt.py --ckpt '$f' --arm ON --envs 256 --eval_decisions 3000 --burnin 2400 --drain 3000 --seed 999"
    done
  done; done
  wait_all
  "$PY" "$HERE/curve_fig1.py" "$O" "$PRE" > "$O/_curve.md" 2>&1; local rc=$?
  cat "$O/_curve.md"; cp "$O/_curve.md" "$O/${PRE}curve.csv" "$RES/" 2>/dev/null; cp "$O/${PRE}curve.pdf" "$RES/" 2>/dev/null
  return $rc
}
fig1_table() {
  sync_out
  { [ -f "$O/_p0_override.txt" ] && { cat "$O/_p0_override.txt"; echo; }
    "$PY" "$HERE/summarize_fig1.py" "$RES" "$PRE"; } > "$O/_fig1.md" 2>&1; local rc=$?
  cat "$O/_fig1.md"; cp "$O/_fig1.md" "$RES/"
  return $rc
}
phase1() {
  if [ "$IMIT" = 0 ]; then   # ★2026-10-07 흉내 없음: P0 대신 preflight + 순위 관문 v4
    step smoke env VESSEL_FORCE_PREFLIGHT=1 VESSEL_SEEDS=43 VESSEL_SMOKE_ARMS="off a6" $TRAIN_GPU bash run_repro.sh smoke
    step rank  rank_v4
  elif ! grep -q "P0 판정: 통과" "$O/_p0.md" 2>/dev/null; then
    [ -f "$O/_p0.md" ] && [ -n "${VESSEL_F_P0_OVERRIDE:-}" ] || { log "★멈춤: P0 통과 기록 없음($O/_p0.md) — phase0 먼저(불통과를 저자 결정으로 넘기려면 VESSEL_F_P0_OVERRIDE)"; exit 1; }
    if [ ! -f "$O/_p0_override.txt" ]; then
      { echo "> ★ P0(흉내 관문, 스펙 §4-2) 불통과 — 저자 결정으로 phase1 진행 ($(date '+%F %T'))"
        echo "> 저자 결정: $VESSEL_F_P0_OVERRIDE"
        echo "> $(grep 'P0 판정' "$O/_p0.md")"; } > "$O/_p0_override.txt"
    fi
    log "★P0 불통과 — 저자 결정으로 진행: $VESSEL_F_P0_OVERRIDE"
  fi
  step trunk    trunk_all
  step gate     gate9
  SEEDS=$(cat "$O/_seeds.txt") || exit 1
  log "갈래 시드: $SEEDS"
  step codec    codec_all
  step branches branches_all
  step eval     eval_all
  [ "$IMIT" = 0 ] && step curve curve_all
  step fig1     fig1_table
  log "phase1 완료 — 결과 $RES/_fig1.md"
}

log "배치 $PRE $MODE $(hostname) $(git -C "$R" log -1 --oneline) 결과사본=$RES action_mode=$VESSEL_ACTION_MODE 흉내=$IMIT 시드=[$SEEDS_ALL]·관문$NGATE 갈래=[$ARMS] OFF선생님=$OFF_T 통신선생님=$COMM_T K=$K OUT=$O"
if [ "$MODE" = phase0 ]; then
  step smoke   env VESSEL_FORCE_PREFLIGHT=1 VESSEL_SEEDS=43 VESSEL_SMOKE_ARMS="off a6" $TRAIN_GPU bash run_repro.sh smoke
  step dagger  dagger_all
  step evalp0  evalp0_all
  step p0      p0_table
  if grep -q "P0 판정: 통과" "$O/_p0.md"; then
    log "P0 통과 → phase1 실행 가능"
    [ "${VESSEL_F_AUTO1:-0}" = 1 ] && phase1
  else
    log "P0 불통과 → 멈춤(스펙 §4: 재시도 없이 보고)"
  fi
else
  phase1
fi
sync_out
