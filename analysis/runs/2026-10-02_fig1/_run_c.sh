#!/bin/bash
# 2026-10-02 배치 c_ — 목표 침로 행동(VESSEL_ACTION_MODE=course) + 선생님 vo56 흉내 시작 (Fig1 준비 단계).
#   스펙 docs/superpowers/specs/2026-10-02-fig1-latent-design.md (feat/fig1-latent). 저자 2026-10-02 "지금 바로 학습"
#   i_ 배치(타각 행동·레이더만) P0 불통과 71.1/28.7 → 행동만 목표 침로로 바꿔 같은 P0 기준(시드마다 도착 ≥ 80 %·충돌 ≤ 15 %, 3/5)으로 다시 봄.
#   통과하면 ARPA 없이 trunk 진행 가능(phase1), 불통과면 ARPA 필요 근거 → 스펙 §2 대로.
#
# 준비(한 번): C:\work\DT_Vessel_v3 에서  git fetch origin && git checkout feat/fig1-latent   (HEAD == HEAD_REQ)
# 실행(Windows Git Bash, 한 줄):
#   bash /c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-02_fig1/_run_c.sh phase0
#   (P0 통과 뒤)  bash /c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-02_fig1/_run_c.sh phase1
#
# phase0 (학습 없음): smoke(preflight 강제) → 시드 43–47 init(--steps 0)·DAgger → 학생 주 평가 → P0 표(_p0.md)
# phase1 (P0 통과 시만): trunk 5개(DAgger 체크포인트에서 --resume_at 0, 9.04M) → 관문(t_ 규칙) → 망각 기록
#                        → 갈래 off·a6·offb → 주 평가 → 판정 표(스펙 2026-09-30 §7, 최선 OFF 풀 i_off·t_off·s_off·x_off)
# 규칙: 단계 실패면 멈춤. 끝난 단계는 다시 돌릴 때 건너뜀(_timeline.txt 의 '끝 <단계>'). HEAD 가 바뀌면 멈춤(→ VESSEL_I_PREFIX=i2_).
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
MODE=${1:-}
case "$MODE" in phase0|phase1) ;; *) echo "사용: $0 phase0|phase1"; exit 1 ;; esac
HEAD_REQ=${VESSEL_C_HEAD:-f3e50fd}      # feat/fig1-latent f3e50fd (2026-10-02). 코드가 바뀌면 새 접두어(c2_)+새 해시
R=/c/work/DT_Vessel_v3
DROP=/c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-02_fig1
IDROP=/c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-02_imitation
TDROP=/c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t
PRE=${VESSEL_C_PREFIX:-c_}
TEACHER=${VESSEL_C_TEACHER:-vo56}
SEEDS_ALL="43 44 45 46 47"
ARMS="off a6 offb"
[ "$HEAD_REQ" != TBD ] || { echo "★멈춤: HEAD_REQ 미설정"; exit 1; }
_busy=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine" 2>/dev/null | grep -c "vessel_gym_train\|dagger_init")
[ "${_busy:-0}" -eq 0 ] || { echo "★멈춤: 학습 python 프로세스 ${_busy}개가 돌고 있음 — 끝난 뒤 실행"; exit 1; }
cd $R || exit 1
[ -z "$(git status --short --untracked-files=no)" ] || { echo "★멈춤: 추적 파일 변경 있음 — git status 확인"; exit 1; }
[ "$(git rev-parse --short HEAD)" = "$(git rev-parse --short $HEAD_REQ 2>/dev/null)" ] || {
  echo "★멈춤: HEAD $(git rev-parse --short HEAD) ≠ $HEAD_REQ — git fetch origin && git checkout feat/fig1-latent"; exit 1; }
cd $R/Python
for f in p6_k6_s0; do [ -f comm_codecs/$f.pt ] || { echo "★멈춤: comm_codecs/$f.pt 없음(a6 코덱)"; exit 1; }; done
PY=/c/Users/OSH/anaconda3/envs/mltest/python.exe
export PYTHONIOENCODING=utf-8 VESSEL_PY=$PY
# ── env = t_ 배치와 1:1 (runs/2026-09-30_t/_run_t.sh:39-48) ──
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0
export VESSEL_COMM_TELEMETRY=1
export VESSEL_CKPT_DIR=$HOME/VESSEL_checkpoints/comm_intent
export VESSEL_BRANCH_WARMUP=2400
unset VESSEL_REQUIRE_TRUNK VESSEL_REUSE_ARMS VESSEL_SKIP_GOLDEN VESSEL_SEEDS VESSEL_NGPU
export VESSEL_RUN_PREFIX=$PRE VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035 VESSEL_RISK_DCPA_GATE_M=48
export VESSEL_ACTION_MODE=course     # ★2026-10-02 목표 침로 행동(스냅샷 action_mode 로 기록·대조)
export VESSEL_OUT_DIR=$PWD/_repro_out_${PRE%_}
CK=$VESSEL_CKPT_DIR
O=$VESSEL_OUT_DIR; TL=$O/_timeline.txt; mkdir -p "$O" "$DROP/out"
TO_DIR=$PWD/_repro_out_t      # t_ 평가 파일(최선 OFF 풀: eval_t_off·eval_s_off·eval_x_off)
HEAD_NOW=$(git -C $R rev-parse HEAD)
if [ -f "$O/_commit.txt" ]; then
  [ "$(cat "$O/_commit.txt")" = "$HEAD_NOW" ] || { echo "★멈춤: 이 배치는 $(cat "$O/_commit.txt") 로 시작했는데 지금 HEAD 는 $HEAD_NOW → VESSEL_I_PREFIX=i2_ 로 새로"; exit 1; }
else
  echo "$HEAD_NOW" > "$O/_commit.txt"
fi
source "$R/Python/common_env.sh"
TRAIN_GPU="VESSEL_JOBS=8 VESSEL_GPU_CAP=2 VESSEL_LAUNCH_GAP=90 VESSEL_VRAM_MARGIN=1500"
EVAL_GPU="VESSEL_JOBS=12 VESSEL_GPU_CAP=3 VESSEL_LAUNCH_GAP=10"
JOBS=${VESSEL_I_JOBS:-5}

log() { echo "$(date '+%F %T') $*" | tee -a "$TL"; }
sync_out() { cp "$O"/*.txt "$O"/*.log "$O"/*.csv "$O"/*.json "$O"/*.md "$DROP/out/" 2>/dev/null; true; }
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
pick_gpu() {    # 남은 메모리가 need + 1.5 GB 이상인 GPU 중 가장 여유 있는 것(없으면 빈 문자열)
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -nr \
    | awk -F, -v need="$1" '$2>=need+1500{print $1; exit}'
}
launch() {      # launch <VRAM MB> <로그> <명령...> : 동시 JOBS 개까지, GPU 고른 뒤 배경 실행. 끝나면 '<로그> rc=N' 을 _status_i.txt 에
  local need=$1 lg=$2; shift 2
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  local g; g=$(pick_gpu "$need"); while [ -z "$g" ]; do sleep 60; g=$(pick_gpu "$need"); done
  ( common_env; export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 VESSEL_MSG_DIM=6 VESSEL_USE_COMM=0
    "$@" > "$lg" 2>&1; echo "$(basename "$lg") rc=$?" >> "$O/_status_i.txt" ) &
  echo "  $(basename "$lg") → GPU $g"
  sleep 20
}
TRUNK_ARGS="--envs 128 --vessels 16 --rollout 32 --ring 1.0 --crossing 0 --max_partners 4"

# ── phase0 ──
dagger_all() {
  : > "$O/_status_i.txt"
  local s
  for s in $SEEDS_ALL; do
    [ -f "$CK/${PRE}dagger_d6_s$s.pt" ] && { echo "  재사용 ${PRE}dagger_d6_s$s.pt"; continue; }
    launch 2500 "$O/${PRE}dagger_d6_s$s.log" bash -c "
      '$PY' -u vessel_gym_train.py --arm OFF --steps 0 --comm_on_at 0 $TRUNK_ARGS --seed $s --save '$CK/${PRE}init_d6_s$s.pt' &&
      '$PY' -u imitation/dagger_init.py --init '$CK/${PRE}init_d6_s$s.pt' --seed $s --teacher $TEACHER --save '$CK/${PRE}dagger_d6_s$s.pt'"
  done
  wait
  for s in $SEEDS_ALL; do
    [ -f "$CK/${PRE}dagger_d6_s$s.pt" ] || { echo "★FAIL DAgger s$s — $O/${PRE}dagger_d6_s$s.log"; return 1; }
    cp "$CK/${PRE}dagger_d6_s$s.imitation.json" "$O/" 2>/dev/null
  done
}
evalp0_all() {
  : > "$O/_status_i.txt"
  local s
  for s in $SEEDS_ALL; do
    grep -q "eps ||" "$O/eval_${PRE}dagger_d6_s$s.txt" 2>/dev/null && continue
    launch 1500 "$O/eval_${PRE}dagger_d6_s$s.txt" \
      "$PY" -u eval/eval_ckpt.py --ckpt "$CK/${PRE}dagger_d6_s$s.pt" --arm OFF --envs 256 --eval_decisions 10000 --burnin 2400 --seed 999
  done
  wait
}
p0_table() {
  sync_out
  "$PY" "$IDROP/summarize_p0.py" "$DROP/out" "$PRE" > "$O/_p0.md" 2>&1; local rc=$?
  cat "$O/_p0.md"; cp "$O/_p0.md" "$DROP/out/"
  return $rc
}

# ── phase1 ──
trunk_all() {
  : > "$O/_status_i.txt"
  local s
  for s in $SEEDS_ALL; do
    [ -f "$CK/${PRE}trunk_d6_s$s.pt" ] && { echo "  재사용 ${PRE}trunk_d6_s$s.pt"; continue; }
    rm -f "$O/${PRE}trunk_d6_s$s.csv" "$O/${PRE}trunk_d6_s$s"_aux.csv "$O/${PRE}trunk_d6_s$s"_ep.csv
    launch 2000 "$O/${PRE}trunk_d6_s$s.log" \
      "$PY" -u vessel_gym_train.py --arm OFF --resume "$CK/${PRE}dagger_d6_s$s.pt" --resume_at 0 --steps 9043968 --comm_on_at 0 \
        $TRUNK_ARGS --seed "$s" --ckpt_every 2 --save "$CK/${PRE}trunk_d6_s$s.pt" --csv "$O/${PRE}trunk_d6_s$s.csv"
  done
  wait
  for s in $SEEDS_ALL; do
    [ -f "$CK/${PRE}trunk_d6_s$s.pt" ] || { echo "★FAIL trunk s$s — $O/${PRE}trunk_d6_s$s.log"; return 1; }
  done
}
gate9() {   # t_ 와 같은 관문(결과 전 고정): 마지막 4창 평균 goal ≥ 30 · oColl ≤ 5. 시드 순서로 앞 3개. G1 은 기록만
  local s L st pass="" g1
  for s in $SEEDS_ALL; do
    L="$O/${PRE}trunk_d6_s$s.log"
    st=$(grep '^\[OFF\] dec=' "$L" 2>/dev/null | tail -4 | sed -E 's/.*goal=([0-9.]+)%.*oColl=([0-9.]+)%.*/\1 \2/' \
         | awk '{g+=$1; o+=$2; n++} END {if (n==4 && g/n>=30 && o/n<=5) v="통과"; else v="불통과"; printf "goal=%.1f oColl=%.1f 창=%d %s", (n?g/n:0), (n?o/n:0), n, v}')
    g1=$(grep '^\[OFF\] dec=' "$L" 2>/dev/null | tail -1 | sed -E 's/.*goal=([0-9.]+)%.*vColl=([0-9.]+)%.*oColl=([0-9.]+)%.*/\1 \2 \3/' \
         | awk '{printf "G1(마지막 창 goal %.1f · vColl %.1f · oColl %.1f) %s", $1, $2, $3, ($1>=90 && $2<=5 && $3<=2) ? "충족" : "미충족"}')
    echo "s$s $st | $g1"
    case "$st" in *" 통과") [ $(echo $pass | wc -w) -lt 3 ] && pass="$pass $s" ;; esac
  done
  pass=$(echo $pass)
  if [ $(echo $pass | wc -w) -lt 3 ]; then echo "trunk 관문: 통과 시드 3개 미만($pass) — 멈춤(기준 안 바꿈, 그대로 보고)"; return 1; fi
  echo "$pass" > "$O/_seeds.txt"; echo "trunk 관문 통과 시드: $pass"
}
forget_check() {   # 기록만(스펙 §5): trunk 1M·2M 근처 창 goal vs 그 시드 P0 학생 goal. 20 pp 넘게 낮으면 'PPO 망각'
  local s L sg
  for s in $SEEDS_ALL; do
    L="$O/${PRE}trunk_d6_s$s.log"
    sg=$(grep -m1 "eps ||" "$O/eval_${PRE}dagger_d6_s$s.txt" 2>/dev/null | sed -E 's/.*goal= *([0-9.]+)%.*/\1/')
    grep '^\[OFF\] dec=' "$L" 2>/dev/null | awk -F'[= |~%]+' -v sg="${sg:-nan}" -v s="$s" '
      {for(i=1;i<=NF;i++){if($i=="dec")d=$(i+1); if($i=="goal")g=$(i+1)}
       if(!a && d+0>=1.0){a=1; g1=g; d1=d} if(!b && d+0>=2.0){b=1; g2=g; d2=d}}
      END{f1=(sg!="nan" && g1+0 < sg-20)?"망각":"-"; f2=(sg!="nan" && g2+0 < sg-20)?"망각":"-";
          printf "s%s 학생(P0) goal %s | dec %s goal %s %s | dec %s goal %s %s\n", s, sg, d1, g1, f1, d2, g2, f2}'
  done
}
watch_stalls() {   # 학습 런 로그가 15분 무진행이면 그 런만 kill(_kill_log.txt) — t_ 와 같음
  while true; do
    sleep 120
    local pids; pids=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'vessel_gym_train' -and \$_.CommandLine -match '${PRE}' } | ForEach-Object { \$m=[regex]::Match(\$_.CommandLine,'${PRE}[a-z0-9_]+_s4\d'); '{0} {1}' -f \$_.ProcessId, \$m.Value }" 2>/dev/null)
    [ -n "$pids" ] || return 0
    echo "$pids" | while read -r pid nm; do
      [ -n "$nm" ] && [ -f "$O/$nm.log" ] || continue
      local age=$(( ($(date +%s) - $(date -r "$O/$nm.log" +%s)) / 60 ))
      if [ "$age" -ge 15 ]; then
        echo "$(date '+%F %T') $nm (PID $pid) kill — 로그 ${age}분 무진행" >> "$O/_kill_log.txt"
        taskkill /F /T /PID "$pid" >/dev/null 2>&1
      fi
    done
  done
}
train_stage() {
  local i rc
  for i in 1 2 3; do
    watch_stalls & local wp=$!
    env VESSEL_SEEDS="$SEEDS" VESSEL_REQUIRE_TRUNK=1 VESSEL_REUSE_ARMS=1 VESSEL_TRAIN_ARMS="$ARMS" $TRAIN_GPU bash run_repro.sh train; rc=$?
    kill $wp 2>/dev/null; wait $wp 2>/dev/null
    [ $rc -eq 0 ] && return 0
    echo "train 시도 $i 실패(rc=$rc) — 재시도"; sleep 30
  done
  return 1
}
quick_table() {   # 스펙 2026-09-30 §7 판정 규칙. 최선 OFF 풀 = i_off + t_off·s_off·x_off(t_ 평가 파일)
  "$PY" "$TDROP/report_parse.py" --dir "$O" --prefix "$PRE" --off off --arms a6 --noise offb --seeds "$(echo $SEEDS | tr ' ' ,)" \
      --extra_off "t_off=$(cygpath -m "$TO_DIR"),s_off=$(cygpath -m "$TO_DIR"),x_off=$(cygpath -m "$TO_DIR")" \
      --json "$O/_primary_table.json" > "$O/_primary_table.md" 2>&1
  cat "$O/_primary_table.md"
}

log "배치 $PRE $MODE $(git -C $R log -1 --oneline) action_mode=$VESSEL_ACTION_MODE teacher=$TEACHER OUT=$O"
if [ "$MODE" = phase0 ]; then
  step smoke   env VESSEL_FORCE_PREFLIGHT=1 VESSEL_LAUNCH_GAP=20 VESSEL_SEEDS=43 VESSEL_SMOKE_ARMS="off a6" $TRAIN_GPU bash run_repro.sh smoke
  step dagger  dagger_all
  step evalp0  evalp0_all
  step p0      p0_table
  grep -q "P0 판정: 통과" "$O/_p0.md" && log "P0 통과 → phase1 실행 가능" || log "P0 불통과 → 멈춤(스펙 §5: 재시도 없이 보고)"
else
  grep -q "P0 판정: 통과" "$O/_p0.md" 2>/dev/null || { log "★멈춤: P0 통과 기록 없음($O/_p0.md) — phase0 먼저"; exit 1; }
  step trunk   trunk_all
  step gate    gate9
  SEEDS=$(cat "$O/_seeds.txt") || exit 1
  log "갈래 시드: $SEEDS"
  step forget  forget_check
  step train1  train_stage
  step eval1   env VESSEL_SEEDS="$SEEDS" $EVAL_GPU bash run_repro.sh eval
  step table1  quick_table
  log "phase1 완료 — 결과 $DROP/out (보고서는 별도 작성)"
fi
sync_out
