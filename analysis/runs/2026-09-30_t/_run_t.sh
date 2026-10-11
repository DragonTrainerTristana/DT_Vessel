#!/bin/bash
# 2026-09-30 배치 t_ — 보상 v3 + 복원형 latent 사다리. 스펙 docs/superpowers/specs/2026-09-30-reward-v3-decode-sweep-design.md (feat/reward-v3)
#
# 실행(Windows Git Bash, 한 줄. Stage A 학습이 끝난 뒤 — GPU 0–2 가 비어야 함):
#   bash /c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t/_run_t.sh
#
# 순서 — 학습 전 검사가 하나라도 FAIL 이면 학습을 시작하지 않고 멈춤:
#   0 guard   = 학습·평가 프로세스 없음 · 깨끗한 트리 · HEAD == HEAD_REQ · 코덱 파일
#   1 smoke   = preflight 강제(검사 14종: +보상 v3·판정기 v2) + 파이프라인(off·a6·a2 1-update trunk→분기→check_branch)
#   2 rank    = 순위 게이트 v2(정책 13 × 설정 4 × 시드 3, GPU 3) → 'DECISION' 줄로 DCPA 게이트 포함 여부 결정(스펙 §4, 결과 무관 규칙)
#   3 evaldiff= eval 출력 바이트 차분(h_off_s43·h_a6_s43, 새 줄 태그만 허용) — rank 와 동시
#   4 trunk   = t_trunk 5개(시드 43–47, 9.04M, 보상 v3) + 감시(3M: TO ≤ 40 % & len ≤ 2000 · 6.2–6.5M: ≥3 trunk goal ≥ 20 % oColl ≤ 10 % TO ≤ 10 %)
#               위반이면 trunk 전부 kill 하고 멈춤(자동 보상 전환 없음 — 저자 결정)
#   5 gate    = 9.04M 관문(마지막 4창 goal ≥ 30 % · oColl ≤ 5 %, 시드 순 앞 3개 → _seeds.txt). 3개 미만이면 멈춤
#   6 train   = wave 1: t_off·t_a6(주)·t_a8 → wave 2: t_a2·t_a4·t_offb(잡음). GPU 당 2개·간격 90 s·워밍업 2400
#   7 eval → 8 evalref(s_off·x_off·h_a6·s_a6 을 v2 판정으로 재채점, 옛 보상 env) → 9 ablate → 10 traj → 11 plots → Dropbox out/
# 규칙: 단계 실패면 즉시 멈춤. 끝난 단계는 다시 돌릴 때 건너뜀(_timeline.txt 의 '끝 <단계>'). HEAD 가 바뀌면 멈춤(→ VESSEL_T_PREFIX=t2_).
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
HEAD_REQ=${VESSEL_T_HEAD:-3aa0636}      # feat/reward-v3 3aa0636 (코드 88f088d + 스펙 기록, 2026-09-30 13:50). 코드가 바뀌면 새 접두어(t2_)+새 해시
R=/c/work/DT_Vessel_v3
DROP=/c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t
PRE=${VESSEL_T_PREFIX:-t_}
TRUNK_SEEDS="43 44 45 46 47"
ARMS1="off a6 offb"      # ★주 비교(OFF · a6 · 재분기 잡음 자) — 먼저 학습·평가해서 핵심 표를 가장 빨리
ARMS2="off a8 a2 a4"       # 보조(a8) + latent 사다리(a2·a4). VESSEL_T_SWEEP_HERE=0 이면 a8 만(사다리는 다른 컴퓨터)
[ "${VESSEL_T_SWEEP_HERE:-1}" = 1 ] || ARMS2="off a8"
ALL_ARMS="off a6 a8 a2 a4 offb"
COMM_ARMS="a6 a8 a2 a4"; [ "${VESSEL_T_SWEEP_HERE:-1}" = 1 ] || COMM_ARMS="a6 a8"
PYB=/c/Users/OSH/anaconda3/python.exe    # matplotlib 있는 base python(그림 전용)
[ "$HEAD_REQ" != TBD ] || { echo "★멈춤: HEAD_REQ 미설정(코드 커밋 뒤 VESSEL_T_HEAD=<hash> 또는 스크립트 수정)"; exit 1; }
# 학습 프로세스만 막음(Stage A 의 evalref·ablate 는 GPU 당 1–2 GB 라 trunk 와 동거 가능 — pick_gpu 의 VRAM 여유 검사가 막아 줌)
_busy=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine" 2>/dev/null | grep -c "vessel_gym_train")
[ "${_busy:-0}" -eq 0 ] || { echo "★멈춤: 학습 python 프로세스 ${_busy}개가 돌고 있음(Stage A train?) — 끝난 뒤 실행"; exit 1; }
cd $R || exit 1
[ -z "$(git status --short --untracked-files=no)" ] || { echo "★멈춤: 추적 파일 변경 있음 — git status 확인"; exit 1; }
[ "$(git rev-parse --short HEAD)" = "$(git rev-parse --short $HEAD_REQ)" ] || { echo "★멈춤: HEAD $(git rev-parse --short HEAD) ≠ $HEAD_REQ"; exit 1; }
cd $R/Python
for f in p6_k6_s0 p6_k2_s0 p6_k4_s0 p12_k8_s0; do [ -f comm_codecs/$f.pt ] || { echo "★멈춤: comm_codecs/$f.pt 없음"; exit 1; }; done
export PYTHONIOENCODING=utf-8 VESSEL_PY=/c/Users/OSH/anaconda3/envs/mltest/python.exe
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0
export VESSEL_COMM_TELEMETRY=1
export VESSEL_CKPT_DIR=$HOME/VESSEL_checkpoints/comm_intent
export VESSEL_BRANCH_WARMUP=2400
unset VESSEL_REQUIRE_TRUNK VESSEL_REUSE_ARMS VESSEL_SKIP_GOLDEN VESSEL_SEEDS VESSEL_NGPU
export VESSEL_RUN_PREFIX=$PRE VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
# ★보상 v3(스펙 §3). DCPA 게이트 값은 rank 단계의 DECISION 으로 확정(48 또는 0). 판정기 v2.
export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035
export VESSEL_RISK_DCPA_GATE_M=${VESSEL_RISK_DCPA_GATE_M:-48}
export VESSEL_OUT_DIR=$PWD/_repro_out_${PRE%_}
O=$VESSEL_OUT_DIR; TL=$O/_timeline.txt; mkdir -p "$O" "$DROP/out"
HEAD_NOW=$(git -C $R rev-parse HEAD)
if [ -f "$O/_commit.txt" ]; then
  [ "$(cat "$O/_commit.txt")" = "$HEAD_NOW" ] || { echo "★멈춤: 이 배치는 $(cat "$O/_commit.txt") 로 시작했는데 지금 HEAD 는 $HEAD_NOW — 코드 바뀜 → VESSEL_T_PREFIX=t2_ 로 새로"; exit 1; }
else
  echo "$HEAD_NOW" > "$O/_commit.txt"
fi
# 학습 단계 GPU 배정: GPU 당 2개(ON 4.8 GB × 2 + 외부 프로세스 여유), 간격 90 s. 평가 단계는 12/3.
TRAIN_GPU="VESSEL_JOBS=8 VESSEL_GPU_CAP=2 VESSEL_LAUNCH_GAP=90 VESSEL_VRAM_MARGIN=1500"   # 남의 점유(GPU당 7–10 GB)가 있어 여유 1.5 GB: ON 4.8+1.5 = 6.3 GB 남은 GPU 에만 배정(3.5 GB 면 GPU 0–2 에 ON 이 하나도 못 들어감). 정지는 watch_stalls 가 회수
EVAL_GPU="VESSEL_JOBS=12 VESSEL_GPU_CAP=3 VESSEL_LAUNCH_GAP=10"

log() { echo "$(date '+%F %T') $*" | tee -a "$TL"; }
sync_out() { cp "$O"/*.txt "$O"/*.log "$O"/*.csv "$O"/*.json "$DROP/out/" 2>/dev/null; cp "$O"/*.png "$O"/*.pdf "$DROP/out/" 2>/dev/null; true; }
bad() {
  grep -q "★FAIL\|FAIL \|실패" "$1" 2>/dev/null && return 0
  for f in "$O"/_status_*.txt; do [ -f "$f" ] && grep -qv 'rc=0' "$f" && return 0; done
  return 1
}
BG=""
bgstep() {
  local nm=$1; shift
  if grep -q " 끝 $nm\$" "$TL" 2>/dev/null; then log "건너뜀 $nm (이미 끝남)"; return 0; fi
  log "시작 $nm (동시)"
  rm -f "$O/_$nm.rc"
  ( "$@" > "$O/_$nm.log" 2>&1; echo $? > "$O/_$nm.rc" ) &
  BG="$BG $nm"
}
bgwait() {
  local nm rc
  wait
  for nm in $BG; do
    rc=$(cat "$O/_$nm.rc" 2>/dev/null || echo 99)
    if [ "$rc" != 0 ] || grep -q "★FAIL\|FAIL \|실패" "$O/_$nm.log" 2>/dev/null; then
      log "★멈춤 $nm rc=$rc — $O/_$nm.log 확인"; sync_out; exit 1; fi
    log "끝 $nm"
  done
  BG=""
}
step() {
  local nm=$1; shift
  if grep -q " 끝 $nm\$" "$TL" 2>/dev/null; then log "건너뜀 $nm (이미 끝남)"; return 0; fi
  log "시작 $nm"
  "$@" > "$O/_$nm.log" 2>&1; local rc=$?
  if [ $rc -ne 0 ] || bad "$O/_$nm.log"; then log "★멈춤 $nm rc=$rc — $O/_$nm.log 확인"; sync_out; exit 1; fi
  log "끝 $nm"; sync_out
}
# ── 순위 게이트 결정(스펙 §4): DECISION 줄을 읽어 DCPA 게이트 값을 확정. 둘 다 실패면 멈춤 ──
rank_decide() {
  local L="$O/_rank.log"
  if grep -q "DECISION: run without DCPA gate" "$L"; then
    export VESSEL_RISK_DCPA_GATE_M=0; log "순위 게이트: v3 불통과·v3−게이트 통과 → DCPA 게이트 없이 진행(RISK_DCPA_GATE_M=0)"
  elif grep -q "DECISION: stop" "$L"; then
    log "★멈춤: 순위 게이트 v3·v3−게이트 모두 불통과 — 저자 결정 필요($L)"; sync_out; exit 1
  elif grep -q "RANK GATE v2: PASS" "$L"; then
    log "순위 게이트 v2 PASS → RISK_DCPA_GATE_M=$VESSEL_RISK_DCPA_GATE_M 그대로"
  else
    log "★멈춤: 순위 게이트 결과 줄을 못 읽음($L)"; sync_out; exit 1
  fi
  echo "$VESSEL_RISK_DCPA_GATE_M" > "$O/_gate_m.txt"
}
# ── trunk 감시(스펙 §5): 3 분마다 로그 파싱. 위반이면 trunk 전부 kill 하고 멈춤 ──
watch_trunks() {
  local s L k
  while true; do
    sleep 180
    local alive=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine" 2>/dev/null | grep -c "${PRE}trunk_d6")
    [ "${alive:-0}" -gt 0 ] || return 0
    for s in $TRUNK_SEEDS; do
      L="$O/${PRE}trunk_d6_s$s.log"; [ -f "$L" ] || continue
      # 3M 검사: dec ≥ 3.0M 인 첫 창에서 TO ≤ 40 % & len ≤ 2000
      k=$(grep '^\[OFF\] dec=' "$L" | awk -F'[= |~%]+' '{for(i=1;i<=NF;i++){if($i=="dec")d=$(i+1); if($i=="len")l=$(i+1); if($i=="TO")t=$(i+1)}; if(d+0>=3.0){ if(t+0>40 || l+0>2000) print "BAD dec="d" TO="t" len="l; else print "OK"; exit}}')
      case "$k" in BAD*) log "★trunk 감시 위반 s$s @3M: $k → 전부 kill·멈춤"; kill_trunks; sync_out; exit 1 ;; esac
    done
    # 6.2–6.5M 검사: 5개 모두 6.5M 을 지났으면 ≥3 개가 goal ≥ 20 · oColl ≤ 10 · TO ≤ 10 (그 구간 창 평균)
    local npass=0 ndone=0
    for s in $TRUNK_SEEDS; do
      L="$O/${PRE}trunk_d6_s$s.log"; [ -f "$L" ] || continue
      k=$(grep '^\[OFF\] dec=' "$L" | awk -F'[= |~%]+' '{for(i=1;i<=NF;i++){if($i=="dec")d=$(i+1); if($i=="goal")g=$(i+1); if($i=="oColl")o=$(i+1); if($i=="TO")t=$(i+1)}; if(d+0>=6.2 && d+0<=6.5){n++; G+=g; Oc+=o; T+=t} if(d+0>6.5) past=1} END{ if(!past) print "WAIT"; else if(n>0 && G/n>=20 && Oc/n<=10 && T/n<=10) print "PASS"; else print "FAIL"}')
      case "$k" in PASS) npass=$((npass+1)); ndone=$((ndone+1)) ;; FAIL) ndone=$((ndone+1)) ;; esac
    done
    if [ "$ndone" -ge 5 ] && [ "$npass" -lt 3 ]; then log "★trunk 감시 위반 @6.5M: 통과 $npass/5 < 3 → 전부 kill·멈춤"; kill_trunks; sync_out; exit 1; fi
    [ "$ndone" -ge 5 ] && { log "trunk 감시 @6.5M 통과 $npass/5"; return 0; }
  done
}
kill_trunks() {
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '${PRE}trunk_d6' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null
  sleep 5
}
gate9() {   # 9.04M 관문(스펙 §5, 결과 전 고정): 마지막 4창 평균 goal ≥ 30 · oColl ≤ 5. 시드 순서로 앞 3개
  local s L st pass=""
  for s in $TRUNK_SEEDS; do
    L="$O/${PRE}trunk_d6_s$s.log"
    st=$(grep '^\[OFF\] dec=' "$L" 2>/dev/null | tail -4 | sed -E 's/.*goal=([0-9.]+)%.*oColl=([0-9.]+)%.*/\1 \2/' \
         | awk '{g+=$1; o+=$2; n++} END {if (n==4 && g/n>=30 && o/n<=5) v="통과"; else v="불통과"; printf "goal=%.1f oColl=%.1f 창=%d %s", (n?g/n:0), (n?o/n:0), n, v}')
    echo "s$s $st"
    case "$st" in *" 통과") [ $(echo $pass | wc -w) -lt 3 ] && pass="$pass $s" ;; esac
  done
  pass=$(echo $pass)
  if [ $(echo $pass | wc -w) -lt 3 ]; then echo "trunk 관문: 통과 시드 3개 미만($pass) — 멈춤(보상·기준 안 바꿈, 그대로 보고)"; return 1; fi
  echo "$pass" > "$O/_seeds.txt"; echo "trunk 관문 통과 시드: $pass"
}
trunk_with_watch() {
  watch_trunks & local wp=$!
  env VESSEL_SEEDS="$TRUNK_SEEDS" $TRAIN_GPU bash run_repro.sh trunk; local rc=$?
  kill $wp 2>/dev/null; wait $wp 2>/dev/null
  grep -q "★trunk 감시 위반" "$TL" && return 1
  return $rc
}
evalref_old() {   # 옛 체크포인트를 v2 판정으로 재채점 — 보상 env 는 옛 값(epReward 비교 유지), 판정만 --role_judge v2
  env VESSEL_SEEDS="43 44 45" VESSEL_RUN_PREFIX= VESSEL_FORWARD_COEF=0.1 VESSEL_TIME_PENALTY=0.07 VESSEL_RISK_DCPA_GATE_M=0 \
      VESSEL_ROLE_JUDGE=end VESSEL_EVAL_EXTRA="--role_judge v2" VESSEL_EVAL_NAMES="s_off x_off h_a6" $EVAL_GPU bash run_repro.sh evalref
}
plots() {
  "$PYB" plotting/plot_reward_v3.py --dir "$O" --prefix "$PRE" --arms off,a6,a8,a2,a4,offb --seeds "$(cat "$O/_seeds.txt" | tr ' ' ,)" --out "$O/_reward_v3" && \
  "$PYB" plotting/plot_ep_return.py --dir "$O" --prefix "$PRE" --arms off,a6,a8,a2,a4,offb --seeds "$(cat "$O/_seeds.txt" | tr ' ' ,)" --out "$O/_ep_return"
}

log "배치 $PRE 시작 $(git -C $R log -1 --oneline) PEN=$VESSEL_ROLE_PROMISE_PEN JUDGE=$VESSEL_ROLE_JUDGE FORWARD=$VESSEL_FORWARD_COEF TIME=$VESSEL_TIME_PENALTY GATE=$VESSEL_RISK_DCPA_GATE_M OUT=$O"
step smoke    env VESSEL_FORCE_PREFLIGHT=1 VESSEL_LAUNCH_GAP=20 VESSEL_SEEDS=43 VESSEL_SMOKE_ARMS="off a6 a2" $TRAIN_GPU bash run_repro.sh smoke
bgstep evaldiff env VESSEL_FORWARD_COEF=0.1 VESSEL_TIME_PENALTY=0.07 VESSEL_RISK_DCPA_GATE_M=0 VESSEL_ROLE_JUDGE=end \
                "$VESSEL_PY" -u verify/test_eval_diff.py --ckpt h_off_s43.pt --arm OFF --ckpt h_a6_s43.pt --arm ON
rank_step() {   # 순위 게이트 v2 — step() 을 쓰지 않음: 출력 리다이렉트가 선행 실행의 _rank.log 를 덮어쓰고, v3 불통과·v3−게이트 통과(정상 분기)의 '★FAIL' 글자에 멈추기 때문
  if grep -q " 끝 rank\$" "$TL" 2>/dev/null; then log "건너뜀 rank (이미 끝남)"; return 0; fi
  local n
  n=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine" 2>/dev/null | grep -c "check_reward_rank")
  if [ "${n:-0}" -gt 0 ]; then
    log "rank: 선행 실행 ${n}개 진행 중 → 끝날 때까지 대기"
    while [ "$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine" 2>/dev/null | grep -c "check_reward_rank")" -gt 0 ]; do sleep 60; done
    sleep 5
  fi
  if ! grep -q "RANK GATE v2: " "$O/_rank.log" 2>/dev/null; then
    log "시작 rank(새로 실행)"
    env CUDA_VISIBLE_DEVICES=3 "$VESSEL_PY" -u verify/check_reward_rank.py --device cuda --jobs 2 --settings v3,v3nogate --out "$O/_reward_rank.json" > "$O/_rank.log" 2>&1
  fi
  grep -q "RANK GATE v2: " "$O/_rank.log" || { log "★멈춤 rank — $O/_rank.log 에 결과 줄 없음"; sync_out; exit 1; }
  log "끝 rank"
}
bgwait
rank_step
[ -f "$O/_gate_m.txt" ] && export VESSEL_RISK_DCPA_GATE_M=$(cat "$O/_gate_m.txt") || rank_decide
step trunk    trunk_with_watch
step gate     gate9
SEEDS=$(cat "$O/_seeds.txt") || exit 1
log "갈래 시드: $SEEDS"
watch_stalls() {   # 학습 런 로그가 15분 무진행이면 그 런만 kill(_kill_log.txt). 외부 GPU 점유(다른 사용자)로 WDDM 페이징에 빠진 런 회수용
  while true; do
    sleep 120
    local pids; pids=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'vessel_gym_train' -and \$_.CommandLine -match '${PRE}' } | ForEach-Object { \$m=[regex]::Match(\$_.CommandLine,'${PRE}[a-z0-9_]+_s4\d'); '{0} {1}' -f \$_.ProcessId, \$m.Value }" 2>/dev/null)
    [ -n "$pids" ] || return 0
    echo "$pids" | while read -r pid nm; do
      [ -n "$nm" ] && [ -f "$O/$nm.log" ] || continue
      local age=$(( ($(date +%s) - $(date -r "$O/$nm.log" +%s)) / 60 ))
      if [ "$age" -ge 15 ]; then
        echo "$(date '+%F %T') $nm (PID $pid) kill — 로그 ${age}분 무진행 (GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '
' ' '))" >> "$O/_kill_log.txt"
        taskkill /F /T /PID "$pid" >/dev/null 2>&1
      fi
    done
  done
}
train_stage() {   # 팔 목록을 최대 3회 시도(정지된 런은 감시가 kill → 다음 시도에서 trunk 부터 재학습, 끝난 런은 재사용)
  local arms="$1" i rc
  for i in 1 2 3; do
    watch_stalls & local wp=$!
    env VESSEL_SEEDS="$SEEDS" VESSEL_REQUIRE_TRUNK=1 VESSEL_REUSE_ARMS=1 VESSEL_TRAIN_ARMS="$arms" $TRAIN_GPU bash run_repro.sh train; rc=$?
    kill $wp 2>/dev/null; wait $wp 2>/dev/null
    [ $rc -eq 0 ] && return 0
    echo "train 시도 $i 실패(rc=$rc) — 정지 kill 뒤 재시도"; sleep 30
  done
  return 1
}
quick_table() {   # 주 비교 표(사전등록 판정 규칙, 스펙 §7) — 로그·Dropbox 에 남김. 판정 문구는 보고서에서
  "$VESSEL_PY" "$DROP/report_parse.py" --dir "$O" --prefix "$PRE" --off off --arms a6 --noise offb --seeds "$(echo $SEEDS | tr ' ' ,)" \
      --extra_off "s_off=$(cygpath -m "$O"),x_off=$(cygpath -m "$O")" --json "$O/_primary_table.json" > "$O/_primary_table.md" 2>&1
  cat "$O/_primary_table.md"
}
eval2() {   # 2차 팔만 평가(1차 팔 재평가 방지): 있는 체크포인트 이름만 evalref 로
  local names="" a
  for a in a8 a2 a4; do ls "$VESSEL_CKPT_DIR/${PRE}${a}_s"*.pt >/dev/null 2>&1 && names="$names ${PRE}$a"; done
  [ -n "$names" ] || { echo "2차 팔 체크포인트 없음"; return 1; }
  env VESSEL_SEEDS="$SEEDS" VESSEL_EVAL_NAMES="$(echo $names)" $EVAL_GPU bash run_repro.sh evalref
}
step train1   train_stage "$ARMS1"
step eval1    env VESSEL_SEEDS="$SEEDS" $EVAL_GPU bash run_repro.sh eval
step table1   quick_table
step train2   train_stage "$ARMS2"
step eval2    eval2
step evalref  evalref_old
step ablate   env VESSEL_SEEDS="$SEEDS" VESSEL_ABLATE_ARMS="$COMM_ARMS" $EVAL_GPU bash run_repro.sh ablate
step traj     env VESSEL_SEEDS="43" VESSEL_TRAJ_ARMS="off a6" $EVAL_GPU bash run_repro.sh traj
step plots    plots
sync_out
log "전 단계 완료 — 결과 $DROP/out (보고서는 별도 작성)"
# Stage A(탐색: s_a6·s_c6, 13:34 외부 GPU 점유로 정지 → kill)를 t_ 뒤에 다시 돌림(코드 9cc8f6d 트리, 자체 드라이버)
if [ "${VESSEL_T_STAGEA_AFTER:-0}" = 1 ]; then
  log "Stage A 재실행 시작(t_ 뒤)"
  bash "$DROP/_run_stageA.sh" > "$O/_stageA_after.log" 2>&1 && log "Stage A 재실행 끝" || log "★Stage A 재실행 실패 — $O/_stageA_after.log"
fi
