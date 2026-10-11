#!/bin/bash
# 2026-09-30 Stage A — 탐색 배치(사후 추가 팔, 확정 근거 아님): 기존 s_ trunk(43·44·45, 9.04M, 보상 = 역할 약속 PEN 20 + 벽 수정)에서
#   s_a6(p6_k6 코덱, 수신측 복원)·s_c6(같은 z, 신경망 직접 읽기) 분기 → 같은 trunk·같은 보상에서 복원 vs 직접 vs s_off.
#   예측(런 전 고정, 스펙 docs/superpowers/specs/2026-09-30-reward-v3-decode-sweep-design.md §2):
#     s_a6 가 s_off 를 goal·vColl·통과최소거리 3/3 이김, fuel·headTravel(도착 평균)은 짐. s_c6 는 goal·vColl 3/3 못 이김.
#   판정 규칙 = s_ 스펙 §7(3/3 + 평균차 > N). 코드 = feat/latent-sweep 9cc8f6d 그대로(변경 0). 워밍업 1200 = s_off 와 동일.
#
# 실행(Windows Git Bash, 한 줄):
#   bash /c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t/_run_stageA.sh
# 단계: guard → train(off 재사용, a6·c6 학습, check_branch) → evalref(s_a6 s_c6 만 — eval 모드는 s_ 전체를 덮어쓰므로 금지)
#       → ablate(a6·c6 × msgzero·shuffle(·decl0), EVAL_DEC 5000 = 빠른 절제) → Dropbox stageA/ 복사. 끝난 단계는 건너뜀.
# GPU: 0–2 만(VESSEL_NGPU=3), GPU 당 2개 상한 + VRAM 검사(외부 프로세스가 GPU 당 ~8 GB 점유 중이면 실질 1개). GPU 3 = 검사 전용.
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
HEAD_REQ=9cc8f6d
R=/c/work/DT_Vessel
DROP=/c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t/stageA
SEEDS="43 44 45"
ARMS="off a6 c6"
_busy=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine" 2>/dev/null | grep -c "vessel_gym_train\|eval_ckpt\|diag_ckpt")
[ "${_busy:-0}" -eq 0 ] || { echo "★멈춤: 학습·평가 python 프로세스 ${_busy}개가 돌고 있음"; exit 1; }
cd $R || exit 1
[ -z "$(git status --short --untracked-files=no)" ] || { echo "★멈춤: 추적 파일 변경 있음 — git status 확인"; exit 1; }
[ "$(git rev-parse --short HEAD)" = "$HEAD_REQ" ] || { echo "★멈춤: HEAD $(git rev-parse --short HEAD) ≠ $HEAD_REQ (Stage A 는 s_ 코드 그대로여야 함)"; exit 1; }
cd $R/Python
for s in $SEEDS; do [ -f "$HOME/VESSEL_checkpoints/comm_intent/s_trunk_d6_s$s.pt" ] || { echo "★멈춤: s_trunk_d6_s$s.pt 없음"; exit 1; }
                    [ -f "$HOME/VESSEL_checkpoints/comm_intent/s_off_s$s.pt" ] || { echo "★멈춤: s_off_s$s.pt 없음(재사용 대상)"; exit 1; }; done
export PYTHONIOENCODING=utf-8 VESSEL_PY=/c/Users/OSH/anaconda3/envs/mltest/python.exe
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0
export VESSEL_COMM_TELEMETRY=1
export VESSEL_CKPT_DIR=$HOME/VESSEL_checkpoints/comm_intent
export VESSEL_NGPU=3 VESSEL_JOBS=6 VESSEL_GPU_CAP=2 VESSEL_LAUNCH_GAP=90 VESSEL_BRANCH_WARMUP=1200
unset VESSEL_REQUIRE_TRUNK VESSEL_REUSE_ARMS VESSEL_SKIP_GOLDEN VESSEL_SEEDS
export VESSEL_RUN_PREFIX=s_ VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
export VESSEL_OUT_DIR=$PWD/_repro_out_s_stageA
O=$VESSEL_OUT_DIR; TL=$O/_timeline.txt; mkdir -p "$O" "$DROP"
echo "$(git rev-parse HEAD)" > "$O/_commit.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$TL"; }
sync_out() { cp "$O"/*.txt "$O"/*.log "$O"/*.csv "$O"/*.json "$DROP/" 2>/dev/null; true; }
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
log "Stage A 시작 $(git log -1 --oneline) PEN=$VESSEL_ROLE_PROMISE_PEN OUT=$O (탐색 배치, 사후 추가 팔)"
step train    env VESSEL_SEEDS="$SEEDS" VESSEL_REQUIRE_TRUNK=1 VESSEL_REUSE_ARMS=1 VESSEL_TRAIN_ARMS="$ARMS" bash run_repro.sh train
step evalref  env VESSEL_SEEDS="$SEEDS" VESSEL_EVAL_NAMES="s_a6 s_c6" bash run_repro.sh evalref
step ablate   env VESSEL_SEEDS="$SEEDS" VESSEL_ABLATE_ARMS="a6 c6" VESSEL_EVAL_DEC=5000 bash run_repro.sh ablate
sync_out
log "Stage A 완료 — 결과 $DROP"
