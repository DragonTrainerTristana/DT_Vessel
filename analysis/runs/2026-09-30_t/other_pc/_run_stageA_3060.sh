#!/bin/bash
# Stage A — 3060 PC 용 (학습만). s_a6·s_c6 × 시드 43·44·45 = 통신 갈래 6개를 기존 s_ 뿌리(통신 없이 9,043,968 결정 학습한 체크포인트)에서 이어 학습.
#   코드 = feat/latent-sweep 9cc8f6d (s_ 배치와 같은 코드·같은 보상). 평가는 서버에서 함(체크포인트 3 MB 짜리 6개를 back/ 으로 돌려보냄).
#   탐색 배치(사후 추가 팔) — 스펙 docs/superpowers/specs/2026-09-30-reward-v3-decode-sweep-design.md §2 의 예측을 런 전에 고정해 둠.
#
# 실행(Git Bash, 한 줄):
#   bash "<Dropbox>/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t/other_pc/_run_stageA_3060.sh"
# 필요하면 앞에 붙임:  VESSEL_REPO=/c/work/DT_Vessel  VESSEL_PY=/c/경로/python.exe  VESSEL_GPU_CAP=1(메모리 빠듯하면 한 번에 1개)
# 다시 실행하면 끝난 갈래는 건너뜀. 로그가 15분 넘게 안 움직이면 알려줄 것(그 런만 죽이고 다시 실행하면 됨).
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
HEAD_REQ=9cc8f6d
R=${VESSEL_REPO:-/c/work/DT_Vessel}
PY=${VESSEL_PY:-python}
B="$(cd "$(dirname "$0")" && pwd)"
CK=$HOME/VESSEL_checkpoints/comm_intent
SEEDS="43 44 45"

if [ ! -d "$R/.git" ]; then
  echo "저장소가 없어서 받음: $R"
  mkdir -p "$(dirname "$R")" && git clone https://github.com/DragonTrainerTristana/DT_Vessel.git "$R" || exit 1
fi
cd "$R" || exit 1
[ -z "$(git status --short --untracked-files=no)" ] || { echo "★멈춤: 추적 파일 변경 있음 — git status 확인"; exit 1; }
git fetch origin --prune && git checkout feat/latent-sweep && git pull --ff-only origin feat/latent-sweep || exit 1
[ "$(git rev-parse --short HEAD)" = "$HEAD_REQ" ] || { echo "★멈춤: HEAD $(git rev-parse --short HEAD) ≠ $HEAD_REQ"; exit 1; }
"$PY" -c "import torch, numpy; assert torch.cuda.is_available(), 'CUDA 안 잡힘'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), '| numpy', numpy.__version__)" \
  || { echo "★멈춤: 이 python 에 torch(CUDA) 가 없음 →  pip install torch --index-url https://download.pytorch.org/whl/cu124  &&  pip install 'numpy<2'"; exit 1; }

# 뿌리·OFF 체크포인트(서버에서 온 바이트 동일 사본) → 로컬. 체크섬이 다르면 멈춤(분기 규약: 같은 뿌리 파일)
mkdir -p "$CK" && cp -n "$B/transfer/ckpt/"*.pt "$CK/"
( cd "$CK" && sha256sum -c "$B/transfer/ckpt/SHA256SUMS.txt" ) || { echo "★멈춤: 체크포인트 체크섬 불일치"; exit 1; }

cd "$R/Python" || exit 1
export PYTHONIOENCODING=utf-8 VESSEL_PY="$PY"
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0
export VESSEL_COMM_TELEMETRY=1
export VESSEL_CKPT_DIR=$CK
# 3060 12 GB: 통신 갈래 1개 ≈ 5 GB. 두 번째는 첫 번째가 메모리를 잡은 뒤(150 s) 남은 VRAM 을 보고 배정(모자라면 첫 런이 끝날 때까지 기다림)
export VESSEL_NGPU=1 VESSEL_JOBS=${VESSEL_JOBS:-2} VESSEL_GPU_CAP=${VESSEL_GPU_CAP:-2} VESSEL_LAUNCH_GAP=150 VESSEL_VRAM_MARGIN=700
export VESSEL_BRANCH_WARMUP=1200          # s_off 와 같은 워밍업
# 골든 비트동일 검사는 서버의 torch 빌드 기준이라 다른 torch 에선 원래 안 맞음 → 건너뜀(미러 검사는 그대로 돎). 보고서에 명시
export VESSEL_SKIP_GOLDEN=1
unset VESSEL_REQUIRE_TRUNK VESSEL_REUSE_ARMS VESSEL_SEEDS
export VESSEL_RUN_PREFIX=s_ VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
export VESSEL_OUT_DIR=$PWD/_repro_out_s_stageA
O=$VESSEL_OUT_DIR; TL=$O/_timeline.txt; mkdir -p "$O" "$B/back"
cp -n "$B/transfer/out_s/"*.csv "$O/" 2>/dev/null        # 뿌리 곡선 CSV(갈래 곡선이 0 부터 이어지게)
log() { echo "$(date '+%F %T') $*" | tee -a "$TL"; }
back() { cp "$O"/*.txt "$O"/*.log "$O"/*.csv "$B/back/" 2>/dev/null; for s in $SEEDS; do for a in a6 c6; do [ -f "$CK/s_${a}_s$s.pt" ] && cp "$CK/s_${a}_s$s.pt" "$B/back/"; done; done; true; }

log "Stage A(3060) 시작 $(git -C "$R" log -1 --oneline) | $("$PY" -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))")"
for try in 1 2 3; do
  env VESSEL_SEEDS="$SEEDS" VESSEL_REQUIRE_TRUNK=1 VESSEL_REUSE_ARMS=1 VESSEL_TRAIN_ARMS="off a6 c6" bash run_repro.sh train > "$O/_train.log" 2>&1
  rc=$?; back
  [ $rc -eq 0 ] && break
  log "train 시도 $try rc=$rc — $O/_train.log 확인(끝난 갈래는 다음 시도에서 재사용)"; sleep 30
done
if [ $rc -eq 0 ]; then
  log "끝 train — 체크포인트 6개를 $B/back/ 에 복사함(서버에서 평가)"
  ( cd "$B/back" && sha256sum s_a6_s4?.pt s_c6_s4?.pt > SHA256SUMS.txt ) 2>/dev/null
  echo "$(date '+%F %T') done" > "$B/back/_DONE.txt"
else
  log "★멈춤 train rc=$rc"
fi
back
