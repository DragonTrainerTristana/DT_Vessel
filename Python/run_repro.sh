#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# 재현 실행 스크립트 (2026-09-05)
#
# 왜 있나: 지금까지 배치는 `runs/m2_ablation/structfix/run_batch.sh` 로 돌렸는데
#   그 파일은 (1) git root(Assets/Scripts) *밖*이라 저장소에 안 올라가고
#   (2) 체크포인트 경로·파이썬 경로가 절대경로로 박혀 있으며
#   (3) GPU 를 4장으로 가정해(`i % 4`) 1~2장 머신에서는 CUDA_VISIBLE_DEVICES=2,3 을
#       받은 런이 조용히 CPU 로 떨어지거나 죽는다.
#   → 저장소를 클론한 제3자가 그대로 돌릴 수 있는 판(version)을 저장소 안에 둔다.
#
# 무엇을 재현하나: ★2026-09-10 부터 **YUGIOH 최종판**(config.py 끝 `YUGIOH` 표 = config 기본값).
#   = 2026-09-04 배치(공유 MoE·attention·중앙 critic·상태복원) + commfix 09-07(leaky·bottleneck·token gain 8·
#     per-module clip·msg_l2 2e-4·comm 300·state_recon 0.05) + 레이더 인코더 망 간 공유 09-10.
#   common_env 는 그 표를 *명시* export 하고 preflight 가 config 기본값과 대조한다(드리프트 시 중단).
#   팔:   OFF(통신 없음) / ON dim6 / ON dim12   × 시드 43·44·45
#   ⚠️그 배치에서 off_s45 는 학습에 실패해 결과에서 제외됐다(최종보상 1.06 vs 형제 1.53·1.59).
#     제외는 통신에 *불리한* 방향이라 보수적 선택이다. 자세한 건 runs/m2_ablation/COMM_PLAN.md §4-B.
#
# 쓰는 법:
#   bash run_repro.sh smoke     # 4만 스텝 짜리 확인용 (몇 분)
#   bash run_repro.sh train     # 본 배치 16.06M × 9런
#   bash run_repro.sh eval      # 학습된 체크포인트 평가
#   bash run_repro.sh random    # 난수 메시지 대조군 (기본 팔에 없음 — 아래 설명)
#
# 환경변수로 바꿀 수 있는 것 (전부 기본값 있음):
#   VESSEL_CKPT_DIR  체크포인트 저장 위치. 기본 $HOME/VESSEL_checkpoints/<날짜>_repro
#                    ⚠️Dropbox 같은 동기화 폴더에 두지 말 것 — 배치 하나가 GB 단위다.
#   VESSEL_PY        파이썬 실행 파일. 기본 python
#   VESSEL_OUT_DIR   CSV·로그 저장 위치. 기본 이 파일 옆 _repro_out/
#   VESSEL_SEEDS     시드 목록. 기본 "43 44 45"
#   VESSEL_NGPU      쓸 GPU 수. 기본은 torch 로 자동 감지(0장이면 1로 두고 CPU)
#   VESSEL_JOBS      동시 실행 프로세스 수. 기본 = NGPU × 2 (VRAM 프로세스당 ~5.2GB 기준)
# ─────────────────────────────────────────────────────────────────────────────
set -u

MODE="${1:-smoke}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${VESSEL_PY:-python}"
OUT="${VESSEL_OUT_DIR:-$HERE/_repro_out}"
CK="${VESSEL_CKPT_DIR:-$HOME/VESSEL_checkpoints/$(date +%Y-%m-%d)_repro}"
SEEDS="${VESSEL_SEEDS:-43 44 45}"

mkdir -p "$OUT" "$CK"

# ── GPU 수 자동 감지 (하드코딩 금지) ────────────────────────────────────────
if [ -n "${VESSEL_NGPU:-}" ]; then
  NGPU="$VESSEL_NGPU"
else
  NGPU=$("$PY" -c "
try:
    import torch; print(max(1, torch.cuda.device_count()))
except Exception:
    print(1)
" 2>/dev/null || echo 1)
fi
JOBS="${VESSEL_JOBS:-$(( NGPU * 2 ))}"

echo "재현 실행 [$MODE]"
echo "  python      : $PY"
echo "  체크포인트  : $CK"
echo "  출력        : $OUT"
echo "  시드        : $SEEDS"
echo "  GPU 수      : $NGPU   동시 실행: $JOBS"
echo

# ── 학습·평가 공통 설정 = YUGIOH (config.py 끝 `YUGIOH` 표와 1:1) ──────────────
# 전부 config 기본값과 같지만 *명시* 한다 — 로그·스냅샷만 보고 설정을 알 수 있게, 그리고 preflight 가 대조하게.
common_env() {
  export VESSEL_USE_ATTENTION=1
  export VESSEL_CENTRAL_CRITIC=1
  export VESSEL_STATE_RECON_COEF=0.05
  export VESSEL_USE_MOE=1
  export VESSEL_MOE_SHARED=1
  export VESSEL_MOE_WIDTH=1.0
  export VESSEL_SHARED_ENCODER=all
  export VESSEL_RADAR_ACT=leaky
  export VESSEL_RADAR_HEAD=bottleneck
  export VESSEL_RADAR_BOTTLENECK_CH=8
  export VESSEL_MSG_LN=1
  export VESSEL_MSG_TOKEN_GAIN=8.0
  export VESSEL_CLIP_PER_MODULE=1
  export VESSEL_MSG_L2=0.0002
  export VESSEL_POS_GROUND=1
  export VESSEL_COMM_RANGE=300
  export VESSEL_MAX_PARTNERS=4
  export VESSEL_RADAR_RANGE=56
  export VESSEL_COLREGS_MODE=unity
  export VESSEL_SIM_COLREGS_COEF=0.45
  export VESSEL_INTENT_K=3
  export VESSEL_THREAT_COEF=0
  export VESSEL_GOAL_COMM_COEF=0
  export VESSEL_INTENT_COEF=0
  export VESSEL_ROLE_COMM_COEF=0
  export VESSEL_COMM_CONSUMER_COEF=0
  export VESSEL_RECON_EMA_FLOOR=0
  export VESSEL_AGG_MODE=sum
  export VESSEL_MSG_GAIN=1.0
  export VESSEL_TIMEOUT_BOOTSTRAP=0
  export VESSEL_MSG_GATE_APPLY=0
}

# ── 사전 검증: 미러가 깨졌으면 돌리지 말 것 ─────────────────────────────────
preflight() {
  # ★YUGIOH 드리프트 검사: common_env 의 export 값 == config.py 기본값 (누가 config 기본값만 바꾸면 여기서 잡힘)
  ( common_env; "$PY" - <<'PYCHK'
import os, json, subprocess, sys
env = {k: v for k, v in os.environ.items() if not k.startswith('VESSEL_')}
code = "import json, config as c; print(json.dumps({k: str(getattr(c, k)) for k in %r}))"
names = ['USE_ATTENTION','CENTRAL_CRITIC','STATE_RECON_COEF','MOE_SHARED','SHARED_ENCODER','RADAR_ACT','RADAR_HEAD',
         'MSG_TOKEN_GAIN','CLIP_PER_MODULE','MSG_L2_COEF','COMM_RANGE','MSG_LN','POS_GROUND','MOE_WIDTH','USE_MOE']
a = json.loads(subprocess.run([sys.executable, '-c', code % names], env=env, capture_output=True, text=True).stdout.strip().splitlines()[-1])
b = json.loads(subprocess.run([sys.executable, '-c', code % names], capture_output=True, text=True).stdout.strip().splitlines()[-1])
bad = [k for k in names if a[k] != b[k]]
print('  YUGIOH 드리프트:', 'PASS (common_env == config 기본값)' if not bad else f'★FAIL {bad}')
sys.exit(1 if bad else 0)
PYCHK
  ) || { echo "preflight 실패: common_env 와 config.py 기본값이 다름 — config.py 끝 YUGIOH 표를 볼 것"; exit 1; }
  echo "[preflight] PPO·통신 미러 검증"
  common_env
  "$PY" -u "$HERE/_verify_ppo_mirror.py"  > "$OUT/_verify_ppo.txt"  2>&1 || { echo "  PPO 미러 FAIL — $OUT/_verify_ppo.txt 확인"; exit 1; }
  "$PY" -u "$HERE/_verify_comm_mirror.py" > "$OUT/_verify_comm.txt" 2>&1 || { echo "  통신 미러 FAIL — $OUT/_verify_comm.txt 확인"; exit 1; }
  grep -q "ALL PASS" "$OUT/_verify_ppo.txt"  || { echo "  PPO 미러가 ALL PASS 가 아님"; exit 1; }
  grep -q "ALL PASS" "$OUT/_verify_comm.txt" || { echo "  통신 미러가 ALL PASS 가 아님"; exit 1; }
  echo "  둘 다 ALL PASS"
  # ★2026-09-10: 기본값 비트동일 골든 + vessel_gym 충실도. VESSEL_SKIP_GOLDEN=1 로 건너뜀(수 분 걸림).
  if [ "${VESSEL_SKIP_GOLDEN:-0}" != "1" ]; then
    echo "[preflight] 골든 비트동일 검사"
    ( cd "$HERE" && env -u VESSEL_STATE_RECON_COEF -u VESSEL_CENTRAL_CRITIC -u VESSEL_USE_ATTENTION \
        "$PY" -u test_golden.py --check ) > "$OUT/_golden.txt" 2>&1 \
      || { echo "  골든 FAIL — $OUT/_golden.txt 확인 (코드가 기본값 결과를 바꿨음)"; exit 1; }
    grep -q "ALL PASS" "$OUT/_golden.txt" || { echo "  골든이 ALL PASS 가 아님"; exit 1; }
    echo "  골든 ALL PASS"
    "$PY" -u "$HERE/test_vessel_gym_fidelity.py" > "$OUT/_fidelity.txt" 2>&1 \
      || { echo "  vessel_gym 충실도 FAIL — $OUT/_fidelity.txt 확인"; exit 1; }
    echo "  충실도 PASS"
  fi
  echo
}

# 동시 실행 수 제한 (GPU 라운드로빈)
GPU_I=0
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 2; done; }

# ── 학습 한 런 ──────────────────────────────────────────────────────────────
# 인자: 이름 arm msg_dim seed steps
train_one() {
  local nm=$1 arm=$2 dim=$3 s=$4 steps=$5
  local gpu=$(( GPU_I % NGPU )); GPU_I=$(( GPU_I + 1 ))
  throttle
  (
    common_env
    export CUDA_VISIBLE_DEVICES=$gpu
    export OMP_NUM_THREADS=2
    export VESSEL_MSG_DIM=$dim
    # ★arm=OFF 런은 VESSEL_USE_COMM=0 을 명시한다.
    #   동작은 어차피 --arm 이 정하지만(comm_active), 이걸 안 주면 config 덤프가 두 팔 모두
    #   use_communication=True 로 찍혀 나중에 로그만 보고 어느 런이 OFF 였는지 구분이 안 된다.
    if [ "$arm" = "OFF" ]; then export VESSEL_USE_COMM=0; else export VESSEL_USE_COMM=1; fi
    "$PY" -u "$HERE/vessel_gym_train.py" \
      --arm "$arm" --comm_on_at "${VESSEL_COMM_ON_AT:-0}" --steps "$steps" \
      --envs 128 --vessels 16 --rollout 32 --ring 1.0 --crossing 2 --max_partners 4 --seed "$s" --ckpt_every "${VESSEL_CKPT_EVERY:-2}" \
      --save "$CK/${nm}_s$s.pt" --csv "$OUT/${nm}_s$s.csv" \
      > "$OUT/${nm}_s$s.log" 2>&1
    echo "${nm}_s$s rc=$?" >> "$OUT/_status_train.txt"
  ) &
}

# ── 평가 한 런 ──────────────────────────────────────────────────────────────
eval_one() {
  local nm=$1 arm=$2 dim=$3 s=$4
  local gpu=$(( GPU_I % NGPU )); GPU_I=$(( GPU_I + 1 ))
  [ -f "$CK/${nm}_s$s.pt" ] || { echo "  건너뜀(체크포인트 없음): ${nm}_s$s"; return; }
  throttle
  (
    common_env
    export CUDA_VISIBLE_DEVICES=$gpu
    export OMP_NUM_THREADS=2
    export VESSEL_MSG_DIM=$dim
    # 집계 방식·중앙critic 등은 eval_ckpt 가 체크포인트의 cfg_snapshot 에서 복원한다(2026-09-05).
    # 그래도 학습과 같은 env 를 주는 편이 안전하다 — 구 체크포인트엔 스냅샷이 없다.
    "$PY" -u "$HERE/eval_ckpt.py" \
      --ckpt "$CK/${nm}_s$s.pt" --arm "$arm" \
      --envs 256 --eval_decisions 10000 --burnin 2400 \
      > "$OUT/eval_${nm}_s$s.txt" 2>&1
    echo "eval_${nm}_s$s rc=$?" >> "$OUT/_status_eval.txt"
  ) &
}

case "$MODE" in
  smoke)
    # 코드가 돌아가는지만 본다. 결과 해석 금지 — 4만 스텝은 수렴이 아니다.
    preflight
    : > "$OUT/_status_train.txt"
    train_one smoke_off OFF 6 43 40000
    train_one smoke_on6 ON  6 43 40000
    wait
    echo "스모크 완료 — $OUT/_status_train.txt 의 rc 가 전부 0 이어야 함"
    cat "$OUT/_status_train.txt"
    ;;

  train)
    # ★VESSEL_TRAIN_ARMS 로 팔 선택 (기본 "off on6 on12"). YUGIOH 6런 = VESSEL_TRAIN_ARMS="off on6".
    #   VESSEL_COMM_ON_AT=9000000 이면 ON 팔이 9M 까지 통신 없이 돌다가 켬(커리큘럼, ABLATION_PLAN §3 B 팔).
    #   그때 VESSEL_CKPT_EVERY=1 로 줘야 .step9M.pt(= 통신 OFF 모델, §4) 가 남는다. OFF 팔엔 comm_on_at 무의미.
    #   학습 중 통신 텔레메트리를 보려면 VESSEL_COMM_TELEMETRY=1 VESSEL_COMM_TELEMETRY_EVERY=5 를 같이 줄 것(ON 팔만 *_comm.csv).
    preflight
    : > "$OUT/_status_train.txt"
    for s in $SEEDS; do
      for arm in ${VESSEL_TRAIN_ARMS:-off on6 on12}; do
        case "$arm" in
          off)  train_one off  OFF 6  "$s" 16056320 ;;
          on6)  train_one on6  ON  6  "$s" 16056320 ;;
          on12) train_one on12 ON  12 "$s" 16056320 ;;
          *) echo "모르는 팔: $arm (off|on6|on12)"; exit 1 ;;
        esac
      done
    done
    wait
    echo "학습 완료"
    cat "$OUT/_status_train.txt"
    ;;

  eval)
    preflight
    : > "$OUT/_status_eval.txt"
    for s in $SEEDS; do
      eval_one off  OFF 6  "$s"
      eval_one on6  ON  6  "$s"
      eval_one on12 ON  12 "$s"
    done
    wait
    echo "평가 완료 — 결과는 $OUT/eval_*.txt"
    cat "$OUT/_status_eval.txt"
    ;;

  random)
    # ★난수 메시지 대조군. 2026-09-04 배치에는 없던 팔이라 기본 재현 대상이 아니다.
    #   통신 ON 이 OFF 를 이겼을 때 그 이득이 메시지 *내용* 때문인지, 메시지 경로가 붙으며
    #   늘어난 파라미터·gradient 경로 때문인지 가른다.
    #   ⚠️난수 스케일을 비교 대상 팔의 실측 others_msg 표준편차에 맞춰야 공정하다.
    #     diag_ckpt.py 가 찍는 msg_sd(텔레메트리 정의)를 읽어 VESSEL_MSG_RANDOM_SD 로 줄 것. (구 _diag_msg_channel.py(→_archive, 현행 diag_ckpt.py) 는 _archive)
    preflight
    : > "$OUT/_status_train.txt"
    for s in $SEEDS; do
      train_one rand RANDOM 6 "$s" 16056320
    done
    wait
    echo "난수 대조군 학습 완료"
    cat "$OUT/_status_train.txt"
    ;;

  diag)
    # ★2026-09-10: 체크포인트 진단 단일 진입점(diag_ckpt.py). 설정은 스냅샷에서, 조우율 게이트 통과 못 하면 숫자 안 냄.
    #   사용: VESSEL_DIAG_CKPTS="a.pt b.pt" bash run_repro.sh diag   (CK 아래 상대경로)
    #   추가 인자: VESSEL_DIAG_ARGS="--burn 1000 --collect 900 --envs 32"
    preflight
    # ⚠️common_env 가 COMM_RANGE=300(YUGIOH) 을 export 하는데 체크포인트가 다른 값(예: 2026-09-04 배치 = 200)으로
    #   학습됐으면 restore_policy 가 중단한다. 그때 VESSEL_DIAG_COMM_RANGE=200 으로 주면 여기서 덮어씀.
    #   값을 모르면 `python ckpt_io.py <ckpt>` 가 스냅샷의 comm_range 를 찍어준다.
    [ -n "${VESSEL_DIAG_COMM_RANGE:-}" ] && export VESSEL_COMM_RANGE="$VESSEL_DIAG_COMM_RANGE"
    : > "$OUT/_status_diag.txt"
    for c in ${VESSEL_DIAG_CKPTS:?VESSEL_DIAG_CKPTS 를 줄 것}; do
      throttle
      n="$(basename "$c" .pt)"
      (
        set +e
        VESSEL_CKPT_DIR="$CK" "$PY" -u "$HERE/diag_ckpt.py" --ckpt "$c" --device "cuda:$(( GPU_I % NGPU ))" \
          --out "$OUT/diag_${n}.json" ${VESSEL_DIAG_ARGS:-} > "$OUT/diag_${n}.txt" 2>&1
        echo "$n rc=$?" >> "$OUT/_status_diag.txt"
      ) &
      GPU_I=$(( GPU_I + 1 ))
    done
    wait
    echo "진단 완료 — $OUT/diag_*.json (rc≠0 은 게이트 실패=숫자 없음)"
    cat "$OUT/_status_diag.txt"
    ;;

  *)
    echo "알 수 없는 모드: $MODE  (smoke | train | eval | random | diag)"
    exit 2
    ;;
esac
