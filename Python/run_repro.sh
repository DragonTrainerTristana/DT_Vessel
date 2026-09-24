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
#   ★2026-09-15 분기 규약 (사용자 지시): 모든 팔은 시드×dim 마다 OFF 로 9,043,968 결정까지 *한 번* 학습한
#     trunk 파일에서 갈라진다 → ON/OFF 가 통신 켜기 전까지 같은 모델임을 구조로 보장.
#     (09-10 배치처럼 시드만 맞춰 따로 처음부터 돌리면 GPU 비결정성으로 2번째 update 부터 갈라짐)
#   ⚠️그 배치에서 off_s45 는 학습에 실패해 결과에서 제외됐다(최종보상 1.06 vs 형제 1.53·1.59).
#     제외는 통신에 *불리한* 방향이라 보수적 선택이다. 자세한 건 runs/m2_ablation/COMM_PLAN.md §4-B.
#
# 쓰는 법:
#   bash run_repro.sh smoke     # trunk 1 update → 갈래 1 update → 분기 검사 (몇 분)
#   bash run_repro.sh train     # 본 배치: trunk(9,043,968) → 갈래 16.06M, 팔 = VESSEL_TRAIN_ARMS
#   bash run_repro.sh eval      # 분기 검사 통과한 체크포인트 평가
#   bash run_repro.sh random    # 난수 메시지 대조군 (같은 trunk 에서 분기, 기본 팔에 없음 — 아래 설명)
#
# 환경변수로 바꿀 수 있는 것 (전부 기본값 있음):
#   VESSEL_CKPT_DIR  체크포인트 저장 위치. 기본 $HOME/VESSEL_checkpoints/<날짜>_repro
#                    ⚠️Dropbox 같은 동기화 폴더에 두지 말 것 — 배치 하나가 GB 단위다.
#   VESSEL_PY        파이썬 실행 파일. 기본 python
#   VESSEL_OUT_DIR   CSV·로그 저장 위치. 기본 이 파일 옆 _repro_out/
#   VESSEL_SEEDS     시드 목록. 기본 "43 44 45"
#   VESSEL_NGPU      쓸 GPU 수. 기본은 torch 로 자동 감지(0장이면 1로 두고 CPU)
#   VESSEL_JOBS      동시 실행 프로세스 수. 기본 = NGPU × 2 (VRAM 프로세스당 ~5.2GB 기준)
#   VESSEL_TRAIN_ARMS  학습 팔. 기본 "off on6 on12" — 통신 팔은 같은 dim 의 OFF 짝이 있어야 시작(on12 ↔ off12, on2/off2(dim 2 짝))
#   VESSEL_BRANCH_WARMUP  갈래 재개 직후 통신 OFF 로 굴리는 에이전트당 결정 수. 기본 1200 (모든 갈래 동일)
#   VESSEL_ALLOW_UNBRANCHED=1  eval 의 분기 검사 FAIL 을 무시 — 규약 이전 옛 배치 재평가 전용, 짝 비교 금지
#   VESSEL_CROSSING  목표 배정. 기본 2 = 대척(agile 배치 재현). 0 = 스폰에서 MIN_GOAL_DIST(400 m) 이상 떨어진 목표 중 무작위
#                    (2026-09-23: open-sea 에서 대척은 16척 경로가 전부 원점 56 m 안을 지나 중앙 난투 → imo 파일럿은 0 으로).
#                    평가는 스냅샷 crossing 을 자동으로 따름. 같은 trunk 묶음 안 crossing 일치는 check_branch 가 검사.
# ─────────────────────────────────────────────────────────────────────────────
set -u
export PYTHONIOENCODING=utf-8   # ★Windows cp949 콘솔로 리다이렉트할 때 한글·기호 print 가 UnicodeEncodeError 로 죽는 것 방지 (2026-09-10)

MODE="${1:-smoke}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${VESSEL_PY:-python}"
OUT="${VESSEL_OUT_DIR:-$HERE/_repro_out}"
CK="${VESSEL_CKPT_DIR:-$HOME/VESSEL_checkpoints/$(date +%Y-%m-%d)_repro}"
SEEDS="${VESSEL_SEEDS:-43 44 45}"

# ★2026-09-15 분기 규약: ON/OFF 는 통신 켜는 지점까지 *같은 체크포인트 파일*(trunk)을 쓴다.
#   update 당 결정 = envs 128 × vessels 16 × rollout 32 — 아래 train_one 의 인자와 같이 움직일 것.
UPDATE_DEC=$(( 128 * 16 * 32 ))                # 65,536
BRANCH_AT=$(( UPDATE_DEC * 138 ))              # 9,043,968 = 138 update 끝 (09-10 배치 comm_on_at 9M 과 같은 update 경계)
TOTAL_STEPS=16056320                           # 245 update (= config.YUGIOH_ARGS --steps)
BR_WARMUP="${VESSEL_BRANCH_WARMUP:-1200}"
if [ -n "${VESSEL_COMM_ON_AT:-}" ]; then
  echo "VESSEL_COMM_ON_AT 는 폐기됨 (2026-09-15 분기 규약). 분기점은 BRANCH_AT=$BRANCH_AT 고정 — 이 환경변수를 지울 것."
  exit 2
fi

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
echo "  분기점      : $BRANCH_AT 결정 (trunk → 갈래, 워밍업 $BR_WARMUP)"
echo "  프로필      : dyn=${VESSEL_DYN_PROFILE:-agile} obstacles=${VESSEL_OBSTACLES:-grid3x3} crossing=${VESSEL_CROSSING:-2}"
echo

# ── 학습·평가 공통 설정 = YUGIOH (config.py 끝 `YUGIOH` 표와 1:1) ──────────────
# ★2026-09-24: common_env() 정본은 common_env.sh 한 곳이다 — smoke_mac.sh 도 같은 파일을 source 한다
#   (예전엔 사본 두 벌이라 export 하나 고치려면 두 번 고쳐야 했음). preflight 가 그 값과 config 기본값을 대조한다.
source "$HERE/common_env.sh"

# ── 사전 검증: 미러가 깨졌으면 돌리지 말 것 ─────────────────────────────────
preflight() {
  # ★2026-09-15: 인터프리터 sanity. torch 없는 python 을 잡으면 아래 검사가 전부 엉뚱한 이유로
  #   실패한다. 특히 드리프트 검사는 자식 stdout 이 비어 "config 기본값이 다름" 으로 오보했었다.
  "$PY" -c "import torch" >/dev/null 2>&1 || {
    echo "preflight 실패: '$PY' 에서 torch 를 import 하지 못함."
    echo "  → VESSEL_PY 로 torch 가 설치된 인터프리터를 지정할 것."
    "$PY" -c "import torch" 2>&1 | tail -3 | sed 's/^/  /'
    exit 1
  }
  # ★YUGIOH 드리프트 검사: common_env 의 export 값 == config.py 기본값 (누가 config 기본값만 바꾸면 여기서 잡힘)
  #   DYN_PROFILE·OBSTACLES·RADAR_RANGE 는 names 에 없다 — common_env 가 바깥 override 를 보존하는
  #   의도된 실험 축이라 여기서 잡으면 imo·용량반응 배치가 시작조차 못 한다(스냅샷·check_branch 가 대신 강제).
  ( common_env; "$PY" - <<'PYCHK'
import os, json, subprocess, sys
env = {k: v for k, v in os.environ.items() if not k.startswith('VESSEL_')}
code = "import json, config as c; print(json.dumps({k: str(getattr(c, k)) for k in %r}))"
names = ['USE_ATTENTION','CENTRAL_CRITIC','STATE_RECON_COEF','MOE_SHARED','SHARED_ENCODER','RADAR_ACT','RADAR_HEAD',
         'MSG_TOKEN_GAIN','CLIP_PER_MODULE','MSG_L2_COEF','COMM_RANGE','MSG_LN','POS_GROUND','MOE_WIDTH','USE_MOE',
         'RADAR_BOTTLENECK_CH','MAX_COMM_PARTNERS','COLREGS_MODE','COLREGS_SIM_COEF','INTENT_K',
         'THREAT_COEF','GOAL_COMM_COEF','INTENT_COEF','ROLE_COMM_COEF','COMM_CONSUMER_COEF','RECON_EMA_FLOOR',
         'AGG_MODE','MSG_GAIN','TIMEOUT_BOOTSTRAP','MSG_GATE_APPLY']
def _dump(e=None):
    # ★2026-09-15: returncode/stderr 를 안 보면 환경 문제(torch 없음·config import 에러)가
    #   빈 stdout -> IndexError 로 터져 "드리프트" 로 오보된다. exit 2 = 환경 문제(드리프트 아님).
    r = subprocess.run([sys.executable, '-c', code % names], env=e, capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    out = (r.stdout or '').strip()
    if r.returncode != 0 or not out:
        print('  YUGIOH 드리프트: ★검사불가 — config import 실패 (rc=%d)' % r.returncode)
        for ln in ((r.stderr or '').strip().splitlines() or ['(stderr 없음)'])[-3:]:
            print('   ', ln)
        sys.exit(2)
    return json.loads(out.splitlines()[-1])
a = _dump(env)
b = _dump()
bad = [k for k in names if a[k] != b[k]]
print('  YUGIOH 드리프트:', 'PASS (common_env == config 기본값)' if not bad else f'★FAIL {bad}')
sys.exit(1 if bad else 0)
PYCHK
  ); _drift_rc=$?
  if [ "$_drift_rc" -eq 2 ]; then
    echo "preflight 실패: 파이썬 환경 문제 — 드리프트 검사가 config 를 import 하지 못함(위 stderr 참고)."
    echo "  → 드리프트 판정이 아님. VESSEL_PY 확인: $PY"
    exit 1
  elif [ "$_drift_rc" -ne 0 ]; then
    echo "preflight 실패: common_env 와 config.py 기본값이 다름 — config.py 끝 YUGIOH 표를 볼 것"
    exit 1
  fi
  echo "[preflight] PPO·통신 미러 검증"
  common_env
  "$PY" -u "$HERE/verify/_verify_ppo_mirror.py"  > "$OUT/_verify_ppo.txt"  2>&1 || { echo "  PPO 미러 FAIL — $OUT/_verify_ppo.txt 확인"; exit 1; }
  "$PY" -u "$HERE/verify/_verify_comm_mirror.py" > "$OUT/_verify_comm.txt" 2>&1 || { echo "  통신 미러 FAIL — $OUT/_verify_comm.txt 확인"; exit 1; }
  grep -q "ALL PASS" "$OUT/_verify_ppo.txt"  || { echo "  PPO 미러가 ALL PASS 가 아님"; exit 1; }
  grep -q "ALL PASS" "$OUT/_verify_comm.txt" || { echo "  통신 미러가 ALL PASS 가 아님"; exit 1; }
  echo "  둘 다 ALL PASS"
  # ★2026-09-10: 기본값 비트동일 골든 + vessel_gym 충실도. VESSEL_SKIP_GOLDEN=1 로 건너뜀(수 분 걸림).
  if [ "${VESSEL_SKIP_GOLDEN:-0}" != "1" ]; then
    echo "[preflight] 골든 비트동일 검사"
    ( env -u VESSEL_STATE_RECON_COEF -u VESSEL_CENTRAL_CRITIC -u VESSEL_USE_ATTENTION \
        "$PY" -u "$HERE/verify/test_golden.py" --check ) > "$OUT/_golden.txt" 2>&1 \
      || { echo "  골든 FAIL — $OUT/_golden.txt 확인 (코드가 기본값 결과를 바꿨음)"; exit 1; }
    grep -q "ALL PASS" "$OUT/_golden.txt" || { echo "  골든이 ALL PASS 가 아님"; exit 1; }
    echo "  골든 ALL PASS"
    "$PY" -u "$HERE/verify/test_vessel_gym_fidelity.py" > "$OUT/_fidelity.txt" 2>&1 \
      || { echo "  vessel_gym 충실도 FAIL — $OUT/_fidelity.txt 확인"; exit 1; }
    echo "  충실도 PASS"
    # ★2026-09-21: 동역학 프로필·시나리오 게이트. 지금 env(agile/grid3x3 든 imo/none 이든) 그대로 돌린다.
    "$PY" -u "$HERE/verify/test_dyn_profile.py" > "$OUT/_dyn_profile.txt" 2>&1 \
      || { echo "  동역학 프로필 FAIL — $OUT/_dyn_profile.txt 확인"; exit 1; }
    grep -q "ALL PASS" "$OUT/_dyn_profile.txt" || { echo "  동역학 프로필이 ALL PASS 가 아님"; exit 1; }
    echo "  동역학 프로필 ALL PASS"
    # ★2026-09-23: 스냅샷 sim 상수(보상 계수·게이트·COLREGS_MODE·에피소드 길이) 기록·대조·복원 게이트.
    "$PY" -u "$HERE/verify/test_sim_snapshot.py" > "$OUT/_sim_snapshot.txt" 2>&1 \
      || { echo "  sim 스냅샷 FAIL — $OUT/_sim_snapshot.txt 확인"; exit 1; }
    grep -q "ALL PASS" "$OUT/_sim_snapshot.txt" || { echo "  sim 스냅샷이 ALL PASS 가 아님"; exit 1; }
    echo "  sim 스냅샷 ALL PASS"
  fi
  echo
}

# 동시 실행 수 제한 (GPU 배정은 아래 pick_gpu)
GPU_I=0
# ★2026-09-15: eval 모드가 체크포인트를 전부 건너뛰고도 exit 0 "평가 완료" 로 보고했었다.
EVAL_N=0        # 실제로 띄운 평가 수
EVAL_MISS=""    # 못 찾은 체크포인트 이름
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 2; done; }

# ── GPU 배정 (2026-09-22) ─────────────────────────────────────────────────
# 예전: GPU_I % NGPU 라운드로빈. 모드마다 0 부터 세서 작업 3개(trunk·random·diag)면 GPU 3 이 항상 놀았고,
#   남의 작업으로 VRAM 이 줄어든 GPU 도 가리지 않았다.
# 지금: 우리 작업 수 최소 GPU → 같으면 남은 VRAM 최대. VESSEL_GPU_PICK=rr 이면 예전 방식.
#   nvidia-smi 가 없거나 실패하면 free=0 → 작업 수만으로 고른다.
#   CUDA_DEVICE_ORDER=PCI_BUS_ID 로 CUDA 번호 = nvidia-smi 번호.
#   결과 영향 없음 — 어느 물리 GPU 에 붙느냐만 바뀐다(갈래 뒤 GPU 비결정성은 §8-1 에서 이미 감수).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
declare -a GPU_PIDS=()
pick_gpu() {
  if [ "${VESSEL_GPU_PICK:-free}" = "rr" ]; then echo $(( GPU_I % NGPU )); return; fi
  local g n f p best=0 best_n=999999 best_f=-1
  local -a free=()
  mapfile -t free < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  for (( g=0; g<NGPU; g++ )); do
    n=0; for p in ${GPU_PIDS[$g]:-}; do kill -0 "$p" 2>/dev/null && n=$(( n + 1 )); done
    f=${free[$g]:-0}; f=${f//[^0-9]/}; f=${f:-0}
    if [ "$n" -lt "$best_n" ] || { [ "$n" -eq "$best_n" ] && [ "$f" -gt "$best_f" ]; }; then
      best=$g; best_n=$n; best_f=$f
    fi
  done
  echo "$best"
}

# ── 학습 한 런 ──────────────────────────────────────────────────────────────
# 인자: 이름 arm msg_dim seed steps [trunk.pt branch_at]
#   trunk 를 주면 분기 갈래: trunk 곡선 CSV 를 복사해 이어 쓰고 --resume 으로 branch_at 부터 학습한다.
#   ★2026-09-15: ON/OFF 비교용 런은 반드시 branch_batch(아래)를 거친다. 직접 부르는 건 trunk·옛 모드뿐.
train_one() {
  local nm=$1 arm=$2 dim=$3 s=$4 steps=$5 trunk=${6:-} br_at=${7:-0}
  throttle
  local gpu; gpu=$(pick_gpu); GPU_I=$(( GPU_I + 1 ))
  echo "  ${nm}_s$s → GPU $gpu"
  (
    common_env
    export CUDA_VISIBLE_DEVICES=$gpu
    export OMP_NUM_THREADS=2
    export VESSEL_MSG_DIM=$dim
    # ★arm=OFF 런은 VESSEL_USE_COMM=0 을 명시한다.
    #   동작은 어차피 --arm 이 정하지만(comm_active), 이걸 안 주면 config 덤프가 두 팔 모두
    #   use_communication=True 로 찍혀 나중에 로그만 보고 어느 런이 OFF 였는지 구분이 안 된다.
    if [ "$arm" = "OFF" ]; then export VESSEL_USE_COMM=0; else export VESSEL_USE_COMM=1; fi
    local run="${nm}_s$s"
    local extra=(--comm_on_at 0)
    if [ -n "$trunk" ]; then
      # 같은 이름의 옛 CSV 가 남아 있으면 그 뒤에 이어 붙으므로 지우고, trunk 곡선(0~branch_at)으로 시작한다.
      local tstem="$OUT/$(basename "${trunk%.pt}")"
      rm -f "$OUT/$run.csv" "$OUT/${run}_aux.csv" "$OUT/${run}_comm.csv"
      [ -f "$tstem.csv" ] && cp "$tstem.csv" "$OUT/$run.csv"
      [ -f "${tstem}_aux.csv" ] && cp "${tstem}_aux.csv" "$OUT/${run}_aux.csv"
      extra=(--resume "$trunk" --resume_at "$br_at" --comm_on_at "$br_at" --resume_warmup "$BR_WARMUP")
    fi
    "$PY" -u "$HERE/vessel_gym_train.py" \
      --arm "$arm" --steps "$steps" "${extra[@]}" \
      --envs 128 --vessels 16 --rollout 32 --ring 1.0 --crossing "${VESSEL_CROSSING:-2}" --max_partners 4 --seed "$s" --ckpt_every "${VESSEL_CKPT_EVERY:-2}" \
      --save "$CK/$run.pt" --csv "$OUT/$run.csv" \
      > "$OUT/$run.log" 2>&1
    echo "$run rc=$?" >> "$OUT/_status_train.txt"
  ) &
  GPU_PIDS[$gpu]="${GPU_PIDS[$gpu]:-} $!"
}

# ── 평가 한 런 ──────────────────────────────────────────────────────────────
eval_one() {
  local nm=$1 arm=$2 dim=$3 s=$4
  [ -f "$CK/${nm}_s$s.pt" ] || { echo "  건너뜀(체크포인트 없음): ${nm}_s$s"; EVAL_MISS="$EVAL_MISS ${nm}_s$s.pt"; return; }
  throttle
  local gpu; gpu=$(pick_gpu); GPU_I=$(( GPU_I + 1 ))
  echo "  eval_${nm}_s$s → GPU $gpu"
  EVAL_N=$(( EVAL_N + 1 ))
  (
    common_env
    export CUDA_VISIBLE_DEVICES=$gpu
    export OMP_NUM_THREADS=2
    export VESSEL_MSG_DIM=$dim
    # 집계 방식·중앙critic 등은 eval_ckpt 가 체크포인트의 cfg_snapshot 에서 복원한다(2026-09-05).
    # 그래도 학습과 같은 env 를 주는 편이 안전하다 — 구 체크포인트엔 스냅샷이 없다.
    "$PY" -u "$HERE/eval/eval_ckpt.py" \
      --ckpt "$CK/${nm}_s$s.pt" --arm "$arm" \
      --envs 256 --eval_decisions 10000 --burnin 2400 \
      > "$OUT/eval_${nm}_s$s.txt" 2>&1
    echo "eval_${nm}_s$s rc=$?" >> "$OUT/_status_eval.txt"
  ) &
  GPU_PIDS[$gpu]="${GPU_PIDS[$gpu]:-} $!"
}

# ── 팔 이름 → "ARM DIM" ─────────────────────────────────────────────────────
arm_spec() {
  case "$1" in
    off)   echo "OFF 6" ;;
    on6)   echo "ON 6" ;;
    on12)  echo "ON 12" ;;
    off12) echo "OFF 12" ;;     # on12 의 짝 — dim12 trunk 에서 분기한 OFF
    on2)   echo "ON 2" ;;
    off2)  echo "OFF 2" ;;      # on2 의 짝 — dim2 trunk 에서 분기한 OFF
    rand)  echo "RANDOM 6" ;;   # 난수 메시지 대조군 (random 모드)
    *) return 1 ;;
  esac
}
off_name() { if [ "$1" = 6 ]; then echo off; else echo "off$1"; fi; }

# ── 분기 검사 (verify/check_branch.py) — 결과 $OUT/_branch_check.txt, ALL PASS 아니면 1 ──
check_branch() {
  "$PY" -u "$HERE/verify/check_branch.py" --trunk_dir "$CK" --csv_dir "$OUT" "$@" > "$OUT/_branch_check.txt" 2>&1
  local rc=$?
  tail -n 25 "$OUT/_branch_check.txt" | sed 's/^/  /'
  [ "$rc" -eq 0 ] && grep -q "ALL PASS" "$OUT/_branch_check.txt"
}

# ── trunk → 갈래 배치 (★2026-09-15 분기 규약, 사용자 지시) ──────────────────
# 인자: "팔 목록" branch_at 총결정 [이름접두]
#   ① 짝 검사: 통신 팔(ON·RANDOM)이 있는 dim 은 같은 dim 의 OFF 갈래가 목록에 있거나 $CK 에 이미 있어야 함
#   ② trunk  : 시드×dim 마다 OFF 로 branch_at 까지 1회 → $CK/<접두>trunk_d<dim>_s<seed>.pt (있으면 재사용)
#   ③ 갈래   : 팔마다 그 trunk 에서 --resume (워밍업은 통신 OFF 로 굴려 모든 갈래 동일)
#   ④ 검증   : check_branch.py — 같은 trunk SHA·seed·dim·branch_at, OFF 짝, 0~branch_at 곡선 글자 일치
#   ⚠️trunk 재사용은 같은 $CK 안에서 같은 코드·설정으로 만든 것만. 학습기가 steps·seed·통신OFF 는 검사하지만
#     보상 계수 같은 설정 차이는 못 잡음.
branch_batch() {
  local arms="$1" br_at=$2 total=$3 pre=${4:-}
  local a spec dim s t dims=""
  for a in $arms; do
    spec=$(arm_spec "$a") || { echo "모르는 팔: $a (off|on6|on12|off12|on2|off2|rand)"; exit 1; }
    dim=${spec#* }
    case " $dims " in *" $dim "*) ;; *) dims="$dims $dim" ;; esac
  done
  for dim in $dims; do
    local has_comm=0 has_off=0
    for a in $arms; do
      spec=$(arm_spec "$a"); [ "${spec#* }" = "$dim" ] || continue
      if [ "${spec% *}" = "OFF" ]; then has_off=1; else has_comm=1; fi
    done
    if [ "$has_comm" = 1 ] && [ "$has_off" = 0 ]; then
      for s in $SEEDS; do
        [ -f "$CK/${pre}$(off_name "$dim")_s$s.pt" ] || {
          echo "분기 규약 위반: dim $dim 통신 팔의 짝 OFF 갈래 '${pre}$(off_name "$dim")_s$s' 가 목록에도 \$CK 에도 없음."
          echo "  → VESSEL_TRAIN_ARMS 에 $(off_name "$dim") 을 넣거나 그 통신 팔을 뺄 것. 짝 없는 통신 팔은 돌리지 않음."
          exit 1; }
      done
    fi
  done
  echo "[branch] 분기점 $br_at · 총 $total 결정 · 워밍업 $BR_WARMUP · dim:$dims · 팔: $arms"
  for s in $SEEDS; do
    for dim in $dims; do
      t="$CK/${pre}trunk_d${dim}_s$s.pt"
      if [ -f "$t" ]; then echo "  trunk 재사용: $(basename "$t")"
      else train_one "${pre}trunk_d$dim" OFF "$dim" "$s" "$br_at"; fi
    done
  done
  wait
  for s in $SEEDS; do
    for dim in $dims; do
      [ -f "$CK/${pre}trunk_d${dim}_s$s.pt" ] || {
        echo "trunk 학습 실패: ${pre}trunk_d${dim}_s$s — $OUT/${pre}trunk_d${dim}_s$s.log 확인. 갈래 안 띄움"; exit 1; }
    done
  done
  for s in $SEEDS; do
    for a in $arms; do
      spec=$(arm_spec "$a")
      train_one "${pre}$a" "${spec% *}" "${spec#* }" "$s" "$total" "$CK/${pre}trunk_d${spec#* }_s$s.pt" "$br_at"
    done
  done
  wait
  local files="" o
  for s in $SEEDS; do
    for a in $arms; do files="$files $CK/${pre}${a}_s$s.pt"; done
    for dim in $dims; do
      o="$CK/${pre}$(off_name "$dim")_s$s.pt"
      case " $files " in *" $o "*) ;; *) [ -f "$o" ] && files="$files $o" ;; esac
    done
  done
  echo "[branch] 분기 검사"
  check_branch $files || { echo "분기 검사 FAIL — $OUT/_branch_check.txt 확인"; exit 1; }
  echo "  분기 검사 ALL PASS"
}

case "$MODE" in
  smoke)
    # 코드가 돌아가는지만 본다. 결과 해석 금지 — 몇 update 는 수렴이 아니다.
    # ★2026-09-15: 분기 경로 전체(trunk 1 update → off·on6 갈래 1 update → 분기 검사)를 통과해야 PASS.
    preflight
    : > "$OUT/_status_train.txt"
    SEEDS=43
    BR_WARMUP=8
    rm -f "$CK"/smoke_trunk_d*_s43.pt   # 스모크는 trunk 학습까지 매번 확인
    branch_batch "off on6" "$UPDATE_DEC" $(( UPDATE_DEC * 2 )) smoke_
    echo "스모크 완료 — $OUT/_status_train.txt 의 rc 가 전부 0 이어야 함"
    cat "$OUT/_status_train.txt"
    ;;

  train)
    # ★2026-09-15 분기 규약 (사용자 지시): trunk(OFF, 9,043,968 결정) → 팔마다 갈래(--resume). branch_batch 참고.
    #   팔 = VESSEL_TRAIN_ARMS (기본 "off on6 on12"). YUGIOH 6런 = VESSEL_TRAIN_ARMS="off on6".
    #   통신 팔은 같은 dim 의 OFF 짝이 있어야 시작함 — on12 는 off12 가 필요(dim12 trunk 는 dim6 과 다른 모델).
    #   9M 통신 OFF 모델 = $CK/trunk_d<dim>_s<seed>.pt (구 .step9M.pt 역할).
    #   학습 중 통신 텔레메트리: VESSEL_COMM_TELEMETRY=1 VESSEL_COMM_TELEMETRY_EVERY=5 (ON 갈래만 *_comm.csv).
    preflight
    : > "$OUT/_status_train.txt"
    branch_batch "${VESSEL_TRAIN_ARMS:-off on6 on12}" "$BRANCH_AT" "$TOTAL_STEPS"
    echo "학습 완료"
    cat "$OUT/_status_train.txt"
    ;;

  eval)
    preflight
    # ★2026-09-15 분기 규약: ON/OFF 짝이 같은 trunk 에서 갈라졌는지 먼저 확인. 아니면 평가 안 함(짝 비교 무효).
    #   규약 이전 옛 배치 재평가만 VESSEL_ALLOW_UNBRANCHED=1 로 우회 — 그 숫자는 ON/OFF 짝 비교에 쓰지 말 것.
    _ev_files=""
    for s in $SEEDS; do
      for nm in off on6 off12 on12 off2 on2 rand; do [ -f "$CK/${nm}_s$s.pt" ] && _ev_files="$_ev_files $CK/${nm}_s$s.pt"; done
    done
    if [ -n "$_ev_files" ]; then
      echo "[eval] 분기 검사"
      if check_branch $_ev_files; then
        echo "  분기 검사 ALL PASS"
      elif [ "${VESSEL_ALLOW_UNBRANCHED:-0}" = "1" ]; then
        echo "  ⚠️분기 검사 FAIL 인데 VESSEL_ALLOW_UNBRANCHED=1 로 진행 — 이 평가는 ON/OFF 짝 비교에 쓰지 말 것"
      else
        echo "평가 중단: 분기 검사 FAIL — $OUT/_branch_check.txt 확인 (규약 이전 옛 배치면 VESSEL_ALLOW_UNBRANCHED=1)"
        exit 1
      fi
    fi
    : > "$OUT/_status_eval.txt"
    for s in $SEEDS; do
      eval_one off  OFF 6  "$s"
      eval_one on6  ON  6  "$s"
      eval_one on12 ON  12 "$s"
      [ -f "$CK/off12_s$s.pt" ] && eval_one off12 OFF 12 "$s"
      [ -f "$CK/on2_s$s.pt" ]  && eval_one on2  ON  2  "$s"
      [ -f "$CK/off2_s$s.pt" ] && eval_one off2 OFF 2  "$s"
      # ★2026-09-22: 난수 대조군도 평가 (스펙 §4 ON > RANDOM 판정). 전에는 random 으로 학습만 하고 평가 목록에 없었다.
      [ -f "$CK/rand_s$s.pt" ] && eval_one rand RANDOM 6 "$s"
    done
    wait
    # ★2026-09-15: 0건이면 실패. 전에는 9건 전부 건너뛰고도 exit 0 "평가 완료" 였다.
    if [ "$EVAL_N" -eq 0 ]; then
      echo
      echo "평가 실패: 체크포인트를 한 건도 못 찾아 0건 평가됨."
      echo "  찾은 곳    : $CK"
      echo "  기대한 이름: {off,on6,on12,off12,on2,off2,rand}_s{$(echo $SEEDS | tr ' ' ',')}.pt"
      echo "  못 찾은 것 :$EVAL_MISS"
      echo "  실제 내용  :"
      if [ -d "$CK" ]; then
        ls -1 "$CK" | sed 's/^/    /'
        [ -n "$(ls -A "$CK" 2>/dev/null)" ] || echo "    (비어 있음)"
      else
        echo "    (디렉터리 없음)"
      fi
      exit 1
    fi
    [ -z "$EVAL_MISS" ] || echo "  ⚠️건너뛴 체크포인트:$EVAL_MISS"
    echo "평가 완료 — $EVAL_N건, 결과는 $OUT/eval_*.txt"
    cat "$OUT/_status_eval.txt"
    # ★2026-09-21: rc≠0 인 평가가 있으면 실패. 전에는 전부 죽어도 "평가 완료" 로 exit 0 이었다.
    _eval_bad=$(grep -cv 'rc=0$' "$OUT/_status_eval.txt")
    if [ "${_eval_bad:-0}" -ne 0 ]; then
      echo
      echo "평가 실패: rc≠0 인 평가 ${_eval_bad}건 — 위 목록에서 rc 를 확인하고 $OUT/eval_*.txt 를 볼 것"
      exit 1
    fi
    ;;

  random)
    # ★난수 메시지 대조군. 2026-09-04 배치에는 없던 팔이라 기본 재현 대상이 아니다.
    #   통신 ON 이 OFF 를 이겼을 때 그 이득이 메시지 *내용* 때문인지, 메시지 경로가 붙으며
    #   늘어난 파라미터·gradient 경로 때문인지 가른다.
    #   ⚠️난수 스케일을 비교 대상 팔의 실측 others_msg 표준편차에 맞춰야 공정하다.
    #     diag_ckpt.py 가 찍는 msg_sd(텔레메트리 정의)를 읽어 VESSEL_MSG_RANDOM_SD 로 줄 것. (구 _diag_msg_channel.py(→_archive, 현행 diag_ckpt.py) 는 _archive)
    #   ★2026-09-15: 난수 팔도 같은 trunk 에서 분기. 짝 OFF 갈래($CK/off_s<seed>.pt)가 있어야 함 → train 먼저.
    preflight
    : > "$OUT/_status_train.txt"
    branch_batch "rand" "$BRANCH_AT" "$TOTAL_STEPS"
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
      gpu=$(pick_gpu)
      n="$(basename "$c" .pt)"
      echo "  diag_${n} → GPU $gpu"
      (
        set +e
        VESSEL_CKPT_DIR="$CK" "$PY" -u "$HERE/eval/diag_ckpt.py" --ckpt "$c" --device "cuda:$gpu" \
          --out "$OUT/diag_${n}.json" ${VESSEL_DIAG_ARGS:-} > "$OUT/diag_${n}.txt" 2>&1
        echo "$n rc=$?" >> "$OUT/_status_diag.txt"
      ) &
      GPU_PIDS[$gpu]="${GPU_PIDS[$gpu]:-} $!"
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
