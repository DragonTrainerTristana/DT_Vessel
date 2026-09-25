# ─────────────────────────────────────────────────────────────────────────────
# preflight 검사 공용 구현 (2026-09-26). run_repro.sh preflight 와 smoke_mac.sh 가 source 한다.
#
# 왜 — 예전엔 모드(smoke·train·eval·random·ablate·traj·diag)마다 검사 7종을 *하나씩 차례로* 다시 돌렸다
#   (골든만 3~4분). 같은 코드로 모드를 5번 돌리면 20분 넘게 같은 검사를 반복했고, 그걸 피하려고
#   사람이 VESSEL_SKIP_GOLDEN=1 을 붙였는데 이건 코드가 바뀌어도 건너뛰는 구멍이었다.
# 지금 —
#   ① 검사 7종을 동시에 돌린다(서로 독립 프로세스. 골든은 케이스마다 1스레드라 결과 바이트 불변).
#   ② 통과하면 '지문'(git HEAD + 미커밋 변경 + 새 .py/.sh + python/torch/플랫폼 + 실험 env)으로 도장을 남기고,
#      같은 지문이면 다음 모드에서 건너뛴다. 코드·env 가 한 글자라도 다르면 다시 돈다 = '커밋당 1회' 규칙의 자동판.
#   강제로 다시: VESSEL_FORCE_PREFLIGHT=1 · 캐시 위치: VESSEL_PREFLIGHT_CACHE(기본 $HOME/.cache/vessel_preflight)
#   VESSEL_SKIP_GOLDEN=1 은 예전 뜻 그대로(미러만 돌림) — 지문에 포함되므로 skip 도장이 전체 도장으로 쓰이지 않음.
# 필요 변수: PY, HERE(Python/ 경로), OUT. 함수 common_env(common_env.sh).
# ─────────────────────────────────────────────────────────────────────────────

# 지문에서 뺄 env — 실행 관리용(검사 결과와 무관). 이게 들어가면 smoke/train/eval 사이에 캐시가 안 맞는다.
VESSEL_FP_IGNORE='^VESSEL_(SEEDS|OUT_DIR|CKPT_DIR|TRAIN_ARMS|JOBS|NGPU|SKIP_GOLDEN|FORCE_PREFLIGHT|PREFLIGHT_CACHE|DIAG_[A-Z_]*|ALLOW_UNBRANCHED|REQUIRE_TRUNK|BRANCH_WARMUP|CKPT_EVERY|GPU_PICK|RUN_PREFIX|COMM_TELEMETRY|COMM_TELEMETRY_EVERY|MSG_RANDOM_SD|TIMING|EVAL_ENVS|EVAL_DEC|VRAM_MARGIN)='

_vessel_hash() { "$PY" -c "import sys,hashlib; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest()[:20])"; }

# 지문 출력(비면 캐시 안 씀: git 이 아니거나 실패). 인자: with_ppo(0|1)
vessel_preflight_fp() {
  local with_ppo=${1:-1} root
  root=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null) || { echo ""; return 0; }
  {
    echo "ppo=$with_ppo skip_golden=${VESSEL_SKIP_GOLDEN:-0}"
    git -C "$root" rev-parse HEAD
    git -C "$root" diff HEAD --binary                       # 미커밋 변경(추적 파일)
    # 새로 만든(추적 안 되는) 코드 파일 — 내용 해시까지
    git -C "$root" ls-files --others --exclude-standard -- '*.py' '*.sh' '*.json' | while IFS= read -r f; do
      echo "$f $(git -C "$root" hash-object "$f" 2>/dev/null)"
    done
    "$PY" -c "import sys, platform, torch; print(sys.executable, sys.version, torch.__version__, platform.platform())" 2>/dev/null
    ( common_env; env | grep '^VESSEL_' | grep -Ev "$VESSEL_FP_IGNORE" | sort )
  } | _vessel_hash
}

# 검사 동시 실행. 인자: with_ppo(0|1). 결과 파일은 $OUT/_*.txt(예전과 같은 이름). 전부 통과면 0.
vessel_run_checks() {
  local with_ppo=${1:-1}
  local -a names=() files=() needs=() pids=()
  local t0; t0=$(date +%s)
  _vc_launch() {   # 이름 결과파일 'ALL PASS 필요'(1|0) 명령...
    local nm=$1 f=$2 need=$3; shift 3
    ( "$@" > "$f" 2>&1 ) &
    pids+=("$!"); names+=("$nm"); files+=("$f"); needs+=("$need")
  }
  common_env
  # 골든은 케이스 5개를 동시에(각 1스레드). 나머지 검사는 스레드를 2개로 묶어 CPU 과점을 막는다.
  local _o=OMP_NUM_THREADS=2
  [ "$with_ppo" = "1" ] && _vc_launch "PPO 미러" "$OUT/_verify_ppo.txt" 1 env $_o "$PY" -u "$HERE/verify/_verify_ppo_mirror.py"
  _vc_launch "통신 미러" "$OUT/_verify_comm.txt" 1 env $_o "$PY" -u "$HERE/verify/_verify_comm_mirror.py"
  if [ "${VESSEL_SKIP_GOLDEN:-0}" != "1" ]; then
    _vc_launch "골든 비트동일" "$OUT/_golden.txt" 1 env -u VESSEL_STATE_RECON_COEF -u VESSEL_CENTRAL_CRITIC -u VESSEL_USE_ATTENTION \
      "$PY" -u "$HERE/verify/test_golden.py" --check --jobs 5
    _vc_launch "vessel_gym 충실도" "$OUT/_fidelity.txt" 0 env $_o "$PY" -u "$HERE/verify/test_vessel_gym_fidelity.py"
    _vc_launch "동역학 프로필" "$OUT/_dyn_profile.txt" 1 env $_o "$PY" -u "$HERE/verify/test_dyn_profile.py"
    _vc_launch "sim 스냅샷" "$OUT/_sim_snapshot.txt" 1 env $_o "$PY" -u "$HERE/verify/test_sim_snapshot.py"
    _vc_launch "COMM_EXT" "$OUT/_comm_ext.txt" 1 env $_o "$PY" -u "$HERE/verify/test_comm_ext.py"
  else
    echo "  (VESSEL_SKIP_GOLDEN=1 — 골든·충실도·동역학·sim·COMM_EXT 건너뜀, 미러만)"
  fi
  local i rc bad=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; rc=$?
    if [ "$rc" -ne 0 ] || { [ "${needs[$i]}" = "1" ] && ! grep -q "ALL PASS" "${files[$i]}"; }; then
      echo "  ★FAIL ${names[$i]} (rc=$rc) — ${files[$i]} 확인"; bad=1
    else
      echo "  PASS  ${names[$i]}"
    fi
  done
  echo "  검사 ${#pids[@]}종 동시 실행 $(( $(date +%s) - t0 ))s"
  return $bad
}

# 캐시 포함 진입점. 인자: with_ppo(0|1) 이름표. 통과(또는 캐시 적중)면 0, 실패면 1.
vessel_preflight_cached() {
  local with_ppo=${1:-1} label=${2:-preflight}
  local cache="${VESSEL_PREFLIGHT_CACHE:-$HOME/.cache/vessel_preflight}" fp=""
  fp=$(vessel_preflight_fp "$with_ppo")
  if [ -n "$fp" ] && [ "${VESSEL_FORCE_PREFLIGHT:-0}" != "1" ] && [ -f "$cache/$fp.ok" ]; then
    echo "[$label] 캐시 적중 — 같은 코드·env 로 이미 ALL PASS ($(head -1 "$cache/$fp.ok")). 건너뜀"
    echo "  (다시 돌리려면 VESSEL_FORCE_PREFLIGHT=1 · 지문 $fp)"
    return 0
  fi
  echo "[$label] 검사 동시 실행$([ -n "$fp" ] && echo " (지문 $fp)" || echo " (git 아님 — 캐시 안 씀)")"
  vessel_run_checks "$with_ppo" || return 1
  if [ -n "$fp" ]; then
    mkdir -p "$cache" && {
      date '+%Y-%m-%d %H:%M:%S'
      git -C "$HERE" log -1 --format='%h %s' 2>/dev/null
      echo "ppo=$with_ppo skip_golden=${VESSEL_SKIP_GOLDEN:-0} out=$OUT"
    } > "$cache/$fp.ok"
  fi
  return 0
}
