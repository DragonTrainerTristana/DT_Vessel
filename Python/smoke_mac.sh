#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Mac 검증 스크립트 (2026-09-14)
#
# PPO 미러는 Mac 미지원이라 제외. 최종 판정은 Windows 의 run_repro.sh smoke 로 한다.
#   왜 — Mac 은 torch↔numpy 비호환(`networks.py`의 `torch.as_tensor(numpy 배열)`
#   이 RuntimeError). `run_repro.sh smoke` 를 Mac 에서 그대로 돌리면 preflight 가
#   PPO 미러(`_verify_ppo_mirror.py`)에서 막혀 뒤에 있는 comm 미러·골든·fidelity 를
#   못 본다. 이 스크립트는 그 세 개만 run_repro.sh 와 같은 방식($PY·$HERE·
#   리다이렉트·exit 처리 그대로)으로 이어서 돌린다.
#   아래 common_env() 블록은 `run_repro.sh:70-102` 의 복사본이다. bash 3.2(macOS
#   기본)에서 `source <(...)` 로 다른 스크립트의 함수를 가져오는 방식이 정의를
#   조용히 누락시켜(실측 — 뒤 검사가 그때 config 기본값과 우연히 같아 실패로도
#   안 드러났다) source 재사용이 불가능해 통째로 복사했다.
#   → run_repro.sh 의 common_env export 를 하나라도 고치면 이 사본도 반드시 같이 고칠 것.
#   동기화 확인: diff <(sed -n '70,102p' run_repro.sh) <(sed -n '37,69p' smoke_mac.sh)
#   (빈 출력 = 동일. 두 파일 중 한쪽 줄이 밀리면 sed 범위를 다시 맞출 것.)
#
# 돌리는 것 (이 순서로, run_repro.sh preflight 와 동일한 호출):
#   1. _verify_comm_mirror.py       (통신 집계 rollout=update 미러)
#   2. test_golden.py --check       (기본값 결과 비트동일 골든)
#   3. test_vessel_gym_fidelity.py  (vessel_gym 물리·obs 충실도)
#
# 쓰는 법: bash smoke_mac.sh
# 환경변수(run_repro.sh 와 동일): VESSEL_PY(기본 python) · VESSEL_OUT_DIR(기본 $HERE/_repro_out)
# ─────────────────────────────────────────────────────────────────────────────
set -u
export PYTHONIOENCODING=utf-8

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${VESSEL_PY:-python}"
OUT="${VESSEL_OUT_DIR:-$HERE/_repro_out}"

mkdir -p "$OUT"

# common_env() — run_repro.sh:70-101 을 그대로 복사한 것. 정본은 run_repro.sh.
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

echo "Mac 검증 [smoke_mac] — PPO 미러 제외 (Mac 미지원)"
echo "  python : $PY"
echo "  출력   : $OUT"
echo

common_env

echo "[1/3] 통신 미러 검증"
"$PY" -u "$HERE/_verify_comm_mirror.py" > "$OUT/_verify_comm.txt" 2>&1 || { echo "  통신 미러 FAIL — $OUT/_verify_comm.txt 확인"; exit 1; }
grep -q "ALL PASS" "$OUT/_verify_comm.txt" || { echo "  통신 미러가 ALL PASS 가 아님"; exit 1; }
echo "  통신 미러 ALL PASS"

echo "[2/3] 골든 비트동일 검사"
( cd "$HERE" && env -u VESSEL_STATE_RECON_COEF -u VESSEL_CENTRAL_CRITIC -u VESSEL_USE_ATTENTION \
    "$PY" -u test_golden.py --check ) > "$OUT/_golden.txt" 2>&1 \
  || { echo "  골든 FAIL — $OUT/_golden.txt 확인 (코드가 기본값 결과를 바꿨음)"; exit 1; }
grep -q "ALL PASS" "$OUT/_golden.txt" || { echo "  골든이 ALL PASS 가 아님"; exit 1; }
echo "  골든 ALL PASS"

echo "[3/3] vessel_gym 충실도 검사"
"$PY" -u "$HERE/test_vessel_gym_fidelity.py" > "$OUT/_fidelity.txt" 2>&1 \
  || { echo "  vessel_gym 충실도 FAIL — $OUT/_fidelity.txt 확인"; exit 1; }
echo "  충실도 PASS"

echo
echo "Mac 검증 완료 — 통신 미러·골든·충실도 전부 PASS (PPO 미러는 제외)"
echo "최종 판정은 Windows 의 run_repro.sh smoke 로 한다."
