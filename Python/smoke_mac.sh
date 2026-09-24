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
#   common_env 는 common_env.sh 를 source 한다 (정본 = 그 파일 하나, 2026-09-24 단일화).
#
# 돌리는 것 (이 순서로, run_repro.sh preflight 와 동일한 호출):
#   1. _verify_comm_mirror.py       (통신 집계 rollout=update 미러)
#   2. test_golden.py --check       (기본값 결과 비트동일 골든)
#   3. test_vessel_gym_fidelity.py  (vessel_gym 물리·obs 충실도)
#   4. test_dyn_profile.py          (동역학 프로필·시나리오. env 를 그대로 물려받음 —
#                                    imo/none 으로 돌리려면 VESSEL_DYN_PROFILE·VESSEL_OBSTACLES 를 밖에서 줄 것)
#   5. test_sim_snapshot.py         (스냅샷 sim 상수 24개 기록·대조·복원)
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

# common_env 는 common_env.sh 를 source 한다 (정본 = 그 파일 하나, run_repro.sh 도 같은 것을 씀).
source "$HERE/common_env.sh"

echo "Mac 검증 [smoke_mac] — PPO 미러 제외 (Mac 미지원)"
echo "  python : $PY"
echo "  출력   : $OUT"
echo

# ★2026-09-15: 인터프리터 sanity (run_repro.sh preflight 와 같은 취지). torch 없는 python 을
#   잡으면 아래 3단계가 전부 "미러 FAIL" 처럼 원인과 무관한 메시지로 끝난다.
"$PY" -c "import torch" >/dev/null 2>&1 || {
  echo "smoke_mac 실패: '$PY' 에서 torch 를 import 하지 못함."
  echo "  → VESSEL_PY 로 torch 가 설치된 인터프리터를 지정할 것."
  "$PY" -c "import torch" 2>&1 | tail -3 | sed 's/^/  /'
  exit 1
}

common_env

echo "[1/5] 통신 미러 검증"
"$PY" -u "$HERE/verify/_verify_comm_mirror.py" > "$OUT/_verify_comm.txt" 2>&1 || { echo "  통신 미러 FAIL — $OUT/_verify_comm.txt 확인"; exit 1; }
grep -q "ALL PASS" "$OUT/_verify_comm.txt" || { echo "  통신 미러가 ALL PASS 가 아님"; exit 1; }
echo "  통신 미러 ALL PASS"

echo "[2/5] 골든 비트동일 검사"
( env -u VESSEL_STATE_RECON_COEF -u VESSEL_CENTRAL_CRITIC -u VESSEL_USE_ATTENTION \
    "$PY" -u "$HERE/verify/test_golden.py" --check ) > "$OUT/_golden.txt" 2>&1 \
  || { echo "  골든 FAIL — $OUT/_golden.txt 확인 (코드가 기본값 결과를 바꿨음)"; exit 1; }
grep -q "ALL PASS" "$OUT/_golden.txt" || { echo "  골든이 ALL PASS 가 아님"; exit 1; }
echo "  골든 ALL PASS"

echo "[3/5] vessel_gym 충실도 검사"
"$PY" -u "$HERE/verify/test_vessel_gym_fidelity.py" > "$OUT/_fidelity.txt" 2>&1 \
  || { echo "  vessel_gym 충실도 FAIL — $OUT/_fidelity.txt 확인"; exit 1; }
echo "  충실도 PASS"

echo "[4/5] 동역학 프로필 검사"
"$PY" -u "$HERE/verify/test_dyn_profile.py" > "$OUT/_dyn_profile.txt" 2>&1 \
  || { echo "  동역학 프로필 FAIL — $OUT/_dyn_profile.txt 확인"; exit 1; }
grep -q "ALL PASS" "$OUT/_dyn_profile.txt" || { echo "  동역학 프로필이 ALL PASS 가 아님"; exit 1; }
echo "  동역학 프로필 ALL PASS"

echo "[5/5] 스냅샷 sim 상수 검사"
"$PY" -u "$HERE/verify/test_sim_snapshot.py" > "$OUT/_sim_snapshot.txt" 2>&1 \
  || { echo "  sim 스냅샷 FAIL — $OUT/_sim_snapshot.txt 확인"; exit 1; }
grep -q "ALL PASS" "$OUT/_sim_snapshot.txt" || { echo "  sim 스냅샷이 ALL PASS 가 아님"; exit 1; }
echo "  sim 스냅샷 ALL PASS"

echo
echo "Mac 검증 완료 — 통신 미러·골든·충실도·동역학 프로필·sim 스냅샷 전부 PASS (PPO 미러는 제외)"
echo "최종 판정은 Windows 의 run_repro.sh smoke 로 한다."
