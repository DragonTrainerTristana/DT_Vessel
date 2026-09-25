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
#   6. test_comm_ext.py             (의도·역할 통신 필드 정의·미러·체크포인트, 2026-09-25)
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

# ★2026-09-26: run_repro.sh preflight 와 같은 구현(preflight_checks.sh) — 검사를 동시에 돌리고 코드·env 지문이 같으면 건너뜀.
#   PPO 미러만 빼고(Mac 미지원) 나머지 6종. 강제로 다시: VESSEL_FORCE_PREFLIGHT=1
source "$HERE/preflight_checks.sh"
vessel_preflight_cached 0 smoke_mac || { echo "smoke_mac 실패 — 위 ★FAIL 파일 확인"; exit 1; }

echo
echo "Mac 검증 완료 — 통신 미러·골든·충실도·동역학 프로필·sim 스냅샷·COMM_EXT 전부 PASS 또는 캐시 적중 (PPO 미러는 제외)"
echo "최종 판정은 Windows 의 run_repro.sh smoke 로 한다."
