#!/bin/bash
# ============================================================================
# Plan C: latentNEW 병렬 학습 (5개 dim, reduced step)
#
# - MSG_DIM ∈ {2, 4, 8, 10, 12}
# - 각 학습: Phase 1 step 4.07M에서 fork → +5M = step 9.07M에서 종료
# - 동시 5개 병렬, 각각 NUM_ENVS=4 (총 20 Unity instance + 5 Python)
# - i7-13700K + 128GB RAM + RTX 3060 기준 약 1.5일 wall time
#
# 실행 방법 (Git Bash):
#   chmod +x launch_latentNEW.sh
#   ./launch_latentNEW.sh
#
# 모니터링:
#   tail -f logs/latentNEW_*.log
#   nvidia-smi -l 5
#
# 중지 (전부 강제 종료):
#   pkill -f "VESSEL_COMM_FOLDER=latentNEW"
# ============================================================================

cd "$(dirname "$0")"
mkdir -p logs

# 상대경로화: 이 스크립트(Assets/Scripts/Python) 기준 프로젝트 루트 = ../../..
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)"

# Phase 1 분기점 모델 (4.07M step)
MODEL_PATH="$ROOT/models/COMM_NON/VesselNavigation_20260114_183130/policy_step_4070000.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Phase 1 base model not found: $MODEL_PATH"
    exit 1
fi

# Server Build 경로 (사용자가 빌드 후 여기 수정 또는 환경변수 export)
# 비어있으면 config.py default (0424 일반 build) 사용
SERVER_ENV_PATH="${VESSEL_ENV_PATH:-}"
# 예시(상대경로): SERVER_ENV_PATH="$ROOT/Build/Vessel_MLAgent.exe"

# 학습 설정
START_STEP=4070000
RUN_STEP=5000000          # 추가 5M step (4.07M → 9.07M)
NUM_ENVS=2                # 5학습 동시 시 GPU 부담 줄이기 (4→2)
N_EPOCH=2                 # PPO update 시간 1/2 (4→2)
DIMS=(2 4 8 10 12)
PORTS=(5000 5100 5200 5300 5400)

echo "======================================================================"
echo "Plan C — latentNEW 병렬 학습 시작"
echo "  MODEL_PATH:      $MODEL_PATH"
echo "  ENV_PATH:        ${SERVER_ENV_PATH:-(config default = 0424 일반 build)}"
echo "  START_STEP:      $START_STEP"
echo "  RUN_STEP:        $RUN_STEP (각 학습 추가 step)"
echo "  NUM_ENVS:        $NUM_ENVS"
echo "  N_EPOCH:         $N_EPOCH"
echo "  MINIBATCH_SIZE:  512 (config default)"
echo "  DIMS:            ${DIMS[*]}"
echo "  PORTS:           ${PORTS[*]}"
echo "======================================================================"

# 학습 5개 spawn (15초 간격으로 init 충돌 방지)
PIDS=()
for i in "${!DIMS[@]}"; do
    d=${DIMS[$i]}
    p=${PORTS[$i]}

    echo "[$(date +%H:%M:%S)] Launching dim=$d on port $p..."

    VESSEL_MSG_DIM=$d \
    VESSEL_COMM_FOLDER="latentNEW/dim_${d}" \
    VESSEL_BASE_PORT=$p \
    VESSEL_RUN_STEP=$RUN_STEP \
    VESSEL_START_STEP=$START_STEP \
    VESSEL_MODEL_PATH="$MODEL_PATH" \
    VESSEL_NUM_ENVS=$NUM_ENVS \
    VESSEL_N_EPOCH=$N_EPOCH \
    VESSEL_ENV_PATH="$SERVER_ENV_PATH" \
    nohup python main.py > "logs/latentNEW_dim${d}.log" 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "    → PID=$PID, log=logs/latentNEW_dim${d}.log"

    if [ $i -lt $((${#DIMS[@]} - 1)) ]; then
        sleep 15
    fi
done

echo ""
echo "======================================================================"
echo "All 5 training processes launched."
echo "PIDs: ${PIDS[*]}"
echo ""
echo "모니터링:"
echo "  tail -f logs/latentNEW_*.log"
echo "  nvidia-smi -l 5"
echo "  htop  (또는 작업 관리자)"
echo ""
echo "종료 (모두 강제):"
echo "  pkill -f 'VESSEL_COMM_FOLDER=latentNEW'"
echo "  또는 각 PID 개별 kill: kill ${PIDS[*]}"
echo "======================================================================"
