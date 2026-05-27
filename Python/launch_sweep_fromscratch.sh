#!/bin/bash
# ============================================================================
# From-scratch 통신 ON sweep: MSG_DIM ∈ {2,4,6,8,10,12} 6개 병렬
#
# - from-scratch: config.py LOAD_MODEL=False → 모델 미로드, step 0부터
#   (옛 모델 dual-scale 호환 불가하므로 Phase 분리 없이 통신 ON 처음부터)
# - 빌드 exe 필요: 에디터는 단일 인스턴스라 병렬 sweep 불가 → VESSEL_USE_EDITOR=0
# - obs 43D + BehaviorType Default 는 현재 C#/prefab 에 반영됨 (빌드에 포함)
#
# 실행 (Git Bash):
#   chmod +x launch_sweep_fromscratch.sh
#   ./launch_sweep_fromscratch.sh
# 모니터링:  tail -f logs/sweep_dim*.log   |   nvidia-smi -l 5
# 종료:      kill <PID들>   또는  pkill -f "VESSEL_COMM_FOLDER=sweep_fromscratch"
# ============================================================================

cd "$(dirname "$0")"
mkdir -p logs

# ⬇⬇⬇ 빌드한 exe 경로로 교체 (또는 export VESSEL_ENV_PATH=... 후 실행) ⬇⬇⬇
ENV_PATH="${VESSEL_ENV_PATH:-c:/Users/sengh/Dropbox/Private_Paper_Project/Vessel/Vessel_MLAgent/Build/0526_ver5/Vessel_MLAgent.exe}"
# 신뢰 outcome 로그 per-run 절대경로 (C#이 씀, Windows 스타일 경로 필요)
OUTDIR="c:/Users/sengh/OneDrive/Desktop/Github/MyUnity/Vessel/Vessel_MLAgent/Assets/Scripts/Python/logs"

if [ ! -f "$ENV_PATH" ]; then
    echo "[ERROR] Build exe not found: $ENV_PATH"
    echo "        Unity에서 빌드 후 .exe 경로를 ENV_PATH 변수에 넣으세요."
    exit 1
fi

# ── 튜닝 가능 ───────────────────────────────────────────────────────────────
RUN_STEP=200000       # run당 step 수 (MaxStep 12000 항해정상화 후 MSG_DIM별 outcome 비교)
NUM_ENVS=1            # run당 Unity 인스턴스 (6 run × 1 = 6 instance; NUM_ENVS=2는 GPU 포화 확인됨)
N_EPOCH=2
DIMS=(2 4 6 8 10 12)
PORTS=(5000 5100 5200 5300 5400 5500)
# ────────────────────────────────────────────────────────────────────────────

echo "===================================================================="
echo "From-scratch 통신 sweep — MSG_DIM ${DIMS[*]}"
echo "  ENV_PATH:  $ENV_PATH"
echo "  RUN_STEP:  $RUN_STEP    NUM_ENVS: $NUM_ENVS    (LOAD_MODEL=False, step0~)"
echo "  PORTS:     ${PORTS[*]}"
echo "===================================================================="

PIDS=()
for i in "${!DIMS[@]}"; do
    d=${DIMS[$i]}
    p=${PORTS[$i]}
    echo "[$(date +%H:%M:%S)] Launching dim=$d on port $p..."

    VESSEL_USE_EDITOR=0 \
    VESSEL_MSG_DIM=$d \
    VESSEL_COMM_FOLDER="sweep_fromscratch/dim_${d}" \
    VESSEL_BASE_PORT=$p \
    VESSEL_RUN_STEP=$RUN_STEP \
    VESSEL_NUM_ENVS=$NUM_ENVS \
    VESSEL_N_EPOCH=$N_EPOCH \
    VESSEL_ENV_PATH="$ENV_PATH" \
    VESSEL_OUTCOME_LOG="$OUTDIR/outcomes_dim${d}.csv" \
    nohup python main.py > "logs/sweep_dim${d}.log" 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "    → PID=$PID, log=logs/sweep_dim${d}.log"
    if [ $i -lt $((${#DIMS[@]} - 1)) ]; then sleep 15; fi   # init 충돌 방지
done

echo ""
echo "All ${#DIMS[@]} runs launched. PIDs: ${PIDS[*]}"
echo "모니터링: tail -f logs/sweep_dim*.log   |   nvidia-smi -l 5"
echo "종료:     kill ${PIDS[*]}   또는  pkill -f 'VESSEL_COMM_FOLDER=sweep_fromscratch'"
