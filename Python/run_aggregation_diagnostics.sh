#!/bin/bash
# Aggregation Diagnostics — 5 experiments
# Each: 5 runs × 1000 steps, time_scale=20

set -e
cd "$(dirname "$0")"

RUNS=5
STEPS=1000
TS=20

# 동일한 model 사용 (Phase 2 v2 sum-of-4 trained)
# USE_COMMUNICATION=True 보장 (config.py default)

run_diag() {
    local TAG=$1
    shift
    local LOG="logs/diag_${TAG}.log"
    echo "=========================================="
    echo "[DIAG] $TAG"
    echo "Env: $@"
    echo "=========================================="
    env $@ \
        VESSEL_DIAG_TAG="${TAG}" \
        VESSEL_SAVE_TRAJECTORY=1 \
        python -u test.py --multitest --runs $RUNS --steps $STEPS --time_scale $TS \
        > "$LOG" 2>&1
    grep -E "Collision Rate|Success Rate|Avg Reward" "$LOG" | head -3
    echo ""
}

# 1. sum-of-4 baseline (verify modified code works)
run_diag "diag_sum4" VESSEL_MAX_PARTNERS=4 VESSEL_AGG_MODE=sum

# 2. nearest-1 raw (no scaling)
run_diag "diag_nn1_raw" VESSEL_MAX_PARTNERS=1 VESSEL_AGG_MODE=sum

# 3. nearest-1 ×4 (magnitude match via scale mode)
run_diag "diag_nn1_x4" VESSEL_MAX_PARTNERS=1 VESSEL_NEAREST_SCALE=4

# 4. mean-of-4 (magnitude /4 of sum-of-4, full info diversity)
run_diag "diag_mean4" VESSEL_MAX_PARTNERS=4 VESSEL_AGG_MODE=mean

# 5. sum-of-4 with msg_gain 0.5 (half power, full diversity)
run_diag "diag_sum4_gain05" VESSEL_MAX_PARTNERS=4 VESSEL_AGG_MODE=sum VESSEL_MSG_GAIN=0.5

echo "=========================================="
echo "[DONE] All 5 diagnostics complete."
echo "Trajectory CSVs in trajectory_data/"
echo "Logs in logs/diag_*.log"
echo "=========================================="
