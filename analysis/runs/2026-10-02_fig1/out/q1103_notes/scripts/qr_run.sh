#!/bin/bash
# qr_run.sh <접두어> <시드> <phase1|extend> [GPU 번호 또는 목록 "0 1"]
cd /c/work/DT_Vessel_v3 || exit 1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3 VESSEL_F_GPU="${4:-0}" VESSEL_F_GPU_CAP=1 VESSEL_F_JOBS=2 VESSEL_F_EVAL_JOBS=6 VESSEL_F_EVAL_GPU_CAP=3   # 11-03 저자 승인: GPU 2장(목록) · 갈래 2개 동시 · 평가 한 장에 3개
export VESSEL_F_IMIT=0 VESSEL_F_PREFIX=$1 VESSEL_F_SEEDS=$2 VESSEL_F_NGATE=1 VESSEL_F_REWARD_RANGE=56
export VESSEL_F_EXTRA_ENV="VESSEL_ENTROPY_BONUS=0 VESSEL_LOGSTD_MIN=-3.0 VESSEL_LOGSTD_INIT0=-2.0 VESSEL_GRAD_TELEMETRY=1"
export VESSEL_F_RANK_OVERRIDE="저자 10-29: 순위 관문 v5 불통과(시드1 (iii) 여유 부족·(v) v4 와 같은 값) 넘기고 성공 경로 진행"
export VESSEL_PY="/c/Users/OSH/anaconda3/envs/mltest/python.exe"
export VESSEL_F_GOLDEN_WAIVE="저자 11-03: 두 번째 기계(LICS-SERVER) 골든 비트동일 불통과는 CPU·torch 차이 — 다른 14종 PASS·학습 코드 = c034422 라 면제하고 진행"
[ "$3" = extend ] && export VESSEL_F_TOTAL=24051712
[ -n "$4" ] && export VESSEL_F_ALLOW_BUSY=1
bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh "$3"
