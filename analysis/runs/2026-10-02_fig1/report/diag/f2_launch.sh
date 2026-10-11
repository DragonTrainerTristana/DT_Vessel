#!/bin/bash
# 2026-10-03 f2_ = Fig1 배치 재실행(같은 코드 9183a0f, 접두어만 f2_). 실험 설정·기준은 _run_f.sh 그대로.
# 실행 환경만 바꿈(10-02 f_ DAgger 정지·TDR 원인 측정 결과):
#   PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3  — config.py 기본(max_split_size_mb:128)이 DAgger 캐시를 14 GB 까지 불림(실사용 3.2 GB)
#   phase0(DAgger·평가, 프로세스당 4.2 GB) = GPU당 2개, phase1(trunk·갈래, 프로세스당 8.8 GB) = GPU당 1개
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
export VESSEL_F_PREFIX=f2_ PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3
cd /c/work/DT_Vessel_v3 || exit 1
RUN=Python/runs_fig1/2026-10-02_fig1/_run_f.sh
VESSEL_F_GPU_CAP=2 bash $RUN phase0 > Python/_f2_phase0_driver.log 2>&1
if grep -q "P0 판정: 통과" Python/_repro_out_f2/_p0.md 2>/dev/null; then
  VESSEL_F_GPU_CAP=1 VESSEL_F_JOBS=4 bash $RUN phase1 > Python/_f2_phase1_driver.log 2>&1
fi
