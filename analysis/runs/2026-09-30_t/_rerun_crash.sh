#!/bin/bash
# 2026-09-30 19:5x 저자 '다시 지금 진행해': GPU1 오류(19:06, nvlddmkm 153)로 죽은 t_a8_s43·t_a2_s46 를 지금 trunk 부터 다시 학습.
#   드라이버(_run_t.sh)는 먼저 멈춤 — 드라이버 재시도(시도 2, 20:30 이후)가 같은 체크포인트·로그를 동시에 쓰지 않게.
#   학습 명령·env 는 _run_t.sh 의 train_stage 와 1:1(팔 off a8 a2 · 시드 43 46 · REUSE → 끝난 갈래는 재사용, 빠진 두 개만 학습).
#   모든 학습 프로세스가 끝나면 드라이버를 다시 띄움 → 끝난 단계 건너뜀 → train2(전부 재사용) → eval2 → plots → 전 단계 완료.
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
R=/c/work/DT_Vessel_v3; PRE=t_
cd $R/Python || exit 1
export PYTHONIOENCODING=utf-8 VESSEL_PY=/c/Users/OSH/anaconda3/envs/mltest/python.exe
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0
export VESSEL_COMM_TELEMETRY=1
export VESSEL_CKPT_DIR=$HOME/VESSEL_checkpoints/comm_intent
export VESSEL_BRANCH_WARMUP=2400
unset VESSEL_REQUIRE_TRUNK VESSEL_REUSE_ARMS VESSEL_SKIP_GOLDEN VESSEL_SEEDS VESSEL_NGPU
export VESSEL_RUN_PREFIX=$PRE VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035
export VESSEL_OUT_DIR=$PWD/_repro_out_${PRE%_}
O=$VESSEL_OUT_DIR; TL=$O/_timeline.txt
export VESSEL_RISK_DCPA_GATE_M=$(cat "$O/_gate_m.txt")
log() { echo "$(date '+%F %T') $*" | tee -a "$TL"; }
TRAIN_GPU="VESSEL_JOBS=8 VESSEL_GPU_CAP=2 VESSEL_LAUNCH_GAP=90 VESSEL_VRAM_MARGIN=1500"
log "재학습 시작: t_a8_s43·t_a2_s46 (trunk 부터 · 팔 off a8 a2 · 시드 43 46 · 끝난 갈래 재사용)"
env CUDA_VISIBLE_DEVICES= VESSEL_NGPU=4 VESSEL_SEEDS="43 46" VESSEL_REQUIRE_TRUNK=1 VESSEL_REUSE_ARMS=1 VESSEL_TRAIN_ARMS="off a8 a2" $TRAIN_GPU bash run_repro.sh train > "$O/_train2_rerun.log" 2>&1
rc=$?
log "재학습 끝 rc=$rc ($(grep -E 't_a8_s43 rc|t_a2_s46 rc' "$O/_status_train.txt" | tr '\n' ' '))"
[ $rc -eq 0 ] || { log "★재학습 실패 rc=$rc — $O/_train2_rerun.log 확인. 드라이버 재시작 안 함"; exit 1; }
while [ "$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'vessel_gym_train' } | Measure-Object | Select-Object -ExpandProperty Count" 2>/dev/null | tr -d '\r')" -gt 0 ]; do sleep 60; done
sleep 30
cp -p $R/Python/_t_driver.log $R/Python/_t_driver_1.log 2>/dev/null
log "드라이버 재시작(끝난 단계 건너뜀 → train2 전부 재사용 → eval2 → plots)"
MSYS_NO_PATHCONV=1 schtasks /run /tn vessel_t >/dev/null 2>&1 || log "★드라이버 재시작 실패 — 수동으로 schtasks /run /tn vessel_t"
