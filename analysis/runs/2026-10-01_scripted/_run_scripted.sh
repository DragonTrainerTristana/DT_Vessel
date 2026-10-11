#!/bin/bash
# 규칙 배 평가 (2026-10-01, 저자 승인 "ㅇㅋ 그리고 아까 지표 4개도 넣어봐" → 2차 "COLREGs 규칙 배 + DCPA·fuel·rudder 점검"
#   → 3차 2026-10-02 "ㅇㅋ 진행": Woerner 2019 COLREGs 채점(colregs_woerner.py) 추가, 규칙 배 5종 + 학습 배 t_off·t_a6 재채점)
#   → 4차 2026-10-02 "ㅇㅋ 진행해줘"(Fig1, 철학 B): Woerner 에 8조 작은 변침 판정(R8) 추가 + 의도 공유(col300i·vo300i) + 규칙 v2(…h)
#   2차 기본 = col56·col300(COLREGs 규칙 + VO, 보는 거리만 다름) + vo56·vo300 재실행(타 지표 추가). 타 지표 줄 있는 칸은 건너뜀 — 학습 없음, 체크포인트 가중치 안 씀
#   질문: 통신 정보(300 m)가 강화학습 없이 규칙 배에서도 도움이 되나. 보고서 지표 9개 전부를 학습 배 평가 코드(eval_ckpt.py) 그대로 기록
#   정책: goal(안 피함) · vo56(56 m·60 s) · vo56h150(56 m·150 s) · vo300(300 m·150 s) · col56·col300(COLREGs 규칙 + VO, 150 s) — 짝은 보는 거리만 다름
#   평가: t_ 주 평가와 같은 조건(--envs 256 --eval_decisions 10000 --burnin 2400 --arm OFF) × 평가 시드 999·1000·1001
#   env 설정 = t_ 배치(_run_evalcurve.sh batch_env 와 1:1). --ckpt 는 env 스냅샷 운반용(t_off_s43.pt), 가중치 미사용
#   이미 끝난 칸은 건너뜀 → 여러 번 돌려도 됨. 끝나면 summarize.py 가 _summary.md 를 씀
# 실행: bash /c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-01_scripted/_run_scripted.sh
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1 PYTHONIOENCODING=utf-8
R=/c/work/DT_Vessel_v3/Python
PY=/c/Users/OSH/anaconda3/envs/mltest/python.exe
DROP=/c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-01_scripted
O=$R/_repro_out_t/scripted
CK=$HOME/VESSEL_checkpoints/comm_intent
CARRIER=$CK/t_off_s43.pt
POLS=${VESSEL_SC_POLS:-"col56h col300ih col300h col300i col56 col300 vo56 vo300 vo300i goal"}   # ★2026-10-02 4차: Fig1 핵심 짝(col56h·col300ih)부터. R8 줄 없는 칸은 자동 재실행
LEARN_ARMS=${VESSEL_SC_LEARN:-"off a6"}; LEARN_SEEDS="43 46 47"   # 학습 배 t_ 같은 체크포인트 재채점(정책 그대로, 평가 시드 999 = 주 평가와 같음)
SEEDS="999 1000 1001"
JOBS=${VESSEL_SC_JOBS:-3}
mkdir -p "$O" "$DROP/out"
[ -f "$CARRIER" ] || { echo "★스냅샷 운반 체크포인트 없음: $CARRIER"; exit 1; }
cd "$R" || exit 1
echo "브랜치 $(git -C "$R/.." rev-parse --abbrev-ref HEAD) $(git -C "$R/.." rev-parse --short HEAD) (feat/reward-v3 또는 feat/imitation-init — eval 코드 같음)"
source "$R/common_env.sh"
batch_env() {   # t_ 배치와 같은 env (_run_evalcurve.sh 와 1:1). eval_ckpt 가 스냅샷과 대조함
  common_env
  export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0 VESSEL_CKPT_DIR=$CK
  export VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
  export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035 VESSEL_RISK_DCPA_GATE_M=48
  export VESSEL_MSG_DIM=6 OMP_NUM_THREADS=1
}
pick_gpu() {    # 남은 메모리 가장 큰 GPU (3.7 GB 이상)
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -nr | awk -F, '$2>=3700{print $1; exit}'
}
for p in $POLS; do
  for s in $SEEDS; do
    out="$O/sc_${p}_e$s.txt"
    grep -q "R8=" "$out" 2>/dev/null && { echo "  건너뜀 $p e$s (끝남, R8 판정 있음)"; continue; }
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
    g=$(pick_gpu); while [ -z "$g" ]; do sleep 60; g=$(pick_gpu); done
    ( batch_env; export CUDA_VISIBLE_DEVICES=$g
      "$PY" -u "$DROP/eval_scripted.py" --py_root "$R" --policy "$p" -- \
        --ckpt "$CARRIER" --arm OFF --envs 256 --eval_decisions 10000 --burnin 2400 --seed "$s" > "$out" 2>&1
      cp "$out" "$DROP/out/" 2>/dev/null ) &
    echo "  평가 $p e$s → GPU $g"
    sleep 20
  done
done
for arm in $LEARN_ARMS; do
  for s in $LEARN_SEEDS; do
    out="$O/lr_t_${arm}_s$s.txt"; ck="$CK/t_${arm}_s$s.pt"; am=OFF; [ "$arm" = off ] || am=ON
    [ -f "$ck" ] || { echo "  없음 $ck"; continue; }
    grep -q "R8=" "$out" 2>/dev/null && { echo "  건너뜀 t_$arm s$s (끝남, R8 판정 있음)"; continue; }
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
    g=$(pick_gpu); while [ -z "$g" ]; do sleep 60; g=$(pick_gpu); done
    ( batch_env; export CUDA_VISIBLE_DEVICES=$g
      "$PY" -u "$DROP/eval_scripted.py" --py_root "$R" --policy learned -- \
        --ckpt "$ck" --arm "$am" --envs 256 --eval_decisions 10000 --burnin 2400 --seed 999 > "$out" 2>&1
      cp "$out" "$DROP/out/" 2>/dev/null ) &
    echo "  학습 배 재채점 t_$arm s$s ($am) → GPU $g"
    sleep 20
  done
done
wait
"$PY" "$DROP/summarize.py" "$DROP/out" > "$DROP/_summary.md" 2>&1
echo "끝 → $DROP/_summary.md"
