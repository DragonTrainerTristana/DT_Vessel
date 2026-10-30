#!/bin/bash
# ablate_comm.sh <prefix> [run] — ★2026-10-30 통신 갈래 메시지 절제(학습 없음, 기록용). 스펙 2026-10-28 §6
#   통신 체크포인트를 평균 행동(기본, VESSEL_F_WHICH=sample 이면 표본)으로 절제 조건마다 평가 → 같은 체크포인트의 기본 평가·OFF 와 표로 비교
#   조건(VESSEL_F_ABL_CONDS, 기본 넷):
#     far56   = --far_drop 56 : 56 m 밖 파트너를 통째로 뺌(위치·코덱 필드) → 파트너 집합이 OFF(ARPA@56)와 같음 = '300 m 통신 전체' 몫
#     shuffle = --field_shuffle : 코덱 필드를 같은 env 다른 수신자 것으로 섞음(위치는 그대로) = '메시지 내용' 몫
#     decl0   = --decl_zero : 역할 선언 칸 [13:18] 만 0 = '역할 선언' 몫(판정기 내부 상태 출처 점검)
#     state   = --comm_groups state : 의도·역할 그룹을 끔(상태만) = '의도·역할' 몫
#   평가 인자 = meaneval_ckpts.sh 와 같음(--envs 256 --eval_decisions 3000 --burnin 2400 --drain 3000 --seed 999) → 같은 체크포인트 기본값과 짝
# 실행(Git Bash, 클론 루트). 조건마다 프로세스를 나눠 동시에:
#   for c in far56 shuffle decl0 state; do CUDA_VISIBLE_DEVICES=1 VESSEL_F_REWARD_RANGE=56 VESSEL_F_ONLY="16 14" VESSEL_F_ABL_CONDS=$c \
#     nohup bash Python/runs_fig1/2026-10-02_fig1/ablate_comm.sh qd_ comm > abl_$c.log 2>&1 & done
#   표만: bash .../ablate_comm.sh qd_ --table
# 결과 = Python/_repro_out_<prefix>/ablate/ (+ Dropbox runs/2026-10-02_fig1/out/ablate_<prefix>/ 사본)
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
PRE=${1:?사용: ablate_comm.sh <prefix> [comm|--table]}; RUN=${2:-comm}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$(git -C "$HERE" rev-parse --show-toplevel) || exit 1
cd "$R/Python" || exit 1
PY=${VESSEL_PY:-$HOME/anaconda3/envs/mltest/python.exe}
[ -x "$PY" ] || PY=$(command -v python3)
export PYTHONIOENCODING=utf-8
CK=${VESSEL_CKPT_DIR:-$HOME/VESSEL_checkpoints/comm_intent}
SEED=${VESSEL_F_SEED:-43}
# VESSEL_F_OUT_DIR = 그 배치 출력 폴더(코덱 기록·기본 평가가 있는 곳). 다른 작업 폴더(worktree)에서 돌릴 때 원 클론의 _repro_out_<prefix> 를 가리킴
O=${VESSEL_F_OUT_DIR:-$PWD/_repro_out_${PRE%_}}; D="$O/ablate"; M="$O/meaneval"; mkdir -p "$D"
EV_ARGS="--arm ON --envs 256 --eval_decisions 3000 --burnin 2400 --drain 3000 --seed 999"
# ── env = meaneval_ckpts.sh 와 같음(n_/q_ 배치 env + comm = p46 코덱) ──
source "$R/Python/common_env.sh"; common_env
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0 VESSEL_CKPT_DIR=$CK
export VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035 VESSEL_RISK_DCPA_GATE_M=48 VESSEL_ACTION_MODE=course
export VESSEL_ROLE_V2_PRIMARY=cum VESSEL_ROLE_V2_RES_F6=1
unset VESSEL_BC_TEACHER VESSEL_BC_COEF VESSEL_BC_DECAY_DEC VESSEL_REWARD_RANGE
unset VESSEL_LOGSTD_INIT0 VESSEL_LR_FINAL VESSEL_LR_DECAY_DEC VESSEL_ENTROPY_BONUS VESSEL_LOGSTD_MIN VESSEL_NOISE_HOLD
[ -z "${VESSEL_F_REWARD_RANGE:-}" ] || export VESSEL_REWARD_RANGE=$VESSEL_F_REWARD_RANGE
if [ "$RUN" != "--table" ]; then
  [ "$RUN" = comm ] || { echo "★멈춤: 절제는 통신 갈래(comm)만"; exit 1; }
  cf=$(sed -n "s/^CODEC_s$SEED=\([^ ]*\) .*/\1/p" "$O/_codec_choice.txt"); sh=$(sed -n "s/^CODEC_s$SEED=.* SHA=\([0-9a-f]*\)$/\1/p" "$O/_codec_choice.txt")
  [ -n "$cf" ] && [ -n "$sh" ] || { echo "★멈춤: 코덱 경로·SHA 없음($O/_codec_choice.txt s$SEED)"; exit 1; }
  export VESSEL_USE_COMM=1 VESSEL_MSG_DIM=6 VESSEL_COMM_LATENT=0.0 VESSEL_AUX_LOSS_SCALE=0.0
  export VESSEL_COMM_FIELDS=intent VESSEL_PARTNER_RANGE=300 VESSEL_COMM_CODEC="$cf" VESSEL_COMM_CODEC_SHA=$sh VESSEL_COMM_CODEC_MODE=decode
  W=${VESSEL_F_WHICH:-mean}
  echo "$(date '+%F %T') ablate $PRE $RUN s$SEED [${VESSEL_F_ABL_CONDS:-far56 shuffle decl0 state}] ckpt=[${VESSEL_F_ONLY:-16}] $W $(git -C "$R" log -1 --oneline) GPU=${CUDA_VISIBLE_DEVICES:-auto}" | tee -a "$D/_log.txt"
  for x in ${VESSEL_F_ONLY:-16}; do
    f="$CK/${PRE}${RUN}_s$SEED.step${x}M.pt"; [ -f "$f" ] || { echo "★멈춤: 체크포인트 없음 $f"; exit 1; }
    for c in ${VESSEL_F_ABL_CONDS:-far56 shuffle decl0 state}; do
      case $c in
        far56) A="--far_drop 56" ;; shuffle) A="--field_shuffle" ;; decl0) A="--decl_zero" ;;
        state) A="--comm_groups state --allow_fields_mismatch" ;; *) echo "★멈춤: 모르는 조건 $c"; exit 1 ;;
      esac
      out="$D/abl_${c}_${W}_${PRE}${RUN}_${x}M.txt"
      grep -q 'epReward=' "$out" 2>/dev/null && continue
      if [ "$W" = mean ]; then "$PY" -u "$HERE/eval_mean.py" "$R/Python" -- --ckpt "$f" $EV_ARGS $A > "$out" 2>&1
      else "$PY" -u eval/eval_ckpt.py --ckpt "$f" $EV_ARGS $A > "$out" 2>&1; fi
      grep -q 'epReward=' "$out" && grep -q '\[eval\] 절제' "$out" || { echo "★FAIL 절제 평가 실패 $out"; exit 1; }
      echo "$(date '+%F %T') $c ${x}M 끝" | tee -a "$D/_log.txt"
    done
  done
fi
# ── 표: 조건별 도착·배 충돌·epReward vs 같은 체크포인트 기본(통신)·OFF (meaneval 파일) ──
"$PY" - "$D" "$M" "$PRE" <<'PYEOF' | tee "$D/_ablate_${PRE}.md"
import os, re, sys, glob
d, m, pre = sys.argv[1:4]
def rd(p):
    if not os.path.exists(p): return None
    t = open(p, encoding='utf-8', errors='replace').read()
    g = re.search(r'goal=\s*([0-9.]+)%\s+vColl=\s*([0-9.]+)%', t); r = re.search(r'epReward=\s*(-?[0-9.]+)', t)
    return (float(g[1]), float(g[2]), float(r[1])) if g and r else None
rows = {}
for f in glob.glob(os.path.join(d, f'abl_*_{pre}comm_*M.txt')):
    mm = re.match(rf'abl_(\w+?)_(mean|sample)_{re.escape(pre)}comm_([0-9.]+)M\.txt$', os.path.basename(f))
    if mm and rd(f): rows[(mm[3], mm[2], mm[1])] = rd(f)
print(f'# 통신 메시지 절제 {pre} — 같은 체크포인트 기본 평가·OFF 와 비교 (기록용, 시드 1개 = 단독 주장 금지)')
print('| 체크포인트 | 행동 | 조건 | 도착 % | 배 충돌 % | epReward | 도착: 기본 통신 − 이 조건 | 도착: 이 조건 − OFF |')
print('|---|---|---|---|---|---|---|---|')
for (x, w, c) in sorted(rows, key=lambda k: (float(k[0]), k[1], k[2])):
    v = rows[(x, w, c)]
    base = rd(os.path.join(m, f'{w}_{pre}comm_{x}M.txt')); off = rd(os.path.join(m, f'{w}_{pre}off_{x}M.txt'))
    db = '–' if not base else f'{base[0] - v[0]:+.1f}'; do = '–' if not off else f'{v[0] - off[0]:+.1f}'
    if c == sorted(k[2] for k in rows if k[0] == x and k[1] == w)[0]:
        if base: print(f'| {x}M | {w} | (기본 통신) | {base[0]:.1f} | {base[1]:.1f} | {base[2]:.0f} | 0 | {"–" if not off else f"{base[0]-off[0]:+.1f}"} |')
        if off: print(f'| {x}M | {w} | (OFF 갈래) | {off[0]:.1f} | {off[1]:.1f} | {off[2]:.0f} | – | 0 |')
    print(f'| {x}M | {w} | {c} | {v[0]:.1f} | {v[1]:.1f} | {v[2]:.0f} | {db} | {do} |')
PYEOF
DROP=$HOME/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-02_fig1/out
[ -d "$DROP" ] && { mkdir -p "$DROP/ablate_${PRE%_}"; cp "$D"/*.txt "$D"/*.md "$DROP/ablate_${PRE%_}/" 2>/dev/null; }
echo "$(date '+%F %T') 끝 → $D/_ablate_${PRE}.md"
