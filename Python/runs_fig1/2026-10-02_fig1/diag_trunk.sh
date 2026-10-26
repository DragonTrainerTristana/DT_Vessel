#!/bin/bash
# diag_trunk.sh <prefix> — ★2026-10-13 D 진단(학습 없음, GPU 1장 ≈45분). 스펙 docs/superpowers/specs/2026-10-13-trunk-fork-design.md 단계 1-D
#   a) 체크포인트 2·4·6·8·9.04M 의 ControlActor logstd(전문가 5)      → logstd.txt
#   b) 4M·9.04M 표본 행동 평가(eval_ckpt) vs 평균 행동 평가(eval_mean)   → eval_sample_*.txt · eval_mean_*.txt
#   c) 9.04M 에서 +1 update 재개(저장 안 함)로 critic 설명분산 EV      → ev.txt
#   결과 = Python/_repro_out_<prefix>/diag_trunk/ (+ Dropbox runs/2026-10-02_fig1/out/diag_trunk_<prefix>/ 사본). 기록용, 관문 아님.
# 실행(Git Bash, 클론 루트): bash Python/runs_fig1/2026-10-02_fig1/diag_trunk.sh n_      (GPU 고르려면 CUDA_VISIBLE_DEVICES=3 앞에)
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
PRE=${1:-n_}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$(git -C "$HERE" rev-parse --show-toplevel) || exit 1
cd "$R/Python" || exit 1
PY=${VESSEL_PY:-$HOME/anaconda3/envs/mltest/python.exe}
[ -x "$PY" ] || { echo "★멈춤: python 없음($PY)"; exit 1; }
export PYTHONIOENCODING=utf-8
CK=${VESSEL_CKPT_DIR:-$HOME/VESSEL_checkpoints/comm_intent}
T="$CK/${PRE}trunk_d6_s43"
for f in "$T.pt" "$T.step2M.pt" "$T.step4M.pt" "$T.step6M.pt" "$T.step8M.pt"; do [ -f "$f" ] || { echo "★멈춤: 없음 $f"; exit 1; }; done
D="$PWD/_repro_out_${PRE%_}/diag_trunk"; mkdir -p "$D"
# ── env = _run_f.sh 의 n_ 배치 env + ENV_ARPA (restore_policy 가 스냅샷과 대조하므로 틀리면 멈춤) ──
source "$R/Python/common_env.sh"; common_env
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0 VESSEL_CKPT_DIR=$CK
export VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035 VESSEL_RISK_DCPA_GATE_M=48 VESSEL_ACTION_MODE=course
export VESSEL_ROLE_V2_PRIMARY=cum VESSEL_ROLE_V2_RES_F6=1
export VESSEL_USE_COMM=1 VESSEL_MSG_DIM=6 VESSEL_COMM_LATENT=0.0 VESSEL_AUX_LOSS_SCALE=0.0 VESSEL_COMM_FIELDS=state VESSEL_PARTNER_RANGE=56
unset VESSEL_COMM_CODEC VESSEL_COMM_CODEC_SHA VESSEL_COMM_CODEC_MODE VESSEL_BC_TEACHER VESSEL_BC_COEF VESSEL_BC_DECAY_DEC
unset VESSEL_REWARD_RANGE VESSEL_LOGSTD_INIT0 VESSEL_LR_FINAL VESSEL_LR_DECAY_DEC VESSEL_ENTROPY_BONUS VESSEL_LOGSTD_MIN VESSEL_NOISE_HOLD
EV_ARGS="--arm ON --envs 256 --eval_decisions 3000 --burnin 2400 --drain 3000 --seed 999"
TARGS="--envs 128 --vessels 16 --rollout 32 --ring 1.0 --crossing 0 --max_partners 8"
echo "$(date '+%F %T') diag_trunk $PRE $(git -C "$R" log -1 --oneline) → $D" | tee "$D/_log.txt"
# a) logstd
"$PY" "$HERE/logstd_dump.py" "$T".step2M.pt "$T".step4M.pt "$T".step6M.pt "$T".step8M.pt "$T.pt" 2>&1 | tee "$D/logstd.txt"
# b) 표본 vs 평균 행동 (4M, 9.04M)
for tag in 4M 9M; do
  [ "$tag" = 4M ] && f="$T.step4M.pt" || f="$T.pt"
  grep -q 'goal=' "$D/eval_sample_$tag.txt" 2>/dev/null || "$PY" -u eval/eval_ckpt.py --ckpt "$f" $EV_ARGS > "$D/eval_sample_$tag.txt" 2>&1
  grep -q 'goal=' "$D/eval_mean_$tag.txt" 2>/dev/null   || "$PY" -u "$HERE/eval_mean.py" "$R/Python" -- --ckpt "$f" $EV_ARGS > "$D/eval_mean_$tag.txt" 2>&1
done
# c) critic 설명분산(EVb = T=32 rollout 부트스트랩 목표에 대한 자기일관성 — MC 수익 적합 판정 아님): 9.04M 을 +1 update(65,536 결정) 재개, 워밍업 2400
grep -q '^\[ppo\] EVb' "$D/ev.txt" 2>/dev/null || VESSEL_GRAD_TELEMETRY=1 "$PY" -u vessel_gym_train.py --arm ON --resume "$T.pt" --resume_at 9043968 --steps 9109504 \
    --comm_on_at 0 --resume_warmup 2400 $TARGS --seed 43 --save "$D/ev_tmp.pt" --csv "$D/ev_tmp.csv" > "$D/ev_full.log" 2>&1
grep -h "^\[ppo\] EVb\|^\[resume\]\|^saved\|거부\|Traceback" "$D/ev_full.log" > "$D/ev.txt"; rm -f "$D/ev_tmp.pt"
# 요약(기록 기준은 스펙 단계 1-D: logstd 9.04M ≤ −1.5 · 평균−표본 goal ≥ 15 pt · 9.04M 평균 < 4M 평균 −5 pt · EVb 는 값만 기록)
"$PY" - "$D" <<'PYEOF' | tee "$D/_diag_trunk.md"
import re, sys, os
d = sys.argv[1]
def goal(f):
    m = re.search(r'goal=\s*([0-9.]+)%', open(os.path.join(d, f), encoding='utf-8', errors='replace').read()); return float(m.group(1)) if m else None
ls = open(os.path.join(d, 'logstd.txt'), encoding='utf-8', errors='replace').read()
m9 = next((float(m.group(1)) for L in ls.splitlines() if 'steps=9043968' in L for m in [re.search(r'mean=(-?[0-9.]+)', L)] if m), None)
ev = open(os.path.join(d, 'ev.txt'), encoding='utf-8', errors='replace').read()
e1 = re.search(r'EVb\(lam=1\)=(-?[0-9.]+)', ev); e1 = float(e1.group(1)) if e1 else None
NA = 'n/a(측정 실패)'
s4, s9, a4, a9 = goal('eval_sample_4M.txt'), goal('eval_sample_9M.txt'), goal('eval_mean_4M.txt'), goal('eval_mean_9M.txt')
f = lambda v: 'n/a' if v is None else f'{v:.1f}'
print('# D 진단 (기록용, 관문 아님)')
print(f'- logstd dim0 9.04M 전문가 평균 = {NA if m9 is None else f"{m9:.3f} → " + ("≤ −1.5 (잡음 스스로 줄었음)" if m9 <= -1.5 else "> −1.5")}')
print(f'- 도착 % 표본/평균: 4M {f(s4)}/{f(a4)} · 9.04M {f(s9)}/{f(a9)}')
if a9 is not None and s9 is not None: print(f'- 9.04M 평균−표본 = {a9 - s9:+.1f} pt → {"흔들림 몫 큼(≥15)" if a9 - s9 >= 15 else "<15"}')
if a9 is not None and a4 is not None: print(f'- 9.04M 평균 − 4M 평균 = {a9 - a4:+.1f} pt → {"정책 퇴화(< −5)" if a9 - a4 < -5 else "퇴화 아님"}')
print(f'- critic EVb(lam=1, T=32 부트스트랩 자기일관성 — 수익 적합 판정 아님) 9.04M = {NA if e1 is None else f"{e1:.3f}"}')
PYEOF
DROP=$HOME/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-02_fig1/out
[ -d "$DROP" ] && { mkdir -p "$DROP/diag_trunk_${PRE%_}"; cp "$D"/*.txt "$D"/*.md "$DROP/diag_trunk_${PRE%_}/" 2>/dev/null; }
echo "$(date '+%F %T') 끝 → $D/_diag_trunk.md" | tee -a "$D/_log.txt"
