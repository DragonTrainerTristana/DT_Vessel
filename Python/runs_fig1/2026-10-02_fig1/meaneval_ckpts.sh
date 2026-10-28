#!/bin/bash
# meaneval_ckpts.sh <prefix> [run ...] — ★2026-10-26 레퍼런스 포팅 P1·P8 무학습 재평가(기록용, 관문·판정 아님)
#   이미 학습된 체크포인트(step*M + 끝 모델)를 같은 조건으로 '표본 행동'(eval_ckpt) 과 '평균 행동'(eval_mean = tanh(mean)) 둘 다 평가하고
#   logstd 를 덤프한 뒤, 체크포인트별 짝 표 + 갈래 마지막 K 점 평균(best·max 선택 없음)을 냄. 근거 = analysis/runs/2026-10-26_ref_port §P1·P8
#   평가 인자 = curve_all 과 같음(--envs 256 --eval_decisions 3000 --burnin 2400 --drain 3000 --seed 999) → 이미 있는 curve_*.txt(표본)는 재사용
#   env = 그 배치(n_·q_ 보상·ARPA@56/300·p46 코덱) — restore_policy 가 스냅샷과 대조해 틀리면 멈춤. f_·f2_(흉내 배치)는 대상 아님
# 실행(Git Bash, 클론 루트). GPU 여러 장이면 run 별로 나눠 동시에:
#   CUDA_VISIBLE_DEVICES=0 bash Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh qb_ trunk_d6
#   CUDA_VISIBLE_DEVICES=1 bash Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh qb_ off
#   CUDA_VISIBLE_DEVICES=2 bash Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh qb_ comm
#   CUDA_VISIBLE_DEVICES=3 VESSEL_F_ARPA_RANGE=300 bash Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh qa_ trunk_d6
#   표만 다시: bash .../meaneval_ckpts.sh qb_ --table
# 결과 = Python/_repro_out_<prefix>/meaneval/ (+ Dropbox runs/2026-10-02_fig1/out/meaneval_<prefix>/ 사본이 있으면)
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
PRE=${1:?사용: meaneval_ckpts.sh <prefix> [trunk_d6|off|comm ...|--table]}; shift
RUNS=${*:-trunk_d6 off comm}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$(git -C "$HERE" rev-parse --show-toplevel) || exit 1
cd "$R/Python" || exit 1
PY=${VESSEL_PY:-$HOME/anaconda3/envs/mltest/python.exe}
[ -x "$PY" ] || PY=$(command -v python3)
export PYTHONIOENCODING=utf-8
CK=${VESSEL_CKPT_DIR:-$HOME/VESSEL_checkpoints/comm_intent}
SEED=${VESSEL_F_SEED:-43}
ARPA_R=${VESSEL_F_ARPA_RANGE:-56}
O="$PWD/_repro_out_${PRE%_}"; D="$O/meaneval"; mkdir -p "$D"
EV_ARGS="--arm ON --envs 256 --eval_decisions 3000 --burnin 2400 --drain 3000 --seed 999"
# ── env = _run_f.sh n_/q_ 배치 env (diag_trunk.sh 와 같음) ──
source "$R/Python/common_env.sh"; common_env
export VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0 VESSEL_CKPT_DIR=$CK
export VESSEL_ROLE_PROMISE_PEN=20 VESSEL_COLREGS_FAR_RANGE=0 VESSEL_COLREGS_FAR_MODE=full
export VESSEL_ROLE_JUDGE=v2 VESSEL_FORWARD_COEF=0 VESSEL_TIME_PENALTY=0.035 VESSEL_RISK_DCPA_GATE_M=48 VESSEL_ACTION_MODE=course
export VESSEL_ROLE_V2_PRIMARY=cum VESSEL_ROLE_V2_RES_F6=1
unset VESSEL_BC_TEACHER VESSEL_BC_COEF VESSEL_BC_DECAY_DEC VESSEL_REWARD_RANGE
unset VESSEL_LOGSTD_INIT0 VESSEL_LR_FINAL VESSEL_LR_DECAY_DEC VESSEL_ENTROPY_BONUS VESSEL_LOGSTD_MIN VESSEL_NOISE_HOLD
[ -z "${VESSEL_F_REWARD_RANGE:-}" ] || export VESSEL_REWARD_RANGE=$VESSEL_F_REWARD_RANGE   # qc_(56 m) 같은 배치용
env_run() {   # env_run <run> : off·trunk = ARPA@$ARPA_R(메시지 없음) · comm = p46 코덱 복원값(_run_f.sh env_comm 과 같음)
  unset VESSEL_COMM_CODEC VESSEL_COMM_CODEC_SHA VESSEL_COMM_CODEC_MODE
  export VESSEL_USE_COMM=1 VESSEL_MSG_DIM=6 VESSEL_COMM_LATENT=0.0 VESSEL_AUX_LOSS_SCALE=0.0
  if [ "$1" = comm ]; then
    local cf sh; cf=$(sed -n "s/^CODEC_s$SEED=\([^ ]*\) .*/\1/p" "$O/_codec_choice.txt"); sh=$(sed -n "s/^CODEC_s$SEED=.* SHA=\([0-9a-f]*\)$/\1/p" "$O/_codec_choice.txt")
    [ -n "$cf" ] && [ -n "$sh" ] || { echo "★멈춤: 코덱 경로·SHA 없음($O/_codec_choice.txt s$SEED)"; return 1; }
    export VESSEL_COMM_FIELDS=intent VESSEL_PARTNER_RANGE=300 VESSEL_COMM_CODEC="$cf" VESSEL_COMM_CODEC_SHA=$sh VESSEL_COMM_CODEC_MODE=decode
  else
    export VESSEL_COMM_FIELDS=state VESSEL_PARTNER_RANGE=$ARPA_R
  fi
}
mtag() { case "$1" in *.step*M.pt) echo "$1" | sed -E 's/.*\.step([0-9.]+)M\.pt$/\1/' ;; *) echo "$2" ;; esac; }
if [ "$RUNS" != "--table" ]; then
  echo "$(date '+%F %T') meaneval $PRE [$RUNS] s$SEED ARPA=$ARPA_R $(git -C "$R" log -1 --oneline) GPU=${CUDA_VISIBLE_DEVICES:-auto} → $D" | tee -a "$D/_log.txt"
  for run in $RUNS; do
    env_run "$run" || exit 1
    base="$CK/${PRE}${run}_s$SEED"; [ "$run" = trunk_d6 ] && last=9.04397 || last=16.0563
    files=$(ls "$base".step*M.pt 2>/dev/null)
    # 끝 모델: trunk 는 늘(9.04M 분기점), 갈래는 step16M 이 없을 때만(qb_ 의 step16M = 끝 모델과 같은 파일 → 이중 계산 막음)
    if [ -f "$base.pt" ] && { [ "$run" = trunk_d6 ] || ! [ -f "$base.step16M.pt" ]; }; then files="$files $base.pt"; fi
    [ -n "$files" ] || { echo "★멈춤: 체크포인트 없음 $base*"; exit 1; }
    "$PY" "$HERE/logstd_dump.py" $files 2>&1 | tee "$D/logstd_${PRE}${run}.txt"
    for f in $files; do
      x=$(mtag "$f" "$last")
      # ★2026-10-28 VESSEL_F_ONLY="4 9.04397" = 그 체크포인트만 · VESSEL_F_WHICH=sample|mean(기본 둘 다) = 프로세스를 나눠 GPU 여러 장에 동시에
      [ -z "${VESSEL_F_ONLY:-}" ] || case " $VESSEL_F_ONLY " in *" $x "*) ;; *) continue ;; esac
      W=${VESSEL_F_WHICH:-both}
      smp="$D/sample_${PRE}${run}_${x}M.txt"; mean="$D/mean_${PRE}${run}_${x}M.txt"
      cur="$O/curve_${PRE}${run}_s${SEED}_${x}M.txt"   # curve_all 이 같은 인자로 잰 표본 평가가 있으면 재사용
      if [ "$W" != mean ] && ! grep -q 'epReward=' "$smp" 2>/dev/null; then
        if grep -q 'epReward=' "$cur" 2>/dev/null; then cp "$cur" "$smp"; echo "  표본 = 기존 curve 재사용 $(basename "$cur")"
        else "$PY" -u eval/eval_ckpt.py --ckpt "$f" $EV_ARGS > "$smp" 2>&1; fi
      fi
      [ "$W" = sample ] || grep -q 'epReward=' "$mean" 2>/dev/null || "$PY" -u "$HERE/eval_mean.py" "$R/Python" -- --ckpt "$f" $EV_ARGS > "$mean" 2>&1
      { [ "$W" = mean ] || grep -q 'epReward=' "$smp"; } && { [ "$W" = sample ] || grep -q 'epReward=' "$mean"; } || { echo "★FAIL 평가 실패 $f (로그 $smp · $mean)"; exit 1; }
      echo "$(date '+%F %T') $run ${x}M 끝" | tee -a "$D/_log.txt"
    done
  done
fi
# ── 표: 체크포인트별 표본/평균 짝 + 갈래 마지막 K 점 평균(K = 그 run 의 step 체크포인트 전부, best·max 선택 없음) ──
"$PY" - "$D" "$PRE" <<'PYEOF' | tee "$D/_meaneval_${PRE}.md"
import os, re, sys, glob
d, pre = sys.argv[1], sys.argv[2]
def read(f):
    t = open(f, encoding='utf-8', errors='replace').read()
    g = re.search(r'goal=\s*([0-9.]+)%\s+vColl=\s*([0-9.]+)%', t); r = re.search(r'epReward=\s*(-?[0-9.]+)', t)
    return (float(g.group(1)), float(g.group(2)), float(r.group(1))) if g and r else None
rows = {}
for f in glob.glob(os.path.join(d, f'mean_{pre}*M.txt')):
    m = re.match(rf'mean_{re.escape(pre)}(.+)_([0-9.]+)M\.txt$', os.path.basename(f))
    s = os.path.join(d, f'sample_{pre}{m.group(1)}_{m.group(2)}M.txt')
    a, b = read(f), (read(s) if os.path.exists(s) else None)
    if a and b: rows.setdefault(m.group(1), []).append((float(m.group(2)), b, a))
print(f'# 무학습 재평가 {pre} — 표본 행동 vs 평균 행동 (기록용, 판정 아님. 시드 1개 = 단독 주장 금지)')
print('| run | 체크포인트 | 도착 % 표본 / 평균 (차) | 배 충돌 % 표본 / 평균 | epReward 표본 / 평균 |')
print('|---|---|---|---|---|')
for run in sorted(rows):
    for x, b, a in sorted(rows[run]):
        print(f'| {run} | {x:g}M | {b[0]:.1f} / {a[0]:.1f} ({a[0]-b[0]:+.1f}) | {b[1]:.1f} / {a[1]:.1f} | {b[2]:.0f} / {a[2]:.0f} |')
print('\n## 마지막 K 점 평균 (갈래 = 분기 뒤 체크포인트 전부, trunk = 2M 이후 전부)')
print('| run | K | 도착 % 표본 / 평균 | 배 충돌 % 표본 / 평균 | epReward 표본 / 평균 |')
print('|---|---|---|---|---|')
for run in sorted(rows):
    pts = sorted(rows[run]); k = len(pts)
    mb = [sum(p[1][i] for p in pts) / k for i in range(3)]; ma = [sum(p[2][i] for p in pts) / k for i in range(3)]
    print(f'| {run} | {k} | {mb[0]:.1f} / {ma[0]:.1f} | {mb[1]:.1f} / {ma[1]:.1f} | {mb[2]:.0f} / {ma[2]:.0f} |')
PYEOF
DROP=$HOME/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-02_fig1/out
[ -d "$DROP" ] && { mkdir -p "$DROP/meaneval_${PRE%_}"; cp "$D"/*.txt "$D"/*.md "$DROP/meaneval_${PRE%_}/" 2>/dev/null; }
echo "$(date '+%F %T') 끝 → $D/_meaneval_${PRE}.md"
