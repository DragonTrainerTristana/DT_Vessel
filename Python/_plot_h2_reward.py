# H2 MSG_DIM 2~12 reward-over-training 그래프 (2026-06-24). 실제 episode_logs.csv(col1=step,col2=avg_reward)만 사용.
# ★조작 없음: 실측 reward 그대로. 6/8/10/12 붕괴면 붕괴한 대로 그린다.
import os, glob, csv, sys, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass

MODELS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models"))
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "figures", "h2_v1"))
os.makedirs(OUT, exist_ok=True)

# 06-24 새벽 완주한 6개 run = launch 순서(DATE_TIME 18:08:40~18:09:00) → MSG_DIM 2,4,6,8,10,12
logs = []
for p in glob.glob(os.path.join(MODELS, "**", "csv_logs", "episode_logs.csv"), recursive=True):
    mt = os.path.getmtime(p)
    # 새 sweep: models 폴더명 VesselNavigation_20260622_1808xx (DATE_TIME), 06-24 새벽 완주
    if "VesselNavigation_20260622_1808" in p or "VesselNavigation_20260622_1809" in p:
        m = re.search(r'VesselNavigation_(\d+_\d+)', p)
        logs.append((m.group(1), p))
logs.sort()   # DATE_TIME 오름차순 = launch 순서
dims = [2,4,6,8,10,12]
assert len(logs) >= 6, f"기대 6개, 발견 {len(logs)}: {[l[0] for l in logs]}"
logs = logs[:6]

def smooth(y, w=15):
    if len(y) < w: return y
    return np.convolve(y, np.ones(w)/w, mode='valid')

plt.figure(figsize=(11,7))
colors = plt.cm.viridis(np.linspace(0,1,6))
print("DATE_TIME        -> MSG_DIM | last avg_reward (매핑 검증: 8/10/12가 낮으면 OK)")
for (dt, path), dim, col in zip(logs, dims, colors):
    steps=[]; rew=[]
    with open(path) as f:
        for r in csv.reader(f):
            if len(r) < 3: continue
            try:
                steps.append(float(r[1])); rew.append(float(r[2]))
            except ValueError:
                continue
    steps=np.array(steps); rew=np.array(rew)
    ys = smooth(rew); xs = steps[len(steps)-len(ys):]
    plt.plot(xs, ys, color=col, lw=2, label=f"MSG_DIM={dim}")
    print(f"  {dt}  -> {dim:2d}      | {np.mean(rew[-20:]):.1f}")

plt.xlabel("training step"); plt.ylabel("avg reward (per interval, smoothed w=15)")
plt.title("H2: reward vs training — MSG_DIM 2~12 (2M, single seed s42, ACTUAL data)")
plt.legend(); plt.grid(alpha=0.3)
outpath = os.path.join(OUT, "h2_reward_msgdim.png")
plt.savefig(outpath, dpi=120, bbox_inches='tight')
print(f"\nsaved: {outpath}")
