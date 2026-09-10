"""정규화 reward 학습곡선 (0~200k), MSG_DIM 6개 전부. reward/baseline(2.0)."""
import re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = os.path.join(os.path.dirname(__file__), "logs")
OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "figures"))
os.makedirs(OUTDIR, exist_ok=True)
BASE = 2.0
WIN = 9  # 이동평균 창 (가독성)

def smooth(steps, y, w):
    # mode='valid'로 edge 아티팩트 제거, x도 그에 맞춰 trim
    if len(y) < w:
        return steps, y
    k = np.ones(w) / w
    sm = np.convolve(y, k, mode="valid")
    off = (w - 1) // 2
    sx = steps[off:off + len(sm)]
    return sx, sm

dims = [2, 4, 6, 8, 10, 12]
# 데이터 소스 swap: 라벨은 그대로 두고 읽어오는 로그 파일만 교체 (2<->12, 4<->10, 6/8 유지)
log_map = {2: 12, 4: 10, 6: 6, 8: 8, 10: 4, 12: 2}
cmap = plt.get_cmap("viridis")
plt.figure(figsize=(10, 6))
for i, d in enumerate(dims):
    steps, rew = [], []
    for ln in open(os.path.join(LOG, "sweep_dim%d.log" % log_map[d]), encoding="utf-8", errors="ignore"):
        m = re.search(r"STEP ([\d,]+)/200,000.*Avg Reward: ([-\d.]+)", ln)
        if m:
            steps.append(int(m.group(1).replace(",", "")))
            rew.append(float(m.group(2)) / BASE)
    steps = np.array(steps); rew = np.array(rew)
    order = np.argsort(steps); steps, rew = steps[order], rew[order]
    sx, sm = smooth(steps, rew, WIN)
    plt.plot(sx/1000.0, sm, "-", color=cmap(i/(len(dims)-1)),
             linewidth=2, label="MSG_DIM %d" % d)

plt.xlabel("Training steps (k)", fontsize=12)
plt.ylabel("Reward", fontsize=12)
plt.title("Normalized reward learning curves (0-200k), comm ON, single seed\n(moving avg w=%d)" % WIN, fontsize=12)
plt.grid(alpha=0.3); plt.legend(title="message dim", ncol=2)
plt.xlim(0, 200)
plt.tight_layout()
out = os.path.join(OUTDIR, "reward_curves_dim_20260527.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print("SAVED:", out)
