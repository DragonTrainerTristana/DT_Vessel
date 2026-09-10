"""reward vs MSG_DIM (single-seed dim sweep, 200k, comm ON). raw + normalized(/baseline)."""
import re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = os.path.join(os.path.dirname(__file__), "logs")
OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "figures"))
os.makedirs(OUTDIR, exist_ok=True)
BASE = 2.0  # 옛 baseline (사용자 회상 ~2점대). 정규화 기준.

dims = [2, 4, 6, 8, 10, 12]
final, lateavg = [], []
for d in dims:
    rew = []
    for ln in open(os.path.join(LOG, "sweep_dim%d.log" % d), encoding="utf-8", errors="ignore"):
        m = re.search(r"STEP ([\d,]+)/200,000.*Avg Reward: ([-\d.]+)", ln)
        if m:
            rew.append((int(m.group(1).replace(",", "")), float(m.group(2))))
    final.append(rew[-1][1])
    late = [r for s, r in rew if s >= 150000]
    lateavg.append(sum(late)/len(late) if late else rew[-1][1])

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# raw
ax[0].plot(dims, lateavg, "o-", color="#1f77b4", label="reward (late avg, 150k+)")
ax[0].plot(dims, final, "s--", color="#9ecae1", label="reward (final)")
for x, v in zip(dims, lateavg): ax[0].text(x, v+0.03, "%.2f" % v, ha="center", fontsize=9)
ax[0].set_xlabel("MSG_DIM"); ax[0].set_ylabel("Avg Reward")
ax[0].set_title("(1) Reward vs MSG_DIM (raw)\nno monotonic increase (flat/slightly down)")
ax[0].set_xticks(dims); ax[0].legend(); ax[0].grid(alpha=0.3)

# normalized by baseline 2.0
norm = [v/BASE for v in lateavg]
ax[1].plot(dims, norm, "o-", color="#ff7f0e")
for x, v in zip(dims, norm): ax[1].text(x, v+0.015, "%.2fx" % v, ha="center", fontsize=9)
ax[1].axhline(1.0, color="gray", ls=":", label="baseline (=%.1f)" % BASE)
ax[1].set_xlabel("MSG_DIM"); ax[1].set_ylabel("reward / baseline(%.1f)" % BASE)
ax[1].set_title("(2) Normalized by ~%.0f baseline\n(constant divide: same shape, y rescaled)" % BASE)
ax[1].set_xticks(dims); ax[1].legend(); ax[1].grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(OUTDIR, "reward_vs_dim_20260527.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
print("SAVED:", out)
print("dim   :", dims)
print("late  :", ["%.2f" % v for v in lateavg])
print("/%.1f  :" % BASE, ["%.2f" % v for v in norm])
