"""정규화 reward vs MSG_DIM (단독). reward/baseline(2.0). 단일 seed dim sweep 200k comm ON."""
import re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = os.path.join(os.path.dirname(__file__), "logs")
OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "figures"))
os.makedirs(OUTDIR, exist_ok=True)
BASE = 2.0

dims = [2, 4, 6, 8, 10, 12]
lateavg = []
for d in dims:
    rew = []
    for ln in open(os.path.join(LOG, "sweep_dim%d.log" % d), encoding="utf-8", errors="ignore"):
        m = re.search(r"STEP ([\d,]+)/200,000.*Avg Reward: ([-\d.]+)", ln)
        if m:
            rew.append((int(m.group(1).replace(",", "")), float(m.group(2))))
    late = [r for s, r in rew if s >= 150000]
    lateavg.append(sum(late)/len(late) if late else rew[-1][1])
norm = [v/BASE for v in lateavg]

plt.figure(figsize=(8, 5.5))
plt.plot(dims, norm, "o-", color="#ff7f0e", linewidth=2, markersize=8)
for x, v in zip(dims, norm):
    plt.text(x, v+0.012, "%.2f" % v, ha="center", fontsize=10)
plt.axhline(1.0, color="gray", ls=":", label="baseline (reward=%.1f)" % BASE)
plt.xlabel("MSG_DIM (message dimension)", fontsize=12)
plt.ylabel("Normalized reward (reward / %.1f)" % BASE, fontsize=12)
plt.title("Normalized reward vs MSG_DIM  (comm ON, 200k, single seed)\nNO monotonic increase with dimension", fontsize=12)
plt.xticks(dims); plt.grid(alpha=0.3); plt.legend()
plt.ylim(0, max(norm)*1.15)
plt.tight_layout()
out = os.path.join(OUTDIR, "reward_normalized_20260527.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print("SAVED:", out)
print("dim :", dims)
print("norm:", ["%.2f" % v for v in norm])
