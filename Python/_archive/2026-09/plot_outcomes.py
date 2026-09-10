"""
Reliable-metric (direct outcome logging) result visualization.
3 panels: (A) MaxStep 4000 vs 12000  (B) MSG_DIM sweep  (C) comm ON vs OFF (multi-seed)
Saves to Vessel_MLAgent/figures/ (outside Assets, not committed)
"""
import csv, os, statistics as st
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = os.path.join(os.path.dirname(__file__), "logs")
OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "figures"))
os.makedirs(OUTDIR, exist_ok=True)

def rates(tag):
    path = os.path.join(LOG, "outcomes_%s.csv" % tag)
    if not os.path.exists(path):
        return None
    rows = [r[2] for r in csv.reader(open(path)) if len(r) >= 4]
    n = len(rows)
    if n == 0:
        return None
    c = Counter(rows)
    f = lambda k: 100.0 * c.get(k, 0) / n
    return dict(goal=f("goal"), vessel=f("collision_vessel"),
               obstacle=f("collision_obstacle"), timeout=f("timeout"), n=n)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
COL = {"goal": "#2ca02c", "vessel": "#d62728", "timeout": "#7f7f7f"}

# ---------- Panel A: MaxStep 4000 vs 12000 ----------
a = ax[0]
ms = {"MaxStep 4000\n(before)": rates("rel2_off"), "MaxStep 12000\n(after)": rates("maxstep12k")}
labels = list(ms.keys()); metrics = ["goal", "vessel", "timeout"]
x = range(len(labels)); w = 0.25
for i, m in enumerate(metrics):
    vals = [ms[l][m] for l in labels]
    a.bar([xx + (i-1)*w for xx in x], vals, w, label=m, color=COL[m])
    for xx, v in zip(x, vals):
        a.text(xx + (i-1)*w, v+0.8, "%.0f" % v, ha="center", fontsize=9)
a.set_xticks(list(x)); a.set_xticklabels(labels)
a.set_ylabel("% of episodes"); a.set_title("(A) MaxStep fix -> navigation works\n(scale 0.2, comm OFF)")
a.legend(); a.set_ylim(0, 90)

# ---------- Panel B: MSG_DIM sweep ----------
b = ax[1]
dims = [2, 4, 6, 8, 10, 12]; dg = [rates("dim%d" % d) for d in dims]
b.plot(dims, [r["goal"] for r in dg], "o-", color=COL["goal"], label="goal %")
b.plot(dims, [r["vessel"] for r in dg], "s-", color=COL["vessel"], label="vessel collision %")
b.plot(dims, [r["timeout"] for r in dg], "^-", color=COL["timeout"], label="timeout %")
b.set_xlabel("MSG_DIM (message dimension)"); b.set_ylabel("% of episodes")
b.set_title("(B) MSG_DIM sweep (comm ON, 200k)\nhigher dim != better (goal down, collision flat)")
b.legend(); b.set_xticks(dims); b.set_ylim(0, 70); b.grid(alpha=0.3)

# ---------- Panel C: comm ON vs OFF (multi-seed mean +/- std) ----------
c = ax[2]
seeds = [42, 43, 44]
off = [rates("off%d" % s) for s in seeds]; on = [rates("on%d" % s) for s in seeds]
mets = ["goal", "vessel", "timeout"]; metlabel = {"goal": "goal", "vessel": "vessel collision", "timeout": "timeout"}
xo = range(len(mets)); w2 = 0.35
off_mean = [st.mean([r[m] for r in off]) for m in mets]; off_std = [st.pstdev([r[m] for r in off]) for m in mets]
on_mean = [st.mean([r[m] for r in on]) for m in mets]; on_std = [st.pstdev([r[m] for r in on]) for m in mets]
c.bar([xx - w2/2 for xx in xo], off_mean, w2, yerr=off_std, capsize=4, label="comm OFF", color="#1f77b4")
c.bar([xx + w2/2 for xx in xo], on_mean, w2, yerr=on_std, capsize=4, label="comm ON", color="#ff7f0e")
for xx, v in zip(xo, off_mean): c.text(xx - w2/2, v+1.2, "%.1f" % v, ha="center", fontsize=9)
for xx, v in zip(xo, on_mean): c.text(xx + w2/2, v+1.2, "%.1f" % v, ha="center", fontsize=9)
c.set_xticks(list(xo)); c.set_xticklabels([metlabel[m] for m in mets])
c.set_ylabel("% of episodes (mean +/- std, 3 seeds)")
c.set_title("(C) comm ON vs OFF (MaxStep 12000)\ncomm: collision unchanged, goal WORSE")
c.legend(); c.set_ylim(0, 70)

plt.tight_layout()
out = os.path.join(OUTDIR, "comm_analysis_20260527.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
print("SAVED:", out)
print("[A] 4000: goal=%.0f timeout=%.0f | 12000: goal=%.0f timeout=%.0f"
      % (ms[labels[0]]["goal"], ms[labels[0]]["timeout"], ms[labels[1]]["goal"], ms[labels[1]]["timeout"]))
print("[C] OFF goal=%.1f vessel=%.1f | ON goal=%.1f vessel=%.1f"
      % (off_mean[0], off_mean[1], on_mean[0], on_mean[1]))
