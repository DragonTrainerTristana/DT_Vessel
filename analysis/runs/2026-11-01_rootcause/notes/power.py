# 5-seed paired decision MC. Assumptions printed. No torch.
import numpy as np
rng=np.random.default_rng(0)
TCRIT={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,10:2.262,12:2.201,15:2.145,20:2.093}  # keyed by n (df=n-1), two-sided .975
def rule33(d):  # spec 2026-10-30 §3-3: wins>=4/5 (scaled: >= ceil(0.8n)) and 95% t CI excludes 0 (sign = d>0 good)
    n=d.shape[1]; m=d.mean(1); s=d.std(1,ddof=1); lo=m-TCRIT[n]*s/np.sqrt(n)
    w=(d>0).sum(1); return (w>=int(np.ceil(0.8*n)))&(lo>0)
def ruleN(d,off,floor):  # summarize_fig1: all pairs win and |mean|>max(OFF range, floor)
    rngoff=off.max(1)-off.min(1); return (d>0).all(1)&(np.abs(d.mean(1))>np.maximum(rngoff,floor))
T=200000
print("== rule §3-3 (n seeds, wins>=0.8n & CI lo>0), d~N(mu,sd)")
for sd in (3,5,7):
    row=[]
    for mu in (0,2,4,6,8,10):
        d=rng.normal(mu,sd,(T,5)); row.append(f"mu{mu}:{rule33(d).mean():.2f}")
    print(f"sd{sd} n5 ", ' '.join(row))
print("== seeds needed (rule33) for mu=6")
for sd in (3,5,7):
    out=[]
    for n in (3,5,7,10,12,15,20):
        if n-1 not in TCRIT: continue
        d=rng.normal(6,sd,(T,n)); out.append(f"n{n}:{rule33(d).mean():.2f}")
    print(f"sd{sd}",' '.join(out))
print("== seeds needed (rule33) for mu=3 (stage2-like/Woerner-like small effect)")
for sd in (2,3,5):
    out=[]
    for n in (5,10,15,20):
        d=rng.normal(3,sd,(T,n)); out.append(f"n{n}:{rule33(d).mean():.2f}")
    print(f"sd{sd}",' '.join(out))
print("== heavy tail: 1 of 5 seeds from collapsed trunk, its diff ~N(mu,20)")
for sd in (3,5):
    row=[]
    for mu in (4,6,8):
        d=rng.normal(mu,sd,(T,5)); d[:,4]=rng.normal(mu,20,T); row.append(f"mu{mu}:{rule33(d).mean():.2f}")
    print(f"sd{sd}",' '.join(row))
print("== N rule (summarize_fig1) at 5 seeds, goal floor 0 / vColl floor 12.1; OFF level ~N(72,sdoff)")
for sdoff in (3,5):
    for floor in (0.0,12.1):
        row=[]
        for mu in (0,4,6,8,12):
            d=rng.normal(mu,5,(T,5)); off=rng.normal(72,sdoff,(T,5)); row.append(f"mu{mu}:{ruleN(d,off,floor).mean():.3f}")
        print(f"sdoff{sdoff} floor{floor}",' '.join(row))
print("== N rule at 1 seed (current 1-seed tables): goal floor 0 -> P(pass|mu=0), vColl floor 12.1 -> P(pass|mu)")
for mu in (0,2,6,10,14):
    d=rng.normal(mu,5,(T,1)); off=rng.normal(72,3,(T,1)); print(f"mu{mu} floor0 {ruleN(d,off,0.0).mean():.2f} floor12.1 {ruleN(d,off,12.1).mean():.2f}")
print("== per-metric 5-seed rule33 using 2-seed stage-1 estimates (mu, sd) -- sd from 2 seeds = very uncertain")
for lab,mu,sd in [("goal mean16M",6.05,5.0),("vColl mean16M(-)",6.2,5.4),("goal lr sample",8.8,4.1),("DCPA mean16M",3.65,0.5),("DCPA if sd3",3.65,3.0),("Woerner lr",0.8,3.7),("head mean16M(-)",5.5,20.5),("stage2 goal lrm",0.95,0.2),("stage2 goal lrm sd2",0.95,2.0)]:
    d=rng.normal(mu,sd,(T,5)); print(f"{lab:22s} mu={mu} sd={sd} P(pass)={rule33(d).mean():.2f}")
