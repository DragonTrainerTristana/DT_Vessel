import csv, math, sys
O='analysis/runs/2026-10-02_fig1/out'
B=9043968
def rd(p):
    out={}
    for r in csv.DictReader(open(p)):
        s=int(r['step'])
        if s>B: out[s]=(int(r['n_ep']),float(r['goal']),float(r['vColl']))
    return out
for d,pre,s in [('qd','qd','43'),('qd2','qd2','44'),('qs','qs','43'),('qs2','qs2','44')]:
    off=rd(f'{O}/{d}/{pre}_off_s{s}_ep.csv'); com=rd(f'{O}/{d}/{pre}_comm_s{s}_ep.csv')
    st=sorted(set(off)&set(com))
    diffs=[com[k][1]-off[k][1] for k in st]
    n=len(diffs); m=sum(diffs)/n; sd=math.sqrt(sum((x-m)**2 for x in diffs)/(n-1))
    # lag-1 autocorr
    ac=sum((diffs[i]-m)*(diffs[i+1]-m) for i in range(n-1))/sum((x-m)**2 for x in diffs)
    neff=n*(1-ac)/(1+ac) if ac<1 else n
    pool=lambda D,ks:(sum(D[k][0]*D[k][1] for k in ks)/sum(D[k][0] for k in ks), sum(D[k][0] for k in ks))
    po,no=pool(off,st); pc,nc=pool(com,st)
    # per-update expected binomial sd of a diff with ~50 eps each
    ne=sum(off[k][0] for k in st)/n
    binsd=math.sqrt(2*0.7*0.3/ne)*100
    # chunks of ~27 updates (4 chunks) and last 16 updates (~1M dec)
    q=n//4; ch=[sum(diffs[i*q:(i+1)*q])/q for i in range(4)]
    l16=sum(diffs[-16:])/16
    win=sum(1 for x in diffs if x>0)
    print(f"{d} s{s}: n_upd={n} eps/upd~{ne:.0f} | per-upd diff mean {m:+.2f} sd {sd:.2f} (binomial-only sd {binsd:.1f}) lag1 ac {ac:.2f} | comm>off {win}/{n} | pooled goal off {po:.1f} (n={no}) comm {pc:.1f} (n={nc}) | 4 chunks {['%+.1f'%c for c in ch]} | last16 {l16:+.1f}")
