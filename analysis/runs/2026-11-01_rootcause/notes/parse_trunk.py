import re,sys
A='analysis/runs/2026-10-02_fig1/out'
runs=[('qd','s43'),('qd2','s44'),('qd3','s45'),('qd4','s46'),('qd5','s47')]
def parse(p):
    ppo={};evb={};grad={};on={}
    cur=None
    for ln in open(p,encoding='utf-8',errors='replace'):
        m=re.search(r'\[ppo\] upd=(\d+) approx_kl=([\d.e-]+) clipfrac=([\d.]+) entropy=([-\d.]+) logstd0_raw=\[([^\]]+)\] lr=([\d.e-]+) dec=([\d.]+)M',ln)
        if m:
            u=int(m.group(1)); ppo[u]=dict(kl=float(m.group(2)),cf=float(m.group(3)),ent=float(m.group(4)),ls=[float(x) for x in m.group(5).split()],lr=float(m.group(6)),dec=float(m.group(7)))
        m=re.search(r'\[ppo\] EVb\(lam=0.95\)=([-\d.]+) EVb\(lam=1\)=([-\d.]+) ret_std=([\d.]+) T=\d+ dec=([\d.]+)M',ln)
        if m: evb[float(m.group(4))]=(float(m.group(1)),float(m.group(2)),float(m.group(3)))
        m=re.search(r'\[grad\] upd=(\d+)\s+norm\s+ctr=([\d.]+)\s+cri=([\d.]+)\s+msg=([\d.]+).*total=([\d.]+)\s+\|\s+clip=([\d.]+)',ln)
        if m: grad[int(m.group(1))]=dict(ctr=float(m.group(2)),cri=float(m.group(3)),msg=float(m.group(4)),tot=float(m.group(5)),clip=float(m.group(6)))
        m=re.search(r'\[ON\] dec=([\d.]+)M \| ep=(\d+).*goal=([\d.]+)% vColl=([\d.]+)% oColl=([\d.]+)% TO=([\d.]+)% \| R=([-\d.]+).*epRet=([-\d.]+)',ln)
        if m: on[float(m.group(1))]=dict(ep=int(m.group(2)),goal=float(m.group(3)),vc=float(m.group(4)),oc=float(m.group(5)),to=float(m.group(6)),R=float(m.group(7)),epR=float(m.group(8)))
    return ppo,evb,grad,on
D={}
for d,s in runs:
    D[s]=parse(f'{A}/{d}/{d}_trunk_d6_{s}.log')
mode=sys.argv[1] if len(sys.argv)>1 else 'ppo'
if mode=='ppo':
    # bin by ~0.5M: average over updates
    import statistics as st
    bins=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9.1)]
    for key in ['kl','cf','ent','ls0','lsmin','ctr','cri','clip','evb','evb1','rstd']:
        print('==',key)
        for s in D:
            ppo,evb,grad,on=D[s]
            row=[]
            for a,b in bins:
                if key in('kl','cf','ent'):
                    v=[x[key] for x in ppo.values() if a<=x['dec']<b]
                elif key=='ls0':
                    v=[x['ls'][0] for x in ppo.values() if a<=x['dec']<b]
                elif key=='lsmin':
                    v=[min(x['ls']) for x in ppo.values() if a<=x['dec']<b]
                elif key in('ctr','cri','clip'):
                    v=[grad[u][key] for u,x in ppo.items() if a<=x['dec']<b and u in grad]
                else:
                    i={'evb':0,'evb1':1,'rstd':2}[key]
                    v=[t[i] for dd,t in evb.items() if a<=dd<b]
                row.append(f'{st.mean(v):8.4f}' if v else '   -    ')
            print(s,' '.join(row))
elif mode=='on':
    for s in D:
        ppo,evb,grad,on=D[s]
        print('==',s)
        print(' '.join(f"{d:.2f}:{v['goal']:.0f}/{v['vc']:.0f}/{v['oc']:.0f}/{v['to']:.0f}" for d,v in sorted(on.items())))
elif mode=='ls':
    for s in D:
        ppo,evb,grad,on=D[s]
        print('==',s)
        for u in sorted(ppo):
            if u%10==0 or u==max(ppo): print(u, ppo[u]['dec'], ppo[u]['ls'], ppo[u]['ent'], ppo[u]['kl'], ppo[u]['cf'])
if mode=='per':
    # per 5-update block: ctr, cri, kl, cf, ent, ls0, evb1, window goal nearest
    for s in D:
        ppo,evb,grad,on=D[s]
        print('==',s, 'blk dec | ctr cri | kl cf | ent ls0 | evb1 rstd | goal vc R epR')
        ev=sorted(evb.items())
        for b in range(0,138,5):
            us=[u for u in range(b,b+5) if u in ppo]
            if not us: continue
            m=lambda f: sum(f(u) for u in us)/len(us)
            d0=ppo[us[0]]['dec']; d1=ppo[us[-1]]['dec']
            e=[t for dd,t in ev if d0-0.01<=dd<=d1+0.01]
            o=[v for dd,v in on.items() if d0-0.01<=dd<=d1+0.01]
            print(f"{d1:5.2f} | {m(lambda u:grad[u]['ctr']):5.2f} {m(lambda u:grad[u]['cri']):6.2f} | {m(lambda u:ppo[u]['kl']):.4f} {m(lambda u:ppo[u]['cf']):.3f} | {m(lambda u:ppo[u]['ent']):6.3f} {m(lambda u:ppo[u]['ls'][0]):6.3f} | {sum(t[1] for t in e)/max(1,len(e)):.3f} {sum(t[2] for t in e)/max(1,len(e)):6.1f} | "+(' '.join(f"{v['goal']:.0f} {v['vc']:.0f} {v['R']:.3f} {v['epR']:.0f}" for v in o)))
if mode=='dec':
    import math
    bins=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9.1)]
    print('enc(=msg) / ctr_head / cri_head  (근사: 미니배치 평균 norm 으로 sqrt(a²-b²))')
    for s in D:
        ppo,evb,grad,on=D[s]
        row=[]
        for a,b in bins:
            us=[u for u,x in ppo.items() if a<=x['dec']<b and u in grad]
            m=lambda k: sum(grad[u][k] for u in us)/len(us)
            e=m('msg'); c=m('ctr'); r=m('cri')
            row.append(f"{e:4.1f}/{math.sqrt(max(0,c*c-e*e)):4.1f}/{math.sqrt(max(0,r*r-e*e)):5.1f}")
        print(s,' | '.join(row))
