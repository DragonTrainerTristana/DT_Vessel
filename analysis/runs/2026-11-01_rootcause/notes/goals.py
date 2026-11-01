import numpy as np, itertools
S=np.array([(250,250),(150,250),(50,250),(-50,250),(-150,250),(-250,250),(-250,150),(-250,50),(-250,-50),(-250,-150),(-250,-250),(-150,-250),(-50,-250),(50,-250),(150,-250),(250,-250),(250,-150),(250,-50),(250,50),(250,150)],float)
G=np.array([(200,200),(100,200),(0,200),(-100,200),(-200,200),(-200,100),(-200,0),(-200,-100),(-200,-200),(-100,-200),(0,-200),(100,-200),(200,-200),(200,-100),(200,0),(200,100)],float)
d=np.linalg.norm(S[:,None]-G[None],axis=-1); V=d>=400
print("valid goals per spawn:",V.sum(1).tolist())
L=[d[i][V[i]] for i in range(20)]; allL=np.concatenate(L); print("trip len min/med/max",allL.min(),np.median(allL),allL.max())
# min distance of straight path to arena centre
c=[]
for i in range(20):
    for g in G[V[i]]:
        a=S[i]; u=(g-a)/np.linalg.norm(g-a); c.append(abs(a[0]*u[1]-a[1]*u[0]))
c=np.array(c); print("path dist to centre: median",np.median(c),"frac<100m",(c<100).mean())
rng=np.random.default_rng(0); sh=[]
for _ in range(20000):
    sp=rng.permutation(20)[:16]; gs=[rng.choice(np.where(V[i])[0]) for i in sp]
    u,cnt=np.unique(gs,return_counts=True); sh.append((cnt[cnt>1]).sum()/16)
print("frac ships whose goal is shared with >=1 other (initial snapshot):",np.mean(sh))
