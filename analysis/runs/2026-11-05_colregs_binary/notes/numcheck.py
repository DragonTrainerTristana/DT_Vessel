import re,glob,os,statistics as st
import os; O=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '2026-10-02_fig1', 'out')
P={43:'qr',44:'qr2',45:'qr3'}
R={}
for s,p in P.items():
  for a in ('off','comm'):
    t=open(f'{O}/{p}/lrmb_{p}_{a}_s{s}.txt').read()
    L=re.search(r'\[binary-COLREGs\] rate=\s*([\d.]+)% \(n=(\d+).*?양보=\s*([\d.]+)%\(n=(\d+)\) 정면=\s*([\d.]+)%\(n=(\d+)\) 추월=\s*([\d.]+)%\(n=(\d+)\) 유지=\s*([\d.]+)%\(n=(\d+)\)',t).groups()
    v=list(map(float,L)); R[(s,a)]=v
    rate,N=v[0],v[1]; roles=[(v[2],v[3]),(v[4],v[5]),(v[6],v[7]),(v[8],v[9])]
    rec=sum(r*n for r,n in roles)/N
    contrib=[r*n/N for r,n in roles]
    nc=[(100-r)*n/N for r,n in roles]
    print(s,a,'rate',rate,'recon %.3f diff %.3f'%(rec,rate-rec),'contrib',['%.2f'%c for c in contrib],'Y+H %.2f'%(contrib[0]+contrib[1]),'S+O %.2f'%(contrib[2]+contrib[3]),'nc share YH %.2f'%(100*(nc[0]+nc[1])/sum(nc)))
# 56m pair drops
for lab in ('양보-유지 쌍결정','HeadOn 쌍결정'):
  ds=[];no=nc_=0
  for s,p in P.items():
    vals={}
    for a in ('off','comm'):
      t=open(f'{O}/{p}/lrm_{p}_{a}_s{s}.txt').read()
      vals[a]=int(re.search(r'\[조율/레이더 안 ≤56m\] '+lab+r' n=(\d+)',t).group(1))
    ds.append(100*(vals['comm']/vals['off']-1)); no+=vals['off']; nc_+=vals['comm']
  print(lab,['%.1f'%d for d in ds],'mean %.1f pooled %.1f'%(st.mean(ds),100*(nc_/no-1)))
