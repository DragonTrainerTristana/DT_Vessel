# 2026-10-26 A–Z 진단 §L6-2: 사전등록 판정식(같은 시드 짝 전부 승 AND |평균차| > N) 통과 확률 MC. 가정 SD 는 f2_·qb_ 로그에서 추정한 값.
import random,statistics as st
def p(eff,n,offb=True,floor=0.0,T=20000,ss=11.6,bs=8.3):
    ok=0
    for _ in range(T):
        off=[];com=[];nb=[]
        for i in range(n):
            s=random.gauss(0,ss); o=s+random.gauss(0,bs); c=s+eff+random.gauss(0,bs)
            off.append(o);com.append(c)
            if offb: nb.append(abs(s+random.gauss(0,bs)-o))
        w=all(c>o for o,c in zip(off,com))
        N=max(max(off)-min(off),(sum(nb)/len(nb)) if nb else 0,floor)
        ok+= w and abs(st.mean(com)-st.mean(off))>N
    return ok/T
random.seed(0)
for n in (1,3,5,10): print('eff17.3',n,p(17.3,n))
print('eff3.3 n3',p(3.3,3)); print('eff0 n1 nooffb',p(0,1,False)); print('eff0 n3 offb',p(0,3))
