import re,sys,statistics as st
OUT=sys.argv[1]
fs={'s43':'qd/qd_trunk_d6_s43.log','s44':'qd2/qd2_trunk_d6_s44.log','s45':'qd3/qd3_trunk_d6_s45.log','s46':'qd4/qd4_trunk_d6_s46.log','s47':'qd5/qd5_trunk_d6_s47.log'}
for s,f in fs.items():
    try: L=open(f"{OUT}/{f}",errors='ignore').read().splitlines()
    except Exception as e: print(s,'missing',e); continue
    kl=[];ls=[];cri=[];ctr=[]
    for l in L:
        m=re.search(r'\[ppo\] upd=(\d+) approx_kl=([\d.]+).*logstd0_raw=\[([-\d. ]+)\]',l)
        if m: kl.append((int(m.group(1)),float(m.group(2)))); ls.append(float(m.group(3).split()[0]))
        m=re.search(r'\[grad\] upd=(\d+)\s+norm\s+ctr=([\d.]+)\s+cri=([\d.]+)',l)
        if m: ctr.append(float(m.group(2))); cri.append(float(m.group(3)))
    k=[v for _,v in kl]
    if not k: print(s,'no ppo'); continue
    # per-quarter mean kl
    n=len(k); q=[st.mean(k[i*n//4:(i+1)*n//4]) for i in range(4)]
    print(s,'n',n,'kl min/med/max %.4f %.4f %.4f'%(min(k),st.median(k),max(k)),'quarter means',['%.4f'%x for x in q],'kl>0.02:',sum(x>0.02 for x in k),
          'logstd0 first/last %.3f %.3f'%(ls[0],ls[-1]),'cri max %.2f med %.3f'%(max(cri),st.median(cri)) if cri else '')
