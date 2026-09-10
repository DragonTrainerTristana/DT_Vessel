# mt10(maxTurnRate=10) 전용 ON vs OFF 분석 — 폴더를 mt10만 명시적으로 잡음(어제 데이터 혼입 방지).
# 17열: 3 steps,4 fuel,6 compliance,8 commandVar,12 headingTravel,13 minDCPA,14 dcpaBelow,15 fuelThrust,16 fuelTurn
import csv, glob, os
import numpy as np

RES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results"))
C = {'steps':3,'fuel':4,'compliance':6,'commandVar':8,'minVD':9,'straightness':11,
     'headingTravel':12,'minDCPA':13,'dcpaBelow':14,'fuelThrust':15,'fuelTurn':16}

def load(prefix, frac=0.25):
    rows=[]
    for s in (42,43,44):
        d=glob.glob(os.path.join(RES, f'*_losL1_{prefix}_s{s}'))   # ★mt10 명시
        if not d: continue
        with open(os.path.join(sorted(d)[-1],'metric.csv'),encoding='utf-8',errors='ignore') as f:
            rr=[r for r in csv.reader(f) if len(r)>=17]
        rows += rr[int(len(rr)*(1-frac)):]
    return rows

on, off = load('commON'), load('commOFF')
print(f"행수: ON={len(on)}, OFF={len(off)}  (17열 확인)")
if not on or not off:
    raise SystemExit("데이터 부족")

# 위협 조우 분리
allb=np.concatenate([[float(r[C['dcpaBelow']]) for r in on],[float(r[C['dcpaBelow']]) for r in off]])
med=np.median(allb)

def stat(rows, name, goal=False, pos=False, threat=None):
    v=[]
    for r in rows:
        if goal and r[2]!='goal': continue
        if threat is not None:
            tb=float(r[C['dcpaBelow']])
            if threat and tb<med: continue
            if not threat and tb>=med: continue
        x=float(r[C[name]])
        if pos and x<0: continue
        v.append(x)
    return np.mean(v) if v else float('nan')

def block(rows_on, rows_off, threat, title):
    print(f"\n=== {title} ===")
    specs=[('goal_rate',None),('fuel','↓'),('fuelThrust','↓'),('fuelTurn','↓'),
           ('headingTravel','↓'),('commandVar','↓'),('compliance','↑'),
           ('minDCPA','↑'),('steps','↓')]
    for name,good in specs:
        if name=='goal_rate':
            on_g=[r for r in rows_on if (threat is None or (float(r[C['dcpaBelow']])>=med)==threat)]
            off_g=[r for r in rows_off if (threat is None or (float(r[C['dcpaBelow']])>=med)==threat)]
            o=100*sum(1 for r in on_g if r[2]=='goal')/max(len(on_g),1)
            f=100*sum(1 for r in off_g if r[2]=='goal')/max(len(off_g),1)
            print(f"  {'goal%':16}{o:>10.2f}{f:>10.2f}{o-f:>+9.2f}")
            continue
        gg = name in ('fuel','fuelThrust','fuelTurn','headingTravel','commandVar','steps')
        o=stat(rows_on,name,goal=gg,pos=(name=='minDCPA'),threat=threat)
        f=stat(rows_off,name,goal=gg,pos=(name=='minDCPA'),threat=threat)
        win='ON' if ((o>f)==(good=='↑')) else ('OFF' if good else '-')
        print(f"  {name:16}{o:>10.2f}{f:>10.2f}{o-f:>+9.2f}   {good or ''} {win}")

block(on, off, None, "전체 (last-25%, maxTR=10)")
block(on, off, True, f"위협 조우만 (dcpaBelow>={med:.0f})")

# ★서사 핵심 판정
print("\n★ 서사 판정 (위협 조우, goal 에피소드):")
ftO=stat(on,'fuelTurn',goal=True,threat=True); ftF=stat(off,'fuelTurn',goal=True,threat=True)
thO=stat(on,'fuelThrust',goal=True,threat=True); thF=stat(off,'fuelThrust',goal=True,threat=True)
print(f"  fuelTurn:   ON={ftO:.1f} OFF={ftF:.1f}  ratio={ftO/ftF:.3f}  {'✓ ON<OFF(rudder 살짝)' if ftO<ftF else '✗ ON>OFF'}")
print(f"  fuelThrust: ON={thO:.1f} OFF={thF:.1f}  ratio={thO/thF:.3f}  {'✓ ≈동일(thrust 그대로)' if abs(thO/thF-1)<0.05 else '△ 차이있음'}")
print(f"  → 서사('rudder만 살짝, thrust 그대로'): {'성립' if ftO<ftF and abs(thO/thF-1)<0.08 else '미성립/부분'}")

# seed-paired headingTravel/fuelTurn 부호
print("\n★ seed-paired 부호 (위협조우, goal):")
for name,good in [('headingTravel','↓'),('fuelTurn','↓'),('fuelThrust','≈'),('commandVar','↓')]:
    signs=[]
    for s in (42,43,44):
        do=glob.glob(os.path.join(RES,f'*_losL1_commON_s{s}')); df=glob.glob(os.path.join(RES,f'*_losL1_commOFF_s{s}'))
        if not do or not df: continue
        def rd(d):
            with open(os.path.join(d,'metric.csv'),encoding='utf-8',errors='ignore') as f:
                rr=[r for r in csv.reader(f) if len(r)>=17]; return rr[int(len(rr)*0.75):]
        ro,rf=rd(sorted(do)[-1]),rd(sorted(df)[-1])
        vo=np.mean([float(r[C[name]]) for r in ro if r[2]=='goal' and float(r[C['dcpaBelow']])>=med])
        vf=np.mean([float(r[C[name]]) for r in rf if r[2]=='goal' and float(r[C['dcpaBelow']])>=med])
        if good=='≈': signs.append(f"{vo/vf:.2f}")
        else: signs.append('ON' if (vo<vf)==(good=='↓') else 'OFF')
    print(f"  {name:16}{good}  {signs}")
