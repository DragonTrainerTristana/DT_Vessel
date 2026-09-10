# 사용자 핵심 목표(COLREGs·smooth·fuel·time)를 위협 조우 에피소드에서 ON vs OFF 정밀 비교.
# metric 15열: 5 rudderVar, 6 compliance, 8 commandVar, 9 minVD, 11 straightness, 12 headingTravel, 13 minDCPA, 14 dcpaBelow
import csv, glob, os
import numpy as np

RES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results"))
COLS = {'rudderVar':5, 'compliance':6, 'commandVar':8, 'minVD':9, 'straightness':11,
        'headingTravel':12, 'minDCPA':13, 'dcpaBelow':14, 'steps':3, 'fuel':4}

def load(d, frac=0.25):
    rows=[]
    with open(os.path.join(d,'metric.csv'),encoding='utf-8',errors='ignore') as f:
        for r in csv.reader(f):
            if len(r)>=15: rows.append(r)
    return rows[int(len(rows)*(1-frac)):]

def arm(prefix):
    out=[]
    for s in (42,43,44):
        out += load(glob.glob(os.path.join(RES,f'*{prefix}_s{s}'))[0])
    return out

on, off = arm('commON'), arm('commOFF')

def col(rows, name, goal_only=False, pos_only=False):
    i=COLS[name]; v=[]
    for r in rows:
        if goal_only and r[2]!='goal': continue
        x=float(r[i])
        if pos_only and x<0: continue
        v.append(x)
    return np.array(v)

# dcpaBelow median으로 위협有/無 분리
allb=np.concatenate([col(on,'dcpaBelow'),col(off,'dcpaBelow')])
med=np.median(allb)

def threat(rows): return [r for r in rows if float(r[COLS['dcpaBelow']])>=med]

print(f"=== 전체 (3seed last-25%) — smooth·COLREGs 중심 ===")
print(f"{'metric':16}{'ON':>11}{'OFF':>11}{'diff':>10}{'방향':>8}")
specs = [('compliance',False,False,'↑'),('straightness',True,False,'↑'),
         ('headingTravel',True,False,'↓'),('commandVar',True,False,'↓'),
         ('rudderVar',True,False,'↓'),('fuel',True,False,'↓'),
         ('steps',True,False,'↓'),('minDCPA',False,True,'↑')]
for name,g,p,good in specs:
    o,f=col(on,name,g,p).mean(),col(off,name,g,p).mean()
    win='ON' if ((o>f)==(good=='↑')) else 'OFF'
    print(f"{name:16}{o:>11.3f}{f:>11.3f}{o-f:>+10.3f}{good+' '+win:>8}")

print(f"\n=== 위협 조우(dcpaBelow≥{med:.0f}) 에피소드만 ===")
ton, toff = threat(on), threat(off)
print(f"  ON n={len(ton)}, OFF n={len(toff)}")
for name,g,p,good in specs:
    o = np.array([float(r[COLS[name]]) for r in ton if (not g or r[2]=='goal') and (not p or float(r[COLS[name]])>=0)]).mean()
    f = np.array([float(r[COLS[name]]) for r in toff if (not g or r[2]=='goal') and (not p or float(r[COLS[name]])>=0)]).mean()
    win='ON' if ((o>f)==(good=='↑')) else 'OFF'
    print(f"{name:16}{o:>11.3f}{f:>11.3f}{o-f:>+10.3f}{good+' '+win:>8}")

# seed-paired 부호 일관성 (smooth·compliance만)
print(f"\n=== seed-paired 부호 (ON-OFF, 위협조우) ===")
for name,g,p,good in [('compliance',False,False,'↑'),('straightness',True,False,'↑'),
                       ('headingTravel',True,False,'↓'),('commandVar',True,False,'↓'),('fuel',True,False,'↓')]:
    signs=[]
    for s in (42,43,44):
        ro=threat(load(glob.glob(os.path.join(RES,f'*commON_s{s}'))[0]))
        rf=threat(load(glob.glob(os.path.join(RES,f'*commOFF_s{s}'))[0]))
        vo=np.array([float(r[COLS[name]]) for r in ro if (not g or r[2]=='goal') and (not p or float(r[COLS[name]])>=0)]).mean()
        vf=np.array([float(r[COLS[name]]) for r in rf if (not g or r[2]=='goal') and (not p or float(r[COLS[name]])>=0)]).mean()
        d=vo-vf; better=(d>0)==(good=='↑')
        signs.append('ON' if better else 'OFF')
    print(f"  {name:16} {good}  s42={signs[0]:4} s43={signs[1]:4} s44={signs[2]:4}  → ON승 {signs.count('ON')}/3")
