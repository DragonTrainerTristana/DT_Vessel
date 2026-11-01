# 2-hull kinematic hand calc with imo constants copied from config.py (c034422). Not vessel_gym.
import math, numpy as np
DT, SUB = 0.04, 10
R_FULL = 2*14.18316; RR = 3.0; MAXR = 30.0; STEER = 10.0
HL, HB = 14.18316/2, 1.92741/2
def corners(p, h):
    f = np.array([math.sin(h), math.cos(h)]); s = np.array([math.cos(h), -math.sin(h)])
    return [p + a*HL*f + b*HB*s for a in (1,-1) for b in (1,-1)], (f, s)
def overlap(p1,h1,p2,h2):
    c1,ax1 = corners(p1,h1); c2,ax2 = corners(p2,h2)
    for ax in ax1+ax2:
        a=[c@ax for c in c1]; b=[c@ax for c in c2]
        if max(a)<min(b) or max(b)<min(a): return False
    return True
class Ship:
    def __init__(s,p,hdg,v): s.p=np.array(p,float); s.h=math.radians(hdg); s.v=v; s.rud=0.0; s.course=s.h
    def sub(s):
        err = (math.degrees(s.course-s.h)+180)%360-180
        cmd = max(-1,min(1,err/STEER))*MAXR
        s.rud += max(-RR*DT, min(RR*DT, cmd-s.rud))
        yaw = (s.rud/MAXR)*s.v/R_FULL
        s.p = s.p + np.array([math.sin(s.h),math.cos(s.h)])*s.v*DT
        s.h += yaw*DT
def run(a,b,act_a,act_b,T=200.0):
    # act_* = (start_time_s, course_change_deg  +=starboard)
    t=0.0; dmin=1e9; coll=False; done_a=done_b=False; a0=a.h; b0=b.h
    while t<T:
        if not done_a and t>=act_a[0]: a.course=a0+math.radians(act_a[1]); done_a=True
        if not done_b and t>=act_b[0]: b.course=b0+math.radians(act_b[1]); done_b=True
        for _ in range(SUB):
            a.sub(); b.sub(); t+=DT
            d=np.linalg.norm(a.p-b.p); dmin=min(dmin,d)
            if d<15 and overlap(a.p,a.h,b.p,b.h): coll=True
        rel=b.p-a.p; rv=np.array([math.sin(b.h),math.cos(b.h)])*b.v-np.array([math.sin(a.h),math.cos(a.h)])*a.v
        if rel@rv>0 and np.linalg.norm(rel)>80: break
    return coll,dmin
def headon(D,v,dpa,dpb,delay=0.4):
    a=Ship((0,0),0,v); b=Ship((0,D),180,v)
    return run(a,b,(delay,dpa),(delay,dpb))
def cross(D,v,dpa,dpb,delay=0.4):
    # a heading north, b heading west coming from a's starboard; both reach origin-ish at same time (worst case)
    # place at equal distance d from conflict point so that separation = D
    d=D/math.sqrt(2); a=Ship((0,-d),0,v); b=Ship((d,0),270,v)
    return run(a,b,(delay,dpa),(delay,dpb))
def min_D(fn,v,dpa,dpb,crit):
    lo,hi=8.0,400.0
    ok=lambda D: crit(*fn(D,v,dpa,dpb))
    if not ok(hi): return float('nan')
    for _ in range(30):
        m=(lo+hi)/2
        if ok(m): hi=m
        else: lo=m
    return hi
nocoll=lambda c,d: not c
dc12=lambda c,d: (not c) and d>=12.0
dc24=lambda c,d: (not c) and d>=24.0
print("v  | headon TTC from 56m(no act) s | cross TTC s")
for v in (0.8,1.3,1.8):
    print(v, round((56-2*HL)/(2*v),1), round((56/math.sqrt(2)-HL)/v,1))
print("\n[headon, zero offset] min start distance D (m) for: no-collision / DCPA>=12 / DCPA>=24 ; maneuver = course change")
for v in (0.8,1.3,1.8):
    for lab,dpa,dpb in (("both +20",20,20),("both +45",45,45),("both +90",90,90),("one +45",45,0),("one +90",90,0)):
        r=[min_D(headon,v,dpa,dpb,c) for c in (nocoll,dc12,dc24)]
        print(f"v={v} {lab:9s}", [round(x,1) for x in r])
print("\n[crossing 90deg, same arrival] give-way=a(+starboard), stand-on b holds")
for v in (0.8,1.3,1.8):
    for lab,dpa,dpb in (("gw +30",30,0),("gw +60",60,0),("gw +90",90,0),("gw -60(port)",-60,0),("both: gw+60 so+?",60,-30)):
        r=[min_D(cross,v,dpa,dpb,c) for c in (nocoll,dc12,dc24)]
        print(f"v={v} {lab:16s}", [round(x,1) for x in r])
