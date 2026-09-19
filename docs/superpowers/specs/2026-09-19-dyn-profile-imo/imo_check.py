import math
DT=0.04; ACCEL=0.1; DECEL=0.04; MAXTR=30.0; DRAG=0.1; DRAGM=0.3; THR=0.1
L=14.18316; DEG=math.pi/180
def mt(a,b,d): return a+max(-d,min(d,b-a))
class S:
    def __init__(s,vm,rr,tf): s.x=s.z=s.h=0.0; s.v=vm; s.r=0.0; s.cmd=0.0; s.tg=vm; s.vm=vm; s.rr=rr; s.tf=tf
    def sub(s):
        s.v=mt(s.v,s.tg,ACCEL*DT if s.tg>s.v else DECEL*DT); s.v=max(0,min(s.vm,s.v))
        s.r=mt(s.r,s.cmd,s.rr*DT); yaw=s.r*(s.v/s.vm)*s.tf
        hr=s.h*DEG; s.x+=math.sin(hr)*s.v*DT; s.z+=math.cos(hr)*s.v*DT; s.h+=yaw*DT
        d=DRAG*DT*(DRAGM if s.tg>=THR else 1); s.v*=1-d
def tf_for(vm,R): return vm/(MAXTR*DEG*R)
for vm in (0.8,1.0,1.8):
    for rr in (3.0,12.0):
        tf=tf_for(vm,2*L)
        # 20/20 zigzag: rudder +20 until heading 20, then -20 until heading -20; record overshoot
        s=S(vm,rr,tf); s.cmd=20; ov1=None; ov2=None; t=0; phase=0; hmax=0; hmin=0
        while t<600:
            s.sub(); t+=DT
            if phase==0 and s.h>=20: phase=1; s.cmd=-20
            if phase==1:
                hmax=max(hmax,s.h)
                if s.h<=-20: phase=2; s.cmd=20; ov1=hmax-20
            if phase==2:
                hmin=min(hmin,s.h)
                if s.h>=20: ov2=-20-hmin; break
        # 10/10 zigzag
        s=S(vm,rr,tf); s.cmd=10; o1=None; t=0; phase=0; hmax=0
        while t<600:
            s.sub(); t+=DT
            if phase==0 and s.h>=10: phase=1; s.cmd=-10
            if phase==1:
                hmax=max(hmax,s.h)
                if s.h<=-10: o1=hmax-10; break
        # stopping: target 0, no astern
        s=S(vm,rr,tf); s.tg=0.0; t=0
        while s.v>0.005 and t<2000: s.sub(); t+=DT
        LV=L/vm
        lim10 = 10 if LV<10 else (20 if LV>=30 else 5+0.5*LV)
        print(f"vm={vm} rr={rr} tf={tf:.4f} L/V={LV:.1f}s  10/10 ov1={o1:.1f}° (lim {lim10:.1f})  20/20 ov1={ov1:.1f}° ov2={ov2:.1f}° (lim 25)  stop track={s.z:.1f}m={s.z/L:.2f}L t={t:.0f}s (lim 15L)")
