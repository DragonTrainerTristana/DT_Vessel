import math
DT=0.04; L=14.18316
def stop(v0, decel, drag, mult_thrust=0.3):
    v=v0; x=0; t=0
    while v>0.005 and t<2000:
        v=max(0.0, v-decel*DT)           # target 0 -> decel
        x+=v*DT; t+=DT
        v*= (1-drag*DT)                  # target<0.1 -> full drag
    return x,t
for (dec,drg,tag) in [(0.04,0.1,'agile'),(0.005,0.007,'imo cand A'),(0.004,0.005,'imo cand B'),(0.003,0.004,'imo cand C')]:
    for v0 in (0.8,1.0,1.8):
        x,t=stop(v0,dec,drg)
        print(f"{tag:12s} DECEL={dec} DRAG={drg} v0={v0}: stop {x:6.1f} m = {x/L:4.2f} L in {t:5.0f} s")
# T_lat6 (cooperative head-on, each ship offsets 6 m) for imo TD4L, RR3, v=1.0 and evasive90 rule
def lat_time(target, R_full=2*L, RR=3.0, v=1.0, maxrud=30.0):
    TF=v/(maxrud*math.pi/180*R_full)   # R constant formula (b)
    rud=0; h=0; x=0; z=0; t=0; cmd=maxrud
    while t<300:
        rud=rud+max(-RR*DT,min(RR*DT,cmd-rud))
        yaw=rud*TF
        hr=math.radians(h); x+=math.sin(hr)*v*DT; z+=math.cos(hr)*v*DT
        h+=yaw*DT; t+=DT
        if h>=90: cmd=0
        if abs(x)>=target: return t
    return None
for tgt in (6,12,14.18):
    print(f"imo TD4L RR3 v1.0: T_lat{tgt:g} = {lat_time(tgt):.1f} s")
print("head-on W at 56 m, closing 2.0 m/s:", 56/2.0, "s ; crossing W:", round(56/math.sqrt(2),1), "s")
