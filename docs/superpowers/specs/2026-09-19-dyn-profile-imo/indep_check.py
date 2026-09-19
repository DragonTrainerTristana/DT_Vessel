# 독립 재구현: 스크립트 구조와 무관하게 해석식 + 미세 dt 적분으로 재산출
import math
DEG=math.pi/180; L=14.18316
def lat(v,RR,TF,tmax,dt=0.001,ret90=False):
    # 타 램프: r(t)=min(30,RR*t); yaw=TF*r (speed=v_max 정상 가정)
    x=z=h=0.0; t=0.0; r=0.0; returned=False
    while t<tmax-1e-9:
        if ret90 and not returned:
            pend=TF*r*r/(2*RR)
            if h+pend>=90: returned=True
        cmd=0.0 if returned else 30.0
        r += max(-RR*dt,min(RR*dt,cmd-r))
        x += v*math.sin(h*DEG)*dt; z += v*math.cos(h*DEG)*dt
        h += TF*r*dt; t+=dt
    return x,z,h
def t_lat(v,RR,TF,target,ret90=True,dt=0.001):
    x=z=h=0.0; t=0.0; r=0.0; returned=False
    while t<400:
        if ret90 and not returned and h+TF*r*r/(2*RR)>=90: returned=True
        cmd=0.0 if returned else 30.0
        r += max(-RR*dt,min(RR*dt,cmd-r))
        x += v*math.sin(h*DEG)*dt; h += TF*r*dt; t+=dt
        if x>=target: return t
    return None
# agile
TF=1.5; v=1.0
R=v/(30*TF*DEG); print('agile v1.0 steady R,D,D/L', R, 2*R, 2*R/L)
for vm in (0.8,1.8): print(' agile v',vm,'D/L',2*vm/(30*TF*DEG)/L)
# imo TD4L RR3 v1.0
R=4*L/2; TF=v/(30*DEG*R); print('imo TF',TF,'yaw_full',30*TF)
x,z,h=lat(v,3.0,TF,28.0); print('imo TD4L RR3 v1.0 x(28s)=',x,'heading',h)
T12=t_lat(v,3.0,TF,12.0); print('T_lat12',T12,'W(HO56)=',56/(2*v),'W/T=',56/(2*v)/T12)
T14=t_lat(v,3.0,TF,L); print('T_lat14',T14,'W/T',28/T14)
T12r=t_lat(v,12.0,v/(30*DEG*R),12.0); print('RR12 T_lat12',T12r)
# 해석 근사: 램프 10s 동안 h=TF*RR*t^2/2, 이후 h=yaw_full*(t-5)
print('h(28) analytic', 30*TF*(28-5))
