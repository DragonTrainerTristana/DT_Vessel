# 어제 2M 완주 데이터에서 "몇 step window부터 ON vs OFF 결론이 안정됐나" 역분석.
# 누적 step 구간별(0-250k, 250-500k, ...)로 ON vs OFF 핵심지표를 보고, 언제부터 부호/크기가 수렴했는지.
import csv, glob, os
import numpy as np

RES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results"))
# 어제 maxTR=30 run (2M 완주)
PAT_ON  = "20260612_1826*_commON_s{}"
PAT_OFF = "20260612_18*_commOFF_s{}"
COLS = {'steps':3, 'fuel':4, 'compliance':6, 'commandVar':8, 'headingTravel':12, 'minDCPA':13, 'dcpaBelow':14}

def load_with_cumstep(prefix):
    """run의 metric을 읽고, 각 에피소드에 누적 agent-step(근사) 부여. 행 순서=시간순."""
    rows = []
    for s in (42,43,44):
        d = glob.glob(os.path.join(RES, prefix.format(s)))
        if not d: continue
        with open(os.path.join(d[0],'metric.csv'),encoding='utf-8',errors='ignore') as f:
            rr = [r for r in csv.reader(f) if len(r)>=15]
        # 누적 step: 에피소드 step수 누적 / 16 agent → training step 근사
        cum = 0
        for r in rr:
            cum += int(r[3])
            rows.append((cum/16.0, r))   # (approx training step, row)
    return rows

on, off = load_with_cumstep(PAT_ON), load_with_cumstep(PAT_OFF)

def window_stats(rows, lo, hi):
    sel = [r for st,r in rows if lo <= st < hi]
    if len(sel) < 50: return None
    g = [r for r in sel if r[2]=='goal']
    f = lambda i,gg=False,pos=False: np.array([float(r[i]) for r in (g if gg else sel) if (not pos or float(r[i])>=0)])
    return {
        'n': len(sel),
        'goal%': 100*len(g)/len(sel),
        'fuel': f(4,True).mean() if g else np.nan,
        'headingTravel': f(12,True).mean() if g else np.nan,
        'commandVar': f(8,True).mean() if g else np.nan,
        'compliance': f(6).mean(),
        'minDCPA': f(13,pos=True).mean(),
        'dcpaBelow': f(14).mean(),
    }

# 250k 구간별 ON vs OFF diff 추이
print("구간별 ON−OFF diff (어제 maxTR=30, 2M 완주) — 언제부터 결론 안정?")
print(f"{'step구간':16}{'goal%':>9}{'headTrav':>10}{'cmdVar':>9}{'compl':>8}{'minDCPA':>9}{'dcpaBel':>9}")
bounds = [(0,250e3),(250e3,500e3),(500e3,750e3),(750e3,1e6),(1e6,1.25e6),(1.25e6,1.5e6),(1.5e6,1.75e6),(1.75e6,2.1e6)]
for lo,hi in bounds:
    so,sf = window_stats(on,lo,hi), window_stats(off,lo,hi)
    if not so or not sf: continue
    label = f"{int(lo/1000)}-{int(hi/1000)}k"
    d = {k: so[k]-sf[k] for k in ('goal%','headingTravel','commandVar','compliance','minDCPA','dcpaBelow')}
    print(f"{label:16}{d['goal%']:>+9.1f}{d['headingTravel']:>+10.0f}{d['commandVar']:>+9.1f}{d['compliance']:>+8.3f}{d['minDCPA']:>+9.2f}{d['dcpaBelow']:>+9.0f}")

print("\n해석: headingTravel(ON 부드러움)·commandVar 부호가 어느 구간부터 안정되는지 = 최소 필요 학습량.")
