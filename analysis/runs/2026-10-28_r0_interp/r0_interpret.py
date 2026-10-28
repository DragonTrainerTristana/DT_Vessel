# 2026-10-28 R0 해석 표 생성(학습·평가 없음). 입력 = analysis 4832f69 의 out/meaneval_qb·meaneval_qa + out/diag_trunk_n(n_ 비교값)
# 사용: python3 r0_interpret.py <out 폴더>   (out = analysis/runs/2026-10-02_fig1/out)
import math, os, re, sys
out = sys.argv[1]
def rd(p):
    return open(p, encoding='utf-8', errors='replace').read() if os.path.exists(p) else None
def ev(t):
    if not t: return None
    g = re.search(r'goal=\s*([0-9.]+)%\s+vColl=\s*([0-9.]+)%\s+oColl=\s*([0-9.]+)%\s+TO=\s*([0-9.]+)%\s+\|\s*(\d+) eps', t)
    r = re.search(r'epReward=\s*(-?[0-9.]+)', t); L = re.search(r'\[goal-ep\].*?len=\s*(\d+)', t)
    fv = re.search(r'첫 변침\(\|δ̄\|>0\.2\)까지 중앙\s*([0-9.]+)s', t); off = re.search(r'오프셋\| 중앙\s*([0-9.]+)deg', t)
    return dict(goal=float(g[1]), vcoll=float(g[2]), to=float(g[4]), eps=int(g[5]), rew=float(r[1]), len=int(L[1]) if L else None,
                turn=float(fv[1]) if fv else None, offset=float(off[1]) if off else None)
def logstd(p):
    d = {}
    for line in (rd(p) or '').splitlines():
        m = re.search(r'step([0-9.]+)M\.pt.*?logstd0=([-0-9. ]+?) mean', line) or re.search(r'(_s43)\.pt.*?logstd0=([-0-9. ]+?) mean', line)
        if m: d[m[1]] = [float(v) for v in m[2].split()]
    return d
sig = lambda l: 120.0 * math.exp(l)
rows = []
for pre, d in (('qb_', 'meaneval_qb'), ('qa_', 'meaneval_qa')):
    ls = logstd(os.path.join(out, d, f'logstd_{pre}trunk_d6.txt'))
    for x in ('2', '4', '6', '8', '9.04397'):
        s = ev(rd(os.path.join(out, d, f'sample_{pre}trunk_d6_{x}M.txt'))); m = ev(rd(os.path.join(out, d, f'mean_{pre}trunk_d6_{x}M.txt')))
        l = ls.get(x) or ls.get('_s43')
        rows.append((pre, x, sig(l[0]) if l else None, s, m))
for x, f in (('4', 'eval_{}_4M.txt'), ('9.04397', 'eval_{}_9M.txt')):
    s = ev(rd(os.path.join(out, 'diag_trunk_n', f.format('sample')))); m = ev(rd(os.path.join(out, 'diag_trunk_n', f.format('mean'))))
    rows.append(('n_', x, None, s, m))
print('# R0 해석 표 (2026-10-28, 학습·평가 없음, 시드 43 하나 — 기록용)\n')
print('## 1. trunk 체크포인트별: None 전문가 σ · 표본/평균 도착 · 평균 행동 세부')
print('| trunk | 체크포인트 | None 전문가 σ 침로 ° | 도착 % 표본 / 평균 (차) | 평균: 배 충돌 % | 평균: 시간초과 % | 평균: epReward | 평균: 도착 ep 길이 | 평균: 첫 변침 s | 평균: 무조우 오프셋 ° |')
print('|---|---|---|---|---|---|---|---|---|---|')
for pre, x, sg, s, m in rows:
    f = lambda v, fmt='{:.1f}': '–' if v is None else fmt.format(v)
    print(f"| {pre} | {x}M | {f(sg)} | {s['goal']:.1f} / {m['goal']:.1f} ({m['goal']-s['goal']:+.1f}) | {m['vcoll']:.1f} | {m['to']:.1f} | {m['rew']:.0f} | {f(m['len'],'{:d}')} | {f(m['turn'])} | {f(m['offset'])} |")
print('\n## 2. qb_ 갈래: 평균 행동 off vs comm (체크포인트별)')
print('| 체크포인트 | off 도착 / 충돌 % | comm 도착 / 충돌 % | comm − off 도착 |')
print('|---|---|---|---|')
for x in ('10', '12', '14', '16'):
    o = ev(rd(os.path.join(out, 'meaneval_qb', f'mean_qb_off_{x}M.txt'))); c = ev(rd(os.path.join(out, 'meaneval_qb', f'mean_qb_comm_{x}M.txt')))
    print(f"| {x}M | {o['goal']:.1f} / {o['vcoll']:.1f} | {c['goal']:.1f} / {c['vcoll']:.1f} | {c['goal']-o['goal']:+.1f} |")
