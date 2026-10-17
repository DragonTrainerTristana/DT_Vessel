# 2026-10-13: 배치 qb_ phase1(T-B trunk → 코덱 → 갈래 off · comm → 주 평가 → 고정 장면 곡선) 결과 PDF. 새 학습·평가 없음.
# 숫자는 전부 로그에서 읽음(runs/2026-10-02_fig1/out/qb/):
#   lr_qb_{off,comm}_s43.txt        주 평가 + Woerner·타 줄 (summarize.parse_all)
#   qb_{trunk_d6,off,comm}_s43.log  학습 창(update 5개) · curve_qb_*_<x>M.txt 체크포인트별 고정 장면 평가
#   _fig1.md · _curve.md · _codec_choice.txt · _branch_check.txt · _gate.log · _timeline.txt · _batch_env.txt
#   ../q_stage1_notes/runnote_phase1.txt (실행 기록)
# 판정 규칙 = summarize_fig1.py 와 같은 식(아래 judge) → _fig1.md 의 '주 지표 통과' 수와 대조. 곡선 M = curve_fig1.py 와 같은 식 → _curve.md 와 대조.
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_fig1_qb.py [출력 폴더]
import glob, io, os, re, subprocess, sys

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.normpath(os.path.join(HERE, '..'))
RUNS = os.path.normpath(os.path.join(F1, '..'))
SC = os.path.join(RUNS, '2026-10-01_scripted')
sys.path.insert(0, os.path.join(RUNS, '2026-09-30_t'))
sys.path.insert(0, SC)
import summarize as S  # noqa: E402

D = os.path.join(F1, 'out', 'qb')
NOTES = os.path.join(F1, 'out', 'q_stage1_notes')
PRE, SEED = 'qb_', 43
NAME = '2026-10-13_q배치_qb_phase1_통신vsOFF_결과'
BRANCH_AT, TOTAL, UPD = 9043968, 16056320, 65536
RUN_MAIN = 'burnin=2400 dec=10000 drain=0 ring=1.0 crossing=0 seed=999 envs=256 vessels=16'
RUN_CURVE = 'burnin=2400 dec=3000 drain=3000 ring=1.0 crossing=0 seed=999 envs=256 vessels=16'
# summarize_fig1.py 와 같은 표(키, 이름, 좋은 방향, 고정 하한)
MAIN = [('vColl', '배끼리 충돌 %', -1, 12.1), ('goal', '도착 %', +1, 0.0), ('dcpa', 'DCPA(통과거리) m', +1, 1.5),
        ('w_score', 'COLREGs(Woerner) %', +1, 0.0), ('fuel', '연료(도착 ep)', -1, 0.0), ('fleet_fuel', '도착 1회당 함대 연료', -1, 0.0)]
POOL_KEYS = ('goal', 'vColl', 'dcpa')
AUX = [('head', '방향 바꾼 총량 °', -1, 0.0), ('rud_trav', '타 이동량 °/ep', -1, 0.0), ('rud_abs', '평균 타각 크기 °', -1, 0.0),
       ('C', '옛 COLREGs %', +1, 0.0), ('TO', '시간초과 %', -1, 0.0), ('oColl', '벽 충돌 %', -1, 0.0), ('epReward', 'epReward', +1, 0.0)]
ND = lambda k: 0 if k in ('fuel', 'fleet_fuel', 'head', 'epReward') else 1  # noqa: E731
M_FRAC = 0.1


def text(*p, base=D):
    with io.open(os.path.join(base, *p), encoding='utf-8', errors='replace') as fh:
        return fh.read()


def f(v, nd=1, sign=False):
    if v is None:
        return 'n/a'
    s = f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'
    return s.replace('-', '−')


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ── 주 평가
def load_main(arm):
    fp = os.path.join(D, f'lr_{PRE}{arm}_s{SEED}.txt')
    txt = text(f'lr_{PRE}{arm}_s{SEED}.txt')
    mm = re.search(r'\[run\] (.*)', txt)
    assert mm and mm.group(1).strip().startswith(RUN_MAIN) and 'R8=' in txt, (fp, mm and mm.group(1))
    hd = re.search(r'\[eval\] ckpt=(\S+) .*? fields=(\w+) .*? partner_range=([\d.]+) ', txt)
    r = S.parse_all(fp)
    assert 'goal' in r and hd.group(1) == f'{PRE}{arm}_s{SEED}.pt', fp
    r['_fields'], r['_prange'] = hd.group(2), hd.group(3)
    return r


off, comm = load_main('off'), load_main('comm')
assert (off['_fields'], off['_prange']) == ('state', '56.0') and (comm['_fields'], comm['_prange']) == ('intent', '300.0'), (off['_fields'], comm['_fields'])
t_off = {}
for fp in sorted(glob.glob(os.path.join(SC, 'out', 'lr_t_off_s*.txt'))):
    r = S.parse_all(fp)
    if 'goal' in r:
        t_off[fp] = r
POOL = {}
for nm, runs in (('off', [off]), ('t_off', list(t_off.values()))):
    g = [r.get('goal') for r in runs if r.get('goal') is not None]
    if runs and g and sum(g) / len(g) >= 50.0:    # 건강한 OFF 만(goal ≥ 50)
        POOL[nm] = {k: mean([r.get(k) for r in runs]) for k in POOL_KEYS}


def judge(k, sg, floor):   # 시드 1개 · offb 없음 → N = max(OFF 시드 범위 0, 고정 하한)
    x, y = off.get(k), comm.get(k)
    if x is None or y is None:
        return None
    w = int((y - x) * sg > 0)
    N = max(0.0, floor)
    ok = w == 1 and abs(y - x) > N
    note, pok = '', None
    if k in POOL_KEYS and POOL:
        who, best = max(((nm, pm[k]) for nm, pm in POOL.items()), key=lambda t: sg * t[1])
        pok = sg * (y - best) > 0
        ok = ok and pok
        note = f"풀 최선 {who} {f(best)} {'넘음' if pok else '못 넘음'}"
    return dict(x=x, y=y, w=w, d=y - x, N=N, ok=ok, note=note)


J = {k: judge(k, sg, fl) for k, _, sg, fl in MAIN + AUX}
WIN_MAIN = sum(1 for k, *_ in MAIN if J[k]['ok'])
DIR_MAIN = sum(J[k]['w'] for k, *_ in MAIN)
DIR_AUX = sum(J[k]['w'] for k, *_ in AUX if J[k])
FIG1 = text('_fig1.md')
m = re.search(r'주 지표 통과: (\d+)/(\d+)', FIG1)
assert m and (int(m.group(1)), int(m.group(2))) == (WIN_MAIN, len(MAIN)), (m and m.groups(), WIN_MAIN)
assert f'같은 trunk 짝 1개: [{SEED}]' in FIG1

# ── 학습 창(update 5개 묶음)
WIN = re.compile(r'^\[ON\] dec=([\d.]+)M \| ep=(\d+) len~(\d+) \| goal=([\d.]+)% vColl=([\d.]+)% oColl=([\d.]+)% TO=([\d.]+)% '
                 r'\| R=(-?[\d.]+) \| (\d+) dec/s \| epRet=(-?[\d.]+)(?: \| lr=([\d.e+-]+))?')
KEYS = ('dec', 'ep', 'len', 'goal', 'vColl', 'oColl', 'TO', 'R', 'dps', 'epRet', 'lr')


def windows(name):
    rows = []
    for ln in text(name).splitlines():
        mm = WIN.match(ln)
        if mm:
            rows.append({k: float(x) for k, x in zip(KEYS, mm.groups())})
    return rows


def saved(name):
    mm = re.search(r'saved -> \S+ \(([\d.]+)M decisions, ([\d.]+)min\)', text(name))
    return float(mm.group(1)), float(mm.group(2))


wm = lambda rows, k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
TR = windows(f'{PRE}trunk_d6_s{SEED}.log')
BR = {a: windows(f'{PRE}{a}_s{SEED}.log') for a in ('off', 'comm')}
SV = {a: saved(f'{PRE}{a}_s{SEED}.log') for a in ('off', 'comm')}
assert len(TR) == 27 and all(len(BR[a]) == 21 and SV[a][0] == 16.06 for a in BR), (len(TR), {a: len(BR[a]) for a in BR}, SV)
for a in BR:
    lg = text(f'{PRE}{a}_s{SEED}.log')
    assert f'[resume] {PRE}trunk_d6_s{SEED}.pt' in lg and f'branch_at={BRANCH_AT}' in lg and '[bc]' not in lg, a
    assert BR[a][0]['lr'] < TR[-1]['lr'] and BR[a][-1]['lr'] < BR[a][0]['lr'], a   # LR 감쇠가 갈래에서 이어짐
SHA = set(re.findall(r'\[branch\] trunk=\S+ sha256=(\w+)', text(f'{PRE}off_s{SEED}.log') + text(f'{PRE}comm_s{SEED}.log')))
BCHK = text('_branch_check.txt')
assert len(SHA) == 1 and 'BRANCH CHECK: ALL PASS' in BCHK and list(SHA)[0] in BCHK
LAST = {a: BR[a][-4:] for a in BR}
TR_LAST = TR[-4:]
B3 = {a: [BR[a][0:7], BR[a][7:14], BR[a][14:21]] for a in BR}     # 갈래 창 7개씩 3구간(결과 보고 고르지 않으려고 3등분)
LOW = {a: min(BR[a], key=lambda w: w['goal']) for a in BR}
N_ABOVE = sum(1 for i in range(21) if BR['comm'][i]['goal'] > BR['off'][i]['goal'])
N_ABOVE_R = sum(1 for i in range(21) if BR['comm'][i]['R'] > BR['off'][i]['R'])

# ── 고정 장면 체크포인트 곡선
EVL = re.compile(r'goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%\s+oColl=\s*([\d.]+)%\s+TO=\s*([\d.]+)%\s+\|\s+(\d+) eps')
PTS = {}
for fp in glob.glob(os.path.join(D, f'curve_{PRE}*_*M.txt')):
    mm = re.match(rf'curve_{PRE}(.+)_([0-9.]+)M\.txt$', os.path.basename(fp))
    txt = io.open(fp, encoding='utf-8', errors='replace').read()
    e, g = re.search(r'epReward=\s*(-?[0-9.]+)', txt), EVL.search(txt)
    assert mm and e and g and RUN_CURVE in txt, fp
    PTS[(mm.group(1), float(mm.group(2)))] = dict(zip(('goal', 'vColl', 'oColl', 'TO', 'eps'), (float(x) for x in g.groups())), epReward=float(e.group(1)))
SER = {a: sorted((x, v) for (r, x), v in PTS.items() if r in (f'trunk_d6_s{SEED}', f'{a}_s{SEED}')) for a in ('off', 'comm')}
assert all(len(SER[a]) == 10 for a in SER) and len(PTS) == 15, (len(PTS), {a: len(SER[a]) for a in SER})


def m_check(c):   # curve_fig1.py 와 같은 식
    run_max, drop = c[0], 0.0
    for v in c:
        run_max = max(run_max, v)
        drop = max(drop, run_max - v)
    rise = max(c) - c[0]
    return drop, rise > 0 and drop <= M_FRAC * rise


CURVE_MD = text('_curve.md')
MC = {}
for a in SER:
    c = [v['epReward'] for _, v in SER[a]]
    drop, ok = m_check(c)
    MC[a] = (c[0], max(c), c[-1], drop, ok)
    row = f"| {a} | {SEED} | {len(c)} | {c[0]:.1f} | {max(c):.1f} | {c[-1]:.1f} | {drop:.1f} | {'우상향' if ok else '★불통과'} |"
    assert row in CURVE_MD, (row, CURVE_MD)
BRPT = {a: sorted((x, v) for (r, x), v in PTS.items() if r == f'{a}_s{SEED}') for a in SER}       # 10 · 12 · 14 · 16 · 16.06M
TRPT = sorted((x, v) for (r, x), v in PTS.items() if r == f'trunk_d6_s{SEED}')                    # 2 · 4 · 6 · 8 · 9.04M
assert len(TRPT) == 5 and all(len(BRPT[a]) == 5 for a in BRPT) and [x for x, _ in BRPT['off']] == [x for x, _ in BRPT['comm']]
for a in BRPT:   # step16M 체크포인트 = 끝 모델(runnote_phase1.txt: steps · 가중치 SHA 같음) → 두 점 값이 같아야 함
    assert BRPT[a][-2][1] == BRPT[a][-1][1], (a, BRPT[a][-2:])
DIST = {a: BRPT[a][:-2] + BRPT[a][-1:] for a in BRPT}     # 서로 다른 모델 4개(10 · 12 · 14 · 16.06M)
C_ABOVE = sum(1 for (x, vo), (_, vc) in zip(DIST['off'], DIST['comm']) if vc['goal'] > vo['goal'])
C_ABOVE_R = sum(1 for (x, vo), (_, vc) in zip(DIST['off'], DIST['comm']) if vc['epReward'] > vo['epReward'])
TR_END = TRPT[-1][1]
TR_BEST = max(TRPT, key=lambda t: t[1]['epReward'])
C_LOW = {a: min(DIST[a], key=lambda t: t[1]['goal']) for a in DIST}

# ── 코덱 · 경과
CODEC = text('_codec_choice.txt')
CH_K = int(re.search(r'CHOSEN_K=(\d+)', CODEC).group(1))
CK = {}
for mm in re.finditer(r"k(\d+) \{'heading_deg_p90': ([\d.]+), 'speed_ms_p90': ([\d.]+), 'intent_deg_p90': ([\d.]+), 'role_acc': ([\d.]+)", CODEC):
    CK[int(mm.group(1))] = [float(x) for x in mm.groups()[1:]]
CPASS = dict((int(k), v) for k, v in re.findall(r'k(\d+)=(통과|불통과)', CODEC))
assert set(CK) == set(CPASS) == {4, 6, 8, 12} and CPASS[CH_K] == '통과'
TLALL = [ln for ln in text('_timeline.txt').splitlines() if ln.startswith('2026-10-13')]
P1 = TLALL[next(i for i, ln in enumerate(TLALL) if 'trunk 모드 완료' in ln) + 1:]
assert f'배치 {PRE} phase1' in P1[0] and 'b6a4341' in P1[0] and any('phase1 완료' in ln for ln in P1) and not any('★' in ln for ln in P1), P1[:2]


def t_of(pat):
    hit = [ln for ln in P1 if ln[20:].startswith(pat)]
    assert len(hit) == 1, (pat, hit)
    return hit[0][11:19]


BENV = text('_batch_env.txt').strip()
NOTE = [ln for ln in text('runnote_phase1.txt', base=NOTES).splitlines()[1:] if ln.strip()]
GATE = re.search(rf'^s{SEED} goal=([\d.]+) vColl=([\d.]+) oColl=([\d.]+) 창=4 통과', text('_gate.log'), re.M)
assert GATE and f"{wm(TR_LAST, 'goal'):.1f}" == GATE.group(1)


# ── 그림(SVG, 로그 값 그대로)
def chart(series, y0, y1, yticks, nd, hlines=(), w=700, h=150, marks=False):
    ml, mr, mt, mb = 40, 8, 16, 18
    X = lambda d: ml + (w - ml - mr) * d / 16.4  # noqa: E731
    Y = lambda v: mt + (h - mt - mb) * (y1 - v) / (y1 - y0)  # noqa: E731
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='100%'>"]
    for t in yticks:
        s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' stroke='#e4e3df' stroke-width='0.6'/>")
        s.append(f"<text x='{ml - 4}' y='{Y(t) + 2.6:.1f}' font-size='7.2' text-anchor='end' fill='#52514e'>{f(t, nd)}</text>")
    for d in range(0, 17, 2):
        s.append(f"<text x='{X(d):.1f}' y='{h - 5}' font-size='7.2' text-anchor='middle' fill='#52514e'>{d}M</text>")
    s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{h - mb}' y2='{h - mb}' stroke='#0b0b0b' stroke-width='0.7'/>")
    bx = X(BRANCH_AT / 1e6)
    s.append(f"<line x1='{bx:.1f}' x2='{bx:.1f}' y1='{mt}' y2='{h - mb}' stroke='#52514e' stroke-width='0.7' stroke-dasharray='3 2'/>")
    s.append(f"<text x='{bx + 3:.1f}' y='{mt + 7}' font-size='7.2' fill='#52514e'>분기 9.04M</text>")
    for hv, hl in hlines:
        s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{Y(hv):.1f}' y2='{Y(hv):.1f}' stroke='#0b0b0b' stroke-width='0.6' stroke-dasharray='2 2'/>")
        s.append(f"<text x='{w - mr - 2}' y='{Y(hv) - 2.5:.1f}' font-size='7.2' text-anchor='end' fill='#0b0b0b'>{hl}</text>")
    lx = ml + 4
    for lab, pts, col, sw, dash in series:
        p = ' '.join(f"{X(d):.1f},{Y(min(max(v, y0), y1)):.1f}" for d, v in pts)
        da = f" stroke-dasharray='{dash}'" if dash else ''
        s.append(f"<polyline points='{p}' fill='none' stroke='{col}' stroke-width='{sw}'{da}/>")
        if marks:
            s += [f"<circle cx='{X(d):.1f}' cy='{Y(min(max(v, y0), y1)):.1f}' r='2.1' fill='{col}'/>" for d, v in pts]
        s.append(f"<line x1='{lx}' x2='{lx + 16}' y1='7' y2='7' stroke='{col}' stroke-width='{sw}'{da}/>")
        s.append(f"<text x='{lx + 19}' y='9.6' font-size='7.6' fill='#0b0b0b'>{lab}</text>")
        lx += 19 + 7.9 * len(lab) + 12
    return ''.join(s) + '</svg>'


C_TR, C_OFF, C_COMM = '#9c9a94', '#d9822b', '#1b6fd6'
wp = lambda rows, k: [(r['dec'], r[k]) for r in rows]  # noqa: E731
join = lambda rows, k: [(TR[-1]['dec'], TR[-1][k])] + wp(rows, k)  # noqa: E731
G_WIN = chart([('trunk 도착', wp(TR, 'goal'), C_TR, 1.4, ''), ('off 도착', join(BR['off'], 'goal'), C_OFF, 1.6, ''), ('comm 도착', join(BR['comm'], 'goal'), C_COMM, 1.6, '')],
              0, 100, [0, 20, 40, 60, 80, 100], 0, hlines=((30.0, 'trunk 관문 30 %'),))
G_WINV = chart([('trunk 배 충돌', wp(TR, 'vColl'), C_TR, 1.4, ''), ('off 배 충돌', join(BR['off'], 'vColl'), C_OFF, 1.6, ''), ('comm 배 충돌', join(BR['comm'], 'vColl'), C_COMM, 1.6, '')],
               0, 100, [0, 20, 40, 60, 80, 100], 0, h=128)
G_WINR = chart([('trunk 결정당 보상', wp(TR, 'R'), C_TR, 1.4, ''), ('off 결정당 보상', join(BR['off'], 'R'), C_OFF, 1.6, ''), ('comm 결정당 보상', join(BR['comm'], 'R'), C_COMM, 1.6, '')],
               -1.4, 0.4, [-1.2, -0.8, -0.4, 0.0, 0.4], 1, hlines=((0.0, '0'),), h=118)
cp = lambda pts, k: [(x, v[k]) for x, v in pts]  # noqa: E731
EPS = [v['epReward'] for v in PTS.values()]
E0, E1 = (min(EPS) // 200) * 200, (max(EPS) // 200 + 1) * 200
G_CEP = chart([('trunk epReward', cp(TRPT, 'epReward'), C_TR, 1.4, ''), ('off epReward', cp(TRPT[-1:] + BRPT['off'], 'epReward'), C_OFF, 1.6, ''),
               ('comm epReward', cp(TRPT[-1:] + BRPT['comm'], 'epReward'), C_COMM, 1.6, '')],
              E0, E1, [E0 + i * (E1 - E0) / 4 for i in range(5)], 0, marks=True, h=140)
G_CGO = chart([('trunk 도착', cp(TRPT, 'goal'), C_TR, 1.4, ''), ('off 도착', cp(TRPT[-1:] + BRPT['off'], 'goal'), C_OFF, 1.6, ''),
               ('comm 도착', cp(TRPT[-1:] + BRPT['comm'], 'goal'), C_COMM, 1.6, '')],
              0, 100, [0, 20, 40, 60, 80, 100], 0, marks=True, h=140)


def judge_table(rows):
    h = ("<table><thead><tr><th>지표</th><th>좋은 방향</th><th>off</th><th>comm</th><th>comm − off</th><th>comm 승/짝</th><th>N</th><th>판정</th></tr></thead><tbody>")
    for k, lab, sg, _ in rows:
        j = J[k]
        if not j:
            h += f"<tr><td>{lab}</td><td colspan='7' class='c'>n/a</td></tr>"
            continue
        cls = 'good' if j['w'] else 'bad'
        h += (f"<tr><td>{lab}</td><td class='c'>{'↑' if sg > 0 else '↓'}</td><td class='c'>{f(j['x'], ND(k))}</td><td class='c'>{f(j['y'], ND(k))}</td>"
              f"<td class='c {cls}'>{f(j['d'], 2 if ND(k) else 0, True)}</td><td class='c {cls}'>{j['w']}/1</td><td class='c'>{f(j['N'], 2)}</td>"
              f"<td class='c'>{'<b>통과</b>' if j['ok'] else '불통과'}{(' · ' + j['note']) if j['note'] else ''}</td></tr>")
    return h + "</tbody></table>"


def role_table():
    rows = [('w_giveway', 'Woerner 양보선'), ('w_headon', 'Woerner 정면'), ('w_overtaking', 'Woerner 추월'), ('w_standon', 'Woerner 유지선'),
            ('rks_v2', '판정기 v2 역할 지킴+안전 %'), ('safe_v2', '판정기 v2 안전 통과 %'), ('minPass', '통과 최소거리 m'), ('len', '에피소드 길이(결정)'), ('eps', '에피소드 수')]
    h = "<table><thead><tr><th>세부(참고 · 판정 아님)</th><th>off</th><th>comm</th><th>comm − off</th></tr></thead><tbody>"
    for k, lab in rows:
        x, y = off.get(k), comm.get(k)
        if x is None or y is None:
            continue
        nd = 0 if k in ('len', 'eps') else 1
        h += f"<tr><td>{lab}</td><td class='c'>{f(x, nd)}</td><td class='c'>{f(y, nd)}</td><td class='c'>{f(y - x, nd, True)}</td></tr>"
    return h + "</tbody></table>"


def blk_table():
    h = ("<table><thead><tr><th>구간(갈래 창 7개씩)</th><th>off 도착 %</th><th>comm 도착 %</th><th>off 배 충돌 %</th><th>comm 배 충돌 %</th>"
         "<th>off 결정당 R</th><th>comm 결정당 R</th></tr></thead><tbody>")
    h += (f"<tr><td class='c'>trunk 끝 4창(7.54–8.85M)</td><td class='c' colspan='2'>{f(wm(TR_LAST, 'goal'))}</td><td class='c' colspan='2'>{f(wm(TR_LAST, 'vColl'))}</td>"
          f"<td class='c' colspan='2'>{f(wm(TR_LAST, 'R'), 3)}</td></tr>")
    for i in range(3):
        bo, bc = B3['off'][i], B3['comm'][i]
        h += (f"<tr><td class='c'>{bo[0]['dec'] - 0.33:.2f}–{bo[-1]['dec']:.2f}M</td><td class='c'>{f(wm(bo, 'goal'))}</td><td class='c'>{f(wm(bc, 'goal'))}</td>"
              f"<td class='c'>{f(wm(bo, 'vColl'))}</td><td class='c'>{f(wm(bc, 'vColl'))}</td><td class='c'>{f(wm(bo, 'R'), 3)}</td><td class='c'>{f(wm(bc, 'R'), 3)}</td></tr>")
    h += (f"<tr class='g'><td class='c'>갈래 끝 4창</td><td class='c'>{f(wm(LAST['off'], 'goal'))}</td><td class='c'>{f(wm(LAST['comm'], 'goal'))}</td>"
          f"<td class='c'>{f(wm(LAST['off'], 'vColl'))}</td><td class='c'>{f(wm(LAST['comm'], 'vColl'))}</td><td class='c'>{f(wm(LAST['off'], 'R'), 3)}</td>"
          f"<td class='c'>{f(wm(LAST['comm'], 'R'), 3)}</td></tr>")
    return h + "</tbody></table>"


def curve_table():
    h = ("<table><thead><tr><th>체크포인트</th><th>런</th><th>도착 %</th><th>배 충돌 %</th><th>벽 충돌 %</th><th>시간초과 %</th><th>epReward</th><th>에피소드 수</th></tr></thead><tbody>")
    for x, v in TRPT:
        h += (f"<tr><td class='c'>{x:g}M</td><td class='c'>trunk</td><td class='c'>{f(v['goal'])}</td><td class='c'>{f(v['vColl'])}</td><td class='c'>{f(v['oColl'])}</td>"
              f"<td class='c'>{f(v['TO'])}</td><td class='c'>{f(v['epReward'], 0)}</td><td class='c'>{v['eps']:.0f}</td></tr>")
    for (x, vo), (_, vc) in zip(BRPT['off'], BRPT['comm']):
        for nm, v in (('off', vo), ('comm', vc)):
            h += (f"<tr><td class='c'>{x:g}M</td><td class='c'>{nm}</td><td class='c'>{f(v['goal'])}</td><td class='c'>{f(v['vColl'])}</td><td class='c'>{f(v['oColl'])}</td>"
                  f"<td class='c'>{f(v['TO'])}</td><td class='c'>{f(v['epReward'], 0)}</td><td class='c'>{v['eps']:.0f}</td></tr>")
    return h + "</tbody></table>"


def codec_table():
    h = ("<table><thead><tr><th>k</th><th>침로 오차 p90 °</th><th>속력 오차 p90 m/s</th><th>의도 오차 p90 °</th><th>역할 정확도</th><th>관문</th></tr></thead><tbody>")
    for k in (4, 6, 8, 12):
        hd, sp, it, ra = CK[k]
        cls = " class='g'" if k == CH_K else ''
        h += (f"<tr{cls}><td class='c'>{k}</td><td class='c'>{f(hd, 2)}</td><td class='c'>{f(sp, 3)}</td><td class='c'>{f(it, 2)}</td><td class='c'>{f(ra * 100, 2)} %</td>"
              f"<td class='c'>{CPASS[k]}{' · 선택' if k == CH_K else ''}</td></tr>")
    return h + "</tbody></table>"


LOSE = [lab for k, lab, *_ in MAIN if not J[k]['w']]
WINL = [lab for k, lab, *_ in MAIN if J[k]['w']]
END_O, END_C = SER['off'][-1][1], SER['comm'][-1][1]
verdict_h1a = DIR_MAIN < len(MAIN) / 2
# 본문 문장이 전제로 삼는 방향(값이 달라지면 문장을 고쳐야 하므로 여기서 멈춤)
assert WIN_MAIN == 0 and not MC['off'][4] and not MC['comm'][4]
assert END_O['goal'] < TR_END['goal'] and END_C['goal'] < TR_END['goal'] and END_O['epReward'] < TR_END['epReward'] and END_C['epReward'] < TR_END['epReward']
assert TR_BEST[0] == TRPT[0][0] and TR_END['epReward'] < TR_BEST[1]['epReward']            # trunk 고정 장면 최고 = 첫 점(2M)
assert all(LOW[a]['goal'] < wm(LAST[a], 'goal') for a in LOW)                               # 최저 창 뒤 끝 4창에서 일부 회복
assert wm(LAST['comm'], 'goal') > wm(LAST['off'], 'goal') and off['goal'] > comm['goal']    # 학습 끝 4창과 주 평가의 순서가 반대
assert (END_O['goal'] > END_C['goal']) == (off['goal'] > comm['goal'])                      # 끝 모델: 곡선 평가와 주 평가 순서 같음
assert all(wm(LAST[a], 'R') < wm(TR_LAST, 'R') for a in LAST)                               # 결정당 보상이 trunk 끝 4창보다 낮음

CSS = """
@page { size: A4; margin: 13mm 13mm 12mm 13mm; }
* { box-sizing: border-box; }
body { font-family: 'Malgun Gothic', sans-serif; color: #0b0b0b; font-size: 9.2pt; line-height: 1.42; margin: 0; }
h1 { font-size: 15pt; margin: 0 0 2px 0; }
.sub1 { color: #52514e; font-size: 8.6pt; margin-bottom: 8px; }
h2 { font-size: 10.8pt; margin: 11px 0 4px 0; padding-top: 5px; border-top: 1px solid #c9c8c3; }
ul { margin: 2px 0 4px 0; padding-left: 15px; }
li { margin: 1.5px 0; }
.box { background: #f4f3ef; border-left: 3px solid #0b0b0b; padding: 6px 9px; margin: 6px 0 4px 0; }
table { border-collapse: collapse; width: 100%; margin: 4px 0 2px 0; font-size: 7.9pt; }
th { text-align: center; font-weight: 600; padding: 2px 4px; border-bottom: 1px solid #0b0b0b; }
td { padding: 1.8px 4px; border-bottom: 1px solid #e4e3df; vertical-align: top; }
td.c { text-align: center; }
td.good { color: #1a6b34; font-weight: 600; } td.bad { color: #a8322a; font-weight: 600; }
tr.g td { background: #efeee9; font-weight: 600; }
code { font-family: Consolas, monospace; font-size: 8pt; background: #f4f3ef; padding: 0 2px; }
.small { color: #52514e; font-size: 7.9pt; }
.fig { margin: 4px 0 0 0; }
.pb { page-break-before: always; }
"""
HTML = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{NAME}</title><style>{CSS}</style></head><body>
<h1>배치 qb_ phase1 — 통신(comm) vs OFF 결과</h1>
<div class='sub1'>2026-10-13 · LICS 서버 · 코드 feat/fig1-latent b6a4341 · 스펙 2026-10-13-trunk-fork-design.md 분기 규칙 2 · 순수 PPO(흉내 없음) + T-B 토글(logstd 초기 −2.0 · LR 3e-4 → 3e-5 감쇠, 두 팔 동일) ·
시드 {SEED} 하나 · trunk {BRANCH_AT:,} → 갈래 {TOTAL:,} 결정 · 갈래 off(ARPA@56) · comm(300 m 메시지, p46 코덱 k={CH_K}) · offb 없음</div>

<div class='box'><b>한 줄 결론</b> — 주 평가에서 <b>통신(comm)이 OFF 를 못 이김</b>: 주 지표 통과 <b>{WIN_MAIN}/{len(MAIN)}</b>, 방향만 봐도 comm 이 앞선 지표 {DIR_MAIN}/{len(MAIN)}
({' · '.join(WINL) if WINL else '없음'}), 뒤진 지표 {len(LOSE)}/{len(MAIN)}. 도착 off {f(off['goal'])} % · comm {f(comm['goal'])} %, 배끼리 충돌 {f(off['vColl'])} % · {f(comm['vColl'])} %.
두 팔 모두 분기 뒤 나빠짐: 고정 장면 도착이 trunk 끝 {f(TR_END['goal'])} % → 끝 모델 off {f(END_O['goal'])} % · comm {f(END_C['goal'])} %.
trunk 자체도 고정 장면에서는 {TR_BEST[0]:g}M 이 최고(도착 {f(TR_BEST[1]['goal'])} % · epReward {f(TR_BEST[1]['epReward'], 0)})였고 그 뒤 내려옴 → 보상 곡선 M(우상향) 판정 off <b>불통과</b> · comm <b>불통과</b>.
반대 방향 자료도 있음: 학습 중 창 값은 갈래 21창 가운데 {N_ABOVE}창에서 comm 도착이 위, 고정 장면 체크포인트(서로 다른 모델 4개)는 comm 위 {C_ABOVE} · off 위 {4 - C_ABOVE} → 재는 방식 · 시점에 따라 순서가 갈림.
시드 1개 · 잡음 자(offb · 시드 범위) 없음 → 어느 방향으로도 단독 주장 못 함. 다음은 저자 결정.</div>

<h2>1. 주 평가 — 끝 모델(16.06M), 256 env · 10000 결정 · burn-in 2400 · 평가 시드 999</h2>
{judge_table(MAIN)}
{judge_table(AUX)}
<ul>
<li>판정식(결과 전 고정, <code>summarize_fig1.py</code>): 같은 시드 짝 전부 승 + |평균차| &gt; N, N = max(OFF 시드 범위, |offb − off| 평균, 고정 하한). 이 배치는 <b>시드 1개 · offb 없음 → N = 고정 하한뿐</b>
(배 충돌 12.1 · DCPA 1.5 · 나머지 0). 스크립트 표 <code>_fig1.md</code> 의 '주 지표 통과: {WIN_MAIN}/{len(MAIN)}' 와 같음</li>
<li>최선 OFF 풀(도착 ≥ 50 % 인 OFF 만): {' · '.join(sorted(POOL)) if POOL else '없음'} — 이 배치 off({f(off['goal'])} %)는 50 % 미만이라 풀에 못 듦. comm 은 풀 최선(도착 {f(POOL['t_off']['goal']) if 't_off' in POOL else 'n/a'} %)에 크게 못 미침</li>
<li>보조 지표: comm 이 앞선 것 {DIR_AUX}/{sum(1 for k, *_ in AUX if J[k])}. epReward off {f(off['epReward'], 0)} · comm {f(comm['epReward'], 0)}</li>
<li>H1a(통신은 OFF 보다 나빠지면 안 됨) 관점: 이 시드 · 이 끝 모델에서는 <b>{'불성립' if verdict_h1a else '방향 엇갈림'}</b>. 단, 아래 §3 처럼 체크포인트마다 순서가 바뀌므로 끝 모델 하나의 차를 통신 효과로 읽을 수 없음</li>
</ul>
{role_table()}

<h2>2. 학습 곡선 — 학습 중 창 값(행동 잡음 포함)</h2>
<div class='fig'>{G_WIN}</div>
<div class='fig'>{G_WINV}</div>
<div class='fig'>{G_WINR}</div>
{blk_table()}
<ul>
<li>분기 직후 첫 창(9.37M): off 도착 {f(BR['off'][0]['goal'])} % · comm {f(BR['comm'][0]['goal'])} % — 같은 trunk 에서 시작</li>
<li>두 팔 모두 내려갔다가 끝에 일부 회복: 최저 창 off {f(LOW['off']['goal'])} % ({LOW['off']['dec']:.2f}M) · comm {f(LOW['comm']['goal'])} % ({LOW['comm']['dec']:.2f}M). 끝 4창 off {' · '.join(f(w['goal']) for w in LAST['off'])} / comm {' · '.join(f(w['goal']) for w in LAST['comm'])}</li>
<li>갈래 21창 가운데 comm 도착 &gt; off 도착 = {N_ABOVE}창, comm 결정당 보상 &gt; off = {N_ABOVE_R}창. 끝 4창 평균 도착은 comm {f(wm(LAST['comm'], 'goal'))} % &gt; off {f(wm(LAST['off'], 'goal'))} % 인데
주 평가(끝 모델)는 off {f(off['goal'])} % &gt; comm {f(comm['goal'])} % 로 반대 — off 가 끝 4창 사이 {f(LAST['off'][0]['goal'])} → {f(LAST['off'][-1]['goal'])} % 로 오르던 중이었음(§3)</li>
<li>결정당 보상은 두 팔 모두 trunk 끝 4창({f(wm(TR_LAST, 'R'), 3)})보다 낮게 끝남: 갈래 끝 4창 off {f(wm(LAST['off'], 'R'), 3)} · comm {f(wm(LAST['comm'], 'R'), 3)}</li>
</ul>

<h2 class='pb'>3. 고정 장면 체크포인트 곡선 — 256 env · 3000 결정 · drain 3000 · 평가 시드 999</h2>
<div class='fig'>{G_CEP}</div>
<div class='fig'>{G_CGO}</div>
{curve_table()}
<table><thead><tr><th>M 판정(스펙 10-07, 결과 전 고정 · 보고만)</th><th>첫 값</th><th>최고</th><th>끝</th><th>최대 하락</th><th>허용(0.1 × (최고 − 첫 값))</th><th>판정</th></tr></thead><tbody>
{''.join(f"<tr><td class='c'>{a} (trunk 5점 + 갈래 5점)</td><td class='c'>{f(MC[a][0], 1)}</td><td class='c'>{f(MC[a][1], 1)}</td><td class='c'>{f(MC[a][2], 1)}</td><td class='c'>{f(MC[a][3], 1)}</td><td class='c'>{f(M_FRAC * (MC[a][1] - MC[a][0]), 1)}</td><td class='c'>{'우상향' if MC[a][4] else '<b>불통과</b>'}</td></tr>" for a in ('off', 'comm'))}
</tbody></table>
<ul>
<li>16M 점과 16.06M 점은 <b>같은 모델</b>임(step16M 체크포인트가 마지막 update 에서 저장됨 — steps · 가중치 SHA 같음, 값도 같음). 갈래의 서로 다른 모델은 4개(10 · 12 · 14 · 16.06M)</li>
<li>그 4개 가운데 comm 도착 &gt; off 도착 = {C_ABOVE}개, comm epReward &gt; off = {C_ABOVE_R}개 → <b>순서가 체크포인트마다 바뀜</b>. 고정 장면 도착 최저: off {f(C_LOW['off'][1]['goal'])} % ({C_LOW['off'][0]:g}M) · comm {f(C_LOW['comm'][1]['goal'])} % ({C_LOW['comm'][0]:g}M)</li>
<li>trunk: 고정 장면 도착 {' → '.join(f(v['goal']) for _, v in TRPT)} % ({' · '.join(f'{x:g}' for x, _ in TRPT)}M), epReward {' → '.join(f(v['epReward'], 0) for _, v in TRPT)} — 첫 점이 최고라 M 의 '최고 − 첫 값' 이 0, 두 팔 모두 불통과</li>
<li>고정 장면 도착: trunk 끝(9.04M) {f(TR_END['goal'])} % → 끝 모델 off {f(END_O['goal'])} % · comm {f(END_C['goal'])} %. epReward {f(TR_END['epReward'], 0)} → {f(END_O['epReward'], 0)} · {f(END_C['epReward'], 0)}</li>
<li>끝 모델은 곡선 평가(drain 3000 · 결정 3000)와 주 평가(drain 0 · 결정 10000)에서 순서가 같음: 도착 곡선 off {f(END_O['goal'])} · comm {f(END_C['goal'])} / 주 평가 {f(off['goal'])} · {f(comm['goal'])}</li>
</ul>

<h2>4. 코덱 관문(p46, 시드 43 trunk 의 레이더 임베딩 + 핵심 16 → k)</h2>
{codec_table()}
<ul><li>공통 k = 관문을 통과하는 가장 작은 k(6 → 8 → 12 순) = <b>{CH_K}</b>. 분기 검사 <code>_branch_check.txt</code>: ALL PASS(같은 trunk sha256 {list(SHA)[0]}, seed {SEED}, branch_at {BRANCH_AT:,})</li></ul>

<h2>5. 경과 · 실행 기록</h2>
<ul>
<li>{t_of('시작 codec')} 코덱 → {t_of('끝 codec')} · 갈래 {t_of('시작 branches')}–{t_of('끝 branches')} (off {f(SV['off'][1])}분 · comm {f(SV['comm'][1])}분) · 주 평가 {t_of('시작 eval')}–{t_of('끝 eval')} ·
곡선 {t_of('시작 curve')}–{t_of('끝 curve')} · 표 {t_of('끝 fig1')}. smoke · trunk · 관문은 단계 1 실행분을 이어 씀(건너뜀)</li>
<li>trunk 관문(단계 1): 마지막 4창 도착 {GATE.group(1)} % · 배 충돌 {GATE.group(2)} % · 벽 충돌 {GATE.group(3)} % = 통과. 실험 env 기록 <code>{BENV}</code></li>
{''.join(f'<li>{ln}</li>' for ln in NOTE)}
</ul>

<h2>6. 한계 — 그대로 읽을 것</h2>
<ul>
<li>시드 1개(43) · offb 없음 → 잡음 크기를 재는 자가 없음. comm − off 차(도착 {f(J['goal']['d'], 1, True)} %p)가 통신 탓인지 재분기 흔들림인지 가를 수 없음.
참고로 f2_(10-06)에서 잰 |offb − off| 는 이 배치에 적용 못 함(다른 trunk · 다른 보상)</li>
<li>체크포인트마다 순서가 바뀜(§3) → 끝 모델 하나로 낸 주 평가 표는 그 시점의 값일 뿐임</li>
<li>두 팔 모두 trunk 보다 나빠진 상태에서의 비교임. 정책이 무너지는 구간의 팔 간 차는 통신 가치의 근거로 약함</li>
<li>T-B 토글 2개(logstd · LR 감쇠)를 같이 켰고 trunk 창 값도 2.6M 최고 뒤 내려오던 중에 분기함(단계 1 보고)</li>
<li>순위 관문은 재실행 없이 n_ 기록과 10-09 저자 override 를 적용(스펙 §3). 원인(왜 분기 뒤 두 팔이 다 내려가는지)은 재지 않았음</li>
<li>관문 기준 · 판정식 · 스크립트는 안 건드렸음</li>
</ul>

<h2>7. 저자 결정 대기</h2>
<ul>
<li>이 결과로는 목표 1(통신 전 지표 우월) · 목표 2(보상 우상향 · 통신 압도) 어느 것도 못 채움. 시드 3개 확증은 '성공 뒤에만' 지시라 해당 없음</li>
<li>스펙 §3 이 적어 둔 다음 후보(저자 결정): 흉내 초기화를 trunk 에만 재도입(갈래 순수 PPO) 또는 보상 재설계. T-A '약한 B'(도착 47.1 %, 기준 50)를 근거로 단계 2(보상 반경 56)를 볼지도 저자 결정</li>
<li>결정 전까지 추가 학습 · 평가는 안 돌림</li>
</ul>
<div class='small'>자료: runs/2026-10-02_fig1/out/qb/ (lr_qb_*.txt · qb_*.log · curve_qb_*.txt · _fig1.md · _curve.md · _codec_choice.txt · _branch_check.txt · _timeline.txt) · out/q_stage1_notes/runnote_phase1.txt ·
서버 Python/_repro_out_qb · 체크포인트 VESSEL_checkpoints/comm_intent/qb_*.pt · 생성 report/make_report_fig1_qb.py · 단계 1 보고 2026-10-13_q배치_단계1_trunk갈림길_결과.pdf</div>
</body></html>"""

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '_build')
os.makedirs(out_dir, exist_ok=True)
out_html = os.path.join(out_dir, NAME + '.html')
out_pdf = os.path.join(out_dir, NAME + '.pdf')
with io.open(out_html, 'w', encoding='utf-8') as fh:
    fh.write(HTML)
edge = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
subprocess.run([edge, '--headless', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={out_pdf}',
                'file:///' + out_html.replace(chr(92), '/')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print('주 지표 통과', WIN_MAIN, '/', len(MAIN), '| 방향 승', DIR_MAIN, '| 보조 방향 승', DIR_AUX, '| 창 comm>off', N_ABOVE, '/21 | 곡선 comm>off', C_ABOVE, '/4',
      '| M off', MC['off'][4], 'comm', MC['comm'][4], '| 풀', sorted(POOL))
print(out_pdf)
