# 2026-10-13: 배치 q_ 단계 1(trunk 갈림길 T-A qa_ · T-B qb_ · D 진단) 결과 PDF. 새 학습·평가 없음. 숫자는 전부 로그에서 읽음:
#   runs/2026-10-02_fig1/out/qa/qa_trunk_d6_s43.log · qb/qb_trunk_d6_s43.log · n_trunk_d6_s43.log   (학습 창 = update 5개, 관문이 읽는 줄)
#   out/<qa|qb>/_gate.log · _timeline.txt · _batch_env.txt · _rank_note.txt · <접두어>trunk_d6_s43_ep.csv
#   out/diag_trunk_n/ (logstd.txt · eval_sample_*.txt · eval_mean_*.txt · ev.txt · _diag_trunk.md)
#   out/q_stage1_notes/ (runnote.txt = 실행 기록 · snap_2M.txt = 2M 체크포인트 스냅샷 읽은 값)
# 판정 기준 = 스펙 docs/superpowers/specs/2026-10-13-trunk-fork-design.md §3 (결과 전 고정)
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_trunk_q.py [출력 폴더]
import csv, io, os, re, subprocess, sys

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', 'out'))
NAME = '2026-10-13_q배치_단계1_trunk갈림길_결과'
SEED = 43
UPD = 65536            # update 하나 = 128 env × 16척 × rollout 32
TRUNK_DEC = 9043968    # = 138 update
GATE_GOAL, GATE_OCOLL, GATE_N = 30.0, 5.0, 4   # 10-07 스펙 §5-3 (_run_f.sh gate9)
B_GOAL, B_VCOLL = 50.0, 49.8                    # 10-13 스펙 §3 B 확정 기준(§5-3 통과 + goal ≥ 50 + vColl ≤ n_ −20 %p)
D_LOGSTD, D_NOISE, D_DECAY = -1.5, 15.0, -5.0   # 10-13 스펙 §3 D 기록 기준


def text(*p):
    with io.open(os.path.join(OUT, *p), encoding='utf-8', errors='replace') as fh:
        return fh.read()


def fm(v, nd=1, sign=False):
    s = f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'
    return s.replace('-', '−')


WIN = re.compile(r'^\[ON\] dec=([\d.]+)M \| ep=(\d+) len~(\d+) \| goal=([\d.]+)% vColl=([\d.]+)% oColl=([\d.]+)% TO=([\d.]+)% '
                 r'\| R=(-?[\d.]+) \| (\d+) dec/s \| epRet=(-?[\d.]+)(?: \| lr=([\d.e+-]+))?')
KEYS = ('dec', 'ep', 'len', 'goal', 'vColl', 'oColl', 'TO', 'R', 'dps', 'epRet', 'lr')


def windows(*p):
    rows = []
    for ln in text(*p).splitlines():
        m = WIN.match(ln)
        if m:
            rows.append({k: (float(x) if x is not None else None) for k, x in zip(KEYS, m.groups())})
    return rows


def saved(*p):
    m = re.search(r'saved -> \S+ \(([\d.]+)M decisions, ([\d.]+)min\)', text(*p))
    return float(m.group(1)), float(m.group(2))


def mean(rows, k):
    return sum(r[k] for r in rows) / len(rows)


def epcsv(*p):
    with io.open(os.path.join(OUT, *p), encoding='utf-8') as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


RUNS = {   # 표시 이름, 창, 학습 시간(분), ep.csv
    'n_': ('n_ (기준, 10-09)', windows('n_trunk_d6_s43.log'), saved('n_trunk_d6_s43.log'), epcsv('n_trunk_d6_s43_ep.csv')),
    'qa_': ('qa_ = T-A (ARPA@300)', windows('qa', 'qa_trunk_d6_s43.log'), saved('qa', 'qa_trunk_d6_s43.log'), epcsv('qa', 'qa_trunk_d6_s43_ep.csv')),
    'qb_': ('qb_ = T-B (logstd −2.0 + LR 감쇠)', windows('qb', 'qb_trunk_d6_s43.log'), saved('qb', 'qb_trunk_d6_s43.log'), epcsv('qb', 'qb_trunk_d6_s43_ep.csv')),
}
for pre, (_, W, (dec, _m), EP) in RUNS.items():
    assert len(W) == 27 and dec == 9.04 and len(EP) == TRUNK_DEC // UPD == 138 and EP[-1]['step'] == TRUNK_DEC, (pre, len(W), dec, len(EP))
    for i, w in enumerate(W):   # 로그 한 줄 = update 5개 묶음
        assert sum(r['n_ep'] for r in EP[5 * i:5 * i + 5]) == w['ep'], (pre, i)
N, QA, QB = RUNS['n_'][1], RUNS['qa_'][1], RUNS['qb_'][1]
assert all(w['lr'] is None for w in N + QA) and all(w['lr'] is not None for w in QB)
for pre in ('qa', 'qb'):
    lg = text(pre, f'{pre}_trunk_d6_s43.log')
    assert '[resume]' not in lg and '[bc]' not in lg and 'DAgger' not in lg, pre   # 처음부터 순수 PPO


def tail3(EP):   # 로그 창에 안 들어간 마지막 update 3개(138 = 27 × 5 + 3)
    T = EP[135:]
    n = sum(r['n_ep'] for r in T)
    return n, sum(r['n_ep'] * r['goal'] for r in T) / n, sum(r['n_ep'] * r['vColl'] for r in T) / n


# ── 관문 다시 계산(스크립트와 같은 식) → _gate.log 와 대조
def gate(W):
    L = W[-GATE_N:]
    g, v, o = mean(L, 'goal'), mean(L, 'vColl'), mean(L, 'oColl')
    return L, g, v, o, (g >= GATE_GOAL and o <= GATE_OCOLL)


G = {pre: gate(RUNS[pre][1]) for pre in RUNS}
for pre in ('qa', 'qb'):
    m = re.search(rf'^s{SEED} goal=([\d.]+) vColl=([\d.]+) oColl=([\d.]+) 창=(\d+) (통과|불통과)', text(pre, '_gate.log'), re.M)
    _, g, v, o, ok = G[pre + '_']
    assert m and (m.group(1), m.group(2), m.group(3), m.group(4)) == (f'{g:.1f}', f'{v:.1f}', f'{o:.1f}', str(GATE_N)), (pre, m and m.groups(), g, v, o)
    assert (m.group(5) == '통과') == ok, pre
assert f"{G['n_'][1]:.1f}" == '27.3' and not G['n_'][4]                     # 10-09 보고와 같은 값
assert abs((G['n_'][2] - 20.0) - B_VCOLL) < 0.05, G['n_'][2]                # B 기준 vColl = n_ 4창 − 20 %p
A_PASS, B_PASS = G['qa_'][4], G['qb_'][4]
B_CONF = A_PASS and G['qa_'][1] >= B_GOAL and G['qa_'][2] <= B_VCOLL
RULE = 1 if B_CONF else (2 if B_PASS else 3)
assert RULE == 2 and A_PASS and G['qa_'][2] <= B_VCOLL and G['qa_'][1] < B_GOAL   # 본문 문장이 이 경우를 전제로 씀

# ── 관문 식·B 식을 모든 4창 위치에 대 봄(기록용, 판정 아님)
def roll(W):
    return [(W[j]['dec'] - 0.33, W[j + GATE_N - 1]['dec'], mean(W[j:j + GATE_N], 'goal'), mean(W[j:j + GATE_N], 'vColl'), mean(W[j:j + GATE_N], 'oColl'))
            for j in range(len(W) - GATE_N + 1)]


ROLL_A = roll(QA)
ROLL_A_B = [r for r in ROLL_A if r[2] >= B_GOAL and r[3] <= B_VCOLL and r[4] <= GATE_OCOLL]
ROLL_A_BEST = max(ROLL_A, key=lambda r: r[2])
ROLL_B = roll(QB)
ROLL_B_BEST = max(ROLL_B, key=lambda r: r[2])
blk = lambda W: [W[0:9], W[9:18], W[18:27]]     # 창 9개씩 3구간(결과 보고 고르지 않으려고 3등분)
peak = lambda W: max(W, key=lambda w: w['goal'])

# ── 경과
def tl(pre):   # trunk 모드 실행분만(같은 접두어 phase1 이 뒤에 이어 씀)
    L = [ln for ln in text(pre, '_timeline.txt').splitlines() if ln.startswith('2026-10-13')]
    end = next(i for i, ln in enumerate(L) if 'trunk 모드 완료' in ln)
    return L[:end + 1]


def t_of(pre, pat):
    hit = [ln for ln in tl(pre) if ln[20:].startswith(pat)]
    assert hit, (pre, pat)
    return hit[0][11:19]


for pre in ('qa', 'qb'):
    assert 'b6a4341' in tl(pre)[0] and f'배치 {pre}_ trunk' in tl(pre)[0], tl(pre)[0]
    assert not any('★' in ln or '건너뜀' in ln for ln in tl(pre)), pre   # 멈춤·재시도·건너뛴 단계 없음
BENV = {pre: text(pre, '_batch_env.txt').strip() for pre in ('qa', 'qb')}
assert BENV['qa'] == 'ARPA=300 RANK=v4 REWARD_RANGE= EXTRA=[]', BENV['qa']
assert BENV['qb'] == 'ARPA=56 RANK=v4 REWARD_RANGE= EXTRA=[VESSEL_LOGSTD_INIT0=-2.0 VESSEL_LR_FINAL=3e-5 VESSEL_LR_DECAY_DEC=16056320]', BENV['qb']
RANKNOTE = text('qa', '_rank_note.txt').strip().lstrip('> ')
NOTE = [ln for ln in text('q_stage1_notes', 'runnote.txt').splitlines()[1:] if ln.strip()]

# ── 2M 스냅샷(직접 연 값)
SNAP = {}
cur = None
for ln in text('q_stage1_notes', 'snap_2M.txt').splitlines():
    m = re.match(r'== (\w+?)trunk_d6_s43\.step2M\.pt', ln)
    if m:
        cur = m.group(1); SNAP[cur] = {'logstd0': []}; continue
    m = re.match(r'\s+(\S+) = (.*)$', ln)
    if m and cur:
        k, v = m.group(1), m.group(2)
        if k.endswith('action_logstd'):
            SNAP[cur]['logstd0'].append(float(v.strip('[]').split(',')[0]))
        else:
            SNAP[cur][k.replace('cfg_snapshot.', '')] = v
assert set(SNAP) == {'qa_', 'qb_', 'n_'} and all(len(s['logstd0']) == 5 and s['steps'] == '2031616' and s['seed'] == '43' for s in SNAP.values())
assert SNAP['qa_']['partner_range'] == '300.0' and SNAP['qb_']['partner_range'] == SNAP['n_']['partner_range'] == '56.0'
assert (SNAP['qb_']['logstd_init0'], SNAP['qb_']['lr_final'], SNAP['qb_']['lr_decay_dec']) == ('-2.0', '3e-05', '16056320.0')
assert (SNAP['qa_']['logstd_init0'], SNAP['qa_']['lr_final'], SNAP['qa_']['lr_decay_dec']) == ('-1.0', '0.0003', '0.0')
for s in SNAP.values():
    assert (s['action_mode'], s['comm_fields'], s['sim.ROLE_V2_PRIMARY'], s['sim.ROLE_V2_RES_F6'], s['sim.REWARD_RANGE'], s['comm_range']) == ('course', 'state', 'cum', '1', 'None', '300.0'), s

# ── D 진단
EVL = re.compile(r'goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%\s+oColl=\s*([\d.]+)%\s+TO=\s*([\d.]+)%\s+\|\s+(\d+) eps')
RUNLN = '[run] burnin=2400 dec=3000 drain=3000 ring=1.0 crossing=0 seed=999 envs=256 vessels=16 maxp=8'


def ev(name):
    t = text('diag_trunk_n', name)
    assert RUNLN in t, name
    m = EVL.search(t)
    return dict(zip(('goal', 'vColl', 'oColl', 'TO', 'eps'), (float(x) for x in m.groups())))


D = {(kind, tag): ev(f'eval_{kind}_{tag}.txt') for kind in ('sample', 'mean') for tag in ('4M', '9M')}
assert '[eval_mean] student drives with tanh(mean)' in text('diag_trunk_n', 'eval_mean_9M.txt')
LS = []
for ln in text('diag_trunk_n', 'logstd.txt').splitlines():
    m = re.search(r'steps=(\d+) logstd0=((?:-?[\d.]+ ?){5})mean=(-?[\d.]+) sigma_course=([\d.]+)deg', ln)
    if m:
        LS.append((int(m.group(1)), [float(x) for x in m.group(2).split()], float(m.group(3)), float(m.group(4))))
assert len(LS) == 5 and LS[-1][0] == TRUNK_DEC
m = re.search(r'EVb\(lam=0\.95\)=(-?[\d.]+) EVb\(lam=1\)=(-?[\d.]+) ret_std=([\d.]+) T=(\d+)', text('diag_trunk_n', 'ev.txt'))
EV95, EV1, RETSTD = float(m.group(1)), float(m.group(2)), float(m.group(3))
assert '거부' not in text('diag_trunk_n', 'ev.txt') and 'Traceback' not in text('diag_trunk_n', 'ev.txt')
D_NOISE9 = D[('mean', '9M')]['goal'] - D[('sample', '9M')]['goal']
D_NOISE4 = D[('mean', '4M')]['goal'] - D[('sample', '4M')]['goal']
D_DEC = D[('mean', '9M')]['goal'] - D[('mean', '4M')]['goal']
dlog = text('diag_trunk_n', '_log.txt').splitlines()
D_T0, D_T1 = dlog[0][11:19], dlog[-1][11:19]
assert '끝' in dlog[-1]


# ── 그림(SVG, 로그 값 그대로)
def chart(series, y0, y1, yticks, unit, hlines=(), shade=None, w=700, h=150):
    ml, mr, mt, mb = 36, 8, 16, 18
    X = lambda d: ml + (w - ml - mr) * d / 9.2
    Y = lambda v: mt + (h - mt - mb) * (y1 - v) / (y1 - y0)
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='100%'>"]
    if shade:
        s.append(f"<rect x='{X(shade[0]):.1f}' y='{mt}' width='{X(shade[1]) - X(shade[0]):.1f}' height='{h - mt - mb}' fill='#efeee9'/>")
    for t in yticks:
        s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' stroke='#e4e3df' stroke-width='0.6'/>")
        s.append(f"<text x='{ml - 4}' y='{Y(t) + 2.6:.1f}' font-size='7.2' text-anchor='end' fill='#52514e'>{fm(t, 0 if unit == '%' else 1)}</text>")
    for d in range(0, 10):
        s.append(f"<text x='{X(d):.1f}' y='{h - 5}' font-size='7.2' text-anchor='middle' fill='#52514e'>{d}M</text>")
    s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{h - mb}' y2='{h - mb}' stroke='#0b0b0b' stroke-width='0.7'/>")
    for hv, hl in hlines:
        s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{Y(hv):.1f}' y2='{Y(hv):.1f}' stroke='#0b0b0b' stroke-width='0.7' stroke-dasharray='2 2'/>")
        s.append(f"<text x='{w - mr - 2}' y='{Y(hv) - 2.5:.1f}' font-size='7.2' text-anchor='end' fill='#0b0b0b'>{hl}</text>")
    lx = ml + 4
    for lab, pts, col, sw, dash in series:
        p = ' '.join(f"{X(d):.1f},{Y(min(max(v, y0), y1)):.1f}" for d, v in pts)
        da = f" stroke-dasharray='{dash}'" if dash else ''
        s.append(f"<polyline points='{p}' fill='none' stroke='{col}' stroke-width='{sw}'{da}/>")
        s.append(f"<line x1='{lx}' x2='{lx + 16}' y1='7' y2='7' stroke='{col}' stroke-width='{sw}'{da}/>")
        s.append(f"<text x='{lx + 19}' y='9.6' font-size='7.6' fill='#0b0b0b'>{lab}</text>")
        lx += 19 + 7.9 * len(lab) + 12
    return ''.join(s) + '</svg>'


pts = lambda rows, k: [(r['dec'], r[k]) for r in rows]
SHADE = (QA[-GATE_N]['dec'] - 0.33, QA[-1]['dec'])
STY = (('n_', N, '#b9b7b0', 1.3, ''), ('qa_ T-A', QA, '#0b0b0b', 1.7, ''), ('qb_ T-B', QB, '#a8322a', 1.5, '5 2'))
C_GOAL = chart([(f'{lab} 도착', pts(W, 'goal'), col, sw, da) for lab, W, col, sw, da in STY], 0, 100, [0, 20, 40, 60, 80, 100], '%',
               hlines=((GATE_GOAL, '관문 도착 30 %'), (B_GOAL, 'B 확정 도착 50 %')), shade=SHADE)
C_VC = chart([(f'{lab} 배 충돌', pts(W, 'vColl'), col, sw, da) for lab, W, col, sw, da in STY], 0, 100, [0, 20, 40, 60, 80, 100], '%',
             hlines=((B_VCOLL, 'B 확정 배 충돌 49.8 %'),), shade=SHADE, h=128)
C_REW = chart([(f'{lab} 결정당 보상', pts(W, 'R'), col, sw, da) for lab, W, col, sw, da in STY], -1.4, 0.4, [-1.2, -0.8, -0.4, 0.0, 0.4], 'R',
              hlines=((0.0, '0'),), shade=SHADE, h=118)


def gate_table():
    h = ("<table><thead><tr><th>런 (시드 43)</th><th>n_ 과 다른 것 하나</th><th>도착 %</th><th>배 충돌 %</th><th>벽 충돌 %</th><th>시간초과 %</th>"
         "<th>결정당 보상 R</th><th>4창 도착 값</th><th>§5-3 관문</th></tr></thead><tbody>")
    for pre, diff in (('n_', '—'), ('qa_', 'OFF 가 300 m 참값을 봄'), ('qb_', 'logstd 초기 −2.0 · LR 3e-4 → 3e-5')):
        L, g, v, o, ok = G[pre]
        h += (f"<tr><td class='l2'>{RUNS[pre][0]}</td><td>{diff}</td><td class='c'><b>{fm(g)}</b></td><td class='c'>{fm(v)}</td><td class='c'>{fm(o)}</td>"
              f"<td class='c'>{fm(mean(L, 'TO'))}</td><td class='c'>{fm(mean(L, 'R'), 3)}</td><td class='c'>{' · '.join(fm(w['goal']) for w in L)}</td>"
              f"<td class='c'>{'통과' if ok else '<b>불통과</b>'}</td></tr>")
    return h + "</tbody></table>"


def blk_table():
    h = ("<table><thead><tr><th>런</th><th>구간(창 9개씩)</th><th>도착 %</th><th>배 충돌 %</th><th>벽 충돌 %</th><th>시간초과 %</th><th>결정당 보상 R</th>"
         "<th>도착 최저–최고</th></tr></thead><tbody>")
    for pre in RUNS:
        for b in blk(RUNS[pre][1]):
            h += (f"<tr><td class='c'>{pre}</td><td class='c'>{b[0]['dec'] - 0.33:.2f}–{b[-1]['dec']:.2f}M</td><td class='c'>{fm(mean(b, 'goal'))}</td><td class='c'>{fm(mean(b, 'vColl'))}</td>"
                  f"<td class='c'>{fm(mean(b, 'oColl'))}</td><td class='c'>{fm(mean(b, 'TO'))}</td><td class='c'>{fm(mean(b, 'R'), 3)}</td>"
                  f"<td class='c'>{fm(min(w['goal'] for w in b))}–{fm(max(w['goal'] for w in b))}</td></tr>")
    return h + "</tbody></table>"


def win_table():
    h = ("<table class='w'><thead><tr><th rowspan='2'>결정(M)</th><th colspan='4'>n_ (기준)</th><th colspan='4'>qa_ T-A</th><th colspan='5'>qb_ T-B</th></tr><tr>"
         + "<th>도착</th><th>배 충돌</th><th>시간초과</th><th>R</th>" * 3 + "<th>lr</th></tr></thead><tbody>")
    for i in range(27):
        cls = " class='g'" if i >= 27 - GATE_N else ''
        h += f"<tr{cls}><td class='c'>{N[i]['dec']:.2f}</td>"
        for W in (N, QA, QB):
            w = W[i]
            h += f"<td class='c'>{fm(w['goal'])}</td><td class='c'>{fm(w['vColl'])}</td><td class='c'>{fm(w['TO'])}</td><td class='c'>{fm(w['R'], 3)}</td>"
        h += f"<td class='c'>{QB[i]['lr']:.2e}</td></tr>"
    return h + "</tbody></table>"


def snap_table():
    h = ("<table><thead><tr><th>런</th><th>partner_range</th><th>logstd_init0</th><th>lr_final</th><th>lr_decay_dec</th><th>2M logstd dim0 (전문가 5)</th></tr></thead><tbody>")
    for pre in ('n_', 'qa_', 'qb_'):
        s = SNAP[pre]
        g = lambda k: s.get(k, '키 없음(옛 코드)')
        h += (f"<tr><td class='c'>{pre}</td><td class='c'>{s['partner_range']}</td><td class='c'>{g('logstd_init0')}</td><td class='c'>{g('lr_final')}</td>"
              f"<td class='c'>{g('lr_decay_dec')}</td><td class='c'>{' · '.join(fm(x, 3) for x in s['logstd0'])}</td></tr>")
    return h + "</tbody></table>"


def d_table():
    h = ("<table><thead><tr><th>체크포인트</th><th>행동</th><th>도착 %</th><th>배 충돌 %</th><th>벽 충돌 %</th><th>시간초과 %</th><th>에피소드 수</th></tr></thead><tbody>")
    for tag, lab in (('4M', '4.06M'), ('9M', '9.04M')):
        for kind, kl in (('sample', '표본(학습 때처럼 잡음 있음)'), ('mean', '평균 행동(잡음 없음)')):
            e = D[(kind, tag)]
            h += (f"<tr><td class='c'>{lab}</td><td>{kl}</td><td class='c'>{fm(e['goal'])}</td><td class='c'>{fm(e['vColl'])}</td><td class='c'>{fm(e['oColl'])}</td>"
                  f"<td class='c'>{fm(e['TO'])}</td><td class='c'>{e['eps']:.0f}</td></tr>")
    return h + "</tbody></table>"


def ls_table():
    h = "<table><thead><tr><th>결정</th><th>logstd dim0 전문가 0–4</th><th>평균</th><th>침로 잡음 σ</th></tr></thead><tbody>"
    for st, vals, mu, sig in LS:
        h += f"<tr><td class='c'>{st / 1e6:.2f}M</td><td class='c'>{' · '.join(fm(x, 3) for x in vals)}</td><td class='c'>{fm(mu, 3)}</td><td class='c'>{fm(sig)}°</td></tr>"
    return h + "</tbody></table>"


ok = lambda b: '충족' if b else '<b>미달</b>'
LA, GA, VA, OA, _ = G['qa_']
LB, GB, VB, OB, _ = G['qb_']
PA, PB, PN = peak(QA), peak(QB), peak(N)
TA, TB = tail3(RUNS['qa_'][3]), tail3(RUNS['qb_'][3])
MA, MB = RUNS['qa_'][2][1], RUNS['qb_'][2][1]

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
table.w { font-size: 6.9pt; }
th { text-align: center; font-weight: 600; padding: 2px 4px; border-bottom: 1px solid #0b0b0b; }
td { padding: 1.8px 4px; border-bottom: 1px solid #e4e3df; vertical-align: top; }
table.w td { padding: 0.4px 3px; }
td.c { text-align: center; } td.l2 { width: 22%; }
tr.g td { background: #efeee9; font-weight: 600; }
code { font-family: Consolas, monospace; font-size: 8pt; background: #f4f3ef; padding: 0 2px; }
.small { color: #52514e; font-size: 7.9pt; }
.fig { margin: 4px 0 0 0; }
.pb { page-break-before: always; }
"""
HTML = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{NAME}</title><style>{CSS}</style></head><body>
<h1>배치 q_ 단계 1 — trunk 갈림길(T-A · T-B) · D 진단 결과</h1>
<div class='sub1'>2026-10-13 · LICS 서버 · 코드 feat/fig1-latent b6a4341 · 스펙 2026-10-13-trunk-fork-design.md · 순수 PPO(흉내 없음) · 시드 {SEED} 하나 ·
128 env × 16척 · rollout 32 · 침로 행동 모드 · 메시지 없음 · 보상 v3 + R1 + R2(n_ 과 동일) · {TRUNK_DEC:,} 결정</div>

<div class='box'><b>한 줄 결론</b> — 두 trunk 모두 §5-3 관문 <b>통과</b>(마지막 4창 도착 qa_ <b>{fm(GA)} %</b> · qb_ <b>{fm(GB)} %</b>, 기준 ≥ {GATE_GOAL:.0f} · 기준선 n_ {fm(G['n_'][1])} %).
가설 B 확정 기준은 <b>미달</b>: qa_ 도착 {fm(GA)} % &lt; {B_GOAL:.0f} %(배 충돌 {fm(VA)} % ≤ {B_VCOLL} % 는 충족) → '약한 B' 로 기록만.
결과 전 고정한 분기 규칙 2(B 미확정 · T-B 통과) 적용 → <b>qb_ trunk 로 코덱 → 갈래 off · comm 진행</b>(19:47 시작). 단계 2(보상 반경 56)는 이 규칙으로는 안 감.
D 진단(n_): 행동 잡음 몫이 큼(9.04M 평균 행동 − 표본 = {fm(D_NOISE9, 1, True)} pt), 평균 정책도 4M 뒤 {fm(D_DEC, 1, True)} pt. 전부 시드 1개 · 학습 중 창 값이라 단독 주장 못 함.</div>

<h2>1. 관문 판정 — 마지막 4창 평균(결과 전 고정)</h2>
{gate_table()}
<ul>
<li>§5-3 관문(10-07 스펙) = 마지막 4창 평균 도착 ≥ {GATE_GOAL:.0f} % · 벽 충돌 ≤ {GATE_OCOLL:.0f} %. 창 = update 5개 = {5 * UPD:,} 결정 동안 끝난 에피소드 결과 비율(학습 중 값 = 행동 잡음 포함)</li>
<li>기록만 하는 G1(마지막 창 도착 ≥ 90 · 배 충돌 ≤ 5 · 벽 충돌 ≤ 2): 둘 다 미충족(qa_ {fm(QA[-1]['goal'])} · {fm(QA[-1]['vColl'])} · {fm(QA[-1]['oColl'])} / qb_ {fm(QB[-1]['goal'])} · {fm(QB[-1]['vColl'])} · {fm(QB[-1]['oColl'])})</li>
<li>로그 창에 안 들어간 마지막 update 3개(8.85–9.04M, 관문 계산 밖 · 참고): qa_ 도착 {fm(TA[1])} % · 배 충돌 {fm(TA[2])} % (에피소드 {TA[0]:.0f}개) / qb_ {fm(TB[1])} % · {fm(TB[2])} % ({TB[0]:.0f}개)</li>
</ul>

<h2>2. 가설 B 확정 기준 · 분기 규칙(스펙 §3, 결과 전 고정)</h2>
<table><thead><tr><th>B 확정 조건(전부 충족해야 함)</th><th>기준</th><th>qa_ 값</th><th>결과</th></tr></thead><tbody>
<tr><td>T-A 가 §5-3 관문 통과</td><td class='c'>도착 ≥ {GATE_GOAL:.0f} · 벽 충돌 ≤ {GATE_OCOLL:.0f}</td><td class='c'>{fm(GA)} · {fm(OA)}</td><td class='c'>{ok(A_PASS)}</td></tr>
<tr><td>마지막 4창 도착</td><td class='c'>≥ {B_GOAL:.0f} %</td><td class='c'><b>{fm(GA)}</b> (기준까지 {fm(GA - B_GOAL, 1, True)} %p)</td><td class='c'>{ok(GA >= B_GOAL)}</td></tr>
<tr><td>마지막 4창 배 충돌(n_ {fm(G['n_'][2])} − 20 %p)</td><td class='c'>≤ {B_VCOLL} %</td><td class='c'>{fm(VA)}</td><td class='c'>{ok(VA <= B_VCOLL)}</td></tr>
</tbody></table>
<ul>
<li>판정: <b>B 미확정</b>(3개 가운데 2개 충족, 도착이 {fm(B_GOAL - GA)} %p 모자람). 4창 도착 값 {' · '.join(fm(w['goal']) for w in LA)} — 마지막 창 {fm(LA[-1]['goal'])} % 가 평균을 내림</li>
<li>분기 규칙(위에서부터 첫 번째로 맞는 것 하나): ① B 확정 → 단계 2 — <b>해당 없음</b> · ② B 미확정 · T-B §5-3 통과 → T-B trunk 로 바로 코덱 → off · comm(시드 43, 접두어 qb_ phase1) — <b>적용</b> ·
③ B 미확정 · T-B 불통과 → 멈춤 — 해당 없음</li>
<li>규칙 ② 문구대로 T-A 는 '약한 B'(§5-3 통과, 도착 30–50 %)로 기록만 함. T-A 는 진단 trunk 라 Fig1 뿌리로 못 씀(<code>_diag_only.txt</code>)</li>
<li>기록용(판정 아님) — B 식을 qa_ 의 모든 4창 위치에 대 보면 {len(ROLL_A)}개 위치 가운데 {len(ROLL_A_B)}개가 충족, 4창 도착 최고 {fm(ROLL_A_BEST[2])} % ({ROLL_A_BEST[0]:.2f}–{ROLL_A_BEST[1]:.2f}M).
기준은 끝 4창으로 고정돼 있어 판정은 미확정 그대로임</li>
</ul>

<h2>3. trunk 학습 곡선(로그 값 그대로)</h2>
<div class='fig'>{C_GOAL}</div>
<div class='fig'>{C_VC}</div>
<div class='fig'>{C_REW}</div>
<div class='small'>가로 = 결정 수. 회색 띠 = 관문이 읽는 마지막 4창. 세 런 모두 보상 정의가 같아(v3 + R1 + R2, 반경 300) 결정당 보상을 같은 축에 둠.</div>
{blk_table()}
<ul>
<li>도착 최고(창): n_ {fm(PN['goal'])} % ({PN['dec']:.2f}M) · qa_ {fm(PA['goal'])} % ({PA['dec']:.2f}M) · qb_ {fm(PB['goal'])} % ({PB['dec']:.2f}M)</li>
<li>qb_ 는 4창 평균 최고 {fm(ROLL_B_BEST[2])} % ({ROLL_B_BEST[0]:.2f}–{ROLL_B_BEST[1]:.2f}M) 뒤 끝 4창 {fm(GB)} % 로 내려옴. 3구간 도착 {' → '.join(fm(mean(b, 'goal')) for b in blk(QB))} %, 배 충돌 {' → '.join(fm(mean(b, 'vColl')) for b in blk(QB))} %</li>
<li>qa_ 3구간 도착 {' → '.join(fm(mean(b, 'goal')) for b in blk(QA))} %, 배 충돌 {' → '.join(fm(mean(b, 'vColl')) for b in blk(QA))} %. 시간초과가 끝 4창 {fm(mean(LA, 'TO'))} % 로 남음(qb_ {fm(mean(LB, 'TO'))} %)</li>
<li>결정당 보상 3구간 평균: n_ {' → '.join(fm(mean(b, 'R'), 3) for b in blk(N))} · qa_ {' → '.join(fm(mean(b, 'R'), 3) for b in blk(QA))} · qb_ {' → '.join(fm(mean(b, 'R'), 3) for b in blk(QB))}.
끝 4창: n_ {fm(mean(G['n_'][0], 'R'), 3)} · qa_ {fm(mean(LA, 'R'), 3)} · qb_ {fm(mean(LB, 'R'), 3)}</li>
<li>두 런 모두 도착 30 % 는 넘겼으나 배끼리 충돌이 끝 4창 {fm(VA)} % · {fm(VB)} % 로 남음. G1 수준(도착 90 %)과는 거리가 큼</li>
</ul>

<h2 class='pb'>4. 창별 값(27창, 굵은 줄 = 관문 4창)</h2>
{win_table()}
<div class='small'>도착 · 배 충돌 · 시간초과 = %. R = 결정당 보상. lr = qb_ 학습률(선형 감쇠, 16.06M 에서 3e-5 가 되는 기울기 → trunk 끝 {QB[-1]['lr']:.2e}). n_ · qa_ 는 3e-4 상수.</div>

<h2>5. 토글이 실제로 들어갔는지 — 2M 체크포인트를 직접 열어 확인</h2>
{snap_table()}
<ul>
<li>세 런 공통(스냅샷): seed 43 · <code>action_mode=course</code> · <code>comm_fields=state</code> · <code>ROLE_V2_PRIMARY=cum</code> · <code>ROLE_V2_RES_F6=1</code> · <code>REWARD_RANGE=None</code>(= COMM_RANGE 300) → 스펙 §3 'n_ 과 다른 것 하나' 그대로</li>
<li>학습 로그 헤더에는 이 값들이 안 찍힘 → 체크포인트로만 확인 가능했음. 접두어별 기록 <code>_batch_env.txt</code>: qa_ <code>{BENV['qa']}</code> · qb_ <code>{BENV['qb']}</code></li>
<li>순위 관문: 재실행 없음. <code>_rank_note.txt</code> — {RANKNOTE}</li>
</ul>

<h2>6. D 진단 — n_ trunk 체크포인트(학습 없음, 기록용 · 관문 아님)</h2>
{d_table()}
{ls_table()}
<table><thead><tr><th>기록 기준(스펙 §3, 결과 전 고정)</th><th>기준</th><th>값</th><th>결과</th></tr></thead><tbody>
<tr><td>logstd dim0 9.04M 전문가 평균</td><td class='c'>≤ {fm(D_LOGSTD)}</td><td class='c'>{fm(LS[-1][2], 3)}</td><td class='c'>{'충족(잡음이 스스로 줄었음)' if LS[-1][2] <= D_LOGSTD else '미달(잡음이 스스로 안 줄었음)'}</td></tr>
<tr><td>9.04M 평균 행동 도착 − 표본 도착</td><td class='c'>≥ {D_NOISE:.0f} pt</td><td class='c'>{fm(D_NOISE9, 1, True)}</td><td class='c'>{'충족(행동 잡음 몫 큼)' if D_NOISE9 >= D_NOISE else '미달'}</td></tr>
<tr><td>9.04M 평균 행동 − 4M 평균 행동</td><td class='c'>&lt; {fm(D_DECAY, 0)} pt</td><td class='c'>{fm(D_DEC, 1, True)}</td><td class='c'>{'충족(평균 정책도 나빠짐)' if D_DEC < D_DECAY else '미달'}</td></tr>
<tr><td>critic EVb(λ=1) 9.04M</td><td class='c'>기준 없음</td><td class='c'>{fm(EV1, 3)} (λ=0.95 {fm(EV95, 3)} · ret_std {fm(RETSTD)})</td><td class='c'>값만 기록</td></tr>
</tbody></table>
<ul>
<li>평가 조건(네 건 모두 <code>[run]</code> 줄 확인): 256 env · 결정 3000 · burn-in 2400 · drain 3000 · 평가 시드 999. D 실행 {D_T0}–{D_T1}</li>
<li>4M 에서도 평균 − 표본 = {fm(D_NOISE4, 1, True)} pt. logstd 는 전문가 0 만 {fm(LS[0][1][0], 3)} → {fm(LS[-1][1][0], 3)} 로 커지고 나머지 넷은 {fm(min(min(v[1:]) for _, v, _, _ in LS), 3)} ~ {fm(max(max(v[1:]) for _, v, _, _ in LS), 3)} 사이</li>
<li>평균 정책 하락 {fm(D_DEC, 1, True)} pt 는 기준 {fm(D_DECAY, 0)} 을 {fm(abs(D_DEC - D_DECAY))} pt 넘긴 값 — 평가 시드 1개라 여유가 작음</li>
<li>EVb 는 T=32 rollout 부트스트랩 목표에 대한 자기일관성임. critic 이 실제 수익을 맞춘다는 근거로는 못 씀(스펙 명시)</li>
</ul>

<h2>7. 경과 · 실행 기록</h2>
<ul>
<li>qa_: {t_of('qa', '시작 smoke')} smoke(preflight 포함) → {t_of('qa', '끝 smoke')} 통과 → trunk {t_of('qa', '시작 trunk')}–{t_of('qa', '끝 trunk')} (학습기 기록 {fm(MA)}분, 평균 {TRUNK_DEC / (MA * 60):.0f} 결정/초) → 관문 {t_of('qa', '끝 gate')}</li>
<li>qb_: {t_of('qb', '시작 smoke')} smoke → {t_of('qb', '끝 smoke')} 통과 → trunk {t_of('qb', '시작 trunk')}–{t_of('qb', '끝 trunk')} ({fm(MB)}분, 평균 {TRUNK_DEC / (MB * 60):.0f} 결정/초) → 관문 {t_of('qb', '끝 gate')}</li>
{''.join(f'<li>{ln}</li>' for ln in NOTE)}
</ul>

<h2>8. 한계 — 그대로 읽을 것</h2>
<ul>
<li>시드 1개(43). 세 런의 차이를 시드 간 흔들림과 가를 수 없음. T-A 의 B 기준 미달 폭({fm(B_GOAL - GA)} %p)도 시드 1개 창 값 위의 숫자임</li>
<li>값은 학습 중 창 통계(행동 잡음 포함). qa_ · qb_ 는 고정 장면 평가를 아직 안 돌았음. qb_ 는 초기 잡음이 작아(logstd −2.0) 창 값이 n_ · qa_ 보다 평균 행동 성능에 가깝게 나올 수 있음 → 창 값끼리의 차를 정책 차로 바로 읽으면 안 됨(미측정)</li>
<li>qb_ 는 토글 2개(logstd 초기값 · LR 감쇠)를 같이 켬 → 어느 쪽 효과인지 이 배치로는 못 가름</li>
<li>qb_ 도착은 창 최고 {fm(PB['goal'])} % ({PB['dec']:.2f}M) · 3구간 {' → '.join(fm(mean(b, 'goal')) for b in blk(QB))} % 로, 가운데 구간 뒤 끝으로 가며 내려옴. 관문은 통과했지만 '안정적으로 오른다'는 근거는 아님. 갈래(9.04M → 16.06M)에서 이어지는지는 phase1 곡선에서 봄</li>
<li>n_ 은 10-09 실행(8d6bc88), qa_ · qb_ 는 10-13 실행(b6a4341). 토글 기본값은 비트동일 규약이지만 실행 날짜 · GPU · 할당기 설정이 다름(§7)</li>
<li>순위 관문은 재실행 없이 n_ 기록(0/3 도구 결함 · 타각 모드 2/3)과 10-09 저자 override 를 그대로 적용함(스펙 §3)</li>
<li>원인(왜 300 m 를 봐도 도착 50 % 에 못 미치는지 · 왜 배 충돌이 45–58 % 로 남는지)은 재지 않았음. 관문 기준 · 스크립트는 안 건드렸음</li>
</ul>

<h2>9. 다음</h2>
<ul>
<li>진행 중: qb_ phase1 = 코덱(p46, 관문 통과하는 가장 작은 k) → 갈래 off · comm(9.04M → 16.06M, 같은 토글) → 분기 검사 → 주 평가 · Woerner → 고정 장면 곡선(M) → 표. 끝나면 별도 PDF</li>
<li>저자 결정 사항(이 보고가 정하지 않음): T-A '약한 B'({fm(GA)} %, 기준 {B_GOAL:.0f}) 를 근거로 단계 2(보상 반경 56)를 따로 돌릴지 — 규칙 ② 로는 안 감 · qb_ phase1 뒤 시드 3개 확증 여부</li>
</ul>
<div class='small'>자료: runs/2026-10-02_fig1/out/ 의 qa/ · qb/ · n_trunk_d6_s43.log · diag_trunk_n/ · q_stage1_notes/ · 서버 Python/_repro_out_qa · _repro_out_qb · _repro_out_n/diag_trunk ·
체크포인트 VESSEL_checkpoints/comm_intent/q[ab]_trunk_d6_s43*.pt · 생성 report/make_report_trunk_q.py</div>
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
print('관문 qa_', fm(GA), fm(VA), fm(OA), '| qb_', fm(GB), fm(VB), fm(OB), '| n_', fm(G['n_'][1]), fm(G['n_'][2]), '| B 확정', B_CONF, '규칙', RULE,
      '| qa_ B 식 충족 위치', len(ROLL_A_B), '/', len(ROLL_A), '| D 잡음', fm(D_NOISE9, 1, True), '퇴화', fm(D_DEC, 1, True), 'EVb1', fm(EV1, 3))
print(out_pdf)
