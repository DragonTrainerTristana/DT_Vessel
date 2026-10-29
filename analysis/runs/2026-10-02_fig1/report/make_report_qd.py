# 2026-10-28: qd_ trunk(위험 비용 56 m + 떨림 수정 4종, 시드 43) 결과 PDF. 스펙 docs/superpowers/specs/2026-10-28-qd-trunk-design.md §2(결과 전 고정).
# 숫자는 전부 로그에서 읽음(runs/2026-10-02_fig1/out/):
#   qd/            qd_trunk_d6_s43.log(학습 창 · [ppo] 계측 줄) · _ep.csv · _gate.log · _timeline.txt · _batch_env.txt · _rank.log · _rank_v5.json
#   meaneval_qd/   sample_/mean_qd_trunk_d6_{4,9.04397}M.txt · logstd_qd_trunk_d6.txt · _meaneval_qd_.md · _log.txt
#   qd_notes/      runnote.txt(실행 기록) · snap_2M.txt(2M 체크포인트 스냅샷 직접 연 값)
#   비교: n_trunk_d6_s43.log · diag_trunk_n/(n_ 4M · 9.04M 같은 인자 평가) · qa/ · qb/ 학습 로그 · meaneval_qa/ · meaneval_qb/(R0)
# 표 값은 meaneval_ckpts.sh 가 만든 _meaneval_qd_.md 행과, 관문 값은 _gate.log 와 대조(assert).
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_qd.py [출력 폴더]
import csv, io, math, os, re, subprocess, sys

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', 'out'))
NAME = '2026-10-28_qd_trunk_위험56m_떨림수정_결과'
RUN_ARGS = 'burnin=2400 dec=3000 drain=3000 ring=1.0 crossing=0 seed=999 envs=256 vessels=16'
SEED = 43
UPD = 65536
TRUNK_DEC = 9043968                               # = 138 update
GATE_GOAL, GATE_OCOLL, GATE_N = 30.0, 5.0, 4      # 10-07 스펙 §5-3 (_run_f.sh gate9)
S1_MEAN9, S2_DROP, S3_GAP = 65.0, 5.0, 10.0       # 10-28 스펙 §2 처방 성공(셋 다)
LOGSTD_MIN = -3.0                                 # qd_ 학습 설정(스냅샷으로 확인)


def text(*p):
    with io.open(os.path.join(OUT, *p), encoding='utf-8', errors='replace') as fh:
        return fh.read()


def f(v, nd=1, sign=False):
    if v is None:
        return 'n/a'
    s = f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'
    return s.replace('-', '−')


# ── 평가 파일
LINE = re.compile(r'^(\S+\.pt)\s+\| arm=ON\s+\| goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%\s+oColl=\s*([\d.]+)%\s+TO=\s*([\d.]+)%\s+\|\s+(\d+) eps\b.*?epReward=\s*(-?[\d.]+)', re.M)
PDL = re.compile(r'\[pair-detail\] 통과최소거리 중앙\s+([\d.]+)m \| 첫 변침\(\|.{1,3}\|>0\.2\)까지 중앙\s+([\d.]+)s \| 무변침 조우\s+([\d.]+)%')
NBL = re.compile(r'\[null-baseline\] 조우 없는 배의 \|목표대비 오프셋\| 중앙\s+([\d.]+)deg')


def parse(rel, ckpt, mean_mode, rr):
    t = text(rel)
    m, run, pd, nb = LINE.search(t), re.search(r'\[run\] (.*)', t), PDL.search(t), NBL.search(t)
    assert m and run and run.group(1).strip().startswith(RUN_ARGS), (rel, run and run.group(1))
    assert m.group(1) == ckpt, (rel, m.group(1), ckpt)
    assert ('[eval_mean] student drives with tanh(mean)' in t) == mean_mode, rel
    assert f'reward_range={rr}' in t, (rel, rr)
    assert pd and nb, rel
    return dict(goal=float(m.group(2)), vColl=float(m.group(3)), oColl=float(m.group(4)), TO=float(m.group(5)), eps=int(m.group(6)),
                epReward=float(m.group(7)), minPass=float(pd.group(1)), turn=float(pd.group(2)), noTurn=float(pd.group(3)), nullOff=float(nb.group(1)))


XS = ('4', '9.04397')
ck_name = lambda pre, x: f'{pre}trunk_d6_s{SEED}.pt' if x == '9.04397' else f'{pre}trunk_d6_s{SEED}.step{x}M.pt'  # noqa: E731
EV = {}   # (pre, x) -> (sample, mean)
for x in XS:
    EV[('qd_', x)] = (parse(os.path.join('meaneval_qd', f'sample_qd_trunk_d6_{x}M.txt'), ck_name('qd_', x), False, '56.0'),
                      parse(os.path.join('meaneval_qd', f'mean_qd_trunk_d6_{x}M.txt'), ck_name('qd_', x), True, '56.0'))
    for pre in ('qb_', 'qa_'):
        d = f'meaneval_{pre[:-1]}'
        EV[(pre, x)] = (parse(os.path.join(d, f'sample_{pre}trunk_d6_{x}M.txt'), ck_name(pre, x), False, '300.0'),
                        parse(os.path.join(d, f'mean_{pre}trunk_d6_{x}M.txt'), ck_name(pre, x), True, '300.0'))
    tag = '4M' if x == '4' else '9M'
    EV[('n_', x)] = (parse(os.path.join('diag_trunk_n', f'eval_sample_{tag}.txt'), ck_name('n_', x), False, '300.0'),
                     parse(os.path.join('diag_trunk_n', f'eval_mean_{tag}.txt'), ck_name('n_', x), True, '300.0'))

# 스크립트 표(_meaneval_qd_.md)와 대조
MD = text('meaneval_qd', '_meaneval_qd_.md')
for x in XS:
    s, a = EV[('qd_', x)]
    row = f"| trunk_d6 | {float(x):g}M | {s['goal']:.1f} / {a['goal']:.1f} ({a['goal'] - s['goal']:+.1f}) | {s['vColl']:.1f} / {a['vColl']:.1f} | {s['epReward']:.0f} / {a['epReward']:.0f} |"
    assert row in MD, row
assert MD.count('\n| ') - 2 == len(XS) + 1, MD   # 체크포인트 행 2 + K 행 1(머리줄 2개 제외)

# logstd 덤프(meaneval 이 체크포인트 전부에 대해 씀)
LSL = re.compile(r'^(\S+\.pt)\s+steps=(\d+) logstd0=([-\d. ]+?) mean=(-?[\d.]+) sigma_course=([\d.]+)deg', re.M)


def sig_of(rel, ck):
    hit = [m for m in LSL.finditer(text(rel)) if m.group(1) == ck]
    assert len(hit) == 1, (rel, ck, len(hit))
    return float(hit[0].group(5))


SIG = {}
for x in XS:
    SIG[('qd_', x)] = sig_of(os.path.join('meaneval_qd', 'logstd_qd_trunk_d6.txt'), ck_name('qd_', x))
    SIG[('qb_', x)] = sig_of(os.path.join('meaneval_qb', 'logstd_qb_trunk_d6.txt'), ck_name('qb_', x))
    SIG[('qa_', x)] = sig_of(os.path.join('meaneval_qa', 'logstd_qa_trunk_d6.txt'), ck_name('qa_', x))
    SIG[('n_', x)] = sig_of(os.path.join('diag_trunk_n', 'logstd.txt'), ck_name('n_', x))
LS_QD = [(int(m.group(2)), float(m.group(5))) for m in LSL.finditer(text('meaneval_qd', 'logstd_qd_trunk_d6.txt'))]
assert len(LS_QD) == 5 and LS_QD[-1][0] == TRUNK_DEC, LS_QD

# ── 학습 창(로그 한 줄 = update 5개)
WIN = re.compile(r'^\[ON\] dec=([\d.]+)M \| ep=(\d+) len~(\d+) \| goal=([\d.]+)% vColl=([\d.]+)% oColl=([\d.]+)% TO=([\d.]+)% '
                 r'\| R=(-?[\d.]+) \| (\d+) dec/s \| epRet=(-?[\d.]+)(?: \| lr=([\d.e+-]+))?')
KEYS = ('dec', 'ep', 'len', 'goal', 'vColl', 'oColl', 'TO', 'R', 'dps', 'epRet', 'lr')


def windows(*p):
    return [{k: (float(x) if x is not None else None) for k, x in zip(KEYS, m.groups())}
            for m in (WIN.match(ln) for ln in text(*p).splitlines()) if m]


def saved(*p):
    m = re.search(r'saved -> \S+ \(([\d.]+)M decisions, ([\d.]+)min\)', text(*p))
    return float(m.group(1)), float(m.group(2))


def mean(rows, k):
    t = 0.0
    for r in rows:
        t += r[k]
    return t / len(rows)


TR = {'qd_': windows('qd', 'qd_trunk_d6_s43.log'), 'n_': windows('n_trunk_d6_s43.log'),
      'qb_': windows('qb', 'qb_trunk_d6_s43.log'), 'qa_': windows('qa', 'qa_trunk_d6_s43.log')}
assert all(len(W) == 27 for W in TR.values()), {k: len(W) for k, W in TR.items()}
QD = TR['qd_']
QD_DEC, QD_MIN = saved('qd', 'qd_trunk_d6_s43.log')
assert QD_DEC == 9.04, QD_DEC
with io.open(os.path.join(OUT, 'qd', 'qd_trunk_d6_s43_ep.csv'), encoding='utf-8') as fh:
    EP = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
assert len(EP) == 138 and EP[-1]['step'] == TRUNK_DEC
for i, w in enumerate(QD):
    assert sum(r['n_ep'] for r in EP[5 * i:5 * i + 5]) == w['ep'], i
LOG_QD = text('qd', 'qd_trunk_d6_s43.log')
assert '[resume]' not in LOG_QD and '[bc]' not in LOG_QD and 'DAgger' not in LOG_QD   # 처음부터 순수 PPO
assert all(w['lr'] is None for w in QD)                                                 # LR 감쇠 없음


def gate(W):
    L = W[-GATE_N:]
    g, v, o = mean(L, 'goal'), mean(L, 'vColl'), mean(L, 'oColl')
    return L, g, v, o, (g >= GATE_GOAL and o <= GATE_OCOLL)


GL, GG, GV, GO, TG = gate(QD)
m = re.search(rf'^s{SEED} goal=([\d.]+) vColl=([\d.]+) oColl=([\d.]+) 창=(\d+) (통과|불통과)', text('qd', '_gate.log'), re.M)
assert m and (m.group(1), m.group(2), m.group(3), m.group(4)) == (f'{GG:.1f}', f'{GV:.1f}', f'{GO:.1f}', str(GATE_N)), (m and m.groups(), GG, GV, GO)
assert (m.group(5) == '통과') == TG
GREF = {pre: gate(TR[pre]) for pre in ('n_', 'qb_', 'qa_')}
blk = lambda W: [W[0:9], W[9:18], W[18:27]]   # noqa: E731  창 9개씩 3구간(결과 보고 고르지 않으려고 3등분)
PEAK = max(QD, key=lambda w: w['goal'])

# ── [ppo] 계측 줄(update 마다)
PPO = re.compile(r'^\[ppo\] upd=(\d+) approx_kl=([\d.]+) clipfrac=([\d.]+) entropy=(-?[\d.]+) logstd0_raw=\[([-\d. ]+)\] lr=([\d.e+-]+) dec=([\d.]+)M', re.M)
PP = [dict(upd=int(m.group(1)), kl=float(m.group(2)), cf=float(m.group(3)), ent=float(m.group(4)),
           ls=[float(v) for v in m.group(5).split()], lr=float(m.group(6)), dec=float(m.group(7))) for m in PPO.finditer(LOG_QD)]
assert [p['upd'] for p in PP] == list(range(138)) and all(len(p['ls']) == 5 for p in PP), len(PP)
assert all(p['lr'] == 3e-4 for p in PP)
sig_eff = lambda raw: 120.0 * math.exp(min(max(raw, LOGSTD_MIN), 0.0))  # noqa: E731  forward 의 clamp 를 지난 σ(침로 °, tanh 전)
PP_BLK = [PP[0:46], PP[46:92], PP[92:138]]
FLOOR = [sum(1 for p in PP if p['ls'][e] <= LOGSTD_MIN) for e in range(5)]   # 하한에 닿은 update 수(전문가별)

# ── 스냅샷(2M 체크포인트 직접 연 값)
SNAP = dict(re.findall(r'^\s+cfg_snapshot\.(\S+) = (.*)$', text('qd_notes', 'snap_2M.txt'), re.M))
assert (SNAP['entropy_bonus'], SNAP['logstd_min'], SNAP['logstd_init0'], SNAP['sim.REWARD_RANGE']) == ('0.0', '-3.0', '-2.0', '56.0'), SNAP
assert (SNAP['partner_range'], SNAP['action_mode'], SNAP['lr_decay_dec'], SNAP['noise_hold'], SNAP['comm_range']) == ('56.0', 'course', '0.0', '0', '300.0'), SNAP
assert (SNAP['sim.ROLE_V2_PRIMARY'], SNAP['sim.ROLE_V2_RES_F6']) == ('cum', '1'), SNAP
BENV = text('qd', '_batch_env.txt').strip()
assert BENV == 'ARPA=56 RANK=v5 REWARD_RANGE=56 EXTRA=[VESSEL_ENTROPY_BONUS=0 VESSEL_LOGSTD_MIN=-3.0 VESSEL_LOGSTD_INIT0=-2.0 VESSEL_GRAD_TELEMETRY=1]', BENV

# ── 순위 관문 v5(학습과 동시, GPU 1)
RLOG = text('qd', '_rank.log')
RSEED = re.findall(r'^\s*\[v5\] seed (\d): (.*?) → (PASS|FAIL)\s*$', RLOG, re.M)
assert len(RSEED) == 3, RSEED
RPASS = 'RANK GATE v2: PASS' in RLOG
assert RPASS == ('RANK GATE v2: ★FAIL' not in RLOG)
RN = sum(v == 'PASS' for *_, v in RSEED)
RSEC = re.search(r'\[v5\] (\d+) rows in (\d+)s', RLOG)
assert RSEC, 'rank rows'

# ── 처방 성공(스펙 §2, 셋 다)
S4, A4 = EV[('qd_', '4')]
S9, A9 = EV[('qd_', '9.04397')]
DROP = A4['goal'] - A9['goal']
GAP9 = A9['goal'] - S9['goal']
C1, C2, C3 = A9['goal'] >= S1_MEAN9, DROP <= S2_DROP, GAP9 <= S3_GAP
SUCCESS = C1 and C2 and C3
NOK = sum((C1, C2, C3))

# ── 경과
TL = [ln for ln in text('qd', '_timeline.txt').splitlines() if ln.startswith('2026-10-28')]
assert '2cd2fbd' in TL[0] and '배치 qd_ trunk' in TL[0], TL[0]


def t_of(pat):
    hit = [ln for ln in TL if ln[20:].startswith(pat)]
    return hit[0][11:19] if hit else None


STOP = [ln[20:] for ln in TL if '★' in ln]
NOTE = [ln[2:] for ln in text('qd_notes', 'runnote.txt').splitlines()[1:] if ln.startswith('- ')]

# ── 그림(SVG, 로그 값 그대로)
C_QD, C_N, C_QB, C_QA = '#a8322a', '#3f8f5a', '#9c9a94', '#7a4fb5'


def chart(series, x0, x1, xticks, y0, y1, yticks, w=700, h=150, hlines=(), shade=None, ynd=0, xfmt=lambda d: f'{d:g}M'):
    ml, mr, mt, mb = 36, 8, 18, 18
    X = lambda d: ml + (w - ml - mr) * (d - x0) / (x1 - x0)  # noqa: E731
    Y = lambda v: mt + (h - mt - mb) * (y1 - min(max(v, y0), y1)) / (y1 - y0)  # noqa: E731
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='100%'>"]
    if shade:
        s.append(f"<rect x='{X(shade[0]):.1f}' y='{mt}' width='{X(shade[1]) - X(shade[0]):.1f}' height='{h - mt - mb}' fill='#efeee9'/>")
    for t in yticks:
        s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' stroke='#e4e3df' stroke-width='0.6'/>")
        s.append(f"<text x='{ml - 4}' y='{Y(t) + 2.6:.1f}' font-size='7.2' text-anchor='end' fill='#52514e'>{f(t, ynd)}</text>")
    for d in xticks:
        s.append(f"<text x='{X(d):.1f}' y='{h - 5}' font-size='7.2' text-anchor='middle' fill='#52514e'>{xfmt(d)}</text>")
    s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{h - mb}' y2='{h - mb}' stroke='#0b0b0b' stroke-width='0.7'/>")
    for hv, hl in hlines:
        s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{Y(hv):.1f}' y2='{Y(hv):.1f}' stroke='#0b0b0b' stroke-width='0.7' stroke-dasharray='2 2'/>")
        s.append(f"<text x='{w - mr - 2}' y='{Y(hv) - 2.5:.1f}' font-size='7.2' text-anchor='end' fill='#0b0b0b'>{hl}</text>")
    lx = ml + 4
    for lab, pts, col, sw, dash, dots in series:
        p = ' '.join(f"{X(d):.1f},{Y(v):.1f}" for d, v in pts)
        da = f" stroke-dasharray='{dash}'" if dash else ''
        s.append(f"<polyline points='{p}' fill='none' stroke='{col}' stroke-width='{sw}'{da}/>")
        if dots:
            s += [f"<circle cx='{X(d):.1f}' cy='{Y(v):.1f}' r='2.0' fill='{'#fff' if dash else col}' stroke='{col}' stroke-width='1'/>" for d, v in pts]
        s.append(f"<line x1='{lx}' x2='{lx + 16}' y1='8' y2='8' stroke='{col}' stroke-width='{sw}'{da}/>")
        s.append(f"<text x='{lx + 19}' y='10.6' font-size='7.4' fill='#0b0b0b'>{lab}</text>")
        lx += 19 + 7.4 * len(lab) + 12
    return ''.join(s) + '</svg>'


wpts = lambda W, k: [(w['dec'], w[k]) for w in W]  # noqa: E731
SHADE = (QD[-GATE_N]['dec'] - 0.33, QD[-1]['dec'])
STY = (('qd_', C_QD, 1.8, ''), ('n_', C_N, 1.2, '4 2'), ('qb_', C_QB, 1.2, ''), ('qa_', C_QA, 1.2, '4 2'))
G_GOAL = chart([(f'{pre} 도착', wpts(TR[pre], 'goal'), col, sw, da, False) for pre, col, sw, da in STY],
               0, 9.2, range(0, 10), 0, 100, [0, 20, 40, 60, 80, 100], hlines=((GATE_GOAL, '관문 도착 30 %'),), shade=SHADE)
G_VC = chart([(f'{pre} 배 충돌', wpts(TR[pre], 'vColl'), col, sw, da, False) for pre, col, sw, da in STY],
             0, 9.2, range(0, 10), 0, 100, [0, 20, 40, 60, 80, 100], h=128, shade=SHADE)
G_TO = chart([(f'{pre} 시간초과', wpts(TR[pre], 'TO'), col, sw, da, False) for pre, col, sw, da in STY],
             0, 9.2, range(0, 10), 0, 60, [0, 20, 40, 60], h=118, shade=SHADE)
EXC = ('#a8322a', '#1b6fd6', '#3f8f5a', '#d9822b', '#7a4fb5')
EXN = ('전문가 0(조우 없음)', '1(정면)', '2(횡단 유지)', '3(횡단 양보)', '4(추월)')
G_SIG = chart([(EXN[e], [(p['dec'], sig_eff(p['ls'][e])) for p in PP], EXC[e], 1.3 if e else 1.8, '', False) for e in range(5)],
              0, 9.2, range(0, 10), 0, 40, [0, 10, 20, 30, 40], h=140, hlines=((sig_eff(LOGSTD_MIN), f'하한 {f(sig_eff(LOGSTD_MIN))}°'),))
KLMAX = max(p['kl'] for p in PP)
G_KL = chart([('approx_kl', [(p['dec'], p['kl']) for p in PP], '#0b0b0b', 1.3, '', False),
              ('clipfrac ÷ 10', [(p['dec'], p['cf'] / 10) for p in PP], '#d9822b', 1.1, '3 2', False)],
             0, 9.2, range(0, 10), 0, max(0.02, math.ceil(KLMAX * 200) / 200), [0, 0.005, 0.01, 0.015, 0.02], h=118, ynd=3)
EVX = [4.06, 9.04]
G_EV = chart([(f'{pre} 평균', [(xx, EV[(pre, x)][1]['goal']) for xx, x in zip(EVX, XS)], col, 1.6, '', True) for pre, col, _, _ in STY]
             + [(f'{pre} 표본', [(xx, EV[(pre, x)][0]['goal']) for xx, x in zip(EVX, XS)], col, 1.2, '3 2', True) for pre, col, _, _ in STY],
             3.3, 9.8, EVX, 0, 80, [0, 20, 40, 60, 80], h=165, hlines=((S1_MEAN9, '성공 1: 9.04M 평균 ≥ 65 %'),),
             xfmt=lambda d: f'{d:.2f}M')

ok = lambda b: '충족' if b else '<b>미달</b>'  # noqa: E731
LAB = {'qd_': 'qd_ (이번)', 'n_': 'n_ (기준)', 'qb_': 'qb_', 'qa_': 'qa_'}
DESC = {'qd_': '위험 비용 56 m · 엔트로피 0 · logstd 하한 −3.0 · 초기 −2.0',
        'n_': '위험 비용 300 m · 토글 없음', 'qb_': '300 m · logstd 초기 −2.0 · LR 감쇠', 'qa_': '300 m · OFF 가 300 m 참값(ARPA@300)'}


def verdict_table():
    rows = [('trunk 관문(10-07 §5-3)', f'마지막 4창 도착 ≥ {f(GATE_GOAL, 0)} % · 벽 ≤ {f(GATE_OCOLL, 0)} %',
             f'도착 {f(GG)} % · 벽 {f(GO)} %(4창 {" · ".join(f(w["goal"]) for w in GL)})', '통과' if TG else '<b>불통과</b>'),
            ('순위 관문 v5(동시 실행)', '보상 순위 규칙 (i)–(v), 시드 3개 모두', f'{RN}/3 시드 통과', '통과' if RPASS else '<b>불통과</b>'),
            ('성공 1', f'9.04M 평균 행동 도착 ≥ {f(S1_MEAN9, 0)} %', f'{f(A9["goal"])} %', ok(C1)),
            ('성공 2', f'4M → 9.04M 평균 행동 도착 하락 ≤ {f(S2_DROP, 0)} pt', f'{f(A4["goal"])} → {f(A9["goal"])} %(하락 {f(DROP, 1, True)} pt{" = 오름" if DROP < 0 else ""})', ok(C2)),
            ('성공 3', f'9.04M 평균 − 표본 도착 ≤ {f(S3_GAP, 0)} pt', f'{f(A9["goal"])} − {f(S9["goal"])} = {f(GAP9, 1, True)} pt', ok(C3)),
            ('처방 성공(셋 다)', '성공 1 · 2 · 3', f'{NOK}/3', '<b>성공</b>' if SUCCESS else '<b>미달</b>')]
    h = "<table><thead><tr><th>항목</th><th>기준(결과 전 고정)</th><th>값</th><th>판정</th></tr></thead><tbody>"
    for r in rows:
        h += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td class='c'>{r[2]}</td><td class='c'>{r[3]}</td></tr>"
    return h + "</tbody></table>"


def ev_table():
    h = ("<table><thead><tr><th>trunk</th><th>체크포인트</th><th>σ 침로 °</th><th>도착 % 표본</th><th>도착 % 평균</th><th>평균 − 표본</th>"
         "<th>배 충돌 % 표본 / 평균</th><th>벽 · 시간초과 % 평균</th><th>첫 변침 s 표본 / 평균</th><th>무조우 오프셋 ° 표본 / 평균</th><th>에피소드 표본 / 평균</th></tr></thead><tbody>")
    for pre in ('qd_', 'n_', 'qb_', 'qa_'):
        for x in XS:
            s, a = EV[(pre, x)]
            cls = " class='ref'" if pre != 'qd_' else ''
            h += (f"<tr{cls}><td>{LAB[pre]}</td><td class='c'>{float(x):.2f}M</td><td class='c'>{f(SIG[(pre, x)])}</td><td class='c'>{f(s['goal'])}</td>"
                  f"<td class='c'><b>{f(a['goal'])}</b></td><td class='c'>{f(a['goal'] - s['goal'], 1, True)}</td><td class='c'>{f(s['vColl'])} / {f(a['vColl'])}</td>"
                  f"<td class='c'>{f(a['oColl'])} · {f(a['TO'])}</td><td class='c'>{f(s['turn'])} / {f(a['turn'])}</td><td class='c'>{f(s['nullOff'])} / {f(a['nullOff'])}</td>"
                  f"<td class='c'>{s['eps']} / {a['eps']}</td></tr>")
    return h + "</tbody></table>"


def gate_table():
    h = ("<table><thead><tr><th>trunk(시드 43)</th><th>n_ 과 다른 것</th><th>도착 %</th><th>배 충돌 %</th><th>벽 %</th><th>시간초과 %</th><th>4창 도착 값</th><th>§5-3</th></tr></thead><tbody>")
    for pre in ('qd_', 'n_', 'qb_', 'qa_'):
        L, g, v, o, p = (GL, GG, GV, GO, TG) if pre == 'qd_' else GREF[pre]
        cls = " class='ref'" if pre != 'qd_' else ''
        h += (f"<tr{cls}><td>{LAB[pre]}</td><td>{DESC[pre]}</td><td class='c'><b>{f(g)}</b></td><td class='c'>{f(v)}</td><td class='c'>{f(o)}</td>"
              f"<td class='c'>{f(mean(L, 'TO'))}</td><td class='c'>{' · '.join(f(w['goal']) for w in L)}</td><td class='c'>{'통과' if p else '불통과'}</td></tr>")
    return h + "</tbody></table>"


def blk_table():
    h = ("<table><thead><tr><th>구간(창 9개씩)</th><th>도착 %</th><th>배 충돌 %</th><th>벽 %</th><th>시간초과 %</th><th>도착 최저–최고</th>"
         "<th>approx_kl</th><th>clipfrac</th><th>σ 전문가 0 °</th><th>σ 전문가 1–4 평균 °</th></tr></thead><tbody>")
    for b, pb in zip(blk(QD), PP_BLK):
        s0 = sum(sig_eff(p['ls'][0]) for p in pb) / len(pb)
        s14 = sum(sig_eff(v) for p in pb for v in p['ls'][1:]) / (4 * len(pb))
        h += (f"<tr><td class='c'>{b[0]['dec'] - 0.33:.2f}–{b[-1]['dec']:.2f}M</td><td class='c'>{f(mean(b, 'goal'))}</td><td class='c'>{f(mean(b, 'vColl'))}</td>"
              f"<td class='c'>{f(mean(b, 'oColl'))}</td><td class='c'>{f(mean(b, 'TO'))}</td><td class='c'>{f(min(w['goal'] for w in b))}–{f(max(w['goal'] for w in b))}</td>"
              f"<td class='c'>{f(sum(p['kl'] for p in pb) / len(pb), 4)}</td><td class='c'>{f(sum(p['cf'] for p in pb) / len(pb), 3)}</td>"
              f"<td class='c'>{f(s0)}</td><td class='c'>{f(s14)}</td></tr>")
    return h + "</tbody></table>"


def rank_table():
    h = "<table><thead><tr><th>시드</th><th>(i)</th><th>(ii)</th><th>(iii)</th><th>(iv)</th><th>(v)</th><th>판정</th></tr></thead><tbody>"
    for sd, body, v in RSEED:
        parts = [p.strip() for p in body.split(' | ')]
        assert len(parts) == 5, parts
        h += f"<tr><td class='c'>{sd}</td>" + ''.join(f"<td class='small'>{p}</td>" for p in parts) + f"<td class='c' style='white-space:nowrap'>{'통과' if v == 'PASS' else '<b>불통과</b>'}</td></tr>"
    return h + "</tbody></table>"


if SUCCESS and RPASS:
    NEXT = '스펙 §4 성공 경로: (a) qd_ 갈래 off · comm · (b) qe_ = qd_ + 연속 위험 예고 PBRS · (c) 하나씩 뺀 trunk 를 병렬로 — 저자 실행 지시 대기'
elif SUCCESS:
    NEXT = '처방 성공 기준은 충족, 순위 관문 v5 불통과 → 스펙 §1 대로 갈래로 안 감, 저자 결정 대기'
else:
    NEXT = '스펙 §4 실패 경로(관문 통과 · 성공 기준 미달 포함): 행동공간 증분형 침로(P14) 설계 · 구현 — 저자 확인 뒤 진행'

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
td.small { font-size: 6.9pt; color: #2b2a28; }
tr.ref td { color: #52514e; background: #f7f6f2; }
code { font-family: Consolas, monospace; font-size: 8pt; background: #f4f3ef; padding: 0 2px; }
.small { color: #52514e; font-size: 7.9pt; }
.fig { margin: 4px 0 0 0; }
.pb { page-break-before: always; }
"""
N9 = EV[('n_', '9.04397')]
HTML = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{NAME}</title><style>{CSS}</style></head><body>
<h1>qd_ trunk — 위험 비용 56 m + 떨림 수정 (시드 43)</h1>
<div class='sub1'>2026-10-28 · LICS 서버 · 코드 feat/fig1-latent 2cd2fbd · 스펙 2026-10-28-qd-trunk-design.md(결과 전 고정) · 흉내 없음 순수 PPO 9.04M 결정(138 update) ·
평가 = meaneval_ckpts.sh(256 env · 3000 결정 · burn-in 2400 · drain 3000 · 평가 시드 999), 4M · 9.04M 만 · 시드 1개 = 단독 주장 아님(다음 실험 선택용)</div>

<div class='box'><b>한 줄 결론</b> — trunk 관문 {'통과' if TG else '<b>불통과</b>'}(마지막 4창 도착 {f(GG)} %) · 순위 관문 v5 {'통과' if RPASS else '<b>불통과</b>'}({RN}/3 시드) ·
<b>처방 성공 {'충족' if SUCCESS else f'미달({NOK}/3)'}</b>: 9.04M 평균 행동 도착 {f(A9['goal'])} %(기준 ≥ 65, n_ {f(N9[1]['goal'])}) {ok(C1)} ·
4M → 9.04M 하락 {f(DROP, 1, True)} pt(≤ 5) {ok(C2)} · 평균 − 표본 격차 {f(GAP9, 1, True)} pt(≤ 10, n_ {f(N9[1]['goal'] - N9[0]['goal'], 1, True)}) {ok(C3)}.
9.04M 표본 도착 {f(S9['goal'])} %(n_ {f(N9[0]['goal'])}) · 배 충돌 표본 {f(S9['vColl'])} / 평균 {f(A9['vColl'])} % · σ 침로 {f(SIG[('qd_', '9.04397')])}°.
네 가지를 한 번에 바꿔서 어느 것이 효과인지는 이 배치로 모름. 다음: {NEXT}.</div>

<h2>1. 판정 — 스펙 §2(결과 전 고정)</h2>
{verdict_table()}
<ul>
<li>처방 성공은 셋 다 충족이어야 함. 평균 행동 = tanh(mean)(<code>eval_mean.py</code>), 표본 = 학습 정책 그대로. 나머지 설정 · 장면 · 평가 시드 같음</li>
<li>순위 관문 v5 = v4 + 보상 반경 56 m, 타각 모드, 학습과 동시(GPU 1, {RSEC.group(1)}행 {int(RSEC.group(2)) // 60}분). 불통과면 기록만 · 갈래 안 감(스펙 §1)</li>
</ul>

<h2>2. 평가 — 4M · 9.04M 표본 / 평균 행동 (회색 = 비교, 다른 날 같은 인자로 잰 값)</h2>
{ev_table()}
<div class='fig'>{G_EV}</div>
<ul>
<li>σ 침로 = <code>logstd_dump.py</code> sigma_course = 120° × exp(행동 0번 logstd 전문가 5개 평균), tanh 전 근사값</li>
<li>첫 변침 = 짝 조우에서 |δ̄| &gt; 0.2 까지 중앙 시간, 무조우 오프셋 = 조우 없는 배의 |목표 대비 침로 오프셋| 중앙(작을수록 목표로 곧게) — 둘 다 보고만</li>
<li>비교 trunk 는 위험 비용 반경이 300 m(qd_ 는 56 m) → 도착 · 충돌 · 시간초과는 같은 지표, epReward 는 정의가 달라 안 비교함(표에 안 넣음)</li>
</ul>

<h2 class='pb'>3. 학습 창 — 관문이 읽는 값(로그 한 줄 = update 5개, 회색 띠 = 관문 4창)</h2>
{gate_table()}
<div class='fig'>{G_GOAL}</div>
<div class='fig'>{G_VC}</div>
<div class='fig'>{G_TO}</div>
<ul>
<li>qd_ 학습 창 도착 최고 {f(PEAK['goal'])} %({f(PEAK['dec'], 2)}M) · 학습 시간 {f(QD_MIN)}분(학습기 기록)</li>
<li>학습 창 값은 그 순간의 표본 정책 · 섞인 장면 값이라 §2 고정 장면 평가와 다름 — 관문은 학습 창, 처방 성공은 고정 장면 평가로 봄(스펙대로)</li>
</ul>

<h2>4. 계측 — logstd · approx_kl(<code>VESSEL_GRAD_TELEMETRY=1</code>, update 마다)</h2>
{blk_table()}
<div class='fig'>{G_SIG}</div>
<div class='fig'>{G_KL}</div>
<ul>
<li>σ = 120° × exp(clamp(logstd, −3.0, 0)) = 정책이 실제로 쓴 침로 잡음(tanh 전). 하한 −3.0(6.0°)에 닿은 update 수: 전문가 0–4 = {' · '.join(str(n) for n in FLOOR)} / 138</li>
<li>전문가 0 = 조우 없음(situation 0). 스펙 §1 근거: qb_ 에서 이 전문가 σ 가 학습 내내 커짐(18.6° → 33.7°, 엔트로피 보너스 0.01) → qd_ 는 보너스 0</li>
<li>LR 3e-4 고정(감쇠 없음) · 2M 스냅샷 직접 열어 확인: entropy_bonus {SNAP['entropy_bonus']} · logstd_min {SNAP['logstd_min']} · logstd_init0 {SNAP['logstd_init0']} · sim.REWARD_RANGE {SNAP['sim.REWARD_RANGE']} · partner_range {SNAP['partner_range']}</li>
</ul>

<h2>5. 순위 관문 v5 — 시드별 규칙</h2>
{rank_table()}

<h2>6. 경과 · 실행 기록</h2>
<ul>
{''.join(f'<li>{ln}</li>' for ln in NOTE)}
{''.join(f'<li><b>{ln}</b></li>' for ln in STOP)}
</ul>

<h2>7. 한계 — 그대로 읽을 것</h2>
<ul>
<li>시드 1개(43) · 평가 시드 1개(999) · 체크포인트당 평가 1회. 같은 체크포인트를 다시 잰 흔들림 크기 모름</li>
<li>네 가지(위험 비용 56 m · 엔트로피 0 · logstd 하한 −3.0 · 초기 −2.0)를 한 번에 바꿈 → 결과가 어느 것 때문인지 모름(스펙 §1)</li>
<li>비교 trunk(n_ · qb_ · qa_)는 보상 반경이 다름 → 같은 보상에서의 비교 아님. 도착 · 충돌 같은 바닥 지표만 나란히 놓음</li>
<li>통신 없는 trunk 하나의 결과임 — 통신 효과 · Fig1 주장 근거 아님</li>
<li>관문 · 판정식 · 스크립트는 안 건드렸음. 명령은 저자가 준 그대로(§6)</li>
</ul>

<h2>8. 다음 — 저자 결정</h2>
<ul>
<li>{NEXT}</li>
<li>결정 전까지 추가 학습 · 평가 안 돌림</li>
</ul>
<div class='small'>자료: runs/2026-10-02_fig1/out/qd/(학습 로그 · ep.csv · _gate.log · _timeline.txt · _rank.log · _rank_v5.json) · out/meaneval_qd/ · out/qd_notes/(runnote.txt · snap_2M.txt · 로그 사본) ·
비교 out/n_trunk_d6_s43.log · diag_trunk_n/ · qa/ · qb/ · meaneval_qa/ · meaneval_qb/ · 생성 report/make_report_qd.py</div>
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
print('trunk 관문', TG, f(GG), '| 순위 v5', RPASS, RN, '| 성공', SUCCESS, (C1, C2, C3), '| 9.04M 평균', A9['goal'], '표본', S9['goal'], '| 4M 평균', A4['goal'])
print(out_pdf)
