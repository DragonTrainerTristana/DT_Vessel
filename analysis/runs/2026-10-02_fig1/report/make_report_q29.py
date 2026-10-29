# 2026-10-29: 병렬 3건 결과 PDF — (a) qd_ phase1(코덱 → 갈래 off · comm → 주 평가 · 곡선 · 표 + 갈래 평균 행동 평가),
#   (b) qe_ trunk(= qd_ + 연속 위험 예고 PBRS c = 410), (c) qd2_ trunk(= qd_ 설정, 시드 44). 스펙 2026-10-28-qd-trunk-design.md §2 · §5(결과 전 고정).
# 숫자는 전부 로그에서 읽음(runs/2026-10-02_fig1/out/):
#   qd/     lr_qd_{off,comm}_s43.txt(주 평가) · qd_{trunk_d6,off,comm}_s43.log(학습 창) · curve_qd_*.txt · _fig1.md · _curve.md · _codec_choice.txt · _branch_check.txt · _timeline.txt
#   qe/ · qd2/   {pre}trunk_d6_s{seed}.log · _gate.log · _timeline.txt · _batch_env.txt
#   meaneval_qd/ · meaneval_qe/ · meaneval_qd2/   sample_/mean_*.txt · logstd_*.txt · _meaneval_*.md
#   q29_notes/   runnote.txt(세션 실행 기록) · snap_2M.txt(체크포인트 스냅샷 직접 연 값)
# 판정식 = summarize_fig1.py · curve_fig1.py · 스펙 §2 와 같은 식 → 스크립트 표(_fig1.md · _curve.md · _meaneval_*.md · _gate.log)와 대조(assert).
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_q29.py [출력 폴더]
import glob, io, math, os, re, subprocess, sys

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.normpath(os.path.join(HERE, '..'))
RUNS = os.path.normpath(os.path.join(F1, '..'))
SC = os.path.join(RUNS, '2026-10-01_scripted')
sys.path.insert(0, os.path.join(RUNS, '2026-09-30_t'))
sys.path.insert(0, SC)
import summarize as S  # noqa: E402

OUT = os.path.join(F1, 'out')
NAME = '2026-10-29_병렬3건_qd갈래_qe_qd2_결과'
BRANCH_AT, TOTAL = 9043968, 16056320
RUN_MAIN = 'burnin=2400 dec=10000 drain=0 ring=1.0 crossing=0 seed=999 envs=256 vessels=16'
RUN_ARGS = 'burnin=2400 dec=3000 drain=3000 ring=1.0 crossing=0 seed=999 envs=256 vessels=16'   # 곡선 · meaneval 같은 인자
GATE_GOAL, GATE_OCOLL, GATE_N = 30.0, 5.0, 4      # 10-07 스펙 §5-3
S1_MEAN9, S2_DROP, S3_GAP, S4_HARM = 65.0, 5.0, 10.0, 5.0   # 10-28 스펙 §2(셋 다) · §5 qe_ 추가 기준(qd_ − 5 pt 이상)
M_FRAC = 0.1
TRK = {'qd_': (43, 'qd', 'meaneval_qd'), 'qd2_': (44, 'qd2', 'meaneval_qd2'), 'qe_': (43, 'qe', 'meaneval_qe')}
XS = ('4', '9.04397')
BX = ('10', '12', '14', '16')


def text(*p):
    with io.open(os.path.join(OUT, *p), encoding='utf-8', errors='replace') as fh:
        return fh.read()


def f(v, nd=1, sign=False):
    if v is None:
        return 'n/a'
    s = f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'
    return s.replace('-', '−')


def mean(xs):
    t, n = 0.0, 0
    for x in xs:
        if x is not None:
            t += x
            n += 1
    return t / n if n else None


wm = lambda rows, k: mean(r[k] for r in rows)  # noqa: E731

# ── 평가 파일(meaneval · 곡선 공통 형식)
LINE = re.compile(r'^(\S+\.pt)\s+\| arm=ON\s+\| goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%\s+oColl=\s*([\d.]+)%\s+TO=\s*([\d.]+)%\s+\|\s+(\d+) eps\b.*?epReward=\s*(-?[\d.]+)', re.M)
PDL = re.compile(r'\[pair-detail\] 통과최소거리 중앙\s+([\d.]+)m \| 첫 변침\(\|.{1,3}\|>0\.2\)까지 중앙\s+([\d.]+)s \| 무변침 조우\s+([\d.]+)%')
NBL = re.compile(r'\[null-baseline\] 조우 없는 배의 \|목표대비 오프셋\| 중앙\s+([\d.]+)deg')


def parse(rel, ckpt, mean_mode):
    t = text(rel)
    m, run, pd, nb = LINE.search(t), re.search(r'\[run\] (.*)', t), PDL.search(t), NBL.search(t)
    assert m and run and run.group(1).strip().startswith(RUN_ARGS), (rel, run and run.group(1))
    assert m.group(1) == ckpt, (rel, m.group(1), ckpt)
    assert ('[eval_mean] student drives with tanh(mean)' in t) == mean_mode, rel
    assert 'reward_range=56.0' in t, rel
    return dict(goal=float(m.group(2)), vColl=float(m.group(3)), oColl=float(m.group(4)), TO=float(m.group(5)), eps=int(m.group(6)),
                epReward=float(m.group(7)), minPass=float(pd.group(1)) if pd else None, turn=float(pd.group(2)) if pd else None,
                nullOff=float(nb.group(1)) if nb else None)


def ck_trunk(pre, x):
    sd = TRK[pre][0]
    return f'{pre}trunk_d6_s{sd}.pt' if x == '9.04397' else f'{pre}trunk_d6_s{sd}.step{x}M.pt'


EV = {}
for pre, (sd, fo, me) in TRK.items():
    for x in XS:
        EV[(pre, x)] = (parse(os.path.join(me, f'sample_{pre}trunk_d6_{x}M.txt'), ck_trunk(pre, x), False),
                        parse(os.path.join(me, f'mean_{pre}trunk_d6_{x}M.txt'), ck_trunk(pre, x), True))
        s, a = EV[(pre, x)]
        row = f"| trunk_d6 | {float(x):g}M | {s['goal']:.1f} / {a['goal']:.1f} ({a['goal'] - s['goal']:+.1f}) | {s['vColl']:.1f} / {a['vColl']:.1f} | {s['epReward']:.0f} / {a['epReward']:.0f} |"
        assert row in text(me, f'_meaneval_{pre}.md'), (pre, row)

LSL = re.compile(r'^(\S+\.pt)\s+steps=(\d+) logstd0=([-\d. ]+?) mean=(-?[\d.]+) sigma_course=([\d.]+)deg', re.M)
SIG = {}
for pre, (sd, fo, me) in TRK.items():
    hit = {m.group(1): float(m.group(5)) for m in LSL.finditer(text(me, f'logstd_{pre}trunk_d6.txt'))}
    for x in XS:
        SIG[(pre, x)] = hit[ck_trunk(pre, x)]

# ── 처방 성공(스펙 §2) · qe_ 추가 기준(§5)
CR = {}
for pre in TRK:
    S4, A4 = EV[(pre, '4')]
    S9, A9 = EV[(pre, '9.04397')]
    c = dict(A4=A4['goal'], A9=A9['goal'], S9=S9['goal'], drop=A4['goal'] - A9['goal'], gap=A9['goal'] - S9['goal'])
    c['C1'], c['C2'], c['C3'] = c['A9'] >= S1_MEAN9, c['drop'] <= S2_DROP, c['gap'] <= S3_GAP
    c['n'] = sum((c['C1'], c['C2'], c['C3']))
    CR[pre] = c
CR['qe_']['C4'] = CR['qe_']['A9'] >= CR['qd_']['A9'] - S4_HARM

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


PPO = re.compile(r'^\[ppo\] upd=(\d+) approx_kl=([\d.]+) clipfrac=([\d.]+) entropy=(-?[\d.]+) logstd0_raw=\[([-\d. ]+)\] lr=([\d.e+-]+) dec=([\d.]+)M', re.M)
sig_eff = lambda raw: 120.0 * math.exp(min(max(raw, -3.0), 0.0))  # noqa: E731

TW, GT, PP, TMIN = {}, {}, {}, {}
for pre, (sd, fo, me) in TRK.items():
    lg = f'{pre}trunk_d6_s{sd}.log'
    TW[pre] = windows(fo, lg)
    assert len(TW[pre]) == 27, (pre, len(TW[pre]))
    TMIN[pre] = saved(fo, lg)
    assert TMIN[pre][0] == 9.04, (pre, TMIN[pre])
    L = TW[pre][-GATE_N:]
    g, v, o = wm(L, 'goal'), wm(L, 'vColl'), wm(L, 'oColl')
    m = re.search(rf'^s{sd} goal=([\d.]+) vColl=([\d.]+) oColl=([\d.]+) 창=(\d+) (통과|불통과)', text(fo, '_gate.log'), re.M)
    assert m and (m.group(1), m.group(2), m.group(3)) == (f'{g:.1f}', f'{v:.1f}', f'{o:.1f}'), (pre, m and m.groups(), g, v, o)
    GT[pre] = dict(L=L, goal=g, vColl=v, oColl=o, TO=wm(L, 'TO'), ok=(g >= GATE_GOAL and o <= GATE_OCOLL))
    assert (m.group(5) == '통과') == GT[pre]['ok']
    lgt = text(fo, lg)
    PP[pre] = [dict(upd=int(m.group(1)), kl=float(m.group(2)), ls=[float(x) for x in m.group(5).split()], lr=float(m.group(6)), dec=float(m.group(7)))
               for m in PPO.finditer(lgt)]
    assert [p['upd'] for p in PP[pre]] == list(range(138)) and all(p['lr'] == 3e-4 for p in PP[pre]), pre
    assert '[resume]' not in lgt and '[bc]' not in lgt, pre

BENV = {pre: text(fo, '_batch_env.txt').strip() for pre, (sd, fo, me) in TRK.items()}
EXP = 'ARPA=56 RANK=v5 REWARD_RANGE=56 EXTRA=[VESSEL_ENTROPY_BONUS=0 VESSEL_LOGSTD_MIN=-3.0 VESSEL_LOGSTD_INIT0=-2.0 VESSEL_GRAD_TELEMETRY=1]'
assert BENV['qd_'] == EXP and BENV['qd2_'] == EXP and BENV['qe_'] == EXP + ' PREVIEW=410', BENV

# 스냅샷(세션이 체크포인트 직접 연 값)
SNAPT = text('q29_notes', 'snap_2M.txt')
SNAP = {}
for blk in re.split(r'^== ', SNAPT, flags=re.M)[1:]:
    nm = blk.split(' ', 1)[0]
    SNAP[nm] = dict(re.findall(r'^\s+cfg_snapshot\.(\S+) = (.*)$', blk, re.M))
q, d2 = SNAP['qe_trunk_d6_s43.step2M.pt'], SNAP['qd2_trunk_d6_s44.step2M.pt']
assert q['sim.PREVIEW_PBRS_COEF'] == '410.0' and 'sim.PREVIEW_PBRS_COEF' not in d2 and d2['seed'] == '44', (q, d2)
for s_ in (q, d2, SNAP['qd_off_s43.pt'], SNAP['qd_comm_s43.pt']):
    assert (s_['entropy_bonus'], s_['logstd_min'], s_['logstd_init0'], s_['sim.REWARD_RANGE'], s_['action_mode']) == ('0.0', '-3.0', '-2.0', '56.0', 'course'), s_
assert (SNAP['qd_off_s43.pt']['comm_fields'], SNAP['qd_off_s43.pt']['partner_range']) == ('state', '56.0')
assert (SNAP['qd_comm_s43.pt']['comm_fields'], SNAP['qd_comm_s43.pt']['partner_range']) == ('intent', '300.0')

# ══ (a) qd_ 갈래 off · comm ══
D = 'qd'
PRE, SEED = 'qd_', 43
MAIN = [('vColl', '배끼리 충돌 %', -1, 12.1), ('goal', '도착 %', +1, 0.0), ('dcpa', 'DCPA(통과거리) m', +1, 1.5),
        ('w_score', 'COLREGs(Woerner) %', +1, 0.0), ('fuel', '연료(도착 ep)', -1, 0.0), ('fleet_fuel', '도착 1회당 함대 연료', -1, 0.0)]
POOL_KEYS = ('goal', 'vColl', 'dcpa')
AUX = [('head', '방향 바꾼 총량 °', -1, 0.0), ('rud_trav', '타 이동량 °/ep', -1, 0.0), ('rud_abs', '평균 타각 크기 °', -1, 0.0),
       ('C', '옛 COLREGs %', +1, 0.0), ('TO', '시간초과 %', -1, 0.0), ('oColl', '벽 충돌 %', -1, 0.0), ('epReward', 'epReward', +1, 0.0)]
ND = lambda k: 0 if k in ('fuel', 'fleet_fuel', 'head', 'epReward') else 1  # noqa: E731


def load_main(arm):
    fp = os.path.join(OUT, D, f'lr_{PRE}{arm}_s{SEED}.txt')
    txt = text(D, f'lr_{PRE}{arm}_s{SEED}.txt')
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
    if runs and g and sum(g) / len(g) >= 50.0:
        POOL[nm] = {k: mean([r.get(k) for r in runs]) for k in POOL_KEYS}


def judge(k, sg, floor):   # summarize_fig1.py 와 같은 식(시드 1개 · offb 없음 → N = 고정 하한)
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
FIG1 = text(D, '_fig1.md')
m = re.search(r'주 지표 통과: (\d+)/(\d+)', FIG1)
assert m and (int(m.group(1)), int(m.group(2))) == (WIN_MAIN, len(MAIN)), (m and m.groups(), WIN_MAIN)
assert f'같은 trunk 짝 1개: [{SEED}]' in FIG1 and '저자 결정' in FIG1

TR = TW['qd_']
BR = {a: windows(D, f'{PRE}{a}_s{SEED}.log') for a in ('off', 'comm')}
SV = {a: saved(D, f'{PRE}{a}_s{SEED}.log') for a in ('off', 'comm')}
assert all(len(BR[a]) == 21 and SV[a][0] == 16.06 for a in BR), ({a: len(BR[a]) for a in BR}, SV)
for a in BR:
    lg = text(D, f'{PRE}{a}_s{SEED}.log')
    assert f'[resume] {PRE}trunk_d6_s{SEED}.pt' in lg and f'branch_at={BRANCH_AT}' in lg and '[bc]' not in lg, a
SHA = set(re.findall(r'\[branch\] trunk=\S+ sha256=(\w+)', text(D, f'{PRE}off_s{SEED}.log') + text(D, f'{PRE}comm_s{SEED}.log')))
BCHK = text(D, '_branch_check.txt')
assert len(SHA) == 1 and 'BRANCH CHECK: ALL PASS' in BCHK and list(SHA)[0] in BCHK
LAST = {a: BR[a][-4:] for a in BR}
B3 = {a: [BR[a][0:7], BR[a][7:14], BR[a][14:21]] for a in BR}     # 창 7개씩 3구간(결과 보고 고르지 않으려고 3등분)
N_ABOVE = sum(1 for i in range(21) if BR['comm'][i]['goal'] > BR['off'][i]['goal'])

# 고정 장면 곡선(표본 행동)
EVL = re.compile(r'goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%\s+oColl=\s*([\d.]+)%\s+TO=\s*([\d.]+)%\s+\|\s+(\d+) eps')
PTS = {}
for fp in glob.glob(os.path.join(OUT, D, f'curve_{PRE}*_*M.txt')):
    mm = re.match(rf'curve_{PRE}(.+)_([0-9.]+)M\.txt$', os.path.basename(fp))
    txt = io.open(fp, encoding='utf-8', errors='replace').read()
    e, g = re.search(r'epReward=\s*(-?[0-9.]+)', txt), EVL.search(txt)
    assert mm and e and g and RUN_ARGS in txt and 'reward_range=56.0' in txt, fp
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


CURVE_MD = text(D, '_curve.md')
MC = {}
for a in SER:
    c = [v['epReward'] for _, v in SER[a]]
    drop, ok = m_check(c)
    MC[a] = (c[0], max(c), c[-1], drop, ok)
    row = f"| {a} | {SEED} | {len(c)} | {c[0]:.1f} | {max(c):.1f} | {c[-1]:.1f} | {drop:.1f} | {'우상향' if ok else '★불통과'} |"
    assert row in CURVE_MD, (row, CURVE_MD)
BRPT = {a: sorted((x, v) for (r, x), v in PTS.items() if r == f'{a}_s{SEED}') for a in SER}       # 10 · 12 · 14 · 16 · 16.06M
TRPT = sorted((x, v) for (r, x), v in PTS.items() if r == f'trunk_d6_s{SEED}')                    # 2 · 4 · 6 · 8 · 9.04M
assert len(TRPT) == 5 and all(len(BRPT[a]) == 5 for a in BRPT)
for a in BRPT:   # step16M = 끝 모델(같은 파일 내용) → 두 점 값 같아야 함
    assert BRPT[a][-2][1] == BRPT[a][-1][1], (a, BRPT[a][-2:])
DIST = {a: BRPT[a][:-2] + BRPT[a][-1:] for a in BRPT}     # 서로 다른 모델 4개(10 · 12 · 14 · 16.06M)
C_ABOVE = sum(1 for (x, vo), (_, vc) in zip(DIST['off'], DIST['comm']) if vc['goal'] > vo['goal'])
TR_END = TRPT[-1][1]

# 갈래 평균 행동(저자 명령 5) — 표본 = 같은 인자 곡선 파일
BM = {}
for a in ('off', 'comm'):
    for x in BX:
        mv = parse(os.path.join('meaneval_qd', f'mean_{PRE}{a}_{x}M.txt'), f'{PRE}{a}_s{SEED}.step{x}M.pt', True)
        sv = dict(BRPT[a])[float(x)]
        BM[(a, x)] = (sv, mv)
        row = f"| {a} | {x}M | {sv['goal']:.1f} / {mv['goal']:.1f} ({mv['goal'] - sv['goal']:+.1f}) | {sv['vColl']:.1f} / {mv['vColl']:.1f} | {sv['epReward']:.0f} / {mv['epReward']:.0f} |"
        assert row in text('meaneval_qd', '_meaneval_qd_.md'), row
BM_ABOVE = sum(1 for x in BX if BM[('comm', x)][1]['goal'] > BM[('off', x)][1]['goal'])
BM_K = {a: mean(BM[(a, x)][1]['goal'] for x in BX) for a in ('off', 'comm')}
BM_KS = {a: mean(BM[(a, x)][0]['goal'] for x in BX) for a in ('off', 'comm')}

# 코덱
CODEC = text(D, '_codec_choice.txt')
CH_K = int(re.search(r'CHOSEN_K=(\d+)', CODEC).group(1))
CPASS = dict((int(k), v) for k, v in re.findall(r'k(\d+)=(통과|불통과)', CODEC))
assert CPASS[CH_K] == '통과'

# 경과
TLQ = [ln for ln in text(D, '_timeline.txt').splitlines() if ln.startswith('2026-10-29')]
assert '배치 qd_ phase1' in TLQ[0] and 'adcb7f2' in TLQ[0] and any('phase1 완료' in ln for ln in TLQ) and not any('★' in ln for ln in TLQ), TLQ[:2]


def t_of(tl, pat):
    hit = [ln for ln in tl if ln[20:].startswith(pat)]
    return hit[-1][11:19] if hit else 'n/a'


TLE = [ln for ln in text('qe', '_timeline.txt').splitlines() if ln.startswith('2026-10-29')]
TLD2 = [ln for ln in text('qd2', '_timeline.txt').splitlines() if ln.startswith('2026-10-29')]
assert 'fa772a3' in TLE[0] and 'adcb7f2' in TLD2[0] and not any('★' in ln for ln in TLE + TLD2), (TLE[0], TLD2[0])
OVR = text(D, '_rank_override.txt')
assert '저자 결정' in OVR and 'FAIL' not in OVR
NOTE = []   # (본문, [하위 항목]) — runnote.txt 순서 그대로
for ln in text('q29_notes', 'runnote.txt').splitlines()[1:]:
    if ln.startswith('- '):
        NOTE.append((ln[2:], []))
    elif ln.startswith('  - '):
        NOTE[-1][1].append(ln[4:])


# ── 그림(SVG, 로그 값 그대로)
def chart(series, x0, x1, xticks, y0, y1, yticks, w=700, h=150, hlines=(), vline=None, ynd=0, xfmt=lambda d: f'{d:g}M'):
    ml, mr, mt, mb = 36, 8, 18, 18
    X = lambda d: ml + (w - ml - mr) * (d - x0) / (x1 - x0)  # noqa: E731
    Y = lambda v: mt + (h - mt - mb) * (y1 - min(max(v, y0), y1)) / (y1 - y0)  # noqa: E731
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='100%'>"]
    for t in yticks:
        s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' stroke='#e4e3df' stroke-width='0.6'/>")
        s.append(f"<text x='{ml - 4}' y='{Y(t) + 2.6:.1f}' font-size='7.2' text-anchor='end' fill='#52514e'>{f(t, ynd)}</text>")
    for d in xticks:
        s.append(f"<text x='{X(d):.1f}' y='{h - 5}' font-size='7.2' text-anchor='middle' fill='#52514e'>{xfmt(d)}</text>")
    s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{h - mb}' y2='{h - mb}' stroke='#0b0b0b' stroke-width='0.7'/>")
    if vline is not None:
        s.append(f"<line x1='{X(vline):.1f}' x2='{X(vline):.1f}' y1='{mt}' y2='{h - mb}' stroke='#52514e' stroke-width='0.7' stroke-dasharray='3 2'/>")
        s.append(f"<text x='{X(vline) + 3:.1f}' y='{mt + 7}' font-size='7.2' fill='#52514e'>분기 9.04M</text>")
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


C_TR, C_OFF, C_COMM = '#9c9a94', '#d9822b', '#1b6fd6'
C_QD, C_QD2, C_QE = '#a8322a', '#3f8f5a', '#7a4fb5'
wp = lambda rows, k: [(r['dec'], r[k]) for r in rows]  # noqa: E731
join = lambda rows, k: [(TR[-1]['dec'], TR[-1][k])] + wp(rows, k)  # noqa: E731
X16 = dict(x0=0, x1=16.4, xticks=range(0, 17, 2), vline=BRANCH_AT / 1e6)
G_WIN = chart([('trunk 도착', wp(TR, 'goal'), C_TR, 1.4, '', False), ('off 도착', join(BR['off'], 'goal'), C_OFF, 1.6, '', False),
               ('comm 도착', join(BR['comm'], 'goal'), C_COMM, 1.6, '', False)], y0=0, y1=100, yticks=[0, 20, 40, 60, 80, 100], **X16)
G_WINV = chart([('trunk 배 충돌', wp(TR, 'vColl'), C_TR, 1.4, '', False), ('off 배 충돌', join(BR['off'], 'vColl'), C_OFF, 1.6, '', False),
                ('comm 배 충돌', join(BR['comm'], 'vColl'), C_COMM, 1.6, '', False)], y0=0, y1=100, yticks=[0, 20, 40, 60, 80, 100], h=128, **X16)
cp = lambda pts, k: [(x, v[k]) for x, v in pts]  # noqa: E731
A9QD = EV[('qd_', '9.04397')][1]
G_CGO = chart([('trunk 표본', cp(TRPT, 'goal'), C_TR, 1.2, '3 2', True), ('off 표본', cp(TRPT[-1:] + DIST['off'], 'goal'), C_OFF, 1.2, '3 2', True),
               ('comm 표본', cp(TRPT[-1:] + DIST['comm'], 'goal'), C_COMM, 1.2, '3 2', True),
               ('off 평균', [(9.04, A9QD['goal'])] + [(float(x), BM[('off', x)][1]['goal']) for x in BX], C_OFF, 1.7, '', True),
               ('comm 평균', [(9.04, A9QD['goal'])] + [(float(x), BM[('comm', x)][1]['goal']) for x in BX], C_COMM, 1.7, '', True)],
              y0=0, y1=100, yticks=[0, 20, 40, 60, 80, 100], h=165, **X16)
EPS = [v['epReward'] for v in PTS.values()]
E0, E1 = (min(EPS) // 200) * 200, (max(EPS) // 200 + 1) * 200
G_CEP = chart([('trunk epReward', cp(TRPT, 'epReward'), C_TR, 1.4, '', True), ('off epReward', cp(TRPT[-1:] + DIST['off'], 'epReward'), C_OFF, 1.6, '', True),
               ('comm epReward', cp(TRPT[-1:] + DIST['comm'], 'epReward'), C_COMM, 1.6, '', True)],
              y0=E0, y1=E1, yticks=[E0 + i * (E1 - E0) / 4 for i in range(5)], h=140, **X16)
STY = (('qd_', C_QD, 1.8, ''), ('qd2_', C_QD2, 1.4, '4 2'), ('qe_', C_QE, 1.4, ''))
LAB = {'qd_': 'qd_ (시드 43)', 'qd2_': 'qd2_ (시드 44)', 'qe_': 'qe_ (예고 PBRS)'}
X9 = dict(x0=0, x1=9.2, xticks=range(0, 10))
G_TG = chart([(f'{LAB[p]} 도착', wp(TW[p], 'goal'), col, sw, da, False) for p, col, sw, da in STY],
             y0=0, y1=100, yticks=[0, 20, 40, 60, 80, 100], hlines=((GATE_GOAL, '관문 도착 30 %'),), **X9)
G_TV = chart([(f'{LAB[p]} 배 충돌', wp(TW[p], 'vColl'), col, sw, da, False) for p, col, sw, da in STY],
             y0=0, y1=100, yticks=[0, 20, 40, 60, 80, 100], h=128, **X9)
G_TS = chart([(f'{LAB[p]} σ 침로(전문가 평균)', [(q_['dec'], mean(sig_eff(v) for v in q_['ls'])) for q_ in PP[p]], col, sw, da, False) for p, col, sw, da in STY],
             y0=0, y1=30, yticks=[0, 10, 20, 30], h=118, **X9)
EVX = [4.06, 9.04]
G_TE = chart([(f'{p} 평균', [(xx, EV[(p, x)][1]['goal']) for xx, x in zip(EVX, XS)], col, 1.7, '', True) for p, col, _, _ in STY]
             + [(f'{p} 표본', [(xx, EV[(p, x)][0]['goal']) for xx, x in zip(EVX, XS)], col, 1.2, '3 2', True) for p, col, _, _ in STY],
             x0=3.3, x1=9.8, xticks=EVX, y0=0, y1=100, yticks=[0, 20, 40, 60, 80, 100], h=160,
             hlines=((S1_MEAN9, '성공 1: 9.04M 평균 ≥ 65 %'),), xfmt=lambda d: f'{d:.2f}M')

ok = lambda b: '충족' if b else '<b>미달</b>'  # noqa: E731


def verdict_table():
    h = "<table><thead><tr><th>건</th><th>항목</th><th>기준(결과 전 고정)</th><th>값</th><th>판정</th></tr></thead><tbody>"
    for pre in ('qd2_', 'qe_'):
        c, gt = CR[pre], GT[pre]
        rows = [('trunk 관문(10-07 §5-3)', f'마지막 4창 도착 ≥ 30 % · 벽 ≤ 5 %', f"도착 {f(gt['goal'])} % · 벽 {f(gt['oColl'])} %", '통과' if gt['ok'] else '<b>불통과</b>'),
                ('성공 1', '9.04M 평균 행동 도착 ≥ 65 %', f"{f(c['A9'])} %", ok(c['C1'])),
                ('성공 2', '4M → 9.04M 평균 행동 하락 ≤ 5 pt', f"{f(c['A4'])} → {f(c['A9'])} %(하락 {f(c['drop'], 1, True)} pt{' = 오름' if c['drop'] < 0 else ''})", ok(c['C2'])),
                ('성공 3', '9.04M 평균 − 표본 ≤ 10 pt', f"{f(c['A9'])} − {f(c['S9'])} = {f(c['gap'], 1, True)} pt", ok(c['C3']))]
        if pre == 'qe_':
            rows.append(('qd_ 를 해치지 않음(§5)', f"9.04M 평균 행동 도착 ≥ qd_ − 5 pt = {f(CR['qd_']['A9'] - S4_HARM)} %", f"{f(c['A9'])} %(qd_ {f(CR['qd_']['A9'])})", ok(c['C4'])))
        tot = c['n'] + (int(c['C4']) if pre == 'qe_' else 0)
        rows.append(('합계', '전부 충족 = 성공', f"{tot}/{len(rows) - 1}", '<b>성공</b>' if tot == len(rows) - 1 else '<b>미달</b>'))
        for i, r in enumerate(rows):
            h += f"<tr>{f'<td rowspan={len(rows)}><b>{LAB[pre]}</b></td>' if i == 0 else ''}<td>{r[0]}</td><td>{r[1]}</td><td class='c'>{r[2]}</td><td class='c'>{r[3]}</td></tr>"
    h += (f"<tr><td><b>qd_ 갈래</b></td><td>Fig1 판정(summarize_fig1.py)</td><td>같은 시드 짝 전부 승 + |차| &gt; N(시드 1개 · offb 없음 → N = 고정 하한)</td>"
          f"<td class='c'>주 지표 통과 {WIN_MAIN}/{len(MAIN)} · 방향 {DIR_MAIN}/{len(MAIN)}</td><td class='c'>{'통과' if WIN_MAIN == len(MAIN) else '<b>미달</b>'}</td></tr>")
    h += (f"<tr><td></td><td>곡선 M(보고만)</td><td>최대 하락 ≤ 0.1 × (최고 − 첫 값)</td><td class='c'>off {'우상향' if MC['off'][4] else '불통과'} · comm {'우상향' if MC['comm'][4] else '불통과'}</td><td class='c'>보고만</td></tr>")
    return h + "</tbody></table>"


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


def bm_table():
    h = ("<table><thead><tr><th>갈래 체크포인트</th><th>off 도착 % 표본 / 평균</th><th>comm 도착 % 표본 / 평균</th><th>평균 comm − off</th>"
         "<th>off 배 충돌 % 평균</th><th>comm 배 충돌 % 평균</th><th>off · comm 시간초과 % 평균</th></tr></thead><tbody>")
    a9s = EV[('qd_', '9.04397')][0]
    h += (f"<tr class='ref'><td class='c'>trunk 9.04M(분기점)</td><td class='c' colspan='2'>{f(a9s['goal'])} / {f(A9QD['goal'])}</td><td class='c'>—</td>"
          f"<td class='c' colspan='2'>{f(A9QD['vColl'])}</td><td class='c'>{f(A9QD['TO'])}</td></tr>")
    for x in BX:
        so, mo = BM[('off', x)]
        sc, mc = BM[('comm', x)]
        d = mc['goal'] - mo['goal']
        h += (f"<tr><td class='c'>{x}M</td><td class='c'>{f(so['goal'])} / <b>{f(mo['goal'])}</b></td><td class='c'>{f(sc['goal'])} / <b>{f(mc['goal'])}</b></td>"
              f"<td class='c {'good' if d > 0 else 'bad'}'>{f(d, 1, True)}</td><td class='c'>{f(mo['vColl'])}</td><td class='c'>{f(mc['vColl'])}</td><td class='c'>{f(mo['TO'])} · {f(mc['TO'])}</td></tr>")
    h += (f"<tr class='g'><td class='c'>4점 평균</td><td class='c'>{f(BM_KS['off'])} / {f(BM_K['off'])}</td><td class='c'>{f(BM_KS['comm'])} / {f(BM_K['comm'])}</td>"
          f"<td class='c'>{f(BM_K['comm'] - BM_K['off'], 1, True)}</td><td class='c' colspan='3'>평균 행동 comm &gt; off: {BM_ABOVE}/4 체크포인트</td></tr>")
    return h + "</tbody></table>"


def ev_table():
    h = ("<table><thead><tr><th>trunk</th><th>체크포인트</th><th>σ 침로 °</th><th>도착 % 표본</th><th>도착 % 평균</th><th>평균 − 표본</th>"
         "<th>배 충돌 % 표본 / 평균</th><th>벽 · 시간초과 % 평균</th><th>첫 변침 s 표본 / 평균</th><th>무조우 오프셋 ° 표본 / 평균</th><th>에피소드 표본 / 평균</th></tr></thead><tbody>")
    for pre in ('qd_', 'qd2_', 'qe_'):
        for x in XS:
            s, a = EV[(pre, x)]
            cls = " class='ref'" if pre == 'qd_' else ''
            h += (f"<tr{cls}><td>{LAB[pre]}</td><td class='c'>{float(x):.2f}M</td><td class='c'>{f(SIG[(pre, x)])}</td><td class='c'>{f(s['goal'])}</td>"
                  f"<td class='c'><b>{f(a['goal'])}</b></td><td class='c'>{f(a['goal'] - s['goal'], 1, True)}</td><td class='c'>{f(s['vColl'])} / {f(a['vColl'])}</td>"
                  f"<td class='c'>{f(a['oColl'])} · {f(a['TO'])}</td><td class='c'>{f(s['turn'])} / {f(a['turn'])}</td><td class='c'>{f(s['nullOff'])} / {f(a['nullOff'])}</td>"
                  f"<td class='c'>{s['eps']} / {a['eps']}</td></tr>")
    return h + "</tbody></table>"


def gate_table():
    h = ("<table><thead><tr><th>trunk</th><th>qd_ 와 다른 것</th><th>도착 %</th><th>배 충돌 %</th><th>벽 %</th><th>시간초과 %</th><th>4창 도착 값</th><th>학습 분</th><th>§5-3</th></tr></thead><tbody>")
    desc = {'qd_': '— (10-28 기록)', 'qd2_': '시드 44', 'qe_': '연속 위험 예고 PBRS c = 410'}
    for pre in ('qd_', 'qd2_', 'qe_'):
        g = GT[pre]
        cls = " class='ref'" if pre == 'qd_' else ''
        h += (f"<tr{cls}><td>{LAB[pre]}</td><td>{desc[pre]}</td><td class='c'><b>{f(g['goal'])}</b></td><td class='c'>{f(g['vColl'])}</td><td class='c'>{f(g['oColl'])}</td>"
              f"<td class='c'>{f(g['TO'])}</td><td class='c'>{' · '.join(f(w['goal']) for w in g['L'])}</td><td class='c'>{f(TMIN[pre][1])}</td><td class='c'>{'통과' if g['ok'] else '불통과'}</td></tr>")
    return h + "</tbody></table>"


def blk_table():
    h = ("<table><thead><tr><th>구간(갈래 창 7개씩)</th><th>off 도착 %</th><th>comm 도착 %</th><th>off 배 충돌 %</th><th>comm 배 충돌 %</th>"
         "<th>off 결정당 R</th><th>comm 결정당 R</th></tr></thead><tbody>")
    TL4 = TR[-4:]
    h += (f"<tr class='ref'><td class='c'>trunk 끝 4창</td><td class='c' colspan='2'>{f(wm(TL4, 'goal'))}</td><td class='c' colspan='2'>{f(wm(TL4, 'vColl'))}</td>"
          f"<td class='c' colspan='2'>{f(wm(TL4, 'R'), 3)}</td></tr>")
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
        h += (f"<tr class='ref'><td class='c'>{x:g}M</td><td class='c'>trunk</td><td class='c'>{f(v['goal'])}</td><td class='c'>{f(v['vColl'])}</td><td class='c'>{f(v['oColl'])}</td>"
              f"<td class='c'>{f(v['TO'])}</td><td class='c'>{f(v['epReward'], 0)}</td><td class='c'>{v['eps']:.0f}</td></tr>")
    for (x, vo), (_, vc) in zip(DIST['off'], DIST['comm']):
        for nm, v in (('off', vo), ('comm', vc)):
            h += (f"<tr><td class='c'>{x:g}M</td><td class='c'>{nm}</td><td class='c'>{f(v['goal'])}</td><td class='c'>{f(v['vColl'])}</td><td class='c'>{f(v['oColl'])}</td>"
                  f"<td class='c'>{f(v['TO'])}</td><td class='c'>{f(v['epReward'], 0)}</td><td class='c'>{v['eps']:.0f}</td></tr>")
    return h + "</tbody></table>"


QD2_OK = GT['qd2_']['ok'] and CR['qd2_']['n'] == 3
QE_OK = GT['qe_']['ok'] and CR['qe_']['n'] == 3 and CR['qe_']['C4']
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
tr.ref td { color: #52514e; background: #f7f6f2; }
code { font-family: Consolas, monospace; font-size: 8pt; background: #f4f3ef; padding: 0 2px; }
.small { color: #52514e; font-size: 7.9pt; }
.fig { margin: 4px 0 0 0; }
.pb { page-break-before: always; }
"""
HTML = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{NAME}</title><style>{CSS}</style></head><body>
<h1>10-29 병렬 3건 — qd_ 갈래 off · comm · qe_ 예고 PBRS · qd2_ 시드 44</h1>
<div class='sub1'>2026-10-29 · LICS 서버 · 스펙 2026-10-28-qd-trunk-design.md §2 · §5(결과 전 고정) · 저자 10-29 결정: 순위 관문 v5 불통과를 넘기고 성공 경로 병렬 ·
코드 qd_ · qd2_ = feat/fig1-latent adcb7f2, qe_ = feat/early-pbrs fa772a3(18c45d8 + 검사 파일 1줄) · 흉내 없음 순수 PPO · 평가 시드 999 · 시드 1–2개 = 단독 주장 아님</div>

<div class='box'><b>한 줄 결론</b> —
<b>qd_ 갈래</b>: 주 평가(끝 모델, 표본 행동) Fig1 주 지표 통과 <b>{WIN_MAIN}/{len(MAIN)}</b>, comm 이 앞선 지표 {DIR_MAIN}/{len(MAIN)} · 도착 off {f(off['goal'])} % · comm {f(comm['goal'])} % · 배 충돌 {f(off['vColl'])} · {f(comm['vColl'])} %.
평균 행동 도착(10–16M 4점): off {f(BM_K['off'])} % · comm {f(BM_K['comm'])} %, comm 위 {BM_ABOVE}/4(차 {' · '.join(f(BM[('comm', x)][1]['goal'] - BM[('off', x)][1]['goal'], 1, True) for x in BX)} pt). 곡선 M off {'우상향' if MC['off'][4] else '불통과'} · comm {'우상향' if MC['comm'][4] else '불통과'}.
<b>qd2_(시드 44)</b>: 관문 {'통과' if GT['qd2_']['ok'] else '불통과'}({f(GT['qd2_']['goal'])} %) · 처방 성공 {CR['qd2_']['n']}/3(9.04M 평균 {f(CR['qd2_']['A9'])} % · 하락 {f(CR['qd2_']['drop'], 1, True)} · 격차 {f(CR['qd2_']['gap'], 1, True)}) → {'<b>qd_ 결과 재현</b>' if QD2_OK else '<b>재현 미달</b>'}.
<b>qe_(예고 PBRS)</b>: 관문 {'통과' if GT['qe_']['ok'] else '불통과'}({f(GT['qe_']['goal'])} %) · 처방 성공 {CR['qe_']['n']}/3 · qd_ 대비 {ok(CR['qe_']['C4'])}(9.04M 평균 {f(CR['qe_']['A9'])} % vs qd_ {f(CR['qd_']['A9'])}) → {'<b>성공</b>' if QE_OK else '<b>미달</b>'}.
다음은 저자 결정.</div>

<h2>1. 판정 — 결과 전 고정 기준</h2>
{verdict_table()}
<ul>
<li>평균 행동 = tanh(mean)(<code>eval_mean.py</code>), 표본 = 학습 정책 그대로. 평가 인자 · 장면 · 평가 시드 같음(256 env · 3000 결정 · burn-in 2400 · drain 3000)</li>
<li>Fig1 판정 · 곡선 M 은 스크립트 표(<code>_fig1.md</code> · <code>_curve.md</code>)와 같은 식으로 다시 계산해 대조함. 순위 관문 v5 는 저자 override(재실행 없음, <code>_rank_override.txt</code>)</li>
</ul>

<h2>2. qd_ 갈래 — 주 평가(끝 모델 16.06M, 표본 행동, 256 env · 10000 결정 · drain 0)</h2>
{judge_table(MAIN)}
{judge_table(AUX)}
<ul>
<li>최선 OFF 풀(도착 ≥ 50 % 인 OFF): {' · '.join(sorted(POOL)) if POOL else '없음'} · 보조 지표 comm 이 앞선 것 {DIR_AUX}/{sum(1 for k, *_ in AUX if J[k])}</li>
<li>코덱 p46 k = <b>{CH_K}</b>({' · '.join(f'k{k} {v}' for k, v in sorted(CPASS.items()))}) · 분기 검사 ALL PASS(trunk sha256 {list(SHA)[0]})</li>
</ul>

<h2 class='pb'>3. qd_ 갈래 — 체크포인트별 평균 행동 · 표본(저자 명령 5 + 고정 장면 곡선)</h2>
{bm_table()}
<div class='fig'>{G_CGO}</div>
<div class='fig'>{G_CEP}</div>
{curve_table()}
<ul>
<li>표본 값 = 곡선 평가 파일(같은 인자). 16M 체크포인트 = 끝 모델(16.06M)과 같은 값 → 서로 다른 갈래 모델 4개(10 · 12 · 14 · 16.06M). 표본 도착 comm &gt; off = {C_ABOVE}/4</li>
<li>곡선 M(epReward, 보고만): off 첫 {f(MC['off'][0], 0)} · 최고 {f(MC['off'][1], 0)} · 끝 {f(MC['off'][2], 0)} · 최대 하락 {f(MC['off'][3], 0)} → {'우상향' if MC['off'][4] else '불통과'} /
comm 첫 {f(MC['comm'][0], 0)} · 최고 {f(MC['comm'][1], 0)} · 끝 {f(MC['comm'][2], 0)} · 최대 하락 {f(MC['comm'][3], 0)} → {'우상향' if MC['comm'][4] else '불통과'}</li>
</ul>

<h2>4. qd_ 갈래 — 학습 중 창 값(표본 행동, 로그 한 줄 = update 5개)</h2>
<div class='fig'>{G_WIN}</div>
<div class='fig'>{G_WINV}</div>
{blk_table()}
<ul><li>갈래 21창 가운데 comm 도착 &gt; off = {N_ABOVE}창 · 갈래 학습 off {f(SV['off'][1])}분 · comm {f(SV['comm'][1])}분(off 는 앞 약 25분 GPU 2 를 qd2_ 와 나눠 써서 느림, §6)</li></ul>

<h2 class='pb'>5. trunk 3개 — qd_(10-28) · qd2_(시드 44) · qe_(예고 PBRS)</h2>
{gate_table()}
{ev_table()}
<div class='fig'>{G_TE}</div>
<div class='fig'>{G_TG}</div>
<div class='fig'>{G_TV}</div>
<div class='fig'>{G_TS}</div>
<ul>
<li>σ 침로 = 120° × exp(clamp(logstd 행동 0번, −3.0, 0)), 전문가 5개 평균(학습기 <code>[ppo]</code> 줄). 표의 σ 는 <code>logstd_dump.py</code> sigma_course(체크포인트)</li>
<li>qe_ 의 epReward 는 예고 보상(c = 410)이 더해져 qd_ · qd2_ 와 정의가 다름 → 안 비교함. 도착 · 충돌 · 시간초과만 나란히 놓음</li>
<li>스냅샷 직접 확인(<code>q29_notes/snap_2M.txt</code>): qe_ sim.PREVIEW_PBRS_COEF {q['sim.PREVIEW_PBRS_COEF']} · qd2_ seed {d2['seed']} · 넷 다 entropy_bonus 0 · logstd_min −3.0 · logstd_init0 −2.0 · REWARD_RANGE 56 · course 모드</li>
</ul>

<h2>6. 경과 · 실행 기록(세션)</h2>
<ul>
{''.join(f"<li>{ln}{('<ul>' + ''.join(f'<li>{c}</li>' for c in sub) + '</ul>') if sub else ''}</li>" for ln, sub in NOTE)}
</ul>

<h2>7. 한계 — 그대로 읽을 것</h2>
<ul>
<li>갈래 비교는 시드 1개(43) · offb 없음 → 잡음 자가 없음. N = 고정 하한뿐</li>
<li>qd2_ 는 qd_ 재현 확인 1개(시드 44). 두 시드 = 단독 주장 못 함</li>
<li>qe_ 는 qd_ 에 예고 보상 하나를 더했지만 시드 43 하나 · 학습 경로가 갈라짐 → qd_ 와의 차가 보상 탓인지 시드 흔들림인지 모름</li>
<li>평가는 체크포인트당 1회 · 평가 시드 1개(999). 같은 체크포인트 재측정 흔들림 크기 모름</li>
<li>16:23 GPU 드라이버 리셋: 그때 돌던 평가는 재부팅 뒤 같은 명령으로 다시 잼. 곡선 comm 10M 점(GPU 3)만 리셋 뒤 정상 종료해 그대로 씀 — 리셋 영향은 확인 못 함(§6)</li>
<li>관문 · 판정식 · 평가 스크립트는 안 건드렸음. 바뀐 것 = qe_ 검사 파일 1줄(인코딩) · qd2_ GPU 번호(2 → 3) · 갈래 표본 짝 파일을 같은 스크립트의 곡선 재사용 경로로 채움(새 평가 없음)</li>
</ul>

<h2>8. 다음 — 저자 결정</h2>
<ul>
<li>스펙 §4 성공 경로에서 남은 것: 하나씩 뺀 trunk · 시드 ×3. 이 결과로 무엇을 할지는 저자 결정</li>
<li>결정 전까지 추가 학습 · 평가 안 돌림</li>
</ul>
<div class='small'>자료: runs/2026-10-02_fig1/out/qd/ · qe/ · qd2/ · meaneval_qd/ · meaneval_qe/ · meaneval_qd2/ · q29_notes/ · 생성 report/make_report_q29.py ·
앞 보고 2026-10-28_qd_trunk_위험56m_떨림수정_결과.pdf</div>
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
print('qd_ 갈래 주 지표', WIN_MAIN, '/', len(MAIN), '방향', DIR_MAIN, '| 평균 행동 comm>off', BM_ABOVE, '/4', f(BM_K['off']), f(BM_K['comm']),
      '| M', MC['off'][4], MC['comm'][4], '| qd2_', GT['qd2_']['ok'], CR['qd2_']['n'], '| qe_', GT['qe_']['ok'], CR['qe_']['n'], CR['qe_']['C4'])
print(out_pdf)
