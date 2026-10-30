# 2026-10-30: qd2_ 갈래(시드 44) + qd_ 통신 메시지 절제 결과 PDF — 저자 명령 블록 '10-30'.
#   (1) qd2_ phase1(코덱 → 갈래 off · comm → 주 평가 · 곡선 · 표) + 갈래 평균 행동 평가(10 · 12 · 14 · 16M)
#   (2) qd_comm 14 · 16M 메시지 절제 4조건(far56 · shuffle · decl0 · state, 평균 행동, 학습 없음, 코드 feat/fig1-latent 5e678dc)
#   스펙 2026-10-28-qd-trunk-design.md §5 · §6(읽는 법 결과 전 고정).
# 숫자는 전부 로그에서 읽음(runs/2026-10-02_fig1/out/):
#   qd2/  lr_qd2_{off,comm}_s44.txt(주 평가) · qd2_{trunk_d6,off,comm}_s44.log · curve_qd2_*.txt · _fig1.md · _curve.md · _codec_choice.txt · _branch_check.txt · _timeline.txt
#   qd/   같은 것(10-29 시드 43, 두 시드 나란히 놓기용)
#   meaneval_qd2/ · meaneval_qd/   sample_/mean_*.txt · _meaneval_*.md
#   ablate_qd/   abl_<조건>_mean_qd_comm_<x>M.txt · _ablate_qd_.md
#   q30_notes/   runnote.txt(세션 실행 기록) · snap_branch.txt(체크포인트 스냅샷 직접 연 값)
# 판정식 = summarize_fig1.py · curve_fig1.py · ablate_comm.sh 표와 같은 식 → 스크립트 표와 대조(assert).
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_q30.py [출력 폴더]
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

OUT = os.path.join(F1, 'out')
NAME = '2026-10-30_qd2갈래_시드44_메시지절제_결과'
BRANCH_AT = 9043968
RUN_MAIN = 'burnin=2400 dec=10000 drain=0 ring=1.0 crossing=0 seed=999 envs=256 vessels=16'
RUN_ARGS = 'burnin=2400 dec=3000 drain=3000 ring=1.0 crossing=0 seed=999 envs=256 vessels=16'   # 곡선 · meaneval · 절제 같은 인자
M_FRAC = 0.1
BX = ('10', '12', '14', '16')
ABL_X = ('14', '16')
ABL_C = ('decl0', 'state', 'shuffle', 'far56')
ABL_FLAG = {'far56': 'far_drop=56m', 'shuffle': 'field_shuffle=True', 'decl0': 'decl_zero=[13:18]', 'state': "comm_groups=['state']"}
ABL_LAB = {'far56': '56 m 밖 파트너 통째로 뺌', 'shuffle': '코덱 필드 섞음(위치 유지)', 'decl0': '역할 선언 칸 0', 'state': '의도 · 역할 그룹 끔(상태만)'}
OFF_BAND, DECL_HALF = 3.0, 0.5    # 스펙 §6 읽는 법(결과 전 고정)


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

# ── 평가 파일(meaneval · 곡선 · 절제 공통 형식)
LINE = re.compile(r'^(\S+\.pt)\s+\| arm=ON\s+\| goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%\s+oColl=\s*([\d.]+)%\s+TO=\s*([\d.]+)%\s+\|\s+(\d+) eps\b.*?epReward=\s*(-?[\d.]+)', re.M)


def parse(rel, ckpt, mean_mode, abl=None):
    t = text(rel)
    m, run = LINE.search(t), re.search(r'\[run\] (.*)', t)
    assert m and run and run.group(1).strip().startswith(RUN_ARGS), (rel, run and run.group(1))
    assert m.group(1) == ckpt, (rel, m.group(1), ckpt)
    assert ('[eval_mean] student drives with tanh(mean)' in t) == mean_mode, rel
    assert 'reward_range=56.0' in t, rel
    ab = re.search(r'\[eval\] 절제: (.*)', t)
    assert (ab is not None) == (abl is not None), rel
    if abl:
        assert ABL_FLAG[abl] in ab.group(1), (rel, ab.group(1))
    return dict(goal=float(m.group(2)), vColl=float(m.group(3)), oColl=float(m.group(4)), TO=float(m.group(5)), eps=int(m.group(6)),
                epReward=float(m.group(7)))


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


MAIN = [('vColl', '배끼리 충돌 %', -1, 12.1), ('goal', '도착 %', +1, 0.0), ('dcpa', 'DCPA(통과거리) m', +1, 1.5),
        ('w_score', 'COLREGs(Woerner) %', +1, 0.0), ('fuel', '연료(도착 ep)', -1, 0.0), ('fleet_fuel', '도착 1회당 함대 연료', -1, 0.0)]
POOL_KEYS = ('goal', 'vColl', 'dcpa')
AUX = [('head', '방향 바꾼 총량 °', -1, 0.0), ('rud_trav', '타 이동량 °/ep', -1, 0.0), ('rud_abs', '평균 타각 크기 °', -1, 0.0),
       ('C', '옛 COLREGs %', +1, 0.0), ('TO', '시간초과 %', -1, 0.0), ('oColl', '벽 충돌 %', -1, 0.0), ('epReward', 'epReward', +1, 0.0)]
ND = lambda k: 0 if k in ('fuel', 'fleet_fuel', 'head', 'epReward') else 1  # noqa: E731
T_OFF = []
for fp in sorted(glob.glob(os.path.join(SC, 'out', 'lr_t_off_s*.txt'))):
    r = S.parse_all(fp)
    if 'goal' in r:
        T_OFF.append(r)


def m_check(c):   # curve_fig1.py 와 같은 식
    run_max, drop = c[0], 0.0
    for v in c:
        run_max = max(run_max, v)
        drop = max(drop, run_max - v)
    rise = max(c) - c[0]
    return drop, rise > 0 and drop <= M_FRAC * rise


EVL = re.compile(r'goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%\s+oColl=\s*([\d.]+)%\s+TO=\s*([\d.]+)%\s+\|\s+(\d+) eps')


def batch(D, PRE, SEED, ME, day):
    """한 배치(갈래 off · comm)를 로그에서 읽고 배치 스크립트 표와 대조."""
    B = dict(D=D, PRE=PRE, SEED=SEED)

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
    POOL = {}
    for nm, runs in (('off', [off]), ('t_off', T_OFF)):
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
        note = ''
        if k in POOL_KEYS and POOL:
            who, best = max(((nm, pm[k]) for nm, pm in POOL.items()), key=lambda t: sg * t[1])
            pok = sg * (y - best) > 0
            ok = ok and pok
            note = f"풀 최선 {who} {f(best)} {'넘음' if pok else '못 넘음'}"
        return dict(x=x, y=y, w=w, d=y - x, N=N, ok=ok, note=note)

    J = {k: judge(k, sg, fl) for k, _, sg, fl in MAIN + AUX}
    B.update(off=off, comm=comm, POOL=POOL, J=J)
    B['WIN_MAIN'] = sum(1 for k, *_ in MAIN if J[k]['ok'])
    B['DIR_MAIN'] = sum(J[k]['w'] for k, *_ in MAIN)
    B['DIR_AUX'] = sum(J[k]['w'] for k, *_ in AUX if J[k])
    B['N_AUX'] = sum(1 for k, *_ in AUX if J[k])
    FIG1 = text(D, '_fig1.md')
    m = re.search(r'주 지표 통과: (\d+)/(\d+)', FIG1)
    assert m and (int(m.group(1)), int(m.group(2))) == (B['WIN_MAIN'], len(MAIN)), (D, m and m.groups(), B['WIN_MAIN'])
    assert f'같은 trunk 짝 1개: [{SEED}]' in FIG1 and '저자 결정' in FIG1

    TR = windows(D, f'{PRE}trunk_d6_s{SEED}.log')
    assert len(TR) == 27, (D, len(TR))
    BR = {a: windows(D, f'{PRE}{a}_s{SEED}.log') for a in ('off', 'comm')}
    SV = {a: saved(D, f'{PRE}{a}_s{SEED}.log') for a in ('off', 'comm')}
    assert all(len(BR[a]) == 21 and SV[a][0] == 16.06 for a in BR), (D, {a: len(BR[a]) for a in BR}, SV)
    for a in BR:
        lg = text(D, f'{PRE}{a}_s{SEED}.log')
        assert f'[resume] {PRE}trunk_d6_s{SEED}.pt' in lg and f'branch_at={BRANCH_AT}' in lg and '[bc]' not in lg, (D, a)
    SHA = set(re.findall(r'\[branch\] trunk=\S+ sha256=(\w+)', text(D, f'{PRE}off_s{SEED}.log') + text(D, f'{PRE}comm_s{SEED}.log')))
    assert len(SHA) == 1 and 'BRANCH CHECK: ALL PASS' in text(D, '_branch_check.txt') and list(SHA)[0] in text(D, '_branch_check.txt')
    B.update(TR=TR, BR=BR, SV=SV, SHA=list(SHA)[0], LAST={a: BR[a][-4:] for a in BR},
             B3={a: [BR[a][0:7], BR[a][7:14], BR[a][14:21]] for a in BR},     # 창 7개씩 3구간(결과 보고 고르지 않으려고 3등분)
             N_ABOVE=sum(1 for i in range(21) if BR['comm'][i]['goal'] > BR['off'][i]['goal']))

    # 고정 장면 곡선(표본 행동)
    PTS = {}
    for fp in glob.glob(os.path.join(OUT, D, f'curve_{PRE}*_*M.txt')):
        mm = re.match(rf'curve_{re.escape(PRE)}(.+)_([0-9.]+)M\.txt$', os.path.basename(fp))
        txt = io.open(fp, encoding='utf-8', errors='replace').read()
        e, g = re.search(r'epReward=\s*(-?[0-9.]+)', txt), EVL.search(txt)
        assert mm and e and g and RUN_ARGS in txt and 'reward_range=56.0' in txt, fp
        PTS[(mm.group(1), float(mm.group(2)))] = dict(zip(('goal', 'vColl', 'oColl', 'TO', 'eps'), (float(x) for x in g.groups())), epReward=float(e.group(1)))
    SER = {a: sorted((x, v) for (r, x), v in PTS.items() if r in (f'trunk_d6_s{SEED}', f'{a}_s{SEED}')) for a in ('off', 'comm')}
    assert all(len(SER[a]) == 10 for a in SER) and len(PTS) == 15, (D, len(PTS), {a: len(SER[a]) for a in SER})
    CURVE_MD = text(D, '_curve.md')
    MC = {}
    for a in SER:
        c = [v['epReward'] for _, v in SER[a]]
        drop, ok = m_check(c)
        MC[a] = (c[0], max(c), c[-1], drop, ok)
        row = f"| {a} | {SEED} | {len(c)} | {c[0]:.1f} | {max(c):.1f} | {c[-1]:.1f} | {drop:.1f} | {'우상향' if ok else '★불통과'} |"
        assert row in CURVE_MD, (D, row)
    BRPT = {a: sorted((x, v) for (r, x), v in PTS.items() if r == f'{a}_s{SEED}') for a in SER}       # 10 · 12 · 14 · 16 · 16.06M
    TRPT = sorted((x, v) for (r, x), v in PTS.items() if r == f'trunk_d6_s{SEED}')                    # 2 · 4 · 6 · 8 · 9.04M
    assert len(TRPT) == 5 and all(len(BRPT[a]) == 5 for a in BRPT)
    for a in BRPT:   # step16M = 끝 모델(같은 파일 내용) → 두 점 값 같아야 함
        assert BRPT[a][-2][1] == BRPT[a][-1][1], (D, a, BRPT[a][-2:])
    DIST = {a: BRPT[a][:-2] + BRPT[a][-1:] for a in BRPT}     # 서로 다른 모델 4개(10 · 12 · 14 · 16.06M)
    B.update(PTS=PTS, MC=MC, TRPT=TRPT, DIST=DIST, BRPT=BRPT,
             C_ABOVE=sum(1 for (x, vo), (_, vc) in zip(DIST['off'], DIST['comm']) if vc['goal'] > vo['goal']))

    # trunk 9.04M 평균 행동(분기점) · 갈래 평균 행동 — 표본 = 같은 인자 곡선 파일
    B['A9'] = parse(os.path.join(ME, f'mean_{PRE}trunk_d6_9.04397M.txt'), f'{PRE}trunk_d6_s{SEED}.pt', True)
    B['S9'] = parse(os.path.join(ME, f'sample_{PRE}trunk_d6_9.04397M.txt'), f'{PRE}trunk_d6_s{SEED}.pt', False)
    BM = {}
    for a in ('off', 'comm'):
        for x in BX:
            mv = parse(os.path.join(ME, f'mean_{PRE}{a}_{x}M.txt'), f'{PRE}{a}_s{SEED}.step{x}M.pt', True)
            sv = dict(BRPT[a])[float(x)]
            BM[(a, x)] = (sv, mv)
            row = f"| {a} | {x}M | {sv['goal']:.1f} / {mv['goal']:.1f} ({mv['goal'] - sv['goal']:+.1f}) | {sv['vColl']:.1f} / {mv['vColl']:.1f} | {sv['epReward']:.0f} / {mv['epReward']:.0f} |"
            assert row in text(ME, f'_meaneval_{PRE}.md'), (ME, row)
    B.update(BM=BM, BM_ABOVE=sum(1 for x in BX if BM[('comm', x)][1]['goal'] > BM[('off', x)][1]['goal']),
             BM_K={a: mean(BM[(a, x)][1]['goal'] for x in BX) for a in ('off', 'comm')},
             BM_KS={a: mean(BM[(a, x)][0]['goal'] for x in BX) for a in ('off', 'comm')})

    CODEC = text(D, '_codec_choice.txt')
    B['CH_K'] = int(re.search(r'CHOSEN_K=(\d+)', CODEC).group(1))
    B['CPASS'] = dict((int(k), v) for k, v in re.findall(r'k(\d+)=(통과|불통과)', CODEC))
    assert B['CPASS'][B['CH_K']] == '통과' and f'CODEC_s{SEED}=' in CODEC
    TL = [ln for ln in text(D, '_timeline.txt').splitlines() if ln.startswith(day)]
    assert f'배치 {PRE} phase1' in TL[0] and 'adcb7f2' in TL[0] and any('phase1 완료' in ln for ln in TL) and not any('★' in ln for ln in TL), (D, TL[:2])
    assert '저자 결정' in text(D, '_rank_override.txt') and 'FAIL' not in text(D, '_rank_override.txt')
    B['TL'] = TL
    return B


Q2 = batch('qd2', 'qd2_', 44, 'meaneval_qd2', '2026-10-30')
Q1 = batch('qd', 'qd_', 43, 'meaneval_qd', '2026-10-29')
assert text('qd2', '_batch_env.txt').strip() == text('qd', '_batch_env.txt').strip() == \
    'ARPA=56 RANK=v5 REWARD_RANGE=56 EXTRA=[VESSEL_ENTROPY_BONUS=0 VESSEL_LOGSTD_MIN=-3.0 VESSEL_LOGSTD_INIT0=-2.0 VESSEL_GRAD_TELEMETRY=1]'

# 스냅샷(세션이 체크포인트 직접 연 값)
SNAP = {}
for blk in re.split(r'^== ', text('q30_notes', 'snap_branch.txt'), flags=re.M)[1:]:
    SNAP[blk.split(' ', 1)[0]] = dict(re.findall(r'^\s+cfg_snapshot\.(\S+) = (.*)$', blk, re.M))
for nm, fl, pr in (('qd2_off_s44.pt', 'state', '56.0'), ('qd2_comm_s44.pt', 'intent', '300.0')):
    s_ = SNAP[nm]
    assert (s_['seed'], s_['entropy_bonus'], s_['logstd_min'], s_['logstd_init0'], s_['sim.REWARD_RANGE'], s_['action_mode'], s_['comm_fields'], s_['partner_range']) == \
        ('44', '0.0', '-3.0', '-2.0', '56.0', 'course', fl, pr), (nm, s_)

# ══ (2) 메시지 절제 — qd_comm 14 · 16M, 평균 행동 ══
ABL = {}
AMD = text('ablate_qd', '_ablate_qd_.md')
for x in ABL_X:
    for c in ABL_C:
        v = parse(os.path.join('ablate_qd', f'abl_{c}_mean_qd_comm_{x}M.txt'), f'qd_comm_s43.step{x}M.pt', True, abl=c)
        ABL[(x, c)] = v
        base, offv = Q1['BM'][('comm', x)][1], Q1['BM'][('off', x)][1]
        row = f"| {x}M | mean | {c} | {v['goal']:.1f} | {v['vColl']:.1f} | {v['epReward']:.0f} | {base['goal'] - v['goal']:+.1f} | {v['goal'] - offv['goal']:+.1f} |"
        assert row in AMD, row
assert '5e678dc' in text('ablate_qd', '_log.txt')
RD = {}
for x in ABL_X:
    base, offv = Q1['BM'][('comm', x)][1]['goal'], Q1['BM'][('off', x)][1]['goal']
    far_d = base - ABL[(x, 'far56')]['goal']
    RD[x] = dict(base=base, off=offv, gain=base - offv, far_off=ABL[(x, 'far56')]['goal'] - offv, far_d=far_d,
                 decl_d=base - ABL[(x, 'decl0')]['goal'])
    RD[x]['R1'] = abs(RD[x]['far_off']) <= OFF_BAND
    RD[x]['R2'] = RD[x]['decl_d'] >= DECL_HALF * far_d
NOTE = []   # (본문, [하위 항목]) — runnote.txt 순서 그대로
for ln in text('q30_notes', 'runnote.txt').splitlines()[1:]:
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


def bars(groups, y0, y1, yticks, w=700, h=170):
    """groups = [(그룹 이름, [(막대 이름, 값, 색), ...]), ...] — 도착 % 막대, 값 그대로."""
    ml, mr, mt, mb = 36, 8, 10, 30
    Y = lambda v: mt + (h - mt - mb) * (y1 - min(max(v, y0), y1)) / (y1 - y0)  # noqa: E731
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='100%'>"]
    for t in yticks:
        s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' stroke='#e4e3df' stroke-width='0.6'/>")
        s.append(f"<text x='{ml - 4}' y='{Y(t) + 2.6:.1f}' font-size='7.2' text-anchor='end' fill='#52514e'>{f(t, 0)}</text>")
    gw = (w - ml - mr) / len(groups)
    for gi, (gname, bs) in enumerate(groups):
        bw = (gw - 24) / len(bs)
        for bi, (bn, v, col) in enumerate(bs):
            x = ml + gi * gw + 12 + bi * bw
            s.append(f"<rect x='{x + 2:.1f}' y='{Y(v):.1f}' width='{bw - 4:.1f}' height='{Y(y0) - Y(v):.1f}' fill='{col}'/>")
            s.append(f"<text x='{x + bw / 2:.1f}' y='{Y(v) - 2.5:.1f}' font-size='7.2' text-anchor='middle' fill='#0b0b0b'>{f(v)}</text>")
            s.append(f"<text x='{x + bw / 2:.1f}' y='{h - mb + 9:.1f}' font-size='7' text-anchor='middle' fill='#52514e'>{bn}</text>")
        s.append(f"<text x='{ml + gi * gw + gw / 2:.1f}' y='{h - 5}' font-size='7.6' text-anchor='middle' fill='#0b0b0b'>{gname}</text>")
    s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{Y(y0):.1f}' y2='{Y(y0):.1f}' stroke='#0b0b0b' stroke-width='0.7'/>")
    return ''.join(s) + '</svg>'


C_TR, C_OFF, C_COMM, C_ABL = '#9c9a94', '#d9822b', '#1b6fd6', '#7aa7e0'
wp = lambda rows, k: [(r['dec'], r[k]) for r in rows]  # noqa: E731
cp = lambda pts, k: [(x, v[k]) for x, v in pts]  # noqa: E731
X16 = dict(x0=0, x1=16.4, xticks=range(0, 17, 2), vline=BRANCH_AT / 1e6)
Y100 = dict(y0=0, y1=100, yticks=[0, 20, 40, 60, 80, 100])


def join(B, a, k):
    return [(B['TR'][-1]['dec'], B['TR'][-1][k])] + wp(B['BR'][a], k)


G_WIN = chart([('trunk 도착', wp(Q2['TR'], 'goal'), C_TR, 1.4, '', False), ('off 도착', join(Q2, 'off', 'goal'), C_OFF, 1.6, '', False),
               ('comm 도착', join(Q2, 'comm', 'goal'), C_COMM, 1.6, '', False)], **Y100, **X16)
G_WINV = chart([('trunk 배 충돌', wp(Q2['TR'], 'vColl'), C_TR, 1.4, '', False), ('off 배 충돌', join(Q2, 'off', 'vColl'), C_OFF, 1.6, '', False),
                ('comm 배 충돌', join(Q2, 'comm', 'vColl'), C_COMM, 1.6, '', False)], h=128, **Y100, **X16)


def g_cgo(B):
    tr, dist = B['TRPT'], B['DIST']
    return chart([('trunk 표본', cp(tr, 'goal'), C_TR, 1.2, '3 2', True), ('off 표본', cp(tr[-1:] + dist['off'], 'goal'), C_OFF, 1.2, '3 2', True),
                  ('comm 표본', cp(tr[-1:] + dist['comm'], 'goal'), C_COMM, 1.2, '3 2', True),
                  ('off 평균', [(9.04, B['A9']['goal'])] + [(float(x), B['BM'][('off', x)][1]['goal']) for x in BX], C_OFF, 1.7, '', True),
                  ('comm 평균', [(9.04, B['A9']['goal'])] + [(float(x), B['BM'][('comm', x)][1]['goal']) for x in BX], C_COMM, 1.7, '', True)],
                 h=165, **Y100, **X16)


EPS = [v['epReward'] for v in Q2['PTS'].values()]
E0, E1 = (min(EPS) // 200) * 200, (max(EPS) // 200 + 1) * 200
G_CEP = chart([('trunk epReward', cp(Q2['TRPT'], 'epReward'), C_TR, 1.4, '', True),
               ('off epReward', cp(Q2['TRPT'][-1:] + Q2['DIST']['off'], 'epReward'), C_OFF, 1.6, '', True),
               ('comm epReward', cp(Q2['TRPT'][-1:] + Q2['DIST']['comm'], 'epReward'), C_COMM, 1.6, '', True)],
              y0=E0, y1=E1, yticks=[E0 + i * (E1 - E0) / 4 for i in range(5)], h=140, **X16)
G_SEED = chart([(f'{lab} {a} 평균', [(9.04, B['A9']['goal'])] + [(float(x), B['BM'][(a, x)][1]['goal']) for x in BX], col, sw, da, True)
                for lab, B, sw, da in (('s43', Q1, 1.2, '3 2'), ('s44', Q2, 1.7, '')) for a, col in (('off', C_OFF), ('comm', C_COMM))],
               x0=8.6, x1=16.4, xticks=[9.04, 10, 12, 14, 16], y0=50, y1=90, yticks=[50, 60, 70, 80, 90], h=150, xfmt=lambda d: f'{d:g}M')
G_ABL = bars([(f'qd_comm {x}M (시드 43) — 평균 행동 도착 %',
               [('기본 통신', RD[x]['base'], C_COMM)] + [(c, ABL[(x, c)]['goal'], C_ABL) for c in ABL_C] + [('OFF 갈래', RD[x]['off'], C_OFF)])
              for x in ABL_X], y0=50, y1=85, yticks=[50, 60, 70, 80])

ok = lambda b: '충족' if b else '<b>미달</b>'  # noqa: E731


def verdict_table():
    h = "<table><thead><tr><th>건</th><th>항목</th><th>기준(결과 전 고정)</th><th>값</th><th>판정</th></tr></thead><tbody>"
    B = Q2
    h += (f"<tr><td rowspan='3'><b>qd2_ 갈래(시드 44)</b></td><td>Fig1 판정(summarize_fig1.py)</td><td>같은 시드 짝 전부 승 + |차| &gt; N(시드 1개 · offb 없음 → N = 고정 하한)</td>"
          f"<td class='c'>주 지표 통과 {B['WIN_MAIN']}/{len(MAIN)} · 방향 {B['DIR_MAIN']}/{len(MAIN)}</td><td class='c'>{'통과' if B['WIN_MAIN'] == len(MAIN) else '<b>미달</b>'}</td></tr>")
    h += (f"<tr><td>곡선 M(보고만)</td><td>최대 하락 ≤ 0.1 × (최고 − 첫 값)</td><td class='c'>off {'우상향' if B['MC']['off'][4] else '불통과'} · comm {'우상향' if B['MC']['comm'][4] else '불통과'}</td><td class='c'>보고만</td></tr>")
    h += (f"<tr><td>갈래 평균 행동(기록용)</td><td>— (판정 기준 없음)</td><td class='c'>comm &gt; off {B['BM_ABOVE']}/4 체크포인트 · 4점 평균 off {f(B['BM_K']['off'])} · comm {f(B['BM_K']['comm'])} %</td><td class='c'>기록용</td></tr>")
    for i, x in enumerate(ABL_X):
        r = RD[x]
        h += (f"<tr>{f'<td rowspan={2 * len(ABL_X)}><b>qd_comm 메시지 절제</b><br>(시드 43, 평균 행동)</td>' if i == 0 else ''}"
              f"<td>{x}M 읽는 법 ①</td><td>far56 이 OFF 갈래 같은 체크포인트 ± {OFF_BAND:g} pt 안 → '이득 = 56 m 밖 통신'</td>"
              f"<td class='c'>far56 {f(ABL[(x, 'far56')]['goal'])} − OFF {f(r['off'])} = {f(r['far_off'], 1, True)} pt</td><td class='c'>{'성립' if r['R1'] else '<b>불성립</b>'}</td></tr>"
              f"<tr><td>{x}M 읽는 법 ②</td><td>decl0 하락 ≥ far56 하락 × ½ → 역할 선언 출처 고쳐야 함(저자에게 보고)</td>"
              f"<td class='c'>decl0 하락 {f(r['decl_d'], 1, True)} vs far56 하락 ½ = {f(DECL_HALF * r['far_d'])}</td><td class='c'>{'<b>성립</b>' if r['R2'] else '불성립'}</td></tr>")
    return h + "</tbody></table>"


def judge_table(B, rows):
    h = ("<table><thead><tr><th>지표</th><th>좋은 방향</th><th>off</th><th>comm</th><th>comm − off</th><th>comm 승/짝</th><th>N</th><th>판정</th></tr></thead><tbody>")
    for k, lab, sg, _ in rows:
        j = B['J'][k]
        if not j:
            h += f"<tr><td>{lab}</td><td colspan='7' class='c'>n/a</td></tr>"
            continue
        cls = 'good' if j['w'] else 'bad'
        h += (f"<tr><td>{lab}</td><td class='c'>{'↑' if sg > 0 else '↓'}</td><td class='c'>{f(j['x'], ND(k))}</td><td class='c'>{f(j['y'], ND(k))}</td>"
              f"<td class='c {cls}'>{f(j['d'], 2 if ND(k) else 0, True)}</td><td class='c {cls}'>{j['w']}/1</td><td class='c'>{f(j['N'], 2)}</td>"
              f"<td class='c'>{'<b>통과</b>' if j['ok'] else '불통과'}{(' · ' + j['note']) if j['note'] else ''}</td></tr>")
    return h + "</tbody></table>"


def seed_table():
    h = ("<table><thead><tr><th>주 지표(끝 모델, 표본)</th><th>좋은 방향</th><th>시드 43 off / comm</th><th>시드 43 차</th><th>시드 44 off / comm</th><th>시드 44 차</th>"
         "<th>comm 승 시드 수</th><th>N</th><th>두 시드 모두 |차| &gt; N</th></tr></thead><tbody>")
    for k, lab, sg, fl in MAIN + AUX:
        a, b = Q1['J'][k], Q2['J'][k]
        if not a or not b:
            continue
        nd, dd = ND(k), (2 if ND(k) else 0)
        both = a['w'] and b['w'] and abs(a['d']) > fl and abs(b['d']) > fl
        cls = lambda j: 'good' if j['w'] else 'bad'  # noqa: E731
        ref = '' if (k, lab, sg, fl) in MAIN else " class='ref'"
        h += (f"<tr{ref}><td>{lab}</td><td class='c'>{'↑' if sg > 0 else '↓'}</td><td class='c'>{f(a['x'], nd)} / {f(a['y'], nd)}</td><td class='c {cls(a)}'>{f(a['d'], dd, True)}</td>"
              f"<td class='c'>{f(b['x'], nd)} / {f(b['y'], nd)}</td><td class='c {cls(b)}'>{f(b['d'], dd, True)}</td><td class='c'>{a['w'] + b['w']}/2</td>"
              f"<td class='c'>{f(fl, 1)}</td><td class='c'>{'예' if both else '아니오'}</td></tr>")
    h += "<tr><td colspan='9' class='small'>회색 행 = 보조 지표. 판정은 배치마다(시드 1개씩) 위 §1 · §2 — 이 표는 두 시드를 나란히 놓은 기록용(시드별 승패 수)</td></tr>"
    return h + "</tbody></table>"


def bm_table(B):
    h = ("<table><thead><tr><th>갈래 체크포인트</th><th>off 도착 % 표본 / 평균</th><th>comm 도착 % 표본 / 평균</th><th>평균 comm − off</th>"
         "<th>off 배 충돌 % 평균</th><th>comm 배 충돌 % 평균</th><th>off · comm 시간초과 % 평균</th></tr></thead><tbody>")
    h += (f"<tr class='ref'><td class='c'>trunk 9.04M(분기점)</td><td class='c' colspan='2'>{f(B['S9']['goal'])} / {f(B['A9']['goal'])}</td><td class='c'>—</td>"
          f"<td class='c' colspan='2'>{f(B['A9']['vColl'])}</td><td class='c'>{f(B['A9']['TO'])}</td></tr>")
    for x in BX:
        so, mo = B['BM'][('off', x)]
        sc, mc = B['BM'][('comm', x)]
        d = mc['goal'] - mo['goal']
        h += (f"<tr><td class='c'>{x}M</td><td class='c'>{f(so['goal'])} / <b>{f(mo['goal'])}</b></td><td class='c'>{f(sc['goal'])} / <b>{f(mc['goal'])}</b></td>"
              f"<td class='c {'good' if d > 0 else 'bad'}'>{f(d, 1, True)}</td><td class='c'>{f(mo['vColl'])}</td><td class='c'>{f(mc['vColl'])}</td><td class='c'>{f(mo['TO'])} · {f(mc['TO'])}</td></tr>")
    h += (f"<tr class='g'><td class='c'>4점 평균</td><td class='c'>{f(B['BM_KS']['off'])} / {f(B['BM_K']['off'])}</td><td class='c'>{f(B['BM_KS']['comm'])} / {f(B['BM_K']['comm'])}</td>"
          f"<td class='c'>{f(B['BM_K']['comm'] - B['BM_K']['off'], 1, True)}</td><td class='c' colspan='3'>평균 행동 comm &gt; off: {B['BM_ABOVE']}/4 체크포인트</td></tr>")
    return h + "</tbody></table>"


def blk_table(B):
    h = ("<table><thead><tr><th>구간(갈래 창 7개씩)</th><th>off 도착 %</th><th>comm 도착 %</th><th>off 배 충돌 %</th><th>comm 배 충돌 %</th>"
         "<th>off 결정당 R</th><th>comm 결정당 R</th></tr></thead><tbody>")
    TL4 = B['TR'][-4:]
    h += (f"<tr class='ref'><td class='c'>trunk 끝 4창</td><td class='c' colspan='2'>{f(wm(TL4, 'goal'))}</td><td class='c' colspan='2'>{f(wm(TL4, 'vColl'))}</td>"
          f"<td class='c' colspan='2'>{f(wm(TL4, 'R'), 3)}</td></tr>")
    for i in range(3):
        bo, bc = B['B3']['off'][i], B['B3']['comm'][i]
        h += (f"<tr><td class='c'>{bo[0]['dec'] - 0.33:.2f}–{bo[-1]['dec']:.2f}M</td><td class='c'>{f(wm(bo, 'goal'))}</td><td class='c'>{f(wm(bc, 'goal'))}</td>"
              f"<td class='c'>{f(wm(bo, 'vColl'))}</td><td class='c'>{f(wm(bc, 'vColl'))}</td><td class='c'>{f(wm(bo, 'R'), 3)}</td><td class='c'>{f(wm(bc, 'R'), 3)}</td></tr>")
    L = B['LAST']
    h += (f"<tr class='g'><td class='c'>갈래 끝 4창</td><td class='c'>{f(wm(L['off'], 'goal'))}</td><td class='c'>{f(wm(L['comm'], 'goal'))}</td>"
          f"<td class='c'>{f(wm(L['off'], 'vColl'))}</td><td class='c'>{f(wm(L['comm'], 'vColl'))}</td><td class='c'>{f(wm(L['off'], 'R'), 3)}</td>"
          f"<td class='c'>{f(wm(L['comm'], 'R'), 3)}</td></tr>")
    return h + "</tbody></table>"


def curve_table(B):
    h = ("<table><thead><tr><th>체크포인트</th><th>런</th><th>도착 %</th><th>배 충돌 %</th><th>벽 충돌 %</th><th>시간초과 %</th><th>epReward</th><th>에피소드 수</th></tr></thead><tbody>")
    for x, v in B['TRPT']:
        h += (f"<tr class='ref'><td class='c'>{x:g}M</td><td class='c'>trunk</td><td class='c'>{f(v['goal'])}</td><td class='c'>{f(v['vColl'])}</td><td class='c'>{f(v['oColl'])}</td>"
              f"<td class='c'>{f(v['TO'])}</td><td class='c'>{f(v['epReward'], 0)}</td><td class='c'>{v['eps']:.0f}</td></tr>")
    for (x, vo), (_, vc) in zip(B['DIST']['off'], B['DIST']['comm']):
        for nm, v in (('off', vo), ('comm', vc)):
            h += (f"<tr><td class='c'>{x:g}M</td><td class='c'>{nm}</td><td class='c'>{f(v['goal'])}</td><td class='c'>{f(v['vColl'])}</td><td class='c'>{f(v['oColl'])}</td>"
                  f"<td class='c'>{f(v['TO'])}</td><td class='c'>{f(v['epReward'], 0)}</td><td class='c'>{v['eps']:.0f}</td></tr>")
    return h + "</tbody></table>"


def abl_table():
    h = ("<table><thead><tr><th>체크포인트</th><th>조건</th><th>무엇을 끊나</th><th>도착 %</th><th>배 충돌 %</th><th>시간초과 %</th><th>epReward</th>"
         "<th>도착: 기본 통신 − 조건</th><th>도착: 조건 − OFF</th><th>통신 이득 중 남은 몫</th></tr></thead><tbody>")
    for x in ABL_X:
        r = RD[x]
        bv, ov = Q1['BM'][('comm', x)][1], Q1['BM'][('off', x)][1]
        h += (f"<tr class='g'><td class='c'>{x}M</td><td class='c'>기본 통신</td><td>—</td><td class='c'>{f(bv['goal'])}</td><td class='c'>{f(bv['vColl'])}</td><td class='c'>{f(bv['TO'])}</td>"
              f"<td class='c'>{f(bv['epReward'], 0)}</td><td class='c'>0</td><td class='c'>{f(r['gain'], 1, True)}</td><td class='c'>100 %</td></tr>")
        for c in ABL_C:
            v = ABL[(x, c)]
            left = (v['goal'] - r['off']) / r['gain'] * 100
            h += (f"<tr><td class='c'>{x}M</td><td class='c'>{c}</td><td>{ABL_LAB[c]}</td><td class='c'><b>{f(v['goal'])}</b></td><td class='c'>{f(v['vColl'])}</td><td class='c'>{f(v['TO'])}</td>"
                  f"<td class='c'>{f(v['epReward'], 0)}</td><td class='c'>{f(r['base'] - v['goal'], 1, True)}</td><td class='c'>{f(v['goal'] - r['off'], 1, True)}</td><td class='c'>{f(left, 0)} %</td></tr>")
        h += (f"<tr class='ref'><td class='c'>{x}M</td><td class='c'>OFF 갈래</td><td>(다른 모델 — 같은 체크포인트 OFF)</td><td class='c'>{f(ov['goal'])}</td><td class='c'>{f(ov['vColl'])}</td>"
              f"<td class='c'>{f(ov['TO'])}</td><td class='c'>{f(ov['epReward'], 0)}</td><td class='c'>—</td><td class='c'>0</td><td class='c'>0 %</td></tr>")
    return h + "</tbody></table>"


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
B = Q2
HTML = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{NAME}</title><style>{CSS}</style></head><body>
<h1>10-30 qd2_ 갈래 off · comm(시드 44) · qd_ 통신 메시지 절제</h1>
<div class='sub1'>2026-10-30 · LICS 서버 · 스펙 2026-10-28-qd-trunk-design.md §5 · §6(읽는 법 결과 전 고정) · 저자 10-30 결정('ㅇㅋ 명령어줘요') ·
코드 qd2_ = feat/fig1-latent adcb7f2(pull 안 함), 절제 = 5e678dc(평가 전용 --far_drop 추가, 별도 작업 폴더) · 흉내 없음 순수 PPO · 평가 시드 999 · 시드 1–2개 = 단독 주장 아님</div>

<div class='box'><b>한 줄 결론</b> —
<b>qd2_ 갈래(시드 44)</b>: 주 평가(끝 모델, 표본) Fig1 주 지표 통과 <b>{B['WIN_MAIN']}/{len(MAIN)}</b>, comm 이 앞선 지표 {B['DIR_MAIN']}/{len(MAIN)} ·
도착 off {f(B['off']['goal'])} · comm {f(B['comm']['goal'])} % · 배 충돌 {f(B['off']['vColl'])} · {f(B['comm']['vColl'])} %.
곡선 M off {'우상향' if B['MC']['off'][4] else '불통과'} · comm {'우상향' if B['MC']['comm'][4] else '불통과'}.
<b>갈래 평균 행동은 시드 43 과 다름</b>: 도착(10–16M 4점) off {f(B['BM_K']['off'])} · comm {f(B['BM_K']['comm'])} %, comm 위 {B['BM_ABOVE']}/4(차 {' · '.join(f(B['BM'][('comm', x)][1]['goal'] - B['BM'][('off', x)][1]['goal'], 1, True) for x in BX)} pt) — 표본은 comm 위 {B['C_ABOVE']}/4.
시드 43(10-29)은 주 지표 {Q1['WIN_MAIN']}/{len(MAIN)} · comm 앞선 지표 {Q1['DIR_MAIN']}/{len(MAIN)} · 평균 행동 comm 위 {Q1['BM_ABOVE']}/4.
<b>메시지 절제(qd_comm, 시드 43)</b>: far56(56 m 밖 파트너 뺌) 도착 14M {f(ABL[('14', 'far56')]['goal'])} · 16M {f(ABL[('16', 'far56')]['goal'])} % = OFF({f(RD['14']['off'])} · {f(RD['16']['off'])}) ± {OFF_BAND:g} pt {'안' if RD['14']['R1'] and RD['16']['R1'] else '밖(일부)'} →
{"<b>통신 이득 = 56 m 밖 통신</b>(스펙 §6 읽는 법)" if RD['14']['R1'] and RD['16']['R1'] else '읽는 법 ① 일부 불성립'} · decl0 하락 {f(RD['14']['decl_d'])} · {f(RD['16']['decl_d'])} pt = 역할 선언 출처 문제 {'있음' if RD['14']['R2'] or RD['16']['R2'] else '없음'} ·
shuffle(내용 섞기)도 이득 대부분 지움. 다음은 저자 결정.</div>

<h2>1. 판정 — 결과 전 고정 기준</h2>
{verdict_table()}
<ul>
<li>평균 행동 = tanh(mean)(<code>eval_mean.py</code>), 표본 = 학습 정책 그대로. 곡선 · 갈래 평균 · 절제 평가 인자 같음(256 env · 3000 결정 · burn-in 2400 · drain 3000 · 시드 999)</li>
<li>Fig1 판정 · 곡선 M · 절제 표는 스크립트 표(<code>_fig1.md</code> · <code>_curve.md</code> · <code>_meaneval_qd2_.md</code> · <code>_ablate_qd_.md</code>)와 같은 식으로 다시 계산해 대조함. 순위 관문 v5 는 저자 override(재실행 없음)</li>
</ul>

<h2>2. qd2_ 갈래 — 주 평가(끝 모델 16.06M, 표본 행동, 256 env · 10000 결정 · drain 0)</h2>
{judge_table(B, MAIN)}
{judge_table(B, AUX)}
<ul>
<li>최선 OFF 풀(도착 ≥ 50 % 인 OFF): {' · '.join(sorted(B['POOL'])) if B['POOL'] else '없음'} · 보조 지표 comm 이 앞선 것 {B['DIR_AUX']}/{B['N_AUX']}</li>
<li>코덱 p46 k = <b>{B['CH_K']}</b>({' · '.join(f'k{k} {v}' for k, v in sorted(B['CPASS'].items()))}) — 시드 43 은 k = {Q1['CH_K']}(같은 관문 규칙, 시드마다 고름) · 분기 검사 ALL PASS(trunk sha256 {B['SHA']})</li>
<li>스냅샷 직접 확인(<code>q30_notes/snap_branch.txt</code>): 두 갈래 seed 44 · entropy_bonus 0 · logstd_min −3.0 · logstd_init0 −2.0 · REWARD_RANGE 56 · course · off = state / 56 m · comm = intent / 300 m</li>
</ul>

<h2 class='pb'>3. 두 시드 나란히 — qd_(시드 43, 10-29) · qd2_(시드 44)</h2>
{seed_table()}
<div class='fig'>{G_SEED}</div>
<ul>
<li>그림 = 갈래 평균 행동 도착(9.04M = trunk 분기점, 실선 시드 44 · 점선 시드 43). 시드 43 4점 comm − off {' · '.join(f(Q1['BM'][('comm', x)][1]['goal'] - Q1['BM'][('off', x)][1]['goal'], 1, True) for x in BX)} ·
시드 44 {' · '.join(f(Q2['BM'][('comm', x)][1]['goal'] - Q2['BM'][('off', x)][1]['goal'], 1, True) for x in BX)} pt</li>
<li>시드 44 는 평균 행동에서 off 가 10 · 12 · 14M 에 앞섬. comm 평균 − 표본 = {' · '.join(f(B['BM'][('comm', x)][1]['goal'] - B['BM'][('comm', x)][0]['goal'], 1, True) for x in BX)} pt(off {' · '.join(f(B['BM'][('off', x)][1]['goal'] - B['BM'][('off', x)][0]['goal'], 1, True) for x in BX)}) — 시드 44 comm 은 뒤 체크포인트에서 평균 행동이 표본보다 낮음</li>
<li>두 시드 주 평가(표본, 끝 모델) 6개 주 지표 중 두 시드 모두 comm 승 = {sum(1 for k, *_ in MAIN if Q1['J'][k]['w'] and Q2['J'][k]['w'])}개. 두 시드는 trunk 도 따로(시드마다 자기 trunk 에서 분기) — 스펙 §8-1 규약 그대로</li>
</ul>

<h2>4. qd2_ 갈래 — 체크포인트별 평균 행동 · 표본(저자 명령 3 + 고정 장면 곡선)</h2>
{bm_table(B)}
<div class='fig'>{g_cgo(B)}</div>
<div class='fig'>{G_CEP}</div>
<ul>
<li>표본 값 = 곡선 평가 파일(같은 인자). 16M 체크포인트 = 끝 모델(16.06M)과 같은 값 → 서로 다른 갈래 모델 4개(10 · 12 · 14 · 16.06M). 표본 도착 comm &gt; off = {B['C_ABOVE']}/4</li>
<li>곡선 M(epReward, 보고만): off 첫 {f(B['MC']['off'][0], 0)} · 최고 {f(B['MC']['off'][1], 0)} · 끝 {f(B['MC']['off'][2], 0)} · 최대 하락 {f(B['MC']['off'][3], 0)} /
comm 첫 {f(B['MC']['comm'][0], 0)} · 최고 {f(B['MC']['comm'][1], 0)} · 끝 {f(B['MC']['comm'][2], 0)} · 최대 하락 {f(B['MC']['comm'][3], 0)}</li>
</ul>
{curve_table(B)}

<h2 class='pb'>5. qd2_ 갈래 — 학습 중 창 값(표본 행동, 로그 한 줄 = update 5개)</h2>
<div class='fig'>{G_WIN}</div>
<div class='fig'>{G_WINV}</div>
{blk_table(B)}
<ul><li>갈래 21창 가운데 comm 도착 &gt; off = {B['N_ABOVE']}창 · 갈래 학습 off {f(B['SV']['off'][1])}분 · comm {f(B['SV']['comm'][1])}분(각자 GPU 하나)</li></ul>

<h2>6. qd_ 통신 메시지 절제 — 이득이 어디서 오나(학습 없음, 기록용)</h2>
{abl_table()}
<div class='fig'>{G_ABL}</div>
<ul>
<li>같은 qd_comm 체크포인트(14 · 16M)에 입력만 바꿔 평균 행동으로 평가. '기본 통신' · 'OFF 갈래' = 10-29 갈래 평균 행동 평가 값(같은 인자). OFF 갈래는 다른 모델(같은 trunk 에서 분기한 OFF)</li>
<li>'통신 이득 중 남은 몫' = (조건 − OFF) ÷ (기본 통신 − OFF). 읽는 법(§6)에 없는 값 — 보고만</li>
<li>far56 = 평가 전용 <code>--far_drop 56</code>(5e678dc, 기본 None = 학습 동작 불변) → 파트너 집합이 OFF(ARPA@56)와 같아짐. 56 m 안 파트너의 메시지는 그대로 받음</li>
</ul>

<h2>7. 경과 · 실행 기록(세션)</h2>
<ul>
{''.join(f"<li>{ln}{('<ul>' + ''.join(f'<li>{c}</li>' for c in sub) + '</ul>') if sub else ''}</li>" for ln, sub in NOTE)}
</ul>

<h2>8. 한계 — 그대로 읽을 것</h2>
<ul>
<li>갈래 비교는 배치마다 시드 1개 · offb 없음 → 잡음 자가 없음. N = 고정 하한뿐. 시드 43 · 44 두 개 = 단독 주장 못 함(스펙 §4 남은 것 = 시드 ×3)</li>
<li>절제는 시드 43 comm 체크포인트 2개만(14 · 16M). 시드 44 절제 안 함. 평가는 조건 · 체크포인트당 1회 · 평가 시드 1개(999)</li>
<li>절제 '조건 − OFF' 는 다른 모델끼리 비교(OFF 갈래). 같은 체크포인트 재측정 흔들림 크기 모름</li>
<li>관문 · 판정식 · 평가 스크립트는 안 건드렸음. 저자 명령 그대로 실행 — 다르게 한 것 없음(§7)</li>
</ul>

<h2>9. 다음 — 저자 결정</h2>
<ul>
<li>스펙 §4 성공 경로에서 남은 것: 하나씩 뺀 trunk · 시드 ×3. 절제 결과로 무엇을 할지는 저자 결정</li>
<li>결정 전까지 추가 학습 · 평가 안 돌림</li>
</ul>
<div class='small'>자료: runs/2026-10-02_fig1/out/qd2/ · qd/ · meaneval_qd2/ · meaneval_qd/ · ablate_qd/ · q30_notes/ · 생성 report/make_report_q30.py ·
앞 보고 2026-10-29_병렬3건_qd갈래_qe_qd2_결과.pdf</div>
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
print('qd2_ 갈래 주 지표', Q2['WIN_MAIN'], '/', len(MAIN), '방향', Q2['DIR_MAIN'], '| 평균 행동 comm>off', Q2['BM_ABOVE'], '/4', f(Q2['BM_K']['off']), f(Q2['BM_K']['comm']),
      '| M', Q2['MC']['off'][4], Q2['MC']['comm'][4], '| 절제 R1', RD['14']['R1'], RD['16']['R1'], 'R2', RD['14']['R2'], RD['16']['R2'])
print(out_pdf)
