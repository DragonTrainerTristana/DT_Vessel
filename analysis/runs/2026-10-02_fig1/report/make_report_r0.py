# 2026-10-27: R0 무학습 재평가(스펙 2026-10-26-ref-port-sigma-trunk-design.md §2) 결과 PDF. 새 학습·평가 없음.
# 숫자는 전부 로그에서 읽음(runs/2026-10-02_fig1/out/):
#   meaneval_qb/ · meaneval_qa/   sample_<run>_<x>M.txt(표본 행동) · mean_<run>_<x>M.txt(평균 행동 tanh(mean)) · logstd_*.txt · _meaneval_*.md · _log.txt · _extra_log.txt
#   diag_trunk_n/                 n_ 9.04M · 4M 같은 인자 평가(10-13 단계 1 D) — 비교 줄
#   qb/curve_qb_*.txt             10-13 phase1 고정 장면 곡선(표본) — qb_ 표본 재사용 확인
#   r0_notes/runnote.txt          실행 기록
# 표 값은 meaneval_ckpts.sh 가 만든 _meaneval_*.md 행과 대조(assert). 본문 문장이 전제로 삼는 방향도 assert.
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_r0.py [출력 폴더]
import glob, io, os, re, subprocess, sys

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', 'out'))
NAME = '2026-10-27_R0_무학습재평가_표본vs평균행동_결과'
RUN_ARGS = 'burnin=2400 dec=3000 drain=3000 ring=1.0 crossing=0 seed=999 envs=256 vessels=16'
SEED = 43


def text(*p):
    with io.open(os.path.join(OUT, *p), encoding='utf-8', errors='replace') as fh:
        return fh.read()


def f(v, nd=1, sign=False):
    if v is None:
        return 'n/a'
    s = f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'
    return s.replace('-', '−')


LINE = re.compile(r'^(\S+\.pt)\s+\| arm=ON\s+\| goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%\s+oColl=\s*([\d.]+)%\s+TO=\s*([\d.]+)%\s+\|\s+(\d+) eps\b.*?epReward=\s*(-?[\d.]+)', re.M)


def parse(rel, ckpt, mean_mode):
    t = text(rel)
    m = LINE.search(t)
    run = re.search(r'\[run\] (.*)', t)
    assert m and run and run.group(1).strip().startswith(RUN_ARGS), (rel, run and run.group(1))
    assert m.group(1) == ckpt, (rel, m.group(1), ckpt)
    assert ('[eval_mean] student drives with tanh(mean)' in t) == mean_mode, rel
    return dict(goal=float(m.group(2)), vColl=float(m.group(3)), oColl=float(m.group(4)), TO=float(m.group(5)),
                eps=int(m.group(6)), epReward=float(m.group(7)))


def ckname(pre, run, x):
    return f'{pre}{run}_s{SEED}.pt' if (run == 'trunk_d6' and x == '9.04397') else f'{pre}{run}_s{SEED}.step{x}M.pt'


XS = {'trunk_d6': ['2', '4', '6', '8', '9.04397'], 'off': ['10', '12', '14', '16'], 'comm': ['10', '12', '14', '16']}
R = {}   # (pre, run) -> [(x, sample, mean)]
for pre, runs in (('qb_', ('trunk_d6', 'off', 'comm')), ('qa_', ('trunk_d6',))):
    d = f'meaneval_{pre[:-1]}'
    for run in runs:
        rows = []
        for x in XS[run]:
            ck = ckname(pre, run, x)
            s = parse(os.path.join(d, f'sample_{pre}{run}_{x}M.txt'), ck, False)
            a = parse(os.path.join(d, f'mean_{pre}{run}_{x}M.txt'), ck, True)
            rows.append((float(x), s, a))
        R[(pre, run)] = rows
# 스크립트 표(_meaneval_*.md)와 대조
for pre in ('qb_', 'qa_'):
    md = text(f'meaneval_{pre[:-1]}', f'_meaneval_{pre}.md')
    n = 0
    for (p, run), rows in R.items():
        if p != pre:
            continue
        for x, s, a in rows:
            row = f"| {run} | {x:g}M | {s['goal']:.1f} / {a['goal']:.1f} ({a['goal'] - s['goal']:+.1f}) | {s['vColl']:.1f} / {a['vColl']:.1f} | {s['epReward']:.0f} / {a['epReward']:.0f} |"
            assert row in md, (pre, row)
            n += 1
    assert md.count('\n| ') - 2 == n + sum(1 for k in R if k[0] == pre), (pre, n)   # 체크포인트 행 + K 행(머리줄 2개 제외)
# qb_ 표본 = 10-13 고정 장면 곡선 재사용(같은 인자) — 값이 같아야 함
for run in ('trunk_d6', 'off', 'comm'):
    for x, s, _ in R[('qb_', run)]:
        cur = parse(os.path.join('qb', f'curve_qb_{run}_s{SEED}_{x:g}M.txt'), ckname('qb_', run, f'{x:g}'), False)
        assert cur == s, (run, x)
for run in ('off', 'comm'):   # step16M = 끝 모델(16.0563M) 같은 파일 → 곡선 값도 같음
    end = parse(os.path.join('qb', f'curve_qb_{run}_s{SEED}_16.0563M.txt'), f'qb_{run}_s{SEED}.pt', False)
    assert end == R[('qb_', run)][-1][1], run

# n_ 비교 줄(10-13 단계 1 D, 같은 인자)
NREF = {}
for x, tag, ck in ((4.0, '4M', f'n_trunk_d6_s{SEED}.step4M.pt'), (9.04397, '9M', f'n_trunk_d6_s{SEED}.pt')):
    NREF[x] = (parse(os.path.join('diag_trunk_n', f'eval_sample_{tag}.txt'), ck, False),
               parse(os.path.join('diag_trunk_n', f'eval_mean_{tag}.txt'), ck, True))

# logstd 덤프
LS = {}
LSL = re.compile(r'^(\S+\.pt)\s+steps=(\d+) logstd0=([-\d. ]+?) mean=(-?[\d.]+) sigma_course=([\d.]+)deg', re.M)
for pre, run in R:
    t = text(f'meaneval_{pre[:-1]}', f'logstd_{pre}{run}.txt')
    rows = [(m.group(1), int(m.group(2)), [float(v) for v in m.group(3).split()], float(m.group(4)), float(m.group(5))) for m in LSL.finditer(t)]
    assert len(rows) == len(R[(pre, run)]), (pre, run, len(rows))
    LS[(pre, run)] = rows
NSIG = {}
for m in LSL.finditer(text('diag_trunk_n', 'logstd.txt')):
    if m.group(1) == f'n_trunk_d6_s{SEED}.step4M.pt':
        NSIG[4.0] = float(m.group(5))
    elif m.group(1) == f'n_trunk_d6_s{SEED}.pt':
        NSIG[9.04397] = float(m.group(5))
assert len(NSIG) == 2, NSIG

# K 점 평균(스크립트와 같은 식: 그 run 의 체크포인트 전부, 선택 없음)
def kavg(rows, which, key):
    # 순차 합(왼쪽부터 더함) — Python 3.12+ 의 sum() 은 보정 합산이라 mltest(3.x 구버전) 표와 끝자리 반올림이 달라짐(42.45 → 42.4 vs 42.5)
    t = 0.0
    for r in rows:
        t += r[which][key]
    return t / len(rows)


for pre in ('qb_', 'qa_'):
    md = text(f'meaneval_{pre[:-1]}', f'_meaneval_{pre}.md')
    for (p, run), rows in R.items():
        if p != pre:
            continue
        k = len(rows)
        row = (f"| {run} | {k} | {kavg(rows, 1, 'goal'):.1f} / {kavg(rows, 2, 'goal'):.1f} | {kavg(rows, 1, 'vColl'):.1f} / {kavg(rows, 2, 'vColl'):.1f} | "
               f"{kavg(rows, 1, 'epReward'):.0f} / {kavg(rows, 2, 'epReward'):.0f} |")
        assert row in md, row

ALL = [(pre, run, x, s, a) for (pre, run), rows in R.items() for x, s, a in rows]
GAP = [a['goal'] - s['goal'] for *_, s, a in ALL]
N_ALL, N_POS, N_10 = len(GAP), sum(g > 0 for g in GAP), sum(g >= 10 for g in GAP)
N_VC = sum(a['vColl'] < s['vColl'] for *_, s, a in ALL)
OFF, COMM = R[('qb_', 'off')], R[('qb_', 'comm')]
QA, QBT = R[('qa_', 'trunk_d6')], R[('qb_', 'trunk_d6')]
W_S = sum(c[1]['goal'] > o[1]['goal'] for o, c in zip(OFF, COMM))
W_A = sum(c[2]['goal'] > o[2]['goal'] for o, c in zip(OFF, COMM))
O14 = OFF[2]
LOO = {nm: [r for r in rows if r[0] != 14.0] for nm, rows in (('off', OFF), ('comm', COMM))}
QA_BEST = max(QA, key=lambda r: r[2]['goal'])
N9 = NREF[9.04397]
T9 = {'n_': N9, 'qb_': (QBT[-1][1], QBT[-1][2]), 'qa_': (QA[-1][1], QA[-1][2])}
# 본문 문장이 전제로 삼는 방향(값이 달라지면 문장을 고쳐야 하므로 여기서 멈춤)
assert O14[0] == 14.0 and O14[2]['goal'] < 10 and O14[1]['goal'] < 10 and O14[2]['goal'] < O14[1]['goal']
assert OFF[-1][1]['goal'] > COMM[-1][1]['goal'] and OFF[-1][2]['goal'] > COMM[-1][2]['goal']          # 끝점: 표본 · 평균 모두 off 위
assert kavg(COMM, 1, 'goal') > kavg(OFF, 1, 'goal') and kavg(COMM, 2, 'goal') > kavg(OFF, 2, 'goal')  # K 점 평균: 둘 다 comm 위
assert W_S == 2 and W_A == 2
assert QBT[0][2]['goal'] - QBT[0][1]['goal'] < 1.0                                                     # qb_ trunk 2M 은 격차 거의 0
T9A = [T9[k][1]['goal'] for k in T9]
T9S = [T9[k][0]['goal'] for k in T9]
assert max(T9A) - min(T9A) < 2.0 and max(T9S) - min(T9S) > 10.0                                       # 평균 행동은 붙고 표본은 벌어짐
G10 = [g for g in GAP if g >= 10]
SIG = {k: LS[k][-1][4] for k in LS}
assert SIG[('qa_', 'trunk_d6')] > 2 * SIG[('qb_', 'trunk_d6')]
GAP9 = {k: T9[k][1]['goal'] - T9[k][0]['goal'] for k in T9}
assert min(GAP9, key=GAP9.get) == 'qb_' and GAP9['qb_'] > 10 and abs(NSIG[9.04397] - SIG[('qa_', 'trunk_d6')]) < 5
QBT_S = [s['goal'] for _, s, _ in QBT]
QBT_A = [a['goal'] for _, _, a in QBT]
assert QA_BEST[0] == 4.0

C_TR, C_OFF, C_COMM, C_QA, C_N = '#9c9a94', '#d9822b', '#1b6fd6', '#7a4fb5', '#3f8f5a'


def chart(series, x0, x1, xticks, y0, y1, yticks, w=700, h=170, vline=None):
    ml, mr, mt, mb = 34, 8, 18, 18
    X = lambda d: ml + (w - ml - mr) * (d - x0) / (x1 - x0)  # noqa: E731
    Y = lambda v: mt + (h - mt - mb) * (y1 - v) / (y1 - y0)  # noqa: E731
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='100%'>"]
    for t in yticks:
        s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' stroke='#e4e3df' stroke-width='0.6'/>")
        s.append(f"<text x='{ml - 4}' y='{Y(t) + 2.6:.1f}' font-size='7.2' text-anchor='end' fill='#52514e'>{f(t, 0)}</text>")
    for d in xticks:
        s.append(f"<text x='{X(d):.1f}' y='{h - 5}' font-size='7.2' text-anchor='middle' fill='#52514e'>{d:.2f}M</text>" if d != int(d) else
                 f"<text x='{X(d):.1f}' y='{h - 5}' font-size='7.2' text-anchor='middle' fill='#52514e'>{d:g}M</text>")
    s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{h - mb}' y2='{h - mb}' stroke='#0b0b0b' stroke-width='0.7'/>")
    if vline:
        bx = X(vline)
        s.append(f"<line x1='{bx:.1f}' x2='{bx:.1f}' y1='{mt}' y2='{h - mb}' stroke='#52514e' stroke-width='0.7' stroke-dasharray='3 2'/>")
        s.append(f"<text x='{bx + 3:.1f}' y='{mt + 8}' font-size='7.2' fill='#52514e'>분기 9.04M</text>")
    lx = ml + 4
    for lab, pts, col, dash in series:
        p = ' '.join(f"{X(d):.1f},{Y(v):.1f}" for d, v in pts)
        da = f" stroke-dasharray='{dash}'" if dash else ''
        s.append(f"<polyline points='{p}' fill='none' stroke='{col}' stroke-width='1.5'{da}/>")
        s += [f"<circle cx='{X(d):.1f}' cy='{Y(v):.1f}' r='2.0' fill='{'#fff' if dash else col}' stroke='{col}' stroke-width='1'/>" for d, v in pts]
        s.append(f"<line x1='{lx}' x2='{lx + 16}' y1='8' y2='8' stroke='{col}' stroke-width='1.5'{da}/>")
        s.append(f"<text x='{lx + 19}' y='10.6' font-size='7.4' fill='#0b0b0b'>{lab}</text>")
        lx += 19 + 7.4 * len(lab) + 12
    return ''.join(s) + '</svg>'


pts = lambda rows, i, k='goal': [(x, r[k]) for x, *rr in rows for r in [rr[i - 1]]]  # noqa: E731
join = lambda rows, i: [(QBT[-1][0], QBT[-1][i]['goal'])] + pts(rows, i)  # noqa: E731
G_QB = chart([('trunk 평균', pts(QBT, 2), C_TR, ''), ('trunk 표본', pts(QBT, 1), C_TR, '3 2'),
              ('off 평균', join(OFF, 2), C_OFF, ''), ('off 표본', join(OFF, 1), C_OFF, '3 2'),
              ('comm 평균', join(COMM, 2), C_COMM, ''), ('comm 표본', join(COMM, 1), C_COMM, '3 2')],
             0, 16.6, [2, 4, 6, 8, 10, 12, 14, 16], 0, 80, [0, 20, 40, 60, 80], vline=9.04397)
NP = sorted(NREF.items())
G_TR = chart([('qa_ 평균', pts(QA, 2), C_QA, ''), ('qa_ 표본', pts(QA, 1), C_QA, '3 2'),
              ('qb_ 평균', pts(QBT, 2), C_TR, ''), ('qb_ 표본', pts(QBT, 1), C_TR, '3 2'),
              ('n_ 평균', [(x, v[1]['goal']) for x, v in NP], C_N, ''), ('n_ 표본', [(x, v[0]['goal']) for x, v in NP], C_N, '3 2')],
             1.5, 9.5, [2, 4, 6, 8, 9.04397], 0, 80, [0, 20, 40, 60, 80], h=150)

LAB = {('qb_', 'trunk_d6'): 'qb_ trunk', ('qb_', 'off'): 'qb_ off', ('qb_', 'comm'): 'qb_ comm', ('qa_', 'trunk_d6'): 'qa_ trunk'}


def main_table():
    h = ("<table><thead><tr><th>런</th><th>체크포인트</th><th>σ 침로 °</th><th>도착 % 표본</th><th>도착 % 평균</th><th>평균 − 표본</th>"
         "<th>배 충돌 % 표본 / 평균</th><th>벽 · 시간초과 % 평균</th><th>epReward 표본 / 평균</th><th>에피소드 수 표본 / 평균</th></tr></thead><tbody>")
    for key in (('qb_', 'trunk_d6'), ('qb_', 'off'), ('qb_', 'comm'), ('qa_', 'trunk_d6')):
        for (x, s, a), lsr in zip(R[key], LS[key]):
            g = a['goal'] - s['goal']
            cls = 'bad' if g >= 10 else ''
            h += (f"<tr><td>{LAB[key]}</td><td class='c'>{x:g}M</td><td class='c'>{f(lsr[4])}</td><td class='c'>{f(s['goal'])}</td><td class='c'>{f(a['goal'])}</td>"
                  f"<td class='c {cls}'>{f(g, 1, True)}</td><td class='c'>{f(s['vColl'])} / {f(a['vColl'])}</td><td class='c'>{f(a['oColl'])} · {f(a['TO'])}</td>"
                  f"<td class='c'>{f(s['epReward'], 0)} / {f(a['epReward'], 0)}</td><td class='c'>{s['eps']} / {a['eps']}</td></tr>")
    for x, (s, a) in NP:
        h += (f"<tr class='ref'><td>n_ trunk (10-13 D)</td><td class='c'>{x:g}M</td><td class='c'>{f(NSIG[x])}</td><td class='c'>{f(s['goal'])}</td><td class='c'>{f(a['goal'])}</td>"
              f"<td class='c bad'>{f(a['goal'] - s['goal'], 1, True)}</td><td class='c'>{f(s['vColl'])} / {f(a['vColl'])}</td><td class='c'>{f(a['oColl'])} · {f(a['TO'])}</td>"
              f"<td class='c'>{f(s['epReward'], 0)} / {f(a['epReward'], 0)}</td><td class='c'>{s['eps']} / {a['eps']}</td></tr>")
    return h + "</tbody></table>"


def order_table():
    def cell(o, c):
        w = 'comm' if c > o else 'off'
        return f"<td class='c'>{f(o)}</td><td class='c'>{f(c)}</td><td class='c'><b>{w}</b> ({f(c - o, 1, True)})</td>"
    h = ("<table><thead><tr><th>추정량</th><th>행동</th><th>off 도착 %</th><th>comm 도착 %</th><th>위 (comm − off)</th></tr></thead><tbody>")
    for i, nm in ((1, '표본'), (2, '평균')):
        h += f"<tr><td>끝점 16M(= 끝 모델)</td><td class='c'>{nm}</td>{cell(OFF[-1][i]['goal'], COMM[-1][i]['goal'])}</tr>"
    for i, nm in ((1, '표본'), (2, '평균')):
        h += f"<tr><td>K 점 평균(10 · 12 · 14 · 16M)</td><td class='c'>{nm}</td>{cell(kavg(OFF, i, 'goal'), kavg(COMM, i, 'goal'))}</tr>"
    for i, nm in ((1, '표본'), (2, '평균')):
        h += f"<tr class='ref'><td>확인용: K 평균에서 14M 뺌</td><td class='c'>{nm}</td>{cell(kavg(LOO['off'], i, 'goal'), kavg(LOO['comm'], i, 'goal'))}</tr>"
    for i, nm, w in ((1, '표본', W_S), (2, '평균', W_A)):
        h += f"<tr><td>체크포인트별 승패(4개)</td><td class='c'>{nm}</td><td class='c' colspan='3'>comm 위 {w} · off 위 {4 - w}</td></tr>"
    return h + "</tbody></table>"


def t9_table():
    h = ("<table><thead><tr><th>trunk 9.04M</th><th>다른 점(같은 보상 · 시드 43)</th><th>σ 침로 °</th><th>도착 % 표본</th><th>도착 % 평균</th><th>평균 − 표본</th><th>배 충돌 % 평균</th></tr></thead><tbody>")
    sig = {'qb_': LS[('qb_', 'trunk_d6')][-1][4], 'qa_': LS[('qa_', 'trunk_d6')][-1][4], 'n_': NSIG[9.04397]}
    desc = {'n_': 'ARPA@56 · 토글 없음', 'qb_': 'ARPA@56 · logstd 초기 −2.0 · LR 감쇠', 'qa_': 'ARPA@300 · 토글 없음'}
    for k in ('n_', 'qb_', 'qa_'):
        s, a = T9[k]
        h += (f"<tr><td class='c'>{k}</td><td>{desc[k]}</td><td class='c'>{f(sig[k]) if k in sig else '—'}</td><td class='c'>{f(s['goal'])}</td><td class='c'>{f(a['goal'])}</td>"
              f"<td class='c'>{f(a['goal'] - s['goal'], 1, True)}</td><td class='c'>{f(a['vColl'])}</td></tr>")
    return h + "</tbody></table>"


NOTE = [ln[2:] for ln in text('r0_notes', 'runnote.txt').splitlines()[1:] if ln.startswith('- ')]
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
td.bad { color: #a8322a; font-weight: 600; }
tr.ref td { color: #52514e; background: #f7f6f2; }
code { font-family: Consolas, monospace; font-size: 8pt; background: #f4f3ef; padding: 0 2px; }
.small { color: #52514e; font-size: 7.9pt; }
.fig { margin: 4px 0 0 0; }
.pb { page-break-before: always; }
"""
KO, KC = kavg(OFF, 2, 'goal'), kavg(COMM, 2, 'goal')
HTML = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{NAME}</title><style>{CSS}</style></head><body>
<h1>R0 무학습 재평가 — 표본 행동 vs 평균 행동 (qb_ · qa_)</h1>
<div class='sub1'>2026-10-27 · LICS 서버 · 코드 feat/fig1-latent 071487b · 스펙 2026-10-26-ref-port-sigma-trunk-design.md §2(기록용 · 관문 아님) · 학습 없음 ·
이미 있는 체크포인트 {N_ALL}개(qb_ trunk 5 · off 4 · comm 4 · qa_ trunk 5) · 시드 {SEED} 하나 · 평가 256 env · 3000 결정 · burn-in 2400 · drain 3000 · 평가 시드 999(고정 장면 곡선과 같은 인자)</div>

<div class='box'><b>한 줄 결론</b> — 평균 행동(tanh(mean), 표본 잡음 없음)이 표본 행동보다 도착이 높은 체크포인트 <b>{N_POS}/{N_ALL}</b>, 그 차가 10 pt 이상 <b>{N_10}/{N_ALL}</b>({f(min(G10), 1, True)} – {f(max(G10), 1, True)} pt).
배끼리 충돌도 평균 행동이 낮은 체크포인트 {N_VC}/{N_ALL}. 예외는 qb_ trunk 2M(+{f(QBT[0][2]['goal'] - QBT[0][1]['goal'])} pt)과 qb_ off 14M(평균 행동도 도착 {f(O14[2]['goal'])} % 로 무너짐).
trunk 3개(n_ · qb_ · qa_)의 9.04M 평균 행동 도착은 {f(T9['n_'][1]['goal'])} · {f(T9['qb_'][1]['goal'])} · {f(T9['qa_'][1]['goal'])} % 로 비슷함 — 표본 행동({f(T9['n_'][0]['goal'])} · {f(T9['qb_'][0]['goal'])} · {f(T9['qa_'][0]['goal'])} %)만 벌어짐.
qa_(ARPA@300) 평균 행동 도착은 4M 이 최고({f(QA_BEST[2]['goal'])} %), 9.04M {f(QA[-1][2]['goal'])} %.
qb_ off · comm 순서는 추정량에 따라 바뀜: 끝점(16M)은 표본 · 평균 모두 off 위, K 점 평균은 둘 다 comm 위(평균 행동 {f(KC)} vs {f(KO)} %) — 이 차는 off 14M 하나에서 나옴(빼면 거의 같음).
시드 1개 · 기록용 → 원인 주장 · 팔 판정 안 함. R1(σ trunk 3팔)은 저자 승인 대기.</div>

<h2>1. 체크포인트별 표본 / 평균 행동</h2>
{main_table()}
<ul>
<li>표본 = 학습 정책 그대로(매 결정 가우시안 표본 → tanh), 평균 = 같은 정책의 tanh(mean)(<code>eval_mean.py</code>, ctr_actor 첫 반환값만 바꿈). 나머지 설정 · 장면 · 평가 시드 같음</li>
<li>σ 침로 = <code>logstd_dump.py</code> 의 sigma_course = 120° × exp(행동 0번 logstd 의 전문가 5개 평균), tanh 전 근사값. qb_ 는 학습 때 logstd 초기 −2.0 이라 17.8–22.0°, qa_ 는 기본 초기값이라 47.2–48.7°</li>
<li>qb_ 표본 값은 10-13 phase1 고정 장면 곡선(<code>curve_qb_*.txt</code>, 같은 인자)을 그대로 씀 — 값이 같음을 확인. qa_ 표본은 이번에 새로 잼</li>
<li>회색 줄 n_ = 10-13 단계 1 D(같은 인자) 값을 비교로 옮겨 적음. 이번에 다시 잰 것 아님</li>
<li>epReward 는 모양 보상(shaped) 합 — 평균 행동이 대부분 양수로 바뀌나 판정 지표 아님</li>
</ul>

<h2>2. 그림 — 도착 % (실선 = 평균 행동, 점선 = 표본 행동)</h2>
<div class='fig'>{G_QB}</div>
<div class='fig'>{G_TR}</div>

<h2 class='pb'>3. 스펙 §2 보고 항목</h2>
<p><b>(a) 체크포인트별 평균 − 표본 격차</b></p>
<ul>
<li>{N_ALL}개 중 양수 {N_POS} · 10 pt 이상 {N_10}. 런별 범위: qb_ trunk {f(min(a['goal'] - s['goal'] for _, s, a in QBT), 1, True)} – {f(max(a['goal'] - s['goal'] for _, s, a in QBT), 1, True)} ·
qb_ off {f(min(a['goal'] - s['goal'] for _, s, a in OFF), 1, True)} – {f(max(a['goal'] - s['goal'] for _, s, a in OFF), 1, True)} ·
qb_ comm {f(min(a['goal'] - s['goal'] for _, s, a in COMM), 1, True)} – {f(max(a['goal'] - s['goal'] for _, s, a in COMM), 1, True)} ·
qa_ trunk {f(min(a['goal'] - s['goal'] for _, s, a in QA), 1, True)} – {f(max(a['goal'] - s['goal'] for _, s, a in QA), 1, True)} pt</li>
<li>qb_ trunk 2M 만 격차 ≈ 0(표본 {f(QBT[0][1]['goal'])} · 평균 {f(QBT[0][2]['goal'])} %). 4M 부터 격차가 생김. 표본 도착 {' → '.join(f(v) for v in QBT_S)} %, 평균 행동 {' → '.join(f(v) for v in QBT_A)} %</li>
<li>스펙 R1 처방 성공 기준 제안(격차 ≤ 10 pt)을 이 자료에 대 보면 9.04M 에서 n_ · qb_ · qa_ 모두 못 미침 — R1 은 이 격차를 줄이는지 보는 실험임</li>
</ul>
<p><b>(b) trunk 9.04M 세 개 비교</b></p>
{t9_table()}
<ul>
<li>평균 행동 도착은 세 trunk 가 {f(min(T9[k][1]['goal'] for k in T9))}–{f(max(T9[k][1]['goal'] for k in T9))} % 로 붙어 있음. 표본 행동은 {f(min(T9S))}–{f(max(T9S))} % 로 벌어짐</li>
<li>σ 침로: n_ {f(NSIG[9.04397])}° · qa_ {f(SIG[('qa_', 'trunk_d6')])}° 는 비슷하고 qb_ {f(SIG[('qb_', 'trunk_d6')])}° 만 작음(logstd 초기 −2.0). 격차는 n_ {f(T9['n_'][1]['goal'] - T9['n_'][0]['goal'], 1, True)} · qa_ {f(T9['qa_'][1]['goal'] - T9['qa_'][0]['goal'], 1, True)} · qb_ {f(T9['qb_'][1]['goal'] - T9['qb_'][0]['goal'], 1, True)} pt — σ 가 작은 qb_ 가 가장 작으나 0 은 아님. 세 trunk 는 보는 거리 · 토글이 같이 달라서 σ 하나의 효과로 못 읽음</li>
<li>qa_ 평균 행동: {' → '.join(f(a['goal']) for _, _, a in QA)} % ({' · '.join(f'{x:g}' for x, _, _ in QA)}M) — 4M 최고 뒤 내려옴. 10-13 단계 1 의 B 확정 기준(도착 ≥ 50)은 학습 창 값 기준이라 이 숫자로 바꾸지 않음</li>
</ul>
<p><b>(c) qb_ off · comm 순서가 평가 방식 · 추정량에 따라 바뀌는지</b></p>
{order_table()}
<ul>
<li>바뀜. 끝점은 off 위(평균 행동 차 {f(COMM[-1][2]['goal'] - OFF[-1][2]['goal'], 1, True)} pt 로 거의 같음), K 점 평균은 comm 위. 체크포인트별로는 표본 · 평균 모두 comm 2 · off 2</li>
<li>K 점 평균의 comm 우세는 off 14M 하나에서 나옴: off 14M 은 표본 {f(O14[1]['goal'])} % · 평균 {f(O14[2]['goal'])} %(배 충돌 {f(O14[2]['vColl'])} %) — <b>평균 행동도 무너짐 → 그 시점 정책 자체 문제</b>(잡음 탓 아님), 16M 에서 회복.
14M 을 빼면(확인용, 판정 아님) 평균 행동 off {f(kavg(LOO['off'], 2, 'goal'))} · comm {f(kavg(LOO['comm'], 2, 'goal'))} %</li>
<li>off · comm 의 σ 침로는 분기 뒤 거의 같게 움직임(16M off {f(SIG[('qb_', 'off')])}° · comm {f(SIG[('qb_', 'comm')])}°) → 두 팔의 잡음 크기 차로는 순서를 설명 못 함</li>
</ul>

<h2>4. 경과 · 실행 기록</h2>
<ul>
{''.join(f'<li>{ln}</li>' for ln in NOTE)}
</ul>

<h2>5. 한계 — 그대로 읽을 것</h2>
<ul>
<li>시드 1개(43) · 평가 시드 1개(999) · 체크포인트당 평가 1회 → 같은 체크포인트를 다시 잰 흔들림 크기 모름. qb_ off 14M 같은 한 점이 평균을 좌우함</li>
<li>평균 행동은 진단용 평가임. 주 평가를 평균 행동으로 바꿀지는 저자 결정(C1) — 이 보고는 둘 다 기록만 함</li>
<li>격차의 원인(침로 모드 잡음 크기 · 잡음 상관 · 정책이 잡음에 맞춰 학습됐는지)은 이 자료로 못 가름. R1 이 그걸 보는 실험임</li>
<li>세 trunk(n_ · qb_ · qa_)는 서로 둘 이상 조건이 달라 단독 비교 안 됨</li>
<li>관문 · 판정식 · 스크립트는 안 건드렸음. 속도용 추가 프로세스는 같은 명령 · 같은 인자(§4)</li>
</ul>

<h2>6. 저자 결정 대기</h2>
<ul>
<li>R1(σ trunk 3팔 ra_ · rb_ · rc_, 75분 동시) 승인 여부 · 처방 성공 기준 확정(제안: 9.04M 표본 도착 ≥ 55 % 이고 격차 ≤ 10 pt)</li>
<li>C1: 주 평가를 평균 행동으로 바꿀지(바꾸면 평가 프로토콜 변경 = 저자 결정)</li>
<li>결정 전까지 추가 학습 · 평가 안 돌림</li>
</ul>
<div class='small'>자료: runs/2026-10-02_fig1/out/meaneval_qb/ · meaneval_qa/(표본 · 평균 평가 로그, logstd, _meaneval_*.md, _log.txt, _extra_log.txt) · out/diag_trunk_n/ · out/qb/curve_qb_*.txt ·
out/r0_notes/(runnote.txt · r0_extra.sh · r0_logs/) · 서버 Python/_repro_out_qb/meaneval · _repro_out_qa/meaneval · 생성 report/make_report_r0.py</div>
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
print('격차>0', N_POS, '/', N_ALL, '| ≥10pt', N_10, '| vColl 평균<표본', N_VC, '| 끝점 off 위 · K 평균 comm 위', f(KO), f(KC), '| 체크포인트 comm 위 표본', W_S, '평균', W_A,
      '| 9.04M 평균', {k: T9[k][1]['goal'] for k in T9})
print(out_pdf)
