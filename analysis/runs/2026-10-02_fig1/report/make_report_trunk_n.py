# 2026-10-09: 배치 n_ trunk 관문 불통과 보고 PDF. 새 학습·평가 없음. 숫자는 전부 배치 로그에서 읽음:
#   runs/2026-10-02_fig1/out/n_trunk_d6_s43.log      (학습 창 = update 5개, 관문이 읽는 줄)
#   runs/2026-10-02_fig1/out/n_trunk_d6_s43_ep.csv   (update 마다 끝난 에피소드 결과) · n_trunk_d6_s43.csv (update 마다 결정당 보상)
#   runs/2026-10-02_fig1/out/_gate.log · _timeline.txt · _rank_override.txt
#   대조: out/f2_trunk_d6_s43.log (10-06 배치 f2_, 흉내 있음, 같은 시드 43)
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_trunk_n.py [출력 폴더]
import csv, io, os, re, subprocess, sys

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', 'out'))
NAME = '2026-10-09_n배치_trunk관문_결과'
SEED = 43
UPD = 65536            # update 하나 = 128 env × 16척 × rollout 32
TRUNK_DEC = 9043968    # = 138 update
GATE_GOAL, GATE_OCOLL, GATE_N = 30.0, 5.0, 4   # _run_f.sh gate9 (t_ 규칙, 결과 전 고정)


def text(*p):
    with io.open(os.path.join(OUT, *p), encoding='utf-8') as fh:
        return fh.read()


def fm(v, nd=1, sign=False):
    s = f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'
    return s.replace('-', '−')


WIN = re.compile(r'^\[ON\] dec=([\d.]+)M \| ep=(\d+) len~(\d+) \| goal=([\d.]+)% vColl=([\d.]+)% oColl=([\d.]+)% TO=([\d.]+)% '
                 r'\| R=(-?[\d.]+) \| (\d+) dec/s \| epRet=(-?[\d.]+)')
KEYS = ('dec', 'ep', 'len', 'goal', 'vColl', 'oColl', 'TO', 'R', 'dps', 'epRet')


def windows(name):
    rows = []
    for ln in text(name).splitlines():
        m = WIN.match(ln)
        if m:
            rows.append(dict(zip(KEYS, (float(x) for x in m.groups()))))
    return rows


def saved(name):
    m = re.search(r'saved -> \S+ \(([\d.]+)M decisions, ([\d.]+)min\)', text(name))
    return float(m.group(1)), float(m.group(2))


def mean(rows, k):
    return sum(r[k] for r in rows) / len(rows)


N = windows('n_trunk_d6_s43.log')
F = windows('f2_trunk_d6_s43.log')
N_DEC, N_MIN = saved('n_trunk_d6_s43.log')
F_DEC, F_MIN = saved('f2_trunk_d6_s43.log')
assert len(N) == 27 and len(F) == 27 and N_DEC == F_DEC == 9.04, (len(N), len(F), N_DEC, F_DEC)
assert 'DAgger' not in text('n_trunk_d6_s43.log') and '[bc]' not in text('n_trunk_d6_s43.log') and '[resume]' not in text('n_trunk_d6_s43.log')
assert '[bc] 흉내 보조손실' in text('f2_trunk_d6_s43.log') and '[resume] f2_dagger_d6_s43.pt' in text('f2_trunk_d6_s43.log')

with io.open(os.path.join(OUT, 'n_trunk_d6_s43_ep.csv'), encoding='utf-8') as fh:
    EP = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
with io.open(os.path.join(OUT, 'n_trunk_d6_s43.csv'), encoding='utf-8') as fh:
    RW = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
assert len(EP) == len(RW) == TRUNK_DEC // UPD == 138 and EP[-1]['step'] == RW[-1]['step'] == TRUNK_DEC
# 로그 한 줄 = update 5개 묶음인지: 끝난 에피소드 수가 ep.csv 5행 합과 같아야 함
for i, w in enumerate(N):
    assert sum(r['n_ep'] for r in EP[5 * i:5 * i + 5]) == w['ep'], (i, w['ep'])
TAIL = EP[5 * len(N):]      # 로그 창에 안 들어간 마지막 update(138 = 27 × 5 + 3)
assert len(TAIL) == 3
TAIL_N = sum(r['n_ep'] for r in TAIL)
TAIL_GOAL = sum(r['n_ep'] * r['goal'] for r in TAIL) / TAIL_N
TAIL_VCOLL = sum(r['n_ep'] * r['vColl'] for r in TAIL) / TAIL_N

# ── 관문 다시 계산(스크립트와 같은 식) → _gate.log 와 대조
LAST = N[-GATE_N:]
G_GOAL, G_OCOLL = mean(LAST, 'goal'), mean(LAST, 'oColl')
G_PASS = G_GOAL >= GATE_GOAL and G_OCOLL <= GATE_OCOLL
gate_txt = text('_gate.log')
m = re.search(rf'^s{SEED} goal=([\d.]+) oColl=([\d.]+) 창=(\d+) (통과|불통과)', gate_txt, re.M)
assert m and (m.group(1), m.group(2), m.group(3)) == (f'{G_GOAL:.1f}', f'{G_OCOLL:.1f}', str(GATE_N)), (m and m.groups(), G_GOAL, G_OCOLL)
assert (m.group(4) == '통과') == G_PASS and not G_PASS and '멈춤' in gate_txt
FLAST = F[-GATE_N:]

# ── 4창 이동 평균(관문 식을 모든 위치에 적용 — 기록용, 판정 아님)
ROLL = [(N[j]['dec'], N[j + GATE_N - 1]['dec'], mean(N[j:j + GATE_N], 'goal'), mean(N[j:j + GATE_N], 'oColl')) for j in range(len(N) - GATE_N + 1)]
ROLL_OK = [r for r in ROLL if r[2] >= GATE_GOAL and r[3] <= GATE_OCOLL]
ROLL_BEST = max(ROLL, key=lambda r: r[2])
N_GE30 = sum(1 for w in N if w['goal'] >= GATE_GOAL)
PEAK = max(N, key=lambda w: w['goal'])
FIRST30 = next(w for w in N if w['goal'] >= GATE_GOAL)
F_FIRST30 = next(w for w in F if w['goal'] >= GATE_GOAL)
RBEST = max(N[1:], key=lambda w: w['R'])       # 첫 창(전부 시간초과) 빼고 가장 높은 결정당 보상
OC_FIRST = next(w for w in N if w['oColl'] > 0)
ZERO_END = next(w for w in N if w['goal'] > 0)  # 도착이 처음 나온 창
ZC = [w['vColl'] for w in N[:N.index(ZERO_END)] if w['vColl'] > w['TO']]   # 도착 0 구간 가운데 배 충돌이 시간초과보다 많은 창
AFTER = [w['goal'] for w in N[N.index(PEAK) + 1:]]
BLK = [N[0:9], N[9:18], N[18:27]]              # 창 9개씩 3구간(구간을 결과 보고 고르지 않으려고 3등분)

# ── 경과(타임라인)
TL = [ln for ln in text('_timeline.txt').splitlines() if ln.startswith('2026-10-09')]


def t_of(pat):
    hit = [ln for ln in TL if pat in ln]
    assert len(hit) == 1, (pat, hit)
    return hit[0][11:16]


T_START, T_TRUNK_END, T_STOP = t_of('시작 trunk'), t_of('끝 trunk'), t_of('★멈춤 gate')
assert any('8d6bc88' in ln and '배치 n_ phase1' in ln for ln in TL) and any('건너뜀 smoke' in ln for ln in TL)
OVR = [ln[2:] for ln in text('_rank_override.txt').splitlines() if ln.startswith('> ')]
assert len(OVR) == 5 and '0/3 seeds' in OVR[2], OVR


# ── 그림(SVG, 로그 값 그대로)
def chart(series, y0, y1, yticks, unit, hline=None, shade=None, w=700, h=150):
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
    if hline is not None:
        s.append(f"<line x1='{ml}' x2='{w - mr}' y1='{Y(hline[0]):.1f}' y2='{Y(hline[0]):.1f}' stroke='#0b0b0b' stroke-width='0.7' stroke-dasharray='2 2'/>")
        s.append(f"<text x='{w - mr - 2}' y='{Y(hline[0]) - 2.5:.1f}' font-size='7.2' text-anchor='end' fill='#0b0b0b'>{hline[1]}</text>")
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
SHADE = (LAST[0]['dec'] - 0.33, LAST[-1]['dec'])
C_OUT = chart([('n_ 도착', pts(N, 'goal'), '#0b0b0b', 1.7, ''), ('n_ 배 충돌', pts(N, 'vColl'), '#a8322a', 1.1, ''),
               ('n_ 시간초과', pts(N, 'TO'), '#6f6e69', 1.0, '4 2'), ('n_ 벽 충돌', pts(N, 'oColl'), '#6f6e69', 1.0, '1 2'),
               ('f2_ 도착(흉내 있음, 대조)', pts(F, 'goal'), '#b9b7b0', 1.3, '')],
              0, 100, [0, 20, 40, 60, 80, 100], '%', hline=(GATE_GOAL, '관문 도착 30 %'), shade=SHADE)
C_REW = chart([('n_ 결정당 보상(창)', pts(N, 'R'), '#0b0b0b', 1.7, ''),
               ('n_ 결정당 보상(update, EMA)', [(r['step'] / 1e6, r['ema_reward']) for r in RW], '#6f6e69', 1.0, '4 2')],
              -1.4, 0.4, [-1.2, -0.8, -0.4, 0.0, 0.4], 'R', hline=(0.0, '0'), shade=SHADE, h=118)


def win_table():
    h = ("<table class='w'><thead><tr><th>결정(M)</th><th>도착 %</th><th>배 충돌 %</th><th>벽 충돌 %</th><th>시간초과 %</th>"
         "<th>결정당 보상 R</th><th>에피소드 수</th><th>평균 길이</th></tr></thead><tbody>")
    for i, w in enumerate(N):
        cls = " class='g'" if i >= len(N) - GATE_N else ''
        h += (f"<tr{cls}><td class='c'>{w['dec']:.2f}</td><td class='c'>{fm(w['goal'])}</td><td class='c'>{fm(w['vColl'])}</td><td class='c'>{fm(w['oColl'])}</td>"
              f"<td class='c'>{fm(w['TO'])}</td><td class='c'>{fm(w['R'], 3)}</td><td class='c'>{w['ep']:.0f}</td><td class='c'>{w['len']:.0f}</td></tr>")
    return h + "</tbody></table>"


def blk_table():
    h = ("<table><thead><tr><th>구간(창 9개씩)</th><th>도착 %</th><th>배 충돌 %</th><th>벽 충돌 %</th><th>시간초과 %</th><th>결정당 보상 R</th>"
         "<th>도착 최저–최고</th></tr></thead><tbody>")
    for b in BLK:
        h += (f"<tr><td class='c'>{b[0]['dec'] - 0.33:.2f}–{b[-1]['dec']:.2f}M</td><td class='c'>{fm(mean(b, 'goal'))}</td><td class='c'>{fm(mean(b, 'vColl'))}</td>"
              f"<td class='c'>{fm(mean(b, 'oColl'))}</td><td class='c'>{fm(mean(b, 'TO'))}</td><td class='c'>{fm(mean(b, 'R'), 3)}</td>"
              f"<td class='c'>{fm(min(w['goal'] for w in b))}–{fm(max(w['goal'] for w in b))}</td></tr>")
    return h + "</tbody></table>"


def cmp_table():
    h = ("<table><thead><tr><th>시드 43 trunk · 마지막 4창 평균</th><th>도착 %</th><th>배 충돌 %</th><th>벽 충돌 %</th><th>시간초과 %</th>"
         "<th>도착 30 % 처음 넘은 창</th><th>도착 최고(창)</th><th>관문</th></tr></thead><tbody>")
    for lab, rows, last, f30 in (('n_ (순수 PPO, 10-09)', N, LAST, FIRST30), ('f2_ (DAgger 시작 + 흉내 보조손실 1→0, 10-06)', F, FLAST, F_FIRST30)):
        pk = max(rows, key=lambda w: w['goal'])
        ok = mean(last, 'goal') >= GATE_GOAL and mean(last, 'oColl') <= GATE_OCOLL
        h += (f"<tr><td class='l2'>{lab}</td><td class='c'>{fm(mean(last, 'goal'))}</td><td class='c'>{fm(mean(last, 'vColl'))}</td>"
              f"<td class='c'>{fm(mean(last, 'oColl'))}</td><td class='c'>{fm(mean(last, 'TO'))}</td><td class='c'>{f30['dec']:.2f}M</td>"
              f"<td class='c'>{fm(pk['goal'])} ({pk['dec']:.2f}M)</td><td class='c'>{'통과' if ok else '<b>불통과</b>'}</td></tr>")
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
table.w { font-size: 7.2pt; }
th { text-align: center; font-weight: 600; padding: 2px 4px; border-bottom: 1px solid #0b0b0b; }
td { padding: 1.8px 4px; border-bottom: 1px solid #e4e3df; vertical-align: top; }
table.w td { padding: 0.4px 4px; }
td.c { text-align: center; } td.l2 { width: 31%; }
tr.g td { background: #efeee9; font-weight: 600; }
code { font-family: Consolas, monospace; font-size: 8pt; background: #f4f3ef; padding: 0 2px; }
.small { color: #52514e; font-size: 7.9pt; }
.fig { margin: 4px 0 0 0; }
.pb { page-break-before: always; }
"""
HTML = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{NAME}</title><style>{CSS}</style></head><body>
<h1>배치 n_ — trunk 관문 불통과 보고</h1>
<div class='sub1'>2026-10-09 · LICS 서버 · 코드 feat/fig1-latent 8d6bc88 · 순수 PPO(DAgger · 흉내 보조손실 없음) · 시드 {SEED} 하나 ·
128 env × 16척 · rollout 32 · 침로 행동 모드 · ARPA@56(메시지 없음) · 보상 v3 + R1 + R2 · {TRUNK_DEC:,} 결정</div>

<div class='box'><b>한 줄 결론</b> — trunk 관문 <b>불통과</b>: 마지막 4창 평균 도착 <b>{fm(G_GOAL)} %</b>(기준 ≥ {GATE_GOAL:.0f} %),
벽 충돌 {fm(G_OCOLL)} %(기준 ≤ {GATE_OCOLL:.0f} %, 충족) → 스펙 2026-10-07 §5-3 대로 멈춤. 코덱 · 갈래 off · comm · 평가 · 곡선 · 표는 안 돎.
도착률은 {PEAK['dec']:.2f}M 에서 {fm(PEAK['goal'])} % 까지 오른 뒤 {min(AFTER):.0f}–{max(AFTER):.0f} % 사이를 오갔고, 배끼리 충돌이 마지막 4창 {fm(mean(LAST, 'vColl'))} % 로 남음.
시드 1개 결과라 단독 주장 못 함. 다음은 저자 결정.</div>

<h2>1. 경과</h2>
<ul>
<li>10-07 17:03 순위 관문 v4 불통과로 멈춤(보고 <code>2026-10-07_n배치_순위관문v4_결과.pdf</code>) → 10-09 저자 결정으로 넘김(스펙 §7-1, 8d6bc88)</li>
<li>10-09 15:13 같은 접두어 n_ 로 다시 실행. smoke 는 건너뜀(10-07 통과분), 순위 관문 재실행 없음. <code>_rank_override.txt</code> 기록:
'{OVR[1]}' · 기록 관문 판정 줄 '{OVR[2]}'</li>
<li>{T_START}–{T_TRUNK_END} trunk 학습({fm(N_MIN)}분, GPU 1장, 평균 {TRUNK_DEC / (N_MIN * 60):.0f} 결정/초). 중단 · 재시도 · 무진행 kill 없음</li>
<li>{T_STOP} trunk 관문 → <b>불통과, 멈춤</b>. 드라이버 정상 종료, 남은 학습 프로세스 없음</li>
</ul>

<h2>2. 관문 판정 — 마지막 4창 평균(결과 전 고정, t_ 규칙)</h2>
<table><thead><tr><th>시드</th><th>도착 % (기준 ≥ {GATE_GOAL:.0f})</th><th>벽 충돌 % (기준 ≤ {GATE_OCOLL:.0f})</th><th>4창 도착 값</th><th>판정</th></tr></thead><tbody>
<tr><td class='c'>{SEED}</td><td class='c'><b>{fm(G_GOAL)}</b> (기준까지 {fm(G_GOAL - GATE_GOAL, 1, True)} %p)</td><td class='c'>{fm(G_OCOLL)}</td>
<td class='c'>{' · '.join(fm(w['goal']) for w in LAST)}</td><td class='c'><b>불통과</b> (0/1 시드)</td></tr></tbody></table>
<ul>
<li>창 = update 5개 = {5 * UPD:,} 결정 동안 끝난 에피소드의 결과 비율(학습 중 값 = 행동 잡음 포함, 고정 장면 평가 아님)</li>
<li>기록만 하는 G1(마지막 창 도착 ≥ 90 · 배 충돌 ≤ 5 · 벽 충돌 ≤ 2): 미충족({fm(N[-1]['goal'])} · {fm(N[-1]['vColl'])} · {fm(N[-1]['oColl'])})</li>
<li>로그 창에 안 들어간 마지막 update 3개(8.85–9.04M, 에피소드 {TAIL_N:.0f}개): 도착 {fm(TAIL_GOAL)} % · 배 충돌 {fm(TAIL_VCOLL)} % — 관문 계산 밖, 참고</li>
</ul>

<h2>3. trunk 학습 곡선(로그 값 그대로)</h2>
<div class='fig'>{C_OUT}</div>
<div class='fig'>{C_REW}</div>
<div class='small'>가로 = 결정 수. 회색 띠 = 관문이 읽는 마지막 4창. 위: 에피소드 결과 비율(%). 아래: 결정당 보상. f2_ 는 보상 정의가 달라(R1 · R2 없음) 아래 그림에 안 넣음.</div>
{blk_table()}
<ul>
<li>도착 0 % 가 {ZERO_END['dec'] - 0.33:.2f}M 까지 이어짐(처음엔 전부 시간초과, 그 뒤 배 충돌 {min(ZC):.0f}–{max(ZC):.0f} %). 도착 30 % 를 처음 넘은 창 = {FIRST30['dec']:.2f}M</li>
<li>도착 최고 = {fm(PEAK['goal'])} % ({PEAK['dec']:.2f}M). 그 뒤 창 {len(N) - N.index(PEAK) - 1}개 가운데 30 % 이상 = {sum(1 for w in N[N.index(PEAK) + 1:] if w['goal'] >= GATE_GOAL)}개. 전체 27창 가운데 30 % 이상 = {N_GE30}개</li>
<li>관문 식(4창 평균)을 모든 위치에 대 보면 {len(ROLL)}개 위치 가운데 {len(ROLL_OK)}개가 기준 충족, 최고 {fm(ROLL_BEST[2])} % ({ROLL_BEST[0] - 0.33:.2f}–{ROLL_BEST[1]:.2f}M).
끝 위치는 {fm(ROLL[-1][2])} %. 관문은 끝 4창으로 고정돼 있어 판정은 불통과 그대로임</li>
<li>시간초과는 100 % → {fm(mean(LAST, 'TO'))} % 로 줄었음. 실패의 대부분은 배끼리 충돌(마지막 4창 {fm(mean(LAST, 'vColl'))} %). 벽 충돌은 {OC_FIRST['dec']:.2f}M 부터 나타나 최고 {fm(max(w['oColl'] for w in N))} %</li>
<li>결정당 보상: 첫 창 {fm(N[0]['R'], 3, True)}(전부 시간초과) → 그 뒤 최고 {fm(RBEST['R'], 3)} ({RBEST['dec']:.2f}M) → 마지막 4창 평균 <b>{fm(mean(LAST, 'R'), 3)}</b>.
3구간 평균 {' → '.join(fm(mean(b, 'R'), 3) for b in BLK)} — trunk 구간에서 보상이 오르지 않음</li>
</ul>

<h2 class='pb'>4. 창별 값(27창, 굵은 줄 = 관문 4창)</h2>
{win_table()}

<h2>5. 대조 — 같은 시드의 f2_ trunk(흉내 있음)</h2>
{cmp_table()}
<ul>
<li>f2_ = DAgger 학생에서 시작 + 흉내 보조손실 1 → 0(3M 결정). n_ = 학습기 기본 초기화에서 시작, 선생님 없음. 둘 다 시드 43 하나, 학습 중 창 값</li>
<li>두 배치는 보상도 다름(n_ 은 R1 누적 주 상대 · R2 해소 종료 F6 을 켬) → 차이를 '흉내 유무' 하나로 못 돌림. 결정당 보상은 직접 비교 못 함</li>
</ul>

<h2>6. 한계 — 그대로 읽을 것</h2>
<ul>
<li>시드 1개(43). 스펙 §5-3: 1시드 결과는 단독 주장 금지. '순수 PPO 는 회피를 못 배운다'는 일반 결론의 근거로는 부족함</li>
<li>값은 학습 중 창 통계임. 고정 장면 평가(256 env · 평가 시드 999)는 관문 뒤 단계라 안 돌았음 → 평균 행동(잡음 없음) 성능은 모름</li>
<li>원인은 재지 않았음. 보상 항별 분해 · critic 설명 분산 · 조우 유형별 결과 어느 것도 이 배치에는 없음</li>
<li>이 배치는 순위 관문 v4 불통과를 저자 결정으로 넘기고 시작한 것임(기록 관문 0/3, 타각 모드 진단 2/3)</li>
<li>관문 기준 · 스크립트는 안 건드렸음. 추가 학습 · 평가도 안 돌렸음</li>
</ul>

<h2>7. 저자 결정 대기</h2>
<ul>
<li>n_ 은 trunk 관문에서 멈춘 상태. 스펙 §7 이 불통과 때 '다음 후보로 보고만' 하라고 적어 둔 것 = 수익 분산 대책(10-07 가설 A)</li>
<li>학습 없이 볼 수 있는 것(승인 필요): 남은 체크포인트 5개(2M · 4M · 6M · 8M · 9.04M)의 고정 장면 평가 — 학습 창 값과 평균 행동 값의 차, 3–4M 뒤 내려가는지 확인</li>
<li>시드 3개 확증(n3_)은 저자 지시가 '성공 뒤에만' 이라 해당 없음</li>
<li>결정 전까지 학습 · 평가는 안 돌림</li>
</ul>
<div class='small'>자료: runs/2026-10-02_fig1/out/ 의 n_trunk_d6_s43.log · _ep.csv · .csv · _gate.log · _timeline.txt · _rank_override.txt · f2_trunk_d6_s43.log ·
서버 Python/_repro_out_n/ · 체크포인트 VESSEL_checkpoints/comm_intent/n_trunk_d6_s43*.pt · 생성 report/make_report_trunk_n.py</div>
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
print('관문 goal', fm(G_GOAL), 'oColl', fm(G_OCOLL), '| 최고', fm(PEAK['goal']), PEAK['dec'], '| 30 이상 창', N_GE30, '/27 | 이동 4창 충족', len(ROLL_OK), '/', len(ROLL),
      '| 마지막 4창 R', fm(mean(LAST, 'R'), 3), '| f2_ 마지막 4창', fm(mean(FLAST, 'goal')), '| 꼬리 3 update', fm(TAIL_GOAL), fm(TAIL_VCOLL))
print(out_pdf)
