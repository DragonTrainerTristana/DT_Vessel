# 2026-10-07: 무통신(off) 보상이 학습할수록 떨어지는 이유 — 진단 결과 PDF. 평가만 돌림(학습 없음), 숫자는 전부 로그에서 읽음:
#   runs/2026-10-02_fig1/out/f2_{off,offb,comm}_s{43,44,45}.csv · _ep.csv                 (학습 곡선, 배치 f2_)
#   runs/2026-10-02_fig1/out/diag_offdrop/ck_{sample,mean}_off_s{43,44,45}[.step12M].txt  ([rdecomp] 줄, report/diag/offdrop/eval_any.py)
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_offdrop.py [출력 폴더]
import csv, io, os, re, subprocess, sys

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', 'out'))
DG = os.path.join(OUT, 'diag_offdrop')
NAME = '2026-10-07_무통신_보상하락_진단'
SEEDS = [43, 44, 45]
BC_END = 9043968 + 3000000
WINS = [('공통 뿌리 6–9.04M', 6e6, 9.043968e6), ('흉내 켜짐 11–12M', 11e6, 12.04e6), ('꺼진 직후 12–13M', 12.04e6, 13e6), ('끝 15–16M', 15e6, 16.1e6)]


def fm(v, nd=1, sign=False):
    s = f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'
    return s.replace('-', '−')


def curve(arm, seed):
    with io.open(os.path.join(OUT, f'f2_{arm}_s{seed}.csv'), encoding='utf-8') as fh:
        r = list(csv.DictReader(fh))
    with io.open(os.path.join(OUT, f'f2_{arm}_s{seed}_ep.csv'), encoding='utf-8') as fh:
        e = list(csv.DictReader(fh))
    return r, e


def win(rows, key, lo, hi):
    v = [float(x[key]) for x in rows if lo <= float(x['step']) < hi and x[key] not in ('', 'nan')]
    return sum(v) / len(v) if v else float('nan')


CUR = {(a, s): curve(a, s) for a in ('off', 'offb', 'comm') for s in SEEDS}


def crow(arm, label, key, which, nd):
    tds = ''
    for _, lo, hi in WINS:
        vs = [win(CUR[(arm, s)][which], key, lo, hi) for s in SEEDS]
        tds += f"<td class='m'>{fm(sum(vs) / 3, nd)}</td><td class='s'>{' · '.join(fm(v, nd) for v in vs)}</td>"
    return f"<tr><td class='l'>{label}</td>{tds}</tr>"


T1 = ("<table><thead><tr><th></th>" + ''.join(f"<th colspan='2'>{n}</th>" for n, _, _ in WINS) + "</tr><tr class='sub'><th></th>"
      + "<th>평균</th><th>시드 43 · 44 · 45</th>" * len(WINS) + "</tr></thead><tbody>"
      + crow('off', '무통신 off — 도착 %', 'goal', 1, 1) + crow('off', '무통신 off — 결정당 보상', 'raw_reward', 0, 3)
      + crow('offb', '무통신 다시 갈래 offb — 도착 %', 'goal', 1, 1) + crow('offb', '무통신 다시 갈래 offb — 결정당 보상', 'raw_reward', 0, 3)
      + crow('comm', '통신 comm — 도착 %', 'goal', 1, 1) + crow('comm', '통신 comm — 결정당 보상', 'raw_reward', 0, 3)
      + "</tbody></table>")


def ck(mode, seed, tag):
    p = os.path.join(DG, f"ck_{mode}_off_s{seed}{'.step12M' if tag == '12M' else ''}.txt")
    with io.open(p, encoding='utf-8', errors='replace') as fh:
        t = fh.read()
    d = {}
    m = re.search(r'goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%.*?headTravel=\s*([\d.]+)deg', t)
    d['goal'], d['vcoll'], d['head'] = [float(x) for x in m.groups()]
    m = re.search(r'1000결정당 보상 R=([+\-\d.]+) = 도착 ([+\-\d.]+) \+ 충돌 ([+\-\d.]+) \+ 시간초과 ([+\-\d.]+) \+ 판정벌점 ([+\-\d.]+) \+ 나머지\(shaping\) ([+\-\d.]+)', t)
    d['R'], d['arr'], d['col'], d['to'], d['jud'], d['shp'] = [float(x) for x in m.groups()]
    m = re.search(r'1000결정당 건수: 도착 ([\d.]+) · 충돌 ([\d.]+) · 시간초과 [\d.]+ · 판정 실패 ([\d.]+) \(좌현 F1 ([\d.]+)', t)
    d['n_arr'], d['n_col'], d['n_fail'], d['n_f1'] = [float(x) for x in m.groups()]
    m = re.search(r'\|침로 명령\| 평균 [\d.]+ \(×120° = ([\d.]+)°\)', t)
    d['cmd'] = float(m.group(1))
    return d


CK = {(m, s, t): ck(m, s, t) for m in ('sample', 'mean') for s in SEEDS for t in ('12M', '16M')}


def krow(label, mode, key, nd=1, sign=False, note=''):
    tds = ''
    for s in SEEDS:
        a, b = CK[(mode, s, '12M')][key], CK[(mode, s, '16M')][key]
        tds += f"<td class='c'>{fm(a, nd, sign)} → {fm(b, nd, sign)}</td>"
    return f"<tr><td class='l'>{label}</td>{tds}<td class='n'>{note}</td></tr>"


T2 = ("<table><thead><tr><th></th><th>시드 43</th><th>시드 44</th><th>시드 45</th><th></th></tr>"
      "<tr class='sub'><th></th><th colspan='3'>12M 체크포인트 → 16M(끝) 체크포인트</th><th></th></tr></thead><tbody>"
      + krow('도착 % (샘플 행동)', 'sample', 'goal', 1, False, '주 평가와 같은 방식')
      + krow('도착 % (평균 행동 = 잡음 없음)', 'mean', 'goal', 1, False, '정책 자체가 나빠졌는지')
      + krow('1000결정당 보상 합계', 'sample', 'R', 0, True)
      + krow('· 도착 보상 (+150/건)', 'sample', 'arr', 0, True)
      + krow('· 충돌 벌점 (−300/건)', 'sample', 'col', 0, True)
      + krow('· 역할 판정 벌점 (−20/건)', 'sample', 'jud', 0, True)
      + krow('· 나머지 (전진·시간·연료·근접·위험)', 'sample', 'shp', 0, True)
      + krow('1000결정당 충돌 건수', 'sample', 'n_col', 3)
      + krow('1000결정당 좌현 변침 벌점(F1) 건수', 'sample', 'n_f1', 3)
      + krow('침로 명령 크기 평균 ° (샘플)', 'sample', 'cmd', 1, False, '목표 방향에서 벗어난 각')
      + krow('침로 명령 크기 평균 ° (평균 행동)', 'mean', 'cmd', 1)
      + krow('뱃머리 돌린 총량 °/도착 (평균 행동)', 'mean', 'head', 0)
      + "</tbody></table>")

SIG = [('43', '0.137 → 0.141'), ('44', '0.136 → 0.147'), ('45', '0.145 → 0.160')]   # 체크포인트 action_logstd 평균의 exp (12M → 16M), 직접 열어 읽음
_NM = {'arr': '도착', 'col': '충돌', 'jud': '판정 벌점', 'shp': '나머지'}
_big = []
for _s in SEEDS:
    _d = {k: CK[('sample', _s, '16M')][k] - CK[('sample', _s, '12M')][k] for k in _NM}
    _k = min(_d, key=_d.get)
    _big.append(f"시드 {_s} {_NM[_k]}({fm(_d[_k], 0, True)})")
BIG = ' · '.join(_big)


def rr(mode, key, tag, nd=1):
    vs = [CK[(mode, s, tag)][key] for s in SEEDS]
    return ' · '.join(fm(v, nd) for v in vs)


def cw(arm, key, which, i, nd):
    vs = [win(CUR[(arm, s)][which], key, WINS[i][1], WINS[i][2]) for s in SEEDS]
    return ' · '.join(fm(v, nd) for v in vs)


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
table { border-collapse: collapse; width: 100%; margin: 4px 0 2px 0; font-size: 8.1pt; }
th { text-align: center; font-weight: 600; padding: 2px 4px; border-bottom: 1px solid #0b0b0b; }
tr.sub th { font-weight: 400; color: #52514e; font-size: 7.5pt; border-bottom: 1px solid #c9c8c3; }
td { padding: 1.8px 4px; border-bottom: 1px solid #e4e3df; vertical-align: top; }
td.l { width: 27%; } td.m { text-align: right; font-weight: 600; white-space: nowrap; }
td.s { text-align: center; color: #52514e; white-space: nowrap; font-size: 7.5pt; } td.c { text-align: center; white-space: nowrap; }
td.n { color: #52514e; font-size: 7.6pt; }
code { font-family: Consolas, monospace; font-size: 8pt; background: #f4f3ef; padding: 0 2px; }
.small { color: #52514e; font-size: 7.9pt; }
"""
HTML = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{NAME}</title><style>{CSS}</style></head><body>
<h1>무통신(off) 보상이 학습할수록 떨어지는 이유 — 진단</h1>
<div class='sub1'>2026-10-07 · 배치 f2_ 학습 곡선 + 무통신 체크포인트(12M · 16M) 평가 12건 · 학습 없음 · 코드 feat/fig1-latent 74b5988</div>

<div class='box'><b>한 줄 결론</b> — '학습할수록'이 아니라 <b>선생님 흉내 손실이 꺼진 뒤(12.04M)</b>부터 떨어짐. 흉내가 켜진 동안은 오르고, 꺼지면 PPO 혼자서는 정책을 지키지 못함.
한 항을 얻으려고 다른 항을 버린 것이 아니라 도착 · 충돌 · 판정 벌점 · 나머지가 전부 나빠짐(3/3 시드). 통신 팔도 같은 시점부터 내려가나 폭이 작음.</div>

<h2>1. 학습 곡선 — 언제 떨어지는가</h2>
<ul>
<li>흉내 보조손실은 갈래 시작(9.04M)에서 1 로 켜져 3M 결정에 걸쳐 0 이 됨 → 12.04M 부터 꺼짐</li>
<li>무통신 도착률: 흉내 켜진 끝 무렵 {cw('off', 'goal', 1, 1, 1)} % → 끝 {cw('off', 'goal', 1, 3, 1)} %. 시드 44 · 45 가 무너지고 43 은 버팀</li>
<li>같은 뿌리에서 다시 갈라 본 무통신(offb)은 시드 44 만 무너짐({cw('offb', 'goal', 1, 3, 1)} %) → 무너지는지 여부가 시드 · 갈래마다 다름(운에 가까움)</li>
<li>통신도 12M 뒤로 내려감: 결정당 보상 {cw('comm', 'raw_reward', 0, 2, 3)} → {cw('comm', 'raw_reward', 0, 3, 3)}. 3/3 시드가 내려가나 폭이 작음</li>
<li>공통 뿌리(흉내가 3M 에 꺼진 뒤 9.04M 까지)는 평평했음. 흉내 없이 꾸준히 오른 구간은 어느 팔에도 없음
(통신 시드 43 · 44 는 꺼진 직후 1M 동안 조금 더 오른 뒤 내려감)</li>
</ul>
{T1}
<div class='small'>학습 중 기록(샘플 행동)의 구간 평균. 괄호 없는 작은 글씨 = 시드 43 · 44 · 45.</div>

<h2>2. 12M 과 16M 체크포인트를 같은 조건으로 평가 — 무엇이 나빠졌는가</h2>
{T2}
<ul>
<li>보상의 네 항이 전부 나빠짐(3/3 시드). 가장 크게 준 항 = {BIG}. 1000결정당 충돌 {rr('sample', 'n_col', '12M', 3)} → {rr('sample', 'n_col', '16M', 3)}</li>
<li>잡음이 커진 탓이 아님: 행동 잡음 σ(침로) 는 {' · '.join(v for _, v in SIG)} 로 거의 그대로. 잡음을 뺀 평균 행동으로 몰아도 도착 {rr('mean', 'goal', '12M')} → {rr('mean', 'goal', '16M')} % → 정책 자체가 나빠짐</li>
<li>덜 돌아서가 아님: 뱃머리 돌린 총량이 3/3 시드에서 늘었음(평균 행동 {rr('mean', 'head', '12M', 0)} → {rr('mean', 'head', '16M', 0)}°).
침로 명령 크기는 샘플 행동 3/3 증가, 평균 행동 2/3 증가(시드 44 는 줄었음). 좌현 변침 벌점(F1)도 3/3 늘어남 → 더 돌면서 더 부딪힘</li>
<li>'나머지' 항 안에서 어느 것(전진·연료·근접·위험)이 줄었는지는 따로 나누지 않았음</li>
</ul>
<div class='small'>평가 = 256 env · burn-in 2400 · 평가 창 2000결정 · 평가 시드 999. 창이 짧아 도착률 값 자체는 주 평가(10000결정)보다 낮게 나옴 → 12M 과 16M 의 비교로만 읽을 것.</div>

<h2>3. 확인된 것 · 아닌 것</h2>
<ul>
<li>보상 순위가 뒤집힌 것은 아님: 같은 보상으로 잰 에피소드 보상이 무통신 선생님 649 &gt; 안 피하는 배 260 &gt; 학습 무통신 3 (10-06 주 평가). 좋은 행동이 보상도 높음</li>
<li>정보가 모자란 것도 아님: 같은 56 m 정보만 쓰는 선생님이 도착 90.6 % · 충돌 8.6 %</li>
<li>따라서 남는 것은 '학습이 보상을 못 따라감'임. 흉내가 꺼진 뒤 PPO 갱신만으로는 정책이 유지되지 않음
(무통신 갈래 6개 중 3개가 끝 구간 도착 60 % 아래로 무너짐: off 44 · 45, offb 44)</li>
</ul>

<h2>4. 왜 PPO 가 못 따라가는가 (가설 — 아직 측정 안 함)</h2>
<ul>
<li>가설 A: 수익의 분산이 큼. 가치 정규화 통계가 평균 +1.4 · 표준편차 129.8(시드 44 공통 뿌리) → 평균 −11.0 · 표준편차 140.6(시드 44 끝).
충돌 −300 · 도착 +150 · 판정 −20 이 드물게 크게 들어와 학습 신호가 잡음에 묻힘</li>
<li>가설 B: 무통신 배가 볼 수 없는 것에 벌점이 붙음. 역할 판정과 위험 비용은 300 m 에서 시작하는데 무통신 배는 56 m 만 봄
→ 관측으로 설명 안 되는 보상. 통신 팔이 덜 내려가는 것과 맞음</li>
<li>둘 다 '학습 신호가 약하다'는 같은 방향이고, 어느 쪽이 큰지는 모름</li>
</ul>

<h2>5. 설계 관점</h2>
<ul>
<li>현재 설계에서 무통신 보상 곡선의 모양 = 평평(공통 뿌리) → 상승(흉내 켜짐) → 하락(흉내 꺼짐). 우상향 곡선이 아님</li>
<li>오르는 구간이 흉내가 켜진 구간과 겹침. 흉내 몫과 PPO 몫을 나눠 재지는 않았음. 통신 팔의 상승도 같은 구간임(10-06 보고와 같음)</li>
<li>학습 없이 더 확인할 수 있는 것(약 30분, 평가만): 12M 체크포인트에서 critic 이 수익을 얼마나 설명하는지(설명 분산)를 무통신 · 통신으로 재어 가설 A · B 를 가름</li>
</ul>
<div class='small'>자료: runs/2026-10-02_fig1/out/diag_offdrop/ (로그 12개) · 스크립트 runs/2026-10-02_fig1/report/diag/offdrop/ · 이 문서 report/make_report_offdrop.py</div>
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
for i, (n, _, _) in enumerate(WINS):
    print(n, '| off 도착', cw('off', 'goal', 1, i, 1), '| offb', cw('offb', 'goal', 1, i, 1), '| comm', cw('comm', 'goal', 1, i, 1),
          '| comm R', cw('comm', 'raw_reward', 0, i, 3))
print(out_pdf)
