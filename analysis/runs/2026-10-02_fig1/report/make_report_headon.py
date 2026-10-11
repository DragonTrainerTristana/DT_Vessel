# 2026-10-07: 정면 조우(head-on)를 왜 안 지키는가 — 진단 결과 PDF. 평가만 돌림(학습 없음), 숫자는 전부 로그에서 읽음:
#   runs/2026-10-02_fig1/out/diag_headon/diag_learned_{off,comm}_s{43,44,45}.txt   ([headon-diag] 줄, report/diag/headon/headon_diag.py)
#   runs/2026-10-02_fig1/out/diag_headon/judge_learned_{off,comm}_s{43,44,45}.txt  ([judge-diag] 줄, report/diag/headon/headon_judge.py)
#   runs/2026-10-01_scripted/out/sc_{goal,vo56h150,vo300i,col300ih}_e{999,1000,1001}.txt  (규칙 배 참고, 10-02 4차)
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_headon.py [출력 폴더]
import io, os, re, subprocess, sys

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.normpath(os.path.join(HERE, '..'))
DG = os.path.join(F1, 'out', 'diag_headon')
SC = os.path.normpath(os.path.join(F1, '..', '2026-10-01_scripted', 'out'))
NAME = '2026-10-07_정면조우_원인진단'
SEEDS = [43, 44, 45]
ARMS = [('off', '무통신(off)'), ('comm', '통신(comm)')]


def lines(path, tag):
    with io.open(path, encoding='utf-8', errors='replace') as fh:
        return [l.split(tag, 1)[1].strip() for l in fh if tag in l]


def num(pat, text):
    m = re.search(pat, text)
    if not m:
        raise SystemExit(f'못 찾음: {pat!r} in {text[:120]!r}')
    return float(m.group(1))


def find(ls, key):
    for l in ls:
        if key in l:
            return l
    raise SystemExit(f'줄 없음: {key}')


def load(arm, seed):
    d = {}
    p = os.path.join(DG, f'diag_learned_{arm}_s{seed}.txt')
    ls = lines(p, '[headon-diag]')
    with io.open(p, encoding='utf-8', errors='replace') as fh:
        w = re.search(r'headon=\s*([\d.]+)%\(n=(\d+) S=([\d.]+) D=([\d.]+) A=([\d.]+) R8=([\d.]+) T=([\d.]+) P=([\d.]+)', fh.read())
    d['score'], d['n'], d['S'], d['D'], d['A'], d['R8'], d['T'], d['P'] = [float(x) for x in w.groups()]
    l = find(ls, '통과 방식:')
    d['pp'], d['ss'] = num(r'좌현대좌현=([\d.]+)%', l), num(r'우현대우현=([\d.]+)%', l)
    d['mx'] = num(r'엇갈림\(한쪽이 앞을 가로지름\)=([\d.]+)%', l)
    d['P_pp'], d['P_ss'] = num(r'P 평균: 좌현대좌현 ([\d.]+)', l), num(r'· 우현대우현 ([\d.]+)', l)
    l = find(ls, '변침 방향:')
    d['ts'], d['tp'] = num(r'주로 우현=([\d.]+)%', l), num(r'주로 좌현=([\d.]+)%', l)
    l = find(ls, '우현 최대 변침 분포:')
    d['port10'] = num(r'좌현 최대:.*≥10° ([\d.]+)%', l)
    for key, nm in (('bs', "'둘 다 우현'"), ('bp', "'둘 다 좌현'"), ('sp', "'한 척 우현·한 척 좌현'")):
        l = find(ls, '두 배 조합 ' + nm)
        d[key], d[key + '_pp'], d[key + '_P'] = num(nm + r' ([\d.]+)%', l), num(r'좌현대좌현 ([\d.]+)%', l), num(r'· P ([\d.]+)', l)
    d['i_pp'] = num(r'좌현대좌현 ([\d.]+)%', find(ls, "시작 '서로 좌현 앞'"))
    d['i_ss'] = num(r'좌현대좌현 ([\d.]+)%', find(ls, "시작 '서로 우현 앞'"))
    d['in56'] = num(r'56 m 안까지 들어온 조우 ([\d.]+)% \(n=', find(ls, '56 m 안까지 들어온 조우 '))
    l = find(ls, '56 m 안까지 들어온 조우:')
    d['in56_P'] = num(r'들어온 조우: 좌현대좌현 [\d.]+% · P ([\d.]+)', l)
    d['out56_P'] = num(r'안 들어온 조우: 좌현대좌현 [\d.]+% · P ([\d.]+)', l)
    d['out56_cpa'] = num(r'최근접 평균 ([\d.]+)m', l)
    d['cpa'] = num(r'최근접 평균=([\d.]+)m', find(ls, '시작거리 평균='))
    js = lines(os.path.join(DG, f'judge_learned_{arm}_s{seed}.txt'), '[judge-diag]')
    i0 = next(i for i, l in enumerate(js) if l.startswith('정면 조우 쌍:'))
    i1 = next(i for i, l in enumerate(js) if l.startswith('교차 쌍'))
    ho, cr = js[i0:i1], js[i1:]
    d['j_n'], d['j_ok'] = num(r'판정 n=(\d+)', ho[0]), num(r'성공\(벌점 없음\) ([\d.]+)%', ho[0])
    d['j_res'] = num(r'\) ([\d.]+)% · 늦은 시작', find(ho, '끝난 방식:'))
    l = find(ho, '실패 유형')
    d['j_F1'], d['j_F4'], d['j_F6'] = num(r'F1 ([\d.]+)%', l), num(r'F4 ([\d.]+)%', l), num(r'F6 ([\d.]+)%', l)
    l = find(ho, '성공 처리된 쌍')
    d['j_ok_both10'], d['j_ok_none10'] = num(r'둘 다 우현 ≥10° ([\d.]+)%', l), num(r'둘 다 <10° ([\d.]+)%', l)
    d['j_ok_res'] = num(r'해소로 끝남 ([\d.]+)%', l)
    l = find(ho, '해소로 끝난 쌍')
    d['j_res_none10'], d['j_res_none2'], d['j_res_len'] = num(r'둘 다 <10° ([\d.]+)%', l), num(r'둘 다 <2° ([\d.]+)%', l), num(r'길이 중앙 ([\d.]+)s', l)
    d['j_mind'] = num(r'최소거리 중앙 ([\d.]+)m', ho[0])
    d['c_ok'], d['c_res'] = num(r'성공\(벌점 없음\) ([\d.]+)%', cr[0]), num(r'\) ([\d.]+)% · 늦은 시작', find(cr, '끝난 방식:'))
    d['c_F6'] = num(r'F6 ([\d.]+)%', find(cr, '실패 유형'))
    return d


R = {a: [load(a, s) for s in SEEDS] for a, _ in ARMS}


def ref(pol):
    out = []
    for e in (999, 1000, 1001):
        p = os.path.join(SC, f'sc_{pol}_e{e}.txt')
        if not os.path.exists(p):
            continue
        with io.open(p, encoding='utf-8', errors='replace') as fh:
            m = re.search(r'headon=\s*([\d.]+)%\(n=\d+ S=([\d.]+) D=[\d.]+ A=[\d.]+ R8=[\d.]+ T=([\d.]+) P=([\d.]+)', fh.read())
        if m:
            out.append([float(x) for x in m.groups()])
    return ([sum(c) / len(c) for c in zip(*out)] if out else None), len(out)


REF = {p: ref(p) for p in ('goal', 'vo56h150', 'vo300i', 'col300ih')}


def fm(v, nd=1):
    return f'{v:.{nd}f}'


def avg(arm, key):
    vs = [r[key] for r in R[arm]]
    return sum(vs) / len(vs)


def rng(arm, key, nd=1):
    vs = [r[key] for r in R[arm]]
    return f'{fm(min(vs), nd)}–{fm(max(vs), nd)}'


def rng2(key, nd=0):
    vs = [r[key] for a, _ in ARMS for r in R[a]]
    return f'{fm(min(vs), nd)}–{fm(max(vs), nd)}'


def cell(arm, key, nd=1):
    vs = [r[key] for r in R[arm]]
    return f"<td class='m'>{fm(sum(vs) / len(vs), nd)}</td><td class='s'>{' · '.join(fm(v, nd) for v in vs)}</td>"


def row(label, key, nd=1, note=''):
    return f"<tr><td class='l'>{label}</td>{cell('off', key, nd)}{cell('comm', key, nd)}<td class='n'>{note}</td></tr>"


def table(rows):
    return ("<table><thead><tr><th></th><th colspan='2'>무통신 (off)</th><th colspan='2'>통신 (comm)</th><th></th></tr>"
            "<tr class='sub'><th></th><th>평균</th><th>시드 43 · 44 · 45</th><th>평균</th><th>시드 43 · 44 · 45</th><th></th></tr></thead><tbody>"
            + ''.join(rows) + "</tbody></table>")


T1 = table([
    row('정면 조우 점수 (Woerner) %', 'score', 1, '주 평가 12.8 → 27.3 과 맞음(검산)'),
    row('· 통과 쪽 항 P', 'P', 1, '점수를 가장 크게 깎는 항'),
    row('· 주로 좌현 변침 아님 항 T', 'T', 1),
    row('· 안전 거리 항 S', 'S', 1, '통신이 오른 몫의 대부분'),
    row('좌현대좌현으로 통과 %', 'pp', 1, '규칙이 원하는 통과'),
    row('우현대우현으로 통과 %', 'ss', 1, 'P ≈ 0 으로 채점됨'),
    row('한쪽이 앞을 가로지름 %', 'mx', 1),
    row('주로 우현으로 돈 배 %', 'ts', 1, '조우 시작 침로 기준, 2° 이상'),
    row('주로 좌현으로 돈 배 %', 'tp', 1),
    row('좌현으로 10° 이상 돈 적 있음 %', 'port10', 1),
    row('두 배 다 우현으로 돎 %', 'bs', 1, '규칙대로 한 조우'),
    row('· 그때 좌현대좌현 통과 %', 'bs_pp', 1, '규칙대로 하면 점수가 나옴'),
    row('두 배 다 좌현으로 돎 %', 'bp', 1, '거울상 약속'),
    row('한 척 우현 · 한 척 좌현 %', 'sp', 1),
    row("시작이 '서로 좌현 앞'일 때 좌현대좌현 %", 'i_pp', 1),
    row("시작이 '서로 우현 앞'일 때 좌현대좌현 %", 'i_ss', 1, '규칙대로면 여기도 높아야 함'),
    row('56 m 안까지 들어온 조우 %', 'in56', 1, '무통신 배가 상대를 볼 수 있던 조우'),
    row('· 안 들어온 조우의 P', 'out56_P', 1),
    row('· 안 들어온 조우의 최근접 거리 m', 'out56_cpa', 0),
])
T2 = table([
    row('정면 쌍 중 벌점 없이 끝남 %', 'j_ok', 1),
    row("· 그중 '해소'로 끝남 %", 'j_ok_res', 1, 'dcpa ≥ 30 m 가 2초(5결정) 이어지면 성공 종료'),
    row('· 그중 두 배 다 우현 10° 미만 %', 'j_ok_none10', 1, '우현 10° 변침 없이 성공 처리'),
    row('· 그중 두 배 다 우현 10° 이상 %', 'j_ok_both10', 1),
    row('해소된 쌍: 두 배 다 우현 2° 미만 %', 'j_res_none2', 1, '우현 변침 기록 없이 해소'),
    row('해소된 쌍: 길이 중앙값 s', 'j_res_len', 0, '조우 시작 뒤 해소까지'),
    row('정면 쌍 최소 거리 중앙값 m', 'j_mind', 0, '판정이 닫힐 때까지'),
    row('벌점: 좌현 5° 초과 변침 (F1) %', 'j_F1', 1, '실제로 걸린 유일한 침로 벌점'),
    row('벌점: 24 m 안 접근 (F4) %', 'j_F4', 1),
    row('벌점: 우현 10° 변침 없음 (F6) %', 'j_F6', 1, '사실상 한 번도 안 걸림'),
    row('참고) 교차 쌍 중 해소로 끝남 %', 'c_res', 1),
    row('참고) 교차 쌍 F6 %', 'c_F6', 1),
])

refs = []
for pol, nm in (('goal', '안 피하는 배 (goal)'), ('vo56h150', 'VO 56 m = 무통신 선생님'), ('vo300i', 'VO 300 m + 의도 = 통신 선생님'),
                ('col300ih', 'COLREGs 규칙 배 (우현만)')):
    v, k = REF[pol]
    if v:
        refs.append(f"<tr><td class='l'>{nm}</td><td class='m'>{fm(v[0])}</td><td class='m'>{fm(v[3])}</td><td class='m'>{fm(v[2])}</td>"
                    f"<td class='m'>{fm(v[1])}</td><td class='n'>평가 시드 {k}개 평균 (10-02 4차 로그)</td></tr>")
for a, nm in ARMS:
    refs.append(f"<tr><td class='l'>학습 {nm}</td><td class='m'>{fm(avg(a, 'score'))}</td><td class='m'>{fm(avg(a, 'P'))}</td>"
                f"<td class='m'>{fm(avg(a, 'T'))}</td><td class='m'>{fm(avg(a, 'S'))}</td><td class='n'>학습 시드 3개 평균 (이번 진단)</td></tr>")
T3 = ("<table><thead><tr class='sub'><th></th><th>정면 점수 %</th><th>P</th><th>T</th><th>S</th><th></th></tr></thead><tbody>"
      + ''.join(refs) + "</tbody></table>")
COL = REF['col300ih'][0]
COL_P, COL_S = (fm(COL[3]), fm(COL[0])) if COL else ('n/a', 'n/a')
P_PP = sorted([avg('off', 'P_pp'), avg('comm', 'P_pp')])

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
table { border-collapse: collapse; width: 100%; margin: 4px 0 2px 0; font-size: 8.3pt; }
th { text-align: center; font-weight: 600; padding: 2px 4px; border-bottom: 1px solid #0b0b0b; }
tr.sub th { font-weight: 400; color: #52514e; font-size: 7.6pt; border-bottom: 1px solid #c9c8c3; }
td { padding: 1.8px 4px; border-bottom: 1px solid #e4e3df; vertical-align: top; }
td.l { width: 31%; } td.m { text-align: right; font-weight: 600; width: 6.5%; white-space: nowrap; }
td.s { text-align: center; color: #52514e; width: 15%; white-space: nowrap; font-size: 7.8pt; } td.n { color: #52514e; font-size: 7.6pt; }
code { font-family: Consolas, monospace; font-size: 8pt; background: #f4f3ef; padding: 0 2px; }
.small { color: #52514e; font-size: 7.9pt; }
.pb { page-break-before: always; }
"""
HTML = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{NAME}</title><style>{CSS}</style></head><body>
<h1>정면 조우(head-on)를 왜 안 지키는가 — 원인 진단</h1>
<div class='sub1'>2026-10-07 · 배치 f2_ 의 끝난 정책 6개(무통신·통신 × 시드 43·44·45)를 평가만 돌림 · 학습 없음 · 코드 feat/fig1-latent 74b5988</div>

<div class='box'><b>한 줄 결론</b> — 학습 배는 '정면이면 우현으로 돈다'를 배우지 않았음. 도는 방향이 배마다 반반이라 좌현대좌현 통과가
무통신 {fm(avg('off', 'pp'))} % · 통신 {fm(avg('comm', 'pp'))} % 에 그침. 학습 보상이 그 행동을 사실상 요구하지 않았고(측정함),
흉내 낸 선생님도 좌우를 가리지 않음(코드 확인).</div>

<h2>1. 배가 실제로 한 행동</h2>
<ul>
<li>정면 점수를 깎는 것은 통과 쪽 항 P 임. 좌현대좌현 통과면 P ≈ {fm(P_PP[0], 0)}–{fm(P_PP[1], 0)}, 우현대우현이면 P ≈ 0 → P 는 거의 '좌현대좌현 통과 비율'</li>
<li>도는 방향에 규칙이 없음: 주로 우현 {rng('off', 'ts', 0)} % (무통신) · {rng('comm', 'ts', 0)} % (통신), 주로 좌현 {rng('off', 'tp', 0)} % · {rng('comm', 'tp', 0)} %.
좌현으로 10° 이상 돈 조우가 두 팔 모두 {rng2('port10')} %</li>
<li>두 배가 다 우현으로 돈 조우에서는 좌현대좌현 {rng('off', 'bs_pp', 0)} % (무통신) · {rng('comm', 'bs_pp', 0)} % (통신) → 규칙대로 하면 점수가 나옴.
그런 조우가 무통신 {rng('off', 'bs', 0)} % · 통신 {rng('comm', 'bs', 0)} % 뿐</li>
<li>시드마다 버릇이 다름: 무통신 시드 45 는 '두 배 다 좌현'이 {fm(R['off'][2]['bp'])} % (거울상 약속 — 서로는 피하지만 채점은 0 점),
시드 44 는 '두 배 다 우현'이 {fm(R['off'][1]['bs'])} %</li>
<li>무통신은 시작 배치를 따라감(서로 우현 앞에서 시작 → 좌현대좌현 {rng('off', 'i_ss', 0)} %). 통신은 시작 배치와 무관하게 약 절반({rng('comm', 'i_ss', 0)} %)
— 멀리서 미리 움직이기는 하나 방향이 우현으로 모이지 않음</li>
</ul>
{T1}
<div class='small'>Woerner 채점과 같은 조우(300 m 안 · 접근 중 · dcpa &lt; 24 m · 역할 = 정면)만 모음. 시드당 정면 조우 {rng2('n')}건.
변침 = 조우 시작 침로에서 벗어난 최대 각.</div>

<h2 class='pb'>2. 원인 1 — 학습 보상이 정면 규칙을 사실상 요구하지 않음 (측정함)</h2>
<ul>
<li>학습 보상의 역할 약속 판정기 v2 가 정면 쌍을 어떻게 닫았는지 셈. 벌점 없이 끝난 정면 쌍의 {rng('off', 'j_ok_res')} % (무통신) · {rng('comm', 'j_ok_res')} % (통신) 가 '해소'로 끝남</li>
<li>해소 = dcpa 가 30 m 이상인 상태가 2초 이어지면 성공으로 닫음. 어느 쪽으로 벌어졌는지 · 우현으로 돌았는지 안 봄 (<code>vessel_gym.py:705, 727</code>)</li>
<li>조우는 300 m 에서 시작하는데 그 거리에서는 침로 몇 도만 달라져도 dcpa 가 30 m 를 넘음(같은 속력 정면이면 한 배 1° ≈ 2.6 m, 계산값).
해소된 쌍은 시작 뒤 {rng('off', 'j_res_len', 0)} 초(무통신) · {rng('comm', 'j_res_len', 0)} 초(통신) 만에 닫힘</li>
<li>해소된 쌍의 {rng('off', 'j_res_none2', 0)} % (무통신) · {rng('comm', 'j_res_none2', 0)} % (통신) 는 두 배 다 우현 변침 기록이 2° 미만
(판정기는 그 상대가 '주 상대'인 결정만 기록하고, 좌현 5° 이내는 허용함)</li>
<li>'우현 10° 이상 안 돌면 벌점'(F6)은 정상 종료에서만 봄 (<code>vessel_gym.py:728–729</code>) → 걸린 비율 {rng('off', 'j_F6')} % · {rng('comm', 'j_F6')} %. 조문은 있으나 작동 안 함</li>
<li>실제로 걸린 침로 벌점은 '좌현 5° 초과'(F1)뿐: {rng('off', 'j_F1', 0)} % · {rng('comm', 'j_F1', 0)} %. 통과 쪽(좌현대좌현)은 보상 어디에도 없음</li>
<li>보상이 가르친 것 = 'dcpa 30 m 를 벌려라, 좌현으로는 돌지 마라'. 채점이 보는 것 = '우현으로 돌아 좌현대좌현으로 지나가라'. 둘이 다름</li>
</ul>
{T2}

<h2>3. 원인 2 — 흉내 낸 선생님이 좌우를 가리지 않음 (코드 확인)</h2>
<ul>
<li>선생님(VO)은 후보 침로 −120°…+120° 를 좌우 똑같이 놓고 비용이 가장 작은 것을 고름 (<code>imitation/vo_teacher.py:43, 95</code>). 우현 우선 없음</li>
<li>그래서 선생님 자신의 P 도 학습 배와 같은 수준(아래 표). 흉내 손실(DAgger + 학습 중 보조손실)은 '가까운 쪽으로 피하기'를 끌어당김</li>
</ul>
{T3}
<div class='small'>규칙 배 줄은 같은 채점으로 잰 10-02 4차 결과이고, 비교 축이 아니라 원인 확인용 참고임.</div>

<h2>4. 원인 3 — 채점 범위와 배가 보는 범위가 다름 (측정함)</h2>
<ul>
<li>채점은 300 m 에서 조우를 시작함 (<code>colregs_woerner.py:119</code>). 무통신 배는 56 m 안만 봄. 정면 조우 중 56 m 안까지 들어온 것은 무통신 {rng('off', 'in56', 0)} %</li>
<li>나머지는 평균 {rng('off', 'out56_cpa', 0)} m 떨어져 지나갔는데도 통과 쪽으로 채점됨(P {rng('off', 'out56_P', 0)}). 무통신 배가 볼 수 없던 조우임</li>
<li>정면 점수 = 여섯 항의 곱(S·D·A·T·P·R8, <code>colregs_woerner.py:179</code>)이라 하나만 낮아도 전체가 낮음.
16척이 동시에 만나는 장면에서는 우현으로만 피하게 짠 규칙 배도 P {COL_P} · 정면 점수 {COL_S} %</li>
</ul>

<h2>5. 확신도 · 한계</h2>
<ul>
<li>'보상이 요구하지 않았다' · '선생님이 좌우 중립이다' = 측정 · 코드로 확인한 사실</li>
<li>'그래서 점수가 낮다' = 가설(확신도 중상). 보상 · 선생님을 고쳐 다시 학습해 봐야 확인됨. 이번에는 학습 안 함</li>
<li>변침 각은 조우 전체 동안의 최대 편차라 목표 쪽 조타 · 다른 배 회피 · 흔들림이 섞여 있음. 조우의 97–99 % 에서 2° 편차가 중앙값 240–250 m 거리에서 이미 생김
(무통신 배가 상대를 보기 전)</li>
<li>진단 평가 창 = 3000결정(주 평가 10000결정보다 짧음), 256 env · burn-in 2400 · 평가 시드 999 · 샘플 행동(주 평가와 같음).
정면 점수 · P 가 주 평가 값과 1.6 안에서 맞음</li>
</ul>

<h2>6. 고치려면 (제안 — 저자 결정 필요)</h2>
<ul>
<li>학습 없이 행동을 바꿀 방법은 없음. 끝난 정책의 행동은 그대로이고, 채점식을 바꿔 점수를 올리는 일은 하지 않음</li>
<li>학습 없이 더 확인할 수 있는 것: 평균 행동 평가(약 40분)로 '흔들림' 몫이 얼마인지 보기. 평가 방식이 바뀌므로 주 지표로 쓰려면 승인 필요</li>
<li>추천 1개(재학습 필요): 판정기 v2 에서 정면 쌍은 '해소'로 닫을 때도 우현 10° 조건을 보게 하고, 선생님을 우현 우선 규칙 배 계열로 바꿈.
보상 · 판정 정의가 바뀌므로 승인 전에는 손대지 않음</li>
</ul>
<div class='small'>자료: runs/2026-10-02_fig1/out/diag_headon/ (로그 12개) · 스크립트 runs/2026-10-02_fig1/report/diag/headon/ · 이 문서 report/make_report_headon.py</div>
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
KEYS = ('score', 'P', 'T', 'S', 'pp', 'ss', 'mx', 'ts', 'tp', 'port10', 'bs', 'bs_pp', 'bp', 'sp', 'i_pp', 'i_ss', 'in56', 'out56_P',
        'j_ok', 'j_ok_res', 'j_ok_none10', 'j_res_none2', 'j_res_len', 'j_F1', 'j_F4', 'j_F6')
for a, nm in ARMS:
    print(nm, {k: round(avg(a, k), 1) for k in KEYS})
print('refs', {p: ([round(x, 1) for x in v[0]] if v[0] else None, v[1]) for p, v in REF.items()})
print(out_pdf)
