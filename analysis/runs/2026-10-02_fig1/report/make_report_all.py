# 2026-10-03: 10-02 ~ 10-03 저자 지시별 결과 종합 + 재분석(평균 행동 주 평가) PDF.
# 숫자는 전부 로그에서 읽음:
#   runs/2026-10-01_scripted/out/sc_*_e*.txt · lr_t_*_s*.txt          규칙 배 4차 (summarize.parse_all)
#   runs/2026-10-02_fig1/out/eval_f2_dagger_d6_s*.txt · _p0.md         f2_ P0 (report_parse.parse_eval)
#   runs/2026-10-02_imitation/out/eval_i_dagger_d6_s*.txt              i_ P0 (10-02 새벽)
#   runs/2026-10-02_fig1/out/diag_meaneval/meaneval_*_s*.txt           재분석: 평균 행동 주 평가 (report/diag/eval_mean.py)
#   runs/2026-10-02_fig1/out/diag_p0/*.log · diag_vram/*.log           라벨 분포·VRAM 측정
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_all.py [출력 폴더]
import io, json, os, re, subprocess, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.normpath(os.path.join(HERE, '..'))
RUNS = os.path.normpath(os.path.join(F1, '..'))
OUT_F = os.path.join(F1, 'out')
T = os.path.join(RUNS, '2026-09-30_t')
SC = os.path.join(RUNS, '2026-10-01_scripted')
IM = os.path.join(RUNS, '2026-10-02_imitation', 'out')
ME = os.path.join(OUT_F, 'diag_meaneval')
PPP = os.path.normpath(os.path.join(RUNS, '..', '..'))
sys.path.insert(0, T)
sys.path.insert(0, SC)
from report_parse import parse_eval  # noqa: E402
import summarize as S  # noqa: E402

NAME = '2026-10-03_지시별_결과_종합'
SEEDS = [43, 44, 45, 46, 47]
ESEEDS = [999, 1000, 1001]
LSEEDS = [43, 46, 47]
P0_GOAL, P0_VCOLL = 80.0, 15.0
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'svg.fonttype': 'none', 'font.size': 8.5,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white'})
RUN_MAIN = 'burnin=2400 dec=10000 drain=0 ring=1.0 crossing=0 seed=999 envs=256 vessels=16'


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f(v, nd=1, sign=False):
    if v is None:
        return 'n/a'
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace('-', '−')


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index('<svg'):]


# ── 1. 규칙 배 4차 ──
WROLE = re.compile(r"(giveway|headon|overtaking|standon)=\s*([\d.]+)%\(n=(\d+) ([^)]*)\)")
MET = S.METRICS + [('w_r8', 'W 8조 작은 변침 R8 %', +1)]
SGN = {k: s for k, _, s in MET}
LAB = {k: l for k, l, _ in MET}


def load_sc(fp):
    r = S.parse_all(fp)
    s = open(fp, encoding='utf-8', errors='replace').read()
    assert 'goal' in r and 'R8=' in s, fp
    num, den = 0.0, 0
    for _, _, n, body in WROLE.findall(s):
        mm = re.search(r'R8=([\d.]+)', body)
        if mm:
            num += float(mm.group(1)) * int(n); den += int(n)
    r['w_r8'] = num / den if den else None
    return r


POLS = ['goal', 'vo56', 'vo56h150', 'vo300', 'vo300i', 'col56', 'col300', 'col300i', 'col56h', 'col300h', 'col300ih']
RS = {p: {s: load_sc(os.path.join(SC, 'out', f'sc_{p}_e{s}.txt')) for s in ESEEDS} for p in POLS}
RS.update({f't_{a}': {s: load_sc(os.path.join(SC, 'out', f'lr_t_{a}_s{s}.txt')) for s in LSEEDS} for a in ('off', 'a6')})


def Ms(p, k):
    return mean([RS[p][s].get(k) for s in RS[p]])


def pair(a, b):
    w = lo = mx = 0
    lose, mixed = [], []
    for k, _, _ in MET:
        ok = [(RS[a][s].get(k), RS[b][s].get(k)) for s in sorted(set(RS[a]) & set(RS[b]))]
        ok = [(x, y) for x, y in ok if x is not None and y is not None]
        if not ok:
            continue
        nw = sum(1 for x, y in ok if (x - y) * SGN[k] > 0)
        nl = sum(1 for x, y in ok if (x - y) * SGN[k] < 0)
        if nw == len(ok):
            w += 1
        elif nl == len(ok):
            lo += 1; lose.append(k)
        else:
            mx += 1; mixed.append(k)
    return w, lo, mx, lose, mixed


PAIRS = [('col300ih', 'col56h', 'COLREGs 규칙 배: 300 m + 의도 vs 56 m'), ('vo300i', 'vo56h150', 'VO 배: 300 m + 의도 vs 56 m (H·dt 같음)'),
         ('vo300i', 'vo56', 'VO 배: 300 m + 의도 vs 56 m (H·dt 도 다름)'), ('vo300', 'vo56h150', 'VO 배: 보는 거리만'),
         ('vo300i', 'vo300', 'VO 배: 의도 공유만'), ('t_a6', 't_off', '학습 배 t_a6 vs t_off (09-30 배치)')]
PR = {(a, b): pair(a, b) for a, b, _ in PAIRS}

# ── 2. P0: f2_ / i_ 학생, 샘플 행동(주 평가) vs 평균 행동(재분석) ──
def load_eval(fp, need_mean=False):
    r = parse_eval(fp)
    txt = open(fp, encoding='utf-8', errors='replace').read()
    m = re.search(r'\[run\] (.*)', txt)
    assert 'goal' in r and m and m.group(1).strip().startswith(RUN_MAIN), (fp, m and m.group(1))
    assert (not need_mean) or re.search(r'\[eval_mean\] mean-action actor calls=\d+', txt), f'평균 행동 표시 없음: {fp}'
    r['p0'] = r['goal'] >= P0_GOAL and r['vColl'] <= P0_VCOLL
    return r


ST = {'f2': {sd: load_eval(os.path.join(OUT_F, f'eval_f2_dagger_d6_s{sd}.txt')) for sd in SEEDS},
      'i': {sd: load_eval(os.path.join(IM, f'eval_i_dagger_d6_s{sd}.txt')) for sd in SEEDS}}
MN = {'f2': {sd: load_eval(os.path.join(ME, f'meaneval_f2_dagger_d6_s{sd}.txt'), True) for sd in SEEDS},
      'i': {sd: load_eval(os.path.join(ME, f'meaneval_i_dagger_d6_s{sd}.txt'), True) for sd in SEEDS}}
p0_md = open(os.path.join(OUT_F, '_p0.md'), encoding='utf-8').read()
P0_LINE = re.search(r'P0 판정: .*', p0_md).group(0)
assert f"{sum(r['p0'] for r in ST['f2'].values())}/5" in P0_LINE, P0_LINE


def A(d, k):
    return mean([r.get(k) for r in d.values()])


def npass(d):
    return sum(1 for r in d.values() if r['p0'])


def diag(name):
    s = open(os.path.join(OUT_F, 'diag_p0', name + '.log'), encoding='utf-8', errors='replace').read()
    return json.loads(re.search(r'^\[diag\] (\{.*\})$', s, re.M).group(1))


DG = {k: diag(k) for k in ('s43_teacher', 's43_student', 's44_student')}


def vram(name):
    s = open(os.path.join(OUT_F, 'diag_vram', name + '.log'), encoding='utf-8', errors='replace').read()
    tr = [float(c) for c in re.findall(r'\[vram\] t=\s*\d+s alloc=[\d.]+ reserved=([\d.]+)', s)]
    end = re.search(r'\[vram\] END peak_alloc=([\d.]+)GB peak_reserved=([\d.]+)GB', s)
    return dict(peak_alloc=float(end.group(1)) if end else None, peak_res=float(end.group(2)) if end else max(tr))


VR = {k: vram(k) for k in ('m2', 'v_gc30', 't_gc30')}
T_GOAL, T_VC = Ms('vo56h150', 'goal'), Ms('vo56h150', 'vColl')       # f2_ 선생님
T56_GOAL, T56_VC = Ms('vo56', 'goal'), Ms('vo56', 'vColl')            # i_ 선생님
print('pairs:', {f'{a} vs {b}': PR[(a, b)][:3] for a, b, _ in PAIRS})
for fam in ('f2', 'i'):
    print(fam, 'sampled', f(A(ST[fam], 'goal')), f(A(ST[fam], 'vColl')), f'{npass(ST[fam])}/5', '| mean', f(A(MN[fam], 'goal')), f(A(MN[fam], 'vColl')), f'{npass(MN[fam])}/5',
          '| per-seed mean goal', [round(MN[fam][s]['goal'], 1) for s in SEEDS], 'vColl', [round(MN[fam][s]['vColl'], 1) for s in SEEDS])


def fig_pairs():
    fig, ax = plt.subplots(figsize=(7.4, 2.3))
    ys = list(range(len(PAIRS)))[::-1]
    for y, (a, b, lab) in zip(ys, PAIRS):
        w, lo, mx, _, _ = PR[(a, b)]
        ax.barh(y, w, color='#2e7d32', height=0.6, zorder=2)
        ax.barh(y, mx, left=w, color='#c9c8c3', height=0.6, zorder=2)
        ax.barh(y, lo, left=w + mx, color='#c0392b', height=0.6, zorder=2)
        ax.text(w / 2, y, str(w), ha='center', va='center', fontsize=7, color='white')
        if lo:
            ax.text(w + mx + lo / 2, y, str(lo), ha='center', va='center', fontsize=7, color='white')
        if mx:
            ax.text(w + mx / 2, y, str(mx), ha='center', va='center', fontsize=7, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{a} vs {b}' for a, b, _ in PAIRS], fontsize=7)
    ax.set_xlabel(f'지표 {len(MET)}개 — 초록 = 왼쪽 정책 3/3 승, 회색 = 갈림, 빨강 = 0/3', fontsize=7.5)
    ax.set_xlim(0, len(MET))
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    return svg(fig)


def fig_p0():
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.axhspan(0, P0_VCOLL, color='#e8f5e9', zorder=0)
    ax.axvline(P0_GOAL, color='#2e7d32', lw=0.8, ls='--', zorder=1)
    ax.axhline(P0_VCOLL, color='#2e7d32', lw=0.8, ls='--', zorder=1)
    ax.text(P0_GOAL + 0.5, 1.0, 'P0 통과 영역', fontsize=6.8, color='#2e7d32')
    for fam, c, lab in (('f2', '#c0392b', 'f2_ 학생(목표 침로·ARPA@56)'), ('i', '#7a5ea8', 'i_ 학생(타각·레이더만)')):
        for sd in SEEDS:
            ax.annotate('', xy=(MN[fam][sd]['goal'], MN[fam][sd]['vColl']), xytext=(ST[fam][sd]['goal'], ST[fam][sd]['vColl']),
                        arrowprops=dict(arrowstyle='->', color=c, lw=0.6, alpha=0.5), zorder=2)
        ax.scatter([r['goal'] for r in ST[fam].values()], [r['vColl'] for r in ST[fam].values()], s=22, color=c, zorder=3, label=lab + ' — 샘플 행동(주 평가)')
        ax.scatter([r['goal'] for r in MN[fam].values()], [r['vColl'] for r in MN[fam].values()], s=22, facecolor='white', edgecolor=c, linewidth=1.1, zorder=3,
                   label=lab + ' — 평균 행동(재분석)')
    for nm, lab, off in (('vo56h150', '선생님 vo56h150', (4, 5)), ('vo56', '선생님 vo56', (4, -10)), ('goal', 'goal 배(안 피함)', (5, -3))):
        ax.scatter([Ms(nm, 'goal')], [Ms(nm, 'vColl')], s=34, color='#eb6834' if nm != 'goal' else '#8a8985', marker='s', edgecolor=INK, linewidth=0.5, zorder=4)
        ax.annotate(lab, (Ms(nm, 'goal'), Ms(nm, 'vColl')), textcoords='offset points', xytext=off, fontsize=6.8, color=INK)
    ax.set_xlabel('도착 % (주 평가 조건)', fontsize=7.5)
    ax.set_ylabel('배끼리 충돌 %', fontsize=7.5)
    ax.set_xlim(35, 100)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=6.3, frameon=False, loc='upper right')
    fig.tight_layout()
    return svg(fig)


RK = [('goal', '도착 %', 1), ('vColl', '배끼리 충돌 %', 1), ('TO', '시간초과 %', 1), ('dcpa', '통과거리 m', 1), ('fuel', '연료(도착 ep)', 0), ('head', '방향 바꾼 총량 °', 0)]


def tbl_rule():
    names = [('vo56', 'vo56 (56 m, H 60 s)'), ('vo56h150', 'vo56h150 (56 m) = f2_ OFF 선생님'), ('vo300', 'vo300 (300 m)'), ('vo300i', 'vo300i (300 m + 의도) = 통신 선생님'),
             ('col56h', 'col56h (COLREGs 규칙, 56 m)'), ('col300ih', 'col300ih (COLREGs 규칙, 300 m + 의도)'), ('goal', 'goal (안 피함)'),
             ('t_off', '학습 t_off (09-30)'), ('t_a6', '학습 t_a6 (09-30)')]
    keys = [('goal', '도착 %', 1), ('vColl', '배끼리 충돌 %', 1), ('dcpa', '통과거리 m', 1), ('w_score', 'COLREGs(Woerner) %', 1), ('fuel', '연료(도착 ep)', 0),
            ('fleet_fuel', '도착 1회당 함대 연료', 0), ('head', '방향 바꾼 총량 °', 0), ('rud_trav', '타 이동량 °/ep', 0)]
    h = ['<table><tr><th>정책</th>' + ''.join(f'<th>{l}</th>' for _, l, _ in keys) + '</tr>']
    for nm, lab in names:
        cls = ' class="ref"' if nm.startswith('t_') else ''
        h.append(f'<tr{cls}><td>{lab}</td>' + ''.join(f'<td>{f(Ms(nm, k), nd)}</td>' for k, _, nd in keys) + '</tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_pairs():
    h = ['<table><tr><th>짝 (왼쪽 = 통신·새 조건)</th><th>무엇이 다른가</th><th>3/3 승</th><th>갈림</th><th>0/3</th><th>0/3 인 지표</th></tr>']
    for a, b, lab in PAIRS:
        w, lo, mx, lose, _ = PR[(a, b)]
        h.append(f'<tr><td>{a} vs {b}</td><td style="text-align:left">{lab}</td><td class="ok">{w}</td><td>{mx}</td><td class="{"bad" if lo else ""}">{lo}</td>'
                 f'<td style="text-align:left">{", ".join(LAB[k] for k in lose) if lose else "없음"}</td></tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_p0(fam):
    h = ['<table><tr><th rowspan="2">시드</th><th colspan="4">샘플 행동 = 주 평가 (P0 판정 근거)</th><th colspan="4">평균 행동 = 재분석 (잡음 없음, 판정 아님)</th></tr>'
         '<tr><th>도착 %</th><th>충돌 %</th><th>방향 °</th><th>P0</th><th>도착 %</th><th>충돌 %</th><th>방향 °</th><th>기준 대입</th></tr>']
    for sd in SEEDS:
        a, b = ST[fam][sd], MN[fam][sd]
        h.append(f'<tr><td>s{sd}</td><td>{f(a["goal"])}</td><td>{f(a["vColl"])}</td><td>{f(a["head"], 0)}</td><td class="{"ok" if a["p0"] else "bad"}">{"통과" if a["p0"] else "불통과"}</td>'
                 f'<td>{f(b["goal"])}</td><td>{f(b["vColl"])}</td><td>{f(b["head"], 0)}</td><td class="{"ok" if b["p0"] else "bad"}">{"충족" if b["p0"] else "미달"}</td></tr>')
    h.append(f'<tr class="ref"><td>평균</td><td>{f(A(ST[fam], "goal"))}</td><td>{f(A(ST[fam], "vColl"))}</td><td>{f(A(ST[fam], "head"), 0)}</td><td>{npass(ST[fam])}/5</td>'
             f'<td>{f(A(MN[fam], "goal"))}</td><td>{f(A(MN[fam], "vColl"))}</td><td>{f(A(MN[fam], "head"), 0)}</td><td>{npass(MN[fam])}/5</td></tr>')
    h.append('</table>')
    return '\n'.join(h)


def gap(fam, tg, tv):
    sg, mg = A(ST[fam], 'goal'), A(MN[fam], 'goal')
    sv, mv = A(ST[fam], 'vColl'), A(MN[fam], 'vColl')
    return dict(noise_goal=mg - sg, rest_goal=tg - mg, noise_vc=sv - mv, rest_vc=mv - tv,
                win_goal=sum(1 for s in SEEDS if MN[fam][s]['goal'] > ST[fam][s]['goal']), win_vc=sum(1 for s in SEEDS if MN[fam][s]['vColl'] < ST[fam][s]['vColl']))


G2, GI = gap('f2', T_GOAL, T_VC), gap('i', T56_GOAL, T56_VC)
f2_vs_i_goal = sum(1 for s in SEEDS if MN['f2'][s]['goal'] > MN['i'][s]['goal'])
f2_vs_i_vc = sum(1 for s in SEEDS if MN['f2'][s]['vColl'] < MN['i'][s]['vColl'])
w1 = PR[('col300ih', 'col56h')]; w2 = PR[('vo300i', 'vo56h150')]; w6 = PR[('t_a6', 't_off')]

html = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>지시별 결과 종합</title>
<style>
@page {{ size: A4; margin: 12mm 11mm; }}
body {{ font-family: "Malgun Gothic", sans-serif; font-size: 9.5pt; line-height: 1.5; color: #111; background: #fff; }}
h1 {{ font-size: 14pt; margin: 0 0 4px; }}
h2 {{ font-size: 11.8pt; margin: 14px 0 6px; border-bottom: 2px solid #333; padding-bottom: 2px; }}
h3 {{ font-size: 10pt; margin: 9px 0 3px; }}
.sub {{ color: #555; font-size: 8.3pt; margin-bottom: 8px; }}
table {{ border-collapse: collapse; margin: 5px 0 7px; font-size: 7.9pt; }}
th, td {{ border: 1px solid #999; padding: 2px 4px; text-align: right; vertical-align: top; }}
th {{ background: #eee; text-align: center; }} td:first-child, th:first-child {{ text-align: left; }}
table.cmd td {{ text-align: left; }}
tr.ref td {{ background: #f6f6f6; color: #444; }}
.key {{ border: 2px solid #1b4f8c; background: #f3f8fe; padding: 6px 12px; margin: 8px 0; }}
.note {{ background: #f4f4f4; border-left: 4px solid #888; padding: 6px 10px; margin: 6px 0; font-size: 8.8pt; }}
.bad {{ color: #b00; font-weight: bold; }} .ok {{ color: #070; font-weight: bold; }}
ul {{ margin: 3px 0 3px 18px; padding: 0; }} li {{ margin: 1px 0; }}
.cap {{ font-size: 8pt; color: #444; margin: 2px 0 8px; }}
.pb {{ page-break-before: always; }}
svg {{ max-width: 100%; height: auto; }}
code {{ font-size: 8pt; }}
</style></head><body>

<h1>2026-10-02 ~ 10-03 지시별 결과 종합 + 재분석</h1>
<div class="sub">Windows LICS-SERVER 세션에서 받은 지시 4건과 결과, 그리고 10-03 저녁 재분석(평균 행동 주 평가). 코드 <code>feat/fig1-latent</code> 9183a0f ·
이 문서 = <code>runs/2026-10-02_fig1/report/make_report_all.py</code> 가 로그에서 직접 읽어 만듦 · 상세본: <code>2026-10-02_규칙배4차_Fig1상한_결과.pdf</code>, <code>2026-10-03_Fig1_f2_흉내P0_결과.pdf</code></div>

<h2>0. 한눈에</h2>
<table class="cmd">
<tr><th>#</th><th>지시</th><th>실행</th><th>결과</th></tr>
<tr><td>1</td><td>10-02 오전 — 규칙 배 4차 36칸(<code>_run_scripted.sh</code>) + 결과는 항상 PDF 로</td><td>10:52–13:27, 36칸 실패 없음 (+ vo56h150 3칸 16:36–17:3x)</td>
<td><b>규칙 배에서는 통신 정보가 이김</b>: vo300i vs vo56h150 = 지표 {len(MET)}개 중 3/3 승 {w2[0]} · 0/3 {w2[1]}. COLREGs 규칙 배 짝(col300ih vs col56h)은 {w1[0]} · {w1[1]} 이고 OFF 쪽이 최선 OFF 아님</td></tr>
<tr><td>2</td><td>10-02 16:3x — feat/fig1-latent 전환 → c_ phase0 → vo56h150 재측정 → P0 통과 시 phase1</td><td>c_ 는 16:37 preflight 에서 멈춤(검사 14종 중 1개 FAIL). vo56h150 은 끝남</td>
<td>c_ 결과 없음. 원인 = 테스트 1개가 타각 행동을 가정(코드는 스펙대로) → Mac 세션이 테스트 고침(2986481). c_ 는 다시 안 돌림(지시)</td></tr>
<tr><td>3</td><td>10-02 17:0x — Fig1 본 배치 f_ 실행(<code>_run_f.sh phase0</code>, P0 통과 시 자동으로 phase1)</td><td>17:04 시작 → DAgger 무진행 → 22:45 TDR → 10-03 00:04 서버 비정상 종료·재부팅</td>
<td><span class="bad">결과 없음</span>(P0 판정 못 냄). 원인 = 메모리 설정 두 가지(§3)</td></tr>
<tr><td>4</td><td>10-03 — "안 터지는 방식으로 다시 돌려봐"</td><td>원인 측정 → f2_(같은 코드·설정, 실행 환경만 수정) 14:27–15:53, 멈춤 없이 끝</td>
<td><span class="bad">{P0_LINE}</span> — 학생 도착 {f(A(ST['f2'], 'goal'))} % · 충돌 {f(A(ST['f2'], 'vColl'))} % (기준 80 · 15). 스펙대로 멈춤 → <b>trunk·코덱·갈래 학습(phase1)은 안 돌았음</b></td></tr>
<tr><td>5</td><td>10-03 저녁 — "다시 분석"</td><td>같은 학생을 잡음 없는 평균 행동으로 주 평가(10칸, 20:2x–)</td>
<td>f2_ 학생 도착 {f(A(ST['f2'], 'goal'))} → {f(A(MN['f2'], 'goal'))} % · 충돌 {f(A(ST['f2'], 'vColl'))} → {f(A(MN['f2'], 'vColl'))} % (기준 대입 {npass(MN['f2'])}/5). i_ 학생 {f(A(ST['i'], 'goal'))} → {f(A(MN['i'], 'goal'))} % · {f(A(ST['i'], 'vColl'))} → {f(A(MN['i'], 'vColl'))} % ({npass(MN['i'])}/5) (§4)</td></tr>
</table>
<div class="key"><ul>
<li><b>학습 결과(Fig1: 통신 vs 무통신 학습 배)는 아직 없음.</b> 이번 이틀 동안 학습은 trunk 까지 한 번도 못 감 — 흉내 0단계(P0)가 관문에서 걸림</li>
<li>확인된 것: ① 규칙 배에서는 300 m + 의도 공유가 56 m 를 거의 모든 지표에서 이김(통신 정보의 가치는 있음) ② 선생님을 흉내 낸 학생은 P0 기준에 못 미침(i_ {npass(ST['i'])}/5, f2_ {npass(ST['f2'])}/5)</li>
<li>재분석: f2_ P0 수치(도착 {f(A(ST['f2'], 'goal'))} %)의 큰 부분은 <b>평가가 학생의 샘플 행동(학습 안 된 잡음)으로 배를 몰기 때문</b>. 잡음을 빼면 f2_ 도착 +{f(G2['noise_goal'], 0)} pp · 충돌 −{f(G2['noise_vc'], 0)} pp ({G2['win_goal']}/5 · {G2['win_vc']}/5 시드).
<b>타각 행동인 i_ 에서는 같은 잡음 제거 효과가 +{f(GI['noise_goal'], 0)} pp 뿐</b> → 잡음 피해는 목표 침로 행동에서 크게 나타남</li>
<li>잡음을 빼면 f2_ 학생(도착 {f(A(MN['f2'], 'goal'))} · 충돌 {f(A(MN['f2'], 'vColl'))} %)이 i_ 학생({f(A(MN['i'], 'goal'))} · {f(A(MN['i'], 'vColl'))} %)보다 {f2_vs_i_goal}/5 시드에서 나음. 그래도 선생님과 도착 {f(G2['rest_goal'], 0)} pp · 충돌 {f(G2['rest_vc'], 0)} pp 차이가 남아(흉내 자체의 부족) 기준 대입 {npass(MN['f2'])}/5</li>
</ul></div>

<h2>1. 지시 1 — 규칙 배 4차 (학습 없음, 통신 정보의 상한)</h2>
{tbl_pairs()}
{fig_pairs()}
<div class="cap">그림 1. 같은 평가 시드 3개 짝 비교. 지표 {len(MET)}개(안전·Woerner COLREGs·옛 COLREGs·연료·조타·보상).</div>
{tbl_rule()}
<ul>
<li><b>VO 배</b>: vo300i 가 vo56h150 을 충돌 {f(Ms('vo56h150', 'vColl'))} → {f(Ms('vo300i', 'vColl'))} % · 도착 {f(Ms('vo56h150', 'goal'))} → {f(Ms('vo300i', 'goal'))} % · Woerner {f(Ms('vo56h150', 'w_score'))} → {f(Ms('vo300i', 'w_score'))} % ·
연료 {f(Ms('vo56h150', 'fuel'), 0)} → {f(Ms('vo300i', 'fuel'), 0)} · 방향 {f(Ms('vo56h150', 'head'), 0)} → {f(Ms('vo300i', 'head'), 0)}° 로 이김. 진 지표: {", ".join(LAB[k] for k in w2[3])}</li>
<li><b>COLREGs 규칙 배</b>: col300ih 는 col56h 는 이기지만 col56h 자체가 vo56(충돌 {f(Ms('vo56', 'vColl'))} %)보다 훨씬 나쁨(충돌 {f(Ms('col56h', 'vColl'))} %) → 이 짝은 '최선 OFF 를 이김'이 아님. col300ih 충돌 {f(Ms('col300ih', 'vColl'))} % 도 vo56 보다 높음</li>
<li>의도 공유만의 효과(vo300i vs vo300): 충돌 {f(Ms('vo300', 'vColl'))} → {f(Ms('vo300i', 'vColl'))} % 로 같고, 연료 {f(Ms('vo300', 'fuel'), 0)} → {f(Ms('vo300i', 'fuel'), 0)} · 방향 {f(Ms('vo300', 'head'), 0)} → {f(Ms('vo300i', 'head'), 0)}° 가 줄어듦</li>
<li>학습 배(09-30 t_ 체크포인트 재채점): Woerner 점수는 가장 높으나({f(Ms('t_off', 'w_score'))} · {f(Ms('t_a6', 'w_score'))} %) 충돌 {f(Ms('t_off', 'vColl'))} · {f(Ms('t_a6', 'vColl'))} %. t_a6 vs t_off = 3/3 승 {w6[0]} · 0/3 {w6[1]}</li>
<li>재현: 3차와 행동이 같은 24칸의 끝 줄이 글자까지 같음(상세본) — 규칙 배 평가는 결정론</li>
</ul>

<div class="pb"></div>
<h2>2. 지시 2·3 — c_ · f_ (결과 없음)</h2>
<ul>
<li><b>c_</b>(레이더만 + 목표 침로 흉내): preflight 의 <code>test_comm_ext::test_intent_timing_is_last_command</code> 1건 FAIL 로 학습 전에 멈춤.
테스트가 '의도 필드 = 날것 a0' 를 가정했는데 목표 침로 모드에서는 env 조타 제어기가 낸 타 비율이 들어감(스펙의 '명령 타각'과 일치, 6/6 확인). 코드 버그 아님 → 테스트 수정(2986481)</li>
<li><b>f_</b>(Fig1 본 배치): DAgger 5시드가 17:12–17:15 부터 무진행 → s47 만 22:44 완료(333분) → 22:45 TDR → 재부팅. P0 평가까지 못 감</li>
</ul>

<h2>3. 지시 4 — "안 터지는 방식" : 원인과 조치</h2>
<ul>
<li>원인 ①: <code>config.py</code> 기본 할당기 설정 <code>max_split_size_mb:128</code> 아래에서 DAgger 프로세스의 PyTorch 캐시가 {f(VR['m2']['peak_res'], 1)} GB 까지 불어남(실사용 최대 {f(VR['v_gc30']['peak_alloc'], 1)} GB). 프로세스 1개만 띄워도 재현</li>
<li>조치 ①: <code>PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3</code> → reserved {f(VR['v_gc30']['peak_res'], 1)} GB 고정·정상 속도. 메모리 관리만 바뀜(계산 결과 불변, 코드·HEAD 그대로)</li>
<li>원인 ②: trunk 학습은 프로세스당 실사용 {f(VR['t_gc30']['peak_alloc'], 1)} GB(reserved {f(VR['t_gc30']['peak_res'], 1)} GB) → 스크립트 기본 'GPU당 2개'면 16 GB 초과. 조치 ②: phase1 은 GPU당 1개·동시 4개</li>
<li>결과: f2_ 는 smoke(preflight 14종) → DAgger 5시드(48분) → 주 평가(34분)까지 멈춤·TDR 없이 끝남. 실험 설정·기준은 <code>_run_f.sh</code> 그대로</li>
</ul>

<h2>4. f2_ P0 결과와 재분석</h2>
<div class="note">P0 판정 근거는 '샘플 행동 = 주 평가' 열뿐(기준·프로토콜은 결과 전에 고정). '평균 행동' 열은 10-03 저녁에 따로 돌린 진단(<code>report/diag/eval_mean.py</code>) — 같은 체크포인트·같은 평가 조건(256 env · 10000결정 · burn-in 2400 · 평가 시드 999)에서
학생이 샘플 대신 tanh(평균) 을 실행하게만 바꿈. '기준 대입' 은 참고용이고 P0 통과가 아님.</div>
<h3>4-1. f2_ 학생 (목표 침로 행동 · ARPA@56 · 선생님 vo56h150: 도착 {f(T_GOAL)} · 충돌 {f(T_VC)} %)</h3>
{tbl_p0('f2')}
<h3>4-2. i_ 학생 (10-02 새벽, 타각 행동 · 레이더만 · 선생님 vo56: 도착 {f(T56_GOAL)} · 충돌 {f(T56_VC)} %)</h3>
{tbl_p0('i')}
{fig_p0()}
<div class="cap">그림 2. 채운 점 = 샘플 행동(주 평가), 빈 점 = 같은 시드의 평균 행동(화살표로 이음). 네모 = 규칙 배 평균. 초록 = P0 통과 영역.</div>
<ul>
<li><b>잡음의 몫</b>: f2_ 도착 +{f(G2['noise_goal'])} pp · 충돌 −{f(G2['noise_vc'])} pp ({G2['win_goal']}/5 · {G2['win_vc']}/5 시드 개선). i_ 도착 {f(GI['noise_goal'], 1, True)} pp · 충돌 {f(-GI['noise_vc'], 1, True)} pp ({GI['win_goal']}/5 · {GI['win_vc']}/5).
DAgger 는 행동 잡음 크기(logstd)를 학습하지 않아 초기값 그대로(침로 명령 표준편차 {f(DG['s43_student']['sampled_std_a0'], 2)} ≈ {f(DG['s43_student']['sampled_std_a0'] * 120, 0)}°) — <code>eval_ckpt.py</code> 는 그 샘플을 실행</li>
<li><b>흉내의 몫</b>: 평균 행동으로도 선생님 대비 f2_ 도착 −{f(G2['rest_goal'])} pp · 충돌 +{f(G2['rest_vc'])} pp, i_ 도착 −{f(GI['rest_goal'])} pp · 충돌 +{f(GI['rest_vc'])} pp</li>
<li><b>f2_ vs i_ (평균 행동, 같은 시드)</b>: f2_ 가 도착 {f2_vs_i_goal}/5 · 충돌 {f2_vs_i_vc}/5 시드에서 앞섬 — 단 두 배치는 행동·입력·선생님이 모두 달라 어느 변경의 효과인지는 못 가림</li>
<li>라벨 불균형(진단 창, 상세본 §3): 선생님 라벨의 약 {f(100 - DG['s43_teacher']['label_nonzero_pct'], 0)} % 가 '그대로 감(0)'. 피함 라벨에서 학생 일치 {f(DG['s43_student']['agree_when_label_nonzero_pct'], 0)}–{f(DG['s43_teacher']['agree_when_label_nonzero_pct'], 0)} %,
변침 크기는 선생님의 절반쯤({f(DG['s43_teacher']['mean_abs_student_when_label_nonzero'], 2)} vs {f(DG['s43_teacher']['mean_abs_label_nonzero'], 2)}). DAgger 로그의 '일치 98 %' 는 이 불균형 때문에 높게 나옴</li>
</ul>

<h2>5. 해석 — 근거가 있는 것과 없는 것</h2>
<div class="note"><ul>
<li>근거 있음: P0 주 평가 수치는 '흉내 품질'과 '학습 안 된 실행 잡음'이 섞인 값. 잡음을 빼면 f2_ 학생은 goal 배 수준({f(Ms('goal', 'goal'))} %)이 아니라 도착 {f(A(MN['f2'], 'goal'))} % 수준</li>
<li>근거 있음: 잡음을 빼도 P0 기준(80 · 15)은 {npass(MN['f2'])}/5(f2_) · {npass(MN['i'])}/5(i_) — 기준 대입일 뿐 판정은 그대로 불통과. f2_ 는 도착 {f(min(r['goal'] for r in MN['f2'].values()))}–{f(max(r['goal'] for r in MN['f2'].values()))} % · 충돌 {f(min(r['vColl'] for r in MN['f2'].values()))}–{f(max(r['vColl'] for r in MN['f2'].values()))} % 로 기준 근처까지는 옴</li>
<li>근거 있음: 학습 안 된 초기 실행 잡음(두 배치 모두 DAgger 가 logstd 를 안 건드림)을 빼면 목표 침로 행동(f2_)에서는 도착 {f(G2['noise_goal'], 0)} pp, 타각 행동(i_)에서는 {f(GI["noise_goal"], 0)} pp 가 회복됨. 단 두 배치는 입력·선생님도 달라 '행동 방식만의 효과'로 단정은 못 함(i_ 의 실행 잡음 크기는 따로 안 쟀음)</li>
<li>확인 안 함: PPO(trunk)를 붙이면 잡음(logstd)이 학습되며 회복되는지, 라벨 불균형을 보정하면 흉내가 선생님에 닿는지, 목표 침로 행동 자체의 득실</li>
<li>CLAUDE.md 에 '서버 불안정으로 다른 기계 실행'이 적혀 있음 — 다른 기계의 f_ 결과가 있으면 이 f2_ 와 같은 설정·시드라 대조 가능(이 문서엔 없음)</li>
</ul></div>

<h2>6. 다음 — 저자가 정할 것</h2>
<ul>
<li>P0 기준·평가 프로토콜은 결과 전에 고정된 것 → 이 문서는 바꾸지 않음. 바꿀지(예: P0 를 평균 행동으로 채점)는 저자 결정</li>
<li>갈림길: ① 흉내 단계에서 잡음(logstd)을 줄이거나 평가를 평균 행동으로 ② 라벨 불균형 보정 ③ 목표 침로 대신 타각으로 되돌림 ④ P0 없이 trunk 로 진행(PPO 가 잡음을 학습) ⑤ 다른 기계 f_ 결과 확인</li>
<li>서버는 비어 있음. 다시 돌리면 P0 까지 약 1.5시간, 끝(trunk → 코덱 → 갈래 → 평가)까지 약 10시간. 실행 환경은 <code>C:\\work\\f2_launch.sh</code> 방식(할당기 env · phase1 GPU당 1개)</li>
</ul>

<h2>7. 산출물</h2>
<ul>
<li><code>Private_Paper_Project/2026-10-02_규칙배4차_Fig1상한_결과.pdf</code> — 지시 1 상세(39칸 전 지표·짝 12개·R8 전후)</li>
<li><code>Private_Paper_Project/2026-10-03_Fig1_f2_흉내P0_결과.pdf</code> — 지시 4 상세(P0 표·라벨 진단·VRAM 측정)</li>
<li>원자료: <code>runs/2026-10-01_scripted/out/</code> · <code>runs/2026-10-02_fig1/out/</code>(<code>diag_p0/</code> · <code>diag_vram/</code> · <code>diag_meaneval/</code>) · 진단 스크립트 <code>runs/2026-10-02_fig1/report/diag/</code></li>
</ul>
</body></html>
'''

OUT = sys.argv[1] if len(sys.argv) > 1 else PPP
out_html = os.path.join(OUT, NAME + '.html')
out_pdf = os.path.join(OUT, NAME + '.pdf')
open(out_html, 'w', encoding='utf-8').write(html)
edge = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
subprocess.run([edge, '--headless', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={out_pdf}',
                'file:///' + out_html.replace(chr(92), '/')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(out_html)
print(out_pdf)
