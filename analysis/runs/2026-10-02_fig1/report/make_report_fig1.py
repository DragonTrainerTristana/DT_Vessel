# 2026-10-06: Fig1 배치 f2_ 결과(통신 comm vs OFF) PDF.
# 숫자는 전부 로그에서 읽음:
#   <dir>/lr_f2_{off,comm,offb}_s*.txt   주 평가 + Woerner·타 줄 (summarize.parse_all) — 기본 dir = runs/2026-10-02_fig1/out
#   <dir>/f2_{off,comm,offb}_s*.csv       학습 곡선(step, raw_reward, ema_reward)
#   <dir>/_fig1.md · _p0_override.txt · _gate.log · _codec_choice.txt · _timeline.txt   (있으면)
#   runs/2026-10-01_scripted/out/sc_*_e*.txt · lr_t_off_s*.txt   규칙 배·옛 학습 배 참고
# 판정 규칙은 summarize_fig1.py 와 같은 식(아래 judge) — _fig1.md 가 있으면 '주 지표 통과' 수가 같은지 대조함.
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_fig1.py [--interim] [자료 폴더] [출력 폴더]
import csv, glob, io, os, re, subprocess, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.normpath(os.path.join(HERE, '..'))
RUNS = os.path.normpath(os.path.join(F1, '..'))
SC = os.path.join(RUNS, '2026-10-01_scripted')
PPP = os.path.normpath(os.path.join(RUNS, '..', '..'))
sys.path.insert(0, os.path.join(RUNS, '2026-09-30_t'))
sys.path.insert(0, SC)
import summarize as S  # noqa: E402

INTERIM = '--interim' in sys.argv
ARGS = [a for a in sys.argv[1:] if a != '--interim']
D = ARGS[0] if len(ARGS) > 0 else os.path.join(F1, 'out')
OUT = ARGS[1] if len(ARGS) > 1 else PPP
PRE = 'f2_'
NAME = '2026-10-06_Fig1_f2_통신vsOFF_결과' + ('_중간' if INTERIM else '')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C_OFF, C_COMM, C_OFFB = '#d9822b', '#1b6fd6', '#9c9a94'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'svg.fonttype': 'none', 'font.size': 8.5,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white'})
RUN_MAIN = 'burnin=2400 dec=10000 drain=0 ring=1.0 crossing=0 seed=999 envs=256 vessels=16'
# summarize_fig1.py 와 같은 표
MAIN = [('vColl', '배끼리 충돌 %', -1, 12.1), ('goal', '도착 %', +1, 0.0), ('dcpa', 'DCPA(통과거리) m', +1, 1.5),
        ('w_score', 'COLREGs(Woerner) %', +1, 0.0), ('fuel', '연료(도착 ep)', -1, 0.0), ('fleet_fuel', '도착 1회당 함대 연료', -1, 0.0)]
POOL_KEYS = ('goal', 'vColl', 'dcpa')
AUX = [('head', '방향 바꾼 총량 °', -1, 0.0), ('rud_trav', '타 이동량 °/ep', -1, 0.0), ('C', '옛 COLREGs %', +1, 0.0),
       ('TO', '시간초과 %', -1, 0.0), ('oColl', '벽 충돌 %', -1, 0.0), ('epReward', 'epReward', +1, 0.0)]
ND = lambda k: 0 if k in ('fuel', 'fleet_fuel', 'head', 'epReward', 'rud_trav') else 1  # noqa: E731


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


def load(d, pre, arm, check=True):
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, f'lr_{pre}{arm}_s*.txt'))):
        m = re.search(r'_s(\d+)\.txt$', fp)
        r = S.parse_all(fp)
        if not (m and 'goal' in r):
            continue
        if check:
            txt = open(fp, encoding='utf-8', errors='replace').read()
            mm = re.search(r'\[run\] (.*)', txt)
            assert mm and mm.group(1).strip().startswith(RUN_MAIN), (fp, mm and mm.group(1))
            assert 'R8=' in txt, fp
        out[int(m.group(1))] = r
    return out


off, comm, offb = load(D, PRE, 'off'), load(D, PRE, 'comm'), load(D, PRE, 'offb')
seeds = sorted(set(off) & set(comm))
assert seeds, '평가 파일 없음'
t_off = load(os.path.join(SC, 'out'), 't_', 'off', check=False)
pool = {'off': off, 'offb': offb, 't_off': t_off}
pool_mean = {}
for nm, runs in pool.items():
    g = [r.get('goal') for r in runs.values() if r.get('goal') is not None]
    if runs and g and sum(g) / len(g) >= 50.0:
        pool_mean[nm] = {k: mean([r.get(k) for r in runs.values()]) for k in POOL_KEYS}


def judge(k, sg, floor):
    pa = [(off[s].get(k), comm[s].get(k)) for s in seeds]
    pa = [(x, y) for x, y in pa if x is not None and y is not None]
    if not pa:
        return None
    w = sum(1 for x, y in pa if (y - x) * sg > 0)
    mo, mc = mean([x for x, _ in pa]), mean([y for _, y in pa])
    rng = max(x for x, _ in pa) - min(x for x, _ in pa)
    nb = [abs(offb[s].get(k) - off[s].get(k)) for s in seeds if s in offb and offb[s].get(k) is not None]
    N = max(rng, (sum(nb) / len(nb)) if nb else 0.0, floor)
    ok = w == len(pa) and abs(mc - mo) > N
    note, pok = '', None
    if k in POOL_KEYS and pool_mean:
        who, best = max(((nm, pm[k]) for nm, pm in pool_mean.items()), key=lambda t: sg * t[1])
        pok = all(sg * (comm[s][k] - best) > 0 for s in seeds if comm[s].get(k) is not None)
        ok = ok and pok
        note = f"풀 최선 {who} {f(best)} {'넘음' if pok else '못 넘음'}"
    return dict(pa=pa, w=w, n=len(pa), mo=mo, mc=mc, d=mc - mo, N=N, rng=rng, nb=(sum(nb) / len(nb)) if nb else None, ok=ok, note=note, good=(mc - mo) * sg > 0)


J = {k: judge(k, sg, fl) for k, _, sg, fl in MAIN + AUX}
win_main = sum(1 for k, _, _, _ in MAIN if J[k] and J[k]['ok'])
dir_main = sum(1 for k, _, _, _ in MAIN if J[k] and J[k]['w'] == J[k]['n'])
print(f"seeds {seeds} | offb seeds {sorted(offb)} | main pass {win_main}/{len(MAIN)} | all-seed wins {dir_main}/{len(MAIN)}")
for k, lab, sg, _ in MAIN + AUX:
    j = J[k]
    if j:
        print(f"  {lab:22s} off {f(j['mo'], ND(k))} comm {f(j['mc'], ND(k))} win {j['w']}/{j['n']} d {f(j['d'], 2)} N {f(j['N'], 2)} {'PASS' if j['ok'] else ''} {j['note']}")
fig1_md = os.path.join(D, '_fig1.md')
MD_LINE = None
if os.path.exists(fig1_md):
    mdt = open(fig1_md, encoding='utf-8').read()
    m = re.search(r'주 지표 통과: (\d+)/(\d+)', mdt)
    if m:
        MD_LINE = m.group(0)
        assert int(m.group(1)) == win_main, (MD_LINE, win_main)
assert INTERIM or MD_LINE, '_fig1.md 없음 — 배치가 안 끝났으면 --interim'
OVR = open(os.path.join(D, '_p0_override.txt'), encoding='utf-8').read().strip() if os.path.exists(os.path.join(D, '_p0_override.txt')) else ''
GATE = open(os.path.join(D, '_gate.log'), encoding='utf-8').read().strip() if os.path.exists(os.path.join(D, '_gate.log')) else ''
CODEC = open(os.path.join(D, '_codec_choice.txt'), encoding='utf-8').read() if os.path.exists(os.path.join(D, '_codec_choice.txt')) else ''
CH_K = re.search(r'CHOSEN_K=(\w+)', CODEC).group(1) if 'CHOSEN_K' in CODEC else '?'
ref = {p: [S.parse_all(os.path.join(SC, 'out', f'sc_{p}_e{s}.txt')) for s in (999, 1000, 1001)] for p in ('vo56h150', 'vo300i', 'goal')}


def R(p, k):
    return mean([r.get(k) for r in ref[p]])


def A(d, k):
    return mean([d[s].get(k) for s in d])


def curve(arm, s):
    fp = os.path.join(D, f'{PRE}{arm}_s{s}.csv')
    if not os.path.exists(fp):
        return [], []
    xs, ys = [], []
    for row in csv.DictReader(open(fp, encoding='utf-8')):
        try:
            xs.append(float(row['step']) / 1e6); ys.append(float(row['ema_reward']))
        except (KeyError, ValueError):
            pass
    return xs, ys


def fig_main():
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.3))
    for ax, (k, lab, sg, _) in zip(axs.flat, MAIN):
        j = J[k]
        for s in seeds:
            x, y = off[s].get(k), comm[s].get(k)
            ax.plot([0, 1], [x, y], color='#bbb', lw=0.9, zorder=1)
            ax.scatter([0], [x], s=22, color=C_OFF, zorder=3)
            ax.scatter([1], [y], s=22, color=C_COMM, zorder=3)
            ax.annotate(f's{s}', (1, y), textcoords='offset points', xytext=(5, -2), fontsize=6, color=INK2)
        for s in sorted(offb):
            if offb[s].get(k) is not None:
                ax.scatter([-0.28], [offb[s][k]], s=14, color=C_OFFB, marker='x', zorder=2)
        if k in ('goal', 'vColl', 'dcpa', 'w_score', 'fuel', 'fleet_fuel'):
            for p, c, ls in (('vo56h150', C_OFF, ':'), ('vo300i', C_COMM, ':')):
                ax.axhline(R(p, k), color=c, lw=0.8, ls=ls, zorder=0)
        ax.set_xlim(-0.6, 1.5)
        ax.set_xticks([-0.28, 0, 1])
        ax.set_xticklabels(['offb', 'off', 'comm'], fontsize=6.8)
        ax.tick_params(axis='y', labelsize=6.5)
        ax.grid(axis='x', visible=False)
        clr = '#070' if j['ok'] else ('#b00' if j['w'] == 0 else INK)
        ax.set_title(f"{lab} {'↑' if sg > 0 else '↓'} · comm {j['w']}/{j['n']}" + (' · 통과' if j['ok'] else ''), fontsize=7, color=clr)
    fig.tight_layout(h_pad=1.0)
    return svg(fig)


def fig_curve():
    fig, axs = plt.subplots(1, len(seeds), figsize=(7.4, 2.4), sharey=True)
    axs = [axs] if len(seeds) == 1 else list(axs)
    for ax, s in zip(axs, seeds):
        for arm, c, lw in (('offb', C_OFFB, 0.9), ('off', C_OFF, 1.3), ('comm', C_COMM, 1.3)):
            xs, ys = curve(arm, s)
            if xs:
                ax.plot(xs, ys, color=c, lw=lw, label=arm)
        ax.axvline(9.043968, color=INK2, lw=0.7, ls=':')
        ax.set_title(f'시드 {s}', fontsize=7.5)
        ax.set_xlabel('결정 수 (M)', fontsize=7)
        ax.tick_params(labelsize=6.5)
    axs[0].set_ylabel('학습 중 보상 (EMA)', fontsize=7)
    axs[0].legend(fontsize=6.5, frameon=False, loc='lower right')
    fig.tight_layout()
    return svg(fig)


def tbl(rows, title):
    h = [f'<h3>{title}</h3>', '<table><tr><th>지표</th><th>off 평균<br><span class="sd">시드별</span></th><th>comm 평균<br><span class="sd">시드별</span></th>'
         '<th>comm 승/짝</th><th>평균차</th><th>N</th><th>N 의 출처<br><span class="sd">off 범위 · |offb−off| · 하한</span></th><th>판정</th></tr>']
    for k, lab, sg, floor in rows:
        j = J[k]
        if not j:
            h.append(f'<tr><td>{lab}</td>' + '<td>n/a</td>' * 7 + '</tr>')
            continue
        nd = ND(k)
        tag = 'ok' if j['w'] == j['n'] else ('bad' if j['w'] == 0 else '')
        h.append(f"<tr><td>{lab} {'↑' if sg > 0 else '↓'}</td>"
                 f"<td>{f(j['mo'], nd)}<br><span class=\"sd\">{'·'.join(f(x, nd) for x, _ in j['pa'])}</span></td>"
                 f"<td>{f(j['mc'], nd)}<br><span class=\"sd\">{'·'.join(f(y, nd) for _, y in j['pa'])}</span></td>"
                 f"<td class=\"{tag}\">{j['w']}/{j['n']}</td><td>{f(j['d'], 2, True)}</td><td>{f(j['N'], 2)}</td>"
                 f"<td><span class=\"sd\">{f(j['rng'], 2)} · {f(j['nb'], 2)} · {f(floor, 1)}</span></td>"
                 f"<td class=\"{'ok' if j['ok'] else 'bad'}\">{'통과' if j['ok'] else '불통과'}{('<br><span class=sd>' + j['note'] + '</span>') if j['note'] else ''}</td></tr>")
    h.append('</table>')
    return '\n'.join(h)


def tbl_runs():
    keys = [('goal', '도착 %'), ('vColl', '충돌 %'), ('TO', '시간초과 %'), ('dcpa', 'DCPA m'), ('w_score', 'Woerner %'), ('fuel', '연료'), ('fleet_fuel', '함대 연료'),
            ('head', '방향 °'), ('rud_trav', '타 이동 °'), ('epReward', 'epReward')]
    h = ['<table><tr><th>런</th>' + ''.join(f'<th>{l}</th>' for _, l in keys) + '</tr>']
    for arm, d in (('off', off), ('comm', comm), ('offb', offb)):
        for s in sorted(d):
            h.append(f'<tr><td>f2_{arm}_s{s}</td>' + ''.join(f'<td>{f(d[s].get(k), ND(k))}</td>' for k, _ in keys) + '</tr>')
        if d:
            h.append(f'<tr class="ref"><td>{arm} 평균</td>' + ''.join(f'<td>{f(A(d, k), ND(k))}</td>' for k, _ in keys) + '</tr>')
    for p, lab in (('vo56h150', '규칙 배 vo56h150 (OFF 선생님)'), ('vo300i', '규칙 배 vo300i (통신 선생님)'), ('goal', '규칙 배 goal (안 피함)')):
        h.append(f'<tr class="ref"><td>{lab}</td>' + ''.join(f'<td>{f(R(p, k), ND(k))}</td>' for k, _ in keys) + '</tr>')
    if t_off:
        h.append('<tr class="ref"><td>옛 학습 t_off (09-30)</td>' + ''.join(f'<td>{f(A(t_off, k), ND(k))}</td>' for k, _ in keys) + '</tr>')
    h.append('</table>')
    return '\n'.join(h)


NOTE = open(os.path.join(HERE, 'fig1_runnote.html'), encoding='utf-8').read().strip() if os.path.exists(os.path.join(HERE, 'fig1_runnote.html')) else ''
wins_all = [lab for k, lab, _, _ in MAIN + AUX if J[k] and J[k]['w'] == J[k]['n']]
lose_all = [lab for k, lab, _, _ in MAIN + AUX if J[k] and J[k]['w'] == 0]
passed = [lab for k, lab, _, _ in MAIN if J[k] and J[k]['ok']]
missing = [f'{a}_s{s}' for a, d in (('offb', offb),) for s in seeds if s not in d]

html = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Fig1 f2_ 통신 vs OFF</title>
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
tr.ref td {{ background: #f6f6f6; color: #444; }}
.sd {{ color: #777; font-size: 6.3pt; white-space: nowrap; }}
.key {{ border: 2px solid #1b4f8c; background: #f3f8fe; padding: 6px 12px; margin: 8px 0; }}
.warn {{ border: 2px solid #b00; background: #fff5f5; padding: 6px 12px; margin: 8px 0; }}
.note {{ background: #f4f4f4; border-left: 4px solid #888; padding: 6px 10px; margin: 6px 0; font-size: 8.8pt; }}
.bad {{ color: #b00; font-weight: bold; }} .ok {{ color: #070; font-weight: bold; }}
ul {{ margin: 3px 0 3px 18px; padding: 0; }} li {{ margin: 1px 0; }}
.cap {{ font-size: 8pt; color: #444; margin: 2px 0 8px; }}
.pb {{ page-break-before: always; }}
svg {{ max-width: 100%; height: auto; }}
code {{ font-size: 8pt; }}
pre {{ font-size: 7.4pt; background: #f7f7f7; padding: 5px 8px; white-space: pre-wrap; margin: 4px 0; }}
</style></head><body>
{('<div class="warn"><b>중간 보고</b> — 평가가 끝난 런만으로 만든 표. 아직 없는 것: ' + (', '.join(missing) if missing else '없음') + ' · 스크립트 최종 표(_fig1.md) 전. offb 가 다 차면 N 이 바뀔 수 있음 → 판정 칸은 잠정</div>') if INTERIM else ''}
<h1>2026-10-06 Fig1 배치 f2_ — 통신(comm) vs OFF 결과</h1>
<div class="sub">스펙 <code>docs/superpowers/specs/2026-10-02-fig1-latent-design.md</code> §3 · 판정 규칙 2026-09-30 §7 · 코드 <code>feat/fig1-latent</code> 74b5988(학습·평가 코드 = 9183a0f) ·
Windows LICS-SERVER 2026-10-06 · 원자료 <code>runs/2026-10-02_fig1/out/</code> · 이 문서 = <code>report/make_report_fig1.py</code> 가 로그에서 직접 읽어 만듦</div>

<h2>0. 요약</h2>
<div class="key"><ul>
<li><b>주 지표 통과 {win_main}/{len(MAIN)}</b>{(' (' + ', '.join(passed) + ')') if passed else ''} — 판정 = 같은 trunk 시드 짝 {len(seeds)}개 전부 승 + |평균차| &gt; N + (도착·충돌·DCPA 는) 최선 OFF 풀도 넘음{' · 스크립트 표: ' + MD_LINE if MD_LINE else ''}</li>
<li>시드 짝 전부 comm 이 앞선 지표({len(wins_all)}): {', '.join(wins_all) if wins_all else '없음'}</li>
<li>시드 짝 전부 comm 이 뒤진 지표({len(lose_all)}): {', '.join(lose_all) if lose_all else '없음'}</li>
<li>평균: 충돌 off {f(J['vColl']['mo'])} → comm {f(J['vColl']['mc'])} % · 도착 {f(J['goal']['mo'])} → {f(J['goal']['mc'])} % · DCPA {f(J['dcpa']['mo'])} → {f(J['dcpa']['mc'])} m ·
Woerner {f(J['w_score']['mo'])} → {f(J['w_score']['mc'])} % · 연료 {f(J['fuel']['mo'], 0)} → {f(J['fuel']['mc'], 0)} · 함대 연료 {f(J['fleet_fuel']['mo'], 0)} → {f(J['fleet_fuel']['mc'], 0)}</li>
<li>참고(규칙 배): OFF 선생님 vo56h150 충돌 {f(R('vo56h150', 'vColl'))} · 도착 {f(R('vo56h150', 'goal'))} %, 통신 선생님 vo300i {f(R('vo300i', 'vColl'))} · {f(R('vo300i', 'goal'))} %</li>
<li><b>전제</b>: 흉내 관문 P0 는 불통과(0/5)였고 저자 결정으로 trunk 로 진행함(아래 §1). trunk 관문·코덱 관문·판정 규칙은 사전등록 그대로</li>
</ul></div>

<h2>1. 무엇을 돌렸나</h2>
{('<pre>' + OVR + '</pre>') if OVR else ''}
<ul>
<li>공통 뿌리 = ARPA@56·메시지 없음, 목표 침로 행동 → DAgger(선생님 vo56h150) → trunk 9.04M(흉내 보조손실 1 → 0, 3M) → 관문 → 시드마다 p46 코덱 → 갈래 16.06M 까지</li>
<li>팔: <b>off</b>(56 m 안 참값 상대 필드, 선생님 vo56h150) · <b>comm</b>(300 m 메시지 = p46 코덱 k={CH_K} 복원값 + 의도 필드, 선생님 vo300i) · <b>offb</b>(off 재분기 = 잡음 자)</li>
<li>평가 = 주 평가(256 env · 10000결정 · burn-in 2400 · 평가 시드 999, <code>[run]</code> 줄 확인) + Woerner COLREGs(R8 포함)·타 지표 줄</li>
</ul>
{('<pre>' + GATE + '</pre>') if GATE else ''}

<h2>2. 판정표</h2>
{tbl(MAIN, '2-1. 주 지표 (스펙 §3)')}
{fig_main()}
<div class="cap">그림 1. 주 지표 — 같은 시드의 off(주황)와 comm(파랑)을 선으로 이음. × = offb(잡음 자). 점선 = 규칙 배 선생님(주황 vo56h150, 파랑 vo300i). 제목 = comm 승/짝.</div>
{tbl(AUX, '2-2. 보조 지표')}
<div class="cap">N = max(off 시드 범위, |offb − off| 평균, 고정 하한). 초록 = 시드 짝 전부 comm 승, 빨강 = 전부 패. '통과' 는 승/짝 전부 + |평균차| &gt; N (+ 풀 조건).</div>

<div class="pb"></div>
<h2>3. 런별 값</h2>
{tbl_runs()}

<h2>4. 학습 곡선 (학습 중 보상)</h2>
{fig_curve()}
<div class="cap">그림 2. 시드별 학습 중 보상(EMA). 세로 점선 = 분기점 9.04M. 학습 중 샘플 행동의 보상이라 고정 장면 평가 곡선은 아님(스펙 §3 의 reward 그림은 체크포인트 고정 평가로 따로 그려야 함).</div>

<h2>5. 읽을 때 주의</h2>
<div class="note"><ul>
<li>시드 {len(seeds)}개. 승/짝과 N 을 같이 볼 것 — 평균차가 N 보다 작으면 off 를 다시 분기해도 나오는 흔들림 범위</li>
<li>comm 과 off 는 선생님이 다름(vo300i vs vo56h150, 스펙 §4-1) — 차이에는 '300 m 메시지 + 의도'와 '다른 선생님의 흉내 보조손실(분기 뒤 3M 동안)'이 같이 들어 있음</li>
<li>P0 불통과 상태에서 시작한 trunk 임(저자 결정). trunk 관문은 5/5 통과, G1(도착 ≥ 90 · 충돌 ≤ 5)은 5시드 모두 미충족(기록만)</li>
<li>옛 COLREGs 지표는 보조. 주 COLREGs 지표 = Woerner(R8 포함)</li>
<li>학습 곡선(그림 2): 보상이 오르는 구간이 흉내 보조손실이 켜진 구간(trunk 0–3M, 갈래 9.04–12.04M)과 겹침. 그 뒤 trunk 는 평평하고 off 는 2/3 시드에서 내려가며 comm 도 더 오르지 않음 → comm 의 이득에 '통신 선생님 vo300i 흉내'의 몫이 얼마인지는 이 배치로 못 가림(가설, 확인 안 함)</li>
<li>off 가 시드마다 크게 다름(도착 {f(min(off[s]['goal'] for s in seeds))}–{f(max(off[s]['goal'] for s in seeds))} %, 낮은 쪽은 안 피하는 goal 배 {f(R('goal', 'goal'))} % 수준). 같은 trunk 에서 off 를 다시 분기한 offb 와의 도착률 차: {', '.join('s' + str(s) + ' ' + f(abs(offb[s]['goal'] - off[s]['goal'])) + ' pp' for s in seeds if s in offb)} — 재분기만으로 이만큼 흔들리는 시드가 있음. 이 흔들림과 off 시드 범위가 N 을 키움</li>
<li>comm 은 offb 와 견줘도 도착 {sum(1 for s in seeds if s in offb and comm[s]['goal'] > offb[s]['goal'])}/{sum(1 for s in seeds if s in offb)} · 충돌 {sum(1 for s in seeds if s in offb and comm[s]['vColl'] < offb[s]['vColl'])}/{sum(1 for s in seeds if s in offb)} 시드에서 앞섬(도착 comm {'·'.join(f(comm[s]['goal']) for s in seeds)} vs offb {'·'.join(f(offb[s]['goal']) for s in seeds if s in offb)})</li>
<li>원인 설명은 확인 안 함 — 이 문서는 표의 숫자만 근거</li>
</ul></div>

<h2>6. 실행 메모</h2>
<ul>
<li>10-06 11:30 phase1 시작 → trunk 5시드 15:00 → 관문(시드 43·44·45) → 코덱 k={CH_K} 15:14 → 갈래 시작</li>
<li>16:56 서버 비정상 종료·재부팅(Kernel-Power 41, 그 전 GPU 드라이버 시간초과 2회) → 17:04 같은 명령으로 재실행. 진행 중이던 갈래 4개는 trunk 에서 다시 시작(끝난 단계는 건너뜀)</li>
<li>실행 환경: <code>PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3</code> · GPU당 1런 · 다른 계정 작업이 6 GB 넘게 쓰는 카드엔 새 런 안 올림(<code>C:\\work\\f2_gpuguard.sh</code>) · 끝난 런 평가를 빈 GPU 에서 먼저 돌림(<code>f2_preeval.sh</code>, 명령·env 는 스크립트 eval 단계와 같음)</li>
{('<li>' + NOTE + '</li>') if NOTE else ''}
</ul>
</body></html>
'''

out_html = os.path.join(OUT, NAME + '.html')
out_pdf = os.path.join(OUT, NAME + '.pdf')
open(out_html, 'w', encoding='utf-8').write(html)
edge = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
subprocess.run([edge, '--headless', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={out_pdf}',
                'file:///' + out_html.replace(chr(92), '/')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(out_html)
print(out_pdf)
