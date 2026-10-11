# 3차(2026-10-02 새벽): Woerner 2019 COLREGs 재채점(규칙 배 5종 + 학습 배 t_off·t_a6) + 흉내 0단계(P0) 결과 정리 PDF.
# 숫자는 전부 로그에서 읽음: runs/2026-10-01_scripted/out/sc_*.txt·lr_t_*.txt (summarize.parse_all = report_parse + Woerner·타·행동 줄),
# runs/2026-10-02_imitation/out/eval_i_dagger_d6_s*.txt·i_dagger_d6_s*.log·_p0.md. 손으로 옮겨 적은 숫자 없음.
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report3.py   (base anaconda = matplotlib 있음)
import glob, io, os, re, subprocess, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True   # Dropbox 폴더에 __pycache__ 안 만듦
HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.normpath(os.path.join(HERE, '..'))                     # runs/2026-10-01_scripted
RUNS = os.path.normpath(os.path.join(SC, '..'))
T = os.path.join(RUNS, '2026-09-30_t')
IM = os.path.join(RUNS, '2026-10-02_imitation', 'out')
PPP = os.path.normpath(os.path.join(RUNS, '..', '..'))               # Dropbox/Private_Paper_Project
sys.path.insert(0, SC)
sys.path.insert(0, T)
import summarize as S  # noqa: E402
from report_parse import parse_eval  # noqa: E402

NAME = '2026-10-02_Woerner재채점_흉내P0_결과'
POLS = ['goal', 'vo56', 'vo300', 'col56', 'col300']
SEEDS = [999, 1000, 1001]
LSEEDS = [43, 46, 47]
COL = {'goal': '#8a8985', 'vo56': '#eb6834', 'vo300': '#2a78d6', 'col56': '#b5651d', 'col300': '#1b4f8c',
       't_off': '#9c9a94', 't_a6': '#6f9fd8', 'i_dagger': '#7a5ea8'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'svg.fonttype': 'none', 'font.size': 8.5,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white'})
MET = S.METRICS
SGN = {k: s for k, _, s in MET}
LAB = {k: l for k, l, _ in MET}
ND = lambda k: S.NDEC.get(k, 1)  # noqa: E731
W_K = ['w_score', 'w_giveway', 'w_headon', 'w_overtaking', 'w_standon']
assert all(k in LAB for k in W_K), [k for k in W_K if k not in LAB]
SAFE_K = ['vColl', 'goal', 'dcpa', 'minPass', 'TO', 'oColl']
OLD_K = ['C', 'C_headon', 'C_standon', 'C_giveway', 'C_overtake', 'rks_v2', 'rks_v2x']
COST_K = ['fuel', 'fleet_fuel', 'fuel_per_m', 'head', 'rud_trav', 'rud_abs', 'epReward']
WN = re.compile(r"\[woerner-COLREGs\] score=\s*([\d.]+)% \(n=(\d+)")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f(v, nd=1, sign=False):
    if v is None:
        return 'n/a'
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace('-', '−')


def load(fp, seed, label):
    r = S.parse_all(fp)
    assert 'goal' in r and r.get('w_score') is not None, f'미완 또는 Woerner 줄 없음: {fp}'
    s = open(fp, encoding='utf-8', errors='replace').read()
    m = re.search(r'\[run\] (.*)', s)
    assert m and m.group(1).strip() == f'burnin=2400 dec=10000 drain=0 ring=1.0 crossing=0 seed={seed} envs=256 vessels=16 maxp=4 dev=cuda', (label, m and m.group(1))
    r['W_n'] = int(WN.search(s).group(2))
    return r


runs = {p: {s: load(os.path.join(SC, 'out', f'sc_{p}_e{s}.txt'), s, p) for s in SEEDS} for p in POLS}
learned = {f't_{a}': {sd: load(os.path.join(SC, 'out', f'lr_t_{a}_s{sd}.txt'), 999, f't_{a}_s{sd}') for sd in LSEEDS} for a in ('off', 'a6')}
# 학습 배 재채점이 기존 주 평가와 같은 숫자인지(재현) — 끝 줄 비교
REPRO = []
for a in ('off', 'a6'):
    for sd in LSEEDS:
        o = re.search(r'goal=.*?eps', open(os.path.join(T, 'out', f'eval_t_{a}_s{sd}.txt'), encoding='utf-8').read()).group(0)
        n = re.search(r'goal=.*?eps', open(os.path.join(SC, 'out', f'lr_t_{a}_s{sd}.txt'), encoding='utf-8').read()).group(0)
        REPRO.append(o == n)
# 흉내 학생(P0)
stud = {}
for sd in (43, 44, 45, 46, 47):
    r = parse_eval(os.path.join(IM, f'eval_i_dagger_d6_s{sd}.txt'))
    lg = open(os.path.join(IM, f'i_dagger_d6_s{sd}.log'), encoding='utf-8', errors='replace').read()
    last = [l for l in lg.splitlines() if l.startswith('[dagger] dec=')][-1]
    m = re.search(r'agree=\s*([\d.]+)%.*?student eps=\d+ goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%', last)
    r['d_agree'], r['d_goal'], r['d_vcoll'] = float(m.group(1)), float(m.group(2)), float(m.group(3))
    r['p0'] = r['goal'] >= 80.0 and r['vColl'] <= 15.0
    stud[sd] = r
p0_md = open(os.path.join(IM, '_p0.md'), encoding='utf-8').read()
P0_LINE = re.search(r'P0 판정: .*', p0_md).group(0)
n_pass = sum(1 for r in stud.values() if r['p0'])
assert ('통과' in P0_LINE) and (f'{n_pass}/5' in P0_LINE), P0_LINE
print('repro learned:', sum(REPRO), '/', len(REPRO), '| P0:', P0_LINE)


def M(src, p, k):
    return mean([src[p][s].get(k) for s in src[p]])


def pair(src, a, b, k, seeds):
    ok = [(src[a][s].get(k), src[b][s].get(k)) for s in seeds]
    ok = [(x, y) for x, y in ok if x is not None and y is not None]
    win = sum(1 for x, y in ok if (x - y) * SGN[k] > 0)
    return mean([x for x, _ in ok]), mean([y for _, y in ok]), win, len(ok)


ALL = dict(runs, **learned)


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index('<svg'):]


def fig_woerner():
    names = POLS + ['t_off', 't_a6']
    labs = ['goal', 'vo56', 'vo300', 'col56', 'col300', 't_off', 't_a6']
    keys = [('w_score', '전체'), ('w_giveway', '양보선'), ('w_headon', '정면'), ('w_overtaking', '추월'), ('w_standon', '유지선')]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    w = 0.12
    for j, nm in enumerate(names):
        for i, (k, _) in enumerate(keys):
            vals = [ALL[nm][s].get(k) for s in ALL[nm]]
            x = i + (j - 3) * w
            ax.bar(x, mean(vals), width=w * 0.92, color=COL[nm], zorder=2, label=(labs[j] if i == 0 else None),
                   hatch=('////' if nm.startswith('t_') else None), edgecolor=('#555' if nm.startswith('t_') else 'none'), linewidth=0.4)
            ax.scatter([x] * len(vals), vals, s=3, color=INK, zorder=3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([l for _, l in keys], fontsize=7.5)
    ax.set_ylabel('Woerner 준수 %', fontsize=7.5)
    ax.set_ylim(0, 105)
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=6.3, ncol=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    return svg(fig)


def fig_tradeoff():
    # Woerner 준수 vs 배끼리 충돌 — 점 = 시드, 큰 점 = 평균
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for ax, (kx, xl) in zip(axs, (('vColl', '배끼리 충돌 %'), ('rud_trav', '타 이동량 °/ep (도착 ep)'))):
        for nm in POLS + ['t_off', 't_a6']:
            xs = [ALL[nm][s].get(kx) for s in ALL[nm]]
            ys = [ALL[nm][s].get('w_score') for s in ALL[nm]]
            ax.scatter(xs, ys, s=8, color=COL[nm], alpha=0.6, zorder=2)
            ax.scatter([mean(xs)], [mean(ys)], s=40, color=COL[nm], edgecolor=INK, linewidth=0.5, zorder=3,
                       marker=('s' if nm.startswith('t_') else 'o'))
            ax.annotate(nm, (mean(xs), mean(ys)), textcoords='offset points', xytext=((5, -10) if nm == 't_off' else (-30, 5) if nm == 't_a6' else (5, 3)), fontsize=6.5, color=INK)
        ax.set_xlabel(xl, fontsize=7.5)
        ax.set_ylabel('Woerner 준수 %', fontsize=7.5)
        if kx == 'rud_trav':
            ax.set_xscale('log')
    fig.tight_layout()
    return svg(fig)


def tbl(keys, src_names):
    h = ['<table><tr><th>정책</th>' + ''.join(f'<th>{LAB[k]}</th>' for k in keys) + '</tr>']
    for nm in src_names:
        cls = ' class="ref"' if nm.startswith('t_') else ''
        cells = []
        for k in keys:
            vals = [ALL[nm][s].get(k) for s in ALL[nm]]
            mv = mean(vals)
            cells.append(f'<td>{f(mv, ND(k))}<br><span class="sd">{"·".join(f(x, ND(k)) for x in vals) if mv is not None else ""}</span></td>')
        lab = nm + ('<br><span class="sd">학습 배</span>' if nm.startswith('t_') else '')
        h.append(f'<tr{cls}><td>{lab}</td>' + ''.join(cells) + '</tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_pair(src, a, b, seeds, keys=None):
    h = [f'<table><tr><th>지표</th><th>{a}</th><th>{b}</th><th>차</th><th>{a} 승/짝</th></tr>']
    for k in (keys or [m[0] for m in MET]):
        ma, mb, w, n = pair(src, a, b, k, seeds)
        if not n:
            continue
        dd = ma - mb
        tag = 'ok' if w == n else ('bad' if w == 0 and dd else '')
        h.append(f'<tr><td>{LAB[k]}</td><td>{f(ma, ND(k))}</td><td>{f(mb, ND(k))}</td><td>{f(dd, ND(k), True)}</td><td class="{tag}">{w}/{n}</td></tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_p0():
    h = ['<table><tr><th>시드</th><th>도착 %</th><th>배끼리 충돌 %</th><th>벽 %</th><th>시간초과 %</th><th>통과거리 m</th><th>COLREGs(옛) %</th>'
         '<th>연료</th><th>방향 바꾼 총량 °</th><th>DAgger 마지막 창<br>도착 · 충돌 · 타 일치</th><th>P0</th></tr>']
    for sd, r in stud.items():
        h.append(f'<tr><td>s{sd}</td><td>{f(r["goal"])}</td><td>{f(r["vColl"])}</td><td>{f(r["oColl"])}</td><td>{f(r["TO"])}</td>'
                 f'<td>{f(r["dcpa"])}</td><td>{f(r["C"])}</td><td>{f(r["fuel"], 0)}</td><td>{f(r["head"], 0)}</td>'
                 f'<td>{f(r["d_goal"])} · {f(r["d_vcoll"])} · {f(r["d_agree"])} %</td>'
                 f'<td class="{"ok" if r["p0"] else "bad"}">{"통과" if r["p0"] else "불통과"}</td></tr>')
    sm = {k: mean([r[k] for r in stud.values()]) for k in ('goal', 'vColl', 'oColl', 'TO', 'dcpa', 'C', 'fuel', 'head')}
    h.append(f'<tr class="ref"><td>평균</td><td>{f(sm["goal"])}</td><td>{f(sm["vColl"])}</td><td>{f(sm["oColl"])}</td><td>{f(sm["TO"])}</td>'
             f'<td>{f(sm["dcpa"])}</td><td>{f(sm["C"])}</td><td>{f(sm["fuel"], 0)}</td><td>{f(sm["head"], 0)}</td><td></td><td></td></tr>')
    for nm, src in (('선생님 vo56', runs), ('학습 t_off', learned)):
        key = 'vo56' if nm.endswith('vo56') else 't_off'
        h.append(f'<tr class="ref"><td>{nm}</td>' + ''.join(f'<td>{f(M(src, key, k), 0 if k in ("fuel", "head") else 1)}</td>'
                 for k in ('goal', 'vColl', 'oColl', 'TO', 'dcpa', 'C', 'fuel', 'head')) + '<td></td><td></td></tr>')
    h.append('</table>')
    return '\n'.join(h), sm


v = {p: {k: M(ALL, p, k) for k, _, _ in MET} for p in ALL}
p0_tbl, sm = tbl_p0()
wcol = pair(runs, 'col300', 'col56', 'w_score', SEEDS)
wvo = pair(runs, 'vo300', 'vo56', 'w_score', SEEDS)
wl = pair(learned, 't_a6', 't_off', 'w_score', LSEEDS)
role_col = [k for k in W_K[1:] if pair(runs, 'col300', 'col56', k, SEEDS)[2] == 3]
role_vo = [k for k in W_K[1:] if pair(runs, 'vo300', 'vo56', k, SEEDS)[2] == 3]
print('W col300-col56', wcol[2:], 'roles 3/3', len(role_col), '| vo', wvo[2:], len(role_vo), '| t_a6-t_off', wl[2:])

html = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Woerner 재채점 결과</title>
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
.sd {{ color: #777; font-size: 6.1pt; white-space: nowrap; }}
.key {{ border: 2px solid #1b4f8c; background: #f3f8fe; padding: 6px 12px; margin: 8px 0; }}
.note {{ background: #f4f4f4; border-left: 4px solid #888; padding: 6px 10px; margin: 6px 0; font-size: 8.8pt; }}
.bad {{ color: #b00; font-weight: bold; }} .ok {{ color: #070; font-weight: bold; }}
ul {{ margin: 3px 0 3px 18px; padding: 0; }} li {{ margin: 1px 0; }}
.cap {{ font-size: 8pt; color: #444; margin: 2px 0 8px; }}
.pb {{ page-break-before: always; }}
.two {{ display: flex; gap: 10px; }} .two > div {{ flex: 1; }}
svg {{ max-width: 100%; height: auto; }}
code {{ font-size: 8pt; }}
</style></head><body>

<h1>2026-10-02 Woerner COLREGs 재채점(규칙 배 · 학습 배) + 흉내 0단계(P0) 결과</h1>
<div class="sub">재채점 = <code>runs/2026-10-01_scripted/</code> (<code>colregs_woerner.py</code> 추가, 저자 10-02 "ㅇㅋ 진행": Woerner 점수가 다음 배치부터 주 COLREGs 지표) ·
흉내 = <code>runs/2026-10-02_imitation/</code> (스펙 <code>2026-10-02-imitation-init-design.md</code> §5) · 코드 <code>feat/imitation-init</code> 0a3e70e (평가 코드는 feat/reward-v3 와 같음) ·
Windows 2026-10-02 02:30–04:20 · 원자료 <code>out/sc_*.txt</code>·<code>out/lr_t_*.txt</code>·<code>2026-10-02_imitation/out/</code> · 이 문서 = <code>runs/2026-10-01_scripted/report/make_report3.py</code> 가 로그에서 직접 읽어 만듦</div>

<h2>0. 요약</h2>
<div class="key"><ul>
<li><b>보는 거리 300 m 가 Woerner 준수도 올림.</b> col300 vs col56: {f(wcol[1])} → {f(wcol[0])} % ({wcol[2]}/{wcol[3]}, 역할 4개 중 {len(role_col)}개 3/3) ·
vo300 vs vo56: {f(wvo[1])} → {f(wvo[0])} % ({wvo[2]}/{wvo[3]}, 역할 {len(role_vo)}개 3/3). 옛 지표에선 col300 이 0/3 이었음(2차 보고서) — 채점 방식에 따라 결론이 뒤집힘</li>
<li><b>학습 배가 Woerner 는 가장 높음</b>: t_off {f(v['t_off']['w_score'])} · t_a6 {f(v['t_a6']['w_score'])} % vs 규칙 배 최고 col300 {f(v['col300']['w_score'])} %.
그러나 충돌 {f(v['t_off']['vColl'])} · {f(v['t_a6']['vColl'])} %, 타 이동량 {f(v['t_off']['rud_trav'], 0)} · {f(v['t_a6']['rud_trav'], 0)}°/ep(규칙 배 {f(min(v[p]['rud_trav'] for p in POLS), 0)}–{f(max(v[p]['rud_trav'] for p in POLS), 0)}) — 준수 점수와 안전·조타가 따로 놂</li>
<li><b>t_a6 vs t_off (같은 trunk 짝)</b>: Woerner {f(wl[1])} → {f(wl[0])} % ({wl[2]}/{wl[3]}) — 차 {f(wl[0] - wl[1], 1, True)} pp, 사실상 차이 없음</li>
<li><b>흉내 0단계 {P0_LINE.replace("P0 판정: ", "P0 ")}</b>. 학생 평균 도착 {f(sm['goal'])} % · 충돌 {f(sm['vColl'])} % (기준 시드마다 ≥ 80 % · ≤ 15 %).
선생님 vo56 {f(v['vo56']['goal'])} · {f(v['vo56']['vColl'])} %, 학습 t_off {f(v['t_off']['goal'])} · {f(v['t_off']['vColl'])} % 사이. 스펙 §5 대로 재시도 없이 멈춤 — phase1 안 돌림, 다음은 저자 결정</li>
<li>재현: 학습 배 6칸 재채점의 끝 줄(도착·충돌·벽·시간초과·ep 수)이 기존 주 평가와 같음({sum(REPRO)}/{len(REPRO)})</li>
</ul></div>

<h2>1. 무엇을 했나</h2>
<ul>
<li><b>Woerner 채점</b>(<code>colregs_woerner.py</code>, Woerner et al. 2019 Autonomous Robots 43:967-991): 배 쌍마다 조우를 잡아 한쪽 배 입장에서 채점(1 = 완전 준수)
<ul>
<li>조우 시작 = 300 m 안(모든 팔 같음) · 접근 중 · dcpa &lt; 24 m · 역할 있음(env COLREGs 역할). 끝 = CPA 지나 멀어짐 / 1.1 × 300 m 밖 / 종료</li>
<li>점수 항: 안전 S(통과거리 12–24 m), 지연 D(첫 2° 변침 거리), 뚜렷함 A(30° 또는 속력 50 %), 방향 T(좌현 0.5), 자세 P(정면 좌현 대 좌현), 유지선 C·V(침로·속력 유지, 17(b) 늦은 행동 면제)</li>
<li>논문이 비워 둔 값은 재채점 전에 고정(* 표시, 코드 헤더). 옛 지표(레이더 안 실제 타각 비율)는 보조 줄로 남음</li>
</ul></li>
<li>재채점 대상: 규칙 배 5종(goal·vo56·vo300·col56·col300) × 평가 시드 999·1000·1001 + 학습 배 t_off·t_a6 × 학습 시드 43·46·47(체크포인트 그대로, 평가 시드 999). 조건 = t_ 주 평가와 같음(21칸 [run] 줄 확인). vo56h150 은 이번에 안 돌림</li>
<li><b>흉내 0단계</b>: 선생님 vo56 을 DAgger 로 흉내 낸 학생(PPO 전) 5개(시드 43–47)를 주 평가로 채점 → P0 기준(결과 전 고정) 시드마다 도착 ≥ 80 % · 충돌 ≤ 15 %, 3/5 이상이면 phase1</li>
</ul>

<h2>2. Woerner 준수</h2>
{tbl(W_K, POLS + ['t_off', 't_a6'])}
{fig_woerner()}
<div class="cap">그림 1. Woerner 준수 — 전체와 역할 4개. 막대 = 평균, 점 = 시드(규칙 배 평가 시드 3개, 학습 배 학습 시드 3개). 빗금 = 학습 배.</div>

<div class="pb"></div>
<h2>3. 안전 · 옛 COLREGs 지표 · 연료 · 조타</h2>
{tbl(SAFE_K, POLS + ['t_off', 't_a6'])}
{tbl(OLD_K, POLS + ['t_off', 't_a6'])}
{tbl(COST_K, POLS + ['t_off', 't_a6'])}
{fig_tradeoff()}
<div class="cap">그림 2. Woerner 준수 vs 배끼리 충돌(왼쪽)·타 이동량(오른쪽, 로그 축). 작은 점 = 시드, 큰 점 = 평균(□ = 학습 배). 학습 배는 준수 점수가 높아도 충돌·조타가 큼.</div>

<div class="pb"></div>
<h2>4. 짝 비교 (같은 시드끼리)</h2>
<div class="two"><div>
<h3>4-1. col300 vs col56 — 보는 거리 (COLREGs 규칙 배)</h3>
{tbl_pair(runs, 'col300', 'col56', SEEDS)}
</div><div>
<h3>4-2. vo300 vs vo56 — 거리 + H + dt (VO)</h3>
{tbl_pair(runs, 'vo300', 'vo56', SEEDS)}
</div></div>
<h3>4-3. t_a6 vs t_off — 학습 배, 같은 trunk 갈래 (학습 시드 43·46·47)</h3>
{tbl_pair(learned, 't_a6', 't_off', LSEEDS)}
<div class="cap">초록 = 3/3, 빨강 = 0/3 이면서 차 있음. 같으면 승 아님. vo56h150 은 이번에 Woerner 를 안 재서 '보는 거리만 다른' VO 짝(vo300 vs vo56h150)은 Woerner 로 비교 못 함.</div>

<div class="pb"></div>
<h2>5. 흉내 0단계 (P0) — {P0_LINE.replace("P0 판정: ", "")}</h2>
{p0_tbl}
<ul>
<li>학생은 학습 OFF 보다 도착·충돌이 크게 나음(도착 {f(sm['goal'])} vs {f(v['t_off']['goal'])} %) 그러나 선생님(vo56) 수준엔 못 미침 → 5시드 모두 기준 미달</li>
<li>DAgger 마지막 창(학생 운전)에서 도착 80 % 넘은 시드(s43·s46)도 주 평가에선 69–73 % — 두 창은 크기·조건이 다름, 차이 원인은 확인 안 함. 판정은 주 평가로만 함(스펙)</li>
<li>스펙 §5: 불통과면 재시도 없이 보고 → phase1(trunk 학습) 안 돌림. 다음 방향은 저자 결정</li>
</ul>

<h2>6. 읽을 때 주의</h2>
<div class="note"><ul>
<li>규칙 배는 보이는 배의 위치·속도 참값 사용, 300 m 는 척수 제한 없음 → 통신 정보의 상한 모사. 학습 배(레이더 ray + 통신 필드)와 직접 짝 비교 아님(시드 구성도 다름)</li>
<li>Woerner 점수는 조우를 300 m 안 dcpa &lt; 24 m 접근부터 잡음 → 56 m 만 보는 배는 조우 시작 뒤 한참 동안 상대를 못 봄. 이 점수에서 300 m 이 유리한 것은 정보 효과이자 채점 창 설계의 결과이기도 함</li>
<li>조우 수(n)가 정책마다 다름(위험해지는 조우 자체가 행동에 따라 달라짐) — 칸별 n 은 원자료 <code>[woerner-COLREGs]</code> 줄</li>
<li>학습 배 Woerner 가 높은 이유는 확인 안 함(예: 이른 변침·좌현 회피 비율 등 세부 항목은 원자료 <code>[woerner-COLREGs/역할]</code> 줄)</li>
<li>규칙 배 시드 간 산포가 아주 작음(결정론 정책) → 3/3 은 'env 배치 3개에서 일관'의 뜻</li>
</ul></div>

<h2>7. 실행 메모</h2>
<ul>
<li>Dropbox 로 받은 4개 파일 최신 확인(woerner 2·LEARN_ARMS 2) → <code>C:\\work\\DT_Vessel_v3</code> 를 feat/imitation-init 0a3e70e 로 전환(작업 트리 깨끗, 평가 코드 동일) 후 두 배치 동시 실행</li>
<li>재채점 02:30–04:20(21칸, 12칸 동시, 실패 없음) · 흉내 0단계 02:30 smoke → 02:40 DAgger → 03:11 학생 평가 → 03:54 P0 불통과로 멈춤</li>
<li>결과 파일 Dropbox 동기화 표시 확인(재채점 22개, 흉내 out 전부)</li>
</ul>
</body></html>
'''

OUT = sys.argv[1] if len(sys.argv) > 1 else PPP
out_html = os.path.join(OUT, NAME + '.html')
out_pdf = os.path.join(OUT, NAME + '.pdf')
open(out_html, 'w', encoding='utf-8').write(html)
edge = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
subprocess.run([edge, '--headless', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={out_pdf}',
                'file:///' + out_html.replace('\\', '/')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(out_html)
print(out_pdf)
