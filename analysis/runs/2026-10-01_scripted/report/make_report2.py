# 규칙 배 평가 2차(2026-10-01 밤, COLREGs 규칙 배 col56·col300 + vo 재실행) 정리 PDF.
# 숫자는 전부 out/sc_*.txt · ../2026-09-30_t/out/eval_t_*.txt 를 summarize.parse_all(= report_parse.parse_eval + 타·행동 줄)로 읽어 씀.
# 손으로 옮겨 적은 숫자 없음. 그림 = matplotlib SVG, PDF = Edge headless.
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report2.py   (base anaconda = matplotlib 있음)
import glob, io, os, re, subprocess, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True   # Dropbox 폴더에 __pycache__ 안 만듦
HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.normpath(os.path.join(HERE, '..'))                     # runs/2026-10-01_scripted
T = os.path.normpath(os.path.join(SC, '..', '2026-09-30_t'))
PPP = os.path.normpath(os.path.join(SC, '..', '..', '..'))           # Dropbox/Private_Paper_Project
sys.path.insert(0, SC)
sys.path.insert(0, T)
import summarize as S  # noqa: E402  (METRICS·parse_all — _summary.md 와 같은 정의)
from report_parse import parse_eval  # noqa: E402

NAME = '2026-10-02_규칙배2차_COLREGs규칙배_결과'
POLS = ['goal', 'vo56', 'vo56h150', 'vo300', 'col56', 'col300']
SEEDS = [999, 1000, 1001]
COL = {'goal': '#8a8985', 'vo56': '#eb6834', 'vo56h150': '#f2a77f', 'vo300': '#2a78d6', 'col56': '#b5651d', 'col300': '#1b4f8c',
       't_off': '#c9c8c4', 't_a6': '#9fbfe6'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'svg.fonttype': 'none', 'font.size': 8.5,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white'})
MET = S.METRICS
SGN = {k: s for k, _, s in MET}
LAB = {k: l for k, l, _ in MET}
ND = lambda k: S.NDEC.get(k, 1)  # noqa: E731
SAFE_K = ['vColl', 'dcpa', 'minPass', 'goal', 'C', 'C_headon', 'C_standon', 'C_giveway', 'C_overtake', 'rks_v2', 'rks_v2x']
COST_K = ['fuel', 'fleet_fuel', 'fuel_per_m', 'head', 'rud_trav', 'rud_abs', 'TO', 'oColl', 'epReward']


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f(v, nd=1, sign=False):
    if v is None:
        return 'n/a'
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace('-', '−')


runs = {}
for p in POLS:
    for s in SEEDS:
        fp = os.path.join(SC, 'out', f'sc_{p}_e{s}.txt')
        r = S.parse_all(fp)
        assert 'goal' in r, f'미완 칸: {fp}'
        txt = open(fp, encoding='utf-8', errors='replace').read()
        m = re.search(r'\[run\] (.*)', txt)
        assert m and m.group(1).strip() == f'burnin=2400 dec=10000 drain=0 ring=1.0 crossing=0 seed={s} envs=256 vessels=16 maxp=4 dev=cuda', (p, s)
        r['_rudline'] = '[scripted-rudder]' in txt
        runs.setdefault(p, {})[s] = r
learned = {}
for arm in ('off', 'a6'):
    for fp in sorted(glob.glob(os.path.join(T, 'out', f'eval_t_{arm}_s4?.txt'))):
        r = parse_eval(fp)
        if 'goal' in r:
            learned.setdefault(f't_{arm}', {})[int(re.search(r'_s(\d+)\.txt$', fp).group(1))] = r
assert sorted(learned) == ['t_a6', 't_off'] and all(sorted(learned[k]) == [43, 46, 47] for k in learned)
# 재실행한 vo56·vo300 이 1차(out_v1_1853)와 지표 줄이 같은지 확인
REPRO = []
for p in ('vo56', 'vo300'):
    for s in SEEDS:
        a = re.search(r'eps \|\|.*', open(os.path.join(SC, 'out_v1_1853', f'sc_{p}_e{s}.txt'), encoding='utf-8').read()).group(0)
        b = re.search(r'eps \|\|.*', open(os.path.join(SC, 'out', f'sc_{p}_e{s}.txt'), encoding='utf-8').read()).group(0)
        REPRO.append(a == b)
print('repro identical:', sum(REPRO), '/', len(REPRO))


def M(p, k, src=None):
    src = src or runs
    return mean([src[p][s].get(k) for s in src[p]])


def pair(a, b, k):
    ok = [(runs[a][s].get(k), runs[b][s].get(k)) for s in SEEDS]
    ok = [(x, y) for x, y in ok if x is not None and y is not None]
    win = sum(1 for x, y in ok if (x - y) * SGN[k] > 0)
    return mean([x for x, _ in ok]), mean([y for _, y in ok]), win, len(ok)


def W(a, b, k):
    _, _, w, n = pair(a, b, k)
    return f'{w}/{n}'


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index('<svg'):]


def fig_main():
    keys = [('vColl', '배끼리 충돌 %  (낮을수록 좋음)'), ('goal', '도착 %  (높을수록)'), ('dcpa', '통과거리(DCPA) m  (높을수록)'),
            ('C', 'COLREGs 준수 %  (높을수록)'), ('fuel', '연료, 도착 ep  (낮을수록)'), ('head', '방향 바꾼 총량 °, 도착 ep  (낮을수록)')]
    names = POLS + ['t_off', 't_a6']
    xs = [0, 1, 2, 3, 4, 5, 6.6, 7.6]
    labs = ['goal', 'vo56', 'vo56\nh150', 'vo300', 'col\n56', 'col\n300', 't_off\n(참고)', 't_a6\n(참고)']
    fig, axs = plt.subplots(2, 3, figsize=(7.6, 4.6))
    for ax, (k, title) in zip(axs.flat, keys):
        top = 0
        for x, nm in zip(xs, names):
            src = runs if nm in runs else learned
            vals = [src[nm][s].get(k) for s in sorted(src[nm])]
            m = mean(vals)
            ref = nm.startswith('t_')
            ax.bar(x, m, width=0.72, color=COL[nm], zorder=2, edgecolor=('#777' if ref else 'none'),
                   hatch=('////' if ref else None), linewidth=0.5)
            ax.scatter([x] * len(vals), vals, s=6, color=INK, zorder=3)
            ax.annotate(f(m, 0 if k in ('fuel', 'head') else 1), (x, max(vals)), textcoords='offset points',
                        xytext=(0, 2), ha='center', fontsize=5.6, color=INK)
            top = max(top, max(vals))
        ax.axvline(5.8, color='#bbb', lw=0.8, ls=':')
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=5.8)
        ax.set_title(title, fontsize=7.6, color=INK)
        ax.grid(axis='x', visible=False)
        ax.set_ylim(0, top * 1.15)
    fig.tight_layout(h_pad=1.2)
    return svg(fig)


def fig_pairs():
    # 정보 효과(300 vs 56 m) 두 규칙 배 나란히: 지표 방향으로 부호 맞춘 상대 차(+ = 300 m 이 나음), 괄호 = 절대 차
    keys = [('vColl', '충돌', ' pp', 1), ('goal', '도착', ' pp', 1), ('dcpa', '통과거리', ' m', 1), ('C', 'COLREGs', ' pp', 1),
            ('rks_v2', '역할 판정기', ' pp', 1), ('fuel', '연료(도착 ep)', '', 0), ('fleet_fuel', '함대 연료\n/도착', '', 0),
            ('head', '방향 바꾼\n총량', '°', 0)]
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    w = 0.38
    for j, (a, b, c, lab) in enumerate((('vo300', 'vo56h150', '#2a78d6', 'VO: vo300 − vo56h150'),
                                        ('col300', 'col56', '#1b4f8c', 'COLREGs 규칙: col300 − col56'))):
        for i, (k, _, unit, nd) in enumerate(keys):
            rel = [SGN[k] * (runs[a][s][k] - runs[b][s][k]) / abs(runs[b][s][k]) * 100 for s in SEEDS]
            m = mean(rel)
            x = i + (j - 0.5) * w
            ax.bar(x, m, width=w * 0.92, color=c, alpha=(1.0 if m > 0 else 0.55), zorder=2,
                   edgecolor=('#c0392b' if m < 0 else 'none'), linewidth=0.8, label=(lab if i == 0 else None))
            ax.annotate(f'{f(m, 0, True)}%\n({f(M(a, k) - M(b, k), nd, True)}{unit})', (x, m), textcoords='offset points',
                        xytext=(0, 2 if m >= 0 else -17), ha='center', fontsize=5.2)
    ax.axhline(0, color='#888', lw=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([lab for _, lab, _, _ in keys], fontsize=6.8)
    ax.set_ylabel('상대 개선 %', fontsize=7)
    ax.grid(axis='x', visible=False)
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_ * 1.35, hi_ * 1.2)
    ax.legend(fontsize=6.5, frameon=False, loc='upper right')
    fig.tight_layout()
    return svg(fig)


def tbl(keys, title_pol='정책'):
    h = [f'<table><tr><th>{title_pol}</th>' + ''.join(f'<th>{LAB[k]}</th>' for k in keys) + '</tr>']
    for nm in POLS + ['t_off', 't_a6']:
        src = runs if nm in runs else learned
        cls = ' class="ref"' if nm.startswith('t_') else ''
        cells = []
        for k in keys:
            vals = [src[nm][s].get(k) for s in sorted(src[nm])]
            mv = mean(vals)
            cells.append(f'<td>{f(mv, ND(k))}<br><span class="sd">{"·".join(f(x, ND(k)) for x in vals) if mv is not None else ""}</span></td>')
        lab = nm + ('<br><span class="sd">학습 배, 참고</span>' if nm.startswith('t_') else '')
        h.append(f'<tr{cls}><td>{lab}</td>' + ''.join(cells) + '</tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_pair(a, b):
    h = [f'<table><tr><th>지표</th><th>{a}</th><th>{b}</th><th>차 ({a} − {b})</th><th>{a} 승/짝</th></tr>']
    for k, lab, _ in MET:
        ma, mb, w, n = pair(a, b, k)
        if not n:
            h.append(f'<tr><td>{lab}</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td></tr>')
            continue
        dd = ma - mb
        tag = 'ok' if w == n else ('bad' if w == 0 and dd else '')
        h.append(f'<tr><td>{lab}</td><td>{f(ma, ND(k))}</td><td>{f(mb, ND(k))}</td><td>{f(dd, ND(k), True)}</td><td class="{tag}">{w}/{n}</td></tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_wins():
    # 짝 6개 × 지표: 승/짝 한눈표
    prs = [p[:2] for p in S.PAIRS]
    h = ['<table><tr><th>지표</th>' + ''.join(f'<th>{a}<br>vs {b}</th>' for a, b in prs) + '</tr>']
    for k, lab, _ in MET:
        cells = []
        for a, b in prs:
            _, _, w, n = pair(a, b, k)
            cls = 'ok' if n and w == n else ('bad' if n and w == 0 and (M(a, k) - M(b, k)) else '')
            cells.append(f'<td class="{cls}" style="text-align:center">{w}/{n}</td>' if n else '<td style="text-align:center">n/a</td>')
        h.append(f'<tr><td>{lab}</td>' + ''.join(cells) + '</tr>')
    h.append('</table>')
    return '\n'.join(h)


def mode_rows():
    rows = []
    for nm in ('col56', 'col300'):
        for s in SEEDS:
            rows.append(f'<tr><td>{nm} e{s}</td><td style="text-align:left">{runs[nm][s].get("mode")}</td></tr>')
    return ''.join(rows)


v = {p: {k: M(p, k) for k, _, _ in MET} for p in POLS}
lo = {p: {k: M(p, k, learned) for k, _, _ in MET} for p in learned}
safe4 = ('vColl', 'goal', 'dcpa', 'minPass')
c_sit = ('C_headon', 'C_standon', 'C_giveway', 'C_overtake')
n_safe_col = sum(1 for k in safe4 if pair('col300', 'col56', k)[2] == 3)
n_safe_vo = sum(1 for k in safe4 if pair('vo300', 'vo56h150', k)[2] == 3)
n_csit_col = sum(1 for k in c_sit if pair('col300', 'col56', k)[2] == 0)
n_csit_c300 = sum(1 for k in c_sit if pair('col300', 'vo300', k)[2] == 0)
rud_note = ', '.join(p for p in POLS if not all(runs[p][s]['_rudline'] for s in SEEDS))
print('safe col', n_safe_col, 'vo', n_safe_vo, '| C sit 0/3 col300-col56', n_csit_col, 'col300-vo300', n_csit_c300, '| no rudder line:', rud_note)

html = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>규칙 배 2차 결과</title>
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
tr.ref td {{ background: #f6f6f6; color: #555; }}
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

<h1>2026-10-01 규칙 배 평가 2차 — COLREGs 규칙 배(col56 · col300) + 타 지표</h1>
<div class="sub">스크립트 <code>runs/2026-10-01_scripted/</code> (<code>eval_scripted.py</code> · <code>colregs_rule.py</code> · <code>_run_scripted.sh</code> · <code>summarize.py</code>, 저자 2차 지시 "COLREGs 규칙 배 + DCPA·fuel·rudder 점검") ·
코드 <code>feat/reward-v3</code> 3aa0636 · 평가기 <code>eval/eval_ckpt.py</code> 그대로 · Windows 실행 2026-10-01 22:38–23:22 (12칸 동시) ·
원자료 <code>out/sc_*.txt</code>, 표 <code>_summary.md</code> · 1차 결과·PDF = <code>out_v1_1853/</code>, <code>2026-10-01_규칙배_평가결과.pdf</code> ·
이 문서 = <code>report/make_report2.py</code> 가 로그에서 직접 읽어 만듦</div>

<h2>0. 요약</h2>
<div class="key"><ul>
<li><b>COLREGs 규칙 배도 300 m 정보로 안전이 좋아짐.</b> col300 vs col56(규칙·H·dt 같음, 보는 거리만 다름) — 충돌·도착·통과거리·조우 최소통과 {n_safe_col}/4 지표 3/3:
충돌 {f(v['col56']['vColl'])} → {f(v['col300']['vColl'])} %, 도착 {f(v['col56']['goal'])} → {f(v['col300']['goal'])} %, 통과거리 {f(v['col56']['dcpa'])} → {f(v['col300']['dcpa'])} m, 역할 판정기 {f(v['col56']['rks_v2'])} → {f(v['col300']['rks_v2'])} % ({W('col300', 'col56', 'rks_v2')})</li>
<li><b>그러나 COLREGs 준수는 300 m 쪽이 낮음 (0/3).</b> {f(v['col56']['C'])} → {f(v['col300']['C'])} %, 상황별 4개 중 {n_csit_col}개가 0/3
(HeadOn {f(v['col56']['C_headon'])} → {f(v['col300']['C_headon'])}, GiveWay {f(v['col56']['C_giveway'])} → {f(v['col300']['C_giveway'])}). 원인 미확인(§6)</li>
<li><b>연료·조타는 300 m 쪽이 더 씀 (0/3)</b>: 도착 ep 연료 {f(v['col56']['fuel'], 0)} → {f(v['col300']['fuel'], 0)}, 방향 바꾼 총량 {f(v['col56']['head'], 0)} → {f(v['col300']['head'], 0)}°,
타 이동량 {f(v['col56']['rud_trav'], 0)} → {f(v['col300']['rud_trav'], 0)}°/ep, 평균 |타각| {f(v['col56']['rud_abs'], 2)} → {f(v['col300']['rud_abs'], 2)}°.
도착 1회당 함대 연료만 {f(v['col56']['fleet_fuel'], 0)} → {f(v['col300']['fleet_fuel'], 0)} ({W('col300', 'col56', 'fleet_fuel')})</li>
<li><b>COLREGs 규칙을 넣으면 안전이 나빠짐.</b> 같은 거리 VO 대비 충돌·도착 0/3 — 56 m: 충돌 {f(v['vo56h150']['vColl'])} → {f(v['col56']['vColl'])} %, 도착 {f(v['vo56h150']['goal'])} → {f(v['col56']['goal'])} % (COLREGs 준수는 {f(v['vo56h150']['C'])} → {f(v['col56']['C'])} %, 3/3) ·
300 m: 충돌 {f(v['vo300']['vColl'])} → {f(v['col300']['vColl'])} %, 도착 {f(v['vo300']['goal'])} → {f(v['col300']['goal'])} %</li>
<li><b>모든 지표에서 이기는 규칙 배 없음</b> — 정보 효과(300 m)는 두 규칙 모두 충돌·도착·통과거리 3/3, 연료(도착 ep)·방향 바꾼 총량 0/3</li>
<li>재현: 재실행한 vo56·vo300 6칸의 지표 줄이 1차와 글자까지 같음({sum(REPRO)}/{len(REPRO)})</li>
</ul></div>

<h2>1. 무엇을 했나</h2>
<ul>
<li>1차(VO 규칙 배 4종)에 저자 2차 지시로 <b>COLREGs 규칙 배 col56·col300</b> 추가 + 타 지표 줄(<code>[scripted-rudder]</code>: 도착 ep 타 이동량·평균 |타각|) 추가. vo56·vo300 은 타 지표 때문에 재실행, goal·vo56h150 은 1차 값 그대로(타 지표 n/a)</li>
<li>방법은 1차와 같음: <code>eval_ckpt.main()</code> 그대로, 정책만 규칙 배로 교체, env = t_ 스냅샷, 가중치 미사용. 평가 <code>--envs 256 --eval_decisions 10000 --burnin 2400</code>, drain 0, 평가 시드 999·1000·1001 (18칸 로그 [run] 줄로 확인)</li>
<li><b>col 규칙</b>(<code>colregs_rule.py</code>): 역할 = env 의 COLREGs 코드(<code>RolePromiseTrackerV2.roles</code>, 보이는 배 R 안). 후보 침로·비용 = VO 와 같음. 배마다 우선순위:
<ul>
<li>피항(보이는 배 dcpa &lt; 24 m·접근, 내 역할 head-on/give-way/overtaking) → 우현만, 15° 이상</li>
<li>유지선 행동(위험 상대의 tcpa ≤ RULE_17B_TIME) → 우현 쪽만 · 유지선(그 전) → 타 0 = 침로 유지</li>
<li>피항 기하인데 위험 아님(dcpa ≥ 24 m) → 좌현 변침 금지 · 그 밖 → VO 전체</li>
</ul></li>
<li>col56 · col300 은 H 150 s · dt 4 s 같음, R 만 56 / 300 m. 추력 항상 1. 보이는 배의 위치·속도 참값 사용</li>
</ul>

<h2>2. 결과 — 안전·준수</h2>
<div class="cap">평균, 작은 글씨 = 시드별(규칙 배 999·1000·1001, 학습 배 43·46·47). 회색 = 학습 배 t_ 주 평가(평가 시드 999), 참고용.</div>
{tbl(SAFE_K)}
<h2>3. 결과 — 연료·조타·기타</h2>
{tbl(COST_K)}
<div class="cap">타 이동량·평균 |타각| 은 2차에서 새로 넣은 줄이라 goal·vo56h150(1차 값 사용)과 학습 배는 n/a.</div>

<div class="pb"></div>
<h2>4. 그림</h2>
{fig_main()}
<div class="cap">그림 1. 규칙 배 6개(막대 = 평균, 점 = 평가 시드 3개)와 학습 배 참고 2개(빗금). 점선 오른쪽은 입력 정보·시드 구성이 다른 참고값.</div>
{fig_pairs()}
<div class="cap">그림 2. 보는 거리 300 m − 56 m 효과, 규칙별(VO · COLREGs 규칙). 막대 = 지표 방향으로 부호 맞춘 상대 차 %(+ = 300 m 이 나음), 괄호 = 절대 차.
상대 %는 기저가 작은 지표(충돌)에서 커 보임. 빨간 테 = 300 m 쪽이 나쁨.</div>

<h2>5. 짝 비교</h2>
<h3>5-1. 승/짝 한눈표 (같은 평가 시드 3짝, 앞 정책 기준)</h3>
{tbl_wins()}
<div class="cap">초록 = 3/3, 빨강 = 0/3 이면서 차 있음. 같으면 승 아님(벽 충돌 0.0 = 0.0 → 0/3). 시간초과 차(0.1–0.2 pp)는 로그 출력 해상도 수준.</div>
<div class="pb"></div>
<div class="two"><div>
<h3>5-2. col300 vs col56 — 보는 거리 효과 (COLREGs 규칙)</h3>
{tbl_pair('col300', 'col56')}
</div><div>
<h3>5-3. col300 vs vo300 — 300 m 에서 COLREGs 규칙 추가 효과</h3>
{tbl_pair('col300', 'vo300')}
</div></div>
<div class="two"><div>
<h3>5-4. col56 vs vo56h150 — 56 m 에서 COLREGs 규칙 추가 효과</h3>
{tbl_pair('col56', 'vo56h150')}
</div><div>
<h3>5-5. vo300 vs vo56h150 — 보는 거리 효과 (VO, 1차와 같음)</h3>
{tbl_pair('vo300', 'vo56h150')}
</div></div>

<h2>6. COLREGs 규칙 배 행동 비율 (burn-in 뒤 배-결정, n = 40,960,000)</h2>
<table><tr><th>칸</th><th>행동 비율</th></tr>{mode_rows()}</table>
<ul>
<li>col300 은 56 m 밖까지 역할을 알아서 '피항 기하(좌현 금지)'·'피항'·'유지' 규칙이 걸리는 결정이 많고(VO 그대로 {runs['col300'][999]['mode'].split()[0]}), col56 은 대부분 VO 그대로({runs['col56'][999]['mode'].split()[0]})</li>
<li>col300 의 COLREGs 준수 하락(특히 HeadOn·GiveWay)과 이 행동 비율의 관계는 <b>확인 안 함</b>. 준수 지표(<code>[encounter-COLREGs/실제타각]</code>)가 조우를 어떤 시점·거리에서 채점하는지와 300 m 에서 미리 움직인 결과가 어긋날 가능성이 있으나 가설임</li>
</ul>

<h2>7. 읽을 때 주의</h2>
<div class="note"><ul>
<li><b>규칙 배는 보이는 배의 위치·속도 참값을 씀</b>(레이더 ray 아님). col300·vo300 은 300 m 안 <b>모든 배</b>(척수 제한 없음) → 학습 통신(가까운 4척·latent)이 줄 수 있는 정보의 상한에 가까운 모사이지 통신 자체가 아님</li>
<li><b>이 COLREGs 규칙은 한 가지 설계</b>(피항 기준 dcpa 24 m, 우현 15° 이상, 유지선 타 0 등). 안전이 나빠진 결과를 'COLREGs 를 지키면 위험해진다'로 일반화하지 않음 — 이 규칙 설계의 결과임</li>
<li>학습 배 줄은 직접 짝 비교 아님(입력·시드 구성·창 절단 비율 다름 — 1차 보고서 §5·§6)</li>
<li>시드 간 산포가 아주 작음(결정론 정책, env 배치만 다름) → 3/3 은 'env 배치 3개에서 일관'의 뜻</li>
<li>epReward 는 끝난 ep 만 셈(drain 0) → 보조로만</li>
<li>연료 지표 3개는 방향이 갈릴 수 있음: 도착 ep 연료(도착한 ep 만) vs 도착 1회당 함대 연료(충돌·시간초과 ep 포함) vs 진행 1 m 당 연료</li>
</ul></div>

<h2>8. 실행 메모</h2>
<ul>
<li>21:01 첫 실행 → 21:12 <b>서버 비정상 종료</b>(블루스크린 0x133 DPC_WATCHDOG_VIOLATION, 21:14 재부팅). 직전 21:06 네트워크 카드(Intel X710, i40ea) 리셋·21:09 OVRService 시간 초과 기록. 원인 미확인(덤프 <code>C:\\WINDOWS\\Minidump\\100126-20171-01.dmp</code>, 관리자 권한 필요)</li>
<li>22:38 재실행(12칸 동시, 서버 빈 상태) → 23:22 끝. 실패 칸 없음. 1차 결과는 <code>out_v1_1853/</code> 에 백업 후 vo56·vo300 덮어씀(지표 동일)</li>
<li>Dropbox: OSH 클라이언트가 자주 꺼짐 → 감시 작업 <code>osh_dropbox_watchdog</code>(5분마다, Explorer 경유 재시작) 운영 중. 결과 19개 파일 동기화 표시 확인</li>
</ul>
</body></html>
'''

out_html = os.path.join(PPP, NAME + '.html')
out_pdf = os.path.join(PPP, NAME + '.pdf')
open(out_html, 'w', encoding='utf-8').write(html)
edge = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
subprocess.run([edge, '--headless', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={out_pdf}',
                'file:///' + out_html.replace('\\', '/')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(out_html)
print(out_pdf)
