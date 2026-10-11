# 규칙 배 평가(2026-10-01) 정리 PDF. 숫자는 전부 out/sc_*.txt · ../2026-09-30_t/out/eval_t_*.txt 를
# report_parse.parse_eval(+ 아래 extra 정규식)로 읽어 씀(손으로 옮겨 적은 숫자 없음). 그림 = matplotlib SVG, PDF = Edge headless.
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report.py   (base anaconda = matplotlib 있음)
# 2026-10-01 19:4x 개정: 읽기 전용 감사(숫자·표현 2관점) 지적 반영 — 문구·제목·주의 사항만, 숫자 출처 동일
import glob, io, os, re, subprocess, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.normpath(os.path.join(HERE, '..'))                     # runs/2026-10-01_scripted
T = os.path.normpath(os.path.join(SC, '..', '2026-09-30_t'))
PPP = os.path.normpath(os.path.join(SC, '..', '..', '..'))           # Dropbox/Private_Paper_Project
sys.path.insert(0, T)
from report_parse import parse_eval  # noqa: E402

NAME = '2026-10-01_규칙배_평가결과'
POLS = ['goal', 'vo56', 'vo56h150', 'vo300']
SEEDS = [999, 1000, 1001]
DESC = {'goal': ('목표로 직진, 안 피함', '—', '—', '—', '하한 기준'),
        'vo56': ('VO 회피', '56 m 안 모든 배 (레이더 범위)', '60 s', '2 s', '= 순위 검사 vo56'),
        'vo56h150': ('VO 회피', '56 m 안 모든 배 (레이더 범위)', '150 s', '4 s', 'vo300 과 보는 거리만 다름'),
        'vo300': ('VO 회피', '300 m 안 모든 배 (통신 반경, 척수 제한 없음)', '150 s', '4 s', '= 순위 검사 vo300')}
COL = {'goal': '#8a8985', 'vo56': '#eb6834', 'vo56h150': '#f2a77f', 'vo300': '#2a78d6', 't_off': '#c9c8c4', 't_a6': '#9fbfe6'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'svg.fonttype': 'none', 'font.size': 8.5,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white'})

# (키, 라벨, 방향 +1 높을수록 좋음 / −1 낮을수록 좋음, 소수 자리)
MET = [('vColl', '배끼리 충돌 %', -1, 1), ('goal', '도착 %', 1, 1), ('dcpa', '통과거리 m', 1, 1), ('C', 'COLREGs 준수 %', 1, 1),
       ('rks_v2', '역할 판정기 %', 1, 1), ('fuel', '연료 (도착 ep)', -1, 0), ('head', '방향 바꾼 총량 ° (도착 ep)', -1, 0),
       ('TO', '시간초과 %', -1, 1), ('oColl', '벽 충돌 %', -1, 1), ('epReward', 'epReward', 1, 0)]
# 보조: (키, 라벨, 방향, 소수)
SUB = [('C_headon', 'HeadOn C %', 1, 1), ('C_standon', 'StandOn C %', 1, 1), ('C_giveway', 'GiveWay C %', 1, 1),
       ('C_overtake', 'Overtake C %', 1, 1), ('noturn', '무변침 조우 %', 0, 1), ('minPass', '조우 최소통과 m', 1, 1),
       ('len', '도착 ep 길이(결정)', -1, 0), ('fuel_per_m', '진행 1 m 당 연료', -1, 3), ('fleet_fuel', '도착 1회당 함대 연료', -1, 0),
       ('rks_v2x', '역할 판정기(해소 제외) %', 1, 1), ('resolved_pct', '해소 비율 %', 0, 1), ('eps', '끝난 ep 수', 0, 0),
       ('cens_pct', '창 절단 미완 ep %', 0, 1)]
SGN = {k: s for k, _, s, _ in MET + SUB}


def extra(fp, r):
    """report_parse 에 없는 줄: 무변침 조우, 창 절단 미완 ep, 해소 비율."""
    s = open(fp, encoding='utf-8', errors='replace').read()
    m = re.search(r'무변침 조우\s*([\d.]+)%', s)
    r['noturn'] = float(m.group(1)) if m else None
    m = re.search(r'\[censored\] 창 끝 미완 에피소드 (\d+)개', s)
    r['cens_pct'] = 100.0 * int(m.group(1)) / (int(m.group(1)) + r['eps']) if m and r.get('eps') else None
    r['resolved_pct'] = 100.0 * r['resolved_v2'] / r['rp2_n'] if r.get('resolved_v2') is not None and r.get('rp2_n') else None
    m = re.search(r'\[run\] (.*)', s)
    r['_run'] = m.group(1).strip() if m else ''
    return r


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
        r = parse_eval(fp)
        assert 'goal' in r, f'미완 칸: {fp}'
        runs.setdefault(p, {})[s] = extra(fp, r)
learned = {}
for arm in ('off', 'a6'):
    for fp in sorted(glob.glob(os.path.join(T, 'out', f'eval_t_{arm}_s4?.txt'))):
        r = parse_eval(fp)
        if 'goal' in r:
            learned.setdefault(f't_{arm}', {})[int(re.search(r'_s(\d+)\.txt$', fp).group(1))] = extra(fp, r)
assert all(len(runs[p]) == 3 for p in POLS)
for p in POLS:   # 12칸 모두 t_ 주 평가와 같은 창인지 로그로 확인
    for s in SEEDS:
        h = runs[p][s]['_run']
        assert h == f'burnin=2400 dec=10000 drain=0 ring=1.0 crossing=0 seed={s} envs=256 vessels=16 maxp=4 dev=cuda', (p, s, h)
assert sorted(learned) == ['t_a6', 't_off'] and all(sorted(learned[k]) == [43, 46, 47] for k in learned)


def M(p, k, src=None):
    src = src or runs
    return mean([src[p][s].get(k) for s in src[p]])


def pair(a, b, k):
    ok = [(runs[a][s].get(k), runs[b][s].get(k)) for s in SEEDS]
    ok = [(x, y) for x, y in ok if x is not None and y is not None]
    win = sum(1 for x, y in ok if (x - y) * SGN[k] > 0)
    return mean([x for x, _ in ok]), mean([y for _, y in ok]), win, len(ok)


def wl(a, b, k, better='낮음'):
    """'a 가 b 보다 k 가 {better} n/3' 용 짝 수 (방향 무시, 단순 비교)."""
    n = sum(1 for s in SEEDS if (runs[a][s][k] < runs[b][s][k]) == (better == '낮음') and runs[a][s][k] != runs[b][s][k])
    return f'{n}/3'


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index('<svg'):]


def fig_main():
    keys = [('vColl', '배끼리 충돌 %  (낮을수록 좋음)'), ('goal', '도착 %  (높을수록)'), ('dcpa', '통과거리 m  (높을수록)'),
            ('C', 'COLREGs 준수 %  (높을수록)'), ('fuel', '연료, 도착 ep  (낮을수록)'), ('head', '방향 바꾼 총량 °, 도착 ep  (낮을수록)')]
    names = POLS + ['t_off', 't_a6']
    xs = [0, 1, 2, 3, 4.6, 5.6]
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.4))
    for ax, (k, title) in zip(axs.flat, keys):
        top = 0
        for x, nm in zip(xs, names):
            src = runs if nm in runs else learned
            vals = [src[nm][s].get(k) for s in sorted(src[nm])]
            m = mean(vals)
            ref = nm.startswith('t_')
            ax.bar(x, m, width=0.7, color=COL[nm], zorder=2, edgecolor=('#777' if ref else 'none'),
                   hatch=('////' if ref else None), linewidth=0.5)
            ax.scatter([x] * len(vals), vals, s=7, color=INK, zorder=3)
            ax.annotate(f(m, 0 if k in ('fuel', 'head') else 1), (x, max(vals)), textcoords='offset points',
                        xytext=(0, 2), ha='center', fontsize=6.3, color=INK)
            top = max(top, max(vals))
        ax.axvline(3.8, color='#bbb', lw=0.8, ls=':')
        ax.set_xticks(xs)
        ax.set_xticklabels(['goal', 'vo56', 'vo56\nh150', 'vo300', 't_off\n(참고)', 't_a6\n(참고)'], fontsize=6.6)
        ax.set_title(title, fontsize=7.8, color=INK)
        ax.grid(axis='x', visible=False)
        ax.set_ylim(0, top * 1.15)
    fig.tight_layout(h_pad=1.2)
    return svg(fig)


def fig_pairs():
    # vo300 − vo56h150 (보는 거리만 다름) 시드별 상대 차, 지표 방향으로 부호 맞춤(+ = vo300 이 나음). 막대 위 = 절대 차
    keys = [('vColl', '충돌', ' pp', 1), ('goal', '도착', ' pp', 1), ('dcpa', '통과거리', ' m', 1), ('C', 'COLREGs', ' pp', 1),
            ('rks_v2', '역할 판정기', ' pp', 1), ('fuel', '연료(도착 ep)', '', 0), ('head', '방향 바꾼 총량', '°', 0),
            ('fleet_fuel', '도착 1회당\n함대 연료', '', 0)]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    for i, (k, lab, unit, nd) in enumerate(keys):
        rel = [SGN[k] * (runs['vo300'][s][k] - runs['vo56h150'][s][k]) / abs(runs['vo56h150'][s][k]) * 100 for s in SEEDS]
        m = mean(rel)
        dabs = M('vo300', k) - M('vo56h150', k)
        ax.bar(i, m, width=0.6, color=('#2a78d6' if m > 0 else '#c0392b'), zorder=2)
        ax.scatter([i] * 3, rel, s=8, color=INK, zorder=3)
        ax.annotate(f'{f(m, 1, True)} %\n({f(dabs, nd, True)}{unit})', (i, m), textcoords='offset points',
                    xytext=(0, 3 if m >= 0 else -20), ha='center', fontsize=6.4)
    ax.axhline(0, color='#888', lw=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([lab for _, lab, _, _ in keys], fontsize=7.0)
    ax.set_ylabel('상대 개선 %', fontsize=7)
    ax.grid(axis='x', visible=False)
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_ * 1.6, hi_ * 1.2)
    fig.tight_layout()
    return svg(fig)


def tbl_main():
    h = ['<table><tr><th>정책</th>' + ''.join(f'<th>{lab}</th>' for _, lab, _, _ in MET) + '</tr>']
    for nm in POLS + ['t_off', 't_a6']:
        src = runs if nm in runs else learned
        cls = ' class="ref"' if nm.startswith('t_') else ''
        cells = []
        for k, _, _, nd in MET:
            vals = [src[nm][s].get(k) for s in sorted(src[nm])]
            cells.append(f'<td>{f(mean(vals), nd)}<br><span class="sd">{"·".join(f(x, nd) for x in vals)}</span></td>')
        lab = nm + ('<br><span class="sd">학습 배, 참고</span>' if nm.startswith('t_') else '')
        h.append(f'<tr{cls}><td>{lab}</td>' + ''.join(cells) + '</tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_pair(a, b, keys=None):
    h = [f'<table><tr><th>지표</th><th>{a} 평균</th><th>{b} 평균</th><th>차 ({a} − {b})</th><th>{a} 승/짝</th></tr>']
    for k, lab, _, nd in (keys or MET):
        ma, mb, w, n = pair(a, b, k)
        dd = None if ma is None or mb is None else ma - mb
        tag = 'ok' if n and w == n else ('bad' if n and w == 0 and dd else '')
        h.append(f'<tr><td>{lab}</td><td>{f(ma, nd)}</td><td>{f(mb, nd)}</td><td>{f(dd, nd, True)}</td>'
                 f'<td class="{tag}">{w}/{n}</td></tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_sub():
    h = ['<table><tr><th>정책</th>' + ''.join(f'<th>{lab}</th>' for _, lab, _, _ in SUB) + '</tr>']
    for nm in POLS + ['t_off', 't_a6']:
        src = runs if nm in runs else learned
        cls = ' class="ref"' if nm.startswith('t_') else ''
        h.append(f'<tr{cls}><td>{nm}</td>' + ''.join(f'<td>{f(M(nm, k, src), nd)}</td>' for k, _, _, nd in SUB) + '</tr>')
    h.append('</table>')
    return '\n'.join(h)


v = {p: {k: M(p, k) for k in [x[0] for x in MET + SUB]} for p in POLS}
lo = {p: {k: M(p, k, learned) for k in [x[0] for x in MET + SUB]} for p in learned}
SAFE = ('vColl', 'goal', 'dcpa')
RULE = ('C', 'rks_v2')
w_safe = [k for k in SAFE if pair('vo300', 'vo56h150', k)[2] == 3]
w_rule = [k for k in RULE if pair('vo300', 'vo56h150', k)[2] == 3]
fuel_rel = (v['vo300']['fuel'] - v['vo56h150']['fuel']) / v['vo56h150']['fuel'] * 100
head_rel = (v['vo300']['head'] - v['vo56h150']['head']) / v['vo56h150']['head'] * 100
print('safe 3/3:', w_safe, '| rule 3/3:', w_rule)
for k in ('C_standon', 'C_giveway', 'C_headon', 'fleet_fuel', 'fuel_per_m', 'len', 'noturn', 'rks_v2x'):
    print(k, pair('vo300', 'vo56h150', k)[2:], f(M('vo300', k), 3), f(M('vo56h150', k), 3))
desc_rows = ''.join(f'<tr><td>{p}</td>' + ''.join(f'<td style="text-align:left">{x}</td>' for x in DESC[p]) + '</tr>' for p in POLS)
# 짝 수(설명 문장용)
n_sto = pair('vo300', 'vo56h150', 'C_standon')[2]
n_gw = sum(1 for s in SEEDS if runs['vo300'][s]['C_giveway'] < runs['vo56h150'][s]['C_giveway'])
n_ho = sum(1 for s in SEEDS if runs['vo300'][s]['C_headon'] < runs['vo56h150'][s]['C_headon'])
n_ff = pair('vo300', 'vo56h150', 'fleet_fuel')[2]
n_fm = pair('vo300', 'vo56h150', 'fuel_per_m')[2]
n_fm56 = pair('vo300', 'vo56', 'fuel_per_m')[2]
n_len = sum(1 for s in SEEDS if runs['vo300'][s]['len'] > runs['vo56h150'][s]['len'])
n_nt = sum(1 for s in SEEDS if runs['vo300'][s]['noturn'] > runs['vo56h150'][s]['noturn'])
n_x = pair('vo300', 'vo56h150', 'rks_v2x')[2]
DUR = '칸당 38–77분(goal 38–54 · vo300 51–72 · vo56 69–70 · vo56h150 72–77)'

html = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>규칙 배 평가 결과</title>
<style>
@page {{ size: A4; margin: 13mm 12mm; }}
body {{ font-family: "Malgun Gothic", sans-serif; font-size: 9.6pt; line-height: 1.5; color: #111; background: #fff; }}
h1 {{ font-size: 14.5pt; margin: 0 0 4px; }}
h2 {{ font-size: 12pt; margin: 16px 0 6px; border-bottom: 2px solid #333; padding-bottom: 2px; }}
h3 {{ font-size: 10.2pt; margin: 10px 0 3px; }}
.sub {{ color: #555; font-size: 8.4pt; margin-bottom: 8px; }}
table {{ border-collapse: collapse; margin: 6px 0 8px; font-size: 8.2pt; }}
th, td {{ border: 1px solid #999; padding: 2px 5px; text-align: right; vertical-align: top; }}
th {{ background: #eee; text-align: center; }} td:first-child, th:first-child {{ text-align: left; }}
tr.ref td {{ background: #f6f6f6; color: #555; }}
.sd {{ color: #777; font-size: 6.4pt; white-space: nowrap; }}
.key {{ border: 2px solid #2a78d6; background: #f3f8fe; padding: 6px 12px; margin: 8px 0; }}
.note {{ background: #f4f4f4; border-left: 4px solid #888; padding: 6px 10px; margin: 6px 0; font-size: 8.9pt; }}
.bad {{ color: #b00; font-weight: bold; }} .ok {{ color: #070; font-weight: bold; }}
ul {{ margin: 3px 0 3px 18px; padding: 0; }} li {{ margin: 1px 0; }}
.cap {{ font-size: 8.1pt; color: #444; margin: 2px 0 8px; }}
.pb {{ page-break-before: always; }}
svg {{ max-width: 100%; height: auto; }}
code {{ font-size: 8.1pt; }}
</style></head><body>

<h1>2026-10-01 규칙 배 평가 결과 — 300 m 참값 정보가 학습 없이도 도움이 되나</h1>
<div class="sub">스크립트 <code>runs/2026-10-01_scripted/</code> (<code>eval_scripted.py</code> · <code>_run_scripted.sh</code> · <code>summarize.py</code>) ·
코드 <code>feat/reward-v3</code> 3aa0636 (<code>C:\\work\\DT_Vessel_v3</code>) · 평가기 <code>eval/eval_ckpt.py</code> 그대로(수정 없음) ·
Windows 실행 2026-10-01 17:33–18:54, 12칸 동시(GPU 4장) · 원자료 <code>runs/2026-10-01_scripted/out/sc_*.txt</code>, 표 <code>_summary.md</code> ·
이 문서 = <code>runs/2026-10-01_scripted/report/make_report.py</code> 가 로그에서 직접 읽어 만듦</div>

<h2>0. 요약</h2>
<div class="key"><ul>
<li><b>학습 없는 규칙 배에서도 300 m 참값 정보 이득 있음.</b> vo300 vs vo56h150(H·dt 같음, 보는 거리만 다름):
안전 지표 {len(w_safe)}/{len(SAFE)}개가 3/3 짝 우세 — 충돌 {f(v['vo56h150']['vColl'])} → {f(v['vo300']['vColl'])} %, 도착 {f(v['vo56h150']['goal'])} → {f(v['vo300']['goal'])} %, 통과거리 {f(v['vo56h150']['dcpa'])} → {f(v['vo300']['dcpa'])} m</li>
<li><b>규칙 지표 2개도 3/3 상승, 다만 내용이 다름.</b> COLREGs 준수 {f(v['vo56h150']['C'])} → {f(v['vo300']['C'])} % 는 전부 StandOn(유지선 침로 유지 {f(v['vo56h150']['C_standon'])} → {f(v['vo300']['C_standon'])} %, {n_sto}/3)에서 옴.
GiveWay·HeadOn 준수는 vo300 이 낮음({n_gw}/3·{n_ho}/3). 역할 판정기 {f(v['vo56h150']['rks_v2'])} → {f(v['vo300']['rks_v2'])} % 도 대부분 '해소'(항상 성공 처리) 몫 — 해소 제외 {f(v['vo56h150']['rks_v2x'])} → {f(v['vo300']['rks_v2x'])} %</li>
<li><b>연료는 기준에 따라 방향이 갈림.</b> 도착 ep 당 연료 {f(v['vo56h150']['fuel'], 0)} → {f(v['vo300']['fuel'], 0)} ({f(fuel_rel, 1, True)} %, 0/3)·방향 바꾼 총량 {f(v['vo56h150']['head'], 0)} → {f(v['vo300']['head'], 0)}° ({f(head_rel, 1, True)} %, 0/3)·도착 ep 길이({n_len}/3 김)는 늘어남.
반대로 도착 1회당 함대 연료 {f(v['vo56h150']['fleet_fuel'], 0)} → {f(v['vo300']['fleet_fuel'], 0)}({n_ff}/3 낮음)·진행 1 m 당 연료({n_fm}/3 낮음)는 vo300 이 나음 — 충돌로 끝나는 ep 가 줄어든 몫</li>
<li><b>56 m 에서 H 60 → 150 s(dt 2 → 4 s 같이 바뀜) 효과는 작음</b>: vo56 → vo56h150 충돌 {f(v['vo56']['vColl'])} → {f(v['vo56h150']['vColl'])} %, 도착 {f(v['vo56']['goal'])} → {f(v['vo56h150']['goal'])} %.
거리 단독 효과는 H=60 인 vo300 칸이 없어 분리 못 함 → '이득 대부분이 거리'는 이 측정 순서 기준임</li>
<li><b>56 m 참값 규칙 배는 학습 OFF 보다 도착·충돌이 크게 나음</b>: vo56 {f(v['vo56']['goal'])} % · {f(v['vo56']['vColl'])} % vs t_off {f(lo['t_off']['goal'])} % · {f(lo['t_off']['vColl'])} %.
통과거리·COLREGs·역할 판정기는 t_off 가 높음. 입력 정보·시드·창 절단 비율이 달라 참고 비교(§6)</li>
</ul></div>

<h2>1. 무엇을 했나</h2>
<ul>
<li>질문(저자, 10-01): 통신 범위 정보가 <b>강화학습 없이</b> 규칙 배에서도 도움이 되나. 보고서 지표 전부를 학습 배 평가 코드(<code>eval_ckpt.py</code>) 그대로 기록</li>
<li>방법: <code>eval_ckpt.main()</code> 을 그대로 부르고 <code>ckpt_io</code> 함수 2개만 감쌈 — 정책만 규칙 배로 바꾸고 env 는 t_ 스냅샷 그대로. 체크포인트 가중치는 안 씀(<code>--ckpt t_off_s43.pt</code> 는 env 설정 운반용)</li>
<li>VO = <code>verify/check_reward_rank.py</code> 의 vo_action 을 글자 그대로 복사(R·H·dt 만 다름). 후보 침로 = 목표 방위 기준 −120…+120° 15° 간격.
비용 = 위험(보이는 배 전부 합산) + 목표 방위 이탈각 비용(90° 당 30) + 후보 바꾸기 벌점 3. 추력 항상 1. COLREGs 는 안 봄</li>
</ul>
<table><tr><th>정책</th><th>행동</th><th>보는 배</th><th>내다보는 시간 H</th><th>dt</th><th>비고</th></tr>
{desc_rows}
</table>
<ul>
<li>평가 조건 = t_ 주 평가와 같음: <code>--arm OFF --envs 256 --eval_decisions 10000 --burnin 2400</code>, drain 0, 배 16척, 평가 시드 999·1000·1001 (12칸 로그 [run] 줄로 확인)</li>
<li>env = t_ 배치: dyn imo · 장애물 없음 · crossing 0 · 레이더 56 m · 통신 반경 300 m · ROLE_PROMISE_PEN 20 · 판정기 v2 · 전진 보너스 0 · 시간 벌점 0.035 · DCPA 게이트 48 m</li>
<li>실행 전 8 env·40결정 스모크 통과(rc=0, 출력 <code>report/_smoke/</code>). 결과 전 코드 읽기 전용 검토: VO 복사·호출 인자·히스테리시스 리셋·행동 전달(tanh 이중 적용 없음)·지표 파싱 이상 없음(<code>report/_review_2026-10-01.md</code>)</li>
</ul>

<h2>2. 결과 — 보고서 지표 10개</h2>
<div class="cap">평균, 아래 작은 글씨 = 시드별 값(규칙 배 = 평가 시드 999·1000·1001 순, 학습 배 = 학습 시드 43·46·47 순). 회색 줄 = 학습 배 t_off · t_a6 기존 주 평가(평가 시드 999) — 참고용.
주의: <code>_summary.md</code> 의 규칙 배 시드별 값은 파일 이름 순(1000·1001·999)임.</div>
{tbl_main()}

<div class="pb"></div>
<h2>3. 그림</h2>
{fig_main()}
<div class="cap">그림 1. 규칙 배 4개(막대 = 평균, 점 = 평가 시드 3개)와 학습 배 참고 2개(빗금, 점 = 학습 시드 3개). 점선 오른쪽은 입력 정보·시드 구성이 다른 참고값(§6). 연료·방향 바꾼 총량은 도착 에피소드 평균.</div>
{fig_pairs()}
<div class="cap">그림 2. vo300 − vo56h150 (보는 거리만 다름). 막대 = 상대 차 %(지표 방향으로 부호 맞춤, + = vo300 이 나음), 괄호 = 절대 차, 점 = 평가 시드별.
상대 %는 기저가 작은 지표에서 커 보임(충돌 +{f(-(M('vo300', 'vColl') - M('vo56h150', 'vColl')) / M('vo56h150', 'vColl') * 100, 0)} % = 절대 {f(M('vo300', 'vColl') - M('vo56h150', 'vColl'), 1)} pp). 시간초과는 차가 로그 출력 해상도(0.1 %) 수준이라 뺌.</div>

<h2>4. 짝 비교 (같은 평가 시드 3짝)</h2>
<h3>4-1. vo300 vs vo56h150 — 보는 거리 56 → 300 m 효과 (H 150 s · dt 4 s 같음)</h3>
{tbl_pair('vo300', 'vo56h150')}
<h3>4-2. vo300 vs vo56 — 거리 + H + dt 합산 효과 (순위 검사 정책 그대로)</h3>
{tbl_pair('vo300', 'vo56')}
<h3>4-3. vo56h150 vs vo56 — H 60 → 150 s · dt 2 → 4 s 효과 (보는 거리 56 m 같음)</h3>
{tbl_pair('vo56h150', 'vo56')}
<div class="cap">승/짝 = 같은 평가 시드끼리 앞 정책이 지표 방향으로 나은 짝 수. 같으면 승 아님(벽 충돌 0.0 = 0.0 → 0/3). 초록 = 3/3, 빨강 = 0/3 이면서 차가 있음.
시간초과 차(0.1 pp)는 로그 출력 해상도(0.1 %) 수준이라 승/짝 의미가 약함.</div>

<div class="pb"></div>
<h2>5. 보조 지표 (평균)</h2>
{tbl_sub()}
<ul>
<li>VO 는 COLREGs 를 모름 → 좌·우 아무 쪽이든 비용이 낮은 쪽으로 돎. 전체 COLREGs 준수(vo56 {f(v['vo56']['C'])} %)가 안 피하는 goal({f(v['goal']['C'])} %, StandOn {f(v['goal']['C_standon'])} %)보다도 낮음</li>
<li>vo300 은 무변침 조우가 더 많음({f(v['vo300']['noturn'])} vs vo56h150 {f(v['vo56h150']['noturn'])} %, {n_nt}/3) — 안 틀어도 되는 조우를 더 많이 가려냄. StandOn 준수 상승과 같은 방향</li>
<li>진행 1 m 당 연료: vo300 이 vo56h150 대비 {n_fm}/3 낮음, vo56 대비 {n_fm56}/3 낮음(차 0.01 단위). 도착 ep 길이: vo300 이 vo56h150 대비 {n_len}/3 김</li>
<li>창 절단 미완 ep 비율(집계 제외): VO 3종 {"–".join(sorted({f(v[p]['cens_pct'], 1) for p in POLS[1:]}))} % (goal {f(v['goal']['cens_pct'])} %) vs 학습 배 t_off {f(lo['t_off']['cens_pct'])} % · t_a6 {f(lo['t_a6']['cens_pct'])} % — 학습 배 쪽에서 긴 ep 가 더 많이 빠짐</li>
</ul>

<h2>6. 읽을 때 주의</h2>
<div class="note"><ul>
<li><b>VO 는 보이는 배의 위치·속도 참값을 씀</b>(레이더 ray 아님) → 정확한 이름은 'ARPA 참값 규칙 배'. vo300 은 300 m 안 <b>모든 배</b>(척수 제한 없음)를 봄. 학습 배 통신은 가까운 4척(maxp 4)·latent 경로이고, 경기장이 600 × 600 m 라 300 m 면 함대 대부분이 보임 → vo300 은 통신 채널이 줄 수 있는 정보의 <b>상한</b>에 가까운 모사이지 통신 자체가 아님</li>
<li><b>학습 배 줄은 직접 짝 비교 아님</b>: 학습 배 = 레이더 ray 입력(+ t_a6 는 통신 필드), 평가 시드 999 × 학습 시드 43·46·47. 규칙 배 = 참값, 평가 시드 999·1000·1001. 창 절단 미완 ep 비율도 다름(§5)</li>
<li><b>학습 배 t_ 와의 관계</b>: t_ 에서도 통신 배(t_a6)의 도착 ep 연료·방향 바꾼 총량이 t_off 보다 큼(표). 다만 t_a6 는 함대 연료({f(lo['t_a6']['fleet_fuel'], 0)} vs {f(lo['t_off']['fleet_fuel'], 0)})도 나빴고 vo300 은 좋아짐 → 숫자 일부의 방향만 같고, 원인이 같다는 근거는 없음</li>
<li><b>시드 간 산포가 아주 작음</b>(예: vo300 충돌 {"·".join(f(runs['vo300'][s]['vColl']) for s in SEEDS)}). 정책이 결정론이라 env 배치만 다름 → 3/3 은 'env 배치 3개에서 일관'의 뜻이고 학습 변동은 없음</li>
<li><b>epReward 는 끝난 ep 만 셈</b>(drain 0) → ep 길이·창 절단 영향. vo300 {f(v['vo300']['epReward'], 0)} vs vo56 {f(v['vo56']['epReward'], 0)} 은 차가 커도 보조로만 볼 것</li>
<li><b>연료·방향 결과는 이 VO 비용 설계 기준</b>(이탈각 비용 90°당 30, 바꾸기 벌점 3). 다른 규칙이면 달라질 수 있음 → 일반화하지 않음</li>
<li>예전 순위 게이트 창(E=32 · burn 600 · T=1500, <code>runs/2026-09-30_t/out/_rank.log</code>) 값(시드 3개 평균) vo56 83.6/8.0 · vo300 87.1/4.5 · goal 48.4/48.3 (도착/충돌 %)과 숫자는 다름(규약 다름). 순서·방향은 같음</li>
<li>참고: 계획서 G1(trunk 9M 마지막 20 update 기준 도착 ≥ 90 % · 충돌 ≤ 5 % · 장애물 ≤ 2 %)에 숫자만 대 보면 vo300 충돌 {f(v['vo300']['vColl'])} % 근소 미달, vo56 {f(v['vo56']['vColl'])} % 미달. 측정 창·대상(학습 OFF)이 달라 판정 아님</li>
</ul></div>

<h2>7. 실행 메모</h2>
<ul>
<li>처음엔 이 서버의 OSH Dropbox 클라이언트가 07:13 쯤부터 꺼져 있어 스크립트 폴더가 없었음 → 17:25 재시작 뒤 동기화돼 실행</li>
<li>스크립트 기본 동시 실행 3 → <code>VESSEL_SC_JOBS=12</code> 로 12칸 한 번에(작업 스케줄러 <code>vessel_scripted</code>, 세션과 분리). 칸당 VRAM ≈ 0.8 GB, GPU 4장 각 7–8 GB / 16 GB → WDDM 오버서브 위험 없었음</li>
<li>CPU 100 %(다른 사용자 job 포함) → {DUR}, 전체 17:33–18:54. 실패·재시작 칸 없음(12/12 끝 줄 확인)</li>
<li>검토에서 나온 잠재 문제 2개(이번 판 영향 없음): ① 드라이버가 도는 중 다시 띄우면 진행 중 칸을 중복 실행해 같은 파일을 덮어씀 ② 실패 칸은 경고 없이 평균에서 빠지는데 꼬리말은 '시드 3개'로 남음</li>
<li>이 문서는 숫자·표현 2관점 읽기 전용 감사 후 개정함(문구·제목·주의만 바뀜, 숫자 출처 동일)</li>
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
