# 4차(2026-10-02 낮): 규칙 배 Fig1 상한 — 의도 공유(…i)·규칙 v2(…h)·Woerner 8조 작은 변침(R8) 결과 정리 PDF.
# 숫자는 전부 로그에서 읽음: runs/2026-10-01_scripted/out/sc_*.txt·lr_t_*.txt (summarize.parse_all = report_parse + Woerner·타·행동 줄),
# R8 추가 전 값 = out_v3_0420/ (3차 로그 백업, 같은 정책·같은 시드). 손으로 옮겨 적은 숫자 없음.
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report4.py [출력 폴더]   (base anaconda = matplotlib 있음)
import io, os, re, subprocess, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True   # Dropbox 폴더에 __pycache__ 안 만듦
HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.normpath(os.path.join(HERE, '..'))                     # runs/2026-10-01_scripted
RUNS = os.path.normpath(os.path.join(SC, '..'))
T = os.path.join(RUNS, '2026-09-30_t')
PPP = os.path.normpath(os.path.join(RUNS, '..', '..'))               # Dropbox/Private_Paper_Project
OLD = os.path.join(SC, 'out_v3_0420')                                # 3차(R8 없음) 로그 백업
sys.path.insert(0, SC)
sys.path.insert(0, T)
import summarize as S  # noqa: E402

NAME = '2026-10-02_규칙배4차_Fig1상한_결과'
POLS = ['goal', 'vo56', 'vo56h150', 'vo300', 'vo300i', 'col56', 'col300', 'col300i', 'col56h', 'col300h', 'col300ih']
LRN = ['t_off', 't_a6']
OLDPOLS = ['goal', 'vo56', 'vo300', 'col56', 'col300']               # 3차에도 있던 정책(행동 같음, 채점만 R8 추가)
LSEEDS0 = [43, 46, 47]
PARTIAL = '--partial' in sys.argv                                     # 배치 도중 중간 보고: R8 줄이 시드 3개 다 있는 정책만
ARGS = [a for a in sys.argv[1:] if a != '--partial']


def done(fp):
    return os.path.exists(fp) and 'R8=' in open(fp, encoding='utf-8', errors='replace').read()


if PARTIAL:
    POLS = [p for p in POLS if all(done(os.path.join(SC, 'out', f'sc_{p}_e{s}.txt')) for s in (999, 1000, 1001))]
    LRN = [a for a in LRN if all(done(os.path.join(SC, 'out', f'lr_{a}_s{sd}.txt')) for sd in LSEEDS0)]
    OLDPOLS = [p for p in OLDPOLS if p in POLS]
    NAME += '_중간'
SEEDS = [999, 1000, 1001]
LSEEDS = [43, 46, 47]
COL = {'goal': '#8a8985', 'vo56': '#eb6834', 'vo300': '#2a78d6', 'vo300i': '#7fb2ea', 'col56': '#b5651d', 'col300': '#1b4f8c',
       'col300i': '#2a9d8f', 'col56h': '#d9822b', 'col300h': '#3d5a99', 'col300ih': '#0b6e4f', 'vo56h150': '#f2a07b',
       't_off': '#9c9a94', 't_a6': '#6f9fd8'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'svg.fonttype': 'none', 'font.size': 8.5,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white'})
MET = S.METRICS + [('w_r8', 'W 8조 작은 변침 R8 %', +1)]
SGN = {k: s for k, _, s in MET}
LAB = {k: l for k, l, _ in MET}
ND = lambda k: S.NDEC.get(k, 1)  # noqa: E731
W_K = ['w_score', 'w_r8', 'w_giveway', 'w_headon', 'w_overtaking', 'w_standon']
SAFE_K = ['vColl', 'goal', 'dcpa', 'minPass', 'TO', 'oColl']
OLD_K = ['C', 'C_headon', 'C_standon', 'C_giveway', 'C_overtake', 'rks_v2', 'rks_v2x']
COST_K = ['fuel', 'fleet_fuel', 'fuel_per_m', 'head', 'rud_trav', 'rud_abs', 'epReward']
assert all(k in LAB for k in W_K + SAFE_K + OLD_K + COST_K)
WN = re.compile(r"\[woerner-COLREGs\] score=\s*([\d.]+)% \(n=(\d+)")
WROLE = re.compile(r"(giveway|headon|overtaking|standon)=\s*([\d.]+)%\(n=(\d+) ([^)]*)\)")
RUN_LINE = 'burnin=2400 dec=10000 drain=0 ring=1.0 crossing=0 seed={} envs=256 vessels=16 maxp=4 dev=cuda'
ENDL = re.compile(r'goal=.*?eps')
MODES = ('VO', 'give-way', 'stand-on keep', 'stand-on act', 'give geometry')
# 짝: (a, b, 제목) — a 가 '통신/새 규칙' 쪽
PAIRS = [('col300ih', 'col56h', 'Fig1 상한: 300 m+의도 vs 56 m (규칙 v2)'),
         ('vo300i', 'vo56h150', 'VO 300 m+의도 vs 56 m (H·dt 같음)'),
         ('vo300i', 'vo56', 'VO 300 m+의도 vs VO 56 m (H·dt 도 다름)'),
         ('vo300', 'vo56h150', 'VO 보는 거리만 (H·dt 같음)'),
         ('vo56h150', 'vo56', 'VO 56 m: H 150 vs 60 s'),
         ('col300h', 'col56h', '보는 거리만 (규칙 v2)'),
         ('col300ih', 'col300h', '의도 공유 (규칙 v2, 300 m)'),
         ('col300i', 'col300', '의도 공유 (규칙 v1, 300 m)'),
         ('vo300i', 'vo300', '의도 공유 (VO, 300 m)'),
         ('col56h', 'col56', '규칙 v2 (56 m)'),
         ('col300h', 'col300', '규칙 v2 (300 m)'),
         ('t_a6', 't_off', '학습 배 a6 vs off (같은 trunk)')]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f(v, nd=1, sign=False):
    if v is None:
        return 'n/a'
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace('-', '−')


def load(fp, seed, label, need_r8=True):
    r = S.parse_all(fp)
    s = open(fp, encoding='utf-8', errors='replace').read()
    assert 'goal' in r and r.get('w_score') is not None, f'미완 또는 Woerner 줄 없음: {fp}'
    assert (not need_r8) or 'R8=' in s, f'R8 판정 없음: {fp}'
    m = re.search(r'\[run\] (.*)', s)
    assert m and m.group(1).strip() == RUN_LINE.format(seed), (label, m and m.group(1))
    r['W_n'] = int(WN.search(s).group(2))
    r['endline'] = ENDL.search(s).group(0)
    num, den = 0.0, 0
    for role, sc, n, body in WROLE.findall(s):
        mm = re.search(r'R8=([\d.]+)', body)
        if mm:
            num += float(mm.group(1)) * int(n)
            den += int(n)
    r['w_r8'] = num / den if den else None                          # 역할 4개 조우 수 가중 평균
    md = r.get('mode')
    if md:
        r['modes'] = {k.strip(): float(v) for k, v in re.findall(r'([A-Za-z][A-Za-z\- ]*)=([\d.]+)%', md)}
    return r


runs = {p: {s: load(os.path.join(SC, 'out', f'sc_{p}_e{s}.txt'), s, p) for s in SEEDS} for p in POLS}
learned = {a: {sd: load(os.path.join(SC, 'out', f'lr_{a}_s{sd}.txt'), 999, f'{a}_s{sd}') for sd in LSEEDS} for a in LRN}
ALL = dict(runs, **learned)
# 3차(R8 없음) — 같은 정책·같은 시드. 행동이 같으면 끝 줄(도착·충돌·벽·시간초과·ep 수)이 글자까지 같아야 함
old = {p: {s: load(os.path.join(OLD, f'sc_{p}_e{s}.txt'), s, p, need_r8=False) for s in SEEDS} for p in OLDPOLS}
old.update({a: {sd: load(os.path.join(OLD, f'lr_{a}_s{sd}.txt'), 999, f'{a}_s{sd}', need_r8=False) for sd in LSEEDS} for a in LRN})
PAIRS = [x for x in PAIRS if x[0] in ALL and x[1] in ALL]
N_CELLS = sum(len(v) for v in ALL.values())
MISSING = [p for p in ['goal', 'vo56', 'vo300', 'vo300i', 'col56', 'col300', 'col300i', 'col56h', 'col300h', 'col300ih', 't_off', 't_a6'] if p not in ALL]
REPRO = {(p, s): old[p][s]['endline'] == ALL[p][s]['endline'] for p in old for s in old[p]}
if 'vo56h150' in ALL:
    for sd in SEEDS:
        o = ENDL.search(open(os.path.join(OLD, f'sc_vo56h150_e{sd}.txt'), encoding='utf-8', errors='replace').read()).group(0)
        REPRO[('vo56h150', sd)] = (o == ALL['vo56h150'][sd]['endline'])
print('repro end lines:', sum(REPRO.values()), '/', len(REPRO))


def M(p, k):
    return mean([ALL[p][s].get(k) for s in ALL[p]])


def pair(a, b, k):
    seeds = sorted(set(ALL[a]) & set(ALL[b]))
    ok = [(ALL[a][s].get(k), ALL[b][s].get(k)) for s in seeds]
    ok = [(x, y) for x, y in ok if x is not None and y is not None]
    win = sum(1 for x, y in ok if (x - y) * SGN[k] > 0)
    lose = sum(1 for x, y in ok if (x - y) * SGN[k] < 0)
    return mean([x for x, _ in ok]), mean([y for _, y in ok]), win, lose, len(ok)


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index('<svg'):]


FIG1_K = ['vColl', 'goal', 'dcpa', 'minPass', 'TO', 'oColl', 'w_score', 'w_r8', 'C', 'rks_v2',
          'fuel', 'fleet_fuel', 'fuel_per_m', 'head', 'rud_trav', 'epReward']


def fig_fig1(a='col300ih', b='col56h'):
    # 지표별 작은 그림: 막대 = 평균, 점 = 평가 시드. 제목 = a 승/짝
    fig, axs = plt.subplots(4, 4, figsize=(7.4, 6.2))
    for ax, k in zip(axs.flat, FIG1_K):
        ma, mb, w, lo, n = pair(a, b, k)
        for i, nm in enumerate((b, a)):
            vals = [ALL[nm][s].get(k) for s in SEEDS]
            ax.bar(i, mean(vals), width=0.62, color=COL[nm], zorder=2)
            ax.scatter([i] * len(vals), vals, s=5, color=INK, zorder=3)
        lo_v = min(min(ALL[nm][s].get(k) for s in SEEDS) for nm in (a, b))
        hi_v = max(max(ALL[nm][s].get(k) for s in SEEDS) for nm in (a, b))
        if lo_v > 0 and (hi_v - lo_v) < 0.25 * hi_v:                 # 차이가 작은 지표는 0 이 아닌 곳부터 보여 줌(축 표시됨)
            pad = max((hi_v - lo_v) * 0.6, 0.02 * hi_v)
            ax.set_ylim(lo_v - pad, hi_v + pad)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([b, a], fontsize=6.5)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(axis='x', visible=False)
        tag = '↑' if SGN[k] > 0 else '↓'
        clr = '#070' if w == n else ('#b00' if lo == n else INK)
        ax.set_title(f'{LAB[k]} {tag}  ·  {a} {w}/{n}', fontsize=6.6, color=clr)
    fig.tight_layout(h_pad=0.9, w_pad=0.6)
    return svg(fig)


def fig_r8():
    # R8 추가 전(3차) → 후(4차) Woerner 전체 점수, 같은 정책·같은 시드
    names = OLDPOLS + LRN
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    for i, nm in enumerate(names):
        b4 = [old[nm][s]['w_score'] for s in old[nm]]
        af = [ALL[nm][s]['w_score'] for s in ALL[nm]]
        ax.scatter([i - 0.12] * len(b4), b4, s=10, color='#bbb', zorder=3)
        ax.scatter([i + 0.12] * len(af), af, s=10, color=COL[nm], zorder=3)
        ax.plot([i - 0.12, i + 0.12], [mean(b4), mean(af)], color=COL[nm], lw=1.6, zorder=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel('Woerner 준수 %', fontsize=7.5)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    return svg(fig)


def tbl(keys, names):
    h = ['<table><tr><th>정책</th>' + ''.join(f'<th>{LAB[k]}</th>' for k in keys) + '</tr>']
    for nm in names:
        cls = ' class="ref"' if nm.startswith('t_') else (' class="hl"' if nm in ('col56h', 'col300ih') else '')
        cells = []
        for k in keys:
            vals = [ALL[nm][s].get(k) for s in ALL[nm]]
            mv = mean(vals)
            cells.append(f'<td>{f(mv, ND(k))}<br><span class="sd">{"·".join(f(x, ND(k)) for x in vals) if mv is not None else ""}</span></td>')
        lab = nm + ('<br><span class="sd">학습 배</span>' if nm.startswith('t_') else '')
        h.append(f'<tr{cls}><td>{lab}</td>' + ''.join(cells) + '</tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_pair(a, b):
    h = [f'<table><tr><th>지표</th><th>{b}</th><th>{a}</th><th>차 ({a} − {b})</th><th>{a} 승/짝</th></tr>']
    for k, _, _ in MET:
        ma, mb, w, lo, n = pair(a, b, k)
        if not n:
            continue
        tag = 'ok' if w == n else ('bad' if lo == n else '')
        h.append(f'<tr><td>{LAB[k]}</td><td>{f(mb, ND(k))}</td><td>{f(ma, ND(k))}</td><td>{f(ma - mb, ND(k), True)}</td><td class="{tag}">{w}/{n}</td></tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_matrix(prs=None):
    prs = PAIRS if prs is None else prs
    h = ['<table class="mx"><tr><th>지표</th>' + ''.join(f'<th>{a}<br>vs {b}<br><span class="sd">{t}</span></th>' for a, b, t in prs) + '</tr>']
    for k, _, _ in MET:
        cells = []
        for a, b, _ in prs:
            ma, mb, w, lo, n = pair(a, b, k)
            if not n:
                cells.append('<td>n/a</td>')
                continue
            tag = 'ok' if w == n else ('bad' if lo == n else '')
            cells.append(f'<td><span class="{tag}">{w}/{n}</span> <span class="sd">{f(ma - mb, ND(k), True)}</span></td>')
        h.append(f'<tr><td>{LAB[k]}</td>' + ''.join(cells) + '</tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_r8():
    h = ['<table><tr><th>정책</th><th>Woerner 전 (3차)</th><th>Woerner 후 (R8 추가)</th><th>차</th><th>R8 항 평균 %</th><th>끝 줄 같음</th></tr>']
    for nm in OLDPOLS + LRN:
        b4 = mean([old[nm][s]['w_score'] for s in old[nm]])
        af = M(nm, 'w_score')
        same = sum(REPRO[(nm, s)] for s in old[nm])
        h.append(f'<tr><td>{nm}</td><td>{f(b4)}</td><td>{f(af)}</td><td>{f(af - b4, 1, True)}</td><td>{f(M(nm, "w_r8"))}</td>'
                 f'<td class="{"ok" if same == len(old[nm]) else "bad"}">{same}/{len(old[nm])}</td></tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_modes():
    names = [p for p in POLS if all(ALL[p][s].get('modes') for s in SEEDS)]
    h = ['<table><tr><th>정책</th>' + ''.join(f'<th>{m}</th>' for m in MODES) + '</tr>']
    for nm in names:
        h.append(f'<tr><td>{nm}</td>' + ''.join(f'<td>{f(mean([ALL[nm][s]["modes"].get(m) for s in SEEDS]))}</td>' for m in MODES) + '</tr>')
    h.append('</table>')
    return '\n'.join(h)


# 요약용 계산(전부 위 표와 같은 함수)
FIG1 = ('col300ih', 'col56h')
assert all(x in ALL for x in FIG1), 'Fig1 짝 미완'
mk = [k for k, _, _ in MET]
f1 = {k: pair(*FIG1, k) for k in mk}
f1_win = [k for k in mk if f1[k][4] and f1[k][2] == f1[k][4]]
f1_lose = [k for k in mk if f1[k][4] and f1[k][3] == f1[k][4]]
f1_mix = [k for k in mk if f1[k][4] and k not in f1_win and k not in f1_lose]
print('Fig1 3/3 win', len(f1_win), 'lose', len(f1_lose), 'mixed', len(f1_mix), 'of', len(mk))
print('  lose:', f1_lose, ' mixed:', f1_mix)
for a, b, t in PAIRS:
    ws = sum(1 for k in mk if pair(a, b, k)[4] and pair(a, b, k)[2] == pair(a, b, k)[4])
    ls = sum(1 for k in mk if pair(a, b, k)[4] and pair(a, b, k)[3] == pair(a, b, k)[4])
    print(f'  {a} vs {b}: 3/3 win {ws} · 0/3 {ls}')


# 최선 OFF 대비(스펙 09-30 §7 '최선 OFF 풀', CLAUDE.md anti-rigging 2원칙): 통신 없는 정책 전부와 col300ih 를 같은 평가 시드로 짝 비교.
# 4차에서 아직 안 끝난 OFF 정책은 3차 로그(out_v3_0420, 행동 같음 — 결정론 정책) 값을 쓰고 표에 출처를 적음. Woerner 는 3차면 R8 전 값이라 비교에서 뺌
OFFPOOL = ['vo56', 'vo56h150', 'col56h', 'col56', 'goal']
BEST_K = ['vColl', 'goal', 'dcpa', 'minPass', 'TO', 'oColl', 'fuel', 'fleet_fuel', 'fuel_per_m', 'head', 'rud_trav', 'epReward']
REFD, REFSRC = {}, {}
for p in OFFPOOL:
    if p in ALL:
        REFD[p], REFSRC[p] = ALL[p], '4차'
    else:
        REFD[p] = {sd: load(os.path.join(OLD, f'sc_{p}_e{sd}.txt'), sd, p, need_r8=False) for sd in SEEDS}
        REFSRC[p] = '3차 로그'


def pair_ref(b, k, a='col300ih'):
    ok = [(ALL[a][sd].get(k), REFD[b][sd].get(k)) for sd in SEEDS]
    ok = [(x, y) for x, y in ok if x is not None and y is not None]
    return (mean([x for x, _ in ok]), mean([y for _, y in ok]), sum(1 for x, y in ok if (x - y) * SGN[k] > 0),
            sum(1 for x, y in ok if (x - y) * SGN[k] < 0), len(ok))


def tbl_best(a='col300ih'):
    keys = BEST_K + (['w_score', 'w_r8'] if all(v == '4차' for v in REFSRC.values()) else [])
    h = [f'<table><tr><th>지표</th><th>{a}</th>' + ''.join(f'<th>{b}<br><span class="sd">{REFSRC[b]}</span></th>' for b in OFFPOOL)
         + f'<th>{a} 가 3/3 이긴 OFF 수</th></tr>']
    for k in keys:
        ma = M(a, k)
        cells, nwin = [], 0
        for b in OFFPOOL:
            _, mb, w, lo, n = pair_ref(b, k, a)
            nwin += (w == n)
            tag = 'ok' if w == n else ('bad' if lo == n else '')
            cells.append(f'<td>{f(mb, ND(k))} <span class="{tag}">{w}/{n}</span></td>')
        h.append(f'<tr><td>{LAB[k]}</td><td>{f(ma, ND(k))}</td>' + ''.join(cells) + f'<td>{nwin}/{len(OFFPOOL)}</td></tr>')
    h.append('</table>')
    return '\n'.join(h)


def best_all(a):
    keys = BEST_K + (['w_score', 'w_r8'] if all(v == '4차' for v in REFSRC.values()) else [])
    allw = [k for k in keys if all(pair_ref(b, k, a)[2] == pair_ref(b, k, a)[4] for b in OFFPOOL)]
    return allw, keys


def RM(p, k):
    return mean([REFD[p][sd].get(k) for sd in SEEDS])


V2 = ('vo300i', 'vo56h150')
v2 = {k: pair(*V2, k) for k in mk}
v2_win = [k for k in mk if v2[k][4] and v2[k][2] == v2[k][4]]
v2_lose = [k for k in mk if v2[k][4] and v2[k][3] == v2[k][4]]
v2_mix = [k for k in mk if v2[k][4] and k not in v2_win and k not in v2_lose]
print('vo300i vs vo56: 3/3 win', len(v2_win), 'lose', v2_lose, 'mixed', v2_mix)


def kl(keys):
    return ', '.join(LAB[k] for k in keys) if keys else '없음'


SUMMARY = os.environ.get('VESSEL_REPORT_SUMMARY_HTML', '')        # 결과 본 뒤 손으로 쓰는 해석 bullet (숫자는 위 계산값만 인용)
if os.path.exists(os.path.join(HERE, 'report4_summary.html')):
    SUMMARY = open(os.path.join(HERE, 'report4_summary.html'), encoding='utf-8').read()


def fill(s):
    # report4_summary.html 안의 {{식}} 을 이 스크립트 값으로 바꿈 — 해석 문장 안의 숫자도 로그에서 옴
    return re.sub(r'\{\{(.+?)\}\}', lambda m: str(eval(m.group(1))), s)


# 실행 메모(시각·사고) — 드라이버 로그 끝 줄과 함께 손으로 적음
RUNNOTE = '드라이버 로그 마지막 줄(이 문서 만든 시점): ' + open(os.path.join('C:/work/DT_Vessel_v3/Python/_repro_out_t', '_scripted_driver.log'), encoding='utf-8').read().strip().splitlines()[-1] \
    if os.path.exists('C:/work/DT_Vessel_v3/Python/_repro_out_t/_scripted_driver.log') else ''
if os.path.exists(os.path.join(HERE, 'report4_runnote.html')):
    RUNNOTE = open(os.path.join(HERE, 'report4_runnote.html'), encoding='utf-8').read().strip()


html = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>규칙 배 4차 결과</title>
<style>
@page {{ size: A4; margin: 12mm 11mm; }}
body {{ font-family: "Malgun Gothic", sans-serif; font-size: 9.5pt; line-height: 1.5; color: #111; background: #fff; }}
h1 {{ font-size: 14pt; margin: 0 0 4px; }}
h2 {{ font-size: 11.8pt; margin: 14px 0 6px; border-bottom: 2px solid #333; padding-bottom: 2px; }}
h3 {{ font-size: 10pt; margin: 9px 0 3px; }}
.sub {{ color: #555; font-size: 8.3pt; margin-bottom: 8px; }}
table {{ border-collapse: collapse; margin: 5px 0 7px; font-size: 7.9pt; }}
table.mx {{ font-size: 7.3pt; }}
th, td {{ border: 1px solid #999; padding: 2px 4px; text-align: right; vertical-align: top; }}
th {{ background: #eee; text-align: center; }} td:first-child, th:first-child {{ text-align: left; }}
tr.ref td {{ background: #f6f6f6; color: #444; }}
tr.hl td {{ background: #f3f8fe; }}
.sd {{ color: #777; font-size: 6.1pt; white-space: nowrap; }}
.key {{ border: 2px solid #1b4f8c; background: #f3f8fe; padding: 6px 12px; margin: 8px 0; }}
.note {{ background: #f4f4f4; border-left: 4px solid #888; padding: 6px 10px; margin: 6px 0; font-size: 8.8pt; }}
.bad {{ color: #b00; font-weight: bold; }} .ok {{ color: #070; font-weight: bold; }}
ul {{ margin: 3px 0 3px 18px; padding: 0; }} li {{ margin: 1px 0; }}
.cap {{ font-size: 8pt; color: #444; margin: 2px 0 8px; }}
.pb {{ page-break-before: always; }}
svg {{ max-width: 100%; height: auto; }}
code {{ font-size: 8pt; }}
</style></head><body>

{('<div class="key" style="border-color:#b00"><b>중간 보고</b> — 36칸 중 ' + str(N_CELLS) + '칸 끝난 시점. 아직 없는 정책: ' + ', '.join(MISSING) + '. 배치 끝나면 최종본으로 바꿈</div>') if PARTIAL else ''}
<h1>2026-10-02 규칙 배 4차 — Fig1 상한(의도 공유 · 규칙 v2) + Woerner 8조 작은 변침(R8)</h1>
<div class="sub">저자 10-02 "ㅇㅋ 진행해줘"(Fig1, 철학 B) · <code>runs/2026-10-01_scripted/</code> · 평가 코드 <code>feat/imitation-init</code> 0a3e70e (feat/reward-v3 와 같음, 학습 없음) ·
Windows 2026-10-02 · 원자료 <code>out/sc_*.txt</code>·<code>out/lr_t_*.txt</code>, R8 전 값 <code>out_v3_0420/</code> · 이 문서 = <code>report/make_report4.py</code> 가 로그에서 직접 읽어 만듦</div>

<h2>0. 요약</h2>
<div class="key"><ul>
<li><b>Fig1 상한 짝 col300ih vs col56h</b>(300 m + 의도 공유 vs 56 m, 규칙 v2 같음): 지표 {len(mk)}개 중 col300ih 3/3 승 <b>{len(f1_win)}</b> · 0/3 <b>{len(f1_lose)}</b> · 갈림 {len(f1_mix)}</li>
<li>col300ih 가 3/3 진 지표: {kl(f1_lose)}</li>
<li>갈린 지표: {kl(f1_mix)}</li>
<li><b>col300ih 가 이긴 쪽</b>: 배끼리 충돌 {f(M('col56h', 'vColl'))} → {f(M('col300ih', 'vColl'))} % · 도착 {f(M('col56h', 'goal'))} → {f(M('col300ih', 'goal'))} % · 통과거리 {f(M('col56h', 'dcpa'))} → {f(M('col300ih', 'dcpa'))} m · Woerner {f(M('col56h', 'w_score'))} → {f(M('col300ih', 'w_score'))} % · 도착 1회당 함대 연료 {f(M('col56h', 'fleet_fuel'), 0)} → {f(M('col300ih', 'fleet_fuel'), 0)}</li>
<li><b>col300ih 가 진 쪽</b>: 도착 ep 연료 {f(M('col56h', 'fuel'), 0)} → {f(M('col300ih', 'fuel'), 0)} · 방향 바꾼 총량 {f(M('col56h', 'head'), 0)} → {f(M('col300ih', 'head'), 0)}° · 타 이동량 {f(M('col56h', 'rud_trav'), 0)} → {f(M('col300ih', 'rud_trav'), 0)}°/ep · 옛 COLREGs {f(M('col56h', 'C'))} → {f(M('col300ih', 'C'))} % · R8 {f(M('col56h', 'w_r8'))} → {f(M('col300ih', 'w_r8'))} % → 저자 목표 '모든 지표 우월' 미달</li>
<li><b>★ OFF 쪽 col56h 가 최선 OFF 아님</b>: vo56({REFSRC['vo56']}) 충돌 {f(RM('vo56', 'vColl'))} % · 도착 {f(RM('vo56', 'goal'))} % vs col56h {f(M('col56h', 'vColl'))} · {f(M('col56h', 'goal'))} %.
col300ih 도 vo56 에 충돌 {pair_ref('vo56', 'vColl')[2]}/3 · 도착 {pair_ref('vo56', 'goal')[2]}/3 승 → anti-rigging 2원칙(최선 OFF 이김)상 이 짝만으로 '통신 > 무통신' 주장 불가 (§2-1)</li>
<li>규칙 v2 는 56 m 에서 거의 그대로: col56h vs col56 충돌 {f(M('col56h', 'vColl'))} vs {f(RM('col56', 'vColl'))} % · 도착 {f(M('col56h', 'goal'))} vs {f(RM('col56', 'goal'))} %</li>
<li><b>VO 쪽 vo300i vs vo56h150</b>(300 m + 의도 공유 vs 56 m, VO 규칙. H 150 s · dt 4 s 같음 = f_ 배치 선생님 짝): 지표 {len(mk)}개 중 vo300i 3/3 승 <b>{len(v2_win)}</b> · 0/3 <b>{len(v2_lose)}</b> · 갈림 {len(v2_mix)}.
충돌 {f(M('vo56h150', 'vColl'))} → {f(M('vo300i', 'vColl'))} % · 도착 {f(M('vo56h150', 'goal'))} → {f(M('vo300i', 'goal'))} % · Woerner {f(M('vo56h150', 'w_score'))} → {f(M('vo300i', 'w_score'))} % · 도착 ep 연료 {f(M('vo56h150', 'fuel'), 0)} → {f(M('vo300i', 'fuel'), 0)} · 방향 바꾼 총량 {f(M('vo56h150', 'head'), 0)} → {f(M('vo300i', 'head'), 0)}°.
진 지표: {kl(v2_lose)} · 갈린 지표: {kl(v2_mix)}. 참고 vo300i vs vo56(H·dt 도 다름): 3/3 승 {sum(1 for k in mk if pair('vo300i', 'vo56', k)[2] == 3)} · 0/3 {sum(1 for k in mk if pair('vo300i', 'vo56', k)[3] == 3)}</li>
<li>최선 OFF 풀(vo56·vo56h150·col56h·col56·goal) 전부를 3/3 이긴 지표 수: vo300i {len(best_all('vo300i')[0])}/{len(best_all('vo300i')[1])} · col300ih {len(best_all('col300ih')[0])}/{len(best_all('col300ih')[1])} (§2-1·2-2)</li>
<li>VO 의도 공유 효과(vo300i vs vo300): 충돌 {f(M('vo300', 'vColl'))} → {f(M('vo300i', 'vColl'))} % · 도착 {f(M('vo300', 'goal'))} → {f(M('vo300i', 'goal'))} % · 도착 ep 연료 {f(M('vo300', 'fuel'), 0)} → {f(M('vo300i', 'fuel'), 0)} · 방향 바꾼 총량 {f(M('vo300', 'head'), 0)} → {f(M('vo300i', 'head'), 0)}° · 타 이동량 {f(M('vo300', 'rud_trav'), 0)} → {f(M('vo300i', 'rud_trav'), 0)}°/ep</li>
<li><b>R8 추가 뒤에도 학습 배 Woerner 가 가장 높음</b>: t_off {f(M('t_off', 'w_score'))} · t_a6 {f(M('t_a6', 'w_score'))} % vs 규칙 배 최고 {max(POLS, key=lambda p: M(p, 'w_score'))} {f(max(M(p, 'w_score') for p in POLS))} %.
R8 로 내려간 폭: t_off {f(M('t_off', 'w_score') - mean([old['t_off'][sd]['w_score'] for sd in LSEEDS]), 1, True)} · t_a6 {f(M('t_a6', 'w_score') - mean([old['t_a6'][sd]['w_score'] for sd in LSEEDS]), 1, True)} · vo300 {f(M('vo300', 'w_score') - mean([old['vo300'][sd]['w_score'] for sd in SEEDS]), 1, True)} · col300 {f(M('col300', 'w_score') - mean([old['col300'][sd]['w_score'] for sd in SEEDS]), 1, True)} pp (§5). 학습 배 충돌 {f(M('t_off', 'vColl'))} · {f(M('t_a6', 'vColl'))} % — 준수 점수와 안전이 따로 놂은 그대로</li>
{('<li>의도 공유 효과(col300ih vs col300h): 충돌 ' + f(M('col300h', 'vColl')) + ' → ' + f(M('col300ih', 'vColl')) + ' % · 도착 ' + f(M('col300h', 'goal')) + ' → ' + f(M('col300ih', 'goal')) + ' % — 작음</li>') if 'col300h' in ALL else ''}
{fill(SUMMARY)}
{('<li>재현: 3차(vo56h150 은 10-01 1차)와 행동이 같은 칸 ' + str(len(REPRO)) + '개의 끝 줄(도착·충돌·벽·시간초과·ep 수) 같음 ' + str(sum(REPRO.values())) + '/' + str(len(REPRO)) + ' — R8 은 채점만 바꿈</li>') if REPRO else ''}
</ul></div>

<h2>1. 무엇을 했나</h2>
<ul>
<li><b>질문</b>: 강화학습 없이 규칙 배만으로, 통신 정보(300 m 보기 + 의도 공유)가 레이더만 쓰는 배(56 m)를 모든 지표에서 이길 수 있나 = Fig1 의 상한 확인</li>
<li><b>의도 공유</b>(…i): 각 배가 직전 결정의 계획 침로를 방송(1결정 늦음) → 받는 배는 상대 경로를 직진이 아니라 그 침로로 도는 것으로 예측. 300 m 안 배만</li>
<li><b>규칙 v2</b>(…h): 충돌까지 tcpa ≤ 150 s(= VO 내다보는 시간 H)인 배에만 COLREGs 동작(양보·유지·좌현 금지). 3차 col300 의 과잉 반응 수정. 56·300 m 같은 값</li>
<li><b>Woerner R8</b>(8조 "작은 변침 반복 피함"): 조우 중 침로 변화 총량 / (2 × 최대 변침) 이 1–2 배면 최대 50 % 감점. 학습 배가 계속 흔들며 Woerner 최고점을 받던 문제 반영(3차 보고서). 모든 역할 점수에 곱함</li>
<li>정책 10종 × 평가 시드 999·1000·1001 + 학습 배 t_off·t_a6 × 학습 시드 43·46·47(체크포인트 그대로, 평가 시드 999) = 36칸 + vo56h150 3칸(R8 채점으로 다시 잼, 17:3x 끝) = {N_CELLS}칸. 조건 = t_ 주 평가와 같음(<code>[run]</code> 줄 확인)</li>
<li>규칙 배는 보이는 배의 위치·속도 참값을 씀(레이더 ray 아님), 300 m 는 척수 제한 없음 → 통신 정보의 상한 모사</li>
</ul>

<div class="pb"></div>
<h2>2. Fig1 상한 짝: col300ih vs col56h</h2>
{fig_fig1()}
<div class="cap">그림 1. 지표별 col56h(주황) vs col300ih(초록). 막대 = 평균, 점 = 평가 시드 3개. 제목 끝 = col300ih 승/짝(초록 = 3/3, 빨강 = 0/3), ↑↓ = 좋은 방향.
차이가 작은 지표는 세로축이 0 에서 시작하지 않음(축 숫자 확인).</div>
{tbl_pair(*FIG1)}

<h3>2-1. 최선 OFF 대비 — col300ih vs 통신 없는 정책 전부 (같은 평가 시드)</h3>
{tbl_best('col300ih')}
<div class="cap">칸 = OFF 정책 평균 + 왼쪽 정책 승/짝(초록 3/3, 빨강 0/3).{" '3차 로그' = 4차 재실행이 아직 안 끝나 3차 값을 씀(같은 정책·같은 시드, 결정론). Woerner 는 3차 값이 R8 전이라 이 표에서 뺌." if any(v != '4차' for v in REFSRC.values()) else " OFF 정책 전부 4차 값."}</div>

{("""<div class="pb"></div>
<h3>2-2. VO 쪽: vo300i vs vo56h150 (f_ 배치 선생님 짝)</h3>""" + fig_fig1('vo300i', 'vo56h150') + """<div class="cap">그림 1b. 지표별 vo56h150 vs vo300i. 읽는 법 그림 1 과 같음. 두 정책은 보는 거리(56 vs 300 m)·의도 공유만 다름(H 150 s · dt 4 s 같음). vo300i vs vo56 짝은 §3 표.</div>""" + tbl_pair('vo300i', 'vo56h150') + "<h3>vo300i vs 통신 없는 정책 전부</h3>" + tbl_best('vo300i')) if 'vo300i' in ALL and 'vo56h150' in ALL else ''}

<div class="pb"></div>
<h2>3. 짝 비교 모음 (같은 시드끼리, 승/짝 · 평균차)</h2>
{tbl_matrix(PAIRS[:5])}
{tbl_matrix(PAIRS[5:])}
<div class="cap">승 = 왼쪽 정책이 좋은 방향으로 앞섬. 초록 = 3/3, 빨강 = 0/3(3짝 모두 짐). 학습 배 짝은 학습 시드 43·46·47(평가 시드 999).</div>

<div class="pb"></div>
<h2>4. 전체 표 (평균, 아래 = 시드별)</h2>
<h3>4-1. Woerner COLREGs (R8 포함)</h3>
{tbl(W_K, POLS + LRN)}
<h3>4-2. 안전</h3>
{tbl(SAFE_K, POLS + LRN)}
<h3>4-3. 옛 COLREGs 지표 · 역할 판정기</h3>
{tbl(OLD_K, POLS + LRN)}
<h3>4-4. 연료 · 조타 · 보상</h3>
{tbl(COST_K, POLS + LRN)}

<div class="pb"></div>
<h2>5. R8 추가 효과 (같은 정책 · 같은 시드, 3차 → 4차)</h2>
{(fig_r8() + '<div class="cap">그림 2. Woerner 전체 점수 R8 추가 전(회색) → 후(색). 점 = 시드, 선 = 평균 이동.</div>' + tbl_r8()) if old else '<p>아직 없음 — 3차에도 있던 정책(goal·vo56·vo300·col56·col300·t_off·t_a6)이 끝나면 채움</p>'}

<h2>6. COLREGs 규칙 배 행동 비율 (burn-in 뒤 배-결정, 시드 평균 %)</h2>
{tbl_modes()}

<h2>7. 읽을 때 주의</h2>
<div class="note"><ul>
<li>규칙 배 시드 간 산포가 아주 작음(결정론 정책) → 3/3 은 'env 배치 3개에서 일관'의 뜻. 효과 크기는 평균차로 같이 볼 것</li>
<li>규칙 배는 참값 위치·속도, 300 m 는 척수 제한 없음 → 학습 배(레이더 ray + 통신 latent)가 받을 정보보다 많음. 상한으로만 읽음</li>
<li>Woerner 조우는 300 m 안 dcpa &lt; 24 m 접근부터 잡음 → 56 m 배는 조우 시작 뒤 한참 상대를 못 봄. 300 m 쪽이 유리한 것은 정보 효과이자 채점 창 설계의 결과이기도 함</li>
<li>R8 띠(1–2 배)·감점 50 % 는 논문이 비워 둔 값 — 결과 전에 고정(<code>colregs_woerner.py</code> 헤더 *)</li>
<li>원인 설명(왜 어떤 지표가 지는지)은 확인 안 함 — 요약의 해석은 표의 숫자만 근거</li>
</ul></div>

<h2>8. 실행 메모</h2>
<ul>
<li>Mac 세션에서 넘겨받은 한 줄 <code>VESSEL_SC_JOBS=12 bash …/_run_scripted.sh</code> 를 Task Scheduler <code>vessel_scripted</code> 로 실행(10:5x 시작, GPU 4장 12칸 동시)</li>
<li>시작 전 3차 로그 백업: <code>out_v3_0420/</code>(Dropbox), <code>_repro_out_t/scripted_v3_0420/</code>(Windows) — 3차 칸은 R8 줄이 없어 전부 다시 돌았음(덮어씀)</li>
<li>{RUNNOTE}</li>
</ul>
</body></html>
'''

OUT = ARGS[0] if ARGS else PPP
out_html = os.path.join(OUT, NAME + '.html')
out_pdf = os.path.join(OUT, NAME + '.pdf')
open(out_html, 'w', encoding='utf-8').write(html)
edge = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
subprocess.run([edge, '--headless', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={out_pdf}',
                'file:///' + out_html.replace(chr(92), '/')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(out_html)
print(out_pdf)
