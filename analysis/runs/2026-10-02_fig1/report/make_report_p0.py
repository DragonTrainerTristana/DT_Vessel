# 2026-10-03: Fig1 배치 f2_ 흉내 0단계(P0) 결과 + 10-02 f_ 정지 사고 원인 측정 정리 PDF.
# 숫자는 전부 로그에서 읽음:
#   runs/2026-10-02_fig1/out/eval_f2_dagger_d6_s*.txt · f2_dagger_d6_s*.imitation.json · _p0.md   (P0, report_parse.parse_eval)
#   runs/2026-10-02_imitation/out/eval_i_dagger_d6_s*.txt                                        (i_ 학생, 10-02 새벽)
#   runs/2026-10-01_scripted/out/sc_*_e*.txt · lr_t_off_s*.txt                                   (규칙 배·학습 t_off, 4차)
#   runs/2026-10-02_fig1/out/diag_p0/*.log  ([diag] JSON 줄, report/diag/label_stats.py)        (라벨 분포·잡음 진단)
#   runs/2026-10-02_fig1/out/diag_vram/*.log ([vram] 줄, report/diag/measure2.py·measure_trunk.py) (VRAM 측정)
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_p0.py [출력 폴더]
import io, json, os, re, subprocess, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.normpath(os.path.join(HERE, '..'))                     # runs/2026-10-02_fig1
RUNS = os.path.normpath(os.path.join(F1, '..'))
OUT_F = os.path.join(F1, 'out')
T = os.path.join(RUNS, '2026-09-30_t')
SC = os.path.join(RUNS, '2026-10-01_scripted')
IM = os.path.join(RUNS, '2026-10-02_imitation', 'out')
PPP = os.path.normpath(os.path.join(RUNS, '..', '..'))
sys.path.insert(0, T)
sys.path.insert(0, SC)
from report_parse import parse_eval  # noqa: E402
import summarize as S  # noqa: E402

NAME = '2026-10-03_Fig1_f2_흉내P0_결과'
SEEDS = [43, 44, 45, 46, 47]
ESEEDS = [999, 1000, 1001]
P0_GOAL, P0_VCOLL = 80.0, 15.0
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'svg.fonttype': 'none', 'font.size': 8.5,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white'})
RUN_LINE = 'burnin=2400 dec=10000 drain=0 ring=1.0 crossing=0 seed=999 envs=256 vessels=16'


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


# ── P0 (f2_) ──
stud = {}
for sd in SEEDS:
    fp = os.path.join(OUT_F, f'eval_f2_dagger_d6_s{sd}.txt')
    r = parse_eval(fp)
    txt = open(fp, encoding='utf-8', errors='replace').read()
    m = re.search(r'\[run\] (.*)', txt)
    assert m and m.group(1).strip().startswith(RUN_LINE), (sd, m and m.group(1))
    j = json.load(open(os.path.join(OUT_F, f'f2_dagger_d6_s{sd}.imitation.json'), encoding='utf-8'))
    assert j['teacher']['name'] == 'vo56h150' and j['teacher']['action_mode'] == 'course' and j['seed'] == sd, j['teacher']
    r['dl'] = j['dagger_last']
    r['code'] = j['code']
    r['p0'] = r['goal'] >= P0_GOAL and r['vColl'] <= P0_VCOLL
    stud[sd] = r
p0_md = open(os.path.join(OUT_F, '_p0.md'), encoding='utf-8').read()
P0_LINE = re.search(r'P0 판정: .*', p0_md).group(0)
n_pass = sum(1 for r in stud.values() if r['p0'])
assert f'{n_pass}/5' in P0_LINE and ('불통과' in P0_LINE) == (n_pass < 3), P0_LINE
CODE = sorted({r['code'] for r in stud.values()})
# ── 참고: i_ 학생(타각 행동·레이더만·선생님 vo56), 규칙 배, 학습 t_off ──
istud = {sd: parse_eval(os.path.join(IM, f'eval_i_dagger_d6_s{sd}.txt')) for sd in SEEDS}
ref = {p: [S.parse_all(os.path.join(SC, 'out', f'sc_{p}_e{s}.txt')) for s in ESEEDS] for p in ('vo56h150', 'vo56', 'goal')}
ref['t_off'] = [S.parse_all(os.path.join(SC, 'out', f'lr_t_off_s{s}.txt')) for s in (43, 46, 47)]
KEYS = [('goal', '도착 %', 1), ('vColl', '배끼리 충돌 %', 1), ('oColl', '벽 %', 1), ('TO', '시간초과 %', 1), ('dcpa', '통과거리 m', 1),
        ('C', 'COLREGs(옛) %', 1), ('fuel', '연료(도착 ep)', 0), ('head', '방향 바꾼 총량 °', 0)]


def M(rs, k):
    return mean([r.get(k) for r in rs])


sm = {k: M(list(stud.values()), k) for k, _, _ in KEYS}
im = {k: M(list(istud.values()), k) for k, _, _ in KEYS}


# ── 진단: 라벨 분포·잡음 ──
def diag(name):
    s = open(os.path.join(OUT_F, 'diag_p0', name + '.log'), encoding='utf-8', errors='replace').read()
    return json.loads(re.search(r'^\[diag\] (\{.*\})$', s, re.M).group(1))


DG = {k: diag(k) for k in ('s43_teacher', 's43_student', 's44_student', 'long_s43_sampled', 'long_s43_mean', 'long_s44_mean', 'long_s43_teacher')}
for k, d in DG.items():
    assert d['samples'] == (2500 if k.startswith('long') else 300) * 128 * 16, (k, d['samples'])


# ── VRAM 측정 ──
def vram(name):
    s = open(os.path.join(OUT_F, 'diag_vram', name + '.log'), encoding='utf-8', errors='replace').read()
    tr = [(float(a), float(b), float(c)) for a, b, c in re.findall(r'\[vram\] t=\s*(\d+)s alloc=([\d.]+) reserved=([\d.]+)', s)]
    end = re.search(r'\[vram\] END peak_alloc=([\d.]+)GB peak_reserved=([\d.]+)GB wall=(\d+)s', s)
    ab = re.search(r'\[vram\] ABORT reserved=([\d.]+) GB > 11 GB at t=(\d+)s', s)
    decs = [(int(a), float(b)) for a, b in re.findall(r'^\[dagger\] dec=\s*(\d+) .*?\|\s*([\d.]+)min\s*$', s, re.M)]
    dps = [int(x) for x in re.findall(r'\| (\d+) dec/s', s)]
    return dict(trace=tr, peak_alloc=float(end.group(1)) if end else None, peak_res=float(end.group(2)) if end else max([t[2] for t in tr] + ([float(ab.group(1))] if ab else [])),
                abort=(float(ab.group(1)), int(ab.group(2))) if ab else None, decs=decs, dps=dps)


VR = {k: vram(k) for k in ('m2', 'v_gc30', 'v_split_gc30', 'v_nosplit', 't_gc30', 't_nosplit', 't_default')}
print('P0:', P0_LINE, '| student mean goal', f(sm['goal']), 'vColl', f(sm['vColl']), '| code', CODE)
print('diag long: sampled', DG['long_s43_sampled']['goal_pct'], DG['long_s43_sampled']['vColl_pct'], '| mean s43', DG['long_s43_mean']['goal_pct'],
      DG['long_s43_mean']['vColl_pct'], '| mean s44', DG['long_s44_mean']['goal_pct'], DG['long_s44_mean']['vColl_pct'], '| teacher', DG['long_s43_teacher']['goal_pct'], DG['long_s43_teacher']['vColl_pct'])
print('vram:', {k: (v['peak_alloc'], v['peak_res'], v['abort']) for k, v in VR.items()})


def fig_p0():
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.axhspan(0, P0_VCOLL, xmin=0, xmax=1, color='#e8f5e9', zorder=0)
    ax.axvline(P0_GOAL, color='#2e7d32', lw=0.8, ls='--', zorder=1)
    ax.axhline(P0_VCOLL, color='#2e7d32', lw=0.8, ls='--', zorder=1)
    ax.text(P0_GOAL + 0.6, 1.0, 'P0 통과 영역 (도착 ≥ 80 · 충돌 ≤ 15)', fontsize=6.8, color='#2e7d32')
    ax.scatter([r['goal'] for r in stud.values()], [r['vColl'] for r in stud.values()], s=26, color='#c0392b', zorder=3, label='f2_ 학생 (목표 침로·ARPA@56·선생님 vo56h150), 시드 5개')
    ax.scatter([r['goal'] for r in istud.values()], [r['vColl'] for r in istud.values()], s=26, color='#7a5ea8', zorder=3, label='i_ 학생 (타각·레이더만·선생님 vo56), 시드 5개')
    for nm, c, mk, off in (('vo56h150', '#eb6834', 's', (4, 4)), ('vo56', '#f2a07b', 's', (4, -10)), ('goal', '#8a8985', 's', (5, -3)), ('t_off', '#52514e', '^', (5, 3))):
        x, y = M(ref[nm], 'goal'), M(ref[nm], 'vColl')
        ax.scatter([x], [y], s=34, color=c, marker=mk, edgecolor=INK, linewidth=0.5, zorder=4)
        ax.annotate({'vo56h150': '선생님 vo56h150', 'vo56': 'vo56', 'goal': 'goal 배(안 피함)', 't_off': '학습 t_off'}[nm], (x, y), textcoords='offset points', xytext=off, fontsize=6.8, color=INK)
    ax.set_xlabel('도착 % (주 평가)', fontsize=7.5)
    ax.set_ylabel('배끼리 충돌 %', fontsize=7.5)
    ax.set_xlim(35, 100)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=6.6, frameon=False, loc='upper right')
    fig.tight_layout()
    return svg(fig)


def fig_diag():
    names = [('long_s43_sampled', '학생 s43\n샘플 행동\n(평가와 같은 방식)', '#c0392b'), ('long_s43_mean', '학생 s43\n평균 행동\n(잡음 없음)', '#e59866'),
             ('long_s44_mean', '학생 s44\n평균 행동\n(잡음 없음)', '#e59866'), ('long_s43_teacher', '선생님\nvo56h150\n(같은 창)', '#eb6834')]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.5))
    for ax, (k, yl, thr) in zip(axs, (('goal_pct', '도착 %', P0_GOAL), ('vColl_pct', '배끼리 충돌 %', P0_VCOLL))):
        for i, (nm, lab, c) in enumerate(names):
            ax.bar(i, DG[nm][k], width=0.62, color=c, zorder=2)
            ax.text(i, DG[nm][k] + 1.2, f(DG[nm][k]), ha='center', fontsize=6.8, color=INK)
        ax.axhline(thr, color='#2e7d32', lw=0.8, ls='--', zorder=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n[1] for n in names], fontsize=6.3)
        ax.set_ylabel(yl, fontsize=7.5)
        ax.set_ylim(0, 100 if k == 'goal_pct' else 60)
        ax.grid(axis='x', visible=False)
    fig.tight_layout()
    return svg(fig)


def fig_vram():
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.5), gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axs[0]
    for nm, lab, c, ls in (('m2', '기본값 max_split_size_mb:128 (config.py)', '#c0392b', '-'), ('v_split_gc30', 'max_split 128 + gc 0.3', '#e59866', '--'),
                           ('v_gc30', 'gc 0.3 만 (f2_ 에 씀)', '#1b6fd6', '-'), ('v_nosplit', 'max_split 16384', '#2a9d8f', ':')):
        tr = VR[nm]['trace']
        ax.plot([t[0] / 60 for t in tr], [t[2] for t in tr], color=c, lw=1.4, ls=ls, label=lab)
    ax.axhline(16.3, color=INK2, lw=0.7, ls=':')
    ax.text(0.2, 15.2, '카드 16 GB', fontsize=6.5, color=INK2)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 17)
    ax.set_xlabel('분', fontsize=7.5)
    ax.set_ylabel('PyTorch reserved GB (DAgger 1개)', fontsize=7.5)
    ax.legend(fontsize=6.2, frameon=False, loc='center right')
    ax = axs[1]
    items = [('DAgger\n실사용 최대', VR['v_gc30']['peak_alloc'], '#9c9a94'), ('DAgger\nreserved\n(gc 0.3)', VR['v_gc30']['peak_res'], '#1b6fd6'),
             ('trunk\n실사용 최대', VR['t_gc30']['peak_alloc'], '#9c9a94'), ('trunk\nreserved', VR['t_gc30']['peak_res'], '#1b6fd6')]
    for i, (lab, v, c) in enumerate(items):
        ax.bar(i, v, width=0.62, color=c, zorder=2)
        ax.text(i, v + 0.25, f(v, 2), ha='center', fontsize=6.8, color=INK)
    ax.axhline(8.15, color='#c0392b', lw=0.8, ls='--')
    ax.text(-0.4, 8.45, '카드 절반 8.15 GB (GPU당 2개 한계)', fontsize=6.2, color='#c0392b')
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([i[0] for i in items], fontsize=6.3)
    ax.set_ylabel('GB (프로세스 1개)', fontsize=7.5)
    ax.set_ylim(0, 11)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    return svg(fig)


def tbl_p0():
    h = ['<table><tr><th>시드</th>' + ''.join(f'<th>{l}</th>' for _, l, _ in KEYS) + '<th>DAgger 마지막 창<br>학생 도착 · 충돌 · 일치 %</th><th>P0</th></tr>']
    for sd, r in stud.items():
        dl = r['dl']
        h.append(f'<tr><td>s{sd}</td>' + ''.join(f'<td>{f(r.get(k), nd)}</td>' for k, _, nd in KEYS)
                 + f'<td>{f(dl["student_goal"])} · {f(dl["student_vColl"])} · {f(dl["agree"])}</td>'
                 + f'<td class="{"ok" if r["p0"] else "bad"}">{"통과" if r["p0"] else "불통과"}</td></tr>')
    h.append('<tr class="ref"><td>f2_ 학생 평균</td>' + ''.join(f'<td>{f(sm[k], nd)}</td>' for k, _, nd in KEYS) + '<td></td><td></td></tr>')
    h.append('<tr class="ref"><td>i_ 학생 평균 (10-02)</td>' + ''.join(f'<td>{f(im[k], nd)}</td>' for k, _, nd in KEYS) + '<td></td><td></td></tr>')
    for nm, lab in (('vo56h150', '선생님 vo56h150'), ('vo56', '규칙 배 vo56'), ('goal', '규칙 배 goal(안 피함)'), ('t_off', '학습 t_off')):
        h.append(f'<tr class="ref"><td>{lab}</td>' + ''.join(f'<td>{f(M(ref[nm], k), nd)}</td>' for k, _, nd in KEYS) + '<td></td><td></td></tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_diag_short():
    rows = [('s43_teacher', '선생님이 몲 · 학생 s43'), ('s43_student', '학생 s43 이 몲(샘플)'), ('s44_student', '학생 s44 이 몲(샘플)')]
    h = ['<table><tr><th>창 (300결정 × 2048척)</th><th>56 m 안에 배 있음 %</th><th>피함 라벨(≠ 0) %</th><th>일치 전체 %</th><th>일치: 라벨 0 %</th><th>일치: 피함 라벨 %</th>'
         '<th>방향 같음: 피함 라벨 %</th><th>|라벨| 평균(피함)</th><th>|학생 평균 행동|(피함)</th><th>실행 잡음 표준편차</th></tr>']
    for k, lab in rows:
        d = DG[k]
        h.append(f'<tr><td>{lab}</td><td>{f(d["ship_within_56m_pct"])}</td><td>{f(d["label_nonzero_pct"])}</td><td>{f(d["agree_all_pct"])}</td>'
                 f'<td>{f(d["agree_when_label_zero_pct"])}</td><td>{f(d["agree_when_label_nonzero_pct"])}</td><td>{f(d["sign_match_when_label_nonzero_pct"])}</td>'
                 f'<td>{f(d["mean_abs_label_nonzero"], 3)}</td><td>{f(d["mean_abs_student_when_label_nonzero"], 3)}</td><td>{f(d["sampled_std_a0"], 3)}</td></tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_diag_long():
    rows = [('long_s43_sampled', '학생 s43 · 샘플 행동(주 평가와 같은 방식)'), ('long_s43_mean', '학생 s43 · 평균 행동(잡음 없음)'),
            ('long_s44_mean', '학생 s44 · 평균 행동(잡음 없음)'), ('long_s43_teacher', '선생님 vo56h150')]
    h = ['<table><tr><th>누가 몲 (2500결정 × 2048척)</th><th>끝난 ep</th><th>도착 %</th><th>배끼리 충돌 %</th><th>피함 라벨 %</th><th>일치: 피함 라벨 %</th></tr>']
    for k, lab in rows:
        d = DG[k]
        h.append(f'<tr><td>{lab}</td><td>{d["eps"]:.0f}</td><td>{f(d["goal_pct"])}</td><td>{f(d["vColl_pct"])}</td><td>{f(d["label_nonzero_pct"])}</td>'
                 f'<td>{f(d["agree_when_label_nonzero_pct"])}</td></tr>')
    h.append('</table>')
    return '\n'.join(h)


def tbl_vram():
    rows = [('m2', 'DAgger', '기본값(config.py) <code>max_split_size_mb:128</code>'), ('v_split_gc30', 'DAgger', '<code>max_split_size_mb:128,garbage_collection_threshold:0.3</code>'),
            ('v_gc30', 'DAgger', '<code>garbage_collection_threshold:0.3</code> ← f2_ 에 씀'), ('v_nosplit', 'DAgger', '<code>max_split_size_mb:16384</code>'),
            ('t_default', 'trunk 학습', '기본값'), ('t_gc30', 'trunk 학습', '<code>garbage_collection_threshold:0.3</code> ← f2_ 에 씀'), ('t_nosplit', 'trunk 학습', '<code>max_split_size_mb:16384</code>')]
    h = ['<table><tr><th>프로세스</th><th>할당기 설정 (PYTORCH_CUDA_ALLOC_CONF)</th><th>실사용 최대 GB</th><th>reserved 최대 GB</th><th>진행</th></tr>']
    for k, kind, conf in rows:
        v = VR[k]
        if kind == 'DAgger':
            last = v['decs'][-1] if v['decs'] else (0, 0.0)
            prog = f'{last[0]}결정 / {f(last[1])}분' + (f' — {v["abort"][1]} s 에 {f(v["abort"][0], 2)} GB 로 자동 중단' if v['abort'] else '') \
                + (' — 이후 15분 넘게 무진행(수동 종료)' if k == 'm2' else '')
        else:
            prog = f'0.33M 결정, {v["dps"][-1] if v["dps"] else "?"} dec/s'
        h.append(f'<tr><td>{kind}</td><td>{conf}</td><td>{f(v["peak_alloc"], 2) if v["peak_alloc"] is not None else f(max(t[1] for t in v["trace"]), 2) + " (추적 중 최대)"}</td>'
                 f'<td class="{"bad" if v["peak_res"] > 11 else ""}">{f(v["peak_res"], 2)}</td><td>{prog}</td></tr>')
    h.append('</table>')
    return '\n'.join(h)


dl_mean = {k: mean([r['dl'][k] for r in stud.values()]) for k in ('student_goal', 'student_vColl', 'agree', 'loss')}
gap_noise = mean([DG['long_s43_mean']['goal_pct'], DG['long_s44_mean']['goal_pct']]) - DG['long_s43_sampled']['goal_pct']

html = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Fig1 f2_ 흉내 P0 결과</title>
<style>
@page {{ size: A4; margin: 12mm 11mm; }}
body {{ font-family: "Malgun Gothic", sans-serif; font-size: 9.5pt; line-height: 1.5; color: #111; background: #fff; }}
h1 {{ font-size: 14pt; margin: 0 0 4px; }}
h2 {{ font-size: 11.8pt; margin: 14px 0 6px; border-bottom: 2px solid #333; padding-bottom: 2px; }}
.sub {{ color: #555; font-size: 8.3pt; margin-bottom: 8px; }}
table {{ border-collapse: collapse; margin: 5px 0 7px; font-size: 7.9pt; }}
th, td {{ border: 1px solid #999; padding: 2px 4px; text-align: right; vertical-align: top; }}
th {{ background: #eee; text-align: center; }} td:first-child, th:first-child {{ text-align: left; }}
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

<h1>2026-10-03 Fig1 배치 f2_ — 흉내 0단계(P0) 결과 + 10-02 정지 사고 원인</h1>
<div class="sub">스펙 <code>docs/superpowers/specs/2026-10-02-fig1-latent-design.md</code> §4 · 코드 <code>feat/fig1-latent</code> {' · '.join(CODE)} (학습 코드는 2986481 과 같음) ·
Windows LICS-SERVER 2026-10-03 14:27–15:53 · 원자료 <code>runs/2026-10-02_fig1/out/</code> · 이 문서 = <code>runs/2026-10-02_fig1/report/make_report_p0.py</code> 가 로그에서 직접 읽어 만듦</div>

<h2>0. 요약</h2>
<div class="key"><ul>
<li><b>{P0_LINE}</b> — 학생 평균 도착 {f(sm['goal'])} % · 배끼리 충돌 {f(sm['vColl'])} % (기준 시드마다 ≥ 80 % · ≤ 15 %). 스펙 §4 대로 재시도 없이 멈춤, <b>phase1(trunk·코덱·갈래) 안 돌림</b></li>
<li>학생은 안 피하는 goal 배(도착 {f(M(ref['goal'], 'goal'))} · 충돌 {f(M(ref['goal'], 'vColl'))} %) 수준. 선생님 vo56h150 {f(M(ref['vo56h150'], 'goal'))} · {f(M(ref['vo56h150'], 'vColl'))} %,
10-02 i_ 학생(타각 행동) {f(im['goal'])} · {f(im['vColl'])} % 보다 낮음</li>
<li>DAgger 로그의 '선생님과 일치 {f(dl_mean['agree'])} %' 는 라벨 불균형 때문: 선생님 라벨의 약 {f(100 - DG['s43_teacher']['label_nonzero_pct'], 0)} % 가 '그대로 감(0)'. 피함 라벨에서 학생 일치는 {f(DG['s43_student']['agree_when_label_nonzero_pct'], 0)}–{f(DG['s43_teacher']['agree_when_label_nonzero_pct'], 0)} %, 변침 크기는 선생님의 절반쯤 (§3)</li>
<li>주 평가는 학생의 <b>샘플 행동</b>으로 몲 — 침로 명령 잡음 표준편차 {f(DG['s43_student']['sampled_std_a0'], 2)}(≈ {f(DG['s43_student']['sampled_std_a0'] * 120, 0)}°, DAgger 는 logstd 를 학습 안 함).
진단 창에서 같은 학생을 평균 행동(잡음 없음)으로 몰면 도착 {f(DG['long_s43_sampled']['goal_pct'])} → {f(DG['long_s43_mean']['goal_pct'])}·{f(DG['long_s44_mean']['goal_pct'])} %, 충돌 {f(DG['long_s43_sampled']['vColl_pct'])} → {f(DG['long_s43_mean']['vColl_pct'])}·{f(DG['long_s44_mean']['vColl_pct'])} %
(같은 창 선생님 {f(DG['long_s43_teacher']['goal_pct'])} · {f(DG['long_s43_teacher']['vColl_pct'])} %). 잡음을 빼도 P0 기준에는 못 미침 (§3, 진단 — 판정 아님)</li>
<li><b>10-02 f_ 정지·TDR 원인 = 실험이 아니라 메모리 설정 두 가지</b>(§4): ① <code>config.py</code> 기본 할당기 설정이 DAgger 캐시를 {f(VR['m2']['peak_res'], 1)} GB 까지 불림(실사용 {f(VR['v_gc30']['peak_alloc'], 1)} GB)
② trunk 학습은 프로세스당 {f(VR['t_gc30']['peak_res'], 1)} GB → GPU당 2개 불가. f2_ 는 환경변수 하나 + GPU당 개수만 바꿔 멈춤 없이 끝남</li>
</ul></div>

<h2>1. 무엇을 돌렸나</h2>
<ul>
<li>f2_ = 10-02 f_ 의 재실행. 실험 설정·기준은 <code>Python/runs_fig1/2026-10-02_fig1/_run_f.sh</code> 그대로(목표 침로 행동 · ARPA@56·메시지 없음 · 선생님 vo56h150 · DAgger 3000결정 · 시드 43–47). 접두어만 f2_(HEAD 가 9183a0f 로 바뀌어 스크립트 규칙대로 새 접두어)</li>
<li>바꾼 것은 실행 환경뿐: <code>PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3</code>(메모리 할당기 — 계산 결과와 무관) · phase0 GPU당 2개 / phase1 GPU당 1개 (<code>report/diag/f2_launch.sh</code>)</li>
<li>순서: smoke(preflight 14종 강제, 통과) 14:27–14:31 → DAgger 5시드 14:31–15:19 → 학생 주 평가 15:19–15:53 → P0 표</li>
<li>P0 = 흉내만 낸 학생(PPO 전)을 주 평가(256 env · 10000결정 · burn-in 2400 · 평가 시드 999, 5칸 <code>[run]</code> 줄 확인)로 채점. 기준은 결과 전에 고정</li>
</ul>

<h2>2. P0 결과</h2>
{tbl_p0()}
{fig_p0()}
<div class="cap">그림 1. 주 평가 도착 vs 배끼리 충돌. 빨강 = f2_ 학생(시드 5개), 보라 = 10-02 i_ 학생, 네모 = 규칙 배 평균(평가 시드 3개), 세모 = 학습 t_off 평균. 초록 = P0 통과 영역.</div>
<ul>
<li>시드 간 차이 작음(도착 {f(min(r['goal'] for r in stud.values()))}–{f(max(r['goal'] for r in stud.values()))} %) → 시드 운이 아니라 5개 모두 같은 양상</li>
<li>방향 바꾼 총량 {f(sm['head'], 0)}°(선생님 {f(M(ref['vo56h150'], 'head'), 0)}°, goal 배 {f(M(ref['goal'], 'head'), 0)}°) · 연료 {f(sm['fuel'], 0)}(선생님 {f(M(ref['vo56h150'], 'fuel'), 0)}) — 학생이 많이 흔들림</li>
<li>i_ 학생과는 행동(타각 vs 목표 침로)·입력(레이더만 vs ARPA@56)·선생님(vo56 vs vo56h150)이 다 달라 어느 것 때문인지는 이 비교로 못 가림</li>
</ul>

<div class="pb"></div>
<h2>3. 진단 — 왜 '일치 98 %' 인데 성적이 낮은가 (측정만, 판정 아님)</h2>
<div class="note">아래는 P0 뒤에 따로 잰 것(<code>report/diag/label_stats.py</code>, 학습·저장 없음). env 128개 · reset 뒤 300결정 버리고 잼 — 주 평가(256 env · 10000결정)와 창이 달라 숫자를 주 평가와 직접 비교하지 말 것. 시드 1–2개.</div>
<b>3-1. 선생님 라벨 분포와 조건부 일치 (300결정 창)</b>
{tbl_diag_short()}
<ul>
<li>일치 = |tanh(학생 평균 a0) − 선생님 a0| &lt; 0.2 (DAgger 로그와 같은 정의). a0 = 목표 방위 기준 침로 변경량 / 120°</li>
<li>라벨 대부분이 0 → 항상 0 을 내도 일치가 90 % 안팎. DAgger 로그의 일치·손실({f(dl_mean['loss'], 2)} 로 평평)은 학생이 피하는지 알려 주지 못함</li>
</ul>
<b>3-2. 같은 학생을 샘플 행동 vs 평균 행동으로 몰기 (2500결정 창)</b>
{tbl_diag_long()}
{fig_diag()}
<div class="cap">그림 2. 진단 창(2500결정)의 도착·충돌. 점선 = P0 기준(참고용 — 이 창은 P0 판정 창이 아님).</div>
<ul>
<li>샘플 행동 → 평균 행동: 도착 +{f(gap_noise, 0)} pp 안팎. 실행 잡음이 성적을 크게 깎음. <code>eval/eval_ckpt.py</code> 는 샘플 행동으로 몲(학습 배 평가와 같은 방식)</li>
<li>평균 행동으로도 선생님과 도착 {f(DG['long_s43_teacher']['goal_pct'] - DG['long_s43_mean']['goal_pct'], 0)} pp 안팎 차이 → 잡음만의 문제가 아님(피함 라벨 일치 {f(DG['long_s43_mean']['agree_when_label_nonzero_pct'], 0)}–{f(DG['long_s44_mean']['agree_when_label_nonzero_pct'], 0)} %)</li>
<li>확인 안 한 것: 잡음을 줄였을 때 주 평가 수치, 라벨 불균형을 보정했을 때 효과, i_ 대비 어느 변경(행동·입력·선생님)이 원인인지</li>
</ul>

<div class="pb"></div>
<h2>4. 10-02 f_ 정지 사고 — 원인 측정과 조치</h2>
<ul>
<li>경과: 10-02 17:08 DAgger 5시드 시작 → 17:12–17:15 무진행(GPU 1·2·3 = 15.7–15.8 GB) → s47 만 22:44 완료(333분) → 22:45 nvlddmkm 153 ×5(TDR), s43·s44 <code>CUDA error: unknown error</code>
→ 10-03 00:04 서버 비정상 종료·00:07 재부팅(Kernel-Power 41, 종료 원인 미확인). P0 판정 없이 끝남</li>
<li>재현: 10-03 단일 프로세스로도 같은 정지(200결정 뒤 무진행, GPU 15.9 GB) → 여러 개를 한 GPU 에 올린 탓만이 아님</li>
</ul>
{tbl_vram()}
{fig_vram()}
<div class="cap">그림 3. 왼쪽 = DAgger 프로세스 1개의 PyTorch reserved 메모리(할당기 설정별). 오른쪽 = 프로세스당 실사용 최대와 reserved(gc 0.3). trunk 는 설정과 무관하게 8.8 GB.</div>
<ul>
<li>원인 ①: <code>config.py</code> 의 기본 <code>max_split_size_mb:128</code> 아래에서 DAgger(ON·파트너 8·미니배치 2048)의 캐시가 실사용의 4배 넘게 쌓임. 이 설정만 빼면 4.2 GB 고정·정상 속도.
할당기 설정은 메모리 관리만 바꿈(계산 결과 불변) → 코드·HEAD 안 바꾸고 환경변수로 덮어씀</li>
<li>원인 ②: trunk 학습 실사용 최대 {f(VR['t_gc30']['peak_alloc'], 2)} GB(할당기와 무관한 실제 수요). 스크립트 선언값 3000 MB · GPU당 2개면 16 GB 초과 → phase1 은 GPU당 1개로 띄우게 해 둠(이번엔 P0 불통과라 실행 안 됨)</li>
<li>f2_ 실행 중 무진행·TDR 없음. DAgger 중 본 GPU 사용량 4.6–9.6 GB(14:57 표본 1회, 다른 계정 작업 포함)</li>
</ul>

<h2>5. 읽을 때 주의</h2>
<div class="note"><ul>
<li>P0 판정 근거는 §2 주 평가뿐. §3 은 다음 결정을 위한 진단(창·시드 수가 다름)</li>
<li>§3 의 '잡음' 설명은 측정된 차이(샘플 vs 평균 행동)까지만 근거 있음. PPO 를 붙이면 어떻게 되는지는 안 돌려 봄</li>
<li>CLAUDE.md 에 '서버 불안정으로 다른 기계 실행'이 적혀 있음 — 다른 기계에서 f_ 가 돌았다면 이 f2_ 와 같은 설정·시드의 중복 실행임(결과 대조 가능)</li>
<li>10-02 f_ 의 s47 DAgger 체크포인트(정지 중 완료)는 이번에 안 씀 — f2_ 는 5시드 전부 새로 흉내</li>
</ul></div>

<h2>6. 다음 — 저자 결정</h2>
<ul>
<li>스펙 §4: P0 불통과면 재시도 없이 보고. 기준·설정 안 바꿈. phase1 은 대기(<code>C:\\work\\f2_launch.sh</code> 는 P0 통과 기록이 있어야 phase1 로 감)</li>
<li>결정할 것(제안 아님, 갈림길만): 잡음(logstd)·평가 방식 / 라벨 불균형 / 행동 방식(목표 침로) 중 무엇을 바꿔 다시 흉내 낼지, 또는 흉내 없이 trunk 로 갈지</li>
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
