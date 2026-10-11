# 2026-10-07: 배치 n_ 순위 관문 v4 불통과 보고 PDF. 학습 없음. 숫자는 전부 관문 JSON 에서 읽음:
#   runs/2026-10-02_fig1/out/_rank_v4.json                         (기록 = 배치가 돌린 관문, 배치 env 그대로 = 침로 행동 모드)
#   runs/2026-10-02_fig1/out/diag_rank/_rank_rudder_v3v4.json       (진단 = 같은 env 에서 VESSEL_ACTION_MODE=rudder 만 바꿈, v3 · v4)
#   runs/2026-10-02_fig1/out/diag_rank/t_reward_rank_2026-09-30.json (09-30 배치 t_ 의 관문 v3, 타각 모드 — 진단 v3 과 대조)
# 실행(Windows): /c/Users/OSH/anaconda3/python.exe make_report_rank_n.py [출력 폴더]
import io, json, os, subprocess, sys

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', 'out'))
NAME = '2026-10-07_n배치_순위관문v4_결과'
SEEDS = [1, 2, 3]


def load(*p):
    with io.open(os.path.join(OUT, *p), encoding='utf-8') as fh:
        return json.load(fh)


REC = load('_rank_v4.json')
DG = load('diag_rank', '_rank_rudder_v3v4.json')
T0 = load('diag_rank', 't_reward_rank_2026-09-30.json')
assert REC['args']['settings'] == ['v4'] and DG['args']['settings'] == ['v3', 'v4'], (REC['args'], DG['args'])
for d in (REC, DG, T0):
    assert (d['args']['seeds'], d['args']['E'], d['args']['burn'], d['args']['T']) == (SEEDS, 32, 600, 1500), d['args']
V4CFG = {'ROLE_PROMISE_PEN': 20.0, 'ROLE_JUDGE': 'v2', 'FORWARD_COEF': 0.0, 'TIME_PENALTY': 0.035, 'RISK_DCPA_GATE_M': 48.0,
         'ROLE_V2_PRIMARY': 'cum', 'ROLE_V2_RES_F6': 1}
assert REC['v4']['cfg'] == V4CFG and DG['v4']['cfg'] == V4CFG, (REC['v4']['cfg'], DG['v4']['cfg'])
assert DG['v3']['cfg'] == dict(V4CFG, ROLE_V2_PRIMARY='risk', ROLE_V2_RES_F6=0), DG['v3']['cfg']


def fm(v, nd=1, sign=False):
    s = f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'
    return s.replace('-', '−')


def row(d, setting, seed, pol):
    r = [x for x in d[setting]['rows'] if x['seed'] == seed and x['policy'] == pol]
    assert len(r) == 1, (setting, seed, pol)
    return r[0]


def per_seed(d, setting, pol, key, nd):
    return ' · '.join(fm(row(d, setting, s, pol)[key], nd) for s in SEEDS)


def verdict_table(d, setting):
    okx = lambda b: '성립' if b else "<b>불성립</b>"
    h = ("<table><thead><tr><th>시드</th><th>(i) 할인 반환 G<br>회피 최소 − 나머지 최대</th><th>(ii) 결정당 보상<br>vo56 − max(goal·stbd·radar)</th>"
         "<th>(iii) 에피소드 반환<br>회피 최소 − 나머지 최대</th><th>(iv) vo300 − vo56<br>G · 결정당</th>"
         "<th>(v) 판정 성공률<br>vo300s &gt; radar &gt; goal</th><th>판정</th></tr></thead><tbody>")
    for v in d['verdict'][setting]:
        i, ii, iii, iv, vv = v['i_G'], v['ii_rdec'], v['iii_ep'], v['iv_info'], v['v_rks']
        h += (f"<tr><td class='c'>{v['seed']}</td>"
              f"<td class='c'>{fm(i['lo'], 2)}[{i['lo_p']}] − {fm(i['hi'], 2)}[{i['hi_p']}] {okx(i['ok'])}</td>"
              f"<td class='c'>{fm(ii['vo56_minus_base3'], 3, True)} (기준 ≥ +0.05) {okx(ii['ok'])}</td>"
              f"<td class='c'>{fm(iii['lo'], 1)}[{iii['lo_p']}] − {fm(iii['hi'], 1)}[{iii['hi_p']}] {okx(iii['ok'])}</td>"
              f"<td class='c'>{fm(iv['dG'], 1, True)} · {fm(iv['dr'], 3, True)} {okx(iv['ok'])}</td>"
              f"<td class='c'>{fm(vv['vo300s'], 3)} · {fm(vv['radar'], 3)} · {fm(vv['goal'], 3)} {okx(vv['ok'])}</td>"
              f"<td class='c'>{'통과' if v['pass'] else '<b>불통과</b>'}</td></tr>")
    n = sum(1 for v in d['verdict'][setting] if v['pass'])
    return h + "</tbody></table>", n


T_REC, N_REC = verdict_table(REC, 'v4')
T_D3, N_D3 = verdict_table(DG, 'v3')
T_D4, N_D4 = verdict_table(DG, 'v4')
assert (N_REC, N_D3, N_D4) == (sum(v['pass'] for v in REC['verdict']['v4']), 3, 2), (N_REC, N_D3, N_D4)
assert REC['gate']['pass'] is False and DG['gate']['v3'] is True and DG['gate']['v4'] is False

# 진단 v3(타각 모드) 이 09-30 t_ 관문 v3 과 같은 값인지 — 42행 전부 대조
POLS = REC['args']['policies']
_dmax = max(abs(row(DG, 'v3', s, p)[k] - row(T0, 'v3', s, p)[k]) for s in SEEDS for p in POLS
            for k in ('G', 'r_per_dec', 'ep_return', 'vColl%', 'goal%', 'oColl%', 'roleKeptSafe'))
SAME_T = _dmax == 0.0


def behave_table():
    pols = [('goal', 'goal(목표로 직진)'), ('vo56', 'vo56(56 m 회피)'), ('vo56s', 'vo56s(우현만)'), ('vo300', 'vo300(300 m 회피)'),
            ('vo300s', 'vo300s(우현만)')]
    h = ("<table><thead><tr><th></th><th colspan='3'>기록 관문(침로 모드 env)</th><th colspan='3'>진단(타각 모드 env)</th></tr>"
         "<tr class='sub'><th>규칙 배 · 시드 1 · 2 · 3</th><th>도착 %</th><th>배 충돌 %</th><th>벽 충돌 %</th>"
         "<th>도착 %</th><th>배 충돌 %</th><th>벽 충돌 %</th></tr></thead><tbody>")
    for p, lab in pols:
        h += f"<tr><td class='l'>{lab}</td>"
        for d in (REC, DG):
            for k in ('goal%', 'vColl%', 'oColl%'):
                h += f"<td class='s'>{per_seed(d, 'v4', p, k, 1)}</td>"
        h += "</tr>"
    return h + "</tbody></table>"


def rks_table():
    pols = [('vo300s', 'vo300s'), ('vo300', 'vo300'), ('vo56', 'vo56'), ('radar', 'radar(레이더 안 규칙)'), ('stbd', 'stbd(무조건 우현)'),
            ('goal', 'goal(안 피함)')]
    h = ("<table><thead><tr><th></th><th colspan='2'>판정 성공률(roleKeptSafe)</th><th colspan='2'>판정 수</th><th colspan='2'>할인 반환 G</th></tr>"
         "<tr class='sub'><th>진단(타각 모드) · 시드 1 · 2 · 3</th><th>v3</th><th>v4</th><th>v3</th><th>v4</th><th>v3</th><th>v4</th></tr></thead><tbody>")
    for p, lab in pols:
        h += f"<tr><td class='l'>{lab}</td>"
        for k, nd in (('roleKeptSafe', 3), ('judged', 0), ('G', 1)):
            for s in ('v3', 'v4'):
                h += f"<td class='s'>{per_seed(DG, s, p, k, nd)}</td>"
        h += "</tr>"
    gap = lambda s: ' · '.join(fm(row(DG, s, x, 'vo300s')['roleKeptSafe'] - row(DG, s, x, 'radar')['roleKeptSafe'], 3, True) for x in SEEDS)
    h += (f"<tr><td class='l'><b>vo300s − radar</b> (규칙 (v) 앞 절)</td><td class='s'>{gap('v3')}</td><td class='s'><b>{gap('v4')}</b></td>"
          "<td></td><td></td><td></td><td></td></tr>")
    return h + "</tbody></table>"


_v1 = [v for v in DG['verdict']['v4'] if v['seed'] == 1][0]['v_rks']
GAP1 = _v1['vo300s'] - _v1['radar']
J1 = (row(DG, 'v4', 1, 'vo300s')['judged'], row(DG, 'v4', 1, 'radar')['judged'])
SEC = (REC['v4']['sec'], DG['v3']['sec'], DG['v4']['sec'])

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
th { text-align: center; font-weight: 600; padding: 2px 4px; border-bottom: 1px solid #0b0b0b; }
tr.sub th { font-weight: 400; color: #52514e; font-size: 7.5pt; border-bottom: 1px solid #c9c8c3; }
td { padding: 1.8px 4px; border-bottom: 1px solid #e4e3df; vertical-align: top; }
td.l { width: 24%; } td.s { text-align: center; color: #33322f; white-space: nowrap; font-size: 7.6pt; }
td.c { text-align: center; font-size: 7.5pt; }
code { font-family: Consolas, monospace; font-size: 8pt; background: #f4f3ef; padding: 0 2px; }
.small { color: #52514e; font-size: 7.9pt; }
"""
HTML = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{NAME}</title><style>{CSS}</style></head><body>
<h1>배치 n_ — 순위 관문 v4 불통과 보고</h1>
<div class='sub1'>2026-10-07 · LICS 서버 · 코드 feat/fig1-latent f42ff8f · 학습 안 함 · 규칙 배 14종 × 시드 3 · 32 env × 16척 · burn-in 600 · 1500 결정</div>

<div class='box'><b>한 줄 결론</b> — 순위 관문 v4 가 <b>{N_REC}/3 시드</b>로 불통과 → 스펙 §5-2 대로 멈춤, trunk 학습 시작 안 함.
관문 도구에 결함 확인(규칙 배는 타각 단위 행동을 내는데 배치 env 는 침로 모드). 결함을 뺀 진단(타각 모드)에서도 v4 는 <b>{N_D4}/3</b>
(시드 1 이 규칙 (v) 불성립) → 어느 쪽으로 읽어도 '시드마다 전부 성립' 기준에 못 미침. 다음은 저자 결정.</div>

<h2>1. 경과</h2>
<ul>
<li>15:23 첫 실행(1046823) → 15:27 preflight '통신 미러'에서 멈춤. 원인 = 검증기 <code>_verify_comm_mirror.py</code> 가 새 토글 2개를 기본값으로 안 묶어 배치 값 <code>cum</code> 이 판정기 없는 p12 케이스로 샘(미러 불일치 아님, 검증기 크래시)</li>
<li>저자 승인 → f42ff8f(검증기 파일만, +4줄) push → 15:41 같은 접두어 n_ 로 다시 실행 → 15:46 smoke 통과(preflight 15종 ALL PASS)</li>
<li>15:46–17:03 순위 관문 v4({fm(SEC[0] / 60, 0)}분) → <b>불통과, 멈춤</b>. 이후 단계(trunk · 코덱 · 갈래 · 평가) 전부 안 돎</li>
<li>17:06–18:22 진단 재측정(v3 · v4, 타각 모드, 각 {fm(SEC[1] / 60, 0)} · {fm(SEC[2] / 60, 0)}분) — 기록용. 관문 판정은 위 불통과 그대로</li>
</ul>

<h2>2. 관문 판정(기록) — 배치 env 그대로(침로 행동 모드), 설정 v4</h2>
{T_REC}
<ul>
<li>v4 = v3 + R1(<code>VESSEL_ROLE_V2_PRIMARY=cum</code>) + R2(<code>VESSEL_ROLE_V2_RES_F6=1</code>). 규칙 (i)–(v) 가 시드마다 전부 성립해야 통과</li>
<li>(i)(ii)(iii) 3/3 시드 불성립: 56 m 회피 배(vo56 · vo56s)가 안 피하는 배(goal · turnless)보다 낮게 나옴. (iv)(v) 는 3/3 성립</li>
</ul>

<h2>3. 관문 도구 결함 — 행동 모드 불일치(코드로 확인)</h2>
<ul>
<li><code>verify/check_reward_rank.py</code> 규칙 배는 a0 를 타각 단위로 냄(<code>steer_to</code> = 침로 오차 / 10°, <code>goal_a0</code> = 오차 / 20°)</li>
<li>실행기 <code>_run_f.sh</code> 가 <code>VESSEL_ACTION_MODE=course</code> 를 export 하고 관문이 그대로 물려받음 → env 는 a0 를 '목표 방위 기준 침로 변경량 / 120°' 로 읽음</li>
<li>관문 스크립트에 행동 모드 처리가 없음. 선생님 코드(<code>imitation/vo_teacher.py</code>)에는 침로 단위 경로가 따로 있음</li>
<li>같은 규칙 배가 env 의 행동 모드만 달라도 다르게 움직임(아래). 침로 모드에서 회피 배의 도착률이 내려가고 벽 충돌이 생김</li>
</ul>
{behave_table()}
<div class='small'>두 열 모두 설정 v4. 진단 열은 배치 env 에서 <code>VESSEL_ACTION_MODE=rudder</code> 하나만 바꿈.</div>

<h2>4. 진단 재측정 — 타각 모드(기록용, 관문 판정 아님)</h2>
<div class='small'>설정 v3 (R1 · R2 끔)</div>
{T_D3}
<div class='small'>설정 v4 (R1 · R2 켬)</div>
{T_D4}
<ul>
<li>v3: {N_D3}/3 통과. 09-30 배치 t_ 의 관문 v3 과 {'42행 전부 같은 값(차 0)' if SAME_T else f'최대 차 {_dmax:.3g}'} → 타각 모드에서는 그날 이후 달라진 것 없음</li>
<li>v4: 규칙 (i)–(iv) 는 3/3 성립(보상 순위는 v3 과 같은 순서). <b>규칙 (v) 가 시드 1 에서 불성립</b>: vo300s − radar = {fm(GAP1, 3, True)}
(판정 수 {J1[0]} · {J1[1]}) → {N_D4}/3</li>
</ul>

<h2>5. v4 가 바꾼 것(진단, 타각 모드)</h2>
{rks_table()}
<ul>
<li>판정 성공률이 모든 규칙 배에서 내려감. 판정 수와 할인 반환은 거의 그대로</li>
<li>잘 피하는 배(vo300s)와 레이더 규칙 배(radar)의 성공률 차가 v3 에서는 3/3 시드 양수였고 v4 에서는 2/3 양수 · 1/3 음수</li>
<li>R1 · R2 중 어느 쪽 몫인지는 나눠 재지 않았음(둘을 함께 켠 설정 하나만 돌림)</li>
</ul>

<h2>6. 한계 — 그대로 읽을 것</h2>
<ul>
<li>기록 관문의 (i)(ii)(iii) 불성립은 행동 모드 불일치와 섞여 있음 → 보상 v4 가 순위를 뒤집는다는 근거로 못 씀</li>
<li>진단은 타각 모드임. 실제 학습은 침로 모드. 규칙 배를 침로 단위로 바꿔서 잰 관문은 아직 없음</li>
<li>시드 1 의 (v) 차는 {fm(GAP1, 3, True)} 로 작음. 규칙 (v) 에는 여유 폭이 없어(부등호만) 작은 차로도 갈림. 기준은 결과 전에 고정된 것이라 안 바꿈</li>
<li>평가 시드 3개 · 정책당 에피소드 약 750–980. 학습 결과 아님</li>
</ul>

<h2>7. 저자 결정 대기</h2>
<ul>
<li>관문 도구를 침로 모드에 맞게 고쳐 v4 를 다시 판정할지(규칙 배 행동을 침로 단위로 변환, 또는 관문만 타각 모드로 고정). 진단 기준으로는 고쳐도 2/3</li>
<li>R2(해소 종료에도 F6) · R1(누적 주 상대)을 따로 켜서 어느 쪽이 (v) 를 흔드는지 볼지(설정당 약 {fm(SEC[2] / 60, 0)}분, 학습 없음)</li>
<li>n_ 을 멈춘 채 보상 · 판정을 다시 볼지</li>
<li>결정 전까지 학습은 안 돌림. 관문 스크립트 · 기준은 안 건드렸음</li>
</ul>
<div class='small'>자료: runs/2026-10-02_fig1/out/_rank.log · _rank_v4.json(기록) · out/diag_rank/(진단 로그 · JSON · 09-30 t_ 관문 JSON) ·
서버 Python/_repro_out_n/ · 실패한 첫 실행 Python/_repro_out_n_try1_1046823/ · 이 문서 report/make_report_rank_n.py</div>
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
print('기록 v4', N_REC, '/3 | 진단 v3', N_D3, '/3 · v4', N_D4, '/3 | 시드1 vo300s−radar', fm(GAP1, 3, True), '| t_ 대조 최대차', _dmax)
print(out_pdf)
