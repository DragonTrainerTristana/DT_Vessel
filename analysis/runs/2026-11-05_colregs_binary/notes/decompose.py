#!/usr/bin/env python3
"""decompose.py — 이진 COLREGs 재평가(스펙 2026-11-02 §9) 수치 분해, 시드 43–45, off vs comm (24M 끝 모델, 평균 행동).

읽기 전용: lrmb_*.txt(이진 줄) + lrm_*.txt(나머지 지표)만 파싱. 학습·평가·torch 없음.
판정은 goals_fig1.py 표(a·a2)뿐 — 여기 나머지는 분해·탐색(판정 아님).
"""
import difflib
import math
import re
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / '2026-10-02_fig1' / 'out'   # analysis/runs/2026-10-02_fig1/out
SEEDS = {43: 'qr', 44: 'qr2', 45: 'qr3'}
ARMS = ('off', 'comm')
T2 = 4.303                       # t(0.975, df 2)
NOISE = re.compile(r'경과|dec/s|UserWarning|untyped_storage')
ROLES_BIN = ('양보', '정면', '추월', '유지')
ROLES_W = ('giveway', 'headon', 'overtaking', 'standon')       # 같은 조우 순서(양보·정면·추월·유지)
NUM = r'([-+]?\d+(?:\.\d+)?)'


def read(kind, seed, arm):
    d = SEEDS[seed]
    return (OUT / d / f'{kind}_{d}_{arm}_s{seed}.txt').read_text(encoding='utf-8').splitlines()


def pick(L, tag, contains=''):
    hits = [l for l in L if l.strip().startswith(tag) and contains in l]
    assert len(hits) == 1, (tag, contains, len(hits))
    return hits[0]


def num(s, label):
    m = re.search(r'(?<!\S)' + re.escape(label) + r'=\s*' + NUM, s)
    return float(m.group(1)) if m else None


def parse(seed, arm):
    L, B = read('lrm', seed, arm), read('lrmb', seed, arm)
    r = {}
    b = pick(B, '[binary-COLREGs]')
    r['bin'] = num(b, 'rate')
    r['bin_n'] = float(re.search(r'\(n=(\d+) 조우', b).group(1))
    for k in ROLES_BIN:
        m = re.search(re.escape(k) + r'=\s*' + NUM + r'%\(n=(\d+)\)', b)
        r[f'bin_{k}'], r[f'n_{k}'] = float(m.group(1)), float(m.group(2))
    main = [l for l in L if '|| [goal-ep]' in l]
    assert len(main) == 1
    main = main[0]
    for k in ('goal', 'vColl', 'oColl', 'TO', 'fuel', 'headTravel', 'minSep', 'len', 'allMinSep', 'epReward',
              'colregs', 'colregsOK'):
        r[k] = num(main, k)
    r['eps'] = float(re.search(r'\|\s*(\d+) eps', main).group(1))
    w = pick(L, '[woerner-COLREGs]')
    r['W'] = num(w, 'score')
    r['W_n'] = float(re.search(r'\(n=(\d+) 조우', w).group(1))
    wr = pick(L, '[woerner-COLREGs/역할]')
    for nm, sc, inner in re.findall(r'(giveway|headon|overtaking|standon)=\s*([\d.]+)%\(([^)]*)\)', wr):
        r[f'W_{nm}'] = float(sc)
        for k, v in re.findall(r'(\S+?)=\s*([-\d.]+)%?', inner):
            r[f'W_{nm}_{k}'] = float(v)
    for zone, tag in (('in', '[조율/레이더 안 ≤56m]'), ('out', '[조율/레이더 밖 56-300m]')):
        g = pick(L, tag, '양보-유지')
        r[f'{zone}_gs_n'] = float(re.search(r'쌍결정 n=(\d+)', g).group(1))
        for lab, key in (('보완(양보 우현+유지 유지)', 'compl'), ('양보 우현', 'gw_stbd'), ('유지 유지', 'so_keep'),
                         ('유지 좌현', 'so_port'), ('둘 다 유지', 'both_keep')):
            r[f'{zone}_{key}'] = num(g, lab) if lab != '보완(양보 우현+유지 유지)' else \
                float(re.search(re.escape(lab) + r'=\s*' + NUM, g).group(1))
        h = pick(L, tag, 'HeadOn 쌍결정')
        r[f'{zone}_ho_n'] = float(re.search(r'HeadOn 쌍결정 n=(\d+)', h).group(1))
        r[f'{zone}_ho_both'] = num(h, '둘 다 우현')
        r[f'{zone}_role_n'] = float(re.search(r'역할 붙은 쌍 n=(\d+)', h).group(1))
        r[f'{zone}_mesh'] = num(h, '역할 맞물림')
    for l in L:
        m = re.search(r'\[pair-heading/Δψ>=\s*(\d+)deg\]', l)
        if m:
            t = int(m.group(1))
            r[f'ph{t}_C'] = num(l, 'C')
            for nm, v, n in re.findall(r'(HeadOn|StandOn|GiveWay|Overtake)=\s*([\d.]+)%\(n=(\d+)\)', l):
                r[f'ph{t}_{nm}'], r[f'ph_n_{nm}'] = float(v), float(n)
        m = re.search(r'\[Rule8/침로or속력>=\s*(\d+)deg\]', l)
        if m:
            t = int(m.group(1))
            r[f'r8{t}_C'] = num(l, 'C')
            for nm, v, n, sp in re.findall(r'(HeadOn|GiveWay|Overtake)=\s*([\d.]+)%\(n=(\d+), 속력만=\s*([\d.]+)%\)', l):
                r[f'r8{t}_{nm}'], r[f'r8{t}_{nm}_spd'] = float(v), float(sp)
    s = pick(L, '[GiveWay/진입DCPA층화')
    for lab, v, n in re.findall(r'(0-10m|10-20m|20-35m|35-m)=\s*([\d.]+)%\(n=(\d+)\)', s):
        r[f'strat_{lab}'], r[f'strat_n_{lab}'] = float(v), float(n)
    raw = pick(L, '[pair-heading/raw]')
    r['gw_dpsi_med'] = float(re.search(r'양보역할 Δψ 중앙\s+' + NUM, raw).group(1))
    r['so_dpsi_med'] = float(re.search(r'유지역할 \|Δψ\| 중앙\s+' + NUM, raw).group(1))
    pd = pick(L, '[pair-detail]')
    r['dcpa'] = float(re.search(r'통과최소거리 중앙\s+' + NUM, pd).group(1))
    r['first_turn_s'] = float(re.search(r'첫 변침\S* 중앙\s+' + NUM, pd).group(1))
    r['no_turn'] = float(re.search(r'무변침 조우\s+' + NUM, pd).group(1))
    oc = pick(L, '[encounter-COLREGs/실제타각]')
    r['oldC'] = num(oc, 'C')
    for tag, p in (('[role-promise]', 'rp1'), ('[role-promise/v2]', 'rp2')):
        g = pick(L, tag)
        for k in ('roleKeptSafe', 'both_comply', 'safe', 'pair_coll', 'discarded', 'started'):
            r[f'{p}_{k}'] = num(g, k)
        r[f'{p}_judged'] = float(re.search(r'judged n=(\d+)', g).group(1))
    sr = pick(L, '[scripted-rudder]')
    ge, ae = sr.split('|')
    r['rud_goal'] = num(ge, 'meanAbsRudder')
    r['rud_all'] = num(ae, 'meanAbsRudder')
    r['rudtrav_goal'] = num(ge, 'rudderTravel')
    return r


D = {(s, a): parse(s, a) for s in SEEDS for a in ARMS}
S3 = list(SEEDS)


def ci(d):
    m = st.mean(d)
    h = T2 * st.stdev(d) / math.sqrt(len(d))
    return m, m - h, m + h


def f(x, nd=1):
    return '–' if x is None else f'{x:.{nd}f}'


def row(name, key=None, hb=True, nd=1, vals=None, show_ci=True):
    """hb: True = 높을수록 좋음, False = 낮을수록 좋음, None = 방향 없음(comm>off 수만 셈)."""
    off = [D[(s, 'off')][key] for s in S3] if vals is None else vals[0]
    com = [D[(s, 'comm')][key] for s in S3] if vals is None else vals[1]
    if any(v is None for v in off + com):
        return f'| {name} | ' + ' · '.join(f(v, nd) for v in off) + ' | ' + ' · '.join(f(v, nd) for v in com) + ' | 값 없음 | | | |'
    d = [c - o for o, c in zip(off, com)]
    m, lo, hi = ci(d)
    if hb is None:
        w = f'comm>off {sum(x > 0 for x in d)}/3 (방향 없음)'
    else:
        w = f'{sum((x > 0) if hb else (x < 0) for x in d)}/3' + (' (낮을수록 좋음)' if hb is False else '')
    ties = sum(x == 0 for x in d)
    if ties:
        w += f' · 동률 {ties}'
    c = f'[{lo:+.2f}, {hi:+.2f}]' if show_ci else ''
    return (f'| {name} | ' + ' · '.join(f(v, nd) for v in off) + ' | ' + ' · '.join(f(v, nd) for v in com) + ' | '
            + ' · '.join(f'{x:+.{nd}f}' for x in d) + f' | {m:+.2f} | {c} | {w} |')


HDR = ('| 지표 | off (43·44·45) | comm (43·44·45) | 짝 차 comm−off | 평균차 | 95 % t-CI (df 2) | 승 |\n'
       '|---|---|---|---|---|---|---|')


def section(t):
    print(f'\n## {t}\n')


# ── 1. 이진 준수율 ─────────────────────────────────────────────
section('1. 이진 COLREGs 준수율 (lrmb, 스펙 §9) — 전체·역할별·조우 수·역할 구성')
print('CI 는 a) 전체 행 말고는 탐색용(판정 아님). a) 판정 = goals_fig1.py 표\n')
print(HDR)
print(row('a) 이진 준수율 전체 %', 'bin'))
for k in ROLES_BIN:
    print(row(f'이진 {k} %', f'bin_{k}'))
print(row('조우 수 전체 (n)', 'bin_n', hb=None, nd=0, show_ci=False))
for k in ROLES_BIN:
    print(row(f'조우 수 {k} (n)', f'n_{k}', hb=None, nd=0, show_ci=False))
for s in S3:
    for a in ARMS:
        r = D[(s, a)]
        r['N'] = sum(r[f'n_{k}'] for k in ROLES_BIN)
        for k in ROLES_BIN:
            r[f'sh_{k}'] = 100 * r[f'n_{k}'] / r['N']
        r['enc_per_ep'] = r['N'] / r['eps']
for k in ROLES_BIN:
    print(row(f'역할 구성 {k} 비중 %', f'sh_{k}', hb=None, nd=2, show_ci=False))
print(row('조우 / 끝난 에피소드', 'enc_per_ep', hb=None, nd=2, show_ci=False))
print(row('끝난 에피소드 수 (eps)', 'eps', hb=None, nd=0, show_ci=False))
print('\n조우 수 합 검사(역할 n 합 = 전체 n, Woerner n 과 같음):')
for s in S3:
    for a in ARMS:
        r = D[(s, a)]
        print(f'- s{s} {a}: Σ역할 n={r["N"]:.0f} · 전체 n={r["bin_n"]:.0f} · Woerner n={r["W_n"]:.0f} · '
              f'역할 n 같음(이진=Woerner)={all(r[f"n_{b}"] == r[f"W_{w}_n"] for b, w in zip(ROLES_BIN, ROLES_W))}')


# ── 2. 구성 vs 비율 분해 ──────────────────────────────────────────
def decompose(rate_keys, rep_key, label):
    section(f'2. 구성 vs 비율 분해 — {label} (역할 비율은 0.1 % 반올림값 → 재구성 전체와 보고값 차 ≤ 0.05 pt)')
    print('rate 효과 = 같은 역할 구성에서 비율만 바꾼 차 · mix 효과 = 남는 차(역할 구성 차). '
          'Kitagawa = 두 팔 평균 구성·평균 비율로 나눈 정확 합\n')
    print('| 시드 | off 보고 | comm 보고 | off 재구성 | comm 재구성 | 전체 차(재구성) | comm@off구성 | rate(off구성) | mix(off구성) '
          '| off@합동구성 | comm@합동구성 | rate(합동구성) | mix(합동, 나머지) | Kitagawa rate | Kitagawa mix |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    agg = {k: [] for k in ('tot', 'rate_o', 'mix_o', 'rate_p', 'mix_p', 'kr', 'km')}
    per_role = {}
    for s in S3:
        o, c = D[(s, 'off')], D[(s, 'comm')]
        so = {k: o[f'n_{k}'] / o['N'] for k in ROLES_BIN}
        sc = {k: c[f'n_{k}'] / c['N'] for k in ROLES_BIN}
        sp = {k: (o[f'n_{k}'] + c[f'n_{k}']) / (o['N'] + c['N']) for k in ROLES_BIN}
        ro = {k: o[rk] for k, rk in zip(ROLES_BIN, rate_keys)}
        rc = {k: c[rk] for k, rk in zip(ROLES_BIN, rate_keys)}
        Ro, Rc = sum(so[k] * ro[k] for k in ROLES_BIN), sum(sc[k] * rc[k] for k in ROLES_BIN)
        Rc_o = sum(so[k] * rc[k] for k in ROLES_BIN)
        Ro_p, Rc_p = sum(sp[k] * ro[k] for k in ROLES_BIN), sum(sp[k] * rc[k] for k in ROLES_BIN)
        kr = sum((so[k] + sc[k]) / 2 * (rc[k] - ro[k]) for k in ROLES_BIN)
        km = sum((sc[k] - so[k]) * (ro[k] + rc[k]) / 2 for k in ROLES_BIN)
        tot = Rc - Ro
        assert abs(kr + km - tot) < 1e-9
        vals = dict(tot=tot, rate_o=Rc_o - Ro, mix_o=Rc - Rc_o, rate_p=Rc_p - Ro_p, mix_p=tot - (Rc_p - Ro_p), kr=kr, km=km)
        for k, v in vals.items():
            agg[k].append(v)
        print(f'| {s} | {o[rep_key]:.1f} | {c[rep_key]:.1f} | {Ro:.2f} | {Rc:.2f} | {tot:+.2f} | {Rc_o:.2f} | {vals["rate_o"]:+.2f} '
              f'| {vals["mix_o"]:+.2f} | {Ro_p:.2f} | {Rc_p:.2f} | {vals["rate_p"]:+.2f} | {vals["mix_p"]:+.2f} | {kr:+.2f} | {km:+.2f} |')
        for k in ROLES_BIN:
            per_role.setdefault(k, []).append(dict(
                dr=rc[k] - ro[k], so=100 * so[k], sc=100 * sc[k], ds=100 * (sc[k] - so[k]),
                c_o=so[k] * (rc[k] - ro[k]), c_p=sp[k] * (rc[k] - ro[k]),
                kr=(so[k] + sc[k]) / 2 * (rc[k] - ro[k]), km=(sc[k] - so[k]) * (ro[k] + rc[k]) / 2))
    print('\n| 분해 항 | 시드별 (43·44·45) | 평균 | 95 % t-CI | >0 시드 |')
    print('|---|---|---|---|---|')
    names = dict(tot='전체 차', rate_o='rate 효과 (off 구성)', mix_o='mix 효과 (off 구성)', rate_p='rate 효과 (합동 구성)',
                 mix_p='mix 효과 (합동 구성, 나머지)', kr='Kitagawa rate', km='Kitagawa mix')
    for k, v in agg.items():
        m, lo, hi = ci(v)
        print(f'| {names[k]} | ' + ' · '.join(f'{x:+.2f}' for x in v) + f' | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {sum(x > 0 for x in v)}/3 |')
    print('\n역할별 기여 (pt, 전체 차에 대해). c_off = off 비중 × 비율차, c_pool = 합동 비중 × 비율차, '
          'K-rate = 평균 비중 × 비율차, K-mix = 비중차 × 평균 비율\n')
    print('| 역할 | 비율차 Δr (43·44·45) | off 비중 % | comm 비중 % | 비중차 pt | c_off | c_pool | K-rate | K-mix | K-rate 평균 | K-mix 평균 | Δr>0 시드 |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k in ROLES_BIN:
        pr = per_role[k]
        j = lambda key, nd=2: ' · '.join(f'{p[key]:+.{nd}f}' if key not in ('so', 'sc') else f'{p[key]:.{nd}f}' for p in pr)
        print(f'| {k} | {j("dr", 1)} | {j("so")} | {j("sc")} | {j("ds")} | {j("c_o")} | {j("c_p")} | {j("kr")} | {j("km")} '
              f'| {st.mean(p["kr"] for p in pr):+.2f} | {st.mean(p["km"] for p in pr):+.2f} | {sum(p["dr"] > 0 for p in pr)}/3 |')


decompose([f'bin_{k}' for k in ROLES_BIN], 'bin', '이진 준수율')
decompose([f'W_{w}' for w in ROLES_W], 'W', 'Woerner (같은 조우·같은 역할 n, 참고)')

# ── 3. Woerner 역할별·성분 ─────────────────────────────────────────
section('3. Woerner 전체·역할별·성분 (lrm) — 성분은 조우 평균(점수 = 성분 곱의 평균이라 성분 차가 점수 차로 더해지진 않음)')
print(HDR)
print(row('a2) Woerner 전체 %', 'W'))
for w in ROLES_W:
    print(row(f'Woerner {w} %', f'W_{w}'))
comps = {'giveway': ('S', 'D', 'A', 'R8', 'T', '선수횡단', '충돌'),
         'headon': ('S', 'D', 'A', 'R8', 'T', 'P', '선수횡단', '충돌'),
         'overtaking': ('S', 'D', 'A', 'R8', 'T', '선수횡단', '충돌'),
         'standon': ('S', 'C', 'C_raw', 'V', 'R8', 'T', '다른의무', '변침')}
lower = {'선수횡단', '충돌'}
nodir = {'다른의무', '변침'}
print('\n성분 (S·D·A·R8·T·P·C·C_raw·V = 0–100 점수, 높을수록 준수 · 선수횡단·충돌 = 비율 %, 낮을수록 좋음 · '
      '다른의무·변침 = 비율 %, 방향 없음). 성분 T 는 overtaking 점수 식에 안 들어감(S·D·A·W)\n')
print(HDR)
for w, cs in comps.items():
    for k in cs:
        print(row(f'{w} {k}', f'W_{w}_{k}', hb=None if k in nodir else (k not in lower)))

# ── 4. 조율 줄 ────────────────────────────────────────────────────
section('4. 조율 줄 (쌍결정 단위, 조우 단위 아님) — 레이더 안 ≤56 m vs 56–300 m')
print(HDR)
for zone, zl in (('in', '≤56m'), ('out', '56–300m')):
    print(row(f'{zl} 양보-유지 쌍결정 n', f'{zone}_gs_n', hb=None, nd=0, show_ci=False))
    print(row(f'{zl} 양보 우현 %', f'{zone}_gw_stbd'))
    print(row(f'{zl} 유지 유지 %', f'{zone}_so_keep'))
    print(row(f'{zl} 유지 좌현 %', f'{zone}_so_port', hb=False))
    print(row(f'{zl} 보완(양보 우현+유지 유지) %', f'{zone}_compl'))
    print(row(f'{zl} 둘 다 유지 %', f'{zone}_both_keep', hb=False))
    print(row(f'{zl} HeadOn 쌍결정 n', f'{zone}_ho_n', hb=None, nd=0, show_ci=False))
    print(row(f'{zl} HeadOn 둘 다 우현 %', f'{zone}_ho_both'))
    print(row(f'{zl} 역할 붙은 쌍 n', f'{zone}_role_n', hb=None, nd=0, show_ci=False))
    print(row(f'{zl} 역할 맞물림 %', f'{zone}_mesh'))

# ── 5. 탐색: 문턱 의존 ──────────────────────────────────────────────
section('5. 탐색 전용 — 판정 아님 (다른 문턱·다른 조우 집합). pair-heading·Rule8 = 쌍 조우(near_risk>0.3 진입, 70 m 이탈), '
        '이진·Woerner 의 300 m 조우와 다른 집합')
print(HDR)
print(row('[탐색] 쌍 조우 n GiveWay', 'ph_n_GiveWay', hb=None, nd=0, show_ci=False))
print(row('[탐색] 쌍 조우 n HeadOn', 'ph_n_HeadOn', hb=None, nd=0, show_ci=False))
print(row('[탐색] 쌍 조우 n StandOn', 'ph_n_StandOn', hb=None, nd=0, show_ci=False))
for t in (5, 10, 15, 20, 30):
    print(row(f'[탐색] pair-heading Δψ≥{t}° GiveWay %', f'ph{t}_GiveWay'))
for t in (5, 10, 15, 20, 30):
    print(row(f'[탐색] pair-heading Δψ≥{t}° HeadOn %', f'ph{t}_HeadOn'))
for t in (5, 10, 15, 20, 30):
    print(row(f'[탐색] pair-heading |Δψ|≤{t}° StandOn %', f'ph{t}_StandOn'))
for t in (5, 10, 20):
    print(row(f'[탐색] Rule8 침로or속력≥{t}° GiveWay %', f'r8{t}_GiveWay'))
    print(row(f'[탐색] Rule8 ≥{t}° GiveWay 속력만 %', f'r8{t}_GiveWay_spd', hb=None, show_ci=False))
for t in (5, 10, 20):
    print(row(f'[탐색] Rule8 침로or속력≥{t}° HeadOn %', f'r8{t}_HeadOn'))
print(row('[탐색] 양보역할 Δψ 중앙 ° (우현+)', 'gw_dpsi_med', hb=None))
print(row('[탐색] 유지역할 |Δψ| 중앙 °', 'so_dpsi_med', hb=None))
print(row('[탐색] 첫 변침까지 중앙 s', 'first_turn_s', hb=None))
print(row('[탐색] 무변침 조우 %', 'no_turn', hb=None))
for lab in ('0-10m', '10-20m', '20-35m', '35-m'):
    print(row(f'[탐색] GiveWay Rule8@10° 진입DCPA {lab} %', f'strat_{lab}'))
    print(row(f'[탐색] GiveWay 진입DCPA {lab} n', f'strat_n_{lab}', hb=None, nd=0, show_ci=False))
print('\n[탐색] 문턱별 comm−off 평균차 pt (시드별 43·44·45). GiveWay/HeadOn = 쌍 조우 중 목표 방위 대비 최대 우현 오프셋 ≥ 문턱, '
      'StandOn = |Δψ| ≤ 문턱 (eval_ckpt.py 옛 클론 ab0cea5 기준 정의)\n')
print('| 줄 | ' + ' | '.join(['5°', '10°', '15°', '20°', '30°']) + ' |')
print('|---|---|---|---|---|---|')
for role in ('GiveWay', 'HeadOn', 'StandOn'):
    cells = []
    for t in (5, 10, 15, 20, 30):
        d = [D[(s, 'comm')][f'ph{t}_{role}'] - D[(s, 'off')][f'ph{t}_{role}'] for s in S3]
        cells.append(f'{st.mean(d):+.1f} ({" · ".join(f"{x:+.1f}" for x in d)})')
    print(f'| pair-heading {role} | ' + ' | '.join(cells) + ' |')
for role in ('GiveWay', 'HeadOn'):
    cells = []
    for t in (5, 10, 15, 20, 30):
        if f'r8{t}_{role}' not in D[(43, 'off')]:
            cells.append('–')
            continue
        d = [D[(s, 'comm')][f'r8{t}_{role}'] - D[(s, 'off')][f'r8{t}_{role}'] for s in S3]
        cells.append(f'{st.mean(d):+.1f} ({" · ".join(f"{x:+.1f}" for x in d)})')
    print(f'| Rule8 {role} | ' + ' | '.join(cells) + ' |')

# ── 6. 역할 약속·옛 COLREGs ──────────────────────────────────────────
section('6. 역할 약속 판정기·옛 COLREGs (lrm)')
print(HDR)
print(row('role-promise/v2 roleKeptSafe %', 'rp2_roleKeptSafe'))
print(row('role-promise/v2 both_comply %', 'rp2_both_comply'))
print(row('role-promise/v2 safe %', 'rp2_safe'))
print(row('role-promise/v2 judged n', 'rp2_judged', hb=None, nd=0, show_ci=False))
print(row('role-promise/v2 pair_coll n', 'rp2_pair_coll', hb=False, nd=0, show_ci=False))
print(row('role-promise (v1) roleKeptSafe %', 'rp1_roleKeptSafe'))
print(row('role-promise (v1) safe %', 'rp1_safe'))
print(row('colregsOK % (결정 단위, 메인 줄)', 'colregsOK'))
print(row('colregs (연속, 메인 줄)', 'colregs', nd=4))
print(row('옛 COLREGs = encounter-COLREGs/실제타각 C % (_fig1_mean "옛 COLREGs")', 'oldC'))

# ── 7. lrmb vs lrm 일치 ──────────────────────────────────────────────
section('7. 평가 결정성 — lrmb vs lrm (진행 줄·torch 경고·[binary-COLREGs]·logstd_min 경고 줄만 빼고 줄 단위 비교)')
for s in S3:
    for a in ARMS:
        L, B = read('lrm', s, a), read('lrmb', s, a)
        core = lambda X: [l for l in X if not NOISE.search(l) and '[binary-COLREGs]' not in l and 'logstd_min' not in l]
        cl, cb = core(L), core(B)
        ntok = sum(len(re.findall(NUM, l)) for l in cl)
        extra_b = [l.strip() for l in B if 'logstd_min' in l]
        extra_l = [l.strip() for l in L if 'logstd_min' in l]
        if cl == cb:
            print(f'- s{s} {a}: **동일** ({len(cl)}줄, 숫자 토큰 {ntok}개) · 뺀 줄: lrmb 에만 logstd_min 경고 {len(extra_b)}줄'
                  f'(lrm {len(extra_l)}줄) + 이진 줄')
        else:
            print(f'- s{s} {a}: **다름**')
            for dl in difflib.unified_diff(cl, cb, 'lrm', 'lrmb', lineterm='', n=0):
                print('    ' + dl)

# ── 8. 사전등록 표 재계산 ────────────────────────────────────────────
section('8. goals_3seed_24M.md 재계산 (1자리 반올림 값 그대로, t = 4.303)')
print(HDR)
print(row('a) COLREGs 이진 준수율 %', 'bin'))
print(row('a2) COLREGs Woerner %', 'W'))
print(row('b) DCPA(통과거리, pair-detail 중앙) m', 'dcpa'))
print(row('c) 평균 타각 크기 ° (goal-ep meanAbsRudder)', 'rud_goal', hb=False, nd=2))
print(row('도착 %', 'goal'))
print(row('배끼리 충돌 %', 'vColl', hb=False))
