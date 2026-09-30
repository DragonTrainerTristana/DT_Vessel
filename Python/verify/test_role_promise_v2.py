"""test_role_promise_v2.py — 역할 약속 판정기 v2(ROLE_JUDGE=v2, 2026-09-30, 스펙 2026-09-30-reward-v3-decode-sweep-design.md §3).

  python verify/test_role_promise_v2.py        # ALL PASS 여야 함 (imo·none 고정, CPU)

rp14 정면, 0 번이 좌현 → F1: 그 결정에 0 번만 −20(1회), 1 번 0. 벌점 결정 = dpsi 가 처음 −5° 를 넘는 결정. 쌍둥이(PEN 1e-9) 보상 차 = PEN
rp15 교차, 유지선이 hold 창(tcpa ≤ 48.8 s) 안에서 >10° 돌면 → F2: 유지선만 1회. 창 밖(이른) 선회는 위반 아님
rp16 정면 직진 → F4: 거리 < 24 m 가 되는 결정에 둘 다 −20(1회). 뒤이은 충돌은 추가 벌점 없음(쌍당 1회), 판정 = 실패·coll
rp17 정면 둘 다 우현 → dcpa ≥ 30 m 5결정 연속이면 해소: 판정 1·성공 1·resolved 1·벌점 0, CPA 통과 전에 종료
rp18 늦은 시작(시작 tcpa < 28 s): 침로 기준 면제 — 둘 다 좌현이어도 벌점 0·성공. 같은 기동을 멀리서 시작하면 F1 둘 다
rp19 실패 뒤 재시작·추가 벌점 없음: 좌현 위반 뒤 300결정 굴려도 벌점 총합 1, 시작 1
rp20 16척 규칙 정책 롤아웃: 판정기 v2 이벤트·결정당 벌점 == 느린 참조 구현(전 결정) + 공허 통과 방지 하한
rp21 같은 롤아웃에서 legacy(end) 판정기 == 독립 end 판정기(병기 경로가 옛 숫자를 바꾸지 않음) · ROLE_JUDGE=end 는 옛 클래스
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
sys.path.insert(0, PYROOT)
sys.path.insert(0, HERE)
for _k in [k for k in os.environ if k.startswith('VESSEL_')]:
    os.environ.pop(_k)
os.environ.update({'VESSEL_DYN_PROFILE': 'imo', 'VESSEL_OBSTACLES': 'none'})

import torch  # noqa: E402
import config as cfg  # noqa: E402
import vessel_gym as vg  # noqa: E402
from test_role_promise import make_env, place, act, HEADON, CROSS, _wrap, PEN  # noqa: E402

RES = []


def check(name, ok, info=''):
    RES.append(bool(ok))
    print(f"  {'PASS' if ok else '★FAIL'}  {name}  {info}")


class _v2:
    """블록 안에서 ROLE_JUDGE=v2 · PEN 켬. 끝나면 원상복구."""
    def __init__(self, pen=PEN):
        self.pen = pen

    def __enter__(self):
        self.saved = (vg.ROLE_JUDGE, vg.ROLE_PROMISE_PEN)
        vg.ROLE_JUDGE, vg.ROLE_PROMISE_PEN = 'v2', self.pen
        return self

    def __exit__(self, *a):
        vg.ROLE_JUDGE, vg.ROLE_PROMISE_PEN = self.saved


def run_v2(ships, policy, steps, N=None, pen=PEN):
    """policy(t, env) -> rud list. 결정마다 (r, d, oc, charges[1,N], events, heading, dist01) 기록."""
    with _v2(pen):
        env = make_env(1, N or len(ships))
        place(env, ships)
        rec = []
        for t in range(steps):
            hdg = env.heading[0].clone()
            a = act(env, policy(t, env))
            _, r, d, oc = env.step(a)
            ev = env._rp.events
            chg = (ev['chg_i'].to(env.dtype).sum(2) + ev['chg_j'].to(env.dtype).sum(1)
                   + ev['crash_i'].to(env.dtype).sum(2) + ev['crash_j'].to(env.dtype).sum(1))
            rec.append(dict(r=r.clone(), d=d.clone(), oc=oc.clone(), chg=chg.clone(), hdg=hdg,
                            dist01=float(env._last_pw['dist'][0, 0, 1]),
                            ev={k: v.clone() for k, v in ev.items()}))
        tot = {k: sum(int(r['ev'][k].sum()) for r in rec) for k in ('judged', 'success', 'fail', 'discard', 'coll', 'start', 'resolved')}
        nf = sum(r['chg'] for r in rec)
        return env, tot, nf, rec


def turn_then_hold(rud0, rud1, t_on):
    return lambda t, env: [rud0 if t < t_on else 0.0, rud1 if t < t_on else 0.0]


def rp14_port_violator_only():
    env, tot, nf, rec = run_v2(HEADON, turn_then_hold(-0.5, 0.5, 30), 260)
    k = next((t for t, r in enumerate(rec) if float(r['chg'][0, 0]) > 0), None)
    # 위반 결정 = 0 번 침로가 시작 침로(0°)에서 −5° 를 처음 넘는 결정(판정기는 step 뒤 heading 으로 봄 → rec[t+1].hdg)
    h0 = float(rec[0]['hdg'][0])
    first = next((t for t in range(len(rec) - 1) if float(_wrap(torch.tensor(float(rec[t + 1]['hdg'][0]) - h0))) < -cfg.ROLE_PORT_TOL_DEG), None)
    ok = nf.tolist() == [[1.0, 0.0]] and k is not None and k == first and tot['judged'] == 1 and tot['fail'] == 1
    ft = int(rec[k]['ev']['fail_type'].max()) if k is not None else -1
    check('rp14 정면 0 번 좌현 → F1 0 번만 1회, 위반 결정에', ok, f"charges={nf.tolist()} 벌점결정={k} 첫 −5° 결정={first} {tot}")
    # 판정은 조우 끝(CPA 3 뒤)에 실패로 잡힘, fail_type 1(좌현)
    kj = next((t for t, r in enumerate(rec) if int(r['ev']['judged'].sum()) > 0), None)
    check('rp14 판정=실패·fail_type=1·벌점 결정 < 판정 결정', kj is not None and k is not None and k < kj
          and int(rec[kj]['ev']['fail_type'][0, 0, 1]) == 1, f"판정결정={kj} fail_type={int(rec[kj]['ev']['fail_type'][0, 0, 1]) if kj is not None else None}")
    # 쌍둥이 보상 차 = PEN·charges (rp4 방식)
    pol = turn_then_hold(-0.5, 0.5, 30)
    with _v2(PEN):
        envP, envT = make_env(1, 2), make_env(1, 2)
        place(envP, HEADON); place(envT, HEADON)
        err, tot_c = 0.0, torch.zeros(1, 2)
        for t in range(260):
            a = act(envP, pol(t, envP))
            vg.ROLE_PROMISE_PEN = PEN
            _, rP, _, _ = envP.step(a)
            vg.ROLE_PROMISE_PEN = 1e-9
            _, rT, _, _ = envT.step(a)
            ev = envP._rp.events
            c = ev['chg_i'].to(envP.dtype).sum(2) + ev['chg_j'].to(envP.dtype).sum(1)
            tot_c += c
            err = max(err, float(((rT - rP) - (PEN - 1e-9) * c).abs().max()))
    check('rp14 쌍둥이 env 보상 차 = PEN·벌점(그 결정)', tot_c.tolist() == [[1.0, 0.0]] and err < 1e-3, f"{tot_c.tolist()} max|오차|={err:.1e}")


def rp15_standon_hold_window():
    # 교차: 0 양보(우현 45결정 → 충분히 돎), 1 유지. 유지선이 창 안(tcpa ≤ 48.8 s)에서 0.6 으로 돌면 F2 유지선만.
    # 양보선(0)은 직진(해소 안 됨), 유지선(1)만 창 안에서 0.6 → F2 유지선만. (양보선이 일찍 충분히 돌면 쌍이 해소돼 F2 기회가 없음 — rp15b)
    st = {'open': None}
    def pol_in_window(t, env):
        tc = float(env._last_pw['tcpa'][0, 0, 1])
        if st['open'] is None and tc <= vg.EARLY_ACTION_TIME and tc > vg.RULE_17B_TIME:
            st['open'] = t
        turn = st['open'] is not None and t < st['open'] + 45          # 창 열린 뒤 45결정 0.6(타속 3°/s·imo 선회 느림 → ~15°, >10°) 뒤 직진
        return [0.0, 0.6 if turn else 0.0]
    env, tot, nf, rec = run_v2(CROSS, pol_in_window, 800)
    kk = [t for t, r in enumerate(rec) if float(r['chg'][0, 1]) > 0]
    ok = nf.tolist() == [[0.0, 1.0]] and len(kk) == 1 and int(rec[kk[0]]['ev']['fail_type'][0, 0, 1]) in (0, 2) if kk else False
    # fail_type 는 판정 결정에만 채움 → 판정 결정에서 2 인지
    kj = next((t for t, r in enumerate(rec) if int(r['ev']['judged'].sum()) > 0), None)
    ft = int(rec[kj]['ev']['fail_type'][0, 0, 1]) if kj is not None else -1
    check('rp15a 유지선이 hold 창 안에서 >10° → F2 유지선만 1회 · fail_type 2', nf.tolist() == [[0.0, 1.0]] and len(kk) == 1 and ft == 2,
          f"charges={nf.tolist()} 벌점결정={kk} 판정={kj} fail_type={ft} {tot}")
    # 창 밖(이른) 선회 = 위반 아님: rp5c 와 같은 기동(둘 다 0.6, 45결정)은 v2 에서 유지선 벌점 없음
    env2, tot2, nf2, rec2 = run_v2(CROSS, turn_then_hold(0.6, 0.6, 45), 320)
    check('rp15b 창 열리기 전 유지선 선회 → v2 는 위반 아님(벌점 0 또는 해소)', float(nf2[0, 1]) == 0.0, f"charges={nf2.tolist()} {tot2}")


def rp16_distance_both_once():
    env, tot, nf, rec = run_v2(HEADON, lambda t, e: [0.0, 0.0], 240)
    k = next((t for t, r in enumerate(rec) if float(r['chg'].sum()) > 0), None)
    first = next((t for t, r in enumerate(rec) if r['dist01'] < cfg.ROLE_SAFE_DIST), None)
    oc = [int(r['oc'][0, 0]) for r in rec]
    hit = vg.OUT_COLLISION_VESSEL in oc
    check('rp16 정면 직진 → F4 거리<24 결정에 둘 다 1회, 이후 충돌은 추가 벌점 없음', nf.tolist() == [[1.0, 1.0]] and k == first and hit
          and tot['coll'] == 1 and tot['fail'] == 1, f"charges={nf.tolist()} 벌점결정={k} 첫<24m={first} 충돌={hit} {tot}")
    kj = next((t for t, r in enumerate(rec) if int(r['ev']['judged'].sum()) > 0), None)
    check('rp16 fail_type=4(거리)', kj is not None and int(rec[kj]['ev']['fail_type'][0, 0, 1]) == 4)


def rp17_resolve():
    env, tot, nf, rec = run_v2(HEADON, turn_then_hold(0.5, 0.5, 30), 260)
    kend = next((t for t, r in enumerate(rec) if int(r['ev']['judged'].sum()) > 0), None)
    # 옛 판정기(end)는 CPA 통과 3결정 뒤에 끝남 — v2 해소는 그보다 앞
    with _v2(0.0):
        pass
    vg.ROLE_PROMISE_PEN = PEN
    envE = make_env(1, 2); place(envE, HEADON)
    kE = None
    for t in range(260):
        envE.step(act(envE, turn_then_hold(0.5, 0.5, 30)(t, envE)))
        if int(envE._rp.events['judged'].sum()) > 0 and kE is None:
            kE = t
    vg.ROLE_PROMISE_PEN = 0.0
    check('rp17 둘 다 우현 → 해소 성공(resolved 1·벌점 0), 옛 판정보다 이른 종료',
          tot['judged'] == 1 and tot['success'] == 1 and tot['resolved'] == 1 and float(nf.sum()) == 0.0
          and kend is not None and kE is not None and kend < kE, f"{tot} v2 종료={kend} end 종료={kE}")


def rp18_late_start():
    # 70 m·3 m/s 접근 → tcpa ≈ 23 s < 28(늦은 시작). 횡 9 m(방위 7° < 10° → 정면 유효, dcpa 9 < 24 → 시작).
    #   둘 다 좌현 전타(서로 반대쪽으로 벌어짐) → ≥ 24 m 안전 통과 → 침로 기준 면제라 벌점 0·성공
    near = [dict(HEADON[0], pos=(0.0, -35.0)), dict(HEADON[1], pos=(9.0, 35.0))]
    env, tot, nf, rec = run_v2(near, turn_then_hold(-1.0, -1.0, 60), 220)
    late = any(bool(r['ev']['late'].any()) for r in rec)
    check('rp18a 늦은 시작: 둘 다 좌현이어도 침로 벌점 0', float(nf.sum()) == 0.0 and tot['judged'] >= 1 and late,
          f"charges={nf.tolist()} late={late} {tot}")
    env2, tot2, nf2, rec2 = run_v2(HEADON, turn_then_hold(-1.0, -1.0, 60), 300)
    check('rp18b 같은 기동, 멀리서 시작(tcpa ≥ 28) → F1 둘 다 1회', nf2.tolist() == [[1.0, 1.0]], f"charges={nf2.tolist()} {tot2}")


def rp19_no_restart():
    env, tot, nf, rec = run_v2(HEADON, turn_then_hold(-0.5, 0.5, 30), 400)
    check('rp19 실패 뒤 재시작·추가 벌점 없음(400결정)', float(nf.sum()) == 1.0 and tot['start'] == 1 and tot['judged'] == 1,
          f"charges={nf.tolist()} {tot}")


class _RefV2:
    """느린 참조 구현 — 쌍별 파이썬 루프로 스펙 §3 v2 규칙을 그대로 옮김(판정기 텐서 코드와 독립)."""

    def __init__(self):
        self.P = {}
        self.t = 0

    def step(self, hdg, dist, raw, tcpa, dcpa, oc, role_raw, risk):
        R = float(cfg.COMM_RANGE)
        E, N = len(hdg), len(hdg[0])
        out, chg = [], [[0.0] * N for _ in range(E)]
        prim = {}
        for e in range(E):
            for i in range(N):
                mx = max(risk[e][i])
                prim[(e, i)] = risk[e][i].index(mx) if mx > 0 else None
        T32 = lambda x: torch.tensor(x, dtype=torch.float32)
        w = lambda x: float(vg._wrap180(T32(x)))
        for key in sorted(self.P):
            e, i, j = key
            p = self.P[key]
            pi, pj = prim[(e, i)] == j, prim[(e, j)] == i
            p['wpi'], p['wpj'] = p['wpi'] or pi, p['wpj'] or pj
            dpi, dpj = w(hdg[e][i] - p['h0i']), w(hdg[e][j] - p['h0j'])
            if pi:
                p['dmaxi'], p['dmini'] = max(p['dmaxi'], dpi), min(p['dmini'], dpi)
            if pj:
                p['dmaxj'], p['dminj'] = max(p['dmaxj'], dpj), min(p['dminj'], dpj)
            tc, dc, d = tcpa[e][i][j], dcpa[e][i][j], dist[e][i][j]
            win = tc <= vg.EARLY_ACTION_TIME and tc > vg.RULE_17B_TIME
            if win and not p['holdi']:
                p['hhi'], p['holdi'] = hdg[e][i], True
            if win and not p['holdj']:
                p['hhj'], p['holdj'] = hdg[e][j], True
            dhi, dhj = abs(w(hdg[e][i] - p['hhi'])), abs(w(hdg[e][j] - p['hhj']))
            if win and pi:
                p['soi'] = max(p['soi'], dhi)
            if win and pj:
                p['soj'] = max(p['soj'], dhj)
            p['mind'] = min(p['mind'], d)
            p['n'] += 1
            p['cpa'] = p['cpa'] + 1 if raw[e][i][j] < 0 else 0
            p['far'] = p['far'] + 1 if d > R else 0
            p['res'] = p['res'] + 1 if (tc > vg.RULE_17B_TIME and dc >= cfg.ROLE_V2_RESOLVE_DCPA_M) else 0
            di, dj = oc[e][i] != 0, oc[e][j] != 0
            pc = oc[e][i] == vg.OUT_COLLISION_VESSEL and oc[e][j] == vg.OUT_COLLISION_VESSEL and d <= vg._PAIR_COLL_DIST
            live = not p['failed']
            course = live and not p['late']
            gi, gj = p['ri'] in (1, 3), p['rj'] in (1, 3)
            F1i = course and gi and pi and dpi < -cfg.ROLE_PORT_TOL_DEG
            F1j = course and gj and pj and dpj < -cfg.ROLE_PORT_TOL_DEG
            F2i = course and p['ri'] == 2 and pi and win and dhi > cfg.ROLE_STANDON_MAX_DEG
            F2j = course and p['rj'] == 2 and pj and win and dhj > cfg.ROLE_STANDON_MAX_DEG
            F4 = live and d < cfg.ROLE_SAFE_DIST
            F5 = live and pc
            fail_now = F1i or F1j or F2i or F2j or F4 or F5
            end_disc = (di or dj) and not pc
            end_norm = (not (di or dj)) and (p['cpa'] >= cfg.ROLE_END_CPA or p['far'] >= cfg.ROLE_END_FAR)
            end_res = live and not fail_now and not (di or dj) and not end_norm and p['res'] >= cfg.ROLE_V2_RESOLVE_N
            endj = end_norm and p['n'] >= cfg.ROLE_MIN_STEPS
            F6i = endj and course and not fail_now and gi and p['wpi'] and p['dmaxi'] < cfg.ROLE_GIVEWAY_MIN_DEG
            F6j = endj and course and not fail_now and gj and p['wpj'] and p['dmaxj'] < cfg.ROLE_GIVEWAY_MIN_DEG
            chgi = F1i or F2i or F4 or F5 or F6i
            chgj = F1j or F2j or F4 or F5 or F6j
            chg[e][i] += float(chgi); chg[e][j] += float(chgj)
            crash_i = end_disc and di and oc[e][i] != vg.OUT_GOAL and not p['failed']
            crash_j = end_disc and dj and oc[e][j] != vg.OUT_GOAL and not p['failed']
            chg[e][i] += float(crash_i); chg[e][j] += float(crash_j)
            newfail = (fail_now or F6i or F6j) and not p['failed']
            failed_new = p['failed'] or newfail
            end_any = pc or end_disc or end_norm or end_res
            judged = endj or pc or end_res or (end_any and failed_new)
            if end_any:
                kind = 'success' if (judged and not failed_new) else ('fail' if judged else 'discard')
                out.append((e, i, j, kind, p['t0'], bool(end_res)))
                del self.P[key]
            else:
                p['failed'] = failed_new
        for e in range(E):
            for i in range(N):
                for j in range(i + 1, N):
                    if (e, i, j) in self.P or oc[e][i] != 0 or oc[e][j] != 0:
                        continue
                    if not (dist[e][i][j] <= R and raw[e][i][j] >= 0 and dcpa[e][i][j] < vg.DCPA_RISK):
                        continue
                    ri, rj = role_raw[e][i][j], role_raw[e][j][i]
                    if ri == 0 and rj in (3, 4):
                        ri = 2
                    if rj == 0 and role_raw[e][i][j] in (3, 4):
                        rj = 2
                    if ri > 0 and rj > 0:
                        self.P[(e, i, j)] = dict(ri=ri, rj=rj, h0i=hdg[e][i], h0j=hdg[e][j], dmaxi=0.0, dmini=0.0,
                                                 dmaxj=0.0, dminj=0.0, soi=0.0, soj=0.0, mind=dist[e][i][j], n=1,
                                                 cpa=0, far=0, res=0, t0=self.t, failed=False,
                                                 late=tcpa[e][i][j] < cfg.ROLE_V2_LATE_START_TCPA_S,
                                                 holdi=False, holdj=False, hhi=0.0, hhj=0.0, wpi=False, wpj=False)
        self.t += 1
        return out, chg


def rp20_reference():
    E, N, T = 3, 16, 1500
    cap = {}
    orig = vg.RolePromiseTrackerV2.update

    def spy(self, env_, pw, outcome):
        to = env_.pos[:, None, :, :] - env_.pos[:, :, None, :]
        h = env_.heading * vg.DEG
        fx, fz = torch.sin(h)[:, :, None], torch.cos(h)[:, :, None]
        gx, gz = torch.sin(h)[:, None, :], torch.cos(h)[:, None, :]
        b = torch.atan2(fz * to[..., 0] - fx * to[..., 1], fx * to[..., 0] + fz * to[..., 1]) / vg.DEG
        ob = torch.atan2(gx * to[..., 1] - gz * to[..., 0], -(gx * to[..., 0] + gz * to[..., 1])) / vg.DEG
        fast = env_.speed[:, :, None] > env_.speed[:, None, :] * 1.1
        cap.update(hdg=env_.heading.tolist(), dist=pw['dist'].tolist(), raw=pw['raw_tcpa'].tolist(),
                   tcpa=pw['tcpa'].tolist(), dcpa=pw['dcpa'].tolist(), oc=outcome.tolist(), risk=pw['risk'].tolist(),
                   role=vg.encounter_role(b, ob, pw['dist'], pw['raw_tcpa'] >= 0, fast, float(cfg.COMM_RANGE)).tolist())
        ret = orig(self, env_, pw, outcome)
        cap['ret'] = ret.tolist()
        return ret
    vg.RolePromiseTrackerV2.update = spy
    ref = _RefV2()
    mism, n = [], {'success': 0, 'fail': 0, 'discard': 0, 'resolved': 0}
    chg_mism, tot_chg = 0, 0.0
    try:
        with _v2(PEN):
            env = make_env(E, N, 21)
            for t in range(T):
                pw0 = env._last_pw
                j = env.danger_idx
                tc = pw0['tcpa'].gather(-1, j.unsqueeze(-1)).squeeze(-1)
                tg = env.goal - env.pos
                err = _wrap(torch.atan2(tg[..., 0], tg[..., 1]) / vg.DEG - env.heading)
                a0 = (err / 20.0).clamp(-1, 1)
                sit = env.situation
                give = (sit == 1) | (sit == 3) | (sit == 4)
                a0 = torch.where(give, torch.full_like(a0, 0.6), a0)
                so = sit == 2
                a0 = torch.where(so & (tc > vg.RULE_17B_TIME), torch.zeros_like(a0), a0)
                a0 = torch.where(so & (tc <= vg.RULE_17B_TIME), torch.full_like(a0, 0.6), a0)
                env.step(torch.stack([a0, torch.ones_like(a0)], dim=-1))
                ev = env._rp.events
                got = set()
                for kind, m in (('success', ev['success']), ('fail', ev['fail']), ('discard', ev['discard'])):
                    for e_, i_, j_ in m.nonzero().tolist():
                        got.add((e_, i_, j_, kind, int(ev['t_start'][e_, i_, j_]), bool(ev['resolved'][e_, i_, j_])))
                want_l, want_chg = ref.step(cap['hdg'], cap['dist'], cap['raw'], cap['tcpa'], cap['dcpa'], cap['oc'],
                                            cap['role'], cap['risk'])
                want = set(want_l)
                if got != want:
                    mism.append((t, sorted(got - want)[:3], sorted(want - got)[:3]))
                if any(abs(a - b) > 1e-6 for ra, rb in zip(cap['ret'], want_chg) for a, b in zip(ra, rb)):
                    chg_mism += 1
                tot_chg += sum(sum(r) for r in want_chg)
                for x in want:
                    n[x[3]] += 1
                    n['resolved'] += int(x[5])
    finally:
        vg.RolePromiseTrackerV2.update = orig
    judged = n['success'] + n['fail']
    nonvac = judged >= 50 and n['success'] >= 10 and n['discard'] >= 5 and n['resolved'] >= 1 and tot_chg >= 20
    check('rp20 16척 규칙 정책: v2 이벤트·결정당 벌점 == 참조 구현(전 결정) + 공허 통과 아님',
          not mism and chg_mism == 0 and nonvac, f"{n} 벌점합={tot_chg:.0f} 불일치 결정 {len(mism)} 벌점 불일치 {chg_mism} {mism[:2]}")


def rp21_legacy_side_by_side():
    # 같은 롤아웃: (a) ROLE_JUDGE=v2 + legacy 병기 (b) ROLE_JUDGE=end 단독 — (a).legacy 이벤트 == (b) 이벤트, (b) 클래스 = 옛 판정기
    E, N, T = 2, 8, 400
    def roll(judge):
        saved = (vg.ROLE_JUDGE, vg.ROLE_PROMISE_PEN)
        vg.ROLE_JUDGE, vg.ROLE_PROMISE_PEN = judge, 0.0
        try:
            env = make_env(E, N, 7)
            tr = env.enable_role_tracker()
            lg = env.enable_legacy_role_tracker()
            out = []
            g = torch.Generator().manual_seed(3)
            for t in range(T):
                a0 = (torch.rand(E, N, generator=g) - 0.5) * 0.8
                env.step(torch.stack([a0, torch.ones_like(a0)], -1))
                src = lg if lg is not None else tr
                ev = src.events
                out.append(tuple(int(ev[k].sum()) for k in ('judged', 'success', 'fail', 'discard')))
            return type(tr), lg, out
        finally:
            vg.ROLE_JUDGE, vg.ROLE_PROMISE_PEN = saved
    ta, lga, oa = roll('v2')
    tb, lgb, ob = roll('end')
    check('rp21 v2+legacy 병기 == end 단독 (이벤트 열 전부) · 클래스', oa == ob and ta is vg.RolePromiseTrackerV2
          and type(lga) is vg.RolePromiseTracker and lgb is None and tb is vg.RolePromiseTracker,
          f"같음={oa == ob} 판정합 v2병기={sum(o[0] for o in oa)} end={sum(o[0] for o in ob)}")


if __name__ == '__main__':
    torch.set_num_threads(1)
    print('=' * 78); print('역할 약속 판정기 v2 단위 테스트 (imo·none, CPU)'); print('=' * 78)
    for fn in (rp14_port_violator_only, rp15_standon_hold_window, rp16_distance_both_once, rp17_resolve, rp18_late_start,
               rp19_no_restart, rp20_reference, rp21_legacy_side_by_side):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            check(f"{fn.__name__} 예외", False, f"{type(e).__name__}: {e}")
    print('=' * 78)
    print(f"VERDICT: {'ALL PASS' if all(RES) else 'FAIL ' + str(RES.count(False))}")
    sys.exit(0 if all(RES) else 1)
