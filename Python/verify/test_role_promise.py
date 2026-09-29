"""test_role_promise.py — 역할 약속 보상(ROLE_PROMISE_PEN, 2026-09-29) 단위 테스트. G2 (스펙 2026-09-29-role-promise-design.md §5).

  python verify/test_role_promise.py        # ALL PASS 여야 함 (imo·none 을 여기서 고정)

rp1  토글을 켜도 같은 행동이면 상태 궤적이 같음(보상만 다름) · 끄면 판정기 자체가 없음
rp2  켜면 결정당 COLREGs 는 가산만 빠짐: r_off − (r_on + PEN·실패수) ≥ 0, 실제로 빠지는 결정이 있음
rp2b 유지선 침로유지 + 중간 속력(옛 합 +0.5−0.5=0) 결정에서 새 보상이 −0.5 항을 그대로 받음(합계 clamp 버그 검출)
rp3  정면: 둘 다 우현 ≥10° + ≥24 m 통과 → 판정 1·성공 1·벌점 0
rp4  정면: 한 배가 좌현 → 두 배 모두 실패 1회, 그 결정 보상 차 = PEN (PEN 1e-9 쌍둥이 env 와 비교)
rp5  교차: 양보선 우현 + 유지선 유지 → 성공 / 유지선이 17(b) 전에 크게 돌면 → 실패
rp6  추월: 추월선 = 4, 추월당하는 배 = 보완 규칙으로 유지(2)
rp7  두 배끼리 충돌 → 길이 무관 판정·실패, −300 도 그대로
rp8  도착으로 끝나면 판정 없이 폐기(벌점 0) · 시간초과·제3선 충돌·벽으로 끝나면 폐기하되 *그 배만* 조우당 실패 1(09-29c 벽 빈틈 수정)
rp9  respawn 한 배는 행·열 모두 비활성(이전 조우를 물려받지 않음)
rp10 CPA 판정 debounce(3 연속) — 흔들려도 1회만 판정
rp11 16척 규칙 정책 롤아웃에서 판정기 이벤트 전체 == 느린 파이썬 참조 구현(쌍별 루프) + 공허 통과 방지 하한
rp12 N=1·N=2 env 에서도 동작 · 판정 상수 = config
rp13 벽 충돌: 조우 중 벽으로 가면 그 배만 조우당 실패 1(벽 탈출 빈틈 막힘)
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
sys.path.insert(0, PYROOT)
for _k in [k for k in os.environ if k.startswith('VESSEL_')]:
    os.environ.pop(_k)
os.environ.update({'VESSEL_DYN_PROFILE': 'imo', 'VESSEL_OBSTACLES': 'none'})

import torch  # noqa: E402
import config as cfg  # noqa: E402
import vessel_gym as vg  # noqa: E402

PEN = 20.0
RES = []


def check(name, ok, info=''):
    RES.append(bool(ok))
    print(f"  {'PASS' if ok else '★FAIL'}  {name}  {info}")


def make_env(E=1, N=2, seed=0):
    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device='cpu', seed=seed, ring_scale=1.0, crossing=0,
                            risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE,
                            farfield_coef=0.0, perpair_coef=-0.15, perpair_exp=3.0)
    env.reset()
    return env


def place(env, ships):
    """env 0 의 배 상태를 직접 넣는다. ships = [dict(pos, hdg, spd, maxs, goal, steps=0)]."""
    for k, s in enumerate(ships):
        env.pos[0, k] = torch.tensor(s['pos'], dtype=env.dtype)
        env.heading[0, k] = s['hdg']
        env.speed[0, k] = s['spd']
        env.max_speed[0, k] = s['maxs']
        env.target_speed[0, k] = s['spd']
        env.rudder[0, k] = 0.0
        env.cmd_rudder[0, k] = 0.0
        env.goal[0, k] = torch.tensor(s['goal'], dtype=env.dtype)
        env.step_count[0, k] = s.get('steps', 0)
    env.prev_dist = torch.linalg.norm(env.goal - env.pos, dim=-1)
    env.prev_dcpa.fill_(-1.0)
    env.prev_danger_idx.fill_(-1)
    env.prev_rudder.zero_()
    env.prev_far_risk.fill_(-1.0)
    env._update_situation()
    if env._rp is not None:
        env._rp.reset()


def act(env, rud, spd=None):
    """rud [N] ∈[-1,1] 명령 타각, spd [N] 목표 속력(m/s, None = 현재 max 의 상태 유지용 현재 속력)."""
    a0 = torch.tensor(rud, dtype=env.dtype).view(1, -1)
    s = env.speed[0:1] if spd is None else torch.tensor(spd, dtype=env.dtype).view(1, -1)
    a1 = (2.0 * s / env.max_speed[0:1] - 1.0).clamp(-1, 1)
    return torch.stack([a0, a1], dim=-1)


def fails_per_ship(env):
    ev = env._rp.events
    f = ev['fail'].to(env.dtype)
    ci, cj = ev['crash_i'].to(env.dtype), ev['crash_j'].to(env.dtype)
    return f.sum(dim=2) + f.sum(dim=1) + ci.sum(dim=2) + cj.sum(dim=1)


def run_scenario(ships, policy, steps, pen=PEN, N=None):
    """policy(t, env) -> rud list. Returns dict of totals and per-step records."""
    vg.ROLE_PROMISE_PEN = pen
    env = make_env(1, N or len(ships))
    place(env, ships)
    tot = {'judged': 0, 'success': 0, 'fail': 0, 'discard': 0, 'coll': 0, 'start': 0}
    nf = torch.zeros(1, env.N)
    rec = []
    for t in range(steps):
        a = act(env, policy(t, env))
        _, r, d, oc = env.step(a)
        ev = env._rp.events
        for k in tot:
            tot[k] += int(ev[k].sum())
        nf += fails_per_ship(env)
        rec.append(dict(r=r.clone(), d=d.clone(), oc=oc.clone(), active=env._rp.active.clone(),
                        role_i=env._rp.role_i.clone(), role_j=env._rp.role_j.clone(), t_start=env._rp.t_start.clone(),
                        ev={k: v.clone() for k, v in ev.items()}))
    vg.ROLE_PROMISE_PEN = 0.0
    return env, tot, nf, rec


# ── 시나리오 (env 0, 위치 m · 침로 deg · 속력 m/s). 목표는 멀리(도착·벽 없음) ──
HEADON = [dict(pos=(0.0, -130.0), hdg=0.0, spd=1.5, maxs=1.5, goal=(0.0, 280.0)),
          dict(pos=(0.0, 130.0), hdg=180.0, spd=1.5, maxs=1.5, goal=(0.0, -280.0))]
CROSS = [dict(pos=(0.0, -150.0), hdg=0.0, spd=1.3, maxs=1.3, goal=(0.0, 280.0)),      # 0: 양보선(상대가 우현)
         dict(pos=(150.0, 0.0), hdg=270.0, spd=1.3, maxs=1.3, goal=(-280.0, 0.0))]     # 1: 유지선
OVERTAKE = [dict(pos=(0.0, -200.0), hdg=0.0, spd=1.6, maxs=1.6, goal=(0.0, 280.0)),    # 0: 추월선(빠름)
            dict(pos=(0.0, -80.0), hdg=0.0, spd=0.8, maxs=1.6, goal=(0.0, 280.0))]     # 1: 추월당하는 배


def turn_then_hold(rud0, rud1, t_on):
    return lambda t, env: [rud0 if t < t_on else 0.0, rud1 if t < t_on else 0.0]


def rp1_rp2():
    E, N = 2, 16
    envA, envB = make_env(E, N, 11), make_env(E, N, 11)
    gA, gB = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    same, dmin, n_pos, fired = True, 1e9, 0, 0
    for _ in range(300):
        aA = torch.rand(E, N, 2, generator=gA) * 2 - 1
        aB = torch.rand(E, N, 2, generator=gB) * 2 - 1
        aA[..., 1] = aA[..., 1].abs(); aB[..., 1] = aB[..., 1].abs()          # 전진 위주(조우가 생기게)
        vg.ROLE_PROMISE_PEN = 0.0
        _, rA, dA, _ = envA.step(aA)
        vg.ROLE_PROMISE_PEN = PEN
        _, rB, dB, _ = envB.step(aB)
        same &= (torch.equal(envA.pos, envB.pos) and torch.equal(envA.heading, envB.heading)
                 and torch.equal(envA.speed, envB.speed) and torch.equal(dA, dB))
        nf = fails_per_ship(envB)
        fired += int(nf.sum())
        diff = rA - (rB + PEN * nf)
        dmin = min(dmin, float(diff.min()))
        n_pos += int((diff > 1e-3).sum())
    vg.ROLE_PROMISE_PEN = 0.0
    check('rp1 켜도 상태 궤적 불변 · 끄면 판정기 없음', same and envA._rp is None and envB._rp is not None)
    check('rp2 r_off − (r_on + PEN·실패) ≥ 0 (가산만 빠짐) · 빠지는 결정 있음',
          dmin > -1e-4 and n_pos > 0, f"min={dmin:.2e} 빠진 배-결정 {n_pos} · 실패 {fired}")


def rp2b_standon_medium_speed():
    """유지선(1) 속력 0.5 < EFFECTIVE_SPEED_MIN 0.7 → sr 0.71(중간) → 옛 식 +0.5(침로유지) −0.5(속도) = 0.
    새 식은 가산 없이 −0.5 → 두 보상 차 = 0.45·0.5·riskw·10 ≥ 2.9. 합계에 clamp 를 걸면 차가 0 이 됨."""
    ships = [dict(pos=(0.0, -110.0), hdg=0.0, spd=1.0, maxs=1.0, goal=(0.0, 280.0)),
             dict(pos=(55.0, 0.0), hdg=270.0, spd=0.5, maxs=1.0, goal=(-280.0, 0.0))]
    envA, envB = make_env(1, 2), make_env(1, 2)
    place(envA, ships); place(envB, ships)
    hit, best = 0, 0.0
    for t in range(250):                                   # 레이더 진입 ≈ 60 s(150결정), 충돌 ≈ 110 s(275결정)
        a = act(envA, [0.0, 0.0], [1.0, 0.5])
        vg.ROLE_PROMISE_PEN = 0.0
        _, rA, _, _ = envA.step(a)
        vg.ROLE_PROMISE_PEN = PEN
        _, rB, _, _ = envB.step(a)
        early = envB._last_pw['tcpa'][0, 1, 0] > vg.RULE_17B_TIME
        if int(envB.situation[0, 1]) == 2 and bool(early):
            d = float(rA[0, 1] - (rB[0, 1] + PEN * fails_per_ship(envB)[0, 1]))
            best = max(best, d)
            hit += int(d > 2.0)
    vg.ROLE_PROMISE_PEN = 0.0
    check('rp2b 유지선 침로유지+중간속력: 새 보상이 −0.5 항을 받음(합계 clamp 버그 검출)', hit > 0,
          f"해당 결정 {hit}개, 최대 차 {best:.2f}")


def rp3_headon_success():
    env, tot, nf, _ = run_scenario(HEADON, turn_then_hold(0.5, 0.5, 30), 260)
    check('rp3 정면 둘 다 우현 → 판정 1·성공 1·실패 0', tot['judged'] == 1 and tot['success'] == 1 and tot['fail'] == 0
          and float(nf.sum()) == 0.0, str(tot))


def rp4_headon_port_fail():
    # 쌍둥이: PEN=20 과 PEN=1e-9(판정기 동일, 벌점만 ~0) — 결정마다 보상 차 = (20 − 1e-9)·실패 수
    pol = turn_then_hold(-0.5, 0.5, 30)
    vg.ROLE_PROMISE_PEN = PEN
    envP, envT = make_env(1, 2), make_env(1, 2)
    place(envP, HEADON); place(envT, HEADON)
    tot_fail, err = torch.zeros(1, 2), 0.0
    for t in range(260):
        a = act(envP, pol(t, envP))
        vg.ROLE_PROMISE_PEN = PEN
        _, rP, _, _ = envP.step(a)
        vg.ROLE_PROMISE_PEN = 1e-9
        _, rT, _, _ = envT.step(a)
        nf = fails_per_ship(envP)
        tot_fail += nf
        err = max(err, float(((rT - rP) - (PEN - 1e-9) * nf).abs().max()))
    vg.ROLE_PROMISE_PEN = 0.0
    check('rp4 정면 한 배 좌현 → 두 배 각각 실패 정확히 1회, 보상 차 = PEN·실패', tot_fail.tolist() == [[1.0, 1.0]] and err < 1e-3,
          f"실패 {tot_fail.tolist()} max|오차|={err:.1e}")


def rp5_crossing():
    _, tot, nf, rec = run_scenario(CROSS, turn_then_hold(0.6, 0.0, 45), 320)
    roles = (int(rec[0]['role_i'][0, 0, 1]), int(rec[0]['role_j'][0, 0, 1]))
    check('rp5a 교차 역할 = (양보 3, 유지 2)', roles == (3, 2), str(roles))
    check('rp5b 양보선 우현 + 유지선 유지 → 성공', tot['judged'] == 1 and tot['success'] == 1, str(tot))
    _, tot2, nf2, _ = run_scenario(CROSS, turn_then_hold(0.6, 0.6, 45), 320)
    check('rp5c 유지선이 17(b) 전에 크게 돔 → 두 배 모두 실패', tot2['judged'] == 1 and tot2['fail'] == 1
          and nf2.tolist() == [[1.0, 1.0]], str(tot2))


def rp6_overtaking():
    _, tot, _, rec = run_scenario(OVERTAKE, turn_then_hold(0.4, 0.0, 40), 400)
    roles = (int(rec[0]['role_i'][0, 0, 1]), int(rec[0]['role_j'][0, 0, 1]))
    check('rp6 추월선 4 · 추월당하는 배 = 보완 규칙 유지 2', roles == (4, 2) and tot['start'] >= 1, f"{roles} {tot}")


def rp7_collision():
    env, tot, nf, rec = run_scenario(HEADON, lambda t, e: [0.0, 0.0], 240)
    oc = [int(r['oc'][0, 0]) for r in rec]
    hit = vg.OUT_COLLISION_VESSEL in oc
    k = oc.index(vg.OUT_COLLISION_VESSEL) if hit else -1
    ok_pen = hit and float(rec[k]['r'][0, 0]) < vg.COLLISION_PENALTY - PEN + 50.0    # −300 −20 + shaping 여유
    check('rp7 두 배끼리 충돌 → 판정·실패 1 (각 배 1회), −300 도 그대로', hit and tot['coll'] == 1 and tot['fail'] == 1
          and nf.tolist() == [[1.0, 1.0]] and ok_pen, f"{tot} r={float(rec[k]['r'][0, 0]) if hit else None}")


def rp8_discard():
    near_goal = [dict(HEADON[0], goal=(0.0, -100.0)), HEADON[1]]                 # 0 번이 조우 중 도착
    _, t1, n1, _ = run_scenario(near_goal, lambda t, e: [0.0, 0.0], 80)
    timeout = [dict(HEADON[0], steps=vg.MAX_EPISODE_STEPS - 400), HEADON[1]]   # 0 번이 40결정 뒤 시간초과
    _, t2, n2, _ = run_scenario(timeout, lambda t, e: [0.0, 0.0], 60)
    third = HEADON + [dict(pos=(1.0, -110.0), hdg=180.0, spd=1.5, maxs=1.5, goal=(0.0, -280.0))]   # 2 번이 0 번과 곧 충돌(횡 1 m < 선폭 1.93 m)
    _, t3, n3, rec3 = run_scenario(third, lambda t, e: [0.0, 0.0, 0.0], 40, N=3)
    disc01 = sum(int(r['ev']['discard'][0, 0, 1]) for r in rec3)
    coll02 = sum(int(r['ev']['coll'][0, 0, 2]) for r in rec3)
    ok = (t1['discard'] >= 1 and t1['judged'] == 0 and float(n1.sum()) == 0                      # 도착: 벌점 없음
          and t2['discard'] >= 1 and t2['judged'] == 0 and n2.tolist() == [[1.0, 0.0]]            # 시간초과: 그 배만 1
          and disc01 == 1 and coll02 == 1 and n3.tolist() == [[2.0, 0.0, 1.0]])                   # 0: (0,2) 충돌 실패 + (0,1) 이탈 1
    check('rp8 도착=폐기·벌점0 / 시간초과·제3선 충돌 = 폐기 + 끝난 배만 조우당 실패', ok,
          f"도착 {t1} n={n1.tolist()} / 시간초과 n={n2.tolist()} / 제3선: (0,1) 폐기 {disc01} (0,2) 충돌 {coll02} n={n3.tolist()}")


def rp9_respawn_reset():
    near_goal = [dict(HEADON[0], goal=(0.0, -100.0)), HEADON[1]]
    _, _, _, rec = run_scenario(near_goal, lambda t, e: [0.0, 0.0], 120)
    k = next((i for i, r in enumerate(rec) if bool(r['d'][0, 0])), None)
    ok = k is not None and not bool(rec[k]['active'][0, 0, :].any()) and not bool(rec[k]['active'][0, :, 0].any())
    later = [r for r in rec[k + 1:]] if k is not None else []
    new_ok = all(int(r['t_start'][0, 0, 1]) > k or not bool(r['active'][0, 0, 1]) for r in later)
    check('rp9 respawn 한 배는 그 결정에 행·열 비활성, 이후 조우는 새로 시작', ok and new_ok, f"respawn 결정 {k}")


class _FakeEnv:
    def __init__(self):
        self.E, self.N, self.device, self.dtype = 1, 2, torch.device('cpu'), torch.float32
        self.pos = torch.tensor([[[0.0, -50.0], [0.0, 50.0]]])
        self.heading = torch.tensor([[0.0, 180.0]])
        self.speed = torch.tensor([[1.0, 1.0]])


def rp10_debounce():
    env = _FakeEnv()
    tr = vg.RolePromiseTracker(env)

    def pw(raw):
        d = torch.tensor([[[0.0, 100.0], [100.0, 0.0]]])
        rt = torch.tensor([[[0.0, raw], [raw, 0.0]]])
        return {'dist': d, 'raw_tcpa': rt, 'tcpa': rt.clamp(min=0), 'dcpa': torch.zeros(1, 2, 2)}
    oc = torch.zeros(1, 2, dtype=torch.long)
    seq = [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, -1.0, 2.0, -1.0, -1.0, 3.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    ends = []
    for raw in seq:
        tr.update(env, pw(raw), oc)
        ends.append(int(tr.events['judged'].sum() + tr.events['discard'].sum()))
    # 시작 1결정(n=1) + 5 → n 7 에서 첫 음수. 음·양 흔들림은 초기화, 3 연속 음수(인덱스 11·12·13)의 셋째에서 종료 1회
    check('rp10 CPA debounce: 3 연속 음수에서만 1회 종료', sum(ends) == 1 and ends.index(1) == 13, str(ends))


def _wrap(x):
    return (x + 180.0) % 360.0 - 180.0


class _Ref:
    """느린 참조 구현 — 쌍별 파이썬 루프로 스펙 §2 를 그대로 옮김(판정기 텐서 코드와 독립). 입력은 판정기가 받은 것과 같은 값."""

    def __init__(self):
        self.P = {}
        self.t = 0

    def step(self, pos, hdg, spd, dist, raw, tcpa, dcpa, oc, role_raw):
        R = float(cfg.COMM_RANGE)
        E, N = len(hdg), len(hdg[0])
        out = []
        for key in sorted(self.P):
            e, i, j = key
            p = self.P[key]
            T32 = lambda x: torch.tensor(x, dtype=torch.float32)
            dpi = float(vg._wrap180(T32(hdg[e][i]) - T32(p['h0i'])))
            dpj = float(vg._wrap180(T32(hdg[e][j]) - T32(p['h0j'])))
            p['dmaxi'], p['dmini'] = max(p['dmaxi'], dpi), min(p['dmini'], dpi)
            p['dmaxj'], p['dminj'] = max(p['dmaxj'], dpj), min(p['dminj'], dpj)
            if tcpa[e][i][j] > vg.RULE_17B_TIME:
                p['soi'], p['soj'] = max(p['soi'], abs(dpi)), max(p['soj'], abs(dpj))
            p['mind'] = min(p['mind'], dist[e][i][j])
            p['n'] += 1
            p['cpa'] = p['cpa'] + 1 if raw[e][i][j] < 0 else 0
            p['far'] = p['far'] + 1 if dist[e][i][j] > R else 0
            di, dj = oc[e][i] != 0, oc[e][j] != 0
            pc = oc[e][i] == vg.OUT_COLLISION_VESSEL and oc[e][j] == vg.OUT_COLLISION_VESSEL and dist[e][i][j] <= vg._PAIR_COLL_DIST
            if pc:
                kind = 'fail'
            elif di or dj:
                kind = 'discard'
            elif p['cpa'] >= cfg.ROLE_END_CPA or p['far'] >= cfg.ROLE_END_FAR:
                if p['n'] < cfg.ROLE_MIN_STEPS:
                    kind = 'discard'
                else:
                    ok = lambda r, mx, mn, so: (mx >= cfg.ROLE_GIVEWAY_MIN_DEG and mn >= -cfg.ROLE_PORT_TOL_DEG) \
                        if r in (1, 3) else (so <= cfg.ROLE_STANDON_MAX_DEG if r == 2 else True)
                    good = ok(p['ri'], p['dmaxi'], p['dmini'], p['soi']) and ok(p['rj'], p['dmaxj'], p['dminj'], p['soj']) \
                        and p['mind'] >= cfg.ROLE_SAFE_DIST
                    kind = 'success' if good else 'fail'
            else:
                continue
            out.append((e, i, j, kind, p['t0']))
            del self.P[key]
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
                                                 cpa=0, far=0, t0=self.t)
        self.t += 1
        return out


def rp11_reference():
    E, N, T = 3, 16, 1500
    vg.ROLE_PROMISE_PEN = PEN
    env = make_env(E, N, 21)
    cap = {}
    orig = vg.RolePromiseTracker.update

    def spy(self, env_, pw, outcome):
        # 판정기가 받은 입력을 그대로 보관(respawn 전 상태)
        to = env_.pos[:, None, :, :] - env_.pos[:, :, None, :]
        h = env_.heading * vg.DEG
        fx, fz = torch.sin(h)[:, :, None], torch.cos(h)[:, :, None]
        gx, gz = torch.sin(h)[:, None, :], torch.cos(h)[:, None, :]
        b = torch.atan2(fz * to[..., 0] - fx * to[..., 1], fx * to[..., 0] + fz * to[..., 1]) / vg.DEG
        ob = torch.atan2(gx * to[..., 1] - gz * to[..., 0], -(gx * to[..., 0] + gz * to[..., 1])) / vg.DEG
        fast = env_.speed[:, :, None] > env_.speed[:, None, :] * 1.1
        cap.update(hdg=env_.heading.tolist(), dist=pw['dist'].tolist(), raw=pw['raw_tcpa'].tolist(),
                   tcpa=pw['tcpa'].tolist(), dcpa=pw['dcpa'].tolist(), oc=outcome.tolist(),
                   role=vg.encounter_role(b, ob, pw['dist'], pw['raw_tcpa'] >= 0, fast, float(cfg.COMM_RANGE)).tolist())
        return orig(self, env_, pw, outcome)
    vg.RolePromiseTracker.update = spy
    ref = _Ref()
    mism, n = [], {'success': 0, 'fail': 0, 'discard': 0}
    try:
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
            a = torch.stack([a0, torch.ones_like(a0)], dim=-1)
            env.step(a)
            ev = env._rp.events
            got = set()
            for kind, m in (('success', ev['success']), ('fail', ev['fail']), ('discard', ev['discard'])):
                for e_, i_, j_ in m.nonzero().tolist():
                    got.add((e_, i_, j_, kind, int(ev['t_start'][e_, i_, j_])))
            want = set(ref.step(None, cap['hdg'], None, cap['dist'], cap['raw'], cap['tcpa'], cap['dcpa'], cap['oc'],
                                cap['role']))
            if got != want:
                mism.append((t, sorted(got - want)[:3], sorted(want - got)[:3]))
            for x in want:
                n[x[3]] += 1
    finally:
        vg.RolePromiseTracker.update = orig
        vg.ROLE_PROMISE_PEN = 0.0
    judged = n['success'] + n['fail']
    nonvac = judged >= 50 and n['success'] >= 10 and n['discard'] >= 5
    check('rp11 16척 규칙 정책: 판정기 이벤트 == 참조 구현(전 결정) + 공허 통과 아님',
          not mism and nonvac, f"{n} 불일치 결정 {len(mism)} {mism[:2]}")


def rp12_small_and_consts():
    ok = True
    vg.ROLE_PROMISE_PEN = PEN
    try:
        for N in (1, 2):
            env = make_env(2, N, 3)
            for _ in range(20):
                env.step(torch.zeros(2, N, 2))
    except Exception as ex:   # noqa: BLE001
        ok = False
        print('    ', ex)
    finally:
        vg.ROLE_PROMISE_PEN = 0.0
    same = all(getattr(vg, k) == getattr(cfg, k) for k in (
        'ROLE_GIVEWAY_MIN_DEG', 'ROLE_PORT_TOL_DEG', 'ROLE_STANDON_MAX_DEG', 'ROLE_SAFE_DIST', 'ROLE_END_CPA',
        'ROLE_END_FAR', 'ROLE_MIN_STEPS'))
    pinned = (cfg.ROLE_GIVEWAY_MIN_DEG, cfg.ROLE_PORT_TOL_DEG, cfg.ROLE_STANDON_MAX_DEG, cfg.ROLE_SAFE_DIST,
              cfg.ROLE_END_CPA, cfg.ROLE_END_FAR, cfg.ROLE_MIN_STEPS) == (10.0, 5.0, 10.0, 24.0, 3, 5, 5)
    check('rp12 N=1·2 동작 · 판정 상수 = config = 스펙 고정값', ok and same and pinned and cfg.ROLE_SAFE_DIST == vg.DCPA_RISK)


def rp13_wall():
    # 동쪽 벽(x 299.5) 옆 정면 조우(0 북행 x=288, 1 남행) → 0 번이 우현 전타로 벽에 박음 → (0,1) 폐기 + 0 번만 실패 1
    ships = [dict(pos=(288.0, -60.0), hdg=0.0, spd=1.5, maxs=1.5, goal=(288.0, 280.0)),
             dict(pos=(288.0, 60.0), hdg=180.0, spd=1.5, maxs=1.5, goal=(288.0, -280.0))]
    vg.ROLE_PROMISE_PEN = PEN
    env = make_env(1, 2)
    place(env, ships)
    nf, hit, started = torch.zeros(1, 2), False, 0
    for t in range(80):
        _, r, d, oc = env.step(act(env, [1.0, 0.0]))
        started += int(env._rp.events['start'].sum())
        nf += fails_per_ship(env)
        if int(oc[0, 0]) == vg.OUT_COLLISION_OBSTACLE:
            hit = True
            break
    vg.ROLE_PROMISE_PEN = 0.0
    check('rp13 조우 중 벽 충돌 → 그 배만 실패 1(벽 탈출 빈틈 막힘)', hit and started >= 1 and nf.tolist() == [[1.0, 0.0]],
          f"벽충돌={hit} 조우시작={started} 실패={nf.tolist()}")


if __name__ == '__main__':
    torch.set_num_threads(1)
    print('=' * 78)
    print('역할 약속 보상(ROLE_PROMISE_PEN) 단위 테스트')
    print('=' * 78)
    rp1_rp2()
    rp2b_standon_medium_speed()
    rp3_headon_success()
    rp4_headon_port_fail()
    rp5_crossing()
    rp6_overtaking()
    rp7_collision()
    rp8_discard()
    rp9_respawn_reset()
    rp10_debounce()
    rp11_reference()
    rp12_small_and_consts()
    rp13_wall()
    print('=' * 78)
    print(f"VERDICT: {'ALL PASS' if all(RES) else 'FAIL ' + str(RES.count(False))}")
    sys.exit(0 if all(RES) else 1)
