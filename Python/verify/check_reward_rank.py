"""check_reward_rank.py — 학습 전 보상 순위 게이트 v2 (2026-09-30, 스펙 2026-09-30-reward-v3-decode-sweep-design.md §4).
   (2026-09-29 G6 판 = 정책 8 · 설정 old/new · 규칙 (i) 만 — 2026-09-29-role-promise-design.md §5. 그 정책 8·규칙 (i) 은 그대로 둠)

질문: 보상이 '실제로 피하는 배'에 '안 피하는 배'보다 높은 점수를 주는가? 학습 없이, 규칙 정책 14개를 학습기와 같은 env·보상으로
굴려 할인 반환(γ = config.DISCOUNT_FACTOR)·결정당 보상·에피소드 반환·roleKeptSafe 를 비교한다.
보상 설정마다 새 서브프로세스(모듈 전역·env 가 섞이지 않게).

  정책 14: goal(목표로 직진) · stbd(레이더 안 상황이 있으면 무조건 우현 — 학습 OFF 의 명령 패턴) · radar(레이더 안 규칙:
        양보·정면·추월 우현, 유지 17(b) 전 유지) · vo56 / vo56s(56 m 안 상대 위치·속도로 궤적 예측 회피, s = 우현만) ·
        seeker(300 m 안 가까운 배 쪽으로 틀어 충돌코스를 만든 뒤 레이더 규칙으로 비킴 = 조우 유도) ·
        wall(가장 가까운 벽으로 직진 = 자폭) · wallesc(평소 목표, 조우가 걸리면 가까운 벽으로 = 벽 탈출)
        ★2026-09-29c wall·wallesc 추가: r_ trunk 가 벽 충돌로 무너진 빈틈(조우 중 벽 충돌 = 판정 폐기) 막힘을 확인
        ★2026-09-30 추가(스펙 §4):
        drift(초기 정책 모사: a0 = tanh(N(0, e^−1))·a1 = tanh(N(0, e^−0.5)) 결정마다 추첨, 전용 Generator(seed) — env.gen 안 씀) ·
        idle(목표 조향 + 목표속도 0.25·max = a1 −0.5) · turnless(목표 조향, 명령 타각 |δ| ≤ 2° = a0 ∈ ±2/MAX_TURN_RATE) ·
        weaver(vo56 + 조우가 걸린 동안(pending) 목표 침로에 ±10° 를 10결정마다 교대로 더함) ·
        vo300 / vo300s(같은 VO, R = 300 m · H = 150 s · dt = 4 s = 통신 정보 상한 참조)
        drift·idle 외 전부 a1 = +1(전속).
  설정 4(서브프로세스마다 env 로 덮어씀, 나머지 VESSEL_* 는 부모 그대로):
        old = PEN 0(배치 X 옛 보상) · s = PEN 20 · 판정기 end(s_ 배치) — 둘 다 기록만
        v3 = PEN 20 · ROLE_JUDGE v2 · FORWARD_COEF 0 · TIME_PENALTY 0.035 · RISK_DCPA_GATE_M 48
        v3nogate = v3 에서 DCPA 게이트만 끔(RISK_DCPA_GATE_M 0)
        old·s 는 v3 키 4개를 기본값(end·0.1·0.07·0)으로 명시 고정 — 부모 셸이 배치용 v3 export 를 물려줘도 옛 보상 그대로.
        자식은 설정의 env 키마다 config.py 가 그 값을 실제로 읽었는지 확인하고, 아니면 키 이름을 찍고 중단(조용히 무시 금지).
  지표(설정·시드·정책): G(할인 반환, 창 끝 300 결정 절단) · r_per_dec · ep_return(기록 창 안에서 끝난 에피소드의 무할인 반환 평균 —
        배마다 t=0 부터 누적, done 에 읽고 0. 초기 스폰의 위상 분산 에피소드도 포함) · eps/vColl/goal/oColl ·
        roleKeptSafe = success/judged(env.enable_role_tracker() 의 판정기 — 설정의 ROLE_JUDGE 가 end/v2 를 고름. PEN 0 에서도 지표로 돈다)
  판정(v3 · v3nogate, 시드마다 전부 성립해야):
        (i)   G: min(vo56, vo56s, vo300, vo300s) − max(나머지 10) > 0.05·|max|
        (ii)  r/dec: vo56 − max(goal, stbd, radar) ≥ 0.05 그리고 drift < goal − 0.2 그리고 idle < goal − 0.2
        (iii) ep_return 순서 = (i)
        (iv)  정보가치: vo300 ≥ vo56 (G 와 r/dec 둘 다)
        (v)   판정기 보정: roleKeptSafe v2 순서 vo300s > radar > goal
  결정(사전등록 §4, 결과 전 고정): v3 통과 → v3 로 진행 / v3 실패 & v3nogate 통과 → 'DECISION: run without DCPA gate' /
        둘 다 실패 → 'DECISION: stop, report'. rc 0 = v3 또는 v3nogate 통과. old·s 에도 같은 규칙을 찍지만 기록만.

  python verify/check_reward_rank.py --out <json>          # 마지막 줄 'RANK GATE v2: PASS' 여야 함. rc 0/1(불통과)/2(자식 중단)
  env: 학습 배치와 같은 VESSEL_*(imo·none·COMM_RANGE 300·COLREGS_FAR_RANGE 0 …) 를 부모에서 물려받는다. 설정 키 5개만 설정별로 덮어씀.
  --settings old,s 처럼 일부만 돌리면 v3·v3nogate 없이는 판정 없음(SKIP, rc 1). --jobs N 으로 설정 서브프로세스를 동시에.
"""
import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
POLICIES = ('goal', 'stbd', 'radar', 'vo56', 'vo56s', 'seeker', 'wall', 'wallesc',
            'drift', 'idle', 'turnless', 'weaver', 'vo300', 'vo300s')
AVOIDERS = ('vo56', 'vo56s', 'vo300', 'vo300s')
OTHERS = tuple(p for p in POLICIES if p not in AVOIDERS)      # 나머지 10
BASE3 = ('goal', 'stbd', 'radar')                              # (ii) 기준선
# 설정 → 자식 env(키 이름은 config.py 의 _env_* 이름 그대로 — ENV2CFG 로 실제 읽힌 값과 대조)
_LEGACY = {'VESSEL_ROLE_JUDGE': 'end', 'VESSEL_FORWARD_COEF': '0.1', 'VESSEL_TIME_PENALTY': '0.07',
           'VESSEL_RISK_DCPA_GATE_M': '0'}
_V3 = {'VESSEL_ROLE_PROMISE_PEN': '20', 'VESSEL_ROLE_JUDGE': 'v2', 'VESSEL_FORWARD_COEF': '0',
       'VESSEL_TIME_PENALTY': '0.035', 'VESSEL_RISK_DCPA_GATE_M': '48'}
SETTINGS = {
    'old': dict(_LEGACY, VESSEL_ROLE_PROMISE_PEN='0'),
    's': dict(_LEGACY, VESSEL_ROLE_PROMISE_PEN='20'),
    'v3': dict(_V3),
    'v3nogate': dict(_V3, VESSEL_RISK_DCPA_GATE_M='0'),
}
SETTING_ORDER = ('old', 's', 'v3', 'v3nogate')
JUDGED = ('v3', 'v3nogate')                                    # 판정 대상. old·s 는 기록만
ENV2CFG = {'VESSEL_ROLE_PROMISE_PEN': 'ROLE_PROMISE_PEN', 'VESSEL_ROLE_JUDGE': 'ROLE_JUDGE',
           'VESSEL_FORWARD_COEF': 'FORWARD_COEF', 'VESSEL_TIME_PENALTY': 'TIME_PENALTY',
           'VESSEL_RISK_DCPA_GATE_M': 'RISK_DCPA_GATE_M'}
REL_MARGIN = 0.05        # (i)(iii): min(회피) − max(나머지) > 0.05·|max|  (부호 무관 여유)
RDEC_MARGIN = 0.05       # (ii): vo56 − max(goal, stbd, radar) ≥ 0.05
LAZY_MARGIN = 0.2        # (ii): drift·idle < goal − 0.2
G_TAIL_CUT = 300         # G 창 끝 절단(γ^300 ≈ 0.05)


def _fmt(x, spec):
    return 'n/a'.rjust(len(f"{0:{spec}}")) if x is None else f"{x:{spec}}"


# ─────────────────────────── child: 한 보상 설정에서 정책 × 시드 ───────────────────────────
def _check_setting(name, want, cfg, vg):
    """Abort with the env key name when config.py (or the vessel_gym mirror) did not take a key of this setting.
    An env key that config.py does not read would otherwise be ignored silently (= a different experiment)."""
    for key, val in want.items():
        attr = ENV2CFG[key]
        if not hasattr(cfg, attr):
            raise SystemExit(f"[{name}] config.py 에 {attr} 없음 → env {key}={val} 가 조용히 무시됨(키 미구현). 중단")
        got = getattr(cfg, attr)
        ok = (str(got).lower() == val.lower()) if isinstance(got, str) else (abs(float(got) - float(val)) < 1e-9)
        if not ok:
            raise SystemExit(f"[{name}] env {key}={val} 를 줬는데 config.{attr}={got!r} 로 읽힘. 중단")
        if hasattr(vg, attr) and getattr(vg, attr) != got:
            raise SystemExit(f"[{name}] vessel_gym.{attr}={getattr(vg, attr)!r} ≠ config.{attr}={got!r} (전역 미러 어긋남). 중단")


def _child(a):
    sys.path.insert(0, PYROOT)
    import torch
    import config as cfg
    import vessel_gym as vg
    want = SETTINGS[a.child]
    _check_setting(a.child, want, cfg, vg)
    assert cfg.DYN_PROFILE == 'imo' and cfg.OBSTACLES_MODE == 'none', (cfg.DYN_PROFILE, cfg.OBSTACLES_MODE)
    torch.set_num_threads(1)
    dev = torch.device(a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    DEG = vg.DEG
    wrap = vg._wrap180

    def goal_bearing(env):
        tg = env.goal - env.pos
        return torch.atan2(tg[..., 0], tg[..., 1]) / DEG

    def goal_a0(env):
        return torch.clamp(wrap(goal_bearing(env) - env.heading) / 20.0, -1, 1)

    def steer_to(env, course):
        # 절대 침로 course[deg] 로 트는 a0 (VO 와 같은 이득: 10° 오차 = 전타)
        return torch.clamp(wrap(course - env.heading) / 10.0, -1, 1)

    def radar_rule(env, a0):
        pw = env._last_pw
        tc = pw['tcpa'].gather(-1, env.danger_idx.unsqueeze(-1)).squeeze(-1)
        sit = env.situation
        gw = (sit == 1) | (sit == 3) | (sit == 4)
        so = sit == 2
        a0 = torch.where(gw, torch.full_like(a0, 0.6), a0)
        a0 = torch.where(so & (tc > vg.RULE_17B_TIME), torch.zeros_like(a0), a0)
        return torch.where(so & (tc <= vg.RULE_17B_TIME), torch.full_like(a0, 0.6), a0)

    def vo_action(env, R, prev, stbd_only, D_safe=30.0, H=60.0, dt=2.0):
        # 스크래치 fleet_vo.vo_action(2026-09-28, 56 m 에서 충돌 6.5–8.6 %) 을 device 인자만 붙여 옮김
        E, N = env.E, env.N
        cands = torch.arange(0.0 if stbd_only else -120.0, 121.0, 15.0, device=dev)
        C = cands.numel()
        gb = goal_bearing(env)
        psi_c = gb.unsqueeze(-1) + cands
        v = env.speed.clamp(min=0.3)
        omega = (v / vg.R_FULL) / DEG if vg.DYN_FORMULA == 'abs' else vg.MAX_YAW_RATE * (env.speed / env.max_speed.clamp(min=1e-6))
        psi = env.heading.unsqueeze(-1).expand(E, N, C).clone()
        p = env.pos.unsqueeze(2).expand(E, N, C, 2).clone()
        h = env.heading * DEG
        vel = torch.stack([torch.sin(h), torch.cos(h)], -1) * env.speed.unsqueeze(-1)
        dist0 = torch.linalg.norm(env.pos.unsqueeze(1) - env.pos.unsqueeze(2), dim=-1)
        eye = torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0)
        seen = (dist0 <= R) & ~eye
        pen_max = None
        for k in range(1, int(H / dt) + 1):
            t = k * dt
            rate = (omega * 0.35 if t <= 5.0 else omega).unsqueeze(-1) * dt
            dpsi = wrap(psi_c - psi)
            psi = psi + torch.clamp(dpsi, -1, 1) * torch.minimum(dpsi.abs(), rate)
            pr = psi * DEG
            p = p + torch.stack([torch.sin(pr), torch.cos(pr)], -1) * (v.unsqueeze(-1).unsqueeze(-1) * dt)
            pj = env.pos + vel * t
            d = torch.linalg.norm(pj.unsqueeze(1).unsqueeze(3) - p.unsqueeze(2), dim=-1)
            pen = 100.0 * (torch.clamp(D_safe - d, min=0) / D_safe) ** 2 / (1.0 + t / 30.0)
            pen_max = pen if pen_max is None else torch.maximum(pen_max, pen)
        risk = (pen_max * seen.unsqueeze(-1)).sum(2)
        cost = risk + cands.abs().view(1, 1, -1) / 90.0 * 30.0 + (cands.view(1, 1, -1) != prev.unsqueeze(-1)).float() * 3.0
        choice = cands[cost.argmin(-1)]
        return steer_to(env, gb + choice), choice

    def seeker_a0(env):
        pw = env._last_pw
        d = pw['dist'] + torch.eye(env.N, device=dev).unsqueeze(0) * 1e9
        rel = env.pos.unsqueeze(1) - env.pos.unsqueeze(2)
        br = wrap(torch.atan2(rel[..., 0], rel[..., 1]) / DEG - env.heading.unsqueeze(-1))
        ahead = (br.abs() < 90.0) & (d <= float(cfg.COMM_RANGE))
        dd = torch.where(ahead, d, torch.full_like(d, 1e9))
        dmin, j = dd.min(-1)
        on_course = ((pw['dcpa'] < vg.DCPA_RISK) & (pw['raw_tcpa'] >= 0) & (d < 1e8)).any(-1)
        toward = torch.clamp(br.gather(-1, j.unsqueeze(-1)).squeeze(-1) / 20.0, -1, 1)
        a0 = torch.where((dmin < 1e8) & ~on_course, toward, goal_a0(env))
        return radar_rule(env, a0)

    def wall_a0(env):
        # 가장 가까운 벽 쪽 방위(동 90 · 서 −90 · 북 0 · 남 180)로 전타
        x, z = env.pos[..., 0], env.pos[..., 1]
        gap = torch.stack([vg.ARENA_INNER - x, vg.ARENA_INNER + x, vg.ARENA_INNER - z, vg.ARENA_INNER + z], -1)
        brg = torch.tensor([90.0, -90.0, 0.0, 180.0], device=dev)[gap.argmin(-1)]
        return torch.clamp(wrap(brg - env.heading) / 20.0, -1, 1)

    def pending(env):
        # 판정기 시작 조건과 같은 모양(300 m · 접근 · dcpa < 24) — 조우가 걸린 배
        pw = env._last_pw
        d = pw['dist'] + torch.eye(env.N, device=dev).unsqueeze(0) * 1e9
        return ((pw['dcpa'] < vg.DCPA_RISK) & (pw['raw_tcpa'] >= 0) & (d <= float(cfg.COMM_RANGE))).any(-1)

    gamma = float(cfg.DISCOUNT_FACTOR)
    turnless_cap = 2.0 / float(vg.MAX_TURN_RATE)                 # 명령 타각 2° (cmd_rudder = a0·MAX_TURN_RATE)
    used = {ENV2CFG[k]: getattr(cfg, ENV2CFG[k]) for k in want}   # 실제 읽힌 값(기록)
    tracker = None
    rows = []
    t0 = time.time()
    for seed in a.seeds:
        for pol in POLICIES:
            env = vg.VesselBatchEnv(num_envs=a.E, n_vessels=16, device=dev, seed=seed, ring_scale=1.0, crossing=0,
                                    risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE,
                                    farfield_coef=cfg.FARFIELD_COEF, perpair_coef=cfg.PERPAIR_COEF, perpair_exp=3.0)
            # 판정기: 설정의 ROLE_JUDGE(end/v2)가 고른 것. reset() 앞에 켜서 reset 이 판정기도 초기화. PEN 0 에서도 지표로 돈다
            tr = env.enable_role_tracker()
            judge_used = getattr(env, '_rp_judge', None) or ('v2' if 'v2' in type(tr).__name__.lower() else 'end')
            if judge_used != cfg.ROLE_JUDGE:
                raise SystemExit(f"[{a.child}] VESSEL_ROLE_JUDGE={cfg.ROLE_JUDGE} 인데 env.enable_role_tracker() 가 "
                                 f"{type(tr).__name__}(judge={judge_used}) 를 만들었음. 중단")
            tracker = type(tr).__name__
            env.reset()
            gen = torch.Generator(device=dev).manual_seed(seed)          # drift 전용 난수(env.gen 과 분리)
            zf = torch.zeros(env.E, env.N, device=dev, dtype=env.dtype)
            prev, ep_sum = zf.clone(), zf.clone()
            R, D = [], []
            cnt = torch.zeros(6, device=dev, dtype=torch.long)          # eps · vColl · goal · oColl · judged · success
            ep_tot = torch.zeros((), device=dev, dtype=env.dtype)        # 창 안에서 끝난 에피소드 반환 합
            for t in range(a.burn + a.T):
                a1 = None
                if pol == 'goal':
                    a0 = goal_a0(env)
                elif pol == 'stbd':
                    a0 = torch.where(env.situation > 0, torch.full_like(prev, 0.6), goal_a0(env))
                elif pol == 'radar':
                    a0 = radar_rule(env, goal_a0(env))
                elif pol == 'seeker':
                    a0 = seeker_a0(env)
                elif pol == 'wall':
                    a0 = wall_a0(env)
                elif pol == 'wallesc':
                    a0 = torch.where(pending(env), wall_a0(env), goal_a0(env))
                elif pol == 'drift':
                    # 초기 정책 모사: 평균 0 · per-dim logstd [−1.0, −0.5](networks ControlActor 초기값) · tanh squash
                    a0 = torch.tanh(torch.randn(env.E, env.N, generator=gen, device=dev, dtype=env.dtype) * math.exp(-1.0))
                    a1 = torch.tanh(torch.randn(env.E, env.N, generator=gen, device=dev, dtype=env.dtype) * math.exp(-0.5))
                elif pol == 'idle':
                    a0 = goal_a0(env)
                    a1 = torch.full_like(a0, -0.5)                       # target_speed = (a1+1)/2·max = 0.25·max
                elif pol == 'turnless':
                    a0 = torch.clamp(goal_a0(env), -turnless_cap, turnless_cap)
                elif pol == 'weaver':
                    a0, prev = vo_action(env, 56.0, prev, False)
                    off = 10.0 if (t // 10) % 2 == 0 else -10.0          # 10결정마다 ±10° 교대
                    a0 = torch.where(pending(env), steer_to(env, goal_bearing(env) + prev + off), a0)
                elif pol in ('vo300', 'vo300s'):
                    a0, prev = vo_action(env, 300.0, prev, pol.endswith('s'), H=150.0, dt=4.0)
                else:
                    a0, prev = vo_action(env, 56.0, prev, pol.endswith('s'))
                if a1 is None:
                    a1 = torch.ones_like(a0)
                act = torch.stack([a0, a1], -1)
                _, r, done, oc = env.step(act)
                prev = torch.where(done, torch.zeros_like(prev), prev)
                ep_sum = ep_sum + r
                if t >= a.burn:
                    ev = tr.events
                    if t == a.burn and (ev is None or 'judged' not in ev or 'success' not in ev):
                        raise SystemExit(f"[{a.child}] 판정기 {tracker}.events 에 judged/success 없음: "
                                         f"{None if ev is None else sorted(ev)}. 중단")
                    R.append(r)
                    D.append(done)
                    cnt = cnt + torch.stack([done.sum(), (oc == vg.OUT_COLLISION_VESSEL).sum(), (oc == vg.OUT_GOAL).sum(),
                                             (oc == vg.OUT_COLLISION_OBSTACLE).sum(), ev['judged'].sum(), ev['success'].sum()])
                    ep_tot = ep_tot + (ep_sum * done.to(ep_sum.dtype)).sum()
                ep_sum = torch.where(done, torch.zeros_like(ep_sum), ep_sum)
            Rt, Dt = torch.stack(R).cpu(), torch.stack(D).cpu().float()
            G = torch.zeros_like(Rt[0])
            Gs = torch.zeros_like(Rt)
            for t in range(Rt.shape[0] - 1, -1, -1):
                G = Rt[t] + gamma * (1.0 - Dt[t]) * G
                Gs[t] = G
            keep = max(1, Rt.shape[0] - G_TAIL_CUT)                       # 창 끝 절단(γ^300≈0.05) 제외
            eps, coll, goal, ocoll, judged, succ = [int(x) for x in cnt.tolist()]
            row = {'seed': seed, 'policy': pol, 'G': float(Gs[:keep].mean()), 'r_per_dec': float(Rt.mean()),
                   'ep_return': (float(ep_tot) / eps) if eps > 0 else None,
                   'eps': eps, 'vColl%': 100.0 * coll / max(eps, 1), 'goal%': 100.0 * goal / max(eps, 1),
                   'oColl%': 100.0 * ocoll / max(eps, 1),
                   'judged': judged, 'success': succ, 'roleKeptSafe': (succ / judged) if judged > 0 else None}
            rows.append(row)
            print(f"  [{a.child}] seed {seed} {pol:8s} G={row['G']:8.2f} r/dec={row['r_per_dec']:6.3f} "
                  f"ep={_fmt(row['ep_return'], '8.1f')} vColl={row['vColl%']:5.1f}% oColl={row['oColl%']:5.1f}% "
                  f"goal={row['goal%']:5.1f}% eps={eps:4d} rks({cfg.ROLE_JUDGE})={_fmt(row['roleKeptSafe'], '5.3f')} "
                  f"j={judged}", flush=True)
    print(f"  [{a.child}] {len(rows)} rows in {time.time() - t0:.0f}s ({tracker}, {used})", flush=True)
    print('JSON ' + json.dumps({'setting': a.child, 'env': want, 'cfg': used, 'pen': float(cfg.ROLE_PROMISE_PEN),
                                'judge': cfg.ROLE_JUDGE, 'tracker': tracker, 'sec': time.time() - t0, 'rows': rows}),
          flush=True)


# ─────────────────────────── parent ───────────────────────────
def _rules(rows):
    """Per-seed rule evaluation (spec §4 (i)-(v)) on one setting's rows -> list of per-seed dicts (None = undefined = fail)."""
    out = []
    for seed in sorted({r['seed'] for r in rows}):
        by = {r['policy']: r for r in rows if r['seed'] == seed}
        G = {p: by[p]['G'] for p in POLICIES}
        rd = {p: by[p]['r_per_dec'] for p in POLICIES}
        ep = {p: by[p]['ep_return'] for p in POLICIES}
        rk = {p: by[p]['roleKeptSafe'] for p in POLICIES}

        def sep(m):
            # (i)/(iii) 모양: min(회피 4) − max(나머지 10) > REL_MARGIN·|max|
            if any(m[p] is None for p in POLICIES):
                return {'lo': None, 'lo_p': None, 'hi': None, 'hi_p': None, 'ok': False}
            lo_p, hi_p = min(AVOIDERS, key=lambda p: m[p]), max(OTHERS, key=lambda p: m[p])
            lo, hi = m[lo_p], m[hi_p]
            return {'lo': lo, 'lo_p': lo_p, 'hi': hi, 'hi_p': hi_p, 'ok': (lo - hi) > REL_MARGIN * abs(hi)}

        r1, r3 = sep(G), sep(ep)
        d_base = rd['vo56'] - max(rd[p] for p in BASE3)
        d_drift, d_idle = rd['drift'] - rd['goal'], rd['idle'] - rd['goal']
        r2 = {'vo56_minus_base3': d_base, 'drift_minus_goal': d_drift, 'idle_minus_goal': d_idle,
              'ok': d_base >= RDEC_MARGIN and d_drift < -LAZY_MARGIN and d_idle < -LAZY_MARGIN}
        r4 = {'dG': G['vo300'] - G['vo56'], 'dr': rd['vo300'] - rd['vo56'],
              'ok': G['vo300'] >= G['vo56'] and rd['vo300'] >= rd['vo56']}
        r5 = {'vo300s': rk['vo300s'], 'radar': rk['radar'], 'goal': rk['goal'],
              'ok': all(rk[p] is not None for p in ('vo300s', 'radar', 'goal')) and rk['vo300s'] > rk['radar'] > rk['goal']}
        out.append({'seed': seed, 'i_G': r1, 'ii_rdec': r2, 'iii_ep': r3, 'iv_info': r4, 'v_rks': r5,
                    'pass': r1['ok'] and r2['ok'] and r3['ok'] and r4['ok'] and r5['ok']})
    return out


def _print_verdict(setting, verdict, judged_setting):
    tag = '' if judged_setting else ' (기록만)'
    for v in verdict:
        r1, r2, r3, r4, r5 = v['i_G'], v['ii_rdec'], v['iii_ep'], v['iv_info'], v['v_rks']
        ok = lambda r: 'ok' if r['ok'] else 'NOT ok'
        print(f"  [{setting}] seed {v['seed']}{tag}: "
              f"(i) G {_fmt(r1['lo'], '8.2f')}[{r1['lo_p']}] − {_fmt(r1['hi'], '8.2f')}[{r1['hi_p']}] {ok(r1)} | "
              f"(ii) r/dec vo56−max3={r2['vo56_minus_base3']:+.3f} drift−goal={r2['drift_minus_goal']:+.3f} "
              f"idle−goal={r2['idle_minus_goal']:+.3f} {ok(r2)} | "
              f"(iii) ep {_fmt(r3['lo'], '8.1f')}[{r3['lo_p']}] − {_fmt(r3['hi'], '8.1f')}[{r3['hi_p']}] {ok(r3)} | "
              f"(iv) vo300−vo56 G={r4['dG']:+.2f} r={r4['dr']:+.3f} {ok(r4)} | "
              f"(v) rks vo300s {_fmt(r5['vo300s'], '.3f')} > radar {_fmt(r5['radar'], '.3f')} > goal {_fmt(r5['goal'], '.3f')} {ok(r5)}"
              f" → {'PASS' if v['pass'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--child', default=None)
    ap.add_argument('--seeds', default='1,2,3')
    ap.add_argument('--E', type=int, default=32)
    ap.add_argument('--burn', type=int, default=600)
    ap.add_argument('--T', type=int, default=1500)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--settings', default=','.join(SETTING_ORDER), help='돌릴 설정(쉼표). v3·v3nogate 가 없으면 판정 없음')
    ap.add_argument('--jobs', type=int, default=1, help='설정 서브프로세스 동시 수(같은 GPU 에 작은 프로세스 N 개)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    a.seeds = [int(s) for s in str(a.seeds).split(',') if s]
    if a.child:
        return _child(a)
    settings = [s for s in str(a.settings).split(',') if s]
    bad = [s for s in settings if s not in SETTINGS]
    if bad:
        print(f"RANK GATE v2: 중단 — 모르는 설정 {bad} (가능: {SETTING_ORDER})")
        return 2

    def run(setting):
        env = dict(os.environ)
        env.update(SETTINGS[setting])
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        cmd = [sys.executable, os.path.abspath(__file__), '--child', setting, '--seeds', ','.join(map(str, a.seeds)),
               '--E', str(a.E), '--burn', str(a.burn), '--T', str(a.T), '--device', a.device]
        return subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')

    res, lock, failed = {}, threading.Lock(), []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(run, s): s for s in settings}
        for fut in concurrent.futures.as_completed(futs):
            setting, p = futs[fut], fut.result()
            with lock:
                sys.stdout.write(p.stdout)
                if p.returncode != 0:
                    sys.stdout.write(p.stderr[-4000:])
                    print(f"RANK GATE v2: 중단 — {setting} 서브프로세스 rc={p.returncode}", flush=True)
                    failed.append(setting)
                    for f in futs:
                        f.cancel()
                else:
                    line = [l for l in p.stdout.splitlines() if l.startswith('JSON ')][-1]
                    res[setting] = json.loads(line[5:])
    if failed:
        return 2
    verdict, passed = {}, {}
    for s in [s for s in SETTING_ORDER if s in res]:
        verdict[s] = _rules(res[s]['rows'])
        _print_verdict(s, verdict[s], s in JUDGED)
        n_ok = sum(v['pass'] for v in verdict[s])
        if s in JUDGED:
            passed[s] = n_ok == len(verdict[s]) and n_ok > 0
            print(f"RANK GATE v2 [{s}]: {'PASS' if passed[s] else '★FAIL'} ({n_ok}/{len(verdict[s])} seeds, judge={res[s]['judge']})")
        else:
            print(f"RANK GATE v2 [{s}] (기록만): {n_ok}/{len(verdict[s])} seeds 가 규칙 (i)–(v) 전부 성립, judge={res[s]['judge']}")
    v3_ok, ng_ok = passed.get('v3', False), passed.get('v3nogate', False)
    if not any(s in res for s in JUDGED):
        gate, decision = False, 'DECISION: 없음 — v3·v3nogate 를 안 돌렸음(기록만)'
        print('RANK GATE v2: SKIP (v3·v3nogate 미실행)')
    else:
        gate = v3_ok or ng_ok
        decision = ('DECISION: run v3 (DCPA gate 48 m)' if v3_ok else
                    'DECISION: run without DCPA gate' if ng_ok else 'DECISION: stop, report')
        print(f"RANK GATE v2: {'PASS' if gate else '★FAIL'}  (v3 {'PASS' if v3_ok else 'FAIL'} · "
              f"v3nogate {'PASS' if ng_ok else 'FAIL' if 'v3nogate' in res else '미실행'})")
    print(decision)
    res['verdict'] = verdict
    res['gate'] = {'v3': v3_ok, 'v3nogate': ng_ok, 'pass': gate, 'decision': decision}
    res['args'] = {'seeds': a.seeds, 'E': a.E, 'burn': a.burn, 'T': a.T, 'device': a.device, 'settings': settings,
                   'policies': POLICIES, 'avoiders': AVOIDERS, 'rules': {'rel_margin': REL_MARGIN, 'rdec_margin': RDEC_MARGIN,
                                                                         'lazy_margin': LAZY_MARGIN, 'G_tail_cut': G_TAIL_CUT}}
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    return 0 if gate else 1


if __name__ == '__main__':
    sys.exit(main())
