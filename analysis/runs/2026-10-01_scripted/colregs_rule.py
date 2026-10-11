"""colregs_rule.py — COLREGs-aware rule ship (2026-10-01, author: "COLREGs 규칙 배 시험 + DCPA·fuel·rudder 점검").

Question: with the same rule, does seeing 300 m (comm range) instead of 56 m (radar range) win every report metric,
in particular COLREGs compliance and DCPA, and what does it cost in fuel and rudder?

col<R>: one rule, only the seen range R differs (col56 vs col300). H 150 s, dt 4 s for both.
  Roles come from the env's own COLREGs code: RolePromiseTrackerV2.roles(env, env._last_pw, R) → role of i toward j
  (head-on / give-way / stand-on / overtaking, with the complement rule), the same geometry as the env situation.
  Candidate courses = goal bearing −120…+120° (15° step); cost = VO cost (check_reward_rank.vo_action, verbatim:
  predicted approach < 30 m penalty over H for every seen ship + deviation 30 per 90° + switch penalty 3).
  Per ship, highest rule wins:
    give-way  (any seen ship with dcpa < 24 m, approaching, my role head-on / give-way / overtaking)
              → starboard only and at least 15° (Rule 14/15/16/8: early, substantial, to starboard)
    stand-on act (stand-on toward a risky ship and its tcpa <= RULE_17B_TIME)
              → starboard side only (Rule 17(b)(c): may act, not to port)
    stand-on  (stand-on toward a risky ship, earlier) → rudder 0 = keep course (Rule 17(a))
    give geometry, not risky (my role give-way toward a seen approaching ship, dcpa >= 24 m)
              → no port turns (candidates >= 0), keeps clear without crossing ahead
    otherwise → full VO (all candidates)
  Thrust +1 always. Ships see TRUE position/velocity of ships within R (not radar rays).

★2026-10-02 intent sharing (col300i / vo300i, Fig1 upper bound, author "ㅇㅋ 진행해줘"):
  every ship broadcasts its planned absolute course (goal bearing + chosen candidate; rudder-0 stand-on = current heading).
  With intent_course given, the VO cost predicts each seen ship turning toward its broadcast course with its own
  turn-rate model (same kinematics as the own-candidate rollout) instead of going straight. The broadcast is the one
  from the previous decision (1-step stale, as a real message). Without intent_course the code is unchanged.
★2026-10-02 rule v2 option react_tcpa (col56h / col300h / col300ih): a ship counts as 'risky' for the COLREGs modes
  only when its tcpa <= react_tcpa (150 s = the VO look-ahead H). Ships further out in time no longer trigger the
  give-way / stand-on / no-port constraints (the over-reaction seen in col300). Same value for 56 m and 300 m.
"""
import torch


def build(vg):
    DEG = vg.DEG
    wrap = vg._wrap180
    SIT_NONE, SIT_HEADON, SIT_STANDON, SIT_GIVEWAY, SIT_OVERTAKING = (vg.SIT_NONE, vg.SIT_HEADON, vg.SIT_STANDON,
                                                                      vg.SIT_GIVEWAY, vg.SIT_OVERTAKING)
    BIG = 1e6

    def goal_bearing(env):
        tg = env.goal - env.pos
        return torch.atan2(tg[..., 0], tg[..., 1]) / DEG

    def steer_to(env, course):
        return torch.clamp(wrap(course - env.heading) / 10.0, -1, 1)

    def vo_cost(env, R, prev, D_safe=30.0, H=150.0, dt=4.0, intent_course=None):
        # = check_reward_rank.vo_action up to `cost` (verbatim), candidates always −120…+120
        # intent_course [E,N] (abs deg) or None: others' predicted path turns toward their broadcast course
        dev = env.pos.device
        E, N = env.E, env.N
        cands = torch.arange(-120.0, 121.0, 15.0, device=dev)
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
        if intent_course is not None:
            psio = env.heading.clone()
            po = env.pos.clone()
        for k in range(1, int(H / dt) + 1):
            t = k * dt
            rate = (omega * 0.35 if t <= 5.0 else omega).unsqueeze(-1) * dt
            dpsi = wrap(psi_c - psi)
            psi = psi + torch.clamp(dpsi, -1, 1) * torch.minimum(dpsi.abs(), rate)
            pr = psi * DEG
            p = p + torch.stack([torch.sin(pr), torch.cos(pr)], -1) * (v.unsqueeze(-1).unsqueeze(-1) * dt)
            if intent_course is None:
                pj = env.pos + vel * t
            else:
                rate_o = (omega * 0.35 if t <= 5.0 else omega) * dt
                dpo = wrap(intent_course - psio)
                psio = psio + torch.clamp(dpo, -1, 1) * torch.minimum(dpo.abs(), rate_o)
                po = po + torch.stack([torch.sin(psio * DEG), torch.cos(psio * DEG)], -1) * (env.speed.unsqueeze(-1) * dt)
                pj = po
            d = torch.linalg.norm(pj.unsqueeze(1).unsqueeze(3) - p.unsqueeze(2), dim=-1)
            pen = 100.0 * (torch.clamp(D_safe - d, min=0) / D_safe) ** 2 / (1.0 + t / 30.0)
            pen_max = pen if pen_max is None else torch.maximum(pen_max, pen)
        risk = (pen_max * seen.unsqueeze(-1)).sum(2)
        cost = risk + cands.abs().view(1, 1, -1) / 90.0 * 30.0 + (cands.view(1, 1, -1) != prev.unsqueeze(-1)).float() * 3.0
        return cands, cost, gb

    def colregs_action(env, R, prev, H=150.0, dt=4.0, intent_course=None, react_tcpa=None):
        """(a0 [E,N], new prev [E,N], mode [E,N] long: 0 VO · 1 give-way · 2 stand-on keep · 3 stand-on act · 4 give geometry,
        broadcast course [E,N] abs deg)."""
        pw = env._last_pw
        N = env.N
        eye = torch.eye(N, dtype=torch.bool, device=env.pos.device).unsqueeze(0)
        role_i, _ = vg.RolePromiseTrackerV2.roles(env, pw, R)                      # [E,N,N] my role toward j
        near = (pw['dist'] <= R) & (pw['raw_tcpa'] >= 0) & ~eye
        risky = near & (pw['dcpa'] < vg.DCPA_RISK)
        if react_tcpa is not None:
            near = near & (pw['tcpa'] <= react_tcpa)
            risky = risky & (pw['tcpa'] <= react_tcpa)
        giver = (role_i == SIT_HEADON) | (role_i == SIT_GIVEWAY) | (role_i == SIT_OVERTAKING)
        give = (risky & giver).any(-1)
        give_geom = (near & giver).any(-1)
        st = risky & (role_i == SIT_STANDON)
        stand = st.any(-1)
        tc = torch.where(st, pw['tcpa'], torch.full_like(pw['tcpa'], BIG)).min(-1).values
        stand_act = stand & (tc <= vg.RULE_17B_TIME)

        cands, cost, gb = vo_cost(env, R, prev, H=H, dt=dt, intent_course=intent_course)
        c = cands.view(1, 1, -1)
        allow_give = (c >= 15.0)                     # starboard, substantial
        allow_stbd = (c >= 0.0)                      # no port turn
        mode = torch.zeros_like(prev, dtype=torch.long)
        mode = torch.where(give_geom, torch.full_like(mode, 4), mode)
        mode = torch.where(stand & ~give, torch.full_like(mode, 2), mode)
        mode = torch.where(stand_act & ~give, torch.full_like(mode, 3), mode)
        mode = torch.where(give, torch.full_like(mode, 1), mode)
        m = mode.unsqueeze(-1)
        allowed = torch.where(m == 1, allow_give, torch.where((m == 3) | (m == 4), allow_stbd, torch.ones_like(allow_give)))
        choice = cands[torch.where(allowed, cost, torch.full_like(cost, BIG)).argmin(-1)]
        a0 = steer_to(env, gb + choice)
        a0 = torch.where(mode == 2, torch.zeros_like(a0), a0)                     # stand-on: rudder 0, keep course
        new_prev = torch.where(mode == 2, prev, choice)
        course = torch.where(mode == 2, env.heading, gb + choice)
        return a0, new_prev, mode, course

    def vo_intent_action(env, R, prev, H=150.0, dt=4.0, intent_course=None):
        """Plain VO (all candidates) with intent-aware prediction of others. (a0, choice, broadcast course)."""
        cands, cost, gb = vo_cost(env, R, prev, H=H, dt=dt, intent_course=intent_course)
        choice = cands[cost.argmin(-1)]
        return steer_to(env, gb + choice), choice, gb + choice

    colregs_action.vo_intent = vo_intent_action
    return colregs_action
