"""vo_teacher.py — scripted VO teacher for imitation start (2026-10-02).

Verbatim copy of `verify/check_reward_rank.py` goal_bearing / goal_a0 / steer_to / vo_action
(feat/reward-v3 3aa0636, line 134). Only change: the device is taken from env instead of a closure.
The same copy is in runs/2026-10-01_scripted/eval_scripted.py (rule-ship eval, 2026-10-01).

vo56 = vo_action(env, R=56, prev, False, H=60, dt=2): sees every ship within 56 m (radar range)
using its TRUE position and velocity, predicts 60 s ahead for 17 candidate courses
(goal bearing -120..+120 deg, 15 deg step), and picks the lowest of
risk + deviation cost (30 per 90 deg) + switch penalty 3. Thrust is always +1.
"""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import torch  # noqa: E402
import vessel_gym as vg  # noqa: E402

DEG = vg.DEG
wrap = vg._wrap180

# teacher used for the imitation start (spec 2026-10-02 §2) — fixed before results
TEACHER: dict = dict(name='vo56', R=56.0, H=60.0, dt=2.0, stbd_only=False)


def goal_bearing(env):
    tg = env.goal - env.pos
    return torch.atan2(tg[..., 0], tg[..., 1]) / DEG


def goal_a0(env):
    return torch.clamp(wrap(goal_bearing(env) - env.heading) / 20.0, -1, 1)


def steer_to(env, course):
    # 절대 침로 course[deg] 로 트는 a0 (VO 와 같은 이득: 10° 오차 = 전타)
    return torch.clamp(wrap(course - env.heading) / 10.0, -1, 1)


def vo_action(env, R, prev, stbd_only, D_safe=30.0, H=60.0, dt=2.0):
    dev = env.pos.device
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


def teacher_action(env, prev):
    """vo56 teacher: (a [E,N,2] in [-1,1], new prev). a = (vo56 rudder, +1 thrust)."""
    a0, prev = vo_action(env, TEACHER['R'], prev, TEACHER['stbd_only'], H=TEACHER['H'], dt=TEACHER['dt'])
    return torch.stack([a0, torch.ones_like(a0)], -1), prev


# ★2026-10-02 Fig1 (spec docs/superpowers/specs/2026-10-02-fig1-latent-design.md §2): teachers in 'course' action units
#   and intent sharing. vo_cost_intent = vo_action's cost with an optional intent-aware prediction of the other ships
#   (same code as runs/2026-10-01_scripted/colregs_rule.py vo_cost, rule-ship round 4): each seen ship turns toward its
#   broadcast course with its own turn-rate model instead of going straight. intent_course=None = straight-line
#   prediction (then the cost equals vo_action's cost; candidates always -120..+120).
TEACHERS = {
    'vo56': dict(R=56.0, H=60.0, dt=2.0, intent=False),
    'vo56h150': dict(R=56.0, H=150.0, dt=4.0, intent=False),
    'vo300i': dict(R=300.0, H=150.0, dt=4.0, intent=True),
}


def vo_cost_intent(env, R, prev, D_safe=30.0, H=150.0, dt=4.0, intent_course=None):
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


class Teacher:
    """Stateful teacher for one env (prev-choice hysteresis per ship; for intent teachers the broadcast courses).

    act(env) -> (a [E,N,2] in the env's action units, choice [E,N] deg). In 'course' mode a0 = choice / COURSE_ACT_DEG
    (the student's action space); in 'rudder' mode a0 = steer_to(goal bearing + choice) as in vo_action.
    after_step(env, a_exec, done): records every ship's broadcast course for the next decision (1-step stale message):
    course = goal bearing (pre-step) + a_exec0 * COURSE_ACT_DEG in 'course' mode, and the respawn heading for done ships.
    """

    def __init__(self, name, env):
        if name not in TEACHERS:
            raise ValueError(f"teacher {name!r} not in {sorted(TEACHERS)}")
        self.name, self.spec = name, TEACHERS[name]
        self.prev = torch.zeros(env.E, env.N, device=env.pos.device, dtype=env.dtype)
        self.course = None
        self._gb = None

    def act(self, env):
        s = self.spec
        if self.course is None:
            self.course = env.heading.clone()
        ic = self.course if s['intent'] else None
        if s['intent'] or s['R'] != 56.0 or s['H'] != 60.0:
            cands, cost, gb = vo_cost_intent(env, s['R'], self.prev, H=s['H'], dt=s['dt'], intent_course=ic)
            choice = cands[cost.argmin(-1)]
        else:
            _, choice = vo_action(env, s['R'], self.prev, False, H=s['H'], dt=s['dt'])
            gb = goal_bearing(env)
        self.prev = choice
        self._gb = gb
        if vg.ACTION_MODE == 'course':
            a0 = (choice / vg.COURSE_ACT_DEG).clamp(-1, 1)
        else:
            a0 = steer_to(env, gb + choice)
        return torch.stack([a0, torch.ones_like(a0)], -1), choice

    def after_step(self, env, a_exec, done):
        gb = self._gb if self._gb is not None else goal_bearing(env)
        if vg.ACTION_MODE == 'course':
            course = gb + a_exec[..., 0].clamp(-1, 1) * vg.COURSE_ACT_DEG
        else:
            course = env.heading.clone()      # rudder mode: no intent available -> current heading
        self.course = torch.where(done, env.heading, course)
        self.prev = torch.where(done, torch.zeros_like(self.prev), self.prev)
