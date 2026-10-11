"""eval_scripted.py — scripted ships through the learned-ship evaluator (2026-10-01).

Question (author, 2026-10-01): does communication information help *before* any RL?
Runs rule-based ships (no network) through eval/eval_ckpt.py unchanged, so every report
metric (fuel, headTravel, COLREGs C, passing distance, timeout, wall, role judge, epReward)
comes from the same code as the learned-ship evaluation of the t_ batch.

How: eval_ckpt.main() is called as is. Two ckpt_io functions are wrapped before the call:
  - make_env_from_snapshot: keeps a handle on the env and resets the VO hysteresis on done.
  - restore_policy: keeps the snapshot (env settings of the t_ batch) but swaps .policy for a
    stub whose ctr_actor returns the scripted action. The checkpoint weights are never used.
No repo file is modified. The checkpoint given with --ckpt only carries the env snapshot.

Policies (VO = check_reward_rank.vo_action, copied verbatim; only R/H/dt differ):
  goal      steer to goal, full speed, no avoidance (reference floor)
  vo56      sees ships <= 56 m (radar range), looks ahead H 60 s, dt 2 s   (= rank-check vo56)
  vo56h150  sees ships <= 56 m, looks ahead H 150 s, dt 4 s               (only range differs from vo300)
  vo300     sees ships <= 300 m (comm range), H 150 s, dt 4 s            (= rank-check vo300)
  col56 / col300  COLREGs rule + VO (colregs_rule.py), H 150 s, dt 4 s; only the seen range differs (2026-10-01 2nd round)
VO reads true positions/velocities of the ships it sees (not radar rays).
Extra lines (2026-10-01): [scripted-rudder] actual rudder travel / mean |rudder| per episode (started after burn-in),
[scripted-mode] share of COLREGs-rule modes (col only).

Usage (Windows, batch env exported, see _run_scripted.sh):
  python eval_scripted.py --py_root C:/work/DT_Vessel_v3/Python --policy vo56 -- \
      --ckpt <t_off_s43.pt> --arm OFF --envs 256 --eval_decisions 10000 --burnin 2400 --seed 999
"""
import argparse
import os
import sys

POLICIES = {
    'goal': None,
    'vo56': dict(kind='vo', R=56.0, H=60.0, dt=2.0),
    'vo56h150': dict(kind='vo', R=56.0, H=150.0, dt=4.0),
    'vo300': dict(kind='vo', R=300.0, H=150.0, dt=4.0),
    # ★2026-10-01 COLREGs 규칙 배(colregs_rule.py): 같은 규칙, 보는 거리만 다름
    'col56': dict(kind='col', R=56.0, H=150.0, dt=4.0),
    'col300': dict(kind='col', R=300.0, H=150.0, dt=4.0),
    # ★2026-10-02 Fig1 상한: 의도 공유(각 배가 직전 결정의 계획 침로를 방송 → 상대 경로를 그 침로로 예측). 보는 거리 300 m
    'col300i': dict(kind='col', R=300.0, H=150.0, dt=4.0, intent=True),
    'vo300i': dict(kind='voi', R=300.0, H=150.0, dt=4.0, intent=True),
    # ★2026-10-02 규칙 v2: 충돌까지 150 s(= H) 넘게 남은 배에는 COLREGs 동작 안 함(과잉 반응 수정). 56·300 같은 값
    'col56h': dict(kind='col', R=56.0, H=150.0, dt=4.0, react_tcpa=150.0),
    'col300h': dict(kind='col', R=300.0, H=150.0, dt=4.0, react_tcpa=150.0),
    'col300ih': dict(kind='col', R=300.0, H=150.0, dt=4.0, intent=True, react_tcpa=150.0),
}
MODE_NAMES = ('VO', 'give-way', 'stand-on keep', 'stand-on act', 'give geometry')
LEARNED = 'learned'   # ★2026-10-02: 정책을 바꾸지 않고(체크포인트 신경망 그대로) Woerner COLREGs·타 지표만 덧붙여 재는 모드


def build(vg, torch):
    """Return (act_fn(env, prev) -> (a0, prev)) helpers bound to vessel_gym."""
    DEG = vg.DEG
    wrap = vg._wrap180

    def goal_bearing(env):
        tg = env.goal - env.pos
        return torch.atan2(tg[..., 0], tg[..., 1]) / DEG

    def goal_a0(env):
        return torch.clamp(wrap(goal_bearing(env) - env.heading) / 20.0, -1, 1)

    def steer_to(env, course):
        return torch.clamp(wrap(course - env.heading) / 10.0, -1, 1)

    def vo_action(env, R, prev, stbd_only, D_safe=30.0, H=60.0, dt=2.0):
        # verbatim copy of verify/check_reward_rank.py vo_action (feat/reward-v3 3aa0636); dev taken from env
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

    return goal_a0, vo_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--py_root', required=True, help='feat/reward-v3 Python dir (has eval/eval_ckpt.py)')
    ap.add_argument('--policy', required=True, choices=sorted(POLICIES) + [LEARNED])
    ap.add_argument('rest', nargs=argparse.REMAINDER, help='-- then eval_ckpt.py args')
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ['--'] else a.rest
    if a.policy != LEARNED and '--arm' in rest and rest[rest.index('--arm') + 1] != 'OFF':
        raise SystemExit('[scripted] --arm must be OFF (scripted ships send no messages)')

    sys.path.insert(0, a.py_root)
    sys.path.insert(0, os.path.join(a.py_root, 'eval'))
    import torch
    import vessel_gym as vg
    import ckpt_io
    import eval_ckpt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import colregs_rule
    import colregs_woerner

    goal_a0, vo_action = build(vg, torch)
    colregs_action = colregs_rule.build(vg)
    WoernerTracker = colregs_woerner.build(vg)
    spec = POLICIES.get(a.policy)
    st: dict = {'env': None, 'prev': None, 'calls': 0, 't': 0}
    burnin = int(rest[rest.index('--burnin') + 1]) if '--burnin' in rest else 1200   # eval_ckpt 기본값과 같음
    acc = {k: 0.0 for k in ('g_n', 'g_trav', 'g_abs', 'g_dec', 'a_n', 'a_trav', 'a_abs', 'a_dec')}
    mode_cnt = [0.0] * len(MODE_NAMES)

    _make = ckpt_io.make_env_from_snapshot

    def make_env(*args, **kw):
        env = _make(*args, **kw)
        st['env'] = env
        z = torch.zeros(env.E, env.N, device=env.pos.device, dtype=env.dtype)
        st['prev'] = z.clone()
        # ★2026-10-01 타 점검: 배마다 에피소드 동안 실제 타각 이동량 Σ|Δδ| 와 Σ|δ| (burn-in 뒤 시작한 에피소드만)
        st['trav'], st['abs'], st['n'] = z.clone(), z.clone(), z.clone()
        st['ep0'] = torch.zeros(env.E, env.N, device=env.pos.device, dtype=torch.long)
        st['woer'] = WoernerTracker(env.E, env.N, env.pos.device, burnin)
        st['course'] = None                               # 방송된 계획 침로(절대 deg), 첫 결정에서 현재 침로로 초기화
        _step = env.step

        def step(actions):
            r0 = env.rudder.clone()
            st['woer'].observe(env, st['t'])              # 결정 시점 상태 s_t 로 조우 갱신·시작
            obs, r, done, oc = _step(actions)
            st['woer'].after_step(done, oc)               # 끝난 배의 조우 마감(충돌 = 최근접 상대)
            if st['course'] is not None:                  # 재스폰한 배의 방송 침로 = 새 침로
                st['course'] = torch.where(done, env.heading, st['course'])
            st['prev'] = torch.where(done, torch.zeros_like(st['prev']), st['prev'])   # same as rank check
            live = ~done                                  # 끝난 배는 이 스텝 안에서 재스폰(타각 0) → 마지막 증분은 뺌
            st['trav'] = st['trav'] + torch.where(live, (env.rudder - r0).abs(), torch.zeros_like(r0))
            st['abs'] = st['abs'] + torch.where(live, env.rudder.abs(), torch.zeros_like(r0))
            st['n'] = st['n'] + 1.0
            fin = done & (st['ep0'] >= burnin)
            g = fin & (oc == vg.OUT_GOAL)
            acc['g_n'] += float(g.sum()); acc['g_trav'] += float(st['trav'][g].sum())
            acc['g_abs'] += float(st['abs'][g].sum()); acc['g_dec'] += float(st['n'][g].sum())
            acc['a_n'] += float(fin.sum()); acc['a_trav'] += float(st['trav'][fin].sum())
            acc['a_abs'] += float(st['abs'][fin].sum()); acc['a_dec'] += float(st['n'][fin].sum())
            zz = torch.zeros_like(st['trav'])
            st['trav'] = torch.where(done, zz, st['trav']); st['abs'] = torch.where(done, zz, st['abs'])
            st['n'] = torch.where(done, zz, st['n'])
            st['t'] += 1
            st['ep0'] = torch.where(done, torch.full_like(st['ep0'], st['t']), st['ep0'])
            return obs, r, done, oc
        env.step = step
        return env

    class ScriptedActor:
        def __call__(self, x, goal, self_s, om, sit):
            env = st['env']
            st['calls'] += 1
            if spec is None:
                a0 = goal_a0(env)
            elif spec['kind'] in ('col', 'voi'):
                if st['course'] is None:
                    st['course'] = env.heading.clone()
                ic = st['course'] if spec.get('intent') else None
                if spec['kind'] == 'col':
                    a0, st['prev'], mode, st['course'] = colregs_action(env, spec['R'], st['prev'], H=spec['H'], dt=spec['dt'],
                                                                         intent_course=ic, react_tcpa=spec.get('react_tcpa'))
                else:
                    a0, st['prev'], st['course'] = colregs_action.vo_intent(env, spec['R'], st['prev'], H=spec['H'],
                                                                             dt=spec['dt'], intent_course=ic)
                    mode = None
                if mode is not None and st['t'] >= burnin:
                    bc = torch.bincount(mode.reshape(-1), minlength=len(MODE_NAMES))
                    for i in range(len(MODE_NAMES)):
                        mode_cnt[i] += float(bc[i])
            else:
                a0, st['prev'] = vo_action(env, spec['R'], st['prev'], False, H=spec['H'], dt=spec['dt'])
            return torch.stack([a0, torch.ones_like(a0)], -1), None, None, None

    class ScriptedPolicy:
        ctr_actor = ScriptedActor()

    _restore = ckpt_io.restore_policy

    def restore(*args, **kw):
        r = _restore(*args, **kw)
        if a.policy != LEARNED:
            r.policy = ScriptedPolicy()
        return r

    ckpt_io.make_env_from_snapshot = make_env
    ckpt_io.restore_policy = restore
    tag = f"[scripted] policy={a.policy} " + ('learned network from --ckpt (unchanged), extra metric lines only' if a.policy == LEARNED else
                                               'goal steer, no avoidance' if spec is None else
                                               f"{'COLREGs rule + VO' if spec['kind'] == 'col' else 'VO'}"
                                               f"{' + intent sharing (others predicted on their broadcast course)' if spec.get('intent') else ''}"
                                               f"{' + react only tcpa<=%gs' % spec['react_tcpa'] if spec.get('react_tcpa') else ''} R={spec['R']:g}m "
                                               f"H={spec['H']:g}s dt={spec['dt']:g}s (true pos/vel of seen ships)")
    print(tag + (' | network from --ckpt IS used' if a.policy == LEARNED else
                 ' | network weights NOT used; --ckpt only supplies the env snapshot'), flush=True)
    sys.argv = ['eval_ckpt.py'] + rest
    eval_ckpt.main()
    def _d(x, n):
        return x / n if n else float('nan')
    print(f"   [scripted-rudder] goal-ep n={acc['g_n']:.0f} rudderTravel={_d(acc['g_trav'], acc['g_n']):.1f}deg/ep "
          f"meanAbsRudder={_d(acc['g_abs'], acc['g_dec']):.2f}deg | all-ep n={acc['a_n']:.0f} "
          f"rudderTravel={_d(acc['a_trav'], acc['a_n']):.1f}deg/ep meanAbsRudder={_d(acc['a_abs'], acc['a_dec']):.2f}deg "
          f"(actual rudder, episodes started after burn-in {burnin})", flush=True)
    print(st['woer'].report(), flush=True)
    if sum(mode_cnt) > 0:
        tot = sum(mode_cnt)
        print("   [scripted-mode] " + ' '.join(f"{nm}={100.0 * c / tot:.1f}%" for nm, c in zip(MODE_NAMES, mode_cnt))
              + f" (ship-decisions after burn-in, n={tot:.0f})", flush=True)
    print(f"{tag} | scripted actor calls={st['calls']}" + ('' if a.policy == LEARNED else
          " (= every decision incl. burn-in; network never called)"), flush=True)


if __name__ == '__main__':
    main()
