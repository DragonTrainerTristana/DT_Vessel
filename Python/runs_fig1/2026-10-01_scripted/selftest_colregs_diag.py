"""selftest_colregs_diag.py — ★2026-11-09 check for colregs_woerner diag (CPU, no vessel_gym, < 1 min).

1) diag=False and diag=True give the same Woerner / binary report (existing lines unchanged).
2) A give-way ship that turns 15° to port toward its goal is classed 'port > 10° toward goal'.
Run: python selftest_colregs_diag.py
"""
import math
import types

import torch

import colregs_woerner

SIT = dict(SIT_HEADON=1, SIT_STANDON=2, SIT_GIVEWAY=3, SIT_OVERTAKING=4)


def fake_vg():
    role = torch.zeros(1, 3, 3, dtype=torch.long)
    role[0, 0, 1], role[0, 1, 0] = SIT['SIT_GIVEWAY'], SIT['SIT_STANDON']
    return types.SimpleNamespace(**SIT, DEG=math.pi / 180.0, DCPA_RISK=24.0, RULE_17B_TIME=15.89, OUT_COLLISION_VESSEL=2,
                                 _wrap180=lambda d: (d + 180.0) % 360.0 - 180.0,
                                 RolePromiseTrackerV2=types.SimpleNamespace(roles=lambda env, pw, r: (role, None)))


def run(diag):
    vg = fake_vg()
    tr = colregs_woerner.build(vg)(1, 3, 'cpu', burnin=0, diag=diag)
    deg = math.pi / 180.0
    pos = torch.tensor([[[0.0, 0.0], [0.0, 200.0], [1000.0, 1000.0]]])
    goal = pos.clone()
    goal[0, 0] = torch.tensor([500 * math.sin(-30 * deg), 500 * math.cos(-30 * deg)])   # ship 0's goal 30° to port
    env = types.SimpleNamespace(pos=pos, goal=goal, speed=torch.ones(1, 3))
    dists = [200, 150, 100, 60, 40, 50, 80]
    for t, dv in enumerate(dists):
        dist = torch.full((1, 3, 3), 1000.0)
        dist[0, 0, 1] = dist[0, 1, 0] = float(dv)
        raw = torch.full((1, 3, 3), 10.0 if t < 5 else -10.0)
        env._last_pw = {'dist': dist, 'raw_tcpa': raw, 'tcpa': raw.clamp(min=0), 'dcpa': torch.full((1, 3, 3), 10.0)}
        env.heading = torch.tensor([[0.0 if t < 2 else -15.0, 180.0, 90.0]])
        tr.observe(env, t)
        tr.after_step(torch.zeros(1, 3, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.long))
    return tr


a, b = run(False), run(True)
assert a.report() == b.report(), 'diag changed the existing report'
d = b.dacc
assert d['gw_n'] == 1 and d['gw_fail'] == 1 and d['gw_port_goal'] == 1 and d['gw_port_other'] == 0, d
assert d['so_n'] == 1 and d['so_fail'] == 0, d
assert b.jit['all'][0] > 0 and abs(b.jit['all'][1] - 15.0) < 1e-4, b.jit
print(b.report())
print(b.diag_report())
print('selftest OK')
