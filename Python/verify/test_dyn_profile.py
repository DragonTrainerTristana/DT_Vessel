"""
VESSEL_DYN_PROFILE 프로필 테스트 (2026-09-21). pytest 없이도 `python3 verify/test_dyn_profile.py` 로 돈다.
numpy 금지(Mac torch↔numpy 비호환).
"""
import math
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import torch  # noqa: E402

import config as cfg  # noqa: E402
import vessel_gym as vg  # noqa: E402
import ckpt_io  # noqa: E402

L = 14.18316


def test_agile_dict_equals_legacy_literals():
    d = cfg.dyn_profile_constants('agile')
    assert d == {
        'formula': 'ratio', 'turn_factor': 1.5, 'r_full': None, 'max_yaw_rate': 45.0, 'rudder_rate': 12.0,
        'accel': 0.1, 'decel': 0.04, 'drag_coef': 0.1,
        'tcpa_risk_denom': 30.0, 'rule_17b_time': 7.0, 'rule_17c_time': 3.5,
        'rule_17b_dist': 18.0, 'rule_17c_dist': 9.0,
        'early_action_time': 21.5, 'substantial_action_time': 11.5,
        'goal_reached': 3.0, 'cmd_mismatch_slack_deg': 0.0,
    }


def test_imo_dict_numbers():
    d = cfg.dyn_profile_constants('imo')
    assert d['formula'] == 'abs' and d['turn_factor'] is None
    assert abs(d['r_full'] - 2.0 * L) < 1e-6
    assert abs(d['max_yaw_rate'] - 1.8 / (2.0 * L) / (math.pi / 180.0)) < 1e-9   # ≈ 3.635 deg/s
    assert d['rudder_rate'] == 3.0 and d['accel'] == 0.01 and d['decel'] == 0.004 and d['drag_coef'] == 0.005
    assert abs(d['tcpa_risk_denom'] - 30.0 * 2.27) < 1e-9
    assert abs(d['rule_17b_time'] - 7.0 * 2.27) < 1e-9 and abs(d['rule_17c_time'] - 3.5 * 2.27) < 1e-9
    assert abs(d['rule_17b_dist'] - 18.0 * 2.27) < 1e-9 and abs(d['rule_17c_dist'] - 9.0 * 2.27) < 1e-9
    assert abs(d['early_action_time'] - 21.5 * 2.27) < 1e-9 and abs(d['substantial_action_time'] - 11.5 * 2.27) < 1e-9
    assert abs(d['goal_reached'] - L / 2.0) < 1e-9
    assert abs(d['cmd_mismatch_slack_deg'] - 1.2) < 1e-9


def test_unknown_profile_raises():
    try:
        cfg.dyn_profile_constants('boat')
    except ValueError:
        return
    raise AssertionError('ValueError 기대')


def test_defaults_are_agile_grid():
    # 이 테스트는 env 없이 돌릴 때만 의미 있음(run_repro 는 common_env 로 명시 export)
    if os.environ.get('VESSEL_DYN_PROFILE') is None:
        assert cfg.DYN_PROFILE == 'agile' and cfg.DYN == cfg.dyn_profile_constants('agile')
    if os.environ.get('VESSEL_OBSTACLES') is None:
        assert cfg.OBSTACLES_MODE == 'grid3x3'
    assert 'VESSEL_DYN_PROFILE' in cfg.YUGIOH and cfg.YUGIOH['VESSEL_DYN_PROFILE'] == 'agile'
    assert 'VESSEL_OBSTACLES' in cfg.YUGIOH and cfg.YUGIOH['VESSEL_OBSTACLES'] == 'grid3x3'


class _profile:
    """모듈 전역을 profile 로 바꿨다가 블록을 나가면 원래대로. 테스트 간 오염 방지."""
    def __init__(self, profile):
        self.profile = profile
    def __enter__(self):
        self.saved, self.saved_p = vg.current_dyn_constants(), vg.DYN_PROFILE
        vg.apply_dyn_constants(cfg.dyn_profile_constants(self.profile), self.profile)
    def __exit__(self, *a):
        vg.apply_dyn_constants(self.saved, self.saved_p)


def _env11(**kw):
    env = vg.VesselBatchEnv(num_envs=1, n_vessels=1, device='cpu', crossing=2, reward_range=300.0)
    env.pos.zero_(); env.heading.zero_(); env.goal.fill_(1e6)
    env.speed.fill_(kw.get('speed', 1.0)); env.max_speed.fill_(kw.get('max_speed', 1.0))
    env.rudder.fill_(kw.get('rudder', 0.0)); env.cmd_rudder.fill_(kw.get('rudder', 0.0))
    env.target_speed.fill_(kw.get('target', kw.get('max_speed', 1.0)))
    return env


def _steady_turn_radius(profile, vmax):
    with _profile(profile):
        env = _env11(speed=vmax, max_speed=vmax, rudder=30.0, target=vmax)
        h0 = float(env.heading[0, 0])
        n = 150 * vg.SUBSTEPS
        for _ in range(n):
            env._substep()
        omega = (float(env.heading[0, 0]) - h0) / (n * vg.DT)          # deg/s
        return float(env.speed[0, 0]) / (omega * math.pi / 180.0)


def test_imo_turn_radius_fixed_across_fleet():
    for vmax in (0.8, 1.0, 1.8):
        R = _steady_turn_radius('imo', vmax)
        assert abs(R - 2.0 * L) / (2.0 * L) < 0.01, (vmax, R)


def test_agile_turn_radius_unchanged():
    R = _steady_turn_radius('agile', 1.0)
    assert abs(R - 1.2732) / 1.2732 < 0.01, R


def _stop_distance(profile, v0=1.0):
    with _profile(profile):
        env = _env11(speed=v0, max_speed=1.0, rudder=0.0, target=0.0)
        for _ in range(600 * vg.SUBSTEPS):           # 최대 240 s
            env._substep()
            if float(env.speed[0, 0]) < 0.005:
                break
        return float(env.pos[0, 0, 1])               # heading 0 = +Z 전진


def test_stop_distance():
    assert abs(_stop_distance('agile') - 5.0) < 0.5
    d = _stop_distance('imo')
    assert abs(d - 70.2) / 70.2 < 0.05, d


def test_yaw_helper_matches_legacy_formula():
    with _profile('agile'):
        rud = torch.tensor([[30.0]]); spd = torch.tensor([[0.5]]); vmax = torch.tensor([[1.0]])
        y = vg.yaw_rate_deg(rud, spd, vmax)
        assert torch.equal(y, rud * (spd / torch.clamp(vmax, min=1e-6)) * 1.5)   # 옛 식과 비트동일
    with _profile('imo'):
        y = vg.yaw_rate_deg(torch.tensor([[30.0]]), torch.tensor([[1.0]]), torch.tensor([[1.0]]))
        assert abs(float(y) - 1.0 / (2.0 * L) / (math.pi / 180.0)) < 1e-6          # ≈ 2.02 deg/s


def test_obs_yaw_norm_bounded():
    for p in ('agile', 'imo'):
        with _profile(p):
            env = _env11(speed=1.8, max_speed=1.8, rudder=30.0, target=1.8)
            obs = env._build_obs()
            assert -1.0 - 1e-6 <= float(obs[0, 0, 363]) <= 1.0 + 1e-6, (p, float(obs[0, 0, 363]))


def test_obstacles_none():
    saved = vg.OBSTACLES_MODE
    vg.OBSTACLES_MODE = 'none'
    try:
        env = vg.VesselBatchEnv(num_envs=2, n_vessels=4, device='cpu', crossing=2, reward_range=300.0)
        assert tuple(env.obstacles.shape) == (0, 2)
        obs = env.reset()
        for _ in range(5):
            obs, r, done, oc = env.step(torch.zeros(2, 4, 2))
        assert torch.isfinite(obs).all() and torch.isfinite(r).all()
        assert not (oc == vg.OUT_COLLISION_OBSTACLE).any()
    finally:
        vg.OBSTACLES_MODE = saved


def test_obstacles_grid_default():
    env = vg.VesselBatchEnv(num_envs=1, n_vessels=2, device='cpu', crossing=2, reward_range=300.0)
    assert tuple(env.obstacles.shape) == (9, 2)


def test_snapshot_records_profile():
    s = ckpt_io.snapshot_config(arm='OFF', msg_dim=6, seed=1, n_envs=2, n_vessels=4, max_partners=4, trunc_boot=False,
                                comm_on_at=0, ring=1.0, crossing=2, rollout=8, trainer='gym')
    assert s['dyn_profile'] == cfg.DYN_PROFILE and s['dyn'] == vg.current_dyn_constants()
    assert s['obstacles'] == cfg.OBSTACLES_MODE
    for k in ('radar_dropout_p', 'radar_dropout_len', 'los_gate', 'max_episode_steps'):
        assert k in s, k


def test_apply_sim_snapshot_legacy_and_mismatch():
    if cfg.DYN_PROFILE != 'agile' or cfg.OBSTACLES_MODE != 'grid3x3':
        return
    saved, saved_p, saved_ob = vg.current_dyn_constants(), vg.DYN_PROFILE, vg.OBSTACLES_MODE
    try:
        # (a) 키 없는 구 스냅샷 = legacy agile/grid3x3 → 현재 기본과 일치, 중단 없음
        notes = []
        eff = ckpt_io.apply_sim_snapshot({}, notes=notes, tag='[t]')
        assert eff['dyn_profile'] == 'agile' and eff['obstacles'] == 'grid3x3' and any('legacy' in n for n in notes)
        # (b) imo 스냅샷 vs agile config → 기본 중단
        snap = {'dyn_profile': 'imo', 'dyn': cfg.dyn_profile_constants('imo'), 'obstacles': 'none', 'radar_range': 56.0}
        try:
            ckpt_io.apply_sim_snapshot(snap, tag='[t]')
        except SystemExit:
            pass
        else:
            raise AssertionError('불일치인데 중단 안 함')
        # (c) allow 면 스냅샷 값이 vessel_gym 에 적용됨
        notes = []
        eff = ckpt_io.apply_sim_snapshot(snap, allow_sim_mismatch=True, notes=notes, tag='[t]')
        assert eff['dyn_profile'] == 'imo' and vg.DYN_FORMULA == 'abs' and abs(vg.R_FULL - 2 * L) < 1e-6
        assert vg.OBSTACLES_MODE == 'none' and any('allow_sim_mismatch' in n for n in notes)
        # (d) radar_range 불일치도 잡힘
        try:
            ckpt_io.apply_sim_snapshot({'radar_range': 28.0}, tag='[t]')
        except SystemExit:
            pass
        else:
            raise AssertionError('radar_range 불일치인데 중단 안 함')
    finally:
        vg.apply_dyn_constants(saved, saved_p); vg.OBSTACLES_MODE = saved_ob


def test_check_branch_rejects_profile_mismatch(tmp=None):
    import subprocess, tempfile
    d = tempfile.mkdtemp()
    base = {'arm': 'OFF', 'seed': 1, 'msg_dim': 6, 'branch_from': 'trunk_d6_s1.pt', 'branch_from_sha256': 'ab' * 32, 'branch_at': 100}
    torch.save({'cfg_snapshot': dict(base, dyn_profile='agile', obstacles='grid3x3'), 'steps': 200}, os.path.join(d, 'off_s1.pt'))
    torch.save({'cfg_snapshot': dict(base, arm='ON', dyn_profile='imo', obstacles='grid3x3'), 'steps': 200}, os.path.join(d, 'on6_s1.pt'))
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, 'check_branch.py'), os.path.join(d, 'off_s1.pt'), os.path.join(d, 'on6_s1.pt')],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    assert r.returncode != 0 and 'dyn' in (r.stdout + r.stderr), (r.returncode, r.stdout[-600:])


TESTS = [test_agile_dict_equals_legacy_literals, test_imo_dict_numbers, test_unknown_profile_raises,
         test_defaults_are_agile_grid, test_imo_turn_radius_fixed_across_fleet, test_agile_turn_radius_unchanged,
         test_stop_distance, test_yaw_helper_matches_legacy_formula, test_obs_yaw_norm_bounded,
         test_obstacles_none, test_obstacles_grid_default,
         test_snapshot_records_profile, test_apply_sim_snapshot_legacy_and_mismatch,
         test_check_branch_rejects_profile_mismatch]


def main():
    fails = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")
    print('VERDICT:', 'ALL PASS' if fails == 0 else f'{fails} FAIL')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
