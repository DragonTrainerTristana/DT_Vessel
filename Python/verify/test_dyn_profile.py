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


TESTS = [test_agile_dict_equals_legacy_literals, test_imo_dict_numbers, test_unknown_profile_raises,
         test_defaults_are_agile_grid]


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
