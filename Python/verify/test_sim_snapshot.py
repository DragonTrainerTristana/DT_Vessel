"""
스냅샷 sim 상수(보상 계수·게이트·COLREGS_MODE·에피소드 길이) 기록·대조·복원 테스트 (2026-09-23).
pytest 없이도 `python3 verify/test_sim_snapshot.py` 로 돈다. numpy 금지(Mac torch↔numpy 비호환).

왜 — 여태 스냅샷은 dyn_profile/obstacles/radar_range 만 대조했고 나머지 24개 상수는 기록도 안 했다.
     다른 보상·게이트로 평가·재개해도 조용히 돌아갔음 ("스냅샷이 유일한 근거" 위반).
"""
import io
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402,F401  (import 순서 확인용 — 아래 모듈들이 어차피 쓴다)

import config as cfg  # noqa: E402
import vessel_gym as vg  # noqa: E402
import ckpt_io  # noqa: E402
from test_dyn_profile import _Skip, _check_branch_on  # noqa: E402  (러너 규약·체크포인트 헬퍼 재사용)


class _sim_consts:
    """apply_sim_snapshot 이 건드리는 config·vessel_gym 상수를 블록 밖에서 원상복구. 테스트 간 오염 방지."""
    def __enter__(self):
        self.cfg_saved = cfg.sim_constants()
        self.vg_saved = {k: getattr(vg, k) for k in cfg.SIM_SNAPSHOT_KEYS if hasattr(vg, k)}
        self.dyn, self.prof = vg.current_dyn_constants(), vg.DYN_PROFILE
        self.obst = vg.OBSTACLES_MODE
        return self

    def __exit__(self, *a):
        for k, v in self.cfg_saved.items():
            setattr(cfg, k, v)
        for k, v in self.vg_saved.items():
            setattr(vg, k, v)
        vg.apply_dyn_constants(self.dyn, self.prof)
        vg.OBSTACLES_MODE = self.obst


def _snap(**extra):
    """현재 env 와 *일치하는* sim 스냅샷 — extra 로 준 것만 달라진다(agile/grid3x3 이든 imo/none 이든 돈다)."""
    s = {'dyn_profile': cfg.DYN_PROFILE, 'obstacles': cfg.OBSTACLES_MODE,
         'dyn': cfg.dyn_profile_constants(cfg.DYN_PROFILE), 'radar_range': float(cfg.RADAR_RANGE),
         'sim': cfg.sim_constants()}
    s.update(extra)
    return s


def test_snapshot_records_all_sim_keys():
    s = ckpt_io.snapshot_config(arm='OFF', msg_dim=6, seed=1, n_envs=2, n_vessels=4, max_partners=4,
                                trunc_boot=False, comm_on_at=0, ring=1.0, crossing=2, rollout=8, trainer='gym')
    assert isinstance(s['sim'], dict)
    assert tuple(s['sim'].keys()) == tuple(cfg.SIM_SNAPSHOT_KEYS), sorted(s['sim'])
    assert s['sim'] == cfg.sim_constants()
    assert len(cfg.SIM_SNAPSHOT_KEYS) == 24, len(cfg.SIM_SNAPSHOT_KEYS)
    # torch.save/JSON 안전한 타입만 (텐서·객체가 섞이면 체크포인트가 커지거나 load 가 깨진다)
    for k, v in s['sim'].items():
        assert v is None or isinstance(v, (bool, int, float, str)), (k, type(v))


def test_env_name_map_is_exact():
    """SIM_ENV_NAMES.get(k, 'VESSEL_'+k) 가 config.py 가 실제로 읽는 이름인지 — 소스 글자로 확인."""
    src = io.open(os.path.join(os.path.dirname(os.path.abspath(cfg.__file__)), 'config.py'),
                  encoding='utf-8').read()
    for k in cfg.SIM_SNAPSHOT_KEYS:
        env = cfg.SIM_ENV_NAMES.get(k, 'VESSEL_' + k)
        assert f"'{env}'" in src, (k, env)
        # 정의 줄에 그 env 이름이 있어야 한다 (VESSEL_MAX_EP_STEPS 처럼 os.environ.get 직접 읽기도 포함)
        line = [ln for ln in src.splitlines() if ln.startswith(k + ' =')]
        assert line and env in line[0], (k, env, line[:1])


def test_mismatch_aborts_and_allow_applies():
    with _sim_consts():
        snap = _snap(sim=dict(cfg.sim_constants(), COLREGS_SIM_COEF=0.30))
        try:
            ckpt_io.apply_sim_snapshot(snap, tag='[t]')
        except SystemExit as e:
            assert 'COLREGS_SIM_COEF' in str(e), str(e)
        else:
            raise AssertionError('sim 상수 불일치인데 중단 안 함')
        notes = []
        eff = ckpt_io.apply_sim_snapshot(snap, allow_sim_mismatch=True, notes=notes, tag='[t]')
        assert cfg.COLREGS_SIM_COEF == 0.30 and vg.COLREGS_SIM_COEF == 0.30
        assert eff['sim_keys_applied'] == len(cfg.SIM_SNAPSHOT_KEYS), eff
    # 블록을 나오면 원래 값
    assert vg.COLREGS_SIM_COEF != 0.30 or cfg.sim_constants()['COLREGS_SIM_COEF'] != 0.30


def test_legacy_snapshot_without_sim():
    with _sim_consts():
        notes = []
        eff = ckpt_io.apply_sim_snapshot(_snap(sim=None), notes=notes, tag='[t]')
        assert eff['sim_keys_applied'] == 0, eff
        assert any('sim 없음' in n for n in notes), notes
        # 'sim' 키 자체가 없는 구 스냅샷도 같다
        notes = []
        eff2 = ckpt_io.apply_sim_snapshot({k: v for k, v in _snap().items() if k != 'sim'},
                                          notes=notes, tag='[t]')
        assert eff2['sim_keys_applied'] == 0, eff2


def test_partial_sim_snapshot_is_not_a_mismatch():
    """구 코드가 상수 하나만 기록했어도 중단하면 안 된다 — 체크포인트에 *있는* 키만 대조한다."""
    with _sim_consts():
        eff = ckpt_io.apply_sim_snapshot(_snap(sim={'FUEL_COEF': cfg.FUEL_COEF}), tag='[t]')
        assert eff['sim_keys_applied'] == 1, eff


def test_missing_key_on_current_side_is_a_mismatch():
    """현재 config 에 없는 키가 체크포인트에 있으면 = 코드가 상수를 지운 것 → 불일치."""
    with _sim_consts():
        try:
            ckpt_io.apply_sim_snapshot(_snap(sim=dict(cfg.sim_constants(), GONE_COEF=1.0)), tag='[t]')
        except SystemExit as e:
            assert 'GONE_COEF' in str(e), str(e)
        else:
            raise AssertionError('현재 config 에 없는 키인데 중단 안 함')


def test_env_lines_emit_sim():
    lines = ckpt_io.env_lines({'sim': cfg.sim_constants(), 'radar_range': 56.0})[0]
    joined = '\n'.join(lines)
    for e in ('export VESSEL_SIM_COLREGS_COEF=', 'export VESSEL_MAX_EP_STEPS=', 'export VESSEL_COLREGS_GATE=',
              'export VESSEL_FUEL_COEF=', 'export VESSEL_RADAR_RANGE='):
        assert e in joined, (e, joined)
    assert 'VESSEL_REWARD_RANGE=None' not in joined, joined   # None = 미설정 → 줄 자체를 내지 않는다
    names = [ln.split('=', 1)[0] for ln in lines]
    assert len(names) == len(set(names)), [n for n in names if names.count(n) > 1]   # 중복 export 금지


def test_make_env_applies_sim():
    """평가 env 도 스냅샷 보상 상수로 만들어져야 한다 (VesselBatchEnv.__init__ 이 cfg 를 직접 읽는 것 포함)."""
    with _sim_consts():
        snap = _snap(sim=dict(cfg.sim_constants(), FUEL_COEF=0.055, FARPAIR_COEF=-0.25),
                     vessels=2, ring=1.0, crossing=2)
        env = ckpt_io.make_env_from_snapshot(snap, device='cpu', num_envs=1, seed=1, tag='[t]')
        assert vg.FUEL_COEF == 0.055 and cfg.FUEL_COEF == 0.055
        assert abs(env.farpair_coef - (-0.25)) < 1e-12, env.farpair_coef


def test_check_branch_rejects_sim_mismatch():
    sim = cfg.sim_constants()
    r = _check_branch_on([('off_s1.pt', {'arm': 'OFF', 'sim': dict(sim)}),
                          ('on6_s1.pt', {'arm': 'ON', 'sim': dict(sim, FUEL_COEF=0.05)})])
    out = r.stdout + r.stderr
    assert r.returncode != 0 and 'sim 불일치' in out, (r.returncode, out[-600:])
    assert 'FUEL_COEF' in out and 'COLREGS_MODE' not in out, out[-600:]   # 다른 키만 찍는다


def test_check_branch_accepts_same_sim():
    sim = cfg.sim_constants()
    r = _check_branch_on([('off_s1.pt', {'arm': 'OFF', 'sim': dict(sim)}),
                          ('on6_s1.pt', {'arm': 'ON', 'sim': dict(sim)})])
    assert r.returncode == 0 and 'ALL PASS' in r.stdout, (r.returncode, r.stdout[-600:])


TESTS = [test_snapshot_records_all_sim_keys, test_env_name_map_is_exact,
         test_mismatch_aborts_and_allow_applies, test_legacy_snapshot_without_sim,
         test_partial_sim_snapshot_is_not_a_mismatch, test_missing_key_on_current_side_is_a_mismatch,
         test_env_lines_emit_sim, test_make_env_applies_sim,
         test_check_branch_rejects_sim_mismatch, test_check_branch_accepts_same_sim]


def main():
    fails = skips = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except _Skip as e:
            skips += 1
            print(f"  SKIP {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")
    print('VERDICT:', 'ALL PASS' if fails == 0 else f'{fails} FAIL', f'(SKIP {skips})' if skips else '')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
