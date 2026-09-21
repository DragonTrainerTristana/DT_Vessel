# VESSEL_DYN_PROFILE=imo — Python·실험 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `VESSEL_DYN_PROFILE=agile|imo`(조종성 IMO 봉투)·`VESSEL_OBSTACLES=grid3x3|none`(open-sea) 토글을 GPU 배치 sim 에 넣고, 스냅샷·복원·분기 검사까지 관통시킨 뒤, 기본값(agile/grid3x3)이 비트동일임을 골든으로 증명한다.

**Architecture:** 상수의 정본은 `config.py`(`dyn_profile_constants(profile)` dict 하나). `vessel_gym.py` 는 그 dict 를 모듈 속성으로 import 하고, 선회식은 `yaw_rate_deg()` 헬퍼 하나로 `_substep`·`_build_obs` 가 공유한다(agile 분기 = 기존 연산 순서 그대로). `ckpt_io` 는 스냅샷에 프로필 이름 + 숫자 dict 를 기록하고, 복원 시 현재 config 와 대조해 불일치면 중단(별도 `allow_sim_mismatch`), 그 뒤 `vg.apply_dyn_constants()` 로 모듈 전역을 덮어쓴다. 학습기 재개·`check_branch` 는 프로필 일치를 강제한다.

**Tech Stack:** Python 3.9/3.10, torch 1.9(Mac CPU)/CUDA(Windows), bash(run_repro.sh). 테스트 = 순수 python 스크립트(`python3 verify/test_*.py`), 골든 = `verify/test_golden.py --check`.

**Spec:** `docs/superpowers/specs/2026-09-19-dyn-profile-imo-design.md` (§3 프로필, §5 보상 규칙, §6 터치포인트, §7 게이트)

## Global Constraints

- 작업 트리 = `/Users/seunghyun/Dropbox/Private_Paper_Project/0702_NewVessel/_dev_dyn` (Dropbox-ignore 된 clone, 브랜치 `feat/dyn-profile-imo`). Dropbox 안 `Assets/Scripts`(main) 는 건드리지 않는다.
- 기본값(`VESSEL_DYN_PROFILE` 미설정 = agile, `VESSEL_OBSTACLES` 미설정 = grid3x3) 은 **비트동일**: 학습기·sim 파일을 건드리는 태스크마다 `python3 verify/test_golden.py --check` → `VERDICT: ALL PASS`.
- `os.environ` 읽기는 `config.py` 에만. 다른 모듈은 `import config as _cfg` 로 받는다(§7 규약).
- 스냅샷 키는 추가만. 삭제·의미 변경 금지.
- 상수 이름은 `vessel_gym` 모듈 속성으로 유지(`vg.RUDDER_RATE` 등 참조처 불변).
- 한국어 주석·docstring 영어. production 코드에 bare `print()` 금지(학습 진행·에러 출력만).
- Mac 은 torch↔numpy 비호환: 테스트에 numpy 쓰지 말 것. `_verify_ppo_mirror.py` 는 Windows 전용(건너뜀).
- 커밋 메시지 끝: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. push 는 `origin feat/dyn-profile-imo`.
- 각 태스크 시작 전 `cd /Users/seunghyun/Dropbox/Private_Paper_Project/0702_NewVessel/_dev_dyn/Python`.

---

## File Structure

| 파일 | 책임 | 변경 |
|---|---|---|
| `Python/config.py` | 프로필 정본: `DYN_PROFILE`, `OBSTACLES_MODE`, `dyn_profile_constants()`, `DYN`, `YUGIOH` 키 | Modify |
| `Python/vessel_gym.py` | 동역학 상수 import, `yaw_rate_deg()`, `apply_dyn_constants()`, `current_dyn_constants()`, cmd_mismatch slack, 장애물 none | Modify |
| `Python/verify/test_dyn_profile.py` | 프로필 단위 테스트(agile==legacy, imo 선회반경·정지거리, obstacles none, 스냅샷 대조) | Create |
| `Python/verify/test_vessel_gym_fidelity.py` | 리터럴 45/0.004 → 프로필 파생값 | Modify |
| `Python/ckpt_io.py` | `snapshot_config` 키, `apply_sim_snapshot()`, `restore_policy(allow_sim_mismatch)`, `make_env_from_snapshot`, `describe`, `_SNAP_TO_ENV` | Modify |
| `Python/vessel_gym_train.py` | 재개 시 sim 일치 검사, 시작 로그 | Modify |
| `Python/verify/check_branch.py` | trunk 묶음 안 `dyn`/`obst` 일치 | Modify |
| `Python/eval/eval_ckpt.py`, `eval/diag_ckpt.py`, `eval/eval_mixed.py`, `eval/measure_regimes.py`, `eval/corridor_run.py` | `allow_sim_mismatch` 전달, restore→env 순서 | Modify |
| `Python/run_repro.sh`, `Python/smoke_mac.sh` | common_env 프로필 export(바깥 값 보존), 프로필 로그, `on2/off2` | Modify |
| `Python/verify/test_golden.py` | `_YUGIOH_CONSTS` 키 2개, CASES 에 agile/grid3x3 명시 핀 | Modify |
| `.claude/CLAUDE.md`, `WINDOWS_RUN.md` | 규약·실행 절차 | Modify |

---

### Task 1: config.py — 프로필 정본

**Files:**
- Modify: `Python/config.py` (import 절 :17-25, vessel_gym 상수 절 끝 :480 근처 `REWARD_RANGE` 다음, `YUGIOH` dict :504-514)
- Create: `Python/verify/test_dyn_profile.py`

**Interfaces:**
- Produces: `config.DYN_PROFILE: str`, `config.OBSTACLES_MODE: str`, `config.SHIP_LEN_M: float`, `config.DYN_K_T: float`, `config.dyn_profile_constants(profile: str) -> dict`, `config.DYN: dict`. dict 키(전부 필수): `formula('ratio'|'abs'), turn_factor, r_full, max_yaw_rate, rudder_rate, accel, decel, drag_coef, tcpa_risk_denom, rule_17b_time, rule_17c_time, rule_17b_dist, rule_17c_dist, early_action_time, substantial_action_time, goal_reached, cmd_mismatch_slack_deg`.

- [ ] **Step 1: Write the failing test**

`Python/verify/test_dyn_profile.py` 를 아래 내용으로 만든다(Task 2 이후 테스트는 뒤 태스크에서 추가).

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 verify/test_dyn_profile.py`
Expected: `FAIL test_agile_dict_equals_legacy_literals: AttributeError: module 'config' has no attribute 'dyn_profile_constants'` (4개 전부 FAIL), exit 1.

- [ ] **Step 3: Implement in config.py**

(a) import 절(:17-25)에 `import math` 추가(없으면).

(b) `REWARD_RANGE = _env_float('VESSEL_REWARD_RANGE', None)` 줄 **바로 뒤**에 추가:

```python
# ── ★동역학 프로필 (2026-09-21, feat/dyn-profile-imo — 스펙 docs/superpowers/specs/2026-09-19-dyn-profile-imo-design.md) ──
#   agile = 현행(전타 yaw 45 °/s, 선회직경 0.18 L). 기본 = 비트동일.
#   imo   = IMO MSC.137(76) 봉투: 선회직경 4 L 전 선박 고정(절대속도 식 yaw=(rudder/30°)·speed/R_FULL), 타속 3 °/s(SOLAS II-1/29),
#           정지거리 ≈ 5 L(DECEL/DRAG/ACCEL), 물리 파생 보상 상수 × k_t 2.27(T_lat12 32.0/14.1 s). 가중치는 불변.
#   vessel_gym 은 DYN 을 모듈 속성으로 import 하고, ckpt_io 는 스냅샷에 'dyn_profile' + 'dyn'(이 dict 숫자) 를 기록·대조·복원한다.
#   state_dict 키에 영향 없음 = 스냅샷이 유일 근거(§5 "조용히 다른 실험" 목록에 해당).
DYN_PROFILE = _env_str('VESSEL_DYN_PROFILE', 'agile').lower()
assert DYN_PROFILE in ('agile', 'imo'), f"VESSEL_DYN_PROFILE={DYN_PROFILE!r} - 'agile' | 'imo'"
OBSTACLES_MODE = _env_str('VESSEL_OBSTACLES', 'grid3x3').lower()   # 'grid3x3'(현행 coastal) | 'none'(open-sea, 벽만)
assert OBSTACLES_MODE in ('grid3x3', 'none'), f"VESSEL_OBSTACLES={OBSTACLES_MODE!r} - 'grid3x3' | 'none'"
SHIP_LEN_M = 14.18316   # 충돌 박스 길이 L (= vessel_gym.SHIP_HALF_LEN × 2). imo 프로필의 길이 단위
DYN_K_T = 2.27          # imo 시간 배율 = T_lat12(imo 32.0 s) / T_lat12(agile 14.1 s) — 스펙 §5, dyn_profiles.py 실측


def dyn_profile_constants(profile):
    """Profile name -> dict of dynamics + physically-derived reward constants.

    Single source for vessel_gym (import), ckpt_io (snapshot record/restore) and tests.
    'agile' values equal the pre-2026-09-21 vessel_gym.py literals (bit-identical golden).
    """
    if profile == 'agile':
        return {
            'formula': 'ratio', 'turn_factor': 1.5, 'r_full': None, 'max_yaw_rate': 45.0, 'rudder_rate': 12.0,
            'accel': 0.1, 'decel': 0.04, 'drag_coef': 0.1,
            'tcpa_risk_denom': 30.0, 'rule_17b_time': 7.0, 'rule_17c_time': 3.5,
            'rule_17b_dist': 18.0, 'rule_17c_dist': 9.0,
            'early_action_time': 21.5, 'substantial_action_time': 11.5,
            'goal_reached': 3.0, 'cmd_mismatch_slack_deg': 0.0,
        }
    if profile == 'imo':
        r_full = 2.0 * SHIP_LEN_M          # 전타 정상 선회반경 = 2 L → 선회직경 4 L (IMO TD ≤ 5 L)
        fleet_vmax = 1.0 * 1.8             # vessel_gym MAX_SPEED_BASE × SPEED_MULT_MAX = 함대 최고속 (obs[363] 분모 기준)
        rudder_rate = 3.0                  # °/s. SOLAS II-1/29 최소 ≈ 2.3
        return {
            'formula': 'abs', 'turn_factor': None, 'r_full': r_full,
            'max_yaw_rate': fleet_vmax / r_full / (math.pi / 180.0),   # ≈ 3.635 °/s
            'rudder_rate': rudder_rate, 'accel': 0.01, 'decel': 0.004, 'drag_coef': 0.005,
            'tcpa_risk_denom': 30.0 * DYN_K_T, 'rule_17b_time': 7.0 * DYN_K_T, 'rule_17c_time': 3.5 * DYN_K_T,
            'rule_17b_dist': 18.0 * DYN_K_T, 'rule_17c_dist': 9.0 * DYN_K_T,
            'early_action_time': 21.5 * DYN_K_T, 'substantial_action_time': 11.5 * DYN_K_T,
            'goal_reached': SHIP_LEN_M / 2.0, 'cmd_mismatch_slack_deg': rudder_rate * 0.4,   # 결정(0.4 s)당 달성 가능 슬루
        }
    raise ValueError(f"dyn_profile_constants: 모르는 프로필 {profile!r} ('agile' | 'imo')")


DYN = dyn_profile_constants(DYN_PROFILE)
```

(c) `YUGIOH` dict(:504-514)의 마지막 항목 `'VESSEL_TIMEOUT_BOOTSTRAP': '0', 'VESSEL_MSG_GATE_APPLY': '0',` 뒤에 추가:

```python
    'VESSEL_DYN_PROFILE': 'agile', 'VESSEL_OBSTACLES': 'grid3x3',   # ★2026-09-21 동역학 프로필·시나리오 (기본 = 비트동일)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 verify/test_dyn_profile.py`
Expected: 4개 `PASS`, `VERDICT: ALL PASS`, exit 0.

- [ ] **Step 5: config import 부작용 확인 + 골든**

Run: `python3 -c "import config as c; print(c.DYN_PROFILE, c.OBSTACLES_MODE, c.DYN['max_yaw_rate'])"` → `agile grid3x3 45.0`
Run: `VESSEL_DYN_PROFILE=imo python3 -c "import config as c; print(round(c.DYN['max_yaw_rate'],4), c.DYN['r_full'])"` → `3.6355 28.36632`
Run: `VESSEL_DYN_PROFILE=boat python3 -c "import config"` → `AssertionError` (exit ≠ 0)
Run: `python3 verify/test_golden.py --check` → `VERDICT: ALL PASS` (config 만 바뀌었으니 당연히 PASS 여야 함 — 아니면 여기서 멈춤)

- [ ] **Step 6: Commit**

```bash
cd /Users/seunghyun/Dropbox/Private_Paper_Project/0702_NewVessel/_dev_dyn
git add Python/config.py Python/verify/test_dyn_profile.py
git commit -m "feat(dyn): config.dyn_profile_constants — VESSEL_DYN_PROFILE(agile|imo)·VESSEL_OBSTACLES 정본 + 단위 테스트

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: vessel_gym.py — 상수 import·선회식 헬퍼·slack·장애물 none (비트동일)

**Files:**
- Modify: `Python/vessel_gym.py` :29-40(동역학 상수), :62(`GOAL_REACHED`), :82-87(COLREGs 시간·거리), :119(`TCPA_RISK_DENOM`), :128 근처(`OBSTACLES_MODE`), :278-282(장애물 생성), :434-437(`_substep` yaw), :694-695(`_build_obs` yaw), :925-926(cmd_mismatch)
- Modify: `Python/verify/test_dyn_profile.py` (테스트 추가)

**Interfaces:**
- Consumes: `config.DYN`, `config.DYN_PROFILE`, `config.OBSTACLES_MODE`
- Produces: `vessel_gym.yaw_rate_deg(rudder, speed, max_speed) -> Tensor`, `vessel_gym.apply_dyn_constants(d: dict, profile: str|None) -> None`, `vessel_gym.current_dyn_constants() -> dict`, 모듈 전역 `DYN_PROFILE, DYN_FORMULA, R_FULL, CMD_MISMATCH_SLACK_DEG, OBSTACLES_MODE` (+ 기존 이름 전부 유지)

- [ ] **Step 1: 테스트 추가 (실패 확인용)**

`verify/test_dyn_profile.py` 의 `import config as cfg` 다음 줄에 `import vessel_gym as vg  # noqa: E402` 추가, `TESTS` 앞에 아래 함수들을 넣고 `TESTS` 리스트에 추가한다.

```python
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
```

`TESTS = [..., test_imo_turn_radius_fixed_across_fleet, test_agile_turn_radius_unchanged, test_stop_distance, test_yaw_helper_matches_legacy_formula, test_obs_yaw_norm_bounded, test_obstacles_none, test_obstacles_grid_default]`

- [ ] **Step 2: 실패 확인**

Run: `python3 verify/test_dyn_profile.py`
Expected: 새 테스트 7개 FAIL (`AttributeError: module 'vessel_gym' has no attribute 'current_dyn_constants'` 등). Task 1 의 4개는 PASS.

- [ ] **Step 3: vessel_gym.py 상수 블록 교체**

`grep -n "^DEG\b\|^DEG =" vessel_gym.py` 로 `DEG` 정의 위치를 확인한다(`_substep` 이 `self.heading * DEG` 를 쓰므로 반드시 있음). :29-40 을 아래로 교체:

```python
DT = 0.04                    # fixedDeltaTime
SUBSTEPS = 10                # DecisionPeriod — 결정(0.4s)당 물리 서브스텝
MAX_SPEED_BASE = 1.0         # GlobalScale.MAX_SPEED
# ★2026-09-21 동역학 프로필 (config.DYN ← VESSEL_DYN_PROFILE=agile|imo). 이름은 모듈 속성으로 유지(vg.RUDDER_RATE 등 참조처 불변).
#   agile 값은 옛 리터럴과 동일(ACCEL 0.1 · DECEL 0.04 · RUDDER_RATE 12 · TURN_FACTOR 1.5 · MAX_YAW_RATE 45 · DRAG 0.1) = 비트동일.
#   imo 는 스펙 §3. ckpt_io.restore_policy / make_env_from_snapshot 이 apply_dyn_constants() 로 스냅샷 값을 덮어쓴다.
DYN_PROFILE = _cfg.DYN_PROFILE
DYN_FORMULA = _cfg.DYN['formula']        # 'ratio'(agile: rudder·speedRatio·TURN_FACTOR) | 'abs'(imo: (rudder/30)·speed/R_FULL)
ACCEL = _cfg.DYN['accel']                # 가속률 (/s 아님, MoveTowards maxDelta = ACCEL*dt)
DECEL = _cfg.DYN['decel']
RUDDER_RATE = _cfg.DYN['rudder_rate']    # deg/s 타각 슬루
MAX_TURN_RATE = 30.0         # deg (명령 타각 상한) — 프로필 무관(obs[365]·보상 정규화 상수)
TURN_FACTOR = _cfg.DYN['turn_factor']    # agile: rudderEff(1.5)·(10/length 2.0)·(beam 0.4/2) = 1.5 / imo: None (식이 다름)
R_FULL = _cfg.DYN['r_full']              # imo: 전타 정상 선회반경 [m] = 2 L / agile: None
MAX_YAW_RATE = _cfg.DYN['max_yaw_rate']  # obs[363] 분모. agile 45 = MAX_TURN_RATE×TURN_FACTOR / imo 함대 최고속 전타 yaw (파생값)
DRAG_COEF = _cfg.DYN['drag_coef']
DRAG_THRUST_MULT = 0.3       # targetSpeed>=0.1이면 drag ×0.3
DRAG_THRUST_THRESH = 0.1     # 절대속도 단위
CMD_MISMATCH_SLACK_DEG = _cfg.DYN['cmd_mismatch_slack_deg']   # 12-b 타속 포화 벌점에서 공제할 '달성 가능 슬루' (agile 0 = 비트동일)
```

- [ ] **Step 4: GOAL_REACHED · COLREGs 상수 · TCPA · OBSTACLES_MODE**

- :62 `GOAL_REACHED = 3.0` → `GOAL_REACHED = _cfg.DYN['goal_reached']   # agile 3.0 (0.21 L) / imo L/2 = 7.09 (스펙 §5)`
- :82-87 여섯 줄을 아래로(주석 유지):

```python
EARLY_ACTION_TIME       = _cfg.DYN['early_action_time']        # Rule 16 조기행동 시점(s) — agile 21.5 / imo ×2.27
SUBSTANTIAL_ACTION_TIME = _cfg.DYN['substantial_action_time']  # Rule 16 충분행동 시점(s) — agile 11.5
RULE_17B_TIME           = _cfg.DYN['rule_17b_time']            # stand-on 이 행동 *가능*해지는 시점(s) — agile 7.0
RULE_17C_TIME           = _cfg.DYN['rule_17c_time']            # stand-on 이 행동 *해야 하는* 시점(s) — agile 3.5
RULE_17B_DIST           = _cfg.DYN['rule_17b_dist']            # agile 18.0 (BASE 90 × 0.2)
RULE_17C_DIST           = _cfg.DYN['rule_17c_dist']            # agile 9.0 (BASE 45 × 0.2)
```

- :119 `TCPA_RISK_DENOM = 30.0` → `TCPA_RISK_DENOM = _cfg.DYN['tcpa_risk_denom']   # agile 30 s / imo ×2.27 (스펙 §5)`
- :128 `OBSTACLE_RADIUS = 20.0` 바로 위에 `OBSTACLES_MODE = _cfg.OBSTACLES_MODE   # 'grid3x3'(현행) | 'none'(open-sea, 2026-09-21)` 추가.

- [ ] **Step 5: 헬퍼 3개 추가** (`_move_toward` 정의 바로 뒤, `grep -n "^def _move_toward" vessel_gym.py`)

```python
def yaw_rate_deg(rudder, speed, max_speed):
    """Yaw rate [deg/s] for the active dynamics profile.

    'ratio' (agile): rudder * (speed/max_speed) * TURN_FACTOR — the legacy expression in its original
    operation order (bit-identical). 'abs' (imo): (rudder/MAX_TURN_RATE) * speed / R_FULL [rad/s] -> deg/s,
    so the steady full-rudder turning radius is R_FULL for every vessel regardless of speed (Nomoto-like).
    """
    if DYN_FORMULA == 'abs':
        return (rudder / MAX_TURN_RATE) * speed / R_FULL / DEG
    speed_ratio = speed / torch.clamp(max_speed, min=1e-6)
    return rudder * speed_ratio * TURN_FACTOR


_DYN_TO_MODULE = {
    'formula': 'DYN_FORMULA', 'turn_factor': 'TURN_FACTOR', 'r_full': 'R_FULL', 'max_yaw_rate': 'MAX_YAW_RATE',
    'rudder_rate': 'RUDDER_RATE', 'accel': 'ACCEL', 'decel': 'DECEL', 'drag_coef': 'DRAG_COEF',
    'tcpa_risk_denom': 'TCPA_RISK_DENOM', 'rule_17b_time': 'RULE_17B_TIME', 'rule_17c_time': 'RULE_17C_TIME',
    'rule_17b_dist': 'RULE_17B_DIST', 'rule_17c_dist': 'RULE_17C_DIST',
    'early_action_time': 'EARLY_ACTION_TIME', 'substantial_action_time': 'SUBSTANTIAL_ACTION_TIME',
    'goal_reached': 'GOAL_REACHED', 'cmd_mismatch_slack_deg': 'CMD_MISMATCH_SLACK_DEG',
}


def apply_dyn_constants(d, profile=None):
    """Overwrite this module's dynamics/reward globals from a profile dict (config.dyn_profile_constants or a
    checkpoint snapshot 'dyn'). Methods read module globals at call time, so this takes effect on the next step
    whether called before or after VesselBatchEnv() (ckpt_io restore path)."""
    g = globals()
    for k, name in _DYN_TO_MODULE.items():
        if k in d:
            g[name] = d[k]
    if profile is not None:
        g['DYN_PROFILE'] = str(profile)


def current_dyn_constants():
    """Return the profile dict currently applied to this module (snapshot recording / tests)."""
    g = globals()
    return {k: g[name] for k, name in _DYN_TO_MODULE.items()}
```

- [ ] **Step 6: `_substep` · `_build_obs` · cmd_mismatch · 장애물**

`_substep` :434-437 의 네 줄

```python
        speed_ratio = self.speed / torch.clamp(self.max_speed, min=1e-6)
        eff_rudder = self.rudder * speed_ratio
        # 6. yawRate [deg/s]
        yaw_rate = eff_rudder * TURN_FACTOR
```
→
```python
        # 4-6. yawRate [deg/s] — 프로필별 식(yaw_rate_deg). agile 은 옛 (rudder·speedRatio)·TURN_FACTOR 연산 순서 그대로(비트동일)
        yaw_rate = yaw_rate_deg(self.rudder, self.speed, self.max_speed)
```

`_build_obs` :694 `yaw_rate = self.rudder * speed_ratio * TURN_FACTOR` → `yaw_rate = yaw_rate_deg(self.rudder, self.speed, self.max_speed)   # 프로필별 식 (agile 비트동일)`. (:693 `speed_ratio` 는 obs[362] 에 그대로 쓰이므로 남긴다.)

cmd_mismatch :925 `_sat = torch.clamp((self.cmd_rudder - self.rudder).abs() / MAX_TURN_RATE, 0, 1)` →
```python
        _dev = (self.cmd_rudder - self.rudder).abs()
        if CMD_MISMATCH_SLACK_DEG > 0.0:
            # ★imo: 결정(0.4 s)당 달성 가능한 슬루(RR×0.4°)는 정책 탓이 아니라 물리 → 공제 (스펙 §5). agile 은 0 = 옛 식 그대로
            _dev = torch.clamp(_dev - CMD_MISMATCH_SLACK_DEG, min=0.0)
        _sat = torch.clamp(_dev / MAX_TURN_RATE, 0, 1)
```

장애물 :278-282 →
```python
        if OBSTACLES_MODE == 'none':
            # ★2026-09-21 open-sea: 장애물 0 → [0,2]. _radar 루프·_obb_circle_hit(.any 빈 축)·LOS 게이트(shape[0]>0 가드) 전부 안전.
            self.obstacles = torch.zeros(0, 2, device=self.device, dtype=self.dtype)
        else:
            # 정적 장애물 9개 (원점 중심 3×3, 반지름 20) — [9,2]
            gx = torch.tensor([-OBSTACLE_GRID_STEP, 0.0, OBSTACLE_GRID_STEP], device=self.device, dtype=self.dtype)
            ox, oz = torch.meshgrid(gx, gx)   # torch<1.10 기본 'ij' indexing
            self.obstacles = torch.stack([ox.reshape(-1), oz.reshape(-1)], dim=-1)  # [9,2]
        self.obstacle_r = OBSTACLE_RADIUS
```

- [ ] **Step 7: 테스트·골든·미러**

Run: `python3 verify/test_dyn_profile.py` → 11개 PASS, `VERDICT: ALL PASS`.
Run: `python3 verify/test_golden.py --check` → `VERDICT: ALL PASS` (★비트동일 게이트. FAIL 이면 Step 3/6 의 연산 순서를 옛 식과 대조).
Run: `python3 verify/_verify_comm_mirror.py` → `VERDICT: ALL PASS`.
Run: `grep -n "TURN_FACTOR\|MAX_YAW_RATE" vessel_gym.py eval/*.py verify/*.py astar_fig9/*.py` → `vessel_gym.py` 밖에서 `TURN_FACTOR` 를 **곱셈에** 쓰는 곳이 있으면 `yaw_rate_deg` 로 바꾼다(fidelity 는 Task 3 에서).

- [ ] **Step 8: Commit**

```bash
git add Python/vessel_gym.py Python/verify/test_dyn_profile.py
git commit -m "feat(dyn): vessel_gym 프로필 상수 import·yaw_rate_deg 헬퍼·cmd_mismatch slack·장애물 none (agile 골든 비트동일)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: fidelity 테스트 파라미터화

**Files:**
- Modify: `Python/verify/test_vessel_gym_fidelity.py` :1-11(docstring), :18-56(`scalar_dynamics_reference` yaw 줄), :87-113(`test_physics_sanity`)

**Interfaces:**
- Consumes: `vg.DYN_FORMULA, vg.R_FULL, vg.MAX_TURN_RATE, vg.DEG, vg.ACCEL, vg.DT, vg.DRAG_COEF, vg.DRAG_THRUST_MULT`

- [ ] **Step 1: 실패 확인 (imo 로 현행 테스트)**

Run: `VESSEL_DYN_PROFILE=imo python3 verify/test_vessel_gym_fidelity.py`
Expected: `AssertionError: 선회율 이상 2.02...` (또는 배칭 불일치) — 리터럴 45 때문.

- [ ] **Step 2: 참조 구현·assert 를 프로필 파생값으로**

`scalar_dynamics_reference` 의
```python
            sr = s / max(1e-6, max_speed)
            yaw = (rudder * sr) * vg.TURN_FACTOR
```
→
```python
            if vg.DYN_FORMULA == 'abs':
                yaw = (rudder / vg.MAX_TURN_RATE) * s / vg.R_FULL / vg.DEG      # imo 절대속도 식
            else:
                sr = s / max(1e-6, max_speed)
                yaw = (rudder * sr) * vg.TURN_FACTOR                            # agile 옛 식
```

`test_physics_sanity` 의 두 assert:
```python
    assert 43.0 < yaw_rate <= 45.5, f"선회율 이상 {yaw_rate}"
```
→
```python
    exp_yaw = 45.0 if vg.DYN_FORMULA == 'ratio' else (30.0 / vg.MAX_TURN_RATE) * 1.0 / vg.R_FULL / vg.DEG
    print(f"  full-rudder yawRate ≈ {yaw_rate:.3f} deg/s (프로필 {vg.DYN_PROFILE} 기대 {exp_yaw:.3f}, drag 로 소폭↓)")
    assert exp_yaw * 0.955 < yaw_rate <= exp_yaw * 1.012, f"선회율 이상 {yaw_rate} (기대 {exp_yaw})"
```
(기존 문구 `print(f"  full-rudder yawRate ≈ ... (이론 ~45 ...)")` 는 지운다.)
```python
    assert 0.0039 < s1 < 0.0041, f"가속 이상 {s1}"
```
→
```python
    exp_s1 = vg.ACCEL * vg.DT * (1.0 - vg.DRAG_COEF * vg.DT * vg.DRAG_THRUST_MULT)
    print(f"  직진 첫 서브스텝 속도 {s1:.6f} (기대 {exp_s1:.6f} = ACCEL·DT 후 drag×0.3)")
    assert abs(s1 - exp_s1) < 0.025 * exp_s1, f"가속 이상 {s1} (기대 {exp_s1})"
```
docstring `(2) 물리 상식: full rudder 선회율 45°/s 수렴, 직진 가속이 accel rate 준수, drag 평형` → `(2) 물리 상식: full rudder 선회율이 프로필 기대값(agile 45°/s, imo 2.02°/s)에 수렴, 직진 가속이 ACCEL 준수, drag 평형`.

- [ ] **Step 3: 양 프로필 PASS**

Run: `python3 verify/test_vessel_gym_fidelity.py` → 전부 PASS(agile).
Run: `VESSEL_DYN_PROFILE=imo python3 verify/test_vessel_gym_fidelity.py` → 전부 PASS.
Run: `VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none python3 verify/test_vessel_gym_fidelity.py` → `test_radar_geometry` 가 정면 장애물을 가정하면 FAIL 할 수 있음. 그 경우 그 테스트 안에서 `if vg.OBSTACLES_MODE == 'none': print('  [3] 장애물 없음 - 건너뜀'); return` 를 맨 앞에 넣는다.

- [ ] **Step 4: Commit**

```bash
git add Python/verify/test_vessel_gym_fidelity.py
git commit -m "test(dyn): fidelity 선회율·가속 assert 를 프로필 파생값 기준으로 (imo/none 도 PASS)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: ckpt_io — 스냅샷 기록·대조·복원 (기존 구멍 수리 포함)

**Files:**
- Modify: `Python/ckpt_io.py` `snapshot_config` :34-82, `Restored.header` :97-106, `restore_policy` 시그니처 :113-114 및 :238-245(스냅샷 유/무 분기 끝) 및 `effective` :315-327, `make_env_from_snapshot` :339-385, `describe` :388-396, `_SNAP_TO_ENV` :400-418
- Modify: `Python/verify/test_dyn_profile.py`

**Interfaces:**
- Produces: `ckpt_io.apply_sim_snapshot(snap, *, allow_sim_mismatch=False, notes=None, tag='[ckpt]') -> dict{dyn_profile, obstacles, radar_range}`; `restore_policy(..., allow_sim_mismatch=False)`; 스냅샷 키 `dyn_profile, dyn, obstacles, radar_dropout_p, radar_dropout_len, los_gate, max_episode_steps`; `Restored.effective['dyn_profile'|'obstacles'|'radar_range']`.

- [ ] **Step 1: 테스트 추가**

`verify/test_dyn_profile.py` 에 추가(`import ckpt_io  # noqa: E402` 도 import 절에):

```python
def test_snapshot_records_profile():
    s = ckpt_io.snapshot_config(arm='OFF', msg_dim=6, seed=1, n_envs=2, n_vessels=4, max_partners=4, trunc_boot=False,
                                comm_on_at=0, ring=1.0, crossing=2, rollout=8, trainer='gym')
    assert s['dyn_profile'] == cfg.DYN_PROFILE and s['dyn'] == vg.current_dyn_constants()
    assert s['obstacles'] == cfg.OBSTACLES_MODE
    for k in ('radar_dropout_p', 'radar_dropout_len', 'los_gate', 'max_episode_steps'):
        assert k in s, k


def test_apply_sim_snapshot_legacy_and_mismatch():
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
```
`TESTS` 에 두 함수 추가. (이 테스트는 env 없이(agile 기본) 돌릴 때 유효 — `VESSEL_DYN_PROFILE=imo` 로 돌리면 (b) 가 뒤집히므로 함수 첫 줄에 `if cfg.DYN_PROFILE != 'agile' or cfg.OBSTACLES_MODE != 'grid3x3': return` 를 넣는다.)

- [ ] **Step 2: 실패 확인**

Run: `python3 verify/test_dyn_profile.py` → 두 테스트 FAIL (`KeyError: 'dyn_profile'`, `AttributeError: apply_sim_snapshot`).

- [ ] **Step 3: snapshot_config 키 추가**

`'msg_gate_apply': bool(cfg.MSG_GATE_APPLY),` 뒤(dict 마지막)에:
```python
        # ★2026-09-21 동역학 프로필·시나리오 + 이전에 빠져 있던 sim 토글 (키에 흔적 없음 = 스냅샷이 유일 근거). 키 추가만.
        'dyn_profile': str(cfg.DYN_PROFILE),
        'dyn': dict(vg.current_dyn_constants()),      # 숫자 dict — 나중에 프로필 정의가 바뀌어도 이 값으로 재현
        'obstacles': str(cfg.OBSTACLES_MODE),
        'radar_dropout_p': float(cfg.RADAR_DROPOUT_P), 'radar_dropout_len': int(cfg.RADAR_DROPOUT_LEN),
        'los_gate': bool(cfg.LOS_GATE), 'max_episode_steps': int(cfg.MAX_EPISODE_STEPS),
```

- [ ] **Step 4: `apply_sim_snapshot` + restore 연결**

`class Restored` 정의 **앞**에 함수 추가:
```python
def apply_sim_snapshot(snap, *, allow_sim_mismatch=False, notes=None, tag='[ckpt]'):
    """Compare the checkpoint's sim settings (dyn_profile, obstacles, radar_range) with the current config and
    apply them to vessel_gym module globals. Missing keys mean legacy (agile / grid3x3). A mismatch aborts unless
    allow_sim_mismatch (deliberately separate from allow_comm_range_mismatch, which 4 eval scripts already set).
    Returns the effective {'dyn_profile', 'obstacles', 'radar_range'}."""
    snap = snap or {}
    notes = notes if notes is not None else []
    ck_dp = str(snap.get('dyn_profile') or 'agile').lower()
    ck_ob = str(snap.get('obstacles') or 'grid3x3').lower()
    ck_rr = snap.get('radar_range')
    bad = []
    if ck_dp != cfg.DYN_PROFILE:
        bad.append(f"dyn_profile ckpt={ck_dp} 현재={cfg.DYN_PROFILE}")
    if ck_ob != cfg.OBSTACLES_MODE:
        bad.append(f"obstacles ckpt={ck_ob} 현재={cfg.OBSTACLES_MODE}")
    if ck_rr is not None and abs(float(ck_rr) - float(vg.RADAR_RANGE)) > 1e-6:
        bad.append(f"radar_range ckpt={ck_rr} 현재={vg.RADAR_RANGE}")
    if bad:
        msg = ("sim 설정 불일치: " + "; ".join(bad)
               + " — VESSEL_DYN_PROFILE / VESSEL_OBSTACLES / VESSEL_RADAR_RANGE 를 학습값으로 주고 다시 실행할 것")
        if not allow_sim_mismatch:
            raise SystemExit(f"{tag} 중단: {msg}")
        notes.append(msg + " (allow_sim_mismatch 로 진행 — 스냅샷 값을 vessel_gym 에 강제 적용)")
    vg.apply_dyn_constants(snap.get('dyn') or cfg.dyn_profile_constants(ck_dp), ck_dp)
    vg.OBSTACLES_MODE = ck_ob
    if ck_rr is not None:
        vg.RADAR_RANGE = float(ck_rr)
    if not snap.get('dyn_profile'):
        notes.append("스냅샷에 dyn_profile 없음(2026-09-21 이전) → legacy 'agile'/'grid3x3' 로 복원")
    return {'dyn_profile': ck_dp, 'obstacles': ck_ob, 'radar_range': float(vg.RADAR_RANGE)}
```

`restore_policy` 시그니처: `allow_arm_mismatch=False, allow_comm_range_mismatch=False, tag='[ckpt]'` → `allow_arm_mismatch=False, allow_comm_range_mismatch=False, allow_sim_mismatch=False, tag='[ckpt]'`. docstring Args 에 한 줄: `allow_sim_mismatch: dyn_profile·obstacles·radar_range 가 스냅샷과 다르면 기본 중단. True 면 스냅샷 값을 강제 적용하고 경고만.`

`if max_partners is None: max_partners = int(cfg.MAX_COMM_PARTNERS)` 줄 **바로 뒤**(스냅샷 유/무 분기가 끝난 자리)에:
```python
    # 4a) ★2026-09-21 sim 설정(동역학 프로필·시나리오·레이더 범위) — 불일치면 중단, 아니면 vessel_gym 전역에 적용
    sim_eff = apply_sim_snapshot(snap, allow_sim_mismatch=allow_sim_mismatch, notes=notes, tag=tag)
```
`effective = {` dict 의 `'ckpt_steps': ...,` 뒤에 `**sim_eff,` 추가. `header()` 의 `f"comm_range={e['comm_range']} max_partners=..."` 앞에 `f"dyn={e.get('dyn_profile')} obst={e.get('obstacles')} "` 추가.

- [ ] **Step 5: make_env_from_snapshot · describe · _SNAP_TO_ENV**

`make_env_from_snapshot` 의 `snap = snap or {}` 바로 뒤:
```python
    # ★2026-09-21: env 생성 *전* 스냅샷의 동역학·시나리오를 vessel_gym 전역에 적용(멱등 — restore_policy 가 이미 했어도 무해).
    if snap.get('dyn') or snap.get('dyn_profile'):
        vg.apply_dyn_constants(snap.get('dyn') or cfg.dyn_profile_constants(str(snap.get('dyn_profile'))),
                               str(snap.get('dyn_profile') or 'agile'))
    if snap.get('obstacles'):
        vg.OBSTACLES_MODE = str(snap['obstacles']).lower()
```
`used = (f"envs={num_envs} ...` 문자열 끝에 `f" dyn={vg.DYN_PROFILE} obst={vg.OBSTACLES_MODE}"` 추가.
`describe` keys 튜플 끝에 `'dyn_profile', 'obstacles'` 추가.
`_SNAP_TO_ENV` 리스트 끝에 `('dyn_profile', 'VESSEL_DYN_PROFILE', str), ('obstacles', 'VESSEL_OBSTACLES', str),` 추가.

- [ ] **Step 6: 테스트·골든**

Run: `python3 verify/test_dyn_profile.py` → 13개 PASS.
Run: `python3 verify/test_golden.py --check` → `VERDICT: ALL PASS` (골든 JSON 의 cfg_snapshot 은 "골든에 있는 키만 비교, 추가 허용").
Run(스냅샷 왕복, 2 update 학습): `VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none python3 vessel_gym_train.py --arm OFF --envs 4 --vessels 4 --rollout 8 --steps 256 --ring 1.0 --crossing 2 --save /tmp/imo_smoke.pt --csv /tmp/imo_smoke.csv --seed 1 2>&1 | tail -3` → 정상 종료.
Run: `python3 ckpt_io.py /tmp/imo_smoke.pt 2>&1 | tail -3` (agile 기본 config) → `중단: sim 설정 불일치: dyn_profile ckpt=imo 현재=agile; obstacles ...` 로 exit ≠ 0.
Run: `VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none python3 ckpt_io.py /tmp/imo_smoke.pt 2>&1 | grep -o "dyn=imo obst=none"` → 출력됨.

- [ ] **Step 7: Commit**

```bash
git add Python/ckpt_io.py Python/verify/test_dyn_profile.py
git commit -m "feat(dyn): 스냅샷에 dyn_profile/dyn/obstacles(+dropout·los·max_steps) 기록, restore 가 대조·복원 (allow_sim_mismatch 별도)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: 학습기 재개 검사·시작 로그 + check_branch

**Files:**
- Modify: `Python/vessel_gym_train.py` `main()` 시작 `[version]` print 다음 줄, 재개 블록 :580-582(`_prev_snap` 직후)
- Modify: `Python/verify/check_branch.py` `load_meta` :41-49, 표 출력 :62-66, 그룹 검사 :74-78
- Modify: `Python/verify/test_dyn_profile.py`

- [ ] **Step 1: 테스트 추가 (check_branch 불일치 FAIL)**

```python
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
```
`TESTS` 에 추가. Run → FAIL(현재는 dyn 을 안 봐서 PASS 로 끝남 = returncode 0).

- [ ] **Step 2: 학습기**

`print(f"[version] ...", flush=True)` 다음 줄에:
```python
    print(f"[sim] dyn_profile={cfg.DYN_PROFILE} obstacles={cfg.OBSTACLES_MODE} formula={cfg.DYN['formula']} "
          f"rudder_rate={cfg.DYN['rudder_rate']} r_full={cfg.DYN['r_full']} goal_reached={cfg.DYN['goal_reached']:.2f}", flush=True)
```
재개 블록 `_prev_snap = _ck.get('cfg_snapshot') or {}` 다음 줄에:
```python
        # ★2026-09-21 sim 설정 일치 — 재개·분기는 같은 동역학·시나리오여야 함(다르면 다른 실험을 이어 붙이는 것). 우회 없음.
        _prev_dp = str(_prev_snap.get('dyn_profile') or 'agile').lower()
        _prev_ob = str(_prev_snap.get('obstacles') or 'grid3x3').lower()
        if _prev_dp != cfg.DYN_PROFILE or _prev_ob != cfg.OBSTACLES_MODE:
            raise SystemExit(f"[resume] 거부: 체크포인트 sim 설정 dyn_profile={_prev_dp} obstacles={_prev_ob} != "
                             f"현재 {cfg.DYN_PROFILE}/{cfg.OBSTACLES_MODE} - VESSEL_DYN_PROFILE/VESSEL_OBSTACLES 를 맞출 것")
```

- [ ] **Step 3: check_branch**

`load_meta` 반환 dict 에 `'dyn': str(snap.get('dyn_profile') or 'agile'), 'obst': str(snap.get('obstacles') or 'grid3x3'),` 추가.
표 헤더 `print(f"{'체크포인트':<28} {'arm':<7} {'seed':>5} {'dim':>4} {'steps':>10} {'branch_at':>10}  trunk")` → 끝에 `  dyn/obst` 추가하고 각 행 print 끝에 `f"  {m['dyn']}/{m['obst']}"` 추가.
그룹 검사 `for key in ('seed', 'msg_dim', 'at'):` → `for key in ('seed', 'msg_dim', 'at', 'dyn', 'obst'):`.

- [ ] **Step 4: 검증**

Run: `python3 verify/test_dyn_profile.py` → 14개 PASS.
Run(agile trunk → imo 갈래 거부): `python3 vessel_gym_train.py --arm OFF --envs 4 --vessels 4 --rollout 8 --steps 256 --ring 1.0 --crossing 2 --save /tmp/agile_trunk.pt --csv /tmp/agile_trunk.csv --seed 1 >/dev/null 2>&1; VESSEL_DYN_PROFILE=imo python3 vessel_gym_train.py --arm ON --resume /tmp/agile_trunk.pt --resume_at 256 --comm_on_at 256 --envs 4 --vessels 4 --rollout 8 --steps 512 --ring 1.0 --crossing 2 --save /tmp/x.pt --csv /tmp/x.csv --seed 1 2>&1 | grep "\[resume\] 거부"` → 출력됨.
Run: `python3 verify/test_golden.py --check` → ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add Python/vessel_gym_train.py Python/verify/check_branch.py Python/verify/test_dyn_profile.py
git commit -m "feat(dyn): 재개·분기 시 dyn_profile/obstacles 일치 강제 (학습기 거부 + check_branch 묶음 검사)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: 평가 스크립트 — `allow_sim_mismatch` 전달·restore→env 순서·diag 게이트 문구

**Files:**
- Modify: `Python/eval/eval_ckpt.py` (restore_policy 호출부 — `grep -n "restore_policy(" eval/eval_ckpt.py`), `Python/eval/diag_ckpt.py` :90-93(restore/env), :153-154(게이트 note), argparse
- Modify: `Python/eval/eval_mixed.py` :123-133, `Python/eval/measure_regimes.py` :72-82, `Python/eval/corridor_run.py` :87-101

- [ ] **Step 1: eval_ckpt · diag_ckpt**

각 argparse 에 `ap.add_argument('--allow_sim_mismatch', action='store_true', help='dyn_profile/obstacles/radar_range 가 스냅샷과 달라도 진행(스냅샷 값 강제 적용, 교차평가 전용)')` 추가하고 `restore_policy(...)` 호출에 `allow_sim_mismatch=args.allow_sim_mismatch,` 를 넘긴다. `diag_ckpt.py:154` 의 note 를 `'restore_policy 가 comm_range/arm/dyn_profile/obstacles/radar_range 불일치에서 중단함'` 으로. 두 스크립트 모두 env 는 `make_env_from_snapshot` 경유이므로 순서 변경 불필요(확인: `grep -n "VesselBatchEnv(" eval/eval_ckpt.py eval/diag_ckpt.py` 가 0건).

- [ ] **Step 2: 직접 env 를 만드는 3개 — restore 를 env 앞으로**

`eval_mixed.py`: `env = vg.VesselBatchEnv(...)` 블록(:123-126)을 잘라 `args.max_partners = _r.max_partners` 줄 **뒤**로 옮긴다. `restore_policy(...)` 호출에 `allow_sim_mismatch=os.environ.get('VESSEL_ALLOW_SIM_MISMATCH', '0') == '1',` 추가.
`measure_regimes.py`: 같은 방식 — `env = ...`(:72-75) 을 `policy = restore_policy(...).policy` 뒤로, 같은 kwarg 추가.
`corridor_run.py`: `env = vg.VesselBatchEnv(...)` 와 `env.obstacles = torch.zeros(...)` 두 줄(:87-90)을 `policy = restore_policy(...).policy` 뒤로, 같은 kwarg 추가. (회랑은 스스로 장애물을 비우므로 `OBSTACLES_MODE` 와 무관 — 주석 한 줄 남김.)
각 이동 위치에 주석: `# ★2026-09-21: restore_policy 가 vessel_gym 동역학 전역을 스냅샷으로 덮어쓰므로 env 는 그 *뒤*에 만든다.`

- [ ] **Step 3: 검증**

Run: `python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in ['eval/eval_ckpt.py','eval/diag_ckpt.py','eval/eval_mixed.py','eval/measure_regimes.py','eval/corridor_run.py']]; print('syntax ok')"`
Run(스냅샷 없는 옛 체크포인트가 그대로 열리는지 — 회귀 확인): `VESSEL_COMM_RANGE=420 VESSEL_CKPT_DIR=/Users/seunghyun/Dropbox/Private_Paper_Project/0702_NewVessel/checkpoints python3 eval/diag_ckpt.py --ckpt ql_SE_START_s42.pt --envs 2 --burn 5 --collect 5 --allow_comm_range_mismatch --device cpu 2>&1 | grep -E "dyn=agile|legacy|게이트" | head -5` → `dyn=agile obst=grid3x3` 헤더와 legacy note 가 보이고 SystemExit 없음(게이트 FAIL 은 조우율 때문이라 무관).
Run: `VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CKPT_DIR=/tmp python3 eval/eval_ckpt.py --ckpt imo_smoke.pt --arm OFF --envs 2 --eval_decisions 20 --burnin 5 2>&1 | grep -c "dyn=imo obst=none"` → `1`.

- [ ] **Step 4: Commit**

```bash
git add Python/eval/eval_ckpt.py Python/eval/diag_ckpt.py Python/eval/eval_mixed.py Python/eval/measure_regimes.py Python/eval/corridor_run.py
git commit -m "fix(eval): allow_sim_mismatch 전달, 직접 env 3곳은 restore 뒤에 생성, diag 게이트 문구

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: run_repro.sh · smoke_mac.sh · test_golden — 프로필 관통

**Files:**
- Modify: `Python/run_repro.sh` `common_env` :88-120, 헤더 echo :77-84, `arm_spec` :257-266, `branch_batch` 에러 문구 :283, `eval` 모드 :376-402, 헤더 주석 :30-35
- Modify: `Python/smoke_mac.sh` :15-18(동기화 diff 범위 주석), `common_env` :37-69
- Modify: `Python/verify/test_golden.py` `BATCH_ENV` :48-59, `CASES` :61-67, `_YUGIOH_CONSTS` :189-194

- [ ] **Step 1: common_env (두 파일 동일하게)**

`export VESSEL_MSG_GATE_APPLY=0` 다음, 닫는 `}` 앞에:
```bash
  # ★2026-09-21 동역학 프로필·시나리오 — 바깥에서 준 값을 보존(기본 agile/grid3x3 = 비트동일).
  #   imo 배치는 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none 을 밖에서 주고, 별도 VESSEL_CKPT_DIR/VESSEL_OUT_DIR 을 쓴다.
  #   preflight 드리프트 검사(names) 대상이 아니다 — 의도된 override 이므로. 학습기·check_branch 가 갈래 간 일치를 강제한다.
  export VESSEL_DYN_PROFILE="${VESSEL_DYN_PROFILE:-agile}"
  export VESSEL_OBSTACLES="${VESSEL_OBSTACLES:-grid3x3}"
```
run_repro 헤더 echo(`echo "  분기점 ..."` 뒤)에 `echo "  프로필      : dyn=${VESSEL_DYN_PROFILE:-agile} obstacles=${VESSEL_OBSTACLES:-grid3x3}"`.
smoke_mac.sh 의 동기화 주석 줄 번호를 실제에 맞게 갱신하고 `diff <(sed -n 'A,Bp' run_repro.sh) <(sed -n 'C,Dp' smoke_mac.sh)` 가 빈 출력인지 확인.

- [ ] **Step 2: arm_spec · eval**

`arm_spec` case 에 `on2)   echo "ON 2" ;;` 와 `off2)  echo "OFF 2" ;;` 추가, `*) return 1` 위. `branch_batch` 의 `echo "모르는 팔: $a (off|on6|on12|off12|rand)"` → `(off|on6|on12|off12|on2|off2|rand)`. 헤더 주석 `VESSEL_TRAIN_ARMS` 설명에 `on2/off2(dim 2 짝)` 추가.
`eval` 모드: `for nm in off on6 off12 on12; do` → `for nm in off on6 off12 on12 off2 on2; do`; eval_one 호출 블록에
```bash
      [ -f "$CK/on2_s$s.pt" ]  && eval_one on2  ON  2  "$s"
      [ -f "$CK/off2_s$s.pt" ] && eval_one off2 OFF 2  "$s"
```
추가.

- [ ] **Step 3: test_golden 명시 핀**

`BATCH_ENV` dict 끝에 `'VESSEL_DYN_PROFILE': 'agile', 'VESSEL_OBSTACLES': 'grid3x3',   # ★2026-09-21 프로필 핀(골든 생성 당시 값)` 추가. `CASES` 의 `default_ON`/`default_OFF` 는 `env={}` 그대로(기본값 = agile 검증이 목적). `_YUGIOH_CONSTS` 끝에 `'DYN_PROFILE', 'OBSTACLES_MODE'` 추가.
`grep -n "def run_case" -A 25 verify/test_golden.py` 로 학습 서브프로세스 env 구성을 확인: 상속 env 에서 `VESSEL_*` 를 지우는지. **지우지 않으면** `run_case` 의 env 조립에서 `{k: v for k, v in os.environ.items() if not k.startswith('VESSEL_')}` 를 베이스로 쓰도록 고친다(바깥에 `VESSEL_DYN_PROFILE=imo` 가 있어도 default 케이스가 agile 을 검증해야 하므로).

- [ ] **Step 4: 검증**

Run: `bash smoke_mac.sh` → comm 미러·골든·fidelity 전부 PASS.
Run: `VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none bash smoke_mac.sh` → 동일하게 PASS(골든 케이스는 agile 핀이라 PASS, fidelity 는 imo 로 PASS).
Run: `python3 verify/test_golden.py --check` 뒤 `python3 -c "import sys; sys.path.insert(0,'verify'); import test_golden as t; print(t.check_defaults_equal_yugioh())"` → `[]`.
Run: `bash -n run_repro.sh && bash -n smoke_mac.sh` → 문법 OK.

- [ ] **Step 5: Commit**

```bash
git add Python/run_repro.sh Python/smoke_mac.sh Python/verify/test_golden.py
git commit -m "feat(dyn): run_repro/smoke_mac 프로필 export(바깥 값 보존)·on2/off2 팔, 골든 프로필 핀·YUGIOH 대조 키

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: 문서 — CLAUDE.md 규약 · WINDOWS_RUN.md 실행 절차

**Files:**
- Modify: `.claude/CLAUDE.md` §5 "키에 영향 없는 옵션" 목록, §7 토글 표, §8 스냅샷 키 문장
- Modify: `WINDOWS_RUN.md` (끝에 절 추가)

- [ ] **Step 1: CLAUDE.md**

§5 "키에 영향 없는 옵션 = 조용히 다른 실험" 목록에 `VESSEL_DYN_PROFILE` · `VESSEL_OBSTACLES` 추가. §7 표에 두 행:
```
| `VESSEL_DYN_PROFILE` | `DYN_PROFILE` | **agile** | config 끝 | 동역학 프로필. `imo` = 선회직경 4 L 고정(절대속도 식)·타속 3°/s·정지 5 L·보상 시간상수 ×2.27(`config.dyn_profile_constants`). 스냅샷 `dyn_profile`+`dyn` 이 유일 근거. 재개·분기·평가는 일치 강제(`allow_sim_mismatch`) — 스펙 `docs/superpowers/specs/2026-09-19-dyn-profile-imo-design.md` |
| `VESSEL_OBSTACLES` | `OBSTACLES_MODE` | **grid3x3** | config 끝 | `none` = open-sea(장애물 0, 벽만). 스냅샷 `obstacles` |
```
§8 "스냅샷 키" 문장에 `2026-09-21: dyn_profile·dyn·obstacles·radar_dropout_p/len·los_gate·max_episode_steps 추가` 한 줄.

- [ ] **Step 2: WINDOWS_RUN.md 끝에 절 추가**

```markdown
## imo open-sea 파일럿 (2026-09-21, feat/dyn-profile-imo) — Git Bash

전제: GitHub clone(Dropbox `.git` 아님). 체크포인트·출력은 agile 배치와 **별도 폴더**.

```bash
git fetch origin && git checkout feat/dyn-profile-imo && git pull
cd Python
# 0) preflight + 기본(agile) 스모크 — 비트동일 확인
bash run_repro.sh smoke
# 1) imo open-sea 스모크
export VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none
export VESSEL_CKPT_DIR=$HOME/VESSEL_checkpoints/imo_opensea VESSEL_OUT_DIR=$PWD/_repro_out_imo
bash run_repro.sh smoke
# 2) 파일럿 학습: trunk(OFF 9,043,968) → off / on6 갈래, 3시드, 통신 텔레메트리 ON
VESSEL_TRAIN_ARMS="off on6" VESSEL_COMM_TELEMETRY=1 bash run_repro.sh train
# 3) 난수 대조군: on6 갈래의 msg_sd 를 diag 로 읽어 sd 로 준다
VESSEL_DIAG_CKPTS="on6_s43.pt on6_s44.pt on6_s45.pt" bash run_repro.sh diag      # _repro_out_imo/diag_on6_s4x.json → telemetry.msg_sd
VESSEL_MSG_RANDOM_SD=<msg_sd 평균> bash run_repro.sh random
# 4) 평가 (분기 검사 자동)
bash run_repro.sh eval
# 5) 차원 스윕 (파일럿 통과 시): dim 2·12 짝
VESSEL_TRAIN_ARMS="off2 on2 off12 on12" bash run_repro.sh train && bash run_repro.sh eval
# 6) coastal 보조: VESSEL_OBSTACLES=grid3x3 로 1)~4) 를 다른 CKPT/OUT 폴더에서
```

판정 기준·지표 = 스펙 §4(사전등록). 결과 표는 `eval_*.txt` 의 goal/vColl/fuel/headTravel/minSep/colregs/colregsOK + 시드별 승패. 결과 보고 기준 바꾸지 말 것.
```

- [ ] **Step 3: Commit + push**

```bash
git add .claude/CLAUDE.md WINDOWS_RUN.md
git commit -m "docs(dyn): CLAUDE.md 토글·스냅샷 규약, WINDOWS_RUN.md imo open-sea 파일럿 절차

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin feat/dyn-profile-imo
```

---

### Task 9: 최종 게이트 (스펙 §7)

- [ ] **Step 1: 전부 한 번에**

```bash
cd /Users/seunghyun/Dropbox/Private_Paper_Project/0702_NewVessel/_dev_dyn/Python
python3 verify/test_dyn_profile.py && \
python3 verify/test_golden.py --check && \
python3 verify/_verify_comm_mirror.py && \
python3 verify/test_vessel_gym_fidelity.py && \
VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none python3 verify/test_vessel_gym_fidelity.py && \
bash smoke_mac.sh && \
VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none bash smoke_mac.sh && echo "GATES: ALL PASS"
```
Expected: 마지막 줄 `GATES: ALL PASS`. 하나라도 FAIL 이면 해당 태스크로 돌아간다(추측 수정 금지).

- [ ] **Step 2: 스냅샷 왕복·분기 거부 재확인** (Task 4 Step 6, Task 5 Step 4 명령 재실행) → 동일 결과.

- [ ] **Step 3: `git status` 깨끗, `git log --oneline -9` 에 태스크 커밋 8개, `git push origin feat/dyn-profile-imo` 완료.**

---

## 범위 밖 (후속 계획)

- C# 미러 + open-sea 씬: `docs/superpowers/plans/2026-09-21-dyn-profile-imo-unity.md` (Windows).
- earlyAvoid/저속 게이트 300 m 확장, 조우 프로필 지표(첫 변침 거리·최대 타각), 밴드 회피 텔레메트리 — 파일럿 결과 뒤.
- imo 골든 케이스(`--regen` 승인).
