"""test_reward_v3.py — 보상 v3 토글(2026-09-30, 스펙 2026-09-30-reward-v3-decode-sweep-design.md §3) 단위 테스트.

  python verify/test_reward_v3.py        # ALL PASS 여야 함 (imo·none 고정, CPU)

v1  토글 기본값(FORWARD 0.1·TIME 0.07·GATE 0·JUDGE end)이면 30결정 롤아웃의 보상·obs·done 이 옛 식 재계산과 torch.equal
    (= 기본값 비트동일. risk_rw 는 게이트 끄면 risk 와 같은 객체)
v2  FORWARD_COEF=0 → 결정당 보상 차 = 정확히 −0.1·sr·SUBSTEPS (다른 항 불변)
v3  TIME_PENALTY=0.035 → 결정당 보상 차 = 정확히 +0.035·SUBSTEPS
v4  RISK_DCPA_GATE_M=48 → risk_rw = risk·clamp((48−dcpa)/24,0,1): dcpa ≤ 24 이면 같음, ≥ 48 이면 0, 30 이면 0.75·risk.
    near_risk·sit·danger_idx·risk 키는 불변(obs·situation 비트동일)
v5  게이트 켠 상태의 보상 차 = 충돌코스(#6)·per-pair(#7) 항 차이만(손으로 재계산과 일치)
v6  ROLE_JUDGE=v2 는 판정기 클래스만 바꾸고(RolePromiseTrackerV2) PEN=0 이면 보상 비트동일
v7  sim 스냅샷 키 4개가 있고 ckpt_io.apply_sim_snapshot 이 vessel_gym 전역을 덮어씀
"""
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

RES = []


def check(name, ok, info=''):
    RES.append(bool(ok))
    print(f"  {'PASS' if ok else '★FAIL'}  {name}  {info}")


def make_env(E=4, N=16, seed=0):
    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device='cpu', seed=seed, ring_scale=1.0, crossing=0,
                            risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE,
                            farfield_coef=0.0, perpair_coef=-0.15, perpair_exp=3.0)
    env.reset()
    return env


def rollout(steps=30, seed=0, **over):
    """토글을 vessel_gym 전역에 직접 걸고(스냅샷 복원과 같은 경로) 규칙 배(목표 조향 + 양보 우현)로 굴린다."""
    saved = {k: getattr(vg, k) for k in over}
    for k, v in over.items():
        setattr(vg, k, v)
    try:
        env = make_env(seed=seed)
        rews, obs_l, pws = [], [], []
        g = torch.Generator().manual_seed(seed)
        for t in range(steps):
            tg = env.goal - env.pos
            a0 = torch.clamp(vg._wrap180(torch.atan2(tg[..., 0], tg[..., 1]) / vg.DEG - env.heading) / 20.0, -1, 1)
            a0 = torch.where(env.situation > 0, torch.full_like(a0, 0.6), a0)
            a0 = a0 + (torch.rand(a0.shape, generator=g) - 0.5) * 0.2
            obs, r, d, oc = env.step(torch.stack([a0, torch.ones_like(a0)], -1))
            rews.append(r.clone()); obs_l.append(obs.clone())
            pws.append({k: (v.clone() if torch.is_tensor(v) else v) for k, v in env._last_pw.items()})
        return torch.stack(rews), torch.stack(obs_l), pws, env
    finally:
        for k, v in saved.items():
            setattr(vg, k, v)


def v1_defaults_bit_identical():
    r0, o0, p0, _ = rollout()
    r1, o1, p1, _ = rollout(FORWARD_COEF=0.1, TIME_PENALTY=0.07, RISK_DCPA_GATE_M=0.0, ROLE_JUDGE='end')
    check('v1 토글 기본값 = 비트동일(보상·obs)', torch.equal(r0, r1) and torch.equal(o0, o1))
    _, _, _, e_last = rollout()
    check('v1 게이트 끄면 risk_rw 는 risk 와 같은 객체', e_last._last_pw['risk_rw'] is e_last._last_pw['risk'])
    check('v1 기본값 확인', cfg.FORWARD_COEF == 0.1 and cfg.TIME_PENALTY == 0.07 and cfg.RISK_DCPA_GATE_M == 0.0
          and cfg.ROLE_JUDGE == 'end', f"{cfg.FORWARD_COEF} {cfg.TIME_PENALTY} {cfg.RISK_DCPA_GATE_M} {cfg.ROLE_JUDGE}")


def v2_forward():
    r0, o0, p0, e0 = rollout()
    r1, o1, p1, e1 = rollout(FORWARD_COEF=0.0)
    # 같은 행동·같은 상태(보상만 다름) → obs 동일, 보상 차 = 0.1·sr·SUBSTEPS. sr 은 step 뒤 속도(보상 계산 시점)
    check('v2 FORWARD 0: 상태 궤적 동일', torch.equal(o0, o1))
    # 보상 차 재계산: _reward 는 step 뒤 speed 로 sr 을 잰다 → obs[362] = speed/max_speed... 대신 env 재구성 없이 사전에 저장 불가 →
    #   차이의 형태로 검사: 모든 결정에서 r0 - r1 ≥ 0 이고 max ≤ 0.1·SUBSTEPS, 첫 결정 차이 = 0.1·sr0·SUBSTEPS(sr 은 obs 로부터)
    diff = r0 - r1
    sr = o0[..., 362]                                    # speed_ratio (obs[362] = speed/max_speed) 결정 뒤
    expect = 0.1 * sr * vg.SUBSTEPS
    check('v2 보상 차 = 0.1·sr·SUBSTEPS (전 결정)', torch.allclose(diff, expect, atol=1e-5), f"max|Δ|={float((diff-expect).abs().max()):.2e}")


def v3_time():
    r0, o0, _, _ = rollout()
    r1, o1, _, _ = rollout(TIME_PENALTY=0.035)
    diff = r1 - r0
    check('v3 TIME 0.035: 상태 동일 · 보상 차 = +0.035·SUBSTEPS', torch.equal(o0, o1)
          and torch.allclose(diff, torch.full_like(diff, 0.035 * vg.SUBSTEPS), atol=1e-5),
          f"min={float(diff.min()):.4f} max={float(diff.max()):.4f}")


def v4_v5_gate():
    r0, o0, p0, _ = rollout()
    r1, o1, p1, _ = rollout(RISK_DCPA_GATE_M=48.0)
    check('v4 게이트: obs·situation 비트동일', torch.equal(o0, o1))
    ok_keys = all(torch.equal(a['risk'], b['risk']) and torch.equal(a['near_risk'], b['near_risk']) and torch.equal(a['sit'], b['sit'])
                  for a, b in zip(p0, p1))
    check('v4 게이트: risk·near_risk·sit 키 불변', ok_keys)
    ok_g = True
    for a in p1:
        g = torch.clamp((48.0 - a['dcpa']) / 24.0, 0.0, 1.0)
        ok_g &= torch.allclose(a['risk_rw'], a['risk'] * g, atol=1e-6)
    check('v4 risk_rw = risk·clamp((48−dcpa)/24,0,1)', bool(ok_g))
    # 손 계산 예: dcpa 24 → g 1, 30 → 0.75, 48 → 0
    g = lambda d: float(torch.clamp((48.0 - torch.tensor(d)) / 24.0, 0, 1))
    check('v4 g(24)=1 g(30)=0.75 g(48)=0', g(24.0) == 1.0 and abs(g(30.0) - 0.75) < 1e-6 and g(48.0) == 0.0)
    # v5 보상 차 = #6 + #7 의 차이만: Δ = SUBSTEPS·[ −0.8(max_rw³ − max³)·1[>0.05] − 0.15·Σ(rw³·1[rw>0.05] − r³·1[r>0.05]) ]
    ok_r = True
    worst = 0.0
    for t in range(len(p0)):
        a, b = p0[t], p1[t]
        m0, m1 = a['risk'].max(-1).values, b['risk_rw'].max(-1).values
        c6 = torch.where(m1 > 0.05, -0.8 * m1 ** 3, torch.zeros_like(m1)) - torch.where(m0 > 0.05, -0.8 * m0 ** 3, torch.zeros_like(m0))
        rj0, rj1 = a['risk'], b['risk_rw']
        s0 = torch.where(rj0 > 0.05, rj0.clamp(max=1.0) ** 3, torch.zeros_like(rj0)).sum(-1)
        s1 = torch.where(rj1 > 0.05, rj1.clamp(max=1.0) ** 3, torch.zeros_like(rj1)).sum(-1)
        c7 = -0.15 * (s1 - s0)
        expect = (c6 + c7) * vg.SUBSTEPS
        d = (r1[t] - r0[t]) - expect
        worst = max(worst, float(d.abs().max()))
        ok_r &= bool(d.abs().max() < 1e-4)
    check('v5 게이트 보상 차 = #6·#7 차이만(손 계산)', bool(ok_r), f"max|Δ|={worst:.2e}")
    any_diff = any(not torch.equal(a['risk_rw'], a['risk']) for a in p1)
    check('v5 게이트가 실제로 뭔가 바꿈(공허 통과 방지)', any_diff)


def v6_judge_class():
    r0, o0, _, e0 = rollout()
    r1, o1, _, e1 = rollout(ROLE_JUDGE='v2')
    check('v6 ROLE_JUDGE=v2 · PEN 0: 보상·obs 비트동일', torch.equal(r0, r1) and torch.equal(o0, o1))
    saved = vg.ROLE_JUDGE
    try:
        vg.ROLE_JUDGE = 'v2'
        e = make_env(E=1, N=4)
        t = e.enable_role_tracker()
        check('v6 v2 → RolePromiseTrackerV2', isinstance(t, vg.RolePromiseTrackerV2) and e._rp_judge == 'v2')
        lg = e.enable_legacy_role_tracker()
        check('v6 legacy 판정기 = end 클래스(별도 객체)', type(lg) is vg.RolePromiseTracker and lg is not t)
        t2 = e.enable_role_tracker(judge='end')
        check('v6 judge=end 강제 → 교체', type(t2) is vg.RolePromiseTracker and e._rp_judge == 'end'
              and e.enable_legacy_role_tracker() is None)
    finally:
        vg.ROLE_JUDGE = saved


def v7_snapshot_keys():
    import ckpt_io
    keys = ('FORWARD_COEF', 'TIME_PENALTY', 'RISK_DCPA_GATE_M', 'ROLE_JUDGE')
    check('v7 SIM_SNAPSHOT_KEYS 에 4키', all(k in cfg.SIM_SNAPSHOT_KEYS for k in keys) and len(cfg.SIM_SNAPSHOT_KEYS) == 33)   # 31 + 2026-10-07 판정기 토글 2
    snap = {'dyn_profile': cfg.DYN_PROFILE, 'obstacles': cfg.OBSTACLES_MODE, 'dyn': cfg.dyn_profile_constants(cfg.DYN_PROFILE),
            'radar_range': float(cfg.RADAR_RANGE), 'sim': dict(cfg.sim_constants())}
    snap['sim'].update({'FORWARD_COEF': 0.0, 'TIME_PENALTY': 0.035, 'RISK_DCPA_GATE_M': 48.0, 'ROLE_JUDGE': 'v2'})
    saved = {k: getattr(vg, k) for k in keys}
    saved_cfg = {k: getattr(cfg, k) for k in keys}
    try:
        ckpt_io.apply_sim_snapshot(snap, allow_sim_mismatch=True)
        check('v7 apply_sim_snapshot 이 vessel_gym 전역을 덮어씀',
              vg.FORWARD_COEF == 0.0 and vg.TIME_PENALTY == 0.035 and vg.RISK_DCPA_GATE_M == 48.0 and vg.ROLE_JUDGE == 'v2',
              f"{vg.FORWARD_COEF} {vg.TIME_PENALTY} {vg.RISK_DCPA_GATE_M} {vg.ROLE_JUDGE}")
    finally:
        for k in keys:
            setattr(vg, k, saved[k]); setattr(cfg, k, saved_cfg[k])


if __name__ == '__main__':
    torch.set_num_threads(1)
    print('=' * 78); print('보상 v3 토글 테스트 (imo·none, CPU)'); print('=' * 78)
    for fn in (v1_defaults_bit_identical, v2_forward, v3_time, v4_v5_gate, v6_judge_class, v7_snapshot_keys):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(f"{fn.__name__} 예외", False, f"{type(e).__name__}: {e}")
    n_ok = sum(RES)
    print(f"\n{'ALL PASS' if all(RES) else '★FAIL'}  {n_ok}/{len(RES)}")
    sys.exit(0 if all(RES) else 1)
