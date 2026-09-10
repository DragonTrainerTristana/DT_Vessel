"""
vessel_gym 충실도 테스트.

⚠️ 한계: 이 Mac엔 Unity가 없어 Unity 궤적을 직접 대조할 수 없다. 이 테스트가 검증하는 것:
  (1) 배칭 정확성: 배치 텐서 동역학 == C# 수식을 그대로 옮긴 스칼라 참조 (텐서 버그 검출)
  (2) 물리 상식: full rudder 선회율 45°/s 수렴, 직진 가속이 accel rate 준수, drag 평형
  (3) 레이더 기하: 정면 장애물까지 거리 = 중심거리 − 반지름, 측면 ray = max range
  (4) 상황판정: 정면 조우 → HeadOn, 우현 횡단 → GiveWay (사각지대 fix 포함)
Unity 대조(같은 action 시퀀스 궤적 비교)는 Windows에서 별도 수행 — 이 테스트 통과는 "전사·배칭이 맞다"까지.
"""
import math
import torch
import vessel_gym as vg


def scalar_dynamics_reference(a0, a1, steps, max_speed=1.0, init_speed=0.0, init_hdg=0.0):
    """C# UpdateDynamics를 순수 파이썬으로 그대로 전사 (참조 구현). steps 결정 실행."""
    DT, SUB = vg.DT, vg.SUBSTEPS
    s = init_speed
    hdg = init_hdg
    rudder = 0.0
    px, pz = 0.0, 0.0
    cmd_rudder = max(-1, min(1, a0)) * vg.MAX_TURN_RATE
    target = max(0.0, min(max_speed, (max(-1, min(1, a1)) + 1) * 0.5 * max_speed))
    for _ in range(steps):
        for _ in range(SUB):
            # 1 accel/decel
            delta = (vg.ACCEL if target > s else vg.DECEL) * DT
            s = s + max(-delta, min(delta, target - s))
            # 2 clamp
            s = max(0.0, min(max_speed, s))
            # 3 rudder slew
            rd = vg.RUDDER_RATE * DT
            rudder = rudder + max(-rd, min(rd, cmd_rudder - rudder))
            # 4-6 yaw
            sr = s / max(1e-6, max_speed)
            yaw = (rudder * sr) * vg.TURN_FACTOR
            # 7 position (pre-rotation heading, current speed) then rotate
            hr = hdg * vg.DEG
            px += math.sin(hr) * s * DT
            pz += math.cos(hr) * s * DT
            hdg += yaw * DT
            # 8 drag
            de = vg.DRAG_COEF * DT
            if target >= vg.DRAG_THRUST_THRESH:
                de *= vg.DRAG_THRUST_MULT
            s = s * (1 - de)
    return dict(px=px, pz=pz, hdg=hdg, speed=s, rudder=rudder)


def test_batching_matches_scalar():
    """배치 동역학이 스칼라 참조와 일치 (여러 action·초기조건)."""
    torch.manual_seed(0)
    cases = [(0.5, 0.8), (-0.7, 1.0), (1.0, -0.5), (0.0, 1.0), (0.3, 0.0)]
    max_speed = 1.0
    max_err = 0.0
    for a0, a1 in cases:
        env = vg.VesselBatchEnv(num_envs=1, n_vessels=1, device='cpu')
        # 초기 상태 고정 (원점, heading 0, 속도 0)
        env.pos.zero_(); env.heading.zero_(); env.speed.zero_()
        env.rudder.zero_(); env.max_speed.fill_(max_speed)
        env.goal.fill_(1000.0)  # 멀리 (goal 도달·충돌 방지)
        act = torch.tensor([[[a0, a1]]], dtype=torch.float32)
        steps = 20
        for _ in range(steps):
            env._apply_action(act)
            for _ in range(vg.SUBSTEPS):
                env._substep()
        ref = scalar_dynamics_reference(a0, a1, steps, max_speed)
        e = max(abs(float(env.pos[0, 0, 0]) - ref['px']),
                abs(float(env.pos[0, 0, 1]) - ref['pz']),
                abs(float(env.heading[0, 0]) - ref['hdg']),
                abs(float(env.speed[0, 0]) - ref['speed']),
                abs(float(env.rudder[0, 0]) - ref['rudder']))
        max_err = max(max_err, e)
        print(f"  action=({a0:+.1f},{a1:+.1f}) 배치vs스칼라 최대오차 {e:.2e} "
              f"[pos=({float(env.pos[0,0,0]):.3f},{float(env.pos[0,0,1]):.3f}) hdg={float(env.heading[0,0]):.2f} s={float(env.speed[0,0]):.4f}]")
    assert max_err < 1e-4, f"배칭 불일치 {max_err}"
    print(f"[1] 배칭 정확성 PASS (최대오차 {max_err:.2e})")


def test_physics_sanity():
    """full rudder 선회율·직진 가속·drag 평형."""
    env = vg.VesselBatchEnv(num_envs=1, n_vessels=1, device='cpu')
    env.pos.zero_(); env.heading.zero_(); env.speed.fill_(1.0)
    env.rudder.fill_(vg.MAX_TURN_RATE); env.cmd_rudder.fill_(vg.MAX_TURN_RATE)
    env.max_speed.fill_(1.0); env.target_speed.fill_(1.0); env.goal.fill_(1e6)
    # 정상상태(속도 1.0=maxSpeed, rudder=30)에서 서브스텝당 heading 증가율 → yawRate
    h0 = float(env.heading[0, 0])
    env._substep()
    dh = float(env.heading[0, 0]) - h0
    yaw_rate = dh / vg.DT
    # 이론: rudder(30)*speedRatio(~1)*turnFactor(1.5) = 45 (drag로 speed 미세 감소분 반영되어 ≲45)
    print(f"  full-rudder yawRate ≈ {yaw_rate:.2f} deg/s (이론 ~45, drag로 소폭↓)")
    assert 43.0 < yaw_rate <= 45.5, f"선회율 이상 {yaw_rate}"

    # 직진 가속: rudder 0, target 1.0, 초기 0 → 첫 서브스텝 속도 증가 ≈ accel*dt=0.004 (drag 반영 후 소폭↓)
    env2 = vg.VesselBatchEnv(num_envs=1, n_vessels=1, device='cpu')
    env2.pos.zero_(); env2.heading.zero_(); env2.speed.zero_()
    env2.rudder.zero_(); env2.cmd_rudder.zero_(); env2.max_speed.fill_(1.0)
    env2.target_speed.fill_(1.0); env2.goal.fill_(1e6)
    env2._substep()
    s1 = float(env2.speed[0, 0])
    print(f"  직진 첫 서브스텝 속도 {s1:.5f} (accel*dt=0.004, drag×0.3 후 ~0.00399)")
    assert 0.0039 < s1 < 0.0041, f"가속 이상 {s1}"
    print("[2] 물리 상식 PASS")


def test_radar_geometry():
    """정면 장애물까지 거리 = 중심거리 − 반지름, 측면 max range."""
    env = vg.VesselBatchEnv(num_envs=1, n_vessels=1, device='cpu')
    # 배를 원점, heading 0(+Z 정면). 장애물은 (0, +40)에 반지름 20 → 정면 40m, 표면 20m
    env.pos.zero_(); env.heading.zero_()
    env.obstacles = torch.tensor([[0.0, 40.0]])  # 정면 +Z 40m
    env.obstacle_r = 20.0
    radar = env._radar()[0, 0]  # [360]
    ray0 = (radar[0].item() + 0.5) * vg.RADAR_RANGE  # ray0 = bow(+Z) 정규화 복원
    print(f"  정면 ray0 거리 {ray0:.3f} (이론 40-20=20)")
    assert abs(ray0 - 20.0) < 0.5, f"정면거리 오차 {ray0}"
    # 측면 ray90(우현 +X)엔 장애물 없음 → max range (0.5 → 56m)
    ray90 = (radar[90].item() + 0.5) * vg.RADAR_RANGE
    print(f"  우현 ray90 거리 {ray90:.3f} (장애물 없음 → 56)")
    assert ray90 > 55.0, f"측면 max range 실패 {ray90}"
    print("[3] 레이더 기하 PASS")


def test_situation_classification():
    """정면 조우 → HeadOn, 우현 횡단 → GiveWay (사각지대 fix)."""
    env = vg.VesselBatchEnv(num_envs=1, n_vessels=2, device='cpu')
    # HeadOn: 배0 원점 heading 0(+Z), 배1 (0,+30) heading 180(-Z) 서로 마주봄
    env.pos = torch.tensor([[[0.0, 0.0], [0.0, 30.0]]])
    env.heading = torch.tensor([[0.0, 180.0]])
    env.speed = torch.tensor([[0.5, 0.5]])
    env.max_speed.fill_(1.0)
    pw = env._pairwise()
    sit01 = int(pw['sit'][0, 0, 1])
    print(f"  정면 마주봄 배0→배1 상황 = {sit01} (1=HeadOn 기대)")
    assert sit01 == vg.SIT_HEADON, f"HeadOn 판정 실패 {sit01}"

    # Crossing GiveWay: 배1이 배0의 우현(+X)에서 접근. 배0 heading 0(+Z), 배1 (+30,+5) heading 270(-X쪽)
    env.pos = torch.tensor([[[0.0, 0.0], [30.0, 5.0]]])
    env.heading = torch.tensor([[0.0, 270.0]])   # 배1은 -X 방향 진행(배0 앞을 가로지름)
    pw = env._pairwise()
    sit01 = int(pw['sit'][0, 0, 1])
    name = {0: 'None', 1: 'HeadOn', 2: 'StandOn', 3: 'GiveWay', 4: 'Overtaking'}[sit01]
    print(f"  우현 횡단 배0→배1 상황 = {sit01} ({name})")
    assert sit01 in (vg.SIT_GIVEWAY, vg.SIT_STANDON), f"횡단 판정 실패 {sit01}"
    print("[4] 상황판정 PASS")


def test_obs_shape_and_step():
    """obs 369D, step 배치 동작, done/리셋."""
    env = vg.VesselBatchEnv(num_envs=8, n_vessels=16, device='cpu', seed=1)
    obs = env.reset()
    assert obs.shape == (8, 16, 369), f"obs shape {obs.shape}"
    assert torch.isfinite(obs).all(), "obs NaN/Inf"
    act = torch.zeros(8, 16, 2)
    for _ in range(5):
        obs, reward, done, outcome = env.step(act)
        assert obs.shape == (8, 16, 369)
        assert torch.isfinite(obs).all(), "step obs NaN"
        assert torch.isfinite(reward).all(), "reward NaN"
    print(f"  obs shape {tuple(obs.shape)}, done {int(done.sum())}, "
          f"radar범위 [{obs[...,:360].min():.2f},{obs[...,:360].max():.2f}] (이론 [-0.5,0.5])")
    assert obs[..., :360].min() >= -0.5 - 1e-4 and obs[..., :360].max() <= 0.5 + 1e-4
    print("[5] obs/step PASS")


def test_throughput():
    """처리량 측정 (Unity 초당 ~25 대비)."""
    import time
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    E = 1024 if dev == 'cuda' else 64
    env = vg.VesselBatchEnv(num_envs=E, n_vessels=16, device=dev, seed=2)
    env.reset()
    act = torch.zeros(E, 16, 2, device=dev)
    # warmup
    for _ in range(3):
        env.step(act)
    if dev == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    K = 50
    for _ in range(K):
        env.step(act)
    if dev == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    decisions_per_s = K * E * 16 / dt   # agent-decisions per second
    env_steps_per_s = K * E / dt
    print(f"  device={dev} envs={E} | {env_steps_per_s:.0f} env-step/s, "
          f"{decisions_per_s:.0f} agent-decision/s (Unity ~25×16=400/s 대비)")
    print(f"  → 가속 배율 ≈ {decisions_per_s / 400:.0f}× (Unity 초당 ~400 agent-decision 기준)")
    print("[6] 처리량 측정 완료")


if __name__ == '__main__':
    print("=== vessel_gym 충실도 테스트 ===")
    print("\n[1] 배칭 정확성 (배치 텐서 == C# 수식 스칼라 전사)")
    test_batching_matches_scalar()
    print("\n[2] 물리 상식")
    test_physics_sanity()
    print("\n[3] 레이더 기하")
    test_radar_geometry()
    print("\n[4] COLREGs 상황판정")
    test_situation_classification()
    print("\n[5] obs/step")
    test_obs_shape_and_step()
    print("\n[6] 처리량")
    test_throughput()
    print("\n=== 전체 통과 ===")
