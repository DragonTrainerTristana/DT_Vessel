"""
unity_fidelity_compare.py — vessel_gym vs Unity 궤적 대조 (Windows 전용, C# 수정 불필요).

원리: obs 369D에 pos/heading/speed/rudder/radar가 이미 있음. 고정 action을 Unity에 먹이고 받은 obs를
      ground-truth 궤적으로 로깅 → 같은 초기조건·action을 vessel_gym에 넣어 스텝별 비교.

★이 Mac 세션에선 Unity 실행 불가라 미검증. Windows에서 mlagents 연결부(TODO)를 채우고 실행할 것.
  vessel_gym 쪽(replay)은 완성. 연결부는 main.py의 UnityEnvironment 사용 패턴을 그대로 가져오면 됨.

실행:
  set VESSEL_ENV_PATH=..\..\Build\0703\Vessel_MLAgent.exe
  set VESSEL_SPEED_MULT_MIN=1.0 & set VESSEL_SPEED_MULT_MAX=1.0 & set VESSEL_USE_COMM=0
  python unity_fidelity_compare.py
"""
import os
import numpy as np
import torch
import vessel_gym as vg

# 고정 action 시퀀스 (결정 단위). 동역학 여러 국면 커버.
ACTION_SEQ = (
    [(0.5, 0.8)] * 20 +     # 우현 선회 + 고추력
    [(-0.7, 1.0)] * 20 +    # 좌현 급선회 + 최대추력
    [(0.0, -0.5)] * 20 +    # 직진 + 감속
    [(1.0, 0.5)] * 20       # 최대 우현타
)


def denorm_state_from_obs(obs_vec, max_speed=1.0):
    """Unity obs 369 → 초기상태 복원 (maxSpeed 고정 전제)."""
    return dict(
        pos_x=float(obs_vec[366]),
        pos_z=float(obs_vec[367]),
        heading=float(obs_vec[364]) * 180.0,   # obs[364]=heading/180
        speed=float(obs_vec[362]) * max_speed,  # obs[362]=speed/maxSpeed
        rudder=float(obs_vec[365]) * vg.MAX_TURN_RATE,  # obs[365]=rudder/30
    )


def extract_traj_row(obs_vec):
    """비교용 스칼라 5개 (pos_x, pos_z, heading, speed, rudder) — 정규화 해제."""
    return np.array([
        obs_vec[366], obs_vec[367],
        obs_vec[364] * 180.0, obs_vec[362], obs_vec[365] * vg.MAX_TURN_RATE
    ], dtype=np.float64)


def run_vessel_gym(init_state, action_seq, max_speed=1.0):
    """vessel_gym를 init_state에서 시작해 action_seq 실행, 결정별 (pos_x,pos_z,heading,speed,rudder) 로깅."""
    env = vg.VesselBatchEnv(num_envs=1, n_vessels=1, device='cpu')
    env.pos[0, 0, 0] = init_state['pos_x']
    env.pos[0, 0, 1] = init_state['pos_z']
    env.heading[0, 0] = init_state['heading']
    env.speed[0, 0] = init_state['speed']
    env.rudder[0, 0] = init_state['rudder']
    env.max_speed[0, 0] = max_speed
    env.goal[0, 0, 0] = 1e6      # 멀리 (goal/충돌 종료 방지)
    env.goal[0, 0, 1] = 1e6
    traj = []
    for (a0, a1) in action_seq:
        act = torch.tensor([[[a0, a1]]], dtype=torch.float32)
        env._apply_action(act)
        for _ in range(vg.SUBSTEPS):
            env._substep()
        traj.append(np.array([
            float(env.pos[0, 0, 0]), float(env.pos[0, 0, 1]),
            float(env.heading[0, 0]), float(env.speed[0, 0]) / max_speed,  # speed 정규화(Unity obs와 맞춤)
            float(env.rudder[0, 0]),
        ], dtype=np.float64))
    return np.array(traj)


def run_unity(action_seq):
    """Unity 빌드에 고정 action을 먹이고 vessel 0의 obs 궤적 로깅 (main.py 연결 패턴 재사용).
    반환: (init_obs_vec[369], traj[T,5])  — traj = extract_traj_row 나열.
    추적 vessel이 중도 종료(충돌/도착)하면 그 시점까지만 잘라 반환."""
    from mlagents_envs.environment import UnityEnvironment, ActionTuple
    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

    env_path = os.environ.get('VESSEL_ENV_PATH')
    if not env_path or not os.path.exists(env_path):
        raise RuntimeError(f"VESSEL_ENV_PATH 없음/무효: {env_path}")
    base_port = int(os.environ.get('VESSEL_BASE_PORT', '5990'))
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_path, side_channels=[channel], worker_id=0,
                           base_port=base_port, timeout_wait=60,
                           additional_args=["-batchmode", "-nographics"])
    try:
        channel.set_configuration_parameters(time_scale=float(os.environ.get('VESSEL_TIME_SCALE', '20')))
        env.reset()
        behavior_name = list(env.behavior_specs)[0]
        # 빌드 함정 가드 (main.py와 동일): obs 크기 확인
        spec = env.behavior_specs[behavior_name]
        obs_len = int(sum(int(np.prod(o.shape)) for o in spec.observation_specs))
        if obs_len != 369:
            raise RuntimeError(f"[BUILD TRAP] Unity obs={obs_len}D != 369D — stale 빌드")

        ds, ts = env.get_steps(behavior_name)
        track_id = int(ds.agent_id[0])          # vessel 0 고정 추적 (agent_id 기준, 인덱스 아님)
        obs0 = np.array(ds.obs[0][ds.agent_id_to_index[track_id]], dtype=np.float64)
        traj = []
        for (a0, a1) in action_seq:
            n = len(ds.agent_id)
            act = np.tile(np.array([[a0, a1]], dtype=np.float32), (n, 1))
            env.set_actions(behavior_name, ActionTuple(continuous=act))
            env.step()
            ds, ts = env.get_steps(behavior_name)
            if track_id in ts.agent_id:
                print(f"  [WARN] 추적 vessel {track_id} 에피소드 종료(충돌/도착) — {len(traj)}결정까지만 비교")
                break
            if track_id not in ds.agent_id_to_index:
                print(f"  [WARN] 추적 vessel {track_id} decision step 부재 — {len(traj)}결정까지만 비교")
                break
            traj.append(extract_traj_row(np.array(ds.obs[0][ds.agent_id_to_index[track_id]], dtype=np.float64)))
        return obs0, np.array(traj)
    finally:
        env.close()


def compare(unity_traj, gym_traj):
    names = ['pos_x', 'pos_z', 'heading', 'speed(norm)', 'rudder']
    n = min(len(unity_traj), len(gym_traj))
    u, g = unity_traj[:n], gym_traj[:n]
    print(f"{'항목':<14}{'최대오차':>12}{'평균오차':>12}")
    ok = True
    for i, nm in enumerate(names):
        d = u[:, i] - g[:, i]
        if nm == 'heading':                       # ±180 랩어라운드 보정 (원형 차)
            d = (d + 180.0) % 360.0 - 180.0
        err = np.abs(d)
        me, ae = err.max(), err.mean()
        flag = '' if me < (5.0 if nm == 'heading' else 1.0) else '  ← 확인'
        if flag:
            ok = False
        print(f"{nm:<14}{me:>12.4f}{ae:>12.4f}{flag}")
    print("\n결과:", "충실 (전사가 Unity와 일치)" if ok else "불일치 — 해당 항 전사 재검토")
    return ok


if __name__ == '__main__':
    # 미설정 시 기본값 (문서 §3.1 결정성 확보 조건)
    _here = os.path.dirname(os.path.abspath(__file__))
    os.environ.setdefault('VESSEL_ENV_PATH',
                          os.path.abspath(os.path.join(_here, '..', '..', '..', 'Build', '0703', 'Vessel_MLAgent.exe')))
    os.environ.setdefault('VESSEL_SPEED_MULT_MIN', '1.0')
    os.environ.setdefault('VESSEL_SPEED_MULT_MAX', '1.0')
    os.environ.setdefault('VESSEL_USE_COMM', '0')
    print("=== vessel_gym vs Unity 충실도 대조 ===")
    print(f"exe = {os.environ['VESSEL_ENV_PATH']}")
    print(f"action 시퀀스 {len(ACTION_SEQ)} 결정\n")
    init_obs, unity_traj = run_unity(ACTION_SEQ)      # Windows에서 동작
    init = denorm_state_from_obs(init_obs)
    print(f"Unity 초기상태: pos=({init['pos_x']:.2f},{init['pos_z']:.2f}) "
          f"hdg={init['heading']:.2f} s={init['speed']:.3f} rud={init['rudder']:.2f}\n")
    gym_traj = run_vessel_gym(init, ACTION_SEQ)
    compare(unity_traj, gym_traj)
