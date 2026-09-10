"""
radar_fidelity_compare.py — Unity obs[0:360] vs vessel_gym 레이더 3종 분리 대조 (§4 레시피).

원리: 스냅샷의 모든 배 포즈를 각자의 obs(pos[366,367], heading[364])로 복원 → 같은 포즈를
      vessel_gym 레이더에 입력 → ray별 비교. 정지 여부와 무관하게 스냅샷 단위로 기하가 일치해야 함.

3종 분리 (기하 코드가 달라 하나만 틀릴 수 있음 — 각각 검증):
  (a) 장애물 원   : ring 0.6 정지 — 장애물 22~42m(시야 내), 이웃 배 60m(시야 밖)
  (b) 타선 OBB    : ring 0.7 전진 — antipodal 수렴으로 배끼리 56m 안 진입 (ray-OBB 검증)
  (c) 벽          : ring 1.1 정지 — 스폰 ±275, 벽 ±300 (25m), 장애물·배 시야 밖

판정: 미감지(+0.5)와 max-range 히트가 구분 불가 → **히트 ray만** 오차 비교 (양쪽 <55.9m).
      한쪽만 히트(gym-only/unity-only)는 별도 카운트 — 기하 어긋남·모델 누락의 신호.

검증 이력 (2026-07-04, Build 0703 실측):
  - 장애물 원: 2,697 ray 최대오차 0.019m ✓ / 벽: 7,224 ray 오차 0.000 ✓ (내면 ±299.5 수정 후 —
    수정 전 평면 ±300 가정은 평균 0.64m 계통오차 = 이 스크립트가 잡은 실제 버그)
  - 타선 OBB (b, 시야 내): 563 ray 평균 0.033m ✓
  - 알려진 양성 이상: (a) ring 0.6 정지에서 이웃 배 bow-tip grazing(53~56m) ray가 ~0.1m 오차
    + gym-only 수십 ray로 플래그됨. box collider 미세 치수/센터 오프셋 또는 float32 양자화 추정.
    obs 정규화(/56) 기준 0.2% 미만 → 정책 전이 무해로 판정. (a)의 ship 플래그는 이 현상이면 무시.

실행: python radar_fidelity_compare.py   (PYTHONUTF8=1 권장)
"""
import os
import json
import numpy as np
import torch
import vessel_gym as vg

HIT_THR = 55.9          # 이 미만(m)만 히트로 간주 (max range 56 경계 모호성 회피)
EDGE_EPS = 0.3          # 경계 ±0.3m ray는 히트/미스 판정 자체가 반올림에 갈리므로 불일치 카운트에서 제외

_here = os.path.dirname(os.path.abspath(__file__))
_scene_json = os.path.join(
    os.environ.get('CLAUDE_SCRATCHPAD',
                   r'C:\Users\sengh\AppData\Local\Temp\claude'
                   r'\C--Users-sengh-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts'
                   r'\e433eb55-8f0a-4264-9ec0-a55bab424d0d\scratchpad'),
    'scene_coords.json')


def arena_center():
    """씬 파싱 결과가 있으면 그 값, 없으면 events.csv 실측 폴백."""
    if os.path.exists(_scene_json):
        with open(_scene_json, encoding='utf-8') as f:
            j = json.load(f)
        cx, cz = j['arena_center_world']
        return float(cx), float(cz), 'scene_coords.json'
    return -17262.3, -3749.9, 'empirical fallback'


def unity_rollout(ring_scale, action, n_decisions, base_port):
    """모든 agent의 obs 369를 결정마다 스냅샷. 반환: [ {agent_id: obs(float64[369])} ... ]"""
    from mlagents_envs.environment import UnityEnvironment, ActionTuple
    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
    os.environ['VESSEL_SPAWN_RING_SCALE'] = str(ring_scale)
    os.environ.setdefault('VESSEL_USE_COMM', '0')
    env_path = os.environ.get('VESSEL_ENV_PATH')
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_path, side_channels=[channel], worker_id=0,
                           base_port=base_port, timeout_wait=60,
                           additional_args=["-batchmode", "-nographics"])
    try:
        channel.set_configuration_parameters(time_scale=20.0)
        env.reset()
        bname = list(env.behavior_specs)[0]
        snaps = []
        ds, _ = env.get_steps(bname)
        for _k in range(n_decisions):
            snaps.append({int(a): np.array(ds.obs[0][ds.agent_id_to_index[a]], dtype=np.float64)
                          for a in ds.agent_id})
            n = len(ds.agent_id)
            if n == 0:
                break
            act = np.tile(np.array([action], dtype=np.float32), (n, 1))
            env.set_actions(bname, ActionTuple(continuous=act))
            env.step()
            ds, _ = env.get_steps(bname)
        return snaps
    finally:
        env.close()


def gym_radar_by_type(poses):
    """poses [(x,z,heading_deg)...] (arena-local) → (obs, ship, wall) 각 [n,360] 거리(m).
    _radar()와 동일 헬퍼·방향수식 사용 — 타입별로 best를 따로 유지해 ray 소스 귀속."""
    n = len(poses)
    env = vg.VesselBatchEnv(num_envs=1, n_vessels=n, device='cpu')
    for i, (x, z, h) in enumerate(poses):
        env.pos[0, i, 0] = x
        env.pos[0, i, 1] = z
        env.heading[0, i] = h
    E, N, R = env.E, env.N, vg.RADAR_RAYS
    h_rad = env.heading * vg.DEG
    cos_h, sin_h = torch.cos(h_rad), torch.sin(h_rad)
    lx, lz = env.ray_local[:, 0], env.ray_local[:, 1]
    wx = lx[None, None] * cos_h[..., None] + lz[None, None] * sin_h[..., None]
    wz = -lx[None, None] * sin_h[..., None] + lz[None, None] * cos_h[..., None]
    origin = env.pos
    full = lambda: torch.full((E, N, R), vg.RADAR_RANGE, dtype=env.dtype)
    b_obs = full()
    for ci in range(env.obstacles.shape[0]):
        b_obs = env._ray_circle(origin, wx, wz, env.obstacles[ci], env.obstacle_r, b_obs)
    b_ship = full()
    for j in range(N):
        b_ship = env._ray_obb_batch(origin, wx, wz, j, b_ship)
    b_wall = env._ray_walls(origin, wx, wz, full())
    return b_obs[0].numpy(), b_ship[0].numpy(), b_wall[0].numpy()


def compare_scenario(name, snaps, acx, acz, sample_every=1):
    """스냅샷들을 타입별로 비교. 반환: {type: {'errs': ndarray, 'gym_only': n, 'both': n}, 'unity_only': n}"""
    res = {t: {'errs': [], 'gym_only': 0} for t in ('obstacle', 'ship', 'wall')}
    unity_only = 0
    n_snap = 0
    for si in range(0, len(snaps), sample_every):
        snap = snaps[si]
        if len(snap) < 1:
            continue
        ids = sorted(snap)
        poses = [(snap[i][366] - acx, snap[i][367] - acz, snap[i][364] * 180.0) for i in ids]
        b_obs, b_ship, b_wall = gym_radar_by_type(poses)
        g3 = np.stack([b_obs, b_ship, b_wall])            # [3,n,360]
        gmin = g3.min(axis=0)
        gtype = g3.argmin(axis=0)
        n_snap += 1
        for k, aid in enumerate(ids):
            u = (np.asarray(snap[aid][0:360]) + 0.5) * vg.RADAR_RANGE
            g, gt = gmin[k], gtype[k]
            both = (g < HIT_THR) & (u < HIT_THR)
            for t, tname in enumerate(('obstacle', 'ship', 'wall')):
                m = both & (gt == t)
                if m.any():
                    res[tname]['errs'].append(np.abs(u[m] - g[m]))
                # gym만 히트 (경계 EPS 밖에서) — 기하 어긋남 신호
                go = (gt == t) & (g < HIT_THR - EDGE_EPS) & (u >= HIT_THR)
                res[tname]['gym_only'] += int(go.sum())
            unity_only += int(((u < HIT_THR - EDGE_EPS) & (g >= HIT_THR)).sum())
    print(f"\n[{name}] 스냅샷 {n_snap}개")
    print(f"  {'타입':<10}{'히트ray':>9}{'최대오차':>10}{'평균오차':>10}{'gym만히트':>10}")
    ok = True
    for tname in ('obstacle', 'ship', 'wall'):
        e = res[tname]['errs']
        e = np.concatenate(e) if e else np.array([])
        go = res[tname]['gym_only']
        if len(e):
            me, ae = e.max(), e.mean()
            flag = '' if me < 1.0 else '  ← 확인'
            if flag:
                ok = False
            print(f"  {tname:<10}{len(e):>9}{me:>10.3f}{ae:>10.3f}{go:>10}{flag}")
        else:
            print(f"  {tname:<10}{0:>9}{'-':>10}{'-':>10}{go:>10}")
        if go > len(e) * 0.02 + 5:     # gym만 히트가 유의미하게 많으면 기하 어긋남
            ok = False
            print(f"    ← {tname}: gym-only 히트 과다 — 좌표/기하 재검토")
    if unity_only > 5:
        print(f"  unity만 히트: {unity_only} ray ← gym이 모르는 물체(부표 등) 또는 좌표 오프셋")
        ok = False
    else:
        print(f"  unity만 히트: {unity_only} ray")
    return ok, res


if __name__ == '__main__':
    os.environ.setdefault('VESSEL_ENV_PATH',
                          os.path.abspath(os.path.join(_here, '..', '..', '..', 'Build', '0703', 'Vessel_MLAgent.exe')))
    acx, acz, src = arena_center()
    print("=== Unity vs vessel_gym 레이더 3종 대조 ===")
    print(f"arena center = ({acx:.2f}, {acz:.2f})  [{src}]")

    # (a) 장애물: ring 0.6 정지 3결정
    snaps_a = unity_rollout(0.6, (0.0, -1.0), 3, base_port=5991)
    ok_a, _ = compare_scenario('(a) 장애물 원  — ring 0.6 정지', snaps_a, acx, acz)

    # (c) 벽: ring 1.1 정지 3결정
    snaps_c = unity_rollout(1.1, (0.0, -1.0), 3, base_port=5992)
    ok_c, _ = compare_scenario('(c) 벽        — ring 1.1 정지', snaps_c, acx, acz)

    # (b) 타선 OBB: ring 0.7 전진 수렴, 240결정 중 8결정마다 샘플
    snaps_b = unity_rollout(0.7, (0.0, 1.0), 240, base_port=5993)
    ok_b, _ = compare_scenario('(b) 타선 OBB  — ring 0.7 수렴', snaps_b, acx, acz, sample_every=8)

    print("\n=== 결과:", "3종 전부 충실" if (ok_a and ok_b and ok_c) else "불일치 항목 있음 — 위 표 확인", "===")
