"""대만↔한국 회랑 항해 — 양방향 통항 궤적 수집.

WorldMap.unity 실측: 부산(-82066,-24438), 대만(-79863,-18304) → 직선 6,517.6 유닛.
그 사이는 GridRegion(스케일 870×927)이 덮는 **장애물 없는 열린 바다**임(섬·벽은 전부 국지 씬에 있음).
그래서 회랑은 장애물 없는 직사각 통항로로 모형화한다.

배치: 절반은 대만 끝, 절반은 부산 끝에서 출발해 서로 반대편으로 간다(양방향 통항).
      crossing=2(대척 배정)로 각 배가 맞은편 끝을 목표로 받는다 → 정면 조우가 구조적으로 발생.

전부 실측 수집임. 곱하거나 합성하는 부분 없음.
"""
import argparse
import math
import os
import time

import numpy as np
import torch


def build_points(n_side, length, width):
    """회랑 양 끝에 n_side 개씩. 부산→대만 방위를 x축으로 놓은 좌표계."""
    half_l, half_w = length / 2.0, width / 2.0
    ys = np.linspace(-half_w, half_w, n_side)
    a = [(-half_l, float(y)) for y in ys]        # 부산 끝
    b = [(+half_l, float(y)) for y in ys]        # 대만 끝
    return tuple(a + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ql_SE_START_s42.pt')
    ap.add_argument('--arm', default='ON', choices=['OFF', 'ORACLE', 'ON'])
    ap.add_argument('--envs', type=int, default=24)
    ap.add_argument('--per_side', type=int, default=20, help='한쪽 끝 스폰 수 (총 척수 = 2배)')
    ap.add_argument('--length', type=float, default=1200.0, help='회랑 길이(m, 심)')
    ap.add_argument('--width', type=float, default=300.0, help='회랑 폭(m, 심)')
    ap.add_argument('--decisions', type=int, default=4000)
    ap.add_argument('--secs', type=float, default=900.0)
    ap.add_argument('--max_partners', type=int, default=4)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='corridor_traj.npz')
    args = ap.parse_args()

    scr = os.path.dirname(os.path.abspath(__file__))
    os.environ.setdefault('VESSEL_MSG_DIM', '6')
    os.environ.setdefault('VESSEL_USE_MOE', '1')
    os.environ.setdefault('VESSEL_MOE_SHARED', '1')
    os.environ.setdefault('VESSEL_MOE_WIDTH', '1.0')
    os.environ.setdefault('VESSEL_USE_COMM', '1')
    os.environ.setdefault('VESSEL_RADAR_RANGE', '56')
    os.environ.setdefault('VESSEL_THREAT_COEF', '0.5')
    os.environ.setdefault('VESSEL_POS_GROUND', '1')
    os.environ.setdefault('VESSEL_USE_ATTENTION', '0')
    os.environ.setdefault('VESSEL_SIM_COLREGS_COEF', '0.45')

    import vessel_gym as vg
    # ── 회랑 기하로 모듈 상수 교체 (env 생성 *전*) ──
    pts = build_points(args.per_side, args.length, args.width)
    vg.SPAWN_PTS_SCENE = pts
    vg.GOAL_PTS_SCENE = pts
    span = max(args.length, args.width) / 2.0 + 60.0
    vg.ARENA_HALF = span
    vg.ARENA_INNER = span - vg.WALL_HALF_THICK
    vg.MIN_GOAL_DIST = args.length * 0.8
    # 여정 length 에 맞춘 시간예산 (실측 0.3835 m/결정, 여유 2.6배)
    est = args.length / 0.3835
    vg.MAX_EPISODE_STEPS = int(est * 2.6) * 10
    import config as cfg
    from networks import CNNPolicy
    from vessel_gym_train import comm_gather, parse_obs, FrameStack, make_others_msg

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    E, N = args.envs, 2 * args.per_side
    torch.manual_seed(args.seed)
    print(f'회랑 {args.length:.0f} x {args.width:.0f} m | {E} env x {N} 척 = {E*N:,} 척 동시')
    print(f'  1항차 예상 {est:.0f} 결정, 시간예산 {vg.MAX_EPISODE_STEPS//10} 결정')

    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device=dev, seed=args.seed,
                            ring_scale=1.0, crossing=2, risk_range=420.0,
                            farfield_coef=0.5, perpair_coef=-0.15, perpair_exp=3.0)
    env.obstacles = torch.zeros(0, 2, device=dev, dtype=env.dtype)   # 열린 바다 — 장애물 없음

    ck_dir = os.environ.get('VESSEL_CKPT_DIR', os.path.join(scr, 'checkpoints'))
    ck = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ck_dir, args.ckpt)
    policy = CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES).to(dev)
    sd = torch.load(ck, map_location=dev)
    policy.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)
    policy.eval()

    fs = FrameStack(E, N, dev)
    obs = env.reset()
    radar, goal, self_s, sit = parse_obs(obs)
    fs.reset_all(radar)

    # ★이 맥은 numpy2 x torch 조합에서 tensor.numpy() 가 막혀 있어 torch 텐서로 모은다.
    T = args.decisions
    P = torch.zeros(T, E, N, 2, dtype=torch.float32)
    OUT = torch.zeros(T, E, N, dtype=torch.int8)
    t0 = time.time()
    for t in range(T):
        with torch.no_grad():
            if args.arm == 'ON':
                om, _ = comm_gather(policy, env, fs.get(), goal, self_s, sit, args.max_partners)
            else:
                om = make_others_msg(env, args.arm, E, N, dev)
            a, _, _, _ = policy.ctr_actor(fs.get(), goal, self_s, om, sit)
        P[t] = env.pos.detach().cpu()
        obs, _, done, outcome = env.step(a)
        OUT[t] = outcome.detach().cpu().to(torch.int8)
        radar, goal, self_s, sit = parse_obs(obs)
        fs.push(radar, done)
        if t % 400 == 0:
            el = time.time() - t0
            print(f'  {t}/{T}  {(t+1)*E*N/max(el,1e-9):,.0f} dec/s  '
                  f'경과 {el:.0f}s', flush=True)
        if args.secs > 0 and time.time() - t0 > args.secs:
            print(f'  [시간예산 도달 — {t} 결정에서 종료]')
            P = P[:t + 1]; OUT = OUT[:t + 1]
            break

    dst = args.out if os.path.isabs(args.out) else os.path.join(scr, args.out)
    if dst.endswith('.npz'):
        dst = dst[:-4] + '.pt'
    torch.save(dict(P=P, OUT=OUT, length=args.length, width=args.width,
                    n_env=E, n_vessel=N, arm=args.arm,
                    spawn=torch.tensor(pts, dtype=torch.float32)), dst)
    n_goal = int((OUT == 1).sum())
    print(f'저장: {dst}  |  shape {P.shape}  |  도착 종료 {n_goal:,}건')


if __name__ == '__main__':
    main()
