"""항해 구간별 계수 측정 — 장거리 연료·시간을 시뮬 없이 합성하기 위한 재료.

왜 이게 성립하는가
------------------
vessel_gym 은 **무기억(memoryless)** 환경임. 항해 내내 누적되는 상태가 없음:
연료탱크·피로·선체열화 없음, prev_dist/prev_rudder/prev_far_risk 는 1스텝 메모리,
max_speed 는 에피소드 상수, step_count 는 timeout 판정에만 쓰임, 프레임스택은 레이더 3장.
→ 장거리 항해 = 짧은 구간의 독립적 이어붙이기. 근사가 아니라 무기억성의 귀결임.

따라서 결정을 두 국면으로 나눠 각각의 결정당 비용을 재면, 임의 거리·임의 밀도의
항해 비용을 합성할 수 있음.

  T (순항) : 반경 DETECTION_RANGE(56m) 안에 타선 없고 레이더에 장애물도 안 잡힘
             → 정책이 매 결정 같은 짓(웨이포인트 방위로 정침·정속). 통계적으로 동일.
  E (조우) : 그 외. 회피 기동이 일어나는 구간. 통신이 일하는 곳도 여기임.

재는 것 (국면별)
---------------
  n           결정 수
  fuel/dec    Σ(v_norm² + 0.5·|rudder_cmd|²) 의 결정당 평균 — eval_ckpt.py:129 와 동일 정의
  sog/dec     목표까지 거리의 감소량(m/결정) = 실효 전진속도
  ─ 추가로 ─
  조우 진입률  거리 1 m 당 T→E 전이 횟수  (밀도 ρ0 에서의 값)
  조우 지속    E 구간 1회의 평균 결정 수

★이 스크립트는 **측정만** 함. 합성은 compose_voyage.py 가 함.
"""
import argparse
import json
import os
import time

import torch

import config as cfg
import vessel_gym as vg
from networks import CNNPolicy
from vessel_gym_train import comm_gather, parse_obs, FrameStack, make_others_msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arm', default='ON', choices=['OFF', 'ORACLE', 'ON'])
    ap.add_argument('--envs', type=int, default=2)
    ap.add_argument('--vessels', type=int, default=16,
                    help='★밀도를 정하는 값. 조우율이 여기 걸리므로 본실험과 같아야 함')
    ap.add_argument('--max_partners', type=int, default=cfg.MAX_COMM_PARTNERS)
    ap.add_argument('--decisions', type=int, default=4000)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--ring', type=float, default=1.0)   # ★2026-08-30: 0.7→1.0 (씬 원본). 0.7 은 goal 8/16 이 장애물에 얹혀 도달 불가였음
    ap.add_argument('--crossing', type=int, default=0)
    ap.add_argument('--out', default='')
    ap.add_argument('--secs', type=float, default=0.0,
                    help='>0 이면 이 초를 넘기면 조기 종료(시간 예산 모드)')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    E, N = args.envs, args.vessels
    torch.manual_seed(args.seed)
    scr = os.path.dirname(os.path.abspath(__file__))
    ck_dir = os.environ.get('VESSEL_CKPT_DIR', os.path.join(scr, 'checkpoints'))
    ck = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ck_dir, args.ckpt)

    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device=dev, seed=args.seed,
                            ring_scale=args.ring, crossing=args.crossing, risk_range=vg.COMM_RANGE, reward_range=vg.COMM_RANGE,   # ★2026-08-30 학습과 동일 반경(200m)
                            farfield_coef=float(os.environ.get('VESSEL_FARFIELD_COEF', '0.0')),   # ★2026-08-30 학습과 동일
                            perpair_coef=-0.15, perpair_exp=3.0)
    # ★2026-09-10: 복원은 ckpt_io.restore_policy 로. 예전엔 msg_ln 만 스니핑하고 attention/pos_ground/radar_head/token_gain 은
    #   복원하지 않아, 그런 ckpt 는 학습과 다른 집계·인코더로 굴렀다(에러 없이). 이제 스냅샷을 전부 적용하고 헤더로 찍는다.
    #   ⚠️과거 이 스크립트가 낸 숫자가 불일치 상태였다면 재실행 시 값이 달라진다 — 그 경우 과거 값이 틀린 것.
    #   comm_range 가 ckpt 와 다르면 중단함. 의도한 것이면 VESSEL_ALLOW_COMM_RANGE_MISMATCH=1.
    from ckpt_io import restore_policy
    policy = restore_policy(ck, dev, arm=None,
                            allow_comm_range_mismatch=os.environ.get('VESSEL_ALLOW_COMM_RANGE_MISMATCH', '0') == '1',
                            tag='[measure_regimes]').policy

    fs = FrameStack(E, N, dev)
    obs = env.reset()
    radar, goal, self_s, sit = parse_obs(obs)
    fs.reset_all(radar)

    BIG = 1e9
    eye = torch.eye(N, device=dev).unsqueeze(0) * BIG

    def act(x, g, s, t):
        with torch.no_grad():
            if args.arm == 'ON':
                om, _ = comm_gather(policy, env, x, g, s, t, args.max_partners)
            else:
                om = make_others_msg(env, args.arm, E, N, dev)
            a, _, _, _ = policy.ctr_actor(x, g, s, om, t)
        return a

    def nearest_sep():
        return (torch.cdist(env.pos, env.pos) + eye).min(dim=-1).values

    def obstacle_near():
        """레이더 범위 안에 장애물 원이 걸리는가 [E,N]."""
        d = torch.linalg.norm(env.pos.unsqueeze(2) - env.obstacles.view(1, 1, -1, 2), dim=-1)
        return (d - env.obstacle_r < vg.RADAR_RANGE).any(dim=-1)

    # ── 누적기: 0=T(순항) 1=E(조우) ──
    n_dec = torch.zeros(2, device=dev, dtype=torch.float64)
    s_fuel = torch.zeros(2, device=dev, dtype=torch.float64)
    s_sog = torch.zeros(2, device=dev, dtype=torch.float64)   # 목표까지 거리 감소량 합
    s_move = torch.zeros(2, device=dev, dtype=torch.float64)  # 실제 이동거리 합
    n_enter = 0.0            # T→E 전이 횟수
    e_run = torch.zeros(E, N, device=dev)       # 현재 E 연속 길이
    e_dur_sum, e_dur_n = 0.0, 0.0
    prev_E = torch.zeros(E, N, dtype=torch.bool, device=dev)

    for _ in range(args.warmup):
        a = act(fs.get(), goal, self_s, sit)
        obs, _, done, _ = env.step(a)
        radar, goal, self_s, sit = parse_obs(obs)
        fs.push(radar, done)

    t0 = time.time()
    it = 0
    while it < args.decisions:
        it += 1
        # step 전 상태로 국면 판정
        inE = (nearest_sep() < vg.DETECTION_RANGE) | obstacle_near()      # [E,N] bool
        d_before = torch.linalg.norm(env.goal - env.pos, dim=-1)
        pos_before = env.pos.clone()
        sr = env.speed / torch.clamp(env.max_speed, min=1e-6)

        a = act(fs.get(), goal, self_s, sit)
        turn01 = a[..., 0].abs().clamp(0, 1)
        fuel = (sr ** 2 + 0.5 * turn01 ** 2).to(torch.float64)

        obs, _, done, outcome = env.step(a)
        d_after = torch.linalg.norm(env.goal - env.pos, dim=-1)
        moved = torch.linalg.norm(env.pos - pos_before, dim=-1)
        # 리스폰(텔레포트)된 배는 이번 결정 통계에서 제외
        ok = ~done
        prog = (d_before - d_after).to(torch.float64)

        for b, m in ((0, (~inE) & ok), (1, inE & ok)):
            if bool(m.any()):
                n_dec[b] += float(m.sum())
                s_fuel[b] += float(fuel[m].sum())
                s_sog[b] += float(prog[m].sum())
                s_move[b] += float(moved[m].to(torch.float64).sum())

        # 조우 진입/지속
        enter = inE & (~prev_E) & ok
        n_enter += float(enter.sum())
        e_run = torch.where(inE & ok, e_run + 1.0, e_run)
        leave = (~inE) & prev_E & ok
        if bool(leave.any()):
            e_dur_sum += float(e_run[leave].sum()); e_dur_n += float(leave.sum())
        e_run = torch.where(inE & ok, e_run, torch.zeros_like(e_run))
        prev_E = torch.where(ok, inE, torch.zeros_like(inE))

        if it % 500 == 0:
            el = time.time() - t0
            print(f'  {it}/{args.decisions}  {it*E*N/el:.0f} dec/s  '
                  f'T {int(n_dec[0])} / E {int(n_dec[1])}', flush=True)
        if args.secs > 0 and time.time() - t0 > args.secs:
            print(f'  [시간예산 {args.secs:.0f}s 도달 — {it} 결정에서 조기 종료]', flush=True)
            break

    nT, nE = float(n_dec[0]), float(n_dec[1])
    tot_move = float(s_move[0] + s_move[1])
    res = dict(
        ckpt=os.path.basename(ck), arm=args.arm, vessels=N, envs=E, ring=args.ring,
        decisions_run=it, agent_decisions=it * E * N, seconds=time.time() - t0,
        T=dict(n=nT, fuel_per_dec=float(s_fuel[0]) / max(nT, 1),
               sog_per_dec=float(s_sog[0]) / max(nT, 1),
               move_per_dec=float(s_move[0]) / max(nT, 1)),
        E=dict(n=nE, fuel_per_dec=float(s_fuel[1]) / max(nE, 1),
               sog_per_dec=float(s_sog[1]) / max(nE, 1),
               move_per_dec=float(s_move[1]) / max(nE, 1)),
        encounter=dict(n_enter=n_enter,
                       enter_per_m=n_enter / max(tot_move, 1e-9),
                       mean_dur_dec=e_dur_sum / max(e_dur_n, 1),
                       closed=e_dur_n),
        frac_E=nE / max(nT + nE, 1),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        print('저장:', args.out)


if __name__ == '__main__':
    main()
