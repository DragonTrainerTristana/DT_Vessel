"""
timeout 진단: timeout난 배들이 목표에서 얼마나 떨어져 있었나?
- 대부분 goal 근처(<20m) → GOAL_REACHED 반경(3m)이 가혹 = 판정 문제
- 대부분 멀리(>100m) → 진짜 교착/시간부족 = 에피소드 길이·보상 문제
사용: python diag_timeout.py --ckpt vg_colregs_s42.pt --arm OFF
"""
import os, argparse
import torch
import config as cfg
import vessel_gym as vg
from networks import CNNPolicy
from vessel_gym_train import comm_gather, parse_obs, FrameStack, make_others_msg

SCR = r"C:/Users/sengh/AppData/Local/Temp/claude/C--Users-sengh-Dropbox-Private-Paper-Project-0702-NewVessel/e8fe723f-83ff-4c13-af4f-54455574c53b/scratchpad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arm', default='OFF')
    ap.add_argument('--envs', type=int, default=96)
    ap.add_argument('--vessels', type=int, default=16)
    ap.add_argument('--decisions', type=int, default=3500)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    E, N = args.envs, args.vessels
    torch.manual_seed(999)
    p = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(SCR, args.ckpt)

    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device=dev, seed=999, ring_scale=1.0,   # ★2026-08-30: 0.7→1.0 (씬 원본)
                            crossing=2,   # ★2026-08-30 학습과 동일 반경(200m)
                            risk_range=vg.COMM_RANGE, reward_range=vg.COMM_RANGE,
                            farfield_coef=0.0, perpair_coef=-0.15, perpair_exp=3.0)
    pol = CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES).to(dev)
    sd = torch.load(p, map_location=dev)
    pol.load_state_dict(sd.get('model_state_dict', sd)); pol.eval()

    fs = FrameStack(E, N, dev)
    obs = env.reset(); radar, goal, self_s, sit = parse_obs(obs); fs.reset_all(radar)
    # 초기 spawn->goal 거리(기준)
    init_d = torch.linalg.norm(env.goal - env.pos, dim=-1).mean().item()

    to_dists, goal_ok = [], 0
    for i in range(args.decisions):
        with torch.no_grad():
            if args.arm == 'ON':
                om, _ = comm_gather(pol, env, fs.get(), goal, self_s, sit, cfg.MAX_COMM_PARTNERS)
            else:
                om = make_others_msg(env, args.arm, E, N, dev)
            a, _, _, _ = pol.ctr_actor(fs.get(), goal, self_s, om, sit)
        d_before = torch.linalg.norm(env.goal - env.pos, dim=-1)   # step 전 거리
        obs, _, done, outcome = env.step(a)
        to_mask = (outcome == vg.OUT_TIMEOUT)
        if int(to_mask.sum()):
            to_dists.append(d_before[to_mask].detach().cpu())
        goal_ok += int((outcome == vg.OUT_GOAL).sum())
        radar, goal, self_s, sit = parse_obs(obs); fs.push(radar, done)

    if not to_dists:
        print("no timeouts"); return
    d = torch.cat(to_dists)
    q = torch.quantile(d, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    print(f"=== timeout 시점 목표까지 거리 (n={len(d)}, spawn 초기거리 평균 {init_d:.0f}m) ===")
    print(f"  중앙값 {q[2]:.1f}m | 10% {q[0]:.1f} | 25% {q[1]:.1f} | 75% {q[3]:.1f} | 90% {q[4]:.1f}")
    for thr in (3, 8, 15, 30, 60):
        print(f"  goal {thr:3d}m 이내에서 timeout: {100.0*float((d <= thr).float().mean()):5.1f}%")
    print(f"  goal 도달 성공: {goal_ok} eps")
    print(f"\n판정: ", end='')
    near = float((d <= 15).float().mean())
    if near > 0.3:
        print(f"목표 근처 timeout이 {near*100:.0f}% → GOAL_REACHED 반경(3m)이 가혹 = 반경 완화 유효")
    else:
        print(f"목표 근처 timeout {near*100:.0f}%뿐 → 진짜 교착/시간부족 = 에피소드 길이·보상 조정 필요")


if __name__ == '__main__':
    main()
