"""메시지 채널 진단 — 차원이 커지면 실제로 더 많은 정보를 나르는가.

한 체크포인트당 한 번 굴려 아래를 전부 잰다:
  eff_rank      실제 만들어진 메시지의 유효 랭크 (참여비). msg_dim 대비 몇 %를 쓰는지
  threat_r2     위협 라벨(메시지가 나르도록 학습된 유일한 대상) 복원 R^2
  sat           tanh 포화율 (|msg|>0.99)
  om_erank      수신자가 실제 받는 집계신호 others_msg 의 유효 랭크
  label_erank   위협 라벨 자체의 고유 차원 (채널이 아니라 과제가 정하는 천장)
  shuf_/zero_   같은 상태에서 메시지만 (섞음 / 0) 으로 바꿨을 때 평균 조타·추력 변화
                (해당 명령의 배치 표준편차로 정규화 → 1.0 = 자기 산포만큼 움직임)

사용:
  VESSEL_MSG_DIM 은 인자로 주는 --dim 이 덮어씀 (import 전에 설정해야 해서 인자로 받음)
  python _diag_msg_channel.py --ckpt <path>.pt --dim 6
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--dim', type=int, required=True, help='그 체크포인트의 VESSEL_MSG_DIM')
ap.add_argument('--envs', type=int, default=32)
ap.add_argument('--burn', type=int, default=300, help='수집 전 굴릴 결정 수')
ap.add_argument('--collect', type=int, default=200)
ap.add_argument('--seed', type=int, default=999)
ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
args = ap.parse_args()

# ★config 가 import 시점에 env 를 읽으므로 import 보다 먼저 설정해야 한다.
os.environ['VESSEL_MSG_DIM'] = str(args.dim)
for k, v in dict(VESSEL_USE_MOE='1', VESSEL_MOE_SHARED='1', VESSEL_MOE_WIDTH='1.0',
                 VESSEL_USE_COMM='1', VESSEL_RADAR_RANGE='56', VESSEL_THREAT_COEF='0.5',
                 VESSEL_POS_GROUND='1', VESSEL_USE_ATTENTION='0', VESSEL_COMM_RANGE='200',
                 VESSEL_COLREGS_MODE='unity', VESSEL_MSG_LN='1',
                 VESSEL_SIM_COLREGS_COEF='0.45').items():
    os.environ[k] = v

import torch
import config as cfg
import vessel_gym as vg
from networks import CNNPolicy
from vessel_gym_train import comm_gather, parse_obs, FrameStack, compute_own_threat

dev = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
E, N = args.envs, 16
torch.manual_seed(args.seed)

env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device=dev, seed=args.seed,
                        ring_scale=1.0, crossing=0,
                        risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE,
                        farfield_coef=0.0, perpair_coef=-0.15, perpair_exp=3.0)
_sd = torch.load(args.ckpt, map_location=dev)
_sd = _sd.get('model_state_dict', _sd)
policy = CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES).to(dev)
policy.load_state_dict(_sd)
policy.eval()

fs = FrameStack(E, N, dev)
obs = env.reset()
radar, goal, self_s, sit = parse_obs(obs)
fs.reset_all(radar)

gen = torch.Generator(device='cpu').manual_seed(args.seed + 3)
MSG, OM, TH, TM, ACT, D_SH, D_Z, NPART = [], [], [], [], [], [], [], []


def mean_action(x, goal, self_s, om, sit):
    """확률적 표본이 아니라 평균 행동 — 메시지만 바꾼 차이를 잡음 없이 재려고."""
    _, mean, _, _, _ = policy.ctr_actor._route(x, goal, self_s, om, sit)
    return mean


for t in range(args.burn + args.collect):
    with torch.no_grad():
        x = fs.get()
        om, parts = comm_gather(policy, env, x, goal, self_s, sit, cfg.MAX_COMM_PARTNERS)
        if t >= args.burn:
            perm = torch.randperm(N, generator=gen).to(dev)   # 같은 씬 안에서 수신자↔메시지 짝만 파괴
            m0 = mean_action(x, goal, self_s, om, sit)
            MSG.append(policy.msg_actor(x, goal, self_s, sit).reshape(-1, args.dim))
            OM.append(om.reshape(-1, args.dim))
            ACT.append(m0.reshape(-1, cfg.CONTINUOUS_ACTION_SIZE))
            D_SH.append((mean_action(x, goal, self_s, om[:, perm], sit) - m0)
                        .reshape(-1, cfg.CONTINUOUS_ACTION_SIZE))
            D_Z.append((mean_action(x, goal, self_s, torch.zeros_like(om), sit) - m0)
                       .reshape(-1, cfg.CONTINUOUS_ACTION_SIZE))
            thr, tmask = compute_own_threat(x.reshape(E * N, -1), cfg.THREAT_K, dev)
            TH.append(thr); TM.append(tmask)
            NPART.append(parts[3].reshape(E * N, -1).sum(-1))   # pmask [E,N,K,1] → 유효 파트너 수
        act, _, _, _ = policy.ctr_actor(x, goal, self_s, om, sit)
    obs, _, done, _ = env.step(act)
    radar, goal, self_s, sit = parse_obs(obs)
    fs.push(radar, done)


def erank(X):
    """참여비 (Σλ)²/Σλ² — 분산이 몇 개 방향에 실려 있는지."""
    Xc = X - X.mean(0, keepdim=True)
    C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    ev = torch.linalg.eigvalsh(C.float()).clamp(min=0)
    return float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12))


MSG = torch.cat(MSG); OM = torch.cat(OM)
ACT = torch.cat(ACT); DSH = torch.cat(D_SH); DZ = torch.cat(D_Z)
TH = torch.cat(TH); TM = torch.cat(TM)
with torch.no_grad():
    pred = policy.threat_decoder(MSG)
mse = float(((pred - TH).pow(2) * TM).sum() / TM.sum().clamp(min=1))
base = TH.sum(0) / TM.sum(0).clamp(min=1)
mse0 = float(((base.unsqueeze(0) - TH).pow(2) * TM).sum() / TM.sum().clamp(min=1))
a_sd = ACT.std(0).clamp(min=1e-8)

print(json.dumps(dict(
    ckpt=os.path.basename(args.ckpt), dim=args.dim, n=int(MSG.shape[0]), device=dev,
    eff_rank=round(erank(MSG), 3), width_used=round(erank(MSG) / args.dim, 3),
    sat=round(float((MSG.abs() > 0.99).float().mean()), 5),
    msg_sd=round(float(MSG.std(0).mean()), 4),
    threat_r2=round(1 - mse / mse0, 4),
    om_erank=round(erank(OM), 3), om_sd=round(float(OM.std(0).mean()), 4),
    label_erank=round(erank(TH * TM), 3),
    partners=round(float(torch.cat(NPART).mean()), 3),
    shuf_rudder=round(float(DSH[:, 0].pow(2).mean().sqrt() / a_sd[0]), 4),
    shuf_thrust=round(float(DSH[:, 1].pow(2).mean().sqrt() / a_sd[1]), 4),
    zero_rudder=round(float(DZ[:, 0].pow(2).mean().sqrt() / a_sd[0]), 4),
    zero_thrust=round(float(DZ[:, 1].pow(2).mean().sqrt() / a_sd[1]), 4),
)))
