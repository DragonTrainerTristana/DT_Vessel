"""통신 절제 대조 — 학습된 통신 정책을 메시지만 바꿔 평가한다.

  normal   학습된 그대로
  shuffle  같은 씬 안에서 수신자↔메시지 짝만 파괴 (분포는 동일, 내용만 무의미)
  zero     others_msg=0 (통신 OFF 팔과 같은 입력)

shuffle 이 normal 과 같으면 메시지 '내용'은 결과에 안 쓰인 것이다.
zero 는 정책이 한 번도 못 본 입력이라 분포이탈이 섞이므로 shuffle 이 더 엄밀한 대조군.

⚠️ --burn 을 2400 결정 이상 줄 것. vessel_gym.py 가 최초 리셋에서만 step_count 를
   U(0, 4500결정) 로 흩뿌리기 때문에(종료 파도 제거용), 짧은 burn-in 은 실력과 무관한
   '인공 시간초과'를 집계창에 섞는다. 실측: burn 250 → 시간초과 18%, burn 2400 → 0%.

사용:
  python _diag_msg_ablate.py --ckpt <path>.pt --dim 6 --mode shuffle
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--dim', type=int, required=True, help='그 체크포인트의 VESSEL_MSG_DIM')
ap.add_argument('--mode', required=True, choices=['normal', 'shuffle', 'zero'])
ap.add_argument('--envs', type=int, default=64)
ap.add_argument('--burn', type=int, default=2400, help='2400 미만으로 낮추지 말 것 (위 경고)')
ap.add_argument('--eval', type=int, default=1600)
ap.add_argument('--seed', type=int, default=1234, help='평가 장면 시드 — 세 조건에 같은 값을 줄 것')
ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
args = ap.parse_args()

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
from vessel_gym_train import comm_gather, parse_obs, FrameStack

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

gen = torch.Generator(device='cpu').manual_seed(args.seed + 7)
cnt = torch.zeros(5, device=dev)
ep_len = torch.zeros(E, N, device=dev)
len_sum, len_n = 0.0, 0
fuel_sum = 0.0

for t in range(args.burn + args.eval):
    with torch.no_grad():
        x = fs.get()
        om, _ = comm_gather(policy, env, x, goal, self_s, sit, cfg.MAX_COMM_PARTNERS)
        if args.mode == 'zero':
            om = torch.zeros_like(om)
        elif args.mode == 'shuffle':
            perm = torch.randperm(N, generator=gen).to(dev)
            om = om[:, perm]
        act, _, _, _ = policy.ctr_actor(x, goal, self_s, om, sit)
    obs, _, done, outcome = env.step(act)
    ep_len += 1
    if t >= args.burn:
        for c in (1, 2, 3, 4):
            cnt[c] += (outcome == c).sum()
        fin = outcome > 0
        if bool(fin.any()):
            len_sum += float(ep_len[fin].sum()); len_n += int(fin.sum())
        fuel_sum += float((act[..., 0] ** 2 + 0.5 * act[..., 1] ** 2).mean())
    ep_len = torch.where(outcome > 0, torch.zeros_like(ep_len), ep_len)
    radar, goal, self_s, sit = parse_obs(obs)
    fs.push(radar, done)

tot = float(cnt[1:].sum())
print(json.dumps(dict(
    ckpt=os.path.basename(args.ckpt), dim=args.dim, mode=args.mode, device=dev,
    eps=int(tot),
    goal=round(100 * float(cnt[1]) / max(tot, 1), 2),
    vcoll=round(100 * float(cnt[2]) / max(tot, 1), 2),
    ocoll=round(100 * float(cnt[3]) / max(tot, 1), 2),
    timeout=round(100 * float(cnt[4]) / max(tot, 1), 2),
    mean_len=round(len_sum / max(len_n, 1), 1),
    fuel_proxy=round(fuel_sum / max(args.eval, 1), 5),
)))
