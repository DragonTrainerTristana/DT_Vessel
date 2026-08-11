"""
혼합 함대 평가 (2026-08) — 16척 중 일부만 통신 장비를 갖춘 상황.

학습은 다시 하지 않는다. 학습이 끝난 정책을 고정해 두고, 평가할 때만 배 일부의
통신을 끊은 뒤 두 무리의 성적을 따로 잰다. 통신 가능한 배가 몇 척 남았느냐를
바꿔가며 반복하면 "몇 대가 장비를 달아야 효과가 나오는가"를 볼 수 있다.

통신을 못 하는 배는 두 종류로 나눠 볼 수 있다.
  radar : 아무것도 못 한다. 보내지도 받지도 못하고 레이더만 쓴다.
  rx    : 듣기만 한다. 남의 메시지는 받지만 자기 것은 못 보낸다(소형선 가정).

★비율 전체를 한 번에 잰다. 환경(배치)마다 다른 비율을 심어 놓고 한 번만 돌리므로,
  비율 7가지를 따로 7번 돌리는 것보다 7배 가깝게 빠르다. burn-in도 한 번만 낸다.

통신 불가 선박은 스폰 링 위에 고르게 흩어 배치한다(한쪽에 몰리면 그 구역만
통신 공백이 되어 비교가 왜곡된다).

사용:
  VESSEL_MSG_DIM=6 VESSEL_USE_MOE=1 VESSEL_MOE_SHARED=1 \
  python eval_mixed.py --ckpt qd_MOE_SE_s42.pt --mode radar --csv mixed_s42.csv
"""
import os, argparse, time
import torch
import config as cfg
import vessel_gym as vg
from networks import CNNPolicy
from vessel_gym_train import comm_gather, parse_obs, FrameStack

# 상대 경로로 넘긴 체크포인트/CSV를 찾을 곳. 절대 경로를 주면 그대로 쓴다.
SCRATCH = os.environ.get('VESSEL_CKPT_DIR',
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints'))


def spread_indices(n_total, k):
    """통신 불가 선박 k척을 0..n_total-1 위에 고르게 배치한 인덱스."""
    if k <= 0:
        return []
    step = n_total / k
    return sorted({int(round(i * step)) % n_total for i in range(k)})


class Acc:
    """한 무리(통신 가능/불가)의 집계기."""

    def __init__(self):
        self.counts = [0, 0, 0, 0, 0]          # _/goal/vColl/oColl/TO
        self.total = 0
        self.g = dict(fuel=0.0, head=0.0, length=0.0, n=0)   # goal 에피소드만
        self.allsep_sum = 0.0
        self.allsep_n = 0
        self.comp_bin = 0.0
        self.comp_n = 0
        self.sit_ok = {k: 0.0 for k in (1, 2, 3, 4)}
        self.sit_n = {k: 0 for k in (1, 2, 3, 4)}

    def row(self):
        if self.total == 0:
            return None
        n = max(self.g['n'], 1)
        return dict(
            eps=self.total,
            goal=self.counts[1] / self.total * 100,
            coll=(self.counts[2] + self.counts[3]) / self.total * 100,
            to=self.counts[4] / self.total * 100,
            fuel=self.g['fuel'] / n,
            head=self.g['head'] / n,
            length=self.g['length'] / n,
            minsep=self.allsep_sum / max(self.allsep_n, 1),
            colregs=100 * self.comp_bin / max(self.comp_n, 1),
            colregs_n=self.comp_n,
            **{f'sit{k}': 100 * self.sit_ok[k] / max(self.sit_n[k], 1) for k in (1, 2, 3, 4)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--sweep', default='2,4,6,8,10,12,14')    # 통신 불가 선박 수 목록
    ap.add_argument('--mode', default='radar', choices=['radar', 'rx'])
    ap.add_argument('--envs_per', type=int, default=16)        # 비율 하나당 환경 수
    ap.add_argument('--vessels', type=int, default=16)
    ap.add_argument('--max_partners', type=int, default=cfg.MAX_COMM_PARTNERS)
    ap.add_argument('--eval_decisions', type=int, default=6000)
    ap.add_argument('--burnin', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--ring', type=float, default=0.7)
    ap.add_argument('--tag', default='')
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    SW = [int(s) for s in args.sweep.split(',') if s.strip()]
    N = args.vessels
    E = len(SW) * args.envs_per
    torch.manual_seed(args.seed)
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(SCRATCH, args.ckpt)

    # 환경 블록마다 다른 비율을 심는다: env e → 설정 e // envs_per
    nocomm = torch.zeros(E, N, dtype=torch.bool, device=dev)
    for ci, k in enumerate(SW):
        idx = spread_indices(N, k)
        nocomm[ci * args.envs_per:(ci + 1) * args.envs_per, idx] = True
    comm = ~nocomm
    send_mask = comm                                    # 송신 불가
    recv_mask = comm if args.mode == 'radar' else torch.ones_like(comm)

    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device=dev, seed=args.seed,
                            ring_scale=args.ring, crossing=2, risk_range=420.0,
                            farfield_coef=0.5, perpair_coef=-0.15, perpair_exp=3.0)
    policy = CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES).to(dev)
    sd = torch.load(ckpt_path, map_location=dev)
    policy.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)
    policy.eval()

    fs = FrameStack(E, N, dev)
    obs = env.reset()
    radar, goal, self_s, sit = parse_obs(obs)
    fs.reset_all(radar)

    def act(x, goal, self_s, sit):
        with torch.no_grad():
            om, _ = comm_gather(policy, env, x, goal, self_s, sit, args.max_partners,
                                send_mask=send_mask, recv_mask=recv_mask)
            action, _, _, _ = policy.ctr_actor(x, goal, self_s, om, sit)
        return action

    BIG = 1e9
    eye = torch.eye(N, device=dev).unsqueeze(0) * BIG

    def wrap180(x):
        return (x + 180.0) % 360.0 - 180.0

    ep_fuel = torch.zeros(E, N, device=dev)
    ep_head = torch.zeros(E, N, device=dev)
    ep_len = torch.zeros(E, N, device=dev)
    ep_minsep = torch.full((E, N), BIG, device=dev)
    prev_head = env.heading.clone()

    # 설정 × 무리별 집계기, 그리고 그에 대응하는 [E,N] 마스크
    acc, grp = {}, {}
    for ci, k in enumerate(SW):
        blk = torch.zeros(E, N, dtype=torch.bool, device=dev)
        blk[ci * args.envs_per:(ci + 1) * args.envs_per] = True
        acc[(k, 'nocomm')] = Acc(); grp[(k, 'nocomm')] = blk & nocomm
        acc[(k, 'comm')] = Acc();   grp[(k, 'comm')] = blk & comm

    t0 = time.time()
    for i in range(args.burnin):
        a = act(fs.get(), goal, self_s, sit)
        obs, _, done, _ = env.step(a)
        radar, goal, self_s, sit = parse_obs(obs)
        fs.push(radar, done)
        prev_head = torch.where(done, env.heading, prev_head)
        if i == 200:
            dps = 201 * E * N / (time.time() - t0)
            print(f"  [{os.path.basename(ckpt_path)}] {dps:.0f} dec/s, "
                  f"ETA ~{(args.burnin + args.eval_decisions) * E * N / dps / 60:.0f}min", flush=True)

    for _ in range(args.eval_decisions):
        a = act(fs.get(), goal, self_s, sit)
        sr = env.speed / torch.clamp(env.max_speed, min=1e-6)
        turn01 = a[..., 0].abs().clamp(0, 1)
        ep_fuel += sr ** 2 + 0.5 * turn01 ** 2
        ep_head += wrap180(env.heading - prev_head).abs()
        prev_head = env.heading.clone()
        ep_minsep = torch.minimum(ep_minsep, (torch.cdist(env.pos, env.pos) + eye).min(dim=-1).values)
        ep_len += 1.0

        _pw = env._last_pw
        if _pw is not None:
            _sit = env.situation
            _rud = a[..., 0]
            _gate = (_pw['risk'].max(dim=-1).values > 0.3) & (_sit > 0)
            if bool(_gate.any()):
                _star = ((_sit == 1) | (_sit == 3) | (_sit == 4)).to(ep_fuel.dtype)
                _hold = (_sit == 2).to(ep_fuel.dtype)
                _viol = _star * torch.clamp(-_rud, min=0.0) + _hold * _rud.abs()
                _ok = (_viol < 0.1).float()
                for key, gm in grp.items():
                    _gg = _gate & gm
                    if not bool(_gg.any()):
                        continue
                    A = acc[key]
                    A.comp_bin += float(_ok[_gg].sum())
                    A.comp_n += int(_gg.sum())
                    for _k in (1, 2, 3, 4):
                        _m = _gg & (_sit == _k)
                        if bool(_m.any()):
                            A.sit_ok[_k] += float(_ok[_m].sum())
                            A.sit_n[_k] += int(_m.sum())

        obs, _, done, outcome = env.step(a)
        term = (outcome != 0)
        if int(term.sum()):
            for key, gm in grp.items():
                A = acc[key]
                for oc in range(1, 5):
                    mask = (outcome == oc) & gm
                    c = int(mask.sum())
                    if not c:
                        continue
                    A.counts[oc] += c
                    A.total += c
                    if oc == 1:
                        A.g['fuel'] += float(ep_fuel[mask].sum())
                        A.g['head'] += float(ep_head[mask].sum())
                        A.g['length'] += float(ep_len[mask].sum())
                        A.g['n'] += c
                tm = term & gm
                if int(tm.sum()):
                    A.allsep_sum += float(ep_minsep[tm].clamp(max=vg.RADAR_RANGE * 4).sum())
                    A.allsep_n += int(tm.sum())
            ep_fuel = torch.where(term, torch.zeros_like(ep_fuel), ep_fuel)
            ep_head = torch.where(term, torch.zeros_like(ep_head), ep_head)
            ep_len = torch.where(term, torch.zeros_like(ep_len), ep_len)
            ep_minsep = torch.where(term, torch.full_like(ep_minsep, BIG), ep_minsep)
        radar, goal, self_s, sit = parse_obs(obs)
        fs.push(radar, done)
        prev_head = torch.where(done, env.heading, prev_head)

    tag = args.tag or os.path.splitext(os.path.basename(ckpt_path))[0]
    nlab = 'Radar-only' if args.mode == 'radar' else 'Receive-only'
    cols = ['eps', 'goal', 'coll', 'to', 'fuel', 'head', 'length', 'minsep',
            'colregs', 'colregs_n', 'sit1', 'sit2', 'sit3', 'sit4']
    lines = []
    print(f"### {tag} | mode={args.mode} | {args.eval_decisions} dec x {args.envs_per} envs/config")
    for k in SW:
        for g, lab in (('nocomm', nlab), ('comm', 'Comm')):
            r = acc[(k, g)].row()
            if r is None:
                continue
            print(f"  {k:2d}{'R' if args.mode == 'radar' else 'Rx'}/{N - k:2d}C {lab:12s}"
                  f" goal={r['goal']:5.1f}% coll={r['coll']:5.2f}% colregsOK={r['colregs']:5.1f}%"
                  f" minSep={r['minsep']:5.1f}m fuel={r['fuel']:6.1f} head={r['head']:6.0f}deg"
                  f" len={r['length']:5.0f} (eps={r['eps']})")
            lines.append([tag, args.mode, k, g] + [f"{r[c]:.4f}" for c in cols])
    if args.csv:
        path = args.csv if os.path.isabs(args.csv) else os.path.join(SCRATCH, args.csv)
        new = not os.path.exists(path)
        with open(path, 'a', encoding='utf-8') as f:
            if new:
                f.write('tag,mode,nocomm,group,' + ','.join(cols) + '\n')
            for ln in lines:
                f.write(','.join(str(v) for v in ln) + '\n')
        print(f"  csv -> {path}")


if __name__ == '__main__':
    main()
