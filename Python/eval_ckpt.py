"""
프리즈-정책 완주 평가 (2026-08) — 학습 로그의 per-window goal%가 '동기화 종료 파도 × 고정-decision 창'
aliasing으로 요동하는 문제를 우회. 체크포인트를 로드해 다수 에피소드를 끝까지 돌려 outcome을 pooling.

핵심: 위상 dephase — 각 배마다 랜덤 burn-in(0~MAX_EP) 후부터 집계 시작 → 종료 파도 smear.
     충분히 길게(에이전트당 여러 에피소드) 돌려 안정적 outcome 분포 획득.

사용: VESSEL_MSG_DIM=6 VESSEL_USE_MOE=1 python eval_ckpt.py --ckpt vg_OFF_s43.pt --arm OFF
     (arm/dim/moe는 체크포인트 학습 시와 동일 env로 맞춰야 함)
"""
import os, sys, argparse
import torch
import config as cfg
import vessel_gym as vg
from networks import CNNPolicy
from vessel_gym_train import comm_gather, parse_obs, FrameStack, make_others_msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arm', default='OFF', choices=['OFF', 'ORACLE', 'ON'])
    ap.add_argument('--envs', type=int, default=256)
    ap.add_argument('--vessels', type=int, default=16)
    ap.add_argument('--max_partners', type=int, default=cfg.MAX_COMM_PARTNERS)
    ap.add_argument('--eval_decisions', type=int, default=8000)  # 에이전트당 ~5 에피소드(1600×5)
    ap.add_argument('--burnin', type=int, default=1700)          # 위상 dephase(>MAX_EP 1600)
    ap.add_argument('--seed', type=int, default=999)             # eval seed(학습과 분리)
    ap.add_argument('--ring', type=float, default=0.7)           # 스폰 링 스케일(학습과 동일해야 함)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    E, N = args.envs, args.vessels
    torch.manual_seed(args.seed)
    scr = os.path.dirname(os.path.abspath(__file__))
    # 상대 경로로 넘기면 VESSEL_CKPT_DIR(기본: 이 파일 옆 checkpoints/)에서 찾는다.
    ckpt_dir = os.environ.get('VESSEL_CKPT_DIR', os.path.join(scr, 'checkpoints'))
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ckpt_dir, args.ckpt)

    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device=dev, seed=args.seed,
                            ring_scale=args.ring, crossing=2, risk_range=420.0,
                            farfield_coef=0.5, perpair_coef=-0.15, perpair_exp=3.0)
    policy = CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES).to(dev)
    sd = torch.load(ckpt_path, map_location=dev)
    policy.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)
    policy.eval()

    fs = FrameStack(E, N, dev)
    obs = env.reset()
    radar, goal, self_s, sit = parse_obs(obs); fs.reset_all(radar)

    counts = torch.zeros(5, device=dev)   # RUNNING/GOAL/vColl/oColl/TO
    total = 0

    def act(x, goal, self_s, sit):
        with torch.no_grad():
            if args.arm == 'ON':
                om, _ = comm_gather(policy, env, x, goal, self_s, sit, args.max_partners)
            else:
                om = make_others_msg(env, args.arm, E, N, dev)
            action, _, _, _ = policy.ctr_actor(x, goal, self_s, om, sit)
        return action

    import math, time
    t0 = time.time()
    DEG = vg.DEG
    BIG = 1e9
    eye = torch.eye(N, device=dev).unsqueeze(0) * BIG

    def wrap180(x):
        return (x + 180.0) % 360.0 - 180.0

    def nearest_sep():
        """각 배의 가장 가까운 타선 거리 [E,N] (step 전 위치 기준, 안전마진 지표)."""
        d = torch.cdist(env.pos, env.pos) + eye
        return d.min(dim=-1).values

    # 에피소드별 누적기 [E,N] (★step 전 유효 상태로 누적 → respawn 텔레포트 오염 없음)
    ep_fuel = torch.zeros(E, N, device=dev)
    ep_head = torch.zeros(E, N, device=dev)      # 총 heading 변화(궤적 굽음 = 덜 부드러움)
    ep_len = torch.zeros(E, N, device=dev)
    ep_minsep = torch.full((E, N), BIG, device=dev)   # 에피소드 중 최소 선박간 거리
    ep_reward = torch.zeros(E, N, device=dev)    # ★에피소드 누적 보상(연속 지표 — 효과크기 측정력↑)
    prev_head = env.heading.clone()
    # outcome별 지표 합계 (goal 에피소드 위주 비교)
    msum = {oc: dict(fuel=0.0, head=0.0, minsep=0.0, length=0.0, reward=0.0, n=0) for oc in range(1, 5)}
    all_minsep_sum = 0.0; all_minsep_n = 0
    # ★COLREGs 준수도 (보상 항과 동일 정의): max_risk>0.3 조우에서
    #   HeadOn(1)/GiveWay(3)/Overtaking(4) → 우현 권장 → 위반 = 좌현 변침량
    #   CrossingStandOn(2) → 침로유지 권장 → 위반 = |변침량|
    #   compliance = 1 - 위반 (0~1). 게이트 밖(위험 없음)은 집계 제외.
    comp_sum = 0.0; comp_n = 0; comp_bin = 0.0
    # ★상황별 준수율 (1=HeadOn, 2=CrossingStandOn, 3=CrossingGiveWay, 4=Overtaking)
    sit_ok = {k: 0.0 for k in (1, 2, 3, 4)}; sit_n = {k: 0 for k in (1, 2, 3, 4)}

    # burn-in: 집계 없이 위상 dephase
    for i in range(args.burnin):
        a = act(fs.get(), goal, self_s, sit)
        obs, _, done, _ = env.step(a)
        radar, goal, self_s, sit = parse_obs(obs); fs.push(radar, done)
        prev_head = torch.where(done, env.heading, prev_head)   # respawn 시 prev 갱신
        if i == 200:
            dps = 201 * E * N / (time.time() - t0)
            eta = (args.burnin + args.eval_decisions) * E * N / dps
            print(f"  [{os.path.basename(ckpt_path)}] {dps:.0f} dec/s, ETA ~{eta:.0f}s", flush=True)

    # 평가: 완주 outcome pooling + 지표
    for _ in range(args.eval_decisions):
        # ── step 전 유효 상태로 지표 누적 ──
        a = act(fs.get(), goal, self_s, sit)
        sr = env.speed / torch.clamp(env.max_speed, min=1e-6)
        turn01 = a[..., 0].abs().clamp(0, 1)
        ep_fuel += sr ** 2 + 0.5 * turn01 ** 2                    # 연료 프록시(보상 항과 동일 형태)
        dh = wrap180(env.heading - prev_head).abs()
        ep_head += dh                                            # 방향 변화 누적
        prev_head = env.heading.clone()
        sep = nearest_sep()
        ep_minsep = torch.minimum(ep_minsep, sep)
        ep_len += 1.0
        # ── ★COLREGs 준수도 (step 전 상황·행동으로 판정; 보상 게이트와 동일 max_risk>0.3) ──
        _pw = env._last_pw
        if _pw is not None:
            _mrisk = _pw['risk'].max(dim=-1).values                  # [E,N]
            _sit = env.situation                                     # [E,N] 0~4
            _rud = a[..., 0]                                         # [-1,1] >0=우현
            _gate = (_mrisk > 0.3) & (_sit > 0)                      # 조우 + 위험
            if bool(_gate.any()):
                _star = ((_sit == 1) | (_sit == 3) | (_sit == 4)).to(r_dtype := ep_fuel.dtype)
                _hold = (_sit == 2).to(r_dtype)
                _viol = _star * torch.clamp(-_rud, min=0.0) + _hold * _rud.abs()
                _comp = (1.0 - _viol).clamp(0.0, 1.0)                # 1=완전준수
                comp_sum += float(_comp[_gate].sum())
                comp_bin += float((_viol[_gate] < 0.1).float().sum())
                comp_n += int(_gate.sum())
                for _k in (1, 2, 3, 4):
                    _m = _gate & (_sit == _k)
                    if bool(_m.any()):
                        sit_ok[_k] += float((_viol[_m] < 0.1).float().sum())
                        sit_n[_k] += int(_m.sum())
        # ── step ──
        obs, rew_step, done, outcome = env.step(a)
        ep_reward += rew_step
        # ── 종료 에피소드: 지표 기록 후 리셋 ──
        for oc in range(1, 5):
            mask = (outcome == oc)
            c = int(mask.sum())
            if c:
                counts[oc] += c; total += c
                msum[oc]['fuel'] += float(ep_fuel[mask].sum())
                msum[oc]['head'] += float(ep_head[mask].sum())
                msum[oc]['minsep'] += float(ep_minsep[mask].clamp(max=vg.RADAR_RANGE * 4).sum())
                msum[oc]['length'] += float(ep_len[mask].sum())
                msum[oc]['reward'] += float(ep_reward[mask].sum())
                msum[oc]['n'] += c
        term = (outcome != 0)
        if int(term.sum()):
            ms = ep_minsep[term].clamp(max=vg.RADAR_RANGE * 4)
            all_minsep_sum += float(ms.sum()); all_minsep_n += int(term.sum())
            ep_fuel = torch.where(term, torch.zeros_like(ep_fuel), ep_fuel)
            ep_head = torch.where(term, torch.zeros_like(ep_head), ep_head)
            ep_len = torch.where(term, torch.zeros_like(ep_len), ep_len)
            ep_minsep = torch.where(term, torch.full_like(ep_minsep, BIG), ep_minsep)
            ep_reward = torch.where(term, torch.zeros_like(ep_reward), ep_reward)
        radar, goal, self_s, sit = parse_obs(obs); fs.push(radar, done)
        prev_head = torch.where(done, env.heading, prev_head)

    if total == 0:
        print(f"{os.path.basename(ckpt_path):26s} | no terminations"); return
    g, v, o, t = (float(counts[i]) / total * 100 for i in (1, 2, 3, 4))
    # goal 에피소드 기준 궤적 지표(공정 비교: 도달한 배들이 얼마나 부드럽고 안전하게 갔나)
    gm = msum[1]
    if gm['n'] > 0:
        gf = gm['fuel'] / gm['n']; gh = gm['head'] / gm['n']
        gs = gm['minsep'] / gm['n']; gl = gm['length'] / gm['n']
    else:
        gf = gh = gs = gl = float('nan')
    amin = all_minsep_sum / max(all_minsep_n, 1)
    # ★연속 지표: 전 에피소드 평균 보상 + COLREGs 준수도(연속 0~1, 이진 준수율)
    r_all = sum(msum[oc]['reward'] for oc in range(1, 5)) / max(total, 1)
    comp = comp_sum / max(comp_n, 1)
    compb = comp_bin / max(comp_n, 1) * 100
    print(f"{os.path.basename(ckpt_path):26s} | arm={args.arm:6s} | "
          f"goal={g:5.1f}%  vColl={v:5.1f}%  oColl={o:5.1f}%  TO={t:5.1f}%  | {total} eps || "
          f"[goal-ep] fuel={gf:6.1f}  headTravel={gh:6.0f}deg  minSep={gs:5.1f}m  len={gl:5.0f}  | allMinSep={amin:5.1f}m "
          f"|| epReward={r_all:8.2f}  colregs={comp:.4f}  colregsOK={compb:5.1f}%  (n={comp_n})"
          + " || " + " ".join(f"sit{k}={100*sit_ok[k]/max(sit_n[k],1):5.1f}%(n={sit_n[k]})"
                              for k in (1, 2, 3, 4)))


if __name__ == '__main__':
    main()
