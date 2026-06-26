"""
완전분리 MoE 스모크 검증 (2026-06-26).

USE_MOE=1이면 MessageActor/ControlActor/Critic 각각이 COLREGs 5상황별로 *코어를 통째 5벌*
(radar_encoder 포함) 보유 → 상황별 완전 독립망. USE_MOE=0이면 코어 1벌(=단일망 baseline).

검증 항목:
  A. 구조: OFF=core(experts 없음), MoE=experts[5] 각각 radar_encoder/fc2/... 독립 보유.
  B. forward/evaluate_actions 동작 + 출력 shape (MoE on/off, comm path).
  C. PPO mirror (rollout forward → evaluate_actions): logprob/value 정합 (sum·attention × MoE on/off).
     ★MoE에선 partner_situations로 MessageActor를 *파트너 상황*으로 재라우팅해야 미러 성립.
  D. others_msg mirror (rollout _get_others_msg vs update 집계) — sender 라우팅 포함.
  E. comm-OFF 불변: others_msg ≡ 0.
  F. ★상황별 grad 격리(완전분리의 핵심): 한 상황만 든 배치 → 그 상황 코어의 *radar_encoder까지* 만
     grad, 다른 상황 코어는 grad=0/None. MessageActor·ControlActor·Critic 모두.
  G. 라우팅이 출력을 바꾼다: 같은 obs라도 situation이 다르면 action_mean이 달라진다(코어가 실제 다름).

실행: python _smoke_fullmoe.py
"""
import os
import sys
import importlib
import math
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
DEV = torch.device('cpu')


def _reload(use_comm=True, use_attention=False, use_moe=False, msg_dim=6,
            consumer_coef=0.0, consumer_coupling=False):
    os.environ['VESSEL_USE_COMM'] = '1' if use_comm else '0'
    os.environ['VESSEL_USE_ATTENTION'] = '1' if use_attention else '0'
    os.environ['VESSEL_USE_MOE'] = '1' if use_moe else '0'
    os.environ['VESSEL_MSG_DIM'] = str(msg_dim)
    os.environ['VESSEL_AGG_MODE'] = 'sum'
    os.environ['VESSEL_COMM_CONSUMER_COEF'] = str(consumer_coef)
    os.environ['VESSEL_COMM_CONSUMER_COUPLING'] = '1' if consumer_coupling else '0'
    os.environ.pop('VESSEL_POS_GROUND', None)
    os.environ.pop('VESSEL_NEAREST_SCALE', None)
    os.environ.pop('VESSEL_MSG_GAIN', None)
    os.environ.pop('VESSEL_ORACLE', None)
    import config
    importlib.reload(config)
    import networks
    importlib.reload(networks)
    return config, networks


def _make_scene(cfg, n_agents=8, seed=0):
    rng = np.random.RandomState(seed)
    F, S = cfg.FRAMES, cfg.STATE_SIZE
    state_dim = F * S
    Kmax = cfg.MAX_COMM_PARTNERS

    states = rng.randn(n_agents, state_dim).astype(np.float32)
    goals = rng.randn(n_agents, cfg.GOAL_SIZE).astype(np.float32)
    selfs = rng.randn(n_agents, cfg.SELF_STATE_SIZE).astype(np.float32)
    positions = (rng.rand(n_agents, 2).astype(np.float32) * (cfg.COMM_RANGE * 0.5))
    # 모든 상황(0~4)이 등장하도록 강제 (grad 격리 테스트가 모든 expert를 보게)
    situations = (np.arange(n_agents) % cfg.NUM_COLREGS_SITUATIONS).astype(np.int64)
    rng.shuffle(situations)

    agent_id_list = list(range(n_agents))
    id_to_idx = {a: i for i, a in enumerate(agent_id_list)}
    pos_map = {a: positions[i] for i, a in enumerate(agent_id_list)}

    def relpos(rx, rz, rhead_deg, px, pz):
        dx, dz = px - rx, pz - rz
        rng_ = math.hypot(dx, dz)
        bearing = math.atan2(dx, dz) - math.radians(rhead_deg)
        return [math.sin(bearing), math.cos(bearing), min(rng_ / cfg.COMM_RANGE, 1.0)]

    comm_partners, comm_relpos = {}, {}
    for a in agent_id_list:
        rx, rz = pos_map[a]
        dists = []
        for b in agent_id_list:
            if b == a:
                continue
            bx, bz = pos_map[b]
            d = np.hypot(rx - bx, rz - bz)
            if d <= cfg.COMM_RANGE:
                dists.append((b, d))
        dists.sort(key=lambda x: x[1])
        partners = [b for b, _ in dists[:Kmax]]
        comm_partners[a] = partners
        rhead = float(selfs[id_to_idx[a]][2]) * 180.0
        rp = [relpos(rx, rz, rhead, pos_map[p][0], pos_map[p][1]) for p in partners]
        comm_relpos[a] = np.array(rp, dtype=np.float32) if rp else np.zeros((0, 3), np.float32)

    # update용 partner 패딩 텐서 (main.training_step 규약: kept_pos 정렬 + p_sits)
    p_states = np.zeros((n_agents, Kmax, state_dim), np.float32)
    p_goals = np.zeros((n_agents, Kmax, cfg.GOAL_SIZE), np.float32)
    p_selfs = np.zeros((n_agents, Kmax, cfg.SELF_STATE_SIZE), np.float32)
    p_mask = np.zeros((n_agents, Kmax), np.float32)
    p_relpos = np.zeros((n_agents, Kmax, 3), np.float32)
    p_sits = np.zeros((n_agents, Kmax), np.int64)
    for i, a in enumerate(agent_id_list):
        partners = comm_partners[a]
        relpos_arr = comm_relpos[a]
        kept_pos = [k for k, p in enumerate(partners) if p in id_to_idx][:Kmax]
        pidx = [id_to_idx[partners[k]] for k in kept_pos]
        for j, k in enumerate(kept_pos):
            pj = pidx[j]
            p_states[i, j] = states[pj]
            p_goals[i, j] = goals[pj]
            p_selfs[i, j] = selfs[pj]
            p_mask[i, j] = 1.0
            p_sits[i, j] = int(situations[pj])
            if relpos_arr is not None and k < len(relpos_arr):
                p_relpos[i, j] = relpos_arr[k]

    return dict(states=states, goals=goals, selfs=selfs, situations=situations,
                agent_id_list=agent_id_list, comm_partners=comm_partners, comm_relpos=comm_relpos,
                p_states=p_states, p_goals=p_goals, p_selfs=p_selfs, p_mask=p_mask,
                p_relpos=p_relpos, p_sits=p_sits, n_agents=n_agents)


def _policy(cfg, networks, seed=0):
    torch.manual_seed(seed)
    pol = networks.CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES).to(DEV)
    pol.eval()
    return pol


def _tens(sc):
    x = torch.as_tensor(sc['states'], dtype=torch.float32).unsqueeze(0)
    goal = torch.as_tensor(sc['goals'], dtype=torch.float32).unsqueeze(0)
    self_s = torch.as_tensor(sc['selfs'], dtype=torch.float32).unsqueeze(0)
    return x, goal, self_s


# ──────────────────────────── A. 구조 ────────────────────────────
def test_structure():
    res = {}
    for moe in (False, True):
        cfg, nn_ = _reload(use_moe=moe)
        pol = _policy(cfg, nn_)
        for name, mod in (('msg', pol.msg_actor), ('ctr', pol.ctr_actor), ('cri', pol.critic)):
            has_experts = hasattr(mod, 'experts')
            has_core = hasattr(mod, 'core')
            n = len(mod.cores())
            # 각 코어가 radar_encoder를 *독립* 보유? (id가 모두 다름)
            radar_ids = [id(c.radar_encoder) for c in mod.cores()]
            indep_radar = len(set(radar_ids)) == len(radar_ids)
            ok = (has_experts == moe) and (has_core == (not moe)) and (n == (5 if moe else 1)) and indep_radar
            res[f'{"MoE" if moe else "OFF"}:{name}'] = (n, has_experts, has_core, indep_radar, ok)
    return res


# ──────────────────────────── B. forward ────────────────────────────
def test_forward():
    res = {}
    for moe in (False, True):
        cfg, nn_ = _reload(use_moe=moe)
        pol = _policy(cfg, nn_)
        sc = _make_scene(cfg)
        x, goal, self_s = _tens(sc)
        sit = torch.as_tensor(sc['situations'], dtype=torch.long).unsqueeze(0) if moe else None
        with torch.no_grad():
            v, a, lp, mean, msg, om = pol.forward(
                x, goal, self_s, return_msg=True,
                comm_partners=sc['comm_partners'], agent_id_list=sc['agent_id_list'],
                comm_relpos=sc['comm_relpos'], situation=sit)
        N = sc['n_agents']
        ok = (tuple(a.shape) == (1, N, cfg.CONTINUOUS_ACTION_SIZE)
              and tuple(v.shape) == (1, N, 1)
              and tuple(msg.shape) == (1, N, cfg.MSG_DIM)
              and torch.isfinite(a).all().item() and torch.isfinite(v).all().item())
        res[f'{"MoE" if moe else "OFF"}'] = (tuple(a.shape), tuple(v.shape), ok)
    return res


# ──────────────────────────── C/D. PPO mirror ────────────────────────────
def _mirror(use_attention, use_moe):
    cfg, nn_ = _reload(use_comm=True, use_attention=use_attention, use_moe=use_moe)
    pol = _policy(cfg, nn_)
    sc = _make_scene(cfg)
    x, goal, self_s = _tens(sc)
    sit = torch.as_tensor(sc['situations'], dtype=torch.long).unsqueeze(0) if use_moe else None

    with torch.no_grad():
        v_r, action, lp_r, mean, msg, om_roll = pol.forward(
            x, goal, self_s, return_msg=True,
            comm_partners=sc['comm_partners'], agent_id_list=sc['agent_id_list'],
            comm_relpos=sc['comm_relpos'], situation=sit)

    x_u = x.transpose(0, 1).contiguous(); goal_u = goal.transpose(0, 1).contiguous()
    self_u = self_s.transpose(0, 1).contiguous(); act_u = action.transpose(0, 1).contiguous()
    sit_u = sit.transpose(0, 1).contiguous() if sit is not None else None
    px = torch.as_tensor(sc['p_states'], dtype=torch.float32)
    pg = torch.as_tensor(sc['p_goals'], dtype=torch.float32)
    ps = torch.as_tensor(sc['p_selfs'], dtype=torch.float32)
    pm = torch.as_tensor(sc['p_mask'], dtype=torch.float32).unsqueeze(-1)
    prp = torch.as_tensor(sc['p_relpos'], dtype=torch.float32)
    p_sits = torch.as_tensor(sc['p_sits'], dtype=torch.long) if use_moe else None

    with torch.no_grad():
        v_u, lp_u, ent, mr, it, tt, gl, rl, cl = pol.evaluate_actions(
            x_u, goal_u, self_u, px, pg, ps, pm, prp, act_u,
            situation=sit_u, partner_situations=p_sits)

    lp_diff = (lp_r.transpose(0, 1).contiguous() - lp_u).abs().max().item()
    v_diff = (v_r.transpose(0, 1).contiguous() - v_u).abs().max().item()
    return lp_diff, v_diff


def test_others_msg_mirror(use_attention, use_moe):
    """rollout _get_others_msg vs update 집계 — sender(파트너) 라우팅 포함."""
    cfg, nn_ = _reload(use_comm=True, use_attention=use_attention, use_moe=use_moe)
    pol = _policy(cfg, nn_)
    sc = _make_scene(cfg)
    x, goal, self_s = _tens(sc)
    sit = torch.as_tensor(sc['situations'], dtype=torch.long).unsqueeze(0) if use_moe else None
    with torch.no_grad():
        msg = pol.msg_actor(x, goal, self_s, sit)
        others_roll = pol._get_others_msg(msg, sc['comm_partners'], sc['agent_id_list'],
                                          sc['comm_relpos'], self_state=self_s, goal=goal)
    px = torch.as_tensor(sc['p_states'], dtype=torch.float32)
    pg = torch.as_tensor(sc['p_goals'], dtype=torch.float32)
    ps = torch.as_tensor(sc['p_selfs'], dtype=torch.float32)
    pm = torch.as_tensor(sc['p_mask'], dtype=torch.float32).unsqueeze(-1)
    prp = torch.as_tensor(sc['p_relpos'], dtype=torch.float32)
    p_sits = torch.as_tensor(sc['p_sits'], dtype=torch.long) if use_moe else None
    with torch.no_grad():
        msg_part = pol.msg_actor(px, pg, ps, p_sits)
        if pol.use_attention:
            q_in = torch.cat([self_s.transpose(0, 1), goal.transpose(0, 1)], dim=-1)
            others_upd = pol.attn.aggregate_batch(q_in, prp, msg_part, pm)
        else:
            others_upd = (msg_part * pm).sum(dim=1, keepdim=True)
    diff = (others_roll.transpose(0, 1).contiguous() - others_upd).abs().max().item()
    return diff


# ──────────────────────────── E. comm-OFF ────────────────────────────
def test_comm_off():
    cfg, nn_ = _reload(use_comm=False, use_moe=True)
    pol = _policy(cfg, nn_)
    sc = _make_scene(cfg)
    x, goal, self_s = _tens(sc)
    sit = torch.as_tensor(sc['situations'], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        msg = pol.msg_actor(x, goal, self_s, sit)
        others = pol._get_others_msg(msg, sc['comm_partners'], sc['agent_id_list'],
                                     sc['comm_relpos'], self_state=self_s, goal=goal)
    return others.abs().max().item()


# ──────────────────────────── F. grad 격리 (핵심) ────────────────────────────
def test_grad_isolation():
    """한 상황만 든 배치 → 그 상황 코어(radar_encoder 포함)만 grad, 나머지 코어 grad=0/None."""
    cfg, nn_ = _reload(use_comm=True, use_moe=True)
    pol = _policy(cfg, nn_)
    NE = cfg.NUM_COLREGS_SITUATIONS
    F, S = cfg.FRAMES, cfg.STATE_SIZE
    N = 4
    target_k = 2   # 이 상황만 배치에 넣음

    x = torch.randn(N, 1, F * S)
    goal = torch.randn(N, 1, cfg.GOAL_SIZE)
    self_s = torch.randn(N, 1, cfg.SELF_STATE_SIZE)
    om = torch.randn(N, 1, cfg.MSG_DIM)
    sit = torch.full((N, 1), target_k, dtype=torch.long)

    results = {}

    def _grad_split(module, fired_idx, fn):
        pol.zero_grad(set_to_none=True)
        out = fn()
        out.sum().backward()
        fired_radar = module.cores()[fired_idx].radar_encoder.conv1.weight.grad
        fired_has = fired_radar is not None and float(fired_radar.abs().sum()) > 0
        others_zero = True
        for k, c in enumerate(module.cores()):
            if k == fired_idx:
                continue
            for p in c.parameters():
                if p.grad is not None and float(p.grad.abs().sum()) > 0:
                    others_zero = False
        return fired_has, others_zero

    # ControlActor: _route mean
    def ctr_fn():
        z, mean, logstd, _, _ = pol.ctr_actor._route(x, goal, self_s, om, sit)
        return mean + z.sum() * 0 + logstd  # mean+logstd로 head·backbone 다 커버
    fh, oz = _grad_split(pol.ctr_actor, target_k, ctr_fn)
    results['ctr_actor'] = (fh, oz, fh and oz)

    # Critic
    def cri_fn():
        return pol.critic(x, goal, self_s, om, sit)
    fh, oz = _grad_split(pol.critic, target_k, cri_fn)
    results['critic'] = (fh, oz, fh and oz)

    # MessageActor (sender 라우팅)
    def msg_fn():
        return pol.msg_actor(x, goal, self_s, sit)
    fh, oz = _grad_split(pol.msg_actor, target_k, msg_fn)
    results['msg_actor'] = (fh, oz, fh and oz)
    return results


# ──────────────────────────── G. 라우팅이 출력을 바꾼다 ────────────────────────────
def test_routing_changes_output():
    cfg, nn_ = _reload(use_comm=True, use_moe=True)
    pol = _policy(cfg, nn_)
    F, S = cfg.FRAMES, cfg.STATE_SIZE
    x = torch.randn(1, 1, F * S); goal = torch.randn(1, 1, cfg.GOAL_SIZE)
    self_s = torch.randn(1, 1, cfg.SELF_STATE_SIZE); om = torch.randn(1, 1, cfg.MSG_DIM)
    means = []
    with torch.no_grad():
        for k in range(cfg.NUM_COLREGS_SITUATIONS):
            sit = torch.full((1, 1), k, dtype=torch.long)
            _, mean, _, _, _ = pol.ctr_actor._route(x, goal, self_s, om, sit)
            means.append(mean.clone())
    # 서로 다른 상황 → 서로 다른 출력 (모든 쌍이 구별돼야)
    max_pair = 0.0
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            max_pair = max(max_pair, float((means[i] - means[j]).abs().max()))
    return max_pair


def main():
    print("=" * 74)
    print("FULL-SEPARATION MoE SMOKE (2026-06-26): 5 cores x 3 nets, radar_encoder per core")
    print("=" * 74)
    allok = True

    print("\n[A] structure (OFF=core / MoE=experts[5], independent radar_encoder)")
    for k, (n, he, hc, ir, ok) in test_structure().items():
        allok &= ok
        print(f"   {k:10s}: cores={n} experts={he} core={hc} indep_radar={ir}  {'PASS' if ok else 'FAIL'}")

    print("\n[B] forward shapes (comm path)")
    for k, (ash, vsh, ok) in test_forward().items():
        allok &= ok
        print(f"   {k:4s}: action{ash} value{vsh}  {'PASS' if ok else 'FAIL'}")

    print("\n[C] PPO logprob/value mirror (rollout forward -> evaluate_actions)")
    for attn in (False, True):
        for moe in (False, True):
            lp, vd = _mirror(attn, moe)
            ok = lp < 1e-4 and vd < 1e-4
            allok &= ok
            tag = f"{'attn' if attn else 'sum':4s}+{'moe' if moe else 'off'}"
            print(f"   {tag:9s}: |lp diff|={lp:.2e} |v diff|={vd:.2e}  {'PASS' if ok else 'FAIL (<1e-4)'}")

    print("\n[D] others_msg mirror (sender routing)")
    for attn in (False, True):
        diff = test_others_msg_mirror(attn, use_moe=True)
        ok = diff < 1e-6
        allok &= ok
        print(f"   {'attn' if attn else 'sum':4s}+moe: max|diff|={diff:.2e}  {'PASS' if ok else 'FAIL (<1e-6)'}")

    print("\n[E] comm-OFF invariance (others_msg==0 under MoE)")
    m = test_comm_off()
    ok = (m == 0.0); allok &= ok
    print(f"   max|others_msg|={m:.2e}  {'PASS' if ok else 'FAIL (!=0)'}")

    print("\n[F] per-situation grad isolation INCLUDING radar_encoder (core of 'full separation')")
    for k, (fh, oz, ok) in test_grad_isolation().items():
        allok &= ok
        print(f"   {k:10s}: fired_core_radar_grad={fh}  other_cores_zero={oz}  {'PASS' if ok else 'FAIL'}")

    print("\n[G] routing changes output (distinct cores per situation)")
    mp = test_routing_changes_output()
    ok = mp > 1e-4; allok &= ok
    print(f"   max pairwise |action_mean diff| over situations = {mp:.3e}  {'PASS' if ok else 'FAIL (~0 => cores identical)'}")

    print("\n" + "=" * 74)
    print("VERDICT:", "ALL PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
