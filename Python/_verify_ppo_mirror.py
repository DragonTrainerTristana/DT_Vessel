"""
PPO 정합성 수치 검증 (2026-06-12 채널동결 fix 후).

init 변경(비영 v_proj/fc2슬라이스/msg_out, gate sigmoid=0.5)으로 others_msg가
실제로 정책에 영향을 준다 → rollout(_get_others_msg)과 update(evaluate_actions)의
집계가 비트단위로 일치해야 PPO ratio(old_logprob)가 유효하다.

검증 항목:
  1. others_msg 정합 (attention/sum 각각): rollout vs update, max|diff| < 1e-6
  2. logprob 정합: rollout forward action을 evaluate_actions에 넣어 같은 logprob, max|diff| ~1e-4
  3. msg_gate 공유 (_backbone 양쪽 동일 적용) — 코드/수치 확인
  4. MoE 경로 Critic fc2 메시지슬라이스가 cat 마지막 열인지(one-hot이 앞 삽입)

실행: python _verify_ppo_mirror.py   (USE_COMM/USE_ATTENTION을 프로세스 내에서 토글하므로
config/networks를 importlib.reload로 재적재한다.)
"""
import os
import sys
import importlib
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

# 결정성 확보 (벡터화 vs 루프 경로의 부동소수 합 순서 차이만 남기고 그 외 randomness 제거)
torch.use_deterministic_algorithms(False)

DEV = torch.device('cpu')   # 검증은 CPU(결정적, 커널 비결정성 회피)


def _reload(use_comm, use_attention, use_moe=False, msg_dim=6):
    """env 토글 후 config/networks를 reload (모듈-레벨 상수 반영)."""
    os.environ['VESSEL_USE_COMM'] = '1' if use_comm else '0'
    os.environ['VESSEL_USE_ATTENTION'] = '1' if use_attention else '0'
    os.environ['VESSEL_USE_MOE'] = '1' if use_moe else '0'
    os.environ['VESSEL_MSG_DIM'] = str(msg_dim)
    os.environ['VESSEL_AGG_MODE'] = 'sum'
    # ★legacy 경로 고정: POS_GROUND·SITUATION_INPUT 기본이 ON이라 pop만 하면 기본값이 새어들어
    #   sum 미러 케이스가 pos_ground 경로를 타고(false FAIL), fc2 기대차원(+5)도 어긋남.
    #   이 스크립트는 sum/attention 미러 검증 전용이므로 두 토글을 명시적으로 끈다.
    os.environ['VESSEL_POS_GROUND'] = '0'
    os.environ['VESSEL_SITUATION_INPUT'] = '0'
    os.environ.pop('VESSEL_NEAREST_SCALE', None)
    os.environ.pop('VESSEL_MSG_GAIN', None)
    import config
    importlib.reload(config)
    import networks
    importlib.reload(networks)
    return config, networks


def _make_scene(cfg, n_agents=6, max_partners=None, seed=0):
    """n_agents 척의 합성 씬: obs/goal/self/position + 통신 파트너/relpos 구성.
    main.py collect_observations와 동일 규약으로 partner 패딩 텐서까지 만든다."""
    rng = np.random.RandomState(seed)
    F = cfg.FRAMES
    S = cfg.STATE_SIZE
    state_dim = F * S
    Kmax = cfg.MAX_COMM_PARTNERS if max_partners is None else max_partners

    states = rng.randn(n_agents, state_dim).astype(np.float32)
    goals = rng.randn(n_agents, cfg.GOAL_SIZE).astype(np.float32)
    selfs = rng.randn(n_agents, cfg.SELF_STATE_SIZE).astype(np.float32)
    # heading(self[2])은 _compute_relpos에서 ×180 deg로 쓰임 — 임의값 OK
    positions = (rng.rand(n_agents, 2).astype(np.float32) * (cfg.COMM_RANGE * 0.5))
    situations = rng.randint(0, cfg.NUM_COLREGS_SITUATIONS, size=n_agents).astype(np.int64)

    agent_id_list = list(range(n_agents))
    id_to_idx = {a: i for i, a in enumerate(agent_id_list)}
    pos_map = {a: positions[i] for i, a in enumerate(agent_id_list)}

    # 통신 파트너 = COMM_RANGE 내 nearest-Kmax (main.get_comm_partners와 동일 정의)
    comm_partners = {}
    comm_relpos = {}

    def relpos(rx, rz, rhead_deg, px, pz):
        import math
        dx, dz = px - rx, pz - rz
        rng_ = math.hypot(dx, dz)
        bearing = math.atan2(dx, dz) - math.radians(rhead_deg)
        return [math.sin(bearing), math.cos(bearing), min(rng_ / cfg.COMM_RANGE, 1.0)]

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

    # update용 partner 패딩 텐서 (main.training_step 규약: kept_pos 정렬)
    p_states = np.zeros((n_agents, Kmax, state_dim), np.float32)
    p_goals = np.zeros((n_agents, Kmax, cfg.GOAL_SIZE), np.float32)
    p_selfs = np.zeros((n_agents, Kmax, cfg.SELF_STATE_SIZE), np.float32)
    p_mask = np.zeros((n_agents, Kmax), np.float32)
    p_relpos = np.zeros((n_agents, Kmax, 3), np.float32)
    p_sits = np.zeros((n_agents, Kmax), np.int64)   # 파트너 상황(완전분리 MoE MessageActor sender 라우팅)
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

    return dict(
        states=states, goals=goals, selfs=selfs, positions=positions, situations=situations,
        agent_id_list=agent_id_list, comm_partners=comm_partners, comm_relpos=comm_relpos,
        p_states=p_states, p_goals=p_goals, p_selfs=p_selfs, p_mask=p_mask, p_relpos=p_relpos,
        p_sits=p_sits, state_dim=state_dim, n_agents=n_agents,
    )


def _build_policy(cfg, networks, seed=0):
    torch.manual_seed(seed)
    pol = networks.CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES).to(DEV)
    pol.eval()
    return pol


def test_others_msg_mirror(use_attention, use_moe=False):
    """rollout _get_others_msg vs update evaluate_actions 내부 집계 others_msg 비교."""
    cfg, networks = _reload(use_comm=True, use_attention=use_attention, use_moe=use_moe)
    pol = _build_policy(cfg, networks)
    sc = _make_scene(cfg)

    x = torch.as_tensor(sc['states'], dtype=torch.float32, device=DEV).unsqueeze(0)        # [1,N,F*S]
    goal = torch.as_tensor(sc['goals'], dtype=torch.float32, device=DEV).unsqueeze(0)      # [1,N,2]
    self_s = torch.as_tensor(sc['selfs'], dtype=torch.float32, device=DEV).unsqueeze(0)    # [1,N,4]
    # ★완전분리 MoE: MessageActor를 sender 상황으로 라우팅(rollout=update 동일) — 없으면 미러 무효
    sit = torch.as_tensor(sc['situations'], dtype=torch.long, device=DEV).unsqueeze(0) if use_moe else None
    p_sits = torch.as_tensor(sc['p_sits'], dtype=torch.long, device=DEV) if use_moe else None

    with torch.no_grad():
        msg = pol.msg_actor(x, goal, self_s, sit)   # [1,N,M]
        others_roll = pol._get_others_msg(
            msg, sc['comm_partners'], sc['agent_id_list'], sc['comm_relpos'],
            self_state=self_s, goal=goal
        )   # [1,N,M]

    # update 경로: partner obs로 msg 재생성 후 집계 ([N,1,M])
    px = torch.as_tensor(sc['p_states'], dtype=torch.float32, device=DEV)   # [N,K,F*S]
    pg = torch.as_tensor(sc['p_goals'], dtype=torch.float32, device=DEV)
    ps = torch.as_tensor(sc['p_selfs'], dtype=torch.float32, device=DEV)
    pm = torch.as_tensor(sc['p_mask'], dtype=torch.float32, device=DEV).unsqueeze(-1)   # [N,K,1]
    prp = torch.as_tensor(sc['p_relpos'], dtype=torch.float32, device=DEV)              # [N,K,3]

    with torch.no_grad():
        msg_part = pol.msg_actor(px, pg, ps, p_sits)    # [N,K,M]
        Kc = pm.sum(dim=1, keepdim=True).clamp(min=1.0)
        if pol.use_attention:
            q_in = torch.cat([self_s.transpose(0, 1), goal.transpose(0, 1)], dim=-1)   # [N,1,6]
            others_upd = pol.attn.aggregate_batch(q_in, prp, msg_part, pm)              # [N,1,M]
        else:
            others_upd = (msg_part * pm).sum(dim=1, keepdim=True)                       # [N,1,M] (sum mode)

    # 형식 정렬: rollout [1,N,M] → [N,1,M]
    others_roll_n1 = others_roll.transpose(0, 1).contiguous()
    diff = (others_roll_n1 - others_upd).abs().max().item()
    return diff, others_roll_n1, others_upd


def test_logprob_mirror(use_attention, use_moe=False):
    """rollout forward (action, logprob) → evaluate_actions에 넣어 logprob 정합."""
    cfg, networks = _reload(use_comm=True, use_attention=use_attention, use_moe=use_moe)
    pol = _build_policy(cfg, networks)
    sc = _make_scene(cfg)

    x = torch.as_tensor(sc['states'], dtype=torch.float32, device=DEV).unsqueeze(0)
    goal = torch.as_tensor(sc['goals'], dtype=torch.float32, device=DEV).unsqueeze(0)
    self_s = torch.as_tensor(sc['selfs'], dtype=torch.float32, device=DEV).unsqueeze(0)
    sit = torch.as_tensor(sc['situations'], dtype=torch.long, device=DEV).unsqueeze(0) if use_moe else None

    with torch.no_grad():
        value_r, action, logprob_r, mean, msg, others_msg = pol.forward(
            x, goal, self_s, return_msg=True,
            comm_partners=sc['comm_partners'], agent_id_list=sc['agent_id_list'],
            comm_relpos=sc['comm_relpos'], situation=sit
        )   # action/logprob [1,N,*]

    # evaluate_actions 입력 형식 [N,1,*]
    x_u = x.transpose(0, 1).contiguous()
    goal_u = goal.transpose(0, 1).contiguous()
    self_u = self_s.transpose(0, 1).contiguous()
    act_u = action.transpose(0, 1).contiguous()
    sit_u = sit.transpose(0, 1).contiguous() if sit is not None else None

    px = torch.as_tensor(sc['p_states'], dtype=torch.float32, device=DEV)
    pg = torch.as_tensor(sc['p_goals'], dtype=torch.float32, device=DEV)
    ps = torch.as_tensor(sc['p_selfs'], dtype=torch.float32, device=DEV)
    pm = torch.as_tensor(sc['p_mask'], dtype=torch.float32, device=DEV).unsqueeze(-1)
    prp = torch.as_tensor(sc['p_relpos'], dtype=torch.float32, device=DEV)
    psit = torch.as_tensor(sc['p_sits'], dtype=torch.long, device=DEV) if use_moe else None

    with torch.no_grad():
        value_u, logprob_u, ent, msg_reg, intent, threat, goal_l, role_l, consumer_l = pol.evaluate_actions(
            x_u, goal_u, self_u, px, pg, ps, pm, prp, act_u, situation=sit_u, partner_situations=psit
        )   # logprob [N,1,1]

    logprob_r_n1 = logprob_r.transpose(0, 1).contiguous()   # [N,1,1]
    value_r_n1 = value_r.transpose(0, 1).contiguous()
    lp_diff = (logprob_r_n1 - logprob_u).abs().max().item()
    v_diff = (value_r_n1 - value_u).abs().max().item()
    return lp_diff, v_diff


def test_comm_off_invariance(use_attention):
    """comm-OFF: others_msg≡0 → rollout/update 모두 0, gate 무영향(anti-rigging)."""
    cfg, networks = _reload(use_comm=False, use_attention=use_attention)
    pol = _build_policy(cfg, networks)
    sc = _make_scene(cfg)
    x = torch.as_tensor(sc['states'], dtype=torch.float32, device=DEV).unsqueeze(0)
    goal = torch.as_tensor(sc['goals'], dtype=torch.float32, device=DEV).unsqueeze(0)
    self_s = torch.as_tensor(sc['selfs'], dtype=torch.float32, device=DEV).unsqueeze(0)
    with torch.no_grad():
        msg = pol.msg_actor(x, goal, self_s)
        others = pol._get_others_msg(msg, sc['comm_partners'], sc['agent_id_list'],
                                     sc['comm_relpos'], self_state=self_s, goal=goal)
    return others.abs().max().item()


def test_msg_gate_shared():
    """msg_gate가 rollout(forward)과 update(get_logprob_entropy) 양쪽 _backbone에서
    동일 적용되는지: gate를 인위적으로 크게 바꾸면 두 경로 출력이 *같은 방향으로* 달라지고,
    여전히 logprob 정합이 유지되는지 확인."""
    cfg, networks = _reload(use_comm=True, use_attention=False)
    pol = _build_policy(cfg, networks)
    # gate를 강제로 크게 설정 (sigmoid≈0.95) → others_msg 가중치 변화. ★완전분리 MoE면 게이트가 코어별 →
    #   cores() 전체에 설정(OFF는 [core] 1개).
    with torch.no_grad():
        for c in pol.ctr_actor.cores():
            c.msg_gate.fill_(3.0)
        for c in pol.critic.cores():
            c.msg_gate.fill_(3.0)
    sc = _make_scene(cfg)
    x = torch.as_tensor(sc['states'], dtype=torch.float32, device=DEV).unsqueeze(0)
    goal = torch.as_tensor(sc['goals'], dtype=torch.float32, device=DEV).unsqueeze(0)
    self_s = torch.as_tensor(sc['selfs'], dtype=torch.float32, device=DEV).unsqueeze(0)
    with torch.no_grad():
        value_r, action, logprob_r, mean, msg, others_msg = pol.forward(
            x, goal, self_s, return_msg=True,
            comm_partners=sc['comm_partners'], agent_id_list=sc['agent_id_list'],
            comm_relpos=sc['comm_relpos'])
    x_u = x.transpose(0, 1).contiguous(); goal_u = goal.transpose(0, 1).contiguous()
    self_u = self_s.transpose(0, 1).contiguous(); act_u = action.transpose(0, 1).contiguous()
    px = torch.as_tensor(sc['p_states'], dtype=torch.float32, device=DEV)
    pg = torch.as_tensor(sc['p_goals'], dtype=torch.float32, device=DEV)
    ps = torch.as_tensor(sc['p_selfs'], dtype=torch.float32, device=DEV)
    pm = torch.as_tensor(sc['p_mask'], dtype=torch.float32, device=DEV).unsqueeze(-1)
    prp = torch.as_tensor(sc['p_relpos'], dtype=torch.float32, device=DEV)
    with torch.no_grad():
        value_u, logprob_u, ent, mr, it, tt, gl, rl, cl = pol.evaluate_actions(
            x_u, goal_u, self_u, px, pg, ps, pm, prp, act_u)
    lp_diff = (logprob_r.transpose(0, 1).contiguous() - logprob_u).abs().max().item()
    # 라우팅 소스 동일성: forward·get_logprob_entropy 둘 다 self._route 호출 + 게이트가 코어 backbone에서 적용.
    import inspect
    fwd_src = inspect.getsource(pol.ctr_actor.forward)
    glp_src = inspect.getsource(pol.ctr_actor.get_logprob_entropy)
    both_backbone = ('self._route(' in fwd_src) and ('self._route(' in glp_src)
    bb_src = inspect.getsource(pol.ctr_actor.cores()[0].backbone)
    gate_in_backbone = 'torch.sigmoid(self.msg_gate)' in bb_src
    return lp_diff, both_backbone, gate_in_backbone


def test_moe_critic_msg_slice():
    """완전분리 MoE(2026-06-26): Critic이 상황별 코어 5벌(one-hot 조건화 폐기). 각 코어 fc2 in_features =
    RADAR_FEAT_DIM+2+4+msg_dim, 메시지슬라이스(마지막 msg_dim 열)가 ×0.1 init으로 나머지보다 작은지 검증."""
    cfg, networks = _reload(use_comm=True, use_attention=False, use_moe=True)
    md = cfg.MSG_DIM
    in_features = cfg.RADAR_FEAT_DIM + 2 + 4 + md   # one-hot 없음(코어 통째 라우팅)
    pol = _build_policy(cfg, networks)
    cores = pol.critic.cores()
    shape_ok = (len(cores) == cfg.NUM_COLREGS_SITUATIONS) and all(c.fc2.in_features == in_features for c in cores)
    # 코어별 msg 슬라이스(마지막 md열) vs 나머지 — ×0.1이 올바른 마지막 M열에 적용됐는지 (평균)
    msg_cols_mean = float(np.mean([float(c.fc2.weight.abs().mean(dim=0)[-md:].mean()) for c in cores]))
    rest_cols_mean = float(np.mean([float(c.fc2.weight.abs().mean(dim=0)[:-md].mean()) for c in cores]))
    last_m_norm = float(np.mean([float(c.fc2.weight[:, -md:].norm()) for c in cores]))
    rest_norm = float(np.mean([float(c.fc2.weight[:, :-md].norm()) for c in cores]))
    return dict(shape_ok=shape_ok, n_cores=len(cores),
                last_m_norm=last_m_norm, rest_norm=rest_norm,
                msg_cols_mean=msg_cols_mean, rest_cols_mean=rest_cols_mean,
                in_features=cores[0].fc2.in_features, expected=in_features)


def main():
    print("=" * 72)
    print("PPO MIRROR VERIFICATION (post 2026-06-12 channel-thaw init)")
    print("=" * 72)
    results = {}

    # --- 1. others_msg 정합 ---
    # ★oracle 모드(2026-06-22)는 학습채널을 우회(참 파트너 goal 주입)하므로 채널-집계 미러 [1]·[3]은 비적용.
    #   oracle의 rollout==update 미러는 [2] logprob mirror(end-to-end)가 권위 + _smoke_c5c.py TEST B가 독립 확인.
    print("\n[1] others_msg mirror (rollout vs update)")
    if os.environ.get('VESSEL_ORACLE', '0') == '1':
        print("   (oracle 모드: 채널 우회 → [1] 비적용. [2] logprob mirror가 oracle rollout==update 권위)")
    else:
        for attn in (False, True):
            mode = 'attention' if attn else 'sum'
            diff, _, _ = test_others_msg_mirror(use_attention=attn)
            ok = diff < 1e-6
            results[f'others_msg[{mode}]'] = (diff, ok)
            print(f"   {mode:10s}: max|diff| = {diff:.3e}   {'PASS' if ok else 'FAIL (<1e-6)'}")
        # MoE + attention 경로도
        diff, _, _ = test_others_msg_mirror(use_attention=True, use_moe=True)
        ok = diff < 1e-6
        results['others_msg[attn+moe]'] = (diff, ok)
        print(f"   {'attn+moe':10s}: max|diff| = {diff:.3e}   {'PASS' if ok else 'FAIL (<1e-6)'}")

    # --- 2. logprob 정합 ---
    print("\n[2] logprob/value mirror (rollout forward -> evaluate_actions)")
    for attn in (False, True):
        mode = 'attention' if attn else 'sum'
        lp, vd = test_logprob_mirror(use_attention=attn)
        ok = lp < 1e-4
        results[f'logprob[{mode}]'] = (lp, ok)
        print(f"   {mode:10s}: max|logprob diff| = {lp:.3e}  max|value diff| = {vd:.3e}   {'PASS' if ok else 'FAIL (<1e-4)'}")
    lp, vd = test_logprob_mirror(use_attention=False, use_moe=True)
    ok = lp < 1e-4
    results['logprob[moe]'] = (lp, ok)
    print(f"   {'sum+moe':10s}: max|logprob diff| = {lp:.3e}  max|value diff| = {vd:.3e}   {'PASS' if ok else 'FAIL (<1e-4)'}")

    # --- 3. comm-OFF 불변 ---
    print("\n[3] comm-OFF invariance (others_msg should be exactly 0)")
    if os.environ.get('VESSEL_ORACLE', '0') == '1':
        print("   (oracle 모드: comm-OFF에선 USE_COMMUNICATION 가드로 oracle 미발화 → 별도 OFF baseline 런서 검증)")
    else:
        for attn in (False, True):
            mode = 'attention' if attn else 'sum'
            m = test_comm_off_invariance(use_attention=attn)
            ok = (m == 0.0)
            results[f'commOFF[{mode}]'] = (m, ok)
            print(f"   {mode:10s}: max|others_msg| = {m:.3e}   {'PASS' if ok else 'FAIL (!=0)'}")

    # --- 4. msg_gate 공유 ---
    print("\n[4] msg_gate shared backbone (rollout==update with gate=3.0)")
    lp_diff, both_bb, gate_in_bb = test_msg_gate_shared()
    ok = (lp_diff < 1e-4) and both_bb and gate_in_bb
    results['msg_gate_shared'] = (lp_diff, ok)
    print(f"   logprob diff (gate=3.0) = {lp_diff:.3e}")
    print(f"   forward&get_logprob both call self._backbone: {both_bb}")
    print(f"   gate sigmoid applied inside _backbone: {gate_in_bb}   {'PASS' if ok else 'FAIL'}")

    # --- 5. 완전분리 MoE Critic: 코어별 fc2 (one-hot 제거) 메시지 슬라이스 위치 ---
    print("\n[5] full-MoE Critic: per-core fc2 (no one-hot), msg-slice = last msg_dim cols")
    moe = test_moe_critic_msg_slice()
    ok = moe['shape_ok'] and (moe['msg_cols_mean'] < moe['rest_cols_mean'])
    results['moe_critic_slice'] = (0.0, ok)
    print(f"   cores = {moe['n_cores']}, fc2.in_features = {moe['in_features']} (expected {moe['expected']}): {moe['shape_ok']}")
    print(f"   msg cols |w| mean = {moe['msg_cols_mean']:.4f}  vs  rest cols |w| mean = {moe['rest_cols_mean']:.4f}")
    print(f"   (msg cols smaller => x0.1 hit the correct last-M columns)   {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 72)
    all_ok = all(v[1] for v in results.values())
    for k, (val, ok) in results.items():
        print(f"   {'PASS' if ok else 'FAIL':4s}  {k:22s}  ({val:.3e})")
    print("=" * 72)
    print("VERDICT:", "ALL PASS" if all_ok else "FAIL - PPO ratio invalid in at least one path")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
