"""_verify_comm_mirror.py — vessel_gym_train 통신 경로의 rollout=update 검증.

기존 `_verify_ppo_mirror.py` 는 networks.py 의 Unity 경로(_get_others_msg vs evaluate_actions)를 본다.
이 파일은 **GPU 배치 학습 경로**(vessel_gym_train.comm_gather vs evaluate_actions)를 본다 — 둘은 다른 코드다.

왜 필요한가: PPO 는 rollout 이 저장한 old_logprob 과 update 가 다시 계산한 logprob 이
같은 입력에서 나왔다고 가정한다. others_msg 가 두 경로에서 달라지면 ratio 가 1 에서 벗어나
정책 변화와 무관하게 clip 이 걸리고, 학습이 조용히 망가진다(에러 없이).

2026-08-31 comm_gather 최적화(메시지를 배당 1회 계산 후 gather) 이후 회귀 방지용.
"""
import os
import sys

import torch


def run_case(name, env_overrides, K, use_masks):
    for k, v in env_overrides.items():
        os.environ[k] = v
    for m in ('config', 'networks', 'vessel_gym', 'vessel_gym_train'):
        sys.modules.pop(m, None)
    import config as cfg
    import vessel_gym as vg
    import vessel_gym_train as T
    from networks import CNNPolicy

    E, N = 8, 16
    torch.manual_seed(0)
    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device='cpu', seed=5, ring_scale=1.0,
                            crossing=0, risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE,
                            farfield_coef=0.0, perpair_coef=-0.15, perpair_exp=3.0)
    pol = CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES)
    fs = T.FrameStack(E, N, 'cpu')
    obs = env.reset()
    r, g, ss, st = T.parse_obs(obs)
    fs.reset_all(r)

    sm = rm = None
    if use_masks:                      # 혼합 함대(일부 선박 통신 불가)
        nc = torch.zeros(E, N, dtype=torch.bool)
        nc[:, ::4] = True
        sm, rm = ~nc, ~nc

    for _ in range(30):                # 위상 분산 + 상황 다양화
        x = fs.get()
        with torch.no_grad():
            om, _ = T.comm_gather(pol, env, x, g, ss, st, K, send_mask=sm, recv_mask=rm)
            a, _, _, _ = pol.ctr_actor(x, g, ss, om, st)
        obs, _, d, _ = env.step(a)
        r, g, ss, st = T.parse_obs(obs)
        fs.push(r, d)

    x = fs.get()
    with torch.no_grad():
        om_roll, (px, pg, ps, pm, pr, psi) = T.comm_gather(
            pol, env, x, g, ss, st, K, send_mask=sm, recv_mask=rm)
        _, logp_roll, _, araw = pol.ctr_actor(x, g, ss, om_roll, st)
        v_roll = pol.critic(x, g, ss, om_roll, st)

    M = E * N
    fl = lambda t: t.reshape(M, *t.shape[2:])
    othr = othrm = None
    if cfg.THREAT_COEF > 0:
        othr, othrm = T.compute_own_threat(fl(x), cfg.THREAT_K, 'cpu')
        othr, othrm = othr.unsqueeze(1), othrm.unsqueeze(1)
    with torch.no_grad():
        v_upd, logp_upd, _, _, _, _, _, _, _ = pol.evaluate_actions(
            fl(x), fl(g), fl(ss), fl(px), fl(pg), fl(ps), fl(pm), fl(pr), fl(araw),
            own_threat=othr, own_threat_mask=othrm,
            situation=fl(st), partner_situations=fl(psi))

    d_lp = float((logp_upd.squeeze(1).squeeze(-1) - logp_roll.reshape(M)).abs().max())
    d_v = float((v_upd.reshape(M) - v_roll.reshape(M)).abs().max())
    ratio = torch.exp(logp_upd.squeeze(1).squeeze(-1) - logp_roll.reshape(M))
    ok = d_lp < 1e-5 and d_v < 1e-4
    print(f"   {'PASS' if ok else '★FAIL'}  {name:28s} logp {d_lp:.2e}  value {d_v:.2e}  "
          f"ratio [{float(ratio.min()):.6f}, {float(ratio.max()):.6f}]")
    return ok


BASE = dict(VESSEL_USE_COMM='1', VESSEL_MOE_SHARED='1', VESSEL_THREAT_COEF='0.5',
            VESSEL_MSG_LN='1', VESSEL_POS_GROUND='1', VESSEL_USE_ATTENTION='0')
CASES = [
    ('MoE + threat + LN',      {**BASE, 'VESSEL_USE_MOE': '1'}, 4, False),
    ('단일망(MoE off)',         {**BASE, 'VESSEL_USE_MOE': '0'}, 4, False),
    ('nearest-1 (K=1)',        {**BASE, 'VESSEL_USE_MOE': '1'}, 1, False),
    ('혼합함대 마스크',           {**BASE, 'VESSEL_USE_MOE': '1'}, 4, True),
    ('threat off',             {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_THREAT_COEF': '0'}, 4, False),
    ('LayerNorm off(옛 구조)',  {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_MSG_LN': '0'}, 4, False),
    ('MSG_DIM=12',             {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_MSG_DIM': '12'}, 4, False),
    # ★attention 케이스 (2026-09-04): 이 케이스가 없던 동안 comm_gather에 attention 분기가 없어
    #   rollout=mean vs update=attention 인 ratio 붕괴를 조용히 놓쳤음. 재발 방지.
    ('attention 집계',          {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_USE_ATTENTION': '1'}, 4, False),
    ('attention + dim12',      {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_USE_ATTENTION': '1',
                                'VESSEL_MSG_DIM': '12'}, 4, False),
    # ★pos_ground=0 케이스 (2026-09-05): 이 케이스가 없던 동안 comm_gather 에 sum/mean 분기가 없어
    #   VESSEL_POS_GROUND=0(sum·mean 대조군)에서 rollout=pos_ground vs update=sum 인 ratio 붕괴를
    #   조용히 놓쳤음. attention 누락(09-04)과 같은 계열의 blocker. 재발 방지.
    ('pos_ground off (sum)',   {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_POS_GROUND': '0',
                                'VESSEL_AGG_MODE': 'sum'}, 4, False),
    ('pos_ground off (mean)',  {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_POS_GROUND': '0',
                                'VESSEL_AGG_MODE': 'mean'}, 4, False),
    # ★msg_gain 케이스 (2026-09-05): gain 이 update(networks) 에만 걸리고 rollout(comm_gather) 에
    #   누락돼 있었음. 1.0 이 아니면 ratio 가 깨짐.
    ('msg_gain 0.5',           {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_MSG_GAIN': '0.5'}, 4, False),
    # ★msg_token_gain 케이스 (2026-09-07): attention 토큰 안에서 msg 를 상수배하는 스위치.
    #   GroundedAttention 안에 있어 rollout·update 가 같은 함수를 타지만, 만약 누군가
    #   comm_gather 쪽에 토큰을 직접 만드는 코드를 넣으면 즉시 갈린다 — 그걸 잡는 케이스.
    #   attention 경로에만 유효하므로 USE_ATTENTION=1 과 함께 건다.
    ('msg_token_gain 8',       {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_USE_ATTENTION': '1',
                                'VESSEL_MSG_TOKEN_GAIN': '8.0'}, 4, False),
    ('token_gain 8 + dim12',   {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_USE_ATTENTION': '1',
                                'VESSEL_MSG_TOKEN_GAIN': '8.0', 'VESSEL_MSG_DIM': '12'}, 4, False),
    # ★공유 인코더 (2026-09-10): MessageActor(·Critic)가 ControlActor 인코더 객체를 씀.
    #   rollout(comm_gather 가 파트너 메시지 생성)·update(evaluate_actions 재생성) 둘 다 같은 객체를 타야 ratio 유효.
    ('shared enc actor',        {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_SHARED_ENCODER': 'actor'}, 4, False),
    ('shared enc all',          {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_SHARED_ENCODER': 'all'}, 4, False),
    ('shared all + attention',  {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_SHARED_ENCODER': 'all', 'VESSEL_USE_ATTENTION': '1'}, 4, False),
    ('shared all, MoE 비공유',   {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_MOE_SHARED': '0', 'VESSEL_SHARED_ENCODER': 'all'}, 4, False),
    # ★ABLATION_PLAN §3 구조 4종 × 최종 인코더 설정(공유 all + bottleneck + leaky) (2026-09-10).
    #   단일망(USE_MOE=0)·얇게(0.416 = bn+SE 에서 단일망과 −0.5% iso-param)·두껍게·공유MoE 가
    #   전부 같은 인코더 객체를 rollout/update 에서 타는지. A~D 배치 전 필수 통과.
    ('플랜 단일망 + SE/bn/leaky',      {**BASE, 'VESSEL_USE_MOE': '0', 'VESSEL_SHARED_ENCODER': 'all',
                                      'VESSEL_RADAR_HEAD': 'bottleneck', 'VESSEL_RADAR_ACT': 'leaky'}, 4, False),
    ('플랜 얇게0.416 + SE/bn/leaky',   {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_MOE_WIDTH': '0.416', 'VESSEL_MOE_SHARED': '0',
                                      'VESSEL_SHARED_ENCODER': 'all', 'VESSEL_RADAR_HEAD': 'bottleneck', 'VESSEL_RADAR_ACT': 'leaky'}, 4, False),
    ('플랜 두껍게 + SE/bn/leaky',      {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_MOE_SHARED': '0',
                                      'VESSEL_SHARED_ENCODER': 'all', 'VESSEL_RADAR_HEAD': 'bottleneck', 'VESSEL_RADAR_ACT': 'leaky'}, 4, False),
    ('플랜 공유MoE + SE/bn/leaky',     {**BASE, 'VESSEL_USE_MOE': '1', 'VESSEL_MOE_SHARED': '1',
                                      'VESSEL_SHARED_ENCODER': 'all', 'VESSEL_RADAR_HEAD': 'bottleneck', 'VESSEL_RADAR_ACT': 'leaky'}, 4, False),
]

if __name__ == '__main__':
    print("=" * 78)
    print("vessel_gym_train 통신 경로 mirror (rollout comm_gather == update evaluate_actions)")
    print("=" * 78)
    results = [run_case(n, e, k, m) for n, e, k, m in CASES]
    print("=" * 78)
    print(f"VERDICT: {'ALL PASS' if all(results) else 'FAIL ' + str(results.count(False))}")
    sys.exit(0 if all(results) else 1)
