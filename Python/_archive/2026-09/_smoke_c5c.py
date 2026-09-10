# C5c(수신측 decode-in-policy) + oracle-OFF 스모크 테스트 (2026-06-22)
# 검증:
#   (1) evaluate_actions가 9-tuple(…, consumer_loss) 반환.
#   (2) C5c: consumer_loss backward가 *수신측* ctr_actor.fc2 메시지슬라이스로 gradient 흐름(단절 해소 실증).
#   (3) oracle: rollout(_get_others_msg) 주입 == update masked-mean 파트너 goal (PPO 미러).
# 사용:
#   C5c grad:  VESSEL_USE_COMM=1 VESSEL_COMM_CONSUMER_COEF=0.05 VESSEL_USE_ATTENTION=1 python _smoke_c5c.py
#   oracle:    VESSEL_USE_COMM=1 VESSEL_ORACLE=1 python _smoke_c5c.py
import os
import torch
import numpy as np

from config import (MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES, STATE_SIZE, GOAL_SIZE,
                    MAX_COMM_PARTNERS, USE_COMMUNICATION, USE_ATTENTION, USE_ORACLE,
                    COMM_CONSUMER_COEF, MSG_L2_COEF)
from networks import CNNPolicy

torch.manual_seed(0)
np.random.seed(0)

print(f"[c5c] comm={USE_COMMUNICATION} oracle={USE_ORACLE} consumer_coef={COMM_CONSUMER_COEF} "
      f"attn={USE_ATTENTION} msg_dim={MSG_DIM} K={MAX_COMM_PARTNERS} goal_size={GOAL_SIZE}")

policy = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES)
# ★완전분리 MoE(2026-06-26): fc2/consumer_decoder/msg_out이 코어로 이동 → 대표 코어로 진단(단일=core, MoE=experts[0]).
ctr_core = policy.ctr_actor.cores()[0]
msg_core = policy.msg_actor.cores()[0]
N, K = 6, MAX_COMM_PARTNERS
obs_dim = FRAMES * STATE_SIZE
ok = True

# ── 공통 입력 ──
x = torch.randn(1, N, obs_dim)
goal = torch.randn(1, N, GOAL_SIZE)
self_state = torch.randn(1, N, 4)
agent_ids = [f"a{i}" for i in range(N)]
# 각 에이전트 파트너 = 다음 2개(순환) → comm_partners/relpos 구성
comm_partners = {aid: [agent_ids[(i + 1) % N], agent_ids[(i + 2) % N]] for i, aid in enumerate(agent_ids)}
comm_relpos = {aid: np.random.randn(len(comm_partners[aid]), 3).astype(np.float32) for aid in agent_ids}

# ── (1) 9-tuple arity (update 경로) ──
px = torch.randn(N, K, obs_dim)
pg = torch.randn(N, K, GOAL_SIZE)
ps = torch.randn(N, K, 4)
pm = (torch.rand(N, K, 1) > 0.3).float()
pr = torch.randn(N, K, 3)
act = torch.tanh(torch.randn(N, 1, CONTINUOUS_ACTION_SIZE)) * 0.9

out = policy.evaluate_actions(
    x.squeeze(0).unsqueeze(1), goal.squeeze(0).unsqueeze(1), self_state.squeeze(0).unsqueeze(1),
    px, pg, ps, pm, pr, act)
arity_ok = (len(out) == 9)
print(f"  (1) evaluate_actions arity = {len(out)} (기대 9) [{'OK' if arity_ok else 'FAIL'}]")
ok = ok and arity_ok
value, logprob, entropy, msg_reg, intent_l, threat_l, goal_l, role_l, consumer_l = out

# ── (2) C5c: consumer_loss grad가 수신측 fc2 메시지슬라이스로 흐르는가 ──
if USE_COMMUNICATION and COMM_CONSUMER_COEF > 0.0 and not USE_ORACLE:
    print(f"  (2) consumer_loss = {float(consumer_l):.5f} (기대 >0)")
    policy.zero_grad()
    consumer_l.backward()
    fc2_slice_grad = ctr_core.fc2.weight.grad[:, -MSG_DIM:].abs().max().item()
    cons_head_grad = ctr_core.consumer_decoder[0].weight.grad.abs().max().item()
    # 생산자(MessageActor)도 attention 경로로 gradient 받는지(폐루프 보너스)
    msgout_grad = (msg_core.msg_out.weight.grad.abs().max().item()
                   if msg_core.msg_out.weight.grad is not None else 0.0)
    s_slice = 'OK' if fc2_slice_grad > 0 else 'FAIL'        # ★C5c 핵심: 수신 정책 메시지슬라이스에 gradient
    s_head = 'OK' if cons_head_grad > 0 else 'FAIL'
    if fc2_slice_grad <= 0 or cons_head_grad <= 0:
        ok = False
    print(f"      grad ctr_actor.fc2[:, -msg:] = {fc2_slice_grad:.3e} [{s_slice}]  ← C5c 수신측 단절 해소")
    print(f"      grad consumer_decoder        = {cons_head_grad:.3e} [{s_head}]")
    print(f"      grad msg_actor.msg_out       = {msgout_grad:.3e} (폐루프: 생산자도 수신손실로 학습)")
else:
    print(f"  (2) consumer_loss = {float(consumer_l):.5f} (coef=0 또는 oracle → 0 기대) "
          f"[{'OK' if float(consumer_l) == 0.0 else 'FAIL'}]")
    ok = ok and (float(consumer_l) == 0.0)

# ── (3) oracle 미러: rollout 주입 == update masked-mean 파트너 goal ──
if USE_ORACLE and not USE_COMMUNICATION:
    # ★가드 검증: COMM=0이면 oracle 미발화(comm-OFF baseline 불변) → others_msg ≡ 0
    with torch.no_grad():
        msg = policy.msg_actor(x, goal, self_state)
        others_roll = policy._get_others_msg(msg, comm_partners, agent_ids, comm_relpos,
                                             self_state=self_state, goal=goal)
    m = others_roll.abs().max().item()
    s_guard = 'OK' if m < 1e-6 else 'FAIL'
    if m >= 1e-6:
        ok = False
    print(f"  (3) oracle guard (COMM=0 → oracle 미발화): max|others_msg| = {m:.3e} [{s_guard}]")
elif USE_ORACLE:
    with torch.no_grad():
        msg = policy.msg_actor(x, goal, self_state)
        others_roll = policy._get_others_msg(msg, comm_partners, agent_ids, comm_relpos,
                                             self_state=self_state, goal=goal)   # [1,N,M]
    # update 형식 파트너 텐서 구성(comm_partners와 일관) → masked-mean
    id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
    pg2 = torch.zeros(N, K, GOAL_SIZE)
    pm2 = torch.zeros(N, K, 1)
    for i, aid in enumerate(agent_ids):
        for j, p in enumerate(comm_partners[aid][:K]):
            pg2[i, j] = goal[0, id_to_idx[p]]
            pm2[i, j, 0] = 1.0
    masked_mean = (pg2 * pm2).sum(dim=1) / pm2.sum(dim=1).clamp(min=1.0)   # [N,GOAL_SIZE]
    roll_goal = others_roll[0, :, :GOAL_SIZE]                              # [N,GOAL_SIZE]
    _rest = others_roll[0, :, GOAL_SIZE:]                                  # MSG_DIM==GOAL_SIZE면 빈 슬라이스
    rest_zero = _rest.abs().max().item() if _rest.numel() > 0 else 0.0
    diff = (roll_goal - masked_mean).abs().max().item()
    s_mirror = 'OK' if diff < 1e-5 else 'FAIL'
    s_rest = 'OK' if rest_zero < 1e-6 else 'FAIL'
    if diff >= 1e-5 or rest_zero >= 1e-6:
        ok = False
    print(f"  (3) oracle mirror |rollout - masked_mean| = {diff:.3e} [{s_mirror}]")
    print(f"      oracle 나머지 차원 0 = {rest_zero:.3e} [{s_rest}]")

print(f"[c5c] {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
