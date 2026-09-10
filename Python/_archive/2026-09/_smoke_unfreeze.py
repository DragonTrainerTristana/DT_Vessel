# 채널 해동(2026-06-12) 스모크 테스트: init에서 통신 채널 양단에 gradient가 실제로 흐르는지.
# 사용: VESSEL_USE_COMM=1 [VESSEL_USE_ATTENTION=0/1] python _smoke_unfreeze.py
import os
import torch
import numpy as np

from config import (MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES, STATE_SIZE,
                    MAX_COMM_PARTNERS, USE_COMMUNICATION, USE_ATTENTION,
                    GOAL_SIZE, SELF_STATE_SIZE, MSG_L2_COEF)
from networks import CNNPolicy

torch.manual_seed(0)
np.random.seed(0)

mode = 'attention' if USE_ATTENTION else 'sum'
print(f"[smoke] comm={USE_COMMUNICATION} mode={mode} msg_dim={MSG_DIM} K={MAX_COMM_PARTNERS}")

policy = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES)
# ★완전분리 MoE(2026-06-26): fc2/msg_gate/msg_out이 코어로 이동 → 대표 코어(단일=core, MoE=experts[0])로 진단.
ctr_core = policy.ctr_actor.cores()[0]
cri_core = policy.critic.cores()[0]
msg_core = policy.msg_actor.cores()[0]
N, K = 5, MAX_COMM_PARTNERS
obs_dim = FRAMES * STATE_SIZE

# ── init 상태 확인 ──
g_ctr = torch.sigmoid(ctr_core.msg_gate).item()
g_cri = torch.sigmoid(cri_core.msg_gate).item()
print(f"  gate init: ctr={g_ctr:.3f} critic={g_cri:.3f} (기대 0.5)")
print(f"  msg_out norm={msg_core.msg_out.weight.norm():.4f} (기대 >0)")
print(f"  v_proj  norm={policy.attn.v_proj.weight.norm():.4f} (기대 >0)")
print(f"  fc2slice ctr={ctr_core.fc2.weight[:, -MSG_DIM:].norm():.4f} critic={cri_core.fc2.weight[:, -MSG_DIM:].norm():.4f} (기대 >0)")

# ── rollout forward (벡터화 attention 경로 포함) ──
x = torch.randn(1, N, obs_dim)
goal = torch.randn(1, N, GOAL_SIZE)
self_state = torch.randn(1, N, SELF_STATE_SIZE)
agent_ids = [f"a{i}" for i in range(N)]
comm_partners = {aid: [p for p in agent_ids if p != aid][:K] for aid in agent_ids}
comm_relpos = {aid: np.random.randn(len(comm_partners[aid]), 3).astype(np.float32)
               for aid in agent_ids}
with torch.no_grad():
    value, action, logprob, mean, msg, others = policy.forward(
        x, goal, self_state, return_msg=True,
        comm_partners=comm_partners, agent_id_list=agent_ids, comm_relpos=comm_relpos)
print(f"  rollout: |msg|={msg.abs().mean():.5f} |others_msg|={others.abs().mean():.5f} (둘 다 >0 기대, comm ON일 때)")

# ── update 경로: evaluate_actions backward → 채널 양단 grad ──
px = torch.randn(N, K, obs_dim)
pg = torch.randn(N, K, GOAL_SIZE)
ps = torch.randn(N, K, SELF_STATE_SIZE)
pm = (torch.rand(N, K, 1) > 0.3).float()
pr = torch.randn(N, K, 3)
act = torch.tanh(torch.randn(N, 1, CONTINUOUS_ACTION_SIZE)) * 0.9

v, lp, ent, msg_reg, intent_loss, threat_loss, goal_loss, role_loss, consumer_loss = policy.evaluate_actions(
    x.squeeze(0).unsqueeze(1), goal.squeeze(0).unsqueeze(1), self_state.squeeze(0).unsqueeze(1),
    px, pg, ps, pm, pr, act)
loss = lp.mean() + v.mean() + MSG_L2_COEF * msg_reg
loss.backward()

def gmax(p):
    return 0.0 if p.grad is None else p.grad.abs().max().item()

checks = {
    'msg_out.weight': gmax(msg_core.msg_out.weight),
    'fc2slice_ctr': (ctr_core.fc2.weight.grad[:, -MSG_DIM:].abs().max().item()
                     if ctr_core.fc2.weight.grad is not None else 0.0),
    'fc2slice_critic': (cri_core.fc2.weight.grad[:, -MSG_DIM:].abs().max().item()
                        if cri_core.fc2.weight.grad is not None else 0.0),
    'gate_ctr': gmax(ctr_core.msg_gate),
    'gate_critic': gmax(cri_core.msg_gate),
    'v_proj.weight': gmax(policy.attn.v_proj.weight),
}
ok = True
for name, g in checks.items():
    if name == 'v_proj.weight' and not USE_ATTENTION:
        print(f"  grad {name:18s} = {g:.3e} (sum 모드: 미사용, 0 정상)")
        continue
    expect = USE_COMMUNICATION
    alive = g > 0
    status = 'OK' if alive == expect else 'FAIL'
    if alive != expect:
        ok = False
    print(f"  grad {name:18s} = {g:.3e} [{status}]")

print(f"[smoke] {'PASS' if ok else 'FAIL'} ({mode})")
raise SystemExit(0 if ok else 1)
