# 채널 건강 텔레메트리(2026-06-30 A2Z 감사) 스모크: main.py에 넣은 grad-norm 캡처 + gate-mean 로직 검증 (Unity 불필요).
# 사용: VESSEL_USE_COMM=1 [VESSEL_USE_ATTENTION=0/1] python _smoke_telemetry.py
import torch
import numpy as np
from config import (MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES, STATE_SIZE,
                    MAX_COMM_PARTNERS, USE_COMMUNICATION, USE_ATTENTION,
                    GOAL_SIZE, SELF_STATE_SIZE, MSG_L2_COEF)
from networks import CNNPolicy

torch.manual_seed(0)
np.random.seed(0)

policy = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES)
N, K = 5, MAX_COMM_PARTNERS
obs_dim = FRAMES * STATE_SIZE
x = torch.randn(N, 1, obs_dim); goal = torch.randn(N, 1, GOAL_SIZE); ss = torch.randn(N, 1, SELF_STATE_SIZE)
px = torch.randn(N, K, obs_dim); pg = torch.randn(N, K, GOAL_SIZE); ps = torch.randn(N, K, SELF_STATE_SIZE)
pm = (torch.rand(N, K, 1) > 0.3).float(); pr = torch.randn(N, K, 3)
act = torch.tanh(torch.randn(N, 1, CONTINUOUS_ACTION_SIZE)) * 0.9

out = policy.evaluate_actions(x, goal, ss, px, pg, ps, pm, pr, act)
v, lp, ent, msg_reg = out[0], out[1], out[2], out[3]
loss = lp.mean() + v.mean() + MSG_L2_COEF * msg_reg
policy.zero_grad()
loss.backward()

# ===== main.py update_policy에 넣은 캡처 로직 그대로 (검증 대상) =====
cg_msgout = cg_fc2ctr = cg_fc2cri = cg_gate = cg_vproj = 0.0
def _gnorm(t):
    return 0.0 if t is None else float(t.norm())
if USE_COMMUNICATION:
    try:
        _mc = policy.msg_actor.cores()
        _md = _mc[0].msg_out.weight.shape[0]
        for _c in _mc:
            cg_msgout += _gnorm(_c.msg_out.weight.grad)
        for _c in policy.ctr_actor.cores():
            _g = _c.fc2.weight.grad
            cg_fc2ctr += 0.0 if _g is None else float(_g[:, -_md:].norm())
            cg_gate += _gnorm(_c.msg_gate.grad)
        for _c in policy.critic.cores():
            _g = _c.fc2.weight.grad
            cg_fc2cri += 0.0 if _g is None else float(_g[:, -_md:].norm())
            cg_gate += _gnorm(_c.msg_gate.grad)
        if getattr(policy, 'attn', None) is not None:
            cg_vproj += _gnorm(policy.attn.v_proj.weight.grad)
    except Exception as e:
        print("CAPTURE EXC:", e)
# =====================================================================

gm_c = [float(torch.sigmoid(c.msg_gate)) for c in policy.ctr_actor.cores()]
gm_v = [float(torch.sigmoid(c.msg_gate)) for c in policy.critic.cores()]

mode = 'attention' if USE_ATTENTION else 'sum'
print(f"[telemetry] comm={USE_COMMUNICATION} mode={mode} msg_dim={MSG_DIM} cores={len(policy.ctr_actor.cores())}")
print(f"  Grad_MsgOut={cg_msgout:.3e} Fc2Ctr={cg_fc2ctr:.3e} Fc2Critic={cg_fc2cri:.3e} Gate={cg_gate:.3e} Vproj={cg_vproj:.3e}")
print(f"  GateMean ctr={sum(gm_c)/len(gm_c):.3f} critic={sum(gm_v)/len(gm_v):.3f} (기대 0.5)")

if USE_COMMUNICATION:
    ok = cg_msgout > 0 and cg_fc2ctr > 0 and cg_fc2cri > 0 and cg_gate > 0 and (cg_vproj > 0 if USE_ATTENTION else cg_vproj == 0)
else:
    ok = cg_msgout == 0 and cg_fc2ctr == 0 and cg_fc2cri == 0 and cg_gate == 0
print(f"[telemetry] {'PASS' if ok else 'FAIL'} ({mode})")
raise SystemExit(0 if ok else 1)
