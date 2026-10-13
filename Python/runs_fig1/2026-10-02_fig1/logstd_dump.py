"""logstd_dump.py — print the ControlActor action_logstd (per MoE expert) of checkpoints. CPU, read-only.
Usage: python logstd_dump.py <ckpt.pt> [<ckpt.pt> ...]   (spec 2026-10-13-trunk-fork-design.md, diag D-a)"""
import math
import sys

import torch

for p in sys.argv[1:]:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    keys = sorted(k for k in sd if k.startswith('ctr_actor') and k.endswith('action_logstd'))
    vals = [float(sd[k].reshape(-1)[0]) for k in keys]
    mean0 = sum(vals) / len(vals) if vals else float('nan')
    sig = 120.0 * math.exp(mean0)   # course mode: sigma(deg) of the course command before tanh squash
    print(f"{p.split('/')[-1]:34s} steps={ck.get('steps')} logstd0={' '.join(f'{v:.3f}' for v in vals)} "
          f"mean={mean0:.3f} sigma_course={sig:.1f}deg", flush=True)
