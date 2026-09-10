# 일회용: 06-02 H1a 게이트fix "검증" 런들의 메시지 채널 동결 여부 실측
import os, glob, re, torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models", "COMM_YES_PHASE3_NEW"))
CANDIDATES = [
    "VesselNavigation_20260601_212408", "VesselNavigation_20260601_212410",
    "VesselNavigation_20260601_212414", "VesselNavigation_20260601_212418",
    "VesselNavigation_20260601_212423", "VesselNavigation_20260601_212427",
    "VesselNavigation_20260602_003254", "VesselNavigation_20260602_003258",
    "VesselNavigation_20260602_003303", "VesselNavigation_20260602_003123",
    "VesselNavigation_20260602_181503", "VesselNavigation_20260602_181626",
]

def stepnum(p):
    m = re.search(r"policy_step_(\d+)\.pth", p)
    return int(m.group(1)) if m else -1

def load_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck

for folder in CANDIDATES:
    d = os.path.join(ROOT, folder)
    pths = sorted(glob.glob(os.path.join(d, "policy_step_*.pth")), key=stepnum)
    if not pths:
        print(f"{folder}: NO CKPT ({len(glob.glob(os.path.join(d,'*')))} entries)")
        continue
    last = pths[-1]
    sd = load_sd(last)
    keys = list(sd.keys())
    mo = sd.get("msg_actor.msg_out.weight")
    msg_dim = mo.shape[0] if mo is not None else -1
    line = f"{folder} step={stepnum(last)} msg_dim={msg_dim}"
    if mo is not None:
        line += f" | msg_out.W={mo.norm().item():.3e}"
    for k, tag in [("ctr_actor.fc2.weight", "ctr.fc2_slice"), ("critic.fc2.weight", "cri.fc2_slice")]:
        if k in sd and msg_dim > 0:
            line += f" | {tag}={sd[k][:, -msg_dim:].norm().item():.3e}"
    for k, tag in [("ctr_actor.msg_gate", "ctr.gate"), ("critic.msg_gate", "cri.gate")]:
        if k in sd:
            line += f" | {tag}={sd[k].item():.3f}"
    if "attn.v_proj.weight" in sd:
        line += f" | v_proj={sd['attn.v_proj.weight'].norm().item():.3e}"
    print(line)
