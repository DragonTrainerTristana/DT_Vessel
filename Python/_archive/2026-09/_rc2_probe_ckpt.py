# RC2 적대적 검증: 실제 sweep 체크포인트에서 메시지 채널 가중치 동결 여부 실측 (일회용)
import os, sys, glob, re, torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models", "COMM_YES_PHASE3_NEW"))

RUNS = {
    # group A: 06-08 sum, intent=0
    "A_msg2": "VesselNavigation_20260608_132523",
    "A_msg6": "VesselNavigation_20260608_132530",
    "A_msg12": "VesselNavigation_20260608_132543",
    # group B: 06-09 attn=1, intent=0.05, farfield=0.3
    "B_msg2": "VesselNavigation_20260609_131039",
    "B_msg6": "VesselNavigation_20260609_131047",
    "B_msg12": "VesselNavigation_20260609_131059",
    # group C: 06-10 attn=1, intent=0, farfield=2.0, crossing
    "C_msg2": "VesselNavigation_20260610_150735",
    "C_msg6": "VesselNavigation_20260610_150739",
    "C_msg12": "VesselNavigation_20260610_150743",
}

def stepnum(p):
    m = re.search(r"policy_step_(\d+)\.pth", p)
    return int(m.group(1)) if m else -1

def load_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck

def norms(sd, msg_dim):
    out = {}
    out["msg_out.W"] = sd["msg_actor.msg_out.weight"].norm().item()
    out["msg_out.b"] = sd["msg_actor.msg_out.bias"].norm().item()
    out["ctr.fc2_msgslice"] = sd["ctr_actor.fc2.weight"][:, -msg_dim:].norm().item()
    out["ctr.fc2_rest"] = sd["ctr_actor.fc2.weight"][:, :-msg_dim].norm().item()
    out["cri.fc2_msgslice"] = sd["critic.fc2.weight"][:, -msg_dim:].norm().item()
    out["v_proj.W"] = sd["attn.v_proj.weight"].norm().item()
    out["v_proj.b"] = sd["attn.v_proj.bias"].norm().item()
    out["ctr.gate"] = sd["ctr_actor.msg_gate"].item()
    out["cri.gate"] = sd["critic.msg_gate"].item()
    out["msgactor.fc2.W"] = sd["msg_actor.fc2.weight"].norm().item()
    out["msgactor.radar.c1"] = sd["msg_actor.radar_encoder.conv1.weight"].norm().item() if "msg_actor.radar_encoder.conv1.weight" in sd else float("nan")
    return out

for tag, folder in RUNS.items():
    d = os.path.join(ROOT, folder)
    pths = sorted(glob.glob(os.path.join(d, "policy_step_*.pth")), key=stepnum)
    if not pths:
        print(f"{tag}: NO CKPT in {folder}")
        continue
    first, last = pths[0], pths[-1]
    msg_dim = int(tag.split("msg")[1])
    sd_f, sd_l = load_sd(first), load_sd(last)
    nf, nl = norms(sd_f, msg_dim), norms(sd_l, msg_dim)
    # msg_actor 본체 동결 여부: first vs last 가중치 최대 절대차
    frozen_keys = [k for k in sd_f if k.startswith("msg_actor.")]
    maxdiff = max((sd_f[k] - sd_l[k]).abs().max().item() for k in frozen_keys)
    print(f"== {tag} ({folder}) steps {stepnum(first)}->{stepnum(last)}")
    print(f"   last: msg_out.W={nl['msg_out.W']:.3e} b={nl['msg_out.b']:.3e} | ctr.fc2_msgslice={nl['ctr.fc2_msgslice']:.3e} (rest={nl['ctr.fc2_rest']:.2f}) | cri.fc2_msgslice={nl['cri.fc2_msgslice']:.3e}")
    print(f"   last: v_proj.W={nl['v_proj.W']:.3e} b={nl['v_proj.b']:.3e} | gate ctr={nl['ctr.gate']:.3f} cri={nl['cri.gate']:.3f} (sig={torch.sigmoid(torch.tensor(nl['ctr.gate'])).item():.4f}/{torch.sigmoid(torch.tensor(nl['cri.gate'])).item():.4f})")
    print(f"   msg_actor first-vs-last max|dW|={maxdiff:.3e} | msgactor.fc2.W first={nf['msgactor.fc2.W']:.3f} last={nl['msgactor.fc2.W']:.3f}")
