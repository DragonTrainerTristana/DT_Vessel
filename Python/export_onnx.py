"""
PyTorch → ONNX (ML-Agents Sentis 호환, 최소 포맷)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (STATE_SIZE, GOAL_SIZE, SELF_STATE_SIZE, COLREGS_SIZE,
                     OBSERVATION_SIZE, FRAMES, MSG_DIM, CONTINUOUS_ACTION_SIZE,
                     PROJECT_ROOT)
from networks import CNNPolicy

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "COMM_YES_PHASE2_v2",
                          "VesselNavigation_20260310_171041", "policy_step_16670000.pth")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "VesselNavigation_16M.onnx")
ASSETS_PATH = os.path.join(PROJECT_ROOT, "Assets", "Models", "VesselNavigation_16M.onnx")


class VesselONNXModel(nn.Module):
    def __init__(self, control_actor):
        super().__init__()
        self.conv1 = control_actor.conv1
        self.conv2 = control_actor.conv2
        self.fc1 = control_actor.fc1
        self.fc2 = control_actor.fc2
        self.fc3 = control_actor.fc3
        self.action_mean_layer = control_actor.action_mean
        self.register_buffer('zero_msg', torch.zeros(1, MSG_DIM))

    def forward(self, obs):
        # obs: [batch, 373]
        batch = obs.shape[0]

        # radar [batch, 360] → 3채널로 복제 [batch, 3, 360]
        r = obs[:, 0:STATE_SIZE].unsqueeze(1)
        radar = torch.cat([r, r, r], dim=1)

        goal = obs[:, STATE_SIZE:STATE_SIZE+GOAL_SIZE]
        ss = obs[:, STATE_SIZE+GOAL_SIZE:STATE_SIZE+GOAL_SIZE+SELF_STATE_SIZE]
        cr = obs[:, STATE_SIZE+GOAL_SIZE+SELF_STATE_SIZE:
                    STATE_SIZE+GOAL_SIZE+SELF_STATE_SIZE+COLREGS_SIZE]
        msg = self.zero_msg.expand(batch, -1)

        a = F.relu(self.conv1(radar))
        a = F.relu(self.conv2(a))
        a = a.reshape(batch, -1)
        a = F.relu(self.fc1(a))
        a = torch.cat([a, goal, ss, cr, msg], dim=1)
        a = torch.tanh(self.fc2(a))
        a = torch.tanh(self.fc3(a))

        mean = torch.clamp(self.action_mean_layer(a), -3.0, 3.0)
        action = torch.tanh(mean)
        return action


def main():
    print(f"Loading: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    policy = CNNPolicy(frames=FRAMES, msg_dim=MSG_DIM, action_size=CONTINUOUS_ACTION_SIZE)
    policy.load_state_dict(ckpt)
    policy.eval()

    model = VesselONNXModel(policy.ctr_actor)
    model.eval()

    dummy = torch.randn(1, OBSERVATION_SIZE)
    with torch.no_grad():
        out = model(dummy)
        print(f"Action: {out}")

    torch.onnx.export(
        model, dummy, OUTPUT_PATH,
        input_names=["obs_0"],
        output_names=["continuous_actions"],
        dynamic_axes={"obs_0": {0: "batch"}, "continuous_actions": {0: "batch"}},
        opset_version=9,
        do_constant_folding=True
    )

    # onnxruntime 검증
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(OUTPUT_PATH)
    print(f"\nInputs:  {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"Outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")
    res = sess.run(None, {'obs_0': np.random.randn(1, OBSERVATION_SIZE).astype(np.float32)})
    print(f"Test: {res[0]}")

    os.makedirs(os.path.dirname(ASSETS_PATH), exist_ok=True)
    shutil.copy2(OUTPUT_PATH, ASSETS_PATH)
    print(f"\nDone: {ASSETS_PATH} ({os.path.getsize(OUTPUT_PATH)/1024/1024:.1f} MB)")
    print(f"Unity: Stacked Vectors=1, Inference Only, Space Size=373")


if __name__ == "__main__":
    main()
