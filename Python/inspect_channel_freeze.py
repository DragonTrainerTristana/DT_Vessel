# 진단 전용(1회성): 학습된 체크포인트에서 통신 채널 파라미터가 init에서 움직였는지 검사.
# 가설: msg_out(0)·v_proj(0)·fc2 msg슬라이스(0) 직렬 zero-init → 채널 gradient 항등 0 → 1M step 후에도 정확히 0.
import sys
import torch

def inspect(path):
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    elif isinstance(sd, dict) and 'policy' in sd:
        sd = sd['policy']
    keys = list(sd.keys())
    print(f"== {path}")
    def stat(k):
        if k in sd:
            t = sd[k]
            print(f"  {k:50s} absmax={t.abs().max().item():.3e}  norm={t.norm().item():.3e}")
        else:
            print(f"  {k:50s} (없음)")
    stat('msg_actor.msg_out.weight')
    stat('msg_actor.msg_out.bias')
    stat('attn.v_proj.weight')
    stat('attn.v_proj.bias')
    stat('attn.k_proj.weight')
    stat('attn.q_proj.weight')
    for k in keys:
        if 'msg_gate' in k:
            print(f"  {k:50s} value={sd[k].item():.6f}  sigmoid={torch.sigmoid(sd[k]).item():.6f}")
    # fc2 메시지 슬라이스 (마지막 msg_dim 열). msg_dim은 msg_out.weight 행수에서 추론.
    md = sd['msg_actor.msg_out.weight'].shape[0]
    for net in ('ctr_actor', 'critic'):
        k = f'{net}.fc2.weight'
        if k in sd:
            sl = sd[k][:, -md:]
            rest = sd[k][:, :-md]
            print(f"  {k} msg슬라이스[{md}열]                    absmax={sl.abs().max().item():.3e}   (참고: 나머지 absmax={rest.abs().max().item():.3e})")
    # MessageActor 본체(radar_encoder/fc2)가 init에서 움직였는지 → intent 없으면 gradient 0이어야 함
    print(f"  (msg_dim 추론값 = {md})")

for p in sys.argv[1:]:
    inspect(p)
