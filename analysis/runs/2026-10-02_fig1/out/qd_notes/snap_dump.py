"""Print the training-relevant cfg_snapshot keys and per-expert action_logstd of checkpoints. CPU, read-only."""
import sys, time
import torch
print(time.strftime('%H:%M:%S'))
KEYS = ('code_version', 'comm_range', 'seed', 'logstd_init0', 'lr_final', 'lr_decay_dec', 'entropy_bonus', 'logstd_min', 'noise_hold',
        'sim.REWARD_RANGE', 'sim.ROLE_V2_PRIMARY', 'sim.ROLE_V2_RES_F6', 'comm_fields', 'partner_range', 'action_mode')
for p in sys.argv[1:]:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    snap = ck.get('cfg_snapshot', {})
    print(f"== {p.replace(chr(92), '/').split('/')[-1]} | top keys: {list(ck)}")
    print(f"   seed = {ck.get('seed')}")
    print(f"   steps = {ck.get('steps')}")
    for k in KEYS:
        v = snap
        for part in k.split('.'):
            v = v.get(part, '키 없음') if isinstance(v, dict) else '키 없음'
        print(f"   cfg_snapshot.{k} = {v}")
    sd = ck['model_state_dict']
    for k in sorted(k for k in sd if k.startswith('ctr_actor') and k.endswith('action_logstd')):
        print(f"   {k} = {[round(float(x), 3) for x in sd[k].reshape(-1)]}")
