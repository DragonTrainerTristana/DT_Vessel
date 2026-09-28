"""test_p6_diff.py — G8a: 역할 약속 변경 전 커밋(기본 ae58b93) 대비 차분 테스트 (2026-09-29, 스펙 §5).

같은 드라이버를 옛 코드(git worktree)와 현재 코드에서 각각 돌려, 토글을 끈 경로가 텐서 단위로 같은지 본다:
  default(YUGIOH, EXT 끔) · EXT intent(코덱 없음) · a6 decode(p6) · c6 direct(p6)
  비교: comm_gather 의 others_msg·prelpos, 정책 행동, env 보상·done 을 30결정 동안 torch.equal.
옛 커밋은 --base 로 바꿀 수 있다. 워크트리는 임시 폴더에 만들고 끝나면 지운다.

  python verify/test_p6_diff.py [--base ae58b93]      # 마지막 줄 VERDICT: ALL PASS
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
REPO = os.path.dirname(PYROOT)

DRIVER = r'''
import os, sys, torch
sys.path.insert(0, sys.argv[1])
import config as cfg, vessel_gym as vg, vessel_gym_train as T, comm_codec
from networks import CNNPolicy
torch.set_num_threads(1)
comm_codec.install(cfg.COMM_CODEC, cfg.COMM_CODEC_SHA, cfg.COMM_CODEC_MODE, 'cpu')
torch.manual_seed(0)
E, N = 6, 16
env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device='cpu', seed=5, ring_scale=1.0, crossing=0,
                        risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE, farfield_coef=0.0,
                        perpair_coef=-0.15, perpair_exp=3.0)
pol = CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES)
fs = T.FrameStack(E, N, 'cpu')
obs = env.reset(); r_, g, ss, st = T.parse_obs(obs); fs.reset_all(r_)
out = {'om': [], 'prel': [], 'a': [], 'r': [], 'd': []}
g_ = torch.Generator().manual_seed(9)
for t in range(30):
    x = fs.get()
    with torch.no_grad():
        om, parts = T.comm_gather(pol, env, x, g, ss, st, 4)
        mean = pol.ctr_actor(x, g, ss, om, st)[2]
    a = torch.tanh(mean) * 0.5 + (torch.rand(E, N, 2, generator=g_) * 2 - 1) * 0.5
    obs, rew, done, oc = env.step(a)
    r_, g, ss, st = T.parse_obs(obs); fs.push(r_, done)
    for k, v in (('om', om), ('prel', parts[4]), ('a', a), ('r', rew), ('d', done)):
        out[k].append(v.clone())
torch.save(out, sys.argv[2])
print('OK', cfg.COMM_EXT, cfg.COMM_CODEC_MODE)
'''

CASES = [
    ('default(EXT off)', {}),
    ('EXT intent', {'VESSEL_COMM_EXT': '1', 'VESSEL_USE_ATTENTION': '1', 'VESSEL_COMM_FIELDS': 'intent'}),
    ('a6 decode p6', {'VESSEL_COMM_EXT': '1', 'VESSEL_USE_ATTENTION': '1', 'VESSEL_COMM_FIELDS': 'intent',
                      'VESSEL_COMM_LATENT': '0.0', 'VESSEL_AUX_LOSS_SCALE': '0.0',
                      'VESSEL_COMM_CODEC': 'comm_codecs/p6_k6_s0.pt', 'VESSEL_COMM_CODEC_SHA': 'fbe4c71a6bf4',
                      'VESSEL_COMM_CODEC_MODE': 'decode'}),
    ('c6 direct p6', {'VESSEL_COMM_EXT': '1', 'VESSEL_USE_ATTENTION': '1', 'VESSEL_COMM_FIELDS': 'intent',
                      'VESSEL_COMM_LATENT': '0.0', 'VESSEL_AUX_LOSS_SCALE': '0.0',
                      'VESSEL_COMM_CODEC': 'comm_codecs/p6_k6_s0.pt', 'VESSEL_COMM_CODEC_SHA': 'fbe4c71a6bf4',
                      'VESSEL_COMM_CODEC_MODE': 'direct'}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='ae58b93')
    a = ap.parse_args()
    import torch
    tmp = tempfile.mkdtemp(prefix='p6diff_')
    old = os.path.join(tmp, 'old')
    subprocess.run(['git', '-C', REPO, 'worktree', 'add', '--detach', old, a.base], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    drv = os.path.join(tmp, 'drv.py')
    with open(drv, 'w', encoding='utf-8') as f:
        f.write(DRIVER)
    res = []
    try:
        for name, extra in CASES:
            env = {k: v for k, v in os.environ.items() if not k.startswith('VESSEL_')}
            env.update({'VESSEL_DYN_PROFILE': 'imo', 'VESSEL_OBSTACLES': 'none', 'OMP_NUM_THREADS': '1',
                        'PYTHONIOENCODING': 'utf-8', 'CUDA_VISIBLE_DEVICES': '-1'})
            env.update(extra)
            outs = {}
            for tag, root in (('old', os.path.join(old, 'Python')), ('new', PYROOT)):
                fp = os.path.join(tmp, f'{tag}.pt')
                p = subprocess.run([sys.executable, drv, root, fp], env=env, cwd=root, capture_output=True, text=True,
                                   encoding='utf-8', errors='replace')
                if p.returncode != 0:
                    print(p.stderr[-3000:])
                    raise SystemExit(f'[p6diff] {name} {tag} 실패 rc={p.returncode}')
                outs[tag] = torch.load(fp)
            same = all(torch.equal(x, y) for k in outs['old'] for x, y in zip(outs['old'][k], outs['new'][k]))
            res.append(same)
            print(f"  {'PASS' if same else '★FAIL'}  {name:18s} base {a.base} vs 현재 — others_msg·prelpos·행동·보상·done 30결정 torch.equal")
    finally:
        subprocess.run(['git', '-C', REPO, 'worktree', 'remove', '--force', old], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"VERDICT: {'ALL PASS' if all(res) else 'FAIL ' + str(res.count(False))}")
    return 0 if all(res) else 1


if __name__ == '__main__':
    sys.exit(main())
