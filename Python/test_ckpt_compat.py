"""test_ckpt_compat.py — 체크포인트 호환 골든 (2026-09-10, 리팩토링 Tier 0.4).

체크포인트가 있는 머신(Windows, VESSEL_CKPT_DIR)에서 돌린다. 이 맥엔 .pt 가 없음.

무엇을 확인하나
  각 .pt 를 ckpt_io.restore_policy 로 열어(strict 로드) 고정 시드 관측 배치 1회 forward →
  ctr_actor 평균행동·critic 값·msg_actor 메시지의 SHA256. 골든과 바이트 단위 비교.
  → 리팩토링(config 통합·죽은 코드 제거)이 12개 체크포인트의 *로드·추론* 을 조금도 바꾸지 않았음을 증명.

쓰는 법 (Windows)
  set VESSEL_CKPT_DIR=C:\\...\\VESSEL_checkpoints
  python test_ckpt_compat.py --regen  cf_OFF_s43.pt cf_OFF_s44.pt ...   # 리팩토링 *전* 커밋(4bcfa4b)에서 1회
  python test_ckpt_compat.py --check  cf_OFF_s43.pt cf_OFF_s44.pt ...   # 리팩토링 후
  인자 없이 주면 VESSEL_CKPT_DIR 의 *.pt 전부.
  ⚠️ --regen 은 리팩토링 전 코드로 만들어야 의미가 있다. 이미 리팩토링 후라면 git worktree 로 4bcfa4b 를 꺼내 거기서 --regen.

주의: comm_range 가 ckpt 와 다르면 restore_policy 가 중단한다(VESSEL_COMM_RANGE 를 ckpt 학습값으로 주고 실행).
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, 'golden', '2026-09-10_ckpt.json')

_FMT = {'torch.float32': 'f', 'torch.float64': 'd', 'torch.int64': 'q', 'torch.int32': 'i', 'torch.bool': 'B'}


def _sha(t):
    import array
    t = t.detach().cpu().contiguous().flatten()
    vals = t.to(torch.uint8).tolist() if t.dtype == torch.bool else t.tolist()
    return hashlib.sha256(array.array(_FMT[str(t.dtype)], vals).tobytes()).hexdigest()


def fingerprint(ckpt, device='cpu'):
    import config as cfg
    import vessel_gym as vg
    from ckpt_io import restore_policy, make_env_from_snapshot
    from vessel_gym_train import comm_gather, parse_obs, FrameStack, make_others_msg
    r = restore_policy(ckpt, device, tag='[compat]')
    E, N = 4, 16
    env = make_env_from_snapshot(r.snap, device=device, num_envs=E, seed=777, n_vessels=N, tag='[compat]')
    fs = FrameStack(E, N, device)
    obs = env.reset()
    radar, goal, self_s, sit = parse_obs(obs)
    fs.reset_all(radar)
    torch.manual_seed(777)
    with torch.no_grad():
        for _ in range(20):                       # 위상 흩기 (결정적)
            x = fs.get()
            om = comm_gather(r.policy, env, x, goal, self_s, sit, r.max_partners)[0] if r.arm == 'ON' \
                else make_others_msg(env, r.arm, E, N, device)
            a, _, _, _ = r.policy.ctr_actor(x, goal, self_s, om, sit)
            obs, _, done, _ = env.step(a)
            radar, goal, self_s, sit = parse_obs(obs)
            fs.push(radar, done)
        x = fs.get()
        om = comm_gather(r.policy, env, x, goal, self_s, sit, r.max_partners)[0] if r.arm == 'ON' \
            else make_others_msg(env, r.arm, E, N, device)
        a, _, _, _ = r.policy.ctr_actor(x, goal, self_s, om, sit)
        v = r.policy.critic(x, goal, self_s, om, sit)
        m = r.policy.msg_actor(x, goal, self_s, sit)
    return {'action': _sha(a), 'value': _sha(v), 'msg': _sha(m), 'others_msg': _sha(om),
            'state_dict': hashlib.sha256(''.join(f'{k}:{_sha(t)}' for k, t in sorted(r.state_dict.items())).encode()).hexdigest(),
            'n_keys': len(r.state_dict), 'arm': r.arm, 'msg_dim': r.msg_dim, 'effective': r.effective}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='*')
    ap.add_argument('--regen', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--device', default='cpu', help='cpu 권장 (GPU 는 커널 비결정 가능)')
    a = ap.parse_args()
    ckpts = a.ckpts or sorted(os.path.basename(p) for p in glob.glob(os.path.join(os.environ.get('VESSEL_CKPT_DIR', ''), '*.pt')))
    if not ckpts:
        print('체크포인트 없음 — 인자로 주거나 VESSEL_CKPT_DIR 설정'); sys.exit(2)
    if a.regen:
        rec = {c: fingerprint(c, a.device) for c in ckpts}
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        json.dump(rec, open(GOLDEN, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'골든 생성 {len(rec)}개 → {GOLDEN}'); return
    gold = json.load(open(GOLDEN, encoding='utf-8')) if os.path.exists(GOLDEN) else {}
    ok = True
    for c in ckpts:
        if c not in gold:
            print(f'  ★없음  {c} (골든에 없음 → --regen)'); ok = False; continue
        cur = fingerprint(c, a.device)
        bad = [k for k in ('action', 'value', 'msg', 'others_msg', 'state_dict', 'n_keys') if gold[c][k] != cur[k]]
        print(f"  {'PASS' if not bad else '★FAIL'}  {c:20s} keys={cur['n_keys']} arm={cur['arm']}" + (f'  차이: {bad}' if bad else ''))
        ok = ok and not bad
    print(f"VERDICT: {'ALL PASS' if ok else 'FAIL'}"); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
