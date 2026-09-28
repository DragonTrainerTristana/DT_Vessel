"""test_eval_diff.py — G8b: eval_ckpt 출력 바이트 차분 (2026-09-29, 스펙 §5). Windows(체크포인트가 있는 곳) 전용.

옛 커밋(기본 ae58b93)의 eval_ckpt.py 와 현재 eval_ckpt.py 로 같은 체크포인트를 작은 창에서 평가해 stdout 을 비교한다.
허용 차이는 ① 진행 표시 줄(경과·ETA·dec/s — 벽시계) ② 새 줄 '[role-promise]' 뿐. 나머지 줄은 글자 하나까지 같아야 한다
(= 역할 약속 판정기를 켜도 기존 지표·난수·상태가 안 바뀜).

  python verify/test_eval_diff.py --ckpt h_off_s43.pt [--ckpt h_a6_s43.pt] [--arm OFF|ON]...
  env: VESSEL_CKPT_DIR·VESSEL_DYN_PROFILE·VESSEL_OBSTACLES·VESSEL_COMM_EXT 등은 호출자(배치 스크립트)에서 물려받는다.
       VESSEL_ROLE_PROMISE_PEN 은 지운다(옛 체크포인트는 그 키가 없음 = 0).
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
NOISE = ('경과', 'ETA', 'dec/s')


def run_eval(root, ckpt, arm, env, extra):
    cmd = [sys.executable, os.path.join(root, 'eval', 'eval_ckpt.py'), '--ckpt', ckpt, '--arm', arm,
           '--envs', '16', '--eval_decisions', '400', '--burnin', '200', '--seed', '999'] + extra
    p = subprocess.run(cmd, env=env, cwd=root, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-3000:])
        raise SystemExit(f'[evaldiff] {root} {ckpt} rc={p.returncode}')
    return [l for l in p.stdout.splitlines() if not any(n in l for n in NOISE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--arm', action='append', default=None, help='ckpt 마다 하나(기본 OFF)')
    ap.add_argument('--base', default='ae58b93')
    a = ap.parse_args()
    arms = a.arm or ['OFF'] * len(a.ckpt)
    assert len(arms) == len(a.ckpt), '--arm 개수 = --ckpt 개수'
    env = {k: v for k, v in os.environ.items() if k != 'VESSEL_ROLE_PROMISE_PEN'}
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    tmp = tempfile.mkdtemp(prefix='evaldiff_')
    old = os.path.join(tmp, 'old')
    subprocess.run(['git', '-C', REPO, 'worktree', 'add', '--detach', old, a.base], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = []
    try:
        for ck, arm in zip(a.ckpt, arms):
            lo = run_eval(os.path.join(old, 'Python'), ck, arm, env, [])
            ln = run_eval(PYROOT, ck, arm, env, [])
            rp = [l for l in ln if '[role-promise]' in l]
            rest = [l for l in ln if '[role-promise]' not in l]
            same = rest == lo and len(rp) == 1 and ln[-1] == rp[0]
            res.append(same)
            print(f"  {'PASS' if same else '★FAIL'}  {ck} ({arm}): 기존 줄 {len(lo)}개 동일={rest == lo} · 새 줄 1개·맨 끝={len(rp) == 1 and ln[-1] == rp[0]}")
            if rp:
                print('        ' + rp[0].strip())
            if rest != lo:
                import difflib
                for d in list(difflib.unified_diff(lo, rest, 'old', 'new', lineterm=''))[:20]:
                    print('        ' + d)
    finally:
        subprocess.run(['git', '-C', REPO, 'worktree', 'remove', '--force', old], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"VERDICT: {'ALL PASS' if all(res) else 'FAIL ' + str(res.count(False))}")
    return 0 if all(res) else 1


if __name__ == '__main__':
    sys.exit(main())
