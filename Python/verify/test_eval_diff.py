"""test_eval_diff.py — G8b: eval_ckpt 출력 바이트 차분 (2026-09-29, 스펙 §5). Windows(체크포인트가 있는 곳) 전용.

옛 커밋(기본 9cc8f6d — 2026-09-30 보상 v3 직전)의 eval_ckpt.py 와 현재 eval_ckpt.py 로 같은 체크포인트를 작은 창에서 평가해 stdout 을 비교한다.
허용 차이는 ① 진행 표시 줄(경과·ETA·dec/s — 벽시계) ② 새 줄 태그 NEW_TAGS('[fuel-diag]'·'[role-promise]'·'[role-promise/v2]') 중
*옛 출력에 없는* 것뿐 — 옛 출력에 이미 있는 태그 줄(9cc8f6d 는 '[role-promise]')은 글자 하나까지 같아야 한다(옛 판정기 줄의 연속성).
나머지 줄도 글자 하나까지 같아야 한다(= 판정기·진단 줄을 더해도 기존 지표·난수·상태가 안 바뀜). 새 줄은 맨 끝에 NEW_TAGS 순서로 온다.

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
# 허용되는 새 줄 태그 = eval 출력 맨 끝 3줄의 순서. '[role-promise]' 는 '[role-promise/v2] …' 줄에 부분 문자열로 들어가지 않는다(']' 위치).
NEW_TAGS = ('[fuel-diag]', '[role-promise]', '[role-promise/v2]')


def _tag(line):
    """Return the NEW_TAGS entry contained in line, or None."""
    for t in NEW_TAGS:
        if t in line:
            return t
    return None


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
    ap.add_argument('--base', default='9cc8f6d')
    a = ap.parse_args()
    arms = a.arm or ['OFF'] * len(a.ckpt)
    assert len(arms) == len(a.ckpt), '--arm 개수 = --ckpt 개수'
    env = {k: v for k, v in os.environ.items() if k != 'VESSEL_ROLE_PROMISE_PEN'}
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    # CPU 고정: GPU 는 실행 간 비결정 연산이 있을 수 있어 같은 코드도 바이트가 달라질 수 있음(거짓 FAIL 방지). 16 env·600 결정이라 CPU 로 충분
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    env['OMP_NUM_THREADS'] = '1'
    tmp = tempfile.mkdtemp(prefix='evaldiff_')
    old = os.path.join(tmp, 'old')
    subprocess.run(['git', '-C', REPO, 'worktree', 'add', '--detach', old, a.base], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = []
    try:
        for ck, arm in zip(a.ckpt, arms):
            lo = run_eval(os.path.join(old, 'Python'), ck, arm, env, [])
            ln = run_eval(PYROOT, ck, arm, env, [])
            # 옛 출력에 없는 태그만 '새 줄'. 옛 출력에 있는 태그 줄은 rest 에 남아 글자 비교를 받는다(옛 판정기 줄 연속성)
            old_tags = {t for t in (_tag(l) for l in lo) if t}
            new_only = [t for t in NEW_TAGS if t not in old_tags]
            added = [l for l in ln if _tag(l) in new_only]
            rest = [l for l in ln if _tag(l) not in new_only]
            tail_ok = ([_tag(l) for l in ln[-len(NEW_TAGS):]] == list(NEW_TAGS)     # 맨 끝 3줄 = NEW_TAGS 순서
                       and len(added) == len(new_only)                             # 새 태그마다 정확히 1줄
                       and all(sum(1 for l in ln if _tag(l) == t) == 1 for t in NEW_TAGS))
            same = rest == lo and tail_ok
            res.append(same)
            print(f"  {'PASS' if same else '★FAIL'}  {ck} ({arm}): 기존 줄 {len(lo)}개 동일={rest == lo} · "
                  f"새 줄 {len(added)}개({','.join(new_only) or '없음'}) 맨 끝 {'/'.join(NEW_TAGS)} 순서={tail_ok}")
            for l in added:
                print('        ' + l.strip())
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
