"""check_branch.py — ON/OFF 분기 규약 검사 (2026-09-15).

규약 (사용자 지시, 예외 없음)
  ON/OFF 는 통신 켜는 지점(9,043,968 결정)까지 *같은 체크포인트 파일*(trunk)을 쓴다.
  OFF 로 거기까지 한 번 학습한 trunk 에서 OFF·ON(·RANDOM) 갈래를 --resume 으로 뻗는다.
  왜: 09-10 배치는 OFF·ON 을 같은 시드로 따로 처음부터 돌렸는데 결정론 설정이 없어 2번째 update(131,072)
  부터 갈라졌고, 9M 체크포인트가 이미 다른 모델이었다(s43 도착 89.1% vs 98.2%, 충돌 7.9% vs 0.8%).

검사 (하나라도 어기면 exit 1, 통과하면 'ALL PASS')
  1. 모든 체크포인트 스냅샷에 branch_from_sha256 · branch_at 이 있다 (= trunk 에서 분기한 런)
  2. 같은 trunk(SHA256) 묶음 안에서 seed · msg_dim · branch_at 이 같다
  3. 통신 팔(ON/RANDOM/ORACLE)이 있는 묶음에는 같은 trunk 의 OFF 갈래가 있다
  4. --trunk_dir 에 trunk 파일이 있으면 SHA256 을 다시 계산해 기록과 대조한다
  5. --csv_dir 에 trunk 곡선 CSV 가 있으면 각 갈래 CSV 의 step<=branch_at 행이 trunk CSV 와 글자까지 같다

쓰는 법
  python verify/check_branch.py [--trunk_dir CK] [--csv_dir OUT] a.pt b.pt ...
  run_repro.sh 가 train·random·smoke 끝과 eval 시작에 자동으로 부른다.

주의: 출력은 ASCII 대시만 쓴다 (U+2014 는 Windows cp949 리다이렉트에서 UnicodeEncodeError).
"""
import argparse
import hashlib
import os
import sys
from collections import defaultdict

import torch

COMM_ARMS = ('ON', 'RANDOM', 'ORACLE')


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_meta(path):
    ck = torch.load(path, map_location='cpu')
    snap = ck.get('cfg_snapshot') or {}
    return {
        'path': path, 'name': os.path.basename(path),
        'arm': snap.get('arm', ck.get('arm')), 'seed': snap.get('seed', ck.get('seed')),
        'msg_dim': snap.get('msg_dim'), 'steps': ck.get('steps'),
        'trunk': snap.get('branch_from'), 'sha': snap.get('branch_from_sha256'), 'at': snap.get('branch_at'),
        'dyn': str(snap.get('dyn_profile') or 'agile'), 'obst': str(snap.get('obstacles') or 'grid3x3'),
    }


def csv_rows_upto(path, at):
    """헤더를 뺀 곡선 CSV 행 중 step <= at 인 것 (문자열 그대로)."""
    rows = []
    with open(path, encoding='utf-8') as f:
        for ln in f.read().splitlines()[1:]:
            if ln and int(ln.split(',', 1)[0]) <= at:
                rows.append(ln)
    return rows


def main():
    ap = argparse.ArgumentParser(description='ON/OFF 분기 규약 검사')
    ap.add_argument('ckpts', nargs='+', help='검사할 갈래 체크포인트들')
    ap.add_argument('--trunk_dir', default=None, help='trunk 파일이 있는 곳 (SHA256 재계산)')
    ap.add_argument('--csv_dir', default=None, help='곡선 CSV 가 있는 곳 (0~branch_at 구간 대조)')
    a = ap.parse_args()

    fails, notes, metas = [], [], []
    for p in a.ckpts:
        if not os.path.exists(p):
            fails.append(f'{p}: 파일 없음')
            continue
        m = load_meta(p)
        metas.append(m)
        if not m['sha'] or m['at'] is None:
            fails.append(f"{m['name']}: 분기 기록 없음(branch_from_sha256/branch_at) - trunk 에서 분기하지 않은 런")

    print(f"{'체크포인트':<28} {'arm':<7} {'seed':>5} {'dim':>4} {'steps':>10} {'branch_at':>10}  trunk  dyn/obst")
    for m in metas:
        sha = (m['sha'] or '-')[:12]
        print(f"{m['name']:<28} {str(m['arm']):<7} {str(m['seed']):>5} {str(m['msg_dim']):>4} "
              f"{str(m['steps']):>10} {str(m['at']):>10}  {m['trunk'] or '-'} ({sha})"
              f"  {m['dyn']}/{m['obst']}")

    groups = defaultdict(list)
    for m in metas:
        if m['sha']:
            groups[m['sha']].append(m)

    for sha, ms in groups.items():
        tag = f"trunk {ms[0]['trunk']} ({sha[:12]})"
        for key in ('seed', 'msg_dim', 'at', 'dyn', 'obst'):
            vals = sorted({str(m[key]) for m in ms})
            if len(vals) > 1:
                fails.append(f'{tag}: {key} 불일치 {vals}')
        arms = {m['arm'] for m in ms}
        comm = sorted(arms & set(COMM_ARMS))
        if comm and 'OFF' not in arms:
            fails.append(f'{tag}: 통신 팔 {comm} 에 같은 trunk 의 OFF 갈래가 없음')
        if not comm:
            notes.append(f'{tag}: 통신 팔 없음(OFF 만) - 비교 대상 없음')
        if a.trunk_dir:
            tp = os.path.join(a.trunk_dir, ms[0]['trunk'] or '')
            if ms[0]['trunk'] and os.path.isfile(tp):
                if file_sha256(tp) != sha:
                    fails.append(f'{tag}: trunk 파일 SHA256 이 기록과 다름 (분기 뒤 덮어써짐?)')
            else:
                notes.append(f'{tag}: trunk 파일 없음 - SHA 재계산 생략(갈래끼리 SHA 일치로만 판정)')
        if a.csv_dir and ms[0]['trunk']:
            at = int(ms[0]['at'])
            tc = os.path.join(a.csv_dir, os.path.splitext(ms[0]['trunk'])[0] + '.csv')
            if os.path.isfile(tc):
                ref = csv_rows_upto(tc, at)
                for m in ms:
                    bc = os.path.join(a.csv_dir, os.path.splitext(m['name'])[0] + '.csv')
                    if not os.path.isfile(bc):
                        notes.append(f"{m['name']}: 곡선 CSV 없음 - 0~{at} 곡선 대조 생략")
                    elif csv_rows_upto(bc, at) != ref:
                        fails.append(f"{m['name']}: 0~{at} 곡선이 trunk CSV 와 다름")
            else:
                notes.append(f'{tag}: trunk 곡선 CSV 없음 - 0~{at} 곡선 대조 생략')

    for n in notes:
        print('  note:', n)
    for f in fails:
        print('  FAIL:', f)
    if fails or not metas:
        print('BRANCH CHECK: FAIL')
        sys.exit(1)
    print(f'BRANCH CHECK: ALL PASS ({len(metas)}개 체크포인트, trunk {len(groups)}개)')


if __name__ == '__main__':
    main()
