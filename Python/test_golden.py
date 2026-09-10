"""test_golden.py — 학습기 '기본값 비트동일' 골든 테스트 (2026-09-10).

왜 있나
  이 저장소는 모든 변경에 "기본값은 비트동일" 을 요구하는데, 그걸 확인하는 건 주석 50여 곳뿐이었다
  (allclose·골든 비교 0건). 리팩토링(config 통합·죽은 코드 제거) 중 조용히 결과가 바뀌면 잡을 수 없다.
  → vessel_gym_train 을 고정 시드·CPU 로 2 update 돌리고 state_dict 텐서별 SHA256 + 학습곡선 CSV +
    cfg_snapshot + Adam 상태를 골든으로 박아 두고, 이후엔 `--check` 로 바이트 단위 비교한다.

쓰는 법
  python test_golden.py --regen        # 골든 생성 (코드 변경 *전* 에만. 명시 승인 필요)
  python test_golden.py --check        # 현재 코드가 골든과 비트동일인지 (리팩토링 각 단계 후)
  python test_golden.py --check --case default_ON   # 한 케이스만
  pytest test_golden.py                 # 위 --check 를 pytest 로

케이스
  default_ON / default_OFF   : env 아무것도 안 줌 = config.py 기본값
  batch_2026_09_04_ON        : run_repro.sh common_env 와 동일 (실제 12런 배치 설정)

주의
  - 학습기 코드는 손대지 않는다. subprocess 로 있는 그대로 돌린다.
  - OMP/MKL 스레드 1개로 고정 (부동소수 합산 순서 비결정성 차단).
  - 골든 파일 Python/golden/*.json 은 git 추적. --regen 은 diff 로 드러난다.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings('ignore')
import torch  # noqa: E402  (torch/numpy 경고는 위에서 차단)

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, 'golden')
TRAIN = os.path.join(HERE, 'vessel_gym_train.py')
STAMP = '2026-09-10'

# run_repro.sh common_env() 와 동일 — 여기 바꾸면 그쪽도 바꿀 것
BATCH_ENV = {
    'VESSEL_STATE_RECON_COEF': '1.0', 'VESSEL_CENTRAL_CRITIC': '1', 'VESSEL_USE_ATTENTION': '1',
    'VESSEL_THREAT_COEF': '0', 'VESSEL_GOAL_COMM_COEF': '0', 'VESSEL_INTENT_COEF': '0',
    'VESSEL_ROLE_COMM_COEF': '0', 'VESSEL_COMM_CONSUMER_COEF': '0',
    'VESSEL_USE_MOE': '1', 'VESSEL_MOE_SHARED': '1', 'VESSEL_MOE_WIDTH': '1.0',
    'VESSEL_POS_GROUND': '1', 'VESSEL_MSG_LN': '1', 'VESSEL_COMM_RANGE': '200',
    'VESSEL_RADAR_RANGE': '56', 'VESSEL_COLREGS_MODE': 'unity', 'VESSEL_SIM_COLREGS_COEF': '0.45',
    'VESSEL_INTENT_K': '3',
}

CASES = {
    'default_ON':          dict(env={}, arm='ON'),
    'default_OFF':         dict(env={}, arm='OFF'),
    'batch_2026_09_04_ON': dict(env=BATCH_ENV, arm='ON'),
    # ★2026-09-10 공유 인코더 — 새 구조. 골든은 도입 시점 코드로 생성.
    'batch_shared_all_ON':   dict(env={**BATCH_ENV, 'VESSEL_SHARED_ENCODER': 'all'}, arm='ON'),
    'batch_shared_actor_ON': dict(env={**BATCH_ENV, 'VESSEL_SHARED_ENCODER': 'actor'}, arm='ON'),
}

# 작게: E=8 N=16 rollout=64 → update 당 8,192 결정(전 에이전트 합). --steps 는 그 합 기준(vessel_gym_train.py:736
# `while total_decisions < args.steps`) 이라 16384 = 정확히 2 update. CPU 2~3분.
TRAIN_ARGS = ['--steps', '16384', '--envs', '8', '--vessels', '16', '--seed', '0', '--rollout', '64']


_FMT = {'torch.float32': 'f', 'torch.float64': 'd', 'torch.int64': 'q', 'torch.int32': 'i',
        'torch.int16': 'h', 'torch.int8': 'b', 'torch.uint8': 'B', 'torch.bool': 'B', 'torch.float16': 'e'}


def _sha(t):
    """numpy 없이 텐서 바이트를 해시 (이 맥은 torch↔numpy 2.x 불일치로 .numpy() 가 안 됨)."""
    import array
    t = t.detach().cpu().contiguous().flatten()
    code = _FMT[str(t.dtype)]
    vals = t.to(torch.uint8).tolist() if t.dtype == torch.bool else t.tolist()
    return hashlib.sha256(array.array(code, vals).tobytes()).hexdigest()


def run_case(name, spec):
    """학습기를 돌리고 골든 레코드(dict)를 만든다."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('VESSEL_')}   # 바깥 VESSEL_* 차단
    env.update({'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'CUDA_VISIBLE_DEVICES': ''})
    env.update(spec['env'])
    with tempfile.TemporaryDirectory() as td:
        save = os.path.join(td, 'g.pt')
        csv = os.path.join(td, 'g_curve.csv')
        cmd = [sys.executable, '-u', TRAIN, '--arm', spec['arm'], '--save', save, '--csv', csv] + TRAIN_ARGS
        r = subprocess.run(cmd, env=env, cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"[{name}] 학습기 실패 rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        ck = torch.load(save, map_location='cpu')
        rec = {
            'stamp': STAMP, 'case': name, 'arm': spec['arm'], 'env': spec['env'], 'train_args': TRAIN_ARGS,
            'state_dict': {k: _sha(v) for k, v in ck['model_state_dict'].items()},
            'cfg_snapshot': ck.get('cfg_snapshot'),
            'value_norm': {k: (float(v) if not hasattr(v, 'tolist') else v.tolist())
                           for k, v in (ck.get('value_norm') or {}).items()},
            'steps': int(ck.get('steps', -1)),
            'curve_csv': open(csv, encoding='utf-8').read() if os.path.exists(csv) else None,
        }
        opt = ck.get('optimizer_state_dict')
        if opt and 'state' in opt:
            rec['adam'] = {str(i): {k: _sha(v) for k, v in st.items() if hasattr(v, 'detach')}
                           for i, st in opt['state'].items()}
        # 보조 CSV(state_recon 켠 케이스)도 있으면 포함
        aux = os.path.splitext(csv)[0] + '_aux.csv'
        rec['aux_csv'] = open(aux, encoding='utf-8').read() if os.path.exists(aux) else None
    return rec


def golden_path(name):
    return os.path.join(GOLDEN_DIR, f'{STAMP}_{name}.json')


def diff(gold, cur):
    """차이 목록. 비면 비트동일."""
    out = []
    for sect in ('state_dict', 'adam', 'value_norm'):
        g, c = gold.get(sect) or {}, cur.get(sect) or {}
        for k in sorted(set(g) | set(c)):
            if g.get(k) != c.get(k):
                out.append(f'{sect}.{k}')
    # cfg_snapshot: 골든에 있는 키만 비교. 키 *추가* 는 허용(구 로더가 무시), 삭제·값변경은 FAIL.
    gs, cs = gold.get('cfg_snapshot') or {}, cur.get('cfg_snapshot') or {}
    for k in sorted(gs):
        if k not in cs or gs[k] != cs[k]:
            out.append(f'cfg_snapshot.{k}')
    for sect in ('steps', 'curve_csv', 'aux_csv'):
        if gold.get(sect) != cur.get(sect):
            out.append(sect)
    return out


def check(names):
    ok_all = True
    for name in names:
        p = golden_path(name)
        if not os.path.exists(p):
            print(f'  ★FAIL  {name:24s} 골든 없음 → --regen 먼저'); ok_all = False; continue
        gold = json.load(open(p, encoding='utf-8'))
        cur = run_case(name, CASES[name])
        d = diff(gold, cur)
        n_sd = len(cur['state_dict'])
        if d:
            ok_all = False
            print(f'  ★FAIL  {name:24s} 차이 {len(d)}건 (state_dict {n_sd}텐서)')
            for x in d[:20]:
                print(f'           {x}')
            if len(d) > 20:
                print(f'           ... +{len(d)-20}')
        else:
            print(f'  PASS   {name:24s} state_dict {n_sd}텐서 · adam · value_norm · curve 전부 비트동일')
    return ok_all


def regen(names):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for name in names:
        rec = run_case(name, CASES[name])
        p = golden_path(name)
        json.dump(rec, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'  생성  {name:24s} → {os.path.relpath(p, HERE)}  (state_dict {len(rec["state_dict"])}텐서)')


# ── pytest 진입점 ──
def test_golden_default_on():
    assert check(['default_ON'])


def test_golden_default_off():
    assert check(['default_OFF'])


def test_golden_batch():
    assert check(['batch_2026_09_04_ON'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--regen', action='store_true', help='골든 생성 (코드 변경 전에만)')
    ap.add_argument('--check', action='store_true', help='현재 코드 vs 골든 비트동일 검사')
    ap.add_argument('--case', default=None, choices=list(CASES), help='한 케이스만')
    a = ap.parse_args()
    names = [a.case] if a.case else list(CASES)
    print('=' * 78)
    print(f'골든 비트동일 테스트  {STAMP}  ({", ".join(names)})')
    print('=' * 78)
    if a.regen:
        regen(names)
        sys.exit(0)
    ok = check(names)
    print('=' * 78)
    print(f"VERDICT: {'ALL PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
