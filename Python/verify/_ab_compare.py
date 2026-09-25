"""_ab_compare.py — 두(세) 학습 산출물의 비트 비교 (2026-09-26, GPU A/B 용).

python verify/_ab_compare.py <dirA> <dirB> [<dirC> ...]
  각 디렉터리에 m.pt (학습기 --save) 와 m_curve.csv (--csv) 가 있어야 한다. 첫 디렉터리를 기준으로
  model_state_dict / optimizer_state_dict / value_norm / steps 텐서별 torch.equal, CSV 바이트 비교.
  전부 같으면 'ALL EQUAL' 과 exit 0, 아니면 다른 키를 찍고 exit 1.
"""
import os
import sys

import torch


def _flat(prefix, obj, out):
    if torch.is_tensor(obj):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flat(f'{prefix}/{k}', v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flat(f'{prefix}/{i}', v, out)
    else:
        out[prefix] = obj


def load(d):
    ck = torch.load(os.path.join(d, 'm.pt'), map_location='cpu', weights_only=False)
    flat = {}
    for key in ('model_state_dict', 'optimizer_state_dict', 'value_norm', 'steps'):
        _flat(key, ck.get(key), flat)
    with open(os.path.join(d, 'm_curve.csv'), 'rb') as f:
        csv = f.read()
    return flat, csv


def main():
    dirs = sys.argv[1:]
    base, base_csv = load(dirs[0])
    bad = 0
    for d in dirs[1:]:
        other, other_csv = load(d)
        keys = sorted(set(base) | set(other))
        diff = []
        for k in keys:
            a, b = base.get(k), other.get(k)
            if torch.is_tensor(a) and torch.is_tensor(b):
                same = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
            else:
                same = (a == b)
            if not same:
                diff.append(k)
        csv_same = base_csv == other_csv
        print(f'{dirs[0]} vs {d}: tensors {len(keys)} differing {len(diff)}; curve_csv {"same" if csv_same else "DIFFERENT"}')
        for k in diff[:20]:
            print('   ', k)
        if diff or not csv_same:
            bad = 1
    print('ALL EQUAL' if not bad else 'NOT EQUAL')
    sys.exit(bad)


if __name__ == '__main__':
    main()
