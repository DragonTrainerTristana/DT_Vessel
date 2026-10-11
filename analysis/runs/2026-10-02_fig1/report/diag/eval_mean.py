# eval_mean.py — 진단 전용(판정 아님): 주 평가(eval/eval_ckpt.py)를 그대로 돌리되 학생이 '평균 행동'(tanh(mean), 샘플 잡음 없음)으로 배를 몲.
#   eval_ckpt 는 ctr_actor 의 샘플 행동을 실행함. 여기서는 ckpt_io.restore_policy 를 감싸 ctr_actor.forward 의 첫 반환값만 tanh(mean) 으로 바꿈.
#   저장소 파일 수정 없음. 사용: python eval_mean.py <Python 폴더> -- <eval_ckpt.py 인자>
import os, sys
py_root, rest = sys.argv[1], sys.argv[3:] if sys.argv[2] == '--' else sys.argv[2:]
sys.path.insert(0, py_root); sys.path.insert(0, os.path.join(py_root, 'eval'))
import torch
import ckpt_io
import eval_ckpt

_restore = ckpt_io.restore_policy
calls = {'n': 0}


def restore(*a, **k):
    r = _restore(*a, **k)
    orig = r.policy.ctr_actor.forward

    def mean_forward(*fa, **fk):
        out = orig(*fa, **fk)
        calls['n'] += 1
        return (torch.tanh(out[2]),) + tuple(out[1:])
    r.policy.ctr_actor.forward = mean_forward
    return r


ckpt_io.restore_policy = restore
print('[eval_mean] student drives with tanh(mean) — no sampling noise (diagnostic, not the P0 protocol)', flush=True)
sys.argv = ['eval_ckpt.py'] + rest
eval_ckpt.main()
print(f"[eval_mean] mean-action actor calls={calls['n']}", flush=True)
