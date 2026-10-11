# eval_any.py — 진단 전용(학습·저장 없음): eval_scripted.py --policy learned 를 그대로 돌리고, 보상을 항목별로 나눠 셈.
#   mode=sample : 주 평가와 같은 샘플 행동.  mode=mean : ctr_actor 첫 반환값만 tanh(mean) 으로 바꿈(잡음 없음, report/diag/eval_mean.py 와 같은 방식).
#   보상 분해(vessel_gym.step 1836–1850행과 같은 식): R = 나머지(shaping) + 150·도착 − 300·충돌 − 50·시간초과 − 20·판정 실패 수.
#     판정 실패 수 = env._rp.events 의 chg_i·chg_j·crash_i·crash_j (RolePromiseTrackerV2.update 733–737행과 같은 합). 읽기만 함.
#   사용: python eval_any.py <scripted 폴더> <sample|mean> -- <eval_scripted.py 인자>
import os, sys
sdrop, mode, rest = sys.argv[1], sys.argv[2], sys.argv[4:]
py_root = rest[rest.index('--py_root') + 1]
burnin = int(rest[rest.index('--burnin') + 1])
sys.path.insert(0, py_root); sys.path.insert(0, os.path.join(py_root, 'eval')); sys.path.insert(0, sdrop)
import torch
import vessel_gym as vg
import ckpt_io

A = {k: 0.0 for k in ('n', 'R', 'goal', 'coll', 'to', 'nf', 'crash', 'f1', 'f2', 'f4', 'f5', 'f6', 'turn', 'spd', 'a0abs', 'near', 'da0')}
ST = {'t': 0, 'prev': None}
_make = ckpt_io.make_env_from_snapshot


def make_env(*a, **k):
    env = _make(*a, **k)
    _step = env.step

    def step(actions):
        out = _step(actions)
        obs, r, done, oc = out
        if ST['t'] >= burnin:
            A['n'] += float(r.numel()); A['R'] += float(r.sum())
            A['goal'] += float((oc == vg.OUT_GOAL).sum())
            A['coll'] += float(((oc == vg.OUT_COLLISION_VESSEL) | (oc == vg.OUT_COLLISION_OBSTACLE)).sum())
            A['to'] += float((oc == vg.OUT_TIMEOUT).sum())
            rp = getattr(env, '_rp', None)
            ev = rp.events if rp is not None else None
            if ev is not None and 'chg_i' in ev:
                ci, cj, xi, xj = ev['chg_i'], ev['chg_j'], ev['crash_i'], ev['crash_j']
                A['nf'] += float(ci.sum() + cj.sum() + xi.sum() + xj.sum())
                A['crash'] += float(xi.sum() + xj.sum())
                ft = rp.fail_type
                for t_, key in ((1, 'f1'), (2, 'f2'), (4, 'f4'), (5, 'f5'), (6, 'f6')):
                    A[key] += float((ci & (ft == t_)).sum() + (cj & (ft == t_)).sum())
            a0 = actions[..., 0].clamp(-1, 1)
            A['a0abs'] += float(a0.abs().sum())
            if ST['prev'] is not None:
                A['da0'] += float((a0 - ST['prev']).abs().sum())
            A['spd'] += float((env.speed / env.max_speed.clamp(min=1e-6)).sum())
            d = env._last_pw['dist']
            eye = torch.eye(env.N, dtype=torch.bool, device=d.device).unsqueeze(0)
            A['near'] += float(((d <= 56.0) & ~eye).any(-1).sum())
        ST['prev'] = actions[..., 0].clamp(-1, 1).clone()
        ST['t'] += 1
        return out
    env.step = step
    return env


ckpt_io.make_env_from_snapshot = make_env
if mode == 'mean':
    _restore = ckpt_io.restore_policy

    def restore(*a, **k):
        r = _restore(*a, **k)
        orig = r.policy.ctr_actor.forward

        def mean_forward(*fa, **fk):
            out = orig(*fa, **fk)
            return (torch.tanh(out[2]),) + tuple(out[1:])
        r.policy.ctr_actor.forward = mean_forward
        return r
    ckpt_io.restore_policy = restore
    print('[eval_any] mean action: tanh(mean), no sampling noise (diagnostic)', flush=True)
else:
    print('[eval_any] sampled action (same as main eval)', flush=True)
import eval_scripted
sys.argv = ['eval_scripted.py'] + rest
eval_scripted.main()

n = max(A['n'], 1.0)
PEN = float(vg.ROLE_PROMISE_PEN)
arr, col, to, jud = vg.ARRIVAL_REWARD * A['goal'] / n, float(vg.COLLISION_PENALTY) * A['coll'] / n, vg.TIMEOUT_PENALTY * A['to'] / n, -PEN * A['nf'] / n
R = A['R'] / n
k = 1000.0     # 1000 결정당으로 표시
print(f"[rdecomp] 1000결정당 보상 R={k * R:+.1f} = 도착 {k * arr:+.1f} + 충돌 {k * col:+.1f} + 시간초과 {k * to:+.1f} + 판정벌점 {k * jud:+.1f} + 나머지(shaping) {k * (R - arr - col - to - jud):+.1f}"
      f" | 배-결정 n={n:.0f} PEN={PEN:g}", flush=True)
print(f"[rdecomp] 1000결정당 건수: 도착 {k * A['goal'] / n:.3f} · 충돌 {k * A['coll'] / n:.3f} · 시간초과 {k * A['to'] / n:.3f} · 판정 실패 {k * A['nf'] / n:.3f}"
      f" (좌현 F1 {k * A['f1'] / n:.3f} · 유지선 F2 {k * A['f2'] / n:.3f} · 24m F4 {k * A['f4'] / n:.3f} · 충돌 F5 {k * A['f5'] / n:.3f} · 우현없음 F6 {k * A['f6'] / n:.3f} · 사고 종료 {k * A['crash'] / n:.3f})", flush=True)
print(f"[rdecomp] 행동: |침로 명령| 평균 {A['a0abs'] / n:.4f} (×120° = {120 * A['a0abs'] / n:.1f}°) · 결정 간 침로 명령 변화 평균 {A['da0'] / n:.4f} (×120° = {120 * A['da0'] / n:.1f}°)"
      f" · 속력비 평균 {A['spd'] / n:.3f} · 56 m 안에 배가 있는 결정 {100 * A['near'] / n:.1f}%", flush=True)
