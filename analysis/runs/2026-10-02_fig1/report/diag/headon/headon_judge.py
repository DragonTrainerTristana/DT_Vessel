# headon_judge.py — 진단 전용(학습·저장 없음): 학습 보상에 쓰인 역할 약속 판정기 v2 가 정면 조우 쌍을 어떻게 처리했는지 집계.
#   env._rp.events(매 결정에 끝난 쌍의 판정)를 읽기만 함. 판정기·평가 코드는 건드리지 않음.
#   사용: python headon_judge.py <scripted 폴더> -- <eval_scripted.py 인자>
import os, sys
sdrop, rest = sys.argv[1], sys.argv[3:]
py_root = rest[rest.index('--py_root') + 1]
burnin = int(rest[rest.index('--burnin') + 1])
sys.path.insert(0, py_root); sys.path.insert(0, os.path.join(py_root, 'eval')); sys.path.insert(0, sdrop)
import torch
import vessel_gym as vg
import ckpt_io

REC = {r: [] for r in ('headon', 'cross')}
CNT = {'start_ho': 0.0, 'disc_ho': 0.0}
_make = ckpt_io.make_env_from_snapshot


def make_env(*a, **k):
    env = _make(*a, **k)
    _step = env.step

    def step(actions):
        out = _step(actions)
        rp = getattr(env, '_rp', None)
        ev = rp.events if rp is not None else None
        if ev is not None and 'fail_type' in ev:
            ok = ev['t_start'] >= burnin
            ri, rj = ev['role_i'], ev['role_j']
            ho = (ri == vg.SIT_HEADON) | (rj == vg.SIT_HEADON)
            cr = ((ri == vg.SIT_GIVEWAY) & (rj == vg.SIT_STANDON)) | ((ri == vg.SIT_STANDON) & (rj == vg.SIT_GIVEWAY))
            CNT['start_ho'] += float((ev['start'] & ((rp.role_i == vg.SIT_HEADON) | (rp.role_j == vg.SIT_HEADON))).sum())
            CNT['disc_ho'] += float((ev['discard'] & ho & ok).sum())
            for nm, sel in (('headon', ho), ('cross', cr)):
                m = ev['judged'] & sel & ok
                if bool(m.any()):
                    f = lambda x: x[m].float()
                    REC[nm].append(torch.stack([f(ev['success']), f(ev['resolved']), f(ev['fail_type']), f(ev['late']), f(ev['coll']),
                                                f(ev['dmax_i']), f(ev['dmin_i']), f(ev['dmax_j']), f(ev['dmin_j']), f(ev['mind']), f(ev['n']),
                                                f(ri), f(rj)], 1).cpu())
        return out
    env.step = step
    return env


ckpt_io.make_env_from_snapshot = make_env
import eval_scripted
sys.argv = ['eval_scripted.py'] + rest
eval_scripted.main()

pc = lambda m: 100.0 * float(m.float().mean()) if m.numel() else float('nan')
for nm, title in (('headon', '정면 조우 쌍'), ('cross', '교차 쌍(양보–유지)')):
    X = torch.cat(REC[nm]) if REC[nm] else torch.zeros(0, 13)
    suc, res, ft, late, coll, dxi, dni, dxj, dnj, mind, n, ri, rj = [X[:, i] for i in range(13)]
    N = X.shape[0]
    if N == 0:
        print(f"[judge-diag] {title}: 판정된 쌍 없음"); continue
    suc, res, late = suc > 0, res > 0, late > 0
    print(f"[judge-diag] {title}: 판정 n={N} | 성공(벌점 없음) {pc(suc):.1f}% · 실패(벌점 −20) {pc(~suc):.1f}% | 길이 중앙 {float(n.median()) * 0.4:.0f}s · 최소거리 중앙 {float(mind.median()):.0f}m")
    print(f"[judge-diag]   끝난 방식: 해소(dcpa≥30 m 가 5결정 연속 → 성공 종료, 우현 변침 여부 안 봄) {pc(res):.1f}% · 늦은 시작(침로 기준 면제) {pc(late):.1f}%")
    print(f"[judge-diag]   실패 유형(전체 쌍 대비): 좌현 변침 F1 {pc(ft == 1):.1f}% · 유지선 변침 F2 {pc(ft == 2):.1f}% · 24 m 안 F4 {pc(ft == 4):.1f}% · 충돌 F5 {pc(ft == 5):.1f}% · 우현 변침 없음 F6 {pc(ft == 6):.1f}%")
    if nm == 'headon':
        both10 = (dxi >= 10) & (dxj >= 10); none10 = (dxi < 10) & (dxj < 10)
        print(f"[judge-diag]   주 상대였던 결정 기준 우현 최대 변침: 둘 다 ≥10° {pc(both10):.1f}% · 한 척만 {pc(~both10 & ~none10):.1f}% · 둘 다 <10° {pc(none10):.1f}%")
        print(f"[judge-diag]   성공 처리된 쌍(n={int(suc.sum())}) 안에서: 둘 다 우현 ≥10° {pc(both10[suc]):.1f}% · 둘 다 <10° {pc(none10[suc]):.1f}% · 해소로 끝남 {pc(res[suc]):.1f}%")
        print(f"[judge-diag]   해소로 끝난 쌍(n={int(res.sum())}) 안에서: 둘 다 <10° {pc(none10[res]):.1f}% · 둘 다 <2° {pc(((dxi < 2) & (dxj < 2))[res]):.1f}% · 길이 중앙 {float(n[res].median()) * 0.4 if bool(res.any()) else float('nan'):.0f}s")
print(f"[judge-diag] 정면 조우 쌍 시작 {CNT['start_ho']:.0f}(burn-in 포함) · 버려짐(판정 안 함) {CNT['disc_ho']:.0f}")
