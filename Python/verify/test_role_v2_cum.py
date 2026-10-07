"""test_role_v2_cum.py — 판정기 v2 의 2026-10-07 토글 두 개 (스펙 2026-10-07-pure-rl-fig1-design.md).

  python verify/test_role_v2_cum.py        # ALL PASS 여야 함 (imo·none 고정, CPU, torch ≥ 2)

cum1 VESSEL_ROLE_V2_PRIMARY=cum: 3척(0 북행 · 1 정면 · 2 우현 교차). 순간 위험 최대는 도중에 2 로 바뀌지만 누적 주 상대는 1 에 머묾.
     self.cum[0,0,1] == 판정기 갱신 전 활성이던 결정의 pw['risk'][0,0,1] 합(손 계산)
cum2 같은 장면에서 role_declaration 이 판정기 주 상대의 고정 역할·선체좌표 위치를 선언
f6a  VESSEL_ROLE_V2_RES_F6=0: 정면, 0 번 직진 · 1 번만 우현 → 해소 성공, 벌점 0 (옛 동작)
f6b  VESSEL_ROLE_V2_RES_F6=1: 같은 기동 → 해소 종료에서 0 번만 F6 1회(fail_type 6), 판정 = 실패
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
sys.path.insert(0, PYROOT)
sys.path.insert(0, HERE)
for _k in [k for k in os.environ if k.startswith('VESSEL_')]:
    os.environ.pop(_k)
os.environ.update({'VESSEL_DYN_PROFILE': 'imo', 'VESSEL_OBSTACLES': 'none'})

import torch  # noqa: E402
import vessel_gym as vg  # noqa: E402
from test_role_promise import make_env, place, act, HEADON, PEN  # noqa: E402
from test_role_promise_v2 import run_v2, turn_then_hold  # noqa: E402

RES = []

# 0 북행, 1 정면(횡 20 m), 2 우현에서 교차(빠름) — 순간 위험 최대가 36결정쯤 1 → 2 로 바뀜(2026-10-07 측정)
THREE = [dict(pos=(0.0, -130.0), hdg=0.0, spd=1.5, maxs=1.5, goal=(0.0, 280.0)),
         dict(pos=(20.0, 130.0), hdg=180.0, spd=1.5, maxs=1.5, goal=(20.0, -280.0)),
         dict(pos=(280.0, 60.0), hdg=270.0, spd=2.2, maxs=2.2, goal=(-280.0, 60.0))]


def check(name, ok, info=''):
    RES.append(bool(ok))
    print(f"  {'PASS' if ok else '★FAIL'}  {name}  {info}")


class _knobs:
    """블록 안에서 판정기 v2 · PEN · 새 토글을 켬. 끝나면 원상복구."""
    def __init__(self, primary='risk', res_f6=0):
        self.new = ('v2', PEN, primary, res_f6)

    def __enter__(self):
        self.saved = (vg.ROLE_JUDGE, vg.ROLE_PROMISE_PEN, vg.ROLE_V2_PRIMARY, vg.ROLE_V2_RES_F6)
        vg.ROLE_JUDGE, vg.ROLE_PROMISE_PEN, vg.ROLE_V2_PRIMARY, vg.ROLE_V2_RES_F6 = self.new

    def __exit__(self, *a):
        vg.ROLE_JUDGE, vg.ROLE_PROMISE_PEN, vg.ROLE_V2_PRIMARY, vg.ROLE_V2_RES_F6 = self.saved


def cum_primary():
    with _knobs(primary='cum'):
        env = make_env(1, 3)
        place(env, THREE)
        inst, cum, manual = [], [], 0.0
        prev_act = None
        for t in range(120):
            env.step(act(env, [0.0, 0.0, 0.0]))
            rp, risk = env._rp, env._last_pw['risk'][0, 0]
            if prev_act is not None and prev_act:
                manual += float(risk[1])
            prev_act = bool((rp.active | rp.active.transpose(1, 2))[0, 0, 1])
            inst.append(int(risk.argmax()) if float(risk.max()) > 0 else -1)
            has, idx = rp.primary()
            cum.append(int(idx[0, 0]) if bool(has[0, 0]) else -1)
        flips = [t for t in range(len(inst)) if inst[t] == 2 and cum[t] == 1]
        check('cum1 순간 최대는 2 로 바뀌어도 누적 주 상대는 1', len(flips) >= 20 and all(c == 1 for c in cum[5:100]),
              f"순간≠누적 결정 {len(flips)} 첫={flips[:1]}")
        got = float(rp.cum[0, 0, 1])
        check('cum1 cum[0,0,1] == 활성 결정의 위험 합', abs(got - manual) < 1e-4 * max(1.0, manual), f"{got:.5f} vs {manual:.5f}")
        topi = torch.zeros(1, 3, 2, dtype=torch.long)
        role, pos = vg.role_declaration(env, topi, torch.ones(1, 3, 2, dtype=torch.bool), None)
        has, idx = rp.primary()
        exp_role = int(rp.role_full()[0, 0, int(idx[0, 0])])
        rel = env.pos[0, int(idx[0, 0])] - env.pos[0, 0]
        h = float(env.heading[0, 0]) * vg.DEG
        stb = float(rel[0]) * torch.cos(torch.tensor(h)) - float(rel[1]) * torch.sin(torch.tensor(h))
        check('cum2 선언 = 판정기 주 상대의 고정 역할·선체좌표 위치',
              int(role[0, 0]) == exp_role and exp_role > 0 and abs(float(pos[0, 0, 0]) - float(stb)) < 1e-3,
              f"role={int(role[0, 0])} 기대={exp_role} stb={float(pos[0, 0, 0]):.2f}/{float(stb):.2f}")


def res_f6():
    pol = turn_then_hold(0.0, 1.0, 40)        # 0 번 직진(양보 의무 안 지킴) · 1 번만 우현 전타
    with _knobs(res_f6=0):
        env, tot, nf, rec = run_v2(HEADON, pol, 260)
    check('f6a 토글 끔: 해소 성공 · 벌점 0', tot['resolved'] == 1 and tot['success'] == 1 and float(nf.sum()) == 0.0,
          f"{tot} charges={nf.tolist()}")
    with _knobs(res_f6=1):
        env, tot, nf, rec = run_v2(HEADON, pol, 260)
    ft = [int(r['ev']['fail_type'].max()) for r in rec if int(r['ev']['judged'].sum()) > 0]
    check('f6b 토글 켬: 해소 종료에서 0 번만 F6 1회 · 판정 실패', tot['resolved'] == 1 and tot['fail'] == 1
          and nf.tolist() == [[1.0, 0.0]] and ft == [6], f"{tot} charges={nf.tolist()} fail_type={ft}")


if __name__ == '__main__':
    torch.set_num_threads(1)
    print('=' * 78); print('판정기 v2 누적 주 상대 · 해소 F6 단위 테스트 (imo·none, CPU)'); print('=' * 78)
    for fn in (cum_primary, res_f6):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            check(f"{fn.__name__} 예외", False, f"{type(e).__name__}: {e}")
    print('=' * 78)
    print(f"VERDICT: {'ALL PASS' if all(RES) else 'FAIL ' + str(RES.count(False))}")
    sys.exit(0 if all(RES) else 1)
