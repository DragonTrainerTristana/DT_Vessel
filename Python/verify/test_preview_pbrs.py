"""test_preview_pbrs.py — 연속 위험 예고 PBRS(2026-10-28, VESSEL_PREVIEW_PBRS_COEF) 단위 테스트.

  python verify/test_preview_pbrs.py        # ALL PASS 여야 함 (imo·none, CPU, ~10 s)

Φ_i = −c·Σ_j g(dcpa_ij)·h(tcpa_ij) ('risk' 쌍 마스크, 300 m), F = γΦ(s′) − Φ(s), 끝난 배·재출현 첫 결정 F = 0.
F 는 같은 장면·같은 시드로 c=1 / c=0 두 번 굴려 보상 차로 잼(PBRS 는 난수를 안 씀 → 두 런 상태 동일).
a  상대 거리가 56 m 를 넘나들 때 Φ 연속(보상 위험 risk_rw 는 56 m 반경이면 끊김 — 대조로만 출력)
b  끝난 배(충돌·시간초과) 결정 F = 0 · 재출현 첫 결정 F = 0 · 안 끝난 배 F = γΦ(재스폰 뒤 상태) − Φ(직전)
c  직진 통과 조우(300 m 밖 → 지나감, 충돌 없음) Σ_t γ^t F_t = γ^T Φ_T − Φ_0 (= 0, 쌓아 먹기 없음) — float64
d  우현 변침(dcpa↑) 결정의 F > 직진 F, 그리고 F > 0
e  계수 0 이면 보상·obs·done·outcome 이 base 2cd2fbd vessel_gym 과 torch.equal (git 없으면 SKIP) + Φ 키 None
info  위험한 정면 조우(옆 간격 3 m) 직진 통과의 Φ 크기와 n_ 보상(반경 56·게이트 48) 충돌코스 #6 비용 합 — c 보정용
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
sys.path.insert(0, PYROOT)
for _k in [k for k in os.environ if k.startswith('VESSEL_')]:
    os.environ.pop(_k)
os.environ.update({'VESSEL_DYN_PROFILE': 'imo', 'VESSEL_OBSTACLES': 'none'})

import torch  # noqa: E402
import config as cfg  # noqa: E402
import vessel_gym as vg  # noqa: E402

RES = []
BASE_REV = '2cd2fbd'
GAMMA = cfg.DISCOUNT_FACTOR


def check(name, ok, info=''):
    RES.append(bool(ok))
    print(f"  {'PASS' if ok else '★FAIL'}  {name}  {info}")


def make_env(E=1, N=2, seed=0, dtype=torch.float32, mod=vg):
    env = mod.VesselBatchEnv(num_envs=E, n_vessels=N, device='cpu', seed=seed, ring_scale=1.0, crossing=0,
                             risk_range=cfg.COMM_RANGE, reward_range=56.0, dtype=dtype,
                             farfield_coef=0.0, perpair_coef=0.0)
    env.reset()
    return env


def place(env, ships):
    """env 0 의 배 상태를 직접 넣고 '진행 중' 상태로 만든다(prev_phi = 현재 Φ, 유효). ships = [dict(pos, hdg, spd, goal, steps)]."""
    for k, s in enumerate(ships):
        env.pos[0, k] = torch.tensor(s['pos'], dtype=env.dtype)
        env.heading[0, k] = s['hdg']
        env.speed[0, k] = s['spd']
        env.max_speed[0, k] = s['spd']
        env.target_speed[0, k] = s['spd']
        env.rudder[0, k] = s.get('rud', 0.0)
        env.cmd_rudder[0, k] = s.get('rud', 0.0)
        env.goal[0, k] = torch.tensor(s['goal'], dtype=env.dtype)
        env.step_count[0, k] = s.get('steps', 0)
    env.prev_dist = torch.linalg.norm(env.goal - env.pos, dim=-1)
    env.prev_dcpa.fill_(-1.0)
    env.prev_danger_idx.fill_(-1)
    env.prev_rudder = env.rudder.clone()
    env.prev_far_risk.fill_(-1.0)
    env._update_situation()
    phi = env._last_pw['preview_phi']
    if phi is not None:
        env.prev_phi = phi.clone()
        env.prev_phi_ok.fill_(True)


def run(ships, steps, coef, rud=None, dtype=torch.float32, gate=None):
    """rud(t) -> [N] 명령 타각 비율(None = 0). 목표 속력 = 현재 최고속(a1 = 1). Returns per-step records."""
    saved = (vg.PREVIEW_PBRS_COEF, vg.RISK_DCPA_GATE_M)
    vg.PREVIEW_PBRS_COEF = coef
    if gate is not None:
        vg.RISK_DCPA_GATE_M = gate
    try:
        env = make_env(1, len(ships), dtype=dtype)
        place(env, ships)
        rec = dict(phi0=(None if env._last_pw['preview_phi'] is None else env._last_pw['preview_phi'].clone()),
                   r=[], d=[], oc=[], phi=[], dist=[], dcpa=[], riskrw=[])
        for t in range(steps):
            a0 = torch.zeros(1, env.N, dtype=dtype) if rud is None else torch.tensor(rud(t), dtype=dtype).view(1, -1)
            _, r, d, oc = env.step(torch.stack([a0, torch.ones_like(a0)], -1))
            pw = env._last_pw
            rec['r'].append(r[0].clone()); rec['d'].append(d[0].clone()); rec['oc'].append(oc[0].clone())
            rec['phi'].append(None if pw['preview_phi'] is None else pw['preview_phi'][0].clone())
            rec['dist'].append(float(pw['dist'][0, 0, 1])); rec['dcpa'].append(float(pw['dcpa'][0, 0, 1]))
            rec['riskrw'].append(pw['risk_rw'][0].max(dim=-1).values.clone())
        return rec
    finally:
        vg.PREVIEW_PBRS_COEF, vg.RISK_DCPA_GATE_M = saved


def pair_F(ships, steps, rud=None, dtype=torch.float32, gate=None):
    """같은 장면 c=1 / c=0 → F_t = r_on − r_off ([T,N]) 와 c=1 기록."""
    on = run(ships, steps, 1.0, rud, dtype, gate)
    off = run(ships, steps, 0.0, rud, dtype, gate)
    F = torch.stack(on['r']) - torch.stack(off['r'])
    same = all(torch.equal(a, b) for a, b in zip(on['oc'], off['oc']))
    return F, on, off, same


def ship(x, z, hdg, spd=1.5, goal=None, steps=0, rud=0.0):
    gz = 280.0 if hdg == 0.0 else -280.0
    return dict(pos=(x, z), hdg=hdg, spd=spd, goal=goal or (x, gz), steps=steps, rud=rud)


# ── a ──
def a_continuity_56():
    eps = 1e-3
    out = {}
    for tag, D in (('in', 56.0 - eps), ('out', 56.0 + eps)):
        dx = 10.0
        z = (D * D - dx * dx) ** 0.5
        vg.PREVIEW_PBRS_COEF = 1.0
        try:
            env = make_env(dtype=torch.float64)
            place(env, [ship(0.0, 0.0, 0.0), ship(dx, z, 180.0)])
            pw = env._last_pw
            out[tag] = (pw['preview_phi'][0].clone(), float(pw['dist'][0, 0, 1]), pw['risk_rw'][0, 0, 1].item())
        finally:
            vg.PREVIEW_PBRS_COEF = 0.0
    (pi, di, ri), (po, do, ro) = out['in'], out['out']
    dphi = (pi - po).abs().max().item()
    check('a Φ 56 m 경계 연속', dphi < 1e-4 and pi[0].item() < -0.5,
          f"dist {di:.4f}/{do:.4f}  Φ0 {pi[0].item():.6f}/{po[0].item():.6f}  |ΔΦ|={dphi:.2e}"
          f"  (대조: risk_rw(반경56) {ri:.4f}/{ro:.4f})")
    check('a 두 배 Φ 대칭(같은 g·h)', abs(pi[0].item() - pi[1].item()) < 1e-9, f"{pi.tolist()}")


# ── b ──
def b_terminal_respawn():
    # 정면 충돌(30 m, 접근 3 m/s)
    F, on, off, same = pair_F([ship(0.0, -15.0, 0.0), ship(0.0, 15.0, 180.0)], 25)
    d = torch.stack(on['d'])
    t_end = int(d[:, 0].nonzero()[0]) if bool(d[:, 0].any()) else -1
    oc_end = int(on['oc'][t_end][0]) if t_end >= 0 else -1
    check('b 충돌로 끝남(같은 결정, c 무관)', t_end > 0 and oc_end == vg.OUT_COLLISION_VESSEL and bool(d[t_end, 1]) and same,
          f"t_end={t_end} outcome={oc_end}")
    if t_end <= 0 or t_end + 1 >= F.shape[0]:
        return
    check('b 끝난 결정 F == 0 (두 배)', bool((F[t_end] == 0).all()), f"F={F[t_end].tolist()}")
    check('b 재출현 첫 결정 F == 0 (두 배)', bool((F[t_end + 1] == 0).all()), f"F={F[t_end + 1].tolist()}")
    pre = F[:t_end]
    phis = [on['phi0'][0]] + on['phi']
    exp = torch.stack([GAMMA * phis[t + 1] - phis[t] for t in range(t_end)])
    check('b 끝나기 전 F = γΦ′ − Φ (0 아님)', bool((pre != 0).all()) and (pre - exp).abs().max().item() < 1e-5,
          f"F[0..2]={[round(v, 5) for v in pre[:3, 0].tolist()]}  max|F−(γΦ′−Φ)|={(pre - exp).abs().max().item():.2e}")

    # 시간초과(배 0 만) — 배 1 은 안 끝남: F_1 = γΦ_1(재스폰 뒤) − Φ_1(직전)
    last = vg.MAX_EPISODE_STEPS - vg.SUBSTEPS
    F2, on2, _, same2 = pair_F([ship(0.0, -60.0, 0.0, steps=last - 2 * vg.SUBSTEPS), ship(5.0, 60.0, 180.0)], 4)
    d2 = torch.stack(on2['d'])
    t2 = int(d2[:, 0].nonzero()[0]) if bool(d2[:, 0].any()) else -1
    ok_to = t2 >= 0 and int(on2['oc'][t2][0]) == vg.OUT_TIMEOUT and not bool(d2[t2, 1]) and same2
    check('b 시간초과(배 0만) 발생', ok_to, f"t={t2}")
    if ok_to:
        phi_prev = (on2['phi0'] if t2 == 0 else on2['phi'][t2 - 1])[1]
        exp1 = GAMMA * on2['phi'][t2][1] - phi_prev
        check('b 시간초과 배 F == 0 · 남은 배 F = γΦ(재스폰 뒤) − Φ', F2[t2, 0].item() == 0.0
              and abs(F2[t2, 1].item() - exp1.item()) < 1e-5 and F2[t2, 1].item() > 0.0,
              f"F={F2[t2].tolist()}  기대(배1)={exp1.item():.5f}  Φ1 {phi_prev.item():.4f}→{on2['phi'][t2][1].item():.4f}")


# ── c ──
def c_telescoping():
    T = 380
    F, on, off, same = pair_F([ship(0.0, -170.0, 0.0), ship(30.0, 170.0, 180.0)], T, dtype=torch.float64)
    phis = torch.stack([on['phi0'][0]] + on['phi'])                       # [T+1, N]
    disc = torch.tensor([GAMMA ** t for t in range(T)], dtype=torch.float64).unsqueeze(-1)
    lhs = (disc * F).sum(0)
    rhs = GAMMA ** T * phis[-1] - phis[0]
    err = (lhs - rhs).abs().max().item()
    nodone = not bool(torch.stack(on['d']).any()) and same
    check('c 조우 동안 끝난 배 없음 · 300 m 밖 시작 · 지나감', nodone and on['dist'][0] > 300.0 and phis[0].abs().max() == 0
          and phis[-1].abs().max() == 0, f"dist {on['dist'][0]:.1f}→min {min(on['dist']):.1f}→{on['dist'][-1]:.1f}")
    check('c Σγ^tF_t = γ^TΦ_T − Φ_0 (= 0)', err < 1e-9 and lhs.abs().max().item() < 1e-9,
          f"Σγ^tF={lhs.tolist()}  |오차|={err:.1e}  minΦ={phis.min().item():.4f}  (할인 없는 ΣF={F.sum(0).tolist()})")
    check('c 퍼텐셜이 실제로 움직임(minΦ < −0.3)', phis.min().item() < -0.3)


# ── d ──
def d_starboard():
    base = [ship(0.0, -100.0, 0.0), ship(-30.0, 100.0, 180.0)]           # 상대가 좌현 30 m 옆으로 지나가는 정면
    turn = [ship(0.0, -100.0, 0.0, rud=30.0), ship(-30.0, 100.0, 180.0)]  # 배 0 우현 전타 중
    Fs, ons, _, _ = pair_F(base, 1, dtype=torch.float64)
    Ft, ont, _, _ = pair_F(turn, 1, rud=lambda t: [1.0, 0.0], dtype=torch.float64)
    fs, ft = Fs[0, 0].item(), Ft[0, 0].item()
    check('d 우현 변침이 dcpa 를 키움', ont['dcpa'][0] > ons['dcpa'][0], f"dcpa 직진 {ons['dcpa'][0]:.3f} 변침 {ont['dcpa'][0]:.3f}")
    check('d 변침 F > 직진 F, 변침 F > 0', ft > fs and ft > 0.0, f"F 직진 {fs:.5f}  변침 {ft:.5f}  Φ0 {ons['phi0'][0, 0].item():.4f}")


# ── e ──
def e_off_equals_base():
    vg.PREVIEW_PBRS_COEF = 0.0
    env = make_env()
    check('e 계수 0 이면 Φ 키 None(연산 없음)', env._last_pw['preview_phi'] is None)
    try:
        src = subprocess.run(['git', '-C', PYROOT, 'show', f'{BASE_REV}:Python/vessel_gym.py'],
                             capture_output=True, text=True, check=True).stdout
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP  e base {BASE_REV} 비교 (git show 실패: {type(e).__name__})")
        return
    td = tempfile.mkdtemp()
    p = os.path.join(td, 'vessel_gym_base.py')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location('vessel_gym_base', p)
    vgb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vgb)

    def roll(mod):
        env = make_env(E=4, N=16, seed=3, mod=mod)
        g = torch.Generator().manual_seed(3)
        out = []
        for _ in range(30):
            tg = env.goal - env.pos
            a0 = torch.clamp(mod._wrap180(torch.atan2(tg[..., 0], tg[..., 1]) / mod.DEG - env.heading) / 20.0, -1, 1)
            a0 = torch.where(env.situation > 0, torch.full_like(a0, 0.6), a0) + (torch.rand(a0.shape, generator=g) - 0.5) * 0.2
            o, r, d, oc = env.step(torch.stack([a0, torch.ones_like(a0)], -1))
            out.append((o.clone(), r.clone(), d.clone(), oc.clone()))
        return out

    a, b = roll(vg), roll(vgb)
    eq = all(all(torch.equal(x, y) for x, y in zip(sa, sb)) for sa, sb in zip(a, b))
    nd = int(sum(int(s[2].sum()) for s in a))
    check(f'e 계수 0 = base {BASE_REV} (30결정 E4 N16 보상·obs·done·outcome torch.equal)', eq, f"끝난 배 {nd}")


# ── info ──
def info_calibration():
    """위험한 정면(옆 간격 3 m) 직진 통과, c=1. n_ 보상 설정(보상 반경 56·DCPA 게이트 48)의 #6 충돌코스 비용 합과 나란히."""
    rec = run([ship(0.0, -170.0, 0.0), ship(3.0, 170.0, 180.0)], 380, 1.0, dtype=torch.float64, gate=48.0)
    phi = torch.stack(rec['phi'])[:, 0]
    dist = torch.tensor(rec['dist'], dtype=torch.float64)
    nz = (phi != 0).nonzero()
    i300 = int(nz[0]) if len(nz) else -1
    i56 = int((dist <= 56.0).nonzero()[0])
    rr = torch.stack(rec['riskrw'])[:, 0]
    cost6 = torch.where(rr > 0.05, 0.8 * rr ** 3, torch.zeros_like(rr)) * vg.SUBSTEPS
    done = bool(torch.stack(rec['d']).any())
    print(f"  INFO  위험 정면(간격 3 m, 접근 3 m/s, c=1, T={vg.PREVIEW_T_S:g}s, G={vg.PREVIEW_DCPA_G:g}m): 끝남={done}")
    print(f"        Φ 첫 비0 = {phi[i300].item():.4f} @ dist {dist[i300].item():.1f} m (300 m 진입 순간 계단)")
    print(f"        Φ @56 m = {phi[i56].item():.4f} · minΦ = {phi.min().item():.4f} · 직진 접근 F ≈ "
          f"{(GAMMA * phi[i56 + 1] - phi[i56]).item():.5f}/결정")
    print(f"        n_ 보상 #6 충돌코스 비용 합(56 m 안, 게이트 48, ×SUBSTEPS) = {cost6.sum().item():.2f} "
          f"({int((cost6 > 0).sum())} 결정)")


if __name__ == '__main__':
    torch.set_num_threads(1)
    print('=' * 78); print('연속 위험 예고 PBRS 테스트 (imo·none, CPU)'); print('=' * 78)
    for fn in (a_continuity_56, b_terminal_respawn, c_telescoping, d_starboard, e_off_equals_base, info_calibration):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(f"{fn.__name__} 예외", False, f"{type(e).__name__}: {e}")
        finally:
            vg.PREVIEW_PBRS_COEF = 0.0
    n_ok = sum(RES)
    print(f"\n{'ALL PASS' if all(RES) else '★FAIL'}  {n_ok}/{len(RES)}")
    sys.exit(0 if all(RES) else 1)
