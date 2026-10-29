"""test_preview_pbrs.py — 연속 위험 예고 PBRS(2026-10-28, VESSEL_PREVIEW_PBRS_COEF) 단위 테스트.

  python verify/test_preview_pbrs.py        # ALL PASS 여야 함 (imo·none, CPU, ~10 s)

φ_ij = g(dcpa)·h(tcpa)·w(dist) (Φ 전용 마스크: 자기·300 m 밖·가림 제외, 멀어지는 쌍 포함), Φ_i = −c·Σ_j φ_ij,
F_i = Σ_j keep_ij·(γφ′_ij − φ_ij), keep_ij = 두 배 모두 직전 유효·이번 결정 안 끝남. 끝난 배·재출현 첫 결정 F = 0.
F 는 env.last_preview_F(실제로 더한 값). (b) 는 같은 장면 c=1 / c=0 보상 차로도 확인(PBRS 는 난수 안 씀).
a  상대 거리가 56 m · 300 m 를 넘나들 때 Φ 연속(보상 위험 risk_rw 는 56 m 반경이면 끊김 — 대조로만 출력)
b  끝난 배(충돌·시간초과) 결정 F = 0 · 재출현 첫 결정 F = 0 (보상 차 == 0 정확히)
c  직진 통과 조우(300 m 밖 → 진입 → 지나감, 충돌 없음) Σ_t γ^t F_t = γ^T Φ_T − Φ_0 (= 0) · 300 m 진입 계단 없음 — float64
d  우현 변침(dcpa↑) 결정의 F > 직진 F, 그리고 F > 0
e  계수 0 이면 보상·obs·done·outcome 이 base 2cd2fbd vessel_gym 과 torch.equal (git 없으면 SKIP) + 키·F None
f  i 와 위험 코스인 j 가 제3의 배 k 와 충돌(300 m 안): 그 결정 i 의 F = 남은 쌍(i,m) 몫만, (i,j) 몫 0 — 옛 배당 값과 대조
g  i 옆 300 m 안 위험 코스로 상대가 재출현: 그 결정·다음 결정 i 의 F = 0 (음의 스파이크 없음) — 옛 배당 값과 대조
h  위험한 정면 조우(옆 간격 3 m) CPA 통과: 멀어지는 쌍 dcpa == 현재 거리 · 통과 구간 max|F| 작음(옛 마스크 +0.999) ·
   Σγ^tF = γ^TΦ_T − Φ_0. INFO = Φ 크기와 n_ 보상(반경 56·게이트 48·per-pair −0.15 exp 3) #6+#7 비용 합 — c 보정용
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


def make_env(E=1, N=2, seed=0, dtype=torch.float32, mod=vg, perpair=(0.0, 1.6)):
    env = mod.VesselBatchEnv(num_envs=E, n_vessels=N, device='cpu', seed=seed, ring_scale=1.0, crossing=0,
                             risk_range=cfg.COMM_RANGE, reward_range=56.0, dtype=dtype,
                             farfield_coef=0.0, perpair_coef=perpair[0], perpair_exp=perpair[1])
    env.reset()
    return env


def place(env, ships):
    """env 0 의 배 상태를 직접 넣고 '진행 중' 상태로 만든다(직전 φ = 현재 φ, 유효). ships = [dict(pos, hdg, spd, maxs, goal, steps)]."""
    for k, s in enumerate(ships):
        env.pos[0, k] = torch.tensor(s['pos'], dtype=env.dtype)
        env.heading[0, k] = s['hdg']
        env.speed[0, k] = s['spd']
        env.max_speed[0, k] = s.get('maxs', s['spd'])
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
    pair = env._last_pw['preview_pair']
    if pair is not None:
        env.prev_phi_pair = pair.clone()
        env.prev_phi_ok.fill_(True)


def acts(env, rud=None, spd=None):
    """rud [N] 명령 타각 비율(None = 0), spd [N] 목표 속력 m/s(None = 최고속)."""
    a0 = torch.zeros(1, env.N, dtype=env.dtype) if rud is None else torch.tensor(rud, dtype=env.dtype).view(1, -1)
    a1 = (torch.ones_like(a0) if spd is None else
          (2.0 * torch.tensor(spd, dtype=env.dtype).view(1, -1) / env.max_speed[0:1] - 1.0).clamp(-1, 1))
    return torch.stack([a0, a1], -1)


def run(ships, steps, coef, rud=None, dtype=torch.float32, gate=None, perpair=(0.0, 1.6)):
    """rud(t) -> [N] 명령 타각 비율(None = 0). 목표 속력 = 최고속. Returns per-step records."""
    saved = (vg.PREVIEW_PBRS_COEF, vg.RISK_DCPA_GATE_M)
    vg.PREVIEW_PBRS_COEF = coef
    if gate is not None:
        vg.RISK_DCPA_GATE_M = gate
    try:
        env = make_env(1, len(ships), dtype=dtype, perpair=perpair)
        place(env, ships)
        p0 = env._last_pw['preview_pair']
        rec = dict(phi0=(None if p0 is None else p0[0].sum(-1)), r=[], d=[], oc=[], phi=[], F=[], dist=[], dcpa=[],
                   riskrw=[])
        for t in range(steps):
            _, r, d, oc = env.step(acts(env, None if rud is None else rud(t)))
            pw = env._last_pw
            rec['r'].append(r[0].clone()); rec['d'].append(d[0].clone()); rec['oc'].append(oc[0].clone())
            rec['phi'].append(None if pw['preview_pair'] is None else pw['preview_pair'][0].sum(-1))
            rec['F'].append(None if env.last_preview_F is None else env.last_preview_F[0].clone())
            rec['dist'].append(float(pw['dist'][0, 0, 1])); rec['dcpa'].append(float(pw['dcpa'][0, 0, 1]))
            rec['riskrw'].append(pw['risk_rw'][0, 0].clone())
        return rec
    finally:
        vg.PREVIEW_PBRS_COEF, vg.RISK_DCPA_GATE_M = saved


def ship(x, z, hdg, spd=1.5, goal=None, steps=0, rud=0.0, maxs=None):
    gz = 280.0 if hdg == 0.0 else -280.0
    return dict(pos=(x, z), hdg=hdg, spd=spd, maxs=maxs or spd, goal=goal or (x, gz), steps=steps, rud=rud)


# ── a ──
def a_continuity():
    eps = 1e-3
    for R in (56.0, 300.0):
        out = {}
        for tag, D in (('in', R - eps), ('out', R + eps)):
            dx = 10.0
            z = (D * D - dx * dx) ** 0.5
            vg.PREVIEW_PBRS_COEF = 1.0
            try:
                env = make_env(dtype=torch.float64)
                place(env, [ship(0.0, 0.0, 0.0), ship(dx, z, 180.0)])
                pw = env._last_pw
                out[tag] = (pw['preview_pair'][0].sum(-1), float(pw['dist'][0, 0, 1]), pw['risk_rw'][0, 0, 1].item())
            finally:
                vg.PREVIEW_PBRS_COEF = 0.0
        (pi, di, ri), (po, do, ro) = out['in'], out['out']
        dphi = (pi - po).abs().max().item()
        nontriv = pi[0].item() < -0.5 if R == 56.0 else True
        check(f'a Φ {R:g} m 경계 연속', dphi < 1e-4 and nontriv,
              f"dist {di:.4f}/{do:.4f}  Φ0 {pi[0].item():.6f}/{po[0].item():.6f}  |ΔΦ|={dphi:.2e}"
              f"  (대조: risk_rw(반경56) {ri:.4f}/{ro:.4f})")
        if R == 56.0:
            check('a 두 배 Φ 대칭(같은 g·h·w)', abs(pi[0].item() - pi[1].item()) < 1e-9, f"{pi.tolist()}")


# ── b ──
def b_terminal_respawn():
    sc = [ship(0.0, -15.0, 0.0), ship(0.0, 15.0, 180.0)]           # 정면 충돌(30 m, 접근 3 m/s)
    on, off = run(sc, 25, 1.0), run(sc, 25, 0.0)
    Fd = torch.stack(on['r']) - torch.stack(off['r'])
    F = torch.stack(on['F'])
    d = torch.stack(on['d'])
    same = all(torch.equal(a, b) for a, b in zip(on['oc'], off['oc']))
    t_end = int(d[:, 0].nonzero()[0]) if bool(d[:, 0].any()) else -1
    oc_end = int(on['oc'][t_end][0]) if t_end >= 0 else -1
    check('b 충돌로 끝남(같은 결정, c 무관)', t_end > 0 and oc_end == vg.OUT_COLLISION_VESSEL and bool(d[t_end, 1]) and same,
          f"t_end={t_end} outcome={oc_end}")
    if t_end <= 0 or t_end + 1 >= F.shape[0]:
        return
    check('b 끝난 결정 F == 0 (두 배, 보상 차 정확히 0)', bool((F[t_end] == 0).all() and (Fd[t_end] == 0).all()),
          f"F={F[t_end].tolist()}")
    check('b 재출현 첫 결정 F == 0 (두 배, 보상 차 정확히 0)', bool((F[t_end + 1] == 0).all() and (Fd[t_end + 1] == 0).all()),
          f"F={F[t_end + 1].tolist()}")
    phis = [on['phi0']] + on['phi']
    exp = torch.stack([GAMMA * phis[t + 1] - phis[t] for t in range(t_end)])
    pre = F[:t_end]
    check('b 끝나기 전 F = γΦ′ − Φ (0 아님) = 보상 차', bool((pre != 0).all()) and (pre - exp).abs().max().item() < 1e-5
          and (Fd[:t_end] - pre).abs().max().item() < 1e-4,
          f"F[0..2]={[round(v, 5) for v in pre[:3, 0].tolist()]}  max|F−(γΦ′−Φ)|={(pre - exp).abs().max().item():.2e}")


# ── c ──
def c_telescoping():
    T = 380
    on = run([ship(0.0, -170.0, 0.0), ship(30.0, 170.0, 180.0)], T, 1.0, dtype=torch.float64)
    F = torch.stack(on['F'])
    phis = torch.stack([on['phi0']] + on['phi'])                       # [T+1, N]
    disc = torch.tensor([GAMMA ** t for t in range(T)], dtype=torch.float64).unsqueeze(-1)
    lhs = (disc * F).sum(0)
    rhs = GAMMA ** T * phis[-1] - phis[0]
    err = (lhs - rhs).abs().max().item()
    nodone = not bool(torch.stack(on['d']).any())
    i1 = int((phis[:, 0] != 0).nonzero()[0])
    dist = torch.tensor(on['dist'], dtype=torch.float64)
    far = (dist >= 240.0) & (torch.arange(T) < int(dist.argmin()))          # 접근 중 감쇠 띠(240–300 m)와 그 밖
    check('c 조우 동안 끝난 배 없음 · 300 m 밖 시작 · 지나감', nodone and on['dist'][0] > 300.0 and phis[0].abs().max() == 0
          and phis[-1].abs().max() == 0, f"dist {on['dist'][0]:.1f}→min {min(on['dist']):.1f}→{on['dist'][-1]:.1f}")
    check('c 300 m 진입 계단 없음(첫 비0 Φ · 240 m 밖 max|F|)', abs(phis[i1, 0].item()) < 0.01
          and F[far].abs().max().item() < 0.01,
          f"첫 비0 Φ={phis[i1, 0].item():.2e} @ dist {on['dist'][i1 - 1]:.2f}  240 m 밖 max|F|={F[far].abs().max().item():.5f}")
    ic = int(F[:, 0].abs().argmax())
    print(f"  INFO  c 전체 max|F|={F[ic, 0].item():+.4f} @ 결정 {ic} (dist {on['dist'][ic]:.1f} m, 최근접 결정 {int(dist.argmin())})")
    check('c Σγ^tF_t = γ^TΦ_T − Φ_0 (= 0)', err < 1e-9 and lhs.abs().max().item() < 1e-9,
          f"Σγ^tF={lhs.tolist()}  |오차|={err:.1e}  minΦ={phis.min().item():.4f}  (할인 없는 ΣF={F.sum(0).tolist()})")
    check('c 퍼텐셜이 실제로 움직임(minΦ < −0.3)', phis.min().item() < -0.3)


# ── d ──
def d_starboard():
    base = [ship(0.0, -100.0, 0.0), ship(-30.0, 100.0, 180.0)]           # 상대가 좌현 30 m 옆으로 지나가는 정면
    turn = [ship(0.0, -100.0, 0.0, rud=30.0), ship(-30.0, 100.0, 180.0)]  # 배 0 우현 전타 중
    s = run(base, 1, 1.0, dtype=torch.float64)
    t = run(turn, 1, 1.0, rud=lambda _: [1.0, 0.0], dtype=torch.float64)
    fs, ft = s['F'][0][0].item(), t['F'][0][0].item()
    check('d 우현 변침이 dcpa 를 키움', t['dcpa'][0] > s['dcpa'][0], f"dcpa 직진 {s['dcpa'][0]:.3f} 변침 {t['dcpa'][0]:.3f}")
    check('d 변침 F > 직진 F, 변침 F > 0', ft > fs and ft > 0.0, f"F 직진 {fs:.5f}  변침 {ft:.5f}  Φ0 {s['phi0'][0].item():.4f}")


# ── e ──
def e_off_equals_base():
    vg.PREVIEW_PBRS_COEF = 0.0
    env = make_env()
    env.step(acts(env))
    check('e 계수 0 이면 쌍 퍼텐셜 키·last_preview_F None(연산 없음)',
          env._last_pw['preview_pair'] is None and env.last_preview_F is None)
    try:
        src = subprocess.run(['git', '-C', PYROOT, 'show', f'{BASE_REV}:Python/vessel_gym.py'],
                             capture_output=True, text=True, encoding='utf-8', check=True).stdout   # Windows 기본 cp949 로 읽으면 한글·기호에서 깨짐
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


def _old_F(prev_pair, pair, prev_ok, done):
    """옛(배당 합) 규칙: F_i = ok_i·(γΦ′_i − Φ_i), ok_i = 직전 유효 & 안 끝남 — 대조용."""
    ok = prev_ok & ~done
    return torch.where(ok, GAMMA * pair.sum(-1) - prev_pair.sum(-1), torch.zeros_like(ok, dtype=pair.dtype))


# ── f ──
def f_third_party_collision():
    """i(0) 북행 · j(1) i 와 정면(200 m, 옆 5 m) · k(2) j 앞 20 m 에서 j 와 정면 → 몇 결정 뒤 j·k 충돌. m(3) i 와 옆 20 m 정면(남은 쌍)."""
    vg.PREVIEW_PBRS_COEF = 1.0
    try:
        env = make_env(N=4, dtype=torch.float64)
        place(env, [ship(0.0, -100.0, 0.0), ship(5.0, 100.0, 180.0), ship(5.0, 80.0, 0.0), ship(-20.0, 150.0, 180.0)])
        hit = None
        for t in range(20):
            prev_pair, prev_ok = env.prev_phi_pair[0].clone(), env.prev_phi_ok[0].clone()
            prev_dij = float(env._last_pw['dist'][0, 0, 1])
            _, r, d, oc = env.step(acts(env))
            if bool(d[0].any()):
                hit = (t, prev_pair, prev_ok, env._last_pw['preview_pair'][0].clone(), d[0].clone(), oc[0].clone(),
                       env.last_preview_F[0].clone(), prev_dij)
                break
    finally:
        vg.PREVIEW_PBRS_COEF = 0.0
    ok_scene = hit is not None and bool(hit[4][1] and hit[4][2]) and not bool(hit[4][0] or hit[4][3]) \
        and int(hit[5][1]) == vg.OUT_COLLISION_VESSEL
    check('f j·k 충돌(i·m 은 진행 중)', ok_scene, '' if hit is None else f"t={hit[0]} done={hit[4].tolist()} oc={hit[5].tolist()}")
    if not ok_scene:
        return
    t, pp, pok, pair, done, _, F, dij = hit
    keep_im = GAMMA * pair[0, 3] - pp[0, 3]
    old = _old_F(pp, pair, pok, done)[0]
    check('f i 의 F = 남은 쌍(i,m) 몫만 · (i,j) 몫 0', abs(F[0].item() - keep_im.item()) < 1e-12 and abs(keep_im.item()) > 1e-4
          and abs(pp[0, 1].item()) > 0.1,
          f"F_i={F[0].item():.5f} (i,m) 몫={keep_im.item():.5f}  직전 φ_ij={pp[0, 1].item():.4f} @ dist {dij:.1f} m"
          f"  옛 배당 F_i={old.item():.5f}")
    check('f 옛 규칙이면 스파이크(|옛 − 새| > 0.1)', abs(old.item() - F[0].item()) > 0.1)


# ── g ──
def g_respawn_near():
    """i(0) (0,100) 북행 1.5 m/s(위쪽 스폰 줄로 접근) · j(1) i 뒤에서 남행(멀어짐, φ_ij = 0)인데 첫 결정에 시간초과 → 재출현.
    재출현 위치가 i 와 300 m 안 위험 코스(φ′ < 0)가 되는 시드를 찾아, 그 결정·다음 결정 i 의 F = 0 확인.
    셋째 결정부터는 (i,j) 쌍이 다시 이어짐(|F| 작음). 재출현 초기 속력이 낮아(0.2–0.5×최고속) i 가 다가가야 h > 0."""
    last = vg.MAX_EPISODE_STEPS - vg.SUBSTEPS
    found = None
    for seed in range(200):
        vg.PREVIEW_PBRS_COEF = 1.0
        try:
            env = make_env(seed=seed, dtype=torch.float64)
            place(env, [ship(0.0, 100.0, 0.0), ship(0.0, -150.0, 180.0, steps=last)])
            spd = None
            p0, ok0 = env.prev_phi_pair[0].clone(), env.prev_phi_ok[0].clone()
            _, r, d, oc = env.step(acts(env, spd=spd))
            p1 = env._last_pw['preview_pair'][0].clone()
            F0, d0 = env.last_preview_F[0].clone(), d[0].clone()
            if not (bool(d0[1]) and not bool(d0[0]) and p1[0, 1].item() < -0.01 and p0[0, 1].item() == 0.0):
                continue
            _, r, d, oc = env.step(acts(env, spd=spd))
            F1 = env.last_preview_F[0].clone()
            _, r, d, oc = env.step(acts(env, spd=spd))
            F2 = env.last_preview_F[0].clone()
            found = (seed, p0, ok0, p1, d0, F0, F1, F2, float(env._last_pw['dist'][0, 0, 1]))
            break
        finally:
            vg.PREVIEW_PBRS_COEF = 0.0
    check('g 300 m 안 위험 코스 재출현 장면 찾음', found is not None, '' if found is None else f"seed={found[0]}")
    if found is None:
        return
    seed, p0, ok0, p1, d0, F0, F1, F2, dist2 = found
    old = _old_F(p0, p1, ok0, d0)[0]
    check('g 재출현 결정·다음 결정 i 의 F = 0 (음의 스파이크 없음)', F0[0].item() == 0.0 and F1[0].item() == 0.0,
          f"F_i {F0[0].item():.5f}/{F1[0].item():.5f}  재출현 φ′_ij={p1[0, 1].item():.4f}  옛 배당 F_i={old.item():.5f}")
    check('g 옛 규칙이면 음의 스파이크 · 셋째 결정 쌍 재개(|F| < 0.02)', old.item() < -0.01 and 0 < abs(F2[0].item()) < 0.02,
          f"셋째 결정 F_i={F2[0].item():.5f} @ dist {dist2:.1f} m")


# ── h ──
def h_cpa_passage():
    """위험한 정면(옆 간격 3 m) 직진 통과, c=1. CPA 통과 구간 F 크기 · telescoping · c 보정용 INFO.
    n_ 보상 설정(반경 56·DCPA 게이트 48·per-pair −0.15 exp 3)의 #6·#7 비용 합과 나란히 출력(Φ·F 는 보상 설정과 무관)."""
    pp = (cfg.PERPAIR_COEF, 3.0)                       # = vessel_gym_train 이 env 에 넘기는 값
    T = 380
    saved = vg.PREVIEW_PBRS_COEF
    vg.PREVIEW_PBRS_COEF = 1.0
    try:                                               # 멀어지는 쌍의 dcpa 가 현재 거리와 같은 값인지(마스크 바꾼 근거)
        env = make_env(dtype=torch.float64)
        place(env, [ship(0.0, 0.0, 0.0), ship(3.0, -40.0, 180.0)])
        pw = env._last_pw
        rec_ok = bool(pw['raw_tcpa'][0, 0, 1] < 0) and torch.equal(pw['dcpa'][0, 0, 1], pw['dist'][0, 0, 1]) \
            and pw['tcpa'][0, 0, 1].item() == 0.0
    finally:
        vg.PREVIEW_PBRS_COEF = saved
    check('h 멀어지는 쌍: tcpa = 0 · dcpa == 현재 거리(비트 같음)', rec_ok)
    rec = run([ship(0.0, -170.0, 0.0), ship(3.0, 170.0, 180.0)], T, 1.0, dtype=torch.float64, gate=48.0, perpair=pp)
    phis = torch.stack([rec['phi0']] + rec['phi'])                     # [T+1, N]
    phi = phis[1:, 0]
    F = torch.stack(rec['F'])                                          # [T, N]
    dist = torch.tensor(rec['dist'], dtype=torch.float64)
    done = bool(torch.stack(rec['d']).any())
    icpa = int(dist.argmin())
    win = dist < 60.0                                                  # CPA 앞뒤 60 m 안
    fmax = F[win].abs().max().item()
    iw = int(torch.where(win, F[:, 0].abs(), torch.zeros_like(dist)).argmax())
    check('h 충돌 없이 지나감 · 끝 Φ = 0', not done and phis[-1].abs().max().item() == 0.0,
          f"최근접 {dist[icpa].item():.2f} m @ 결정 {icpa} · 끝 dist {dist[-1].item():.1f} m")
    check('h CPA 통과 구간(60 m 안) max|F| < 0.1 (옛 마스크 +0.999)', fmax < 0.1,
          f"max|F|={fmax:.4f} @ 결정 {iw} (dist {dist[iw].item():.1f} m, CPA 뒤 {iw - icpa} 결정) · 전체 max|F|={F.abs().max().item():.4f}")
    disc = torch.tensor([GAMMA ** t for t in range(T)], dtype=torch.float64).unsqueeze(-1)
    lhs = (disc * F).sum(0)
    rhs = GAMMA ** T * phis[-1] - phis[0]
    check('h Σγ^tF_t = γ^TΦ_T − Φ_0 (= 0)', (lhs - rhs).abs().max().item() < 1e-9 and lhs.abs().max().item() < 1e-9,
          f"Σγ^tF={lhs[0].item():.2e}  할인 없는 ΣF={F[:, 0].sum().item():.4f}")
    nz = (phi != 0).nonzero()
    i300 = int(nz[0]) if len(nz) else -1
    i56 = int((dist <= 56.0).nonzero()[0])
    rr = torch.stack(rec['riskrw'])                                    # [T,N] 배 0 의 상대별 risk_rw
    mr = rr.max(dim=-1).values
    cost6 = torch.where(mr > 0.05, 0.8 * mr ** 3, torch.zeros_like(mr)) * vg.SUBSTEPS
    cost7 = -pp[0] * torch.where(rr > 0.05, rr.clamp(max=1.0) ** pp[1], torch.zeros_like(rr)).sum(-1) * vg.SUBSTEPS
    print(f"  INFO  위험 정면(간격 3 m, 접근 3 m/s, c=1, T={vg.PREVIEW_T_S:g}s, G={vg.PREVIEW_DCPA_G:g}m, "
          f"TAPER={vg.PREVIEW_TAPER_M:g}m)")
    print(f"        Φ 첫 비0 = {phi[i300].item():.2e} @ dist {dist[i300].item():.1f} m · Φ @240 m 안쪽 첫 결정 = "
          f"{phi[int((dist <= 240.0).nonzero()[0])].item():.4f}")
    print(f"        Φ @56 m = {phi[i56].item():.4f} · minΦ = {phi.min().item():.4f} · 직진 접근 F @56 m ≈ {F[i56 + 1, 0].item():.5f}/결정")
    print(f"        n_ 보상(56 m 안, 게이트 48, ×SUBSTEPS) #6 충돌코스 합 = {cost6.sum().item():.2f} · #7 per-pair 합 = "
          f"{cost7.sum().item():.2f} · #6+#7 = {(cost6 + cost7).sum().item():.2f} ({int((cost6 > 0).sum())} 결정, per-pair {pp})")


if __name__ == '__main__':
    torch.set_num_threads(1)
    print('=' * 78); print('연속 위험 예고 PBRS 테스트 (imo·none, CPU)'); print('=' * 78)
    for fn in (a_continuity, b_terminal_respawn, c_telescoping, d_starboard, e_off_equals_base,
               f_third_party_collision, g_respawn_near, h_cpa_passage):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(f"{fn.__name__} 예외", False, f"{type(e).__name__}: {e}")
        finally:
            vg.PREVIEW_PBRS_COEF = 0.0
    n_ok = sum(RES)
    print(f"\n{'ALL PASS' if all(RES) else '★FAIL'}  {n_ok}/{len(RES)}")
    sys.exit(0 if all(RES) else 1)
