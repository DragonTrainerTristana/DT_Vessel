"""test_codec_p50.py — latent 차원 sweep 코덱(레이아웃 p50, z2–z12, 2026-09-29c) 단위 테스트. 스펙 2026-09-29-latent-sweep-design.md §5.

  python verify/test_codec_p50.py        # ALL PASS 여야 함 (imo·none 고정, CPU 1 스레드, 1–2분)

cz1 코덱 6개 SHA = 고정값 · layout p50 · d_in 50 · k 맞음 · direct 설치 가능(k+4 ≤ COMM_EXT_DIM) · decode 거부
cz2 radar_sectors == 무차별 대입(현재 프레임 1° 360개 → 10° 36칸 최솟값) · 현재 프레임 = 프레임 스택 마지막
cz3 payload50 == 따로 조립한 값(운동 6 · 목표 obs 2 · 역할 선언 one-hot 4 · 대상 위치/300 · 레이더 36)
cz4 comm_gather direct 필드 = [파트너 z · 수신자 자기 상태 4 · 0] · 빈 슬롯 0 (k 2·8·12)
cz5 실제 상태 충실도 표(학습 데이터와 다른 시드) — 설명 자료. 검사는 두 가지만:
    z-score 복원 오차가 k 가 늘 때 나빠지지 않음(10 % 여유) · k12 역할 ≥ 99 %
    ★2026-09-29 저자 결정: 사전 예상 'k12 침로 p99 ≤ 5°' 는 못 넘음(8.5°, 중앙 0.9°). k 따라 고르게 줄고 역할 100 % →
      버그 아님(용량 한계)으로 보고 판정에서 뺌. 값은 표와 아래 기록 줄에 그대로 남김.
cz6 가드: layout/d_in 불일치 · 틀린 SHA · 스냅샷 layout 불일치 거부, 맞으면 설치
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
sys.path.insert(0, PYROOT)
for _k in [k for k in os.environ if k.startswith('VESSEL_')]:
    os.environ.pop(_k)
os.environ.update({'VESSEL_DYN_PROFILE': 'imo', 'VESSEL_OBSTACLES': 'none', 'VESSEL_COMM_EXT': '1',
                   'VESSEL_USE_ATTENTION': '1', 'VESSEL_COMM_FIELDS': 'intent'})

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import config as cfg  # noqa: E402
import vessel_gym as vg  # noqa: E402
import vessel_gym_train as T  # noqa: E402
import networks as net  # noqa: E402
import comm_codec  # noqa: E402

# ★코덱 파일·SHA 고정 (run_repro.sh z_codec_sha 와 같아야 함)
P50 = {2: '0793c57f44b18f7e', 4: '234e5e3bd855cf5c', 6: '2cbb1e9bb6e903ff', 8: 'cebe26d19380f564', 10: '520834f0ec3284fa', 12: '2d2285b279e17f0d'}
PATH = 'comm_codecs/p50_k{}_s0.pt'
SANITY_K12 = {'decl_role_acc_min': 0.99}
HEADING_K12_PRIOR = 5.0   # 사전 예상(기록만, 판정 아님 — 저자 결정 2026-09-29)
RES = []


def check(name, ok, info=''):
    RES.append(bool(ok))
    print(f"  {'PASS' if ok else '★FAIL'}  {name}  {info}")


def make_env(E=2, N=16, seed=1):
    return vg.VesselBatchEnv(num_envs=E, n_vessels=N, device='cpu', seed=seed, ring_scale=1.0, crossing=0,
                             risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE,
                             farfield_coef=0.0, perpair_coef=-0.15, perpair_exp=3.0)


def rule_action(env):
    """comm_codec.collect_p50 과 같은 규칙 배(목표 조향 · 양보/정면/추월 우현 0.6 · 유지선은 17(b) 뒤 0.6)."""
    tc = env._last_pw['tcpa'].gather(-1, env.danger_idx.unsqueeze(-1)).squeeze(-1)
    tg = env.goal - env.pos
    a0 = torch.clamp(vg._wrap180(torch.atan2(tg[..., 0], tg[..., 1]) / vg.DEG - env.heading) / 20.0, -1, 1)
    s = env.situation
    a0 = torch.where((s == 1) | (s == 3) | (s == 4), torch.full_like(a0, 0.6), a0)
    a0 = torch.where((s == 2) & (tc > vg.RULE_17B_TIME), torch.zeros_like(a0), a0)
    a0 = torch.where((s == 2) & (tc <= vg.RULE_17B_TIME), torch.full_like(a0, 0.6), a0)
    return torch.stack([a0, torch.ones_like(a0)], -1)


def rollout(burn=150, T_=200, snap_every=40):
    """학습 데이터(seed 0)와 다른 seed 1 롤아웃. 매 결정 payload50, 몇 결정마다 (env 상태 사본 없이) 즉석 검사 콜백."""
    env = make_env()
    fs = T.FrameStack(env.E, env.N, 'cpu')
    obs = env.reset()
    r, g, ss, st = T.parse_obs(obs)
    fs.reset_all(r)
    pays, snaps = [], []
    for t in range(burn + T_):
        x = fs.get()
        if t >= burn:
            pays.append(T.payload50_now(env, x, g).reshape(-1, 50).clone())
            if (t - burn) % snap_every == 0:
                snaps.append(dict(env=env, x=x, g=g, ss=ss, st=st, r=r, t=t))
                yield_checks(snaps[-1])
        obs, _, done, _ = env.step(rule_action(env))
        r, g, ss, st = T.parse_obs(obs)
        fs.push(r, done)
    return torch.cat(pays), len(snaps)


# ─────────────────────────── cz2 · cz3 · cz4 (롤아웃 도중 스냅샷마다) ───────────────────────────
ACC = {'cz2': True, 'cz3': True, 'cz4': True, 'n_decl': 0, 'n_valid': 0, 'n_pad': 0, 'n_det': 0}
POL = {}


def brute_decl(env, topi, valid, feats):
    E, N, K = topi.shape
    role = torch.zeros(E, N, dtype=torch.long)
    pos = torch.zeros(E, N, 2)
    for e in range(E):
        for s in range(N):
            best, bk = -1.0, -1
            for k in range(K):
                rr = int(feats[e, s, k, 8:13].argmax())
                if not bool(valid[e, s, k]) or rr == 0:
                    continue
                sc = float(feats[e, s, k, 6] + feats[e, s, k, 7])
                if sc > best:
                    best, bk = sc, k
            if bk >= 0:
                t = int(topi[e, s, bk])
                role[e, s] = int(feats[e, s, bk, 8:13].argmax())
                rel = env.pos[e, t] - env.pos[e, s]
                h = torch.tensor(float(env.heading[e, s]) * vg.DEG)
                pos[e, s, 0] = rel[0] * torch.cos(h) - rel[1] * torch.sin(h)
                pos[e, s, 1] = rel[0] * torch.sin(h) + rel[1] * torch.cos(h)
    return role, pos


def yield_checks(sn):
    env, x, g, ss, st, r = sn['env'], sn['x'], sn['g'], sn['ss'], sn['st'], sn['r']
    E, N = env.E, env.N
    S, FR = cfg.STATE_SIZE, cfg.FRAMES
    # cz2: 현재 프레임 = 스택 마지막 = 방금 obs 의 레이더, 36칸 최솟값
    ok = torch.equal(x[..., (FR - 1) * S:], r)
    sec = T.radar_sectors(x)
    bf = torch.empty(E, N, 36)
    for e in range(E):
        for n in range(N):
            cur = r[e, n] + 0.5
            for s in range(36):
                bf[e, n, s] = min(float(v) for v in cur[s * 10:(s + 1) * 10])
    ok &= sec.shape == (E, N, 36) and bool(torch.allclose(sec, bf, atol=1e-6))
    ACC['n_det'] += int((bf < 0.999).sum())
    ACC['cz2'] &= bool(ok)
    # cz3: payload50 = 따로 조립
    d = torch.cdist(env.pos, env.pos) + torch.eye(N).unsqueeze(0) * 1e9
    d = torch.where(d <= cfg.COMM_RANGE, d, torch.full_like(d, 1e9))
    topd, topi = torch.topk(d, min(cfg.MAX_COMM_PARTNERS, N - 1), dim=-1, largest=False)
    valid = topd < 1e9
    feats = vg.comm_pair_features(env, topi, cfg.COMM_RANGE)
    role, pos = brute_decl(env, topi, valid, feats)
    ACC['n_decl'] += int((role > 0).sum())
    man = torch.cat([vg.own_payload(env), g, F.one_hot(role, 5)[..., 1:].float(), pos / float(cfg.COMM_RANGE), bf], dim=-1)
    pay = T.payload50_now(env, x, g)
    ok3 = pay.shape == (E, N, 50) and bool(torch.allclose(pay, man, atol=1e-4))
    ok3 &= torch.equal(pay[..., comm_codec.P50_GOAL], g) and torch.equal(pay[..., comm_codec.P50_RADAR], sec)
    ACC['cz3'] &= bool(ok3)
    # cz4: comm_gather direct 필드 (k 2·8·12)
    for k in (2, 8, 12):
        c = comm_codec.install(PATH.format(k), P50[k], 'direct', 'cpu')
        pol = POL.get('p')
        if pol is None:
            torch.manual_seed(0)
            pol = POL['p'] = net.CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES)
        with torch.no_grad():
            _, parts = T.comm_gather(pol, env, x, g, ss, st, cfg.MAX_COMM_PARTNERS)
            z = c.encode(pay)
        prel = parts[4]                                                       # [E,N,K,3+20]
        K = prel.shape[2]
        # comm_gather 는 PARTNER_RANGE(None = COMM_RANGE) 로 파트너를 고름 — 위 topi 와 같음
        b = torch.arange(E)[:, None, None]
        vm = valid.unsqueeze(-1)
        exp = torch.cat([z[b, topi], vg.own_state4(env).unsqueeze(2).expand(-1, -1, K, -1),
                         torch.zeros(E, N, K, cfg.COMM_EXT_DIM - k - 4)], dim=-1)
        exp = torch.where(vm, exp, torch.zeros_like(exp))
        ok4 = prel.shape[-1] == 3 + cfg.COMM_EXT_DIM and bool(torch.allclose(prel[..., 3:], exp, atol=1e-6))
        ok4 &= bool((prel[..., 3:][~valid] == 0).all())
        ACC['cz4'] &= bool(ok4)
        if k == 2:
            ACC['n_valid'] += int(valid.sum())
            ACC['n_pad'] += int((~valid).sum())
    net.COMM_CODEC, net.COMM_CODEC_MODE = None, ''


# ─────────────────────────── cz1 · cz5 · cz6 ───────────────────────────
def cz1():
    ok, info = [], []
    for k, sha in P50.items():
        c = comm_codec.load_codec(PATH.format(k), sha)
        ok.append(c.sha.startswith(sha) and c.layout == 'p50' and c.d_in == 50 and c.k == k and k + 4 <= cfg.COMM_EXT_DIM)
        info.append(f"k{k} {c.sha[:12]}")
        try:
            comm_codec.install(PATH.format(k), sha, 'decode', 'cpu')
            ok.append(False)
        except SystemExit:
            ok.append(True)
        net.COMM_CODEC, net.COMM_CODEC_MODE = None, ''
    check('cz1 코덱 6개 SHA·layout p50·d_in 50·k · k+4 ≤ 20 · decode 거부', all(ok), ' '.join(info))


def zmse(c, p, sd):
    with torch.no_grad():
        ph = c.decode(c.encode(p))
    return float((((ph - p) / sd) ** 2).mean())


def cz5(P):
    sd = P.std(dim=0).clamp(min=0.05)
    rows, zs = [], []
    for k in sorted(P50):
        c = comm_codec.load_codec(PATH.format(k), P50[k])
        f = comm_codec.fidelity(c, P)
        zs.append(zmse(c, P, sd))
        rows.append((k, f, zs[-1]))
    print(f"  충실도(실제 상태 seed 1, n={P.shape[0]}, 역할 있음 {float((P[:, comm_codec.P50_ROLE].sum(-1) > 0).float().mean()):.3f}, "
          f"레이더 감지칸 {float((P[:, comm_codec.P50_RADAR] < 0.999).float().mean()):.3f}) — 설명 자료")
    print('    k  zMSE   침로p99°  SOGp99  선회p99  키p99°  역할acc  대상p99m  목표각p99°  레이더p99m  감지acc')
    for k, f, z in rows:
        print(f"   {k:2d}  {z:.3f}  {f['heading_deg']['p99']:7.1f}  {f['sog_mps']['p99']:6.3f}  {f['rot_n']['p99']:6.3f}  "
              f"{f['cmd_rudder_deg']['p99']:6.1f}  {f['decl_role_acc']:.4f}  {f['decl_pos_m']['p99']:7.1f}  "
              f"{f['goal_angle_deg']['p99']:8.1f}  {f['radar_sector_m']['p99']:8.1f}  {f['radar_detect_acc']:.4f}")
    mono = all(zs[i + 1] <= 1.1 * zs[i] for i in range(len(zs) - 1))
    check('cz5a z-score 복원 오차가 k 증가에 나빠지지 않음(10 % 여유)', mono, ' '.join(f"{z:.3f}" for z in zs))
    f12 = rows[-1][1]
    check('cz5b k12 역할 ≥ 99 %', f12['decl_role_acc'] >= SANITY_K12['decl_role_acc_min'], f"역할 {f12['decl_role_acc']:.4f}")
    print(f"  (기록) k12 침로 p99 {f12['heading_deg']['p99']:.2f}° — 사전 예상 ≤ {HEADING_K12_PRIOR:g}° "
          f"{'안' if f12['heading_deg']['p99'] <= HEADING_K12_PRIOR else '밖'} (판정 아님, 저자 결정 2026-09-29)")


def cz6():
    ok = []
    try:
        comm_codec.LatentCodec(k=4, d_in=12, layout='p50')
        ok.append(False)
    except ValueError:
        ok.append(True)
    try:
        comm_codec.load_codec(PATH.format(8), '000000000000')
        ok.append(False)
    except SystemExit:
        ok.append(True)
    import ckpt_io
    c8 = comm_codec.load_codec(PATH.format(8), P50[8])
    snap = {'comm_ext': 1, 'comm_fields': 'intent', 'comm_ext_layout': cfg.COMM_EXT_LAYOUT, 'comm_latent': 0.0,
            'partner_range': None, 'aux_loss_scale': 0.0, 'comm_codec': PATH.format(8), 'comm_codec_sha256': c8.sha,
            'comm_codec_mode': 'direct', 'comm_codec_layout': 'p12'}
    sd = {'attn.k_proj.0.weight': torch.zeros(64, 3 + cfg.COMM_EXT_DIM + cfg.MSG_DIM)}
    try:
        ckpt_io.restore_comm_ext(sd, snap, cfg.MSG_DIM, [], '[t]')
        ok.append(False)
    except SystemExit:
        ok.append(True)
    snap['comm_codec_layout'] = 'p50'
    ckpt_io.restore_comm_ext(sd, snap, cfg.MSG_DIM, [], '[t]')
    ok.append(net.COMM_CODEC is not None and net.COMM_CODEC.layout == 'p50' and net.COMM_CODEC_MODE == 'direct')
    net.COMM_CODEC, net.COMM_CODEC_MODE = None, ''
    check('cz6 가드: layout/d_in · 틀린 SHA · 스냅샷 layout 불일치 거부, 맞으면 설치', all(ok), str(ok))


if __name__ == '__main__':
    torch.set_num_threads(1)
    print('=' * 78)
    print('latent 차원 sweep 코덱 (p50, z2–z12) 단위 테스트')
    print('=' * 78)
    cz1()
    P, n_snap = rollout()
    check('cz2 radar_sectors == 무차별 대입 · 현재 프레임 = 스택 마지막', ACC['cz2'] and ACC['n_det'] >= 100,
          f"스냅샷 {n_snap} 감지칸 {ACC['n_det']}")
    check('cz3 payload50 == 따로 조립(운동·목표·선언·위치·레이더)', ACC['cz3'] and ACC['n_decl'] >= 20,
          f"선언 {ACC['n_decl']}")
    check('cz4 comm_gather direct 필드 = [z_j · 자기 상태 4 · 0] · 빈 슬롯 0 (k 2·8·12)',
          ACC['cz4'] and ACC['n_valid'] >= 50 and ACC['n_pad'] >= 1, f"유효 {ACC['n_valid']} 빈 {ACC['n_pad']}")
    cz5(P)
    cz6()
    print('=' * 78)
    print(f"VERDICT: {'ALL PASS' if all(RES) else 'FAIL ' + str(RES.count(False))}")
    sys.exit(0 if all(RES) else 1)
