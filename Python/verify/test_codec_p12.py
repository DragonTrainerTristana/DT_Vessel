"""test_codec_p12.py — 역할 선언 latent(코덱 레이아웃 p12, 2026-09-29) 단위 테스트. G3 + 기능 게이트(스펙 §3·§5).

  python verify/test_codec_p12.py        # ALL PASS 여야 함 (imo·none 고정)

cp1 p12 파일 SHA = 고정값 · p6 로드·SHA·meta dict 는 이전 그대로
cp2 송신 선언(대상·역할·위치) == 무차별 대입 참값 (실제 롤아웃 상태)
cp3 코덱 통과 후 선언 역할 복원 ≥ 99.5 % (실제 상태) · 합성 holdout 역할 정확도 ≥ 99.5 %      ← 기능 게이트 ①
cp4 선언이 대상 배에만 도착: 미탐 ≤ 1 % · 오탐(대상 아닌 배가 받음) ≤ 2 %, 선언 단위             ← 기능 게이트 ②
cp5 [13:18] 을 뺀 필드는 같은 복원 운동값의 decode 경로와 같음 · 받는 배 역할 추정 일치율 ≥ 0.97   ← 기능 게이트 ③
cp6 가드: p12+direct 거부 · layout/d_in 불일치 거부 · 틀린 SHA 거부 · 스냅샷 layout 불일치 거부
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
sys.path.insert(0, PYROOT)
for _k in [k for k in os.environ if k.startswith('VESSEL_')]:
    os.environ.pop(_k)
os.environ.update({'VESSEL_DYN_PROFILE': 'imo', 'VESSEL_OBSTACLES': 'none', 'VESSEL_COMM_EXT': '1',
                   'VESSEL_USE_ATTENTION': '1'})

import torch  # noqa: E402
import config as cfg  # noqa: E402
import vessel_gym as vg  # noqa: E402
import networks as net  # noqa: E402
import comm_codec  # noqa: E402

CODEC12, SHA12 = 'comm_codecs/p12_k8_s0.pt', 'ed43ecc4d2a3'
CODEC6, SHA6 = 'comm_codecs/p6_k6_s0.pt', 'fbe4c71a6bf4'
G = comm_codec.P12_FUNC_GATE
RES = []


def check(name, ok, info=''):
    RES.append(bool(ok))
    print(f"  {'PASS' if ok else '★FAIL'}  {name}  {info}")


def make_env(E=4, N=16, seed=13):
    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device='cpu', seed=seed, ring_scale=1.0, crossing=0,
                            risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE,
                            farfield_coef=0.0, perpair_coef=-0.15, perpair_exp=3.0)
    env.reset()
    return env


def partners(env, K=4):
    """comm_gather 와 같은 파트너 선택(자기 제외·반경 밖 제외·거리순 top-K)."""
    d = torch.cdist(env.pos, env.pos) + torch.eye(env.N).unsqueeze(0) * 1e9
    d = torch.where(d <= cfg.COMM_RANGE, d, torch.full_like(d, 1e9))
    topd, topi = torch.topk(d, K, dim=-1, largest=False)
    return topd, topi, topd < 1e9


def rule_policy(env):
    tg = env.goal - env.pos
    err = vg._wrap180(torch.atan2(tg[..., 0], tg[..., 1]) / vg.DEG - env.heading)
    a0 = (err / 20.0).clamp(-1, 1)
    give = (env.situation == 1) | (env.situation == 3) | (env.situation == 4)
    a0 = torch.where(give, torch.full_like(a0, 0.6), a0)
    return torch.stack([a0, torch.ones_like(a0)], dim=-1)


def states(n_snap=16, every=25, burn=150):
    env = make_env()
    out = []
    for t in range(burn + n_snap * every):
        env.step(rule_policy(env))
        if t >= burn and (t - burn) % every == 0:
            out.append(env)
            snap = make_env()
            for a in ('pos', 'heading', 'speed', 'rudder', 'cmd_rudder', 'target_speed', 'max_speed', 'goal'):
                setattr(snap, a, getattr(env, a).clone())
            snap._update_situation()
            out[-1] = snap
    return out


def cp1():
    c12 = comm_codec.load_codec(CODEC12, SHA12)
    c6 = comm_codec.load_codec(CODEC6, SHA6)
    meta6 = comm_codec.LatentCodec(k=6).meta()
    ok_meta = list(meta6) == ['layout', 'k', 'hidden', 'bits', 'd_in'] and meta6['layout'] == 'p6'
    check('cp1 p12 SHA 고정값 · p6 로드·SHA·meta 이전 그대로',
          c12.sha.startswith(SHA12) and c12.layout == 'p12' and c12.d_in == 12 and c12.k == 8
          and c6.sha.startswith(SHA6) and c6.layout == 'p6' and ok_meta, f"p12 {c12.sha[:12]} p6 {c6.sha[:12]}")
    return c12


def brute_decl(env, topi, valid, feats):
    E, N, K = topi.shape
    role = torch.zeros(E, N, dtype=torch.long)
    pos = torch.zeros(E, N, 2)
    for e in range(E):
        for s in range(N):
            best, bk = -1.0, -1
            for k in range(K):
                r = int(feats[e, s, k, 8:13].argmax())
                if not bool(valid[e, s, k]) or r == 0:
                    continue
                sc = float(feats[e, s, k, 6] + feats[e, s, k, 7])
                if sc > best:
                    best, bk = sc, k
            if bk >= 0:
                t = int(topi[e, s, bk])
                role[e, s] = int(feats[e, s, bk, 8:13].argmax())
                rel = env.pos[e, t] - env.pos[e, s]
                h = float(env.heading[e, s]) * vg.DEG
                pos[e, s, 0] = rel[0] * torch.cos(torch.tensor(h)) - rel[1] * torch.sin(torch.tensor(h))
                pos[e, s, 1] = rel[0] * torch.sin(torch.tensor(h)) + rel[1] * torch.cos(torch.tensor(h))
    return role, pos


def cp2_to_cp5(c12, snaps):
    n_decl = n_role_ok = n_miss = n_tgt_seen = n_false_decl = 0
    brute_ok, other_ok, agree_n, agree_ok = True, True, 0, 0
    for env in snaps:
        topd, topi, valid = partners(env)
        feats = vg.comm_pair_features(env, topi, cfg.COMM_RANGE)
        role, pos = vg.role_declaration(env, topi, valid, feats)
        brole, bpos = brute_decl(env, topi, valid, feats)
        brute_ok &= torch.equal(role, brole) and bool(torch.allclose(pos, bpos, atol=1e-3))
        with torch.no_grad():
            p12 = vg.own_payload12(env, role, pos)
            snd = comm_codec.decode_sender(c12.decode(c12.encode(p12)))
        has = role > 0
        n_decl += int(has.sum())
        n_role_ok += int((snd['decl_role'] == role)[has].sum())
        base = vg.comm_pair_features(env, topi, cfg.COMM_RANGE, sender=snd)
        ext = vg.apply_declaration(env, topi, base, snd)
        other_ok &= torch.equal(ext[..., :13], base[..., :13]) and torch.equal(ext[..., 18:], base[..., 18:])
        # 받는 배 역할 추정(복원 운동값) vs 참: 내 역할 [8:13]
        mr_t, mr_h = feats[..., 8:13].argmax(-1), base[..., 8:13].argmax(-1)
        agree_n += int(valid.sum()); agree_ok += int((mr_t == mr_h)[valid].sum())
        # 선언 단위 도착: 송신 s 의 대상 t. t 가 s 를 파트너로 가지면 t 쪽 슬롯이 선언 역할이어야 함(미탐),
        #   s 를 파트너로 가진 다른 수신자 슬롯은 없음이어야 함(오탐)
        got = ext[..., 13:18].argmax(-1)                                    # [E,N,K] 수신자가 받은 선언 역할
        E, N, K = topi.shape
        tgt = torch.full((E, N), -1, dtype=torch.long)
        for e in range(E):
            for s in range(N):
                if int(role[e, s]) == 0:
                    continue
                rel = env.pos[e] - env.pos[e, s]
                h = float(env.heading[e, s]) * vg.DEG
                stb = rel[:, 0] * torch.cos(torch.tensor(h)) - rel[:, 1] * torch.sin(torch.tensor(h))
                fwd = rel[:, 0] * torch.sin(torch.tensor(h)) + rel[:, 1] * torch.cos(torch.tensor(h))
                d_ = (stb - pos[e, s, 0]) ** 2 + (fwd - pos[e, s, 1]) ** 2
                d_[s] = 1e18
                t = int(d_.argmin())
                false_hit = False
                for i in range(N):
                    sl = (topi[e, i] == s) & valid[e, i]
                    if not bool(sl.any()):
                        continue
                    k = int(sl.nonzero()[0])
                    rcv = int(got[e, i, k])
                    if i == t:
                        n_tgt_seen += 1
                        n_miss += int(rcv != int(role[e, s]))
                    elif rcv != 0:
                        false_hit = True
                n_false_decl += int(false_hit)
    acc_real = n_role_ok / max(n_decl, 1)
    fid = torch.load(os.path.join(PYROOT, CODEC12), map_location='cpu')['fidelity_holdout']
    miss, fals = n_miss / max(n_tgt_seen, 1), n_false_decl / max(n_decl, 1)
    agree = agree_ok / max(agree_n, 1)
    check('cp2 송신 선언(대상·역할·위치) == 무차별 대입 참값', brute_ok and n_decl >= 50, f"선언 {n_decl}")
    check('cp3 선언 역할 복원 ≥ 99.5 % (실제 상태·합성 holdout) [기능 게이트 ①]',
          acc_real >= G['decl_role_acc_min'] and fid['decl_role_acc'] >= G['decl_role_acc_min'],
          f"실제 {acc_real:.4f} (n={n_decl}) holdout {fid['decl_role_acc']:.4f}")
    check('cp4 선언이 대상에만 도착: 미탐 ≤ 1 % · 오탐 ≤ 2 % [기능 게이트 ②]',
          miss <= G['match_miss_max'] and fals <= G['match_false_max'] and n_tgt_seen >= 50,
          f"미탐 {miss:.4f} (대상 {n_tgt_seen}) 오탐 {fals:.4f} (선언 {n_decl})")
    check('cp5 [13:18] 밖 필드 = decode 경로 · 받는 배 역할 추정 일치 ≥ 0.97 [기능 게이트 ③]',
          other_ok and agree >= G['recv_role_agree_min'], f"일치 {agree:.4f} (n={agree_n})")


def cp6(c12):
    ok = []
    try:
        comm_codec.install(CODEC12, SHA12, 'direct', 'cpu')
        ok.append(False)
    except SystemExit:
        ok.append(True)
    try:
        comm_codec.LatentCodec(k=8, d_in=6, layout='p12')
        ok.append(False)
    except ValueError:
        ok.append(True)
    try:
        comm_codec.load_codec(CODEC12, '000000000000')
        ok.append(False)
    except SystemExit:
        ok.append(True)
    import ckpt_io
    snap = {'comm_ext': 1, 'comm_fields': 'intent', 'comm_ext_layout': cfg.COMM_EXT_LAYOUT, 'comm_latent': 0.0,
            'partner_range': None, 'aux_loss_scale': 0.0, 'comm_codec': CODEC12, 'comm_codec_sha256': c12.sha,
            'comm_codec_mode': 'decode', 'comm_codec_layout': 'p6'}
    sd = {'attn.k_proj.0.weight': torch.zeros(64, 3 + cfg.COMM_EXT_DIM + cfg.MSG_DIM)}
    try:
        ckpt_io.restore_comm_ext(sd, snap, cfg.MSG_DIM, [], '[t]')
        ok.append(False)
    except SystemExit:
        ok.append(True)
    snap['comm_codec_layout'] = 'p12'
    ckpt_io.restore_comm_ext(sd, snap, cfg.MSG_DIM, [], '[t]')
    ok.append(net.COMM_CODEC is not None and net.COMM_CODEC.layout == 'p12' and net.COMM_CODEC_MODE == 'decode')
    net.COMM_CODEC, net.COMM_CODEC_MODE = None, ''
    check('cp6 가드: p12+direct · layout/d_in · 틀린 SHA · 스냅샷 layout 불일치 거부, 맞으면 설치', all(ok), str(ok))


if __name__ == '__main__':
    torch.set_num_threads(1)
    print('=' * 78)
    print('역할 선언 latent (p12) 단위 테스트 + 기능 게이트')
    print('=' * 78)
    c = cp1()
    cp2_to_cp5(c, states())
    cp6(c)
    print('=' * 78)
    print(f"VERDICT: {'ALL PASS' if all(RES) else 'FAIL ' + str(RES.count(False))}")
    sys.exit(0 if all(RES) else 1)
