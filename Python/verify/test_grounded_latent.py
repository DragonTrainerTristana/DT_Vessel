"""test_grounded_latent.py — grounded latent 코덱 + 원거리 COLREGs 보상 (2026-09-28) 단위 테스트.

  python verify/test_grounded_latent.py        # ALL PASS 여야 함 (imo·none 환경을 여기서 고정)

1. comm_pair_features(sender=참값) 가 sender=None 과 비트동일 (송신자 값 대입 경로가 같은 연산)
2. 코덱: 로드 SHA 가드·프로필 가드·양자화 결정론·no-grad/정책 밖·복원 충실도, decode 필드 ≈ 참 필드
3. 원거리 COLREGs: 끄면 far_sit=None, 켜면 상태 궤적 불변(보상만 다름)이고 보상 차이는 원거리 채점 대상 배에만
4. ckpt_io.restore_comm_ext: 스냅샷에 코덱 있으면 설치, 없으면 None 으로 리셋(앞 체크포인트 값 누출 없음)
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

CODEC = 'comm_codecs/p6_k6_s0.pt'
SHA = 'fbe4c71a6bf4'
RES = []


def check(name, ok, info=''):
    RES.append(bool(ok))
    print(f"  {'PASS' if ok else '★FAIL'}  {name}  {info}")


def make_env(E=8, N=16, seed=5):
    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device='cpu', seed=seed, ring_scale=1.0, crossing=0,
                            risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE,
                            farfield_coef=0.0, perpair_coef=-0.15, perpair_exp=3.0)
    env.reset()
    return env


def rand_actions(E, N, g):
    return torch.rand(E, N, 2, generator=g) * 2 - 1


def topk_partners(env, K=4):
    d = torch.cdist(env.pos, env.pos) + torch.eye(env.N).unsqueeze(0) * 1e9
    d = torch.where(d <= cfg.COMM_RANGE, d, torch.full_like(d, 1e9))
    return torch.topk(d, K, dim=-1, largest=False)


def t1_sender_identity():
    env = make_env()
    g = torch.Generator().manual_seed(1)
    for _ in range(40):
        env.step(rand_actions(env.E, env.N, g))
    _, topi = topk_partners(env)
    f0 = vg.comm_pair_features(env, topi, cfg.COMM_RANGE)
    snd = {'heading': env.heading, 'speed': env.speed,
           'rot': vg.yaw_rate_deg(env.rudder, env.speed, env.max_speed) / vg.MAX_YAW_RATE,
           'cmd_rudder': env.cmd_rudder, 'target_speed': env.target_speed}
    f1 = vg.comm_pair_features(env, topi, cfg.COMM_RANGE, sender=snd)
    check('1 sender=참값 == sender=None (비트동일)', torch.equal(f0, f1), f"max|d|={float((f0 - f1).abs().max()):.1e}")
    return env, topi, f0


def t2_codec(env, topi, f0):
    c = comm_codec.load_codec(CODEC, SHA)
    check('2a 코덱 로드·SHA 일치', c.sha.startswith(SHA), c.sha[:12])
    try:
        comm_codec.load_codec(CODEC, '000000000000')
        check('2b 틀린 SHA 거부', False)
    except SystemExit:
        check('2b 틀린 SHA 거부', True)
    check('2c 파라미터 requires_grad 전부 False', all(not p.requires_grad for p in c.parameters()))
    p = vg.own_payload(env)
    z1, z2 = c.encode(p), c.encode(p)
    lv = 2 ** c.bits - 1
    on_grid = torch.allclose((z1 + 1) * 0.5 * lv, torch.round((z1 + 1) * 0.5 * lv), atol=1e-4)
    check('2d 양자화 결정론·격자 위', torch.equal(z1, z2) and on_grid and z1.shape[-1] == 6, f"shape={tuple(z1.shape)}")
    fid = comm_codec.fidelity(c, p.reshape(-1, 6))
    check('2e 실제 env 상태 복원 충실도 (heading p99 < 3°, SOG p99 < 0.08 m/s)',
          fid['heading_deg']['p99'] < 3.0 and fid['sog_mps']['p99'] < 0.08,
          f"heading p50/p99 {fid['heading_deg']['p50']:.2f}/{fid['heading_deg']['p99']:.2f}° "
          f"sog p99 {fid['sog_mps']['p99']:.3f}")
    snd = comm_codec.decode_sender(c.decode(z1))
    f1 = vg.comm_pair_features(env, topi, cfg.COMM_RANGE, sender=snd)
    role_t = f0[..., 8:18].argmax(-1)
    role_h = f1[..., 8:18].argmax(-1)
    agree = float((role_t == role_h).float().mean())
    dk = (f1[..., 0:6] - f0[..., 0:6]).abs()                # 운동 성분(침로차 sin/cos·SOG·ROT·상대속도)
    dr = (f1[..., 6:8] - f0[..., 6:8]).abs()                # dcpa/tcpa 위험(문턱·접근 여부에 민감 — 서술만)
    q99 = lambda t: float(t.flatten().quantile(0.99))
    check('2f decode 필드: 역할 일치율 ≥ 0.97, 운동 성분 p99 오차 < 0.05', agree >= 0.97 and q99(dk) < 0.05,
          f"role agree={agree:.3f} kin p99={q99(dk):.4f} max={float(dk.max()):.3f} | risk p99={q99(dr):.3f} max={float(dr.max()):.3f}")
    try:
        os.environ['VESSEL_DYN_PROFILE'] = 'imo'
        _old = cfg.DYN_PROFILE
        cfg.DYN_PROFILE = 'agile'
        try:
            comm_codec.load_codec(CODEC, SHA)
            ok = False
        except SystemExit:
            ok = True
        cfg.DYN_PROFILE = _old
        check('2g 다른 동역학 프로필에서 로드 거부', ok)
    finally:
        pass


def t3_far_reward():
    E, N = 8, 16
    envA, envB = make_env(E, N, 7), make_env(E, N, 7)
    gA, gB = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    vg.COLREGS_FAR_RANGE = 0.0
    envA.step(rand_actions(E, N, gA))
    envB.step(rand_actions(E, N, gB))                      # 두 env 를 같은 행동·같은 상태로 맞춰 둠
    check('3a 끄면 far_sit=None', envA._last_pw['far_sit'] is None)
    same_state, diff_outside, n_used, n_diff = True, 0.0, 0, 0
    for _ in range(120):
        aA, aB = rand_actions(E, N, gA), rand_actions(E, N, gB)
        vg.COLREGS_FAR_RANGE = 0.0
        _, rA, dA, _ = envA.step(aA)
        vg.COLREGS_FAR_RANGE = 300.0
        _, rB, dB, _ = envB.step(aB)
        same_state &= torch.equal(envA.pos, envB.pos) and torch.equal(envA.heading, envB.heading)
        uf = (envB.situation == 0) & (envB.far_situation > 0) & (envB.far_max_risk > vg.COLREGS_RISK_GATE)
        live = ~(dA.bool() | dB.bool())
        m_out = live & ~uf
        if m_out.any():
            diff_outside = max(diff_outside, float((rA - rB)[m_out].abs().max()))
        n_used += int((live & uf).sum())
        n_diff += int(((rA - rB).abs() > 1e-6)[live & uf].sum())
    vg.COLREGS_FAR_RANGE = 0.0
    check('3b 켜도 상태 궤적 불변(보상만 다름)', same_state)
    check('3c 원거리 채점 대상 아닌 배는 보상 동일', diff_outside < 1e-5, f"max|d|={diff_outside:.1e}")
    check('3d 원거리 채점이 실제로 발화', n_used > 0 and n_diff > 0, f"대상 {n_used} 배·결정, 보상 달라진 {n_diff}")


def t4_restore():
    import ckpt_io
    notes = []
    snap = {'comm_ext': 1, 'comm_fields': 'intent', 'comm_ext_layout': cfg.COMM_EXT_LAYOUT, 'comm_latent': 0.0,
            'partner_range': None, 'aux_loss_scale': 0.0,
            'comm_codec': CODEC, 'comm_codec_sha256': comm_codec.load_codec(CODEC, SHA).sha, 'comm_codec_mode': 'direct'}
    tok = 3 + cfg.COMM_EXT_DIM + cfg.MSG_DIM
    sd = {'attn.k_proj.0.weight': torch.zeros(64, tok)}
    ckpt_io.restore_comm_ext(sd, snap, cfg.MSG_DIM, notes, '[t]')
    ok1 = net.COMM_CODEC is not None and net.COMM_CODEC_MODE == 'direct'
    snap2 = dict(snap, comm_codec='', comm_codec_sha256='', comm_codec_mode='')
    ckpt_io.restore_comm_ext(sd, snap2, cfg.MSG_DIM, notes, '[t]')
    ok2 = net.COMM_CODEC is None and net.COMM_CODEC_MODE == ''
    check('4 restore: 코덱 스냅샷 → 설치 / 코덱 없음 → None 리셋', ok1 and ok2)


if __name__ == '__main__':
    print('=' * 78)
    print('grounded latent 코덱 + 원거리 COLREGs 단위 테스트')
    print('=' * 78)
    env, topi, f0 = t1_sender_identity()
    t2_codec(env, topi, f0)
    t3_far_reward()
    t4_restore()
    print('=' * 78)
    print(f"VERDICT: {'ALL PASS' if all(RES) else 'FAIL ' + str(RES.count(False))}")
    sys.exit(0 if all(RES) else 1)
