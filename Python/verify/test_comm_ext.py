"""
의도·역할 통신(COMM_EXT, 2026-09-25) 테스트. pytest 없이도 `python3 verify/test_comm_ext.py` 로 돈다.
스펙: docs/superpowers/specs/2026-09-25-comm-intent-design.md. numpy 금지(Mac torch↔numpy 비호환).

검사 대상 = 저자 의도가 코드에 그대로 실리는가:
  좌표·침로·속력(state) / 내 역할·상대가 선언한 역할(role) / 상대 명령 타각·속력(intent) 이 정의대로 계산되고,
  rollout 과 update 가 같은 값을 쓰며(미러), 체크포인트가 그 설정을 잃지 않는다.
config 는 import 시점에 env 를 읽으므로, 여기서는 ckpt_io 처럼 networks 모듈 전역을 바꿔 EXT 정책을 만든다.
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
# ★config 는 import 시점에 env 를 읽는다. preflight 는 배치 env(VESSEL_COMM_EXT=1 등)를 export 한 채 이 테스트를 부르므로
#   import 전에 통신 토글을 기본값으로 고정한다(_verify_ppo_mirror 와 같은 방식). EXT 구성은 아래에서 networks 전역으로 만든다.
os.environ['VESSEL_COMM_EXT'] = '0'
os.environ['VESSEL_COMM_FIELDS'] = 'latent'
for _k in ('VESSEL_COMM_LATENT', 'VESSEL_PARTNER_RANGE', 'VESSEL_AUX_LOSS_SCALE'):
    os.environ.pop(_k, None)
import torch  # noqa: E402

import config as cfg  # noqa: E402
import networks as net  # noqa: E402
import vessel_gym as vg  # noqa: E402
import vessel_gym_train as T  # noqa: E402
import ckpt_io  # noqa: E402

_KEYS = ('COMM_EXT', 'COMM_FIELDS', 'COMM_GROUPS', 'COMM_LATENT', 'PARTNER_RANGE', 'USE_ATTENTION')


def _save_globals():
    return {k: getattr(net, k) for k in _KEYS}


def _load_globals(g):
    for k, v in g.items():
        setattr(net, k, v)


def _set_ext(fields='intent', latent=1.0, partner=None):
    net.COMM_EXT = True
    net.USE_ATTENTION = True
    net.COMM_FIELDS = fields
    net.COMM_GROUPS = tuple(cfg.COMM_FIELDS_TO_GROUPS[fields])
    net.COMM_LATENT = float(latent)
    net.PARTNER_RANGE = partner


def _policy(seed=0):
    torch.manual_seed(seed)
    return net.CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES)


def _env(E=1, N=2, seed=3):
    return vg.VesselBatchEnv(num_envs=E, n_vessels=N, device='cpu', seed=seed, crossing=2,
                             risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE)


def _place(env, pos, hdg_deg, spd, vmax=1.8):
    """env 상태를 직접 놓는다. pos [E,N,2] · hdg [E,N] deg · spd [E,N]."""
    env.pos = torch.tensor(pos, dtype=env.dtype)
    env.heading = torch.tensor(hdg_deg, dtype=env.dtype)
    env.speed = torch.tensor(spd, dtype=env.dtype)
    env.max_speed = torch.full_like(env.speed, vmax)
    env.rudder = torch.zeros_like(env.speed)
    env.cmd_rudder = torch.zeros_like(env.speed)
    env.target_speed = env.speed.clone()


def _obs(env, E, N):
    fs = T.FrameStack(E, N, 'cpu')
    obs = env._build_obs()
    radar, goal, self_s, sit = T.parse_obs(obs)
    fs.reset_all(radar)
    return fs.get(), goal, self_s, sit


# ─────────────────────────── 구조 ───────────────────────────
def test_defaults_are_off_and_bit_identical_structure():
    assert cfg.COMM_EXT is False and cfg.COMM_FIELDS == 'latent' and cfg.COMM_LATENT == 1.0
    assert cfg.PARTNER_RANGE is None and cfg.AUX_LOSS_SCALE == 1.0
    assert net.COMM_GROUPS == ()
    pol = _policy()
    sd = pol.state_dict()
    assert pol.relpos_dim == 3
    assert 'attn.k_proj.weight' in sd and not any(k.startswith('attn.k_proj.0') for k in sd)
    assert tuple(sd['attn.k_proj.weight'].shape) == (cfg.ATTN_DIM, 3 + cfg.MSG_DIM)


def test_ext_structure_and_small_init():
    g = _save_globals()
    try:
        _set_ext()
        pol = _policy()
        sd = pol.state_dict()
        assert pol.relpos_dim == 3 + cfg.COMM_EXT_DIM == 23
        tok = 3 + cfg.COMM_EXT_DIM + cfg.MSG_DIM
        assert tuple(sd['attn.k_proj.0.weight'].shape) == (net.COMM_EXT_MLP_HIDDEN, tok)
        assert tuple(sd['attn.v_proj.2.weight'].shape) == (cfg.MSG_DIM, net.COMM_EXT_MLP_HIDDEN)
        assert float(sd['attn.v_proj.2.bias'].abs().max()) == 0.0
        bound = 1.0 / math.sqrt(net.COMM_EXT_MLP_HIDDEN)          # nn.Linear 기본 init 상한
        assert float(sd['attn.v_proj.2.weight'].abs().max()) <= 0.1 * bound + 1e-7
        assert tuple(sd['msg_encoder.0.weight'].shape)[1] == 3 + cfg.COMM_EXT_DIM + cfg.MSG_DIM
    finally:
        _load_globals(g)


def test_ext_requires_attention():
    g = _save_globals()
    try:
        _set_ext()
        net.USE_ATTENTION = False
        try:
            _policy()
        except RuntimeError as e:
            assert 'USE_ATTENTION' in str(e)
        else:
            raise AssertionError('EXT + attention off 가 조용히 통과함')
    finally:
        _load_globals(g)


# ─────────────────────────── 필드 정의 (저자 의도) ───────────────────────────
def test_role_cascade_matches_pairwise_dense():
    """게이트를 끄고 반경 56 m 로 부르면 역할 = env._pairwise()['sit'] (전 쌍). 조우가 실제로 많아야 의미 있음."""
    E, N = 2, 16
    env = _env(E, N, seed=5)
    torch.manual_seed(11)
    _place(env, (torch.rand(E, N, 2) * 90 - 45).tolist(), (torch.rand(E, N) * 360).tolist(),
           (0.3 + torch.rand(E, N) * 1.5).tolist())
    pw = env._pairwise()
    b = torch.arange(E)[:, None, None]
    ii = torch.arange(N)[None, :, None]
    dmat = torch.cdist(env.pos, env.pos) + torch.eye(N)[None] * 1e9
    _, topi = torch.topk(dmat, N - 1, dim=-1, largest=False)            # 자기 제외 전원
    feat = vg.comm_pair_features(env, topi, vg.DETECTION_RANGE, role_gate=False)
    role = feat[..., 8:13].argmax(-1)
    ref = pw['sit'][b, ii, topi]
    nz = int((ref != 0).sum())
    assert nz >= 40, f'조우 쌍이 너무 적어 검사가 무의미함 ({nz})'
    assert bool((role == ref).all()), f'불일치 {int((role != ref).sum())} / {role.numel()}'
    # 상대가 선언한 역할(j→i) = 전치 위치의 _pairwise 역할
    their = feat[..., 13:18].argmax(-1)
    ref_t = pw['sit'].transpose(1, 2)[b, ii, topi]
    assert bool((their == ref_t).all())


def test_headon_fields():
    """정면 마주침: A(0,0) 북향 · B(0,40) 남향, 둘 다 1.0 m/s."""
    env = _env(1, 2)
    _place(env, [[[0.0, 0.0], [0.0, 40.0]]], [[0.0, 180.0]], [[1.0, 1.0]])
    f = vg.comm_pair_features(env, torch.tensor([[[1], [0]]]), 300.0)[0, 0, 0]   # A 가 받는 B
    assert abs(float(f[0])) < 1e-5 and abs(float(f[1]) + 1.0) < 1e-5            # Δψ = 180°
    assert abs(float(f[2]) - 1.0 / 1.8) < 1e-6                                   # sog
    assert abs(float(f[4])) < 1e-5 and abs(float(f[5]) + 2.0 / 3.6) < 1e-5       # 상대속도: 우현 0, 전방 −2 m/s
    assert abs(float(f[6]) - 1.0) < 1e-5                                          # dcpa 0 → 위험 1
    tcpa = 40.0 / 2.0
    assert abs(float(f[7]) - 1.0 / (1.0 + tcpa / vg.TCPA_RISK_DENOM)) < 1e-5
    assert int(f[8:13].argmax()) == vg.SIT_HEADON and int(f[13:18].argmax()) == vg.SIT_HEADON


def test_crossing_roles_giveway_standon():
    """교차: A(0,0) 북향, B(30,30) 서향 → B 가 A 의 우현 → A = GiveWay, B = StandOn. 둘 다 서로 맞물려 선언."""
    env = _env(1, 2)
    _place(env, [[[0.0, 0.0], [30.0, 30.0]]], [[0.0, 270.0]], [[1.0, 1.0]])
    topi = torch.tensor([[[1], [0]]])
    f = vg.comm_pair_features(env, topi, 300.0)
    a_gets, b_gets = f[0, 0, 0], f[0, 1, 0]
    assert int(a_gets[8:13].argmax()) == vg.SIT_GIVEWAY and int(a_gets[13:18].argmax()) == vg.SIT_STANDON
    assert int(b_gets[8:13].argmax()) == vg.SIT_STANDON and int(b_gets[13:18].argmax()) == vg.SIT_GIVEWAY


def test_role_gate_removes_safe_passing():
    """기하로는 교차(A 양보선)지만 dcpa ≥ 24 m 로 비껴 가면 역할 None — 'so as to involve risk of collision' 게이트.
    A(0,0) 북향, B(80,30) 서향, 둘 다 1.0 m/s → tcpa 55 s, dcpa ≈ 35 m. (정면 평행 비껴가기는 기존 cascade 가
    이미 '우현 대 우현 clear' 로 None 을 주므로 게이트 검사에 쓸 수 없음.)"""
    env = _env(1, 2)
    _place(env, [[[0.0, 0.0], [80.0, 30.0]]], [[0.0, 270.0]], [[1.0, 1.0]])
    topi = torch.tensor([[[1], [0]]])
    f = vg.comm_pair_features(env, topi, 300.0)[0, 0, 0]
    assert float(f[6]) == 0.0                                                    # dcpa 35 > 24 → 위험 0
    assert int(f[8:13].argmax()) == vg.SIT_NONE and int(f[13:18].argmax()) == vg.SIT_NONE
    g = vg.comm_pair_features(env, topi, 300.0, role_gate=False)[0, 0, 0]
    assert int(g[8:13].argmax()) == vg.SIT_GIVEWAY and int(g[13:18].argmax()) == vg.SIT_STANDON   # 게이트 끄면 기하 역할


def test_starboard_beam_relpos_and_receding():
    """우현 정횡 파트너는 relpos sin≈1. 멀어지는 배는 CPA 위험 0."""
    E, N = 1, 2
    g = _save_globals()
    try:
        _set_ext()
        pol = _policy()
        env = _env(E, N)
        _place(env, [[[0.0, 0.0], [40.0, 0.0]]], [[0.0, 90.0]], [[1.0, 1.0]])   # B 는 동쪽에서 동향(멀어짐)
        x, goal, self_s, sit = _obs(env, E, N)
        with torch.no_grad():
            _, parts = T.comm_gather(pol, env, x, goal, self_s, sit, 1)
        pr = parts[4][0, 0, 0]
        assert abs(float(pr[0]) - 1.0) < 1e-5 and abs(float(pr[1])) < 1e-5
        assert float(pr[3 + 6]) == 0.0 and float(pr[3 + 7]) == 0.0
    finally:
        _load_globals(g)


def test_intent_timing_is_last_command():
    """결정 t 의 intent = t−1 에 내린 명령 (동시결정이라 미래 누설 없음). 명령 속력은 절대값/1.8."""
    E, N = 1, 3
    env = _env(E, N)
    env.reset()
    a = torch.tensor([[[0.5, 0.2], [-1.3, -0.4], [0.0, 1.0]]])
    env.step(a)
    dmat = torch.cdist(env.pos, env.pos) + torch.eye(N)[None] * 1e9
    _, topi = torch.topk(dmat, N - 1, dim=-1, largest=False)
    f = vg.comm_pair_features(env, topi, 1e9)
    b = torch.arange(E)[:, None, None]
    a0 = a[..., 0].clamp(-1, 1)
    ts = ((a[..., 1].clamp(-1, 1) + 1) * 0.5 * env.max_speed).clamp(max=env.max_speed)
    assert torch.allclose(f[..., 18], a0[b, topi], atol=1e-6)
    assert torch.allclose(f[..., 19], (ts / 1.8)[b, topi], atol=1e-6)


def test_padding_finite_and_group_masks():
    E, N, K = 1, 3, 4
    g = _save_globals()
    try:
        _set_ext('intent')
        pol = _policy()
        env = _env(E, N)
        _place(env, [[[0.0, 0.0], [30.0, 10.0], [280.0, -280.0]]], [[0.0, 200.0, 45.0]], [[1.0, 1.2, 0.8]])
        x, goal, self_s, sit = _obs(env, E, N)
        with torch.no_grad():
            om, parts = T.comm_gather(pol, env, x, goal, self_s, sit, K)
        pm, pr = parts[3], parts[4]
        assert pr.shape[-1] == 23 and bool(torch.isfinite(pr).all()) and bool(torch.isfinite(om).all())
        pad = pm.squeeze(-1) <= 0
        assert bool(pad.any()) and bool((pr[pad] == 0).all())
        for groups, zero_idx in ((('state', 'role'), [18, 19]), (('state',), list(range(8, 20))), ((), list(range(20)))):
            with torch.no_grad():
                _, p2 = T.comm_gather(pol, env, x, goal, self_s, sit, K, groups=groups)
            assert bool((p2[4][..., 3:][..., zero_idx] == 0).all()), groups
    finally:
        _load_globals(g)


def test_field_shuffle_keeps_valid_pool_and_derangement():
    """field-shuffle(평가 절제) = 같은 env 안 *유효* (수신자,슬롯) 항목끼리만 섞고 자기 자리로는 안 돌아옴.
    유효 슬롯 값의 모음(분포)은 그대로, 패딩·반경 밖·자기 쌍 값은 절대 안 들어옴 (리뷰 2026-09-25 M1/E1 회귀 방지)."""
    E, N, K = 3, 10, 4
    g = _save_globals()
    try:
        _set_ext('state', 0.0, 56.0)                                       # ARPA@56: 파트너 0~1척인 배가 많아 패딩이 흔함
        pol = _policy()
        env = _env(E, N, seed=9)
        torch.manual_seed(21)
        _place(env, (torch.rand(E, N, 2) * 200 - 100).tolist(), (torch.rand(E, N) * 360).tolist(),
               (0.4 + torch.rand(E, N) * 1.2).tolist())
        x, goal, self_s, sit = _obs(env, E, N)
        with torch.no_grad():
            _, p0 = T.comm_gather(pol, env, x, goal, self_s, sit, K)
            _, p1 = T.comm_gather(pol, env, x, goal, self_s, sit, K, ext_shuffle_gen=torch.Generator().manual_seed(3))
        vm = p0[3].squeeze(-1) > 0
        e0, e1 = p0[4][..., 3:], p1[4][..., 3:]
        assert bool((e1[~vm] == 0).all())                                  # 패딩은 여전히 0
        n_multi = 0
        for e in range(E):
            v0, v1 = e0[e][vm[e]], e1[e][vm[e]]                             # [n,20]
            n = v0.shape[0]
            key = lambda t: sorted(tuple(round(float(z), 6) for z in r) for r in t)   # noqa: E731
            assert key(v0) == key(v1), f'env {e}: 유효 슬롯 값 모음이 바뀜(외부 값 유입)'
            if n >= 2:
                n_multi += 1
                same = (v0 - v1).abs().sum(-1) < 1e-9
                # 두 슬롯 값이 우연히 같은 경우를 빼면 자기 자리로 돌아온 슬롯이 없어야 함
                dup = torch.tensor([int(((v0 - v0[i]).abs().sum(-1) < 1e-9).sum()) > 1 for i in range(n)])
                assert not bool((same & ~dup).any()), f'env {e}: 고정점 있음'
        assert n_multi >= 1, '유효 슬롯 2개 이상인 env 가 없어 검사가 무의미함'
    finally:
        _load_globals(g)


# ─────────────────────────── 미러 (rollout = update) ───────────────────────────
def _mirror_once(fields, latent, partner, send=None, recv=None):
    E, N, K = 3, 8, 4
    _set_ext(fields, latent, partner)
    pol = _policy(1)
    env = _env(E, N, seed=7)
    env.reset()
    fs = T.FrameStack(E, N, 'cpu')
    radar, goal, self_s, sit = T.parse_obs(env._build_obs())
    fs.reset_all(radar)
    for _ in range(25):                                   # 속도·타각·조우가 생기게 굴림
        x = fs.get()
        with torch.no_grad():
            om, _ = T.comm_gather(pol, env, x, goal, self_s, sit, K, send_mask=send, recv_mask=recv)
            a, _, _, _ = pol.ctr_actor(x, goal, self_s, om, sit)
        obs, _, d, _ = env.step(a)
        radar, goal, self_s, sit = T.parse_obs(obs)
        fs.push(radar, d)
    x = fs.get()
    with torch.no_grad():
        om, (px, pg, ps, pm, pr, psi) = T.comm_gather(pol, env, x, goal, self_s, sit, K, send_mask=send, recv_mask=recv)
        _, lp, _, araw = pol.ctr_actor(x, goal, self_s, om, sit)
        M = E * N
        f = lambda t: t.reshape(M, *t.shape[2:])  # noqa: E731
        _, lp_u, *_ = pol.evaluate_actions(f(x), f(goal), f(self_s), f(px), f(pg), f(ps), f(pm), f(pr), f(araw),
                                           situation=f(sit), partner_situations=f(psi))
    return float((lp_u.reshape(-1) - lp.reshape(-1)).abs().max())


def test_mirror_all_arms():
    g = _save_globals()
    try:
        for fields, latent, partner in (('latent', 1.0, None), ('state', 1.0, None), ('intent', 1.0, None),
                                        ('state', 0.0, 56.0)):                     # 마지막 = ARPA@56
            d = _mirror_once(fields, latent, partner)
            assert d < 1e-5, (fields, latent, partner, d)
        send = torch.tensor([True, True, False, True, True, False, True, True])
        recv = torch.tensor([True, False, True, True, False, True, True, True])
        d = _mirror_once('intent', 1.0, None, send=send, recv=recv)               # 혼합 함대
        assert d < 1e-5, ('mixed', d)
    finally:
        _load_globals(g)


def test_latent_zero_removes_message_influence():
    """ARPA@56(latent 0): 메시지를 바꿔도 others_msg 가 안 변함 = 통신 없이 추적 정보만."""
    E, N, K = 1, 4, 3
    g = _save_globals()
    try:
        _set_ext('state', 0.0, 56.0)
        pol = _policy()
        env = _env(E, N)
        _place(env, [[[0.0, 0.0], [20.0, 20.0], [-25.0, 10.0], [5.0, -30.0]]], [[0.0, 225.0, 90.0, 10.0]],
               [[1.0, 1.1, 0.9, 1.2]])
        x, goal, self_s, sit = _obs(env, E, N)
        with torch.no_grad():
            om1, _ = T.comm_gather(pol, env, x, goal, self_s, sit, K)
            om2, _ = T.comm_gather(pol, env, x, goal, self_s, sit, K, msg_override=torch.randn(E, N, cfg.MSG_DIM))
        assert torch.equal(om1, om2)
    finally:
        _load_globals(g)


def test_unity_path_raises_for_ext():
    g = _save_globals()
    try:
        _set_ext()
        pol = _policy()
        msg = torch.zeros(1, 2, cfg.MSG_DIM)
        try:
            pol._get_others_msg(msg, comm_partners={0: [1], 1: [0]}, agent_id_list=[0, 1],
                                comm_relpos={0: [[0.0, 1.0, 0.1]], 1: [[0.0, 1.0, 0.1]]},
                                self_state=torch.zeros(1, 2, 4), goal=torch.zeros(1, 2, 2))
        except RuntimeError as e:
            assert 'gym' in str(e)
        else:
            raise AssertionError('Unity 경로가 EXT 를 조용히 받음')
    finally:
        _load_globals(g)


# ─────────────────────────── 체크포인트 ───────────────────────────
def _save_ckpt(pol, path, arm='ON'):
    snap = ckpt_io.snapshot_config(arm=arm, msg_dim=cfg.MSG_DIM, seed=0, n_envs=1, n_vessels=2, max_partners=4,
                                   trunc_boot=False)
    torch.save({'model_state_dict': pol.state_dict(), 'cfg_snapshot': snap, 'steps': 0}, path)
    return snap


def test_ckpt_roundtrip_and_guards():
    g = _save_globals()
    tmp = tempfile.mkdtemp()
    try:
        _set_ext('intent', 1.0, None)
        p_ext = os.path.join(tmp, 'ext.pt')
        snap = _save_ckpt(_policy(), p_ext)
        assert snap['comm_ext'] == 1 and snap['comm_fields'] == 'intent' and snap['comm_ext_layout'] == cfg.COMM_EXT_LAYOUT
        _load_globals(g)                                                     # 기본(EXT 0)으로 되돌린 뒤 복원
        r = ckpt_io.restore_policy(p_ext, 'cpu', tag='[t]')
        assert r.policy.relpos_dim == 23 and net.COMM_EXT and net.COMM_GROUPS == ('state', 'role', 'intent')
        assert r.effective['comm_fields'] == 'intent'
        # 절제 override 는 allow 가 있어야
        try:
            ckpt_io.restore_policy(p_ext, 'cpu', tag='[t]', comm_groups=('state', 'role'))
        except SystemExit:
            pass
        else:
            raise AssertionError('allow 없이 필드 override 가 통과함')
        r2 = ckpt_io.restore_policy(p_ext, 'cpu', tag='[t]', comm_groups=('state', 'role'), allow_fields_mismatch=True)
        assert net.COMM_GROUPS == ('state', 'role') and r2.effective['comm_groups'] == ['state', 'role']
        # comm_fields 가 빠진 EXT 스냅샷은 추정하지 않고 중단
        sd = torch.load(p_ext)
        sd['cfg_snapshot'].pop('comm_fields')
        p_bad = os.path.join(tmp, 'bad.pt')
        torch.save(sd, p_bad)
        try:
            ckpt_io.restore_policy(p_bad, 'cpu', tag='[t]')
        except SystemExit:
            pass
        else:
            raise AssertionError('comm_fields 없는 EXT 체크포인트가 복원됨')
        # ARPA 설정(latent 0, 반경 56) 복원 + 그 뒤 EXT 0 체크포인트를 열면 전역이 새지 않음
        _set_ext('state', 0.0, 56.0)
        p_arpa = os.path.join(tmp, 'arpa.pt')
        _save_ckpt(_policy(), p_arpa)
        _load_globals(g)
        ckpt_io.restore_policy(p_arpa, 'cpu', tag='[t]')
        assert net.COMM_LATENT == 0.0 and net.PARTNER_RANGE == 56.0 and net.COMM_GROUPS == ('state', 'role')
        p0 = os.path.join(tmp, 'plain.pt')
        _load_globals(g)
        _save_ckpt(_policy(), p0)
        _set_ext('intent', 0.0, 56.0)                                        # 일부러 오염시킨 뒤
        r0 = ckpt_io.restore_policy(p0, 'cpu', tag='[t]')
        assert r0.policy.relpos_dim == 3 and not net.COMM_EXT and net.COMM_GROUPS == ()
        assert net.COMM_LATENT == 1.0 and net.PARTNER_RANGE is None
    finally:
        _load_globals(g)


def test_env_lines_export_comm_keys():
    g = _save_globals()
    try:
        _set_ext('state', 0.0, 56.0)
        snap = ckpt_io.snapshot_config(arm='ON', msg_dim=cfg.MSG_DIM, seed=0, n_envs=1, n_vessels=2,
                                       max_partners=4, trunc_boot=False)
        lines, _, _ = ckpt_io.env_lines(snap)
        txt = '\n'.join(lines)
        for want in ('VESSEL_COMM_EXT=1', 'VESSEL_COMM_FIELDS=state', 'VESSEL_COMM_LATENT=0.0', 'VESSEL_PARTNER_RANGE=56.0'):
            assert want in txt, want
        # partner_range None = COMM_RANGE → unset 줄(셸 잔여값 차단). 통신 키 없는 구 스냅샷 → legacy 확정값 (리뷰 M3)
        _set_ext('intent', 1.0, None)
        snap2 = ckpt_io.snapshot_config(arm='ON', msg_dim=cfg.MSG_DIM, seed=0, n_envs=1, n_vessels=2,
                                        max_partners=4, trunc_boot=False)
        l2, u2, _ = ckpt_io.env_lines(snap2)
        assert 'unset VESSEL_PARTNER_RANGE' in l2 and 'VESSEL_PARTNER_RANGE' not in u2
        l3, u3, _ = ckpt_io.env_lines({'arm': 'ON'})
        assert 'export VESSEL_COMM_EXT=0' in l3 and 'unset VESSEL_PARTNER_RANGE' in l3 and 'VESSEL_COMM_EXT' not in u3
    finally:
        _load_globals(g)


TESTS = [test_defaults_are_off_and_bit_identical_structure, test_ext_structure_and_small_init,
         test_ext_requires_attention, test_role_cascade_matches_pairwise_dense, test_headon_fields,
         test_crossing_roles_giveway_standon, test_role_gate_removes_safe_passing,
         test_starboard_beam_relpos_and_receding, test_intent_timing_is_last_command,
         test_padding_finite_and_group_masks, test_field_shuffle_keeps_valid_pool_and_derangement, test_mirror_all_arms, test_latent_zero_removes_message_influence,
         test_unity_path_raises_for_ext, test_ckpt_roundtrip_and_guards, test_env_lines_export_comm_keys]


def main():
    fails = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")
    print('VERDICT:', 'ALL PASS' if fails == 0 else f'{fails} FAIL')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
