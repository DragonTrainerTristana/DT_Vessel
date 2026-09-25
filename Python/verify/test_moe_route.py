"""
MoE 라우팅 helper `networks._moe_buckets`(2026-09-26, design_trainer.md T2) 비트동일 테스트.
pytest 없이 `python verify/test_moe_route.py` 로 돈다. CPU 와 (있으면) cuda:0 둘 다.

검사 대상 = 옛 mask 루프 `for k: mask=(sit==k); if mask.any(): out[mask] = expert_k(x[mask])` 를
stable argsort + bincount + .tolist() 1회로 바꾼 것이 forward·backward 모두 *비트동일*인가.
  - 참조 = 옛 루프를 이 파일 안에 그대로 옮긴 사본(production 토글 없음, ref_* 함수).
  - 비교 = MessageActor.forward / ControlActor._route / ControlActor.get_logprob_entropy /
    ControlActor.consumer_decode / Critic.forward 의 모든 출력 + 정책 파라미터 *전부*의 grad + 입력 grad
    → torch.equal. 빈 전문가(행 0개)의 비공유 파라미터는 양쪽 다 grad None 이어야 함(Adam 의미 보존:
    0행으로 돌리면 zeros grad 가 생겨 모멘트 감쇠·step 증가 — 루프는 그 전문가를 얼려 둔다).
  - 구성 = config 기본값(USE_MOE=1·MOE_SHARED=1·SHARED_ENCODER=all·MOE_FAST=0) + coupling ON 정책 1개.
  - 덤: StateReconDecoder.loss 의 통계 갱신(T3e, float() sync 제거)도 옛 코드 사본과 버퍼·손실·raw 를 대조.
CUDA 에선 결정론 커널을 켜고 판정한다. 참조 vs 참조 대조군이 먼저 통과해야 하고, 갈리면 그 장치는 판정 불가.
"""
import copy
import os
import sys
import types

# verify/ 로 내려온 뒤에도 Python/ 루트의 networks 를 찾으려면 sys.path 에 넣어야 함 (test_vessel_gym_fidelity 와 같은 패턴)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
# cuBLAS 결정론 workspace 는 torch import 전에 잡아야 한다
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
# ★config 는 import 시점에 env 를 읽는다. preflight 가 배치 env 를 export 한 채 부를 수 있으므로
#   라우팅 구조를 정하는 키는 기본값(MoE 켜짐·전문가 간 인코더 공유·망 간 공유·루프 경로)으로 고정한다.
for _k in ('VESSEL_USE_MOE', 'VESSEL_MOE_SHARED', 'VESSEL_MOE_WIDTH', 'VESSEL_SHARED_ENCODER',
           'VESSEL_MOE_FAST', 'VESSEL_COMM_CONSUMER_COUPLING'):
    os.environ.pop(_k, None)
import torch  # noqa: E402

import config as cfg  # noqa: E402
import networks as net  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)   # cumsum 등은 경고만(참조 vs 참조 대조군이 판정)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

K = net.NUM_COLREGS_SITUATIONS
B, N = 4, 16            # M = 64 행
SEED_W = 12345          # 손실 가중치 난수(new·ref 에 같은 값)


# ───────────────────────── 참조: 2026-09-26 이전 mask 루프 사본 ─────────────────────────
def _sit_vec(situation, M, num_experts, device):
    if situation is not None:
        return situation.reshape(M).long().clamp(0, num_experts - 1)
    return torch.zeros(M, dtype=torch.long, device=device)


def ref_msg_forward(ma, x, goal, self_state, situation):
    """옛 MessageActor.forward 의 MoE 루프(구 networks.py :620-626)."""
    batch_size, n_agent, _ = x.shape
    M = batch_size * n_agent
    x_f = x.reshape(M, -1)
    goal_f = goal.reshape(M, -1)
    self_f = self_state.reshape(M, -1)
    sit_oh = net._situation_onehot(situation, M, x_f) if net.SITUATION_INPUT else None
    sit = _sit_vec(situation, M, ma.num_experts, x.device)
    msg = x_f.new_zeros(M, ma.msg_dim)
    for k in range(ma.num_experts):
        mask = (sit == k)
        if mask.any():
            msg[mask] = ma.experts[k](x_f[mask], goal_f[mask], self_f[mask],
                                      sit_oh[mask] if sit_oh is not None else None)
    return msg.view(batch_size, n_agent, ma.msg_dim)


def ref_route(ca, x, goal, self_state, others_msg, situation):
    """옛 ControlActor._route 의 MoE 루프(구 :778-796). 바인딩해서 get_logprob_entropy 에도 꽂는다."""
    batch_size, n_agent, _ = x.shape
    M = batch_size * n_agent
    x_f = x.reshape(M, -1)
    goal_f = goal.reshape(M, -1)
    self_f = self_state.reshape(M, -1)
    om_f = others_msg.reshape(M, -1)
    sit_oh = net._situation_onehot(situation, M, x_f) if net.SITUATION_INPUT else None
    sit = _sit_vec(situation, M, ca.num_experts, x.device)
    z = x_f.new_zeros(M, ca.core_hidden)
    mean = x_f.new_zeros(M, ca.action_size)
    logstd = x_f.new_zeros(M, ca.action_size)
    dec_full = None
    for k in range(ca.num_experts):
        mask = (sit == k)
        if mask.any():
            zk = ca.experts[k].backbone(x_f[mask], goal_f[mask], self_f[mask], om_f[mask],
                                        sit_oh[mask] if sit_oh is not None else None)
            mk, lk, dk = ca.experts[k].head(zk)
            z[mask] = zk
            mean[mask] = mk
            logstd[mask] = lk
            if dk is not None:
                if dec_full is None:
                    dec_full = x_f.new_zeros(M, dk.shape[-1])
                dec_full[mask] = dk
    ca._cache_dec(z, dec_full)
    return z, mean, logstd, batch_size, n_agent


def ref_consumer_decode(ca, z, situation):
    """옛 ControlActor.consumer_decode 의 MoE 루프(구 :821-826)."""
    M = z.shape[0]
    out_dim = ca.consumer_k * net.GOAL_SIZE
    sit = _sit_vec(situation, M, ca.num_experts, z.device)
    out = z.new_zeros(M, out_dim)
    for k in range(ca.num_experts):
        mask = (sit == k)
        if mask.any():
            out[mask] = ca.experts[k].consumer_decoder(z[mask])
    return out


def ref_critic_forward(cr, x, goal, self_state, others_msg, situation, global_feat):
    """옛 Critic.forward 의 MoE 루프(구 :994-1001)."""
    batch_size, n_agent, _ = x.shape
    M = batch_size * n_agent
    x_f = x.reshape(M, -1)
    goal_f = goal.reshape(M, -1)
    self_f = self_state.reshape(M, -1)
    om_f = others_msg.reshape(M, -1)
    gf_f = global_feat.reshape(M, global_feat.shape[-2], 6) if global_feat is not None else None
    sit_oh = net._situation_onehot(situation, M, x_f) if net.SITUATION_INPUT else None
    sit = _sit_vec(situation, M, cr.num_experts, x.device)
    v = x_f.new_zeros(M, 1)
    for k in range(cr.num_experts):
        mask = (sit == k)
        if mask.any():
            v[mask] = cr.experts[k](x_f[mask], goal_f[mask], self_f[mask], om_f[mask],
                                    sit_oh[mask] if sit_oh is not None else None,
                                    gf_f[mask] if gf_f is not None else None)
    return v.view(batch_size, n_agent, 1)


def ref_state_recon_loss(dec, msg, goal, self_state, situation, own_threat, own_threat_mask,
                         own_future, own_future_mask):
    """옛 StateReconDecoder.loss 사본(구 :320-380): float(stat_inited) 분기 + raw float. 나머지는 현재와 같음."""
    F = torch.nn.functional
    self = dec
    Nn = msg.shape[0]
    sit_oh = F.one_hot(situation.reshape(Nn).long().clamp(0, self.num_sit - 1),
                       self.num_sit).float().unsqueeze(1)
    tm = torch.ones_like(own_threat) if own_threat_mask is None else own_threat_mask
    fm = torch.ones_like(own_future) if own_future_mask is None else own_future_mask
    tgt = torch.cat([goal, self_state, sit_oh, own_threat, own_future], dim=-1)
    msk = torch.cat([torch.ones_like(goal), torch.ones_like(self_state),
                     torch.ones_like(sit_oh), tm, fm], dim=-1)
    tf = tgt.reshape(Nn, -1); mf = msk.reshape(Nn, -1)
    if self.training:
        with torch.no_grad():
            cnt = mf.sum(0).clamp(min=1.0)
            bm = (tf * mf).sum(0) / cnt
            bv = (((tf - bm).pow(2)) * mf).sum(0) / cnt
            valid = (mf.sum(0) > 0)
            if net._RECON_LEGACY_STAT:
                valid = torch.ones_like(valid)
            if float(self.stat_inited) == 0.0:
                self.run_mean.copy_(torch.where(valid, bm, self.run_mean))
                self.run_var.copy_(torch.where(valid, bv.clamp(min=1e-8), self.run_var))
                self.stat_inited.fill_(1.0)
            else:
                new_m = self.run_mean * (1 - self.momentum) + self.momentum * bm
                new_v = self.run_var * (1 - self.momentum) + self.momentum * bv.clamp(min=1e-8)
                self.run_mean.copy_(torch.where(valid, new_m, self.run_mean))
                self.run_var.copy_(torch.where(valid, new_v, self.run_var))
    sd = self.run_var.clamp(min=1e-6).sqrt()
    z = ((tf - self.run_mean) / sd) * mf
    pred = self.net(msg).reshape(Nn, -1)
    se = (pred - z).pow(2) * mf
    raw, total = {}, 0.0
    for gi, (g, (a, b)) in enumerate(self._slices().items()):
        gl = se[:, a:b].sum() / mf[:, a:b].sum().clamp(min=1.0)
        raw[g] = float(gl.detach())
        w_pre = self.loss_ema[gi].detach().clone()
        if self.training:
            with torch.no_grad():
                self.loss_ema[gi] = (1 - self.momentum) * self.loss_ema[gi] + self.momentum * gl.detach()
        w = w_pre if self.ema_pre else self.loss_ema[gi].detach()
        if net._RECON_EMA_FLOOR > 0.0:
            w = w.clamp(min=net._RECON_EMA_FLOOR)
        total = total + gl / (w + 1e-4)
    return total / len(self.GROUPS), raw


# ───────────────────────── 비교 도구 ─────────────────────────
def _same(name, a, b):
    if a is None and b is None:
        return
    assert (a is None) == (b is None), f'{name}: None 불일치 (new={a is None}, ref={b is None})'
    assert a.dtype == b.dtype and a.shape == b.shape, f'{name}: dtype/shape 불일치 {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}'
    if not torch.equal(a, b):
        d = (a.double() - b.double()).abs().max().item()
        raise AssertionError(f'{name}: torch.equal 실패, max|diff|={d:.3e}')


def _same_dict(tag, ga, gb):
    assert ga.keys() == gb.keys(), f'{tag}: 키 집합 불일치'
    for n in ga:
        _same(f'{tag}/{n}', ga[n], gb[n])


def _zero(policy, inputs):
    policy.zero_grad(set_to_none=True)
    for t in inputs.values():
        t.grad = None


def _grads(policy, inputs):
    g = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in policy.named_parameters()}
    for n, t in inputs.items():
        g['input:' + n] = None if t.grad is None else t.grad.detach().clone()
    return g


def _weights(outs, device):
    gen = torch.Generator().manual_seed(SEED_W)
    return [None if o is None else torch.randn(o.shape, generator=gen, dtype=o.dtype).to(device) for o in outs]


def _run(policy, inputs, fn, device):
    """fn() → 텐서 튜플(None 허용). loss = Σ(out·w).sum() 역전파 → (출력 사본, grad dict, RNG 불변 여부)."""
    _zero(policy, inputs)
    rng0 = torch.get_rng_state()
    crng0 = torch.cuda.get_rng_state(device) if device.startswith('cuda') else None
    outs = fn()
    rng_same = torch.equal(rng0, torch.get_rng_state()) and (
        crng0 is None or torch.equal(crng0, torch.cuda.get_rng_state(device)))
    ws = _weights(outs, device)
    loss = sum((o * w).sum() for o, w in zip(outs, ws) if o is not None)
    loss.backward()
    return [None if o is None else o.detach().clone() for o in outs], _grads(policy, inputs), rng_same


def _check_empty_experts(tag, grads, prefix, present, must_have):
    """빈 전문가 k: prefix{k}. 의 비공유(radar_encoder 제외) 파라미터 전부 grad None.
    있는 전문가 k: must_have 접미사 파라미터는 grad non-None."""
    for k in range(K):
        names = [n for n in grads if n.startswith(f'{prefix}{k}.') and 'radar_encoder' not in n]
        assert names, f'{tag}: {prefix}{k}. 파라미터가 없음'
        for n in names:
            if k in present:
                if any(n.endswith(s) for s in must_have):
                    assert grads[n] is not None, f'{tag}: 전문가 {k} 는 행이 있는데 {n} grad None'
            else:
                assert grads[n] is None, f'{tag}: 빈 전문가 {k} 의 {n} grad 가 None 이 아님'


def _present(situation, M):
    if situation is None:
        return {0}
    s = situation.reshape(M).long().clamp(0, K - 1)
    return set(torch.unique(s).tolist())


def _situations(device):
    """이름 → situation 텐서. 빈 버킷을 강제하는 케이스 포함."""
    M = B * N
    gen = torch.Generator().manual_seed(7)
    pick = lambda vals: torch.tensor(vals, dtype=torch.long)[torch.randint(len(vals), (M,), generator=gen)]
    s13 = pick([0, 2, 4]); s13[0] = 0; s13[1] = 2; s13[2] = 4                         # 1·3 비어 있음
    s_all = pick([0, 1, 2, 3, 4]); s_all[:5] = torch.arange(5)                         # 전부 있음
    s_clamp = s_all.clone(); s_clamp[5] = 7; s_clamp[6] = -1                          # clamp 경로(7→4, -1→0)
    s_last = torch.full((M,), 4, dtype=torch.long)                                    # 마지막 전문가만
    s_skew = pick([0] * 60 + [1, 3]); s_skew[0] = 1; s_skew[1] = 3                    # 실제 분포(상황0 대다수)
    return {
        'empty13': s13.view(B, N).to(device),
        'none': None,
        'all': s_all.view(B, N).to(device),
        'all_shape3': s_all.view(B, N, 1).to(device),
        'clamp': s_clamp.view(B, N).to(device),
        'only4': s_last.view(B, N).to(device),
        'skew': s_skew.view(B, N).to(device),
    }


def _inputs(device, hidden, seed=0):
    gen = torch.Generator().manual_seed(seed)
    M = B * N
    x = (torch.rand(B, N, cfg.FRAMES * net.STATE_SIZE, generator=gen) - 0.5).to(device)
    goal = torch.rand(B, N, net.GOAL_SIZE, generator=gen).to(device)
    self_s = (torch.rand(B, N, net.SELF_STATE_SIZE, generator=gen) * 2 - 1).to(device)
    om = (torch.randn(B, N, cfg.MSG_DIM, generator=gen) * 0.5).to(device).requires_grad_(True)
    gf = torch.randn(B, N, N, 6, generator=gen).to(device)
    raw = torch.randn(B, N, cfg.CONTINUOUS_ACTION_SIZE, generator=gen).to(device)
    z = torch.randn(M, hidden, generator=gen).to(device).requires_grad_(True)
    return dict(x=x, goal=goal, self_s=self_s, om=om, gf=gf, raw=raw, z=z)


def _policy(seed, device):
    torch.manual_seed(seed)
    pol = net.CNNPolicy(cfg.MSG_DIM, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES).to(device)
    assert pol.ctr_actor.use_moe and pol.critic.use_moe and pol.msg_actor.use_moe, 'USE_MOE=1 이어야 함'
    assert not net._moe_fast_on(pol.ctr_actor), 'MOE_FAST=0(루프 경로) 이어야 함'
    return pol


# ───────────────────────── 테스트 ─────────────────────────
def test_buckets(device):
    """_moe_buckets 의 (k, idx) == (sit==k).nonzero() 오름차순, 빈 k 는 건너뜀, k 오름차순."""
    M = B * N
    for name, sit in _situations(device).items():
        s = _sit_vec(sit, M, K, device)
        seen = []
        for k, idx in net._moe_buckets(s, K):
            assert idx.dtype == torch.long and idx.dim() == 1
            _same(f'buckets[{name}] k={k}', idx, (s == k).nonzero().squeeze(1))
            seen.append(k)
        assert seen == [k for k in range(K) if bool((s == k).any())], f'buckets[{name}]: 건너뛰기/순서 {seen}'
    # 2026-09-25 트렁크 분포와 비슷한 큰 벡터(65,536 행, 상황0 98.7%)에서도 동일
    gen = torch.Generator().manual_seed(99)
    big = torch.where(torch.rand(65536, generator=gen) < 0.987, torch.zeros(65536, dtype=torch.long),
                      torch.randint(1, K, (65536,), generator=gen)).to(device)
    for k, idx in net._moe_buckets(big, K):
        _same(f'buckets[big] k={k}', idx, (big == k).nonzero().squeeze(1))


def _run_all_nets(pol, inp, sit, device, use_ref):
    """다섯 경로를 (use_ref 면 옛 루프로) 돌려 {경로: (outs, grads, rng_same)} 반환."""
    ma, ca, cr = pol.msg_actor, pol.ctr_actor, pol.critic
    x, goal, ss, om, gf, raw, z = (inp[k] for k in ('x', 'goal', 'self_s', 'om', 'gf', 'raw', 'z'))
    res = {}
    # 1) MessageActor
    fn = (lambda: (ref_msg_forward(ma, x, goal, ss, sit),)) if use_ref else (lambda: (ma(x, goal, ss, sit),))
    res['msg'] = _run(pol, {}, fn, device)
    # 2) ControlActor._route (+ 캐시된 dec)
    def fn_route():
        r = ref_route(ca, x, goal, ss, om, sit) if use_ref else ca._route(x, goal, ss, om, sit)
        c = ca._dec_cache
        assert (c is None) or (c[0] is r[0])
        return (r[0], r[1], r[2], None if c is None else c[1])
    res['route'] = _run(pol, {'om': om}, fn_route, device)
    # 3) get_logprob_entropy: 참조는 _route 만 옛 루프로 바꿔 꽂음(꼬리는 production 코드 그대로)
    def fn_lpe():
        if use_ref:
            ca._route = types.MethodType(ref_route, ca)
        try:
            lp, ent, mean, zz = ca.get_logprob_entropy(x, goal, ss, om, raw, sit)
        finally:
            if use_ref:
                del ca._route
        ca._dec_cache = None
        return (lp, ent, mean, zz)
    res['lpe'] = _run(pol, {'om': om}, fn_lpe, device)
    # 4) consumer_decode (coupling OFF 면 pop_consumer_dec 가 이리로 떨어짐)
    fn = (lambda: (ref_consumer_decode(ca, z, sit),)) if use_ref else (lambda: (ca.consumer_decode(z, sit),))
    res['consumer'] = _run(pol, {'z': z}, fn, device)
    # 5) Critic, global_feat 있음/없음
    for tag, g in (('critic_gf', gf), ('critic_nogf', None)):
        fn = (lambda g=g: (ref_critic_forward(cr, x, goal, ss, om, sit, g),)) if use_ref \
            else (lambda g=g: (cr(x, goal, ss, om, sit, g),))
        res[tag] = _run(pol, {'om': om}, fn, device)
    return res


def test_policy(device, coupling):
    """config 기본 정책(coupling=False) / coupling ON 정책: 모든 경로 출력·grad torch.equal + 빈 전문가 grad None."""
    old = net.COMM_CONSUMER_COUPLING
    net.COMM_CONSUMER_COUPLING = coupling
    try:
        pol = _policy(1 if coupling else 0, device)
    finally:
        net.COMM_CONSUMER_COUPLING = old
    assert pol.ctr_actor.experts[0].consumer_coupling == coupling
    inp = _inputs(device, pol.ctr_actor.core_hidden)
    M = B * N
    n_cases = 0
    for name, sit in _situations(device).items():
        present = _present(sit, M)
        ref_a = _run_all_nets(pol, inp, sit, device, use_ref=True)
        ref_b = _run_all_nets(pol, inp, sit, device, use_ref=True)      # 대조군: 참조 vs 참조
        new = _run_all_nets(pol, inp, sit, device, use_ref=False)
        for path in ref_a:
            oa, ga, _ = ref_a[path]
            ob, gb, _ = ref_b[path]
            on, gn, rng_same = new[path]
            tag = f'{device}/coupling={int(coupling)}/{name}/{path}'
            for i, (p, q) in enumerate(zip(oa, ob)):
                try:
                    _same(f'{tag}/control-out{i}', p, q)
                except AssertionError as e:
                    raise AssertionError(f'대조군(참조 vs 참조) 불일치 — 이 장치는 결정론이 안 잡혀 판정 불가: {e}')
            try:
                _same_dict(f'{tag}/control-grad', ga, gb)
            except AssertionError as e:
                raise AssertionError(f'대조군(참조 vs 참조) 불일치 — 이 장치는 결정론이 안 잡혀 판정 불가: {e}')
            assert len(on) == len(oa)
            for i, (p, q) in enumerate(zip(on, oa)):
                _same(f'{tag}/out{i}', p, q)
            _same_dict(f'{tag}/grad', gn, ga)
            assert rng_same, f'{tag}: 새 경로가 RNG 를 소비함'
            n_cases += 1
        # 빈 전문가 grad None (양쪽 다 — 참조는 위 grad 동일성으로 같음이 보장됨)
        _check_empty_experts(f'{device}/{name}/msg', new['msg'][1], 'msg_actor.experts.', present, ('fc2.weight',))
        must_ctr = ('fc2.weight', 'action_mean.weight') + (('consumer_decoder.0.weight',) if coupling else ())
        _check_empty_experts(f'{device}/{name}/route', new['route'][1], 'ctr_actor.experts.', present, must_ctr)
        _check_empty_experts(f'{device}/{name}/lpe', new['lpe'][1], 'ctr_actor.experts.', present, must_ctr)
        _check_empty_experts(f'{device}/{name}/consumer', new['consumer'][1], 'ctr_actor.experts.', present,
                             ('consumer_decoder.0.weight',))
        _check_empty_experts(f'{device}/{name}/critic', new['critic_gf'][1], 'critic.experts.', present,
                             ('fc2.weight', 'value_out.weight'))
        # 이 케이스가 실제로 뭔가 검사했는지: 빈 전문가가 있는 케이스는 정말 비어 있어야 함
        if name in ('empty13', 'none', 'only4', 'skew'):
            assert len(present) < K, f'{name}: 빈 전문가가 없음 — 케이스 설계 오류'
        if coupling:
            assert new['route'][0][3] is not None, 'coupling ON 인데 dec 캐시가 None'
    return n_cases


def test_state_recon(device):
    """T3e: StateReconDecoder.loss 통계 갱신(torch.where 분기) == 옛 float() 분기. 첫 호출·EMA·전부 마스크 성분."""
    pol = _policy(0, device)
    assert pol.state_recon is not None, 'STATE_RECON_COEF>0 (기본 0.05) 이어야 인스턴스가 있음'
    dec_new = pol.state_recon
    dec_ref = copy.deepcopy(dec_new)
    gen = torch.Generator().manual_seed(3)
    Nn = 48
    tk, ik = dec_new.dims['threat'], dec_new.dims['future']
    losses_new, losses_ref = [], []
    for call in range(4):
        msg = torch.randn(Nn, 1, cfg.MSG_DIM, generator=gen).to(device)
        goal = torch.rand(Nn, 1, net.GOAL_SIZE, generator=gen).to(device)
        ss = torch.randn(Nn, 1, net.SELF_STATE_SIZE, generator=gen).to(device)
        sit = torch.randint(0, K, (Nn, 1), generator=gen).to(device)
        thr = torch.randn(Nn, 1, tk, generator=gen).to(device)
        fut = torch.randn(Nn, 1, ik, generator=gen).to(device) * 0.01
        tmask = (torch.rand(Nn, 1, tk, generator=gen) < 0.7).float().to(device)
        fmask = (torch.rand(Nn, 1, ik, generator=gen) < 0.7).float().to(device)
        if call == 1:
            fmask = torch.zeros_like(fmask)       # future 성분 전부 마스크 → valid False → 통계 불변 경로
        args = (goal, ss, sit, thr, tmask, fut, fmask)
        t_new, r_new = dec_new.loss(msg, *args)
        t_ref, r_ref = ref_state_recon_loss(dec_ref, msg, *args)
        _same(f'{device}/recon[{call}]/total', t_new.detach(), t_ref.detach())
        assert list(r_new.keys()) == list(r_ref.keys()) == list(dec_new.GROUPS), 'raw 키·순서 변경'
        for g in r_new:
            assert isinstance(r_new[g], torch.Tensor) and r_new[g].dim() == 0 and not r_new[g].requires_grad, \
                f'raw[{g}] 는 detached 0-d tensor 여야 함'
            assert float(r_new[g]) == r_ref[g], f'{device}/recon[{call}]/raw[{g}] {float(r_new[g])!r} vs {r_ref[g]!r}'
        for bn in ('run_mean', 'run_var', 'stat_inited', 'loss_ema'):
            _same(f'{device}/recon[{call}]/{bn}', getattr(dec_new, bn), getattr(dec_ref, bn))
        losses_new.append(t_new); losses_ref.append(t_ref)
    # 역전파도 같은 grad (net 파라미터)
    gn = torch.autograd.grad(sum(losses_new), list(dec_new.net.parameters()))
    gr = torch.autograd.grad(sum(losses_ref), list(dec_ref.net.parameters()))
    for i, (a, b) in enumerate(zip(gn, gr)):
        _same(f'{device}/recon/grad{i}', a, b)
    assert float(dec_new.stat_inited) == 1.0


def main():
    devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
    print(f'torch {torch.__version__} | devices {devices} | USE_MOE={cfg.USE_MOE} MOE_SHARED={cfg.MOE_SHARED} '
          f'SHARED_ENCODER={cfg.SHARED_ENCODER} MOE_FAST={int(net._MOE_FAST)} SITUATION_INPUT={cfg.SITUATION_INPUT} '
          f'CENTRAL_CRITIC={cfg.CENTRAL_CRITIC}', flush=True)
    assert cfg.USE_MOE and cfg.MOE_SHARED and cfg.SHARED_ENCODER == 'all' and not net._MOE_FAST, \
        '이 테스트는 config 기본값(USE_MOE=1·MOE_SHARED=1·SHARED_ENCODER=all·MOE_FAST=0) 전제'
    fails = 0
    for dev in devices:
        for name, fn in (('buckets', lambda: test_buckets(dev)),
                         ('policy(coupling=0)', lambda: test_policy(dev, False)),
                         ('policy(coupling=1)', lambda: test_policy(dev, True)),
                         ('state_recon', lambda: test_state_recon(dev))):
            try:
                r = fn()
                extra = f' ({r} 경로×케이스 torch.equal)' if isinstance(r, int) else ''
                print(f'[PASS] {dev:6s} {name}{extra}', flush=True)
            except Exception as e:   # noqa: BLE001 — 실패 사유를 전부 모아 보여준다
                fails += 1
                print(f'[FAIL] {dev:6s} {name}: {type(e).__name__}: {e}', flush=True)
    print('ALL PASS' if fails == 0 else f'FAILED ({fails})', flush=True)
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
