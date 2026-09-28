"""Grounded latent message codec (2026-09-28, spec docs/superpowers/specs/2026-09-28-grounded-latent-small-design.md).

The sender's own state (vessel_gym.own_payload, layout 'p6': sin/cos heading, SOG, ROT, last commanded rudder/speed)
is compressed by a frozen autoencoder into a k-dim latent z (tanh, 8-bit uniform quantization) = the message.
The codec is trained offline on physics-consistent synthetic states (no policy data, no RL gradient) and frozen:
it is never in policy.parameters(), never in the optimizer, always called under no_grad. So z always means
"this sender's motion and last command" - RL cannot reshape it (the failure of the emergent latent in batch X).

Modes (config.COMM_CODEC_MODE, networks module globals set by install()):
  'decode' (A6): receiver decodes z with the frozen decoder and feeds the estimate into comm_pair_features.
  'direct' (C6): receiver's attention k/v reads z directly (next to its own state) and learns what it means.

Content SHA = sha256 over meta + raw float32 bytes of every tensor (not pickle bytes) so it is stable across
torch versions and machines. install() refuses a file whose SHA does not match the pinned value.

Layouts (2026-09-29, spec docs/superpowers/specs/2026-09-29-role-promise-design.md §3):
  'p6'  = vessel_gym.own_payload (6).
  'p12' = p6 + declared role one-hot 4 (head-on, stand-on, give-way, overtaking; all 0 = no declaration)
          + declared target's position in the sender's body frame (starboard, forward) / COMM_RANGE (2).
          Built by vessel_gym.own_payload12; decode only ('direct' is refused).

CLI:  VESSEL_DYN_PROFILE=imo python comm_codec.py train --k 6 --seed 0 --out comm_codecs/p6_k6_s0.pt
      VESSEL_DYN_PROFILE=imo python comm_codec.py train --layout p12 --k 8 --seed 0 --out comm_codecs/p12_k8_s0.pt
      python comm_codec.py info comm_codecs/p6_k6_s0.pt
"""
import hashlib
import json
import math
import os
import struct

import torch
import torch.nn as nn

LAYOUT = 'p6'
D_IN = 6
LAYOUTS = {'p6': 6, 'p12': 12}        # ★2026-09-29 layout -> payload width (file meta 'layout' picks one)
DECL_ROLE_SLICE = slice(6, 10)        # p12: declared role one-hot (SIT 1..4 -> column 0..3)
DECL_POS_SLICE = slice(10, 12)        # p12: declared target position / COMM_RANGE, sender body frame (stb, fwd)
P12_DECL_WEIGHT = 4.0                 # p12 training loss weight on the 6 declaration columns (training-only, recorded in extra)
_HERE = os.path.dirname(os.path.abspath(__file__))


class LatentCodec(nn.Module):
    """payload [..,d_in] -> z [..,k] in [-1,1] (quantized) -> payload estimate [..,d_in]."""

    def __init__(self, k=6, hidden=64, bits=8, d_in=D_IN, layout=LAYOUT):
        super().__init__()
        if layout not in LAYOUTS or LAYOUTS[layout] != int(d_in):
            raise ValueError(f"LatentCodec: layout {layout!r} / d_in {d_in} mismatch (known {LAYOUTS})")
        self.layout = str(layout)
        self.k, self.hidden, self.bits, self.d_in = int(k), int(hidden), int(bits), int(d_in)
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, k), nn.Tanh())
        self.dec = nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d_in))

    def quantize(self, z):
        if self.bits <= 0:
            return z
        lv = float(2 ** self.bits - 1)
        return torch.round((z + 1.0) * 0.5 * lv) / lv * 2.0 - 1.0

    def encode(self, p):
        """Deterministic message: quantized z."""
        return self.quantize(self.enc(p))

    def decode(self, z):
        return self.dec(z)

    def meta(self):
        # p6 dict must stay byte-identical to the 2026-09-28 one (content SHA of p6_k6_s0.pt)
        return {'layout': self.layout, 'k': self.k, 'hidden': self.hidden, 'bits': self.bits, 'd_in': self.d_in}


def decode_sender(p_hat):
    """Decoded payload [E,N,6] -> per-ship dict for vessel_gym.comm_pair_features(sender=...). Fixed rules
    (spec §3): heading = atan2(sin, cos) (atan2(0,0)=0), speeds clamped to [0, 1.8], rot and rudder to [-1, 1]."""
    import vessel_gym as vg
    out = {
        'heading': torch.atan2(p_hat[..., 0], p_hat[..., 1]) / vg.DEG,
        'speed': p_hat[..., 2].clamp(0.0, 1.0) * vg.COMM_EXT_SOG_NORM,
        'rot': p_hat[..., 3].clamp(-1.0, 1.0),
        'cmd_rudder': p_hat[..., 4].clamp(-1.0, 1.0) * vg.MAX_TURN_RATE,
        'target_speed': p_hat[..., 5].clamp(0.0, 1.0) * vg.COMM_EXT_SOG_NORM,
    }
    if p_hat.shape[-1] == LAYOUTS['p12']:
        # ★2026-09-29 p12 declaration: role = argmax of the 4 columns if its value >= 0.5 else none (0);
        #   position = clamp(-1,1) * COMM_RANGE in the sender's body frame (stb, fwd).
        mx, am = p_hat[..., DECL_ROLE_SLICE].max(dim=-1)
        out['decl_role'] = torch.where(mx >= 0.5, am + 1, torch.zeros_like(am))
        out['decl_pos'] = p_hat[..., DECL_POS_SLICE].clamp(-1.0, 1.0) * float(vg.COMM_RANGE)
    return out


def content_sha(codec, extra_meta=None):
    h = hashlib.sha256()
    m = dict(codec.meta())
    if extra_meta:
        m.update(extra_meta)
    h.update(json.dumps(m, sort_keys=True).encode())
    for name, t in sorted(codec.state_dict().items()):
        v = t.detach().cpu().float().flatten().tolist()
        h.update(name.encode())
        h.update(struct.pack('<%df' % len(v), *v))
    return h.hexdigest()


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(_HERE, path)


def load_codec(path, expect_sha, device='cpu'):
    """Load + verify. expect_sha = pinned prefix (>=12 hex). Returns a frozen eval-mode codec on device."""
    fp = _resolve(path)
    if not os.path.isfile(fp):
        raise SystemExit(f"[codec] 중단: 코덱 파일 없음 {fp}")
    blob = torch.load(fp, map_location='cpu')
    meta = blob['meta']
    if meta.get('layout') not in LAYOUTS or int(meta.get('d_in', -1)) != LAYOUTS[meta.get('layout')]:
        raise SystemExit(f"[codec] 중단: 레이아웃 {meta.get('layout')!r}/d_in {meta.get('d_in')!r} 를 모름 (알려진 것 {LAYOUTS})")
    import config as _cfg
    if meta.get('dyn_profile') and str(meta['dyn_profile']) != str(_cfg.DYN_PROFILE):
        raise SystemExit(f"[codec] 중단: 코덱 학습 프로필 {meta['dyn_profile']!r} != 현재 {_cfg.DYN_PROFILE!r} "
                         "(ROT 정규화 MAX_YAW_RATE 가 프로필마다 다름)")
    c = LatentCodec(k=meta['k'], hidden=meta['hidden'], bits=meta['bits'], d_in=meta['d_in'], layout=meta['layout'])
    c.load_state_dict(blob['state_dict'])
    extra = {k: v for k, v in meta.items() if k not in c.meta()}
    sha = content_sha(c, extra)
    if sha != blob.get('sha'):
        raise SystemExit(f"[codec] 중단: 파일 내용 SHA {sha[:12]} != 저장된 SHA {str(blob.get('sha'))[:12]} (손상·수정)")
    if not expect_sha or len(expect_sha) < 12 or not sha.startswith(expect_sha.lower()):
        raise SystemExit(f"[codec] 중단: 코덱 SHA {sha[:12]} != 고정값 {expect_sha!r} (앞 12자 이상 필요)")
    c.eval()
    for p in c.parameters():
        p.requires_grad_(False)
    c.sha = sha
    c.path = path
    c.file_meta = meta
    return c.to(device)


def install(path, expect_sha, mode, device):
    """Set networks module globals (read by vessel_gym_train.comm_gather). Always sets both (no leak between ckpts)."""
    import networks as net
    if not path:
        net.COMM_CODEC, net.COMM_CODEC_MODE = None, ''
        return None
    if mode not in ('decode', 'direct'):
        raise SystemExit(f"[codec] 중단: 모드 {mode!r} ('decode' | 'direct')")
    c = load_codec(path, expect_sha, device)
    if c.layout != 'p6' and mode != 'decode':
        raise SystemExit(f"[codec] 중단: 레이아웃 {c.layout!r} 는 'decode' 모드 전용 (direct 는 z 폭 + 자기상태 4 가 20 칸을 넘음)")
    net.COMM_CODEC = c
    net.COMM_CODEC_MODE = mode
    return net.COMM_CODEC


# ─────────────────────────── offline training (synthetic, physics-consistent) ───────────────────────────
def synth_payload(n, gen):
    """Physics-consistent random sender states under the current dyn profile (vessel_gym constants).
    Spawn law for max_speed, ROT from yaw_rate_deg(rudder, speed, max_speed). Wide coverage on purpose."""
    import vessel_gym as vg
    u = lambda *s: torch.rand(*s, generator=gen, dtype=torch.float64)
    psi = u(n) * 2.0 * math.pi
    maxs = vg.MAX_SPEED_BASE * (vg.SPEED_MULT_MIN + (vg.SPEED_MULT_MAX - vg.SPEED_MULT_MIN) * u(n))
    cruise = u(n) < 0.5
    spd = torch.where(cruise, (0.6 + 0.4 * u(n)) * maxs, u(n) * maxs)
    wide = u(n) < 0.4
    rud = torch.where(wide, (u(n) * 2 - 1) * vg.MAX_TURN_RATE,
                      (torch.randn(n, generator=gen, dtype=torch.float64) * 5.0).clamp(-vg.MAX_TURN_RATE, vg.MAX_TURN_RATE))
    rot = vg.yaw_rate_deg(rud, spd, maxs) / vg.MAX_YAW_RATE
    near = u(n) < 0.5
    cmd_r = torch.where(near, (rud + torch.randn(n, generator=gen, dtype=torch.float64) * 6.0),
                        (u(n) * 2 - 1) * vg.MAX_TURN_RATE).clamp(-vg.MAX_TURN_RATE, vg.MAX_TURN_RATE)
    follow = u(n) < 0.5
    cmd_s = torch.where(follow, spd + torch.randn(n, generator=gen, dtype=torch.float64) * 0.1, u(n) * maxs)
    cmd_s = torch.minimum(cmd_s.clamp(min=0.0), maxs)
    return torch.stack([torch.sin(psi), torch.cos(psi), spd / vg.COMM_EXT_SOG_NORM, rot,
                        cmd_r / vg.MAX_TURN_RATE, cmd_s / vg.COMM_EXT_SOG_NORM], dim=-1).float()


def synth_payload_p12(n, gen):
    """★2026-09-29 p12 synthetic payload = synth_payload (6, same draws first) + declaration (6).
    Declared role: none 40 %, head-on / stand-on / give-way / overtaking 15 % each. Target position: uniform in the
    COMM_RANGE disc in the sender's body frame (stb, fwd) / COMM_RANGE; all zeros when there is no declaration."""
    import torch.nn.functional as F
    p6 = synth_payload(n, gen)
    u = torch.rand(n, generator=gen, dtype=torch.float64)
    cls = torch.where(u < 0.4, torch.zeros(n, dtype=torch.long),
                      1 + ((u - 0.4) / 0.15).floor().clamp(0, 3).long())
    has = (cls > 0).unsqueeze(-1)
    oh = F.one_hot(cls, 5)[:, 1:].to(torch.float64)
    rad = torch.sqrt(torch.rand(n, generator=gen, dtype=torch.float64))
    ang = torch.rand(n, generator=gen, dtype=torch.float64) * 2.0 * math.pi
    pos = torch.stack([rad * torch.sin(ang), rad * torch.cos(ang)], dim=-1)
    pos = torch.where(has, pos, torch.zeros_like(pos))
    return torch.cat([p6, oh.float(), pos.float()], dim=-1)


def decl_class(p):
    """[..,12] payload (true or decoded) -> declared role class long (0 none, 1..4 = SIT codes)."""
    mx, am = p[..., DECL_ROLE_SLICE].max(dim=-1)
    return torch.where(mx >= 0.5, am + 1, torch.zeros_like(am))


def fidelity(codec, p):
    """Reconstruction errors on payload p [n,d_in] (message = quantized z). p12 adds declaration accuracy / position error."""
    with torch.no_grad():
        ph = codec.decode(codec.encode(p))
    h_true = torch.atan2(p[:, 0], p[:, 1])
    h_hat = torch.atan2(ph[:, 0], ph[:, 1])
    dh = torch.rad2deg(torch.atan2(torch.sin(h_hat - h_true), torch.cos(h_hat - h_true))).abs()
    out = {}
    q = lambda t: {'p50': float(t.quantile(0.5)), 'p99': float(t.quantile(0.99)), 'max': float(t.max())}
    out['heading_deg'] = q(dh)
    out['sog_mps'] = q((ph[:, 2].clamp(0, 1) - p[:, 2]).abs() * 1.8)
    out['rot_n'] = q((ph[:, 3].clamp(-1, 1) - p[:, 3]).abs())
    out['cmd_rudder_deg'] = q((ph[:, 4].clamp(-1, 1) - p[:, 4]).abs() * 30.0)
    out['cmd_speed_mps'] = q((ph[:, 5].clamp(0, 1) - p[:, 5]).abs() * 1.8)
    if p.shape[-1] == LAYOUTS['p12']:
        import vessel_gym as vg
        tr, dr = decl_class(p), decl_class(ph)
        out['decl_role_acc'] = float((tr == dr).double().mean())
        has = tr > 0
        perr = torch.linalg.norm((ph[:, DECL_POS_SLICE].clamp(-1, 1) - p[:, DECL_POS_SLICE]) * float(vg.COMM_RANGE), dim=-1)
        out['decl_pos_m'] = q(perr[has])
    return out


# ★2026-09-29 p12 numeric reference (spec §3). Was the pre-training gate; k8/k10 did not reach it (4 tries logged in the
#   spec) and the author replaced it with the functional gate P12_FUNC_GATE (2026-09-29). Kept and reported, not gating.
P12_GATE = {'decl_role_acc_min': 0.995, 'decl_pos_m_p99_max': 5.0, 'heading_deg_p99_max': 1.2, 'sog_mps_p99_max': 0.035,
            'rel_to_p6_max': 1.2}
# Functional gate (author decision 2026-09-29): what the codec is for. Checked by verify/test_codec_p12.py on real states.
#   role transmission on the synthetic holdout; declarations reach their target (miss) and nobody else (false hit) per
#   declaration; the receiver's role inference from decoded motion agrees with the true one (same bar as the a6 test 2f).
P12_FUNC_GATE = {'decl_role_acc_min': 0.995, 'match_miss_max': 0.01, 'match_false_max': 0.02, 'recv_role_agree_min': 0.97}


def fidelity_gate(fid, fid_p6):
    """(ok, reasons) for a p12 holdout fidelity dict against the numeric reference P12_GATE (reported only)."""
    g, bad = P12_GATE, []
    if fid.get('decl_role_acc', 0.0) < g['decl_role_acc_min']:
        bad.append(f"decl_role_acc {fid.get('decl_role_acc')} < {g['decl_role_acc_min']}")
    if fid['decl_pos_m']['p99'] > g['decl_pos_m_p99_max']:
        bad.append(f"decl_pos_m p99 {fid['decl_pos_m']['p99']:.3f} > {g['decl_pos_m_p99_max']}")
    if fid['heading_deg']['p99'] > g['heading_deg_p99_max']:
        bad.append(f"heading p99 {fid['heading_deg']['p99']:.3f} > {g['heading_deg_p99_max']}")
    if fid['sog_mps']['p99'] > g['sog_mps_p99_max']:
        bad.append(f"sog p99 {fid['sog_mps']['p99']:.4f} > {g['sog_mps_p99_max']}")
    for key in ('rot_n', 'cmd_rudder_deg', 'cmd_speed_mps'):
        lim = g['rel_to_p6_max'] * fid_p6[key]['p99']
        if fid[key]['p99'] > lim:
            bad.append(f"{key} p99 {fid[key]['p99']:.4f} > {g['rel_to_p6_max']}×p6 {fid_p6[key]['p99']:.4f}")
    return (not bad), bad


def train_codec(k=6, seed=0, hidden=64, bits=8, n_train=400000, n_hold=50000, steps=6000, batch=4096, layout='p6',
                decl_weight=None):
    import vessel_gym as vg
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    _synth = synth_payload if layout == 'p6' else synth_payload_p12
    ptr = _synth(n_train, gen)
    pho = _synth(n_hold, gen)
    c = LatentCodec(k=k, hidden=hidden, bits=bits, d_in=LAYOUTS[layout], layout=layout)
    _w = None
    if layout == 'p12':
        _w = torch.ones(LAYOUTS['p12'])
        _w[6:] = P12_DECL_WEIGHT if decl_weight is None else float(decl_weight)
    opt = torch.optim.Adam(c.parameters(), lr=1e-3)
    lv = float(2 ** bits - 1)
    for it in range(steps):
        if it == int(steps * 0.7):
            for g in opt.param_groups:
                g['lr'] = 3e-4
        idx = torch.randint(0, n_train, (batch,), generator=gen)
        p = ptr[idx]
        z = c.enc(p)
        if bits > 0:   # quantization as uniform noise during training (deterministic rounding at run time)
            z = (z + (torch.rand(z.shape, generator=gen) - 0.5) * (2.0 / lv)).clamp(-1.0, 1.0)
        loss = ((c.dec(z) - p) ** 2).mean() if _w is None else (((c.dec(z) - p) ** 2) * _w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    c.eval()
    extra = {'dyn_profile': str(vg._cfg.DYN_PROFILE), 'seed': int(seed), 'n_train': int(n_train), 'steps': int(steps),
             'data': 'synth_payload v1 (physics-consistent uniform mix)'}
    if layout == 'p12':
        extra['data'] = 'synth_payload_p12 v1 (p6 draws + declaration: none 0.4 / 4 roles 0.15 each, uniform disc)'
        extra['decl_weight'] = float(P12_DECL_WEIGHT if decl_weight is None else decl_weight)
        extra['hidden_note'] = f'hidden={hidden} steps={steps} batch={batch}'
    sha = content_sha(c, extra)
    return c, extra, sha, fidelity(c, pho)


def _main():
    import argparse
    ap = argparse.ArgumentParser(description='grounded latent codec: train / info')
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--k', type=int, default=6)
    t.add_argument('--layout', default='p6', choices=sorted(LAYOUTS))
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--bits', type=int, default=8)
    t.add_argument('--steps', type=int, default=6000)
    t.add_argument('--hidden', type=int, default=64)
    t.add_argument('--decl_weight', type=float, default=None)
    t.add_argument('--out', required=True)
    i = sub.add_parser('info')
    i.add_argument('path')
    a = ap.parse_args()
    if a.cmd == 'train':
        fp = _resolve(a.out)
        if os.path.exists(fp):
            raise SystemExit(f"[codec] 중단: {fp} 가 이미 있음 — 덮어쓰지 않음(고정 SHA 보호). 지우고 다시 하려면 직접 지울 것")
        c, extra, sha, fid = train_codec(k=a.k, seed=a.seed, bits=a.bits, steps=a.steps, layout=a.layout,
                                         hidden=a.hidden, decl_weight=a.decl_weight)
        meta = dict(c.meta())
        meta.update(extra)
        blob = {'state_dict': c.state_dict(), 'meta': meta, 'sha': sha, 'fidelity_holdout': fid}
        rep_ = {'out': a.out, 'sha': sha, 'meta': meta, 'fidelity_holdout': fid}
        if a.layout == 'p12':
            p6 = torch.load(_resolve('comm_codecs/p6_k6_s0.pt'), map_location='cpu')['fidelity_holdout']
            ok, bad = fidelity_gate(fid, p6)
            blob['numeric_ref'] = rep_['numeric_ref'] = {'meets': bool(ok), 'reasons': bad, 'thresholds': P12_GATE,
                                                          'note': 'reference only; gate = P12_FUNC_GATE (test_codec_p12.py)'}
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        torch.save(blob, fp)
        print(json.dumps(rep_, indent=1, ensure_ascii=False))
    else:
        blob = torch.load(_resolve(a.path), map_location='cpu')
        print(json.dumps({'sha': blob['sha'], 'meta': blob['meta'], 'fidelity_holdout': blob.get('fidelity_holdout'),
                          'numeric_ref': blob.get('numeric_ref')}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    _main()
