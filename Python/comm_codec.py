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
LAYOUTS = {'p6': 6, 'p12': 12, 'p50': 50, 'p46': 46}   # ★2026-09-29 layout -> payload width (file meta 'layout' picks one)
DECL_ROLE_SLICE = slice(6, 10)        # p12: declared role one-hot (SIT 1..4 -> column 0..3)
DECL_POS_SLICE = slice(10, 12)        # p12: declared target position / COMM_RANGE, sender body frame (stb, fwd)
P12_DECL_WEIGHT = 4.0                 # p12 training loss weight on the 6 declaration columns (training-only, recorded in extra)
# ★2026-09-29c p50 = sender payload for the latent-dimension sweep (spec 2026-09-29-latent-sweep-design.md):
#   [0:6] own_payload · [6:8] own goal obs (d/(d+150), signed angle/180) · [8:12] declared role one-hot ·
#   [12:14] declared target position / COMM_RANGE (sender body frame) · [14:50] own radar in 36 sectors of 10 deg
#   (current frame, min over the 10 one-degree rays = nearest return in the sector as a fraction of radar range;
#   1.0 = nothing). Direct mode only. Codecs are trained on REAL states (comm_codec.py collect), not synthetic draws.
P50_GOAL = slice(6, 8)
P50_ROLE = slice(8, 12)
P50_POS = slice(12, 14)
P50_RADAR = slice(14, 50)
# ★2026-10-02 p46 = Fig1 message (spec docs/superpowers/specs/2026-10-02-fig1-latent-design.md §2):
#   [0:30] sender radar embedding from a FROZEN copy of the trunk's radar encoder (stored in the codec file, part of the SHA) ·
#   [30:36] own_payload · [36:38] intent (planned course − heading)/120, offset from goal bearing · [38:42] declared role ·
#   [42:44] declared target position / COMM_RANGE · [44:46] goal direction sin/cos. Decode only. Inputs are z-scored
#   inside the codec (mu/sd buffers from the training data). Loss weights (fixed in the spec): motion+intent x10,
#   role+target x4, goal x1, radar embedding x0.1.
P46_RADAR, P46_MOTION, P46_INTENT = slice(0, 30), slice(30, 36), slice(36, 38)
P46_ROLE, P46_POS, P46_GOAL = slice(38, 42), slice(42, 44), slice(44, 46)
P46_FIELD_W = {'radar': 0.1, 'motion': 10.0, 'intent': 10.0, 'role': 4.0, 'pos': 4.0, 'goal': 1.0}
# spec §4 codec gate (holdout, fixed before results): intent course err p90 <= 10 deg, heading err p90 <= 5 deg,
#   speed err p90 <= 0.1 m/s, declared-role accuracy >= 95 %
P46_GATE = {'intent_deg_p90': 10.0, 'heading_deg_p90': 5.0, 'speed_ms_p90': 0.1, 'role_acc': 0.95}
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


def _radar_cfg_now(frames, n_rays, out_dim, width):
    import networks as net
    return {'frames': int(frames), 'n_rays': int(n_rays), 'out_dim': int(out_dim), 'width': float(width),
            'head': str(net._RADAR_HEAD), 'leaky': bool(net._RADAR_LEAKY), 'bottleneck_ch': int(net._RADAR_BOTTLENECK_CH)}


class LatentCodec46(LatentCodec):
    """★2026-10-02 p46 codec = frozen radar-encoder copy (embeds the sender's radar stack -> 30) + z-scored AE 46 -> k.
    The radar encoder copy, mu/sd and the AE are all in state_dict -> all inside the content SHA."""

    def __init__(self, k=6, hidden=64, bits=8, radar_cfg=None):
        super().__init__(k=k, hidden=hidden, bits=bits, d_in=LAYOUTS['p46'], layout='p46')
        import networks as net
        rc = dict(radar_cfg)
        now = _radar_cfg_now(rc['frames'], rc['n_rays'], rc['out_dim'], rc['width'])
        for key in ('head', 'leaky', 'bottleneck_ch'):
            if now[key] != rc[key]:
                raise SystemExit(f"[codec] 중단: p46 레이더 인코더 {key}={rc[key]!r} != 현재 networks {now[key]!r} "
                                 "(체크포인트 스냅샷으로 networks 전역을 먼저 복원할 것)")
        self.radar_cfg = rc
        self.radar_enc = net.RadarEncoder(rc['frames'], rc['n_rays'], rc['out_dim'], rc['width'])
        self.register_buffer('mu', torch.zeros(LAYOUTS['p46']))
        self.register_buffer('sd', torch.ones(LAYOUTS['p46']))

    def meta(self):
        m = super().meta()
        m['radar_cfg'] = dict(self.radar_cfg)
        return m

    def encode(self, p):
        return self.quantize(self.enc((p - self.mu) / self.sd))

    def decode(self, z):
        return self.dec(z) * self.sd + self.mu

    def embed(self, x_flat):
        """x_flat [M, frames*n_rays] (the sender's stacked radar obs) -> [M, out_dim]."""
        return self.radar_enc(x_flat)


def decode_sender(p_hat):
    """Decoded payload [E,N,6] -> per-ship dict for vessel_gym.comm_pair_features(sender=...). Fixed rules
    (spec §3): heading = atan2(sin, cos) (atan2(0,0)=0), speeds clamped to [0, 1.8], rot and rudder to [-1, 1]."""
    import vessel_gym as vg
    o = 30 if p_hat.shape[-1] == LAYOUTS['p46'] else 0          # ★2026-10-02 p46: motion at [30:36]
    out = {
        'heading': torch.atan2(p_hat[..., o + 0], p_hat[..., o + 1]) / vg.DEG,
        'speed': p_hat[..., o + 2].clamp(0.0, 1.0) * vg.COMM_EXT_SOG_NORM,
        'rot': p_hat[..., o + 3].clamp(-1.0, 1.0),
        'cmd_rudder': p_hat[..., o + 4].clamp(-1.0, 1.0) * vg.MAX_TURN_RATE,
        'target_speed': p_hat[..., o + 5].clamp(0.0, 1.0) * vg.COMM_EXT_SOG_NORM,
    }
    if p_hat.shape[-1] == LAYOUTS['p46']:
        # ★2026-10-02 p46: intent (planned course change, sender bow frame, deg) + declaration (same rules as p12)
        out['intent_deg'] = p_hat[..., P46_INTENT.start].clamp(-1.5, 1.5) * vg.COURSE_ACT_DEG
        mx, am = p_hat[..., P46_ROLE].max(dim=-1)
        out['decl_role'] = torch.where(mx >= 0.5, am + 1, torch.zeros_like(am))
        out['decl_pos'] = p_hat[..., P46_POS].clamp(-1.0, 1.0) * float(vg.COMM_RANGE)
        return out
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
    if meta['layout'] == 'p46':
        c = LatentCodec46(k=meta['k'], hidden=meta['hidden'], bits=meta['bits'], radar_cfg=meta['radar_cfg'])
    else:
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
    if c.layout == 'p46' and mode != 'decode':
        raise SystemExit("[codec] 중단: 레이아웃 'p46' 은 'decode' 전용 (스펙 2026-10-02 §2: 56 m 밖 상대 필드를 복원값으로 채움)")
    if c.layout == 'p50' and mode != 'direct':
        raise SystemExit("[codec] 중단: 레이아웃 'p50' 은 'direct' 전용 (목표·레이더 필드용 복원 규칙 없음 — 수신 신경망이 z 를 직접 읽음)")
    # ★2026-09-29 p12 도 direct(C8) 허용 — 수신 신경망이 z(8) 를 직접 읽고 선언의 뜻·대상 매칭을 스스로 학습(저자: latent 우선).
    #   z 폭 + 자기상태 4 ≤ COMM_EXT_DIM 은 comm_gather 가 검사(k=8 → 8+4+0패딩 8 = 20).
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
    if p.shape[-1] == LAYOUTS['p50']:
        import vessel_gym as vg
        cls = lambda t: torch.where(t[..., P50_ROLE].max(-1).values >= 0.5, t[..., P50_ROLE].argmax(-1) + 1,
                                    torch.zeros(t.shape[:-1], dtype=torch.long))
        tr, dr = cls(p), cls(ph)
        out['decl_role_acc'] = float((tr == dr).double().mean())
        has = tr > 0
        perr = torch.linalg.norm((ph[:, P50_POS].clamp(-1, 1) - p[:, P50_POS]) * float(vg.COMM_RANGE), dim=-1)
        out['decl_pos_m'] = q(perr[has]) if bool(has.any()) else q(torch.zeros(1))
        out['goal_dist_ratio'] = q((ph[:, 6] - p[:, 6]).abs())
        out['goal_angle_deg'] = q((ph[:, 7] - p[:, 7]).abs() * 180.0)
        rd, rh = p[:, P50_RADAR], ph[:, P50_RADAR].clamp(0, 1)
        det = rd < 0.999
        out['radar_sector_m'] = q(((rh - rd).abs() * float(vg.RADAR_RANGE))[det]) if bool(det.any()) else q(torch.zeros(1))
        out['radar_detect_acc'] = float(((rh < 0.999) == det).double().mean())
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
                decl_weight=None, data=None):
    import vessel_gym as vg
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    if layout == 'p50':
        # ★실제 상태(collect 로 모은 파일)로 학습. 뒤섞은 뒤 90 % 학습 / 10 % holdout
        if not data:
            raise SystemExit("[codec] p50 은 --data <collect 파일> 이 필요함(실제 상태로 학습)")
        P = torch.load(_resolve(data), map_location='cpu')['payload'].float()
        perm = torch.randperm(P.shape[0], generator=gen)
        P = P[perm]
        nh = max(1, P.shape[0] // 10)
        pho, ptr = P[:nh], P[nh:]
        n_train = int(ptr.shape[0])
    else:
        _synth = {'p6': synth_payload, 'p12': synth_payload_p12}[layout]
        ptr = _synth(n_train, gen)
        pho = _synth(n_hold, gen)
    c = LatentCodec(k=k, hidden=hidden, bits=bits, d_in=LAYOUTS[layout], layout=layout)
    _w = None
    if layout == 'p12':
        _w = torch.ones(LAYOUTS['p12'])
        _w[6:] = P12_DECL_WEIGHT if decl_weight is None else float(decl_weight)
    elif layout == 'p50':
        # ★z-score MSE(값마다 학습 데이터 표준편차로 나눔, 하한 0.05) — 크기가 작은 값(선회율 등)을 통째로 버리지 않게.
        #   움직임 6·역할 선언·대상 위치는 ×decl_weight(기본 4): 상대에게 COLREGs·DCPA 에 필요한 핵심이 레이더 36칸에 묻히지 않게.
        _w = 1.0 / ptr.std(dim=0).clamp(min=0.05) ** 2
        _dw = P12_DECL_WEIGHT if decl_weight is None else float(decl_weight)
        _w[0:6] = _w[0:6] * _dw
        _w[P50_ROLE] = _w[P50_ROLE] * _dw
        _w[P50_POS] = _w[P50_POS] * _dw
    opt = torch.optim.Adam(c.parameters(), lr=1e-3)
    lv = float(2 ** bits - 1)
    for it in range(steps):
        if it == int(steps * 0.7):
            for g in opt.param_groups:
                g['lr'] = 3e-4
        idx = torch.randint(0, int(ptr.shape[0]), (batch,), generator=gen)
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
    elif layout == 'p50':
        extra['data'] = f'real states: {os.path.basename(str(data))} (comm_codec.py collect)'
        extra['loss'] = f'z-score MSE (std floor 0.05), motion+role+target x{P12_DECL_WEIGHT if decl_weight is None else decl_weight}'
        extra['hidden_note'] = f'hidden={hidden} steps={steps} batch={batch}'
    sha = content_sha(c, extra)
    return c, extra, sha, fidelity(c, pho)


def collect_p50(out, envs=8, burn=200, T=1000, seed=0):
    """★2026-09-29c real sender payloads (p50) from a rule-following fleet (goal steering; give-way/head-on/overtaking
    turn starboard 0.6; stand-on holds until 17(b) then 0.6) in the imo/open-sea env — same partner choice, role
    declaration, radar sectors and goal obs as comm_gather. Deterministic on CPU. Saves {'payload': [M,50], 'meta'}."""
    import config as cfg
    import vessel_gym as vg
    import vessel_gym_train as T_
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = vg.VesselBatchEnv(num_envs=envs, n_vessels=16, device='cpu', seed=seed, ring_scale=1.0, crossing=0,
                            risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE,
                            farfield_coef=0.0, perpair_coef=-0.15, perpair_exp=3.0)
    fs = T_.FrameStack(env.E, env.N, 'cpu')
    obs = env.reset()
    r, g, ss, st = T_.parse_obs(obs)
    fs.reset_all(r)
    out_p = []
    for t in range(burn + T):
        x = fs.get()
        if t >= burn:
            out_p.append(T_.payload50_now(env, x, g).reshape(-1, 50).clone())
        pw = env._last_pw
        tc = pw['tcpa'].gather(-1, env.danger_idx.unsqueeze(-1)).squeeze(-1)
        tg = env.goal - env.pos
        a0 = torch.clamp(vg._wrap180(torch.atan2(tg[..., 0], tg[..., 1]) / vg.DEG - env.heading) / 20.0, -1, 1)
        sit = env.situation
        a0 = torch.where((sit == 1) | (sit == 3) | (sit == 4), torch.full_like(a0, 0.6), a0)
        a0 = torch.where((sit == 2) & (tc > vg.RULE_17B_TIME), torch.zeros_like(a0), a0)
        a0 = torch.where((sit == 2) & (tc <= vg.RULE_17B_TIME), torch.full_like(a0, 0.6), a0)
        obs, _, done, _ = env.step(torch.stack([a0, torch.ones_like(a0)], -1))
        r, g, ss, st = T_.parse_obs(obs)
        fs.push(r, done)
    P = torch.cat(out_p)
    fp = _resolve(out)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    torch.save({'payload': P, 'meta': {'envs': envs, 'burn': burn, 'T': T, 'seed': seed, 'dyn_profile': str(cfg.DYN_PROFILE),
                                        'policy': 'rule fleet (goal + starboard give-way, 17b stand-on)'}}, fp)
    print(json.dumps({'out': out, 'n': int(P.shape[0]), 'role_frac': float((P[:, P50_ROLE].sum(-1) > 0).float().mean()),
                      'radar_detect_frac': float((P[:, P50_RADAR] < 0.999).float().mean())}))


# ─────────────────────────── ★2026-10-02 p46 (Fig1): collect real states → train → holdout gate ───────────────────────────
def trunk_radar_encoder(policy):
    """The control actor's radar encoder (shared with critic/msg actor under SHARED_ENCODER=all; MOE_SHARED=1 -> one
    object for all experts). Returned as-is; callers deep-copy it."""
    ca = policy.ctr_actor
    core = ca.experts[0] if hasattr(ca, 'experts') else ca.core
    return core.radar_encoder


def payload46(enc, env, x, topi, valid, part_r):
    """Sender payload [E,N,46] at decision time: frozen radar embedding of the sender's own stacked radar (x [E,N,F*S])
    + vessel_gym.own_core16 (declaration chosen among the sender's partner slots, as p12/p50)."""
    import vessel_gym as vg
    E, N = x.shape[0], x.shape[1]
    emb = enc(x.reshape(E * N, -1)).reshape(E, N, -1)
    drole, dpos = vg.role_declaration(env, topi, valid, vg.comm_pair_features(env, topi, part_r))
    core = vg.own_core16(env, drole, dpos)
    return torch.cat([emb.to(core.dtype), core], dim=-1)


def partners_within(env, R, K):
    """Nearest-K partner indices within R (same rule as vessel_gym_train.comm_gather). Returns topi [E,N,K], valid."""
    N = env.N
    d = torch.cdist(env.pos, env.pos) + torch.eye(N, device=env.pos.device).unsqueeze(0) * 1e9
    d = torch.where(d <= R, d, torch.full_like(d, 1e9))
    topd, topi = torch.topk(d, min(K, N - 1), dim=-1, largest=False)
    return topi, topd < 1e9


def collect_p46(ckpt, out, envs=16, burn=300, T=1500, seed=0, teacher='vo300i', K=8, every=2, device=None):
    """Real p46 payloads from a fleet driven by `teacher` (imitation/vo_teacher.Teacher) in the trunk checkpoint's env
    (snapshot: course action mode, reward, dyn). The radar embedding comes from a deep copy of the trunk's radar encoder,
    which is saved with the data and becomes the codec's frozen encoder."""
    import copy
    import sys as _sys
    _sys.path.insert(0, os.path.join(_HERE, 'imitation'))
    import vessel_gym as vg
    from ckpt_io import restore_policy, make_env_from_snapshot
    from vessel_gym_train import parse_obs, FrameStack
    from vo_teacher import Teacher
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    r = restore_policy(ckpt, dev, tag='[collect46]')
    enc = copy.deepcopy(trunk_radar_encoder(r.policy)).eval()
    for q in enc.parameters():
        q.requires_grad_(False)
    env = make_env_from_snapshot(r.snap, device=dev, num_envs=envs, seed=seed, n_vessels=16, tag='[collect46]')
    if vg.ACTION_MODE != 'course':
        raise SystemExit("[collect46] 중단: trunk 스냅샷 action_mode 가 course 가 아님(의도 칸이 비게 됨)")
    tch = Teacher(teacher, env)
    obs = env.reset()
    radar, goal, self_s, sit = parse_obs(obs)
    fs = FrameStack(envs, env.N, dev)
    fs.reset_all(radar)
    R = float(vg.COMM_RANGE)
    rows = []
    with torch.no_grad():
        for t in range(burn + T):
            a, _ = tch.act(env)
            if t >= burn and (t - burn) % every == 0:
                topi, valid = partners_within(env, R, K)
                rows.append(payload46(enc, env, fs.get(), topi, valid, R).reshape(-1, LAYOUTS['p46']).float().cpu())
            obs, _, done, _ = env.step(a)
            tch.after_step(env, a, done)
            radar, goal, self_s, sit = parse_obs(obs)
            fs.push(radar, done)
    P = torch.cat(rows, 0)
    rc = _radar_cfg_now(enc.frames, enc.n_rays, enc.fc.out_features, 1.0)
    torch.save({'payload': P, 'radar_enc': {k: v.cpu() for k, v in enc.state_dict().items()}, 'radar_cfg': rc,
                'meta': {'ckpt': os.path.basename(str(ckpt)), 'teacher': teacher, 'envs': envs, 'burn': burn, 'T': T,
                         'seed': seed, 'K': K, 'every': every, 'dyn_profile': str(vg._cfg.DYN_PROFILE)}}, out)
    print(f"[collect46] {P.shape[0]} payloads -> {out}", flush=True)


def fidelity46(c, P):
    """Holdout errors for the spec §4 gate. role_acc = exact class match over samples where the true OR the decoded
    declaration is non-empty (fixed 2026-10-02 before any codec was trained; plain accuracy is inflated by 'none')."""
    with torch.no_grad():
        ph = c.decode(c.encode(P))

    def ang(a, b):
        return ((a - b + 180.0) % 360.0 - 180.0).abs()
    m0 = P46_MOTION.start
    hd = ang(torch.atan2(ph[:, m0], ph[:, m0 + 1]) * 180.0 / math.pi, torch.atan2(P[:, m0], P[:, m0 + 1]) * 180.0 / math.pi)
    spd = (ph[:, m0 + 2] - P[:, m0 + 2]).abs() * 1.8
    it = (ph[:, P46_INTENT.start].clamp(-1.5, 1.5) - P[:, P46_INTENT.start]).abs() * 120.0

    def role(x):
        mx, am = x[:, P46_ROLE].max(dim=-1)
        return torch.where(mx >= 0.5, am + 1, torch.zeros_like(am))
    rt, rh = role(P), role(ph)
    sel = (rt > 0) | (rh > 0)
    acc = float((rt[sel] == rh[sel]).float().mean()) if bool(sel.any()) else float('nan')
    q = lambda v: float(v.quantile(0.9))
    return {'heading_deg_p90': q(hd), 'speed_ms_p90': q(spd), 'intent_deg_p90': q(it), 'role_acc': acc,
            'role_n': int(sel.sum()), 'n': int(P.shape[0])}


def gate46(fid):
    g = P46_GATE
    ok = (fid['intent_deg_p90'] <= g['intent_deg_p90'] and fid['heading_deg_p90'] <= g['heading_deg_p90']
          and fid['speed_ms_p90'] <= g['speed_ms_p90'] and fid['role_acc'] >= g['role_acc'])
    return bool(ok)


def train_p46(data, k=6, seed=0, hidden=64, bits=8, steps=6000, batch=4096, hold_frac=0.1):
    """AE 46 -> k on real payloads (collect_p46 file). z-score inside the codec (sd floor 0.05), field weights P46_FIELD_W."""
    import vessel_gym as vg
    blob = torch.load(_resolve(data), map_location='cpu')
    P = blob['payload'].float()
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(P.shape[0], generator=gen)
    nh = max(1, int(P.shape[0] * hold_frac))
    pho, ptr = P[perm[:nh]], P[perm[nh:]]
    torch.manual_seed(seed)
    c = LatentCodec46(k=k, hidden=hidden, bits=bits, radar_cfg=blob['radar_cfg'])
    c.radar_enc.load_state_dict(blob['radar_enc'])
    c.mu.copy_(ptr.mean(0))
    c.sd.copy_(ptr.std(0).clamp(min=0.05))
    fw = torch.ones(LAYOUTS['p46'])
    for sl, key in ((P46_RADAR, 'radar'), (P46_MOTION, 'motion'), (P46_INTENT, 'intent'), (P46_ROLE, 'role'),
                    (P46_POS, 'pos'), (P46_GOAL, 'goal')):
        fw[sl] = P46_FIELD_W[key]
    pn = (ptr - c.mu) / c.sd
    params = list(c.enc.parameters()) + list(c.dec.parameters())       # radar encoder copy stays frozen
    for q in c.radar_enc.parameters():
        q.requires_grad_(False)
    opt = torch.optim.Adam(params, lr=1e-3)
    lv = float(2 ** bits - 1)
    c.train()
    for it in range(steps):
        if it == int(steps * 0.7):
            for g in opt.param_groups:
                g['lr'] = 3e-4
        idx = torch.randint(0, int(pn.shape[0]), (batch,), generator=gen)
        x = pn[idx]
        z = c.enc(x)
        if bits > 0:
            z = (z + (torch.rand(z.shape, generator=gen) - 0.5) * (2.0 / lv)).clamp(-1.0, 1.0)
        loss = (((c.dec(z) - x) ** 2) * fw).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    c.eval()
    extra = {'dyn_profile': str(vg._cfg.DYN_PROFILE), 'seed': int(seed), 'n_train': int(ptr.shape[0]), 'steps': int(steps),
             'data': f"real states: {os.path.basename(str(data))} ({blob.get('meta')})",
             'loss': f"z-score MSE (sd floor 0.05), field weights {P46_FIELD_W}",
             'hidden_note': f'hidden={hidden} steps={steps} batch={batch}'}
    sha = content_sha(c, extra)
    return c, extra, sha, fidelity46(c, pho)


def sweep_p46(data, out_dir, ks=(4, 6, 8, 12), seed=0, steps=6000):
    """Train p46 codecs for every k, apply the spec §4 gate and the fixed selection rule:
    k = 6 if it passes, else the smallest passing of 8, 12; none -> stop (chosen_k None)."""
    os.makedirs(_resolve(out_dir), exist_ok=True)
    rep = {}
    for k in ks:
        fp = os.path.join(_resolve(out_dir), f'p46_k{k}_s{seed}.pt')
        if os.path.exists(fp):
            raise SystemExit(f"[p46] 중단: {fp} 가 이미 있음 — 덮어쓰지 않음(고정 SHA 보호)")
        c, extra, sha, fid = train_p46(data, k=k, seed=seed, steps=steps)
        meta = dict(c.meta())
        meta.update(extra)
        ok = gate46(fid)
        torch.save({'state_dict': c.state_dict(), 'meta': meta, 'sha': sha, 'fidelity_holdout': fid,
                    'gate': {'pass': ok, 'thresholds': P46_GATE}}, fp)
        rep[k] = {'path': fp, 'sha': sha, 'fidelity_holdout': fid, 'pass': ok}
        print(f"[p46] k={k} sha={sha[:16]} {fid} gate={'통과' if ok else '불통과'}", flush=True)
    chosen = 6 if rep.get(6, {}).get('pass') else next((k for k in (8, 12) if rep.get(k, {}).get('pass')), None)
    rep['chosen_k'] = chosen
    print(f"[p46] 선택 k = {chosen if chosen is not None else '없음 → 멈추고 보고(스펙 §4)'}", flush=True)
    return rep


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
    t.add_argument('--data', default=None, help='p50: collect 로 모은 실제 상태 파일')
    c_ = sub.add_parser('collect', help='p50 학습용 실제 상태 수집(규칙 배 롤아웃, CPU 1 스레드)')
    c_.add_argument('--out', required=True)
    c_.add_argument('--envs', type=int, default=8)
    c_.add_argument('--burn', type=int, default=200)
    c_.add_argument('--T', type=int, default=1000)
    c_.add_argument('--seed', type=int, default=0)
    t.add_argument('--out', required=True)
    c46 = sub.add_parser('collect46', help='★2026-10-02 p46 실제 상태 수집(trunk 체크포인트 + 선생님 함대)')
    c46.add_argument('--ckpt', required=True)
    c46.add_argument('--out', required=True)
    c46.add_argument('--envs', type=int, default=16)
    c46.add_argument('--burn', type=int, default=300)
    c46.add_argument('--T', type=int, default=1500)
    c46.add_argument('--seed', type=int, default=0)
    c46.add_argument('--teacher', default='vo300i')
    s46 = sub.add_parser('sweep46', help='★2026-10-02 p46 코덱 k=4·6·8·12 학습 + 관문 + 선택 규칙')
    s46.add_argument('--data', required=True)
    s46.add_argument('--out_dir', default='comm_codecs')
    s46.add_argument('--seed', type=int, default=0)
    s46.add_argument('--steps', type=int, default=6000)
    s46.add_argument('--report', default=None)
    i = sub.add_parser('info')
    i.add_argument('path')
    a = ap.parse_args()
    if a.cmd == 'collect46':
        collect_p46(a.ckpt, a.out, envs=a.envs, burn=a.burn, T=a.T, seed=a.seed, teacher=a.teacher)
        return
    if a.cmd == 'sweep46':
        rep_ = sweep_p46(a.data, a.out_dir, seed=a.seed, steps=a.steps)
        if a.report:
            with open(a.report, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in rep_.items()}, f, ensure_ascii=False, indent=1)
        print(f"CHOSEN_K={rep_['chosen_k']}", flush=True)
        return
    if a.cmd == 'train':
        fp = _resolve(a.out)
        if os.path.exists(fp):
            raise SystemExit(f"[codec] 중단: {fp} 가 이미 있음 — 덮어쓰지 않음(고정 SHA 보호). 지우고 다시 하려면 직접 지울 것")
        c, extra, sha, fid = train_codec(k=a.k, seed=a.seed, bits=a.bits, steps=a.steps, layout=a.layout,
                                         hidden=a.hidden, decl_weight=a.decl_weight, data=a.data)
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
    elif a.cmd == 'collect':
        collect_p50(a.out, envs=a.envs, burn=a.burn, T=a.T, seed=a.seed)
    else:
        blob = torch.load(_resolve(a.path), map_location='cpu')
        print(json.dumps({'sha': blob['sha'], 'meta': blob['meta'], 'fidelity_holdout': blob.get('fidelity_holdout'),
                          'numeric_ref': blob.get('numeric_ref')}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    _main()
