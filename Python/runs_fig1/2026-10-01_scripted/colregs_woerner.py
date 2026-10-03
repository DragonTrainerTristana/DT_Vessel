"""colregs_woerner.py — encounter-based COLREGs compliance score after Woerner et al. 2019 (2026-10-02).

Reference: K. Woerner, M. R. Benjamin, M. Novitzky, J. J. Leonard, "Quantifying protocol evaluation for autonomous
collision avoidance", Autonomous Robots 43:967-991, 2019 (Algorithms 4-14). Author approval 2026-10-02 ("ㅇㅋ 진행"):
this score is the primary COLREGs metric from the next batch on; the old in-radar rudder-fraction metric stays as a
secondary line. Paper defaults are used as written; values the paper leaves open are fixed here BEFORE any rescoring
and marked (*).

Encounter (per ordered pair i→j, scored from ship i's side):
  start  the first decision with dist <= R_ENC (300 m, same for every arm), approaching (raw_tcpa >= 0), dcpa < 24 m
         (env DCPA_RISK) and role_i != none (env COLREGs roles, RolePromiseTrackerV2.roles at R_ENC). Role is fixed at start.
         r_detect = range at start, h0 / v0 = own heading / speed at start.
  end    past CPA and opening (raw_tcpa < 0 and dist > r_cpa + 0.5 m), or dist > 1.1 R_ENC, or i or j terminates.
         Encounters started during burn-in or still open at the end of the run are not counted.
Scores (1 = fully compliant):
  safety S      = clip((r_cpa − R_NM) / (R_PREF − R_NM), 0, 1); 0 if i collided with j.
                  (*) R_NM = 12 m (env SAFE_PASSING, Rule 8(d)), R_PREF = 24 m (env DCPA_RISK)
  delay  D      = 1 − 0.5 · (r_detect − r_man) / r_detect, r_man = range when |heading change| first >= 2°,
                  capped at r_detect (acting before the encounter = no delay); no maneuver → r_man = 0.   (Alg. 5; (*) 50 %)
  apparent A    = 1 if course penalty < 30 % or speed penalty < 30 %, else 1 − min(course, speed penalty);
                  course penalty = 0.5 · max(0, 30° − |Δψ|max) / 30°, speed penalty = 0.5 · max(0, 0.5 − Δv) / 0.5,
                  Δv = (v0 − vmin) / v0.                                                      (Alg. 6-8, defaults)
  direction T   = 0.5 if the turn was mainly to port (−min Δψ > max Δψ and −min Δψ >= 2°), else 1.   ((*) 50 %)
  pose P        = ((sin β_cpa − 1)/2)² · ((sin α_cpa − 1)/2)², β = bearing of j from i, α = bearing of i from j
                  (port-to-port passing → 1).                                                 (Eq. 13)
  stand-on C    = 1 − 0.5 · clip((|Δψ|max − 2°)/(30° − 2°), 0, 1)                            (Alg. 10, defaults)
                  excused (C = 1) if i had a give-way type duty to another ship during the encounter ("maneuvering for
                  another COLREGS obligation", Sec. 4.3) or the first >= 2° change came at tcpa <= RULE_17B_TIME
                  (Rule 17(a)(ii)/(b)); in that late case a port turn still gets T = 0.5 (Rule 17(c)).
  stand-on V    = speed: no penalty if slow-down (v0 − vmin) < 0.2 m/s, else 1 − 0.5 · (v0 − vmin)/v0.  (Alg. 11 slowing term)
                  (*) speed-up is NOT penalized: every ship spawns at 0.2–0.5 of its max speed and accelerates to cruise for
                  ~100 s (ACCEL 0.01), i.e. normal navigation (Alg. 9 line 6), and scripted ships always use full thrust.
  Also reported: C_raw = stand-on course score without the 'other duty' / 'late' excuses (transparency).
  ★2026-10-02 Rule 8 "a succession of small alterations of course and/or speed should be avoided" (Woerner Sec. 4.3/4.5:
  "several small turns resulting in a larger effective turn should also be penalized"), added after the first rescoring
  showed constantly turning learned ships scoring highest (author "ㅇㅋ 진행"):
  small-alteration W = 1 − 0.5 · clip(travel / (2 · max(|Δψ|max, 2°)) − 1, 0, 1), travel = Σ|heading change per decision|
                  over the encounter. One clean turn-out-and-back gives travel ≈ 2 · |Δψ|max → W = 1; travel ≥ 4 · |Δψ|max → W = 0.5.
                  (*) ratio band 1–2 and 50 % max. Applied to every role (stand-on only when |Δψ|max >= 2°).
  give-way (crossing)  = S · D · A · T · W      head-on = S · D · A · T · P · W      overtaking = S · D · A · W
  stand-on (crossing or overtaken) = S · C · V · T · W   (W = Rule 8 small alterations, below; added 2026-10-02)
Also reported: bow-crossing share of give-way encounters (i passed ahead of j: |bearing of i from j's bow| < 90° at CPA).
"""

import torch

R_ENC = 300.0
R_NM, R_PREF = 12.0, 24.0
MAN_DEG = 2.0
APP_DEG = 30.0


def build(vg):
    SIT_HEADON, SIT_STANDON, SIT_GIVEWAY, SIT_OVERTAKING = vg.SIT_HEADON, vg.SIT_STANDON, vg.SIT_GIVEWAY, vg.SIT_OVERTAKING
    wrap = vg._wrap180
    DEG = vg.DEG
    ROLES = {SIT_GIVEWAY: 'giveway', SIT_HEADON: 'headon', SIT_OVERTAKING: 'overtaking', SIT_STANDON: 'standon'}

    class WoernerTracker:
        def __init__(self, E, N, device, burnin):
            self.E, self.N, self.burnin = E, N, burnin
            zf = lambda v=0.0: torch.full((E, N, N), v, device=device)
            self.act = torch.zeros(E, N, N, dtype=torch.bool, device=device)
            self.count = torch.zeros(E, N, N, dtype=torch.bool, device=device)
            self.role = torch.zeros(E, N, N, dtype=torch.long, device=device)
            self.r0, self.h0, self.v0 = zf(), zf(), zf()
            self.dmax, self.dmin, self.vmin, self.vmax = zf(), zf(), zf(), zf()
            self.rman, self.tcman = zf(-1.0), zf(-1.0)
            self.rcpa, self.bcpa, self.ocpa = zf(1e9), zf(), zf()
            self.multi = torch.zeros(E, N, N, dtype=torch.bool, device=device)
            self.eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
            self.near = torch.zeros(E, N, dtype=torch.long, device=device)
            self.trav = zf()                                    # Σ|Δheading| per decision during the encounter (Rule 8)
            self.hprev = None
            self.acc = {}           # role name -> dict of sums

        def _bearings(self, env):
            to_other = env.pos[:, None, :, :] - env.pos[:, :, None, :]
            dx, dz = to_other[..., 0], to_other[..., 1]
            h = env.heading * DEG
            fx, fz = torch.sin(h)[:, :, None], torch.cos(h)[:, :, None]
            gx, gz = torch.sin(h)[:, None, :], torch.cos(h)[:, None, :]
            b = torch.atan2(fz * dx - fx * dz, fx * dx + fz * dz) / DEG              # j seen from i (+ starboard)
            ob = torch.atan2(gx * dz - gz * dx, -(gx * dx + gz * dz)) / DEG          # i seen from j
            return b, ob

        def observe(self, env, t):
            """Call with the decision-time state s_t (before env.step)."""
            pw = env._last_pw
            dist, raw, tcpa, dcpa = pw['dist'], pw['raw_tcpa'], pw['tcpa'], pw['dcpa']
            hi = env.heading[:, :, None].expand(-1, -1, self.N)
            if self.hprev is not None:
                step_turn = wrap(env.heading - self.hprev).abs()[:, :, None].expand(-1, -1, self.N)
                self.trav = torch.where(self.act, self.trav + step_turn, self.trav)
            self.hprev = env.heading.clone()
            vi = env.speed[:, :, None].expand(-1, -1, self.N)
            b, ob = self._bearings(env)
            a = self.act
            dev = wrap(hi - self.h0)
            self.dmax = torch.where(a, torch.maximum(self.dmax, dev), self.dmax)
            self.dmin = torch.where(a, torch.minimum(self.dmin, dev), self.dmin)
            self.vmin = torch.where(a, torch.minimum(self.vmin, vi), self.vmin)
            self.vmax = torch.where(a, torch.maximum(self.vmax, vi), self.vmax)
            first = a & (self.rman < 0) & (dev.abs() >= MAN_DEG)
            self.rman = torch.where(first, dist, self.rman)
            self.tcman = torch.where(first, tcpa, self.tcman)
            closer = a & (dist < self.rcpa)
            self.rcpa = torch.where(closer, dist, self.rcpa)
            self.bcpa = torch.where(closer, b, self.bcpa)
            self.ocpa = torch.where(closer, ob, self.ocpa)
            giver = (self.role == SIT_GIVEWAY) | (self.role == SIT_HEADON) | (self.role == SIT_OVERTAKING)
            n_give = (a & giver).sum(-1, keepdim=True)                                # i's active give-way type duties
            self.multi = self.multi | (a & (self.role == SIT_STANDON) & (n_give > 0))
            # geometric end: past CPA and opening, or out of range
            end = a & (((raw < 0) & (dist > self.rcpa + 0.5)) | (dist > 1.1 * R_ENC))
            self._finish(end, torch.zeros_like(end))
            # start
            role_i, _ = vg.RolePromiseTrackerV2.roles(env, pw, R_ENC)
            start = (~self.act) & (~self.eye) & (dist <= R_ENC) & (raw >= 0) & (dcpa < vg.DCPA_RISK) & (role_i > 0)
            self.act = self.act | start
            self.count = torch.where(start, torch.full_like(self.count, t >= self.burnin), self.count)
            self.role = torch.where(start, role_i, self.role)
            self.r0 = torch.where(start, dist, self.r0)
            self.h0 = torch.where(start, hi, self.h0)
            self.v0 = torch.where(start, vi, self.v0)
            self.vmin = torch.where(start, vi, self.vmin)
            self.vmax = torch.where(start, vi, self.vmax)
            self.dmax = torch.where(start, torch.zeros_like(self.dmax), self.dmax)
            self.trav = torch.where(start, torch.zeros_like(self.trav), self.trav)
            self.dmin = torch.where(start, torch.zeros_like(self.dmin), self.dmin)
            self.rman = torch.where(start, torch.full_like(self.rman, -1.0), self.rman)
            self.tcman = torch.where(start, torch.full_like(self.tcman, -1.0), self.tcman)
            self.rcpa = torch.where(start, dist, self.rcpa)
            self.bcpa = torch.where(start, b, self.bcpa)
            self.ocpa = torch.where(start, ob, self.ocpa)
            self.multi = torch.where(start, torch.zeros_like(self.multi), self.multi)
            dd = dist.masked_fill(self.eye, 1e9)
            self.near = dd.argmin(-1)                                                  # nearest ship at s_t (collision partner)

        def after_step(self, done, outcome):
            """Call after env.step: close encounters of ships that terminated (collision with the nearest ship → S = 0)."""
            di = done[:, :, None].expand(-1, -1, self.N)
            dj = done[:, None, :].expand(-1, self.N, -1)
            end = self.act & (di | dj)
            coll_i = (outcome == vg.OUT_COLLISION_VESSEL)
            jidx = torch.arange(self.N, device=done.device).view(1, 1, -1)
            iidx = torch.arange(self.N, device=done.device).view(1, -1, 1)
            hit_ij = coll_i[:, :, None] & (self.near[:, :, None] == jidx)              # [i,j]: i collided and j was i's nearest
            hit_ji = coll_i[:, None, :] & (self.near[:, None, :] == iidx)              # [i,j]: j collided and i was j's nearest
            self._finish(end, end & (hit_ij | hit_ji))

        def _finish(self, end, collided):
            m = end & self.count
            if bool(m.any()):
                maxabs = torch.maximum(self.dmax, -self.dmin)
                W = 1.0 - 0.5 * (self.trav / (2.0 * maxabs.clamp(min=MAN_DEG)) - 1.0).clamp(0, 1)   # Rule 8 small alterations
                Ws = torch.where(maxabs < MAN_DEG, torch.ones_like(W), W)
                S = ((self.rcpa - R_NM) / (R_PREF - R_NM)).clamp(0, 1)
                S = torch.where(collided, torch.zeros_like(S), S)
                rman = torch.where(self.rman < 0, torch.zeros_like(self.rman), torch.minimum(self.rman, self.r0))
                D = 1.0 - 0.5 * (self.r0 - rman) / self.r0.clamp(min=1e-6)
                pc = 0.5 * (APP_DEG - maxabs).clamp(min=0) / APP_DEG
                dv = ((self.v0 - self.vmin) / self.v0.clamp(min=1e-6)).clamp(min=0)
                pv = 0.5 * (0.5 - dv).clamp(min=0) / 0.5
                A = torch.where((pc < 0.3) | (pv < 0.3), torch.ones_like(pc), 1.0 - torch.minimum(pc, pv))
                port = (-self.dmin > self.dmax) & (-self.dmin >= MAN_DEG)
                T = torch.where(port, torch.full_like(pc, 0.5), torch.ones_like(pc))
                sb, so = torch.sin(self.bcpa * DEG), torch.sin(self.ocpa * DEG)
                P = ((sb - 1.0) / 2.0) ** 2 * ((so - 1.0) / 2.0) ** 2
                late = (self.tcman >= 0) & (self.tcman <= vg.RULE_17B_TIME)
                C = 1.0 - 0.5 * ((maxabs - MAN_DEG) / (APP_DEG - MAN_DEG)).clamp(0, 1)
                Craw = C
                C = torch.where(self.multi | late, torch.ones_like(C), C)
                Ts = torch.where(late & port, torch.full_like(pc, 0.5), torch.ones_like(pc))
                slow = (self.v0 - self.vmin).clamp(min=0)
                V = torch.where(slow < 0.2, torch.ones_like(pc), (1.0 - 0.5 * slow / self.v0.clamp(min=1e-6)).clamp(0, 1))
                bow = self.ocpa.abs() < 90.0
                score = torch.where(self.role == SIT_GIVEWAY, S * D * A * T * W,
                        torch.where(self.role == SIT_HEADON, S * D * A * T * P * W,
                        torch.where(self.role == SIT_OVERTAKING, S * D * A * W, S * C * V * Ts * Ws)))
                Wr = torch.where(self.role == SIT_STANDON, Ws, W)
                for rv, nm in ROLES.items():
                    k = m & (self.role == rv)
                    n = float(k.sum())
                    if n == 0:
                        continue
                    a = self.acc.setdefault(nm, {'n': 0.0, 'score': 0.0, 'S': 0.0, 'D': 0.0, 'A': 0.0, 'T': 0.0, 'P': 0.0,
                                                 'C': 0.0, 'Craw': 0.0, 'V': 0.0, 'R8': 0.0, 'bow': 0.0, 'multi': 0.0, 'coll': 0.0, 'man': 0.0})
                    a['n'] += n
                    for key, val in (('score', score), ('S', S), ('D', D), ('A', A), ('T', T if nm != 'standon' else Ts),
                                     ('P', P), ('C', C), ('Craw', Craw), ('V', V), ('R8', Wr), ('bow', bow.float()), ('multi', self.multi.float()),
                                     ('coll', collided.float()), ('man', (self.rman >= 0).float())):
                        a[key] += float(val[k].sum())
            self.act = self.act & ~end

        def report(self):
            tot_n = sum(a['n'] for a in self.acc.values())
            tot_s = sum(a['score'] for a in self.acc.values())
            pc = lambda a, k: 100.0 * a[k] / a['n'] if a['n'] else float('nan')
            head = (f"   [woerner-COLREGs] score={100.0 * tot_s / tot_n if tot_n else float('nan'):5.1f}% (n={tot_n:.0f} 조우, "
                    f"시작 = 300 m 안 dcpa<24 m 접근, 두 팔 같은 기준; Woerner 2019)")
            parts = []
            for nm in ('giveway', 'headon', 'overtaking', 'standon'):
                a = self.acc.get(nm)
                if not a:
                    parts.append(f"{nm}=n/a")
                    continue
                if nm == 'standon':
                    parts.append(f"{nm}={pc(a, 'score'):5.1f}%(n={a['n']:.0f} S={pc(a, 'S'):.1f} C={pc(a, 'C'):.1f} C_raw={pc(a, 'Craw'):.1f} V={pc(a, 'V'):.1f} R8={pc(a, 'R8'):.1f} "
                                 f"T={pc(a, 'T'):.1f} 다른의무={pc(a, 'multi'):.1f}% 변침={pc(a, 'man'):.1f}%)")
                else:
                    parts.append(f"{nm}={pc(a, 'score'):5.1f}%(n={a['n']:.0f} S={pc(a, 'S'):.1f} D={pc(a, 'D'):.1f} A={pc(a, 'A'):.1f} R8={pc(a, 'R8'):.1f} "
                                 f"T={pc(a, 'T'):.1f}" + (f" P={pc(a, 'P'):.1f}" if nm == 'headon' else '')
                                 + f" 선수횡단={pc(a, 'bow'):.1f}% 충돌={pc(a, 'coll'):.1f}%)")
            return head + "\n   [woerner-COLREGs/역할] " + ' '.join(parts)

    return WoernerTracker
