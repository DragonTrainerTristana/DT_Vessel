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
★2026-11-04 binary compliance (author request; spec 2026-11-02 §9, fixed before any binary value was computed): same encounters,
  one yes/no per encounter, window = the whole encounter from its 300 m start, threshold BIN_DEG = 10°:
  give-way / head-on  = (max starboard heading change >= 10° or speed cut >= 50 %) and max port change <= 10° and no collision
  stand-on            = |heading change|max <= 10° (excused: late action at tcpa <= RULE_17B_TIME or another give-way duty, as C) and no collision
  overtaking          = no collision and closest range >= R_NM (12 m)
★2026-11-09 diagnostics (author OK 11-09 'Windows 명령어'; eval-only, --colregs_diag, diag=False = no extra state, all
  lines above unchanged): why give-way / head-on encounters fail the binary rule, where the first >= 10° turns happen,
  and whether heading changes follow neighbour-list changes. No definition or verdict changes; exploratory only.
  failure class (give-way / head-on, % of those encounters): collision | port > 10° toward goal (at the first port > 10°
  decision the heading points closer to the goal bearing than h0 did) | port > 10° other | no >= 10° starboard or 50 %
  speed cut and no port > 10° (split by closest range >= R_PREF). Also: start range r0, first starboard / port >= 10°
  range (> 56 m share), closed by ship termination (arrival / other collision / timeout) share.
  jitter: per ship-decision |Δheading| after burn-in (ships that respawned on the previous step excluded), split by
  'radar empty' (no ship within 56 m) and by whether the 300 m nearest-8 list or the 56 m list changed since the last
  decision. Lists are geometric (true positions), the same for both arms — not the network's message inputs.
"""

import torch

R_ENC = 300.0
R_NM, R_PREF = 12.0, 24.0
MAN_DEG = 2.0
APP_DEG = 30.0
BIN_DEG = 10.0     # ★2026-11-04 binary compliance threshold (spec 2026-11-02 §9)
R_RADAR, R_LIST, K_LIST = 56.0, 300.0, 8   # ★2026-11-09 diag only: radar range, comm range, max partners (eval header max_partners=8)


def build(vg):
    SIT_HEADON, SIT_STANDON, SIT_GIVEWAY, SIT_OVERTAKING = vg.SIT_HEADON, vg.SIT_STANDON, vg.SIT_GIVEWAY, vg.SIT_OVERTAKING
    wrap = vg._wrap180
    DEG = vg.DEG
    ROLES = {SIT_GIVEWAY: 'giveway', SIT_HEADON: 'headon', SIT_OVERTAKING: 'overtaking', SIT_STANDON: 'standon'}

    class WoernerTracker:
        def __init__(self, E, N, device, burnin, diag=False):
            self.E, self.N, self.burnin, self.diag = E, N, burnin, diag
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
            if diag:                # ★2026-11-09 diagnostics (header)
                self.rsb, self.rport = zf(-1.0), zf(-1.0)       # range at the first starboard / port change >= BIN_DEG
                self.pgoal = torch.zeros(E, N, N, dtype=torch.bool, device=device)
                self.prev300 = self.prev56 = self.fresh = None
                self.dacc, self.jit = {}, {}

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
            if self.diag:
                self._jitter(env, dist, t)
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
            if self.diag:
                g = env.goal - env.pos
                gb = (torch.atan2(g[..., 0], g[..., 1]) / DEG)[:, :, None].expand(-1, -1, self.N)   # goal bearing (compass)
                fsb = a & (self.rsb < 0) & (dev >= BIN_DEG)
                fpt = a & (self.rport < 0) & (dev <= -BIN_DEG)
                self.rsb = torch.where(fsb, dist, self.rsb)
                self.rport = torch.where(fpt, dist, self.rport)
                self.pgoal = torch.where(fpt, wrap(gb - hi).abs() < wrap(gb - self.h0).abs(), self.pgoal)
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
            if self.diag:
                self.rsb = torch.where(start, torch.full_like(self.rsb, -1.0), self.rsb)
                self.rport = torch.where(start, torch.full_like(self.rport, -1.0), self.rport)
                self.pgoal = torch.where(start, torch.zeros_like(self.pgoal), self.pgoal)
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
            if self.diag:
                self.fresh = done.clone()                       # respawned this step → skip its next |Δheading|
            self._finish(end, end & (hit_ij | hit_ji), term=end)

        def _jitter(self, env, dist, t):
            """★2026-11-09 diag: |Δheading| per ship-decision vs neighbour-list changes (header)."""
            dd = dist.masked_fill(self.eye, float('inf'))
            lists = []
            for r in (R_LIST, R_RADAR):
                v, ix = dd.masked_fill(dd > r, float('inf')).topk(min(K_LIST, self.N - 1), dim=-1, largest=False)
                m = torch.zeros(self.E, self.N, self.N, dtype=torch.bool, device=dist.device)
                lists.append(m.scatter(-1, ix, torch.isfinite(v)))
            if self.prev300 is not None and self.hprev is not None and t >= self.burnin:
                ok = ~self.fresh if self.fresh is not None else torch.ones_like(env.heading, dtype=torch.bool)
                turn = wrap(env.heading - self.hprev).abs()
                empty = ~(dd <= R_RADAR).any(-1)
                c300 = (lists[0] != self.prev300).any(-1)
                c56 = (lists[1] != self.prev56).any(-1)
                for key, msk in (('all', ok), ('empty', ok & empty), ('empty_c300', ok & empty & c300),
                                 ('empty_s300', ok & empty & ~c300), ('c56', ok & c56), ('s56', ok & ~c56)):
                    s = self.jit.setdefault(key, [0.0, 0.0])
                    s[0] += float(msk.sum())
                    s[1] += float(turn[msk].sum())
            self.prev300, self.prev56 = lists

        def _finish(self, end, collided, term=None):
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
                # ★2026-11-04 binary compliance (§9): one yes/no per encounter over the whole 300 m window
                give_ok = ((self.dmax >= BIN_DEG) | (dv >= 0.5)) & (-self.dmin <= BIN_DEG)
                so_ok = (maxabs <= BIN_DEG) | self.multi | late
                ov_ok = self.rcpa >= R_NM
                binok = torch.where((self.role == SIT_GIVEWAY) | (self.role == SIT_HEADON), give_ok,
                        torch.where(self.role == SIT_OVERTAKING, ov_ok, so_ok)) & (~collided)
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
                                                 'C': 0.0, 'Craw': 0.0, 'V': 0.0, 'R8': 0.0, 'bow': 0.0, 'multi': 0.0, 'coll': 0.0, 'man': 0.0,
                                                 'bin': 0.0})
                    a['n'] += n
                    for key, val in (('score', score), ('S', S), ('D', D), ('A', A), ('T', T if nm != 'standon' else Ts),
                                     ('P', P), ('C', C), ('Craw', Craw), ('V', V), ('R8', Wr), ('bow', bow.float()), ('multi', self.multi.float()),
                                     ('coll', collided.float()), ('man', (self.rman >= 0).float()), ('bin', binok.float())):
                        a[key] += float(val[k].sum())
                if self.diag:
                    self._diag_finish(m, collided, term, dv, late, binok)
            self.act = self.act & ~end

        def _diag_finish(self, m, collided, term, dv, late, binok):
            """★2026-11-09 diag: failure classes and first-turn ranges (header). Sums only, no per-encounter storage."""
            term = torch.zeros_like(m) if term is None else term
            port = (-self.dmin > BIN_DEG) & ~collided
            act10 = (self.dmax >= BIN_DEG) | (dv >= 0.5)
            give = (self.role == SIT_GIVEWAY) | (self.role == SIT_HEADON)
            so = (self.role == SIT_STANDON) & ~self.multi & ~late
            rows = {
                'gw_n': m & give, 'gw_fail': m & give & ~binok, 'gw_coll': m & give & collided,
                'gw_port_goal': m & give & port & self.pgoal, 'gw_port_other': m & give & port & ~self.pgoal,
                'gw_small_safe': m & give & ~collided & ~port & ~act10 & (self.rcpa >= R_PREF),
                'gw_small_close': m & give & ~collided & ~port & ~act10 & (self.rcpa < R_PREF),
                'gw_port_far': m & give & port & (self.rport > R_RADAR),
                'gw_port_after_sb': m & give & port & (self.rsb > self.rport),          # starboard >= 10° came first (larger range)
                'gw_term': m & give & term & ~collided, 'gw_r0_far': m & give & (self.r0 > R_RADAR),
                'gw_sb': m & give & (self.rsb >= 0), 'gw_sb_far': m & give & (self.rsb > R_RADAR),
                'so_n': m & so, 'so_fail': m & so & ~binok, 'so_coll': m & so & collided,
                'so_port': m & so & ~collided & (-self.dmin > BIN_DEG), 'so_sb': m & so & ~collided & (self.dmax > BIN_DEG),
            }
            for key, msk in rows.items():
                self.dacc[key] = self.dacc.get(key, 0.0) + float(msk.sum())
            for key, val, msk in (('gw_r0_sum', self.r0, rows['gw_n']), ('gw_sb_r_sum', self.rsb, rows['gw_sb'])):
                self.dacc[key] = self.dacc.get(key, 0.0) + float(val[msk].sum())

        def diag_report(self):
            """★2026-11-09 diag lines (exploratory, not a verdict)."""
            d, j = self.dacc, self.jit
            p = lambda k, n: 100.0 * d.get(k, 0.0) / d[n] if d.get(n) else float('nan')
            mj = lambda k: (j[k][1] / j[k][0], j[k][0]) if j.get(k) and j[k][0] else (float('nan'), 0.0)
            gw = (f"   [colregs-diag/양보·정면 탈락] n={d.get('gw_n', 0):.0f} 탈락={p('gw_fail', 'gw_n'):.1f}% = 충돌 {p('gw_coll', 'gw_n'):.1f}"
                  f" + 좌현>10°(목표쪽) {p('gw_port_goal', 'gw_n'):.1f} + 좌현>10°(기타) {p('gw_port_other', 'gw_n'):.1f}"
                  f" + 우현·감속<문턱·좌현 없음(통과≥24m {p('gw_small_safe', 'gw_n'):.1f} / <24m {p('gw_small_close', 'gw_n'):.1f}) %"
                  f" | 좌현>10° 첫 시점 56 m 밖 {p('gw_port_far', 'gw_n'):.1f}% · 우현≥10° 뒤 좌현 {p('gw_port_after_sb', 'gw_n'):.1f}%"
                  f" | 종료(도착·다른 충돌·시간초과)로 닫힘 {p('gw_term', 'gw_n'):.1f}% (모두 조우 대비 %)")
            first = (f"   [colregs-diag/첫 변침] 양보·정면 시작거리 평균 {d.get('gw_r0_sum', 0) / d['gw_n'] if d.get('gw_n') else float('nan'):.1f} m"
                     f" (56 m 밖 시작 {p('gw_r0_far', 'gw_n'):.1f}%) | 우현≥10° 있음 {p('gw_sb', 'gw_n'):.1f}%"
                     f" (56 m 밖 {p('gw_sb_far', 'gw_n'):.1f}%, 평균 거리 {d.get('gw_sb_r_sum', 0) / d['gw_sb'] if d.get('gw_sb') else float('nan'):.1f} m)")
            so = (f"   [colregs-diag/유지 탈락, 면제 없는 조우] n={d.get('so_n', 0):.0f} 탈락={p('so_fail', 'so_n'):.1f}% = 충돌 {p('so_coll', 'so_n'):.1f}"
                  f" · 좌현>10° {p('so_port', 'so_n'):.1f} · 우현>10° {p('so_sb', 'so_n'):.1f} % (좌·우 겹칠 수 있음)")
            jl = [f"{lbl} {mj(k)[0]:.2f}°(n={mj(k)[1]:.0f})" for k, lbl in
                  (('all', '전체'), ('empty', '레이더 빔'), ('empty_c300', '빔+300m목록 바뀜'), ('empty_s300', '빔+안 바뀜'),
                   ('c56', '56m목록 바뀜'), ('s56', '56m 안 바뀜'))]
            jit = "   [colregs-diag/흔들림] 배-결정당 평균 |Δheading| " + ' · '.join(jl) + " (목록 = 실제 위치 기준 nearest-8, 두 팔 같음)"
            return '\n'.join((gw, first, so, jit))

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
            tot_b = sum(a.get('bin', 0.0) for a in self.acc.values())
            kr = {'giveway': '양보', 'headon': '정면', 'overtaking': '추월', 'standon': '유지'}
            bparts = ' '.join(f"{kr[nm]}={pc(self.acc[nm], 'bin'):5.1f}%(n={self.acc[nm]['n']:.0f})" for nm in kr if nm in self.acc)
            binl = (f"\n   [binary-COLREGs] rate={100.0 * tot_b / tot_n if tot_n else float('nan'):5.1f}% (n={tot_n:.0f} 조우, 300 m 조우 전체, "
                    f"문턱 {BIN_DEG:.0f}°, 스펙 2026-11-02 §9) {bparts}")
            return head + "\n   [woerner-COLREGs/역할] " + ' '.join(parts) + binl

    return WoernerTracker
