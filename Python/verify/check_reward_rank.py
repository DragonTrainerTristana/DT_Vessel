"""check_reward_rank.py — 학습 전 보상 순위 게이트 G6 (2026-09-29, 스펙 2026-09-29-role-promise-design.md §5).

질문: 보상이 '실제로 피하는 배'에 '안 피하는 배'보다 높은 점수를 주는가? 학습 없이, 규칙 정책 6개를 학습기와 같은 env·보상으로
굴려 할인 반환(γ = config.DISCOUNT_FACTOR)을 비교한다. 보상 설정마다 새 서브프로세스(모듈 전역·env 가 섞이지 않게).

  정책: goal(목표로 직진) · stbd(레이더 안 상황이 있으면 무조건 우현 — 학습 OFF 의 명령 패턴) · radar(레이더 안 규칙:
        양보·정면·추월 우현, 유지 17(b) 전 유지) · vo56 / vo56s(56 m 안 상대 위치·속도로 궤적 예측 회피, s = 우현만) ·
        seeker(300 m 안 가까운 배 쪽으로 틀어 충돌코스를 만든 뒤 레이더 규칙으로 비킴 = 조우 유도)
  보상: old = 배치 X 와 같은 옛 보상 / new = VESSEL_ROLE_PROMISE_PEN=20
  판정(new 만, 시드마다): min(G[vo56], G[vo56s]) − max(G[goal·stbd·radar·seeker]) > 0.05·|max(...)|  (부호 무관 여유)
  old 결과는 기록만(옛 보상은 이 순서가 뒤집혀 있을 수 있음 — 그게 이번 수정의 이유).

  python verify/check_reward_rank.py --out <json>          # 마지막 줄 'RANK GATE: PASS' 여야 함. rc 0/1
  env: 학습 배치와 같은 VESSEL_*(imo·none·COMM_RANGE 300 …) 를 부모에서 물려받는다. ROLE_PROMISE_PEN 만 설정별로 덮어씀.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
POLICIES = ('goal', 'stbd', 'radar', 'vo56', 'vo56s', 'seeker')
AVOIDERS, OTHERS = ('vo56', 'vo56s'), ('goal', 'stbd', 'radar', 'seeker')


# ─────────────────────────── child: 한 보상 설정에서 정책 × 시드 ───────────────────────────
def _child(a):
    sys.path.insert(0, PYROOT)
    import torch
    import config as cfg
    import vessel_gym as vg
    assert cfg.DYN_PROFILE == 'imo' and cfg.OBSTACLES_MODE == 'none', (cfg.DYN_PROFILE, cfg.OBSTACLES_MODE)
    torch.set_num_threads(1)
    dev = torch.device(a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    DEG = vg.DEG
    wrap = vg._wrap180

    def goal_bearing(env):
        tg = env.goal - env.pos
        return torch.atan2(tg[..., 0], tg[..., 1]) / DEG

    def goal_a0(env):
        return torch.clamp(wrap(goal_bearing(env) - env.heading) / 20.0, -1, 1)

    def radar_rule(env, a0):
        pw = env._last_pw
        tc = pw['tcpa'].gather(-1, env.danger_idx.unsqueeze(-1)).squeeze(-1)
        sit = env.situation
        gw = (sit == 1) | (sit == 3) | (sit == 4)
        so = sit == 2
        a0 = torch.where(gw, torch.full_like(a0, 0.6), a0)
        a0 = torch.where(so & (tc > vg.RULE_17B_TIME), torch.zeros_like(a0), a0)
        return torch.where(so & (tc <= vg.RULE_17B_TIME), torch.full_like(a0, 0.6), a0)

    def vo_action(env, R, prev, stbd_only, D_safe=30.0, H=60.0, dt=2.0):
        # 스크래치 fleet_vo.vo_action(2026-09-28, 56 m 에서 충돌 6.5–8.6 %) 을 device 인자만 붙여 옮김
        E, N = env.E, env.N
        cands = torch.arange(0.0 if stbd_only else -120.0, 121.0, 15.0, device=dev)
        C = cands.numel()
        gb = goal_bearing(env)
        psi_c = gb.unsqueeze(-1) + cands
        v = env.speed.clamp(min=0.3)
        omega = (v / vg.R_FULL) / DEG if vg.DYN_FORMULA == 'abs' else vg.MAX_YAW_RATE * (env.speed / env.max_speed.clamp(min=1e-6))
        psi = env.heading.unsqueeze(-1).expand(E, N, C).clone()
        p = env.pos.unsqueeze(2).expand(E, N, C, 2).clone()
        h = env.heading * DEG
        vel = torch.stack([torch.sin(h), torch.cos(h)], -1) * env.speed.unsqueeze(-1)
        dist0 = torch.linalg.norm(env.pos.unsqueeze(1) - env.pos.unsqueeze(2), dim=-1)
        eye = torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0)
        seen = (dist0 <= R) & ~eye
        pen_max = None
        for k in range(1, int(H / dt) + 1):
            t = k * dt
            rate = (omega * 0.35 if t <= 5.0 else omega).unsqueeze(-1) * dt
            dpsi = wrap(psi_c - psi)
            psi = psi + torch.clamp(dpsi, -1, 1) * torch.minimum(dpsi.abs(), rate)
            pr = psi * DEG
            p = p + torch.stack([torch.sin(pr), torch.cos(pr)], -1) * (v.unsqueeze(-1).unsqueeze(-1) * dt)
            pj = env.pos + vel * t
            d = torch.linalg.norm(pj.unsqueeze(1).unsqueeze(3) - p.unsqueeze(2), dim=-1)
            pen = 100.0 * (torch.clamp(D_safe - d, min=0) / D_safe) ** 2 / (1.0 + t / 30.0)
            pen_max = pen if pen_max is None else torch.maximum(pen_max, pen)
        risk = (pen_max * seen.unsqueeze(-1)).sum(2)
        cost = risk + cands.abs().view(1, 1, -1) / 90.0 * 30.0 + (cands.view(1, 1, -1) != prev.unsqueeze(-1)).float() * 3.0
        choice = cands[cost.argmin(-1)]
        return torch.clamp(wrap(gb + choice - env.heading) / 10.0, -1, 1), choice

    def seeker_a0(env):
        pw = env._last_pw
        d = pw['dist'] + torch.eye(env.N, device=dev).unsqueeze(0) * 1e9
        rel = env.pos.unsqueeze(1) - env.pos.unsqueeze(2)
        br = wrap(torch.atan2(rel[..., 0], rel[..., 1]) / DEG - env.heading.unsqueeze(-1))
        ahead = (br.abs() < 90.0) & (d <= float(cfg.COMM_RANGE))
        dd = torch.where(ahead, d, torch.full_like(d, 1e9))
        dmin, j = dd.min(-1)
        on_course = ((pw['dcpa'] < vg.DCPA_RISK) & (pw['raw_tcpa'] >= 0) & (d < 1e8)).any(-1)
        toward = torch.clamp(br.gather(-1, j.unsqueeze(-1)).squeeze(-1) / 20.0, -1, 1)
        a0 = torch.where((dmin < 1e8) & ~on_course, toward, goal_a0(env))
        return radar_rule(env, a0)

    gamma = float(cfg.DISCOUNT_FACTOR)
    rows = []
    for seed in a.seeds:
        for pol in POLICIES:
            env = vg.VesselBatchEnv(num_envs=a.E, n_vessels=16, device=dev, seed=seed, ring_scale=1.0, crossing=0,
                                    risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE,
                                    farfield_coef=cfg.FARFIELD_COEF, perpair_coef=cfg.PERPAIR_COEF, perpair_exp=3.0)
            env.reset()
            prev = torch.zeros(env.E, env.N, device=dev)
            R, D = [], []
            eps = coll = goal = 0
            for t in range(a.burn + a.T):
                if pol == 'goal':
                    a0 = goal_a0(env)
                elif pol == 'stbd':
                    a0 = torch.where(env.situation > 0, torch.full_like(prev, 0.6), goal_a0(env))
                elif pol == 'radar':
                    a0 = radar_rule(env, goal_a0(env))
                elif pol == 'seeker':
                    a0 = seeker_a0(env)
                else:
                    a0, prev = vo_action(env, 56.0, prev, pol.endswith('s'))
                act = torch.stack([a0, torch.ones_like(a0)], -1)
                _, r, done, oc = env.step(act)
                prev = torch.where(done, torch.zeros_like(prev), prev)
                if t >= a.burn:
                    R.append(r.cpu()); D.append(done.cpu())
                    eps += int(done.sum()); coll += int((oc == vg.OUT_COLLISION_VESSEL).sum())
                    goal += int((oc == vg.OUT_GOAL).sum())
            Rt, Dt = torch.stack(R), torch.stack(D).float()
            G = torch.zeros_like(Rt[0])
            Gs = torch.zeros_like(Rt)
            for t in range(Rt.shape[0] - 1, -1, -1):
                G = Rt[t] + gamma * (1.0 - Dt[t]) * G
                Gs[t] = G
            keep = max(1, Rt.shape[0] - 300)                                  # 창 끝 절단(γ^300≈0.05) 제외
            rows.append({'seed': seed, 'policy': pol, 'G': float(Gs[:keep].mean()), 'r_per_dec': float(Rt.mean()),
                         'eps': eps, 'vColl%': 100.0 * coll / max(eps, 1), 'goal%': 100.0 * goal / max(eps, 1)})
            print(f"  [{a.child}] seed {seed} {pol:7s} G={rows[-1]['G']:8.2f} r/dec={rows[-1]['r_per_dec']:6.3f} "
                  f"vColl={rows[-1]['vColl%']:5.1f}% goal={rows[-1]['goal%']:5.1f}% eps={eps}", flush=True)
    print('JSON ' + json.dumps({'setting': a.child, 'pen': float(cfg.ROLE_PROMISE_PEN), 'rows': rows}), flush=True)


# ─────────────────────────── parent ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--child', default=None)
    ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--E', type=int, default=16)
    ap.add_argument('--burn', type=int, default=600)
    ap.add_argument('--T', type=int, default=1200)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    a.seeds = [int(s) for s in str(a.seeds).split(',') if s]
    if a.child:
        return _child(a)
    res = {}
    for setting, pen in (('old', '0'), ('new', '20')):
        env = dict(os.environ)
        env['VESSEL_ROLE_PROMISE_PEN'] = pen
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        cmd = [sys.executable, os.path.abspath(__file__), '--child', setting, '--seeds', ','.join(map(str, a.seeds)),
               '--E', str(a.E), '--burn', str(a.burn), '--T', str(a.T), '--device', a.device]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        sys.stdout.write(p.stdout)
        if p.returncode != 0:
            sys.stdout.write(p.stderr[-4000:])
            print(f"RANK GATE: 중단 — {setting} 서브프로세스 rc={p.returncode}")
            return 2
        line = [l for l in p.stdout.splitlines() if l.startswith('JSON ')][-1]
        res[setting] = json.loads(line[5:])
    ok_all, verdict = True, []
    for seed in a.seeds:
        G = {r['policy']: r['G'] for r in res['new']['rows'] if r['seed'] == seed}
        lo, hi = min(G[p] for p in AVOIDERS), max(G[p] for p in OTHERS)
        ok = (lo - hi) > 0.05 * abs(hi)
        ok_all &= ok
        verdict.append({'seed': seed, 'min_avoider': lo, 'max_other': hi, 'pass': ok})
        Go = {r['policy']: r['G'] for r in res['old']['rows'] if r['seed'] == seed}
        print(f"  seed {seed} new: min(vo)={lo:8.2f} max(others)={hi:8.2f} → {'ok' if ok else 'NOT ok'}   "
              f"| old (기록만): min(vo)={min(Go[p] for p in AVOIDERS):8.2f} max(others)={max(Go[p] for p in OTHERS):8.2f}")
    res['verdict'] = verdict
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"RANK GATE: {'PASS' if ok_all else '★FAIL'}")
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
