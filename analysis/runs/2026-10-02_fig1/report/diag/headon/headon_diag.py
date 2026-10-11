# headon_diag.py — 진단 전용(학습·저장 없음): 정면 조우(head-on)에서 배가 실제로 어떻게 움직였는지 집계.
#   runs_fig1/2026-10-01_scripted/eval_scripted.py 를 그대로 돌리되, colregs_woerner 의 조우 추적기를 상속해
#   정면 조우가 끝날 때마다 (시작 때 상대 방위 · 최근접 때 서로의 방위 · 우현/좌현 최대 변침 · 최근접 거리)를 따로 모음.
#   채점식·평가 코드는 건드리지 않음. 사용: python headon_diag.py <scripted 폴더> -- <eval_scripted.py 인자>
import math, os, sys
sdrop, rest = sys.argv[1], sys.argv[3:]
sys.path.insert(0, sdrop)
import colregs_woerner as cw
import torch

_build = cw.build
KEEP = {}


def build(vg):
    Base = _build(vg)

    class Diag(Base):
        def __init__(self, E, N, device, burnin):
            super().__init__(E, N, device, burnin)
            z = lambda: torch.zeros(E, N, N, device=device)
            self.b0, self.ob0, self.nmax = z(), z(), z()
            self.rec = []
            self.r56 = torch.zeros(E, N, N, dtype=torch.bool, device=device)      # 조우 중 56 m 안에 들어온 적 있음
            self.h56, self.b56, self.dmax56, self.dmin56 = z(), z(), z(), z()
            KEEP['t'] = self

        def observe(self, env, t):
            before = self.act.clone()
            super().observe(env, t)
            started = self.act & ~before
            b, ob = self._bearings(env)
            self.b0 = torch.where(started, b, self.b0)
            self.ob0 = torch.where(started, ob, self.ob0)
            d = env._last_pw['dist']
            n56 = ((d <= 56.0) & ~self.eye).sum(-1).float()[:, :, None].expand(-1, -1, self.N)   # i 주변 56 m 안 배 수
            self.nmax = torch.where(started, n56, torch.where(self.act, torch.maximum(self.nmax, n56), self.nmax))
            # 56 m(무통신 배가 상대를 보기 시작하는 거리) 진입 뒤의 변침만 따로
            self.r56 = self.r56 & ~started
            hi = env.heading[:, :, None].expand(-1, -1, self.N)
            first = self.act & (d <= 56.0) & ~self.r56
            self.r56 = self.r56 | first
            self.h56 = torch.where(first, hi, self.h56)
            self.b56 = torch.where(first, b, self.b56)
            dev = (hi - self.h56 + 180.0) % 360.0 - 180.0
            on = self.act & self.r56
            zz = torch.zeros_like(dev)
            self.dmax56 = torch.where(first, zz, torch.where(on, torch.maximum(self.dmax56, dev), self.dmax56))
            self.dmin56 = torch.where(first, zz, torch.where(on, torch.minimum(self.dmin56, dev), self.dmin56))

        def _finish(self, end, collided):
            m = end & self.count & (self.role == vg.SIT_HEADON)
            if bool(m.any()):
                T = lambda x: x.transpose(1, 2)
                vt = (T(self.act) & (T(self.role) == vg.SIT_HEADON)).float()          # 상대도 나를 정면 조우로 추적 중인가
                self.rec.append(torch.stack([self.b0[m], self.ob0[m], self.bcpa[m], self.ocpa[m], self.dmax[m], self.dmin[m],
                                             self.rcpa[m], collided[m].float(), self.nmax[m], self.r0[m],
                                             self.rman[m], T(self.dmax)[m], T(self.dmin)[m], vt[m],
                                             self.r56[m].float(), self.b56[m], self.dmax56[m], self.dmin56[m]], 1).cpu())
            super()._finish(end, collided)

    return Diag


cw.build = build
import eval_scripted
sys.argv = ['eval_scripted.py'] + rest
eval_scripted.main()

X = torch.cat(KEEP['t'].rec) if KEEP['t'].rec else torch.zeros(0, 18)
b0, ob0, bc, oc, dmax, dmin, rcpa, coll, nmax, r0, rman, dmaxT, dminT, vT, r56, b56, dmax56, dmin56 = [X[:, i] for i in range(18)]
n = X.shape[0]
D = math.pi / 180.0
P = ((torch.sin(bc * D) - 1) / 2) ** 2 * ((torch.sin(oc * D) - 1) / 2) ** 2
pc = lambda m: 100.0 * float(m.float().mean()) if m.numel() else float('nan')
mean = lambda v, m=None: float(v[m].mean()) if (m is None or bool(m.any())) and v.numel() else float('nan')
passPP, passSS = (bc < 0) & (oc < 0), (bc > 0) & (oc > 0)
passMX = ~(passPP | passSS)
tS = (dmax >= 2) & (dmax >= -dmin)
tP = (-dmin >= 2) & (-dmin > dmax)
tN = ~(tS | tP)
iPP, iSS = (b0 < 0) & (ob0 < 0), (b0 > 0) & (ob0 > 0)
iMX = ~(iPP | iSS)
print(f"[headon-diag] n={n} 시작거리 평균={mean(r0):.0f}m 최근접 평균={mean(rcpa):.1f}m 충돌={pc(coll > 0):.1f}% P 평균={100 * mean(P):.1f}")
print(f"[headon-diag] 통과 방식: 좌현대좌현={pc(passPP):.1f}% 우현대우현={pc(passSS):.1f}% 엇갈림(한쪽이 앞을 가로지름)={pc(passMX):.1f}%"
      f" | P 평균: 좌현대좌현 {100 * mean(P, passPP):.1f} · 우현대우현 {100 * mean(P, passSS):.1f} · 엇갈림 {100 * mean(P, passMX):.1f}")
print(f"[headon-diag] 변침 방향: 주로 우현={pc(tS):.1f}% 주로 좌현={pc(tP):.1f}% 2° 미만(안 돎)={pc(tN):.1f}% | 우현 최대 평균={mean(dmax):.1f}° 좌현 최대 평균={mean(-dmin):.1f}°")
print(f"[headon-diag] 시작 때 배치: 서로 좌현 앞(그대로 가면 좌현대좌현)={pc(iPP):.1f}% 서로 우현 앞(그대로 가면 우현대우현)={pc(iSS):.1f}% 그 밖={pc(iMX):.1f}%")
for nm, im in (('서로 좌현 앞', iPP), ('서로 우현 앞', iSS), ('그 밖', iMX)):
    if bool(im.any()):
        print(f"[headon-diag]   시작 '{nm}' (n={int(im.sum())}): 좌현대좌현 {pc(passPP[im]):.1f}% · 우현대우현 {pc(passSS[im]):.1f}% · 엇갈림 {pc(passMX[im]):.1f}%"
              f" | 우현 변침 {pc(tS[im]):.1f}% · 좌현 변침 {pc(tP[im]):.1f}% · 안 돎 {pc(tN[im]):.1f}% | P {100 * mean(P, im):.1f} | 충돌 {pc(coll[im] > 0):.1f}%")
for nm, km in (('주변 56 m 에 배 1척 이하', nmax <= 1), ('주변 56 m 에 배 2척 이상', nmax >= 2)):
    if bool(km.any()):
        print(f"[headon-diag]   {nm} (n={int(km.sum())}, {pc(km):.1f}%): 좌현대좌현 {pc(passPP[km]):.1f}% · P {100 * mean(P, km):.1f} · 좌현 변침 {pc(tP[km]):.1f}%")

# ── 추가: 첫 변침 거리, 변침 크기 분포, 두 배의 변침 조합 ──
man = rman >= 0
print(f"[headon-diag] 첫 2° 변침 때 거리: 변침한 조우 {pc(man):.1f}% · 그 거리 평균 {mean(rman, man):.0f}m · 중앙값 {float(rman[man].median()) if bool(man.any()) else float('nan'):.0f}m"
      f" | 56 m 밖에서 첫 변침 {pc(rman[man] > 56):.1f}% · 150 m 밖 {pc(rman[man] > 150):.1f}%")
print(f"[headon-diag] 우현 최대 변침 분포: <2° {pc(dmax < 2):.1f}% · 2–10° {pc((dmax >= 2) & (dmax < 10)):.1f}% · 10–30° {pc((dmax >= 10) & (dmax < 30)):.1f}% · ≥30° {pc(dmax >= 30):.1f}%"
      f" | 좌현 최대: <2° {pc(-dmin < 2):.1f}% · 2–5° {pc((-dmin >= 2) & (-dmin < 5)):.1f}% · 5–10° {pc((-dmin >= 5) & (-dmin < 10)):.1f}% · ≥10° {pc(-dmin >= 10):.1f}%")
both = vT > 0
if bool(both.any()):
    tST = ((dmaxT >= 2) & (dmaxT >= -dminT))[both]; tPT = ((-dminT >= 2) & (-dminT > dmaxT))[both]
    s_, p_ = tS[both], tP[both]
    pp_ = passPP[both]
    for nm, k in (('둘 다 우현', s_ & tST), ('둘 다 좌현', p_ & tPT), ('한 척 우현·한 척 좌현', (s_ & tPT) | (p_ & tST)),
                  ('한 척만 돎', (s_ | p_) ^ (tST | tPT)), ('둘 다 안 돎', ~(s_ | p_ | tST | tPT))):
        if bool(k.any()):
            print(f"[headon-diag]   두 배 조합 '{nm}' {pc(k):.1f}% (n={int(k.sum())}): 좌현대좌현 {pc(pp_[k]):.1f}% · P {100 * mean(P[both], k):.1f} · 충돌 {pc(coll[both][k] > 0):.1f}%")
# 판정기 v2 기준(우현 10° 이상 + 좌현 5° 이내)을 채운 조우 비율
okj = (dmax >= 10) & (dmin >= -5)
print(f"[headon-diag] 보상 판정기 기준(우현 ≥10° 그리고 좌현 ≤5°) 충족 {pc(okj):.1f}% | 그 안에서 좌현대좌현 {pc(passPP[okj]):.1f}% · P {100 * mean(P, okj):.1f}"
      f" | 미충족에서 좌현대좌현 {pc(passPP[~okj]):.1f}% · P {100 * mean(P, ~okj):.1f}")

# ── 추가: 상대가 어느 쪽에 있을 때 어느 쪽으로 도는가 (조우 전체 / 56 m 진입 뒤) ──
def turn_by_side(tag, side, mx, mn, sel, th):
    ts = (mx >= th) & (mx >= -mn); tp = (-mn >= th) & (-mn > mx)
    for nm, k in (('상대가 좌현 앞', sel & (side < -2)), ('상대가 우현 앞', sel & (side > 2)), ('상대가 정선수(±2°)', sel & (side.abs() <= 2))):
        if bool(k.any()):
            print(f"[headon-diag]   {tag} {nm} (n={int(k.sum())}, {100.0 * float(k.sum()) / max(1.0, float(sel.sum())):.1f}%): "
                  f"우현 변침 {pc(ts[k]):.1f}% · 좌현 변침 {pc(tp[k]):.1f}% · {th:g}° 미만 {pc((~(ts | tp))[k]):.1f}% | 좌현대좌현 {pc(passPP[k]):.1f}% · 충돌 {pc(coll[k] > 0):.1f}%")
allm = torch.ones_like(b0, dtype=torch.bool)
print("[headon-diag] 조우 전체(300 m 시작 침로 기준, 5° 이상 변침):")
turn_by_side('전체', b0, dmax, dmin, allm, 5.0)
in56 = r56 > 0
print(f"[headon-diag] 56 m 안까지 들어온 조우 {pc(in56):.1f}% (n={int(in56.sum())}) — 56 m 진입 침로 기준, 5° 이상 변침:")
turn_by_side('56 m 뒤', b56, dmax56, dmin56, in56, 5.0)
print(f"[headon-diag] 56 m 안까지 들어온 조우: 좌현대좌현 {pc(passPP[in56]):.1f}% · P {100 * mean(P, in56):.1f} · 충돌 {pc(coll[in56] > 0):.1f}% | 안 들어온 조우: 좌현대좌현 {pc(passPP[~in56]):.1f}% · P {100 * mean(P, ~in56):.1f} · 최근접 평균 {mean(rcpa, ~in56):.0f}m")
