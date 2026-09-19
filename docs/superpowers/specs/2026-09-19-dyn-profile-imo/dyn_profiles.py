"""
dyn_profiles.py — vessel_gym._substep 식을 그대로 옮긴 순수 파이썬 기동 스파이크 (numpy/torch 금지).
식 출처: Python/vessel_gym.py :29-40 (상수), :182-184 (_move_toward), :415-420 (_apply_action), :422-448 (_substep)
        Agent/VesselDynamics.cs :98-157 (UpdateDynamics) 와 순서 동일.
"""
import math

# ── vessel_gym.py :29-40 상수 그대로 ──
DT = 0.04
SUBSTEPS = 10
MAX_SPEED_BASE = 1.0
ACCEL = 0.1
DECEL = 0.04
RUDDER_RATE_AGILE = 12.0
MAX_TURN_RATE = 30.0
TURN_FACTOR_AGILE = 1.5
DRAG_COEF = 0.1
DRAG_THRUST_MULT = 0.3
DRAG_THRUST_THRESH = 0.1
# vessel_gym.py :124-126 충돌 박스 / :88 SAFE_PASSING
L = 14.18316
SAFE_PASSING = 12.0
DEG = math.pi / 180.0


def move_toward(a, b, max_delta):          # vessel_gym.py :182-184
    return a + max(-max_delta, min(max_delta, b - a))


class Ship:
    def __init__(self, max_speed):
        self.x = 0.0
        self.z = 0.0
        self.heading = 0.0          # deg, 0 = +z, 시계방향 +
        self.speed = 0.0
        self.rudder = 0.0
        self.cmd_rudder = 0.0
        self.target_speed = 0.0
        self.max_speed = max_speed

    def apply_action(self, a0, a1):        # vessel_gym.py :415-420
        a0 = max(-1.0, min(1.0, a0))
        a1 = max(-1.0, min(1.0, a1))
        self.cmd_rudder = a0 * MAX_TURN_RATE
        self.target_speed = max(0.0, min(self.max_speed, (a1 + 1) * 0.5 * self.max_speed))

    def substep(self, rudder_rate, turn_factor):   # vessel_gym.py :422-448
        delta = ACCEL * DT if self.target_speed > self.speed else DECEL * DT
        self.speed = move_toward(self.speed, self.target_speed, delta)
        self.speed = max(0.0, min(self.max_speed, self.speed))
        self.rudder = move_toward(self.rudder, self.cmd_rudder, rudder_rate * DT)
        speed_ratio = self.speed / max(self.max_speed, 1e-6)
        eff_rudder = self.rudder * speed_ratio
        yaw_rate = eff_rudder * turn_factor                     # deg/s
        h = self.heading * DEG
        v_used = self.speed                                     # 회전 前 heading·현재속도(드래그 前)
        self.x += math.sin(h) * v_used * DT
        self.z += math.cos(h) * v_used * DT
        self.heading += yaw_rate * DT
        drag = DRAG_COEF * DT
        if self.target_speed >= DRAG_THRUST_THRESH:
            drag *= DRAG_THRUST_MULT
        self.speed *= (1 - drag)
        return v_used, yaw_rate


def warmup(ship, rr, tf, seconds=200.0):
    """전속 직진 정상상태까지 굴림(타 0)."""
    ship.apply_action(0.0, 1.0)
    for _ in range(int(seconds / DT)):
        ship.substep(rr, tf)
    ship.x = ship.z = 0.0
    ship.heading = 0.0


def run_maneuver(max_speed, rr, tf, a0, t_max, evasive90=False):
    """t=0 에 타 명령 a0(전속 유지). evasive90: heading 90° 도달을 예측해 타 0 으로 복귀(결정 0.4 s 단위).
    반환 rows: (t, x, z, heading, v_used, yaw, rudder)"""
    s = Ship(max_speed)
    warmup(s, rr, tf)
    rows = [(0.0, 0.0, 0.0, 0.0, s.speed, 0.0, 0.0)]
    n = int(round(t_max / DT))
    cmd = a0
    returned = False
    for i in range(n):
        if i % SUBSTEPS == 0:                                   # 결정 경계(0.4 s)에서만 명령 갱신
            if evasive90 and not returned:
                # 타를 0 으로 되돌리는 동안 누적될 heading = TF·r²/(2·RR)
                pending = tf * s.rudder * s.rudder / (2.0 * rr)
                if s.heading + pending >= 90.0:
                    cmd = 0.0
                    returned = True
            s.apply_action(cmd, 1.0)
        v_used, yaw = s.substep(rr, tf)
        rows.append(((i + 1) * DT, s.x, s.z, s.heading, v_used, yaw, s.rudder))
    return rows


def interp_at_heading(rows, h_target, col):
    """heading 이 h_target 을 처음 넘는 지점에서 col 값을 선형보간. 없으면 None."""
    for k in range(1, len(rows)):
        h0, h1 = rows[k - 1][3], rows[k][3]
        if h0 < h_target <= h1:
            f = (h_target - h0) / (h1 - h0) if h1 != h0 else 0.0
            return rows[k - 1][col] + f * (rows[k][col] - rows[k - 1][col])
    return None


def time_at_lateral(rows, x_target):
    for k in range(1, len(rows)):
        x0, x1 = rows[k - 1][1], rows[k][1]
        if x0 < x_target <= x1:
            f = (x_target - x0) / (x1 - x0)
            return rows[k - 1][0] + f * (rows[k][0] - rows[k - 1][0])
    return None


def value_at_time(rows, t, col):
    k = int(round(t / DT))
    return rows[k][col] if k < len(rows) else None


def path_length_until_heading(rows, h_target):
    d = 0.0
    for k in range(1, len(rows)):
        d += rows[k][4] * DT
        if rows[k][3] >= h_target:
            return d
    return None


def steady_turn(rows):
    """heading 720° 이후 한 바퀴(720→1080) 평균 v, ω 로 R = v/ω. 실측 폭(max x - min x)도 반환."""
    seg = [r for r in rows if 720.0 <= r[3] <= 1080.0]
    if len(seg) < 10:
        return None, None, None
    v = sum(r[4] for r in seg) / len(seg)
    w = sum(r[5] for r in seg) / len(seg)
    R = v / (w * DEG)
    width = max(r[1] for r in seg) - min(r[1] for r in seg)
    return R, v, width


def tf_for_td(td_L, max_speed):
    """정상 선회 R = max_speed/(30·TF·π/180) 를 뒤집어 TF = max_speed/(30·(π/180)·R)."""
    R = td_L * L / 2.0
    return max_speed / (MAX_TURN_RATE * DEG * R)


def fmt(v, nd=2):
    return '—' if v is None else f'{v:.{nd}f}'


def main():
    max_speeds = [0.8, 1.0, 1.8]
    profiles = [('agile', None, RUDDER_RATE_AGILE)]
    for td in (3, 4, 5):
        for rr in (3, 5, 12):
            profiles.append((f'imo TD{td}L RR{rr}', td, float(rr)))

    results = []
    for name, td, rr in profiles:
        for v in max_speeds:
            tf = TURN_FACTOR_AGILE if td is None else tf_for_td(td, v)
            t_max = 60.0 if td is None else 1400.0 / v      # 3바퀴 이상 돌도록
            hard = run_maneuver(v, rr, tf, 1.0, t_max)
            ten = run_maneuver(v, rr, tf, 10.0 / MAX_TURN_RATE, min(t_max, 400.0))
            ev = run_maneuver(v, rr, tf, 1.0, 400.0, evasive90=True)
            R, v_ss, width = steady_turn(hard)
            adv = interp_at_heading(hard, 90.0, 2)
            trn = interp_at_heading(hard, 90.0, 1)
            tdm = interp_at_heading(hard, 180.0, 1)
            t30 = interp_at_heading(hard, 30.0, 0)
            t90 = interp_at_heading(hard, 90.0, 0)
            init_d = path_length_until_heading(ten, 10.0)
            lat = {t: value_at_time(hard, t, 1) for t in (7, 14, 28, 56)}
            lat_ev = {t: value_at_time(ev, t, 1) for t in (7, 14, 28, 56)}
            tl12 = time_at_lateral(ev, SAFE_PASSING)
            tl14 = time_at_lateral(ev, L)
            tl12_hard = time_at_lateral(hard, SAFE_PASSING)
            tl14_hard = time_at_lateral(hard, L)
            h_final_ev = ev[-1][3]
            results.append(dict(name=name, td=td, rr=rr, v=v, tf=tf, R=R, v_ss=v_ss, width=width,
                                adv=adv, trn=trn, tdm=tdm, t30=t30, t90=t90, init_d=init_d,
                                lat=lat, lat_ev=lat_ev, tl12=tl12, tl14=tl14,
                                tl12_hard=tl12_hard, tl14_hard=tl14_hard, h_final_ev=h_final_ev))

    print('## (a) 정상 선회 (전타 30°·전속, R 고정 — TF 를 max_speed 별로 역산)')
    print('| profile | v_max | TF | yaw_full [°/s] | v_ss(선회중) | R [m] | D [m] | D [L] | 실측 폭 [m] |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in results:
        print(f"| {r['name']} | {r['v']} | {r['tf']:.4f} | {MAX_TURN_RATE*r['tf']:.2f} | {fmt(r['v_ss'],4)} | "
              f"{fmt(r['R'],2)} | {fmt(2*r['R'],2)} | {fmt(2*r['R']/L,3)} | {fmt(r['width'],2)} |")

    print('\n## (b) advance / transfer / tactical diameter (전타 30°) — IMO: advance ≤ 4.5 L, TD ≤ 5 L')
    print('| profile | v_max | advance [m] | adv [L] | transfer [m] | trn [L] | TD [m] | TD [L] | t90 [s] | IMO |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in results:
        ok = (r['adv'] is not None and r['tdm'] is not None and r['adv'] / L <= 4.5 and r['tdm'] / L <= 5.0)
        print(f"| {r['name']} | {r['v']} | {fmt(r['adv'])} | {fmt(r['adv']/L if r['adv'] else None,2)} | "
              f"{fmt(r['trn'])} | {fmt(r['trn']/L if r['trn'] else None,2)} | {fmt(r['tdm'])} | "
              f"{fmt(r['tdm']/L if r['tdm'] else None,2)} | {fmt(r['t90'])} | {'PASS' if ok else 'FAIL'} |")

    print('\n## (c) IMO 초기선회: 10° 타 → heading 10° 까지 track 거리 ≤ 2.5 L  /  (d) 전타 후 heading 30° 도달시간')
    print('| profile | v_max | init dist [m] | [L] | ≤2.5L | t30 [s] |')
    print('|---|---|---|---|---|---|')
    for r in results:
        d = r['init_d']
        print(f"| {r['name']} | {r['v']} | {fmt(d)} | {fmt(d/L if d else None,3)} | "
              f"{'PASS' if d is not None and d/L <= 2.5 else 'FAIL'} | {fmt(r['t30'])} |")

    print('\n## (e) 전타 유지 시 횡변위 x(t) [m] (원 진행방향 수직) — 괄호: 90° 변침 후 타 복귀(evasive90) 변형')
    print('| profile | v_max | t=7 | t=14 | t=28 | t=56 | evasive90 최종 heading |')
    print('|---|---|---|---|---|---|---|')
    for r in results:
        cells = ' | '.join(f"{fmt(r['lat'][t])} ({fmt(r['lat_ev'][t])})" for t in (7, 14, 28, 56))
        print(f"| {r['name']} | {r['v']} | {cells} | {r['h_final_ev']:.1f}° |")

    print('\n## (f) 경고시간 비 W/T — W = r_d/closing, T_lat12 = 횡 12 m(SAFE_PASSING), T_lat14 = 횡 14.18 m(1 L). 기동 = evasive90')
    print('closing: head-on 2v · crossing √2 v · overtaking 1.0 m/s(1.8 vs 0.8 고정)')
    rds = (56, 84, 112, 168)
    hdr = '| profile | v_max | T_lat12 [s] | T_lat14 [s] | ' + ' | '.join(f'HO{rd}' for rd in rds) + ' | ' + \
          ' | '.join(f'CR{rd}' for rd in rds) + ' | ' + ' | '.join(f'OT{rd}' for rd in rds) + ' |'
    for which, key in (('W/T_lat12', 'tl12'), ('W/T_lat14', 'tl14')):
        print(f'\n### {which}')
        print(hdr)
        print('|' + '---|' * (4 + 3 * len(rds)))
        for r in results:
            T = r[key]
            v = r['v']
            cells = []
            for closing in (2 * v, math.sqrt(2) * v, 1.0):
                for rd in rds:
                    W = rd / closing
                    cells.append('—' if T is None else f'{W/T:.2f}')
            print(f"| {r['name']} | {v} | {fmt(r['tl12'])} | {fmt(r['tl14'])} | " + ' | '.join(cells) + ' |')

    print('\n### 참고: 전타 *유지* 시 T_lat12 / T_lat14 (도달 못 하면 —)')
    print('| profile | v_max | T_lat12 hard | T_lat14 hard |')
    print('|---|---|---|---|')
    for r in results:
        print(f"| {r['name']} | {r['v']} | {fmt(r['tl12_hard'])} | {fmt(r['tl14_hard'])} |")

    print('\n## (g) agile 추정 검증 (v_max=1.0)')
    a = [r for r in results if r['name'] == 'agile' and r['v'] == 1.0][0]
    print(f"- 해석 R = v/ω = 1.0/(45°/s) = {1.0/(45*DEG):.4f} m → D = {2.0/(45*DEG):.4f} m = {2.0/(45*DEG)/L:.4f} L")
    print(f"- 시뮬 정상 R = {a['R']:.4f} m, D = {2*a['R']:.4f} m = {2*a['R']/L:.4f} L, 실측 폭 {a['width']:.4f} m")
    print(f"- 시뮬 tactical diameter(180°) = {a['tdm']:.4f} m = {a['tdm']/L:.4f} L (타 램프 과도 포함)")
    print(f"- 해석 t30: heading = 1.5·∫12t dt = 9t² → t = √(30/9) = {math.sqrt(30/9):.3f} s ; 시뮬 t30 = {a['t30']:.3f} s")

    print('\n## (h) TF 를 v=1.0 에서 한 번만 정하면 (상수 TF) 다른 max_speed 의 정상 선회직경')
    print('| TD 목표 | TF(v=1.0) | D@0.8 [L] | D@1.0 [L] | D@1.8 [L] | IMO TD≤5L @1.8 |')
    print('|---|---|---|---|---|---|')
    for td in (3, 4, 5):
        tf = tf_for_td(td, 1.0)
        ds = [2 * v / (MAX_TURN_RATE * tf * DEG) / L for v in max_speeds]
        print(f"| {td}L | {tf:.4f} | {ds[0]:.2f} | {ds[1]:.2f} | {ds[2]:.2f} | {'PASS' if ds[2] <= 5 else 'FAIL'} |")


if __name__ == '__main__':
    main()
