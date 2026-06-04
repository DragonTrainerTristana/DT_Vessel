"""
timeout 에피소드가 "안전한 느림"인지 "위험한 배회(near-miss)"인지 진단.
condition(ON/OFF) × outcome(goal/timeout/collision) 별로 메트릭 분해.
- fuel/step: 속도·기동 강도 (높으면 적극 기동, 낮으면 거의 정지)
- compliance: COLREGs 준수 (timeout인데 낮으면 = 위험 기하에서 배회)
- cmdVar/step: 타 사용 빈도
- occlRate: 가림 위협 비율
※ 기존 로그엔 '최근접거리/near-miss'가 없음 → 이건 간접 증거. 직접 증거는 instrumentation 필요(별도).
"""
import csv
import os

RUN_DIR = r"C:\Users\sengh\OneDrive\Desktop\Github\MyUnity\Vessel\Vessel_MLAgent\run_logs"
TAIL_FRAC = 0.30

RUNS = {
    "OFF": ["base_off_mt.csv", "off_s43_mt.csv", "off_s44_mt.csv"],
    "ON":  ["on_s42_mt.csv", "on_s43_mt.csv", "on_s44_mt.csv"],
}
C_OUTCOME, C_STEPS, C_FUEL, C_RUDVAR, C_COMP, C_OCCL, C_CMDVAR = 2, 3, 4, 5, 6, 7, 8


def load_window(fn):
    path = os.path.join(RUN_DIR, fn)
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for r in csv.reader(f):
            if len(r) >= 9:
                rows.append(r)
    return rows[int(len(rows) * (1 - TAIL_FRAC)):]


def main():
    print(f"=== timeout 안전성 진단 (수렴 마지막 {int(TAIL_FRAC*100)}%, 3-seed pooled) ===\n")
    for cond, files in RUNS.items():
        rows = []
        for fn in files:
            rows.extend(load_window(fn))
        # outcome 그룹화
        groups = {}
        for r in rows:
            groups.setdefault(r[C_OUTCOME], []).append(r)
        total = len(rows)
        print(f"--- {cond}  (총 {total} ep) ---")
        print(f"  {'outcome':<20}{'n':>6}{'%':>7}{'steps':>8}{'fuel/st':>9}{'cmdVar/st':>11}{'comp':>7}{'occl':>7}")
        for oc in ["goal", "timeout", "collision_vessel", "collision_obstacle"]:
            g = groups.get(oc, [])
            if not g:
                print(f"  {oc:<20}{0:>6}{0.0:>7.1f}")
                continue
            n = len(g)

            def m(idx, per_step=False):
                vals = []
                for r in g:
                    try:
                        v = float(r[idx])
                        if per_step:
                            s = float(r[C_STEPS])
                            if s > 0:
                                vals.append(v / s)
                        else:
                            vals.append(v)
                    except (ValueError, IndexError):
                        pass
                return sum(vals) / len(vals) if vals else float("nan")

            print(f"  {oc:<20}{n:>6}{100.0*n/total:>7.1f}{m(C_STEPS):>8.0f}"
                  f"{m(C_FUEL, True):>9.3f}{m(C_CMDVAR, True):>11.4f}{m(C_COMP):>7.3f}{m(C_OCCL):>7.3f}")
        print()

    print("[판독] timeout 행에서:")
    print("  - fuel/step이 goal과 비슷↑ = 적극 기동 중(정지 아님). 낮으면 = 거의 멈춤.")
    print("  - compliance가 goal보다 *낮으면* = 위험 기하에서 배회(=near-miss 의심, 통신 충돌감소가 착시 가능).")
    print("  - compliance가 goal과 비슷/높으면 = 안전하게 느린 것(통신 충돌감소가 진짜에 가까움).")
    print("  ※ 확정은 '최근접 타선 거리/near-miss step' instrumentation 필요(rebuild).")


if __name__ == "__main__":
    main()
