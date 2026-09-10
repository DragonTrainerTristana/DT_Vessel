"""
timeout 에피소드가 "안전한 느림"인지 "위험한 배회(near-miss)"인지 진단.
condition(ON/OFF) × outcome(goal/timeout/collision) 별로 메트릭 분해.
- fuel/step: 속도·기동 강도 (높으면 적극 기동, 낮으면 거의 정지)
- compliance: COLREGs 준수 (timeout인데 낮으면 = 위험 기하에서 배회)
- cmdVar/step: 타 사용 빈도
- occlRate: 가림 위협 비율
※ 기존 로그엔 '최근접거리/near-miss'가 없음 → 이건 간접 증거. 직접 증거는 instrumentation 필요(별도).
metric csv 읽기는 metric_io.read_metric (열이름 접근: steps, fuel, commandVar, compliance, occlRate).
2026-09-10 metric_io 로 교체, 위치 인덱스 사용 금지.
"""
import os

import numpy as np

from metric_io import read_metric, Metric

RUN_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "run_logs"))
TAIL_FRAC = 0.30

RUNS = {
    "OFF": ["base_off_mt.csv", "off_s43_mt.csv", "off_s44_mt.csv"],
    "ON":  ["on_s42_mt.csv", "on_s43_mt.csv", "on_s44_mt.csv"],
}


def nrows(m):
    return len(m["outcome"]) if "outcome" in m else 0


def sub(m, lo, hi):
    """행 구간 [lo:hi) — 열이름 그대로 유지한 부분 Metric."""
    return Metric({k: v[lo:hi] for k, v in m.items()})


def concat(ms):
    """여러 Metric 을 행 방향으로 합침(공통 열만 — 세대가 다른 파일 pooled 허용)."""
    ms = [m for m in ms if m is not None and nrows(m)]
    if not ms:
        return Metric()
    keys = set(ms[0]).intersection(*ms[1:])
    return Metric({k: np.concatenate([m[k] for m in ms]) for k in keys})


def load_window(fn):
    path = os.path.join(RUN_DIR, fn)
    if not os.path.exists(path):
        return None
    m = read_metric(path)
    n = m.n
    return sub(m, int(n * (1 - TAIL_FRAC)), n)


def select(m, mask):
    return Metric({k: v[mask] for k, v in m.items()})


def main():
    print(f"=== timeout 안전성 진단 (수렴 마지막 {int(TAIL_FRAC*100)}%, 3-seed pooled) ===\n")
    for cond, files in RUNS.items():
        pooled = concat([load_window(fn) for fn in files])
        total = nrows(pooled)
        print(f"--- {cond}  (총 {total} ep) ---")
        print(f"  {'outcome':<20}{'n':>6}{'%':>7}{'steps':>8}{'fuel/st':>9}{'cmdVar/st':>11}{'comp':>7}{'occl':>7}")
        for oc in ["goal", "timeout", "collision_vessel", "collision_obstacle"]:
            g = select(pooled, pooled["outcome"] == oc) if total else Metric()
            n = nrows(g)
            if not n:
                print(f"  {oc:<20}{0:>6}{0.0:>7.1f}")
                continue

            def m(col, per_step=False):
                if col not in g:
                    return float("nan")
                v = g[col]
                if per_step:
                    s = g["steps"]
                    ok = s > 0
                    vals = (v[ok] / s[ok]).tolist()
                else:
                    vals = v.tolist()
                return sum(vals) / len(vals) if vals else float("nan")

            print(f"  {oc:<20}{n:>6}{100.0*n/total:>7.1f}{m('steps'):>8.0f}"
                  f"{m('fuel', True):>9.3f}{m('commandVar', True):>11.4f}{m('compliance'):>7.3f}{m('occlRate'):>7.3f}")
        print()

    print("[판독] timeout 행에서:")
    print("  - fuel/step이 goal과 비슷↑ = 적극 기동 중(정지 아님). 낮으면 = 거의 멈춤.")
    print("  - compliance가 goal보다 *낮으면* = 위험 기하에서 배회(=near-miss 의심, 통신 충돌감소가 착시 가능).")
    print("  - compliance가 goal과 비슷/높으면 = 안전하게 느린 것(통신 충돌감소가 진짜에 가까움).")
    print("  ※ 확정은 '최근접 타선 거리/near-miss step' instrumentation 필요(rebuild).")


if __name__ == "__main__":
    main()
