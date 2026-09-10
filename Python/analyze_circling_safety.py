"""
circling / near-miss 진단 (★13열 이상 메트릭의 진단 컬럼 사용 — 보상에 비연결된 로그 전용 지표).

사용 컬럼(이름):
  minVesselDist   에피소드 최근접 타선거리 (작을수록 위험노출↑; -1=타선 미접근)
  nearMissSteps   minDist<NEAR_MISS_DIST(6m) 결정 수 (충돌 직전 배회 직접 증거)
  straightness    순변위/경로길이 ∈[0,1] (낮을수록 빙빙/우회; ~1=직선 통과)
  headingTravel   누적 |Δheading|(도) (per-step↑ = 지속 선회=circling)

판독:
  - straightness 낮고 headingTravel/step 높으면 = circling (회피 강화의 부작용).
  - timeout 행에서 nearMiss 높고 minVesselDist 작으면 = '위험한 배회'(안전한 느림 아님).
  - condition(ON/OFF) 비교: ON이 충돌만 줄이고 straightness↓·nearMiss↑면 = 가짜 안전(착시).

metric csv 읽기는 metric_io.read_metric (세대 자동 판별). 진단 컬럼 없는 9열 파일은 데이터 없음으로 처리.
2026-09-10 metric_io 로 교체, 위치 인덱스 사용 금지.
"""
import os
import sys

import numpy as np

from metric_io import read_metric, Metric

# Windows 콘솔(cp949)에서 em-dash·한글 출력 시 UnicodeEncodeError 방지 — UTF-8 강제
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

RUN_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "run_logs"))
RUNS = {
    "OFF": ["base_off_mt.csv", "off_s43_mt.csv", "off_s44_mt.csv"],
    "ON":  ["on_s42_mt.csv", "on_s43_mt.csv", "on_s44_mt.csv"],
}
TAIL_FRAC = 0.30
DIAG_COLS = ("minVesselDist", "nearMissSteps", "straightness", "headingTravel")


def nrows(m):
    return len(m["outcome"]) if "outcome" in m else 0


def sub(m, lo, hi):
    """행 구간 [lo:hi) — 열이름 그대로 유지한 부분 Metric."""
    return Metric({k: v[lo:hi] for k, v in m.items()})


def select(m, mask):
    return Metric({k: v[mask] for k, v in m.items()})


def concat(ms):
    """여러 Metric 을 행 방향으로 합침(공통 열만 — 세대가 다른 파일 pooled 허용)."""
    ms = [m for m in ms if m is not None and nrows(m)]
    if not ms:
        return Metric()
    keys = set(ms[0]).intersection(*ms[1:])
    return Metric({k: np.concatenate([m[k] for m in ms]) for k in keys})


def load_window(path):
    """수렴 구간(마지막 TAIL_FRAC). 진단 컬럼이 없는 세대(9열)는 None — 구버전엔 진단컬럼 없음."""
    if not os.path.exists(path):
        return None
    m = read_metric(path)
    if not all(m.has(c) for c in DIAG_COLS):
        return None
    n = m.n
    return sub(m, int(n * (1 - TAIL_FRAC)), n)


def summarize(g):
    """outcome 그룹의 진단지표 평균."""
    def mean(col, per_step=False, skip_neg=False):
        if col not in g:
            return float("nan")
        v = g[col]
        ok = np.ones(len(v), dtype=bool)
        if skip_neg:
            ok &= ~(v < 0)
        if per_step:
            s = g["steps"]
            ok &= s > 0
            vals = (v[ok] / s[ok]).tolist()
        else:
            vals = v[ok].tolist()
        return sum(vals) / len(vals) if vals else float("nan")
    return {
        "minVD": mean("minVesselDist", skip_neg=True),
        "nearMiss_ps": mean("nearMissSteps", per_step=True),
        "straight": mean("straightness"),
        "headTravel_ps": mean("headingTravel", per_step=True),
    }


def main():
    args = sys.argv[1:]
    runs = {"FILES": args} if args else RUNS

    print(f"=== circling/near-miss 진단 (수렴 마지막 {int(TAIL_FRAC*100)}%, condition별 pooled) ===")
    print("  straightness↓+headTravel/st↑ = circling | minVD작음+nearMiss/st↑ = 위험한 배회\n")
    for cond, files in runs.items():
        pooled = concat([load_window(fn if os.path.isabs(fn) else os.path.join(RUN_DIR, fn)) for fn in files])
        total = nrows(pooled)
        if not total:
            print(f"--- {cond}: (13열 데이터 없음 — 새 빌드로 재학습 후 생성) ---\n")
            continue
        print(f"--- {cond}  (총 {total} ep) ---")
        print(f"  {'outcome':<20}{'n':>6}{'%':>7}{'straight':>10}{'headTrv/st':>12}{'minVD':>8}{'nearMiss/st':>13}")
        for oc in ["goal", "timeout", "collision_vessel", "collision_obstacle"]:
            g = select(pooled, pooled["outcome"] == oc)
            ng = nrows(g)
            if not ng:
                continue
            s = summarize(g)
            print(f"  {oc:<20}{ng:>6}{100.0*ng/total:>7.1f}{s['straight']:>10.3f}"
                  f"{s['headTravel_ps']:>12.4f}{s['minVD']:>8.2f}{s['nearMiss_ps']:>13.4f}")
        print()
    print("[해석] timeout 행 straightness가 낮고 headTravel/st 높으면 = '우물쭈물 배회=circling'(=사고 직전).")
    print("       goal 행 straightness 높으면 = 깔끔한 통과. ON vs OFF에서 ON이 더 낮으면 통신이 궤적 악화.")


if __name__ == "__main__":
    main()
