"""
circling / near-miss 진단 (★새 13열 메트릭 사용 — 보상에 비연결된 로그 전용 지표).

새 컬럼:
  [9]  minVesselDist   에피소드 최근접 타선거리 (작을수록 위험노출↑; -1=타선 미접근)
  [10] nearMissSteps   minDist<NEAR_MISS_DIST(6m) 결정 수 (충돌 직전 배회 직접 증거)
  [11] straightness    순변위/경로길이 ∈[0,1] (낮을수록 빙빙/우회; ~1=직선 통과)
  [12] headingTravel   누적 |Δheading|(도) (per-step↑ = 지속 선회=circling)

판독:
  - straightness 낮고 headingTravel/step 높으면 = circling (회피 강화의 부작용).
  - timeout 행에서 nearMiss 높고 minVesselDist 작으면 = '위험한 배회'(안전한 느림 아님).
  - condition(ON/OFF) 비교: ON이 충돌만 줄이고 straightness↓·nearMiss↑면 = 가짜 안전(착시).
"""
import csv
import os
import sys
import statistics as st

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
C_OUTCOME, C_STEPS = 2, 3
C_MINVD, C_NEARMISS, C_STRAIGHT, C_HEADTRAVEL = 9, 10, 11, 12


def load_window(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="ignore") as f:
        rows = [r for r in csv.reader(f) if len(r) >= 13]   # 13열(신규)만 — 구버전 9열은 진단컬럼 없음
    return rows[int(len(rows) * (1 - TAIL_FRAC)):]


def fval(r, idx):
    try:
        return float(r[idx])
    except (ValueError, IndexError):
        return None


def summarize(group):
    """outcome 그룹의 진단지표 평균."""
    def mean(idx, per_step=False, skip_neg=False):
        vals = []
        for r in group:
            v = fval(r, idx)
            if v is None or (skip_neg and v < 0):
                continue
            if per_step:
                s = fval(r, C_STEPS)
                if s and s > 0:
                    vals.append(v / s)
            else:
                vals.append(v)
        return sum(vals) / len(vals) if vals else float("nan")
    return {
        "minVD": mean(C_MINVD, skip_neg=True),
        "nearMiss_ps": mean(C_NEARMISS, per_step=True),
        "straight": mean(C_STRAIGHT),
        "headTravel_ps": mean(C_HEADTRAVEL, per_step=True),
    }


def main():
    args = sys.argv[1:]
    runs = {"FILES": args} if args else RUNS

    print(f"=== circling/near-miss 진단 (수렴 마지막 {int(TAIL_FRAC*100)}%, condition별 pooled) ===")
    print("  straightness↓+headTravel/st↑ = circling | minVD작음+nearMiss/st↑ = 위험한 배회\n")
    for cond, files in runs.items():
        rows = []
        for fn in files:
            path = fn if os.path.isabs(fn) else os.path.join(RUN_DIR, fn)
            rows.extend(load_window(path))
        if not rows:
            print(f"--- {cond}: (13열 데이터 없음 — 새 빌드로 재학습 후 생성) ---\n")
            continue
        groups = {}
        for r in rows:
            groups.setdefault(r[C_OUTCOME], []).append(r)
        total = len(rows)
        print(f"--- {cond}  (총 {total} ep) ---")
        print(f"  {'outcome':<20}{'n':>6}{'%':>7}{'straight':>10}{'headTrv/st':>12}{'minVD':>8}{'nearMiss/st':>13}")
        for oc in ["goal", "timeout", "collision_vessel", "collision_obstacle"]:
            g = groups.get(oc, [])
            if not g:
                continue
            s = summarize(g)
            print(f"  {oc:<20}{len(g):>6}{100.0*len(g)/total:>7.1f}{s['straight']:>10.3f}"
                  f"{s['headTravel_ps']:>12.4f}{s['minVD']:>8.2f}{s['nearMiss_ps']:>13.4f}")
        print()
    print("[해석] timeout 행 straightness가 낮고 headTravel/st 높으면 = '우물쭈물 배회=circling'(=사고 직전).")
    print("       goal 행 straightness 높으면 = 깔끔한 통과. ON vs OFF에서 ON이 더 낮으면 통신이 궤적 악화.")


if __name__ == "__main__":
    main()
