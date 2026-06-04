"""
한 run 폴더(또는 results 상위 폴더)를 통째로 분석 — 학습 후 '무엇이 문제였나' 바로 확인.
각 run의 metric.csv(13열) + events.csv를 읽어 수렴구간(마지막 30%) ground-truth 리포트.
reward 임계값 추정 금지 — 오직 실제 outcome/연료/궤적으로만.

사용:
  python analyze_run.py <build>\results            # 아래 run 폴더 전부 스캔 + 비교
  python analyze_run.py <build>\results\<run>       # 단일 run

metric.csv 13열: 0 id,1 ep,2 outcome,3 steps,4 fuel,5 rudderVar,6 comp,7 occl,8 cmdVar,
                 9 minVesselDist,10 nearMissSteps,11 straightness,12 headingTravel
events.csv     : id,ep,outcome,step,startX,startZ,endX,endZ,heading,speed
"""
import csv
import os
import sys
import math

# Windows 콘솔(cp949)에서 em-dash·한글 출력 시 UnicodeEncodeError 방지 — UTF-8 강제
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

TAIL_FRAC = 0.30
NCHUNK = 5
C_OUTCOME, C_STEPS, C_FUEL, C_RUDVAR, C_COMP, C_OCCL, C_CMDVAR = 2, 3, 4, 5, 6, 7, 8
C_MINVD, C_NEARMISS, C_STRAIGHT, C_HEADTRAVEL = 9, 10, 11, 12
NEAR_CENTER_R = 3.0   # 원통 근처 판정 반경(m, 튜닝). 충돌이 중심에 몰리는지.


def read_csv(path, min_cols):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [r for r in csv.reader(f) if len(r) >= min_cols]


def fnum(r, idx):
    try:
        return float(r[idx])
    except (ValueError, IndexError):
        return None


def outcome_counts(rows):
    c = {"goal": 0, "collision_vessel": 0, "collision_obstacle": 0, "timeout": 0}
    for r in rows:
        c[r[C_OUTCOME]] = c.get(r[C_OUTCOME], 0) + 1
    return c


def pct(c, n, *keys):
    return 100.0 * sum(c.get(k, 0) for k in keys) / n if n else 0.0


def per_step_mean(rows, idx, skip_neg=False):
    v = []
    for r in rows:
        x, s = fnum(r, idx), fnum(r, C_STEPS)
        if x is None or s is None or s <= 0 or (skip_neg and x < 0):
            continue
        v.append(x / s)
    return sum(v) / len(v) if v else float("nan")


def mean_col(rows, idx, skip_neg=False):
    v = [fnum(r, idx) for r in rows]
    v = [x for x in v if x is not None and not (skip_neg and x < 0)]
    return sum(v) / len(v) if v else float("nan")


def late_collapse(rows):
    n = len(rows)
    csz = max(1, n // NCHUNK)
    coll = []
    for i in range(NCHUNK):
        lo = i * csz
        hi = (i + 1) * csz if i < NCHUNK - 1 else n
        ch = rows[lo:hi]
        coll.append(pct(outcome_counts(ch), len(ch), "collision_vessel", "collision_obstacle"))
    mono = all(coll[i] <= coll[i + 1] + 1e-6 for i in range(NCHUNK - 1))
    flag = (coll[-1] > max(coll[0], 1.0) * 1.15) or mono
    return coll, flag


def chunk_trend(rows):
    """시계열 5등분 chunk별 goal%/충돌%(vColl+oColl)/timeout% + 좋아짐/나빠짐 판정.
    충돌이 핵심 안전지표 → 충돌 상승을 우선 경계. 초기 2 chunk vs 최근 2 chunk 비교."""
    n = len(rows)
    csz = max(1, n // NCHUNK)
    chunks = []
    for i in range(NCHUNK):
        lo = i * csz
        hi = (i + 1) * csz if i < NCHUNK - 1 else n
        ch = rows[lo:hi]
        m = len(ch)
        if m == 0:
            chunks.append(None)
            continue
        cc = outcome_counts(ch)
        chunks.append({
            "ep": m,
            "goal": pct(cc, m, "goal"),
            "coll": pct(cc, m, "collision_vessel", "collision_obstacle"),
            "to": pct(cc, m, "timeout"),
        })
    valid = [c for c in chunks if c]
    if len(valid) < 2:
        return chunks, 0.0, 0.0, "데이터부족"
    early = valid[:2]
    late = valid[-2:]
    d_goal = (sum(c["goal"] for c in late) / len(late)) - (sum(c["goal"] for c in early) / len(early))
    d_coll = (sum(c["coll"] for c in late) / len(late)) - (sum(c["coll"] for c in early) / len(early))
    _, collapse = late_collapse(rows)
    # 판정: 충돌 상승(또는 LATE 붕괴)·goal 하락 = 나빠짐 / 충돌 하락·goal 상승 = 좋아짐 / 그 외 횡보
    if collapse or d_coll >= 5.0 or d_goal <= -5.0:
        verdict = "나빠지는중 ▼"
    elif (d_coll <= -3.0 and d_goal >= -1.0) or (d_goal >= 5.0 and d_coll <= 2.0):
        verdict = "좋아지는중 ▲"
    else:
        verdict = "횡보 ―"
    return chunks, d_goal, d_coll, verdict


def analyze_events(path):
    rows = read_csv(path, 10)
    coll = []
    for r in rows:
        if r[C_OUTCOME].startswith("collision"):
            x, z = fnum(r, 6), fnum(r, 7)   # endX,endZ
            if x is not None and z is not None:
                coll.append(math.hypot(x, z))
    if not coll:
        return None
    return {
        "n": len(coll),
        "mean_dist": sum(coll) / len(coll),
        "near_center": sum(1 for d in coll if d < NEAR_CENTER_R) / len(coll),
    }


def analyze_run(run_dir):
    rows = read_csv(os.path.join(run_dir, "metric.csv"), 9)
    if not rows:
        return None
    n = len(rows)
    tail = rows[int(n * (1 - TAIL_FRAC)):]
    m = len(tail)
    c = outcome_counts(tail)
    _, collapse = late_collapse(rows)
    chunks, d_goal, d_coll, verdict = chunk_trend(rows)
    return {
        "name": os.path.basename(run_dir.rstrip("\\/")),
        "ep": n,
        "chunks": chunks,
        "d_goal": d_goal,
        "d_coll": d_coll,
        "verdict": verdict,
        "goal": pct(c, m, "goal"),
        "vColl": pct(c, m, "collision_vessel"),
        "oColl": pct(c, m, "collision_obstacle"),
        "timeout": pct(c, m, "timeout"),
        "fuel_ps": per_step_mean(tail, C_FUEL),
        "cmdVar_ps": per_step_mean(tail, C_CMDVAR),
        "comp": mean_col(tail, C_COMP),
        "straight": mean_col(tail, C_STRAIGHT),
        "headTrv_ps": per_step_mean(tail, C_HEADTRAVEL),
        "minVD": mean_col(tail, C_MINVD, skip_neg=True),
        "nearMiss_ps": per_step_mean(tail, C_NEARMISS),
        "collapse": collapse,
        "ev": analyze_events(os.path.join(run_dir, "events.csv")),
    }


def main():
    if len(sys.argv) < 2:
        print("usage: python analyze_run.py <results_dir | run_dir>")
        return
    root = sys.argv[1]
    if os.path.exists(os.path.join(root, "metric.csv")):
        runs = [root]
    else:
        runs = [os.path.join(root, d) for d in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, d))
                and os.path.exists(os.path.join(root, d, "metric.csv"))]
    results = [x for x in (analyze_run(r) for r in runs) if x]
    if not results:
        print(f"(metric.csv 든 run 폴더 없음: {root})")
        return

    print(f"=== run 분석 (수렴 마지막 {int(TAIL_FRAC*100)}%, ground-truth) — {len(results)} runs ===\n")
    hdr = (f"{'run':<28}{'ep':>6}{'goal':>6}{'vColl':>6}{'oColl':>6}{'TO':>5}"
           f"{'fuel/s':>8}{'cmdV/s':>8}{'strght':>7}{'minVD':>7}{'nearM/s':>8}{'flag':>7}")
    print(hdr)
    print("-" * len(hdr))
    for x in results:
        print(f"{x['name']:<28}{x['ep']:>6}{x['goal']:>6.1f}{x['vColl']:>6.1f}{x['oColl']:>6.1f}"
              f"{x['timeout']:>5.1f}{x['fuel_ps']:>8.3f}{x['cmdVar_ps']:>8.4f}{x['straight']:>7.3f}"
              f"{x['minVD']:>7.2f}{x['nearMiss_ps']:>8.4f}{'LATE' if x['collapse'] else 'ok':>7}")

    print("\n--- run별 시계열 추세 (전체 5등분 초기→후기, run 하나씩) — 좋아지는중/나빠지는중 ---")
    for x in results:
        def fmt(key):
            return "→".join(f"{c[key]:.0f}" if c else "·" for c in x["chunks"])
        print(f"  {x['name']:<28} {x['verdict']}")
        print(f"      goal%  {fmt('goal'):<26} (Δ{x['d_goal']:+.1f})")
        print(f"      coll%  {fmt('coll'):<26} (Δ{x['d_coll']:+.1f})  [vColl+oColl]")
        print(f"      TO%    {fmt('to'):<26}")

    print("\n--- 충돌 공간 분포 (events.csv: 충돌이 어디서 나나) ---")
    for x in results:
        if x["ev"]:
            print(f"  {x['name']:<28} 충돌 {x['ev']['n']}건 | 중심거리 평균 {x['ev']['mean_dist']:.1f}m | "
                  f"중심<{NEAR_CENTER_R:.0f}m {100*x['ev']['near_center']:.0f}% (원통 근처면 obstacle 회피 문제)")
        else:
            print(f"  {x['name']:<28} (events 없음/충돌0)")

    print("\n[판독]")
    print("  추세 판정: 충돌↑ 또는 LATE붕괴 또는 goal↓ = 나빠지는중 ▼ / 충돌↓·goal유지 또는 goal↑↑ = 좋아지는중 ▲ / 그 외 횡보 ―")
    print("  (초기 데이터는 chunk당 표본 적어 노이즈 큼 → 횡보로 나오기 쉬움. step 쌓일수록 추세 굳어짐.)")
    print("  flag=LATE → 그 run 아직 악화 중(수렴 아님, 더 학습 필요). ok면 수렴구간 판정 신뢰.")
    print("  straight↓ + headTrv/s↑  = circling(빙글빙글). minVD작음 + nearMiss/s↑ = 위험한 배회.")
    print("  ON vs OFF: ON이 vColl·oColl·fuel/s·nearMiss/s에서 OFF를 *초과(나쁨)*면 = 통신 유해(H1a 위배=버그).")


if __name__ == "__main__":
    main()
