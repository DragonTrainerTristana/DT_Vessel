"""
타속 슬루 도입 후 from-scratch 3-seed comm ON/OFF 실험 ground-truth 평가.
METRIC_LOG(9열: id,ep,outcome,steps,fuel,rudderVar,compliance,occlRate,commandVar)을
수렴 구간(마지막 TAIL_FRAC)만 잘라 outcome률 + 메트릭 평균을 condition별(ON/OFF) mean±std로 비교.
reward 임계값 추정 금지 — 오직 실제 outcome/연료/타사용으로만 판정.
"""
import csv
import os
import statistics as st

RUN_DIR = r"C:\Users\sengh\OneDrive\Desktop\Github\MyUnity\Vessel\Vessel_MLAgent\run_logs"
TAIL_FRAC = 0.30  # 각 run 마지막 30% 에피소드 = 수렴 구간

RUNS = {
    "OFF": {
        "s42": "base_off_mt.csv",
        "s43": "off_s43_mt.csv",
        "s44": "off_s44_mt.csv",
    },
    "ON": {
        "s42": "on_s42_mt.csv",
        "s43": "on_s43_mt.csv",
        "s44": "on_s44_mt.csv",
    },
}

# metric csv 열 인덱스
C_OUTCOME, C_STEPS, C_FUEL, C_RUDVAR, C_COMP, C_OCCL, C_CMDVAR = 2, 3, 4, 5, 6, 7, 8


def parse_run(path):
    """run 1개의 수렴 구간 통계 반환."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for r in csv.reader(f):
            if len(r) < 9:
                continue  # 슬루 이전 8열 stale 행 방어
            rows.append(r)
    if not rows:
        return None
    n = len(rows)
    tail = rows[int(n * (1 - TAIL_FRAC)):]
    m = len(tail)
    if m == 0:
        return None

    # outcome 분포
    outc = {}
    for r in tail:
        o = r[C_OUTCOME]
        outc[o] = outc.get(o, 0) + 1

    def fmean(idx):
        vals = []
        for r in tail:
            try:
                vals.append(float(r[idx]))
            except (ValueError, IndexError):
                pass
        return sum(vals) / len(vals) if vals else float("nan")

    def per_step_mean(idx):
        # 에피소드별 (값/steps)을 구해 평균 → 에피소드 길이 정규화(합산 metric의 길이 편향 제거)
        vals = []
        for r in tail:
            try:
                stp = float(r[C_STEPS])
                if stp > 0:
                    vals.append(float(r[idx]) / stp)
            except (ValueError, IndexError):
                pass
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "episodes_total": n,
        "episodes_window": m,
        "goal_pct": 100.0 * outc.get("goal", 0) / m,
        "vcoll_pct": 100.0 * outc.get("collision_vessel", 0) / m,
        "ocoll_pct": 100.0 * outc.get("collision_obstacle", 0) / m,
        "timeout_pct": 100.0 * outc.get("timeout", 0) / m,
        "steps": fmean(C_STEPS),
        "fuel": fmean(C_FUEL),
        "rudderVar": fmean(C_RUDVAR),
        "commandVar": fmean(C_CMDVAR),
        "fuel_ps": per_step_mean(C_FUEL),
        "rudderVar_ps": per_step_mean(C_RUDVAR),
        "commandVar_ps": per_step_mean(C_CMDVAR),
        "compliance": fmean(C_COMP),
        "occlRate": fmean(C_OCCL),
        "_outc": outc,
    }


def agg(cond_runs):
    """seed들에 대해 mean±std."""
    keys = ["goal_pct", "vcoll_pct", "ocoll_pct", "timeout_pct",
            "steps", "fuel", "rudderVar", "commandVar",
            "fuel_ps", "rudderVar_ps", "commandVar_ps", "compliance", "occlRate"]
    out = {}
    for k in keys:
        vals = [r[k] for r in cond_runs if r is not None]
        if vals:
            out[k] = (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)
        else:
            out[k] = (float("nan"), 0.0)
    return out


def main():
    print(f"=== 타속 슬루 from-scratch 실험 평가 (수렴 구간 = 마지막 {int(TAIL_FRAC*100)}%) ===\n")
    per_cond = {}
    for cond, seeds in RUNS.items():
        print(f"--- {cond} ---")
        run_stats = []
        for sd, fn in seeds.items():
            path = os.path.join(RUN_DIR, fn)
            if not os.path.exists(path):
                print(f"  {sd}: (없음 {fn})")
                run_stats.append(None)
                continue
            s = parse_run(path)
            run_stats.append(s)
            if s is None:
                print(f"  {sd}: (데이터 없음)")
                continue
            print(f"  {sd}: ep={s['episodes_total']} win={s['episodes_window']} | "
                  f"goal={s['goal_pct']:.1f}% vColl={s['vcoll_pct']:.1f}% "
                  f"oColl={s['ocoll_pct']:.1f}% timeout={s['timeout_pct']:.1f}% | "
                  f"fuel={s['fuel']:.3f} rudVar={s['rudderVar']:.2f} cmdVar={s['commandVar']:.2f} "
                  f"comp={s['compliance']:.3f} steps={s['steps']:.0f}")
        per_cond[cond] = agg(run_stats)
        print()

    # 비교표
    print("=== condition 비교 (3-seed mean±std) ===")
    metrics = [
        ("goal_pct", "goal %", "높을수록 좋음"),
        ("vcoll_pct", "vessel충돌 %", "낮을수록 좋음"),
        ("ocoll_pct", "장애물충돌 %", "낮을수록 좋음"),
        ("timeout_pct", "timeout %", "낮을수록 좋음"),
        ("fuel_ps", "fuel/step ★", "낮을수록 좋음 (H1 핵심·길이정규화)"),
        ("commandVar_ps", "cmdVar/step ★", "낮을수록 부드러움 (H1 핵심·길이정규화)"),
        ("rudderVar_ps", "rudVar/step ★", "낮을수록 부드러움(실제·길이정규화)"),
        ("fuel", "fuel(합산)", "참고: 에피소드 길이에 비례(편향)"),
        ("commandVar", "cmdVar(합산)", "참고: 길이 편향"),
        ("compliance", "COLREGs준수", "높을수록 좋음"),
        ("occlRate", "occlRate", "가림 비율"),
        ("steps", "ep길이", "참고"),
    ]
    hdr = f"{'metric':<16}{'OFF':>18}{'ON':>18}{'Δ(ON-OFF)':>14}   note"
    print(hdr)
    print("-" * len(hdr))
    for key, label, note in metrics:
        om, osd = per_cond["OFF"][key]
        nm, nsd = per_cond["ON"][key]
        delta = nm - om
        print(f"{label:<16}{om:>10.2f}±{osd:<6.2f}{nm:>10.2f}±{nsd:<6.2f}{delta:>+14.2f}   {note}")

    print("\n[해석 가이드] H1a: comm-ON이 OFF보다 *나쁘면* = 버그(정보가치≥0 위배). "
          "ON이 fuel·commandVar·vColl에서 OFF 이하면 통신 도움/무해, 초과면 유해.")


if __name__ == "__main__":
    main()
