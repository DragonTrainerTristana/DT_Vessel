"""
수렴 평가 단일 게이트 (★mid-training 착시 차단 — 과거 4번 속은 미관측 변수 방지).

규율:
  1) 판정값은 *마지막 TAIL_FRAC(30%)* 수렴 구간에서만 읽는다.
  2) 시간순 5등분 chunk로 충돌 추세를 본다 → 마지막 chunk 충돌 > 첫 chunk 충돌(×1.15) 또는
     단조증가면 LATE_COLLAPSE 플래그(아직 나빠지는 중 → 어떤 조기 snapshot도 신뢰 금지).
  3) timeout도 함께 본다(소심/freeze 정책을 '안전'으로 오인 방지).

usage:
  python convergence_gate.py                       # 기본 RUN_DIR의 FILES 매핑
  python convergence_gate.py a.csv b.csv ...       # 명시 파일들(tag=파일명)
metric csv = VESSEL_METRIC_LOG (13열). 이 스크립트는 outcome(2)·steps(3)만 사용 → 구버전 9열도 호환.
"""
import csv
import os
import sys

# Windows 콘솔(cp949)에서 em-dash·한글 출력 시 UnicodeEncodeError 방지 — UTF-8 강제
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

RUN_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "run_logs"))
FILES = {
    "OFF_s42": "base_off_mt.csv", "OFF_s43": "off_s43_mt.csv", "OFF_s44": "off_s44_mt.csv",
    "ON_s42": "on_s42_mt.csv", "ON_s43": "on_s43_mt.csv", "ON_s44": "on_s44_mt.csv",
}
C_OUTCOME, C_STEPS = 2, 3
NCHUNK = 5
TAIL_FRAC = 0.30
COLLAPSE_RATIO = 1.15   # 마지막 chunk 충돌이 첫 chunk의 이 배수 초과면 붕괴 의심


def load_rows(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [r for r in csv.reader(f) if len(r) >= 4]


def outcome_pct(rows):
    n = len(rows)
    c = {"goal": 0, "collision_vessel": 0, "collision_obstacle": 0, "timeout": 0}
    steps = []
    for r in rows:
        c[r[C_OUTCOME]] = c.get(r[C_OUTCOME], 0) + 1
        try:
            steps.append(float(r[C_STEPS]))
        except (ValueError, IndexError):
            pass
    coll = c["collision_vessel"] + c["collision_obstacle"]
    avg_s = sum(steps) / len(steps) if steps else 0
    return {
        "n": n,
        "goal": 100.0 * c["goal"] / n if n else 0,
        "vColl": 100.0 * c["collision_vessel"] / n if n else 0,
        "oColl": 100.0 * c["collision_obstacle"] / n if n else 0,
        "coll": 100.0 * coll / n if n else 0,
        "timeout": 100.0 * c["timeout"] / n if n else 0,
        "avgSteps": avg_s,
    }


def analyze(tag, rows):
    n = len(rows)
    if n < NCHUNK:
        print(f"\n=== {tag}: 에피소드 부족({n}) ===")
        return
    print(f"\n=== {tag}  (총 {n} ep) ===")
    print(f"  {'chunk':<7}{'n':>6}{'goal':>7}{'vColl':>7}{'oColl':>7}{'coll':>7}{'timeout':>8}{'avgSteps':>9}")
    csz = n // NCHUNK
    chunk_coll = []
    for i in range(NCHUNK):
        lo = i * csz
        hi = (i + 1) * csz if i < NCHUNK - 1 else n
        s = outcome_pct(rows[lo:hi])
        chunk_coll.append(s["coll"])
        print(f"  {i+1}/{NCHUNK:<5}{s['n']:>6}{s['goal']:>7.1f}{s['vColl']:>7.1f}"
              f"{s['oColl']:>7.1f}{s['coll']:>7.1f}{s['timeout']:>8.1f}{s['avgSteps']:>9.0f}")

    # 수렴 구간(마지막 30%)
    tail = rows[int(n * (1 - TAIL_FRAC)):]
    t = outcome_pct(tail)
    print(f"  {'[수렴]':<7}{t['n']:>6}{t['goal']:>7.1f}{t['vColl']:>7.1f}"
          f"{t['oColl']:>7.1f}{t['coll']:>7.1f}{t['timeout']:>8.1f}{t['avgSteps']:>9.0f}   ← 판정은 여기만")

    # LATE_COLLAPSE 판정
    first, last = chunk_coll[0], chunk_coll[-1]
    # monotonic은 '단조 비감소 + 실제 상승'일 때만 — 완전 평탄(예: 전 chunk 0% = 이상적 run)을
    # 붕괴로 오판하지 않도록 last > first 조건을 함께 요구.
    monotonic = (last > first + 1e-6) and all(
        chunk_coll[i] <= chunk_coll[i + 1] + 1e-6 for i in range(NCHUNK - 1))
    collapse = (last > max(first, 1.0) * COLLAPSE_RATIO) or monotonic
    if collapse:
        print(f"  ⚠️ LATE_COLLAPSE: 충돌 {first:.1f}%→{last:.1f}% (아직 악화 중). "
              f"조기 snapshot 신뢰 금지, 더 학습 후 재평가.")
    else:
        print(f"  ✅ 안정: 충돌 추세 {first:.1f}%→{last:.1f}% (붕괴 아님). 수렴 구간 판정 신뢰 가능.")


def main():
    args = sys.argv[1:]
    if args:
        items = [(os.path.basename(p), p) for p in args]
    else:
        items = [(tag, os.path.join(RUN_DIR, fn)) for tag, fn in FILES.items()]

    print(f"=== 수렴 게이트 (마지막 {int(TAIL_FRAC*100)}%만 판정, 5등분 추세로 LATE_COLLAPSE 탐지) ===")
    for tag, path in items:
        rows = load_rows(path)
        if rows is None:
            print(f"\n=== {tag}: (파일 없음 {path}) ===")
            continue
        analyze(tag, rows)
    print("\n[규율] mid-training이 좋아도 채택 금지. LATE_COLLAPSE면 그 run은 '아직 나빠지는 중'.")


if __name__ == "__main__":
    main()
