"""
각 run의 outcome 분포를 시간순 5등분으로 보여줘 후반 붕괴/추세 탐지.
(통신 후반 붕괴 의심 — 수렴 평가는 *맨 끝* 구간으로 해야 함.)
"""
import csv
import os

RUN_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "run_logs"))
FILES = {
    "OFF_s42": "base_off_mt.csv", "OFF_s43": "off_s43_mt.csv", "OFF_s44": "off_s44_mt.csv",
    "ON_s42": "on_s42_mt.csv", "ON_s43": "on_s43_mt.csv", "ON_s44": "on_s44_mt.csv",
}
C_OUTCOME, C_STEPS = 2, 3
NCHUNK = 5


def main():
    for tag, fn in FILES.items():
        path = os.path.join(RUN_DIR, fn)
        if not os.path.exists(path):
            print(f"{tag}: (없음)"); continue
        rows = [r for r in csv.reader(open(path, encoding="utf-8", errors="ignore")) if len(r) >= 9]
        n = len(rows)
        if n == 0:
            print(f"{tag}: (빈 파일)"); continue
        print(f"\n=== {tag}  (총 {n} ep) — 시간순 {NCHUNK}등분 outcome % ===")
        print(f"  {'chunk':<8}{'n':>6}{'goal':>7}{'vColl':>7}{'oColl':>7}{'timeout':>8}{'avgSteps':>9}")
        csz = n // NCHUNK
        for i in range(NCHUNK):
            lo = i * csz
            hi = (i + 1) * csz if i < NCHUNK - 1 else n
            chunk = rows[lo:hi]
            m = len(chunk)
            if m == 0:
                continue
            cnt = {"goal": 0, "collision_vessel": 0, "collision_obstacle": 0, "timeout": 0}
            steps = []
            for r in chunk:
                cnt[r[C_OUTCOME]] = cnt.get(r[C_OUTCOME], 0) + 1
                try:
                    steps.append(float(r[C_STEPS]))
                except (ValueError, IndexError):
                    pass
            avg_s = sum(steps) / len(steps) if steps else 0
            print(f"  {i+1}/{NCHUNK:<6}{m:>6}{100*cnt['goal']/m:>7.1f}"
                  f"{100*cnt['collision_vessel']/m:>7.1f}{100*cnt['collision_obstacle']/m:>7.1f}"
                  f"{100*cnt['timeout']/m:>8.1f}{avg_s:>9.0f}")


if __name__ == "__main__":
    main()
