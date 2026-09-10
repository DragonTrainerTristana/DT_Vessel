"""
각 run의 outcome 분포를 시간순 5등분으로 보여줘 후반 붕괴/추세 탐지.
(통신 후반 붕괴 의심 — 수렴 평가는 *맨 끝* 구간으로 해야 함.)
metric csv 읽기는 metric_io.read_metric (outcome·steps 열만 사용, 세대 자동 판별).
2026-09-10 metric_io 로 교체, 위치 인덱스 사용 금지.
"""
import os

from metric_io import read_metric, Metric, OUTCOMES

RUN_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "run_logs"))
FILES = {
    "OFF_s42": "base_off_mt.csv", "OFF_s43": "off_s43_mt.csv", "OFF_s44": "off_s44_mt.csv",
    "ON_s42": "on_s42_mt.csv", "ON_s43": "on_s43_mt.csv", "ON_s44": "on_s44_mt.csv",
}
NCHUNK = 5


def sub(m, lo, hi):
    """행 구간 [lo:hi) — 열이름 그대로 유지한 부분 Metric."""
    return Metric({k: v[lo:hi] for k, v in m.items()})


def main():
    for tag, fn in FILES.items():
        path = os.path.join(RUN_DIR, fn)
        if not os.path.exists(path):
            print(f"{tag}: (없음)"); continue
        mt = read_metric(path)
        n = mt.n
        if n == 0:
            print(f"{tag}: (빈 파일)"); continue
        print(f"\n=== {tag}  (총 {n} ep) — 시간순 {NCHUNK}등분 outcome % ===")
        print(f"  {'chunk':<8}{'n':>6}{'goal':>7}{'vColl':>7}{'oColl':>7}{'timeout':>8}{'avgSteps':>9}")
        csz = n // NCHUNK
        for i in range(NCHUNK):
            lo = i * csz
            hi = (i + 1) * csz if i < NCHUNK - 1 else n
            chunk = sub(mt, lo, hi)
            m = len(chunk["outcome"])
            if m == 0:
                continue
            cnt = {k: 0 for k in OUTCOMES}
            for o in chunk["outcome"].tolist():
                cnt[o] = cnt.get(o, 0) + 1
            steps = chunk["steps"].tolist()
            avg_s = sum(steps) / len(steps) if steps else 0
            print(f"  {i+1}/{NCHUNK:<6}{m:>6}{100*cnt['goal']/m:>7.1f}"
                  f"{100*cnt['collision_vessel']/m:>7.1f}{100*cnt['collision_obstacle']/m:>7.1f}"
                  f"{100*cnt['timeout']/m:>8.1f}{avg_s:>9.0f}")


if __name__ == "__main__":
    main()
