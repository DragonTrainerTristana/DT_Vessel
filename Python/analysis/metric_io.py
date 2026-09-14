"""metric_io.py — VESSEL_METRIC_LOG CSV 리더 1벌 (2026-09-10).

왜 있나
  metric.csv 는 9열(초기) → 13열 → 15열 → 17열로 자랐고, 열 인덱스 딕셔너리가 분석 스크립트 11벌에
  각자 박혀 있었다. 포맷이 자랄 때마다 구 스크립트가 *에러 없이* 엉뚱한 열을 읽는다.
  → 열 수로 세대를 판별하고 **이름**으로만 접근한다. 위치 인덱스는 이 파일 밖에서 쓰지 않는다.

열 정의 (VesselAgent.cs:903-904, 뒤로 추가만 됨 — 앞 열 위치 불변)
  9열  : agentId, episodeIndex, outcome, steps, fuel, rudderVar, compliance, occlRate, commandVar
  13열 : + minVesselDist, nearMissSteps, straightness, headingTravel          (near-miss/circling 진단)
  15열 : + minDCPA, dcpaBelowSteps
  17열 : + fuelThrust, fuelTurn                                                (fuel 분해, 합=fuel 불변식)
  ★뒤 8열은 진단 전용 — 보상에 절대 비연결(제약3).

쓰는 법
  from metric_io import read_metric
  m = read_metric('metric.csv')          # dict: 열이름 → np.ndarray (문자열 열은 object)
  m['outcome'], m['fuel'], m.ncols, m.gen  # 세대(9/13/15/17)
  m.has('minDCPA')                       # 이 파일에 그 열이 있나
"""
import csv
import os

COLS_9 = ['agentId', 'episodeIndex', 'outcome', 'steps', 'fuel', 'rudderVar', 'compliance', 'occlRate', 'commandVar']
COLS_13 = COLS_9 + ['minVesselDist', 'nearMissSteps', 'straightness', 'headingTravel']
COLS_15 = COLS_13 + ['minDCPA', 'dcpaBelowSteps']
COLS_17 = COLS_15 + ['fuelThrust', 'fuelTurn']
GENERATIONS = {9: COLS_9, 13: COLS_13, 15: COLS_15, 17: COLS_17}
STR_COLS = {'outcome'}
INT_COLS = {'agentId', 'episodeIndex', 'steps', 'nearMissSteps', 'dcpaBelowSteps'}

OUTCOMES = ('goal', 'collision_vessel', 'collision_obstacle', 'timeout')


class Metric(dict):
    """열이름 → 배열. .ncols/.gen/.path/.n 부가정보."""
    def has(self, col):
        return col in self

    def outcome_rates(self):
        """outcome 별 비율(%) — 항상 이 함수로 (직접 세면 라벨 오타가 조용히 0 이 됨)."""
        import numpy as np
        oc = self['outcome']
        n = len(oc)
        return {k: float((oc == k).sum()) / n * 100.0 if n else float('nan') for k in OUTCOMES}


def read_metric(path, *, strict=True):
    """metric CSV → Metric. 헤더가 있으면 헤더로, 없으면 열 수로 세대 판별.

    strict=True: 열 수가 9/13/15/17 이 아니면 예외. False 면 앞에서부터 아는 이름만 붙이고 나머지는 col_N.
    """
    import numpy as np
    with open(path, encoding='utf-8', newline='') as f:
        rows = [r for r in csv.reader(f) if r and not (len(r) == 1 and not r[0].strip())]
    if not rows:
        m = Metric(); m.ncols = 0; m.gen = 0; m.path = path; m.n = 0
        return m
    # 헤더 판별: 첫 행 첫 칸이 숫자가 아니면 헤더
    header = None
    try:
        float(rows[0][0])
    except ValueError:
        header = [h.strip() for h in rows[0]]
        rows = rows[1:]
    ncols = len(rows[0]) if rows else (len(header) if header else 0)
    if header is None:
        if ncols in GENERATIONS:
            header = GENERATIONS[ncols]
        elif strict:
            raise ValueError(f"{path}: 열 {ncols}개 — 아는 세대(9/13/15/17) 아님. 포맷이 바뀌었으면 metric_io.py 부터 갱신")
        else:
            base = max(k for k in GENERATIONS if k <= ncols) if ncols >= 9 else 0
            header = (GENERATIONS[base] if base else []) + [f'col_{i}' for i in range(base, ncols)]
    bad = [i for i, r in enumerate(rows) if len(r) != ncols]
    if bad and strict:
        raise ValueError(f"{path}: 열 수가 다른 행 {len(bad)}개 (예: {bad[0]+1}행)")
    rows = [r for r in rows if len(r) == ncols]
    m = Metric()
    for j, name in enumerate(header):
        col = [r[j] for r in rows]
        if name in STR_COLS:
            m[name] = np.array(col, dtype=object)
        elif name in INT_COLS:
            m[name] = np.array([int(float(x)) for x in col], dtype=np.int64)
        else:
            m[name] = np.array([float(x) for x in col], dtype=np.float64)
    m.ncols = ncols; m.gen = ncols if ncols in GENERATIONS else 0; m.path = path; m.n = len(rows)
    return m


def read_outcome(path):
    """VESSEL_OUTCOME_LOG (agentId,episodeIndex,outcome,steps) → Metric(4열)."""
    return read_metric(path, strict=False)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('사용: python metric_io.py <metric.csv>'); sys.exit(2)
    m = read_metric(sys.argv[1])
    print(f'{os.path.basename(m.path)}: {m.n}행 {m.ncols}열 (세대 {m.gen})')
    for k, v in m.outcome_rates().items():
        print(f'  {k:20s} {v:6.2f}%')
