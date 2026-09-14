# metric_io.py · analyze_run.py · analyze_trajectory.py · analyze_circling_safety.py · analyze_timeout_safety.py · convergence_gate.py — 이동됨 (2026-09-14)

원위치 `Python/metric_io.py` · `Python/analyze_run.py` · `Python/analyze_trajectory.py` ·
`Python/analyze_circling_safety.py` · `Python/analyze_timeout_safety.py` · `Python/convergence_gate.py`.
근거는 `docs/PLAN.md` "analysis" 그룹(4단계).

## 역할

metric CSV(`metric_io.read_metric`)만 읽는 순수 분석 도구군. 모델·체크포인트 안 엶.
`run_repro.sh`/`smoke_mac.sh` 가 안 부르는 완전한 잎 — 학습·검증 파이프라인과 무관.

## 이동 이유

- fan-out 1(`metric_io`), 섬 밖 소비자 0. `analyze_run.py` `analyze_trajectory.py`
  `analyze_circling_safety.py` `analyze_timeout_safety.py` `convergence_gate.py` 5개가
  전부 `metric_io.read_metric`/`Metric`/`OUTCOMES` 만 가져다 씀.

## 이동 직전 grep 결과 (2026-09-14)

- `run_repro.sh` · `smoke_mac.sh` — 이 6개 파일을 `$HERE` 상대경로로 호출하는 곳 0건
  (smoke_mac.sh는 comm 미러·golden·fidelity 3개만 돌리고 analysis 그룹은 아예 안 건드림).
- `metric_io.py`/`analyze_*.py`/`convergence_gate.py` 를 이 6개 밖에서 `import` 하는 곳 0건.
- 코드·셸 전수 재검색으로 옛 경로(`Python/metric_io.py` 등) 참조 0건 확인(이동 후).

## sys.path 보정 — 불필요

- 6개 파일 전부 `config`/`networks` 등 `Python/` 루트 모듈을 import 하지 않음(표준 라이브러리 +
  `numpy` + 서로 간 `from metric_io import ...` 뿐). `metric_io.py` 도 같이 `analysis/` 로 내려와
  같은 디렉터리에 있으므로, 직접 실행 시 파이썬이 스크립트 자신의 디렉터리를 `sys.path[0]` 에
  자동으로 넣어 `from metric_io import ...` 가 보정 없이 그대로 풀림. 1단계(unity-island)처럼
  `sys.path.insert` 를 추가할 필요 없음.

## 상대경로 보정 — RUN_DIR 4곳 (필수, 했음)

`analyze_circling_safety.py:33` · `analyze_timeout_safety.py:19` · `analyze_trajectory.py:12` ·
`convergence_gate.py:30` 의 `RUN_DIR = <repo>/../../../run_logs` 가 `Python/` 루트 기준 3단계 `..`
였음. `analysis/` 로 한 단계 더 내려왔으므로 4단계 `..` 로 보정(위 4개 파일 전부 수정, 주석 남김).
이동 전/후 문자열이 동일한 절대경로로 해석되는지 python으로 직접 비교해 확인함:
`.../0702_NewVessel/run_logs` — 동일.

6개 파일 각각 직접 실행(`python3 Python/analysis/<file>.py`)으로 `ImportError`/`ModuleNotFoundError`
없음을 확인함(인자 없이 돌면 "파일 없음"/"0건" 계열 정상 출력 — 실측 데이터가 이 맥에 없어서일 뿐,
import·경로 해석 실패 아님).

## astar_fig9/paper_style.py:10 재확인 (PLAN.md 4단계 risk 항목)

MAP.md:202 문구("`astar_fig9/paper_style.py:10` … analysis 그룹 이동 시 반드시 함께 수정할 것")를
재확인함. 이 shim은 `../plotting/paper_style.py` 를 참조하는데, `astar_fig9/` 와 `plotting/` 은
둘 다 `Python/` 바로 아래 그대로 있고(이번 안에서 이동 안 함), `analysis/` 는 이 둘의 상대 위치와
무관한 별도 신설 디렉터리임. 따라서 analysis 그룹 이동은 이 shim 경로에 **영향 없음** — 실제로
수정할 필요 없이 그대로 성립함을 확인함(`Python/astar_fig9/`·`Python/plotting/` 디렉터리 목록 직접
대조). MAP.md 원문의 "analysis 그룹" 표현은 이 항목에 한해 부정확해 보이나, MAP.md는 이번 세션
수정 대상이 아니라 정정하지 않음.

## 되살릴 때

`git mv Python/analysis/{metric_io.py,analyze_run.py,analyze_trajectory.py,analyze_circling_safety.py,analyze_timeout_safety.py,convergence_gate.py} Python/`
후 위 4개 파일의 RUN_DIR `..` 를 3단계로 되돌리고 이 README 삭제.
