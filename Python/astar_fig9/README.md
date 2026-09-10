# astar_fig9

Fig9(Global) 그림 빌더. 2026-09-10 정리:
- `corridor_run.py`·`eval_astar_global.py`·`worldmap_extract.py` 사본 삭제 — 정본은 `Python/` 상위. 사본은 2026-08-31 `msg_ln` fix 가 빠진 구버전이었음.
- `paper_style.py` 는 `plotting/paper_style.py` 를 가리키는 shim.
- `make_fig9_*.py` 는 상위 스크립트의 산출물(json/csv)만 소비함. 실행 순서는 `Python/EVAL_ASTAR_README.md`.
