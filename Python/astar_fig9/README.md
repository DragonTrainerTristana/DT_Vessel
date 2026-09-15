# astar_fig9

Fig9(Global) 그림 빌더. 2026-09-10 정리:
- `corridor_run.py`·`worldmap_extract.py` 사본 삭제 — 정본은 `Python/` 상위. 사본은 2026-08-31 `msg_ln` fix 가 빠진 구버전이었음.
- `paper_style.py` 는 `plotting/paper_style.py` 를 가리키는 shim.
- `make_fig9_*.py` 는 상위 스크립트의 산출물(json/csv)만 소비함. 실행 순서는 `Python/EVAL_ASTAR_README.md`.

## eval_astar_global.py 편입 (2026-09-15, PLAN.md 6단계 / git refactor(5))

- `Python/eval_astar_global.py` → 이 디렉터리로 이동. MAP.md 근거: "`Python/` 루트에 있으나
  astar_fig9 세트의 일부, 소비자는 `make_fig9_from_eval.py`". `corridor_run.py`는 이번 단계에서
  안 움직임(eval 그룹, PLAN.md 7단계 소속) — 위 항목의 "정본은 Python/ 상위"는 corridor_run.py에만 해당.
- `config`·`vessel_gym`·`networks`·`vessel_gym_train`·`ckpt_io` 는 여전히 `Python/` 루트에 있어
  `sys.path.insert(0, .../..)` 를 파일 상단에 추가함(unity-island·fidelity와 동일 패턴).
  기본 체크포인트 경로(`VESSEL_CKPT_DIR` 미설정 시 `<파일위치>/checkpoints`)도 깊이 +1 보정해
  이동 전과 같은 `Python/checkpoints` 를 가리키게 함.
- `paper_style.py` shim(`../plotting/paper_style.py` 상대참조, MAP.md [MOVE-RISK])은 이번 편입과 무관 —
  `plotting/`·`astar_fig9/` 두 디렉터리 자체는 옮기지 않는 것이 PLAN.md의 확정 결정이라 상대 위치가
  그대로 유지됨. 편입 후 `make_fig9_from_eval.py` 를 실행해 shim 이 여전히 뜨는 것으로 확인함.
- 이동 전 경로(`Python/eval_astar_global.py`) 참조는 저장소 전체 grep 결과 `docs/MAP.md`·`docs/PLAN.md`
  (계획/맵 문서, 정적 스냅샷 — 이전 단계 선례상 이동마다 갱신 안 함) 외 0건. `EVAL_ASTAR_README.md` 의
  실행 예시 명령만 새 경로로 고침.
