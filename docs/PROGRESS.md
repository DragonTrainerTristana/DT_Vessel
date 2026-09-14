# PROGRESS

근거 = `docs/PLAN.md` 작업 순서. 단계 추가·변경 금지(0번만 신규).

## 체크리스트

- [ ] **0. 드리프트 검사를 common_env 31개 전부로 확장**
      (현재 preflight 는 15개만 검사. 나머지 16개 — `RADAR_BOTTLENECK_CH` `MAX_PARTNERS` `RADAR_RANGE` `COLREGS_MODE` `SIM_COLREGS_COEF` `INTENT_K` `THREAT_COEF` `GOAL_COMM_COEF` `INTENT_COEF` `ROLE_COMM_COEF` `COMM_CONSUMER_COEF` `RECON_EMA_FLOOR` `AGG_MODE` `MSG_GAIN` `TIMEOUT_BOOTSTRAP` `MSG_GATE_APPLY` — 은 어긋나도 PASS)
      → 확장 후 **현재 어긋난 값이 있는지 먼저 확인**. 어긋나면 고치기 전에 보고.
      완료 조건: `bash run_repro.sh smoke` PASS + preflight 가 31개를 대조하고 `YUGIOH 드리프트: PASS` 출력.

- [ ] **1. shared + core-train 동결** (`run_repro.sh` `config.py` `networks.py` `vessel_gym.py` `vessel_gym_train.py` `ckpt_io.py` — 이동 없음)
      완료 조건: smoke PASS + `YUGIOH 드리프트: PASS` + config.py 를 건드렸으면 `test_golden.py --check` 가 `ALL PASS`.

- [ ] **2. unity-island → `_archive/unity_path_2026-09/`** (`main.py` `frame_stack.py` `memory.py` `functions.py` `obs_utils.py`)
      완료 조건: smoke PASS(코어 무영향) + 끊긴 엣지 0 + `_archive` 에 사유·복구법 README.

- [ ] **3. deprecated → `_archive/deprecated_2026-09/`** (`_smoke_fullmoe.py` `inspect_channel_freeze.py`)
      완료 조건: smoke PASS + preflight 출력이 2단계와 동일 + README 에 `_smoke_fullmoe` 상시 FAIL 사실 기재.

- [ ] **4. analysis → `Python/analysis/`** (`metric_io.py` + analyze 4종 + `convergence_gate.py`)
      완료 조건: smoke PASS(무변화) + 이동한 6개 **각각 직접 실행**해 `metric_io` import 와 RUN_DIR 이 이전과 같은 절대경로로 해석되는지 확인.

- [ ] **5. fidelity → `Python/fidelity/`** (`radar_fidelity_compare.py` `unity_fidelity_compare.py`)
      완료 조건: smoke PASS + 두 파일이 계산하는 Build 경로 문자열이 이동 전과 같은 곳을 가리킴(Build 실물 없으면 문자열만).

- [ ] **6. plotting 위치 확정 + astar_fig9 편입** (`eval_astar_global.py` → `astar_fig9/`, `write_fig_code.py:13` 수정)
      완료 조건: `plotting/regenerate_all.py` 1회 완주 + `make_fig9_from_eval.py` 가 shim 경유로 `paper_style` 을 올림 + `eval_astar_global.py` CLI 1회 실행.

- [ ] **7. eval → `Python/eval/`** (`eval_ckpt.py` `eval_mixed.py` `diag_ckpt.py` `measure_regimes.py` `corridor_run.py` `compose_voyage.py` + `run_repro.sh:183,269` 동시 수정)
      완료 조건: smoke PASS + `bash run_repro.sh eval` 1회 + `VESSEL_DIAG_CKPTS=… bash run_repro.sh diag` 1회(게이트 3개 통과해 JSON 산출).

- [ ] **8. verify → `Python/verify/`** (`_verify_ppo_mirror.py` `_verify_comm_mirror.py` `test_golden.py` `test_vessel_gym_fidelity.py` `test_ckpt_compat.py` `golden/` + `run_repro.sh:122,123,131,135` 동시 수정)
      완료 조건: smoke 가 **새 경로의 preflight 4관문 전부** 통과 + `_golden.txt` `_verify_ppo.txt` `_verify_comm.txt` 에 `ALL PASS` grep 성공 + 골든 JSON 이 이동 전과 같은 파일을 읽음. `test_ckpt_compat.py` 는 미검증으로 명시 기록.

## 판정 규칙

- Mac 에서는 comm 미러·골든·fidelity 까지만 근거로 삼는다.
- 최종 판정 smoke 는 Windows 에서 돌린다 (`_verify_ppo_mirror.py` 가 Windows 전용).
- `VESSEL_SKIP_GOLDEN=1` 로 돌린 실행은 완료 판정에 쓰지 않는다.
- 단계마다 반드시 커밋한다. Windows full smoke 실패 시 `git bisect` 로 좁힌다.

## 로그

| 날짜 | 단계 | 결과 | 커밋 |
|---|---|---|---|
| | | | |
