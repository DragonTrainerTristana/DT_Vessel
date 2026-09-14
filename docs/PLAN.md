# PLAN — Python 리팩토링 실행안

작성 2026-09-14. 근거 = `docs/MAP.md` + `Python/run_repro.sh`(common_env 포함) 2개뿐.
Python 소스는 안 엶 → MAP.md 에 없는 것은 **확인 필요** 로 남김. 대안 비교·장단점 토론 없음. 결정과 순서만.

전제(고정):
- 학습 정본 = `run_repro.sh` 경로. 오케스트레이터는 셸, Python 은 워커.
- run_repro.sh 가 워커를 `$HERE` 상대경로로 부름 → **파일을 옮기면 run_repro.sh 도 같은 커밋에서 고침.**
- `__init__.py` 0개(MAP.md "import 위험 지점") → 서브디렉터리로 내려간 워커는 `import config` 가 안 잡힘. **이동 단계마다 sys.path 확보가 필수 작업임.**
- 검증은 이미 있는 것만 씀: 드리프트 검사 · `_verify_ppo_mirror.py` · `_verify_comm_mirror.py` · `test_golden.py --check` · `test_vessel_gym_fidelity.py`. 새로 안 만듦.

---

## 기능 그룹 최종안

| 그룹명 | 목표 경로 | 소속 파일 | 근거 |
|---|---|---|---|
| **shared** | `Python/` (이동 없음) | `run_repro.sh` · `config.py` | 오케스트레이터와 드리프트 검사의 짝. `config.py:29` PROJECT_ROOT 가 `../../..` 라 깊이가 바뀌면 전 경로가 어긋남 |
| **core-train** | `Python/` (이동 없음) | `networks.py` · `vessel_gym.py` · `vessel_gym_train.py` · `ckpt_io.py` | fan-in 최상위(12·13·9·9) + `ckpt_io ↔ vessel_gym_train` [CYCLE] 지연 import. 옮기면 전 워커의 sys.path 가 동시에 흔들림 |
| **eval** | `Python/eval/` | `eval_ckpt.py` · `eval_mixed.py` · `diag_ckpt.py` · `measure_regimes.py` · `corridor_run.py` · `compose_voyage.py` | 전부 `ckpt_io + config + vessel_gym + vessel_gym_train` 묶음을 함께 import 하는 같은 소비자군. `measure_regimes → compose_voyage` 는 사람이 숫자를 옮기는 파이프라 같은 방에 둠 |
| **verify** | `Python/verify/` | `_verify_ppo_mirror.py` · `_verify_comm_mirror.py` · `test_golden.py` · `test_vessel_gym_fidelity.py` · `test_ckpt_compat.py` · `golden/` | run_repro.sh preflight 4관문 + 골든 데이터. 한 덩어리로 움직여야 preflight 수정이 1곳으로 끝남 |
| **analysis** | `Python/analysis/` | `metric_io.py` · `analyze_run.py` · `analyze_trajectory.py` · `analyze_circling_safety.py` · `analyze_timeout_safety.py` · `convergence_gate.py` | metric CSV 만 읽고 모델 안 엶. fan-out 1(`metric_io`). run_repro.sh 가 안 부르는 완전한 잎 |
| **fidelity** | `Python/fidelity/` | `radar_fidelity_compare.py` · `unity_fidelity_compare.py` | Unity↔gym 대조 2종. `../../../Build` 상대경로 [DUP] 를 공유 |
| **plotting** | `Python/plotting/` (이동 없음) | 현행 13개 | `regenerate_all.py` 가 `cwd=FIG` 로 돌려 `import paper_style` 를 성립시키는 구조 + `astar_fig9/paper_style.py:10` 이 `../plotting/` 을 고정 전제. 옮길 이득 없음 |
| **astar_fig9** | `Python/astar_fig9/` (이동 없음, 1개 편입) | 현행 4개 + `eval_astar_global.py` 편입 | MAP.md "`Python/` 루트에 있으나 astar_fig9 세트의 일부, 소비자는 `make_fig9_from_eval.py`", 04 에 과거 위치 이력 있음 |
| **unity-island** | `Python/_archive/unity_path_2026-09/` | `main.py` · `frame_stack.py` · `memory.py` · `functions.py` · `obs_utils.py` | 사용자 전제. 셋 다 fan-in 1(=main), `functions` 는 2(main·memory) — 섬 밖 소비자 0 |
| **deprecated** | `Python/_archive/deprecated_2026-09/` | `_smoke_fullmoe.py` · `inspect_channel_freeze.py` | 아래 "_archive 로 보낼 목록" 참조 |

배분 검산: `Python/` 직속 현행 32개 = shared·core 5 + eval 6 + verify 5 + analysis 6 + fidelity 2 + astar 편입 1 + unity-island 5 + deprecated 2. **미배정 0개.**

---

## VESSEL_* 환경변수 인벤토리

**동결 대상** = 이름 · 기본값 · [FIXED] 여부 3개가 리팩토링 전후로 동일해야 함.
`[FIXED]` = run_repro.sh 가 조건 없이 `export` 해 사용자 값을 덮어씀 → 밖에서 못 바꿈.

### A. common_env — YUGIOH 31개 (전부 [FIXED])

`:70-102`. 전부 `export X=값` 형식이라 호출 전에 무엇을 주든 덮어씀. 읽는 위치는 전부 `config.py`(MAP.md: 모든 env 토글의 단일 정본).

| 이름 | 기본값 | 읽는 위치 | 의미 | 드리프트 |
|---|---|---|---|---|
| `VESSEL_USE_ATTENTION` | 1 | :71 → config | GroundedAttention 집계 | ✔ |
| `VESSEL_CENTRAL_CRITIC` | 1 | :72 → config | CTDE critic (critic fc2 shape 결정자) | ✔ |
| `VESSEL_STATE_RECON_COEF` | 0.05 | :73 → config | 상태복원 aux 계수 (>0 이면 state_recon 키 생성) | ✔ |
| `VESSEL_USE_MOE` | 1 | :74 → config | 상황별 코어 5벌 (`experts.` 접두 결정자) | ✔ |
| `VESSEL_MOE_SHARED` | 1 | :75 → config | 전문가 간 레이더 인코더 공유 | ✔ |
| `VESSEL_MOE_WIDTH` | 1.0 | :76 → config | iso-param 폭 | ✔ |
| `VESSEL_SHARED_ENCODER` | all | :77 → config | 인코더 망 간 공유(키 불변 → 스냅샷이 유일 근거) | ✔ |
| `VESSEL_RADAR_ACT` | leaky | :78 → config | 인코더 활성 | ✔ |
| `VESSEL_RADAR_HEAD` | bottleneck | :79 → config | 인코더 헤드(키 결정자) | ✔ |
| `VESSEL_RADAR_BOTTLENECK_CH` | 8 | :80 → config | 1×1 bottleneck 채널 | ✔ |
| `VESSEL_MSG_LN` | 1 | :81 → config | msg_ln LayerNorm(키 결정자) | ✔ |
| `VESSEL_MSG_TOKEN_GAIN` | 8.0 | :82 → config | attention 토큰 안 msg 상수배 | ✔ |
| `VESSEL_CLIP_PER_MODULE` | 1 | :83 → config | 망별 grad clip | ✔ |
| `VESSEL_MSG_L2` | 0.0002 | :84 → config(`MSG_L2_COEF`) | 메시지 L2 | ✔ |
| `VESSEL_POS_GROUND` | 1 | :85 → config | relpos+msg_encoder mean 집계 | ✔ |
| `VESSEL_COMM_RANGE` | 300 | :86 → config | 통신 반경 = 보상 반경 | ✔ |
| `VESSEL_MAX_PARTNERS` | 4 | :87 → config(`MAX_COMM_PARTNERS`) | nearest-K | ✔ |
| `VESSEL_RADAR_RANGE` | 56 | :88 → config | obs[0:360] 정규화 상수. 바뀌면 관측 의미가 바뀜 | ✔ |
| `VESSEL_COLREGS_MODE` | unity | :89 → config | COLREGs 판정 모드 | ✔ |
| `VESSEL_SIM_COLREGS_COEF` | 0.45 | :90 → config(`COLREGS_SIM_COEF`) | 보상 내 COLREGs 계수 | ✔ |
| `VESSEL_INTENT_K` | 3 | :91 → config | intent 디코더 출력 K(shape 결정자) | ✔ |
| `VESSEL_THREAT_COEF` | 0 | :92 → config | 구 aux 디코더 계수 | ✔ |
| `VESSEL_GOAL_COMM_COEF` | 0 | :93 → config | 구 aux 디코더 계수 | ✔ |
| `VESSEL_INTENT_COEF` | 0 | :94 → config | 구 aux 디코더 계수 | ✔ |
| `VESSEL_ROLE_COMM_COEF` | 0 | :95 → config | 구 aux 디코더 계수 | ✔ |
| `VESSEL_COMM_CONSUMER_COEF` | 0 | :96 → config | consumer 디코더 계수 | ✔ |
| `VESSEL_RECON_EMA_FLOOR` | 0 | :97 → config | state_recon EMA 하한 | ✔ |
| `VESSEL_AGG_MODE` | sum | :98 → config | others_msg 집계 모드 | ✔ |
| `VESSEL_MSG_GAIN` | 1.0 | :99 → config | 집계 끝단 상수배 | ✔ |
| `VESSEL_TIMEOUT_BOOTSTRAP` | 0 | :100 → config | 타임아웃 부트스트랩 | ✔ |
| `VESSEL_MSG_GATE_APPLY` | 0 | :101 → config | 게이트 적용 | ✔ |

- **드리프트 검사 커버리지 = 31개 전부**(2026-09-14 15→31 확장 완료, preflight `:111-115` 의 name 목록). 확장 시점 실행 결과 전부 PASS — common_env export 값과 config 기본값이 어긋난 항목 없음.
- 골든 실행만 예외: `:130` 이 `VESSEL_STATE_RECON_COEF` · `VESSEL_CENTRAL_CRITIC` · `VESSEL_USE_ATTENTION` 3개를 `env -u` 로 **지우고** `test_golden.py --check` 를 돌림. 이 3개 제거를 빠뜨리면 골든이 깨짐.

### B. 팔·런마다 스크립트가 정하는 것 ([FIXED])

| 이름 | 값 | 읽는 위치 | 의미 |
|---|---|---|---|
| `VESSEL_MSG_DIM` | off 6 / on6 6 / on12 12 / rand 6 / smoke 6 | :156(train) · :180(eval) | 메시지 차원. `train_one`/`eval_one` 3번째 인자 |
| `VESSEL_USE_COMM` | arm=OFF → 0, 그 외 1 | :160 | 로그 식별용. 실동작은 `--arm` 이 정함(주석 :157-159) |

### C. 밖에서 바꿀 수 있는 것 (오버라이드 가능)

| 이름 | 기본값 | 읽는 위치 | 의미 |
|---|---|---|---|
| `VESSEL_PY` | `python` | :40 | 파이썬 실행 파일 |
| `VESSEL_OUT_DIR` | `$HERE/_repro_out` | :41 | CSV·로그 출력. **$HERE 종속 — run_repro.sh 이동 시 바뀜** |
| `VESSEL_CKPT_DIR` | `$HOME/VESSEL_checkpoints/<날짜>_repro` | :42 · :269(diag 에 인라인 주입) | 체크포인트 위치. 워커측 미설정 기본값은 `Python/checkpoints` [DUP] `eval_mixed.py:31-32` · `ckpt_io.py:441` |
| `VESSEL_SEEDS` | `43 44 45` | :43 | 시드 목록 |
| `VESSEL_NGPU` | torch 자동감지(0장이면 1) | :48-49 | GPU 수 |
| `VESSEL_JOBS` | `NGPU × 2` | :58 | 동시 실행 프로세스 수 |
| `VESSEL_SKIP_GOLDEN` | 0 | :128 | 1 이면 골든+fidelity 건너뜀. **완료 조건 판정 때 쓰면 안 됨** |
| `VESSEL_COMM_ON_AT` | 0 | :162 | 통신 커리큘럼 전환 스텝 |
| `VESSEL_CKPT_EVERY` | 2 | :163 | 체크포인트 주기 |
| `VESSEL_TRAIN_ARMS` | `off on6 on12` | :211 | train 모드 팔 선택 |
| `VESSEL_DIAG_CKPTS` | **없음(필수, `:?`)** | :264 | diag 대상 체크포인트 목록(CK 하위 상대경로) |
| `VESSEL_DIAG_ARGS` | 빈 값 | :270 | diag_ckpt 추가 인자 |
| `VESSEL_DIAG_COMM_RANGE` | 미설정 | :262 | 설정 시 `VESSEL_COMM_RANGE` 를 **덮어씀** — [FIXED] COMM_RANGE 의 유일한 탈출구(diag 모드 한정) |

### D. 주석에만 있고 run_repro.sh 가 export 하지 않는 것 (사용자가 직접 줘야 함)

| 이름 | 읽는 위치 | 의미 |
|---|---|---|
| `VESSEL_MSG_RANDOM_SD` | :243 주석 | RANDOM 팔 난수 sd. diag_ckpt 의 msg_sd 를 보고 사람이 넣음 |
| `VESSEL_COMM_TELEMETRY` · `VESSEL_COMM_TELEMETRY_EVERY` | :207 주석 | 학습 중 통신 텔레메트리(`*_comm.csv`) |

### E. env 아닌 [FIXED] 인자·변수 (같이 동결)

- 학습 CLI `:162-164` : `--envs 128 --vessels 16 --rollout 32 --ring 1.0 --crossing 2 --max_partners 4`, `--steps` = smoke 40000 / train·random 16056320. `--max_partners 4` 는 `VESSEL_MAX_PARTNERS=4` 와 **이중 정의 [DUP]**.
- 평가 CLI `:185` : `--envs 256 --eval_decisions 10000 --burnin 2400`.
- 비-VESSEL env : `PYTHONIOENCODING=utf-8`(:36) · `CUDA_VISIBLE_DEVICES`(:154·:178, 라운드로빈) · `OMP_NUM_THREADS=2`(:155·:179).

### F. run_repro.sh 밖의 VESSEL_* (MAP.md 기재분, 참고)

`VESSEL_LOG_DIR` · `VESSEL_FIG_XMAX` (`plotting/regenerate_all.py:12-13` 이 자식에 주입, SCR [DUP] 8곳) · `VESSEL_LOAD_MODEL`(`main.py:915-949`).
이 3개는 run_repro.sh 가 안 건드림 → 동결 대상 밖. plotting/unity-island 단계에서만 봄.

---

## 작업 순서

1. **shared + core-train 동결** — 모든 단계의 완료 조건이 여기서 나옴. `config.py:29` PROJECT_ROOT 와 [CYCLE] 때문에 "이동 없음"을 먼저 확정해야 아래가 성립함.
2. **unity-island → _archive** — fan-in 이 섬 안에서 닫혀 있어(소비자 = main.py 하나) 참조 끊기 없이 통째로 빠짐. 그래프가 5개 줄어 뒤 단계가 쉬워짐.
3. **deprecated → _archive** — `_smoke_fullmoe.py` · `inspect_channel_freeze.py`. import 0 / 피의존 0 인 완전 고립.
4. **analysis → `analysis/`** — run_repro.sh 가 안 부르는 잎. 오케스트레이터를 안 건드리는 첫 실이동이라 sys.path 확보 방식을 여기서 확정함.
5. **fidelity → `fidelity/`** — 같은 성격의 잎 2개. 4단계에서 정한 sys.path 방식을 그대로 적용.
6. **plotting 위치 확정 + astar_fig9 편입** — 두 디렉터리는 안 옮김을 확정하고, `eval_astar_global.py` 만 `astar_fig9/` 로 넣음. `write_fig_code.py:13` 깨진 경로도 여기서 고침.
7. **eval → `eval/`** — run_repro.sh `eval`·`diag` 모드가 `$HERE` 로 부르는 대상 → 셸과 동시 수정. 잎보다 위, 하네스보다 아래.
8. **verify → `verify/`** — preflight 관문 자체. 1~7 을 **원래 자리의 하네스**로 검증한 뒤 마지막에 옮기고, 옮긴 직후 자기 자신으로 재검증.

---

## 각 단계의 완료 조건

**공통 기본: `bash run_repro.sh smoke` 통과** (rc 전부 0).
smoke 는 preflight 를 포함하므로 **드리프트 검사 + 미러 2종 + 골든 + fidelity 가 이미 다 들어감**. `VESSEL_SKIP_GOLDEN=1` 로 돌린 실행은 완료 판정에 쓰지 않음.

⚠️ **실행 플랫폼 확인 필요** — `_verify_ppo_mirror.py` 는 Windows 전용(Mac 은 torch↔numpy 비호환, 루트 규약 §8 기준). Mac 에서는 preflight 가 이 관문에서 멈출 수 있음 → **완료 판정 smoke 는 Windows 에서 돌리는 것을 기본으로 함.** Mac 에서 작업하는 단계는 `_verify_comm_mirror.py` + `test_golden.py --check` + `test_vessel_gym_fidelity.py` 까지만 근거로 쓰고 그 사실을 기록.

| 단계 | 완료 조건 |
|---|---|
| 1. shared+core | `bash run_repro.sh smoke` PASS. preflight 출력에 `YUGIOH 드리프트: PASS` 문자열 확인(2026-09-14부터 31개 전부 대조, 수동 대조 불필요). config.py 를 한 줄이라도 건드렸으면 `test_golden.py --check` 가 `ALL PASS` 여야 함(기본값 비트동일). |
| 2. unity-island 격리 | smoke PASS(= 코어 무영향 확인). `main.py` 외 소비자가 없던 4개라 끊긴 엣지 0 이어야 함. `ckpt_io.snapshot_config` 는 남으므로 Unity `.pth` 스냅샷 키 정의는 보존됨 |
| 3. deprecated 격리 | smoke PASS. preflight 가 `_smoke_fullmoe.py` 를 안 부르므로 출력이 2단계와 동일해야 함 |
| 4. analysis 이동 | smoke PASS(무변화 확인) + **이동한 6개 각각 직접 실행**해 `metric_io` import 와 RUN_DIR 해석이 이전과 같은 절대경로인지 확인. run_repro.sh 는 이 그룹을 안 부르므로 smoke 만으로는 회귀가 안 잡힘 |
| 5. fidelity 이동 | smoke PASS + `radar_fidelity_compare.py` · `unity_fidelity_compare.py` 가 계산하는 Build 경로 문자열이 이동 전과 같은 곳을 가리키는지 확인(Build 실물 없으면 경로 문자열만) |
| 6. plotting/astar | `plotting/regenerate_all.py` 1회 완주 + `astar_fig9/make_fig9_from_eval.py` 가 shim 경유로 `paper_style` 을 올리는지 확인. `eval_astar_global.py` 는 이동 후 CLI 1회 실행 |
| 7. eval 이동 | **`bash run_repro.sh smoke` PASS 필수** + 체크포인트가 있으면 `bash run_repro.sh eval` · `VESSEL_DIAG_CKPTS=… bash run_repro.sh diag` 각 1회. diag 는 게이트 3개를 통과해 JSON 이 나와야 함(rc≠0 = 숫자 없음) |
| 8. verify 이동 | `bash run_repro.sh smoke` 가 **새 경로의 preflight 4관문 전부** 통과 + `_golden.txt`·`_verify_ppo.txt`·`_verify_comm.txt` 에 `ALL PASS` grep 성공 + 골든 JSON 이 이동 전과 같은 파일을 읽는지 확인. `test_ckpt_compat.py` 는 Windows 체크포인트가 필요해 이 저장소에서 검증 불가 — **미검증으로 명시 기록** |

---

## 각 단계의 리스크

**모든 이동 단계 공통**
- `__init__.py` 0개 + sys.path 조작 없음 → 서브디렉터리로 내려간 파일은 `import config`/`import metric_io` 가 **ImportError 로 즉사**. 4단계에서 방식을 정하고 5~8 이 그대로 따름.
- `config.py:347-348` : import 만 해도 `models/<COMM_FOLDER>/…/logs` 를 만듦. 이동 검증 중 빈 폴더가 생기는 건 정상 — 실패 신호로 오독하지 말 것.

**1. shared + core-train**
- `config.py:29` PROJECT_ROOT = `../../..`. config 가 한 칸이라도 내려가면 `models/` `trajectory_data/` `figures/` 경로가 전부 어긋남 → 이동 금지 근거.
- [CYCLE] `ckpt_io.py:145,183` ↔ `vessel_gym_train.py:676` 지연 import. 둘 다 루트 유지라 이번엔 안 터지지만, 어느 한쪽만 옮기면 지연 import 가 새 경로를 못 찾음.
- 드리프트 검사 = 31/31 전부 대조(2026-09-14 확장, 확장 시점 PASS). 이전엔 15/31 만 봐서 나머지 16개가 어긋나도 못 잡았으나 지금은 해소됨.
- [DUP] `Python/smoke_mac.sh`(2026-09-14 신규)의 `common_env()` 가 `run_repro.sh:70-102` 를 그대로 복사한 사본임. bash 3.2(macOS 기본)에서 `source <(...)` 로 함수를 재사용하는 방식이 정의를 조용히 누락시켜(실측) 어쩔 수 없이 복사함 — 지금은 두 파일 모두 손대는 사람이 smoke_mac.sh 헤더의 diff 커맨드로 직접 동기화를 확인해야 함. **shared 를 손댈 때 이 복사를 해소할 후보**(예: 진짜 공용 소스로 뽑아내거나 다른 재사용 방식을 다시 찾는 것).

**2. unity-island → _archive**
- `main.py:20` `from config import *` 와일드카드 — 이름 출처 추적 불가. 격리 자체는 안전하나, **Unity 경로가 CLAUDE.md 상 ground-truth 판정관**이라 되돌릴 수 있게 `_archive` 안에 README(사유·복구법)를 남길 것(export_onnx/worldmap 선례와 동일).
- Unity 체크포인트 `.pth` + `_reward_rms.npz` 형식 [DUP] 의 생산자가 사라짐. `ckpt_io.snapshot_config` 는 남으므로 키 정의는 보존됨 — 소비자 없는 코드가 되는 것만 인지.
- `obs_utils.py` 는 obs 369D 정의 [DUP] 6곳 중 하나 → 격리해도 **나머지 5곳은 그대로 살아 있음**(계약은 안 바뀜).

**3. deprecated → _archive**
- `_smoke_fullmoe.py:46-48` 의 `importlib.reload(config)` — 프로세스 내 모듈 전역 교체. 격리로 이 부작용이 없어지는 쪽이라 리스크는 감소.
- `_smoke_fullmoe.py` 는 C절이 2026-09-05 `action_raw` 변경 이후 상시 FAIL. 격리 후 누가 "검증이 사라졌다"고 오해하지 않게 README 에 상시 FAIL 사실을 적을 것.

**4. analysis → `analysis/`**
- **[MOVE-RISK]** RUN_DIR = `<repo>/../../../run_logs` **[DUP] 4곳** (`analyze_circling_safety.py:32` · `analyze_timeout_safety.py:18` · `analyze_trajectory.py:11` · `convergence_gate.py:29`). 깊이 +1 → 4곳 전부 한 칸 조정. 하나 빠뜨리면 조용히 다른 폴더를 읽음.
- **[MOVE-RISK]** MAP.md 문구: "`astar_fig9/paper_style.py:10` … **analysis 그룹 이동 시 반드시 함께 수정할 것**". 본 안에서는 plotting/astar 를 안 옮기므로 실제 수정은 6단계에서 확인하되, **4단계 종료 시 이 상대경로가 여전히 성립하는지 반드시 재확인**.
- `analyze_trajectory.py:12-15` · `convergence_gate.py` 의 run 태그→CSV `FILES` 딕셔너리 고정 — 경로 기준이 바뀌면 매핑이 헛돎.

**5. fidelity → `fidelity/`**
- **[MOVE-RISK]** `unity_fidelity_compare.py:144` · `radar_fidelity_compare.py:168` **[DUP]** : `../../../Build/0703/Vessel_MLAgent.exe`. 깊이 +1 → 2곳 동시 수정.
- `radar_fidelity_compare.py:37-38` : Windows temp 절대경로 2줄(이 맥에 없음). 이동과 무관하게 이미 깨진 값 — 고치려면 사용자 확인 필요(데이터 소스 결정).
- `unity_fidelity_compare.py` 는 Unity 연결부가 TODO 로 비어 있음(자체 docstring). "동작 확인"이 애초에 불가 → 완료 조건은 import 성공·경로 문자열까지.

**6. plotting 위치 확정 + astar_fig9 편입**
- **[MOVE-RISK] 핵심** `astar_fig9/paper_style.py:10` 이 `../plotting/paper_style.py` 를 `spec_from_file_location` 으로 **파일 경로 직결** — 두 디렉터리 상대 위치가 고정 전제. 둘 다 안 옮기는 것이 본 안의 결정이지만, `eval_astar_global.py` 편입으로 `astar_fig9/` 내용물이 바뀌므로 편입 후 1회 실행으로 shim 이 여전히 뜨는지 확인.
- `plotting/*.py` 7개는 sys.path 조작 없이 `import paper_style` → **CWD 가 `plotting/` 이어야 함**. `regenerate_all.py:32` 의 `cwd=FIG` 가 유일한 보증. 직접 실행으로 검증하면 위치에 따라 깨짐(코드 문제 아님).
- `plotting/write_fig_code.py:13` : `PY = FIG/../Assets/Scripts/Python` 이 **이미 깨져 있음**(실제 `Python/Assets/Scripts/Python`). 이번 단계에서 고침 — 단 "무엇을 가리켜야 하는지"는 MAP.md 에 없음 → **확인 필요**.
- `plotting/make_ablation_rewards.py:29` : Windows 스크래치패드 절대경로(세션 UUID 포함) 기본값. 이 맥에 없음 → 실행 검증 불가. 대체 값은 **확인 필요**.
- `eval_astar_global.py` 는 `ckpt_io + vessel_gym_train` 묶음을 import 하는 평가군 → `astar_fig9/` 로 내려가면 sys.path 확보가 반드시 필요. 소비자 `make_fig9_from_eval.py` 는 import 가 아니라 **출력 텍스트 줄 파싱**이라 경로 결합 방식은 **확인 필요**.

**7. eval → `eval/`**
- **$HERE 상대경로 호출** : `run_repro.sh:183`(eval_ckpt.py) · `:269`(diag_ckpt.py) 를 **같은 커밋에서** 새 경로로 고칠 것. 고치지 않으면 eval/diag 모드가 "파일 없음"으로 죽음(smoke 는 train 만 부르므로 **smoke 통과로는 안 잡힘** — 7단계 완료 조건에 eval·diag 1회 실행을 넣은 이유).
- **[MOVE-RISK]** `eval_mixed.py:31-32` 의 `VESSEL_CKPT_DIR` 미설정 기본값 `Python/checkpoints` **[DUP]** (`ckpt_io.py:441` 과 짝). `ckpt_io` 는 루트에 남고 `eval_mixed` 만 내려가므로 **두 기본값이 서로 다른 폴더를 가리키게 됨** → 동시 수정.
- `eval_mixed.py → plotting/make_mixed_fleet.py` 는 `mixed_fleet.csv` 파일 전달(import 아님). 출력 경로가 상대라면 깊이 변화로 어긋남 — **확인 필요**.
- `measure_regimes.py → compose_voyage.py` 는 사람이 숫자를 옮기는 경계라 코드상 깨질 것 없음. 같은 방에 두는 이유가 이것뿐임을 문서에 남길 것.
- `corridor_run.py:3-5` 의 부산·대만 실측 좌표는 docstring+코드 리터럴 — 이동과 무관하지만 건드리지 말 것(논문 수치).
- `diag_ckpt.py:42` 가 outcome 매핑을 재정의 **[DUP]**(`vessel_gym.py:170-174` · `metric_io.py:32` 와 3중). 이동 중 통합 유혹 금지 — 게이트 판정에 영향.
- `run_repro.sh:262` 의 `VESSEL_DIAG_COMM_RANGE` 탈출구가 살아 있어야 구 체크포인트 진단이 가능.

**8. verify → `verify/`**
- **$HERE 상대경로 호출 4곳** : `run_repro.sh:122`(_verify_ppo_mirror) · `:123`(_verify_comm_mirror) · `:131`(test_golden, `cd "$HERE"` 안에서 `test_golden.py` 를 **상대 이름으로** 호출) · `:135`(test_vessel_gym_fidelity). 네 줄 동시 수정.
- **가장 위험** `test_golden.py` 는 `cwd=HERE` 로 `vessel_gym_train.py` 를 subprocess 실행함. test_golden 이 `verify/` 로 내려가면 그 HERE 가 `verify/` 가 되어 **학습기를 못 찾음**. run_repro.sh 의 `cd "$HERE"` 와 test_golden 내부의 HERE 를 **둘 다** 봐야 함.
- 골든 데이터 `Python/golden/2026-09-10_*.json` 의 경로 산출 방식은 MAP.md 에 없음 → **확인 필요**. `golden/` 을 함께 옮길지, 루트에 둘지는 이 확인 후 결정.
- `_verify_ppo_mirror.py:47-49` 의 `importlib.reload(config)`/`reload(networks)` — 모듈 전역을 갈아끼움. 경로가 바뀌면 reload 대상이 다른 모듈이 될 수 있음.
- preflight `:130` 의 `env -u` 3개 제거를 옮기는 과정에서 빠뜨리면 골든이 FAIL 로 바뀜(코드 변경 없이).
- `test_ckpt_compat.py` : `test_*.py` 인데 테스트 함수 0개 → pytest 가 수집만 하고 아무것도 안 돎. `verify/` 로 옮기면 수집 범위가 바뀔 수 있음. 이름 변경은 **사용자 확인 필요**(실행 지시가 바뀜).
- 이 단계는 **자기 자신이 안전망**이므로 반드시 마지막. 이동 직후 smoke 1회로 하네스 자체를 재검증할 것.

---

## _archive 로 보낼 목록

| 경로 | 근거 |
|---|---|
| `Python/main.py` → `_archive/unity_path_2026-09/` | 사용자 전제(main.py 섬). fan-out 7 로 최다지만 **피의존은 섬 안뿐** |
| `Python/frame_stack.py` → 위 | fan-in 1 = main.py |
| `Python/memory.py` → 위 | fan-in 1 = main.py. 이름은 "Replay Buffer" 지만 실제는 on-policy rollout 버퍼 |
| `Python/functions.py` → 위 | fan-in 2 = main.py · memory.py (섬 내부) |
| `Python/obs_utils.py` → 위 | fan-in 1 = main.py. docstring 이 참조하는 `test.py` 는 이미 삭제됨 |
| `Python/_smoke_fullmoe.py` → `_archive/deprecated_2026-09/` | C절이 2026-09-05 `action_raw` 변경 이후 **상시 FAIL**, 판정 근거로 안 씀. preflight 미호출, 파이프라인 연결점 0 |
| `Python/inspect_channel_freeze.py` → `_archive/deprecated_2026-09/` | 1회성 진단. import 0 / 피의존 0. 2026-09-14 `__main__` 가드로 부작용은 이미 해소 |

**이미 격리됨(기록만)** : `_archive/deprecated_2026-09/export_onnx.py`(현행 networks 와 불일치·실행 불가, ONNX 배포 경로 자체가 없음) · `worldmap_extract.py` + `worldmap_geometry.json`(소비자 0건).

**격리 여부 확인 필요**
- `Python/unity_fidelity_compare.py` — Unity 연결부 TODO 미완성 + 고립. fidelity/ 잔류로 잡아뒀으나 unity-island 격리와 함께 보낼지 확인 필요.
- `Python/test_ckpt_compat.py` — Windows 전용이라 이 저장소에서 안 돎. verify/ 잔류로 잡아뒀으나 판단 확인 필요.

**MAP.md 범위 밖(존재만 확인, 처분 확인 필요)** — MAP.md 는 `.py` 51개만 다룸. `Python/` 직속에 아래가 함께 쌓여 있음:
- 레거시 실행 스크립트 : `run_sweep*.ps1`(7) · `run_experiment.ps1` · `run_eval_gt.ps1` · `launch_*.{ps1,sh}`(3) · `run_aggregation_diagnostics.sh` · `install_mlagents.bat`
- 삭제된 `.py` 의 Unity `.meta` 잔해 다수(`plot_*.py.meta` · `_smoke_*.py.meta` 등)
- 산출물·부산물 : `vessel_gym_{OFF_s1,ON_s1,ON_s42}.pt` · `channel_A4.jsonl` · `debug.log` · `relu_explain.html` · `example_message_logs.csv`
- 문서 : `EXPERIMENT_STATUS.md`(루트 규약상 **인용 금지** 낡은 문서) · `something.md` · `colregs_compliance_metric.tex`
→ 전부 본 안의 8단계에 **포함하지 않음**. 처분은 별도로 물어볼 것.
