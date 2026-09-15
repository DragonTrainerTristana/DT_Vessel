# MAP (Python)

작성 2026-09-14. 근거 = `docs/_raw/01~04` + 확인용 소스 열람. 대상 = 작성 시점 Python 현행 51개(`Python/` 34, `Python/plotting/` 13, `Python/astar_fig9/` 4). `Python/_archive/` 제외.
2026-09-14 격리 2건(`export_onnx.py` · `worldmap_extract.py`) 이후 현행은 49개(`Python/` 32, `plotting/` 13, `astar_fig9/` 4). 아래 본문은 격리분을 `_archive/deprecated_2026-09/` 경로로 표기함.
_raw 결손 2건 보정함 — 02 의존그래프는 `Python/test_golden.py`(경로에 "golden" 포함 → 제외필터에 걸림)와 `Python/astar_fig9/paper_style.py`(모듈명 충돌로 드롭)가 빠져 있고, 02 PART 3(C#)은 셸 워드분할 실패로 깨져 있음. 아래는 둘 다 직접 확인해 반영한 것임.

## 기능 그룹

fan-in / 인접리스트(02 §2·§5)로 묶음. 이름이 아니라 엣지를 따름.

### 공용 코어 (fan-in 최상위, 자신은 거의 의존 없음)
- `Python/config.py` : 모든 상수·차원·env 토글의 단일 정본 (fan-in 18, 내부 의존 0)
- `Python/networks.py` : MessageActor/ControlActor/Critic + RadarEncoder + MoE 정의 (fan-in 12, → config)
- `Python/vessel_gym.py` : C# 물리·레이더·COLREGs를 옮긴 GPU 배치 시뮬 (fan-in 13, → config)
- `Python/metric_io.py` : metric CSV 9/13/15/17열 세대 자동판별 리더 (fan-in 5, 의존 0)
- `Python/plotting/paper_style.py` : 그림 rcParams·평활·시드밴드 (fan-in 8, 의존 0)

### 학습 — GPU 배치 경로 (현행 주 경로)
- `Python/vessel_gym_train.py` : 배치 PPO 루프 + comm_gather + comm_telemetry (fan-in 9 / fan-out 4)
- `Python/ckpt_io.py` : 체크포인트 저장·복원·평가 env 생성의 단일 구현 (fan-in 9 / fan-out 4)

### 학습 — Unity 경로 (별도 섬, 공용 코어만 공유)
- `Python/main.py` : ML-Agents 연결 + 자체 PPO 루프 (fan-out 7, 최다)
- `Python/frame_stack.py` : Unity용 다중 에이전트 프레임스택 (fan-in 1 = main)
- `Python/memory.py` : Unity용 rollout 버퍼 (fan-in 1 = main)
- `Python/functions.py` : GAE + RunningMeanStd (fan-in 2 = main, memory)
- `Python/obs_utils.py` : Unity obs 369D 파싱 + 통신 파트너 선택 (fan-in 1 = main)

### 평가·진단 (전부 `ckpt_io + config + vessel_gym + vessel_gym_train` 묶음을 함께 import)
- `Python/eval_ckpt.py` : 프리즈 정책 완주 평가, 위상 dephase 후 outcome pooling
- `Python/eval_mixed.py` : 혼합 함대 평가(장비 보유 비율 전체를 1회에)
- `Python/eval_astar_global.py` : A* 전역경로 + 국소회피 평가 (Fig9 실측)
- `Python/measure_regimes.py` : 순항/조우 국면별 결정당 비용 측정
- `Python/corridor_run.py` : 대만↔부산 회랑 양방향 통항 궤적 수집
- `Python/diag_ckpt.py` : 체크포인트 통신지표 진단 단일 진입점(게이트 3개)
- `Python/test_ckpt_compat.py` : 체크포인트 로드·추론 SHA256 골든 (Windows 전용)

### 분석 (metric CSV만 읽음, 모델 안 엶)
- `Python/analyze_run.py` : run 폴더 통째 수렴구간 리포트
- `Python/convergence_gate.py` : 수렴 판정 단일 게이트(마지막 30% + 5등분 추세)
- `Python/analyze_trajectory.py` : outcome 분포 시간순 5등분
- `Python/analyze_circling_safety.py` : circling / near-miss 진단
- `Python/analyze_timeout_safety.py` : timeout이 안전한 느림인지 위험한 배회인지
- 넷 다 `metric_io` 하나만 import (fan-out 1). 03 중복데이터상 서로 13~38라인 클론 다수

### 그림
- `Python/plotting/plot_h1_paired.py` `plot_h2_dimension.py` `plot_moe_axis.py` `plot_nqi.py` `plot_task_return.py` `plot_all_curves.py` `make_reward_folder.py` : `paper_style`만 import
- `Python/plotting/make_ablation_rewards.py` `build_final.py` `make_mixed_fleet.py` `write_fig_code.py` : paper_style도 import 안 함(자체 matplotlib)
- `Python/plotting/regenerate_all.py` : 위 8개를 subprocess로 순차 실행
- `Python/astar_fig9/make_fig9_from_eval.py` `make_fig9_paper.py` `route_astar_real.py` : Fig9 전용 세트

### 검증·골든
- `Python/_verify_ppo_mirror.py` : Unity 경로 rollout=update 미러 (→ config, networks)
- `Python/_verify_comm_mirror.py` : gym 경로 rollout=update 미러 (→ config, networks, vessel_gym, vessel_gym_train)
- `Python/_smoke_fullmoe.py` : MoE 완전분리 스모크 (→ config, networks)
- `Python/test_golden.py` : 학습기 기본값 비트동일 골든. **import 0 — subprocess로만 학습기 호출**
- `Python/test_vessel_gym_fidelity.py` : 배치 텐서 동역학 == 스칼라 참조 (→ vessel_gym)

### 시뮬 충실도 대조 (Unity ↔ gym)
- `Python/radar_fidelity_compare.py` : obs[0:360] 레이더 3종(장애물원·타선OBB·벽) 분리 대조
- `Python/unity_fidelity_compare.py` : 같은 action 시퀀스 궤적 대조. Unity 연결부 TODO 상태

### 내보내기·지오메트리 (독립)
- `Python/_archive/deprecated_2026-09/export_onnx.py` : ONNX 내보내기 (→ config, networks). 현행 networks와 불일치 — **2026-09-14 격리**(`export_onnx.README.md`). 아래 "이름과 실제 역할" 4번
- `Python/_archive/deprecated_2026-09/worldmap_extract.py` : WorldMap.unity YAML → `worldmap_geometry.json` — **2026-09-14 격리**(json 도 함께 이동, `worldmap.README.md`)
- `Python/compose_voyage.py` : measure_regimes 계수로 임의 거리·밀도 비용 합성
- `Python/inspect_channel_freeze.py` : 통신 채널 파라미터가 init에서 움직였는지 1회성 검사

## 진입점

### CLI (argparse)
- `Python/vessel_gym_train.py` : GPU 배치 PPO 학습. `--arm {OFF,ORACLE,ON,RANDOM} --envs --vessels --rollout --steps --ring --crossing --seed --save`
- `Python/eval_ckpt.py` : 프리즈 정책 완주 평가
- `Python/eval_mixed.py` : 혼합 함대 평가 → mixed_fleet.csv
- `Python/eval_astar_global.py` : A* + 국소회피 평가
- `Python/diag_ckpt.py` : 체크포인트 진단 → JSON + CSV
- `Python/measure_regimes.py` : 국면별 계수 측정
- `Python/corridor_run.py` : 회랑 궤적 수집
- `Python/compose_voyage.py` : 장거리 비용 합성
- `Python/ckpt_io.py` : 라이브러리지만 `python ckpt_io.py <ckpt>` 로 스냅샷 덤프
- `Python/test_golden.py` : `--check` / `--regen`
- `Python/test_ckpt_compat.py` : `--check` / `--regen` (Windows)
- `Python/astar_fig9/route_astar_real.py` `make_fig9_from_eval.py` `make_fig9_paper.py`

### 인자 없는 실행 (env 또는 고정 경로로 제어)
- `Python/main.py` : Unity ML-Agents 학습·평가. 전부 `VESSEL_*` env로 제어
- `Python/_verify_ppo_mirror.py` `_verify_comm_mirror.py` `_smoke_fullmoe.py` : 미러·스모크 검증기
- `Python/test_vessel_gym_fidelity.py` : 직접 실행 겸 pytest 대상
- `Python/analyze_run.py` : sys.argv로 run 폴더 경로
- `Python/analyze_trajectory.py` `analyze_circling_safety.py` `analyze_timeout_safety.py` `convergence_gate.py` : 고정 RUN_DIR + 내장 FILES 매핑
- `Python/radar_fidelity_compare.py` `Python/unity_fidelity_compare.py` (`export_onnx.py` · `worldmap_extract.py` 는 2026-09-14 `_archive/deprecated_2026-09/` 로 격리)
- `Python/metric_io.py` : 자체 점검용 `__main__`
- `Python/plotting/*.py` 12개 + `regenerate_all.py` : 그림 생성
- `Python/inspect_channel_freeze.py` : `__main__` 가드 안에서 `sys.argv[1:]`를 돌려 실행(2026-09-14 가드 추가). import 해도 실행 안 됨

### 셸 오케스트레이터 (import 그래프 밖)
- `Python/run_repro.sh` : preflight(미러 2종 → test_golden → fidelity) → vessel_gym_train → eval_ckpt → diag_ckpt 순으로 호출

## 경계를 넘는 의존

- **[CYCLE] `ckpt_io` ↔ `vessel_gym_train`** : `ckpt_io.py:145,183` 이 함수 안에서 `import vessel_gym_train as _vgt` 해 모듈 전역(MSG_RANDOM_SD 등)을 스냅샷 값으로 덮어씀 / `vessel_gym_train.py:676` 이 함수 안에서 `from ckpt_io import snapshot_config`. 양쪽 다 지연 import라 import 시점엔 안 터짐. pylint도 같은 1건만 보고함
- `eval_ckpt` `eval_mixed` `eval_astar_global` `measure_regimes` `corridor_run` `diag_ckpt` `test_ckpt_compat` → `vessel_gym_train` : `comm_gather` `parse_obs` `FrameStack` `make_others_msg` 를 그대로 가져다 씀 — 평가가 학습기의 rollout 집계 구현에 직접 묶여 있음
- 같은 7개 → `ckpt_io` : `restore_policy` `make_env_from_snapshot` `describe`
- `vessel_gym_train` → `vessel_gym` : VesselBatchEnv 생성·step
- `vessel_gym_train` `ckpt_io` `_verify_*` `_smoke_fullmoe` → `networks` : `CNNPolicy` (`export_onnx` 도 같은 엣지였으나 2026-09-14 격리)
- `main.py` → `config` : `from config import *` (와일드카드 — 이름 출처가 추적 안 됨)
- `main.py` → `ckpt_io` : `snapshot_config` (Unity 체크포인트도 같은 스냅샷 키를 쓰게 하는 지점)
- `memory` → `functions` : `calculate_returns`(GAE)
- `networks` `vessel_gym` `obs_utils` → `config` : 차원·토글 상수
- analyze 4종 + `convergence_gate` → `metric_io` : `read_metric` `Metric` `OUTCOMES`
- plotting 7종 → `plotting/paper_style` : `apply` `ema` `roll` `run_curve` `seed_band`
- `astar_fig9/make_fig9_from_eval` → `astar_fig9/paper_style`(shim) → `plotting/paper_style` : shim이 파일 경로로 직접 로드
- **import 아닌 경계** — 그래프에 안 나타남:
  - `plotting/regenerate_all.py` → plotting 8개 : subprocess, `cwd=FIG`
  - `test_golden.py` → `vessel_gym_train.py` : subprocess, `cwd=HERE`
  - `eval_mixed.py` → `plotting/make_mixed_fleet.py` : mixed_fleet.csv 파일 전달
  - `eval_astar_global.py` → `astar_fig9/make_fig9_from_eval.py` : 출력 텍스트 줄을 파싱
  - `measure_regimes.py` → `compose_voyage.py` : 측정 계수를 사람이 옮겨 넣음
  - `run_repro.sh` → 학습기·평가기 전부

## Unity 배포 계약 (동결 대상)

### 관측 벡터 — 369D
| index | dim | 내용 | 정규화 |
|---|---|---|---|
| `[0:360]` | 360 | 레이더 raw ray 1° | `dist/RADAR_RANGE − 0.5`, 미감지 `+0.5` |
| `[360]` | 1 | goal distance | `d/(d + goalNormK)` |
| `[361]` | 1 | goal angle (SignedAngle) | `/180` |
| `[362]` | 1 | speed | `/maxSpeed` |
| `[363]` | 1 | yaw rate | `/MaxYawRate` |
| `[364]` | 1 | heading | −180~180 변환 후 `/180` |
| `[365]` | 1 | rudder angle | `/maxTurnRate` |
| `[366:368]` | 2 | position x, z | **정규화 없음(원시 월드 좌표)**. 네트워크 입력 제외 — 통신 파트너·relpos 계산용 |
| `[368]` | 1 | COLREGs situation 0~4 | 정수 그대로. 1-step stale(의도). MoE 라우팅 키 + one-hot 5D 입력 |

- 네트워크 실입력 = radar 3프레임×360 + 2 + 4 + one-hot 5 (position 제외)
- 정의 위치 **[DUP] 6곳** — 하나만 고치면 조용히 어긋남:
  - `Python/config.py:44-66` : RADAR_RAYS 360 / STATE_SIZE 360 / GOAL_SIZE 2 / SELF_STATE_SIZE 4 / POSITION_SIZE 2 / SITUATION_SIZE 1 / OBSERVATION_SIZE 369 / FRAMES 3 / COLREGS_SIZE 0
  - `Python/vessel_gym.py:675-709` `_build_obs` — 조립 순서 정본(gym 경로)
  - `Python/vessel_gym_train.py:36-42` `parse_obs` — 슬라이스 숫자 `0:360 / 360:362 / 362:366 / 368` 을 리터럴로 박음
  - `Python/obs_utils.py:9-33` `parse_observation` — STATE_SIZE 기준 산출(Unity 경로)
  - `Agent/VesselAgent.cs:958-1004` `CollectObservations` + `:333` `VectorObservationSize = radarObsSize + 2 + 4 + 2 + 1`
  - `GlobalScale.cs:105` `RADAR_RAYS = 360`

### 행동 공간
- 타입: 연속 2D, squashed Gaussian (Normal 샘플 → tanh)
- 범위: 출력 `[-1, 1]`. `action_mean` clamp `±3.0` (`networks.py:810`, tanh(3)≈0.995), `logstd` clamp `[-2.3, 0]` → std 0.1~1.0 (`networks.py:811`, 재계산은 `:841`)
- per-dim logstd 초기값 `[-1.0, -0.5]` (`networks.py:641`)
- 차원 상수: `CONTINUOUS_ACTION_SIZE = 2` (`Python/config.py:54`)
- 의미·환경 해석 **[DUP] 2곳** (수식 동일):
  - `Python/vessel_gym.py:415-420` `_apply_action` : `a[0]·MAX_TURN_RATE` = 명령 타각, `(a[1]+1)·0.5·max_speed` = 목표속도
  - `Agent/VesselAgent.cs:489-498` `OnActionReceived` : `Clamp(a[0],-1,1)·maxTurnRate` / `(Clamp(a[1],-1,1)+1)/2·maxSpeed`
- pre-tanh `action_raw` 를 rollout이 저장하고 update가 그대로 재사용 (`networks.py:803-827`, `:829-848`) — atanh 역변환 안 씀

### 관측 정규화
- 방식: **러닝 통계 없음.** 전 원소가 obs 생성 시점에 고정 상수로 나눠짐(위 표). 관측용 정규화 파라미터를 학습하거나 저장하는 코드 없음
- 다만 스케일 상수 자체는 env로 바뀜 — `RADAR_RANGE` 기본 56.0 (`Python/config.py:456`, `VESSEL_RADAR_RANGE`). 바뀌면 `[0:360]` 의 의미가 바뀜
- 저장 위치: `cfg_snapshot` 의 `radar_range` 키 (`Python/ckpt_io.py:72` `snapshot_config`) — 체크포인트 안에 들어감. 복원은 `restore_policy` / `make_env_from_snapshot` 이 이 값으로
- **보상** 정규화는 별개이고 저장 방식이 두 갈래 [DUP]:
  - gym: `ValueNorm` 상태를 체크포인트 `value_norm` 키에 (`vessel_gym_train.py:1021,1075`)
  - Unity: `reward_rms` 를 체크포인트가 아닌 별도 `*_reward_rms.npz` 로 (`main.py:875-876`)

### 모델 내보내기
- **현재 실동작하는 배포 경로는 ONNX가 아님.** C# 전체에 Barracuda/Sentis/NNModel 참조 0건 — Unity 안에서 추론하지 않음. `Python/main.py:915-949` 가 `VESSEL_LOAD_MODEL=1` 에서 `torch.load` → `load_state_dict` 로 정책을 올리고 mlagents_envs 로 Unity를 구동함
- `Python/_archive/deprecated_2026-09/export_onnx.py`(2026-09-14 격리) 는 **현행 `networks.py` 와 맞지 않음** — 실행 시 깨짐:
  - `control_actor.conv1 / conv2 / fc1` 참조(`:25-28`). 현행에서 `conv1/conv2` 는 `RadarEncoder`(`networks.py:207-208`) 안에만 있고 ControlActor엔 없음
  - 입력을 `obs: [batch, 373]` 로 가정(`:32`). 현행 계약은 369D
  - `COLREGS_SIZE` 슬라이스를 씀(`:40-41`). 현행 값은 0
  - frame-stack 3프레임을 같은 프레임 3복제로 위조(`:37-38`)
  - 출력 파일명 고정: `<PROJECT_ROOT>/models/VesselNavigation_16M.onnx`, `<PROJECT_ROOT>/Assets/Models/…` (`:17-18`)
- 체크포인트 형식 **[DUP] 2종** (경로마다 키·확장자가 다름):
  - gym `.pt` (`vessel_gym_train.py:1021`, `:1075`) : `model_state_dict, arm, seed, steps, comm_active, value_norm, cfg_snapshot, optimizer_state_dict`
  - Unity `.pth` (`main.py:866-876`) : `model_state_dict, msg_anneal_step, arm, cfg_snapshot` + 옆에 `_reward_rms.npz`
  - `cfg_snapshot` 생성기는 `Python/ckpt_io.py:34` `snapshot_config` 하나 — gym `_cfg_snapshot()` 과 Unity `_unity_snapshot()` 둘 다 이걸 부름. 여기 담기는 것: arm·msg_dim·code_version·use_attention·pos_ground·central_critic·state_recon_coef·use_moe·moe_shared·moe_width·msg_ln·comm_range·max_partners·comm_on_at·ring·crossing·vessels·envs·rollout·seed·radar_act·radar_head·radar_bottleneck_ch·radar_range·shared_encoder·situation_input·radar_feat_dim·attn_dim·보조손실 계수 전부

## 설정·경로·하이퍼파라미터가 하드코딩된 곳

- `Python/plotting/make_ablation_rewards.py:29` : 데이터 소스 기본값이 Windows 스크래치패드 절대경로(세션 UUID 포함). 이 맥에 존재하지 않음
- `Python/radar_fidelity_compare.py:37-38` : 같은 성격의 Windows temp 절대경로 2줄
- `Python/plotting/write_fig_code.py:13` : `PY = FIG/../Assets/Scripts/Python`. 이 파일 실제 위치가 `Assets/Scripts/Python/plotting/` 이라 `..` = `Python/` → `Python/Assets/Scripts/Python` 은 없음. **현재 깨진 경로**
- `RUN_DIR = <repo>/../../../run_logs` **[DUP] 4곳** : `analyze_circling_safety.py:32` `analyze_timeout_safety.py:18` `analyze_trajectory.py:11` `convergence_gate.py:29`
- `SCR = VESSEL_LOG_DIR 또는 FIG/_data` + `metrics_v2.txt` **[DUP] 8곳** : `plotting/` 의 `build_final.py:13,80` `make_mixed_fleet.py:23` `plot_all_curves.py:19,112` `plot_h1_paired.py:16,37` `plot_moe_axis.py:23,45` `plot_nqi.py:17` `plot_task_return.py:23,38` `make_reward_folder.py:25,116,219`
- `Python/plotting/regenerate_all.py:12-13` : `VESSEL_LOG_DIR=FIG/_data`, `VESSEL_FIG_XMAX=15000000` 를 자식 프로세스에 주입
- `Python/config.py:29` : `PROJECT_ROOT = <파일위치>/../../..`
- `Python/_archive/deprecated_2026-09/export_onnx.py:17-18` : 출력 파일명 `VesselNavigation_16M.onnx` 고정
- `Python/_archive/deprecated_2026-09/worldmap_extract.py:16` : `Assets/Scenes/WorldMap.unity` (씬 파일은 존재·빌드 등록 상태. 스크립트만 격리)
- `Python/unity_fidelity_compare.py:144` · `Python/radar_fidelity_compare.py:168` **[DUP]** : Unity 빌드 기본값 `../../../Build/0703/Vessel_MLAgent.exe`
- `Python/eval_mixed.py:31-32` · `Python/ckpt_io.py:441` **[DUP]** : `VESSEL_CKPT_DIR` 미설정 시 `Python/checkpoints`
- `Python/corridor_run.py:3-5` : 부산(-82066,-24438)·대만(-79863,-18304)·직선 6517.6 유닛을 docstring과 코드에 실측값으로 박음
- `Python/analyze_trajectory.py:12-15` · `convergence_gate.py` : run 태그 → CSV 파일명 `FILES` 딕셔너리 고정
- `Python/vessel_gym_train.py:36-42` : obs 슬라이스 인덱스를 config 상수 대신 숫자로 **[DUP]** (config.py:44-66 과 이중 정의)
- `Python/vessel_gym.py:170-174` : outcome 코드 0~4를 모듈 상수로. `Python/diag_ckpt.py:42` 가 같은 매핑을 이름으로 다시 정의 **[DUP]**, `Python/metric_io.py:32` `OUTCOMES` 도 같은 4종을 문자열로 다시 정의 **[DUP]**
- `Python/config.py` 전체 : 보상 12항 계수·PPO 상수·YUGIOH 기본값이 리터럴. 단일 출처라 의도된 하드코딩임

## import 위험 지점

- **패키지 아님 — `__init__.py` 0개** (`Python/`, `plotting/`, `astar_fig9/` 전부). `import config` `import networks` 류가 전부 CWD 또는 sys.path에 의존. 실행 위치가 `Python/` 이어야 동작
- `Python/plotting/*.py` 7개 : `import paper_style` 인데 sys.path 조작 없음 → **CWD가 `Python/plotting/` 이어야 함**. `regenerate_all.py:32` 가 `cwd=FIG` 로 돌려 이 조건을 만족시키는 구조. 직접 실행하면 위치에 따라 깨짐
- `Python/astar_fig9/make_fig9_from_eval.py:32-34` : `sys.path.insert(0, HERE)` 후 `import paper_style` → `astar_fig9/paper_style.py`(shim)가 잡힘
- `Python/astar_fig9/paper_style.py:7-11` : `importlib.spec_from_file_location` 으로 `../plotting/paper_style.py` 를 파일 경로로 직접 로드. sys.path 방식이면 자기 자신이 잡혀 순환하기 때문 — **`plotting/` 과 이 파일의 상대 위치가 고정 전제**
- **이름 충돌은 실행 경로상 발생 불가**: shim 이 정본을 `spec_from_file_location('_plotting_paper_style', ...)` 로 **다른 모듈명**으로 올리고, `sys.path` 를 건드리는 파일은 `make_fig9_from_eval.py` 하나뿐이라 한 프로세스에 `plotting/` 과 `astar_fig9/` 가 동시에 올라가지 않음 (02 의존그래프가 astar_fig9 쪽을 드롭한 건 정적 분석 도구의 모듈명 중복 처리이고, 런타임 충돌과는 별개)
- **실제 위험 [MOVE-RISK]**: `astar_fig9/paper_style.py:10` 이 `../plotting/paper_style.py` 를 상대경로로 참조 — **두 디렉터리의 상대 위치가 고정 전제**. plotting 또는 astar_fig9 이동 시 반드시 함께 수정할 것
- `Python/_archive/deprecated_2026-09/export_onnx.py:9` : `sys.path.insert(0, 자기 디렉터리)`
- `Python/main.py:20` : `from config import *` — 이름이 어디서 왔는지 추적 불가. config에 상수 추가 시 main의 지역명과 조용히 충돌 가능
- `Python/_smoke_fullmoe.py:46-48` · `Python/_verify_ppo_mirror.py:47-49` : `importlib.reload(config)` / `reload(networks)` 로 프로세스 내 재적재. 모듈 전역을 갈아끼우므로 같은 프로세스의 다른 코드가 영향받음
- `Python/ckpt_io.py:145,183` : 함수 내부 지연 `import vessel_gym_train` (순환 회피용). 최상위로 올리면 순환이 실제로 터짐
- `Python/vessel_gym_train.py:676` : 함수 내부 `from ckpt_io import snapshot_config` — 위와 짝
- `Python/inspect_channel_freeze.py:40-42` : `if __name__ == "__main__":` 가드 안의 `for p in sys.argv[1:]: inspect(p)` — 2026-09-14 가드 추가로 import 시 실행·argv 탈취 없음(해소)
- `__file__` 기준 상대 경로로 프로젝트 밖을 가리키는 파일 : `config.py:29`(`../../..`), `analyze_circling_safety.py:32` `analyze_timeout_safety.py:18` `analyze_trajectory.py:11` `convergence_gate.py:29`(`../../../run_logs`), `_archive/deprecated_2026-09/worldmap_extract.py:16`(`../../Scenes` — 격리로 깊이가 달라져 현재 어긋남), `unity_fidelity_compare.py:144` `radar_fidelity_compare.py:168`(`../../../Build`). **`Python/` 디렉터리를 옮기면 전부 깨짐**
- `Python/plotting/write_fig_code.py:13` : 위 목록과 같은 방식이지만 **이미 어긋나 있음**
- `Python/config.py:347-348` : import만 해도 `models/<COMM_FOLDER>/VesselNavigation_<시각>/logs` 디렉터리를 생성하는 부작용. config를 읽기만 하는 스크립트도 빈 폴더를 남김

## 이름과 실제 역할이 다른 곳

- `Python/memory.py` : "Experience Replay Buffer" → 실제는 **on-policy PPO rollout 버퍼**. replay 아님(update 후 폐기, 재샘플링 없음)
- `Python/functions.py` : 범용 유틸 이름 → 실제는 `calculate_returns`(GAE) + `RunningMeanStd` 둘뿐
- `Python/obs_utils.py` : docstring "main.py와 test.py에서 공통으로 사용 — 중복 제거" → `Python/test.py` 는 삭제됨(04 §A에 `[삭제됨]`). 현재 소비자는 main.py 하나(fan-in 1)
- `Python/_archive/deprecated_2026-09/export_onnx.py` : ONNX 내보내기 → **현행 networks.py로는 실행 불가**. 없는 속성(`control_actor.conv1/conv2/fc1`) 참조 + obs 373D 가정 + COLREGS_SIZE(=0) 슬라이스. 사실상 죽은 경로 — 2026-09-14 격리
- `Python/test_ckpt_compat.py` : `test_*.py` 라 pytest가 수집하지만 `test_` 로 시작하는 함수가 0개 → 수집돼도 아무 것도 실행 안 됨. 실제로는 Windows 전용 CLI 도구
- `Python/_smoke_fullmoe.py` `_verify_ppo_mirror.py` `_verify_comm_mirror.py` : `_` 접두(비공개 모듈 관례)인데 셋 다 독립 실행 스크립트이고 `run_repro.sh preflight` 가 직접 호출하는 필수 관문
- `Python/_smoke_fullmoe.py` : "스모크 검증"이지만 C절(PPO 미러)은 2026-09-05 `action_raw` 변경 이전에 작성돼 현재 상시 FAIL — 판정 근거로 쓰이지 않음
- `Python/astar_fig9/paper_style.py` : 모듈처럼 보이나 `plotting/paper_style.py` 로 넘기는 8줄 shim
- `Python/plotting/write_fig_code.py` : "그림 코드를 쓴다"가 아니라 **99_FINAL 각 폴더에 HOWTO.md + 소스 사본을 배포**하는 문서화 도구
- `Python/plotting/build_final.py` : 그림 1장 생성기가 아니라 **99_FINAL/ 폴더 조립기**(막대그림 생성 + 파일 복사)
- `Python/plotting/make_reward_folder.py` 와 `make_ablation_rewards.py` : 이름이 달라 별개로 보이나 **둘 다 같은 `20_REWARD/` 에 씀**. 03 중복데이터상 서로 28라인 클론
- `Python/corridor_run.py` : "run"이 학습 실행이 아니라 **궤적 수집 하네스**. `torch.save`로 `.pt` 를 쓰지만 체크포인트가 아니라 궤적 텐서(`:129`)
- `Python/convergence_gate.py` : 학습 중 게이트가 아니라 **사후 metric CSV 판독기**. 학습기가 이 파일을 부르지 않음
- `Python/measure_regimes.py` ↔ `Python/compose_voyage.py` : 이름이 짝으로 안 보이나 실제로는 1→2 데이터 파이프(국면 계수 측정 → 장거리 합성). import 엣지 없음 — 사람이 숫자를 옮김
- `Python/eval_astar_global.py` : `Python/` 루트에 있으나 `astar_fig9/` 세트의 일부. 04 §B에 `Python/astar_fig9/eval_astar_global.py` 이력이 남아 있음(위치 이동). 소비자는 `astar_fig9/make_fig9_from_eval.py`
- `Python/vessel_gym_train.py` docstring : "Stage 2(학습된 통신 ON)의 배치 집계는 별도 작업" → `comm_gather`(`:261`)로 **이미 구현됨**. docstring이 낡음
- `Python/networks.py` docstring : "COLREGs one-hot도 제거됨(vessel-label leak)" → `SITUATION_INPUT` 기본 ON 이라 **one-hot 5D가 세 망 fc2에 실제로 들어감**. docstring이 낡음
- `Python/test_golden.py` : 학습기를 import하지 않아 02 의존그래프상 고립으로 보이지만, subprocess로 `vessel_gym_train.py` 를 돌리는 **가장 강한 회귀 방어선**
- `Python/inspect_channel_freeze.py` : "inspect"인데 부작용이 실행이었음 — 2026-09-14 `__main__` 가드 추가로 해소
- `Python/diag_ckpt.py` ↔ `Python/measure_regimes.py` : 이름상 무관하나 03에 10라인 클론(`diag_ckpt.py:94-103` ↔ `measure_regimes.py:77-141`) — burn-in/텔레메트리 루프가 겹침

## 어디에도 안 묶이는 파일

02 §4 고립 모듈(의존 0·피의존 0) + 확인 결과:

- `Python/compose_voyage.py` : import 0 / 피의존 0. measure_regimes 출력을 사람이 CLI 인자로 옮겨 넣는 구조라 코드 경계가 없음
- `Python/inspect_channel_freeze.py` : 1회성 진단. import 0 / 피의존 0. `__main__` 가드 추가(2026-09-14)로 스크립트로 분류됨
- `Python/_archive/deprecated_2026-09/worldmap_extract.py` : 출력 `worldmap_geometry.json`(1068줄)을 **읽는 코드 0건** — Python·C#(`*.cs`)·설정/빌드 파일 전부 확인. 생산자만 있고 소비자 없음 → 2026-09-14 json 과 함께 격리. 씬 `Assets/Scenes/WorldMap.unity` 는 빌드 등록 상태로 살아 있고, 씬 실측 좌표 기록은 `Python/corridor_run.py:3-5` docstring 에 남아 있음
- `Python/plotting/build_final.py` `make_mixed_fleet.py` `make_ablation_rewards.py` `write_fig_code.py` : 같은 `plotting/` 안인데 `paper_style` 도 안 씀(자체 matplotlib 설정). `regenerate_all.py` 의 SCRIPTS 목록에 `build_final` `make_mixed_fleet` `write_fig_code` 는 빠져 있어 실행 경로도 분리됨
- `Python/plotting/regenerate_all.py` : 자신은 아무 것도 import 안 하고 subprocess로만 8개를 엮음 — 그래프상 고립이지만 실제로는 plotting 전체의 진입점
- `Python/astar_fig9/make_fig9_paper.py` : Natural Earth 해안선 데이터 필요. `astar_fig9/data/` 의 `coast_ea.json`(2.5MB) · `route_astar.json`(368KB) 은 **실데이터가 든 정상 파일임** — 줄 수가 1인 것은 줄바꿈 없는 단일 행 JSON 이기 때문(2026-09-14 확인). 데이터 결손 아님
- `Python/astar_fig9/route_astar_real.py` : 독립 A* CLI. `eval_astar_global.py` 와 7라인 클론이 있으나 import 관계는 없음
- `Python/test_ckpt_compat.py` : 실행에 Windows 체크포인트(`VESSEL_CKPT_DIR`)가 필요해 이 저장소 안에서는 돌지 않음
- `Python/unity_fidelity_compare.py` : Unity 연결부가 TODO로 비어 있다고 자체 docstring에 명시 — 미완성 상태로 고립
- `Python/inspect_channel_freeze.py` `Python/_smoke_fullmoe.py` : 둘 다 과거 특정 사건(채널 동결·MoE 분리) 검증용이라 현재 파이프라인에 연결점 없음

---
읽은 파일 수: 58 (Python 현행 51 + Unity 배포 계약 확인용 C# 3 + docs/_raw 4)
