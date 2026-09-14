# main.py · frame_stack.py · memory.py · functions.py · obs_utils.py — 격리됨 (2026-09-14)

원위치 `Python/main.py` · `Python/frame_stack.py` · `Python/memory.py` · `Python/functions.py` · `Python/obs_utils.py`.
근거는 `docs/PLAN.md` "unity-island" 그룹(1단계).

## 역할 — Unity ground-truth 판정 경로

`main.py` + ML-Agents(C# `Agent/` `Navigation/` `Management/`)로 도는 학습·평가 경로.
`.claude/CLAUDE.md` §1 기준 **sim2sim 판정관** — GPU 배치(`vessel_gym.py`+`vessel_gym_train.py`)와는
`config.py`+`networks.py` 만 공유하고 나머지는 완전히 별개 구현.

## 격리 이유 — 섬 안에서 fan-in 닫힘

- `main.py` 를 import 하는 코드 0건(엔트리포인트).
- `frame_stack.py` · `memory.py` · `obs_utils.py` fan-in 1 = `main.py` 뿐.
- `functions.py` fan-in 2 = `main.py` · `memory.py` (섬 내부끼리만).
- 섬 밖 소비자 0건. 격리해도 끊기는 엣지 없음.

## 격리 직전 grep 결과 (2026-09-14)

- `run_repro.sh` · `smoke_mac.sh` — 이 5개 파일을 `$HERE` 상대경로로 호출하는 곳 0건.
- 살아있는 Python 코드(core-train·eval·verify·fidelity 등) — 실제 `import`/`from … import` 0건.
  `ckpt_io.py` · `config.py` · `networks.py` · `vessel_gym_train.py` · `_verify_ppo_mirror.py` ·
  `unity_fidelity_compare.py` 에 있는 `main.py`/`memory.py` 언급은 전부 **주석**(이식 출처 설명)뿐이라
  격리와 무관하게 그대로 유효함.
- C# `Agent/VesselAgent.cs:329` — `frame_stack.py` 언급 1건이지만 **주석**("프레임 스태킹은
  Python(frame_stack.py)이 담당"). 코드상 의존 아님, 그대로 정확한 설명으로 남음.
- **범위 밖(이번 세션에서 안 고침)** — 레거시 실행 스크립트 6개가 여전히 `python main.py` 를
  옛 위치(`Python/` 루트) 기준으로 호출함: `install_mlagents.bat` · `launch_sweep_fromscratch.sh` ·
  `launch_latentNEW.sh` · `run_eval_gt.ps1` · `run_experiment.ps1` · `launch_editor_dim6.ps1`.
  `docs/PLAN.md` "MAP.md 범위 밖" 절이 이 스크립트들을 8단계 어디에도 포함하지 않고
  "처분은 별도로 물어볼 것"이라 명시함 — 이번 1단계 범위 밖이라 손대지 않음. 되살릴 때 또는
  이 스크립트들을 실제로 쓸 때는 `cd Python/_archive/unity_path_2026-09 && python main.py` 로
  바꾸거나 셸에서 새 경로로 옮겨줘야 함.
- `docs/_raw/03_duplicate_code.json` — 리팩토링 전 분석 스냅샷(생성 시점 기록). 어떤 코드도
  런타임에 읽지 않음 — 옛 경로가 남아 있어도 무해.
- `Python/_archive/2026-09/test.py` · `_smoke_telemetry.py` — 이전(09-10) 리팩토링 때 이미 격리된
  죽은 코드. `test.py` 가 `frame_stack`/`obs_utils` 를 여전히 `import` 하지만, 애초에 중첩 폴더에서
  sys.path 보정 없이 실행 불가능한 상태로 격리돼 있었음(격리 이전부터 죽어 있었음) — 이번 이동으로
  새로 깨진 것 아님.

## sys.path 보정 (PLAN.md "이동 단계마다 sys.path 확보 필수")

- `main.py` 가 `from config import *` / `from networks import CNNPolicy` 로 루트 모듈을 씀.
  `__init__.py` 가 없어 서브디렉터리로 내려오면 이 import 가 그냥 깨짐.
- `main.py` 상단(CUDA allocator 설정 직후, 다른 import 이전)에 아래 두 줄을 추가함:
  ```python
  import sys
  sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
  ```
  이걸로 `Python/` 이 다시 sys.path 에 들어와 `config`/`networks` import 가 예전과 동일하게 해석됨.
- `obs_utils.py` 도 `from config import STATE_SIZE, COMM_RANGE, MAX_COMM_PARTNERS` 를 쓰지만,
  `main.py` 를 통해서만 import 되므로(fan-in 1) `main.py` 의 sys.path 보정을 그대로 물려받음 —
  별도 수정 불필요.
- `frame_stack.py` · `memory.py` · `functions.py` 는 `numpy`/표준 라이브러리만 써서 보정 불필요.

## Unity 체크포인트 호환은 유지됨

- `ckpt_io.snapshot_config` 는 `Python/` 루트에 그대로 남아 있어 Unity `.pth`/`_reward_rms.npz`
  스냅샷 키 정의는 보존됨. 이 격리로 사라지는 건 **생산자**(main.py 학습 루프)뿐, 키 스키마 아님.
- `obs_utils.py` 는 obs 369D 정의 [DUP] 6곳 중 1곳 — 나머지 5곳(`VesselAgent.cs` · `vessel_gym.py` ·
  `vessel_gym_train.py` · `config.py` · `networks.py`)은 그대로 살아 있어 계약 자체는 안 바뀜.

## 되살릴 때

1. `git mv Python/_archive/unity_path_2026-09/{main.py,frame_stack.py,memory.py,functions.py,obs_utils.py} Python/`
2. `main.py` 상단에 추가한 sys.path 두 줄 제거(원래 위치로 돌아가면 불필요 — 남겨둬도 자기 자신을
   가리키게 돼 무해하지만 지우는 게 깔끔함).
3. 위 "범위 밖" 레거시 스크립트 6개는 애초에 안 고쳤으므로 그대로 다시 맞음.
