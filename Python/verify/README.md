# _verify_ppo_mirror.py · _verify_comm_mirror.py · test_golden.py · test_vessel_gym_fidelity.py · test_ckpt_compat.py · golden/ — 이동됨 (2026-09-15)

원위치 `Python/_verify_ppo_mirror.py` · `Python/_verify_comm_mirror.py` · `Python/test_golden.py` ·
`Python/test_vessel_gym_fidelity.py` · `Python/test_ckpt_compat.py` · `Python/golden/`.
근거는 `docs/PLAN.md` "verify" 그룹(작업순서 **8단계** — PLAN.md·PROGRESS.md 둘 다 마지막 단계로 못박음.
이번 세션 지시문도 "verify 그룹"이라고만 했고 번호를 붙이지 않아 정정할 오기 없음).

## 역할 — `run_repro.sh preflight` 4관문 그 자체

1~7단계 전부를 이 하네스로 검증해 왔음. 이번엔 하네스 자신을 옮기고, 옮긴 뒤 **자기 자신으로
재검증**하는 순서. `run_repro.sh preflight`가 순서대로 부르는 4관문:
- `_verify_ppo_mirror.py` : Unity 경로 rollout=update 미러 (Windows 전용, Mac은 torch↔numpy 비호환)
- `_verify_comm_mirror.py` : GPU 배치 경로 rollout=update 미러
- `test_golden.py --check` : 학습기 기본값 비트동일 골든 (5케이스, `golden/*.json` 대조)
- `test_vessel_gym_fidelity.py` : vessel_gym 배치 텐서 동역학 == C# 스칼라 참조

`test_ckpt_compat.py`는 preflight가 안 부르는 별도 CLI 도구(Windows 체크포인트 전용, 아래 별도 절).

## 이동 직전 grep 결과 (2026-09-15)

- `run_repro.sh`가 이 그룹을 `$HERE` 상대경로로 부르는 곳 4곳(구 :125,126,133-134,138) —
  `_verify_ppo_mirror.py` · `_verify_comm_mirror.py` · `test_golden.py`(구 `cd "$HERE" && ... test_golden.py`)
  · `test_vessel_gym_fidelity.py`. PLAN.md가 예상한 지점과 내용은 같았으나 **줄 번호는 125/126/133-134/138로
  이미 밀려 있었음**(PLAN.md가 122/123/131/135라 적은 시점 이후 6·7단계에서 eval 그룹이 먼저 이동하며
  파일 길이가 바뀜) — 번호가 아니라 실제 파일을 읽어 확인 후 진행.
- `smoke_mac.sh`는 `_verify_comm_mirror.py` · `test_golden.py` · `test_vessel_gym_fidelity.py` 3곳을
  자체 복사본(`common_env()`)과 함께 부름(구 :79,84-85,91) — 같은 3곳 수정.
- 저장소 전체(`*.py *.sh *.md *.ps1 *.bat`)에서 옛 경로 참조 — `docs/MAP.md`·`docs/PLAN.md`(계획서,
  이전 단계 선례상 이동 때마다 갱신 안 함)·`docs/_raw/*`(리팩토링 전 정적 스냅샷, 09-14 unity-island
  README와 동일 판단으로 무해)·`.claude/CLAUDE.md`(bare 파일명 프로즈만, analysis·fidelity·eval 선례와
  동일 기준으로 갱신 안 함) 제외하고 아래 3곳에서 실제 깨지는 참조를 찾아 고침:
  - `WINDOWS_RUN.md`(2곳, `python Python\_verify_ppo_mirror.py` 실행 커맨드 + 트러블슈팅 표)
  - `Python/SIM2SIM_HANDOFF.md`(1곳, `python test_vessel_gym_fidelity.py` 실행 커맨드)
  - `.claude/agents/qa-engineer.md`(5곳)·`.claude/agents/refactorer.md`(1곳) — 표·Test Execution
    절이 `Python/test_golden.py` 등 **경로를 명시**한 실행 커맨드였음(analysis/eval 그룹 관련 언급은
    전부 경로 없는 프로즈라 이전 단계에서 안 고쳤던 것과 대비됨 — 이 그룹만 경로 명시가 있어 고침 대상).
  - `runs/ABLATION_PLAN.md` 등 `runs/*.md`, `Build/0703/HANDOFF.md`는 git root(`Assets/Scripts`) **밖**이라
    이 저장소가 추적하지 않음 — 이전 7단계 전부 이 폴더 밖을 손대지 않은 것과 동일 기준으로 안 고침
    (`Build/0703/HANDOFF.md:44`에 `python Python\_verify_ppo_mirror.py` 옛 참조가 남아 있음, 참고로 기록).

## __file__ 상대경로 결합 전수 검사 (PLAN.md "확인 필요" 항목 포함)

**가장 위험했던 지점 — `test_golden.py`의 `TRAIN`·`cwd`.** 학습기(`vessel_gym_train.py`)는
core-train 그룹이라 `Python/` 루트에 그대로 있음(동결). `test_golden.py`가 `verify/`로 내려오며
`HERE = dirname(__file__)`가 `verify/`가 되므로:
- (구 :40) `TRAIN = os.path.join(HERE, 'vessel_gym_train.py')` 는 `verify/vessel_gym_train.py`를
  가리켜 **학습기를 못 찾게 됨** — `PYROOT = os.path.dirname(HERE)`를 새로 도입해
  `TRAIN = os.path.join(PYROOT, 'vessel_gym_train.py')`로 수정.
- (구 :95, :198, :206) 세 `subprocess.run(..., cwd=HERE, ...)` 전부 `cwd=PYROOT`로 수정. 이 중
  `_config_dump`/`check_defaults_equal_yugioh`(구 :198, :206)는 `python -c "import config as c..."`를
  돌리는데, `-c` 실행은 `sys.path[0]`이 `''`(=cwd)로 잡혀 **cwd가 곧 config 탐색 경로** — `cwd=HERE`
  그대로 뒀으면 `verify/`에서 `import config`가 실패했을 지점. `cwd=PYROOT`로 고쳐 해소.
- `GOLDEN_DIR = os.path.join(HERE, 'golden')`(구 :39)는 `golden/`이 같이 이동하므로 **그대로 정확**
  (PYROOT 아님, HERE 유지) — TRAIN과 반대 방향이라 실수하기 쉬운 지점, 둘을 분리해 확인함.
- 실측: `python3 -c`로 `TRAIN`이 `.../Python/vessel_gym_train.py`(존재 확인)를, `GOLDEN_DIR`이
  `.../Python/verify/golden`(존재 확인)을 가리킴을 직접 대조 — 이동 전과 동일한 절대경로.
  이후 실제로 `test_golden.py --check`를 돌려 학습기 subprocess가 `.../Python/vessel_gym_train.py`
  경로로 기동됨을 `ps aux`로 확인함(아래 "완료 조건 확인").
- `test_golden.py` 안 docstring의 "골든 파일 Python/golden/\*.json" 문구도 `Python/verify/golden/*.json`으로
  같이 고침(코드 밖 설명이지만 같은 파일 안이라 방치하면 바로 옆 코드와 모순).

**`test_ckpt_compat.py`**: `HERE`는 golden과 함께 이동해 그대로 정확
(`GOLDEN = os.path.join(HERE, 'golden', '2026-09-10_ckpt.json')`). 학습기가 아니라 `config`·
`vessel_gym`·`ckpt_io`·`vessel_gym_train`을 **직접 import**(`fingerprint()` 함수 안, 지연 import)하므로
sys.path 보정이 새로 필요 — 아래 절.

**`os.path.exists` 가드로 조용히 건너뛰는 코드**: `test_golden.py`의 `golden_path(existing=True)`가
플랫폼별 파일(`*.{sys.platform}.json`)이 없으면 무접미사 legacy 파일로 폴백하는 `os.path.exists` 분기
(구 :128-131, 이동 후에도 동일 로직) — 이건 **플랫폼 분기**(Mac↔Windows 골든 분리, 2026-09-10 기존 설계)
목적이지 디렉터리 이동과 무관. 이동으로 새로 생긴 `os.path.exists` 가드는 없음(전수 grep 확인,
`test_ckpt_compat.py`의 `if os.path.exists(GOLDEN)` 도 마찬가지로 기존 설계 그대로 — golden이 같이
이동해 이 조건 결과 자체가 바뀌지 않음).

## sys.path 보정 (필수, 했음)

- `_verify_ppo_mirror.py` : `import config`/`import networks`가 `_reload()` 함수 안 지연 import.
  `import` 블록(`import os / import sys / import importlib / import numpy / import torch`) 사이,
  `sys` 임포트 직후에 삽입 — 함수 호출보다 먼저 모듈 로드 시 실행되면 되므로 위치 자체는
  자유롭지만 기존 그룹들과 같은 자리(초반 import 블록)에 둠.
- `_verify_comm_mirror.py` : `import config as cfg`/`vessel_gym`/`vessel_gym_train`/`networks`가
  전부 `run_case()` 함수 안 지연 import. 동일 패턴 삽입.
- `test_vessel_gym_fidelity.py` : `import vessel_gym as vg`가 **모듈 최상단**(지연 아님) — 원래
  `os`/`sys`를 import 안 했어서 같이 추가하고, `import vessel_gym` **앞**에 삽입.
- `test_ckpt_compat.py` : `config`/`vessel_gym`/`ckpt_io`/`vessel_gym_train`이 전부 `fingerprint()`
  함수 안 지연 import. 기존 `sys`/`os` import 블록 뒤, `HERE` 정의 다음 줄에 삽입.
- `test_golden.py` : project 모듈을 **직접 import 하지 않음**(subprocess로만 학습기 호출) — sys.path
  보정 불필요, 위 TRAIN/cwd 보정만으로 충분.

패턴은 analysis(4단계)·fidelity(5단계)·eval(7단계)와 동일:
```python
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
```

- 확인: `python3 -c "import sys, os; sys.path.insert(...); import config, networks, vessel_gym"`으로
  세 모듈의 `__file__`이 전부 `Python/config.py`·`Python/networks.py`·`Python/vessel_gym.py`(루트,
  `verify/` 아님)를 가리킴을 확인.
- 추가로 `_verify_ppo_mirror.py`를 **직접 실행**해 sys.path 보정이 실전에서도 통하는지 확인:
  `networks._get_others_msg`(others_msg sum 미러)가 `max|diff|=8.941e-08 PASS`로 통과 —
  트레이스백에 `.../Python/networks.py`가 정확히 잡힘(모듈을 못 찾는 게 아니라 로직까지 도달).
  이후 attention 케이스에서 `torch.as_tensor(idx_mat)` → `RuntimeError: Could not infer dtype of
  numpy.int64`로 죽는 것은 **이동과 무관한 기존 Mac torch↔numpy 비호환**(루트 CLAUDE.md §8,
  memory `mac-torch-numpy-incompat` 그대로) — 이 검증기가 Windows 전용인 이유 그 자체이며, 이동 전
  Python 루트에서도 동일하게 실패했을 조건. 회귀 아님.

## 셸 호출부 수정 (필수, 했음)

`run_repro.sh` (`preflight()` 함수):
- (구 :125) `"$HERE/_verify_ppo_mirror.py"` → `"$HERE/verify/_verify_ppo_mirror.py"`
- (구 :126) `"$HERE/_verify_comm_mirror.py"` → `"$HERE/verify/_verify_comm_mirror.py"`
- (구 :133-134) `( cd "$HERE" && env -u ... "$PY" -u test_golden.py --check )` →
  `( env -u ... "$PY" -u "$HERE/verify/test_golden.py" --check )` — `test_golden.py`가 이제
  `PYROOT` 기준으로 학습기를 스스로 찾으므로 `cd "$HERE"`가 불필요해져 제거하고 다른 3곳과
  같은 절대경로 스타일로 통일(동작은 그대로, `env -u` 3개 unset은 유지).
- (구 :138) `"$HERE/test_vessel_gym_fidelity.py"` → `"$HERE/verify/test_vessel_gym_fidelity.py"`

`smoke_mac.sh`:
- (구 :79) `"$HERE/_verify_comm_mirror.py"` → `"$HERE/verify/_verify_comm_mirror.py"`
- (구 :84-85) `( cd "$HERE" && env -u ... test_golden.py --check )` → `run_repro.sh`와 동일하게
  `cd` 제거 + `"$HERE/verify/test_golden.py"` 절대경로.
- (구 :91) `"$HERE/test_vessel_gym_fidelity.py"` → `"$HERE/verify/test_vessel_gym_fidelity.py"`
- `common_env()` 복사본(구 :37-69)은 이번 이동과 무관 — 손대지 않음. 동기화 확인:
  `diff <(sed -n '70,102p' run_repro.sh) <(sed -n '37,69p' smoke_mac.sh)` → 빈 출력(동일, 헤더의
  sed 범위 그대로 유효).

## test_ckpt_compat.py 격리 여부 — 판단 보고 (실행 안 함, 승인 대기)

PLAN.md가 이 단계 소관으로 남긴 판단. 현재 상태와 두 옵션:

- **현재**: `test_*.py` 이름 규칙을 따르지만 `test_` 로 시작하는 함수가 0개(`fingerprint`/`main`뿐) →
  pytest가 `verify/`에서 수집은 하되 실행할 테스트가 없어 그냥 통과 취급됨(실패도 아니고 검증도 아닌
  침묵). `pytest.ini`의 `testpaths = Python`은 재귀 수집이라 `verify/`로 옮겨도 이 특성 자체는
  이동 전(`Python/` 루트)과 동일하게 유지됨 — **이동이 이 문제를 새로 만들지도, 없애지도 않음.**
- **옵션 A (그대로 유지, 추천)**: verify/ 그룹의 나머지 4개와 이름·역할(체크포인트 로드·추론 골든)이
  동질적이라 한 디렉터리에 두는 것 자체는 자연스러움. pytest 수집-무실행은 이동 전부터 있던
  독립적인 결함이라 이번 이동 스코프에서 고칠 필요는 없음.
- **옵션 B (이름 변경)**: `ckpt_compat_check.py` 등으로 개명해 pytest가 애초에 수집 대상으로 안 보게
  함. `docs/PLAN.md` 8단계 risk 항목이 이미 지적한 대로 **이름 변경은 실행 지시가 바뀌는
  일**(Windows에서 `python test_ckpt_compat.py --check ckpt들...`로 부르던 사람이 있으면 깨짐) —
  사용자 확인 없이 하지 않음.
- 이번 세션은 **이동만 하고 이름은 그대로 유지**(옵션 A 방향으로 처리, B는 승인 대기 상태로 보류).
  Windows 체크포인트가 이 저장소에 없어 실행 자체를 검증할 수 없다는 점은 PROGRESS.md 완료조건
  문구대로 **미검증으로 명시**.
- sys.path 보정만 적용(위 절) — 로직·CLI 인터페이스 변경 없음.

## 완료 조건 확인

- **`VESSEL_PY=python3 bash smoke_mac.sh`** (2026-09-15, Mac) — **`EXIT_CODE=0`.** 전 구간 새 `verify/`
  경로로 실행됨(`ps aux`로 `test_golden.py --check`가 `.../Python/verify/test_golden.py`로,
  그 안 학습기 subprocess가 `.../Python/vessel_gym_train.py --arm ON --steps 16384 ...`로 —
  즉 **PYROOT 경로에서** — 기동됨을 실측 확인, TRAIN 경로 보정이 실전에서 작동함).
  ```
  [1/3] 통신 미러 검증        →  통신 미러 ALL PASS   (18케이스: sum/mean/attention/pos_ground off/
                                shared enc all·actor/MoE 공유·비공유/YUGIOH 구조 4종 등, `_verify_comm.txt`)
  [2/3] 골든 비트동일 검사    →  골든 ALL PASS
    PASS  config 기본값 == YUGIOH
    PASS  default_ON               state_dict 319텐서 · adam · value_norm · curve 전부 비트동일
    PASS  default_OFF              state_dict 319텐서 · adam · value_norm · curve 전부 비트동일
    PASS  batch_2026_09_04_ON      state_dict 289텐서 · adam · value_norm · curve 전부 비트동일
    PASS  batch_shared_all_ON      state_dict 289텐서 · adam · value_norm · curve 전부 비트동일
    PASS  batch_shared_actor_ON    state_dict 289텐서 · adam · value_norm · curve 전부 비트동일
    VERDICT: ALL PASS
  [3/3] vessel_gym 충실도 검사 →  충실도 PASS (6종: 배칭 정확성·물리 상식·레이더 기하·상황판정·
                                obs/step·처리량 34× — `_fidelity.txt`)
  ```
  **골든 5케이스가 이동 전(Python/golden/) 골든 JSON과 비트동일 — "경로가 바뀌어도 결과는 같아야
  한다"는 완료 조건을 실측으로 만족.**
- **`_verify_ppo_mirror.py` 직접 실행**(Mac, smoke_mac.sh가 원래 제외하는 대상): sys.path 보정으로
  `config`/`networks`를 정확한 루트 경로에서 찾아 첫 검증(others_msg sum 미러, `max|diff|=8.941e-08
  PASS`)까지 통과 — 이동으로 인한 회귀 없음을 코드 실행으로 직접 확인. attention 케이스에서
  Mac 전용 `torch.as_tensor(numpy.int64)` 비호환으로 죽는 것은 이동 전부터 있던 플랫폼 제약(위
  "__file__ 상대경로" 절 참조) — Windows 전용이라 ALL PASS 판정 자체는 Windows에서 함(루트 규약
  §8, PLAN.md 판정 규칙과 동일).
- 골든 JSON이 이동 전과 같은 파일을 읽는지: `golden_path()`가 `GOLDEN_DIR = os.path.join(HERE,
  'golden')`(HERE=`verify/`)로 계산 → `verify/golden/2026-09-10_default_ON.json` 등 기존 파일과
  100% 동일 경로·동일 파일(내용 변경 없이 `git mv`로만 이동) — 파일 자체가 안 바뀌었으므로 비트동일은
  구조적으로 보장됨. `test_golden.py --check`가 실제로 그 경로를 읽어 diff 없이 통과하는지는 위 실행
  결과로 확인.
- 저장소 전체 옛 경로 참조 0건 — 위 "이동 직전 grep 결과" 절 + 아래 재확인 grep으로 검증
  (git 추적 대상 한정, docs/MAP.md·docs/PLAN.md·docs/_raw·`.claude/CLAUDE.md` bare 프로즈·git-외부
  runs/·Build/ 는 선례대로 제외).

## 되살릴 때

```
git mv Python/verify/{_verify_ppo_mirror.py,_verify_comm_mirror.py,test_golden.py,test_vessel_gym_fidelity.py,test_ckpt_compat.py} Python/
git mv Python/verify/golden Python/golden
```
후:
- `_verify_ppo_mirror.py`·`_verify_comm_mirror.py`·`test_vessel_gym_fidelity.py`·`test_ckpt_compat.py`의
  `sys.path.insert` 삽입 줄 제거.
- `test_golden.py`의 `PYROOT = os.path.dirname(HERE)` 줄 제거, `TRAIN`/3곳의 `cwd=PYROOT`를
  `cwd=HERE`(=`TRAIN = os.path.join(HERE, 'vessel_gym_train.py')`)로 되돌리고 docstring의
  `Python/verify/golden` 문구를 `Python/golden`으로 되돌림.
- `run_repro.sh`·`smoke_mac.sh`의 `verify/` 접두어 4+3곳 제거(구 스타일로 되돌리려면 `test_golden.py`
  호출부에 `cd "$HERE" &&`를 다시 붙여도 되고, 절대경로 그대로 둬도 무방 — 둘 다 동작).
- `WINDOWS_RUN.md`·`Python/SIM2SIM_HANDOFF.md`·`.claude/agents/qa-engineer.md`·
  `.claude/agents/refactorer.md`의 `verify/` 접두어 제거.
- 이 README 삭제.
