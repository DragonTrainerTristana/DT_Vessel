# radar_fidelity_compare.py · unity_fidelity_compare.py — 이동됨 (2026-09-14)

원위치 `Python/radar_fidelity_compare.py` · `Python/unity_fidelity_compare.py`.
근거는 `docs/PLAN.md` "fidelity" 그룹(작업순서 5단계, git 커밋 넘버링 refactor(4)).

## 역할

Unity ↔ vessel_gym 대조 도구 2종.
- `radar_fidelity_compare.py` — obs[0:360] 레이더 3종(장애물원·타선OBB·벽) 분리 대조. 2026-07-04 Build 0703 실측으로 검증 이력 있음(헤더 참조).
- `unity_fidelity_compare.py` — 고정 action 시퀀스로 동역학 궤적(pos/heading/speed/rudder) 대조. vessel_gym 쪽(replay)은 완성, Unity 연결부는 TODO 미완성 상태(자체 docstring 명시) — 이 Mac 세션에서도 검증 불가.

## 이동 이유

Unity↔gym 대조라는 같은 목적, `../../../Build/0703/Vessel_MLAgent.exe` 상대경로 [DUP]를 공유(PLAN.md). 둘 다 `mlagents_envs.UnityEnvironment`를 직접 연결하며 `main.py`(unity-island)를 경유하지 않는 독립 도구군.

## 이동 직전/직후 grep 결과 (2026-09-14)

- **중요 정정**: 이번 세션 지시에는 "이 그룹은 검증 하네스에 속하고 `test_vessel_gym_fidelity.py`를 `run_repro.sh` preflight와 `smoke_mac.sh`가 호출한다"는 전제가 있었으나, 실제로 grep한 결과 **`run_repro.sh`·`smoke_mac.sh`가 `radar_fidelity_compare.py`/`unity_fidelity_compare.py`를 호출하는 곳은 0건**이다. `smoke_mac.sh` [3/3]이 실제로 부르는 `test_vessel_gym_fidelity.py`는 이름이 비슷할 뿐 PLAN.md상 **verify 그룹(8단계, 이번 세션 범위 밖)의 완전히 다른 파일**이며, 이 두 파일과는 import도 실행 관계도 0건(양방향 grep 확인). 즉 이 fidelity 그룹의 이동은 검증 하네스에 구조적으로 영향을 주지 않는다 — shell 호출부 수정 대상 자체가 없었음.
- 저장소 전체(`*.py *.sh *.md *.ps1 *.bat`)에서 `Python/radar_fidelity_compare.py` / `Python/unity_fidelity_compare.py` 문자열 경로 참조 — `docs/MAP.md`·`docs/PLAN.md`(계획/맵 문서, 정적 스냅샷, 이전 단계 선례상 이동 때마다 갱신하지 않음) 외 0건. 레거시 `.ps1`/`.bat`/`.sh` 실행 스크립트 참조 0건. `WINDOWS_RUN.md` 참조 0건.
- `Python/SIM2SIM_HANDOFF.md`(line 41, 98, 100)에 두 파일명이 경로 없이 프로즈로 언급됨 — 디렉터리 무관한 스크립트 이름 서술이라 이동과 무관하게 유효, 수정 불필요(unity-island README의 동일 판단과 같은 기준).

## sys.path 보정 (필수, 했음)

- 둘 다 `import vessel_gym as vg` — `Python/` 루트 모듈. `fidelity/`로 한 단계 내려오면 `ModuleNotFoundError`.
- 각 파일 상단(`import os` 직후, 다른 import 이전)에 unity-island(1단계)와 동일 패턴 추가:
  ```python
  import sys
  sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
  ```
  (unity-island는 `_archive/unity_path_2026-09/`로 2단계 깊이라 `"..", ".."`, fidelity는 `fidelity/`로 1단계 깊이라 `".."` 한 번만.)
- `python3 -c "import radar_fidelity_compare, unity_fidelity_compare"`로 `vg.__file__`이 정확히 `Python/vessel_gym.py`를 가리킴을 확인함.

## 상대경로 보정 — Build 경로 [DUP] 2곳 (필수, 했음)

- `radar_fidelity_compare.py`(구 :168) · `unity_fidelity_compare.py`(구 :144)의 `_here/../../../Build/0703/Vessel_MLAgent.exe`가 `Python/` 루트 기준 3단계 `..`였음. `fidelity/`로 한 단계 더 내려왔으므로 4단계 `..`로 보정.
- 이동 전/후 절대경로가 동일함을 python으로 직접 비교해 확인: 양쪽 다 `.../0702_NewVessel/Build/0703/Vessel_MLAgent.exe` — 동일.
- `radar_fidelity_compare.py`의 `_scene_json` Windows temp 절대경로 fallback(구 :37-38)은 `_here`와 무관한 하드코딩값이라 손대지 않음 — 이동과 무관하게 이미 깨져 있던 값(이 Mac에 해당 경로 없음). PLAN.md 5단계 risk에 기록된 대로 고치려면 데이터 소스 결정이 필요해 사용자 확인 없이 건드리지 않음.
- `unity_fidelity_compare.py` 헤더 docstring의 수동 실행 예시(`set VESSEL_ENV_PATH=..\..\Build\...`, 2단계)는 애초에 실제 기본값(3단계, 지금은 4단계)과 안 맞던 이동과 무관한 기존 표기 불일치 — 이번 이동으로 새로 생긴 문제 아니라 범위 밖으로 두고 손대지 않음.

## unity_fidelity_compare.py 격리 여부 판단 (PLAN.md "확인 필요" — 판단만, 실행 안 함)

**판단: unity-island(`_archive/unity_path_2026-09/`)로 같이 보내지 않고 fidelity/에 잔류시키는 것이 맞다.**

근거:
- import 그래프상 unity-island 5개 파일(main.py·frame_stack.py·memory.py·functions.py·obs_utils.py)과 연결이 양방향 0건. `unity_fidelity_compare.py`는 `mlagents_envs.UnityEnvironment`를 직접 연결하지 `main.py`를 경유하지 않는다.
- "Unity에 연결한다"는 특성은 `radar_fidelity_compare.py`도 동일하게 가짐(`mlagents_envs.UnityEnvironment` 직접 사용) — fidelity 그룹 공통 특성이지 `unity_fidelity_compare.py`만의 특성이 아니다. 이 기준으로 격리한다면 `radar_fidelity_compare.py`도 같이 가야 하는데, PLAN.md는 그쪽을 "확인 필요"로 잡지 않았다 — 비대칭적 근거.
- unity-island의 소속 기준은 "fan-in이 섬 안에서 닫힘"(main.py가 유일 소비자, 섬 밖 소비자 0)이었다. `unity_fidelity_compare.py`는 애초에 main.py의 소비자도 피소비자도 아니라 이 기준 자체가 적용되지 않는다.
- TODO 미완성은 완성도 문제이지 소속(어느 디렉터리에 있어야 하는가) 문제가 아니다.

## test_ckpt_compat.py — 이번 세션 범위 밖 (판단 보고 생략)

PLAN.md verify 그룹(8단계) 소속. 이번 세션은 fidelity 그룹(5단계)만 다루고, 세션 제약상 "eval·verify·plotting/astar는 범위 밖"이라 손대지 않음 — 판단은 verify 단계 세션 소관.

## 완료 조건 확인

- Import 테스트로 두 파일 모두 `ModuleNotFoundError` 없이 로드되고 `vg.__file__`이 올바른 `Python/vessel_gym.py`를 가리킴을 확인.
- 두 파일이 계산하는 Build 경로 문자열이 이동 전과 같은 절대경로를 가리킴(위 확인). Build 실물이 이 Mac에 없어 실제 Unity 연결 대조는 여전히 Windows 필요 — 이동과 무관하게 원래도 그랬음.
- `VESSEL_PY=python3 bash smoke_mac.sh` — 2026-09-14 실행, `EXIT_CODE=0`(통신 미러 ALL PASS · 골든 ALL PASS · 충실도 PASS). 단, 위 "이동 직전/직후 grep 결과"에 적었듯 이 스크립트는 fidelity 그룹 파일을 애초에 실행하지 않으므로, 통과해도 "이동한 파일이 실제로 실행돼 검증됐다"는 근거는 아니다 — import 테스트 + Build 경로 문자열 대조가 이 그룹의 실질적 완료 조건이다.
- 저장소 전체 옛 경로(`Python/radar_fidelity_compare.py`·`Python/unity_fidelity_compare.py`) 참조 0건(docs/MAP.md·docs/PLAN.md의 계획서상 기록 제외 — 선례상 이동 때마다 갱신하지 않음).

## 되살릴 때

```
git mv Python/fidelity/{radar_fidelity_compare.py,unity_fidelity_compare.py} Python/
```
후 두 파일의 `sys.path.insert` 2줄 삭제, Build 경로 `..` 4단계를 3단계로 되돌리고 이 README 삭제.
