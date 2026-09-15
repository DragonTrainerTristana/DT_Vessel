# eval_ckpt.py · eval_mixed.py · diag_ckpt.py · measure_regimes.py · corridor_run.py · compose_voyage.py — 이동됨 (2026-09-15)

원위치 `Python/eval_ckpt.py` · `Python/eval_mixed.py` · `Python/diag_ckpt.py` ·
`Python/measure_regimes.py` · `Python/corridor_run.py` · `Python/compose_voyage.py`.
근거는 `docs/PLAN.md` "eval" 그룹(작업순서 **7단계**, git 커밋 넘버링 예정 refactor(6)).

⚠️ 세션 지시문은 이 작업을 "6단계(eval)"라 불렀으나, `docs/PLAN.md`·`docs/PROGRESS.md`
둘 다 eval 그룹을 **7단계**로 못박고 있다(6단계는 plotting 위치 확정 + astar_fig9 편입 —
이미 이 저장소에 커밋 안 된 상태로 존재하는 `Python/astar_fig9/`·`Python/plotting/write_fig_code.py`
변경분이 그것이고, 그쪽 README(`astar_fig9/README.md`)에도 "corridor_run.py는 eval 그룹,
PLAN.md 7단계 소속"이라고 이미 정확히 적혀 있음). 지시문 안의 내용(eval_ckpt.py :183,
diag_ckpt.py :269, eval_mixed→make_mixed_fleet CSV)은 전부 PLAN.md 7단계 항목과 정확히
일치하므로 **번호가 아니라 내용을 따라 7단계로 진행함.** 번호 자체는 세션 지시문의 오기로 보임 —
바로잡아 보고.

## 역할

`ckpt_io + config + vessel_gym + vessel_gym_train` 묶음을 함께 import 하는 평가·진단 소비자군.
- `eval_ckpt.py` : 프리즈 정책 완주 평가(outcome pooling)
- `eval_mixed.py` : 혼합 함대 평가 → `mixed_fleet.csv`
- `diag_ckpt.py` : 체크포인트 통신지표 진단 단일 진입점(게이트 3개)
- `measure_regimes.py` → `compose_voyage.py` : 국면별 계수 측정 → 장거리 비용 합성(사람이 숫자를 옮기는 파이프, import 엣지 없음 — 같은 방에 둠)
- `corridor_run.py` : 대만↔부산 회랑 궤적 수집

## 이동 직전/직후 grep 결과 (2026-09-15)

- `run_repro.sh` 가 이 그룹 중 `eval_ckpt.py`(구 :186 `eval_one` 함수 안)와
  `diag_ckpt.py`(구 :272 `diag` 모드 안)를 `$HERE` 상대경로로 **실제로 호출함** — 지시문의
  전제가 맞았음. 두 줄 다 `$HERE/eval/...` 로 같은 커밋 단위로 고침(아래 "셸 호출부 수정").
  `eval_mixed.py`·`measure_regimes.py`·`corridor_run.py`·`compose_voyage.py` 는 `run_repro.sh`가
  호출하지 않음(사람이 직접 실행하는 도구) — 셸 수정 대상 아님.
- `run_repro.sh` 안의 나머지 `diag_ckpt.py` 문자열(구 :246, :258)은 주석일 뿐 호출이 아니라 그대로 둠.
- 저장소 전체(`*.py *.sh *.md *.ps1 *.bat`)에서 옛 경로(`Python/eval_ckpt.py` 등, `HERE/eval_ckpt.py` 등)
  문자열 참조 — `docs/MAP.md`(계획/맵 문서, 정적 스냅샷, 이전 단계 선례상 이동 때마다 갱신 안 함)와
  `Python/_archive/deprecated_2026-09/worldmap.README.md`(이미 격리된 문서가 삭제된
  `worldmap_extract.py`의 이력을 설명하며 `corridor_run.py:3-5` 를 프로즈로 가리킴, 실행 경로 아님)
  외 0건. `WINDOWS_RUN.md` 참조 0건. `.claude/CLAUDE.md`·`.claude/agents/*.md` 는 전부 경로 없이
  파일명만 프로즈로 언급 — 이전 단계(analysis·fidelity) 선례와 동일 기준으로 갱신 안 함.
- `smoke_mac.sh` 는 이 6개를 아예 안 부름(통신 미러·골든·충실도 3개만) — PLAN.md 7단계 완료조건
  문구("eval·diag 경로는 smoke 로 안 잡힘")와 일치, 그래서 아래처럼 직접 실행으로 따로 확인함.

## sys.path 보정 (필수, 했음) — `compose_voyage.py` 제외

- `eval_ckpt.py`·`eval_mixed.py`·`diag_ckpt.py`·`measure_regimes.py`·`corridor_run.py` 5개는
  전부 `import config`/`import vessel_gym`/`from networks import ...`/`from vessel_gym_train import ...`
  (`diag_ckpt.py`는 추가로 `from ckpt_io import ...`) — `Python/` 루트 모듈. `eval/` 로 한 단계
  내려오면 `ModuleNotFoundError`. 각 파일 상단(가능한 한 이른 지점, `import` 블록 앞)에
  unity-island(1단계)·fidelity(5단계)·astar_fig9 편입(6단계)과 동일 패턴 추가:
  ```python
  import sys
  sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
  ```
  `eval_mixed.py`·`measure_regimes.py`·`corridor_run.py` 는 원래 `sys`를 import 안 했어서 같이 추가.
  `corridor_run.py`는 해당 import 들이 `main()` 안 지연 import 라 sys.path 만은 모듈 최상단(로드 시
  항상 실행)에 둠 — 지연 import 시점보다 먼저 확보돼야 하므로.
- `compose_voyage.py` 는 `config`/`vessel_gym`/`networks`/`ckpt_io`/`vessel_gym_train` 어느 것도
  import 하지 않음(measure_regimes 가 낸 JSON만 `--on`/`--off` CLI 인자로 읽는 순수 소비자,
  MAP.md "import 0 / 피의존 0"과 일치) — **수정 없음, 이동만.**
- import 테스트로 5개 모두 `ModuleNotFoundError` 없이 로드되고 `config.__file__`/`vessel_gym.__file__`/
  `networks.__file__`/`ckpt_io.__file__` 이 전부 정확히 `Python/config.py` 등 루트 파일을 가리킴을
  확인함(이동 전과 동일 모듈 객체 — 중복 로드 아님).

## 상대경로 보정 — ckpt_dir 4곳 + corridor_run.py 출력경로 (필수, 했음)

- `eval_ckpt.py`(구 :65-67) · `eval_mixed.py`(구 :31-32, 변수명 `SCRATCH`) · `measure_regimes.py`(구 :62-63) ·
  `corridor_run.py`(구 :45, :84) 전부 자기 `__file__` 기준으로 `VESSEL_CKPT_DIR` 미설정 시
  기본 체크포인트 위치를 `<파일위치>/checkpoints` 로 잡음. `Python/` 루트 기준 0단계였던 것이
  `eval/` 로 한 단계 내려오며 깊이 +1 → 전부 `..` 한 단계 보정(`os.path.normpath(os.path.join(...,
  '..'))`), 이동 전과 같은 `Python/checkpoints` 를 가리키게 함.
- **`corridor_run.py` 는 같은 `scr` 변수를 ckpt_dir 뿐 아니라 출력(`--out`, 구 :126의 `dst`)에도
  재사용함** — 기본 출력 위치도 `<파일위치>` 옆이었으므로 같은 한 번의 보정으로 출력 위치도
  이동 전과 동일한 `Python/` 를 가리키게 됨(별도 수정 지점 아님, 한 곳만 고치면 됨).
- `diag_ckpt.py` 는 자체 `__file__` 기준 경로 계산이 없음 — `--ckpt` 상대경로 해석은 전부
  `ckpt_io.restore_policy`(`Python/` 루트에 그대로 있는 core-train 그룹, 동결)가 자신의 `__file__`
  기준으로 처리하고, `run_repro.sh` diag 모드는 항상 `VESSEL_CKPT_DIR="$CK"`(절대경로)를 명시
  주입함 — **수정 불필요**, 확인만 함.
- `compose_voyage.py` 의 `os.path.exists(p)` 는 CLI `--on`/`--off` 로 받은 경로를 그대로 검사할 뿐
  `__file__` 무관 — **수정 불필요**.
- 이동 전/후 절대경로가 동일한지 python으로 직접 비교해 확인함: 5개 파일 전부
  `.../Assets/Scripts/Python/checkpoints` (와 corridor_run.py 의 출력 기본값 `.../Python/`) — 동일.

## eval_mixed.py → plotting/make_mixed_fleet.py CSV 경로 결합 (PLAN.md "확인 필요" 항목, 확인함)

- `eval_mixed.py --csv` 가 상대경로면 `os.path.join(SCRATCH, args.csv)`(SCRATCH = 위에서 보정한
  `VESSEL_CKPT_DIR` 기본값, 즉 `Python/checkpoints`)에 씀. 반면 소비자
  `plotting/make_mixed_fleet.py:23` 의 기본 읽기 위치는 `VESSEL_LOG_DIR` 기본값
  `Python/plotting/_data/mixed_fleet.csv` — **이 둘은 이동 이전부터 서로 다른 기본 디렉터리였음**
  (import 관계가 아니라 파일 전달이라 애초에 강제되는 짝이 아니었음, MAP.md 기술과 일치).
  즉 실사용은 항상 `--csv` 에 명시 경로를 주거나 `VESSEL_LOG_DIR` 를 맞춰 왔다는 뜻.
- 이번 이동으로 바뀌는 것은 **없음** — `SCRATCH`(=eval_mixed.py 쪽 기본값)를 위 보정으로
  이동 전과 같은 절대경로에 고정했으므로, 두 스크립트 간의 기존 관계(사람이 경로를 맞춰 줘야 함)가
  이동 전과 똑같이 유지됨. "확인 필요"로 남았던 이유(깊이 변화로 어긋날 수 있음)는 보정으로 해소.

## 셸 호출부 수정 (필수, 했음)

`run_repro.sh`:
- (구 :186) `"$PY" -u "$HERE/eval_ckpt.py" \` → `"$PY" -u "$HERE/eval/eval_ckpt.py" \` (`eval_one` 함수)
- (구 :272) `VESSEL_CKPT_DIR="$CK" "$PY" -u "$HERE/diag_ckpt.py" ...` → `.../eval/diag_ckpt.py` (`diag` 모드)

`smoke_mac.sh` 는 이 그룹을 안 부르므로 수정 대상 없음(위 grep 결과 참조).

## diag_ckpt.py:42 outcome 매핑 [DUP] — 손대지 않음

PLAN.md 7단계 risk 항목("이동 중 통합 유혹 금지, 게이트 판정에 영향")대로
`vessel_gym.py:170-174`·`metric_io.py:32` 와의 3중 정의를 그대로 둠. 이동만 했고 로직 무변경.

## corridor_run.py 실측 좌표 — 손대지 않음

PLAN.md 7단계 risk 항목대로 부산·대만 실측 좌표(docstring + 코드 리터럴)는 논문 수치라
손대지 않음(이동으로 내용 변경 없음, 위치만 이동).

## 완료 조건 확인

- `VESSEL_PY=python3 bash smoke_mac.sh` — 2026-09-15 실행, `EXIT_CODE=0`
  (통신 미러 ALL PASS · 골든 ALL PASS · 충실도 PASS). 단, 위에 적었듯 이 스크립트는 eval 그룹
  파일을 실행하지 않으므로 "이동한 파일이 실제로 돌아간다"는 근거는 못 됨 — 아래 직접 실행이
  그 근거임.
- **`eval_ckpt.py` 직접 실행 확인** (2026-09-15, Mac, CPU): 로컬에 남아 있던 pre-YUGIOH 체크포인트
  `Python/vessel_gym_OFF_s1.pt`(cfg_snapshot 있음, comm_range=200 학습분)를
  `VESSEL_CKPT_DIR=<Python 루트> VESSEL_COMM_RANGE=200 python3 eval/eval_ckpt.py --ckpt
  vessel_gym_OFF_s1.pt --arm OFF --envs 4 --vessels 4 --eval_decisions 200 --burnin 50 --device cpu`
  로 실행 → import·경로 해석 통과, `ckpt_io.restore_policy` 가 스냅샷을 읽어 헤더를 찍고
  평가 루프가 끝까지 돌아 결과 줄을 출력함(4×4×200 짜리 소규모 창이라 종료 에피소드가
  없는 것은 정상 — "결과 숫자가 맞다"는 주장이 아니라 "파이프라인이 새 경로에서 끝까지
  실행된다"는 확인용).
- **`diag_ckpt.py` 직접 실행 확인** (2026-09-15, 동일 체크포인트): `--envs 4 --vessels 4 --burn 50
  --collect 100 --min_sit_rate 0.0 --device cpu` 로 실행 → 게이트 2개(`sit_rate`·`config_match`) PASS,
  JSON+CSV 산출 확인. (`--expect_vcoll` 게이트는 값을 안 줘서 스킵 — 정상.) `--min_sit_rate 0.0` 은
  4×4 소규모 창이라 조우율이 0%였기 때문에 게이트를 통과시키려 준 것으로, 이 역시 실측 진단
  수치를 주장하는 게 아니라 파이프라인 확인용. 실행 후 산출 JSON/CSV 는 삭제함(검증용 임시 출력).
- 저장소 전체 옛 경로 참조 0건 (위 grep 결과 절 참조, docs/MAP.md·archive 문서의 정적/역사적 언급 제외).

## 되살릴 때

```
git mv Python/eval/{eval_ckpt.py,eval_mixed.py,diag_ckpt.py,measure_regimes.py,corridor_run.py,compose_voyage.py} Python/
```
후 `eval_ckpt.py`·`eval_mixed.py`·`diag_ckpt.py`·`measure_regimes.py`·`corridor_run.py` 의
`sys.path.insert` 2줄 삭제, `eval_ckpt.py`/`measure_regimes.py`/`corridor_run.py` 의 `scr` `..`
보정과 `eval_mixed.py` 의 `SCRATCH` `..` 보정을 되돌리고, `run_repro.sh` 구 :186/:272 의
`eval/` 접두어 2곳을 제거하고 이 README 삭제.
