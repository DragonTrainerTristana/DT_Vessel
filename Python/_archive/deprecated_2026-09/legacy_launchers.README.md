# 레거시 실행 스크립트 6개 — 격리됨 (2026-09-14)

원위치 `Python/`. 사용자 지시로 2단계(deprecated 격리) 세션에 추가 편입 — `docs/PLAN.md` 의
"MAP.md 범위 밖" 절에 이름만 올라 있고 8단계 어디에도 배정 안 돼 있던 것을 이번에 처분함.

대상: `install_mlagents.bat` · `launch_sweep_fromscratch.sh` · `launch_latentNEW.sh` ·
`run_eval_gt.ps1` · `run_experiment.ps1` · `launch_editor_dim6.ps1`.

## 격리 이유 — 전부 옛 위치의 `python main.py` 를 불러 현재 동작하지 않음

- `launch_editor_dim6.ps1` 마지막 줄이 `python main.py`, 나머지도 `main.py` 를 CWD=`Python/`
  기준으로 직접 실행하거나(`install_mlagents.bat` 안내 문구) 그 실행을 감싼 것.
- `main.py` 는 이미 이전 세션(1단계, 커밋 `dcaef15`)에서 `Python/_archive/unity_path_2026-09/`
  로 격리됨. 즉 이 6개는 **1단계 시점부터 이미 죽어 있었음** — 이번 이동으로 새로 깨진 것 아님.
- 정본 실행 경로는 `run_repro.sh`(`.claude/CLAUDE.md` §1) 하나뿐. 이 6개는 전부 옛 sweep/단발
  실험 시대의 개별 런처로, run_repro.sh 이전 워크플로.

## 격리 직전 grep 결과 (2026-09-14)

- `run_repro.sh` · `smoke_mac.sh` — 이 6개를 호출하는 곳 0건.
- 살아있는 코드에서 이 6개를 실행/참조하는 곳 0건. 문서(`docs/PLAN.md` `docs/MAP.md` `docs/_raw/*`)
  언급만 있고, 전부 격리 대상 존재를 기록한 정적 텍스트임(런타임 무관).

## ⚠️ 범위 밖으로 확인됨 — `run_sweep_*.ps1` 7개가 `run_experiment.ps1` 을 여전히 참조함

`run_sweep_attention.ps1` · `run_sweep_commgate.ps1` · `run_sweep_moe.ps1` · `run_sweep.ps1` ·
`run_sweep_rushfix.ps1` · `run_sweep_c5c.ps1` · `run_sweep_msgdim.ps1` (전부 `Python/` 에 그대로
남아 있음)이 `"$PSScriptRoot\run_experiment.ps1"` 형태로 같은 폴더의 `run_experiment.ps1` 을
자식 프로세스로 spawn함. 이번 이동으로 그 참조가 끊김.

**단, 이 7개도 이미 죽어 있던 코드임** — `run_experiment.ps1` 자신이 `python main.py` 를 부르고
`main.py` 는 1단계에서 이미 격리됐으므로, `run_experiment.ps1` 이 `Python/` 루트에 있었어도
어차피 동작 안 했음. 이번 이동이 "동작하던 것을 깬" 게 아니라 "이미 죽은 참조의 위치가 바뀐" 것.
이 7개는 `docs/PLAN.md` "MAP.md 범위 밖" 절의 `run_sweep*.ps1`(7)·`run_aggregation_diagnostics.sh`
와 같은 처분 대기 묶음 — **이번 세션 범위 밖, 처분은 별도 승인 후.**

## 되살릴 때

1. `git mv Python/_archive/deprecated_2026-09/{install_mlagents.bat,launch_sweep_fromscratch.sh,launch_latentNEW.sh,run_eval_gt.ps1,run_experiment.ps1,launch_editor_dim6.ps1} Python/`
2. 이것만으로는 여전히 안 돎 — `main.py` 가 `_archive/unity_path_2026-09/` 에 있는 한 `python main.py`
   호출이 전부 실패함. 같이 되살리려면 `main.py` 등 unity-island 5개 파일도 원위치해야 함
   (`_archive/unity_path_2026-09/README.md` "되살릴 때" 참고).
3. 되살린 뒤에도 `run_sweep_*.ps1` 7개가 `run_experiment.ps1` 을 같은 폴더 기준으로 찾으므로
   경로 문제는 자동 해소됨(전부 `Python/` 루트로 복귀하는 경우).
