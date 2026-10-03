# runs_fig1 — Fig1 배치 f_ 실행 묶음 (2026-10-03)

- 목적: Dropbox 없이 GitHub 클론만으로 Fig1 배치(f_)를 아무 Windows 기계에서 돌림
- 스펙: `docs/superpowers/specs/2026-10-02-fig1-latent-design.md`
- 실행(Git Bash, 클론 루트, 브랜치 `feat/fig1-latent`, 얕은 클론 금지):
  - `VESSEL_F_AUTO1=1 bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh phase0`
  - python 경로가 `$HOME/anaconda3/envs/mltest/python.exe` 가 아니면 `VESSEL_PY=<경로>` 를 앞에 붙임
- 결과: `Python/_repro_out_f/` (git 밖). `$HOME/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-02_fig1/` 가 있으면 표·로그 사본을 그 `out/` 에도 씀
- 폴더 구성 = Dropbox `runs/` 와 같은 상대 경로(스크립트끼리 `../` 로 서로 import)
  - `2026-10-02_fig1/` 실행 스크립트 `_run_f.sh` · 표 `summarize_fig1.py`
  - `2026-10-02_imitation/summarize_p0.py` P0 표
  - `2026-10-01_scripted/` 평가 래퍼 `eval_scripted.py`(Woerner·타 지표 줄) · `colregs_*.py` · `summarize.py` · `out/` 참고 결과(규칙 배 vo56·goal, 학습 t_off 재채점)
  - `2026-09-30_t/report_parse.py` 평가 로그 파서 · `out/` 참고 결과(t_off 주 평가)
- 파일 내용은 2026-10-02 Dropbox 사본과 같음. 이 실행의 정본은 이 폴더
