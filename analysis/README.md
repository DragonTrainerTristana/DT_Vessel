# analysis — 결과 분석 묶음 (2026-09-30 – 10-13)

브랜치 `analysis` 전용 폴더. 코드는 `feat/fig1-latent` 8d6bc88 그대로이고, 이 폴더만 더함.
10-13 q_ 배치는 `feat/fig1-latent` b6a4341 코드로 돎 — 그 코드 변경은 이 브랜치에 안 넣음(코드·스펙은 `feat/fig1-latent` 에서 볼 것).
새 분석은 날짜별로 이 폴더에 계속 추가함(PDF 는 `pdf/<날짜>_…`, 자료·생성기는 `runs/` 아래 원본 구조 그대로).
학습·평가 코드는 여기서 안 바꿈. 새 실험 없음 — 이미 돈 배치의 로그와 그것을 읽어 만든 보고서임.

## 들어 있는 것

| 폴더 | 내용 |
|---|---|
| `pdf/` | 결과 보고서 PDF 16개 (09-30 – 10-13) |
| `runs/2026-09-30_t/` | 보상 v3 + 판정기 v2 배치 t_ — 로그·평가 출력·원인 분석(`rootcause_2026-09-30.md`)·보고서 조립 스크립트 |
| `runs/2026-10-01_scripted/` | 규칙 배 평가 1–4차 — 평가 출력·요약·보고서 생성기 |
| `runs/2026-10-02_imitation/` | 흉내 시작 P0 (i_) — 로그·판정 표 |
| `runs/2026-10-02_fig1/` | Fig1 배치 f_ · f2_ · n_ · q_(qa_ · qb_) — `out/`(배치 로그·관문 기록·진단 출력), `report/`(보고서 생성기·진단 스크립트) |

원본 위치 = Dropbox `Private_Paper_Project/0702_NewVessel/runs/` (폴더 구조 그대로 옮김).
체크포인트(`.pt`)는 안 넣음.

## PDF ↔ 생성기

| PDF (`pdf/`) | 생성기 (`runs/` 아래) |
|---|---|
| 2026-09-30_보상v3_t배치결과 | `2026-09-30_t/build_t_report.py` (+ `t_report_data.py` · `report_parse.py`) |
| 2026-10-01_통신_COLREGs_쉬운설명 | `2026-09-30_t/easy_report/build.py` · `make_figs.py` |
| 2026-10-01_규칙배_평가결과 | `2026-10-01_scripted/report/make_report.py` |
| 2026-10-02_규칙배2차_COLREGs규칙배_결과 | `2026-10-01_scripted/report/make_report2.py` |
| 2026-10-02_Woerner재채점_흉내P0_결과 | `2026-10-01_scripted/report/make_report3.py` |
| 2026-10-02_규칙배4차_Fig1상한_결과 | `2026-10-01_scripted/report/make_report4.py` |
| 2026-10-03_Fig1_f2_흉내P0_결과 | `2026-10-02_fig1/report/make_report_p0.py` |
| 2026-10-03_지시별_결과_종합 | `2026-10-02_fig1/report/make_report_all.py` |
| 2026-10-06_Fig1_f2_통신vsOFF_결과 | `2026-10-02_fig1/report/make_report_fig1.py` |
| 2026-10-06_f2_학습곡선 | `2026-10-02_fig1/report/diag/curve_train.py` |
| 2026-10-07_정면조우_원인진단 | `2026-10-02_fig1/report/make_report_headon.py` (측정 `report/diag/headon/`) |
| 2026-10-07_무통신_보상하락_진단 | `2026-10-02_fig1/report/make_report_offdrop.py` (측정 `report/diag/offdrop/eval_any.py`) |
| 2026-10-07_n배치_순위관문v4_결과 | `2026-10-02_fig1/report/make_report_rank_n.py` (자료 `out/_rank_v4.json` · `out/diag_rank/`) |
| 2026-10-09_n배치_trunk관문_결과 | `2026-10-02_fig1/report/make_report_trunk_n.py` (자료 `out/n_trunk_d6_s43.*` · `out/_gate.log`) |
| 2026-10-13_q배치_단계1_trunk갈림길_결과 | `2026-10-02_fig1/report/make_report_trunk_q.py` (자료 `out/qa/` · `out/qb/` · `out/diag_trunk_n/` · `out/q_stage1_notes/`) |
| 2026-10-13_q배치_qb_phase1_통신vsOFF_결과 | `2026-10-02_fig1/report/make_report_fig1_qb.py` (자료 `out/qb/` · `out/q_stage1_notes/runnote_phase1.txt`) |

## 다시 만들기

- Windows · base anaconda python · Edge headless(HTML → PDF, 글꼴 Malgun Gothic)
- 10-07 이후 생성기는 출력 폴더를 인자로 받음: `python runs/2026-10-02_fig1/report/make_report_trunk_n.py <출력 폴더>`
- 그 전 생성기는 출력 경로(Dropbox `Private_Paper_Project`)와 서버 경로(`C:\work\DT_Vessel_v3` 등)가 스크립트에 적혀 있음 — 다른 기계에서는 경로를 고쳐야 돎
- 진단 스크립트(`report/diag/`)는 체크포인트와 GPU 가 필요함. 출력은 `out/diag_*/` 에 이미 있음

## 읽을 때 주의

- 숫자는 로그 값 그대로임. 시드 1개 결과는 단독 주장 못 함(n_ · q_ 배치 = 시드 43 하나, q_ 는 offb 없음)
- `out/` 의 접두어 없는 파일(`_gate.log` · `_timeline.txt` 등)은 n_ 배치 것임. f2_ 때 값은 `out/_f2_unprefixed_2026-10-06/`, q_ 배치 값은 `out/qa/` · `out/qb/` 안에 따로 있음
- 진단(`diag_*`)은 기록용 측정임. 관문 판정은 배치가 남긴 기록(`_rank_v4.json` · `_gate.log` · `out/qa|qb/_gate.log`)이 정본. `diag_trunk_n/` 은 q_ 스펙 단계 1 의 D 항목(n_ 체크포인트 진단)임 — 분기 규칙은 T-A · T-B 관문 기록만 씀
- `out/q_stage1_notes/` = 세션이 실행 중 직접 본 기록(시작 시각·VRAM·우선순위·실행 env·2M 스냅샷 토글). 배치 스크립트가 쓴 파일 아님
- 설계·사전등록은 `docs/superpowers/specs/` 에 있음
