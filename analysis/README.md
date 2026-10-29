# analysis — 결과 분석 묶음 (2026-09-30 – 10-28)

브랜치 `analysis` 전용 폴더. 코드는 `feat/fig1-latent` 8d6bc88 그대로이고, 이 폴더만 더함.
10-13 q_ 배치는 `feat/fig1-latent` b6a4341, 10-27 R0 재평가는 071487b, 10-28 qd_ trunk 는 2cd2fbd 코드로 돎 — 그 코드 변경은 이 브랜치에 안 넣음(코드·스펙은 `feat/fig1-latent` 에서 볼 것).
새 분석은 날짜별로 이 폴더에 계속 추가함(PDF 는 `pdf/<날짜>_…`, 자료·생성기는 `runs/` 아래 원본 구조 그대로).
학습·평가 코드는 여기서 안 바꿈. 새 실험 없음 — 이미 돈 배치의 로그와 그것을 읽어 만든 보고서임.

## 들어 있는 것

| 폴더 | 내용 |
|---|---|
| `pdf/` | 결과 보고서 PDF 23개 (09-30 – 10-29) |
| `runs/2026-09-30_t/` | 보상 v3 + 판정기 v2 배치 t_ — 로그·평가 출력·원인 분석(`rootcause_2026-09-30.md`)·보고서 조립 스크립트 |
| `runs/2026-10-01_scripted/` | 규칙 배 평가 1–4차 — 평가 출력·요약·보고서 생성기 |
| `runs/2026-10-02_imitation/` | 흉내 시작 P0 (i_) — 로그·판정 표 |
| `runs/2026-10-26_a2z/` | 10-26 A–Z 진단(읽기 전용 종합) — 보고서 원본 `report.md`·렌즈 6개 검증 결과 `findings_lenses.json`·판정식 MC `mc_pass_prob.py`·PDF 생성기 |
| `runs/2026-10-26_ref_port/` | 10-26 레퍼런스 포팅 — 논문·코드 원문에서 뽑은 설정과 검증 판정 `references_verified.json`·포팅 명세와 리뷰 `port_spec.json`·보고서 원본 `report.md`(Zotero 컬렉션 MNMHENXE) |
| `runs/2026-10-28_r0_interp/` | 10-28 R0 해석 — 잡음 격차·trunk 별 평균 행동 하락 모양(직진형·우회형·불안정)·통신 vs OFF 평균 행동, 표 생성기 `r0_interpret.py`(입력 = `runs/2026-10-02_fig1/out/meaneval_*`·`diag_trunk_n`) |
| `runs/2026-10-28_early_avoid/` | 10-28 미리 피하는 보상 A–Z(설계 분석, 학습 없음) — 시간 척도·조건 5·후보 7(A–G)·연속 위험 예고 PBRS 제안 |
| `runs/2026-10-29_qd_interp/` | 10-29 qd_ 해석 — 성공 기준 3/3·학습 곡선·PPO 계측·순위 관문 v5 불통과 내용((iii) 여유 부족 · (v) v4 와 같은 값) |
| `runs/2026-10-02_fig1/` | Fig1 배치 f_ · f2_ · n_ · q_(qa_ · qb_) · R0 무학습 재평가(10-27) · qd_ trunk(10-28) — `out/`(배치 로그·관문 기록·진단 출력), `report/`(보고서 생성기·진단 스크립트) |

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
| 2026-10-26_A2Z_진단_할일 | `2026-10-26_a2z/make_pdf.sh report.md <출력.pdf>` (Mac: pandoc + Chrome headless; 자료 `findings_lenses.json` · Dropbox `runs/` 로그) |
| 2026-10-26_레퍼런스_포팅 | `2026-10-26_ref_port/make_pdf.sh report.md <출력.pdf>` (자료 `references_verified.json` · `port_spec.json`) |
| 2026-10-27_R0_무학습재평가_표본vs평균행동_결과 | `2026-10-02_fig1/report/make_report_r0.py` (자료 `out/meaneval_qb/` · `out/meaneval_qa/` · `out/diag_trunk_n/` · `out/qb/curve_qb_*.txt` · `out/r0_notes/`) |
| 2026-10-28_R0_해석_다음단계 | `2026-10-28_r0_interp/make_pdf.sh report.md <출력.pdf>` (표 = `python3 r0_interpret.py runs/2026-10-02_fig1/out`) |
| 2026-10-28_미리피하기_보상_AtoZ | `2026-10-28_early_avoid/make_pdf.sh report.md <출력.pdf>` (설계 문서, 자료 = 코드·스펙 인용) |
| 2026-10-28_qd_trunk_위험56m_떨림수정_결과 | `2026-10-02_fig1/report/make_report_qd.py <출력 폴더>` (자료 `out/qd/` · `out/meaneval_qd/` · `out/qd_notes/` · 비교 `out/n_trunk_d6_s43.log` · `out/diag_trunk_n/` · `out/qa/` · `out/qb/` · `out/meaneval_qa/` · `out/meaneval_qb/`) |
| 2026-10-29_qd_해석_순위관문 | `2026-10-29_qd_interp/make_pdf.sh report.md <출력.pdf>` (자료 = `runs/2026-10-02_fig1/out/qd/`·`meaneval_qd/`) |

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
- `out/meaneval_q*/` = `meaneval_ckpts.sh` 출력(표본 / 평균 행동 짝, 기록용 · 관문 아님). `_extra_log.txt` = 속도용 추가 프로세스(같은 명령 · 같은 파일 이름) 기록, `out/r0_notes/` = 세션 실행 기록 · 추가 프로세스 스크립트 · 드라이버 로그
- `out/qd/` = qd_ trunk 배치 출력(코드 `feat/fig1-latent` 2cd2fbd, 스펙 `2026-10-28-qd-trunk-design.md`). 순위 관문 v5 기록 = `out/qd/_rank.log` · `_rank_v5.json` · `_rankchk.log`. `out/qd_notes/` = 세션 실행 기록(`runnote.txt`) · 2M 스냅샷(`snap_2M.txt`) · 드라이버 · 평가 로그
- 10-29 병렬 3건(스펙 `2026-10-28-qd-trunk-design.md` §5): `out/qd/` 에 qd_ phase1(갈래 off · comm) 출력이 더해짐(코드 adcb7f2), `out/qe/` = qe_ trunk(예고 PBRS c = 410, `feat/early-pbrs` fa772a3), `out/qd2/` = qd2_ trunk(시드 44, adcb7f2), `out/meaneval_qe/` · `out/meaneval_qd2/` = 평가. `out/q29_notes/` = 세션 실행 기록 · 스냅샷 · 드라이버 · 평가 로그 · 16:23 GPU 드라이버 리셋으로 죽은 평가 로그(`failed_1623/`) · CPU 로 미리 계산한 Fig1 표(`_fig1_cpu_session.md`, 배치 스크립트 표 아님). `out/_qe_try1_18c45d8/` · `out/_qd2_try1_gpu2/` = 다시 띄우기 전 첫 시도(학습 · 체크포인트 없음 / 0.98M 에서 멈춤). ★qd_ 곡선 5점 · 갈래 평균 행동 평가 8개는 GPU 복구 뒤 다시 돌려야 함(이 커밋 기준 미완)
- 설계·사전등록은 `docs/superpowers/specs/` 에 있음
