# _archive/2026-09 — 2026-09-10 리팩토링 때 현역에서 뺀 스크립트

전부 **참조 0건**(다른 스크립트·셸·문서가 부르지 않음)이고 현행 경로가 대체함. 지우지 않고 여기 둔 이유는
(a) 그림·표의 출처 증빙, (b) 회귀 fix 의 증거, (c) gitignore 돼 있어 지우면 복구 불가였음(plot_*/gen_*/mock_*).
**여기 있는 건 돌리지 말 것.** 설정이 현행과 안 맞고, 일부는 잘못된 조건으로 측정한 것임(아래 "인용 금지" 표시).

현행 대체물: 체크포인트 진단 → `diag_ckpt.py` / 체크포인트 복원 → `ckpt_io.py` / metric CSV → `metric_io.py` /
논문 그림 → `plotting/` / 평가 → `eval_ckpt.py`·`eval_mixed.py`.

## 왜 뺐나 — 파일별

### 잘못된 조건으로 측정 — 인용 금지
| 파일 | 사유 |
|---|---|
| `_diag_msg_channel.py` | env 12개 강제 대입(attention OFF·comm 200), crossing=0 리터럴. 실제 런은 attention ON·300·crossing 2. `runs/m2_ablation/diag/README_진단.md §1` 표의 출처 → 그 표 전체 재측정 대상. 지표는 `comm_telemetry` 로 흡수됨 |
| `_diag_msg_ablate.py` | 위와 같은 env 블록 복붙. zero/shuffle 은 `comm_telemetry` 의 act_zero/act_shuf 와 정의가 다름(집계 후 조작) |
| `diag_timeout.py` | 타인 머신 절대경로 박힘, `--arm` choices 없음 → `vessel_gym_train.py:89` 런타임 가드가 생긴 원인. `runs/diag_timeout_why.py` 가 후속 |

### 일회용 — 그날 데이터·그 체크포인트 전용
| 파일 | 사유 |
|---|---|
| `_probe_0602_gatefix.py`, `_rc2_probe_ckpt.py` | 체크포인트 경로 12개·6개를 본문에 박음. 기능은 `inspect_channel_freeze.py` 가 일반화 |
| `_midcheck_0613.py` | 스스로 "중간점검(1회성)·채택 금지" |
| `_convergence_point_0613.py` | 폴더 패턴 `20260612_1826*` 박음 |
| `_analyze_h2.py`, `_plot_h2_reward.py` | 2026-06-24 H2 MSG_DIM 스윕 전용(17열 인덱스 박음). H2 그림 출처 |
| `_mt10_analysis.py`, `_losL1_analysis.py`, `_smooth_mt10.py`, `_smooth_colregs_0613.py`, `_threat_mt10.py` | 같은 로직 × glob 5벌(헤더 복붙). 2026-06 스윕 전용. `_threat_subset_0613.py` 는 `_threat_mt10.py` 와 md5 동일이라 삭제함 |
| `eval_slew_exp.py` | 9열 구 METRIC_LOG 포맷 가정 |

### 회귀 fix 의 증거 (그 fix 가 되돌려지지 않았다는 증빙)
| 파일 | 사유 |
|---|---|
| `_smoke_unfreeze.py` | 2026-06-12 채널 해동(양단 gradient) |
| `_smoke_c5c.py` | 2026-06-22 C5c 수신측 decode |
| `_smoke_telemetry.py` | 2026-06-30 main.py(Unity) grad-norm/gate 텔레메트리 |
| `_density_regime_check.py` | 학습 전 기하 Monte-Carlo (정책·env 안 씀). ring 레버 결정 근거 |
| `_smoke_moe_fc2.py` | 스스로 `[DEPRECATED 2026-06-26]` 선언한 7줄 shim → 삭제함 |

### Unity 시대(2026-02~05) 분석·플롯 — `plotting/` 로 세대교체
| 파일 | 사유 |
|---|---|
| `test.py` (2357줄) | Unity `main.py` 테스터. 구 371D obs 계약, `policy.forward` 4번째 인자 시그니처 불일치. Unity 재측정 시 유일한 경로일 수 있어 보관 |
| `analyze_longhaul.py`, `analyze_existing_time_fuel.py`, `analyze_efficiency.py`, `analyze_mixed.py` | 2026-03~04 분석. `analyze_existing_time_fuel.py` 는 스스로 "임시 분석" |
| `plot_*.py` 32개 | 전부 gitignore(`Python/plot_*.py`)·참조 0. `plotting/`(2026-08, paper_style 공통화)이 대체. `plot_5metrics_commcompare.py:4` 가 "옛 gen_5graphs 계열은 'COMM ON≥OFF 보정'·환경 보간이 박혀 있어 사용 금지" 라 명시 — 방법론 전환 기록으로 가치 |
| `gen_5graphs_final.py`, `gen_5graphs_v2.py`, `gen_mixed_graphs.py` | 위 "사용 금지" 대상. gitignore 라 보관 |
| `mock_longhaul_estimate.py` | 헤더 `*** MOCK / PRIOR ESTIMATE *** 실제 실험 결과 아님`. `eval_astar_global.py` 가 대체. gitignore 라 보관 |
| `compute_metrics.py` | "Narrow Channel: 보간 추정값(CSV 없음)" 자기 명시 = 합성 수치 → 삭제함(추적 중이라 git 에 있음) |

### astar_fig9/ stale 포크
`astar_fig9/{corridor_run,eval_astar_global,worldmap_extract}.py` 는 `Python/` 상위판의 사본이었고 2026-08-31 `msg_ln` fix 가 빠진 구버전 → 삭제. `astar_fig9/paper_style.py` 는 `plotting/paper_style.py` 를 import 하는 shim 으로 교체. `make_fig9_*.py` 는 산출물(json/csv)만 소비하므로 영향 없음.

## 삭제한 것 (git 스냅샷 `4bcfa4b` 에서 복구 가능)
`_smoke_moe_fc2.py`, `_threat_subset_0613.py`(md5 중복), `compute_metrics.py`, `astar_fig9/{corridor_run,eval_astar_global,worldmap_extract}.py`, 디스크의 `*.bak_*` 12개.
