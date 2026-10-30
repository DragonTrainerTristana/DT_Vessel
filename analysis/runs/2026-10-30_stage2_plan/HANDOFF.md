# 인계 — 2단계 실행 중 (2026-10-30 저녁, 다음 세션용)

새 세션은 이 순서로 읽음: 루트 `CLAUDE.md` §0 마지막 항목들 → 이 파일 → 스펙 `docs/superpowers/specs/2026-10-30-stage2-far-colregs-design.md`(feat/fig1-latent c034422, §3 = 결과 전 고정 기준)

## 1. 지금 상태

- 저자 승인: 묘책(이 폴더 `report.md`) 그대로 진행 · MoE 구조 유지 · MoE 종류 비교(단일망·공유·얇게 iso-param)는 통신 효과 확정 뒤 Fig2
- 코드 = `feat/fig1-latent` **c034422** (기본값 비트동일 확인)
  - 역할 약속 판정기 + 원거리 COLREGs `penalty` 같이 쓰기 허용(`config.py` assert · `vessel_gym._reward`)
  - `eval_ckpt.py --sim_from_env COLREGS_FAR_RANGE,COLREGS_FAR_MODE` = trunk 체크포인트를 2단계 보상으로 평가(허용 키 2개만)
  - `eval_scripted.py --policy learned --mean_action` = 평균 행동 주 평가(`lrm_*` 파일)
  - `_run_f.sh stage2` + `VESSEL_F_TRUNK_FROM` = trunk·관문 기록·코덱 재사용 → 갈래 → 분기 검사 → 주 평가(lr·lrm) → 곡선 → `_fig1.md`·`_fig1_mean.md`
  - `meaneval_ckpts.sh` 에 `VESSEL_F_STAGE2=1`
- Windows 에 전달한 배치(Windows 클론 `C:\work\DT_Vessel_v3`, c034422):

| 접두어 | 무엇 | GPU |
|---|---|---|
| `qs_` | 시드 43 2단계 갈래 off·comm (qd_ trunk, 코덱 k=8 재사용) | 0 |
| `qs2_` | 시드 44 2단계 갈래 (qd2_ trunk, k=6) | 1 |
| `qd3_` → `qd5_` | 시드 45 → 47 trunk(qd_ 방법, trunk 모드) | 2 |
| `qd4_` | 시드 46 trunk | 3 |
| 절제 | 시드 44 qd2_comm 14·16M far56·shuffle·decl0·state (worktree `C:\work\DT_Vessel_abl` = 5e678dc, pull 금지) | 3 |

- 끝난 뒤 명령(같은 블록): qs_·qs2_ 갈래 평균 행동 meaneval(10·12·14·16M) · 각 trunk 4M·9.04M meaneval
- 결과는 Windows 세션이 이 `analysis` 브랜치에 올림 → `git fetch` 뒤 이 커밋 이후 것을 봄. 실행이 실제로 시작됐는지는 아직 모름(이 파일 작성 시점)

## 2. 결과 오면 할 분석 (순서)

1. **안전(스펙 §3-1)**: `meaneval_qs*/_meaneval_*.md` 의 off 16M 평균 행동 도착 ≥ s43 **64.4** · s44 **63.8** %
2. **45–47 행선지(§3-2)**: `_fig1_mean.md`(lrm) 에서 시드 43·44 **둘 다** 통신이 Woerner·도착·배 충돌 앞섬 + 안전 통과 → 45–47 = `stage2`(접두어 qs3_·qs4_·qs5_, `VESSEL_F_TRUNK_FROM=qd3_` 등, 코덱은 runner 가 새로 고름). 하나라도 아니면 대안 A = 그 trunk 접두어로 `phase1`(1단계 갈래)
   - 이 규칙의 세부는 세션이 정했고 저자에게 '결과 전 이의 있으면 말해 달라'고 알림 — 이의 오면 결과 보기 전 스펙 고침
3. **2×2**: {off, comm} × {1단계 qd_·qd2_, 2단계 qs_·qs2_}. 주의: 1단계 갈래는 주 평가가 표본(`lr_`)뿐 → 평균 행동 Woerner 비교하려면 qd_·qd2_ 끝 모델에 `eval_scripted.py --policy learned --mean_action` 4회 더 필요(1단계 env, 명령 아직 안 줌)
4. **COLREGs 원인 확인**: Woerner 성분(T 방향·D·A 등) + 평가 줄 `[조율/레이더 밖 56-300m]`(양보 우현·둘 다 우현·유지 좌현)을 qd_ vs qs_ 로. 가설 = 2단계에서 통신 배만 56–300 m 좌현이 줄고 일찍 우현
5. **곡선 M**(`_curve.md`, 2단계 보상, trunk 점 포함) 두 팔 우상향인지 — 보고만
6. **시드 44 절제**(`_repro_out_qd2/ablate/_ablate_qd2_.md`): 10-28 스펙 §6 읽는 법 = far56 이 OFF ± 3 pt 안이면 '이득 = 56 m 밖 통신' · decl0 하락이 far56 하락의 ½ 이상이면 역할 선언 출처 문제(저자 보고)
7. **trunk 45–47**: 관문(`_gate.log` 마지막 4창 도착 ≥ 30·벽 ≤ 5) + 9.04M 평균 행동 도착(그 시드 안전 기준값 = 이것 − 5)
8. 보고는 `analysis/runs/<날짜>_.../` 에 PDF(make_pdf.sh 방식) 올림. 시드별 승패 같이, 시드 1–2개 단독 주장 금지

## 3. 아직 안 만든 코드 (평가 전용, 학습 무관)

- **이른 행동 지표**: 조우별 첫 우현 ≥ 10° 변침 때 거리·tcpa, 56 m 밖에서 풀린 조우 비율 → `colregs_woerner.py`/`eval_scripted.py` 에 줄 추가 → 끝 모델만 다시 평가
- **MoE 보고**: 팔별 전문가 사용 몫 · 첫 우현 변침이 어느 전문가에서 났는지(통신 = 레이더 접촉 전 '조우 없음' 전문가면 이른 행동의 직접 증거)
- **5 시드 판정기**(§3-3): 시드별 짝 차이 평균 ± 95 % CI(t, df 4) + 승 수, 성공 = 승 ≥ 4/5 · CI 0 안 걸침. 주 지표 = 배 충돌·도착·DCPA·Woerner·방향 바꾼 총량, lrm 기준
- 나중(저자 우선순위): Fig2 MoE 종류 비교 — 같은 2단계 방식, 2–3 시드. 과거 9월 B1–B4(옛 환경, 도착 94–96 % 천장)는 MoE 우위 없었음

## 4. 주의

- Mac = 읽기·쓰기·1–2분 단일 프로세스 테스트만. Mac 파이썬 = `~/.venvs/vessel2/bin/python`(시스템 python3 는 torch 1.9 + numpy 2 깨짐)
- Mac 작업 클론 = `_dev_dyn`(feat/fig1-latent). 분석 브랜치는 scratch worktree 로 커밋
- runner 는 배치 도중 학습·평가 코드가 바뀐 커밋으로 같은 접두어 이어 돌리기를 거부 → 코드 고치면 새 접두어. runs_fig1·docs 만 바뀐 커밋은 허용
- 무작위 행동에서 원거리 벌점 결정당 최대 −9.7 (평균 미측정) — 2단계 epReward 가 1단계보다 낮아 보이는 건 정의 차이. 비교는 같은 보상 곡선 안에서만
