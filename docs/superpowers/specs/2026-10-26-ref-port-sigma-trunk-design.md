# 2026-10-26 레퍼런스 포팅 — 무학습 재평가 + σ trunk 3팔 (설계 초안, ★저자 승인 대기)

근거: `Private_Paper_Project/2026-10-26_A2Z_진단_할일.pdf`(진단) · `2026-10-26_레퍼런스_포팅.pdf`(논문·코드 원문 대조, analysis/runs/2026-10-26_ref_port).
코드 = 브랜치 `feat/fig1-latent`(b6a4341 위, 토글 4개 + 재평가 스크립트, 기본값 비트동일). 결과 보기 전 고정본. 행동·입력·보상·코덱·지표는 2026-10-07·10-13 스펙 그대로.

## 1. 왜

- 학습 OFF 가 '안 피하는 goal 배'(50.3 %) 수준을 못 넘음. 같은 정보의 규칙 배 vo56 은 90.0 %
- 원인 후보 1순위 = 침로 모드 탐색 잡음: a0×120°, 침로 오차 10° = 전타, logstd 하한 −2.3(σ 12°)도 문턱보다 큼. 엔트로피 squash 항이 θ 상수라 보너스가 logstd 를 늘 +1 로 밂(n_ None 전문가 −0.83 → −0.19)
- n_ 9.04M: 평균 행동 54.6 % vs 표본 28.8 %
- 레퍼런스: CleanRL·SB3·gSDE PPO 엔트로피 0, Andrychowicz 2021 §3.8 보너스 이득 없음·하한 작게, gSDE 재표집 간격(저역통과 시스템), 다수 문헌이 평균 행동으로 평가

## 2. 단계 R0 — 무학습 재평가 (학습 없음, Windows GPU 4장 ≈ 2 h)

- `meaneval_ckpts.sh`: qb_ trunk(2·4·6·8·9.04M)·off·comm(10·12·14·16M), qa_ trunk(2–9.04M)을 표본/평균 행동 짝으로 평가 + logstd 덤프 + 갈래 마지막 K 점 평균
- 기록만(관문 아님). 보고 = 체크포인트별 평균−표본 격차, qa_ 평균 행동 도착, qb_ off·comm 순서가 평가 방식·추정량(끝점 vs K 점 평균)에 따라 바뀌는지

## 3. 단계 R1 — σ trunk 3팔 (시드 43, 75분 동시, GPU 3장)

| 런 | 접두어 | n_ 과 다른 것 | 묻는 것 |
|---|---|---|---|
| A | `ra_` | `VESSEL_ENTROPY_BONUS=0` | 엔트로피 압력만 빼면 σ 가 스스로 줄고 도착이 오르나 |
| B | `rb_` | `VESSEL_ENTROPY_BONUS=0 VESSEL_LOGSTD_MIN=-3.0` | 하한이 전타 문턱(10°) 아래(6°)로 내려갈 수 있으면 더 나아지나 |
| C | `rc_` | `VESSEL_NOISE_HOLD=4`(1.6 s 유지) | 엔트로피는 그대로 두고 잡음을 저역통과로 만들면 나아지나 |

- 세 팔 모두 `VESSEL_GRAD_TELEMETRY=1`(로그만). 보상 = n_ 그대로(v3 + R1 cum + R2 F6, 반경 300). 순위 관문 재실행 없음(보상 동일 → n_ 기록·10-09 override 적용, `_rank_note.txt`)
- 비교 기준선(같은 시드·같은 보상): n_ 27.3 % · qb_ 41.9 %(학습 창 마지막 4창)
- **관문(결과 전 고정)**: 10-07 스펙 §5-3 그대로(마지막 4창 도착 ≥ 30 % · oColl ≤ 5 %)
- **처방 성공 기준(결과 전 고정, 제안값 — 저자 확정 필요)**: 9.04M 표본 행동 도착 ≥ 55 %(goal 배 50.3 위) **이고** 평균−표본 격차 ≤ 10 pt. 둘 다 `meaneval_ckpts.sh` 로 잼
- ★표본 평가의 잡음 과정(10-26 리뷰): 평가(eval_ckpt·meaneval)는 세 팔 모두 매 결정 i.i.d. 표본. C(`rc_`)는 학습 때 잡음을 4결정 유지하므로 C 의 '표본' 값은 학습 행동 정책과 다른 잡음으로 잰 값임(복원 로그에 경고가 찍힘). C 는 성공 기준을 '평균 행동 도착'과 '학습 창 관문'으로 함께 보고, 표본 값은 참고로만 적음 — 평가를 유지 잡음으로 바꿀지는 저자 결정(바꾸면 eval_ckpt 토글 추가, 기본 비트동일)
- 보고만: update 별 approx_kl·clipfrac·logstd0_raw 추이, 4M→9.04M 평균 행동 변화
- 분기(위에서부터 첫 규칙 하나): ① 성공 팔이 있으면 그중 표본 도착 최고 팔 하나로 off·comm 갈래(같은 접두어 phase1) ② 관문만 통과한 팔이 있으면 멈추고 보고 ③ 전부 불통과면 멈추고 보고. 다음 후보(저자 결정) = 행동공간 증분형(P14) · 보상 반경 56 m(P6, 10-13 B 미확정 → override 필요)

## 4. 하지 않는 것

- 보상 가중치·반경·지표 정의·평가 프로토콜 변경(주 평가를 평균 행동으로 바꿀지는 저자 결정 C1, 이 배치는 둘 다 기록)
- 흉내 재도입·3시드 선행·offb
- 미니배치·clip·target_kl(P7·P16)은 R1 의 approx_kl 이 0.02 를 넘는 증거가 있을 때만 제안

## 5. 실행 (Windows LICS-SERVER, 클론 루트, `git pull` 뒤)

```
# R0 (학습 없음)
CUDA_VISIBLE_DEVICES=0 bash Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh qb_ trunk_d6
CUDA_VISIBLE_DEVICES=1 bash Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh qb_ off
CUDA_VISIBLE_DEVICES=2 bash Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh qb_ comm
CUDA_VISIBLE_DEVICES=3 VESSEL_F_ARPA_RANGE=300 bash Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh qa_ trunk_d6
# R1 (승인 뒤, 앞 명령의 헤더 로그가 찍힌 뒤 다음 명령)
VESSEL_F_IMIT=0 VESSEL_F_ALLOW_BUSY=1 VESSEL_F_GPU=0 VESSEL_F_PREFIX=ra_ VESSEL_F_EXTRA_ENV="VESSEL_ENTROPY_BONUS=0 VESSEL_GRAD_TELEMETRY=1" bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh trunk
VESSEL_F_IMIT=0 VESSEL_F_ALLOW_BUSY=1 VESSEL_F_GPU=1 VESSEL_F_PREFIX=rb_ VESSEL_F_EXTRA_ENV="VESSEL_ENTROPY_BONUS=0 VESSEL_LOGSTD_MIN=-3.0 VESSEL_GRAD_TELEMETRY=1" bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh trunk
VESSEL_F_IMIT=0 VESSEL_F_ALLOW_BUSY=1 VESSEL_F_GPU=2 VESSEL_F_PREFIX=rc_ VESSEL_F_EXTRA_ENV="VESSEL_NOISE_HOLD=4 VESSEL_GRAD_TELEMETRY=1" bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh trunk
# R1 끝난 뒤 각 팔 9.04M 평가(표본/평균): CUDA_VISIBLE_DEVICES=<g> bash Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh ra_ trunk_d6  (rb_·rc_ 같음)
```

## 6. 솔직한 확률

- R0 은 원인 몫을 가를 확률 높음(학습 없음, 어느 결과든 정보)
- R1 에서 성공 기준을 넘는 팔이 나올 확률 ≈ 35–45 %. 잡음이 '수준'만 설명하고 평균 정책의 직진 표류(보상−안전 불일치)는 못 고칠 수 있음 — 그 경우 다음은 보상 쪽(P6) 또는 행동공간(P14)
