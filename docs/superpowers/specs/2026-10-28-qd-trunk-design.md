# 2026-10-28 qd_ trunk — 위험 비용 56 m + 떨림 수정 (설계·사전등록)

저자 결정(2026-10-28 대화)
- "ㅇㅋ 그렇게 가는게 맞다": 위험 비용(충돌코스 #6·per-pair #7) = 56 m, 두 팔 동일 · 역할 판정기 v2 = 300 m 유지(벌점만)
- "보상을 바꿔도 돼요, 내가 원하는 논리대로 가는게 가장 중요": 논리 = 통신 배는 멀리서 미리 알아 일찍·작게 꺾음 → COLREGs·DCPA·매끄러움이 나아짐, 보상은 두 팔 동일
- 시간 부족: trunk 1개 먼저 → 잘 되면 병렬

근거: `Private_Paper_Project/2026-10-27_R0_무학습재평가_표본vs평균행동_결과.pdf` · `2026-10-28_R0_해석_다음단계.pdf` · `2026-10-28_미리피하기_보상_AtoZ.pdf`. 코드 = `feat/fig1-latent` 071487b + 이 문서·`meaneval_ckpts.sh` 체크포인트 선택(runs_fig1·docs 만).
10-13 스펙 단계 2(B 확정 시 qc_)의 조건(B 확정)은 미충족이었음 → 이 배치는 그 분기가 아니라 저자 결정으로 하는 새 trunk 임.

## 1. qd_ 가 바꾸는 것 (n_ 대비, 시드 43)

| 항목 | n_ | qd_ | 근거 |
|---|---|---|---|
| 위험 비용 반경 | 300 m | **56 m**(`VESSEL_F_REWARD_RANGE=56`, 두 팔 동일) | R0: n_ 직진형·qa_ 우회형 둘 다 300 m 비용과 맞음 · 문헌은 감지 범위 안에서만 |
| 역할 판정기 v2 | 300 m | 300 m 그대로 | 저자 결정 |
| 엔트로피 보너스 | 0.01 | **0** | None 전문가 σ 가 학습 내내 커짐(qb_ 18.6° → 33.7°) · CleanRL·SB3 기본 0 |
| logstd 하한 | −2.3(12°) | **−3.0(6°)** | 하한이 전타 문턱 10° 보다 큼 |
| logstd 초기 dim0 | −1.0(44°) | **−2.0(16°)** | qb_ 2M 격차 ≈ 0(σ 가장 작을 때) |
| 계측 | 없음 | `VESSEL_GRAD_TELEMETRY=1`(로그만) | KL·clipfrac·전문가별 logstd |

- 네 가지를 한 번에 바꿈(시간 때문) → 성공해도 어느 것이 효과인지는 이 배치로 모름 → 성공 시 하나씩 뺀 trunk 를 병렬로(§4)
- 순위 관문 v5(= v4 + 보상 반경 56, 타각 모드) 를 trunk 와 동시에 다른 GPU 에서 돌림. 불통과면 결과는 기록만 하고 갈래로 안 감(저자 결정 대기)

## 2. 관문·성공 기준 (결과 전 고정 — 세션 제안, 저자 이의 없으면 확정)

- trunk 관문(기존 10-07 §5-3 그대로): 학습 창 마지막 4창 도착 ≥ 30 % · oColl ≤ 5 %
- 처방 성공(셋 다):
  1. 9.04M **평균 행동** 도착 ≥ 65 %(n_ 54.6 보다 +10)
  2. 4M → 9.04M 평균 행동 도착 하락 ≤ 5 pt(중간 하락 없음)
  3. 9.04M 평균 − 표본 도착 격차 ≤ 10 pt(떨림 해결)
- 평가 = `meaneval_ckpts.sh`(R0 와 같은 인자 · 시드 999), 4M·9.04M 만. 평균 행동 = tanh(mean), 표본 = 학습 정책 그대로
- 보고만: 표본 도착·배 충돌·시간초과·첫 변침·무조우 오프셋·logstd·approx_kl 추이
- 시드 1개 → 단독 주장 아님. 다음 실험 선택용

## 3. 실행 (Windows, `C:\work\DT_Vessel_v3`, Git Bash)

```
cd /c/work/DT_Vessel_v3
git fetch origin && git checkout feat/fig1-latent && git pull
VESSEL_F_IMIT=0 VESSEL_F_ALLOW_BUSY=1 VESSEL_F_GPU=0 VESSEL_F_PREFIX=qd_ VESSEL_F_REWARD_RANGE=56 \
  VESSEL_F_RANK_PARALLEL=1 VESSEL_F_RANK_GPU=1 \
  VESSEL_F_EXTRA_ENV="VESSEL_ENTROPY_BONUS=0 VESSEL_LOGSTD_MIN=-3.0 VESSEL_LOGSTD_INIT0=-2.0 VESSEL_GRAD_TELEMETRY=1" \
  nohup bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh trunk > qd_driver.log 2>&1 &
# trunk 끝('trunk 모드 완료' 또는 관문 결과) 뒤 평가 4개 동시:
S=Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh
for a in "0 4 sample" "1 4 mean" "2 9.04397 sample" "3 9.04397 mean"; do set -- $a
  CUDA_VISIBLE_DEVICES=$1 VESSEL_F_REWARD_RANGE=56 VESSEL_F_ONLY=$2 VESSEL_F_WHICH=$3 nohup bash $S qd_ trunk_d6 > qd_eval_$2_$3.log 2>&1 &
done
# 넷 다 끝나면: VESSEL_F_REWARD_RANGE=56 bash $S qd_ --table
```

## 4. 다음 (결과 뒤)

- 성공 → 병렬: (a) qd_ 갈래 off·comm(같은 접두어 phase1) (b) qe_ = qd_ + 연속 위험 예고 PBRS(브랜치 `feat/early-pbrs`, `2026-10-28_미리피하기_보상_AtoZ` §4) trunk·갈래 — PBRS 있/없 OFF 비교로 OFF 를 해치지 않는지 확인 (c) 하나씩 뺀 trunk(원인 분해). 확증은 그 뒤 시드 3개
- 실패(관문 통과·성공 기준 미달 포함) → 행동공간 증분형 침로(P14) 설계·구현으로 감
