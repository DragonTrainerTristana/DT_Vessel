# 2026-10-13 trunk 갈림길 실험 q_ — 설계·사전등록

저자 결정(2026-10-13 대화, plan mode 승인): n_ trunk 불통과 뒤 "지금 어떻게 해야 할지 설계". 설계 workflow(사실 3·설계안 3·비판 2·종합 1, 읽기 전용)의 종합안을 저자가 4건 승인.
코드 = 브랜치 `feat/fig1-latent`(8d6bc88 위). 이 문서가 결과 전 고정본. 나머지(행동·입력·코덱·지표·판정 규칙)는 `2026-10-07-pure-rl-fig1-design.md`·`2026-10-02-fig1-latent-design.md` 그대로.

## 1. 왜 (사실, 출처 = 10-13 설계 workflow 사실 묶음)

- n_(10-09): 순수 PPO trunk 마지막 4창 도착 27.3 %(관문 30), 배 충돌 65–76 %, 결정당 보상 −0.63 → −1.02 하락. 3.6M 최고 43.1 % 뒤 하락하며 시간초과가 충돌로 바뀜(`runs/2026-10-02_fig1/out/n_trunk_d6_s43.log`)
- 보상 산술(침로 모드, v3+R1+R2): 위험 없는 직진 +2.75/결정(전진 +1.80 · 각도 +1.50 · 시간 −0.35 · 연료 −0.20). 정면 조우 1건의 회피−직진 할인 반환 차 ≈ +205 ≈ 수익 잡음 σ(ValueNorm 130–140)의 1.5배 → **보상 크기는 충분**
- 그러나 그 이득의 ≈ 90 %가 300→56 m 구간 위험 비용(직진 −866 중 −534)이고 OFF(ARPA@56)는 그 원인을 못 봄. GAE 실효창 1/(1−γλ) = 16.8 결정 안에서는 변침−직진 어드밴티지 ≈ 0(전타 중 −0.3~−0.5/결정, dcpa 는 25결정 뒤 움직임) → PPO 는 관측으로 설명되는 직진 shaping 만 따라감 = **가설 B(관측 못 하는 비용)**. t_ 천장(도착 53 %)·f2_ 갈래 퇴화(77→62 %, 흉내 꺼진 뒤)도 같은 설명
- 보조: 침로 모드 초기 잡음(logstd −1.0 은 타각 기준 주석 → 침로 ×120° 로 σ 40°, 결정의 82 %가 전타 임계 10° 초과)·LR 상수 3e-4(후반 드리프트)
- 기각(비판 2건): 보상 가중치 조정(CLAUDE.md 금지 항목·창 안 차 여전히 ≈0)·밀도 커리큘럼·trunk 연장(실행↔조건 매핑 변경, t_/f2_/n_ 비교 단절, 전환점 V 예약)·λ/γ(09-30 기각 인접)·N_EPOCH(stale advantage 만 2배)

## 2. 저자 답 (10-13)

| 질문 | 답 |
|---|---|
| 다음 75분 | T-A + T-B 동시 |
| 가설 B 확정 시 | **위험 비용(충돌코스 #6·per-pair #7)만 56 m 로, 판정기 v2 는 300 m 유지**(두 팔 동일). 10-07 "300 m 유지" 결정을 이 증거로 번복 |
| T-B 내용 | logstd 초기 dim0 −1.0 → −2.0 + LR 3e-4 → 3e-5 선형 감쇠(16.06M 까지) |
| 진단 D | 두 trunk 와 동시에 돌림(학습 없음) |

## 3. 단계 1 — 75분 동시 3건 (시드 43, Windows GPU 3장)

| 런 | 접두어 | n_ 과 다른 것 하나 | 묻는 것 |
|---|---|---|---|
| T-A | `qa_` | `VESSEL_PARTNER_RANGE=300`(참값 상태 필드, 메시지 없음). Fig1 뿌리 불가(진단) | OFF 가 300 m 를 보면 순수 PPO 가 배우나 → 가설 B |
| T-B | `qb_` | `VESSEL_LOGSTD_INIT0=-2.0 VESSEL_LR_FINAL=3e-5 VESSEL_LR_DECAY_DEC=16056320` | 최적화가 지렛대인가 |
| D | `n_` | 학습 없음: 체크포인트 2·4·6·8·9.04M logstd 추세 · 4M·9.04M 표본 vs 평균 행동 평가 · 9.04M +1 update 재개로 critic 설명분산 EV | n_ 정책이 잡음인지 퇴화인지, critic 이 수익을 맞추는지 |

- 보상 = n_ 그대로(v3 + R1 cum + R2 F6, 반경 300). 순위 관문 재실행 없음(보상 동일 → n_ 기록 0/3(도구 결함)·타각 진단 2/3·10-09 저자 override 를 그대로 적용, `_rank_note.txt` 로 남김)
- **관문(결과 전 고정):** 두 trunk 모두 10-07 스펙 §5-3(마지막 4창 goal ≥ 30 % · oColl ≤ 5 %). 기준선 n_ = 27.3 / 0.6, vColl 69.8
- **B 확정 기준:** T-A 가 §5-3 관문을 통과(oColl ≤ 5 포함)하고 마지막 4창 goal ≥ 50 % **이고** 4창 vColl ≤ 49.8 %(n_ −20 %p). gate9 가 4창 vColl 평균을 기록함
- **분기(결과 전 고정, 위에서부터 첫 번째로 맞는 규칙 하나만 적용):**
  1. B 확정 → 단계 2. T-B 도 §5-3 통과면 그 토글을 단계 2 에 같이 켬
  2. B 미확정 · T-B §5-3 통과 → T-B trunk 로 바로 gate → codec → off·comm(시드 43, 같은 접두어 `qb_` phase1). T-A 가 §5-3 통과했어도(goal 30–50) '약한 B' 로 기록만
  3. B 미확정 · T-B 불통과 → 멈추고 보고(T-A 통과 여부 무관, T-A 는 기록만). 다음 후보(저자 결정): 흉내 초기화를 trunk 에만 재도입(갈래 순수 PPO) 또는 보상 재설계
- D 는 보고만. 기록 기준: logstd 9.04M 전문가 평균 ≤ −1.5 / 평균행동 goal − 표본 goal ≥ 15 pt / 9.04M 평균행동 < 4M 평균행동 −5 pt. EVb(λ=1)는 T=32 rollout 의 부트스트랩 목표에 대한 설명분산 = critic 자기일관성이지 MC 수익 적합이 아니므로 값만 기록(기준 없음)
- qa_·qb_ 의 순위 관문: 보상이 n_ 과 같으므로 재실행 없이 n_ 기록(0/3 도구 결함 · 타각 모드 2/3)과 10-09 저자 override 를 그대로 적용(`_rank_note.txt`). 이 적용 범위는 10-13 저자에게 명시 보고

## 4. 단계 2 — B 확정 시 (저자 10-13 사전 승인)

- T-C `qc_`: OFF = ARPA@56 그대로 + `VESSEL_REWARD_RANGE=56`(충돌코스 #6·per-pair #7 이 레이더 안에서만, **두 팔 동일**) + T-B 토글(통과 시). 판정기 v2(R1·R2)는 COMM_RANGE 300 그대로 → 통신 배만 지킬 수 있는 작은 항(−20/건)으로 남음 = 통신 가치. 역할 선언·메시지 내용 불변
- 순위 관문 **v5**(= v4 + REWARD_RANGE 56, 타각 모드, 규칙 (i)–(v) 그대로)를 T-C trunk 와 병렬(다른 GPU, ≈ 88분). v5 불통과면 T-C 는 기록만 하고 멈춤(대체·override 없음 — 새 보상이므로 10-09 override 는 적용 안 됨)
- T-C 관문 = §5-3 그대로 → gate → codec → off·comm(시드 43) → 평가·곡선(M)·표(10-07 스펙 그대로) → 통과(저자 판단) 시 시드 43·44·45
- 보상 정의 변경이므로 이전 배치(t_·f2_·n_·qa_·qb_)와 결정당 보상·epReward 는 직접 비교 못 함(결과 %·DCPA·Woerner 는 비교 가능)
- 구현 규약(10-13 리뷰 반영): `vessel_gym._pairwise` 의 `risk` 키(판정기 v2 주 상대·역할 선언·eval COLREGs 게이트)는 반경 `risk_range`(= COMM_RANGE 300) 그대로, `reward_range` 는 보상용 `risk_rw` 에만 적용. 둘이 같으면 옛 경로와 비트동일. `eval/eval_mixed.py`·`comm_codec.collect_p50` 은 300 고정이라 56 모델에 쓰지 않음(q_ 는 collect46 만 씀)

## 5. 구현 (전부 토글, 기본값 = 비트동일)

- `Python/config.py`: `LOGSTD_INIT0`(−1.0) · `LR_FINAL`(= LEARNING_RATE) · `LR_DECAY_DEC`(0). `REWARD_RANGE` 는 이미 있음(sim 스냅샷 키)
- `Python/networks.py`: ControlActor logstd 초기 dim0 = `config.LOGSTD_INIT0`
- `Python/vessel_gym.py`: `risk`(판정·선언·평가) 반경 = `risk_range`, `risk_rw`(보상 #6·#7) 반경 = `reward_range`(다를 때만 따로 계산)
- `Python/vessel_gym_train.py`: env `reward_range` = `REWARD_RANGE` 가 있으면 그 값, 없으면 COMM_RANGE(옛대로) · `LR_DECAY_DEC>0` 이면 update 마다 lr 선형 보간(재개 뒤 Adam 복원 lr 도 덮음) · 재개 시 스냅샷의 `logstd_init0·lr_final·lr_decay_dec` 가 현재와 다르면 거부 · `VESSEL_GRAD_TELEMETRY=1` 이면 GAE 뒤 `[ppo] EVb(lam=0.95)·EVb(lam=1)·ret_std·T` 출력
- `Python/ckpt_io.py`: 스냅샷 키 `logstd_init0·lr_final·lr_decay_dec` 추가 · `make_env_from_snapshot` 이 스냅샷 sim.REWARD_RANGE 를 env 에 넘김(없으면 COMM_RANGE)
- `Python/verify/check_branch.py`: 같은 trunk 묶음 안 `opt`(위 3키) 일치 강제 · `verify/check_reward_rank.py`: 설정 `v5`, env 가 REWARD_RANGE 를 실제로 쓰고 `used['reward_range']` 로 기록 · `verify/_verify_comm_mirror.py` BASE 에 새 키 4개 기본 고정
- `Python/runs_fig1/2026-10-02_fig1/_run_f.sh`: MODE `trunk`(IMIT=0 전용) · `VESSEL_F_ARPA_RANGE`(trunk 모드만 56 아님 허용, `_diag_only.txt`) · `VESSEL_F_EXTRA_ENV`(화이트리스트 3키, 전역 export → trunk·off·comm 동일) · `VESSEL_F_REWARD_RANGE`(56 만 → 관문 v5) · `VESSEL_F_RANK_PARALLEL`(관문 PID·rc 별도, `_rank_rc.txt`) · `VESSEL_F_ALLOW_BUSY` · `VESSEL_F_GPU`/`VESSEL_F_RANK_GPU`(동시 배치는 GPU 명시) · 접두어별 `_batch_env.txt` 기록·대조(이어갈 때 토글 누락 차단) · 결과 사본은 `out/<접두어>/` · smoke 는 토글 없는 env 로
- 새 `diag_trunk.sh`·`logstd_dump.py`·`eval_mean.py`(Dropbox `report/diag/eval_mean.py` 사본, 평균 행동 평가)

## 6. 실행 (Windows LICS-SERVER, 클론 루트, `git pull` 뒤)

```
CUDA_VISIBLE_DEVICES=3 bash Python/runs_fig1/2026-10-02_fig1/diag_trunk.sh n_                                                   # D (GPU 3, ≈45분)
VESSEL_F_IMIT=0 VESSEL_F_ALLOW_BUSY=1 VESSEL_F_GPU=0 VESSEL_F_PREFIX=qa_ VESSEL_F_ARPA_RANGE=300 bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh trunk
VESSEL_F_IMIT=0 VESSEL_F_ALLOW_BUSY=1 VESSEL_F_GPU=1 VESSEL_F_PREFIX=qb_ VESSEL_F_EXTRA_ENV="VESSEL_LOGSTD_INIT0=-2.0 VESSEL_LR_FINAL=3e-5 VESSEL_LR_DECAY_DEC=16056320" bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh trunk
# (두 번째 명령은 첫 명령의 헤더 로그가 찍힌 뒤 시작 — 같은 클론의 git fetch 겹침 방지)
# 분기 2(T-B 로 진행): 같은 접두어·같은 EXTRA_ENV(다르면 _batch_env.txt 대조·학습기 재개 검사가 멈춤)
VESSEL_F_IMIT=0 VESSEL_F_PREFIX=qb_ VESSEL_F_EXTRA_ENV="VESSEL_LOGSTD_INIT0=-2.0 VESSEL_LR_FINAL=3e-5 VESSEL_LR_DECAY_DEC=16056320" bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh phase1
# 분기 1(B 확정): 단계 2 — 관문 v5 는 GPU 2 에서 병렬
VESSEL_F_IMIT=0 VESSEL_F_PREFIX=qc_ VESSEL_F_REWARD_RANGE=56 VESSEL_F_RANK_PARALLEL=1 VESSEL_F_RANK_GPU=2 VESSEL_F_EXTRA_ENV="<T-B 통과 시 같은 값>" bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh phase1
```
- 단계 1 벽시계 ≈ 80분. 단계 2 ≈ 4.5시간(trunk ∥ 관문 90 + 코덱 15 + 갈래 90 + 평가·곡선 60)

## 7. 하지 않는 것

- 보상 가중치(CMD_MISMATCH·전진·충돌)·밀도 커리큘럼·trunk 연장·λ/γ·N_EPOCH·흉내 재도입·offb·3시드 선행 — 단계 1·2 모두 금지
- Fig1 OFF 정의(ARPA@56)는 안 바꿈. T-A 는 진단 trunk(check_branch 의 no_msg 규칙 그대로라 갈래 뿌리로 못 씀)

## 8. 솔직한 확률

- 단계 1 이 원인을 가를 확률 높음(어느 결과든 정보). Fig1 까지 ≈ 40 %
- B 확정 뒤에도 T-C 가 못 배우면 관측 탓이 아니라 PPO 자체 → 다음 후보 = 흉내 초기화를 trunk 에만(갈래 순수 PPO), 저자 결정
