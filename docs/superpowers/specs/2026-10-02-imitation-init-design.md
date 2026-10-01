# 2026-10-02 선생님(vo56) 흉내로 시작하는 trunk — 설계·사전등록 (배치 i_)

저자 승인 2026-10-01 (plan mode, "쉽게 설명해봐 그리고 ultrathink 설계도안" → 승인). 코드 = 브랜치 `feat/imitation-init` (feat/reward-v3 3aa0636 위, 새 파일만).

## 1. 왜

- 규칙 배 평가 2026-10-01 (`runs/2026-10-01_scripted/`, 학습 없음, 평가 시드 3개 3/3)
  - vo56 (56 m 참값 VO): 도착 90.0 %·배끼리 충돌 9.3 %
  - vo300 (300 m 참값 VO): 도착 94.3 %·충돌 5.1 %
  - goal (안 피함): 50.3 %·49.7 %
- 학습 OFF t_off = 53.1 %·42.2 % ≈ 안 피하는 배 수준 → 피하는 법을 못 배움. G1(도착 ≥ 90 %·충돌 ≤ 5 %) 불통과
- 통신 정보에 가치가 있다는 건 규칙 배로 확인됨. 남은 문제 = 학습이 그 수준에 못 감
- 가설(측정 안 함): imo 타 슬루 3°/s = 결정당 1.2°. 결정마다 바뀌는 탐색 잡음(a0 표준편차 ≈ 0.35 = ±10°)은 타가 0 근처에서 맴돌게 만듦 → '20–30초 꾸준히 틀기'가 우연히 거의 안 나옴
- 계획서 §11 "OFF 유능성 미달 → 다음은 교사 모방 시작안", next_steps ⑯(나) "전 팔 공통 교사 모방 시작 trunk(교사 vo56)"를 구체화한 것임

## 2. 바꾸는 것 — 하나뿐: trunk 의 시작 가중치

나머지(보상 v3·env·평가·팔 정의·분기 규약 §8-1)는 t_ 와 같음. env export 는 `runs/2026-09-30_t/_run_t.sh:39-48` 과 1:1.

1. **init**: `vessel_gym_train.py --arm OFF --steps 0 ... --seed S --save i_init_d6_sS.pt` (학습기가 만든 0 스텝 체크포인트 = 형식·스냅샷 그대로)
2. **DAgger** (`Python/imitation/dagger_init.py`, 기본값 = 사전등록 값)
   - 선생님 = vo56: `verify/check_reward_rank.py` vo_action 글자 그대로 복사(`Python/imitation/vo_teacher.py`), R 56 m·H 60 s·dt 2 s, 추력 +1. 56 m 안 배의 위치·속도 참값을 씀(레이더 ray 아님) — OFF 가 원래 감지하는 범위 밖 정보는 없음
   - env = trunk 와 같은 env(envs 128 × 16척, 스냅샷에서 생성), 결정 D = 3000
   - 운전: 배마다 에피소드 시작 때 확률 β 로 선생님 / 학생(샘플 행동). β = 1(결정 0–500) → 직선 감소 → 0(결정 2000) → 0(3000까지)
   - 라벨: 모든 상태에 선생님 행동 a*. 링 버퍼 64결정(131,072 샘플). 결정마다 미니배치 2048 × 4번
   - 손실 = 가중 MSE(tanh(mean), clip(a*, ±0.995)), 가중 [타 1.0, 추력 0.1]. Adam lr 3e-4, ctr_actor 파라미터만, grad clip 0.5. logstd 미학습(init [-1, −0.5] 유지)
3. **critic 예열**: actor 고정, 학생 샘플 행동 rollout 32 × 128 × 16 을 15번. GAE·ValueNorm·value loss·minibatch 512·2 epoch = vessel_gym_train 과 같은 식. critic 전용 파라미터만(actor 와 공유하는 레이더 인코더는 고정)
4. **저장**: init dict 에서 model_state_dict·value_norm 만 교체, optimizer_state_dict 제거(trunk 는 새 Adam), steps 0. cfg_snapshot 에 `init_imitation` 기록 + sidecar `i_dagger_d6_sS.imitation.json`
5. **trunk**: `vessel_gym_train.py --arm OFF --resume i_dagger_d6_sS.pt --resume_at 0 --steps 9043968 --comm_on_at 0` + t_ trunk 와 같은 인자. 학습기 수정 없음(resume_at ≠ comm_on_at 이라 분기 검사 대상 아님, value_norm 로드, Adam 새로)
   - 주의: trunk 체크포인트 스냅샷에는 `init_imitation` 이 안 이어짐(학습기 미수정). 출처 = `i_` 접두어 + sidecar JSON + 로그
6. **갈래**: `run_repro.sh` branch_batch 그대로(VESSEL_REQUIRE_TRUNK=1 → `i_trunk_d6_sS.pt` 재사용, 곡선 CSV 복사, check_branch). 팔 = `off a6 offb`, 워밍업 2400. 갈래 뒤에는 선생님 없음

## 3. 공정성

- 선생님은 공통 trunk 앞(0 스텝 이전)에만 → OFF·통신이 같은 시작점(§8-1 유지)
- 선생님 정보 ⊆ 56 m(레이더 범위). 통신 정보(300 m)는 안 씀
- OFF 를 강하게 만드는 쪽(기준선 원칙 2) → 통신이 이기기 더 어려워짐. 그대로 받아들임

## 4. 실행

- Windows 한 줄: `bash /c/Users/OSH/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-02_imitation/_run_i.sh phase0`, P0 통과 뒤 `... phase1`
- phase0 = 스모크(preflight 강제) → init·DAgger 시드 43–47 → 주 평가(eval_ckpt `--arm OFF --envs 256 --eval_decisions 10000 --burnin 2400 --seed 999`) → P0 표
- phase1 = trunk 5개 → 관문 → 갈래 off·a6·offb → 주 평가 → 판정 표

## 5. 사전등록 (결과 전 고정 — 결과 보고 안 바꿈)

- **P0 (phase0 관문)**: 학생(DAgger 직후, PPO 전) 주 평가에서 시드마다 도착 ≥ 80 % 그리고 배끼리 충돌 ≤ 15 %. 5시드 중 3개 이상 통과 → phase1
  - 불통과면 멈추고 보고. 하이퍼파라미터 바꿔 재시도 안 함. 다음 후보(저자 결정): OFF 입력을 56 m 상대 목록(ARPA)으로 · 행동을 '목표 방향 기준 침로 오프셋'으로
- **trunk 관문** = t_ 와 같은 규칙: 마지막 4창 평균 goal ≥ 30 %·oColl ≤ 5 %, 시드 순서 앞 3개 → 갈래 시드. 3개 미만이면 멈춤
- **G1** (trunk 마지막 창 도착 ≥ 90 %·충돌 ≤ 5 %·장애물 ≤ 2 %): 판정·기록만, 멈춤 조건 아님
- **망각 감시(기록만)**: trunk 로그 1M·2M 근처 창의 goal 이 그 시드 P0 학생 goal 보다 20 pp 넘게 낮으면 'PPO 망각'으로 보고. 창 측정과 주 평가는 규약이 달라 참고값. 고치려면 학습기 수정(BC 보조손실) 필요 → 이번 범위 밖
- **판정**: 2026-09-30 스펙 §7 그대로 — 같은 시드 짝 3/3 + 평균차 > N, 시드별 승패 병기, 최선 OFF 풀 {i_off, t_off, s_off, x_off}(goal ≥ 50 인 것만)
- **보고**: i_off 와 t_off 를 나란히(흉내 시작의 OFF 효과), i_a6 vs i_off(통신 효과)

## 6. 정직한 기대 (주관)

- 학생이 선생님에 가까우면(도착 80–90 %) OFF 가 크게 강해짐
- 규칙 배의 통신 이득은 도착 +3.6 pp·충돌 −3.5 pp 로 작았음 → 학습 배도 비슷하거나 더 작을 것. 잡음 N(충돌 12.1 pp)보다 작으면 통과 아님
- 통신 갈래는 보상 그대로라 '크게 돌아가기'를 다시 배울 수 있음 → 연료·방향은 여전히 질 수 있음

## 7. 확인 기록 (Mac, 2026-10-01, CPU 작은 설정 — 숫자 해석 금지)

- `dagger_init.py --smoke` rc=0: BC 손실 0.387 → 0.282 → 0.116, 타 일치율 27.5 → 48.3 %, 저장 키 = arm·cfg_snapshot·comm_active·model_state_dict·seed·steps·value_norm (optimizer 없음)
- 학습기 `--resume i_dagger_smoke.pt --resume_at 0 --steps 4096` rc=0: "[resume] ... | Adam 없음(재축적)", 저장 steps 4096·arm OFF·comm_active False
- `eval_ckpt.py --ckpt i_dagger_smoke.pt --arm OFF` rc=0
- vo_teacher.vo_action 본문 == check_reward_rank.vo_action 본문(device 줄·주석 제외 31줄 일치)
