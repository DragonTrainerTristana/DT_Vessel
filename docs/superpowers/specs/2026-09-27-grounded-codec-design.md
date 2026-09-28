# 오토인코더 grounded latent 통신(동결 화자 코덱 + 학습 청자) 설계 + 사전등록 — 2026-09-27 (새 주기), 09-28 C 전환

> **결과 전 작성. 저자 승인 전 효력 없음.**
> 잠금 = §0-0 순서의 2단계(이 문서 커밋·해시 기록). 잠금 뒤 변경은 §13 이력에만 추가하고, 판정·코덱·청자 구조를 바꾸는 변경은 금지(§7-9).

- 상태: **개정안 v2.1 — 09-28 저자 결정 'C 로 바꿈' 반영**(§0-6) + **C 전환분 검토 2관점(구현·사전등록) blocking 4건 반영**(§13). §11 저자 결정 대기
  - 직전판 = A 중심 설계(09-27, 3관점 검토 blocking 13건 반영, 991줄). 사본 `scratchpad/grounded_codec_spec_A_version.md`(스크래치패드 = 임시 폴더 → 잠금 때 스펙 폴더로 복사 보존)
  - A 판 검토에서 나온 사전등록 수리(잠금 순서·사전 탐색 공개·동률·열등 정의·다중 비교 문구·H1a 기저율·버그 재실행·시드 추가 금지·붕괴 규칙·앵커 보호·fail-closed generator·팔 구분 문자열·sit [P]·traj cmd t−1)는 전부 C 팔로 옮김
  - C 전환분 검토: 구현·사전등록 관점은 받아 반영함. **인과성 관점 검토는 아직 없음 + 이 반영본 자체도 재검토 전** → §11-29 [필수]
- 기준 코드: `origin/perf/bit-identical-speedup` @7013a3a(배치 X 산출물을 낸 코드, da23ad0 의 후손). 아래 file:line 은 전부 7013a3a 기준(`git show 7013a3a:Python/<파일>`, git root = `_dev_dyn`)
- 구현 브랜치: **7013a3a 에서 새로 딴 브랜치**(가칭 `feat/grounded-codec`). 지금 `_dev_dyn` 체크아웃(`feat/comm-intent` @da23ad0)은 7013a3a 와 eval_ckpt 551줄·vessel_gym 242줄이 다름 → 거기서 구현하면 G-C0 재평가 대조가 성립 안 함
- 형식: 직전 사전등록 `2026-09-25-comm-intent-design.md` 의 절 구성을 따름

---

## 0. 배경·결정 이력

### 0-0. 잠금 순서 (이 순서를 어기면 그 단계부터 무효, §13 에 기록)

1. §11 의 **[필수]** 항목 전부 저자 확정(09-28 C 세부 §11-22–31 포함, §11-29 재검토 반영 포함)
2. 이 문서 커밋 → 커밋 해시를 §13 '잠금 기록'에 적음 = **잠금**
3. 구현 + §9 테스트 PASS(테스트는 고정 seed 합성 코덱 사용) → 커밋
   - 구현 전에 **7013a3a 별도 체크아웃**(Mac·Windows 각각)에서 기준 산출물 생성: 새 골든 `ext_v1_off`(+ 승인 시 `ext_v1_intent_ON`), 분기 재개 대조 기준(9-33). 새 코드로 만든 기준은 무효
4. Windows 에서 코덱 데이터 덤프(3의 커밋 코드로) → 데이터 내용 SHA 기록
5. 코덱 3개 학습 → `Python/comm_codecs/*.pt` 커밋 → 내용 SHA 를 §13 에 기록 → 팔 env 블록(§4-7)의 SHA 핀 자리 채움(그 외 글자 변경 금지)
6. 오프라인 충실도표 생성(§6 코덱 충실도 항목, holdout + traj, z 토큰 지도학습 상한 probe·z 성분별 표준편차 포함 — 보고 전용) → 커밋 = 동결
7. G-C0 재사용 팔 재평가(2단계, §7-5)
8. 신규 27런(9팔 × 3시드) 한 배치 제출(§5-2)

- 선언(작성 시점 사실)
  - 잠금 전 25D 페이로드 코덱 학습 0회, 코덱 데이터 덤프 0회, 25D 충실도 측정 0회
  - **C 청자(z 토큰 + 수신자 자기 상태, k/v 3×256) RL 학습 0회**. 청자 용량 근거는 지도학습 상한 탐색(`sup`, 운동 부분 10D 토큰)뿐(§0-3)
  - z 토큰·T2 형식 토큰 지도학습 probe 실행 0회(z 토큰 probe 는 6단계에 보고 전용으로 계획, §6)
  - A 판(A2·A6·A12 등) 학습 0회 — A 결과를 보고 C 로 바꾼 것 아님
- 충실도표(6단계)를 본 뒤 λ·τ·threat 정의·복호 규칙·k·청자 구조를 바꾸지 않음. 예외는 §3-11 의 '코덱 단계 버그' 뿐
- 잠금 전 사전 탐색(4D/6D 소형 AE, 지도학습 상한 등)은 §0-3 에 전부 공개

### 0-1. 왜 새 주기 문서인가

- 이 실험은 계획서(`runs/2026-09-24_plan`)에 없음 → 루트 CLAUDE.md §0 에 따라 저자 승인이 먼저임
- 직전 스펙 §6 중단 규칙 "이번 주기에 채널·보상 변형 추가 없음" → 배치 X 주기 안에서는 채널을 못 바꿈 → **새 주기 사전등록으로 분리**
- 설계 시점 = 배치 X 결과(09-26)와 A/B 실험(09-27)을 본 **뒤**. 논문에 시점 명시. 이 계열을 'onl6 를 고친 것'으로 쓰지 않음
- 채널 변형 주기 수 공개: 이번이 **3번째**(1차 latent 파일럿 → 배치 X COMM_EXT → 이번). 09-28 A→C 전환은 같은 주기 안, 학습 전 변경이라 주기 수를 늘리지 않음. 이 사실(설계안 2개 중 C 채택, 시점)도 공개. 이번 결과는 이후 모든 보고에 앞 두 주기 결과와 같이 적음(§7-13)
- 기존 H2 판정(불지지, F(5,12)=0.39)과 별개. H2 를 뒤집는 근거로 안 씀(대상이 창발 latent vs 동결 코덱 latent + 학습 청자로 다름)

### 0-2. 배치 X 근거 (원자료 `runs/2026-09-26_comm_ext_batch/_FOLLOWUP_0927.md`, `_abl_dvcoll.md`)

| 팔 | vColl s43 | s44 | s45 | 평균 | 비고 |
|---|---|---|---|---|---|
| off | 45.2 | 45.3 | 43.5 | 44.7 | 시드 범위 1.8 |
| arpa6 | 49.4 | 46.7 | 44.1 | 46.7 | OFF 대비 3/3 나쁨 |
| onl6 | 52.0 | 51.7 | 47.5 | 50.4 | 3/3 나쁨, +5.7 pp → **G6(H1a) FAIL** |
| ons6 | 32.5 | 41.1 | 37.7 | 37.1 | 3/3 좋음 |
| oni6 | 36.1 | 30.3 | 26.4 | 30.9 | 3/3 좋음 |
| rand(A/B 후속) | 45.5 | 43.0 | 38.2 | 42.2 | OFF 대비 2/3 좋음 |

- ons6 latent0 절제 ΔvColl −3.4/−1.4/−4.9(3/3 개선) → 지금의 창발 latent 는 해로움
- ons6 스냅샷 `comm_latent=1.0`(FOLLOWUP §3 표) → **기존 ons6 는 무병목 상한이 될 수 없음** → 상한 팔 C∞ 를 새로 학습
- ons6·oni6 는 명시 쌍별 필드(ext v1) + 청자 k/v 1×64. 이번 C 는 필드 없이 청자가 배움 → ons6 크기의 이득을 C 의 기대값으로 쓰지 않음(§7-11)
- oni6 intent0 절제 +0.3/+0.7/+0.0 → 평가 시점 의도 필드 기여는 작음
- 재학습 잡음(같은 trunk·같은 설정, 1차 off vs G6 off): vColl 짝차 4.1/5.8/12.4 pp(1차 조건, RMS 8.2)
- trunk: `x_trunk_d6_s43/44/45` (sha 9827ae54e5cd / f3eb18bc0661 / d393ae2fd737), 분기점 9,043,968, 끝 16,056,320, 갈래 1개 ≈ 1 h(x_ons6_s45 61.5 min 실측, k/v 1×64 기준)
- 레이더 dropout: 배치 X 15개 갈래 diag JSON 의 스냅샷 줄 전부 `radar_dropout_p=0.0`. 재개 때 sim 상수 불일치를 거부하므로(config.py:556 SIM_SNAPSHOT_KEYS) trunk 도 같음

### 0-3. 사전 탐색 이력·선택 편향 공개 (잠금 전, 이 스펙 설계에 쓰임)

- 전부 앞 설계 검토(wf3) rl 관점이 스크래치패드에서 돌린 것. 스크립트·출력: `ae.py`→`ae_state.txt`·`ae_intent.txt`, `ae_uniform.py`→`ae_uniform.txt`, `sup.py`→`sup.txt`, `sens.py`, `flicker_dist.py` (스크래치패드 = 임시 폴더 → 잠금 때 스펙 폴더로 복사해 보존). 09-28 개정 때 추가로 돌린 것은 위양성 시뮬(`rv/h1a_8.py`·`rv/fpr2.py`, 데이터 안 씀)과 파라미터 계산(`kv_params.py`)뿐

| 탐색 | 개수 | 학습 데이터 → 평가 데이터 | 본 것(요지) |
|---|---|---|---|
| 4D 운동 AE(`ae_state`, sinψ·cosψ·SOG·ROT) | AE 16개 = k{1,2,3,4} × 잡음{0, 0.05} × 과제항 λ{0,1} + 해석적 절단 2개 | traj_x_off_s43·s44 → traj_x_off_s45·**x_ons6_s44·x_oni6_s45**(처치 팔 궤적) | k=2 λ0: \|dh\| 중앙 1.47–1.58°, 역할 붙은 쌍 뒤집힘 10.1–41.0% / k=2 λ1: 0.47–0.48°, 1.8–10.5% / 해석적 절단 k=2(ψ·SOG, rot=0): 뒤집힘 0% |
| 6D 운동+명령 AE(`ae_intent`) | AE 10개 = k{2..6} × λ{0,1}, 잡음 0 | 같은 train → x_oni6_s45 만 | k=2 λ1: \|dv\| 중앙 0.226 m/s, 역할 붙은 쌍 뒤집힘 37.3–68.6% / k=2 λ0: \|dh\| 10.06°, 43.1–68.6% / k≥3 λ1: 4.5–10.2% |
| 합성 균일 데이터 AE(`ae_uniform`) | k=3·4 | 균일 합성 → x_ons6_s44 | k=3 꼬리 \|dh\| p99.9 16.5°, 최대 144° → trunk rollout 데이터 채택 근거 |
| **지도학습 상한(`sup`)** | k/v 1×64·3×256 × 표본 2e5–2e7 | 쌍 3.05M(x_off_s43·s44) → 0.51M(x_ons6_s44) | 1×64·2e7 에서 역할 정확도 28.4–67.1%, 3×256·2e7 에서 91.7–94.6% → **청자 용량 결정 근거(§4-3)** |
| 민감도·자연 역할 변동(`sens`, `flicker_dist`) | – | traj | 침로 1°·SOG 0.01 m/s 부터 역할 뒤집힘 시작, 자연 변동 0.27–0.63% → τ 자릿수 근거 |

- 탐색 AE 의 과제항은 이번 L_pair 와 다름: 연속 쌍 특성 8개에 가중 [1,1,1,1,3,3,10,3]×50, 역할 CE 없음, 학습 4000 step·lr 2e-3 cosine
- `sup` 입력 = 원시 토큰 10D [relpos 3(방위 sin·cos, d/300), 파트너 sinψ·cosψ·SOG/1.8·rot, 자기 sinψ·cosψ·SOG/1.8] = 이번 C∞ 토큰의 운동 부분과 같은 구성(자기 ROT 없음). 라벨 = 참 dcpa_risk·내 역할·상대속도 2D. z 입력·RL 학습은 안 봄
  - 학습 = Adam lr 1e-3, 배치 1024 복원 추출, step 수 = 표본/1024(2e6 → 약 1,950 step, 2e7 → 약 19,500 step) — `sup.py` 확인
  - 파트너 침로가 sin/cos 로 들어간 토큰임 → **T2 의 스칼라 ψ/180 형식에는 이 근거가 안 닿음**(§3-10·§4-3)
- **인과성 결함 공개**: 6D 탐색의 cmd 는 traj `cmd` = 결정 t 의 행동 a_t(eval_ckpt.py:551) → 송신 시점보다 한 칸 미래 값(§3-6). 그 측정의 cmd 성분 수치는 참고로만 씀
- 문구 정정: 초안의 "λ=1 스윕 안 함" → "**이 25D 코덱은 스윕 안 함. 4D/6D 탐색에서 λ 0/1 비교는 봤음**"
- 선택 편향 공개
  - 저자 결정 3(과제 가중)은 위 탐색(λ1 이 4D k=2 뒤집힘을 줄임)을 본 뒤 나옴
  - 저자 결정 1 의 명령(cmd) 포함은 oni6(30.9) 결과를 본 뒤 나옴. wf3 은 같은 이유(oni6 vs ons6 2/3, intent0 +0.3)로 intent 제외를 권했음(wf3_digest §2-1)
  - 처치 팔(ons6·oni6) 궤적을 탐색 평가에 썼음. 이번 코덱 학습 데이터에는 안 씀(§3-6)
  - **09-28 C 전환과 청자 용량 3×256 은 `sup` 결과(1×64 한계)를 본 뒤 정함**. wf3 은 C 를 '진단용 1팔'로 권했음(성공 가능성 A > C 추정) — 저자가 논문 주장 때문에 C 를 주 실험으로 정함(§0-6)
- 논문 방법 절에 이 표 요약을 넣음(의무)

### 0-4. 저자 결정 (2026-09-27, 이 스펙의 전제)

1. **접근 = A-확장**: 송신 배 j 가 결정 시점에 알 수 있는 값 묶음 P_j(약 25D)를 동결 코덱으로 k차원 z 로 압축해 보냄. 수신 배는 동결 디코더로 P̂_j 복원 → `_pair_core` 로 쌍별 필드 계산, 나머지는 토큰에 덧붙임
   - **09-28 결정 6 으로 주 실험에서 대체.** A 경로는 k=6 하나(A6)만 보험·읽기 진단으로 남김(§4-12). 송신측(P_j·동결 코덱)은 그대로
2. **k ∈ {2, 6, 12}** — 저자 원문 "latent msg 는 2~12차원, 현재랑 동일하게" → C2·C6·C12 로 유지
3. **코덱 손실 = 과제 가중**: 성분별 z-score MSE(P 전체) + 쌍 특성 항(복원값으로 계산한 ext 필드가 참값과 맞도록). λ 결과 전 고정. 논문 명칭 '과제 인지 코덱(task-aware codec)' — 유지
4. **T2 팔 포함**: 코덱 없이 원값 2개(침로 ψ·SOG)만 보냄 — 유지
   - 수신측 처리('rot=0·cmd=0·나머지 0 으로 쌍별 계산')는 09-28 C 전환으로 **'청자가 원값 2개를 직접 읽음'**으로 바뀜(§3-10)
5. **창발 latent 끔**(COMM_LATENT=0), 통신 팔 보조손실 0(AUX_LOSS_SCALE=0), **off·arpa6 은 배치 X 런 재사용**(전제: 새 코드에서 OFF 경로 골든 PASS) — 유지
   - 09-28 검토로 전제를 구체화: 기존 골든 5케이스는 전부 EXT=0·agile·grid3x3·처음부터 2 update(test_golden.py:56-66, 바깥 VESSEL_* 는 :88 에서 지움) → x_off·x_arpa6·offr 의 실제 경로(EXT=1 1×64 attn·imo·none·crossing 0·trunk 분기 재개 vessel_gym_train.py:672-724)를 안 덮음. **전제 = 새 골든 `ext_v1_off` PASS(9-3) + 분기 재개 대조 비트동일(9-33)**

- 앞 검토(wf3) 권고와 다른 점: 검토는 z-score MSE 단독·k∈{1,2}·payload 4D·intent 제외·A 주 실험을 권했음. 저자가 과제 가중·{2,6,12}·25D·cmd 포함·(09-28) C 주 실험으로 정함 → 이 스펙은 저자 결정을 따르고 대가(§3-5, §4-3, §7-11, §8)를 기록

### 0-5. 인과성 확인 요약

- `compute_own_future`(vessel_gym_train.py:182-221)는 rollout 뒤 버퍼의 미래 위치 `pos[t+h]` 로 만드는 사후 라벨(:1087-1089 에서 호출) → 송신 시점 t 에 없음 → **페이로드에서 제외**
- 나머지 성분은 전부 §2-2 표에서 '결정 t 에 송신자가 가진 값'인지 코드로 확인. 미래 정보 누설 없음
- 단 `sit` 는 자기 계측이 아니라 타선 참 운동학 유래 → [P] 로 분류(§2-2, §8-3)
- **수신자 자기 상태 토큰 own4**(§2-5) = 결정 t 의 수신자 env 값, P[0:4] 와 같은 함수·같은 시점 → 자기 계기 [S], 미래 정보 없음
- 저장 궤적(traj)으로 P 를 다시 만들 때는 명령 시점을 한 칸 당김(§3-6)

### 0-6. 저자 결정 (2026-09-28) — 원문 'ㅇㅇ C로 바꿔줘'

6. **주 실험 = C**(동결 화자 코덱 + 학습 청자)
   - 이유(전달된 저자 논리): 논문 포인트 **'적은 차원 latent msg 로도 이득'** 을 정직하게 주장하려면 받는 배의 신경망이 z 를 직접 읽고 뜻을 학습해야 함. A 는 고정 복원기 + 손 공식이라 '압축 전송'으로 읽힘
7. **A 는 k=6 하나만 보험(A6)**. A2·A12·A형 S∞·A형 T2 는 폐기

- 전달된 C 설계 전제(바꾸지 않음)
  - 송신 = A 와 같은 페이로드 p25·같은 동결 코덱 E(과제 가중 손실, k ∈ {2,6,12}, 8bit). **C(k) 와 A6 는 같은 z**(같은 코덱 내용 SHA)
  - 수신 = 토큰 [relpos 3, z_k(양자화값), 수신자 자기 상태, msg×0] → GroundedAttention k/v MLP 가 학습. 디코더·`_pair_core`·ext 필드 경로 안 씀
  - 수신자 자기 상태를 토큰에 넣는 이유 = 쌍별 관계(상대속도·CPA·역할)를 집계 전에 만들 수 있게(구조 한계 ②, §4-1)
  - 공정성 = P0 도 같은 토큰 구성(relpos + 수신자 자기 상태, z 자리 0) → C(k) vs P0 차이는 z 뿐
  - 팔 = P0·C2·C6·C12·C∞·R6·T2·A6·offr(신규) + off·arpa6(재사용) + ons6·oni6·onl6(보고만)
  - 청자 k/v 용량 = 모든 통신 팔 동일, 결과 전 하나로 고정
  - 판정 = 확증 비교 C6 vs off 1개
  - 명칭 = 'autoencoder-grounded latent communication (동결 화자 코덱 + 학습 청자)', 'emergent' 금지
- 출처 표기: 저자 원문은 'ㅇㅇ C로 바꿔줘' 한 줄. 위 전제는 그 앞에 제시된 C 안 내용(워크플로 전달). **이 문서가 정한 세부**(청자 k/v 3×256·own4 구성·P0 슬롯 폭 6·C∞ 입력 형식·T2 표현·토큰 폭 규칙)와 09-28 검토 반영분('latent' 문구 규칙·offr 조건)은 §11-22–31 저자 확인 전 효력 없음
- 시점: 이번 주기 학습 0회 상태에서 바꿈(결과 전)

---

## 1. 저자 의도 → 코드 → 측정 대응

| 저자 의도 | 코드 | 드러나는 곳(측정) |
|---|---|---|
| 송신 배가 결정 시점에 아는 것만 보냄 | `own_payload(env, x, goal, sit)` [E,N,25] (§2, 새 함수) — 결정 t 의 env 상태·자기 obs 만 읽음 | 테스트 9-7(가용성·행 국소성), §8-3 ORACLE 분류표 |
| **적은 차원 latent 로 보냄** | 동결 인코더 E → tanh → 8bit 양자화(§3), k ∈ {2,6,12} | 전송 형식. 표기 = 코드 k×8 bit + 위치 무손실(§3-3) |
| 통신 채널 전체(위치 공유 + own4 토큰 + 학습 청자 + z)가 이득인가 | cz 팔 전체 경로 | **확증 C6 vs off**(G-C2) — 답하는 것은 '채널 전체 효과'(§5-4) |
| **받는 배 신경망이 z 를 직접 읽고 뜻을 학습**(= 저자 논문 포인트 '적은 차원 latent 로도 이득') | 토큰 [relpos 3, z_k, own4, msg×0] → GroundedAttention k/v(3×256) 학습(§4-1·§4-3). D·`_pair_core` 안 씀 | **G-C3 ①(C6 vs P0)·②(C6 vs R6)** + slot-shuffle·msgzero 개입(§6). 'latent' 문구 허용 조건 = §7-4 |
| 쌍별 관계를 집계 전에 만들 수 있게 | 수신자 own4 를 토큰에(§2-5, 구조 한계 ②) | own0 절제(보조), P0 대조 |
| 이득이 z 의 송신자별 내용에서 왔는지 | P0(같은 토큰·같은 shape, z 자리 0), R6(z 를 송신자 단위 derangement) | G-C3 ① → 'latent' 문구 허용, ① + ② → '송신자별 내용에 귀속'(§7-4) |
| 압축 가치 | C2 vs T2(같은 16 bit·같은 폭·같은 격자) | G-C4(**k=2 에서만**) |
| 손 공식 읽기 vs 학습 읽기 | A6: 같은 z → 동결 D → `_pair_core` → ext 39(§4-12) | 읽기 진단 G-C5(A6 vs C6) |
| 병목 없는 상한 | C∞: 참 P 25D 를 청자가 직접 읽음 | 상한 게이트 G-C1(C∞ vs off) |
| 과제 인지 코덱 | 손실 = z-score MSE + λ·쌍 특성 항(λ=1 고정, §3-4) | 오프라인 충실도표(G-C0) |
| 창발 latent·ON 전용 aux 제거 | COMM_LATENT=0, AUX_LOSS_SCALE=0 | H1a(§7-7), 통신 팔 목적함수 = OFF 와 대칭(aux 는 comm_active 일 때만 켜짐 :986·:991·:1135-1138 확인) |
| off·arpa6 재사용, 잡음 추정 | 같은 trunk SHA 묶음, 새 코드의 v1 경로 비트동일(처음부터·분기 재개 둘 다), 파일 SHA 전후 대조, offr 재학습 3런 | 골든 5케이스 + **새 골든 `ext_v1_off`(9-3) + 분기 재개 대조(9-33)** + 재평가 2단계(G-C0) + §5-2 앵커 보호 + §7-10 |

---

## 2. 페이로드 P_j 정의 (레이아웃 'p25', D = 25) — 송신측, A 판과 같음

### 2-1. 결정 t 의 타이밍 (코드 확인)

- `VesselBatchEnv.step`(vessel_gym.py:1268-1318): `_apply_action`(:1270, cmd 세팅) → 서브스텝 10회(:1271-1272) → `_update_situation`(:1274) → 레이더 → 보상·종료 → 재스폰 시 `_update_situation` 재호출(:1307) → `_build_obs`(:1316/:1318)
- rollout: `x = fs.get()`(:997) → `comm_gather(...)`(:1000-1002) → 행동 표본 → `env.step`(:1020). `compute_own_threat` 도 step 전(:1043)
  - comm_gather 가 읽는 env 상태 = obs_t 를 만든 상태(송신 P_j·수신 own4_i 둘 다)
  - `env.cmd_rudder`·`env.target_speed` = t−1 행동의 명령(다음 `_apply_action` 전, vessel_gym.py:587-588 은 step 안에서만 갱신)
- 재스폰된 배는 재스폰 값(:559-564)이 곧 결정 t 상태
- gym 의 situation 은 1-step stale 이 아님(:1274·:1307 에서 post-step 재계산 후 obs 로)

### 2-2. 성분 표

| idx | 이름 | 식 | 출처(file:line @7013a3a) | 결정 t 가용 근거 | 분류(§8-3) |
|---|---|---|---|---|---|
| 0 | sinψ | sin(env.heading_j·DEG) | 상태 vessel_gym.py:405, 적분 :607(누적각) | 자선 선수방위. obs[364]=wrap180(ψ)/180(:945)의 다른 표현 | [S] |
| 1 | cosψ | cos(env.heading_j·DEG) | 위와 같음 | ±180 불연속 회피 | [S] |
| 2 | sog | env.speed_j / 1.8 (COMM_EXT_SOG_NORM, :245) | :406, 갱신 :596-612 | 자선 대수속력 계기. obs[362] 는 speed/max_speed(:942)라 정책 obs 에 절대값 없음 | [S] 계기값 |
| 3 | rot | yaw_rate_deg(rudder_j, speed_j, max_speed_j) / MAX_YAW_RATE | :195-206, comm_pair_features :327 과 같은 식 | obs[363] 과 같은 식(:943-944) = 자기 obs | [S] |
| 4 | cmd_rudder | env.cmd_rudder_j / 30 (MAX_TURN_RATE) | 세팅 :587 | t−1 자기 명령. 재스폰 직후 0(:564) | [S] |
| 5 | cmd_speed | env.target_speed_j / 1.8 | 세팅 :588 | t−1 자기 명령. 재스폰 직후 U(0.2,0.5)×max(:562) | [S] |
| 6 | goal_dist | obs_j[360] = d/(d+150) | :934 | 자기 obs(자기 위치·목표) | [S] |
| 7 | goal_angle | obs_j[361] = SignedAngle(선수, 목표)/180 | :935-941 | 자기 obs. **송신자 선수 기준 각** | [S] |
| 8:13 | sit one-hot 5 | one_hot(obs_j[368]) [None,HeadOn,StandOn,GiveWay,Overtaking] | `_update_situation` :907-917 → `_pairwise` :781-808 → cascade :824-857 → obs :947-956 | 송신자 obs 에 있는 값이지만 **자기 계측 아님**: `_pairwise` 가 모든 배의 env.heading·env.speed 참값으로 bearing·rel_vel·dcpa 를 계산하고 56 m 안 최위험 상대의 상황을 고름 | **[P] 유래**(56 m 안 타선 참 운동학, OFF obs 에도 있음) |
| 13:25 | threat top-3 × [sin a, cos a, dist, closing] | `compute_own_threat(x_j, 3)` | vessel_gym_train.py:239-260 | 자기 레이더 frame stack(:45-62)의 최신·직전 프레임만 | [S] |

- **sit 중계의 뜻**: 수신자 입장에서 송신자의 56 m 안 제3선(수신자 레이더 밖일 수 있음)에 대한 참 운동학 유래 판정이 넘어옴 → 특권 정보 전달. 저자 결정 1 이 '상황'을 넣었으므로 성분은 유지하고 **표기만 [P]** 로 함(§8-2, §8-3, §11-10). C∞ 는 이 성분을 직접 읽고, C(k)·A6 는 코덱이 남긴 만큼 받음
- **threat 정의(코드로 확인)**
  - cur = 최신 프레임 + 0.5 ∈ [0,1](:247), prev = 직전 프레임(:248)
  - ray 인덱스 = 선수 기준 각(deg), ray0 = 선수 +Z, 시계방향(= 우현 +)(vessel_gym.py:439-441, 회전 :641-645) → **송신자 선체 좌표**
  - 거리 = dist/RADAR_RANGE(56), 레이더 대상 = 타선 OBB·벽(±299.5)·장애물(이번 regime 은 none)(:631-670)
  - top-3 = **ray 단위** 최근접 3개(:253). 표적 단위 아님 → 한 척이 여러 ray 를 차지하면 3칸이 같은 표적일 수 있음(L 14.18 m 선박이 20 m 거리면 약 40° — 계산값). 실효 정보는 12D 보다 작을 것(추정). NMS 로 바꿀지는 §11-6
  - closing = prev − cur, **같은 ray 인덱스**(:255). 표적 추적 아님 → 다른 물체가 ray 에 들어오면 값이 튐(추정)
  - 미감지 = cur ≥ 0.999 → 칸 전체 0(:257-258). 감지 칸은 sin²+cos²=1 이라 (0,0,·,·) 로 구분 가능
  - 재스폰 직후 frame stack 3장 모두 현재 프레임(:55-58) → closing 0
  - 레이더 dropout = 0.0(배치 X 스냅샷 확인, §0-2)

### 2-3. 뺀 후보와 이유

| 후보 | 결정 | 이유 |
|---|---|---|
| own_future(미래 궤적) | **제외** | 사후 라벨(§0-5) |
| 실제 타각 rudder(obs[365]) | 제외 | `_pair_core` 는 rot 만 씀. imo 에서 rot = (rudder/30)·SOG/R_FULL(:203) → SOG 와 함께 중복 |
| speed_ratio(obs[362]) | 제외 | SOG·max_speed 의 함수 |
| max_speed | 제외 | agile 식만 필요(:204-206), 이번 regime 은 imo |
| 위치 x,z(obs[366:368]) | P 밖 | 모든 통신 팔에서 relpos 로 **코덱 밖 무손실 공유**(기존 comm 팔 가정 유지). bit 표기에 따로 적음(§3-3) |
| 레이더 원시 360 | 제외 | 저자 결정 범위 밖. 위협 요약(top-3)만 |
| RadarEncoder 특징 30D | 제외 | 정책 파라미터의 함수 → 동결 코덱 원칙 위반 |

- **총 D = 25** (운동 4 + 명령 2 + 목표 2 + 상황 5 + 위협 12)
- 본질 자유도(추정): 운동·명령 5, 목표 2, 상황 범주 1, 위협 3–9 → 약 11–17. k=6·12 는 운동·명령에 사실상 무손실일 것(추정, §7-11)
- P 의 모든 성분은 식 자체로 [−1, 1] 안(sin/cos, /1.8, /MAX_YAW_RATE, /30, d/(d+150), /180, one-hot, [0,1) 거리, [−1,1] closing) → C∞ 는 이 식 그대로 읽음(§2-4)

### 2-4. 정규화

- 코덱 입력 = 성분별 z-score: (p_c − μ_c)/s_c. μ_c·s_c 는 **코덱 학습 데이터 train 분할에서 한 번 계산해 meta 에 고정**. s_c = max(std_c, 1e-3)
- 코덱 3개가 같은 μ·s 를 씀
- 평가·학습 중 재계산 없음(배치 통계 정규화 금지 규약 — Assets/Scripts/.claude/CLAUDE.md §6)
- **z-score 는 코덱 내부 전용.** C∞ 청자 입력은 §2-2 식 그대로의 P(z-score 아님)
  - 이유: z-score 는 분산 작은 성분(드문 sit one-hot, closing 등)을 s_c 바닥 1e-3 까지 키울 수 있음(최대 ×1000) → 청자 입력 크기가 팔마다 달라짐. z(tanh) 는 (−1,1), P 식도 [−1,1] → 값 범위가 C(k)·C∞·T2 에서 같은 자릿수

### 2-5. 수신자 자기 상태 own4 (C 토큰 전용, 09-28 신규)

- own4_i = [sin(ψ_i·DEG), cos(ψ_i·DEG), speed_i/1.8, yaw_rate_deg(rudder_i, speed_i, max_speed_i)/MAX_YAW_RATE]
  - **P[0:4] 와 같은 식·같은 함수**(`own_motion4(env)` 하나를 own_payload 와 토큰이 같이 부름). 배 i 의 own4_i 는 i 가 송신자일 때의 P_i[0:4] 와 비트 동일(테스트 9-8)
- 명령 2(cmd)·목표·상황·위협은 넣지 않음
  - 이유: 쌍 관계(상대속도·CPA·역할)는 두 배의 현재 운동과 상대 위치로 정해짐. `comm_pair_features` 가 쓰는 수신자 입력도 pos_i·h_i·spd_i 뿐(vessel_gym.py:288-292). 수신자 자기 목표·상황은 이미 자기 obs(query·fc2)에 있음
  - ROT_i 는 `_pair_core` 에 안 들어가지만 송신 운동 4성분과 같은 식으로 맞추려고 넣음(대칭). obs[363] 과 같은 값이라 새 정보 아님
- 새 정보 여부
  - sin/cos ψ_i: obs[364] wrap180(ψ)/180 의 연속 표현 → 정보는 같고 표현만 다름. 세계 좌표 → z·C∞·T2 가 담은 송신자 세계 침로와 결합해 Δψ 를 만들 수 있음
  - ROT_i: obs[363] 과 같음
  - **SOG_i 절대값: 정책 obs 에 없음**(obs[362] 는 비율) → 통신 경로(토큰)로만 들어오는 새 정보. P0 에도 똑같이 있으므로 C(k) vs P0 에서 상쇄(§5-4)
- 분류 [S](수신자 자선 계기, 잡음 0 이상화)
- 유효 파트너가 있는 슬롯에만 들어감(패딩 where, :349). 파트너 0 이면 context 0(networks.py:542-543) → own4 경로도 없음

---

## 3. 코덱 — 송신측, A 판과 같음(참조 팔만 C 로 바꿈)

### 3-1. 구조

- E: 25 → 128 → 128 → k (ReLU, 출력 tanh). D: k → 128 → 128 → 25 (ReLU, 출력 선형 = z-score 공간)
- 파라미터(계산값): 39,705 + 257k → k=2 **40,219** / k=6 **41,247** / k=12 **42,789**
- 정책 파라미터와 분리. Adam·clip 그룹·state_dict 어디에도 안 들어감(§3-8)
- 실행 때 쓰는 부분: C(k)·R6 는 **E 만**(D 는 cdc_* 텔레메트리 전용). A6 는 E·D. 학습 때 L_pair 에 D 필요(§3-4)

### 3-2. k 별 코덱 3개 별도 (nested dropout 1개 대신) — 선택과 이유 (§11-3 확정 대기)

- **선택: k 마다 별도 코덱 3개**, 같은 데이터·레시피·seed
- 이유
  - 속도-왜곡 곡선의 각 점은 그 rate 의 최선 코덱이어야 공정함. C2 vs T2 에서 C2 가 prefix 제약으로 약해지면 압축 가치를 과소평가함
  - nested dropout 은 k 표본 분포가 새 손잡이가 됨 → 결과 전 고정할 값이 늘어남
  - '코덱 3개 SHA 고정'(§7-9)과 대응이 단순함
- 대가
  - 정보 포함관계(z₂ ⊂ z₆)가 구조로 보장되지 않음. 설계상 기대로만 둠
  - k 사이 충실도가 비단조로 나와도 **재학습 안 함**(그대로 보고)
  - k 마다 z 의 '좌표계'가 다름 → C(k) 청자는 k 마다 따로 배움(C 는 팔마다 새로 학습하므로 문제 없음, 교차평가 불가 — §6)

### 3-3. 채널: tanh + 8bit 균일 양자화, bit 표기 규칙

- z = tanh(E(p̃)) ∈ (−1,1)^k
- 격자: 256 단계, Δ = 2/255, 값 = −1 + iΔ
- 학습(코덱): z̃ = z + u, u ~ U(−Δ/2, Δ/2). 실행: q = −1 + round((z+1)/Δ)·Δ (결정론). **청자 토큰에 들어가는 값 = q**
- 전송량(송신자 1척·결정 1회당 방송)

| 팔 | 전송 | bit |
|---|---|---|
| C2 / T2 | 코드 2칸 / 원값 2칸 | 16 |
| C6 / R6 / A6 | 코드 6칸(같은 코덱) | 48 |
| C12 | 코드 12칸 | 96 |
| C∞ | P 25 float32 | 명목 800 — 대역 주장 안 함 |
| P0 | 없음(위치만) | 0 |

- **표기 규칙(논문·표 전부)**: 항상 '코드 k×8 bit + 위치 무손실(float32 2개, 코덱 밖)'로 병기. **'16 bit 통신' 단독 표기 금지**. 수신자 own4 는 자기 계기값이라 전송량에 안 셈(명시)

### 3-4. 손실 (결과 전 고정)

- 전체: **L = L_P + λ·L_pair, λ = 1.0**
- **L_P**: (1/25) Σ_c (p̂_c − p_c)², z-score 공간, 25성분 균등
- **L_pair**: 데이터 속 실제 유효 쌍 (i,j)(반경 300 m 안 nearest-4, 덤프 때 comm_gather 와 같은 규칙으로 저장)마다
  - f = `_pair_core`(수신자 i 참값, pos_j 참값, P_j 참값 복호) — 목표값
  - f̂ = `_pair_core`(수신자 i 참값, pos_j 참값, P̂_j 를 §3-9 로 복호) — 학습과 A6 실행이 같은 복호 함수
  - L_pair = ½·L_cont + ½·L_role
  - **L_cont**: ext v1 연속 성분 10개 [0:8](sin/cos Δψ, sog, rot, relvel 2, dcpa_risk, tcpa_risk) + [18:20](cmd 2). (1/10) Σ_c ((f̂_c − f_c)/σ_c)², σ_c = train 쌍의 참 f_c 표준편차(meta 고정, 바닥 1e-3)
  - **L_role**: 내 역할(i→j)·상대 역할(j→i) one-hot 을 **soft cascade CE** 로. (CE_my + CE_their)/(2·ln 5)
- 수신자 참값을 쓰는 곳은 **오프라인 목표값 계산뿐**. 실행 때 E 의 입력은 P_j 하나 → 배포 경로로 새는 정보 없음
- **C 에서의 뜻**: L_pair 는 'z 가 쌍 계산에 필요한 운동 정보를 우선 담게' 하는 화자 쪽 손실. C 의 청자는 D·`_pair_core` 를 안 쓰고 z 를 직접 읽음 → 청자가 배우는 읽기는 이 손실과 무관(설계자가 정한 것은 z 의 뜻, 청자가 배우는 것은 z 를 쓰는 법)
- **soft cascade** (`encounter_role` vessel_gym.py:249-274 를 확률 연산으로 옮긴 것, 코덱 학습 도구 전용)
  - σ_τ(x) = sigmoid(x/τ). B = 내 방위(수신자 참값), OB = 상대 방위(ψ̂ 의존)
  - approach: still 이면 1, 아니면 σ(raw_tcpa/τ_t). clear_pp = σ_a(−10−B)·σ_a(−10−OB), clear_ss = σ_a(B−10)·σ_a(OB−10). v = [dist≤R]·[\|B\|≤100]·approach·(1−clear_pp)·(1−clear_ss)
  - h = σ_a(15−\|B\|)·σ_a(15−\|OB\|). stern = σ_a(\|OB\|−112.5). fast = σ_s(spd_i − 1.1·SOĜ_j) (their 쪽은 i↔j·B↔OB 교환)
  - p_HO = v·h. ot = stern·(fast + (1−fast)·σ_a(5−\|B\|)). p_OT = v·(1−h)·ot
  - p_X = v·(1−h)·(1−ot)·σ_a(112.5−\|B\|). gw = σ_a(\|B\|−5)·σ_a(B) + (1−σ_a(\|B\|−5))·σ_a(−OB). p_GW = p_X·gw, p_SO = p_X·(1−gw)
  - 게이트: g = σ_d(24 − dcpâ)(DCPA_RISK=24 :127). 비None 확률 × g, p_None = 1 − 합
  - CE = −log(clamp(p_참역할, 1e-6))
  - **hard 로 남는 경계(gradient 없음)**: my_role 의 dist≤R·\|B\|≤100 (수신자 참값만) / their_role 은 인자 교환이라 hard 조건이 \|OB\|≤100 → **ψ̂ 의존** / `still`(vessel_gym.py:306, rel_speed²<1e-4) → **SOĜ 의존**. 이 경계를 넘나드는 복원 오차는 L_role 로 교정되지 않음
- 온도(결과 전 고정): τ_a = 2°, τ_s = 0.02 m/s, τ_t = 1 s, τ_d = 1 m
- 표본 가중: 배치 = 송신 표본 4096개 균일 추출 → L_P 는 이 표본에, L_pair 는 이 송신자들이 들어간 **데이터 속 실제 유효 쌍 전부**에(빈도 그대로)

### 3-5. λ·온도 값과 근거 (결과 전 고정, 튜닝 없음)

- λ = 1.0: L_P(평균 예측 시 1), L_cont(표준화, 평균 예측 시 약 1), L_role(균등 예측 시 1) 을 O(1) 로 맞춘 뒤 대등 가중. **이 25D 코덱은 스윕 안 함**. 4D/6D 탐색에서 λ 0/1 비교는 봤음(§0-3)
- ½·½ 배분도 같은 이유
- 온도: 민감도 탐색(침로 1°·SOG 0.01 m/s 부터 역할 뒤집힘, §0-3)과 같은 자릿수. hard 역할과의 일치율을 데이터로 재서 보고(테스트 9-20). 기준 미달이어도 안 바꿈
- 대가(결과 전 기록)
  - 쌍 항은 운동·명령 6성분에만 걸림 → k=2 코덱은 운동에 몰리고 목표·상황·위협을 거의 버릴 것(추정). 설계자가 넣은 과제 지식 → 명칭 '과제 인지'(§8-1)
  - 빈도 가중이라 먼 쌍에 끌림: x_off_s43 eval 기준 56–300 m 101,575쌍 vs ≤56 m 12,554쌍. 침로 오차 δψ 는 dcpa 에 약 d·δψ 로 들어감(300 m·1.5° ≈ 7.9 m, 문턱 24 m — 계산값) → 충실도표는 거리대별로 나눠 동결하고 **근거리(≤56 m) 값을 주 보고**로 둠. 게이트 형태(hard, 24 m)는 그대로

### 3-6. 학습 데이터·오프라인 점검 데이터

- 출처: trunk 3개(`x_trunk_d6_s43/44/45`, 통신 OFF, 9,043,968 결정). **처치 팔 rollout 안 씀**
- 굴리기: `ckpt_io.restore_policy` + `make_env_from_snapshot`(ckpt_io.py:299, :537), arm OFF, eval_ckpt 와 같은 방식(정책 표본 행동, 전역 seed 고정)
  - seed: trunk s43→1001, s44→1002, s45→1003(평가 seed 999·학습 seed 43–45 와 분리)
  - envs 64, vessels 16, crossing = 스냅샷(0), burn-in 2400 결정, 수집 1500 결정, 2결정마다 저장 → 750 스냅샷
  - 표본: 3 × 64 × 750 × 16 ≈ 2.30M 송신 표본, 쌍은 ×(유효 파트너 수)
- **덤프 시점 규칙**: 덤프 도구(`tools/dump_codec_data.py`)는 학습기와 같은 `own_payload` 를 **env.step 전**(comm_gather 와 같은 자리)에 부름. 테스트 9-7 이 확인: P[4:6] == 직전 행동의 명령(_apply_action 식), 재스폰 행 == 재스폰 값
- 저장 항목(스냅샷마다): P_all [E,N,25], 수신자 참값 pos·heading·speed, topi·pmask(반경 300, K=4)
- 크기: 약 124 B/표본 → 약 285 MB(계산값). **git 밖** `$VESSEL_CKPT_DIR/codec_data/`. 데이터 내용 SHA 를 코덱 meta 에 기록
- holdout: **env 단위 10%** — 각 trunk 의 env 인덱스 ≡ 0 (mod 10) 인 env 전체(7개/trunk). 보고 전용, 모델 선택에 안 씀
- 플랫폼: trunk 파일이 Windows 에만 있음 → 덤프는 Windows. GPU 덤프 + 파일 SHA 동결(초안) vs CPU 결정론은 §11-15
- **분포 이동 점검(보고 전용)**: Mac 의 `traj_x_off/ons6/oni6_*.pt`(16 env × 1500 결정, burn-in 0, seed 999)로 P 재구성
  - **명령 시점 한 칸 당김**: traj `cmd[t]` 는 결정 t 의 행동 a_t(eval_ckpt.py:551 이 :516 에서 뽑은 a 를 :672 env.step 전에 저장). 결정 t 의 P[4:6] 은 **traj cmd[t−1]** 로 만듦: cmd_r = clamp(a0,−1,1)·30/30, cmd_s = clamp((clamp(a1,−1,1)+1)/2·max_speed, 0, max_speed)/1.8 (vessel_gym.py:584-588 과 같은 식)
  - **재스폰 제외**: traj `outcome[t−1]` ≠ 0 인 배의 결정 t 는 재구성 불가(cmd_r=0, cmd_s=U(0.2,0.5)·max 난수) → 제외하고 제외 수 보고. t=0 도 제외
  - threat: 레이더가 traj 에 없으므로 pos·heading 으로 `_radar` 재계산(장애물 none·벽). 재스폰 직후 결정은 closing 0(vessel_gym_train.py:55-58 과 같음)
  - goal: pos·heading·goal 위치로 obs[360:362] 식(:934-941) 재계산
  - sit: traj `situation` 그대로(post-step 값 = 결정 t obs 와 같음)

### 3-7. 학습 설정·재현성·내용 SHA

- Adam lr 1e-3, 75% 지점에서 1e-4, 배치 4096 송신 표본, **40 epoch 고정**(조기 종료 없음), seed 0, 초기화 PyTorch 기본
- CPU, `torch.use_deterministic_algorithms(True)`, 스레드 1. `.numpy()` 안 씀(Mac torch↔numpy 비호환)
- 소요(추정): 코덱 1개 CPU 30–60 min
- 내용 SHA = sha256(정렬된 state_dict 키마다 이름·dtype·shape·원시 바이트(test_golden `_sha` 방식, test_golden.py:80-86) + meta JSON(sort_keys)). pickle 바이트 아님
- 같은 데이터·seed·**같은 플랫폼·torch** → 같은 SHA(테스트 9-19). 플랫폼 간 동일성은 (미확인) — 보장 안 함
- meta: k, D=25, layout 'p25', μ[25]·s[25], σ_pair[10], λ, τ 4개, bits 8, 구조 문자열, epoch·lr·batch·seed, data_sha256, torch 버전·플랫폼

### 3-8. 동결·저장·장치·난수 격리

- 동결: `requires_grad_(False)`, `eval()`, 호출은 `torch.no_grad()` 안에서만. `policy.parameters()`(Adam :642)에 없음 → 갱신 경로 0
- C 에서 청자 gradient 는 토큰의 q 값까지만 감(q 는 no_grad 로 만든 상수) → E 로 가는 경로 0. 화자 쪽 z 의 뜻은 RL 중 안 바뀜
- 위치: networks 모듈 전역 `COMM_CODEC`(정책 밖). rollout 의 comm_gather 만 읽음. ckpt_io 가 덮어씀
- **로드 시점**: config import 때 로드 안 함. config 에는 경로·SHA **문자열만**. 로드는 학습기 `main()` 과 `ckpt_io.restore_*` 에서
- **장치**: 모듈 전역이라 `policy.to(device)` 로 안 옮겨짐 → 로드 함수가 device 인자를 받아 명시적으로 옮김
- **난수 격리**: 코덱 nn.Module 생성·로드는 `CNNPolicy` 생성 **뒤**, 또는 `torch.random.fork_rng` 안에서만 → 코덱 팔(cc2·cc6·cc12·cr6·ca6)과 비코덱 팔(cp0·ccinf·ct2)의 CNNPolicy 초기값이 **같은 shape 끼리 동일**(cp0 = cc6 = cr6, cc2 = ct2). 코덱 비활성이면 **아예 생성 안 함**(골든 default 보호). 테스트 9-14 가 확인
- 파일: `Python/comm_codecs/tac_p25_k{2,6,12}.pt`(약 170 KB, **git 추적**). 폴더 이름은 stdlib `codecs` 와 겹치지 않게 `comm_codecs/`
  - 학습 전 커밋 → preflight 지문(`git diff HEAD --binary`, preflight_checks.sh:27)에 들어감. 미추적 .pt 는 지문에 안 들어감(:29 는 .py/.sh/.json 만) → **커밋 전 배치 금지**
  - run_repro.sh 팔 spec 에 코덱 SHA 를 핀으로 적음 → 지문이 run_repro.sh 로 SHA 변경을 잡음
- 체크포인트 최상위 키 `comm_codec` = {state_dict, meta, sha256} 내장(:1249·:1306 두 저장 지점 모두). `model_state_dict` 밖 → 코덱은 strict 로드·키 결정자 표에 영향 없음(청자 k/v 는 영향 있음 — §4-5)
- 평가는 체크포인트 blob 으로 복원(env 경로 아님). 교차평가만 CLI 인자로(§4-7)

### 3-9. 복호 규칙 (결과 전 고정) — 쓰는 곳: 코덱 학습 L_pair·A6 실행·cdc_* 텔레메트리·C∞ 교차평가

- D 출력 → 역정규화 p̂ = μ + s·d
- ψ̂ = atan2(p̂₀, p̂₁)(rad). p̂₀²+p̂₁² < 1e-12 이면 0
- SOĜ = clamp(p̂₂, 0, 1)·1.8. ROT̂ = clamp(p̂₃, −1, 1). cmd_r̂ = clamp(p̂₄, −1, 1). cmd_ŝ = clamp(p̂₅, 0, 1)
- goal: clamp(p̂₆, 0, 1), clamp(p̂₇, −1, 1)
- sit: one_hot(argmax p̂₈:₁₃) (§11-5 확정 대기)
- threat 칸 m: (ŝ,ĉ) 노름 < 0.5 이면 칸 전체 0(미감지). 아니면 (ŝ,ĉ) 단위벡터로 정규화, dist clamp [0, 0.999], closing clamp [−1, 1]
- 결과를 P 형식 25D(sin ψ̂, cos ψ̂, SOĜ/1.8, …)로 다시 조립하는 함수 `decode_p25` 하나를 A6·C∞ 교차평가가 같이 씀(테스트 9-31)
- 역할 게이트: 기존 그대로 dcpâ < 24 m(`role_gate=True`, vessel_gym.py:319-322). their_role 은 '상대가 선언'이 아니라 '상대 관점 기하 역할(수신측 계산)'
- **C 팔 실행 경로에는 복호 없음**(청자가 q 를 직접 읽음)

### 3-10. T2 부호 (코덱 없음, C 판 — 09-28 재정의)

- 슬롯 2칸 = [ψ_n, v_n]
  - ψ_n = wrap180(ψ_j)/180 (송신자 obs[364] 와 같은 식, vessel_gym.py:236·:945)
  - v_n = 2·(speed_j/1.8) − 1
  - 둘 다 **z 와 같은 격자**로 양자화(§3-3: 256 단계, Δ = 2/255, 결정론 반올림) → ψ 해상도 360/255 = 1.41°(오차 ≤ 0.71°), SOG 해상도 1.8/255 = 0.00706 m/s(오차 ≤ 0.0035) — 계산값
- 16 bit = C2 와 같음. 슬롯 폭·값 범위(−1,1)·격자가 C2 와 같음 → C2 vs T2 차이 = 사상(동결 코덱 vs 원값 절단)
- **표현 결정: ψ 는 스칼라 ψ/180**(sin/cos 아님)
  - 이유: 2칸 제약. sin/cos 로 쓰면 침로가 2칸을 다 써서 SOG 가 빠짐 → 저자 결정 4(원값 ψ·SOG) 위반
  - 속도벡터(SOG·sinψ, SOG·cosψ)도 2칸·연속이지만 원값이 아니라 설계한 부호 → '원값 절단' 뜻이 흐려짐 → 채택 안 함(대안으로 §11-26)
- 대가(결과 전 기록)
  - ψ_n 은 ±180° 에서 불연속(179° 와 −179° 가 입력 양 끝). 청자 k/v 가 이 불연속을 배워야 함
    - 단 청자에게 필요한 특징(sin πψ_n, cos πψ_n)은 ψ_n 의 연속 주기함수라 실제 부담은 이 서술보다 작을 수 있음(추정). 아래 caveat 은 보수적으로 유지
    - 청자 용량 근거 `sup` 는 파트너 침로를 sin/cos 로 넣은 토큰 → 스칼라 ψ/180 형식에는 근거가 안 닿음(§0-3). 잠금 전 T2 형식 sup 는 안 돌림(돌리면 §0-3 공개 필수)
  - C2 의 z 는 코덱이 sinψ·cosψ 를 입력으로 학습해 침로에 대해 연속 사상일 수 있음(k=2 에 원형 임베딩 가능 — 추정) → **C2 > T2 가 나와도 '과제 인지 압축' 효과와 '연속 표현' 효과가 섞임**. 해석문에 병기(§7-12)
  - rot·cmd·목표·상황·위협 자리가 아예 없음 → A 판 T2 의 '결측 0 오독' 문제(A 판 §11-17) 해당 없음
- T2 는 k=2 비교 전용

### 3-11. 코덱 단계 버그와 튜닝의 구분 (결과 전 고정)

- **버그**(수정·재학습 허용): holdout L_P ≥ 1.0(상수 평균 예측 이하) 또는 테스트 9-10(항등 복호) 실패 또는 SHA 재현(9-19) 실패
  - 허용 범위: 코드 수정 → 같은 하이퍼파라미터로 재학습. 이력 §13 기록
- 그 밖의 재학습(충실도가 기대보다 나쁨 등)은 **금지**. k·λ·τ·epoch·구조 불변

---

## 4. 수신측·구조 (C 기준, A6 경로는 §4-12)

### 4-1. C 수신 토큰 (layout 'cz{W}', W = 슬롯 폭)

- token_ij = [relpos_ij 3 | slot_j W | own4_i 4 | msg_j×0 6]
  - relpos: 기존 그대로(vessel_gym_train.py:313-317 — 수신자 선체 방위 sin·cos + dist/COMM_R)
  - slot_j: 송신자 j 의 전송값. 팔별(§4-4 표). 순서 = 저자 전제 [relpos, z, 자기 상태, msg]
  - own4_i: 수신자 i 자기 운동(§2-5). 같은 행의 K 슬롯에 같은 값
  - msg×0: COMM_LATENT=0 → aggregate_batch 에서 ×0(networks.py:533-535). msg_dim 6 자리는 남음(키·폭 규약)
- 코드상 ext = [slot W, own4 4] → COMM_EXT_DIM = W + 4, relpos_dim = 3 + W + 4(networks.py:1071 식 그대로), 토큰 = relpos_dim + 6(networks.py:505)
- 패딩: 기존 규칙 그대로 `torch.where(pmask>0, ext·gmask, 0)`(vessel_gym_train.py:349) → own4 도 유효 슬롯에만. 파트너 0 이면 context 0(networks.py:542-543)
- **D·`_pair_core`·ext20 경로 안 씀.** 쌍별 관계는 k/v 가 relpos·slot·own4 로 학습
- **구조 한계 ② 해소**: 지금 수신자 상태는 query(q_in = [self_s 4, goal 2], vessel_gym_train.py:375, networks.py:1377)에만 들어가 attention 가중치만 바꿈. value 는 파트너 토큰만의 함수 → 쌍별 관계(상대속도·CPA·역할)를 집계 전에 못 만듦. own4 를 토큰에 넣으면 k(token_ij)·v(token_ij) 가 (i, j) 쌍의 함수가 됨
- 좌표계
  - relpos: 수신자 선체 좌표
  - z 가 담은 침로·C∞/T2 의 ψ, own4 의 ψ_i: 세계 좌표 → Δψ 는 청자가 곱셈 상호작용으로 배워야 함. 고정 좌표 변환 안 넣음(학습 청자 원칙)
  - C∞ 의 goal·threat 성분: 송신자 선수·선체 좌표 그대로(A 판 §4-3 과 같은 성격, 변환 없음)

### 4-2. `comm_gather` 분기 (vessel_gym_train.py:318-350)

```
if policy.relpos_dim > 3:
    L = net_mod.COMM_EXT_LAYOUT
    if L == 'v1':
        ext = vg.comm_pair_features(env, topi, PART_R)          # 기존 그대로 (off·arpa6·offr·보고만 팔)
        (기존 field-shuffle :322-343 그대로)
    elif L == 'v2p':
        ext = a6_ext(...)                                         # A6 — §4-12
    else:                                                        # 'cz{W}' — C 계열
        if ext_shuffle_gen is not None: raise RuntimeError(...)   # 행 단위 field-shuffle 은 own4 까지 섞음 → cz 에서 금지
        own = vg.own_motion4(env)                                # [E,N,4] = P[...,0:4] 와 같은 함수
        own_ent = own.unsqueeze(2).expand(E, N, Kc, 4)            # 수신자(행) 값을 K 슬롯에
        if net_mod.COMM_FIELDS == 'own':                         # P0: 슬롯 계산 안 함
            slot = zeros([E, N, Kc, W])
        else:
            P = own_payload(env, x, goal, sit)                   # [E,N,25] 송신자당 1회
            with torch.no_grad():
                mode = net_mod.COMM_PAYLOAD
                if   mode == 'codec':  s_all = quant(E(norm(P)))            # [E,N,k]   C2·C6·C12·R6
                elif mode == 'true':   s_all = P                            # [E,N,25]  C∞ (§2-2 식 그대로)
                elif mode == 'trunc2': s_all = quant(t2_code(env))          # [E,N,2]   T2 (§3-10)
                elif mode == 'recon':  s_all = decode_p25(D(quant(E(norm(P)))))   # [E,N,25] 평가 전용(C∞ 교차, §6)
                slot = s_all[b, topi]                                       # [E,N,Kc,W] 항목 단위 gather
                if net_mod.COMM_SLOT_SHUFFLE:                              # R6(학습) / slot-shuffle(평가)
                    if slot_shuffle_gen is None: raise RuntimeError(...)    # fail-closed (아래)
                    slot = sender_derange(slot, topi, pmask, slot_shuffle_gen)
        assert slot.shape[-1] == W
        ext = cat([slot, own_ent], -1)                           # [E,N,Kc,W+4]
    # 이후 그룹 마스크(:344-349)·where·cat(:350) 그대로. 그룹 표는 net_mod.COMM_EXT_GROUPS(레이아웃별)에서 읽음
```

- 슬롯 셔플은 **슬롯 열만** 옮김(relpos·own4 는 제자리) → 받는 z 만 다른 송신자 것이 됨
- 그룹 마스크가 지금 `cfg.COMM_EXT_GROUPS`(config 상수)를 읽음(:347) → `net_mod.COMM_EXT_GROUPS` 로 바꿈. v1 값은 같으므로 비트동일
- **같은 폭 팔 사이 전역 드리프트 가드**(09-28 구현 검토)
  - 문제: 분기가 networks 전역(COMM_EXT_LAYOUT·COMM_FIELDS·COMM_PAYLOAD·COMM_SLOT_SHUFFLE)을 읽음. cp0·cc6·cr6 는 셋 다 cz6(T 19)라 전역이 이전 ckpt 값으로 남아도 shape 에러가 안 남 → C6 가 R6·P0 로 조용히 바뀜
  - 규칙: `CNNPolicy.__init__` 이 그 시점 전역으로 `policy.comm_sig = (layout, fields, payload, codec SHA, slot_shuffle, kv_hidden, kv_depth)` 를 속성으로 박음. comm_gather 첫 줄에서 현재 전역과 비교 → 다르면 **RuntimeError**. v1 정책도 같은 검사(값은 기존 전역 그대로라 비트동일)
  - 테스트 9-16 에 cr6→cc6·cc6→cp0 순서 복원 사례 추가
- 수신자 자기 상태 own4 는 **rollout comm_gather 에서만 env 로 계산**됨(09-28 구현 검토가 코드로 확인): prelpos 에 cat → buf['prel'](vessel_gym_train.py:1041) → update 는 fprel(:1082·:1107) 재사용. `evaluate_actions` 는 env 를 안 받아 재계산 경로가 구조적으로 없음(networks.py:1372-1377 은 폭 검사만)
- **셔플 generator fail-closed 규칙**(A 판 규칙 그대로, 이름만 slot 으로)
  - 인자 이름을 기존 `ext_shuffle_gen`(field-shuffle, None = 안 섞음 관례 :322)과 분리한 `slot_shuffle_gen` 으로 둠
  - `net_mod.COMM_SLOT_SHUFFLE` 이 켜져 있는데 `slot_shuffle_gen is None` 이면 **RuntimeError**. 조용히 섞기 없는 C6 채널로 도는 사고(R6 가 C6 가 됨 → G-C3 오염, '4번째 미러 사고' 계열) 방지
  - 호출부와 전용 generator (전부 CPU `torch.Generator`, 전역 RNG 불간섭)

| 호출부 | file:line @7013a3a | generator |
|---|---|---|
| rollout | vessel_gym_train.py:1000-1002 | G_train = seed + 200003. 재개 시 (seed, resume_at) 로 재시드, 스냅샷 `comm_shuffle_seed` 기록 |
| last-value | :1054-1055 | G_train 같은 객체 이어 씀 |
| 텔레메트리 base | :465 | G_tel = 호출마다 새로 seed + 300007 + update 번호 → 한 텔레메트리 안 모든 호출이 같은 순열(차이 = 끈 그룹만). 텔레메트리 on/off 가 G_train 을 안 건드림 |
| 텔레메트리 zero·shuf | :546, :548 | G_tel 규칙 같음 |
| 텔레메트리 그룹 | :565 | G_tel 규칙 같음 |
| eval | eval/eval_ckpt.py:129-130 | G_eval = eval seed + 200003, 결정마다 이어 씀 |
| diag | eval/diag_ckpt.py:111 | G_diag = seed + 200003 |
| 혼합 함대 | eval/eval_mixed.py:148 | G_eval 규칙. 이번 계획에 혼합 평가 없음 |
| 미러 검증기 | verify/_verify_comm_mirror.py:54·:62 | R6 케이스는 전용 generator 를 넘김(테스트 9-39). 안 넘기면 fail-closed 에러가 나는지도 같이 확인 |
| **미연결 호출부**(generator 안 넘김) | eval/corridor_run.py:118 · eval/measure_regimes.py:99 · astar_fig9/eval_astar_global.py:452 · verify/test_ckpt_compat.py:62·:69 · verify/test_comm_ext.py(:200 등 9곳) · `_archive/2026-09/` 3곳 | 이번 계획에 안 씀. **R6 ckpt 를 넣으면 fail-closed RuntimeError 가 나는 것이 의도**(섞기 없는 C6 로 조용히 도는 것보다 나음). test_ckpt_compat 에는 cc6 합성 ckpt(정상 동작)·cr6 합성 ckpt(RuntimeError 기대) 사례를 추가(테스트 9-41) |

### 4-3. 청자 k/v 용량 (결과 전 고정, 모든 통신 팔 동일)

- **결정: k_proj·v_proj = 은닉 256 × 3층 MLP(ReLU)**. 신규 통신 8팔(cp0·cc2·cc6·cc12·ccinf·cr6·ct2·ca6) **전부 같은 구조**. 재사용 off·arpa6·offr·보고만 팔은 기존 1×64 그대로(체크포인트 구조, off·offr 는 attention 미사용)
  - k: T → 256 → 256 → 256 → 32(d_attn). v: T → 256 → 256 → 256 → 6. v 마지막 층 ×0.1·bias 0(기존 소진폭 규약 networks.py:514-516 을 `v_proj[-1]` 로)
  - q_proj(6→32 선형)·d_attn 32·softmax 집계(aggregate_batch :527-543)·msg_encoder 구조 불변
  - 초기화 = PyTorch 기본(기존 k/v 와 같음). 활성 ReLU(기존과 같음). 재튜닝 없음
- 근거 (§0-3 `sup.txt`, off s43·s44 쌍 3.05M 학습 → ons6_s44 쌍 0.51M 평가, 입력 = C∞ 토큰의 운동 부분 10D, 정답 라벨 지도학습)

| k/v | 표본 | 역할 정확도(참 역할 있는 쌍) ≤56 / 56–150 / 150–300 m | dcpa_risk MAE ≤56 / 56–150 / 150–300 m (상수 예측) |
|---|---|---|---|
| 1×64 | 2e5 | 0.0 / 0.0 / 0.0% | 0.121 / 0.096 / 0.078 (0.124 / 0.075 / 0.065) |
| 1×64 | 2e6 | 40.2 / 16.3 / 0.9% | 0.101 / 0.089 / 0.082 |
| 1×64 | 2e7 | 67.1 / 64.0 / 28.4% | 0.099 / 0.086 / 0.080 |
| 3×256 | 2e5 | 50.2 / 23.6 / 1.3% | 0.088 / 0.083 / 0.076 |
| 3×256 | 2e6 | 77.7 / 77.3 / 75.8% | 0.041 / 0.027 / 0.031 |
| 3×256 | 2e7 | 91.7 / 94.6 / 92.6% | 0.019 / 0.013 / 0.015 |

  - 1×64(현 k/v)는 정답 라벨 2e7 개로도 역할 28–67%, 56 m 밖 dcpa MAE 가 상수 예측보다 나쁨 → 청자 학습의 **상한부터 막힘**. 이 크기로 C 가 지면 '압축·정보' 탓인지 '청자 용량' 탓인지 못 가름
  - 3×256 은 모든 표본 크기에서 1×64 보다 나음(2e5 에서도) → 표본 효율 면에서 키워 손해 본 증거 없음(지도학습 한정)
  - **측정한 크기는 이 둘뿐.** 2×128 등 중간값은 측정 안 함 → 측정한 쪽 중 상한에 닿는 3×256 을 택함. 결과 보고 안 바꿈
  - 보조(확인한 서지): Lin et al. 2021 의 청자도 받은 메시지를 3층 MLP 로 처리(§12)
  - 한계: sup 는 정답 라벨 지도학습. RL 신호는 훨씬 약하고, 집계 뒤 6D value 병목·7M 결정 예산을 거침 → 3×256 이어도 RL 로 같은 정확도에 닿는다는 보장 없음(추정). 학습량 비교는 §7-11
  - 근거 범위: sup 토큰은 파트너 침로 sin/cos·운동 부분만 → z 토큰(C(k))·T2 스칼라 ψ/180·C∞ 의 goal·sit·threat 부분에는 직접 근거 아님. z 토큰 상한은 코덱 동결 뒤 보고 전용 probe 로만 봄(§6, 설계 불변)
- 대가·위험 (결과 전 기록)
  - 파라미터(attn + msg_encoder, 계산값 `kv_params.py`): 1×64 이면 5,452–10,732 → 3×256 이면 **282,060–300,012**(§4-4 표). 배치 X 정책 합계 약 29.4만(YUGIOH 287,571 에 EXT k/v 차이를 더한 계산값 — 실측은 구현 때 §4-4 에 채움) → 신규 통신 팔 정책이 약 2배
  - off 는 통신 경로가 없으니 이 차이 전체가 통신 경로에 있음 → 확증 비교 문장에 '통신 경로(청자 용량 포함)' 병기, 용량 효과는 **P0(같은 용량, z 없음)** 로 통제(§5-4)
  - **깊이 변경 = 키 결정자**(attn.k_proj.{4,6}·v_proj.{4,6} 새 키) → trunk 분기 때 Adam 파라미터 개수가 달라져 `opt.load_state_dict` 가 ValueError(파라미터 그룹 크기 검사) → 이름 기준 재배치 필요(§4-5)
  - 기본 초기화 + 3층 + v ×0.1 → 초기 value 출력이 1×64 보다 작음(층마다 2차 모멘트 약 1/6 — 계산 추정). Adam 이 step 크기를 정규화하므로 06-12 류 동결은 아닐 것(추정). 스모크(§9-26)에서 k/v grad 가 0 아님만 확인, 결과 보고 초기화 안 바꿈
  - grad clip: `CLIP_PER_MODULE=1` 에서 attn 은 '나머지' 묶음(구성 vessel_gym_train.py:917-933 중 :931-933, 적용 :1212-1215)에 들어가 0.5 로 잘림. 3×256 이면 그 묶음 norm 이 커질 수 있음 — 규칙 그대로(재튜닝 없음)
    - 그 묶음의 다른 구성원(보조 디코더)은 AUX_LOSS_SCALE=0 이라 grad 0, msg_encoder 는 attention 경로 밖이라 grad None → 사실상 attn 만 남음 → 3×256 이 세 망(msg·ctr·critic)의 clip 계수를 누르지 않음(09-28 사전등록 검토가 코드로 확인)
  - RL 안정성·학습 시간 증가 (미확인)
  - A6 도 같은 3×256 → A 판(1×64)과 다름. 대신 A6 vs C6 에서 청자 용량이 같아짐(§5-4)

### 4-4. 팔별 토큰 폭·shape·파라미터 (계산값) + 공정성

| 팔 | layout | 슬롯 W | 슬롯 내용 | relpos_dim | 토큰 T | k/v | attn+msg_encoder 파라미터 | 첫 층 초기화 bound 1/√T |
|---|---|---|---|---|---|---|---|---|
| cp0 (P0) | cz6 | 6 | 0 (fields own) | 13 | 19 | 3×256 | 284,236 | 0.229 |
| cc2 (C2) | cz2 | 2 | q(E₂) | 9 | 15 | 3×256 | 282,060 | 0.258 |
| cc6 (C6) | cz6 | 6 | q(E₆) | 13 | 19 | 3×256 | 284,236 | 0.229 |
| cc12 (C12) | cz12 | 12 | q(E₁₂) | 19 | 25 | 3×256 | 287,500 | 0.200 |
| ccinf (C∞) | cz25 | 25 | P (float32) | 32 | 38 | 3×256 | 294,572 | 0.162 |
| cr6 (R6) | cz6 | 6 | 다른 송신자의 q(E₆) | 13 | 19 | 3×256 | 284,236 | 0.229 |
| ct2 (T2) | cz2 | 2 | [ψ_n, v_n] 격자값 | 9 | 15 | 3×256 | 282,060 | 0.258 |
| ca6 (A6) | v2p | – | ext 39(D 복호 → `_pair_core` 20 + 덧붙임 19) | 42 | 48 | 3×256 | 300,012 | 0.144 |
| off·offr·arpa6(참고) | v1 | – | ext 20 | 23 | 29 | 1×64 | 7,692 (off·offr 미사용) | 0.186 |

- 파라미터 = k + v + q_proj 224 + msg_encoder. msg_encoder 는 attention 경로에서 안 쓰임(shape 만 폭 따라 바뀜). 표 값·첫 층 bound 는 09-28 구현·사전등록 검토가 `kv_params.py` 재실행으로 일치 확인(실측은 구현 때)
- **짝 비교별 shape 일치**
  - C6 vs P0: **동일**(T 19, 같은 seed → 초기값 텐서 비트 동일, 테스트 9-14) → 차이 = z 값뿐
  - C6 vs R6: **동일** → 차이 = 송신자–z 짝
  - C2 vs T2: **동일**(T 15) → 차이 = 슬롯 사상
  - C6 vs A6: 다름(19 vs 48, 파라미터 +15,776 = 5.6%) → 읽기 진단에 폭·토큰 내용 차이가 섞임(§5-4)
  - C(k) vs off: off 는 attention 미사용 → 통신 경로 전체가 차이(구조상 불가피, P0 로 통제)
  - C2·C12 vs P0: 폭 다름(15·25 vs 19) → '귀속 미분리(폭 포함)'로만
  - C(k) vs C∞: 폭 다름 → R(k) 서술용만
- **P0 슬롯 폭 = 6 으로 정한 이유**: 확증·귀속 비교(C6 vs P0)에서 shape·초기값까지 같게. 슬롯을 없애면(T 13) 첫 층 fan_in·초기화 bound 가 달라져 'z 뿐' 전제가 흐려짐. 0 열은 입력이 0 이라 가중치 gradient 0 → 학습 내내 기여 0(초기값 그대로)
- **네이티브 폭(팔마다 다름) vs 공통 폭 패딩(모두 T 48 등)**: 패딩을 택하지 않음
  - 패딩하면 shape·파라미터 수는 모두 같아지나, 실제 입력 폭 대비 초기화 bound 가 작아져(1/√48) 좁은 팔일수록 실효 초기 크기가 작아짐 → 팔마다 다른 실효 초기화가 다시 생김
  - 판정에 쓰는 짝(C6–P0·C6–R6·C2–T2)은 네이티브 폭에서 이미 shape 동일
  - 폭이 다른 비교는 부 비교·서술로만 둠
- 초기 스케일 차이 외의 공정성 문제: C∞ 의 슬롯 25칸 중 sit one-hot·threat 는 [P]·송신자 좌표 성분 → C∞ 는 상한(ORACLE-ref)으로만 씀(§8-3)

### 4-5. trunk 분기 재초기화 (A 판 규칙을 팔별 shape 로 일반화)

- 바뀌는 키(trunk = v1·1×64, 신규 = 토큰 T·3×256)

| 키 | trunk | 신규 |
|---|---|---|
| attn.k_proj.0.weight / .bias | (64, 29) / (64) | (256, T) / (256) |
| attn.k_proj.2.weight / .bias | (32, 64) / (32) | (256, 256) / (256) |
| attn.k_proj.4.* | 없음 | (256, 256) / (256) |
| attn.k_proj.6.* | 없음 | (32, 256) / (32) |
| attn.v_proj.0.* | (64, 29) / (64) | (256, T) / (256) |
| attn.v_proj.2.* | (6, 64) / (6) | (256, 256) / (256) |
| attn.v_proj.4.* | 없음 | (256, 256) / (256) |
| attn.v_proj.6.* | 없음 | (6, 256) / (6) ×0.1·bias 0 |
| msg_encoder.0.weight | (32, 29) | (32, T) |
| msg_encoder.0.bias·.2.* | (32)·(6,32)·(6) | 같음 |
| attn.q_proj.* | (32, 6)·(32) | 같음 |

  - T: cc2·ct2 15 / cp0·cc6·cr6 19 / cc12 25 / ccinf 38 / ca6 48
- **trunk 의 attn·msg_encoder 가 초기값 그대로인 근거(코드 확인)**
  - trunk 구간은 comm_active False(:977) → rollout om = 0(:1004, last value :1057)
  - update 는 else 분기 `ctr_actor.get_logprob_entropy` + `critic` 만(:1139-1151) → attn·msg_encoder 는 그래프 밖 → `.grad` None
  - Adam 은 grad None 파라미터에 state 를 안 만들고 갱신도 안 함 → 값 불변. clip 은 grad None 을 건너뜀
  - torch 1.9 실측(스크래치패드 `adam_shape_probe.py`): 모양이 바뀐 파라미터에 state 가 없으면 load 통과 / 있으면 첫 step 에서 크기 불일치 RuntimeError
  - 실제 trunk 3개의 `optimizer_state_dict['state']` 에 해당 파라미터가 없는지 학습 전 Windows 에서 확인(테스트 9-14)
- **발동 조건(09-28 추가)**: 분기점 재개에서 이번 런의 (layout, kv_hidden, kv_depth) 가 trunk 스냅샷과 **다를 때만** 아래 재초기화·Adam 재배치를 탐. 같으면(offr·arpa6 형 = v1·(64, 1)) **기존 경로 그대로**(:713 strict 로드 → :722-724 `opt.load_state_dict`) — 새 코드 줄을 한 줄도 안 지나감. 7013a3a 와 비트동일인지는 테스트 9-33 이 확인
- **재초기화 규칙** (`load_state_dict`(:713) 앞)
  - **화이트리스트 모듈 = attn.k_proj·attn.v_proj·msg_encoder** — 이 모듈의 키는 shape 이 같아도 **통째로** 이번 런 새 초기값(규칙 단순화, trunk 값도 어차피 초기값)
  - 새 값 = 이번 런이 `CNNPolicy` 를 만들 때(:641, `torch.manual_seed(seed)` :627 뒤)의 초기값. 화이트리스트 밖 키 = trunk 값(q_proj 포함)
  - 허용 조건(하나라도 어기면 SystemExit)
    - ① 분기점 `resume_at == comm_on_at > 0`(:689)
    - ② trunk `comm_active` False(:697)
    - ③ trunk 스냅샷 comm_ext=1·layout v1·kv (64, 1)(스냅샷 없으면 키 스니핑)
    - ④ 키 집합·shape 차이가 화이트리스트 모듈 안뿐
    - ⑤ trunk Adam state 에 화이트리스트 파라미터 없음
    - ⑥ 화이트리스트 밖 파라미터의 trunk Adam state shape == 새 파라미터 shape
  - **방식은 병합 state_dict 의 strict 로드만**: 새 policy 의 state_dict 를 복사하고 화이트리스트 밖 키를 trunk 값으로 덮은 dict 를 `policy.load_state_dict(strict=True)`. **모듈 교체 금지**(:642 Adam 이 옛 파라미터 객체를 잡고 있어 그 층이 조용히 학습 안 됨). 테스트 9-14 가 Adam param_groups 객체 id == policy.parameters() 확인
- **Adam 이름 기준 재배치** (깊이 변경으로 파라미터 개수가 달라서 필요 — `opt.load_state_dict`(:722-724)는 그룹 크기가 다르면 ValueError)
  - trunk 파라미터 이름 순서 = trunk 스냅샷 설정(v1·1×64 등)으로 만든 CNNPolicy 골격의 `named_parameters()` 순서. 골격은 `torch.random.fork_rng`(CPU·CUDA) 안에서 만들고 버림 → 전역 난수 불변
  - **골격용 전역 저장·복원**(09-28 구현 검토): 골격을 만들려면 networks·config 전역(COMM_EXT_LAYOUT·COMM_EXT_DIM·COMM_EXT_GROUPS·COMM_FIELDS·COMM_PAYLOAD·COMM_CODEC·COMM_SLOT_SHUFFLE·COMM_EXT_MLP_HIDDEN·COMM_EXT_MLP_DEPTH)을 trunk 값으로 바꿔야 함 → 컨텍스트 관리자 하나가 **바꾸기 전 값을 전부 저장 → trunk 값 설정 → 골격 생성 → finally 에서 전부 복원**. 복원 뒤 전역 == 이번 런 값, `policy.comm_sig` 검사(§4-2) 통과를 assert
    - 대안(trunk state_dict 의 화이트리스트 밖 키로 이름 순서를 직접 만듦)은 채택 안 함: state_dict 에 버퍼(state_recon run_mean 등)가 섞이고, 공유 인코더(SHARED_ENCODER='all')·MOE_SHARED 의 중복 제거가 로드 뒤 저장소 동일성에 의존 → `named_parameters()` 순서와 어긋날 위험
  - 공유 인코더(SHARED_ENCODER='all')는 `named_parameters()` 의 중복 제거 규칙이 양쪽 같음(같은 스냅샷 설정)
  - 새 state = 화이트리스트 밖이고 이름이 같은 파라미터마다 trunk 의 exp_avg·exp_avg_sq·step 을 새 인덱스로 옮김. 화이트리스트는 state 없음
  - param_groups 하이퍼파라미터(lr·betas·eps·weight_decay·amsgrad) = trunk 값, params 목록 = 새 인덱스 → `opt.load_state_dict(재배치본)`
  - 기록: 스냅샷 `branch_adam_remap` = {moved, trunk_state_n, whitelist_state: 0}
  - 크래시 재개(분기 뒤)는 체크포인트가 이미 새 구조 → 기존 `opt.load_state_dict` 그대로
- 같은 seed 면 같은 shape 팔끼리 화이트리스트 초기값 동일(생성 전 난수 소비가 팔과 무관, §3-8 난수 격리)
- 학습 난수열: GPU 행동 표본·randperm 은 CUDA RNG, 초기화는 CPU RNG(:641) → 폭·깊이 차이가 GPU 난수열을 안 바꿈(코드 읽기로 확인, 실행 확인은 미확인). 워밍업 1200(:794-804)은 om 0
- 기록: 스냅샷 `branch_reinit` = {키: [old shape 또는 None, new shape]}, `branch_reinit_sha256`(화이트리스트 새 텐서 내용 SHA), 로그 `[branch] reinit …`
- 크래시 재개: 분기 키 상속(vessel_gym_train.py:711-712)에 `branch_reinit`·`branch_reinit_sha256`·`branch_adam_remap` 추가
- **지금 코드가 막는 곳**(셋 다 고침): 학습기 strict 로드(:713) RuntimeError, `restore_comm_ext` 폭·layout 검사 SystemExit(ckpt_io.py:254-263), check_branch 는 layout·kv 를 안 봐서 그대로 통과
- **check_branch 호환** (verify/check_branch.py)
  - 묶음 키 비교(:128)의 comm_ext 는 전 팔 1 → 통과. seed·msg_dim 6·branch_at·dyn·obst·crossing·sim 도 같음
  - 추가: layout·kv 를 묶음 표시·검사에 포함. layout·kv 가 trunk 와 다른 갈래는 `branch_reinit` 필수·화이트리스트 모듈 키만(없으면 FAIL)
  - **'통신 팔끼리 kv 동일' 검사 대상 = layout ∈ {cz2, cz6, cz12, cz25, v2p} 인 갈래만**. 재사용 arpa6(v1·64×1)·off·offr(v1)는 대상 밖 → 이 규칙으로 FAIL 나지 않음. 대상 안에서 kv 가 다르면 FAIL
  - trunk CSV 행 일치 검사(검사 5)는 그대로
  - 재사용 x_off·x_arpa6 의 곡선 CSV 는 새 OUT 에 없으면 검사 5 가 note 로 건너뜀 → **배치 X OUT 에서 `x_off_s*.csv`·`x_off_s*_aux.csv`·`x_arpa6_s*.csv`·`x_arpa6_s*_aux.csv`·`x_arpa6_s*_comm.csv` 를 새 OUT 으로 복사**(복사 전후 sha256 기록, 원본 쓰기 금지) → 재사용 팔도 0~branch_at 곡선 대조를 받음. off·arpa6 은 학습 대상이 아니라 train_one 이 이 사본을 지우지 않음(§5-2)

### 4-6. 기존 latent·aux 처리

- COMM_LATENT=0 → `aggregate_batch` 에서 msg 항 ×0(networks.py:533-534). msg_actor 는 계산되지만 tanh 유계라 0×유한 = 0 → grad 정확히 0
- AUX_LOSS_SCALE=0 → aux 전체 ×0, 라벨 버퍼·state_recon 생략(:984-991, :1135-1138)
- msg_actor·StateReconDecoder·consumer_decoder·msg_encoder·4디코더는 무조건 생성 키 → 유지·미사용(제거 금지 규약)
- MSG_TOKEN_GAIN 8 은 msg 항에만 걸림 → 효과 0. 재튜닝 금지
- 'latent' 라는 말은 이 문서에서 **동결 코덱 코드 z** 만 가리킴(창발 latent 채널은 꺼져 있음)

### 4-7. 토글·config·networks 전역·스냅샷·재개 검사·변경 목록

- config.py(:437-455 절 뒤, 새 env 키는 config 에만)

| env | 상수 | 기본 | 의미 |
|---|---|---|---|
| VESSEL_COMM_EXT_LAYOUT | COMM_EXT_LAYOUT | 'v1' | 지금 상수(:443) → env 구동. 'v2p'(A6) · 'cz2'·'cz6'·'cz12'·'cz25'(C 계열). **shape 결정자** |
| (파생) | COMM_EXT_DIM, COMM_EXT_GROUPS | v1: 20 / :446 표 | v2p: 39, 그룹 state(0,8)·role(8,18)·intent(18,20)·goal(20,22)·sit(22,27)·threat(27,39) / cz{W}: W+4, 그룹 slot(0,W)·own(W,W+4). cz25 는 보조 그룹 p_motion(0,4)·p_cmd(4,6)·p_goal(6,8)·p_sit(8,13)·p_threat(13,25) 추가 |
| VESSEL_COMM_FIELDS | COMM_FIELDS | 'latent' | v1: 기존 {latent, state, intent} / v2p: {full} / cz: {own, full}. own = P0, full = slot+own |
| VESSEL_COMM_PAYLOAD | COMM_PAYLOAD | '' | v2p: codec / cz full: codec \| true \| trunc2. 'recon'·v2p 'true' 는 평가 CLI 전용 |
| VESSEL_COMM_CODEC | COMM_CODEC_PATH | '' | 코덱 파일 경로 문자열. '' = 끔 |
| VESSEL_COMM_CODEC_SHA | COMM_CODEC_SHA | '' | 기대 내용 SHA. 다르면 중단 |
| VESSEL_COMM_CODEC_BITS | COMM_CODEC_BITS | 8 | 이번 주기 8 고정(T2 격자도 같음) |
| VESSEL_COMM_SLOT_SHUFFLE | COMM_SLOT_SHUFFLE | 0 | 1 = R6(학습). 평가 slot-/z-shuffle 은 CLI |
| VESSEL_COMM_KV_HIDDEN | COMM_KV_HIDDEN | 64 | 지금 networks.py:109 상수 → config. **shape 결정자**. networks 전역 이름은 기존 `COMM_EXT_MLP_HIDDEN` 그대로 둠(test_comm_ext.py:100-103 이 이 이름을 읽음 → 별칭 없이 한 이름만, 드리프트 없음) |
| VESSEL_COMM_KV_DEPTH | COMM_KV_DEPTH | 1 | 은닉층 수. **키 결정자**(1 = 지금 구조, 생성 코드 경로 비트동일). networks 전역 = 새 `COMM_EXT_MLP_DEPTH` |
| VESSEL_ARM_TAG | ARM_TAG | '' | 표시 전용(offr 만 'offr'). 구조·학습 영향 없음 |

- **kv 두 키의 뜻 범위**: comm_ext=1 일 때만 뜻이 있음. EXT=0 ckpt 는 선형 attn(키 `attn.k_proj.weight`, networks.py:518-519)이라 kv 값은 무시하고 스냅샷에 legacy (64, 1) 로 적힘

- assert(모르는 값이 조용히 0 으로 가는 사고 방지)
  - v2p·cz → COMM_EXT=1·USE_ATTENTION=1·COMM_LATENT=0·AUX_LOSS_SCALE=0
  - cz{W}: fields own → W == 6·payload·codec 키 금지 / fields full → payload codec(코덱 k == W) · true(W == 25) · trunc2(W == 2) 중 하나 필수
  - v2p: fields full·payload codec(k=6) 필수
  - payload codec → 경로·SHA 필수 / SLOT_SHUFFLE=1 → layout cz6·payload codec / trunc2 → BITS 8 / v1 에서 payload·codec·shuffle 키 금지
- **팔 env 는 `comm_variant_env`(run_repro.sh:344-355)로만** 줌. LAYOUT·PAYLOAD·CODEC·KV 를 배치 전역으로 export 하면 assert(LATENT=0·AUX=0 요구) 때문에 trunk·eval 프로세스가 import 때 죽고, KV 는 재사용 팔 복원을 흐림
- **팔 env 블록 (결과 전 고정, 이 글자 그대로 구현)** — 기존 규칙 '전부 명시, 바깥 셸 값이 팔을 조용히 바꾸지 못하게'(run_repro.sh:344-347)를 새 키까지 넓힘. assert 는 빈 문자열 '' 을 '없음'으로 읽음

```
comm_variant_env() {
  # 기본 줄: 기존 3개 + 새 키 전부 명시 (v1·코덱 없음·1×64)
  export VESSEL_COMM_FIELDS=latent VESSEL_COMM_LATENT=1.0 VESSEL_AUX_LOSS_SCALE=1.0
  export VESSEL_COMM_EXT_LAYOUT=v1 VESSEL_COMM_PAYLOAD= VESSEL_COMM_CODEC= VESSEL_COMM_CODEC_SHA= \
         VESSEL_COMM_CODEC_BITS=8 VESSEL_COMM_SLOT_SHUFFLE=0 VESSEL_COMM_KV_HIDDEN=64 VESSEL_COMM_KV_DEPTH=1 VESSEL_ARM_TAG=
  unset VESSEL_PARTNER_RANGE
  _C="VESSEL_COMM_LATENT=0.0 VESSEL_AUX_LOSS_SCALE=0.0 VESSEL_COMM_KV_HIDDEN=256 VESSEL_COMM_KV_DEPTH=3"   # 신규 통신 8팔 공통
  case "${1:-}" in
    (arpa6·onl6·ons6·oni6·on6a0 — 기존 줄 그대로)
    offr)  export VESSEL_ARM_TAG=offr ;;
    cp0)   export $_C VESSEL_COMM_EXT_LAYOUT=cz6  VESSEL_COMM_FIELDS=own ;;
    cc2)   export $_C VESSEL_COMM_EXT_LAYOUT=cz2  VESSEL_COMM_FIELDS=full VESSEL_COMM_PAYLOAD=codec \
                  VESSEL_COMM_CODEC=comm_codecs/tac_p25_k2.pt  VESSEL_COMM_CODEC_SHA=<§13 k=2 내용 SHA> ;;
    cc6)   export $_C VESSEL_COMM_EXT_LAYOUT=cz6  VESSEL_COMM_FIELDS=full VESSEL_COMM_PAYLOAD=codec \
                  VESSEL_COMM_CODEC=comm_codecs/tac_p25_k6.pt  VESSEL_COMM_CODEC_SHA=<§13 k=6 내용 SHA> ;;
    cc12)  export $_C VESSEL_COMM_EXT_LAYOUT=cz12 VESSEL_COMM_FIELDS=full VESSEL_COMM_PAYLOAD=codec \
                  VESSEL_COMM_CODEC=comm_codecs/tac_p25_k12.pt VESSEL_COMM_CODEC_SHA=<§13 k=12 내용 SHA> ;;
    ccinf) export $_C VESSEL_COMM_EXT_LAYOUT=cz25 VESSEL_COMM_FIELDS=full VESSEL_COMM_PAYLOAD=true ;;
    cr6)   export $_C VESSEL_COMM_EXT_LAYOUT=cz6  VESSEL_COMM_FIELDS=full VESSEL_COMM_PAYLOAD=codec \
                  VESSEL_COMM_CODEC=comm_codecs/tac_p25_k6.pt  VESSEL_COMM_CODEC_SHA=<§13 k=6 내용 SHA> VESSEL_COMM_SLOT_SHUFFLE=1 ;;
    ct2)   export $_C VESSEL_COMM_EXT_LAYOUT=cz2  VESSEL_COMM_FIELDS=full VESSEL_COMM_PAYLOAD=trunc2 ;;
    ca6)   export $_C VESSEL_COMM_EXT_LAYOUT=v2p  VESSEL_COMM_FIELDS=full VESSEL_COMM_PAYLOAD=codec \
                  VESSEL_COMM_CODEC=comm_codecs/tac_p25_k6.pt  VESSEL_COMM_CODEC_SHA=<§13 k=6 내용 SHA> ;;
  esac
}
is_ext_arm()  : 기존 4팔 + cp0 cc2 cc6 cc12 ccinf cr6 ct2 ca6
arm_spec()    : offr → "OFF 6", 신규 통신 8팔 → "ON 6" (msg_dim 6 = trunk 와 같은 dim 묶음)
eval_arm()    : off·offr → OFF / rand → RANDOM / 나머지 → ON   (새 함수 — 지금 eval 은 off 만 OFF 로 보냄, run_repro.sh:502-509)
```

  - 코덱 경로는 `Python/` 기준 상대 경로 → config 가 `os.path.dirname(__file__)` 기준으로 풂. 스냅샷에는 basename·내용 SHA 만
  - `<§13 … SHA>` 자리는 §0-0 5단계에서만 채움. 채운 뒤 바뀌면 preflight 지문(run_repro.sh 글자)이 잡음
  - **팔 이름 → 평가 arm 표**: off·offr → OFF, rand → RANDOM, arpa6·onl6·ons6·oni6·신규 통신 8팔 → ON. 지금 코드대로면 offr 가 ON 으로 평가돼 restore_policy 가 arm 불일치로 중단함 → `eval_arm()` 로 고치고 dry-run 테스트(9-40)에 넣음
- **교차평가 코덱·payload 는 CLI 인자**(`--codec <경로> --codec_sha <SHA> --payload_override <mode> --slot_shuffle` + `allow_codec_override`)로만. VESSEL_* env 로 주지 않음(CLAUDE.md §8 스크립트 env 세팅 금지)
- networks 전역(:99-109 옆): COMM_EXT_LAYOUT, COMM_EXT_DIM(기존), COMM_EXT_GROUPS, COMM_FIELDS, COMM_PAYLOAD, COMM_CODEC(객체), COMM_CODEC_BITS, COMM_SLOT_SHUFFLE, COMM_EXT_MLP_HIDDEN(기존 :109, config COMM_KV_HIDDEN 값), COMM_EXT_MLP_DEPTH(새, config COMM_KV_DEPTH 값). GroundedAttention(:501) 에 `mlp_depth` 인자, 생성 호출(:1081-1082)이 전역을 읽음
- ckpt_io
  - `snapshot_config`(:91-98 뒤) 키 추가만: comm_payload, comm_codec_name, comm_codec_sha256, comm_codec_k, comm_codec_bits, comm_codec_lambda, comm_codec_data_sha256, comm_slot_shuffle, comm_shuffle_seed, comm_kv_hidden, comm_kv_depth, arm_tag, branch_reinit, branch_reinit_sha256, branch_adam_remap. comm_ext_layout·comm_ext_dim 은 기존 키(:93-94)에 새 값. 골든은 스냅샷 키 추가를 허용(test_golden.py:143-147) → `--regen` 필요 없음
  - `restore_comm_ext`(:243-296)
    - 폭 검사(:254-260)·layout 검사(:261-263)를 **스냅샷 layout 기준 표**로. 스냅샷 layout 으로 net.COMM_EXT_LAYOUT·DIM·GROUPS 설정(CNNPolicy() 전)
    - **kv 복원**: 스냅샷 comm_kv_hidden·depth → 없으면 키 스니핑: `attn.k_proj.weight` 가 있으면 선형 attn(EXT=0, kv 뜻 없음) / 없으면 **깊이 = attn.k_proj 안 Linear 개수 − 1**(1×64 는 k_proj.{0,2} = Linear 2개 → 깊이 1, networks.py:512), 폭 = k_proj.0.weight 행 수 → 배치 X ckpt 는 (64, 1). CNNPolicy() 전에 설정(순서 규약 §8). 배치 X ckpt 에는 comm_kv_* 스냅샷 키가 없어 G-C0 2단계의 x_arpa6 복원이 이 스니핑을 탐 → 테스트 9-16 이 x_arpa6 형 합성 ckpt 로 (64, 1) 복원 확인
    - 그룹 이름 하드코딩 {'state','role','intent'}(:276-279) → 레이아웃 그룹 표 기준으로(slot0·own0·p_*0·goal0·sit0·threat0 절제 허용)
    - 스냅샷에 codec SHA 가 있는데 blob 없음·SHA 불일치 → 중단. 키 없음 → v1·payload 없음(legacy)
    - **항상 전부 다시 설정**(:248 원칙): blob 없는 ckpt 면 `COMM_CODEC=None`, `COMM_PAYLOAD=''`, `COMM_SLOT_SHUFFLE=0` 으로 명시 리셋, kv 도 매번 설정 → 한 프로세스에서 코덱·3×256 ckpt 다음 비코덱·1×64 ckpt 를 열어도 값이 안 샘(테스트 9-16)
  - `env_lines`(:630-633, :682-691)에 새 키 줄
- 재개 검사 `_cur_comm`(vessel_gym_train.py:684-686)에 comm_ext_layout·comm_payload·comm_codec_sha256·comm_codec_bits·comm_slot_shuffle·comm_kv_hidden·comm_kv_depth·arm_tag 추가, legacy 값('v1', None, None, None, 0, 64, 1, '')도(:687). 분기점은 comm_ext 만 봄(:689-690) + §4-5 재초기화 규칙
- **그룹 이름 확장 때 같이 고칠 곳**: config assert(:448-452, 'own'·'full' 허용), `COMM_FIELDS_TO_GROUPS`(:447, 레이아웃별 표로), 텔레메트리 그룹 루프(vessel_gym_train.py:558-561), run_repro ablate 의 `--comm_groups` 목록(:604-613)
- `compute_own_threat` 의 `_THREAT_ANG` 캐시(vessel_gym_train.py:249-251)에 device 비교 추가(버그 수정 — CPU→GPU 순서 호출 시 에러). 값 불변
- run_repro.sh
  - arm_spec(:328-341)·`comm_variant_env`(:344-355)·is_ext_arm(:356)·이름 목록(:379·:486·:504·:632)·ablate(:604-613)에 신규 9팔, 팔 spec 에 코덱 SHA·kv 핀(위 env 블록). offr 는 arm_spec 'OFF 6' + `VESSEL_ARM_TAG=offr`
  - eval 대상 목록 제한: 지금 eval 은 접두어 x_ 의 기존 팔 전부를 '있으면' 평가(:502-509) → 새 env `VESSEL_EVAL_ARMS`(기본 = 기존 목록 → 옛 동작 비트동일)로 이번 11팔만. 팔 → 평가 arm 은 `eval_arm()`(위 표)
  - trunk 없으면 중단: **기존 가드 `VESSEL_REQUIRE_TRUNK=1`(run_repro.sh:406-408)을 배치 env 로 켬**(새로 만들 것 없음)
  - **재사용 팔 존재 검사(새)**: 배치 시작 때 `$CK/x_off_s{43,44,45}.pt`·`x_arpa6_s{43,44,45}.pt`·`x_trunk_d6_s{43,44,45}.pt` 9개가 모두 있어야 진행(없으면 중단). 지금 짝 검사는 offr 가 목록에 있으면 has_off=1 이라 x_off 존재를 안 봄(§5-2)
  - 스모크 팔 목록 하드코딩(:453) → cz·v2p end-to-end 스모크 모드 추가(§9-26)
  - **dry-run 모드(새)**: 7013a3a 에 dry-run 코드 없음(`grep dry` 0건) → `VESSEL_DRY_RUN=1` 이면 train_one·eval_one 이 실행 대신 명령·env·저장 경로만 적음. 9-30·9-40 이 이것을 씀
- preflight_checks.sh: `VESSEL_FP_IGNORE`(:17)에 `EVAL_ARMS|REUSE_ARMS|DRY_RUN` 추가(실행 관리용 — 안 넣으면 train↔eval 사이 지문이 달라 검사가 매번 다시 돔, 결과 오류는 아님). TELEMETRY·TELEMETRY_EVERY 는 이미 들어 있음
- 텔레메트리 열(뒤에만 추가): act_zero_slot·act_zero_own(cz), act_zero_goal·act_zero_sit·act_zero_threat(v2p), 코덱 팔(cc2·cc6·cc12·cr6·ca6) cdc_psi_q50·cdc_sog_q50·cdc_role_agree(D 로 복호해 실행 중 분포 이동 감시, 기록 전용), cr6 전용 **r6_fb**(직전 행 이후 rollout 결정 중 fallback env 비율)·**r6_self**(유효 항목 중 j′ = i 비율) — rollout 의 G_train 셔플에서 센 값(텔레메트리 자체 셔플 아님)
  - 텔레메트리는 ON 팔만 기록(vessel_gym_train.py:960 `args.arm == 'ON'`) → offr 는 `_comm.csv` 없음(정상)
  - 학습기는 텔레메트리 예외를 삼키고 파일을 닫음(:1277-1279) → 버그 점검 체크리스트에 '행이 끝까지 있음'·'실패 줄 0' 을 넣음(§7-7)
  - eval: R6·slot-shuffle 평가 로그에 `[slot_shuffle] fallback=… self=…` 한 줄 + eval meta 에 같은 값
- **문서 갱신(구현 커밋에 같이)**: `Assets/Scripts/.claude/CLAUDE.md` §4(k/v 깊이·cz 토큰)·§5(COMM_KV_DEPTH = 키 결정자, COMM_EXT_LAYOUT·COMM_KV_HIDDEN = shape 결정자, 코덱 blob 은 model_state_dict 밖)·§7(새 env 표)

### 4-8. 팔 구분 표시 (로그·검사표·meta)

- 문제: 지금 `Restored.header`(ckpt_io.py:232)와 `_variant`(verify/check_branch.py:64-76)는 fields·latent·반경·aux 만 찍음 → 신규 팔이 'full,L0,aux0' 으로 같게 보임. 스냅샷 arm 이 전부 ON(check_branch.py:58 주석)이라 구분 근거는 이 키들뿐. 루트 CLAUDE.md §2 '실행 이름으로 구조 판단 금지'
- 규칙: **layout·kv·payload·codec k·codec 내용 SHA 앞 12자·bits·shuffle·arm_tag** 를 세 곳 모두에 찍음
  - `Restored.header`(eval·diag 로그 첫 줄)
  - `_variant`(check_branch 표)
  - eval_ckpt traj/eval meta(eval/eval_ckpt.py:832)
- `_variant` 형식(v1 팔은 지금 문자열 그대로 → 옛 검사표 불변. arm_tag 가 있을 때만 뒤에 붙임)

| 팔 | _variant |
|---|---|
| off(재사용) | latent (지금 그대로) |
| offr | latent,tag:offr |
| arpa6(재사용) | state,L0,R56,aux0 (지금 그대로) |
| cp0 | own,L0,aux0,cz6,kv256x3 |
| cc2 / cc6 / cc12 | full,L0,aux0,cz{2\|6\|12},kv256x3,P:codec:k{2\|6\|12}:b8:<sha12> |
| ccinf | full,L0,aux0,cz25,kv256x3,P:true |
| cr6 | full,L0,aux0,cz6,kv256x3,P:codec:k6:b8:<sha12>:shuf |
| ct2 | full,L0,aux0,cz2,kv256x3,P:trunc2:b8 |
| ca6 | full,L0,aux0,v2p,kv256x3,P:codec:k6:b8:<sha12> |

- off 와 offr 는 구조가 같게 설계됨 → 구분은 arm_tag(표시)·파일 SHA·학습 커밋. 판정 앵커는 off 고정(§7-2)
- 테스트 9-27: 11팔 문자열이 서로 다름(header·_variant·meta 각각)

### 4-9. 미러 보장 논증

- COMM_EXT 와 같은 **구조적 미러**(구현 검토가 A 판에서 코드로 확인한 경로 그대로)
  - slot·own4·(A6 의 P̂·쌍별 필드·덧붙임)는 rollout 의 comm_gather 에서만 계산 → prelpos 하나를 집계와 반환에 같이 씀(:350) → 버퍼 'prel'(:1041) → update 는 저장값 재사용(:1082, :1107 → networks.py:1372-1377)
  - `evaluate_actions` 는 폭 검사만(:1372-1374), 집계는 rollout 과 같은 `aggregate_batch`(:527) → 고칠 곳 없음. k/v 깊이는 모듈 안 구조라 rollout·update 가 같은 모듈을 탐
- 코덱은 no_grad·결정론 반올림 → update 그래프 밖. own4 는 env 값
- R6 셔플은 rollout 에서만 난수 사용, 결과가 prelpos 로 저장 → update 는 같은 값
- last-value·텔레메트리·eval·diag 가 모두 comm_gather 를 탐 → generator 규칙(§4-2 표)만 지키면 자동 일치
- 검증: `_verify_comm_mirror` CASES·`test_mirror_all_arms` 에 신규 팔 추가(테스트 9-12), Windows `_verify_ppo_mirror` 1회
- 09-28 C 전환분 구현 검토가 이 논증을 코드로 재확인함(own4·slot 은 rollout 에서만 계산, update 재계산 경로 없음 — §4-2 끝 줄)

### 4-10. Unity 경로

- `_get_others_msg` 는 relpos_dim ≠ 3 이면 RuntimeError(networks.py:1158-1161) → cz·v2p 도 자동으로 막힘. gym 전용
- C#·obs 369D 무변경

### 4-11. 크래시 재개 절차 (배치 X 의 중간 재개·CSV 절단 선례 — `_FOLLOWUP_0927.md` 곡선 연속성 표)

1. 프로세스가 실제로 죽었는지 확인(루트 CLAUDE.md §5) → 락 정리
2. 마지막 `ckpt_every` 체크포인트 선택, 그 `steps` 확인
3. 곡선 CSV·aux·comm CSV 를 그 steps 행까지 절단(awk, 배치 X 와 같은 방식), 절단 행 수 기록
4. 재개 실행: 스냅샷의 codec SHA·payload·layout·kv·shuffle·arm_tag 가 팔 env 와 같은지 `_cur_comm` 이 대조(다르면 거부)
5. G_train 을 (seed, resume_at) 로 재시드, `comm_shuffle_seed` 기록
6. 분기 키·reinit·adam_remap 키 상속 확인(스냅샷 비교)
7. 재개 사실을 `_status_train.txt`·§13 에 기록. 재개한 런도 판정에 그대로 씀

### 4-12. A6 경로 (보험·읽기 진단 — A 판 설계 그대로, 청자 k/v 만 공통 3×256)

- **'보험'의 뜻(09-28 정리, §11-27)**: A6 는 헤드라인·확증에 못 씀(§7-4) → 논문 주장을 지켜 주는 기능은 없음. 여기서 보험 = ① C 가 지면 다음 주기 방향(손 공식 읽기 쪽)을 정할 근거 ② 같은 z 에서 읽기 방식 비교(G-C5 읽기 진단). 논문 표기는 '읽기 진단 팔(A6)'

#### 4-12-1. `comm_pair_features` 쪼개기 (vessel_gym.py:277-339) — A6 실행과 코덱 학습 L_pair 가 같이 씀

- 셋으로 나눔
  - `own_motion(env)` → 참 h(rad)·spd(m/s)·rot_n·cmd_n·ts_n [E,N]. `own_motion4`(§2-5)는 이 값에서 [sin h, cos h, spd/1.8, rot_n] 을 만듦
  - `_pair_core(pos_i, h_i, spd_i, pos_j, h_j, spd_j, rot_j, cmdr_j, cmds_j, partner_range, role_gate, dt)` — 현 :286-338 연산을 **순서 그대로**
  - `comm_pair_features(env, topi, partner_range, role_gate=True)` — 시그니처 불변. gather 후 `_pair_core` 호출
- **참값 경로 비트동일 조건**
  - core 가 h_j 를 **rad** 로 받음(:291 `env.heading[b,topi]*DEG` 그대로). sin/cos 를 받으면 약 1e-7 차이 → 역할 문턱 흔들림
  - rot_j·cmd 는 래퍼가 현 식(:327, :336-337)으로 계산해 넘김
  - **텐서 모양·브로드캐스트 형태도 같게**: i 쪽은 [E,N,1] 유지(원래 코드와 같은 브로드캐스트). CPU 에서 sin/atan2 는 벡터 경로와 스칼라 꼬리 경로가 1 ulp 다를 수 있음 → 테스트 9-1 에 홀수 크기 E·N·K 포함
- 조율 진단 GT(eval/eval_ckpt.py:314)는 계속 참값 래퍼 → GT 역할 오염 없음
- 파트너 위치 pos_j·수신자 참값은 전 경로에서 env 참값(위치 공유 가정)

#### 4-12-2. v2p 분기 (`a6_ext`)

```
P = own_payload(env, x, goal, sit)                               # [E,N,25]
with torch.no_grad():
    mode = net_mod.COMM_PAYLOAD
    if mode == 'codec':                                          # A6 학습·기본 평가
        q_ent = quant(E(norm(P)))[b, topi]                       # [E,N,Kc,6] — C6 와 같은 코덱·같은 q
        if net_mod.COMM_SLOT_SHUFFLE:                            # 평가 z-shuffle 전용(학습 팔 없음)
            if slot_shuffle_gen is None: raise RuntimeError(...)
            q_ent = sender_derange(q_ent, topi, pmask, slot_shuffle_gen)
        Ph_ent = decode_p25(D(q_ent))                            # [E,N,Kc,25] — 다시 gather 하지 않음
        ext20 = vg._pair_core(수신자 참값, pos_j 참값, Ph_ent 의 운동·명령)
    elif mode == 'true':                                         # 평가 CLI 전용 교차(A6 ckpt 에 참 P)
        Ph_ent = P[b, topi]
        ext20 = vg.comm_pair_features(env, topi, PART_R)
ext = cat([ext20, Ph_ent[..., 6:25]])                            # [E,N,Kc,39]
(기존 field-shuffle 은 A6 평가에서만 허용 — 행 단위, own4 없음)
```

- z-shuffle 은 **항목 단위 코드 q_ent** 에 적용 → 쌍별 필드는 **참 relpos + 다른 송신자의 운동**으로 다시 계산됨(기존 field-shuffle 은 필드 행을 통째로 옮김 — 다름)

#### 4-12-3. A6 덧붙는 부분(목표·상황·위협)의 좌표계

- goal: **송신자 선수 기준** 각 + 거리 d/(d+150)
- sit: 좌표 없음. 송신자의 56 m 안 최위험 상대 기준 상황 — 그 상대가 수신자인지 제3선인지 구분 정보 없음
- threat: **송신자 선체 좌표** ray 방위·/56 거리·같은 ray closing
- 처리: **변환 없이 토큰에 그대로**(저자 결정 1). 수신 k/v 가 relpos·rel_heading(ext[0:2])와 함께 학습으로 씀. 수신측 고정 좌표 변환은 대안으로만(§11-7)
- A6 는 수신자 own4 를 토큰에 따로 넣지 않음(A 판 그대로). 수신자 절대속력은 ext 항등식으로 들어옴(§5-4)

---

## 5. 팔

### 5-1. 팔 표 (이름은 가칭, 전부 `x_` 접두어 = 배치 X trunk 재사용)

| 팔 | 상태 | layout | fields | payload | 슬롯·bit | k/v | partner_range | 의미 |
|---|---|---|---|---|---|---|---|---|
| off | 재사용 | v1 | latent | – | – | 1×64(미사용) | – | 레이더만. **최선 OFF·판정 앵커**(§7-2) |
| arpa6 | 재사용 | v1 | state | – | 참값 필드 56 m | 1×64 | 56 | 레이더 안 설계 특징(통신 없음), 참고 |
| offr | 신규 | v1 | latent | – | – | 1×64(미사용) | – | off 재학습 3런 = **잡음 추정 전용**(앵커 아님, §7-10) |
| cp0 | 신규 | cz6 | own | – | 0 bit | 3×256 | 300 | **P0**: 위치 + 수신자 own4, z 자리 0 |
| cc2 | 신규 | cz2 | full | codec k=2 | 16 bit | 3×256 | 300 | C2 |
| cc6 | 신규 | cz6 | full | codec k=6 | 48 bit | 3×256 | 300 | **C6 = 확증 비교 처치** |
| cc12 | 신규 | cz12 | full | codec k=12 | 96 bit | 3×256 | 300 | C12 |
| ccinf | 신규 | cz25 | full | true | P 25 float32 | 3×256 | 300 | **C∞**: 코덱 없는 참 P 를 청자가 직접 = 상한 |
| cr6 | 신규 | cz6 | full | codec k=6 + slot-shuffle | 48 bit, 짝 끊김 | 3×256 | 300 | **R6**: 송신자–z 짝 정보 없음 |
| ct2 | 신규 | cz2 | full | trunc2 | ψ·SOG 원값 16 bit | 3×256 | 300 | **T2** 원값 절단 |
| ca6 | 신규 | v2p | full | codec k=6 → D → `_pair_core` | 48 bit(C6 와 같은 z) | 3×256 | 300 | **A6**: 보험·읽기 진단 |
| ons6·oni6·onl6 | 보고만 | v1 | – | – | – | 1×64 | – | 앵커로 대체 안 함 |

- **폐기(09-28 C 전환)**: ca2(A2)·ca12(A12)·csinf(A형 S∞: 참 P 를 `comm_pair_features`+덧붙임으로)·A형 ct2(수신측 `_pair_core` 로 계산하던 T2)
- 전 신규 통신 팔의 bit 는 '+ 위치 무손실(float32 2개)' 병기(§3-3)
- **R6 가 끊는 것과 남기는 것**
  - 끊음: 송신자–z 짝(= z 와 그 송신자의 relpos·수신자 own4 와의 결합)
  - 남음: env 안 유효 z 모음(순열), z 주변분포, 참 relpos, 수신자 own4(절대속력 포함)
- R6 짝 끊기 절차(학습·평가 공통, 결과 전 고정)
  - 대상: 유효 (수신자 i, 슬롯) 항목. 원 코드 = q(z_j)
  - env 마다 항목을 송신자 j 로 묶어 일렬로 둠(묶음 순서·묶음 안 순서 = 전용 generator 난수) → **가장 큰 묶음 크기 b_max 칸 순환 이동** → 각 항목이 다른 송신자의 코드를 받음(M ≥ 2·b_max 면 같은 묶음으로 안 돌아옴 — 계산으로 확인)
  - 기존 field-shuffle(항목 단위 한 칸 이동)을 z 에 쓰면 같은 송신자의 다른 항목에서 같은 z 를 받아 정보가 남음 → 송신자 단위로 바꾼 이유
  - M < 2·b_max 인 env 는 그 결정의 슬롯을 0(P0 와 같은 입력), 빈도 기록
  - 받은 코드가 수신자 자기 것인 항목(j′ = i)은 허용, 빈도 기록
  - generator: §4-2 표
- T2 는 k=2 비교 전용. R6 은 k=6 만. A6 는 k=6 만

### 5-2. 공통 학습 인자·배치 실행 규칙 (off 앵커 보호)

- 인자(배치 X 와 동일)
  - `VESSEL_COMM_EXT=1 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0`, 레이더 56, MSG_DIM 6, COMM_LATENT=0, AUX_LOSS_SCALE=0
  - `--resume x_trunk_d6_s<seed>.pt --resume_at 9043968 --comm_on_at 9043968 --resume_warmup 1200 --steps 16056320 --envs 128 --vessels 16 --rollout 32 --ring 1.0 --crossing 0 --max_partners 4 --seed <43|44|45> --ckpt_every 2`
  - 신규 통신 8팔은 `comm_variant_env` 가 `VESSEL_COMM_KV_HIDDEN=256 VESSEL_COMM_KV_DEPTH=3` 을 같이 줌. offr 는 kv 기본(64, 1) = x_off 와 같은 spec
  - **텔레메트리 핀**: `VESSEL_COMM_TELEMETRY=1 VESSEL_COMM_TELEMETRY_EVERY=5` 를 배치 env 로. 근거 = 배치 X `x_ons6_s43_comm.csv` 행 간격 327,680 결정 = 5 update(첫 행 9,371,648). 두 키는 preflight 지문 제외 목록에 이미 있음(preflight_checks.sh:17) → 켜도 지문 불변
- **off·arpa6 은 `VESSEL_TRAIN_ARMS` 에서 뺌**. "off 를 넣으면 기존 x_off 는 학습 건너뜀"은 사실과 다름
  - 근거: branch_batch 는 `VESSEL_REUSE_ARMS=1` 일 때만 건너뜀(run_repro.sh:424-426). 그 외에는 train_one 이 곡선 CSV 를 먼저 지우고(:287) `$CK/x_off_s*.pt` 에 다시 저장(:295) → 사전등록 앵커 off(44.7)가 재학습본으로 조용히 바뀜(재학습 잡음 4.1/5.8/12.4 pp)
  - `VESSEL_TRAIN_ARMS="cp0 cc2 cc6 cc12 ccinf cr6 ct2 ca6 offr"`
  - offr 는 새 이름(`$CK/x_offr_s*.pt`) → 기존 x_off 를 안 건드림
  - **정정(09-28 사전등록 검토)**: 짝 검사(run_repro.sh:386-399)는 목록 안에 OFF 팔(offr = 'OFF 6')이 있으면 has_off=1 이 되어 `$CK/x_off_s*.pt` 존재 검사를 건너뜀 → 실제로는 '기존 x_off 로 통과'가 아니라 **'offr 로 통과'**. check_branch 의 OFF 짝 검사도 offr 로 채워짐. check_branch 파일 목록(:435-436)은 `$CK/x_off_s*.pt` 가 있으면 자동 포함
  - 그래서 x_off·x_arpa6 누락을 이 경로로는 못 잡음 → **배치 시작 때 재사용 파일 9개 존재 검사(§4-7 run_repro 변경)** 를 명시 항목으로 둠. 뒤의 앵커 SHA 전후 대조도 누락을 잡음
- **새 `VESSEL_OUT_DIR`**(가칭 `runs/2026-09-2x_grounded_codec/`). 배치 X OUT(`runs/2026-09-26_comm_ext_batch/`)에 쓰기 금지
  - 이유: 같은 OUT 이면 eval 이 `eval_x_off_s*.txt` 를 덮어써서(:321) G-C0 대조 기준값이 사라지고, train 모드가 `_status_train.txt` 를 비움(:449)
  - 새 OUT 에는 trunk CSV 3쌍(`x_trunk_d6_s*.csv`, `*_aux.csv`)을 복사 — train_one 이 갈래 CSV 시작으로 씀(:288-289)
  - 재사용 x_off·x_arpa6 의 곡선·aux·comm CSV 도 복사(check_branch 검사 5 용, §4-5). 전부 복사 전후 sha256 기록, 배치 X OUT 쪽은 읽기만
- **앵커 파일 SHA 전후 대조**: 배치 전·후 `$CK/x_off_s*.pt`·`x_arpa6_s*.pt`·`x_trunk_d6_s*.pt` 9개 sha256 기록·대조. 하나라도 다르면 배치 무효
- `VESSEL_REQUIRE_TRUNK=1`(기존 가드), `VESSEL_REUSE_ARMS` 는 중단된 배치 재개 때만 1
- eval 은 `VESSEL_EVAL_ARMS="off arpa6 offr cp0 cc2 cc6 cc12 ccinf cr6 ct2 ca6"`, eval 시작 check_branch 에 x_arpa6 포함. 평가 arm 은 `eval_arm()`: **off·offr → OFF**, 나머지 9팔(arpa6 + 신규 통신 8팔) → ON(§4-7 표)
- **신규 9팔 × 3시드 = 27런을 한 배치로 한 번에 제출**(동시 수는 VESSEL_JOBS 로만 제한). 유망한 팔만 골라 계속하는 일 금지. 게이트는 해석 순서일 뿐 실행 여부를 안 바꿈

### 5-3. eval 조건 (배치 X 와 동일)

- envs 256, burn-in 2400, 10000 결정/에이전트, seed 999, crossing 스냅샷, `run_repro.sh eval`
- 재사용 off·arpa6 은 2단계 재평가(G-C0, §7-5). offr 는 신규 팔과 같이 기본 eval

### 5-4. 비교 논리

- **수신자 자기 절대속력 경로 — C 비교에서 해소되는지 확인(09-28)**
  - A 판 문제: 쌍별 필드가 수신자 참 절대속력 spd_i 를 씀(vessel_gym.py:303-304 rvx = gx·spd_j − fx·spd_i, relvel 필드 :332-333, faster :317-318) → 필드끼리 **spd_i = 1.8·f[2]·f[1] − 3.6·f[5]** 항등식 성립 → A·S∞·T2·R6 는 절대속력을 받는데 P0(ext 전부 0)는 못 받음 → A(k) vs P0 '귀속 미분리'
  - **C 판**: cz 레이아웃 팔 전부(P0·C2·C6·C12·C∞·R6·T2)가 토큰에 own4(SOG_i 절대값 포함)를 가짐. P0 도 같음(fields own = own 그룹 켜짐)
  - own4 가 들어가는 조건(유효 파트너 슬롯, pmask)은 파트너 선택(300 m 참 거리 nearest-4)에만 의존하고 z 와 무관 → 같은 상태면 P0 와 C(k) 에서 같은 슬롯에 같은 값
  - **결론: C(k)·C∞·T2·R6 vs P0 에서는 이 경로가 양쪽에 같게 있어 해소됨.** C6 vs P0 는 shape·초기값까지 같아 차이 = z 값(내용 + 주변분포)뿐
  - **해소 안 되는 비교**: C(k) vs off(off 는 절대속력·relpos·통신 경로 모두 없음 → 확증 비교 = '통신 채널 전체' 효과), P0 vs off, A6 vs C6(A6 는 항등식으로만 spd_i 를 받고 세계 침로 ψ_i·ROT_i 는 없음, C6 는 own4 로 명시)
  - 그래서 저자 논문 포인트 '적은 차원 latent 로도 이득'은 **C6 vs off 만으로 답해지지 않음** → 'latent' 문구 허용 조건을 G-C3 ① 에 묶음(§7-4)
  - 근거 확인: 09-28 구현 검토가 own4 경로(comm_gather → prelpos → buf['prel'] → update 재사용)와 이 결론을 코드로 확인. 사전등록 검토가 C6=P0=R6·C2=T2 의 shape·초기값 동일, P0 의 0 열 grad 0, own4 가 pmask 에만 의존함을 재현
  - 남는 일반 한계: 학습 중 궤적이 팔마다 갈라지므로 '같은 상태면 같은 값'은 입력 규칙의 동일성일 뿐 분포 동일성은 아님(모든 짝 비교 공통)

| 비교 | 같은 것 | 다른 것 | 답하는 질문 |
|---|---|---|---|
| C∞ vs off | trunk | 무손실 P 청자 + 위치 + own4 + 청자 용량 | 상한 게이트(G-C1) |
| **C6 vs off** | trunk | 48 bit z 청자 + 위치 + own4 + 청자 용량 | **확증 비교(G-C2)** |
| C2 / C12 vs off | trunk | 16 / 96 bit + 같은 경로 | 부 비교 |
| C6 vs P0 | 위치·own4·토큰 폭 19·k/v shape·초기값 텐서 | z 값(내용 + 주변분포) | 귀속 ①(G-C3) |
| C6 vs R6 | 위치·own4·폭·초기값·z 주변분포·env 안 z 모음 | 송신자–z 짝(z 와 relpos·own4 의 결합) | 귀속 ②(G-C3) |
| C2 vs T2 | 16 bit·폭 15·격자·초기값·위치·own4 | 동결 코덱 z vs 원값 ψ·SOG(연속성 차이 포함, §3-10) | 압축 가치(G-C4, k=2 만) |
| C(k) vs C∞ | 위치·own4·k/v 구조 | 압축 손실 + 폭(첫 층 fan_in) | 회복률 R(k)(서술) |
| A6 vs C6 | trunk·같은 z(같은 코덱 SHA)·k/v 구조 | 읽기(D + `_pair_core` + 덧붙임 vs 학습)·토큰 폭 48 vs 19·own4 유무 | 읽기 진단(G-C5) |
| A6 vs off | trunk | 48 bit z + 손 공식 읽기 + 위치 + 용량 | 부 비교(보험) |
| P0 vs off | trunk | 위치 + own4 + 청자 용량 + 채널 경로 | 위치·자기 상태 효과 |
| C2 / C12 vs P0 | 위치·own4 | z + 폭(15·25 vs 19) | 부 비교, 귀속 미분리(폭 포함) |
| offr vs off | trunk·설정 | 재학습(새 코드·이번 배치 GPU) | 잡음 추정 |
| C∞ vs arpa6 | 일부 정보 | 반경·필드 형식·청자 용량·폭·latent | 참고만(차이가 여러 개 섞임) |

---

## 6. 평가·절제·기전 진단 (목록 고정, 서술 전용)

| 절제·진단 | 대상 | 건수 | 뜻 |
|---|---|---|---|
| 기본 eval | 신규 9팔(offr 포함) | 27 | §7 판정 |
| 재사용 팔 재평가 2단계(G-C0) | off·arpa6 × {7013a3a, 새 코드} | 12 | 재현·코드 동일성 |
| msgzero(others_msg 전부 0, `--arm OFF --allow_arm_mismatch`) | 신규 통신 8팔 | 24 | 통신 경로 의존도 |
| slot-shuffle(R6 방식 송신자 단위를 평가 때만, G_eval, 슬롯 열만) | cc2·cc6·cc12·ccinf·ct2 | 15 | 슬롯 짝 끊기(Lowe 2019 개입) |
| z-shuffle(A6, q_ent 에 같은 방식) | ca6 | 3 | 같은 개입을 A6 에 |
| field-shuffle(ext39 전체, 기존 derangement) | ca6 | 3 | A 판 개입 유지 |
| 그룹 0 — **보조** | cc6 {slot0, own0} · ccinf {slot0, own0, p_motion0, p_cmd0, p_goal0, p_sit0, p_threat0} · ca6 {state0, role0, intent0, goal0, sit0, threat0} | 45 | '0 값 입력'(학습 때 없던 값)이라 보조로만 |
| 교차: C∞ ckpt 에 복원 P̂_k(`--payload_override recon`, k=2·6·12) | ccinf | 9 | 복원 오차에 대한 **의존도일 뿐 이득 아님** |
| 교차: A6 ckpt 에 참 P(`--payload_override true`) | ca6 | 3 | 복호 오차를 없앴을 때 변화(의존도) |

- C(k) 체크포인트는 슬롯 폭·z 좌표계가 k 마다 달라 **payload 교체 교차평가 불가**(§3-2). 의존도는 slot-shuffle·msgzero 로만
- 교차평가는 CLI `--codec/--codec_sha/--payload_override/--slot_shuffle` + `allow_codec_override` 로만, notes·header 에 기록
- 조율 진단·충돌 분해: 기존 eval 출력 그대로 — **기전 설명 전용**, 판정에 안 씀
- 코덱 충실도(오프라인 G-C0 표 + 평가 창, 코덱 팔·교차평가)
  - 성분별 오차 분위수(q50/90/99/99.9/max): ψ(deg)·SOG·ROT·cmd 2·goal 2, sit 일치율, threat 감지 일치율·방위 오차
  - 역할 일치율(my/their, 참·복호 중 하나라도 None 아닌 쌍 기준), dcpa 게이트 뒤집힘률
  - **전부 거리대별**: ≤56 / 56–150 / 150–300 m. **≤56 m 를 주 보고**, 쌍 수 병기
  - 기준선: 같은 창의 자연 역할 변동(0.27–0.63%, §0-3)
  - C 에서의 뜻: 'z 에 복원 가능한 정보가 얼마나 남았나'(화자 쪽 상한). 청자가 그만큼 읽었다는 뜻 아님 — 명시
  - traj 기반 점검은 §3-6 규칙(명령 한 칸 당김·재스폰 제외·goal 재계산)
  - **z 토큰 지도학습 상한 probe(보고 전용, 09-28 추가)**: `sup` 와 같은 라벨(참 dcpa_risk·내 역할·상대속도)·같은 3×256·같은 학습 레시피(배치 1024, 표본 2e5·2e6·2e7), 토큰만 [relpos 3, q(E_k), own4] 로 바꿈(k = 2·6·12) + 같은 형식의 T2 토큰·C∞ 토큰 1개씩. 데이터 = 코덱 train/holdout 분할 그대로(처치 팔 궤적 안 씀). **설계·판정 불변** — G-C5 3행·G-C4 해석에서 '압축 손실 vs 청자 학습 부담'을 가르는 결과 전 자료로만
  - **z 성분별 표준편차**(holdout, k 별) + C∞ 의 P 성분별·T2 2칸 표준편차를 같은 표에 → 값 범위뿐 아니라 퍼짐이 같은 자릿수인지 보고(§2-4 는 범위만 비교했음). 결과 보고 정규화 안 바꿈
- 합계 141건(기본 27 + 재평가 12 + 절제·교차 102). 소요(추정)는 동시 실행 수에 따라 약 3.4–7 h(§10). z 토큰 probe 는 오프라인 CPU 라 건수 밖

---

## 7. 사전등록 판정

### 7-1. 지표

- 주 지표: 선박충돌률 **vColl**(OBB 사건 판정 = 보상과 독립)
- 부 지표: goal·timeout·allMinSep
- 보조(보상 결합 표시): fuel·headTravel·Rule8@10·colregs 계열
- 근거로 안 씀: epReward·학습곡선

### 7-2. 최선 OFF

- 직전 스펙 §8-6 (a) 팔 단위 = 시드 평균 vColl 이 낮은 팔 = **off**(44.7 < arpa6 46.7). 결과 전 고정. 짝 승패·평균차·시드 범위 모두 off 기준
- offr 는 앵커로 안 씀. offr 가 off 보다 좋게·나쁘게 나와도 앵커 교체 금지(§7-9)

### 7-3. 주장 기준 (강화) — §11-12 채택 대기

- **짝 승** = 같은 시드에서 처치 팔 vColl(eval 보고값, 소수 1자리) **<** off vColl. **같으면 승 아님**('동률'로 표기, 패도 아님)
- **강화 기준** = **3/3 짝 승 그리고 평균차 > max(off 시드 범위 1.8, MDE 6) = 6.0 pp**. 평균차 = 시드별 보고값 차의 평균(반올림 없음)
- 기존 기준(평균차 > off 범위 1.8)으로 본 결과도 병기. 주장은 강화 기준으로만
- 결과 전에 정하고 통신에 불리한 방향이라 사후 강화 아님
- 3/3 이지만 평균차 ≤ 6 → '방향 일치, 크기 판정 불가'. 3/3 미만 → '판정 불가'
- 두 처치 팔끼리 비교(C6 vs P0·R6·A6, C2 vs T2, C(k) vs C∞, offr vs off)도 짝 승 정의는 같음(off 자리에 비교 대상 팔). 'A 가 B 를 이김' = A 가 B 대비 강화 기준 통과
  - **처치 팔끼리의 문턱 = 고정 6.0 pp**(이 문서 규칙, §11-12 채택 대상에 포함). 비교 대상 팔의 시드 범위는 문턱에 안 넣음 → 결과를 본 뒤 문턱을 고를 여지 없음. 예: P0 가 한 시드에서 붕괴해 시드 범위가 11 pp 여도 문턱은 6.0 pp 그대로(붕괴 대조는 §7-8 이 따로 처리)
  - 병기하는 '기존 기준'도 처치 팔끼리는 **고정 1.8 pp**(off 시드 범위 수치를 그대로 씀)
  - off 대비 문턱 max(1.8, 6) = 6.0 은 off 값이 이미 정해져 있어 원래 고정임 → 모든 비교의 문턱이 6.0 pp 로 같음

### 7-4. 확증 비교와 부 비교 (다중 비교)

- **확증 비교 = C6 vs off 1개**(G-C2). 헤드라인·초록에 쓸 수 있는 것은 이것뿐(G-C1 통과 전제)
- **확증 비교가 답하는 것 = '통신 채널 전체'(위치 공유 + 수신자 자기 상태 토큰 + 학습 청자 3×256 + z)의 효과**(§5-4). 저자 논문 포인트 '적은 차원 latent 로도 이득'은 이것만으로 검정되지 않음(09-28 사전등록 검토 blocking)
- **'latent' 문구 규칙 — 안 (a), §11-30 저자 택1 대기(권장 = a)**
  - '6차원 latent 통신 이득'·'적은 차원 latent 로도 이득'·'받는 배가 z 를 읽어 이득' 문구 = **G-C2 통과 그리고 G-C3 ①(C6 vs P0) 강화 기준 통과**일 때만 헤드라인·초록에 씀. '송신자별 z 내용에 귀속' = 추가로 G-C3 ②(C6 vs R6) 통과
  - G-C2 만 통과(G-C3 ① 미통과): 헤드라인 주어 = **'위치 공유 + 수신자 자기 상태 토큰 + 학습 청자 + 6차원 코덱 z 로 된 통신 채널'**. **'z 기여 미확인'** 필수 병기. 'latent' 가 이득의 주어가 되는 문장 금지
  - latent 주장 쪽 위양성: 두 비교 교집합이라 확증 1개보다 안 늘어남(검정력만 줄어듦)
  - 안 (b)(대안): 확증을 교집합 검정 **C6 > off ∧ C6 > P0**(둘 다 강화 기준)로 바꿈. 통과 못 하면 채널 헤드라인도 없음. (a) 와 latent 문구 조건은 같고, 차이는 'G-C2 만 통과' 때 채널 헤드라인을 허용하느냐뿐
  - (b) 채택 시 고칠 곳: 이 절 첫 줄, §7-5 G-C2, §7-12 표 3행(채널 헤드라인 줄 → '확증 미통과: 헤드라인 없음' 줄), §1 표
- 부 비교(다중 비교 보정 없음): C2·C12 vs off, C2·C12 vs P0, C2 vs T2, C(k) vs C∞, P0 vs off, T2 vs off, **A6 vs off**, A6 vs C6, R(k)
- **부 비교만 통과**(예: C6 실패인데 C2·C12·A6 중 하나 통과) → 헤드라인·초록에 안 씀. 본문에 '부 비교(다중 비교 보정 없음)'로 표기하고 무효과 위양성률을 병기(§7-10 표 전체 범위 σ_d 3.2–8.2: k 3개 중 하나 통과 0.16–15.7%, C6 또는 A6 중 하나 0.11–11.7%)
- 결과 뒤 '최적 k' 를 헤드라인으로 올리지 않음. A6 를 결과 뒤 확증으로 올리지 않음(§11-27)

### 7-5. 게이트 (해석 순서)

- **G-C0 구현·충실도 (학습 전)**
  - §9 테스트 전부 PASS
  - 코덱 3개 SHA 커밋·§13 기록, 오프라인 충실도표 동결(§0-0)
  - 재사용 팔 재평가 2단계(off·arpa6 6건씩)
    - 1단계: **7013a3a 코드**, 배치에 쓸 같은 GPU → 배치 X eval 값(FOLLOWUP §2 표, goal/vColl 소수 1자리)과 대조 = GPU 비결정성 확인
    - 2단계: **새 코드**, 같은 GPU → 1단계 값과 대조 = 코드 변경 효과만
    - 2단계 ≠ 1단계면 원인 보고 전 진행 금지. 1단계 ≠ 배치 X 일 때 처리는 §11-13
  - 충실도가 나쁘게 나와도 k·코덱·λ·청자 구조 안 바꿈(버그는 §3-11)
- **G-C1 상한 게이트**: C∞ vs off, **§7-3 강화 기준**
  - 실패 → 결론 "학습 청자가 무손실 P 를 읽어도 이득이 이 재학습에서 재현 안 됨". C 계열 주장 안 함(수치는 전부 보고)
  - G-C1 실패인데 C6 가 강화 기준 통과 → '상한 미통과 상태의 C6 이득 — 원인 미상(잡음, 좁은 입력이 청자에게 배우기 쉬움 등)'으로 보고, 헤드라인 금지
- **G-C2 확증**: C6 vs off, 강화 기준. 답하는 것 = 통신 채널 전체 효과. 'latent' 문구는 §7-4 규칙
  - offr 조건(§7-10, §11-31 채택 시): offr vs off 가 어느 방향이든 강화 기준을 통과하면 G-C2 헤드라인에 **C6 vs offr 강화 기준 통과**를 추가 조건으로 붙임(앵커 교체 아님, 조건 하나 추가)
- **G-C3 정보 귀속**
  - 해석 문장은 **G-C2 통과 때만** 씀. G-C2 미통과면 C6 vs P0·R6 결과는 수치만(부 비교 표기)
  - C6 vs P0, C6 vs R6 를 각각 강화 기준으로(문턱 고정 6.0 pp, §7-3)
  - **둘 다 통과해야 'z 의 송신자별 내용에 귀속'**
  - C6 vs P0 만 통과 → 'z 입력 효과는 있으나 송신자별 내용인지 z 주변분포(입력 추가) 효과인지 가르지 못함'
  - C6 vs R6 만 통과 → '짝 정보 효과 방향. P0 대비 크기 판정 불가(R6 자체가 해로웠을 가능성)'
  - 하나라도 3/3 이지만 ≤ 6 pp → '방향 일치, 크기 판정 불가'. 3/3 미만 → '귀속 불가(위치·자기 상태 토큰·청자 용량으로 설명 가능)'
  - C2·C12 vs P0 는 '귀속 미분리(폭 차이 포함)'로만 표기(R6 대조 없음)
  - 대조 팔(P0·R6)이 H1a FAIL 이면 그 대조로 얻은 통과에 '대조 H1a FAIL 동반'(§7-7)
- **G-C4 서술**
  - 압축 가치 C2 vs T2 — 판정 순서대로 첫 번째로 맞는 줄 하나(09-28: '둘 다 짐'·상대 H1a 를 먼저 봄)

| 순서 | 조건 | 해석 문장 |
|---|---|---|
| 1 | C2·T2 모두 off 대비 강화 기준 미통과 | '둘 다 off 대비 미통과': 압축 가치 판정 안 함. 두 팔 사이 차이는 수치만 적음 |
| 2 | 한 팔이 다른 팔을 강화 기준으로 이기고, 진 팔이 H1a FAIL(§7-7) | '상대 H1a FAIL 의존': "진 팔(C2 또는 T2)이 off 보다 해로웠음" 만 씀. 압축 가치 해석 문장 금지 |
| 3 | C2 가 T2 를 강화 기준으로 이김 | '동결 코덱 latent > 원값 절단(16 bit, 이 설정) — 연속 표현 효과 섞임(§3-10)' |
| 4 | T2 가 C2 를 강화 기준으로 이김 | '압축 가치 없음(이 설정)' |
| 5 | 그 외 | '구별 안 됨' |

  - 모든 줄에 필수 병기: 두 팔 각각의 off 대비 판정(강화 기준)·H1a 상태. 1행을 먼저 보므로 3·4행에 오면 이긴 팔은 off 대비 통과임(1행을 지났으면 적어도 한 팔이 통과. 진 팔이 통과했다면 이긴 팔은 시드마다 진 팔보다 낮고 평균차도 더해져 역시 통과)
  - 회복률 R(k)(§7-6)
  - C(k) vs C∞: **'열등' = C∞ 가 C(k) 를 강화 기준으로 이김**. 그 외 '구별 안 됨'. **'동등' 금지**(3시드로 비열등 입증 불가)
  - 압축 가치는 **k=2 에서만** 시험됨. C6·C12 는 운동·명령에 사실상 무손실일 것(추정) → '압축 가치' 주장의 근거로 C6 를 쓰지 않음. C6 의 주장(§7-4 문구 규칙 통과 때만)은 '25D 자기 상태를 6차원(48 bit) latent 로 보내고 청자가 읽는 통신의 이득'. 규칙 미통과면 '통신 채널 이득, z 기여 미확인'
- **G-C5 읽기 진단 (서술, A6 vs C6 — 같은 z, 손 공식 읽기 vs 학습 읽기)**
  - 판정 순서대로 첫 번째로 맞는 줄 하나(09-28 사전등록 검토: '둘 다 짐'·상대 H1a 를 먼저 봄 — 해로운 상대 팔 덕분에 저자 가설을 지지하는 문장이 나오는 구조를 막음)

| 순서 | 조건 | 해석 문장 |
|---|---|---|
| 1 | C6·A6 모두 off 대비 강화 기준 미통과 | '둘 다 짐': k=6 z 를 어느 방식으로 읽어도 이득 확인 안 됨. 두 팔 사이 차이는 수치만(읽기 방식 해석 안 함). C∞ 통과면 '압축 손실 또는 청자 학습 부담', C∞ 미통과면 '채널 경로·청자 전체 문제'로 서술 |
| 2 | 한 팔이 다른 팔을 강화 기준으로 이기고, 진 팔이 H1a FAIL(§7-7) | '상대 H1a FAIL 의존': "진 팔이 off 보다 해로웠음" 만 씀. 읽기 방식 해석 문장 금지 |
| 3 | C6 가 A6 를 강화 기준으로 이김 | 'C 이김': 같은 z 에서 학습 읽기가 고정 복원기 + 손 공식보다 나음(이 설정). 설계 쌍 특성이 제약이었을 가능성 |
| 4 | A6 가 C6 를 강화 기준으로 이김 | 'A 이김': 손 공식 읽기가 나음 → 청자 학습 부담이 병목일 가능성. C 의 실패·약세를 '압축 손실'로 쓰지 않음(C∞ 결과 병기) |
| 5 | 그 외 | '구별 안 됨': 읽기 방식 차이를 3시드로 가르지 못함 |

  - 모든 줄에 필수 병기: **C6·A6 각각의 off 대비 판정(강화 기준)·H1a 상태**. 1행을 먼저 보므로 3·4행에 오면 이긴 팔은 off 대비 통과임(G-C4 와 같은 이유)
  - 섞이는 것(결과 전 기록): 토큰 폭 48 vs 19(파라미터 +5.6%), A6 의 덧붙은 goal·sit·threat 복원값·쌍별 필드 vs C6 의 own4·원 z. '읽기 방식' = 수신 경로 전체 차이
  - 보고 전용 z 토큰 probe(§6)의 k=6 상한이 3·4·1행 서술에 참고로 붙음(판정 안 바꿈)
  - A6 붕괴 시드로 얻은 C6 의 짝 승(또는 반대)은 '대조 붕괴 의존'(§7-8)

### 7-6. 회복률

- R(k) = (v̄_off − v̄_C(k)) / (v̄_off − v̄_C∞), 시드 평균값
- 시드별 R_s 병기. 분모 < 6 pp 면 '정의 안 함'. 0–1 로 자르지 않음
- 서술 지표. 판정은 G-C2·C3 로만

### 7-7. H1a

- 신규 통신 8팔(cp0·cc2·cc6·cc12·ccinf·cr6·ct2·ca6) 각각 vs off: **3/3 나쁘고(동률은 나쁨 아님) 평균차 > off 범위 1.8 이면 FAIL**(기존 문턱 유지). offr 는 통신 팔 아님 → 대상 아님(대신 §7-10 기준선 점검)
- **기저율 병기**: 무효과여도 팔 1개 FAIL 확률 9.0–12.3%, **8팔 중 하나 이상 FAIL 36.2–45.8%**(σ_d 3.2–8.2, off 를 짝으로 공유, 200k 반복 — 스크래치패드 `rv/h1a_8.py`. A 판 7팔 값 34.2–43.3% 를 8팔로 다시 계산)
- P0·R6 가 FAIL 이어도 H1a 위반으로 기록(정보 없는 입력도 무시 가능해야 함). 3×256 청자가 무정보 입력을 잡음으로 쓰면 FAIL 위험이 커질 수 있음(추정)
- H1a FAIL 팔은 같은 표에 표시하고 **G-C2·C3 판정은 그대로 함**. P0·R6 만 FAIL 이고 C6 통과면 C6 주장은 유지하되 '대조 H1a FAIL 동반 — 채널 경로 자체가 해로울 수 있음, 귀속 판정이 통신에 유리하게 기울 수 있음'을 같이 적음
- **H1a 의존 규칙의 범위(09-28 확장)**: 위 '대조 H1a FAIL 동반'(G-C3 의 P0·R6)에 더해 **G-C4 의 C2↔T2, G-C5 의 C6↔A6 에서 진 팔이 H1a FAIL 이면 '상대 H1a FAIL 의존'**(§7-5 표 2행) — 읽기 방식·압축 가치 해석 문장 대신 '상대 팔이 해로웠음'만 씀
- **버그 점검(결과 방향과 무관, 전 팔 1회, 상한 반나절)** — 체크리스트 결과 전 고정. **맹검 순서**(09-28): (가) 는 eval 을 돌리기 **전**에 학습 산출물만 보고 끝냄. (나) 는 eval 뒤지만 결과 수치 줄을 안 엶
  - (가) eval 전
    - [1] `_status_train.txt` rc=0 전 런, 재개 기록 확인
    - [2] check_branch ALL PASS, 11팔 `_variant` 서로 다름, 신규 통신 팔(layout cz*·v2p) kv 동일
    - [3] 스냅샷 codec SHA == §13, layout·payload·kv·shuffle·arm_tag 가 팔 표와 일치, 신규 8팔에 branch_reinit·branch_adam_remap 기록 있음, offr 에는 없음(발동 조건 §4-5)
    - [4] 곡선·aux·comm CSV NaN 0
    - [5] 텔레메트리: **ON 팔 전 런의 `_comm.csv` 마지막 행 step ≥ 15,728,640(= 끝 16,056,320 − 5 update)**, 학습 로그에 `[telemetry] 실패` 줄 0(학습기가 예외를 삼키고 파일을 닫음, vessel_gym_train.py:1277-1279). cr6 의 r6_fb·r6_self, 코덱 팔 cdc_* 기록 있음(문턱 없음, 이상 시 조사)
    - [7] 배치 커밋에서 golden `--check`(새 케이스 포함)·`_verify_comm_mirror`·9-33 분기 대조 재실행
    - [8] 앵커 파일 SHA 전후 일치(§5-2), 복사한 CSV SHA 일치
  - (나) eval 뒤(결과 줄 안 봄)
    - `_status_eval.txt` 의 rc 열만 확인
    - [6] eval 로그의 **header 줄만 grep**(restore header·comm 줄 패턴, vColl 등 결과 줄 제외 — 패턴은 잠금 뒤·배치 전에 배치 X eval 로그(이미 아는 결과)에 걸어 결과 줄이 안 나오는지 확인, 테스트 9-40)해 팔 표와 대조
- **버그를 찾아 고치면**: 수정 커밋으로 **신규 27런 전체 재실행**(같은 trunk·시드). 수정 전·후 둘 다 같은 표에 보고, 주 판정 = 수정 후. 수정이 OFF 경로를 바꾸면(골든 변화) off·arpa6 재사용 전제가 깨짐 → 중단, 저자 결정
- **결과를 본 뒤 찾은 버그**(체크리스트 밖 포함): 수정·재실행 규칙은 위와 같음. 추가로 **발견 시각, 무엇을 보다가 찾았는지, 그때 본 결과의 방향(통신에 유리·불리)** 을 §13 에 적음. 결과가 나쁠 때만 더 파는 비대칭을 드러내려는 것. 결과 방향과 무관하게 같은 규칙
- 재튜닝 금지

### 7-8. 붕괴 시드 처리 (결과 전 고정, §11-18 확인 대기)

- **붕괴 표시** = 한 시드의 vColl 이 같은 팔 나머지 두 시드 평균보다 **10 pp 이상 높음**
  - 근거(과거 자료, 이번 결과 전): 선례 G6 off_s45 57.1(나머지 평균 42.85, +14.3), onl6g1_s45 57.5(+12.75). 배치 X 15런·A/B·G6-2 비붕괴 최대 편차 +7.75(oni6 s43)
  - 보조 기록: 곡선 마지막 2M 결정 goal EMA(판정에 안 씀)
- **시드는 빼지 않고 전부 보고**. 주 판정은 3시드 그대로
- 대조 팔(P0·R6·T2, G-C4 의 C∞, G-C5 에서 상대 팔)의 붕괴 시드로 얻은 짝 승 → **'대조 붕괴 의존'** 표시 + 그 시드 뺀 2시드 결과 병기(2시드 결과는 주장에 안 씀)
  - **통과가 붕괴 시드에 걸려 있으면 판정을 낮춤**(09-28, 결과 전 고정): 강화 기준 통과(3/3 + 평균차 > 6.0)가 나왔어도 그 대조 붕괴 시드를 빼면 **2/2 짝 승이 아니거나 2시드 평균차 ≤ 6.0 pp** 이면 → 주장 문장 대신 **'판정 불가(대조 붕괴 의존)'**. 통신에 불리한 쪽으로만 작동하는 규칙
  - 붕괴 시드를 빼도 2/2·평균차 > 6.0 이면 판정 유지 + '대조 붕괴 의존' 표시만
- 처치 팔(C(k)·C∞·A6 vs off)의 붕괴 시드는 그대로 패로 셈(통신에 불리한 쪽 — Assets/Scripts/.claude/CLAUDE.md §10)
- off 는 재사용이라 값이 이미 정해짐(45.2/45.3/43.5, 붕괴 아님). offr 붕괴는 표시만(앵커 아님)

### 7-9. 중단 규칙·변형 상한·시드

- 중단: G-C1 실패 → C 계열 결론 '재현 안 됨'. G-C2 실패 → 'k=6 latent 청자에서 유지 안 됨'. 어느 쪽이든 이번 주기에 **k·λ·τ·bits·손실·코덱 구조·epoch·payload 구성·청자 k/v 용량·토큰 구성·token gain·보상 재튜닝 금지**
- 앵커 교체 금지: C∞ 가 나쁘게 나와도 ons6·oni6 로 바꿔 끼우지 않음. 최선 OFF 는 off 고정(offr 로 안 바꿈)
- 변형 상한: 신규 학습 팔 9개(통신 8 + offr, 목록 고정), 코덱 3개(SHA 학습 전 고정), 청자 구조 1개(3×256), 평가 목록 §6 고정
- **동결 목록**(결과 전): 파트너 선택(300 m nearest-4), eval 조건(§5-3), 지표 정의(§7-1), 복호 규칙(§3-9), R6 절차·fallback(§5-1), generator 규칙(§4-2), 붕괴 문턱(§7-8), 버그 체크리스트(§7-7), **토큰 구성·순서·own4 정의(§2-5·§4-1), P0 슬롯 폭 6(§4-4), C∞ 입력 = P 식 그대로(§2-4), T2 표현·격자(§3-10), 청자 k/v 3×256·ReLU·기본 초기화·v 마지막 ×0.1(§4-3), 재초기화·Adam 재배치 규칙·발동 조건(§4-5)**, **09-28 추가: 처치 팔끼리 문턱 고정 6.0·1.8 pp(§7-3), 'latent' 문구 규칙(§7-4, §11-30 택한 안), G-C4·G-C5 판정 순서·필수 병기(§7-5), offr 조건(§11-31 택한 안), 붕괴 의존 낮춤 규칙(§7-8), 버그 점검 맹검 순서(§7-7), 팔 env 블록(§4-7, SHA 자리만 5단계에서 채움), 텔레메트리 핀(§5-2)**
- **시드: 이번 주기 시드 추가 없음.** 추가는 다음 주기 사전등록으로만 하고, 이번 3시드 판정이 주 판정으로 남음(경계선 결과에서만 시드를 늘리는 선택적 중단 방지)
- 결과 뒤 추가 실험(예: 청자 용량 스윕, B)은 새 사전등록(다음 주기)으로만

### 7-10. 잡음·위양성·검정력 병기 (논문 표에 같이)

- off 시드 범위 1.8 pp(43.5–45.3)
- 같은 trunk·같은 설정 재학습 짝차 4.1/5.8/12.4 pp(1차 조건, RMS 8.2)
- **offr − off 짝차 3개(이번 배치·새 코드) = 재학습 잡음의 이번 배치 추정치** — 위 1차 조건 값과 병기. 판정 문턱은 안 바꿈. offr 가 off 대비 강화 기준을 통과하거나(더 좋음) off 가 offr 를 이기면 → '같은 설정 재학습만으로 기준 통과 수준 차이' 를 모든 판정 문장에 병기
  - offr 가 x_off 와 같은 코드 경로라는 전제의 근거 = 새 골든 `ext_v1_off`(9-3) + 분기 재개 대조(9-33, v1 OFF 갈래가 7013a3a 와 비트동일)
  - **offr 판정 연결(§11-31 저자 결정 대기, 권장 = 채택)**: offr vs off 가 어느 방향이든 강화 기준 통과면 G-C2 헤드라인에 C6 vs offr 강화 기준 통과를 추가 조건으로 붙임. 이유 = 신규 팔은 전부 이번 배치(새 코드·이번 GPU), off 는 배치 X 것 → 배치 간 차이가 C6 vs off 를 부풀릴 수 있음. 앵커는 그대로 off(교체 아님). 통신에 불리한 쪽으로만 작동
- MDE ≈ 6 pp(3시드 짝차, 직전 스펙 §6)
- 3/3 부호검정 단측 p = 0.125 → '유의' 표현 금지
- 1회 붕괴 선례: G6 off_s45 57.1, onl6g1_s45 57.5
- eval 표본오차: 에피소드 수만 기준 이항 SE ≈ 0.3 pp(선박 간 상관 무시 — 실제는 더 큼, 추정)
- 절제 Δ 는 같은 ckpt 라 학습 잡음이 없지만 의존도지 이득 아님(예: ons6 msgzero +17.7 인데 ons6−rand 짝차 s44 1.9·s45 0.5)
- 시뮬(무효과·효과 모두 정규 가정, off 를 짝으로 공유, 짝차 SD σ_d, 반복 200k/100k. 스크래치패드 `rv/fpr.py`·`rv/h1a_8.py`·`rv/fpr2.py`)

| σ_d 가정 | 근거 | 강화 기준 위양성(비교 1개) | k 3개 중 하나 통과 | C6 또는 A6 중 하나 통과 | 검정력 효과 4 / 7.6 / 10 pp | H1a FAIL 팔 1개 / 8팔 중 ≥1 |
|---|---|---|---|---|---|---|
| 3.2 | 앞 검토 하한 | 0.05% | 0.16% | 0.11% | – | 9.0% / 36.2% |
| 4.5 | 앞 검토 상한 | 0.9% | 2.5% | 1.8% | 0.20 / 0.70 / 0.92 | 11.0% / 42.2% |
| 6.0 | 중간 | 3.3% | 8.2% | 5.9% | – | – |
| 8.2 | 재학습 짝차 RMS | 6.6% | 15.7% | 11.7% | 0.24 / 0.49 / 0.66 | 12.3% / 45.8% |

- 효과 7.6 pp = ons6 평균 이득 크기. 검정력 0.49–0.70 → 진짜 효과가 있어도 놓칠 확률이 큼(결과 전 기록). C 의 기대 효과는 ons6 보다 작을 가능성(§7-11) → 검정력은 이 표보다 낮을 수 있음

### 7-11. 결과 전 예측 기록 (판정에 안 씀, 전부 추정)

- 정보 순서 **P0 ≤ C2 ≤ C6 ≤ C12 ≤ C∞** 는 설계상 기대. 코덱이 k 별로 따로 학습돼 엄밀한 포함관계 아님. **성능 순서 보장 아님**
- **C 는 A 보다 성공 확률이 낮을 수 있음**(09-28 갱신)
  - 청자가 RL 신호만으로 z 해석·Δψ 회전·CPA·역할을 분기 후 약 7M 결정 안에 배워야 함. sup 에서 3×256 도 정답 라벨 2e6 에 76–78%, 2e5 에 1–50% → RL 로는 더 어려움
  - 학습량 수치(계산값): 분기 뒤 7,012,352 결정 = 107 update × 2 epoch × 128 미니배치(512) ≈ **27,400 step**, 3×256 청자는 처음부터 이 안에 배워야 함. sup 는 정답 라벨로 2e6 = 약 1,950 step·2e7 = 약 19,500 step(배치 1024). step 수는 RL 쪽이 많지만 신호는 advantage 가중 정책 gradient 가 value 6D·gate·fc2 를 거쳐 오는 간접 신호 + 워밍업 1200 결정은 om 0 → 같은 정확도를 기대할 근거 없음(추정)
  - wf3 주관 추정(3/3 승 기준, 청자 1×64): A(k≥3) 0.7–0.9, C 0.25–0.45(회복률 10–60%). **이번 강화 기준(6 pp)·청자 3×256 에 맞춘 새 확률 수치는 만들지 않음**(근거 없음). 방향만: 강화 기준이라 둘 다 wf3 값보다 낮을 것, 3×256 이 C 쪽 격차를 줄일 수는 있으나 크기 모름
  - 같은 이유로 A6 vs C6 는 'A 이김' 쪽 가능성이 더 높다고 봄
- C∞ 가 off 를 이길 가능성은 C6 보다 높음(정보 무손실). 단 ons6(명시 쌍별 필드) 37.1 수준은 기대 안 함(청자 학습 부담). 배치 X 선례 arpa6(56 m 참값 필드)은 off 를 못 이김 → C 의 이득은 레이더 밖 파트너 정보에서 나와야 함
- C6 vs C12: 구별 안 될 가능성 높음(3시드)
- C2 vs T2: 방향 모름. T2 는 운동 정밀도가 높지만 ψ 불연속, C2 는 25D 를 2칸에 넣어 운동 정밀도를 잃음(6D 탐색 k=2 λ1 \|dv\| 중앙 0.226 m/s·역할 붙은 쌍 뒤집힘 37–69%) → 약하게 T2 ≥ C2 쪽
- P0·R6 는 off 와 비슷하거나 약간 나음(rand 42.2 선례). 반대로 H1a FAIL 위험도 있음(위)
- C∞ 의 goal·sit·threat 부분 기여(p_goal0·p_sit0·p_threat0)는 작을 것(좌표 변환 학습 부담, oni6 intent0 +0.3 선례)
- C(k) > C∞ 가 나오면 **'압축이 도움' 주장 안 함, 원인 미상**(잡음, 좁은 입력이 청자에게 배우기 쉬움 등 후보만 적음 — 원인을 결과 전에 단정하지 않음)

### 7-12. 결과별 주장 문장 템플릿 (결과 전 고정)

- 공통 범위 문구(직전 스펙 §6 이식): "radar-only 학습 정책이 무너지는 regime(imo + 레이더 56 m, 16척)에서 공유 X 가 ground-truth 를 Δ 만큼 바꿈"
- G1: 직전 스펙 §8-4 G1' 판정을 그대로 물려받음. 미채택이면 'G1 FAIL regime' 을 문장에 병기
- 공통 병기: 비트 표기(§3-3), 시드별 값·승패 수, '통신 경로(위치 공유·수신자 자기 상태 토큰·청자 용량 포함)' 문구(§5-4), offr 잡음(§7-10)

- 표 규칙(09-28): **'…' 로 앞 줄 문장을 물려받지 않음** — 줄마다 문장 전체를 적음. 'latent 가 주어인 문장'은 G-C3 ① 통과 줄(1·2행)에만 있음(§7-4 안 (a))

| 결과 | 문장 |
|---|---|
| G-C1·G-C2 통과, G-C3 ①·② 모두 통과 | "송신자 자기 값 25D 를 동결 과제 인지 코덱으로 6차원(6×8 bit, + 위치 무손실) latent 로 보내고 받는 배가 그 z 를 직접 읽도록 학습한 통신이 같은 trunk OFF 대비 vColl 을 Δ pp 낮춤(3/3). 이득은 위치·자기 상태 토큰(P0)·짝 끊은 z(R6) 대비로도 유지 → z 의 송신자별 내용에 귀속" |
| G-C1·G-C2 통과, G-C3 ① 통과·② 미통과 | "송신자 자기 값 25D 를 동결 과제 인지 코덱으로 6차원(6×8 bit, + 위치 무손실) latent 로 보내고 받는 배가 그 z 를 직접 읽도록 학습한 통신이 같은 trunk OFF 대비 vColl 을 Δ pp 낮춤(3/3). 같은 토큰에서 z 만 뺀 대조(P0) 대비로도 이득이 있으나, 송신자별 z 내용인지 z 입력 추가(주변분포) 효과인지는 가르지 못함" |
| G-C1·G-C2 통과, G-C3 ① 미통과(② 통과 여부 무관) | "위치 공유 + 수신자 자기 상태 토큰 + 학습 청자(3×256) + 6차원 코덱 z(6×8 bit)로 된 통신 채널이 같은 trunk OFF 대비 vColl 을 Δ pp 낮춤(3/3). **z 기여 미확인** — 같은 토큰에서 z 만 뺀 대조(P0) 대비 판정 불가라, 이득이 위치·자기 상태 토큰·청자 용량에서 왔을 수 있음." ② 통과면 "짝 끊은 z(R6) 대비 이득 방향은 있음(R6 자체가 해로웠을 가능성 병기)" 추가 |
| G-C1 통과, G-C2 실패 | "무손실 P 를 직접 읽는 학습 청자(C∞)는 이득이 있으나 k=6 코덱 z 를 읽는 채널에서 유지 안 됨" + G-C5 줄 + 부 비교는 '부 비교(다중 비교 보정 없음)'로만 |
| G-C1 실패 | "학습 청자가 무손실 P 를 읽어도 이득이 이 재학습에서 재현 안 됨. C 계열 주장 없음" + C6 가 통과했으면 '상한 미통과 상태의 C6 이득 — 원인 미상'(헤드라인 금지) + A6 결과는 부 비교·읽기 진단으로만 |
| 전부 실패 | "이 설정에서 동결 코덱 z 를 쓰는 통신은 OFF 를 이기지 못함(안 도움)" |

- 1–3행 공통: offr 조건(§7-10, §11-31 채택 시)이 걸리면 'C6 vs offr 도 통과' 를 문장에 붙이고, 못 통과면 헤드라인 없음('판정 불가(재학습 잡음 수준)')
- G-C3 판정이 붕괴 시드에 걸려 있으면 §7-8 규칙(판정 불가로 낮춤)이 먼저
- G-C4·G-C5 는 위 문장 뒤에 서술로 붙임(§7-5 표 문구 그대로, 필수 병기 포함)
- 금지 문구: '유능한 기준선 대비 통신 이득', '현실 선박 일반화', '유의', '동등', **'emergent'**, 'learned communication'·'메시지 학습'(단독 — 화자는 학습 안 됨), '16 bit 통신' 단독, 부 비교의 헤드라인화, **G-C3 ① 미통과 때 'latent 로도 이득'·'6차원 latent 통신 이득'**

### 7-13. 보고 의무

- 진 팔·모든 지표 포함 전부 보고. 조건이 다른 결과를 한 평균으로 합치지 않음
- 이전 결과(같은 표, 순서대로)

| 결과 | 조건 | 수치 |
|---|---|---|
| 1차 파일럿 | EXT=0, aux 0.05, crossing 2 | on6 vColl 50.5/51.0/52.0 vs off 47.2/48.4/44.7 → 3/3 악화 |
| 1차 RANDOM | 같음 | vColl 53.7/47.7/50.7 |
| G6-2 | 1차 trunk, aux 0, crossing 2 | vColl on6a0 41.3/53.6/52.1 vs G6 off 43.1/42.6/57.1. goal on6a0−off +1.0/−13.4/+2.2(평균 −3.4, 2/3 ≥) |
| 배치 X onl6 | EXT v1, aux 0, crossing 0 | vColl 50.4, 0/3, +5.7 pp → **G6(H1a) FAIL 유지** |
| onl6 msgzero | 같은 ckpt | vColl 49.2/43.3/47.5(평균 46.7) |
| 후속 A onl6g1 | token gain 1 | vColl 44.4/45.1/57.5(49.0) |
| onl6g1 msgzero | 같은 ckpt | vColl 43.1/40.9/50.4(44.8) |
| 후속 B rand | RANDOM sd 0.20 | vColl 42.2, off 대비 2/3 |
| ons6 latent0 절제 | 같은 ckpt | ΔvColl −3.4/−1.4/−4.9(3/3 개선) |
| ons6·oni6 | EXT v1, latent 1 | 37.1 / 30.9 (3/3) — 이번 앵커 아님 |

- 판정 문구: 'RL 만으로 학습된 latent 채널은 이 regime 에서 H1a 위반(사전등록 G6 FAIL)' 유지
- 설계 시점(09-27 결과 뒤)·채널 변형 주기 수(3번째)·§0-3 탐색 이력·**09-28 A→C 전환(학습 전, 설계안 2개 중 C 채택)** 명시. 이번 결과는 이후 보고에 앞 주기 결과와 병기

---

## 8. 한계·정직성

### 8-1. 명칭

- 논문 명칭: **'autoencoder-grounded latent communication (동결 화자 코덱 + 학습 청자)'** — 한국어 '오토인코더 grounded latent 통신'
- 코덱 명칭: '과제 인지 코덱(task-aware codec)'. 'pair-feature-weighted' 병기 여부는 §11-9
- 본문 필수 문장: "화자(송신) 쪽 z 의 뜻은 오프라인에서 한 번 학습해 동결한 코덱이 정함. '과제 인지'는 설계자가 정한 쌍 특성(COLREGs 쌍별 필드) 충실도를 코덱 손실에 넣었다는 뜻이며, 과제 보상으로 메시지를 학습한 압축(IMAC 류)이 아님. 받는 배의 attention k/v 가 그 z 를 읽는 법을 RL 로 학습함"
- **'emergent' 금지**. 'learned communication'·'메시지 학습' 단독 금지(화자는 학습 안 됨). 쓸 수 있는 말 = '학습 청자(learned listener)', '동결 화자 코덱(frozen speaker codec)'
- 'latent' 는 '동결 코덱 코드 z'라는 뜻으로만
- their_role → '상대 관점 기하 역할(수신측 계산)'(A6·코덱 학습에만 해당). 역할 명칭은 직전 스펙 §8-3 저자 결정 그대로

### 8-2. 이상화

- 지연 0, 패킷 손실 0, 매 결정(0.4 s) 갱신 — AIS(2–10 s)·VHF 보다 좋음
- 송신 값·수신자 own4 는 시뮬레이터 참값(계기 잡음 0)
- 위치는 코덱 밖 float32 무손실
- **전 선박 위치 방송 가정**: 파트너 선택 topi 는 300 m 안 모든 배의 참 거리 cdist 로 뽑음(vessel_gym_train.py:289-297). pmask(파트너 수)도 참 거리 유래
- **sit 중계**: 송신자 sit 는 56 m 안 타선 참 운동학 유래(§2-2) → C∞(직접)·C(k)·A6(코덱이 남긴 만큼)에서 수신자에게 56 m 밖 제3선에 대한 특권 정보가 넘어갈 수 있음. OFF obs 의 자기 sit 도 같은 성격(전 팔 공통 이상화)
- 중앙 critic 전역 상태(build_global_feat, vessel_gym_train.py:225 — 모든 배의 참 위치·침로·속도비): 배치 X central_critic=True. 학습 전용·전 팔 공통, 실행 정책 입력 아님
- gym 전용(Unity 판정관 미러 없음)

### 8-3. ORACLE 분류 ([S] 송신자·수신자 자기 계측 / [P] 특권 참값 또는 그 유래)

| 입력 | 누가 가진 값 | 분류 | 비고 |
|---|---|---|---|
| P 운동 ψ·SOG·ROT | 송신자 자선 계기(env 참값) | [S] | 잡음 0(이상화). SOG 는 정책 obs 에 없지만 자선 계기값 |
| P 명령 cmd 2 | 송신자 자기 지령(t−1) | [S] | |
| P goal | 송신자 자기 obs(자기 위치·목표) | [S] | |
| P threat | 송신자 자기 레이더 frame stack | [S] | |
| **P sit** | 송신자 obs 이지만 `_pairwise` 가 56 m 안 타선 참 heading·speed 로 계산 | **[P] 유래** | arpa6 필드([P] 이상화 레이더 추적)와 같은 기준. OFF obs 에도 있음. 중계 시 제3선 특권 정보 전달 |
| **수신자 own4(토큰)** | 수신자 자선 계기 | [S] | SOG_i 절대값은 obs 에 없고 토큰으로만 들어옴 — P0 포함 cz 전 팔 공통(§2-5·§5-4) |
| T2 ψ·SOG | 송신자 자선 계기 | [S] | |
| 파트너 위치(relpos) | 송신자 GPS 가정, 무손실 | [S] 이상화 | 코덱 밖 |
| 파트너 선택(300 m nearest-4)·pmask | env 참 거리 | [S] 환원 가능 | 전 선박 위치 방송 가정하에 |
| A6 의 수신자 참값(`_pair_core` 의 pos_i·ψ_i·SOG_i) | 수신자 자선 계기 | [S] | 절대 SOG 는 항등식으로만(§5-4) |
| C∞ 의 P | 위 성분 무손실(sit [P] 포함) | 상한(ORACLE-ref) | 처치 주장에 안 씀, 상한으로만 |
| arpa6 필드(56 m) | env 참값을 레이더 반경 안에만 | [P] 이상화 레이더 추적 | 잡음·지연 0 |
| critic global_feat | 모든 배 참 위치·침로·속도비 | [P] | 학습 전용·전 팔 공통·실행 입력 아님 |
| 코덱 학습 데이터 | OFF trunk rollout | 오프라인 | 처치 정책 안 봄 |
| 코덱 쌍 항 목표값의 수신자 참값 | 오프라인 데이터 | 오프라인 | 실행 경로 입력 아님(§3-4) |
| USE_ORACLE(미사용) | env 평균 goal 주입 | [P] | 이번 팔에 없음 |

- ORACLE 재정의 채택 여부와 sit [P]·own4 [S] 분류 확인은 §11-10. 채택이 결과 뒤라는 시점 명시

### 8-4. 약속 가능 / 불가

- 요청 원문(wf3 기록): **"latent가 무조건 의미있어지는 설계"** → 뜻을 셋으로 나눔
  - (a) **되돌려 읽을 수 있음(z ↔ 송신자 물리량)** — 보장함. 코덱 동결이라 대응이 정의상 고정(D 로 복원). 단 충실도표 범위만큼이고, k=2 는 25D 중 운동 위주만 복원됨
  - (b) **충돌 감소 이득** — 보장 못 함
  - (c) **학습으로 생긴 의미** — 화자 쪽은 해당 없음(설계 코덱). **청자 쪽 '읽는 법'은 학습됨 = C 의 주장 대상**. 청자가 z 를 실제로 쓰는지는 msgzero·slot-shuffle·R6 개입으로만 보임(Lowe 2019)
- **약속 가능**
  - RL 이 코드를 해로운 채널로 빚는 경로 차단: no_grad, LATENT=0, aux 0 → z 의 뜻은 RL 중 불변
  - 구조적 미러, OFF·v1 경로 비트동일, 스냅샷 SHA·팔 구분 표시(§4-8)로 '조용히 다른 실험' 방지
  - P0·C∞·R6·T2·A6 로 양쪽을 막아 어떤 결과든 해석 문장이 나옴(§7-12 템플릿)
  - 사전등록한 팔·지표 전부 보고(onl6·1차·G6·A/B·§0-3 탐색 포함). 결과 뒤 k·시드·지표·청자 구조 선택 없음
- **약속 불가**
  - 어떤 C(k) 든 off·P0·R6 를 이긴다는 것
  - C6 vs off 통과만으로 '적은 차원 latent 로도 이득'(z 기여는 C6 vs P0 로만 — §7-4)
  - 청자가 z 에서 어떤 물리량을 읽는지(기전) — 서술만
  - 청자 용량 3×256 이 최선이라는 것(측정 2점 중 택함)
  - 3시드로 k 곡선 모양·꺾이는 점·통계적 유의성
  - 'emergent'·'learned communication' 명칭
  - '과제가 요구하는 전송량(rate)' 주장 — 설계자가 정한 왜곡 척도의 rate–distortion 임
  - k=6·12 에서의 '압축 가치' 주장(k=2 만 시험)
  - 유능한 기준선(G1 FAIL 그대로), 실선·지연 있는 AIS 로의 일반화
  - onl6 H1a FAIL 을 소급해 지우는 것
  - 결과를 본 뒤 '최적 k' 고르기, A6 를 확증으로 올리기

### 8-5. 3시드 검출력

- MDE vColl ≈ 6 pp. 짝차 SD 는 두 가정(앞 검토 3.2–4.5, 재학습 RMS 8.2)을 §7-10 표에 병기 + offr 추정치
- 판정 가능한 건 사실상 'C(k)/C∞/A6 vs off' 뿐(ons6·oni6 폭 7.6·13.8 pp 선례)
- 인접 k 사이 2–5 pp 차를 가르려면 k 마다 시드 10–35개(앞 검토 추정) → k 곡선 모양은 주장 안 함. C(k) 는 교차평가 불가(§6)라 의존도도 slot-shuffle 로만
- 3/3 규칙만으로는 약한 효과(rand 수준 Δ 2.5 pp)도 3/3 확률 약 0.36(σ_d 4.5 가정, 앞 검토) → 강화 기준과 P0·R6 대조가 필요한 이유

### 8-6. 리뷰어 예상 질문과 답 위치

| 질문 | 답 | 위치 |
|---|---|---|
| ① k=6·12 는 운동·명령(자유도 5)에 사실상 무손실인데 '적은 차원'인가 | 25D → 6D 전송은 차원 축소임. 압축 가치(원값 대비)는 k=2(C2 vs T2)에서만 시험. C6 는 '6차원 latent 청자 통신 이득'으로만 주장하고, 그 문구도 C6 > P0 통과 때만 | §7-4, §7-5 G-C4, §8-4 |
| ①′ C6 vs off 이득이 z 덕분인지, 위치·자기 상태·큰 청자 덕분인지 | 확증 비교는 채널 전체 효과라고 명시. 'latent' 문구는 C6 > P0(같은 토큰·shape, z 만 0) 통과 때만 허용 | §5-4, §7-4, §7-12 |
| ② 무너진 기준선(G1 FAIL, OFF 충돌 45%) 대비 이득 아닌가 | 맞음. 주장 범위 문구로 regime 을 한정, '유능한 기준선 대비' 금지 | §7-12 |
| ③ 결과 보고 설계했나, 주기를 반복했나, A→C 를 왜 바꿨나 | 설계 시점(결과 뒤)·주기 수(3번째)·사전 탐색·09-28 전환(학습 전) 전부 공개, 잠금 순서 고정 | §0-0, §0-1, §0-3, §0-6 |
| ④ 이득이 z 정보 때문인가, 입력 추가·잡음 정규화 때문인가(rand 42.2) | P0(같은 토큰·shape, z 0)·R6(짝 끊은 z) 대조. 수신자 절대속력 경로는 P0 에도 있어 상쇄 | §5-4, §7-5 G-C3 |
| ⑤ 위치 float32 무손실 공유라 bit 수가 무의미하지 않나 | bit 표기에 위치를 항상 병기, 단독 표기 금지. P0 가 위치만의 효과 | §3-3 |
| ⑥ 3시드 검정력 | 위양성·검정력 두 가정 + offr 병기 | §7-10, §8-5 |
| ⑦ 대조 팔이 붕괴하면 통신이 공짜로 이기지 않나 | 붕괴 표시·'대조 붕괴 의존' 병기 규칙 | §7-8 |
| ⑧ 통신 팔 청자(3×256)가 커서 이긴 것 아닌가 | P0·R6 가 같은 용량. 확증 문장에 '청자 용량 포함' 병기 | §4-3, §4-4, §5-4 |
| ⑨ 왜 손 공식(A) 대신 학습 청자(C)인가, C 가 A 보다 못하면 | 같은 z 로 A6 를 둠. 해석표 결과 전 고정('둘 다 짐'·상대 H1a FAIL 을 먼저 봄) | §7-5 G-C5 |
| ⑨′ 신규 팔은 이번 배치, off 는 옛 배치 — 배치 차이 아닌가 | offr(같은 설정 재학습)를 이번 배치에 둠. offr 가 off 와 6 pp 이상 다르면 C6 vs offr 통과도 요구(§11-31 채택 시) | §7-10 |
| ⑩ C2 vs T2 는 표현(ψ 불연속) 차이 아닌가 | 결과 전 기록, 해석문 병기 | §3-10 |
| ⑪ Lin et al. 2021 과 무엇이 다른가 | 동결·오프라인·과제 가중 화자, 저차원 자기 상태, 쌍별 청자 토큰 | §12 |

---

## 9. 테스트·검증 목록

- 테스트용 코덱은 고정 seed 로 만든 **합성 코덱**(실제 코덱은 §0-0 5단계 뒤에 나옴). 실제 코덱 3개는 커밋 뒤 SHA 재계산·충실도 검사로 따로 확인

1. **쪼개기 동일성**(A6·코덱 학습용): 쪼개기 전후 `comm_pair_features` torch.equal — 기존 test_comm_ext 상태 + 임의 조밀 상태(E=4 N=16, **홀수 크기 E=3·N=7·K=3 포함**, imo·agile, 정지·후퇴·패딩·반경 경계). 쪼개기 전 코드 출력 해시 파일과 대조
2. 골든 5케이스 `verify/test_golden.py --check` PASS(기본·legacy 경로 비트동일). **이 5케이스는 전부 EXT=0·agile·grid3x3·처음부터 2 update**(test_golden.py:56-66) → off·arpa6·offr 재사용 전제는 9-3·9-33 이 따로 덮음
3. **[필수] 새 골든 `ext_v1_off`**(EXT=1·imo·none·crossing 0·OFF = x_off·offr 실제 설정) + (저자 승인 시, §11-11) `ext_v1_intent_ON`(EXT=1·imo·none·intent)
   - **7013a3a 별도 체크아웃 코드로** 생성(§0-0 3단계). 골든은 플랫폼별(test_golden.py:123-131) → **Mac·Windows 각각**, `--regen` 은 새 케이스만. 이후 새 코드에서 `--check`
   - 케이스 env = x_off_s43 스냅샷 env_lines 와 같은 값(EXT·DYN_PROFILE·OBSTACLES·CROSSING·레이더 56·MSG_DIM 6 등). 차이는 규모(envs 8·steps 16384)뿐. `run_case` 가 바깥 VESSEL_* 를 지우므로(:88) 케이스 env 에 전부 적음. `--crossing 0` 등 인자가 필요하면 CASES 에 `args` 필드를 추가(기존 5케이스는 args 없음 → 불변)
4. 기존 `verify/test_comm_ext.py` 전부 PASS
5. **k/v 토글**: KV (64, 1)(기본)이면 GroundedAttention 생성 코드 경로·state_dict 키·초기값이 7013a3a 와 비트동일(모듈 단위 대조 — 학습 루프·재개 경로는 안 봄 → 9-3·9-33 이 맡음). (256, 3) 이면 키 attn.k_proj.{0,2,4,6}·v_proj.{0,2,4,6}, v 마지막 층 ×0.1·bias 0, 순전파 유한
6. **cz·v2p 구조**: §4-4 표대로 relpos_dim·토큰 폭, v1 기본값 불변, 키 이름 규칙
7. **가용성·국소성**
   - own_payload 가 (env, x, goal, sit) 만 받음(미래 버퍼 접근 불가)
   - P[0:3] == (sin, cos)(env.heading·DEG)·env.speed/1.8 비트 단위, P[3] == obs[363], P[6:8] == obs[360:362], P[8:13] == one_hot(obs[368]), P[13:25] == compute_own_threat(x)
   - **행 국소성**: 배 k 의 heading·speed·rudder·cmd·max_speed 만 바꿨을 때 P[:, j≠k, 0:6]·own4[:, j≠k] 불변(타선 참값을 읽는 실수 차단)
   - 덤프 도구가 env.step 전 시점에 부름: P[4:6] == 직전 행동의 명령(_apply_action 식), 재스폰 행 == 재스폰 값
8. **C 토큰 조립**: cz 팔 ext[..., W:W+4] == 수신자 행의 own4(모든 유효 슬롯), own4_i == P_i[0:4] 비트 동일, 패딩 슬롯 0. slot == 팔 규칙(codec: q(E(norm P_j)) gather / true: P_j / trunc2: 격자값), P0 slot 전부 0. cz 에서 `ext_shuffle_gen` 을 주면 RuntimeError
9. 양자화: 실행 결정론, 256 단계, 오차 ≤ Δ/2. 잡음은 코덱 학습 모드에서만. T2 도 같은 격자
10. 복호 규칙: P̂ = P(항등) 이면 운동 필드 allclose(atol 1e-5), 역할 일치율 기록(비트동일 아님 명시). (0,0) 침로·clamp 경계·threat 미감지 규칙에서 값 유한
11. 동결: 코덱 파라미터가 policy.parameters()·Adam·clip 그룹에 없음, 출력 requires_grad False, 코덱 유무로 policy state_dict 키 불변. 코덱이 지정 device 에 있음. C 팔 청자 loss 의 grad 가 코덱에 0
12. 미러: `_verify_comm_mirror` CASES·`test_mirror_all_arms` 에 cp0·cc2·cc6·cc12·ccinf·ct2·cr6·ca6 추가, 기존 \|Δlogp\| 문턱, ALL PASS
13. **slot-shuffle·R6**: env 안 유효 z 모음 보존, 같은 송신자 z 를 받는 항목 0, fallback env 는 슬롯 0, j′ = i 빈도 기록, **relpos·own4 열은 안 움직임**, 전역 RNG 상태 전후 동일, 재개 재시드 재현. **SHUFFLE=1 이고 gen None 이면 §4-2 표의 generator 연결 호출부 9곳 모두 RuntimeError**(미연결 호출부는 9-41). 텔레메트리 on/off 가 G_train 난수열을 안 바꿈
14. **분기 재초기화(일반화)**: 작은 v1·1×64 OFF trunk → cz2·cz6·cz12·cz25·v2p × 3×256 분기에서 화이트리스트 모듈만 새 값, 나머지 torch.equal, 화이트리스트 밖 키·shape 차이 → SystemExit, 화이트리스트에 Adam state 가 있으면 SystemExit(합성), **Adam param_groups 객체 id == policy.parameters()**(모듈 교체 금지 확인), **같은 shape 팔끼리 초기값 SHA 동일(cp0 = cc6 = cr6, cc2 = ct2 — 코덱 난수 격리 포함)**, 스냅샷 branch_reinit 기록, 크래시 재개가 reinit 키 상속, 분기점 아닌 재개에서 layout·kv 불일치면 거부. **실제 trunk 3개** Adam state 확인(Windows)
15. **Adam 이름 기준 재배치**: 옮긴 파라미터마다 exp_avg·exp_avg_sq·step 이 trunk 의 같은 이름 텐서와 torch.equal, 화이트리스트 state 없음, 골격 생성 전후 전역 RNG(CPU·CUDA) 동일, 첫 step 에러 없음, 옮긴 개수 == trunk state 개수. **재배치 뒤 networks·config 전역이 이번 런 값 그대로, comm_gather 출력 폭 불변, `policy.comm_sig` 검사 통과**(골격 전역 저장·복원, §4-5). 골격 생성 중 예외가 나도 finally 복원(합성 예외로 확인)
16. ckpt_io: 코덱 blob 왕복, 스냅샷에 SHA 있는데 blob 없음·SHA 불일치 → SystemExit, legacy ckpt → v1·payload 없음·kv (64,1)(스니핑), eval 이 blob 으로 복원, CLI override 가 notes 에 남음. **한 프로세스에서 코덱·3×256 ckpt → 비코덱·1×64 ckpt 순서로 열어도 COMM_CODEC=None·payload=''·kv 리셋**. **같은 폭 팔 순서 복원 cr6→cc6·cc6→cp0**: 두 번째 복원 뒤 COMM_SLOT_SHUFFLE=0·fields 가 맞게 바뀌고, 전역을 일부러 옛 값으로 되돌리면 comm_gather 가 `comm_sig` 불일치 RuntimeError(§4-2). x_arpa6 형 합성 ckpt(스냅샷에 comm_kv_* 없음, k_proj.{0,2})가 스니핑으로 (64, 1). slot·own·p_*·goal·sit·threat 그룹 override 허용
17. 재개 가드: 새 키 포함 `_cur_comm` 불일치 → 거부, 크래시 재개가 코덱 SHA·kv 유지
18. check_branch: 신규 9팔 + 재사용 x_off·x_arpa6 이 같은 trunk SHA 묶음에서 ALL PASS(재사용 arpa6 의 v1·64×1 이 kv 동일 규칙에 안 걸림 — 대상 = layout cz*·v2p). layout·kv 다른데 branch_reinit 없는 갈래 → FAIL. cz*·v2p 갈래끼리 kv 다르면 FAIL. 복사한 x_off·x_arpa6 곡선 CSV 로 검사 5 가 note 없이 돎
19. 코덱 학습 재현: 같은 데이터·seed·플랫폼 → 같은 내용 SHA. 데이터 SHA 기록
20. soft cascade: τ 고정값에서 hard 역할 일치율을 train 쌍으로 측정·보고. 연속 필드는 hard core 와 allclose
21. preflight: 추적 파일인 `comm_codecs/*.pt` 변경과 run_repro.sh 의 코덱 SHA·kv 핀 변경이 지문을 바꿈. (팔별 env 는 common_env 만 수집하는 지문에 안 들어감 — preflight_checks.sh). `VESSEL_EVAL_ARMS`·`REUSE_ARMS`·`DRY_RUN` 만 바꾼 train→eval 전환에서 지문 불변(캐시 적중)
22. Unity 경로: cz·v2p 체크포인트로 `_get_others_msg` 호출 시 RuntimeError
23. cz·v2p + LATENT=0 에서 msg_override·latent_zero 가 결과에 영향 없음
24. Windows preflight 1회(`_verify_ppo_mirror` 는 Windows 전용)
25. 재사용 팔 재평가 2단계 일치(G-C0)
26. **end-to-end 스모크**(cc6·ccinf·ca6 각 1회): trunk 복사본 → 재초기화·Adam 재배치 분기 → `ckpt_every` blob 저장 → eval → check_branch. k/v grad norm 0 아님·NaN 0 확인(기록만). **Windows 에서 ON 갈래 1개의 VRAM(torch reserved·nvidia-smi 델타)·update 당 시간 실측 → §10 에 채움**. VRAM 이 run_repro `VRAM_TRAIN_ON=4800`(:257)을 넘으면 그 값을 실측으로 갱신(OOM 방지 — 결과 영향 없음)
27. **팔 구분**: 11팔 `_variant`·header·meta 문자열이 서로 다름
28. `_THREAT_ANG` device: CPU 호출 뒤 GPU 호출(가능한 플랫폼) 에러 없음, 값 불변
29. traj 재구성: 합성 env 로 만든 traj 에서 §3-6 규칙(cmd[t−1]·재스폰 제외)으로 만든 P[4:6] 이 own_payload 와 같음
30. 앵커 보호: 배치 스크립트 dry-run(`VESSEL_DRY_RUN=1`, 새로 구현 — 7013a3a 에 없음)에서 `$CK/x_off_s*.pt`·`x_arpa6_s*.pt` 에 쓰기 없음, offr 는 `x_offr_s*.pt` 로, 새 OUT 사용. 재사용 파일 9개 중 하나를 치우면 배치가 시작 전 중단
31. `decode_p25` 공용: C∞ 교차평가 'recon' 슬롯 == A6 의 Ph_ent(같은 q 에서) 값 동일
32. T2 슬롯: ψ_n == 격자(wrap180(heading)/180)(누적각 입력 포함), ±180 경계 유한, v_n 범위 [−1, 1]
33. **[필수] 분기 재개 대조(7013a3a vs 새 코드, 09-28 구현 검토 blocking)**
   - 입력: 7013a3a 로 만든 작은 v1·1×64 OFF trunk 1개(EXT=1·imo·none·crossing 0, envs 8, 1 update, 고정 seed, CPU)
   - 갈래 둘: ① v1 OFF(offr 형) ② v1 ON fields state·partner_range 56·LATENT 0·AUX 0(arpa6 형), 각각 `--resume <trunk> --resume_at <trunk steps> --comm_on_at <같은 값> --resume_warmup <작은 값>` 으로 2 update
   - 같은 trunk 파일로 **7013a3a 체크아웃과 새 코드에서 각각** 돌림 → state_dict SHA(test_golden `_sha` 방식)·Adam state(exp_avg·exp_avg_sq·step)·곡선 CSV·aux CSV·스냅샷(새 키 제외) **비트동일**
   - 새 코드 쪽 스냅샷에 branch_reinit·branch_adam_remap 키가 없어야 함(발동 조건 §4-5 — 같은 shape 갈래에서 no-op)
   - Mac 필수, Windows 는 preflight 때 1회. offr 가 x_off 와 같은 코드 경로라는 전제(§5-4·§7-10)의 근거
34. 텔레메트리·diag: `comm_telemetry`·`eval/diag_ckpt.py` 가 cz2·cz6·cz12·cz25·v2p 합성 정책에서 예외 없이 돌고 새 열(act_zero_slot·act_zero_own, v2p 그룹 열, cdc_*, cr6 의 r6_fb·r6_self)이 생김. 옛 열 순서 불변
35. **절제·교차 CLI 스모크(Mac, 작은 합성 ckpt)**: §6 표의 조합 전부 — msgzero, slot-shuffle(cc2·cc6·cc12·ccinf·ct2), z-shuffle·field-shuffle(ca6), 그룹 0(slot0·own0·p_*0·state0·role0·intent0·goal0·sit0·threat0), recon 교차(ccinf × 합성 코덱 k 2·6·12, `--codec/--codec_sha`), A6 true 교차 — 각 1회 끝까지 돌고 header·notes 에 기록. 지금은 27 GPU-h 뒤에야 처음 실행되는 경로임
36. **R6 평가가 실제로 섞음**: cr6 합성 ckpt 를 eval 로 열면 복원된 COMM_SLOT_SHUFFLE=1 + G_eval 로 섞임. 같은 상태에서 섞기를 강제로 끈 경로와 om 이 다름(섞기 없는 C6 로 평가되는 사고 방지), 같은 G_eval seed 두 번이면 om 동일
37. **CLI override 순서**: restore_policy 가 스냅샷으로 전역을 리셋한 **뒤** CLI(`--payload_override`·`--codec`·`--slot_shuffle`·그룹 0)가 설정됨. 순서를 뒤집은 가짜 경로에서는 CLI 값이 사라짐을 확인(테스트가 순서를 실제로 잡는지)
38. **새 테스트의 env 고정**: 새로 만드는 테스트 파일은 import 전에 새 env 키(LAYOUT·PAYLOAD·CODEC·CODEC_SHA·CODEC_BITS·SLOT_SHUFFLE·KV_HIDDEN·KV_DEPTH·ARM_TAG)를 기본값으로 고정(test_comm_ext.py:18-21 방식). preflight 가 배치 env 를 export 한 채 불러도 결과 불변
39. `_verify_comm_mirror` 의 cr6 케이스가 전용 generator 를 넘김. 안 넘긴 변형은 fail-closed RuntimeError(검증기가 조용히 섞기 없는 경로를 검사하지 않음)
40. **run_repro dry-run**: `VESSEL_DRY_RUN=1` 로 train·eval 을 돌려 ① 팔 → 평가 arm 대응(off·offr → OFF, 나머지 → ON), ② 팔별 env 블록이 §4-7 글자와 같음(SHA 자리 포함), ③ 재사용 파일 존재 검사·앵커 쓰기 없음, ④ 버그 점검 [6] 의 header grep 패턴을 배치 X eval 로그에 걸어 결과 줄(vColl·goal 등)이 한 줄도 안 나옴
41. `verify/test_ckpt_compat.py`: cc6 합성 ckpt 가 기존 호출(:62·:69)로 에러 없이 돎, cr6 합성 ckpt 는 generator 없는 호출이라 RuntimeError(의도, §4-2 미연결 호출부)

---

## 10. 작업량 (전부 추정)

- Mac
  - 코드 약 1,900줄: vessel_gym 쪼개기·own_motion4 ~100, own_payload ~60, `state_codec.py`(코덱·양자화·복호·decode_p25·soft cascade·SHA·저장·device) ~270, `tools/dump_codec_data.py` ~150, `tools/train_codec.py` ~200, comm_gather cz·v2p 분기·slot-shuffle·fail-closed ~150, config ~70, networks(k/v 깊이·전역) ~40, ckpt_io(복원·kv 스니핑·리셋·header) ~130, 학습기(재초기화·Adam 재배치·재개 키·blob 저장·generator) ~180, eval_ckpt(충실도·slot-/z-shuffle·CLI override·meta) ~150, run_repro(팔·EVAL_ARMS·스모크) ~90, check_branch ~50, 테스트 ~480
  - **09-28 구현 검토: 위 1,900줄·4–5일은 낙관적** → 빠진 항목 +500–700줄: 오프라인 충실도표 생성기(거리대별·분위수·역할 일치율) + traj→P 재구성(레이더 재계산, §3-6) + z 토큰 probe(§6), run_repro dry-run 모드(7013a3a 에 없음), `_verify_comm_mirror`·test_comm_ext·test_ckpt_compat 수정 + 합성 코덱 픽스처, comm_telemetry 새 열·cdc_*·r6_*, diag·eval_mixed generator 연결, `comm_sig` 가드·골격 전역 저장·복원, 9-33 분기 대조 하네스, CLAUDE.md §4/§5/§7 갱신
  - **합계 약 2,400–2,600줄, 구현·리뷰 6–7일**. 새 골든 케이스 생성·9-33 기준 산출(7013a3a 별도 체크아웃) Mac·Windows 각 CPU 수 분 + 체크아웃 준비
- Windows (단계마다 동시 실행 가정을 적음. VRAM 은 배치 X 실측 ON(EXT) 4.5–5 GB/프로세스(run_repro.sh:36-37) 기준 — **3×256 청자의 VRAM·시간 증가는 미확인, 9-26 스모크에서 재어 이 절을 채움**)
  - 7013a3a 별도 체크아웃 준비(G-C0 1단계·골든·9-33 용): 수십 분
  - trunk 3개 Adam state 확인·데이터 덤프: 1–2 h
  - 코덱 학습 3개: CPU 30–60 min 씩(병렬 가능) + 충실도표·z probe
  - preflight 1회, G-C0 재평가 12건(eval 동시 8 가정) 약 1 h
  - 학습 27런 × 약 1 h = 약 27 GPU-h(갈래당 시간 증가 가능 — 미확인). **동시 8**(JOBS 기본 NGPU×2, GPU_CAP 기본 ⌈JOBS/NGPU⌉)이면 4 웨이브 약 4–5 h 벽시계. VRAM 이 늘면 GPU 당 동시 수가 줄어 더 걸림
  - eval 27건(1건 약 28 min, eval_x_ons6_s43 로그 실측): 동시 8 이면 약 1.6 h, 동시 16 이면 약 0.8 h
  - 절제·교차평가 102건: 배치 X 절제 51건 1 h 41 min 실측(그때 동시 약 14–16 수준)에 비례 → 같은 동시 수면 약 3.4 h, 동시 8 이면 약 7 h
- 합계 벽시계(Windows): 약 10–16 h(체크아웃·G-C0·학습·eval·절제, 동시 수 가정에 따라) + 코덱 준비(덤프·학습·충실도) 반나절
- 버그 재실행이 생기면(§7-7): 신규 27런 재학습 + eval 로 **+약 5–7 h**(같은 동시 수 가정)

---

## 11. 저자 결정 대기

- **[필수]** = 잠금(§0-0 2단계) 전에 정해야 함. 판정·코덱·청자를 바꾸는 항목

1. **[필수]** 계획서 밖 새 주기 승인, 이 문서 확정 위치·이름(09-28 C 전환판)
2. 팔 이름(가칭 cp0·cc2·cc6·cc12·ccinf·cr6·ct2·ca6·offr)과 논문 표기(P0·C2·C6·C12·C∞·R6·T2·A6)
3. **[필수]** 코덱 k 별 3개 별도(이 문서) vs nested dropout 1개
4. **[필수]** λ = 1, ½·½ 배분, τ 4개 값(§3-4·§3-5)
5. **[필수]** sit 복호 argmax one-hot(이 문서) vs softmax 확률 — 이제 코덱 학습 L_pair·A6·교차평가에만 영향
6. **[필수]** threat 정의: ray 단위 top-3 유지(이 문서) vs 표적 단위 NMS(바꾸면 StateRecon threat 그룹과 정의가 달라짐)
7. **[필수]** A6 덧붙는 부분 좌표: 그대로(이 문서, 저자 결정 1) vs 수신측 고정 좌표 변환 — A6 전용으로 축소
8. **[필수]** 절제 목록 규모(§6, 141건) + 보고 전용 오프라인 추가분(z 토큰 지도학습 probe·z 성분별 표준편차, §6 — 설계·판정 불변)
9. 명칭 'autoencoder-grounded latent communication (동결 화자 코덱 + 학습 청자)', 코덱 '과제 인지 코덱', §8-1 본문 필수 문장, 'pair-feature-weighted' 병기 여부(IMAC 류 보상 기반 압축으로 오독될 위험), 'latent' 사용 범위
10. **[필수]** ORACLE 재정의([S]/[P]) 채택, **sit 를 [P] 유래로, 수신자 own4 를 [S] 로 표기하는 것 확인**
11. **[필수]** 골든 새 케이스 `--regen` 승인, Mac·Windows 각각(새 케이스만, 7013a3a 체크아웃 코드로): **`ext_v1_off` = 필수**(09-28 구현 검토 blocking — 재사용 off·arpa6·offr 의 실제 경로를 기존 5케이스가 안 덮음, §0-4-5·9-3), `ext_v1_intent_ON` = 선택. 분기 재개 대조 9-33 은 골든 파일을 안 바꾸므로 regen 승인 대상 아님
12. **[필수]** 강화 판정 기준(6 pp, 동률 = 승 아님) 채택 — **처치 팔끼리도 문턱 고정 6.0 pp·병기 기준 고정 1.8 pp**(§7-3, 09-28 사전등록 검토 blocking)
13. **[필수]** G-C0 1단계(7013a3a 재평가)가 배치 X 값과 다를 때 처리(이 문서: 2단계 불일치만 중단, 1단계 불일치는 GPU 비결정성으로 기록 후 진행 — 확정 필요)
14. 확정(09-28 전제): offr 3런 포함, 판정 앵커는 배치 X off, offr 는 잡음 추정·기준선 점검 전용
15. **[필수]** 코덱 데이터 덤프: GPU + 파일 SHA 동결(이 문서) vs CPU 결정론
16. 해소(09-28 C 설계): P0 가 own4 를 가져 수신자 절대속력 경로가 C 비교에서 맞춰짐(§5-4). A6 비교에만 남음
17. 해당 없음(09-28): A형 T2 의 결측 0 문제 — C형 T2 는 결측 칸 자체가 없음(§3-10)
18. **[필수]** 붕괴 문턱 10 pp·처리 규칙(§7-8) 확인 — 09-28 추가: 통과가 대조 붕괴 시드에 걸려 있으면(빼면 2/2 아님 또는 평균차 6.0 이하) '판정 불가(대조 붕괴 의존)'로 낮춤(이 문서 권장값)
19. **[필수]** 버그 수정 시 재실행 범위 = 신규 27런 전체(§7-7) 확인
20. 논문 Fig 팔·라벨·run 매핑 변경(루트 CLAUDE.md §1)
21. '다음 주' 칸 후보: 지연·패킷 손실 강건성, 표적 단위 위협, B(학습 송신 인코더), 청자 용량 스윕(1×64·2×128·3×256), 청자 표현 probe(v 출력으로 쌍 특성 회귀), 추가 시드(전부 다음 주기 사전등록으로만)
22. **[필수]** 청자 k/v 용량 = 3×256(이 문서, §4-3) vs 1×64(A 판·ons6 와 같음) vs 다른 값 — 모든 통신 팔 동일 조건은 고정
23. **[필수]** own4 구성 = [sinψ, cosψ, SOG/1.8, ROT], cmd 제외(이 문서, §2-5)
24. **[필수]** P0 슬롯 = 폭 6 영(C6 와 같은 shape, 이 문서) vs 슬롯 없음(§4-4)
25. **[필수]** C∞ 입력 = §2-2 식 그대로의 P(이 문서) vs 코덱 z-score(§2-4)
26. **[필수]** T2 표현 = [wrap180(ψ)/180, 2·SOG/1.8−1], z 와 같은 격자(이 문서) vs 속도벡터(SOG·sinψ, SOG·cosψ) — 저자 결정 4 문구와 관련되므로 저자만 바꿀 수 있음(§3-10)
27. **[필수]** A6 의 지위 = 부 비교 + 읽기 진단(이 문서) vs 공동 확증(C6 또는 A6 통과 시 주장 — 위양성 0.11–11.7% 로 늘어남, §7-10). 이 문서 안이면 '보험'의 뜻 = 다음 주기 방향 근거 + 읽기 진단(논문 주장 보호 기능 없음, §4-12). 논문 표기 '읽기 진단 팔'
28. **[필수]** 토큰 폭 = 팔별 네이티브(이 문서) vs 공통 폭 패딩(§4-4)
29. **[필수]**(09-28 권고 → 필수로 올림) **C 전환분 재검토 잠금 전 1회**. 구현·사전등록 관점 1차 검토는 받아 반영함(§13). 남은 것 = **인과성 관점 검토(C 수신 경로: own4·slot 시점·R6 절차)** + 이 반영본(v2.1) 재확인. 잠금 뒤에는 청자 구조 변경이 금지(§7-9)라 새 수신 경로 결함을 잠금 뒤에 못 고침
30. **[필수]** 'latent' 문구 규칙(§7-4, 09-28 사전등록 검토 blocking) — (a) **권장**: 확증은 C6 vs off 그대로, 'latent 로도 이득' 문구는 C6 > P0(G-C3 ①) 통과 때만, 미통과면 '통신 채널 이득, z 기여 미확인' / (b) 확증 자체를 C6 > off 그리고 C6 > P0 로(검정력 줄어듦, 채널 헤드라인 없음). latent 문구 조건은 두 안이 같음
31. **[필수]** offr 판정 연결(§7-10) — **권장 = 채택**: offr vs off 가 어느 방향이든 6 pp 기준을 넘으면 C6 헤드라인에 C6 vs offr 통과를 추가 조건으로. 통신에 불리한 쪽으로만 작동. 불채택이면 지금처럼 병기만

---

## 12. 선행연구 대응 (앞 검토에서 원문·공식 페이지로 확인한 서지만. '초록' = 초록만 확인)

- Lin, Huh, Stauffer, Lim, Isola, "Learning to Ground Multi-Agent Communication with Autoencoders", NeurIPS 34 (2021), arXiv 2110.15349 (전문, 스크래치패드 `lin2021.txt`)
  - 구조: speaker 모듈 = 자기 관측 이미지 오토인코더, 메시지 = 그 부호. listener 모듈 = 관측 + 받은 메시지로 행동을 RL(A3C)로 학습. AE 재구성 손실은 listener 정책 gradient 손실과 **함께(jointly) 최적화**(§4.1–4.2) = AE 가 RL 과 같은 기간 계속 학습. 메시지는 이산 기호(길이 10, §5.5). listener 는 메시지를 선형 embedding → concat → **3층 MLP** 로 128D 특징을 만듦(§4.2)
  - AE grounding 이 공통 언어에 충분하다고 주장. AE 위에 RL 을 더 얹은 ae-rl-comm 은 "consistently performed worse"(FindGoal 제외)
  - **대응: C = 화자 AE + 청자 RL 구조와 같은 계열**
  - 차이: 이쪽 화자 코덱은 **동결**(OFF trunk 데이터로 오프라인 한 번 학습, RL 중 갱신 없음) · **과제 가중 손실**(쌍 특성 항) · 저차원 자기 상태 25D(픽셀 아님) · 8bit 연속 격자 코드(이산 기호 아님) · 청자 = 쌍별 attention 토큰(수신자 자기 상태 포함)
- Jayalath, Morad, Prorok, "Generalising Multi-Agent Cooperation through Task-Agnostic Communication", DARS 2024, arXiv 2403.06750
  - 사전학습 AE 를 동결하고 정책을 학습("Keeping the autoencoder weights frozen, we train policies") → **동결 원칙이 같음**
  - 차이: 그쪽은 과제 무관 집합 AE(수신측), 이쪽은 송신자별·과제 인지 손실·청자는 z 를 쌍별 토큰으로 직접 읽음. 실행 중 복원오차로 분포 이동 감지 → cdc_* 텔레메트리로 대응
- 과제 기준 압축: IMAC(Wang, He, Yu, Qiu, An, Rabinovich, ICML 2020, PMLR 119:9908–9918), NDQ(Wang, Wang, Zheng, Zhang, ICLR 2020, arXiv 1910.05366), VQ-VIB(Tucker, Levy, Shah, Zaslavsky, NeurIPS 2022)
  - 과제 gradient 로 메시지를 학습하며 정보량 정규화. 이쪽의 '과제 인지'는 **오프라인 쌍 특성 충실도**일 뿐 과제 보상 압축이 아님 → 명시(§8-1)
- Lowe, Foerster, Boureau, Pineau, Dauphin, "On the Pitfalls of Measuring Emergent Communication", AAMAS 2019
  - positive listening 은 개입으로 보여야 함 → msgzero·slot-shuffle·z-shuffle·R6
- Eccles, Bachrach, Lever, Lazaridou, Graepel, NeurIPS 2019, arXiv 1912.05676: 화자·청자 공동 탐색이 어려움 → 동결 화자는 청자 입력 분포를 정상으로 만듦(청자만 학습)
- Foerster, Assael, de Freitas, Whiteson(RIAL/DIAL), NIPS 2016, arXiv 1605.06676: 채널로 gradient 를 보내는 방식 — 이번엔 안 씀(청자 gradient 는 z 에서 멈춤)
- Das et al., TarMAC, ICML 2019(초록): 비슷한 attention 집계. SOTA 주장 안 함
- Kim, Park, Sung, Intention Sharing, ICLR 2021(초록): 의도 메시지 attention 압축 — intent 필드 대조 선례
- 수신자 상태를 k/v 토큰에 넣어 쌍별 계산을 집계 전에 하게 하는 구체적 선례는 앞 검토 검색 범위에서 못 찾음(TarMAC·ATOC·Intention Sharing 은 같은 attention 계열) — '없음' 아님
- 해상
  - Wang & Zhao, Ocean Eng 320:120244 (2025)(전문): 서로 안 보이는 2척, 학습 메시지 3원소, AWGN/BSC 잡음 → end-to-end 학습 저차원 통신 선례. 이쪽은 동결 코덱 + 8bit 양자화 + 학습 청자
  - Ding, Meng, He, Li, JNCA 251:104511 (2026)(초록): 상태·TCPA/DCPA·양보/유지 역할 공유 + MAPPO → ons6·A6 형식과 거의 같음 → 명시 필드 공유 자체의 새로움은 제한적
  - Chen, Ma, Xu, Chen, Wang, JMSE 9(10):1056 (2021)(초록): 타선 상태를 관측으로 받음 = 명시 공유(C∞ 와 가까움)
  - Xu et al., PLOS One 21(6):e0345950 (2026)(초록): 과거 AIS + LSTM 으로 통신 없이 타선 의도 추론 → intent 필드 대조 선례
- 인용 보류: Draz et al., Ocean Eng 345:123626 (2026) — 초록 못 구함, 내용 주장 금지
- 차별점(검색 범위 안에서 선례 **못 찾음** — '없음' 아님): 송신자 자기 페이로드를 과제 가중 동결 코덱으로 2–12차원 latent 로 압축 → 받는 배 attention k/v 가 자기 상태와 함께 z 를 직접 읽도록 RL 학습 → 같은 z 의 손 공식 읽기(A6)·원값(T2)·무정보(P0)·짝 끊기(R6)·무손실(C∞) 대조, radar-only 가 무너지는 regime(16척, r56, imo)

---

## 13. 변경 이력·잠금 기록

- 잠금 기록(빈칸 = 아직 안 함)
  - 문서 잠금 커밋 해시: ____
  - 구현 커밋 해시: ____
  - 코덱 데이터 SHA(s43/s44/s45): ____ / ____ / ____
  - 코덱 내용 SHA(k=2 / 6 / 12): ____ / ____ / ____
  - 충실도표 커밋 해시: ____
  - 배치 제출 커밋·시각: ____
- 2026-09-27 초안 작성(결과 전). 저자 결정 5개(§0-4) 반영, 앞 설계 검토(wf3 '권장 설계') 구조 채택
- 2026-09-27 3관점 검토(구현·인과성·사전등록) 반영, 결과 전
  - 구현: off 앵커 덮어쓰기 방지(§5-2), R6 generator fail-closed·호출부 표(§4-2), 팔 구분 표시(§4-8)
  - 인과성: sit [P] 유래 재분류(§2-2·§8-2·§8-3), 수신자 절대속력 경로 반영·귀속 문구 정정(§5-4·§7-5), traj 명령 시점 한 칸 당김·재스폰 제외(§3-6)
  - 사전등록: 잠금 순서(§0-0), 사전 탐색 이력·선택 편향 공개(§0-3), 기준 없는 판정 3곳 확정(§7-3·§7-5), 확증 비교 1개·부 비교 표기(§7-4), H1a 기저율·버그 수정 경로(§7-7), 시드 추가 금지(§7-9), 붕괴 처리(§7-8)
  - 권고 반영: 구현 브랜치 명시, 기존 VESSEL_REQUIRE_TRUNK 가드 사용, 그룹 이름 확장 변경 목록, 복원 리셋·device·난수 격리, 병합 state_dict 재초기화·모듈 교체 금지, 쪼개기 브로드캐스트 조건, 의사코드 정정, 팔 env·CLI 교차평가 규칙, 합성 코덱 테스트, 골든 플랫폼별, G-C0 2단계, 스모크·이름 목록, `comm_codecs/` 폴더명, `_THREAT_ANG` device, 크래시 재개 절차, soft cascade hard 경계, T2 결측 해석, R6 표기, 거리대별 충실도, 비트 표기 규칙, 주장 템플릿, 검정력 두 가정, 보고표 보강, 코덱 버그/튜닝 구분, 동결 목록 확장, 리뷰어 질문 표, radar_dropout_p 0.0 확인
- **2026-09-28 C 전환(결과 전, 이번 주기 학습 0회)** — 저자 원문 'ㅇㅇ C로 바꿔줘'(§0-6)
  - A 중심판 사본: `scratchpad/grounded_codec_spec_A_version.md`(잠금 때 스펙 폴더로 복사 보존)
  - 제목·명칭: 'autoencoder-grounded latent communication (동결 화자 코덱 + 학습 청자)', 'emergent' 금지 유지·'learned communication' 단독 금지(§8-1)
  - 송신측(§2·§3) 유지. 추가: own4 정의(§2-5), C∞ 입력 = P 식 그대로(§2-4), T2 재정의(§3-10: 청자가 원값 2개 직접, ψ/180, z 격자)
  - 수신측(§4) 재작성: cz{W} 토큰 [relpos, slot, own4, msg×0](§4-1), comm_gather cz 분기·slot-shuffle(§4-2), 청자 k/v 3×256 결정·sup 근거(§4-3), 팔별 폭·파라미터 표·공정성(§4-4), 재초기화 화이트리스트 모듈화·Adam 이름 기준 재배치(§4-5), 새 env·스냅샷 키(KV_HIDDEN·KV_DEPTH·SLOT_SHUFFLE·ARM_TAG)(§4-7), A 경로는 A6 하위 절로(§4-12)
  - 팔(§5): 신규 = cp0·cc2·cc6·cc12·ccinf·cr6·ct2·ca6·offr(27런). **폐기 = ca2·ca12·csinf(A형 S∞)·A형 ct2**. 수신자 절대속력 경로가 C 비교(vs P0)에서 해소됨을 확인해 적음(§5-4)
  - 평가(§6): 141건. C(k) payload 교체 교차평가 불가 명시, slot-shuffle·C∞ recon 교차 추가
  - 판정(§7): 확증 = C6 vs off, 귀속 = C6 vs P0·R6, 압축 = C2 vs T2, 상한 = C∞ vs off, 읽기 진단 G-C5(A6 vs C6 해석표) 신설, A6 vs off = 부 비교. H1a 8팔 기저율 36.2–45.8% 재계산, C6|A6 위양성 추가, offr 잡음 병기, 예측 갱신(C 성공 확률 A 보다 낮을 수 있음, 새 확률 수치 안 만듦)
  - A 판 사전등록 수리 전부 유지(잠금 순서·사전 탐색 공개·동률·열등 정의·다중 비교 문구·H1a 기저율·버그 재실행·시드 추가 금지·붕괴·앵커 보호·fail-closed generator·팔 구분 문자열·sit [P]·traj cmd t−1)
  - §11: 22–29 신설(C 세부 확인·재검토 권고), 14·16·17 상태 갱신. §12: Lin 2021 대응(화자 AE + 청자 RL, 차이 = 동결·오프라인·과제 가중) 보강
  - 새로 돌린 것: 위양성 시뮬 2개·파라미터 계산만(데이터·학습 없음, §0-3)
- **2026-09-28 C 전환분 검토 2관점(구현·사전등록) 반영(결과 전, 학습 0회)** — 판정 조건: 두 관점 모두 '조건부 통과'
  - 구현 blocking 1: 재사용 off·arpa6·offr 비트동일 검증이 실제 설정(EXT=1·imo·none·crossing 0·분기 재개)을 안 덮음 → 새 골든 `ext_v1_off` [필수](7013a3a 체크아웃, Mac·Windows)(§9-3·§11-11), 분기 재개 대조 9-33 [필수] 신설, 재초기화·재배치 발동 조건(같은 shape 이면 no-op, §4-5), §0-4-5·§1 전제 문구 정정
  - 사전등록 blocking 1(주장↔확증): 확증 C6 vs off = 채널 전체 효과로 명시, 'latent' 문구는 G-C3 ① 통과 때만(안 a, §7-4) / 대안 b 병기 → §11-30 저자 택1. §1 대응표를 G-C3·slot-shuffle 로, §7-12 표에서 '…' 물려받기 없앰, §7-5 C6 주장 문구 조건부
  - 사전등록 blocking 2(문턱 모호): 처치 팔끼리 문턱 고정 6.0 pp·병기 기준 고정 1.8 pp(§7-3)
  - 사전등록 blocking 3(해석표 사후 여지): G-C4·G-C5 표를 '둘 다 짐' → '상대 H1a FAIL 의존' → 승패 순으로 재배열, 두 팔의 off 대비 판정·H1a 상태 필수 병기(§7-5), H1a 의존 규칙 범위 확장(§7-7)
  - 권고 반영(구현): kv 스니핑 'Linear 개수 − 1'(§4-7), 같은 폭 팔 전역 드리프트 가드 `comm_sig`(§4-2), 골격 전역 저장·복원(§4-5), 텔레메트리 핀·r6_fb/r6_self 열·끝 행 검사(§4-7·§5-2·§7-7), 팔 → 평가 arm 표(offr = OFF), comm_variant_env 새 키 명시 export·9팔 env 블록 고정(§4-7), comm_gather 미연결 호출부 표(§4-2), 테스트 9-34–41, check_branch kv 규칙 대상(cz*·v2p)·재사용 CSV 복사(§4-5), FP_IGNORE 추가, 라인 인용 정정(clip :931-933, `_THREAT_ANG` :249-251), 학습량 수치(§7-11), kv 키 뜻 범위·전역 이름 `COMM_EXT_MLP_HIDDEN` 유지(§4-7), 작업량 +500–700줄·+2일·Windows 단계별 가정(§10), 확인 사항 근거 병기(§4-2·§4-9·§5-4)
  - 권고 반영(사전등록): §5-2 짝 검사 사실 정정(offr 로 통과) + 재사용 파일 9개 존재 검사, offr 판정 연결 → §11-31, §11-29 [필수]로, 버그 점검 맹검 순서·결과 뒤 버그 기록 규칙(§7-7), T2 sup 근거 범위 명시(§0-3·§3-10·§4-3), z 토큰 probe·z 표준편차(보고 전용, §6), C(k) > C∞ 원인 미상 문구(§7-5·§7-11), A6 '보험' 뜻 정리(§4-12·§11-27), 붕괴 의존 시 판정 낮춤(§7-8), 위양성 범위 σ_d 3.2–8.2 로 통일(§7-4·§11-27), §10 동시 실행 가정·VRAM 미확인 명시
  - 안 한 것: 인과성 관점 C 전환분 검토(입력 없음) → §11-29 [필수]로 남김. T2 형식 sup 는 안 돌림(선택 사항, 돌리면 §0-3 공개)
- **결과 뒤 발견 버그 기록 형식**(§7-7): 날짜·시각 / 발견 경위(무엇을 보다가) / 그때 본 결과 방향(통신에 유리·불리·안 봄) / 수정 커밋 / 재실행 범위
