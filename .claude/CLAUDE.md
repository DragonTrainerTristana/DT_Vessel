# Vessel Multi-Agent RL — 구조·설정·진단 규약

2026-09-10 전면 재작성(코드에서 직접 뽑음). 이전 2026-06-02판(obs 59D·MLP·게이트 −3·MoE 없음)은 전부 폐기.
규칙(허락·정직성·말투)은 루트 `CLAUDE.md`. **현재 상태·할 일은 여기 안 씀 → `runs/STATUS.md`만.**
라인 번호는 2026-09-10 `refactor/2026-09-10` 기준. "(미확인)" 표시는 코드로 못 확인한 값.

---

## 1. 프로젝트 — 학습 경로 두 개

다중 선박이 shared policy PPO로 목표까지 항해하며 COLREGs 준수·충돌 회피. 연구 질문 = 에이전트 간 latent 메시지 통신이 협력 회피를 개선하는가.

| 경로 | 파일 | 역할 |
|---|---|---|
| **GPU 배치 (현행 주 경로)** | `vessel_gym.py`(VesselBatchEnv) + `vessel_gym_train.py` | 탐색·절제실험 전부. `run_repro.sh`가 돌리는 것 |
| **Unity** | `main.py` + ML-Agents(C# `Agent/`, `Navigation/`, `Management/`) | ground-truth 판정(sim2sim 판정관). `VESSEL_OUTCOME_LOG`/`VESSEL_METRIC_LOG` |

- **두 경로가 공유하는 것 = `config.py` + `networks.py` 뿐.** 체크포인트 shape 호환은 이 둘 때문임.
- 나머지는 각자 구현(같은 이름이라도 다른 코드):

| 기능 | GPU 배치 | Unity |
|---|---|---|
| frame stack | `vessel_gym_train.FrameStack` (:43) | `frame_stack.MultiAgentFrameStack` |
| GAE | `vessel_gym_train.batched_gae` (:95) + `ValueNorm` (:116) | `memory.Memory` (dones/truncateds 경계) |
| PPO 루프 | `vessel_gym_train.py` :846-901 | `main.ppo_update` (:81) |
| others_msg(rollout) | `vessel_gym_train.comm_gather` (:259) | `networks.CNNPolicy._get_others_msg` (:1032) |
| 체크포인트 스냅샷 | `_cfg_snapshot()` (:672) | `main._unity_snapshot` (:725) → `ckpt_io.snapshot_config` |
| 평가·진단 | `eval_ckpt.py` · `eval_mixed.py` · `diag_ckpt.py` (전부 `ckpt_io` 경유) | `VESSEL_LOAD_MODEL=1 VESSEL_TRAIN=0` |

- `vessel_gym_train.py` 헤더 docstring은 "ON arm 배치 집계는 별도 작업"이라 적혀 있으나 `comm_gather`가 이미 구현됨 — docstring이 낡음.

---

## 2. 가설·정직성 (문구 유지)

> 통신은 **자동으로 좋아지지 않음**(학습돼야 하고 실패할 수 있음). 아래는 *증명 대상*이며 "통신이 이기도록 강제"가 아니라 **공정하게 입증**하는 것이 목표.

### H1. 통신 ON > 통신 OFF (reward · fuel↓ · 궤적 간소화)
- **H1a (불변식·반드시 성립): comm-ON ≥ comm-OFF — 통신은 절대 더 나빠지면 안 됨.** 근거: 정보가치 비음수 — 에이전트는 메시지를 무시할 수 있음(게이트·fc2 메시지 가중치 0 → comm-OFF와 동일). **comm-ON이 OFF보다 나쁘면 = 구현/최적화 버그(반드시 고침).**
- **H1b (목표·조건부): comm-ON > comm-OFF.** 자동 아님 — task가 통신을 *필요*로 할 때만(국소 인지 부족/협응 모호). 완전관측에선 ON = OFF가 최선.
- 순서: H1a 먼저(안 나빠지게) → H1b(도움되게 regime 부여).

**anti-rigging 3원칙 (필수):**
1. **Ground-truth로 평가** — shaped reward 아닌 실제 충돌률·연료·궤적·COLREGs 준수.
2. **공정한 baseline** — comm-ON은 *불구화 안 된 최선의 comm-OFF*를 이겨야 함. OFF를 인위적으로 나쁘게 만들면 = rigging(무효).
3. **통신이 필요한 조건에서만** — 완전관측 COLREGs는 기하로 최적행동이 결정돼 통신 잉여. 국소 인지 부족·협응 모호 regime에서 시험.

### H2. latent 차원 ↑ → 통신 질 ↑
- **판정 완료: 불지지.** `runs/m2_ablation/RESULT.md` — 18런(C1~C6, 차원 2~12) 수렴평가 ANOVA **F(5,12)=0.39** (임계 3.11). 붕괴 시드 없음.
- H1이 성립한 *후에만* 의미 있음(통신이 안 쓰이면 차원은 무의미). **H1 선행 필요.**

---

## 3. obs 계약 — 369D

| index | dim | 내용 | 네트워크 입력 | frame-stack |
|---|---|---|---|---|
| `[0:360]` | 360 | 레이더 raw ray 1°. 값 = dist/range − 0.5, 미감지 +0.5 | RadarEncoder | **×3** (Conv 채널축 = 시간) |
| `[360:362]` | 2 | goal: dist `d/(d+k)`, angle/180 | ✓ | ✗ |
| `[362:366]` | 4 | self: speed, yawRate, heading, rudder (정규화) | ✓ | ✗ |
| `[366:368]` | 2 | position x, z | **제외** — 통신 파트너·relpos 계산용 | ✗ |
| `[368]` | 1 | COLREGs situation 0~4 (None/HeadOn/CrossingStandOn/CrossingGiveWay/Overtaking). 1-step stale(의도) | **MoE 라우팅 키 + one-hot 5D 입력**(`SITUATION_INPUT`) | ✗ |

- 네트워크 실입력 = radar 3×360 + 2 + 4 + one-hot 5.
- **동시 수정 파일 (차원·순서 바꾸면 전부):**

| 파일 | 위치 |
|---|---|
| `Agent/VesselAgent.cs` | `CollectObservations` :958-1000 (radar :974, goal/self :981-990, pos :993-994, situation :1000), `Initialize` VectorObservationSize :333 |
| `Python/vessel_gym.py` | `_build_obs` :674-707 (조립 :698-706) |
| `Python/vessel_gym_train.py` | `parse_obs` :34-40 |
| `Python/obs_utils.py` | `parse_observation` :10-33 (Unity 경로) |
| `Python/config.py` | :44-66 (RADAR_RAYS/STATE_SIZE 360, GOAL 2, SELF 4, POSITION 2, SITUATION 1, OBSERVATION_SIZE 369, FRAMES 3 :55) |
| `Python/networks.py` | fc2 입력 :482 / :604 / :861, `SIT_INPUT_DIM` :149 |

- Unity 빌드 obs 크기 ≠ 369면 연결 시 RuntimeError(build trap, `main.py`).

---

## 4. 네트워크 (`networks.py`)

세 망 **MessageActor / ControlActor / Critic 각자 독립 인스턴스**(RadarEncoder 포함, 공유 없음). MoE ON이면 망마다 코어 5벌.

| 구성요소 | 코드 | 내용 |
|---|---|---|
| **RadarEncoder** | :163-218 | Conv1D circular ×3: 3→32 k5 s2 (:183), 32→64 k5 s2 (:184), 64→64 k3 s2 (:185) → 360→180→90→45. 옵션 1×1 bottleneck `reduce` 64→8 (`VESSEL_RADAR_HEAD=bottleneck`, :187-190) → flatten(2880, bottleneck이면 360) → Linear→`RADAR_FEAT_DIM`=30 (:194). 활성 ReLU, `VESSEL_RADAR_ACT=leaky`로 LeakyReLU(0.01) (:34, :204) |
| **인코더 공유** `SHARED_ENCODER` (config, 2026-09-10) | `'all'` **기본(YUGIOH)**=msg·cri←ctr / `'0'` 세 망 각자(legacy, 12런까지) / `'actor'` msg←ctr. `networks._share_encoder_across`, CNNPolicy.__init__ 세 망 생성 직후 | 12런 실측 메시지망 인코더 출력 산포 = 조타망의 1/50~1/1000(학습 신호 1/2000) → 조타 손실이 인코더를 직접 학습시키게. 파라미터: 0→356,607 / actor 322,089 / all 287,571(배치+bottleneck). 골든 `batch_shared_{all,actor}_ON`, 미러 4케이스 |
| **MoE** | `USE_MOE` 기본 **ON** (config :235), 전문가 5 (:236), `MOE_WIDTH` 1.0 (:244), `MOE_SHARED` 0 (:249) | 코어 통째 hard-routing(obs[368]). ModuleList :529/:671/:896. `_share_radar_encoder` (:130-135) = **MOE_SHARED=1일 때 전문가 5벌끼리만** 인코더 공유(세 망 간 공유 아님) |
| **situation one-hot** | `SITUATION_INPUT` 기본 ON (config :264) → `SIT_INPUT_DIM`=5 (:149), `_situation_onehot` :152 | 세 망 fc2에 concat. MoE에서도 입력(코어 내 상수, 무해) |
| **MessageActor 코어** | :467-511 | radar30 + goal2 + self4 + sit5 = **fc2 41**→128 ReLU → `msg_ln` LayerNorm(**기본 ON**, `VESSEL_MSG_LN`, :490, 2026-08-31 tanh 포화 방지) → `msg_out` 128→MSG_DIM tanh. `msg_out` **×0.1 소진폭**(:495-497) |
| **ControlActor 코어** | :587-651 | **fc2 47**(41+msg 6)→128 tanh → fc3 128→64 tanh → action_mean 64→2 (×0.1, :621-622). fc2 **메시지 슬라이스 ×0.1** (:607-608). `msg_gate` 초기 **0.0 = sigmoid 0.5 중립** (:610). per-dim logstd [−1.0, −0.5] (:625). mean clamp ±3, logstd [−2.3, 0] (:794-795, :824-825). `consumer_decoder` 128→64→K·2 항상 생성(:613, K=min(COMM_CONSUMER_K 3, MAX_PARTNERS 4)=3) |
| **Critic 코어** | :846-882 | **fc2 47**→128 ReLU → value 1. `CENTRAL_CRITIC=1`이면 `glob_enc` 6→64→64 per-ship mean-pool 추가 → **fc2 111** (:858-861). 메시지 슬라이스 ×0.1 (:862-863), `msg_gate` 0.0 (:864) |
| **집계 모듈** | `msg_encoder` 9→32→6 (:990-992, POS_GROUND 기본 ON config :117) · `GroundedAttention` q 6→32 / k 9→32 / v 9→6 ×0.1 (:413-464, USE_ATTENTION 기본 **OFF** config :107) | 둘 다 항상 생성 |
| **디코더 5** | intent :1005 · threat :1012 · goal :1017 · role :1023 (항상 생성) · state_recon :1027 (계수>0일 때만) | 계수 기본 **전부 0** (config :128/:144/:157/:186/:165). consumer 계수 0 (:199) → `consumer_decoder` 미사용 |
| **CTDE** | `CENTRAL_CRITIC` 기본 OFF (config :172) | 켜면 `CNNPolicy.forward(global_feat=…)` 필수 — 안 넘기면 zeros 경고(:1233) |
| **행동** | 2D squashed Gaussian. `action_raw`(pre-tanh) 저장→update 재사용(:786-844) | `[0]` rudder, `[1]` thrust |

**설계 결정(기각 이력 포함):**
- 게이트 **−3 init + 개방 페널티(0.02) 설계는 2026-06-12 기각** — 채널 grad 0 상태에서 페널티만 작용해 −8까지 단조 폐쇄(흡수상태, 체크포인트 실측). 현재 중립 0 + 페널티 0 (config :93-96).
- **zero-init 기각** — 생산측(msg_out)·소비측(fc2 슬라이스·v_proj) zero가 직렬곱 새들 형성 → 채널 grad 항등 0, 1M step 동결(실측). 전부 ×0.1 소진폭.
- `msg_ln` — 시드 절반이 mean|tanh|=1.000 포화로 학습 정지(실측) → LayerNorm으로 pre-tanh O(1) 고정.

**파라미터 수 (2026-09-10 실측, torch 1.9 CPU, `CNNPolicy.parameters()` 합):**

| 구성 | 총합 | msg / ctr / critic | 공용(attn+msg_encoder+4디코더) | state_dict 키 |
|---|---|---|---|---|
| **기본값**(MoE5 폭1.0 shared0 LN1 SIT1) | **1,827,999** | 580,020 / 663,885 / 579,360 | 4,734 | 261 |
| 기본 + MOE_SHARED=1 | 512,823 | 141,628 / 225,493 / 140,968 | 4,734 | 261 |
| USE_MOE=0 (단일) | 369,387 | 116,004 / 132,777 / 115,872 | 4,734 | 73 |
| MOE_WIDTH=0.44 | 363,564 | 114,195 / 130,740 / 113,895 | 4,734 | 261 |
| 2026-09-04 배치(shared1 attn1 cc1 sr1) | 581,847 | 141,628 / 225,493 / 204,968 | 9,758 | 289 |
| **★YUGIOH(2026-09-10 최종) = 12런 구성 + `SHARED_ENCODER=all`** | **287,571** | 인코더 1벌(34,518). 같은 기준 구조 4종: 단일망 92,935 / 얇게 **폭 0.32** 92,163(−0.8%) / 두껍게(MOE_SHARED=0) 425,643 / 공유 287,571 — `runs/ABLATION_PLAN.md` §3 | 9,758 | 319 |
| 09-04 배치 + `RADAR_HEAD=bottleneck(ch=8)` = 12런 실제 구성 | **356,607** | 인코더당 fc 2880→30(86,430) → 1×1 conv(520)+fc 360→30(10,830), 3벌 −225,240 | 9,758 | 319 |

- README.md의 364,397 / 1,821,985 / 358,270 = **3망 합(공용 모듈 제외)·msg_ln 없음** 기준 — 위 실측과 정합(LN 제외 단일 3망 합 = 364,397).
- 0908 문서·`Vessel_신경망_층별명세`의 356,607 = 위 bottleneck 행 (2026-09-10 CPU 실측 재현). 12런 평가 헤더 `head=bottleneck(ch=8) act=leaky` 로 확인.

---

## 5. state_dict 키 결정자 (체크포인트 호환의 핵심)

**키 생성/삭제·shape를 바꾸는 것** — 다르면 strict 로드 실패. `ckpt_io.restore_policy`가 키 스니핑으로 복원(§8).

| 결정자 | 코드 | 키 영향 |
|---|---|---|
| `VESSEL_MSG_LN` (기본 1) | networks :490 | `msg_actor.*.msg_ln.{weight,bias}` 생성/부재. ckpt_io가 키로 스니핑 |
| `SHARED_ENCODER` (config) | `_share_encoder_across` | **키·shape 불변** — 공유 텐서가 msg_actor/ctr_actor/critic 접두어로 동일 사본 저장(MOE_SHARED 와 같음). 구 ckpt(3벌 다름)를 공유 모델에 strict 로드하면 마지막 접두어만 남는 조용한 사고 → `ckpt_io` 가 스냅샷 `shared_encoder` 로 복원(스냅샷 없으면 '0') |
| `USE_MOE` (config :235) | :529/:534, :671/:676, :896/:901 | 접두 `experts.{0..4}.` ↔ `core.` |
| `CENTRAL_CRITIC` (config :172) | :858-861 | `critic.*.glob_enc.{0,2}.*` 생성 + critic fc2 (128,47)→(128,111) |
| `STATE_RECON_COEF>0` (config :165) | :1027 | `state_recon.net.{0,2}.*` + 버퍼 `run_mean/run_var/stat_inited/loss_ema` (조건부 생성) |
| `VESSEL_RADAR_HEAD=bottleneck` | :47-48, :187-190 | `*.radar_encoder.reduce.*` 생성 + fc (30,2880)→(30,360) |
| **무조건 생성(계수 0·미사용이어도 키 있음)** | `consumer_decoder` :613 · intent/threat/goal/role 디코더 :1005/:1012/:1017/:1023 · `msg_encoder` :990 · `attn` :999 | **제거 금지** — 지우면 기존 체크포인트 전부 strict 로드 깨짐 |
| `MOE_SHARED` / `_share_radar_encoder` | :130-135 | **키 불변** — 5벌 동일 사본 저장(load 호환). 파라미터 수만 다름. RUNS.md는 텐서 동일성으로 공유 여부 판정 |
| `VESSEL_MOE_FAST` / `_bmm_*` | :88, :91-128 | **키 불변** — 매 forward stack. 단 비트동일 아님(~1e-7) |
| fc2 concat 순서 | ctr :635-636, critic :870-879 | **[radar, goal, self, sit, (glob), msg] — msg 항상 마지막.** `[:, -msg_dim:]` 인덱싱(×0.1 init·텔레메트리)이 의존 |

**shape만 바꾸는 옵션** (키 이름 동일, 차원 불일치 → from-scratch):
`MSG_DIM` (config :53) · `MOE_WIDTH` (:244, `_w` :138) · `SITUATION_INPUT` (:264, fc2 ±5) · `ATTN_DIM` (:108) · `RADAR_FEAT_DIM` (:49) · `INTENT_K`/`THREAT_K` (디코더 out) · `COMM_CONSUMER_K` (consumer out, ≤MAX_PARTNERS) · `COMM_CONSUMER_COUPLING` (fc3 in 128→134) · `VESSEL_RADAR_BOTTLENECK_CH`.

**키에 영향 없는 옵션 = "조용히 다른 실험"** (가중치에 흔적 없음, **cfg_snapshot이 유일한 근거**):
`USE_ATTENTION` · `POS_GROUND` · `VESSEL_AGG_MODE` · `VESSEL_NEAREST_SCALE` · `VESSEL_MSG_GAIN` · `VESSEL_MSG_TOKEN_GAIN` · `VESSEL_RADAR_ACT` · `VESSEL_RECON_EMA_FLOOR/PRE/LEGACY_STAT` · `COMM_RANGE` · `MAX_COMM_PARTNERS` · 모든 손실 계수 · `USE_ORACLE` · `USE_COMMUNICATION` · 보상·시뮬 상수 전부.

---

## 6. rollout = update 미러 (PPO ratio 유효 조건)

others_msg 집계가 **세 곳에 복제**돼 있음. 한 곳만 고치면 ratio가 에러 없이 조용히 깨짐.

| 복제 | 코드 | 역할 |
|---|---|---|
| `vessel_gym_train.comm_gather` | :259 (분기 :325-345) | GPU 경로 rollout |
| `networks.CNNPolicy._get_others_msg` | :1032 (oracle :1057, attention :1084, pos_ground :1116, sum/mean/scale :1178-1185, mean-field fallback :1188) | Unity 경로 rollout |
| `networks.CNNPolicy.evaluate_actions` | :1246 (oracle :1280, attention :1304, pos_ground :1308, sum/mean/scale :1313-1318) | 양 경로 공통 update |

- **우선순위: attention > pos_ground > sum·mean·scale**, 끝에 `msg_gain`. 세 곳 모두 같은 순서·같은 함수형.
- MoE 라우팅도 미러 대상: 파트너 메시지는 저장된 `partner_situations`로 재생성(:1302), 자기 행동은 저장된 `situation`으로 재라우팅.
- **검증기 둘 다 ALL PASS 필수** — `_verify_ppo_mirror.py`(Unity 경로, VERDICT :443) · `_verify_comm_mirror.py`(gym 경로, :123). `run_repro.sh preflight`(:91-112, ALL PASS grep :96-99)가 grep으로 강제.
- **"한 곳만 고치면 4번째 사고"** — 과거 3건:

| # | 일자 | 사고 | 기록 |
|---|---|---|---|
| 1 | 2026-09-04 | `comm_gather`에 attention 분기 없음 → rollout=mean / update=attention | `runs/m2_ablation/COMM_PLAN.md` :197-198 |
| 2 | 2026-09-05 | `POS_GROUND=0`이면 rollout=pos_ground / update=sum (blocker) | `vessel_gym_train.py` :319-322 |
| 3 | 2026-09-05 | `VESSEL_MSG_GAIN`이 update에만 걸림 | `vessel_gym_train.py` :323 |

- 배치 통계 정규화 금지(rollout E·N vs update 미니배치 통계가 달라짐) — 상수배만(:59).

---

## 7. 설정

**원칙: `config.py`가 정본 — 2026-09-10 통합 완료** (커밋 348018b·f6998a2). `networks.py`·`vessel_gym.py`·`vessel_gym_train.py`의 비주석 `os.environ` 읽기 **0개**. 새 env 키는 `config.py`에만 추가하고 각 모듈은 import 만 한다.

**★YUGIOH 최종판 (2026-09-10 저녁) = config 기본값.** env 를 하나도 안 주면 12런 배치(commfix) + 인코더 망 간 공유 설정으로 돈다. 그날 바뀐 기본값 11개는 아래 표에 ★(괄호 = legacy 값). 정의·출처는 `config.py` 끝 `CODE_VERSION / YUGIOH / YUGIOH_ARGS / YUGIOH_LEGACY`. `test_golden default_ON/OFF` = YUGIOH(이날 재생성, 319텐서), `batch_2026_09_04_ON` 은 legacy 11개를 **명시 핀**해 과거 골든 그대로 PASS. `ckpt_io.restore_policy` 는 스냅샷에 키가 없으면 YUGIOH 가 아니라 legacy 로 복원(§8). 옛 기본값으로 돌리려면 `YUGIOH_LEGACY` 를 env 로 주면 됨. 스냅샷 `code_version` 키로 판별.

| 어디 | 무엇 | 비고 |
|---|---|---|
| `config.py` 끝 "config 통합" 절 | 레이더 `RADAR_ACT/HEAD/BOTTLENECK_CH` · 집계 `MSG_LN MSG_TOKEN_GAIN AGG_MODE NEAREST_SCALE MSG_GAIN MSG_RANDOM_SD` · `RECON_EMA_FLOOR RECON_LEGACY_STAT MOE_FAST` · 학습기 전용 12(`VALNORM_BETA FARFIELD/PERPAIR_COEF TIMEOUT_BOOTSTRAP GRAD_TELEMETRY CLIP_PER_MODULE MSG_GATE_APPLY NOCOMM_* BLIND_WARN_AFTER COMM_TELEMETRY*`) · vessel_gym 시뮬·보상 27 | 이름·기본값은 통합 전과 동일 — `test_golden` 비트동일로 확인 |
| `networks.py` 모듈 전역 `MSG_LN AGG_MODE NEAREST_SCALE MSG_GAIN _RADAR_LEAKY/_HEAD/_BOTTLENECK_CH _MSG_TOKEN_GAIN _RECON_EMA_FLOOR` | config 값으로 초기화. **rollout(`_get_others_msg`, `vessel_gym_train.comm_gather`)과 update(`evaluate_actions`)가 같은 객체를 읽음** = 미러의 근거 | `ckpt_io.restore_policy`가 체크포인트 스냅샷으로 **이 전역을** 덮어씀. import 후 `os.environ` 변경은 무효 |
| `vessel_gym_train.MSG_RANDOM_SD` | RANDOM 팔 난수 sd | 위와 같은 규약 |
| 남은 env 직접 읽기 | `VESSEL_CKPT_DIR`(경로, ckpt_io) · Unity `main.py` 6개(SEED·PROFILE·GRAPHICS·ALLOW_PARTIAL_LOAD·RECV_ONLY_COUNT·METRIC_LOG) · C# 34개 | 경로·실행 인자·C# 은 통합 범위 밖 |

**주요 토글 (`config.py`, 기본값은 코드 확인):**

| env | config 상수 | 기본 | 라인 | 의미 |
|---|---|---|---|---|
| `VESSEL_USE_COMM` | `USE_COMMUNICATION` | **1** | :275 | 통신 ON/OFF. OFF = others_msg≡0 |
| `VESSEL_MSG_DIM` | `MSG_DIM` | 6 | :53 | 메시지 차원(≥GOAL_SIZE 2 assert :976) |
| `VESSEL_COMM_RANGE` | `COMM_RANGE` | **300 ★**(legacy 200) | :83 | 통신 반경 = 보상 반경(2026-08-30 420→200). `vessel_gym.py` :61도 같은 env |
| `VESSEL_MAX_PARTNERS` | `MAX_COMM_PARTNERS` | 4 | :88 | nearest-K |
| `VESSEL_USE_MOE` | `USE_MOE` | **1** | :235 | 상황별 코어 5벌. 단일망 baseline은 0 명시 |
| `VESSEL_MOE_WIDTH` / `VESSEL_MOE_SHARED` | `MOE_WIDTH` / `MOE_SHARED` | 1.0 / **1 ★**(legacy 0) | :244 / :249 | iso-param 폭 / 전문가 간 인코더 공유 |
| `VESSEL_SITUATION_INPUT` | `SITUATION_INPUT` | 1 | :264 | one-hot 5 입력(0 = fc2 36/42 ablation) |
| `VESSEL_POS_GROUND` | `POS_GROUND` | 1 | :117 | relpos+msg_encoder mean 집계 |
| `VESSEL_USE_ATTENTION` / `VESSEL_ATTN_DIM` | `USE_ATTENTION` / `ATTN_DIM` | **1 ★**(legacy 0) / 32 | :107 / :108 | GroundedAttention |
| `VESSEL_CENTRAL_CRITIC` | `CENTRAL_CRITIC` | **1 ★**(legacy 0) | :172 | CTDE critic |
| `VESSEL_STATE_RECON_COEF` | `STATE_RECON_COEF` | **0.05 ★**(legacy 0.0) | :165 | 통합 상태복원 aux (>0이면 구 5계수는 0으로) |
| `VESSEL_INTENT/THREAT/GOAL_COMM/ROLE_COMM/COMM_CONSUMER_COEF` | 각 `*_COEF` | 0.0 | :128/:144/:157/:186/:199 | 구 aux 디코더 계수 |
| `VESSEL_ORACLE` | `USE_ORACLE` | 0 | :220 | 참 파트너 goal 주입 통제군(comm 승리 주장에 쓰지 않음) |
| `VESSEL_MSG_L2` / `VESSEL_MSG_GATE_L2` / `VESSEL_MSG_LR` | `MSG_L2_COEF` / `MSG_GATE_COEF` / `MSG_LR_SCALE` | **0.0002 ★**(legacy 0.001) / **0.0** / 1.0 | :92 / :96 / :89 | 게이트 페널티 0 = 06-12 기각 설계의 잔재(ablation 전용) |
| `VESSEL_RADAR_FEAT_DIM` | `RADAR_FEAT_DIM` | 30 | :49 | 인코더 출력 |
| `VESSEL_SHARED_ENCODER` | `SHARED_ENCODER` | **all ★**(legacy 0) | config 끝 | 레이더 인코더 망 간 공유(§4). 키·shape 불변 — 스냅샷이 유일한 근거 |
| `VESSEL_RADAR_ACT` / `VESSEL_RADAR_HEAD` / `VESSEL_RADAR_BOTTLENECK_CH` | `RADAR_ACT` / `RADAR_HEAD` / `RADAR_BOTTLENECK_CH` | **leaky ★**(relu) / **bottleneck ★**(flat) / 8 | config 끝 | 붕괴 완화(COLLAPSE_ROOTCAUSE §5) / fc fan-in 2880→360(커밋 fe60596). head 는 키 결정자 |
| `VESSEL_MSG_TOKEN_GAIN` / `VESSEL_CLIP_PER_MODULE` | `MSG_TOKEN_GAIN` / `CLIP_PER_MODULE` | **8.0 ★**(1.0) / **1 ★**(0) | config 끝 | attention 토큰 안 msg 상수배 / 망별 grad clip 0.5 |
| `VESSEL_LOAD_MODEL` / `VESSEL_TRAIN` / `VESSEL_MODEL_PATH` | `LOAD_MODEL` / `TRAIN_MODE` / `MODEL_PATH` | 0 / 1 / — | :280-284 | Unity 경로 로드·eval |
| `VESSEL_USE_EDITOR` / `VESSEL_NUM_ENVS` / `VESSEL_BASE_PORT` / `VESSEL_TIME_SCALE` | — | 1 / 2 / 5004 / 100 | :314-317 | Unity 환경 |
| PPO 상수 | γ 0.99 · λ 0.95 · LR 3e-4 · BATCH 2048 · `N_EPOCH` 2 · `MINIBATCH_SIZE` 512 · clip 0.2 · entropy 0.01 · value 0.5 · grad 0.5 | | :290-299 | gym 경로는 rollout 길이를 `--rollout`(기본 64)으로 받고 나머지는 config 사용(:820, :846-901) |

- `run_repro.sh common_env()` = **YUGIOH 를 명시 export**(config 기본값과 동일). `preflight` 가 export 값과 config 기본값을 대조해 드리프트면 중단. 학습 인자 `--envs 128 --vessels 16 --rollout 64 --ring 1.0 --crossing 2 --max_partners 4 --steps 16056320`(= `config.YUGIOH_ARGS`). 2026-09-04 배치 설정은 `test_golden.py BATCH_ENV`(legacy 핀 포함)에만 남아 있음.
- `config.py` import 시 `models/<COMM_FOLDER>/VesselNavigation_<시각>/logs` 디렉토리 생성 부작용(:347-348) — 스크립트에서 import만 해도 빈 폴더 생김.

---

## 8. 진단·평가 규약 (`ckpt_io.py`, `diag_ckpt.py`, `test_golden.py` — 전부 2026-09-10)

배경: 체크포인트를 여는 스크립트 9개 중 스냅샷을 읽는 건 2개뿐이었고 나머지 + `runs/m2_ablation/diag/` 10개는 env를 손으로 박아 **학습과 다른 설정으로 측정 → 측정 2회 무효.**

| 규약 | 구현 |
|---|---|
| 체크포인트는 **`ckpt_io.restore_policy()`로만** 연다 | 순서 고정: `torch.load` → msg_ln(:127)/msg_dim/radar_head(:143) 키 스니핑 → `networks` 모듈 전역 덮어쓰기(USE_ATTENTION·POS_GROUND·CENTRAL_CRITIC·STATE_RECON_COEF·_MSG_TOKEN_GAIN·_RADAR_*) → **그 다음** `CNNPolicy()`(:208) → strict 로드. `CNNPolicy.__init__`이 전역을 그 시점에 읽으므로 순서 바꾸면 무효 |
| 평가 env는 **`ckpt_io.make_env_from_snapshot()`으로만** | ring·crossing·vessels·farfield·perpair를 스냅샷에서. override는 전부 로그 |
| **`VESSEL_*`를 스크립트에서 직접 세팅 금지** | 스냅샷과 어긋나는 import-시점 값(comm_range·arm)은 기본 **중단**. 의도한 교차평가만 `allow_*` |
| 진단은 **`diag_ckpt.py`로만** | 지표 정의 = `vessel_gym_train.comm_telemetry` 하나(재구현 금지). `restore → make_env → burn → 게이트 → telemetry → JSON+CSV` |
| **게이트 3개 통과 못 하면 숫자 안 냄** | ① 조우율(sit≠0) ≥ 5% (`--min_sit_rate`) ② 설정 == 스냅샷(restore가 불일치 시 중단) ③ `--expect_vcoll` 주면 창 vColl이 평가값 ±50% 안 |
| **검증 안 된 숫자 보고 금지** | 조우율 낮은 창·다른 설정으로 잰 숫자가 두 번 보고 후 철회됨 — 그게 `diag_ckpt.py`가 생긴 이유 |
| **골든 테스트** `test_golden.py --check` | 학습기(`vessel_gym_train.py`·`networks.py`·`config.py`·`vessel_gym.py`) **변경마다.** 케이스 `default_ON` / `default_OFF` / `batch_2026_09_04_ON`, 고정시드 CPU 2 update → state_dict SHA256·곡선 CSV·스냅샷·Adam 비트 비교. 골든 `Python/golden/2026-09-10_*.json`(git 추적). `--regen`은 명시 승인 필요 |
| **스냅샷 키** | `ckpt_io.snapshot_config` — 추가 자유, **삭제·의미 변경 금지** |
| Unity ground-truth | `VESSEL_OUTCOME_LOG`(goal/collision_vessel/collision_obstacle/timeout) · `VESSEL_METRIC_LOG` **17열**(`VesselAgent.cs` :903-904: agentId,episodeIndex,outcome,steps,fuel,rudderVar,complianceMean,occlRate,commandVar,minVesselDist,nearMissSteps,straightness,headingTravel,minDCPA,dcpaBelowSteps,fuelThrust,fuelTurn). 뒤 8열 = 진단 전용, 보상 비연결 |

- `cfg_snapshot` 없는 체크포인트(2026-09-05 이전) = 집계 방식(attention/pos_ground)을 키로 알 수 없음 → **legacy 기본**(attention 0·pos_ground 1·token_gain 1·relu·agg sum)으로 감(YUGIOH 기본 아님). comm_range 불명이면 **중단** — `VESSEL_COMM_RANGE=<학습값>` + `allow_comm_range_mismatch` 로만 진행. **조용히 틀릴 수 있음** 명시 보고.
- **2026-09-10 YUGIOH 에서 발견·수정**: `restore_policy` 가 `use_moe / moe_width / moe_shared` 를 복원 안 했음 → YUGIOH 기본(공유 MoE)으로 만들면 단일망·얇게는 strict 실패, 두껍게(MOE_SHARED=0)는 5벌 인코더가 한 객체에 덮여 **마지막 전문가만 남는 조용한 오염**. 지금은 스냅샷 → 없으면 키(`experts.`)·전문가 0/1 텐서 동일성·conv/fc 채널 수(폭 역산)로 스니핑. 구조 4종 × {정상/구 스냅샷/스냅샷 없음} 12건 시뮬 통과.
- 시드 1개 단독 주장 금지, 평균엔 시드별 승패 수 동반(루트 CLAUDE.md §2).

---

- ⚠️ `_smoke_fullmoe.py` C절(PPO 미러 sum)은 2026-09-05 action_raw 변경 이전 작성 — 리팩토링 **전부터** FAIL(|lp diff| 1.9e-1, 리팩토링 전 코드로 재현). 낡은 검사이니 판정에 쓰지 말 것. 권위는 `_verify_ppo_mirror.py`(Windows 전용, Mac 은 torch↔numpy 비호환)·`_verify_comm_mirror.py`. D절도 Mac 에서 numpy 크래시.

## 9. 권위 문서 (주제별 정본 1개)

| 주제 | 정본 | 기준일 |
|---|---|---|
| **현재 상태·할 일** | `runs/STATUS.md` | 08-26 기준 + **08-31 갱신**(ring 0.7 원인 확정·1-1 철회 포함) |
| 배치 실행 계획 | `runs/ABLATION_PLAN.md` | 08-31 확정 |
| 통신 계획·사전등록 | `runs/m2_ablation/COMM_PLAN.md` | 09-04 |
| 붕괴 원인(off_s45, dying ReLU) | `Python/COLLAPSE_ROOTCAUSE.md` | 09-05 |
| H2 판정 | `runs/m2_ablation/RESULT.md` | 09-01 정리 (ANOVA F(5,12)=0.39) |
| 그림 ↔ 런 ↔ 설정 매핑 | `Python/plotting/RUNS.md` | 파일 08-20 (구조 열은 체크포인트 직접 열어 판정) |
| sim2sim 핸드오프 | `Python/SIM2SIM_HANDOFF.md` | 07-04 작성 |
| 신경망 층별 설명·근거 | `README.md` (git root) | 07-03 — **MoE 기본값(현재 ON)·msg_ln·CTDE·state_recon 미반영, 파라미터 수는 3망 합 기준** |
| 재현 실행 | `Python/run_repro.sh` | 09-05 |

**낡은 문서 — 인용 금지:**

| 파일 | 이유 |
|---|---|
| `Python/EXPERIMENT_STATUS.md` | 06-02판. obs 59D·게이트 −3 시대 |
| `Python/plotting/STATUS.md` | 08-26. `runs/STATUS.md`의 옛 사본(08-31 갱신 없음) |
| `runs/m2_ablation/diag/README_진단.md` | 09-01. **잘못된 설정으로 측정**(env 손으로 박음 — `ckpt_io.py` 헤더). 결론 인용 금지 |
| 이 파일의 2026-06-02판 | git `2e02e89`에 기록만 남김 |

---

## 10. 코드 규칙·과학적 정직성

**코드 규칙**
- **C#**: PascalCase(클래스/메서드), camelCase(지역). 주석 한국어. `[Header]` public 필드. `Debug.Log` 금지(`Debug.LogWarning`만, setup 에러). C# 변경 → **재빌드** 필수(Editor는 자동 반영).
- **Python**: snake_case. 주석 한국어/docstring 영어. **모든 상수·경로·차원은 `config.py`**(§7 예외는 통합 대상). production 코드에 bare `print()` 금지(학습 진행/에러 출력만).
- **기본값 = 비트동일 원칙**: 모든 새 기능은 토글, 끈 상태가 없는 상태와 비트동일. `test_golden.py --check`로 확인.
- **데이터 위치**: `models/`, `trajectory_data/`, `figures/`, 체크포인트는 `Assets/` *밖*(Unity 무한 import 방지). 체크포인트는 Dropbox 밖(`VESSEL_CKPT_DIR`).
- **GitHub**: git root = `Assets/Scripts/`. C# 파일 복사 금지(Unity 중복 컴파일). 원본 직접 `git add`.
- 학습기 본체(`vessel_gym_train.py` `vessel_gym.py` `networks.py` `main.py` `eval_ckpt.py` `eval_mixed.py` `ckpt_io.py` `diag_ckpt.py` `test_golden.py`) 수정 전 담당 확인 — 동시 작업 중인 경우 있음.

**과학적 정직성 (제1원칙)**
통신이 도우면 ground-truth로 입증, **안 도우면 정직하게 "안 도움"이 결론.** baseline을 불구화해 통신을 이기게 만들지 않음. H1/H2는 *달성할 목표*지 *조작으로 만들 결과*가 아님. 시드 제외는 통신에 불리한 방향으로만, 그리고 제외 전에 코드를 먼저 의심(`COLLAPSE_ROOTCAUSE.md`).
