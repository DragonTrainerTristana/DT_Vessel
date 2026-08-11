"""
Vessel Navigation ML-Agent Configuration
하이퍼파라미터 및 환경 설정

병렬 학습 시 환경변수 override 가능 (부분 목록 — 통신/attention/MoE/intent/보상 레버 전체는 CLAUDE.md env 토글 표 참조):
    VESSEL_MSG_DIM        MSG_DIM (default 6)
    VESSEL_COMM_FOLDER    COMM_FOLDER (default COMM_YES_PHASE3_NEW)
    VESSEL_BASE_PORT      BASE_PORT (default 5004)
    VESSEL_RUN_STEP       RUN_STEP (default 30000000)
    VESSEL_START_STEP     START_STEP (default 0, from-scratch)
    VESSEL_MODEL_PATH     MODEL_PATH (절대 경로)
    VESSEL_ENV_PATH       Unity build exe 경로 (default 0424)
    VESSEL_NUM_ENVS       NUM_ENVS (default 2, 에디터 모드면 무시되고 1)
    VESSEL_N_EPOCH        N_EPOCH (default 2)
    VESSEL_USE_EDITOR     '1'=Unity Editor 직결 학습(기본), '0'=빌드 exe 병렬
"""
import os

# CUDA allocator 최적화 (torch import 전 반드시 설정)
# fragmentation 50%↓, expandable segments로 OOM 회피
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                       'max_split_size_mb:128,expandable_segments:True')

import torch
import datetime

# 프로젝트 루트 경로 (Assets/Scripts/Python/ → 3단계 상위)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

# 환경변수 헬퍼 (병렬 학습 시 dim/path 외부 주입용)
def _env_int(key, default):
    v = os.environ.get(key)
    return int(v) if v is not None else default

def _env_str(key, default):
    # 빈 문자열도 default 사용 (launch script에서 VESSEL_ENV_PATH="" 전달 시 default 적용)
    v = os.environ.get(key)
    return v if v else default

# ============================================================================
# Network Architecture (GitHub 방식 - 메시지 교환)
# ============================================================================
RADAR_RAYS = 360                # ★Radar raw ray 수 (2026-06-04: C# min-pool 제거, 360 ray 그대로 송신 → Python Conv1D 학습압축)
STATE_SIZE = RADAR_RAYS         # Radar state 크기 = raw ray 360 (압축은 networks.RadarEncoder가 담당). C# GlobalScale.RADAR_RAYS와 반드시 일치
# ★RadarEncoder 신경망 압축 출력 차원 (2026-06-08: 360 ray → 30D 학습형 압축).
#   옛 min-pool은 360→30섹터 *고정·데이터손실*. 신경망은 30D로 압축하되 "무엇을 남길지" 학습 → 손실 최소화.
#   기본 30(옛 섹터수와 동일 차원). VESSEL_RADAR_FEAT_DIM로 override(예: 256 = 차원 비교 실험). 변경 시 from-scratch.
RADAR_FEAT_DIM = _env_int('VESSEL_RADAR_FEAT_DIM', 30)
GOAL_SIZE = 2                   # Goal (distance, angle)
SELF_STATE_SIZE = 4             # Self state (speed, yaw_rate, heading, rudder) - 네트워크 입력용
COLREGS_SIZE = 0               # obs 내 별도 슬롯 없음. 보상 shaping + MoE 라우팅 + obs[368] one-hot 정책 입력(기본 ON, SITUATION_INPUT)
MSG_DIM = _env_int('VESSEL_MSG_DIM', 6)   # 메시지 차원 (env override 가능)
CONTINUOUS_ACTION_SIZE = 2      # 행동 공간 차원 (rudder, thrust)
FRAMES = 3                      # Frame stacking 개수 (radar에만 적용 → RadarEncoder 입력 채널 = FRAMES)

# Unity에서 보내는 관측값 구조 (radar 360 raw ray 송신 → Python RadarEncoder가 압축; ARPA 제거 2026-06-05):
# [0:360]    Radar (360 raw ray min-distance, 1°)  ← 유일하게 frame-stack 대상(×3)
# [360:362]  Goal (distance, angle)
# [362:366]  Self state (speed, yaw_rate, heading, rudder)
# [366:368]  Position (x, z) - 통신 범위 계산용, 학습 제외
# [368:369]  Situation (COLREGs 상황 0~4) - MoE 라우터 + one-hot 5D 정책 입력(★기본 ON, VESSEL_SITUATION_INPUT=0으로 ablation)
# ★ARPA(구 [366:387]) 제거: 충돌기하는 360 raw ray + frame-stack(RadarEncoder Conv1D가 bearing-rate 학습)이 대체
POSITION_SIZE = 2               # Position (x, z) - 통신 범위 계산용, 학습 제외
SITUATION_SIZE = 1              # COLREGs 상황 라우팅 인덱스 (0~4). MoE 라우터 전용 — 네트워크 feature 입력 제외
OBSERVATION_SIZE = STATE_SIZE + GOAL_SIZE + SELF_STATE_SIZE + POSITION_SIZE + SITUATION_SIZE  # 369D

# ============================================================================
# Scale (C#의 GlobalScale과 반드시 일치해야 함)
# ============================================================================
VESSEL_SCALE = 0.2              # 0.3→0.2 (C# GlobalScale와 동기화. 0.3은 충돌 78% 학습불가였음). COMM_RANGE=2100×0.2=420m 자동. MAP_SCALE 0.1 유지
MAP_SCALE = 0.1                 # VESSEL_SCALE과 통일 (센서↔맵 스케일 정합, C# GlobalScale.MAP_SCALE과 일치)

# ============================================================================
# Communication Settings
# ============================================================================
def _env_float(key, default):
    v = os.environ.get(key)
    return float(v) if v is not None else default

COMM_RANGE = _env_float('VESSEL_COMM_RANGE', 2100 * VESSEL_SCALE) # 통신 범위 420m (C# GlobalScale.COMM_RANGE와 매칭. COLREGS_DETECTION(보상 56m)과는 분리)
# ★기본 4 (2026-07-03 사용자 결정으로 8→4 원복): nearest-4 within COMM_RANGE.
#   이력: 원래 4 → 2026-06-12에 far-band(56~420m) 커버 목적으로 8로 확대(활성 씬 16척, 밴드 점유 8~10척 실측)
#   → 2026-07-03에 4로 원복. far-band 커버가 필요한 실험은 VESSEL_MAX_PARTNERS=8로 명시 설정할 것.
#   ⚠️ COMM_CONSUMER_K는 이 값으로 clamp됨 → 기본 4에서는 consumer K ≤ 4 (msgdim 스윕의 K=6 설정은 4로 축소됨).
MAX_COMM_PARTNERS = _env_int('VESSEL_MAX_PARTNERS', 4)   # nearest-N within COMM_RANGE
MSG_LR_SCALE = _env_float('VESSEL_MSG_LR', 1.0)   # 3.0→1.0 (zero-init으로 0에서 자라는 구조: 빠른 LR은 노이즈만↑). env override
# 메시지 L2 정규화: 통신이 쓸모없으면 메시지를 0으로 우아하게 수렴(불안정 붕괴 방지).
# 통신이 도움되면 페널티 무릅쓰고 nonzero 유지 → "comm 유용성 자가검증". env로 튜닝.
MSG_L2_COEF = _env_float('VESSEL_MSG_L2', 0.001)
# 메시지 게이트 개방 페널티 — ★기본 0 (2026-06-12 채널동결 fix): 0.02 + 닫힌 init(−3) 조합은 채널
# grad가 항등 0인 상태에서 페널티만 작용해 게이트를 −8까지 단조 폐쇄(흡수상태, 체크포인트 실측).
# 게이트는 이제 중립 init(0)의 자유 다이얼(networks.py) — 페널티는 ablation용으로만 env 재활성.
MSG_GATE_COEF = _env_float('VESSEL_MSG_GATE_L2', 0.0)

# ============================================================================
# 위치 grounding + Attention 집계 (sum/mean 대체)
# ============================================================================
# receiver의 [self_state ⊕ goal]로 query, 각 partner의 [상대위치(sin,cos,거리) ⊕ msg]로
# key/value → softmax 가중선택(sum의 무차별 합 대신 "누가·어디서·지금 얼마나 중요한지" 반영).
# 출력차원 dv=MSG_DIM이라 ControlActor/Critic의 게이트·fc2 메시지슬롯 *불변*(인터페이스 동일, 연산만 추가).
# v_proj 소진폭 init(×0.1, 2026-06-12 fix) — zero-init은 직렬 곱 새들로 채널을 영구 동결시켰음(실측).
# ⚠️ rollout(aggregate_single)·update(aggregate_batch) 동일 함수형이라 PPO ratio 유효.
# 기본 OFF(=기존 sum) → 켜기 전 빌드/baseline과 100% 동일(anti-rigging). H1b regime에서 ON 비교.
USE_ATTENTION = _env_str('VESSEL_USE_ATTENTION', '0') == '1'
ATTN_DIM = _env_int('VESSEL_ATTN_DIM', 32)   # attention query/key 내부차원 (head 1개)

# ★위치 grounding (2026-07-03 기본 ON 승격, 사용자 결정): 수신 메시지에 발신자의 [상대방위 sin·cos, 거리/420]
#   3D를 붙여 msg_encoder(9→32→6 MLP)로 인코딩 후 mean 집계. "내용(학습 6D) + 주소(물리 3D)" 분리.
# 근거: 상대위치는 수신자가 국소 계산 가능(전송 불요, 채널 대역폭 0) + 실제 AIS가 위치를 메시지에 동봉 →
#   현실적 관측. comm-OFF는 others_msg≡0이라 불변(공정성 유지). 방향 귀속("어느 쪽이 위험한가")을 가능하게 함.
# 우선순위: USE_ATTENTION=1 > POS_GROUND=1 > sum/mean/scale. sum 대조군은 VESSEL_POS_GROUND=0 명시.
# ⚠️ 켜/끄면 통신 arm의 집계 함수가 달라짐 = from-scratch 대상 (msg_encoder 가중치는 항상 존재하므로
#   체크포인트 shape은 호환되나 학습 의미가 다름).
POS_GROUND = _env_str('VESSEL_POS_GROUND', '1') == '1'

# ============================================================================
# Intent self-supervised (메시지 = sender의 미래의도; Phase 2)
# ============================================================================
# 메시지 latent이 sender의 *미래 K-step 궤적/heading*을 디코드가능하게 인코딩하도록 self-supervised
# 보조손실을 건다 → radar+frame-stack(관측된 운동학)이 구조적으로 못 주는 *미래 maneuver* 정보를 담아 시간축에서 비잉여.
# ★self-prediction(자기 미래를 자기 메시지로 예측): receiver step과 정렬 → 파트너 정렬 silent-failure 0.
# ★라벨 = trajectory에서 추출한 *실제 미래변위*(self-supervised), reward·advantage와 완전분리 = anti-rigging.
# ★정책/가치 경로 무오염(별도 IntentDecoder head). receiver는 여전히 게이트로 메시지 무시 가능 = H1a 보존.
# default 0.0 = OFF = 기존과 비트동일(IntentDecoder 미호출, own_future 미계산).
INTENT_COEF = _env_float('VESSEL_INTENT_COEF', 0.0)
INTENT_K = _env_int('VESSEL_INTENT_K', 3)            # 예측 미래시점 개수 (h=HORIZON×{1..K})
INTENT_HORIZON = _env_int('VESSEL_INTENT_HORIZON', 12)  # 시점 간격(step). 12/24/36 = 1.2/2.4/3.6초(0.4s/결정)
INTENT_POS_SCALE = _env_float('VESSEL_INTENT_POS_SCALE', 56.0)  # 변위 정규화(≈radar 56m); 디코더 타깃 O(1)화

# ============================================================================
# Threat-relay self-supervised (메시지 = sender가 본 제3 위협 기하; L1, Phase 2)
# ============================================================================
# 메시지 latent이 sender의 *ego-radar가 본 top-K 최근접 위협*의 기하(방위·거리·closing)를
# 디코드가능하게 인코딩하도록 self-supervised 보조손실을 건다 → occlusion(VESSEL_LOS_GATE)으로
# receiver가 못 보는 제3 위협을 sender 메시지로 복원 가능 → 통신이 *필수*가 되는 CTDE-lite regime.
# ★라벨 = sender 자기 ego-radar 360 ray의 top-K 최근접 위협 [sin방위,cos방위,거리/maxrange,closing].
#   privileged 아님 — occlusion으로 가린 위협은 ray에 막혀 안 잡힘 = sender가 실제 본 것만(정직, anti-rigging).
# ★reward·advantage와 완전분리(self-supervised) = anti-rigging. 별도 ThreatDecoder head → 정책/가치 무오염.
# ★receiver는 게이트로 메시지 무시 가능 = H1a 보존. MessageActor로만 gradient.
# default 0.0 = OFF = 기존과 비트동일(ThreatDecoder 미호출, own_threat 라벨 미계산).
THREAT_COEF = _env_float('VESSEL_THREAT_COEF', 0.0)
THREAT_K = _env_int('VESSEL_THREAT_K', 3)            # 복원할 top-K 최근접 위협 개수 (거리순)

# ============================================================================
# Goal-broadcast self-supervised (메시지 = sender의 목적지/의도; L4, Phase 2)
# ============================================================================
# ★사용자 핵심 비전(2026-06-16): 통신 거리(420m) >> 레이더(56m)라 56~420m 배는 "레이더로 못 보지만
#   통신으로 잡히는" 배 = 정보 비대칭 *이미 존재*(섬/occlusion 불요). 그 먼 배들의 *목적지/진로*를
#   통신으로 알면 "쟤는 저리 가니까 나는 미리 살짝" 멀리서 부드럽게 협응(COLREGs↑·연료↓·궤적smooth↑).
# ★메시지 latent이 sender의 *목적지(goal: 거리·방위)*를 디코드가능하게 인코딩하도록 self-supervised.
#   라벨 = sender 자기 goal(obs에 이미 있음 → 별도 라벨·재빌드 불요, self-prediction). reward 비연결=anti-rigging.
# ★별도 GoalDecoder head → 정책/가치 무오염(H1a 보존). receiver는 게이트로 무시 가능.
# default 0.0 = OFF = 비트동일(GoalDecoder 미호출).
GOAL_COMM_COEF = _env_float('VESSEL_GOAL_COMM_COEF', 0.0)

# ============================================================================
# Role-broadcast self-supervised (메시지 = sender의 COLREGs 상황/역할; Phase 2)
# ============================================================================
# ★사용자 비전(C): 통신으로 "누가 양보(give-way)·누가 직진(stand-on)"을 미리 공유 → COLREGs 협응↑.
#   메시지 latent이 sender의 *COLREGs 상황/역할(situation: 0~4)*을 디코드가능하게 인코딩하도록 self-supervised
#   분류손실(cross-entropy)을 건다. situation = sender가 본 가장 위험한 배와의 상황(MoE 라우터와 동일 ground-truth).
# ★receiver가 attention으로 sender 역할을 알면 협응: "쟤가 give-way(3)니 난 stand-on(직진) 유지",
#   "쟤가 head-on(1)이니 나도 우현" — 메시지에 역할 인코딩 → fc2가 학습으로 활용(Threat/GoalDecoder 동일 원리).
# ★라벨 = sender 자기 situation(이미 obs/메모리·evaluate_actions 인자 → 별도 라벨 텐서·재빌드 불요, self-prediction).
#   reward·advantage와 완전분리(self-supervised) = anti-rigging. 별도 RoleDecoder head → 정책/가치 무오염(H1a 보존).
# ★comm-OFF는 others_msg≡0이라 RoleDecoder 무영향(불변). receiver는 게이트로 무시 가능 = H1a 보존.
# default 0.0 = OFF = 비트동일(RoleDecoder 미호출, situation 라벨 미사용).
ROLE_COMM_COEF = _env_float('VESSEL_ROLE_COMM_COEF', 0.0)

# ============================================================================
# ★C5c: 수신측 decode-in-policy (2026-06-22, 4차 논검 진단)
# ============================================================================
# ★근본 진단: 기존 aux decoder(intent/threat/goal/role)는 전부 own_msg=msg_actor()를 재실행 →
#   gradient가 *생산자(MessageActor)*에게만 흐름. 수신자 정책(ControlActor)이 others_msg를 *쓰도록*
#   강제하는 loss 항이 없었음 = "채널이 z 인코딩"과 "정책이 z 사용"의 구조적 단절(C5c). 채널이 LIVE여도
#   수신정책엔 약한 암묵 PPO 신호만 → density regime서 comm 무승부의 binding 원인(체크포인트·코드 실증).
# ★fix: ControlActor.consumer_decoder가 backbone z(=fc2(...,gated others_msg))로 *파트너 의도(goal)*를
#   복원 → MSE 손실이 fc2 메시지슬라이스로 흘러 "수신 정책이 파트너 의도를 표상"하게 강제(수신측 gradient).
#   producer측 GoalDecoder(메시지에 자기 goal 인코딩)와 짝 → 송수신 폐루프. reward·advantage 무연결=anti-rigging.
# ★comm-OFF는 others_msg≡0이라 consumer 손실 미가산(USE_COMM 게이트) → OFF 불변(공정). default 0=OFF.
COMM_CONSUMER_COEF = _env_float('VESSEL_COMM_CONSUMER_COEF', 0.0)
# ★H2 per-slot 재구성 (2026-06-22, H2-scaling-by-design): consumer가 masked-MEAN(2D, MSG_DIM≥2서 포화=flat 보장)
#   대신 **nearest-K 파트너 각각의 의도(goal)를 슬롯별 복원** → K개 의도가 MSG_DIM others_msg 하나를 공유 =
#   rate-distortion 병목이 MSG_DIM → 차원↑ = 더 많은 이웃 의도 전달 → H2 스케일 가능. K=1이면 nearest만(mean 아님).
#   ⚠️ ground-truth 스케일엔 per-pair 보상(T4, C#)+행동head coupling도 필요(이것만으론 aux MSE만 스케일).
#   anti-Schelling: z에서만 디코드(relpos는 attention 주소로만, 재구성 입력/타깃 아님). 천장=내재의도차원×동시이웃수.
COMM_CONSUMER_K = _env_int('VESSEL_COMM_CONSUMER_K', 3)   # 복원 nearest-K 슬롯 (consumer_coef>0일 때만 의미)
# ★H2 행동-head coupling (2026-06-22): consumer 재구성(decode)을 fc3 입력에 concat → "재구성↓ = 행동↑"이
#   forward 항등이 됨 → ground-truth가 aux MSE를 *추종*(단순 공유-z aux는 "정렬되길 바람"일 뿐). decode는 z에서 산출 →
#   forward·get_logprob_entropy(둘 다 _head 경유) 동일 → PPO mirror 보존. default 0=OFF(fc3 입력 128=baseline 비트동일).
#   ★consumer_coef>0와 함께 써야 의미(coupling만 켜고 미supervised면 decode가 노이즈로 행동에 유입). ground-truth 스케일엔 T4(C#)도 필요.
COMM_CONSUMER_COUPLING = _env_str('VESSEL_COMM_CONSUMER_COUPLING', '0') == '1'

# ============================================================================
# ★oracle-OFF 통제 arm (2026-06-22): C1·C4·C5 판정 분리용 진단 계기
# ============================================================================
# ★others_msg를 학습채널 대신 *참 파트너 goal*(privileged, 채널 우회)로 주입 → 수신자가 학습된 메시지
#   디코딩 없이 파트너 의도를 직접 받음. 3분할 판정: oracle≈OFF→task가 comm 불필요(C1/C4 거짓, 종료);
#   oracle>OFF & ON≈OFF→C5가 병목(채널학습 문제, C5c 수정 가치); oracle>OFF & ON≈oracle→comm 정당승리.
# ★진단/통제용(comm-ON arm 아님) — comm 승리 주장에 쓰지 않음 = anti-rigging. rollout·update 미러(networks.py).
# ★권장: VESSEL_ORACLE=1 + VESSEL_USE_COMM=1 + aux/consumer COEF=0(정보 직접이라 불요). default 0=OFF.
USE_ORACLE = _env_str('VESSEL_ORACLE', '0') == '1'

# ============================================================================
# COLREGs Mixture-of-Experts (상황별 정책 head hard-routing)
# ============================================================================
# Unity가 판정한 COLREGs 상황(cachedDangerSituation, obs[389]=마지막 슬롯, 0~4)으로 정책 head를 hard-route.
#   0=None(조우없음/항해) 1=HeadOn 2=CrossingStandOn 3=CrossingGiveWay 4=Overtaking (COLREGsHandler enum 일치).
# 공유 backbone(radar 인지 + 통신 융합 → z 128D) 위에 상황별 maneuvering head(fc3→μ,σ) 5개.
#   단일 정책이 4상황의 상충하는 회피규칙(head-on→우현 / stand-on→유지 / give-way→우현+감속 / overtake→keep clear)을
#   평균내며 간섭하는 걸 방지 → 상황별 전문화.
# ★2026-08-04 기본 ON 승격(사용자 결정): 무설정 학습이 MoE를 쓴다. 단일망 baseline은 VESSEL_USE_MOE=0으로
#   명시 실행 = 기존 단일망과 *비트동일*(기존 체크포인트 strict 로드 가능) = anti-rigging 공정 baseline.
#   MoE(5x/iso)가 이 single을 ground-truth로 이겨야 진짜(H4). Critic은 USE_MOE=1일 때만 상황 one-hot 조건화.
# ★라우터=privileged 상황(보상과 동일 ground-truth) → CTDE 일관. situation은 transition마다 저장돼
#   rollout==update 동일 라우팅(PPO ratio 유효, 메시지 집계 일관성과 같은 원리).
USE_MOE = _env_str('VESSEL_USE_MOE', '1') == '1'   # ★Thick_MoE 브랜치: MoE 기본 ON (폭 1.0 = 전문가마다 단일망과 같은 폭, 총 ~5배)
NUM_COLREGS_SITUATIONS = 5   # None/HeadOn/CrossingStandOn/CrossingGiveWay/Overtaking
# ★iso-parameter MoE (2026-07-03): MoE 전문가 코어의 내부 폭 배수 (conv 채널·radar feat·hidden·fc3에 적용).
#   1.0(기본) = 기존 MoE — 코어당 단일망과 동일 폭, 총 파라미터 약 5배.
#   0.44 = 5코어 합계 ≈ 단일망 파라미터 → "상황별 전문화 효과"를 "용량 5배 효과"와
#   분리하는 공정 비교 arm (single vs moe5x vs moe-iso 3-arm 가능).
#   실측(2026-07-03 스모크): 0.44 → 3망 합 358,270 vs 단일 364,397 (−1.7%). 0.45는 반올림 계단으로 +7.9% → 0.44 권장.
#   USE_MOE=1일 때만 적용, 단일망(USE_MOE=0)은 항상 폭 1.0 = 기존과 비트동일.
#   메시지 6D·행동 2D·상황 one-hot 등 외부 인터페이스 불변. 폭 변경 시 체크포인트 비호환 = from-scratch.
MOE_WIDTH = _env_float('VESSEL_MOE_WIDTH', 1.0)
# ★공유지각 MoE (2026-08-08, H4 설계수정): RadarEncoder(코어 파라미터 ~82.5%) 1벌을 5개 전문가가 공유,
#   결정부(fc2·gate·consumer·fc3·head)만 상황별 5벌. iso-MoE 붕괴의 근본원인(지배상황 코어의
#   지각 용량 1/5 축소 — 실측 8.5M goal 3%)을 제거: 지각은 전체 데이터로 학습, 라우팅은 결정 계층만 특화.
#   총 파라미터 ≈ 단일망 ×1.7 (5x의 ×5 대비). USE_MOE=1일 때만 유효, MOE_WIDTH=1.0과 함께 쓸 것.
MOE_SHARED = _env_str('VESSEL_MOE_SHARED', '0') == '1'

# ============================================================================
# ★COLREGs situation 정책 입력 (2026-07-02 도입, 2026-07-03 기본 ON 승격): obs[368] 상황(0~4)을
#   one-hot 5D로 MessageActor·ControlActor·Critic fc2에 직접 입력 (MoE 라우팅과 별개, 병행 가능).
# ============================================================================
# 근거: 상황 판정은 자기 센서 반경(56m=레이더 범위) 내 정보로 산출되고, 실선도 ARPA/AIS가 동일한
#   상황 분류를 제공 → 현실적 관측으로 간주. 통신 ON/OFF 양 arm에 동일 적용이라 anti-rigging 안전
#   (오히려 통신이 전달할 정보를 공짜로 주는 방향 = 통신 가치 평가엔 보수적).
# ★기본 1 = 표준 아키텍처 (사용자 결정 2026-07-03): MoE OFF=단일망+situation 입력,
#   MoE ON=상황별 5벌+각 코어에도 situation 입력(expert 내 상수라 중복이나 무해).
#   =0은 ablation 전용(situation 입력 제거, fc2 36/42 구조 — 07-03 이전 체크포인트 호환).
# ⚠️ 기본 변경으로 fc2 입력 +5 (Msg 41 / Ctrl·Critic 47) → 이전 체크포인트와 shape 불일치 = from-scratch.
# ⚠️ 라벨 결함(1스텝 stale, argmax-risk 1척 기준)이 정책 입력으로 직결됨을 인지.
#   (0~5° None 사각지대는 2026-07-03 COLREGsHandler.AnalyzeSituation fix로 폐쇄 — 재빌드 필요)
SITUATION_INPUT = _env_str('VESSEL_SITUATION_INPUT', '1') == '1'

# Terminal reward 판별 threshold (C# collisionPenalty=-100, spinningPenalty=-80 기준)
COLLISION_REWARD_THRESHOLD = -150  # collision: reward < -150 (collisionPenalty -300과 짝. 충돌 종료~-300≪-150, 정상종료~±2 → 분류 신뢰)
SPINNING_REWARD_THRESHOLD = -50    # spinning: -90 < reward < -50

# ============================================================================
# Training Phase (2-Phase Learning)
# ============================================================================
# Phase 1: USE_COMMUNICATION = False (자기 obs만으로 기본 navigation 학습)
# Phase 2: USE_COMMUNICATION = True (msg 통신 추가해서 fine-tune)
USE_COMMUNICATION = _env_str('VESSEL_USE_COMM', '1') == '1'   # 통신 ON/OFF (env 토글: ON vs OFF 비교용. 기본 ON)

# ============================================================================
# Training Mode
# ============================================================================
LOAD_MODEL = (_env_str('VESSEL_LOAD_MODEL', '0') == '1')   # env로 켜면 MODEL_PATH 로드(관찰/이어학습용). 기본 from-scratch
TRAIN_MODE = (_env_str('VESSEL_TRAIN', '1') == '1')   # 0=eval: PPO 업데이트 스킵 — LOAD_MODEL=1과 조합해
                                                      #   학습된 정책의 ground-truth 평가 (sim2sim 판정관 단계, 2026-07-06)
_default_model_path = os.path.join(PROJECT_ROOT, "models", "COMM_NON", "VesselNavigation_20260419_194205", "policy_step_3220000.pth")
MODEL_PATH = _env_str('VESSEL_MODEL_PATH', _default_model_path)
START_STEP = _env_int('VESSEL_START_STEP', 0)   # from-scratch=0 (LOAD_MODEL=False면 main.py가 0으로 강제)

# ============================================================================
# PPO Hyperparameters
# ============================================================================
DISCOUNT_FACTOR = 0.99          # Gamma: 할인율 (0.95 → 0.99, 장기 보상 중시)
GAE_LAMBDA = 0.95               # GAE lambda: advantage estimation 파라미터
LEARNING_RATE = 3e-4            # Adam optimizer learning rate (1e-4 → 3e-4)
BATCH_SIZE = 2048               # Mini-batch 크기 (4096 → 2048, 더 자주 업데이트)
N_EPOCH = _env_int('VESSEL_N_EPOCH', 2)  # PPO epoch (4→2: update 시간 1/2)
MINIBATCH_SIZE = _env_int('VESSEL_MINIBATCH_SIZE', 512)  # PPO mini-batch (BATCH_SIZE는 rollout 길이로 분리)
EPSILON = 0.2                   # PPO clipping epsilon
ENTROPY_BONUS = 0.01            # Entropy bonus 계수 (0.005 → 0.01, 탐색 유지)
CRITIC_LOSS_WEIGHT = 0.5        # Value loss 가중치
MAX_GRAD_NORM = 0.5             # Gradient clipping norm

# ============================================================================
# Training Schedule
# ============================================================================
RUN_STEP = _env_int('VESSEL_RUN_STEP', 30000000)   # env override 가능 (eval도 env로 길이 지정 — 옛 'else 0' 분기는 TRAIN_MODE 상수 시절 사문)
MAX_STEPS = RUN_STEP            # main.py가 start_step + MAX_STEPS 까지 학습
UPDATE_INTERVAL = BATCH_SIZE    # PPO 업데이트 간격

# ============================================================================
# Unity Environment
# ============================================================================
# Editor 모드: Unity Editor에서 ▶Play 로 학습 (빌드 불필요, 최신 C# 즉시 반영).
# main.py가 file_name=None으로 연결 → Unity에서 Play 누르면 학습 시작.
# 에디터는 인스턴스 1개만 가능 → NUM_ENVS 자동 1. 빌드로 되돌리려면 VESSEL_USE_EDITOR=0.
USE_EDITOR = _env_str('VESSEL_USE_EDITOR', '1') == '1'          # 기본 ON (편집 즉시 반영)
NUM_ENVS = 1 if USE_EDITOR else _env_int('VESSEL_NUM_ENVS', 2)  # 에디터=1, 빌드=병렬
BASE_PORT = _env_int('VESSEL_BASE_PORT', 5004)   # env override 가능 (병렬 학습 시 충돌 회피)
TIME_SCALE = _env_float('VESSEL_TIME_SCALE', 100.0)   # 시뮬 속도. 관찰 시 VESSEL_TIME_SCALE=2~3 (천천히 보이게)
# ★cross-platform: 프로젝트 루트 기준 상대경로 (Mac/Windows 공용).
#   ⚠️ Mac에선 이 .exe가 실행 안 됨 → 기본 VESSEL_USE_EDITOR=1(에디터 직결)로 학습하거나
#      Mac 빌드를 VESSEL_ENV_PATH로 지정할 것.
_default_env_path = os.path.join(PROJECT_ROOT, "Build", "Vessel_MLAgent.exe")
ENV_PATH = _env_str('VESSEL_ENV_PATH', _default_env_path)   # Server build 경로로 env override 가능

# ============================================================================
# Paths and Logging
# ============================================================================
ENV_NAME = "VesselNavigation"
DATE_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 이어서 학습할 폴더 (None이면 새 폴더 생성)
RESUME_FOLDER = None            # Phase 1: 새 폴더 생성

# 통신 모드에 따라 저장 경로 분리 (env override로 dim별 분리 가능)
COMM_FOLDER = _env_str('VESSEL_COMM_FOLDER', "COMM_YES_PHASE3_NEW")

if RESUME_FOLDER and LOAD_MODEL:
    SAVE_PATH = os.path.join(PROJECT_ROOT, "models", COMM_FOLDER, RESUME_FOLDER)
else:
    SAVE_PATH = os.path.join(PROJECT_ROOT, "models", COMM_FOLDER, f"{ENV_NAME}_{DATE_TIME}")

LOG_DIR = os.path.join(SAVE_PATH, 'logs')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 디렉토리 생성
# exist_ok: DATE_TIME이 초 단위라 같은 초에 두 학습을 띄우면 폴더 이름이 겹친다.
# 검사-후-생성 사이에 다른 프로세스가 먼저 만들면 FileExistsError로 죽어버려 실행이 통째로 날아갔다.
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

VALUE_LOSS_COEF = CRITIC_LOSS_WEIGHT   # main.py PPO total-loss에서 사용 (CRITIC_LOSS_WEIGHT alias)

def get_config_dict():
    """설정을 딕셔너리로 반환 (로깅용)"""
    return {
        'state_size': STATE_SIZE,
        'observation_size': OBSERVATION_SIZE,
        'msg_dim': MSG_DIM,
        'use_communication': USE_COMMUNICATION,
        'use_moe': USE_MOE,
        'moe_width': MOE_WIDTH,
        'moe_shared': MOE_SHARED,
        'situation_input': SITUATION_INPUT,
        'num_colregs_situations': NUM_COLREGS_SITUATIONS,
        'use_attention': USE_ATTENTION,
        'attn_dim': ATTN_DIM,
        'pos_ground': POS_GROUND,
        'intent_coef': INTENT_COEF,
        'intent_k': INTENT_K,
        'intent_horizon': INTENT_HORIZON,
        'threat_coef': THREAT_COEF,
        'threat_k': THREAT_K,
        'continuous_action_size': CONTINUOUS_ACTION_SIZE,
        'frames': FRAMES,
        'learning_rate': LEARNING_RATE,
        'discount_factor': DISCOUNT_FACTOR,
        'gae_lambda': GAE_LAMBDA,
        'batch_size': BATCH_SIZE,
        'n_epoch': N_EPOCH,
        'epsilon': EPSILON,
        'entropy_bonus': ENTROPY_BONUS,
        'critic_loss_weight': CRITIC_LOSS_WEIGHT,
        'max_grad_norm': MAX_GRAD_NORM,
        'max_steps': MAX_STEPS,
        'update_interval': UPDATE_INTERVAL,
        'device': str(DEVICE),
        'time_scale': TIME_SCALE,
        # ★comm arm 식별 필드 (2026-06-12): 과거엔 snapshot에 없어서 run 식별이 가중치 포렌식에 의존했음
        'seed': os.environ.get('VESSEL_SEED', '42'),
        'agg_mode': os.environ.get('VESSEL_AGG_MODE', 'sum'),
        'comm_range': COMM_RANGE,
        'max_comm_partners': MAX_COMM_PARTNERS,
        'msg_l2_coef': MSG_L2_COEF,
        'msg_gate_coef': MSG_GATE_COEF,
        'msg_lr_scale': MSG_LR_SCALE,
        'radar_feat_dim': RADAR_FEAT_DIM,
        'env_farfield_coef': os.environ.get('VESSEL_FARFIELD_COEF', '(unset=0)'),
        'env_risk_range': os.environ.get('VESSEL_RISK_RANGE', '(unset=56)'),
        'env_radar_range': os.environ.get('VESSEL_RADAR_RANGE', '(unset=default)'),
        'env_crossing': os.environ.get('VESSEL_CROSSING', '(unset=0)'),
        # ★density regime 식별 필드 (2026-06-21): spawn 기하 레버(C# 적용). 미설정=baseline 비트동일.
        #   런 식별·재현용 — 보상/네트워크 무관(289행 주석과 동일 목적). VESSEL_USE_COMM과 독립.
        'env_spawn_ring_scale': os.environ.get('VESSEL_SPAWN_RING_SCALE', '(unset=1.0)'),
        'env_vessel_count': os.environ.get('VESSEL_VESSEL_COUNT', '(unset=scene)'),
        'env_speed_mult_min': os.environ.get('VESSEL_SPEED_MULT_MIN', '(unset=0.8)'),
        'env_speed_mult_max': os.environ.get('VESSEL_SPEED_MULT_MAX', '(unset=1.8)'),
        # ★C5c·oracle 식별 (2026-06-22): 수신측 decode-in-policy 계수 + oracle 통제 arm 플래그.
        #   goal/role_comm은 producer측 인코딩 계수 — C5c sweep은 GOAL_COMM(생산)+CONSUMER(수신) 짝이라 함께 기록.
        'goal_comm_coef': GOAL_COMM_COEF,
        'role_comm_coef': ROLE_COMM_COEF,
        'comm_consumer_coef': COMM_CONSUMER_COEF,
        'comm_consumer_k': COMM_CONSUMER_K,
        'comm_consumer_coupling': COMM_CONSUMER_COUPLING,
        'use_oracle': USE_ORACLE
    }
