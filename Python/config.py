"""
Vessel Navigation ML-Agent Configuration
하이퍼파라미터 및 환경 설정
"""
import torch
import os
import datetime

# 프로젝트 루트 경로 (Assets/Scripts/Python/ → 3단계 상위)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

# ============================================================================
# Network Architecture (GitHub 방식 - 메시지 교환)
# ============================================================================
STATE_SIZE = 360                # Radar state 크기 (360 rays, 1도 간격)
GOAL_SIZE = 2                   # Goal (distance, angle)
SELF_STATE_SIZE = 4             # Self state (speed, yaw_rate, heading, rudder) - 네트워크 입력용
COLREGS_SIZE = 5                # COLREGs one-hot (None, HeadOn, CrossingStandOn, CrossingGiveWay, Overtaking)
MSG_DIM = 6                     # 메시지 차원 (6D로 압축)
CONTINUOUS_ACTION_SIZE = 2      # 행동 공간 차원 (rudder, thrust)
FRAMES = 3                      # Frame stacking 개수

# Unity에서 보내는 관측값 구조:
# [0:360]   Radar (360 rays)
# [360:362] Goal (distance, angle)
# [362:366] Self state (speed, yaw_rate, heading, rudder)
# [366:371] COLREGs (5D one-hot)
# [371:373] Position (x, z) - 통신 범위 계산용, 학습 제외
POSITION_SIZE = 2               # Position (x, z) - 통신 범위 계산용, 학습 제외
OBSERVATION_SIZE = STATE_SIZE + GOAL_SIZE + SELF_STATE_SIZE + COLREGS_SIZE + POSITION_SIZE  # 373D

# ============================================================================
# Scale (C#의 GlobalScale과 반드시 일치해야 함)
# ============================================================================
VESSEL_SCALE = 0.1              # 배 자체 크기 (길이/속도/센서)
MAP_SCALE = 1.0                 # 월드맵 내 활동 영역 (spawn zone, goal distance)

# ============================================================================
# Communication Settings
# ============================================================================
COMM_RANGE = 300 * VESSEL_SCALE # 통신 범위 (미터) - 배 센서 기반 → VESSEL_SCALE
MAX_COMM_PARTNERS = 4           # 최대 통신 파트너 수
MSG_ANNEAL_STEPS = 500000       # 메시지 기여도 0→1 선형 증가 스텝 수 (Phase 2 전환 안정화)
MSG_LR_SCALE = 3.0              # MessageActor 학습률 배수 (untrained → 빠르게 학습)
COLREGS_LOSS_COEF = 0.1         # COLREGs classifier auxiliary loss 계수

# Terminal reward 판별 threshold (C# collisionPenalty=-100, spinningPenalty=-80 기준)
COLLISION_REWARD_THRESHOLD = -90   # collision: reward < -90
SPINNING_REWARD_THRESHOLD = -50    # spinning: -90 < reward < -50

# ============================================================================
# Training Phase (2-Phase Learning)
# ============================================================================
# Phase 1: USE_COMMUNICATION = False (자기 obs만으로 기본 navigation 학습)
# Phase 2: USE_COMMUNICATION = True (msg 통신 추가해서 fine-tune)
USE_COMMUNICATION = True        # Phase 3: 통신 ON (test용)

# ============================================================================
# Training Mode
# ============================================================================
LOAD_MODEL = True               # Phase 3: 최근 모델 로드
TRAIN_MODE = True               # 학습 모드
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "COMM_NON", "VesselNavigation_20260419_194205", "policy_step_3220000.pth")
START_STEP = 3220000            # Phase 3: 이전 학습 이어서

# ============================================================================
# PPO Hyperparameters
# ============================================================================
DISCOUNT_FACTOR = 0.99          # Gamma: 할인율 (0.95 → 0.99, 장기 보상 중시)
GAE_LAMBDA = 0.95               # GAE lambda: advantage estimation 파라미터
LEARNING_RATE = 3e-4            # Adam optimizer learning rate (1e-4 → 3e-4)
BATCH_SIZE = 2048               # Mini-batch 크기 (4096 → 2048, 더 자주 업데이트)
N_EPOCH = 4                     # PPO epoch 수
EPSILON = 0.2                   # PPO clipping epsilon
ENTROPY_BONUS = 0.01            # Entropy bonus 계수 (0.005 → 0.01, 탐색 유지)
CRITIC_LOSS_WEIGHT = 0.5        # Value loss 가중치
MAX_GRAD_NORM = 0.5             # Gradient clipping norm

# ============================================================================
# Training Schedule
# ============================================================================
RUN_STEP = 30000000 if TRAIN_MODE else 0  # 전체 학습 스텝
MAX_STEPS = RUN_STEP            # 전체 학습 스텝 (30,000,000)
UPDATE_INTERVAL = BATCH_SIZE    # PPO 업데이트 간격 (N_STEP)

# ============================================================================
# Unity Environment
# ============================================================================
NUM_ENVS = 4                    # 병렬 환경 수 (빌드 exe 사용)
BASE_PORT = 5004                # Unity 통신 시작 포트
TIME_SCALE = 100.0              # 시뮬레이션 속도 (headless 빌드용)
ENV_PATH = r"c:\Users\sengh\Dropbox\Private_Paper_Project\Vessel\Vessel_MLAgent\Build\0424\Vessel_MLAgent.exe"

# ============================================================================
# Paths and Logging
# ============================================================================
ENV_NAME = "VesselNavigation"
DATE_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 이어서 학습할 폴더 (None이면 새 폴더 생성)
RESUME_FOLDER = None            # Phase 1: 새 폴더 생성

# 통신 모드에 따라 저장 경로 분리
COMM_FOLDER = "COMM_YES_PHASE3_NEW"  # Phase 3: 통신 ON 새 폴더

if RESUME_FOLDER and LOAD_MODEL:
    SAVE_PATH = os.path.join(PROJECT_ROOT, "models", COMM_FOLDER, RESUME_FOLDER)
else:
    SAVE_PATH = os.path.join(PROJECT_ROOT, "models", COMM_FOLDER, f"{ENV_NAME}_{DATE_TIME}")

LOG_DIR = os.path.join(SAVE_PATH, 'logs')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 디렉토리 생성
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# ============================================================================
# Aliases (backward compatibility)
# ============================================================================
N_STEP = BATCH_SIZE
VALUE_LOSS_COEF = CRITIC_LOSS_WEIGHT
ENTROPY_COEF = ENTROPY_BONUS
GRAD_CLIP_MAX_NORM = MAX_GRAD_NORM
MSG_ACTION_SPACE = MSG_DIM  # backward compatibility

def get_config_dict():
    """설정을 딕셔너리로 반환 (로깅용)"""
    return {
        'state_size': STATE_SIZE,
        'observation_size': OBSERVATION_SIZE,
        'msg_dim': MSG_DIM,
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
        'time_scale': TIME_SCALE
    }
