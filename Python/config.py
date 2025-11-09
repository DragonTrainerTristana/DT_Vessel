"""
Vessel Navigation ML-Agent Configuration
하이퍼파라미터 및 환경 설정
"""
import torch
import os
import datetime

# ============================================================================
# Network Architecture (Updated to MDPI 2024)
# ============================================================================
STATE_SIZE = 180                # 단일 에이전트 state 크기 (30 regions × 6 params)
MSG_ACTION_SPACE = 6            # 메시지 공간 차원
CONTINUOUS_ACTION_SIZE = 2      # 행동 공간 차원 (rudder, thrust)
FRAMES = 3                      # Frame stacking 개수
N_AGENT = 4                     # 최대 이웃 에이전트 수
NEIGHBOR_STATE_SIZE = 35        # 이웃 state 크기 (compressed radar=24, vessel=4, goal=3, fuzzy_colregs=4)

# ============================================================================
# Training Mode
# ============================================================================
LOAD_MODEL = False              # 저장된 모델 로드 여부
TRAIN_MODE = True               # 학습 모드 (False: 평가 모드)

# ============================================================================
# PPO Hyperparameters
# ============================================================================
DISCOUNT_FACTOR = 0.95          # Gamma: 할인율
LEARNING_RATE = 1e-4            # Adam optimizer learning rate
BATCH_SIZE = 2048               # Mini-batch 크기
N_EPOCH = 4                     # PPO epoch 수
EPSILON = 0.2                   # PPO clipping epsilon
ENTROPY_BONUS = 0.005           # Entropy bonus 계수
CRITIC_LOSS_WEIGHT = 0.5        # Value loss 가중치
MAX_GRAD_NORM = 0.5             # Gradient clipping norm

# ============================================================================
# Training Schedule
# ============================================================================
MAX_STEPS = 2000                # 에피소드당 최대 스텝
RUN_STEP = 30000000 if TRAIN_MODE else 0  # 전체 학습 스텝
UPDATE_INTERVAL = BATCH_SIZE    # PPO 업데이트 간격 (N_STEP)
SAVE_INTERVAL = 100             # 모델 저장 간격 (에피소드)
NUM_EPISODES = RUN_STEP // MAX_STEPS  # 총 에피소드 수

# ============================================================================
# Unity Environment
# ============================================================================
WORKER_ID = 0                   # Unity 환경 worker ID
BASE_PORT = 5005                # Unity 통신 포트
TIME_SCALE = 3.0                # 시뮬레이션 속도 (1.0 = 실시간, 20.0 = 20배속)

# ============================================================================
# Paths and Logging
# ============================================================================
ENV_NAME = "VesselNavigation"
DATE_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = os.path.join(".", "models", f"{ENV_NAME}_{DATE_TIME}")
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

def get_config_dict():
    """설정을 딕셔너리로 반환 (로깅용)"""
    return {
        'state_size': STATE_SIZE,
        'msg_action_space': MSG_ACTION_SPACE,
        'continuous_action_size': CONTINUOUS_ACTION_SIZE,
        'frames': FRAMES,
        'n_agent': N_AGENT,
        'learning_rate': LEARNING_RATE,
        'discount_factor': DISCOUNT_FACTOR,
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
