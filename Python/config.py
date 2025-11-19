"""
Vessel Navigation ML-Agent Configuration
하이퍼파라미터 및 환경 설정
"""
import torch
import os
import datetime

# ============================================================================
# Network Architecture (GitHub + COLREGs + Full Neighbor Obs)
# ============================================================================
STATE_SIZE = 360                # Radar state 크기 (360 rays)
SELF_STATE_SIZE = 6             # Self state (goal_dist, goal_angle, speed, yaw_rate, heading, rudder)
NEIGHBOR_OBS_SIZE = 371         # Neighbor observation (360 radar + 2 goal + 2 speed + 5 colregs + 1 heading + 1 rudder)
MAX_NEIGHBORS = 4               # 최대 이웃 에이전트 수
COLREGS_SIZE = 5                # COLREGs one-hot (None, HeadOn, CrossingStandOn, CrossingGiveWay, Overtaking)
MSG_ACTION_SPACE = 6            # 메시지 공간 차원 (sigmoid: 6D, tanh: 6D → total 12D)
CONTINUOUS_ACTION_SIZE = 2      # 행동 공간 차원 (rudder, thrust)
FRAMES = 3                      # Frame stacking 개수
N_AGENT = MAX_NEIGHBORS         # 최대 이웃 에이전트 수 (호환성)

# Total observation: 360 (self radar) + 6 (self state) + 4×371 (neighbors) + 5 (colregs) = 1855D
OBSERVATION_SIZE = STATE_SIZE + SELF_STATE_SIZE + (MAX_NEIGHBORS * NEIGHBOR_OBS_SIZE) + COLREGS_SIZE

# ============================================================================
# Training Mode
# ============================================================================
LOAD_MODEL = False              # 저장된 모델 로드 여부
TRAIN_MODE = True               # 학습 모드 (False: 평가 모드)
MODEL_PATH = None               # 로드할 모델 경로 (예: "./models/VesselNavigation_20251117_123456/policy_episode_1000.pth")

# ============================================================================
# PPO Hyperparameters
# ============================================================================
DISCOUNT_FACTOR = 0.95          # Gamma: 할인율
GAE_LAMBDA = 0.95               # GAE lambda: advantage estimation 파라미터
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
WORKER_ID = 0                   # Unity 환경 worker ID (Editor는 0 필수)
BASE_PORT = 5004                # Unity 통신 포트 (Unity 기본값)
TIME_SCALE = 3.0                # 시뮬레이션 속도 (1.0 = 실시간, 10.0 = 10배속)

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
