import torch
import os
import datetime

# 환경 및 학습 하이퍼파라미터

# 레이더 갯수를 좀 더 늘려야 할 것 같음 
# 현재는 1'간격으로 (나중에도 마찬가지) 45도 방향으로 압축해서 학습하지만, min, avg, max로만 해서 정확도가 낮을 수 있음

STATE_SIZE = 36  # 기본 관측 차원 (레이더 24 + 상태 4 + 목표 3 + COLREGs 4 + 위험도 1)

MSG_ACTION_SPACE = 6  # latent msg 3+3 concat한 것 
CONTINUOUS_ACTION_SIZE = 2 # Rudder, Thrust -1 ~ 1값

# Stack은 3으로 바꿔야함 (Dynamic Model의 특성상) 
FRAMES = 3  # 프레임 스택 크기 (Stack 3)
N_AGENT = 4  # 최대 통신 가능 선박 수

# 모델 로드 및 학습 모드 설정 (중요하지는 않음, Editor 상에서 돌릴꺼임)
LOAD_MODEL = False
TRAIN_MODE = True

# PPO 하이퍼파라미터 (8월 12일 수정)
DISCOUNT_FACTOR = 0.95   # 감가율 (0.99 → 0.95로 조정함)
LEARNING_RATE = 1e-4    # 학습률 (3e-4 → 1e-4로 감소)
N_STEP = 2048           # 업데이트 간격 (1000 → 2048로 증가)
BATCH_SIZE = 2048       # 배치 크기 (유지)

# EPOCH가 4면 너무 학습속도가 느릴 수 있나? 너무 느리면 추후에 3으로 줄이자
N_EPOCH = 4             # 에포크 수 (3 → 4로 증가)
EPSILON = 0.2           # PPO 클리핑 파라미터 (유지)
ENTROPY_BONUS = 0.005   # 엔트로피 보너스 (0.01 → 0.005로 감소)
CRITIC_LOSS_WEIGHT = 0.5  # 크리틱 손실 가중치 (유지)

# 학습 설정
GRAD_CLIP_MAX_NORM = 0.5  # 그래디언트 클리핑
RUN_STEP = 30000000 if TRAIN_MODE else 0  # 총 실행 스텝
TEST_STEP = 100000     # 테스트 스텝
PRINT_INTERVAL = 10000  # 출력 간격
SAVE_INTERVAL = 50000  # 모델 저장 간격

# 추가 학습 파라미터
MAX_STEPS = 2000       # 에피소드당 최대 스텝 수 1000 -> 2000으로 증가시킴

NUM_EPISODES = RUN_STEP // MAX_STEPS  # 총 에피소드 수 10000000 // 2000 = 5000개
UPDATE_INTERVAL = N_STEP  # 업데이트 간격
VALUE_LOSS_COEF = CRITIC_LOSS_WEIGHT  # 가치 손실 계수
ENTROPY_COEF = ENTROPY_BONUS  # 엔트로피 계수
MAX_GRAD_NORM = GRAD_CLIP_MAX_NORM  # 최대 그래디언트 노름

# 환경 설정
ENV_NAME = "AirCombatRL" # 이제 이건 필요 없음
DATE_TIME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
SAVE_PATH = os.path.join(".", "saved_models", ENV_NAME, "PPO", DATE_TIME)
# 언젠가 GPU로 바꿀 수 있도록... (CUDA가 진짜 설정 너무 어려움)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unity Editor 환경 설정 
WORKER_ID = 0
BASE_PORT = 5005
TIME_SCALE = 3.0

# Unity Build exe 환경 설정 (안씀)
# WORKER_ID = 1
# BASE_PORT = 5006
# TIME_SCALE = 1.0

# 로깅 설정
LOG_DIR = os.path.join(SAVE_PATH, 'logs')

# 모델 저장 경로가 없으면 생성
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    os.makedirs(LOG_DIR)

def get_config_dict():
    """설정값들을 딕셔너리 형태로 반환"""
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
        'max_steps': MAX_STEPS,
        'device': str(DEVICE)
    }