import torch
import os
import datetime

# 환경 및 학습 하이퍼파라미터
STATE_SIZE = 46  # 24(섹터레이더) + 4(상태) + 3(목표) + 12(COLREGs×3) + 3(위험도×3)
MSG_ACTION_SPACE = 3  # MessageActor의 액션 공간 크기 
CONTINUOUS_ACTION_SIZE = 2  # ControlActor의 액션 공간 크기
FRAMES = 4  # 프레임 스택 크기
N_AGENT = 4  # 에이전트 수

# 모델 로드 및 학습 모드 설정
LOAD_MODEL = False
TRAIN_MODE = True

# PPO 하이퍼파라미터
DISCOUNT_FACTOR = 0.995  # 감가율
LEARNING_RATE = 3e-4    # 학습률
N_STEP = 2048          # 업데이트 간격 (배치 크기와 동일하게)
BATCH_SIZE = 2048      # 배치 크기
N_EPOCH = 3             # 에포크 수
EPSILON = 0.2           # PPO 클리핑 파라미터
ENTROPY_BONUS = 0.01    # 엔트로피 보너스
CRITIC_LOSS_WEIGHT = 0.5  # 크리틱 손실 가중치

# 학습 설정
GRAD_CLIP_MAX_NORM = 0.5  # 그래디언트 클리핑
RUN_STEP = 30000000 if TRAIN_MODE else 0  # 총 실행 스텝
TEST_STEP = 100000     # 테스트 스텝
PRINT_INTERVAL = 10    # 10 에피소드마다 출력
SAVE_INTERVAL = 100    # 에피소드 기준으로 저장 (100 에피소드마다)

# 추가 학습 파라미터
MAX_STEPS = 1000       # 에피소드당 최대 스텝 수
NUM_EPISODES = RUN_STEP // MAX_STEPS  # 총 에피소드 수
UPDATE_INTERVAL = N_STEP  # 업데이트 간격
VALUE_LOSS_COEF = CRITIC_LOSS_WEIGHT  # 가치 손실 계수
ENTROPY_COEF = ENTROPY_BONUS  # 엔트로피 계수
MAX_GRAD_NORM = GRAD_CLIP_MAX_NORM  # 최대 그래디언트 노름

# 환경 설정
ENV_NAME = "VesselProject"
DATE_TIME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
SAVE_PATH = os.path.join(".", "saved_models", ENV_NAME, "PPO", DATE_TIME)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unity ML-Agents 환경 설정
WORKER_ID = 1
BASE_PORT = 5006
TIME_SCALE = 10.0

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