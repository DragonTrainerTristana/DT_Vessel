"""
Observation 파싱 및 통신 유틸리티 함수
main.py와 test.py에서 공통으로 사용 — 중복 제거
"""
import numpy as np
from config import STATE_SIZE, COMM_RANGE, MAX_COMM_PARTNERS


def parse_observation(obs_raw):
    """
    369D observation 파싱 (STATE_SIZE=360 raw ray 기준, 인덱스는 STATE_SIZE로 자동 산출):
    [0:360]    Radar (360 raw ray min-distance, 1°) ← frame-stack 대상. Python RadarEncoder가 Conv1D로 압축.
    [360:362]  Goal (distance, angle)
    [362:366]  Self state (speed, yaw_rate, heading, rudder)
    [366:368]  Position (x, z) - 통신 범위 계산용, 학습 제외
    [368]      Situation (COLREGs 상황 0~4) - MoE 라우터 전용, 학습 feature 제외
    ★ARPA 제거(2026-06-05): 충돌기하는 360 raw ray + frame-stack(RadarEncoder)이 학습.
    ★구버전 빌드 호환: situation 슬롯 없으면 0(None)으로 폴백 → USE_MOE=0/단일 head로 동작.
    """
    idx = STATE_SIZE  # 360
    state = obs_raw[:idx]

    goal = np.array(obs_raw[idx:idx + 2], dtype=np.float32)            # 2D
    self_state = np.array(obs_raw[idx + 2:idx + 6], dtype=np.float32)  # 4D (speed, yaw, heading, rudder)
    _p = idx + 6
    position = obs_raw[_p:_p + 2]                                      # x, z
    # situation: MoE 라우팅 인덱스. 구버전 빌드면 슬롯 부재 → 0(None) 폴백(역호환).
    situation = int(round(float(obs_raw[_p + 2]))) if len(obs_raw) > _p + 2 else 0

    # 네트워크 입력 (position·situation 제외) = STATE_SIZE(360) + 2 + 4 = 366D (radar는 RadarEncoder가 Conv1D 압축)
    obs_full = np.concatenate([state, goal, self_state])

    return state, goal, self_state, obs_full, position, situation


def get_comm_partners(my_id, my_pos, all_positions):
    """통신 범위 내 가장 가까운 파트너들 반환"""
    distances = []
    for other_id, other_pos in all_positions.items():
        if other_id == my_id:
            continue
        dist = np.sqrt((my_pos[0] - other_pos[0])**2 + (my_pos[1] - other_pos[1])**2)
        if dist <= COMM_RANGE:
            distances.append((other_id, dist))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:MAX_COMM_PARTNERS]]
