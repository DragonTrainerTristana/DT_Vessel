"""
Observation 파싱 및 통신 유틸리티 함수
main.py와 test.py에서 공통으로 사용 — 중복 제거
"""
import numpy as np
from config import STATE_SIZE, COMM_RANGE, MAX_COMM_PARTNERS


def parse_observation(obs_raw):
    """
    373D observation 파싱 (STATE_SIZE=360 기준):
    [0:360]      Radar (360 rays, 1도 간격)
    [360:362]    Goal (distance, angle)
    [362:366]    Self state (speed, yaw_rate, heading, rudder)
    [366:371]    COLREGs (5D one-hot)
    [371:373]    Position (x, z) - 통신용, 학습 제외
    """
    idx = STATE_SIZE  # 360
    state = obs_raw[:idx]

    goal_distance = obs_raw[idx]
    goal_angle = obs_raw[idx + 1]
    speed = obs_raw[idx + 2]
    yaw_rate = obs_raw[idx + 3]
    heading = obs_raw[idx + 4]
    rudder = obs_raw[idx + 5]

    goal = np.array([goal_distance, goal_angle], dtype=np.float32)
    self_state = np.array([speed, yaw_rate, heading, rudder], dtype=np.float32)  # 4D
    colregs = obs_raw[idx + 6:idx + 11]
    position = obs_raw[idx + 11:idx + 13]  # x, z (통신 범위 계산용)

    # 전체 observation (position 제외) = 371D
    obs_full = np.concatenate([state, [goal_distance, goal_angle, speed, yaw_rate,
                                        heading, rudder], colregs])

    return state, goal, self_state, colregs, obs_full, position


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
