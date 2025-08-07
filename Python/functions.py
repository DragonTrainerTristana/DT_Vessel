import numpy as np
import torch
import math

def log_normal_density(x, mean, log_std, std):
    """가우시안 확률 밀도의 로그값을 계산하는 함수 
    
    Args:
        x: 실제 행동 값
        mean: 평균값
        log_std: 표준편차의 로그값
        std: 표준편차
    
    Returns:
        log_density: 확률 밀도의 로그값
    """
    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 * np.log(2 * np.pi) - log_std
    log_density = log_density.sum(dim=-1, keepdim=True)
    return log_density

def calculate_returns(rewards, dones, last_value, values, gamma=0.99):
    """할인된 리턴값 계산
    
    Args:
        rewards: 각 스텝의 보상
        dones: 각 스텝의 종료 여부
        last_value: 마지막 상태의 가치
        values: 각 상태의 가치 추정값
        gamma: 할인율
    
    Returns:
        returns: 계산된 할인 리턴값
    """
    returns = np.zeros_like(rewards)
    gae = 0
    next_value = last_value
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * 0.95 * (1 - dones[t]) * gae
        returns[t] = gae + values[t]
        next_value = values[t]
        
    return returns

def transform_buffer(buff):
    """경험 버퍼를 학습에 사용할 형태로 변환
    
    Args:
        buff: 수집된 경험 데이터
        
    Returns:
        변환된 배치 데이터들 (상태, 목표, 속도, 행동, 보상, 종료 여부 등)
    """
    s_batch, goal_batch, speed_batch = [], [], []
    a_batch, r_batch, d_batch, l_batch, v_batch = [], [], [], [], []
    
    for e in buff:
        s_temp, goal_temp, speed_temp = [], [], []
        
        for state in e[0]:
            s_temp.append(state[0])
            goal_temp.append(state[1])
            speed_temp.append(state[2])
            
        s_batch.append(s_temp)
        goal_batch.append(goal_temp)
        speed_batch.append(speed_temp)
        
        a_batch.append(e[1])
        r_batch.append(e[2])
        d_batch.append(e[3])
        l_batch.append(e[4])
        v_batch.append(e[5])

    return map(np.asarray, [s_batch, goal_batch, speed_batch, 
                           a_batch, r_batch, d_batch, l_batch, v_batch])

class RunningMeanStd:
    """상태 정규화를 위한 이동 평균과 표준편차를 계산하는 클래스"""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        """새로운 데이터로 평균과 분산 업데이트"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """모멘트를 이용한 업데이트 수행"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

def normalize_state(state, running_mean_std):
    """상태 정규화
    
    Args:
        state: 정규화할 상태
        running_mean_std: RunningMeanStd 인스턴스
    
    Returns:
        normalized_state: 정규화된 상태
    """
    return (state - running_mean_std.mean) / np.sqrt(running_mean_std.var + 1e-8)