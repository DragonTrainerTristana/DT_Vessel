"""
Utility Functions for PPO Training
"""
import numpy as np
import torch

def log_normal_density(x, mean, log_std, std):
    """
    가우시안 확률 밀도의 로그값 계산

    Args:
        x: 실제 행동 값 [batch, action_dim]
        mean: 평균값 [batch, action_dim]
        log_std: 표준편차의 로그값 [batch, action_dim]
        std: 표준편차 [batch, action_dim]

    Returns:
        log_density: 확률 밀도의 로그값 [batch, 1]
    """
    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 * np.log(2 * np.pi) - log_std
    log_density = log_density.sum(dim=-1, keepdim=True)
    return log_density


def calculate_returns(rewards, dones, last_value, values, gamma=0.99, gae_lambda=0.95):
    """
    GAE (Generalized Advantage Estimation)를 사용한 할인된 리턴 계산

    Args:
        rewards: 각 스텝의 보상 [T]
        dones: 각 스텝의 종료 여부 [T]
        last_value: 마지막 상태의 가치 (스칼라)
        values: 각 상태의 가치 추정값 [T]
        gamma: 할인율 (기본값: 0.99)
        gae_lambda: GAE 람다 파라미터 (기본값: 0.95)

    Returns:
        returns: 계산된 할인 리턴값 [T]
    """
    returns = np.zeros_like(rewards)
    gae = 0
    next_value = last_value

    for t in reversed(range(len(rewards))):
        # TD error 계산: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # GAE 계산: A_t = δ_t + γ*λ*A_{t+1}
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae

        # Return 계산: R_t = A_t + V(s_t)
        returns[t] = gae + values[t]

        next_value = values[t]

    return returns


class RunningMeanStd:
    """
    Welford's online algorithm을 사용한 이동 평균과 표준편차 계산
    상태 정규화에 사용됨
    """
    def __init__(self, epsilon=1e-4, shape=()):
        """
        Args:
            epsilon: 0으로 나누는 것을 방지하기 위한 작은 값
            shape: 데이터의 shape (기본값: 스칼라)
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        """
        새로운 배치 데이터로 평균과 분산 업데이트

        Args:
            x: 새로운 데이터 배치 [batch, ...]
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        Welford's algorithm을 사용한 모멘트 업데이트

        Args:
            batch_mean: 배치 평균
            batch_var: 배치 분산
            batch_count: 배치 크기
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # 새로운 평균 계산
        new_mean = self.mean + delta * batch_count / tot_count

        # 새로운 분산 계산
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        # 업데이트
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def normalize_state(state, running_mean_std):
    """
    RunningMeanStd를 사용한 상태 정규화

    Args:
        state: 정규화할 상태 [...]
        running_mean_std: RunningMeanStd 인스턴스

    Returns:
        normalized_state: 정규화된 상태 (평균=0, 표준편차=1)
    """
    return (state - running_mean_std.mean) / np.sqrt(running_mean_std.var + 1e-8)
