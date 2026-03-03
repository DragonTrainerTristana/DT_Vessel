"""
Utility Functions for PPO Training
"""
import numpy as np


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
        # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # GAE: A_t = δ_t + γ*λ*A_{t+1}
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae

        # Return: R_t = A_t + V(s_t)
        returns[t] = gae + values[t]

        next_value = values[t]

    return returns


class RunningMeanStd:
    """
    Welford's online algorithm을 사용한 이동 평균과 표준편차 계산
    상태 정규화에 사용됨
    """
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def normalize_state(state, running_mean_std):
    """RunningMeanStd를 사용한 상태 정규화"""
    return (state - running_mean_std.mean) / np.sqrt(running_mean_std.var + 1e-8)
