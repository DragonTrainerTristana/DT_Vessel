"""
Utility Functions for PPO Training
"""
import numpy as np


def calculate_returns(rewards, dones, last_value, values, gamma=0.99, gae_lambda=0.95,
                      truncateds=None):
    """
    GAE (Generalized Advantage Estimation)를 사용한 할인된 리턴 계산

    Args:
        rewards: 각 스텝의 보상 [T]
        dones: 각 스텝의 *진짜* 종료 여부 [T] (goal/collision) → bootstrap 0
        last_value: 마지막 상태의 가치 (스칼라)
        values: 각 상태의 가치 추정값 [T]
        gamma: 할인율 (기본값: 0.99)
        gae_lambda: GAE 람다 파라미터 (기본값: 0.95)
        truncateds: 각 스텝의 timeout 절단 여부 [T] (None이면 전부 False = 기존 동작 비트동일).
            True면 절단 → V를 bootstrap(미래가치 0으로 안 만듦)하되 GAE 역전파는 경계에서 절단.
            ★ truncated는 흐름의 강제중단(관측중단)이라 미래가치가 존재 → done(진짜종료)과 반드시 구분.

    Returns:
        returns: 계산된 할인 리턴값 [T]
    """
    returns = np.zeros_like(rewards)
    gae = 0
    next_value = last_value
    if truncateds is None:
        truncateds = np.zeros_like(dones)

    for t in reversed(range(len(rewards))):
        if dones[t]:
            # 진짜 종료(goal/collision): 미래가치 없음 → bootstrap 0, GAE 역전파 절단.
            boot = 0.0
            cont = 0.0
        elif truncateds[t]:
            # timeout 절단: 흐름은 계속됐을 것 → 자기 value로 bootstrap(V(s_next) 근사, 버퍼끝 처리와 동일 규약).
            #   에피소드 경계라 GAE 역전파는 절단(다음 step은 respawn한 새 에피소드).
            boot = values[t]
            cont = 0.0
        else:
            # 진행 중: 다음 상태 value로 bootstrap, GAE 역전파 지속.
            boot = next_value
            cont = 1.0

        # TD error: δ_t = r_t + γ*boot - V(s_t)
        delta = rewards[t] + gamma * boot - values[t]
        # GAE: A_t = δ_t + γ*λ*cont*A_{t+1}
        gae = delta + gamma * gae_lambda * cont * gae
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
