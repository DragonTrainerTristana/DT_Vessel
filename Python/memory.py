"""
Experience Replay Buffer for Multi-Agent PPO (GitHub 방식)
- neighbor_obs 제거 (메시지 교환은 Python 내부에서 처리)
"""
import numpy as np


class AgentMemory:
    """개별 에이전트의 경험을 저장하는 메모리"""

    def __init__(self):
        """메모리 초기화"""
        self.states = []          # [frames * STATE_SIZE]
        self.goals = []           # [2]
        self.self_states = []     # [4] speed, yaw_rate, heading, rudder
        self.colregs = []         # [5]
        self.others_msgs = []     # [msg_dim] - 이웃에게 받은 메시지 합
        self.actions = []         # [action_size]
        self.rewards = []         # scalar
        self.dones = []           # bool
        self.values = []          # scalar
        self.logprobs = []        # scalar
        self.episode_done = False

    def clear(self):
        """모든 저장된 경험 삭제"""
        self.states.clear()
        self.goals.clear()
        self.self_states.clear()
        self.colregs.clear()
        self.others_msgs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.logprobs.clear()
        self.episode_done = False

        import gc
        gc.collect()

    def add(self, state, goal, self_state, colregs, others_msg, action, reward, done, value, logprob):
        """새로운 경험 추가"""
        if not self.episode_done:
            self.states.append(state)
            self.goals.append(goal)
            self.self_states.append(self_state)
            self.colregs.append(colregs)
            self.others_msgs.append(others_msg)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.values.append(value)
            self.logprobs.append(logprob)

            if done:
                self.episode_done = True

    def mark_done(self, final_reward=0):
        """
        에피소드 종료 마킹 및 최종 보상 추가
        - 마지막 경험의 reward에 final_reward를 더함
        - 마지막 경험의 done을 True로 설정
        """
        if len(self.rewards) > 0:
            self.rewards[-1] += final_reward
            self.dones[-1] = True
            self.episode_done = True


class Memory:
    """여러 에이전트의 경험을 관리하는 중앙 메모리"""

    def __init__(self):
        """메모리 초기화"""
        self.agent_memories = {}  # {agent_id: AgentMemory}

    def clear(self):
        """모든 에이전트 메모리 삭제"""
        for agent_memory in self.agent_memories.values():
            agent_memory.clear()
        self.agent_memories.clear()

        import gc
        gc.collect()

    def add_agent_experience(self, agent_id, state, goal, self_state, colregs, others_msg,
                             action, reward, done, value, logprob):
        """특정 에이전트의 경험 추가"""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = AgentMemory()

        self.agent_memories[agent_id].add(
            state, goal, self_state, colregs, others_msg, action, reward, done, value, logprob
        )

    def get_active_agents(self):
        """활성 상태인 에이전트 ID 목록 반환"""
        return [agent_id for agent_id, memory in self.agent_memories.items()
                if not memory.episode_done]

    def get_all_experiences(self):
        """
        모든 에이전트의 경험을 하나의 배치로 통합
        ✅ GAE를 에이전트별로 계산한 후 통합

        Returns:
            dict: 통합된 경험 데이터 (returns 포함)
                - states: [N, frames * STATE_SIZE]
                - goals: [N, 2]
                - speeds: [N, 2]
                - colregs: [N, 5]
                - actions: [N, action_size]
                - rewards: [N]
                - dones: [N]
                - values: [N]
                - logprobs: [N]
                - returns: [N] - 에이전트별로 계산된 GAE returns
        """
        from functions import calculate_returns
        from config import DISCOUNT_FACTOR, GAE_LAMBDA

        states, goals, self_states, colregs, others_msgs = [], [], [], [], []
        actions, rewards, dones, values, logprobs = [], [], [], [], []
        returns = []

        for agent_memory in self.agent_memories.values():
            if len(agent_memory.states) == 0:
                continue

            # 에이전트별로 GAE 계산
            agent_rewards = np.array(agent_memory.rewards)
            agent_dones = np.array(agent_memory.dones)
            agent_values = np.array(agent_memory.values)

            # 에피소드가 끝났으면 last_value=0, 아니면 마지막 value 사용
            last_value = 0 if agent_memory.episode_done else agent_values[-1]

            agent_returns = calculate_returns(
                agent_rewards, agent_dones, last_value, agent_values,
                DISCOUNT_FACTOR, GAE_LAMBDA
            )

            # 데이터 통합
            states.extend(agent_memory.states)
            goals.extend(agent_memory.goals)
            self_states.extend(agent_memory.self_states)
            colregs.extend(agent_memory.colregs)
            others_msgs.extend(agent_memory.others_msgs)
            actions.extend(agent_memory.actions)
            rewards.extend(agent_memory.rewards)
            dones.extend(agent_memory.dones)
            values.extend(agent_memory.values)
            logprobs.extend(agent_memory.logprobs)
            returns.extend(agent_returns)

        return {
            'states': np.array(states) if states else np.array([]),
            'goals': np.array(goals) if goals else np.array([]),
            'self_states': np.array(self_states) if self_states else np.array([]),
            'colregs': np.array(colregs) if colregs else np.array([]),
            'others_msgs': np.array(others_msgs) if others_msgs else np.array([]),
            'actions': np.array(actions) if actions else np.array([]),
            'rewards': np.array(rewards) if rewards else np.array([]),
            'dones': np.array(dones) if dones else np.array([]),
            'values': np.array(values) if values else np.array([]),
            'logprobs': np.array(logprobs) if logprobs else np.array([]),
            'returns': np.array(returns) if returns else np.array([])
        }

    def reset_for_new_episode(self):
        """새 에피소드 시작 전 종료된 에이전트만 메모리에서 제거"""
        self.agent_memories = {agent_id: memory
                              for agent_id, memory in self.agent_memories.items()
                              if not memory.episode_done}
