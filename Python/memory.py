"""
Experience Replay Buffer for Multi-Agent PPO
"""
import numpy as np


class AgentMemory:
    """개별 에이전트의 경험을 저장하는 메모리 (COLREGs 정보 포함)"""

    def __init__(self):
        """메모리 초기화"""
        self.states = []
        self.goals = []
        self.speeds = []
        self.colregs_situations = []  # COLREGs 상황 정보 [4]
        self.neighbor_infos = []  # {' obs': [...], 'mask': [...]} 형태
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logprobs = []
        self.episode_done = False  # 에피소드 종료 플래그

    def clear(self):
        """모든 저장된 경험 삭제"""
        self.states.clear()
        self.goals.clear()
        self.speeds.clear()
        self.colregs_situations.clear()
        self.neighbor_infos.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.logprobs.clear()
        self.episode_done = False

        # 강제 가비지 컬렉션
        import gc
        gc.collect()

    def add(self, state, goal, speed, colregs_situations, neighbor_info, action, reward, done, value, logprob):
        """
        새로운 경험 추가 (COLREGs 정보 포함)

        Args:
            state: Frame-stacked state [STATE_SIZE * FRAMES]
            goal: Goal information [2]
            speed: Speed information [2]
            colregs_situations: COLREGs 상황 [4]
            neighbor_info: 이웃 정보 dict {'obs': [...], 'mask': [...]}
            action: 행동 [CONTINUOUS_ACTION_SIZE]
            reward: 보상 (scalar)
            done: 종료 여부 (bool)
            value: 가치 추정 (scalar)
            logprob: 행동 로그 확률 (scalar)
        """
        if not self.episode_done:  # 종료된 에피소드는 경험 추가 안 함
            self.states.append(state)
            self.goals.append(goal)
            self.speeds.append(speed)
            self.colregs_situations.append(colregs_situations)
            self.neighbor_infos.append(neighbor_info)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.values.append(value)
            self.logprobs.append(logprob)

            if done:
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

        # 강제 가비지 컬렉션
        import gc
        gc.collect()

    def add_agent_experience(self, agent_id, state, goal, speed, colregs_situations, neighbor_info,
                            action, reward, done, value, logprob):
        """
        특정 에이전트의 경험 추가 (COLREGs 정보 포함)

        Args:
            agent_id: 에이전트 ID
            state: Frame-stacked state [STATE_SIZE * FRAMES]
            goal: Goal information [2]
            speed: Speed information [2]
            colregs_situations: COLREGs 상황 [4]
            neighbor_info: 이웃 정보 dict {'obs': [...], 'mask': [...]}
            action: 행동 [CONTINUOUS_ACTION_SIZE]
            reward: 보상 (scalar)
            done: 종료 여부 (bool)
            value: 가치 추정 (scalar)
            logprob: 행동 로그 확률 (scalar)
        """
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = AgentMemory()

        self.agent_memories[agent_id].add(
            state, goal, speed, colregs_situations, neighbor_info, action, reward, done, value, logprob)

    def get_active_agents(self):
        """
        활성 상태인 에이전트 ID 목록 반환

        Returns:
            list: 아직 종료되지 않은 에이전트 ID 리스트
        """
        return [agent_id for agent_id, memory in self.agent_memories.items()
                if not memory.episode_done]

    def get_all_experiences(self):
        """
        모든 에이전트의 경험을 하나의 배치로 통합 (COLREGs 정보 포함)

        Returns:
            dict: 통합된 경험 데이터
                - states: [N, STATE_SIZE * FRAMES]
                - goals: [N, 2]
                - speeds: [N, 2]
                - colregs_situations: [N, 4]
                - neighbor_infos: [N, {'obs': [...], 'mask': [...]}]
                - actions: [N, CONTINUOUS_ACTION_SIZE]
                - rewards: [N]
                - dones: [N]
                - values: [N]
                - logprobs: [N]
        """
        states, goals, speeds, colregs_situations, neighbor_infos = [], [], [], [], []
        actions, rewards, dones, values, logprobs = [], [], [], [], []

        for agent_memory in self.agent_memories.values():
            states.extend(agent_memory.states)
            goals.extend(agent_memory.goals)
            speeds.extend(agent_memory.speeds)
            colregs_situations.extend(agent_memory.colregs_situations)
            neighbor_infos.extend(agent_memory.neighbor_infos)
            actions.extend(agent_memory.actions)
            rewards.extend(agent_memory.rewards)
            dones.extend(agent_memory.dones)
            values.extend(agent_memory.values)
            logprobs.extend(agent_memory.logprobs)

        return {
            'states': np.array(states) if states else np.array([]),
            'goals': np.array(goals) if goals else np.array([]),
            'speeds': np.array(speeds) if speeds else np.array([]),
            'colregs_situations': np.array(colregs_situations) if colregs_situations else np.array([]),
            'neighbor_infos': neighbor_infos,  # list of dicts
            'actions': np.array(actions) if actions else np.array([]),
            'rewards': np.array(rewards) if rewards else np.array([]),
            'dones': np.array(dones) if dones else np.array([]),
            'values': np.array(values) if values else np.array([]),
            'logprobs': np.array(logprobs) if logprobs else np.array([])
        }

    def reset_for_new_episode(self):
        """
        새 에피소드 시작 전 종료된 에이전트만 메모리에서 제거
        (아직 활성인 에이전트는 유지)
        """
        self.agent_memories = {agent_id: memory
                              for agent_id, memory in self.agent_memories.items()
                              if not memory.episode_done}
