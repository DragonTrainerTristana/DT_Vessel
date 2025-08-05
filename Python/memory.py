import numpy as np

class AgentMemory:
    """개별 에이전트의 경험을 저장하는 메모리 클래스"""
    def __init__(self):
        self.states = []
        self.goals = []
        self.speeds = []
        self.neighbor_infos = []  # 이웃 정보 저장
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logprobs = []
        self.episode_done = False  # 에피소드 종료 여부
    
    def clear(self):
        """메모리 초기화"""
        self.__init__()
    
    def add(self, state, goal, speed, neighbor_info, action, reward, done, value, logprob):
        """새로운 경험 추가"""
        if not self.episode_done:  # 에피소드가 종료되지 않은 경우에만 추가
            self.states.append(state)
            self.goals.append(goal)
            self.speeds.append(speed)
            self.neighbor_infos.append(neighbor_info)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.values.append(value)
            self.logprobs.append(logprob)
            
            if done:
                self.episode_done = True

class Memory:
    """여러 에이전트의 경험을 관리하는 메모리 클래스"""
    def __init__(self):
        self.agent_memories = {}  # 에이전트 ID를 키로 사용
    
    def clear(self):
        """모든 에이전트 메모리 초기화"""
        self.__init__()
    
    def add_agent_experience(self, agent_id, state, goal, speed, neighbor_info, 
                            action, reward, done, value, logprob):
        """에이전트의 경험 추가"""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = AgentMemory()
        
        self.agent_memories[agent_id].add(
            state, goal, speed, neighbor_info, action, reward, done, value, logprob)
    
    def get_active_agents(self):
        """활성 상태인 에이전트 ID 목록 반환"""
        return [agent_id for agent_id, memory in self.agent_memories.items() 
                if not memory.episode_done]
    
    def get_all_experiences(self):
        """학습을 위한 모든 경험 데이터 수집"""
        states, goals, speeds, neighbor_infos = [], [], [], []
        actions, rewards, dones, values, logprobs = [], [], [], [], []
        
        for agent_memory in self.agent_memories.values():
            states.extend(agent_memory.states)
            goals.extend(agent_memory.goals)
            speeds.extend(agent_memory.speeds)
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
            'neighbor_infos': np.array(neighbor_infos) if neighbor_infos else np.array([]),
            'actions': np.array(actions) if actions else np.array([]),
            'rewards': np.array(rewards) if rewards else np.array([]),
            'dones': np.array(dones) if dones else np.array([]),
            'values': np.array(values) if values else np.array([]),
            'logprobs': np.array(logprobs) if logprobs else np.array([])
        }

    def reset_for_new_episode(self):
        """새 에피소드를 위해 종료된 에이전트만 제거"""
        self.agent_memories = {agent_id: memory 
                              for agent_id, memory in self.agent_memories.items()
                              if not memory.episode_done}