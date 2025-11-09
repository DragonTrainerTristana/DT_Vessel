"""
Frame Stacking for Temporal Information
프레임 스택을 사용하여 시간적 정보를 포착
"""
import numpy as np
from collections import deque


class FrameStackBuffer:
    """단일 에이전트의 프레임 스택 버퍼"""

    def __init__(self, n_frames, state_size):
        """
        Args:
            n_frames: 스택할 프레임 개수
            state_size: 단일 프레임 state 크기
        """
        self.n_frames = n_frames
        self.state_size = state_size
        self.buffer = deque(maxlen=n_frames)
        self.reset()

    def reset(self):
        """버퍼 초기화 (0으로 채움)"""
        self.buffer.clear()
        for _ in range(self.n_frames):
            self.buffer.append(np.zeros(self.state_size, dtype=np.float32))

    def update(self, state):
        """
        새 프레임으로 업데이트

        Args:
            state: 새로운 state [state_size]
        """
        self.buffer.append(state.copy())

    def get_stacked(self):
        """
        스택된 프레임 반환

        Returns:
            stacked_frames: [state_size * n_frames] 1D 배열
        """
        return np.concatenate(list(self.buffer))


class MultiAgentFrameStack:
    """여러 에이전트의 프레임 스택 관리"""

    def __init__(self, n_frames, state_size):
        """
        Args:
            n_frames: 스택할 프레임 개수
            state_size: 단일 프레임 state 크기
        """
        self.n_frames = n_frames
        self.state_size = state_size
        self.agent_buffers = {}  # {agent_id: FrameStackBuffer}

    def update(self, agent_id, state):
        """
        특정 에이전트의 프레임 업데이트

        Args:
            agent_id: 에이전트 ID
            state: 새로운 state [state_size]

        Returns:
            stacked_frames: [state_size * n_frames] 1D 배열
        """
        if agent_id not in self.agent_buffers:
            self.agent_buffers[agent_id] = FrameStackBuffer(self.n_frames, self.state_size)

        self.agent_buffers[agent_id].update(state)
        return self.agent_buffers[agent_id].get_stacked()

    def reset_agent(self, agent_id):
        """특정 에이전트 버퍼 리셋"""
        if agent_id in self.agent_buffers:
            self.agent_buffers[agent_id].reset()

    def remove_agent(self, agent_id):
        """에이전트 제거"""
        if agent_id in self.agent_buffers:
            del self.agent_buffers[agent_id]

    def clear_all(self):
        """모든 에이전트 버퍼 제거"""
        self.agent_buffers.clear()
