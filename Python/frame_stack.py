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
            state: 새로운 state [state_size] (parse_observation이 매번 새 array 반환하므로 copy 불필요)
        """
        self.buffer.append(state)

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
        # ★누수 방어(2026-06-18): ML-Agents agent_id는 EndEpisode(respawn)마다 새 값 →
        #   agent_buffers 키가 무한 누적(remove_agent는 terminal_steps에서만 호출, terminal 누락 시 영구 잔존).
        #   이번 update에서 '본' agent_id를 기록 → retire_unseen()이 한 동안 안 보인 stale 키를 GC.
        #   respawn 빈번한 commOFF(충돌 7.9%)가 ~23k ep서 크래시한 근본(buffer dict가 키당 deque×369D×3 누적).
        self._seen_this_pass = set()

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

        self._seen_this_pass.add(agent_id)
        self.agent_buffers[agent_id].update(state)
        return self.agent_buffers[agent_id].get_stacked()

    def retire_unseen(self):
        """
        이번 collect pass에서 한 번도 update되지 않은(=더 이상 활성 아닌) agent_id 버퍼 제거.
        terminal_steps 정리(remove_agent)가 어떤 이유로든 누락돼도 stale 키가 무한 쌓이지 않게 하는 안전망.
        매 training_step의 collect 직후 호출. 활성 배(매 step update됨)는 절대 제거 안 됨(안전).
        ★respawn 횟수 비례 누수 차단 → commOFF 장기(2M) 완주 가능.
        """
        active = self._seen_this_pass
        stale = [aid for aid in self.agent_buffers if aid not in active]
        for aid in stale:
            del self.agent_buffers[aid]
        self._seen_this_pass = set()
        return len(stale)

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
