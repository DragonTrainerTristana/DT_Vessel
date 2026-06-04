"""
Experience Replay Buffer for Multi-Agent PPO (GitHub 방식)
- 메시지 교환은 Python 내부에서 처리
- ★ 통신 gradient 수정: 각 경험에 '통신 파트너들의 obs'를 함께 저장 →
  PPO update 때 MessageActor를 파트너 obs로 재실행하여 sender→receiver gradient를 흐르게 함.
- ★ 경험 누락 버그 수정: episode_done 게이트 제거. 한 rollout window 안에서 충돌→리스폰한
  배의 다음 에피소드 경험도 버리지 않고 누적(GAE는 dones[]로 경계 구분).
"""
import numpy as np


class AgentMemory:
    """개별 에이전트의 경험을 저장하는 메모리"""

    def __init__(self):
        self.states = []          # [frames * STATE_SIZE]
        self.goals = []           # [2]
        self.self_states = []     # [4]
        self.arpas = []           # [21] - label-blind 충돌기하
        self.actions = []         # [action_size]
        self.rewards = []         # scalar
        self.dones = []           # bool
        self.values = []          # scalar
        self.logprobs = []        # scalar
        # 통신 파트너 obs (sender→receiver gradient 재구성용). 각 [K, dim], K=MAX_COMM_PARTNERS
        self.partner_states = []
        self.partner_goals = []
        self.partner_selfs = []
        self.partner_arpas = []
        self.partner_masks = []   # [K] (1=유효 파트너, 0=padding)
        self.partner_relpos = []  # [K,3] 상대방위(sin,cos)+거리 (위치 grounding)
        self.positions = []       # [2] 자기 위치(x,z) — intent 미래라벨 계산용(get_all_experiences)
        self.situations = []      # scalar COLREGs 상황(0~4) — MoE head 라우팅(rollout==update 동일 라우팅용)

    def clear(self):
        for lst in (self.states, self.goals, self.self_states, self.arpas,
                    self.actions, self.rewards, self.dones, self.values, self.logprobs,
                    self.partner_states, self.partner_goals, self.partner_selfs,
                    self.partner_arpas, self.partner_masks, self.partner_relpos, self.positions,
                    self.situations):
            lst.clear()

    def add(self, state, goal, self_state, arpa, action, reward, done, value, logprob,
            partner_states, partner_goals, partner_selfs, partner_arpas, partner_mask, partner_relpos,
            position, situation):
        """새로운 경험 추가 (episode_done 게이트 제거 - 모든 경험 누적, 경계는 dones[]가 표시)"""
        self.states.append(state)
        self.goals.append(goal)
        self.self_states.append(self_state)
        self.arpas.append(arpa)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.logprobs.append(logprob)
        self.partner_states.append(partner_states)
        self.partner_goals.append(partner_goals)
        self.partner_selfs.append(partner_selfs)
        self.partner_arpas.append(partner_arpas)
        self.partner_masks.append(partner_mask)
        self.partner_relpos.append(partner_relpos)
        self.positions.append(position)
        self.situations.append(situation)

    def mark_done(self, final_reward=0):
        """
        에피소드 종료 마킹 및 최종 보상 추가 (마지막 경험에).
        ⚠️ 이후 add()를 막지 않음 → 같은 window 내 새 에피소드 경험도 계속 누적됨.
        """
        if len(self.rewards) > 0:
            self.rewards[-1] += final_reward
            self.dones[-1] = True


class Memory:
    """여러 에이전트의 경험을 관리하는 중앙 메모리"""

    def __init__(self):
        self.agent_memories = {}  # {agent_id: AgentMemory}

    def clear(self):
        for agent_memory in self.agent_memories.values():
            agent_memory.clear()
        self.agent_memories.clear()

    def add_agent_experience(self, agent_id, state, goal, self_state, arpa,
                             action, reward, done, value, logprob,
                             partner_states, partner_goals, partner_selfs, partner_arpas, partner_mask,
                             partner_relpos, position, situation):
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = AgentMemory()
        self.agent_memories[agent_id].add(
            state, goal, self_state, arpa, action, reward, done, value, logprob,
            partner_states, partner_goals, partner_selfs, partner_arpas, partner_mask, partner_relpos,
            position, situation
        )

    def get_all_experiences(self):
        """
        모든 에이전트의 경험을 하나의 배치로 통합 (GAE는 에이전트별 계산).

        Returns dict: states/goals/self_states/arpas/actions/rewards/dones/values/logprobs/returns
                      + partner_states/partner_goals/partner_selfs/partner_arpas/partner_masks
        """
        from functions import calculate_returns
        from config import (DISCOUNT_FACTOR, GAE_LAMBDA,
                            INTENT_COEF, INTENT_K, INTENT_HORIZON, INTENT_POS_SCALE)
        import math

        states, goals, self_states, arpas = [], [], [], []
        actions, rewards, dones, values, logprobs = [], [], [], [], []
        p_states, p_goals, p_selfs, p_arpas, p_masks, p_relpos = [], [], [], [], [], []
        own_future, own_future_mask = [], []
        sits = []
        returns = []

        for agent_memory in self.agent_memories.values():
            if len(agent_memory.states) == 0:
                continue

            agent_rewards = np.array(agent_memory.rewards)
            agent_dones = np.array(agent_memory.dones)
            agent_values = np.array(agent_memory.values)

            # 버퍼 마지막 step이 terminal이면 bootstrap=0, 아니면 마지막 value로 truncated bootstrap
            last_value = 0 if (len(agent_dones) > 0 and agent_dones[-1]) else agent_values[-1]

            agent_returns = calculate_returns(
                agent_rewards, agent_dones, last_value, agent_values,
                DISCOUNT_FACTOR, GAE_LAMBDA
            )

            # ★ intent 미래라벨(self-supervised): 각 step t의 미래 K지점 변위를 t시점 body frame으로 회전.
            #   done 경계를 넘으면 mask=0(에피소드 누설 방지). INTENT_COEF=0이면 zeros(=OFF 비트동일).
            L = len(agent_memory.states)
            fut = np.zeros((L, INTENT_K * 3), dtype=np.float32)
            fut_mask = np.zeros((L, INTENT_K * 3), dtype=np.float32)
            if INTENT_COEF > 0.0 and L > 0:
                pos = np.asarray(agent_memory.positions, dtype=np.float32).reshape(L, -1)[:, :2]  # [L,2]
                head_deg = np.asarray([s[2] for s in agent_memory.self_states], dtype=np.float32) * 180.0
                dn = agent_dones.astype(bool)
                for t in range(L):
                    hr = math.radians(float(head_deg[t])); ch, sh = math.cos(hr), math.sin(hr)
                    for k in range(INTENT_K):
                        j = t + (k + 1) * INTENT_HORIZON
                        if j >= L:
                            break  # 더 긴 horizon도 무효
                        if dn[t:j].any():
                            continue  # 에피소드 경계 넘음 → 무효(respawn 텔레포트 라벨 방지)
                        dx = float(pos[j, 0] - pos[t, 0]); dz = float(pos[j, 1] - pos[t, 1])
                        loc_star = dx * ch - dz * sh   # 국소 우현(starboard) 성분
                        loc_fwd = dx * sh + dz * ch    # 국소 전방(forward) 성분
                        dh = ((float(head_deg[j]) - float(head_deg[t]) + 180.0) % 360.0) - 180.0  # wrap[-180,180]
                        b = k * 3
                        fut[t, b + 0] = loc_star / INTENT_POS_SCALE
                        fut[t, b + 1] = loc_fwd / INTENT_POS_SCALE
                        fut[t, b + 2] = dh / 180.0
                        fut_mask[t, b:b + 3] = 1.0

            states.extend(agent_memory.states)
            goals.extend(agent_memory.goals)
            self_states.extend(agent_memory.self_states)
            arpas.extend(agent_memory.arpas)
            actions.extend(agent_memory.actions)
            rewards.extend(agent_memory.rewards)
            dones.extend(agent_memory.dones)
            values.extend(agent_memory.values)
            logprobs.extend(agent_memory.logprobs)
            p_states.extend(agent_memory.partner_states)
            p_goals.extend(agent_memory.partner_goals)
            p_selfs.extend(agent_memory.partner_selfs)
            p_arpas.extend(agent_memory.partner_arpas)
            p_masks.extend(agent_memory.partner_masks)
            p_relpos.extend(agent_memory.partner_relpos)
            own_future.extend(fut)
            own_future_mask.extend(fut_mask)
            sits.extend(agent_memory.situations)
            returns.extend(agent_returns)

        def _arr(lst):
            return np.array(lst) if lst else np.array([])

        return {
            'states': _arr(states),
            'goals': _arr(goals),
            'self_states': _arr(self_states),
            'arpas': _arr(arpas),
            'actions': _arr(actions),
            'rewards': _arr(rewards),
            'dones': _arr(dones),
            'values': _arr(values),
            'logprobs': _arr(logprobs),
            'returns': _arr(returns),
            'partner_states': _arr(p_states),
            'partner_goals': _arr(p_goals),
            'partner_selfs': _arr(p_selfs),
            'partner_arpas': _arr(p_arpas),
            'partner_masks': _arr(p_masks),
            'partner_relpos': _arr(p_relpos),
            'own_future': _arr(own_future),
            'own_future_mask': _arr(own_future_mask),
            'situations': np.array(sits, dtype=np.int64) if sits else np.array([], dtype=np.int64),
        }
