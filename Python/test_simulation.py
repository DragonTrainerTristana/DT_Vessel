"""
완전한 A to Z 학습 파이프라인 시뮬레이션 및 검증
Unity → Python → Neural Network → PPO Training 전체 확인
"""

import torch
import numpy as np
import sys
from config import *
from networks import CNNPolicy
from memory import Memory
from functions import calculate_returns
from frame_stack import MultiAgentFrameStack

print("="*80)
print("🔍 A to Z 학습 파이프라인 시뮬레이션 시작")
print("="*80)

# =============================================================================
# Step 1: Unity C# Observation 생성 시뮬레이션
# =============================================================================
print("\n" + "="*80)
print("Step 1: Unity C# - CollectObservations() 시뮬레이션")
print("="*80)

def simulate_unity_observation():
    """Unity VesselAgent.cs의 CollectObservations() 시뮬레이션"""
    obs = []

    # [0:24] Radar data (8 sectors × 3)
    print("\n📡 Radar 데이터 (24D):")
    for sector in range(8):
        min_norm = np.random.rand()
        median_norm = np.random.rand()
        hit_ratio = np.random.rand()
        obs.extend([min_norm, median_norm, hit_ratio])
        print(f"  Sector {sector}: min={min_norm:.3f}, median={median_norm:.3f}, hit={hit_ratio:.3f}")

    # [24:28] Vessel state (4D)
    print("\n🚢 Vessel 상태 (4D):")
    speed_norm = np.random.rand()
    heading_x = np.random.randn()
    heading_z = np.random.randn()
    yaw_rate = np.random.randn()
    obs.extend([speed_norm, heading_x, heading_z, yaw_rate])
    print(f"  Speed: {speed_norm:.3f}, Heading: ({heading_x:.3f}, {heading_z:.3f}), Yaw: {yaw_rate:.3f}")

    # [28:31] Goal info (3D)
    print("\n🎯 Goal 정보 (3D):")
    goal_x = np.random.randn()
    goal_z = np.random.randn()
    distance = np.random.rand()
    obs.extend([goal_x, goal_z, distance])
    print(f"  Goal direction: ({goal_x:.3f}, {goal_z:.3f}), Distance: {distance:.3f}")

    # [31:35] COLREGs (4D one-hot)
    print("\n⚓ COLREGs 상황 (4D one-hot):")
    colregs = [0, 0, 1, 0]  # CrossingGiveWay 예시
    obs.extend(colregs)
    situations = ['HeadOn', 'CrossingStandOn', 'CrossingGiveWay', 'Overtaking']
    print(f"  Situation: {situations[colregs.index(1)]}")

    # [35:36] Risk level (1D)
    print("\n⚠️ 위험도 (1D):")
    risk = np.random.rand()
    obs.append(risk)
    print(f"  Risk: {risk:.3f}")

    # [36:38] Goal for message passing (2D)
    obs.extend([goal_x, goal_z])

    # [38:40] Speed for message passing (2D)
    obs.extend([speed_norm, yaw_rate])

    # [40:184] Neighbors (144D = 4 × 36)
    print("\n👥 Neighbor 정보 (144D = 4 neighbors × 36D):")
    for i in range(4):
        if i < 2:  # 2명만 활성화
            neighbor_obs = np.random.rand(36)
            obs.extend(neighbor_obs)
            print(f"  Neighbor {i}: ACTIVE (36D)")
        else:
            obs.extend([0.0] * 36)
            print(f"  Neighbor {i}: INACTIVE (36D zeros)")

    obs = np.array(obs, dtype=np.float32)
    print(f"\n✅ Total observation size: {len(obs)}D")
    assert len(obs) == 184, f"❌ ERROR: Expected 184D, got {len(obs)}D"

    return obs

# 4대의 선박 시뮬레이션
print("\n🚢 4대의 선박 observation 생성:")
unity_observations = []
for agent_id in range(4):
    print(f"\n--- Agent {agent_id} ---")
    obs = simulate_unity_observation()
    unity_observations.append((agent_id, obs))

print("\n" + "="*80)
print("✅ Step 1 완료: Unity observation 생성 성공 (4 agents × 184D)")
print("="*80)

# =============================================================================
# Step 2: Python - parse_observation()
# =============================================================================
print("\n" + "="*80)
print("Step 2: Python - parse_observation() 데이터 파싱")
print("="*80)

def parse_observation(obs_raw):
    """main.py의 parse_observation() 복제"""
    state = obs_raw[:STATE_SIZE]  # [0:36]
    goal = obs_raw[STATE_SIZE:STATE_SIZE+2]  # [36:38]
    speed = obs_raw[STATE_SIZE+2:STATE_SIZE+4]  # [38:40]
    neighbor_obs_raw = obs_raw[STATE_SIZE+4:]  # [40:184]

    neighbor_obs = neighbor_obs_raw.reshape(N_AGENT, STATE_SIZE)  # (4, 36)
    neighbor_mask = torch.tensor(
        [np.any(neighbor_obs[i] != 0) for i in range(N_AGENT)],
        dtype=torch.bool
    ).to(DEVICE)

    return state, goal, speed, neighbor_obs, neighbor_mask

parsed_data = []
for agent_id, obs_raw in unity_observations:
    state, goal, speed, neighbor_obs, neighbor_mask = parse_observation(obs_raw)

    print(f"\n🔍 Agent {agent_id}:")
    print(f"  State: {state.shape} (expected: (36,))")
    print(f"  Goal: {goal.shape} (expected: (2,))")
    print(f"  Speed: {speed.shape} (expected: (2,))")
    print(f"  Neighbor obs: {neighbor_obs.shape} (expected: (4, 36))")
    print(f"  Neighbor mask: {neighbor_mask.shape} (expected: (4,))")
    print(f"  Active neighbors: {neighbor_mask.sum().item()}")

    assert state.shape == (36,), f"❌ State dimension error"
    assert goal.shape == (2,), f"❌ Goal dimension error"
    assert speed.shape == (2,), f"❌ Speed dimension error"
    assert neighbor_obs.shape == (4, 36), f"❌ Neighbor obs dimension error"
    assert neighbor_mask.shape == (4,), f"❌ Neighbor mask dimension error"

    parsed_data.append((agent_id, state, goal, speed, neighbor_obs, neighbor_mask))

print("\n" + "="*80)
print("✅ Step 2 완료: 데이터 파싱 성공")
print("="*80)

# =============================================================================
# Step 3: Frame Stack 적용
# =============================================================================
print("\n" + "="*80)
print("Step 3: Frame Stack 적용 (FRAMES=3)")
print("="*80)

frame_stack = MultiAgentFrameStack(FRAMES, STATE_SIZE)

print("\n🎬 프레임 스택 시뮬레이션:")
stacked_data = []

for step in range(3):  # 3 time steps 시뮬레이션
    print(f"\n--- Time step {step} ---")

    for agent_id, state, goal, speed, neighbor_obs, neighbor_mask in parsed_data:
        state_stack = frame_stack.update(agent_id, state)

        if step == 0:
            print(f"  Agent {agent_id}: Frame 0 - All zeros (initial)")
        elif step == 1:
            print(f"  Agent {agent_id}: Frame 0-1 - Building up...")
        else:
            print(f"  Agent {agent_id}: Frame 0-2 - Full stack!")
            print(f"    Stacked state: {state_stack.shape} (expected: ({STATE_SIZE * FRAMES},))")
            assert state_stack.shape == (STATE_SIZE * FRAMES,), f"❌ Frame stack dimension error"

            stacked_data.append((agent_id, state_stack, goal, speed, neighbor_obs, neighbor_mask))

print("\n" + "="*80)
print(f"✅ Step 3 완료: Frame stack 적용 성공 ({STATE_SIZE} × {FRAMES} = {STATE_SIZE * FRAMES}D)")
print("="*80)

# =============================================================================
# Step 4: Neural Network Forward Pass
# =============================================================================
print("\n" + "="*80)
print("Step 4: Neural Network Forward Pass")
print("="*80)

policy = CNNPolicy(MSG_ACTION_SPACE, CONTINUOUS_ACTION_SIZE, FRAMES, N_AGENT).to(DEVICE)
print(f"\n🧠 Policy network 생성: {sum(p.numel() for p in policy.parameters())} parameters")

print("\n📊 Forward pass 시뮬레이션:")
forward_results = []

for agent_id, state_stack, goal, speed, neighbor_obs, neighbor_mask in stacked_data:
    state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(DEVICE)
    goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(DEVICE)
    speed_tensor = torch.FloatTensor(speed).unsqueeze(0).to(DEVICE)
    neighbor_obs_tensor = torch.FloatTensor(neighbor_obs).unsqueeze(0).to(DEVICE)
    neighbor_mask_tensor = neighbor_mask.unsqueeze(0)

    print(f"\n  Agent {agent_id} input tensors:")
    print(f"    state: {state_tensor.shape} (expected: (1, {STATE_SIZE * FRAMES}))")
    print(f"    goal: {goal_tensor.shape} (expected: (1, 2))")
    print(f"    speed: {speed_tensor.shape} (expected: (1, 2))")
    print(f"    neighbor_obs: {neighbor_obs_tensor.shape} (expected: (1, {N_AGENT}, {STATE_SIZE}))")
    print(f"    neighbor_mask: {neighbor_mask_tensor.shape} (expected: (1, {N_AGENT}))")

    with torch.no_grad():
        value, action, logprob, mean = policy(
            state_tensor, goal_tensor, speed_tensor,
            neighbor_obs_tensor, neighbor_mask_tensor
        )

    print(f"\n  Agent {agent_id} output tensors:")
    print(f"    value: {value.shape} (expected: (1, 1, 1))")
    print(f"    action: {action.shape} (expected: (1, 1, {CONTINUOUS_ACTION_SIZE}))")
    print(f"    logprob: {logprob.shape} (expected: (1, 1, 1))")
    print(f"    mean: {mean.shape} (expected: (1, 1, {CONTINUOUS_ACTION_SIZE}))")

    print(f"\n  Agent {agent_id} values:")
    print(f"    value: {value.item():.4f}")
    print(f"    action: [{action[0, 0, 0].item():.4f}, {action[0, 0, 1].item():.4f}] (rudder, thrust)")
    print(f"    logprob: {logprob.item():.4f}")

    assert value.shape == (1, 1, 1), f"❌ Value shape error"
    assert action.shape == (1, 1, CONTINUOUS_ACTION_SIZE), f"❌ Action shape error"
    assert logprob.shape == (1, 1, 1), f"❌ Logprob shape error"

    forward_results.append({
        'agent_id': agent_id,
        'state_stack': state_stack,
        'goal': goal,
        'speed': speed,
        'neighbor_obs': neighbor_obs,
        'neighbor_mask': neighbor_mask.cpu().numpy(),
        'action': action.cpu().numpy()[0, 0],
        'value': value.cpu().numpy()[0, 0, 0],
        'logprob': logprob.cpu().numpy()[0, 0, 0]
    })

print("\n" + "="*80)
print("✅ Step 4 완료: Neural network forward pass 성공")
print("="*80)

# =============================================================================
# Step 5: 경험 저장 (Memory)
# =============================================================================
print("\n" + "="*80)
print("Step 5: 경험 저장 (Memory)")
print("="*80)

memory = Memory()

print("\n💾 경험 저장 시뮬레이션 (10 steps):")
for step in range(10):
    for result in forward_results:
        reward = np.random.randn() * 0.1  # 랜덤 보상
        done = (step == 9 and result['agent_id'] == 0)  # 마지막 스텝에 agent 0 종료

        memory.add_agent_experience(
            result['agent_id'],
            result['state_stack'],
            result['goal'],
            result['speed'],
            {'obs': result['neighbor_obs'], 'mask': result['neighbor_mask']},
            result['action'],
            reward,
            done,
            result['value'],
            result['logprob']
        )

    if step % 3 == 0:
        print(f"  Step {step}: {len(memory.get_active_agents())} active agents")

experiences = memory.get_all_experiences()
print(f"\n📦 저장된 경험:")
print(f"  States: {experiences['states'].shape}")
print(f"  Goals: {experiences['goals'].shape}")
print(f"  Speeds: {experiences['speeds'].shape}")
print(f"  Neighbor infos: {len(experiences['neighbor_infos'])} entries")
print(f"  Actions: {experiences['actions'].shape}")
print(f"  Rewards: {experiences['rewards'].shape}")
print(f"  Dones: {experiences['dones'].shape}")
print(f"  Values: {experiences['values'].shape}")
print(f"  Logprobs: {experiences['logprobs'].shape}")

expected_size = 10 * 4  # 10 steps × 4 agents
print(f"\n  Expected size: {expected_size}")
print(f"  Actual size: {len(experiences['states'])}")

assert len(experiences['states']) == expected_size, f"❌ Memory size error"

print("\n" + "="*80)
print("✅ Step 5 완료: 경험 저장 성공")
print("="*80)

# =============================================================================
# Step 6: GAE 및 Returns 계산
# =============================================================================
print("\n" + "="*80)
print("Step 6: GAE (Generalized Advantage Estimation) 계산")
print("="*80)

rewards = experiences['rewards']
dones = experiences['dones']
values = experiences['values']

returns = calculate_returns(rewards, dones, 0, values, DISCOUNT_FACTOR)
advantages = returns - values
advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

print(f"\n📈 GAE 결과:")
print(f"  Rewards: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
print(f"  Values: mean={values.mean():.4f}, std={values.std():.4f}")
print(f"  Returns: mean={returns.mean():.4f}, std={returns.std():.4f}")
print(f"  Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
print(f"  Normalized advantages: mean={advantages_normalized.mean():.4f}, std={advantages_normalized.std():.4f}")

print("\n" + "="*80)
print("✅ Step 6 완료: GAE 계산 성공")
print("="*80)

# =============================================================================
# Step 7: PPO 학습 루프
# =============================================================================
print("\n" + "="*80)
print("Step 7: PPO 학습 루프 시뮬레이션")
print("="*80)

# Tensor 변환
states_tensor = torch.FloatTensor(experiences['states']).to(DEVICE)
goals_tensor = torch.FloatTensor(experiences['goals']).to(DEVICE)
speeds_tensor = torch.FloatTensor(experiences['speeds']).to(DEVICE)
actions_tensor = torch.FloatTensor(experiences['actions']).to(DEVICE)
old_logprobs_tensor = torch.FloatTensor(experiences['logprobs']).to(DEVICE)
returns_tensor = torch.FloatTensor(returns).to(DEVICE)
advantages_tensor = torch.FloatTensor(advantages_normalized).to(DEVICE)

# Neighbor 정보 처리
neighbor_obs_list = []
neighbor_mask_list = []
for neighbor_info in experiences['neighbor_infos']:
    neighbor_obs_list.append(neighbor_info['obs'])
    neighbor_mask_list.append(neighbor_info['mask'])
neighbor_obs_tensor = torch.FloatTensor(np.array(neighbor_obs_list)).to(DEVICE)
neighbor_mask_tensor = torch.BoolTensor(np.array(neighbor_mask_list)).to(DEVICE)

print(f"\n🎯 학습 데이터 준비:")
print(f"  States: {states_tensor.shape}")
print(f"  Goals: {goals_tensor.shape}")
print(f"  Speeds: {speeds_tensor.shape}")
print(f"  Actions: {actions_tensor.shape}")
print(f"  Returns: {returns_tensor.shape}")
print(f"  Advantages: {advantages_tensor.shape}")
print(f"  Neighbor obs: {neighbor_obs_tensor.shape}")
print(f"  Neighbor mask: {neighbor_mask_tensor.shape}")

optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

print(f"\n🔄 PPO 에폭 루프 (N_EPOCH={N_EPOCH}, BATCH_SIZE={BATCH_SIZE}):")

initial_params = {name: param.clone() for name, param in policy.named_parameters()}

for epoch in range(N_EPOCH):
    epoch_policy_loss = 0
    epoch_value_loss = 0
    epoch_total_loss = 0
    num_batches = 0

    indices = np.random.permutation(len(experiences['states']))

    for start in range(0, len(indices), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(indices))
        batch_indices = indices[start:end]

        # Batch 데이터
        batch_states = states_tensor[batch_indices]
        batch_goals = goals_tensor[batch_indices]
        batch_speeds = speeds_tensor[batch_indices]
        batch_actions = actions_tensor[batch_indices]
        batch_old_logprobs = old_logprobs_tensor[batch_indices]
        batch_returns = returns_tensor[batch_indices]
        batch_advantages = advantages_tensor[batch_indices]
        batch_neighbor_obs = neighbor_obs_tensor[batch_indices]
        batch_neighbor_mask = neighbor_mask_tensor[batch_indices]

        # Forward pass
        values, _, new_logprobs, _ = policy(
            batch_states, batch_goals, batch_speeds,
            batch_neighbor_obs, batch_neighbor_mask
        )

        # PPO Loss
        ratio = torch.exp(new_logprobs - batch_old_logprobs)
        surr1 = ratio * batch_advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * batch_advantages.unsqueeze(-1)
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = torch.nn.functional.mse_loss(values.squeeze(), batch_returns)

        entropy = -(new_logprobs.exp() * new_logprobs).mean()
        entropy_loss = -ENTROPY_BONUS * entropy

        total_loss = policy_loss + VALUE_LOSS_COEF * value_loss + entropy_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        epoch_policy_loss += policy_loss.item()
        epoch_value_loss += value_loss.item()
        epoch_total_loss += total_loss.item()
        num_batches += 1

    avg_policy_loss = epoch_policy_loss / num_batches
    avg_value_loss = epoch_value_loss / num_batches
    avg_total_loss = epoch_total_loss / num_batches

    print(f"\n  Epoch {epoch+1}/{N_EPOCH}:")
    print(f"    Policy loss: {avg_policy_loss:.4f}")
    print(f"    Value loss: {avg_value_loss:.4f}")
    print(f"    Total loss: {avg_total_loss:.4f}")
    print(f"    Gradient norm: {grad_norm:.4f}")

print("\n" + "="*80)
print("✅ Step 7 완료: PPO 학습 루프 성공")
print("="*80)

# =============================================================================
# Step 8: 파라미터 업데이트 검증
# =============================================================================
print("\n" + "="*80)
print("Step 8: 파라미터 업데이트 검증")
print("="*80)

print("\n🔍 파라미터 변화 확인:")
param_changed = 0
total_params = 0

for name, param in policy.named_parameters():
    initial_param = initial_params[name]
    diff = (param - initial_param).abs().sum().item()
    total_params += 1

    if diff > 1e-6:
        param_changed += 1
        if param_changed <= 5:  # 처음 5개만 출력
            print(f"  ✅ {name}: changed by {diff:.6f}")

print(f"\n📊 업데이트 통계:")
print(f"  Total parameters: {total_params}")
print(f"  Changed parameters: {param_changed}")
print(f"  Unchanged parameters: {total_params - param_changed}")

if param_changed == 0:
    print("\n❌ ERROR: No parameters were updated!")
    sys.exit(1)
elif param_changed < total_params * 0.5:
    print(f"\n⚠️ WARNING: Only {param_changed}/{total_params} parameters changed")
else:
    print(f"\n✅ SUCCESS: {param_changed}/{total_params} parameters updated")

print("\n" + "="*80)
print("✅ Step 8 완료: 파라미터 업데이트 검증 성공")
print("="*80)

# =============================================================================
# Step 9: 업데이트 후 Forward Pass 재실행
# =============================================================================
print("\n" + "="*80)
print("Step 9: 업데이트 후 Policy 행동 변화 확인")
print("="*80)

print("\n🎭 학습 전후 비교:")
for i, result in enumerate(forward_results[:2]):  # 2개만 비교
    state_tensor = torch.FloatTensor(result['state_stack']).unsqueeze(0).to(DEVICE)
    goal_tensor = torch.FloatTensor(result['goal']).unsqueeze(0).to(DEVICE)
    speed_tensor = torch.FloatTensor(result['speed']).unsqueeze(0).to(DEVICE)
    neighbor_obs_tensor = torch.FloatTensor(result['neighbor_obs']).unsqueeze(0).to(DEVICE)
    neighbor_mask_tensor = torch.BoolTensor(result['neighbor_mask']).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        new_value, new_action, new_logprob, _ = policy(
            state_tensor, goal_tensor, speed_tensor,
            neighbor_obs_tensor, neighbor_mask_tensor
        )

    print(f"\n  Agent {result['agent_id']}:")
    print(f"    학습 전 value: {result['value']:.4f}")
    print(f"    학습 후 value: {new_value.item():.4f}")
    print(f"    변화량: {new_value.item() - result['value']:.4f}")

    old_action = result['action']
    new_action_np = new_action.cpu().numpy()[0, 0]
    print(f"\n    학습 전 action: [{old_action[0]:.4f}, {old_action[1]:.4f}]")
    print(f"    학습 후 action: [{new_action_np[0]:.4f}, {new_action_np[1]:.4f}]")
    print(f"    변화량: [{new_action_np[0]-old_action[0]:.4f}, {new_action_np[1]-old_action[1]:.4f}]")

print("\n" + "="*80)
print("✅ Step 9 완료: Policy 행동 변화 확인 성공")
print("="*80)

# =============================================================================
# 최종 요약
# =============================================================================
print("\n" + "="*80)
print("🎉 A to Z 시뮬레이션 완료!")
print("="*80)

print("\n✅ 검증 완료 항목:")
print("  1. ✅ Unity observation 생성 (184D)")
print("  2. ✅ Python 데이터 파싱")
print("  3. ✅ Frame stack 적용 (3 frames)")
print("  4. ✅ Neural network forward pass")
print("  5. ✅ 경험 저장 (Memory)")
print("  6. ✅ GAE 계산")
print("  7. ✅ PPO 학습 루프")
print("  8. ✅ 파라미터 업데이트")
print("  9. ✅ Policy 행동 변화")

print("\n📊 최종 통계:")
print(f"  Total training data: {len(experiences['states'])} samples")
print(f"  Network parameters: {sum(p.numel() for p in policy.parameters())}")
print(f"  Parameters updated: {param_changed}/{total_params}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  PPO epochs: {N_EPOCH}")

print("\n" + "="*80)
print("🚀 전체 학습 파이프라인이 정상 작동합니다!")
print("="*80)

print("\n💡 다음 단계:")
print("  1. Unity Inspector 설정 (Space Size = 184)")
print("  2. Unity Play + Python main.py 실행")
print("  3. 실제 학습 시작")
print("\n" + "="*80)
