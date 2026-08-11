import os
# CUDA allocator 최적화 — 반드시 torch import 전에 설정 (fragmentation 50%↓)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                       'max_split_size_mb:128,expandable_segments:True')

import random
import torch
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import csv
import math
import time
import threading
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from config import *
from networks import CNNPolicy
from memory import Memory
from functions import RunningMeanStd
from frame_stack import MultiAgentFrameStack
from obs_utils import parse_observation, get_comm_partners

# Conv1D 성능 부스트 (kernel autotune + TF32)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except AttributeError:
    pass

# ★구간별 프로파일 (VESSEL_PROFILE=1): rollout 시간이 어디에 드는지 update마다 출력.
#   collect=obs 파싱(Phase1), infer=GPU 추론(Phase2), store=경험 저장(Phase2), env_step=Unity 시뮬+IPC(Phase3)
PROFILE = os.environ.get('VESSEL_PROFILE', '0') == '1'
PROF = {'collect': 0.0, 'infer': 0.0, 'store': 0.0, 'env_step': 0.0, 'n': 0}

def setup_logging():
    """로깅 설정"""
    csv_dir = os.path.join(SAVE_PATH, 'csv_logs')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    episode_log_file = os.path.join(csv_dir, 'episode_logs.csv')
    training_log_file = os.path.join(csv_dir, 'training_logs.csv')
    message_log_file = os.path.join(csv_dir, 'message_logs.csv')

    with open(episode_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'avg_reward', 'total_reward',
            'collision_count', 'collision_rate', 'success_count', 'success_rate',
            'episode_length', 'active_agents', 'learning_rate'
        ])

    with open(training_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'policy_loss', 'value_loss', 'entropy_loss',
            'colregs_loss', 'total_loss', 'learning_rate', 'gradient_norm'
        ])

    # 메시지 로그 (6D 메시지 + 메타데이터)
    with open(message_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'total_steps', 'agent_id',
            'msg_0', 'msg_1', 'msg_2', 'msg_3', 'msg_4', 'msg_5',
            'others_msg_0', 'others_msg_1', 'others_msg_2', 'others_msg_3', 'others_msg_4', 'others_msg_5',
            'n_agents', 'colregs_situation'
        ])

    return episode_log_file, training_log_file, message_log_file



# parse_observation, get_comm_partners → obs_utils.py로 이동


def ppo_update(policy, optimizer, memory, writer, total_steps, training_log_file):
    """PPO 학습 업데이트"""
    experiences = memory.get_all_experiences()

    if len(experiences['states']) == 0:
        return

    n_samples = len(experiences['states'])

    # ✅ Returns는 이미 에이전트별로 계산되어 있음 (memory.get_all_experiences에서)
    states = experiences['states']
    values = experiences['values']
    returns = experiences['returns']  # 에이전트별 GAE 계산 완료

    # Advantages 계산
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Tensor로 변환 (own obs: [N,1,dim])
    states_tensor = torch.as_tensor(states, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    goals_tensor = torch.as_tensor(experiences['goals'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    self_states_tensor = torch.as_tensor(experiences['self_states'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    actions_tensor = torch.as_tensor(experiences['actions'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    old_logprobs_tensor = torch.as_tensor(experiences['logprobs'], dtype=torch.float32, device=DEVICE)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)
    advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)

    # ★ 통신 파트너 obs: [N,K,dim] (sender→receiver gradient 재구성용) ★
    p_states_tensor = torch.as_tensor(experiences['partner_states'], dtype=torch.float32, device=DEVICE)
    p_goals_tensor = torch.as_tensor(experiences['partner_goals'], dtype=torch.float32, device=DEVICE)
    p_selfs_tensor = torch.as_tensor(experiences['partner_selfs'], dtype=torch.float32, device=DEVICE)
    p_masks_tensor = torch.as_tensor(experiences['partner_masks'], dtype=torch.float32, device=DEVICE).unsqueeze(-1)  # [N,K,1]
    p_relpos_tensor = torch.as_tensor(experiences['partner_relpos'], dtype=torch.float32, device=DEVICE)  # [N,K,3] 위치 grounding
    p_sits_tensor = torch.as_tensor(experiences['partner_situations'], dtype=torch.long, device=DEVICE)  # [N,K] 파트너 COLREGs 상황(완전분리 MoE MessageActor 라우팅)
    # ★ intent self-supervised 미래라벨 [N,1,K*3] (INTENT_COEF=0이면 zeros → 미사용)
    own_future_tensor = torch.as_tensor(experiences['own_future'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    own_future_mask_tensor = torch.as_tensor(experiences['own_future_mask'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    # ★ L1 threat-relay 라벨 [N,1,K*4] (THREAT_COEF=0이면 zeros → 미사용)
    own_threat_tensor = torch.as_tensor(experiences['own_threat'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    own_threat_mask_tensor = torch.as_tensor(experiences['own_threat_mask'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    # ★ COLREGs 상황 [N] (MoE head 라우팅; rollout 저장값 그대로 → 동일 head 재라우팅 = PPO ratio 유효)
    situations_tensor = torch.as_tensor(experiences['situations'], dtype=torch.long, device=DEVICE)

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy_loss = 0
    total_intent_loss = 0
    total_threat_loss = 0
    total_goal_loss = 0
    total_role_loss = 0
    total_consumer_loss = 0
    total_msg_reg = 0
    total_loss = 0
    num_updates = 0
    grad_norm = 0.0
    # ★통신 채널 건강 텔레메트리 누적 (2026-06-30 A2Z 감사: weight norm만으론 "동결=grad 0" vs "압력부족=grad 미미" 구분 불가)
    cg_msgout = cg_fc2ctr = cg_fc2cri = cg_gate = cg_vproj = cg_msgenc = 0.0
    def _gnorm(t):
        return 0.0 if t is None else float(t.norm())

    for epoch in range(N_EPOCH):
        indices = np.random.permutation(n_samples)

        # PPO mini-batch (rollout BATCH_SIZE보다 작게 → GPU peak memory 1/4)
        for start in range(0, n_samples, MINIBATCH_SIZE):
            end = min(start + MINIBATCH_SIZE, n_samples)
            batch_indices = indices[start:end]

            batch_states = states_tensor[batch_indices]
            batch_goals = goals_tensor[batch_indices]
            batch_self_states = self_states_tensor[batch_indices]
            batch_actions = actions_tensor[batch_indices]
            batch_old_logprobs = old_logprobs_tensor[batch_indices]
            batch_returns = returns_tensor[batch_indices]
            batch_advantages = advantages_tensor[batch_indices]
            # 파트너 obs (통신 gradient)
            batch_p_states = p_states_tensor[batch_indices]
            batch_p_goals = p_goals_tensor[batch_indices]
            batch_p_selfs = p_selfs_tensor[batch_indices]
            batch_p_masks = p_masks_tensor[batch_indices]
            batch_p_relpos = p_relpos_tensor[batch_indices]
            batch_p_sits = p_sits_tensor[batch_indices]
            batch_own_future = own_future_tensor[batch_indices]
            batch_own_future_mask = own_future_mask_tensor[batch_indices]
            batch_own_threat = own_threat_tensor[batch_indices]
            batch_own_threat_mask = own_threat_mask_tensor[batch_indices]
            batch_situations = situations_tensor[batch_indices]

            values, logprobs, dist_entropy, msg_reg, intent_loss, threat_loss, goal_loss, role_loss, consumer_loss = policy.evaluate_actions(
                batch_states, batch_goals, batch_self_states,
                batch_p_states, batch_p_goals, batch_p_selfs, batch_p_masks,
                batch_p_relpos, batch_actions, batch_own_future, batch_own_future_mask,
                own_threat=batch_own_threat, own_threat_mask=batch_own_threat_mask,
                situation=batch_situations, partner_situations=batch_p_sits
            )

            values = values.squeeze(-1).squeeze(-1)
            logprobs = logprobs.squeeze(-1).squeeze(-1)

            ratio = torch.exp(logprobs - batch_old_logprobs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, batch_returns)
            entropy_loss = -ENTROPY_BONUS * dist_entropy

            # ★ 메시지 L2 정규화 (통신 안정화: 쓸모없으면 메시지→0으로 수렴). comm-OFF면 msg_reg=0.
            loss = policy_loss + VALUE_LOSS_COEF * value_loss + entropy_loss + MSG_L2_COEF * msg_reg
            # ★ 메시지 게이트 개방 페널티 (ON만): sigmoid(gate)를 0쪽으로 압박 → 메시지가 advantage를
            #   유의하게 줄일 때만 gate가 열림 = value-of-information≥0를 수렴까지 보장(H1a 위배 근본수정).
            #   comm-OFF는 others_msg≡0이라 gate 무영향 → loss에 더해도 OFF 학습 불변(anti-rigging).
            if USE_COMMUNICATION:
                # 완전분리 MoE면 활성 코어(단일=1, MoE=5)들의 sigmoid(gate) 합 → 모든 게이트로 grad.
                gate_open = policy.ctr_actor.gate_open_sum() + policy.critic.gate_open_sum()
                loss = loss + MSG_GATE_COEF * gate_open
            # ★ intent self-supervised 손실 (ON·COEF>0만): 메시지가 sender 미래의도 인코딩하도록.
            #   별도 head라 정책/가치 무오염; OFF/COEF=0이면 intent_loss=0 → loss 불변(비트동일).
            if USE_COMMUNICATION and INTENT_COEF > 0.0:
                loss = loss + INTENT_COEF * intent_loss
            # ★ L1 threat-relay 손실 (ON·COEF>0만): 메시지가 sender가 본 제3 위협 기하 인코딩하도록.
            #   별도 head라 정책/가치 무오염; OFF/COEF=0이면 threat_loss=0 → loss 불변(비트동일).
            if USE_COMMUNICATION and THREAT_COEF > 0.0:
                loss = loss + THREAT_COEF * threat_loss
            # ★ L4 goal-broadcast 손실 (ON·COEF>0만): 메시지가 sender 목적지(의도) 인코딩하도록.
            #   별도 head라 정책/가치 무오염; OFF/COEF=0이면 goal_loss=0 → loss 불변(비트동일).
            if USE_COMMUNICATION and GOAL_COMM_COEF > 0.0:
                loss = loss + GOAL_COMM_COEF * goal_loss
            # ★ Role-broadcast 손실 (C, ON·COEF>0만): 메시지가 sender COLREGs 역할(give-way/stand-on) 인코딩하도록.
            #   별도 head라 정책/가치 무오염; OFF/COEF=0이면 role_loss=0 → loss 불변(비트동일).
            if USE_COMMUNICATION and ROLE_COMM_COEF > 0.0:
                loss = loss + ROLE_COMM_COEF * role_loss
            # ★C5c consumer 손실 (ON·COEF>0만): 수신 정책이 파트너 의도(goal)를 표상하도록 강제(수신측 gradient).
            #   기존 aux는 생산자만 학습 → 수신 단절(C5c)이 comm 무승부의 binding 원인. OFF/COEF=0이면 미가산(비트동일).
            if USE_COMMUNICATION and COMM_CONSUMER_COEF > 0.0:
                loss = loss + COMM_CONSUMER_COEF * consumer_loss

            optimizer.zero_grad()
            loss.backward()
            # ★채널별 grad L2 (clip 전 = 진짜 신호). comm-OFF면 스킵=불변. MoE는 코어 합산, 미라우팅 코어 grad=None→0.
            if USE_COMMUNICATION:
                try:
                    _mc = policy.msg_actor.cores()
                    _md = _mc[0].msg_out.weight.shape[0]
                    for _c in _mc:
                        cg_msgout += _gnorm(_c.msg_out.weight.grad)
                    for _c in policy.ctr_actor.cores():
                        _g = _c.fc2.weight.grad
                        cg_fc2ctr += 0.0 if _g is None else float(_g[:, -_md:].norm())
                        cg_gate += _gnorm(_c.msg_gate.grad)
                    for _c in policy.critic.cores():
                        _g = _c.fc2.weight.grad
                        cg_fc2cri += 0.0 if _g is None else float(_g[:, -_md:].norm())
                        cg_gate += _gnorm(_c.msg_gate.grad)
                    if getattr(policy, 'attn', None) is not None:
                        cg_vproj += _gnorm(policy.attn.v_proj.weight.grad)
                    # ★pos_ground(기본 집계) 활성 경로 감시: attention OFF일 땐 v_proj가 죽은 경로라
                    #   msg_encoder grad가 실제 채널 생존 신호.
                    if getattr(policy, 'msg_encoder', None) is not None:
                        cg_msgenc += _gnorm(policy.msg_encoder[0].weight.grad)
                except Exception:
                    pass
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_intent_loss += float(intent_loss)
            total_threat_loss += float(threat_loss)
            total_goal_loss += float(goal_loss)
            total_role_loss += float(role_loss)
            total_consumer_loss += float(consumer_loss)
            total_msg_reg += float(msg_reg)
            total_loss += loss.item()
            num_updates += 1

    if num_updates > 0:
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        avg_total_loss = total_loss / num_updates

        writer.add_scalar('Loss/Policy', avg_policy_loss, total_steps)
        writer.add_scalar('Loss/Value', avg_value_loss, total_steps)
        writer.add_scalar('Loss/Entropy', avg_entropy_loss, total_steps)
        writer.add_scalar('Loss/Intent', total_intent_loss / num_updates, total_steps)
        writer.add_scalar('Loss/Threat', total_threat_loss / num_updates, total_steps)
        writer.add_scalar('Loss/GoalComm', total_goal_loss / num_updates, total_steps)
        writer.add_scalar('Loss/RoleComm', total_role_loss / num_updates, total_steps)
        writer.add_scalar('Loss/CommConsumer', total_consumer_loss / num_updates, total_steps)
        writer.add_scalar('Loss/MsgReg', total_msg_reg / num_updates, total_steps)
        writer.add_scalar('Loss/Total', avg_total_loss, total_steps)

        # ★통신 채널 건강 (2026-06-30 A2Z 감사). grad≈0 단조 = 동결(구조사, 현 코드론 불가) / 미미 = 압력부족 감쇠.
        #   GateMean이 0.5→0 하강 = 수신측 메시지 외면. comm-OFF면 전부 스킵.
        if USE_COMMUNICATION:
            writer.add_scalar('Comm/Grad_MsgOut', cg_msgout / num_updates, total_steps)
            writer.add_scalar('Comm/Grad_Fc2Ctr', cg_fc2ctr / num_updates, total_steps)
            writer.add_scalar('Comm/Grad_Fc2Critic', cg_fc2cri / num_updates, total_steps)
            writer.add_scalar('Comm/Grad_Gate', cg_gate / num_updates, total_steps)
            writer.add_scalar('Comm/Grad_Vproj', cg_vproj / num_updates, total_steps)
            writer.add_scalar('Comm/Grad_MsgEncoder', cg_msgenc / num_updates, total_steps)
            try:
                _gm_c = [float(torch.sigmoid(c.msg_gate)) for c in policy.ctr_actor.cores()]
                _gm_v = [float(torch.sigmoid(c.msg_gate)) for c in policy.critic.cores()]
                writer.add_scalar('Comm/GateMean_Ctr', sum(_gm_c) / len(_gm_c), total_steps)
                writer.add_scalar('Comm/GateMean_Critic', sum(_gm_v) / len(_gm_v), total_steps)
            except Exception:
                pass

        with open(training_log_file, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                0, total_steps, avg_policy_loss, avg_value_loss, avg_entropy_loss,
                0.0, avg_total_loss, LEARNING_RATE, grad_norm
            ])

    # GPU peak memory 해제 (5 process 동시 학습 시 fragmentation 방지)
    del states_tensor, goals_tensor, self_states_tensor
    del actions_tensor, old_logprobs_tensor, returns_tensor, advantages_tensor
    del p_states_tensor, p_goals_tensor, p_selfs_tensor, p_masks_tensor, p_relpos_tensor
    del p_sits_tensor
    del own_future_tensor, own_future_mask_tensor, situations_tensor
    del own_threat_tensor, own_threat_mask_tensor
    del experiences   # numpy dict 사본도 즉시 해제 (RAM 7-10GB 누적 방지)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _compute_relpos(rx, rz, rheading_deg, px, pz):
    """파트너 j의 수신자 i 기준 상대위치 = [sin(방위), cos(방위), 거리/COMM_RANGE].
    방위 = i의 heading 기준 (Unity: heading 0=+Z, 시계방향; 방향벡터=(sinθ,cosθ))."""
    dx, dz = px - rx, pz - rz
    rng = math.hypot(dx, dz)
    bearing = math.atan2(dx, dz) - math.radians(rheading_deg)
    return [math.sin(bearing), math.cos(bearing), min(rng / COMM_RANGE, 1.0)]


def collect_observations(env, env_idx, behavior_name, frame_stack):
    """
    Phase 1: Unity에서 observation 수집
    Returns: env_data dict or None
    """
    try:
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        n_agents = len(agent_ids)
        if n_agents == 0:
            return {
                'env_idx': env_idx,
                'n_agents': 0,
                'decision_steps': decision_steps,
                'terminal_steps': terminal_steps,
                'empty': True
            }

        batch_states = []
        batch_goals = []
        batch_self_states = []
        batch_positions = {}
        batch_situations = []   # COLREGs 상황(0~4) — MoE 라우팅(agent_id_list 정렬)
        agent_id_list = []
        global_id_list = []
        rewards = []

        # enumerate로 직접 인덱스 사용 (O(n) -> O(1))
        for idx, agent_id in enumerate(agent_ids):
            obs_raw = decision_steps.obs[0][idx]
            state, goal, self_state, _, position, situation = parse_observation(obs_raw)

            global_id = f"env{env_idx}_{agent_id}"
            state_stacked = frame_stack.update(agent_id, state)

            batch_states.append(state_stacked)
            batch_goals.append(goal)
            batch_self_states.append(self_state)
            batch_positions[agent_id] = position
            batch_situations.append(situation)
            agent_id_list.append(agent_id)
            global_id_list.append(global_id)
            rewards.append(decision_steps.reward[idx])

        # 통신 파트너 계산 + 상대위치(위치 grounding용)
        comm_partners = {}
        comm_relpos = {}
        _aididx = {aid: k for k, aid in enumerate(agent_id_list)}
        # ★ receive-only 에이전트(부분 참여): 수신은 정상(자기 partner 유지), 송신 안 함.
        #   VESSEL_RECV_ONLY_COUNT개(정렬된 agent_id 앞쪽 N개)를 receive-only로 지정 →
        #   다른 모든 배의 partner 목록에서 제외(아무도 이 배 메시지 수신X). comm-OFF와 full-comm 사이 조건.
        _recv_only_n = int(os.environ.get('VESSEL_RECV_ONLY_COUNT', 0))
        _recv_only_ids = set(sorted(agent_id_list)[:_recv_only_n]) if _recv_only_n > 0 else set()
        for agent_id in agent_id_list:
            partners = get_comm_partners(agent_id, batch_positions[agent_id], batch_positions)
            if _recv_only_ids:
                partners = [p for p in partners if p not in _recv_only_ids]  # receive-only는 송신X → partner 제외
            comm_partners[agent_id] = partners
            rx, rz = batch_positions[agent_id]
            rheading = float(batch_self_states[_aididx[agent_id]][2]) * 180.0   # heading_norm → deg
            rp = [_compute_relpos(rx, rz, rheading, batch_positions[p][0], batch_positions[p][1]) for p in partners]
            comm_relpos[agent_id] = np.array(rp, dtype=np.float32) if rp else np.zeros((0, 3), dtype=np.float32)

        return {
            'env_idx': env_idx,
            'n_agents': n_agents,
            'batch_states': batch_states,
            'batch_goals': batch_goals,
            'batch_self_states': batch_self_states,
            'agent_id_list': agent_id_list,
            'global_id_list': global_id_list,
            'comm_partners': comm_partners,
            'comm_relpos': comm_relpos,
            'batch_positions': batch_positions,
            'batch_situations': batch_situations,
            'rewards': rewards,
            'decision_steps': decision_steps,
            'terminal_steps': terminal_steps,
            'empty': False
        }
    except Exception as e:
        print(f"  [ERROR] collect_observations env {env_idx}: {e}")
        return None


def send_actions(env, env_idx, behavior_name, decision_steps, actions_for_env):
    """
    Phase 3: Unity에 action 전송 (병렬 실행)
    """
    try:
        action_tuple = ActionTuple(continuous=actions_for_env)
        env.set_actions(behavior_name, action_tuple)
        env.step()
        return True
    except Exception as e:
        print(f"  [ERROR] send_actions env {env_idx}: {e}")
        return False


def setup_environments():
    """Unity 환경 생성 및 연결"""
    envs = []
    channels = []
    behavior_names = []

    if USE_EDITOR:
        print(f"\n[INFO] EDITOR mode - Unity Editor 에서 Play 를 누르세요 "
              f"(port {BASE_PORT}, connect wait max 120s)...", flush=True)
    else:
        print(f"\n[INFO] Creating {NUM_ENVS} Unity environments (headless build)...", flush=True)

    for i in range(NUM_ENVS):
        channel = EngineConfigurationChannel()
        if USE_EDITOR:
            # 에디터 직결: file_name=None → Python이 Play 누를 때까지 대기.
            # 에디터는 단일 인스턴스(worker_id=0)만 가능, batchmode 인자 없음.
            env = UnityEnvironment(
                file_name=None,
                side_channels=[channel],
                worker_id=0,
                base_port=BASE_PORT,
                timeout_wait=600,
            )
        else:
            # VESSEL_GRAPHICS=1: 창 띄워서 관찰 (headless 끔). 기본은 headless(빠름).
            _gfx = os.environ.get('VESSEL_GRAPHICS') == '1'
            _addargs = [] if _gfx else ["-batchmode", "-nographics"]
            env = UnityEnvironment(
                file_name=ENV_PATH,  # 빌드된 exe 사용 (병렬 학습)
                side_channels=[channel],
                worker_id=i,
                base_port=BASE_PORT + i,
                timeout_wait=60,
                additional_args=_addargs
            )
        channel.set_configuration_parameters(time_scale=TIME_SCALE)
        env.reset()
        behavior_name = list(env.behavior_specs)[0]

        # ★빌드 함정 가드 (2026-06-28): stale 빌드(옛 obs 크기)가 연결되면 obs_utils가 situation=0으로
        #   폴백 → MoE가 expert0로만 라우팅 = 무의미한 학습을 몇 시간 돌리는 시간낭비. Unity가 보고하는
        #   실제 obs 벡터 크기가 config.OBSERVATION_SIZE(369)와 다르면 *연결 즉시 실패*시켜 차단.
        #   (이 프로젝트 트리에서 빌드한 exe만 연결할 것 — obs 차원이 다르면 stale 빌드.)
        _spec = env.behavior_specs[behavior_name]
        _obs_len = int(sum(int(np.prod(o.shape)) for o in _spec.observation_specs))
        if _obs_len != OBSERVATION_SIZE:
            raise RuntimeError(
                f"[BUILD TRAP] Unity obs={_obs_len}D != config OBSERVATION_SIZE={OBSERVATION_SIZE}D. "
                f"stale 빌드 의심 — 현재 프로젝트에서 재빌드 필요(obs 차원 불일치). "
                f"MoE면 situation 슬롯 부재로 expert0만 라우팅됨.")

        envs.append(env)
        channels.append(channel)
        behavior_names.append(behavior_name)
        print(f"  Environment {i}: port {BASE_PORT + i} - OK", flush=True)
    
    return envs, channels, behavior_names


def setup_training(policy):
    """학습에 필요한 optimizer, writer, log 파일 설정"""
    # Phase 2: MessageActor에 높은 학습률
    if USE_COMMUNICATION:
        msg_params = list(policy.msg_actor.parameters())
        msg_ids = set(id(p) for p in msg_params)
        other_params = [p for p in policy.parameters() if id(p) not in msg_ids]
        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': LEARNING_RATE},
            {'params': msg_params, 'lr': LEARNING_RATE * MSG_LR_SCALE}
        ])
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    print(f"[OK] Policy Network: {sum(p.numel() for p in policy.parameters())} parameters")

    writer = SummaryWriter(log_dir=os.path.join(SAVE_PATH, 'logs'))
    episode_log_file, training_log_file, message_log_file = setup_logging()

    return optimizer, writer, episode_log_file, training_log_file, message_log_file


def training_step(step, envs, behavior_names, frame_stacks, policy, memory,
                  reward_rms, reward_buffer, stats, interval_stats,
                  agent_rewards, agent_episode_steps):
    """한 스텝의 obs 수집 + 추론 + action 전송 + terminal 처리 (환경 I/O 병렬)"""
    total_agents = 0
    last_env_actions = None
    _t0 = time.time()
    _infer_local = 0.0

    # ========== Phase 1: 환경에서 observation 수집 (병렬) ==========
    env_results = [None] * NUM_ENVS

    def _collect(env_idx):
        env_results[env_idx] = collect_observations(
            envs[env_idx], env_idx, behavior_names[env_idx],
            frame_stacks[env_idx]
        )

    threads = [threading.Thread(target=_collect, args=(i,)) for i in range(NUM_ENVS)]
    for t in threads: t.start()
    for t in threads: t.join()
    _t1 = time.time()
    if PROFILE:
        PROF['collect'] += _t1 - _t0

    env_data_list = [r for r in env_results if r is not None]

    # ★누수 방어(2026-06-18): collect에서 본 활성 agent_id 외의 stale frame_stack 버퍼 GC.
    #   ML-Agents agent_id는 respawn마다 새 값 → remove_agent(terminal)가 누락되면 무한 누적.
    #   commOFF(충돌 7.9% → respawn 빈번)가 ~23k ep서 크래시한 근본 차단. 활성 배는 매 step
    #   update되므로 절대 제거 안 됨(안전). 빈 env(n_agents=0)는 _seen 비어 retire skip(전체 GC 방지).
    for ed in env_data_list:
        if not ed.get('empty', True):
            frame_stacks[ed['env_idx']].retire_unseen()

    # ========== Phase 2: 환경별 분리 추론 (통신 파트너 적용) ==========
    actions_per_env = {}

    for env_data in env_data_list:
        if env_data['empty']:
            continue

        env_idx = env_data['env_idx']
        n = env_data['n_agents']
        total_agents += n

        # 환경별로 텐서 생성
        states_tensor = torch.as_tensor(np.array(env_data['batch_states']), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        goals_tensor = torch.as_tensor(np.array(env_data['batch_goals']), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        self_states_tensor = torch.as_tensor(np.array(env_data['batch_self_states']), dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # ★ comm_partners와 agent_id_list를 forward에 전달 (rollout others_msg = nearest-K sum) ★
        # ★ situation: COLREGs 상황(0~4)으로 MoE head 라우팅. 저장값과 동일 → update 재라우팅 정합(PPO).
        situation_tensor = torch.as_tensor(env_data['batch_situations'], dtype=torch.long, device=DEVICE).unsqueeze(0)
        _ti = time.time()
        with torch.no_grad():
            values, actions, logprobs, means, msg, others_msg = policy.forward(
                states_tensor, goals_tensor, self_states_tensor,
                return_msg=True,
                comm_partners=env_data['comm_partners'],
                agent_id_list=env_data['agent_id_list'],
                comm_relpos=env_data['comm_relpos'],
                situation=situation_tensor
            )

        env_actions = np.asarray(actions.squeeze(0).cpu().detach())
        env_values = np.asarray(values.squeeze(0).cpu().detach())
        env_logprobs = np.asarray(logprobs.squeeze(0).cpu().detach())
        _infer_local += time.time() - _ti
        last_env_actions = env_actions  # 로깅용

        # ★통신 계기판: rollout 메시지 크기 누적 (3000-step 블록에서 평균 내 로깅)
        if USE_COMMUNICATION:
            interval_stats['msg_abs_sum'] += float(msg.abs().mean())
            interval_stats['others_abs_sum'] += float(others_msg.abs().mean())
            interval_stats['msg_stat_count'] += 1

        # agent_id -> index 매핑 (O(1) 검색용)
        agent_id_to_idx = {aid: i for i, aid in enumerate(env_data['agent_id_list'])}
        comm_partners = env_data['comm_partners']
        agent_id_list = env_data['agent_id_list']
        state_dim = len(env_data['batch_states'][0])  # F*S

        # 경험 저장 (+ 통신 파트너 obs: sender→receiver gradient 재구성용)
        for i in range(n):
            global_id = env_data['global_id_list'][i]
            raw_reward = env_data['rewards'][i]

            reward_buffer.append(raw_reward)
            normalized_reward = raw_reward / (np.sqrt(reward_rms.var) + 1e-8)

            if global_id not in agent_rewards:
                agent_rewards[global_id] = 0
                agent_episode_steps[global_id] = 0
            agent_rewards[global_id] += raw_reward
            agent_episode_steps[global_id] += 1
            stats['total_reward'] += raw_reward
            interval_stats['reward_sum'] += raw_reward
            interval_stats['reward_count'] += 1

            # 통신 파트너 obs 수집 (nearest-K, zero-pad). rollout others_msg와 동일 파트너 집합.
            partners = comm_partners.get(agent_id_list[i], [])
            relpos_arr = env_data['comm_relpos'].get(agent_id_list[i])
            # ★ kept_pos: partners 중 agent_id_to_idx 생존 *위치* — pidx와 relpos를 같은 위치에서 추출해
            #   정렬을 구성적으로 보장(rollout _get_others_msg와 동일 규약; 중간 파트너 필터돼도 relpos[j]↔obs[j] 불변).
            #   현재 partners ⊆ agent_id_list라 필터 no-op(kept_pos=전체) → 기존 동작과 비트 동일.
            kept_pos = [k for k, p in enumerate(partners) if p in agent_id_to_idx][:MAX_COMM_PARTNERS]
            pidx = [agent_id_to_idx[partners[k]] for k in kept_pos]
            p_states = np.zeros((MAX_COMM_PARTNERS, state_dim), dtype=np.float32)
            p_goals = np.zeros((MAX_COMM_PARTNERS, GOAL_SIZE), dtype=np.float32)
            p_selfs = np.zeros((MAX_COMM_PARTNERS, SELF_STATE_SIZE), dtype=np.float32)
            p_mask = np.zeros((MAX_COMM_PARTNERS,), dtype=np.float32)
            p_relpos = np.zeros((MAX_COMM_PARTNERS, 3), dtype=np.float32)
            # ★완전분리 MoE: 파트너 COLREGs 상황(0~4) 저장 → update가 MessageActor를 *파트너 situation*으로
            #   재라우팅(rollout msg 생성과 동일 sender 코어) → PPO ratio 유효. padding 슬롯=0(None, mask로 제외).
            p_sits = np.zeros((MAX_COMM_PARTNERS,), dtype=np.int64)
            for j, k in enumerate(kept_pos):
                pj = pidx[j]
                p_states[j] = env_data['batch_states'][pj]
                p_goals[j] = env_data['batch_goals'][pj]
                p_selfs[j] = env_data['batch_self_states'][pj]
                p_mask[j] = 1.0
                p_sits[j] = int(env_data['batch_situations'][pj])
                if relpos_arr is not None and k < len(relpos_arr):
                    p_relpos[j] = relpos_arr[k]

            memory.add_agent_experience(
                global_id,
                env_data['batch_states'][i],
                env_data['batch_goals'][i],
                env_data['batch_self_states'][i],
                env_actions[i],
                normalized_reward,
                False,
                env_values[i, 0],
                env_logprobs[i, 0],
                p_states, p_goals, p_selfs, p_mask, p_relpos, p_sits,
                np.asarray(env_data['batch_positions'][agent_id_list[i]], dtype=np.float32),
                int(env_data['batch_situations'][i])
            )

        # Unity로 보낼 action 준비 (dict 사용으로 O(1) 검색)
        decision_steps = env_data['decision_steps']
        full_actions = np.zeros((len(decision_steps.agent_id), CONTINUOUS_ACTION_SIZE))
        for i, agent_id in enumerate(decision_steps.agent_id):
            if agent_id in agent_id_to_idx:
                full_actions[i] = env_actions[agent_id_to_idx[agent_id]]
        actions_per_env[env_idx] = full_actions

        # Terminal steps 처리 (최종 보상 저장 + done=True 설정)
        for t_idx, agent_id in enumerate(env_data['terminal_steps'].agent_id):
            raw_reward = env_data['terminal_steps'].reward[t_idx]
            global_id = f"env{env_idx}_{agent_id}"

            reward_buffer.append(raw_reward)
            normalized_reward = raw_reward / (np.sqrt(reward_rms.var) + 1e-8)

            if global_id in memory.agent_memories:
                memory.agent_memories[global_id].mark_done(normalized_reward)

            if global_id in agent_rewards:
                agent_rewards[global_id] += raw_reward
            else:
                agent_rewards[global_id] = raw_reward

            stats['total_reward'] += raw_reward
            interval_stats['reward_sum'] += raw_reward
            interval_stats['reward_count'] += 1

            interval_stats['terminal_count'] += 1

            if raw_reward > 0:
                stats['success_count'] += 1
                interval_stats['success_count'] += 1
            elif raw_reward < COLLISION_REWARD_THRESHOLD:
                stats['collision_count'] += 1
                interval_stats['collision_count'] += 1
            elif raw_reward < SPINNING_REWARD_THRESHOLD:
                stats['spinning_count'] += 1
                interval_stats['spinning_count'] += 1

            # 에피소드 카운터 리셋
            if global_id in agent_rewards:
                del agent_rewards[global_id]
            if global_id in agent_episode_steps:
                del agent_episode_steps[global_id]

            frame_stacks[env_idx].remove_agent(agent_id)

    _t2 = time.time()
    if PROFILE:
        PROF['infer'] += _infer_local
        PROF['store'] += (_t2 - _t1) - _infer_local

    # ========== Phase 3: 환경에 action 전송 (병렬) ==========
    def _send(env_data):
        env_idx = env_data['env_idx']
        if env_data['empty']:
            envs[env_idx].step()
        elif env_idx in actions_per_env:
            send_actions(
                envs[env_idx], env_idx, behavior_names[env_idx],
                env_data['decision_steps'], actions_per_env[env_idx]
            )
        else:
            envs[env_idx].step()

    threads = [threading.Thread(target=_send, args=(ed,)) for ed in env_data_list]
    for t in threads: t.start()
    for t in threads: t.join()
    if PROFILE:
        PROF['env_step'] += time.time() - _t2
        PROF['n'] += 1

    return total_agents, last_env_actions


def log_and_save(step, total_agents, last_env_actions, policy, optimizer,
                 memory, writer, reward_rms, reward_buffer, stats, interval_stats,
                 episode_log_file, training_log_file, training_start_time=None):
    """통계 출력 + 모델 저장 + tensorboard 로깅"""

    # Reward RMS 업데이트
    if len(reward_buffer) >= 100:
        reward_rms.update(np.array(reward_buffer))
        reward_buffer.clear()

    # PPO 업데이트 (message annealing 폐기 - 통신 즉시 100%, dead code 제거됨)
    if TRAIN_MODE and step > 0 and step % UPDATE_INTERVAL == 0:
        update_start = time.time()
        print(f"\n[UPDATE] PPO update at step {step}")
        ppo_update(policy, optimizer, memory, writer, step, training_log_file)
        memory.clear()
        elapsed = time.time() - training_start_time
        update_time = time.time() - update_start
        steps_per_sec = step / max(elapsed, 1)
        eta_hours = (RUN_STEP - step) / max(steps_per_sec, 1) / 3600
        print(f"  Update: {update_time:.1f}s | Total: {elapsed/60:.1f}min | {steps_per_sec:.0f} steps/s | ETA: {eta_hours:.1f}h")
        if PROFILE and PROF['n'] > 0:
            _n = PROF['n']
            _tot = PROF['collect'] + PROF['infer'] + PROF['store'] + PROF['env_step']
            print("  [PROFILE] per-iter: collect(parse) {:.2f}ms | infer(GPU) {:.2f}ms | store {:.2f}ms | "
                  "env_step(Unity+IPC) {:.2f}ms | rollout-sum {:.2f}ms | update(amort) {:.2f}ms".format(
                      1000 * PROF['collect'] / _n, 1000 * PROF['infer'] / _n, 1000 * PROF['store'] / _n,
                      1000 * PROF['env_step'] / _n, 1000 * _tot / _n, 1000 * update_time / UPDATE_INTERVAL), flush=True)
            for _k in ('collect', 'infer', 'store', 'env_step'):
                PROF[_k] = 0.0
            PROF['n'] = 0

    # 간단한 상태 출력 (500 스텝마다)
    if step % 500 == 0:
        if last_env_actions is not None and len(last_env_actions) > 0:
            # action 분포 확인 (rudder, thrust)
            rudder_mean = last_env_actions[:, 0].mean()
            thrust_mean = last_env_actions[:, 1].mean()
            print(f"[STEP {step}] Agents: {total_agents}, Action: rudder={rudder_mean:.3f}, thrust={thrust_mean:.3f}", flush=True)
        else:
            print(f"[STEP {step}] Agents: {total_agents}", flush=True)

    # 진행 상황 출력
    if step > 0 and step % 3000 == 0:
        avg_reward = interval_stats['reward_sum'] / max(interval_stats['reward_count'], 1)
        print(f"[STEP {step:,}/{MAX_STEPS:,}] Envs: {NUM_ENVS}, Agents: {total_agents}, "
              f"Collisions: {interval_stats['collision_count']} (total: {stats['collision_count']}), "
              f"Spinning: {interval_stats['spinning_count']} (total: {stats['spinning_count']}), "
              f"Success: {interval_stats['success_count']} (total: {stats['success_count']}), "
              f"Avg Reward: {avg_reward:.4f}")

        avg_normalized = avg_reward / (np.sqrt(reward_rms.var) + 1e-8)
        terminal_count = interval_stats['terminal_count']
        collision_rate = interval_stats['collision_count'] / max(terminal_count, 1)
        success_rate = interval_stats['success_count'] / max(terminal_count, 1)

        writer.add_scalar('Reward/Step_Raw', avg_reward, step)
        writer.add_scalar('Reward/Step_Normalized', float(avg_normalized), step)
        writer.add_scalar('Reward/RMS_Std', float(np.sqrt(reward_rms.var)), step)
        writer.add_scalar('Collision/Interval', interval_stats['collision_count'], step)
        writer.add_scalar('Collision/Total', stats['collision_count'], step)
        writer.add_scalar('Collision/Rate', collision_rate, step)
        writer.add_scalar('Success/Interval', interval_stats['success_count'], step)
        writer.add_scalar('Success/Total', stats['success_count'], step)
        writer.add_scalar('Success/Rate', success_rate, step)
        writer.add_scalar('Agents/Total', total_agents, step)

        # ★통신 채널 계기판 (2026-06-12 채널동결 사후대책): "채널이 살아있는가"를 학습 중 직접 관측.
        #   gate=듣는 정도(sigmoid), msg_abs=말하는 크기, *_norm=채널 가중치 성장(동결이면 init에서 불변).
        #   fc2_critic_rest 대비 fc2slice_critic 비율 >0.5면 critic 과조건화 경보(06-01 naive 실측 재발 감시).
        if USE_COMMUNICATION:
            md = policy.msg_dim
            n_msg = max(interval_stats['msg_stat_count'], 1)
            # 완전분리 MoE면 활성 코어(단일=1, MoE=5) 평균으로 집계 → comm_stats.csv 스키마 불변.
            gate_ctr = policy.ctr_actor.mean_gate_sigmoid()
            gate_cri = policy.critic.mean_gate_sigmoid()
            msg_abs = interval_stats['msg_abs_sum'] / n_msg
            others_abs = interval_stats['others_abs_sum'] / n_msg
            msg_out_norm = policy.msg_actor.mean_msg_out_norm()
            vproj_norm = float(policy.attn.v_proj.weight.norm())
            msgenc_norm = float(policy.msg_encoder[0].weight.norm())   # pos_ground 활성 경로 가중치
            fc2_ctr = policy.ctr_actor.fc2_msg_slice_norm()
            fc2_cri = policy.critic.fc2_msg_slice_norm()
            fc2_cri_rest = policy.critic.fc2_rest_norm()
            writer.add_scalar('Comm/gate_ctr', gate_ctr, step)
            writer.add_scalar('Comm/gate_critic', gate_cri, step)
            writer.add_scalar('Comm/msg_abs_mean', msg_abs, step)
            writer.add_scalar('Comm/others_msg_abs_mean', others_abs, step)
            writer.add_scalar('Comm/msg_out_norm', msg_out_norm, step)
            writer.add_scalar('Comm/vproj_norm', vproj_norm, step)
            writer.add_scalar('Comm/msg_encoder_norm', msgenc_norm, step)
            writer.add_scalar('Comm/fc2slice_ctr_norm', fc2_ctr, step)
            writer.add_scalar('Comm/fc2slice_critic_norm', fc2_cri, step)
            comm_stats_file = stats.get('comm_stats_file')
            if comm_stats_file:
                with open(comm_stats_file, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        step, f"{gate_ctr:.6f}", f"{gate_cri:.6f}", f"{msg_abs:.6f}",
                        f"{others_abs:.6f}", f"{msg_out_norm:.6f}", f"{vproj_norm:.6f}",
                        f"{fc2_ctr:.6f}", f"{fc2_cri:.6f}", f"{fc2_cri_rest:.6f}",
                        f"{msgenc_norm:.6f}"
                    ])

        # Episode CSV 로깅
        with open(episode_log_file, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                terminal_count, step, avg_reward, interval_stats['reward_sum'],
                interval_stats['collision_count'], collision_rate,
                interval_stats['success_count'], success_rate,
                interval_stats['reward_count'], total_agents, LEARNING_RATE
            ])

        interval_stats['reward_sum'] = 0
        interval_stats['reward_count'] = 0
        interval_stats['collision_count'] = 0
        interval_stats['spinning_count'] = 0
        interval_stats['success_count'] = 0
        interval_stats['terminal_count'] = 0
        interval_stats['msg_abs_sum'] = 0.0
        interval_stats['others_abs_sum'] = 0.0
        interval_stats['msg_stat_count'] = 0

    # 모델 저장 (reward_rms 포함)
    if TRAIN_MODE and step > 0 and step % 10000 == 0:
        save_path = os.path.join(SAVE_PATH, f'policy_step_{step}.pth')
        checkpoint = {
            'model_state_dict': policy.state_dict(),
            'msg_anneal_step': getattr(policy, 'msg_anneal_step', 0),
        }
        torch.save(checkpoint, save_path)
        rms_save = save_path.replace('.pth', '_reward_rms.npz')
        np.savez(rms_save, mean=reward_rms.mean, var=reward_rms.var, count=reward_rms.count)
        print(f"  Model saved: {save_path}")
        writer.flush()


def main():
    # 재현성을 위한 random seed 고정 (VESSEL_SEED로 multi-seed 비교 가능, 기본 42)
    _seed = int(os.environ.get('VESSEL_SEED', '42'))
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(_seed)
    print(f"[INFO] Random seed = {_seed}", flush=True)

    print("=" * 80)
    print("[START] Vessel ML-Agent Training (Multi-Instance)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Number of Environments: {NUM_ENVS}")
    print(f"Observation Size: {OBSERVATION_SIZE}D (per agent)")
    print(f"Message Dimension: {MSG_DIM}D")
    print(f"Communication Range: {COMM_RANGE}m")
    print(f"Max Communication Partners: {MAX_COMM_PARTNERS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Steps: {MAX_STEPS}")
    print("=" * 80)

    # 여러 Unity 환경 생성
    envs, channels, behavior_names = setup_environments()

    # Reward normalization
    reward_rms = RunningMeanStd()
    reward_buffer = []

    # 정책 네트워크
    policy = CNNPolicy(MSG_DIM, CONTINUOUS_ACTION_SIZE, FRAMES).to(DEVICE)

    if LOAD_MODEL and MODEL_PATH and os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            saved_state = checkpoint['model_state_dict']
            if 'msg_anneal_step' in checkpoint:
                policy.msg_anneal_step = checkpoint['msg_anneal_step']
                print(f"  [OK] Loaded msg_anneal_step = {policy.msg_anneal_step}")
        else:
            saved_state = checkpoint  # 이전 형식 호환
        model_state = policy.state_dict()
        filtered = {k: v for k, v in saved_state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in saved_state if k not in filtered]
        model_state.update(filtered)
        policy.load_state_dict(model_state)
        if skipped:
            print(f"[WARN] Skipped (shape mismatch): {skipped}")
        print(f"[OK] Model loaded: {MODEL_PATH} ({len(filtered)}/{len(saved_state)} layers)")

        # ★동결 체크포인트 감지 (2026-06-12): 06-02~06-11 구버전 ckpt는 채널 가중치 0/게이트 −8로
        #   동결돼 있어 로드 시 해동 init을 덮어씀 → 통신이 조용히 죽은 채 재개됨. 통신 실험은 from-scratch 필수.
        if USE_COMMUNICATION:
            _gate_sig = policy.ctr_actor.mean_gate_sigmoid()
            if (_gate_sig < 0.05
                    or float(policy.attn.v_proj.weight.norm()) < 1e-4
                    or policy.msg_actor.mean_msg_out_norm() < 1e-4):
                print(f"[WARN] 동결 체크포인트 감지 (gate sigmoid={_gate_sig:.4f}) — "
                      f"채널 해동 init이 덮어써졌습니다. 통신 실험은 from-scratch를 사용하세요.", flush=True)

        # reward_rms 복원
        rms_path = MODEL_PATH.replace('.pth', '_reward_rms.npz')
        if os.path.exists(rms_path):
            rms_data = np.load(rms_path)
            reward_rms.mean = rms_data['mean']
            reward_rms.var = rms_data['var']
            reward_rms.count = float(rms_data['count'])
            print(f"[OK] Reward RMS loaded: std={np.sqrt(reward_rms.var):.4f}")

    optimizer, writer, episode_log_file, training_log_file, message_log_file = setup_training(policy)

    # 학습 설정 기록
    try:
        from config import get_config_dict
        config_dict = get_config_dict()
        config_path = os.path.join(SAVE_PATH, 'config_snapshot.txt')
        with open(config_path, 'w') as f:
            for k, v in config_dict.items():
                f.write(f"{k}: {v}\n")
        print(f"  [OK] Config saved to {config_path}")
    except Exception as e:
        print(f"  [WARN] Config dump failed: {e}")

    memory = Memory()
    frame_stacks = [MultiAgentFrameStack(FRAMES, STATE_SIZE) for _ in range(NUM_ENVS)]

    # 통계
    stats = {
        'collision_count': 0,
        'spinning_count': 0,
        'success_count': 0,
        'total_reward': 0,
    }
    agent_rewards = {}
    agent_episode_steps = {}

    interval_stats = {
        'reward_sum': 0,
        'reward_count': 0,
        'collision_count': 0,
        'spinning_count': 0,
        'success_count': 0,
        'terminal_count': 0,
        'msg_abs_sum': 0.0,       # ★통신 계기판: rollout 메시지 |평균| 누적
        'others_abs_sum': 0.0,
        'msg_stat_count': 0,
    }

    # ★통신 계기판 CSV (2026-06-12 채널동결 사후대책): 결과 폴더(VESSEL_METRIC_LOG 옆)에 채널 생존
    #   신호를 기록 → 사후 가중치 포렌식 없이 "채널이 살아 있었나/게이트가 열렸나"를 데이터로 판정.
    if USE_COMMUNICATION:
        _metric_log = os.environ.get('VESSEL_METRIC_LOG')
        _comm_dir = os.path.dirname(_metric_log) if _metric_log else os.path.join(SAVE_PATH, 'csv_logs')
        os.makedirs(_comm_dir, exist_ok=True)
        stats['comm_stats_file'] = os.path.join(_comm_dir, 'comm_stats.csv')
        with open(stats['comm_stats_file'], 'w', newline='') as f:
            csv.writer(f).writerow([
                'step', 'gate_ctr', 'gate_critic', 'msg_abs_mean', 'others_msg_abs_mean',
                'msg_out_norm', 'vproj_norm', 'fc2slice_ctr_norm', 'fc2slice_critic_norm',
                'fc2_critic_rest_norm', 'msg_encoder_norm'
            ])

    # 시작 스텝 설정 (이어서 학습할 때 이전 스텝 누적)
    start_step = START_STEP if LOAD_MODEL else 0
    print(f"\n[TRAINING] Running for {MAX_STEPS} steps (starting from step {start_step:,})...")
    print(f"[INFO] Threaded parallel loop with {NUM_ENVS} environment(s)")

    # 타이밍 측정용 (rolling average, 무한 증가 방지)
    step_times = deque(maxlen=1000)
    training_start_time = time.time()

    for step in range(start_step, start_step + MAX_STEPS):
        step_start = time.time()

        total_agents, last_env_actions = training_step(
            step, envs, behavior_names, frame_stacks, policy, memory,
            reward_rms, reward_buffer, stats, interval_stats,
            agent_rewards, agent_episode_steps
        )

        step_time = time.time() - step_start
        step_times.append(step_time)

        log_and_save(
            step, total_agents, last_env_actions, policy, optimizer,
            memory, writer, reward_rms, reward_buffer, stats, interval_stats,
            episode_log_file, training_log_file, training_start_time
        )

    # 최종 통계
    print(f"\n{'=' * 80}")
    print(f"[DONE] Training completed!")
    print(f"  Total Steps: {MAX_STEPS}")
    print(f"  Total Environments: {NUM_ENVS}")
    print(f"  Total Collisions: {stats['collision_count']}")
    print(f"  Total Success: {stats['success_count']}")
    print(f"  Avg Reward: {stats['total_reward'] / max(len(agent_rewards), 1):.2f}")
    print(f"{'=' * 80}")

    # 환경 종료
    for env in envs:
        env.close()
    writer.close()


if __name__ == "__main__":
    main()
