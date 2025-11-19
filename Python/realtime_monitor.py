"""
실시간 학습 모니터링 도구

Unity ML-Agents 학습 중 실시간으로 메트릭과 latent message를 모니터링합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from collections import deque
import time
from pathlib import Path
import json

class RealtimeMonitor:
    """실시간 학습 모니터링"""

    def __init__(self, window_size=100):
        self.window_size = window_size

        # 데이터 버퍼
        self.rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.message_diversity = deque(maxlen=window_size)
        self.colregs_violations = deque(maxlen=window_size)

        # 플롯 초기화
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Vessel RL Training Monitor', fontsize=14)

    def update_metrics(self, reward, episode_length, messages, colregs_data):
        """메트릭 업데이트"""

        self.rewards.append(reward)
        self.episode_lengths.append(episode_length)

        # 메시지 다양성 계산 (표준편차)
        if messages is not None:
            diversity = np.std(messages)
            self.message_diversity.append(diversity)

        # COLREGs 위반 계산
        if colregs_data is not None:
            # 위험 상황 수 계산
            violations = np.sum(colregs_data > 0.7)
            self.colregs_violations.append(violations)

    def update_plots(self, frame):
        """실시간 플롯 업데이트"""

        # Clear axes
        for ax in self.axes.flat:
            ax.clear()

        # 1. Reward
        ax = self.axes[0, 0]
        if len(self.rewards) > 0:
            ax.plot(self.rewards, 'b-', alpha=0.7)
            ax.axhline(y=np.mean(self.rewards), color='r',
                      linestyle='--', label=f'Mean: {np.mean(self.rewards):.2f}')
            ax.set_title('Episode Rewards')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. Episode Length
        ax = self.axes[0, 1]
        if len(self.episode_lengths) > 0:
            ax.plot(self.episode_lengths, 'g-', alpha=0.7)
            ax.axhline(y=np.mean(self.episode_lengths), color='r',
                      linestyle='--', label=f'Mean: {np.mean(self.episode_lengths):.0f}')
            ax.set_title('Episode Lengths')
            ax.set_ylabel('Steps')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. Message Diversity
        ax = self.axes[1, 0]
        if len(self.message_diversity) > 0:
            ax.plot(self.message_diversity, 'purple', alpha=0.7)
            ax.set_title('Message Diversity (Std Dev)')
            ax.set_ylabel('Diversity')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)

        # 4. COLREGs Violations
        ax = self.axes[1, 1]
        if len(self.colregs_violations) > 0:
            ax.bar(range(len(self.colregs_violations)),
                  self.colregs_violations, color='red', alpha=0.7)
            ax.set_title('COLREGs High-Risk Situations')
            ax.set_ylabel('Count')
            ax.set_ylim([0, max(self.colregs_violations) + 1])

        plt.tight_layout()
        return self.axes

    def start_monitoring(self, update_interval=1000):
        """모니터링 시작"""
        print("Starting real-time monitoring...")
        print("Press Ctrl+C to stop")

        ani = FuncAnimation(self.fig, self.update_plots,
                          interval=update_interval, blit=False)
        plt.show()

        return ani


class MessageLogger:
    """Unity에서 메시지 데이터 로깅"""

    def __init__(self, log_path="message_logs"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(exist_ok=True)

        self.current_episode = 0
        self.message_buffer = []

    def log_message(self, agent_id, message_data, timestamp):
        """메시지 로깅"""
        entry = {
            "episode": self.current_episode,
            "agent_id": agent_id,
            "timestamp": timestamp,
            "message": message_data.tolist() if isinstance(message_data, np.ndarray) else message_data
        }
        self.message_buffer.append(entry)

        # 버퍼가 일정 크기 이상이면 저장
        if len(self.message_buffer) >= 100:
            self.save_buffer()

    def save_buffer(self):
        """버퍼를 파일로 저장"""
        filename = self.log_path / f"messages_ep{self.current_episode}.json"

        with open(filename, 'w') as f:
            json.dump(self.message_buffer, f, indent=2)

        print(f"Saved {len(self.message_buffer)} messages to {filename}")
        self.message_buffer = []

    def new_episode(self):
        """새 에피소드 시작"""
        if self.message_buffer:
            self.save_buffer()
        self.current_episode += 1

    def load_messages(self, episode=None):
        """저장된 메시지 로드"""
        if episode is None:
            # 모든 에피소드 로드
            all_messages = []
            for file in self.log_path.glob("messages_ep*.json"):
                with open(file, 'r') as f:
                    messages = json.load(f)
                    all_messages.extend(messages)
            return all_messages
        else:
            # 특정 에피소드 로드
            filename = self.log_path / f"messages_ep{episode}.json"
            if filename.exists():
                with open(filename, 'r') as f:
                    return json.load(f)
            return []


def analyze_message_patterns(message_logs):
    """
    로그된 메시지에서 패턴 분석

    Args:
        message_logs: MessageLogger에서 로드한 메시지 로그
    """
    print("\n=== Message Pattern Analysis ===")

    if not message_logs:
        print("No message logs found")
        return

    # DataFrame으로 변환
    df = pd.DataFrame(message_logs)

    # 에피소드별 통계
    print(f"\nTotal messages: {len(df)}")
    print(f"Episodes: {df['episode'].nunique()}")
    print(f"Unique agents: {df['agent_id'].nunique()}")

    # 메시지를 numpy 배열로 변환
    messages = np.array(df['message'].tolist())

    # 시간에 따른 메시지 변화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. 메시지 평균값 변화
    ax = axes[0]
    message_means = np.mean(messages, axis=1)
    ax.plot(message_means, alpha=0.7)
    ax.set_title('Message Mean Over Time')
    ax.set_xlabel('Message Index')
    ax.set_ylabel('Mean Value')

    # 2. 메시지 다양성 변화
    ax = axes[1]
    message_stds = np.std(messages, axis=1)
    ax.plot(message_stds, color='orange', alpha=0.7)
    ax.set_title('Message Diversity Over Time')
    ax.set_xlabel('Message Index')
    ax.set_ylabel('Std Dev')

    # 3. Agent별 메시지 패턴
    ax = axes[2]
    for agent_id in df['agent_id'].unique()[:5]:  # 처음 5개 에이전트만
        agent_messages = df[df['agent_id'] == agent_id]['message'].values
        if len(agent_messages) > 0:
            agent_msg_array = np.array(agent_messages.tolist())
            agent_mean = np.mean(agent_msg_array, axis=0)
            ax.plot(agent_mean, label=f'Agent {agent_id}', alpha=0.7)

    ax.set_title('Message Pattern by Agent')
    ax.set_xlabel('Message Dimension')
    ax.set_ylabel('Average Value')
    ax.legend()

    plt.tight_layout()
    plt.savefig('message_patterns.png', dpi=150)
    print("Message patterns saved to message_patterns.png")

    return messages


if __name__ == "__main__":
    print("Vessel RL Real-time Monitoring Tools")
    print("=" * 60)

    # 옵션 선택
    print("\nSelect mode:")
    print("1. Real-time monitoring")
    print("2. Analyze logged messages")
    print("3. Generate test data")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        # 실시간 모니터링
        monitor = RealtimeMonitor()

        # 테스트 데이터로 시뮬레이션
        print("\nSimulating with test data...")
        for i in range(100):
            reward = np.random.normal(0, 1)
            episode_length = np.random.randint(100, 500)
            messages = np.random.randn(35)
            colregs = np.random.rand(4)

            monitor.update_metrics(reward, episode_length, messages, colregs)
            time.sleep(0.1)

        monitor.start_monitoring()

    elif choice == "2":
        # 로그 분석
        logger = MessageLogger()
        messages = logger.load_messages()

        if messages:
            analyze_message_patterns(messages)
        else:
            print("No message logs found. Run training first!")

    elif choice == "3":
        # 테스트 데이터 생성
        logger = MessageLogger()

        print("\nGenerating test message logs...")
        for ep in range(5):
            logger.new_episode()
            for _ in range(50):
                agent_id = np.random.randint(0, 4)
                message = np.random.randn(35)
                timestamp = time.time()
                logger.log_message(agent_id, message, timestamp)

        logger.save_buffer()
        print("Test data generated!")