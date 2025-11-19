"""
Unity ML-Agents Vessel Navigation 강화학습 분석 도구

이 스크립트는 학습된 선박 에이전트의 행동과 메시지 패싱을 분석합니다.
특히 multi-agent 환경에서의 latent message 패턴을 시각화합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import os
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path

# 학습 결과 경로
RESULTS_PATH = "results/vessel_training_002"  # ML-Agents 학습 결과 폴더

class VesselRLAnalyzer:
    """선박 강화학습 결과 분석기"""

    def __init__(self, run_id="vessel_training_002"):
        self.run_id = run_id
        self.results_path = Path(f"results/{run_id}")

        # 메시지 구조 정의
        self.message_structure = {
            "compressed_radar": (0, 24),    # 8 regions × 3
            "vessel_state": (24, 28),       # 4D
            "goal_info": (28, 31),          # 3D
            "fuzzy_colregs": (31, 35)       # 4D
        }

    def load_tensorboard_data(self):
        """TensorBoard 로그에서 학습 메트릭 로드"""
        tb_path = self.results_path / "VesselBehavior"

        if not tb_path.exists():
            print(f"TensorBoard logs not found at {tb_path}")
            return None

        ea = event_accumulator.EventAccumulator(str(tb_path))
        ea.Reload()

        metrics = {}

        # 주요 메트릭 추출
        scalar_tags = ea.Tags()['scalars']
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            metrics[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }

        return metrics

    def analyze_latent_messages(self, message_data):
        """
        Latent message 패턴 분석

        Args:
            message_data: 에이전트들 간 교환된 메시지 데이터 (N × 35)
        """
        print("\n=== Latent Message Analysis ===")

        # 1. 메시지 컴포넌트별 분석
        components_analysis = {}

        for comp_name, (start, end) in self.message_structure.items():
            comp_data = message_data[:, start:end]

            # 통계 분석
            mean_vals = np.mean(comp_data, axis=0)
            std_vals = np.std(comp_data, axis=0)

            components_analysis[comp_name] = {
                "mean": mean_vals,
                "std": std_vals,
                "variance_ratio": np.var(comp_data, axis=0) / np.var(message_data)
            }

            print(f"\n{comp_name.upper()}:")
            print(f"  Mean range: [{np.min(mean_vals):.3f}, {np.max(mean_vals):.3f}]")
            print(f"  Std range: [{np.min(std_vals):.3f}, {np.max(std_vals):.3f}]")
            print(f"  Information content: {np.mean(std_vals):.3f}")

        # 2. PCA 분석 - 메시지의 주요 패턴
        pca = PCA(n_components=10)
        pca_result = pca.fit_transform(message_data)

        print(f"\n=== PCA Analysis ===")
        print(f"Explained variance ratio (top 5): {pca.explained_variance_ratio_[:5]}")
        print(f"Total variance explained (10 components): {np.sum(pca.explained_variance_ratio_):.2%}")

        # 3. 메시지 클러스터링 패턴
        return components_analysis, pca_result

    def visualize_communication_patterns(self, message_data, save_path="communication_analysis.png"):
        """통신 패턴 시각화"""

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Vessel Communication Pattern Analysis', fontsize=16)

        # 1. 메시지 컴포넌트 히트맵
        ax = axes[0, 0]
        sns.heatmap(message_data[:100].T, cmap='coolwarm', ax=ax, cbar=True)
        ax.set_title('Message Components Heatmap (100 samples)')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Message Dimension')

        # 2. Radar 정보 분포 (8 regions)
        ax = axes[0, 1]
        radar_data = message_data[:, :24].reshape(-1, 8, 3)
        mean_radar = np.mean(radar_data, axis=0)

        # Radar plot
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        for i, param_name in enumerate(['Min Dist', 'TCPA', 'DCPA']):
            values = np.concatenate([mean_radar[:, i], [mean_radar[0, i]]])
            ax.plot(angles, values, 'o-', label=param_name)

        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.set_title('Average Radar Pattern (8 regions)')
        ax.legend()

        # 3. COLREGs 상황 분포
        ax = axes[0, 2]
        colregs_data = message_data[:, 31:35]
        colregs_names = ['Overtaking', 'Head-on', 'Crossing-Give', 'Crossing-Stand']

        mean_colregs = np.mean(colregs_data, axis=0)
        ax.bar(colregs_names, mean_colregs)
        ax.set_title('Average COLREGs Situation Weights')
        ax.set_ylabel('Weight')
        ax.tick_params(axis='x', rotation=45)

        # 4. PCA 2D 투영
        ax = axes[1, 0]
        pca = PCA(n_components=2)
        pca_2d = pca.fit_transform(message_data)

        scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1],
                           c=np.arange(len(pca_2d)), cmap='viridis', alpha=0.6)
        ax.set_title(f'PCA Projection (explained: {sum(pca.explained_variance_ratio_):.1%})')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax, label='Time step')

        # 5. 메시지 컴포넌트 상관관계
        ax = axes[1, 1]
        correlation = np.corrcoef(message_data.T)
        sns.heatmap(correlation[:10, :10], cmap='coolwarm', center=0,
                   square=True, ax=ax, vmin=-1, vmax=1)
        ax.set_title('Message Dimension Correlations (first 10)')

        # 6. Vessel State 시계열
        ax = axes[1, 2]
        vessel_state = message_data[:, 24:28]
        time_window = min(200, len(vessel_state))

        for i, label in enumerate(['Speed', 'Heading X', 'Heading Z', 'Yaw Rate']):
            ax.plot(vessel_state[:time_window, i], label=label, alpha=0.7)

        ax.set_title('Vessel State Time Series')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Normalized Value')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nVisualization saved to {save_path}")

        return fig

    def analyze_colregs_compliance(self, episode_data):
        """COLREGs 규정 준수 분석"""

        print("\n=== COLREGs Compliance Analysis ===")

        # Fuzzy COLREGs weights 추출 (마지막 4D)
        colregs_weights = episode_data[:, 31:35]

        # 각 상황별 발생 빈도
        situations = ['Overtaking', 'Head-on', 'Crossing-Give', 'Crossing-Stand']

        # 주요 상황 판별 (가장 높은 weight)
        dominant_situations = np.argmax(colregs_weights, axis=1)

        print("\nSituation Frequency:")
        for i, situation in enumerate(situations):
            count = np.sum(dominant_situations == i)
            percentage = (count / len(dominant_situations)) * 100
            print(f"  {situation}: {count} ({percentage:.1f}%)")

        # 위험도 변화
        risk_levels = np.max(colregs_weights, axis=1)

        print(f"\nRisk Level Statistics:")
        print(f"  Mean: {np.mean(risk_levels):.3f}")
        print(f"  Max: {np.max(risk_levels):.3f}")
        print(f"  Episodes with high risk (>0.7): {np.sum(risk_levels > 0.7)}")

        return colregs_weights, risk_levels

    def generate_analysis_report(self, message_data, save_path="analysis_report.txt"):
        """종합 분석 리포트 생성"""

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Vessel Multi-Agent RL Analysis Report\n")
            f.write("=" * 60 + "\n\n")

            # 1. 데이터 개요
            f.write("1. DATA OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total samples: {len(message_data)}\n")
            f.write(f"Message dimensions: {message_data.shape[1]}\n")
            f.write(f"Run ID: {self.run_id}\n\n")

            # 2. 메시지 구조 분석
            f.write("2. MESSAGE STRUCTURE ANALYSIS\n")
            f.write("-" * 40 + "\n")

            for comp_name, (start, end) in self.message_structure.items():
                comp_data = message_data[:, start:end]
                f.write(f"\n{comp_name.upper()} [{end-start}D]:\n")
                f.write(f"  Mean: {np.mean(comp_data):.4f}\n")
                f.write(f"  Std: {np.std(comp_data):.4f}\n")
                f.write(f"  Min: {np.min(comp_data):.4f}\n")
                f.write(f"  Max: {np.max(comp_data):.4f}\n")

                # 정보 엔트로피 계산
                if np.std(comp_data) > 0:
                    entropy = -np.sum(comp_data * np.log(comp_data + 1e-10)) / len(comp_data)
                    f.write(f"  Information entropy: {entropy:.4f}\n")

            # 3. 통신 효율성
            f.write("\n3. COMMUNICATION EFFICIENCY\n")
            f.write("-" * 40 + "\n")

            # Zero-value ratio (통신 효율성 지표)
            zero_ratio = np.sum(np.abs(message_data) < 0.01) / message_data.size
            f.write(f"Near-zero value ratio: {zero_ratio:.2%}\n")

            # Message diversity
            pca = PCA(n_components=min(10, message_data.shape[1]))
            pca.fit(message_data)
            effective_dims = np.sum(pca.explained_variance_ratio_ > 0.01)
            f.write(f"Effective dimensions (PCA): {effective_dims}\n")
            f.write(f"Top 5 PC variance: {pca.explained_variance_ratio_[:5]}\n")

            print(f"\nAnalysis report saved to {save_path}")

    def run_full_analysis(self, message_data=None):
        """전체 분석 실행"""

        # 테스트 데이터 생성 (실제로는 Unity에서 로그된 데이터 사용)
        if message_data is None:
            print("Generating synthetic data for demonstration...")
            message_data = self.generate_synthetic_data(1000)

        # 1. TensorBoard 데이터 로드
        tb_metrics = self.load_tensorboard_data()

        # 2. Latent message 분석
        components, pca_result = self.analyze_latent_messages(message_data)

        # 3. 시각화
        self.visualize_communication_patterns(message_data)

        # 4. COLREGs 준수 분석
        colregs, risks = self.analyze_colregs_compliance(message_data)

        # 5. 리포트 생성
        self.generate_analysis_report(message_data)

        return components, pca_result, colregs, risks

    def generate_synthetic_data(self, n_samples):
        """테스트용 합성 데이터 생성"""
        np.random.seed(42)

        data = np.zeros((n_samples, 35))

        # Compressed radar (24D)
        for i in range(8):
            # Min distance
            data[:, i*3] = np.random.exponential(0.3, n_samples)
            # TCPA
            data[:, i*3 + 1] = np.random.normal(0.5, 0.2, n_samples)
            # DCPA
            data[:, i*3 + 2] = np.random.normal(0.5, 0.15, n_samples)

        # Vessel state (4D)
        data[:, 24] = np.random.uniform(0, 1, n_samples)  # Speed
        data[:, 25:27] = np.random.normal(0, 0.5, (n_samples, 2))  # Heading
        data[:, 27] = np.random.normal(0, 0.2, n_samples)  # Yaw rate

        # Goal info (3D)
        data[:, 28:30] = np.random.normal(0, 0.7, (n_samples, 2))  # Direction
        data[:, 30] = np.random.uniform(0, 1, n_samples)  # Distance

        # Fuzzy COLREGs (4D)
        colregs = np.random.dirichlet([1, 1, 1, 1], n_samples)
        data[:, 31:35] = colregs

        # Clip values
        data = np.clip(data, -1, 1)

        return data


if __name__ == "__main__":
    print("Starting Vessel RL Analysis...")
    print("=" * 60)

    analyzer = VesselRLAnalyzer()

    # 전체 분석 실행
    components, pca_result, colregs, risks = analyzer.run_full_analysis()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("\nGenerated files:")
    print("  - communication_analysis.png : Visual analysis")
    print("  - analysis_report.txt : Detailed report")
    print("\nNext steps:")
    print("1. Run actual training with Unity")
    print("2. Export message logs from Unity")
    print("3. Re-run this analysis with real data")