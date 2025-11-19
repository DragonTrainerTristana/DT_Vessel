"""
Latent Message Deep Analysis Tool

선박 간 통신에서 학습된 latent message의 의미와 패턴을 심층 분석합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import networkx as nx

class LatentMessageAnalyzer:
    """Latent Message 심층 분석기"""

    def __init__(self):
        # 메시지 구조 정의 (35D)
        self.message_components = {
            "radar_min_dist": (0, 8),      # 8 regions min distance
            "radar_tcpa": (8, 16),         # 8 regions TCPA
            "radar_dcpa": (16, 24),        # 8 regions DCPA
            "vessel_speed": (24, 25),      # 1D speed
            "vessel_heading": (25, 27),    # 2D heading vector
            "vessel_yaw": (27, 28),        # 1D yaw rate
            "goal_direction": (28, 30),    # 2D goal direction
            "goal_distance": (30, 31),     # 1D goal distance
            "colregs_overtaking": (31, 32),    # 1D
            "colregs_headon": (32, 33),        # 1D
            "colregs_crossing_give": (33, 34),  # 1D
            "colregs_crossing_stand": (34, 35)  # 1D
        }

    def analyze_information_flow(self, message_sequence):
        """
        메시지 시퀀스에서 정보 흐름 분석

        Args:
            message_sequence: T × N × 35 (시간 × 에이전트 × 메시지)
        """
        print("\n=== Information Flow Analysis ===")

        T, N, D = message_sequence.shape

        # 1. Mutual Information between agents
        mutual_info = np.zeros((N, N))

        for i in range(N):
            for j in range(i+1, N):
                # 두 에이전트 간의 상호 정보량
                agent_i = message_sequence[:, i, :].flatten()
                agent_j = message_sequence[:, j, :].flatten()

                # Pearson correlation as proxy for MI
                corr = np.corrcoef(agent_i, agent_j)[0, 1]
                mutual_info[i, j] = mutual_info[j, i] = abs(corr)

        print(f"Average mutual information: {np.mean(mutual_info):.3f}")
        print(f"Max mutual information: {np.max(mutual_info):.3f}")

        # 2. Temporal dynamics
        message_changes = np.diff(message_sequence, axis=0)
        change_magnitude = np.linalg.norm(message_changes, axis=2)

        print(f"\nTemporal dynamics:")
        print(f"  Average message change: {np.mean(change_magnitude):.3f}")
        print(f"  Message stability: {1 - np.std(change_magnitude):.3f}")

        return mutual_info, change_magnitude

    def decode_semantic_meaning(self, messages):
        """
        메시지의 의미론적 해석

        Args:
            messages: N × 35 메시지 배열
        """
        print("\n=== Semantic Decoding ===")

        interpretations = {}

        for comp_name, (start, end) in self.message_components.items():
            comp_data = messages[:, start:end]

            # 컴포넌트별 주요 패턴
            mean_val = np.mean(comp_data)
            std_val = np.std(comp_data)
            dominant_idx = np.argmax(np.mean(np.abs(comp_data), axis=0))

            interpretation = {
                "mean": mean_val,
                "std": std_val,
                "activity_level": std_val / (abs(mean_val) + 1e-6),
            }

            # 특정 컴포넌트에 대한 해석
            if "radar" in comp_name:
                if mean_val < 0.3:
                    interpretation["meaning"] = "Close obstacles detected"
                elif mean_val > 0.7:
                    interpretation["meaning"] = "Clear path"
                else:
                    interpretation["meaning"] = "Moderate obstacle density"

            elif "colregs" in comp_name:
                if mean_val > 0.5:
                    situation = comp_name.split("_")[1]
                    interpretation["meaning"] = f"Frequent {situation} situations"
                else:
                    interpretation["meaning"] = "Low frequency situation"

            elif comp_name == "vessel_speed":
                interpretation["meaning"] = f"Average speed: {mean_val*100:.1f}%"

            elif comp_name == "goal_distance":
                if mean_val < 0.3:
                    interpretation["meaning"] = "Close to goal"
                elif mean_val > 0.7:
                    interpretation["meaning"] = "Far from goal"
                else:
                    interpretation["meaning"] = "Medium distance to goal"

            interpretations[comp_name] = interpretation

        # Print interpretations
        for comp, interp in interpretations.items():
            print(f"\n{comp.upper()}:")
            print(f"  Activity: {interp['activity_level']:.2f}")
            if 'meaning' in interp:
                print(f"  Interpretation: {interp['meaning']}")

        return interpretations

    def find_communication_clusters(self, messages):
        """
        메시지 클러스터링으로 통신 패턴 발견

        Args:
            messages: N × 35 메시지 배열
        """
        print("\n=== Communication Pattern Clustering ===")

        # 표준화
        scaler = StandardScaler()
        messages_scaled = scaler.fit_transform(messages)

        # 1. K-means clustering
        n_clusters = min(5, len(messages) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(messages_scaled)

        # 2. DBSCAN for anomaly detection
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(messages_scaled)

        n_anomalies = np.sum(dbscan_labels == -1)

        print(f"K-means clusters found: {n_clusters}")
        print(f"Anomalous messages (DBSCAN): {n_anomalies} ({n_anomalies/len(messages)*100:.1f}%)")

        # 클러스터별 특성
        cluster_characteristics = {}
        for i in range(n_clusters):
            cluster_msgs = messages[kmeans_labels == i]
            if len(cluster_msgs) > 0:
                # 주요 특성 추출
                radar_activity = np.mean(cluster_msgs[:, :24])
                colregs_focus = np.argmax(np.mean(cluster_msgs[:, 31:35], axis=0))
                speed_avg = np.mean(cluster_msgs[:, 24])

                cluster_characteristics[i] = {
                    "size": len(cluster_msgs),
                    "radar_activity": radar_activity,
                    "dominant_colregs": ["overtaking", "head-on", "crossing-give", "crossing-stand"][colregs_focus],
                    "avg_speed": speed_avg
                }

        print("\nCluster characteristics:")
        for cluster_id, chars in cluster_characteristics.items():
            print(f"  Cluster {cluster_id} ({chars['size']} messages):")
            print(f"    - Dominant COLREGs: {chars['dominant_colregs']}")
            print(f"    - Avg speed: {chars['avg_speed']*100:.1f}%")
            print(f"    - Radar activity: {chars['radar_activity']:.2f}")

        return kmeans_labels, dbscan_labels, cluster_characteristics

    def analyze_emergence(self, early_messages, late_messages):
        """
        학습 초기와 후기 메시지 비교로 emergent communication 분석

        Args:
            early_messages: 학습 초기 메시지
            late_messages: 학습 후기 메시지
        """
        print("\n=== Emergent Communication Analysis ===")

        # 1. 정보 엔트로피 변화
        early_entropy = stats.entropy(early_messages.flatten() + 1e-10)
        late_entropy = stats.entropy(late_messages.flatten() + 1e-10)

        print(f"Entropy change: {early_entropy:.3f} → {late_entropy:.3f}")
        print(f"Information structuring: {(early_entropy - late_entropy)/early_entropy*100:.1f}%")

        # 2. 차원 축소 비교
        pca = PCA(n_components=10)
        early_pca = pca.fit_transform(early_messages)
        early_var = pca.explained_variance_ratio_

        late_pca = pca.fit_transform(late_messages)
        late_var = pca.explained_variance_ratio_

        print(f"\nDimensionality reduction:")
        print(f"  Early: Top 3 PC explain {np.sum(early_var[:3])*100:.1f}%")
        print(f"  Late: Top 3 PC explain {np.sum(late_var[:3])*100:.1f}%")

        # 3. 메시지 일관성
        early_consistency = 1 - np.mean(np.std(early_messages, axis=0))
        late_consistency = 1 - np.mean(np.std(late_messages, axis=0))

        print(f"\nMessage consistency:")
        print(f"  Early: {early_consistency:.3f}")
        print(f"  Late: {late_consistency:.3f}")
        print(f"  Improvement: {(late_consistency - early_consistency)/early_consistency*100:+.1f}%")

        # 4. 컴포넌트별 진화
        component_evolution = {}
        for comp_name, (start, end) in self.message_components.items():
            early_comp = early_messages[:, start:end]
            late_comp = late_messages[:, start:end]

            # KL divergence로 분포 변화 측정
            early_hist, _ = np.histogram(early_comp.flatten(), bins=20)
            late_hist, _ = np.histogram(late_comp.flatten(), bins=20)

            early_hist = early_hist + 1e-10
            late_hist = late_hist + 1e-10
            early_hist = early_hist / np.sum(early_hist)
            late_hist = late_hist / np.sum(late_hist)

            kl_div = stats.entropy(late_hist, early_hist)
            component_evolution[comp_name] = kl_div

        print("\nComponent evolution (KL divergence):")
        sorted_evolution = sorted(component_evolution.items(), key=lambda x: x[1], reverse=True)
        for comp, kl in sorted_evolution[:5]:
            print(f"  {comp}: {kl:.3f}")

        return component_evolution

    def visualize_latent_space(self, messages, labels=None, save_path="latent_space.png"):
        """
        Latent space 시각화
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Latent Message Space Analysis', fontsize=16)

        # 1. t-SNE visualization
        ax = axes[0, 0]
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(messages[:min(500, len(messages))])

        if labels is not None:
            scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1],
                               c=labels[:len(tsne_result)], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6)

        ax.set_title('t-SNE Projection')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

        # 2. Component importance
        ax = axes[0, 1]
        component_std = []
        component_names = []

        for comp_name, (start, end) in self.message_components.items():
            comp_data = messages[:, start:end]
            component_std.append(np.mean(np.std(comp_data, axis=0)))
            component_names.append(comp_name.replace('_', '\n'))

        ax.barh(component_names, component_std)
        ax.set_title('Component Activity Levels')
        ax.set_xlabel('Average Std Dev')

        # 3. Message correlation matrix
        ax = axes[0, 2]
        corr_matrix = np.corrcoef(messages.T)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title('Message Dimension Correlations')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Dimension')
        plt.colorbar(im, ax=ax)

        # 4. Radar pattern visualization
        ax = axes[1, 0]
        radar_data = messages[:, :24].reshape(-1, 8, 3)
        mean_radar = np.mean(radar_data, axis=0)

        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        for i, param in enumerate(['Min Dist', 'TCPA', 'DCPA']):
            values = np.concatenate([mean_radar[:, i], [mean_radar[0, i]]])
            ax.plot(angles, values, 'o-', label=param)

        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.legend(loc='upper right')
        ax.set_title('Average Radar Pattern')

        # 5. COLREGs distribution
        ax = axes[1, 1]
        colregs_data = messages[:, 31:35]
        colregs_names = ['Overtaking', 'Head-on', 'Cross-Give', 'Cross-Stand']

        positions = np.arange(len(colregs_names))
        box_data = [colregs_data[:, i] for i in range(4)]

        bp = ax.boxplot(box_data, labels=colregs_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax.set_title('COLREGs Weight Distribution')
        ax.set_ylabel('Weight')
        ax.tick_params(axis='x', rotation=45)

        # 6. Information flow network
        ax = axes[1, 2]
        if len(messages) > 10:
            # Calculate pairwise distances
            distances = pdist(messages[:20], 'euclidean')
            dist_matrix = squareform(distances)

            # Create network
            G = nx.Graph()
            threshold = np.median(distances)

            for i in range(min(20, len(messages))):
                G.add_node(i)
                for j in range(i+1, min(20, len(messages))):
                    if dist_matrix[i, j] < threshold:
                        G.add_edge(i, j, weight=1/dist_matrix[i, j])

            pos = nx.spring_layout(G)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, alpha=0.7)
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
            ax.set_title('Message Similarity Network')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nVisualization saved to {save_path}")

        return fig

    def generate_analysis_report(self, messages, save_path="latent_analysis_report.md"):
        """
        Markdown 형식의 상세 분석 리포트 생성
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("# Latent Message Analysis Report\n\n")

            f.write("## Executive Summary\n\n")
            f.write(f"- Total messages analyzed: {len(messages)}\n")
            f.write(f"- Message dimensionality: {messages.shape[1]}\n")
            f.write(f"- Average message norm: {np.mean(np.linalg.norm(messages, axis=1)):.3f}\n\n")

            # Semantic analysis
            f.write("## Semantic Analysis\n\n")
            interpretations = self.decode_semantic_meaning(messages)

            for comp, interp in interpretations.items():
                f.write(f"### {comp.replace('_', ' ').title()}\n")
                f.write(f"- Activity level: {interp['activity_level']:.2f}\n")
                if 'meaning' in interp:
                    f.write(f"- Interpretation: {interp['meaning']}\n")
                f.write("\n")

            # Clustering analysis
            f.write("## Communication Patterns\n\n")
            kmeans_labels, dbscan_labels, clusters = self.find_communication_clusters(messages)

            for cluster_id, chars in clusters.items():
                f.write(f"### Cluster {cluster_id}\n")
                f.write(f"- Size: {chars['size']} messages\n")
                f.write(f"- Dominant situation: {chars['dominant_colregs']}\n")
                f.write(f"- Average speed: {chars['avg_speed']*100:.1f}%\n")
                f.write(f"- Radar activity: {chars['radar_activity']:.2f}\n\n")

            # Recommendations
            f.write("## Analysis Recommendations\n\n")
            f.write("1. **Message Efficiency**: ")
            zero_ratio = np.sum(np.abs(messages) < 0.01) / messages.size
            if zero_ratio > 0.3:
                f.write("High zero-value ratio suggests potential for message compression\n")
            else:
                f.write("Messages are efficiently utilizing available bandwidth\n")

            f.write("2. **Emergent Patterns**: ")
            pca = PCA(n_components=5)
            pca.fit(messages)
            if np.sum(pca.explained_variance_ratio_) > 0.8:
                f.write("Strong low-dimensional structure detected - agents have learned efficient encoding\n")
            else:
                f.write("High dimensional complexity - consider longer training for better convergence\n")

            f.write("3. **Safety Compliance**: ")
            colregs_weights = messages[:, 31:35]
            high_risk = np.sum(np.max(colregs_weights, axis=1) > 0.7) / len(messages)
            if high_risk > 0.2:
                f.write(f"High risk situations frequent ({high_risk*100:.1f}%) - review reward function\n")
            else:
                f.write(f"Good safety performance with {high_risk*100:.1f}% high-risk situations\n")

        print(f"\nAnalysis report saved to {save_path}")


def run_comprehensive_analysis():
    """전체 분석 실행"""
    print("=" * 60)
    print("Comprehensive Latent Message Analysis")
    print("=" * 60)

    analyzer = LatentMessageAnalyzer()

    # 테스트 데이터 생성 (실제로는 Unity에서 로드)
    print("\nGenerating synthetic data for demonstration...")
    np.random.seed(42)

    # 학습 초기 메시지
    early_messages = np.random.randn(100, 35) * 0.5

    # 학습 후기 메시지 (더 구조화된 패턴)
    late_messages = np.random.randn(100, 35) * 0.3

    # Add structure to late messages
    late_messages[:, :8] = np.random.exponential(0.3, (100, 8))  # Radar distances
    late_messages[:, 31:35] = np.random.dirichlet([1, 1, 1, 1], 100)  # COLREGs

    # 시간 시퀀스 생성
    T, N = 50, 4  # 50 timesteps, 4 agents
    message_sequence = np.random.randn(T, N, 35) * 0.4

    print("\n1. Analyzing information flow...")
    mutual_info, changes = analyzer.analyze_information_flow(message_sequence)

    print("\n2. Decoding semantic meaning...")
    interpretations = analyzer.decode_semantic_meaning(late_messages)

    print("\n3. Finding communication clusters...")
    kmeans_labels, dbscan_labels, clusters = analyzer.find_communication_clusters(late_messages)

    print("\n4. Analyzing emergent communication...")
    evolution = analyzer.analyze_emergence(early_messages, late_messages)

    print("\n5. Visualizing latent space...")
    analyzer.visualize_latent_space(late_messages, kmeans_labels)

    print("\n6. Generating analysis report...")
    analyzer.generate_analysis_report(late_messages)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("\nGenerated files:")
    print("  - latent_space.png : Visual analysis of message space")
    print("  - latent_analysis_report.md : Detailed analysis report")


if __name__ == "__main__":
    run_comprehensive_analysis()