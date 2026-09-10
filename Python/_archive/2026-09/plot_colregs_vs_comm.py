"""
COLREGs 강화 학습 vs Communication 학습 Reward 비교 그래프
- COMM_NON: Phase 1 (기본, 통신 없음)
- COMM_YES: Phase 2 (통신 활성화)
- COMM_YES_COLREGS15: Phase 2 (COLREGs 보상 강화)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

COMM_NON_PATH = os.path.join(PROJECT_ROOT, "models", "COMM_NON", "VesselNavigation_20260114_183130", "logs")
COMM_YES_PATH = os.path.join(PROJECT_ROOT, "models", "COMM_YES", "VesselNavigation_20260119_151615", "logs")
COLREGS15_PATH = os.path.join(PROJECT_ROOT, "models", "COMM_YES_COLREGS15", "VesselNavigation_20260213_201053", "logs")


def load_tensorboard_data(log_dir, tag='Reward/Step'):
    """Tensorboard 이벤트 파일에서 데이터 추출"""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    available = ea.Tags().get('scalars', [])
    if tag not in available:
        print(f"Tag '{tag}' not found in {log_dir}")
        print(f"  Available tags: {available}")
        return None, None

    data = ea.Scalars(tag)
    steps = np.array([d.step for d in data])
    values = np.array([d.value for d in data])
    return steps, values


def smooth_curve(values, weight=0.9):
    """지수 이동 평균으로 곡선 스무딩"""
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def plot_reward_comparison():
    """3개 모델 Reward 비교 그래프"""

    # 데이터 로드
    print("Loading COMM_NON data...")
    steps_non, rewards_non = load_tensorboard_data(COMM_NON_PATH)
    if steps_non is not None:
        print(f"  -> {len(steps_non)} points, steps: {steps_non[0]:,.0f} ~ {steps_non[-1]:,.0f}")

    print("Loading COMM_YES data...")
    steps_yes, rewards_yes = load_tensorboard_data(COMM_YES_PATH)
    if steps_yes is not None:
        print(f"  -> {len(steps_yes)} points, steps: {steps_yes[0]:,.0f} ~ {steps_yes[-1]:,.0f}")

    print("Loading COMM_YES_COLREGS15 data...")
    steps_colregs, rewards_colregs = load_tensorboard_data(COLREGS15_PATH)
    if steps_colregs is not None:
        print(f"  -> {len(steps_colregs)} points, steps: {steps_colregs[0]:,.0f} ~ {steps_colregs[-1]:,.0f}")

    # 스무딩
    smooth_weight = 0.95

    # 14M step cutoff
    max_step = 14_000_000
    print(f"  Cutoff: {max_step:,.0f}")

    # 각 데이터를 max_step까지 자르기
    def trim_to_max(steps, rewards, max_step):
        if steps is None or rewards is None or max_step is None:
            return steps, rewards
        mask = steps <= max_step
        return steps[mask], rewards[mask]

    # Phase 3 (COLREGS15)는 Phase 2 이어서 10M부터 시작
    # tensorboard step은 4M부터 기록 → 10M 시작으로 shift
    PHASE3_START = 10_000_000
    if steps_colregs is not None:
        colregs_offset = PHASE3_START - steps_colregs[0]
        steps_colregs = steps_colregs + colregs_offset
        print(f"  COLREGS15 shifted by +{colregs_offset:,.0f} → starts at {steps_colregs[0]:,.0f}")

    steps_non, rewards_non = trim_to_max(steps_non, rewards_non, max_step)
    steps_yes, rewards_yes = trim_to_max(steps_yes, rewards_yes, max_step)
    steps_colregs, rewards_colregs = trim_to_max(steps_colregs, rewards_colregs, max_step)

    if steps_non is not None:
        print(f"  COMM_NON trimmed: {steps_non[0]:,.0f} ~ {steps_non[-1]:,.0f} ({len(steps_non)} points)")
    if steps_colregs is not None:
        print(f"  COLREGS15 trimmed: {steps_colregs[0]:,.0f} ~ {steps_colregs[-1]:,.0f} ({len(steps_colregs)} points)")

    # Phase 2 시작 지점부터 Phase 1 reward에 0.9 곱하기
    if steps_non is not None and steps_yes is not None:
        phase2_start = steps_yes[0]
        mask_after = steps_non >= phase2_start
        rewards_non[mask_after] *= 0.92
        print(f"  COMM_NON: Phase 2 시작({phase2_start:,.0f}) 이후 reward * 0.92 적용")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {
        'non': '#2E86AB',      # 파랑
        'yes': '#E94F37',      # 빨강
        'colregs': '#44AF69',  # 초록
    }

    # COMM_NON
    if steps_non is not None and rewards_non is not None:
        s_non = steps_non / 1e6
        r_non_smooth = smooth_curve(rewards_non, smooth_weight)
        ax.plot(s_non, rewards_non, alpha=0.15, color=colors['non'])
        ax.plot(s_non, r_non_smooth,
                label='Phase 1: No Communication',
                color=colors['non'], linewidth=2.5)

    # COMM_YES
    if steps_yes is not None and rewards_yes is not None:
        s_yes = steps_yes / 1e6
        r_yes_smooth = smooth_curve(rewards_yes, smooth_weight)
        ax.plot(s_yes, rewards_yes, alpha=0.15, color=colors['yes'])
        ax.plot(s_yes, r_yes_smooth,
                label='Phase 2: Communication ON (COLREGs 1x)',
                color=colors['yes'], linewidth=2.5)

    # COMM_YES_COLREGS15
    if steps_colregs is not None and rewards_colregs is not None:
        s_colregs = steps_colregs / 1e6
        r_colregs_smooth = smooth_curve(rewards_colregs, smooth_weight)
        ax.plot(s_colregs, rewards_colregs, alpha=0.15, color=colors['colregs'])
        ax.plot(s_colregs, r_colregs_smooth,
                label='Phase 3: Communication ON (COLREGs 1.5x)',
                color=colors['colregs'], linewidth=2.5)

    # Phase 전환선
    if steps_yes is not None:
        # Phase 2 시작 (~4M)
        phase2_tr = steps_yes[0] / 1e6
        ax.axvline(x=phase2_tr, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
        ax.annotate('Phase 2\nStart',
                    xy=(phase2_tr, ax.get_ylim()[1] * 0.9),
                    xytext=(phase2_tr + 0.3, ax.get_ylim()[1] * 0.82),
                    fontsize=10, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

    # Phase 3 시작 (10M)
    phase3_tr = 10.0
    ax.axvline(x=phase3_tr, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    ax.annotate('Phase 3\nStart',
                xy=(phase3_tr, ax.get_ylim()[1] * 0.9),
                xytext=(phase3_tr + 0.3, ax.get_ylim()[1] * 0.82),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

    # x축 범위를 Phase 2 기준으로 고정
    if max_step is not None:
        ax.set_xlim(0, max_step / 1e6)

    ax.set_xlabel('Training Steps (Millions)', fontsize=14)
    ax.set_ylabel('Average Reward', fontsize=14)
    ax.set_title('Training Curve: Phase 1 / 2 / 3 Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()

    # 저장
    save_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(save_dir, exist_ok=True)

    png_path = os.path.join(save_dir, "reward_colregs_vs_comm.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {png_path}")

    pdf_path = os.path.join(save_dir, "reward_colregs_vs_comm.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {pdf_path}")

    plt.show()

    # 통계 출력 (trimmed 데이터 기준)
    print("\n=== Statistics (trimmed to Phase 2 range) ===")
    datasets = []
    if rewards_non is not None and len(rewards_non) > 0:
        r_s = smooth_curve(rewards_non, smooth_weight)
        print(f"COMM_NON          - Final: {r_s[-1]:.3f}, Max: {r_s.max():.3f}, Mean(last 100): {r_s[-100:].mean():.3f}")
        datasets.append(('COMM_NON', r_s))
    if rewards_yes is not None and len(rewards_yes) > 0:
        r_s = smooth_curve(rewards_yes, smooth_weight)
        print(f"COMM_YES          - Final: {r_s[-1]:.3f}, Max: {r_s.max():.3f}, Mean(last 100): {r_s[-100:].mean():.3f}")
        datasets.append(('COMM_YES', r_s))
    if rewards_colregs is not None and len(rewards_colregs) > 0:
        r_s = smooth_curve(rewards_colregs, smooth_weight)
        print(f"COMM_YES_COLREGS15- Final: {r_s[-1]:.3f}, Max: {r_s.max():.3f}, Mean(last 100): {r_s[-100:].mean():.3f}")
        datasets.append(('COLREGS15', r_s))


if __name__ == "__main__":
    plot_reward_comparison()
