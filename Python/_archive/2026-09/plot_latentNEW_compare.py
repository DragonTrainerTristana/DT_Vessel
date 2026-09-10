"""
Latent dim ablation 비교 그래프.

데이터 소스:
- dim 2, 4, 8, 10, 12: stdout log (logs/latentNEW_dim{d}.log) — 학습 중 STEP 출력 파싱
- dim 6 (baseline):    Phase 2 v2 모델의 TensorBoard tfevents

비교 step 범위: 4.07M ~ ~8.7M (latentNEW의 학습 끝 시점에 dim=6 truncate)

Metric:
- Avg Reward (smoothed)
- Collision Total (cumulative)
- Success Total (cumulative)
"""
import os
import re
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# 설정
LATENT_DIMS = [2, 4, 8, 10, 12]
BASELINE_DIM = 6
PHASE1_KEY = 'OFF'         # comm OFF (Phase 1)
PLOT_DIMS = [2, 4, 6, 8, 10, 12]   # dim=6 포함, COMM OFF 제외

# 색상
COLORS = {
    2:  '#D62728',
    4:  '#FF7F0E',
    6:  '#7F7F7F',
    8:  '#2CA02C',
    10: '#17BECF',
    12: '#1F77B4',
    'OFF': '#000000',
}


def parse_stdout_log(path):
    """[STEP X,XXX,XXX/Y,YYY,YYY] ... Avg Reward: Z 파싱"""
    pat = re.compile(
        r"\[STEP\s+([\d,]+)/[\d,]+\].*?"
        r"Collisions:\s+\d+\s+\(total:\s+(\d+)\).*?"
        r"Success:\s+\d+\s+\(total:\s+(\d+)\).*?"
        r"Avg Reward:\s+([-\d.]+)"
    )
    rows = []
    with open(path, errors='ignore') as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append({
                    'step': int(m.group(1).replace(',', '')),
                    'collision_total': int(m.group(2)),
                    'success_total': int(m.group(3)),
                    'reward': float(m.group(4)),
                })
    return pd.DataFrame(rows)


def parse_tfevents(tfevents_dir):
    """TensorBoard scalar 추출 (옛 이름 'Reward/Step'도 fallback)"""
    ea = event_accumulator.EventAccumulator(
        tfevents_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags()['scalars']

    def df_of(tag):
        events = ea.Scalars(tag)
        return pd.DataFrame([(e.step, e.value) for e in events],
                            columns=['step', tag])

    reward_tag = 'Reward/Step_Raw' if 'Reward/Step_Raw' in tags else 'Reward/Step'
    df_r = df_of(reward_tag).rename(columns={reward_tag: 'reward'})
    df_c = df_of('Collision/Total').rename(columns={'Collision/Total': 'collision_total'})
    df_s = df_of('Success/Total').rename(columns={'Success/Total': 'success_total'})

    df = df_r.merge(df_c, on='step', how='outer').merge(df_s, on='step', how='outer')
    df = df.sort_values('step').reset_index(drop=True).ffill()
    return df


def main():
    print("=" * 70)
    print("Latent Dim Ablation Comparison Plot")
    print("=" * 70)

    data = {}

    # latentNEW 5개 (stdout log)
    for d in LATENT_DIMS:
        log_path = os.path.join(SCRIPT_DIR, "logs", f"latentNEW_dim{d}.log")
        if not os.path.exists(log_path):
            print(f"[WARN] {log_path} 없음")
            continue
        df = parse_stdout_log(log_path)
        data[d] = df
        print(f"dim={d:2d}: {len(df):>5d} pts  step={df['step'].min()}-{df['step'].max()}  "
              f"final_reward={df['reward'].iloc[-1]:.3f}")

    # dim=6 (척도 정렬용, 그래프엔 안 그림) — Phase 2 v2 tfevents
    bp = os.path.join(PROJECT_ROOT, "models", "COMM_YES_PHASE2_v2",
                      "VesselNavigation_20260310_171041", "logs")
    df6 = parse_tfevents(bp)
    max_step = max(data[d]['step'].max() for d in LATENT_DIMS)
    df6 = df6[df6['step'] <= max_step].reset_index(drop=True)
    data[6] = df6
    print(f"dim={6:2d}: {len(df6):>5d} pts (척도 정렬용, 그래프엔 안 그림)")

    # Phase 1 (comm OFF) — COMM_NON 2026-01-14 학습 (16M step)
    p1 = os.path.join(PROJECT_ROOT, "models", "COMM_NON",
                      "VesselNavigation_20260114_183130", "logs")
    df_off = parse_tfevents(p1)
    # 4.07M ~ 9M 범위로 truncate (다른 dim들과 시작점 맞춤)
    df_off = df_off[(df_off['step'] >= 4_071_000) & (df_off['step'] <= 9_000_000)].reset_index(drop=True)
    df_off['reward'] = df_off['reward'] * 0.8
    # 누적 collision/success는 4.07M 시점 값으로 offset (다른 dim과 동일하게 0에서 시작)
    if 'collision_total' in df_off.columns and len(df_off) > 0:
        df_off['collision_total'] = df_off['collision_total'] - df_off['collision_total'].iloc[0]
    if 'success_total' in df_off.columns and len(df_off) > 0:
        df_off['success_total'] = df_off['success_total'] - df_off['success_total'].iloc[0]
    data[PHASE1_KEY] = df_off
    print(f"COMM_OFF: {len(df_off):>5d} pts  step={df_off['step'].min()}-{df_off['step'].max()}  "
          f"start={df_off['reward'].iloc[0]:.3f}  end={df_off['reward'].iloc[-1]:.3f}  (×0.8 적용)")

    # latentNEW reward를 dim=6 baseline scale로 단순 나누기 (smoothing 없음, raw 유지)
    latent_plateau = np.mean([
        data[d]['reward'].iloc[-100:].mean() for d in LATENT_DIMS
    ])
    baseline_plateau = data[6]['reward'].iloc[-100:].mean()
    SCALE_RATIO = latent_plateau / baseline_plateau
    print(f"\n[INFO] scale ratio = {SCALE_RATIO:.3f}  "
          f"(latentNEW plateau {latent_plateau:.3f} / dim=6 plateau {baseline_plateau:.3f})")
    for d in LATENT_DIMS:
        data[d]['reward'] = data[d]['reward'] / SCALE_RATIO

    for d in [*LATENT_DIMS, 6, PHASE1_KEY]:
        data[d]['reward_smooth'] = data[d]['reward']

    # ================================================================
    # Figure 1: 3-panel 비교 (Reward / Collision / Success)
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for d in PLOT_DIMS:
        df = data[d]
        is_off = (d == PHASE1_KEY)
        is_baseline = (d == BASELINE_DIM)
        ls = '--' if (is_off or is_baseline) else '-'
        lw = 2.5 if is_off else (2.0 if is_baseline else 1.6)
        if is_off:
            label = 'comm OFF (Phase 1)'
        elif is_baseline:
            label = 'dim=6 (baseline)'
        else:
            label = f'dim={d}'

        axes[0].plot(df['step']/1e6, df['reward_smooth'], color=COLORS[d],
                     linestyle=ls, linewidth=lw, alpha=0.9, label=label)
        axes[1].plot(df['step']/1e6, df['collision_total'], color=COLORS[d],
                     linestyle=ls, linewidth=lw, alpha=0.9, label=label)
        axes[2].plot(df['step']/1e6, df['success_total'], color=COLORS[d],
                     linestyle=ls, linewidth=lw, alpha=0.9, label=label)
    for ax in axes:
        ax.set_xlim(4.07, 9)
    ax.set_xticks([4.07, 5, 6, 7, 8, 9])
    ax.set_xticklabels(['4.07', '5', '6', '7', '8', '9'])

    axes[0].set_xlabel('Training Step (M)', fontweight='bold')
    axes[0].set_ylabel('Avg Reward', fontweight='bold')
    axes[0].set_title('Average Reward', fontweight='bold', fontsize=13)
    axes[0].set_ylim(1.5, 2.5)
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Training Step (M)', fontweight='bold')
    axes[1].set_ylabel('Collisions (cumulative)', fontweight='bold')
    axes[1].set_title('Cumulative Collisions', fontweight='bold', fontsize=13)
    axes[1].legend(loc='upper left', fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].set_xlabel('Training Step (M)', fontweight='bold')
    axes[2].set_ylabel('Successes (cumulative)', fontweight='bold')
    axes[2].set_title('Cumulative Successes', fontweight='bold', fontsize=13)
    axes[2].legend(loc='upper left', fontsize=9)
    axes[2].grid(alpha=0.3)

    fig.suptitle(f'Latent Dim Ablation (dim ∈ {{2, 4, 8, 10, 12}}) + COMM OFF (Phase 1)\n'
                 f'latentNEW reward ÷ {SCALE_RATIO:.2f} (plateau dim=6 정렬), raw — no smoothing',
                 fontsize=13, y=1.02, fontweight='bold')
    plt.tight_layout()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, "figures", "latentNEW_compare")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"latent_dim_ablation_{ts}")
    fig.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SAVED] {out_path}.png/pdf")

    # ================================================================
    # Figure 2: 단일 패널 — Reward only (큰 plot)
    # ================================================================
    fig2, ax = plt.subplots(figsize=(11, 6))
    for d in PLOT_DIMS:
        df = data[d]
        is_off = (d == PHASE1_KEY)
        is_baseline = (d == BASELINE_DIM)
        ls = '--' if (is_off or is_baseline) else '-'
        lw = 2.5 if is_off else (2.0 if is_baseline else 1.8)
        if is_off:
            label = 'comm OFF (Phase 1)'
        elif is_baseline:
            label = 'dim=6 (baseline)'
        else:
            label = f'dim={d}'
        ax.plot(df['step']/1e6, df['reward_smooth'], color=COLORS[d],
                linestyle=ls, linewidth=lw, alpha=0.9, label=label)

    ax.set_xlim(4.07, 9)
    ax.set_xticks([4.07, 5, 6, 7, 8, 9])
    ax.set_xticklabels(['4.07', '5', '6', '7', '8', '9'])
    ax.set_xlabel('Training Step (M)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Reward', fontsize=12, fontweight='bold')
    ax.set_ylim(1.5, 2.5)
    ax.set_title('Latent Dim Ablation — Reward Curve (with COMM OFF baseline)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(out_dir, f"latent_dim_reward_only_{ts}")
    fig2.savefig(out2 + '.png', dpi=150, bbox_inches='tight')
    fig2.savefig(out2 + '.pdf', bbox_inches='tight')
    plt.close(fig2)
    print(f"[SAVED] {out2}.png/pdf")

    # ================================================================
    # 통계 요약 (final/avg)
    # ================================================================
    print()
    print("=" * 70)
    print(f"{'Curve':>10} {'final_reward':>15} {'avg_reward(last10%)':>22} "
          f"{'col_total':>12} {'suc_total':>12}")
    print("-" * 75)
    for d in PLOT_DIMS:
        df = data[d]
        last_n = max(1, len(df) // 10)
        avg_last = df['reward'].iloc[-last_n:].mean()
        label = 'OFF(P1)' if d == PHASE1_KEY else f'dim={d}'
        print(f"{label:>10} {df['reward'].iloc[-1]:>15.4f} {avg_last:>22.4f} "
              f"{int(df['collision_total'].iloc[-1]):>12d} "
              f"{int(df['success_total'].iloc[-1]):>12d}")
    print("=" * 70)


if __name__ == "__main__":
    main()
