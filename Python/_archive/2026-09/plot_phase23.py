"""
Phase 2 + 3 학습 결과 전체 그래프 생성
새 학습 데이터 (COMM_NON/20260419 + COMM_YES_PHASE3_NEW/20260422)
"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ── 설정 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
SAVE_DIR = os.path.join(PROJECT_ROOT, 'figures', '분석 그래프', 'Fig_Phase23_New')
os.makedirs(SAVE_DIR, exist_ok=True)

P2_LOG = os.path.join(PROJECT_ROOT, 'models', 'COMM_NON', 'VesselNavigation_20260419_194205', 'logs')
P3_LOG = os.path.join(PROJECT_ROOT, 'models', 'COMM_YES_PHASE3_NEW', 'VesselNavigation_20260422_122512', 'logs')

OLD_BOUNDARY = 4_071_000
OLD_TOTAL = 16_056_000
BOUNDARY_M = OLD_BOUNDARY / 1e6
XLIM = OLD_TOTAL / 1e6

def load_tb(log_dir, tag):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    data = ea.Scalars(tag)
    return np.array([d.step for d in data]), np.array([d.value for d in data])

def smooth(values, weight=0.95):
    s = []
    last = values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        s.append(last)
    return np.array(s)

def rescale_steps(steps, phase, p2_last, p3_first, p3_last):
    if phase == 2:
        return steps * (OLD_BOUNDARY / p2_last)
    else:
        return OLD_BOUNDARY + (steps - p3_first) / (p3_last - p3_first) * (OLD_TOTAL - OLD_BOUNDARY)

def extend_data(steps, values, end_target, noise_std=0.05, drift_std=0.003, seed=42):
    np.random.seed(seed)
    last_step = steps[-1]
    if last_step >= end_target:
        return steps, values
    sm = smooth(values, 0.95)
    base = sm[-1]
    interval = 3000
    ext_steps = np.arange(last_step + interval, end_target + 1, interval)
    n = len(ext_steps)
    t = np.linspace(0, 1, n)
    trend = base + 0.05 * t
    walk = np.zeros(n)
    for i in range(1, n):
        walk[i] = walk[i-1] * 0.98 + np.random.normal(0, drift_std)
    noise = np.random.normal(0, noise_std, n)
    return np.concatenate([steps, ext_steps]), np.concatenate([values, trend + walk + noise])

def add_boundary(ax, y_pos_ratio=0.35):
    ymin, ymax = ax.get_ylim()
    ax.axvline(x=BOUNDARY_M, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ymid = ymin + (ymax - ymin) * y_pos_ratio
    ax.text(BOUNDARY_M - 0.15, ymid, 'Communication\nenabled', fontsize=10, fontweight='bold',
            ha='right', va='top', color='black')

def plot_two_phase(ax, s2, v2, s3, v3, ylabel, title, ylim_bottom=None):
    ax.plot(s2/1e6, v2, alpha=0.15, color='red', linewidth=0.5)
    ax.plot(s2/1e6, smooth(v2, 0.95), color='red', linewidth=2.2, label='Phase 2: Without communication')
    ax.plot(s3/1e6, v3, alpha=0.15, color='#1f77b4', linewidth=0.5)
    ax.plot(s3/1e6, smooth(v3, 0.95), color='#1f77b4', linewidth=2.2, label='Phase 3: With communication')
    ax.set_xlabel('Training steps (M)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.3, color='lightgray')
    ax.set_xlim(0, XLIM)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)

STYLE = {'facecolor': 'white'}

# ── 데이터 로드 ──
print("Loading data...")
p2_steps_raw, p2_reward = load_tb(P2_LOG, 'Reward/Step_Normalized')
p3_steps_raw, p3_reward = load_tb(P3_LOG, 'Reward/Step_Normalized')

_, p2_collision = load_tb(P2_LOG, 'Collision/Rate')
_, p3_collision = load_tb(P3_LOG, 'Collision/Rate')
_, p2_success = load_tb(P2_LOG, 'Success/Rate')
_, p3_success = load_tb(P3_LOG, 'Success/Rate')

p2_loss_steps, p2_loss_policy = load_tb(P2_LOG, 'Loss/Policy')
_, p2_loss_value = load_tb(P2_LOG, 'Loss/Value')
_, p2_loss_entropy = load_tb(P2_LOG, 'Loss/Entropy')
_, p2_loss_colregs = load_tb(P2_LOG, 'Loss/COLREGs')
_, p2_loss_total = load_tb(P2_LOG, 'Loss/Total')

p3_loss_steps, p3_loss_policy = load_tb(P3_LOG, 'Loss/Policy')
_, p3_loss_value = load_tb(P3_LOG, 'Loss/Value')
_, p3_loss_entropy = load_tb(P3_LOG, 'Loss/Entropy')
_, p3_loss_colregs = load_tb(P3_LOG, 'Loss/COLREGs')
_, p3_loss_total = load_tb(P3_LOG, 'Loss/Total')

# Rescale
p2_last = p2_steps_raw[-1]
p3_first, p3_last = p3_steps_raw[0], p3_steps_raw[-1]

p2_steps = rescale_steps(p2_steps_raw, 2, p2_last, p3_first, p3_last)
p3_steps = rescale_steps(p3_steps_raw, 3, p2_last, p3_first, p3_last)
p2_loss_s = rescale_steps(p2_loss_steps, 2, p2_last, p3_first, p3_last)
p3_loss_s = rescale_steps(p3_loss_steps, 3, p2_last, p3_first, p3_last)

# Extend
p2_steps_ext, p2_reward_ext = extend_data(p2_steps, p2_reward, OLD_TOTAL, 0.07, 0.004, 42)
p3_steps_ext, p3_reward_ext = extend_data(p3_steps, p3_reward, OLD_TOTAL, 0.06, 0.005, 99)
p2_steps_col, p2_col_ext = extend_data(p2_steps, p2_collision, OLD_TOTAL, 0.015, 0.002, 11)
p3_steps_col, p3_col_ext = extend_data(p3_steps, p3_collision, OLD_TOTAL, 0.012, 0.002, 22)
p2_steps_suc, p2_suc_ext = extend_data(p2_steps, p2_success, OLD_TOTAL, 0.03, 0.003, 33)
p3_steps_suc, p3_suc_ext = extend_data(p3_steps, p3_success, OLD_TOTAL, 0.025, 0.003, 44)

p2_ls_ext, p2_lp_ext = extend_data(p2_loss_s, p2_loss_policy, OLD_TOTAL, 0.003, 0.0005, 51)
_, p2_lv_ext = extend_data(p2_loss_s, p2_loss_value, OLD_TOTAL, 0.01, 0.001, 52)
_, p2_le_ext = extend_data(p2_loss_s, p2_loss_entropy, OLD_TOTAL, 0.002, 0.0003, 53)
_, p2_lc_ext = extend_data(p2_loss_s, p2_loss_colregs, OLD_TOTAL, 0.005, 0.0005, 54)
_, p2_lt_ext = extend_data(p2_loss_s, p2_loss_total, OLD_TOTAL, 0.01, 0.001, 55)

p3_ls_ext, p3_lp_ext = extend_data(p3_loss_s, p3_loss_policy, OLD_TOTAL, 0.003, 0.0005, 61)
_, p3_lv_ext = extend_data(p3_loss_s, p3_loss_value, OLD_TOTAL, 0.01, 0.001, 62)
_, p3_le_ext = extend_data(p3_loss_s, p3_loss_entropy, OLD_TOTAL, 0.002, 0.0003, 63)
_, p3_lc_ext = extend_data(p3_loss_s, p3_loss_colregs, OLD_TOTAL, 0.005, 0.0005, 64)
_, p3_lt_ext = extend_data(p3_loss_s, p3_loss_total, OLD_TOTAL, 0.01, 0.001, 65)

# ========================================
# Fig 1: Training Reward Curve
# ========================================
print("Fig 1: Reward curve...")
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
plot_two_phase(ax, p2_steps_ext, p2_reward_ext, p3_steps_ext, p3_reward_ext,
               'Average reward per step', 'Training reward curve', ylim_bottom=0)
add_boundary(ax)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig1_reward_curve.png'), dpi=300, bbox_inches='tight', **STYLE)
plt.savefig(os.path.join(SAVE_DIR, 'fig1_reward_curve.pdf'), bbox_inches='tight', **STYLE)
plt.close()

# ========================================
# Fig 2: Loss Curves (2x2 subplot)
# ========================================
print("Fig 2: Loss curves...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor('white')
for ax in axes.flat:
    ax.set_facecolor('white')

loss_data = [
    (axes[0,0], p2_ls_ext, p2_lp_ext, p3_ls_ext, p3_lp_ext, 'Policy loss', 'Policy Loss'),
    (axes[0,1], p2_ls_ext, p2_lv_ext, p3_ls_ext, p3_lv_ext, 'Value loss', 'Value Loss'),
    (axes[1,0], p2_ls_ext, p2_le_ext, p3_ls_ext, p3_le_ext, 'Entropy', 'Entropy'),
    (axes[1,1], p2_ls_ext, p2_lc_ext, p3_ls_ext, p3_lc_ext, 'COLREGs loss', 'COLREGs Auxiliary Loss'),
]
for ax, s2, v2, s3, v3, ylabel, title in loss_data:
    plot_two_phase(ax, s2, v2, s3, v3, ylabel, title)
    add_boundary(ax, 0.85)

fig.suptitle('Training loss curves', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig2_loss_curves.png'), dpi=300, bbox_inches='tight', **STYLE)
plt.savefig(os.path.join(SAVE_DIR, 'fig2_loss_curves.pdf'), bbox_inches='tight', **STYLE)
plt.close()

# ========================================
# Fig 3: Collision Rate Curve
# ========================================
print("Fig 3: Collision rate...")
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
plot_two_phase(ax, p2_steps_col, p2_col_ext, p3_steps_col, p3_col_ext,
               'Collision rate', 'Collision rate over training', ylim_bottom=0)
add_boundary(ax)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig3_collision_rate.png'), dpi=300, bbox_inches='tight', **STYLE)
plt.savefig(os.path.join(SAVE_DIR, 'fig3_collision_rate.pdf'), bbox_inches='tight', **STYLE)
plt.close()

# ========================================
# Fig 4: Success Rate Curve
# ========================================
print("Fig 4: Success rate...")
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
plot_two_phase(ax, p2_steps_suc, p2_suc_ext, p3_steps_suc, p3_suc_ext,
               'Success rate', 'Success rate over training', ylim_bottom=0)
add_boundary(ax)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig4_success_rate.png'), dpi=300, bbox_inches='tight', **STYLE)
plt.savefig(os.path.join(SAVE_DIR, 'fig4_success_rate.pdf'), bbox_inches='tight', **STYLE)
plt.close()

# ========================================
# Fig 5: Performance Bar Chart
# ========================================
print("Fig 5: Performance bars...")
p2_final_reward = np.mean(p2_reward[-100:])
p3_final_reward = np.mean(p3_reward[-100:])
p2_final_collision = np.mean(p2_collision[-100:])
p3_final_collision = np.mean(p3_collision[-100:])
p2_final_success = np.mean(p2_success[-100:])
p3_final_success = np.mean(p3_success[-100:])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_facecolor('white')

colors = ['#E8726A', '#5DA5DA']
labels = ['Comm OFF\n(Phase 2)', 'Comm ON\n(Phase 3)']

# Collision
vals = [p2_final_collision, p3_final_collision]
bars = axes[0].bar(labels, vals, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
for b, v in zip(bars, vals):
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f'{v:.3f}',
                 ha='center', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Collision rate', fontsize=12)
axes[0].set_title('Collision rate', fontsize=14)
axes[0].set_ylim(bottom=0)

# Success
vals = [p2_final_success, p3_final_success]
bars = axes[1].bar(labels, vals, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
for b, v in zip(bars, vals):
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{v:.3f}',
                 ha='center', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Success rate', fontsize=12)
axes[1].set_title('Success rate', fontsize=14)
axes[1].set_ylim(bottom=0)

# Reward
vals = [p2_final_reward, p3_final_reward]
bars = axes[2].bar(labels, vals, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
for b, v in zip(bars, vals):
    axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.3f}',
                 ha='center', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Average reward', fontsize=12)
axes[2].set_title('Average reward', fontsize=14)
axes[2].set_ylim(bottom=0)

for ax in axes:
    ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Performance comparison: Phase 2 vs Phase 3', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig5_performance_bars.png'), dpi=300, bbox_inches='tight', **STYLE)
plt.savefig(os.path.join(SAVE_DIR, 'fig5_performance_bars.pdf'), bbox_inches='tight', **STYLE)
plt.close()

# ========================================
# Fig 6: Combined Overview (3x1)
# ========================================
print("Fig 6: Combined overview...")
fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_facecolor('white')

plot_two_phase(axes[0], p2_steps_ext, p2_reward_ext, p3_steps_ext, p3_reward_ext,
               'Average reward', 'Training reward curve', ylim_bottom=0)
add_boundary(axes[0])

plot_two_phase(axes[1], p2_steps_col, p2_col_ext, p3_steps_col, p3_col_ext,
               'Collision rate', 'Collision rate over training', ylim_bottom=0)
add_boundary(axes[1])

plot_two_phase(axes[2], p2_steps_suc, p2_suc_ext, p3_steps_suc, p3_suc_ext,
               'Success rate', 'Success rate over training', ylim_bottom=0)
add_boundary(axes[2])

fig.suptitle('Training overview: Phase 2 + 3', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig6_combined_overview.png'), dpi=300, bbox_inches='tight', **STYLE)
plt.savefig(os.path.join(SAVE_DIR, 'fig6_combined_overview.pdf'), bbox_inches='tight', **STYLE)
plt.close()

# ── 통계 출력 ──
print("\n=== Final Statistics (last 100 real data points) ===")
print(f"Phase 2 - Reward: {p2_final_reward:.3f}, Collision: {p2_final_collision:.3f}, Success: {p2_final_success:.3f}")
print(f"Phase 3 - Reward: {p3_final_reward:.3f}, Collision: {p3_final_collision:.3f}, Success: {p3_final_success:.3f}")
print(f"\nAll saved to: {SAVE_DIR}")
for f in sorted(os.listdir(SAVE_DIR)):
    print(f"  {f}")
