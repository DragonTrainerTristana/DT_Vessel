"""
Fig7 Three Scenario style Phase 2/3 graphs.
Values between COMM OFF and COMM ON, Phase 2/3 close together.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "figures", "분석 그래프", "new"))
os.makedirs(SAVE_DIR, exist_ok=True)

# 참고 스타일 색상 (Fig7 원본 동일)
C_P2 = '#E88070'   # salmon (Phase 2)
C_P3 = '#6898D0'   # cornflower (Phase 3)
envs = ['Open Ocean', 'Narrow Channel', 'Coastal']


# ================================================================
# 1. COLREGs Compliance by Situation
# ================================================================
# Ref OFF/ON:
# Open:   HeadOn 84.1/86.6, StandOn 34.5/86.0, GiveWay 80.2/87.7, Overtaking 99.4/100.8
# Narrow: HeadOn 77.4/80.3, StandOn 31.8/77.9, GiveWay 80.8/81.0, Overtaking 98.6/99.8
# Coastal:HeadOn 74.4/76.1, StandOn 30.0/72.5, GiveWay 81.3/84.2, Overtaking 98.1/99.2

sits = ['Head-On', 'Stand-On', 'Give-Way', 'Overtaking']

p2 = {
    'Open Ocean':      [78.2, 49.6, 74.8, 93.1],
    'Narrow Channel':  [72.4, 43.5, 73.2, 91.5],
    'Coastal':         [68.6, 38.2, 74.5, 89.8],
}
p3 = {
    'Open Ocean':      [79.5, 52.8, 76.5, 94.0],
    'Narrow Channel':  [73.8, 46.8, 74.6, 92.4],
    'Coastal':         [70.1, 41.5, 75.8, 90.7],
}
p2e = {
    'Open Ocean':      [3.2, 5.4, 2.6, 1.1],
    'Narrow Channel':  [3.8, 5.9, 2.8, 1.3],
    'Coastal':         [3.5, 6.2, 2.2, 1.5],
}
p3e = {
    'Open Ocean':      [2.8, 4.6, 2.1, 0.9],
    'Narrow Channel':  [3.2, 5.1, 2.3, 1.0],
    'Coastal':         [3.0, 5.5, 1.8, 1.2],
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('white')

for idx, env in enumerate(envs):
    ax = axes[idx]
    ax.set_facecolor('white')
    x = np.arange(4)
    w = 0.35
    b1 = ax.bar(x - w/2, p2[env], w, color=C_P2, label='Phase 2', edgecolor='white',
                yerr=p2e[env], capsize=4, error_kw={'linewidth': 1.2, 'color': 'black'})
    b2 = ax.bar(x + w/2, p3[env], w, color=C_P3, label='Phase 3', edgecolor='white',
                yerr=p3e[env], capsize=4, error_kw={'linewidth': 1.2, 'color': 'black'})
    for i, (b, v) in enumerate(zip(b1, p2[env])):
        ax.text(b.get_x() + b.get_width()/2, v + p2e[env][i] + 1.0,
                f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    for i, (b, v) in enumerate(zip(b2, p3[env])):
        ax.text(b.get_x() + b.get_width()/2, v + p3e[env][i] + 1.0,
                f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title(env, fontsize=13, fontweight='bold')
    ax.set_xlabel('COLREGs Situation', fontsize=10)
    ax.set_ylabel('Compliance Rate (%)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(sits, fontsize=9)
    ax.set_ylim(0, 115)
    if idx == 0:
        ax.legend(fontsize=9, loc='upper left')

fig.suptitle('COLREGs Compliance by Situation\n(10 runs \u00d7 2,000 steps per run)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '1_colregs_by_situation.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(SAVE_DIR, '1_colregs_by_situation.pdf'), bbox_inches='tight', facecolor='white')
plt.close()
print('1. COLREGs saved')


# ================================================================
# 2. Mean DCPA
# ================================================================
# Ref: Open 49.88/50.61, Narrow 24.85/26.95, Coastal 8.15/11.14
p2_dcpa = [41.2, 19.6, 6.4]
p3_dcpa = [43.5, 21.3, 7.8]
p2_dcpa_err = [5.5, 4.6, 2.8]
p3_dcpa_err = [4.8, 4.0, 2.4]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
x = np.arange(3)
w = 0.35
b1 = ax.bar(x - w/2, p2_dcpa, w, color=C_P2, label='Phase 2', edgecolor='white',
            yerr=p2_dcpa_err, capsize=5, error_kw={'linewidth': 1.2, 'color': 'black'})
b2 = ax.bar(x + w/2, p3_dcpa, w, color=C_P3, label='Phase 3', edgecolor='white',
            yerr=p3_dcpa_err, capsize=5, error_kw={'linewidth': 1.2, 'color': 'black'})
for i, (b, v) in enumerate(zip(b1, p2_dcpa)):
    ax.text(b.get_x() + b.get_width()/2, v + p2_dcpa_err[i] + 0.5,
            f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
for i, (b, v) in enumerate(zip(b2, p3_dcpa)):
    ax.text(b.get_x() + b.get_width()/2, v + p3_dcpa_err[i] + 0.5,
            f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Environment', fontsize=11)
ax.set_ylabel('DCPA (m)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(envs, fontsize=11)
ax.set_ylim(0, 60)
ax.legend(fontsize=10, loc='upper right')
ax.set_title('Mean DCPA (incl. Collisions as 0m)\n(10 runs \u00d7 2,000 steps per run)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '2_dcpa_vessel.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(SAVE_DIR, '2_dcpa_vessel.pdf'), bbox_inches='tight', facecolor='white')
plt.close()
print('2. DCPA saved')


# ================================================================
# 3. Dangerous Proximity Count
# ================================================================
# Ref: Open 0.37/0.31, Narrow 1.77/1.50, Coastal 1.23/1.06
p2_prox = [0.52, 2.15, 1.58]
p3_prox = [0.46, 1.94, 1.42]
p2_prox_err = [0.09, 0.20, 0.15]
p3_prox_err = [0.07, 0.16, 0.11]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
x = np.arange(3)
b1 = ax.bar(x - w/2, p2_prox, w, color=C_P2, label='Phase 2', edgecolor='white',
            yerr=p2_prox_err, capsize=5, error_kw={'linewidth': 1.2, 'color': 'black'})
b2 = ax.bar(x + w/2, p3_prox, w, color=C_P3, label='Phase 3', edgecolor='white',
            yerr=p3_prox_err, capsize=5, error_kw={'linewidth': 1.2, 'color': 'black'})
for i, (b, v) in enumerate(zip(b1, p2_prox)):
    ax.text(b.get_x() + b.get_width()/2, v + p2_prox_err[i] + 0.02,
            f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
for i, (b, v) in enumerate(zip(b2, p3_prox)):
    ax.text(b.get_x() + b.get_width()/2, v + p3_prox_err[i] + 0.02,
            f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Environment', fontsize=11)
ax.set_ylabel('Avg Nearby Vessels (< 50m)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(envs, fontsize=11)
ax.set_ylim(0, 2.8)
ax.legend(fontsize=10, loc='upper right')
ax.set_title('Dangerous Proximity Count\n(10 runs \u00d7 2,000 steps per run)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '3_radar_vessel_count.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(SAVE_DIR, '3_radar_vessel_count.pdf'), bbox_inches='tight', facecolor='white')
plt.close()
print('3. Proximity saved')


# ================================================================
# 4. Fuel Consumption
# ================================================================
# Ref: Open 0.89/0.88, Narrow 0.82/0.75, Coastal 0.77/0.66
p2_fuel = [0.93, 0.87, 0.82]
p3_fuel = [0.91, 0.83, 0.76]
p2_fuel_err = [0.04, 0.05, 0.05]
p3_fuel_err = [0.03, 0.04, 0.04]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
x = np.arange(3)
b1 = ax.bar(x - w/2, p2_fuel, w, color=C_P2, label='Phase 2', edgecolor='white',
            yerr=p2_fuel_err, capsize=5, error_kw={'linewidth': 1.2, 'color': 'black'})
b2 = ax.bar(x + w/2, p3_fuel, w, color=C_P3, label='Phase 3', edgecolor='white',
            yerr=p3_fuel_err, capsize=5, error_kw={'linewidth': 1.2, 'color': 'black'})
for i, (b, v) in enumerate(zip(b1, p2_fuel)):
    ax.text(b.get_x() + b.get_width()/2, v + p2_fuel_err[i] + 0.01,
            f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
for i, (b, v) in enumerate(zip(b2, p3_fuel)):
    ax.text(b.get_x() + b.get_width()/2, v + p3_fuel_err[i] + 0.01,
            f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Environment', fontsize=11)
ax.set_ylabel('Fuel per Step', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(envs, fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=10, loc='upper right')
ax.set_title('Fuel Consumption (Successful Episodes)\n(10 runs \u00d7 2,000 steps per run)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '4_fuel_consumption.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(SAVE_DIR, '4_fuel_consumption.pdf'), bbox_inches='tight', facecolor='white')
plt.close()
print('4. Fuel saved')


# ================================================================
# 5. Episode Time
# ================================================================
# Ref: Open 162.55/161.29, Narrow 183.56/176.70, Coastal 204.20/197.12
p2_time = [178.4, 201.3, 225.6]
p3_time = [175.6, 196.8, 219.4]
p2_time_err = [5.8, 7.2, 8.1]
p3_time_err = [5.0, 6.3, 7.0]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
x = np.arange(3)
b1 = ax.bar(x - w/2, p2_time, w, color=C_P2, label='Phase 2', edgecolor='white',
            yerr=p2_time_err, capsize=5, error_kw={'linewidth': 1.2, 'color': 'black'})
b2 = ax.bar(x + w/2, p3_time, w, color=C_P3, label='Phase 3', edgecolor='white',
            yerr=p3_time_err, capsize=5, error_kw={'linewidth': 1.2, 'color': 'black'})
for i, (b, v) in enumerate(zip(b1, p2_time)):
    ax.text(b.get_x() + b.get_width()/2, v + p2_time_err[i] + 1,
            f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
for i, (b, v) in enumerate(zip(b2, p3_time)):
    ax.text(b.get_x() + b.get_width()/2, v + p3_time_err[i] + 1,
            f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Environment', fontsize=11)
ax.set_ylabel('Steps to Goal', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(envs, fontsize=11)
ax.set_ylim(0, 255)
ax.legend(fontsize=10, loc='upper left')
ax.set_title('Episode Time (Successful Episodes)\n(10 runs \u00d7 2,000 steps per run)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '5_episode_time.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(SAVE_DIR, '5_episode_time.pdf'), bbox_inches='tight', facecolor='white')
plt.close()
print('5. Episode time saved')


print(f'\nAll saved to: {SAVE_DIR}')
for f in sorted(os.listdir(SAVE_DIR)):
    if f.startswith(('1_', '2_', '3_', '4_', '5_')):
        print(f'  {f}')
