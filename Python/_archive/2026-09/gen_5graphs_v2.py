"""
0330 계획에 따른 5개 그래프 생성 (수정 버전)
A-1: COLREGs 4상황별, A-2: DCPA (vessel-vessel only), A-3: 위험 근접 선박 수 (50m)
B-1: Fuel, B-2: Episode Time
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
TRAJ_DIR = os.path.join(PROJECT_ROOT, "trajectory_data")
OUT_DIR = os.path.join(PROJECT_ROOT, "figures", "efficiency_0330")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_FILES = {
    'Open Ocean': {
        'OFF': os.path.join(TRAJ_DIR, 'open_ocean_compare_commOFF_10x2000_20260319_015000.csv'),
        'ON':  os.path.join(TRAJ_DIR, 'open_ocean_compare_commON_10x2000_20260319_015000.csv'),
    },
    'Coastal': {
        'OFF': os.path.join(TRAJ_DIR, 'narrow_compare_commOFF_10x2000_20260318_163447.csv'),
        'ON':  os.path.join(TRAJ_DIR, 'narrow_compare_commON_10x2000_20260318_163447.csv'),
    },
}

DANGER_THRESHOLD = 50.0

def load_runs(csv_path):
    df = pd.read_csv(csv_path)
    total_steps = df['step'].max() + 1
    steps_per_run = total_steps // 10
    runs = []
    for r in range(10):
        s, e = r * steps_per_run, (r + 1) * steps_per_run
        runs.append(df[(df['step'] >= s) & (df['step'] < e)].copy())
    return runs

def compute_colregs_per_situation(run_df):
    results = {}
    for situation in ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']:
        rows = run_df[run_df['colregs_name'] == situation]
        if len(rows) == 0:
            results[situation] = np.nan
            continue
        if situation in ['HeadOn', 'CrossGiveWay']:
            compliant = (rows['rudder'] > 0).sum()
        elif situation == 'CrossStandOn':
            compliant = (rows['rudder'].abs() < 0.2).sum()
        elif situation == 'Overtaking':
            compliant = (rows['rudder'].abs() > 0.01).sum()
        results[situation] = compliant / len(rows) * 100
    return results

def compute_dcpa_vessel(run_df):
    pivot_x = run_df.pivot_table(index='step', columns='agent_id', values='x', aggfunc='first')
    pivot_z = run_df.pivot_table(index='step', columns='agent_id', values='z', aggfunc='first')
    agents = sorted(pivot_x.columns)
    if len(agents) < 2:
        return np.nan
    encounter_mins = []
    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            ai, aj = agents[i], agents[j]
            dx = pivot_x[ai].values - pivot_x[aj].values
            dz = pivot_z[ai].values - pivot_z[aj].values
            dists = np.sqrt(dx**2 + dz**2)
            valid = ~np.isnan(dists)
            dists_v = dists[valid]
            if len(dists_v) == 0:
                continue
            in_range = dists_v < 100.0
            changes = np.diff(in_range.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1
            if in_range[0]: starts = np.concatenate([[0], starts])
            if in_range[-1]: ends = np.concatenate([ends, [len(dists_v)]])
            for s, e in zip(starts, ends):
                if e > s:
                    encounter_mins.append(np.min(dists_v[s:e]))
    if len(encounter_mins) == 0:
        return np.nan
    return np.mean(encounter_mins)

def compute_danger_vessel_count(run_df):
    pivot_x = run_df.pivot_table(index='step', columns='agent_id', values='x', aggfunc='first')
    pivot_z = run_df.pivot_table(index='step', columns='agent_id', values='z', aggfunc='first')
    agents = sorted(pivot_x.columns)
    if len(agents) < 2:
        return 0.0
    x_vals = pivot_x[agents].values
    z_vals = pivot_z[agents].values
    n_steps, n_agents = x_vals.shape
    counts = np.zeros(n_steps)
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j: continue
            dx = x_vals[:, i] - x_vals[:, j]
            dz = z_vals[:, i] - z_vals[:, j]
            dists = np.sqrt(dx**2 + dz**2)
            valid = ~np.isnan(dists)
            in_danger = (dists < DANGER_THRESHOLD) & valid
            counts += in_danger
    return counts.sum() / (n_steps * n_agents)

def get_episodes(run_df):
    episodes = []
    for aid in run_df['agent_id'].unique():
        agent_data = run_df[run_df['agent_id'] == aid].sort_values('step')
        if len(agent_data) < 2:
            continue
        goal_dists = agent_data['goal_dist'].values
        boundaries = [0]
        for k in range(1, len(goal_dists)):
            if goal_dists[k] > goal_dists[k-1] + 0.3:
                boundaries.append(k)
        boundaries.append(len(goal_dists))
        for b in range(len(boundaries)-1):
            ep = agent_data.iloc[boundaries[b]:boundaries[b+1]]
            if len(ep) < 5:
                continue
            success = ep['goal_dist'].iloc[-1] < 0.075
            episodes.append({'data': ep, 'success': success})
    return episodes

def compute_fuel(ep_data):
    speed = ep_data['speed'].abs().values
    rudder = ep_data['rudder'].values
    delta_rudder = np.abs(np.diff(rudder))
    return np.sum(speed) + 0.5 * np.sum(delta_rudder)

# ============================================================
# 분석 실행
# ============================================================
print("=" * 60)
print("분석 시작 (수정 버전 v2)")
print("=" * 60)

all_results = {}
for env_name, csvs in CSV_FILES.items():
    print(f"\n--- {env_name} ---")
    all_results[env_name] = {}
    for comm, path in csvs.items():
        runs = load_runs(path)
        colregs_runs = {'HeadOn': [], 'CrossStandOn': [], 'CrossGiveWay': [], 'Overtaking': []}
        dcpa_runs, danger_runs, fuel_runs, time_runs = [], [], [], []

        for rid, run_df in enumerate(runs):
            cr = compute_colregs_per_situation(run_df)
            for sit in colregs_runs:
                if not np.isnan(cr.get(sit, np.nan)):
                    colregs_runs[sit].append(cr[sit])
            dcpa = compute_dcpa_vessel(run_df)
            if not np.isnan(dcpa): dcpa_runs.append(dcpa)
            danger_runs.append(compute_danger_vessel_count(run_df))
            for ep in get_episodes(run_df):
                if ep['success']:
                    fuel_runs.append(compute_fuel(ep['data']))
                    time_runs.append(len(ep['data']))

        result = {
            'colregs': {s: (np.mean(v), np.std(v)) if v else (np.nan, np.nan) for s, v in colregs_runs.items()},
            'dcpa': (np.mean(dcpa_runs), np.std(dcpa_runs)) if dcpa_runs else (np.nan, np.nan),
            'danger_count': (np.mean(danger_runs), np.std(danger_runs)),
            'fuel': (np.mean(fuel_runs), np.std(fuel_runs)) if fuel_runs else (np.nan, np.nan),
            'episode_time': (np.mean(time_runs), np.std(time_runs)) if time_runs else (np.nan, np.nan),
        }
        all_results[env_name][comm] = result
        print(f"  [{comm}] DCPA={result['dcpa'][0]:.1f}m, Danger={result['danger_count'][0]:.2f}, "
              f"Fuel={result['fuel'][0]:.1f}, Time={result['episode_time'][0]:.0f}, SuccessEps={len(fuel_runs)}")

# 결과 출력
print("\n" + "=" * 80)
for env in ['Open Ocean', 'Coastal']:
    print(f"\n--- {env} ---")
    for sit in ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']:
        off = all_results[env]['OFF']['colregs'][sit]
        on = all_results[env]['ON']['colregs'][sit]
        print(f"  COLREGs {sit:15s}: OFF={off[0]:.1f}%  ON={on[0]:.1f}%")
    for key, label in [('dcpa','DCPA'), ('danger_count','Danger<50m'), ('fuel','Fuel'), ('episode_time','Time')]:
        off = all_results[env]['OFF'][key]
        on = all_results[env]['ON'][key]
        print(f"  {label:20s}: OFF={off[0]:.1f}  ON={on[0]:.1f}")

# ============================================================
# 5개 그래프 생성
# ============================================================
envs_csv = ['Open Ocean', 'Coastal']
width = 0.35

# 1. COLREGs 4상황별
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
situations = ['HeadOn', 'CrossStandOn', 'CrossGiveWay', 'Overtaking']
sit_labels = ['Head-On', 'Stand-On', 'Give-Way', 'Overtaking']
x = np.arange(len(situations))

for idx, env in enumerate(envs_csv):
    ax = axes[idx]
    off_v = [all_results[env]['OFF']['colregs'][s][0] for s in situations]
    off_e = [all_results[env]['OFF']['colregs'][s][1] for s in situations]
    on_v = [all_results[env]['ON']['colregs'][s][0] for s in situations]
    on_e = [all_results[env]['ON']['colregs'][s][1] for s in situations]
    off_v = [0 if np.isnan(v) else v for v in off_v]
    on_v = [0 if np.isnan(v) else v for v in on_v]
    off_e = [0 if np.isnan(v) else v for v in off_e]
    on_e = [0 if np.isnan(v) else v for v in on_e]

    b1 = ax.bar(x - width/2, off_v, width, yerr=off_e, label='COMM OFF', color='#E74C3C', alpha=0.85, capsize=4)
    b2 = ax.bar(x + width/2, on_v, width, yerr=on_e, label='COMM ON', color='#3498DB', alpha=0.85, capsize=4)
    for bar, val in zip(b1, off_v):
        if val > 0: ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+2, f'{val:.0f}%', ha='center', fontsize=8, fontweight='bold')
    for bar, val in zip(b2, on_v):
        if val > 0: ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+2, f'{val:.0f}%', ha='center', fontsize=8, fontweight='bold')
    ax.set_title(env, fontsize=13, fontweight='bold')
    ax.set_ylabel('Compliance (%)')
    ax.set_xticks(x); ax.set_xticklabels(sit_labels, fontsize=9)
    ax.set_ylim(0, 115); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

plt.suptitle('A-1. COLREGs Compliance by Situation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '1_colregs_by_situation.png'), dpi=200)
plt.savefig(os.path.join(OUT_DIR, '1_colregs_by_situation.pdf'))
plt.close()
print("\n[OK] 1_colregs_by_situation")

# Helper for single bar charts
def bar_chart(metric_key, title, ylabel, filename, fmt='{:.1f}'):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(envs_csv))
    off_v = [all_results[e]['OFF'][metric_key][0] for e in envs_csv]
    off_e = [all_results[e]['OFF'][metric_key][1] for e in envs_csv]
    on_v = [all_results[e]['ON'][metric_key][0] for e in envs_csv]
    on_e = [all_results[e]['ON'][metric_key][1] for e in envs_csv]
    b1 = ax.bar(x-width/2, off_v, width, yerr=off_e, label='COMM OFF', color='#E74C3C', alpha=0.85, capsize=5)
    b2 = ax.bar(x+width/2, on_v, width, yerr=on_e, label='COMM ON', color='#3498DB', alpha=0.85, capsize=5)
    for bar, val in zip(b1, off_v):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+max(off_e)*0.3, fmt.format(val), ha='center', fontsize=10, fontweight='bold')
    for bar, val in zip(b2, on_v):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+max(on_e)*0.3, fmt.format(val), ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(envs_csv, fontsize=11)
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename+'.png'), dpi=200)
    plt.savefig(os.path.join(OUT_DIR, filename+'.pdf'))
    plt.close()
    print(f"[OK] {filename}")

bar_chart('dcpa', 'A-2. Average DCPA (Vessel-Vessel)', 'DCPA (m)', '2_dcpa_vessel_only')
bar_chart('danger_count', 'A-3. Dangerous Proximity Count (< 50m)', 'Avg Vessels within 50m', '3_danger_vessel_count', fmt='{:.2f}')
bar_chart('fuel', 'B-1. Fuel Consumption (successful episodes)', 'Fuel (normalized)', '4_fuel_consumption', fmt='{:.0f}')
bar_chart('episode_time', 'B-2. Episode Time (successful episodes)', 'Steps', '5_episode_time', fmt='{:.0f}')

print(f"\n완료! 폴더: {OUT_DIR}")
