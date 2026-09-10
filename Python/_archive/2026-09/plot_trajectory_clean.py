"""
Clean Trajectory Visualization v3
- 주인공 궤적만 표시
- 감지 시점에 이웃 위치(점) + step 번호
- Comm ON: 같은 step에서 radar + comm 원 동시 표시
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

TRAJ_OFF = os.path.join(PROJECT_ROOT, "trajectory_data", "commOFF.csv")
TRAJ_ON = os.path.join(PROJECT_ROOT, "trajectory_data", "commON.csv")

SAVE_DIR = os.path.join(PROJECT_ROOT, "figures", "분석 그래프", "Fig4_Trajectory", "v2_clean")
os.makedirs(SAVE_DIR, exist_ok=True)

RADAR_RANGE = 60.0
COMM_RANGE = 90.0

COLREGS_LABELS = {1: 'HeadOn', 2: 'CrossStandOn', 3: 'CrossGiveWay', 4: 'Overtaking'}
COLREGS_COLORS = {1: '#E74C3C', 2: '#3498DB', 3: '#2ECC71', 4: '#F39C12'}

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
})


def extract_first_episode(df, agent_id):
    a = df[df.agent_id == agent_id].reset_index(drop=True)
    dx = a.x.diff().abs().fillna(0)
    dz = a.z.diff().abs().fillna(0)
    jumps = (dx > 50) | (dz > 50)
    first_jump = jumps[jumps].index[0] if jumps.any() else len(a)
    return a.iloc[:first_jump]


def find_densest_step(df, target_aid, ep):
    """
    주변 배가 가장 많은 step 찾기 (comm 범위 기준)
    Returns: {step, tx, tz, neighbors: [{ox, oz, dist, colregs, aid}]}
    """
    steps = ep.step.values
    best = None
    best_count = 0

    for step in steps[::2]:  # 2step 간격 탐색
        target_row = ep[ep.step == step]
        if len(target_row) == 0:
            continue
        tx, tz = target_row.x.values[0], target_row.z.values[0]

        others = df[(df.step == step) & (df.agent_id != target_aid)]
        neighbors = []
        for _, row in others.iterrows():
            dist = np.sqrt((tx - row.x)**2 + (tz - row.z)**2)
            if dist <= COMM_RANGE:
                neighbors.append({
                    'ox': row.x, 'oz': row.z, 'dist': dist,
                    'colregs': int(row.colregs), 'aid': int(row.agent_id),
                })

        if len(neighbors) > best_count:
            best_count = len(neighbors)
            best = {'step': int(step), 'tx': tx, 'tz': tz, 'neighbors': neighbors}

    return best


def plot_scenario(df_off, df_on, aid, scenario_num):
    ep_off = extract_first_episode(df_off, aid)
    ep_on = extract_first_episode(df_on, aid)

    if len(ep_off) < 10 or len(ep_on) < 10:
        print(f"  Scenario {scenario_num}: insufficient data")
        return None

    # 주변 배가 가장 많은 시점 찾기
    dense_off = find_densest_step(df_off, aid, ep_off)
    dense_on = find_densest_step(df_on, aid, ep_on)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, ep, dense, title, is_comm in [
        (ax1, ep_off, dense_off, 'Without Communication', False),
        (ax2, ep_on, dense_on, 'With Communication', True),
    ]:
        color_main = '#3498DB' if is_comm else '#E74C3C'
        style = '--' if is_comm else '-'

        # 주인공 궤적만
        ax.plot(ep.x.values, ep.z.values, style, color=color_main,
                linewidth=2.5, alpha=0.9, zorder=3)

        # 시작점 / 끝점
        ax.plot(ep.x.iloc[0], ep.z.iloc[0], 'o', color=color_main,
                markersize=14, zorder=6, markeredgecolor='black', markeredgewidth=1.2)
        ax.plot(ep.x.iloc[-1], ep.z.iloc[-1], 's', color=color_main,
                markersize=14, zorder=6, markeredgecolor='black', markeredgewidth=1.2)
        ax.annotate('START', xy=(ep.x.iloc[0], ep.z.iloc[0]),
                    fontsize=9, fontweight='bold',
                    xytext=(8, 10), textcoords='offset points')

        # === 주변 배가 가장 많은 시점에서 범위 원 + 이웃 표시 ===
        if dense:
            tx0, tz0 = dense['tx'], dense['tz']
            step = dense['step']

            # Radar 원
            ax.add_patch(Circle((tx0, tz0), RADAR_RANGE, fill=False,
                                color='#FF6B6B', linestyle='--', linewidth=1.8,
                                alpha=0.7, zorder=2))
            ax.annotate(f'Radar {RADAR_RANGE:.0f}m',
                        xy=(tx0, tz0 + RADAR_RANGE), fontsize=8,
                        color='#FF6B6B', ha='center', fontweight='bold',
                        xytext=(0, 6), textcoords='offset points')

            if is_comm:
                # Comm 원
                ax.add_patch(Circle((tx0, tz0), COMM_RANGE, fill=False,
                                    color='#4ECDC4', linestyle=':', linewidth=1.8,
                                    alpha=0.7, zorder=2))
                ax.annotate(f'Comm {COMM_RANGE:.0f}m',
                            xy=(tx0, tz0 + COMM_RANGE), fontsize=8,
                            color='#4ECDC4', ha='center', fontweight='bold',
                            xytext=(0, 6), textcoords='offset points')

            # 원 중심 (주인공 위치) + step 표시
            ax.plot(tx0, tz0, '*', color=color_main, markersize=15,
                    zorder=6, markeredgecolor='black', markeredgewidth=0.8)
            ax.annotate(f'step {step}', xy=(tx0, tz0),
                        fontsize=8, fontweight='bold', color=color_main,
                        xytext=(-10, -18), textcoords='offset points', ha='center')

            # 이웃 선박 위치 + 라벨
            for nb in dense['neighbors']:
                in_radar = nb['dist'] <= RADAR_RANGE
                nb_color = '#FF6B6B' if in_radar else '#4ECDC4'
                if nb['colregs'] in COLREGS_COLORS:
                    nb_color = COLREGS_COLORS[nb['colregs']]

                # 주인공 ↔ 이웃 연결선
                ax.plot([tx0, nb['ox']], [tz0, nb['oz']],
                        color=nb_color, alpha=0.4, linewidth=1, linestyle='-', zorder=2)

                # 이웃 마커
                ax.plot(nb['ox'], nb['oz'], '^', color=nb_color,
                        markersize=14, markeredgecolor='black',
                        markeredgewidth=0.8, zorder=5)

                # 라벨: 거리 + COLREGs
                dist_str = f"{nb['dist']:.0f}m"
                if nb['colregs'] in COLREGS_LABELS:
                    dist_str += f"\n{COLREGS_LABELS[nb['colregs']]}"
                range_type = "radar" if in_radar else "comm"
                dist_str += f"\n({range_type})"

                ax.annotate(dist_str,
                            xy=(nb['ox'], nb['oz']),
                            fontsize=7, color=nb_color, fontweight='bold',
                            xytext=(10, 5), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                      edgecolor=nb_color, alpha=0.85))

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)

    fig.suptitle(f'Scenario {scenario_num}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        path = os.path.join(SAVE_DIR, f'scenario_{scenario_num}.{ext}')
        fig.savefig(path, facecolor='white', bbox_inches='tight')
    off_nb = len(dense_off['neighbors']) if dense_off else 0
    on_nb = len(dense_on['neighbors']) if dense_on else 0
    print(f"  Saved: scenario_{scenario_num} "
          f"(OFF: step {dense_off['step'] if dense_off else '-'}, {off_nb} neighbors | "
          f"ON: step {dense_on['step'] if dense_on else '-'}, {on_nb} neighbors)")
    plt.close(fig)

    return {'scenario': scenario_num, 'agent_id': aid,
            'off_len': len(ep_off), 'on_len': len(ep_on)}


def plot_overlay(df_off, df_on, aid, scenario_num):
    """Comm OFF vs ON 같은 축에 겹쳐서 회피 경로 차이 비교"""

    ep_off = extract_first_episode(df_off, aid)
    ep_on = extract_first_episode(df_on, aid)

    if len(ep_off) < 10 or len(ep_on) < 10:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Comm OFF (실선, 빨강)
    ax.plot(ep_off.x.values, ep_off.z.values, '-', color='#E74C3C',
            linewidth=2.5, alpha=0.9, label=f'Comm OFF (len={len(ep_off)})', zorder=3)
    ax.plot(ep_off.x.iloc[0], ep_off.z.iloc[0], 'o', color='#E74C3C',
            markersize=14, zorder=6, markeredgecolor='black', markeredgewidth=1.2)
    ax.plot(ep_off.x.iloc[-1], ep_off.z.iloc[-1], 's', color='#E74C3C',
            markersize=14, zorder=6, markeredgecolor='black', markeredgewidth=1.2)

    # Comm ON (점선, 파랑)
    ax.plot(ep_on.x.values, ep_on.z.values, '--', color='#3498DB',
            linewidth=2.5, alpha=0.9, label=f'Comm ON (len={len(ep_on)})', zorder=3)
    ax.plot(ep_on.x.iloc[0], ep_on.z.iloc[0], 'o', color='#3498DB',
            markersize=14, zorder=6, markeredgecolor='black', markeredgewidth=1.2)
    ax.plot(ep_on.x.iloc[-1], ep_on.z.iloc[-1], 's', color='#3498DB',
            markersize=14, zorder=6, markeredgecolor='black', markeredgewidth=1.2)

    # START 라벨
    ax.annotate('START', xy=(ep_off.x.iloc[0], ep_off.z.iloc[0]),
                fontsize=10, fontweight='bold',
                xytext=(8, 10), textcoords='offset points')

    # COLREGs 마커 (OFF: 원, ON: 다이아몬드)
    added_labels = set()
    for ep, mk in [(ep_off, 'o'), (ep_on, 'D')]:
        for c in [1, 2, 3, 4]:
            pts = ep[ep.colregs == c]
            if len(pts) > 0:
                label = COLREGS_LABELS[c] if c not in added_labels else None
                ax.scatter(pts.x.values, pts.z.values, c=COLREGS_COLORS[c],
                           s=30, alpha=0.5, zorder=4, edgecolors='none',
                           marker=mk, label=label)
                added_labels.add(c)

    # 경로 차이가 큰 구간에 화살표로 표시
    # 같은 step에서 OFF vs ON 위치 차이가 가장 큰 지점 찾기
    common_steps = set(ep_off.step.values) & set(ep_on.step.values)
    if common_steps:
        max_diff = 0
        best_step = None
        for s in sorted(common_steps)[::5]:
            row_off = ep_off[ep_off.step == s]
            row_on = ep_on[ep_on.step == s]
            if len(row_off) == 0 or len(row_on) == 0:
                continue
            diff = np.sqrt((row_off.x.values[0] - row_on.x.values[0])**2 +
                           (row_off.z.values[0] - row_on.z.values[0])**2)
            if diff > max_diff:
                max_diff = diff
                best_step = s

        if best_step is not None and max_diff > 5:
            row_off = ep_off[ep_off.step == best_step]
            row_on = ep_on[ep_on.step == best_step]
            xo, zo = row_off.x.values[0], row_off.z.values[0]
            xn, zn = row_on.x.values[0], row_on.z.values[0]

            # 양쪽 경로의 같은 step 위치를 연결
            ax.annotate('', xy=(xn, zn), xytext=(xo, zo),
                        arrowprops=dict(arrowstyle='<->', color='#888888',
                                        linewidth=1.5, linestyle='-'))
            mx, mz = (xo + xn) / 2, (zo + zn) / 2
            ax.annotate(f'step {best_step}\n({max_diff:.0f}m gap)',
                        xy=(mx, mz), fontsize=8, color='#555555',
                        fontweight='bold', ha='center',
                        xytext=(15, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='#888888', alpha=0.9))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'Scenario {scenario_num}: Trajectory Comparison (Overlay)',
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(SAVE_DIR, f'scenario_{scenario_num}_overlay.{ext}')
        fig.savefig(path, facecolor='white', bbox_inches='tight')
    print(f"  Saved: scenario_{scenario_num}_overlay (max gap: {max_diff:.0f}m)")
    plt.close(fig)


def main():
    print("=" * 60)
    print("  Clean Trajectory v3")
    print(f"  Save dir: {SAVE_DIR}")
    print("=" * 60)

    df_off = pd.read_csv(TRAJ_OFF)
    df_on = pd.read_csv(TRAJ_ON)
    print(f"  Comm OFF: {len(df_off)} rows, {df_off.agent_id.nunique()} agents")
    print(f"  Comm ON:  {len(df_on)} rows, {df_on.agent_id.nunique()} agents")

    target_agents = [0, 3]

    for i, aid in enumerate(target_agents):
        print(f"\n[Scenario {i + 1}] Agent {aid}")
        plot_scenario(df_off, df_on, aid, i + 1)
        plot_overlay(df_off, df_on, aid, i + 1)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
