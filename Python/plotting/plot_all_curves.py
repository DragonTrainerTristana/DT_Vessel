"""
★모든 학습 케이스의 training curve 정리 — 10_CURVES/ 폴더 일괄 생성.
  10_CURVES/individual/<label>.png : run 하나당 곡선 1장 (raw 지터 + EMA, 최종 eval 성능 주석)
  10_CURVES/group_<axis>.png       : 실험 축별 묶음 비교
  10_CURVES/00_contact_sheet.png   : 전체 run 한눈에 (컨택트 시트)
  10_CURVES/INDEX.md               : run 목록 + 설정 + 최종 eval 지표 표
데이터: scratchpad의 학습 CSV(step,raw_reward,ema_reward) + metrics_v2.txt(frozen eval).
"""
import os, re, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import paper_style as ps

FIG = os.path.dirname(os.path.abspath(__file__))
# 데이터 경로: 프로젝트 내 _data(고정 사본)를 우선 사용 → 임시폴더가 지워져도 재생성 가능
_LOCAL = os.path.join(FIG, '_data')
SCR = os.environ.get('VESSEL_LOG_DIR', _LOCAL if os.path.isdir(_LOCAL) else
    os.path.join(FIG, 'logs'))
OUT = os.path.join(FIG, '10_CURVES')
IND = os.path.join(OUT, 'individual')
os.makedirs(IND, exist_ok=True)
ps.apply()
WARMUP, EMA_A = 3, 0.10

# (csv 파일, 표시 라벨, eval 라벨, 실험축, 설정 설명)
RUNS = [
    # ── basis: MoE-5x, 16척 ──
    ('base_off.csv',        'BASIS OFF s42',   'OFF_s42',   'basis',  'MoE-5x, no comm'),
    ('base_off_s43.csv',    'BASIS OFF s43',   'OFF_s43',   'basis',  'MoE-5x, no comm'),
    ('base_off_s44.csv',    'BASIS OFF s44',   'OFF_s44',   'basis',  'MoE-5x, no comm'),
    ('base_comm.csv',       'BASIS COMM s42',  'COMM_s42',  'basis',  'MoE-5x, comm@9M, dim 6'),
    ('base_comm_s43.csv',   'BASIS COMM s43',  'COMM_s43',  'basis',  'MoE-5x, comm@9M, dim 6'),
    ('base_comm_s44.csv',   'BASIS COMM s44',  'COMM_s44',  'basis',  'MoE-5x, comm@9M, dim 6'),
    # ── 메시지 차원 ──
    ('v2_DIM2.csv',   'DIM2 s42',  'DIM2',      'dimension', 'msg dim 2'),
    ('q_DIM2_s43.csv','DIM2 s43',  'DIM2_s43',  'dimension', 'msg dim 2'),
    ('q_DIM2_s44.csv','DIM2 s44',  'DIM2_s44',  'dimension', 'msg dim 2'),
    ('v2_DIM4.csv',   'DIM4 s42',  'DIM4',      'dimension', 'msg dim 4'),
    ('qe_DIM4_s43.csv','DIM4 s43', 'DIM4_s43',  'dimension', 'msg dim 4'),
    ('qe_DIM4_s44.csv','DIM4 s44', 'DIM4_s44',  'dimension', 'msg dim 4'),
    ('v2_DIM8.csv',   'DIM8 s42',  'DIM8',      'dimension', 'msg dim 8'),
    ('qe_DIM8_s43.csv','DIM8 s43', 'DIM8_s43',  'dimension', 'msg dim 8'),
    ('qe_DIM8_s44.csv','DIM8 s44', 'DIM8_s44',  'dimension', 'msg dim 8'),
    ('v2_DIM10.csv',  'DIM10 s42', 'DIM10',     'dimension', 'msg dim 10'),
    ('qh_DIM10_s43.csv','DIM10 s43','DIM10_s43','dimension', 'msg dim 10'),
    ('qh_DIM10_s44.csv','DIM10 s44','DIM10_s44','dimension', 'msg dim 10'),
    ('v2_DIM12.csv',  'DIM12 s42', 'DIM12',     'dimension', 'msg dim 12'),
    ('q_DIM12_s43.csv','DIM12 s43','DIM12_s43', 'dimension', 'msg dim 12'),
    ('q_DIM12_s44.csv','DIM12 s44','DIM12_s44', 'dimension', 'msg dim 12'),
    # ── 메시지 집계 / 통신 타이밍 ──
    ('v2_NEAR1.csv',    'NEAR1 s42',  'NEAR1',      'aggregation', 'nearest-1 (MoE-5x)'),
    ('qd_NEAR1_s43.csv','NEAR1 s43',  'NEAR1_s43',  'aggregation', 'nearest-1 (MoE-5x)'),
    ('qe_NEAR1_s44.csv','NEAR1 s44',  'NEAR1_s44',  'aggregation', 'nearest-1 (MoE-5x)'),
    ('qf_SE_NEAR1_s42.csv','MoE-shared NEAR1 s42','SE_NEAR1_s42','aggregation','nearest-1 (MoE-shared)'),
    ('qf_SE_NEAR1_s43.csv','MoE-shared NEAR1 s43','SE_NEAR1_s43','aggregation','nearest-1 (MoE-shared)'),
    ('qf_SE_NEAR1_s44.csv','MoE-shared NEAR1 s44','SE_NEAR1_s44','aggregation','nearest-1 (MoE-shared)'),
    ('v2_START.csv',  'COMM from start', 'START',      'protocol', 'comm from step 0'),
    ('q_PURECOMM_s42.csv','PURECOMM s42','PURECOMM_s42','protocol', 'comm, no aux loss'),
    ('q_PURECOMM_s43.csv','PURECOMM s43','PURECOMM_s43','protocol', 'comm, no aux loss'),
    ('q_PURECOMM_s44.csv','PURECOMM s44','PURECOMM_s44','protocol', 'comm, no aux loss'),
    # ── MoE 아키텍처 ──
    ('q_MOE_SINGLE.csv',   'SINGLE s42', 'MOE_SINGLE',     'moe', 'single network (no routing)'),
    ('qd_MOE_SINGLE_s43.csv','SINGLE s43','MOE_SINGLE_s43','moe', 'single network (no routing)'),
    ('qd_MOE_SINGLE_s44.csv','SINGLE s44','MOE_SINGLE_s44','moe', 'single network (no routing)'),
    ('q_MOE_ISO.csv',      'MoE-iso s42','MOE_ISO',        'moe', 'iso-param MoE (width 0.44)'),
    ('qd_MOE_SE_s42.csv',  'MoE-shared s42', 'MOE_SE_s42',     'moe', 'MoE-shared + comm'),
    ('qd_MOE_SE_s43.csv',  'MoE-shared s43', 'MOE_SE_s43',     'moe', 'MoE-shared + comm'),
    ('qd_MOE_SE_s44.csv',  'MoE-shared s44', 'MOE_SE_s44',     'moe', 'MoE-shared + comm'),
    ('qf_SE_OFF_s42.csv',  'MoE-shared OFF s42', 'SE_OFF_s42',     'moe', 'MoE-shared, no comm'),
    ('qf_SE_OFF_s43.csv',  'MoE-shared OFF s43', 'SE_OFF_s43',     'moe', 'MoE-shared, no comm'),
    ('qf_SE_OFF_s44.csv',  'MoE-shared OFF s44', 'SE_OFF_s44',     'moe', 'MoE-shared, no comm'),
    # ── 보상 설계 ──
    ('q_COLREGSOFF.csv', 'COLREGs OFF', 'COLREGSOFF', 'reward', 'COLREGs term removed'),
    ('qi_VA_OFF_s42.csv',  'VA OFF s42',  'VA_OFF_s42',  'reward', 'value-aligned reward, no comm'),
    ('qi_VA_OFF_s43.csv',  'VA OFF s43',  'VA_OFF_s43',  'reward', 'value-aligned reward, no comm'),
    ('qi_VA_OFF_s44.csv',  'VA OFF s44',  'VA_OFF_s44',  'reward', 'value-aligned reward, no comm'),
    ('qi_VA_COMM_s42.csv', 'VA COMM s42', 'VA_COMM_s42', 'reward', 'value-aligned reward + comm'),
    ('qi_VA_COMM_s43.csv', 'VA COMM s43', 'VA_COMM_s43', 'reward', 'value-aligned reward + comm'),
    ('qi_VA_COMM_s44.csv', 'VA COMM s44', 'VA_COMM_s44', 'reward', 'value-aligned reward + comm'),
    # ── 제한시계(안개) ──
    ('qd_FOG28_OFF_s42.csv',  'FOG28 OFF',   'FOG28_OFF_s42',  'fog', 'radar 28 m, no comm'),
    ('qd_FOG28_COMM_s42.csv', 'FOG28 COMM',  'FOG28_COMM_s42', 'fog', 'radar 28 m, comm@9M'),
    ('qd_FOG28_START_s42.csv','FOG28 START', 'FOG28_START_s42','fog', 'radar 28 m, comm from start'),
    # ── 고밀도(혼잡) ──
    ('qj_DENSE_OFF_s42.csv',  'DENSE OFF s42',  'DENSE_OFF_s42@16M',  'dense', '20 vessels, ring 0.6, no comm'),
    ('qj_DENSE_COMM_s42.csv', 'DENSE COMM s42', 'DENSE_COMM_s42@16M', 'dense', '20 vessels, ring 0.6, comm@9M'),
    ('qj_DENSE_OFF_s43.csv',  'DENSE OFF s43',  'DENSE_OFF_s43@16M',  'dense', '20 vessels, ring 0.6, no comm'),
    ('qj_DENSE_COMM_s43.csv', 'DENSE COMM s43', 'DENSE_COMM_s43@16M', 'dense', '20 vessels, ring 0.6, comm@9M'),
    ('qj_DENSE_OFF_s44.csv',  'DENSE OFF s44',  'DENSE_OFF_s44@16M',  'dense', '20 vessels, ring 0.6, no comm'),
    ('qj_DENSE_COMM_s44.csv', 'DENSE COMM s44', 'DENSE_COMM_s44@16M', 'dense', '20 vessels, ring 0.6, comm@9M'),
]
AXIS_TITLE = {
    'basis': 'Baseline: MoE-5x, comm OFF vs ON',
    'dimension': 'Message dimension (2–12)',
    'aggregation': 'Message aggregation: nearest-4 vs nearest-1',
    'protocol': 'Communication protocol variants',
    'moe': 'MoE architecture variants',
    'reward': 'Reward design variants',
    'fog': 'Restricted visibility (radar 28 m)',
    'dense': 'Dense traffic (20 vessels, ring 0.6)',
}
EVPAT = re.compile(r'\[([\w@.]+)\].*?goal=\s*([\d.]+)%.*?vColl=\s*([\d.]+)%.*?oColl=\s*([\d.]+)%'
                   r'.*?fuel=\s*([\d.]+)\s+headTravel=\s*([\d.]+)deg.*?epReward=\s*(-?[\d.]+)'
                   r'\s+colregs=\s*([\d.]+)\s+colregsOK=\s*([\d.]+)%')


def load_evals():
    d = {}
    p = os.path.join(SCR, 'metrics_v2.txt')
    if not os.path.exists(p):
        return d
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = EVPAT.search(ln)
        if m:
            d[m.group(1)] = dict(goal=float(m.group(2)),
                                 coll=float(m.group(3)) + float(m.group(4)),
                                 fuel=float(m.group(5)), head=float(m.group(6)),
                                 epr=float(m.group(7)), ok=float(m.group(9)))
    return d


def load(fn):
    p = os.path.join(SCR, fn)
    if not os.path.exists(p):
        return None
    st, rw = [], []
    with open(p, encoding='utf-8') as f:
        next(f, None)
        for ln in f:
            q = ln.strip().split(',')
            if len(q) == 3:
                try:
                    st.append(float(q[0])); rw.append(float(q[1]))
                except ValueError:
                    pass
    if len(st) <= WARMUP + 5:
        return None
    return np.array(st[WARMUP:]), np.array(rw[WARMUP:])


def main():
    ev = load_evals()
    have = []
    for fn, lab, evlab, axis, desc in RUNS:
        r = load(fn)
        if r is None:
            continue
        have.append((fn, lab, evlab, axis, desc, r))
    print(f'학습 곡선 {len(have)}개 발견')

    # ── 1) 개별 곡선 ──
    for fn, lab, evlab, axis, desc, (x, y) in have:
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ps.run_curve(ax, x, y, '#2a8f3a', a=EMA_A, lw=1.5)
        ax.axvline(9e6, color='#bbbbbb', ls='--', lw=0.8)
        ax.set_xlabel('Training steps'); ax.set_ylabel('Average training reward')
        e = ev.get(evlab)
        sub = (f'goal {e["goal"]:.1f}%  coll {e["coll"]:.1f}%  fuel {e["fuel"]:.0f}  '
               f'COLREGs {e["ok"]:.1f}%  epR {e["epr"]:.0f}') if e else '(no eval yet)'
        ax.set_title(f'{lab}  —  {desc}\n{sub}', fontsize=9)
        ax.set_xlim(0, 16.6e6)
        fig.tight_layout()
        safe = lab.replace(' ', '_')
        fig.savefig(os.path.join(IND, f'{safe}.png'), bbox_inches='tight')
        plt.close(fig)

    # ── 2) 축별 묶음 ──
    for axis in sorted(set(h[3] for h in have)):
        runs = [h for h in have if h[3] == axis]
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        cols = cm.viridis(np.linspace(0, 0.88, len(runs)))
        for (fn, lab, evlab, ax_, desc, (x, y)), c in zip(runs, cols):
            ax.plot(x, ps.ema(y, EMA_A), color=c, lw=1.3, label=lab)
        ax.axvline(9e6, color='#bbbbbb', ls='--', lw=0.8)
        ax.set_xlabel('Training steps'); ax.set_ylabel('Average training reward')
        ax.set_title(AXIS_TITLE.get(axis, axis), fontsize=10.5)
        ax.legend(fontsize=7.5, ncol=2 if len(runs) > 6 else 1)
        ax.set_xlim(0, 16.6e6)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'group_{axis}.png'), bbox_inches='tight')
        plt.close(fig)

    # ── 3) 컨택트 시트 ──
    n = len(have); ncol = 6; nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 1.75 * nrow))
    for ax, (fn, lab, evlab, axis, desc, (x, y)) in zip(axes.ravel(), have):
        ax.plot(x, ps.ema(y, EMA_A), color='#2a8f3a', lw=1.0)
        ax.axvline(9e6, color='#cccccc', ls='--', lw=0.6)
        ax.set_title(lab, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.suptitle(f'All training runs ({n})  — average training reward vs steps '
                 '(dashed = communication ON at 9M)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(os.path.join(OUT, '00_contact_sheet.png'), bbox_inches='tight')
    plt.close(fig)

    # ── 4) INDEX.md ──
    with open(os.path.join(OUT, 'INDEX.md'), 'w', encoding='utf-8') as f:
        f.write('# Training curves — all runs\n\n')
        f.write('- `00_contact_sheet.png` — all runs at a glance\n')
        f.write('- `group_<axis>.png` — grouped by experiment axis\n')
        f.write('- `individual/<label>.png` — one figure per run\n\n')
        f.write('NOTE: window-averaged training reward is noisy (synchronized-termination '
                'aliasing). Judge performance from the frozen-eval columns '
                '(goal / coll / epReward).\n\n')
        f.write('| run | axis | configuration | goal% | coll% | fuel | COLREGs% | epReward |\n')
        f.write('|---|---|---|---|---|---|---|---|\n')
        for fn, lab, evlab, axis, desc, _ in have:
            e = ev.get(evlab)
            if e:
                f.write(f'| {lab} | {axis} | {desc} | {e["goal"]:.1f} | {e["coll"]:.1f} | '
                        f'{e["fuel"]:.0f} | {e["ok"]:.1f} | {e["epr"]:.0f} |\n')
            else:
                f.write(f'| {lab} | {axis} | {desc} | – | – | – | – | – |\n')
    print(f'saved 10_CURVES/  (individual {len(have)}, groups {len(set(h[3] for h in have))}, '
          'contact sheet, INDEX.md)')


if __name__ == '__main__':
    main()
