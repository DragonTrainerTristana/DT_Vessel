"""
전체 그림 재생성 — 파라미터 고정, 데이터는 Figures/_data 사본 사용.
사용법:  python regenerate_all.py
  * 임시 폴더(scratchpad)가 지워져도 _data 사본으로 동일하게 재생성됨
  * 새 학습 결과를 반영하려면 _data에 해당 .log/.csv/metrics_v2.txt를 갱신할 것
"""
import os, subprocess, sys

FIG = os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ)
ENV['PYTHONIOENCODING'] = 'utf-8'
ENV['VESSEL_LOG_DIR'] = os.path.join(FIG, '_data')   # 고정 데이터 경로
ENV['VESSEL_FIG_XMAX'] = '15000000'                  # 곡선 표시 상한 15M

SCRIPTS = [
    'make_ablation_rewards.py',    # 20_REWARD/ablation_reward_*  (ablation 축별 reward 곡선)
    'make_reward_folder.py',       # 20_REWARD/reward_learning_curve*, reward_episode_return
    'plot_all_curves.py',          # 10_CURVES/  (개별 run 아카이브)
    'plot_moe_axis.py',            # 00_MAIN/moe_axis
    'plot_h1_paired.py',           # 00_MAIN/h1_paired
    'plot_nqi.py',                 # 00_MAIN/nqi_comm*
    'plot_h2_dimension.py',        # 00_MAIN/h2_dimension
    'plot_task_return.py',         # 00_MAIN/task_return_*
]

if __name__ == '__main__':
    fails = []
    for s in SCRIPTS:
        p = os.path.join(FIG, s)
        if not os.path.exists(p):
            print(f'[건너뜀] {s} 없음'); continue
        r = subprocess.run([sys.executable, p], cwd=FIG, env=ENV,
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        tag = 'OK ' if r.returncode == 0 else 'FAIL'
        print(f'[{tag}] {s}')
        for ln in (r.stdout or '').strip().splitlines():
            print('      ' + ln)
        if r.returncode != 0:
            fails.append(s)
            print('      ' + (r.stderr or '').strip().splitlines()[-1] if r.stderr else '')
    print('\n완료' if not fails else f'\n실패: {fails}')
