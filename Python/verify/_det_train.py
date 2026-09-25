"""_det_train.py — GPU 비트동일 A/B 용 결정론 래퍼 (2026-09-26).

왜: 학습기는 결정론 플래그를 안 건다(운영 배치는 cuDNN 원자연산 때문에 같은 GPU 에서도 실행마다 비트가 다름 —
    .claude/CLAUDE.md §8-1). 코드 변경이 수학을 안 바꿨는지 GPU 에서 확인하려면 옛 코드·새 코드 *둘 다* 같은 결정론
    커널로 돌려 state_dict·Adam·곡선 CSV 를 비교해야 한다. 이 래퍼가 그 플래그만 걸고 vessel_gym_train.main() 을 부른다.
쓰는 법:  python verify/_det_train.py <Python 루트> <vessel_gym_train.py 인자...>
    <Python 루트> = 옛 코드(C:/work/DT_Vessel/Python) 또는 새 코드(worktree) 의 Python 디렉터리.
    CUBLAS_WORKSPACE_CONFIG 는 cuBLAS 핸들이 생기기 전에 있어야 하므로 import 전에 건다.
검증 범위 밖: 운영 커널 자체(비결정론) — 여기서 증명하는 건 '같은 op 열 + 결정론 커널 ⇒ old == new'.
"""
import os
import sys

root, argv = sys.argv[1], sys.argv[2:]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.chdir(root)
sys.path.insert(0, root)
import vessel_gym_train as T  # noqa: E402  (config 가 torch 보다 먼저 import 되는 운영 순서 그대로)
import torch  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)   # cumsum(compute_own_future) 등은 경고만
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sys.argv = ['vessel_gym_train.py'] + argv
T.main()
