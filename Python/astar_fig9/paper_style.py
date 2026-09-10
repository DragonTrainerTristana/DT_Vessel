"""astar_fig9/paper_style.py — plotting/paper_style.py 의 shim (2026-09-10).

예전엔 바이트 동일한 사본이었음. 사본은 어느 쪽을 고쳤는지가 또 쟁점이 되므로 정본 하나만 두고 여기서 끌어옴.
make_fig9_from_eval.py 가 `sys.path.insert(0, HERE); import paper_style` 로 부르는 것 그대로 동작.
(sys.path 로 하면 이 파일 자신이 'paper_style' 로 잡혀 순환이 되므로 경로로 직접 로드.)
"""
import importlib.util as _ilu
import os as _os

_SRC = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'plotting', 'paper_style.py')
_spec = _ilu.spec_from_file_location('_plotting_paper_style', _SRC)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('_')})
del _ilu, _os, _SRC, _spec, _mod
