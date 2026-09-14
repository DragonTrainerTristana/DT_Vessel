# _smoke_fullmoe.py — 격리됨 (2026-09-14)

원위치 `Python/_smoke_fullmoe.py`. 근거는 `docs/PLAN.md` "deprecated" 그룹(2단계) + `.claude/CLAUDE.md` §8.

## 역할

완전분리 MoE(USE_MOE=1, 상황별 코어 5벌) 스모크 검증. A~G 7개 절 — 구조·forward shape·PPO
mirror·others_msg mirror·comm-OFF 불변·상황별 grad 격리·라우팅 효과.

## 격리 이유 — C절이 리팩토링 이전부터 상시 FAIL, 판정 권위 아님

- C절(PPO mirror sum) 은 2026-09-05 `action_raw` 변경 **이전**에 작성됨 — 리팩토링 전 코드로
  재현해도 FAIL(|lp diff| 1.9e-1). 리팩토링이 깬 게 아니라 애초에 낡은 검사.
- D절(others_msg mirror) 도 Mac 에서 numpy 크래시로 안 돎.
- `.claude/CLAUDE.md` §8 이 이미 "낡은 검사이니 판정에 쓰지 말 것"이라 명시. 권위는
  `_verify_ppo_mirror.py`(Windows 전용) · `_verify_comm_mirror.py` — 이 둘로 대체됨.
- `run_repro.sh` preflight 가 이 파일을 호출하는 곳 **0건** (grep 확인, 2026-09-14). 파이프라인
  연결점 없는 고립 스크립트.
- `importlib.reload(config)` 같은 프로세스 내 모듈 전역 교체 부작용이 있던 파일이라, 격리로
  이 부작용이 다른 스크립트에 영향을 줄 여지도 함께 없어짐.

## 되살릴 때

1. `git mv Python/_archive/deprecated_2026-09/_smoke_fullmoe.py Python/`
2. C·D절은 여전히 낡은 가정 기준이므로 **그대로 믿지 말 것** — 되살리는 목적이 "권위 있는
   검증기로 쓰기"라면 `_verify_ppo_mirror.py`/`_verify_comm_mirror.py` 를 먼저 볼 것.
