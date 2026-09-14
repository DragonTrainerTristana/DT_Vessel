# inspect_channel_freeze.py — 격리됨 (2026-09-14)

원위치 `Python/inspect_channel_freeze.py`. 근거는 `docs/PLAN.md` "deprecated" 그룹(2단계).

## 역할

1회성 진단 스크립트. 체크포인트를 열어 통신 채널 파라미터(msg_out·v_proj·fc2 msg 슬라이스)가
init 이후 움직였는지 검사 — "직렬 zero-init → grad 항등 0 → 1M step 후에도 그대로 0" 가설 확인용.

## 격리 이유 — 1회성 진단, 고립

- import 하는 코드 0건 / 이 파일이 import 하는 프로젝트 모듈 0건(torch·sys 뿐).
- `run_repro.sh`·`smoke_mac.sh` 가 호출하는 곳 0건(grep 확인, 2026-09-14).
- 2026-09-14 `__main__` 가드가 이미 추가돼 있어 import 시 부작용(즉시 실행)은 이미 해소됨
  (`docs/PLAN.md` "deprecated → _archive" 근거 표).
- 가설 자체는 `networks.py`의 zero-init 기각 설계 결정(전부 ×0.1 소진폭 초기화로 대체,
  `.claude/CLAUDE.md` §4 "설계 결정" 참고)으로 이미 코드에 반영·해소된 과거 진단.

## 되살릴 때

`git mv Python/_archive/deprecated_2026-09/inspect_channel_freeze.py Python/` — 특별한 경로
의존성 없음(체크포인트 경로는 CLI 인자로 받음), sys.path 보정 불필요.
