# export_onnx.py — 격리됨 (2026-09-14)

원위치 `Python/export_onnx.py`. 근거는 전부 `docs/MAP.md`.

## 격리 이유 — 현행 `networks.py` 와 불일치

- `control_actor.conv1 / conv2 / fc1` 을 참조함(`:25-28`). 현행 `ControlActor` 에 **이 세 속성이 없음**.
  `conv1`/`conv2` 는 `RadarEncoder`(`networks.py:207-208`) 안에만 있음
- 즉 실행하면 깨짐. 사실상 죽은 경로임

## 잘못된 가정 3가지

- **입력 차원** — 입력을 `obs: [batch, 373]` 으로 가정함(`:32`). 현행 관측 계약은 **369D**
- **COLREGS 슬라이스** — `COLREGS_SIZE` 로 슬라이스함(`:40-41`). 현행 `COLREGS_SIZE` 값은 **0**
- **프레임스택** — 3프레임 스택을 **같은 프레임 3복제로 위조**함(`:37-38`). 실제 프레임스택이 아님

## 참고 — 현재 배포 경로는 ONNX 가 아님

- C# 전체에 Barracuda/Sentis/NNModel 참조 **0건**. Unity 안에서 추론하지 않음
- 실동작 경로 = `Python/main.py:915-949` 가 `VESSEL_LOAD_MODEL=1` 에서
  `torch.load` → `load_state_dict` 로 정책을 올리고 mlagents_envs 로 Unity 를 구동함
- 출력 파일명이 `<PROJECT_ROOT>/models/VesselNavigation_16M.onnx` 로 고정돼 있음(`:17-18`)
- `:9` 에서 `sys.path.insert(0, 자기 디렉터리)` 함 — 위치를 옮기면 그 전제도 달라짐

## 다시 만들 때

**이 파일을 기준으로 삼지 말 것.** 위 4개 가정(속성명·373D·COLREGS 슬라이스·프레임 복제)이
전부 현행과 어긋나 있어, 고쳐 쓰는 것보다 현행 `networks.py` · `config.py` 관측 계약에서
새로 쓰는 쪽이 맞음.

## 격리 시점 확인

- 격리 직전 grep 결과 `export_onnx` 를 import 하는 Python 코드 **0건** (참조는 `docs/` 와
  `.claude/agents/qa-engineer.md` 의 언급뿐)
