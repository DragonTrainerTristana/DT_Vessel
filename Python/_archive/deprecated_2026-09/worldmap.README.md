# worldmap_extract.py · worldmap_geometry.json — 격리됨 (2026-09-14)

원위치 `Python/worldmap_extract.py` · `Python/worldmap_geometry.json`.
역할 = `WorldMap.unity` YAML → 항해 가능 영역 지오메트리 추출(`worldmap_geometry.json` 생산).

## 격리 이유 — 생산자만 있고 소비자 0건

- `worldmap_geometry.json` 을 **읽는 코드 0건**. 참조는 생산자 자신의 두 줄
  (`worldmap_extract.py:6` docstring, `:173` 출력 경로)뿐임
- `worldmap_extract` 를 import 하는 코드 **0건**
- 확인 범위 = 프로젝트 루트 전체(`.git` 제외), 대소문자 무시:
  - Python — 0건
  - C# (`*.cs`) — 0건
  - 설정·빌드류 (`json/asmdef/sh/bat/yaml/yml/txt/meta/cfg/toml/ini`) — 0건

## 씬은 별개로 살아 있음

- `Assets/Scenes/WorldMap.unity` 는 정상 사용 중.
  `ProjectSettings/EditorBuildSettings.asset:12` 에 빌드 씬으로 등록돼 있음
- 이 격리는 **추출 스크립트와 그 출력 json 만** 대상임. 씬과는 무관
- `worldmap_extract.py:16-17` 이 `__file__` 기준 상대경로로 씬을 찾음 —
  이 폴더로 옮겨져 깊이가 달라졌으므로 되살릴 때 경로부터 고쳐야 함

## 좌표 출처 기록은 남아 있음

- 씬에서 뽑은 실측 좌표는 `Python/corridor_run.py:3-5` docstring 에 기록돼 있음
  (부산 −82066,−24438 / 대만 −79863,−18304 / 직선 6,517.6 유닛)
- 즉 이 파일들을 격리해도 좌표 근거는 소실되지 않음
