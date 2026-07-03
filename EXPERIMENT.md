# Light_MoE arm

COLREGs 상황별 완전분리 MoE, 코어 내부 폭 0.44배 — 5코어 합계가 단일망과 등가 파라미터 (3망 합 358,270 vs 단일 364,397).

- 이 브랜치 기본값: `VESSEL_USE_MOE=1`, `VESSEL_MOE_WIDTH=0.44`
- 코어 내부 차원: conv 14/28/28, 레이더 특징 13, hidden 56, fc3 28 (외부 인터페이스 불변)
- 코드는 msgComparision과 동일 — config 기본값 커밋 하나 차이. 베이스 수정은 msgComparision에 반영 후 이 브랜치를 rebase.
- 비교 arm: single(msgComparision 기본) / Thick_MoE(용량 5배) / Light_MoE(등파라미터) — 전문화 효과와 용량 효과의 분리 측정
