# Thick_MoE arm

COLREGs 상황별 완전분리 MoE, 코어당 단일망과 동일 폭 (총 파라미터 ~5배: 3망 합 1,821,985).

- 이 브랜치 기본값: `VESSEL_USE_MOE=1`, `VESSEL_MOE_WIDTH=1.0`
- 코드는 msgComparision과 동일 — config 기본값 커밋 하나 차이. 베이스 수정은 msgComparision에 반영 후 이 브랜치를 rebase.
- 비교 arm: single(msgComparision 기본) / Thick_MoE(용량 5배) / Light_MoE(등파라미터, 폭 0.44)
