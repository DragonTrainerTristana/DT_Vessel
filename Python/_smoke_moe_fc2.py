"""
[DEPRECATED 2026-06-26] 옛 MoE = '공유 radar backbone + fc2부터 상황별 분기(expert_fc2/head_fc3)'.
완전분리 MoE로 교체됨(MessageActor/ControlActor/Critic 각각 상황별 코어 5벌, radar_encoder 포함 독립).
이 스모크가 검증하던 expert_fc2/head_fc3/critic one-hot 구조는 더 이상 존재하지 않는다.

→ 새 권위 테스트: _smoke_fullmoe.py (구조·forward·PPO mirror·grad 격리[radar 포함]·라우팅).
   PPO mirror 일반(sum/attention/comm-OFF)은 _verify_ppo_mirror.py.
이 파일은 옛 이름 호환을 위해 _smoke_fullmoe를 그대로 실행한다.
"""
import sys
import _smoke_fullmoe

if __name__ == '__main__':
    sys.exit(_smoke_fullmoe.main())
