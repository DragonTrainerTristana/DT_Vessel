# ─────────────────────────────────────────────────────────────────────────────
# common_env() 정본 = 이 파일 하나 (2026-09-24). 예전엔 run_repro.sh 와 smoke_mac.sh 가 각자
#   사본을 들고 있어 export 하나 고칠 때마다 두 번 고쳐야 했다(누락 시 조용히 다른 설정).
# run_repro.sh 와 smoke_mac.sh 가 `source "$HERE/common_env.sh"` 로 가져다 쓴다.
# 값은 config.py 끝 `YUGIOH` 표와 1:1 — 전부 config 기본값이지만 *명시* export 한다
#   (로그·스냅샷만 보고 설정을 알 수 있게). run_repro.sh preflight 의 드리프트 검사 대상.
#   단 아래 `${VAR:-기본}` 형태 3개(DYN_PROFILE·OBSTACLES·RADAR_RANGE)는 의도된 실험 축이라
#   drift-check 의 `names`(= 기본값과 같은가) 목록에서 빠져 있다. 대신 preflight 의 'override 축'
#   검사가 *실제 적용값* 을 대조하고(config 가 그 값을 못 읽으면 FAIL) 기본값과 다르면 ★ 로 찍는다
#   — 셸에 남아 있던 옛 export 가 배치를 조용히 바꾸지 못하게.
# ─────────────────────────────────────────────────────────────────────────────
common_env() {
  export VESSEL_USE_ATTENTION=1
  export VESSEL_CENTRAL_CRITIC=1
  export VESSEL_STATE_RECON_COEF=0.05
  export VESSEL_USE_MOE=1
  export VESSEL_MOE_SHARED=1
  export VESSEL_MOE_WIDTH=1.0
  export VESSEL_SHARED_ENCODER=all
  export VESSEL_RADAR_ACT=leaky
  export VESSEL_RADAR_HEAD=bottleneck
  export VESSEL_RADAR_BOTTLENECK_CH=8
  export VESSEL_MSG_LN=1
  export VESSEL_MSG_TOKEN_GAIN=8.0
  export VESSEL_CLIP_PER_MODULE=1
  export VESSEL_MSG_L2=0.0002
  export VESSEL_POS_GROUND=1
  export VESSEL_COMM_RANGE=300
  export VESSEL_MAX_PARTNERS=4
  # ★2026-09-24 레이더 사거리 — 바깥에서 준 값을 보존(기본 56 = YUGIOH 값, 안 주면 불변).
  #   용량반응 스윕(설계 스펙 `docs/superpowers/specs/2026-09-19-dyn-profile-imo-design.md` §8-5,
  #   RADAR_RANGE ∈ {56,84,112,168}) 용. obs 정규화(dist/range−0.5)가 같이 바뀌므로 값마다 trunk 를
  #   따로 학습할 것 — 갈래끼리 섞지 말 것(스냅샷·check_branch 가 묶음 안 일치를 강제).
  #   DYN_PROFILE 과 같이 드리프트 검사 names 대상은 아니지만, preflight 'override 축' 검사가
  #   적용값을 대조·표시한다(run_repro.sh preflight 의 'override 축'). 이 축 자체는 저자 확인 대기 — 되돌리려면
  #   아래 줄을 `export VESSEL_RADAR_RANGE=56` 으로 고정하고 names 에 'RADAR_RANGE' 를 되돌리면 됨.
  export VESSEL_RADAR_RANGE="${VESSEL_RADAR_RANGE:-56}"
  export VESSEL_COLREGS_MODE=unity
  export VESSEL_SIM_COLREGS_COEF=0.45
  export VESSEL_INTENT_K=3
  export VESSEL_THREAT_COEF=0
  export VESSEL_GOAL_COMM_COEF=0
  export VESSEL_INTENT_COEF=0
  export VESSEL_ROLE_COMM_COEF=0
  export VESSEL_COMM_CONSUMER_COEF=0
  export VESSEL_RECON_EMA_FLOOR=0
  export VESSEL_AGG_MODE=sum
  export VESSEL_MSG_GAIN=1.0
  export VESSEL_TIMEOUT_BOOTSTRAP=0
  export VESSEL_MSG_GATE_APPLY=0
  # ★2026-09-21 동역학 프로필·시나리오 — 바깥에서 준 값을 보존(기본 agile/grid3x3 = 비트동일).
  #   imo 배치는 VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none 을 밖에서 주고, 별도 VESSEL_CKPT_DIR/VESSEL_OUT_DIR 을 쓴다.
  #   preflight 드리프트 검사(names) 대상이 아니다 — 의도된 override 이므로. 학습기·check_branch 가 갈래 간 일치를 강제한다.
  export VESSEL_DYN_PROFILE="${VESSEL_DYN_PROFILE:-agile}"
  export VESSEL_OBSTACLES="${VESSEL_OBSTACLES:-grid3x3}"
  # ★2026-09-25 의도·역할 통신 구조(COMM_EXT) — 배치 단위 실험 축(바깥 값 보존, 기본 0 = 비트동일).
  #   1 이면 attention 토큰에 파트너 상태·역할·명령 20차원이 붙고 k/v 가 MLP → 새 trunk 필요(run_repro 가 접두어 x_ 로 분리).
  #   팔별 내용(COMM_FIELDS·COMM_LATENT·PARTNER_RANGE·AUX_LOSS_SCALE)은 run_repro.sh comm_variant_env 가 런마다 export.
  #   스펙: docs/superpowers/specs/2026-09-25-comm-intent-design.md
  export VESSEL_COMM_EXT="${VESSEL_COMM_EXT:-0}"
}
