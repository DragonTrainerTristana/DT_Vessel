#!/bin/bash
# 2026-10-03: Fig1 배치 f_ 실행 스크립트 정본은 저장소로 옮김 → Python/runs_fig1/2026-10-02_fig1/_run_f.sh (브랜치 feat/fig1-latent 9183a0f~)
#   GitHub 클론만으로 아무 Windows 기계에서 돎(Dropbox 불필요). 이 파일은 옛 서버(/c/work/DT_Vessel_v3)용 바로가기
#   다른 기계: 클론 루트에서  VESSEL_F_AUTO1=1 bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh phase0
exec bash "${VESSEL_F_REPO:-/c/work/DT_Vessel_v3}/Python/runs_fig1/2026-10-02_fig1/_run_f.sh" "$@"
