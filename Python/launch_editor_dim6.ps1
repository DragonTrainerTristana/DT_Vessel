# ============================================================================
# 에디터 직결 학습 (단일 인스턴스): MSG_DIM=6, from-scratch, 통신 ON
#
# sweep(빌드 6병렬)과 달리 에디터는 인스턴스 1개뿐 → dim 하나만 학습 가능.
# 학습 과정을 Unity 에디터에서 눈으로 직접 보려고 사용.
#
# ※ config.py 기본값이 이미 에디터+MSG_DIM6+from-scratch라
#    사실 `python main.py` 만 실행해도 동일. 이 스크립트는 이전 sweep 의
#    환경변수 잔재(VESSEL_USE_EDITOR=0 등)를 확실히 비워 안전하게 고정.
#
# 사용법:
#   1) Unity 에디터에서 학습 씬 열기
#   2) PowerShell:  ./launch_editor_dim6.ps1
#   3) 콘솔에 Unity 연결 대기 메시지가 뜨면 → 에디터에서 ▶Play
#   종료: Ctrl+C  또는  에디터 ■Stop
#
#   다른 차원으로 보고 싶으면 아래 VESSEL_MSG_DIM 값만 바꾸면 됨.
# ============================================================================
Set-Location $PSScriptRoot

# 에디터 + MSG_DIM=6 명시 (config.py 기본값과 동일하지만 잔재 방어용으로 강제)
$env:VESSEL_USE_EDITOR = "1"
$env:VESSEL_MSG_DIM    = "6"

# sweep 전용 환경변수 잔재 제거 (남아 있으면 빌드/병렬/엉뚱한 저장폴더로 샐 수 있음)
"VESSEL_NUM_ENVS","VESSEL_COMM_FOLDER","VESSEL_RUN_STEP","VESSEL_ENV_PATH","VESSEL_BASE_PORT","VESSEL_START_STEP" |
    ForEach-Object { Remove-Item "Env:$_" -ErrorAction SilentlyContinue }

Write-Host "[Editor] USE_EDITOR=1  MSG_DIM=6  from-scratch(LOAD_MODEL=False)  통신 ON" -ForegroundColor Cyan
Write-Host "[Editor] 곧 'Unity 연결 대기' 상태가 되면 에디터에서 ▶Play 누르세요." -ForegroundColor Cyan
python main.py
