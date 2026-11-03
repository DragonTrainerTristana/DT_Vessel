#!/bin/bash
# 2026-11-03 qr4_(시드 46) = 저자 명령 2) 'GPU 2장 이상' 줄 그대로, GPU 0·1 목록(11-03 저자 승인 'GPU 4장 다 쓰게 재시작', 실행기 a9d154b).
# nohup 대신 Task Scheduler(vessel_qr4) 로 띄움 — Claude 세션이 끝나도 안 죽게. 명령·env 는 같음
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
cd /c/work/DT_Vessel_v3 || exit 1
echo "===== $(date '+%F %T') launch qr4_ (HEAD $(git rev-parse --short HEAD)) =====" >> run_qr4.log
bash -c 'bash /c/work/qr_run.sh qr4_ 46 phase1 "0 1" && bash /c/work/qr_run.sh qr4_ 46 extend "0 1"' >> run_qr4.log 2>&1
echo "===== $(date '+%F %T') driver exit rc=$? =====" >> run_qr4.log
