#!/bin/bash
# 2026-11-03 qr5_(시드 47) = 저자 명령 2) 'GPU 2장 이상' 줄 그대로, GPU 2·3 목록(11-03 저자 승인, 실행기 a9d154b). qr4_ 첫 smoke 끝난 뒤 띄움.
# nohup 대신 Task Scheduler(vessel_qr5) 로 띄움 — Claude 세션이 끝나도 안 죽게. 명령·env 는 같음
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
cd /c/work/DT_Vessel_v3 || exit 1
echo "===== $(date '+%F %T') launch qr5_ (HEAD $(git rev-parse --short HEAD)) =====" >> run_qr5.log
bash -c 'bash /c/work/qr_run.sh qr5_ 47 phase1 "2 3" && bash /c/work/qr_run.sh qr5_ 47 extend "2 3"' >> run_qr5.log 2>&1
echo "===== $(date '+%F %T') driver exit rc=$? =====" >> run_qr5.log
