#!/bin/bash
cd "$(dirname "$0")"; rm -f _jdone.txt
bash run_judge.sh off_s43 0 3000 & sleep 15
bash run_judge.sh comm_s43 1 3000 & sleep 15
bash run_judge.sh off_s44 2 3000 & sleep 15
bash run_judge.sh comm_s44 3 3000 & sleep 15
bash run_judge.sh off_s45 2 3000 & sleep 15
bash run_judge.sh comm_s45 3 3000 &
wait; echo ALLDONE >> _jdone.txt
