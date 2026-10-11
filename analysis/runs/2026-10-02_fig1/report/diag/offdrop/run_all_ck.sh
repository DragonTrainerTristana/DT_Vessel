#!/bin/bash
cd "$(dirname "$0")"; rm -f _ckdone.txt
g=0
for job in "off_s43.step12M sample" "off_s44.step12M sample" "off_s45.step12M sample" "off_s43 sample" "off_s44 sample" "off_s45 sample" \
           "off_s43.step12M mean" "off_s44.step12M mean" "off_s45.step12M mean" "off_s43 mean" "off_s44 mean" "off_s45 mean"; do
  set -- $job
  bash run_ck.sh $1 $((g % 4)) 2000 $2 &
  g=$((g + 1)); sleep 12
done
wait; echo ALLDONE >> _ckdone.txt
