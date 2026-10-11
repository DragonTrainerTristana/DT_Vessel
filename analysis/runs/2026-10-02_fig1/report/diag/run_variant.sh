#!/bin/bash
# run_variant.sh <name> <gpu> <alloc_conf>  — 단일 DAgger 700결정, VRAM 추적
source "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/vram/env_f.sh"
W=$(cygpath -m "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/vram")
export CUDA_VISIBLE_DEVICES=$2 PYTORCH_CUDA_ALLOC_CONF="$3"
/c/Users/OSH/anaconda3/envs/mltest/python.exe -u "$W/measure2.py" "$W/$1.pt" --decisions 700 --log_every 100 --value_updates 1 --rollout 4 > "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/vram/$1.log" 2>&1
echo "$1 rc=$? conf=$3 :: $(grep -E '^\[vram\] (END|ABORT)' "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/vram/$1.log" | tail -1) :: $(grep '^\[dagger\] dec=' "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/vram/$1.log" | tail -1 | awk '{print $3, $NF}')"
