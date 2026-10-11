#!/bin/bash
# run_trunk_variant.sh <name> <gpu> <alloc_conf|DEFAULT>
source "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/vram/env_f.sh"
W=$(cygpath -m "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/vram")
export CUDA_VISIBLE_DEVICES=$2 VESSEL_BC_TEACHER=vo56h150 VESSEL_BC_COEF=1.0 VESSEL_BC_DECAY_DEC=3000000
[ "$3" = DEFAULT ] || export PYTORCH_CUDA_ALLOC_CONF="$3"
/c/Users/OSH/anaconda3/envs/mltest/python.exe -u "$W/measure_trunk.py" "$1" "$W" > "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/vram/$1.log" 2>&1
echo "$1 rc=$? conf=$3 :: $(grep -E '^\[vram\] (END|ABORT)' "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/vram/$1.log" | tail -1) :: $(grep -E '^\[ON\] dec=' "/c/Users/OSH/AppData/Local/Temp/2/claude/C--Users-OSH-Dropbox-Private-Paper-Project-0702-NewVessel-Assets-Scripts/9e297c7a-7e0d-477f-98d7-1f664196ab38/scratchpad/vram/$1.log" | tail -1 | cut -c1-140)"
