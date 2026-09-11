#!/bin/bash
# W10 G0 회귀(구 더미 + W3b params 그대로, 패치판 스크립트) → 진단 v2 경로 스모크(구 더미). 순차.
cd /home/cgxr/Documents/Robotics/RoArm_Project
OUT=claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix; PY=~/miniconda3/envs/roarm/bin/python
PILE=claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz
echo "[$(date +%H:%M:%S)] stage regression_sphere gpu_free_MiB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" >> $OUT/run.log
T0=$(date +%s); timeout -k 30 900 $PY sim_deme_scoop_s1.py --params claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w3_deme_scoop/b_diverge/params_fixnorm_plunge25.json --pile $PILE --out $OUT/regression_sphere --seed 460 > $OUT/regression_sphere/stdout.txt 2> $OUT/regression_sphere/stderr.txt; RC=$?; T1=$(date +%s)
echo "[$(date +%H:%M:%S)] stage regression_sphere rc=$RC wall=$(( T1 - T0 ))s" >> $OUT/run.log
echo "[$(date +%H:%M:%S)] stage smoke_diag_sphere" >> $OUT/run.log
T0=$(date +%s); timeout -k 30 1200 $PY sim_deme_scoop_s1.py --params $OUT/params_smoke_diag_sphere.json --pile $PILE --out $OUT/smoke_diag_sphere --seed 460 > $OUT/smoke_diag_sphere/stdout.txt 2> $OUT/smoke_diag_sphere/stderr.txt; RC=$?; T1=$(date +%s)
echo "[$(date +%H:%M:%S)] stage smoke_diag_sphere rc=$RC wall=$(( T1 - T0 ))s" >> $OUT/run.log
