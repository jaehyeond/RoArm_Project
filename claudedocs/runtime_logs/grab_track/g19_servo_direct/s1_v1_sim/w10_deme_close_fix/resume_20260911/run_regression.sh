#!/bin/bash
# W10 G0: current source, existing sphere params/pile; run only after the lens process exits.
set -u
cd /home/cgxr/Documents/Robotics/RoArm_Project || exit 2
W10_BASE=claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix
W10_REG="$W10_BASE/regression_sphere_resume_20260911"
mkdir "$W10_REG" || exit 2
W10_T0=$(date +%s)
echo "[$(date +%H:%M:%S)] stage regression_sphere_resume_20260911 START max_wall=1200s" >> "$W10_BASE/run.log"
timeout -k 30 1200 /home/cgxr/miniconda3/envs/roarm/bin/python sim_deme_scoop_s1.py \
    --params claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w3_deme_scoop/b_diverge/params_fixnorm_plunge25.json \
    --pile claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz \
    --out "$W10_REG" --seed 460 > "$W10_REG/stdout.txt" 2> "$W10_REG/stderr.txt"
W10_RC=$?
W10_T1=$(date +%s)
echo "[$(date +%H:%M:%S)] stage regression_sphere_resume_20260911 rc=$W10_RC wall=$((W10_T1-W10_T0))s stalled=0" >> "$W10_BASE/run.log"
exit "$W10_RC"
