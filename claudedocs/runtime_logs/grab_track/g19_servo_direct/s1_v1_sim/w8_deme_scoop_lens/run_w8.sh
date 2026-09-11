#!/bin/bash
# W8 순차 실행(GPU 공유 — 동시 1개). 사용: ./run_w8.sh regression c xp50 xm50
# regression = 구 더미 seed 460 + W3b params 그대로(G2). 셀 = 렌즈 더미, params_w8_cell_<셀>.json, 상한 2400 s.
cd /home/cgxr/Documents/Robotics/RoArm_Project
OUT=claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w8_deme_scoop_lens
B=claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w3_deme_scoop/b_diverge
LENS=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz
SPH=claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz
PY=~/miniconda3/envs/roarm/bin/python
for S in "$@"; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "[$(date +%H:%M:%S)] stage $S gpu_free_MiB=$FREE" >> $OUT/run.log
  if [ "$FREE" -lt 3072 ]; then echo "GPU free < 3 GB, abort stage $S" >> $OUT/run.log; continue; fi
  if [ "$S" = "regression" ]; then
    mkdir -p $OUT/regression_sphere
    T0=$(date +%s)
    timeout -k 30 600 $PY sim_deme_scoop_s1.py --params $B/params_fixnorm_plunge25.json --pile $SPH --out $OUT/regression_sphere --seed 460 > $OUT/regression_sphere/stdout.txt 2> $OUT/regression_sphere/stderr.txt
    RC=$?
  else
    mkdir -p $OUT/cell_$S
    T0=$(date +%s)
    timeout -k 30 2400 $PY sim_deme_scoop_s1.py --params $OUT/params_w8_cell_$S.json --pile $LENS --out $OUT/cell_$S --seed 460 > $OUT/cell_$S/stdout.txt 2> $OUT/cell_$S/stderr.txt
    RC=$?
  fi
  echo "[$(date +%H:%M:%S)] stage $S rc=$RC wall=$(( $(date +%s) - T0 ))s" >> $OUT/run.log
done
echo "[$(date +%H:%M:%S)] DONE $*" >> $OUT/run.log
