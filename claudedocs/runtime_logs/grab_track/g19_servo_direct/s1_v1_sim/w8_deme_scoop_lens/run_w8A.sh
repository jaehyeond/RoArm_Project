#!/bin/bash
# W8 옵션 A 실행기: params_w8A_cell_<셀>.json (dt_sync_close_s 1 ms) → cell_A_<셀>/. 사용: ./run_w8A.sh c xp50 xm50
cd /home/cgxr/Documents/Robotics/RoArm_Project
OUT=claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w8_deme_scoop_lens
LENS=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz
PY=~/miniconda3/envs/roarm/bin/python
for S in "$@"; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "[$(date +%H:%M:%S)] stage A_$S gpu_free_MiB=$FREE" >> $OUT/run.log
  if [ "$FREE" -lt 3072 ]; then echo "GPU free < 3 GB, abort stage A_$S" >> $OUT/run.log; continue; fi
  mkdir -p $OUT/cell_A_$S; T0=$(date +%s)
  timeout -k 30 2700 $PY sim_deme_scoop_s1.py --params $OUT/params_w8A_cell_$S.json --pile $LENS --out $OUT/cell_A_$S --seed 460 > $OUT/cell_A_$S/stdout.txt 2> $OUT/cell_A_$S/stderr.txt
  echo "[$(date +%H:%M:%S)] stage A_$S rc=$? wall=$(( $(date +%s) - T0 ))s" >> $OUT/run.log
done
echo "[$(date +%H:%M:%S)] DONE A $*" >> $OUT/run.log
