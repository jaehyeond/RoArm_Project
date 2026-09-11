#!/bin/bash
# seed 3회 순차 실행 (GPU 공유 — 동시 1개). 사용: ./run_seeds.sh 460 461 462
cd /home/cgxr/Documents/Robotics/RoArm_Project
OUT=claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w3_deme_scoop
for S in "$@"; do
  timeout -k 30 420 ~/miniconda3/envs/roarm/bin/python sim_deme_scoop_s1.py --params $OUT/params_provisional_plunge10.json --out $OUT --seed $S > $OUT/stdout_seed$S.txt 2> $OUT/stderr_seed$S.txt
  echo "seed $S rc=$?" >> $OUT/run_seeds.log
done
echo DONE >> $OUT/run_seeds.log
