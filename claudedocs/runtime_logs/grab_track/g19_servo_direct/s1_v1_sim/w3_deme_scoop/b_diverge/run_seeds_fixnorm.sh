#!/bin/bash
# W3b ② seed 순차 실행(GPU 공유 — 동시 1개). 사용: ./run_seeds_fixnorm.sh 460 461 462
cd /home/cgxr/Documents/Robotics/RoArm_Project
OUT=claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w3_deme_scoop/b_diverge
mkdir -p $OUT/scoop_fixnorm
for S in "$@"; do
  timeout -k 30 420 ~/miniconda3/envs/roarm/bin/python sim_deme_scoop_s1.py --params $OUT/params_fixnorm_plunge25.json --out $OUT/scoop_fixnorm --seed $S > $OUT/scoop_fixnorm/stdout_seed$S.txt 2> $OUT/scoop_fixnorm/stderr_seed$S.txt
  echo "seed $S rc=$?" >> $OUT/scoop_fixnorm/run.log
done
echo DONE >> $OUT/scoop_fixnorm/run.log
