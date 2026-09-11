#!/bin/bash
# W8 옵션 E 실행기: params_w8E_cell_<셀>.json (폐합 sync 1 ms + 물림 가드 3 N) → cell_E_<셀>/. 사용: ./run_w8E.sh c xp50 xm50
# (run_w8A.sh 의 rc 로깅 버그 — echo 안 $(date) 가 $? 를 덮음 — 수정: RC 를 먼저 저장)
cd /home/cgxr/Documents/Robotics/RoArm_Project
OUT=claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w8_deme_scoop_lens
LENS=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz
PY=~/miniconda3/envs/roarm/bin/python
for S in "$@"; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "[$(date +%H:%M:%S)] stage E_$S gpu_free_MiB=$FREE" >> $OUT/run.log
  if [ "$FREE" -lt 3072 ]; then echo "GPU free < 3 GB, abort stage E_$S" >> $OUT/run.log; continue; fi
  mkdir -p $OUT/cell_E_$S; T0=$(date +%s)
  timeout -k 30 2700 $PY sim_deme_scoop_s1.py --params $OUT/params_w8E_cell_$S.json --pile $LENS --out $OUT/cell_E_$S --seed 460 > $OUT/cell_E_$S/stdout.txt 2> $OUT/cell_E_$S/stderr.txt
  RC=$?; T1=$(date +%s)
  echo "[$(date +%H:%M:%S)] stage E_$S rc=$RC wall=$(( T1 - T0 ))s" >> $OUT/run.log
done
echo "[$(date +%H:%M:%S)] DONE E $*" >> $OUT/run.log
