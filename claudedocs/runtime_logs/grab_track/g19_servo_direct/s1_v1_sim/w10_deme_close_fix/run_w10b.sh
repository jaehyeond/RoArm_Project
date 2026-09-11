#!/bin/bash
# W10 셀 실행기(b: 진행 = stdout·timeline 최신 mtime). 사용: ./run_w10.sh <stage> [max_wall_s]   (stage = diag_c | DE_c | DE_E1e8_c → params_w10_<stage>.json → cell_<stage>/)
# 규칙: GPU free ≥ 3 GB 아니면 60 s 대기 반복(최대 30 회), 300 s 무진행(stdout 갱신 없음) 스톨 가드 + 1 회 재시도, rc/벽시계를 run.log 에 기록.
cd /home/cgxr/Documents/Robotics/RoArm_Project
OUT=claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix
LENS=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz
PY=~/miniconda3/envs/roarm/bin/python
S=$1; MAXW=${2:-5400}; STALL=300
log() { echo "[$(date +%H:%M:%S)] $*" >> $OUT/run.log; }
for attempt in 1 2; do
  for k in $(seq 1 30); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$FREE" -ge 3072 ] && break
    log "stage $S attempt $attempt gpu_free_MiB=$FREE < 3072 → 60 s 대기 ($k/30)"; sleep 60
  done
  if [ "$FREE" -lt 3072 ]; then log "stage $S attempt $attempt ABORT gpu_free_MiB=$FREE after 30 min wait"; exit 3; fi
  D=$OUT/cell_$S; mkdir -p $D; T0=$(date +%s)
  log "stage $S attempt $attempt START gpu_free_MiB=$FREE max_wall=${MAXW}s"
  timeout -k 30 $MAXW $PY sim_deme_scoop_s1.py --params $OUT/params_w10_$S.json --pile $LENS --out $D --seed 460 > $D/stdout.txt 2> $D/stderr.txt &
  TPID=$!; STALLED=0
  while kill -0 $TPID 2>/dev/null; do
    sleep 30
    M=0; for f in $D/stdout.txt $D/timeline_seed460.json; do [ -f $f ] && { t=$(stat -c %Y $f); [ $t -gt $M ] && M=$t; }; done; AGE=$(( $(date +%s) - M ))
    if [ $AGE -gt $STALL ]; then
      log "stage $S attempt $attempt STALL: stdout/timeline ${AGE}s 무진행 → kill"; STALLED=1
      pkill -TERM -P $TPID; kill -TERM $TPID 2>/dev/null; sleep 10; pkill -KILL -P $TPID 2>/dev/null; kill -KILL $TPID 2>/dev/null
      break
    fi
  done
  wait $TPID; RC=$?; T1=$(date +%s)
  log "stage $S attempt $attempt rc=$RC wall=$(( T1 - T0 ))s stalled=$STALLED"
  if [ $STALLED -eq 1 ] && [ $attempt -eq 1 ]; then mv $D ${D}_try1_stall; log "stage $S partial → cell_${S}_try1_stall/ ; 재시도"; continue; fi
  exit $RC
done
