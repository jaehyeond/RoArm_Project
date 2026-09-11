#!/bin/bash
# W10 스톨 가드(실행 중인 셀에 붙임). 사용: ./guard_w10.sh <stage> <attempt> <cell_dir> <python_pid> <timeout_pid> <T0_epoch>
# 진행 = stdout.txt 와 timeline_seed460.json 중 최신 mtime. 300 s 무진행이면 kill. 종료 시 run.log 에 rc/벽시계 기록(run_w10.sh 형식).
cd /home/cgxr/Documents/Robotics/RoArm_Project
OUT=claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix
S=$1; A=$2; D=$3; PPID_=$4; TPID=$5; T0=$6; STALL=300; STALLED=0
log() { echo "[$(date +%H:%M:%S)] $*" >> $OUT/run.log; }
newest() { local m=0; for f in $D/stdout.txt $D/timeline_seed460.json; do [ -f $f ] && { t=$(stat -c %Y $f); [ $t -gt $m ] && m=$t; }; done; echo $m; }
while kill -0 $PPID_ 2>/dev/null; do
  sleep 30; AGE=$(( $(date +%s) - $(newest) ))
  if [ $AGE -gt $STALL ]; then log "stage $S attempt $A STALL: stdout/timeline ${AGE}s 무진행 → kill"; STALLED=1; kill -TERM $PPID_; sleep 10; kill -KILL $PPID_ 2>/dev/null; kill -TERM $TPID 2>/dev/null; break; fi
done
sleep 3; RC=$(cat $D/rc.txt 2>/dev/null || echo "?"); T1=$(date +%s)
[ -f $D/scoop_s1_seed460.json ] && RC=0
[ $STALLED -eq 1 ] && RC=143
log "stage $S attempt $A rc=$RC wall=$(( T1 - T0 ))s stalled=$STALLED (guard_w10)"
