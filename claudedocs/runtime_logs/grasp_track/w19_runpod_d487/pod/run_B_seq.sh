#!/bin/bash
# pod B 본 실행 순차: runB_seed461 → runB_seed462 (사용자 GO 2026-09-17 16:2x KST). 두 셀은 독립 반복이라 앞 셀 rc 와 무관하게 다음 셀을 돌린다(실패 증거 보존).
export LC_ALL=C
PY=/home/cgxr/miniconda3/envs/roarm/bin/python
POD=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/pod
LOG=/workspace/w19_runs; mkdir -p "$LOG"
for s in runB_seed461 runB_seed462; do
  echo "[$(date -u +%FT%TZ)] START $s" >> "$LOG/runB.log"
  "$PY" -B "$POD/run_w19.py" --step "$s" --allow-go-step > "$LOG/$s.runner.out" 2> "$LOG/$s.runner.err"
  echo "[$(date -u +%FT%TZ)] END $s rc=$?" >> "$LOG/runB.log"
done
echo "RUNB_SEQ_DONE" >> "$LOG/runB.log"
