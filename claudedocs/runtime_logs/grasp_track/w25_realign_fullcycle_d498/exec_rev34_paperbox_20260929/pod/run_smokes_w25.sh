#!/bin/bash
# pod 측 smoke 3종 순차 실행 — 앞 단계 rc 0 일 때만 다음. 본 실행(run_paperbox_full_cycle)은 여기 없음(GO 후 별도).
# 사용: bash run_smokes_w25.sh <COMMANDS_w25_pod*.json>
export LC_ALL=C
PY=/home/cgxr/miniconda3/envs/roarm/bin/python
EXEC=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/exec_rev34_paperbox_20260929
CMD=${1:?COMMANDS json}
LOG=/workspace/w25_smokes; mkdir -p "$LOG"
for s in smoke_import_300 smoke_sphere_regression smoke_settle_full; do
  echo "[$(date -u +%FT%TZ)] START $s" >> "$LOG/smokes.log"
  "$PY" -B "$EXEC/pod/run_w25.py" --commands "$CMD" --step "$s" > "$LOG/$s.runner.out" 2> "$LOG/$s.runner.err"
  rc=$?
  echo "[$(date -u +%FT%TZ)] END $s rc=$rc" >> "$LOG/smokes.log"
  if [ "$rc" -ne 0 ]; then echo "STOP_AT $s rc=$rc" >> "$LOG/smokes.log"; exit "$rc"; fi
done
echo "ALL_SMOKES_OK" >> "$LOG/smokes.log"
