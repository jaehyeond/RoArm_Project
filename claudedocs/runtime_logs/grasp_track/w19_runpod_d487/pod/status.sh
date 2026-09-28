#!/bin/bash
# pod 상태 한눈에: GPU · 러너 상태 · sim stdout 꼬리. 사용: bash status.sh <attempt_dir>
ATT=${1:?attempt dir}
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
echo "--- RUN_STATUS ---"; cat "$ATT/RUN_STATUS.json" 2>/dev/null | head -20
echo "--- stdout tail ---"; tail -n 8 "$ATT"/*.stdout.txt 2>/dev/null
echo "--- stderr tail ---"; tail -n 5 "$ATT"/*.stderr.txt 2>/dev/null
echo "--- processes ---"; pgrep -af "sim_w13_full_cycle|sim_deme_scoop_s1|run_w19" || echo "(none)"
