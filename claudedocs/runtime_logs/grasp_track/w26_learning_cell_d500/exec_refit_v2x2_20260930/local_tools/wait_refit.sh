#!/bin/bash
# 로컬: refit pod(1대) chain.log 의 새 사건(FAIL/CELL_RUNNER_EXIT/CELL2_RUNNER_EXIT)만 기다린다. EVERY 초 간격, MAXS 초 뒤 REARM.
H=root@213.173.108.87; P=19620; PAT="FAIL|NOT_OK|CELL_RUNNER_EXIT|CELL2_RUNNER_EXIT"
sshp() { ssh -i $HOME/.ssh/id_ed25519 -p $P -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes $H "$1" 2>/dev/null; }
BASE=$(sshp "grep -c -E '$PAT' /workspace/w26_smokes/chain.log"); echo "baseline=$BASE"; t0=$(date +%s)
while true; do
  out=$(sshp "export LC_ALL=C; tail -n 1 /workspace/w26_smokes/chain.log; for f in /home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w26_learning_cell_d500/exec_refit_v2x2_20260930/runs/refit_*/cell_01/cell.stdout.txt; do [ -f \$f ] && tail -n 1 \$f | cut -c1-120; done; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader")
  n=$(sshp "grep -c -E '$PAT' /workspace/w26_smokes/chain.log")
  echo "[$(date +%H:%M:%S)] $(echo "$out" | tr '\n' ' ' | cut -c1-400)"
  [ -n "$n" ] && [ "$n" -gt "${BASE:-0}" ] && { echo "EVENT"; exit 0; }
  [ $(( $(date +%s) - t0 )) -ge ${MAXS:-3300} ] && { echo "REARM"; exit 0; }
  sleep ${EVERY:-600}
done
