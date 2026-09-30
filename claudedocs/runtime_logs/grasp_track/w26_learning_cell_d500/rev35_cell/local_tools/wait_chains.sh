#!/bin/bash
# 로컬: 두 pod 의 chain.log·셀 러너 상태를 EVERY 초마다 보고, **시작 뒤 새로 생긴** 사건(실패·셀 종료) 또는 MAXS 초 경과 시 끝낸다.
declare -A HOST=([pod1]=root@162.43.172.177 [pod2]=root@157.157.221.30); declare -A PORT=([pod1]=10528 [pod2]=56787)
declare -A BASE
PAT="FAIL|NOT_OK|CELL_RUNNER_EXIT|CELL2_RUNNER_EXIT"
sshp() { ssh -i $HOME/.ssh/id_ed25519 -p ${PORT[$1]} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes ${HOST[$1]} "$2" 2>/dev/null; }
for p in pod1 pod2; do BASE[$p]=$(sshp $p "grep -c -E '$PAT' /workspace/w26_smokes/chain.log"); done
echo "baseline pod1=${BASE[pod1]} pod2=${BASE[pod2]}"
t0=$(date +%s)
while true; do
  ev=""
  for p in pod1 pod2; do
    out=$(sshp $p "export LC_ALL=C; tail -n 1 /workspace/w26_smokes/chain.log; for f in /home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w26_learning_cell_d500/rev35_cell/runs/pod*/cell_01/cell.stdout.txt; do tail -n 1 \$f | cut -c1-120; done; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader")
    n=$(sshp $p "grep -c -E '$PAT' /workspace/w26_smokes/chain.log")
    echo "[$(date +%H:%M:%S)] $p | $(echo "$out" | tr '\n' ' ' | cut -c1-480)"
    [ -n "$n" ] && [ "$n" -gt "${BASE[$p]:-0}" ] && ev="$ev $p"
  done
  [ -n "$ev" ] && { echo "EVENT:$ev"; exit 0; }
  [ $(( $(date +%s) - t0 )) -ge ${MAXS:-1500} ] && { echo "REARM"; exit 0; }
  sleep ${EVERY:-300}
done
