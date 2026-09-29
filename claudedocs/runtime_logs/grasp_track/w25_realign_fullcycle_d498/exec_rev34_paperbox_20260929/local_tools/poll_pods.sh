#!/bin/bash
# 로컬 10분 폴러(setsid nohup 로 분리). pod 별 RUN_STATUS.state·마지막 결정줄·GPU 를 runs/POLL_LOG.txt 에 한 줄씩 남기고,
# 터미널 상태(completed_rc0/failed_nonzero_rc/halted_timeout_not_success/killed_after_grace/aborted_*/runner_exception)면 runs/<tag>.DONE 를 쓴다.
# pgrep 자기매치 금지: 프로세스 확인은 RUN_STATUS 의 child_pid 로 kill -0.
EXEC=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/exec_rev34_paperbox_20260929
LOG=$EXEC/runs/POLL_LOG.txt
declare -A HOST=([podA_4090]=root@157.157.221.29 [podB_pro6000x2]=root@152.236.142.250)
declare -A PORT=([podA_4090]=31486 [podB_pro6000x2]=15541)
while true; do
  alldone=1
  for tag in podA_4090 podB_pro6000x2; do
    [ -f "$EXEC/runs/$tag.DONE" ] && continue
    alldone=0
    SSHO="-i $HOME/.ssh/id_ed25519 -p ${PORT[$tag]} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o BatchMode=yes"
    out=$(ssh $SSHO ${HOST[$tag]} "export LC_ALL=C; R=/workspace/w25_out/runs/$tag/run_01; python3 - <<'PY' 2>/dev/null
import json,os,signal
d=json.load(open('$R/RUN_STATUS.json'.replace('\$R','/workspace/w25_out/runs/$tag/run_01')))
st=d['state']; pid=d.get('child_pid'); alive=None
if pid:
    try: os.kill(int(pid),0); alive=True
    except OSError: alive=False
print('state='+st, 'child_alive='+str(alive), 'elapsed_s='+str(d.get('elapsed_s')), 'rc='+str(d.get('rc')), 'wall_s='+str(d.get('wall_s')))
PY
grep -E '▣ 결정|abort|발산|Traceback|Error|WALL_CAP' /workspace/w25_out/runs/$tag/run_01/run_paperbox_full_cycle.stdout.txt 2>/dev/null | tail -n 1 | cut -c1-160
tail -n 1 /workspace/w25_out/runs/$tag/run_01/run_paperbox_full_cycle.stdout.txt 2>/dev/null | cut -c1-120
tail -n 1 /workspace/w25_out/runs/$tag/run_01/run_paperbox_full_cycle.stderr.txt 2>/dev/null | cut -c1-120
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '" 2>/dev/null)
    rc=$?
    line="$(date '+%m-%d %H:%M:%S') $tag ssh_rc=$rc | $(echo "$out" | tr '\n' ' | ')"
    echo "$line" >> "$LOG"
    if echo "$out" | grep -q -E 'state=(completed_rc0|failed_nonzero_rc|halted_timeout_not_success|killed_after_grace|aborted_hash_or_stray|aborted_before_launch|runner_exception)'; then
      echo "$line" > "$EXEC/runs/$tag.DONE"; echo "DONE $tag" >> "$LOG"
    fi
  done
  [ $alldone -eq 1 ] && { echo "ALL_DONE $(date '+%m-%d %H:%M:%S')" >> "$LOG"; exit 0; }
  sleep 600
done
