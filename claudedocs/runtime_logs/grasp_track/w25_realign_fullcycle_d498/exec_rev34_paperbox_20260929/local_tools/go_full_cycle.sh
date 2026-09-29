#!/bin/bash
# 로컬 실행 → pod 에서 본 실행을 setsid nohup 으로 기동(GO 뒤에만). 러너 하드 상한 115,200 s 가 바깥 watchdog.
# 사용: TAG=podA_4090 CMD=COMMANDS_w25_podA_4090.json SSH_HOST=root@IP SSH_PORT=NNNN bash go_full_cycle.sh
set -u
: "${TAG:?}"; : "${CMD:?}"; : "${SSH_HOST:?}"; : "${SSH_PORT:?}"
EXEC=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/exec_rev34_paperbox_20260929
SSHO="-i $HOME/.ssh/id_ed25519 -p $SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes"
ssh $SSHO "$SSH_HOST" "export LC_ALL=C PYTHONDONTWRITEBYTECODE=1; mkdir -p /workspace/w25_runs; cd /workspace/w25_runs && setsid nohup /home/cgxr/miniconda3/envs/roarm/bin/python -B $EXEC/pod/run_w25.py --commands $EXEC/$CMD --step run_paperbox_full_cycle --allow-go-step > /workspace/w25_runs/run_01.runner.out 2> /workspace/w25_runs/run_01.runner.err < /dev/null & echo GO_LAUNCHED_RUNNER_PID \$!; sleep 20; cat /workspace/w25_runs/run_01.runner.out; echo ---; cat /workspace/w25_out/runs/$TAG/run_01/RUN_STATUS.json 2>/dev/null | head -12"
