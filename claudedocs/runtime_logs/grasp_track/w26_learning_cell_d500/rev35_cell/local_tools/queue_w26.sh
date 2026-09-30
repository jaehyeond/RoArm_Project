#!/bin/bash
# pod 측 순차 대기열(09-30, 5090·4090 재고 0 → V2·V4 를 같은 장비에서 순차): chain.log 에 CELL_RUNNER_EXIT 가 찍힐 때까지 기다린 뒤
# 동결 꾸러미의 다음 COMMANDS 로 셀 1회. smoke·G0 는 같은 pod 에서 이미 통과(장비 동일). 재시도 0. 사용: bash queue_w26.sh <다음 COMMANDS json>
export LC_ALL=C
PY=/home/cgxr/miniconda3/envs/roarm/bin/python
R=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w26_learning_cell_d500/rev35_cell
NEXT=${1:?COMMANDS json}; LOG=/workspace/w26_smokes/chain.log
while ! grep -q CELL_RUNNER_EXIT $LOG; do sleep 30; done
TAG=$($PY -c "import json;print(json.load(open('$NEXT'))['pod_tag'])")
echo "QUEUE_GO $TAG $(date -u +%FT%TZ)" >> $LOG
$PY -B $R/pod/run_w26.py --commands $NEXT --step cell --allow-go-step > /workspace/w26_runs/cell2.runner.out 2> /workspace/w26_runs/cell2.runner.err
echo "CELL2_RUNNER_EXIT $? $TAG $(date -u +%FT%TZ)" >> $LOG
