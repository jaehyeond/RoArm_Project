#!/bin/bash
# pod 측 연결 실행(사용자 RunPod 승인 09-30 범위): 부트스트랩 BOOTSTRAP_OK 대기 → smoke 2종(rc0 연쇄) → G0 3항목(발산 없음·268~362알·servo_stall)
# 판정 → 통과 시에만 셀 GO. pop(5 m/s) 항목은 G0 규칙이 아니므로 관측만(W19 선례). 사용: bash chain_w26.sh <COMMANDS json>
export LC_ALL=C
PY=/home/cgxr/miniconda3/envs/roarm/bin/python
R=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w26_learning_cell_d500/rev35_cell
CMD=${1:?COMMANDS json}; LOG=/workspace/w26_smokes; mkdir -p $LOG /workspace/w26_runs
for i in $(seq 1 120); do grep -q BOOTSTRAP_OK /workspace/w26_bootstrap/bootstrap.stdout.txt 2>/dev/null && break; sleep 10; done
grep -q BOOTSTRAP_OK /workspace/w26_bootstrap/bootstrap.stdout.txt || { echo "BOOTSTRAP_NOT_OK $(date -u +%FT%TZ)" >> $LOG/chain.log; exit 1; }
echo "BOOTSTRAP_OK_SEEN $(date -u +%FT%TZ)" >> $LOG/chain.log
bash $R/pod/run_smokes_w26.sh $CMD || { echo "SMOKES_FAIL $(date -u +%FT%TZ)" >> $LOG/chain.log; exit 2; }
TAG=$($PY -c "import json;print(json.load(open('$CMD'))['pod_tag'])")
J=$(ls $R/runs/$TAG/smoke_02_sphere_regression/*.json | grep -v -E 'RECEIPT|STATUS' | head -1)
$PY - "$J" >> $LOG/chain.log 2>&1 <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); n = d["capture"]["n_in_cavity"]; st = d["door"]["stops"]
ok = d["diverged"] is False and 268 <= n <= 362 and bool(st) and all(s["reason"] == "servo_stall" for s in st)
print("G0", n, [s["reason"] for s in st], "v_max", d["pops"]["v_particle_max_m_s"], "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || { echo "G0_FAIL $(date -u +%FT%TZ)" >> $LOG/chain.log; exit 3; }
echo "G0_PASS -> GO cell $(date -u +%FT%TZ)" >> $LOG/chain.log
$PY -B $R/pod/run_w26.py --commands $CMD --step cell --allow-go-step > /workspace/w26_runs/cell.runner.out 2> /workspace/w26_runs/cell.runner.err
echo "CELL_RUNNER_EXIT $? $(date -u +%FT%TZ)" >> $LOG/chain.log
