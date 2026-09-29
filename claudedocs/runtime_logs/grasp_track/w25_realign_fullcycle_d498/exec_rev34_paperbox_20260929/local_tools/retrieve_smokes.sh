#!/bin/bash
# 로컬 실행. pod 의 smoke 3종 산출 + /workspace/w25_smokes 로그를 exec/runs/<tag>/ 로 rsync(_obj 포함) 하고 판정 2개를 로컬에서 계산한다.
# 사용: TAG=podA_4090 SSH_HOST=root@IP SSH_PORT=NNNN bash retrieve_smokes.sh
set -u
: "${TAG:?}"; : "${SSH_HOST:?}"; : "${SSH_PORT:?}"
EXEC=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/exec_rev34_paperbox_20260929
PY=/home/cgxr/miniconda3/envs/roarm/bin/python
SSHO="-i $HOME/.ssh/id_ed25519 -p $SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes"
mkdir -p "$EXEC/runs/$TAG"
rsync -a -q -e "ssh $SSHO" "$SSH_HOST:/workspace/w25_out/runs/$TAG/" "$EXEC/runs/$TAG/" && echo RSYNC_SMOKES_OK
rsync -a -q -e "ssh $SSHO" "$SSH_HOST:/workspace/w25_smokes/" "$EXEC/runs/$TAG/pod_smokes_logs/" && echo RSYNC_SMOKE_LOGS_OK
rsync -a -q -e "ssh $SSHO" "$SSH_HOST:/workspace/w25_bootstrap/" "$EXEC/runs/$TAG/pod_bootstrap_logs/" && echo RSYNC_BOOTSTRAP_LOGS_OK
for s in smoke_01_import300 smoke_02_sphere_regression smoke_03_settle_full; do
  printf '%s state=%s rc=%s wall=%s\n' "$s" "$($PY -c "import json;d=json.load(open('$EXEC/runs/$TAG/$s/RUN_STATUS.json'));print(d['state'])" 2>/dev/null)" \
    "$($PY -c "import json;d=json.load(open('$EXEC/runs/$TAG/$s/EXECUTION_RECEIPT.json'));print(d.get('rc'))" 2>/dev/null)" \
    "$($PY -c "import json;d=json.load(open('$EXEC/runs/$TAG/$s/EXECUTION_RECEIPT.json'));print(d.get('wall_s'))" 2>/dev/null)"
done
R=$(ls "$EXEC/runs/$TAG/smoke_02_sphere_regression"/*.json 2>/dev/null | grep -v -E "RECEIPT|STATUS" | head -1)
[ -n "$R" ] && $PY "$EXEC/pod/check_sphere_regression.py" "$R" "$EXEC/runs/$TAG/smoke_02_sphere_regression/G0_CHECK.json" | grep -E '"pass"|n_in_cavity|mass_g|wall_seconds|v_particle' 
N=$(ls "$EXEC/runs/$TAG/smoke_03_settle_full"/*.npz 2>/dev/null | head -1)
[ -n "$N" ] && $PY "$EXEC/pod/settle_speed_ratio.py" "$N" "$EXEC/runs/$TAG/smoke_03_settle_full/SETTLE_SPEED.json" | grep -A4 '"settle"' | grep -E "wall_s|sim_s|wall_per_sim_s"
