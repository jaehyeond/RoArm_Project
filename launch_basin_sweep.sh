#!/bin/bash
# (alpha') Skill 3 basin-of-attraction sweep — 5/14
#
# Goal: skip scripted Skill 0/1/2 entirely. Force-set env at (P6v14a training
# entry + perturbation). Run Skill 3 (P6v14a model_499) inference.
# Measure where P6v14a's release behavior generalizes.
#
# 6-point grid: dx=[0,+15,+30,+45]mm × dz=[0,+20]mm (selected subset, see chain_skills.py main()).
# Each run: 200 step Skill 3 inference. Total ~6 min after sim_app init.

set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && { echo FAIL_root_unset; exit 1; }
[[ "$(whoami)" != "sogang_jhki" ]] && { echo FAIL_user; exit 1; }
[[ "$(hostname)" != "JHPark-container" ]] && { echo FAIL_host; exit 1; }

# chain_skills.py md5 guard (synced 5/14 morning).
EXPECTED_CHAIN_MD5="2615070a54471cbfe2fea2f40d61b817"
ACTUAL_CHAIN_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/chain_skills.py" | awk '{print $1}')
[[ "$ACTUAL_CHAIN_MD5" != "$EXPECTED_CHAIN_MD5" ]] && { echo "FAIL chain md5: $ACTUAL_CHAIN_MD5 != $EXPECTED_CHAIN_MD5"; exit 1; }
echo "GUARD-OK chain_md5=$ACTUAL_CHAIN_MD5"

eval "$(micromamba shell hook --shell bash)"
micromamba activate "$ROARM_B200_ROOT/envs/isaacsim_5_1"
export OMNI_KIT_ACCEPT_EULA=YES
cd "$ROARM_B200_ROOT/code"

exec python -u -m roarm_rl.chain_skills \
    --basin_sweep \
    --model_path "$ROARM_B200_ROOT/logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt" \
    --basin_steps 200
