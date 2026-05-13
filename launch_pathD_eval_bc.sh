#!/bin/bash
# Path D Phase D.3 v2 (5/15 evening) — Eval release_bc with per-env gripper_q@s capture.
#
# v1 → v2 delta: eval_release_bc.py now records gripper_q at the EXACT step
# `_place_success_flag` rises (per env), so we can split nominal success into
# CLEAN (gripper_q@s < 0.4 rad, true direct-path release) vs counter-path artifact.
#
# v1 result (nominal): 175/256 = 68.36% but 89.1% s≥50 + clean_at_end=0 → suspect.
#
# PASS gate (CLEAN rate, design doc Path D.3):
#   ≥50% CLEAN : publishable
#   ≥30% CLEAN : proceed to subskill expansion
#   10-30%     : BC capacity 확장 or procedural demo pivot
#   <10%       : PATH D FAIL → SkillGen/MimicGen procedural pivot
#
# Md5 verify (BC ckpt + env unchanged; only eval script bumped):
#   roarm_stack_env.py    = ff31c5a32e85cc61bec39302a9b739c0
#   eval_release_bc.py    = 1c9c7c0c4afc920ae1fbea9261d6704e   # bumped v1→v2
#   release_bc.pt         = 688c3b0bd7f6e50b45334e51a571406e

set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && { echo FAIL_root_unset; exit 1; }
[[ "$(whoami)" != "sogang_jhki" ]] && { echo FAIL_user; exit 1; }
[[ "$(hostname)" != "JHPark-container" ]] && { echo FAIL_host; exit 1; }

EXPECTED_ENV_MD5="ff31c5a32e85cc61bec39302a9b739c0"
ACTUAL_ENV_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py" | awk '{print $1}')
[[ "$ACTUAL_ENV_MD5" != "$EXPECTED_ENV_MD5" ]] && { echo "FAIL env md5: $ACTUAL_ENV_MD5"; exit 1; }
EXPECTED_EVAL_MD5="1c9c7c0c4afc920ae1fbea9261d6704e"
ACTUAL_EVAL_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/eval_release_bc.py" | awk '{print $1}')
[[ "$ACTUAL_EVAL_MD5" != "$EXPECTED_EVAL_MD5" ]] && { echo "FAIL eval md5: $ACTUAL_EVAL_MD5"; exit 1; }
EXPECTED_BC_MD5="688c3b0bd7f6e50b45334e51a571406e"
ACTUAL_BC_MD5=$(md5sum "$ROARM_B200_ROOT/data/release_bc.pt" | awk '{print $1}')
[[ "$ACTUAL_BC_MD5" != "$EXPECTED_BC_MD5" ]] && { echo "FAIL bc md5: $ACTUAL_BC_MD5"; exit 1; }
echo "GUARD-OK env_md5=$ACTUAL_ENV_MD5 eval_md5=$ACTUAL_EVAL_MD5 bc_md5=$ACTUAL_BC_MD5"

eval "$(micromamba shell hook --shell bash)"
micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
export OMNI_KIT_ACCEPT_EULA=YES
cd $ROARM_B200_ROOT/code

OUTPUT_DIR="$ROARM_B200_ROOT/logs/roarm_rl/pathD_eval_bc_v2"
mkdir -p "$OUTPUT_DIR"

exec python -u -m roarm_rl.eval_release_bc \
    --bc_ckpt $ROARM_B200_ROOT/data/release_bc.pt \
    --num_envs 256 \
    --num_episodes 1 \
    --curriculum_xy_thresh 0.05 \
    --curriculum_z_thresh 0.04 \
    --episode_length_s 2.0 \
    --seed 0 \
    --output "$OUTPUT_DIR/eval_metrics.pt"
