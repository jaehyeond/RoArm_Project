#!/bin/bash
# Phase 1.B-α P6v14a (5/12) — Phase 0a pre-grasp init (Option α).
#
# WHY P6v14 FAILED (final iter 999):
#   jackpot_fire 0.0000 / 1000 iter / 800M steps. Agent hovers at xy_offset=0.044
#   (just inside xy_thresh=0.05) but z_offset=0.053 (> z_thresh=0.04). Sponge held
#   5cm above table, gripper hard-closed (open_rate 0.065 FLAT). 6th farming pattern:
#   "boundary hover" at the threshold boundary, exactly as user warned in P6v13 session.
#
# P6v14a CHANGES vs P6v14:
#   (1) curriculum_pregrasp=True — IK pose with TCP +5cm above target, gripper q=0.8
#       (closed > grasp_thresh 0.4), _grasped=True latched. _update_grasp_attach pins
#       sponge to TCP each step. Agent's ONLY task: open gripper → sponge falls 5cm
#       → stage 4 success_now (xy<0.05 AND z<0.04 AND open AND stable) fires.
#   (2) curriculum_disable_nearzone_cap REMOVED — cap KEPT (d<0.1 cap to 2.0).
#       In Phase 0a sponge starts at d≈0.05 < 0.1 → cap immediately active. Hover at
#       start = 2.0×200 = 400 reward. Release path 5 + 8×190 = 1525 reward. +281% margin.
#   (3) curriculum_xy/z_thresh 0.05/0.04 (P6v14 same) — relaxed for first-fire feasibility.
#   (4) curriculum_spawn_min_r/max_r OMITTED — pregrasp branch overrides sponge spawn.
#
# IK PRE-COMPUTE (roarm_kinematics.ik_dls, err=0.30mm):
#   target_above = (0.280, -0.0435, +0.0614) world
#   joints_rad = [-0.1541, +0.4109, +2.0177, +0.2213, 0.0, 0.8(gripper override)]
#   gripper=0.8 (vs IK 0.524=30°) ensures q > grasp_thresh 0.4 reliably.
#
# RESUME: P6v14/model_999. Recent policy with transport skill at xy=0.044
#   (boundary stuck). Pre-grasp removes hover obstacle (sponge already at target xy).
#   reset_std=1.5 + entropy_coef=0.001 same hyperparams.
#
# SANITY GATE:
#   iter 1-5: jackpot_fire_rate > 0.001 (≥4 fires per 4096 envs).
#   iter 10-20: jackpot_fire_rate > 0.05 (≥200 fires).
#   If iter 5 fails → critical: exploration NOT solved by structural fix → consider
#   ζ (Pure RL intractable, pivot to BC+RL hybrid).
#
# Md5: roarm_stack_env.py = bc3e17967cccc97c96601a12f18efb7d
#      train_ppo.py        = 21675a050b810295b64bcae812fe976e

set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && { echo FAIL_root_unset; exit 1; }
[[ "$(whoami)" != "sogang_jhki" ]] && { echo FAIL_user; exit 1; }
[[ "$(hostname)" != "JHPark-container" ]] && { echo FAIL_host; exit 1; }

EXPECTED_ENV_MD5="bc3e17967cccc97c96601a12f18efb7d"
ACTUAL_ENV_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py" | awk '{print $1}')
[[ "$ACTUAL_ENV_MD5" != "$EXPECTED_ENV_MD5" ]] && { echo "FAIL env md5: $ACTUAL_ENV_MD5 != $EXPECTED_ENV_MD5"; exit 1; }
EXPECTED_TRAIN_MD5="21675a050b810295b64bcae812fe976e"
ACTUAL_TRAIN_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/train_ppo.py" | awk '{print $1}')
[[ "$ACTUAL_TRAIN_MD5" != "$EXPECTED_TRAIN_MD5" ]] && { echo "FAIL train md5: $ACTUAL_TRAIN_MD5 != $EXPECTED_TRAIN_MD5"; exit 1; }
echo "GUARD-OK env_md5=$ACTUAL_ENV_MD5 train_md5=$ACTUAL_TRAIN_MD5"

eval "$(micromamba shell hook --shell bash)"
micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
export OMNI_KIT_ACCEPT_EULA=YES
cd $ROARM_B200_ROOT/code

exec python -u -m roarm_rl.train_ppo \
    --task stack \
    --num_envs 4096 \
    --max_iterations 500 \
    --reward_phase 6 \
    --seed 0 \
    --resume $ROARM_B200_ROOT/logs/roarm_rl/p6v14_curriculum_p0_resumeP6v13/model_999.pt \
    --reset_std 1.5 \
    --entropy_coef 0.001 \
    --episode_length_s 2.0 \
    --curriculum_pregrasp \
    --curriculum_xy_thresh 0.05 \
    --curriculum_z_thresh 0.04 \
    --experiment_name p6v14a_pregrasp_resumeP6v14
