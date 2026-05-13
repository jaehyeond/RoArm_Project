#!/bin/bash
# Phase 1.B-α P6v15 (5/14) — P6v14c minimal diff: reset_actor_bias_idx=5 (gripper).
#
# CONTEXT (5/14 session, post-P6v14c FAIL analysis):
#   P6v14c iter 0 (P6v14a/model_499 resume): stage4=0.3653, gripper_open=0.7835
#     → starting policy good. iter 1: stage4=0.3446, gripper_open=0.21 (-74% in 1 iter),
#     grasped=0.6696 (+730%). iter 10: stage4=0.0037. PPO 1-iter catastrophic forgetting.
#   Round 2 research (5/14 morning, 6 agents) confirmed: NVIDIA Isaac Lab 9/14 community
#     manipulation repos = PPO from scratch (NVIDIA-default). PPO is NOT dead. Our 4 fail
#     likely = impl gap, not algo limit.
#
# ROOT CAUSE HYPOTHESIS (5/14):
#   P6v14a/model_499 ckpt's actor.6.bias[5] (gripper output) likely close-biased (positive).
#   P6v14c resume did NOT include --reset_actor_bias_idx 5 → starting policy already
#   skewed toward grasp → iter 1 PPO update lock-in to grasp farming. Same root cause
#   that triggered P6v5→P6v6 fix when --reset_actor_bias_idx flag was first introduced.
#
# MINIMAL DIFF (vs P6v14c, single-variable isolation):
#   ✅ ADD: --reset_actor_bias_idx 5 (gripper output, restores P(open)=P(close)=50/50)
#   ✅ ALL OTHER FLAGS IDENTICAL TO P6v14c (post_grasp_cap=3.0, pregrasp_hover, annulus 0.05-0.07)
#
# SANITY GATE (compare to P6v14c metrics.csv):
#   iter 0:  stage4 ≈ 0.37 ± 0.05 AND gripper_open ≈ 0.50 (was 0.78 — lower after bias reset)
#   iter 1:  gripper_open > 0.40 (was 0.21) ← key forgetting test
#   iter 10: stage4 > 0.10 (was 0.004) ← key recovery test
#   iter 50: stage4 > 0.25 (was 0.011) ← cleared escape from farm
#   FAIL → root cause is NOT actor bias. Pivot to Path B (RPL frozen BC + residual).
#
# Md5 (unchanged from P6v14c — code sync verified 5/14):
#   roarm_stack_env.py = ff31c5a32e85cc61bec39302a9b739c0
#   train_ppo.py        = 7b3a8e2b0e463ab0d0f5983fe102b8ee

set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && { echo FAIL_root_unset; exit 1; }
[[ "$(whoami)" != "sogang_jhki" ]] && { echo FAIL_user; exit 1; }
[[ "$(hostname)" != "JHPark-container" ]] && { echo FAIL_host; exit 1; }

EXPECTED_ENV_MD5="ff31c5a32e85cc61bec39302a9b739c0"
ACTUAL_ENV_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py" | awk '{print $1}')
[[ "$ACTUAL_ENV_MD5" != "$EXPECTED_ENV_MD5" ]] && { echo "FAIL env md5: $ACTUAL_ENV_MD5 != $EXPECTED_ENV_MD5"; exit 1; }
EXPECTED_TRAIN_MD5="7b3a8e2b0e463ab0d0f5983fe102b8ee"
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
    --resume $ROARM_B200_ROOT/logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt \
    --reset_std 2.0 \
    --reset_actor_bias_idx 5 \
    --entropy_coef 0.003 \
    --episode_length_s 2.0 \
    --curriculum_pregrasp_hover \
    --curriculum_post_grasp_cap \
    --curriculum_spawn_min_r 0.05 \
    --curriculum_spawn_max_r 0.07 \
    --curriculum_xy_thresh 0.05 \
    --curriculum_z_thresh 0.04 \
    --experiment_name p6v15_pathA_bias_reset_resumeP6v14a
