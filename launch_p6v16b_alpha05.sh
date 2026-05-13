#!/bin/bash
# Phase 1.B-α P6v16b (5/14 evening) — RPL alpha sweep, conservative residual α=0.05.
#
# CONTEXT (5/14 P6v16 alpha=0.3 partial FAIL):
#   P6v16 (alpha=0.3) iter 0 stage4=0.36 (starting policy good) → iter 10 = 0.003
#   (RPL did NOT prevent forgetting). iter 499 = 0.0271 (vs P6v14c 0.0105 = 2.6×
#   improvement = late recovery signal).
#   Diagnosis: alpha=0.3 lets residual ±0.3 per dim — enough capacity for PPO to
#   effectively override BC. Need much smaller residual band to keep BC anchor
#   stronger during early PPO updates.
#
# HYPOTHESIS (this run α=0.05):
#   alpha=0.05 → residual ±0.05 per dim (action range ~tanh ~±1). BC dominates
#   action ~95%. PPO can only nudge — forgetting prevented by construction.
#   PASS gate (iter 10 stage4 ≥ 0.20): RPL is the right framework → α sweep
#   continues + 1000 iter long train. FAIL (< 0.05) but alpha=0.10 also FAIL
#   → Path D entry.
#
# Md5 (same as P6v16 — only CLI args differ):
#   roarm_stack_env.py   = ff31c5a32e85cc61bec39302a9b739c0
#   train_ppo.py         = a5fa7482d98fd66b13e4b272454237df
#   residual_actor.py    = f47d16d7dc43339a5cc3d5a549138ea6

set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && { echo FAIL_root_unset; exit 1; }
[[ "$(whoami)" != "sogang_jhki" ]] && { echo FAIL_user; exit 1; }
[[ "$(hostname)" != "JHPark-container" ]] && { echo FAIL_host; exit 1; }

EXPECTED_ENV_MD5="ff31c5a32e85cc61bec39302a9b739c0"
ACTUAL_ENV_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py" | awk '{print $1}')
[[ "$ACTUAL_ENV_MD5" != "$EXPECTED_ENV_MD5" ]] && { echo "FAIL env md5: $ACTUAL_ENV_MD5"; exit 1; }
EXPECTED_TRAIN_MD5="a5fa7482d98fd66b13e4b272454237df"
ACTUAL_TRAIN_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/train_ppo.py" | awk '{print $1}')
[[ "$ACTUAL_TRAIN_MD5" != "$EXPECTED_TRAIN_MD5" ]] && { echo "FAIL train md5: $ACTUAL_TRAIN_MD5"; exit 1; }
EXPECTED_RES_MD5="f47d16d7dc43339a5cc3d5a549138ea6"
ACTUAL_RES_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/policies/residual_actor.py" | awk '{print $1}')
[[ "$ACTUAL_RES_MD5" != "$EXPECTED_RES_MD5" ]] && { echo "FAIL residual md5: $ACTUAL_RES_MD5"; exit 1; }
echo "GUARD-OK env_md5=$ACTUAL_ENV_MD5 train_md5=$ACTUAL_TRAIN_MD5 residual_md5=$ACTUAL_RES_MD5"

eval "$(micromamba shell hook --shell bash)"
micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
export OMNI_KIT_ACCEPT_EULA=YES
cd $ROARM_B200_ROOT/code

# P6v16b — RPL alpha=0.05 (conservative residual capacity).
exec python -u -m roarm_rl.train_ppo \
    --task stack \
    --num_envs 4096 \
    --max_iterations 500 \
    --reward_phase 6 \
    --seed 0 \
    --residual_mode \
    --residual_bc_ckpt $ROARM_B200_ROOT/logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt \
    --residual_alpha 0.05 \
    --residual_hidden 64,32 \
    --entropy_coef 0.003 \
    --episode_length_s 2.0 \
    --curriculum_pregrasp_hover \
    --curriculum_post_grasp_cap \
    --curriculum_spawn_min_r 0.05 \
    --curriculum_spawn_max_r 0.07 \
    --curriculum_xy_thresh 0.05 \
    --curriculum_z_thresh 0.04 \
    --experiment_name p6v16b_pathB_RPL_alpha005
