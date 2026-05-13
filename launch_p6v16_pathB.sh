#!/bin/bash
# Phase 1.B-α P6v16 Path B (5/14) — Residual Policy Learning (Silver 2018).
#
# CONTEXT (5/14 session, Path A REJECTED):
#   P6v15 (=P6v14c + reset_actor_bias_idx 5) produced metrics ≈ bit-identical to
#   P6v14c. Gripper bias close-lock hypothesis REJECTED. Iter 0→1 gripper_open
#   -73% in BOTH runs. iter 5-10 cascade collapse to stage4=0.003.
#   Root cause: PPO advantage prefers low-variance grasp basin (8th farming),
#   reward shape engineering exhausted (cap=0/3.0/∞ all collapse). Algorithmic.
#
# RPL DESIGN (Path B, structural fix):
#   actor = ResidualMLPWrapper(bc_actor_frozen, residual_mlp, alpha=0.3)
#     bc_actor:    P6v14a/model_499 actor (256→128→64→6), requires_grad=False
#     residual_mlp: small MLP (28 → 64 → 32 → 6), zero-init final layer
#     alpha = 0.3 → at start residual=0, action=BC. Range ±0.3 max deviation per dim.
#   PPO trains residual + critic + std (NOT bc).
#   By construction: forgetting impossible because bc weights never updated.
#
# Hypothesis (Round 2 evidence: Silver 2018 + Ankile 2025 ResFiT):
#   iter 0: stage4 ≈ 0.36 (matches P6v14c/P6v15 starting policy = P6v14a base)
#   iter 1-10: stage4 STAYS ≥ 0.20 (vs P6v15 collapse to 0.003)
#                — key RPL test: PPO update on RESIDUAL only, BC release path preserved
#   iter 50-499: stage4 improves to 0.50+ as residual learns small place/release corrections
#   FAIL → root cause deeper than forgetting (e.g. critic value func divergence)
#         → pivot to Path D (P6v14a + release BC) or Path E (Diffusion no RL)
#
# Md5 (new — train_ppo.py changed + residual_actor.py added):
#   roarm_stack_env.py   = ff31c5a32e85cc61bec39302a9b739c0 (unchanged)
#   train_ppo.py         = a5fa7482d98fd66b13e4b272454237df (P6v16 residual_mode CLI)
#   residual_actor.py    = f47d16d7dc43339a5cc3d5a549138ea6 (new file)

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

# Path B (RPL) — IMPORTANT: --residual_mode replaces --resume. BC ckpt = P6v14a/model_499.
exec python -u -m roarm_rl.train_ppo \
    --task stack \
    --num_envs 4096 \
    --max_iterations 500 \
    --reward_phase 6 \
    --seed 0 \
    --residual_mode \
    --residual_bc_ckpt $ROARM_B200_ROOT/logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt \
    --residual_alpha 0.3 \
    --residual_hidden 64,32 \
    --entropy_coef 0.003 \
    --episode_length_s 2.0 \
    --curriculum_pregrasp_hover \
    --curriculum_post_grasp_cap \
    --curriculum_spawn_min_r 0.05 \
    --curriculum_spawn_max_r 0.07 \
    --curriculum_xy_thresh 0.05 \
    --curriculum_z_thresh 0.04 \
    --experiment_name p6v16_pathB_RPL_alpha03
