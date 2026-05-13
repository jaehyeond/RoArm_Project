#!/bin/bash
# Phase 1.B-α P6v14c (5/13 evening) — Phase 0a' pre-grasp HOVER bridge.
#
# CONTEXT (5/13 evening session, post-P6v14b FAIL):
#   P6v14b Phase 0b (cold-start full chain from 80-150mm annulus) FAIL: stage4_success
#   0.0 across 1000 iter / 4096 env / 4.1M episodes / jackpot 0. Catastrophic forgetting
#   of release skill (gripper_open 0.578 → 0.066 in 5 iter). 8th farming pattern =
#   "stage 2 grasp-hold outside cap zone (d>0.1, stage 2=5.28/step)".
#
# PHASE 0a' DESIGN (P6v14c):
#   Bridge P6v14a (sponge in hand at target → release only) ↔ P6v14b (cold full chain).
#   - TCP at P6v14a IK pose (5cm above target), gripper OVERRIDE to OPEN q=0.0
#   - Sponge on table near target via annulus 0.05-0.07 (5-7cm from target xy)
#   - _grasped=False, _was_grasped=False (sponge NOT in hand at start)
#   - Agent task: descend (5cm) → close gripper → grasp → drag (2cm) → release
#   - P6v14a release-aware policy resumed; descent+grasp+drag = ~25-step new skill
#
# ANTI-FARMING: --curriculum_post_grasp_cap (NEW P6v14c env flag).
#   Stage 2 r = post_grasp_cap_value (default 3.0) ALWAYS when is_grasped.
#   Overrides nearzone_cap (broader: d any). Kills P6v14b's 8th "grasp+move-away" farm.
#   cap=3.0 > stage 1 max (2.0) → PPO grasp gradient preserved (+1.0/step jump).
#
# PRE-LAUNCH MATH (C1 protocol, ALL paths computed):
#   Initial: TCP +5cm above sponge, d_tcp_sponge≈0.078, d_sponge_target=0.05-0.07
#   reach_r(0.078) = 2×(1-tanh(0.39)) = 1.26
#
#   Path A  (hover, no descent):   1.26×200          = 252
#   Path A' (descent, no grasp):   16.3 + 380        = 396
#   Path A''(grasp + hold, cap=3): 16.3 + 189×3.0    = 583   ← P6v14b's farm, BOUNDED
#   Path B  (full release):        16.3 + 42 + 16.5
#                                  + 28 + 150jackpot
#                                  + 168×8 = 1344    = 1597  ← target
#
#   Margin B/A'' = 1597/583 = +174% SAFE (C1 protocol pass, ≥+100%).
#   Gradient positive at every transition: +144 (A→A'), +187 (A'→A''), +1014 (A''→B).
#
# TRIVIAL JACKPOT AVOIDED:
#   annulus_min=0.05 = on_target_xy_thresh strict boundary (d<0.05 strict fail).
#   sponge_z=table center = target_z (d_z=0, on_target_z passes). Need d_xy fail only.
#
# OOD RESUME RISK:
#   P6v14a obs distrib: TCP+5cm, gripper q=0.8, sponge in hand. Phase 0a' obs:
#   TCP+5cm, gripper q=0.0, sponge on table. Mitigated via entropy 0.003 + reset_std 2.0.
#
# SANITY GATE (DUAL, C1 권장):
#   iter 50:  stage4_success > 0.05 AND grasped_frac > 0.10  (descent+grasp working)
#   iter 200: stage2_grasp_frac < 0.85 (anti-farming check, post_grasp_cap working)
#   iter 500: stage4_success > 0.30 AND grasped_frac > 0.20  (release path emerging)
#   FAIL → BC pretraining pivot (B1 backup) OR Phase 0a' tighter (annulus 0.04-0.05)
#
# Md5:
#   roarm_stack_env.py = ff31c5a32e85cc61bec39302a9b739c0 (post_grasp_cap + pregrasp_hover)
#   train_ppo.py        = 7b3a8e2b0e463ab0d0f5983fe102b8ee (new CLI flags)

set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && { echo FAIL_root_unset; exit 1; }
[[ "$(whoami)" != "sogang_jhki" ]] && { echo FAIL_user; exit 1; }
[[ "$(hostname)" != "JHPark-container" ]] && { echo FAIL_host; exit 1; }

EXPECTED_ENV_MD5="ff31c5a32e85cc61bec39302a9b739c0"
ACTUAL_ENV_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py" | awk '{print $1}')
[[ "$ACTUAL_ENV_MD5" != "$EXPECTED_ENV_MD5" ]] && { echo "FAIL env md5: $ACTUAL_ENV_MD5 != $EXPECTED_ENV_MD5 (P6v14c env edits not synced)"; exit 1; }
EXPECTED_TRAIN_MD5="7b3a8e2b0e463ab0d0f5983fe102b8ee"
ACTUAL_TRAIN_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/train_ppo.py" | awk '{print $1}')
[[ "$ACTUAL_TRAIN_MD5" != "$EXPECTED_TRAIN_MD5" ]] && { echo "FAIL train md5: $ACTUAL_TRAIN_MD5 != $EXPECTED_TRAIN_MD5 (P6v14c CLI flags not synced)"; exit 1; }
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
    --entropy_coef 0.003 \
    --episode_length_s 2.0 \
    --curriculum_pregrasp_hover \
    --curriculum_post_grasp_cap \
    --curriculum_spawn_min_r 0.05 \
    --curriculum_spawn_max_r 0.07 \
    --curriculum_xy_thresh 0.05 \
    --curriculum_z_thresh 0.04 \
    --experiment_name p6v14c_phase0a_prime_hover_resumeP6v14a
