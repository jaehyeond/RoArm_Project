#!/bin/bash
# Phase 1.B-α P6v14 (5/12) — Curriculum (Option B) Phase 0.
#
# CRITIQUE OF P6v6→v13 PURE SHAPING APPROACH:
#   7 reward-shape iterations created 5 farming local opts but produced 0 jackpot_fire
#   over 800M steps. Stage 4 joint AND prob ≈ 0 from random π → PPO never sees release
#   signal → no gradient. Shape fixes don't address exploration; curriculum does.
#
# P6v14 CHANGES vs P6v13 (env code, identity-preserving for legacy R1-R4 default):
#   (1) cfg: curriculum_spawn_min_r / max_r — sponge spawned in annulus around target
#       instead of R1-R4. min_r=0.08 > xy_thresh=0.05 prevents iter-0 trivial jackpot.
#   (2) cfg: curriculum_xy_thresh / z_thresh — relax stage-3/4 thresholds in Phase 0 to
#       make success_now (xy<0.05 AND z<0.04 AND gripper_open AND stable) reachable.
#   (3) cfg: curriculum_disable_nearzone_cap — Phase 0 disables P6v12 stage-2 d<0.1 cap
#       so transport gradient is unblocked into the release zone. Re-enabled in Phase 1/2.
#
# PHASE 0 NUMERICAL MARGINS (200 step ep, 8/step stage 4 sustained, 5 jackpot):
#   Close-hover grasped d=0.08 (just outside target):  5.86 × 200 = 1172
#   Stage-3 close-hover d=0.04 (V2 cap 3.0 still active): 3.0 × 200 = 600
#   Release path (success at step 50):  5 + 8×150 = 1205 → DOMINANT (+33mm margin over hover)
#   Spawn d∈[0.08, 0.15], xy_offset min = 0.08 > 0.05 thresh → no iter-0 trigger.
#
# RESUME: P6v13/model_999 (latest, grasp_frac 0.865 strong; zone avoidance neutralized by
#   curriculum spawn near target = no zone to avoid). reset_std=1.5 + entropy_coef=0.001
#   matches P6v13 hyperparams. NO bias reset (P6v13 grasp skill kept; curriculum changes
#   landscape, PPO must re-explore which gripper state pays off in new annulus regime).
#
# SANITY GATE iter 5: jackpot_fire_rate > 0.001 (≥4 fires across 4096 envs in single iter).
#   P6v13 final jackpot_fire = 0/1000 iter. Phase 0 should fire ≥ once in first few iters
#   (release-feasible regime). If still 0 after iter 20 → escalate: add --reset_actor_bias_idx 5.
#
# Md5 verify: roarm_stack_env.py = 453acf68eac6a778c16eebb81c4131ef
#             train_ppo.py        = 2a2b7e93e4932f49f2d0b73e439096f6

set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && { echo FAIL_root_unset; exit 1; }
[[ "$(whoami)" != "sogang_jhki" ]] && { echo FAIL_user; exit 1; }
[[ "$(hostname)" != "JHPark-container" ]] && { echo FAIL_host; exit 1; }

EXPECTED_ENV_MD5="453acf68eac6a778c16eebb81c4131ef"
ACTUAL_ENV_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py" | awk '{print $1}')
[[ "$ACTUAL_ENV_MD5" != "$EXPECTED_ENV_MD5" ]] && { echo "FAIL env md5 mismatch: $ACTUAL_ENV_MD5 != $EXPECTED_ENV_MD5"; exit 1; }
EXPECTED_TRAIN_MD5="2a2b7e93e4932f49f2d0b73e439096f6"
ACTUAL_TRAIN_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/train_ppo.py" | awk '{print $1}')
[[ "$ACTUAL_TRAIN_MD5" != "$EXPECTED_TRAIN_MD5" ]] && { echo "FAIL train md5 mismatch: $ACTUAL_TRAIN_MD5 != $EXPECTED_TRAIN_MD5"; exit 1; }
echo "GUARD-OK env_md5=$ACTUAL_ENV_MD5 train_md5=$ACTUAL_TRAIN_MD5"

eval "$(micromamba shell hook --shell bash)"
micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
export OMNI_KIT_ACCEPT_EULA=YES
cd $ROARM_B200_ROOT/code

exec python -u -m roarm_rl.train_ppo \
    --task stack \
    --num_envs 4096 \
    --max_iterations 1000 \
    --reward_phase 6 \
    --seed 0 \
    --resume $ROARM_B200_ROOT/logs/roarm_rl/p6v13_v2_etav2_v3_velrelax_resumeP6v12/model_999.pt \
    --reset_std 1.5 \
    --entropy_coef 0.001 \
    --episode_length_s 2.0 \
    --curriculum_spawn_min_r 0.08 \
    --curriculum_spawn_max_r 0.15 \
    --curriculum_xy_thresh 0.05 \
    --curriculum_z_thresh 0.04 \
    --curriculum_disable_nearzone_cap \
    --experiment_name p6v14_curriculum_p0_resumeP6v13
