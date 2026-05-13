#!/bin/bash
# Phase 1.B-α P6v14b (5/13 evening) — Phase 0b cold-start short-transport full chain.
#
# CONTEXT (5/13 evening session):
#   P6v14a Phase 0a (pre-grasp init) DECISIVE SUCCESS: stage4_success 0.778, gripper_open
#   0.578, jackpot first fires (0.0044) after 7 failed iterations. Bootstrap signal
#   established. Now remove training-wheel scaffolding: agent must learn full chain
#   reach → grasp → transport → release from cold start with short distance to target.
#
# CROSS-VERIFICATION (5/13 evening, 3 agents):
#   A1 Manipulation: Phase 0a→0b catastrophic forgetting risk HIGH (P6v14a learned
#     release only, not full chain). grasped_frac 0.39 = some grasp skill survived.
#   A2 Sim2Real: Isaac Lab 공식 Stack = BC only, NVIDIA 자체도 RL stacking 어렵게 봄.
#     Reverse curriculum (Florensa 2017, CASHER ICLR 2024) = Phase 0a 정합.
#   C1 Experiment: cap KEPT 산수 SAFE — d=0.053 hover stage2 cap=2.0 → 400 reward
#     vs release path 5+8×150 = 1205 → margin +201%. Boundary hover farming blocked.
#
# BUG #2 FIX EMBEDDED (env md5 changed):
#   roarm_stack_env.py P6v14b 추가: upright check (sz_world_z > 0.90).
#   Without this, tipped sponge z_center drops near table → z_offset < thresh → stage
#   3/4 fire on tipping → 8th reward farming pattern. Both stage 3 gate and stage 4
#   success_now now require upright. Logging: upright_rate + sponge_z_axis_world_z_mean.
#
# CONFIG (vs P6v14a):
#   (1) curriculum_pregrasp REMOVED — agent grasps from scratch
#   (2) curriculum_spawn_min_r=0.08 max_r=0.15 — annulus close to target
#   (3) curriculum_xy_thresh=0.05 z_thresh=0.04 — relaxed (same as Phase 0a)
#   (4) curriculum_disable_nearzone_cap OMITTED — cap KEPT (anti-farming)
#   (5) resume P6v14a/model_499 — warm-start release-aware policy
#   (6) max_iterations 1000 (C1 recommendation; was 500 for P6v14a)
#
# SANITY GATE (DUAL, C1 권장):
#   iter 50: stage4_success > 0.05 AND grasped_frac > 0.05
#   iter 200: stage2_grasp_frac < 0.80 (rolling diagnostic — no farming)
#   iter 500: stage4_success > 0.50 AND grasped_frac > 0.30
#   FAIL → Phase 0a' insertion (sponge at table near target, pregrasp height 3cm)
#         OR BC pretraining pivot (B1 backup plan)
#
# PRE-LAUNCH MATH (C1 verified):
#   d=0.053 (P6v14 boundary hover farming point):
#     Stage 2 with cap = 2.0/step × 200 = 400 reward
#     Release path = 5 (just_on_target) + 8 × 150 (stage 4 latched) = 1205 reward
#     Margin = +201% — SAFE, boundary hover blocked
#   d=0.08 (spawn min): stage 2 same cap 2.0 → 400 vs release 1205 → +201% SAFE
#   d=target: stage 4 = 8/step → 8 × 200 = 1600 (success latch) dominates
#
# Md5:
#   roarm_stack_env.py = 286dc7e47ca27431d12b488d12ef9886 (Bug #2 fix included)
#   train_ppo.py        = 21675a050b810295b64bcae812fe976e (unchanged from P6v14a)

set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && { echo FAIL_root_unset; exit 1; }
[[ "$(whoami)" != "sogang_jhki" ]] && { echo FAIL_user; exit 1; }
[[ "$(hostname)" != "JHPark-container" ]] && { echo FAIL_host; exit 1; }

EXPECTED_ENV_MD5="286dc7e47ca27431d12b488d12ef9886"
ACTUAL_ENV_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py" | awk '{print $1}')
[[ "$ACTUAL_ENV_MD5" != "$EXPECTED_ENV_MD5" ]] && { echo "FAIL env md5: $ACTUAL_ENV_MD5 != $EXPECTED_ENV_MD5 (Bug #2 fix not synced to B200)"; exit 1; }
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
    --max_iterations 1000 \
    --reward_phase 6 \
    --seed 0 \
    --resume $ROARM_B200_ROOT/logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt \
    --reset_std 1.5 \
    --entropy_coef 0.001 \
    --episode_length_s 2.0 \
    --curriculum_spawn_min_r 0.08 \
    --curriculum_spawn_max_r 0.15 \
    --curriculum_xy_thresh 0.05 \
    --curriculum_z_thresh 0.04 \
    --experiment_name p6v14b_phase0b_resumeP6v14a
