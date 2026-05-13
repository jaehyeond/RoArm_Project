#!/bin/bash
# Path D Phase D.1 (5/14 evening) — Generate release demos from P6v14a rollout.
#
# CONTEXT (5/14 evening Path B FAIL → Path D entry):
#   P6v16/b/c RPL alpha sweep (0.30/0.05/0.10) — all 3 alpha FAIL (iter 10 stage4
#   bit-identical to P6v14c ~0.003). RPL framework REJECTED — forgetting cause
#   is log_std (PPO learnable), not residual capacity. Path D = "P6v14a release
#   policy is already good (iter 0 stage4=0.37 in P6v14c eval) → freeze P6v14a
#   as pick/release expert + train tiny release-only BC + state-machine deploy".
#
# THIS RUN:
#   - Init via curriculum_pregrasp (matches P6v14a training distribution exactly).
#   - 256 envs × 1 episode = 256 trials.
#   - Expected ~95 successful demos (P6v14a stage4_rate ≈ 0.37).
#   - PASS gate: ≥50 demos → proceed to Phase D.2 (BC train).
#   - FAIL gate: <50 demos → bump num_envs or relax xy/z thresh.
#
# Md5 verify:
#   roarm_stack_env.py                  = ff31c5a32e85cc61bec39302a9b739c0
#   gen_release_demos_from_rollout.py   = 8d02cc6f19c3a14c4817beff7dca0cd5  (v2: TensorDict obs fix)

set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && { echo FAIL_root_unset; exit 1; }
[[ "$(whoami)" != "sogang_jhki" ]] && { echo FAIL_user; exit 1; }
[[ "$(hostname)" != "JHPark-container" ]] && { echo FAIL_host; exit 1; }

EXPECTED_ENV_MD5="ff31c5a32e85cc61bec39302a9b739c0"
ACTUAL_ENV_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py" | awk '{print $1}')
[[ "$ACTUAL_ENV_MD5" != "$EXPECTED_ENV_MD5" ]] && { echo "FAIL env md5: $ACTUAL_ENV_MD5"; exit 1; }
EXPECTED_GEN_MD5="8d02cc6f19c3a14c4817beff7dca0cd5"
ACTUAL_GEN_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/gen_release_demos_from_rollout.py" | awk '{print $1}')
[[ "$ACTUAL_GEN_MD5" != "$EXPECTED_GEN_MD5" ]] && { echo "FAIL gen md5: $ACTUAL_GEN_MD5"; exit 1; }
echo "GUARD-OK env_md5=$ACTUAL_ENV_MD5 gen_md5=$ACTUAL_GEN_MD5"

eval "$(micromamba shell hook --shell bash)"
micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
export OMNI_KIT_ACCEPT_EULA=YES
cd $ROARM_B200_ROOT/code

OUTPUT_DIR="$ROARM_B200_ROOT/logs/roarm_rl/pathD_demogen_v1"
mkdir -p "$OUTPUT_DIR"

exec python -u -m roarm_rl.gen_release_demos_from_rollout \
    --checkpoint $ROARM_B200_ROOT/logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt \
    --num_envs 256 \
    --num_episodes 1 \
    --curriculum_xy_thresh 0.05 \
    --curriculum_z_thresh 0.04 \
    --episode_length_s 2.0 \
    --seed 0 \
    --output "$OUTPUT_DIR/release_demos_v1.pt"
