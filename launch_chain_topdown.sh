#!/bin/bash
# (delta) Top-down chain run — 5/14
#
# Goal: full chain (Skill 0/1a/1b/1c/2/3/4) with top-down Skill 0/1 (TCP +150mm
# clearance above sponge → straight-down vertical descent), tight Skill 1b tol
# (0.005 rad), Skill 3 early-terminate after release + 15 buffer steps, Skill 4
# retreat to TCP +150mm above place to avoid sponge knock-away.
#
# Same sponge_xy=(0.25, -0.04) as 5/13 chain run for apples-to-apples comparison.

set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && { echo FAIL_root_unset; exit 1; }
[[ "$(whoami)" != "sogang_jhki" ]] && { echo FAIL_user; exit 1; }
[[ "$(hostname)" != "JHPark-container" ]] && { echo FAIL_host; exit 1; }

EXPECTED_CHAIN_MD5="c6e610216197994c6b7d2b6625d87560"
ACTUAL_CHAIN_MD5=$(md5sum "$ROARM_B200_ROOT/code/roarm_rl/chain_skills.py" | awk '{print $1}')
[[ "$ACTUAL_CHAIN_MD5" != "$EXPECTED_CHAIN_MD5" ]] && { echo "FAIL chain md5: $ACTUAL_CHAIN_MD5 != $EXPECTED_CHAIN_MD5"; exit 1; }
echo "GUARD-OK chain_md5=$ACTUAL_CHAIN_MD5"

eval "$(micromamba shell hook --shell bash)"
micromamba activate "$ROARM_B200_ROOT/envs/isaacsim_5_1"
export OMNI_KIT_ACCEPT_EULA=YES
cd "$ROARM_B200_ROOT/code"

exec python -u -m roarm_rl.chain_skills \
    --sponge_xy 0.25 -0.04 \
    --episode 1 \
    --model_path "$ROARM_B200_ROOT/logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt" \
    "$@"
