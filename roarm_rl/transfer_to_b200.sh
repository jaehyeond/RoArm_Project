#!/usr/bin/env bash
# Transfer roarm_rl/ package to B200 server.
#
# Run from local /home/cgxr/Documents/Robotics/RoArm_Project/:
#   bash roarm_rl/transfer_to_b200.sh
set -e

REPO=/home/cgxr/Documents/Robotics/RoArm_Project
B200_ROOT_REMOTE='/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200'
DEST="$B200_ROOT_REMOTE/code/roarm_rl"

cd "$REPO"
TGZ=/tmp/roarm_rl.tgz

tar czf "$TGZ" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  roarm_rl/

scp "$TGZ" JHPark:/tmp/

ssh JHPark "set -e
source $B200_ROOT_REMOTE/env.sh
[[ -z \"\$ROARM_B200_ROOT\" ]] && exit 1
[[ \"\$(whoami)\" != 'sogang_jhki' ]] && exit 1
mkdir -p \$ROARM_B200_ROOT/code
cd \$ROARM_B200_ROOT/code
rm -rf roarm_rl
tar xzf /tmp/roarm_rl.tgz
ls -la \$ROARM_B200_ROOT/code/roarm_rl/
"

echo ""
echo "Transferred. To run sanity test on B200:"
echo "  ssh JHPark 'set -e; source $B200_ROOT_REMOTE/env.sh; \\"
echo "    micromamba activate \$ROARM_B200_ROOT/envs/isaacsim_5_1; \\"
echo "    export OMNI_KIT_ACCEPT_EULA=YES; \\"
echo "    cd \$ROARM_B200_ROOT/code; \\"
echo "    python -m roarm_rl.test_sanity --num_envs 1 --steps 200'"
