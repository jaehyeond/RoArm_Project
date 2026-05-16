#!/usr/bin/env bash
# P6v17: learned transport/release from realistic G2-A four-source attached starts.
#
# This intentionally spends B200 GPU time on the current failing surface:
# source-to-target attached transport + release.  It does not tune Skill 1b/1c
# and does not add scripted release variants.
set -euo pipefail

cd "$(dirname "$0")"

python -m roarm_rl.train_ppo \
  --task stack \
  --reward_phase 7 \
  --num_envs 4096 \
  --max_iterations 500 \
  --seed 17 \
  --experiment_name roarm_stack_p7v3_g2a_attached_transport_release \
  --episode_length_s 2.0 \
  --entropy_coef 0.003 \
  --curriculum_attached_transport_release \
  --curriculum_attached_start_jitter_rad 0.01 \
  "$@"
