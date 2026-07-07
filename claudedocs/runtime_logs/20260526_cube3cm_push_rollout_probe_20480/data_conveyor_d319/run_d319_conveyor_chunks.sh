#!/usr/bin/env bash
set -euo pipefail

ROOT="claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
ACTOR="${ROOT}/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt"
BASE_OUT="${ROOT}/data_conveyor_d319/tap10cm"
SCRIPT="sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py"

run_chunk() {
  local bin_dir="$1"
  local chunk="$2"
  local seed="$3"
  local static_friction="$4"
  local dynamic_friction="$5"
  local tag="$6"
  local out_dir="${BASE_OUT}/${bin_dir}/chunk_${chunk}"
  local summary="${out_dir}/closed_loop_recovery_summary_${tag}.json"
  if [[ -s "${summary}" ]]; then
    echo "[d319-conveyor] skip existing ${summary}"
    return 0
  fi
  echo "[d319-conveyor] run ${bin_dir} chunk=${chunk} seed=${seed} friction=${static_friction}/${dynamic_friction}"
  PYTHONPATH="/home/cgxr/Documents/Robotics/RoArm_Project" \
  OMNI_KIT_ACCEPT_EULA=YES \
  conda run -n isaaclab --no-capture-output python "${SCRIPT}" \
    --actor_checkpoint "${ACTOR}" \
    --num_envs 100 \
    --steps 580 \
    --seed "${seed}" \
    --reset_pose_source env_hook \
    --d256_reset_sample_mode random \
    --rl_action_mode candidate8_diffik_target_residual \
    --policy_action_space 3 \
    --exec_source zero \
    --candidate8_hybrid_stop_after_useful \
    --cube_static_friction "${static_friction}" \
    --cube_dynamic_friction "${dynamic_friction}" \
    --out_dir "${out_dir}" \
    --artifact_tag "${tag}"
}

run_chunk "bin_low_0p7_0p9" "00" "31901" "0.8" "0.6" "d319_bin_low_chunk00"
run_chunk "bin_low_0p7_0p9" "01" "31902" "0.8" "0.6" "d319_bin_low_chunk01"
run_chunk "bin_low_0p7_0p9" "02" "31903" "0.8" "0.6" "d319_bin_low_chunk02"
run_chunk "bin_mid_0p9_1p2" "00" "31911" "1.05" "0.84" "d319_bin_mid_chunk00"
run_chunk "bin_mid_0p9_1p2" "01" "31912" "1.05" "0.84" "d319_bin_mid_chunk01"
run_chunk "bin_upper_1p2_1p6" "00" "31921" "1.4" "1.12" "d319_bin_upper_chunk00"
run_chunk "bin_upper_1p2_1p6" "01" "31922" "1.4" "1.12" "d319_bin_upper_chunk01"
run_chunk "bin_upper_1p2_1p6" "02" "31923" "1.4" "1.12" "d319_bin_upper_chunk02"

echo "[d319-conveyor] done"
