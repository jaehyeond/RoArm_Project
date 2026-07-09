#!/usr/bin/env bash
set -euo pipefail

ROOT="claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
ACTOR="${ROOT}/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt"
BASE_OUT="${ROOT}/data_conveyor_d321/tap10cm_envcsv"
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
  local env_csv="${out_dir}/closed_loop_recovery_envs_${tag}.csv"
  if [[ -s "${summary}" && -s "${env_csv}" ]]; then
    echo "[d321-conveyor-envcsv] skip existing ${summary}"
    return 0
  fi
  echo "[d321-conveyor-envcsv] run ${bin_dir} chunk=${chunk} seed=${seed} friction=${static_friction}/${dynamic_friction}"
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
    --artifact_tag "${tag}" \
    --out_env_csv "${env_csv}"
}

# Low bin static friction: 0.70-0.88, dynamic friction = 0.75 * static.
run_chunk "bin_low_0p7_0p9" "00" "321010" "0.70" "0.525" "d321_envcsv_bin_low_chunk00"
run_chunk "bin_low_0p7_0p9" "01" "321011" "0.72" "0.540" "d321_envcsv_bin_low_chunk01"
run_chunk "bin_low_0p7_0p9" "02" "321012" "0.74" "0.555" "d321_envcsv_bin_low_chunk02"
run_chunk "bin_low_0p7_0p9" "03" "321013" "0.76" "0.570" "d321_envcsv_bin_low_chunk03"
run_chunk "bin_low_0p7_0p9" "04" "321014" "0.78" "0.585" "d321_envcsv_bin_low_chunk04"
run_chunk "bin_low_0p7_0p9" "05" "321015" "0.80" "0.600" "d321_envcsv_bin_low_chunk05"
run_chunk "bin_low_0p7_0p9" "06" "321016" "0.82" "0.615" "d321_envcsv_bin_low_chunk06"
run_chunk "bin_low_0p7_0p9" "07" "321017" "0.84" "0.630" "d321_envcsv_bin_low_chunk07"
run_chunk "bin_low_0p7_0p9" "08" "321018" "0.86" "0.645" "d321_envcsv_bin_low_chunk08"
run_chunk "bin_low_0p7_0p9" "09" "321019" "0.88" "0.660" "d321_envcsv_bin_low_chunk09"

# Mid bin static friction: 0.90-1.17, dynamic friction = 0.80 * static.
run_chunk "bin_mid_0p9_1p2" "00" "321110" "0.90" "0.720" "d321_envcsv_bin_mid_chunk00"
run_chunk "bin_mid_0p9_1p2" "01" "321111" "0.93" "0.744" "d321_envcsv_bin_mid_chunk01"
run_chunk "bin_mid_0p9_1p2" "02" "321112" "0.96" "0.768" "d321_envcsv_bin_mid_chunk02"
run_chunk "bin_mid_0p9_1p2" "03" "321113" "0.99" "0.792" "d321_envcsv_bin_mid_chunk03"
run_chunk "bin_mid_0p9_1p2" "04" "321114" "1.02" "0.816" "d321_envcsv_bin_mid_chunk04"
run_chunk "bin_mid_0p9_1p2" "05" "321115" "1.05" "0.840" "d321_envcsv_bin_mid_chunk05"
run_chunk "bin_mid_0p9_1p2" "06" "321116" "1.08" "0.864" "d321_envcsv_bin_mid_chunk06"
run_chunk "bin_mid_0p9_1p2" "07" "321117" "1.11" "0.888" "d321_envcsv_bin_mid_chunk07"
run_chunk "bin_mid_0p9_1p2" "08" "321118" "1.14" "0.912" "d321_envcsv_bin_mid_chunk08"
run_chunk "bin_mid_0p9_1p2" "09" "321119" "1.17" "0.936" "d321_envcsv_bin_mid_chunk09"

echo "[d321-conveyor-envcsv] done"
