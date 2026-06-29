#!/usr/bin/env bash
set -euo pipefail

ROOT="claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/overshoot_control_d296"
CHECKPOINT="claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/model_0.pt"
D256_CSV="claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/rl_transition_preflight_d256/ppo_actor_prior_teacher_rows_d256.csv"

run_variant() {
  local tag="$1"
  local seed="$2"
  shift 2
  local outdir="${ROOT}/${tag}"
  mkdir -p "${outdir}"
  echo "[d296] start ${tag}"
  env OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. conda run -n isaaclab --no-capture-output \
    python sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py \
    --checkpoint "${CHECKPOINT}" \
    --num_envs 32 \
    --eval_steps 580 \
    --seed "${seed}" \
    --episode_length_s 6.0 \
    --action_scale 0.04 \
    --max_joint_delta_per_step_rad 0.04 \
    --joint_target_lead_limit_rad 0.06 \
    --joint_delta_reference joint_pos \
    --d256_reset_csv_path "${D256_CSV}" \
    --d256_reset_frame_index 0 \
    --d256_reset_sample_mode random \
    --fixed_push_dir_x 1.0 \
    --fixed_push_dir_y 0.0 \
    --tap_contact_proxy_mode link5_collision_aabb \
    --bc_teacher_blend 0.0 \
    --bc_teacher_imitation_reward_scale 0.0 \
    --vertical_gate_mode min_contact \
    --min_useful_seen_rate 0.90 \
    --max_overshoot_seen_rate 0.05 \
    --max_joint_delta_cap_rate 0.25 \
    --min_mean_disp_xy_m 0.0005 \
    --min_max_disp_xy_m 0.001 \
    --min_disp_xy_ge_1mm_rate 0.25 \
    "$@" \
    --out_json "${outdir}/teacher_off_policy_eval_summary_${tag}.json" \
    --out_md "${outdir}/teacher_off_policy_eval_summary_${tag}.md" \
    --out_csv "${outdir}/teacher_off_policy_eval_steps_${tag}.csv" \
    --artifact_tag "${tag}"
  echo "[d296] done ${tag}"
}

run_variant stop_disp003_random_seed29603_d296 29603 --tap_stop_after_disp_m 0.003
run_variant stop_disp003_random_seed29604_d296 29604 --tap_stop_after_disp_m 0.003
run_variant exec_clip050_random_seed29603_d296 29603 --exec_action_clip_abs 0.5
run_variant exec_clip050_random_seed29604_d296 29604 --exec_action_clip_abs 0.5
