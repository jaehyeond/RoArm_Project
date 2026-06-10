# START_HERE.md

Last updated: 2026-06-10 KST (D225 current truth: after D224, ran only local no-training base-only probes for the professor 10cm/0.72kg cube tap RL branch, keeping the 6mm tap objective fixed. Broad xy randomization up to +/-15cm did not produce a clean 30-60% base success boundary; seed1020 n64 stayed success_episode about 0.78-0.89, while overshoot/target quality degraded. Fixed pose-bin probes found the real weak region: high lateral +y=0.15 with close x is target-band failing despite normal IK/contact/reaction. Examples: x=0.09,y=0.15 and x=0.14,y=0.15 both had success_event_count=0, target_band=0, overshoot=0, ik_reset_rate=1.0, candidate6_numeric_ok=1.0; x=0.19,y=0.15 and beyond recovered success events. D224 geometry still stands; D223 same-contract L2 still failed; no Large PPO/dataset/VLA/action-teacher/RoArm.)

2026-06-10 latest base failure boundary state: D225 supersedes D224 as the active research step but does not alter D224 geometry. User directed the branch to stop blind PPO scaling and hold `policy_target_disp_m=0.006` while sweeping initial cube pose. Ran local RTX4090/cuda:0 no-training `max_iterations=0` base-only probes only; no code changes, PPO learning, Large PPO, dataset, VLA, action-teacher, RoArm, SSH/B200, or Track A. Coarse xy randomization at seed1020 n64 with half-extents +/-3, +/-5, +/-7, +/-10, +/-15cm stayed base-success high by `success_episode_rate` (`0.8889`, `0.8690`, `0.7941`, `0.78125`, `0.8091`) and was not a clean RL operating point; quality worsened with overshoot up to `0.21875` and low target-band rates. Fixed pose-bin seed1021 n32 isolated the real weak region: `(x=0.09,y=0.15)` and `(x=0.14,y=0.15)` have `success_event_count=0`, `target_band=0`, contact/reaction `1.0`, overshoot `0.0`, `ik_reset_rate=1.0`, and Candidate6 numeric OK `1.0`, so this is not IK/contact failure but target-band under/quality failure. `(x=0.09,y=0.0)` succeeds (`success_episode_rate=0.9926`) and `(x=0.39,y=0.15)` succeeds cleanly (`success_episode_rate=1.0`, target_band `0.6875`), so the weak region is specifically close-x + high lateral +y. Next valid step is not broad random L2/Large PPO; it is a small, explicitly approved L1 on a weak-bin fixed/narrow curriculum around the close-x/high-y region, with same-run base comparison and strict overshoot/target-band gates.

2026-06-10 latest TCP/contact geometry check: D224 supplements D223 for the professor 10cm/0.72kg cube tap RL branch. User's TCP suspicion was important and partially right, but not as a missing 115mm offset. `sim_scripts/roarm_kinematics.py` defines `link5_to_tcp` as `[0,0,0.115428]`; `local_assets/roarm_m3/urdf/roarm_m3.urdf` defines fixed `link5_to_hand_tcp` at the same xyz/rpy; `roarm_rl/roarm_stack_env.py` computes `_tcp_pos_w = link5_pos + quat(link5)*TCP_LOCAL_OFFSET_M`; Candidate6 built-in DiffIK commands link5 by subtracting the same TCP offset from the desired TCP target before calling Isaac DiffIK; numeric IK reset uses `fk_tcp` with the same offset. Therefore IK is not solving for bare link5 origin. Existing positive-control telemetry also reported `actual_fk_vs_sim_tcp_err_mm_final=0.0`. The real nuance is contact geometry: link5 collision AABB max local z is `0.11988562011718751`, while hand_tcp local z is `0.115428`, so the foremost collision surface is about `4.46mm` beyond the hand_tcp point. This explains why old strict `tcp_point` shortfall numbers were too pessimistic for physical contact. The old x240 reach split remains consistent: applied FK shortfall about `3.46mm`, actual lag about `5.6mm`, total actual shortfall about `8.96-9.53mm`; it is not a pure TCP-origin bug. Current PPO smoke sets `tap_contact_proxy_mode=link5_collision_aabb`, so current RL success/contact uses the collision proxy rather than raw hand_tcp point. D223 remains current for learning: same-contract L2 failed, Large PPO remains unjustified, dataset/VLA/action-teacher/RoArm remain blocked.

2026-06-09 latest randomized robustness state: D214 supersedes D213 for the professor 10cm/0.72kg cube tap RL branch. Added default-off cube position randomization to the Candidate6 residual PPO smoke runner and kept fixed behavior intact: fixed seed982 line 3 reports zero randomization and line 4 reports zero-policy success/contact/reaction `1.0/1.0/1.0` with overshoot `0.0`. Randomization exposes a real robustness gap: xy +/-3cm seed978 line 4 falls to `0.359375/0.359375/0.359375` with overshoot `0.046875`; xy +/-1cm seed979 line 4 falls to `0.109375/0.109375/0.109375` with overshoot `0.015625`; x-only and y-only +/-1cm screens also fail. A small xy +/-1cm residual PPO L1 (`32*64*20=40,960` steps, seed983) did not recover it: line 4 base pre-eval was `0.125/0.125/0.125`, line 7 post-eval was only `0.0625/0.0625/0.0625`, and line 8 reports `training_smoke_pass=False`, `policy_task_pass=False`. Added reset IK metrics to the tap env log; xy +/-1cm resetmetric seed985 line 4 reports `ik_reset_rate_min=1.0`, `ik_reset_err_mm_max=1.316048622`, Candidate6 active/numeric `1.0/1.0`, but success/contact/reaction still `0.109375/0.109375/0.109375`, so the failure is not reset IK or numeric DiffIK activation. Current verdict: randomization is the right research direction, but blind larger PPO is not. The next pass route is a controller/trajectory robustness candidate for randomized cube poses, then PPO after the base manifold is not catastrophically brittle. Large dataset/VLA/RoArm/action-teacher claims remain NO.

2026-06-09 previous residual PPO checkpoint-promotion state: D213 superseded D212 for the professor 10cm/0.72kg cube tap RL branch. Ran local RTX4090/cuda:0 loaded-checkpoint promotion validation for L3 `model_499.pt`; no training, no geometry/contact/controller/action-wrapper changes, no dataset, no robot/RoArm, no SSH/B200, no Track A. All runs kept `rl_action_mode=candidate6_diffik_residual_joint`, `tap_success_terminate=True`, residual scale `0.002`, previous-target-base, near-face target path, AABB contact, hand TCP proxy, `init_noise_std=0.2`, and contract violations `0`. Independent seed evals at `num_envs=32` for seeds `974`, `975`, and `976` all produced the same partial result: line 7 reports finite loaded-policy eval, success/contact/reaction `0.90625/0.90625/0.90625`, overshoot `0.0`, lead-limit `0.0`, joint-delta cap `0.0`, residual max `0.000655551`; line 8 reports `policy_task_pass=True` but this is not strict all-env promotion. Env-scale seed `977`, `num_envs=64`, dropped further to success/contact/reaction `0.859375/0.859375/0.859375` with overshoot/lead/cap still `0.0`. Zero-policy pre-eval remained `1.0` in these runs, so the learned L3 residual is the likely source of partial degradation. Verdict: L3 `model_499.pt` is a single-run ladder PASS but checkpoint promotion/reproducibility validation is NOT strict PASS. Large dataset/VLA/RoArm/action-teacher claims remain NO. Next research step should be checkpoint selection or residual regularization, not dataset/RoArm.

2026-06-09 previous residual PPO learning-ladder state: D212 superseded D211 for the professor 10cm/0.72kg cube tap RL branch. Ran the controlled same-contract Candidate6 residual PPO ladder locally on RTX4090/cuda:0, no SSH/B200, no Track A, no dataset generation, no robot/RoArm. L1 default PPO noise `0.8` was a health-warning case, not a promotion: `20,480` steps completed but post-eval success/contact/reaction dropped to `0.75` and TensorBoard reward fell `1.417 -> 0.120`, while residual nearly hit the `0.002rad` scale. Added `--ppo_init_noise_std` to the PPO smoke runner and reran L1b with `init_noise_std=0.2`: summary line 7 PASSed post-eval success/contact/reaction `1.0/1.0/1.0`, overshoot/lead/cap `0.0`, residual max `0.000330671`, and TensorBoard reward stayed `1.813 -> 1.851`. L2 `102,400` steps with the same contract PASSed: summary line 7 reports success/contact/reaction `1.0/1.0/1.0`, overshoot/lead/cap `0.0`, residual max `0.000343330`, TensorBoard reward `1.806 -> 1.961`, and checkpoint `model_49.pt`. L3 `1,024,000` steps with the same contract PASSed: summary line 7 reports post-eval success/contact/reaction `1.0/1.0/1.0`, overshoot `0.0`, lead-limit `0.0`, joint-delta cap `0.0`, residual max `0.000655551`, and checkpoint `model_499.pt`; TensorBoard reward ran `1.813 -> 1.957` with max `2.105`, policy noise std `0.200 -> 0.0619`, residual max `0.000659 -> 0.000283`, overshoot/lead/cap all `0`. This unblocked only the fixed-contract residual-action PPO learning branch to checkpoint promotion/reproducibility validation. Raw joint-delta PPO, action-teacher dataset claims, large dataset/VLA, and RoArm deployment remain NO.

2026-06-09 previous residual PPO promotion state: D211 superseded D210 for the professor 10cm/0.72kg cube tap RL branch. Promoted the D210 `model_2.pt` only through fixed-contract eval: no training, no geometry/contact/controller/action wrapper changes, no dataset, no robot, no SSH/B200, no Track A. Independent reset-seed loaded-checkpoint evals PASSed for seeds `967`, `968`, and `969` at `num_envs=8`: each summary line 2 reports contract violations `0`; line 6 reports `tap_success_max=1.0`, `contact_seen_max=1.0`, `reaction_seen_max=1.0`, `overshoot_max=0.0`, candidate6 active/numeric `1.0/1.0`, residual max `0.000413679`, lead-limit rate `0.0`, and joint-delta cap rate `0.0`; line 7 reports `policy_task_pass=True` and keeps `large_dataset_rl_roarm_unblocked=NO`, `action_teacher_dataset=NO`. Small env-scale loaded-checkpoint eval also PASSed at seed966 `num_envs=16`, with line 6 reporting the same success/contact/reaction/overshoot and residual metrics. Audit verdict line 7 is `CANDIDATE6_RESIDUAL_SUCCESS_TERMINATE_PROMOTION_VALIDATION_PASS`; line 8 sets the next step to a controlled residual PPO learning ladder under the same contract. This unblocks only the residual-action-path pilot PPO branch; raw joint-delta scale-up, action-teacher dataset claims, large dataset, VLA, and RoArm deployment remain NO.

2026-06-09 previous residual PPO state: D210 superseded D209 for the professor 10cm/0.72kg cube tap RL branch. Ran the next tiny local RTX4090/cuda:0 PPO smoke under the D209 action-path contract: `rl_action_mode=candidate6_diffik_residual_joint`, `tap_success_terminate=True`, `candidate6_diffik_residual_scale_rad=0.002`, seed966, `num_envs=8`, `max_iterations=3`, `num_steps_per_env=64`, eval steps `580`. Contract violations were `0`; `model_0.pt`, `model_1.pt`, and `model_2.pt` were written under `ppo_runs/cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_ppo_smoke/seed966_env8_it3/`. Training summary line 7 reports `training_smoke_pass=True` and `policy_task_pass=True`; line 6 reports post-eval `tap_success_max=1.0`, `contact_seen_max=1.0`, `reaction_seen_max=1.0`, `overshoot_max=0.0`, `candidate6_active_rate_max=1.0`, `candidate6_numeric_ok_rate_min=1.0`, residual max `0.000413679`, lead-limit rate `0.0`, and joint-delta cap rate `0.0`. Reloaded-checkpoint posthoc eval also PASSed from `model_2.pt`: posthoc line 6 repeats success/contact/reaction `1.0/1.0/1.0`, overshoot `0.0`, candidate6 active/numeric `1.0/1.0`, and residual max `0.000413679`; line 7 reports `policy_task_pass=True`. Audit verdict line 7 is `CANDIDATE6_RESIDUAL_SUCCESS_TERMINATE_TINY_PPO_SMOKE_PASS`. This still is not large PPO/RL, dataset, action-teacher, or RoArm readiness; `large_dataset_rl_roarm_unblocked=NO` and `action_teacher_dataset=NO` remain current. Next valid step is fixed-contract promotion validation across independent reset seeds and small env scale under this residual action-path contract, not raw joint-delta scale-up.

2026-06-09 previous RL action-path state: D209 superseded D208 for the professor 10cm/0.72kg cube tap RL branch. The key critique was valid: Candidate6 PASS used built-in DiffIK direct target application, while the PPO env default action path was raw joint-delta. Added default-off `rl_action_mode=candidate6_diffik_residual_joint` in `roarm_rl/roarm_cube_push_env.py`: the base target is Candidate6 near-face built-in DiffIK with previous-target-base, 0.010rad step clip, 0.060rad lead limit, AABB contact, hand TCP proxy, and the policy action is only a small residual (`candidate6_diffik_residual_scale_rad=0.002`). Added smoke-runner args/contract logging for the residual action mode and `tap_success_terminate`. Static bridge audit line 8 pins the new env/smoke lines. First local RTX4090/cuda:0 no-training bridge preflight with `tap_success_terminate=False` proved action-path transfer but failed strict quality after success (`candidate6_active_rate_max=1.0`, `numeric_ok_rate_min=1.0`, `tap_success_max=1.0`, but `overshoot_max=0.625`). The pass route is success termination/hold, not more raw joint-delta PPO: the second no-training preflight with `tap_success_terminate=True` PASSed strict zero-policy task under the Candidate6 residual action path (`tap_contact_seen_max=1.0`, `reaction_seen_max=1.0`, `tap_success_max=1.0`, `tap_overshoot_max=0.0`, `zero_policy_task_pass=True`, contract violations `0`). This is still not large PPO/RL, dataset, action-teacher, or RoArm readiness; `large_dataset_rl_roarm_unblocked=NO` and `action_teacher_dataset=NO` remain current.

2026-06-09 previous RL smoke state: D208 superseded D207 "pilot RL smoke/design unblocked" for the professor 10cm/0.72kg cube tap RL branch. Added `roarm_rl/train_cube_tap10cm_ppo_smoke.py` as a Candidate6 fixed-contract PPO smoke runner; it is not a dataset generator or RoArm path. Preflight PASSed under the fixed Candidate6 env contract (`cube=(0.240,0.000)`, push dir `(+1,0)`, `tap_contact_proxy_mode=link5_collision_aabb`, `tool_contact_proxy_mode=hand_tcp`, `precontact_clearance_m=0.040`, `episode_length_s=6.08`, eval steps `580`, policy target displacement `0.006`, step clip `0.010`, lead limit `0.060`, `scripted_teacher_blend=0.0`) with `preflight_pass=True`, finite zero/untrained policy rollouts, and contract violations `0`. Tiny PPO smoke ran local RTX4090/cuda:0 with seed966, `num_envs=8`, `max_iterations=3`, `num_steps_per_env=64`; checkpoint `model_2.pt` exists and `training_smoke_pass=True`. The first smoke summary underreported policy task success due smoke-script log-key mapping; corrected loaded-checkpoint posthoc eval now supersedes it: `tap_contact_seen_max=1.0`, `reaction_seen_max=1.0`, `tap_success_max=1.0`, `tap_overshoot_max=0.0`, and `policy_task_pass=True`. Quality caveat remains: posthoc details include `tap_disp_max=3.32072377204895e-05`, `tcp_cube_dist_min_m=0.08132576197385788`, `target_lead_limit_rate_max=0.5`, and `joint_delta_cap_rate_max=0.5`, so this is fixed-contract tiny RL policy evidence only. `large_dataset_rl_roarm_unblocked=NO` and `action_teacher_dataset=NO` remain current.

2026-06-09 previous promotion state: D207 supersedes D206 "planning only" for the professor 10cm/0.72kg cube tap RL branch. Candidate6 is now fixed as the Stage-0 positive-control contract: `cube=(0.240,0.000)`, push dir `(+1,0)`, `controller_mode=isaac_builtin_diffik_step_clipped_direct_apply`, `target_path_mode=near_face_goal`, `builtin_diffik_target_base_mode=previous_joint_target`, `tap_contact_proxy_mode=link5_collision_aabb`, `tool_contact_proxy_mode=hand_tcp`, `precontact_clearance_m=0.040`, `episode_length_s=6.08`, `steps=580`, step clip `0.010`, lead limit `0.060`, default arm drive `80/4/2.5/3.14`. Added `sim_scripts/cube10cm_tap_rl_candidate6_promotion_audit.py`. A direct base-Python launch was BLOCKED by `ModuleNotFoundError: No module named 'isaaclab'` and is not physics evidence; valid local RTX4090/cuda:0 runs used `conda run -n isaaclab --no-capture-output python -u -m roarm_rl.test_positive_control_cube_tap10cm`. Stage0A multi-seed fixed-geometry validation PASSed for existing seed962 plus new seed963/964/965, all `num_envs=2`, with initial contact `0.0`, first contact step `162`, first success step `333`, actual contact rows `343`, `tap_success=0.5`, no overshoot/termination/truncation, no action fields, and contract violations `0`. Stage0B small env-scale validation PASSed for seed962 `num_envs=8`: first contact step `162`, first success step `331`, actual contact rows `1358`, `tap_success=0.375`, `contact_seen=1.0`, overshoot `0.0`, term/trunc `0/0`, no action fields, contract violations `0`. Audit verdict: `candidate6_promotion_validation_pass=True`, `pilot_rl_smoke_design_unblocked=True`, but `large_dataset_rl_roarm_unblocked=NO` and `action_teacher_dataset=NO`. Next valid step is a tiny pilot RL smoke/design using this fixed AABB-contact env contract, not a large dataset/PPO scale-up/RoArm deployment.

2026-06-09 previous detail-trace patch: implemented default-off `--reach_trace_detail_json` in `roarm_rl/test_positive_control_cube_tap10cm.py` and added/reran local-only static audit `sim_scripts/cube10cm_tap_rl_reach_trace_detail_patch_contract_audit.py`; no GPU/runtime/data/training/robot/SSH/B200/Track A. Audit line 1 confirms local static only. Line 2 verifies the D203 basis remains command final face gap `0.005999971m` with applied/actual inside rows `0/0`. Line 3 verifies the parser arg at line `890`, default-off behavior, separate detail JSON, and no control change. Line 4 verifies target-base fields: previous target state/update, raw/clipped delta, and post-step `joint_pos_target_after_arm_rad`. Line 5 verifies actuator telemetry fields: joint vel/acc, computed/applied torque, effort limit, velocity limit. Line 6 verifies schema guard: `contains_action_fields=false` and `action_teacher_dataset=false`. Line 7 fixes next step: only with explicit approval, run one tiny same-contract nearface x240 h580 ep608 step-clipped built-in DiffIK repeat with both basic and detail traces; contact-gate relaxation is not next. Line 8 verdict is `READY_LOCAL_ONLY_DEFAULT_OFF_DETAIL_TRACE_PATCH`. Strict contact/RL positive-control, DiffIK action dataset, PPO/RL, large dataset, and RoArm remain blocked.

2026-06-09 previous remaining-blocker design: added and ran local-only `sim_scripts/cube10cm_tap_rl_remaining_blocker_decomposition_design.py`; no GPU/runtime/data/training/robot/SSH/B200/Track A. Design line 2 anchors the near-face failure: status FAIL, RL contact-gated FAIL, professor evidence PASS, `steps_executed=580`, no truncation, command final face gap `0.005999971m`, command inside steps `223..579` / `714` rows, but applied/actual inside rows `0/0`. Line 3 ranks remaining hypotheses: rank1 `TARGET_BASE_ACCUMULATION_OR_APPLIED_TARGET_GENERATION`, rank2 `ACTUATOR_DRIVE_FOLLOW`, rank3 `PRECONTACT_RESET_INITIAL_OFFSET`; contact gate relaxation is not next. Line 4 pins target-base evidence and code: actual joint source line `364`, raw delta `370`, step clip `372`, actual-base target line `373`, `target_full` from actual line `384`, applied best face gap `-0.014022207m`, applied shortfall `0.004022207m`, final target FK error `25.145485846mm`, raw delta final `0.095429182rad`, clipped final `0.010000000rad`. Line 5 says reset offset is lower priority but not cleared: initial command/applied/actual face gaps `-0.019955199/-0.020112728/-0.021178227m`, actual-command bias `-0.001223028m`, reset IK err `1.065392971mm`. Line 6 says actuator follow is real secondary: follow max `0.010870218rad`, actual step max `0.001367390rad`, ratio `0.125792337`, extra actual shortfall over applied `0.004937029m`, with env/actuator lines pinned. Line 7 designs a default-off `--reach_trace_detail_json` patch: absent means no output/control change; fields must include reset snapshot, command/applied/actual scalar gaps, per-joint raw/clipped/target/actual/follow arrays, joint velocity/acceleration, torque/limits. Line 8 decision tree: previous-target counterfactual enters band -> design previous-target-base runtime; applied enters but actual misses/torque saturates -> actuator-follow runtime; reset bias >2-3mm -> reset/precontact recalibration; otherwise FK/tool-frame visual overlay. Dataset/RL/RoArm remain blocked.

2026-06-09 previous near-face target-path runtime: user explicitly approved the next tiny candidate. First sandbox launch wrote a BLOCKED log (`ModuleNotFoundError: No module named roarm_rl`, plus sandbox CUDA device errors), so it is not a physics result. Reran once local unsandboxed with `PYTHONPATH=/home/cgxr/Documents/Robotics/RoArm_Project`, same x240 h580 ep608 step-clipped built-in DiffIK contract, only `--target_path_mode near_face_goal`; no dataset/training/robot/SSH/B200/Track A. Runtime summary line 1 is still `status=FAIL`; line 2 confirms 10cm/0.72kg, `episode_length_s=6.08`, `env_max_episode_length=608`; line 3 confirms `steps_executed=580`, `cube_xy=(0.24,0.0)`, controller `isaac_builtin_diffik_step_clipped_direct_apply`, `target_path_mode=near_face_goal`, direct target apply true, step clip `0.01`; line 5 still has `contact_seen=0.0`, `tap_success=0.0`, professor weak reaction seen `1.0`; line 8 actual face gap max is only `-0.018959235m` with shortfall `0.008959235m`; line 9 keeps professor evidence PASS but RL contact-gated FAIL; line 10 proves near-face command applied (`target_face_gap_final=0.005999971`, inside final `1.0`) but applied FK error remains `25.145485846mm`. Added posthoc `sim_scripts/cube10cm_tap_rl_nearface_target_path_result_audit.py`: audit line 3 shows legacy command final face gap `0.105999991 -> 0.005999971` and command inside rows `714`; line 5 shows applied/actual inside rows remain `0/0`; line 6 shows final FK error improved by `-101.912614319mm` but actual best shortfall improved only `-0.000003666m`. Verdict line 7: `NEAR_FACE_TARGET_PATH_APPLIED_BUT_STRICT_CONTACT_STILL_FAILS_ACTUAL_TCP_PRECONTACT`. Do not relax contact gate or start dataset/RL/RoArm. Next local-only design must isolate target-base accumulation vs precontact reset/initial offset vs actuator/drive follow telemetry.

2026-06-09 previous first-button target-path audit: user challenged whether the initial controller/code setup is wrong because a tap should place TCP on the object face and solve IK there. Added default-off `--target_path_mode {legacy_far_face_through,near_face_goal}` in `roarm_rl/test_positive_control_cube_tap10cm.py` while preserving legacy default; no GPU/runtime/data/training/robot/SSH/B200/Track A. Added and ran local-only `sim_scripts/cube10cm_tap_rl_target_path_first_button_audit.py`. Audit line 2 pins both target paths: external DLS path lines `160-164`, built-in DiffIK path lines `330-334`, parser line `683`. Line 3 shows the first-button mismatch math: for a 10cm cube, legacy final face gap is `0.106000000m` (`cube_size + goal_push`) while the near-face tap/push goal should end at `0.006000000m`; the legacy path is `4.846153846x` longer than near-face. Line 4 proves existing x240 followed legacy far-face-through (`command_final_face_gap_m=0.105999991`, `matches_legacy_far_face=TRUE`, `matches_near_face_goal=FALSE`). Line 5 shows the consequence: applied/actual inside rows remain `0/0`, applied FK error grows from `0.122874349mm` to `127.058100165mm`. Verdict line 6: `FIRST_BUTTON_MISMATCH_LEGACY_FAR_FACE_THROUGH_TARGET_FOR_10CM_TAP`. The next runtime candidate, only with explicit approval, is the same x240 h580 ep608 step-clipped built-in DiffIK contract plus `--target_path_mode near_face_goal` and reach trace. Do not relax contact gate; dataset/RL/RoArm remain blocked.

2026-06-09 previous horizon-vs-clip check: added and ran local-only `sim_scripts/cube10cm_tap_rl_horizon_vs_clip_interpretation_audit.py` from existing x240 trace/sanity only; no GPU/runtime/data/training/robot/SSH/B200/Track A. Summary line 2 shows this is not an episode cutoff: `steps_executed=580`, `max_steps=580`, `terminated_count=0`, `truncated_count=0`, command in-band steps `46..137`, final step `579`. Line 3 shows command keeps progressing through the cube (`post_delta_m=0.096020695`, final gap `0.105999991m`). But line 4 shows applied FK moves away after the contact window (`inside_delta_m=-0.001222481`, `post_delta_m=-0.005073384`, final `-0.019887484m`), and line 5 shows actual TCP also moves away (`inside_delta_m=-0.001244128`, `post_delta_m=-0.005120277`, final `-0.025572997m`). Line 6 shows the controller mismatch: raw delta `0.427774668rad`, clipped delta `0.010000000rad`, actual joint step `0.001492023rad`. Verdict line 7: `NOT_A_SIMPLE_MORE_HORIZON_STEPS_FIX`. Increasing the same horizon alone is not the right next unblock; next is target-generation contract design or an Isaac Sim render with explicit overlay markers if a new runtime is approved.

2026-06-09 previous visual check: generated and rendered local-only x240 reach-contract visual audit (`cube10cm_tap_rl_reach_contract_visual_audit.{svg,html,png,json,summary.out}`) from the existing per-step trace; no GPU/runtime/data/training/robot/SSH/B200/Track A. Visual summary line 3 confirms command target enters the contact band (`184` rows), while applied FK and actual TCP have 0 inside rows. Line 4 shows the first in-band step visually/numerically: command gap `-0.009789657m` is inside the ±10mm band, but applied FK is still `-0.013591619m` and actual TCP is `-0.019208591m`; line 5 mid step keeps command near zero while applied/actual remain outside; line 6 final step has command far through the cube (`0.105999991m`) but applied/actual still precontact. The PNG was rendered with headless Chrome and visually inspected: blue command line crosses the green contact band, orange applied-FK and red actual-TCP lines stay below it. This answers the visual question: yes, the picture makes the cause obvious, and it supports target-generation clip plus actuator lag rather than contact-gate or cube-mass-first explanations.

2026-06-09 previous cube10cm root-cause audit: added and ran local-only `sim_scripts/cube10cm_tap_rl_reach_contract_root_cause_audit.py`; no GPU/runtime/data/training/robot/SSH/B200/Track A. Root-cause summary line 2 identifies the primary cause as `STEP_CLIPPED_CURRENT_JOINT_BASED_TARGET_GENERATION`: built-in DiffIK raw delta max is `0.427774668rad`, but the harness clips to `0.010000000rad`, keeps target delta from actual at `~0.010000005rad`, and final target FK error grows to `127.058100165mm`. Line 3 pins the code basis: current actual joint pos line `358`, DiffIK compute `362`, raw delta `364`, step clip `366`, `arm_joint_target = joint_pos_arm + clipped_delta_arm` `367`, `target_full` seeded from actual joint pos `378`, target assignment `379`, and IsaacLab DiffIK returns `joint_pos + delta_joint_pos` at installed source line `174`. Line 4 shows the contact effect: command target crosses (`184` rows / `92` steps), but applied FK and actual TCP both have 0 inside rows; first command-applied miss is `0.003801962m` and command-inside mean is `0.014288675m`. Line 5 identifies secondary cause `POSITION_DRIVE_ACTUAL_TCP_LAG`: target follow max `0.010857821rad`, actual step max `0.001492023rad`, only `0.149202267` of target lead per 0.01s control step. Line 6 ties this to the env/actuator contract: direct override `roarm_cube_push_env.py:633-638`, `set_joint_position_target` `:753`, `decimation=2`, `dt=1/200`, arm stiffness/damping/effort/velocity `80/4/2.5/3.14`, and IsaacLab implicit actuator PD handled in simulation. Line 8 keeps exact effort/stiffness/damping split unresolved until torque/per-joint telemetry. Next local-only unblock is a default-off target-generation contract design separating schedule/clip/base from actuator-follow dynamics; do not relax contact gate, do not use x=0.285 as a shortcut, and do not start dataset/RL/RoArm.

2026-06-09 previous cube10cm applied-target/TCP reach-contract diagnosis: added and ran local-only `sim_scripts/cube10cm_tap_rl_applied_target_tcp_reach_contract_diagnosis.py` from existing x250/x240 per-step traces; no GPU runtime, dataset/training, robot, SSH/B200, or Track A. Diagnosis line 2 pins the code contract: target path `test_positive_control_cube_tap10cm.py:327-332`, built-in DiffIK compute `:362`, step clip `:366`, `target_full` `:379`, applied FK trace `:413`, env direct target override `roarm_cube_push_env.py:633-638`, `set_joint_position_target` `:753`, post-step actual trace `test_positive_control_cube_tap10cm.py:895`, and contact proxy `roarm_cube_push_env.py:1103-1105`. Line 3 shows x240 command target enters the contact band for 184 rows / 92 steps (`46..137`). Line 4 shows applied joint-target FK enters 0 rows, best face gap `-0.013457759m`, best shortfall `0.003457759m`, final FK error `127.058100165mm`. Line 5 shows actual TCP enters 0 rows, best face gap `-0.018961143m`, best shortfall `0.008961143m`. Line 6 splits the miss: first command-band step has command-applied `0.003801962m` and applied-actual `0.005616973m`; across the command-inside window those are `0.014288675m` and `0.005629399m`. Line 7 shows follow remains near the 0.010rad step clip while actual joint step is only ~0.0014-0.0015rad. Verdict line 9: `TARGET_FULL_FK_NEVER_REACHES_FACE_BAND_AND_ACTUAL_TCP_LAGS_TARGET_FULL`.

2026-06-09 previous cube10cm fixed-pose audit/test: user questioned whether the 10cm cube fixed pose was still a 3cm-center pose. Added local design audit `sim_scripts/cube10cm_tap_rl_same_center_vs_same_face_pose_audit.py` and ran exactly one approved x240 tiny runtime. Design audit line 2 shows current same-center `x=0.250` gives 10cm near face `x=0.200` and previous actual shortfall `0.009534182m`; line 3 rejects same-3cm-center near-face `x=0.285` for +x because it moves the face farther, and identifies same-3cm-xmin near-face `x=0.240`; line 4 selects `fixed_cube_x_m=0.240`, `y=0.000`, changing only pose. Runtime line 3 confirms x240 with the same built-in step-clipped DiffIK h580 ep608 contract; line 5/9 still fail contact/tap while professor evidence PASSes. Result audit line 6 shows applied FK still has 0 inside rows, shortfall only improved `0.004059910 -> 0.003457759m`; line 7 shows actual TCP still has 0 inside rows, shortfall only improved `0.009534182 -> 0.008961143m`; line 9 verdict is `X240_POSE_IMPROVES_FACE_SHORTFALL_BUT_STILL_NO_CONTACT`. Fixed pose was part of the problem but not sufficient; do not use x=0.285 for +x, do not relax contact gate, and do not start dataset/RL/RoArm. Next local-only unblock remains applied joint-target/TCP reach contract diagnosis.

2026-06-09 previous cube10cm per-step reach trace repeat: after explicit approval, ran exactly one local RTX4090/cuda:0 h580 ep608 reach-trace repeat with only `--reach_trace_json` added to the previously selected continuous step-clipped built-in DiffIK contract. Runtime summary line 1 is still status `FAIL`; line 5 keeps contact/tap `0.0` while professor weak reaction evidence is seen; line 6 confirms no episode cap regression (`terminated_count=0`, `truncated_count=0`); line 9 keeps professor evidence `PASS` and RL contact-gated positive-control `FAIL`. Trace result audit line 2 verifies the separate telemetry artifact (`cube10cm_tap_rl_per_step_reach_trace_v1`, `action_teacher_dataset=False`, `1160/1160` rows, steps `0..579`, envs `[0,1]`). Audit line 4 shows the command target entered the contact band for 184 rows / 92 unique steps (`first_step=46`, `last_step=137`), but line 5 shows applied joint-target FK entered 0 rows (`face_gap_max=-0.014059910`, best shortfall `0.004059910m`, final target FK error `127.704326062mm`) and line 6 shows actual TCP entered 0 rows (`face_gap_max=-0.019534182`, best shortfall `0.009534182m`) while lateral/vertical remain small. Verdict line 8: `APPLIED_AND_ACTUAL_REACH_NEVER_ENTER_CONTACT_BAND`.

2026-06-09 previous cube10cm per-step reach trace patch: default-off `--reach_trace_json` is implemented in the 10cm tap positive-control harness and locally audited as `READY_LOCAL_ONLY`; no runtime was launched in that patch step. The patch writes a separate telemetry JSON only when the arg is provided, marks `action_teacher_dataset=false`, and records per-step/per-env command target gap, applied joint-target FK gap, actual TCP gap, joint follow, cube reaction, and terminated/truncated flags. Static audit line 3 verifies code readiness (`reach_trace_arg=700`, trace writer `591`, applied FK metric `462`, row-count metadata `1127`); line 4 verifies schema/default-off/separate-json and no action-teacher dataset; line 5 recorded exactly one designed-but-not-run h580 ep608 repeat with `reach_trace_json` as the only change.

2026-06-09 latest cube10cm reach-contract audit: target/actual contact trajectory audit was run local-only from existing logs/code. It found the strict failure is now specifically an along-face reach/trajectory contract gap, not lateral or vertical gate margin: contact contract is face band `0.010m`, lateral limit `0.065m`, vertical limit `0.070m`; ep608 actual lateral max is only `0.000231256m` and vertical max `0.020352287m`, both inside gate, but actual face-gap max remains `-0.019535881m` with shortfall min `0.009535881m`. The command target crosses the contact band (`command_target_inside_max=1.0`, command target face-gap min `-0.019782793m`, final `0.105999991m`), while actual TCP stays outside. This pattern is stable across step120/h580/direct telemetry/slow240. Critical limitation: current JSON has only min/max/final trace stats, no full step timeline, and built-in step-clipped applied joint-target FK gap is unavailable (`target_fk_err_mm_final=nan`). Therefore do not relax contact gate yet. Next local unblock is to patch a default-off per-step reach trace that records command target gap, applied joint-target FK gap, actual TCP gap, joint follow, cube reaction, and done flags; run one tiny repeat only after explicit approval. Dataset/RL/PPO/large dataset/RoArm remain BLOCKED.

2026-06-09 latest cube10cm tap RL unblock state: the default-off episode-length override path was designed, patched, and run once on the already selected step-clipped h580 contract. The override worked: runtime summary line 2 shows `episode_length_s=6.08` and `env_max_episode_length=608`, line 6 shows `terminated_count=0` and `truncated_count=0`, and the posthoc audit line 3 records `continuous_horizon_valid=True` plus `episode_cap_blocker_resolved=True` against the previous h580 `truncated_count=8`. This resolves D191's episode-cap blocker, but it does NOT unblock RL/contact success: runtime summary line 5 still has `contact_seen=0.0` and `tap_success=0.0`, audit line 4 shows the actual best contact shortfall worsened/no-better (`shortfall_min=0.009535881m` versus step120 `0.009376528m` and previous h580 `0.009437712m`), and audit line 7 sets `CONTINUOUS_STEP_CLIPPED_DIFFIK_H580_STILL_OUTSIDE_STRICT_CONTACT_BAND`. Professor weak physical evidence remains PASS, but strict contact-gated positive-control, DiffIK action dataset, tiny action dataset dry run, PPO/RL, large dataset, and RoArm remain BLOCKED. Next local-only unblock is target/actual contact trajectory and reach-contract audit design; do not relax the contact gate or start dataset/RL/RoArm from this result.

2026-06-09 latest cube10cm tap RL unblock state: md was stale at slow240, now corrected. Built-in IsaacLab `DifferentialIKController` parity was added as default-off harness modes. Full `joint_pos_des` direct apply FAILED: target path was OK (`target_inside_max=1.0`, FK-vs-Isaac TCP error `0.0mm`) but contact/tap stayed `0.0` and full-target actuator follow was worse (`follow_final=0.447358370rad`), so do not call this clean parity. The 3cm-style step-clipped built-in DiffIK mode fixed the follow-lag portion (`follow_final=0.008570671rad`, clipped delta `0.010000000rad`) and preserved professor weak evidence PASS, but strict contact/tap remained `0.0`; actual best face gap stayed outside band (`-0.019376528m`, shortfall `0.009376528m`). A local 3cm horizon/progress design selected h580 (`steps 120->580`, `closed_loop_push_steps 72->580`) because 3cm used 580 steps / 6.08s, but the tiny h580 runtime still FAILED and revealed a new contract blocker: the 10cm env still truncates at its 1.2s episode cap (`truncated_count=8`), so h580 did NOT prove a continuous 5.8s horizon. Current primary blocker is `ENV_EPISODE_LENGTH_1P2S_TRUNCATES_H580_HORIZON_TEST`; next local-only unblock is a default-off episode-length override design before repeating step-clipped horizon. Contact-gate relaxation, dataset/RL/PPO/large dataset/RoArm remain BLOCKED.

2026-06-09 slow240 update: one explicit local tiny direct-IK-apply runtime changed only `closed_loop_push_steps 72 -> 240`; geometry/action wrapper knobs stayed unchanged. It still FAILED contact-gated positive-control (`contact=0.0`, `tap=0.0`) while preserving professor weak physical reaction (`professor_seen=1.0`). Follow error improved (`0.362854958 -> 0.170210123rad`) and shortfall improved only `0.000342280m`, but actual TCP remained outside the contact band (`slow240_shortfall_min=0.009191336m`). Verdict is `FAIL_SLOW240_IMPROVES_FOLLOW_BUT_NOT_CONTACT`; DiffIK action dataset, tiny action dataset dry run, PPO/RL, large dataset, and RoArm remain BLOCKED. Next allowed work is local-only design for the next timing/contact diagnostic, not dataset/RL/RoArm.

2026-06-06 local update: `sim_scripts/cube10cm_reaction_window_contract_audit.py` defines contact-anchored reaction windows as the next data-unit label contract. seed957/949/950 existing traces pass 16/16 windows; seed948 negative control fails 0/16 due missing contact anchor. Quality-tier v2 is now active: seed957 and seed949 are all Tier B, seed950 is 10 Tier B + 6 Tier C, seed948 is 16 Rejected. Clean DiffIK teacher is a quality tier, not the absolute tap/reaction filter. This is local/posthoc only and does not authorize 1024/10240/data/RL/VLA/Track A.

2026-06-07 tier-matrix update: `sim_scripts/cube10cm_reaction_window_tier_matrix.py` now joins reaction-window audits back to trace CSV direction/workspace metadata. Existing seed948/949/950/957 matrix initially had 64 candidate windows, 48 accepted (`acceptance_rate=0.75`), 42 Tier B, 6 Tier C, 16 Rejected, and zero Tier A. After explicit approval, ran one tiny local IsaacLab y+ coverage screen seed958: reaction/contact/no-posewrite/no-overshoot PASS, reaction-window 16/16, all Tier C (`follow_p95_to_cap_p95=1.151057652`, clip mean `1.0`). Then ran one tiny local y- coverage screen seed959: reaction-window 16/16, 11 Tier B + 5 Tier C (`follow_p95_to_cap_p95=1.006378446`, clip mean `1.0`). Then ran y+ cap050 seed960, changing only `max_diffik_joint_step_rad 0.035 -> 0.050`: reaction/contact/no-posewrite/no-overshoot PASS and reaction-window 16/16, but still all Tier C (`follow_p95_to_cap_p95=1.141746044`, clip mean `1.0`). Then ran y+ stiffness600 seed961, changing only `arm_stiffness_override 400 -> 600`: reaction/contact/no-posewrite/no-overshoot PASS and reaction-window 16/16, but still all Tier C (`follow_p95_to_cap_p95=1.200965473`, clip mean `1.0`). Updated matrix now has 128 candidate windows, 112 accepted (`acceptance_rate=0.875`), 53 Tier B, 59 Tier C, 16 Rejected, zero Tier A. y+ is now 51/51 accepted but all Tier C, so cap-only and stiffness-only cleanup did not fix y+ follow-lag quality. y- is no longer under-sampled (`20/20` accepted, 12 Tier B + 8 Tier C). x- remains config-mixed: seed948 old x- geometry is 16/16 Rejected while seed949 height050 x- is 16/16 Tier B and seed950 x- is 5/5 Tier B. Matrix readiness remains `ready_for_1024_or_data=false`; the matrix labels this as `direction_x-_config_mixed_acceptance_rate=0.567568_inspect_audit_direction` instead of implying direction-only x- failure. Do not start 1024/10240/data.

2026-06-07 local y+ failure diagnosis: added `sim_scripts/cube10cm_yplus_tierc_failure_diagnostic.py`, a local/posthoc-only per-window comparison across existing seed949/950/957/958/959/960/961 reaction-window traces. Result: y+ Tier C windows (`n=51`) have follow p95/cap p95 `1.223191874` versus Tier B non-y+ baseline `1.030052730`, but raw IK delta p95 is lower (`0.174502504` vs `0.280128609`, ratio `0.622168298`), so the simple "bigger raw IK demand" hypothesis is not supported. y+ contact/reaction is strong and early: mean max XY displacement is `0.012257356m`, `10.223485837x` Tier B non-y+ baseline, and anchor/contact step is about `80.876804` steps earlier with phase alpha near zero. Verdict: support `yplus_geometry_follow_coupling`, not `simple_raw_ik_demand`; next local question is why y+ contacts so early and moves the cube ~10x more while follow lag exceeds Tier B. No GPU/data/1024/RL/VLA/Track A.

2026-06-07 local y+ early-contact geometry audit: added `sim_scripts/cube10cm_yplus_early_contact_geometry_audit.py`, another local/posthoc-only audit over existing reaction-window traces. It confirms the stronger interpretation: y+ does not merely have a unique measured-contact lead; instead, it accumulates large object reaction inside the approach/pre-anchor window. y+ Tier C windows have first reaction step mean `46.039216`, measured contact step mean `181.176471`, and anchor about `40.235294` steps before push phase starts; non-y+ Tier B baseline has contact after push start (`anchor_minus_push_start_mean=40.641509`). In the 24 steps before the contact anchor, y+ mean max XY displacement is `0.012257356m` versus baseline `0.000895049m` (`13.694612400x`), and tip is `13.261459369deg` versus `1.004456226deg` (`13.202625486x`). Initial target along/lateral are the same nominal `-0.060/-0.020m`, but y+ target z is at cube side-center (`~0m` above cube z) while the mixed non-y+ baseline averages `0.034905637m`; y- shares low side-center yet stays low pre-anchor, so low height alone is not sufficient. Verdict: support `yplus_preanchor_reaction_accumulation` and `yplus_approach_phase_geometry_hypothesis`; next local step is not GPU but a config audit/proposed tiny screen around y+ precontact/lateral/height/timing.

2026-06-07 local y+ precontact candidate audit: added `sim_scripts/cube10cm_yplus_precontact_candidate_audit.py`, a local/config-only pre-runtime audit for the narrow next y+ screen. It keeps the professor tap/reaction objective, changes no trace data, and proposes exactly one tiny candidate: fixed y+ seed958-like geometry with only `precontact_clearance_m 0.010 -> 0.020`. Existing evidence remains y+ `51` windows, pre-anchor displacement ratio `13.694612400`, tip ratio `13.202625486`, anchor `40.235294` steps before push start, raw delta ratio `0.622168298`, follow ratio `1.187431643`. Nominal target geometry changes from pre-target along `-0.060000m` to `-0.070000m`; through target remains `-0.040000m`; push path increases `0.020000m -> 0.030000m`. Verdict: `supports_precontact_first=True`, `candidate_is_tiny_one_variable_change=True`; height/lateral first are rejected for now. GPU was NOT run; next seed would be seed962 only after explicit approval.

2026-06-07 local y+ pre020 runtime result: after explicit approval and passing guards, ran exactly one local IsaacLab 16-env fixed y+ seed962 screen changing only `precontact_clearance_m 0.010 -> 0.020`. Reaction/contact/no-posewrite/no-overshoot PASSed (`reaction_gate_pass=true`, reaction/contact `1.0/1.0`, overshoot `0.0`, posewrite `0`), but this is not data/1024 readiness: `teacher_quality_ready=false`, `diffik_clip_rate_mean=1.0`, final TCP error `0.051811996m`, `controlled_push_rate=0.5625`, and low-motion `1.0`. Reaction-window audit accepted 16/16 windows with 2 Tier B + 14 Tier C, zero rejected, follow p95/cap p95 `1.160505840`. Compared to seed958/960/961, pre020 delayed contact/anchor and reduced y+ pre24 reaction (`pre24_disp_mean=0.005104796m`, `pre24_tip_mean=5.079945deg`, anchor after push start by `27.1875` steps) but also weakened reaction strength (`max_disp_along_push_mean=0.002923813m`, `max_tip_angle_mean=9.205450deg`). Updated seed962-inclusive matrix: 144 windows, 128 accepted, 55 Tier B, 73 Tier C, 16 Rejected, zero Tier A; y+ is 67/67 accepted but only 2 Tier B + 65 Tier C. Verdict: `precontact=0.020` partially reduces early/pre-anchor y+ accumulation but mostly moves the failure toward weaker late contact while preserving DiffIK/follow quality blockers. Next research step is local-only timing/contact-strength separation around y+ geometry; do not run another GPU/precontact sweep, 1024/10240/data, RL/VLA, Track A, or B200.

2026-06-07 local y+ pre020 failure-shift audit: added `sim_scripts/cube10cm_yplus_pre020_failure_shift_audit.py`, a no-GPU/no-data audit comparing seed958/960/961/962 with the same per-window timing/pre24 metrics. Result: seed962 pre020 is a real early-reaction reduction (`pre24_disp_vs_prev_mean=0.415034926`, `pre24_tip_vs_prev_mean=0.381066167`, anchor after push start), but it also weakens reaction strength (`max_disp_vs_prev_mean=0.661469914`, `max_tip_vs_prev_mean=0.376103186`, controlled push `0.5625`) and quality remains blocked. Verdict: `pre020_reduces_preanchor_reaction=True`, `pre020_weakens_reaction_strength=True`, `quality_still_blocked=True`. Next research action remains local-only: design a path/timing/contact-strength candidate before any further tiny GPU; do not continue blind precontact/lateral/height/actuator/DLS/cap sweeps.

2026-06-07 local y+ contact-strength correction: `sim_scripts/cube10cm_yplus_contact_strength_candidate_audit.py` now treats final 1mm retention as a secondary diagnostic only, not a task objective and not a next-GPU trigger. The audit still records seed962 max 1mm gate `1.0`, final 1mm gate `0.5625`, retention `0.462406074` versus prior y+ mean `0.737681608`, but the corrected verdict is `final_retention_primary_objective=False` and `selected_next_candidate=NONE_FROM_FINAL_RETENTION_ALONE`. If stronger 2-3mm transient push is explicitly requested later, `contact_stop_disp_m 0.001 -> 0.002` can be treated as an optional diagnostic candidate, not as a fix for final-position retention. Current judging order remains reaction/contact/no-posewrite/no-overshoot first, optional max 1/2/3mm transient strength second, quality tier third, final 1cm secondary only if explicitly requested. GPU was NOT run.

2026-06-07 local y+ transient tap-strength audit: added `sim_scripts/cube10cm_yplus_transient_tap_strength_audit.py`, a no-GPU/no-data audit that intentionally excludes final position as a success gate. Result: seed962 pre020 PASSes the primary 1mm tap event (`contact=1.0`, `reaction=1.0`, `overshoot=0.0`, max 1mm `1.0`) and has majority 2mm transient strength (`0.8125`), but 3mm is not reliable (`0.5`). Compared with seed958/960/961, seed962 is deliberately less aggressive: max displacement ratio `0.661469914`, tip ratio `0.376103186`, z ratio `0.492015176`. Next order: if 1-2mm tap is enough, stop y+ contact-geometry tuning and keep quality-tier metadata separate; if 3mm is explicitly required, define that transient target first and then propose exactly one local candidate. Do not use final 1cm/final retention.

2026-06-07 dataset/RL/robot readiness gate: added `sim_scripts/cube10cm_dataset_rl_robot_readiness_audit.py`, then created only the allowed local event-label manifest with `sim_scripts/cube10cm_event_label_dataset_manifest.py`. Readiness result: primary event and 1-2mm objective are ready (`contact=1.0`, `reaction=1.0`, `overshoot=0.0`, max1mm `1.0`, max2mm `0.8125`), but action-teacher dataset is NOT ready (`clean_teacher=false`, 2 Tier B + 14 Tier C, clip mean `1.0`, follow p95/cap `1.160505840`). Pipeline gates: `event_label_dataset_ready=True`, but `large_isaaclab_dataset_ready=False`, `isaaclab_rl_ready=False`, `roarm_m3_pro_deploy_ready=False`. Generated manifest is schema/label-only, not train data: 16 events, 16 contact, 16 reaction, 0 overshoot, window-level transient counts 16/13/7 for 1/2/3mm, and it explicitly excludes final 1cm/final retention. Do not run dataset generation/RL/robot deploy from this state.

2026-06-08 local DiffIK action-dataset blocker audit: added `sim_scripts/cube10cm_diffik_action_dataset_blocker_audit.py` to answer whether the IsaacLab built-in DifferentialIK action-teacher dataset can start. Verdict: event-label dataset remains `READY_LOCAL_ONLY` with 16 events and 1/2/3mm counts 16/13/7, but `differential_ik_action_teacher_dataset=BLOCKED`, `large_isaaclab_dataset=BLOCKED`, `isaaclab_rl=BLOCKED`, and `roarm_m3_pro=BLOCKED`. Evidence: clean teacher false, 2B+14C, clip mean `1.0`, follow p95/cap `1.160505840`, final TCP error `0.051811996m`, trace modes `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, `ACTUATOR_TARGET_TRACKING_LAG`; old dataset builder still has final controlled/low-motion/success filters, and existing RL env is 3cm/20g relocation-oriented. Next unblock order: keep 1-2mm tap objective, use event-label manifest only, write a 10cm tap-specific dataset-builder preflight without final-success filters, resolve/explicitly gate DiffIK teacher quality, validate 10cm tap RL env random sanity, then train, then robot safety/replay. No GPU/data/training/robot/B200/SSH was run.

2026-06-08 local unblock step: added `sim_scripts/cube10cm_tap_reaction_dataset_builder_preflight.py` and `sim_scripts/cube10cm_diffik_teacher_quality_policy_gate.py`. The preflight unblocks the old-builder conflict locally: it builds a 16-row event-label preview from the manifest, keeps contact/reaction/overshoot/transient/tier fields, and passes the forbidden gate check (`forbidden_present=[]`, `uses_final_success_filter=NO`, `uses_final_1cm_or_retention=NO`). This is still not an action dataset. Teacher quality policy remains blocked by default: strict clean action teacher is BLOCKED (`clean_teacher=false`, Tier A/B/C = 0/2/14, clip mean `1.0`, follow p95/cap `1.160505840`, final TCP error `0.051811996m`), Tier-B-only action teacher is `BLOCKED_INSUFFICIENT_ROWS` (2/16), and Tier-B/C noisy action teacher requires an explicit policy exception. Therefore large IsaacLab dataset/RL/RoArm remain blocked unless teacher quality is improved/retested or an explicit noisy-teacher exception is recorded first. No GPU/data/training/robot/B200/SSH was run.

2026-06-08 teacher-quality revalidation: added `sim_scripts/cube10cm_teacher_quality_revalidation_audit.py` and `sim_scripts/cube10cm_tierb_action_dryrun_preview.py`. Local revalidation swept anchor-relative action-row slices on existing seed962 trace. The official `[-24,+48]` reaction window is 16/16 accepted but 2B+14C with clip mean `1.0` and follow p95/cap `1.140652384`; the best trimmed action-row policy is `contact_to_p16` `[0,+16]`, 16/16 accepted and 16B+0C with follow p95/cap `0.251552037`, but still clip mean `1.0` and zero Tier A. So the Tier-C follow-lag blocker is partly a row-window definition issue, but clean teacher remains blocked by command clipping/control tracking. Built only a tiny local Tier-B action dry-run preview: 16 events, 66 sparse trace rows, no forbidden final/success fields, action abs p95/max `0.007rad`, all rows clip-any. Actual action-teacher dataset is NOT built; large dataset/RL/RoArm remain blocked. No GPU/data/training/robot/B200/SSH was run.

2026-06-08 visual/sim sanity update: added `sim_scripts/cube10cm_visual_sanity_trace_storyboard.py` and `sim_scripts/cube10cm_visual_sim_sanity_audit.py`, and fixed `sim_scripts/cube3cm_push_diffik_render_trace.py` so replay rendering tolerates string trace columns and uses trace `cube_size_*_m` instead of the old hardcoded 3cm cube scale. The live `--record_video` attempt produced no frames/summary and is not evidence. The trace replay renderer produced a 98-frame env0 MP4 with 10cm cube size, real local RoArm STL mesh, `physics_recomputed=false`, `training=false`, and `dataset_generation=false`; MP4 decode passes. Critical verdict: `visual_contact_replay_pass=True`, but `clean_tap_visual_verified=False` because the contact frame still has large vertical target error (`tcp_z=0.100452900`, `target_z=0.049999580`, delta `0.050453320m`), TCP target error `0.050612349m`, and `clip_any=1`. Visual sanity therefore does not unblock action-teacher dataset, RL, or RoArm-M3-Pro.

2026-06-08 contact-frame geometry mismatch audit: added `sim_scripts/cube10cm_contact_frame_geometry_mismatch_audit.py`, a local trace-only audit over all 16 seed962 first-contact rows. It confirms the 5cm visual blocker is not a missing TCP-local-offset compensation bug: `tcp_link5_offset_check consistent=True`, max offset-consistency error `0.000000007m`. The actual problem is side-center target not being reached under clipping: first-contact `tcp_minus_target_z_mean=0.052857013m`, matching `link5_minus_target_z_mean=0.052857012m`; z accounts for `0.983196354` of TCP-target error; TCP is near the live cube top in 16/16 contacts (`tcp_above_live_cube_center_z_mean=0.048793540m`, `tcp_below_live_cube_top_z_mean=0.001206460m`, `tcp_near_top_10mm_rate=1.0`, `tcp_near_center_10mm_rate=0.0`); clipping at first contact is `1.0` and always maxes at `link1_to_link2`. Verdict: `SIDE_CENTER_TARGET_NOT_TRACKED_TCP_CONTACTS_NEAR_TOP_UNDER_CLIPPING`. Do not fix this by simply lifting target z: prior seed944 height050 reduced TCP error but had contact evidence `0.0` and reaction gate false. Next work remains local teacher contact-geometry/control-tracking design; no dataset/RL/RoArm.

2026-06-08 teacher contact-frame design + one negative-control runtime: added `sim_scripts/cube10cm_teacher_contact_frame_design_audit.py` to compare three candidate teacher criteria. Result: `true_side_center_tcp` is semantically correct but current seed962 fails tracking (`side_center_z_reached_10mm_rate=0.0`, z err mean `0.052857013m`); `upper_edge_contact_proxy` best explains current visual contact (`upper_z_reached_10mm_rate=1.0`, z err mean `0.001206460m`) but teaches top contact; `tool_oriented_side_contact_proxy` is selected as teacher criterion because it preserves side-contact semantics, though current DiffIK is `command_type=position` and cannot validate orientation/proxy from trace alone. After guards and explicit local runtime approval, ran exactly one 16-env top-margin negative control changing only `--tcp_height_mode top_margin` on seed962 geometry. It improved target tracking (`diffik_clip_rate_mean=0.495833354`, final TCP err `0.011275044m`, reaction gate teacher_quality `READY`) and reaction-window quality to 16 Tier B, but weakened the tap (`max_disp=0.001112372m`, final disp `0.000045329m`, controlled push `0.0`, low-motion `1.0`) and still contacts upper/top proxy (`tcp_above_live_cube_center_z_mean=0.057616640m`, `tcp_below_live_cube_top_z_mean=-0.007616640m`). Runtime comparison verdict: `upper_edge_proxy_tracking_improved=True`, `upper_edge_proxy_tap_strength_weakened=True`, `upper_edge_proxy_selected_as_teacher=False`, selected teacher criterion remains `tool_oriented_side_contact_proxy`. Detailed session: `claudedocs/session_20260608_cube10cm_teacher_contact_frame_design.md`. No dataset/RL/RoArm/B200/SSH.

2026-06-08 local tool/contact-proxy + orientation preflight: added `sim_scripts/cube10cm_tool_contact_proxy_orientation_preflight.py`, a no-GPU/no-data/no-robot audit over existing seed962 side-center and top-margin traces plus local URDF/STL collision assets. It confirms FK reconstruction is trustworthy (`fk_tcp_err_p95=0.000001505m` side-center; `0.000001510m` top-margin), so mesh-proxy calculations are meaningful. Current hand TCP is not side-center (`side_center_dist_mean=0.056645103m`, z err `0.048793540m`), gripper collision is not the contact proxy (`side_center_dist_mean=0.094208332m`, AABB overlap `0.0`), and best existing collision proxy is link5, not gripper (`mesh_mode=link5_collision`, label `corner_011`, overlap `1.0`) but still upper/offset (`side_center_dist_mean=0.040192105m`, z err `0.034543147m`, near10/near20 `0.0/0.0`). For side-center only, retargeting that stable link5 corner would reduce current link5 target error (`0.053769121m -> 0.040192105m`, ratio `0.748512386`) but requires moving link5 down about `0.034543147m` and remains orientation/contact-semantics unvalidated because the probe is position-only and trace has no link5 quaternion. In top-margin, the same proxy is worse (`ratio=3.801698970`), confirming top-margin is not the teacher path. Verdict: do NOT run GPU yet; next is local code preflight for `link5_collision:corner_011` proxy with pose or trace support before any tiny runtime. No dataset/RL/RoArm/B200/SSH.

2026-06-08 link5 proxy pose/trace code-contract update: added default-preserving probe support for `--tool_contact_proxy_mode link5_collision_corner_011`, `--diffik_command_type position|pose`, and `--diffik_pose_quat_mode current_link5|initial_link5`, plus trace/summary columns for tool proxy target error and link5 quaternions. Added `sim_scripts/cube10cm_link5_proxy_pose_trace_contract_audit.py`, a local-only/no-runtime audit. Result: default hand-TCP position path is preserved, runtime mapping is present, 29 required trace fields and 7 summary keys are present, and code contract is ready for exactly one tiny runtime consideration (`code_contract_ready_for_one_tiny_runtime_consideration=True`). Critical decision: pose support exists but first tiny runtime should NOT use pose yet (`pose_first_runtime_recommended=False`) because that would mix proxy retargeting with a 6D pose constraint on a 5-joint arm. The only next candidate to consider, after explicit approval, is fixed seed962 y+ pre020 geometry with `tool_contact_proxy_mode hand_tcp -> link5_collision_corner_011` and `command_type=position`, keeping lateral/height/actuator/DLS/cap unchanged. Dataset/RL/RoArm remain blocked.

2026-06-08 one tiny link5-corner runtime result: after guards and explicit approval, ran exactly one local IsaacLab 16-env fixed seed962 y+ pre020 screen changing only `--tool_contact_proxy_mode link5_collision_corner_011` with `--diffik_command_type position`. Reaction/contact/no-posewrite/no-overshoot PASSed (`reaction=1.0`, `contact=1.0`, `overshoot=0.0`, posewrite `0`), and proxy tracking improved (`diffik_clip_rate_mean 1.0 -> 0.515544884`, final TCP err `0.051811996m -> 0.038090036m`, final tool-proxy target err `0.002636088m`). Reaction windows stayed 16/16 and quality improved from 2B+14C to 16B; follow p95/cap improved `1.160505840 -> 0.201606750`. But this is NOT dataset/RL readiness: clean teacher is still false because window clip mean is `0.666666667` and reaction strength weakened (`max_disp 0.002923813m -> 0.001431603m`, speed `0.127446551 -> 0.024424722m/s`, 5/10/20/30mm rates all `0.0`, low_motion `1.0`). Verdict: proxy retargeting solved part of the tracking/quality-tier problem but moved toward a weaker 1mm tap. Do not scale dataset/RL/RoArm; next should be visual proxy-contact inspection or exactly one strength-preserving proxy variant only if needed.

2026-06-08 visual proxy-contact inspection: added `sim_scripts/cube10cm_link5corner_visual_proxy_contact_inspection.py`, a local trace-only HTML/SVG + JSON inspection over the link5-corner runtime. It confirms the proxy is not top contact anymore: contact proxy z is near live cube center (`proxy_minus_cube_center_z_mean=0.000392607m`, side-center z near5mm rate `1.0`) and about `0.049607393m` below cube top (`proxy_not_top_rate=1.0`). However, the weak tap is real: at contact the proxy remains outside/grazing the live approach face (`proxy_gap_to_live_side_face_mean=-0.006247562m`, outside rate `1.0`), the target is also already outside the live face (`-0.002771095m`, outside rate `1.0`), contact and stop are the same rollout step (`1.0`), and max displacement/speed stay weak (`0.001431603m`, `0.024424722m/s`, low-motion `1.0`). Verdict: side-center proxy semantics improved, but clean tap strength is NOT visually verified; dataset/RL/RoArm remain blocked. If 1mm tap is enough, stop contact-geometry GPU tuning and keep quality-tier metadata separate; if 2-3mm tap is required, define that transient requirement first and design one local strength-preserving contact timing/through candidate before any runtime.

2026-06-08 10cm tap RL preflight/policy gate: clarified that Isaac Lab itself is OK under local GPU evidence; CPU/sandbox diagnostic failure must not be promoted over the accepted local RTX 4090/cuda:0 sanity. The trace-derived visual inspection and the new `RoArm-CubeTap10cm-Direct-v0` wrapper are separate layers: the former explains link5 contact geometry, the latter prevents RL objective mismatch. Added `sim_scripts/cube10cm_tap_rl_preflight_policy_gate.py`, a local-only consolidation over the event-label manifest, noisy Tier-B teacher policy, visual proxy-contact JSON, and runtime gate audit. Result: event-label/quality-tier metadata is `READY_LOCAL_ONLY`, env wrapper is `READY_LOCAL_PREFLIGHT_ONLY`, weak 1mm is the only verified objective evidence, and strong 2-3mm is not required by current evidence. Still blocked: strict action teacher, noisy Tier-B action-teacher exception, tiny action dataset, PPO/RL, large dataset, and RoArm. Next allowed local-only work is reward/done/log contract freeze plus a scripted positive-control tap sanity design; no new GPU runtime without explicit approval.

2026-06-08 10cm tap RL contract freeze + positive-control design: added `sim_scripts/cube10cm_tap_rl_contract_positive_control_design.py`, a local design/static audit only. Result: objective/reaction contract frozen (`final_1cm_required_default=NO`, tap target `0.001m`, overshoot `0.020m`, contact-gated reaction true), reward/done contract frozen with final-success leak count `0`, log contract frozen with separate raw reaction/contact context/reaction seen/overshoot/success logs. A future scripted positive-control sanity is designed but NOT run: `RoArm-CubeTap10cm-Direct-v0`, `cuda:0`, `num_envs=2`, `max_steps=120`, scripted TCP DifferentialIK-to-joint-delta actions, pass only if contact/reaction_context/reaction_seen/tap_success are >0 while overshoot=0 and final flag=0. This unblocks only consideration of one explicitly approved tiny positive-control runtime; PPO/RL, large dataset, action teacher, tiny action dataset, and RoArm remain blocked.

2026-06-08 10cm tap RL positive-control result: after explicit user approval, added and ran `roarm_rl/test_positive_control_cube_tap10cm.py` as a tiny local RTX4090/cuda:0 IsaacLab sanity using local USD, `RoArm-CubeTap10cm-Direct-v0`, `num_envs=2`, `max_steps=120`, fixed cube `(0.250,0.000)`, push dir `(+1,0)`, side-center TCP target (`tcp_top_margin_m=-0.050`), and no dataset/PPO/robot/B200/SSH/Track A. First launch was `BLOCKED` before env creation by a harness-only `TerrainImporterCfg(use_terrain_origins)` incompatibility; fixed the harness and reran. Actual rollout `FAIL`ed the positive-control gate: reset IK and teacher goal were OK (`1.0/1.0`), but contact never registered (`contact_seen=0.0`), raw reaction speed fired without contact context (`reaction_signal=1.0`, `reaction_context=0.0`, `reaction_seen=0.0`), tap success stayed `0.0`, overshoot stayed `0.0`, max disp was only `0.000824004m`, and final face gap was still outside the contact band (`-0.021077018m`, shortfall `0.011077018m`). This is useful negative evidence: the wrapper correctly blocked a false positive reaction-without-contact, but positive-control tap is NOT passed; PPO/RL, large dataset, action teacher, tiny action dataset, and RoArm remain blocked. Added local failure audit and revised one-candidate design: `external_closed_loop` controller mode is code-ready but NOT run; any new GPU runtime requires explicit approval.

2026-06-08 10cm tap RL positive-control visual contact audit + strict candidate correction: added `sim_scripts/cube10cm_tap_rl_positive_control_visual_contact_audit.py`, a local existing-log-only PNG/SVG/HTML inspection of the failed positive-control contact frame. Important limitation: it reconstructs only reset/final scalar geometry, not a per-step video trace. It confirms contact stayed `0.0` because the controller did not close the live approach-face gap: initial/final along were `-0.070252299m/-0.071077018m`, initial/final face gap `-0.020252299m/-0.021077018m`, final shortfall to the `[-0.010,+0.010]m` contact band `0.011077018m`. Lateral and height were not the blockers (`final_lateral=0.000003905m`, `final_vertical_offset=0.000538668m`), while gap delta `-0.000824720m` nearly cancels cube disp `0.000824004m` (`abs=0.000000715m`). Critically corrected the revised candidate design: the selected next candidate is now strict `changed_knobs=1` (`controller_mode=builtin_teacher->external_closed_loop`) with smoothing/scale left at env defaults (`0.25/0.35`); the older strength variant with smoothing/scale `1.0/1.0` is `NOT_SELECTED_NOT_RUN` because it would mix variables. Verdict remains: wrapper false-positive guard PASSed, positive-control tap did NOT pass, PPO/RL/large dataset/action teacher/RoArm remain blocked; any new GPU runtime still requires explicit approval.

2026-06-08 strict external-closed-loop positive-control runtime result: after explicit user approval, ran exactly one local RTX4090/cuda:0 tiny positive-control runtime with existing local USD, `num_envs=2`, `max_steps=120`, `seed=962`, strict `controller_mode=external_closed_loop`, smoothing/defaults unchanged (`action_smoothing_alpha=0.25`, `contact_joint_delta_scale=0.35`, `closed_loop_push_steps=72`), and no dataset/PPO/robot/B200/SSH/Track A. Result again `FAIL`: external IK solved (`closed_loop_ik_ok_rate=1.0`, mean err `0.617469787mm`) but contact still stayed `0.0`, raw reaction signal `1.0`, reaction context/seen/tap success `0.0`, overshoot `0.0`, max disp `0.000824124m`, speed `0.077135637m/s`. Visual/failure audits show final face gap improved only `0.000558525m` versus builtin but remained outside band (`external_final_face_gap=-0.020518493m`, shortfall `0.010518493m`); lateral/vertical were OK. Corrected harness pass gate locally so external mode uses `closed_loop_ik_ok_rate` rather than stale `teacher_goal_ok_rate`; verdict is unchanged because contact was still zero. Added tap-wrapper actuator diagnostic log keys for future runs (`cube_push_tcp_cube_dist_m`, `cube_push_joint_delta_abs_mean`, `cube_push_contact_slowdown_mean`, `cube_push_teacher_blend_mean`), but did NOT run another GPU after instrumentation. Next is local-only design/instrumentation of one actuation-limit candidate; any further runtime requires new explicit approval.

2026-06-08 strict external instrumented action-path runtime: after user asked to proceed, ran one additional local RTX4090/cuda:0 tiny strict external sanity using the newly added action-path logs. Result still `FAIL`; contact/reaction context/tap success stayed `0.0`. New line 7 evidence: `tcp_cube_dist_m=0.070519388`, `joint_delta_abs_mean=0.005000000`, `contact_slowdown_mean=1.0`, `teacher_blend_mean=0.0`, `action_penalty=-0.015`. Interpretation changed: contact slowdown is NOT active, and the mean joint delta is below the `0.010rad` per-step cap, so direct slowdown/cap blame is not supported by this final-scalar evidence. The live face gap still remains the blocker (`final_face_gap=-0.020518493m`, shortfall `0.010518493m`). Added per-step aggregate trace instrumentation to the harness (`face_gap min/max/final`, contact-band shortfall min/final, tcp_dist min, joint_delta max, controller trace stats) but did NOT launch another GPU after that. Differential IK action dataset has NOT been built; dataset/RL/RoArm remain blocked until contact-gated positive-control passes and teacher/action quality policy is resolved.

2026-06-08 strict external TCP-progress instrumented sanity: after user asked to verify whether face gap ever closes, ran one local RTX4090/cuda:0 tiny strict external sanity using the new per-step aggregate trace. Result still `FAIL`. Trace line 8: initial face gap `-0.020252299m`, best/closest face gap `-0.019507330m`, worst `-0.024245869m`, final `-0.020518493m`; best improvement from initial was only `0.000744969m`, and the best contact-band shortfall remained `0.009507330m` (`face_gap_near_band=False`). Action-path line 7 stayed unchanged (`contact_slowdown_mean=1.0`, `joint_delta_abs_max=0.005000000`). Interpretation: the TCP does move slightly toward the band mid-run, so it is not a total action-mapping no-op, but it never gets close enough for timing/contact-band to be the main explanation. The next local-only candidate should target action progress/gain/target application, not dataset/RL/RoArm. Any new GPU runtime requires explicit approval.

2026-06-08 action-progress candidate design: added `sim_scripts/cube10cm_tap_rl_action_progress_candidate_design.py`, a local design/static audit only. It reads the TCP-progress result audit and runtime JSON plus the harness/env source; it launches no GPU runtime and generates no dataset/training/robot action. Result: `code_ready=True`, `basis_ok=True`, previous contact/reaction context/tap success all `0.0`, best face-gap improvement only `0.000744969m`, best shortfall `0.009507330m`, previous smoothing `0.25`, closed-loop alpha final `1.0`, slowdown `1.0`, joint-delta abs max `0.005000000`. Selected next candidate is exactly one runtime knob: `action_smoothing_alpha 0.25 -> 1.0`, while controller mode, closed-loop steps, contact delta scale, geometry, side-center height, precontact, through distance, and gates remain fixed. `contact_joint_delta_scale`, `closed_loop_push_steps`, goal push/contact band, and `joint_delta_reference` are not selected first. This only makes a future tiny cuda:0 positive-control runtime ready for explicit approval; DiffIK action dataset/tiny dry run/PPO/large dataset/RoArm remain blocked.

2026-06-08 action_smoothing_alpha=1.0 positive-control result: after explicit approval, ran one tiny local RTX4090/cuda:0 runtime with `RoArm-CubeTap10cm-Direct-v0`, `num_envs=2`, `max_steps=120`, `seed=962`, `controller_mode=external_closed_loop`, and the single changed knob `action_smoothing_alpha 0.25 -> 1.0`; no dataset/PPO/robot/B200/SSH/Track A. Result still `FAIL`: contact `0.0`, reaction context/seen/tap success `0.0`, overshoot `0.0`, raw reaction signal `1.0`, max disp `0.000820309m`, final face gap `-0.020514678m`, best shortfall `0.009533616m`. Comparison audit shows smoothing did not improve contact progress: best improvement delta vs baseline `-0.000026286m`, best shortfall worsened by `0.000026286m`, final gap changed only `+0.000003815m`, and the mean joint-delta trace stayed `0.005000000`. Visual audit confirms final TCP remains outside the contact band while lateral/height are OK. Verdict: `FAIL_SMOOTHING_NOT_ROOT_CAUSE`; DiffIK action dataset/tiny dry run/PPO/large dataset/RoArm remain blocked. Critical correction: previous `joint_delta_abs_max` summaries were max-over-time of mean joint delta, not per-joint cap evidence; added env/harness logs for per-joint delta max, cap rate, action abs mean/max, target lead abs mean/max, and target lead-limit rate for the next approved diagnostic.

2026-06-08 cap/target-lead diagnostic result: after user asked to proceed, ran one tiny local RTX4090/cuda:0 diagnostic using the new per-joint/action/lead logs, default strict external settings (`action_smoothing_alpha=0.25`, `contact_joint_delta_scale=0.35`, `closed_loop_push_steps=72`), no dataset/PPO/robot/B200/SSH/Track A. Result still `FAIL`, but the action-path blocker is now clearer: action command is saturated (`action_abs_max=1.0`, mean `0.5`), per-joint delta hits the env cap (`joint_delta_abs_max=0.010000000`, `joint_delta_cap_rate=0.5`), slowdown is inactive (`1.0`), and target lead-limit appears only as secondary trace evidence (`target_lead_limit_rate_trace=0.333333343`, final `0.0`). Result audit verdict: `CAP_ACTION_SATURATION_PRIMARY_HYPOTHESIS`; visual audit still shows final TCP outside contact band with lateral/height OK. Added `--max_joint_delta_per_step_rad` default-off harness override and `sim_scripts/cube10cm_tap_rl_cap_only_candidate_design.py`. The next candidate is local-designed but NOT run: exactly one knob, `max_joint_delta_per_step_rad 0.010 -> 0.040`, while action scale, smoothing, controller mode, geometry, target reference/lead limit, and gates stay fixed. Dataset/RL/RoArm remain blocked; any new cap040 cuda:0 runtime requires explicit approval.

2026-06-08 cap040 positive-control result: after explicit approval, ran exactly one tiny local RTX4090/cuda:0 runtime with `RoArm-CubeTap10cm-Direct-v0`, `num_envs=2`, `max_steps=120`, `seed=962`, `controller_mode=external_closed_loop`, and the single changed knob `max_joint_delta_per_step_rad 0.010 -> 0.040`; no dataset/PPO/robot/B200/SSH/Track A. Result still `FAIL`: contact/reaction context/tap success remained `0.0`, raw reaction signal `1.0`, overshoot `0.0`, and final face gap remained outside the band (`-0.020518493m`, shortfall `0.010518493m`). Cap override did apply (`joint_delta_abs_max_trace 0.010000000 -> 0.039999995`, cap rate `0.5 -> 0.0`), but best shortfall and best face gap did not improve at all (`0.009507330m`, `-0.019507330m` both unchanged). Visual audit confirms the final TCP is still outside the contact band while lateral/height are OK. Verdict: cap-only is falsified as the primary blocker; next local-only work is exactly one target-application candidate design around `joint_target_lead_limit` or `joint_delta_reference`. DiffIK action dataset, tiny action dry run, PPO/RL, large dataset, and RoArm remain blocked.

2026-06-08 target-application candidate design: added a default-off `--joint_target_lead_limit_rad` harness override and `sim_scripts/cube10cm_tap_rl_target_application_candidate_design.py`; no GPU runtime, no dataset/PPO/robot/B200/SSH/Track A. Local design audit is `READY_FOR_EXPLICIT_RUNTIME_APPROVAL_ONLY`: cap040 is the baseline, cap stays fixed at `0.040`, and the single changed knob for any future tiny runtime is `joint_target_lead_limit_rad 0.060 -> 0.120`. Basis is cap040 line evidence: cap-only falsified, cap no longer active, lead-limit observed (`target_lead_abs_max_trace=0.069168568`, `target_lead_limit_rate_trace=0.5`). Not selected first: `joint_delta_reference` because it changes target-base semantics and likely needs matching harness action-base review; action scale/geometry/smoothing/cap-only are rejected. Dataset/RL/RoArm and DiffIK action dataset remain blocked; any `cap040_lead120` runtime requires explicit approval.

2026-06-08 Isaac Lab source cross-check correction: local installed Isaac Lab task-space action code computes `joint_pos_des` from the IK controller and directly calls `set_joint_position_target`, while the current positive-control harness computes a TCP target/IK solution but then rewraps it as normalized RL action before env smoothing/action-scale/cap/lead-limit. Added default-off `external_closed_loop_direct_apply` support in the tap harness/env plus `sim_scripts/cube10cm_tap_rl_direct_ik_apply_candidate_design.py`; no GPU/data/training/robot/B200/SSH/Track A. Verdict: `direct_ik_apply_positive_control` is the next designed diagnostic, and `cap040_lead120` is reserve. Purpose: separate target geometry/IK failure from RL action target-application failure before more lead/cap sweeps or any DiffIK action dataset. Dataset/RL/RoArm remain blocked; any direct-IK-apply cuda:0 runtime requires explicit approval.

2026-06-08 direct-IK-apply tiny runtime result: after explicit approval, ran exactly one local RTX4090/cuda:0 `external_closed_loop_direct_apply` positive-control with `RoArm-CubeTap10cm-Direct-v0`, `num_envs=2`, `max_steps=120`, `seed=962`; no dataset/PPO/robot/B200/SSH/Track A. Result still `FAIL`: direct apply was active and RL action path was bypassed (`action_abs_max_trace=0`, cap/lead-limit rates `0`), closed-loop IK remained OK (`1.0`, mean err `0.617469787mm`), but contact/reaction/tap success stayed `0.0`, max displacement was only `0.000922590m`, best face gap stayed outside the band (`-0.019533616m`, best shortfall `0.009533616m`), and visual audit shows initial/best/final TCP all outside the contact band while lateral/vertical are OK. Verdict: wrapper-only explanation is falsified for this target/path; next is target-geometry/FK-frame/actuator-follow telemetry, not lead/cap/action-scale tuning. DiffIK action dataset/tiny dry run/PPO/RL/large/RoArm remain blocked.

2026-06-08 direct-IK telemetry candidate design: added local-only posthoc/visual audit `sim_scripts/cube10cm_tap_rl_direct_ik_apply_result_audit.py`, viewed the generated PNG, and added default-preserving telemetry in `roarm_rl/test_positive_control_cube_tap10cm.py` for target face gap, target inside-band rate, IK target FK error, actual-FK-vs-Isaac-TCP frame error, direct joint follow error, and actual joint step. Added `sim_scripts/cube10cm_tap_rl_direct_ik_telemetry_candidate_design.py`; no new GPU in this design step. Candidate is `READY_FOR_EXPLICIT_RUNTIME_APPROVAL_ONLY`: repeat the same direct-IK-apply tiny run with zero control-knob changes purely to collect telemetry. `cap040_lead120`, `joint_delta_reference`, action scale/smoothing, geometry changes, dataset/RL/RoArm all remain blocked/reserve until this cause split is known.

2026-06-08 professor physical-reaction gate separation: corrected the policy/code interpretation after user clarified that weak physical object reaction is acceptable for the professor 10cm/0.72kg push/tap objective. The tap env now logs a separate `professor_physical_reaction_*` evidence path with weak displacement/speed/z thresholds, while preserving the stricter contact-gated `tap_success` gate for RL/dataset/RoArm. Re-ran only local posthoc/policy audits, not GPU/runtime/data/training/robot/SSH/B200. Result: `professor_physical_reaction_evidence=READY_PROFESSOR_EVIDENCE_ONLY`, event-label metadata remains `READY_LOCAL_ONLY`, and RL contact-gated positive-control remains `RUN_FAILED`; reason is explicitly `rl_contact_gated_positive_control_blocks_dataset_rl_not_professor_physical_evidence`. Direct-IK posthoc audit reclassifies the direct run as physical reaction evidence PASS (`max_disp_along_m=0.000922590`, `max_speed_mps=0.008078601`, overshoot `0.0`) while still failing contact-gated tap. Therefore current allowed next step is professor-facing evidence/package or local RL blocker debug only; do not claim clean action teacher, dataset, PPO/RL, large dataset, or RoArm readiness.

Latest correction: for the professor 10cm/0.72kg push/tap branch, final 1cm
displacement is no longer the primary objective if the real task only needs a
tap/reaction. Treat measured contact, transient displacement, cube lift/z delta,
and cube speed as the primary reaction-event evidence; keep final displacement as
a secondary relocation metric. The reaction audit script confirms seed938 FAILs
because contact evidence is `0.0`, while seed939 and seed940 PASS reaction gate.
seed940 has `max_disp_mean_m=0.010990217`, transient gate `1.0`, final gate
`0.0`, no overshoot, and `teacher_quality_ready=false`. The approved seed941
randomized 16-env screen did not pass the reaction gate: reaction evidence was
`1.0`, but contact evidence was only `0.625`; no-posewrite and no-overshoot were
OK, and teacher quality still failed due TCP error/clipping. The approved cap050
seed942 diagnostic also failed: contact evidence dropped to `0.5`, transient gate
to `0.1875`, final gate to `0.0`, and teacher quality stayed false. The fixed-y+
seed943 diagnostic confirmed y+ is a weak direction bucket: reaction `0.9375`,
contact evidence `0.375`, no overshoot, and teacher quality false. The local
posthoc y+ geometry/reach audit now shows this is not merely reaction-rate
noise: contact rows average max displacement `0.010986278m`, no-contact rows
only `0.000069159m`, and the workspace is asymmetric (`cube_y0_m<=0` contact
`0.625` versus `cube_y0_m>0` contact `0.125`; `cube_x0_m<0.25` contact
`0.111111` versus `cube_x0_m>=0.25` contact `0.714286`). Traced envs still hold
large final vertical TCP-target error (`0.043-0.070m`) at side-center targets,
so next work remains y+ target path/reach/lateral-height/actuator tracking, not
RL/data scale-up. A follow-up trace-path/actuator audit shows target world-y
motion is about `0.020000m` and final target z stays at the start-cube
side-center height, while final TCP error is mostly z error and both contact and
no-contact traced groups keep `clip_any=1.0`. The seed944 height050 y+ screen
then rejected the height-only shortcut: contact evidence `0.0`, reaction
`0.6875`, final TCP error `0.022889409m`, clip `0.949198734`, no posewrite, and
no overshoot. The seed945 good-workspace y+ screen (`fixed_cube_x_m=0.295`,
`fixed_cube_y_m=-0.044`) restored reaction/contact to `1.0/1.0` with no
overshoot/posewrite, but teacher quality remains false because final TCP error
is `0.065514732m` and DiffIK clip is `1.0`. The default-preserving
`--base_lateral_offset_m` patch let seed946 test the lateral error directly; at
`-0.020m`, good-workspace y+ reached reaction/contact `1.0/1.0`, final 1cm gate
`1.0`, max displacement `0.011251196m`, no overshoot/posewrite, but teacher
quality still false from final TCP error `0.062820967m` and clip `1.0`.
To remove the confusing legacy filename issue, future professor 10cm DiffIK
commands should prefer `sim_scripts/cube10cm_push_diffik_probe.py`. That wrapper
does not duplicate physics logic; it injects 10cm/0.72kg tap/reaction defaults
into the shared DiffIK probe and lets explicit command-line overrides win. Its
default displacement/stop/gate values are `0.001m`, not a fixed final 1cm
relocation objective. After explicit approval, one tiny local stiff600 screen
changed only `--arm_stiffness_override 600` inside the seed946 geometry. It kept
tap/reaction PASS (`reaction/contact=1.0/1.0`, no posewrite, no overshoot), but
it is not an improvement over seed946: final relocation secondary is false, max
displacement fell to `0.004667487m`, final displacement to `0.003845483m`, clip
stayed `1.0`, and teacher quality stayed false despite TCP error improving to
`0.046062952m`. Do not treat lower TCP error alone as progress. A follow-up
approved direction-generalization screen seed947 kept seed946 goodxy
(`x=0.295,y=-0.044`) and lateral `-0.020m` but released `fixed_push_dir`; it
FAILed reaction gate because contact evidence was only `0.5625`, with no
posewrite and no overshoot. Direction split shows y+ is the only controlled
direction (`controlled=1.0`, but low-motion `0.75`), x- had contact `0/7`, and
x+/y- had contact but controlled `0`. This is strong evidence seed946 is a
y+-specific/contact-geometry pocket, not a direction-general teacher.

This is the rolling current-state dashboard. It is not full history. Durable
rules live in `claudedocs/DECISIONS.md`; experiment history lives in
`claudedocs/EXPERIMENT_LEDGER.md`; detailed logs live in `claudedocs/session_*.md`.

Do **not** use `HANDOFF.md` or `TASKS.md` as current state.

## B200 Retired / Backup Truth

- NHN/Sogang B200 access expired on 2026-05-22 at 23:59 KST, and the user
  reported B200 now shows disconnect on 2026-05-23 KST. Future research must
  not depend on entering B200 through SSH or on B200-only paths.
- Do not copy, request, or depend on `.ssh` private material. We preserved
  research artifacts, logs, code snapshots, checkpoints, env specs, and wandb
  cache; not login secrets.
- Track A B200 evidence is locally preserved and verified: B200 `/tmp/p7_branch_b_*`
  ↔ `b200_backup_20260522_final/tmp_p7` has 494 files, path+size hash
  `c308d1a682560cf51136cdd1a018c50ce2e7b488f1a0d4620e31abf7de80cfd4`,
  and file-content aggregate hash
  `cca0586b77c36ee79532d0640f9a35b2f1056654ab2758f256ea2bc1f149a4ae`.
- Track A B200 `sim_scripts` snapshot is locally preserved and verified:
  53 non-pycache files, path+size hash
  `98563bbc3d27426351abd13272a88537009372b2c709b46d2a5021560c5ea23a`,
  file-content aggregate hash
  `fefe4c873c1e45ec4cb95226a2c1a0d53860e4eca926c93d3da1b9887c9ca83f`.
- Track B B200 outputs are locally preserved, but split across
  `b200_backup_20260522_final/outputs`, `b200_backup_20260521`, and
  `openvla_oft_b200_pulls`. Do not assume
  `b200_backup_20260522_final/outputs/openvla_oft_v6_b200` is complete; the
  complete OpenVLA full checkpoints live in `openvla_oft_b200_pulls`.
- Full verification details:
  `claudedocs/session_20260522_b200_retirement_track_a_b_backup_verified.md`
  and `b200_backup_20260522_final/README_BACKUP.md`.

## Current Truth

**Track A active line**: P7/Branch B normalized 3cm cube grasp primitive.

- **Urgent professor push/tap branch (2026-05-26)** is separate from Track A
  grasp. Added `sim_scripts/cube3cm_push_rollout_probe.py` md5
  `8d329b79106e7ca2c03fa91b7ac87170` and ran local IsaacLab 3cm cube push/tap
  scripted rollouts at 16, 1024, 5120, and 20480 trials. The 20480 run lives in
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/`.
  Runtime stdout md5 `2aad344f08f95c880e43bc0d7f655998`; summary md5
  `5c9278450b5531afb7b0ca2a1fed46ee`; per-env CSV md5
  `4c2864301bea8e2ae798a8f77adf23ab`; audit md5
  `3e0096ba54e7cc0ec0e55b1b26a50b8e`. Runtime line 20 confirms no grasp,
  no attach/object posewrite, no training, no dataset; line 21 confirms 6D
  normalized joint-delta action semantics. Line 42 reports
  `total_trials=20480`, `ik_ok_rate=1.0000`, `disp_xy_mean_m=0.031809`,
  `disp_xy_p95_m=0.089702`, `moved_5mm_rate=0.8774`,
  `push_positive_1mm_rate=0.9086`, zero action saturation, zero grasp/attach/
  posewrite. Audit lines 5-21 show important caveats: low-motion rate
  `0.073340`, direction asymmetry, and high-speed/outlier impacts up to
  `0.521036748m` displacement and `4.549609073m/s`.
- **Professor cube-push learned-policy follow-up (2026-05-26)**: added a separate
  no-attach cube-push RL env/entry/eval path, not Track A grasp. Current
  `roarm_rl/roarm_cube_push_env.py` md5 `34254ac2fc3ede7a7844bd434fc9781d`;
  `train_cube_push_ppo.py` md5 `6dd733710ae3ec1e69cf8ad9e944948b`;
  `eval_cube_push_policy.py` md5 `bec3214d391862e4196d221775f4477d`.
  Static audit line 37 PASS and eval override check line 22 PASS. V4
  action-smoothed/velocity-limited policy improved frozen 1k impact from the
  speed-guard baseline but still had impact `0.245576787`
  (`ppo_smooth_limit_model49_eval1024_audit.out:3-17`). V5 scripted-teacher
  warm-start improved training logs, but teacher-off frozen 1k eval regressed to
  impact `0.257686676` and clean success `0.095168375`
  (`ppo_teacher_warmstart_model49_eval1024_audit.out:3-17`). V6 policy-only
  contact-speed curriculum reduced frozen 1k impact to `0.153782895`, but clean
  success was only `0.110197368`, so verdict remains
  `CONTACT_SPEED_MODEL49_EVAL_IMPROVED_BUT_NO_10K`
  (`ppo_contact_speed_model49_eval1024_audit.out:3-18`). Teacher-on diagnostic
  was also weak/unsafe: impact `0.162448980`, clean success `0.067755102`,
  verdict `TEACHER_ON_DIAGNOSTIC_UNSAFE_OR_WEAK_NOT_LEARNED_NO_10K`
  (`ppo_contact_speed_teacher_on_eval1024_audit.out:3-18`). Therefore do **not**
  run learned-policy 10k/100k scaling yet. The professor's "known endpoint
  means use IK near the cube first" request should now be pursued more
  literally with an **IsaacLab built-in DifferentialIKController** cube-push
  probe: provide end-effector/TCP targets near the cube, let IsaacLab compute
  joint targets from the live Jacobian, then audit physics push/tap results.
  The existing RoArm `ik_dls` path is valid IK evidence but is not the same as
  IsaacLab built-in Differential IK. Do this before any new learned-policy
  10k/100k scaling or Track A runtime.
- **Professor cube-push IsaacLab Differential IK follow-up (2026-05-27)**:
  added `sim_scripts/cube3cm_push_diffik_probe.py` md5
  `cbb2176a80ed2a2c55552d0d98bc9ab9`,
  `cube3cm_push_diffik_audit.py` md5
  `5ed85775e31f805f4d43885a1de80246`, and
  `cube3cm_push_diffik_posthoc.py` md5
  `6bfc8ea3eac942d0af4c8fc852738f0e`. The short 16-env smoke was a useful
  negative: mechanism PASS but low-motion `1.000000000` and final TCP error
  `0.161282191m` (`diffik_probe_smoke16_seed777_audit.out:1-6`). With longer
  reach/horizon, 16-env reached controlled `0.937500000`, low-motion
  `0.062500000`, impact `0` (`diffik_probe_reach16_seed778_audit.out:1-6`).
  Frozen 1024 eval then ran headless with IsaacLab built-in
  `DifferentialIKController`, no RoArm-local IK control loop, no grasp, no
  attach/object posewrite, no training, and no dataset. Audit lines 1-6 report
  mechanism PASS, controlled `0.892578125`, impact `0.023437500`,
  low-motion `0.136718750`, `disp_xy_mean_m=0.034856980`, max speed
  `1.931515932m/s`, and final TCP error mean `0.028779610m`.
  Posthoc line 7 identifies weak direction `(1, 0)` with controlled
  `0.633333333`; line 8 identifies worst initial-position grid `(1, 1)` by
  low+impact. This is an IsaacLab scripted Differential IK physics result, not
  PPO/VLA learning and not Track A grasp success.
- **Professor cube-push Differential IK trajectory v2 (2026-05-27)**: updated
  `sim_scripts/cube3cm_push_diffik_probe.py` with default-preserving
  `--trajectory_variant v2` for the weak `(1, 0)` direction: closer precontact,
  lower TCP target height, shorter push-through, longer approach/push horizon,
  and smaller per-step DiffIK joint cap. Smoke16 seed780 exited 0 and audit
  PASS, but had `v2_posx_env_count=0`, so it was only a mechanism smoke.
  Reach16 seed779 included 6 `(1,0)` envs and audit lines 1-6 PASS with
  controlled `1.000000000`, impact `0`, low-motion `0.062500000`. Frozen 1024
  seed779 then exited 0 and audit lines 1-6 PASS with controlled
  `0.932617188`, impact `0.038085938`, low-motion `0.051757812`, success marker
  `0.580078125`, final TCP error `0.024324538`, and clip rate `0.666682201`.
  Compare-to-v1 lines 2-4 show the mixed result: overall controlled/low/final
  TCP improved, `(1,0)` controlled improved `0.633333333 -> 0.785185185` and
  low-motion improved `0.274074074 -> 0.085185185`, but `(1,0)` impact worsened
  `0.088888889 -> 0.144444444` and success marker fell. Grid `(1,1)` improved
  low-motion `0.304687500 -> 0.023437500`, but gained nonzero impact
  `0 -> 0.031250000`. Verdict: useful scripted Differential IK physics
  evidence, not learned policy, not Track A grasp, not dataset/teacher-ready.
- **Professor cube-push Differential IK trajectory v3 (2026-05-27)**: current
  `sim_scripts/cube3cm_push_diffik_probe.py` md5
  `dc6ca5a222f0bd9437d5f83bf5449729` adds default-preserving
  `--trajectory_variant v3` for `(1,0)`: lower contact height, shorter
  push-through, longer/slower pos-x horizon, and lower pos-x joint-step cap.
  Smoke16 seed780 audit lines 1-6 PASS but had `v3_posx_env_count=0`, so it was
  mechanism-only. Reach16 seed779 included 6 `(1,0)` envs and audit lines 1-6
  PASS with controlled `1.000000000`, impact `0`, low-motion `0.062500000`.
  Frozen 1024 seed779 audit lines 1-6 PASS with controlled `0.969726562`,
  impact `0.004882812`, low-motion `0.035156250`, success marker
  `0.604492188`, final TCP error `0.023551417`, and zero posewrite. Posthoc
  line 6 reports `(1,0)` controlled `0.929629630`, impact `0.014814815`,
  low-motion `0.088888889`. Compare lines 2-3 show v3 improves over v2 on
  overall impact `0.038085938 -> 0.004882812` and `(1,0)` impact
  `0.144444444 -> 0.014814815`; line 9 shows remaining `(1,0)` impacts are
  still tip-angle outliers only. Critical caveat: `(1,0)` success marker dropped
  to `0.314814815` and clip is `1.000000000`, so v3 is a strong scripted
  physics/statistics candidate but not automatically a dataset teacher.
- **Professor cube-push Differential IK v3 10,240-trial audit (2026-05-27)**:
  ran local IsaacLab `num_envs=1024`, `episodes=10`, seed779, total 10,240
  scripted DiffIK trials. Stdout lines 20-21 confirm built-in
  `DifferentialIKController`, no RoArm-local IK loop, no training/dataset/grasp/
  posewrite, `trajectory_variant=v3`. Audit lines 1-6 PASS with
  `csv_rows=10240`, controlled `0.943164062`, impact `0.007519531`,
  low-motion `0.042480469`, success marker `0.594824219`, final TCP error
  `0.023529604`, and zero posewrite. Posthoc line 6: `(1,0)` n=2566,
  controlled `0.874512860`, impact `0.012860483`, low-motion `0.122759158`,
  success marker `0.296570538`. Compare-to-1024 lines 2-3: overall impact stays
  below 1%, `(1,0)` impact stays below 2%, but `(1,0)` low-motion worsens and
  success remains weak. Verdict: professor-style 10k scripted push/tap robustness
  evidence PASS; still not learned policy, not Track A grasp, not dataset-ready.
- **Professor cube-push Differential IK v3 visualization replay (2026-05-27)**:
  generated a professor-facing MP4 from a captured v3 trace; this is a replay
  artifact for visualization, not a new training run, dataset generation, or
  fresh physics recomputation. Current `cube3cm_push_diffik_probe.py` md5 is
  `1e39836eb02a22c12e084a4279e6b4e7`; replay renderer
  `sim_scripts/cube3cm_push_diffik_render_trace.py` md5 is
  `2adb116ae2c441420873a8384d3a7b17`. Selected single-env case is
  `diffik_probe_v3_reach16_seed779.csv` line 5: env_id `3`, direction `(1,0)`,
  start `(x,y)=(0.353590250,-0.073313951)m`, displacement
  `0.036002159m`, controlled `1`, impact `0`, low-motion `0`, success marker
  `1`. Trace summary lines 46-49 show trace CSV, env_id `3`, 145 trace frames,
  and `training=false`. Render stdout line 447 confirms `frames=145`, MP4 path,
  `training=NO`, `dataset_generation=NO`, `physics_recomputed=NO`. Render
  summary lines 19-27 confirm `30fps`, `1280x720`, 145 written frames, output
  path, `physics_recomputed=false`, `training=false`. A corrected four-direction
  parallel replay now uses env_id `[0,3,4,7]` for directions `(0,-1)`, `(1,0)`,
  `(0,1)`, `(-1,0)` respectively; render stdout line 447 confirms
  `frames=145 env_count=4`, and render summary lines 12-19/91-112 confirm white
  background, black actual RoArm URDF STL mesh, gray table, pink cube, `30fps`,
  145 frames, 2x2 layout, `physics_recomputed=false`, and `training=false`.
  Earlier black FK-proxy render artifacts were rejected/superseded because they
  were not actual RoArm geometry.
- **Professor cube-push DiffIK dataset v2 + BC learned rollout (2026-05-28)**:
  added all-env trace capture and auditable dataset/BC tooling. Current md5s:
  `cube3cm_push_diffik_probe.py` `2342f31701e91af57d0f311db4eeec87`,
  dataset builder `a0d18ef1b34415c96d036ba42952e37e`, dataset audit
  `f677ea14809a0a2091bc13c5254d4fae`, BC train
  `df03abb00188cfd9b644b0ef410a0e14`, BC rollout
  `a56e9a96feaad196fef6e1081c0116ec`, rollout audit
  `121721561bcde8df141f56207dcab14d`. Source v3.1 1024 trace seed779 audit
  lines 1-6: controlled `0.964843750`, impact `0.004882812`, low-motion
  `0.034179688`, success `0.611328125`; posthoc line 6 still marks `(1,0)`
  weak. Dataset build lines 1-6 selected 320 final-success teacher trajectories
  into 46,400 rows, balanced 80 per direction, split `224/48/48`, candidate YES.
  Dataset audit lines 1-7 PASS full state-action dataset with schema/finite/split/
  direction/mechanism OK and final controlled/success `1.0`, impact/low `0.0`.
  BC train lines 1-4 PASS checkpoint with test MSE `0.007494668` and mean test
  MAE `0.000745819rad`. Learned BC rollout 1024 seed883 audit lines 1-6 PASS
	  without DiffIK: controlled `0.945312500`, impact `0.012695312`, low-motion
	  `0.026367188`, success `0.648437500`, posewrite 0. Caveat remains: `(1,0)`
	  line 5 success `0.453488372`, impact `0.038759690`. This is professor-branch
	  teacher-filtered state-action BC, not Track A grasp, not PPO/RL/VLA, not image
	  dataset, and not 10k/100k learned-policy robustness.
- **Professor cube-push v3.2 teacher sweep + bucket-balanced BC v2 (2026-05-28)**:
  added `cube3cm_push_diffik_bucket_audit.py` and extended the dataset builder/audit
  for `direction_posx_bucket` selection. v3.2 teacher parameter sweeps were useful
  negatives: t270/p036 raised `(1,0)` success but also raised `(1,0)` impact, while
  t270/p030 and t257/p034 were not robust across seed790/seed791. Dataset v3 was
  built without new physics from the v3.1 all-env trace: build lines 1-6 selected
  180 final-success teacher trajectories, 26,100 rows, 45 per direction, and
  `(1,0)` low/mid/high-x `15/15/15`. Dataset audit lines 1-8 PASS, including
  `balance_mode=direction_posx_bucket`, bucket OK, split bucket OK. BC v2_bucket
  train lines 1-4 PASS with test MSE `0.022629632`, mean test MAE
  `0.001227073rad`. 1024 learned rollout seed883 with default clip PASS overall:
  controlled `0.961914062`, impact `0.015625000`, low-motion `0.016601562`,
  success `0.679687500`; `(1,0)` success improved to `0.527131783`, but bucket
  audit fails because low_x impact is `0.068493151` and high_x impact is
  `0.061855670`. Clip `0.035` improved overall success to `0.689453125` but still
  failed low_x impact. Verdict: learned BC quality improved, but per-bucket safety
  gate blocks PPO/RL scale-up.
- **Professor cube-push safety-aware BC v3 gate (2026-05-28)**: extended
  `cube3cm_push_diffik_train_bc.py` with optional safety-weighted BC loss and
  `cube3cm_push_bc_policy_rollout.py` with auditable per-bucket action scaling /
  smoothing; fixed bucket-audit learned-policy reporting. Current md5s:
  train `cbe7c2e8d44fe7a92cb2ba69f29b518a`, rollout
  `aa0b5ef06db903058724a71f61225f0b`, bucket audit
  `c69ff72a4a31228868169016ab2f2d08`. Safety BC train lines 1-5 PASS with test
  MSE `0.018879525`, mean MAE `0.001225158rad`; checkpoint md5
  `03b159809ddca64aad6d6449b7f44876`. 1024 frozen learned rollout seed883
  lines 1-6 PASS overall: controlled `0.953125000`, impact `0.004882812`,
  low-motion `0.030273438`, success `0.662109375`; bucket lines 7-10 PASS with
  low_x impact `0.041095890`, high_x impact `0`. Cross-seed seed884 lines 1-6
  PASS overall: controlled `0.943359375`, impact `0.010742188`, low-motion
  `0.024414062`, success `0.662109375`; bucket lines 7-10 PASS with low_x impact
  `0.035714286`, high_x impact `0`. Critical caveat: low_x low-motion is still
  high (`0.315068493` seed883, `0.261904762` seed884), so this unblocks only the
  next small safety-aware learned-policy gate, not 10k/100k robustness or PPO
  scale-up without explicit approval.
- **Professor cube-push safety-aware PPO warm-start smoke (2026-05-28)**: after
  explicit approval, added default-off BC teacher/imitation warm-start support to
  `roarm_rl/roarm_cube_push_env.py` md5 `9806c1fcfb4666355f825418da5b7d75`,
  `train_cube_push_ppo.py` md5 `5466a9c9d40a7f09d397fbffa7cdb878`,
  `eval_cube_push_policy.py` md5 `fa68ee654c969aff7938867894acf125`, optional
  PPO bucket audit support md5 `62f74ce38c9a44f0f0790e00559f634a`, and new
  `cube3cm_push_ppo_rollout_audit.py` md5 `b92260c8f0986c1b6bfe233fcf417d01`.
  Prior docs say system Python failed with missing `gymnasium`, but current local
  stderr citation is a mismatch; verified valid runtime was `conda run -n isaaclab`
  on `cuda:0`. Smoke12 training seed885 completed 73,728 timesteps and wrote
  `model_11.pt` md5 `c9f945a4d1eacd817d4733e7d9b7e48e`; training stdout lines
  47-78 confirm BC teacher blend `0.35`, imitation reward `0.30`, no attach/no
  dataset, and `cuda:0`. Teacher-off frozen 1024 eval seed886 is a mechanism PASS
  but performance FAIL: audit lines 1-5 show controlled `0.470703125`, impact
  `0.087890625`, low-motion `0.344726562`, success `0.078125000`; bucket lines
  1-10 fail with `(-1,0)` impact `0.230483271` and `(1,0)` success
  `0.070833333`. Teacher-on short/safety-limited diagnostic is also weak
  (success `0.050781250`, impact `0.097656250`). Direct-like teacher-on diagnostic
  with 6s horizon/home reset/relaxed action loop partially recovers controlled
  `0.792968750`, impact `0.054687500`, success `0.417968750` and passes the loose
	  posx bucket screen, proving the checkpoint is not dead but the PPO action-loop
	  warm-start bridge/curriculum is mismatched. Do not call PPO `model_11.pt` a
	  successful learned policy; do not run PPO 10k/100k scale-up from it.
- **Professor cube-push BC teacher bridge redesign diagnostics (2026-05-29)**:
  added default-preserving bridge controls: `joint_delta_reference` and
  `bc_teacher_phase_timing`. Current md5s: env
  `a0483108ef0fc8ab2f27a58b6edd8c13`, train
  `7032616ded5617b546149227f4c0d110`, eval
  `b10fad43cfd3b0ca543390ad6011135f`. Static checks passed. Teacher-on
  direct-step/joint-pos 128-env seed889 recovered overall but failed low_x bucket:
  controlled `0.984375000`, impact `0.007812500`, success `0.601562500`, low_x
  success `0.133333333`. Low_x scale `1.0` seed890 passed the small teacher-on
  bridge screen: controlled `0.992187500`, impact `0.007812500`, low-motion
  `0.007812500`, success `0.765625000`, and bucket PASS with low_x success
  `0.538461538`. But this is teacher-on only. Existing `model_11.pt` under the
  new action loop is teacher-off zero-motion at 128 (controlled `0`, low-motion
  `1`, success `0`). A tiny 128-env smoke8 PPO distillation run wrote `model_7.pt`
  md5 `5ed5ac34dc624ac8c660d9176378b357`, but imitation MSE stayed around
  `0.56-0.59` and teacher-off 128 still failed with controlled `0`, low-motion
  `1`, success `0`. Verdict: bridge mismatch is partly fixed, but PPO actor
  learning is still not solved. Do **not** run teacher-off 1024, PPO scale-up,
  10k/100k, dataset generation, or Track A runtime from these PPO checkpoints.
  Next valid step is a true supervised actor/normalized-action distillation or
  stronger actor initialization before PPO.
- **Professor cube-push rsl_rl actor distillation gate (2026-05-29)**: added
  `roarm_rl/distill_cube_push_actor.py`, which collects the BC teacher's
  normalized joint-delta actions through the same direct-step/joint-pos loop and
  writes a normal rsl_rl checkpoint. Distillation seed894 used 128 envs x 600
  steps = 76,800 samples from `model_7.pt`; checkpoint
  `model_actor_distill.pt` md5 `57811cfb054ca7ac39b134d1d97cd543`.
  Supervised fit improved val MSE `0.169735238 -> 0.000794161`, but the only
  allowed teacher-off 128 audit seed895 still failed: controlled `0.101562500`,
  impact `0`, low-motion `0.929687500`, success `0.031250000`; bucket audit
  failed with low_x/mid_x success `0` and high_x success `0.076923077`. Verdict:
  actor no longer outputs pure zero, but closed-loop push is still effectively
  low-motion. Do **not** run teacher-off 1024, PPO scale-up, 10k/100k learned
  robustness, dataset generation, or Track A runtime from this checkpoint.
  Next valid step is a closed-loop/action-target analysis of why low-MSE one-step
  normalized action imitation collapses to low-motion, or a stronger rollout-
  aware actor initialization; then repeat only teacher-off 128.
- **Professor cube-push waypoint actor 128 gate (2026-05-29)**: trace showed
  the first actor-distilled checkpoint failed because actor-visited states diverged
  from the teacher and because the actor observation did not expose the teacher's
  phase/moving TCP waypoint. Added default-off `policy_obs_target_mode` with
  `bc_teacher_tcp_target`, plus trace/analysis tools. `model_actor_waypoint.pt`
  improved teacher-off 128 seed901 to controlled `0.679687500`, impact `0`,
  low-motion `0.242187500`, success `0.273437500`, but bucket failed low_x.
  Waypoint + on-policy DAgger1 improved seed904 to controlled `0.968750000`,
  impact `0`, success `0.617187500`, but bucket still failed low_x success
  `0.117647059`. The low_x-scale `1.3` on-policy checkpoint
  `model_actor_waypoint_lowx130.pt` md5 `606d19fff713e7468d395af4a027d08a`
  passed the first teacher-off 128 first-episode gate seed906: audit controlled
  `0.937500000`, impact `0`, low-motion `0.093750000`, success `0.546875000`;
  bucket audit PASS with `(1,0)` low_x success `0.571428571`, mid_x success
  `0`, high_x success `0.428571429`, and zero impact. After explicit approval,
  the same checkpoint passed three teacher-off 1024 first-episode overall/
  per-bucket gates with no teacher action blend: seed907 controlled
  `0.924804688`, impact `0`, low-motion `0.109375000`, success `0.511718750`;
  seed908 controlled `0.925781250`, impact `0`, low-motion `0.114257812`,
  success `0.523437500`; seed909 controlled `0.920898438`, impact `0`,
  low-motion `0.125976562`, success `0.506835938`. This is now a 3x1024
  teacher-off learned-policy gate PASS using waypoint observations, but still
  not 10k/100k robustness, not dataset readiness, and not Track A/PPO/VLA
  final success. Follow-up candidate screen showed high_x actor scale `1.0`
  fails 128 bucket, gain `0.045` is a 3x1024 PASS non-canonical deployment
  candidate with only modest/mixed improvement, and gain `0.050` is mixed
  after a seed907 pilot; canonical remains `model_actor_waypoint_lowx130.pt`
  with gain `0.040`. After explicit approval, a single-stage 10240-env audit
  failed during IsaacLab env creation before policy rollout, so it is not a
  policy failure. The fallback 10x1024 sharded first-episode teacher-off audit
  on seeds 912-921 produced 10,240 rows and PASSed mechanism and posx bucket
  gates: controlled `0.927148437`, impact `0.000097656`, low-motion
	  `0.106054687`, success `0.524902344`; low_x success `0.406947891`, mid_x
	  success `0.183497537`, high_x success `0.213625866`. This supports a
	  sharded 10k teacher-off robust learned-policy gate PASS, but still not
	  dataset readiness, not Track A evidence, and not PPO/RL/VLA final success.
- **Professor cube-push metric reframe and target-extension probe (2026-06-02)**:
  code/log review shows the old `success_marker` is a strict task marker, not the
  only professor-relevant push/tap metric. In code, the cube is fixed at
  `CUBE_SIZE_M=0.030` and mass `0.020kg`; env success requires controlled push,
  no impact, `disp_along >= 0.030m`, target-distance tolerance, and speed cap.
  The sharded 10k threshold analysis shows the canonical actor is much stronger
  at stable smaller pushes than the 3cm marker alone suggests: for direction
  `(1,0)`, `disp_ge_5mm=0.906199678`, `disp_ge_10mm=0.842592593`,
  `disp_ge_20mm=0.770531401`, but `disp_ge_30mm=0.266505636`. Posx buckets show
  mid/high are near-perfect at 10mm and strong at 20mm, then fall sharply at
  30mm. Therefore report professor push/tap evidence as `1/5/10/20/30mm`,
  `disp/object_size`, controlled, no-impact, and low-motion, not as 3cm success
  alone. Added default-preserving code knobs for weighted actor distillation and
  mid/high BC teacher push-through overrides. The weighted mid/high actor
  candidate passed one-step fit but failed the 128 posx bucket screen
  (`low_x=0.083333333`, `mid_x=0.090909091`, `high_x=0.833333333`) and is
  rejected. The target-extension probe on the canonical checkpoint improved
  same-seed mid_x/high_x (`mid_x=0.272727273`, `high_x=1.000000000`) but still
  failed due low_x (`0.166666667`), so do not scale it. Before changing cube
  size, define whether `10*10*10` means mm or cm and whether mass is measured,
  density-preserving, or deliberately fixed as a diagnostic. A 2026-06-04
  local-only update to `sim_scripts/cube3cm_push_diffik_bucket_audit.py` now emits
  the hierarchical report directly. The generated sharded-10k report logs
  `cube_size_m=0.030000`, `cube_mass_kg=0.020000`, density `740.741kg/m^3`,
  no-impact, `disp/object_size`, and displacement-only 1/5/10/20/30mm columns
  in
  `model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_hierarchical_bucket.out`.
  Overall `disp/object_size_mean=0.775020338`; forward `(1,0)` reports
  `disp/object_size_mean=0.743007598`, 5mm `0.906199678`, 10mm `0.842592593`,
  20mm `0.770531401`, and 30mm `0.266505636`.
- **Professor cube10cm DiffIK gate prep (2026-06-04)**: the next professor
  request is interpreted as a separate 10cm/0.72kg object push/tap diagnostic,
  not as Track A and not as a requirement to push a full 10cm object length.
  The 1cm gate is now explicit. The 128 v1 gate, fixed16 side-center gate, and
  settled-start fixed16 rerun all failed with `diffik_clip_rate_mean=1.0` and
  low/no motion. The settled-start patch fixed the reset-buffer z mismatch but
  not the rollout.
  A follow-up 4-env trace diagnostic added per-step link5/TCP targets, raw and
  clipped deltas, robot targets, and actuator follow errors. cap `0.120` and
  drive-boost controls prove the 10cm/0.72kg object can move, but they overshoot
  by 5-9cm and are not teacher-ready. Code review also found the old through
  target was far-face/cross-object (`cube + half + push_through`), so
  `--through_target_mode near_face` was added. Near-face default/long-approach
  runs still failed to contact (`disp_ge_gate_rate=0.0`, final TCP error about
  `0.061m`, min TCP-cube about `0.083m`), while near-face drive boost passed the
  gate but moved about `0.050m`. Next valid work is one tiny controlled near-face
  contact controller with actuator/step scheduling and displacement/contact stop;
  not randomization, 128/1024/10k, dataset generation, PPO/RL, VLA, or Track A.
- Track A goal: first make the sim/Isaac Lab contact primitive reliable, then
  move toward broad sim/lab dataset collection and learning.
- Dataset generation and training are blocked until close_26 proxy audit PASS,
  then hold-lift PASS, then small pilot dataset/replay PASS.
- v4/v5/v6/v7 rigid/offset variants, soft-contact material-only, virtual
  compression+damping, target-guarded v1 through v7, and the first pre-fix v8
  runtime have all failed close_26 posthoc audit. None is grasp success.
- This is an Isaac proxy/contact primitive failure, not proof that the real robot
  cannot grasp the foam cube.
- The RL-to-expert-to-rollout-to-demo pipeline is valid only after a Track A
  Stage 0 no-attach contact gate exists. Existing default Pick/Stack PPO envs use
  kinematic attach / posewrite and must not be used as Track A no-attach expert
  evidence.
- Latest Track A B200 v6 `close_26` runtime is FAIL, not grasp success.
- 2026-05-26 step-plan update: the professor-style
  RL→expert→rollout→dataset→learning pipeline is still the right high-level
  path, but only after Stage 0 no-attach contact primitive PASS. Added local
  static design artifact
  `sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py`
  md5 `14a462526945f3c5bca1c5e8c3e13525`; it reports that v6 pre-freeze recovery
  rows increased target error by `0.000626m` and support gap by `0.000319m`.
- 2026-05-26 v7 active-recovery code/readiness update: added default-off v7
  finite-difference TCP recovery, matching audit support, and readiness negative
  controls. Local static checks pass, including synthetic v7 PASS, v7 no-active
  recovery rejection, archived v6 log rejection as v7, and readiness
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`. No runtime was run.
- 2026-05-26 approved local v7 runtime attempt was blocked by infrastructure, not
  physics: IsaacLab metadata emitted, but local CUDA/NVIDIA access failed before
  environment creation and no close step/aggregate lines were produced. The
  immediate audit correctly failed. Preserved logs live in
  `claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/`.
- 2026-05-26 RunPod/Codex continuation setup: Claude had RunPod MCP configured,
  but Codex did not. Added `[mcp_servers.runpod]` to
  `/home/cgxr/.codex/config.toml` from Claude's RunPod MCP config, with the
  `RUNPOD_API_KEY` value not printed. Backup:
  `/home/cgxr/.codex/config.toml.bak_runpod_20260526` md5
  `1ef4acf6f1c92a64b9bbd79a2e35b7e7`. Same-session `tool_search` still did not
  expose `mcp__runpod__...`, so each new Codex session must verify loaded tools
  before using RunPod MCP. A later Codex session did expose `mcp__runpod__...`
  and `list_pods` returned no GPU pods.
- 2026-05-26 post-reboot local CUDA update: user rebooted the local Ubuntu PC.
  Boot time now `2026-05-26 14:08`; host NVIDIA kernel/userspace now match at
  `580.159.03`. Host `nvidia-smi` and `conda run -n isaaclab` CUDA checks pass
  only when run outside the default Codex sandbox. The default Codex sandbox
  hides `/dev/nvidia*`, so sandboxed `nvidia-smi` still fails; this is not a host
  CUDA failure. v7 readiness still reports
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`. The old `/tmp` RunPod overlay is
  gone after reboot; recreate it if RunPod is needed. Top local backup USD md5
  remains `4497024d25abab11de5c50e144124553`.
- 2026-05-26 post-reboot v7 runtime/audit result: exactly one local
  close_26-only v7 active-recovery runtime ran with escalated Codex execution and
  immediate audit. This is a real physics/audit FAIL, not a CUDA infrastructure
  block. Logs:
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/`.
  Runtime stdout md5 `621d00b9d157b4e70178c28f94ca4c7f`; audit stdout md5
  `406b96557d94418f16273e517ec4d69b`. Runtime lines 389-391 show v7 active
  recovery did trigger (3 writes, 0 IK failures, negative counter-gap deltas),
  but runtime line 392 is first support hard-freeze (`counter_gap=0.002048m >
  0.002m`, target `0.002962m` still inside gate), line 393 is first target+
  support breach (`target=0.003059m`, gap `0.002104m`), and line 424 aggregate
  has close_reached NO, 31 hard freezes, attach/posewrite 0, telemetry-only YES,
  success_claim NO. Audit line 19 fails close_reached; line 32 fails
  hard-freezes-zero; lines 54-56 fail hard-freeze/fixed-target/fixed-support;
  line 66 `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- 2026-05-26 v7 failure static analysis result: analyzer
  `sim_scripts/p7_branch_b_cube2cm_v7_failure_analyzer.py` md5
  `e13605f058cd1908ff3d863e8239fbc4`; output
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v7_failure_static_analysis.out`
  md5 `0fbf57f32473fa253ee1082b888bdcb1`. It parsed 45 close rows,
  found 3 v7 active rows, and classified audit mismatch NO, late trigger YES,
  candidate prediction mismatch YES, weak TCP follow YES, contact geometry
  suspect YES, and hard-safety lockout after active YES. Active followups
  predicted target/gap improvement but observed target and support gap worsening
  with negative TCP follow ratios.
- 2026-05-26 v8 observed-recovery static design result: static-only script
  `sim_scripts/p7_branch_b_cube2cm_target_guarded_v8_observed_recovery_static_design.py`
  md5 `56a382377b7fb0f0c6391bf59163af0d`; output
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_observed_recovery_static_design.out`
  md5 `c14e80ec5fc69c6e6e17925d61f81d0b`. Output lines 1-2 verify v7 runtime
  and v7 analysis md5s. Line 3 finds first projected reserve trigger at runtime
  line 386 / close step 9, line 4 shows v7 first active at runtime line 389 /
  step 12, and lines 5-7 show the projected reserve trigger was 3 steps before
  v7 active and 6 steps before first support breach. Lines 8-11 reject unchanged
  v7 by observed response/TCP follow; lines 21-28 define the v8 design contract
  and explicitly report `RUNTIME_READY=NO`, `STATIC_V8_DESIGN_DONE=YES`.
- 2026-05-26 pre-runtime v8 runtime-candidate static readiness result:
  default-off v8 candidate and matching audit/readiness support were implemented;
  at that point no v8 physics runtime had run. Runtime probe md5
  `7e6dfc35bbfeacb5d1689f2f175e5120`;
  audit md5 `8dbf621c983ec03f46e5d52843781fda`; readiness md5
  `a31ced20b754a4a42058349525d1a435`. Readiness output
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_runtime_candidate_readiness.out`
  md5 `6a2a62808451175b65e5d522b695b8b6`: lines 1-2 local/static/no forbidden
  mechanisms, lines 3-4 wiring/metadata guard PASS, lines 5-13 negative controls
  PASS, line 14 synthetic v8 PASS accepted, line 16 future command uses preserved
  local backup USD path, and line 19 `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
  A separate v8 audit of the preserved post-reboot v7 runtime
  `v8_rejects_post_reboot_v7_audit.out` md5
  `cb082918d92a0f95b585ade432c34730` correctly fails: lines 5/14/15 reject
  metadata, lines 20/30 reject close/hard-freeze success, and lines 53/55/57/58
  reject missing v8 reserve trigger, observed worsening, non-positive TCP follow,
  and missing counter-contact modeling.
- 2026-05-26 approved local v8 runtime/audit result: exactly one local
  close_26-only v8 observed-recovery runtime ran with escalated Codex
  GPU/Isaac execution and immediate audit. It FAILED; this is not grasp success.
  Logs:
  `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/`.
  Runtime stdout md5 `74095570c2d6a60abdf522c2413735db`; audit stdout md5
  `7cd38eddb1dc9c925b01948cbc5cb416`. Audit line 20 fails
  `close_reached`; lines 26/35-36 fail because virtual damping was inactive and
  wrote zero velocity updates; lines 45-49/53 show no recovery/trigger; line 60
  reports `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`. Static failure analysis
  output md5 `7e81773b91d39658a3ec5c6eaf878f0c`: first hard freeze is runtime
  line 384 / step 7, target/support breach is runtime line 392 / step 15, and
  v8 trigger/recovery were never seen.
- 2026-05-26 post-fail v8 code fix: pre-fix v8 inherited target-guarded close
  activation but missed virtual damping activation. Added v8 to the
  `virtual_damping_active` mechanism and added a readiness regression check.
  Runtime probe md5 is now `acae0ca2e85a522dd4ac8fb583cb8fb8`; readiness md5
  is now `dc2bdaa8d882f12b5cc901a677caccc0`. Post-fix readiness output
  `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/readiness_after_v8_damping_fix.out`
  md5 `b652520a81792bf12373ff742cdba6b5`, lines 5 and 20 report the new
  damping-inheritance check PASS and `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
  No second/post-fix runtime has run.

**Track B/OpenVLA** is separate. Do not use Track B training or eval status as
Track A contact success evidence. Latest Track B P3 result remains: best deploy
ckpt = step 7500; steps 10000+ are collapsed and must not be deployed.
Track B data/continuation assets are backed up locally; P5 real robot deploy is
still pending local reboot/CUDA verification and user approval for robot motion.

**Track B Cube Task Pivot (2026-05-26, user-confirmed)**: sponge → **cube 3×3×3cm
× 5개 → 3+2 pyramid stacking** (L1=3, L2=2) 신규 task. Camera = Azure Kinect 고정
v6 동일 viewpoint. Sponge HARD RULES #19/#20/#24 자동 SUPERSEDED (HARD RULE #18
사용자 명시 정정 우선). Track A 직접 비교 → **sim demo 증강으로 재포지셔닝**
(Track A close_26 PASS 후 cube stacking sim demos co-training).

Hyperparam 갱신 (P3 7500→10000 collapse 회피): per_gpu_batch=8 + grad_accum=4 →
**effective batch=32** (vanilla OpenVLA-OFT LoRA 최소 권장치, 우리 P2 effective
8은 1/4였음). LR `5e-4` → **`2.5e-4`** (½, linear scaling 보수적). grad_clip_norm
=1.0, warmup 1K step, cosine. RunPod **A100 80GB**, 30K step ~8h ~$13.

데이터 신규 수집: **250ep (200 cube stacking + 50 cube pick), 일 50ep × 5일**,
ep당 ~400fr → 80K frames (v6 6942fr 대비 11.5×, task horizon 10× 근거).

7-phase plan: P0 cube+gripper calib (0.5일) → P1 데이터 수집 (4일, mid γ-gate)
→ P2 LeRobot 변환 (0.5일) → P3 RunPod 학습 (1일) → P4 12-ckpt offline eval rank
(0.5일) → P5 real multi-position deploy (0.5일) → P6 Track A close_26 PASS 후 sim
demo co-train (별개 trace) → P7 비교 paper (1일). 상세:
`claudedocs/session_20260526_track_b_cube_task_pivot_plan.md`, ledger row 123.
v6 sponge ckpt 7500 deploy (P5 pending CUDA reboot)는 별개 보존, cube pivot과 무관.

**Track B Cube P0 EXECUTION (2026-05-26, P0.1+P0.2 done, blocked at HW disconnect)**:
스크립트 3개 신규 — `safety_p0_guards.py`(G1-G10+DryRunArm, move_joints speed>200
ValueError), `trajectory_p0_gripper_sweep.py`(Gauge+stall auto-stop+cube release),
`hw_p0_sanity.py`(P0.1+pose_ctrl smoke default-OFF). 전부 py_compile+dry-run OK.
**포트 serial 검증**: Leader=USB0/`7842…`, Follower=USB1/`ee7a…` (5/22 기록 일치, 재연결 시 by-id 재검증 필수).
**P0.1 PASS**: max_diff 1.93°≤3° / Kinect cube 224-resize ~6-11px (P1 전 가시성 flag).
**P0.2 anchor lock**: 30mm cube→moving-jaw state **~37.88°** 정지, hold Y, **grip cmd
target ~28** (cmd 0-5 금지=서보 stall, P1 분포 "0-5°" 폐기). pad 없음 시사.
**fixed-jaw URDF 확증**: gripper movable joint `link5_to_gripper_link` 1개뿐 → cube
center≠TCP 중심선 → P0.3/0.4/L-F는 lateral offset 반영 필수.
**세션말 두 팔 USB 동시 단선** (`/dev/ttyUSB*`+by-id 소실, lsusb CP210x 둘 다 없음,
Kinect 정상) → pose_ctrl smoke/P0.3부터 재연결 대기. 사용자: 팔 전원/USB허브 확인.
다음: 재연결→by-id 재검증→`hw_p0_sanity.py --pose-ctrl-smoke`(P0.3 IK 게이팅)→
P0.3 fixed-jaw 반영→P0.4 grasp z→P0.7 L-F 5/5→P1. 상세:
`claudedocs/session_20260526_track_b_cube_p0_execution.md`,
`~/.claude/.../memory/tech_cube_grasp_anchors.md`(P0.1/0.2 실측 반영).

**P0 plan 4-agent 교차검증 + 사용자 결정 5건 확정 (2026-05-26)**: 코드/데이터
변경 없음, 메모리 docs 4개만. P0 순서 = P0.0 전제(pad/mass/matte) → P0.1 HW
sanity(max_diff≤3° + cube 가시성 + v6 viewpoint remount sanity + pose_ctrl smoke)
→ P0.2 jaw sweep(**Gauge 방식**: arm HOME 고정+수동 cube+gripper cmd만, object-
agnostic curve) → P0.3 approach angle(scripted joint, FK z guard) → P0.4 grasp z
(**FK primary**, 후보 +8/+12/+15mm tipping 비교) → P0.5 stacking z(L-F 관찰만)
→ P0.6 pyramid jaw 간섭 → P0.7 gate(L-F single pick 5/5) → P0.8 lock-in. 결정
5건: ① cube 5개 보유 ② P0.2 Gauge(future task 재사용) ③ camera v6 유지(환경 동일
→재calib 불필요) ④ IK는 joint 직접+pose_ctrl 보조 ⑤ grasp z tipping 비교 + P0.5
경량화. 정정: cube top z=**+18mm world**(table -12.12+30), grasp z FK 실측
+9~15mm, wrist_pitch +14° shift. 다음: P0.0 hands-on → deploy-agent P0.1/P0.2
스크립트(safety_p0_guards.py G1-G10 + Gauge sweep). 상세:
`claudedocs/session_20260526_track_b_cube_p0_plan_crossvalidated.md`, ledger row
126, `~/.claude/.../memory/tech_cube_grasp_anchors.md`. sponge anchor SUPERSEDED.

**Track B P4 result — 2026-05-22 ~17:00 KST (deploy prep + offline + hw sanity
all PASS, real deploy pending CUDA reboot + openvla-7b 14GB download)**:

- Built `deploy_openvla_oft.py` 561 lines mirroring `deploy_smolvla.py` 4/9 Plan 3
  SUCCESS setup (INIT_POS [0,0,90,0,0,5] HOME, JOINT_SPEED_CAPS
  [500,500,500,300,300,300], gripper-only unlock pattern `arm.gripper_angle_ctrl(
  angle, speed=1000, acc=0)` directly after `joints_angle_ctrl`, Z_FLOOR=-130mm,
  DIST_MAX=420mm, Follower-only `--port /dev/ttyUSB0` blocked). Inference path
  replaces SmolVLA with OpenVLA-OFT (224×224 PIL RGB + language prompt, no state
  input, chunk (8,6) BOUNDS_Q99-denorm via `vla.predict_action`).
- Inline `L1RegressionActionHead` (deploy_openvla_oft.py:78-138) bypasses
  `prismatic.models.__init__` → `vlas` → `vla.materialize` → `dlimp` chain.
  See `claudedocs/DECISIONS.md` D086 for full rationale.
- Offline sanity 1+2+3 PASS (CPU only):
  1. Inline L1 head strict-load from B200 ckpt 7500 `action_head--7500_checkpoint.pt`
     after `module.` prefix strip: missing=0, unexpected=0, 134,328,326 params,
     forward (1,48,4096)→(1,8,6) OK.
  2. `dataset_statistics.json` key `roarm_v6_pick` q01/q99 for all 6 joints inside
     JOINT_LIMITS even at ±1.0 saturation.
  3. Script `ast.parse` PASS + 3 critical sub-imports OK.
- Hardware sanity 4+5 PASS:
  4. Kinect 720P NFOV_UNBINNED 1-frame capture (1280×720×3 BGR) →
     `logs/hw_sanity_20260522/kinect_sanity_frame.png`.
  5. Follower `/dev/ttyUSB1` (serial `ee7a06468e98ef1194edca63a8793231`, Leader
     USB0 serial `7842202ff8d9ef11b33f513dc8728757` per
     `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/tech_leader_follower_setup.md`)
     → torque ON → INIT_POS reached in 0.5s, max_diff=1.93°,
     FK pose x=353 y=2 z=204 mm (Z_FLOOR/DIST_MAX safe).
- Blockers for Step 6 real deploy:
  - CUDA driver mismatch `Failed to initialize NVML / NVML lib 580.159 / Error 804
    forward compatibility` → `torch.cuda.is_available()=False`. Fix = `sudo reboot`
    (no PC power-cycle needed).
  - `openvla/openvla-7b` HF cache 14 GB download in background at pinned revision
    `47a0ec7fc4ec123775a391911046cf33cf9ed83f`, ~2 GB / 14 GB at session end.
- `roarm` conda env additions: `peft 0.18.0` (`--no-deps`), `rich 15.0.0`,
  `timm 0.9.16` (HARD RULE #15 pin), prismatic editable from
  `/home/cgxr/Documents/Robotics/openvla-oft/`.
- Full detail: `claudedocs/session_20260522_track_b_p4_deploy_prep_offline_hw_sanity.md`.
- DECISIONS: D086 OpenVLA-OFT local inference deps + inline action head pattern.
- Ledger row: 2026-05-22 (Track B P4 deploy prep ...) at line 118.
- Next session: `sudo reboot` → verify CUDA → resume snapshot_download → GPU
  dry-run sanity (1-chunk inference) → Kinect dry-run → real deploy
  (multi-position, head-to-head vs SmolVLA v6 4/9 Plan 3 SUCCESS baseline).
  Verbatim continuation prompt in P4 session doc.

**Track B P4.5 — 2026-05-22 ~19:00 KST (post-P4 verification, real deploy
aborted at Step 0, reboot still pending)**:

- Session entered under premise "reboot done after P4, proceed to GPU dry-run
  + real deploy". Premise verified **FALSE**: `uptime`=1d 20:02,
  `who -b`=2026-05-20 22:53. No reboot between P4 prep (today 2026-05-22) and
  this session.
- `nvidia-smi` still returns `Failed to initialize NVML: Driver/library
  version mismatch / NVML library version: 580.159`. Kernel module
  `580.126.09`, userspace `libnvidia-ml.so.580.159.03`. Same P4 Blocker (a).
- `conda run -n roarm python -c "import torch; print(torch.cuda.is_available())"`
  → `False` (Error 804 forward compatibility).
- `openvla/openvla-7b` HF cache 14 GB on disk (17 blobs) but
  `snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f/` only shows
  `model-00003-of-00003.safetensors` symlink; 00001/00002 symlinks not
  finalized. Likely byte-complete in blobs; next session must re-run
  `snapshot_download` for idempotent fixup.
- No `deploy_openvla_oft.py` change, no env change, no Isaac, no RL, no robot
  command, no Track A file touched. User explicitly chose "Reboot 후 새 세션
  (권장)" via AskUserQuestion.
- Full detail: `claudedocs/session_20260522_track_b_p4_5_reboot_blocked.md`
  (includes verbatim continuation prompt for the post-reboot Track B P5 real
  deploy session).
- Ledger row: 2026-05-22 (Track B P4.5 post-P4 verification ...) at line 119.
- No new DECISIONS entry (no durable lesson — operational reboot omission,
  not a new rule).
- Next: user runs `sudo reboot` from terminal (assistant cannot run sudo
  autonomously). After ~1 min, new Claude Code session, paste P4.5 doc's
  continuation prompt to start Track B P5.

## Latest Verified Track A B200 Evidence

User approved exactly one close_26-only v6 projected-guard runtime on B200 GPU0,
followed immediately by v6 posthoc audit. It failed.

- Runtime stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out`
  md5 `9a4f8825a88ee3c9d93d83e5b9a28b41`, 430 lines.
- Runtime stderr:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.err`
  md5 `947cab475a1eff6ad2f3ccea6505d8c4`, 3 lines / 377 bytes.
- Audit stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out`
  md5 `480a3355864937763eb665e086aadbb0`, 58 lines.
- Audit stderr md5 `d41d8cd98f00b204e9800998ecf8427e` (empty).

Key v6 runtime lines:

- line 43: strict diagnostic-only, close_26-only, v6 flag YES, no training,
  constraints, SurfaceGripper, transport/release, gate tuning, posewrite, or
  success claim.
- line 45: `mode=target_guarded_micro_close_v6_projected_guard_diagnostic`
  and separate-approval marker YES.
- lines 393-397: v6 correctly blocked advance when projected support/target
  margins went unsafe; recovery writes continued with IK OK.
- line 398: first hard freeze/support-gate breach. `target_error=0.002914m`
  was still inside the fixed 0.003m target gate, but counter support gap was
  `0.002075m > 0.002m`; hard freeze YES.
- line 399: both fixed gates were breached: target error `0.003052m > 0.003m`
  and support gap `0.002146m > 0.002m`.
- lines 427-428: posthoc FAIL, 4 advances, 41 holds, zero zero-backlog holds,
  zero safety rollbacks, 12 recovery writes, 0 IK failures, 29 hard freezes,
  `close_reached=NO`, attach/posewrite zero, telemetry-only YES,
  success_claim NO.

Key v6 audit lines:

- line 18: `close_reached pass=NO`.
- lines 27-30: zero zero-backlog holds, zero safety rollbacks, positive recovery
  writes, and zero IK failures all PASS.
- line 31: hard safety freezes zero FAIL (`value=29`).
- lines 51-53: hard freeze / fixed target / fixed support criteria FAIL from
  runtime lines 398-426.
- lines 54-56: recovery present, preemptive trigger seen, and IK OK all PASS.
- line 58: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

Interpretation:

- v6 fixed the specific v5 mistake at old line 394: it did not advance once the
  projection went unsafe.
- v6 still failed because the recovery/hold behavior did not reduce target and
  support error fast enough after advance was blocked. The first failure is a
  support-gate hard freeze at runtime line 398, followed by target+support breach
  at line 399.
- Runtime exit 0 is not success. Success requires audit line 58 to be YES.

## Previous Track A Evidence To Keep In Mind

- v5 runtime/audit remain FAIL: stdout md5
  `f93ddaa75920a560777f8f9c8fae26f0`, audit md5
  `7709c2bc37424bc7c3874e978b34d104`. v5 line 394 advanced while support margin
  was too small; v6 corrected that specific advance decision but not the overall
  close_26 outcome.
- D083: RL-to-expert-to-rollout-to-demo is valid only after a no-attach Stage 0
  contact gate. Existing attach-based Pick/Stack PPO envs are not Track A
  evidence.
- D084: v5 recovery writes alone were insufficient; next advance needed projected
  fixed target/support margin checks.
- D085: v6 projection alone is also insufficient; once projection blocks advance,
  the mechanism must actively recover target/support before hard freeze.

## Current Direction

1. Do not rerun v2, v3, v4, v5, v6, or v7 unchanged.
2. For the professor's immediate cube3cm push/tap branch, keep it separate from
   Track A grasp/dataset/training. The canonical waypoint actor
   `model_actor_waypoint_lowx130.pt` at gain `0.040` has a sharded 10k
   teacher-off first-episode robustness gate PASS, but it is not dataset-ready,
   not Track A evidence, and not PPO/RL/VLA final success.
3. The professor-branch metric cleanup is implemented in the bucket audit/report:
   use the hierarchical sharded-10k output to report `1/5/10/20/30mm`,
   `disp/object_size`, controlled, no-impact, and low-motion. Treat the old 3cm
   `success_marker` as a strict task marker, not the sole objective.
4. Future professor-branch code edits must start with a short code review of
   reset, target generation, action/clipping, logging, and metrics before patching.
5. The professor-branch 10cm/0.72kg push/tap objective is now reaction-first,
   not final-displacement-first. The old 128 v1, fixed16 side-center, and
   settled-start fixed16 gates still failed as final 1cm displacement gates.
   Later 4-env diagnostics narrowed the controller issue: far-face/cross-object
   TCP targets are too aggressive; near-face geometry reduces TCP error; default
   actuator/step control fails to contact; drive boost can move the object. The
   measured-stop freeze seed940 run is a fixed-geometry reaction-event PASS under
   the clarified push/tap criterion: measured contact `1.0`, transient max
   displacement mean `0.010990217m`, transient 1cm rate `1.0`, no overshoot, but
   final displacement gate `0.0` and DiffIK clipping/lag remain. The local
   reaction audit is implemented. The approved seed941 randomized 16-env screen
   FAILed because contact evidence was only `0.625` despite reaction evidence
   `1.0`; no-posewrite and overshoot checks were OK, and teacher quality stayed
   false. Local direction buckets showed `x+` and `y-` contacted in seed941,
   while `y+` and `x-` were weak; cap-only seed942 worsened contact evidence to
   `0.5` and did not remove clipping/lag. The fixed-y+ seed943 screen confirmed
   y+ is weak: reaction `0.9375`, contact evidence `0.375`, no overshoot, final
   TCP error about `0.0706m`, and clip `1.0`. The local y+ geometry/reach audit
   shows contact/no-contact is also workspace-position dependent and vertically
   under-reached at the traced side-center target. The y+ trace-path audit shows
   target world-y advance is `0.019999981m`, final target z is essentially start
   cube z, final z error dominates TCP error, and clip_any is `1.0` for both
   contact and no-contact traced groups. The approved height050 y+ discriminator
   seed944 improved final TCP error to `0.022889409m` but killed contact evidence
   (`0.0`) and failed reaction (`0.6875`), so target-height-only is not the
   recovery path. The approved seed945 good-workspace screen fixed cube
   `x=0.295,y=-0.044` and PASSed the reaction gate (`reaction=1.0`,
   `contact=1.0`, no overshoot/posewrite), but teacher quality still failed
   (`final_tcp_err=0.065514732m`, clip `1.0`). The seed946 lateral `-0.020m`
   screen in that good workspace is the strongest 10cm y+ teacher-candidate
   evidence so far: reaction/contact `1.0/1.0`, final and transient 1cm gates
   `1.0`, no overshoot/posewrite, but teacher quality still false
   (`final_tcp_err=0.062820967m`, clip `1.0`). Next work is actuator/IK tracking
   cleanup or a tiny robustness check around this exact candidate, not cap-only
   escalation, dataset generation, or RL scale-up yet.
6. Do not run Track A dataset generation, PPO/training, rollout, hold-lift,
   transport/release, constraints, SurfaceGripper, or gate tuning from this
   result.
7. v7 active recovery is implemented and diagnostic telemetry works, but the
   post-reboot close_26 audit FAILED. It is not grasp success.
8. The first approved v8 runtime FAILED before recovery could trigger. Do not
   rerun that pre-fix v8 state. A post-fail static fix now makes v8 inherit
   virtual damping and reports `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`, but
   this is not physics validation. The next valid Track A action is exactly one
   post-fix close_26-only v8 runtime only after explicit user approval, followed
   immediately by v8 audit. In Codex, any future GPU/Isaac command still needs
   `sandbox_permissions=require_escalated` because the default sandbox hides
   `/dev/nvidia*` even though host CUDA is healthy.
9. Runtime PASS is not enough for Track A data; next gate is hold-lift.
10. Track A dataset/training remain blocked until close_26 PASS + hold-lift PASS + small
   pilot dataset/replay PASS. Then proceed: no-attach RL env → random sanity →
   PPO smoke → expert rollout → pilot dataset → replay/audit → large dataset →
   BC/VLA/IL training.
11. Do not plan future work around B200 SSH. Use local backups plus local/RunPod
   GPUs. Any remote compute should start by rebuilding/verifying env and smoke
   tests from backed-up artifacts.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` D083-D124
4. `claudedocs/EXPERIMENT_LEDGER.md` latest rows
5. `claudedocs/session_20260522_track_a_v6_projected_guard_runtime_fail.md`
6. `claudedocs/session_20260522_track_a_contact_rl_stage0_preflight.md`
7. `claudedocs/session_20260526_track_a_stage0_to_dataset_step_plan.md`
8. `claudedocs/session_20260526_track_a_v7_active_recovery_static_readiness.md`
9. `claudedocs/session_20260526_track_a_v7_local_runtime_cuda_blocked.md`
10. `claudedocs/session_20260522_b200_retirement_track_a_b_backup_verified.md`
11. `claudedocs/session_20260523_b200_disconnected_next_steps.md`
12. `b200_backup_20260522_final/README_BACKUP.md`
13. `sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py`
14. `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
15. `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
16. `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
17. `claudedocs/session_20260526_runpod_mcp_codex_registration_and_next_prompt.md`
18. `claudedocs/session_20260526_track_a_cuda_reboot_codex_sandbox_ready.md`
19. `claudedocs/session_20260526_track_a_v7_active_recovery_runtime_fail.md`
20. `claudedocs/session_20260526_track_a_v7_failure_static_analysis.md`
21. `claudedocs/session_20260526_track_a_v8_observed_recovery_static_design.md`
22. `sim_scripts/p7_branch_b_cube2cm_target_guarded_v8_observed_recovery_static_design.py`
23. `claudedocs/session_20260526_track_a_v8_runtime_candidate_static_readiness.md`
24. `claudedocs/session_20260526_track_a_v8_runtime_fail_and_damping_wiring_fix.md`
25. `claudedocs/session_20260606_cube10cm_reaction_window_contract.md`
26. `claudedocs/session_20260604_cube10cm_diffik_teacher_gate_prep.md`
27. `claudedocs/session_20260604_cube3cm_hierarchical_bucket_audit.md`
28. `claudedocs/session_20260602_cube3cm_push_metric_reframe_targetext.md`
29. `claudedocs/session_20260529_cube3cm_waypoint_actor_gate.md`
30. `claudedocs/session_20260529_cube3cm_actor_distillation_gate.md`
31. `claudedocs/session_20260529_cube3cm_bc_teacher_bridge_redesign.md`
32. `claudedocs/session_20260528_cube3cm_safety_rl_warmstart.md`
33. `claudedocs/session_20260526_cube3cm_push_rollout_probe_professor_request.md`
34. `sim_scripts/cube3cm_push_rollout_probe.py`
35. `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runtime.out`
34. `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/rollout_stats_audit.out`
35. `claudedocs/session_20260526_cube3cm_push_rl_reward_curriculum.md`
36. `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_speed_guard_model49_eval1024_audit.out`

## Do Not Trust As Current

- `HANDOFF.md`
- `TASKS.md`
- Any claim that runtime exit code 0 means grasp success
- Any claim that target-guarded v1 through v6 passed close_26
- Any claim that target-guarded v7 passed close_26
- Any claim that v5, v6, or v7 should be rerun unchanged
- Any claim that v7 candidate-level selected margins or negative counter-gap
  deltas prove observed runtime recovery
- Any claim that v8 observed-recovery static design is a physics result or means
  v8 is runtime-ready
- Any claim that v8 readiness YES means close_26 physics PASS
- Any claim that the first approved v8 runtime passed close_26
- Any claim that pre-fix v8 should be rerun unchanged, or that v8 readiness before
  the damping-inheritance check proved virtual damping was active
- Any claim that the professor cube3cm push/tap rollout is Track A grasp success,
  PPO/VLA training output, or dataset readiness
- Any claim that the professor cube-push PPO checkpoint is a successful learned
  policy before teacher-off 128 and then 1024 overall/per-bucket audits pass
- Any claim that `model_actor_distill.pt` is a successful learned policy; its
  teacher-off 128 audit failed low-motion/per-bucket gates
- Any claim that `model_actor_waypoint_lowx130.pt` is PPO/RL/VLA final success,
  dataset-ready, or Track A evidence; it has a sharded 10k teacher-off learned
  policy gate PASS for the separate professor cube3cm branch, but not those
  broader claims
- Any claim that the old 3cm `success_marker` is the professor push/tap objective
  by itself. Report hierarchical displacement thresholds and `disp/object_size`
  together with controlled/no-impact.
- Any claim that final 1cm displacement is still the only professor 10cm/0.72kg
  push/tap success criterion. Under the clarified objective, reaction/contact,
  transient displacement, z lift, and speed matter first; final displacement is
  secondary unless the task is relocation.
- Any future 10cm wrapper/default command that encodes final 1cm relocation as the
  primary objective. The wrapper defaults are tap/reaction `0.001m`; 1cm remains a
  secondary relocation/transient diagnostic only when explicitly used.
- Any claim that one lucky success from massive IsaacLab randomization is a
  learned policy, teacher-data readiness, or sim-to-real evidence by itself.
- Any claim that speed-only motion without contact evidence is a push/tap PASS.
  The reaction gate requires contact evidence too.
- Any claim that seed941 randomized 16-env screen passed the reaction gate. It
  failed on contact evidence (`0.625 < 1.0`) and remains teacher-quality false.
- Any claim that simply increasing DiffIK joint-step cap solves the randomized
  10cm reaction gate. seed942 cap050 failed worse on contact evidence (`0.5`).
- Any claim that y+ randomized failures are noise only. Fixed-y+ seed943 also
  failed contact evidence (`0.375`) with no overshoot and high clip/error.
- Any claim that y+ can be fixed by relabeling speed/z reaction as success. The
  local geometry audit shows no-contact rows had only `0.000069159m` mean max
  push displacement despite reaction-like speed/z signals, and traced envs kept
  large vertical TCP-target error at the side-center target.
- Any claim that y+ failure is caused by a missing y target advance. The local
  trace-path audit shows target world-y moves about `0.020000m`; the unresolved
  issue is vertical reach/actuator clipping and workspace-conditioned contact.
- Any claim that +5cm target height fixes fixed-y+ 10cm contact. seed944
  height050 reduced final TCP error but produced contact evidence `0.0`,
  reaction `0.6875`, and reaction gate FAIL.
- Any claim that seed945 good-workspace reaction PASS means 10cm teacher/data/RL
  readiness. It passes contact/reaction, but teacher quality is still false due
  final TCP error `0.065514732m` and DiffIK clip `1.0`.
- Any claim that seed946 lateral `-0.020m` makes 10cm data/RL ready. It is the
  best y+ candidate so far and passes final 1cm reaction/relocation gates, but
  teacher quality remains false because final TCP error is `0.062820967m` and
  DiffIK clip is `1.0`.
- Any claim that the professor 10cm branch must directly call the legacy
  `cube3cm_push_diffik_probe.py` filename. Prefer the new 10cm wrapper entrypoint
  for future 10cm commands, while keeping the shared engine and old logs intact.
- Any claim that the weighted mid/high actor candidate or the target-extension
  probe is ready for 1024/10k scale-up; both failed the 128 posx bucket screen
- Any Track B/OpenVLA training status as evidence for Track A contact success
- Any claim that existing default Pick/Stack PPO envs produce Track A-valid
  no-attach contact experts
- Any plan that requires new B200 SSH access after 2026-05-22 23:59 KST
- Any assumption that `.ssh` secrets were or should be copied as research data
- Any claim that Codex RunPod MCP is available or unavailable without checking
  both `/home/cgxr/.codex/config.toml` and the currently loaded tool namespace
- Any assumption that all complete Track B outputs live under
  `b200_backup_20260522_final/outputs` alone
- Any use of stale RunPod pod `az53n8t8alp8pz` from 2026-05-06 unless the user
  explicitly confirms it is current and active
- Any claim that default Codex sandbox `nvidia-smi` failure means host CUDA is
  still broken. Post-reboot host CUDA is healthy; default sandbox hides
  `/dev/nvidia*`.
- Any assumption that `/tmp/track_a_v7_active_recovery_runpod_overlay_20260526.tar.gz`
  still exists after reboot. `/tmp` is volatile; recreate the overlay if RunPod
  is needed.

## Current Dirty/Untracked Note

Dirty/untracked state is expected. Do not revert it unless explicitly requested.
Track B/OpenVLA files may be present; they are separate from Track A verdicts.

## Continuation Prompt For Next Session

```
Read CLAUDE.md first, then follow Current-State Protocol exactly.

한국어로 브리핑하고, 비판적/분석적으로 진행.
기억만으로 말하지 말고 반드시 파일/라인과 로컬 백업 로그 라인을 확인.
HANDOFF.md / TASKS.md 사용 금지.
기존 dirty/untracked 상태를 임의로 되돌리지 말 것.
B200은 만료/disconnect 상태다. 절대 ssh JHPark / B200 재접속 / 추가 pull / .ssh 복사 시도 금지.
local + RunPod + 로컬 백업만 사용.

Start by running:
git status --short --untracked-files=all

Must read:
1. CLAUDE.md
2. START_HERE.md
3. claudedocs/DECISIONS.md D083-D106
4. claudedocs/EXPERIMENT_LEDGER.md latest rows
5. claudedocs/session_20260522_track_a_v6_projected_guard_runtime_fail.md
6. claudedocs/session_20260522_track_a_contact_rl_stage0_preflight.md
7. claudedocs/session_20260526_track_a_stage0_to_dataset_step_plan.md
8. claudedocs/session_20260526_track_a_v7_active_recovery_static_readiness.md
9. claudedocs/session_20260526_track_a_v7_local_runtime_cuda_blocked.md
10. b200_backup_20260522_final/README_BACKUP.md
11. sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py
12. sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py
13. sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py
14. sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py
15. claudedocs/session_20260526_runpod_mcp_codex_registration_and_next_prompt.md
16. claudedocs/session_20260526_track_a_cuda_reboot_codex_sandbox_ready.md
17. claudedocs/session_20260526_track_a_v7_active_recovery_runtime_fail.md
18. claudedocs/session_20260526_track_a_v7_failure_static_analysis.md
19. claudedocs/session_20260526_track_a_v8_observed_recovery_static_design.md
20. sim_scripts/p7_branch_b_cube2cm_target_guarded_v8_observed_recovery_static_design.py
21. claudedocs/session_20260526_track_a_v8_runtime_candidate_static_readiness.md
22. claudedocs/session_20260526_track_a_v8_runtime_fail_and_damping_wiring_fix.md
23. claudedocs/session_20260526_cube3cm_push_rollout_probe_professor_request.md
24. sim_scripts/cube3cm_push_rollout_probe.py
25. claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runtime.out
26. claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/rollout_stats_audit.out
27. claudedocs/session_20260526_cube3cm_push_rl_reward_curriculum.md
28. claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_speed_guard_model49_eval1024_audit.out
29. claudedocs/session_20260527_cube3cm_diffik_v3_10k_audit.md
30. claudedocs/session_20260527_cube3cm_diffik_v3_visualization.md
31. claudedocs/session_20260528_cube3cm_diffik_dataset_bc_policy.md
32. claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_state_action_dataset_v2_1024_seed779_audit.out
33. claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_state_action_dataset_v2_1024_seed779_bc_train.out
34. claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/bc_mlp_joint_delta_v1_rollout1024_seed883_audit.out

Current Track A state:
- v6 close_26 audit FAIL, not grasp success.
- Runtime stdout:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out
  md5 9a4f8825a88ee3c9d93d83e5b9a28b41
- Audit stdout:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out
  md5 480a3355864937763eb665e086aadbb0
- Reverify runtime lines 43,45,393-399,427-428 and audit lines 18,27-31,51-58 before citing.
- Interpretation: v6 blocked unsafe projected advance but recovery writes did not reduce target/support margins. First support failure line 398; first target+support breach line 399.

Professor pipeline decision:
- RL→expert→rollout→dataset→learning is valid only after Stage 0 no-attach contact primitive.
- Do NOT use existing RoArm-Pick/Stack PPO envs as clean Track A expert sources because they use kinematic attach / write_root_pose_to_sim.
- Do NOT start PPO/training/dataset/rollout first.

Current v7 status:
- Default-off v7 active target/support recovery candidate is implemented and has
  now been physics-tested locally after reboot.
- It uses finite-difference TCP candidate sweep with current object pose and robot joint target writes only.
- Objective: maximize minimum fixed target/support margin; reduce counter gap before next close advance while keeping target error inside fixed 3mm gate.
- Matching audit/readiness support and negative controls exist.
- Local static checks already passed: py_compile, git diff --check, synthetic v7 pass, v7 no-active-recovery reject, archived v6 log reject as v7, readiness.
- Approved local runtime attempt on 2026-05-26 did not produce a physics result: local IsaacLab emitted v7 metadata, then CUDA failed before env creation.
- Preserved local-block logs:
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/runtime.out
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/runtime.err
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/audit.out
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/cuda_check.txt
- Reverify runtime.out lines 28,31,33,35; runtime.err lines 9-16,32-52; audit.out lines 3,16,18,29-30,35; cuda_check.txt lines 1-3,12-13 before citing.
- User rebooted the local Ubuntu PC after that blocked attempt. Host CUDA is now healthy: boot time 2026-05-26 14:08, NVIDIA kernel/userspace 580.159.03, host nvidia-smi OK, isaaclab torch CUDA True/device_count 1 when run outside the default Codex sandbox.
- Default Codex sandbox still hides /dev/nvidia*, so sandboxed nvidia-smi fails. This is a sandbox device visibility issue, not host CUDA failure. Run GPU/Isaac commands in Codex with sandbox_permissions=require_escalated.
- Post-reboot readiness still reports READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES.
- Post-reboot close_26-only v7 runtime/audit FAILED:
  claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/runtime.out
  md5 621d00b9d157b4e70178c28f94ca4c7f
  claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/audit.out
  md5 406b96557d94418f16273e517ec4d69b
- Reverify runtime.out lines 389-393,423-424 and audit.out lines 19,32,54-56,60-66 before citing.
- Interpretation: v7 active recovery did trigger and passed its v7-specific audit checks, but fixed support failed first at runtime line 392 and fixed target failed at line 393. This is not close_26 success and v7 must not be rerun unchanged.

Current v8 status:
- Static-only v8 observed-recovery design is done:
  sim_scripts/p7_branch_b_cube2cm_target_guarded_v8_observed_recovery_static_design.py
  md5 56a382377b7fb0f0c6391bf59163af0d
  claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_observed_recovery_static_design.out
  md5 c14e80ec5fc69c6e6e17925d61f81d0b
- Reverify v8 output lines 1-28 before citing.
- Interpretation: v8 static design finds projected reserve depletion at runtime line 386 / step 9, before v7 first active at line 389 / step 12 and first support breach at line 392 / step 15. It rejects unchanged v7 by observed-response/TCP-follow checks. This is not a physics result: output line 27 says RUNTIME_READY=NO.
- Default-off v8 runtime candidate + matching v8 audit/readiness were implemented:
  sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py pre-fix md5 7e6dfc35bbfeacb5d1689f2f175e5120
  sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py md5 8dbf621c983ec03f46e5d52843781fda
  sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py pre-fix md5 a31ced20b754a4a42058349525d1a435
  claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_runtime_candidate_readiness.out md5 6a2a62808451175b65e5d522b695b8b6
- Reverify readiness lines 1-19 and v8_rejects_post_reboot_v7_audit.out lines 5,14,15,20,30,53,55,57,58,60 before citing.
- The first approved v8 close_26 runtime has now been run and FAILED:
  claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/runtime.out
  md5 74095570c2d6a60abdf522c2413735db
  claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/audit.out
  md5 7cd38eddb1dc9c925b01948cbc5cb416
  claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/v8_runtime_failure_static_analysis.out
  md5 7e81773b91d39658a3ec5c6eaf878f0c
- Reverify runtime.out lines 4,6,8,423-424; audit.out lines 20,26,35-36,45-49,53,60; and v8_runtime_failure_static_analysis.out lines 4,8-12 before citing.
- Post-fail static fix: v8 now inherits virtual damping. Current runtime probe md5 acae0ca2e85a522dd4ac8fb583cb8fb8; readiness md5 dc2bdaa8d882f12b5cc901a677caccc0; readiness_after_v8_damping_fix.out md5 b652520a81792bf12373ff742cdba6b5 lines 5 and 20 PASS. No post-fix v8 runtime has run.

Current RunPod/Codex state:
- Claude has RunPod MCP configured, but Codex initially did not.
- Codex config was updated at /home/cgxr/.codex/config.toml with [mcp_servers.runpod], command npx, args ["-y", "@runpod/mcp-server@latest"], and RUNPOD_API_KEY copied from Claude config without printing the value.
- Backup exists: /home/cgxr/.codex/config.toml.bak_runpod_20260526 md5 1ef4acf6f1c92a64b9bbd79a2e35b7e7.
- Same-session tool_search after config edit initially did not expose mcp__runpod__..., but later Codex sessions did expose mcp__runpod__. In this session, tool_search exposed mcp__runpod__ and `list_pods(computeType=GPU)` returned `[]`. Still verify loaded tools before claiming RunPod MCP can be used.
- Do not use stale RunPod pod az53n8t8alp8pz from 2026-05-06 unless the user explicitly confirms it is current and active.
- The old minimal RunPod overlay at /tmp/track_a_v7_active_recovery_runpod_overlay_20260526.tar.gz was lost after reboot because /tmp is volatile. Recreate it if RunPod is needed.
- Local backup top USD path:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd
  md5 4497024d25abab11de5c50e144124553.

Current professor cube10cm push/tap branch:
- Separate from Track A close_26 grasp and separate from the older 3cm dataset/BC
  line. The active objective is tap/reaction first, not final 1cm relocation.
- Future 10cm DiffIK commands should use:
  `sim_scripts/cube10cm_push_diffik_probe.py`
  while keeping `sim_scripts/cube3cm_push_diffik_probe.py` as the shared legacy
  engine for old tools/logs.
- Guard before any new 10cm runtime:
  `python sim_scripts/cube10cm_tap_objective_contract_audit.py`
  The latest guard JSON says contract `professor_cube10cm_tap_reaction`,
  final 1cm relocation default `false`, tap defaults `0.001m`, explicit 1cm
  override allowed, and verdict PASS.
- Latest object-level candidate remains seed946:
  reaction/contact `1.0/1.0`, no posewrite, no overshoot, final relocation
  secondary pass, but `teacher_quality_ready=false`.
- Latest actuator-tracking screen stiff600:
  reaction/contact `1.0/1.0`, no posewrite, no overshoot, but only tap-scale
  displacement (`max_disp_mean_m=0.004667487`) and `final_relocation_pass=false`;
  `teacher_quality_ready=false`, clip `1.0`, final TCP error `0.046062952m`.
  This does not supersede seed946 as the strongest object-level candidate.
- Latest direction-generalization screen seed947:
  same goodxy/lateral recipe as seed946 but random directions. It FAILed reaction
  gate: contact evidence `0.5625`, reaction `1.0`, no posewrite, no overshoot.
  Direction split: y+ controlled `1.0` but low-motion `0.75`; x- contact `0.0`;
  x+/y- contact `1.0` but controlled `0.0`. This blocks 1024/10240/data.
- Latest x- contact-geometry screen seed948:
  fixed goodxy/lateral with `fixed_push_dir=[-1,0]` and only
  `push_through_m=0.020`. It FAILed harder: contact evidence `0.0`, reaction
  `1.0`, no posewrite, no overshoot, max displacement `0.000009052m`, final
  displacement `-0.000160992m`, clip `1.0`, final TCP error `0.062163922m`.
  This falsifies the simple "deeper near-face target fixes x-" hypothesis.
- Latest x- reach/IK feasibility screen seed949:
  fixed goodxy/lateral and fixed x-, but with `tcp_center_height_offset_m=0.050`.
  It restored tap contact/reaction: reaction/contact `1.0/1.0`, no posewrite,
  no overshoot, contact stop `1.0`, final TCP error `0.012976003m`, clip
  `0.460576925`, and next-step audit `teacher_quality_ready=true`. However it is
  still tap-scale only: final displacement `0.001272157m`, max displacement
  `0.001294456m`, low-motion `1.0`, final relocation secondary false. This is a
  reach/height clue, not 1024/data readiness.
- Direction-specific height support is now in the shared DiffIK probe:
  `--xneg_tcp_center_height_offset_m`, `--xpos_tcp_center_height_offset_m`,
  `--yneg_tcp_center_height_offset_m`, and `--ypos_tcp_center_height_offset_m`.
  Both push and retract side-center targets use the per-env height.
- Latest random-direction heldout seed950:
  goodxy/lateral plus only `xneg_tcp_center_height_offset_m=0.050` produced
  reaction/contact `1.0/1.0`, no posewrite, no overshoot, and all directions had
  contact evidence. x- was fixed (`controlled=1.0`, clip `0.457949`), but x+
  remained controlled `0.0`, clip `1.0`, final TCP error `0.058328m`; teacher
  quality remained false and data/1024 stayed blocked.
- Latest fixed x+ height050 screens:
  seed951 passed tap/reaction contact (`1.0/1.0`, no posewrite, no overshoot) but
  final displacement was only `0.000011876m`, clip `1.0`, and teacher false.
  seed952 slowed the x+ schedule to `320/180/80`; it improved some tracking means
  and final TCP error to `0.008242317m`, but clip remained `1.0`, joint1 clip rate
  `0.986637931`, low-motion `1.0`, and teacher false.
  seed953 changed only `dls_lambda` from `0.010` to `0.030`; tap/reaction stayed
  PASS, final TCP error stayed low at `0.008274879m`, but clip remained `1.0`,
  joint1 clip rate `0.979591837`, low-motion `1.0`, and teacher false.
  seed954 changed only `max_diffik_joint_step_rad` from `0.035` to `0.050`;
  tap/reaction stayed PASS and final 1mm rate improved to `0.1875`, clip dropped
  to `0.870833352`, but joint1 actuator follow mean worsened to `0.032731688rad`,
  low-motion stayed `1.0`, and teacher false.
  seed955 tried midpoint cap `0.0425`; tap/reaction stayed PASS, but clip stayed
  `1.0` and final 1mm rate was `0.0`, so this is not the compromise.
  seed956 kept cap `0.050` and changed only stiffness `400 -> 600`; tap/reaction
  stayed PASS, aggregate clip improved to `0.766826950`, and phase-split pre-stop
  clip improved to `0.570247934`, but teacher false, final 1mm rate `0.0625`,
  and post-stop freeze clip remained `1.0`.
  seed957 kept cap `0.050`/stiffness `600` and changed only effort limit
  `25 -> 35`; tap/reaction stayed PASS, aggregate clip improved only slightly to
  `0.749519262`, pre-stop clip to `0.534360190`, final 1mm rate `0.0`, and
  teacher false.
- Trace diagnostic now includes `phase_splits`:
  `sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py` separates
  pre-stop motion from post-stop freeze. This is diagnostic only; it does not
  relax the conservative next-step/data gates.
- Phase-window audit on seed957 rejects a freeze-only explanation:
  `full_clip_rate=0.749519262`, `pre_stop_clip_rate=0.534360190`,
  `post_stop_clip_rate=1.0`, teacher clip max `0.5`, and conclusion
  `PRE_STOP_ACTUATOR_IK_CLIP_STILL_BLOCKS`. The official gate is conservative,
  but pre-stop tracking is still slightly above threshold.
- Reaction audit no longer computes implicit default 1cm relocation. The default
  is tap-only: `tap_gate_disp_m=0.001`, `final_relocation_pass=None`. A 1cm
  relocation diagnostic now requires explicit `--final_relocation_disp_m 0.010`.
- Reaction-window contract audit/builder is now the local bridge from professor
  tap/reaction objective to data-unit labels. It anchors short windows on contact
  and requires contact evidence + reaction signal + no posewrite/training/attach
  + no overshoot; final 1cm relocation is not required. Existing-log cross-checks:
  seed957/seed949/seed950 accepted 16/16 windows, while seed948 accepted 0/16
  because no contact anchor existed. Clip/follow remain metadata and a separate
  clean-DiffIK teacher diagnostic, not the default reaction-window reject reason.
- Quality-tier v2 is the active interpretation: Tier A is clean DiffIK teacher,
  Tier B is reaction-valid with follow OK but clip high, Tier C is reaction-valid
  with actuator follow lag, and Rejected is not a valid reaction window. Current
  existing-log tiers: seed957 all B, seed949 all B, seed950 10 B + 6 C, seed948
  all Rejected.
- Latest next-step audit now says:
  `NARROW_ACTUATOR_IK_TRACKING_CLEANUP_INSIDE_WORKING_TAP_GEOMETRY`
  for seed959. Do not start 1024/data: y- reaction/contact is valid, but DiffIK
  clip is still `1.0`, final TCP error is `0.037017073m`, and actuator follow lag
  remains in the trace diagnostic.

Next concrete step:
1. Do not start dataset generation, PPO/RL, VLA, Track A, 1024/10k scale-up, or
   broad random search from seed946.
2. Before any new 10cm GPU runtime, run and cite:
   `python sim_scripts/cube10cm_tap_objective_contract_audit.py`
   and
   `python sim_scripts/cube10cm_next_research_step_audit.py`
3. Do not repeat cap0425 or stiffness/effort changes as success claims. The best
   fixed x+ actuator screen so far is seed957, and it is still teacher false.
4. Use `sim_scripts/cube10cm_reaction_window_contract_audit.py` on existing traces
   to decide which contact/reaction windows would be retained, with clip/follow
   metadata preserved.
5. First tier-distribution matrix is now available:
   `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_reaction_window_tier_matrix_existing_seeds.json`
   and `.csv`. Interpret x- with the audit/config split, not direction-only
   aggregation: seed948 x- is rejected, seed949/seed950 x- are valid Tier B.
6. Latest approved tiny y+ coverage screen seed958 added 16/16 accepted reaction
   windows, all Tier C. This answers y+ lucky-contact concern negatively but
   confirms follow-lag quality risk for y+.
7. Latest approved tiny y- screen seed959 added 16/16 accepted reaction windows,
   split 11 Tier B + 5 Tier C. y- coverage is no longer the immediate gap.
8. Latest approved tiny y+ quality screen seed960 changed only
   `max_diffik_joint_step_rad 0.035 -> 0.050`. It kept reaction/contact valid but
   stayed 16/16 Tier C, so cap-only y+ cleanup is not enough.
9. Latest approved tiny y+ stiffness screen seed961 changed only
   `arm_stiffness_override 400 -> 600`. It kept reaction/contact valid but stayed
   16/16 Tier C, so stiffness-only y+ cleanup is not enough.
10. Latest local per-window diagnosis found y+ Tier C is not a simple larger raw
   IK delta problem: y+ raw delta p95 is lower than the non-y+ Tier B baseline
   (`0.174502504` vs `0.280128609`), but follow/cap is higher (`1.223191874`
   vs `1.030052730`), contact anchors about `80.876804` steps earlier, and max
   XY displacement is `10.223485837x` the baseline.
11. Latest early-contact geometry audit confirms y+ accumulates large object
   reaction before/around the measured-contact anchor inside the approach phase:
   pre-anchor displacement is `13.694612400x` baseline and tip is
   `13.202625486x` baseline.
12. Latest local precontact candidate audit fixed the one-variable GPU candidate:
   fixed y+ seed958-like geometry, only `precontact_clearance_m 0.010 -> 0.020`,
   seed962. It was a hypothesis test for approach/pre-anchor reaction
   accumulation, not a data-scale step.
13. seed962 pre020 runtime PASSed reaction/contact/no-posewrite/no-overshoot but
   did not solve quality or data readiness: reaction-window 16/16 accepted, 2
   Tier B + 14 Tier C, follow p95/cap p95 `1.160505840`, clip mean `1.0`,
   teacher quality false, controlled push only `0.5625`.
14. The pre020 intervention reduced y+ pre24 reaction and moved anchor after push
   start, but also weakened reaction strength; treat it as failure-mode movement,
   not a clean fix.
15. Latest local failure-shift audit confirms the same: seed962 pre24
   displacement/tip fell to `0.415034926x`/`0.381066167x` of the seed958/960/961
   mean, but max displacement/tip also fell to `0.661469914x`/`0.376103186x`;
   quality remains blocked.
16. Corrected contact-strength interpretation: seed962 final 1mm retention drop is
   not a primary failure because the active objective is tap/reaction/contact. The
   local audit now records `final_retention_primary_objective=False` and
   `selected_next_candidate=NONE_FROM_FINAL_RETENTION_ALONE`; use max transient
   1/2/3mm only if contact-strength is explicitly requested.
17. Latest transient tap-strength audit excludes final position as a success gate:
   seed962 primary 1mm tap event PASSes (`contact=1.0`, `reaction=1.0`,
   `overshoot=0.0`, max 1mm `1.0`), 2mm transient is majority `0.8125`, and 3mm
   transient is not reliable `0.5`.
18. Next order is now explicit: if 1-2mm tap is enough, stop y+ contact-geometry
   tuning and keep quality-tier metadata separate; if 3mm is required, define that
   transient target first, then propose exactly one local candidate. Do not use
   final 1cm/final retention.
19. Dataset/RL/robot readiness audit says event-label manifest is allowed, but
   action-teacher dataset, large IsaacLab dataset, IsaacLab RL, and RoArm-M3-Pro
   deploy are blocked: `clean_teacher=false`, 2B+14C, clip mean `1.0`,
   follow p95/cap `1.160505840`, existing RL env is still 3cm-oriented, and the
   old dataset builder uses final-success filters.
20. Created only a local event-label manifest: 16 reaction-window events, contact
   16, reaction 16, overshoot 0, window-level transient counts 16/13/7 for
   1/2/3mm. This is not action data, not LeRobot/RLDS, not training data.
21. Latest DiffIK action-dataset blocker audit confirms the exact pipeline state:
   event-label manifest is `READY_LOCAL_ONLY`, but DifferentialIK action-teacher
   dataset, large IsaacLab dataset, IsaacLab RL, and RoArm-M3-Pro deploy are still
   blocked. Code-level conflicts are also confirmed: the old dataset builder uses
   final controlled/low-motion/success filters, and the existing RL env is a
   3cm/20g relocation task.
22. Latest 10cm tap/reaction dataset-builder preflight locally bypasses the old
   final-success builder conflict: 16 preview rows, contact 16, reaction 16,
   overshoot 0, transient 1/2/3mm counts 16/13/7, forbidden final/success fields
   absent. This is still not an action-teacher dataset or training data.
23. Latest DiffIK teacher-quality policy gate keeps action dataset blocked by
   default: strict clean teacher fails, Tier-B-only has only 2 rows, and Tier-B/C
   noisy action teacher requires an explicit policy exception before any dry-run.
24. Latest teacher-quality revalidation improves the seed962 action-row policy:
   trimming to contact anchor `[0,+16]` gives 16/16 Tier B and removes Tier C
   follow lag, but clip remains `1.0` and Tier A remains zero. This supports a
   tiny Tier-B dry-run preview only, not a clean action dataset.
25. Latest Tier-B action dry-run preview has 16 events and 66 sparse trace rows,
   no forbidden final/success fields, and action abs p95/max `0.007rad`; it marks
   actual action-teacher dataset `NOT_BUILT` and keeps large dataset/RL/RoArm
   blocked.
26. Latest visual/sim sanity check produced a valid 10cm replay video from
   existing seed962 env0 trace, but did not verify a clean tap. The live
   `--record_video` path failed with zero frames; the trace replay path passed
   MP4/frame checks (`98` frames, `1280x720`, 10cm cube, `physics_recomputed=false`),
   yet the contact frame still has `tcp_minus_target_z=0.050453320m`,
   `tcp_target_err_before=0.050612349m`, and `clip_any=1`. Treat this as visual
   contact evidence plus teacher-quality blocker evidence, not dataset/RL/robot
   readiness.
27. Latest contact-frame geometry mismatch audit confirms all 16 seed962 first
   contact rows have the same blocker: side-center target is not reached, TCP
   contacts near the cube top, z error dominates TCP-target error, and clipping
   is saturated. This is not a missing TCP-local-offset compensation bug. The
   target-side-center code path is active, but the actual first-contact geometry
   is upper/top contact under `link1_to_link2` clipping.
28. Latest teacher contact-frame design audit tested all three criteria. Current
   `true_side_center_tcp` is the right semantic object goal but fails tracking;
   `upper_edge_contact_proxy` explains the current contact but teaches top contact;
   the selected teacher criterion is `tool_oriented_side_contact_proxy`.
29. The one approved top-margin negative-control runtime improved target tracking
   and summary teacher quality, but weakened the tap and kept upper/top contact.
   Therefore it is NOT the dataset teacher criterion and does NOT unlock
   dataset/RL/RoArm.
30. The approved link5-corner position runtime was executed once. It PASSed
   reaction/contact/no-posewrite/no-overshoot and improved proxy tracking plus
   reaction-window tier distribution, but it weakened reaction strength and still
   is not a clean DiffIK teacher. Treat it as evidence that proxy retargeting is
   promising but not yet data/RL/RoArm-ready.
31. The visual proxy-contact inspection confirms the link5-corner proxy is
   side-center-height rather than top contact, but the point stays outside/grazing
   the live approach face and contact_stop freezes on the contact step. This is a
   weak 1mm tap, not a clean stronger 2-3mm teacher.
32. Do not run 1024/10240 trace/data from seed946/seed947/seed948/seed949/seed950/
   seed951/seed952/seed953/seed954/seed955/seed956/seed957/seed958/seed959/seed960/seed961/seed962. Direction-specific
   contact is improving, but low-motion, DiffIK clipping, post-stop freeze
   clipping, and actuator follow lag still block teacher/data quality.
33. Any next GPU experiment still requires explicit approval. Do not repeat the
   same link5-corner position run, do not jump to pose first, and do not scale
   dataset/RL/RoArm. If 1mm tap is sufficient, stop contact-geometry GPU tuning;
   if 2-3mm is required, define that transient target first and design exactly
   one local strength-preserving contact timing/through candidate.
34. Judge the result by reaction/contact/no-posewrite/no-overshoot first; report
   final 1cm only as secondary relocation evidence.
```
