# Session 2026-06-10 - Cube10cm Tap RL Candidate8 Target Residual

## Scope

- Active branch: professor 10cm/0.72kg cube push/tap RL robustness.
- Not Track A. No B200/SSH/reconnect/pull/.ssh copy.
- Goal: move beyond harmful joint residual PPO by testing a lower-dimensional task-space residual around the Candidate6/Candidate7 DiffIK base.

## Verified Starting State

- `START_HERE.md:5` / D215 said xy +/-3cm is the useful randomized robustness task, with base around `success_episode_rate=0.7210084033613445`, overshoot `0.0625`.
- D215 also said joint-residual PPO L1 must not scale because three xy +/-3cm attempts degraded success and/or overshoot.
- RoArm arm control in the env uses five arm joints plus separate gripper, so a 3D position residual is a better next test than 6D joint residual or orientation residual.

## Code Changes

- Added `candidate8_diffik_target_residual` action mode in `roarm_rl/roarm_cube_push_env.py`.
- The policy action is interpreted as task-space TCP target residual before DiffIK:
  - action[0] -> forward residual along push direction
  - action[1] -> lateral residual perpendicular to push direction
  - action[2] -> height residual
  - no orientation residual
- Zero action calls the same Candidate6/Candidate7 DiffIK target generator, so zero-policy Candidate8 should match the base controller.
- Added residual telemetry:
  - aggregate target residual max/mean
  - forward/lateral/height residual maxima
- Added smoke-runner CLI/config/logging for Candidate8 and a base-relative L1 health readout:
  - `l1_health_pass`
  - `l2_scale_candidate`
  - success episode delta
  - overshoot delta

## Static Verification

- `python3 -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_tap10cm_ppo_smoke.py` passed.
- `git diff --check` passed.

## Runtime Results

### Candidate8 Zero-Policy Preflight

Files:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate8_target_residual_xy3cm_zero_preflight_seed1011_summary.out`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate7_currentpose_xy3cm_same_seed1011_base_compare_summary.out`

Contract:

- xy randomization +/-3cm
- `candidate6_diffik_cube_reference_mode=current_pose`
- `tap_success_terminate=True`
- Candidate8 default scales: forward `0.004m`, lateral `0.012m`, height `0.004m`
- `max_iterations=0`, no training

Result:

- Candidate8 zero-policy line 4: `success_episode_rate=0.7422360248447205`, `overshoot_max=0.125`.
- Candidate7/base same-seed comparison line 4: `success_episode_rate=0.7422360248447205`, `overshoot_max=0.125`.
- Interpretation: zero-action Candidate8 equals the base controller. Implementation bridge is valid.

### Candidate8 Default 3D Target Residual L1

File:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate8_target_residual_xy3cm_l1_40k_seed1012_summary.out`

Contract:

- `num_envs=32`, `num_steps_per_env=64`, `max_iterations=20`
- total transitions: `32*64*20=40960`
- PPO init noise `0.2`
- forward/lateral/height scales: `0.004/0.012/0.004m`

Result:

- line 4 pre: `success_episode_rate=0.7350993377483444`, `overshoot_max=0.0625`, `reward_mean_per_step=0.08071071922940659`.
- line 7 post: `success_episode_rate=0.6845425867507886`, `overshoot_max=0.53125`, `reward_mean_per_step=0.10252326592032251`.
- line 9 base-relative: `success_episode_delta=-0.050556750997555744`, `overshoot_delta=0.46875`, `signal_seen=True`, `l1_health_pass=False`, `l2_scale_candidate=False`.

Interpretation:

- PPO found a reward-improving behavior that worsens the actual quality gate.
- This is not an argument for more steps; it is an action/reward/termination contract problem.

### Candidate8b Conservative Lateral/Height L1

File:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate8b_latheight_xy3cm_l1_40k_seed1013_summary.out`

Contract:

- Same xy +/-3cm task.
- PPO init noise `0.1`.
- forward scale `0.0m`, lateral `0.006m`, height `0.002m`.

Result:

- line 4 pre: `success_episode_rate=0.6772151898734177`, `overshoot_max=0.125`.
- line 7 post: `success_episode_rate=0.6725352112676056`, `overshoot_max=0.5`.
- line 9 base-relative: `success_episode_delta=-0.004679978605812041`, `overshoot_delta=0.375`, `target_residual_abs_max_max=0.0010023463983088732`, `l1_health_pass=False`, `l2_scale_candidate=False`.

Interpretation:

- Even tiny lateral/height residuals can increase overshoot under the current reward/termination contract.
- Candidate8 action-space plumbing is still useful, but Candidate8 as-is must not scale to L2.

## Decision

- Candidate8 validates the task-space residual bridge, not the training contract.
- Do not run Candidate8 L2/large PPO from the current reward/action settings.
- Do not generate dataset, VLA data, action-teacher data, or RoArm deployment artifacts from these policies.

## Next Research Step

Candidate9 should keep the same low-dimensional DiffIK target residual idea but fix safety/credit assignment before learning scale-up:

- Make the PPO objective align with the gate: success without overshoot should dominate transient displacement reward.
- Gate or phase-limit residuals so the policy cannot create late/post-contact overshoot while still collecting transient reward.
- Keep zero-action equal to Candidate7/base.
- Rerun only L1 health first; L2 is allowed only if same-run base success is preserved or improved and overshoot is not worse.

## Candidate9 Safety/Credit Assignment Screen

### Code Changes

- Added default-off `candidate8_diffik_target_residual_zero_after_contact` in
  `roarm_rl/roarm_cube_push_env.py`.
- When enabled, the Candidate8 target residual is multiplied by zero after
  `_tap_contact_seen` becomes true. The base Candidate6/Candidate7 DiffIK target
  remains active; this only gates the learned residual.
- Added CLI/config/contract logging in
  `roarm_rl/train_cube_tap10cm_ppo_smoke.py`.
- This remains forward/lateral/height target residual only. No orientation
  residual and no action-teacher dataset path were added.

### Static Verification

- `python3 -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_tap10cm_ppo_smoke.py` passed.
- `git diff --check` passed.

### Candidate9a - Zero After Contact Only

File:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate9a_zeroaftercontact_xy3cm_l1_40k_seed1013_summary.out`

Contract:

- Same xy +/-3cm, seed1013, `num_envs=32`, `num_steps_per_env=64`,
  `max_iterations=20`.
- `candidate8_diffik_target_residual_zero_after_contact=True`.
- Reward stayed at Candidate8b values:
  `tap_transient_disp_reward_scale=40.0`, `tap_overshoot_penalty_scale=12.0`,
  `action_penalty_scale=0.005`.
- Residual scales: forward `0.0m`, lateral `0.006m`, height `0.002m`.

Result:

- line 4 pre: `success_episode_rate=0.6772151898734177`,
  `overshoot_max=0.125`.
- line 7 post: `success_episode_rate=0.6725352112676056`,
  `overshoot_max=0.5`.
- line 9 base-relative: `success_episode_delta=-0.004679978605812041`,
  `overshoot_delta=0.375`, `l1_health_pass=False`,
  `l2_scale_candidate=False`.

Interpretation:

- Zeroing residual only after contact is not enough. The failure is effectively
  identical to Candidate8b.

### Candidate9b - Reward-Safe + Zero After Contact

File:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate9b_rewardsafe_zeroaftercontact_xy3cm_l1_40k_seed1013_summary.out`

Contract:

- Same xy +/-3cm, seed1013 L1.
- `candidate8_diffik_target_residual_zero_after_contact=True`.
- Reward-safe weights:
  `tap_transient_disp_reward_scale=0.0`, `tap_overshoot_penalty_scale=120.0`,
  `action_penalty_scale=0.05`.
- Residual scales: forward `0.0m`, lateral `0.006m`, height `0.002m`.

Result:

- line 4 pre: `success_episode_rate=0.6772151898734177`,
  `overshoot_max=0.125`, `reward_mean_per_step=-0.5916546382055395`.
- line 7 post: `success_episode_rate=0.7161716171617162`,
  `overshoot_max=0.25`, `reward_mean_per_step=-0.4926078331307508`.
- line 9 base-relative: `success_episode_delta=0.03895642728829851`,
  `overshoot_delta=0.125`, `target_residual_abs_max_max=0.0009049359941855073`,
  `l1_health_pass=False`, `l2_scale_candidate=False`.

Interpretation:

- This is the first useful direction after Candidate8: success improves and
  overshoot is lower than Candidate9a/Candidate8b.
- It is still not a pass because same-run base overshoot is `0.125` and post
  overshoot is `0.25`.
- Do not scale Candidate9b to L2 yet.

### Candidate9c - Half-Scale Reward-Safe

File:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate9c_halfscale_rewardsafe_zeroaftercontact_xy3cm_l1_40k_seed1013_summary.out`

Contract:

- Same reward-safe settings as Candidate9b.
- Residual scales halved: forward `0.0m`, lateral `0.003m`, height `0.001m`.

Result:

- line 4 pre: `success_episode_rate=0.6772151898734177`,
  `overshoot_max=0.125`.
- line 7 post: `success_episode_rate=0.6572327044025157`,
  `overshoot_max=0.59375`.
- line 9 base-relative: `success_episode_delta=-0.01998248547090198`,
  `overshoot_delta=0.46875`, `l1_health_pass=False`,
  `l2_scale_candidate=False`.

Interpretation:

- Smaller residual scale is not automatically safer. Do not continue a naive
  scale grid.

## D217 Decision

- Candidate9b is the best partial branch, but there is still no L1 health pass.
- The blocker is now narrower: low-dimensional task residual can improve success,
  but credit/safety still allows overshoot worse than same-run base.
- Next pass route is not more audit, not larger PPO, and not a residual-scale
  sweep. It should be a stricter overshoot-safe success/residual design:
  - disable residual before the overshoot margin using displacement or reaction
    state, not only after contact;
  - or make overshoot-free success the only meaningful positive learning credit;
  - then rerun one L1 screen and require same-run base success preserved/improved
    and overshoot not worse.
- Dataset, action-teacher, VLA, large PPO, and RoArm deployment remain blocked.

## Candidate10 Early Displacement Gate Screen

### Code Changes

- Added default-off gates to the Candidate8 target residual path:
  - `candidate8_diffik_target_residual_zero_after_reaction`
  - `candidate8_diffik_target_residual_zero_after_disp_m`
- These gates affect only the learned target residual. Candidate6/Candidate7
  base DiffIK remains active.
- The implementation avoids Python `bool(torch.all(...))` inside the GPU step
  loop; the residual gate multiplies by an active mask only when a gate is enabled.
- Added smoke-runner CLI/config/contract logging for the two new fields.

### Static / Launch Verification

- `python3 -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_tap10cm_ppo_smoke.py` passed.
- `python3 roarm_rl/train_cube_tap10cm_ppo_smoke.py --help` showed the new
  `candidate8_diffik_target_residual_zero_after_*` args.
- `git diff --check` passed.
- Direct system Python launch failed before physics with
  `ModuleNotFoundError: No module named 'gymnasium'`.
- Valid local runtime used `conda run -n isaaclab python -m roarm_rl.train_cube_tap10cm_ppo_smoke ...`.

### Candidate10a - Displacement Gate 0.006m + Candidate9b Reward-Safe

File:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate10a_dispgate006_rewardsafe_zeroaftercontact_xy3cm_l1_40k_seed1013_summary.out`

Contract:

- Same xy +/-3cm, seed1013 L1 (`num_envs=32`, `num_steps_per_env=64`,
  `max_iterations=20`).
- Candidate9b reward-safe settings:
  `tap_transient_disp_reward_scale=0.0`, `tap_overshoot_penalty_scale=120.0`,
  `action_penalty_scale=0.05`.
- Residual scales: forward `0.0m`, lateral `0.006m`, height `0.002m`.
- Gates: `candidate8_zero_after_contact=True`,
  `candidate8_zero_after_disp_m=0.006`,
  `candidate8_zero_after_reaction=False`.

Result:

- line 3 contract violations `0`.
- line 4 pre: `success_episode_rate=0.6772151898734177`,
  `overshoot_max=0.125`.
- line 7 post: `success_episode_rate=0.6946107784431138`,
  `overshoot_max=0.46875`, `reward_mean_per_step=-0.5920655317102752`.
- line 9 base-relative: `success_episode_delta=0.01739558856969614`,
  `overshoot_delta=0.34375`, `target_residual_abs_max_max=0.0008083037100732327`,
  `l1_health_pass=False`, `l2_scale_candidate=False`.

Interpretation:

- Displacement-gating at the nominal target displacement did not solve the
  overshoot problem. It made the result worse than Candidate9b.
- Candidate9b remains the best partial branch, but it still is not scale-ready.
- The simple-gate path is now weak evidence: post-contact gate failed, half-scale
  failed, and displacement gate failed.

## D218 Decision

- Do not run Candidate10a or Candidate9b to L2/large PPO as a promotion path.
- Do not continue naive residual-scale grids or simple post-contact/displacement
  gate stacking.
- The next useful pass route has to be state-aware:
  - either shrink the actual policy action space from the inherited 6D scaffold
    to the 3 used target-residual axes;
  - or add pose-binned success/overshoot evidence and enable/condition residual
    only where same-run base is weak under xy +/-3cm randomization.
- Dataset, action-teacher, VLA, large PPO, and RoArm deployment remain blocked.

## D219 Action-Space Correction Patch - Static Only

User correction:

- The agreed core change was not "add another gate" but "make the policy action
  space actually 3D target residual axes".
- Candidate8/9/10 had still run through the inherited 6D policy action scaffold:
  only the first three outputs were interpreted as forward/lateral/height target
  residuals, then gates/scale variants were stacked on top.
- That is not the same as a real 3D policy action space.

Implemented correction:

- `roarm_rl/train_cube_tap10cm_ppo_smoke.py` now sets `cfg.action_space=3` when
  `rl_action_mode=candidate8_diffik_target_residual`.
- The smoke summary contract now logs `policy_action_space=3`.
- The smoke runner no longer exposes post-contact/reaction/displacement gate CLI
  args.
- `roarm_rl/roarm_cube_push_env.py` rejects
  `candidate8_diffik_target_residual` unless `cfg.action_space == 3`.
- The env step rejects any non-3D action tensor in this mode.
- Env-level guard still rejects post-contact, reaction, or displacement gates if
  they are manually enabled on this clean 3D branch; do not add a
  Candidate11b-style gate/scale workaround.
- Internal teacher joint buffers now use `self._robot.num_joints` instead of
  `self.cfg.action_space`, so reducing policy action space to 3 does not break
  reset/teacher 6D joint-state storage.

Static verification:

- `python3 -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_tap10cm_ppo_smoke.py` passed.
- `git diff --check` passed.
- No GPU/IsaacLab runtime was run for this correction.

Next required gate:

- Run exactly one corrected zero-action 3D target-residual preflight against a
  same-seed Candidate7/base compare.
- Required pass: zero action in corrected 3D mode must equal Candidate7/base on
  the key metrics before PPO. If not equal, treat as implementation bug and stop.
- Only after that preflight passes may one L1 PPO screen be considered.
- L2/large PPO, dataset generation, action-teacher claims, VLA, and RoArm remain
  blocked.

## D220 Corrected 3D Zero-Action Preflight Runtime

Runtime scope:

- User approved the next runtime: corrected 3D zero-action preflight versus
  same-seed Candidate7/base.
- Ran exactly one new local RTX4090/cuda:0 preflight:
  `conda run -n isaaclab python -m roarm_rl.train_cube_tap10cm_ppo_smoke ...`
  with `rl_action_mode=candidate8_diffik_target_residual`,
  `num_envs=32`, `seed=1011`, `max_iterations=0`, eval steps `580`, xy +/-3cm
  randomization, current-pose cube reference, previous-target base, near-face
  target path, and clean 3D action-space contract.
- No PPO training, checkpoint load, dataset, VLA, action-teacher, RoArm, SSH/B200,
  or Track A work was run.

New output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate8_3daction_zero_preflight_seed1011_summary.out`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate8_3daction_zero_preflight_seed1011_summary.json`

Corrected 3D summary:

- line 1: `max_iterations=0`, `num_envs=32`, `seed=1011`, `device=cuda:0`.
- line 3: `rl_action_mode=candidate8_diffik_target_residual`,
  `policy_action_space=3`, xy +/-3cm randomization, current-pose cube reference,
  clean target-residual scales, `violations=0`.
- line 4: `success_episode_rate=0.7422360248447205`,
  `tap_success_max=0.28125`, `success_event_count=239.0`,
  `success_event_rate_per_env=1.0`, `contact_seen_max=0.28125`,
  `reaction_seen_max=0.28125`, `overshoot_max=0.125`,
  `reward_mean_per_step=0.10634467779936727`,
  `face_gap_final_m=-0.013820353895425797`,
  `tcp_dist_min_m=0.0825260579586029`, `ik_reset_rate_min=1.0`,
  `ik_reset_err_mm_max=1.2306809425354004`,
  `candidate6_active_rate_max=1.0`,
  `candidate6_numeric_ok_rate_min=1.0`,
  `candidate6_hold_success_rate_max=0.0`.
- line 8: `preflight_pass=True`, `bridge_preflight_pass=True`,
  `training_smoke_pass=None`, `policy_task_pass=None`.

Same-seed Candidate7/base comparison:

- Compared against existing
  `cube10cm_tap_rl_candidate7_currentpose_xy3cm_same_seed1011_base_compare_summary.out`
  line 4.
- JSON comparison produced exact `0.0` diffs for:
  `tap_success_max`, `success_event_count`, `success_event_rate_per_env`,
  `success_episode_rate`, `tap_contact_seen_max`, `reaction_seen_max`,
  `tap_overshoot_max`, `reward_mean_per_step`, `tap_contact_face_gap_m_final`,
  `tcp_cube_dist_m_min`, `ik_endpoint_reset_rate_min`, `ik_reset_err_mm_max`,
  `candidate6_diffik_active_rate_max`, `candidate6_diffik_numeric_ok_rate_min`,
  `candidate6_diffik_hold_success_rate_max`, and
  `candidate8_diffik_target_residual_abs_max_max`.

Verdict:

- `CORRECTED_3D_TARGET_RESIDUAL_ZERO_EQUALS_BASE_PREFLIGHT_PASS`.
- The D219 implementation gate is cleared: corrected 3D policy action-space zero
  action equals Candidate7/base for the same seed.
- Next valid runtime, only with explicit approval, is one clean corrected 3D L1
  health screen. Do not add gates or half-scale variants.
- L2/large PPO, dataset generation, VLA, action-teacher claims, and RoArm
  deployment remain blocked.

## D221 Corrected 3D L1 Health Screen

Runtime scope:

- User explicitly approved one clean corrected 3D L1 health screen after the D220
  zero-action preflight passed.
- Ran exactly one local RTX4090/cuda:0 PPO screen:
  `conda run -n isaaclab python -m roarm_rl.train_cube_tap10cm_ppo_smoke ...`
  with `rl_action_mode=candidate8_diffik_target_residual`,
  `policy_action_space=3`, xy +/-3cm randomization, seed1012,
  `num_envs=32`, `num_steps_per_env=64`, `max_iterations=20`,
  `ppo_init_noise_std=0.2`, current-pose cube reference, previous-target base,
  near-face target path, and no gates.
- No L2/large PPO, dataset generation, VLA, action-teacher, RoArm, SSH/B200, or
  Track A work was run.

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate8_3daction_xy3cm_l1_40k_seed1012_summary.out`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate8_3daction_xy3cm_l1_40k_seed1012_summary.json`

Result:

- line 3: `policy_action_space=3`, xy +/-3cm, no gate fields,
  contract violations `0`.
- line 4 pre: `success_episode_rate=0.7350993377483444`,
  `overshoot_max=0.0625`, `reward_mean_per_step=0.08071071922940659`,
  reset/numeric bridge OK.
- line 7 post: `success_episode_rate=0.7027777777777777`,
  `overshoot_max=0.4375`, `reward_mean_per_step=0.07069900840007026`,
  `candidate8_target_residual_abs_max_max=0.0028551355935633183`,
  forward/lateral/height residual max
  `0.0009385579032823443 / 0.0028551355935633183 / 0.0009315494680777192`.
- line 8: `training_smoke_pass=False`, `policy_task_pass=False`,
  `large_dataset_rl_roarm_unblocked=NO`, `action_teacher_dataset=NO`.
- line 9: `success_episode_delta=-0.03232155997056663`,
  `overshoot_delta=0.375`, `signal_seen=True`, `l1_health_pass=False`,
  `l2_scale_candidate=False`.

Verdict:

- `CORRECTED_3D_TARGET_RESIDUAL_L1_HEALTH_FAIL_NO_L2`.
- The corrected 3D action space is implemented and zero=base passed, but the
  learned 3D residual still worsens overshoot and reduces success under xy +/-3cm.
- Do not run L2/large PPO.
- Do not respond by adding gates, half-scale grids, or displacement gates as a
  reflex.
- Next work must diagnose why the true 3D residual still learns overshoot before
  any new scale-up runtime.
