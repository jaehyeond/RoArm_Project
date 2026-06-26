# 2026-06-20 Cube10cm Top-View Contact Proxy Correction D267-D270

## Scope

- Branch: professor 10cm / 0.72kg cube top-view visual trajectory dataset path.
- No PPO learning, no rendering, no RoArm deployment, no RunPod, no B200/SSH.
- Goal: verify whether D264-D266 contact failures were real behavior failures or
  contact-proxy false negatives.

## Code Review Finding

The visual top-view renderer records labels through `inner_env._tap_terms()`.
The renderer uses `train_cube_tap10cm_ppo_smoke._apply_candidate6_contract()`,
which sets:

- `tap_contact_proxy_mode = "link5_collision_aabb"`

Therefore the D247/D256 visual labels are not raw TCP-point contact labels.
D264-D266 used `_push_terms().tcp_cube_dist < 0.055m`, so those probes were
measuring a stricter and different gate.

## Probe Fixes

Updated:

- `sim_scripts/cube10cm_top_view_d256_state_sequence_probe.py`
- `sim_scripts/cube10cm_top_view_d256_action_replay_probe.py`
- `sim_scripts/cube10cm_top_view_teacher_rollout_probe.py`
- `roarm_rl/train_cube_push_ppo.py`

The probes now use `_tap_terms().tap_contact_proxy` for tap10cm contact and log
the older TCP threshold separately as `tcp_threshold_contact_rate`.

`train_cube_push_ppo.py` now exposes:

- `--tap_contact_proxy_mode {tcp_point,link5_collision_aabb}`

For `env_kind=tap10cm`, the corrected default is `link5_collision_aabb`.

## Results

### D267 Recorded-State Sequence

Output:

- `d256_state_sequence_probe_d267_aabb/tap10cm/d256_state_sequence_summary_d267_aabb.json`
- `d256_state_sequence_probe_d267_tcppoint/tap10cm/d256_state_sequence_summary_d267_tcppoint.json`

Same D256 recorded states:

| proxy | contact_rate | tap_useful_rate | tcp_threshold_contact_rate | min TCP-cube dist min |
|---|---:|---:|---:|---:|
| `link5_collision_aabb` | `1.0` | `1.0` | `0.0` | `0.06270913034677505` |
| `tcp_point` | `0.0` | `0.0` | `0.0` | `0.06270913034677505` |

Interpretation:

- D266 was a TCP-only false negative for the visual dataset contract.
- Recorded D256 states are compatible with the AABB contact proxy.

### D268 Direct D256 Action Replay

Output:

- `d256_action_replay_probe_d268_aabb_hold3/tap10cm/d256_action_replay_summary_d268_aabb_hold3.json`

Result:

- `tap_contact_proxy_mode`: `link5_collision_aabb`
- contact rate: `1.0`
- tap useful rate: `1.0`
- TCP-threshold contact rate: `0.0`
- max disp along mean/max:
  `0.006767723709344864` / `0.017127275466918945`
- min TCP-cube distance mean/min/max:
  `0.07518836855888367` / `0.06179572641849518` /
  `0.09923214465379715`

Interpretation:

- D256 action replay is not failing the dataset contact proxy.
- It only fails the raw TCP-distance threshold, which is not the dataset label
  contract.

### D269 D257 Teacher-Only

Output:

- `teacher_rollout_probe_d269_aabb_d256_initial/tap10cm/teacher_rollout_probe_summary_d269_aabb_d256_initial.json`

Result:

- reset: D256 initial pose
- `tap_contact_proxy_mode`: `link5_collision_aabb`
- contact rate: `0.71875`
- tap useful rate: `0.71875`
- TCP-threshold contact rate: `0.0`
- max disp along mean/max:
  `0.0014523034915328026` / `0.01252603530883789`
- raw delta clip exceed rate: `0.20877155172413794`
- action cap rate: `0.13050466954022988`

Interpretation:

- Teacher-only is no longer contact-zero under the correct proxy.
- However, teacher-only displacement is much weaker than direct D256 action
  replay, so this is not learned policy success and not RoArm readiness.
- Contact/useful can appear from initial AABB contact context, so promotion
  gates must include displacement, overshoot, action saturation, and TensorBoard
  reward/task scalars.

### D270 Offline D256 Contact Audit

Output:

- `d256_contact_contract_audit_d270/d256_contact_contract_audit_d270.json`

All D256 train-clean teacher rows:

- rows: `142978`
- `tap_contact_proxy` rate: `0.8646784820042245`
- `tap_contact_seen` rate: `0.9137559624557624`
- `tap_reaction_seen` rate: `0.9137559624557624`
- `tap_overshoot_seen` rate: `0.0`
- `tcp_sphere_055` rate: `0.0`
- `tcp_point_face_band` rate: `0.0`

Interpretation:

- Raw TCP-point contact would mark the entire D256 teacher table as no-contact.
- That is incompatible with the visual dataset contract.
- AABB/tool-surface contact is the correct proxy for this branch unless a new
  `tool_surface_union` proxy is explicitly implemented and revalidated.

## Corrected Tiny PPO Candidate

Prepared but not run:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/ppo_data_prior_smoke_command_d270_corrected_tap10cm_aabb.txt`

Required characteristics:

- `env_kind=tap10cm`
- `tap_contact_proxy_mode=link5_collision_aabb`
- fixed `+x` push direction
- `bc_teacher_feature_target_mode=env_target`
- `PYTHONPATH=.`
- tiny smoke only, followed by TensorBoard scalar gate

## Verdict

`D270_TCP_CONTACT_GATE_FALSE_NEGATIVE_AABB_CONTRACT_RESTORED_NO_LONG_PPO`

Do not run long PPO. The next valid runtime, only with explicit approval, is a
tiny corrected tap10cm+AABB PPO smoke, then TensorBoard scalar gate. Do not claim
learned policy or RoArm readiness until teacher-off eval passes.

## Verification

- `python3 -m py_compile` passed for modified scripts.
- `git diff --check` passed.
- No Python/Isaac/PPO probe process remained after runs.
- GPU returned to observed baseline around `2509MiB` used / `13436MiB` free.
