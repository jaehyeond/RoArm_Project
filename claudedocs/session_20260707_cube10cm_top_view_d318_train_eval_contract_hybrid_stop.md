# D318 train/eval contract diagnosis and hybrid stop result

Date: 2026-07-07 KST

Scope: professor 10cm / 0.72kg cube top-view tap/push branch only. Local host GPU
only. No B200/SSH/pull/RoArm/VLA/high-friction training/raw joint-delta revival.

Verdict:

`D318_HYBRID_STOP_LONG_HORIZON_IMPROVES_ZERO_ACTION_MATCH_NO_PROMOTION`

## Current-state checks

Followed the repo-grounded current-state protocol before runtime work:

- `CLAUDE.md`
- `START_HERE.md`
- `claudedocs/DECISIONS.md`
- `claudedocs/EXPERIMENT_LEDGER.md`
- D317 session doc:
  `claudedocs/session_20260707_cube10cm_top_view_d317_reward_v2_domain_randomized_ppo.md`
- D290 probe:
  `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`
- Runtime env:
  `roarm_rl/roarm_cube_push_env.py`
- PPO train harness:
  `roarm_rl/train_cube_push_ppo.py`

Initial git status was dirty only after D318 edits/results were created; no
pre-existing work was reverted.

## Contract diff

| Axis | D317 PPO collection path | D290 promotion probe path | D318 finding |
|---|---|---|---|
| Episode length | Tap env default is `1.2s` (`roarm_rl/roarm_cube_push_env.py:72`). | Probe default is `6.0s` (`sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py:149`). | Short collection can hide late over-push. |
| Rollout horizon | D317 used `num_steps_per_env=64`; CLI wiring is at `roarm_rl/train_cube_push_ppo.py:152,727-729`. | D290 ran `580` env steps. | This was the main score mismatch. |
| Termination/stop | D317 reward-v2 kept useful/overshoot termination and tap stop off. | D290 also disabled useful/overshoot termination and tap stop in the promotion commands. | Policy had to keep behaving after useful tap. |
| Sampling | PPO collection uses stochastic actor updates. | D290 uses deterministic inference policy (`sim_scripts/...d290...py:448,665`). | Sampling is secondary; horizon dominates. |
| Reset distribution | Training path had D256 reset inactive. | D290 used `reset_pose_source=env_hook`, D256 random frame-0 reset (`sim_scripts/...d290...py:143,359-362`). | D256 reset makes the long-horizon result harsher. |
| Friction | D317 training randomized static friction `[0.7,1.6]`. | Promotion low-friction row fixed `0.8/0.6`. | Training distribution alone did not solve stop timing. |
| Action mode | `candidate8_diffik_target_residual`, 3D residual. | Same action mode. | The train/eval action mode was not the mismatch. |

## Code changes

- Added default-off `candidate8_hybrid_stop_after_useful` to env config
  (`roarm_rl/roarm_cube_push_env.py:155-161`).
- In candidate8 mode, when strict useful is reached
  (`contact_seen & reaction_seen & max_xy >= 1mm & no overshoot`), latch the
  current joint positions as hold targets and stop applying further residual
  targets (`roarm_rl/roarm_cube_push_env.py:1240-1266`).
- Added train-harness eval-only support for checkpoint evaluation without PPO
  updates (`roarm_rl/train_cube_push_ppo.py:91-99`, `:899-1021`).
- Added D290 CLI support and summary fields for candidate8 hybrid stop and
  low-motion / `>=1mm` displacement metrics
  (`sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py:194-197`,
  `:1119-1142`, `:1215-1218`).

This is a structural stop/termination repair for the observed train/eval
contract failure, not reward tuning. It is also not a learned-policy promotion.

## Controlled diagnosis

Same D317 `seed31713 model_225.pt`, fixed friction `0.8/0.6`, deterministic
policy, 580 steps:

| Path | Hybrid stop | Reset | Strict useful | Overshoot | Low-motion | Mean max XY |
|---|---:|---|---:|---:|---:|---:|
| Train harness eval-only | off | default train reset | `22/32` | `10/32` | `0/32` | `16.310mm` |
| Train harness eval-only | on | default train reset | `32/32` | `0/32` | `0/32` | `7.376mm` |
| D290 probe | off | D256 random env_hook | `8/32` | `24/32` | not logged in old summary | `27.073mm` |
| D290 probe | on | D256 random env_hook | `31/32` | `1/32` | `0/32` | summary path below |

D318 therefore confirms the D317 failure was not just a reward-scale issue.
The long-horizon no-stop/stillness requirement was a real contract mismatch.
D256 reset distribution worsened the failure, but the train harness itself also
overshot at 580 steps without hybrid stop.

Key summaries:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/contract_diag_d318/tap10cm/train_harness_seed31713_model225_fixed_friction_580_det/eval_only_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/contract_diag_d318/tap10cm/train_harness_seed31713_model225_fixed_friction_580_det_hybrid/eval_only_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/contract_diag_d318/tap10cm/d290_seed31713_model225_fixed_friction_580/closed_loop_recovery_summary_d318_d290_seed31713_model225_fixed_friction_580.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/contract_diag_d318/tap10cm/d290_seed31713_model225_fixed_friction_580_hybrid/closed_loop_recovery_summary_d318_d290_seed31713_model225_fixed_friction_580_hybrid.json`

## Hybrid PPO run

Ran one seed as requested:

- seed: `31813`
- iterations: `300`
- envs: `64`
- action mode: `candidate8_diffik_target_residual`
- hybrid stop: on
- reward: D317 reward-v2 values unchanged
- friction randomization: static `[0.7,1.6]`, dynamic ratio `0.8`

Collection-final was not a promotion signal:

| Metric | Result |
|---|---:|
| strict useful | `36/64` |
| overshoot | `18/64` |
| mean/max XY | `16.671/45.304mm` |

Training output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/`

## Checkpoint sweep

D290 fresh32, 580-step, D256 random env_hook, hybrid stop enabled, same seed
`31813`, evaluated checkpoints `0,25,...,275,299` on low-friction and nominal
rows.

Every checkpoint produced effectively the same result:

| Row | Strict useful | Overshoot | Low-motion | Mean max XY | Hybrid latch |
|---|---:|---:|---:|---:|---:|
| friction `0.8/0.6` | `30/32` | `2/32` | `0/32` | `9.522mm` | `32/32` |
| nominal `1.5/1.2` | `32/32` | `0/32` | `0/32` | about `1.35mm` | `32/32` |

This fails Promotion Criteria V1 because low-friction overshoot is `6.25%`,
above the `<=5%` threshold.

More importantly, a zero-action D290 diagnostic matched the policy sweep:

| Row | Actor source | Strict useful | Overshoot | Low-motion | Mean max XY |
|---|---|---:|---:|---:|---:|
| friction `0.8/0.6` | zero action | `30/32` | `2/32` | `0/32` | `9.521mm` |
| nominal `1.5/1.2` | zero action | `32/32` | `0/32` | `0/32` | `1.336mm` |

Therefore D318 does not prove learned policy contribution. The improved D290
long-horizon behavior is dominated by base candidate8 DiffIK target plus hybrid
stop, not by the PPO residual policy.

Checkpoint sweep root:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d318_hybrid_checkpoint_eval/tap10cm/`

## Interpretation

D318 fixed an evaluation contract bug, but it also exposed a new research
constraint:

- Hybrid stop is useful as a long-horizon safety/termination layer.
- It can make low-friction long-horizon evaluation nearly pass.
- It currently collapses policy credit: all checkpoints and zero-action control
  produce the same D290 metrics.

That means the next valid learning question is not "train longer" and not
"retune reward again." The next question is how to make the learned action affect
the part that still matters, for example stop margin, approach/contact selection,
push direction, or bounded displacement target, while keeping D290 long-horizon
promotion criteria as the judge.

## Promotion criteria V1 verdict

| Criterion | D318 best observed |
|---|---:|
| low-friction strict useful `>=65%` | pass: `93.75%` |
| low-friction overshoot `<=5%` | fail: `6.25%` |
| low-friction low-motion `<=10%` | pass: `0%` |
| nominal strict useful `>=90%` | pass: `100%` |
| reproducible across 3 seeds | not attempted; policy contribution unproven |

Final: no `PROMOTION_CANDIDATE`, no learned-policy promotion.

## Next action

Do not run a longer PPO ladder from D318. The next failure-capable experiment
should isolate policy-controllable parameters:

1. Freeze the hybrid stop layer as the evaluator/safety layer, but make one of
   these parameters policy-controlled and logged: stop margin, target displacement
   band, push direction, or approach/contact offset.
2. Include a zero-action baseline in every promotion table.
3. Promote only if policy beats zero-action on low-friction overshoot/useful
   while preserving nominal performance.

