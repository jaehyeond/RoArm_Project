# D321 Goal-Conditioned Primitive Action Space Draft

Date: 2026-07-08 KST

This is a design-only document. D321 does not train PPO/RL. The purpose is to
define the next action space where zero-action can no longer solve the task by
construction.

## Objective

Move from a fixed +x zero-residual script generator to a goal-conditioned
primitive generator:

```text
condition = push direction + target displacement band
policy output = bounded primitive parameters
controller = DiffIK/TCP execution + hybrid stop + physicality gate
```

The 10cm cube remains a fixture for validating the data factory, not the final
industrial task.

## Condition Inputs

The condition should be explicit and logged per episode:

| Field | Values | Meaning |
|---|---|---|
| `push_direction_id` | `{0,1,2,3}` | Direction class |
| `push_direction_xy` | `{+x,-x,+y,-y}` | Object/world-frame push direction for cube fixture |
| `target_disp_band_id` | `{0,1,2}` | Target displacement band |
| `target_disp_band_m` | `{[0.003,0.007], [0.007,0.015], [0.015,0.050]}` | Desired accepted displacement range |
| `friction_bin_id` | `{low, mid}` first | D321 producer bins before upper-bin learning |

The first training curriculum should start with `+x` only, because D319/D320
showed +x is currently stable and non-+x zero-action is not.

## Learnable Primitive Parameters

Candidate policy output should be low-dimensional and bounded:

| Parameter | Draft range | Reason |
|---|---:|---|
| `approach_offset_along_m` | `[-0.020, 0.020]` | Adjust pre-contact distance along push axis |
| `approach_offset_lateral_m` | `[-0.020, 0.020]` | Compensate face/edge contact geometry |
| `push_depth_m` | `[0.000, 0.080]` | Main displacement-control parameter |
| `stop_margin_m` | `[0.000, 0.020]` | Early/late handoff to hybrid stop |
| `height_offset_m` | `[-0.006, 0.006]` | Small vertical contact correction |

`push_depth_m` and `stop_margin_m` are the first priority. `height_offset_m`
should remain frozen unless contact failures show vertical sensitivity.

## Zero-Action Baseline

Every evaluation must include zero-action for the same condition. In this
action space, zero-action is a single fixed primitive parameter vector under the
requested condition, not a policy. A learned policy only contributes if it beats
the zero-action condition-matched baseline.

## Reward Draft

The reward must be aligned with the post-hoc label filter and physicality gate:

| Term | Direction |
|---|---|
| `strict_useful` | Positive reward for contact + reaction + displacement >=1mm + no overshoot |
| `target_band_match` | Positive plateau inside requested target displacement band |
| `band_distance` | Smooth penalty outside the target band |
| `overshoot` | Explicit penalty for displacement beyond accepted bound |
| `solver_outlier` | Terminal rejection / large penalty when max XY >=300mm |
| `control_effort` | Small regularizer on parameter magnitude |

Do not make transient displacement a large monotonic reward. D316/D317 showed
that this creates cliff-edge behavior.

## Curriculum

| Stage | Scope | Entry condition |
|---|---|---|
| 1 | `+x`, low/mid friction | D321 producer bins pass >=90% with label filter |
| 2 | `-x` and `-y` | D320 direction probe showed partial feasibility |
| 3 | `+y` | D320 direction probe weakest row; train last |
| 4 | upper friction | Only after direction stability; upper contains mixed physical failures and solver outliers |

Upper-bin rows must keep the 300mm physicality gate. The upper bin is an RL
contribution target only after solver outliers are excluded.

## Evaluation Protocol

Use a fresh-process D290-style probe with matched train/eval contract:

```text
fresh32 x 4 directions x 2 friction bins x target bands
```

Required report columns:

- condition fields;
- zero-action baseline metrics;
- policy metrics;
- contact/reaction/useful;
- overshoot;
- low-motion <1mm;
- solver_outlier >=300mm;
- displacement distribution;
- final/current contact proxy.

Promotion is not deployment promotion. For the data factory, the generator
criterion is label-filter pass rate and diversity. For policy contribution, the
policy must beat zero-action under the same condition.

## Non-Goals

- No raw joint-delta fallback.
- No controller hand-condition patching as a substitute for learning.
- No upper-bin data production until physical failures and solver outliers are
  separated.
- No VLA/RoArm deployment claim from this design.
