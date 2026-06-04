# Session: 2026-06-04 Cube10cm DiffIK Teacher Gate Prep

## Scope

- Professor cube push/tap branch only.
- No B200/SSH/reconnect/pull or `.ssh` copy.
- No Track A grasp/dataset/training/runtime.
- One local GPU IsaacLab tiny 128 runtime was run after explicit approval.
- No 1024/10k scale-up, dataset generation, PPO scale-up, VLA training, or
  Track A runtime was run.

## Term Definitions Used In This Session

- `GPU escalated`: Codex default sandbox hides `/dev/nvidia*`; any IsaacLab/PhysX
  command that needs the local GPU must be run with
  `sandbox_permissions=require_escalated`.
- `tiny 128 gate`: a small 128-environment smoke gate, not training and not a
  dataset run. It checks whether the configured object/controller/trajectory can
  produce stable controlled pushes before any larger 1024/10240 data collection.
- `teacher`: here means the scripted IsaacLab `DifferentialIKController` path
  that produces TCP targets and joint targets from live PhysX Jacobians.
- `teacher-off`: in the learned-policy actor/eval branch, this means disabling
  teacher blending. For this DiffIK probe, the more precise term is
  `DiffIK teacher gate`, because the controller itself is the diagnostic teacher
  and the script already sets training/dataset generation false.

## Verified Context

- Current 3cm env constants remain fixed in `roarm_rl/roarm_cube_push_env.py`:
  `CUBE_SIZE_M=0.030`, mass `0.020kg`, default target displacement `0.040m`,
  strict success displacement `0.030m`.
- Existing 2026-06-04 hierarchical sharded-10k audit remains 3cm evidence only:
  it logs `cube_size_m=0.030000`, `cube_mass_kg=0.020000`, density
  `740.741kg/m^3`, and displacement-only 1/5/10/20/30mm columns.
- Professor's current request is interpreted as a separate 10cm cube/object
  diagnostic, not as Track A grasp and not as a requirement to move the object
  one full object length.

## Code Prep

- Updated `sim_scripts/cube3cm_push_diffik_probe.py` without changing its default
  3cm behavior:
  - added `--cube_size_m`, `--cube_mass_kg`, `--cube_push_target_disp_m`,
    `--cube_success_disp_m`, and `--gate_disp_m`;
  - sets the IsaacLab cuboid size, mass, and z-center from these args before
    `gym.make`;
  - logs cube size, mass, density, object-size reference, target displacement,
    success displacement, and gate displacement;
  - computes the TCP precontact/through points from the configured object size,
    not the old imported 3cm constant;
  - records `disp_over_object_size`, `disp_ge_gate_rate`, and
    1/5/10/20/30mm threshold rates.

## Local Verification

- `python -m py_compile sim_scripts/cube3cm_push_diffik_probe.py`: PASS.
- `python sim_scripts/cube3cm_push_diffik_probe.py --help`: PASS and showed the
  new size/mass/gate args.
- `git diff --check`: PASS.

## Approved Tiny 128 Runtime

After explicit approval, ran:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u \
  sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 128 \
  --episodes 1 \
  --seed 930 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010 \
  --out_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_gate128_seed930.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_gate128_seed930_summary.json
```

Result: FAIL as a 1cm gate.

- Summary lines 11-25: controlled `0.1875`, `disp_along_push_mean_m=-0.0001524518520454876`,
  `disp_ge_gate_rate=0.0`.
- Summary lines 27-34: only the 1mm displacement tier reached `0.1875`; 5/10/20/30mm all `0.0`.
- Summary lines 41-52: final TCP target error mean `0.15134897292591631m`,
  low-motion `1.0`, max cube speed mean `0.0004937104492910294m/s`,
  min TCP-cube distance mean `0.13646007765782997m`.
- Summary lines 56 and 63: posewrite during rollout `0` and rollout object
  posewrite false, so the failure is not hidden object pose-writing.
- CSV had 128 rows plus header. Direction analysis showed every direction had
  `ge10mm=0.0`, `low=1.0`, and `diffik_clip_rate=1.0`; mean final TCP target
  error was roughly `0.14-0.17m` by direction.

Interpretation:

- The 10cm/0.72kg object did not fail because 1cm is too strict. It failed
  earlier: the default v1 trajectory/controller did not reliably reach/contact
  the larger object, and the object barely moved.
- Do not scale this to 1024/10k data, dataset generation, PPO/RL, VLA, or Track A.
- The next valid work is small geometry/control diagnosis: contact height,
  precontact/through distance, horizon, and DiffIK joint-step clipping for a
  10cm object, then another tiny gate only after explicit approval.

## Post-Fail Code Review Before Further Edits

After the failed 128 gate, reviewed the relevant code before making another
change.

Findings:

1. Reset z was still 3cm-specific. `sim_scripts/cube3cm_push_diffik_probe.py`
   set `env_cfg.sponge.init_state.pos` from the requested 10cm size, but
   `roarm_rl/roarm_cube_push_env.py` reset used the module constant
   `CUBE_CENTER_Z`, derived from `CUBE_SIZE_M=0.030`. For a 10cm object this is
   a 35mm z-center error.
2. The first 10cm gate mixed geometry validation with random x/y and random push
   direction. That is useful for robustness later, but it is too broad for
   debugging whether one easy, centered push works.
3. The current probe target height mode is a top-margin path. For the professor's
   "push the front face" mental model, the next diagnostic should be able to use
   side-face center height.
4. The failed run's `diffik_clip_rate_mean=1.0` and final TCP target error mean
   `0.15134897292591631m` should be interpreted as a reach/path issue before
   treating mass/friction as the limiting factor.

## Static Patch After Review

Patched without running another IsaacLab/GPU runtime:

- `roarm_rl/roarm_cube_push_env.py`
  - added config fields `cube_size_x_m`, `cube_size_y_m`, `cube_size_z_m`;
  - reset now uses `TABLE_Z + cube_size_z_m / 2` instead of the 3cm
    `CUBE_CENTER_Z` constant;
  - local precontact/teacher target helpers now compute half extents from the
    configured object size;
  - added optional fixed push direction config.
- `sim_scripts/cube3cm_push_diffik_probe.py`
  - passes the configured cube size into the env config, not just the spawn
    object;
  - added `--fixed_cube_x_m`, `--fixed_cube_y_m`, and `--fixed_push_dir`;
  - added `--tcp_height_mode top_margin|side_center`;
  - logs `table_z_m`, `cube_center_z_m`, and per-row `cube_z0_m` so the reset
    height is auditable from logs.

Static verification:

- `python -m py_compile roarm_rl/roarm_cube_push_env.py sim_scripts/cube3cm_push_diffik_probe.py`: PASS.
- `python sim_scripts/cube3cm_push_diffik_probe.py --help`: PASS and showed the
  fixed-position, fixed-direction, and TCP-height-mode args.
- `git diff --check`: PASS.
- Direct `conda run -n isaaclab python -c "from roarm_rl..."` import was not a
  valid check because IsaacLab/Omniverse modules require `SimulationApp` /
  `AppLauncher` before import. No runtime claim is made from that failed import.

## Next Candidate Shape

Do not run until explicit approval with GPU escalation. Start smaller than 128:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u \
  sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 931 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --fixed_cube_x_m 0.300 \
  --fixed_cube_y_m 0.000 \
  --fixed_push_dir 1 0 \
  --tcp_height_mode side_center \
  --precontact_clearance_m 0.010 \
  --push_through_m 0.010 \
  --approach_steps 220 \
  --push_steps 90 \
  --post_steps 40 \
  --max_diffik_joint_step_rad 0.035 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010
```

Interpretation:

- This is not dataset generation, not PPO/RL, not VLA, and not Track A.
- First check `cube_z0_m`, `fixed_push_dir`, `tcp_height_mode`, final TCP target
  error, min TCP-cube distance, and only then the 1cm displacement gate.
- If this fixed easy case fails, do not add randomization; debug geometry/reach
  with trace/video.

## Previous Gate Candidate

This was the initial candidate and has now failed:

```bash
python sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 128 \
  --episodes 1 \
  --seed 930 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010
```

Expected interpretation:

- Primary metric is not 10cm displacement. It is whether a 10cm/0.72kg object can
  be pushed about 1cm in a controlled, no-impact, low-motion-safe way.
- Report `disp_ge_gate_rate`, 1/5/10/20/30mm rates, `disp/object_size`,
  controlled, no-impact, low-motion, TCP target error, and posewrite count.
- The failed result blocks immediate 1024/10240 scale-up, dataset generation,
  PPO scale-up, VLA training, and Track A runtime.
