# t3y_workspace1 preregistration — workspace-wide parallel PhysX grasp sweep

Status: **PREREGISTERED / NOT RUN**
Case: `g0b_d420`
Canonical tag: `t3y_workspace1`
Date: 2026-08-11 KST

## 1. What this case asks

The question is not whether one fixed cylinder pose can be reached.  It is whether the
unchanged D29 x H50 upright cylinder can be grasped and lifted by contact physics at
some object positions and some reachable tool tilts.  The gripper must use the frozen
attempt3 split collision asset (64 convex-hull parts on `link5` and 64 on
`gripper_link`); kinematic object attachment is forbidden.

This is a failure-capable experiment.  A valid all-fail result would rule out the
sampled workspace/form cells under the fixed controller.  A success is evidence only
for its measured cell and controls, not for the whole workspace or real hardware.

이번 case의 신규 변수: `[object planar position, reachable grasp form/tilt stratum]`

Everything else is a deterministic control: upright object yaw 0, top-centre grasp
point (D419 unchanged), radial approach azimuth, fixed cylinder-material coefficients
and controller gains, p13-provided
theta-specific descend margin, and the finite q5 targets selected below.  Here
"friction" means only the cylinder material coefficients authored by p14
(`static=0.40`, `dynamic=0.30`); jaw/support materials and combine behavior remain
the full-SHA-pinned environment/USD defaults.  Effective pair friction is unmeasured
and not claimed.

## 2. Frozen inputs and non-negotiable guards

Before Isaac starts, the runner must hard-fail on any of the following:

1. `numpy != 1.26.0`, `psutil != 5.9.8`, `scipy != 1.15.3`, Isaac Sim
   distribution not `5.1.0.0`, Isaac Lab distribution not `2.3.0`, or Rerun SDK
   not `0.34.1`.
2. Any decision-bearing repo-local source differs from this full-SHA manifest:

   | source | SHA-256 |
   |---|---|
   | `sim_scripts/p10_g0b_t3t_cyld29h50_tilted_close_sweep_grasp_probe.py` | `63c6b2127d969e3291da6943eab6da1037034c154a8f21fe447519cbcb2f6cff` |
   | `sim_scripts/g0b_t3_attempt3_jaw_throat_occlusion_readonly_vertex_audit.py` | `bca4f898023f63f21d540483a169499760038c582ce3a7919d7622e77946e1c3` |
   | `sim_scripts/roarm_kinematics.py` | `af4cc2d3c124ba4a8a3a6899ab1c8d676e127199bcf687b8e31fee690a011a97` |
   | `roarm_rl/__init__.py` | `270819a1bba4aa43723ca2257cf7ed7da160e492dcb17607e71ee7cf1a08e940` |
   | `roarm_rl/roarm_stack_env.py` | `726a57f4be83276fda1bb5b3eaf07a17f56d5c16cdbc0441bc14fb5c794a697d` |
   | `roarm_rl/viz_debug.py` | `4b5f821ad43652f529dfaa2f92b2826d9cd4973635e34521cc2b3a93ab0193d0` |
   | `roarm_rl/rerun_contract.py` | `aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e` |

   The manifest is checked before p10 planning/import and again after physics and
   immediately before results serialization.
3. Any local USD layer in the recursive attempt3 sublayer/reference/payload
   composition differs from the exact manifest below, or the recursively resolved
   local layer set is not exactly these five:

   | composed layer | SHA-256 |
   |---|---|
   | `roarm_m3.usd` | `a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff` |
   | `configuration/roarm_m3_base.usd` | `ea0ee8f258e935799cf927b8c67e871f935c09b3c9be4f971006937334a11841` |
   | `configuration/roarm_m3_physics.usd` | `043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503` |
   | `configuration/roarm_m3_robot.usd` | `2227536fcb8c9dae1aa9cc1cf422350fcf85e662eed97fe9ea48535c6b4aa65d` |
   | `configuration/roarm_m3_sensor.usd` | `3f44081f42b452bc5f9791a8df1c37e00ba5a6dc98a9e49e065c7acacdda0d0f` |

   The ignored USD files are hash-checked before planning, recursively resolved by
   `pxr.UsdUtils.ComputeAllDependencies` after Kit starts and before environment
   construction, and checked again after physics/final replay.  The sole admissible
   unresolved normalized identifier set is exactly `{OmniPBR.mdl}`, and only after
   all `t3y_workspace_preflight2_prereg.md` semantic gates pass: NVIDIA runtime
   built-in membership/configuration and resolver-version recording; no MDL in a
   real Sdf sublayer/reference/payload arc; exactly eight authored
   `info:mdl:sourceAsset` attributes on `Shader` prims in the pinned base layer; and
   an all-eight authoring-layer `UsdShade.Shader.GetSourceAsset("mdl")` cross-check.
   Any other
   MDL, path-qualified same basename, missing/unresolved USD, mixed unresolved set,
   or external composition arc remains fatal.  This is not a sixth USD layer, a
   blanket `.mdl` ignore, a search-path change, or a USD edit.
4. `t3x_bite81_results.json` is absent, is not
   full SHA-256
   `d1460c9d80e5f65f4ab9d85a7851b29876ef7ae0ca9e28d34bd93ddb91f0170a`, is not
   `physics_handoff.schema_version == 1`, is not source tag `t3x_bite81`, does not
   state `run_physx_regardless_of_bilateral == true`, or lacks exact controls for
   theta `{6,15,24,35,60,69}`.  The full p13 results SHA is recorded at runtime;
   no interpolation to an unmeasured theta is permitted.  The p13 producer must
   also have `run_valid == true`, `source_freeze.stable_at_end == true`, gates
   `X1_input_sha`, `X1_env_pins`, `X2_asset_identity_64_plus_64`,
   `X3_n10_theta29_35_regression`, and `X6_source_stable` all true, and
   `n10_regression.pass == true`.  These producer-validity fields are copied into
   p14 provenance.  p13 Rerun technical PASS is deliberately not a scientific
   handoff blocker because observability failure does not rewrite geometry data.
5. Runtime USD stage audit is not exactly 64 enabled `part_*` collision bodies for
   each jaw, enabled collision bodies are not parts-only, or either legacy collider
   is not present exactly once and disabled.  The pinned jaw extractor must also
   report `hull_ok=true` for all 64 parts of both jaws; raw-point fallback from a
   failed `scipy.spatial.ConvexHull` is forbidden.
6. Exact `link5` and `gripper_link` contact reporters are not armed with read-back
   threshold 0.0 on **every cloned environment body**, or either separate
   jaw-to-support force-only sensor fails to resolve one sensor body and one ground
   filter.  Canonical tensor shapes are exactly object force/contact
   `(1024,1,3,3)` and each jaw-ground force `(1024,1,1,3)`; force-only jaw sensors
   must leave `contact_pos_w` unallocated (`None`).
7. Any canonical output path already exists (G0, exit 3).  Existing evidence is
   never overwritten or renamed.

`bilateral_window_exists` is deliberately **not** an admission gate.  Static mesh
bite is not force closure.  A p13 unilateral-only or no-window row remains a labelled
negative-control PhysX trial, so p14 still answers the user's physics question.

## 3. Population and controls

### Workspace positions

Use an 8 x 8 cell-centre grid in every frozen p10 `SOURCE_REGIONS` rectangle:

| region | x (m) | y (m) |
|---|---:|---:|
| R1 | [0.150, 0.250] | [-0.220, -0.130] |
| R2 | [0.150, 0.250] | [+0.070, +0.200] |
| R3 | [0.330, 0.430] | [-0.220, -0.100] |
| R4 | [0.330, 0.430] | [+0.050, +0.200] |

This gives 256 systematic positions.  In each region, replace the nearest grid cell
with the corresponding exact frozen `seed0_S1..S4` point, keeping 256 total positions.

### Grasp forms

- `near_top_down`: theta 6 deg and 15 deg.  This is not theta=0; the prohibited fully
  vertical T2/T2b experiment is not rerun.
- `oblique`: theta 24 deg and 35 deg.
- `high_tilt`: theta 60 deg and 69 deg.  These are evaluated only where p14's own
  per-pose IK passes.  A p13 generic control does not grant position-independent IK.

For every position/form pair, p14 runs every distinct p13 interior q5 target (normally
25/50/75% of the exact window) plus one measured no-bite or least-bite q5 row derived
from p13's same-theta curve.  Rows are labelled `bilateral`,
`unilateral_negative_control`, `no_window`, `no_bite_negative_control`,
`least_bite_negative_control`, or (when that q5 has no finite depth solution)
`no_depth_solution_fallback_negative_control`; none is silently promoted to
bilateral or to a measured no-bite condition.

The descend target uses the exact same-theta, same-q5 p13 curve row's `delta_m` (no
theta/q5 interpolation).  If any q5 source row—including an interior target—has no
finite intersection depth, it uses only the finite exact-theta
`physics_handoff.controls[].grasp_surface_margin_m` fallback and always changes the
row label to `no_depth_solution_fallback_negative_control`.  p14 then solves approach,
descend, and lift IK independently for every workspace position and q5 control.
`physics_handoff.controls[].q_descend_deg` is expected to be null because the handoff
explicitly requires pose-specific IK.

No theta interpolation, random pose jitter, friction randomization, controller-gain
randomization, object-yaw randomization, side-midpoint grasp, or altered object
dimension is in scope.

## 4. Execution and measured signals

All IK-valid trials run in deterministic batches of 1024 environments on the local
GPU.  The environment is reused between batches, but every batch resets robot,
cylinder, sensor history, and velocities.  Cylinder material coefficients and
controller gains are authored once and remain fixed across resets; p14 does not
re-author jaw/support materials.  A short final batch is padded with inactive
duplicates and padding is excluded from every denominator.

Fixed schedule per trial:

| phase | steps | meaning |
|---|---:|---|
| settle | 120 | cylinder rests on support; positive sensor control |
| approach | 300 | home to approach IK target |
| descend | 500 | approach to theta/q5-specific descend target |
| close | 300 | close to exact p13 q5 control |
| hold | 120 | maintain closure |
| lift | 500 | world-vertical 25 mm TCP lift |

The inherited simulator timestep remains 1/200 s, but p14 fixes environment
`decimation=1`.  The doubled step counts preserve p11's physical phase durations
while making each recorded runner step exactly one PhysX step; the replay therefore
does not hide the first of two contact substeps behind the RL default decimation 2.

Physics metrics include per-filter normal-force vectors for support, fixed jaw and
moving jaw; averaged contact-point positions; pre-close jaw contact; both-jaw contact;
whole-close object drift; waypoint arrival error; q5 tracking error; object position,
orientation and tilt; raw centre rise; tilt-corrected lowest-point lift-off; and the
actual link5-to-support and gripper_link-to-support force vectors at every PhysX step.

This jaw-to-support measurement is reactive hardening, not speculative scope growth:
p13 directly observed minimum jaw-hull table clearances of `-7.471`, `-9.529`, and
`-9.148 mm` with `table_penetration_count` values `6198`, `8134`, and `7775` in the
relevant seed0_S3/S4/r=0.45 candidates.  The original object sensor cannot observe those
jaw-ground pairs.  Therefore p14 keeps the original object sensor and adds two exact,
force-only sensors: `link5 -> support` and `gripper_link -> support`.

`ContactSensorCfg.max_contact_data_count_per_prim` is fixed at 256.  In installed
Isaac Lab 2.3 this allocates `256 * num_envs * num_sensor_bodies` raw contact records,
so a 1024-env, one-body object sensor has capacity 262,144.  This is intentionally
larger than the 128 total jaw convex parts; p11's value 16 was not used as contact-point
authority.  The object sensor and each of the two jaw-ground sensors independently
receive this capacity.  Each raw contact-count peak is recorded every step, and any
one sensor reaching its allocated total invalidates the batch as saturated.  Raw
count tensor shapes are hard-gated as object `(1024,3)` and each jaw-ground sensor
`(1024,1)` before per-environment maxima are accumulated.

The settle positive control expects the cylinder's median support force to fall in
`m*g +/- 35%`, with at least 90% of active environments in band.  An invalid batch
cannot yield a scientific grasp success even if its object moved.

The canonical workspace verdict additionally requires **every** active population
batch to pass that positive control, no contact buffer to saturate, and every
feasible trial to have `measurement_valid == true`.  `measurement_valid` means the
force readout is trustworthy: it depends only on per-environment settle control,
batch positive control, and raw-buffer integrity.  A reliably measured jaw/support
force above 0.02 N is **not** a sensor failure; it is a valid physical task failure
that remains in the `n_valid` denominator, cannot be success, and is labelled
`JAW_SUPPORT_CONTACT_FAIL` when the measurement controls pass.

If even one trial/batch is measurement-invalid, the global verdict is
`MEASUREMENT_INVALID_PARTIAL_BATCHES`; valid strata remain diagnostic only and
neither a global success rate nor a global `NO_*` claim is permitted.  Every
reported rate uses `n_valid` as its denominator (`null` when `n_valid == 0`) and
reports `n_invalid` separately.  A pure regression smoke must prove that a
partial-invalid population cannot authorize a global workspace claim.  Primary
lift and bilateral-contact statistics use the stricter `task_clear =
measurement_valid & jaw_support_contact_pass` subset; all-measured and
support-collision statistics are reported separately so table-assisted rise cannot
be presented as a free-space grasp.

## 5. Preregistered gates and classifications

A trial is `PHYSICS_LIFT_SUCCESS` only if all are true:

1. all three per-pose IK waypoints pass p10's 3 mm position / 5 deg direction gates;
   and wrist roll q4 stays inside the V6 distribution clip `[-90,+90] deg` at
   approach, descend, and lift (p10's wider URDF `[-180,+180]` clamp alone is not
   admission);
2. its batch passes the support-force positive control;
3. descend TCP arrival error <= 3 mm, max arm-joint arrival error <= 5 deg,
   close q5 error <= 3 deg, and commanded lift produces >= 15 mm TCP rise;
4. no jaw normal force > 0.02 N occurs before the close phase;
5. the maximum actual `link5 -> support` and `gripper_link -> support` force over
   the entire executed episode is <= 0.02 N.  Any exceedance is a reliably measured
   task failure: it forbids success but remains `measurement_valid=true` when the
   positive controls pass, stays in the valid denominator, and emits
   `JAW_SUPPORT_CONTACT_FAIL`.  If a positive-control/buffer failure co-occurs,
   primary label `MEASUREMENT_INVALID` has priority while a reason flag still records
   the observed jaw/support contact; this blocks table-assisted lift false positives
   without making the whole broad sweep self-invalidating;
6. both jaws exceed 0.01 N **in the same PhysX step** during close and again
   during lift.  The authority scalar for each phase is
   `max_t(min(F_fixed(t), F_moving(t))) > 0.01 N`; AND-ing the two jaws'
   independent phase maxima is forbidden because staggered one-jaw contacts
   would become a false two-jaw grasp.  The independent fixed/moving maxima are
   diagnostic values only.  A pure regression smoke must prove staggered
   contacts -> false and simultaneous contacts -> true before Isaac starts;
7. tilt-corrected lift-off is > 6.000 mm; and
8. final cylinder tilt is below `atan(29/50) = 30.1137 deg` (a tipped cylinder is
   not counted as lifted).

Mechanism labels are also emitted for failed trials: `NO_JAW_CONTACT`,
`ONE_JAW_ONLY`, `STAGGERED_JAW_CONTACT`, `BILATERAL_LOST_IN_LIFT`,
`BOTH_JAWS_NO_LIFT`, `PRECLOSE_COLLISION`, `ARRIVAL_FAIL`,
`TIPPED_NOT_LIFTED`, `JAW_SUPPORT_CONTACT_FAIL`, and `MEASUREMENT_INVALID`.
These classifications are more
informative than one global 0/1 rate and determine the next experiment.

The global verdict partitions simultaneous contact by support clearance.  Priority
is: reproducible success; task-clear close-phase bilateral contact; task-clear
lift-only bilateral contact; bilateral contact co-occurring with jaw/support
collision; jaw/support collision with no free bilateral contact; and finally true
no-bilateral contact.  Therefore a table-assisted bilateral row can be reported as
`BILATERAL_CONTACT_COOCCURS_WITH_JAW_SUPPORT_COLLISION_NO_VALID_GRASP` but can be
neither a free-space grasp nor evidence for a global `NO_SIMULTANEOUS_*` claim.

The population decision is stratified by region, exact theta/form and q5 control.
It must never be collapsed into "top-down always fails" or "side grasp works".
The grasp point remains the top-centre; theta is the approach/tool-axis angle.
`summary.by_theta_q5` is a direct exact-control partition keyed by theta, q5,
window kind, grasp margin, descend-margin source, and q5 source (float64 hex identity
is retained).  Every row reports `n/n_valid/n_invalid`, valid-denominator success,
same-step close/lift bilateral counts and min-force authority statistics,
jaw-support failure count/force maximum, and mechanism counts.  Min-force statistics
are emitted for the measurement-valid, jaw-support-clear subset and again as
explicitly labelled all-row diagnostics.  Each row also reports
`n_jaw_support_fail`, support-fail bilateral counts, and the maximum per-environment
fixed/moving raw jaw-ground contact count.  Batch-total raw peaks remain separate
saturation authorities.

## 6. Replay, provenance, and observability (D341)

After the population run, select deterministic representatives: best
measurement-valid/support-clear lift per form, the maximum jaw-support-force failure
row when one exists, best task-clear both-jaw/no-lift failure, an explicit task-clear
lift-only row when one exists, one unilateral/no-window control, one no-bite control,
and the global worst pre-close collision (deduplicated).  Replay those exact plans
and record **every physics step**.  Population and replay must match on
`measurement_valid`, jaw-support clear/fail, arrival, preclose, close bilateral,
lift bilateral, success, and primary mechanism.  Any gate-class mismatch yields
`REPLAY_GATE_CLASS_NONREPRODUCIBLE`; success-false equality alone is insufficient.

Required canonical artifacts under `g0b_d420/`, all prefixed by the run tag:

- `*_results.json`, `*_plan.json`, `*_trace.npz`
- `*_timeline.rrd`, `*_timeline.rbl`, `*_rerun_validation.json`
- `*_inspection.png`, `*_decision_snapshot.png`
- `*_script.py.txt`, `*_argv.txt`

The RRD contains the D29 x H50 cylinder mesh, target and actual tool frames, actual
cylinder/link5/moving-jaw poses, and both actual collision-body point clouds extracted
through the full-SHA-pinned JAW authority.  The inspection clouds use its 0.5 mm
hull-surface samples with stride 16 and an explicitly reported 20,000-point/body cap;
they are body-local static geometry driven by the replay's actual link transforms, not
two body-origin dots.  All 64 parts per jaw must report successful convex-hull surface
sampling; even one raw-point fallback hard-fails before RRD emission and is recorded
in both results and Rerun-validation provenance.  The RRD also contains full step and simulation-time timelines,
q targets and actuals, decision scalars, averaged object-contact points, object/jaw
forces, and jaw-support force arrows.  The exact component contract requires
`CoordinateFrame:frame` on both body-local jaw point clouds and
`Transform3D:parent_frame` plus `Transform3D:child_frame` on the world, cylinder,
link5, and moving-jaw transforms; missing frame-graph components fail D341.  Rerun
spatial values are inspection copies only; the native callback NPZ/JSON values decide
the gate.  The SDK/CLI version, footer verification, exact entity/timeline/component
contract, fixed RBL export, and headless screenshot must pass.  The generated image
must then be actually viewed and observations written in the session document before
claiming D341 visual completion.

Each ground sensor's raw `buffer_count` is reshaped to the exact environment axis and
reduced every PhysX step.  The per-environment fixed/moving raw-count maxima are saved
in population/replay metrics and the full representative trace, while batch-total
peaks remain the separate capacity-saturation gate.

The executed p14 start bytes are captured before planning; its executed copy and argv
are written after physics stops.  Every local Python source in Section 2, the p13
result, protocol, and all five recursively composed attempt3 USD layers are
full-SHA-checked at start, after physics, and after Rerun immediately before final
serialization.  Both start and final manifests, recursive USD resolver records,
package versions, GPU identity, stage inventory, schedule and all CLI arguments are
written to results.

The installed Isaac Sim 5.1 lifecycle follows D367/D375: `env.close()` must return
internally, but `SimulationApp.close()` may terminate Python inside framework release.
Therefore p14 fsyncs every RRD/result artifact first, writes a full-SHA-bound preclose
sentinel and phase-prefix hash, appends exactly one close-start, and makes app close its
last normal call.  Results deliberately keep `cleanup.pass=false`,
`scientific_verdict=null`, and
`internal_lifecycle_verdict=PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL`; the measured
workspace branch is only `scientific_verdict_preclose_candidate`.  A post-return
success marker is forbidden.  Return/raise writes a failure marker and nonzero path;
an earlier failure writes the marker, finishes the env-close attempt, and still calls
the graceful terminal close.  The marker remains fatal even if raw exit becomes 0.
No `skip_cleanup` path exists.

## 7. Safe launch (do not run from a foreground tool timeout)

Run only after `t3x_bite81_results.json` is complete and inspected **and** the separate
`t3y_workspace_preflight2_prereg.md` instrumentation preflight passes.  Preflight1
is retained only as failed forward-only evidence and must not be rerun.  Do not edit
p14, p10, p13 or the handoff results while the process is alive.

```bash
conda activate isaaclab
(
set -o errexit
set -o pipefail
set -o noclobber
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_stdout.log
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_supervisor_pid.txt
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_python_pid.txt
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_pgid.txt
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_supervisor_contract.json
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_exit_status.txt
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_nvidia_smi_before.csv
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_nvidia_smi_after.csv
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_terminal_attestation.json
printf '%s\n' '{"artifact":"T3Y_EXTERNAL_TIMEOUT_SUPERVISOR_V1","automatic_retry_count":0,"foreground":true,"kill_after_seconds":20,"preserve_status":false,"term_signal":"TERM","timeout_seconds":21600}' \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_supervisor_contract.json
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_nvidia_smi_before.csv
nohup setsid bash -c '
  set +e
  set -o noclobber
  t3y_w1_supervisor_pid=$$
  t3y_w1_pgid="$(ps -o pgid= -p "$t3y_w1_supervisor_pid" | tr -d "[:space:]")"
  if test -z "$t3y_w1_pgid" || test "$t3y_w1_pgid" != "$t3y_w1_supervisor_pid"; then
    printf "setsid self-audit failed: pid=%s pgid=%s\n" \
      "$t3y_w1_supervisor_pid" "$t3y_w1_pgid" >&2
    exit 126
  fi
  printf "%s\n" "$t3y_w1_supervisor_pid" \
    > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_supervisor_pid.txt \
    || exit 126
  printf "%s\n" "$t3y_w1_pgid" \
    > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_pgid.txt \
    || exit 126
  timeout --foreground --signal=TERM --kill-after=20s 21600s bash -c "
    set -o noclobber
    printf \"%s\\n\" \"\$\$\" \\
      > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_python_pid.txt \\
      || exit 126
    exec python sim_scripts/p14_g0b_t3y_cyld29h50_workspace_parallel_physics_sweep.py \\
      --num_envs 1024 --grid_side 8 --run_label workspace1 --plan_workers 28 \\
      --handoff_sha256 d1460c9d80e5f65f4ab9d85a7851b29876ef7ae0ca9e28d34bd93ddb91f0170a \\
      --protocol_path claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_prereg.md \\
      --protocol_sha256 <WORKSPACE1_PROTOCOL_SHA256>
  "
  t3y_w1_python_status=$?
  printf "%s\n" "$t3y_w1_python_status" \
    > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_exit_status.txt
  exit "$t3y_w1_python_status"
' > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_stdout.log 2>&1 &
)
```

The protocol placeholder must be replaced only with p14's frozen
`WORKSPACE1_PREREG_SHA256`.  The p14 G0 guard reserves exit status, failure, phase,
preclose sentinel, and terminal-attestation paths.  The supervisor self-records `$$`
and its PGID after `setsid`; the inner Bash records its own PID then `exec`s Python, so
that PID remains the Python identity.  GNU `timeout --foreground` preserves the exact
setsid process group.  Exit 0 proves the 21600 s watchdog, TERM, 20 s KILL escalation,
and nonzero child paths were unused; 124/137/143 or missing status is FAIL.  There is
one worker and no retry.  During the run, monitor the recorded process group and
stdout; never restart under the same tag.

After termination, do not invoke the attestor until the supervisor has actually exited.
The scientific result remains unauthorized while the result is only a preclose
candidate.  The offline attestor is the sole terminal authority; it never launches
Isaac/PhysX and writes a forward-only attestation only after checking all bindings and
residue.

```bash
(
set -o errexit
set -o pipefail
python sim_scripts/p14_g0b_t3y_cyld29h50_workspace_parallel_physics_sweep.py \
  --external_terminal_attest workspace1
)
```

The attestor requires result↔sentinel↔artifact-manifest and phase-prefix full-hash
binding, exact sentinel/PID/PGID identity, exactly one sentinel-durable row followed by
exactly one close-start bound to that sentinel, exit 0, failure marker/warning absence,
supervisor/Python/PGID residue 0, a successful fresh `nvidia-smi`, the Python PID absent,
and an empty after-minus-before GPU PID set.  It also requires Rerun technical PASS but
returns only `TERMINAL_ATTESTED_PENDING_MANUAL_VISUAL`.  The inspection PNG must then
be actually viewed and recorded in the session document before the preclose scientific
candidate can be promoted.  A missing/failed item invalidates the run regardless of
partial JSON/RRD.  Launch and attestation blocks are fail-fast (`errexit` + `pipefail`).
The attestation records terminal-lifecycle and internal/Rerun previsual booleans
separately; command PASS requires both, so process cleanliness cannot overwrite a
measurement or observability failure.

## 8. Decision boundary

- Any reproducible success opens a narrow confirmatory physics case around that
  measured pose/form, followed by a separate sim-to-real safety review.  It does not
  authorize hardware motion.
- Both-jaw contact without lift motivates one new variable only (for example the
  Isaac 5.1 gripping solver ordering or friction), not an unregistered redesign.
- Bilateral contact only during lift is reported as
  `BILATERAL_CONTACT_ONLY_DURING_LIFT_NO_VALID_GRASP`; it must not be collapsed into
  a close-phase grasp.  The global no-contact verdict is allowed only when neither
  close nor lift contains simultaneous bilateral contact in either the task-clear or
  support-colliding measured subsets.
- Jaw/support collision is a valid observed task failure, not a measurement failure.
  Bilateral contact that co-occurs with it is reported separately and cannot motivate
  a free-space grip promotion until a new authorized geometry/controller variable
  removes the support contact.
- No contact across a form means geometry/arrival must be fixed before friction.
- Broad valid all-fail closes only this top-centre, fixed-controller design.  Moving
  the grasp point to the cylinder side remains a D419/professor authorization issue.
