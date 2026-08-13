# t3y_workspace_preflight1 — jaw/support sensor runtime preflight

Status: **PREREGISTERED / NOT RUN**
Case: `g0b_d420`
Canonical preflight tag: `t3y_workspace_preflight1`
Date: 2026-08-11 KST

## 1. Purpose and scope

This is a small, failure-capable **instrumentation preflight**, not a scientific
workspace experiment.  p13 observed jaw-hull/table penetrations of `-7.471`,
`-9.529`, and `-9.148 mm` with `table_penetration_count` values `6198`, `8134`, and
`7775` in seed0_S3/S4/r=0.45 candidates.  The existing object contact sensor cannot measure
jaw-to-table force.  Before the 1024-environment workspace run, this preflight tests
the installed Isaac Sim 5.1 / Isaac Lab 2.3 runtime path for two new exact force-only
contact sensors:

- `link5 -> support`
- `gripper_link -> support`

이번 preflight의 신규 연구 변수: `[]` (measurement instrumentation only)

Its shortened schedule and 2 x 2 regional grid cannot support a grasp-feasibility,
success-rate, top-down, or all-fail claim.  The runner must set
`scientific_authoritative=false`, serialize `scientific_verdict=null`, and put any
workspace branch under `diagnostic_workspace_branch` only.  Results, RRD metadata,
Rerun recording ID, and the decision snapshot must all display
`INSTRUMENTATION_PREFLIGHT_ONLY`; the diagnostic branch must not enter the experiment
ledger as a workspace result.

## 2. Frozen inputs and exact invocation

The runner hard-gates the same p10, p13, package, scipy hull sampler, seven-file
repo-local Python source manifest, exact recursive five-layer attempt3 USD composition,
and 64+64 collision/hull-surface identities as `t3y_workspace1`.  In addition, the
full p13 result SHA is:

`d1460c9d80e5f65f4ab9d85a7851b29876ef7ae0ca9e28d34bd93ddb91f0170a`

The preflight protocol file itself is full-SHA pinned by the p14 source and must be
passed explicitly with `--protocol_path` and `--protocol_sha256`.  The exact protocol
SHA is reported in the final implementation briefing after this file is frozen; do
not substitute a freshly computed hash after editing the file.

Exact noncanonical preflight controls:

- `num_envs=128`, `grid_side=2`, `plan_workers=8`
- steps: settle 120, approach 300, descend 500, close 30, hold 20, lift 30
- `settle_stat_tail=60`, `episode_length_s=120.0`, `contact_capacity=256`
- `descend_open_deg=88.30998496351378`, `approach_clearance_m=0.040`,
  `lift_delta_m=0.025`, IK position/direction gates `0.003 m / 5.0 deg`
- cylinder-authored static/dynamic material coefficients `0.40/0.30`;
  stiffness/damping `100.0/5.0`.  Jaw/support material and combine behavior remain
  full-SHA-pinned environment/USD defaults; effective pair friction is unmeasured
  and not claimed
- GPU capacities: found/lost pairs `2^23`, total aggregate pairs `2^23`, collision
  stack `2^28`, max rigid contacts `2^23`
- exact p13 result/path, protocol path/hash, and every value above are hard compared;
  any other `run_label` is rejected rather than inheriting canonical authority

The 300+500 movement steps are deliberately not the earlier 20+30 draft.  At
`dt=0.005 s` and the articulation velocity limit, the shortened draft could finish
before the known S3/S4 high-tilt support-collision poses were reached, allowing a
silent all-zero ground reporter to look healthy.

## 3. Runtime PASS/FAIL contract

The preflight passes instrumentation only when all of the following are observed:

1. object reporter count is exactly 1 and its threshold read-back is 0.0;
2. robot jaw spawn reporter count is exactly 2, exact bodies are `link5` and
   `gripper_link`, both threshold read-backs are 0.0, and an exhaustive runtime
   stage audit finds threshold-zero ContactReportAPI on both bodies in all 128
   cloned environments (256 reporter bodies total);
3. attempt3 runtime stage remains exactly 64 enabled split convex parts per jaw with
   exactly one disabled legacy collider per jaw; all 64 parts per jaw also have
   successful convex-hull surface sampling with zero raw-point fallbacks;
4. object force and contact-point tensors are exactly `(128,1,3,3)` with one body and
   three filters, while each jaw-ground force tensor is exactly `(128,1,1,3)` with
   one body and one support filter; both force-only jaw sensors must have
   `contact_pos_w is None`;
5. object, fixed-jaw-ground, and moving-jaw-ground batch-total raw contact-count peaks are
   recorded every step, remain below their respective capacity `256*128=32768`, and
   no CUDA/PhysX contact-buffer error occurs.  Each ground sensor's raw count is also
   hard-checked as `(128,1)`, while object raw count is `(128,3)`; fixed/moving
   per-environment maxima are retained independently;
6. representative `trace.npz` contains every-step
   `replay_trace_jaw_ground_force_w_n` and `replay_trace_jaw_ground_raw_count` with
   fixed/moving force vectors and per-environment raw counts;
7. result metrics contain both jaw-ground episode maxima,
   per-environment raw-count maxima, `jaw_support_contact_pass`, and the
   `JAW_SUPPORT_CONTACT_FAIL` classification path.  A reliable >0.02 N value is a
   measurement-valid task failure and success-false; if positive-control failure
   co-occurs, `MEASUREMENT_INVALID` remains the primary label but a reason flag must
   preserve the jaw/support observation;
8. RRD exact components include `CoordinateFrame:frame` for both body-local jaw
   clouds and `parent_frame`/`child_frame` for all world-to-body transforms; jaw-ground
   force and per-environment raw-count scalars/arrows are present on the full replay
   timeline; RRD/PNG metadata says `INSTRUMENTATION_PREFLIGHT_ONLY`; and
9. at least one feasible exact `seed0_S3`/`seed0_S4` high-tilt row produces both
   nonzero jaw-ground force and nonzero **same-environment** raw count, and at least
   one such row exceeds 0.02 N.  Missing witness IK or all-zero witness output is a
   preflight FAIL, never a silent PASS; and
10. every population batch and representative replay passes the support-force
    positive control, and representative replay reproduces every gate class and
    primary mechanism; and
11. every repo-local source and recursively composed USD layer end hash equals its
    start pin, with the complete start/final manifests in results.

Because this revision restores the full 120-step settle and 60-step statistic tail,
every population batch and the representative replay must pass the support-force
positive control; population/replay gate classes and mechanisms must also reproduce.
A canonical workspace run remains blocked unless those controls, reporter/filter
shapes, raw-buffer integrity, same-environment nonzero witness, >0.02 N task-failure
witness, and RRD/source contracts all pass.  If API construction, filter resolution,
positive control, replay class, or witness response fails, stop and author a new
forward-only preflight tag; do not weaken or silently remove the sensor.

## 4. Detached launch

Replace `<PREFLIGHT1_PROTOCOL_SHA256>` only with the frozen value reported by p14's
`PREFLIGHT1_PREREG_SHA256` constant.  Do not run in a foreground tool timeout.

```bash
conda activate isaaclab
set -o noclobber
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight1_stdout.log
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight1_pid.txt
nohup python sim_scripts/p14_g0b_t3y_cyld29h50_workspace_parallel_physics_sweep.py \
  --run_label workspace_preflight1 --num_envs 128 --grid_side 2 --plan_workers 8 \
  --settle_steps 120 --approach_steps 300 --descend_steps 500 --close_steps 30 \
  --hold_steps 20 --lift_steps 30 --settle_stat_tail 60 --contact_capacity 256 \
  --handoff_sha256 d1460c9d80e5f65f4ab9d85a7851b29876ef7ae0ca9e28d34bd93ddb91f0170a \
  --protocol_path claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight1_prereg.md \
  --protocol_sha256 <PREFLIGHT1_PROTOCOL_SHA256> \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight1_stdout.log 2>&1 &
printf '%s\n' "$!" > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight1_pid.txt
```

Expected p14 artifacts use prefix `t3y_workspace_preflight1_`:

- `results.json`, `plan.json`, `trace.npz`
- `timeline.rrd`, `timeline.rbl`, `rerun_validation.json`
- `inspection.png`, `decision_snapshot.png`
- `script.py.txt`, `argv.txt`
- plus the detached `stdout.log`, `pid.txt`, and a separately captured
  `nvidia_smi_before.txt`

After exit, inspect `results.json`, buffer peaks, filter maps, the RRD validation, and
the actual `inspection.png`.  The machine result is either
`INSTRUMENTATION_PREFLIGHT_FAIL` or
`INSTRUMENTATION_PREFLIGHT_RUNTIME_PASS_PENDING_VISUAL_INSPECTION`; only the latter
plus actual screenshot inspection is the technical GO for canonical
`t3y_workspace1`.  `scientific_verdict` remains null.  The user has already granted
Isaac execution authorization; this does not authorize any hardware action.
