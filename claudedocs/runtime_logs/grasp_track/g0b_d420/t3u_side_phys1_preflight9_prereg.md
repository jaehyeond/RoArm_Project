# p16 v8 / t3u_side_phys1 preregistration — preflight9-bound canonical PhysX grasp

Status: **BLOCKED ON SUCCESSFUL `t3u_side_preflight9` + TERMINAL + MANUAL VISUAL / NOT RUN**  
Case: `g0b_d420`  
Canonical prefix: `t3u_side_phys1`  
Date: 2026-08-12 KST

`t3u_side_phys1_prereg.md` (SHA-256
`c52a31bddf6cfd64700074c66d0b6c1d43736379f37c581842334ce06819bbb2`) is
**retired historical evidence and non-executable**. This file,
`t3u_side_phys1_preflight9_prereg.md`, is the sole canonical protocol. P16 v8 pins both
bytes but its parser can select only this new prereg for `side_phys1` and explicitly
rejects the retired path. The formerly active
`t3u_side_phys1_preflight3_prereg.md` (SHA-256
`b1b20f9e8eee24950f53c663f3712d787f77ac697cb66ada87b0502b17c51faf`) is also
retired/non-executable after its preflight launch died inside the Codex sandbox; neither
old document can be selected by the v7 parser. The later
`t3u_side_phys1_preflight4_prereg.md` (SHA-256
`6ccc5616d35abd8863c7bf48dc005cb7e058daf32414fd51df65d7f08a46466f`)
is likewise retired/non-executable after preflight4 failed before contract/child creation.
Only this preflight9-bound document can authorize `side_phys1`.

## 1. Scientific question

Under the user-approved sim-only D419 exception, can the fixed-base RoArm-M3-Pro grasp
the upright D29 x H50, 24.83 g cylinder across its side midpoint and lift it by contact
physics, without kinematic attachment, desk/support collision, non-jaw shoving, or
self-collision?

This is a local existence test, not a workspace atlas.  One valid passing row is
sufficient for `SIDE_MIDPOINT_GRASP_PASS_IN_FIXED_SIM_CONTROLS`.  If none passes, the
specific preregistered failure or no-evidence branch in section 3 must be reported rather
than collapsed into a generic label.  Neither result establishes real hardware
performance or measured friction.

이번 case의 신규 변수: `[side-midpoint grasp point, SDG candidate pose]`.

## 2. Frozen inheritance from instrumentation preflight

Sections 2 through 9 of `t3u_side_preflight9_prereg.md` are inherited exactly except:

- this tag is scientifically authoritative after all validity, lifecycle, Rerun, and
  visual gates pass;
- canonical environment count is `64`; exactly eight p15 candidates crossed with five
  frozen pinch offsets produce exactly 40 ordered planned rows.  The active set is hard
  pinned to exactly 10 ordered rows: candidates 5 and 7, each with offsets 0..4.  This
  equality is checked before environment creation and again after transformed-hull
  clearance; any other nonempty IK set is drift and aborts rather than silently changing
  the experiment.  All 30 preregistered IK/frame rejects and padded rows remain reported
  but excluded from PhysX denominators.  The first padding slot after the 10 active rows
  is the inherited witness and is always excluded from task counts.  Zero rows after the
  independent static-clearance gate selects the explicit no-safe-planned-path verdict;
- the frozen pre-physics planner found exactly 10/40 IK/frame-feasible rows: all five
  offsets for p15 candidates 5 and 7.  Candidate 5 is the first passing candidate in
  canonical p15 order and remains the preflight/canonical visual representative; this
  selection happened before any p16 Isaac/PhysX step and may not be replaced by whichever
  row later has the best physical outcome;
- candidate order, five geometry-derived offset controls, controller, full phase
  durations, contact schemas, representative selection, and render cadence must be
  byte-for-byte equivalent to the passing preflight contract;
- every external supervisor/lifecycle filename substitutes the prefix
  `t3u_side_phys1` for `t3u_side_preflight9`; the detached no-retry contract and its
  timeouts are otherwise exact. The entire wrapper must be submitted as one host-
  authorized `exec_command` with `sandbox_permissions=require_escalated`. Before G0 is
  consumed it rejects a sandbox PID 1, checks NVIDIA access and frozen hashes. V11 repeats
  that guard before outputs and binds PID-1 bytes/hash, scoped self/own-PID namespace
  device/inode/readlink, boot ID and no-TTY session-leader identity into contract/outcome
  for terminal recomputation. Self/own-PID consistency is explicitly not PID-1 namespace
  equality or host proof; PID-1 cmdline plus the full ancestor forbidden-token walk is
  the sandbox-rejection authority. The wrapper
  waits at least two and at most ten seconds before its exact second liveness guard; a
  mismatch/timeout performs bounded TERM-to-`$!` then KILL/reap and never retries;
- the exact full SHA-256 of the passing preflight result, terminal attestation, manual
  visual inspection, all supervisor/raw-lifecycle files, RRD/RBL/NPZ/PNGs/render manifest
  and MP4, p15 handoff, both preregistration files, p16 source/supervisor, pinned p14
  helper, jaw extractor, kinematics, environment sources, URDF, and all five attempt3
  USD layers are recorded before AppLauncher and verified unchanged after physics.
  Canonical reloads the preflight result/plan/trace/render/outcome files and recomputes
  exact result, render, lifecycle, and supervisor semantic contracts with the same pure
  raw verifier used by terminal attestation.  This includes exact child schemas/commands,
  PID/PGID/SID/TTY and raw wait status, timestamps, signal/timeout/retry/cleanup fields,
  process and GPU residue, stdout/launcher/phase hashes, every outcome binding, and the
  stored attestation's exact equality to the recomputation.  A terminal attestation
  `pass=true` or arbitrary all-true map is insufficient;
- Supervisor V11 applies the same raw-zero-is-insufficient rule to both children.
  Canonical admission requires the embedded physics preclose gate and post-hoc render
  gate to equal independent recomputation.  The render gate rechecks exactly 234 frame
  files, every NPZ/state/clock mapping, and independent decoding of all PNGs plus every
  MP4 frame with count/size/fps/hash gates, zero physics,
  exact durable render phase/stdout completion, and physics-finalize/render-start/end
  dependency equality.  Physics phases have exact row schemas/finite ordered times and
  stdout PREClose is reconstructed byte-for-byte from the result.  The NPZ must encode
  exact steps `1..2340`, `step/200` times and frozen phase arrays; render sampling must be
  steps `10..2340` by tens, with the exact 491-label zero-physics clock schedule
  (250 app updates, 240 Replicator calls, one new-stage audit).  Numeric schema is strict:
  Boolean values can never stand in for zero-valued integer counts or float clocks,
  deltas, frame/phase fields, fidelity errors, render-phase observations, or decode
  count/size/fps fields.  The dependency map is
  the exact profile-specific full physics set, including every p15 bound file and, for
  canonical, all passing-preflight files—not a reduced render-only subset.  Raw-zero
  semantic failure maps to reserved status 125 and a
  render-abort attestation is always `pass=false`/non-promotable;
- raw lifecycle schema is likewise strict and shared across success, physics-abort,
  render-abort, terminal recomputation, and canonical admission.  PID/PGID/SID must be
  integers greater than one; attempt/raw-wait/exit/signal/combined-status fields must be
  exact non-Boolean integers; times must be finite ordered floats. Recursive V11
  supervisor-contract and outcome equality, plus terminal-attestation equality, compares JSON type as well
  as value, so `false == 0` and `true == 1` can never authorize promotion;
- the preflight and canonical subjects both hard-gate the actual composed fixed-base
  metatype/root joint and full-step fixed-body stability, all 15 non-adjacent self pairs
  (including fixed `world` versus five non-adjacent moving bodies), and actual composed/
  runtime joint limits plus post-clamp applied-target equality to parsed URDF controls.
- both profiles also use the identical preflight9 self-collision gate: unique
  `/root_joint` ArticulationRootAPI + PhysxArticulationAPI per clone; typed authored
  property-stack `True` stronger than the pinned attempt3 source-layer `False`; exact
  Isaac Lab root PhysX-view identity/count/check. The deprecated Dynamic Control query,
  handle and retry path are absent. Instead, exactly two same-process raw PhysX frames
  must distinguish the frozen positive pose from HOME in every clone. Before either
  diagnostic, all 15 pair views must read `sensor_count=N`, `filter_count=1`, ordered
  env0..envN-1 subject paths, exactly one concrete env0 filter representative equal to
  the expected target, independently exact regex resolution to all env0..envN-1 stage
  targets, raw-count shape `(N,1)`, and actual capacity `N*256`; scene flags are
  explicitly `replicate_physics=True`,
  `filter_collisions=True`, `clone_in_fabric=False`. Positive overlap
  set exactly `{link2__link4, link2__link5}`, both geometric inradii `>=5 mm`, both raw
  counts `1 <= count < 256`, per-view total strictly below actual buffer capacity, and
  forces `>0.02 N`; all other 13 pairs plus support/object channels
  zero/`<=1e-8 N`; then HOME all 15 pairs and support/object channels zero/`<=1e-8 N`
  with frozen signed `link2__link4` separation `<=-60 mm`. After HOME write and before
  its step, direct root-PhysX DOF/link transforms must match HOME within `1e-7 rad`,
  show zero overlaps for all 15 pairs, and actual separation `>=60 mm` in every clone.
  The 15 view identities/capacities are reread after both controls and must equal their
  pre-control report apart from epoch/clock. Each frame advances exactly one
  `0.005 s` callback and both engine clocks while task counters remain zero.  This is a
  two-pose behavioral instrumentation proof, not proof of every possible manifold; the
  unchanged 15-pair full-step task trace remains scientific evidence.
- before local
  task step 1, the exact object/robot state and authoritative PhysX targets are restored,
  `scene.write_data_to_sim()` plus zero-physics `sim.forward()` is called, all sensor
  private buffers/timestamps and task latches are reset and gated, and task clock
  baselines are recorded.  Because the object root PhysX view uses simulation-view
  subspace root `/`, its transform expectation is the world position
  `OBJECT_CENTER_M + env_origin`; a pure nonzero-origin regression must match all eight
  world rows and reject the env-local candidate for env1..7.  The first task step must
  show fresh timestamps.  The local
  scientific trace remains exactly `1..2340` (`11.700 s`); diagnostic `2`, task `2340`,
  and combined physical steps `2342` are recorded separately.  No global clock is
  rewritten or represented as zero after the diagnostic frames.
- the preflight9 lifecycle repair is inherited exactly: `reward_buf` must be absent at
  the post-diagnostic pre-task re-baseline because Isaac Lab 2.3 creates it in the first
  `DirectRLEnv.step()`. The first unchanged task step must return a finite all-zero
  `(N,)` tensor exactly equal to the newly created `env.reward_buf`; no fake buffer is
  synthesized. Durable self-contact `pair_rows` validation requires the exact 15-key
  set and every row semantic but is intentionally independent of JSON key order.

The failed `t3u_side_preflight1` remains immutable negative evidence: it executed zero
task-physics steps and exposed an instrumentation defect, not a grasp result.  Its
authored USD rest transforms (`q=0`) had been compared against hard-coded HOME
`q2=90 deg`; the old p10 compact chain also rounded the URDF's `0.051959 m` and
`1.5708 rad` literals.  Preflight2 and canonical therefore use the exact decimal frozen
URDF origin/RPY/axis chain for every 2,340-sample collider-clearance transform.  They
separately gate (a) default-time authored/rest USD frames against exact URDF `q=0` and
(b) same-epoch PhysX articulation joint positions against actual body poses.  p10 remains
IK provenance only and its delta is diagnostic; no clearance, contact, pose, or success
threshold was relaxed.

The failed `t3u_side_preflight2` is also immutable and non-promotable. It executed zero
task-physics steps because the old instrumentation read `/Robot` instead of discovering
the schema prim `/Robot/root_joint`. Its terminal-attestation/failure/outcome/phase/exit
hashes are pinned by preflight9 and canonical solely as predecessor provenance. They are
not a grasp verdict, science artifact, or successful-preflight substitute.

The attempted `t3u_side_preflight3` is immutable launch-infrastructure evidence, not a
preflight result. Its original launch-authority inventory is only the frozen
preregistration and zero-byte launcher log; V5 contract/PID/PGID and every Isaac/science
output are absent.
Because pinned V5 fsyncs those lifecycle files before the physics fork, the preserved
inventory proves Supervisor/Isaac never started and task-physics steps are zero. P16 v8
pins the v2 source, V5, both preflight3 preregs, and zero launcher SHA; it admits
canonical only through a successful `side_preflight9` result.
The later `preflight4_frozen_audit` child accidentally invoked V6 with this retired
profile, creating only a 3,074-byte argparse `SystemExit(2)` failure marker (SHA-256
`218ec29911134acaca1d472762fa27341f87fed136bd39849099c2eeca35ebcc`). It is pinned as
posthoc non-science audit contamination with null child/outcome, never as original launch
authority or promotion evidence; it is preserved and not deleted.

`t3u_side_preflight4` is also immutable infrastructure evidence only. Its exact prefix
contains its 30,676-byte prereg
(`6b413e343630cbac6dbec458769aac9310c9caea3cfedfb436d0f3582ac2ea13`),
the 1,397-byte Supervisor V6 PermissionError record
(`50cd5e0eec3444e44862dc0885137389c8073decbfdf7fbbe8d2a55b8bbf66b5`),
and its 786-byte matching launcher traceback
(`3b37b2967c6dcb702f71dde28a8c3dd1d2069a7ec7f15a650f91667096bca2e9`).
V6 failed while statting inaccessible `/proc/1/ns/pid`, before contract/PID/PGID and
before its child fork; child/outcome are null and task-physics steps are zero. P16 v8
pins those files plus frozen v3 source, V6 and the retired preflight4 canonical prereg.
None can substitute for successful preflight9 terminal/render/manual evidence.

`t3u_side_preflight5` is immutable startup/readback failure evidence only.  Frozen p16
v4 and Supervisor V7 are SHA-256
`f019d55b437c93e53a2f6820af633821765c24a8741cd170fe3b4d189dc4a4ad` and
`b344b49fb955a833ef4eee92c48f4ef7cf95ffdda4e4cef58cd806a681d15fcd`.
The exact prereg/failure/outcome/phase/terminal hashes are respectively
`319376d827f92355a51c71a0397f3aeace6f6a70c4ce4c3a41a8d8e7aa3c349b`,
`0a051340dc4b448032fa4ebceee7927229497e4d5799502fbf90c71604746b5b`,
`cd44a132735d001b05f4c93bf9c9bdf05c76cded8a877b86d9f87570a24191d6`,
`ea99ff199f00f3fe28fc0d0dfd28655c8163c0d410f178cd716ce43a89df8d76`,
and `d99a0f19d946d149d3307134cd79305b4ba5a1858662758d5a867717dbf9a84e`.
Its child and supervisor were reaped, render never started, and GPU inventories were
unchanged.  It passed the USD/root-view checks but Dynamic Control returned zero
candidates in all eight clone containers before the local task loop.  Its raw child
status zero became semantic status `125`; it is neither a self-collision observation nor
a grasp verdict, and it cannot substitute for a successful preflight9.

`t3u_side_preflight6` is also immutable instrumentation failure evidence. Frozen p16 v5
and Supervisor V8 hashes are
`b6eb67cbec8e11752b926d8d04498c3d29fd993b8ac87b5aabbc207c92d06458` and
`8cd7946b7dfb826a2fce8a9a9580603a945037aa48aa591d8979fc58ba03d9b2`;
its prereg/canonical-prereg hashes are
`198c81869ff8a547edb5bbc497e0c080864b39cf7ae47db676a03ba7d5028375`
and `9dcaeee6840edeea81b0e7b7a1b92aa2415f57f03c1173be921692dda7556cc0`.
It advanced exactly one non-task activation frame but Dynamic Control again returned
zero articulation candidates in all eight clones, then failed before `run_physics`.
Failure/outcome/phase/exit/terminal hashes are
`43e086551ee54063795fd915d5fa8c0dfd927855090928ca69f2518f609ab245`,
`da8161d632b6da1ba48e8a6c25a3cca240461bfd03b4f6fefc8c6561793adbe7`,
`ce6771980c37ab13dc4ed7f5ff52348be34d00ed750c14c2492ae73c274143d4`,
`a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca`,
and `3d13364e2cb3113c69485e31aa12f30e1403ccef323852eb12bf58d464094d08`.
It has task steps zero, render zero, exit `125`, `pass=false`, and promotion false; it
cannot substitute for preflight9.

`t3u_side_preflight7` is immutable representation-audit failure evidence only. Frozen
p16 v6 and Supervisor V9 hashes are
`aabac6c76985682e32376195d187134da028bb6cc768148e883fbd56c18b3dbe` and
`9f1cf1be075fe052f8d2db196be9a14207d80dc82f46a58a364a88513bacb716`;
its prereg/canonical-prereg hashes are
`c2b17be775ad4df465c967d3cbdf08c571eed3cfbd21325265798a906b0d6e96`
and `1453c33642b5d32e2e24dba66da5732240afc6484af7142ddc7953c97b1efbbd`.
It observed one logical filter as the env0 concrete representative while the configured
regex independently resolved all eight valid stage targets, then aborted before either
behavioral diagnostic or the task schedule. Failure/outcome/phase/exit/terminal hashes
are `3b84af93ab399725a2fab220d9dd5883d6f5286bfa0241c63bec04b15d5bc01a`,
`26e161620c725c45e24beaf163fe8221bff061ffbf667ea2fac811259bda64f0`,
`43788a37e63e000bf00d30346f6fdef9d152eef51041ca1ea604420cb028e2de`,
`a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca`,
and `41a65b75453e4f97a326ec8ae4e966a09070812a1e8374696f3a46f421e4a8ea`.
It has render zero, semantic exit `125`, valid abort attestation, no science verdict,
and cannot substitute for preflight9.

Any preflight failure or contract drift blocks this canonical run; it cannot be bypassed
by changing `run_label`, environment count, candidate list, offsets, phase lengths, or
camera selection.

## 3. Scientific verdict ladder

Before physics, failure of the exact 40-planned/10-active IK contract is a source or
dependency drift abort, not a scientific branch.  If the exact IK rows exist but any is
removed by the preregistered 1 mm enabled-collider/support clearance over all 2,340
command samples, return `NO_PLANNED_STATIC_CLEARANCE_FEASIBLE_SIDE_PATH`.  That means
PhysX grasp evidence is absent and may not be presented as a physical grasp failure.

If PhysX executes, every active row is assigned exactly one causal label in the progress
order below, and branch masks/counts are serialized.  Population selection is existential:
`measurement-invalid in any row` blocks all claims; otherwise `success in any row` proves
the local existence result; otherwise the **deepest populated failure branch** is the
global verdict.  Thus one early-failing row cannot hide other rows that reached bilateral
lift contact.  The ordered row branches are:

1. `MEASUREMENT_INVALID` — any row has invalid numerical/contact instrumentation or the
   run-level lifecycle, provenance, Rerun, or visual evidence fails.  A scientific task
   verdict is forbidden.
2. `SIDE_MIDPOINT_GRASP_PASS_IN_FIXED_SIM_CONTROLS` — at least one valid row passes every
   exact gate.  Report every passing candidate/offset, not only the best video.
3. `TRACKING_OR_ARRIVAL_GATE_FAIL` — no success and a valid row fails grasp/lift arm
   arrival or commanded TCP-rise tracking.  Contact cannot hide a tracking failure.
4. `PREMATURE_JAW_CONTACT_BLOCKS_SIDE_GRASP` — no earlier branch and a valid arrived row
   contacts the object with either jaw before close above `0.02 N`.
5. `NONJAW_OR_SUPPORT_COLLISION_BLOCKS_SIDE_GRASP` — no earlier branch and a row is
   blocked by moving-link/support, non-jaw/object, or one of the 15 gated self contacts.
6. `NO_BILATERAL_SIDE_CONTACT` — this row is valid, arrived, task-clear, but lacks
   same-step bilateral contact during close.  It can be the population verdict only when
   no row reaches any later branch.
7. `BILATERAL_CONTACT_LOST_BEFORE_LIFT` — this row closes bilaterally but does not retain
   same-step bilateral contact during lift.  It is selected only when no row gets farther.
8. `BILATERAL_CONTACT_BUT_NO_CORRECTED_LIFT` — bilateral contact survives into lift for
   this row, but corrected object rise does not strictly exceed `6 mm`; it is selected only
   when no row reaches a later branch.
9. `OBJECT_LIFTED_BUT_TIPPED` — corrected lift passes for this row but final tilt fails;
   it is selected only when no success or later residual exact-gate branch exists.
10. `OTHER_EXACT_GATE_FAIL` — only a residual exact-gate combination not captured above;
    masks and row labels must expose which condition remained false.

Corrected lift is the cylinder bottom-equivalent centre rise after subtracting its own
settled rest pose, not TCP motion.  Bilateral force authority is the same-step scalar
`max_t(min(F_fixed(t),F_moving(t)))`; independent per-jaw maxima cannot prove a grasp.

## 4. Required outputs and scope warning

Required prefix outputs include `_results.json`, `_plan.json`, `_trace.npz`,
`_timeline.rrd`, `_timeline.rbl`, `_rerun_validation.json`, `_inspection.png`,
`_decision_snapshot.png`, `_rgb_frames_manifest.json`, `_side_grasp.mp4`, `_script.py.txt`, `_argv.txt`, lifecycle
phase/sentinel/attestation records, and a separate manual visual inspection record.

Canonical uses the same frozen no-retry supervisor and lifecycle modes with only the
registered profile changed. The preflight9 host wrapper in its section 8 is normative:
this block likewise must be one `require_escalated` host command; its omitted guard
lines are inherited verbatim with `p16_profile=side_phys1`.

```bash
(
  set -euo pipefail
  set -o noclobber
  p16_repo=/home/cgxr/Documents/Robotics/RoArm_Project
  p16_case_dir=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/g0b_d420
  p16_profile=side_phys1
  p16_prefix="t3u_${p16_profile}"
  p16_pid1_hex="$(od -An -tx1 -v /proc/1/cmdline | tr -d ' \n')"
  p16_pid1_text="$(tr '\0' ' ' </proc/1/cmdline)"
  case "${p16_pid1_text}" in
    *bwrap*|*--die-with-parent*|*codex-linux-sandbox*)
      echo "HOST_GUARD sandbox PID1 forbidden: ${p16_pid1_text}" >&2
      exit 4
      ;;
  esac
  [[ -n "${p16_pid1_hex}" ]] || { echo "HOST_GUARD empty PID1 cmdline" >&2; exit 4; }
  [[ -c /dev/nvidiactl ]] || { echo "HOST_GUARD /dev/nvidiactl missing" >&2; exit 4; }
  timeout 15s nvidia-smi >/dev/null
  echo 'bf36d2acbe9cb7fb6cb6721e8dca90ce4f7623fab5813449f459893131fae29b  sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v11.py' | (cd "${p16_repo}" && sha256sum -c -) >/dev/null
  echo '67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384  claudedocs/runtime_logs/grasp_track/g0b_d420/t3s_side_sdg2_candidates.json' | (cd "${p16_repo}" && sha256sum -c -) >/dev/null
  /home/cgxr/miniconda3/envs/isaaclab/bin/python \
    "${p16_repo}/sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v8.py" \
    --prelaunch_guard side_phys1 >/dev/null
  p16_suffixes=(
    results.json plan.json trace.npz timeline.rrd timeline.rbl
    rerun_validation.json decision_snapshot.png inspection.png
    rgb_frames_manifest.json side_grasp.mp4 script.py.txt argv.txt phase.jsonl
    render_phase.jsonl
    preclose_sentinel.json terminal_attestation.json manual_visual_inspection.json
    failure.json render_failure.json exit_status.txt stdout.log supervisor_launcher.log
    supervisor_pid.txt physics_python_pid.txt render_python_pid.txt pgid.txt
    supervisor_contract.json supervisor_outcome.json nvidia_smi_before.csv
    nvidia_smi_supervisor_end.csv nvidia_smi_after.csv supervisor_failure.json
  )
  for p16_suffix in "${p16_suffixes[@]}"; do
    p16_target="${p16_case_dir}/${p16_prefix}_${p16_suffix}"
    if [[ -e "${p16_target}" ]]; then
      echo "G0 existing target: ${p16_target}" >&2
      exit 3
    fi
  done
  if [[ -e "${p16_case_dir}/${p16_prefix}_rgb_frames" ]]; then
    echo "G0 existing frame directory" >&2
    exit 3
  fi
  p16_wait_reapable() {
    local p16_wait_deadline=$((SECONDS + 20))
    local p16_stat p16_tail p16_state
    while (( SECONDS < p16_wait_deadline )); do
      if [[ ! -r "/proc/${p16_supervisor_pid}/stat" ]]; then
        wait "${p16_supervisor_pid}" 2>/dev/null || true
        return 0
      fi
      if p16_stat="$(<"/proc/${p16_supervisor_pid}/stat")"; then
        p16_tail="${p16_stat##*) }"
        p16_state="${p16_tail%% *}"
        if [[ "${p16_state}" == Z ]]; then
          wait "${p16_supervisor_pid}" 2>/dev/null || true
          return 0
        fi
      fi
      sleep 1
    done
    return 1
  }
  p16_abort_launch() {
    echo "HOST_LIVENESS_FAIL reason=$1 pid=${p16_supervisor_pid}; no retry" >&2
    kill -TERM "${p16_supervisor_pid}" 2>/dev/null || true
    if ! p16_wait_reapable; then
      kill -KILL "${p16_supervisor_pid}" 2>/dev/null || true
      if ! p16_wait_reapable; then
        echo "HOST_LIVENESS_UNREAPED pid=${p16_supervisor_pid}" >&2
        exit 7
      fi
    fi
    exit 6
  }
  p16_monotonic_ns() {
    /home/cgxr/miniconda3/envs/isaaclab/bin/python -c \
      'import time; print(time.monotonic_ns())'
  }
  p16_liveness_start_ns="$(p16_monotonic_ns)"
  p16_liveness_deadline_ns=$((p16_liveness_start_ns + 10000000000))
  nohup setsid /home/cgxr/miniconda3/envs/isaaclab/bin/python \
  /home/cgxr/Documents/Robotics/RoArm_Project/sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v11.py \
    --profile "${p16_profile}" \
    --candidates_sha256 67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384 \
    </dev/null >"${p16_case_dir}/${p16_prefix}_supervisor_launcher.log" 2>&1 &
  p16_supervisor_pid=$!
  echo "LAUNCH_ACCEPTED pid=${p16_supervisor_pid} profile=${p16_profile} retry=0"
  if ! sleep 2; then
    p16_abort_launch minimum_survival_sleep_interrupted
  fi
  while true; do
    if ! p16_liveness_now_ns="$(p16_monotonic_ns)"; then
      p16_abort_launch monotonic_clock
    fi
    (( p16_liveness_now_ns < p16_liveness_deadline_ns )) || p16_abort_launch timeout
    if [[ -s "${p16_case_dir}/${p16_prefix}_supervisor_contract.json" &&
          -s "${p16_case_dir}/${p16_prefix}_supervisor_pid.txt" &&
          -s "${p16_case_dir}/${p16_prefix}_pgid.txt" ]]; then
      if ! /home/cgxr/miniconda3/envs/isaaclab/bin/python \
        "${p16_repo}/sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v8.py" \
        --launch_liveness_guard side_phys1 \
        --supervisor_pid "${p16_supervisor_pid}" >/dev/null; then
        p16_abort_launch second_guard
      fi
      if ! p16_liveness_now_ns="$(p16_monotonic_ns)"; then
        p16_abort_launch monotonic_clock
      fi
      (( p16_liveness_now_ns < p16_liveness_deadline_ns )) || p16_abort_launch deadline
      exit 0
    fi
    kill -0 "${p16_supervisor_pid}" 2>/dev/null || p16_abort_launch early_exit
    if ! sleep 0.25; then
      p16_abort_launch polling_sleep_interrupted
    fi
  done
)
```

After the supervisor and both child groups are reaped, run exactly:

```bash
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  /home/cgxr/Documents/Robotics/RoArm_Project/sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v8.py \
  --terminal_attest side_phys1
```

The block above is the complete G0/no-retry wrapper, not permission to overwrite the
launcher log or any tag artifact.

The frozen detached supervisor V11 source SHA-256 is
`bf36d2acbe9cb7fb6cb6721e8dca90ce4f7623fab5813449f459893131fae29b` at
`sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v11.py`; canonical accepts only
the exact `--profile side_phys1 --candidates_sha256
67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384` argv shown above.

Candidate index 5 (`side_sdg_005_raw_025092`) / nominal offset remains the video
representative even if another row
is scientifically successful.  Every passing row is still reported in JSON; the video
cannot select or tune the best outcome after physics.

Scope warning serialized verbatim:

> A PASS is one fixed analytic-cylinder pose under unmeasured placeholder material
> coefficients and the parsed URDF joint envelope.  It does not authorize hardware,
> does not measure real friction, does not validate an actual finite desk, does not
> reopen prior top-centre verdicts, and does not yet constitute Isaac Lab learning.
