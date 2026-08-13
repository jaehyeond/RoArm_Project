# p16 v8 / t3u_side_preflight9 preregistration — behavioral self-collision control

Status: **PREREGISTERED / NOT RUN / STATIC RE-AUDIT REQUIRED**  
Case: `g0b_d420`  
Canonical prefix: `t3u_side_preflight9`  
Date: 2026-08-12 KST

## 1. Why this forward-only tag exists

`t3u_side_preflight3` never reached Supervisor V5.  Its original launch-authority prefix
contained only its 22,626-byte preregistration and a zero-byte
`supervisor_launcher.log`; every
contract/PID/PGID/phase/failure/outcome/science/render artifact and frame directory is
absent, and the matching process inventory is empty.  The frozen V5 source writes and
fsyncs the supervisor contract, PID and PGID before forking the physics child.  Their
absence therefore establishes **Supervisor not started, Isaac child not started, and
0 task-physics steps**.  The shell accepted a background PID inside Codex's bubblewrap
PID namespace, but `bwrap --die-with-parent` destroyed that namespace when the tool call
returned.  This is launch-infrastructure evidence, not a grasp or instrumentation
result, and the tag is retired with no retry.

At 13:10:47 KST, the independent `preflight4_frozen_audit` child violated its read-only
brief by invoking historical Supervisor V6 with the retired `side_preflight3` profile. Argparse
rejected it with `SystemExit(2)` before `main()` and before any supervisor/Isaac child,
but V6's outer diagnostic handler created a third 3,074-byte
`t3u_side_preflight3_supervisor_failure.json` (SHA-256
`218ec29911134acaca1d472762fa27341f87fed136bd39849099c2eeca35ebcc`, mtime-ns
`1786507847537740266`). It is preserved, never deleted, and classified only as
`posthoc_static_audit_contamination_non_science`; it cannot alter the original two-file
launch inference or satisfy any preflight/science/promotion requirement. P16 v8 now
requires the exact three-entry inventory and independently checks that file's schema,
argv, `SystemExit(2)`, null child/outcome and frozen bytes.

`t3u_side_preflight4` then used the correctly authorized host wrapper and did start
Supervisor V6, but V6 failed before writing its contract/PID/PGID and before forking an
Isaac child.  The exact prefix is only these three files:

- frozen `t3u_side_preflight4_prereg.md`: 30,676 bytes, SHA-256
  `6b413e343630cbac6dbec458769aac9310c9caea3cfedfb436d0f3582ac2ea13`;
- `t3u_side_preflight4_supervisor_failure.json`: 1,397 bytes, SHA-256
  `50cd5e0eec3444e44862dc0885137389c8073decbfdf7fbbe8d2a55b8bbf66b5`;
- `t3u_side_preflight4_supervisor_launcher.log`: 786 bytes, SHA-256
  `3b37b2967c6dcb702f71dde28a8c3dd1d2069a7ec7f15a650f91667096bca2e9`.

The launcher bytes exactly equal the failure traceback.  It records
`PermissionError: /proc/1/ns/pid`, the exact `side_preflight4` V6 argv,
`last_child_outcome=null`, empty active-child/cleanup state and no signal.  Frozen V6
attempted that inaccessible `stat` before contract/PID/PGID creation and before the
physics fork.  Therefore Supervisor Python started, but physics/render child count and
task-physics steps are both zero.  This is host-context instrumentation failure, not a
self-collision or grasp result, and preflight4 is retired with no retry.

Preflight5 changed only that observed host-context readback.  It never read or statted
`/proc/1/ns/pid`.  Before consuming any output, frozen p16 v4 checked that
`/proc/self/ns/pid` and `/proc/<its-own-pid>/ns/pid` are both accessible and have the
same positive device/inode and readlink.  Frozen Supervisor V7 made the same check for its
own PID.  This is labelled only `supervisor_self_namespace_consistent`: it is a
self/own-PID procfs consistency check and **not** evidence of equality with PID 1 or of
host execution.  Sandbox rejection authority remains the PID-1 command-line guard plus
the complete visible ancestor walk for exact `bwrap`, `codex-linux-sandbox`, and
`--die-with-parent` tokens.

Preflight5 then launched correctly, created one physics child, and reaped both that child
and Supervisor V7.  Its raw physics wait status was zero, but the semantic result was the
reserved failure status `125`; no render child started and there was no NVIDIA-process
residue.  Before the task schedule, all eight USD articulation roots, root PhysX views,
and requested settings passed, while the deprecated Dynamic Control getter returned no
articulation candidate for any of the eight clone containers.  The exact frozen failure
is `SELF_COLLISION_DYNAMIC_CONTROL_FAIL`.  This is a startup/readback failure, not a
self-collision observation and not a side-grasp result.  Control flow never entered the
local `1..2340` task schedule; the terminal attestation deliberately leaves the claimed
physics-step count and authority absent because Kit startup activity existed outside
that task schedule.

Preflight6 performed the one activation frame exactly as preregistered, but the
deprecated Dynamic Control API still returned zero articulation objects for all eight
clones and aborted before `run_physics`.  Its exact 17-file prefix, frozen p16 v5,
Supervisor V8, failure, outcome and terminal attestation are immutable negative
instrumentation evidence.  In particular, task steps are exactly zero, the one
diagnostic step is not grasp evidence, render count is zero, exit is reserved `125`, and
promotion is false.

Preflight7 used the replacement behavioral gate in frozen p16 v6 and correctly reached
the pre-control identity audit before either diagnostic frame or the 2,340-step task.
It aborted because PhysX reported the one logical replicated filter as the concrete
representative `['/World/envs/env_0/Robot/link2']`, while the p7 gate incorrectly
required the authored regex/glob spelling.  The same audit independently resolved the
regex to the exact eight valid stage targets env0..env7.  Raw physics child exit zero was
mapped to reserved semantic status `125`; render count was zero, the supervisor and
child were reaped, no fresh NVIDIA PID appeared, and terminal attestation is valid with
`pass=false`, `promotion_allowed=false`, and no science/task-step claim.  This is an
observed replicated-view representation mismatch, not a self-contact or grasp result.

P16 v8 inherits p7's removal of the deprecated Dynamic Control query and both frozen
behavioral poses unchanged.  Its only reactive change is the pre-control logical-filter
identity rule: for each of the 15 pair views, `filter_paths` must be exactly the single
concrete env0 representative for the expected target body; the configured regex must
independently resolve to the exact ordered env0..envN-1 target set on the stage.  It does
not accept an arbitrary concrete path, regex text, glob text, or a partial resolver set.
Before any task step it
executes exactly two raw `env.sim.step(render=False)` diagnostic frames in the same
process.  The first frame commands `[0,0,165,90,0,45] deg`; frozen attempt3 convex hulls
and exact URDF FK must show exactly two overlaps, `link2__link4` (intersection inradius
`>=5 mm`; independently derived `5.652101947 mm`) and `link2__link5` (`>=5 mm`;
independently derived `6.371574011 mm`), and no other nonadjacent pair. Before either
diagnostic step, every one of the 15 self-pair rigid-contact views must independently
report `sensor_count=8`, `filter_count=1`, ordered env0..env7 subject paths, exactly one
env0 concrete filter representative equal to the expected target, regex stage
resolution exactly env0..env7, raw-count shape `(8,1)`, and actual buffer capacity
`8*256=2048`.
The scene configuration must read back strict Boolean `replicate_physics=True`,
`filter_collisions=True`, and `clone_in_fabric=False`. All eight clones must report
per-env raw count `1 <= count < 256`, total raw count strictly below the view's actual
`2048`-entry buffer, and force strictly `>0.02 N` for both expected pairs,
while the other 13 pairs, every moving-link/support pair, and every robot/object pair
remain raw zero and `<=1e-8 N`.  Pure geometry must also show moving-collider floor
clearance `>=71 mm` and cylinder separation `>=395 mm`, so those contacts cannot explain
the positive response.

The second frame commands exact HOME `[0,0,90,0,0,88.3099849635] deg`.  Frozen geometry
must have zero positive self intersections and a signed `link2__link4` separation margin
`<=-60 mm`; all 15 runtime self pairs, moving-link/support pairs and robot/object pairs
must be raw zero and `<=1e-8 N` in all eight clones.  Each phase must advance exactly one
Physics callback, SimulationManager step/time and SimulationContext step/time by
`0.005 s`, while all task counters remain zero.  Total diagnostic count is exactly two.
This proves only that the preregistered overlapping poses are sensed and HOME is clear;
it is not a proof of every possible pairwise manifold.
After writing HOME and before the negative step, direct root-PhysX DOF positions and
link transforms must equal the written HOME within `1e-7 rad`, show an empty overlap set
for all 15 pairs in every clone, and show an actual `link2__link4` separating margin
`>=60 mm`. After both diagnostics, the same 15 view identities and capacities are read
again and must equal the pre-control report exactly apart from the named epoch/clock.

Those diagnostic frames are never task samples.  Before task step 1, p16 must call
`env.reset()`, restore the exact cylinder centre/identity quaternion/zero velocity and
robot HOME/zero velocity, write HOME as the articulation position target, restore the
unchanged `100/5` stiffness/damping, call `scene.write_data_to_sim()` plus zero-physics
`sim.forward()`, reset every sensor, and clear every task counter/latch.  The re-baseline
gate reads actual DOF position, velocity, and position target from the root PhysX view;
object transform and velocity from the object root PhysX view; and private sensor
timestamps, invalidation flags, and force/contact buffers.  That rigid-body view uses
simulation-view subspace root `/`, so `get_transforms()` is gated in world coordinates:
the expected position is the already-written `OBJECT_CENTER_M + env_origin`, never the
env-local centre alone.  A pure eight-environment regression uses nonzero origins for
env1..7 and requires the world expectation to match all eight while the local
expectation fails exactly env1..7.  Cached tensors alone cannot pass the runtime gate.
The first task step must then demonstrate fresh counters and timestamps.
Because the global `SimulationManager` clock has no setter, its post-rebaseline value is
recorded as the task baseline rather than falsified to zero.  The unchanged task trace is
still local steps `1..2340` and exactly `11.700 s`; accounting reports diagnostic `2`,
task `2340`, and combined physical steps `2342` separately.

The no-retry wrapper must still be sent as one host-authorized `exec_command` with
`sandbox_permissions=require_escalated`.  Before consuming any G0 output path it also
checks the NVIDIA device, `nvidia-smi`, and frozen V11/prereg/candidate inputs. Supervisor
V11 repeats the ancestor rejection before writing any artifact and records PID-1 command
bytes/hash, the scoped self/own-PID namespace evidence, boot ID, PID=PGID=SID and no TTY
for terminal/canonical recomputation. After at least two monotonic shell seconds and
before ten seconds, the wrapper reruns an output-free v7 guard requiring the exact
PID/PGID/session, argv, full V11 contract and context. Any early exit,
mismatch or timeout sends TERM to the exact `$!`, waits/reaps for at most 20 seconds,
then sends KILL and performs one final bounded 20-second reap; unreaped state exits 7.
This is a reactive behavioral self-collision control/reset repair only: p16 v8's
scientific subject, task commands, thresholds, trajectory and contact gates are
unchanged from frozen preflight6.

Earlier, `t3u_side_preflight2` was a real failure-capable preflight, but it aborted before the
first task-physics step.  Its frozen `failure.json` reported
`SELF_COLLISION_READBACK_FAIL`: the old code read
`/World/envs/env_i/Robot`, while the composed articulation schemas and
`physxArticulation:enabledSelfCollisions` property are on the uniquely discovered
`/World/envs/env_i/Robot/root_joint`.  All old readback values were therefore `None`.
This proves an instrumentation path error only.  It does **not** prove that PhysX
self-collision was disabled, and it contains no grasp success/failure observation.
The frozen V4 supervisor correctly treated raw child exit zero plus a failure marker as
semantic failure, emitted reserved exit `125`, started no render child, and the terminal
attestation remained valid evidence with `pass=false`, `promotion_allowed=false`, and
no claimed physics-step count.

Preflight8 then executed both preregistered diagnostic frames successfully and generated
the five-row plan, but aborted before task step 1 because its re-baseline called
`env.reward_buf.zero_()`. Installed Isaac Lab 2.3 allocates the reset/counter buffers in
`DirectRLEnv.__init__`, while `reward_buf` is first assigned from `_get_rewards()` inside
`DirectRLEnv.step()`. The frozen p8 terminal/failure/plan evidence is pinned and means
diagnostic steps `2`, task steps `0`, render child `0`, and no grasp verdict.

This preflight9 tag inherits all earlier reactive instrumentation repairs byte-for-byte.  It does not change the cylinder,
candidate, jaw geometry, controller, trajectory, contact threshold, clearance threshold,
lift threshold, tilt threshold, friction placeholders, or success definition.  The
scientific variable count added by this tag is zero. Its runtime repair deletes only the
invalid pre-first-step `reward_buf` clear: re-baseline requires the attribute absent,
then the unchanged first task `env.step` requires its returned reward and newly created
`env.reward_buf` to have shape `(N,)`, be finite/all-zero, and be tensor-equal. The
durable self-contact filter validator additionally ignores only JSON object key order:
the exact 15-key set, length, types, and every row semantic remain mandatory, so missing
or mutated rows fail. The failure-capable experiment
already performed in this research session is preflight2; preflight9 may run only after
the new bytes receive an independent static GO.

Immutable predecessor evidence (historical/non-promotable only):

- frozen preflight2 p16 source: `5c6132b68651549b2c54c9216a09ecfb4210e9b74ee1c3ba9ddf96f667dcf789`;
- frozen preflight2 supervisor: `527b06e5b9a090f4207c5f9ac5feb539c4b26f4c23f48ac59e4d802a153fa365`;
- `t3u_side_preflight2_prereg.md`: `e02b927edc493f4912ad9dbc5c9bd5713e4181c4e6512f0d61e50c62328bf329`;
- `t3u_side_preflight2_failure.json`: `f17e0c3a3f48c9a52ffea572b52957164b8e0adb54af1d2c9cbfe766ce88c4a3`;
- `t3u_side_preflight2_supervisor_outcome.json`: `443dd6a18ef7a0074a0ca04c64a3a6bcf55711991f403d4dea4ef9e733b56210`;
- `t3u_side_preflight2_phase.jsonl`: `010ae83487eb2cac6fc496ed9070cbe90242a3a8dd6f5079e90c93eb18e20ccb`;
- `t3u_side_preflight2_exit_status.txt`: `a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca`;
- `t3u_side_preflight2_terminal_attestation.json`: `6fbab4dc67a800d7a3d649fc4bf72fea2ad3dbffe5a57961a0284e96c923c58b`;
- retired old canonical prereg `t3u_side_phys1_prereg.md`:
  `c52a31bddf6cfd64700074c66d0b6c1d43736379f37c581842334ce06819bbb2`.
- frozen preflight3 p16 v2 source:
  `b9f987eef7f62527a64a80900a9811e73eea7a8d02885e2e820192af456f64ac`;
- frozen preflight3 Supervisor V5:
  `998865694378509549841cac6fd1d486d49abf1ef8f53a5d74d423657213db5d`;
- `t3u_side_preflight3_prereg.md` (22,626 bytes):
  `4c5a068c28f54e5ba13313c55cac350f6aaff38fe10d52db7451dc962b5067a0`;
- retired `t3u_side_phys1_preflight3_prereg.md`:
  `b1b20f9e8eee24950f53c663f3712d787f77ac697cb66ada87b0502b17c51faf`;
- the original preflight3 launch's only generated run-prefix artifact,
  zero-byte `t3u_side_preflight3_supervisor_launcher.log`:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- posthoc audit-contamination `t3u_side_preflight3_supervisor_failure.json`:
  `218ec29911134acaca1d472762fa27341f87fed136bd39849099c2eeca35ebcc`
  (3,074 bytes; not original launch authority, not science).
- frozen preflight4 p16 v3 source:
  `f03561858e12841d4b3eef3047083d69e96791136dbaa8e76bc0e9eb178e1d2a`;
- frozen preflight4 Supervisor V6:
  `40f46f3f94bf1926294831e4d41106b98fb9b69efd1cdb82d977e6be899f0f2f`;
- `t3u_side_preflight4_prereg.md`:
  `6b413e343630cbac6dbec458769aac9310c9caea3cfedfb436d0f3582ac2ea13`;
- retired `t3u_side_phys1_preflight4_prereg.md`:
  `6ccc5616d35abd8863c7bf48dc005cb7e058daf32414fd51df65d7f08a46466f`;
- preflight4 `/proc/1/ns/pid` failure and matching launcher:
  `50cd5e0eec3444e44862dc0885137389c8073decbfdf7fbbe8d2a55b8bbf66b5`
  and `3b37b2967c6dcb702f71dde28a8c3dd1d2069a7ec7f15a650f91667096bca2e9`.
- frozen preflight5 p16 v4 source:
  `f019d55b437c93e53a2f6820af633821765c24a8741cd170fe3b4d189dc4a4ad`;
- frozen preflight5 Supervisor V7:
  `b344b49fb955a833ef4eee92c48f4ef7cf95ffdda4e4cef58cd806a681d15fcd`;
- `t3u_side_preflight5_prereg.md`:
  `319376d827f92355a51c71a0397f3aeace6f6a70c4ce4c3a41a8d8e7aa3c349b`;
- retired `t3u_side_phys1_preflight5_prereg.md`:
  `9415c0703897c1d3548c2db126c6a285e4c3418032fb71b6c973e5b9d4bb6e44`;
- preflight5 failure, supervisor outcome, phase, exit status, and terminal abort
  attestation respectively:
  `0a051340dc4b448032fa4ebceee7927229497e4d5799502fbf90c71604746b5b`,
  `cd44a132735d001b05f4c93bf9c9bdf05c76cded8a877b86d9f87570a24191d6`,
  `ea99ff199f00f3fe28fc0d0dfd28655c8163c0d410f178cd716ce43a89df8d76`,
  `a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca`,
  and `d99a0f19d946d149d3307134cd79305b4ba5a1858662758d5a867717dbf9a84e`.
- frozen preflight6 p16 v5 source and Supervisor V8:
  `b6eb67cbec8e11752b926d8d04498c3d29fd993b8ac87b5aabbc207c92d06458` and
  `8cd7946b7dfb826a2fce8a9a9580603a945037aa48aa591d8979fc58ba03d9b2`;
- preflight6 prereg and retired canonical prereg:
  `198c81869ff8a547edb5bbc497e0c080864b39cf7ae47db676a03ba7d5028375` and
  `9dcaeee6840edeea81b0e7b7a1b92aa2415f57f03c1173be921692dda7556cc0`;
- preflight6 failure/outcome/phase/exit/terminal attestation:
  `43e086551ee54063795fd915d5fa8c0dfd927855090928ca69f2518f609ab245`,
  `da8161d632b6da1ba48e8a6c25a3cca240461bfd03b4f6fefc8c6561793adbe7`,
  `ce6771980c37ab13dc4ed7f5ff52348be34d00ed750c14c2492ae73c274143d4`,
  `a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca`,
  and `3d13364e2cb3113c69485e31aa12f30e1403ccef323852eb12bf58d464094d08`.
  Its stdout hash is the directly measured
  `50476db40a8c03594d09a2091ef8c01100afae196bc602a280146d933d94b1cf`;
  the earlier handoff spelling with a leading `b` was not a file hash and is rejected.
- frozen preflight7 p16 v6 source and Supervisor V9:
  `aabac6c76985682e32376195d187134da028bb6cc768148e883fbd56c18b3dbe` and
  `9f1cf1be075fe052f8d2db196be9a14207d80dc82f46a58a364a88513bacb716`;
- preflight7 prereg and retired canonical prereg:
  `c2b17be775ad4df465c967d3cbdf08c571eed3cfbd21325265798a906b0d6e96` and
  `1453c33642b5d32e2e24dba66da5732240afc6484af7142ddc7953c97b1efbbd`;
- preflight7 failure/outcome/phase/exit/terminal attestation:
  `3b84af93ab399725a2fab220d9dd5883d6f5286bfa0241c63bec04b15d5bc01a`,
  `26e161620c725c45e24beaf163fe8221bff061ffbf667ea2fac811259bda64f0`,
  `43788a37e63e000bf00d30346f6fdef9d152eef51041ca1ea604420cb028e2de`,
  `a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca`,
  and `41a65b75453e4f97a326ec8ae4e966a09070812a1e8374696f3a46f421e4a8ea`.
  Its exact 17-file prefix is immutable, contains no science artifact, and is pinned as
  the non-promotable predecessor that motivates only this representation repair.

Those files are historical inputs only and must never be overwritten, renamed, repaired,
used as science artifacts, or used as a `pass=true` promotion condition.

## 2. Frozen physical subject inherited without relaxation

Except for the six reactive instrumentation changes in sections 1 and 3 through 7, sections
2 through 7 of `t3u_side_preflight1_prereg.md` remain the physical and measurement
contract:

- fixed-base RoArm attempt3 asset; actual 64+64 enabled convex-hull jaw parts;
- upright analytic cylinder `D=0.029 m`, `H=0.050 m`, `mass=0.02483 kg`, centre
  `[0.4235072423787768,0.17237803311822986,0.025] m`;
- placeholder cylinder material static/dynamic/restitution `0.40/0.30/0.0`, with no
  effective-pair or real-friction claim;
- p15 side_sdg2 candidate index 5, ID `side_sdg_005_raw_025092`, crossed with all five
  frozen pinch offsets: exactly five planned and five active rows in eight environments;
- link5 `+X` closure, `+Y` world-up, `+Z` radial approach; physical pinch-centre offset
  `+4.474150434 mm`; q5 open `88.3099849635 deg`, sole close command `22 deg`;
- HOME -> elevated outward pregrasp -> near-side staging -> final horizontal midpoint ->
  close -> hold -> vertical lift, exactly 2,340 samples at 200 Hz;
- transformed enabled-collider minimum support clearance `>=1.0 mm` at every sample;
  final adverse pitch `<=1 deg`; no attach; parsed true URDF limits; full fixed-base,
  applied-target, reporter, support, non-jaw/object and all 15 nonadjacent self-contact
  gates;
- same-step bilateral jaw force strictly `>0.01 N`, collision/contact gate `>0.02 N`,
  corrected object lift strictly `>6 mm`, and final tilt strictly below the unchanged
  cylinder tip angle;
- full-step authoritative NPZ plus RRD/RBL/two PNGs; render is isolated, post-hoc,
  zero-physics and produces an exact frame manifest plus MP4.

Preflight3 through preflight7 remain non-scientific. Even a fully valid preflight9 may promote only
instrumentation readiness, never a side-grasp conclusion.

## 3. Reactive repair A — authored setting plus behavioral positive/negative proof

The old `/Robot` attribute-path assumption is removed. Before any task-physics sample,
p16 v8 performs one fail-closed contract with five mutually bound parts:

1. It opens the pinned attempt3 source USD and finds exactly one
   `UsdPhysics.ArticulationRootAPI` prim at suffix `/root_joint`. The same prim must carry
   `PhysxSchema.PhysxArticulationAPI`, a typed Bool attribute with an authored opinion,
   and the pinned `configuration/roarm_m3_physics.usd` property spec must explicitly
   author strict Boolean `False`. This distinguishes the source setting from the schema
   fallback, whose default is itself `True`.
2. For each clone container `/World/envs/env_i/Robot`, it traverses the subtree and
   discovers exactly one articulation-root prim at pinned relative suffix `/root_joint`.
   The same two APIs must be applied. The composed Bool must resolve to strict Python
   `True`, have an authored opinion, and its strongest authored default must be explicit
   `True` at a stronger property-stack index than the exact pinned source-layer `False`.
   A stronger spec need not live in an env_i layer because clone inheritance may resolve
   through env0; spec-layer paths are recorded rather than hard-coded.
3. Isaac Lab's actual `root_physx_view` must have a non-null backend, `check() is True`,
   exactly the requested clone count, and an ordered `prim_paths` list identical to the
   roots discovered in step 2.
4. Before any diagnostic step, all 15 actual self-pair rigid-contact views and the three
   explicit scene-cloning flags pass the exact identity/count/path/shape/capacity gate.
   The identical view contract is reread after the controls and before task rebaseline.
5. Pure frozen geometry evaluates all 15 nonadjacent pairs at the registered positive
   pose and HOME.  The positive pair set must be exactly
   `{link2__link4, link2__link5}`, both inradii must be `>=5 mm`, moving geometry must be
   `>=71 mm` above support and `>=395 mm` from the cylinder. HOME must have no overlap
   and `link2__link4` signed separation `<=-60 mm`.
6. Exactly two same-process raw PhysX frames then supply the behavioral check described
   in section 1: both expected pairs contact in every clone at the positive pose, all
   other pairs and support/object channels are clear, then all channels are clear at
   HOME. The runtime articulation body transforms independently re-evaluate both
   expected inradii `>=5 mm`; after HOME is written and before its step, direct PhysX
   DOF/link transforms must show all 15 pairs non-overlapping and `link2__link4 >=60 mm`
   in every clone. Positive raw counts must stay strictly below both the per-prim and
   actual total view capacities.

Missing/multiple roots, wrong suffix, missing PhysX API, fallback-only value, missing
authored opinion, Boolean alias, source-layer mismatch, stronger `False`, PhysX-view
identity/count failure, geometry-set drift, an absent/unexpected diagnostic contact,
support/object contamination, clock/counter drift, or failed HOME negative control
aborts before the first task step as instrumentation invalidity. The authored setting
alone remains non-proof. The two registered behavioral poses test that the active
sensor/control path can distinguish overlap from clearance; the unchanged 2,340-step
tensors for all 15 nonadjacent pairs remain the scientific-task evidence authority.

## 4. Reactive repair B — exact URDF clearance FK and two distinct pose audits

The transformed-collider clearance authority is now parsed directly from the frozen
`local_assets/roarm_m3/urdf/roarm_m3.urdf` bytes (SHA-256
`64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2`).
P16 verifies the exact ordered joint names, parent/child links, joint types, `+Z` axes,
and decimal origin xyz/RPY values, then evaluates all 2,340 command samples with that
chain.  P10 remains the IK solver only.  Its q=0 delta from exact URDF is serialized as a
diagnostic and can never authorize a clearance transform.

Two different observations must not be conflated:

1. **Authored/rest audit.** `UsdGeom` default-time body transforms and every composed
   revolute joint's local frames are used to recover the authored joint coordinates.
   These must be q=0 within `2e-5 deg`, have pure-axis residual `<=5e-7`, and each of the
   six moving bodies must match exact-decimal URDF q=0 within the unchanged
   `1e-6 m` / `1e-6` rotation-matrix gates.  These fields must say authored/rest, never
   current PhysX or HOME.
2. **Same-epoch runtime audit.** In one articulation data timestamp, p16 reads actual
   `joint_pos`, `body_pos_w`, and `body_quat_w` for every clone.  Exact-decimal URDF FK at
   each observed joint vector must match the corresponding actual body pose, after
   subtracting that clone's environment origin, within `5e-6 m`, `1e-5` rotation-matrix
   component, and `1e-6` quaternion-norm error.  Tensor shapes, finiteness, body/joint
   order, and unchanged timestamp are hard gates.

Failure of either audit aborts before any task step.  It is instrumentation invalidity,
not desk collision or grasp failure.  No old tolerance was widened: the first audit
retains the original `1e-6` gates; the second is a new independent float-runtime
readback gate.

As an independent no-Kit/no-PhysX countercheck before freeze, the actual composed USD
joint local frames were also evaluated at every one of the five planned command
schedules: `5 x 2,340 x 6 = 70,200` body transforms.  Exact URDF FK versus composed
`L0 * Rz(q) * inverse(L1)` differed by at most `2.4063157388066466e-08 m` in translation
and `1.6798151730723632e-07` in a rotation-matrix component; all six `L1` transforms were
identity.  This demonstrates the q-nonzero motion-map equivalence for the frozen bytes;
it is a static derivation, not physics evidence or a relaxed runtime threshold.

## 5. Reactive repair C — raw exit zero is not physics success

Preflight1 proved that `SimulationApp.close()` can terminate with raw wait status zero
after p16 has already written a failure marker. Supervisor V11 therefore defines physics
semantic success as all of the following, recomputed after the child group is reaped:

- raw ordinary exit zero, no timeout/signal action, and empty child process group;
- `failure.json` absent;
- nonempty `results.json`, `plan.json`, `trace.npz`, RRD, RBL, Rerun validation,
  decision PNG, inspection PNG, frozen source, argv, phase log, and preclose sentinel;
- result/plan/profile/argv/source identity exact;
- sentinel binds result, NPZ, Rerun validation, source, p15 and tag hashes;
- result binds plan, NPZ, RRD, RBL, validation, and both PNG hashes;
- exactly one ordered `run_claim -> results_durable -> preclose_sentinel_durable ->
  simulation_app_close_start` sequence (optionally followed only by
  `simulation_app_close_returned`), with an exact key set for every row, finite
  nondecreasing timestamps, `failure_marker_exists=false`, and source/prereg/p15/result/
  sentinel/internal-verdict fields recomputed from the actual files;
- exactly one complete stdout line reconstructed from the result rather than token
  searched: profile, `scientific_verdict_preclose_candidate`, Boolean success count,
  and the frozen active denominator (`5` here, `10` canonical) must all match byte for
  byte.

Only this combined gate starts the render child.  If raw status is zero but the semantic
gate fails, render attempt count is zero and combined exit status is reserved value
`125`; a nonzero raw child status remains its own normalized value.  The outcome embeds
the complete `T3U_PHYSICS_PRECLOSE_SEMANTIC_GATE_V1` recomputation.  Terminal attestation
and canonical promotion independently rerun the same pinned pure verifier; they do not
trust its stored boolean.

## 6. Reactive repair D — raw exit zero is not render success

The same Kit close behavior can occur in the isolated render child.  Render success is
therefore no longer raw wait status alone.  After the render child group is reaped,
Supervisor V11 independently loads p16 v8 only as a pure file validator and requires:

- ordinary raw exit zero, no timeout/signal action, and an empty child process group;
- no `failure.json`, plus nonempty result/plan/NPZ/frozen-source/phase/manifest/MP4;
- exactly 234 uniquely named `frame_0000.png` through `frame_0233.png` files, each bound
  to its manifest row, frozen NPZ sample, clock snapshot, body/object transform and joint
  state;
- the full manifest schema and exact 20 fps cadence; then an independent supervisor-side
  decode of all 234 PNGs and every MP4 frame, checking PNG format/mode/resolution and
  MP4 count/frame-byte-length/resolution/fps in addition to hashes; zero observed physics
  callbacks/clock delta/scenes/explicit step calls; and physics-finalize ->
  render-start -> render-end dependency hashes all independently recomputed;
- the authoritative NPZ cadence is independently regenerated as physics steps
  `1..2340`, simulation time `step/200`, and the exact seven phase-id/phase-step arrays;
  the 234 rendered samples must therefore be exactly steps `10,20,...,2340`;
- the clock-audit ledger has exactly 491 ordered labels: one `context.new_stage`, 250
  `app.update`, and 240 Replicator calls, including the six warm-ups and all 234 frames.
  A shortened but internally self-consistent audit list is invalid.  Every count/step
  field is a strict JSON integer (Boolean forbidden), and every time/delta/error/fps
  field is a finite JSON float (Boolean forbidden), including all 491 before/after
  snapshots, frame indices/phase IDs/fidelity values, the render phase, and independent
  decode counts/size/fps;
- an immutable physics `phase.jsonl` ending at SimulationApp close, plus a separate
  one-row `render_phase.jsonl` containing exactly `render_trace_durable` and binding the
  manifest/MP4 hashes; and one profile/count/path-exact `RENDER_TRACE_COMPLETE` record.

The outcome embeds `T3U_RENDER_POSTHOC_SEMANTIC_GATE_V1`.  A raw-zero render missing or
contradicting any evidence becomes reserved combined status `125`, never supervisor
PASS.  A nonzero raw render status remains its normalized value.  Terminal attestation
and canonical promotion rerun the pinned pure render verifier and demand exact equality
with the stored gate; neither trusts `manifest.pass`, `outcome.pass`, or a stored map.
All supervisor-side `nvidia-smi` probes are bounded by 15 seconds.  The independent
PNG/MP4 decode is a CPU semantic check after the render child has exited; that decode is
not subject to the 15-second `nvidia-smi` timeout.  Its paired ffprobe metadata check
and ffmpeg RGB framehash decode each have a separate fixed 120-second timeout.  The
supervisor rechecks its TERM/INT flag immediately after each physics/render semantic
gate and before spawning render or writing a successful outcome; a signal observed
during a bounded decoder can therefore never produce combined PASS.

## 7. Reactive repair E — useful terminal evidence for an aborted run

If preflight9 aborts before science artifacts exist, `--terminal_attest
side_preflight9` takes the upstream-failure branch before scanning for successful-run
artifacts.  It requires and binds the failure marker, frozen source/argv/phase, raw child
wait status, semantic-gate failure, reserved nonzero combined status, render-not-started,
PID/PGID cleanup, supervisor contract/outcome, file hashes, and GPU before/end/after
inventories.  `nvidia-smi` has a 15-second hard timeout.

On valid failure evidence it writes forward-only
`t3u_side_preflight9_terminal_attestation.json` with artifact
`T3U_EXTERNAL_TERMINAL_ABORT_ATTESTATION_V2`, `attestation_valid=true`,
`promotion_allowed=false`, and `pass=false`.  Missing NPZ/RRD/PNG/MP4 is then
reported as absent science, not as an attestor crash and not as zero physics steps.
This failure attestation can never satisfy the canonical promotion schema.

If physics preclose is complete but rendering fails, p16 writes forward-only
`render_failure.json` plus a separate `render_failure` render-phase record before
calling `SimulationApp.close()` when
an exception is observable.  Supervisor semantic failure remains the independent
authority even if Kit suppresses that Python exception or a completed-looking manifest
fails recomputation.  The terminal verifier separately rechecks both child raw wait
records, the passing physics gate, failing render gate, exact phase/failure evidence,
reserved-or-raw nonzero combined status, bindings, process groups and GPU inventories.
Valid evidence produces `T3U_EXTERNAL_TERMINAL_RENDER_ABORT_ATTESTATION_V1` with
`pass=false`, `scientific_artifacts_complete=false`, and promotion forbidden.  It can
never be mistaken for the successful `T3U_EXTERNAL_TERMINAL_ATTESTATION_V4` schema.
If close/finally prevents the child from writing a render failure row, exact absence of
both render failure files is accepted only together with the recomputed failing render
gate and nonzero/reserved supervisor outcome; absence can never become render PASS.
All pure JSON/phase/raw-wait/binding and GPU-text parsing is completed before the
one-shot `nvidia_smi_after.csv` file is created.
Keeping render failure/phase bytes separate is essential: it leaves the already-passing
physics failure marker and phase hash byte-identical, so terminal recomputation can
compare the pre-render physics gate exactly rather than pretending later appends existed.

Every successful-run, physics-abort, render-abort, and canonical-promotion verifier uses
one shared strict raw-lifecycle decoder.  PID/PGID/SID are JSON integers greater than
one; attempt counts, raw wait status, exit/signal values and combined status are exact
non-Boolean integers; timestamps/durations are finite JSON floats in monotonic order;
TTY, timeout, reap and pass fields are exact Booleans.  The decoder reconstructs
`waitpid` exit/signal meaning rather than trusting stored decoded fields, and validates
the exact child command, process group, session, cleanup actions, GPU PID inventories,
and supervisor identity.  Recursive JSON comparisons require both type and value to
match, so `false` can never alias integer zero and `true` can never alias integer one in
the V11 supervisor contract, stored semantic gates, outcome, or terminal attestation.
Canonical promotion reruns this same verifier and a second direct raw-lifecycle check.
The V11 contract and outcome bind the strict host context: terminal and canonical decode
PID-1 raw command bytes/SHA, require no forbidden sandbox token, validate the scoped
self/own-PID namespace device/inode/readlink fields and explicit non-host-proof label,
require a valid boot ID, and demand type-and-value-identical context copies. They do not
claim or attempt a PID-1 namespace comparison.

## 8. Forward-only outputs and detached execution

All outputs use only `t3u_side_preflight9_*` under
`claudedocs/runtime_logs/grasp_track/g0b_d420/`.  Existing preflight1/p15/p14/p10 files
are read-only.  The supervisor performs one physics attempt and at most one render
attempt, with no retry or tag reuse.

Exact launch block after independent static GO follows.  This entire block is submitted
as **one** `exec_command` with `sandbox_permissions="require_escalated"`; executing the
same shell text in the default sandbox is forbidden.  The host/device/hash guard runs
before redirection, so a bad launch context cannot consume the forward-only tag.

```bash
(
  set -euo pipefail
  set -o noclobber
  p16_repo=/home/cgxr/Documents/Robotics/RoArm_Project
  p16_case_dir=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/g0b_d420
  p16_profile=side_preflight9
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
    --prelaunch_guard side_preflight9 >/dev/null
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
        --launch_liveness_guard side_preflight9 \
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

After the recorded supervisor and child group are reaped, run exactly:

```bash
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  /home/cgxr/Documents/Robotics/RoArm_Project/sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v8.py \
  --terminal_attest side_preflight9
```

The external terminal top-level artifact name intentionally remains the already-audited
`T3U_EXTERNAL_TERMINAL_ATTESTATION_V4`; its nested supervisor contract/outcome must be
V11 and host-context schema V2. If and only if that attestation passes, the user manually opens both PNGs,
the RRD, and the full MP4 and writes a hash-bound
`t3u_side_preflight9_manual_visual_inspection.json`. Canonical remains blocked until all
successful-run terminal and manual fields are independently recomputed and PASS.

## 9. Frozen dependency pins

- supervisor source SHA-256:
  `bf36d2acbe9cb7fb6cb6721e8dca90ce4f7623fab5813449f459893131fae29b` at
  `sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v11.py`;
- executable p16 source path:
  `sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v8.py`;
- p15 source: `250a3f406f83d3b0cc95be7ccdc666d043e28eb5b5c0f9fb25e450e26ee17240`;
- p15 prereg: `23acb036cd1a26f577cff8145ef4031f1c4075af3e4e60f1df28a42d86da8330`;
- p15 candidates: `67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384`;
- p14 helper: `fcaa7b1c6aeea65cd7fd335d9cd17ee5424a53d81764f67642d074a28e3e0133`;
- p10 IK/planner: `63c6b2127d969e3291da6943eab6da1037034c154a8f21fe447519cbcb2f6cff`;
- attempt3 jaw extractor: `bca4f898023f63f21d540483a169499760038c582ce3a7919d7622e77946e1c3`;
- frozen URDF: `64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2`.

P16 pins the final full SHA-256 of this prereg and the canonical prereg.  It records its
own executed-source hash rather than embedding a self-reference.  One profile-specific
path function defines the complete decision dependency set: p16 source/supervisor and
both preregs, p10/p14/p15 and every bound p15 config/output/inspection/manual/exit/stdout/
PID file, the preflight1 historical witnesses, workspace witnesses, environment sources,
URDF, jaw extractor, all five attempt3 USD layers, and every immutable preflight2/3/4/5/6/7
predecessor file/hash listed in section 1. Canonical adds every passing preflight9
science/render/lifecycle/manual file. The exact key set and hashes must match
physics start, physics finalize, render start, render end, and terminal-time current
rehash; editing or adding a failure marker between these boundaries aborts.
