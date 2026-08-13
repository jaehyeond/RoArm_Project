# t3y_workspace_preflight2 — jaw/support sensors + built-in MDL resolver repair

Status: **PREREGISTERED / NOT RUN**
Case: `g0b_d420`
Canonical preflight tag: `t3y_workspace_preflight2`
Date: 2026-08-11 KST

## 1. Why this is a new forward-only preflight

`workspace_preflight1` planned 384 trials and found 215 IK-feasible trials, then
failed before environment construction.  `pxr.UsdUtils.ComputeAllDependencies`
opened all five pinned local USD layers but returned NVIDIA's built-in visual
material identifier `OmniPBR.mdl` as unresolved.  p14 incorrectly treated every
unresolved item as a missing USD file.  The Kit log then entered
`SimulationApp.close` at about 4.96 s and stopped after framework-release began; the
detached PID was terminated after it remained blocked.  There are no preflight1
environment, batch, result, RRD, or inspection artifacts.

Evidence:

- `t3y_workspace_preflight1_plan.json` (`n_planned=384`, `n_feasible=215`)
- `t3y_workspace_preflight1_stdout.log` (the `OmniPBR.mdl` warning)
- Isaac Kit log
  `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20260811_165430.log`
  (five local layers opened, repeated built-in MDL warnings, cleanup start)

All preflight1 files are immutable failure evidence.  This protocol creates only
the new `t3y_workspace_preflight2_*` namespace.  p14 explicitly rejects the retired
`workspace_preflight1` run label.

This remains a small, failure-capable **instrumentation preflight**, not a scientific
workspace experiment.  이번 preflight의 신규 연구 변수: `[]` (runtime provenance
classification repair only).  It does not change object pose sampling, IK, PhysX,
contact gates, jaw geometry, friction authorship, or the preflight1 schedule.

## 2. Exact MDL classification repair; no USD gate relaxation

The exact five local composed USD layers and their full SHA-256 values remain those
in `t3y_workspace1_prereg.md`.  `ComputeAllDependencies` must still discover exactly
that five-layer set.  Every local layer is full-SHA-checked before planning, after
physics, and after Rerun.  A missing, extra, changed, or unresolved `.usd`, `.usda`,
or `.usdc` remains an immediate FAIL.

The only admissible unresolved normalized identifier set is exactly:

```text
{OmniPBR.mdl}
```

Normalization uses `Sdf.AssetPath.path`; the fallback removes only USD's outer
`@...@` display delimiters.  It does not use basename extraction, path resolution,
case folding, URL decoding, or search-path substitution.  Therefore
`materials/OmniPBR.mdl`, any other `.mdl`, an unknown unresolved item, and any mixture
of `OmniPBR.mdl` with a missing USD are fatal.  Raw and normalized records, duplicates,
classification, and the exact-set result are serialized.

The exception is accepted only if all semantic checks pass:

1. `Tf.GetEnvSetting("OMNI_USD_RESOLVER_MDL_BUILTIN_PATHS")` contains the exact
   identifier `OmniPBR.mdl`; its module set equals the process environment set.
2. The installed `OMNI_USD_RESOLVER_MDL_BUILTIN_BYPASS` configuration is exactly the
   string `1`.  This is recorded as configuration, not interpreted as an
   enable/disable claim because NVIDIA's installed comment and variable name are
   semantically inverted.
3. OpenUSD version, Ar resolver type, and enabled extension IDs/versions are recorded;
   OpenUSD must be exactly `0.24.5`, and extensions must be
   `omni.usd=1.13.10`, `omni.usd.config=1.0.6`,
   `omni.usd.libs=1.0.1`, and `omni.usd_resolver=1.0.0`.
4. For each of the five pinned layers, true composition arcs are enumerated directly
   from `subLayerPaths` and every authored `PrimSpec` reference/payload list-op bucket
   (explicit/added/prepended/appended/deleted/ordered).  Their
   nonempty paths must equal `Sdf.Layer.GetCompositionAssetDependencies()`.  No MDL
   identifier may appear as a sublayer, reference, or payload.
5. Full Sdf-spec traversal must find exactly eight authored MDL asset attributes, all
   in `configuration/roarm_m3_base.usd`, all named
   `info:mdl:sourceAsset`, typed `asset`, owned by `Shader` prims, with exact value
   `OmniPBR.mdl` and empty file-resolution path.  The five layers' ASCII export count
   of `@OmniPBR.mdl@` must equal those eight records (base 8, every other layer 0).
6. Every layer containing an authored record is opened as its own `Usd.Stage`; all
   eight prim paths must resolve through `UsdShade.Shader.Get`, and
   `GetSourceAsset("mdl")` must equal that record's exact `OmniPBR.mdl` identifier.
   Opening the authoring base layer (rather than only the robot root composition)
   keeps all eight sibling shader specs visible to the UsdShade API check.

No USD is edited, no resolver search path is changed, `OmniPBR.mdl` is not counted as
a sixth USD layer, and there is no blanket `.mdl` ignore.  Installed NVIDIA evidence
for this interpretation is:

- `omni.usd.config-1.0.6.../omni/usd_config/extension.py:97-99,159-166`
  (configuration and built-in module enumeration)
- `omni.usd-1.13.10.../omni/usd/tests/test_usd_bootstrap.py:25-37`
  (`OmniPBR.mdl` membership check through `Tf.GetEnvSetting`)
- `omni.usd_resolver-1.0.0.../config/extension.toml:1-16`
  (resolver extension identity/version)

## 3. Frozen inputs and exact invocation

All p10/p13/package/source/64+64 hull/contact contracts in
`t3y_workspace_preflight1_prereg.md` remain in force.  In particular the full p13
result SHA is:

`d1460c9d80e5f65f4ab9d85a7851b29876ef7ae0ca9e28d34bd93ddb91f0170a`

Exact preflight2 controls:

- `num_envs=128`, `grid_side=2`, `plan_workers=8`
- steps: settle 120, approach 300, descend 500, close 30, hold 20, lift 30
- `settle_stat_tail=60`, `episode_length_s=120.0`, `contact_capacity=256`
- `descend_open_deg=88.30998496351378`, `approach_clearance_m=0.040`,
  `lift_delta_m=0.025`, IK gates `0.003 m / 5.0 deg`
- cylinder-authored static/dynamic coefficients `0.40/0.30`; gains `100.0/5.0`
- GPU capacities: found/lost pairs `2^23`, aggregate pairs `2^23`, collision stack
  `2^28`, max rigid contacts `2^23`

The protocol path, full protocol SHA, p13 path/SHA, run label, and every value above
are hard compared.  Unregistered labels are rejected.  The protocol SHA placeholder
below must be replaced only at launch with p14's frozen
`PREFLIGHT2_PREREG_SHA256` constant.

## 4. Instrumentation PASS/FAIL contract

All eleven runtime checks in preflight1 Section 3 remain mandatory: exact
object/jaw reporter inventory and shapes, all-clone threshold-zero audit, raw-buffer
non-saturation, per-environment raw counts, full trace, jaw-support task-failure
classification, frame-graph/RRD contract, known S3/S4 high-tilt same-jaw force+raw
witness above 0.02 N, all population/replay positive controls, replay gate-class
equality, and source/dependency stability.

Preflight2 adds five explicit PASS fields: exact five local USDs, exact unresolved
MDL allowset, runtime built-in membership/configuration, absence of MDL composition
arcs, and exclusive authored/visible UsdShade sourceAsset semantics.  Any one failing
causes `INSTRUMENTATION_PREFLIGHT_FAIL` or a pre-result failure marker.  Pure startup
regressions additionally require:

- exact `OmniPBR.mdl` -> allowed;
- another MDL and path-qualified same basename -> fatal;
- missing USD -> fatal;
- `OmniPBR.mdl` mixed with missing USD -> fatal;
- `OmniPBR.mdl` mixed with another MDL -> fatal.

p14 follows the durable D367/D375 terminal lifecycle contract.  It first requires
`env.close()` to return without exception.  It then finalizes and fsyncs the trace,
RRD, RBL, validation, both PNGs, frozen script/argv and result.  The result is not a
terminal PASS: it records `cleanup.pass=false`, the environment close as returned,
`SimulationApp.close_attempted=false`, and exact internal lifecycle verdict
`PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL`.  It also leaves
`scientific_verdict=null`.

Next p14 writes and fsyncs `preclose_sentinel.json`, which binds the full result SHA
and byte count, the complete artifact SHA/size manifest, source/protocol/p13 hashes,
the environment-close state, and a SHA/byte-count prefix of `phase.jsonl`.  Only after
that durable sentinel exists does p14 append exactly one
`simulation_app_close_start` phase and call `SimulationApp.close()` as its last normal
Python call.  On this installed Isaac Sim 5.1 stack, normal framework release may end
the process without returning, so a post-return success marker is neither expected
nor required.  A return or raised `BaseException` writes `failure.json` and forces a
nonzero path.  Any earlier exception also writes the marker before interpreter
teardown; after the environment-close attempt, the top-level failure handler appends a
failure-path close-start and still attempts the same graceful terminal app close.  A
failure marker is fatal even if that terminal framework cleanup masks the primary
exception with raw exit status 0.  If the close returns or raises, the handler forces a
nonzero path.  No `skip_cleanup` path exists.

Terminal completion belongs only to the external supervisor/attestor.  A pure
regression rejects a false post-return cleanup PASS, an env-close failure, a missing
app, a mutated result SHA, a mutated artifact manifest, and any premature internal
terminal PASS.  This repairs the locally observed D367 category error: installed
`SimulationApp.close()` reaches `shutdown_and_release_framework()` and may terminate
Python before a post-return write.

As before, results must set `scientific_authoritative=false` and
`scientific_verdict=null`; RRD/PNG/result scope is
`INSTRUMENTATION_PREFLIGHT_ONLY`.  Only (a) internal checks all true, (b) external
terminal attestation PASS, and (c) actual inspection of the generated PNG is technical
GO for canonical `workspace1`.

## 5. Detached launch

```bash
conda activate isaaclab
(
set -o errexit
set -o pipefail
set -o noclobber
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_stdout.log
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_supervisor_pid.txt
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_python_pid.txt
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_pgid.txt
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_supervisor_contract.json
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_exit_status.txt
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_nvidia_smi_before.csv
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_nvidia_smi_after.csv
test ! -e claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_terminal_attestation.json
printf '%s\n' '{"artifact":"T3Y_EXTERNAL_TIMEOUT_SUPERVISOR_V1","automatic_retry_count":0,"foreground":true,"kill_after_seconds":20,"preserve_status":false,"term_signal":"TERM","timeout_seconds":7200}' \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_supervisor_contract.json
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_nvidia_smi_before.csv
nohup setsid bash -c '
  set +e
  set -o noclobber
  t3y_pf2_supervisor_pid=$$
  t3y_pf2_pgid="$(ps -o pgid= -p "$t3y_pf2_supervisor_pid" | tr -d "[:space:]")"
  if test -z "$t3y_pf2_pgid" || test "$t3y_pf2_pgid" != "$t3y_pf2_supervisor_pid"; then
    printf "setsid self-audit failed: pid=%s pgid=%s\n" \
      "$t3y_pf2_supervisor_pid" "$t3y_pf2_pgid" >&2
    exit 126
  fi
  printf "%s\n" "$t3y_pf2_supervisor_pid" \
    > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_supervisor_pid.txt \
    || exit 126
  printf "%s\n" "$t3y_pf2_pgid" \
    > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_pgid.txt \
    || exit 126
  timeout --foreground --signal=TERM --kill-after=20s 7200s bash -c "
    set -o noclobber
    printf \"%s\\n\" \"\$\$\" \\
      > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_python_pid.txt \\
      || exit 126
    exec python sim_scripts/p14_g0b_t3y_cyld29h50_workspace_parallel_physics_sweep.py \\
      --run_label workspace_preflight2 --num_envs 128 --grid_side 2 --plan_workers 8 \\
      --settle_steps 120 --approach_steps 300 --descend_steps 500 --close_steps 30 \\
      --hold_steps 20 --lift_steps 30 --settle_stat_tail 60 --contact_capacity 256 \\
      --handoff_sha256 d1460c9d80e5f65f4ab9d85a7851b29876ef7ae0ca9e28d34bd93ddb91f0170a \\
      --protocol_path claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_prereg.md \\
      --protocol_sha256 <PREFLIGHT2_PROTOCOL_SHA256>
  "
  t3y_pf2_python_status=$?
  printf "%s\n" "$t3y_pf2_python_status" \
    > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_exit_status.txt
  exit "$t3y_pf2_python_status"
' > claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace_preflight2_stdout.log 2>&1 &
)
```

Expected p14 artifacts use prefix `t3y_workspace_preflight2_`: results, plan, trace,
RRD, RBL, Rerun validation, inspection PNG, decision snapshot, frozen script, argv,
phase journal, preclose sentinel, and (only on an exception) `failure.json`.  The shell
supervisor adds stdout, supervisor/Python PID, PGID, its exact timeout contract, exit
status, and the before-GPU inventory.  The offline attestor creates the fresh after-GPU
inventory and `terminal_attestation.json`.  p14's G0 guard reserves exit status,
sentinel, phase, failure, and terminal-attestation paths.  The supervisor writes exit
status only after GNU `timeout` returns from the Python process.  Exact status 0 means
the 7200 s watchdog, TERM, 20 s KILL escalation, and all nonzero paths were unused;
status 124/137/143 or an absent status is FAIL.  There is no retry.
The Bash supervisor records its own `$$` and PGID only after `setsid` has completed;
the outer shell's race-prone `$!` is deliberately not called the supervisor PID.

Wait until the supervisor has actually exited; do not use the attestor as a polling
command.  Then run the offline external attestor **before** opening the PNG or
considering canonical `workspace1`:

```bash
(
set -o errexit
set -o pipefail
python sim_scripts/p14_g0b_t3y_cyld29h50_workspace_parallel_physics_sweep.py \
  --external_terminal_attest workspace_preflight2
)
```

The attestor is offline: it never launches Isaac or PhysX.  It full-hash checks the
live p14 against the executed/frozen copy; verifies result↔sentinel↔artifact-manifest
and phase-prefix binding; requires exactly one close-start and no post-return PASS;
requires exit 0, no marker/warning, supervisor/Python/PGID residue 0; runs a fresh
successful `nvidia-smi`; requires the exact Python PID absent and after-minus-before
GPU PID set empty; and writes a forward-only terminal attestation.  Its only PASS is
`TERMINAL_ATTESTED_PENDING_MANUAL_VISUAL`, not instrumentation or science.  It reports
`terminal_lifecycle_pass` separately from the internal/Rerun previsual gate so a clean
exit cannot rewrite a failed measurement as PASS; command exit 0 requires both.  Then inspect
the semantic resolver/contact reports, internal checks, RRD validation, and the actual
`inspection.png`.  Record that human observation in the session document.  The user
has already authorized Isaac execution; this protocol authorizes no robot hardware
action.
