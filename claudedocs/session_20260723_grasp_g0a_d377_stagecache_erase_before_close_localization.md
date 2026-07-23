# D377 — D375 StageCache Erase-before-close localization

Date: 2026-07-23 KST  
Case: `g0a_d377`  
Attempt: `attempt1_stagecache_erase_before_close_localization`  
User authorization: one forward-only worker, no retry, one explicit StageCache erase after
successful PhysX detach, frozen D375 acquisition workload, no q5/physics/contact/cylinder science  
Frozen formal verdict: `D377_STAGECACHE_ERASE_BEFORE_CLOSE_LOCALIZATION_FAIL_STOP`  
Result branch: `UPSTREAM_WORKLOAD_MISMATCH_ERASE_EFFECT_NULL`  
`g0a_pass=false`

## What and why

D375 completed the P34 callback and property-query acquisition but did not return from the final
Isaac/Kit shutdown boundary. Its supervisor waited `900s`, sent SIGTERM, waited another `20s`,
then required SIGKILL. D376 proved that the last observed boundary was framework release/process
exit, but could not identify the exact native blocker.

Installed Omni PhysX helper source performs two lifecycle actions together: detach the USD stage
from PhysX and erase that stage from `UsdUtils.StageCache`. D375 detached but did not erase its
custom in-memory stage. D377 therefore tested one new variable only:

`이번 case의 신규 변수: [explicit_stagecache_erase_after_physx_detach_v1]`

The question was operational, not geometric or physical: if the exact D375 termination workload
is repeated and the stage is removed from StageCache exactly once after detach, does the process
reach a clean external exit under a bounded watchdog?

## Frozen scope

- Reused D375's P34 derivative asset and acquisition workload.
- Callback requests `34`; property queries `2`; link5/gripper_link collider rows `17/19`.
- Preserved the Python `stage` reference, `SimulationApp.close()` API, `fastShutdown`, asset,
  material, mass, actuator, physics settings, and timeline STOP/time zero.
- Worker count `1`, automatic retry `0`, watchdog `120s`, SIGTERM grace `20s`, SIGKILL only if
  still alive.
- Full geometry classifier was intentionally not run.
- Physics steps, q5 commands/samples, contacts, cylinder creates/writes, SimulationContext,
  reset, public forward, timeline play/commit, automatic decomposition sweep, USD writes,
  target/IK/path changes, and physical-setting changes were all required to remain `0`.
- D334 sidecar and D375/D376 evidence paths were immutable.

## Step-by-step execution

### 1. Boot and repository boundary

- Read `AGENTS.md`, `START_HERE.md`, DECISIONS and ledger, then the referenced D375/D376 evidence.
- Verified `HEAD == origin/master == e30f7f99d44252f509e383627738f3ad7967ea93`, subject `D375`.
- The worktree was clean after the user's D375 push and before approved D376 work. Existing D376
  changes were preserved; no commit or push was performed.

### 2. Installed-stack and official-source cross-check

- GPU: NVIDIA GeForce RTX 4090 Laptop GPU, compute capability `8.9`, driver `580.159.03`,
  memory total/free `16376/15465 MiB` at preregistration.
- Installed Isaac Sim `5.1.0.0`, Isaac Lab `2.3.0`, Kit `107.3.3`, Omni PhysX `107.3.26`.
- Installed PhysX helper source pairs stage detach with `StageCache.Erase(stage)`; its registered
  SHA-256 is `d7e62f...0dcc1` and the relevant installed-source lines are `595-599`.
- `StageCache.Erase(stage)` removes cache membership. Because D377 deliberately retained a Python
  stage reference, it does not prove destruction of the stage object.
- NVIDIA Isaac Sim 6.0 bug 5948099 is later-version mechanism evidence only. D377 does not claim
  that D375 was exactly that bug.

Official references registered before execution:

- NVIDIA, *Isaac Sim 5.1 SimulationApp API*:
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html
- NVIDIA, *Isaac Sim 5.1 Release Notes*:
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html
- NVIDIA, *Kit 107.3.1 UsdUtils API*:
  https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/107.3.1/pxr/UsdUtils.html
- NVIDIA, *Kit 107.3.1 Usd StageCache API*:
  https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/107.3.1/pxr/Usd.html
- NVIDIA, *Isaac Sim 6.0 Release Notes*:
  https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html

### 3. Preregistration and failure-capable controls

- Preregistration checks `17/17` PASS and negative controls `7/7` PASS.
- No pre-existing Isaac worker was found. The pre-existing GPU process PID `4159377`, `390 MiB`,
  was `mcp_memory_service.server`, not an Isaac worker; it was not touched.
- Registered source hashes:
  - controller `e6e18a2cbda79d8545da36661501231f5604bd7017981236d49897a933148df4`
  - worker `1d7f56de664885278d4a2bf35866203e682ce8a9f54ca3d6b70fd4592bfc259e`
  - `roarm_rl/viz_debug.py`
    `a21a1d4d64db51963bef753704b2848f1ed4cc0bbfce4caf754330ad5a769a84`
  - `roarm_rl/rerun_contract.py`
    `aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e`
- Static scope attestation found Erase/detach/close counts `1/1/1`; physics step, forward,
  timeline play/commit, reset, and contact collection were all `0`. A generic AST `update` hit was
  `hashlib.digest.update`, not an Isaac application update pump; the explicit app-update gate passed.

Preregistration:
`claudedocs/runtime_logs/grasp_track/g0a_d377/attempt1_stagecache_erase_before_close_localization/d377_preregistration.json`
SHA-256 `01cc3f55e2bb5e7718b5eed210501a37622bc731dbe9e4bc623f97e91f885d87`.

### 4. Single actual worker

- Worker/retry: `1/0`.
- SimulationApp launch: `1`.
- PhysX stage attach/detach: `1/1`.
- Callback requests: `34`; property queries: `2`.
- Link5 property rows: rigid body VALID plus `17/17` collider rows VALID.
- Gripper-link property rows: rigid body VALID plus `19/19` collider rows VALID.
- Worker protocol: PASS.
- Timeline was STOP at time zero before and after the inherited workload.

The exact Erase observation was:

1. Before Erase, cache `Contains(stage)=true`.
2. The registered cache ID was valid and found the same stage.
3. `UsdUtils.StageCache.Get().Erase(stage)` was called once, immediately after successful PhysX
   detach.
4. It returned `true`.
5. After Erase, `Contains(stage)=false`, the old ID was invalid, and lookup by the old ID was absent.
6. The Python `stage` reference was still retained.

The worker then reached the close-start marker and exited externally with return `0` in
`6.733121555997059s`. It did not time out; SIGTERM and SIGKILL were not sent; process-group and
worker-GPU residue were empty. The close-return marker is absent, which is expected under the
inherited D367 rule: external supervisor process exit is the terminal authority.

Supervisor evidence:
`claudedocs/runtime_logs/grasp_track/g0a_d377/attempt1_stagecache_erase_before_close_localization/d377_worker_supervisor.json`
SHA-256 `1c0a4754da7fa0bae748e6c1095a1c39982ab50cef6202c6918f742fb635ce49`.

Raw/preclose evidence:

- `d377_worker_raw_summary.json` SHA-256
  `f14d2cf38cffc03a3121719a4dac0a62d612b46926a5ff6afcc10cd143717fb1`
- `d377_worker_preclose_sentinel.json` SHA-256
  `4a312e20b4444f84897864e013b1f4eb74ce57aeb52377bfaa0ee9e3d068cc89`

### 5. Frozen formal analysis

The preregistered workload comparator produced these hashes:

- D375 selected canonical SHA-256:
  `ec930163ac2a9cdbf7342630dccd34d5467fa3618dfd0d6213066fbaa12b0b7b`
- D377 selected canonical SHA-256:
  `758504733115b8740a972fe99ea63f9303d5759505d03a29e1e9c9570fa13c81`

Because they differed, the registered fail-closed logic did not attribute the clean exit to the
Erase variable. It froze:

- verdict `D377_STAGECACHE_ERASE_BEFORE_CLOSE_LOCALIZATION_FAIL_STOP`
- branch `UPSTREAM_WORKLOAD_MISMATCH_ERASE_EFFECT_NULL`
- `lifecycle_localization_pass=false`
- formal causal interpretation `null`

This artifact is immutable and was not rewritten after inspection.

Localization evidence:
`claudedocs/runtime_logs/grasp_track/g0a_d377/attempt1_stagecache_erase_before_close_localization/d377_stagecache_erase_localization_evidence.json`
SHA-256 `556db509206fd99507b68f0ce6d686ba3dbb15708309e475e4344107da0777b2`.

### 6. Independent post-result diff audit

The formal mismatch was then examined read-only by independent audits without changing any D377
artifact or rerunning Isaac. The selected-signature differences were exactly `68`:

- Callback witness SHA differences `34`. Each underlying witness JSON differed at exactly one
  leaf: `request_return_repr`, containing a run-specific Python object memory address.
- Live-inventory differences `34`. Each differed only at `prototype_path_diagnostic`, whose
  generated `__Prototype_N` ordinal changed between runs. The semantic suffix, live path subject,
  collision enable, approximation, hull limit, Float32 point hash, face-index/count hashes, and
  typed min-thickness values were otherwise exact.

Additional checks:

- Callback witness filenames exact `34/34`.
- Callback payload exact `34/34`: vertices `314`, indices `1016`, original polygons `262`.
- Property-query raw differences `40`: opaque runtime `path_id` values `38` and elapsed seconds
  `2` only. Mass, COM, inertia, principal axes, volume, AABB, local pose, semantic path, result,
  and collider counts were exact.
- Authored readback, owner structure, mass base-versus-derivative, inspection-stage mass, and
  canonical data outside the collision subtree were exact.

Excluding only the two run-dependent selected-signature fields gives the same independent digest
for D375 and D377:

`28aadb5ff26270039df58f7cd06080bf7afcdec001402e886a6edf1483fdfe31`.

Therefore the registered mismatch was a comparator false negative, not a geometry or termination-
workload change. This is a post-result diagnosis, so it cannot retroactively turn D377 into PASS.
A new forward-only offline preregistration is required to make the corrected authority formal.

## Visualization result

- Exact board: `1920x1080`, `176856B`, SHA-256
  `6e1c2433970fabe69eddd66d7cd6b31ac23636455ec476146b3b6c9ec85a59b3`.
- Save-only RRD: `67991B`, SHA-256
  `f9616da3f6e5080dc3589fba0535cc7a7473781c32c81f1777cebb8fdb0b5d3f`.
- RBL: `43615B`, SHA-256
  `6bc0c53d485e4831c693bf6c497c997e7e69b5642de2c0ea3f7786c591d7a82a`.
- Rerun SDK/CLI `0.34.1`, footer, entity, duration-timeline, required-component and one headless
  Viewer invocation checks passed.
- Requested logical Viewer size was `1920x1080`; HiDPI physical capture was `3840x2160`, SHA-256
  `50b021a483887ca55b8eafaa81da3bbe6a5db25a81c60d3a8437ad88fc4aa807`.
- Original-resolution inspection found the board readable and unclipped, but the Rerun lower
  Korean boundary text rendered as square missing glyphs. Manual and visualization completion
  therefore failed.
- The board also displays the frozen registered `workload=False` result. Because the post-result
  audit found that result to be a comparator false negative, this board must not be used as a
  corrected professor-facing conclusion.

Completion summary:
`claudedocs/runtime_logs/grasp_track/g0a_d377/attempt1_stagecache_erase_before_close_localization/d377_completion_summary.json`
SHA-256 `762d13ad790231a9bd810d37f29ee42b4d09b604e5fabe4dbae1b3ebb6f5c5dc`.

Manual inspection:
`claudedocs/runtime_logs/grasp_track/g0a_d377/attempt1_stagecache_erase_before_close_localization/d377_manual_visual_inspection.json`
SHA-256 `8118548e45e827da4a0a8c11887ed449cb7fe5958f300b47619499283055773d`.

## Plain-language verdict

This time Isaac did not hang. The same meaningful callback/property workload was executed, the
stage was removed from StageCache once, and the worker exited normally in about `6.73s`. That is
a strong observation that retained StageCache membership may have triggered the D375 terminal
hang in this workload.

However, the predeclared equality test accidentally treated changing memory addresses and internal
prototype numbers as meaningful data. Its formal gate therefore failed before it was allowed to
credit the clean exit to Erase. The correct scientific record is both facts together:

1. Operational clean exit after one Erase: observed.
2. Formal single-variable causal support: still null because the preregistered authority was
   defective.

D373 also exited normally without Erase, so Erase is not proven universally necessary. D377 did
not run the full P34 live-identity classifier, cylinder contact, tipping, or grasp. Consequently
full P34 identity, A64/P34 physical equivalence, tipping causality, current-pose closure, and grasp
feasibility remain null; `g0a_pass=false`.

## Next authorization boundary

Recommended but not approved:

`D378 [d377_ephemeral_identifier_provenance_and_workload_authority_repair]`

Offline-only scope:

1. Read immutable D375/D377 raw summaries and callback witnesses.
2. Preregister exclusion of only run-dependent diagnostics from termination-workload identity:
   callback `request_return_repr`/derived witness SHA, generated `prototype_path_diagnostic`, and
   already diagnosed property `path_id`/elapsed fields where property equality is evaluated.
3. Independently bind corrected authoritative payload counts and digest.
4. Produce an ASCII-only corrected board/Rerun projection so missing Korean glyphs cannot obscure
   the boundary.

Isaac/PhysX launch, q5, physics, contact, cylinder, target/IK/path, collider regeneration, and
physical-setting changes remain forbidden in D378. A repaired full P34 live-identity classifier
and any cylinder physics comparison each require later separate approval.
