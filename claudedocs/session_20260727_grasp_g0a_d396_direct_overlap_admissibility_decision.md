# 2026-07-27 — Grasp G0a D396 direct-overlap admissibility decision

## What and why

`D396 [d388_reanchor_direct_pre_float32_nonoverlap_admissibility_decision]`
asked one narrow, falsifiable question:

> Do the two D389 pre-Float32 overlap calculations that actually completed
> make the frozen D388 re-anchor candidate inadmissible under D388's
> zero-positive-overlap contract?

This is the forward-only decision layer requested after D395. It does not edit
the immutable D388 or D389 records. It also does not use D395's mixed-authority
hybrid table to decide the candidate.

이번 case의 신규 변수:

1. `d388_direct_pre_float32_positive_pair_nonoverlap_admissibility_decision_v1`

This session satisfies the progress rule because the registered decision could
have failed on witness identity, numeric disagreement, a directional
calculation below the gate, a Qhull fallback, or artifact/visualization
integrity. It was not a validation that could not change the decision.

## Frozen inputs and authority

- HEAD and `origin/master` at execution:
  `d354d46134fe002073642441a7d24c99fe579edd`
- script SHA-256:
  `f59afb17090f90f356fc2a2c10e5ed73df42f4a3248b25f6781fa90f1ad74f2b`
- external execution authority SHA-256:
  `eb8517cd8b29fb944709f0dad1e4de2cae2fd1d54836d76e0ef109dd88cb9f87`
- preregistration SHA-256:
  `6e4b78ddaad88cc8062604b30a338a7b7b2443a8608af8f32cbfde127ff8c8e9`
- decision authority:
  D389 pair 5 (`UPPER 1-2`) and pair 26 (`LOWER 2-3`) only
- D395 role:
  frozen input integrity and background only; hybrid-table authority `false`

The authority bound nine immutable D388/D389/D395 inputs by SHA-256 and bound
the worktree outside the forward-only D396 output directory by canonical
status hash
`60c572c77b4789a5214810b22aca54c46ab6eee7d0bac890dce1bf5b0a45f26f`
over 165 status rows.

## Observable procedure

1. A static read-only review checked input hashes, direct-pair identities,
   authority self-consistency, exact file inventories, one-worker/no-retry
   enforcement, and Rerun/manual-inspection contracts. Verdict: no blocker.
2. `prepare` verified all authority, frozen-input, schema, dependency, font,
   and Git checks before writing the preregistration.
3. The offline worker was launched exactly once. It did not replay clipping or
   a solver; it extracted the already-completed strict and two directional
   D389 records for the two registered pairs.
4. Each witness had to satisfy:
   strict calculation PASS, both directional calculations PASS, every volume
   above `1e-18m^3`, positive volume, and no Qhull fallback.
5. The registered controls tested upper-only, lower-only, both enabled, both
   masked, D389-integrity failure, and opposite generic D395 outcomes.
6. Canonical evidence and display geometry were committed before visualization.
7. An exact 1920x1080 board and save-only RRD/RBL were generated. The RRD/RBL
   passed footer, entity, timeline, and component validation. The Viewer was
   invoked once.
8. Both images were actually opened and inspected. Eleven manual checks passed,
   and their exact hashes were bound before finalization.
9. Finalization rechecked the full artifact chain and exact 11-phase order.

## Numeric result

Canonical evidence SHA-256:
`9cd315f69d4ce6a2b6b25addbfb589f2be0bdb3ea9e35b47069bea9be0580c1f`.

Frozen gate:

- positive-volume threshold: `1e-18m^3`
- allowed positive-overlap count: `0`

Upper pair 5, children 1-2:

- strict volume: `6.4038856253626914e-15m^3`
- gate ratio: `6403.885625362691`
- signed inradius: `0.06985495742125113nm`
- left-by-right / right-by-left:
  `6.403885673976526e-15 / 6.403885506278743e-15m^3`
- strict and both directions: PASS, positive, no Qhull fallback

Lower pair 26, children 2-3:

- strict volume: `2.4130456372851684e-15m^3`
- gate ratio: `2413.045637285168`
- signed inradius: `0.025803862627049094nm`
- left-by-right / right-by-left:
  `2.413045456355167e-15 / 2.413045613703222e-15m^3`
- strict and both directions: PASS, positive, no Qhull fallback

Controls:

- upper-only: candidate admissible `false`
- lower-only: candidate admissible `false`
- both enabled: candidate admissible `false`
- both masked: `null`, never promoted to PASS
- D389 integrity failure: `null`
- generic D395 PASS or FAIL background: decision remains `false`

Numeric verdict:

`D396_D388_REANCHOR_DIRECT_PRE_FLOAT32_NONOVERLAP_INADMISSIBILITY_CERTIFICATE_PASS`

## Execution and visualization

- worker/retry/signal: `1/0/0`
- worker return/runtime: `0 / 0.4722126529086381s`
- Viewer/retry: `1/0`
- phase contract: `11/11`, exact and forward-monotonic
- exact board: `1920x1080`, SHA-256
  `d634b1ff328dd8016ee522cc9a1f464e9f3b231e73020dcd98e48ce28cd29618`
- RRD/RBL: strict validation PASS
- Viewer screenshot: `3840x2160`, SHA-256
  `799303904458038131e56f14ae9d28338815336415cf3cc9a96b2640ed5d7d26`
- manual inspection: `11/11` PASS

The Rerun screenshot contains a sandbox message-proxy permission toast. It
does not cover the two child point sets or authority metadata, and the Viewer
returned successfully. The red center points are explicitly display markers;
their rendered radius is not the physical overlap extent.

## Interpretation and nonclaims

The D388 re-anchor candidate is not admissible under its own registered
zero-positive-overlap contract. Either one of the two actual completed D389
witnesses is sufficient to reject it.

This is the correct forward-only reflection of D389 into the candidate
decision. The immutable D389 seam records are not rewritten, and the other
nine indeterminate seam records remain indeterminate. Resolving them is not
required to rescue this specific candidate because it already has two
independent contract-breaking witnesses.

The overlaps are microscopic numerical design geometry. D396 does **not**
claim:

- visible or manufacturing-scale penetration;
- authored-to-cooked or live PhysX overlap;
- contact, grasp, stability, or tipping behavior;
- that the D395 hybrid table is adopted;
- that D388 or D389 retroactively passes.

Scope counters were zero for clipping/solver replay, geometry or partition
change, collider/USD, Isaac/PhysX/Warp/CUDA, cylinder, physics, q5, contact,
grasp, target/IK/path, hardware, and process signals.

Operational verdict:

`D396_D388_REANCHOR_NONOVERLAP_INADMISSIBILITY_DECISION_COMPLETE_NO_MATERIALIZATION`

- `materializable_candidate=false`
- `g0a_pass=false`

## NVIDIA-stack entry review

Installed local versions were cross-checked before reviewing the later
materialization/physics boundary:

- Isaac Sim `5.1.0.0`
- Isaac Lab `2.3.0`
- Kit `107.3.3`, USD `24.05`
- Omni PhysX integration extension `107.3.26`

The version-matched NVIDIA guidance says that USD is parsed into PhysX,
required cooking/decomposition occurs, and only then does simulation stepping
begin. It recommends primitives first when adequate, then convex meshes; a
rigid body can own multiple collider prims in its subtree. It also requires
mass, center of mass, inertia, and principal axes to be handled explicitly or
understood as inferred properties.

Therefore D396 does not authorize USD or physics. The minimum next candidate
is a separate offline design whose shared boundaries guarantee zero-volume
overlap while retaining the frozen surface, void, clearance, count, and bounds
contracts. Only after that passes may a separate live identity/cook readback
case be considered.

Even after live identity, a product-representative 29x50mm physics case needs
measured mass and preferably measured dimensions/tolerance, center of mass or
a justified model, inertia, and jaw-cylinder plus table-cylinder friction.
Isaac can run with inferred defaults, but such a run would not be evidence for
the real diffuser before those properties are justified.

## Sources

- `sim_scripts/cyl34_top_view_d396_d388_reanchor_direct_pre_float32_nonoverlap_admissibility_decision.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d396/attempt1_d388_reanchor_direct_pre_float32_nonoverlap_admissibility_decision/d396_execution_authority.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d396/attempt1_d388_reanchor_direct_pre_float32_nonoverlap_admissibility_decision/d396_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d396/attempt1_d388_reanchor_direct_pre_float32_nonoverlap_admissibility_decision/d396_direct_overlap_admissibility_evidence.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d396/attempt1_d388_reanchor_direct_pre_float32_nonoverlap_admissibility_decision/d396_offline_worker_supervisor.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d396/attempt1_d388_reanchor_direct_pre_float32_nonoverlap_admissibility_decision/d396_rerun_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d396/attempt1_d388_reanchor_direct_pre_float32_nonoverlap_admissibility_decision/d396_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d396/attempt1_d388_reanchor_direct_pre_float32_nonoverlap_admissibility_decision/d396_completion_summary.json`
- NVIDIA Omni Physics 107.3, *Simulation Control*:
  `https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/simulation_control/simulation_control.html`
- NVIDIA Omni Physics 107.3, *Colliders*:
  `https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html`
- NVIDIA Omni Physics 107.3, *Rigid Bodies*:
  `https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html`
- NVIDIA Isaac Sim 5.1.0, *Reference Architecture and Task Groupings*:
  `https://docs.isaacsim.omniverse.nvidia.com/5.1.0/introduction/reference_architecture.html`
- NVIDIA Isaac Sim 5.1.0, *ImportConfig*:
  `https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/structisaacsim_1_1asset_1_1importer_1_1urdf_1_1_import_config.html`
- NVIDIA Isaac Sim 5.1.0, *Tutorial 10: Rig Closed-Loop Structures*:
  `https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/rig_closed_loop_structures.html`
