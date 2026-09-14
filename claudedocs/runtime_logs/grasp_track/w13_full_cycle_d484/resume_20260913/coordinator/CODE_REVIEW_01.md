# Root integration review 01 — preflight NOT accepted

2026-09-13. Main reviewed the actual files below while the two Orca worktrees continued CPU work. These findings are source inspection unless a separate execution receipt is named. They are not new physics outcomes.

Producer prefix: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/`.

## Runner

Read all249 lines of `rev11/src/run_production.py`, SHA `0c8c9d45c391cfa87780230908ad228fb9112aeab92187195a992012e244b3f5`, and all COMMANDS.json.

- Refusal of an existing attempt raises SystemExit, then its handler writes RUN_STATUS into the refused path.
- 32400-second cap followed by1200-second grace permits33600 seconds; total approved physics bound is32400 including cleanup.
- Timed-out child returning0 can continue to postprocessing and all_steps_rc0.
- Exceptions after child launch lack finally-owned-group cleanup.
- time.time is used for duration; use monotonic for elapsed budgets.
- Repeated/arbitrary --step requests are not rejected before output writes/launch.

Independent isolated CPU adverse tests requested in msg_0fb7c90d65f7. Never invoke this draft on an actual attempt to test refusal.

## Isaac renderer

Read all772 lines of `rev11/src/isaac_replay_w13.py`, SHA `f17db482b0fb30c7e4d0ad0d8460eaa971486ac7c6417eb1284143664958e35c`.

- :649 and :666 call sim.step while metadata claims no_physics_during_replay. This is scene physics, distinct from the absent new DEME particle solve.
- Synthetic robot pose moves while :662 chooses the previous raw sync for the S1 shell/door, so tool and synthetic robot can separate. Raw production geometry must remain raw; synthetic pose geometry must be explicitly labeled synthetic.
- Output refusal checks manifest/video only, not existing frames/render_plan; :696 saves PNG in that possibly preexisting folder.
- Missing S1/partial frames/IK failure/time-budget stop can still reach an OK token.
- Camera definition still matches cropped readiness03; full-object bounds/framing and actual source floor representation need review before another render.
- Startup/ffmpeg/close are not bounded by the inner per-frame budget check. Outer launcher must record and bound the full process including cleanup.

### Version-matched NVIDIA evidence

- Installed pip `isaaclab==2.3.0`; extension `isaaclab/source/isaaclab/config/extension.toml:4` is0.47.2 (different version category). `find_spec` resolved the pip package without importing/starting Isaac. Its `__init__.py` exposes the packaged AppLauncher.
- Producer readiness reports Isaac Sim `5.1.0-rc.19+release.26219.9c81211b.gl`; it is not silently replaced by a generic release label.
- Official title: **isaaclab.sim — Isaac Lab2.3.0 API**, https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.sim.html .
- Official source: **isaaclab.sim.simulation_context — Isaac Lab2.3.0**, https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/sim/simulation_context.html .
- Local `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py:466` updates kinematics/Fabric; :507 step advances physics; :551 render refreshes display and calls forward while playSimulations is disabled around app update.
- This supports a render/kinematics-only correction, not a claim that the corrected scene already passed. Record any initialization steps separately and verify displayed-frame simulation time does not advance.

## Rerun exporter

Read all291 lines of `rev11/src/w13_rerun_export.py`, SHA `3b82bd5008bcca783fec2737230245889ca89be9065ec978b58408317276acf6`.

- Contact entities/loop/metadata contain fixed0/door1/tray2 only, omitting receiving-bin3. coverage.contacts_logged nevertheless copies len(ci), not the number actually logged.
- Exporter reads bridge_planned_target_pos_m; frozen bridge_only_01 sim :1077 emits bridge_planned_fixed_pos_m and separate door arrays. This would fail when the bridge data is present.
- Missing certificate/worst cell on abort must not invent a PASS contract or nonfinite success scalar. Actual available decision evidence must still be finalized honestly.

Producer received msg_b2c0bdf6cf77. Required correction is schema integration with a clearly labeled CPU fixture, not another particle run.

## Partial accepted evidence, not full preflight

- Root actually reproduced two oracle01 inconsistencies; root subsequently ran frozen oracle03 30/30 rc0. See oracle_03_root_receipt.json.
- Root rehashed bridge_only_01 SHA256SUMS:11 entries including CHECKPOINT.md allOK. Producer's46/46 fixture uses test-onlyl; not a production clearance verdict.
- Independent audit executed the frozen nested callsite with CPU fakes: certificate reject0calls; first-sync reject0; second-sync reject1 prior call; positive2calls. Root read `audit/BRIDGE_ONLY_01_REVIEW.md` and accepted this bounded control-flow evidence. Enclosing run exception/finalization remains unexecuted by that particular test.
- Root recomputed the source-derived lattice and world bound; see lattice_root_receipt.json. New exact domain matching and final numerical envelope remain integration requirements.

No production GO. All further GPU/render/physics launches remain on HOLD pending a reviewed exact command and immutable candidate.

## Erratum 2026-09-13 14:38 KST — bin contact premise

The Rerun item above incorrectly assumed existing receiving-bin role3 contact records. Root inspected current sim :297, :534 and :1145: only fixed/door/tray trackers exist; bin mesh `mb` exists but has no contact tracker. The producer's `bin:null` metadata honestly means **not observed**, not zero contacts. The root's original ID3-exists premise is withdrawn; original review is preserved.

Root msg_cd2ae8394287 authorizes observation-only integration for the already-existing bin mesh: tracker/getter, raw contact points/forces, count and norm-of-vector-sum scalar, explicit semantic role3 distinct from actual engine ownerID, with corresponding Rerun and independent audit handling. No new geometry, physical parameter, path, timing, controller or execution authority. Old raw files remain bin-not-observed. GPU HOLD remains active.
