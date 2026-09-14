# W13 resume — 2026-09-13, contract revision 1

이번 case의 신규 변수: [] — 승인된 W13 전체 사이클 통합을 이어가며 추가 물리/전략 변수를 도입하지 않는다.

## User authority and ownership

User approved the next work after the criteria briefing: bridge integration → Isaac full-cycle replay readiness → long-run execution approval → production/replay/verification, supervised here using appropriate Orca worktrees. The coordinator interprets this as the previously offered maximum 9 hours for ONE production physics process, conditional on preflight acceptance. No automatic retry or duration extension. Root issues exact technical GO after independent review; do not ask the user the same cost question again.

Main = `/home/cgxr/Documents/Robotics/RoArm_Project`. Only root writes main state/relay/session/ledger and this coordinator directory. You are not alone: preserve others' changes; never revert, move, overwrite, merge or commit old work. Worktree state docs are stale; read main versions.

- Claude `claude-opus-5`, worktree `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle`, OWNS ONLY `claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/**`.
- Codex `gpt-5.6-sol/high`, worktree `/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit`, OWNS ONLY `claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/**`.
- Exact Orca executable `/home/cgxr/.local/bin/orca-ide`. Never bare `orca` (GNOME screen reader). Use only new live injected Task/Dispatch authority. Read and ACK coordinator mail at checkpoints and before costly commands; send means enqueued, not read. No nested workers. Heartbeats per preamble. Use Orca ask/reply for GO/questions, not a local question UI. Send worker_done only after whole assigned outcome or a genuine terminal blocker; no self-release.

Forbidden: all hardware queries including T105/serial, motion/PID/torque/cameras; training, A/B/C, dt/material/seed/strategy sweeps, package installation, old outputs or source edits, commit/push, public sharing. GPU is producer-only, with DEME and Isaac sequential. CPU audit may run independently.

## Read / preserve

Main AGENTS, START, DECISIONS_ACTIVE, LEDGER_RECENT; main `claudedocs/session_20260912_w13_full_cycle.md` and old `.../w13_full_cycle_d484/coordinator/{CONTRACT.md,REPORT_w13_partial.md}`. Producer old `implementation/HANDOFF_NOTE_01.md`, `preflight_01/rev10_frozen/`, old canonical `attempts/smoke_wall_regression_01/` are immutable reference. Auditor old `audit/{bounded_clearance_01,runtime_bridge_guard_01,wall_smoke_result_01}/` are immutable reference. Old guard/test SHA values must be checked before copying; do not run an old test that rewrites its evidence.

Reuse the established HOME/S1/FK/local-3mm adapter and frozen fixture; do not repeat already-accepted derivations absent an observed defect. W11 dt1e-6 is provisional, not convergence/default promotion. Preserve seed460, all20000 clumps, pile/template, S1 geometry, contact law/E/friction/density/servo protections and existing frozen W13 motion/timings/bin/fixture. Output paths and observability/integration plumbing may change. Any required physics/path/fixture change needs coordinator review before implementation. No second W10/W11/old-wall baseline run.

## A. Bridge integration, then renderer preparation

1. Copy required rev10 sources into the NEW owned directory, preserving imports with no hardware side effects. Integrate reviewed guard from old audit at the actual `resume_fk_here()` result, BEFORE either bridge starts physics. Validate actual W11 pose→actual q_res alignment and actual q_res→post_lift joint path; never substitute a convenient endpoint path. Hold solver/time/particles unchanged during pure CPU validation. FAIL/INCONCLUSIVE => `CLEARANCE_UNCERTIFIED`, finalize raw/RRD and stop, no extra hold/reset/retry/path optimization. Exact input/decision/time and finalization must be testable with a fake solver counter and adversarial fixtures, not only a guard unit test.
2. Require frozen separating-gap > analytic movement bound + explicit numeric epsilon and applicable joint limits. The bounded 3584-interval old proof excludes these two bridges; do not redo its already-proven parts unless a dependency actually changed. Review old guard fully; 5 synthetic PASS claims are not accepted sight unseen.
3. AFTER integrated bridge tests, prepare Isaac full-cycle renderer from W12 reference, with source-time/phase/robot/tool/bin and target/actual markers. Each saved particle frame maps once with no invented intermediate particle dynamics. Include transition and close/reclose/carry/discharge/return frame support. Use stored actual door quaternion, consistent source/display transforms and corrected 2.057714892mm/deg metadata if needed; no old7.7822 error. Prescribed robot display is not actuation verification.
4. One bounded renderer-readiness test (<=8 selected existing W13 raw frames and/or explicitly synthetic phase/pose fixtures, <=600s, new path) is permitted before production; no new particle solve. Label synthetic inputs prominently and never count them as full-cycle result. Use no BasicWriter, no unbounded dumps. Preserve failed attempts. Follow version-matched NVIDIA official/source evidence for API claims; verify installed versions/pins without installation.

## B. Criteria reconciliation before output

Publish ONE machine-readable `criteria.json` with origin, purpose, value/unit, severity and scientific limitation for every threshold, used or identically checked by producer and audit. Freeze criteria/code/params/mesh/commands/input/output paths BEFORE production.

- 5m/s is the existing sampled-speed WARNING; >20m/s is the existing Python pop-stop. Do not turn warning5 into an undocumented hard audit failure. Do not change actual engine protection parameters. Equality operators must match source and boundary controls. Old wall max0.5mm/p990.1mm is an uncalibrated descriptive comparison, not a full-cycle acceptance test; do not rerun it or claim causal equality.
- No surface-overlap-zero rule. Near/overlap/center-crossing/contact are different observations. Full-cycle particles legitimately leave the source box, so do not apply stationary source containment globally. Recompute spatial classification from particle coordinates/quaternions and actual tray/bin geometry independently; scalar metadata alone is not independent geometric verification.
- Every original ID must appear exactly once in a disjoint inventory (source, receiving, tool, spill, in-flight, ambiguous). Unknown geometry remains ambiguous; ambiguity is not corrupted accounting. Do not demand final global ambiguous/in-flight0 to certify raw integrity. Check count/mass conservation with canonical template mass, never supplied expected mass alone.
- Separate geometry-inside, settled delivery and task success. Previously proposed operational window>=0.25s, spacing<=.05s, speed<=.005m/s, displacement<=1mm and stable definite-bin IDs is UNCALIBRATED, not universal rest. Use existing frozen hold/timings; if insufficient, report lower/upper bounds and missing settlement proof. No extra wait to manufacture a pass. Do not require an invented minimum delivered mass. Zero delivery is an honest unsuccessful experiment, not a reason to change conditions.
- Observed engine sync times are authority; requested duration sum is separate. Float32 particle-frame times use justified representation tolerance, never `exact_same_time` language. Dense sync and sparse particle timelines are distinct, with transition coverage and ID correspondence checked.
- Scientific verdict, data/visual completeness and workflow completion are separate. Process rc0 or script selfPASS cannot prove delivery. Full-cycle absence must keep full-cycle coverage unmet. Synthetic-only success cannot pass production checks.
- Add actual negative controls: mutate pose/phase/time/duplicate/lost IDs/bin transform/force mapping and inject a changed-byte input against an unchanged expected SHA. `hash != zeros` is NOT a corruption test. No threshold adjustment after outcomes.

## C. Preflight evidence and exact GO

Producer publishes `PREFLIGHT.md`, `criteria.json`, preserved revision/param diff, boundary/integration test receipt, small renderer test/inspection, actual full-cycle schema, fixed commands, prospective manifest, timeout/finalization strategy, output/storage estimates. Real failed commands remain listed. Audit independently reviews frozen snapshots and returns GO/NO-GO with concrete source/line defects. Resolve focused source-level defects in NEW revision paths, do not cycle through stylistic rewrites. Physics/output already executed => preserve; no automatic rerun.

Implement a runner that checks full SHA256 immediately before execution, rejects existing output/log files, writes actual subprocess rc/start/end/timeout/stdout/stderr receipts, and enforces max32400s physics wall time. Graceful signal handling must flush partial raw/RRD before a bounded hard stop; no detached orphan processes. Heartbeat/log progress cannot replace solver progress. Require final JSON/run status even on planned guard abort. Freeze what is included in the budget; Isaac/Rerun postprocessing each bounded separately, GPU sequential.

Root sends GO naming ONE immutable production revision + argv + fresh attempt after producer readiness AND independent preflight acceptance. Cost permission exists; no production until technical GO. One continuous HOME→approach→descend/close/lift/reclose→carry→one fixed discharge→close→HOME. Start original initial pile once with continuous solver state; never restart from incomplete final NPZ, attach/remove/respawn particles, reset velocities, or splice a decorative transfer movie.

## D. Production, replay, audit and return

Producer runs the one approved production process, sends actual start/progress/phase/process exit. Record all executed phases, actual poses/door/contact points+forces, source/receiver/tool inventories, pre/post maps with observation times, sim vs wall time. On failure preserve full available evidence; stop for coordinator decision instead of searching for success. Export Rerun0.34.1 file-sink-before-log/finalization/footer/exact entity/timeline/component/RBL + actual decision inspection. Use `roarm_rl.viz_debug` where practical. Replay actual saved production data in Isaac into new outputs, source-time annotated video and map arrays; incomplete physics must be visibly labeled incomplete. No physics during replay.

Audit checks final immutable manifest, independent raw geometry/time/ID/accounting/contact/criteria, source-to-Rerun/Isaac mapping, coverage and adverse controls. Producer check modes and audit check modes MUST be read-only (`-B`, no rewritten RESULT/manifest/PNG), fail on missing data and distinguish actual production from fixtures.

Required command interface (absolute NEW owned root as CWD; roarm Python unless visual runtime declared):
- producer `verify_resume.py --check-preflight` => `W13R_PREFLIGHT_VERIFIED`; `--check-run` => `W13R_FULL_CYCLE_RECORDED` only when real full phase coverage exists; `--check-visual` => `W13R_PRODUCTION_VISUAL_VERIFIED`.
- auditor `verify_resume_audit.py --check-preflight` => `W13R_INDEPENDENT_PREFLIGHT_VERIFIED`; `--check-results` => `W13R_INDEPENDENT_RESULT_VERIFIED`; `--check-negative-controls` => `W13R_NEGATIVE_CONTROLS_VERIFIED`.
- Each oracle may fail honestly. A completed full cycle with zero delivery must report that result; a partial/guard-aborted cycle does not emit full-cycle token. Own REPORT.md lists actual commands, numerical findings with raw paths, exact limitations and changed files. Root re-runs reviewed checks, actually inspects decision images, updates main state/relay, and resolves worker lifecycle.

## Roles now

Claude starts A1 bridge integration, then A3 renderer, continuing through D under GO. Codex starts independent old-guard review + criteria/negative checker in parallel; then reviews producer immutable preflight, later actual output. No duplicate implementation owners. Request only choices needed to advance; do not ask permission for CPU checks already in this contract.
