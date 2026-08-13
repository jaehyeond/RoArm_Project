# p16 v12 / t3u_side_preflight13 preregistration — validator isolation repair

Status: PREREGISTERED / NOT RUN / G0 BLOCKED UNTIL INDEPENDENT PURE AUDIT

Run profile: `side_preflight13`

Canonical prefix: `t3u_side_preflight13`

Executable: `sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v12.py`

Detached supervisor: `sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v15.py`

Supervisor V15 SHA-256:
`64cece0c57ce0b5fb713f67c69efac6724e5b31ba16f0c0d0454294442aebeb3`.
This preregistration's final SHA-256 and the canonical preregistration's final SHA-256
are pinned by p16 v12 before source freeze. The final p16 v12 SHA-256 is the last member
of that one-way closure and is published in the freeze receipt; neither prereg embeds a
self-referential p16 hash.

이번 case의 신규 변수: [side-midpoint grasp point, SDG candidate pose].

Those two scientific variables remain exactly the D419-approved variables. Preflight13
is a reactive validator-only repair after preflight12 exposed historical Python module
cache contamination and over-coupled witness/numeric semantic leaves. It does not change
physics, control, CUDA serialization, lifecycle, rendering, or scientific classification.
This document does not authorize Isaac, PhysX, Kit, render, or hardware execution.

## 1. Frozen scientific and lifecycle inheritance

Unless this document explicitly replaces a validator clause below, every byte-level
contract in `t3u_side_preflight12_prereg.md` (SHA-256
`9817283fc77fb14e503bca6b6e560d1cf14ad80b5e7c64daa0b05c16739391a9`)
is inherited unchanged. In particular:

- the analytic upright D29 x H50 mm, 24.83 g cylinder, fixed-base RoArm, support,
  materials, gravity, PhysX settings, parsed URDF limits, contact reporters, collision
  filters, target poses, candidate order, and thresholds are unchanged;
- preflight has exactly 8 environments, 5 active scientific rows, witness slot 5, and
  padding slots 6 and 7; the active denominator is exactly 5;
- phases remain settle 120, approach 400, stage 400, descend 400, close 400, hold 120,
  lift 500 at 200 Hz: exactly 2 diagnostic callbacks, 2,340 fresh task callbacks, and
  2,342 combined callbacks;
- contact gate remains strictly greater than 0.02 N, jaw-load gate 0.01 N, lift gate
  6.0 mm, and the frozen tip/settle/contact/count/quaternion/fixed-base gates remain exact;
- the CUDA-derived safe witness command remains float32 degrees
  `[22.147554397583008, 54.00971603393555, 84.68325805664062,
  -26.58652114868164, 90.0, 66.40000915527344]`, with the same six uint32 words;
- exactly one physics attempt and, only after raw-zero plus a complete all-true semantic
  preclose gate, at most one render attempt are allowed; retry count remains zero;
- timeouts, raw wait status, PID/SID/PGID, signal, reap, GPU-before/end/after, phase ledger,
  preclose, RRD/RBL/footer, PNG, MP4, render cadence, and terminal attestation contracts
  are inherited without relaxation.

The p15 candidate source remains SHA-256
`67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384`.

## 2. Exact immutable preflight12 retirement

Preflight12 is observed evidence, not a successful or promotable experiment. Its prefix
contains exactly the following 25 files; every full hash is normative:

| File suffix | SHA-256 |
|---|---|
| argv.txt | e1557c4c6a68b2ad26a475ae71cf51cb4c9273888398368f1e0d54bc71677eda |
| decision_snapshot.png | 938b3c8126b17402a0de1974171b5e62d9912cb7f2437ffce90b0e86e25aedca |
| exit_status.txt | a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca |
| inspection.png | ead72aa94005429d0e97ac1448d3ce61abbabc1587170f8bb5b0acc4804431ca |
| nvidia_smi_after.csv | 4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0 |
| nvidia_smi_before.csv | 4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0 |
| nvidia_smi_supervisor_end.csv | 4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0 |
| pgid.txt | 882064bde568a4379c2b7effcce895105eb3968d4e5da93e07290f5b6911fb23 |
| phase.jsonl | dd7f36a8ea94e4abce82cf4730fc7351f045626d321613e65637f62259edffa4 |
| physics_python_pid.txt | 8c2cdadb893b285e77fbaac3f5494bdc5619a18f1d94bd90165df4797c085608 |
| plan.json | f9776a461177ccc8b99fb98da9694fc916637f48ba751db2373ae330606b7fcd |
| preclose_sentinel.json | 52399bebf3787eb31d300b31b46b48891db6d5f44770d6da4fa1a03297d3a5ed |
| prereg.md | 9817283fc77fb14e503bca6b6e560d1cf14ad80b5e7c64daa0b05c16739391a9 |
| rerun_validation.json | f22171f40539eff5c7d8691ff6f9ee6f04078506bfc079181075839ebda76fa3 |
| results.json | 3be6849426fb46cecfae419f5b1886f7c807b0141b427b7a2b0a0d8f0d8df0dc |
| script.py.txt | 1e85d5213a37550143367dc0e52ffc9c00616a7a0c0afde01fb6154df653f044 |
| stdout.log | f8d338cf6501effda5fa0a1c4da0924861095ba23bddbbf4197daea4a3b55c1b |
| supervisor_contract.json | d3854f5bb94cbddd8171d7e80bb145b5be901ff5ab61e5fd414663e15a3600e9 |
| supervisor_launcher.log | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| supervisor_outcome.json | 7340a600a08d875b48307987585962afafa7afc665a8630a77c15b394812c10f |
| supervisor_pid.txt | 882064bde568a4379c2b7effcce895105eb3968d4e5da93e07290f5b6911fb23 |
| terminal_attestation.json | 6ffe9dddb7a333a27497a631b8343f224b29954662b7df7bca5771238e20008a |
| timeline.rbl | 71698b4590c3f1c22d6cacb1967a66d58c389b6717d6ba89d3e879d224e43611 |
| timeline.rrd | 60834f8e5792f06e705deeffdb85fc18ddda68c15d485087ef9d65ac9e2dabcb |
| trace.npz | 60969f9d3359fc918b193811cb74d2e10ed0427f59f96125346c2ec7abf0fa9a |

Frozen dependencies are p16 v11 SHA-256
`1e85d5213a37550143367dc0e52ffc9c00616a7a0c0afde01fb6154df653f044`,
Supervisor V14 SHA-256
`dae90e385db3b830c1d5369fa1ea31c1595ff6a6250b0e7d35a05c04a541a888`,
and canonical prereg SHA-256
`3386376e191addcafef893b9ed3698b244daa2b352349f90bc536341212893eb`.

The exact retirement must reproduce the original stored context: bind the frozen generic
module cache name to pinned v11 for the entire frozen V14 semantic-gate and terminal
recomputation, assert path and SHA, then restore the prior binding by object identity.
Nested historical calls must continue to observe v11 in this scope. The exact stored
five false result leaves are:

- `numeric_trace_metrics_quaternions_counts_recomputed`;
- `retired_preflight10_completed_preclose_semantic_rejection_exact_and_nonpromotable`;
- `runtime_instrumentation_recomputed_not_trusted`;
- `source_prereg_p15_and_dependency_pins_exact`;
- `witness_event_latch_target_only_retreat_recomputed`.

The only false outer physics-gate leaf is
`pinned_result_semantic_validator_exact_all_true`. Retirement additionally requires raw
physics exit 0, combined exit 125, render attempts 0, diagnostic callbacks 2, task
callbacks 2,340, exact results/trace/outcome/terminal bytes, clean GPU deltas, successful
reaping, terminal attestation valid but pass false, no promotion, no scientific verdict,
and explicit `historical_cache_contamination_reproduced_not_promoted=true`.

## 3. Module loading and cache isolation

Supervisor V15 loads current p16 using a cache key derived from both resolved path and
full source SHA-256. It verifies the cached/loaded module's `__file__` and file SHA before
use. A pre-existing generic module name cannot select the current validator.

Historical scopes are separate and must never be combined:

1. p10 retirement loads frozen Supervisor V12 and pinned p16 v9 under unique keys, forces
   the frozen generic cache name to v9 for that call, validates identity, then restores;
2. p12 retirement loads frozen Supervisor V14 and pinned p16 v11 under unique keys, forces
   the generic cache name to v11 for its entire call, validates identity, then restores.

Pure adversarial tests must preload the generic name in different orders with absent,
v9, v11, current, and poison modules. Both historical results must remain exact and every
touched binding must be restored by object identity. Any preload-order dependence fails
G0 and no Isaac launch is allowed.

## 4. Witness command-only isolation contract

Cross-run realized joint state, velocity, force, and contact bytes are diagnostic only;
they are not command-isolation equality gates. Current-run numeric, limit, quaternion,
raw-count/capacity, contact-position, contact/filter, fixed-base, and callback gates remain
fully authoritative.

The witness semantic pass is recomputed without any `(environment_count, active_count)`
shortcut. It requires all of the following for preflight and canonical profiles:

- every active planned and applied target row exactly equals the active trace view;
- the five candidate-5 active planned and applied command rows exactly equal both frozen
  preflight10 and frozen preflight12 command bytes; one ULP fails;
- the same-step active target slice is unchanged by the excluded witness override;
- active-target mutation count is exactly zero;
- AST inspection of the task loop proves the override branch only assigns
  `target[witness_slot] = witness_safe_target` and increments its witness counter;
- the branch performs no state/object write, teleport, reset, forward, or extra step,
  and the task loop has exactly one `env.step` call;
- witness event, one-way latch, next-step activation, safe command, limits, and snapshots
  still independently recompute from all 2,340 current-run rows.

Cross-run actual-state differences may be reported only under a field explicitly marked
`diagnostic_only__not_a_witness_or_science_gate`.

## 5. Numeric integrity and measurement validity

The numeric semantic leaf is exactly the independently recomputed raw numeric report plus
`metrics.numeric_integrity`. It must not require the aggregate `measurement_valid` flag.
A one-ULP mutation of an authoritative current numeric value must still make the numeric
leaf false.

`measurement_valid` remains mandatory for runtime instrumentation, classification, and
scientific authority. It is independently recomputed as the conjunction of positive
control, contact buffers, witness command-isolation pass, and numeric integrity, and the
stored aggregate must equal that recomputation exactly. This separation repairs circular
classification coupling; it does not relax any runtime or scientific gate.

## 6. Admission, verdict, and forward-only boundary

G0 requires AST/syntax PASS, exact current profile/path identities, the p13 prereg hash
closure, all historical retirement checks, order-independent cache preload tests, the
pure actual-p12 witness/numeric regression, active-command one-ULP rejection, current
numeric mutation rejection, exact zero existing `t3u_side_preflight13_*` runtime outputs,
and no stale target process/GPU PID attributable to this tag.

If separately authorized after G0, execution remains host-detached `nohup setsid` using
the pinned IsaacLab Python. Raw zero alone never admits render or promotion. Every exact
semantic leaf must be true; otherwise Supervisor V15 returns reserved semantic status 125,
starts no render child, and records a nonpromotable outcome. A preflight PASS establishes
instrumentation readiness only and never a scientific grasp verdict. Canonical remains
blocked by `t3u_side_phys1_preflight13_prereg.md`.

Only these forward files belong to this repair: p16 v12, Supervisor V15, this prereg, and
the p13 canonical prereg. Preflight1 through preflight12 sources and artifacts are
immutable. No Isaac/PhysX/render/hardware run is authorized by this document.
