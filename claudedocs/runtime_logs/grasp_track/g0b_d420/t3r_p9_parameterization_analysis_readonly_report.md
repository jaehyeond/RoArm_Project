# p9 Asset-Identity Parameterization Analysis (READ-ONLY)

Target: `/home/cgxr/Documents/Robotics/RoArm_Project/sim_scripts/p9_g0b_t3_cyld29h50_top_center_vertical_close_sweep_grasp_probe.py`

- **sha256 verification: PASS** — measured `99c99c65da75d5b77fff5c777ebf6d5628c6cbf3cdd528b156ff461d79dc2412` == required pin.
- **Line count: 1780** (`wc -l`). No file was edited; no simulation was run.
- External evidence consulted (read-only): `roarm_rl/roarm_stack_env.py:96-99` (env-side `ROARM_M3_USD_PATH` consumption), `sim_scripts/cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair.py:88-105` (full part prim-path scope), `claudedocs/runtime_logs/grasp_track/g0b_d420/t3_grasp2_stdout.log` (actual body_checks values: link5=64, gripper_link=64).

---

## 1. Asset-identity coupling inventory

Every site where the attempt3 path, root sha, physics-layer sha, part count 64, part naming, legacy-collider fragment, or asset-specific prim path appears or flows. **[B] = behavior-bearing, [D] = documentation/comment only.**

### 1.1 Module docstring [D]

| Lines | Quote (abridged) |
|---|---|
| 10 | `DESCEND run at --descend_open_deg (default = frozen OPEN; attempt3 measured` |
| 19–22 | `D-3  gripper collision body = frozen attempt3 asset REUSE ... ROARM_M3_USD_PATH -> g0a_d344 attempt3 64+64-part USD, root+physics layer sha pinned, stage audit (64 enabled part_* + exactly 1 disabled legacy node_STL_BINARY_* per body) hard-fails before any physics step.` |
| 29–30 | `env default USD path is the retired B200 path (HARD RULE #27) -> this probe sets ROARM_M3_USD_PATH itself and aborts on any /NHNHOME resolution.` |
| 76–77 | `MINOR the /NHNHOME guard now checks the post-import effective cfg.robot.spawn.usd_path` |
| 97 | `claudedocs/runtime_logs/grasp_track/g0b_d420/.` (output dir — case identity, see §5) |

### 1.2 Constants block [B] — primary hardcode cluster

| Lines | Quote |
|---|---|
| 137 | `# D-3: frozen attempt3 collision asset (REUSE ONLY — re-decomposition banned, D415).` [D] |
| 138–142 | `ATTEMPT3_USD = (REPO / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3" / "roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd")` |
| 143 | `ATTEMPT3_ROOT_SHA256 = "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff"` |
| 144 | `ATTEMPT3_PHYSICS_LAYER = ATTEMPT3_USD.parent / "configuration/roarm_m3_physics.usd"` — **second hardcode inside this line**: the relative sub-path `configuration/roarm_m3_physics.usd` (asset internal layout assumption) |
| 145 | `ATTEMPT3_PHYSICS_SHA256 = "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503"` |
| 146–149 | `BODY_PATHS = {"link5": "/World/envs/env_0/Robot/link5", "gripper_link": "/World/envs/env_0/Robot/gripper_link"}` — stage prim paths; `/World/envs/env_0/Robot` is the env spawn root (env-owned), the trailing body names are asset skeleton names |
| 150 | `EXPECTED_PART_COUNT = 64` — **single constant applied to BOTH bodies** (answering the per-body question: counts do NOT differ today; 64+64, confirmed in t3_grasp2 stdout) |
| 151 | `LEGACY_COLLIDER_FRAGMENT = "node_STL_BINARY_"` |

**Note on part naming**: the string `d338_convex_parts` does **not appear anywhere in p9**. The audit's part-classification predicate is leaf-name-only (`"part_" in leaf`, line 617), scoped by the body path prefix. The actual stage layout is `/World/envs/env_0/Robot/<body>/collisions/d338_convex_parts/part_NNN` (evidence: `cyl34_top_view_d339_...py:105`). This matters for the incremental-arm mode design (§3, arg 8–10): new `part_NNN` prims in a *new* namespace under the same body would be counted by the current predicate and correctly fail `part_count_64` (fail-closed), so the parameterized audit must classify by namespace fragment *before* the leaf-name test.

### 1.3 Stage-audit functions [B]

| Lines | Quote |
|---|---|
| 573–575 | `# D-3 stage audit (copied from cyl34_top_view_d334...:197-239 inventory + cyl34_top_view_d349...:921-934 body checks, adapted to the attempt3 layout).` [D] |
| 581 | `body_path = BODY_PATHS[body_label]` |
| 614 | `for body in BODY_PATHS:` |
| 617 | `enabled_parts = [r for r in enabled if "part_" in r["path"].rsplit("/", 1)[-1]]` — part-naming predicate |
| 618 | `legacy = [r for r in rows if LEGACY_COLLIDER_FRAGMENT in r["path"]]` |
| 621 | `"enabled_part_count": len(enabled_parts),` |
| 622 | `"part_count_64": len(enabled_parts) == EXPECTED_PART_COUNT,` — **the literal "64" is baked into the JSON/log KEY NAME**, not only the value |
| 623 | `"enabled_only_parts": len(enabled) == len(enabled_parts),` |
| 624–625 | `"legacy_rows": len(legacy), "disabled_legacy_exact_one": len(legacy) == 1 and legacy[0]["collision_enabled"] is False,` |
| 627–630 | `audit_pass = all(row["part_count_64"] and row["enabled_only_parts"] and row["disabled_legacy_exact_one"] for row in body_checks.values())` |

### 1.4 Pre-Isaac guard block [B]

| Lines | Quote |
|---|---|
| 815 | `# ---- D-3 / D-6 guards: frozen USD injection + sha pins, before any Isaac import` [D] |
| 816 | `if not ATTEMPT3_USD.exists() or not ATTEMPT3_PHYSICS_LAYER.exists():` |
| 817 | `print(f"[{LOG}] G0B_T3_GRASP_VERDICT=USD_GUARD_FAIL missing attempt3 asset", flush=True)` — asset name in log string |
| 819–820 | `root_sha = _sha256_file(ATTEMPT3_USD)` / `physics_sha = _sha256_file(ATTEMPT3_PHYSICS_LAYER)` |
| 821 | `if root_sha != ATTEMPT3_ROOT_SHA256 or physics_sha != ATTEMPT3_PHYSICS_SHA256:` — the sha equality gate |
| 823 | `f"...USD_GUARD_FAIL sha mismatch root={root_sha} physics={physics_sha}"` |
| 830 | `os.environ["ROARM_M3_USD_PATH"] = str(ATTEMPT3_USD)` — injection consumed at import time by `roarm_stack_env.py:96-99` (`USD_PATH = os.environ.get("ROARM_M3_USD_PATH", "/NHNHOME/...")`) |
| 832 | `f"[{LOG}] usd_guard PASS path={ATTEMPT3_USD} root_sha={root_sha[:16]} physics_sha={physics_sha[:16]}"` |

### 1.5 Banner + effective-path guard [B]

| Lines | Quote |
|---|---|
| 863 | `"q5_convention=LARGE_IS_OPEN(D-1) marker=probe_patched(D-2) collision_asset=attempt3_frozen(D-3)"` — asset identity claim in the run banner |
| 900 | `effective_usd = str(cfg.robot.spawn.usd_path)` |
| 901 | `if effective_usd != str(ATTEMPT3_USD) or "/NHNHOME" in effective_usd:` — exact string equality against the constant + HARD RULE #27 check |
| 903–904 | `f"...USD_GUARD_FAIL effective usd_path={effective_usd} " "(expected attempt3 injection; HARD RULE #27)"` |
| 909 | `print(f"[{LOG}] usd_effective PASS cfg.robot.spawn.usd_path={effective_usd}", ...)` |

### 1.6 Stage-audit invocation [B]

| Lines | Quote |
|---|---|
| 952 | `audit_pass, body_checks = _audit_collision_bodies(base_env)` |
| 953 | `print(f"[{LOG}] usd_stage_audit pass={_yes(audit_pass)} body_checks={json.dumps(body_checks)}", ...)` — `part_count_64` key appears verbatim in stdout |
| 954–957 | `if not audit_pass: print(f"[{LOG}] G0B_T3_GRASP_VERDICT=USD_AUDIT_FAIL", ...); _close_all(); return 3` |

### 1.7 Rerun summary document [B — content of a shipped artifact]

| Lines | Quote |
|---|---|
| 1529–1530 | `f"- collision asset (D-3): attempt3 64+64 frozen, root {ATTEMPT3_ROOT_SHA256[:16]}, " f"physics {ATTEMPT3_PHYSICS_SHA256[:16]}, stage audit pass={audit_pass}\n"` — prints the **PIN constants** (not the measured shas; equivalent only because gate 821 already enforced equality) and the literal `64+64` |

### 1.8 Results JSON [B — provenance record]

| Lines | Quote |
|---|---|
| 1627 | `"artifact": "G0B_T3_CYLD29H50_TOP_CENTER_VERTICAL_GRASP_V1",` (schema version — see §4) |
| 1628 | `"case": "g0b_d420",` (case identity, §5) |
| 1631–1637 | `"usd": {"path": str(ATTEMPT3_USD), "root_sha256": root_sha, "physics_sha256": physics_sha, "stage_audit_pass": audit_pass, "body_checks": body_checks}` — records **measured** values only; the *expected* pins/counts are recorded nowhere in JSON (only implied by gate PASS) |

### 1.9 Comments that look asset-related but are NOT [D — do-not-touch warnings for the editor]

- **686** `# Attempt3 (t3_grasp3): attempt2 measured that a FULL-OPEN (88.31 deg)` and **695** `# Attempt2 (t3_grasp2): ...` — these "Attempt" numbers are **run-attempt numbering of the t3 probe legs**, not the attempt3 collision *asset*. Do not rename when parameterizing.
- **327** `@dataclass(frozen=True)` / 129, 692, 761, 766 "frozen OPEN" — D-1 gripper convention, unrelated to the frozen asset.

---

## 2. Flow map (constant → consumers → output fields)

| Constant | Consumed by (gates/logic) | Recorded in (outputs) |
|---|---|---|
| `ATTEMPT3_USD` (138–142) | existence gate 816; sha source 819; env injection 830 (→ `roarm_stack_env.py:96` at import 848); effective-path equality gate 901 | usd_guard PASS log 832; fail logs 817/823/903; results JSON `usd.path` 1632 |
| `ATTEMPT3_ROOT_SHA256` (143) | sha equality gate 821 (exit 3 on mismatch) | summary_md 1529 (truncated pin); measured twin `root_sha` → log 832, JSON `usd.root_sha256` 1633 |
| `ATTEMPT3_PHYSICS_LAYER` (144, derived from 138–142 + relative sub-path) | existence gate 816; sha source 820 | (path itself never logged — only its sha) |
| `ATTEMPT3_PHYSICS_SHA256` (145) | sha equality gate 821 | summary_md 1530; measured twin `physics_sha` → log 832, JSON `usd.physics_sha256` 1634 |
| `BODY_PATHS` (146–149) | inventory scope filter 581–586; audit iteration 614 | body names are the KEYS of `body_checks` → stdout 953, JSON 1636 |
| `EXPECTED_PART_COUNT` (150) | per-body equality check 622 → conjunction 627–630 → `audit_pass` → exit-3 gate 954–957 | `body_checks[*]["part_count_64"]` (name + value) → stdout 953, JSON 1636; literal `64+64` text in summary_md 1529 |
| `"part_"` leaf predicate (617) | defines `enabled_part_count` 621 and `enabled_only_parts` 623 | `body_checks` fields → stdout 953, JSON 1636 |
| `LEGACY_COLLIDER_FRAGMENT` (151) | legacy row filter 618 → `disabled_legacy_exact_one` 625 → `audit_pass` | `body_checks[*]["legacy_rows"|"disabled_legacy_exact_one"]` → stdout 953, JSON 1636 |
| `audit_pass` (627) | exit gate 954–957 (`USD_AUDIT_FAIL`, return 3, before any physics step) | stdout 953; summary_md 1530; JSON `usd.stage_audit_pass` 1635 |
| literal `attempt3` log strings (817, 863, 904, 1529) | none (display only) | stdout / rrd metadata — **identity claims an adversarial reviewer will grep**; must become parameter-derived or they will lie for B/F/D runs |

Ordering (must be preserved by any edit): argparse (754) → validation (756–797) → tag/artifact guard (799–813) → **existence+sha gate (816–826)** → **env-var injection (830)** → AppLauncher (838–840) → `import roarm_rl` (848, reads env var at module import) → RERUN version gate (853) → cfg composition (897) → **effective-path gate (900–908)** → `gym.make` (944) → **stage audit (952–957)** → physics.

---

## 3. Minimal parameterization design (argparse; defaults = frozen attempt3 values)

Design principle: **one audit code path, two modes.** Mode A (all defaults / no incremental args) is decision-identical to today. Mode B (incremental arm) activates *additional* gates; nothing is ever relaxed. All equality gates stay `==`.

Move lines 138–151 into `main()` after argparse (or keep module constants as DEFAULT_* and bind from args) — the guard block 816+ then reads only args-derived values.

### Arg 1: `--asset_usd`
- type: `str` (→ `Path(v).resolve()` immediately; resolve is REQUIRED to keep the 901 string-equality semantics — the current constant is resolved via `REPO = Path(...).resolve()` at 115)
- default: `str(REPO / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd")`
- lines changed: 138–142 (constant → default), 816, 817, 819, 830, 832, 901, 903, 1632
- strictness: unchanged — existence + sha gates still hard-fail (exit 3) before any Isaac import.

### Arg 2: `--asset_physics_usd`
- type: `str`, default: `None` → derived as `asset_usd.parent / "configuration/roarm_m3_physics.usd"` (preserves line 144 behavior exactly); explicit path for arms whose repair layer lives elsewhere
- lines changed: 144, 816, 820
- strictness: unchanged (existence + sha gate). Deriving-by-default means an arm that silently lacks the physics layer still aborts at 816.

### Arg 3: `--asset_root_sha256`
- type: `str`, default: `"a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff"`
- validation (add next to 791): `re.fullmatch(r"[0-9a-f]{64}", v)` else ValueError — prevents a truncated/uppercase pin (a wrong-format pin would fail closed anyway, but the explicit check makes the error attributable)
- lines changed: 143, 821, 823, 1529
- strictness: equality gate unchanged. **Fail-closed cross-check**: if `--asset_usd` is non-default and the shas are left at defaults, gate 821 fails (attempt3 pins cannot match a different file). No skip flag exists or may be added.

### Arg 4: `--asset_physics_sha256`
- type: `str`, default: `"043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503"`; same regex validation
- lines changed: 145, 821, 823, 1530
- strictness: identical to Arg 3.

### Arg 5: `--asset_label`
- type: `str`, default: `"attempt3"`, validated `[A-Za-z0-9_]+` (reuse the 791 tag pattern)
- purpose: makes the identity claims in logs truthful for new arms
- lines changed: 817 → `f"...missing {args.asset_label} asset"`; 863 → `f"collision_asset={args.asset_label}_frozen(D-3)"`; 904 → `f"(expected {args.asset_label} injection; HARD RULE #27)"`; 1529 → `f"- collision asset (D-3): {args.asset_label} {n}+{n} frozen, ..."`
- strictness: display-only; under defaults every string is byte-identical to today.

### Arg 6: `--expected_part_count`
- type: `int`, default: `64`; validation `>= 1`
- applied per body, uniformly, exactly as today (answer to "per body?": the current script uses ONE constant for BOTH bodies; body_checks confirmed 64/64)
- lines changed: 150, 622 (`== args.expected_part_count`), 1529 (`{n}+{n}`)
- key naming: change 622's key to the dynamic `f"part_count_{args.expected_part_count}"` and 628 to read the same dynamic key. Under defaults this is byte-identical (`part_count_64`); for a 62-part arm it self-describes (`part_count_62`) instead of lying. (Alternative — a stable key `part_count_expected` — is cleaner schema but changes default-run bytes; see §4.)
- strictness: `==` preserved; a `>=` or `min()` anywhere here is a review-rejectable weakening.

### Arg 7 (optional, forward provision): `--expected_part_count_per_body`
- type: repeatable `str` `"body=count"` (e.g. `link5=62 gripper_link=64`), default: absent → uniform Arg 6 applies
- validation: every key must be an existing `BODY_PATHS` key; every audited body must be either covered by the uniform count or listed — no silent fallback mixing
- lines changed: 622 (lookup per body)
- rationale: a plug-deactivation-style arm (e.g. link5 part_029/030 disabled) changes only link5's enabled count. Not needed for a pure-additive F/D arm; include only if such an arm is actually planned, otherwise defer (Variable Ladder: do not implement future-looking ideas — listing here as design option, decision to lead).

### Arg 8: `--new_part_namespace_fragment` (mode B switch)
- type: `str`, default: `None` (mode A). When given: non-empty, `[A-Za-z0-9_]+`, must NOT contain `"part_"` and must differ from both the legacy fragment and the original fragment (classification disjointness — see gates below)
- lines changed: 617–630 region (audit classification), 953 (body_checks gains fields), 1529 (summary text gains `+N new parts in <ns>`)

### Arg 9: `--expected_new_part_count` (required iff Arg 8 given)
- type: repeatable `str` `"body=count"` — **every body in `BODY_PATHS` must be listed explicitly** (0 is allowed but must be written; no implicit default → forces a conscious per-body declaration)
- lines changed: audit function (new exact-count gate)

### Arg 10: `--original_part_namespace_fragment`
- type: `str`, default: `"d338_convex_parts"` — factually verified as the real stage scope: `/World/envs/env_0/Robot/<body>/collisions/d338_convex_parts/part_NNN` (`cyl34_top_view_d339_...py:105`)
- **consumed only in mode B** (mode A keeps the exact 617 predicate, guaranteeing byte-equivalent default behavior)

### Arg 11: `--legacy_collider_fragment`
- type: `str`, default: `"node_STL_BINARY_"`, non-empty validation (an empty fragment would match every row and explode `legacy_rows` — fail-closed but misattributed)
- lines changed: 151, 618

### Mode B audit semantics (the "original 64+64 verbatim preserved AND N new parts in namespace X" contract)

Replace the single classification at 617 with a three-way partition of `enabled` rows per body, evaluated in this order:
1. `new_rows` = rows whose path contains `new_part_namespace_fragment`
2. `orig_rows` = rows whose path contains `original_part_namespace_fragment`
3. `stray_rows` = everything else

Gates (ALL must hold, per body; each is `==`):
- `orig_part_count_ok`: `len([r in orig_rows if "part_" in leaf])` **== `expected_part_count`** (64 — unchanged strictness on the original set)
- `orig_only_parts`: every `orig_rows` member passes the `part_` leaf test (mirror of today's `enabled_only_parts`, scoped)
- `new_part_count_ok`: `len(new_rows)` **== declared per-body count** (exact; not `>=`, not `<=`)
- `no_stray_enabled`: `len(stray_rows) == 0` (replaces/strengthens `enabled_only_parts` globally)
- `namespace_disjoint`: no row matches BOTH fragments (hard error → `USD_AUDIT_FAIL`, catches fragment-substring footguns)
- `disabled_legacy_exact_one`: unchanged (exactly 1 disabled legacy row per body)
- mode-B-only strengthening: `no_undeclared_disabled`: disabled rows that are neither the legacy node nor (if Arg 7-style allowlisting is adopted) explicitly declared → FAIL. (In mode A this stays unchecked to preserve today's semantics — see §4 latent-gap note.)

**Scope limit to state in the prereg**: the in-stage audit proves *structure* (counts/namespaces/enable-state). "Original 64+64 **verbatim**" (bit-level geometry identity) is proven by the **sha pins of the new asset's layers** against the externally-audited authoring gate ledger (D426 3-condition contract), not by this script. The parameterization keeps that division of labor; it does not claim in-stage geometric verbatim-ness.

### Provenance additions to results JSON (all modes)

Add one block (see §4 for the byte-equivalence tension):

```json
"asset_params": {
  "asset_label": ..., "asset_usd": ..., "asset_physics_usd": ...,
  "expected_root_sha256": ..., "expected_physics_sha256": ...,
  "expected_part_count": ..., "expected_part_count_per_body": ...,
  "original_part_namespace_fragment": ..., "new_part_namespace_fragment": ...,
  "expected_new_part_count": ..., "legacy_collider_fragment": ...,
  "audit_mode": "frozen_default" | "incremental_arm"
}
```

Without this, a results.json from a B/F/D run is indistinguishable from a default run except by the sha values — insufficient for the (B,F,D)×{pass, fail-consistent, off-prediction} attribution matrix.

---

## 4. Risks / invariants (silent-weakening audit)

1. **Sha pins must never become optional.** Defaults ARE the attempt3 pins, so `--asset_usd` changed + pins forgotten = guaranteed `USD_GUARD_FAIL` (fail-closed by construction). Do not add any `--skip_sha`/`--no_verify` escape hatch; do not accept empty-string shas (regex gate).
2. **`==` must stay `==`** at: sha gate 821, part count 622, new-part count (new), legacy `== 1` at 625, effective-path equality 901. Any `>=`, `in`-set, or prefix-match relaxation is a gate weakening.
3. **`/NHNHOME` check at 901 is independent of parameterization** (HARD RULE #27) and must remain unconditional — it must not be folded into the label/path equality in a way that a custom `--asset_usd` bypasses it.
4. **Path resolution**: 901 compares `str(cfg.robot.spawn.usd_path) != str(ATTEMPT3_USD)`. If the new arg is not `.resolve()`d, a symlinked/relative user path passes the sha gate (same bytes) but fails or — worse under future edits — string-compares unequal semantics. Resolve at argparse time; keep the equality on the resolved string.
5. **Ordering invariants** (breaking any silently voids the audit):
   - sha gate (816–826) BEFORE `os.environ` injection (830) BEFORE `AppLauncher` (838) BEFORE `import roarm_rl` (848) — the env module snapshots `ROARM_M3_USD_PATH` at import (`roarm_stack_env.py:96-99`); a post-import injection would silently load the retired B200 default path (only the 901 effective check would catch it).
   - stage audit (952) must stay AFTER `gym.make` (944) and BEFORE any commanded physics (first `env.step` inside settle loop 1131).
6. **JSON/log key `part_count_64`** (622 → stdout 953 → results 1636): the "64" lives in the KEY. Dynamic key `f"part_count_{n}"` preserves default bytes and self-describes; a frozen literal key would lie for a 62-part arm; a renamed stable key changes default-run output. Decision must be explicit in the prereg. Whatever is chosen, 628 must read the same key it writes (a mismatch would make `audit_pass` throw `KeyError` — loud, but attribute it correctly).
7. **summary_md 1529–1530 prints the PIN constants, not the measured shas.** Equivalent today only because gate 821 enforces equality first. When parameterizing, print the args (or the measured values) — but never print a stale module constant while running a different asset: the rrd metadata panel would then assert the wrong identity (D341 visual-inspection evidence would lie).
8. **Banner strings 817/863/904/1529 must derive from `--asset_label`** or they misattribute B/F/D runs to attempt3 in stdout — the exact thing an adversarial reviewer greps.
9. **Byte-equivalence tension**: the task requires default runs to preserve behavior EXACTLY (audit-semantics byte-equivalence) and simultaneously requires provenance fields. Gate decisions, exit codes, stdout gate lines, and body_checks content can be kept byte-identical under defaults; the `asset_params` JSON block is a schema addition visible in default runs too. Options: (a) add the block unconditionally and bump `"artifact"` V1→V2 (1627) — honest, but breaks cross-run schema greps; (b) add unconditionally, keep V1 — schema drift within a version tag; (c) add only in mode B — default runs stay byte-identical but a default run no longer proves its parameters were default. Recommendation: **(a)**, declared in `t3_prereg.md` before the first parameterized run. This is a lead/prereg decision, not an editor decision.
10. **Latent audit gaps that parameterization must NOT silently "fix" in mode A** (D-8-style: gate revision is a separate preregistered leg):
    - Disabled non-legacy rows are currently unconstrained (a row with `collision_enabled=False`, leaf `part_029`, no legacy fragment → counted by NO check; only the enabled-count drop catches full deactivation of a previously-enabled part). Mode A must keep this; mode B may strengthen (new gates are additive there).
    - `collision_enabled = None` (attr unset) is treated as `True` (605) — USD default semantics; keep.
    - `approximation` is recorded (601–607) but no gate consumes it — a repair arm swapping convexHull→boundingCube on one part would pass the stage audit (sha pin is the only defense). Note for the adversarial review; do not add a gate in this parameterization.
    - Bodies not in `BODY_PATHS` are never audited — an arm adding colliders under `link4` would be invisible to the stage audit (again: sha pin + authoring-gate ledger are the defense). State explicitly in the prereg.
11. **Tag/artifact overwrite guard (799–813) is the only collision defense for multi-arm runs** in the shared `g0b_d420` folder — each arm/leg needs a fresh `--tag`; the guard aborts (exit 3) on any existing artifact, unchanged.
12. **`--expected_new_part_count` must require explicit per-body values** (0 included). A default of 0 for unlisted bodies would let a typo'd body name silently skip the new-part gate for that body.

---

## 5. Other hardcodes (scope classification)

| Lines | Item | Scope |
|---|---|---|
| 122 | `LOG = "g0b_t3_grasp"` stdout namespace | **OUT** — case identity, prereg-frozen |
| 123–124 | `RERUN_VERSION = "0.34.1"`, `RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"` | **OUT** — D341 environment pin (version gate 853, validation 1751–1752) |
| 126 | `TABLE_Z = -0.012117` | **OUT** — measured physical constant (D-5) |
| 127 | `HOME_ARM_DEG` | **OUT** — HARD RULE #1 lineage |
| 130–131 | `Q5_OPEN_RAD = 1.5413` | **OUT** — D-1 frozen convention (validation 765–776 consumes) |
| 134–135 | `GRASP_MARKER_DIST_M = 0.030` (CLI-overridable via `--marker_dist_m` 700), `GRASP_MARKER_Q5_MAX_DEG = 41.40` (hardcoded; flows to 767, 879, 971, 1527, 1677) | **OUT** — D-2 prereg marker contract |
| 155–166 | `SOURCE_REGIONS`, `FOUR_SPONGE_SEED0_SOURCES` | **OUT** — p7/T2 prereg spawn poses |
| 168 | `OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d420"` | **OUT** — case folder per Variable Ladder forward-only rule; per-leg separation is `--tag` (752, 800–805). New arms stay in this folder with fresh tags |
| 670–753 | object tuple (D29×H50 size, 0.02483 kg, μ 0.40/0.30, rest 0.0), all gates, close sweep band | **OUT** — already CLI args with prereg-frozen defaults (D-4); not part of asset identity |
| 944 | `gym.make("RoArm-Stack-Direct-v0", ...)` | **OUT** — env identity |
| 1102–1103 | `app_id = f"roarm_g0b_{args.tag}"`, `recording_id=f"g0b_d420_{args.tag}"` | **OUT** — case+tag namespace; tag-scoped, no collision under fresh tags |
| 1627–1628 | `"artifact": "G0B_T3_..._V1"`, `"case": "g0b_d420"` | **OUT** as identity, but the `V1` version is **touched** if provenance fields are added (§4.9) |
| 1719–1754 | Rerun entity/timeline/component contracts | **OUT** — D341 observability contract, asset-independent |
| TCP offset | not in this script (`roarm_stack_env.py:86` `TCP_LOCAL_OFFSET_M`) | **OUT** — env-owned |

**IN SCOPE for this parameterization (asset identity only)**: lines 138–151 (constants), 581/614/617/618/621–630 (audit), 816–832 (guard), 863 (banner), 901–904 (effective check), 1529–1530 (summary), 1632–1636 (JSON usd block), plus docstring lines 19–22/29–30/137/573–575 as text updates.

---

## Cited line-number ranges (jump list for the editor)

Docstring: 10, 19–22, 29–30, 76–77, 97 · Constants: 122–124, 126–127, 130–131, 134–135, 137–151, 155–166, 168 · Audit fns: 573–575, 577–609 (581, 594–607), 612–631 (614, 617, 618, 621–625, 627–630) · argparse/validation: 669–754, 756–797 (esp. 700, 752, 791) · tag guard: 799–813 · USD guard: 815–834 (816, 817, 819–826, 830, 832) · banner: 858–864 (863) · effective check: 897–909 (900–904, 909) · stage audit call: 951–957 · summary_md: 1522–1545 (1529–1530) · blueprint/artifacts: 1102–1103, 1583–1597 · results JSON: 1626–1717 (1627–1637, 1661–1683, 1715) · validation: 1719–1755 · non-asset "Attempt" comments (do not rename): 686, 695.

**Counts**: 26 behavior-bearing asset-identity sites + 10 documentation-only sites. Proposed args: 6 core (`--asset_usd`, `--asset_physics_usd`, `--asset_root_sha256`, `--asset_physics_sha256`, `--asset_label`, `--expected_part_count`) + 4 incremental-arm mode B (`--new_part_namespace_fragment`, `--expected_new_part_count`, `--original_part_namespace_fragment`, `--legacy_collider_fragment`) + 1 optional forward provision (`--expected_part_count_per_body`).
