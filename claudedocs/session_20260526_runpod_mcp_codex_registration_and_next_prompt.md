# 2026-05-26 RunPod MCP Codex Registration And Next Prompt

## Scope

Operational continuity only. No Isaac runtime, no PPO/training, no rollout, no
dataset generation, no hold-lift, no transport/release, no constraints, no
SurfaceGripper, no B200 SSH/reconnect/pull, no `.ssh` copy, and no success claim.

## Why This Session Exists

The user pointed out that Claude currently has RunPod MCP available and asked
why Codex could not use it. The answer was not "Codex cannot use MCP"; it was
that the current Codex configuration did not have a RunPod MCP server registered.

This distinction matters for future sessions:

- Do not claim RunPod MCP is unavailable from memory alone.
- Do not claim RunPod MCP is available from config alone.
- Verify both the local MCP config and the current session's loaded tool
  namespace before deciding whether to use RunPod MCP or manual SSH/SCP.

## Evidence Verified

- Claude config `/home/cgxr/.claude.json` contains an MCP server named
  `runpod` with command `npx`, args `["-y", "@runpod/mcp-server@latest"]`, and
  env key `RUNPOD_API_KEY`. The key value was not printed.
- Before the update, Codex config `/home/cgxr/.codex/config.toml` contained MCP
  servers for context7, filesystem, sequential-thinking, memory, fetch, github,
  and arxiv, but no RunPod server.
- Added Codex config block:

```toml
[mcp_servers.runpod]
command = "npx"
args = ["-y", "@runpod/mcp-server@latest"]
env = { RUNPOD_API_KEY = "<redacted>" }
startup_timeout_sec = 30
```

- Backup made before editing:
  `/home/cgxr/.codex/config.toml.bak_runpod_20260526`
  md5 `1ef4acf6f1c92a64b9bbd79a2e35b7e7`.
- Post-edit redacted verification:
  `/home/cgxr/.codex/config.toml:71` has `[mcp_servers.runpod]`;
  line 73 has `@runpod/mcp-server@latest`;
  line 74 has `RUNPOD_API_KEY` present.
- Re-running `tool_search` in the same Codex session still did not expose an
  `mcp__runpod__...` namespace. Interpretation: this Codex session likely needs
  restart/new-session tool loading before RunPod MCP becomes callable.

## Track A Runtime Package Prepared

A minimal RunPod overlay was created and verified at:

`/tmp/track_a_v7_active_recovery_runpod_overlay_20260526.tar.gz`

md5:

`c5133dc4120e07595d8c6f060608345e`

It includes:

- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
- required helper scripts including `sim_scripts/roarm_kinematics.py`
- `roarm_rl/__init__.py`, `roarm_rl/roarm_stack_env.py`, and minimal agent cfg
  files needed for gym env registration/import
- full local backup USD directory
  `b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/`,
  not just the top USD file

Verified top USD:

`b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd`

md5:

`4497024d25abab11de5c50e144124553`

The top USD is a USD crate that references files under `configuration/`, so a
future RunPod transfer should carry the whole
`p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/` directory.

Overlay checks performed:

- archive contains no `HANDOFF.md`, no `TASKS.md`, no `.ssh`, no `JHPark`, no
  `__pycache__`, and no `.pyc`;
- extraction preserved top USD md5;
- Python compile checks passed after extraction;
- `conda run -n isaaclab` import of `roarm_rl` registration passed on the local
  IsaacLab env.

## Current Track A Truth Remains Unchanged

- v6 close_26 projected-guard audit is FAIL, not grasp success.
- v7 active recovery is implemented/static-ready only, not physics-validated.
- The approved 2026-05-26 local v7 runtime attempt was blocked by local
  CUDA/NVIDIA infrastructure before close steps; it is neither v7 contact
  success nor v7 contact failure.
- Dataset/training remain blocked until close_26 PASS, then hold-lift PASS, then
  small pilot dataset/replay PASS.

## Next Valid Action

In a new Codex session:

1. Read `CLAUDE.md`, `START_HERE.md`, `DECISIONS.md`, latest
   `EXPERIMENT_LEDGER.md`, and this session file.
2. Run `git status --short --untracked-files=all`.
3. Check whether `mcp__runpod__...` tools are actually loaded. If not, verify
   `/home/cgxr/.codex/config.toml:71-75` redacted and restart/new-session if
   needed.
4. Do not use stale RunPod pod `az53n8t8alp8pz` from 2026-05-06 unless the user
   explicitly confirms it is current and active.
5. Do not use B200 or `ssh JHPark`.
6. On CUDA-valid RunPod/local, run close_26-only v7 active recovery with the
   local backup USD path, then immediately run the v7 audit.
7. If audit fails, analyze exact first failing runtime/audit lines before any
   rerun.
8. If audit passes, do not start dataset/training; next gate is hold-lift PASS.
