# Session 2026-05-22 — OpenVLA-OFT v6 Offline Eval — PARTIAL (3/12 ckpts)

## TL;DR

Track B P3 (offline eval of 12 OpenVLA-OFT v6 LoRA ckpts on B200) is still
IN FLIGHT. Original ETA 14:21 KST was based on dryrun-derived 0.62s/frame.
Measured ckpt 2500 actual is **1.21s/frame on holdout + 1.17s/frame on
train_sanity + 36.5s load = ~19.7 min/ckpt**, so revised ETA **~16:14 KST**.
Deadline corrected to **23:59 KST** per user 2026-05-22 12:30 KST (the in-progress
doc's 15:00 KST was wrong). Three checkpoints have finished:
**24.18° → 22.41° → 22.16°** on holdout `l2_step0_mean`. Holdout L2 is
monotonically decreasing so far, so the R-OFT-2 early-overfit hypothesis is
NOT confirmed yet — needs 15K/20K/25K/30K data to see turnaround.

This session is ending at ~89% context per HARD RULE #11 (no /half-clone). The
eval continues on B200; the next session must pull final JSON and finalize.

## Verified State at Session End (KST ~13:33)

### Eval process

- B200 nohup PID **3507531**, ALIVE at 13:30:28 KST.
- Process: `python -u eval_offline_v6.py --base_model openvla/openvla-7b
  --checkpoint_root .../outputs/openvla_oft_v6_b200 --dataset_repo_id roarm_v6_pick
  --dataset_root .../data/lerobot_dataset_v6 --holdout_episodes 45 46 47 48 49
  --train_sanity_episodes 0 1 --output .../openvla_oft_v6_eval_20260522_121028.json
  --dtype bfloat16`.
- Elapsed at 13:30: 1h 20min 12s.
- RSS 2.4 GB, %CPU 1509% (15 cores), GPU mem 16410 MiB, util mostly 0%/sample
  10s = single 25% spike → CPU/data-pipeline bound, not GPU compute.
- B200 NVML mismatch persists → use `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05`
  for any nvidia-smi call (D024).

### Completed ckpts (3/12)

| step | holdout.l2_step0_mean | holdout.l2_chunk | train_sanity.l2_step0 | gap_step0 |
|---:|---:|---:|---:|---:|
| 2500 | 24.1829° | 23.8242° | 13.5086° | 10.6743° |
| 5000 | 22.4112° | TBD     | TBD       | TBD       |
| 7500 | 22.1624° | TBD     | TBD       | TBD       |

Source: B200 stdout `grep step= /tmp/openvla_oft_v6_eval_full_20260522_121028.out`
gives 3 `step=N l2_step0_mean=X` lines as of 13:29 KST. ckpt 5000/7500 full
detail in JSON (only ckpt 2500 detail pulled to Lenovo so far).

### Ckpt 2500 per-joint pattern (memorization ratio holdout/train_sanity)

| joint | holdout MAE | train MAE | ratio |
|---|---:|---:|---:|
| J0 base    | 11.613° | 2.634° | **4.41×** |
| J1 shoulder| 2.674°  | 4.896° | 0.55× |
| J2 elbow   | 6.459°  | 4.529° | 1.43× |
| J3 wrist_p | 6.522°  | 5.938° | 1.10× |
| J4 wrist_r | 11.917° | 2.130° | **5.59×** |
| J5 gripper | 4.215°  | 4.679° | 0.90× |

Memorization is asymmetric: J0 (base rotation) and J4 (wrist roll), the two
largest-range joints with highest training variance, show clear overfit on
seen episodes already at step 2500. The other 4 joints are roughly balanced
(0.55-1.43×). z-score per joint 0.09-0.47 — normalized error is moderate
even though degrees look large.

### Files

- `openvla_oft_roarm/eval_offline_v6.py` md5 `3ef67f2b558547623f6b3d03e8e98b4f`
- `openvla_oft_roarm/launch_eval_offline_v6.sh` md5 `8976cf43d41e150679e9663e2e471ebf`
- `openvla_oft_roarm/rank_eval_v6.py` md5 `10edd0485c4c754e0ac9ad4ca7370cf5`
  (new ranker; reads JSON, prints 6 tables, identifies best/early-best/final,
  flags overfit; sanity-tested on ckpt 2500 partial).
- Partial JSON pulled `openvla_oft_b200_pulls/openvla_oft_v6_eval_20260522_121028_partial.json`
  sha256 `75cb3a920850a6fb416f919597e9f5f456e2cfc202caaec461f68100ad1c2caf`
  (only ckpt 2500 entry; pulled 12:30 KST before ckpt 5000/7500 written).
- B200 JSON current size: 10644 bytes at 13:11:58 (3 ckpts written).
- B200 `outputs/openvla_oft_v6_eval/` contains dryrun.json, dryrun2.json,
  dryrun3.json, dryrun4.json, openvla_oft_v6_eval_20260522_121028.json.
- Background watcher Bash ID `bdz39iyyv` STILL ALIVE at 13:28 (52+ min). Output
  at `/tmp/claude-1000/-home-cgxr-Documents-Robotics-RoArm-Project/67d354fa-2c10-4041-8ff9-a39239b58712/tasks/bdz39iyyv.output`.
  Heartbeats every 3 min. Will exit when PID 3507531 dies + dump tail of
  `/tmp/openvla_oft_v6_eval_full_20260522_121028.out` + outputs dir listing.

## ETA Math

- Per-ckpt actual: 36.5s load + 741 fr × 1.21s/fr + 207 fr × 1.17s/fr +
  ~5s write = **~1183s ≈ 19.7 min**.
- 12 ckpts × 19.7 min = 236 min from kickoff 12:10:28 KST → **~16:06 KST**.
- Plus alignment with measured ckpt 5000 start 12:30:11 = +~20 min per ckpt
  × 11 remaining ckpts from there = **~16:14 KST**.
- 23:59 KST deadline → ~7.7h slack.

## What Next Session Must Do

1. **SSH check**: `ssh JHPark 'set -e; source env.sh; [[ -z "$ROARM_B200_ROOT" ]] && exit 1; (ps -p 3507531 -o pid= >/dev/null && echo ALIVE) || echo DONE; tail -60 /tmp/openvla_oft_v6_eval_full_20260522_121028.out; ls -la $ROARM_B200_ROOT/outputs/openvla_oft_v6_eval/'`
2. **Pull final JSON** to `openvla_oft_b200_pulls/openvla_oft_v6_eval_20260522_121028.json`.
3. **Rank**: `python openvla_oft_roarm/rank_eval_v6.py openvla_oft_b200_pulls/openvla_oft_v6_eval_20260522_121028.json`.
4. Identify: `best_overall`, `best_early(<=10K)`, `final(highest)`. Report
   R-OFT-2 overfit flags (gap widens / holdout worsens).
5. Write `claudedocs/session_20260522_openvla_oft_offline_eval_v6_result.md`
   (the actual successor — this partial doc is a checkpoint, not the result).
6. Append `claudedocs/EXPERIMENT_LEDGER.md` row "2026-05-22 (Track B P3 OpenVLA-OFT v6
   offline eval COMPLETE)" with best ckpt + gap evidence + JSON sha256.
7. Update `START_HERE.md` Track B section. **Do not touch Track A timestamp/sections** —
   the parallel session owns Track A v4 recovery FAIL state (lines 1-148 as of 13:30 KST).
8. Prepend MEMORY.md recent-session entry for P3 result. If Track A already
   prepended a v4 entry, then archive the oldest per HARD RULE #8 5-slot rule
   (move full body to `MEMORY_archive_20260522.md`, leave one-line index pointer).

## HARD RULE Compliance This Session

- ✅ #4: All metrics cross-verifiable from B200 log line numbers + JSON sha256.
- ✅ #11: `/half-clone` refused twice (85% and 89% hooks). End-of-session
  protocol invoked early at 89% (before 95% emergency threshold).
- ✅ #14: All ssh commands have fail-fast guard (`set -e`, `source env.sh`,
  ROARM_B200_ROOT/user check).
- ✅ #15: torch nightly cu128 + transformers 4.57.6 (from in-progress doc).
- ✅ #18: User explicit corrections respected — deadline 23:59 KST (not 15:00),
  action = "그대로 계속" not kill. No reinterpretation.
- N/A #17/#26: No Isaac Sim / RL / sim render involvement.

## Track A Boundary

Track A v4 close_26 runtime FAIL was recorded by a parallel session into
START_HERE.md lines 1-148 between 12:35 KST and 13:35 KST. This session did
not touch any Track A scripts, audits, or session docs. Track A and Track B
remain independent per the user's explicit two-track design.
