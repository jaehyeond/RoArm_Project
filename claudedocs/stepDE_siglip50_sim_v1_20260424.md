# Step D SigLIP 50-ep + Step E sim_v1 LeRobot v3 Dataset (2026-04-24)

## Step D — 50-episode SigLIP distribution (full, n=6942 frames)

Command (already finished in prior session, background task `bhhyi7tys`):
```
conda run -n roarm python sim_real_compare.py --all --max-frames 300 \
  --sim-dir sim_renders_v2 --output sim_real_compare_v2_all50.json
```

### Frame-level (n=6942)

| Stat | Value |
|---|---|
| mean | **0.7232** |
| std | 0.0300 |
| min / max | 0.6075 / 0.8247 |
| frames ≥ 0.70 | 5404/6942 (77.8%) |
| frames ≥ 0.75 | 1329/6942 (19.1%) |
| frames < 0.65 | 39/6942 (0.6%) |

### Episode-level (n=50)

| Stat | Value |
|---|---|
| episode-mean of means | **0.7222 ± 0.0157** |
| median | 0.7201 |
| min ep | ep45 = 0.6909 (LEFT) |
| max ep | ep11 = 0.7622 (RIGHT) |
| GO episodes (≥0.70) | **48/50** |
| NO-GO | ep43=0.6911 (LEFT), ep45=0.6909 (LEFT) |

### Zone breakdown

| Zone | N | mean | std |
|---|---:|---:|---:|
| FAR_RIGHT | 11 | 0.7406 | 0.0057 |
| RIGHT | 19 | 0.7218 | 0.0142 |
| CENTER | 5 | 0.7170 | 0.0056 |
| LEFT | 15 | 0.7111 | 0.0126 |

**LEFT is systematically worst; both NO-GO episodes are LEFT.** Likely driver: Kinect calibration RMSE residual (10mm) + sponge pose recovery may be slightly less accurate on the camera's LEFT side. Still well above 0.70 threshold overall.

### Sample-5 extrapolation check (from 2026-04-24 earlier session)

| | Sample 5 (predicted) | Full 50 (actual) |
|---|---:|---:|
| mean | 0.7212 | 0.7222 |
| std | 0.0128 | 0.0157 |

**Actual inside predicted CI → extrapolation validated.** Sample approach was justified.

### Verdict: GO
48/50 eps ≥ 0.70, overall frame mean 0.7232, both NO-GO eps only 0.009 below cut-off. Proceeding to Step E.

---

## Step E — sim_v1 LeRobot v3 dataset

### Script
`sim_scripts/sim_to_lerobot.py`

### Strategy
- `meta/info.json`, `meta/tasks.parquet`, `data/chunk-000/file-000.parquet` → **copy from v6** (replay of same trajectory → state/action/timestamps identical)
- `videos/observation.images.top/chunk-000/file-000.mp4` → **re-encode** from sim PNGs via **av1_nvenc** (RTX 4090 Ada HW encode)
- `meta/episodes/chunk-000/file-000.parquet` → copy + overwrite `stats/observation.images.top/*` per-ep
- `meta/stats.json` → copy + overwrite `observation.images.top` (aggregate of 50 per-ep stats)

### Execution (2026-04-24 18:26-18:27, 72s total)

| Phase | Duration | Result |
|---|---|---|
| Copy static | <1s | 3 files copied |
| MP4 encode (av1_nvenc, 5 Mbit, yuv420p) | **7.0s @ 1044 fps** | 6942 frames, 90.2 MB |
| Per-ep image stats (50 × ~100 sample frames) | 50.1s | 50 dicts with `mean/std/min/max/q01..q99/count` |
| Episodes parquet write | <1s | 101599 bytes |
| stats.json write | <1s | 9829 bytes |
| LeRobotDataset validation | <2s | shape/keys match v6 |

### Output tree
```
sim_v1/
├── data/chunk-000/file-000.parquet             359 KB  (copy of v6)
├── meta/
│   ├── episodes/chunk-000/file-000.parquet     102 KB  (v6 + sim image stats)
│   ├── info.json                                2.6 KB (copy)
│   ├── stats.json                               9.8 KB (v6 + sim image aggregate)
│   └── tasks.parquet                            2.2 KB (copy)
└── videos/observation.images.top/chunk-000/file-000.mp4   90.2 MB (av1 yuv420p)
```
Total: **87 MB** (vs v6 75 MB; sim MP4 slightly larger due to uniform-dome-light bitrate allocation).

### Stats sanity (aggregate image mean/std)

| | v6 (real) | sim_v1 |
|---|---:|---:|
| mean R | 0.445 | 0.755 |
| mean G | 0.437 | 0.754 |
| mean B | 0.441 | 0.755 |
| std R | 0.0058 | 0.0012 |
| max | 1.00 | 0.953 |
| min | 0.00 | 0.00 |

**Sim is ~70% brighter than real** (dome-light white void vs real black couch/wall/dark table). **No saturation in sim** (max 0.953). Consistent with SigLIP gap sources observed in Step 5c.

**Note**: per-ep image std=0 in both v6 and sim_v1 (same `estimate_num_samples` behavior in LeRobot's running quantile stats on ≤200-frame episodes). Aggregate std is non-zero. Mirrored from v6 faithfully.

### LeRobotDataset validation

```
N eps: 50, N frames: 6942
features: ['observation.images.top', 'observation.state', 'action', ...]

ds[0]:     ep=0,  frame=0,   img_mean=0.7481, img_max=0.9608
ds[103]:   ep=1,  frame=0,   img_mean=0.7481, img_max=0.9608
ds[3000]:  ep=22, frame=8,   img_mean=0.7503, img_max=0.9608
ds[6941]:  ep=49, frame=142, img_mean=0.7512, img_max=0.9608
```

- Image tensor: (3, 720, 1280) float32, range [0.004, 0.961] ✓
- observation.state: (6,) float32 ✓ (identical to v6 — same replay)
- action: (6,) float32 ✓
- Episode boundaries correct ✓
- Lengths preserved (ep0=103, ep11=179, ep45=146) ✓

### Key decision

**sim_v1 is a drop-in replacement for v6 at the state/action level** (identical trajectories, only images differ). This enables **three-condition training with minimal dataset-side complexity**:

- **A (real-only)**: `--dataset.root=lerobot_dataset_v6`
- **B (sim-only)**: `--dataset.root=sim_v1`
- **C (real+sim)**: requires MultiLeRobotDataset or parquet concatenation (Step F design)

### Files produced
- `sim_v1/` (entire dataset)
- `sim_scripts/sim_to_lerobot.py` (rerunnable)
- `logs/stepE_sim_v1.log` / `.err` (execution trace)

---

## What's next (NOT started this session)

### Step F — 3-condition training
Design choices pending user approval:
1. **Condition C (real+sim) joining strategy**:
   - Option 1: LeRobot MultiLeRobotDataset (two repo_ids, dataset_mixing probabilities)
   - Option 2: Concatenate parquets + MP4s into a single `real_plus_sim/` dataset (100 eps, 13884 frames). Requires re-aggregating stats and re-indexing `index`/`episode_index`.
   - Option 3: Curriculum — pretrain on sim, fine-tune on real.
2. **Step count**: CLAUDE.md and MEMORY suggest 20K per condition. RunPod ~16.5h total, ~$5.6.
3. **Eval protocol**: 3-zone × 5 deploy trials per checkpoint (matches 2026-04-14 Stage 1 baseline evaluation).

### Remaining known gaps (Step 5c → Step 9 optional hardening)
- Table material override (dark gray → white): +5 min, likely +0.03 SigLIP → mean ~0.75
- HDRI dome light: +10 min, likely +0.02-0.04 SigLIP
- Robot link colors (gray → black): lower priority

Not blocking Step F; only needed if Condition B (sim-only) training plateaus early or Condition C shows poor real-transfer.
