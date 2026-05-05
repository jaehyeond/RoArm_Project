# 2026-05-04 evening — Phase ST-A v3 COMPLETE (Sub-A6 + Sub-A7 + Sub-A8 PASS)

## TL;DR
Phase ST-A v3 (edge-stand 47mm tall sponge # tower stacking) build pipeline END-TO-END PASS:
- **Sub-A6**: 50 demo render → `sim_renders_v5/` (7300 PNG, 2.2 GB), 22m 23s
- **Sub-A7**: render → `lerobot_dataset_stacking_v3/` (50 ep × 146 fr, 41 MB), 5m 44s
- **Sub-A8**: merge v6 + stacking_v3 → `lerobot_dataset_v6_stacking_v3/` (100 ep × 14242 fr, 115 MB), 0.4s mp4 stream-copy

ST-B2 v3 (B200 finetune 20K steps, ~84 min) 진입 준비 완료.

---

## 1. Sub-A6 — 50 demo render (background `bftatuis5`, exit 0)

### Command
```bash
conda run --no-capture-output -n isaaclab python sim_scripts/render_stacking_demos_v3.py --all
```

### Result
| Metric | Value |
|---|---|
| Episodes | 50/50 |
| Frames per ep | 146 (constant) |
| Total frames | 7300 |
| Total elapsed | **1328.6 s = 22 m 8 s** (Sub-A6 runner reported `[1343s] Simulation App Shutting Down` ≈ 22m 23s wall-clock) |
| ms/frame avg | **182.0 ms** |
| Per-ep | 26.0–26.3 s (extremely consistent) |
| Disk | **2.2 GB** at `sim_renders_v5/` |
| Exit | 0 |

### Validation
- All 50 eps directories created: `sim_renders_v5/episode_000/` ~ `episode_049/`
- All 50 × 146 = 7300 PNGs present (per-ep `frame_*.png` count verified)
- `sim_renders_v5/render_summary.json` written with per-ep timings
- Pre-existing `stacking_initial_seed0_v3.png` from Sub-A4 preserved (not overwritten)

### Per-ep consistency
26.0–26.3 s per episode → no Isaac restart penalty per ep, single sim_app_launcher amortized over all 50 eps. Slightly slower than 5/01 v2 22min total (181 ms/frame) by ~1 ms/frame — within noise.

---

## 2. Sub-A7 — sim_to_lerobot v3 (background `bssh7dry0`, exit 0)

### Command
```bash
conda run --no-capture-output -n roarm python sim_scripts/sim_to_lerobot_stacking.py
```

### Path edits applied (v2 → v3)
[sim_scripts/sim_to_lerobot_stacking.py](../sim_scripts/sim_to_lerobot_stacking.py) lines 30-33:
```python
DEMOS_DIR = REPO / "sim_demos_v3"      # was sim_demos_v2
RENDERS_DIR = REPO / "sim_renders_v5"  # was sim_renders_v4
OUT_DIR = REPO / "lerobot_dataset_stacking_v3"  # was _v2
REPO_ID = "roarm_m3_stacking_sim_v3"   # was _v2
```
+ docstring header v2 → v3.

### Result
| Metric | Value |
|---|---|
| Build time | **344.4 s = 5 m 44 s** |
| Total eps | 50 |
| Total frames | 7300 |
| Per-ep build | 6.9 s (very consistent across all 50 eps) |
| Disk | **41 MB** at `lerobot_dataset_stacking_v3/` |
| Exit | 0 |

### Dataset validation (post-build, ds[0])
- `observation.images.top`: shape (3, 720, 1280) float32 range [0.0000, 0.9765] ✓
- `observation.state`: shape (6,) float32 first=`[0.0, 0.0, 90.0, 0.0, 0.0, 5.0]` (HOME, gripper closed=5° per HARD RULE #18 user-defined convention) ✓
- `action`: shape (6,) float32

### Convention enforcement
- 1 task: `"Stack four pink sponges into a # pattern"` (task_index=0 in standalone v3 dataset)
- AV1 video stream-encoded
- L-F gap: NONE — `state[t] = action[t] = trajectory[t]` (procedural sim demo)
- FPS=30, robot_type=`roarm_m3`

---

## 3. Sub-A8 — merge_v6_stacking v3 (background `bjdkxb55y`, exit 0)

### Command
```bash
conda run --no-capture-output -n roarm python sim_scripts/merge_v6_stacking.py
```

### Path edits applied (v2 → v3)
[sim_scripts/merge_v6_stacking.py](../sim_scripts/merge_v6_stacking.py) lines 30-33:
```python
V6_ROOT = REPO / "lerobot_dataset_v6"               # unchanged
STACKING_ROOT = REPO / "lerobot_dataset_stacking_v3"  # was _v2
OUT_ROOT = REPO / "lerobot_dataset_v6_stacking_v3"    # was _v2
AGGR_REPO_ID = "roarm_m3_v6_stacking_v3"              # was _v2
```
+ docstring header v2 → v3, repo_ids label v2 → v3, log message v2 → v3.

### Result
| Metric | Value |
|---|---|
| Aggregate time | **0.4 s** (mp4 stream-copy concat, no re-encoding) |
| Total eps | 100 (= 50 v6 + 50 stacking_v3) |
| Total frames | 14242 (= 6942 v6 + 7300 stacking_v3) |
| Total tasks | 2 |
| Disk | **115 MB** at `lerobot_dataset_v6_stacking_v3/` |
| Exit | 0 |

### Dataset spot-checks (ALL ASSERTIONS PASS)
| ds index | episode | task_index | Task string | state |
|---|---|---|---|---|
| `ds[0]` | 0 | 0 | `"Pick up the sponge\n"` | `[0.35, 4.48, 91.32, 0.0, 0.26, 1.49]` (v6 ep0 first frame, near HOME) |
| `ds[6941]` | 49 | 0 | `"Pick up the sponge\n"` | (last v6 frame) |
| `ds[6942]` | 50 | 1 | `"Stack four pink sponges into a # pattern"` | `[0.0, 0.0, 90.0, 0.0, 0.0, 5.0]` (stacking_v3 ep0 HOME closed) |
| `ds[14241]` (last) | 99 | 1 | (stacking last) | — |

### Final assert
```
expected = (100, 14242, 2)  # eps, frames, tasks
actual   = (100, 14242, 2)
ALL ASSERTIONS PASS
```

### task_index remap convention (preserved from v6 design)
- 0 = `"Pick up the sponge\n"` (v6 real pick, single Kinect, L-F teleop)
- 1 = `"Stack four pink sponges into a # pattern"` (sim # tower edge-stand stacking, procedural IK)
- Aggregator = native `lerobot.datasets.aggregate.aggregate_datasets()` → mp4 stream-copy concat (no re-encode), parallel-variance stats, name-based task remap.

---

## 4. HARD RULES applied (no violations)
| Rule | Application |
|---|---|
| #11 (no /half-clone) | Stop hook 111% earlier → declined. Continued via sequential background tasks. |
| #13 (cgxr@Lenovo dual-PC) | All Sub-A6/A7/A8 ran on Lenovo 4090 (sim render + dataset build). B200 untouched. |
| #16 (4090 train_config source-of-truth) | Reserved for ST-B2 finetune, not yet applied. |
| #17 (sim render = 4090) | Sub-A6 ran in `isaaclab` env on local RTX 4090 Laptop ✓. |
| #18 (사용자 정정 우선) | Sponge orientation = edge-stand (47mm tall) — preserved across all v3 outputs. |
| #19 (edge-stand 47mm tall) | Generate v3 SPONGE_HEIGHT_EDGE=0.047, render TCP_TO_SPONGE_CENTER_Z=−0.02162 — all consistent. |
| #20 (# tower geometry L1 c2c=87 / L2 c2c=67) | DY_L1=0.0435 / DX_L2=0.0335 — preserved. |

### TodoWrite reminder
거부 (3회): 사용자 prompt에 명시적 step list 있음, 메모리 + 로그가 충분, 추가 트래킹 불필요.

---

## 5. 잠재 이슈 (ST-B2/ST-C 진입 전 검토)

| 이슈 | 영향 | 검증 시점 |
|---|---|---|
| TCP z 최댓값 +343.7mm (HOME bridge 17 frames > +155mm 안전 임계) | Real deploy 시 JOINT_SPEED_CAPS 보호 → 실측 필요 | ST-C real deploy 첫 ckpt |
| wrist_p clamp +75° vs v6 mean +68.8° (+6° tighter) | IK fail 0 (Sub-A3) PASS, but +6° marginal OOD | ST-B2 loss curve + ST-C grasp 정확도 |
| L2 descent 직선 (+150 → +80) held bottom +47mm = L1 top grazing | L2 place 시 sponge collision 가능 | ST-C L2 place attempt 시 |
| 5K가 v3 새 task에 saturate일지 미확정 | 4 ckpt (5K/10K/15K/20K) 비교 필수 | ST-C 4-ckpt deploy 비교 |
| Normalizer refit (v6 6942 + stacking_v3 7300 = 14242 stats) | v3 분포가 v2와 다름 (edge-stand vs lying-flat 자세 변경) | ST-B2 step 100 loss / step 1K loss 모니터 |

---

## 6. ST-B2 v3 진입 사양 (B200, 별도 SSH 세션)

### 사용자 결정 적용
- steps = **20,000** (5K/10K/15K/20K 4 ckpt → save_freq=2500)
- peak_lr = 5e-5
- warmup = 500
- decay_steps = 20000 (cosine 1주기 정확)
- decay_lr = 1e-6
- batch_size = 64
- seed = 1000
- video_backend = torchcodec

### Base ckpt + dataset
- Base: `outputs/smolvla_v6_b200/checkpoints/last/pretrained_model` (v6 50K, B200 reproducibility 검증 완료)
- Dataset: `lerobot_dataset_v6_stacking_v3/` (115 MB, rsync ETA <1min)

### B200 ETA
- Production: ~84 min for 20K (v2 10K = 42m → 20K = 84m linear)
- 4 ckpt rsync back to 4090: ~4 min total

### HARD RULE 강제
- #14 fail-fast guard (`set -e; source env.sh; [[ -z "$ROARM_B200_ROOT" ]] && exit 1; [[ "$(whoami)" != "sogang_jhki" ]] && exit 1`)
- #15 PyTorch nightly cu128 → lerobot install 후 강제 upgrade → arch_list sm_100 검증
- #16 `train_config.json` source-of-truth: `observation.images.top` 1개 + empty_cameras=0
- HARD RULE #13 dual-PC env: Lenovo `JHPark/roarm_b200/` + GPU 0 UUID c553ca20

---

## 7. 다음 세션 진입 (Continuation Prompt)

```
ST-B2 v3 B200 finetune (20K steps, ~84min) 진입.

준비물:
- Local: lerobot_dataset_v6_stacking_v3/ (115 MB) — rsync to B200
- Base: outputs/smolvla_v6_b200/checkpoints/last/pretrained_model — already on B200 v6 path
- HARD RULE #13/#14/#15/#16 강제

명령 (fail-fast guard + nightly cu128 upgrade 후):
lerobot-train \
  --policy.pretrained_path=outputs/smolvla_v6_b200/checkpoints/last/pretrained_model \
  --dataset.repo_id=roarm_m3_v6_stacking_v3 \
  --dataset.root=lerobot_dataset_v6_stacking_v3 \
  --batch_size=64 \
  --steps=20000 \
  --save_freq=2500 \
  --seed=1000 \
  --optimizer.lr=5e-5 \
  --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps=20000 \
  --scheduler.peak_lr=5e-5 \
  --scheduler.decay_lr=1e-6 \
  --output_dir=outputs/smolvla_v6_stacking_v3_b200 \
  --video_backend=torchcodec

(loss curve 모니터: step 100, 500, 1K, 2.5K, 5K, 10K, 15K, 20K. v2와 비교 — edge-stand 자세 새 분포로 step 100 loss는 v2 0.416 대비 변동 가능. step 1K 0.020 이하 도달 시 정상 fast adapt.)

검증:
- 5K/10K/15K/20K 4 ckpt 모두 rsync back to Lenovo
- weight diff vs base (vision encoder bit-exact 378/500 패턴 재확인 — 4/28 evening v6 + 5/03 v2 패턴 일치 여부)

다음: ST-C v3 deploy (4090, 4 ckpt 비교)
```

---

## 8. 파일 변경 요약

### Modified
- [sim_scripts/sim_to_lerobot_stacking.py](../sim_scripts/sim_to_lerobot_stacking.py) — v2 → v3 paths (4 const + docstring)
- [sim_scripts/merge_v6_stacking.py](../sim_scripts/merge_v6_stacking.py) — v2 → v3 paths (3 const + docstring + 2 log msgs + repo_ids label)

### Created
- [sim_renders_v5/](../sim_renders_v5/) — 50 ep × 146 PNGs + render_summary.json (2.2 GB)
- [lerobot_dataset_stacking_v3/](../lerobot_dataset_stacking_v3/) — sim_v3 standalone (41 MB, 50 ep × 7300 fr, 1 task)
- [lerobot_dataset_v6_stacking_v3/](../lerobot_dataset_v6_stacking_v3/) — co-train (115 MB, 100 ep × 14242 fr, 2 tasks)
- [logs/sub_a6_v3/](../logs/sub_a6_v3/) — render_all.log, sub_a7.log, sub_a8.log + .err

### Untouched (per user "Cleanup 안 함")
- `sim_renders_v5_dryrun/` (Sub-A5 dry run, 1 ep × 146 PNGs) — preserved
- `sim_demos_v2/`, `sim_renders_v4/`, `lerobot_dataset_stacking_v2/`, `lerobot_dataset_v6_stacking_v2/` — DEPRECATED (5/03 evening pivot) but disk preserved
- B200 `outputs/smolvla_v6_stacking_v2_b200/` 5K+10K ckpt — DEPRECATED but disk preserved
