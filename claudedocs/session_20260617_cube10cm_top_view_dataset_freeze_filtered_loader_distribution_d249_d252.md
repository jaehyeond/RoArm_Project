# Session 2026-06-17 - Dataset freeze + filtered loader + distribution D249-D252

## Scope

- Active branch: professor 10cm / 0.72kg cube top-view visual trajectory dataset.
- User asked to verify documentation and proceed step-by-step from the recommended
  order:
  1. dataset freeze,
  2. filtered LeRobot view,
  3. filtered dataloader smoke,
  4. train/eval distribution check.
- This session read existing D246-D248 artifacts and wrote small reproducibility
  manifests/checks only.
- No render, training, deletion, move, archive, PPO, L2, Large PPO,
  VLA/SmolVLA fine-tuning, action-teacher, RoArm deployment, RunPod runtime,
  B200/SSH/pull, `.ssh` copy, or Track A work was run.

## Documentation Recheck

Before new artifacts:

- `START_HERE.md` had the correct D248 current result near the current-truth
  section.
- `claudedocs/DECISIONS.md` had D248.
- `claudedocs/EXPERIMENT_LEDGER.md` had the D248 row.
- `claudedocs/session_20260617_cube10cm_top_view_label_package_d248.md` existed.

Issue found:

- The lower `START_HERE.md` `Active Direction` / `Must Read First` section still
  contained stale D234-era text about a 100 episode render and old must-read
  order. This session updated those sections to D252 current truth.

## Step 1 - Dataset Freeze

Added:

```text
sim_scripts/cube10cm_top_view_freeze_dataset.py
```

Run:

```bash
python3 sim_scripts/cube10cm_top_view_freeze_dataset.py
```

Output:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/dataset_freeze_d249
```

Key result:

- Freeze id: `cube10cm_top_view_0_999_v0_1_d249`
- `dataset_freeze_manifest_d249.json`: status `PASS`
- `dataset_card_d249.md`: plain-language dataset card
- `sha256_manifest_d249.tsv`: `24` primary files, total `1089314018` bytes
- LeRobot frozen root:
  `cube10cm_top_view_visual_0_999_d242/lerobot_dataset_av1_d247`
- Split package:
  `cube10cm_top_view_visual_0_999_d242/label_package_d248`

Important caveat:

- Raw PNG files remain preserved but were not individually SHA256 hashed.
- Reason: primary frozen artifact is LeRobot MP4+parquet plus split manifests;
  raw PNGs are large source/debug frames and still occupy the dominant disk cost.

## Step 2 - Filtered LeRobot Views

Added:

```text
sim_scripts/cube10cm_top_view_build_filtered_views.py
```

Run:

```bash
python3 sim_scripts/cube10cm_top_view_build_filtered_views.py
```

Output:

```text
cube10cm_top_view_visual_0_999_d242/filtered_views_d250
```

Definitions and counts:

- `train_clean_positive`: 학습용 정상 성공 예시.
  - `737` episodes
  - `143715` frames
  - 뜻: 카메라 검증을 통과했고, 접촉과 큐브 반응이 있으며, 과하게 밀지 않은 예시.
- `eval_clean_holdout`: 평가용 정상 보류 예시.
  - `82` episodes
  - `15990` frames
  - 뜻: 학습 가능한 정상 성공 데이터 중 일부를 일부러 빼둔 시험 문제.
- `eval_overshoot_diagnostic`: 과하게 민 케이스 진단용 평가 데이터.
  - `167` episodes
  - `32565` frames
  - 뜻: 접촉과 반응은 있지만 큐브를 과하게 민 케이스. 기본 positive train에는 넣지 않는다.
- `quarantine_camera_fail`: 카메라 기준 실패 격리 데이터.
  - `14` episodes
  - `2730` frames
  - 뜻: 카메라 투영/coverage 기준 실패로 train/eval에서 제외할 데이터.

Coverage check:

- All four views together cover exactly `195000` frame indices.

## Step 3 - Filtered Dataloader Smoke

Added:

```text
sim_scripts/cube10cm_top_view_filtered_dataloader_smoke.py
```

Run:

```bash
HF_HOME=/tmp/roarm_hf_cache HF_DATASETS_CACHE=/tmp/roarm_hf_datasets_cache conda run -n lerobot --no-capture-output python -u sim_scripts/cube10cm_top_view_filtered_dataloader_smoke.py
```

Output:

```text
filtered_views_d250/dataloader_smoke_d251/filtered_dataloader_smoke_summary.json
```

Result:

- status `PASS`
- dataset root:
  `cube10cm_top_view_visual_0_999_d242/lerobot_dataset_av1_d247`
- repo id: `roarm_cube10cm_top_view_0_999_d247`
- video backend: `pyav`
- total frames: `195000`
- total episodes: `1000`
- All four splits sampled through LeRobot.
- Decoded image shape: `[3,720,1280]`
- state/action shape: `[6]` / `[6]`

Runtime caveat:

- The command emitted urllib/requests dependency warnings and a torchvision video
  deprecation warning. These did not block loading or decoding.
- Local torchcodec remains not trusted; use explicit `video_backend=pyav`.

## Step 4 - Split Distribution Check

Added:

```text
sim_scripts/cube10cm_top_view_split_distribution_check.py
```

Run:

```bash
python3 sim_scripts/cube10cm_top_view_split_distribution_check.py
```

Output:

```text
split_distribution_d252
```

Result:

- status `PASS`
- 학습용 정상 성공 예시는 sampled workspace x/y range를 포함한다:
  - x `0.09000000357627869..0.38999998569488525`
  - y `-0.10000000149011612..0.15000000596046448`
- 과하게 민 케이스 진단용 평가 데이터 is concentrated at higher y:
  - mean y `0.08234862930247348`
  - boundary y `>=0.12`: `79/167`
- 학습용 정상 성공 예시:
  - mean y `-0.008028418002612232`
  - boundary y `>=0.12`: `59/737`
- 카메라 기준 실패 격리 데이터:
  - x `0.14000000059604645..0.16500000655651093`
  - all `14` in mid-y band
  - episode `721` remains the strongest camera coverage warning from D248.

Interpretation:

- Train data is not confined to a tiny local region; it spans the sampled
  workspace.
- Overshoot is not random noise only; it is position-dependent, especially high-y
  and boundary-y. This should be reported separately in any later evaluation.
- Camera-fail quarantine should remain excluded before training.

## Decision

`D249_D252_DATASET_FREEZE_FILTERED_VIEW_DATALOADER_DISTRIBUTION_PASS`

This proves dataset/loader readiness for a future training preflight plan. It
does not prove model performance.

Default next data usage:

- Train on `train_clean_positive` only by default.
- Evaluate normal success on `eval_clean_holdout`.
- Report overshoot behavior separately on `eval_overshoot_diagnostic`.
- Exclude `quarantine_camera_fail`.

Still blocked until explicit approval:

- Actual SmolVLA/VLA fine-tuning.
- PPO/L2/Large PPO.
- Action-teacher work.
- RoArm deployment.
- RunPod runtime.
- Raw PNG cleanup, archive, move, or deletion.
- Additional 1000/10000 generation.
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy.
- Track A work.

## Sources

- `START_HERE.md`
- `claudedocs/DECISIONS.md`
- `claudedocs/EXPERIMENT_LEDGER.md`
- `sim_scripts/cube10cm_top_view_freeze_dataset.py`
- `sim_scripts/cube10cm_top_view_build_filtered_views.py`
- `sim_scripts/cube10cm_top_view_filtered_dataloader_smoke.py`
- `sim_scripts/cube10cm_top_view_split_distribution_check.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/dataset_freeze_d249/dataset_freeze_manifest_d249.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/dataset_freeze_d249/dataset_card_d249.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/filtered_views_d250/filtered_views_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/filtered_views_d250/dataloader_smoke_d251/filtered_dataloader_smoke_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/split_distribution_d252/split_distribution_summary.json`
