# Cube10cm Top-View Method Pipeline D254

Date: 2026-06-17 KST

## Purpose

This document reframes the current professor branch correctly.

The goal is not just to show example data to the professor. The goal is to
show that the proposed method pipeline is coherent:

1. define a physically reproducible top-view camera contract;
2. generate 10cm cube tap/push visual trajectories in Isaac Lab;
3. validate labels after render from measured contact/reaction/overshoot/camera
   metrics;
4. store the result in a training-compatible LeRobot format;
5. split the corpus into train/eval/quarantine by numeric post-render labels;
6. only then connect the dataset to a model training/evaluation path.

## What Professor Method Means Here

"Pipeline" means a repeatable research method, not a one-off dataset preview.

In Korean:

- `pipeline`: 파이프라인. 연구 절차를 단계별로 고정한 흐름이다. 여기서는
  카메라 계약, Isaac Lab 생성, 라벨 검증, LeRobot 저장, 학습/평가 연결까지의
  전체 방법을 뜻한다.
- `camera contract`: 카메라 계약. 실제 Azure Kinect를 다시 설치해도 같은
  시야를 재현할 수 있도록 높이, roll/pitch/yaw, flip/crop, workspace coverage,
  fps, self-occlusion 기준을 문서화한 규칙이다.
- `post-render label validation`: 렌더 후 라벨 검증. manifest에 적은 의도
  라벨을 그대로 믿지 않고, 실제 렌더된 trajectory에서 접촉, 반응, 과밀기,
  카메라 투영/가시성 수치를 계산해서 최종 라벨을 붙이는 단계다.
- `LeRobot`: HuggingFace LeRobot 데이터셋 포맷과 로더. 여기서는
  `mp4 + parquet` 저장 구조와 official training data path를 뜻한다.
- `SmolVLA training smoke`: 짧은 모델 학습 배관 테스트. 성능 주장용 학습이
  아니라, 데이터가 실제 `lerobot-train`으로 들어가 loss/backprop까지 깨지지
  않는지 보는 최소 runtime test다.

## Current Evidence Chain

### Step 1 - Camera Contract Direction

D232 established the professor branch as a camera-calibrated visual trajectory
dataset path, not PPO promotion.

Key decisions:

- raw visual target is Azure-Kinect-compatible `1280x720`;
- `224x224` is only model preprocessing;
- old `sim_scripts/kinect_calib.yaml` intrinsics may inform camera intrinsics,
  but old hand-eye extrinsics are invalid for the new top-view setup;
- primary dataset storage should be LeRobot-style video+parquet, not raw PNG at
  scale.

Sources:

- `claudedocs/session_20260612_camera_contract_visual_dataset_disk_audit_d232.md`
- `claudedocs/DECISIONS.md` D232

### Step 2 - Isaac Lab Generation

D246 is the actual Isaac Lab data generation stage.

What it produced:

- 1000 episodes;
- 195000 frames;
- raw top-view RGB at `1280x720`;
- target `30fps`;
- raw PNG bytes `51386208295`;
- elapsed render time about `7.88h`;
- no model training.

Why this matters:

- This is the "sim generates visual trajectory data" part of the professor
  method.
- It is not SmolVLA training.
- It is not PPO.

Sources:

- `claudedocs/session_20260615_cube10cm_top_view_0_999_render_labels_d246.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/render_summary.json`

### Step 3 - Post-Render Label Validation

D246 also converted intended sampling buckets into measured labels.

Current label result:

- `clean_useful_tap=819`;
- `contact_reaction_with_overshoot=167`;
- `camera_quality_fail=14`.

Important principle:

- manifest labels are intended sampling buckets;
- final train/eval/quarantine labels come only from post-render numeric
  validation.

Sources:

- `claudedocs/session_20260615_cube10cm_top_view_0_999_render_labels_d246.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/postrender_label_validation_d246/label_validation_summary.json`

### Step 4 - LeRobot Conversion

D247 converted the rendered corpus to LeRobot v3 AV1+parquet.

What it proved:

- LeRobot root exists at
  `cube10cm_top_view_visual_0_999_d242/lerobot_dataset_av1_d247`;
- `195000` frames / `1000` episodes load with `video_backend=pyav`;
- codec is `av1`, `yuv420p`, `30fps`;
- final LeRobot root is about `540M`;
- arbitrary PNG extraction from MP4 was verified.

Why this matters:

- This is the "industry-standard storage/training-compatible format" part of
  the professor method.
- It also proves PNG can be extracted on demand, without storing all frames as
  PNG at scale.

Sources:

- `claudedocs/session_20260616_cube10cm_top_view_0_999_lerobot_metadata_d247.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/lerobot_validation_summary.json`

### Step 5 - Train/Eval/Quarantine Packaging

D248-D252 turned labels into usable split lists and checked loader/distribution.

Current split definitions:

- `train_clean_positive`: 학습용 정상 성공 예시.
  - camera-pass and clean useful tap only.
  - `737` episodes / `143715` frames.
- `eval_clean_holdout`: 평가용 정상 보류 예시.
  - clean useful tap held out from training.
  - `82` episodes / `15990` frames.
- `eval_overshoot_diagnostic`: 과하게 민 케이스 진단용 평가 데이터.
  - camera-pass but over-pushed trajectories.
  - `167` episodes / `32565` frames.
- `quarantine_camera_fail`: 카메라 기준 실패 격리 데이터.
  - excluded by default.
  - `14` episodes / `2730` frames.

Why this matters:

- This is the "training-ready curation" part of the method.
- Overshoot data is not positive training data.
- Camera-fail data is not hidden; it is isolated and auditable.

Sources:

- `claudedocs/session_20260617_cube10cm_top_view_label_package_d248.md`
- `claudedocs/session_20260617_cube10cm_top_view_dataset_freeze_filtered_loader_distribution_d249_d252.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/label_package_d248/split_package_summary.json`

### Step 6 - Training-Input Preflight, Not Training

D253 checked whether the selected train split can enter the official LeRobot
training data path.

What it proved:

- official LeRobot `DatasetConfig.episodes` can select
  `train_clean_positive`;
- selected data is `737` episodes / `143715` frames;
- first DataLoader batch shape is image `[4,3,720,1280]`, state `[4,6]`,
  action `[4,6]`;
- local AV1 decode should use `video_backend=pyav`.

What it did not prove:

- it did not run SmolVLA fine-tuning;
- it did not produce a model checkpoint;
- it did not evaluate model performance;
- it did not use Isaac Lab for training.

Sources:

- `claudedocs/session_20260617_cube10cm_top_view_training_preflight_d253.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/training_preflight_d253/training_preflight_summary_d253.json`

## Correct Interpretation of SmolVLA Smoke

SmolVLA smoke is not the professor method itself.

It is an optional next gate after the dataset pipeline is framed:

- It checks the training loop can consume the dataset.
- It checks loss/backprop does not crash.
- It does not prove that the policy is useful.
- It does not replace offline evaluation on `eval_clean_holdout` and
  `eval_overshoot_diagnostic`.
- It is not Isaac Lab training.

Therefore the correct order is:

1. lock the method pipeline and deliverable framing;
2. decide whether a 50-step SmolVLA smoke is needed as a training-loop sanity
   check;
3. if smoke is approved and passes, design offline evaluation over held-out
   splits;
4. only after that consider a longer model run.

## What To Tell Professor

Short framing:

> 교수님 피드백대로, 10cm cube tap/push를 top-view camera contract 기반으로
> Isaac Lab에서 생성하고, 렌더 후 contact/reaction/overshoot/camera 기준으로
> 라벨을 확정한 뒤, LeRobot 표준 mp4+parquet 포맷으로 학습 가능한 데이터셋을
> 만들었습니다. 지금 산출물은 단순 이미지 예시가 아니라, sim generation부터
> label validation, train/eval/quarantine split, model training input까지 이어지는
> pipeline proof입니다.

More explicit:

> 아직 모델 성능을 주장하는 단계는 아닙니다. 현재까지 증명한 것은 데이터 생성
> 및 학습 입력 파이프라인입니다. 다음 단계는 50-step training smoke로 학습
> loop 연결을 확인하거나, 먼저 held-out evaluation 설계를 고정하는 것입니다.

## Current Recommendation

Do not present D253 as "we trained SmolVLA."

Present the current state as:

- method pipeline built through training-input preflight;
- no model-performance claim yet;
- next runtime, if approved, is a minimal 50-step SmolVLA smoke only to verify
  training-loop connectivity;
- evaluation split usage still needs a separate offline evaluation script.

## Blocked Until Explicit Approval

- 50-step SmolVLA training smoke.
- 20k SmolVLA candidate training.
- Any RunPod/H100 job.
- Any extra Isaac Lab render or 1000/10000 expansion.
- Any PPO/L2/Large PPO/action-teacher/RoArm deployment work.
- Any deletion/archive/move/cleanup.
- Any B200/SSH/pull/.ssh copy.
