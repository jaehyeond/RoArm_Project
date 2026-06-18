# Session 2026-06-17 - Cube10cm Top-View Training Preflight D253

## Scope

Active branch:

- Professor 10cm / 0.72kg cube top-view visual trajectory dataset branch.

This session continued after D249-D252. It did not run Track A, PPO/L2/Large
PPO, VLA/SmolVLA fine-tuning, action-teacher work, RoArm deployment, RunPod,
B200/SSH/pull, deletion, archive, move, render, or additional dataset
generation.

## Current-State Checks

- `git status --short --untracked-files=all --branch` was run first.
- `CLAUDE.md` confirms the Current-State Protocol:
  - `START_HERE.md` and `claudedocs/` are the current truth.
  - `HANDOFF.md` and `TASKS.md` are stale unless `START_HERE.md` points to
    them.
- `START_HERE.md` before this session said the valid next work was a training
  preflight plan/dry-run only, not actual fine-tuning.

## LeRobot Source Audit

The important question was whether our split can be used by official LeRobot
training without copying the dataset.

Result: yes, at the episode level.

Evidence from installed LeRobot source:

- `DatasetConfig` has `episodes: list[int] | None` and validates that the list
  is non-negative and duplicate-free.
- `make_dataset()` passes `cfg.dataset.episodes` and
  `cfg.dataset.video_backend` into `LeRobotDataset`.
- `LeRobotDataset` stores the requested `episodes` list and reports selected
  `num_frames` / `num_episodes`.
- `lerobot_train.py` passes `dataset.episodes` into `EpisodeAwareSampler` when
  the policy needs an episode-aware sampler.

Critical correction:

- A draft 50-step smoke command used `save_freq=0`.
- The official train loop computes `step % cfg.save_freq`, so `save_freq=0`
  could divide by zero even when `save_checkpoint=false`.
- D253 proposed smoke command therefore uses `save_freq=50`.

## D253 Script

Added:

- `sim_scripts/cube10cm_top_view_training_preflight.py`

It reads the D248-D252 artifacts, verifies split integrity, opens the approved
training split through `lerobot.datasets.factory.make_dataset`, samples decoded
frames, checks a small DataLoader batch, and writes proposed commands for a
future approved training run.

It does not train.

## Command Run

```bash
env HF_HOME=/tmp/roarm_hf_cache HF_DATASETS_CACHE=/tmp/roarm_hf_datasets_cache conda run -n lerobot --no-capture-output python -u sim_scripts/cube10cm_top_view_training_preflight.py --force
```

Non-blocking warnings:

- `requests` dependency warning from the base conda import path.
- torchvision video deprecation warning. This is expected for the current
  `pyav` AV1 decode path and did not fail D253.

## Output

Output root:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/training_preflight_d253`

Main files:

- `training_preflight_summary_d253.json`
- `training_preflight_brief_d253.md`
- `train_clean_positive_episode_ids_d253.txt`
- `eval_clean_holdout_episode_ids_d253.txt`
- `eval_overshoot_diagnostic_episode_ids_d253.txt`
- `quarantine_camera_fail_episode_ids_d253.txt`
- `proposed_smolvla_train_smoke_50_steps_d253.txt`
- `proposed_smolvla_train_candidate_20000_steps_d253.txt`

## Result

`training_preflight_summary_d253.json` status:

- `PASS`

Selected train split:

- `train_clean_positive`
- Korean definition: 학습용 정상 성공 예시.
- Meaning: camera-pass and clean useful tap episodes only.
- Episodes: `737`
- Frames: `143715`
- Selection mechanism: LeRobot `DatasetConfig.episodes` /
  `LeRobotDataset(..., episodes=...)`.
- Video backend: `pyav`.

Factory/DataLoader checks:

- `factory_path`: `lerobot.datasets.factory.make_dataset`
- selected episodes: `737`
- selected frames: `143715`
- first DataLoader batch:
  - image shape `[4,3,720,1280]`
  - state shape `[4,6]`
  - action shape `[4,6]`

Sampled selected training frames:

- dataset index `0`: episode `1`, frame `0`
- dataset index `71857`: episode `448`, frame `97`
- dataset index `143714`: episode `999`, frame `194`

## Korean Term Definitions

- `train_clean_positive`: 학습용 정상 성공 예시.
  - 카메라 기준을 통과했고, 로봇이 큐브에 접촉/반응을 만들었고, 너무 많이
    밀지 않은 정상 성공 데이터다.
- `eval_clean_holdout`: 평가용 정상 보류 예시.
  - 정상 성공 데이터 중 일부를 학습에서 빼고 나중에 모델 검증에 쓰는 목록이다.
- `eval_overshoot_diagnostic`: 과하게 민 케이스 진단용 평가 데이터.
  - 카메라 기준은 통과했지만 큐브를 너무 많이 밀어버린 데이터다. 좋은
    정답으로 학습시키지 않고, 모델이 이런 실패 경향을 보이는지 따로 본다.
- `quarantine_camera_fail`: 카메라 기준 실패 격리 데이터.
  - 카메라 projection/reprojection/coverage 기준을 통과하지 못한 데이터다.
    학습과 평가에서 기본 제외한다.

## Critical Limitation

`lerobot-train` takes one dataset input. Its `eval_freq` means environment
rollout evaluation, not this dataset's held-out split.

Therefore:

- `train_clean_positive` can feed future training directly.
- `eval_clean_holdout` is not automatically used by `lerobot-train`.
- `eval_overshoot_diagnostic` is not automatically used by `lerobot-train`.
- A separate offline evaluation script is still required after an approved
  checkpoint exists.

## Verdict

`D253_TRAINING_PREFLIGHT_PASS_NO_TRAINING`

The dataset is ready to be selected by official LeRobot training code, but no
model has been trained and no model-performance claim exists.

## Next Blocked Items

Still blocked until explicit approval:

- 50-step SmolVLA training smoke.
- 20k-step SmolVLA candidate training.
- Offline held-out evaluation script after a checkpoint exists.
- RunPod/H100 runtime.
- PPO/L2/Large PPO/action-teacher/RoArm deployment.
- Deletion/archive/move/cleanup.
- Additional rendering or 1000/10000 expansion.
- B200/SSH/pull/.ssh copy.
- Track A work.
