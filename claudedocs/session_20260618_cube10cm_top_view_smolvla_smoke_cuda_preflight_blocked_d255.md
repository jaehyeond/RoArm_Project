# Session 2026-06-18 - Cube10cm Top-View SmolVLA Smoke CUDA Preflight Blocked D255

## Scope

Active branch:

- Professor 10cm / 0.72kg cube top-view visual trajectory dataset / camera-contract
  branch.

This session followed the D254 recommendation order and checked the next possible
runtime step: a minimal 50-step SmolVLA training-loop smoke.

This session did not run Isaac Sim render, generate data, delete/archive/move
files, start RunPod, connect to B200/SSH, pull code, run PPO/L2/Large PPO,
run action-teacher work, run RoArm deployment, or start actual model fine-tuning.

## Current-State Checks

- `git status --short --untracked-files=all --branch` was run first.
- `CLAUDE.md` says `START_HERE.md`, `claudedocs/DECISIONS.md`,
  `claudedocs/EXPERIMENT_LEDGER.md`, and referenced session logs are the current
  truth, while `HANDOFF.md` and `TASKS.md` are stale unless explicitly referenced.
- `START_HERE.md` D254 said:
  - the professor branch is method-pipeline-ready through training-input
    preflight;
  - SmolVLA smoke is optional training-loop connectivity verification;
  - actual SmolVLA/VLA fine-tuning, PPO, RunPod, deletion, additional rendering,
    and RoArm work remain blocked unless explicitly approved.
- `claudedocs/cube10cm_top_view_method_pipeline_d254.md` says SmolVLA smoke is
  not the professor method itself, not Isaac Lab training, and not a policy
  usefulness proof.

## Intended Next Step

The intended approved next step was:

- run the D253 proposed 50-step SmolVLA smoke;
- use only `train_clean_positive` episodes;
- use `dataset.video_backend=pyav`;
- avoid checkpoints with `save_checkpoint=false`;
- use `save_freq=50` because `save_freq=0` can divide by zero in the official
  training loop.

This is a connectivity smoke only. In Korean terms:

- `training-loop smoke` means "학습 코드가 데이터셋을 읽고 forward/backward/loss
  계산을 아주 짧게 통과하는지 보는 연결성 점검"이다.
- It does not mean "성능이 좋아졌다" or "교수님 방법이 완성됐다".

## Pre-Run Resource Check

Disk:

- Command: `df -h . /tmp`
- Result: project root and `/tmp` are on the same filesystem.
- Size/used/free: `590G / 529G / 31G`.
- Use: `95%`.

GPU host:

- Command:
  `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv`
- Result: `NVIDIA GeForce RTX 4090 Laptop GPU`.
- Memory: `16376 MiB` total, `2509 MiB` used, `13436 MiB` free.
- Utilization: `19%`.

Output directory collision:

- `outputs/smolvla_cube10cm_top_view_d253_smoke` did not exist.
- `outputs/smolvla_cube10cm_top_view_d253_candidate` did not exist.

Process check:

- No active Isaac/Kit/render/`lerobot-train`/torchrun-style process was found
  for this branch before starting a smoke.

## CUDA Environment Audit

The smoke was not started because the local training environment failed CUDA
preflight.

### `lerobot` env

Observed:

- `torch`: `2.10.0+cu128`
- `torch.version.cuda`: `12.8`
- `torch.cuda.is_available()`: `False`
- `torch.cuda.device_count()`: `0`
- Warning: `Can't initialize NVML`
- `CUDA_VISIBLE_DEVICES`: unset
- `PYTORCH_CUDA_ALLOC_CONF`: unset

Important interpretation:

- This environment has the `lerobot-train` CLI and is the environment used by
  D253 training-input preflight.
- But it cannot see CUDA.
- Running the 50-step smoke here would either fail or silently become an
  invalid CPU/environment test.

### Quick override attempts in `lerobot` env

All remained CUDA false:

- `PYTORCH_NVML_BASED_CUDA_CHECK=0`: available `False`, count `0`.
- `PYTORCH_NVML_BASED_CUDA_CHECK=1`: available `False`, count `0`.
- `CUDA_VISIBLE_DEVICES=0`: available `False`, count `0`.
- `CUDA_MODULE_LOADING=LAZY`: available `False`, count `0`.

### `isaaclab` env

Observed:

- `torch`: `2.7.0+cu128`
- `torch.version.cuda`: `12.8`
- `torch.cuda.is_available()`: `True`
- `torch.cuda.device_count()`: `1`
- Device: `NVIDIA GeForce RTX 4090 Laptop GPU`

Important interpretation:

- Host GPU and driver are not globally broken.
- The CUDA failure is environment-specific.
- However `isaaclab` does not provide the `lerobot-train` CLI, so using it for
  this smoke would require dependency changes, not a simple runtime launch.

### Other envs

- `roarm`: has `lerobot-train`, but CUDA is also false with the same NVML-style
  failure.
- `openvla`: cannot import `torch`, so it is not a valid local SmolVLA smoke
  environment.

## Verdict

`D255_SMOLVLA_SMOKE_BLOCKED_LEROBOT_ENV_CUDA_FALSE_NO_TRAINING`.

The 50-step SmolVLA smoke was not run.

Critical reasoning:

- D254 explicitly frames SmolVLA smoke as optional connectivity verification.
- A valid connectivity smoke must test the actual intended path:
  LeRobot train + `train_clean_positive` + `pyav` + GPU.
- The current local `lerobot` environment has the training entrypoint but cannot
  see CUDA.
- The current local `isaaclab` environment can see CUDA but does not have the
  LeRobot training entrypoint.
- Therefore running now would produce a misleading result.

## Written Artifact

Added:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/training_smoke_preflight_d255/smolvla_smoke_cuda_preflight_summary_d255.json`

## Next Valid Options

Option A - local env repair:

- Repair or recreate the local LeRobot training environment so:
  - `torch.cuda.is_available()` is `True`;
  - `torch.cuda.device_count()` is `1`;
  - `lerobot-train` exists;
  - `dataset.video_backend=pyav` still works.
- Then rerun only the 50-step smoke.

Option B - install LeRobot training deps into a CUDA-working env:

- Use `isaaclab` or a fresh env as the base only after explicit dependency-change
  approval.
- This is riskier because dependency churn can disturb the IsaacLab stack.

Option C - RunPod/H100:

- Run the 50-step smoke remotely only with explicit runtime/cost approval.
- Stop the pod immediately after artifacts are copied back.

Option D - skip smoke for professor packet:

- Keep D254 as the professor-facing method-pipeline deliverable.
- This is defensible because D254 is not a model-performance claim.

## Still Blocked

- Actual SmolVLA/VLA fine-tuning beyond a minimal smoke.
- 20k candidate training.
- Offline evaluation script execution until an approved checkpoint exists.
- Additional Isaac render or 1000/10000 expansion.
- Raw cleanup/delete/archive/move.
- PPO/L2/Large PPO/action-teacher/RoArm deployment.
- RunPod runtime unless separately approved.
- B200/SSH/pull/.ssh copy.
