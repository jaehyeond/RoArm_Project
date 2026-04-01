---
name: V5 Proprioceptive Echo Analysis (2026-03-31)
description: VGST test revealed model copies state.base to action.base. Quantifies why the v5 dataset structure makes this optimal behavior.
type: project
---

## Finding: Proprioceptive Echo Is Optimal For v5 Dataset

VGST test showed SmolVLA copies state input to base output, ignoring the image.
Analysis of 13,470 frames from `lerobot_dataset_v5/data/chunk-000/file-000.parquet` confirms
this is not a model failure — it is the optimal response to the dataset structure.

**Why:** Episodes start at approach pose (already facing the sponge), so base joint
never needs to change direction from state to action. The model has no need for image.

**How to apply:** Any new data collection MUST start episodes from home (base=0) if
we want the model to learn to use the camera for directional reasoning.

## Key Numbers

| Metric | Value |
|---|---|
| r(state.base, action.base) | 0.9996 |
| Frames where \|action-state\| < 0.5° | 88.6% (11,939/13,470) |
| Frames where \|action-state\| < 1.0° | 90.8% |
| Frames where \|diff\| > 5° | 0.4% (55 frames) |
| Frames where \|diff\| > 10° | 0.007% (1 frame) |
| Sign-crossing frames (state vs action) | 0 of 13,470 |
| MAE if model echoes state.base | 0.29° |

## Per-Joint Echo Strength (r values)

| Joint | r | <0.5° | >5° |
|---|---|---|---|
| base | 0.9996 | 88.6% | 0.4% |
| shoulder | 0.9977 | 70.9% | 0.2% |
| elbow | 0.9987 | 83.0% | 4.1% |
| wrist_pitch | 0.9983 | 77.7% | 5.3% |
| wrist_roll | 0.9990 | 92.9% | 2.4% |
| gripper | 0.9874 | 82.4% | 9.0% |

Gripper has the WEAKEST echo signal (r=0.9874, >5° for 9% of frames) — gripper is
the joint where the model MOST needs the image. Base has the STRONGEST echo.

## First-10-Frame Analysis

- Mean |action-state| in first 10 frames: 0.043°
- Episodes with max first-10 diff > 2°: 2 / 136
- Episodes with max first-10 diff > 5°: 0 / 136
- Episodes start ALREADY AT APPROACH POSE → first frames have near-zero divergence

## Left/Right Episode Analysis

- LEFT episodes (min action.base < -15°): 29 episodes
- RIGHT episodes (max action.base > 15°): 53 episodes
- LEFT: frames where state.base > 0 AND action.base < 0: 1 frame (0.03%)
- RIGHT: frames where state.base < 0 AND action.base > 0: 5 frames (0.1%)
- Sign-crossing frames with strong divergence (both > 5° from zero): 0 frames

## Root Cause

Episodes start at approach pose (mean start_state.base = 8.72°, std = 37.42°).
The arm is ALREADY rotated to face the sponge at frame 0.
Therefore state.base ≈ action.base throughout the entire episode.

A model that outputs `action.base = state.base` achieves 0.29° MAE and will appear
to succeed at deployment — but has learned NOTHING about where the sponge is from
the image.

## Required Fix (Data Collection)

Episodes MUST start from home position (base~0, shoulder~2.5°, elbow~90°).
Only then will there be a phase where:
  - state.base = 0 (home)
  - action.base = target_base (e.g. -30° for a left sponge)
These are the frames where image information is NECESSARY to predict the correct action.
Without this, any model can achieve low training loss through proprioceptive echo.

## Analysis Scripts
- `data_v5_proprioceptive_echo_analysis.py` — proprioceptive echo standalone analysis
- `data_v5_deployment_failure_analysis.py` — full V5 vs V3 deployment failure comparison (2026-03-31)

## V5 vs V3 Confirmed Comparison (2026-03-31 from parquet)

| Metric | V5 | V3 |
|---|---|---|
| r(state.base, action.base) | 0.9996 | 0.9992 |
| \|delta_base\| > 2° | 6.3% | 7.5% |
| Sign-crossing frames | 0.40% | 0.26% |
| Episode start shoulder | 44.1° mean | 2.8° mean |
| Starts near home (sh<10°) | 0/136 (0%) | 74/74 (100%) |
| Base range in first 30 frames | mean 1.74° | mean 23.69° |
| Episodes no early approach | 91/136 (66.9%) | 15/74 (20.3%) |
| Gripper opens in first 50fr | 136/136 (100%) | 32/74 (43%) |
| Gripper closes in first 50fr | 110/118 (93%) | 0/10 (0%) |
| Gripper q10 | 16.2° | 1.7° |
| Gripper <10° (% frames) | 5.8% | 30.6% |
