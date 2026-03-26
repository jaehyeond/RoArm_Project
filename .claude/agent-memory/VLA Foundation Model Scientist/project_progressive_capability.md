---
name: Progressive VLA capability building — SmolVLA + RoArm-M3
description: Research synthesis on multi-position, multi-object, sequential tasks for SmolVLA. Data requirements, language conditioning, evaluation metrics. (2026-03-25)
type: project
---

## Context
SmolVLA + RoArm-M3. Stage 1 complete (100% success, 74 ep, single sponge).
Researched: multi-position, multi-object, sequential tasks.
Reference file: `model_progressive_vla_capability.py`

**Why:** Needed to plan Stage 2-4 of capability building for CoRL paper.
**How to apply:** Use episode counts, language format, and curriculum order below.

---

## Episode Requirements (verified)

| Stage | Config | Episodes | Steps |
|-------|--------|----------|-------|
| Stage 2: multi-position | 5 zones × 30ep | 150 | 200K |
| Stage 3: multi-object | 3 objects × 50ep | 150 | 200K |
| Stage 4: sequential (optional) | 100ep | 100 | 200K |

Critical: BALANCED episodes per zone/object mandatory — MEAN_STD normalization breaks with imbalance.

---

## Language Conditioning (verified from source)

- Tokenizer: SmolVLM2-500M-Video-Instruct, max 48 tokens
- REQUIRED: task string must end with `\n` (SmolVLANewLineProcessor)
- Good format: `"Pick up the red sponge\n"` (~7 tokens)
- Each object = separate task_index in dataset
- task_index handled by DataLoader, NOT model — no code changes needed

SigLIP zero-shot capability:
- Frozen SigLIP CAN distinguish colors and common objects (PIVOT 2402.07872 confirms)
- Action Expert must be trained to USE language → must have training data per object
- No held-out zero-shot objects — all objects must be in training data

---

## Sequential Task Capacity

SmolVLA 450M CANNOT do single-prompt multi-step (no memory across chunks).
Best approach: Subtask decomposition
- Phase 1: task="Pick up the sponge\n"
- Phase 2: task="Place sponge in box\n"
- Phase 3: task="Pick up the cup\n"
- Subtask switching: heuristic (gripper state) or human interrupt

LeRobot subtask annotation: supported via meta/subtasks.parquet
Note: SmolVLA does NOT auto-read subtask field — must pass as "task" at inference.

---

## Curriculum Order

Stage 1 (done): single object, single position
Stage 2 (next): single object, 5 zones
Stage 3: 3 objects, 5 zones (350-450ep? or 150ep with mixed zones)
Stage 4 (optional CoRL): sequential pick-place

Each stage adds ONE new generalization variable (spatial, language, temporal).
This mirrors RT-2 evaluation structure — appropriate for CoRL methods paper.

---

## Data Mixing Strategy

- Do NOT mix old stage-1 data with new stages (camera OOD, different MEAN_STD)
- Balanced collection → natural uniform sampling in LeRobot DataLoader
- Always train from smolvla_base pretrained (78.3% vs 51.7% from scratch)
- No per-task sampling weights available in LeRobot by default

---

## Evaluation Metrics (CoRL-grade)

Tier 1 (required):
1. Per-zone success rate (spatial generalization heatmap)
2. Language conditioning accuracy (distractor test: red+blue object, pick one)
3. Overall success rate per task

Tier 2 (important):
4. Failure mode taxonomy (Approach/Grasp/Lift/Language failure types)
5. Chunk-to-chunk joint discontinuity (< 2° target)
6. Action diversity score (std across 50-step chunk, > 5° = active model)

Tier 3 (ablation):
7. Offline L2 vs online success (known mismatch — interesting finding)
8. Denoising variance as uncertainty proxy (N=5 sample_actions() runs)
   → forward(reduction='none') already implemented in source code
