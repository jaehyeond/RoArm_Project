---
name: SmolVLA architecture constraints and research findings
description: Architectural details of SmolVLA 450M that affect research design and paper claims
type: project
---

## Architecture Summary
- 350M frozen VLM (SigLIP vision + SmolLM2 language) — pretrained on SO-100 ONLY
- 100M trainable Action Expert — flow matching, 10 denoising steps, Beta(1.5,1.0) noise
- 6-DOF → zero-pad 32-dim → process → unpad — means 26/32 dims always zero during training
- chunk_size = n_action_steps = 50 (default) — 1.67s at 30fps
- Image input: resized to 512x512 (config default: resize_imgs_with_padding=(512,512))

## Vision Encoder Details (verified from source)
- SigLIP variant: SmolVLMVisionConfig, hidden_size=768, image_size=512, patch_size=16, 12 layers
- Pixel-shuffle connector: scale_factor=4 → compresses 1024 raw patches → 64 tokens per image
- Per-camera token count: 64 tokens (after connector compression)
- Language tokens: tokenizer_max_length=48
- State tokens: 1 (after state_proj linear)

## Prefix Sequence Lengths
| Cameras | Image tokens | +lang+state | Prefix total | +chunk (action) | Total seq |
|---------|-------------|-------------|--------------|-----------------|-----------|
| 1 camera | 64 | +49 | 113 | +50 | 163 |
| 2 cameras | 128 | +49 | 177 | +50 | 227 |
| 3 cameras | 192 | +49 | 241 | +50 | 291 |

## smolvla_base Pretrained Config (verified from HuggingFace cache)
- TRAINED WITH 3 CAMERAS: camera1, camera2, camera3 (all 256x256 inputs, resized to 512)
- empty_cameras=0 (no fallback for missing cameras in base)
- prefix_length=0, pad_language_to="max_length"
- attention_mode="cross_attn" (Action Expert cross-attends to VLM KV cache)
- num_vlm_layers=16 (uses first 16 of 32 SmolLM2 text layers)
- num_expert_layers=0 (0 means same as VLM = 16)
- self_attn_every_n_layers=2 (self-attn every 2 layers, cross-attn otherwise)
- expert_width_multiplier=0.75 → expert hidden = 960 * 0.75 = 720

## Multi-Camera Support (verified from source code)
- SmolVLA NATIVELY supports multiple cameras — `prepare_images()` iterates over `config.image_features` dict
- Each camera is encoded SEPARATELY by SigLIP, then ALL embeddings concatenated in sequence and fed jointly into VLM transformer (early fusion via sequence concatenation)
- `empty_cameras` config handles missing cameras (filled with -1 padding, mask=0)
- LeRobot v3 format: separate video stream per camera key — multi-camera storage works natively
- There is NO per-camera encoder — same frozen SigLIP weights process all camera images
- CRITICAL: smolvla_base was pretrained with 3 cameras (camera1/2/3). Single-camera fine-tuning
  discards 2/3 of the pretraining visual capacity. The model adapts via training to this mismatch.

## Key VLM Architecture Details
- SmolLM2 text model: hidden_size=960, num_attention_heads=15, num_key_value_heads=5, head_dim=64, 32 layers total
- SmolVLA uses first 16 layers only (num_vlm_layers=16)
- Cross-attention mode: Action Expert cross-attends to frozen VLM KV cache
  - VLM processes images + language + state → fills KV cache once per inference
  - Action Expert queries VLM cache for all 10 denoising steps → efficient inference
- Inference: VLM forward pass ONCE (fills KV), then 10x Expert denoising steps (cheap)

## Adding a 2nd Camera: Technical Impact
1. Config change: add new key to input_features dict (e.g., "observation.images.wrist")
2. Data: ALL existing episodes invalid — must recollect with both cameras
3. Fine-tuning from smolvla_base: supported, the pretrained model already saw 3 cameras
   - OOD mismatch: base saw camera1/2/3 on SO-100; yours will be different camera positions/names
   - Normalization: new camera key gets its own MEAN_STD stats computed from your dataset
4. Memory: +64 tokens per camera → sequence grows by 64, attention O(n²) cost increases
5. Estimated VRAM increase (batch=64): roughly +0.8GB for 2-cam vs 1-cam (KV cache dominated)
6. Data requirement: 2-camera training needs MORE data — model must correlate viewpoints
   - Estimated: 1.5-2x the single-camera requirement (150→200+ episodes for 2-cam)

## Key Research-Relevant Findings

### SmolVLA as ablation instrument
SmolVLA is the ONLY open VLA trainable full-FT on consumer GPU (16GB) without LoRA/quantization.
This is its primary research value for single-GPU studies.
OpenVLA 7B requires LoRA. pi0 3B requires quantization.
Full fine-tune = no LoRA confounds = clean ablations.

### OOD gap is maximal for SmolVLA
Pretrained on SO-100 only (11,132 episodes, 128 datasets, all SO-100).
RoArm-M3 was NOT in pretraining. This is the maximum OOD transfer case.
Why it matters: if it works anyway, it's a stronger result than OpenVLA working
(OpenVLA pretrained on OXE with diverse embodiments).

### Action chunking finding (empirical, confirmed 2026-02-25)
4-chunk open-loop (4x50 steps) = 5/5 success
1-step closed-loop = 0/5 (drift failure)
This contradicts the standard advice that "closed-loop is always better."
Explanation: per-step noise accumulates; committing to a chunk lets gripper signal propagate.
Research question: when does closed-loop fail and why?

### Pretraining transfer value
Pretrained vs scratch: 78.3% vs 51.7% success (from MEMORY.md).
Even for OOD embodiments, pretraining still provides value.
The value comes from SigLIP's visual representations, not the action representations.

### Zero-padding research implication
For 6-DOF arm, 26/32 action dims are always zero.
Action Expert spends capacity learning to ignore them.
A 6-dim-native Action Expert would be more efficient.
Interesting thesis direction — NOT CoRL material (requires retraining from scratch).

## Untested Architectural Questions
1. SigLIP feature separability for novel objects (cup/box/tool vs sponge) — 2 hour test, do immediately
2. Beta(1.5,1.0) noise schedule optimality for RoArm-M3 dynamics vs SO-100
3. 224x224 vs 384x384 input resolution effect on grasping precision
4. Optimal chunk size for different manipulation subtasks (pick=shorter? push=longer?)

## What NOT to change in architecture
- Do NOT modify LeRobot source code
- Do NOT change noise schedule without retraining from smolvla_base
- Do NOT remove VLM freezing (training VLM requires 10x more compute)

## Deep Source Analysis Findings (2026-03-31)

### Exact State Injection (confirmed from modeling_smolvla.py lines 571-697)
- state → pad(6→32) → state_proj Linear(32→960, TRAINED) → 1 token in prefix
- att_mask=1 for state: image/language CANNOT attend to state token
- Action Expert cross-attends to ALL prefix KV — state gets 1 slot vs 64 image slots
- state_proj is the ONLY fully trained projection in the prefix pathway

### Proprioceptive Echo — Architectural Root Cause Confirmed
The frozen visual pathway (SigLIP + connector = 100% frozen) vs trained state_proj creates
a gradient imbalance. If state → action is a near-valid predictor (center-biased data),
state_proj optimizes for echo. Visual gradients cannot compensate — they are zero.
This is architecturally expected behavior under bad data conditions. Not a bug.

SO100 avoids echo because HOME start makes state at t=0 uninformative.
v5 had no HOME start → state encoded zone → echo was the Bayes-optimal strategy.

### Fixes Without Source Modification
1. Delta actions in convert script + deploy script: breaks echo shortcut [BEST]
2. State dropout (30% episodes → dataset_mean) in convert script [GOOD]
3. Balanced zones + HOME start: necessary but insufficient alone

### Official Recipe Key Numbers (confirmed from smolvla.mdx)
- 50 ep (5pos × 10rep), batch=64, steps=20K → works on SO100
- scheduler: warmup=1K, decay=30K > steps=20K → INTENTIONAL, LR ends at ~50% peak
- v5 200K steps = 952 epochs = echo amplification, not improvement
- v6 must match: ~200fr/ep, 5zone × 10rep, HOME start, 20K steps

### Full analysis script
`/home/cgxr/Documents/Robotics/RoArm_Project/model_smolvla_architecture_critique.py`
