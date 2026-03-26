---
name: project_visual_prompting_hw
description: Digital vs Physical visual prompting — hardware feasibility analysis for each physical marker option (2026-03-25)
type: project
---

# Visual Prompting Hardware Feasibility (2026-03-25)

## Physical Marker Options — Ranked

| Option | Azure Kinect 720p | SigLIP recognition | Cost | Repro | Robustness | Overall |
|--------|------------------|--------------------|------|-------|------------|---------|
| ArUco/AprilTag | EXCELLENT | HIGH (printed symbols on internet) | ~$0 | EXCELLENT | MEDIUM | **BEST** |
| Colored tape/stickers | GOOD | HIGH (colored regions common) | ~$2 | GOOD | MEDIUM | **BEST** |
| LED ring/strip | GOOD | MEDIUM (bright ring novel) | ~$10 | GOOD | LOW (power) | MEDIUM |
| Laser pointer | MARGINAL | LOW (dot tiny at 720p, OOD) | $5 | POOR | VERY LOW | WORST |
| Smartphone screen | GOOD | HIGH (screens on internet) | ~$0 | MEDIUM | LOW (glare) | MEDIUM |
| Projector | POOR (washout) | LOW (projected light = OOD) | $200+ | POOR | VERY LOW | WORST |

## Key Hardware Findings

### Azure Kinect 720p (1280x720) → SigLIP 224x224
- Downscale factor: ~5.7x
- 1cm object at 500mm distance → ~13.7 pixels at 720p → ~2.4 pixels at 224px
- ArUco (10cm) at 500mm → ~137 px at 720p → ~24 px at 224px — CLEARLY VISIBLE
- Laser dot (3-5mm) at 500mm → ~4-7 px at 720p → ~0.7-1.2 px at 224px — BARELY VISIBLE
- Colored tape (3cm strip) at 500mm → ~41 px at 720p → ~7 px at 224px — VISIBLE

### SigLIP Frozen Encoder — Critical Constraint
- SigLIP trained on internet images: HAS seen arrows, circles, colored tape, markers
- SigLIP HAS NOT seen: laser dots on robot workspaces, projected light patterns
- ArUco/AprilTag: mixed — QR-code-like patterns may get generic feature response
- Colored regions (red tape, yellow sticker): HIGH probability of semantic response
- Conclusion: colored tape/stickers are the safest bet for SigLIP recognition

### Laser Pointer — Hardware FAIL
1. Dot size at 720p: 3-5mm → 4-7px → after 5.7x downscale = <2px at 224px: sub-pixel
2. Speckle pattern: coherent light = granular noise, not clean dot
3. Azure Kinect depth IR at 850nm overlaps red laser → depth sensor interference possible
4. Ambient light washout: red laser invisible under bright LED lighting
5. SigLIP never saw "red dot on gray table surface" as meaningful marker
6. Reproducibility: dot position changes with hand tremor, table angle, mounting jitter

### Digital Overlay — Implementation Analysis
- Pipeline: Azure Kinect RGB → OpenCV overlay (cv2.circle, cv2.arrowedLine) → resize to 224x224 → SigLIP
- Difficulty: LOW. ~10 lines of OpenCV. No new hardware. No calibration changes.
- Key question: does overlay pixel position match SigLIP's semantic attention?
- SigLIP has seen circles/arrows in internet images → HIGH probability of feature response
- Controlled injection: can place circle at exact target location every time → reproducible

## Fair Comparison Experiment Design

### Conditions
1. Baseline: RGB + text prompt only ("pick up the red block")
2. Digital: RGB + cv2.circle at target centroid + text
3. Physical-ArUco: RGB showing ArUco marker placed next to target + text
4. Physical-tape: RGB showing colored tape on target zone + text

### Controls Needed
- Same camera position (do NOT remount between conditions)
- Same lighting (draw curtains, use artificial light)
- Same number of demonstrations per condition (20-30 ep each)
- Same task and same target objects
- Test: n=20 trials per condition, binary success/failure

### Confound Warning
Physical markers provide SPATIAL INFORMATION that digital overlays can't unless calibrated.
ArUco provides 6-DoF pose → model implicitly gets target location even without VLA understanding.
This must be controlled: digital overlay must also be at the same spatial location (not random).

## Digital Prompting Implementation Cost
- Time: 2-3 hours to integrate into collect_data_manual.py
- No hardware changes needed
- Azure Kinect extrinsics already partially calibrated (calibrate_azure_kinect.py)
- For fair comparison: need camera-to-robot transform to project 3D target point to 2D pixel

**Why:** Physical prompting options vary enormously in SigLIP visibility and reproducibility.
**How to apply:** Recommend colored tape (immediate test) + digital overlay as two main conditions. Skip laser entirely.
