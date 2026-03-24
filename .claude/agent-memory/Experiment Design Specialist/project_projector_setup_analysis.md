---
name: project_projector_setup_analysis
description: Ceiling beam projector + SAM3 setup analysis — application brainstorm and critical evaluation for CoRL 2026
type: project
---

Setup: Unity scene projected via ceiling beam projector onto table. SAM3 segments projected light vs robot vs real objects. Azure Kinect sees the projected content.

Key insight: the projector's unique value is that projected content is visible to BOTH the human demonstrator AND the Azure Kinect camera simultaneously. Any idea that only helps the human (not the camera) can be replaced by a cheaper screen.

## Top 3 Ranked Applications

### Rank 1: Projected Task Context for Multi-Task Learning
- Projects different colored zones/symbols during collection AND inference
- The projected content becomes part of the camera observation fed to SmolVLA's SigLIP backbone
- Enables dynamic task specification for a fixed policy without retraining
- CoRL framing: "Projector-Conditioned VLA: Dynamic Task Specification via Workspace-Projected Context at Zero Policy Cost"
- CRITICAL PRE-CONDITION: SigLIP must actually attend to projected zones. Test with cosine similarity of embeddings before committing. If SigLIP ignores it, this fails.

### Rank 2: Projected Background Texture Augmentation
- Projects different textures onto table surface DURING collection (10ep × 5 textures)
- Creates REAL images with real lighting on augmented backgrounds
- Claim: real projected texture > GenAug post-hoc augmentation because SigLIP feature space differs for real vs. rendered textures
- Experiment: Condition A (baseline) vs B (projected textures) vs C (post-hoc GenAug) → if B >> C, strong projector necessity argument
- HIGH feasibility, no new architecture assumptions required

### Rank 3: Spatial Target Projection (already in plan)
- Projects target circles at specific (x,y) positions for operator guidance
- Camera sees the target → enables automatic placement error verification (projected target position = ground truth reference)
- This is the AR-guided demo collection idea, now with clearer mechanism for why projector > screen

## Rejected Ideas
- Projected object appearance variation: requires real-time tracking of projection onto moving object. Not feasible.
- Non-expert teaching interface: screen works as well, no novelty.
- Episode replay projection: human-factors result, not ML contribution.
- Projecting onto the robot itself: arm shadows corrupt projection at critical moments.

## Pre-Conditions to Validate This Week
1. Projector-camera registration quality: can SAM3 segment projected light from object/table in room lighting? (1-2 hours to test)
2. SigLIP projected content sensitivity: cosine similarity test with colored zones at 3 positions. If embeddings identical → Rank 1 fails.
3. Shadow-free workspace: at which joint angles does arm occlude projection? Defines usable area within 25cm reach.

## Integration with Existing Experiment Plan
Rank 2 + Rank 3 can be combined as conditions in the existing AR guidance experiment (4-condition structure already designed):
- A: 50ep baseline, no guidance, no augmentation
- B: 50ep + offline GenAug-style (already planned)
- C: 50ep + AR spatial target circles (Rank 3)
- D: 50ep + AR spatial targets + projected texture variety (Rank 2 + 3 combined)

Rank 1 requires a separate PoC experiment first (1 week). Only commit to it if SigLIP sensitivity test passes.

**Why:** Student has Unity/XR expertise and owns both projector hardware and Azure Kinect. The projector setup creates a uniquely testable claim about real-image augmentation that post-hoc methods cannot match.
**How to apply:** Always check whether the projected content is in the camera frame and whether SigLIP encodes it. If not, the projector reverts to "human guidance only" and loses its primary technical justification.
