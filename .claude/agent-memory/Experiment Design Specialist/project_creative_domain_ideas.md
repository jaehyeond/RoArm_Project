---
name: project_creative_domain_ideas
description: Creative domain brainstorm (2026-03-24) — 11 ideas across manufacturing/healthcare/education/food/entertainment/metaverse with projector + XR angle. Priority stack and CoRL recommendation included.
type: project
---

## Context
Generated 2026-03-24. Cloud GPU now available (any-size VLA possible). Ceiling projector key new hardware variable.

## Priority Stack for CoRL 2026 (May 28)

| Rank | Idea | Feasibility | Key Pre-condition |
|------|------|-------------|-------------------|
| 1 | Projected Texture Augmentation (real light > GenAug) | HIGH | SigLIP gate test |
| 2 | Projected Task Context for Multi-Task VLA | HIGH | SigLIP gate test |
| 3 | Multi-Arm + Projected Zone Coordination | MEDIUM-HIGH | 3-arm workspace layout test |
| 4 | Laser Pointer as Attention/Pointing Mechanism | HIGH | None |

## Decisive Gate Test (do this first, 2 hours)
- Collect 20 images: projected colored zone at position A / position B / absent
- Compute SigLIP cosine similarity between zone-present and zone-absent
- < 0.80 similarity: SigLIP detects projection → proceed with Ideas 1+2
- > 0.95 similarity: SigLIP blind → Idea 1 fails, Idea 9 partially valid (real photon argument holds)

## Top 3 New Ideas (not in prior research notes)

### Idea 9: Projected Texture Augmentation (STRONGEST NEW IDEA)
- Problem: VLA trained on white table fails on new backgrounds. GenAug edits pixels but not SigLIP features.
- Solution: project 5 different textures onto table during collection (10ep × 5 textures)
- Claim: projected textures = real photons → real SigLIP feature diversity (physically grounded)
- Baseline comparison: same 50ep + post-hoc GenAug
- Experiment: Condition A (baseline), B (projected textures), C (GenAug post-hoc). Test all 3 on OOD background.
- Projector necessity: camera is looking AT the table with real reflected light. Screen cannot do this.
- Existing work gap: no papers on physical projection for real-light background augmentation in VLA context

### Idea 1: Projected Task Context for Multi-Task (already partially in projector analysis)
- Project colored zone = task specification visible to BOTH human and camera
- Change projection = change task. No retraining.
- Experiment: 3 tasks × 50ep each, zone visible in training. Test with zone present/absent.

### Idea 7: Laser Pointer as Attention Mechanism
- Operator points at target → laser dot appears in camera frame → "pick the highlighted object"
- Replace language prompt with laser (x,y) position
- Training: 50ep per object, all with laser dot visible
- High feasibility, strong "simple idea, clear result" structure
- Check Molmo (pointing model, 2024) overlap before claiming gap

## Rejected for CoRL (wrong venue or timeline)
- Healthcare rehab robot (HRI, post-CoRL)
- Human draws trajectory on table (HRI, post-CoRL)
- Digital twin fidelity thresholds (ICRA, thesis Chapter 5)
- Projected correction (HRI, user study required)
- Cross-embodiment via projected morphology (speculative, high risk)

## CoRL Recommendation: Combine Idea 9 + Idea 1
Title: "Projected Workspace Conditioning for Consumer VLA: Real-Light Augmentation and Dynamic Task Specification via Ceiling Projector"
- Contribution 1: projected textures = physically-grounded appearance augmentation (Idea 9)
- Contribution 2: projected zone = zero-cost task re-specification (Idea 1)
- Unifying claim: $200 projector solves appearance generalization + multi-task specification simultaneously

**Why:** User has projector, Unity expertise, and existing AR-guided collection work. This extends that work with a projector-specific scientific claim (real photons > pixel editing) that is both novel and testable in the existing 4-condition experiment structure.
**How to apply:** Do NOT commit until SigLIP gate test passes. All projection conditioning relies on SmolVLA's frozen SigLIP backbone actually encoding projected content.
