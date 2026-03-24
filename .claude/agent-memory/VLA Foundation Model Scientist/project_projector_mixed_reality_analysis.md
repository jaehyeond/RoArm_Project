---
name: Projector Mixed Reality Robot System Analysis (2026-03-24)
description: Critical analysis of Unity projector + real robot + SAM2 architecture proposal. 4 failure modes identified, 1 genuine research direction isolated.
type: project
---

## Proposal Summary
Unity renders scene → ceiling projector projects onto table → real robot + real objects on table → SAM2 segments → no SigLIP/VLA

## Verdict: NOT viable as proposed. Core architecture is undefined.

## 4 Critical Failure Points

### 1. Architecture hole: what model actually drives the robot?
- If no VLA/SigLIP, what produces motor commands?
- SAM2 provides masks only — segmentation is NOT a policy
- No precedent found in literature for projector-based VLA policy substitution
- Student has not defined this

### 2. SigLIP domain gap is WORSE with projector, not eliminated
- SigLIP frozen on real images (ImageNet/LAION)
- Projected scene = low-lumen, washed-out, no 3D cues, flat onto table
- SplatSim (3DGS) achieves cosine dist ~0.1-0.2; projector output likely WORSE than Isaac rasterizer (0.6-0.8)
- Projected images are NOT photorealistic — this makes the domain gap larger, not smaller
- "Don't need SigLIP" is not a solution; it's not knowing what to replace it with

### 3. SAM2 cannot reliably segment projected light from real objects
- SAM2 is trained on object boundaries defined by geometry + surface normals
- Projected light has NO geometry — it is illumination change on existing surface
- Projected color/pattern on table is semantically same as "table with different lighting"
- SAM2 failure modes confirmed: camouflage (object blends with projected background), flat texture (no edge cues), and the "tablecloth problem" (SAM segments the projection as a region, not as an object)
- No literature found validating SAM2 reliability on projected-light vs real-object distinction

### 4. Technology looking for a problem
- Student admits "don't know what to project or what task robot should do"
- This is the classic TRL-0 trap: build hardware first, find problem later
- No genuine robotics problem has been identified that this setup uniquely solves

## What Does Exist (Verified)

| Related Work | What it does | Why different |
|---|---|---|
| RoomAlive (2014, Microsoft) | Projector maps onto room for gaming | No robot, no manipulation |
| AR workspace preparation (arXiv 2311.05562) | AR for human-robot collaboration legibility | No policy learning |
| AR-guided data collection (IDEA 1, our work) | AR on phone/HMD to guide human demonstrator | Robot acts on real objects only |
| Structured light robot sensing | Depth + shape estimation using projector patterns | Not for scene creation |

## The Genuine Kernel (if any)

The one potentially valid use case: projecting a TARGET MARKER (circle, color patch) onto the table surface to tell the robot WHERE to pick/place. This is:
- Technically simpler than full scene projection
- Avoids the model architecture problem (still use SmolVLA, just add a spatial prompt)
- Conceptually close to "visual goal conditioning" literature

But this is NOT what the student described. The student described projecting a full Unity scene as if it replaces real objects.

## Connection to Our Research (IDEA 1)

AR-Guided Demo Collection (our IDEA 1) is the correct interpretation of the same intuition:
- AR overlay on demonstrator's phone/tablet = guides where to place real objects
- Real objects remain real — no domain gap issue
- Policy still runs on real camera images — SigLIP works as intended
- This is validated; projector-robot is not

## Why: The projector creates a domain gap problem that the student believes it solves.
## How to apply: When user brings this idea up, redirect to AR-guided collection (IDEA 1) as the correct implementation of the same intuition.
