---
name: CoRL 2026 Research Claim Verification (2026-03-23)
description: Fact-check of 5 claimed research gaps for CoRL 2026 submission. All 5 required significant qualification. 3 were outright false.
type: project
---

# CoRL 2026 Research Claim Verification

**Date verified: 2026-03-23**

## Claim 1: "Nobody has used LLM/VLM to automate demo collection for VLA"
**VERDICT: FALSE**
- AutoRT (Google DeepMind, 2401.12963): LLM selects tasks, VLM verifies success, real robot fleet
- SOAR (CoRL 2024): VLM-in-the-loop autonomous practice, successful rollouts become training data
- RoboGen (2311.01455): GPT-4 generates tasks + demo collection pipeline
- GenSim/GenSim2 (2310.01361, NeurIPS 2024): GPT-4 generates simulation code + demos at scale
- RoboCasa (2406.02523): LLM generates 100+ task variants, auto-collects demos

**Defensible narrow gap:** VLM-guided data curation loop on a SINGLE consumer robot ($130), no simulator, no fleet.

## Claim 2: "Sim-to-real VLA doesn't work (image-based)"
**VERDICT: PARTIALLY TRUE**
- Real2Render2Real (2505.09601, CoRL 2025): 73% real-robot success from purely synthetic training images
- RoboSplat (2504.13175, RSS 2025): 3DGS novel views improve real-robot generalization
- SplatSim (2409.10161, ICRA 2025): 3DGS renders → real robot transfer works

**What is still hard:** Plain IsaacGym/MuJoCo (non-photorealistic) → real VLA still fails. SmolVLA specifically would have triple domain gap (pretrained on real SO-100 → fine-tuned on simulated RoArm → deployed on real RoArm).

## Claim 3: "Consumer-arm factory testbed = nobody"
**VERDICT: PARTIALLY TRUE (weak)**
- LIBERO, FurnitureBench, ALOHA exist for structured tabletop tasks
- No paper uses 3+ identical $100-200 arms in logistics-style workflow with VLA policies
- But this is a system/benchmark contribution, not a data-efficiency contribution

## Claim 4: "VLA data augmentation is unsolved"
**VERDICT: FALSE**
- MimicGen (2310.17596, CoRL 2023, NeurIPS 2024 oral): 10x-100x demo amplification via trajectory retargeting
- RoCoDA (2411.13031): Counterfactual augmentation specifically for VLA
- TGM-VLA (2603.00615, March 2026): Task-guided mixup, 3x data efficiency improvement
- GenAug, Rosie, CACTI: image-level augmentation for imitation learning

**Real gap:** Trajectory-level augmentation WITHOUT a simulator for unstructured grasping tasks.
Our finding: collection-time spatial diversity > post-hoc augmentation.

## Claim 5: "Multi-robot fleet data collection = unstudied"
**VERDICT: FALSE**
- AutoRT: Up to 20 real robots, pooled for VLA training
- DROID (2403.12945, RSS 2024): 86 identical robot setups, 564 hours
- Seed2Scale (2603.08260): Explicit fleet-size scaling study (warehouse robots)
- Open X-Embodiment: 22 embodiments data pooling

**Real gap:** N=1/2/3 fleet scaling for consumer-class arms ($130 each). We have the hardware to test this.

## Why: 2026-03-10 pattern recurred
User was right to have this verified. Claims were too broad. The defensible contributions are:
1. FK-depth + gripper phase + static frame = collection-time quality metrics (not post-hoc)
2. v1(50ep, bad quality)=0% vs v3(74ep, good quality)=100% = empirical data point
3. Real-robot-only self-improve loop (no simulator) on sub-$500 hardware
4. OOD embodiment scaling: episodes × quality × steps interaction study

**How to apply:** Never claim any of these 5 claims as gaps in future work without re-verifying.
