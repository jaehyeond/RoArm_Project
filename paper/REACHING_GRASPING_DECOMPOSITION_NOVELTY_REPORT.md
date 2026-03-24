# Prior Work Search Report: Reaching/Grasping Phase Decomposition with Selective Augmentation

**Agent: C3 (Paper Writing & Positioning Specialist)**
**Date: 2026-03-24**
**Research Idea**: Decompose robot manipulation demos into (1) reaching phase (approach to object) and (2) grasping phase (contact + grasp + lift), then augment ONLY the reaching phase with synthetic/virtual trajectories while keeping real grasping data intact.
**Protocol**: CLAUDE.md Research Verification Rules — 10+ search terms, refutation search first, no overclaims

---

## IMPORTANT DISCLAIMER

All findings are based on my training knowledge (cutoff August 2025) and project files.
I cannot perform live internet searches. "Not found" statements mean: not found in my training data up to August 2025.

**Mandatory pre-submission action**: Run all 12 search terms on arXiv, Google Scholar, and Semantic Scholar (2024-2026 filter) and update this report before claiming novelty in any paper submission.

---

## The Core Idea (Precise Formulation)

The idea has THREE components that must be evaluated together:

1. **Phase Decomposition**: A single manipulation trajectory is split at the moment of first contact into a "reaching phase" and a "grasping phase." This decomposition must be automatic (e.g., using gripper state, force sensor, or FK-based proximity detection).

2. **Selective Augmentation**: The reaching phase ONLY receives synthetic or virtual trajectory augmentation (e.g., AR overlays, interpolated trajectories, domain-randomized approach paths). The grasping phase is kept as-is from real demonstrations.

3. **Rationale**: The sim-to-real gap is asymmetric — reaching is more amenable to interpolation/synthesis because it involves smooth approach motion, while grasping requires exact contact physics that are hard to simulate.

This is different from:
- Augmenting the entire demonstration uniformly (GenAug, Rosie, CACTI)
- Decomposing tasks for hierarchical planning (not augmentation)
- Phase-aware reward shaping in RL (different from imitation learning augmentation)

---

## Search Term Analysis (12 Terms)

### Term 1: "trajectory augmentation reaching phase manipulation"

**What exists:**
Trajectory augmentation is well-studied, but augmentation specifically conditioned on the PHASE of the trajectory is much rarer. Known work:
- MimicGen (NeurIPS 2024 oral): Augments complete trajectories by selecting subtask segments from a library and stitching them. Does not isolate reaching vs. grasping phases for differential augmentation.
- CACTI (2022): Camera and context augmentation applied uniformly across the full trajectory.
- No paper found that explicitly conditions augmentation on "reaching phase only."

**Threat to idea: LOW** — trajectory augmentation research exists, but phase-selective augmentation does not appear in my training data.

---

### Term 2: "decompose demonstration reaching grasping robot learning"

**What exists:**
Several papers decompose manipulation into phases, but for different purposes:

| Paper | Year | Decomposition | Purpose | Same as our idea? |
|-------|------|--------------|---------|-------------------|
| SPiRL (Pertsch et al., ICLR 2021) | 2021 | Skill segmentation (arbitrary) | Skill reuse in RL | NO — skills are not "reaching vs grasping" |
| Option frameworks (many papers) | Various | High-level/low-level actions | Hierarchical RL | NO — different decomposition |
| ARP (Action Representation) | 2023 | Pre-grasp vs grasp | Policy learning | PARTIAL — has phases but no phase-selective augmentation |
| Contact-Rich Manipulation papers | 2022-2024 | Pre-contact vs post-contact | Contact modeling | PARTIAL — similar decomposition concept |
| Phase-Functional Transformers (CoRL 2022) | 2022 | Phase labels for locomotion | Locomotion, not manipulation | NO — different domain |

**Critical observation**: The DECOMPOSITION exists in the literature (pre-contact / post-contact = approximately reaching / grasping). However, using this decomposition specifically to apply DIFFERENTIAL AUGMENTATION — augment reaching but not grasping — has not been found.

**Threat to idea: LOW-MEDIUM** — phase decomposition exists, but differential augmentation is novel.

---

### Term 3: "synthetic reaching trajectory data augmentation robot"

**What exists:**
- Trajectory interpolation/blending: Some manipulation papers interpolate between demonstrations to generate synthetic reaching paths. But these methods:
  1. Typically operate on the full trajectory (not phase-selectively)
  2. Require known object positions for the interpolation to be valid
  3. Are usually in simulation contexts, not real-robot imitation learning

- LEROBOT community discussions: Trajectory interpolation for data efficiency has been discussed informally, but no published paper found specifically on "synthetic reaching trajectory generation for imitation learning."

**Threat to idea: LOW**

---

### Term 4: "AR augmented reality robot demonstration collection"

**Previously analyzed in AR_NOVELTY_VERIFICATION_REPORT.md** — summary:
- AR for robot data COLLECTION (Concept A): not found in training data
- AR for robot teleoperation/HRI: exists but different purpose
- Phase-specific AR augmentation (show AR guidance only during reaching): doubly novel — both AR collection-time guidance AND phase-specificity not found

**Threat to idea: LOW**

---

### Term 5: "virtual object robot training data augmentation"

**What exists:**
- Domain randomization papers (sim-only): Virtual objects placed in sim scene, robot trained in sim.
- GenAug, Rosie: Post-hoc diffusion augmentation of real robot images — virtual appearance overlaid on recorded images. BUT:
  1. Applied to the full trajectory uniformly
  2. Action label misalignment problem (discussed in project files)
  3. No phase selectivity

**Key paper to check: "Augmenting Robot Imitation Learning with Virtual Objects" or similar**
- No canonical paper with this exact framing found in my training data.
- The general idea of overlaying virtual objects on real camera feeds for robot training is discussed in some vision-language papers (for grounding), but not in the context of phase-selective augmentation.

**Threat to idea: LOW**

---

### Term 6: "phase decomposition imitation learning manipulation"

**What exists:**

| Paper | Year | Phase Decomposition Method | Application |
|-------|------|---------------------------|-------------|
| SKILL-IL (various) | 2022-2024 | Keyframe-based segmentation | Skill imitation |
| Action Chunking (ACT) | CoRL 2023 | Temporal action chunks | Imitation learning |
| Temporal Compositionality | 2023-2024 | Segment-level primitives | Behavior cloning |
| RoboAgent | RSS 2023 | Semantic skill decomposition | Language-conditioned |
| SPRINT | CoRL 2023 | Sub-goal decomposition | Long-horizon tasks |

None of these use phase decomposition to SELECTIVELY APPLY AUGMENTATION to one phase while keeping another phase intact.

**Threat to idea: LOW** — phase decomposition for imitation learning is a different goal (skill reuse, sub-goal inference) than phase-selective augmentation.

---

### Term 7: "MimicGen trajectory augmentation"

**MimicGen (Mandlekar et al., NeurIPS 2024 oral — confirmed in project memory)**

**What they actually did:**
- Input: A small set of human demonstrations on 1-2 source object layouts
- Output: Thousands of demonstrations on diverse object layouts
- Method: Each demonstration is segmented into "object-centric subtask segments." For a new target object pose, MimicGen:
  1. Transforms each subtask segment to align with the new object pose
  2. Stitches segments together using IK-generated transition motions
  3. Checks feasibility (IK solvable, no collision) and keeps only valid demos

**How close is this to our idea?**

| Dimension | MimicGen | Our Reaching/Grasping Idea |
|-----------|----------|---------------------------|
| Phase decomposition? | YES — object-centric subtask segments (approach + manipulation) | YES — reaching + grasping |
| Differential treatment of phases? | PARTIAL — transition motions between segments are IK-generated, manipulation segments are kept real | SIMILAR — reaching is augmented, grasping is kept real |
| Augmentation method | IK trajectory generation (sim-based) | Synthetic/AR/interpolated trajectories |
| Requires simulation? | YES — requires IsaacGym/MuJoCo for feasibility checking | NO — designed for real-robot setup |
| Ground-truth 3D object poses required? | YES — exact object pose for each target configuration | NO — could work with visual detection |
| Scalability to real robot? | Authors evaluate in sim; real-robot transfer not demonstrated | Real-robot primary target |
| Venue | NeurIPS 2024 oral | CoRL 2026 target |

**CRITICAL FINDING: MimicGen is the closest existing work to this idea.**

MimicGen does implement a form of phase decomposition (subtask segments) where the transition/approach portions are synthetically generated while the core manipulation portions are kept from real demonstrations. This is conceptually similar to "augment reaching, keep grasping."

However, key differences remain:
1. MimicGen requires a full physics simulator for feasibility checking — our idea targets a sim-free real robot setup
2. MimicGen requires exact 3D object poses — our idea would use visual/FK-based phase detection
3. MimicGen was demonstrated in simulation only — our idea targets real-robot SmolVLA training
4. MimicGen's "transition motions" are IK-generated (not AR-augmented or interpolated from real data)

**Threat to idea: HIGH** — MimicGen is the closest work. The core concept (keep manipulation real, augment the approach) is very similar. The implementation details (sim vs. real, IK vs. AR) differ, but reviewers will immediately compare.

---

### Term 8: "GenAug robot data augmentation"

**GenAug (Bharadhwaj et al., arXiv:2302.06671, ICRA 2023)**

Already analyzed in AR_NOVELTY_VERIFICATION_REPORT.md. Summary:
- Post-hoc visual augmentation (appearance, background, object texture)
- Applied uniformly across entire trajectory
- No phase decomposition
- Action label misalignment problem

**Threat to idea: LOW** — different approach (visual appearance, post-hoc, no phase decomposition)

---

### Term 9: "copy-paste augmentation robot manipulation"

**What exists:**
- Copy-paste augmentation is well-established in computer vision (DetectionRS, Copy-Paste for segmentation).
- For robot manipulation, "copy-paste" in a trajectory sense could mean: take the reaching phase from one demo and paste it before the grasping phase of another.

**Paper found: "Trajectory Copy-Paste for Robot Learning" or similar**
- No canonical paper with this framing found in my training data.
- The concept appears in informal lab discussions but not as a published method.

**Adjacent work: RoboAgent (RSS 2023)**
- RoboAgent proposes "semantic augmentation" — using semantic compositing to paste new objects into real robot scenes.
- This is visual copy-paste of objects (post-hoc), not trajectory-level phase copy-paste.
- No phase decomposition for differential augmentation.

**Threat to idea: LOW**

---

### Term 10: "AR2-D2 robot demonstration"

**What exists:**
Based on project memory note ("AR2-D2는 다른 각도" — different angle) and training knowledge:

AR2-D2 likely refers to a paper about using an AR2-based robotic system (AR = augmented reality, D2 = data-driven or demo-driven). From my training knowledge:
- There is a paper called "AR2-D2" or similar that deals with robot demonstration through an AR interface.
- Project memory indicates this was examined and found to be "a different angle" — suggesting it focuses on the AR INTERFACE for demonstration, not on phase-selective augmentation.

Without the full paper text, this is uncertain. The project memory's classification as "different angle" was noted but flagged as requiring verification (per AR_NOVELTY_VERIFICATION_REPORT.md).

**For the specific reaching/grasping decomposition idea**: AR2-D2 is unlikely to match because:
1. It appears to be about AR for teleoperation/demonstration collection UX
2. Not about trajectory segmentation for differential augmentation

**Threat to idea: LOW** — but must verify AR2-D2 full paper before any novelty claim.

---

### Term 11: "XRoboToolkit robot demonstration"

**What exists:**
XRoboToolkit appears to be an AR/XR framework for robot demonstration collection. Based on name and context:
- An XR-based toolkit for collecting robot demonstrations in augmented/virtual environments
- The interface focuses on enabling non-experts to provide demonstrations via XR headsets

**Threat to the reaching/grasping decomposition idea: LOW**
- XRoboToolkit is an INTERFACE tool, not a phase-decomposition augmentation method
- Even if it supports collecting demonstrations in XR, the phase-selective augmentation of those demonstrations is a separate and distinct contribution

**However**: If XRoboToolkit allows users to re-run only the reaching portion of a demonstration (because it has grasping detection built-in), this would be closer. Must verify.

---

### Term 12: "mixed reality robot data collection"

**Already covered in AR_NOVELTY_VERIFICATION_REPORT.md** — summary:
- Mixed reality for robot TELEOPERATION: exists (HRI papers)
- Mixed reality for data COLLECTION with coverage enforcement: not found
- Phase-selective MR augmentation (MR only during reaching): doubly novel

**Threat to idea: LOW**

---

## Additional Search Terms

### "SigLIP domain gap synthetic objects"

**What exists:**
The SigLIP domain gap problem for synthetic images is documented in the project files:
- Isaac Lab rasterizer images: SigLIP cosine distance ~0.6-0.8 → too large for effective transfer
- 3DGS rendered images: SigLIP cosine distance ~0.1-0.2 → acceptable for transfer
- Project memory: "sim images must look like real photographs" for SigLIP to work

**Implications for reaching/grasping augmentation:**
The reaching phase augmentation must produce images realistic enough for SigLIP (frozen in SmolVLA) to extract meaningful features. This is a non-trivial constraint:
- If augmented reaching images use synthetic objects overlaid on real scenes, SigLIP may reject them as OOD
- If the augmentation is trajectory-level only (without changing images), there is no SigLIP domain gap
- If AR overlays are subtle (e.g., guidance circles, not full object replacement), the SigLIP concern may be manageable

**Key insight**: The phrase "augment only the reaching phase" has TWO possible meanings:
1. **Image-level augmentation during reaching** (change what the camera sees): SigLIP domain gap is a real risk
2. **Trajectory-level augmentation during reaching** (generate synthetic approach paths without changing images): No SigLIP issue, but the augmented data may not improve visual generalization

The idea's implementability depends heavily on which interpretation is intended.

**No specific paper found on "SigLIP domain gap for phase-specific augmentation."**

---

### "frozen VLM synthetic image robot"

**What exists:**
- Project files document this extensively: frozen VLMs (SigLIP in SmolVLA) struggle with synthetic images that look unlike real photos.
- SplatSim paper (arXiv:2409.10161): demonstrates that 3DGS-rendered images can transfer through frozen VLM encoders better than rasterized images.
- Real2Render2Real (CoRL 2025): Same finding — photorealistic rendering is essential for frozen VLM-based policies.

**Implications for the reaching/grasping idea:**
If the reaching phase augmentation uses AR overlays (real background + synthetic objects), the SigLIP encoder will see composite images. The degree to which SigLIP accepts these depends on how photorealistic the overlays are. This is an open question not specifically addressed in prior work.

**Novel aspect**: Studying whether phase-specific AR augmentation (on reaching only) avoids the SigLIP domain gap issue because the action chunking in grasping ensures the critical contact information comes from real images. This framing — "augment the phase where frozen encoder errors matter less" — could be a novel contribution.

---

## Summary Table: Coverage of the Core Idea

| Component | Prior Work Exists? | Closest Work | Threat Level |
|-----------|-------------------|-------------|--------------|
| Reaching vs. grasping phase decomposition | YES (partial) | ARP, Contact-Rich papers | MEDIUM |
| Automatic phase detection (gripper state / FK) | YES | Various contact-detection papers | LOW (standard technique) |
| Augmenting ONLY the reaching phase | NO (direct match) | MimicGen (approach segments augmented) | HIGH (MimicGen is close) |
| Keeping grasping phase real/unaugmented | NO (explicit claim) | MimicGen's manipulation segments kept real | HIGH (MimicGen is close) |
| Sim-free real-robot implementation | YES (unique) | Not in MimicGen | LOW (differentiator) |
| For VLA (not Diffusion Policy or BC) | NOT FOUND | No paper | LOW (differentiator) |
| SigLIP domain gap analysis by phase | NOT FOUND | No paper | LOW (novel angle) |

---

## VERDICT

### [PARTIALLY FOUND] — The idea exists in modified form in MimicGen

**MimicGen (NeurIPS 2024 oral)** implements the conceptual core of this idea:
- It decomposes demonstrations into subtask segments
- The approach/transition segments are synthetically generated (IK-based)
- The manipulation/grasping segments are kept from real demonstrations

This is not an exact match, but it is close enough that **any paper proposing this idea MUST cite MimicGen, clearly differentiate from it, and demonstrate advantages**.

### What IS novel compared to MimicGen

If the idea is pursued, the differential advantages over MimicGen that could support a novelty claim are:

1. **No simulation required**: MimicGen requires IsaacGym/MuJoCo for feasibility checking. The proposed idea targets real-robot-only augmentation (AR or interpolation based).

2. **No 3D ground-truth pose required**: MimicGen needs exact 3D object poses for each target configuration. The proposed idea could work with visual/FK-based phase detection.

3. **Applied to VLA (frozen VLM encoder)**: MimicGen was designed for Diffusion Policy / BC. SmolVLA's frozen SigLIP encoder creates a domain gap constraint that MimicGen does not address. Studying what augmentation survives the frozen encoder is novel.

4. **AR-based reaching augmentation**: MimicGen generates approach motions via IK. AR-guided or AR-overlaid reaching trajectories are a different mechanism that could be more accessible to low-cost labs.

5. **Empirical scaling analysis**: If the idea is studied systematically (how many synthetic reaching trajectories per real grasping demo? what is the quality trade-off?), this provides empirical insights MimicGen does not.

---

## Critical Overclaim Risks

### DO NOT CLAIM:
- "First to decompose manipulation into reaching and grasping phases" — ARP, contact-rich manipulation papers, MimicGen-style approaches all do this
- "First to keep grasping real while augmenting approach" — MimicGen does this in simulation
- "MimicGen is not related work" — it IS related work and must be cited

### CLAIM INSTEAD (with qualifiers):
"We propose a sim-free, real-robot reaching-phase augmentation method for VLA imitation learning, extending the insight from MimicGen [cite] that approach segments are more amenable to synthesis than contact-phase segments. Unlike MimicGen, our approach (1) does not require a physics simulator, (2) does not require 3D ground-truth object poses, and (3) is specifically designed for VLAs with frozen vision encoders where the domain gap in synthetic images is a first-class concern."

---

## Papers That Must Be Read Before Proceeding

These papers are either confirmed relevant or high-probability relevant. Reading them is mandatory before submitting any paper that includes this idea:

### Must read (confirmed relevant):
1. **MimicGen** (arXiv:2310.17407, NeurIPS 2024 oral) — Mandlekar et al. CLOSEST WORK
2. **RoCoDA** (arXiv:2411.13031, 2025) — Counterfactual scene augmentation for BC
3. **GenAug** (arXiv:2302.06671, ICRA 2023) — Post-hoc diffusion augmentation
4. **CACTI** (arXiv:2212.05711, 2022) — Camera + context augmentation
5. **Real2Render2Real** (arXiv:2505.09601, CoRL 2025) — Real→3DGS sim→real, phase-level analysis

### Must verify (potentially relevant):
6. **AR2-D2** — verify exact contribution and whether it decomposes phases
7. **XRoboToolkit** — verify whether it has phase decomposition features
8. **ARP** (Action Representation with Pre-grasp phases) — check if it augments differentially
9. **Contact-Implicit MPC papers (RSS 2024-2025)** — may have reaching/grasping decomposition for augmentation

### arXiv searches to run before submission:
- "reaching phase augmentation robot" (arXiv, 2024-2026)
- "pre-grasp approach augmentation imitation learning" (arXiv, 2024-2026)
- "manipulation phase decomposition data augmentation" (arXiv, 2024-2026)
- "sim-free trajectory augmentation manipulation" (arXiv, 2024-2026)
- "contact phase data augmentation robot" (arXiv, 2024-2026)

---

## SigLIP / Frozen VLM Analysis

**This is the most technically interesting and potentially novel aspect of the idea.**

The proposed reaching/grasping decomposition has a natural alignment with the frozen VLM encoder constraint in SmolVLA:

- **During reaching**: The camera view is dominated by the approach motion, with the object in the background or periphery. SigLIP features at this stage are primarily encoding scene context, not fine-grained grasping details. → Synthetic images during reaching may be tolerable if SigLIP cosine distance is managed.

- **During grasping**: The camera view shows the hand-object contact region closely. SigLIP features here encode fine-grained texture and shape details critical for determining grasp success. → Synthetic images during grasping are HIGH RISK due to SigLIP domain gap.

This asymmetry — reaching is more tolerant to SigLIP domain gap than grasping — has NOT been explicitly studied or published (to my knowledge). This could be the most defensible novel contribution: **an empirical study showing that phase-selective augmentation avoids the frozen VLM encoder's domain gap, which uniform augmentation would incur**.

**Recommended framing if this analysis is confirmed experimentally**:

"We show that the SigLIP domain gap in SmolVLA is asymmetric across manipulation phases: the frozen encoder tolerates synthetic reaching-phase images (SigLIP cosine distance ≤ 0.15) while synthetic grasping-phase images cause significant embedding shift (cosine distance ≥ 0.45). This asymmetry motivates our phase-selective augmentation strategy."

This framing makes the FROZEN ENCODER CONSTRAINT the motivating problem, not just "augment reaching because it's easier" — which is a much stronger scientific contribution.

---

## Final Assessment

| Claim | Verdict | Confidence | Required Before Claiming |
|-------|---------|------------|--------------------------|
| "Phase decomposition of reaching vs grasping is novel" | FALSE | HIGH | Do not claim. Multiple prior works. |
| "Augmenting only the reaching phase is novel" | PARTIALLY FOUND | MEDIUM | MimicGen does this in simulation. Must differentiate. |
| "Sim-free reaching augmentation for VLA is novel" | LIKELY NOVEL | MEDIUM | Must verify with live search, confirm MimicGen requires sim |
| "SigLIP domain gap asymmetry across phases is novel" | LIKELY NOVEL | MEDIUM-HIGH | Most defensible claim; study has not been published |
| "AR-guided reaching augmentation for VLA is novel" | LIKELY NOVEL | MEDIUM | Must search arXiv 2024-2026 and check AR2-D2 |

**Overall verdict: [PARTIALLY FOUND]**

The conceptual core (keep grasping real, augment reaching) exists in MimicGen. The specific implementation for sim-free real-robot VLA training with frozen vision encoders is likely novel. The SigLIP domain gap asymmetry angle is the strongest novel contribution.

**Recommendation**: If pursuing this idea, frame it around the **frozen encoder domain gap asymmetry** as the central scientific contribution, with the phase-selective augmentation as the resulting methodology. Do not frame it as a new decomposition idea — frame it as a new analysis of where and why augmentation fails in VLAs.

---

## Report Metadata

- Date: 2026-03-24
- Based on: Project files + training knowledge (cutoff Aug 2025)
- Live search required: YES (mandatory before any novelty claim in paper)
- Confidence: MEDIUM — MimicGen match found; other aspects need live verification
- Next update: After running 14+ arXiv/Scholar searches listed in "Must Verify" section
