# Comprehensive Landscape Survey: AR / Projector / Laser x Robot Manipulation x VLA/Foundation Models

**Agent: C3 (Paper Writing & Positioning Specialist)**
**Date: 2026-03-24**
**Protocol: CLAUDE.md Research Verification Rules — 10+ search terms per category, refutation search first**
**Knowledge cutoff: August 2025. Papers after this date require live search.**
**Purpose: Exhaustive novelty check before any AR/projector-related claim in CoRL 2026 paper**

---

## IMPORTANT DISCLAIMER

This report is based entirely on training knowledge (cutoff August 2025). I cannot perform live internet searches.
Every "not found" statement means "not found in training corpus." Pre-submission, all sections marked [VERIFY]
must be checked against arXiv, Semantic Scholar, and conference proceedings (2023-2026).

---

## CRITICAL CONCEPT DISTINCTION (repeated from AR_NOVELTY_VERIFICATION_REPORT.md)

**Concept A (our potential claim)**: AR overlay shown to human demonstrator DURING data collection to guide
WHERE objects are placed → workspace coverage enforcement. AR is NOT in training images. Action labels valid.

**Concept B (existing literature)**: Post-hoc alteration of RECORDED demo images. GenAug, Rosie, CACTI,
RoboSplat, RoCoDA are all Concept B. Training images are augmented, but this causes action-label misalignment.

These are fundamentally different. Almost all literature is Concept B. Concept A appears sparse.

---

## CATEGORY A: Projector-Guided Robotics

### A1. Spatial Augmented Reality (SAR) — Foundational

| Title | Venue/Year | What it does | VLA? | Hardware | Gap / Threat |
|-------|-----------|--------------|------|---------|-------------|
| "Shader Lamps: Animating Real Objects With Image-Based Illumination" (Raskar et al.) | EGWR 2001 | Projects textures onto real objects — foundational SAR | No | Overhead projector | Historical context only. 25-year-old field. VERY LOW threat. |
| Industrial SAR assembly guidance (Boeing, Airbus style) | ISMAR 2010s | Projector-guided assembly for aircraft parts | No | Industrial projector + CAD model | Industrial only; no learning. VERY LOW. |
| "Projection Mapping for Interactive Environments" | SIGGRAPH/CHI various | Dynamic projection onto non-planar surfaces | No | Projector + depth sensor | Art/HCI; not robot learning. VERY LOW. |

**SAR field assessment**: Mature (25+ years), but almost entirely for human assembly guidance and artistic/HCI applications. No canonical paper uses projectors to guide robot ML training data collection.

### A2. GreenAug — CLOSEST KNOWN PAPER to Concept A

**Title**: "GreenAug: Green Screen Augmentation Enables Scene Randomisation for Simulated and Real Robot Training"
**Venue**: CoRL 2024
**arXiv ID**: [VERIFY — not confirmed in training data]
**What it does**: Places robot in front of a physical green screen backdrop during data collection. At training time, composites random real-world background images behind the robot. Policy trained on this background-augmented data.
**VLA?**: Evaluated with behavioral cloning and diffusion-based policies. Not specifically a VLA paper.
**Hardware**: Robot arm + physical green screen cloth (~$30-50 setup).
**Key limitation**: Only BACKGROUND diversity, not foreground/object placement diversity. Green screen is passive (no interactive feedback to human demonstrator). Does not address spatial workspace coverage.
**Threat to Concept A**: LOW-MEDIUM.

**Critical differentiation for paper**:
"GreenAug provides passive collection-time background substitution. Our approach uses real-time AR feedback to
actively guide WHERE the demonstrator places objects, enforcing spatial workspace coverage — targeting
distribution coverage rather than appearance diversity. GreenAug is a passive backdrop choice; our approach is
an active, interactive guidance mechanism."

### A3. SAR for Robot Workspace (HRI-adjacent)

| Title | Venue/Year | What it does | VLA? | Notes |
|-------|-----------|--------------|------|-------|
| "Robot State Legibility via Projection" | HRI 2013-2020 various | Project robot's intended path onto floor for human legibility | No | Ceiling projector; not for learning |
| "Projected AR for Human-Robot Collaboration" | HRI, ROMAN | Project task info onto workspace surface | No | Different goal: HRC communication |
| "LightGuide" (Sodhi et al., ~2012) | — | Wrist-mounted projector for hand motion guidance | No | Wrist projector; MEDIUM risk if adapted for robot demos. [VERIFY] |
| "Workspace projection for assembly guidance" | IROS/ICRA ~2015-2020 | Project next-step instructions onto workspace | Some use language models in late versions | Different goal: step-by-step assembly, not spatial coverage for ML |

---

## CATEGORY B: Laser + Robot

### B1. Laser Pointer for Task Specification

| Title | Venue/Year | What it does | VLA? | Threat |
|-------|-----------|--------------|------|--------|
| "Laser Pointer Interaction for Robot Control" | IROS/ICRA 2010-2020 various | Human points laser at target; robot moves to detected laser dot | No (pre-VLA era) | VERY LOW |
| "Natural Human-Robot Interaction Using Gesture and Laser" | HRI 2015-2020 | Gesture + laser for robot task specification | No | VERY LOW |
| "Robot Instruction via Laser Pointer and Speech" | HRI various | Multimodal instruction: laser + speech | No | VERY LOW |

### B2. Structured Light for Robot Perception

Structured light (including Azure Kinect NFOV depth) is used everywhere as a SENSING tool, not as a manipulation LEARNING guidance tool. Not directly relevant to our claim.

| Area | Relevance |
|------|---------|
| Structured light depth sensing | LOW — perception tool |
| Laser stripe scanning (industrial inspection) | VERY LOW — quality control |
| Time-of-flight for grasping | LOW — input modality, not training guidance |

### B3. Industrial Laser Projection Systems

| Company/Paper | What it does | VLA? | Relevance |
|--------------|--------------|------|---------|
| LAP Laser (Germany) | Laser projectors for assembly templates | No | Industrial, $10K-100K, no ML |
| Virtek Vision | Laser templating for aerospace | No | Industrial only |
| "Laser Guidance for Human-Robot Assembly" | IROS/CASE | Project laser assembly guides | No VLA | Industrial |

**Finding**: No papers found using consumer laser projectors for VLA fine-tuning or robot ML data collection.
Laser pointer + robot = pre-2020 HRI literature, no foundation model involvement.

---

## CATEGORY C: AR + Robot Learning (Beyond Teleoperation)

### C1. AR for Robot Policy Learning — Direct Hits

#### AR2-D2 — HIGHEST RISK UNKNOWN

**What we know from project memory**: "AR2-D2는 다른 각도" (different angle). Someone previously searched and
concluded it is a different angle. But given the 2026-03-10 incident (4/5 false gaps), this CANNOT be accepted
without verification.

**Possible interpretations**:
1. AR2-D2 = social robot with AR display face → NOT a threat (appearance/sociability research)
2. AR2-D2 = AR for robot data demonstration collection → DIRECT THREAT to Concept A
3. AR2-D2 = AR for robot task specification → LOW threat (different goal)

**arXiv ID**: NOT CONFIRMED in training data.

**MANDATORY ACTION #1**: Search "AR2-D2 robot" on arXiv + Semantic Scholar. Read abstract. Determine
exact contribution. Do not claim novelty until verified.

#### Other C1 Papers

| Title | Venue/Year | What it does | VLA? | Threat |
|-------|-----------|--------------|------|--------|
| "Robot Learning from Demonstration via AR Guidance" (general category) | HRI 2020-2024 various | AR interface showing HOW to demonstrate | No (pre-VLA) | LOW: UI focus, not distribution coverage |
| "Mixed Reality Robot Learning" | HRI, ROMAN various | MR headsets for robot programming | Some NLP | LOW: programming interface, not ML training data |
| "MOSAIC: MR for Robot Skill Acquisition" | ~2023-2024 | Possible AR for skill transfer | Unknown | MEDIUM — [VERIFY] |

### C2. AR for Data Augmentation — Post-hoc (Concept B, NOT a threat to Concept A)

All established, well-studied. Full analysis in AR_NOVELTY_VERIFICATION_REPORT.md:

| Paper | arXiv ID | Threat to Concept A |
|-------|---------|-------------------|
| GenAug (ICRA 2023) | 2302.06671 | LOW |
| Rosie (2023) | 2309.11386 | LOW |
| CACTI (2022) | 2212.05711 | LOW |
| RoboSplat (RSS 2025) | 2504.13175 | LOW |
| RoCoDA (2025) | 2411.13031 | LOW |
| TGM-VLA (2026) | 2603.00615 | LOW |

**All operate on RECORDED data. None guide human demonstrators in real-time. All are Concept B.**

### C3. AR for Robot Task Specification (VLA era)

| Title | arXiv ID | VLA? | Threat |
|-------|---------|------|--------|
| RoboPoint (Yuan et al., 2024) | 2406.10721 | Yes (VLM) | LOW: predicts affordance points visually, no physical AR hardware |
| SpatialVLM (Chen et al., ECCV 2024) | 2406.13537 | Yes (VLM) | LOW: spatial reasoning VLM, no projector/AR hardware |
| "SayPlan" / spatial projection task planning | ~2023-2024 | LLM/VLM for planning | LOW: planning, not data collection |

### C4. Holographic / HoloLens Robot Research

| Area | Papers exist? | VLA involvement? | Threat |
|------|--------------|-----------------|--------|
| HoloLens for robot programming | YES (HRI 2018-2024) | No | VERY LOW |
| Holographic shared autonomy | YES (HRI 2021-2023) | No | VERY LOW |
| MR for robot teleoperation | YES (HRI, IROS) | No VLA | VERY LOW |
| AR for robot intent visualization | YES (HRI 2019-2024) | No | VERY LOW |

None of these focus on training data distribution or VLA fine-tuning.

---

## CATEGORY D: Projection + AI / SAR + ML

### D1. Projection Mapping + Machine Learning

| Title | Venue/Year | What it does | VLA? | Relevance |
|-------|-----------|--------------|------|---------|
| DeProCams / Neural Projector-Camera Systems | CVPR/ECCV 2020-2024 | DL for projector geometric/color compensation | No | ML FOR projectors (not ML WITH projectors for robot learning) |
| ProCams field (general) | CVPR, ICCV | ML for projector appearance correction | No | VERY LOW |
| "Projection mapping + interactive ML" | CHI 2022-2024 | Interactive ML interfaces via projection | No robot | VERY LOW |

**Finding**: ML + projector in the computer vision literature = improving projector visual output quality.
NOT using projectors to guide robot manipulation learning. Completely different domain.

### D2. SAR + Machine Learning — Notable Gap

No canonical paper found that:
1. Uses projectors to project guidance/target information onto a real robot workspace
2. Uses this projected information to improve robot manipulation learning
3. Involves VLA or foundation models in any way

This gap appears real but must be confirmed with live search (see verification checklist below).

---

## CATEGORY E: Conference-Specific Analysis

### CoRL 2024
- **GreenAug** (confirmed): collection-time green screen, post-training augmentation. NEAREST paper.
- No AR/projector paper targeting VLA fine-tuning found.

### CoRL 2025 (up to Aug 2025 knowledge)
- **Real2Render2Real** (2505.09601): Real → Gaussian Splatting → sim. No physical AR.
- **RoboSplat** (RSS 2025): 3DGS novel view augmentation. Post-hoc, no AR hardware.
- **AirExo-2**: low-cost data collection hardware. No AR/projector.
- [VERIFY]: Full CoRL 2025 accepted papers list.

### ICRA 2025
- AR for robot teleoperation: some papers exist, no VLA training focus.
- Structured light for grasping: perception papers only.
- [VERIFY]: Full ICRA 2025 proceedings.

### HRI 2025 — HIGHEST RISK CONFERENCE
HRI is where AR + robot papers are most likely published. From training knowledge:
- AR for robot programming by demo: YES — multiple papers. NO VLA involvement.
- AR visualization of robot state: YES — multiple papers.
- AR-guided data collection for ML: NOT FOUND in training data.
- [VERIFY]: **MUST CHECK HRI 2025 proceedings** before any Concept A claim.

### RSS 2025
- RoboSplat: post-hoc augmentation, no AR hardware.
- No AR/projector + VLA paper found.

### IROS 2024/2025
- Laser pointer for task specification: older literature exists (no VLA).
- AR for assembly: industrial focus.
- [VERIFY]: IROS 2024 + 2025 proceedings.

### CHI/ISMAR 2025
- Projector + ML for HCI: exists but no robotics manipulation focus.
- [VERIFY]: ISMAR 2025 (AR conference — most likely venue for AR + robot interaction papers).

---

## COMPREHENSIVE PAPER TABLE (All Found)

### Group 1: Confirmed relevant, well-characterized

| # | Title | Venue/Year | arXiv ID | VLA? | AR/Proj type | Threat |
|---|-------|-----------|---------|------|------------|--------|
| 1 | Domain Randomization (Tobin et al.) | IROS 2017 | 1703.06907 | No | Sim randomization | LOW |
| 2 | GenAug | ICRA 2023 | 2302.06671 | BC | Post-hoc diffusion | LOW |
| 3 | Rosie | 2023 | 2309.11386 | BC | Post-hoc diffusion | LOW |
| 4 | CACTI | 2022 | 2212.05711 | BC | Post-hoc camera+context | LOW |
| 5 | GreenAug | CoRL 2024 | [VERIFY] | BC/diffusion | Collection-time green screen | LOW-MEDIUM |
| 6 | RoboSplat | RSS 2025 | 2504.13175 | VLA-adjacent | Post-hoc 3DGS | LOW |
| 7 | RoCoDA | 2025 | 2411.13031 | VLA-adjacent | Post-hoc counterfactual | LOW |
| 8 | Real2Render2Real | CoRL 2025 | 2505.09601 | VLA-adjacent | Real→sim→real | LOW |
| 9 | SOAR | CoRL 2024 | 2404.11617 | VLA | No AR (autonomous practice) | LOW-MEDIUM |
| 10 | RoboPoint | 2024 | 2406.10721 | Yes (VLM) | No physical AR | LOW |
| 11 | SpatialVLM | ECCV 2024 | 2406.13537 | Yes (VLM) | No physical AR | LOW |
| 12 | UMI | RSS 2024 | — | BC/diffusion | No AR (handheld hardware) | LOW |
| 13 | DAgger (Ross et al.) | AISTATS 2011 | — | No (IL) | No AR (interactive demos) | LOW |

### Group 2: Partially relevant, need verification

| # | Title | Venue/Year | arXiv ID | VLA? | AR/Proj type | Threat |
|---|-------|-----------|---------|------|------------|--------|
| 14 | **AR2-D2** | UNKNOWN | NOT FOUND | UNKNOWN | UNKNOWN | **HIGH — MUST VERIFY IMMEDIATELY** |
| 15 | AR for Robot PbD (general category) | HRI 2020-2024 | Various | No | HoloLens AR | LOW |
| 16 | MR robot teleoperation | HRI, ROMAN | Various | No | HoloLens MR | LOW |
| 17 | "MOSAIC MR for robot skills" | ~2023-2024 | NOT CONFIRMED | Unknown | MR headset | MEDIUM — [VERIFY] |
| 18 | LightGuide (wrist projector) | ~2012 | NOT FOUND | No | Wrist projector | LOW-MEDIUM — [VERIFY] |
| 19 | "Projected affordance robot" (HRI category) | HRI, IROS | Various | Some VLMs | Short-range projector | LOW-MEDIUM |

### Group 3: Tangentially related (cite as background context)

| # | Title | Connection | Threat |
|---|-------|-----------|--------|
| 20 | Shader Lamps SAR (Raskar 2001) | Foundational SAR paper | VERY LOW |
| 21 | Industrial SAR (Boeing, Airbus style) | Mature industrial application | VERY LOW |
| 22 | Neural ProCams (CVPR 2020-2024) | ML for projector systems (not robot learning) | VERY LOW |
| 23 | Robot state legibility via projection | HRI context | VERY LOW |
| 24 | Active learning for robot demonstrations | Spirit: coverage enforcement (no AR) | LOW |
| 25 | Inner Monologue, ProgPrompt, TidyBot | LLM-based planning (no AR) | LOW |

---

## STRUCTURAL ANALYSIS: Why VLA + AR is Sparse

The intersection is sparse for structural reasons that should be understood:

1. **Frozen visual encoder problem**: VLAs like SmolVLA use SigLIP (frozen at training time).
   AR overlays visible IN training images = distribution shift = model confusion at deployment.
   Example: if a target circle is projected onto the table during collection (visible in camera images),
   the model learns to expect that circle — which won't be there at deployment.

2. **Benchmark non-compatibility**: VLA papers are evaluated on LIBERO, CALVIN, BridgeData. None use
   AR hardware. AR papers would be incomparable to these benchmarks.

3. **Hardware complexity**: Standard VLA papers use RGB cameras (RealSense, webcam). AR requires
   projector or HoloLens — additional hardware that reduces reproducibility.

4. **Train-test gap**: AR guidance visible in training images creates a systematic gap.

**Concept A elegantly avoids ALL of these problems**:
- AR is shown to the human on a separate monitor/display, NOT projected into the physical scene
- Robot camera sees normal real-world images (no AR overlay visible)
- Training images are clean (no AR artifacts)
- Deployment images match training images
- Compatible with any VLA and any frozen visual encoder

This design distinction MUST be stated explicitly in the paper. It explains why Concept A is VLA-compatible.

---

## MANDATORY PRE-SUBMISSION VERIFICATION CHECKLIST

### Priority 1 — BLOCKING (cannot submit without these)

- [ ] **arXiv**: "AR2-D2 robot" — title, arXiv ID, exact contribution, date
- [ ] **arXiv**: "augmented reality demonstration collection robot learning" (2023-2026)
- [ ] **arXiv**: "mixed reality data collection robot imitation learning" (2023-2026)
- [ ] **HRI 2025 proceedings**: any paper on AR + data collection for robot ML

### Priority 2 — HIGH importance (do before writing related work section)

- [ ] **arXiv**: "AR workspace coverage robot" (2023-2026)
- [ ] **IROS 2024 proceedings**: AR + robot manipulation data collection
- [ ] **CoRL 2025 full proceedings**: AR guidance for robot learning
- [ ] **ICRA 2025 proceedings**: AR + robot demonstration collection
- [ ] **Semantic Scholar**: "mixed reality robot learning data" (filter: 2023-2026)

### Priority 3 — Complete the picture

- [ ] **arXiv**: "projection guided robot learning" (2022-2026)
- [ ] **arXiv**: "projector robot workspace annotation" (2022-2026)
- [ ] **Google Scholar**: "spatial augmented reality imitation learning" (2022-2026)
- [ ] **ISMAR 2024/2025 proceedings**: AR + robot interaction
- [ ] **CHI 2025**: projection mapping + ML + robot
- [ ] **ROMAN 2024/2025**: AR for robot demonstration collection

### Priority 4 — Lower risk, confirms completeness

- [ ] **arXiv**: "laser pointer robot task learning foundation model"
- [ ] **arXiv**: "GreenAug VLA" or "green screen robot VLA"
- [ ] **IEEE Xplore IROS/ICRA 2025**: "projector robot learning"
- [ ] **SIGGRAPH 2024/2025**: "projector mapping robot"
- [ ] **LightGuide** exact title and arXiv ID verification
- [ ] **MOSAIC MR** exact title verification

---

## RISK MAP

| Risk | Severity | Description | Mitigation |
|------|----------|-------------|-----------|
| AR2-D2 does exactly Concept A | HIGH | Unknown paper, high relevance if AR + data collection | Verify BEFORE claiming anything |
| HRI 2025 has AR data collection paper | MEDIUM | HRI is prime venue for this topic | Check proceedings |
| GreenAug reviewer extension argument | MEDIUM | "GreenAug + AR overlay is incremental" | Differentiate: spatial coverage vs. appearance diversity |
| MOSAIC or similar MR paper | MEDIUM-LOW | Unknown paper status | Verify title |
| IROS/ICRA 2025 workshop paper | LOW-MEDIUM | Workshop papers may have informal AR+collection work | Check workshop proceedings |
| GenAug/CACTI reviewer argument | LOW | "Obvious extension" argument | Defend via action-label correctness + coverage enforcement distinction |

---

## POSITIONING RECOMMENDATIONS

### If AR2-D2 and HRI 2025 checks are CLEAR:

**Claimable statement (with "to our knowledge" qualifier)**:
> "To our knowledge, we are the first to use real-time AR overlay guidance during robot demonstration
> collection to enforce spatial workspace coverage, distinct from post-hoc visual augmentation approaches
> (GenAug, Rosie, CACTI, RoboSplat) in two key respects: (1) our approach modifies human demonstrator
> behavior rather than recorded images, and (2) action labels are correct by construction since the human
> always demonstrates in the real physical environment."

### Paper comparison structure:

**vs. GenAug/Rosie/CACTI/RoboSplat** (post-hoc Concept B):
"Prior work augments the visual appearance of recorded demonstrations. We instead guide the spatial
distribution of demonstrations during collection itself. This distinction matters: post-hoc appearance
augmentation can introduce action-label misalignment when object positions change in augmented images,
a problem our collection-time guidance avoids by construction."

**vs. GreenAug** (closest prior work):
"GreenAug makes a passive collection-time environmental design choice (green screen backdrop for
background substitution). Our approach uses interactive, real-time AR feedback to guide WHERE objects
are placed — targeting spatial distribution coverage rather than background appearance diversity."

**vs. SOAR** (coverage enforcement, different method):
"SOAR autonomously practices in states where the policy underperforms, requiring an existing base policy.
Our approach guides initial data collection from scratch, before any policy has been trained."

**vs. Simulation domain randomization**:
"Standard domain randomization diversifies data in simulation. We achieve comparable spatial diversity
in the real world without any simulator, avoiding the sim-to-real gap entirely."

### If AR2-D2 does Concept A (fallback):

1. Differentiate on VLA-specific OOD embodiment: "We specifically address SmolVLA's frozen SigLIP encoder"
2. Differentiate on scaling characterization: "We quantify HOW MUCH coverage guidance improves success rates"
3. Differentiate on consumer hardware: "$130 arm + RTX 4090 consumer GPU"
4. Last resort: Drop AR contribution, focus on data quality + scaling laws (independent value)

---

## FINAL ASSESSMENT TABLE

| Claim | Verdict | Confidence | Required Action |
|-------|---------|------------|-----------------|
| "Post-hoc visual augmentation is novel" | FALSE | HIGH | Do not claim |
| "Real-time AR guidance during demo collection is novel" | LIKELY TRUE | MEDIUM | Must verify AR2-D2 + HRI 2025 |
| "Workspace coverage enforcement via AR is novel for VLA" | LIKELY TRUE | MEDIUM | Same as above |
| "AR is novel for robot interaction" | FALSE | HIGH | Do not claim |
| "Projector for VLA fine-tuning is novel" | LIKELY TRUE | MEDIUM-HIGH | Verify MOSAIC, conference proceedings |
| "Laser + VLA combination is novel" | LIKELY TRUE | HIGH | Low practical value; laser not in our system |
| "GreenAug already solves our problem" | FALSE | HIGH | Clear differentiation: spatial vs. appearance |

**Overall**: The specific claim — using AR guidance at collection time for spatial workspace coverage, applied to VLA fine-tuning on OOD embodiment — appears to be in the LIKELY NOVEL category with MEDIUM confidence.
Confidence can be raised to HIGH only after live verification of AR2-D2 and HRI 2025 proceedings.

---

## UPDATE LOG

- 2026-03-24 (C3): Initial report created. 25 papers catalogued across 5 categories. Knowledge cutoff Aug 2025.
  All findings pending live search verification.
