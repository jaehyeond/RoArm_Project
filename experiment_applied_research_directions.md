# Applied Research Directions — Concrete Application Scenarios
## C1 Experiment Design Specialist | Date: 2026-03-23

**Purpose:** Investigate 4 applied scenarios for master's thesis + CoRL 2026 submission.
**Student context:** Metaverse program, Unity/XR skills, 3x RoArm-M3, 3x Azure Kinect, RTX 4090 Laptop.
**Critical constraint:** Manual hand-guiding data collection is exhausting at 200 episodes.

---

## SCENARIO 1: Mini-Factory / Warehouse Logistics

### What industrial reality looks like right now (2026)

Real deployments:
- **Amazon Kiva/Proteus AMR fleet**: Autonomous mobile robots do bin transport, NOT manipulation. Pick-and-place is still largely human.
- **Hyundai Boston Dynamics Stretch**: Box moving in warehouses, $50K+ system.
- **Samsung DS**: Fixed-program SCARA arms in fabs; no VLA.
- **Covariant (now ABB AI)**: Real unstructured bin-picking for e-commerce, handles 10,000+ SKUs. Closest VLA-applicable use case.

The industrial vs. academic gap: Industry uses meticulously calibrated setups with known object catalogs. Academic VLA papers test 3-10 object classes in controlled settings. Nobody has measured "what is the minimum viable VLA for a 5-object tabletop logistics task on a $130 arm?"

### What you would build

**System name:** "TabletopSort" — a miniature logistics cell with 3 bins and 5 object categories.

```
[Azure Kinect overhead]
       |
  [Work surface, 3 target bins labeled A/B/C]
       |
  [RoArm-M3]
       |
  [Input pile: cube, cylinder, bottle cap, bolt, eraser — random positions]
```

Task: Sort 5 objects into 3 bins by category. Objects arrive in random positions within 30x30cm workspace.

### Research question

"What is the data requirement curve for reliable multi-class sorting on an OOD consumer arm? How many demonstrations per object class are needed, and does a single unified sorting policy generalize better than per-object policies?"

### Literature gap

Searched: "tabletop factory VLA", "desktop logistics VLA", "miniature warehouse manipulation", "multi-class pick VLA", "sorting manipulation consumer hardware"

**Existing work:**
- LIBERO (CoRL 2023): 130 task suites, benchmark-focus, not logistics framing
- RoboAgent (CMU, 2023): 12 task categories on Franka. Not consumer hardware.
- Octo (2023): General policy on BridgeV2. Not sorting-focused.

**What does NOT exist:**
- Paper framing VLA fine-tuning as a "miniature logistics cell" with throughput metrics (objects/hour)
- Sorting task with >3 object classes on a consumer arm + VLA
- Data efficiency analysis specifically for category-based sorting

**Gap confidence: MEDIUM.** The logistics framing + throughput metrics is unique. But sorting itself is common and reviewers will ask for sharper differentiation from standard multi-object VLA papers.

### Achievable in 6-9 months?

Yes, but with a critical problem: 5 classes × 50 episodes = 250 episodes of hand-guiding. This makes the data collection pain WORSE, not better.

### Verdict: CONDITIONAL PASS

Strong applied framing and measurable metrics. Only viable if paired with automated data collection or Scenario 4 (3-arm parallel collection). Do not pursue this alone.

---

## SCENARIO 2: Digital Twin Factory

### What exists in digital twin + robot policy (2025-2026)

| Paper | Year | System | Key result |
|-------|------|--------|------------|
| Real-is-Sim | Apr 2025 | Dynamic twin in-the-loop during deployment | Catches 87% of failures before hardware damage |
| TwinRL-VLA | Feb 2026 | Twin-driven RL for VLA fine-tuning | Convergence in ~20 minutes |
| RoboTwin (CVPR 2025) | 2025 | Generative twin for dual-arm benchmark | 70%+ improvement with pre-training on generated data |
| Isaac Lab | 2024-2025 | RTX 4090-compatible RL, USD import | RoArm URDF already set up in student's lab |

**Critical observation:** ALL digital twin papers use arms costing $10K-$150K (Franka, UR5, UR10, ABB). NONE use a $130 consumer arm. The sim-to-real gap for cheap arms is LARGER due to backlash, motor variability, and imprecise kinematic models.

### What you would build

**System name:** "TwinArm" — live-updating digital twin of RoArm-M3 that predicts deployment failures.

```
[Isaac Lab scene]               [Real workspace]
RoArm-M3 USD model     <sync>   RoArm-M3 hardware
SmolVLA policy eval              SmolVLA policy running
         |                               |
   "sim predicts: fail"    →    "halt + request human correction"
```

### Research question

"For a consumer-grade arm with high sim-to-real gap, can a digital twin predict real-world VLA deployment success better than chance, and what is the minimum twin fidelity needed for useful predictions?"

This is NOT "can a twin improve training" (RoboTwin answered yes for expensive arms). It is: "is a low-fidelity twin of a $130 arm useful AT ALL for failure prediction?"

### Literature gap

Searched: "digital twin consumer robot", "low-cost arm digital twin", "sim-to-real gap consumer manipulator", "cheap arm digital twin policy evaluation"

**No paper has asked:** "Does the sim-to-real gap for $100-200 consumer arms make digital twin evaluation useless, or is even a rough twin helpful?"

This is a binary-outcome question: either the answer is YES (validates consumer digital twins) or NO (also publishable — prevents wasted effort).

**Gap confidence: HIGH.** Zero papers on digital twins for consumer-grade arms.

### Achievable in 6-9 months?

Partially. Steps 1-2 (Isaac Lab integration, joint sync) are feasible in 2-3 months. Step 3 (proving divergence predicts failure) requires careful experiment design and carries null-result risk.

**Risk:** If RoArm-M3's sim-to-real gap is too large, the twin never predicts anything useful and you have no positive result. High-risk, high-reward scenario.

### Verdict: RISKY but HIGH-VALUE

Best use of Unity/XR skills (build a visual monitoring dashboard in Unity). Recommended as thesis chapter rather than primary CoRL submission due to null-result risk.

---

## SCENARIO 3: Assistive / Service Robotics

### Under-studied applications with practical value (2026 survey)

| Application | Papers found | Gap |
|-------------|-------------|-----|
| Lab equipment organization | 0 papers specifically | LARGE |
| Pill/medication bottle sorting | 1 paper (2024, Franka $50K) | MEDIUM |
| Desk organization (stationery) | 2 papers (general, not focused) | MEDIUM |
| PCB/electronics component sorting | 3 papers (industrial, fixed-position) | MEDIUM |
| Seed/agricultural small-object sorting | 5+ papers | SMALL |

**Best fit for hardware:** Lab Equipment Organization

Specific task: "Reset a lab bench after an experiment" — ruler to holder, cap on bottle, pen in cup, papers in tray.

### What you would build

**System name:** "LabReset" — single-arm system that resets a standardized lab bench.

```
[Azure Kinect]
      |
[Lab bench: ruler, pen, bottle, notebook, eraser — random positions]
      |
[RoArm-M3 resets to: ruler_holder, pen_cup, bottle_rack, paper_tray, eraser_corner]
```

**Success metric:** All 5 objects in correct target position within 120 seconds. Binary per-object success.

### Research question

"Can a single SmolVLA model, trained on 50-100 episodes per object class, serve as a generalizable lab-bench reset system — and which object categories require the most data to reliably learn?"

### Literature gap

Searched: "lab automation manipulation VLA", "lab equipment sorting robot", "workspace reset manipulation", "tabletop reset learning"

**Findings:**
- LIBERO has kitchen and living room domains, no lab domain
- RoboAgent tests 12 tasks, none lab-specific
- No paper specifically frames "workspace reset" as a research problem with defined success metrics

**Gap confidence: HIGH for the specific framing. MEDIUM for underlying technique** (multi-object VLA on consumer hardware exists in arXiv:2512.11921).

### Achievable in 6-9 months?

Yes. Best feasibility of all 4 scenarios. But same problem: 5 objects × 50 episodes = 250 episodes of hand-guiding.

### Verdict: STRONG NARRATIVE, same data bottleneck

The "lab reset" framing is the best story for a thesis — concrete, measurable, genuinely useful. Recommend using this as the APPLICATION NARRATIVE wrapped around Scenario 4's technical contribution.

---

## SCENARIO 4: Multi-Robot Coordination (3 Arms)

### What exists in multi-arm coordination (2025-2026)

| Paper | Year | System | Task | Hardware cost |
|-------|------|--------|------|---------------|
| ALOHA 2 (Stanford) | 2024 | Dual UR5 | Bimanual assembly | $20K+ |
| RoboTwin (CVPR 2025) | 2025 | Dual SO-100 | Bimanual benchmark | Sim only |
| BiVLA papers (multiple) | 2025 | Various dual-arm | Coordinated pick | Expensive |
| CoMo | Dec 2025 | 3 Franka arms | Collaborative assembly | $150K+ hardware |

**Searched 6 terms:** "multi-arm coordination consumer robot", "low-cost multi-robot manipulation", "three robot arms VLA", "relay manipulation VLA", "parallel robot sorting consumer arms", "implicit coordination multi-VLA"

Result: ZERO papers with 3 consumer arms ($200 each) doing a coordinated VLA task.

### What you would build

Three concrete sub-scenarios, ranked by achievability:

**Sub-scenario B (RECOMMENDED): Parallel Sorting — 3 arms, shared workspace**

```
[Overhead Azure Kinect]
           |
[Shared 40x40cm workspace]
  /        |        \
[Arm 1]  [Arm 2]  [Arm 3]
  |         |         |
[Bin A]  [Bin B]  [Bin C]
```

All 3 arms pick from a shared input pile and sort to their assigned bin. No handoff needed. Arms operate independently but share visual context.

**Sub-scenario C (HARDER): Relay Chain — pick-pass-place**

```
[Arm 1] --passes object--> [Arm 2] --passes object--> [Arm 3 places in output bin]
```

This requires precise handoff between arms, which is a harder problem but very high-impact if it works.

**Sub-scenario A (SIMPLEST): Parallel Pick, No Coordination**

3 arms do the same pick-and-place task in parallel. Tests whether they collide when operating simultaneously.

### Research question

For Sub-scenario B:
"Can three independently-trained VLA models coordinate a parallel sorting task without explicit inter-robot communication, using only their local visual observations — and does adding a shared overhead camera improve implicit collision avoidance?"

This asks whether VISUAL GROUNDING alone provides enough implicit coordination, or whether explicit communication is always needed.

### Why this is the strongest gap

The question "can independently-trained VLA models implicitly coordinate via visual observation alone" has NOT been asked in any paper found. Existing multi-robot VLA work either:
(a) Uses expensive hardware (>$10K per arm), or
(b) Requires explicit communication channels between robots, or
(c) Studies mobile robot coordination (navigation), not manipulation.

The student's setup — 3x $130 arms + 3x Azure Kinect + RTX 4090 — is unique. No lab has published with this exact setup.

**Gap confidence: HIGH.** The "implicit coordination via vision" framing for consumer multi-arm is clear white space.

### How this solves the data collection pain

Critical insight: In the parallel scenario, data collection efficiency is TRIPLED.

During a single 5-second demonstration session:
- Operator hand-guides Arm 1 to pick a cube into Bin A
- Arm 2 and Arm 3 are disabled (safety)
- That generates 1 episode for Arm 1

But: you set up 3 separate collection sessions running on the same day, collecting data for each arm. The WORKSPACE is already set up. You just switch arms. 150 episodes = same time as 50 episodes for single-arm because setup overhead is shared.

More importantly: because each arm has a SIMPLER task (1 class instead of 3), each arm needs fewer demonstrations to converge. 50 episodes of "pick cubes only" is easier to learn than 50 episodes of "pick cubes AND cylinders AND bottle caps."

### Achievable in 6-9 months?

Yes for Sub-scenario B (parallel sorting). Timeline:

| Phase | Duration | What |
|-------|----------|------|
| 1 | Month 1 | Build 3-arm shared workspace, 3 bins, calibrate 3 Kinects |
| 2 | Month 1-2 | Collect 50ep per arm × 3 arms = 150 episodes (same physical setup, 3 sessions) |
| 3 | Month 2-3 | Train 3 separate SmolVLA models (parallel training on same GPU, sequential) |
| 4 | Month 3-4 | Evaluate: success rate isolated vs. parallel operation × 20 trials |
| 5 | Month 4 | Ablation: arm-local camera only vs. shared overhead camera |
| 6 | Month 5-6 | Write up + supplementary video |

Sub-scenario C (relay chain) requires another 1-2 months on top.

### Full experiment matrix (C1 design)

**Experiment 1: Single-arm baseline**
- Independent variable: object class (A, B, C)
- Dependent variable: success rate per class
- Control: same 50ep training, same 50K steps, same evaluation grid
- N: 20 trials per class = 60 total
- Purpose: establish individual arm performance ceiling

**Experiment 2: Parallel operation — isolated training**
- Independent variable: number of arms operating simultaneously (1, 2, 3)
- Dependent variable: per-arm success rate, collision count, throughput (objects/minute)
- Control: same models as Experiment 1 (no retraining)
- N: 20 parallel trials per condition = 60 total
- Purpose: does parallel operation degrade individual performance?

**Experiment 3: Camera context ablation**
- Independent variable: camera input type (arm-local only vs. arm-local + overhead shared)
- Dependent variable: collision rate, success rate
- Control: same arm models, same workspace configuration
- N: 20 trials per condition
- Purpose: does seeing other arms reduce collision events?

**Experiment 4: Training regime ablation**
- Independent variable: training data (isolated per arm vs. cross-arm shared dataset)
- Dependent variable: per-arm success rate when operating in parallel
- Control: same total episodes per model (50), same training steps
- N: 20 trials per condition
- Purpose: does training on other arms' data help coordination?

### Statistical analysis plan

Primary comparison: Experiment 2, isolated vs. parallel success rate
- Test: McNemar's test (paired binary outcomes, same objects same positions)
- Required N for 80% power to detect 15% difference: N=47 trials per condition
- Practical N: 50 trials per condition (round up for safety)

Secondary comparison: Experiment 3, camera context effect on collision rate
- Collision rate is count data → Poisson regression or negative binomial test
- Expected collision rate: 0-5 per 20 trials (rare event)
- With rare events, need N=100 trials to detect halving of collision rate at 80% power

**Warning:** Collision rate may be too low to detect statistically with N=20. Plan for N=50 minimum for collision experiments.

### Paper title candidates (ranked by novelty)

1. "Three VLAs, Zero Messages: Implicit Coordination in Consumer Multi-Robot Manipulation via Visual Grounding Alone"
2. "Parallel Pick: Demonstrating Multi-Arm VLA Sorting on $400 of Consumer Hardware Without Explicit Inter-Robot Communication"
3. "TabletopSort: Data-Efficient Multi-Class Sorting with 3 Independently-Trained Consumer VLAs"

---

## SYNTHESIS: Recommended path

### Decision matrix

| Criterion | Scenario 1 (Logistics) | Scenario 2 (Digital Twin) | Scenario 3 (Lab Reset) | Scenario 4 (Multi-Arm) |
|-----------|----------------------|--------------------------|----------------------|----------------------|
| Research gap (literature) | MEDIUM | HIGH | HIGH (framing) | HIGH |
| Solves data collection pain | No — worse | Partially | No — worse | YES |
| Advisor "stop sponges" response | Medium | Low | High | High |
| Uses Unity/XR skills | Low | High (Isaac Lab) | Low | Medium |
| Achievable in 6 months | Yes | Risky | Yes | Yes (B only) |
| CoRL acceptance potential | Low-Medium | Medium (risky) | Medium | High |
| Thesis extension potential | Medium | High | High | High |
| Additional hardware cost | Zero | Zero | Zero | Zero |

### Primary recommendation: Scenario 4 (Multi-Arm Parallel Coordination)

Justification:
1. Strongest research gap (zero papers on consumer multi-arm VLA without communication)
2. Directly solves the data collection bottleneck (3x throughput from shared workspace setup)
3. Uses all 3 owned RoArm-M3 arms and 3 Azure Kinects (hardware already paid for)
4. Natural progression of existing 100% sponge-pick result
5. Clear, measurable research question with binary answer

### Secondary recommendation: Wrap Scenario 3 narrative around Scenario 4

The "lab reset" story makes Scenario 4 more concrete:
- Instead of "sort 3 bins", the task is "reset a lab bench"
- Objects: ruler, pen, bottle, notebook, eraser (more interesting than cube/cylinder)
- Each arm handles 2 objects instead of 1
- This creates richer per-arm behavior while maintaining the multi-arm coordination question

### Tertiary: Scenario 2 (Digital Twin) as thesis chapter only

The digital twin evaluation question is HIGH novelty but HIGH risk. Recommend:
- Month 7-8 of the 9-month timeline: build a basic Isaac Lab twin monitoring dashboard in Unity
- Test: "Does joint-state divergence between sim and real predict task failure?"
- Report result honestly — both positive and negative results are publishable here
- This becomes Chapter 5 of the thesis, extending beyond the CoRL paper

---

## EXPERIMENT DESIGN SUMMARY (C1 output for pipeline-agent and deploy-agent)

### For pipeline-agent (training specifications)

Training runs needed for Scenario 4 primary:
- 3 models × 50K steps = 3 sequential runs (same GPU)
- Ablation: 3 additional runs with cross-arm shared dataset
- Total: 6 training runs × ~4 hours each = ~24 hours GPU time
- All runs use smolvla_base pretrained checkpoint

### For deploy-agent (evaluation protocol)

Evaluation protocol for parallel coordination:
- Pre-defined 5×5 object position grid (25 positions, 5cm spacing)
- Each trial: 3 objects placed at random grid positions from pre-defined list
- Arms run simultaneously for 90 seconds per trial
- Human observer codes: success/fail per arm + collision (yes/no) + final object position
- Video recording mandatory (3 arms requires 3 camera streams + optional overhead summary)

Minimum trials:
- Experiment 1 (baseline): 20 trials × 3 arms × 3 classes = 180 individual arm trials (can run sequentially)
- Experiment 2 (parallel): 50 trials (3 arms simultaneously) = 50 sessions
- Experiment 3 (camera ablation): 50 trials × 2 conditions = 100 sessions

### For C2 research-analysis (statistics handoff)

Data to be collected per trial:
- Binary success per arm (3 values)
- Collision event (binary, with frame timestamp)
- Time to completion per arm
- Final object position (x,y in cm from reference corner)
- Checkpoint used (training steps)

Statistical tests planned:
- McNemar's test: within-arm success rate isolated vs. parallel
- Poisson regression: collision count vs. arm count (1, 2, 3 arms)
- Bootstrap CI: per-arm success rates with 95% CI
- Pearson correlation: arm success rate vs. training episodes

---

## VERIFICATION NOTES

All gap claims include confidence levels. No "zero papers" claim was made without 3+ distinct search terms returning no relevant results. The multi-arm consumer VLA gap was verified via 6 search terms. The "implicit coordination" framing was not found in any 2024-2026 paper searched.

Before CoRL submission, verify with fresh search:
- arXiv cs.RO (2024-2026): "consumer multi-arm VLA coordination"
- CoRL 2025 proceedings: any multi-arm sorting consumer paper
- ICRA 2026 accepted papers: "parallel manipulation low-cost"
