# Robotics Data Ecosystem Intelligence Report (March 2026)

**Agent**: DATA-OSINT
**Date**: 2026-03-07
**Scope**: Datasets, benchmarks, competitions, data infrastructure, and relevance to RoArm M3 + SmolVLA pipeline

---

## Executive Summary

The robotics data ecosystem has undergone a massive transformation between 2023-2026, driven by the convergence of VLA (Vision-Language-Action) foundation models and a new wave of open-source datasets. Key trends:

1. **Consolidation around foundation models**: pi0, GR00T N1, SmolVLA, OpenVLA, RoboVLMs -- all follow the pattern of VLM backbone + action head (diffusion/flow-matching)
2. **Cross-embodiment pretraining is now standard**: Open X-Embodiment proved that training on 22+ robots helps each individual robot
3. **Data quantity thresholds are clear**: 50 episodes is the minimum for single-task in-distribution; 150+ for OOD embodiment (our case)
4. **Community-driven data collection is the new frontier**: HuggingFace LeRobot Hub has democratized robotic data sharing
5. **Sim-to-real and real2sim evaluation are converging**: SIMPLER, ManiSkill3, and RoboCasa365 enable scalable evaluation

**Bottom line for our project**: Our RoArm M3 is an OOD embodiment for SmolVLA (pretrained only on SO-100). We need 100-150 high-quality episodes and 200K+ training steps. The 74-episode v3 dataset with 100% success proves the pipeline works; spatial diversity is the next frontier.

---

## 1. Major Robotics Datasets

### 1.1 Comparison Table

| Dataset | Org | Type | Size | Robots | Format | Access | Year |
|---------|-----|------|------|--------|--------|--------|------|
| **Open X-Embodiment** | Google DeepMind + 34 labs | Multi-task manipulation | 1M+ trajectories, 60 datasets | 22 robots (Franka, xArm, UR5, etc.) | RLDS (TF Datasets) | Open, Apache 2.0 | 2023 |
| **DROID** | Toyota Research + 13 institutions | Manipulation (in-the-wild) | 76K trajectories, 350h, 564 scenes, 86 tasks | Franka Panda 7-DOF | RLDS (TF Datasets), HuggingFace | Open | 2024 |
| **BridgeData V2** | UC Berkeley (RAIL) | Manipulation | 60,096 trajectories, 24 environments | WidowX 250 6-DOF | RLDS, LeRobot | Open | 2023 |
| **RoboCasa / RoboCasa365** | UT Austin | Kitchen manipulation (sim) | 600h+ human demos + 1600h+ synthetic, 365 tasks | Mobile manipulators, humanoids | robosuite format | Open (CC BY) | 2024/2026 |
| **LIBERO** | UT Austin | Lifelong learning manipulation | 130 tasks, 4 suites | Franka Panda (sim) | robosuite / HDF5 | Open | 2023 |
| **RLBench** | Imperial College London | Manipulation | 100+ tasks | Franka Panda (CoppeliaSim) | Custom (PyRep) | Open | 2020 |
| **MetaWorld** | Farama Foundation (ex-Berkeley) | Multi-task / Meta-RL | 50 tasks (MT1/MT10/MT50/ML1/ML10/ML45) | Sawyer (MuJoCo) | Gymnasium | Open | 2019/2025 |
| **ManiSkill 3** | UCSD (Hao Su Lab) | Manipulation + locomotion | Wide range of tasks, GPU-parallel | Multiple (Franka, humanoids, mobile) | SAPIEN, LeRobot-compatible | Open (Apache 2.0) | 2025 |
| **CALVIN** | Univ. of Freiburg | Long-horizon language-conditioned | Simulated benchmark | Franka Panda (sim) | Custom | Open | 2022 |
| **robomimic** | UT Austin / Stanford | Manipulation (from demos) | Multiple task datasets | Franka Panda (sim) | HDF5 | Open | 2021 |
| **community_dataset_v1** | HuggingFace LeRobot | SmolVLA pretraining | 128 datasets, 11,132 episodes | SO-100 (exclusively) | LeRobot v2.1 (Parquet + MP4) | Open (HF Hub) | 2025 |
| **LeRobot Hub** | HuggingFace + Community | Multi-robot manipulation | 1000s of datasets | SO-100, SO-101, ALOHA, Koch, xArm, etc. | LeRobot v2.1 (Parquet + MP4) | Open (HF Hub) | 2024-present |

### 1.2 Detailed Analysis

#### Open X-Embodiment (OXE)
- **Organization**: Google DeepMind, collaboration with 34 research labs worldwide
- **Scale**: 1M+ real robot trajectories, 22 robot embodiments, 527 skills, 160,266 tasks
- **Key insight**: RT-1-X trained on this mixture outperforms single-dataset models by 50% in small-data regimes
- **Format**: RLDS (TensorFlow Datasets for RL) -- standardized action format as 7D vector (x, y, z, roll, pitch, yaw, gripper)
- **URL**: https://robotics-transformer-x.github.io/
- **Dataset enrollment**: Open via Google Form for contributing new datasets
- **Key papers**: RT-1 (Brohan et al. 2022), RT-2 (Zitkovich et al. 2023), RT-X (Open X-Embodiment Collaboration 2023)
- **Relevance to us**: HIGH. Our RoArm M3 data could potentially be contributed. However, the 7D action format (Cartesian) differs from our 6-DOF joint-space actions. Conversion would be needed.

#### DROID (Distributed Robot Interaction Dataset)
- **Organization**: Toyota Research Institute + Stanford, Berkeley, CMU, etc. (13 institutions)
- **Scale**: 76K demonstrations, 350 hours, 564 scenes, 86 tasks, collected by 50 data collectors across North America, Asia, Europe over 12 months
- **Robot**: Franka Panda 7-DOF, standardized setup with 2x Zed 2 stereo + wrist-mounted Zed Mini + Oculus Quest 2 for VR teleoperation
- **Key result**: Co-training with DROID improves success rate by 22% in-distribution and 17% OOD vs Open-X
- **Camera calibrations**: 36K episodes have improved calibrations (April 2025 update)
- **Language annotations**: 3 natural language annotations per episode for 95% of successful episodes (75K episodes, Dec 2024 update)
- **Format**: RLDS (TF Datasets), also available on HuggingFace
- **URL**: https://droid-dataset.github.io/
- **Relevance to us**: MEDIUM. Shows the power of diverse data collection. Their VR teleoperation approach (Quest 2) is state-of-art for data quality. Our hand-guiding approach is lower quality but zero additional hardware cost.

#### BridgeData V2
- **Organization**: UC Berkeley (RAIL Lab)
- **Scale**: 60,096 trajectories, 24 environments
- **Robot**: WidowX 250 6-DOF -- low-cost arm similar in spirit to our setup
- **Key insight**: Performance improves with more data and higher capacity models; training on more skills improves generalization
- **Format**: RLDS, also converted to LeRobot format
- **URL**: https://rail-berkeley.github.io/bridgedata
- **Relevance to us**: HIGH. The WidowX 250 is a low-cost arm like RoArm M3. Their approach of collecting data across many environments with a portable arm matches what we could do. Their ~60K trajectories across 24 environments is the gold standard for low-cost arm data.

#### community_dataset_v1 (SmolVLA Pretraining Data)
- **Organization**: HuggingFace LeRobot community
- **Scale**: 128 datasets, 11,132 episodes
- **Robot**: Exclusively SO-100 (confirmed in SmolVLA paper)
- **Format**: LeRobot v2.1 (Parquet + MP4)
- **Key insight**: SmolVLA was pretrained ONLY on SO-100 data. RoArm M3 is completely OOD for this pretrained model
- **URL**: https://huggingface.co/datasets/lerobot/community_dataset_v1
- **Relevance to us**: CRITICAL. This is what our SmolVLA model was pretrained on. Understanding its composition helps explain why we need more data (OOD embodiment gap). The 6-DOF -> zero-pad 32-DOF architecture handles dimension mismatch, but kinematic/dynamic properties differ significantly.

#### RoboCasa365
- **Organization**: UT Austin (Yuke Zhu lab)
- **Scale**: 365 tasks, 2,500 kitchen environments, 600h+ human demos, 1600h+ synthetic
- **Robot**: Mobile manipulators, humanoids, quadrupeds
- **Format**: robosuite-compatible
- **Published at ICLR 2026**
- **Key features**: Supports Diffusion Policy, pi0, GR00T; 10 foundational skills; LLM-guided composite task generation
- **Relevance to us**: LOW for direct use, but HIGH conceptually. Shows the trend toward massive sim datasets with AI-generated diversity. Could be useful if we ever move to sim-to-real.

---

## 2. VLA Foundation Models Landscape

| Model | Org | Params | Architecture | Pretraining Data | Action Space | Year |
|-------|-----|--------|-------------|-----------------|--------------|------|
| **pi0** | Physical Intelligence | 3B VLM + action expert | VLM + Flow Matching | OXE + proprietary (8 robots) | Continuous, up to 50Hz | 2024 |
| **pi0.5** | Physical Intelligence | 3B+ | VLM + Flow Matching (improved) | Larger mixture | Continuous | 2025 |
| **GR00T N1** | NVIDIA | 2B | SigLip2 + T5 + DiT (flow matching) | Cross-embodiment (real + synthetic + human video) | Continuous, multi-embodiment | 2025 |
| **SmolVLA** | HuggingFace | 450M | SmolVLM + Flow Matching Action Expert | community_dataset_v1 (SO-100 only, 11K episodes) | Continuous, 50-step action chunks | 2025 |
| **OpenVLA** | Stanford/Berkeley | 7B | Llama-based VLM | OXE subset | Discrete tokens | 2024 |
| **RoboVLMs** | Tsinghua/ByteDance | Various (8 backbones tested) | VLM + various action heads | Multiple | Various | 2024/2026 |
| **RoboUniView** | Various | VLM-based | Unified view representation | CALVIN/OXE | Continuous | 2024 |
| **ACT** | Stanford (Tony Zhao) | ~30M | Transformer encoder-decoder | Per-task demos (no pretraining) | Action chunking (100-step) | 2023 |
| **Diffusion Policy** | Columbia/Toyota | ~50-300M | Diffusion-based | Per-task demos | Continuous | 2023 |

### Key Takeaways for VLA Selection
1. **SmolVLA (our choice)** is the most accessible: 450M params, single GPU training, community data
2. **pi0/pi0.5** is the most capable but fully proprietary data/weights
3. **GR00T N1** is open-weight but non-commercial license (NVIDIA OneWay)
4. **The architecture converging toward**: VLM backbone + flow matching / diffusion action head
5. **Action chunking is standard**: 50-step chunks (SmolVLA), not per-step prediction

---

## 3. Benchmarks and Evaluation

### 3.1 Simulation Benchmarks

| Benchmark | Org | Tasks | Robot | Evaluation Type | Used By |
|-----------|-----|-------|-------|-----------------|---------|
| **LIBERO** | UT Austin | 130 tasks in 4 suites (spatial, object, goal, long-horizon) | Franka Panda | Success rate, lifelong learning metrics | SmolVLA, pi0Fast, GR00T, LeRobot official |
| **SIMPLER** | Google / Stanford / UCSD | Real2sim manipulation | Google Robot, WidowX | Sim-real correlation, success rate | RT-X, Octo, OpenVLA |
| **CALVIN** | Univ. of Freiburg | Long-horizon language-conditioned (34 tasks) | Franka Panda | Chained task completion rate (1-5 tasks) | RoboUniView (SOTA 96.2%), many VLAs |
| **MetaWorld** | Farama Foundation | 50 tasks (MT1/10/50, ML1/10/45) | Sawyer | Success rate, multi-task/meta-learning | LeRobot official benchmark, RL algorithms |
| **RLBench** | Imperial College | 100+ manipulation tasks | Franka Panda (CoppeliaSim) | Success rate, few-shot generalization | PerAct, RVT, many IL methods |
| **ManiSkill 3** | UCSD | Diverse manipulation + locomotion | Multiple embodiments | Success rate, GPU-parallel eval | VLAs, RL baselines, RSS 2025 |
| **RoboCasa365** | UT Austin | 365 kitchen tasks | Mobile manipulators | Success rate, multi-task | pi0, GR00T, Diffusion Policy |
| **FrankaKitchen** | Stanford/Berkeley | 4 subtasks in kitchen | Franka Panda | Multi-task completion | D4RL, offline RL |

### 3.2 How VLAs Are Actually Evaluated

**Simulation evaluation** (increasingly standard):
- LIBERO is the de facto VLA benchmark -- SmolVLA, pi0Fast, GR00T N1 all report LIBERO numbers
- SIMPLER provides real2sim correlation for policies already deployed on real robots
- ManiSkill 3 and MetaWorld are integrated into LeRobot's `lerobot-eval` CLI

**Real-world evaluation** (still the gold standard):
- Success rate over N trials (typically 10-50 per condition)
- Variations tested: object position, distractor objects, lighting, novel objects
- No standardized protocol exists -- every paper defines its own eval

**Gap between sim and real**:
- SIMPLER showed "strong correlation" between sim and real for policies like RT-1-X and Octo
- But for fine-grained manipulation (our case), sim-real gap remains significant
- Our experience: offline L2=2.53 degrees (good) but initial deployment failed -- confirming that offline metrics are insufficient

### 3.3 Relevance to Our Setup

We cannot directly use most simulation benchmarks because:
- They use Franka Panda / Sawyer (not RoArm M3)
- They assume specific sim environments (CoppeliaSim, MuJoCo, SAPIEN)

Our evaluation is necessarily **real-world only** unless we create an Isaac Lab sim of RoArm M3 (partially started). Key metrics we should track:
1. Real-world success rate per spatial zone (5-zone evaluation)
2. Gripper close timing accuracy
3. Grasp depth (FK z-height at grasp moment)
4. OOD generalization (novel object positions)

---

## 4. Data Collection Methods and Teleoperation

### 4.1 State of the Art (2026)

| Method | Quality | Cost | Speed | Used By |
|--------|---------|------|-------|---------|
| **VR Teleoperation (Quest 2/3)** | Highest | $300-500 headset + custom mount | ~13 sec/episode | DROID, pi0, ALOHA |
| **Leader-Follower (dual arm)** | High | 2x robot cost | ~10-15 sec/episode | ALOHA, ACT, LeRobot |
| **SpaceMouse** | Medium-High | $200-400 | Slower, learning curve | BridgeData V2, many labs |
| **Hand-guiding (torque off)** | Medium | Zero (uses existing robot) | ~5-10 sec/episode | Our approach, some LeRobot users |
| **Keyboard/Gamepad** | Low | ~$30 | Slowest, low quality | Prototyping only |
| **Phone teleoperation** | Medium | Zero (phone app) | Limited DOF | LeRobot phone teleop |
| **Automated (RL in sim)** | Variable | Compute cost only | Fastest per episode | RoboCasa365 synthetic data |

### 4.2 Our Method: Hand-Guiding

**Advantages**:
- Zero additional hardware cost
- Intuitive for simple tasks (pick, place)
- Fast setup, immediate feedback

**Disadvantages**:
- Hand occlusion of camera view (minimal issue per our analysis -- 2/43 episodes affected)
- Lower trajectory quality than leader-follower (estimated 20% more data needed to compensate)
- Difficult for bimanual or complex tasks
- Speed limited by human motor control

**Recommendation**: For our current single-arm pick-place task, hand-guiding is adequate. For more complex tasks or if scaling beyond 150 episodes, consider adding a leader arm for leader-follower teleoperation.

### 4.3 ALOHA Teleoperation System

The ALOHA system (ACT paper, Tony Zhao et al. 2023) established the gold standard for low-cost teleoperation:
- **Static ALOHA**: 2x ViperX 300 6-DOF arms, leader-follower, ~$20K total
- **Mobile ALOHA**: Added mobile base, whole-body teleoperation, ~$32K
- **Key finding**: 50 demonstrations per task + co-training with existing data = 80-90% success on fine manipulation
- **Data format**: Compatible with LeRobot Hub

### 4.4 Key Data Collection Papers

| Paper | Key Finding | Data Requirement |
|-------|-------------|-----------------|
| ACT (2023) | Action chunking + 50 demos = 80-90% success | 10 minutes of data per task |
| Mobile ALOHA (2024) | Co-training with static ALOHA data boosts mobile tasks by up to 90% | 50 demos per task |
| SmolVLA (2025) | 5 positions x 10 reps = 50 episodes, in-distribution | 50 episodes minimum; 25 insufficient |
| DROID (2024) | 76K diverse demos improve generalization by 22% over OXE | More diversity > more quantity |
| BridgeData V2 (2023) | Performance improves with more data AND higher capacity | ~60K trajectories across 24 environments |

---

## 5. Competitions and Challenges (2025-2026)

### 5.1 Major Active Competitions

| Competition | Year | Location | Focus | Hardware |
|-------------|------|----------|-------|----------|
| **RoboCup 2026** | Jun 30 - Jul 6, 2026 | Songdo Convensia, Incheon, South Korea | Soccer, @Home, Industrial, Rescue, Junior | Various (Unitree, Fourier, custom) |
| **RoboCup 2025** | 2025 | Salvador, Brazil | Same leagues | Same |
| **ICRA 2026 Workshops** | Jun 2026 | (with RoboCup) | SCR@HOME -- Human-Centered Robotic Autonomy | Various |
| **Amazon Picking/Stowing Challenge** | Ongoing (evolved) | Various | Warehouse manipulation | Industrial arms |
| **NeurIPS Robot Learning Workshop** | Dec 2025 | San Diego | Learning-based manipulation | Varies |

### 5.2 RoboCup 2026 -- South Korea (CRITICAL for us)

RoboCup 2026 is **in South Korea (Incheon)**, making it the most relevant competition for our Korean-based project:
- **First time in South Korea** -- huge milestone
- **Organized by**: Korea Association of AI Robot Industry (KAR) + Incheon Technopark (ITP)
- **Expected**: 3,000+ participants, 15,000 visitors
- **Leagues**: RoboCupSoccer, RoboCup@Home, RoboCupIndustrial, RoboCupRescue, RoboCupJunior
- **Sponsors**: MathWorks, Unitree, Fourier, Booster, PAL Robotics, JP Morgan
- **RoboCup@Home** is the most relevant league for manipulation research -- autonomous service robots performing household tasks
- **RoboCupIndustrial** focuses on warehouse/logistics manipulation

### 5.3 Korean Robotics Ecosystem

South Korea is a major robotics hub:
- **KAIST** (Korea Advanced Institute of Science and Technology): Leading robotics research, contributed `kaist_nonprehensile` dataset to LeRobot Hub
- **Samsung, Hyundai, LG**: Major corporate R&D in service robots and cobots
- **Korea Institute of Robot and Convergence (KIRO)**: Government-backed robotics institute
- **Korean Robot Competition**: Annual domestic competition organized by KIRO
- **Korea Association of AI Robot Industry (KAR)**: Organizing RoboCup 2026
- **KIRIA (Korea Institute of Robotics & Technology Convergence)**: Standards and testing

---

## 6. Data Infrastructure

### 6.1 Data Formats

| Format | Used By | Structure | Pros | Cons |
|--------|---------|-----------|------|------|
| **LeRobot v2.1** | SmolVLA, LeRobot Hub, community | Parquet (state/action) + MP4 (video) | HF Hub integration, streaming, visualization | Newer format, less legacy tooling |
| **RLDS** | OXE, DROID, BridgeData V2 | TensorFlow Datasets | Standard for cross-embodiment, Google ecosystem | TF dependency, complex setup |
| **HDF5** | robomimic, LIBERO | Hierarchical binary | Fast random access, mature | Large files, no streaming |
| **robosuite** | RoboCasa, robomimic | MuJoCo-based | Standard for sim manipulation | Sim-only |
| **Custom/pickle** | Many academic datasets | Varies | Quick to create | Not interoperable |

### 6.2 HuggingFace Hub for Robotics

The HuggingFace Hub has become the central repository for robotics data:
- **LeRobot datasets**: Thousands of datasets from community contributors
- **Dataset viewer**: Browser-based visualization of robot data (video + joint states)
- **Streaming**: Can train on datasets without downloading full data
- **Format**: Standardized LeRobot v2.1 format (Parquet + MP4)
- **Key datasets on Hub**: ALOHA (static/mobile), community_dataset_v1, BridgeData V2 (converted), LIBERO (image version), KAIST nonprehensile, many custom robot datasets

### 6.3 Our Data Pipeline

```
collect_data_manual.py          # Azure Kinect + torque-off hand-guiding
        |
        v
convert_to_lerobot_v3.py       # Convert to LeRobot v2.1 format (Parquet + MP4)
        |
        v
HuggingFace Hub (optional)      # Push to Hub for sharing/backup
        |
        v
lerobot-train (run_official_train.py)  # Train SmolVLA
```

**Could we contribute our data?**
- YES. Our LeRobot v2.1 format is Hub-compatible
- Our RoArm M3 data would be the FIRST RoArm M3 dataset on the Hub
- This would benefit the community (new embodiment) and us (visibility, potential collaboration)
- Requirement: Clean data, proper task descriptions, metadata

---

## 7. Data Scaling Laws for Robotics

### 7.1 How Much Data Is Enough?

| Scenario | Data Required | Evidence |
|----------|--------------|----------|
| **In-distribution, pretrained model, single task** | 50 episodes (5 positions x 10 reps) | SmolVLA paper, SO-100 experiments |
| **In-distribution, from scratch** | 90-284 episodes | Diffusion Policy paper |
| **OOD embodiment, pretrained model** | 100-150+ episodes, 200K+ steps | Our experience + SmolVLA paper extrapolation |
| **OOD embodiment, from scratch** | 500+ episodes | Estimated, no direct evidence |
| **Multi-task generalist** | 10K-100K+ episodes | OXE, DROID, pi0 |
| **Cross-embodiment foundation model** | 100K-1M+ episodes | OXE (1M+), pi0 (proprietary), GR00T N1 |

### 7.2 Key Scaling Insights

1. **Quality > Quantity**: Our v1 (50 episodes, bad quality) = 0% success. Our v3 (74 episodes, good quality) = 100% success. A 2x improvement in data quality was more impactful than 2x more data.

2. **Diversity > Repetition**: DROID showed that 564 diverse scenes beat focused repetition. For single-task, 5 distinct positions with 10 reps each outperforms 50 reps at 1 position.

3. **Co-training helps**: Mobile ALOHA showed that co-training with existing static ALOHA data can boost success rates by up to 90%. SmolVLA's pretraining on community_dataset_v1 provides a similar co-training effect.

4. **More data = more robust, not necessarily more capable**: Beyond a threshold, additional data primarily improves OOD robustness, not peak in-distribution performance.

5. **Episode quality criteria** (our validated list):
   - 7 phases: start -> approach+open -> pregrasp -> descend(open) -> grasp -> lift -> return
   - 5-6 seconds duration (our data) vs 13 seconds (official SmolVLA demos)
   - Gripper opens during Phase 2, holds open until Phase 5
   - Static frame deduplication reduces "stay still" learning bias

### 7.3 Emerging Trend: Synthetic Data

- **RoboCasa365**: 1600h+ synthetic data alongside 600h+ human data
- **GR00T N1**: Trained on mix of real trajectories + human videos + synthetic data
- **Isaac Lab/Sim**: GPU-parallel simulation enables millions of episodes
- **Verdict**: Synthetic data works for pretraining/co-training but cannot replace real data for fine-tuning on specific hardware

---

## 8. Emerging Trends (2025-2026)

### 8.1 Foundation Model Wars

The robotics foundation model landscape is consolidating rapidly:

| Tier | Models | Data Strategy | Business Model |
|------|--------|--------------|----------------|
| **Closed/Commercial** | pi0/pi0.5 (Physical Intelligence) | Proprietary data from 8+ robots | Startup (raised $400M+) |
| **Open-weight/Restricted** | GR00T N1 (NVIDIA) | Mix of real + synthetic + video | Non-commercial license |
| **Fully Open** | SmolVLA (HuggingFace), OpenVLA (Stanford) | Community data, open pretraining | Apache 2.0 |
| **Academic** | ACT, Diffusion Policy, RoboVLMs | Per-paper datasets | Research only |

### 8.2 Architecture Convergence

All leading VLAs in 2025-2026 converge on:
```
Pre-trained VLM (vision + language understanding)
        |
        v
Action Expert (flow matching / diffusion)
        |
        v
Action Chunk (50-100 steps into the future)
```

Key architectural choices:
- **VLM backbone**: SigLip/PaLI (Google), SmolVLM (HF), Llama-based (Meta/open)
- **Action generation**: Flow matching (pi0, SmolVLA, GR00T N1) > Diffusion > Discrete tokens
- **Action representation**: Continuous joint angles (most), discrete bins (OpenVLA -- now considered suboptimal)

### 8.3 Cross-Embodiment is the New Normal

- OXE established: training on multiple robots helps each individual robot
- GR00T N1 trains on humanoids + mobile manipulators + single arms simultaneously
- SmolVLA supports up to 32-DOF robots via zero-padding
- Implication for us: Contributing our RoArm M3 data to shared pools could benefit both us and others

### 8.4 Real2Sim Evaluation is Maturing

- SIMPLER (Google/Stanford) showed strong sim-real correlation
- ManiSkill 3 provides GPU-parallel evaluation infrastructure
- LeRobot integrates `lerobot-eval` for LIBERO and MetaWorld benchmarks
- Trend: Standard benchmarks are becoming a requirement for publication

---

## 9. Relevance Assessment for RoArm M3 + Azure Kinect + SmolVLA

### 9.1 Where We Fit in the Ecosystem

| Aspect | Our Setup | Ecosystem Norm | Gap |
|--------|-----------|---------------|-----|
| **Robot** | RoArm M3 (6-DOF, low-cost) | Franka/xArm/SO-100 | OOD embodiment for all pretrained models |
| **Camera** | Azure Kinect (1x, 720P) | 2-3 cameras (stereo + wrist) | Fewer viewpoints; higher depth quality |
| **VLA** | SmolVLA (450M) | pi0 (3B), GR00T (2B) | Smaller but trainable on single GPU |
| **Data** | 74 episodes (v3), hand-guiding | 50+ (in-dist), 10K+ (generalist) | Adequate for single-task, need 100-150 for spatial diversity |
| **Training** | RTX 4090 Laptop, batch_size=64 | A100/H100 cluster | Sufficient for SmolVLA; cannot run pi0/GR00T |
| **Evaluation** | Real-world only | Sim + real | No sim benchmark available |
| **Data format** | LeRobot v2.1 | LeRobot v2.1 / RLDS | Fully compatible with HF Hub |

### 9.2 Actionable Opportunities

1. **Contribute RoArm M3 data to HuggingFace Hub**
   - First RoArm M3 dataset on the Hub
   - Increases visibility and potential for cross-embodiment benefits
   - Our LeRobot v2.1 format is already Hub-compatible

2. **Use LIBERO as a sim benchmark**
   - LeRobot supports `lerobot-eval` on LIBERO
   - Could validate SmolVLA fine-tuning approach in simulation before real-world
   - Would not require RoArm M3 sim model

3. **Attend RoboCup 2026 in Incheon**
   - In South Korea, physically accessible
   - RoboCup@Home league is relevant to manipulation research
   - Networking with Korean robotics community (KAIST, KAR)

4. **Consider adding a wrist camera**
   - DROID and pi0 both benefit significantly from wrist-mounted cameras
   - Would provide close-up view for grasp precision
   - USB camera (Logitech C920 or similar) would suffice

5. **Explore co-training with existing datasets**
   - Mobile ALOHA showed that co-training with related data boosts performance
   - Could co-train with SO-100 data from LeRobot Hub (same SmolVLA pretraining distribution)
   - Would require matching action dimensions (both are 6-DOF)

### 9.3 What NOT to Do

1. **Do not switch to pi0 or GR00T N1** -- We cannot train these on our RTX 4090 Laptop (need A100+), and pi0 weights are not fully open
2. **Do not try to convert our data to RLDS format** -- LeRobot v2.1 is the correct format for SmolVLA
3. **Do not pursue sim-to-real** unless we first create a high-fidelity RoArm M3 URDF in Isaac Lab (partially started but not validated)
4. **Do not pursue massive data collection (1000+ episodes)** -- Quality and diversity matter more than raw quantity for single-task

---

## 10. Dataset Comparison Matrix (Quick Reference)

### Real-World Datasets Ranked by Relevance to Our Setup

| Rank | Dataset | Why Relevant | How to Use |
|------|---------|-------------|-----------|
| 1 | **community_dataset_v1** | SmolVLA pretraining data (SO-100) | Already used via `smolvla_base` pretrained weights |
| 2 | **LeRobot Hub (ALOHA)** | Same LeRobot format, similar tasks | Potential co-training, reference for data quality |
| 3 | **BridgeData V2** | Low-cost arm, similar philosophy | Study their data collection methodology (24 environments) |
| 4 | **DROID** | Gold standard for in-the-wild manipulation data | Study their VR teleoperation approach, camera setup |
| 5 | **Open X-Embodiment** | Cross-embodiment pretraining standard | Potential future contribution of our RoArm M3 data |

### Simulation Benchmarks Ranked by Accessibility

| Rank | Benchmark | LeRobot Integration | GPU Requirement |
|------|-----------|---------------------|-----------------|
| 1 | **LIBERO** | Official (`lerobot-eval`) | Moderate (single GPU) |
| 2 | **MetaWorld** | Official (`lerobot-eval`) | Low (CPU possible) |
| 3 | **ManiSkill 3** | Partial (VLA baselines available) | NVIDIA GPU (Vulkan) |
| 4 | **CALVIN** | Not official, but data on Hub | Moderate |
| 5 | **RLBench** | Not integrated | CoppeliaSim + GPU |

---

## 11. Key Papers Reference List

| Paper | Year | Key Contribution | arXiv |
|-------|------|-----------------|-------|
| Open X-Embodiment (RT-X) | 2023 | 1M+ trajectories, 22 robots, cross-embodiment transfer | 2310.08864 |
| DROID | 2024 | 76K demos, 564 scenes, in-the-wild data collection | 2403.12945 |
| BridgeData V2 | 2023 | 60K trajectories, low-cost arm | 2308.12952 |
| pi0 | 2024 | VLM + flow matching, generalist robot policy | 2410.24164 |
| SmolVLA | 2025 | 450M params, community-driven, single GPU training | 2506.01844 |
| GR00T N1 | 2025 | Open foundation model for humanoids | 2503.14734 |
| ACT (ALOHA) | 2023 | Action chunking with transformers, leader-follower | 2304.13705 |
| Mobile ALOHA | 2024 | Whole-body teleoperation, co-training | 2401.02117 |
| LIBERO | 2023 | Lifelong learning benchmark, 130 tasks | 2306.03310 |
| SIMPLER | 2024 | Real2sim evaluation with strong correlation | 2405.05941 |
| CALVIN | 2022 | Long-horizon language-conditioned benchmark | 2112.03227 |
| RoboVLMs | 2024/2026 | Systematic study of VLA design choices | 2412.14058 |
| RoboCasa365 | 2026 | 365 kitchen tasks, 2500 environments | ICLR 2026 |
| ManiSkill 3 | 2025 | GPU-parallel sim, multi-embodiment | RSS 2025 |
| robomimic | 2021 | What matters in learning from demos | 2108.03298 |

---

## 12. Conclusions and Recommendations

### The Robotics Data Ecosystem in 2026: State Summary

1. **We are in the "GPT-3 era" of robotics**: Foundation models (pi0, GR00T, SmolVLA) are showing that scaling data and compute works, but we are far from GPT-4-level generalization
2. **Data is the bottleneck, not algorithms**: The same architectures (flow matching + VLM) work everywhere; what differs is data quality, quantity, and diversity
3. **Open-source is winning**: LeRobot/HuggingFace ecosystem has democratized robotics data collection and model training
4. **Cross-embodiment transfer works but has limits**: Helpful for pretraining, but fine-tuning on target embodiment is still required
5. **Evaluation is still fragmented**: No unified real-world benchmark exists; sim benchmarks (LIBERO, MetaWorld) are becoming standard for VLA comparison

### Specific Recommendations for Our Project

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| **P0** | Collect 150 total episodes (5-zone spatial diversity) | 2-3 sessions | Enable robust 5-zone deployment |
| **P1** | Push our dataset to HuggingFace Hub | 1 hour | First RoArm M3 data; community contribution |
| **P1** | Run LIBERO eval with our SmolVLA checkpoint | 2-3 hours | Standardized comparison metric |
| **P2** | Attend/follow RoboCup 2026 (Incheon) | Travel | Networking, Korean robotics community |
| **P2** | Add wrist-mounted USB camera | $30-50 | Better grasp-phase visual feedback |
| **P3** | Explore co-training with SO-100 Hub data | 1 day experiment | Potential performance boost |
| **P3** | Contribute to OXE dataset enrollment | Application form | Cross-embodiment visibility |

---

*Report compiled from primary sources: project websites, arxiv papers, HuggingFace Hub, GitHub repositories. All URLs verified as of 2026-03-07.*
