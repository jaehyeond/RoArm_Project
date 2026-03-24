# Projection / AR / Laser + Robot Manipulation: Complete Landscape

> Generated: 2026-03-25. Systematic search across arXiv, Google Scholar, Semantic Scholar, Exa, Brave.
> Scope: Papers using projectors, lasers, or spatial augmented reality WITH robot manipulation/HRI.

---

## CATEGORY A: Physical Projection onto Robot Workspace (Projector/Laser Hardware)

These are papers that use an ACTUAL PHYSICAL PROJECTOR or LASER to project information into the real robot workspace.

---

### A1. Vogel et al. — Projection-Based Safety System (Fraunhofer IFF)
- **Title**: "Towards safe physical human-robot collaboration: A projection-based safety system"
- **Authors**: Christian Vogel, Maik Poggendorf, Christoph Walter, Norbert Elkmann
- **Venue**: IEEE/RSJ IROS 2011
- **arXiv ID**: N/A (IEEE)
- **Summary**: Projector-camera system creates a dynamic light barrier around the robot workspace. When a human breaches the projected safety zone, the camera detects deviation and triggers a robot stop. Pioneering work in projector-based safety for HRC.
- **Uses projection/AR/laser?**: YES — physical projector projects safety zones onto workspace
- **Robot**: Industrial robot arm
- **Policy model**: N/A (safety system, not learning)

**Follow-up** (2017): "Safeguarding and supporting future human-robot cooperative manufacturing processes by a projection- and camera-based technology" — Vogel, Walter, Elkmann. Procedia Manufacturing 11:39-46. Extends to "speed and separation monitoring" with dynamic safety spaces + arbitrary info visualization + interaction functionalities.

**Follow-up** (2013): "A Projection-based Sensor System for Safe Physical Human-Robot Collaboration" — IROS 2013. Incorporates joint positions/velocities for minimal, well-shaped safety spaces.

---

### A2. Chadalavada et al. — Robot Intention Projection on Floor
- **Title**: "That's on my mind! Robot to human intention communication through on-board projection on shared floor space"
- **Authors**: Ravi T. Chadalavada, Henrik Andreasson, Robert Krug, Achim J. Lilienthal
- **Venue**: IEEE RO-MAN 2015
- **arXiv ID**: N/A (IEEE)
- **Summary**: Autonomous forklift with an onboard LED projector projects navigation intention patterns (Line, Arrow, Blinking Arrow) onto the shared floor. Significantly improved human comfort and productivity during encounters.
- **Uses projection/AR/laser?**: YES — LED projector mounted on robot projects onto floor
- **Robot**: Autonomous forklift
- **Policy model**: N/A (intent communication)

**Follow-up** (2020): "Bi-directional navigation intent communication using spatial augmented reality and eye-tracking glasses for improved safety in human-robot interaction" — Chadalavada et al. Robotics and Computer-Integrated Manufacturing. Extends to bidirectional communication with eye-tracking.

---

### A3. Wengefeld et al. — Laser Projection for Robot Intention
- **Title**: "A Laser Projection System for Robot Intention Communication and Human Robot Interaction"
- **Authors**: Tim Wengefeld, Dominik Höchemer, Horst-Michael Gross
- **Venue**: IEEE RO-MAN 2020
- **arXiv ID**: N/A (IEEE)
- **Summary**: Robot-mounted laser projection system for communicating robot intention and enabling interaction. Laser projections are bright, high-contrast, and work well in both indoor and outdoor environments.
- **Uses projection/AR/laser?**: YES — laser projector on robot
- **Robot**: Mobile service robot
- **Policy model**: N/A (intent communication)

---

### A4. Seet et al. — Laser Graphics for Robot Deployment (NTU/IntechOpen)
- **Title**: "Laser Graphics in Augmented Reality Applications for Real-World Robot Deployment"
- **Authors**: Gerald Seet, Viatcheslav Iastrebov, Dinh Quang Huy, Pang Wee-Ching
- **Venue**: IntechOpen book chapter, 2016
- **arXiv ID**: N/A
- **Summary**: Develops a laser projection-based Spatial AR system for mobile robots. Projects line graphics and text onto surfaces to augment reality with the robot's intention, status, and planned path. Laser-generated outline-graphics are viable for natural environments with bright ambient lighting.
- **Uses projection/AR/laser?**: YES — laser projector on robot
- **Robot**: Mobile robotic platform
- **Policy model**: N/A (visualization)

---

### A5. Andersen & Boegh — Projector-Based Robot Programming (Aalborg)
- **Title**: "Intuitive task programming of stud welding robots for ship construction"
- **Authors**: Rasmus S. Andersen, Simon Boegh et al.
- **Venue**: ISR 2015 / Aalborg University
- **arXiv ID**: N/A
- **Summary**: Projector mounted on robot end-effector projects stud positions directly onto ship walls before welding. Non-expert operators can program, verify, and reprogram the robot's task on-site using an IMU pointing device. Projection mapping enables intuitive spatial programming.
- **Uses projection/AR/laser?**: YES — projector on end-effector projects onto work surface
- **Robot**: Industrial stud welding robot
- **Policy model**: N/A (task programming)

**Related** (2016): "Facilitating Programming of Vision-Equipped Robots through Robotic Skills and Projection Mapping" — Andersen. Aalborg Universitetsforlag. PhD thesis extending the concept.

---

### A6. Projection-Based AR for Electronic Assembly
- **Title**: "Projection-Based Augmented Reality Assistance for Manual Electronic Component Assembly Processes"
- **Authors**: (MDPI Applied Sciences 2025)
- **Venue**: Applied Sciences, 2025
- **Summary**: Uses projector-based AR to guide manual assembly of electronic components. Projects step-by-step instructions directly onto the workspace.
- **Uses projection/AR/laser?**: YES — projector onto assembly workspace
- **Robot**: Manual assembly guidance (human-centered)
- **Policy model**: N/A

---

### A7. Projector-Based AR for Robot Milling
- **Title**: "Projector-based Augmented Reality support for shop-floor programming of industrial robot milling operations"
- **Venue**: IEEE 2022
- **Summary**: Uses a projector to display AR information for programming industrial robot milling operations on the shop floor.
- **Uses projection/AR/laser?**: YES — projector
- **Robot**: Industrial milling robot
- **Policy model**: N/A (programming support)

---

### A8. LARS — Light Augmented Reality System for Collective Robotics
- **Title**: "LARS: A Light-Augmented Reality System for Collective Robotic Interaction"
- **Authors**: Mohsen Raoufi, Pawel Romanczuk, Heiko Hamann
- **Venue**: Sensors 2025; arXiv preprint 2024
- **arXiv ID**: 2411.00007
- **Summary**: Open-source framework that uses a projector to create dynamic visual stimuli (gradients, fields, trails) in the physical environment where real swarm robots operate. Enables indirect robot-robot communication (stigmergy) and makes hidden robot states visible to humans. Cross-platform, marker-free.
- **Uses projection/AR/laser?**: YES — overhead projector projects onto robot arena
- **Robot**: Miniature to mid-sized swarm robots
- **Policy model**: N/A (swarm interaction, stigmergy)
- **Code**: https://github.com/mohsen-raoufi/LARS

---

### A9. Tschulik et al. — SAR for Heavy Machinery with Laser
- **Title**: "Spatial augmented reality for heavy machinery using laser projections"
- **Authors**: Maximilian Tschulik, Thomas Kernbauer, Philipp Fleck, Clemens Arth
- **Venue**: Computers & Graphics, 2025
- **Summary**: Uses laser projections for spatial augmented reality on heavy machinery. Laser projections are bright enough for outdoor industrial use.
- **Uses projection/AR/laser?**: YES — laser projection
- **Robot**: Heavy machinery (construction/industrial)
- **Policy model**: N/A

---

### A10. ProjecTA — Robot with In-Situ Projection
- **Title**: "ProjecTA: A Semi-Humanoid Robotic Teaching Assistant with In-Situ Projection for Guided Tours"
- **Authors**: Hanqing Zhou, Yichuan Zhang, Zihan Zhang, Wei Zhang, Chao Wang, Pengcheng An
- **Venue**: arXiv 2026
- **arXiv ID**: 2601.11328v2
- **Summary**: Semi-humanoid robot with a body-mounted projector that projects content directly onto real-world objects during walk-and-talk tours, instead of using a screen. Reduces extraneous cognitive load.
- **Uses projection/AR/laser?**: YES — onboard projector projects onto real objects
- **Robot**: Semi-humanoid teaching robot
- **Policy model**: N/A (educational HRI)

---

### A11. Torielli et al. — Laser-Guided Robot Assistance for Disabled
- **Title**: "A Laser-guided Interaction Interface for Providing Effective Robot Assistance to People with Upper Limbs Impairments"
- **Authors**: Davide Torielli, Liana Bertoni, Luca Muratore et al.
- **Venue**: IEEE RA-L 2025; arXiv 2025
- **arXiv ID**: 2503.15987
- **Summary**: Head-wearable laser pointing device enables users with upper limb impairments to control a robotic arm. User points the laser at objects/locations; robot executes reaching and manipulation.
- **Uses projection/AR/laser?**: YES — head-mounted laser pointer for task specification
- **Robot**: Robotic arm (assistive)
- **Policy model**: N/A (direct control interface)

---

### A12. Liu et al. — Laser Pointer for Wheelchair Robotic Arm
- **Title**: "Object Affordance-Based Implicit Interaction for Wheelchair-Mounted Robotic Arm Using a Laser Pointer"
- **Authors**: Yaxin Liu, Yan Liu, Yufeng Yao, Ming Zhong
- **Venue**: Sensors 2023
- **Summary**: Laser pointer used to specify objects for a wheelchair-mounted robotic arm. Combines laser pointing with affordance recognition for implicit interaction.
- **Uses projection/AR/laser?**: YES — laser pointer for object selection
- **Robot**: Wheelchair-mounted robotic arm
- **Policy model**: Affordance-based (not VLA)

---

### A13. Kaiser et al. — Laser Pointing to Control a Robot
- **Title**: "Laser pointing to control a robot"
- **Authors**: B. Kaiser, R.A. Tauro, H. Woern
- **Venue**: 2009
- **Summary**: Laser pointer tool controls an industrial 6-axis robot. User points laser at objects to be grasped; robot detects laser spot and executes grasp.
- **Uses projection/AR/laser?**: YES — laser pointer for task specification
- **Robot**: Industrial 6-axis serial robot
- **Policy model**: N/A (direct control)

---

### A14. Ishii et al. — Laser Gesture Interface for Robot Control
- **Title**: "Designing Laser Gesture Interface for Robot Control"
- **Authors**: Ishii et al.
- **Venue**: INTERACT 2009 (Springer)
- **Summary**: Laser pointer UI with stroke gesture recognition (lasso, dwelling gestures for object selection, stroke gestures for commands).
- **Uses projection/AR/laser?**: YES — laser pointer gestures
- **Robot**: General robot
- **Policy model**: N/A

---

### A15. PATI — Projection-Based Table-Top Interface
- **Title**: "PATI: A Projection-Based Augmented Table-Top Interface for Robot Programming"
- **Venue**: ACM IUI 2019
- **Summary**: Projects an interactive interface directly onto a table surface for intuitive robot programming. Users can specify tasks through projected visuals.
- **Uses projection/AR/laser?**: YES — projector onto table-top
- **Robot**: Robot arm
- **Policy model**: N/A (programming interface)

---

### A16. Projection-Based Interaction Partner Clarification
- **Title**: "A projection-based approach for clarifying interaction partners in human-robot communication"
- **Venue**: Frontiers in Robotics and AI, 2025
- **Summary**: Robot-mounted projector illuminates the area around both the robot and the person it is addressing, clarifying who the robot is communicating with. Improves feeling of being directly addressed.
- **Uses projection/AR/laser?**: YES — robot-mounted projector
- **Robot**: Social robot
- **Policy model**: N/A (communication)

---

### A17. AR Robot Programming (Thoo et al.)
- **Title**: "Online and Offline Robot Programming via Augmented Reality Workspaces"
- **Authors**: Yong Joon Thoo, Jeremie Maceiras, Philip Abbet et al.
- **Venue**: arXiv 2021
- **arXiv ID**: 2107.01884
- **Summary**: AR workspaces for robot programming, reducing costs of reprogramming. Various interfaces including head-mounted and projected.
- **Uses projection/AR/laser?**: YES (AR workspace)
- **Robot**: Industrial robot
- **Policy model**: N/A

---

## CATEGORY B: Visual/Computational Affordance Maps (Not Physical Projection)

These papers create "projected" affordance/value maps COMPUTATIONALLY but do NOT use a physical projector.

---

### B1. VoxPoser
- **Title**: "VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models"
- **Authors**: Wenlong Huang, Chen Wang, Ruohan Zhang et al.
- **Venue**: CoRL 2023
- **arXiv ID**: 2307.05973
- **Summary**: Uses LLMs + VLMs to compose 3D value maps (affordance + avoidance) in voxel space, then synthesizes trajectories via model-predictive control. Zero-shot, no training required.
- **Uses projection/AR/laser?**: NO — computational 3D value maps, not physical projection
- **Robot**: Franka Panda
- **Policy model**: LLM + VLM + MPC (zero-shot)

---

### B2. RT-Affordance
- **Title**: "RT-Affordance: Affordances are Versatile Intermediate Representations for Robot Manipulation"
- **Authors**: Soroush Nasiriany, Sean Kirmani, Tianli Ding et al.
- **Venue**: arXiv 2024
- **arXiv ID**: 2411.02704
- **Summary**: Affordance images as intermediate representations guide robot manipulation. An affordance model generates 2D visual affordance maps, then a policy executes based on those affordance images.
- **Uses projection/AR/laser?**: NO — computational affordance images
- **Robot**: Google robots
- **Policy model**: Two-stage: affordance model + policy network

---

### B3. RT-Sketch
- **Title**: "RT-Sketch: Goal-Conditioned Imitation Learning from Hand-Drawn Sketches"
- **Authors**: Priya Sundaresan, Quan Vuong, Jiayuan Gu et al.
- **Venue**: arXiv 2024
- **arXiv ID**: 2403.02709
- **Summary**: Uses hand-drawn sketches as goal representations for robot manipulation. Sketches are less ambiguous than language and less over-specified than images.
- **Uses projection/AR/laser?**: NO — digital sketches as input, not physical projection
- **Robot**: Google robots
- **Policy model**: Goal-conditioned imitation learning

---

### B4. RT-Trajectory
- **Title**: "RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches"
- **Authors**: Jiayuan Gu, Sean Kirmani, Paul Wohlhart et al.
- **Venue**: arXiv 2023
- **arXiv ID**: 2311.01977
- **Summary**: Uses 2D trajectory sketches (drawn on images) as intermediate representations. Enables generalization to novel motion trajectories at test time.
- **Uses projection/AR/laser?**: NO — digital trajectory sketches overlaid on images
- **Robot**: Google robots
- **Policy model**: RT-2 variant conditioned on trajectory sketches

---

### B5. RoboPoint
- **Title**: "RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics"
- **Authors**: Wentao Yuan, Jiafei Duan, Valts Blukis et al.
- **Venue**: arXiv 2024
- **arXiv ID**: 2406.10721
- **Summary**: VLM fine-tuned to predict precise 2D action points on images for robot manipulation. Trained on synthetic data generated from 3D environments.
- **Uses projection/AR/laser?**: NO — computational point prediction
- **Robot**: Various (Franka, etc.)
- **Policy model**: Fine-tuned VLM

---

### B6. SpatialVLA
- **Title**: "SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model"
- **Authors**: Delin Qu, Haoming Song, Qizhi Chen et al.
- **Venue**: arXiv 2025
- **arXiv ID**: 2501.15830
- **Summary**: Proposes spatial understanding as the key for robot manipulation. Incorporates 3D spatial representations (depth, normals, etc.) into a VLA model to improve manipulation accuracy.
- **Uses projection/AR/laser?**: NO — computational spatial representations
- **Robot**: Various
- **Policy model**: VLA with spatial representations

---

### B7. Dream2Real
- **Title**: "Dream2Real: Zero-Shot 3D Object Rearrangement with Vision-Language Models"
- **Authors**: Ivan Kapelyukh, Yifei Ren, Ignacio Alzugaray et al.
- **Venue**: arXiv 2023
- **arXiv ID**: 2312.04533
- **Summary**: Robot autonomously constructs a 3D digital twin, generates rearrangements in simulation using VLMs, then executes in reality. "Dreaming" the solution before doing it.
- **Uses projection/AR/laser?**: NO — simulation-based
- **Robot**: Franka Panda
- **Policy model**: VLM + simulation + execution

---

## CATEGORY C: Visual Augmentation for Robot Learning (Generative/Digital)

These use generative models to augment training data digitally, not physical projection.

---

### C1. ROSIE
- **Title**: "Scaling Robot Learning with Semantically Imagined Experience"
- **Authors**: Tianhe Yu, Ted Xiao, Austin Stone et al.
- **Venue**: RSS 2023; arXiv 2023
- **arXiv ID**: 2302.11550
- **Summary**: Uses text-to-image diffusion models (Imagen Editor) to perform aggressive data augmentation via inpainting: novel objects, backgrounds, distractors. Policy trained on augmented data can perform novel tasks.
- **Uses projection/AR/laser?**: NO — digital image augmentation
- **Robot**: Google robots (RT-1)
- **Policy model**: RT-1 trained on augmented data

---

### C2. GenAug
- **Title**: "GenAug: Retargeting behaviors to unseen situations via Generative Augmentation"
- **Authors**: Zoey Chen, Sho Kiami, Abhishek Gupta et al.
- **Venue**: arXiv 2023
- **arXiv ID**: 2302.06671
- **Summary**: Uses generative models to augment robot manipulation datasets, enabling generalization to unseen scenarios without collecting new physical data.
- **Uses projection/AR/laser?**: NO — digital generative augmentation
- **Robot**: Various
- **Policy model**: Imitation learning on augmented data

---

### C3. GNFactor
- **Title**: "GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields"
- **Authors**: Yanjie Ze, Ge Yan, Yueh-Hua Wu et al.
- **Venue**: CoRL 2023
- **arXiv ID**: 2308.16891
- **Summary**: Combines 3D neural radiance fields with semantic features for multi-task robot learning. Generalizable NeRF provides 3D understanding.
- **Uses projection/AR/laser?**: NO — neural 3D representations
- **Robot**: Real robot (Franka)
- **Policy model**: PerAct variant with NeRF features

---

### C4. MimicPlay
- **Title**: "MimicPlay: Long-Horizon Imitation Learning by Watching Human Play"
- **Authors**: Chen Wang, Linxi Fan, Jiankai Sun et al.
- **Venue**: CoRL 2023
- **arXiv ID**: 2302.12422
- **Summary**: Learns long-horizon manipulation from human play videos. Extracts high-level plan from human demonstrations, then low-level policy executes with robot.
- **Uses projection/AR/laser?**: NO
- **Robot**: Franka
- **Policy model**: Hierarchical: human video plan + robot policy

---

### C5. LIV
- **Title**: "LIV: Language-Image Representations and Rewards for Robotic Control"
- **Authors**: Yecheng Jason Ma, William Liang, Vaidehi Som et al.
- **Venue**: ICML 2023
- **arXiv ID**: 2306.00958
- **Summary**: Unified objective for vision-language representation and reward learning from action-free videos with text. Learns reward functions from human videos.
- **Uses projection/AR/laser?**: NO
- **Robot**: Various simulated + real
- **Policy model**: RL with learned LIV reward

---

## CATEGORY D: AR (Head-Mounted) for Robot Interaction

These use head-mounted AR (HoloLens, etc.), NOT physical projection.

---

### D1. Spot-On — MR Interface for Multi-Robot Cooperation
- **Title**: "Spot-On: A Mixed Reality Interface for Multi-Robot Cooperation"
- **Authors**: Tim Engelbracht, Petar Lukovic, Tjark Behrens et al. (ETH Zurich)
- **Venue**: arXiv 2025
- **arXiv ID**: 2505.22539
- **Summary**: MR framework allowing multiple quadruped robots to operate via MR interface. Supports collaborative tasks with drawers, doors, light switches.
- **Uses projection/AR/laser?**: MR headset (not physical projector)
- **Robot**: Quadruped robots (Spot)
- **Policy model**: N/A (teleoperation interface)

---

### D2. Chu & Weng — AR Robot Programming by Demonstration
- **Title**: "Experimental analysis of augmented reality interfaces for robot programming by demonstration in manufacturing"
- **Authors**: Chih-Hsing Chu, Chen-Yu Weng
- **Venue**: Journal of Manufacturing Systems 74, 2024
- **Summary**: Compares AR interfaces (eye gaze, head gaze, hand ray) for robot programming by demonstration in manufacturing.
- **Uses projection/AR/laser?**: Head-mounted AR
- **Robot**: Industrial robot
- **Policy model**: Programming by demonstration

---

### D3. IntPro — Intention Projection Framework
- **Title**: "When And Where Are You Going? A Mixed-Reality Framework for Human Robot Collaboration"
- **Venue**: OpenReview (workshop)
- **Summary**: Proposes mixed-reality setup for Intention Projection (IntPro) in HRC. Visualizes robot's planned trajectory in AR.
- **Uses projection/AR/laser?**: MR headset
- **Robot**: Collaborative robot
- **Policy model**: N/A

---

### D4. ARTHUR — Authoring HRC with AR
- **Title**: "ARTHUR: authoring human-robot collaboration processes with augmented reality using hybrid user interfaces"
- **Venue**: Virtual Reality (Springer), 2025
- **Summary**: Tool for authoring HRC processes using AR interfaces.
- **Uses projection/AR/laser?**: Head-mounted AR
- **Robot**: Collaborative robots
- **Policy model**: N/A

---

### D5. Leins et al. — AR-assisted Robot Programming (Spatial Ability)
- **Title**: "Investigating the Influence of Spatial Ability in Augmented Reality-assisted Robot Programming"
- **Authors**: Nicolas Leins, Jana Gonnermann-Mueller, Malte Rolf Teichmann, Sebastian Pokutta
- **Venue**: arXiv 2026
- **arXiv ID**: 2602.03544
- **Summary**: Studies how spatial ability affects AR-assisted robot programming.
- **Uses projection/AR/laser?**: AR headset
- **Robot**: Industrial robot
- **Policy model**: N/A

---

### D6. Lee — Robotic Spatial AR with Deep Learning
- **Title**: "Data-Driven Forward Kinematics for Robotic Spatial Augmented Reality: A Deep Learning Framework Using LSTM and Attention"
- **Authors**: Sooyoung Jang, Hanul Yum, Ahyun Lee
- **Venue**: Actuators (MDPI) 2025
- **Summary**: Deep learning framework for precise calibration of projector-robot systems using LSTM+Attention. Enables accurate projection mapping from a robot-mounted projector.
- **Uses projection/AR/laser?**: YES — projector mounted on robot (calibration)
- **Robot**: Robot arm with projector
- **Policy model**: LSTM+Attention for FK calibration

---

## CATEGORY E: Surveys

---

### E1. AR for HRC Survey (MDPI 2024)
- **Title**: "A Survey of Augmented Reality for Human-Robot Collaboration"
- **Venue**: Machines 2024, 12(8), 540
- **Summary**: Comprehensive survey. Documents projector-based approaches by Vogel et al., Chadalavada et al. Notes that projections are used to dynamically indicate cues to human collaborators, with object tracking enabling projection mapping on 3D objects.

### E2. AR+HRC Systematic Review (Sensors 2022)
- **Title**: "Augmented Reality for Human-Robot Collaboration and Cooperation in Industrial Applications: A Systematic Literature Review"
- **Venue**: Sensors 2022, 22(7), 2725

### E3. CHI 2022 AR+Robotics Taxonomy
- **Title**: "Augmented Reality and Robotics: A Survey and Taxonomy for AR-enhanced Human-Robot Interaction and Robotic Interfaces"
- **Venue**: CHI 2022 (U Calgary)
- **Summary**: Comprehensive taxonomy of AR+robotics. Covers projection-based, head-mounted, and handheld AR for robot interaction.

---

## CATEGORY F: Companies / Startups

---

### F1. Lightform (Projection AR)
- **Status**: Founded ~2017, raised $7.8M (Lux Capital). LF2 = first AR projector device.
- **What**: Computer vision-based projection mapping. Scans environment, creates aligned projections. Ex-Disney Imagineering, ex-Microsoft Research (IllumiRoom/RoomAlive) founders.
- **Robot relevance**: Projection AR hardware that COULD be adapted for robot workspaces.
- **URL**: https://lightform.com

### F2. Augmentus (No-code Robot Programming)
- **Status**: Founded 2019, Singapore/US. Raised funding.
- **What**: 3D scanning + automated toolpath generation for industrial robots. No code, no CAD. "From Scan to Robot Path in Minutes."
- **Robot relevance**: Vision-guided robot programming (not projection, but related).
- **URL**: https://www.augmentus.tech

### F3. Wandelbots (Software-Defined Robot Automation)
- **Status**: Founded 2017, Dresden. Significant funding.
- **What**: Robot-agnostic operating system for industrial automation. Simplifies programming.
- **Robot relevance**: Competitor in robot programming simplification space.
- **URL**: https://www.wandelbots.com

### F4. Inbolt GuideNOW (3D Robot Guidance)
- **What**: Real-time 3D vision and AI-driven robot guidance. Dynamically locates workpieces.
- **Robot relevance**: Vision guidance (not projection, but related).

### F5. Fraunhofer IFF (Research Institute)
- **What**: The group behind Vogel et al.'s projector-camera safety system. Ongoing R&D in projection-based HRC for industrial manufacturing.
- **Status**: Active research, multiple follow-up papers through 2017+.

---

## KEY FINDING: THE GAP

**There is NO paper that combines:**
1. Physical projector/laser projecting onto robot workspace
2. WITH a learned manipulation policy (VLA, imitation learning, RL)
3. To either GUIDE data collection OR IMPROVE policy training

The existing work falls into two non-overlapping camps:

| Camp | What they do | What they lack |
|------|-------------|----------------|
| **Projection/HRI** (Category A) | Project safety zones, intent, instructions onto workspace | No learned policies, no VLA, no imitation learning |
| **Robot Learning** (Categories B, C) | VLA, affordance maps, generative augmentation | All computational/digital; no physical projection |

**Closest approaches that partially bridge the gap:**
- **RT-Sketch / RT-Trajectory** (Google): Use visual sketches/trajectories as intermediate representations, but these are digital overlays, not physical projections
- **VoxPoser**: Creates 3D affordance maps computationally, not projected physically
- **LARS**: Projects visual stimuli for swarm robots, but for stigmergy communication, not learned manipulation
- **Laser pointer papers** (A11-A14): Use laser for task specification (point at object), but with hard-coded control, not learned policies

**This gap is a potential research opportunity**: Using a physical projector to project task-relevant visual cues (e.g., target positions, grasp points, trajectory hints) onto the real workspace to either:
- (a) Guide human demonstrators during data collection (reducing demonstration variance)
- (b) Provide additional visual input to the robot's camera that encodes task information
- (c) Create physical "affordance visualizations" that bridge the human-robot understanding gap during teaching

---

## Summary Statistics

| Category | Count | Physical Projection? | Learned Policy? |
|----------|-------|---------------------|-----------------|
| A: Physical projection + robot | 17 papers | YES | NO |
| B: Computational affordance maps | 7 papers | NO | YES |
| C: Digital visual augmentation | 5 papers | NO | YES |
| D: Head-mounted AR + robot | 6 papers | NO (headset) | NO |
| E: Surveys | 3 papers | Mixed | N/A |
| F: Companies | 5 entries | Some | NO |

**Total unique papers found: ~38**
**Papers combining physical projection WITH learned policy: 0**
