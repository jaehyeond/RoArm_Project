# Robot Hardware Landscape: Intelligence Report (Early 2026)

Agent: BODY-HUMINT | Compiled: 2026-03-07
Focus: Physical robot platforms, manufacturers, pricing, software ecosystems, VLA readiness.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Humanoid Robots](#2-humanoid-robots)
3. [Research-Grade Robot Arms](#3-research-grade-robot-arms)
4. [Low-Cost / DIY Robot Arms](#4-low-cost--diy-robot-arms)
5. [Dexterous Hands](#5-dexterous-hands)
6. [Mobile Platforms](#6-mobile-platforms)
7. [Quadruped Robots](#7-quadruped-robots)
8. [LeRobot Ecosystem Map](#8-lerobot-ecosystem-map)
9. [VLA-Ready Hardware Comparison](#9-vla-ready-hardware-comparison)
10. [Where RoArm M3 Fits](#10-where-roarm-m3-fits)
11. [Key Software Ecosystems](#11-key-software-ecosystems)
12. [What Top Labs Actually Use](#12-what-top-labs-actually-use)
13. [Convergence Trends](#13-convergence-trends)
14. [Cheapest VLA Research Setup](#14-cheapest-vla-research-setup)
15. [Korean Robot Companies](#15-korean-robot-companies)
16. [Chinese Robot Companies](#16-chinese-robot-companies)

---

## 1. Executive Summary

The robot hardware landscape in early 2026 is defined by three macro-trends:

1. **Humanoid Gold Rush**: At least 15 companies are racing to deploy humanoid robots. Tesla, Figure, 1X, Unitree, Agility, Apptronik, and Fourier are the most capitalized. Valuations are astronomical (Figure: $39B), but commercial deployments remain limited to controlled warehouse/factory settings.

2. **Democratization of Research Arms**: The SO-100/SO-101 (HuggingFace/LeRobot standard) and Koch v1.1 have created a sub-$500 research arm category. This is where VLA research is actually happening at scale across the community.

3. **Software Eats Hardware**: LeRobot has emerged as the de-facto standard for robot learning research, supporting 10+ hardware platforms with a unified API. Physical Intelligence's Pi0, NVIDIA's GR00T, and HuggingFace's SmolVLA all run through or interoperate with it. The robot you use matters less than the software ecosystem it plugs into.

**Bottom line for our project**: RoArm M3 Pro ($200-300 range, 6-DOF, ESP32) sits in a unique but awkward position -- more capable than SO-100 but not integrated into LeRobot natively. Our custom SmolVLA pipeline bridges this gap, but community support is thin.

---

## 2. Humanoid Robots

### Master Comparison Table

| Robot | Company | Country | DOF | Height | Weight | Price (est.) | Status (early 2026) | Key Backer |
|-------|---------|---------|-----|--------|--------|-------------|---------------------|------------|
| **Optimus (Gen 2/3)** | Tesla | US | ~28 body + 22 DOF hands (Gen 3) | 173cm | 57kg | ~$30K (target) | Limited factory deployment | Self-funded |
| **Figure 02** | Figure AI | US | 16 DOF hands + body | ~170cm | N/A | N/A (RaaS model) | BMW plant testing | Bezos, Microsoft, NVIDIA ($39B val.) |
| **Figure 03** | Figure AI | US | 35 DOF total | ~170cm | N/A | N/A | Announced Oct 2025 | Same |
| **Helix** | Figure AI | US | 35 DOF, dual-GPU | ~170cm | N/A | N/A | VLA-native (Helix VLA model) | Same |
| **NEO Gamma** | 1X Technologies | Norway/US | N/A | ~170cm | N/A | $20K (pre-order) / $499/mo | Pre-orders opened Oct 2025 | OpenAI fund, Samsung, EQT |
| **EVE** | 1X Technologies | Norway/US | Wheeled + arms | ~160cm | N/A | N/A (B2B only) | Deployed for logistics | Same |
| **Atlas (Electric)** | Boston Dynamics | US (Hyundai) | 28+ DOF | 150cm | ~80kg | Not for sale | R&D, demos with Hyundai | Hyundai Motor Group |
| **Digit** | Agility Robotics | US | Bipedal + arms | ~175cm | N/A | RaaS model | Amazon warehouse pilot, Spanx factory | Amazon, GXO |
| **Apollo** | Apptronik | US | N/A | ~173cm | N/A | N/A | Pilot deployments | NASA heritage (Valkyrie team) |
| **H1** | Unitree | China | 20+ DOF | ~180cm | N/A | ~$90K (list) | Shipping, research | HongShan, Shunwei |
| **G1** | Unitree | China | 23-43 DOF | ~127cm | ~35kg | $16,000 | Mass production | Same |
| **H2** | Unitree | China | 31 DOF | 180cm | N/A | N/A | Announced 2025 | Same |
| **R1** | Unitree | China | 20-26 DOF | <123cm | ~29kg | From $4,900 | Announced 2025 | Same |
| **GR-1** | Fourier | China | N/A | 165cm | N/A | N/A | Small-batch delivery to universities | IDG Capital, Saudi Aramco |
| **GR-2** | Fourier | China | N/A | N/A | N/A | N/A | In development | Same |
| **HUBO-2 Plus** | Rainbow Robotics | Korea | 40+ DOF | ~130cm | N/A | ~$400K (research) | 15 units sold globally | KAIST spinoff |

### Key Observations

**Tesla Optimus**: Still primarily teleoperated in demos despite 4 years of development. Gen 3 hands jumped to 22 DOF (from 11 DOF in Gen 2). Head of program (Milan Kovac) resigned June 2025, replaced by Autopilot lead Ashok Elluswamy -- signaling a pivot to more AI-driven control. Target price $30K would be transformative if achieved but timeline remains unclear.

**Figure AI**: The most aggressively funded humanoid startup ($39B valuation as of Sept 2025). Ended OpenAI partnership in 2025, developing own VLA (Helix VLA). BotQ factory aiming for 12K units/year. Figure 03 (Oct 2025) features tactile sensors detecting 3-gram forces and embedded palm cameras. The Helix architecture is notable: System 2 (high-level VLA planning, 7-9 Hz) + System 1 (low-level motor control, 200 Hz).

**1X Technologies (NEO)**: The most consumer-facing play -- $20K pre-order or $499/month subscription for home use. Honest about current limitations: most tasks still teleoperated (VR headset), with plan to collect training data from early adopters to improve autonomy. This is effectively "Tesla FSD for humanoids" -- ship hardware early, collect data, iterate.

**Unitree**: The price disruptor. G1 at $16K is the cheapest humanoid available for purchase. New R1 at $4,900 is a smaller wheeled-arm hybrid. H2 at 31 DOF with 2070 TOPS compute chip positions it for VLA deployment. Unitree is the only company shipping humanoids at scale to individual researchers.

**Agility (Digit)**: First commercial Robot-as-a-Service (RaaS) humanoid contract with GXO Logistics. Focused on warehouse tote-moving, not general-purpose. Amazon pilot is the highest-profile commercial humanoid deployment.

**Boston Dynamics (Atlas Electric)**: Revealed April 2024 after retiring hydraulic Atlas. Not for sale -- BD is focused on commercial Spot/Stretch ecosystem. Atlas Electric demonstrates superhuman joint range of motion. Owned by Hyundai Motor Group.

**Fourier (GR-1/GR-2)**: Originally a rehabilitation robotics company (exoskeletons). GR-1 delivered to universities in small quantities. Chinese government interest (Xi Jinping inspection tour, Dec 2023). Background in medical robotics gives unique angle on human-safe design.

---

## 3. Research-Grade Robot Arms

### Master Comparison Table

| Robot Arm | Manufacturer | DOF | Payload | Reach | Price | Software | Open-Source HW | VLA Papers |
|-----------|-------------|-----|---------|-------|-------|----------|---------------|------------|
| **Franka FR3** | Franka Robotics (now Agile Robots) | 7 | 3kg | 855mm | ~$30K-40K | libfranka, ROS/ROS2, MoveIt | URDF yes | RT-2, SayCan, many |
| **UR5e** | Universal Robots (Teradyne) | 6 | 5kg | 850mm | ~$25K-35K | URScript, ROS/ROS2, UR+ ecosystem | URDF yes | Moderate |
| **UR10e** | Universal Robots | 6 | 12.5kg | 1300mm | ~$35K-45K | Same as UR5e | URDF yes | Some |
| **UR3e** | Universal Robots | 6 | 3kg | 500mm | ~$20K-25K | Same as UR5e | URDF yes | Some |
| **KUKA iiwa 7/14** | KUKA (Midea Group) | 7 | 7/14kg | 800/820mm | ~$50K-80K | Sunrise, ROS | URDF yes | DLR, industrial |
| **xArm 6** | UFactory | 6 | 5kg | 700mm | $5,299 | Python/C++ SDK, ROS/ROS2 | URDF yes | Growing |
| **xArm 7** | UFactory | 7 | 3.5kg | 700mm | $5,494 | Same | URDF yes | Growing |
| **ViperX 300** | Trossen Robotics | 6 | ~0.75kg | 700mm | ~$5,000 | Dynamixel SDK, ROS/ROS2 | URDF, CAD | ALOHA (Stanford) |
| **WidowX 250** | Trossen Robotics | 6 | ~0.4kg | 525mm | ~$3,500 | Same | URDF, CAD | Bridge V2, RT-X |

### Key Analysis

**Franka (now part of Agile Robots)**: The most important development -- Franka Emika had financial difficulties and was acquired by Agile Robots in 2023. Agile Robots (a German-Chinese company) now operates it. The Franka Panda/FR3 remains THE reference arm for manipulation research. It appears in more robotics papers than any other arm. 7-DOF with torque sensing at every joint. The acquisition means long-term support stability is uncertain -- researchers are hedging by also supporting xArm and UR.

**Universal Robots**: Market leader with 40-50% cobot market share as of 2022. Revenue $311M (2021). UR+ ecosystem with 400+ certified peripherals. Most used in industry; less common in VLA research (mainly used in NVIDIA labs). Extremely reliable but expensive for pure research use.

**UFactory xArm**: The "sweet spot" for research -- industrial quality at 1/5 the price of Franka/UR. Open Python/C++ SDK, ROS2 support. Growing adoption in VLA research labs. The 0.1mm repeatability is genuinely impressive at this price. Harmonic drives + DC servomotors (same technology as Franka, just cheaper). This is where the market is shifting for budget-conscious research labs.

**Trossen ViperX/WidowX**: Built on Dynamixel servos. The ViperX 300 is the arm used in the original ALOHA system (Stanford). WidowX 250 is used in Bridge V2 dataset (one of the largest robot manipulation datasets). Strong in the LeRobot ecosystem. Lower precision than xArm but deeply embedded in the research community.

**KUKA iiwa**: Premium industrial arm. 7-DOF with excellent torque sensing. Owned by Midea Group (China) since 2016. Mainly used in German research labs (DLR) and industrial applications. Too expensive for most academic VLA research.

---

## 4. Low-Cost / DIY Robot Arms

### Master Comparison Table

| Robot | Creator | DOF | Price (BOM) | Motors | Controller | LeRobot Native | VLA Tested |
|-------|---------|-----|-------------|--------|------------|---------------|------------|
| **SO-100** | HuggingFace / LeRobot | 6 | ~$110-200 | Feetech STS3215 | Raspberry Pi / USB | YES (reference) | SmolVLA, ACT, DP |
| **SO-101** | HuggingFace / LeRobot | 6 | ~$110-200 | Feetech STS3215 | Same | YES (reference) | SmolVLA, ACT, DP |
| **Koch v1.1** | Jess Moss / LeRobot | 6 | ~$200-300 | Dynamixel XL330/430 | USB2Dynamixel | YES | ACT, DP |
| **RoArm M3 Pro** | Waveshare | 6 (5+1) | ~$200-300 | TTL bus servos | ESP32 | NO (custom integration) | SmolVLA (our project) |
| **ALOHA (low-cost)** | Stanford (Zipeng Fu) | 2x6 (bimanual) | ~$10K-20K | Dynamixel | Custom | Partial | ACT (native) |
| **Mobile ALOHA** | Stanford (Zipeng Fu) | 2x6 + mobile base | ~$25K-30K | Dynamixel + base | Custom | Partial | ACT + co-training |
| **OpenARM** | Community | 6 | ~$150-250 | Feetech | RPi | YES | Limited |
| **HopeJR** | Community | 6 | ~$150 | Feetech | RPi | YES | Limited |
| **LeKiwi** | LeRobot | Mobile base | ~$200-300 | Feetech | RPi | YES | SmolVLA |

### LeRobot Officially Supported Hardware (as of v0.4.4+)

From the LeRobot README and hardware integration guide:
- **SO100** / **SO101** -- Reference arms (Feetech STS3215)
- **Koch** -- Dynamixel-based
- **LeKiwi** -- Mobile base platform
- **HopeJR** -- Community arm
- **OMX** -- Community arm
- **EarthRover** -- Mobile platform
- **OpenARM** -- Community arm
- **Reachy2** (Pollen Robotics) -- Full humanoid
- **Unitree G1** -- Humanoid
- **Gamepad / Keyboard / Phone** -- Teleoperation devices

### Motor Ecosystem

| Motor Family | Protocol | Key Models | Used By | Price/motor |
|-------------|----------|------------|---------|-------------|
| **Feetech STS/SMS** | TTL (Protocol 0) | STS3215, STS3250, SM8512BL | SO-100/101, OpenARM, HopeJR | $8-25 |
| **Feetech SCS** | TTL (Protocol 1) | SCS0009 | Grippers | $5-10 |
| **Dynamixel** | TTL (Protocol 2.0) | XL330, XL430, XM430, XM540, XC430 | Koch, ALOHA, ViperX, WidowX | $25-250 |
| **Waveshare TTL bus** | Custom TTL | Unnamed (12-bit encoder) | RoArm M3 | Integrated |

### Key Observations

**SO-100/SO-101 is the new standard**: HuggingFace designed these as the "Hello World" of robot arms. 3D-printed structure + Feetech STS3215 servos + Raspberry Pi. Total BOM under $200 for a leader-follower pair. This is what SmolVLA was pretrained and validated on. The SO-100 is to robot learning what MNIST was to deep learning -- the universal benchmark.

**Koch v1.1**: Dynamixel-based alternative, slightly higher quality motors. Created by Jess Moss (LeRobot team member). Better torque and precision than SO-100 but more expensive motors.

**ALOHA**: Stanford's bimanual system using two ViperX 300 arms. The ACT policy (Action Chunking with Transformers) was developed on this platform. Low-cost ALOHA uses cheaper components but still runs $10K+. Mobile ALOHA adds a mobile base for whole-body teleoperation. 50 demonstrations per task is the standard.

**RoArm M3 Pro (our robot)**: Waveshare's offering. ESP32 MCU, dual-drive shoulder, 360-degree base, 1m workspace diameter. Explicitly mentions LeRobot compatibility on product page. However, NOT natively integrated in LeRobot codebase -- our project is one of the first to run VLA (SmolVLA) on this platform. The ESP32 controller with custom TTL protocol is the main integration barrier.

---

## 5. Dexterous Hands

### Master Comparison Table

| Hand | Organization | DOF | Fingers | Tactile | Price | Software | Open-Source |
|------|-------------|-----|---------|---------|-------|----------|-------------|
| **Shadow Dexterous Hand** | Shadow Robot Co. (UK) | 20 | 5 | Hall effect + optional BioTac | ~$100K-200K | ROS, ROS2 | URDF, ROS packages |
| **Allegro Hand** | Wonik Robotics (Korea) | 16 | 4 | Joint torque | ~$15K-25K | ROS, SDK | URDF |
| **LEAP Hand** | CMU (Deepak Pathak) | 16 | 4 | None (vision-based) | ~$2,000 (BOM) | Python, ROS | Fully open (CAD, code, BOM) |
| **Ability Hand** | PSYONIC | 6+ DOF | 5 | Pressure sensors (vibration feedback) | ~$10K-20K (prosthetic) | USB-C, EMG compatible | No |
| **Figure 03 Hand** | Figure AI | 16+ DOF per hand | 5 | 3-gram force tactile + palm cameras | N/A | Proprietary (Helix) | No |
| **Tesla Gen 3 Hand** | Tesla | 22 DOF | 5 | N/A | N/A | Proprietary | No |
| **DexHand (various)** | Multiple labs | Varies | Varies | Varies | DIY | ROS | Some |

### Key Observations

**LEAP Hand (CMU)**: The most important development for research. Published at RSS 2023 by Kenneth Shaw, Ananye Agarwal, and Deepak Pathak. Fully open-source: 3D-printed structure, off-the-shelf motors, ~$2K BOM. 16 DOF, 4 fingers. Designed explicitly for robot learning (sim-to-real RL). This is the "SO-100 of dexterous hands."

**Shadow Dexterous Hand**: The gold standard for 20+ years. 20 DOF, 24 joints, both pneumatic and electric models. Used by OpenAI for the famous Rubik's cube solving demo. NASA, Carnegie Mellon, Bielefeld University are users. ROS-based software. Price ($100K+) limits it to well-funded labs.

**Allegro Hand**: Korean-made (Wonik Robotics). 16 DOF, 4 fingers. The most popular research hand after Shadow due to more reasonable pricing (~$15-25K). Strong ROS support.

**PSYONIC Ability Hand**: Originally a prosthetic hand (with FDA clearance). 5 fingers with pressure sensing and vibration feedback. 490g, USB-C charging. Being explored for robotics applications. 32 grip patterns. IP64 rated. The prosthetics-to-robotics pipeline is an interesting trend.

**Trend**: The field is bifurcating -- expensive precision hands (Shadow, Allegro) for labs that need them, and cheap open-source hands (LEAP) for the ML research community that needs scale over precision. The humanoid companies (Figure, Tesla) are developing proprietary hands with integrated tactile sensing, which may eventually become available as standalone products.

---

## 6. Mobile Platforms

### Master Comparison Table

| Platform | Company | Type | DOF (manip) | Price | Software | Key Use Case |
|----------|---------|------|-------------|-------|----------|-------------|
| **Stretch 3** | Hello Robot | Mobile manipulator | 7 (arm) + 2 (head) + 1 (gripper) + 2 (base) | $24,950 | ROS 2, Python SDK | Home assistive, Embodied AI research |
| **TIAGo** | PAL Robotics (Spain) | Mobile manipulator | 7 (arm) + head + base | ~$50K-80K | ROS/ROS2 | Service robotics, research |
| **PR2** | Willow Garage (discontinued) | Dual-arm mobile | 2x7 (arms) + base | N/A (discontinued) | ROS | Legacy research platform |
| **Husky** | Clearpath (Rockwell) | UGV base | 0 (base only) | ~$20K-30K | ROS/ROS2 | Outdoor research |
| **Jackal** | Clearpath | UGV base | 0 (base only) | ~$10K-15K | ROS/ROS2 | Indoor/outdoor research |
| **Dingo** | Clearpath | Indoor mobile base | 0 (base only) | ~$8K-12K | ROS/ROS2 | Indoor research |
| **Mobile ALOHA** | Stanford | Mobile bimanual | 2x6 (arms) + base | ~$25K-30K | Custom + ROS | Whole-body teleoperation |
| **LeKiwi** | LeRobot | Mobile base | 0 (add SO-100) | ~$200-300 | LeRobot native | Low-cost mobile research |

### Key Observations

**Hello Robot Stretch 3**: The most successful dedicated mobile manipulator for research. $24,950 all-inclusive (arm, gripper, RGBD cameras, computer, lidar, speakers). 24.5 kg total weight. Huge Embodied AI community -- included in Open-X Embodiment and RT-X datasets (Google DeepMind). The telescoping arm design is unique: compact (33x34cm footprint) but reaches floor-to-cabinet. 7 DOF manipulator. ROS 2 and Python SDK. Web teleop, dex teleop kit available. This is arguably the most "ready for VLA" mobile platform.

**PAL Robotics TIAGo**: European standard for service robotics research. More expensive than Stretch, larger, stronger arm. Used extensively in EU-funded research projects.

**Clearpath Robotics**: Acquired by Rockwell Automation. The Husky/Jackal/Dingo are mobile BASE platforms (no arm) -- you mount your own manipulator. Standard in outdoor and agricultural robotics research. ROS/ROS2 native. Took over PR2 support when Willow Garage shut down.

**Mobile ALOHA**: Stanford's research platform for whole-body bimanual mobile manipulation. Two ViperX 300 arms on a mobile base with whole-body teleoperation. 50 demonstrations per task with co-training achieves up to 90% success. Not commercially available -- you build it yourself.

---

## 7. Quadruped Robots

### Comparison Table

| Robot | Company | Price | Weight | Payload | Key Feature |
|-------|---------|-------|--------|---------|-------------|
| **Spot** | Boston Dynamics | ~$75K | 32kg | 14kg | Gold standard, Spot SDK, arm option |
| **Go2** | Unitree | ~$1,600-2,800 | 15kg | 8kg | Consumer-grade, LLM integration |
| **Go2-W** | Unitree | Similar | Similar | Similar | Wheeled-leg variant |
| **B2** | Unitree | ~$15K-25K | ~60kg | 40kg | Industrial, high payload |
| **B2-W** | Unitree | Similar | Similar | Similar | Wheeled-leg industrial |
| **A2** | Unitree | N/A | ~42kg | Heavy | 5h/20km endurance, dual LiDAR |

### Key Observations

**Spot** remains the industry standard but at a premium price. Spot SDK is well-documented, and an arm accessory is available. Owned by Hyundai Motor Group (same as Boston Dynamics).

**Unitree Go2** disrupted the market at ~$1,600 for the base model. It has become the most popular quadruped in research and hobbyist communities. Built-in 4D LiDAR and LLM (GPT) integration in premium models. Military use has been reported (USMC training, Chinese military drills), though Unitree denies direct military sales.

---

## 8. LeRobot Ecosystem Map

LeRobot (HuggingFace) has become the de-facto standard for robot learning research. Here is the full ecosystem as of v0.4.4+:

### Supported Policies

| Category | Models |
|----------|--------|
| Imitation Learning | ACT, Diffusion Policy, VQ-BeT |
| Reinforcement Learning | HIL-SERL, TDMPC |
| VLA Models | **Pi0Fast**, **Pi0.5**, **GR00T N1.5**, **SmolVLA**, **XVLA** |

### Supported Hardware

| Category | Platforms |
|----------|----------|
| Arms (leader-follower) | SO-100, SO-101, Koch v1.1, OpenARM, HopeJR, OMX |
| Mobile | LeKiwi, EarthRover |
| Humanoid | Reachy2 (Pollen), Unitree G1 |
| Teleop devices | Gamepad, Keyboard, Phone |

### Motor Support

| Family | Supported Models |
|--------|-----------------|
| Feetech (Protocol 0) | STS3215, STS3250, SM8512BL |
| Feetech (Protocol 1) | SCS0009 |
| Dynamixel (Protocol 2.0) | XL330-M077, XL330-M288, XL430-W250, XM430-W350, XM540-W270, XC430-W150 |

### Integration Path for New Robots

LeRobot provides a `Robot` base class. To integrate a new robot:
1. Subclass `Robot` and `RobotConfig`
2. Define `observation_features` and `action_features`
3. Implement `connect()`, `disconnect()`, `get_observation()`, `send_action()`
4. If using Feetech or Dynamixel motors, use built-in `FeetechMotorsBus` or `DynamixelMotorsBus`
5. For other motors (like RoArm M3's ESP32 TTL), write a custom communication wrapper

### Dataset Format

LeRobotDataset v3: Parquet files (state/action) + MP4 video (cameras). Hosted on HuggingFace Hub. Thousands of datasets available for pre-training.

---

## 9. VLA-Ready Hardware Comparison

The critical question: which hardware can actually run Vision-Language-Action models in a closed loop?

### VLA Deployment Requirements

1. Camera (RGB, ideally 720p+, 30fps)
2. Joint state feedback (position at minimum, torque ideal)
3. Position or velocity control interface
4. < 200ms inference latency acceptable (most VLAs: 50-150ms)
5. Software bridge to policy framework (LeRobot, custom)

### VLA Readiness Matrix

| Platform | Camera | Joint Feedback | Control | Inference Latency OK | LeRobot Native | VLA Demonstrated | Cost |
|----------|--------|---------------|---------|---------------------|----------------|-----------------|------|
| SO-100/101 | USB/RPi cam | Feetech position | Position | Yes | YES | SmolVLA, ACT, DP | $200 |
| Koch v1.1 | USB cam | Dynamixel position | Position | Yes | YES | ACT, DP | $300 |
| RoArm M3 Pro | External (Kinect) | TTL position | Position | Yes | NO (custom) | SmolVLA (our work) | $250 |
| xArm 6 | External | Position + torque | Position/velocity | Yes | NO | Pi0, ACT (labs) | $5,300 |
| Franka FR3 | External | Position + torque | Position/torque | Yes | NO | RT-2, many | $35,000 |
| UR5e | External | Position + torque | Position/velocity | Yes | NO | Some | $30,000 |
| ViperX 300 | External | Dynamixel position | Position | Yes | Partial | ACT (ALOHA) | $5,000 |
| Stretch 3 | Built-in RGBD x2 | Position + force | Position | Yes | NO | RT-X, Open-X | $25,000 |
| Unitree G1 | Built-in | Position + torque | Position | Yes | YES | GR00T | $16,000 |

### Key Insight

The cheapest fully VLA-ready setup today is: **SO-100 leader-follower pair + USB camera + LeRobot** at approximately $200-400 total. This is what SmolVLA was pretrained on (community_dataset_v1: 128 datasets, 11,132 episodes, ALL on SO-100 platform).

---

## 10. Where RoArm M3 Fits

### Positioning Analysis

```
Price Scale:
$100 -------- $500 -------- $5K -------- $30K -------- $100K+
  |              |              |              |              |
SO-100        RoArm M3       xArm 6        Franka FR3     Shadow+Franka
Koch v1.1     (Our robot)    ViperX 300     UR5e           (Top tier)
LeKiwi                       WidowX 250
```

### RoArm M3 vs Competitors

| Feature | SO-100 | RoArm M3 Pro | xArm 6 |
|---------|--------|-------------|--------|
| DOF | 6 | 6 (5+1) | 6 |
| Payload at reach | ~50g | 200g (at 0.5m) | 5kg |
| Workspace diameter | ~0.5m | 1.0m | 1.4m |
| Repeatability | ~1-2mm | 0.088 deg (joint) | 0.1mm |
| Motor type | Feetech STS3215 | TTL bus servo (12-bit) | Harmonic drive + DC servo |
| Controller | RPi / USB | ESP32 (WiFi/BLE/UART) | Arm controller + SDK |
| ROS support | Via LeRobot | ROS2 compatible | Native ROS/ROS2 |
| LeRobot native | YES | NO | NO |
| Price | ~$100-200 | ~$200-300 | $5,299 |
| VLA track record | Extensive (SmolVLA pretrained) | Our project only | Growing |

### Strengths of RoArm M3

1. **Dual-drive shoulder**: Doubled torque at the most loaded joint -- SO-100 lacks this
2. **1m workspace**: 2x the SO-100 workspace -- more practical for real tasks
3. **ESP32 versatility**: WiFi, BLE, UART, I2C -- more connectivity than any other arm in its class
4. **Waveshare support**: Active product with tutorials, wiki, firmware updates
5. **Explicit LeRobot mention**: Waveshare product page references LeRobot compatibility
6. **12-bit encoder**: 0.088 degree repositioning accuracy -- much better than STS3215

### Weaknesses of RoArm M3

1. **No native LeRobot integration**: The #1 barrier. Custom pipeline required.
2. **ESP32 communication overhead**: Serial JSON protocol adds latency vs direct motor bus
3. **SDK quirks**: print(data) spam, intermittent None returns, background thread errors
4. **No torque sensing**: Position-only feedback. Cannot do force-controlled tasks.
5. **Small community**: We may be the only group running VLA on this platform.
6. **OOD for SmolVLA**: Pretrained exclusively on SO-100. RoArm M3 = out-of-distribution embodiment.

### Recommendation

RoArm M3 Pro is a **competent but lonely** platform. It is objectively better hardware than SO-100 (bigger workspace, better encoders, dual-drive shoulder) but has 100x less community support. The optimal strategy is:

1. **Continue using RoArm M3 for our VLA research** (sunk cost + working pipeline)
2. **Consider contributing a LeRobot integration** (would benefit us + community)
3. **Get an SO-100 pair as a reference platform** (~$200) to validate against community baselines
4. **Do not switch to xArm/Franka** unless budget allows ($5K-35K)

---

## 11. Key Software Ecosystems

### Framework Landscape

| Framework | Focus | Robot Support | Policy Support | Org |
|-----------|-------|--------------|----------------|-----|
| **LeRobot** | End-to-end robot learning | 10+ platforms | ACT, DP, SmolVLA, Pi0, GR00T | HuggingFace |
| **ROS 2** | General robotics middleware | Universal | N/A (infrastructure) | Open Robotics |
| **MoveIt 2** | Motion planning | UR, Franka, many | Classical planning | PickNik |
| **Isaac Lab** | GPU-parallel sim + RL | URDF/USD any | RL (PPO, SAC) | NVIDIA |
| **MuJoCo** | Physics simulation | MJCF/URDF any | Any (via API) | Google DeepMind |
| **ManiSkill 3** | GPU-parallel manipulation | SAPIEN robots | Any | Hillbot/SAPIEN |
| **Robosuite** | Manipulation benchmarks | Franka, Sawyer | IL, RL | Stanford |
| **DROID** | Distributed data collection | Franka (primary) | Any | Toyota Research |
| **Robot Control Stack** | Lean sim-to-real | Franka, UR5e, xArm, SO101 | Any | ICRA 2026 |

### The Software Stack for VLA Research (2026)

```
Layer 4: VLA Model         [SmolVLA | Pi0 | GR00T | OpenVLA]
                                    |
Layer 3: Training Framework [LeRobot | custom PyTorch]
                                    |
Layer 2: Data Format       [LeRobotDataset v3 (Parquet + MP4)]
                                    |
Layer 1: Robot Interface   [LeRobot Robot class | ROS 2 | custom SDK]
                                    |
Layer 0: Hardware          [SO-100 | Koch | xArm | Franka | RoArm M3 | ...]
```

LeRobot is becoming the "PyTorch of robotics" -- the framework that connects everything. Physical Intelligence (Pi0), NVIDIA (GR00T), and HuggingFace (SmolVLA) all publish LeRobot-compatible models.

---

## 12. What Top Labs Actually Use

### Hardware by Lab

| Lab / Company | Primary Hardware | Backup/Alt | Budget Tier |
|---------------|-----------------|------------|-------------|
| **Google DeepMind** | Custom RT-X platforms, Everyday Robots (retired) | UR5, Stretch | Unlimited |
| **Physical Intelligence (Pi0)** | Franka FR3 (primary), UR5e, multiple | Various | High |
| **Stanford REAL Lab (ACT/ALOHA)** | ViperX 300 (ALOHA), Mobile ALOHA | Franka | Medium-High |
| **Stanford IRIS Lab** | Franka FR3 | xArm | High |
| **CMU (LEAP Hand, ReinFlow)** | LEAP Hand, Franka FR3 | Allegro | Medium-High |
| **HuggingFace (SmolVLA)** | SO-100/SO-101 | Community hardware | Low |
| **NVIDIA (GR00T)** | Unitree G1, custom humanoids | Franka, UR | Unlimited |
| **Toyota Research (Diffusion Policy)** | Franka FR3 | UR5 | High |
| **Columbia (Diffusion Policy)** | Franka FR3 | UR | High |
| **UC Berkeley (SERL)** | Franka FR3 | - | High |
| **MIT (various)** | Franka FR3, Spot | UR, Allegro | High |
| **Tsinghua/Shanghai AI Lab** | Various Chinese arms, Franka | xArm | Medium |

### Key Takeaway

**Franka FR3 is the most popular arm in top research labs by a wide margin.** It appears in papers from Physical Intelligence, Stanford, CMU, Toyota Research, Columbia, UC Berkeley, and MIT. The torque sensing at every joint and excellent ROS integration make it the reference platform.

**For VLA specifically, the trend is bifurcating:**
- **Big labs**: Franka FR3 + multiple cameras + extensive compute
- **Community/democratized**: SO-100/101 + USB camera + consumer GPU (via LeRobot)

Our RoArm M3 project sits between these two worlds.

---

## 13. Convergence Trends

### Hardware Standardization

1. **Motor convergence**: The research community is converging on two motor families:
   - **Feetech STS3215** (low-cost): SO-100/101, OpenARM, HopeJR
   - **Dynamixel XL/XM series** (mid-range): Koch, ALOHA, ViperX, WidowX
   - Everything else (including RoArm M3's custom TTL servos) is non-standard

2. **Camera convergence**: Moving toward standard USB cameras or Intel RealSense for research. Azure Kinect (our camera) was discontinued by Microsoft but remains capable. The VLA community mostly uses simple RGB webcams (no depth needed for current VLAs).

3. **Form factor convergence**: 6-DOF tabletop arms with parallel gripper is the standard for manipulation research. Leader-follower teleoperation for data collection.

4. **Compute convergence**: NVIDIA Jetson (edge) or RTX consumer GPUs for inference. Training on cloud/workstation with A100/H100. RTX 4090 is the sweet spot for VLA fine-tuning.

### Software Convergence

1. **LeRobot as the hub**: Training, data, evaluation, hardware interface -- all converging into LeRobot
2. **LeRobotDataset v3**: Becoming the standard data format (Parquet + MP4)
3. **HuggingFace Hub**: Centralized hosting for robot datasets and trained policies
4. **VLA as the model architecture**: SmolVLA, Pi0, GR00T all share similar architecture patterns (VLM backbone + action head)

### What is NOT converging

1. **Humanoid hardware**: Every company has a radically different design. No standard.
2. **Hand/gripper design**: Parallel grippers, soft grippers, multi-fingered hands -- no convergence
3. **Mobile base**: Every platform is different (differential drive, omnidirectional, legged)
4. **Communication protocols**: ROS 2, custom serial, HTTP/WebSocket, CAN bus -- fragmented

---

## 14. Cheapest VLA Research Setup

### Tier 1: Minimum Viable ($200-400)

| Component | Option | Cost |
|-----------|--------|------|
| Robot arm (follower) | SO-100 (3D printed + Feetech STS3215 x5) | ~$100 |
| Robot arm (leader) | SO-100 leader kit | ~$100 |
| Camera | USB webcam (720p+) | $20-50 |
| Compute (training) | Google Colab Pro+ or existing GPU | $50/mo or $0 |
| Compute (inference) | Raspberry Pi 5 or laptop | $80 or $0 |
| Software | LeRobot (free, open-source) | $0 |
| **Total** | | **$200-400** |

This is the setup SmolVLA was pretrained on. It works.

### Tier 2: Comfortable Research ($500-1,500)

| Component | Option | Cost |
|-----------|--------|------|
| Robot arm | Koch v1.1 (Dynamixel, better quality) | ~$300-500 |
| Camera | Intel RealSense D435i (RGB-D) | ~$300 |
| Compute | RTX 4060 Ti laptop or desktop | ~$500-800 |
| Software | LeRobot | $0 |
| **Total** | | **$1,100-1,600** |

### Tier 3: Our Setup (~$2,000-3,000)

| Component | Option | Cost |
|-----------|--------|------|
| Robot arm | RoArm M3 Pro (Waveshare) | ~$250 |
| Camera | Azure Kinect DK (720p RGB + depth) | ~$400 (discontinued, secondhand) |
| Compute | RTX 4090 Laptop (16GB VRAM) | ~$2,000+ |
| Software | LeRobot + custom pipeline | $0 |
| **Total** | | **~$2,650+** |

Better compute than Tier 1/2, but non-standard hardware creates integration overhead.

### Tier 4: Lab Standard ($30,000-50,000)

| Component | Option | Cost |
|-----------|--------|------|
| Robot arm | Franka FR3 | ~$35,000 |
| Camera | 2-3x RealSense D435i | ~$900 |
| Compute | Workstation with A6000/A100 | ~$10,000+ |
| Software | LeRobot + ROS 2 + MoveIt | $0 |
| **Total** | | **~$46,000+** |

This is what Physical Intelligence, Stanford, CMU, etc. use.

---

## 15. Korean Robot Companies

| Company | Product | Type | Notable |
|---------|---------|------|---------|
| **Rainbow Robotics** | HUBO-2 Plus, cobots | Humanoid, cobots | KAIST spinoff, DARPA Robotics Challenge winner (2015), publicly traded (KRX: 277810) |
| **Wonik Robotics** | Allegro Hand | Dexterous hand | Most popular research hand after Shadow, ~$15-25K |
| **Doosan Robotics** | M/H/A series cobots | Collaborative arms | Publicly traded, industrial focus |
| **Hyundai / Boston Dynamics** | Atlas, Spot | Humanoid, quadruped | Hyundai owns Boston Dynamics since 2020 |
| **Samsung** | Various research robots | Research | Samsung NEXT invested in 1X Technologies |
| **KAIST HUBO Lab** | HUBO research | Humanoid research | Academic, foundation of Rainbow Robotics |
| **Naver Labs** | AMBIDEX, various | Service robots | Naver's robotics division, indoor delivery |

### Key Observations

Korea's robotics strength is in **industrial cobots** (Doosan, Rainbow) and **research hands** (Wonik/Allegro). Hyundai's ownership of Boston Dynamics makes Korea the de-facto home of the world's most famous robot company. Rainbow Robotics is notable as the only publicly traded humanoid robot company in Korea, with roots in KAIST's DARPA-winning HUBO team.

---

## 16. Chinese Robot Companies

| Company | Product | Type | Notable |
|---------|---------|------|---------|
| **Unitree** | G1, H1, H2, R1, Go2, B2, A2 | Humanoid, quadruped | Price disruptor, IPO preparation, ~500 employees |
| **Fourier** | GR-1, GR-2 | Humanoid | Rehab robotics background, Xi Jinping inspection |
| **UFactory** | xArm 5/6/7, Lite 6 | Collaborative arms | Best price/performance ratio for research arms |
| **KUKA** (Chinese-owned) | iiwa, various industrial | Industrial arms | Owned by Midea Group since 2016 |
| **Agile Robots** (German-Chinese) | Franka integration | Research arm | Acquired Franka Robotics in 2023 |
| **Galbot** | Various | General | RL-focused manipulation startup |
| **Agilex Robotics** | Mobile bases + arms | Mobile manipulation | Scout, Ranger platforms, popular in research |
| **Waveshare** | RoArm M3 series | Low-cost arms | Our robot manufacturer, Shenzhen-based |
| **Feetech** | STS3215, SCS series | Servo motors | Powers SO-100/101 and many DIY arms |
| **UBTECH** | Walker X | Humanoid | Consumer/education market |
| **XPeng Robotics** | PX5 | Humanoid | EV company's robotics division |
| **Zhiyuan (AGIBOT)** | A2 arms + humanoid | Research arms | $1B+ funding, ex-TP-Link founders |

### Key Observations

China dominates the **low-cost hardware supply chain** for robot learning research:
- **Feetech** makes the motors that power SO-100/101 (the LeRobot standard)
- **UFactory** makes the xArm (best price/performance research arm)
- **Unitree** makes the cheapest humanoid (G1 at $16K) and quadruped (Go2 at $1.6K)
- **Waveshare** makes our RoArm M3

The Chinese humanoid robot sector is exploding: Unitree, Fourier, UBTECH, XPeng, and Zhiyuan/AGIBOT are all actively developing humanoids, often with substantial government backing. Unitree is preparing for IPO. The combination of manufacturing capability + government support + academic talent (Tsinghua, Shanghai Jiao Tong, CUHK) makes China the most dynamic geography for robot hardware innovation.

---

## Appendix: Timeline of Key Events (2024-2026)

| Date | Event |
|------|-------|
| Apr 2024 | Boston Dynamics retires hydraulic Atlas, reveals electric Atlas |
| Aug 2024 | Figure 02 introduced, Unitree G1 mass production at $16K |
| Aug 2024 | 1X NEO Beta introduced |
| Oct 2024 | Tesla "We, Robot" event -- Optimus mostly teleoperated |
| Oct 2025 | Figure 03 introduced (tactile + palm cameras) |
| Oct 2025 | 1X NEO pre-orders open ($20K / $499/mo) |
| Sep 2025 | Figure AI raises $1B at $39B valuation |
| 2023 | Agile Robots acquires Franka Robotics |
| 2025 | LeRobot v0.4.4 with SO-101, Pi0Fast, GR00T N1.5, XVLA support |
| Feb 2025 | 1X NEO Gamma (updated design) |
| Jun 2025 | Tesla Optimus program head resigns |
| 2025 | Unitree announces R1 ($4,900), H2 (31 DOF), A2 |

---

*Last updated: 2026-03-07*
*Sources: Wikipedia (Tesla Optimus, Figure AI, Unitree Robotics, Atlas, 1X Technologies, Agility Robotics, Fourier Intelligence, Universal Robots, Clearpath Robotics, Shadow Dexterous Hand, Rainbow Robotics, KUKA), manufacturer websites (UFactory, Hello Robot, PSYONIC, Apptronik, Franka Robotics, Waveshare, Unitree Shop), LeRobot GitHub README and hardware integration documentation, Mobile ALOHA project page, LEAP Hand paper (RSS 2023).*
