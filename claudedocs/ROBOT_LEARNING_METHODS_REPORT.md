# Robot Learning Methods: Comprehensive Intelligence Report

**Date:** 2026-03-07
**Agent:** LEARN-OPS
**Scope:** All major robot learning approaches as of early 2026

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Imitation Learning](#1-imitation-learning)
3. [Reinforcement Learning](#2-reinforcement-learning)
4. [Vision-Language-Action Models (VLAs)](#3-vision-language-action-models-vlas)
5. [Sim-to-Real Transfer](#4-sim-to-real-transfer)
6. [World Models](#5-world-models)
7. [Hybrid Approaches](#6-hybrid-approaches)
8. [Data Efficiency Techniques](#7-data-efficiency-techniques)
9. [Frameworks & Libraries](#8-frameworks--libraries)
10. [Master Comparison Table](#9-master-comparison-table)
11. [Practical Recommendations](#10-practical-recommendations-1-robot--rtx-4090)
12. [Unsolved Problems](#11-unsolved-problems)
13. [References](#12-references)

---

## Executive Summary

As of early 2026, robot learning is in the midst of a paradigm shift. The field has bifurcated into two dominant tracks:

1. **VLA/IL-first track**: Pre-trained Vision-Language-Action models fine-tuned with small amounts of task-specific demonstration data. This is the current mainstream for manipulation (pi0, SmolVLA, OpenVLA, GR00T). Flow matching has largely replaced diffusion as the action generation backbone.

2. **RL-in-the-loop track**: RL is no longer used standalone for manipulation from scratch. Instead, it serves as (a) a fine-tuning stage for IL/VLA policies (HIL-SERL, ReinFlow), (b) the dominant paradigm for locomotion (sim-to-real PPO), or (c) a component of model-based planning (TD-MPC2, DreamerV3).

**The current best approach for manipulation:** Pre-trained VLA (pi0 or SmolVLA) fine-tuned on 50-150 task-specific demonstrations, optionally refined with RL.

**Is RL still relevant?** Yes, but its role has shifted. Pure RL from scratch for real-robot manipulation is largely dead. RL thrives in simulation (locomotion, dexterous hands), as a fine-tuning method for IL policies, and in model-based variants for sample-efficient learning.

---

## 1. Imitation Learning

### 1.1 Behavioral Cloning (BC)

| Attribute | Details |
|-----------|---------|
| **Category** | Imitation Learning (supervised) |
| **Key papers** | Pomerleau (1989) ALVINN; Torabi et al. (2018) survey |
| **Core idea** | Supervised learning: map observations to actions from expert demos |
| **Implementation** | Any ML framework; robomimic, LeRobot |
| **GPU requirements** | Minimal (single GPU, even CPU feasible for small models) |
| **Data requirements** | 10-100 demonstrations for simple tasks; quality > quantity |
| **Training time** | Minutes to hours |

**Strengths:**
- Simplest possible approach; easy to implement and debug
- Fast training; no reward engineering required
- Works surprisingly well for short-horizon, unimodal tasks

**Weaknesses:**
- Compounding error: small mistakes accumulate (distribution shift)
- Cannot handle multimodal demonstrations (averages conflicting actions)
- No recovery behavior; if the policy drifts, it cannot correct
- Fundamentally limited by demonstration quality

**Real-world successes:**
- ALVINN (1989): autonomous highway driving
- Many industrial pick-and-place with fixed object positions
- Still used as the inner loop of most modern methods (ACT, VLAs are all fundamentally BC)

**Current frontier:** BC itself is "solved" — the frontier is making it work better via action chunking (ACT), diffusion/flow matching denoising, and VLM pre-training.

---

### 1.2 DAgger (Dataset Aggregation)

| Attribute | Details |
|-----------|---------|
| **Category** | Interactive Imitation Learning |
| **Key papers** | Ross et al. (2011) "A Reduction of Imitation Learning to Online Learning" |
| **Core idea** | Iteratively collect corrections from expert on policy's own state distribution |
| **Implementation** | Custom; no standard library |
| **GPU requirements** | Same as BC |
| **Data requirements** | Starts with 10-50 demos, then iterative corrections |

**Strengths:**
- Provably addresses compounding error problem of BC
- Converges to expert-level performance with sufficient iterations
- Theoretical guarantees (no-regret)

**Weaknesses:**
- Requires interactive expert (human must watch and correct in real-time)
- Extremely tedious for human operators
- Rarely used in modern manipulation research (superseded by better BC variants)

**Real-world successes:**
- Drone racing (Loquercio et al., 2021)
- Surgical robotics (limited)

**Current frontier:** HG-DAgger (Human-Gated DAgger) reduces expert burden. Largely superseded by IL+RL hybrid approaches like SERL.

---

### 1.3 ACT (Action Chunking with Transformers)

| Attribute | Details |
|-----------|---------|
| **Category** | Imitation Learning (generative, chunked) |
| **Key papers** | Zhao et al. (2023) "Learning Fine Manipulation with ACT" (arXiv: 2304.13705) |
| **Core idea** | CVAE + Transformer predicts action *chunks* (sequences) to reduce compounding error |
| **Implementation** | LeRobot (`lerobot-train --policy=act`), original ALOHA codebase |
| **GPU requirements** | Single GPU (RTX 3090/4090 sufficient); ~2-4 GB VRAM |
| **Data requirements** | 10-50 demonstrations (10 minutes of teleoperation) |
| **Training time** | ~1-2 hours for 50 demos |
| **Inference speed** | ~5ms per chunk (very fast) |

**Strengths:**
- Extremely data-efficient: 50 demos for fine manipulation at 80-90% success
- Action chunking elegantly solves temporal correlation and compounding error
- CVAE captures multimodal action distributions
- Low compute requirements; runs on consumer hardware
- Proven on ALOHA platform for bimanual tasks
- Simple architecture; well-understood

**Weaknesses:**
- Limited generalization to novel objects/scenes (no semantic understanding)
- No language conditioning in base version
- Needs consistent demonstration style (sensitive to demo quality variance)
- Does not leverage pre-trained visual representations

**Real-world successes:**
- ALOHA: cable routing, battery insertion, condiment cups (80-90% success)
- ALOHA 2: more complex bimanual tasks
- Widely reproduced by community on SO-100, Koch, custom arms
- Most popular policy for low-cost robot arms as of 2025

**Current frontier:** ACT is being superseded by VLA-based methods for generalization, but remains the gold standard for single-task fine manipulation with limited data. LeRobot's most battle-tested policy.

---

### 1.4 Diffusion Policy

| Attribute | Details |
|-----------|---------|
| **Category** | Imitation Learning (denoising diffusion) |
| **Key papers** | Chi et al. (2023) "Diffusion Policy" (arXiv: 2303.04137, RSS 2023) |
| **Core idea** | Model visuomotor policy as conditional denoising diffusion process over action space |
| **Implementation** | LeRobot (`lerobot-train --policy=diffusion`), diffusion-policy repo |
| **GPU requirements** | Single GPU; ~4-8 GB VRAM; inference slower than ACT |
| **Data requirements** | 50-200 demonstrations |
| **Training time** | 2-8 hours |
| **Inference speed** | ~100-200ms (10-50 denoising steps) |

**Strengths:**
- Handles multimodal action distributions naturally (key advantage over BC)
- 46.9% average improvement over prior SOTA across 12 tasks
- Graceful handling of high-dimensional action spaces
- Training stability superior to GANs
- Receding horizon control enables long-horizon tasks

**Weaknesses:**
- Inference is slow due to iterative denoising (50-100ms+)
- More data-hungry than ACT for same task
- No language conditioning in base version
- DDPM/DDIM scheduling requires careful tuning
- Being superseded by flow matching (faster, simpler)

**Real-world successes:**
- Peg insertion, pushing T-block, sauce pouring
- UMI (Universal Manipulation Interface): in-the-wild manipulation
- Widely used as baseline in manipulation research

**Current frontier:** Flow matching variants (pi0, SmolVLA) are replacing diffusion due to faster inference and simpler training. Consistency distillation for faster inference. 3D diffusion policies using point clouds.

---

### 1.5 Flow Matching Policy

| Attribute | Details |
|-----------|---------|
| **Category** | Imitation Learning (continuous normalizing flows) |
| **Key papers** | Lipman et al. (2023) "Flow Matching for Generative Modeling"; Black et al. (2024) "pi0" (arXiv: 2410.24164) |
| **Core idea** | Learn vector field that transports noise to action distribution; simpler than diffusion |
| **Implementation** | LeRobot (SmolVLA uses flow matching), pi0 codebase |
| **GPU requirements** | Depends on backbone; SmolVLA ~10GB, pi0 ~40GB+ |
| **Data requirements** | 50-150 demos (with pre-trained backbone) |
| **Training time** | Hours to days depending on scale |
| **Inference speed** | ~50-100ms (5-10 denoising steps, faster than diffusion) |

**Strengths:**
- Simpler training objective than diffusion (ODE vs SDE)
- Faster inference: fewer denoising steps needed (5-10 vs 50-100)
- Naturally integrates with VLM backbones (pi0 architecture)
- Handles multimodal distributions like diffusion
- Current state-of-the-art for VLA action heads

**Weaknesses:**
- Relatively new; less community experience than diffusion
- Still iterative at inference (not single-pass like ACT)
- Requires careful action chunk size tuning

**Real-world successes:**
- pi0 (Physical Intelligence): laundry folding, box assembly, table cleaning across multiple robot platforms
- SmolVLA (HuggingFace): manipulation tasks on SO-100
- RoArm-M3 project (this project!): sponge grasping at 100% success rate

**Current frontier:** Flow matching is THE action head for VLAs in 2025-2026. ReinFlow proposes RL fine-tuning of flow matching policies. Consistency flow matching for single-step inference.

---

### 1.6 GAIL (Generative Adversarial Imitation Learning)

| Attribute | Details |
|-----------|---------|
| **Category** | Inverse RL / Adversarial IL |
| **Key papers** | Ho & Ermon (2016) "Generative Adversarial Imitation Learning" |
| **Core idea** | Train discriminator to distinguish expert vs policy, use as reward for RL |
| **Implementation** | stable-baselines3 (partial), imitation library |
| **GPU requirements** | Same as RL backbone (PPO/SAC) |
| **Data requirements** | 5-50 demonstrations + online interaction |

**Strengths:**
- Learns reward function implicitly; no reward engineering
- Can surpass BC by leveraging online interaction
- Theoretically elegant (occupancy measure matching)

**Weaknesses:**
- GAN training instability applies (mode collapse, oscillation)
- Requires online environment interaction (real or sim)
- Much more complex to implement than BC
- Largely superseded by diffusion/flow matching for manipulation

**Real-world successes:**
- Limited real-world deployment; mostly simulation benchmarks
- Some success in locomotion

**Current frontier:** Mostly historical interest. GAIL's core insight (distribution matching) lives on in IRL-based methods, but direct GAIL usage is rare in 2026.

---

## 2. Reinforcement Learning

### 2.1 Model-Free RL (PPO, SAC, TD3)

#### PPO (Proximal Policy Optimization)

| Attribute | Details |
|-----------|---------|
| **Category** | On-policy, model-free RL |
| **Key papers** | Schulman et al. (2017) |
| **Implementation** | stable-baselines3, CleanRL, rl_games, RSL-RL |
| **GPU requirements** | CPU for simple envs; GPU for parallel sim (IsaacGym: thousands of envs) |
| **Data requirements** | Millions to billions of timesteps in simulation |
| **Training time** | Minutes (IsaacGym parallel) to days (single env) |

**Strengths:**
- Extremely robust and stable; works out-of-the-box
- THE standard for sim-to-real locomotion (quadrupeds, humanoids)
- Scales massively with GPU-parallel simulation
- Simple to implement and tune

**Weaknesses:**
- Very sample-inefficient (needs millions of samples)
- Impractical for real-robot learning (too many samples needed)
- On-policy: cannot reuse old data
- Reward engineering is non-trivial

**Real-world successes:**
- Agility Robotics: bipedal locomotion
- ANYmal quadruped (ETH Zurich): rough terrain traversal
- OpenAI Rubik's cube (with massive domain randomization)
- Nearly all legged robot locomotion in 2024-2026

#### SAC (Soft Actor-Critic)

| Attribute | Details |
|-----------|---------|
| **Category** | Off-policy, model-free RL |
| **Key papers** | Haarnoja et al. (2018) |
| **Implementation** | stable-baselines3, CleanRL, SERL |
| **GPU requirements** | Single GPU sufficient |
| **Data requirements** | 10K-1M timesteps (more efficient than PPO) |
| **Training time** | Hours for sim tasks; 25-50 minutes real-robot with SERL |

**Strengths:**
- Off-policy: can reuse all past experience (sample efficient)
- Maximum entropy framework promotes exploration
- Can learn from demonstrations (SAC+demos)
- Basis of SERL for real-robot RL

**Weaknesses:**
- More complex than PPO; more hyperparameters
- Can be unstable with image observations
- Still needs thousands of real-world trials without good initialization

**Real-world successes:**
- SERL: PCB assembly, cable routing in 25-50 minutes of real training
- Dexterous in-hand manipulation
- Robot soccer

#### TD3 (Twin Delayed DDPG)

| Attribute | Details |
|-----------|---------|
| **Category** | Off-policy, model-free RL |
| **Key papers** | Fujimoto et al. (2018) |
| **Implementation** | stable-baselines3, CleanRL |
| **Data requirements** | Similar to SAC |

**Status:** Largely superseded by SAC in robotics. TD3+BC variant is important for offline RL.

---

### 2.2 Model-Based RL

#### DreamerV3

| Attribute | Details |
|-----------|---------|
| **Category** | Model-based RL (learned world model) |
| **Key papers** | Hafner et al. (2023) "Mastering Diverse Domains" (arXiv: 2301.04104) |
| **Core idea** | Learn latent world model, plan by imagining future trajectories |
| **Implementation** | dreamerv3 repo, sheeprl |
| **GPU requirements** | Single GPU (RTX 3090+ recommended) |
| **Data requirements** | 100K-1M steps (10-100x more efficient than PPO) |
| **Training time** | Hours to days |

**Strengths:**
- Single configuration works across 150+ diverse tasks
- 10-100x more sample efficient than model-free RL
- First algorithm to collect diamonds in Minecraft from scratch
- Learns from pixels with no task-specific tuning
- General-purpose: Atari, DMC, Minecraft, robotics

**Weaknesses:**
- World model quality limits policy quality (model bias)
- Complex implementation (RSSM, KL balancing, symlog)
- Slower per-step than model-free (model rollouts)
- Not widely adopted for real-robot manipulation (world model inaccuracies compound)

**Real-world successes:**
- Primarily simulation benchmarks (DMC, Atari, Crafter, Minecraft)
- Limited real-robot deployment; more promise than practice for manipulation

**Current frontier:** DreamerV3 + foundation model features. Scaling world models. Multi-task world models.

#### TD-MPC2

| Attribute | Details |
|-----------|---------|
| **Category** | Model-based RL (implicit world model + planning) |
| **Key papers** | Hansen et al. (2024) "TD-MPC2" (arXiv: 2310.16828, ICLR 2024) |
| **Core idea** | Latent-space trajectory optimization with decoder-free world model |
| **Implementation** | tdmpc2 repo, LeRobot (`lerobot-train --policy=tdmpc`) |
| **GPU requirements** | Single GPU for small models; multi-GPU for 317M agent |
| **Data requirements** | 100K-1M online steps |

**Strengths:**
- Strong results across 104 tasks, 4 domains with single hyperparameters
- Scales: single 317M parameter agent across 80 tasks
- Implicit (decoder-free) world model avoids reconstruction artifacts
- In LeRobot (official support)

**Weaknesses:**
- Online RL (needs environment interaction)
- Planning at test time adds latency
- Limited real-world validation

**Real-world successes:**
- Primarily simulation; promising for sim-to-real pipeline

**Current frontier:** Scaling model-based agents. Multi-embodiment multi-task models. Foundation world models.

---

### 2.3 Offline RL

| Method | Key Paper | Core Idea |
|--------|-----------|-----------|
| **CQL** | Kumar et al. (2020) | Conservative Q-learning: penalize Q-values for OOD actions |
| **IQL** | Kostrikov et al. (2022) | Implicit Q-learning: avoid querying OOD actions entirely |
| **TD3+BC** | Fujimoto & Gu (2021) | TD3 with BC regularization term |
| **Decision Transformer** | Chen et al. (2021) | Sequence modeling of (R,s,a) tuples |

| Attribute | Details |
|-----------|---------|
| **Category** | RL from fixed datasets (no online interaction) |
| **Implementation** | d3rlpy, CORL, robomimic |
| **GPU requirements** | Single GPU |
| **Data requirements** | 100-10,000 demonstrations or mixed-quality data |

**Strengths:**
- No online interaction needed (safe; uses existing data)
- Can leverage suboptimal data (not just expert demos)
- Theoretically principled (conservative estimation)
- Good for fine-tuning pre-trained policies

**Weaknesses:**
- Performance limited by dataset coverage
- Distribution shift at deployment is fundamental challenge
- Often underperforms BC on expert-only data
- Cal-QL and other improvements needed for practical use

**Real-world successes:**
- Robot kitchen tasks (Bridge dataset)
- Autonomous driving (offline RL on logged driving data)
- Fine-tuning VLA policies

**Current frontier:** Offline-to-online fine-tuning (Cal-QL). Using offline RL to improve VLA policies. FOWM (Fine-tuning Offline World Models): pretrain world model offline, fine-tune online.

---

### 2.4 RL Fine-Tuning of VLAs

This is the most active research frontier in robot learning as of 2026.

#### RLPF (RL for Policy Fine-tuning)

| Attribute | Details |
|-----------|---------|
| **Category** | Hybrid IL+RL |
| **Key papers** | Emerging area, multiple groups (2025-2026) |
| **Core idea** | Use RL (PPO/GRPO) to fine-tune IL-pretrained VLA policies |
| **Data requirements** | IL pretraining data + online/offline RL episodes |

#### ReinFlow

| Attribute | Details |
|-----------|---------|
| **Category** | RL fine-tuning of flow matching policies |
| **Key papers** | Emerging (2025) |
| **Core idea** | Apply REINFORCE/PPO-style gradients through flow matching denoising |
| **Challenge** | Gradient through multi-step denoising is non-trivial |

#### HIL-SERL (Human-in-the-Loop SERL)

| Attribute | Details |
|-----------|---------|
| **Category** | Real-robot RL with human demonstrations |
| **Key papers** | Luo et al. (2024) SERL (arXiv: 2401.16013, ICRA 2024) |
| **Core idea** | Off-policy RL (SAC) with demo buffer + learned reward + human resets |
| **Implementation** | SERL library, LeRobot (HIL-SERL integration) |
| **GPU requirements** | Single GPU (RTX 3090/4090) |
| **Data requirements** | 20-50 demos + 25-50 minutes of autonomous practice |
| **Training time** | 25-50 minutes real-world |

**Strengths:**
- Fastest real-robot learning: policies in under 1 hour
- Near-perfect success rates with extreme robustness
- Emergent recovery and correction behaviors (RL advantage over pure IL)
- Open-source, well-documented
- Addresses IL's fundamental weakness (no self-correction)

**Weaknesses:**
- Needs real-time environment interaction (cannot train offline)
- Requires reward function (learned from demos or engineered)
- Reset mechanism needed
- Currently limited to Franka-class robots with good control

**Real-world successes:**
- PCB board assembly: 100% success in 25 min training
- Cable routing: near-perfect in 40 min
- Object relocation with perturbation recovery
- Best reported real-robot RL results as of 2025

**Current frontier:** HIL-SERL is being integrated into LeRobot. Combining VLA pre-training + SERL fine-tuning. Scaling to more complex tasks.

---

## 3. Vision-Language-Action Models (VLAs)

VLAs are the dominant paradigm for generalizable manipulation as of 2026.

### 3.1 Architecture Taxonomy

```
VLA Architecture
├── Autoregressive token-based
│   ├── RT-2 (Google, 2023): PaLM-E → action tokens
│   ├── OpenVLA (2024): Llama2 + DINOv2/SigLIP → action tokens
│   └── Octo (2024): Transformer → action tokens
├── Flow matching action head
│   ├── pi0 (Physical Intelligence, 2024): PaLI-based VLM → flow matching
│   ├── SmolVLA (HuggingFace, 2025): SmolVLM → flow matching expert
│   └── pi0-FAST (2025): faster variant
└── Diffusion action head
    ├── GR00T (NVIDIA, 2025): foundation → diffusion head
    └── Octo (option for diffusion head)
```

### 3.2 Key VLA Models

| Model | Params | Open? | Action Head | Pre-train Data | Fine-tune Data | GPU Req |
|-------|--------|-------|-------------|----------------|----------------|---------|
| **RT-2-X** | 55B | No | Autoregressive | Internet + OXE | Task-specific | TPU pod |
| **OpenVLA** | 7B | Yes | Autoregressive | OXE (970K demos) | 50-200 demos | 1x A100 (LoRA: 1x RTX 4090) |
| **pi0** | ~3B | Partial | Flow matching | Internet + multi-robot | 50-100 demos | 4x A100 |
| **SmolVLA** | 450M | Yes | Flow matching | community_dataset_v1 (SO-100) | 50-150 demos | 1x RTX 4090 |
| **GR00T N1.5** | ~2B | Yes | Diffusion | Multi-robot | Task-specific | 1x A100 |
| **Octo** | 93M | Yes | Diffusion/token | OXE | 50-200 demos | 1x RTX 3090 |
| **XVLA** | - | Yes | - | - | - | - |
| **pi0-FAST** | ~3B | Yes | Flow matching | Same as pi0 | 50-100 demos | 1x RTX 4090 |

### 3.3 VLA Detailed Analysis

#### pi0 (Physical Intelligence)

| Attribute | Details |
|-----------|---------|
| **Key paper** | Black et al. (2024) arXiv: 2410.24164, RSS 2025 |
| **Architecture** | Pre-trained VLM + flow matching action expert |
| **Pre-training** | Internet-scale vision-language + diverse robot data (single-arm, dual-arm, mobile) |
| **Fine-tuning** | 50-100 task-specific demos |
| **Inference** | ~50-100ms |

**Strengths:**
- Best generalization results across multiple robot platforms
- Zero-shot capabilities after pre-training
- Language-conditioned: follows natural language instructions
- Can be directed by high-level VLM planner
- Handles dexterous tasks (laundry, box assembly)

**Weaknesses:**
- Partially open (weights released for pi0-FAST only)
- Requires significant pre-training compute
- Fine-tuning still needed for reliable deployment

**Real-world successes:**
- Laundry folding, table cleaning, box assembly
- Multi-platform: single-arm, dual-arm, mobile manipulator
- Most impressive real-world VLA demos as of 2025

#### OpenVLA

| Attribute | Details |
|-----------|---------|
| **Key paper** | Kim et al. (2024) arXiv: 2406.09246 |
| **Architecture** | Llama 2 + DINOv2 + SigLIP |
| **Size** | 7B parameters |
| **Fine-tuning** | LoRA enables consumer GPU fine-tuning |

**Strengths:**
- Fully open-source (weights, code, training)
- Outperforms RT-2-X (55B) by 16.5% with 7x fewer parameters
- LoRA fine-tuning on consumer GPUs
- Strong language grounding

**Weaknesses:**
- Autoregressive action generation (discretized, slower)
- 7B parameters still large for real-time on edge devices
- Action discretization loses precision

#### SmolVLA (Your Current Method)

| Attribute | Details |
|-----------|---------|
| **Architecture** | SmolVLM (vision-language) + Flow Matching Action Expert |
| **Size** | 450M parameters (smallest VLA) |
| **Pre-training** | community_dataset_v1 (128 datasets, 11,132 episodes, all SO-100) |
| **Action space** | Up to 32-DOF (zero-padded) |
| **Inference** | ~100ms (10 denoising steps) |

**Strengths:**
- Smallest VLA: runs on single RTX 4090
- Open-source in LeRobot
- Flow matching action head (SOTA)
- Proven on real hardware (including your RoArm-M3!)

**Weaknesses:**
- Pre-trained only on SO-100 (OOD for other robots)
- Smaller VLM limits language understanding
- Needs more data for OOD embodiments (150+ episodes)

---

## 4. Sim-to-Real Transfer

### 4.1 Domain Randomization

| Attribute | Details |
|-----------|---------|
| **Category** | Sim-to-Real |
| **Key papers** | Tobin et al. (2017); OpenAI Rubik's cube (2019) |
| **Core idea** | Randomize simulation parameters so policy generalizes to real-world |
| **Implementation** | IsaacGym/IsaacLab, MuJoCo, PyBullet |

**What to randomize:**
- Physics: friction, mass, damping, motor strength
- Visual: lighting, textures, colors, camera pose
- Dynamics: actuator delays, noise, backlash

**Strengths:**
- Simple concept; effective in practice
- THE standard for sim-to-real locomotion
- No real-world data needed for initial policy

**Weaknesses:**
- "Reality gap" for contact-rich manipulation (friction, deformation)
- Over-randomization → conservative policies
- Under-randomization → fails to transfer
- Tuning randomization ranges requires expertise

**Real-world successes:**
- Quadruped locomotion (ANYmal, Unitree): near-universal adoption
- Dexterous in-hand manipulation (OpenAI, NVIDIA)
- Drone racing

### 4.2 System Identification

| Attribute | Details |
|-----------|---------|
| **Core idea** | Measure real-world physical parameters, match simulation accurately |
| **Approach** | Parameter estimation, Bayesian optimization of sim parameters |

**Strengths:** More targeted than domain randomization; less conservative policies.
**Weaknesses:** Labor-intensive; parameters may change over time.

### 4.3 Domain Adaptation

| Attribute | Details |
|-----------|---------|
| **Core idea** | Learn domain-invariant representations that transfer sim→real |
| **Methods** | Adversarial adaptation, visual randomization, style transfer |

**Current status:** Less popular than domain randomization. Visual randomization (changing textures/colors) is the practical approach.

### 4.4 Real-to-Sim-to-Real (3DGS-based)

| Attribute | Details |
|-----------|---------|
| **Category** | Emerging (2024-2025) |
| **Key papers** | EnerVerse (2025, NeurIPS); GRUtopia (2024) |
| **Core idea** | Reconstruct real scene as 3D Gaussian Splats → train in reconstructed sim → deploy |

**Strengths:**
- Photorealistic simulation from real scenes
- Bridges sim-to-real visual gap
- Enables training in exact replica of deployment environment

**Weaknesses:**
- 3DGS reconstruction quality varies
- Physics simulation in reconstructed scenes is hard
- Compute-intensive reconstruction

**Current frontier:** Combining 3DGS with physics engines. 4D Gaussian Splatting for dynamic scenes. EnerVerse's self-reinforcing data loop.

### 4.5 Progressive Networks / Sim-to-Real Fine-tuning

| Attribute | Details |
|-----------|---------|
| **Core idea** | Pre-train in sim, fine-tune specific layers on real data |
| **Methods** | Progressive nets (Rusu et al., 2017), adapter layers, LoRA-style |

**Current status:** This is essentially what VLA fine-tuning does — pre-train on diverse data (including sim), fine-tune on task-specific real data.

---

## 5. World Models

### 5.1 JEPA (Joint Embedding Predictive Architecture)

| Attribute | Details |
|-----------|---------|
| **Category** | Self-supervised world model |
| **Key papers** | LeCun (2022) "A Path Towards Autonomous Machine Intelligence"; Bardes et al. (2024) V-JEPA |
| **Core idea** | Predict representations (not pixels) of future states; avoid generative modeling |
| **Implementation** | V-JEPA (Meta, open-source) |
| **GPU requirements** | Multi-GPU for pre-training; single GPU for fine-tuning |

**Strengths:**
- Avoids wasteful pixel prediction (predict abstract representations)
- Theoretically principled (energy-based models)
- Self-supervised: no labels needed
- V-JEPA shows strong video understanding

**Weaknesses:**
- Not yet directly applied to robot control (representation learning only)
- Gap between representation quality and policy quality
- LeCun's full vision (hierarchical JEPA) not yet realized

**Real-world robotics application:** Primarily as a pre-training objective for visual encoders. Not a complete robot learning method.

### 5.2 Video Prediction Models for Robotics

| Attribute | Details |
|-----------|---------|
| **Key papers** | UniPi (Du et al., 2023); SuSIE (Black et al., 2023) |
| **Core idea** | Use video generation models to predict future visual states, then plan |

**Strengths:**
- Leverages powerful video generation models (Sora-like)
- Can plan long-horizon tasks by "imagining" outcomes
- Language-conditioned planning

**Weaknesses:**
- Video prediction is computationally expensive
- Extracting actions from predicted videos is non-trivial
- Accumulated prediction errors over long horizons

### 5.3 UniSim (Google)

| Attribute | Details |
|-----------|---------|
| **Key papers** | Yang et al. (2023) "Learning Interactive Real-World Simulators" |
| **Core idea** | Neural simulator that generates realistic visual observations given actions |

**Strengths:**
- Can simulate unseen environments
- Enables RL training in learned simulator
- Handles diverse visual domains

**Weaknesses:**
- Requires massive training data and compute
- Physics accuracy limited by training data
- Not publicly available

### 5.4 Genie 2 (DeepMind)

| Attribute | Details |
|-----------|---------|
| **Key papers** | DeepMind blog (2024) |
| **Core idea** | Action-controllable world model trained on video; generates interactive 3D worlds |

**Strengths:**
- Generates consistent, playable 3D environments from single image
- Could enable unlimited RL training environments
- Impressive visual quality

**Weaknesses:**
- Not open-source
- Physics simulation quality unknown
- Gap between generated worlds and real-world manipulation

### 5.5 World Models for Robotics Planning (FOWM)

| Attribute | Details |
|-----------|---------|
| **Key paper** | Feng et al. (2023) "Finetuning Offline World Models" (arXiv: 2310.16029, CoRL 2023 Oral) |
| **Core idea** | Pre-train world model on offline data, fine-tune online with uncertainty-aware planning |

**Strengths:**
- Few-shot fine-tuning to new tasks (even unseen tasks)
- Works even with limited offline data
- Epistemic uncertainty for safe online exploration
- Validated on real robot

**Weaknesses:**
- Planning adds latency
- Model quality limits policy quality
- Complex implementation

**Real-world successes:**
- Real-robot manipulation fine-tuning with minimal online data
- One of few world model papers with real-robot results

---

## 6. Hybrid Approaches

### 6.1 IL + RL Fine-Tuning

| Attribute | Details |
|-----------|---------|
| **Key methods** | RLPD (Ball et al., 2023); HIL-SERL; Cal-QL |
| **Core idea** | Pre-train with IL (BC/ACT/VLA), fine-tune with RL for robustness |

This is the most promising direction as of 2026:

```
[Pre-trained VLA] → [Fine-tune IL on task demos] → [RL fine-tune for robustness]
     pi0/SmolVLA        50-150 demos                   25-50 min real practice
```

**Why it works:** IL provides a good initialization (avoids RL's exploration problem), RL improves robustness, recovery, and handles distribution shift.

### 6.2 Residual RL

| Attribute | Details |
|-----------|---------|
| **Key papers** | Johannink et al. (2019); Silver et al. (2019) |
| **Core idea** | Learn a base policy (analytical/IL) + RL residual correction |

```
action = base_policy(obs) + residual_RL(obs)
```

**Strengths:**
- Base policy provides safety and structure
- RL only needs to learn small corrections
- Much faster convergence than full RL

**Weaknesses:**
- Base policy quality limits overall performance
- Residual can destabilize base policy
- Tuning the balance is tricky

**Real-world successes:**
- Assembly tasks with analytical base controllers
- Track2Act (Bharadhwaj et al., 2024): web-video track prediction + residual policy

### 6.3 Language-Conditioned Learning

| Attribute | Details |
|-----------|---------|
| **Key methods** | RT-2, SayCan, GRIF, VLAs |
| **Core idea** | Use language instructions to condition policy behavior |

**Current status:** Integral to all modern VLAs. Language provides task specification, generalization across tasks, and human-robot interface.

### 6.4 Reward Shaping from VLMs

| Attribute | Details |
|-----------|---------|
| **Key methods** | VLM-RM, CLIP-based rewards, LLM reward code generation |
| **Core idea** | Use VLMs to automatically generate reward signals for RL |

**Strengths:**
- Eliminates manual reward engineering
- Language-specified reward functions
- Enables RL for novel tasks without custom rewards

**Weaknesses:**
- VLM reward signals are noisy and can be hacked
- Reward misspecification remains a fundamental risk

**Current frontier:** Using GPT-4/Claude to write reward functions in code (Eureka, L2R). Using CLIP similarity as dense reward. Active research area.

### 6.5 Demonstration-Augmented RL

| Attribute | Details |
|-----------|---------|
| **Key methods** | RLPD, DDPGfD, DQfD |
| **Core idea** | Add demonstrations to RL replay buffer; bias exploration toward expert behavior |

**Strengths:** Dramatically accelerates RL learning. Standard practice in SERL.
**Weaknesses:** Demo quality matters; bad demos can slow learning.

---

## 7. Data Efficiency Techniques

### 7.1 Data Augmentation for Robotics

| Technique | Description | Effectiveness |
|-----------|-------------|---------------|
| **Random crop** | Random image crops during training | High (standard) |
| **Color jitter** | Random brightness/contrast/saturation | Medium |
| **Random erasing** | Random patches erased from image | Medium |
| **Image translation** | Small random shifts | High (especially for RL from pixels) |
| **Action noise** | Small Gaussian noise on actions | Low-Medium |
| **Viewpoint augmentation** | Synthetic camera angle variation | Not used by SmolVLA; spatial diversity preferred |

**Key finding from SmolVLA:** Image augmentation is NOT used. Instead, spatial diversity in demonstrations (5 positions x 10 reps) provides sufficient variation. This is a departure from earlier work.

### 7.2 Few-Shot Imitation

| Method | Key Idea | Data Needed |
|--------|----------|-------------|
| **Meta-learning (MAML)** | Learn initialization that adapts in few gradient steps | 5-10 demos per task (but many tasks for meta-training) |
| **One-shot imitation** | Condition on single demo video | 1 demo (but needs large pre-training) |
| **In-context learning** | Provide demo as context to VLA | 1-5 demos in prompt |

**Current status:** Few-shot imitation is largely superseded by VLA fine-tuning, which achieves similar goals more effectively.

### 7.3 Cross-Embodiment Transfer

| Attribute | Details |
|-----------|---------|
| **Key papers** | Open X-Embodiment (2024, arXiv: 2310.08864); RT-X |
| **Core idea** | Train on data from many robot types; transfer to new robots |
| **Implementation** | Open X-Embodiment dataset, OXE; pi0, OpenVLA |

**Key findings from OXE:**
- Positive transfer across embodiments (RT-2-X improves on most robots)
- Shared visual/language representations transfer well
- Action spaces must be normalized/adapted per embodiment
- More data diversity → better generalization

**Implications for RoArm-M3:**
- SmolVLA pretrained on SO-100 → OOD for RoArm
- Cross-embodiment transfer works but with performance penalty
- Need 100-150+ episodes to compensate for embodiment gap (vs 50 for in-distribution)

### 7.4 Hindsight Relabeling

| Attribute | Details |
|-----------|---------|
| **Key papers** | HER (Andrychowicz et al., 2017); HIQL (Park et al., 2024) |
| **Core idea** | Relabel failed trajectories with achieved goals as intended goals |

**Strengths:**
- Converts failures into useful training data
- Dramatically improves sample efficiency for goal-conditioned RL
- No additional data needed

**Weaknesses:**
- Only works for goal-conditioned settings
- Does not apply directly to language-conditioned VLAs

---

## 8. Frameworks & Libraries

### 8.1 LeRobot (HuggingFace)

| Attribute | Details |
|-----------|---------|
| **URL** | https://github.com/huggingface/lerobot |
| **Version** | 0.4.4+ (as of early 2026) |
| **Language** | Python (PyTorch) |
| **License** | Apache 2.0 |
| **Supported policies** | ACT, Diffusion, VQ-BeT, SmolVLA, pi0-FAST, pi0.5, GR00T N1.5, XVLA, HIL-SERL, TD-MPC |
| **Supported hardware** | SO-100, Koch, LeKiwi, Reachy2, Unitree G1, and more |
| **GPU requirements** | Varies by policy (RTX 3090+ for VLAs) |

**Strengths:**
- Most comprehensive open-source robotics framework
- Standardized dataset format (Parquet + MP4)
- HuggingFace Hub integration (share datasets and models)
- Both IL and RL policies supported
- Active community and development
- Hardware-agnostic Robot interface
- Supports sim benchmarks (LIBERO, MetaWorld)

**Weaknesses:**
- Rapidly evolving API (breaking changes between versions)
- VLA training still requires significant VRAM
- Limited sim-to-real pipeline (primarily real-world focused)

**Best for:** Anyone doing real-robot learning. The default choice in 2026.

### 8.2 robomimic

| Attribute | Details |
|-----------|---------|
| **URL** | https://robomimic.github.io/ |
| **Focus** | Imitation learning benchmarking |
| **Policies** | BC, BC-RNN, HBC, IRIS, IQL |
| **GPU requirements** | Single GPU |

**Strengths:** Well-designed benchmarks; reproducible experiments; good for research comparisons.
**Weaknesses:** Primarily simulation; limited real-robot support; less actively maintained than LeRobot.

### 8.3 stable-baselines3

| Attribute | Details |
|-----------|---------|
| **URL** | https://stable-baselines3.readthedocs.io/ |
| **Focus** | Model-free RL algorithms |
| **Algorithms** | PPO, SAC, TD3, A2C, DQN, HER |
| **GPU requirements** | CPU or single GPU |

**Strengths:** Reliable, well-tested, great documentation, Gymnasium integration.
**Weaknesses:** No model-based RL; no IL; not robotics-specific; primarily for single-environment RL.

### 8.4 rl_games

| Attribute | Details |
|-----------|---------|
| **URL** | https://github.com/Denys88/rl_games |
| **Focus** | GPU-accelerated RL for IsaacGym |
| **Algorithms** | PPO (primary), SAC |
| **GPU requirements** | NVIDIA GPU required (IsaacGym) |

**Strengths:** Fastest PPO implementation for massively parallel sim; standard for IsaacGym.
**Weaknesses:** Tightly coupled to IsaacGym; limited algorithm selection.

### 8.5 RSL-RL (ETH Zurich)

| Attribute | Details |
|-----------|---------|
| **URL** | https://github.com/leggedrobotics/rsl_rl |
| **Focus** | Legged robot RL (locomotion) |
| **Algorithms** | PPO |
| **GPU requirements** | NVIDIA GPU (IsaacGym/IsaacLab) |

**Strengths:**
- Battle-tested on ANYmal and other quadrupeds
- Clean implementation optimized for locomotion
- Standard in legged robotics community
- Used with IsaacLab

**Weaknesses:** PPO only; locomotion-focused; not for manipulation.

### 8.6 CleanRL

| Attribute | Details |
|-----------|---------|
| **URL** | https://github.com/vwxyzjn/cleanrl |
| **Focus** | Single-file RL implementations for research |
| **Algorithms** | PPO, SAC, TD3, DQN, and many variants |
| **GPU requirements** | Single GPU |

**Strengths:** Incredibly readable (single-file implementations); great for understanding algorithms; WandB integration; well-documented.
**Weaknesses:** Not production-ready; single-file design limits modularity.

### 8.7 IsaacGym / IsaacLab

| Attribute | Details |
|-----------|---------|
| **URL** | https://isaac-sim.github.io/IsaacLab/ |
| **Focus** | GPU-parallel physics simulation for robot learning |
| **GPU requirements** | RTX 3090+ (simulation runs on GPU) |

**Strengths:**
- 1000-10,000 parallel environments on single GPU
- Photorealistic rendering (IsaacSim)
- Comprehensive robot models (arms, quadrupeds, humanoids)
- State-of-the-art for sim-to-real locomotion
- IsaacLab 2.x with modular task design

**Weaknesses:**
- NVIDIA GPU lock-in
- Complex setup (Isaac Sim is heavy)
- Contact-rich manipulation physics less accurate than MuJoCo
- Steep learning curve

**Best for:** Sim-to-real locomotion, dexterous manipulation in sim, RL research requiring massive parallelism.

### 8.8 TorchRL (Meta)

| Attribute | Details |
|-----------|---------|
| **URL** | https://github.com/pytorch/rl |
| **Focus** | Composable, modular RL library in PyTorch |
| **Algorithms** | PPO, SAC, TD3, REDQ, DreamerV3, Decision Transformer |

**Strengths:** PyTorch-native; modular design (TensorDict); good for custom research; multi-process data collection.
**Weaknesses:** Less mature than stable-baselines3; smaller community; steeper learning curve.

---

## 9. Master Comparison Table

### 9.1 Method Comparison

| Method | Category | Data Needed | GPU (Train) | GPU (Infer) | Real-Robot Viable? | Best For |
|--------|----------|-------------|-------------|-------------|-------------------|----------|
| **BC** | IL | 10-100 demos | Any | Any | Yes | Simple tasks |
| **ACT** | IL | 10-50 demos | 1x 3090 | 1x any | Yes | Fine manipulation, low-cost |
| **Diffusion Policy** | IL | 50-200 demos | 1x 3090 | 1x 3090 | Yes | Multimodal tasks |
| **SmolVLA** | VLA+FM | 50-150 demos | 1x 4090 | 1x 4090 | Yes | Language-conditioned manip |
| **OpenVLA** | VLA | 50-200 demos | 1x A100 (LoRA: 4090) | 1x 4090 | Yes | Generalist manipulation |
| **pi0** | VLA+FM | 50-100 demos | 4x A100 | 1x A100 | Yes | Dexterous, multi-platform |
| **PPO** | RL | 1M+ steps (sim) | 1x GPU (parallel) | CPU | Sim-to-real only | Locomotion |
| **SAC/SERL** | RL | 20 demos + 50 min | 1x 4090 | 1x 4090 | Yes (best) | Robust manipulation |
| **DreamerV3** | MBRL | 100K+ steps | 1x 3090 | 1x 3090 | Limited | General sim tasks |
| **TD-MPC2** | MBRL | 100K+ steps | 1x 3090 | 1x 3090 | Limited | Multi-task sim |
| **Offline RL (IQL)** | Offline RL | 100-10K demos | 1x GPU | 1x GPU | Yes | Leveraging existing data |
| **HIL-SERL** | IL+RL | 20 demos + 25 min | 1x 4090 | 1x 4090 | Yes (best) | Robust manipulation |

### 9.2 Data Requirements Comparison

| Method | Minimum Demos | Recommended Demos | Online Interaction? | Time to Policy |
|--------|--------------|-------------------|--------------------|--------------:|
| BC | 5 | 50 | No | ~30 min |
| ACT | 10 | 50 | No | ~1-2 hours |
| Diffusion Policy | 20 | 100 | No | ~2-8 hours |
| SmolVLA (in-dist) | 25 | 50 | No | ~3-5 hours |
| SmolVLA (OOD) | 50 | 150 | No | ~5-10 hours |
| OpenVLA (LoRA) | 20 | 100 | No | ~2-6 hours |
| SERL | 20 | 50 + 50 min RL | Yes | ~1.5 hours |
| PPO (sim) | 0 | 0 (reward only) | Yes (sim) | ~1-4 hours |
| Offline RL | 100 | 1000+ | No | ~2-8 hours |

### 9.3 Compute Requirements

| Method | VRAM (Train) | VRAM (Infer) | Can Use RTX 4090? |
|--------|-------------|-------------|-------------------|
| ACT | 2-4 GB | 1-2 GB | Yes (easily) |
| Diffusion Policy | 4-8 GB | 2-4 GB | Yes |
| SmolVLA | 10 GB (bs=64) | 4-6 GB | Yes |
| OpenVLA (full) | 40+ GB | 14+ GB | No (need A100) |
| OpenVLA (LoRA) | 12-16 GB | 14+ GB | Barely |
| pi0-FAST | 12-16 GB | 6-10 GB | Yes |
| PPO (IsaacGym) | 4-12 GB | 1-2 GB | Yes |
| SAC (SERL) | 4-8 GB | 2-4 GB | Yes |
| DreamerV3 | 8-16 GB | 4-8 GB | Yes |

---

## 10. Practical Recommendations (1 Robot + RTX 4090)

Given the constraints of a master's student with:
- 1 robot arm (RoArm-M3, 6-DOF)
- RTX 4090 Laptop (15.6 GB VRAM)
- Azure Kinect camera
- Linux (Ubuntu 22.04)

### What You Can Realistically Do

#### Tier 1: Immediately Feasible (Already Doing)

| Method | Feasibility | Notes |
|--------|-------------|-------|
| **SmolVLA (LeRobot)** | Proven | Already achieved 100% sponge grasping |
| **ACT (LeRobot)** | Easy | Fastest to train; best for single-task |
| **Diffusion Policy (LeRobot)** | Easy | Good baseline comparison |

#### Tier 2: Feasible with Effort

| Method | Feasibility | Notes |
|--------|-------------|-------|
| **pi0-FAST (LeRobot)** | Feasible | Larger model but fits on 4090; better generalization |
| **HIL-SERL (LeRobot)** | Feasible | Need to implement reward + reset for RoArm |
| **OpenVLA (LoRA)** | Tight fit | 7B model, LoRA fine-tuning barely fits |
| **IsaacLab RL** | Feasible | Already set up; need URDF refinement |

#### Tier 3: Research Exploration

| Method | Feasibility | Notes |
|--------|-------------|-------|
| **VLA + RL fine-tuning** | Ambitious | SmolVLA pretrain → SERL-style fine-tune |
| **Residual RL** | Medium | Analytical base + learned residual |
| **Offline RL on your data** | Easy to try | IQL on collected demos; compare to BC |

#### Tier 4: Beyond Scope

| Method | Why Not |
|--------|---------|
| pi0 (full) | Needs 4x A100 |
| DreamerV3 on real robot | Too sample-inefficient for real |
| GR00T N1.5 | Needs A100+ |
| RT-2-X | Not open source |

### Recommended Research Path

```
Phase 1 (Current): SmolVLA → Expand data → Multi-position generalization
                    Status: DONE (100% success on sponge)

Phase 2 (Next):     Compare ACT vs SmolVLA vs Diffusion Policy
                    Same task, same data, quantitative comparison
                    Contribution: Cross-method comparison on low-cost OOD arm

Phase 3 (Advanced): SmolVLA + HIL-SERL fine-tuning
                    IL pre-training + RL robustness
                    Contribution: First VLA+RL on sub-$500 robot arm

Phase 4 (Stretch):  Sim-to-real with IsaacLab
                    PPO in sim → fine-tune real
                    Contribution: Complete pipeline comparison IL vs sim-to-real
```

### Master's Thesis Angle Suggestions

1. **"SmolVLA on OOD Embodiments"**: Systematic study of VLA transfer to robots not in pre-training data. Data efficiency curves, failure modes, mitigation strategies. You already have unique data.

2. **"IL vs RL vs Hybrid for Low-Cost Arms"**: Compare ACT, SmolVLA, SERL, and SmolVLA+SERL on the same tasks. No one has done this on a sub-$500 arm.

3. **"Data Collection Strategies for VLA Fine-tuning"**: How many demos? What diversity? Hand-guiding vs teleoperation? Your experience with 50ep failure → 74ep success is publishable data.

---

## 11. Unsolved Problems

### Fundamental Challenges

| Problem | Status | Why It Matters |
|---------|--------|----------------|
| **Generalization across objects** | Partially solved by VLAs | VLAs generalize to seen categories but struggle with truly novel objects |
| **Contact-rich manipulation** | Unsolved | Deformable objects, in-hand manipulation, assembly with tight tolerances |
| **Long-horizon planning** | Active research | Chaining multiple skills reliably over minutes of execution |
| **Safety guarantees** | Unsolved | No learned policy has formal safety certificates |
| **Sample efficiency** | Improving | Still need 50+ demos per task (vs humans who learn from 1-2 examples) |
| **Sim-to-real for manipulation** | Partially solved | Locomotion is solved; manipulation transfer remains brittle |
| **Multi-task generalization** | Active research | VLAs can do it but performance degrades vs single-task |
| **Closed-loop drift** | Active research | Your RoArm experience: small errors compound |
| **Real-time adaptation** | Emerging | Adapting to changed conditions during execution |
| **Reward specification** | Improving | VLM-based rewards are promising but noisy |

### Open Questions for 2026-2027

1. **Will scaling VLAs solve manipulation like scaling LLMs solved language?** Unknown. The data bottleneck is fundamentally different (physical data is expensive).

2. **Is simulation good enough for manipulation?** Contact physics accuracy remains the bottleneck. 3DGS and neural physics are promising but unproven.

3. **Can RL recover from IL's blind spots?** SERL says yes for simple tasks. Whether this scales to dexterous manipulation is unclear.

4. **How much human data do we actually need?** The community is split between "collect millions of demos" (Open X-Embodiment) and "learn from internet video + minimal real data" (Track2Act).

5. **Will world models replace model-free approaches?** DreamerV3 and TD-MPC2 show promise, but model-free SERL achieves better real-robot results with less complexity.

---

## 12. References

### Imitation Learning
- **ACT**: Zhao et al., "Learning Fine Manipulation with Action Chunking Transformers", 2023 (arXiv: 2304.13705)
- **Diffusion Policy**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023 (arXiv: 2303.04137)
- **GAIL**: Ho & Ermon, "Generative Adversarial Imitation Learning", NeurIPS 2016
- **DAgger**: Ross et al., "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning", AISTATS 2011

### VLAs
- **pi0**: Black et al., "pi0: A Vision-Language-Action Flow Model for General Robot Control", RSS 2025 (arXiv: 2410.24164)
- **OpenVLA**: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", 2024 (arXiv: 2406.09246)
- **SmolVLA**: HuggingFace, LeRobot 0.4.4 (2025)
- **RT-2**: Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", 2023
- **Open X-Embodiment**: Open X-Embodiment Collaboration, 2024 (arXiv: 2310.08864)
- **BYOVLA**: Hancock et al., "Bring Your Own VLA", 2024 (arXiv: 2410.01971)

### Reinforcement Learning
- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- **SAC**: Haarnoja et al., "Soft Actor-Critic", ICML 2018
- **DreamerV3**: Hafner et al., "Mastering Diverse Domains through World Models", 2023 (arXiv: 2301.04104)
- **TD-MPC2**: Hansen et al., "TD-MPC2: Scalable, Robust World Models", ICLR 2024 (arXiv: 2310.16828)
- **SERL**: Luo et al., "SERL: A Software Suite for Sample-Efficient Robotic RL", ICRA 2024 (arXiv: 2401.16013)
- **CQL**: Kumar et al., "Conservative Q-Learning for Offline RL", NeurIPS 2020
- **IQL**: Kostrikov et al., "Offline RL with Implicit Q-Learning", ICLR 2022

### World Models & Sim-to-Real
- **FOWM**: Feng et al., "Finetuning Offline World Models in the Real World", CoRL 2023 Oral (arXiv: 2310.16029)
- **EnerVerse**: Huang et al., "EnerVerse: Generative Robotics Foundation Model", NeurIPS 2025 (arXiv: 2501.01895)
- **V-JEPA**: Bardes et al., "V-JEPA: Latent Video Prediction for Visual Representation Learning", Meta 2024
- **GRUtopia**: Wang et al., "GRUtopia: Dream General Robots in a City at Scale", 2024

### Frameworks
- **LeRobot**: https://github.com/huggingface/lerobot
- **robomimic**: https://robomimic.github.io/
- **stable-baselines3**: https://stable-baselines3.readthedocs.io/
- **CleanRL**: https://github.com/vwxyzjn/cleanrl
- **IsaacLab**: https://isaac-sim.github.io/IsaacLab/
- **RSL-RL**: https://github.com/leggedrobotics/rsl_rl
- **TorchRL**: https://github.com/pytorch/rl

### Surveys
- Yao et al., "Language-Conditioned Robot Manipulation: A Survey", 2024 (arXiv: 2312.10807)
- ViNT: Shah et al., "Visual Navigation Transformer", CoRL 2023 (arXiv: 2306.14846)
- Track2Act: Bharadhwaj et al., 2024 (arXiv: 2405.01527)

---

## Appendix: Glossary

| Term | Definition |
|------|-----------|
| **VLA** | Vision-Language-Action model: multimodal model mapping images+language to robot actions |
| **Flow Matching** | Generative modeling by learning vector fields; faster than diffusion |
| **Action Chunking** | Predicting sequences of future actions instead of single actions |
| **OOD** | Out-of-distribution: inputs different from training data |
| **Closed-loop** | Re-observe and re-plan at every step |
| **Open-loop** | Execute planned action chunk without re-observation |
| **CVAE** | Conditional Variational Autoencoder |
| **LoRA** | Low-Rank Adaptation: parameter-efficient fine-tuning |
| **OXE** | Open X-Embodiment: large-scale multi-robot dataset |
| **Sim-to-Real** | Training in simulation, deploying on real robot |
| **Domain Randomization** | Randomizing simulation parameters for robust transfer |
| **HIL** | Human-in-the-Loop |
| **MBRL** | Model-Based Reinforcement Learning |
