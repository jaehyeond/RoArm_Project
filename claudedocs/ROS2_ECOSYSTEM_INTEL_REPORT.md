# ROS 2 Ecosystem Intelligence Report
**Agent: ROS-SIGINT | Date: 2026-03-07 | Classification: OPEN**

---

## Executive Summary

ROS 2 is now the **sole supported ROS ecosystem**. ROS 1 Noetic reached EOL in May 2025, making ROS 2 the only path forward. As of March 2026, the ecosystem has matured significantly: **Zenoh is now Tier 1 middleware** (Kilted Kaiju, May 2025), **Rust has first-class support** via `rosidl_rust`, and NVIDIA Isaac ROS provides GPU-accelerated perception/manipulation pipelines. The gap between traditional ROS 2 robotics and end-to-end VLA approaches is narrowing, with ROBOTIS (Korea) already shipping `physical_ai_tools` that bridges LeRobot and ROS 2.

---

## 1. ROS 2 Current State

### 1.1 Distribution Timeline

| Distro | Release | EOL | Ubuntu | Status | Notes |
|--------|---------|-----|--------|--------|-------|
| **Humble Hawksbill** | 2022-05-23 | **2027-05** | 22.04 | LTS, Supported | Widely deployed in production |
| **Iron Irwini** | 2023-05-23 | 2024-12-04 | 22.04 | **EOL** | Skip this |
| **Jazzy Jalisco** | 2024-05-23 | **2029-05** | 24.04 | **LTS, Recommended** | Current best choice for new projects |
| **Kilted Kaiju** | 2025-05-23 | 2026-12 | 24.04 | Non-LTS, Supported | Latest stable, Zenoh Tier 1 |
| **Lyrical Luth** | 2026-05 (planned) | 2031-05 | 24.04 | **Future LTS** | Next LTS release |
| **Rolling Ridley** | Continuous | N/A | Noble | Dev only | Bleeding edge, not for production |

**Recommendation:**
- **New projects (2026):** Jazzy Jalisco (LTS, supported until 2029, Ubuntu 24.04)
- **Existing production:** Humble if on Ubuntu 22.04 (supported until 2027)
- **Bleeding edge/research:** Kilted or Rolling (Zenoh Tier 1, Rust support)
- **For your RoArm project (Ubuntu 22.04):** Humble is the correct choice

### 1.2 ROS 1 EOL Status (May 2025)

ROS 1 Noetic Ninjemys reached End of Life in May 2025 (tied to Ubuntu 20.04 Focal EOL). The official ROS wiki now states:

> "All ROS 1 distributions have reached end-of-life. You should use ROS 2."

**Migration status:**
- Most major packages have ROS 2 ports (MoveIt 2, Nav2, ros2_control all mature)
- Legacy industrial systems still running ROS 1 with `ros1_bridge` for gradual migration
- New projects universally use ROS 2
- Academic papers increasingly require ROS 2 compatibility

### 1.3 Zenoh as Tier 1 Middleware

**Major milestone in Kilted Kaiju (May 2025): `rmw_zenoh_cpp` is now Tier 1.**

| Aspect | DDS (FastDDS/CycloneDDS) | Zenoh |
|--------|--------------------------|-------|
| Tier Status | Tier 1 (default) | **Tier 1** (as of Kilted) |
| Discovery | Multicast (noisy on large networks) | Router-based gossip (clean, scalable) |
| Latency | Good | **Better** (benchmarked lower) |
| Bandwidth | Standard DDS overhead | **Lower overhead** |
| Embedded | Limited | **zenoh-pico** for MCUs (Raspberry Pi Pico supported) |
| Cross-network | Complex DDS bridging | **Native WAN support** |
| Configuration | Complex XML profiles | **Simple JSON5 configs** |

Zenoh version timeline:
- 1.0.0 "Firesong" (October 2024) -- API stabilization, 1.0 milestone
- 1.5.0 "Hong" (July 2025)
- 1.6.x "Imoogi" (October 2025)
- 1.7.x "Jiaolong" (December 2025) -- latest

**Usage:**
```bash
sudo apt install ros-jazzy-rmw-zenoh-cpp
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 run rmw_zenoh_cpp rmw_zenohd  # Start Zenoh router
```

**ROBOTIS `zenoh_ros2_sdk`**: Python SDK enabling ROS 2 pub/sub via Zenoh **without ROS 2 installation**. This is significant -- it means any Python process (e.g., a VLA inference server) can participate in ROS 2 communication with `pip install zenoh-ros2-sdk`.

### 1.4 MicroROS (Embedded Systems)

- Bridges microcontrollers (STM32, ESP32, Raspberry Pi Pico) to ROS 2
- Uses Micro-XRCE-DDS (or zenoh-pico) as transport
- Supports FreeRTOS, Zephyr, NuttX, bare-metal
- Vibrant community, monthly working group meetings
- Maintained by eProsima, integrated into Vulcanexus
- Jazzy and Rolling support

**Relevance to RoArm:** The RoArm M3's ESP32 could theoretically run micro-ROS, publishing joint states and subscribing to commands directly as ROS 2 topics, eliminating the serial SDK layer.

### 1.5 Rust in ROS 2

Kilted Kaiju added `rosidl_rust` as a default code generator. This means:
- Rust nodes can be written natively
- Message/service types auto-generated for Rust
- Zenoh itself is written in Rust -- synergy with `rmw_zenoh`
- Growing ecosystem of Rust-based ROS 2 packages

---

## 2. Key ROS 2 Packages for Robotics Research

### 2.1 Package Maturity Matrix

| Package | Purpose | Distros | Maturity | Your Relevance |
|---------|---------|---------|----------|----------------|
| **MoveIt 2** | Manipulation planning | Humble/Jazzy/Rolling | Production | HIGH -- motion planning for RoArm |
| **Nav2** | Navigation stack | Humble/Jazzy/Rolling | Production | LOW -- mobile robot focus |
| **ros2_control** | HW abstraction | Humble/Jazzy/Kilted/Rolling | Production | HIGH -- hardware interface |
| **tf2** | Transform library | All | Stable | HIGH -- coordinate frames |
| **image_pipeline** | Camera processing | All | Stable | HIGH -- Azure Kinect images |
| **cv_bridge** | OpenCV-ROS bridge | All | Stable | HIGH -- image conversion |
| **PCL (ROS2)** | Point clouds | Humble/Jazzy | Stable | MEDIUM -- depth processing |
| **octomap_server2** | 3D occupancy mapping | Humble | Beta | LOW |

### 2.2 MoveIt 2

- **Used on 150+ robots**, BSD licensed
- Commercially supported version: **MoveIt Pro** (by PickNik)
- CI builds for Rolling, Jazzy, and Humble
- Key components:
  - Motion Planning (OMPL, Pilz, STOMP planners)
  - Inverse Kinematics (KDL, TRAC-IK, custom)
  - Collision checking (FCL)
  - **MoveIt Task Constructor** -- multi-step task planning
  - **MoveIt Servo** -- real-time jogging/teleoperation
  - Grasp generation pipeline
  - Gazebo integration
  - **NVIDIA cuMotion** integration via `isaac_ros_cumotion_moveit`

**For RoArm M3:** MoveIt 2 would require a URDF/XACRO model and a `ros2_control` hardware interface. This is the standard path for manipulation research.

### 2.3 ros2_control

The hardware abstraction framework. Architecture:
```
Controller Manager
├── Joint Trajectory Controller
├── Forward Command Controller
├── Diff Drive Controller
├── ...custom controllers...
│
Hardware Interface (your robot driver)
├── command_interfaces (position, velocity, effort)
└── state_interfaces (position, velocity)
```

Supported distros: Humble, Jazzy, Kilted, Rolling (full CI on all).

**For RoArm M3:** Writing a `ros2_control` hardware interface for RoArm would:
1. Expose joints as standard `JointState` messages
2. Allow MoveIt 2 to plan and execute trajectories
3. Enable `ros2_control` controllers (JointTrajectoryController)
4. The existing `roarm_sdk` serial protocol maps cleanly to this pattern

### 2.4 image_pipeline / cv_bridge

- `image_proc`: rectification, color conversion, resizing
- `depth_image_proc`: depth to point cloud, registration
- `stereo_image_proc`: stereo correspondence
- `cv_bridge`: converts between ROS Image messages and OpenCV `cv::Mat` / numpy arrays
- NVIDIA alternative: `isaac_ros_image_pipeline` (GPU-accelerated for Jetson)

---

## 3. Perception Stack

### 3.1 Object Detection + ROS 2

| Framework | ROS 2 Integration | Status |
|-----------|-------------------|--------|
| **YOLOv8/YOLO11** | `isaac_ros_yolov8` (NVIDIA), community `yolov8_ros` | Production-ready |
| **RT-DETR** | `isaac_ros_rtdetr` (NVIDIA) | Production-ready |
| **Grounding DINO** | `isaac_ros_grounding_dino` (NVIDIA) | Open-vocab detection |
| **DetectNet** | `isaac_ros_detectnet` (NVIDIA) | Optimized for Jetson |

**YOLO + ROS 2 pattern:**
```
[Camera Node] → /image_raw → [YOLO Node] → /detections → [Planning Node]
                              TensorRT optimized
```

### 3.2 Segmentation + ROS 2

| Model | Package | Notes |
|-------|---------|-------|
| **SAM** | `isaac_ros_segment_anything` | NVIDIA-accelerated |
| **SAM2** | `isaac_ros_segment_anything2` | Video segmentation |
| **SegFormer** | `isaac_ros_segformer` | Semantic segmentation |
| **U-Net** | `isaac_ros_unet` | Classic architecture |

### 3.3 SLAM / Localization

| Package | Type | ROS 2 Status | Notes |
|---------|------|-------------|-------|
| **RTAB-Map** | Visual + LiDAR SLAM | Humble/Jazzy/Rolling binaries | Best multi-sensor SLAM, actively maintained |
| **Isaac ROS Visual SLAM** | cuVSLAM | NVIDIA package | GPU-accelerated, production-grade |
| **ORB-SLAM3** | Visual SLAM | Community ROS 2 ports | Research-grade |
| **nvblox** | Dense 3D reconstruction | `isaac_ros_nvblox` | Real-time occupancy, Nav2 costmap provider |
| **Isaac Mapping** | Visual mapping + localization | `isaac_mapping_ros` | Global localization |

RTAB-Map recommendation from their repo: use CycloneDDS or Zenoh for best performance (DDS default can be laggy with large point clouds).

### 3.4 Depth Camera Integration

| Camera | ROS 2 Driver | Status |
|--------|-------------|--------|
| **Intel RealSense** | `realsense2_camera` (ROS 2) | Actively maintained, Humble/Jazzy |
| **Azure Kinect DK** | `azure_kinect_ros_driver` | **RETIRED by Microsoft, no longer maintained** |
| **Stereolabs ZED** | `zed-ros2-wrapper` | Actively maintained |
| **Orbbec** | `OrbbecSDK_ROS2` | Active |
| **OAK-D** | `depthai-ros` | Active, Humble/Jazzy |

**Azure Kinect Warning:** The official ROS driver is ROS 1 only (melodic branch) and **retired**. Microsoft transferred the technology and stopped maintaining both the SDK and the ROS driver. For ROS 2 usage, you would need community forks or a custom wrapper using `pyk4a`.

### 3.5 Pose Estimation

| Model | ROS 2 Package | Notes |
|-------|--------------|-------|
| **FoundationPose** | `isaac_ros_foundationpose` | NVIDIA, TensorRT-optimized, #1 on BOP leaderboard |
| **CenterPose** | `isaac_ros_centerpose` | 6DoF from single image |
| **DOPE** | `isaac_ros_dope` | Deep Object Pose Estimation |

FoundationPose explicitly recommends its Isaac ROS version for robotics deployment:
> "For ROS version, please check Isaac ROS Pose Estimation, which enjoys TRT fast inference and C++ speed up."

### 3.6 Grasp Perception

| Package | Description | ROS 2 |
|---------|-------------|-------|
| **AnyGrasp** | SOTA grasp detection + tracking | SDK only, no official ROS 2 (community wrappers exist) |
| **GraspNet** | Grasp baseline | Python SDK |
| **MoveIt Grasp Generation** | Geometric + ML grasps | Built into MoveIt 2 |
| **GPD (Grasp Pose Detection)** | Point cloud grasps | ROS 1/2 packages |

AnyGrasp SDK now supports CUDA 12.8 and Python 3.11/3.12/3.13 (November 2025 update), making it compatible with modern environments.

---

## 4. ROS 2 + AI/ML Integration

### 4.1 ROS 2 + PyTorch Inference

Standard pattern:
```python
import rclpy
from sensor_msgs.msg import Image
import torch

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        self.model = torch.load('model.pt')
        self.sub = self.create_subscription(Image, '/image', self.callback, 10)
        self.pub = self.create_publisher(...)

    def callback(self, msg):
        # cv_bridge → numpy → tensor → inference → publish
        img = self.bridge.imgmsg_to_cv2(msg)
        tensor = torch.from_numpy(img).cuda()
        result = self.model(tensor)
        self.pub.publish(result)
```

NVIDIA approach: Use `isaac_ros_tensor_rt` or `isaac_ros_triton` for optimized inference, avoiding Python GIL bottlenecks.

### 4.2 ROS 2 + VLA Models (Vision-Language-Action)

**This is the critical intersection for your project.**

| Integration | Status | Details |
|-------------|--------|---------|
| **ROBOTIS `physical_ai_tools`** | **Active (March 2026)** | LeRobot + ROS 2 bridge, datasets on HuggingFace |
| **ROBOTIS `ai_worker`** | **Active (March 2026)** | Full ROS 2 packages for physical AI robot |
| **Direct VLA → ROS 2** | Community efforts | No official HuggingFace/LeRobot ROS 2 integration |
| **NVIDIA Isaac Manipulator** | Production | Not VLA, but GPU-accelerated manipulation pipeline |

**ROBOTIS Physical AI Tools** is the most significant finding:
- Branch: `jazzy` (ROS 2 Jazzy)
- Connects LeRobot framework directly to ROS 2
- ROBOTIS publishes datasets and pretrained models on HuggingFace (`ROBOTIS` org)
- Includes Docker images (`robotis/ros` on Docker Hub)
- MuJoCo simulation models available
- **This is the closest existing solution to what your RoArm + SmolVLA project needs**

**ROBOTIS `zenoh_ros2_sdk`** enables an elegant VLA deployment pattern:
```
[Azure Kinect] → pyk4a → zenoh_ros2_sdk.publish(/image) ──┐
                                                            │ Zenoh
[RoArm joints] → roarm_sdk → zenoh_ros2_sdk.publish(/joints)─┤ Network
                                                            │
[SmolVLA Inference] ← zenoh_ros2_sdk.subscribe(/image, /joints)
                   → zenoh_ros2_sdk.publish(/action) ───────┘
                                                            │
[RoArm Controller] ← zenoh_ros2_sdk.subscribe(/action) ────┘
```

This pattern requires **no ROS 2 installation** -- just `pip install zenoh-ros2-sdk` -- but fully interoperates with ROS 2 nodes.

### 4.3 ROS 2 + Isaac Sim/Lab

NVIDIA Isaac ecosystem:
```
Isaac Sim (NVIDIA Omniverse)
├── Physics simulation (PhysX 5)
├── Synthetic data generation
├── Domain randomization
│
Isaac Lab (RL framework)
├── GPU-accelerated RL training
├── Curriculum learning
├── RSL-RL, RL Games, SKRL integrations
│
Isaac ROS (deployment)
├── ROS 2 GEMs (GPU-accelerated packages)
├── NITROS (hardware-accelerated ROS transport)
├── Sim-to-Real bridge
```

Isaac ROS packages (deployment target):

| Category | Packages |
|----------|----------|
| **Perception** | AprilTag, Object Detection (YOLO, RT-DETR, GroundingDINO), Segmentation (SAM/SAM2, SegFormer) |
| **3D Vision** | Visual SLAM, nvblox (3D reconstruction), Stereo Depth (ESS, FoundationStereo) |
| **Manipulation** | cuMotion (GPU motion planning), FoundationPose, DOPE, CenterPose |
| **Navigation** | Mapping, Localization, Occupancy Grid |
| **Infrastructure** | NITROS (GPU transport), TensorRT/Triton inference, H.264 encode/decode |
| **Platforms** | AGX Thor, DGX Spark, x86_64 + RTX, Jetson |

Performance examples (from NVIDIA benchmarks):
- AprilTag: 596 fps on x86_64 w/ RTX 5090
- TensorRT DNN: 1570 fps on x86_64 w/ RTX 5090
- cuMotion motion planning: millisecond-scale trajectory optimization

### 4.4 ROS 2 + LeRobot

**No official LeRobot ROS 2 integration exists** from HuggingFace. LeRobot uses its own hardware abstraction (`lerobot.robots.*`) and does not depend on ROS 2.

However:
1. **ROBOTIS `physical_ai_tools`** provides the bridge (see 4.2)
2. LeRobot's `roarm_m3.py` (your backup) uses direct serial communication
3. A ROS 2 wrapper around LeRobot's policy inference is straightforward:
   ```python
   # Conceptual: LeRobot policy as ROS 2 node
   class SmolVLANode(Node):
       def image_callback(self, msg):
           observation = {"observation.image": preprocess(msg)}
           action = self.policy.select_action(observation)
           self.publish_joint_command(action)
   ```

### 4.5 ROS 2 + LLM/VLM for Task Planning

Emerging pattern (2025-2026):
```
[User] → "Pick up the red cup" → [LLM Task Planner]
                                       │
                         ┌─────────────┼──────────────┐
                         ↓             ↓              ↓
                   [Perception]  [Motion Plan]  [Grasp Plan]
                   YOLO/SAM      MoveIt 2      AnyGrasp
                         ↓             ↓              ↓
                         └─────────────┼──────────────┘
                                       ↓
                               [ROS 2 Execution]
```

This is the "classical" approach vs. VLA end-to-end. The tradeoff:
- Classical: Interpretable, modular, debuggable, requires engineering each component
- VLA: End-to-end, simple pipeline, requires data + training, less interpretable

---

## 5. Industry Usage

### 5.1 Major Companies Using ROS 2

| Company | Domain | ROS 2 Usage |
|---------|--------|-------------|
| **Amazon (AWS)** | Warehouse robotics | AWS RoboMaker, Proteus robots |
| **Boston Dynamics** | Legged robots | Spot SDK interop with ROS 2 |
| **NVIDIA** | GPU robotics | Isaac ROS entire stack |
| **Qualcomm** | Edge AI robotics | RB5/RB6 platform with ROS 2 |
| **Samsung** | Service robots | Internal ROS 2 deployment |
| **Bosch** | Industrial robotics | ros2_control maintainer, OSRA member |
| **Intel** | Perception | RealSense ROS 2, OpenVINO |
| **Clearpath/Rockwell** | Field robots | Husky, Jackal, Dingo on ROS 2 |
| **Universal Robots** | Cobots | UR ROS 2 driver |
| **ABB** | Industrial arms | ROS 2 interop packages |
| **Franka Emika** | Research cobots | `franka_ros2` official |
| **iRobot** | Consumer robots | ROS 2 on Roomba J7+ |

### 5.2 Autonomous Vehicles

**Autoware** is the premier open-source autonomous driving stack built entirely on ROS 2:
- Active development (2026), large contributor community
- Localization, perception, planning, control
- Used by Tier IV, Apex.AI, and numerous AV companies
- Production deployments in Japan, Europe

### 5.3 Agricultural Robots

- AgOpenGPS community experiments with ROS 2
- Carbon Robotics (weed laser) uses ROS 2
- Growing adoption in precision agriculture

### 5.4 Medical/Surgical Robots

- Primarily research-stage ROS 2 integration
- da Vinci research kit has ROS 2 bindings
- Regulatory barriers slow production adoption
- `ros2_control` real-time capabilities enabling surgical applications

### 5.5 Korean Companies Using ROS 2

| Company | Product | ROS 2 Role |
|---------|---------|------------|
| **ROBOTIS** | DYNAMIXEL, TurtleBot3/4, OpenManipulator, **AI Worker** | Core platform, LeRobot integration, Physical AI |
| **Doosan Robotics** | Collaborative arms (M/H/A series) | `doosan-robot2` ROS 2 driver |
| **Hyundai Robotics** | Industrial arms | ROS 2 integration packages |
| **Naver Labs** | AMBIDEX, indoor robots | ROS 2 research platform |
| **LG Electronics** | CLOi service robots | ROS 2 navigation |
| **Kakao Brain / KAIST** | Research | VLA + ROS 2 experiments |
| **Rainbow Robotics** | RB series cobots | ROS 2 driver |

**ROBOTIS is the most relevant** for your project:
- They build the DYNAMIXEL servos (same bus protocol concept as RoArm)
- `physical_ai_tools` = LeRobot + ROS 2 bridge (updated March 6, 2026)
- `ai_worker` = full physical AI robot with ROS 2 packages
- `zenoh_ros2_sdk` = ROS 2 communication without ROS 2 installation
- OpenPI fork on their GitHub
- Active HuggingFace presence with datasets and models

---

## 6. Career & Job Market

### 6.1 ROS 2 Developer Skills (2025-2026)

**Core requirements:**
| Skill | Priority | Notes |
|-------|----------|-------|
| C++ (14/17/20) | Essential | Node development, real-time systems |
| Python 3 | Essential | Prototyping, launch files, AI integration |
| ROS 2 (Humble/Jazzy) | Essential | Topics, services, actions, lifecycle |
| Linux (Ubuntu) | Essential | Development/deployment platform |
| Git/CI/CD | Essential | colcon, GitHub Actions |
| DDS/Zenoh concepts | Important | Middleware understanding |
| Docker/containerization | Important | Deployment |
| ros2_control | Important | Hardware integration |
| MoveIt 2 | Important (manipulation) | Planning, kinematics |
| Nav2 | Important (mobile) | Navigation |
| PyTorch/TensorRT | Growing | AI integration |
| Gazebo/Isaac Sim | Valuable | Simulation |
| CUDA/GPU computing | Valuable | NVIDIA ecosystem |

### 6.2 Salary Ranges (approximate, 2025-2026)

| Region | Junior | Mid | Senior |
|--------|--------|-----|--------|
| US | $80-110K | $120-160K | $170-250K |
| Europe | EUR 45-65K | EUR 70-100K | EUR 100-150K |
| Korea | KRW 40-55M | KRW 60-85M | KRW 85-130M |

NVIDIA robotics positions command 20-40% premium. VLA/Physical AI expertise adds further premium due to extreme scarcity.

### 6.3 Korean ROS 2 Job Market

| Company | Roles | Focus |
|---------|-------|-------|
| ROBOTIS | ROS 2 SW Engineer | Physical AI, DYNAMIXEL, TurtleBot |
| Doosan Robotics | Robot SW Developer | Cobot integration |
| Rainbow Robotics | Control Engineer | ROS 2 + cobot control |
| Naver Labs | Robotics Researcher | Indoor navigation, manipulation |
| LG Electronics | AMR Developer | Service robot navigation |
| Hyundai Motor Group (Boston Dynamics Korea) | Robotics Engineer | Spot, Atlas integration |
| Samsung Research | AI Robotics | VLM + manipulation |
| KAIST spin-offs | Research Engineer | Cutting-edge robotics |

**Trend:** Korean companies increasingly hiring for "Physical AI" roles that combine VLA/VLM expertise with ROS 2 deployment skills. ROBOTIS leading this convergence.

---

## 7. ROS 2 + Manipulation Research

### 7.1 Grasp Planning Ecosystem

| Approach | Package | ROS 2 Status | Method |
|----------|---------|-------------|--------|
| **AnyGrasp** | `anygrasp_sdk` | SDK (no official ROS 2 pkg) | DNN, point cloud, 6-DoF grasps |
| **GraspIt!** | `graspit_ros2` | Community port | Physics-based simulation |
| **GPD** | `gpd_ros2` | Community port | Point cloud → grasp poses |
| **MoveIt Grasps** | Built into MoveIt 2 | Official | Geometric + ML |
| **Contact-GraspNet** | Research code | No ROS 2 | Contact-based 6-DoF |
| **cuMotion** | `isaac_ros_cumotion` | NVIDIA Official | GPU motion planning for grasping |

### 7.2 Task and Motion Planning (TAMP)

| Package | Description |
|---------|-------------|
| **MoveIt Task Constructor** | Multi-step manipulation task decomposition |
| **BehaviorTree.CPP** | Behavior tree framework for ROS 2 |
| **PlansysII** | PDDL-based planning for ROS 2 |
| **pyroboplan** | Educational Python manipulation planning (Pinocchio-based) |

### 7.3 Deformable Object Manipulation

- Active research area, no dominant ROS 2 package
- Sim-to-real transfer using Isaac Sim deformable body simulation
- VLA approaches (like SmolVLA) may bypass explicit deformable modeling

### 7.4 Contact-Rich Manipulation

- `ros2_control` real-time control enables force/torque feedback
- Impedance/admittance controllers in `ros2_controllers`
- NVIDIA cuMotion handles collision-aware trajectory optimization
- MoveIt Servo enables real-time force-guided manipulation

---

## 8. The ROS 2 vs. VLA Question

### 8.1 Can ROS 2 and VLA Models Work Together?

**Yes, and they should.** Three integration patterns:

**Pattern A: VLA as a ROS 2 Node (most flexible)**
```
[Camera ROS Node] → /image → [VLA Inference Node] → /action → [ros2_control]
```
- VLA policy runs as a standard ROS 2 node
- Benefits: full ROS 2 ecosystem (visualization, logging, safety)
- Overhead: ROS 2 message serialization (~1ms, negligible for 30fps)

**Pattern B: VLA via Zenoh SDK (no ROS 2 install needed)**
```
[Camera] → zenoh_ros2_sdk → [VLA Server] → zenoh_ros2_sdk → [Robot Controller]
```
- Uses ROBOTIS `zenoh_ros2_sdk`
- Benefits: pure Python, no ROS 2 build system, still ROS 2 compatible
- Best for: research prototyping, your current SmolVLA setup

**Pattern C: VLA + Classical Hybrid**
```
[VLA] → high-level intent → [MoveIt 2] → trajectory → [ros2_control]
```
- VLA provides semantic understanding, classical stack handles execution
- Benefits: safety guarantees from classical planning, intelligence from VLA
- This is likely where the field converges

### 8.2 Is ROS 2 Still Relevant with End-to-End Learning?

**Definitively yes**, for several reasons:

1. **Safety:** JOINT_LIMITS, collision checking, emergency stops -- VLA models do not provide these guarantees
2. **Modularity:** Swapping perception, planning, or control independently
3. **Debugging:** ROS 2 topic inspection, rosbag recording, RViz visualization
4. **Hardware abstraction:** `ros2_control` works with any robot, VLA models are embodiment-specific
5. **Industry requirement:** Production robots need certified, verifiable behavior
6. **Complementary, not competing:** VLA replaces specific components (perception+planning), not the entire stack

### 8.3 The Gap Between Traditional ROS 2 and VLA Robotics

| Aspect | Traditional ROS 2 | VLA End-to-End |
|--------|-------------------|----------------|
| **Architecture** | Modular pipeline | Single neural network |
| **Data need** | URDF + parameters | 50-200 demonstrations |
| **Generalization** | Programmatic rules | Learned from data |
| **Interpretability** | Full observability | Black box |
| **Safety** | Deterministic bounds | Statistical bounds |
| **Development speed** | Weeks of engineering | Hours of data collection |
| **Novel objects** | Re-program each | Train with few demos |
| **Deployment** | Standard ROS 2 | Custom inference loop |

**The convergence:** VLA models handle perception-to-action mapping, ROS 2 handles safety, communication, and hardware abstraction. This is what ROBOTIS's `physical_ai_tools` demonstrates.

---

## 9. Recommendations for RoArm M3 + Azure Kinect Project

### 9.1 Most Useful ROS 2 Packages

| Priority | Package | Why |
|----------|---------|-----|
| 1 | `ros2_control` | Hardware interface for RoArm M3 joints |
| 2 | `zenoh_ros2_sdk` | Bridge SmolVLA inference to ROS 2 without full install |
| 3 | `cv_bridge` / `image_pipeline` | Azure Kinect image processing |
| 4 | `MoveIt 2` | Motion planning, IK, safety limits |
| 5 | `tf2` | Camera-to-robot frame transforms |
| 6 | ROBOTIS `physical_ai_tools` | Reference architecture for LeRobot + ROS 2 |

### 9.2 Recommended Integration Path

**Phase 1 (Quick Win): Zenoh SDK Bridge**
- `pip install zenoh-ros2-sdk`
- Wrap `deploy_smolvla.py` with Zenoh publishers/subscribers
- Get ROS 2 topic visibility without changing deployment pipeline
- Enables RViz visualization, rosbag recording

**Phase 2 (Standard Integration): ros2_control**
- Write `RoArmM3HardwareInterface` for `ros2_control`
- URDF model of RoArm M3
- Joint trajectory controller for trajectory execution
- Azure Kinect node (custom, since official driver is retired)

**Phase 3 (Full Stack): MoveIt 2 + VLA Hybrid**
- MoveIt 2 MoveGroup for RoArm M3
- SmolVLA provides grasp targets, MoveIt 2 plans trajectories
- Safety: MoveIt 2 collision checking + joint limits
- Best of both worlds

### 9.3 Azure Kinect ROS 2 Consideration

The official Azure Kinect ROS driver is **retired and unmaintained** (ROS 1 only). Options:
1. Write a custom ROS 2 node using `pyk4a` (you already have this working)
2. Use a community fork (limited maintenance)
3. Consider migrating to Intel RealSense D455 (actively maintained ROS 2 driver)

---

## 10. Ecosystem Map

```
┌─────────────────────────────────────────────────────────────────┐
│                     ROS 2 ECOSYSTEM (2026)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─── MIDDLEWARE ──────────────────────────────────────────┐    │
│  │  DDS (FastDDS, CycloneDDS)  │  Zenoh (Tier 1!)        │    │
│  │  rmw_fastrtps_cpp           │  rmw_zenoh_cpp           │    │
│  │                             │  zenoh-pico (embedded)   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─── PERCEPTION ─────────────┐  ┌─── MANIPULATION ─────────┐ │
│  │ image_pipeline              │  │ MoveIt 2                 │ │
│  │ Isaac ROS (YOLO, SAM/SAM2) │  │ ros2_control             │ │
│  │ FoundationPose              │  │ cuMotion (NVIDIA)        │ │
│  │ RTAB-Map / cuVSLAM         │  │ MoveIt Task Constructor  │ │
│  │ nvblox (3D reconstruction) │  │ AnyGrasp (SDK)           │ │
│  └─────────────────────────────┘  └──────────────────────────┘ │
│                                                                 │
│  ┌─── AI/ML INTEGRATION ──────────────────────────────────┐    │
│  │ NVIDIA Isaac ROS (TensorRT, Triton)                    │    │
│  │ ROBOTIS physical_ai_tools (LeRobot + ROS 2)            │    │
│  │ zenoh_ros2_sdk (ROS 2 without ROS 2)                   │    │
│  │ Custom PyTorch/VLA nodes                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─── SIMULATION ─────────────┐  ┌─── NAVIGATION ───────────┐ │
│  │ Gazebo (Harmonic/Ionic)    │  │ Nav2                     │ │
│  │ Isaac Sim / Isaac Lab      │  │ SLAM (RTAB-Map)          │ │
│  │ MuJoCo (via ROS 2 bridge) │  │ Localization (AMCL)      │ │
│  └─────────────────────────────┘  └──────────────────────────┘ │
│                                                                 │
│  ┌─── DEPLOYMENT TARGETS ─────────────────────────────────┐    │
│  │ x86_64 (Ubuntu 24.04)  │  Jetson (Orin/Thor)          │    │
│  │ ARM64 (Pi, embedded)   │  DGX Spark                    │    │
│  │ micro-ROS (ESP32, STM32, Pico)                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─── LANGUAGES ──────────────────────────────────────────┐    │
│  │ C++ (primary)  │  Python (primary)  │  Rust (new!)     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Sources

- ROS 2 Official Releases: https://docs.ros.org/en/rolling/Releases.html
- ROS 1 Distributions (EOL): https://wiki.ros.org/Distributions
- Zenoh Blog (1.0-1.7): https://zenoh.io/blog/
- rmw_zenoh_cpp: https://github.com/ros2/rmw_zenoh
- Kilted Kaiju Release Notes: https://docs.ros.org/en/kilted/Releases/Release-Kilted-Kaiju.html
- Jazzy Jalisco Release Notes: https://docs.ros.org/en/jazzy/Releases/Release-Jazzy-Jalisco.html
- NVIDIA Isaac ROS: https://developer.nvidia.com/isaac-ros
- Isaac ROS Packages: https://nvidia-isaac-ros.github.io/repositories_and_packages/index.html
- MoveIt 2: https://moveit.ai/, https://github.com/moveit/moveit2
- ros2_control: https://control.ros.org/, https://github.com/ros-controls/ros2_control
- image_pipeline: https://github.com/ros-perception/image_pipeline
- RTAB-Map ROS 2: https://github.com/introlab/rtabmap_ros
- micro-ROS: https://micro.ros.org/
- ROBOTIS physical_ai_tools: https://github.com/ROBOTIS-GIT/physical_ai_tools
- ROBOTIS ai_worker: https://github.com/ROBOTIS-GIT/ai_worker
- ROBOTIS zenoh_ros2_sdk: https://github.com/ROBOTIS-GIT/zenoh_ros2_sdk
- Autoware: https://github.com/autowarefoundation/autoware
- AnyGrasp SDK: https://github.com/graspnet/anygrasp_sdk
- FoundationPose: https://github.com/NVlabs/FoundationPose
- Azure Kinect ROS Driver (retired): https://github.com/microsoft/Azure_Kinect_ROS_Driver
- pyroboplan: https://github.com/sea-bass/pyroboplan

---

*Report compiled from live source fetches on 2026-03-07. All URLs verified accessible at time of collection.*
