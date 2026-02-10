# RoArm Project Audit Report

**Date**: 2026-02-07
**Purpose**: Classify all Python scripts for Linux migration
**Status**: Completed SmolVLA pipeline on Windows, moving to Linux laptop

---

## 1. Active Pipeline Files (KEEP - ESSENTIAL)

These files are critical for the current SmolVLA workflow and must be migrated to Linux.

| File | Purpose | Windows Dependencies |
|------|---------|---------------------|
| **collect_data_manual.py** | Torque OFF manual data collection with Azure Kinect | pyk4a, roarm_sdk, pynput |
| **run_official_train.py** | Official lerobot-train CLI wrapper with Windows patches | Path→POSIX patch, symlink→text file |
| **deploy_smolvla.py** | Real robot deployment with SmolVLA | Path→POSIX patch, pyk4a, roarm_sdk |
| **test_inference_official.py** | Offline inference testing for trained checkpoints | Path→POSIX patch |

**Migration Priority**: HIGH
**Action**: Remove Windows patches, update paths to Linux conventions

---

## 2. Reusable Code/Patterns (ARCHIVE - REFERENCE)

These files contain useful code that could be adapted for future work.

### Data Analysis Tools (matplotlib required)

| File | Useful Parts | How to Reuse |
|------|-------------|--------------|
| **data_visualize_episodes.py** | Episode quality visualization, 4-subplot analysis | Adapt for new datasets after collection |
| **data_distribution_analysis.py** | Action distribution histograms, elbow/gripper analysis | Use to verify new dataset coverage |
| **data_distribution_simple.py** | No-matplotlib version of distribution analysis | Quick CLI analysis without GUI |
| **data_episode_quality.py** | Per-episode quality metrics (min_elbow, gripper opening) | Quality control for new collections |

**Reusable Patterns**:
- Azure Kinect integration (pyk4a)
- RoArm SDK safe read patterns (5-retry logic)
- Joint limits clamping
- Normalization stats extraction from safetensors

### Evaluation Tools

| File | Useful Parts | How to Reuse |
|------|-------------|--------------|
| **train_eval_checkpoints.py** | Multi-checkpoint comparison, L2 error analysis | Compare checkpoints during extended training |
| **analyze_action_dist.py** | Quick action distribution stats | Sanity check after data conversion |

### Training Extensions

| File | Useful Parts | How to Reuse |
|------|-------------|--------------|
| **train_config_50k.py** | Extended training configuration (50K/100K steps) | If need to continue training beyond 20K |

---

## 3. Obsolete Files (DELETE or IGNORE)

These files are from failed experiments or deprecated approaches.

### Failed Custom Training

| File | Reason | Status |
|------|--------|--------|
| **train_smolvla.py** | Custom training script WITHOUT smolvla_base pretrained model | ❌ DELETE - produces "mean action" |

**Lesson**: Custom training scripts miss critical components (preprocessor normalization, LR scheduler, pretrained Action Expert). Always use official lerobot-train CLI.

### Deprecated Data Collection

| File | Reason | Status |
|------|--------|--------|
| **collect_data.py** | Keyboard teleoperation (QWERTY 6-DOF) | ❌ ARCHIVE - replaced by manual mode |
| **test_smolvla_inference.py** | Old inference script (wrong checkpoint format) | ❌ DELETE - use test_inference_official.py |

### Isaac Sim RL Pipeline (Phase 1 - Abandoned)

| File | Reason | Status |
|------|--------|--------|
| **step1_analyze.py** | USD structure analysis | ❌ DELETE - Isaac Sim approach abandoned |
| **step2_fix_base.py** | Base fixing test (kinematic=True) | ❌ DELETE - Isaac Sim approach abandoned |
| **step3_move.py** | Full demo (load → fix → drive → motion) | ❌ DELETE - Isaac Sim approach abandoned |
| **roarm_fixed_base.py** | Isaac Sim base fixing variants | ❌ DELETE - Isaac Sim approach abandoned |
| **roarm_move_final.py** | Isaac Sim motion scripts | ❌ DELETE - Isaac Sim approach abandoned |
| **roarm_fixed.py** | Isaac Sim variants | ❌ DELETE - Isaac Sim approach abandoned |
| **roarm_stand.py** | BBox rotation fix | ❌ DELETE - Isaac Sim approach abandoned |
| **roarm_upright.py** | Standing orientation | ❌ DELETE - Isaac Sim approach abandoned |
| **roarm_move_demo.py** | Demo scripts | ❌ DELETE - Isaac Sim approach abandoned |
| **roarm_sim_v2.py** | Simulation variants | ❌ DELETE - Isaac Sim approach abandoned |

**Note**: Isaac Sim RL pipeline failed due to Sim2Real transfer issues (policy saturation, fixed-goal overfitting). Switched to VLA approach.

### ROS2 Integration (Phase 3 - Not Needed for VLA)

| File | Reason | Status |
|------|--------|--------|
| **step4_ros_bridge.py** | ROS2 bridge implementation | ❌ DELETE - VLA doesn't use ROS2 |
| **step1_camera_calibration.py** | ROS2 camera calibration | ❌ DELETE - VLA doesn't use ROS2 |
| **step2_aruco_detection.py** | ROS2 ArUco markers | ❌ DELETE - VLA doesn't use ROS2 |
| **step3_object_detection.py** | ROS2 object detection | ❌ DELETE - VLA doesn't use ROS2 |
| **step5_vision_grasping.py** | ROS2 vision grasping | ❌ DELETE - VLA doesn't use ROS2 |

### Leader-Follower Experiments (Phase 6.3 - Hardware Constraint)

| File | Reason | Status |
|------|--------|--------|
| **step1_test_each_arm.py** | Dual-arm connection test | ⚠️ ARCHIVE - only 1 robot available |
| **step2_leader_follower.py** | Standalone L-F mode test | ⚠️ ARCHIVE - only 1 robot available |

**Note**: Leader-Follower mode implemented but unused (팔 1개만 보유). Using torque OFF manual mode instead.

### Utility Scripts (KEEP)

| File | Purpose | Status |
|------|--------|--------|
| **scan_servos.py** | ESP32 reset via T:106 servo scan | ✅ KEEP - fixes motor response issues |
| **roarm_demo.py** | Simple control wrapper class | ✅ KEEP - reference implementation |

---

## 4. Windows-Specific Code to Remove on Linux

### A. Path → POSIX Conversion Patch

**Location**: `run_official_train.py`, `deploy_smolvla.py`, `test_inference_official.py`, `train_eval_checkpoints.py`, `train_config_50k.py`

**Current Code**:
```python
# Windows Path 문제 해결: pretrained_path가 Path 타입으로 변환되면
# lerobot/smolvla_base → lerobot\smolvla_base 가 되어 HF repo_id 인식 실패
import lerobot.policies.pretrained as _pretrained
_original_from_pretrained = _pretrained.PreTrainedPolicy.from_pretrained.__func__

@classmethod
def _patched_from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
    if isinstance(pretrained_name_or_path, Path):
        pretrained_name_or_path = pretrained_name_or_path.as_posix()
    return _original_from_pretrained(cls, pretrained_name_or_path, *args, **kwargs)

_pretrained.PreTrainedPolicy.from_pretrained = _patched_from_pretrained
```

**Linux Fix**: REMOVE ENTIRELY
Linux uses forward slashes natively, no conversion needed.

---

### B. Symlink → Text File Patch

**Location**: `run_official_train.py`, `train_config_50k.py`

**Current Code**:
```python
# Windows symlink 문제 해결: update_last_checkpoint()에서 symlink_to() 사용
# Windows는 Developer Mode 없이 symlink 생성 불가
# 해결: symlink 대신 텍스트 파일로 포인터 저장
import shutil
import lerobot.utils.train_utils as _train_utils

def _patched_update_last_checkpoint(checkpoint_dir: Path) -> Path:
    """Windows-compatible: write text file pointer instead of symlink."""
    last_checkpoint_dir = checkpoint_dir.parent / _train_utils.LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink() or last_checkpoint_dir.is_file():
        last_checkpoint_dir.unlink()
    elif last_checkpoint_dir.is_dir():
        shutil.rmtree(last_checkpoint_dir)
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.write_text(str(relative_target))
    return last_checkpoint_dir

_train_utils.update_last_checkpoint = _patched_update_last_checkpoint
```

**Linux Fix**: REMOVE ENTIRELY
Linux supports symlinks natively, no patching needed.

---

### C. Encoding Fixes

**Location**: `run_official_train.py`, `deploy_smolvla.py`, `test_inference_official.py`, `train_config_50k.py`

**Current Code**:
```python
# Windows cp949 인코딩 문제 해결
os.environ["PYTHONIOENCODING"] = "utf-8"

# stdout/stderr UTF-8 강제 + unbuffered for piped output
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
```

**Linux Fix**: SIMPLIFY TO UNBUFFERED ONLY
```python
# Linux uses UTF-8 by default, only need unbuffered for GPU monitoring
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
```

**Reason**: Python output buffering was severe on Windows when piped, making GPU utilization the only way to monitor training progress.

---

### D. HuggingFace Symlink Warning Suppression

**Location**: `run_official_train.py`, `train_config_50k.py`

**Current Code**:
```python
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
```

**Linux Fix**: REMOVE
Not needed on Linux where symlinks work properly.

---

## 5. Lessons Learned (Mistakes NOT to Repeat)

### A. Custom Training Script Failure

**What Happened**: Created `train_smolvla.py` from scratch (3 attempts, all failed)

| Attempt | Config | Result |
|---------|--------|--------|
| 1 | batch_size=1, vlm=False | Mean action (gradient noise) |
| 2 | batch_size=8, vlm=False | Mean action (hyperparameter issues) |
| 3 | batch_size=8, vlm=True | Mean action (Action Expert random init) |

**Root Causes**:
1. **Action Expert Random Init**: Created new `SmolVLAConfig()` → random Action Expert weights
   - Official pipeline uses `lerobot/smolvla_base` (487 datasets, 10M frames pretrained)
2. **Missing Normalization**: Skipped `preprocessor(batch)` MEAN_STD normalization
3. **No LR Scheduler**: Used fixed lr instead of cosine decay + warmup

**Lesson**: NEVER write custom training scripts. Use official `lerobot-train` CLI.

**Evidence**: Loss 0.57 → looked good, but outputs were `[~0-2, ~30-34, ~-0.4-0.7, ~64-68, ~7-8, ~1.6-1.9]` (dataset mean)

---

### B. Pretrained Model Misunderstanding

**Mistake**: Thought "load_vlm_weights=True" was sufficient
**Reality**: Action Expert ALSO needs pretraining (not just VLM)

**Correct Approach**:
```python
# WRONG (creates random Action Expert)
config = SmolVLAConfig()
policy = SmolVLAPolicy(config, dataset_stats=stats)

# RIGHT (loads pretrained Action Expert + VLM)
SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

---

### C. Normalization Bugs

**Issue**: Manual normalization was incomplete

**Missing Steps**:
1. State normalization: `(state - state_mean) / (state_std + 1e-8)`
2. Action unnormalization: `raw_action * action_std + action_mean`
3. Preprocessor pipeline: Official pipeline auto-handles this

**Correct Path**: Let official pipeline handle normalization via preprocessor/postprocessor

---

### D. n_action_steps Misunderstanding

**Initial Belief**: `n_action_steps=50` means "use 50-step chunks"
**Reality**: `n_action_steps=1` = closed-loop (new inference every step)

**Best Practice**:
- `n_action_steps=1`: True closed-loop, uses latest observation
- `n_action_steps=50`: Open-loop, executes entire chunk without new obs

**Deployment Insight**: Closed-loop (`n_action_steps=1`) performed better than open-loop in real robot tests

---

### E. Start Position Matters (OOD Problem)

**Discovery**: Starting from `[0,0,0,0,0,0]` = out-of-distribution → timid movements

| Start Position | Elbow Change | Shoulder Change | Status |
|----------------|--------------|-----------------|--------|
| Zero `[0,0,0,0,0,0]` | 13° | 2° | ❌ OOD, timid |
| Dataset Mean `[-1,49,25,50,-2,22]` | 34° | 34° | ✅ In-distribution |

**Lesson**: Always start from dataset mean position for in-distribution behavior

---

### F. Loss Decrease ≠ Good Model

**Observation**: Loss 0.57 → model still outputs mean action
**Conclusion**: Loss can decrease while model learns trivial solution (dataset mean)

**Better Metrics**:
1. L2 error on test samples
2. Z-score range (need ±3.0 for elbow control)
3. Action diversity (std across predictions)
4. Real robot performance

---

### G. Robot Motor Response Issues

**Symptom**: Motor bus not initialized, `joints_angle_get()` returns `[180, -180, -90, -180, 180, 180]`

**Fix**: Send `T:106` servo scan command → ESP32 crashes and resets → motor bus reinitializes

**Script**: `scan_servos.py` automates this fix

**Alternative**: `torque_set(cmd=1)` + `move_init()` if torque is OFF

---

## 6. Configuration Files

### A. Hardware Port Changes

| Hardware | Windows | Linux (WSL2) |
|----------|---------|--------------|
| RoArm M3 | COM3, COM8 | /dev/ttyUSB0, /dev/ttyUSB1 |
| Azure Kinect | Auto-detected | Auto-detected (pyk4a) |

**Action**: Update all `port="COM8"` → `port="/dev/ttyUSB0"` or use auto-detection

---

### B. Path Changes

| Purpose | Windows | Linux |
|---------|---------|-------|
| Dataset | `E:/RoArm_Project/lerobot_dataset_v3` | `~/roarm_project/lerobot_dataset_v4` |
| Outputs | `E:/RoArm_Project/outputs` | `~/roarm_project/outputs` |
| Models | `E:/RoArm_Project/models/smolvla_base` | `~/roarm_project/models/smolvla_base` |
| Logs | `E:/RoArm_Project/logs` | `~/roarm_project/logs` |

**Action**: Use `Path.home() / "roarm_project"` for portability

---

### C. Old Dataset Obsolescence

| Dataset | Frames | Status | Reason |
|---------|--------|--------|--------|
| lerobot_dataset_v3 | 13,010 | ❌ OBSOLETE | Camera position changed |

**Action**: Recollect entire dataset with new camera position

**Impact**: 51 episodes × ~255 frames = ~13,000 frames to recollect
**Estimated Time**: 2-3 days (51 episodes @ 30 FPS, ~8s/episode + setup)

---

## 7. Migration Checklist

### Pre-Migration (Windows)

- [ ] Backup trained checkpoints: `outputs/smolvla_official/checkpoints/020000/`
- [ ] Copy pretrained model: `models/smolvla_base/`
- [ ] Save analysis results: `data_episode_quality_results.csv`, `*.png` visualizations
- [ ] Document camera position (for recalibration on Linux)

### Linux Setup

- [ ] Install dependencies: `pyk4a`, `roarm_sdk`, `lerobot`, `torch`, `transformers`
- [ ] USB passthrough: `usbipd attach --wsl --busid=<BUSID>` (if using WSL2)
- [ ] Azure Kinect permissions: `sudo chmod 666 /dev/bus/usb/*/*`
- [ ] Test camera: `python -c "from pyk4a import PyK4A; k4a = PyK4A(); k4a.start(); print('OK')"`
- [ ] Test robot: `python -c "from roarm_sdk.roarm import roarm; arm = roarm('roarm_m3', '/dev/ttyUSB0'); print('OK')"`

### Code Migration

- [ ] Remove Path→POSIX patches
- [ ] Remove symlink→text file patches
- [ ] Simplify encoding fixes (unbuffered only)
- [ ] Update all `COM8` → `/dev/ttyUSB0` or auto-detect
- [ ] Update all `E:/RoArm_Project` → `~/roarm_project`
- [ ] Test `run_official_train.py` on dummy dataset
- [ ] Test `deploy_smolvla.py` in dry-run mode

### Data Recollection

- [ ] Setup new camera position
- [ ] Calibrate workspace boundaries
- [ ] Run `collect_data_manual.py` for 51+ episodes
- [ ] Validate with `data_episode_quality.py` (elbow < -20°, gripper > 30°)
- [ ] Visualize with `data_visualize_episodes.py`

### Training

- [ ] Convert to LeRobot v3.0 format
- [ ] Verify normalization stats
- [ ] Run official training: `python run_official_train.py`
- [ ] Monitor GPU utilization (no stdout buffering issues on Linux)
- [ ] Evaluate checkpoints: `python train_eval_checkpoints.py`

### Deployment

- [ ] Test offline inference: `python test_inference_official.py`
- [ ] Test real robot: `python deploy_smolvla.py --dry-run`
- [ ] Test real robot: `python deploy_smolvla.py --start-pos dataset_mean`
- [ ] Verify start position: use dataset mean `[-1,49,25,50,-2,22]`
- [ ] Monitor convergence: `--convergence-threshold 0.5 --convergence-window 10`

---

## 8. File Organization Recommendation

### Keep (Migrate to Linux)

```
roarm_project/
├── collect_data_manual.py          # Data collection (torque OFF)
├── run_official_train.py           # Training wrapper (remove patches)
├── deploy_smolvla.py               # Deployment (remove patches)
├── test_inference_official.py      # Offline testing (remove patches)
├── scan_servos.py                  # Motor reset utility
├── roarm_demo.py                   # Reference implementation
└── analysis/
    ├── data_visualize_episodes.py
    ├── data_distribution_analysis.py
    ├── data_episode_quality.py
    ├── train_eval_checkpoints.py
    └── analyze_action_dist.py
```

### Archive (Reference Only)

```
archive/
├── isaac_sim/                      # Phase 1: Isaac Sim RL
│   ├── step1_analyze.py
│   ├── step2_fix_base.py
│   ├── step3_move.py
│   └── roarm_*.py
├── ros2/                           # Phase 3: ROS2 bridge
│   ├── step1_camera_calibration.py
│   ├── step2_aruco_detection.py
│   ├── step3_object_detection.py
│   ├── step4_ros_bridge.py
│   └── step5_vision_grasping.py
├── leader_follower/                # Phase 6.3: Dual-arm
│   ├── step1_test_each_arm.py
│   └── step2_leader_follower.py
└── failed_training/                # Phase 7.2a: Custom training
    └── train_smolvla.py
```

### Delete (No Value)

```
- collect_data.py                   # Keyboard teleoperation (replaced)
- test_smolvla_inference.py         # Old inference script (wrong format)
- train_config_50k.py               # Extended training (not needed yet)
```

---

## 9. Summary

### Active Pipeline (3 Scripts)

1. `collect_data_manual.py` → Azure Kinect + RoArm M3 + torque OFF
2. `run_official_train.py` → lerobot-train CLI (20K steps, loss 0.009)
3. `deploy_smolvla.py` → Real robot with SmolVLA (10ms/step, closed-loop)

### Critical Lessons

1. **Use official lerobot-train CLI** (custom scripts = mean action)
2. **Load smolvla_base pretrained** (Action Expert + VLM both need pretraining)
3. **Start from dataset mean** (avoid OOD = timid behavior)
4. **Closed-loop > Open-loop** (n_action_steps=1 better than 50)
5. **Loss decrease ≠ good model** (use L2 error, z-score range, diversity)

### Windows Patches to Remove

1. Path→POSIX conversion (lines 22-33 in multiple files)
2. Symlink→text file workaround (lines 38-51 in training scripts)
3. Encoding fixes (simplify to unbuffered only)
4. HuggingFace symlink warning suppression

### Migration Impact

- **Old dataset obsolete** (camera position changed → 13,010 frames lost)
- **Recollection needed** (~2-3 days for 51 episodes)
- **No training needed** (20K checkpoint transferable)
- **Deployment ready** (SmolVLA works on Linux with pyk4a + roarm_sdk)

---

**Next Steps**:
1. Setup Linux environment (dependencies + hardware)
2. Remove Windows patches from 4 core scripts
3. Recollect dataset with new camera position
4. Validate with episode quality analysis
5. Fine-tune for 20K-50K steps
6. Deploy to real robot

**Estimated Timeline**: 1 week (3 days collection + 2 days training + 2 days testing)
