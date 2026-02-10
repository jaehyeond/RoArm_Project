# RoArm M3 SmolVLA Project - Complete Dependency Map

Generated: 2026-02-07

## Python Environment

- **Python Version**: 3.10+ (tested on 3.11.x)
- **Virtual Environment**: `E:\RoArm_Project\.venv`
- **CUDA Version**: 12.4+ (check: `python -c "import torch; print(torch.version.cuda)"`)
- **GPU**: RTX 4070 Ti SUPER

## Dependency Categories

### 1. Core ML / Deep Learning

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `torch` | >=2.1.0 | PyTorch deep learning framework | All training/inference scripts |
| `transformers` | >=4.35.0 | HuggingFace Transformers (SmolVLM) | SmolVLA policy |
| `safetensors` | >=0.4.0 | Fast tensor serialization | Checkpoint loading |
| `accelerate` | >=0.25.0 | Training acceleration | LeRobot training |
| `lerobot` | latest | Vision-Language-Action framework | All VLA scripts |

**Installation:**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers safetensors accelerate
pip install git+https://github.com/huggingface/lerobot.git
```

### 2. Computer Vision

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `opencv-python` | >=4.8.0 | Image processing, display, I/O | All camera scripts |
| `pyk4a` | >=1.5.0 | Azure Kinect DK Python wrapper | Data collection, deployment |

**Installation:**
```powershell
pip install opencv-python pyk4a
```

**System Dependencies:**
- **Windows**: [Azure Kinect SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/releases)
- **Linux**: `sudo apt install libk4a1.4-dev`

### 3. Robot Hardware / Serial Communication

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `pyserial` | >=3.5 | Serial communication (USB-UART) | All robot control scripts |
| `roarm_sdk` | vendor | RoArm M3 Pro SDK (Waveshare) | All robot control scripts |

**Installation:**
```powershell
pip install pyserial
# roarm_sdk: Install from vendor package or local wheel
pip install roarm_sdk  # or: pip install path/to/roarm_sdk-*.whl
```

### 4. Data Processing

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `numpy` | >=1.24.0 | Numerical operations | All scripts |
| `pandas` | >=2.0.0 | Dataset analysis (parquet) | data_episode_quality.py |
| `pyarrow` | >=14.0.0 | Parquet file I/O | LeRobot datasets |

**Installation:**
```powershell
pip install numpy pandas pyarrow
```

### 5. User Interface / Input

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `pynput` | >=1.7.6 | Keyboard input handling | collect_data_manual.py |

**Installation:**
```powershell
pip install pynput
```

### 6. Training / Logging Utilities

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `tqdm` | >=4.65.0 | Progress bars | Training scripts |
| `rich` | >=13.0.0 | Terminal formatting | LeRobot CLI |

**Installation:**
```powershell
pip install tqdm rich
```

### 7. Optional Analysis Tools

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `jupyter` | >=1.0.0 | Notebook environment | Interactive analysis |
| `matplotlib` | >=3.7.0 | Plotting | Data visualization |
| `seaborn` | >=0.12.0 | Statistical plots | Analysis notebooks |
| `scipy` | >=1.10.0 | Scientific computing | Statistical analysis |

**Installation:**
```powershell
pip install jupyter matplotlib seaborn scipy
```

## Script-Specific Import Map

### `deploy_smolvla.py` (Real Robot Deployment)

**Imports:**
```python
# Standard library
import os, sys, argparse, time, csv, logging
from datetime import datetime
from pathlib import Path

# ML/DL
import torch
from safetensors.torch import load_file
import lerobot.policies.pretrained
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Computer vision
import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A

# Robot hardware
from roarm_sdk.roarm import roarm
import serial  # For servo scan fallback
```

**Critical Dependencies:**
- `torch` + CUDA (GPU inference)
- `lerobot` (SmolVLA policy)
- `pyk4a` (Azure Kinect)
- `roarm_sdk` (robot control)
- `opencv-python` (display overlay)

### `collect_data_manual.py` (Data Collection)

**Imports:**
```python
# Standard library
import os, json, time, datetime, logging

# Computer vision
import numpy as np
import cv2
import pyk4a
from pyk4a import Config, PyK4A

# Robot hardware
from roarm_sdk.roarm import roarm

# User input
from pynput import keyboard
```

**Critical Dependencies:**
- `pyk4a` (camera capture)
- `roarm_sdk` (robot state reading)
- `pynput` (keyboard control)
- `opencv-python` (live preview)

### `run_official_train.py` (Training Pipeline)

**Imports:**
```python
# Standard library
import os, sys, shutil
from pathlib import Path

# LeRobot internals
import lerobot.policies.pretrained
import lerobot.utils.train_utils
from lerobot.scripts.lerobot_train import main
```

**Critical Dependencies:**
- `lerobot` (full training pipeline)
- `torch` + CUDA (GPU training)
- `transformers` (model architecture)
- `accelerate` (distributed training)

**Windows-Specific Patches:**
- Path → POSIX conversion for HuggingFace repo_id
- Symlink → text file for checkpoint pointer (Developer Mode not required)

### `train_config_50k.py` (Extended Training)

**Imports:**
```python
# Standard library
import os, sys, shutil
from pathlib import Path

# LeRobot internals
import lerobot.policies.pretrained
import lerobot.utils.train_utils
from lerobot.scripts.lerobot_train import main
```

**Same as `run_official_train.py`**

### `train_eval_checkpoints.py` (Checkpoint Evaluation)

**Imports:**
```python
# Standard library
import os, sys, argparse, json
from pathlib import Path
from typing import Dict, List

# ML/DL
import torch
import numpy as np
from safetensors.torch import load_file

# LeRobot
import lerobot.policies.pretrained
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
```

**Critical Dependencies:**
- `torch` + CUDA (inference)
- `lerobot` (dataset + policy)
- `safetensors` (checkpoint loading)

### `data_episode_quality.py` (Dataset Analysis)

**Imports:**
```python
# Data processing
import pandas as pd
import numpy as np
from pathlib import Path
```

**Critical Dependencies:**
- `pandas` (parquet reading)
- `pyarrow` (backend for parquet)

### `scan_servos.py` + `reset_robot.py` (Robot Diagnostics)

**Imports:**
```python
# Standard library
import sys, time

# Hardware
import serial
```

**Critical Dependencies:**
- `pyserial` (low-level serial communication)

## Windows vs Linux Differences

### Packages with Platform Differences

| Package | Windows | Linux | Notes |
|---------|---------|-------|-------|
| `pyserial` | COM ports | /dev/ttyUSB* | Port naming only |
| `pyk4a` | Requires SDK MSI | Requires apt package | System-level dependency |
| All others | ✅ Same | ✅ Same | Cross-platform |

### Windows-Only Issues (Patched in Code)

1. **Path → POSIX conversion** (`run_official_train.py`)
   - HuggingFace repo_id uses forward slashes
   - Windows Path objects use backslashes
   - **Solution**: Monkey-patch `from_pretrained()` to convert Path to POSIX string

2. **Symlink without Developer Mode** (`run_official_train.py`)
   - LeRobot checkpoints use symlinks
   - Windows requires Developer Mode for `os.symlink()`
   - **Solution**: Replace symlink with text file pointer

3. **cp949 Encoding** (all scripts)
   - Windows default encoding is cp949 (Korean locale)
   - **Solution**: `sys.stdout.reconfigure(encoding="utf-8")`

4. **Serial Port Names** (all robot scripts)
   - Windows: `COM3`, `COM8`
   - Linux: `/dev/ttyUSB0`, `/dev/ttyUSB1`
   - **Solution**: Pass port as command-line argument

### LeRobot Dataset Format (Parquet)

The LeRobot dataset uses Apache Parquet format with specific schema:

**File Structure:**
```
lerobot_dataset_v3/
├── data/
│   └── chunk-000/
│       └── file-000.parquet
├── meta_data/
│   ├── info.json
│   └── stats.safetensors
└── videos/
    └── episode_000000.mp4 (optional)
```

**Parquet Schema (read by pandas):**
```python
df = pd.read_parquet("data/chunk-000/file-000.parquet")

# Columns:
# - episode_index: int
# - frame_index: int
# - timestamp: float
# - action: array[6] (joint angles)
# - observation.state: array[6] (current state)
# - observation.images.top: image data (stored separately or embedded)
# - task: string (task description)
```

**Dependencies for Parquet:**
- `pandas>=2.0.0`
- `pyarrow>=14.0.0` (backend)

## Installation Order (Fresh Setup)

### 1. System-Level Dependencies

**Windows:**
```powershell
# Install CUDA Toolkit 12.4+
# https://developer.nvidia.com/cuda-downloads

# Install Azure Kinect SDK
# https://github.com/microsoft/Azure-Kinect-Sensor-SDK/releases
```

**Linux (Ubuntu 22.04):**
```bash
# Install CUDA
sudo apt install nvidia-cuda-toolkit

# Install Azure Kinect SDK
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
sudo apt update
sudo apt install libk4a1.4-dev
```

### 2. Python Virtual Environment

```powershell
# Create venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux)
source .venv/bin/activate
```

### 3. Core ML Packages (CUDA)

```powershell
# Install PyTorch with CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
python -c "import torch; print(torch.version.cuda)"          # Should print 12.4
```

### 4. LeRobot Framework

```powershell
# Install from GitHub (latest)
pip install git+https://github.com/huggingface/lerobot.git

# OR local editable install (for development)
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

### 5. All Other Dependencies

```powershell
pip install -r requirements_core.txt
```

### 6. Vendor Packages

```powershell
# RoArm SDK (install from vendor or local wheel)
pip install roarm_sdk

# OR from local wheel
pip install path/to/roarm_sdk-*.whl
```

### 7. Verify Installation

```powershell
# Check all critical imports
python -c "
import torch
import transformers
import lerobot
import pyk4a
import cv2
from roarm_sdk.roarm import roarm
print('✅ All critical imports successful')
"

# Check GPU
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

## Quick Reference

### Check Installed Packages

```powershell
# List all packages
pip list

# Check specific package version
pip show torch
pip show lerobot
```

### Update Packages

```powershell
# Update specific package
pip install --upgrade torch
pip install --upgrade lerobot

# Update from requirements
pip install --upgrade -r requirements_core.txt
```

### Export Current Environment

```powershell
# Full freeze (includes all dependencies)
pip freeze > requirements_full.txt

# Manual export (recommended for portability)
# Edit requirements_core.txt with tested versions
```

## Troubleshooting

### Azure Kinect SDK Not Found

**Symptom:** `ImportError: DLL load failed while importing _k4a`

**Solution (Windows):**
1. Download SDK MSI from [Microsoft](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/releases)
2. Install to default location (`C:\Program Files\Azure Kinect SDK v1.4.1`)
3. Add to PATH: `C:\Program Files\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\bin`
4. Restart terminal

**Solution (Linux):**
```bash
sudo apt install libk4a1.4-dev
```

### CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Add to top of script
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

### RoArm SDK Import Error

**Symptom:** `ModuleNotFoundError: No module named 'roarm_sdk'`

**Solution:**
```powershell
# Check if installed
pip show roarm_sdk

# If not installed, install from vendor package
pip install roarm_sdk

# OR build/install from local source
cd path/to/roarm_sdk_source
pip install .
```

### LeRobot Dataset Loading Slow

**Symptom:** Dataset initialization takes >10 seconds

**Solution:**
- Ensure `pyarrow` is installed (faster parquet backend)
- Use `num_workers=0` on Windows (multiprocessing issue)
- Disable video loading if not needed

## File Locations

- **Core requirements**: `E:\RoArm_Project\requirements_core.txt`
- **Virtual environment**: `E:\RoArm_Project\.venv`
- **LeRobot datasets**: `E:\RoArm_Project\lerobot_dataset_v3`
- **Model checkpoints**: `E:\RoArm_Project\outputs\smolvla_official\checkpoints`
- **Pre-trained models**: `E:\RoArm_Project\models\smolvla_base`

---

**Last Updated**: 2026-02-07
**Maintainer**: RoArm M3 SmolVLA Project
