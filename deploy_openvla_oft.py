"""
OpenVLA-OFT 7B (LoRA + L1 action head) 실제 로봇 배포.

학습: B200, v6 LeRobot (50ep, "Pick up the sponge"), 30K steps, batch=8, LR 5e-4, LoRA r=32.
오프라인 eval (12/30K 중 8 ckpt, 2026-05-22): best = step 7500, holdout L2 22.16°,
catastrophic collapse 7500→10000 (22.16°→70.07°, both train+holdout). Do NOT deploy 10K+.

SmolVLA v6 4/9 Plan 3 baseline과 동일한 하드웨어/안전 셋업 (head-to-head 비교용):
- Follower 전용 (/dev/ttyUSB1) — Leader (USB0)에는 명령 안 보냄
- INIT_POS [0,0,90,0,0,5] (HOME, v6 학습 분포 시작 위치)
- JOINT_SPEED_CAPS gripper-only unlock (Plan 3 핵심)
- Z_FLOOR/DIST workspace safety

OpenVLA-OFT vs SmolVLA 차이:
- Input: RGB 224×224 PIL + language prompt만 (state 사용 안 함)
- Output: action chunk (8, 6) unnormalized (BOUNDS_Q99 denorm은 vla.predict_action 안에서)
- Inference time: ~1.2s/frame (B200) → 4090에서 ~3-5s/frame 예상 → chunk-mode default (n_action_steps=8 → effective ~2 Hz)

사용 예:
    # chunk-mode (default, 권장)
    python deploy_openvla_oft.py --max-steps 64

    # closed-loop (매 step 새 추론, 매우 느림 — 비교용)
    python deploy_openvla_oft.py --n-action-steps 1 --max-steps 32

    # Dry-run (로봇 명령 안 보냄, sanity)
    python deploy_openvla_oft.py --dry-run --max-steps 8

    # CPU mode (driver 미해결 시)
    python deploy_openvla_oft.py --device cpu --dry-run --max-steps 1
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import csv
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# SDK 로그 억제 + DataProcessor monkey-patch (CLAUDE.md 패턴, lambda 금지)
logging.getLogger("BaseController").setLevel(logging.CRITICAL)


def _patch_sdk_silence():
    from roarm_sdk.common import DataProcessor, JsonCmd, handle_m3_feedback

    def _silent_process(self, data, genre):
        if not data:
            return None
        res, valid = [], []
        if genre == JsonCmd.FEEDBACK_GET:
            valid = [data["x"], data["y"], data["z"]]
            if self.type == "roarm_m3":
                valid = handle_m3_feedback(valid, data)
        else:
            valid = data
        res.append(valid)
        return res

    DataProcessor._process_received = _silent_process


# --------------------------------------------------------------------------- #
# Constants — RoArm M3 / v6 dataset                                           #
# --------------------------------------------------------------------------- #

# RoArm M3 joint range (safety)
JOINT_LIMITS = [
    (-180, 180),
    (-110, 110),
    (-70, 190),
    (-110, 110),
    (-180, 180),
    (-10, 100),
]

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]

# Per-joint speed caps. Distal joints (wrist_pitch, wrist_roll, gripper) = 300 max.
# SDK does not support per-joint speed, so min() is sent for joints_angle_ctrl.
# Plan 3 (2026-04-09 SmolVLA SUCCESS): gripper 호출을 별도로 speed=1000 unlock.
JOINT_SPEED_CAPS = [500, 500, 500, 300, 300, 300]

# Workspace safety (deploy_smolvla.py:83-84, 2026-03-26 실측 기반)
Z_FLOOR_DEPLOY = -130  # mm — 책상 -120 + 10mm 여유
DIST_MAX_DEPLOY = 420  # mm — 책상 밖 이탈 방지

# HOME (v6 학습 분포 시작 위치). 4/9 Plan 3 SUCCESS와 동일.
INIT_POS = [0, 0, 90, 0, 0, 5]

# OpenVLA-OFT / v6
BASE_MODEL_DEFAULT = "openvla/openvla-7b"
BASE_REVISION_DEFAULT = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
CHECKPOINT_PATH_DEFAULT = (
    "openvla_oft_b200_pulls/"
    "openvla-7b+roarm_v6_pick+b8+lr-0.0005+lora-r32+dropout-0.0--v6_30k--7500_chkpt"
)
UNNORM_KEY_DEFAULT = "roarm_v6_pick"

# ROARM_M3 constants (prismatic.vla.constants.ROARM_M3_CONSTANTS과 일치)
ACTION_DIM = 6
NUM_ACTIONS_CHUNK = 8
IMAGE_SIZE = 224


# --------------------------------------------------------------------------- #
# L1RegressionActionHead inline (avoids dlimp/RLDS chain via prismatic.models) #
# --------------------------------------------------------------------------- #
import torch.nn as nn


class _MLPResNetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.ffn(x)


class _MLPResNet(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([_MLPResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for blk in self.mlp_resnet_blocks:
            x = blk(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class L1RegressionActionHead(nn.Module):
    """Inline copy of prismatic.models.action_heads.L1RegressionActionHead.

    Bypasses dlimp / RLDS / diffusers chain triggered by `prismatic.models.__init__`.
    Architecture matches the trained head exactly (MLPResNet num_blocks=2,
    input_dim=hidden_dim*ACTION_DIM, output_dim=action_dim).
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 4096, action_dim: int = ACTION_DIM):
        super().__init__()
        self.action_dim = action_dim
        self.model = _MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.model(rearranged)


# --------------------------------------------------------------------------- #
# sdpa class-attr patch (D071) — must run before AutoModel instantiation       #
# --------------------------------------------------------------------------- #
def apply_sdpa_class_attr_patch():
    import importlib

    for mod_name in ("prismatic.extern.hf.modeling_prismatic",):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name in dir(mod):
            if not name.endswith("PreTrainedModel"):
                continue
            cls = getattr(mod, name, None)
            if isinstance(cls, type) and "PrismaticPreTrainedModel" in cls.__name__:
                cls._supports_sdpa = True


def ensure_roarm_constants(unnorm_key: str):
    """Inject `--dataset_name <key>` into argv so prismatic constants pick ROARM_M3."""
    if "roarm" not in unnorm_key.lower():
        raise SystemExit(f"--unnorm_key must contain 'roarm' (got {unnorm_key})")
    if not any("roarm" in a.lower() for a in sys.argv):
        sys.argv.insert(1, f"--dataset_name={unnorm_key}")


# --------------------------------------------------------------------------- #
# Robot helpers (mirror deploy_smolvla.py)                                    #
# --------------------------------------------------------------------------- #
def clamp_joints(angles):
    return [max(lo, min(hi, a)) for a, (lo, hi) in zip(angles, JOINT_LIMITS)]


def get_safe_speed(base_speed: int) -> int:
    return min(base_speed, min(JOINT_SPEED_CAPS))


def get_robot_angles(arm, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            angles = arm.joints_angle_get()
            if angles is not None and len(angles) >= 6:
                return list(angles[:6])
        except (KeyError, TypeError, AttributeError):
            if attempt < max_retries - 1:
                time.sleep(0.05)
    return None


def get_robot_fk_pose(arm, max_retries: int = 3):
    for _ in range(max_retries):
        try:
            pose = arm.pose_get()
            if pose and len(pose) >= 3:
                return pose
        except Exception:
            time.sleep(0.05)
    return None


# --------------------------------------------------------------------------- #
# Model load                                                                  #
# --------------------------------------------------------------------------- #
def load_openvla_oft(
    base_model: str,
    revision: str,
    checkpoint_dir: Path,
    unnorm_key: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """Load openvla-7b base + LoRA adapter + L1 action head + norm stats inject."""
    print("=" * 60)
    print("OpenVLA-OFT 7B 모델 로딩")
    print("=" * 60)
    print(f"  base   = {base_model}@{revision}")
    print(f"  ckpt   = {checkpoint_dir}")
    print(f"  device = {device}, dtype = {dtype}")

    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right",
        revision=revision,
        local_files_only=False,  # 첫 실행 시 HF cache 다운로드 허용
    )
    apply_sdpa_class_attr_patch()
    t0 = time.time()
    vla = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        revision=revision,
        local_files_only=False,
    ).to(device)
    apply_sdpa_class_attr_patch()  # second pass after dynamic class realisation
    print(f"  base load: {time.time()-t0:.1f}s")

    t0 = time.time()
    vla = PeftModel.from_pretrained(vla, str(checkpoint_dir / "lora_adapter"))
    vla = vla.to(device=device, dtype=dtype).eval()
    print(f"  lora load: {time.time()-t0:.1f}s")

    # Inject norm_stats (BOUNDS_Q99 denorm uses q01/q99 from chkpt dataset_statistics.json)
    import json

    stats_path = checkpoint_dir / "dataset_statistics.json"
    with stats_path.open() as f:
        raw_stats = json.load(f)
    norm_stats = {}
    for k, v in raw_stats.items():
        norm_stats[k] = {
            "action": {
                "q01": np.asarray(v["action"]["q01"], dtype=np.float32),
                "q99": np.asarray(v["action"]["q99"], dtype=np.float32),
                "mean": np.asarray(v["action"]["mean"], dtype=np.float32),
                "std": np.asarray(v["action"]["std"], dtype=np.float32),
                "min": np.asarray(v["action"]["min"], dtype=np.float32),
                "max": np.asarray(v["action"]["max"], dtype=np.float32),
                "mask": np.ones(len(v["action"]["q01"]), dtype=bool),
            }
        }
    underlying = vla
    if hasattr(vla, "base_model") and hasattr(vla.base_model, "model"):
        underlying = vla.base_model.model
    underlying.norm_stats = norm_stats
    try:
        vla.norm_stats = norm_stats
    except Exception:
        pass
    try:
        vla.base_model.norm_stats = norm_stats
    except Exception:
        pass

    # Action head
    llm_dim = (
        vla.config.text_config.hidden_size
        if hasattr(vla.config, "text_config")
        else vla.config.hidden_size
    )
    action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    ah_files = list(checkpoint_dir.glob("action_head--*_checkpoint.pt"))
    if not ah_files:
        raise SystemExit(f"No action_head checkpoint in {checkpoint_dir}")
    ah_path = sorted(ah_files)[0]
    print(f"  action_head = {ah_path.name}")
    state = torch.load(ah_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    action_head.load_state_dict(state, strict=True)
    action_head = action_head.to(device=device, dtype=dtype).eval()

    if unnorm_key not in norm_stats:
        raise SystemExit(
            f"unnorm_key '{unnorm_key}' not in dataset_statistics keys {list(norm_stats.keys())}"
        )
    print(f"  unnorm_key = {unnorm_key}  q01={norm_stats[unnorm_key]['action']['q01']}")
    print(f"                              q99={norm_stats[unnorm_key]['action']['q99']}")
    print("  로딩 완료.")
    return vla, action_head, processor


# --------------------------------------------------------------------------- #
# Inference                                                                   #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def predict_chunk(
    vla,
    action_head,
    processor,
    image_bgr: np.ndarray,
    task: str,
    unnorm_key: str,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """Run OpenVLA-OFT and return (NUM_ACTIONS_CHUNK, ACTION_DIM) unnormalized action chunk."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb).convert("RGB")
    if pil.size != (IMAGE_SIZE, IMAGE_SIZE):
        pil = pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    prompt = f"In: What action should the robot take to {task.lower()}?\nOut:"
    inputs = processor(prompt, pil).to(device, dtype=dtype)
    action, _ = vla.predict_action(
        **inputs,
        unnorm_key=unnorm_key,
        do_sample=False,
        action_head=action_head,
    )
    return np.asarray(action, dtype=np.float32)


# --------------------------------------------------------------------------- #
# CSV logger / convergence detector (mirror deploy_smolvla.py)                 #
# --------------------------------------------------------------------------- #
class ConvergenceDetector:
    def __init__(self, threshold: float = 0.5, window: int = 10):
        self.threshold = threshold
        self.window = window
        self.delta_history = []

    def update(self, current_angles, prev_angles):
        deltas = [abs(c - p) for c, p in zip(current_angles, prev_angles)]
        max_delta = max(deltas)
        self.delta_history.append(max_delta)
        if len(self.delta_history) > self.window:
            self.delta_history.pop(0)
        return deltas, max_delta

    def is_converged(self) -> bool:
        if len(self.delta_history) < self.window:
            return False
        return all(d < self.threshold for d in self.delta_history)


class CSVLogger:
    FIELDS = [
        "step", "timestamp", "chunk_idx", "chunk_step",
        *JOINT_NAMES,
        *[f"state_{n}" for n in JOINT_NAMES],
        *[f"delta_{n}" for n in JOINT_NAMES],
        "max_delta", "convergence", "inference_ms",
        "fk_x", "fk_y", "fk_z",
    ]

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.file = None
        self.writer = None

    def open(self):
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.file.flush()

    def log(self, step, chunk_idx, chunk_step, action, state, deltas, max_delta,
            convergence, inference_ms, fk_pose):
        if self.writer is None:
            return
        if state is None:
            state = [float("nan")] * 6
        fk_x, fk_y, fk_z = (fk_pose[0], fk_pose[1], fk_pose[2]) if fk_pose else (0, 0, 0)
        row = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "chunk_idx": chunk_idx,
            "chunk_step": chunk_step,
            **{JOINT_NAMES[i]: action[i] for i in range(6)},
            **{f"state_{JOINT_NAMES[i]}": state[i] for i in range(6)},
            **{f"delta_{JOINT_NAMES[i]}": deltas[i] for i in range(6)},
            "max_delta": max_delta,
            "convergence": convergence,
            "inference_ms": inference_ms,
            "fk_x": fk_x, "fk_y": fk_y, "fk_z": fk_z,
        }
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


# --------------------------------------------------------------------------- #
# OpenCV overlay                                                              #
# --------------------------------------------------------------------------- #
def draw_overlay(frame, step, max_steps, chunk_idx, action, max_delta, elapsed, task, inference_ms):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Step {step}/{max_steps} (chunk {chunk_idx})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if inference_ms is not None:
        cv2.putText(frame, f"{inference_ms:.0f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"t={elapsed:.1f}s d={max_delta:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, name in enumerate(JOINT_NAMES):
        cv2.putText(frame, f"{name[:3]}:{action[i]:+6.1f}", (w - 160, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    cv2.putText(frame, task, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="OpenVLA-OFT 7B 실제 로봇 배포 (ckpt 7500)")
    p.add_argument("--port", default="/dev/ttyUSB1",
                   help="Follower 시리얼 (USB1=Follower; USB0=Leader 사용 금지)")
    p.add_argument("--task", default="Pick up the sponge", help="언어 태스크")
    p.add_argument("--max-steps", type=int, default=64,
                   help="총 step 수 (chunk-mode 64 = 8 chunks × 8)")
    p.add_argument("--n-action-steps", type=int, default=NUM_ACTIONS_CHUNK,
                   choices=[1, 2, 4, 8],
                   help="chunk 안에서 사용할 action 개수 (8=chunk-mode default, 1=closed-loop)")
    p.add_argument("--speed", type=int, default=500,
                   help="joints_angle_ctrl 속도 (실 effective = min(speed, JOINT_SPEED_CAPS))")
    p.add_argument("--acc", type=int, default=200, help="joints_angle_ctrl 가속도")
    p.add_argument("--hz", type=float, default=2.0,
                   help="제어 루프 Hz (chunk-mode: chunk 안에서는 명령 sleep, 새 inference 사이는 GPU 시간)")
    p.add_argument("--dry-run", action="store_true", help="로봇 명령 안 보냄")
    p.add_argument("--start-pos", default="init",
                   choices=["init", "current", "zero"],
                   help="init=[0,0,90,0,0,5] HOME (v6 분포), current=현재유지, zero=[0]*6")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--checkpoint", default=CHECKPOINT_PATH_DEFAULT)
    p.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    p.add_argument("--revision", default=BASE_REVISION_DEFAULT)
    p.add_argument("--unnorm-key", default=UNNORM_KEY_DEFAULT)
    p.add_argument("--log-csv", nargs="?", const="auto", default=None,
                   help="CSV 경로 (없으면 auto: logs/deploy_oft_YYYYMMDD_HHMMSS.csv)")
    p.add_argument("--save-frames-dir", default=None, help="step별 Kinect frame PNG 저장 경로")
    p.add_argument("--no-kinect", action="store_true",
                   help="Kinect 없이 sanity (검은 frame 사용, dry-run과 함께)")
    p.add_argument("--convergence-threshold", type=float, default=0.5)
    p.add_argument("--convergence-window", type=int, default=10)
    args = p.parse_args()

    ensure_roarm_constants(args.unnorm_key)

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA 미가용 → CPU 전환 (OpenVLA-OFT 7B CPU 추론은 매우 느림)")
        args.device = "cpu"
    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    if device.type == "cpu" and args.dtype != "float32":
        print("NOTE: CPU에서 bf16/fp16 지원 제한적. float32 권장 (하지만 더 느림).")

    # Safety: Follower-only
    if args.port == "/dev/ttyUSB0":
        raise SystemExit("ERROR: /dev/ttyUSB0은 Leader. Follower는 /dev/ttyUSB1.")

    # CSV
    csv_logger = None
    if args.log_csv:
        if args.log_csv == "auto":
            Path("logs").mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = Path("logs") / f"deploy_oft_{ts}.csv"
        else:
            csv_path = Path(args.log_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_logger = CSVLogger(csv_path)
        csv_logger.open()
        print(f"CSV log: {csv_path}")

    frames_dir = None
    if args.save_frames_dir:
        frames_dir = Path(args.save_frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Frames: {frames_dir}/step_NNNN.png")

    conv = ConvergenceDetector(args.convergence_threshold, args.convergence_window)

    # 1. Model
    checkpoint_dir = Path(args.checkpoint).resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"Checkpoint dir missing: {checkpoint_dir}")
    vla, action_head, processor = load_openvla_oft(
        args.base_model, args.revision, checkpoint_dir, args.unnorm_key, device, dtype,
    )

    # 2. Robot (Follower-only)
    arm = None
    if not args.dry_run:
        _patch_sdk_silence()
        from roarm_sdk.roarm import roarm as RoArm
        print(f"\nFollower 연결: {args.port}")
        arm = RoArm(roarm_type="roarm_m3", port=args.port, baudrate=115200)
        time.sleep(0.5)
        angs = get_robot_angles(arm)
        if angs is None:
            print("Follower 응답 없음. ESP32 reset 시도...")
            import serial
            ser = serial.Serial(args.port, 115200, timeout=2)
            ser.write(b'{"T":106}\n')
            time.sleep(1)
            ser.close()
            print("리셋 완료. 전원 OFF→ON 후 다시 실행.")
            return
        print(f"  현재 관절: {[round(a, 1) for a in angs]}")
        # Torque ON (Follower)
        arm.torque_set(cmd=1)
        time.sleep(0.3)

    # 3. Kinect
    k4a = None
    if not args.no_kinect:
        print("\nAzure Kinect 연결...")
        import pyk4a
        from pyk4a import Config, PyK4A
        k4a = PyK4A(Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        ))
        k4a.start()
        time.sleep(1)
        cap = k4a.get_capture()
        if cap.color is None:
            print("Kinect 프레임 없음!")
            k4a.stop()
            if arm is not None:
                arm.disconnect()
            return
        print(f"  Kinect 프레임 shape: {cap.color.shape}")

    # 4. Start position
    if args.start_pos == "init":
        start = INIT_POS
    elif args.start_pos == "zero":
        start = [0] * 6
    else:
        start = None
    if start is not None and arm is not None and not args.dry_run:
        print(f"\nFollower → {start}")
        arm.joints_angle_ctrl(angles=start, speed=300, acc=100)
        for i in range(20):
            time.sleep(0.5)
            cur = get_robot_angles(arm)
            if cur is None:
                continue
            md = max(abs(c - t) for c, t in zip(cur, start))
            if md < 5.0:
                print(f"  도달 ({(i+1)*0.5:.1f}s, max diff={md:.1f}°)")
                break
        else:
            print(f"  WARNING: 10s 후 미도달 (current={cur})")

    # 5. Loop
    n_act = args.n_action_steps
    n_chunks = math.ceil(args.max_steps / n_act)
    print("\n" + "=" * 60)
    print(f"  Task            : \"{args.task}\"")
    print(f"  Checkpoint      : {checkpoint_dir}")
    print(f"  Max steps       : {args.max_steps}  (chunks: {n_chunks} × n_action_steps={n_act})")
    print(f"  Device / dtype  : {device} / {dtype}")
    print(f"  Speed cap       : {get_safe_speed(args.speed)} (main joints), gripper unlocked to 1000 (Plan 3)")
    print(f"  Dry-run         : {args.dry_run}")
    print("  [Ctrl+C 종료]")
    print("=" * 60)

    loop_interval = 1.0 / args.hz
    start_time = time.time()
    inference_times = []
    prev_angles = start if start is not None else (get_robot_angles(arm) if arm else [0]*6)
    global_step = 0
    abort = False

    try:
        for chunk_idx in range(n_chunks):
            if abort or global_step >= args.max_steps:
                break

            # Capture
            if k4a is not None:
                cap = k4a.get_capture()
                frame_bgr = np.ascontiguousarray(cap.color[:, :, :3]) if cap.color is not None else None
                if frame_bgr is None:
                    print(f"  [chunk {chunk_idx}] Kinect 프레임 누락, skip")
                    continue
            else:
                frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)

            current_angles = get_robot_angles(arm) if arm is not None else prev_angles

            # Inference
            t0 = time.time()
            chunk = predict_chunk(
                vla, action_head, processor,
                image_bgr=frame_bgr, task=args.task,
                unnorm_key=args.unnorm_key, device=device, dtype=dtype,
            )
            inference_ms = (time.time() - t0) * 1000.0
            inference_times.append(inference_ms)
            print(f"\n=== chunk {chunk_idx+1}/{n_chunks}  inference={inference_ms:.0f}ms  shape={chunk.shape} ===")
            print(f"  chunk[0] (will execute first): {[round(float(x), 2) for x in chunk[0]]}")
            if n_act > 1:
                print(f"  chunk[{n_act-1}] (will execute last):  {[round(float(x), 2) for x in chunk[n_act-1]]}")

            # Execute first n_act actions of the chunk
            for ci in range(n_act):
                if global_step >= args.max_steps:
                    break
                loop_t0 = time.time()
                global_step += 1
                step = global_step

                action_raw = chunk[ci, :ACTION_DIM].tolist()
                action_clamped = clamp_joints(action_raw)

                deltas, max_delta = conv.update(action_clamped, prev_angles)
                is_conv = conv.is_converged()

                # Safety
                fk_pose = get_robot_fk_pose(arm) if arm is not None else None
                if fk_pose is not None:
                    if fk_pose[2] < Z_FLOOR_DEPLOY:
                        print(f"\n  !!! Z_FLOOR BREACH FK z={fk_pose[2]:.1f}mm < {Z_FLOOR_DEPLOY}")
                        abort = True
                        break
                    dist = math.sqrt(fk_pose[0] ** 2 + fk_pose[1] ** 2)
                    if dist > DIST_MAX_DEPLOY:
                        print(f"\n  !!! DIST BREACH {dist:.0f}mm > {DIST_MAX_DEPLOY}")
                        abort = True
                        break

                # Send to Follower (Plan 3 gripper unlock pattern)
                if not args.dry_run and arm is not None:
                    arm.joints_angle_ctrl(
                        angles=action_clamped,
                        speed=get_safe_speed(args.speed),
                        acc=args.acc,
                    )
                    arm.gripper_angle_ctrl(
                        angle=action_clamped[5], speed=1000, acc=0,
                    )

                # CSV / display
                if csv_logger is not None:
                    state_log = get_robot_angles(arm, max_retries=2) if arm is not None and not args.dry_run else None
                    csv_logger.log(
                        step=step, chunk_idx=chunk_idx, chunk_step=ci,
                        action=action_clamped, state=state_log,
                        deltas=deltas, max_delta=max_delta,
                        convergence=is_conv,
                        inference_ms=inference_ms if ci == 0 else 0.0,
                        fk_pose=fk_pose,
                    )
                if frames_dir is not None and ci == 0:
                    cv2.imwrite(str(frames_dir / f"step_{step:04d}.png"), frame_bgr)

                marker = " [CONV]" if is_conv else ""
                print(f"  [{step:3d}/{args.max_steps}] "
                      f"Act:[{action_clamped[0]:6.1f},{action_clamped[1]:6.1f},"
                      f"{action_clamped[2]:6.1f},{action_clamped[3]:6.1f},"
                      f"{action_clamped[4]:6.1f},{action_clamped[5]:6.1f}] "
                      f"d={max_delta:5.1f}{marker}")

                # Display
                if k4a is not None:
                    disp = cv2.resize(frame_bgr, (640, 360))
                    draw_overlay(
                        disp, step=step, max_steps=args.max_steps, chunk_idx=chunk_idx,
                        action=action_clamped, max_delta=max_delta,
                        elapsed=time.time() - start_time, task=args.task,
                        inference_ms=inference_ms if ci == 0 else None,
                    )
                    cv2.imshow("OpenVLA-OFT Deploy", disp)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        print("\n  [ESC 종료]")
                        abort = True
                        break

                prev_angles = action_clamped
                elapsed = time.time() - loop_t0
                if elapsed < loop_interval:
                    time.sleep(loop_interval - elapsed)

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C 종료]")

    finally:
        print("\n정리...")
        cv2.destroyAllWindows()
        if k4a is not None:
            k4a.stop()
        if arm is not None:
            if not args.dry_run:
                print(f"  Follower → HOME {INIT_POS}")
                arm.joints_angle_ctrl(angles=INIT_POS, speed=300, acc=100)
                time.sleep(2)
            arm.disconnect()
        if csv_logger is not None:
            csv_logger.close()
            print(f"  CSV 저장: {csv_logger.filepath}")

        if inference_times:
            print(f"\n추론 통계 ({len(inference_times)} chunks):")
            print(f"  평균: {np.mean(inference_times):.0f} ms")
            print(f"  min/max: {np.min(inference_times):.0f} / {np.max(inference_times):.0f} ms")
            print(f"  총 steps: {global_step}")
            print(f"  총 시간: {time.time() - start_time:.1f}s")

        print("완료.")


if __name__ == "__main__":
    main()
