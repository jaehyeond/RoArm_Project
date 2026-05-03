"""
SmolVLA 실제 로봇 배포 스크립트

공식 lerobot-train CLI로 학습된 50K steps 체크포인트 사용 (v3: 74 에피소드, "Pick up the sponge")
Azure Kinect DK + RoArm M3 Pro 실시간 추론

사용법:
    # 실제 로봇 실행
    python deploy_smolvla.py

    # 포트 지정 (default = Follower /dev/ttyUSB1, deploy 대상)
    python deploy_smolvla.py --port /dev/ttyUSB1

    # 태스크 변경
    python deploy_smolvla.py --task "Pick up the sponge"

    # 최대 스텝 수 변경
    python deploy_smolvla.py --max-steps 200

    # Dry-run (로봇 전송 없이 추론만)
    python deploy_smolvla.py --dry-run

    # Action scaling (z-score amplification)
    python deploy_smolvla.py --action-scale 1.5

    # Convergence detection
    python deploy_smolvla.py --convergence-threshold 0.5 --convergence-window 10 --convergence-action warn

    # CSV logging
    python deploy_smolvla.py --log-csv logs/deployment.csv

    # Custom checkpoint
    python deploy_smolvla.py --checkpoint outputs/smolvla_v3_sponge/checkpoints/010000/pretrained_model

    # EMA smoothing (reduce jitter from per-step re-inference)
    python deploy_smolvla.py --n-action-steps 5 --ema-alpha 0.4

    # Smoothing reference:
    #   alpha=1.0  → no smoothing (raw model output each step)
    #   alpha=0.4  → moderate smoothing (recommended with n_action_steps=1)
    #   alpha=0.2  → heavy smoothing (very stable but slow to respond)
    # Note: with n_action_steps >= 5, within-chunk motion is already smooth;
    #       EMA still helps at chunk boundaries (re-inference jitter).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import time
import cv2
import math
import numpy as np
import torch
import logging
import csv
from datetime import datetime
from pathlib import Path
from safetensors.torch import load_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# SDK 로그 억제
logging.getLogger('BaseController').setLevel(logging.CRITICAL)

# RoArm M3 관절 범위 (안전 제한)
JOINT_LIMITS = [
    (-180, 180),   # 0: Base rotation
    (-110, 110),   # 1: Shoulder
    (-70,  190),   # 2: Elbow (비대칭)
    (-110, 110),   # 3: Wrist pitch
    (-180, 180),   # 4: Wrist roll
    (-10,  100),   # 5: Gripper
]

# Workspace 안전 제한 (2026-03-26 실측)
# FK z: 실측 책상 z ≈ -95~-121mm, 스펀지 상면 ≈ -90mm
# 잡기 시 그리퍼는 z ≈ -100~-115mm까지 내려감 → 안전 하한은 책상 표면 아래로
Z_FLOOR_DEPLOY = -130  # mm — 책상(-120) + 10mm 여유. 이 아래 = 책상 관통
DIST_MAX_DEPLOY = 420  # mm — 이 이상은 책상 밖 위험 (실측 안전 범위)

CHECKPOINT_PATH = "outputs/smolvla_v6/checkpoints/020000/pretrained_model"

# HOME position (v6 데이터 수집 시작 위치 = 배포 시작 위치)
# v6는 100% 에피소드가 HOME에서 시작 → 배포도 HOME에서 시작해야 in-distribution
# v3 성공: init 시작. v5 실패: dataset_mean 시작 (에피소드 시작 위치 아님)
# 2026-05-03: gripper 0→5 변경 — stacking_v2 task=1 ep[0] state[0]=[0,0,90,0,0,5.0] 정확 매치.
#   v6 pick task 별도 deploy 시 0 또는 1.5로 되돌릴 것.
INIT_POS = [0, 0, 90, 0, 0, 5]

# DATASET_MEAN_POS: 학습 후 stats.json에서 업데이트 (v6 수집 전이므로 placeholder)
# 주의: dataset_mean은 궤적 중간값이지 시작 위치가 아님. 배포 시작에 사용 금지!
DATASET_MEAN_POS = [0, 0, 90, 0, 0, 0]  # v6 학습 후 실제 값으로 교체

# 관절 이름 (로깅/출력용)
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]


def clamp_joints(angles):
    """관절 각도를 안전 범위로 클램핑"""
    return [max(lo, min(hi, a)) for a, (lo, hi) in zip(angles, JOINT_LIMITS)]


# Per-joint speed limits: distal joints (wrist_pitch, wrist_roll, gripper) = 300 max
# Prevents Wrist_R runaway observed in v1 deployment
JOINT_SPEED_CAPS = [500, 500, 500, 300, 300, 300]


def get_safe_speed(base_speed):
    """per-joint speed cap 적용. 가장 제한적인 값 반환 (SDK가 per-joint speed 미지원)"""
    return min(base_speed, min(JOINT_SPEED_CAPS))


def get_robot_angles(arm, max_retries=5):
    """로봇 관절 각도 읽기 (재시도 로직)"""
    for attempt in range(max_retries):
        try:
            angles = arm.joints_angle_get()
            if angles is not None and len(angles) >= 6:
                return list(angles[:6])
        except (KeyError, TypeError, AttributeError):
            if attempt < max_retries - 1:
                time.sleep(0.05)
    return None


def get_robot_fk_z(arm, max_retries=3):
    """FK z값 읽기 (책상 충돌 감지용)"""
    for _ in range(max_retries):
        try:
            pose = arm.pose_get()
            if pose and len(pose) >= 3:
                return pose[2]
        except Exception:
            time.sleep(0.05)
    return None


def load_model(checkpoint_path, device):
    """공식 체크포인트 모델 + 정규화 통계 로드"""
    print("=" * 60)
    print("SmolVLA 모델 로딩")
    print("=" * 60)

    # 모델 로드
    print(f"\n  체크포인트: {checkpoint_path}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  파라미터: {total_params:,}")

    # 정규화 통계 로드
    pre_stats = load_file(
        f"{checkpoint_path}/policy_preprocessor_step_5_normalizer_processor.safetensors"
    )
    post_stats = load_file(
        f"{checkpoint_path}/policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )

    stats = {
        "action_mean": post_stats["action.mean"].to(device),
        "action_std": post_stats["action.std"].to(device),
        "state_mean": pre_stats["observation.state.mean"].to(device),
        "state_std": pre_stats["observation.state.std"].to(device),
    }

    print(f"  Action mean: {stats['action_mean'].cpu().numpy()}")
    print(f"  Action std:  {stats['action_std'].cpu().numpy()}")
    print(f"  State mean:  {stats['state_mean'].cpu().numpy()}")
    print(f"  State std:   {stats['state_std'].cpu().numpy()}")

    # 토크나이저
    processor = policy.model.vlm_with_expert.processor
    tokenizer = processor.tokenizer

    print("  모델 로드 완료!")
    return policy, tokenizer, stats


def tokenize_task(tokenizer, task_text, device):
    """태스크 텍스트를 토큰화 (공식 SmolVLANewLineProcessor와 동일하게 \n 추가)"""
    if not task_text.endswith("\n"):
        task_text = f"{task_text}\n"
    tokenized = tokenizer(
        [task_text],
        max_length=48,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "tokens": tokenized["input_ids"].to(device),
        "mask": tokenized["attention_mask"].bool().to(device),
    }


def build_observation(image_bgr, state_angles, lang, stats, device):
    """카메라 프레임 + 로봇 상태 → 모델 입력 구성"""
    # 이미지: BGR→RGB, HWC→CHW, [0,255]→[0,1], batch dim
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    if (w, h) != (1280, 720):
        image_rgb = cv2.resize(image_rgb, (1280, 720))
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

    # 상태: normalize (MEAN_STD)
    state_tensor = torch.tensor(state_angles, dtype=torch.float32).to(device)
    state_norm = (state_tensor - stats["state_mean"]) / (stats["state_std"] + 1e-8)

    return {
        "observation.images.top": image_tensor.unsqueeze(0).to(device),
        "observation.state": state_norm.unsqueeze(0),
        "observation.language.tokens": lang["tokens"],
        "observation.language.attention_mask": lang["mask"],
    }


def unnormalize_action(raw_action, stats, action_scale=1.0):
    """정규화된 액션 → 실제 관절 각도 (with optional scaling)

    Args:
        raw_action: normalized action (z-scores)
        stats: dict with action_mean and action_std
        action_scale: multiplier for z-scores before unnormalization (default=1.0)

    Returns:
        unnormalized action tensor
    """
    # Scale z-scores first, then unnormalize
    scaled_z = raw_action * action_scale
    return scaled_z * stats["action_std"] + stats["action_mean"]


def compute_z_scores(action, stats):
    """실제 관절 각도 → z-scores"""
    return (action - stats["action_mean"]) / (stats["action_std"] + 1e-8)


class ConvergenceDetector:
    """관절 변화량 추적 및 수렴 감지"""

    def __init__(self, threshold=0.5, window=10):
        self.threshold = threshold
        self.window = window
        self.delta_history = []

    def update(self, current_angles, prev_angles):
        """관절 변화량 계산 및 이력 업데이트"""
        deltas = [abs(c - p) for c, p in zip(current_angles, prev_angles)]
        max_delta = max(deltas)
        self.delta_history.append(max_delta)
        if len(self.delta_history) > self.window:
            self.delta_history.pop(0)
        return deltas, max_delta

    def is_converged(self):
        """최근 N스텝이 모두 threshold 미만이면 수렴"""
        if len(self.delta_history) < self.window:
            return False
        return all(d < self.threshold for d in self.delta_history)

    def get_recent_max(self):
        """최근 window 스텝의 최대 변화량"""
        return max(self.delta_history) if self.delta_history else 0.0


def add_convergence_noise(angles, noise_std=2.0):
    """수렴 탈출용 작은 랜덤 노이즈 추가 (관절 범위 고려)"""
    noisy = [a + np.random.randn() * noise_std for a in angles]
    return clamp_joints(noisy)


class CSVLogger:
    """실시간 CSV 로깅"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.writer = None

        # 헤더
        self.fieldnames = [
            "step", "timestamp",
            "base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper",
            "state_base", "state_shoulder", "state_elbow", "state_wrist_pitch", "state_wrist_roll", "state_gripper",
            "z_base", "z_shoulder", "z_elbow", "z_wrist_pitch", "z_wrist_roll", "z_gripper",
            "delta_base", "delta_shoulder", "delta_elbow", "delta_wrist_pitch", "delta_wrist_roll", "delta_gripper",
            "max_delta", "convergence_detected", "inference_ms",
            "zone", "fk_x", "fk_y", "fk_z",
        ]

    def open(self):
        """파일 열기 및 헤더 쓰기"""
        self.file = open(self.filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def log_step(self, step, angles, z_scores, deltas, max_delta, convergence, inference_ms,
                 zone="", fk_x=0, fk_y=0, fk_z=0, state=None):
        """한 스텝 로깅. state=현재 팔로워 실제 관절(선택)."""
        if self.writer is None:
            return

        if state is None:
            state = [float("nan")] * 6

        row = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **{f"{name}": angles[i] for i, name in enumerate(JOINT_NAMES)},
            **{f"state_{name}": state[i] for i, name in enumerate(JOINT_NAMES)},
            **{f"z_{name}": z_scores[i].item() if hasattr(z_scores[i], 'item') else z_scores[i]
               for i, name in enumerate(JOINT_NAMES)},
            **{f"delta_{name}": deltas[i] for i, name in enumerate(JOINT_NAMES)},
            "max_delta": max_delta,
            "convergence_detected": convergence,
            "inference_ms": inference_ms,
            "zone": zone,
            "fk_x": fk_x,
            "fk_y": fk_y,
            "fk_z": fk_z,
        }
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """파일 닫기"""
        if self.file:
            self.file.close()


def draw_overlay(frame, step, max_steps, z_scores, convergence, elapsed_time, task, inference_ms=None):
    """OpenCV 디스플레이에 정보 오버레이

    Args:
        frame: BGR image (will be modified in-place)
        step: current step number
        max_steps: total steps
        z_scores: tensor or list of z-scores (6 joints)
        convergence: True if converged
        elapsed_time: total elapsed seconds
        task: task description string
        inference_ms: optional inference time in ms
    """
    h, w = frame.shape[:2]

    # 상단: 스텝 카운터
    cv2.putText(frame, f"Step {step}/{max_steps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 추론 시간
    if inference_ms is not None:
        cv2.putText(frame, f"{inference_ms:.0f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 경과 시간
    cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 수렴 상태
    if convergence:
        cv2.putText(frame, "CONVERGED", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Z-scores (오른쪽 상단)
    z_arr = z_scores.cpu().numpy() if hasattr(z_scores, 'cpu') else np.array(z_scores)
    y_offset = 30
    for i, (name, z_val) in enumerate(zip(JOINT_NAMES, z_arr)):
        # Color coding: green (|z| < 1), yellow (1-2), red (>2)
        abs_z = abs(z_val)
        if abs_z < 1.0:
            color = (0, 255, 0)  # green
        elif abs_z < 2.0:
            color = (0, 255, 255)  # yellow
        else:
            color = (0, 0, 255)  # red

        text = f"{name[:3]}: {z_val:+.2f}"
        cv2.putText(frame, text, (w - 150, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 하단: 태스크 설명
    cv2.putText(frame, task, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="SmolVLA 실제 로봇 배포")
    parser.add_argument("--port", default="/dev/ttyUSB1", help="로봇 시리얼 포트 (Follower=USB1, deploy 대상)")
    parser.add_argument("--task", default="Pick up the sponge", help="태스크 설명")
    parser.add_argument("--max-steps", type=int, default=300, help="최대 추론 스텝 (에피소드 평균 255프레임)")
    parser.add_argument("--speed", type=int, default=500, help="로봇 모터 속도 (0-1000)")
    parser.add_argument("--acc", type=int, default=200, help="로봇 모터 가속도 (0-500)")
    parser.add_argument("--hz", type=float, default=10.0, help="제어 루프 주파수")
    parser.add_argument("--dry-run", action="store_true", help="로봇에 명령 전송 안함")
    parser.add_argument("--start-pos", default="init",
                        choices=["zero", "dataset_mean", "current", "init"],
                        help="시작 위치: init=HOME [0,0,90,0,0,0] (v6 기본값 — 수집 시작 위치와 일치), "
                             "dataset_mean=학습데이터 전체 평균 (비권장: mid-trajectory), "
                             "zero=[0]*6, current=현재위치 유지")
    parser.add_argument("--open-loop", action="store_true",
                        help="Open-loop: chunk(50 actions)를 순서대로 실행 (chunk 내에서는 새 관측 안함)")
    parser.add_argument("--n-chunks", type=int, default=4,
                        help="Open-loop 모드에서 실행할 chunk 수 (4=200steps, v6 에피소드 ~200프레임 커버)")
    parser.add_argument("--n-action-steps", type=int, default=5,
                        help="chunk에서 사용할 action 수 (1=매 스텝 새 추론, 5=권장(재추론+부드러움 균형), 50=공식 기본값)")
    parser.add_argument("--device", default="cuda", help="cuda 또는 cpu")

    # NEW: Action scaling
    parser.add_argument("--action-scale", type=float, default=1.0,
                        help="z-score 증폭 배율 (1.0=기본, >1.0=더 큰 동작, <1.0=더 작은 동작)")

    # NEW: EMA smoothing
    parser.add_argument("--ema-alpha", type=float, default=1.0,
                        help="EMA 스무딩 계수 (1.0=스무딩 없음, 0.4=권장, 0.2=강한 스무딩). "
                             "공식: smoothed = alpha*new + (1-alpha)*prev. "
                             "n-action-steps=1일 때 매 스텝 새 노이즈로 인한 떨림 억제에 효과적.")

    # NEW: Convergence detection
    parser.add_argument("--convergence-threshold", type=float, default=0.5,
                        help="수렴 감지 임계값 (관절 변화량 도 기준)")
    parser.add_argument("--convergence-window", type=int, default=10,
                        help="수렴 감지 윈도우 (연속 N스텝 동안 threshold 미만이면 수렴)")
    parser.add_argument("--convergence-action", default="warn",
                        choices=["warn", "noise", "stop"],
                        help="수렴 감지 시 행동: warn=경고만, noise=노이즈 추가, stop=중단")

    # NEW: CSV logging
    parser.add_argument("--log-csv", type=str, default=None, nargs='?', const='auto',
                        help="CSV 로그 파일 경로 (미지정시 자동 생성: logs/deploy_YYYYMMDD_HHMMSS.csv)")

    # NEW: Custom checkpoint
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="SmolVLA 체크포인트 경로")

    # NEW (2026-05-03): Kinect frame dump for offline analysis
    parser.add_argument("--save-frames-dir", type=str, default=None,
                        help="closed-loop 매 step Kinect BGR frame을 PNG로 저장할 디렉토리 (None=저장 안 함)")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 불가, CPU 전환")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # CSV 로거 초기화
    csv_logger = None
    if args.log_csv:
        if args.log_csv == 'auto':
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = log_dir / f"deploy_{timestamp}.csv"
        else:
            csv_path = Path(args.log_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)

        csv_logger = CSVLogger(csv_path)
        csv_logger.open()
        print(f"CSV 로그: {csv_path}")

    # NEW (2026-05-03): Frame dump directory
    frames_dir = None
    if args.save_frames_dir:
        frames_dir = Path(args.save_frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Frame 저장: {frames_dir}/step_NNNN.png")

    # Convergence detector 초기화
    conv_detector = ConvergenceDetector(
        threshold=args.convergence_threshold,
        window=args.convergence_window
    )

    # 1. 모델 로드
    policy, tokenizer, stats = load_model(args.checkpoint, device)
    lang = tokenize_task(tokenizer, args.task, device)

    # 2. 로봇 연결
    print(f"\n로봇 연결 ({args.port})...")
    from roarm_sdk.roarm import roarm as RoArm
    arm = RoArm(roarm_type="roarm_m3", port=args.port, baudrate=115200)
    time.sleep(0.5)

    # 연결 확인
    angles = get_robot_angles(arm)
    if angles is None:
        print("로봇 연결 실패! 서보 스캔 시도...")
        import serial
        ser = serial.Serial(args.port, 115200, timeout=2)
        ser.write(b'{"T":106}\n')
        time.sleep(1)
        ser.close()
        print("ESP32 리셋 완료. 전원 OFF→ON 후 다시 실행하세요.")
        return
    print(f"  현재 관절: {[f'{a:.1f}' for a in angles]}")

    # 3. Azure Kinect 연결
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

    # 카메라 테스트
    capture = k4a.get_capture()
    if capture.color is None:
        print("카메라 프레임 읽기 실패!")
        k4a.stop()
        arm.disconnect()
        return
    print(f"  프레임 크기: {capture.color.shape}")

    # 4. 초기 위치로 이동
    if args.start_pos == "zero":
        start_angles = [0, 0, 0, 0, 0, 0]
    elif args.start_pos == "dataset_mean":
        print("  WARNING: dataset_mean is mid-trajectory average. 배포 시작에 부적합.")
        print("           --start-pos init 사용 권장 (v6 에피소드 시작 = HOME)")
        start_angles = DATASET_MEAN_POS
    elif args.start_pos == "init":
        # v6: 모든 에피소드가 HOME [0,0,90,0,0,0]에서 시작 → in-distribution
        start_angles = INIT_POS
    else:  # "current"
        start_angles = None

    if start_angles is not None:
        print(f"\n로봇 초기 위치로 이동: {start_angles}")
        arm.joints_angle_ctrl(angles=start_angles, speed=300, acc=100)
        # 이동 완료 대기: 목표 도달까지 polling (최대 10초)
        for wait_i in range(20):
            time.sleep(0.5)
            cur = get_robot_angles(arm)
            if cur is not None:
                diffs = [abs(cur[j] - start_angles[j]) for j in range(6)]
                max_diff = max(diffs)
                if max_diff < 5.0:
                    print(f"  도달 완료! (max diff={max_diff:.1f}°, {(wait_i+1)*0.5:.1f}초)")
                    break
                if wait_i % 4 == 3:
                    print(f"  이동 중... max_diff={max_diff:.1f}° ({(wait_i+1)*0.5:.1f}초)")
        else:
            cur = get_robot_angles(arm)
            if cur is not None:
                diffs = [abs(cur[j] - start_angles[j]) for j in range(6)]
                print(f"  WARNING: 10초 후 max_diff={max(diffs):.1f}° (목표 미도달)")
                print(f"  현재: {[f'{c:.1f}' for c in cur]}")
                print(f"  목표: {start_angles}")
    else:
        print("\n현재 위치에서 시작")

    # 5. 추론 루프
    print("\n" + "=" * 60)
    print(f"  Task: \"{args.task}\"")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Control Hz: {args.hz}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  Start pos: {args.start_pos} → {start_angles}")
    print(f"  Open-loop: {args.open_loop}" + (f" ({args.n_chunks} chunks × 50 = {args.n_chunks*50} steps)" if args.open_loop else ""))
    print(f"  n_action_steps: {args.n_action_steps} ({'매 스텝 새 추론' if args.n_action_steps == 1 else f'{args.n_action_steps}스텝마다 추론'})")
    print(f"  Action scale: {args.action_scale}x")
    print(f"  EMA alpha: {args.ema_alpha} ({'no smoothing' if args.ema_alpha == 1.0 else f'smoothing ON (alpha={args.ema_alpha})'})")
    print(f"  Convergence: threshold={args.convergence_threshold}°, window={args.convergence_window}, action={args.convergence_action}")
    if csv_logger:
        print(f"  CSV logging: {csv_logger.filepath}")
    print(f"  [Ctrl+C 종료]")
    print("=" * 60)

    # n_action_steps 설정 (1이면 매 스텝 새 추론 = 진짜 closed-loop)
    policy.config.n_action_steps = args.n_action_steps
    policy.reset()
    step = 0
    loop_interval = 1.0 / args.hz
    inference_times = []
    start_time = time.time()
    prev_angles = start_angles if start_angles is not None else get_robot_angles(arm)
    convergence_detected = False

    # EMA state: None means "no prior smoothed value yet" → first step uses raw action
    ema_smoothed = None

    try:
        with torch.inference_mode():
            if args.open_loop:
                # ===== Open-Loop 모드 (Multi-Chunk) =====
                # 각 chunk(50 steps) 시작 시 새 관측 → chunk 내에서는 관측 없이 실행
                # 기본 4 chunks = 200 steps → 전체 에피소드 커버 (학습 에피소드 평균 178 프레임)
                n_chunks = args.n_chunks
                total_open_steps = n_chunks * 50
                print(f"\n  [Open-Loop] {n_chunks} chunks × 50 steps = {total_open_steps} steps 예정")

                prev_angles = start_angles if start_angles is not None else get_robot_angles(arm)
                global_step = 0
                abort = False

                for chunk_idx in range(n_chunks):
                    if abort:
                        break

                    # 각 chunk 시작 시 새 관측 촬영
                    print(f"\n  === Chunk {chunk_idx+1}/{n_chunks} ===")
                    capture = k4a.get_capture()
                    frame_bgr = np.ascontiguousarray(capture.color[:, :, :3])

                    current_angles = get_robot_angles(arm)
                    if current_angles is None:
                        print("  상태 읽기 실패!")
                        raise KeyboardInterrupt

                    t0 = time.time()
                    obs = build_observation(frame_bgr, current_angles, lang, stats, device)

                    batch = policy._prepare_batch(obs)
                    raw_chunk = policy._get_action_chunk(batch, noise=None)
                    chunk = unnormalize_action(raw_chunk, stats, action_scale=args.action_scale)
                    t1 = time.time()

                    chunk_np = chunk.cpu().numpy().squeeze()  # (50, 6+)
                    n_steps = chunk_np.shape[0]
                    inference_ms = (t1 - t0) * 1000
                    inference_times.append(inference_ms)
                    print(f"  Chunk 생성: {chunk_np.shape}, {inference_ms:.0f}ms")

                    # Chunk 미리보기 (처음과 마지막 5개만)
                    print(f"  --- Chunk {chunk_idx+1} Preview ---")
                    for i in [0, 1, 2, 3, 4, n_steps-3, n_steps-2, n_steps-1]:
                        if i < n_steps:
                            a = chunk_np[i, :6]
                            marker = "..." if i == n_steps-3 and n_steps > 8 else ""
                            if i == 5 and n_steps > 8:
                                print(f"    ... (steps 6-{n_steps-2} omitted)")
                            print(f"    [{global_step+i+1:3d}] [{a[0]:7.1f},{a[1]:7.1f},{a[2]:7.1f},"
                                  f"{a[3]:7.1f},{a[4]:7.1f},{a[5]:7.1f}]")

                    # Chunk 실행
                    for i in range(n_steps):
                        loop_start = time.time()
                        global_step += 1
                        step = global_step

                        action_clamped = clamp_joints(chunk_np[i, :6].tolist())

                        action_tensor = torch.tensor(action_clamped, dtype=torch.float32, device=device)
                        z_scores = compute_z_scores(action_tensor, stats)

                        deltas, max_delta = conv_detector.update(action_clamped, prev_angles)
                        is_converged = conv_detector.is_converged()

                        if is_converged and not convergence_detected:
                            convergence_detected = True
                            print(f"\n  [CONVERGENCE at step {step}]")
                            if args.convergence_action == "stop":
                                print("      Stopping execution")
                                abort = True
                                break

                        if not args.dry_run:
                            arm.joints_angle_ctrl(
                                angles=action_clamped,
                                speed=get_safe_speed(args.speed),
                                acc=args.acc,
                            )
                            # [PLAN3-GRIPPER-UNLOCK 2026-04-09] gripper만 속도 unlock 격리 실험
                            # 학습 수집=speed=0(무제한). 배포 cap 300=26°/s → close phase timeout.
                            # 두 번째 호출이 joints_angle_ctrl의 gripper 명령을 덮어씀.
                            # main 관절은 그대로 26°/s, gripper만 88°/s.
                            arm.gripper_angle_ctrl(
                                angle=action_clamped[5],
                                speed=1000,
                                acc=0,
                            )

                        if csv_logger:
                            follower_state = get_robot_angles(arm, max_retries=2) if not args.dry_run else None
                            csv_logger.log_step(
                                step=step,
                                angles=action_clamped,
                                z_scores=z_scores.cpu().numpy(),
                                deltas=deltas,
                                max_delta=max_delta,
                                convergence=is_converged,
                                inference_ms=inference_ms if i == 0 else 0,
                                state=follower_state,
                            )

                        convergence_marker = " [CONV]" if is_converged else ""
                        print(
                            f"  [{step:3d}/{total_open_steps}] "
                            f"Act:[{action_clamped[0]:6.1f},{action_clamped[1]:6.1f},"
                            f"{action_clamped[2]:6.1f},{action_clamped[3]:6.1f},"
                            f"{action_clamped[4]:6.1f},{action_clamped[5]:6.1f}] "
                            f"d={max_delta:5.1f}"
                            f"{convergence_marker}"
                        )
                        prev_angles = action_clamped

                        capture = k4a.get_capture()
                        if capture.color is not None:
                            display = cv2.resize(capture.color[:, :, :3], (640, 360))
                            elapsed_time = time.time() - start_time
                            draw_overlay(
                                frame=display,
                                step=step,
                                max_steps=total_open_steps,
                                z_scores=z_scores,
                                convergence=is_converged,
                                elapsed_time=elapsed_time,
                                task=f"[OPEN-LOOP {chunk_idx+1}/{n_chunks}] {args.task}",
                                inference_ms=inference_ms if i == 0 else None
                            )
                            cv2.imshow("SmolVLA Deploy", display)

                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            print("\n  [ESC 종료]")
                            abort = True
                            break

                        elapsed = time.time() - loop_start
                        if elapsed < loop_interval:
                            time.sleep(loop_interval - elapsed)

            else:
                # ===== Closed-Loop 모드 (NEW: enhanced with all features) =====
                while step < args.max_steps:
                    loop_start = time.time()
                    step += 1

                    # 카메라 프레임 캡처
                    capture = k4a.get_capture()
                    if capture.color is None:
                        print(f"  [{step:3d}] 카메라 프레임 없음, 건너뜀")
                        continue
                    frame_bgr = np.ascontiguousarray(capture.color[:, :, :3])  # BGRA → BGR

                    # NEW (2026-05-03): Save raw Kinect frame for offline analysis
                    if frames_dir is not None:
                        cv2.imwrite(str(frames_dir / f"step_{step:04d}.png"), frame_bgr)

                    # 로봇 상태 읽기
                    current_angles = get_robot_angles(arm)
                    if current_angles is None:
                        print(f"  [{step:3d}] 상태 읽기 실패, 건너뜀")
                        continue

                    # 관측 구성 + 추론
                    t0 = time.time()
                    obs = build_observation(frame_bgr, current_angles, lang, stats, device)
                    raw_action = policy.select_action(obs)
                    # NEW: Apply action scaling
                    action = unnormalize_action(raw_action, stats, action_scale=args.action_scale)
                    t1 = time.time()

                    inference_ms = (t1 - t0) * 1000
                    inference_times.append(inference_ms)

                    # 액션 추출
                    action_np = action.cpu().numpy().squeeze()[:6]

                    # NEW: EMA smoothing (reduces per-step jitter from different noise seeds)
                    # Applied BEFORE clamping so the smoothed value is still physically meaningful.
                    # On the very first step ema_smoothed is None → initialize from raw action.
                    # With n_action_steps > 1 the within-chunk motion is already smooth;
                    # EMA only activates at chunk boundaries when a new inference occurs.
                    alpha = args.ema_alpha
                    if alpha < 1.0:
                        if ema_smoothed is None:
                            ema_smoothed = action_np.copy()
                        else:
                            ema_smoothed = alpha * action_np + (1.0 - alpha) * ema_smoothed
                        action_np = ema_smoothed

                    # 안전 클램핑
                    action_clamped = clamp_joints(action_np.tolist())

                    # NEW: Compute z-scores for display
                    action_tensor = torch.tensor(action_clamped, dtype=torch.float32, device=device)
                    z_scores = compute_z_scores(action_tensor, stats)

                    # NEW: Convergence detection
                    deltas, max_delta = conv_detector.update(action_clamped, prev_angles)
                    is_converged = conv_detector.is_converged()

                    if is_converged and not convergence_detected:
                        convergence_detected = True
                        print(f"\n  ⚠️  [CONVERGENCE DETECTED at step {step}]")
                        print(f"      Max delta < {args.convergence_threshold}° for {args.convergence_window} steps")

                        if args.convergence_action == "stop":
                            print("      Stopping execution (--convergence-action stop)")
                            break
                        elif args.convergence_action == "noise":
                            print("      Adding noise to break plateau (--convergence-action noise)")
                            action_clamped = add_convergence_noise(action_clamped)
                            convergence_detected = False  # reset flag

                    # Workspace 안전 체크 (책상 충돌 + 범위 이탈 방지)
                    fk_z = get_robot_fk_z(arm)
                    if fk_z is not None and fk_z < Z_FLOOR_DEPLOY:
                        print(f"\n  !!! Z_FLOOR BREACH: FK z={fk_z:.1f}mm < {Z_FLOOR_DEPLOY}mm")
                        print(f"      긴급 정지! 로봇이 책상 아래로 내려가려 함.")
                        break
                    # FK dist 체크 (책상 밖 이탈 방지)
                    fk_pose = None
                    try:
                        fk_pose = arm.pose_get()
                    except Exception:
                        pass
                    if fk_pose and len(fk_pose) >= 2:
                        fk_dist_now = math.sqrt(fk_pose[0]**2 + fk_pose[1]**2)
                        if fk_dist_now > DIST_MAX_DEPLOY:
                            print(f"\n  !!! DIST BREACH: {fk_dist_now:.0f}mm > {DIST_MAX_DEPLOY}mm")
                            print(f"      긴급 정지! 팔이 책상 밖으로 나가려 함.")
                            break

                    # 로봇에 전송
                    if not args.dry_run:
                        arm.joints_angle_ctrl(
                            angles=action_clamped,
                            speed=get_safe_speed(args.speed),
                            acc=args.acc,
                        )
                        # [PLAN3-GRIPPER-UNLOCK 2026-04-09] gripper만 속도 unlock (설명: open-loop 경로 참조)
                        arm.gripper_angle_ctrl(
                            angle=action_clamped[5],
                            speed=1000,
                            acc=0,
                        )

                    # NEW: CSV logging
                    if csv_logger:
                        csv_logger.log_step(
                            step=step,
                            angles=action_clamped,
                            z_scores=z_scores.cpu().numpy(),
                            deltas=deltas,
                            max_delta=max_delta,
                            convergence=is_converged,
                            inference_ms=inference_ms
                        )

                    # 상태 출력 (enhanced)
                    convergence_marker = " [CONV]" if is_converged else ""
                    print(
                        f"  [{step:3d}] "
                        f"Act:[{action_clamped[0]:6.1f},{action_clamped[1]:6.1f},"
                        f"{action_clamped[2]:6.1f},{action_clamped[3]:6.1f},"
                        f"{action_clamped[4]:6.1f},{action_clamped[5]:6.1f}] "
                        f"d={max_delta:5.1f} "
                        f"{inference_ms:4.0f}ms"
                        f"{convergence_marker}"
                    )

                    # NEW: Enhanced OpenCV display
                    display = cv2.resize(frame_bgr, (640, 360))
                    elapsed_time = time.time() - start_time
                    draw_overlay(
                        frame=display,
                        step=step,
                        max_steps=args.max_steps,
                        z_scores=z_scores,
                        convergence=is_converged,
                        elapsed_time=elapsed_time,
                        task=args.task,
                        inference_ms=inference_ms
                    )
                    cv2.imshow("SmolVLA Deploy", display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        print("\n  [ESC 종료]")
                        break

                    prev_angles = action_clamped

                    # 루프 타이밍 유지
                    elapsed = time.time() - loop_start
                    if elapsed < loop_interval:
                        time.sleep(loop_interval - elapsed)

    except KeyboardInterrupt:
        print("\n\n  [Ctrl+C 종료]")

    finally:
        # 정리
        print("\n정리 중...")
        cv2.destroyAllWindows()
        k4a.stop()

        if not args.dry_run:
            home = start_angles if start_angles is not None else [0, 0, 0, 0, 0, 0]
            print(f"  로봇 초기 위치 복귀: {home}")
            arm.joints_angle_ctrl(angles=home, speed=args.speed, acc=args.acc)
            time.sleep(2)

        arm.disconnect()

        # NEW: Close CSV logger
        if csv_logger:
            csv_logger.close()
            print(f"  CSV 로그 저장 완료: {csv_logger.filepath}")

        # 통계
        if inference_times:
            print(f"\n추론 통계:")
            print(f"  총 스텝: {step}")
            print(f"  평균 추론: {np.mean(inference_times):.1f}ms")
            print(f"  최소/최대: {np.min(inference_times):.1f}/{np.max(inference_times):.1f}ms")

        # NEW: Convergence summary
        if conv_detector.delta_history:
            print(f"\n수렴 통계:")
            print(f"  최근 최대 delta: {conv_detector.get_recent_max():.2f}°")
            print(f"  수렴 감지 횟수: {'Yes' if convergence_detected else 'No'}")

        print("완료!")


if __name__ == "__main__":
    main()
