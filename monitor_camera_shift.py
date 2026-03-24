"""
Camera Shift Diagnostic Monitor
B3 Deployment & Safety Specialist

PURPOSE: Before running a full deployment after camera repositioning,
run this script to determine whether the camera shift is large enough
to cause OOD failures.

QUICK TEST (< 5 minutes):
    python monitor_camera_shift.py --checkpoint outputs/smolvla_v3_sponge/checkpoints/050000/pretrained_model

WHAT IT DOES:
    1. Captures a current live frame from Azure Kinect
    2. Loads a reference frame from a training episode
    3. Computes structural similarity (SSIM) and pixel-space delta
    4. Runs model inference on both frames and compares action distributions
    5. Issues a GO / CAUTION / STOP recommendation

FAILURE MODES this catches:
    - Spatial offset (sponge appears in different pixel location)
    - Background texture shift (robot base, workspace edge moved into frame)
    - Lighting change (drastic histogram shift)
    - Action z-score deviation (model outputs OOD actions for current view)

DOES NOT replace:
    - Full deployment test
    - Human visual inspection (run --show to see frame comparison)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
import time
import logging
import numpy as np
import cv2
import torch
from pathlib import Path

logging.getLogger("BaseController").setLevel(logging.CRITICAL)

# ─── Constants from deploy_smolvla.py (READ-ONLY, not imported) ───────────────

DATASET_MEAN_POS = [0, 30, 59, 41, -2, 26]   # v3 74-ep action.mean (degrees)
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]

# Action z-score thresholds from successful deployment logs (2026-02-25)
# In 5/5 successful runs, per-joint z-scores stayed within these ranges.
# Source: manual inspection of deploy_20260225_*.csv
SAFE_ZSCORE_RANGE = {
    "base":        (-2.5,  2.5),
    "shoulder":    (-2.5,  2.5),
    "elbow":       (-2.5,  2.5),
    "wrist_pitch": (-2.5,  2.5),
    "wrist_roll":  (-2.5,  2.5),
    "gripper":     (-2.5,  2.5),
}

# SSIM threshold: similarity below this indicates problematic camera shift
# 1.0 = identical, 0.0 = completely different
# In practice: minor camera shift → SSIM ~0.85-0.95
# Catastrophic shift → SSIM < 0.70
SSIM_CAUTION_THRESHOLD = 0.80
SSIM_STOP_THRESHOLD    = 0.65

# Pixel-space mean absolute difference (0-255 scale)
# Same scene, slightly different angle: MAE ~15-25
# Object moved or background changed: MAE > 40
MAE_CAUTION_THRESHOLD = 30.0
MAE_STOP_THRESHOLD    = 50.0


# ─── Image similarity metrics ─────────────────────────────────────────────────

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index between two BGR images.
    Resize to same shape if needed.
    Returns scalar in [-1, 1], higher is more similar.
    """
    # Convert to grayscale
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def compute_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean absolute pixel error after resize-to-same."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return float(np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32))))


def compute_histogram_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Color histogram correlation (per-channel, averaged).
    1.0 = identical distribution, 0.0 = uncorrelated.
    Catches lighting changes even when SSIM is acceptable.
    """
    scores = []
    for ch in range(3):
        h1 = cv2.calcHist([img1], [ch], None, [64], [0, 256])
        h2 = cv2.calcHist([img2], [ch], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        scores.append(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    return float(np.mean(scores))


# ─── Reference frame loading ──────────────────────────────────────────────────

def load_reference_frame(lerobot_dataset_path: str) -> np.ndarray | None:
    """
    Load a representative training frame from the LeRobot v3 dataset.
    Uses the first episode's middle frame as reference.
    Returns BGR numpy array or None if not found.
    """
    dataset_path = Path(lerobot_dataset_path)

    # Try video files (LeRobot v3 stores frames as MP4)
    # Actual structure: videos/observation.images.top/chunk-*/file-*.mp4
    video_dirs = list(dataset_path.glob("videos/observation.images.top/chunk-*/*.mp4"))
    if not video_dirs:
        # Fallback patterns
        video_dirs = list(dataset_path.glob("videos/chunk-*/observation.images.top/*.mp4"))
    if not video_dirs:
        video_dirs = list(dataset_path.glob("**/*.mp4"))

    if not video_dirs:
        print(f"  [WARN] No video files found in {lerobot_dataset_path}")
        return None

    # Pick the first video file (episode 0 or earliest)
    video_path = sorted(video_dirs)[0]
    print(f"  Reference video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Jump to middle of episode — avoids init position and captures task execution
    target_frame = max(0, total_frames // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"  [WARN] Could not read frame {target_frame} from {video_path}")
        return None

    print(f"  Loaded frame {target_frame}/{total_frames} ({frame.shape[1]}x{frame.shape[0]})")
    return frame


# ─── Camera capture ───────────────────────────────────────────────────────────

def capture_live_frame() -> np.ndarray | None:
    """
    Capture one frame from Azure Kinect.
    Returns BGR numpy array or None on failure.
    """
    try:
        import pyk4a
        from pyk4a import Config, PyK4A

        k4a = PyK4A(Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        ))
        k4a.start()

        # Warm up: discard first 5 frames (auto-exposure settling)
        live_frame = None
        for _ in range(6):
            capture = k4a.get_capture()
            if capture.color is not None:
                live_frame = capture.color[:, :, :3].copy()  # BGRA → BGR
        k4a.stop()

        if live_frame is None:
            print("  [ERROR] Kinect returned no color frames")
        return live_frame

    except ImportError:
        print("  [ERROR] pyk4a not installed. Run: pip install pyk4a")
        return None
    except Exception as e:
        print(f"  [ERROR] Kinect capture failed: {e}")
        return None


# ─── Model inference comparison ───────────────────────────────────────────────

def run_inference_comparison(
    checkpoint_path: str,
    ref_frame_bgr: np.ndarray,
    live_frame_bgr: np.ndarray,
    robot_state: list,
    task_text: str,
    device: str,
) -> dict:
    """
    Run SmolVLA inference on both reference and live frames.
    Compare action z-scores to detect distribution shift.
    Returns dict with per-joint z-scores and deviation flags.
    """
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"  [WARN] Cannot run inference comparison: {e}")
        return {"error": str(e)}

    print(f"\n  Loading checkpoint: {checkpoint_path}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    pre_stats = load_file(
        f"{checkpoint_path}/policy_preprocessor_step_5_normalizer_processor.safetensors"
    )
    post_stats = load_file(
        f"{checkpoint_path}/policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )

    stats = {
        "action_mean": post_stats["action.mean"].to(device),
        "action_std":  post_stats["action.std"].to(device),
        "state_mean":  pre_stats["observation.state.mean"].to(device),
        "state_std":   pre_stats["observation.state.std"].to(device),
    }

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenized = tokenizer(
        [task_text],
        max_length=48,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    lang = {
        "tokens": tokenized["input_ids"].to(device),
        "mask": tokenized["attention_mask"].bool().to(device),
    }

    def build_obs(frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if (img_rgb.shape[1], img_rgb.shape[0]) != (1280, 720):
            img_rgb = cv2.resize(img_rgb, (1280, 720))
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        state_t = torch.tensor(robot_state, dtype=torch.float32).to(device)
        state_norm = (state_t - stats["state_mean"]) / (stats["state_std"] + 1e-8)
        return {
            "observation.images.top": img_t.unsqueeze(0).to(device),
            "observation.state": state_norm.unsqueeze(0),
            "observation.language.tokens": lang["tokens"],
            "observation.language.attention_mask": lang["mask"],
        }

    results = {}
    with torch.no_grad():
        for label, frame in [("reference", ref_frame_bgr), ("live", live_frame_bgr)]:
            obs = build_obs(frame)
            raw = policy.select_action(obs)
            if isinstance(raw, torch.Tensor):
                raw = raw.squeeze().cpu().float()
            else:
                raw = torch.tensor(raw, dtype=torch.float32)

            # Unnormalize to degrees
            action_deg = (raw * stats["action_std"].cpu() + stats["action_mean"].cpu()).numpy()
            # Z-score in action space
            z_scores = ((raw).numpy())  # raw IS the z-score (normalized action)

            results[label] = {
                "action_deg": action_deg,
                "z_scores": z_scores,
            }
            print(f"\n  [{label}] actions (degrees): " +
                  ", ".join(f"{n}={v:.1f}" for n, v in zip(JOINT_NAMES, action_deg)))
            print(f"  [{label}] z-scores:          " +
                  ", ".join(f"{n}={v:.2f}" for n, v in zip(JOINT_NAMES, z_scores)))

    # Compute deviation between reference and live
    if "reference" in results and "live" in results:
        z_ref  = results["reference"]["z_scores"]
        z_live = results["live"]["z_scores"]
        z_delta = np.abs(z_live - z_ref)
        results["z_delta"] = z_delta
        results["max_z_delta"] = float(z_delta.max())
        results["mean_z_delta"] = float(z_delta.mean())

        print(f"\n  z-score delta (|live - ref|): " +
              ", ".join(f"{n}={v:.2f}" for n, v in zip(JOINT_NAMES, z_delta)))
        print(f"  Max z-delta: {results['max_z_delta']:.3f}   Mean z-delta: {results['mean_z_delta']:.3f}")

        # Flag joints that exceed safe range for the live frame
        oob_joints = []
        for i, name in enumerate(JOINT_NAMES):
            lo, hi = SAFE_ZSCORE_RANGE[name]
            z = float(z_live[i])
            if z < lo or z > hi:
                oob_joints.append(f"{name}={z:.2f}")
        results["oob_joints"] = oob_joints

    return results


# ─── Recommendation engine ────────────────────────────────────────────────────

def make_recommendation(
    ssim: float,
    mae: float,
    hist_corr: float,
    inference_results: dict,
) -> tuple[str, list[str]]:
    """
    Returns (verdict, reasons).
    verdict: "GO" | "CAUTION" | "STOP"
    """
    reasons = []
    severity = 0  # 0=GO, 1=CAUTION, 2=STOP

    # ── Image similarity ──────────────────────────────────────────────────────
    if ssim < SSIM_STOP_THRESHOLD:
        reasons.append(f"SSIM={ssim:.3f} < {SSIM_STOP_THRESHOLD} (severe scene change)")
        severity = max(severity, 2)
    elif ssim < SSIM_CAUTION_THRESHOLD:
        reasons.append(f"SSIM={ssim:.3f} < {SSIM_CAUTION_THRESHOLD} (moderate scene change)")
        severity = max(severity, 1)
    else:
        reasons.append(f"SSIM={ssim:.3f} OK (scene similar)")

    if mae > MAE_STOP_THRESHOLD:
        reasons.append(f"MAE={mae:.1f} > {MAE_STOP_THRESHOLD} (large pixel difference)")
        severity = max(severity, 2)
    elif mae > MAE_CAUTION_THRESHOLD:
        reasons.append(f"MAE={mae:.1f} > {MAE_CAUTION_THRESHOLD} (moderate pixel difference)")
        severity = max(severity, 1)
    else:
        reasons.append(f"MAE={mae:.1f} OK")

    if hist_corr < 0.7:
        reasons.append(f"Histogram correlation={hist_corr:.3f} (lighting changed significantly)")
        severity = max(severity, 1)
    else:
        reasons.append(f"Histogram correlation={hist_corr:.3f} OK")

    # ── Inference comparison ──────────────────────────────────────────────────
    if "error" in inference_results:
        reasons.append(f"Inference skipped: {inference_results['error']}")
    elif "max_z_delta" in inference_results:
        max_d = inference_results["max_z_delta"]
        if max_d > 2.0:
            reasons.append(f"Max action z-delta={max_d:.2f} (model sees very different scene)")
            severity = max(severity, 2)
        elif max_d > 1.0:
            reasons.append(f"Max action z-delta={max_d:.2f} (model sees moderately different scene)")
            severity = max(severity, 1)
        else:
            reasons.append(f"Max action z-delta={max_d:.2f} OK (model output consistent)")

        oob = inference_results.get("oob_joints", [])
        if oob:
            reasons.append(f"OOB joints in live frame: {oob}")
            severity = max(severity, 1)

    verdict_map = {0: "GO", 1: "CAUTION", 2: "STOP"}
    return verdict_map[severity], reasons


# ─── Visualization ────────────────────────────────────────────────────────────

def show_side_by_side(ref_frame: np.ndarray, live_frame: np.ndarray, title: str = "Shift Check"):
    """Display reference and live frames side by side. Press any key to close."""
    h = max(ref_frame.shape[0], live_frame.shape[0])
    r = cv2.resize(ref_frame,  (int(ref_frame.shape[1]  * h / ref_frame.shape[0]),  h))
    l = cv2.resize(live_frame, (int(live_frame.shape[1] * h / live_frame.shape[0]), h))
    combined = np.hstack([r, l])
    # Label each half
    cv2.putText(combined, "REFERENCE (training)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(combined, "LIVE (current camera)", (r.shape[1] + 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.imshow(title, combined)
    print("\n  [Visualization] Press any key in the window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_side_by_side(ref_frame: np.ndarray, live_frame: np.ndarray, output_path: str):
    """Save reference and live frames side by side as PNG."""
    h = max(ref_frame.shape[0], live_frame.shape[0])
    r = cv2.resize(ref_frame,  (int(ref_frame.shape[1]  * h / ref_frame.shape[0]),  h))
    l = cv2.resize(live_frame, (int(live_frame.shape[1] * h / live_frame.shape[0]), h))
    combined = np.hstack([r, l])
    cv2.putText(combined, "REFERENCE", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(combined, "LIVE", (r.shape[1] + 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.imwrite(output_path, combined)
    print(f"  Saved comparison: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Camera Shift Diagnostic (B3 Safety Monitor)")

    parser.add_argument(
        "--dataset",
        default="lerobot_dataset_v4",
        help="Path to LeRobot v3 dataset directory (for reference frame). Default: lerobot_dataset_v4"
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/smolvla_v3_sponge/checkpoints/050000/pretrained_model",
        help="Checkpoint path for inference comparison (optional but recommended)"
    )
    parser.add_argument(
        "--task",
        default="Pick up the sponge",
        help="Task language instruction"
    )
    parser.add_argument(
        "--ref-image",
        default=None,
        help="Path to a specific reference PNG/JPG instead of loading from dataset"
    )
    parser.add_argument(
        "--live-image",
        default=None,
        help="Path to a specific live PNG/JPG instead of capturing from Kinect (for offline testing)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show side-by-side comparison window"
    )
    parser.add_argument(
        "--save-comparison",
        default=None,
        metavar="PATH",
        help="Save side-by-side comparison PNG to this path"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip model inference comparison (faster, image metrics only)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Camera Shift Diagnostic  [B3 Deployment Safety]")
    print("=" * 60)

    # ── Step 1: Load reference frame ─────────────────────────────────────────
    print("\n[1/4] Loading reference frame (training distribution)...")

    if args.ref_image:
        ref_frame = cv2.imread(args.ref_image)
        if ref_frame is None:
            print(f"  [ERROR] Cannot read --ref-image: {args.ref_image}")
            sys.exit(1)
        print(f"  Loaded: {args.ref_image} ({ref_frame.shape[1]}x{ref_frame.shape[0]})")
    else:
        ref_frame = load_reference_frame(args.dataset)
        if ref_frame is None:
            print("  [ERROR] No reference frame available.")
            print("  Provide --ref-image PATH or ensure --dataset path is correct.")
            sys.exit(1)

    # ── Step 2: Capture live frame ────────────────────────────────────────────
    print("\n[2/4] Capturing live frame (current camera position)...")

    if args.live_image:
        live_frame = cv2.imread(args.live_image)
        if live_frame is None:
            print(f"  [ERROR] Cannot read --live-image: {args.live_image}")
            sys.exit(1)
        print(f"  Loaded: {args.live_image} ({live_frame.shape[1]}x{live_frame.shape[0]})")
    else:
        live_frame = capture_live_frame()
        if live_frame is None:
            print("  [ERROR] Live capture failed. Use --live-image for offline test.")
            sys.exit(1)
        print(f"  Captured: {live_frame.shape[1]}x{live_frame.shape[0]}")

    # ── Step 3: Image similarity metrics ─────────────────────────────────────
    print("\n[3/4] Computing image similarity metrics...")

    t0 = time.time()
    ssim     = compute_ssim(ref_frame, live_frame)
    mae      = compute_mae(ref_frame, live_frame)
    hist_corr = compute_histogram_correlation(ref_frame, live_frame)
    t1 = time.time()

    print(f"  SSIM:                  {ssim:.4f}  (1.0=identical, >{SSIM_CAUTION_THRESHOLD}=OK)")
    print(f"  Pixel MAE:             {mae:.2f}   (<{MAE_CAUTION_THRESHOLD}=OK, >{MAE_STOP_THRESHOLD}=STOP)")
    print(f"  Histogram correlation: {hist_corr:.4f}  (>0.7=OK)")
    print(f"  Computed in {(t1 - t0)*1000:.0f}ms")

    # ── Step 4: Inference comparison ─────────────────────────────────────────
    inference_results = {}
    if not args.skip_inference:
        print("\n[4/4] Running model inference comparison...")
        print(f"  Device: {args.device}")
        print(f"  Task: '{args.task}'")
        inference_results = run_inference_comparison(
            checkpoint_path=args.checkpoint,
            ref_frame_bgr=ref_frame,
            live_frame_bgr=live_frame,
            robot_state=DATASET_MEAN_POS,
            task_text=args.task,
            device=args.device,
        )
    else:
        print("\n[4/4] Inference comparison skipped (--skip-inference)")

    # ── Recommendation ────────────────────────────────────────────────────────
    verdict, reasons = make_recommendation(ssim, mae, hist_corr, inference_results)

    print("\n" + "=" * 60)
    print(f"VERDICT: {verdict}")
    print("=" * 60)
    for r in reasons:
        print(f"  {'[!]' if '[!]' not in r else ''} {r}")

    print()
    if verdict == "GO":
        print("  Camera shift is within acceptable bounds.")
        print("  Proceed with deployment. Monitor first 50 steps closely.")
        print("  Recommended: run 1 trial manually before full evaluation.")
    elif verdict == "CAUTION":
        print("  Camera shift detected. Deployment is risky but potentially OK.")
        print("  Actions:")
        print("    1. Visually confirm sponge is in expected workspace position")
        print("    2. Place sponge in the EXACT same position as training (use tape marks)")
        print("    3. Run a single trial with --max-steps 50 (first chunk only)")
        print("    4. If elbow drifts upward or wrist_roll exceeds 2σ, abort immediately")
        print("    5. Consider recapturing 10-20 episodes from new camera position")
    elif verdict == "STOP":
        print("  Camera shift is large enough to cause OOD failures.")
        print("  DO NOT deploy without one of the following:")
        print("    A. Restore camera to exact original position (use reference image as guide)")
        print("    B. Recollect data from new camera position (minimum 30-40 new episodes)")
        print("  Rationale: SmolVLA is highly sensitive to observation distribution shift.")
        print("  The 74-ep model has narrow visual coverage; large viewpoint changes")
        print("  are 4σ+ OOD and will reproduce the Wrist_R runaway failure mode.")

    # ── Visualization ─────────────────────────────────────────────────────────
    if args.save_comparison:
        save_side_by_side(ref_frame, live_frame, args.save_comparison)

    if args.show:
        show_side_by_side(ref_frame, live_frame)

    # ── Summary line (machine-parseable) ─────────────────────────────────────
    print(f"\nSUMMARY ssim={ssim:.4f} mae={mae:.2f} hist={hist_corr:.4f} verdict={verdict}")
    return 0 if verdict in ("GO", "CAUTION") else 1


if __name__ == "__main__":
    sys.exit(main())
