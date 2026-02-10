"""
Extended Training Configuration for SmolVLA: 50K-100K steps

Resume from 20K checkpoint with optimized hyperparameters for
precise joint control learning, especially elbow and gripper.

Key improvements over initial 20K training:
1. Extended training: 50K or 100K steps (vs 20K)
2. Lower learning rate in later stages (cosine decay continues)
3. More frequent checkpoints for analysis
4. Gradient clipping maintained at 10.0

Critical findings from 20K checkpoint analysis:
- Model outputs conservative z-scores (±1.5 range max)
- Elbow needs z=-3.04 for target angle -64°
- Gripper needs wider range for open/close
- Current loss: 0.009, but action diversity still limited
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def train_50k(output_suffix: str = "50k"):
    """
    Resume training from 20K checkpoint for 50K total steps.

    Args:
        output_suffix: Suffix for output directory (e.g., "50k", "100k")
    """
    checkpoint_20k = Path("E:/RoArm_Project/outputs/smolvla_official/checkpoints/020000/pretrained_model/train_config.json")

    if not checkpoint_20k.exists():
        raise FileNotFoundError(
            f"20K checkpoint not found at {checkpoint_20k}\n"
            "Run run_official_train.py first to complete 20K training."
        )

    # Calculate additional steps
    if output_suffix == "50k":
        total_steps = 50000
    elif output_suffix == "100k":
        total_steps = 100000
    else:
        raise ValueError(f"Unknown suffix: {output_suffix}. Use '50k' or '100k'")

    additional_steps = total_steps - 20000

    print("=" * 70)
    print(f"SmolVLA Extended Training Configuration: {total_steps} steps")
    print("=" * 70)
    print(f"Resuming from: {checkpoint_20k.parent}")
    print(f"Additional steps: {additional_steps} (current: 20K → target: {total_steps//1000}K)")
    print(f"Output: outputs/smolvla_{output_suffix}")
    print()
    print("Optimizer settings (from config.json):")
    print("  lr: 0.0001 (initial), cosine decay to 2.5e-06")
    print("  warmup: 1000 steps (already completed)")
    print("  decay_steps: 30000 (20K done, 10K remaining for 50K)")
    print("  grad_clip_norm: 10.0")
    print()
    print("Checkpoint strategy:")
    print("  save_freq: 2500 (more frequent for analysis)")
    print("  Expected checkpoints: 22.5K, 25K, 27.5K, ..., 50K")
    print("=" * 70)

    # Build training command
    sys.argv = [
        "lerobot-train",
        f"--config_path={checkpoint_20k}",
        "--resume=true",
        f"--steps={total_steps}",  # Total steps (not additional)
        f"--output_dir=outputs/smolvla_{output_suffix}",
        "--save_freq=2500",  # More frequent checkpoints for analysis
        "--log_freq=100",
        "--eval_freq=-1",
        "--batch_size=8",  # Keep same as initial training
        "--num_workers=0",
    ]

    print("Launching lerobot-train with arguments:")
    for arg in sys.argv[1:]:
        print(f"  {arg}")
    print()


def train_100k():
    """Resume training from 20K checkpoint for 100K total steps."""
    train_50k(output_suffix="100k")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended SmolVLA training configuration")
    parser.add_argument(
        "--steps",
        type=str,
        choices=["50k", "100k"],
        default="50k",
        help="Total training steps (50K or 100K)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without launching training",
    )

    args = parser.parse_args()

    # Setup configuration
    if args.steps == "50k":
        train_50k()
    else:
        train_100k()

    if args.dry_run:
        print("\nDry run complete. Use --no-dry-run or remove flag to start training.")
        sys.exit(0)

    # Launch training
    from lerobot.scripts.lerobot_train import main
    main()
