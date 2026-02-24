"""
SmolVLA Batch Size VRAM Test
RTX 4090 Laptop (15.6 GB VRAM)

Tests forward + backward pass at batch sizes: 8, 16, 32, 64
Reports peak VRAM usage and whether each batch size fits.

Architecture reference:
  - Model: SmolVLA (450M total, ~100M trainable — Action Expert only)
  - Image: (B, 3, H, W) -> resized to 512x512 with padding inside model
  - State:  (B, 1, 32) — max_state_dim=32, padded from 6 joints
  - Action: (B, 50, 32) — chunk_size=50, max_action_dim=32
  - Language tokens: (B, 48) — tokenizer_max_length=48
  - 1 camera, 1 obs step (n_obs_steps=1)

Usage:
    conda run -n roarm python3 train_batch_size_test.py
"""

import os
import gc
import sys

# Must be set BEFORE importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

# Add lerobot source to path
sys.path.insert(0, "/home/cgxr/Documents/Robotics/RoArm_Project/lerobot/src")

DEVICE = "cuda"
BATCH_SIZES_TO_TEST = [8, 16, 32, 64]

# Input shape constants (from SmolVLAConfig and lerobot training pipeline)
IMG_H, IMG_W = 480, 640          # Raw camera resolution before model's internal resize to 512x512
MAX_STATE_DIM = 32               # max_state_dim (6 joints padded to 32)
MAX_ACTION_DIM = 32              # max_action_dim (6 joints padded to 32)
CHUNK_SIZE = 50                  # chunk_size / n_action_steps
TOKEN_MAX_LEN = 48               # tokenizer_max_length
N_OBS_STEPS = 1                  # n_obs_steps


def clear_cuda():
    """Clear all CUDA memory caches between tests."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def get_vram_stats():
    """Return current and peak VRAM usage in GB."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    return allocated, reserved, peak


def build_dummy_batch(batch_size: int, device: str) -> dict:
    """
    Build a dummy batch matching the SmolVLA training input format.

    Based on modeling_smolvla.py forward():
      - Images: list of tensors (B, 3, H, W) in range [0, 1]
      - OBS_STATE: (B, n_obs_steps, max_state_dim)
      - ACTION: (B, chunk_size, max_action_dim)
      - OBS_LANGUAGE_TOKENS: (B, token_max_len) int64
      - OBS_LANGUAGE_ATTENTION_MASK: (B, token_max_len) int64
    """
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    batch = {
        # Camera image: raw resolution, model will resize internally to 512x512
        "observation.images.top": torch.rand(
            batch_size, 3, IMG_H, IMG_W, dtype=torch.float32, device=device
        ),
        # Robot state: 6 joints normalized, padded to max_state_dim=32
        OBS_STATE: torch.randn(
            batch_size, N_OBS_STEPS, MAX_STATE_DIM, dtype=torch.float32, device=device
        ),
        # Target actions: chunk of 50 steps, 6 joints padded to max_action_dim=32
        ACTION: torch.randn(
            batch_size, CHUNK_SIZE, MAX_ACTION_DIM, dtype=torch.float32, device=device
        ),
        # Language tokens (tokenized task description, e.g. "Pick up the white box\n")
        OBS_LANGUAGE_TOKENS: torch.randint(
            0, 32000, (batch_size, TOKEN_MAX_LEN), dtype=torch.int64, device=device
        ),
        # Attention mask: all True (no padding in this dummy batch).
        # IMPORTANT: SmolVLA's make_att_2d_masks requires BOOL dtype,
        # not int64. The real pipeline converts via TokenizerProcessorStep.
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(
            batch_size, TOKEN_MAX_LEN, dtype=torch.bool, device=device
        ),
    }
    return batch


def test_batch_size(policy, batch_size: int, total_vram_gb: float) -> dict:
    """
    Run a single forward + backward pass at the given batch_size.
    Returns a result dict with VRAM stats and success/failure.
    """
    print(f"\n{'='*60}")
    print(f"Testing batch_size = {batch_size}")
    print(f"{'='*60}")

    clear_cuda()

    try:
        # Build dummy batch
        batch = build_dummy_batch(batch_size, DEVICE)

        vram_after_batch, _, _ = get_vram_stats()
        print(f"  VRAM after building batch: {vram_after_batch:.2f} GB")

        # Forward pass (training mode computes flow matching loss)
        policy.train()
        loss, loss_dict = policy.forward(batch)

        vram_after_fwd, _, _ = get_vram_stats()
        print(f"  VRAM after forward pass:   {vram_after_fwd:.2f} GB")
        print(f"  Loss value: {loss.item():.4f}")

        # Backward pass (computes gradients for Action Expert only)
        loss.backward()

        vram_after_bwd, vram_reserved, peak_vram = get_vram_stats()
        print(f"  VRAM after backward pass:  {vram_after_bwd:.2f} GB")
        print(f"  VRAM reserved (PyTorch):   {vram_reserved:.2f} GB")
        print(f"  Peak VRAM allocated:       {peak_vram:.2f} GB")
        print(f"  Total GPU VRAM:            {total_vram_gb:.2f} GB")
        print(f"  Utilization:               {peak_vram/total_vram_gb*100:.1f}%")
        print(f"  Headroom remaining:        {total_vram_gb - peak_vram:.2f} GB")

        # Clear gradients before next test
        policy.zero_grad()

        result = {
            "batch_size": batch_size,
            "success": True,
            "peak_vram_gb": peak_vram,
            "vram_reserved_gb": vram_reserved,
            "utilization_pct": peak_vram / total_vram_gb * 100,
            "headroom_gb": total_vram_gb - peak_vram,
            "loss": loss.item(),
            "error": None,
        }

    except torch.cuda.OutOfMemoryError as e:
        _, _, peak_vram = get_vram_stats()
        print(f"  !! OUT OF MEMORY at batch_size={batch_size} !!")
        print(f"  Peak VRAM before OOM: {peak_vram:.2f} GB")
        result = {
            "batch_size": batch_size,
            "success": False,
            "peak_vram_gb": peak_vram,
            "vram_reserved_gb": None,
            "utilization_pct": None,
            "headroom_gb": None,
            "loss": None,
            "error": str(e)[:200],
        }

    except Exception as e:
        print(f"  !! Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        result = {
            "batch_size": batch_size,
            "success": False,
            "peak_vram_gb": None,
            "vram_reserved_gb": None,
            "utilization_pct": None,
            "headroom_gb": None,
            "loss": None,
            "error": f"{type(e).__name__}: {str(e)[:300]}",
        }

    # Always clear after each test
    clear_cuda()
    return result


def main():
    print("=" * 70)
    print("SmolVLA Batch Size VRAM Test")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU:     {gpu_name}")
    print(f"VRAM:    {total_vram_gb:.2f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")

    # Load SmolVLA model
    print("\nLoading SmolVLA from lerobot/smolvla_base ...")
    print("(This takes ~30 seconds on first load, uses HuggingFace cache)")

    # Suppress noisy output during load
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import FeatureType, PolicyFeature

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

    # Set input_features so the policy knows which image keys to look for
    # (normally set by the training pipeline via dataset metadata)
    policy.config.input_features = {
        "observation.images.top": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, IMG_H, IMG_W),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(MAX_STATE_DIM,),
        ),
    }
    policy.config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(MAX_ACTION_DIM,),
        ),
    }

    policy = policy.to(DEVICE)

    # Model parameter report
    total_params = sum(p.numel() for p in policy.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad) / 1e6
    frozen_params = total_params - trainable_params
    print(f"\nModel loaded successfully:")
    print(f"  Total parameters:     {total_params:.1f}M")
    print(f"  Trainable (AE only):  {trainable_params:.1f}M")
    print(f"  Frozen (VLM):         {frozen_params:.1f}M")

    vram_model, _, _ = get_vram_stats()
    print(f"  VRAM after model load: {vram_model:.2f} GB")

    print(f"\nTest input shapes:")
    print(f"  Image:   (B, 3, {IMG_H}, {IMG_W}) -> internal resize to 512x512")
    print(f"  State:   (B, {N_OBS_STEPS}, {MAX_STATE_DIM})")
    print(f"  Action:  (B, {CHUNK_SIZE}, {MAX_ACTION_DIM})")
    print(f"  Tokens:  (B, {TOKEN_MAX_LEN})")
    print(f"  Testing: {BATCH_SIZES_TO_TEST}")

    # Run tests at each batch size
    results = []
    for bs in BATCH_SIZES_TO_TEST:
        result = test_batch_size(policy, bs, total_vram_gb)
        results.append(result)
        # Even if OOM occurs, continue testing other batch sizes
        # (expandable_segments:True helps recovery)

    # Print summary table
    print("\n")
    print("=" * 75)
    print("SUMMARY: SmolVLA VRAM Usage per Batch Size")
    print("=" * 75)
    print(f"GPU: {gpu_name}  |  Total VRAM: {total_vram_gb:.2f} GB")
    print()
    header = f"{'BS':>4} | {'Status':>7} | {'Peak VRAM':>10} | {'Util%':>7} | {'Headroom':>9} | {'Loss':>8}"
    print(header)
    print("-" * 55)

    for r in results:
        if r["success"]:
            status = "OK"
            peak_str = f"{r['peak_vram_gb']:.2f} GB"
            util_str = f"{r['utilization_pct']:.1f}%"
            headroom_str = f"{r['headroom_gb']:.2f} GB"
            loss_str = f"{r['loss']:.4f}"
        else:
            status = "OOM"
            peak_str = f"{r['peak_vram_gb']:.2f} GB" if r["peak_vram_gb"] else "N/A"
            util_str = "FAIL"
            headroom_str = "N/A"
            loss_str = "N/A"

        print(f"{r['batch_size']:>4} | {status:>7} | {peak_str:>10} | {util_str:>7} | {headroom_str:>9} | {loss_str:>8}")

    # Recommendations
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print()
    print("RECOMMENDATIONS:")
    print("-" * 55)

    if not successful:
        print("  ERROR: All batch sizes failed. Check GPU state.")
    else:
        max_ok_bs = max(r["batch_size"] for r in successful)
        best = next(r for r in successful if r["batch_size"] == max_ok_bs)

        print(f"  Maximum batch_size fitting in VRAM: {max_ok_bs}")
        print(f"  Peak VRAM at bs={max_ok_bs}: {best['peak_vram_gb']:.2f} GB "
              f"({best['utilization_pct']:.1f}% of {total_vram_gb:.2f} GB)")

        # VRAM scaling analysis
        if len(successful) >= 2:
            r1, r2 = successful[0], successful[1]
            delta_vram = r2["peak_vram_gb"] - r1["peak_vram_gb"]
            delta_bs = r2["batch_size"] - r1["batch_size"]
            per_sample_mb = (delta_vram / delta_bs) * 1000
            base_vram = r1["peak_vram_gb"] - (per_sample_mb/1000) * r1["batch_size"]
            print()
            print(f"  VRAM scaling analysis:")
            print(f"    Base VRAM (model weights + framework): ~{base_vram:.2f} GB")
            print(f"    Per-sample activation VRAM: ~{per_sample_mb:.0f} MB/sample")
            print(f"    (This is for training with backward pass)")

        # Gradient accumulation advice
        print()
        if max_ok_bs >= 64:
            print(f"  batch_size=64 fits! You can use the official recommended config:")
            print(f"    lerobot-train --batch_size=64 --steps=200000")
        else:
            accum_factor = 64 // max_ok_bs
            effective_steps_multiplier = 64 // max_ok_bs
            print(f"  Official recommendation is batch_size=64.")
            print(f"  Your max is batch_size={max_ok_bs}.")
            print()
            print(f"  GRADIENT ACCUMULATION STATUS:")
            print(f"    lerobot-train does NOT support --gradient_accumulation_steps")
            print(f"    (Searched entire lerobot source: zero matches for 'gradient_accumulation')")
            print()
            print(f"  RECOMMENDED WORKAROUNDS (choose one):")
            print(f"    Option A — Use batch_size={max_ok_bs} with proportionally more steps:")
            print(f"      Effective bs ratio: {max_ok_bs}/{64} = {max_ok_bs/64:.3f}")
            print(f"      If official uses 200K steps at bs=64, use:")
            print(f"      batch_size={max_ok_bs}, steps={200_000 * accum_factor:,}")
            print(f"      (Same total samples seen = {200_000 * 64:,} samples)")
            print()
            print(f"    Option B — Use Accelerate multi-GPU (if you have multiple GPUs)")
            print(f"      Effective batch = batch_size x num_GPUs")
            print()
            print(f"    Option C — Use batch_size={max_ok_bs} as-is for our 100-episode dataset")
            print(f"      100 episodes x ~145 frames = ~14,500 frames")
            print(f"      At bs={max_ok_bs}: epoch = {14500//max_ok_bs} steps")
            print(f"      100K steps = {100_000//(14500//max_ok_bs):.0f} epochs (good for fine-tuning)")

    if failed:
        print()
        print(f"  Failed batch sizes: {[r['batch_size'] for r in failed]}")
        for r in failed:
            if r["error"]:
                print(f"    bs={r['batch_size']}: {r['error'][:100]}")

    print()
    print("=" * 75)
    print("NOTE: lerobot-train gradient_accumulation support:")
    print("  Searched lerobot/src/lerobot/ for 'gradient_accumulation' -> 0 matches")
    print("  The TrainPipelineConfig dataclass has no such field.")
    print("  Accelerate is used internally but no accumulation steps exposed via CLI.")
    print("=" * 75)


if __name__ == "__main__":
    main()
