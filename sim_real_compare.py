"""
Compare Sim-rendered images vs Real Azure Kinect images using SigLIP cosine similarity.

Phase 1 Go/No-Go criterion: cosine similarity >= 0.7

Usage:
    conda activate roarm  # needs transformers + torch
    python sim_real_compare.py --sim-dir sim_renders/episode_000 --episode 0
    python sim_real_compare.py --sim-dir sim_renders --all --output results/sim_real_compare.json
"""
import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from glob import glob

import torch
import pandas as pd
from PIL import Image


VIDEO_PATH = str(Path(__file__).parent / "lerobot_dataset_v6" / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4")
PARQUET_PATH = str(Path(__file__).parent / "lerobot_dataset_v6" / "data" / "chunk-000" / "file-000.parquet")


def parse_args():
    parser = argparse.ArgumentParser(description="Sim vs Real SigLIP cosine similarity")
    parser.add_argument("--sim-dir", type=str, default="sim_renders",
                        help="Directory with sim renders (episode_NNN/frame_NNNN.png)")
    parser.add_argument("--episode", type=int, nargs="+", default=[0])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=str, default="sim_real_compare_results.json")
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224",
                        help="SigLIP model for feature extraction")
    parser.add_argument("--video", type=str, default=VIDEO_PATH)
    parser.add_argument("--parquet", type=str, default=PARQUET_PATH)
    parser.add_argument("--max-frames", type=int, default=50,
                        help="Max frames per episode to compare")
    return parser.parse_args()


def extract_video_frames(video_path, frame_indices):
    """Extract specific frames from video using ffmpeg/av."""
    try:
        import av
    except ImportError:
        print("Installing av...")
        os.system("pip install av")
        import av

    container = av.open(video_path)
    stream = container.streams.video[0]

    frames = {}
    target_set = set(frame_indices)

    for i, frame in enumerate(container.decode(stream)):
        if i in target_set:
            img = frame.to_ndarray(format="rgb24")
            frames[i] = img
        if i > max(target_set):
            break

    container.close()
    return frames


def compute_siglip_features(images, model, processor, device):
    """Compute SigLIP features for a batch of PIL images."""
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    # L2 normalize
    features = outputs / outputs.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    args = parse_args()

    # Load parquet for frame index mapping
    print(f"Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)

    # Determine episodes
    if args.all:
        # Find all episode dirs in sim-dir
        ep_dirs = sorted(glob(os.path.join(args.sim_dir, "episode_*")))
        ep_list = [int(os.path.basename(d).split("_")[1]) for d in ep_dirs]
    else:
        ep_list = args.episode

    print(f"Episodes to compare: {ep_list}")

    # Load SigLIP model
    print(f"Loading SigLIP model: {args.model}")
    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()
    print(f"SigLIP loaded on {device}")

    results = {
        "model": args.model,
        "video": args.video,
        "sim_dir": args.sim_dir,
        "episodes": {},
    }

    for ep_idx in ep_list:
        ep_dir = os.path.join(args.sim_dir, f"episode_{ep_idx:03d}")
        if not os.path.exists(ep_dir):
            print(f"  Episode {ep_idx}: sim dir not found, skipping")
            continue

        # Find sim frames
        sim_frames = sorted(glob(os.path.join(ep_dir, "frame_*.png")))
        if not sim_frames:
            print(f"  Episode {ep_idx}: no sim frames, skipping")
            continue

        # Parse frame indices from filenames
        sim_frame_indices = []
        for f in sim_frames:
            fname = os.path.basename(f)
            idx = int(fname.replace("frame_", "").replace(".png", ""))
            sim_frame_indices.append(idx)

        # Map to global video frame indices
        ep_df = df[df["episode_index"] == ep_idx]
        ep_start_global = ep_df["index"].iloc[0]

        # Limit number of frames
        if len(sim_frame_indices) > args.max_frames:
            step = len(sim_frame_indices) // args.max_frames
            sim_frame_indices = sim_frame_indices[::step][:args.max_frames]
            sim_frames = [os.path.join(ep_dir, f"frame_{i:04d}.png") for i in sim_frame_indices]

        # Global video frame indices
        video_frame_indices = [ep_start_global + i for i in sim_frame_indices]

        print(f"\n  Episode {ep_idx}: {len(sim_frame_indices)} frames to compare")
        print(f"    Local frames: {sim_frame_indices[:5]}...")
        print(f"    Global video frames: {video_frame_indices[:5]}...")

        # Extract real frames from video
        print(f"    Extracting real frames from video...")
        real_frames = extract_video_frames(args.video, video_frame_indices)
        print(f"    Got {len(real_frames)} real frames")

        # Compute similarities
        similarities = []
        for local_idx, global_idx in zip(sim_frame_indices, video_frame_indices):
            sim_path = os.path.join(ep_dir, f"frame_{local_idx:04d}.png")
            if not os.path.exists(sim_path):
                continue
            if global_idx not in real_frames:
                continue

            # Load images
            sim_img = Image.open(sim_path).convert("RGB")
            real_img = Image.fromarray(real_frames[global_idx])

            # Compute features
            sim_feat = compute_siglip_features([sim_img], model, processor, device)
            real_feat = compute_siglip_features([real_img], model, processor, device)

            cos_sim = cosine_similarity(sim_feat[0], real_feat[0])
            similarities.append({
                "frame": local_idx,
                "cosine_similarity": cos_sim,
            })

            if len(similarities) <= 3 or len(similarities) % 10 == 0:
                print(f"    Frame {local_idx}: cosine={cos_sim:.4f}")

        if similarities:
            cos_values = [s["cosine_similarity"] for s in similarities]
            ep_result = {
                "num_compared": len(similarities),
                "mean_cosine": float(np.mean(cos_values)),
                "std_cosine": float(np.std(cos_values)),
                "min_cosine": float(np.min(cos_values)),
                "max_cosine": float(np.max(cos_values)),
                "go_nogo": "GO" if np.mean(cos_values) >= 0.7 else "NO-GO",
                "per_frame": similarities,
            }
            results["episodes"][str(ep_idx)] = ep_result
            print(f"    Episode {ep_idx} summary: mean={ep_result['mean_cosine']:.4f}, "
                  f"min={ep_result['min_cosine']:.4f}, max={ep_result['max_cosine']:.4f} "
                  f"→ {ep_result['go_nogo']}")
        else:
            print(f"    Episode {ep_idx}: no frames compared")

    # Overall summary
    all_cos = []
    for ep_data in results["episodes"].values():
        all_cos.extend([s["cosine_similarity"] for s in ep_data["per_frame"]])

    if all_cos:
        results["overall"] = {
            "mean_cosine": float(np.mean(all_cos)),
            "std_cosine": float(np.std(all_cos)),
            "min_cosine": float(np.min(all_cos)),
            "max_cosine": float(np.max(all_cos)),
            "num_frames": len(all_cos),
            "go_nogo": "GO" if np.mean(all_cos) >= 0.7 else "NO-GO",
        }
        print(f"\n=== OVERALL: mean={results['overall']['mean_cosine']:.4f}, "
              f"go/no-go: {results['overall']['go_nogo']} ===")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
