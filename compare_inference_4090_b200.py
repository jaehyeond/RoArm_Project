"""4090 vs B200 inference action diff (controlled noise + same input).

Critical design:
- Both models loaded on GPU simultaneously
- For each sample: same seed before each select_action call
- Flow matching denoising 10 steps -> deterministic given seed + weights
- Diff measures impact of cuDNN/cuBLAS GPU noise (~8% rel L2 in Action Expert MLP)
  on actual action prediction
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset


CKPT_4090_SRC = "outputs/smolvla_v6/checkpoints/last/pretrained_model"
CKPT_B200_SRC = "outputs/smolvla_v6_b200/checkpoints/last/pretrained_model"
PATCH_DIR = ".inference_compare_patched"
DATASET_ROOT = "lerobot_dataset_v6"
TASK_TEXT = "Pick up the sponge"
SEED = 42
TEST_INDICES = [0, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 6900]

OUT_PATH = Path("claudedocs/inference_diff_4090_vs_b200.json")
OUT_PATH.parent.mkdir(exist_ok=True)


def set_seed(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    np.random.seed(s)


def make_patched_ckpt(src, dst):
    """symlink everything from src to dst, but patch config.json to drop
    compile_model/compile_mode keys (LOCAL lerobot SmolVLAConfig doesn't know them)."""
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        src_path = os.path.abspath(os.path.join(src, fn))
        dst_path = os.path.join(dst, fn)
        if fn == "config.json":
            with open(src_path) as f:
                c = json.load(f)
            removed = [k for k in ("compile_model", "compile_mode") if c.pop(k, None) is not None]
            with open(dst_path, "w") as f:
                json.dump(c, f, indent=2)
            if removed:
                print(f"  patched {dst}: removed {removed}")
        else:
            if os.path.lexists(dst_path):
                os.remove(dst_path)
            os.symlink(src_path, dst_path)
    return dst


def load_policy(ckpt_path, device):
    policy = SmolVLAPolicy.from_pretrained(ckpt_path)
    policy.to(device)
    policy.eval()
    return policy


def load_norm(ckpt_path, device):
    pre = load_file(f"{ckpt_path}/policy_preprocessor_step_5_normalizer_processor.safetensors")
    post = load_file(f"{ckpt_path}/policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    return {
        "state_mean": pre["observation.state.mean"].to(device),
        "state_std": pre["observation.state.std"].to(device),
        "action_mean": post["action.mean"].to(device),
        "action_std": post["action.std"].to(device),
    }


def build_batch(sample, device, state_mean, state_std, lang_tokens, lang_mask):
    batch = {}
    skip = {"action", "task", "episode_index", "frame_index", "timestamp",
            "index", "task_index", "next.done", "next.reward"}
    for key, val in sample.items():
        if key in skip:
            continue
        if isinstance(val, torch.Tensor):
            batch[key] = val.unsqueeze(0).to(device)
    if "observation.state" in batch:
        batch["observation.state"] = (batch["observation.state"] - state_mean) / (state_std + 1e-8)
    batch["observation.language.tokens"] = lang_tokens
    batch["observation.language.attention_mask"] = lang_mask
    return batch


def predict(policy, batch, action_mean, action_std, seed):
    policy.reset()
    set_seed(seed)  # set AFTER reset so flow-matching noise is deterministic
    with torch.inference_mode():
        raw = policy.select_action(batch)
    action = raw * action_std + action_mean
    return action.cpu().numpy().squeeze(), raw.cpu().numpy().squeeze()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print("\nLoading dataset...")
    dataset = LeRobotDataset(repo_id="roarm_m3_pick", root=Path(DATASET_ROOT))
    print(f"  frames={len(dataset)}, episodes={dataset.num_episodes}")

    print("\nPatching configs (drop compile_model/compile_mode for LOCAL lerobot)...")
    ckpt_4090 = make_patched_ckpt(CKPT_4090_SRC, f"{PATCH_DIR}/4090")
    ckpt_b200 = make_patched_ckpt(CKPT_B200_SRC, f"{PATCH_DIR}/b200")

    print(f"\nLoading 4090 model: {ckpt_4090}")
    t0 = time.time()
    pol_4090 = load_policy(ckpt_4090, device)
    norm_4090 = load_norm(ckpt_4090, device)
    print(f"  loaded in {time.time()-t0:.1f}s, VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"\nLoading B200 model: {ckpt_b200}")
    t0 = time.time()
    pol_b200 = load_policy(ckpt_b200, device)
    norm_b200 = load_norm(ckpt_b200, device)
    print(f"  loaded in {time.time()-t0:.1f}s, VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Tokenize task once (both models share same tokenizer, identical pretrained_path)
    processor = pol_4090.model.vlm_with_expert.processor
    tokenizer = processor.tokenizer
    tok = tokenizer([TASK_TEXT], max_length=48, truncation=True,
                    padding="max_length", return_tensors="pt")
    lang_tokens = tok["input_ids"].to(device)
    lang_mask = tok["attention_mask"].bool().to(device)

    indices = [i for i in TEST_INDICES if i < len(dataset)]
    print(f"\nTest indices: {indices}")

    rows = []
    for idx in indices:
        sample = dataset[idx]
        gt_action = sample["action"].numpy() if "action" in sample else None

        # Build batch using 4090 norm stats (identical to B200 — verified bit-exact earlier)
        batch_4090 = build_batch(sample, device, norm_4090["state_mean"], norm_4090["state_std"],
                                  lang_tokens, lang_mask)
        batch_b200 = build_batch(sample, device, norm_b200["state_mean"], norm_b200["state_std"],
                                  lang_tokens, lang_mask)

        a4, r4 = predict(pol_4090, batch_4090, norm_4090["action_mean"], norm_4090["action_std"], SEED)
        ab, rb = predict(pol_b200, batch_b200, norm_b200["action_mean"], norm_b200["action_std"], SEED)

        diff = a4[:6] - ab[:6]
        diff_l2 = float(np.linalg.norm(diff))
        diff_max = float(np.max(np.abs(diff)))
        per_joint = [float(d) for d in diff]

        gt_l2_4090 = float(np.linalg.norm(a4[:6] - gt_action[:6])) if gt_action is not None else None
        gt_l2_b200 = float(np.linalg.norm(ab[:6] - gt_action[:6])) if gt_action is not None else None

        row = {
            "idx": idx,
            "action_4090": [float(v) for v in a4[:6]],
            "action_b200": [float(v) for v in ab[:6]],
            "gt_action": [float(v) for v in gt_action[:6]] if gt_action is not None else None,
            "diff_per_joint": per_joint,
            "diff_l2": diff_l2,
            "diff_max_abs": diff_max,
            "gt_l2_4090": gt_l2_4090,
            "gt_l2_b200": gt_l2_b200,
        }
        rows.append(row)
        print(f"\n[idx={idx:>5d}]")
        print(f"  4090:    [{', '.join(f'{v:>7.2f}' for v in a4[:6])}]")
        print(f"  B200:    [{', '.join(f'{v:>7.2f}' for v in ab[:6])}]")
        if gt_action is not None:
            print(f"  GT:      [{', '.join(f'{v:>7.2f}' for v in gt_action[:6])}]")
        print(f"  diff:    [{', '.join(f'{d:>+7.4f}' for d in per_joint)}]")
        print(f"  diff L2={diff_l2:.4f}, max|diff|={diff_max:.4f}, "
              f"gt_L2 4090={gt_l2_4090:.3f} B200={gt_l2_b200:.3f}")

    # Aggregate
    diffs_l2 = np.array([r["diff_l2"] for r in rows])
    diffs_max = np.array([r["diff_max_abs"] for r in rows])
    per_joint_arr = np.array([r["diff_per_joint"] for r in rows])  # (N, 6)
    gt_l2_4090 = np.array([r["gt_l2_4090"] for r in rows if r["gt_l2_4090"] is not None])
    gt_l2_b200 = np.array([r["gt_l2_b200"] for r in rows if r["gt_l2_b200"] is not None])

    summary = {
        "n_samples": len(rows),
        "seed": SEED,
        "diff_l2_mean": float(np.mean(diffs_l2)),
        "diff_l2_max": float(np.max(diffs_l2)),
        "diff_l2_min": float(np.min(diffs_l2)),
        "diff_max_abs_overall": float(np.max(diffs_max)),
        "per_joint_max_abs": [float(np.max(np.abs(per_joint_arr[:, j]))) for j in range(6)],
        "per_joint_mean_abs": [float(np.mean(np.abs(per_joint_arr[:, j]))) for j in range(6)],
        "gt_l2_mean_4090": float(np.mean(gt_l2_4090)),
        "gt_l2_mean_b200": float(np.mean(gt_l2_b200)),
        "gt_l2_diff": float(np.mean(gt_l2_b200) - np.mean(gt_l2_4090)),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    joint_names = ["Base", "Shoulder", "Elbow", "Wrist_P", "Wrist_R", "Gripper"]
    print(f"\n4090 vs B200 action diff (N={summary['n_samples']}, seed={SEED}):")
    print(f"  L2 diff:           mean={summary['diff_l2_mean']:.4f}  max={summary['diff_l2_max']:.4f}")
    print(f"  max|diff| overall: {summary['diff_max_abs_overall']:.4f} deg")
    print(f"\nPer-joint max|diff|:")
    for j, name in enumerate(joint_names):
        print(f"  {name:<10}: max={summary['per_joint_max_abs'][j]:.4f}  mean={summary['per_joint_mean_abs'][j]:.4f}")
    print(f"\nGT L2 (lower=better fit to ground truth):")
    print(f"  4090: {summary['gt_l2_mean_4090']:.4f}")
    print(f"  B200: {summary['gt_l2_mean_b200']:.4f}")
    print(f"  diff: {summary['gt_l2_diff']:+.4f} (positive = B200 worse)")

    # Verdict
    print("\n" + "=" * 60)
    if summary['diff_max_abs_overall'] < 0.1:
        verdict = "EXCELLENT — bit-exact-equivalent inference"
    elif summary['diff_max_abs_overall'] < 1.0:
        verdict = "PASS — sub-degree diff, deploy-equivalent"
    elif summary['diff_max_abs_overall'] < 5.0:
        verdict = "MARGINAL — diff exceeds typical closed-loop drift threshold"
    else:
        verdict = "FAIL — significant divergence, deploy NOT equivalent"
    print(f"VERDICT: {verdict}")
    print("=" * 60)

    out = {"summary": summary, "rows": rows, "verdict": verdict,
           "ckpt_4090": CKPT_4090_SRC, "ckpt_b200": CKPT_B200_SRC, "task": TASK_TEXT}
    with OUT_PATH.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
