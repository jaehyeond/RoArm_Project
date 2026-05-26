"""Offline eval of OpenVLA-OFT 7B LoRA checkpoints against v6 LeRobot holdout.

Loads each checkpoint (LoRA adapter + L1 action head) on top of the base
`openvla/openvla-7b` model, runs single-image inference on every frame of the
selected holdout episodes, and aggregates per-checkpoint metrics:

  * `l2_step0`         L2 norm of (predicted chunk[0] - GT action[t]) in degrees
  * `mae_per_joint`    Mean absolute error per joint, degrees
  * `zscore_per_joint` MAE / dataset std per joint (from stats.json)
  * `diversity`        Std of predicted chunk[0] per joint across frames
  * `l2_step0_train`   Optional sanity check against a training subset

Designed to run on B200 (sm_100, torch nightly cu128, transformers 4.57.6, no
flash-attn). The script applies the openvla-oft `_supports_sdpa` class-attr
patch monkey-style before instantiating any prismatic model so the same code
works whether or not the cached HF snapshot has been patched in place.

Usage (B200):
    python eval_offline_v6.py \
        --base_model openvla/openvla-7b \
        --checkpoint_root outputs/openvla_oft_v6_b200 \
        --dataset_repo_id roarm_v6_pick \
        --dataset_root /NHNHOME/.../JHPark/roarm_b200/data/lerobot_dataset_v6 \
        --holdout_episodes 45 46 47 48 49 \
        --train_sanity_episodes 0 1 2 3 4 \
        --output results/openvla_oft_v6_eval.json

The `--dataset_repo_id roarm_v6_pick` substring "roarm" triggers
`prismatic/vla/constants.py:detect_robot_platform` to load ROARM_M3 constants
(ACTION_DIM=6, NUM_ACTIONS_CHUNK=8).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


# --------------------------------------------------------------------------- #
# sdpa class-attr patch (D071)                                                #
# --------------------------------------------------------------------------- #
def apply_sdpa_class_attr_patch() -> None:
    """Override `_supports_sdpa` as class attr on the prismatic base class.

    The stock prismatic `modeling_prismatic.py` exposes `_supports_sdpa` as a
    `@property` that reads `self.language_model._supports_sdpa`. Transformers
    >=4.50 reads `cls._supports_sdpa` during `super().__init__` before any
    submodule attribute is set, which crashes with AttributeError. Setting a
    bool class attr resolves the lookup without altering runtime semantics.
    """
    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_VISION_2_SEQ_MAPPING,  # noqa: F401  # warm up registry
        )
    except ImportError:
        pass

    try:
        import transformers_modules  # noqa: F401
    except Exception:
        pass

    def _patch_module(mod) -> None:
        for name in dir(mod):
            if not name.endswith("PreTrainedModel"):
                continue
            cls = getattr(mod, name, None)
            if cls is None:
                continue
            if isinstance(cls, type) and "PrismaticPreTrainedModel" in cls.__name__:
                cls._supports_sdpa = True

    import importlib
    candidate_modules = [
        "prismatic.extern.hf.modeling_prismatic",
    ]
    for mod_name in candidate_modules:
        try:
            mod = importlib.import_module(mod_name)
            _patch_module(mod)
        except Exception:
            continue


# --------------------------------------------------------------------------- #
# ROARM_M3 constants trigger                                                  #
# --------------------------------------------------------------------------- #
def ensure_roarm_constants(repo_id: str) -> None:
    """Inject `--dataset_name <repo_id>` into sys.argv so
    `prismatic/vla/constants.py:detect_robot_platform` picks ROARM_M3 (action_dim=6)."""
    if "roarm" not in repo_id.lower():
        raise SystemExit(
            f"--dataset_repo_id must contain 'roarm' (got {repo_id}); "
            "ROARM_M3 constants detection relies on it."
        )
    if not any("roarm" in a.lower() for a in sys.argv):
        sys.argv.insert(1, f"--dataset_name={repo_id}")


# --------------------------------------------------------------------------- #
# Holdout dataset                                                             #
# --------------------------------------------------------------------------- #
class HoldoutFrames:
    """Iterable over (image, language, gt_chunk_unnormalized) for selected episodes."""

    def __init__(
        self,
        repo_id: str,
        root: Path,
        episodes: List[int],
        num_action_chunk: int = 8,
        action_dim: int = 6,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.root = Path(root)
        self.num_chunk = int(num_action_chunk)
        self.action_dim = int(action_dim)
        self.episodes = sorted(int(e) for e in episodes)

        self.ds = LeRobotDataset(repo_id=repo_id, root=str(self.root))
        ep_meta = self.ds.meta.episodes
        self._ep_from = np.asarray(ep_meta["dataset_from_index"], dtype=np.int64)
        self._ep_to = np.asarray(ep_meta["dataset_to_index"], dtype=np.int64)
        self.num_episodes_total = int(self._ep_from.shape[0])

        self.frames: List[Tuple[int, int, int]] = []
        for ep in self.episodes:
            if ep >= self.num_episodes_total:
                raise SystemExit(
                    f"Episode {ep} requested but dataset has only {self.num_episodes_total}"
                )
            a, b = int(self._ep_from[ep]), int(self._ep_to[ep])
            for gi in range(a, b):
                self.frames.append((ep, gi, b))

        stats_path = self.root / "meta" / "stats.json"
        with stats_path.open() as f:
            stats = json.load(f)
        self.action_mean = np.asarray(stats["action"]["mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action"]["std"], dtype=np.float32)
        self.action_q01 = np.asarray(stats["action"]["q01"], dtype=np.float32)
        self.action_q99 = np.asarray(stats["action"]["q99"], dtype=np.float32)

        print(
            f"[HoldoutFrames] repo={repo_id} eps={self.episodes} "
            f"total_frames={len(self.frames)} chunk={self.num_chunk}"
        )

    def __len__(self) -> int:
        return len(self.frames)

    def _gt_chunk(self, gi: int, ep_to: int) -> np.ndarray:
        out = np.zeros((self.num_chunk, self.action_dim), dtype=np.float32)
        last = None
        for k in range(self.num_chunk):
            j = gi + k
            if j < ep_to:
                a = self.ds[int(j)]["action"]
                last = a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
            out[k] = last
        return out.astype(np.float32)

    def __getitem__(self, i: int) -> Dict:
        ep, gi, ep_to = self.frames[i]
        sample = self.ds[int(gi)]
        img_t = sample["observation.images.top"]
        if torch.is_tensor(img_t):
            img_np = (img_t.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        else:
            img_np = np.asarray(img_t)
        task_str = str(sample.get("task", "Pick up the sponge")).rstrip("\n").strip()
        if not task_str:
            task_str = "Pick up the sponge"
        return {
            "ep": ep,
            "frame": gi,
            "image": img_np,
            "task": task_str,
            "gt_chunk": self._gt_chunk(gi, ep_to),
        }


# --------------------------------------------------------------------------- #
# Checkpoint loader                                                           #
# --------------------------------------------------------------------------- #
def discover_checkpoints(root: Path) -> List[Tuple[int, Path]]:
    pairs: List[Tuple[int, Path]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        suffix = sub.name.rsplit("--", 1)[-1]
        if not suffix.endswith("_chkpt"):
            continue
        try:
            step = int(suffix.replace("_chkpt", ""))
        except ValueError:
            continue
        pairs.append((step, sub))
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_vla_with_lora(
    base_model: str,
    chkpt_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module, "AutoProcessor", Dict]:
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    # Pin the commit so transformers does NOT re-download fresh modeling/processing
    # over the patched fork files in the local HF cache.
    revision = os.environ.get(
        "OPENVLA_REVISION",
        "47a0ec7fc4ec123775a391911046cf33cf9ed83f",
    )
    print(f"[load] base={base_model}@{revision}")
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right",
        revision=revision,
        local_files_only=True,
    )
    apply_sdpa_class_attr_patch()
    vla = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        revision=revision,
        local_files_only=True,
    ).to(device)
    apply_sdpa_class_attr_patch()  # second pass after dynamic class realisation

    print(f"[load] lora_adapter={chkpt_dir / 'lora_adapter'}")
    vla = PeftModel.from_pretrained(vla, str(chkpt_dir / "lora_adapter"))
    # Do not call merge_and_unload — D073 keeps merge off for inference safety
    # under PEFT 0.18; PeftModel forward path works directly.
    vla = vla.to(device=device, dtype=dtype).eval()

    stats_path = chkpt_dir / "dataset_statistics.json"
    with stats_path.open() as f:
        raw_stats = json.load(f)
    norm_stats: Dict[str, Dict] = {}
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
    # The actual prismatic model that owns `predict_action` / `_check_unnorm_key`
    # is at vla.base_model.model after PeftModel wrap. Set on every level we can
    # reach so __getattr__ chains still resolve.
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

    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

    llm_dim = vla.config.text_config.hidden_size if hasattr(vla.config, "text_config") else vla.config.hidden_size
    action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)

    ah_files = list(chkpt_dir.glob("action_head--*_checkpoint.pt"))
    if not ah_files:
        raise SystemExit(f"No action_head checkpoint in {chkpt_dir}")
    ah_path = sorted(ah_files)[0]
    print(f"[load] action_head={ah_path}")
    state = torch.load(ah_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
    action_head.load_state_dict(state)
    action_head = action_head.to(device=device, dtype=dtype).eval()

    cfg = {
        "ACTION_DIM": ACTION_DIM,
        "NUM_ACTIONS_CHUNK": NUM_ACTIONS_CHUNK,
        "action_head_path": str(ah_path),
        "lora_adapter_path": str(chkpt_dir / "lora_adapter"),
    }
    return vla, action_head, processor, cfg


# --------------------------------------------------------------------------- #
# Inference per frame                                                         #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def predict_chunk(
    vla,
    action_head,
    processor,
    image_uint8: np.ndarray,
    task: str,
    unnorm_key: str,
    device: torch.device,
    dtype: torch.dtype,
    image_size: int = 224,
) -> np.ndarray:
    """Return predicted action chunk shape (NUM_ACTIONS_CHUNK, ACTION_DIM), unnormalized."""
    pil = Image.fromarray(image_uint8).convert("RGB")
    if pil.size != (image_size, image_size):
        pil = pil.resize((image_size, image_size), Image.BILINEAR)
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
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def compute_metrics(
    preds_step0: np.ndarray,  # (N, action_dim)
    gts_step0: np.ndarray,    # (N, action_dim)
    preds_chunk_mean: np.ndarray,  # (N, action_dim) avg over chunk
    gts_chunk_mean: np.ndarray,    # (N, action_dim) avg over chunk
    action_std: np.ndarray,
) -> Dict:
    diff0 = preds_step0 - gts_step0
    l2_step0 = np.linalg.norm(diff0, axis=1)
    mae_step0 = np.mean(np.abs(diff0), axis=0)
    z_step0 = mae_step0 / np.maximum(action_std, 1e-6)

    diff_c = preds_chunk_mean - gts_chunk_mean
    l2_chunk_avg = np.linalg.norm(diff_c, axis=1)
    mae_chunk = np.mean(np.abs(diff_c), axis=0)

    diversity_step0 = np.std(preds_step0, axis=0)
    return {
        "l2_step0_mean": float(np.mean(l2_step0)),
        "l2_step0_median": float(np.median(l2_step0)),
        "l2_step0_p95": float(np.percentile(l2_step0, 95)),
        "l2_step0_std": float(np.std(l2_step0)),
        "l2_chunk_avg_mean": float(np.mean(l2_chunk_avg)),
        "mae_per_joint_step0": [float(x) for x in mae_step0],
        "mae_per_joint_chunk_avg": [float(x) for x in mae_chunk],
        "zscore_per_joint_step0": [float(x) for x in z_step0],
        "diversity_per_joint_step0": [float(x) for x in diversity_step0],
        "n_frames": int(preds_step0.shape[0]),
    }


# --------------------------------------------------------------------------- #
# Eval driver                                                                 #
# --------------------------------------------------------------------------- #
def eval_one_checkpoint(
    base_model: str,
    chkpt_dir: Path,
    holdout: HoldoutFrames,
    train_sanity: HoldoutFrames | None,
    device: torch.device,
    dtype: torch.dtype,
    repo_id: str,
) -> Dict:
    t0 = time.time()
    vla, action_head, processor, cfg = load_vla_with_lora(base_model, chkpt_dir, dtype, device)
    load_secs = time.time() - t0

    def _run(frames: HoldoutFrames) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        preds_step0 = np.zeros((len(frames), holdout.action_dim), dtype=np.float32)
        preds_chunk_mean = np.zeros_like(preds_step0)
        gts_step0 = np.zeros_like(preds_step0)
        gts_chunk_mean = np.zeros_like(preds_step0)
        for i in range(len(frames)):
            f = frames[i]
            chunk = predict_chunk(
                vla, action_head, processor,
                image_uint8=f["image"], task=f["task"],
                unnorm_key=repo_id, device=device, dtype=dtype,
            )
            preds_step0[i] = chunk[0]
            preds_chunk_mean[i] = chunk.mean(axis=0)
            gts_step0[i] = f["gt_chunk"][0]
            gts_chunk_mean[i] = f["gt_chunk"].mean(axis=0)
            if (i + 1) % 100 == 0:
                print(f"  frame {i+1}/{len(frames)}")
        return preds_step0, preds_chunk_mean, gts_step0, gts_chunk_mean

    print(f"[eval] holdout {len(holdout)} frames")
    t1 = time.time()
    p0, pc, g0, gc_ = _run(holdout)
    holdout_secs = time.time() - t1
    holdout_metrics = compute_metrics(p0, g0, pc, gc_, holdout.action_std)
    holdout_metrics["seconds"] = holdout_secs

    train_metrics = None
    if train_sanity is not None and len(train_sanity) > 0:
        print(f"[eval] train_sanity {len(train_sanity)} frames")
        t2 = time.time()
        p0t, pct, g0t, gct = _run(train_sanity)
        train_metrics = compute_metrics(p0t, g0t, pct, gct, train_sanity.action_std)
        train_metrics["seconds"] = time.time() - t2

    out = {
        "checkpoint_dir": str(chkpt_dir),
        "load_seconds": load_secs,
        "config": cfg,
        "holdout": holdout_metrics,
        "train_sanity": train_metrics,
    }

    del vla, action_head, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--checkpoint_root", required=True)
    ap.add_argument("--dataset_repo_id", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--holdout_episodes", type=int, nargs="+", required=True)
    ap.add_argument("--train_sanity_episodes", type=int, nargs="*", default=[])
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--only_steps", type=int, nargs="*", default=None,
                    help="If set, evaluate only checkpoints whose step is in this list.")
    args = ap.parse_args()

    ensure_roarm_constants(args.dataset_repo_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    if device.type == "cpu":
        raise SystemExit("GPU required (no CUDA detected).")

    checkpoint_root = Path(args.checkpoint_root)
    pairs = discover_checkpoints(checkpoint_root)
    if args.only_steps:
        wanted = set(args.only_steps)
        pairs = [(s, p) for s, p in pairs if s in wanted]
    if not pairs:
        raise SystemExit(f"No checkpoints found under {checkpoint_root}")
    print(f"[main] found {len(pairs)} checkpoints: {[s for s, _ in pairs]}")

    holdout = HoldoutFrames(
        repo_id=args.dataset_repo_id,
        root=Path(args.dataset_root),
        episodes=args.holdout_episodes,
    )
    train_sanity = None
    if args.train_sanity_episodes:
        train_sanity = HoldoutFrames(
            repo_id=args.dataset_repo_id,
            root=Path(args.dataset_root),
            episodes=args.train_sanity_episodes,
        )

    out: Dict = {
        "started_at": _dt.datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "device": str(device),
        "dtype": str(dtype),
        "n_checkpoints": len(pairs),
        "holdout_episodes": args.holdout_episodes,
        "train_sanity_episodes": args.train_sanity_episodes,
        "action_std_from_stats": [float(x) for x in holdout.action_std],
        "per_checkpoint": [],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for step, sub in pairs:
        print(f"\n=== checkpoint step={step} dir={sub.name} ===")
        try:
            res = eval_one_checkpoint(
                args.base_model, sub, holdout, train_sanity, device, dtype, args.dataset_repo_id
            )
            res["step"] = step
            out["per_checkpoint"].append(res)
            with output_path.open("w") as f:
                json.dump(out, f, indent=2)
            print(f"  step={step} l2_step0_mean={res['holdout']['l2_step0_mean']:.4f}")
        except Exception as e:  # noqa: BLE001
            err = {"step": step, "checkpoint_dir": str(sub), "error": repr(e)}
            out["per_checkpoint"].append(err)
            with output_path.open("w") as f:
                json.dump(out, f, indent=2)
            print(f"  step={step} ERROR: {e!r}")

    out["finished_at"] = _dt.datetime.utcnow().isoformat() + "Z"
    with output_path.open("w") as f:
        json.dump(out, f, indent=2)

    best = sorted(
        (r for r in out["per_checkpoint"] if "holdout" in r),
        key=lambda r: r["holdout"]["l2_step0_mean"],
    )
    print("\n=== ranking by holdout.l2_step0_mean (lower is better) ===")
    for r in best:
        print(
            f"  step={r['step']:>6d}  l2_step0_mean={r['holdout']['l2_step0_mean']:.4f}  "
            f"l2_chunk_avg_mean={r['holdout']['l2_chunk_avg_mean']:.4f}  "
            f"sanity_l2={r['train_sanity']['l2_step0_mean'] if r['train_sanity'] else 'NA'}"
        )
    print(f"\nwrote {output_path} sha256={hashlib.sha256(output_path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
