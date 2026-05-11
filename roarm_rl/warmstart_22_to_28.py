"""Warm-start a 22-dim Phase 1.A checkpoint -> 28-dim Phase 1.B-alpha checkpoint.

Phase 1.A obs = 22 dim (joint_pos[6] + joint_vel[6] + sponge_pos[3] + sponge_quat[4] + tcp_to_sponge[3]).
Phase 1.B-alpha obs = 28 dim (Phase 1.A + target_pos_local[3] + sponge_to_target[3]).

Strategy:
  - Auto-detect any state_dict tensor of shape [..., 22] -> expand to [..., 28]:
      * 2-D Linear weight [out, 22] -> [out, 28] with new 6 cols zero-init
      * 1-D normalizer mean/var [22] -> [28] with new 6 dims default (mean=0, var=1)
  - Other tensors unchanged.
  - Optimizer state_dict size will mismatch -> drop optimizer (start fresh).

Usage:
  python -m roarm_rl.warmstart_22_to_28 \
      --in /path/to/phase1a_ckpt.pt \
      --out /path/to/phase1b_warmstart.pt \
      --inspect_only      # to inspect keys/shapes only

NOTE: this script does NOT require Isaac Sim. Pure torch.
"""
from __future__ import annotations

import argparse
import torch


OLD_DIM = 22
NEW_DIM = 28


def expand_2d(w: torch.Tensor, mode: str = "zero", default: float = 0.0) -> torch.Tensor:
    """Expand 2-D weight [out, OLD_DIM] -> [out, NEW_DIM]. New cols init to `default`."""
    assert w.dim() == 2 and w.shape[1] == OLD_DIM
    new_w = torch.full((w.shape[0], NEW_DIM), default, dtype=w.dtype, device=w.device)
    new_w[:, :OLD_DIM] = w
    if mode == "small_random":
        new_w[:, OLD_DIM:] = torch.randn(w.shape[0], NEW_DIM - OLD_DIM,
                                         dtype=w.dtype, device=w.device) * 0.01
    return new_w


def expand_1d(v: torch.Tensor, default: float) -> torch.Tensor:
    """Expand 1-D vector [OLD_DIM] -> [NEW_DIM] with new entries set to `default`."""
    assert v.dim() == 1 and v.shape[0] == OLD_DIM
    new_v = torch.empty(NEW_DIM, dtype=v.dtype, device=v.device)
    new_v[:OLD_DIM] = v
    new_v[OLD_DIM:] = default
    return new_v


def warmstart(ckpt_in: str, ckpt_out: str, mode: str = "zero", inspect_only: bool = False):
    print(f"[warmstart] loading: {ckpt_in}")
    ckpt = torch.load(ckpt_in, map_location="cpu", weights_only=False)

    print(f"[warmstart] top-level keys: {list(ckpt.keys())}")

    # Locate model state dict (rsl_rl 3.1.2 uses 'model_state_dict')
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        sd_key = "model_state_dict"
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        sd_key = "state_dict"
    else:
        # whole ckpt may BE the state_dict
        sd = ckpt
        sd_key = None

    print(f"[warmstart] state_dict keys ({len(sd)}):")
    expanded_2d, expanded_1d_keys, unchanged = [], [], []
    for k, v in sd.items():
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
        marker = ""
        if isinstance(v, torch.Tensor):
            if v.dim() == 2 and v.shape[1] == OLD_DIM:
                marker = "  <-- expand 2-D"
                expanded_2d.append(k)
            elif v.dim() == 1 and v.shape[0] == OLD_DIM:
                marker = "  <-- expand 1-D"
                expanded_1d_keys.append(k)
            else:
                unchanged.append(k)
        print(f"    {k}: {shape}{marker}")

    print(f"\n[warmstart] expand 2-D: {expanded_2d}")
    print(f"[warmstart] expand 1-D: {expanded_1d_keys}")
    print(f"[warmstart] {len(unchanged)} keys unchanged.")

    if inspect_only:
        print("[warmstart] inspect_only=True -> not writing.")
        return

    # Apply expansion
    for k in expanded_2d:
        old = sd[k]
        kl = k.lower()
        if "var" in kl or "_std" in kl:
            d2 = 1.0
        elif "mean" in kl:
            d2 = 0.0
        else:
            d2 = 0.0  # Linear weight: new cols zero-init
        sd[k] = expand_2d(old, mode=mode, default=d2)
        print(f"[warmstart] expanded 2-D '{k}' (default={d2}): "
              f"{tuple(old.shape)} -> {tuple(sd[k].shape)}")

    for k in expanded_1d_keys:
        # Heuristic: running_mean default=0, running_var default=1, others default=0
        if "var" in k.lower():
            default = 1.0
        elif "mean" in k.lower():
            default = 0.0
        else:
            default = 0.0
        old = sd[k]
        sd[k] = expand_1d(old, default=default)
        print(f"[warmstart] expanded 1-D '{k}' (default={default}): "
              f"{tuple(old.shape)} -> {tuple(sd[k].shape)}")

    # Optimizer: drop (size will mismatch)
    if "optimizer_state_dict" in ckpt:
        print("[warmstart] dropping optimizer_state_dict (will be re-created on resume).")
        del ckpt["optimizer_state_dict"]

    # Re-attach if state_dict was nested
    if sd_key is not None:
        ckpt[sd_key] = sd

    print(f"[warmstart] saving: {ckpt_out}")
    torch.save(ckpt, ckpt_out)
    print("[warmstart] DONE.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="ckpt_in", required=True)
    parser.add_argument("--out", dest="ckpt_out", required=False, default=None)
    parser.add_argument("--mode", choices=["zero", "small_random"], default="zero",
                        help="New 6-dim weight init: zero (default) or small_random (std 0.01)")
    parser.add_argument("--inspect_only", action="store_true",
                        help="Print keys/shapes without writing")
    args = parser.parse_args()

    if not args.inspect_only and args.ckpt_out is None:
        parser.error("--out required unless --inspect_only")

    warmstart(args.ckpt_in, args.ckpt_out, mode=args.mode, inspect_only=args.inspect_only)


if __name__ == "__main__":
    main()
