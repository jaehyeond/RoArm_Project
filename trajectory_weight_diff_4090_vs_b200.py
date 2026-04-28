"""4090 vs B200 weight L2 diff across training trajectory (5K/10K/15K/20K)."""
import json
from pathlib import Path

import torch
from safetensors import safe_open

STEPS = ["005000", "010000", "015000", "last"]
STEP_LABELS = {"005000": "5K", "010000": "10K", "015000": "15K", "last": "20K"}


def load_state(path):
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def diff_state(s4, sb):
    assert set(s4.keys()) == set(sb.keys()), "key mismatch"
    rows = []
    for k in s4.keys():
        a = s4[k].float()
        b = sb[k].float()
        if a.shape != b.shape:
            print(f"  SHAPE MISMATCH {k}: {a.shape} vs {b.shape}")
            continue
        diff = a - b
        l2 = float(diff.norm())
        mx = float(diff.abs().max())
        ref = float(a.norm())
        rel = l2 / max(ref, 1e-12)
        rows.append({"key": k, "rel_l2": rel, "l2": l2, "max_abs": mx,
                     "numel": int(a.numel()), "ref_l2": ref})
    total_l2 = sum(r["l2"] ** 2 for r in rows) ** 0.5
    total_max = max(r["max_abs"] for r in rows)
    bit_exact = sum(1 for r in rows if r["rel_l2"] < 1e-7)
    # Separate trainable Action Expert MLP layers (visible from key prefix)
    trainable = [r for r in rows if "lm_expert.layers" in r["key"] and "mlp" in r["key"]]
    trainable_l2 = sum(r["l2"] ** 2 for r in trainable) ** 0.5
    trainable_mean_rel = sum(r["rel_l2"] for r in trainable) / max(len(trainable), 1)
    return {
        "global_l2": total_l2,
        "global_max_abs": total_max,
        "bit_exact": bit_exact,
        "n_total": len(rows),
        "n_trainable_mlp": len(trainable),
        "trainable_l2": trainable_l2,
        "trainable_mean_rel": trainable_mean_rel,
        "top10": sorted(rows, key=lambda r: -r["rel_l2"])[:10],
    }


def main():
    results = {}
    for s in STEPS:
        p4 = f"outputs/smolvla_v6/checkpoints/{s}/pretrained_model/model.safetensors"
        pb = f"outputs/smolvla_v6_b200/checkpoints/{s}/pretrained_model/model.safetensors"
        if not Path(p4).exists():
            print(f"MISSING 4090 {s}: {p4}")
            continue
        if not Path(pb).exists():
            print(f"MISSING B200 {s}: {pb}")
            continue
        print(f"Loading {s}...", flush=True)
        s4 = load_state(p4)
        sb = load_state(pb)
        d = diff_state(s4, sb)
        results[STEP_LABELS[s]] = d
        print(f"  {STEP_LABELS[s]}: global_l2={d['global_l2']:.4f}, "
              f"max|diff|={d['global_max_abs']:.4e}, "
              f"bit_exact={d['bit_exact']}/{d['n_total']}, "
              f"trainable_mean_rel={d['trainable_mean_rel']:.4f}", flush=True)

    print()
    print("=" * 110)
    print("Trajectory weight diff: 4090 vs B200 (cuDNN/cuBLAS noise across training)")
    print("=" * 110)
    print(f"{'step':<6} {'global_L2':>12} {'global_max':>14} {'bit_exact':>14} "
          f"{'trainable_L2':>14} {'tr_mean_rel':>12} {'top1_rel':>12} top1_key")
    for label, d in results.items():
        top = d["top10"][0]
        print(f"{label:<6} {d['global_l2']:>12.4f} {d['global_max_abs']:>14.4e} "
              f"{d['bit_exact']:>5}/{d['n_total']:<8} "
              f"{d['trainable_l2']:>14.4f} {d['trainable_mean_rel']:>12.4e} "
              f"{top['rel_l2']:>12.4e} {top['key']}")

    # Save full results
    out_path = Path("claudedocs/trajectory_weight_diff_4090_vs_b200.json")
    out_path.parent.mkdir(exist_ok=True)
    serializable = {
        label: {k: v for k, v in d.items()}
        for label, d in results.items()
    }
    with out_path.open("w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Trajectory analysis
    print()
    print("=" * 110)
    print("Trajectory analysis (does noise grow / saturate?)")
    print("=" * 110)
    labels = list(results.keys())
    for i, label in enumerate(labels):
        d = results[label]
        if i > 0:
            prev = results[labels[i-1]]
            growth_l2 = d["global_l2"] - prev["global_l2"]
            growth_max = d["global_max_abs"] - prev["global_max_abs"]
            print(f"  {labels[i-1]} -> {label}: "
                  f"L2 +{growth_l2:+.4f}, max|diff| +{growth_max:+.4e}")
        else:
            print(f"  {label} baseline: L2={d['global_l2']:.4f}, max|diff|={d['global_max_abs']:.4e}")


if __name__ == "__main__":
    main()
