"""Train a small BC joint-delta policy on a filtered DiffIK dataset."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def make_model(torch, in_dim: int, out_dim: int, hidden_dim: int, hidden_layers: int):
    layers = []
    last = in_dim
    for _ in range(int(hidden_layers)):
        layers.append(torch.nn.Linear(last, int(hidden_dim)))
        layers.append(torch.nn.ReLU())
        last = int(hidden_dim)
    layers.append(torch.nn.Linear(last, out_dim))
    return torch.nn.Sequential(*layers)


def row_bucket(row: dict[str, str]) -> str:
    return row.get("posx_x_bucket", "not_posx")


def row_is_posx(row: dict[str, str]) -> bool:
    return int(round(float(row.get("push_dx", 0.0)))) == 1 and int(round(float(row.get("push_dy", 0.0)))) == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", type=Path, required=True)
    ap.add_argument("--manifest_json", type=Path, required=True)
    ap.add_argument("--out_model", type=Path, required=True)
    ap.add_argument("--metrics_json", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--hidden_layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0e-4)
    ap.add_argument("--seed", type=int, default=779)
    ap.add_argument("--loss_mode", choices=("mse", "safety_l2"), default="mse")
    ap.add_argument("--posx_sample_weight", type=float, default=1.0)
    ap.add_argument("--lowx_sample_weight", type=float, default=1.0)
    ap.add_argument("--highx_sample_weight", type=float, default=1.0)
    ap.add_argument("--action_l2_weight", type=float, default=0.0)
    ap.add_argument("--posx_action_l2_weight", type=float, default=0.0)
    ap.add_argument("--lowx_action_l2_weight", type=float, default=0.0)
    ap.add_argument("--highx_action_l2_weight", type=float, default=0.0)
    ap.add_argument("--action_excess_limit_rad", type=float, default=0.0)
    ap.add_argument("--action_excess_weight", type=float, default=0.0)
    ap.add_argument("--posx_action_excess_weight", type=float, default=0.0)
    ap.add_argument("--lowx_action_excess_weight", type=float, default=0.0)
    ap.add_argument("--highx_action_excess_weight", type=float, default=0.0)
    args = ap.parse_args()

    import torch

    torch.manual_seed(int(args.seed))
    manifest = json.loads(args.manifest_json.read_text())
    feature_columns = list(manifest["feature_columns"])
    target_columns = list(manifest["target_columns"])
    rows = list(csv.DictReader(args.dataset_csv.open(newline="")))
    if not rows:
        raise ValueError(f"empty dataset: {args.dataset_csv}")
    missing = [c for c in feature_columns + target_columns + ["split"] if c not in rows[0]]
    if missing:
        raise ValueError(f"dataset missing columns: {missing}")

    by_split = {
        "train": [r for r in rows if r["split"] == "train"],
        "val": [r for r in rows if r["split"] == "val"],
        "test": [r for r in rows if r["split"] == "test"],
    }
    if not all(by_split.values()):
        raise ValueError({k: len(v) for k, v in by_split.items()})

    def tensorize(items: list[dict[str, str]], cols: list[str]) -> "torch.Tensor":
        return torch.tensor([[float(row[col]) for col in cols] for row in items], dtype=torch.float32)

    x_train = tensorize(by_split["train"], feature_columns)
    y_train = tensorize(by_split["train"], target_columns)
    x_val = tensorize(by_split["val"], feature_columns)
    y_val = tensorize(by_split["val"], target_columns)
    x_test = tensorize(by_split["test"], feature_columns)
    y_test = tensorize(by_split["test"], target_columns)

    def sample_weights(items: list[dict[str, str]]) -> "torch.Tensor":
        weights = []
        for row in items:
            weight = 1.0
            if row_is_posx(row):
                weight *= float(args.posx_sample_weight)
                bucket = row_bucket(row)
                if bucket == "low_x":
                    weight *= float(args.lowx_sample_weight)
                elif bucket == "high_x":
                    weight *= float(args.highx_sample_weight)
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)

    def safety_weights(items: list[dict[str, str]], base: float, posx: float, lowx: float, highx: float) -> "torch.Tensor":
        weights = []
        for row in items:
            weight = float(base)
            if row_is_posx(row):
                weight += float(posx)
                bucket = row_bucket(row)
                if bucket == "low_x":
                    weight += float(lowx)
                elif bucket == "high_x":
                    weight += float(highx)
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)

    train_sample_w = sample_weights(by_split["train"])
    l2_train_w = safety_weights(
        by_split["train"],
        float(args.action_l2_weight),
        float(args.posx_action_l2_weight),
        float(args.lowx_action_l2_weight),
        float(args.highx_action_l2_weight),
    )
    excess_train_w = safety_weights(
        by_split["train"],
        float(args.action_excess_weight),
        float(args.posx_action_excess_weight),
        float(args.lowx_action_excess_weight),
        float(args.highx_action_excess_weight),
    )

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    x_train_n = (x_train - x_mean) / x_std
    x_val_n = (x_val - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    model = make_model(torch, len(feature_columns), len(target_columns), int(args.hidden_dim), int(args.hidden_layers))
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = torch.nn.MSELoss()

    def supervised_loss(pred_n: "torch.Tensor", target_n: "torch.Tensor", sample_w: "torch.Tensor") -> "torch.Tensor":
        per_row = torch.mean((pred_n - target_n) ** 2, dim=1)
        return torch.sum(per_row * sample_w) / torch.clamp(torch.sum(sample_w), min=1.0e-6)

    def safety_loss(pred_n: "torch.Tensor", l2_w: "torch.Tensor", excess_w: "torch.Tensor") -> "torch.Tensor":
        if args.loss_mode != "safety_l2":
            return pred_n.new_tensor(0.0)
        pred_rad = pred_n * y_std + y_mean
        out = pred_n.new_tensor(0.0)
        if float(torch.max(l2_w).item()) > 0.0:
            per_row_l2 = torch.mean(pred_rad**2, dim=1)
            out = out + torch.mean(per_row_l2 * l2_w)
        if float(args.action_excess_limit_rad) > 0.0 and float(torch.max(excess_w).item()) > 0.0:
            excess = torch.relu(torch.abs(pred_rad) - float(args.action_excess_limit_rad))
            per_row_excess = torch.mean(excess**2, dim=1)
            out = out + torch.mean(per_row_excess * excess_w)
        return out

    with torch.no_grad():
        baseline_val = loss_fn(torch.zeros_like(y_val_n), y_val_n).item()
        baseline_test = loss_fn(torch.zeros_like(y_test_n), y_test_n).item()
    first_train = math.nan
    best_val = float("inf")
    best_state = None
    n_train = x_train_n.shape[0]
    batch_size = max(1, int(args.batch_size))
    for epoch in range(int(args.epochs)):
        model.train()
        order = torch.randperm(n_train)
        epoch_losses = []
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            pred = model(x_train_n[idx])
            bc_loss = supervised_loss(pred, y_train_n[idx], train_sample_w[idx])
            reg_loss = safety_loss(pred, l2_train_w[idx], excess_train_w[idx])
            loss = bc_loss + reg_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        if epoch == 0:
            first_train = sum(epoch_losses) / len(epoch_losses)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val_n), y_val_n).item()
        if val_loss < best_val:
            best_val = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_train_n = model(x_train_n)
        pred_val_n = model(x_val_n)
        pred_test_n = model(x_test_n)
        train_mse = loss_fn(pred_train_n, y_train_n).item()
        val_mse = loss_fn(pred_val_n, y_val_n).item()
        test_mse = loss_fn(pred_test_n, y_test_n).item()
        pred_test = pred_test_n * y_std + y_mean
        test_mae_rad = torch.mean(torch.abs(pred_test - y_test), dim=0)
        test_mae_mean_rad = float(torch.mean(test_mae_rad).item())
        test_max_abs_err_rad = float(torch.max(torch.abs(pred_test - y_test)).item())

    pass_bc = (
        train_mse < first_train
        and val_mse < baseline_val * 0.85
        and test_mse < baseline_test * 0.85
        and test_mae_mean_rad < 0.01
    )
    verdict = "PASS_BC_TRAINED_CHECKPOINT" if pass_bc else "FAIL_BC_TRAINED_CHECKPOINT"
    safety_config = {
        "loss_mode": args.loss_mode,
        "posx_sample_weight": float(args.posx_sample_weight),
        "lowx_sample_weight": float(args.lowx_sample_weight),
        "highx_sample_weight": float(args.highx_sample_weight),
        "action_l2_weight": float(args.action_l2_weight),
        "posx_action_l2_weight": float(args.posx_action_l2_weight),
        "lowx_action_l2_weight": float(args.lowx_action_l2_weight),
        "highx_action_l2_weight": float(args.highx_action_l2_weight),
        "action_excess_limit_rad": float(args.action_excess_limit_rad),
        "action_excess_weight": float(args.action_excess_weight),
        "posx_action_excess_weight": float(args.posx_action_excess_weight),
        "lowx_action_excess_weight": float(args.lowx_action_excess_weight),
        "highx_action_excess_weight": float(args.highx_action_excess_weight),
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "x_mean": x_mean.squeeze(0),
        "x_std": x_std.squeeze(0),
        "y_mean": y_mean.squeeze(0),
        "y_std": y_std.squeeze(0),
        "dataset_csv": str(args.dataset_csv),
        "manifest_json": str(args.manifest_json),
        "verdict": verdict,
        "safety_config": safety_config,
    }
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out_model)

    metrics = {
        "verdict": verdict,
        "dataset_csv": str(args.dataset_csv),
        "manifest_json": str(args.manifest_json),
        "out_model": str(args.out_model),
        "rows": len(rows),
        "train_rows": len(by_split["train"]),
        "val_rows": len(by_split["val"]),
        "test_rows": len(by_split["test"]),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "epochs": int(args.epochs),
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        **safety_config,
        "first_train_mse_norm": float(first_train),
        "final_train_mse_norm": float(train_mse),
        "baseline_val_mse_norm": float(baseline_val),
        "final_val_mse_norm": float(val_mse),
        "baseline_test_mse_norm": float(baseline_test),
        "final_test_mse_norm": float(test_mse),
        "test_mae_mean_rad": test_mae_mean_rad,
        "test_max_abs_err_rad": test_max_abs_err_rad,
        "supervised_bc_checkpoint": verdict.startswith("PASS"),
        "learned_policy": verdict.startswith("PASS"),
        "rollout_validated": False,
        "full_dataset_ready": manifest.get("full_dataset_candidate") is True,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(
        "diffik_bc_train line1 "
        f"rows={len(rows)} train_rows={len(by_split['train'])} val_rows={len(by_split['val'])} "
        f"test_rows={len(by_split['test'])} epochs={int(args.epochs)}"
    )
    print(
        "diffik_bc_train line2 "
        f"first_train_mse_norm={first_train:.9f} final_train_mse_norm={train_mse:.9f} "
        f"baseline_val_mse_norm={baseline_val:.9f} final_val_mse_norm={val_mse:.9f}"
    )
    print(
        "diffik_bc_train line3 "
        f"baseline_test_mse_norm={baseline_test:.9f} final_test_mse_norm={test_mse:.9f} "
        f"test_mae_mean_rad={test_mae_mean_rad:.9f} test_max_abs_err_rad={test_max_abs_err_rad:.9f}"
    )
    print(
        "diffik_bc_train line4 "
        f"model_path={args.out_model} verdict={verdict} learned_policy_checkpoint={'YES' if pass_bc else 'NO'} "
        "rollout_validated=NO"
    )
    print(
        "diffik_bc_train line5 "
        f"loss_mode={args.loss_mode} posx_sample_weight={float(args.posx_sample_weight):.6f} "
        f"lowx_sample_weight={float(args.lowx_sample_weight):.6f} highx_sample_weight={float(args.highx_sample_weight):.6f} "
        f"action_l2_weight={float(args.action_l2_weight):.6f} posx_action_l2_weight={float(args.posx_action_l2_weight):.6f} "
        f"lowx_action_l2_weight={float(args.lowx_action_l2_weight):.6f} highx_action_l2_weight={float(args.highx_action_l2_weight):.6f} "
        f"action_excess_limit_rad={float(args.action_excess_limit_rad):.6f}"
    )
    return 0 if pass_bc else 2


if __name__ == "__main__":
    raise SystemExit(main())
