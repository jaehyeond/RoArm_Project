"""Small BC smoke test for a DiffIK step-level trace.

This validates that the pilot trace can feed a supervised learner. It is not a
policy-quality, rollout, or deployment claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FEATURE_COLUMNS = [
    "push_dx",
    "push_dy",
    "phase_alpha",
    "cube_x_m",
    "cube_y_m",
    "cube_z_m",
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "arm_joint_0_rad",
    "arm_joint_1_rad",
    "arm_joint_2_rad",
    "arm_joint_3_rad",
    "arm_joint_4_rad",
    "gripper_joint_rad",
]
TARGET_COLUMNS = [
    "joint_delta_0_rad",
    "joint_delta_1_rad",
    "joint_delta_2_rad",
    "joint_delta_3_rad",
    "joint_delta_4_rad",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_csv", type=Path, required=True)
    ap.add_argument("--metrics_json", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=779)
    args = ap.parse_args()

    import torch

    torch.manual_seed(int(args.seed))
    rows = list(csv.DictReader(args.trace_csv.open(newline="")))
    if not rows:
        raise ValueError(f"empty trace csv: {args.trace_csv}")
    missing = [c for c in FEATURE_COLUMNS + TARGET_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    env_ids = sorted({int(r["env_id"]) for r in rows})
    test_count = max(1, len(env_ids) // 5)
    test_envs = set(env_ids[-test_count:])
    train_rows = [r for r in rows if int(r["env_id"]) not in test_envs]
    test_rows = [r for r in rows if int(r["env_id"]) in test_envs]

    def tensorize(items: list[dict[str, str]], columns: list[str]) -> torch.Tensor:
        return torch.tensor([[float(r[c]) for c in columns] for r in items], dtype=torch.float32)

    x_train = tensorize(train_rows, FEATURE_COLUMNS)
    y_train = tensorize(train_rows, TARGET_COLUMNS)
    x_test = tensorize(test_rows, FEATURE_COLUMNS)
    y_test = tensorize(test_rows, TARGET_COLUMNS)

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train_n = (x_train - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    model = torch.nn.Sequential(
        torch.nn.Linear(len(FEATURE_COLUMNS), 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, len(TARGET_COLUMNS)),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        baseline_test_mse = loss_fn(torch.zeros_like(y_test_n), y_test_n).item()
    first_train_mse = None
    for epoch in range(int(args.epochs)):
        model.train()
        pred = model(x_train_n)
        loss = loss_fn(pred, y_train_n)
        if first_train_mse is None:
            first_train_mse = float(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_mse = loss_fn(model(x_train_n), y_train_n).item()
        test_mse = loss_fn(model(x_test_n), y_test_n).item()

    verdict = "PASS_BC_PIPELINE_SMOKE"
    if not (test_mse < baseline_test_mse and train_mse < float(first_train_mse)):
        verdict = "FAIL_BC_PIPELINE_SMOKE"

    metrics = {
        "verdict": verdict,
        "trace_csv": str(args.trace_csv),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "env_count": len(env_ids),
        "test_envs": sorted(test_envs),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "epochs": int(args.epochs),
        "first_train_mse_norm": float(first_train_mse),
        "final_train_mse_norm": float(train_mse),
        "baseline_test_mse_norm": float(baseline_test_mse),
        "final_test_mse_norm": float(test_mse),
        "learned_policy": False,
        "rollout_validated": False,
        "full_dataset_ready": False,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(
        "diffik_bc_smoke line1 "
        f"rows={len(rows)} train_rows={len(train_rows)} test_rows={len(test_rows)} "
        f"env_count={len(env_ids)} epochs={int(args.epochs)}"
    )
    print(
        "diffik_bc_smoke line2 "
        f"first_train_mse_norm={float(first_train_mse):.9f} "
        f"final_train_mse_norm={train_mse:.9f} "
        f"baseline_test_mse_norm={baseline_test_mse:.9f} "
        f"final_test_mse_norm={test_mse:.9f}"
    )
    print(
        "diffik_bc_smoke line3 "
        f"verdict={verdict} learned_policy=NO rollout_validated=NO full_dataset_ready=NO"
    )
    return 0 if verdict.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
