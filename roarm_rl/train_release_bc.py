"""Path D Phase D.2 — Train release BC from clean demos.

Input : release_demos_v1_clean.pt (Filter1-pass demos, action clipped to [-1, 1]).
Window: [0, min(success_step + window_post, T-1)] — approach + release window,
        excludes post-success drift where policy oscillates.
Arch  : 28 → hidden → Tanh(6). Tanh bounds output to env's effective action range.
Loss  : MSE on clipped actions (matches env's clip(action, -1, 1) behavior).
Opt   : Adam lr=1e-3 wd=1e-4, batch=32, early stop on val plateau.
Split : Train/val split BY DEMO (not by pair) to avoid leakage.

Other-agent insights applied:
  - Small demo budget → weight_decay + early stop (regularization)
  - Subskill BC (release-only) per Path D design
  - State-only 28-dim (no rendering — HARD RULE #17)

Run locally:
  conda activate roarm
  python -m roarm_rl.train_release_bc \
      --demos claudedocs/pathD_data/release_demos_v1_clean.pt \
      --output_dir claudedocs/pathD_data/bc_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_frac", type=float, default=0.20)
    parser.add_argument("--early_stop_patience", type=int, default=50)
    parser.add_argument("--window_post_succ", type=int, default=5,
                        help="Window per demo: [0, min(s+window_post_succ, T-1)] inclusive.")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = torch.load(args.demos, weights_only=False)
    obs = d["obs"]
    action = d["action"]
    succ_step = d["success_step"]
    N, T, obs_dim = obs.shape
    act_dim = action.shape[-1]
    print(f"[bc] Loaded {N} demos  T={T}  obs={obs_dim}  act={act_dim}")
    print(f"[bc] action range  : [{action.min():.3f}, {action.max():.3f}]  "
          f"(env clip [-1,1]; |raw|>1 frac={d.get('filter', {})})")

    pairs_obs, pairs_act = [], []
    for i in range(N):
        s = int(succ_step[i].item())
        end = min(s + args.window_post_succ + 1, T)
        pairs_obs.append(obs[i, :end])
        pairs_act.append(action[i, :end])
    total_pairs = sum(p.shape[0] for p in pairs_obs)
    print(f"[bc] Window [0, s+{args.window_post_succ}]: total {total_pairs} pairs "
          f"(mean {total_pairs/N:.1f} per demo)")

    n_val = max(1, int(round(N * args.val_frac)))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
    val_ids = sorted(perm[:n_val].tolist())
    train_ids = sorted(perm[n_val:].tolist())
    train_obs = torch.cat([pairs_obs[i] for i in train_ids], dim=0)
    train_act = torch.cat([pairs_act[i] for i in train_ids], dim=0)
    val_obs = torch.cat([pairs_obs[i] for i in val_ids], dim=0)
    val_act = torch.cat([pairs_act[i] for i in val_ids], dim=0)
    print(f"[bc] Train : {train_obs.shape[0]} pairs ({len(train_ids)} demos, ids={train_ids})")
    print(f"[bc] Val   : {val_obs.shape[0]} pairs ({len(val_ids)} demos, ids={val_ids})")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[bc] device: {device}")

    model = nn.Sequential(
        nn.Linear(obs_dim, args.hidden),
        nn.ELU(),
        nn.Linear(args.hidden, act_dim),
        nn.Tanh(),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bc] arch  : Linear({obs_dim}→{args.hidden}) ELU Linear({args.hidden}→{act_dim}) Tanh")
    print(f"[bc] params: {n_params}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    train_obs_d = train_obs.to(device)
    train_act_d = train_act.to(device)
    val_obs_d = val_obs.to(device)
    val_act_d = val_act.to(device)

    train_losses, val_losses = [], []
    best_val, best_epoch = float("inf"), -1
    best_state = None
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(train_obs_d.shape[0], device=device)
        ep_loss, n_b = 0.0, 0
        for b_start in range(0, train_obs_d.shape[0], args.batch_size):
            b = idx[b_start:b_start + args.batch_size]
            pred = model(train_obs_d[b])
            loss = loss_fn(pred, train_act_d[b])
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += loss.item()
            n_b += 1
        ep_loss /= max(1, n_b)
        train_losses.append(ep_loss)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_obs_d), val_act_d).item()
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"[bc] ep={epoch:>4}  train={ep_loss:.5f}  val={val_loss:.5f}  "
                  f"best_val={best_val:.5f}@{best_epoch}")

        if patience >= args.early_stop_patience:
            print(f"[bc] Early stop @ epoch {epoch} (no val improve for {args.early_stop_patience} eps)")
            break

    torch.save({
        "model_state_dict": best_state,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden": args.hidden,
        "arch": f"Linear({obs_dim}→{args.hidden}) ELU Linear({args.hidden}→{act_dim}) Tanh",
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(train_losses),
        "train_demo_ids": train_ids,
        "val_demo_ids": val_ids,
        "args": vars(args),
    }, out_dir / "release_bc.pt")
    np.savez(out_dir / "losses.npz",
             train=np.array(train_losses), val=np.array(val_losses))

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(val_obs_d).cpu()
    val_act_cpu = val_act_d.cpu()
    per_dim_mse = ((val_pred - val_act_cpu) ** 2).mean(dim=0)
    per_dim_corr = []
    for k in range(act_dim):
        vp = val_pred[:, k] - val_pred[:, k].mean()
        va = val_act_cpu[:, k] - val_act_cpu[:, k].mean()
        denom = (vp.std() * va.std()) + 1e-8
        per_dim_corr.append(((vp * va).mean() / denom).item())

    names = ["base", "shoulder", "elbow", "wrist_p", "wrist_r", "gripper"]
    print()
    print("=== BC TRAIN SUMMARY ===")
    print(f"epochs_run    : {len(train_losses)}")
    print(f"best_val_loss : {best_val:.5f} @ epoch {best_epoch}")
    print(f"final_train   : {train_losses[-1]:.5f}")
    print(f"{'dim':<5}{'name':<12}{'val_mse':>10}{'corr':>10}")
    for k, name in enumerate(names):
        print(f"{k:<5}{name:<12}{per_dim_mse[k].item():>10.5f}{per_dim_corr[k]:>10.4f}")
    print(f"saved         : {out_dir / 'release_bc.pt'}")


if __name__ == "__main__":
    main()
