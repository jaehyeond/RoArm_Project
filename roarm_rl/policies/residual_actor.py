"""Residual Policy Learning (Silver et al., 2018, arxiv 1812.06298) for rsl_rl ActorCritic.

Architecture:
    final_action = clip(π_BC(obs) + alpha × π_residual(obs), -1, +1)
    π_BC: pretrained MLP (frozen weights, requires_grad=False)
    π_residual: small trainable MLP (28 → 64 → 32 → 6 by default)

PPO trains ONLY π_residual. By construction, zero forgetting of BC initialization:
    - BC weights never updated (requires_grad=False → grad=None at optimizer.step)
    - Residual α controls how far PPO can deviate (0.3 = ±0.3 max per action dim)

Round 2 evidence (5/14):
    - 4 PPO RL attempts on RoArm-M3 stack: P6v14a/b/c + P6v15 all collapse via
      "8th farming" (grasp-hold reward farming).
    - iter 0 stage4=0.36 (good) → iter 10 stage4=0.003 (98% collapse).
    - Root cause: PPO advantage prefers low-variance grasp basin over high-variance
      release path. Even desired_kl=0.005 + adaptive schedule + bias reset insufficient.
    - RPL (Silver 2018) provides STRUCTURAL fix: BC never overwritten.

Usage (drop-in replacement for ActorCritic.actor):
    bc_actor = trained MLP (loaded from P6v14a/model_499.pt)
    residual_mlp = nn.Sequential(...)  # small MLP, 28 → 64 → 32 → 6
    actor_critic.actor = ResidualMLPWrapper(bc_actor, residual_mlp, alpha=0.3)
    # BC frozen automatically inside __init__.

References:
    - Silver et al. (2018) "Residual Policy Learning" arXiv:1812.06298
    - Johannink et al. (2019) "Residual Reinforcement Learning for Robot Control"
    - Ankile et al. (2025) "Residual Off-Policy RL for Finetuning BC Policies" arxiv:2509.19301
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualMLPWrapper(nn.Module):
    """Frozen BC base + trainable residual.

    Outputs: final_action_pre_clip = bc(x) + alpha * residual(x).
    PPO sees this as the actor mean. std parameter (in ActorCritic) is unchanged
    (separately learnable).

    Args:
        bc_mlp: nn.Module — pretrained policy (frozen here).
        residual_mlp: nn.Module — small trainable head, same in/out dims as bc_mlp.
        alpha: float — residual scale (0.3 default). Lower = closer to BC, higher = freer.
    """

    def __init__(self, bc_mlp: nn.Module, residual_mlp: nn.Module, alpha: float = 0.3):
        super().__init__()
        self.bc_mlp = bc_mlp
        self.residual_mlp = residual_mlp
        self.alpha = alpha
        for p in self.bc_mlp.parameters():
            p.requires_grad = False
        self.bc_mlp.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            bc_out = self.bc_mlp(x)
        res_out = self.residual_mlp(x)
        return bc_out + self.alpha * res_out

    def train(self, mode: bool = True):
        super().train(mode)
        self.bc_mlp.eval()  # always keep BC in eval mode (no dropout/BN updates)
        return self


def build_residual_mlp(num_obs: int, num_actions: int,
                       hidden_dims=(64, 32), activation: str = "elu") -> nn.Module:
    """Build a small MLP for the residual head. Zero-init final layer so initial
    action == BC (residual contributes 0 at start)."""
    act_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers = []
    in_dim = num_obs
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(act_fn())
        in_dim = h
    final = nn.Linear(in_dim, num_actions)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


def install_residual_actor(actor_critic, bc_state_dict: dict,
                           alpha: float = 0.3,
                           residual_hidden: tuple = (64, 32)) -> dict:
    """Replace actor_critic.actor with ResidualMLPWrapper.

    Steps:
        1. Load BC state_dict into actor_critic (strict=False) — populates actor,
           actor_obs_normalizer, critic, std.
        2. Snapshot actor as bc_mlp (independent module).
        3. Build new residual MLP (zero-init final layer).
        4. Wrap: actor_critic.actor = ResidualMLPWrapper(bc_mlp, residual_mlp, alpha).
        5. Critic is NOT frozen (trained from scratch — value func needs PPO updates).

    Returns: dict with debug info {'bc_params', 'residual_params', 'alpha'}.
    """
    import copy
    ret = actor_critic.load_state_dict(bc_state_dict, strict=False)
    missing = list(ret[0]) if isinstance(ret, tuple) else []
    unexpected = list(ret[1]) if isinstance(ret, tuple) else []
    print(f"[residual] BC load missing={missing} unexpected={unexpected}")

    bc_actor_orig = actor_critic.actor
    bc_actor = copy.deepcopy(bc_actor_orig)

    # Infer num_obs from BC actor's first Linear
    first_linear = None
    for m in bc_actor.modules():
        if isinstance(m, nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        raise RuntimeError("BC actor has no Linear layer")
    num_obs = first_linear.in_features

    last_linear = None
    for m in bc_actor.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    num_actions = last_linear.out_features

    residual_mlp = build_residual_mlp(num_obs, num_actions, residual_hidden, "elu")
    residual_mlp = residual_mlp.to(next(bc_actor.parameters()).device)

    wrapper = ResidualMLPWrapper(bc_actor, residual_mlp, alpha=alpha)
    actor_critic.actor = wrapper

    bc_params = sum(p.numel() for p in bc_actor.parameters())
    res_params = sum(p.numel() for p in residual_mlp.parameters())
    print(f"[residual] alpha={alpha} bc_params={bc_params} (frozen) residual_params={res_params} (trainable)")
    return {"bc_params": bc_params, "residual_params": res_params, "alpha": alpha}
