"""Filter demos by gripper_open at success_step (obs[5] < -0.4).

Rationale: env has TWO paths to set _place_success_flag:
  (1) DIRECT path: success_now = on_target_upright & gripper_open & sponge_stable
  (2) COUNTER path: _place_counter >= 50 (gripper_open NOT required, line 302 design)

For release-BC training, we want ONLY demos from DIRECT path (genuine gripper-open release).
COUNTER-path successes are physics artifacts (sponge ends up in target zone without release).

obs[5] = scaled gripper q. Joint range [-0.175, 1.745] rad. grasp_gripper_thresh=0.4 rad.
  q=0.4 rad → obs[5] ≈ -0.40. So gripper_open ↔ obs[5] < -0.40.
"""
import torch
import numpy as np

d = torch.load("release_demos_v1.pt", weights_only=False)
obs = d["obs"]
action = d["action"]
succ = d["success_step"]

N, T, _ = obs.shape
GRIPPER_OPEN_THRESH = -0.40  # scaled obs[5] threshold

# Filter 1: gripper_open at success_step (DIRECT path indicator)
gripper_at_s = torch.tensor([obs[i, int(succ[i].item()), 5].item() for i in range(N)])
clean_mask_v1 = gripper_at_s < GRIPPER_OPEN_THRESH

# Filter 2: gripper trajectory shows OPENING during window (delta < 0 from s-5 to s+5 capped)
def safe_idx(i, s, off):
    return max(0, min(T-1, s+off))
delta_g = torch.tensor([
    obs[i, safe_idx(i, int(succ[i].item()), 3), 5].item()
    - obs[i, safe_idx(i, int(succ[i].item()), -5), 5].item()
    for i in range(N)
])
opening_mask = delta_g < -0.02  # opening: obs[5] decreases (more open)

clean_mask = clean_mask_v1  # primary filter
print(f"=== FILTER RESULTS ===")
print(f"Total demos          : {N}")
print(f"Filter1 (open@s)     : {int(clean_mask_v1.sum())}/{N}")
print(f"  threshold obs[5] < {GRIPPER_OPEN_THRESH}")
print(f"Filter2 (opening @win): {int(opening_mask.sum())}/{N}")
print(f"  threshold Δobs[5] < -0.02 over [s-5, s+3]")
print(f"Intersection         : {int((clean_mask_v1 & opening_mask).sum())}/{N}")

print(f"\n=== Per-demo breakdown ===")
print(f"{'demo':<6}{'s':<6}{'g@s':>10}{'Δg':>10}{'clean':>8}{'opening':>10}")
for i in range(N):
    g = obs[i, int(succ[i].item()), 5].item()
    dg = delta_g[i].item()
    c = "✓" if clean_mask_v1[i].item() else "✗"
    o = "✓" if opening_mask[i].item() else "✗"
    print(f"{i:<6}{int(succ[i].item()):<6}{g:>10.3f}{dg:>10.3f}{c:>8}{o:>10}")

print(f"\n=== success_step distribution (CLEAN only) ===")
clean_succ = succ[clean_mask].numpy()
hist, edges = np.histogram(clean_succ, bins=[0, 20, 40, 60, 80, 100, 150, 200])
for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
    print(f"  [{e0:>4}, {e1:>4}): {h:>3} {'#' * int(h)}")
print(f"  mean = {clean_succ.mean():.1f}  median = {int(np.median(clean_succ))}  "
      f"min = {clean_succ.min()}  max = {clean_succ.max()}")

# Save filtered demos
clean_ids = clean_mask.nonzero(as_tuple=False).flatten()
clean_obs = obs[clean_ids]
clean_act = action[clean_ids]
clean_succ_steps = succ[clean_ids]

# Also clip actions to [-1, 1] (raw policy mean is unbounded, env clips internally)
clean_act_clipped = torch.clamp(clean_act, -1.0, 1.0)

torch.save({
    "obs": clean_obs,
    "action": clean_act_clipped,
    "action_raw": clean_act,
    "success_step": clean_succ_steps,
    "filter": {"obs5_at_s_lt": GRIPPER_OPEN_THRESH, "n_orig": N, "n_clean": int(clean_mask.sum())},
    "meta": d["meta"],
}, "release_demos_v1_clean.pt")
print(f"\n=== SAVED ===")
print(f"release_demos_v1_clean.pt  ({clean_obs.shape[0]} demos)")
print(f"  obs.shape    : {tuple(clean_obs.shape)}")
print(f"  action.shape : {tuple(clean_act_clipped.shape)} (clipped to [-1, 1])")
print(f"  action_raw saturated frac: {((clean_act.abs() > 1.0).float().mean().item()):.2%}")
