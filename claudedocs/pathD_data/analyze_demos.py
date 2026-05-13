"""Path D Phase D.1 sanity analysis — load release_demos_v1.pt and verify:
  - Tensor shapes match expected (N, 200, 28) obs + (N, 200, 6) act
  - Action range [-1, 1] (env clip_actions=1.0)
  - Gripper dim (action[:, :, 5]) shows OPEN delta near success_step
  - Obs[5] (scaled gripper q) transitions from closed (high) to open (low) near success
  - Obs[14] (sponge_z_local) drops near success
  - success_step distribution stats
  - Recommendation: window strategy + size

Obs layout (per env code lines 439-450):
  [0:6]   = dof_pos_scaled[6]
  [6:12]  = joint_vel[6] * 0.1
  [12:15] = sponge_pos_local[3]
  [15:19] = sponge_quat_w[4]
  [19:22] = tcp_to_sponge[3]
  [22:25] = target_local[3]
  [25:28] = sponge_to_target[3]
"""
import torch
import numpy as np

d = torch.load("release_demos_v1.pt", weights_only=False)
obs = d["obs"]
action = d["action"]
succ = d["success_step"]
meta = d["meta"]

N, T, obs_dim = obs.shape
_, _, act_dim = action.shape
print(f"=== SHAPES ===")
print(f"obs        : {tuple(obs.shape)}")
print(f"action     : {tuple(action.shape)}")
print(f"success_step: {tuple(succ.shape)}  range=[{succ.min().item()}, {succ.max().item()}]")
print(f"meta keys  : {list(meta.keys())}")
print(f"meta success_rate: {meta['success_rate']:.4f}")

print(f"\n=== ACTION DIST (per dim, range check) ===")
print(f"{'dim':<5}{'min':>10}{'max':>10}{'mean':>10}{'std':>10}{'name':>20}")
names = ['base', 'shoulder', 'elbow', 'wrist_p', 'wrist_r', 'gripper']
for i in range(act_dim):
    a = action[:, :, i].flatten()
    print(f"{i:<5}{a.min().item():>10.3f}{a.max().item():>10.3f}{a.mean().item():>10.3f}{a.std().item():>10.3f}{names[i]:>20}")

print(f"\n=== GRIPPER ACTION (idx 5) NEAR success_step ===")
print(f"For each demo, show action[5] at [s-5, s-3, s-1, s, s+1, s+3, s+5]")
print(f"{'demo':<6}{'s':<6}{'s-5':>8}{'s-3':>8}{'s-1':>8}{'s':>8}{'s+1':>8}{'s+3':>8}{'s+5':>8}")
for i in range(min(N, 10)):
    s = int(succ[i].item())
    row = []
    for off in [-5, -3, -1, 0, 1, 3, 5]:
        idx = max(0, min(T-1, s+off))
        row.append(f"{action[i, idx, 5].item():>8.3f}")
    print(f"{i:<6}{s:<6}" + "".join(row))

print(f"\n=== OBS[5] (scaled gripper q) NEAR success_step (closed=+, open=-) ===")
print(f"{'demo':<6}{'s':<6}{'s-5':>8}{'s-3':>8}{'s-1':>8}{'s':>8}{'s+1':>8}{'s+3':>8}{'s+5':>8}")
for i in range(min(N, 10)):
    s = int(succ[i].item())
    row = []
    for off in [-5, -3, -1, 0, 1, 3, 5]:
        idx = max(0, min(T-1, s+off))
        row.append(f"{obs[i, idx, 5].item():>8.3f}")
    print(f"{i:<6}{s:<6}" + "".join(row))

print(f"\n=== OBS[14] (sponge_z_local) NEAR success_step ===")
print(f"{'demo':<6}{'s':<6}{'s-5':>8}{'s-3':>8}{'s-1':>8}{'s':>8}{'s+1':>8}{'s+3':>8}{'s+5':>8}")
for i in range(min(N, 10)):
    s = int(succ[i].item())
    row = []
    for off in [-5, -3, -1, 0, 1, 3, 5]:
        idx = max(0, min(T-1, s+off))
        row.append(f"{obs[i, idx, 14].item():>8.3f}")
    print(f"{i:<6}{s:<6}" + "".join(row))

print(f"\n=== success_step distribution (histogram bins of 20) ===")
hist, edges = np.histogram(succ.numpy(), bins=10)
for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
    bar = "#" * int(h)
    print(f"  [{int(e0):>4}, {int(e1):>4}): {h:>3} {bar}")

print(f"\n=== WINDOW RECOMMENDATION ===")
print(f"Median success_step = {int(succ.median().item())}  → release happens early on average")
print(f"With window [s-5, s+5] = 11 steps per demo, total = {N * 11} (obs, action) pairs")
print(f"With full trajectory T={T}, total = {N * T} pairs (high hover ratio, may dilute signal)")

# Check if action[5] (gripper) is consistently negative (open) at success_step
gripper_at_s = torch.stack([action[i, int(succ[i].item()), 5] for i in range(N)])
print(f"\naction[gripper] @ success_step: mean={gripper_at_s.mean():.3f}  std={gripper_at_s.std():.3f}")
print(f"  → all 20 demos negative (open command)? {(gripper_at_s < 0).all().item()}")
print(f"  → fraction negative: {(gripper_at_s < 0).float().mean().item():.2f}")

# Check obs[5] (gripper q) transition pattern: pre-s closed, post-s open
obs_g_pre = torch.stack([obs[i, max(0, int(succ[i].item())-5), 5] for i in range(N)])
obs_g_post = torch.stack([obs[i, min(T-1, int(succ[i].item())+5), 5] for i in range(N)])
print(f"obs[gripper_q_scaled] pre-s (s-5): mean={obs_g_pre.mean():.3f}  std={obs_g_pre.std():.3f}")
print(f"obs[gripper_q_scaled] post-s (s+5): mean={obs_g_post.mean():.3f}  std={obs_g_post.std():.3f}")
print(f"  → transition delta (post - pre): mean={(obs_g_post - obs_g_pre).mean():.3f}")
