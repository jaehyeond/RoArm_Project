"""W13 사전검토 ③ — D324 결정 시점 시각 진단(정적 기하). DEME 실행 0.
전 경로 립 궤적·트레이·용기·HOME/취점/배출 마커를 XY·XZ·배출 상세 3패널로 그린다.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MAIN = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(MAIN)); sys.path.insert(0, str(HERE))
import sim_deme_scoop_s1 as W11SRC
import w13_kinematics as K
import w13_fk as FK

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
P = dict(W11SRC.DEFAULT); P.update(K.W13_DEFAULT); P.update(json.load(open(sys.argv[2])))
q_open = P["door_open_servo_deg"] - P["servo_zero_offset_deg"]
fixed_m, door_m, lip_f, lip_d, hinge_off, L5mm, _ = W11SRC.load_tool(P, q_open)
P["lip_l5_mm"] = [float(v) for v in L5mm]
Vt = np.vstack([np.asarray(fixed_m.vertices, float), np.asarray(door_m.vertices, float) + hinge_off])
# rev34: 더미 경로 = 3번째 인자(없으면 rev32 더미). 트레이·자리·어댑터·펠릿면은 params 의 rev34 키를 따른다.
pile = np.load(sys.argv[3] if len(sys.argv) > 3 else
               "/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/"
               "pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz", allow_pickle=True)
box, _tray_info = FK.w25_tray_bounds(np.asarray(pile["box_bounds_m"], float), P, fail_closed=False)
box_top = float(box[2, 1])
site_xy, site_info = FK.w25_scoop_site(P, P["lip_l5_mm"])
site_w11 = (0.0, 0.0) if site_info.get("rev32_path") else site_xy
z_surf_pre = W11SRC.surface_z(np.asarray(pile["positions_m"], float), np.asarray(pile["radii_m"], float),
                              *((P["scoop_x_mm"] / 1000.0, 0.0) if site_info.get("rev32_path") else site_xy), P)
z_lip0 = z_surf_pre + (P["w25_door_open_gap_mm"] if P.get("w25_proc_open_at_surface") else P["approach_gap_mm"]) / 1000.0
ad, ad_info, frame_info = FK.build_adapter_w25(box, z_surf_pre, P, P["lip_l5_mm"])
R_sc = np.eye(3) if frame_info["rev32_path"] else np.asarray(frame_info["R_scoop_owner_box"], float)
dz_bot = (z_surf_pre - P["plunge_mm"] / 1000.0) - z_lip0
dz_top = ((z_surf_pre + P["w25_proc_lift_to_surface_mm"] / 1000.0) - z_lip0
          if P.get("w25_proc_lift_to_surface_mm") is not None else dz_bot + P["lift_mm"] / 1000.0)
r0 = float(np.hypot(*FK.lip_pose(ad_info["reference_pose_q5"], P["lip_l5_mm"])[0][:2]))
z_travel = float(ad_info["floor_robot_z_m"] + P["travel_cm"] / 100.0 - ad_info["t_robot_m"][2])
z_surf5 = float(ad_info["pellet_robot_z_m"] + 0.05 - ad_info["t_robot_m"][2])
tray = K.tray_mesh(box, P["tray_wall_t_mm"] / 1000.0)
wps0, _ = FK.build_waypoints(ad, r0, z_travel, z_lip0, z_surf5, 0.0, P["place_base_deg"])
bc = np.asarray([w for w in wps0 if w["name"] == "place_target"][0]["lip_world_m"])[:2]
binm, bi = K.bin_mesh(P, bc)
z_rel = bi["rim_z_m"] + P["release_clearance_mm"] / 1000.0
wps, wx = FK.build_waypoints(ad, r0, z_travel, z_lip0, z_surf5, z_rel, P["place_base_deg"])
Wp = {w["name"]: w for w in wps}
order = ["initial_home", "p1_tool_vertical", "above_pile_travel", "surface_plus_50mm", "approach_gap",
         "post_lift_travel", "place_retract_base0", "place_retract_base90", "place_extend_travel",
         "place_target", "place_up_travel", "return_retract_base90", "return_retract_base0", "return_home"]
path, tools = [], {}
for k in range(len(order) - 1):
    q0 = np.asarray(Wp[order[k]]["q5"], float); q1 = np.asarray(Wp[order[k + 1]]["q5"], float)
    for t in np.linspace(0, 1, 60):
        p, Rm = ad.owner_pose((q0 + (q1 - q0) * t).tolist()); path.append(p)
col = [np.array([site_w11[0], site_w11[1], z_lip0 + dz]) for dz in
       np.linspace(0, dz_bot, 40).tolist() + np.linspace(dz_bot, dz_top, 40).tolist()]
path = np.asarray(path); col = np.asarray(col)
for nm in ("initial_home", "approach_gap", "place_target", "above_pile_travel"):
    p, Rm = ad.owner_pose(Wp[nm]["q5"]); tools[nm] = (Rm @ Vt.T).T + p
tools["w11_plunge"] = (R_sc @ Vt.T).T + np.array([site_w11[0], site_w11[1], z_lip0 + dz_bot])

fig, ax = plt.subplots(1, 3, figsize=(19, 6))
tb, bb = tray.bounds, binm.bounds
ax[0].add_patch(plt.Rectangle((box[0, 0] * 1000, box[1, 0] * 1000), (box[0, 1] - box[0, 0]) * 1000,
                              (box[1, 1] - box[1, 0]) * 1000, fill=False, ec="tab:green", lw=2, label="source tray (inner)"))
ax[0].add_patch(plt.Circle((bc[0] * 1000, bc[1] * 1000), bi["inner_r_m"] * 1000, fill=False, ec="tab:red", lw=2, label="receiving bin (inner)"))
ax[0].plot(path[:, 0] * 1000, path[:, 1] * 1000, "-", c="0.4", lw=1.2, label="lip path (joint interp)")
ax[0].plot(*(np.asarray(wx["base_axis_world_xy_m"]) * 1000), "k^", ms=9, label="robot base axis")
for nm, c_ in (("initial_home", "tab:blue"), ("approach_gap", "tab:orange"), ("place_target", "tab:red")):
    p = np.asarray(Wp[nm]["lip_world_m"]) * 1000
    ax[0].plot(p[0], p[1], "o", c=c_, ms=8); ax[0].annotate(nm, (p[0], p[1]), fontsize=7)
for nm, T in tools.items():
    ax[0].scatter(T[:, 0] * 1000, T[:, 1] * 1000, s=1, alpha=0.35)
ax[0].set_title(f"top view (box/DEME XY, mm) — convention {frame_info['box_frame_convention']}"); ax[0].set_aspect("equal"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[1].plot(path[:, 0] * 1000, path[:, 2] * 1000, "-", c="0.4", lw=1.2, label="lip path")
ax[1].plot(col[:, 0] * 1000, col[:, 2] * 1000, "-", c="tab:purple", lw=2.5, label="W11 scoop column (frozen)")
ax[1].axhline(box_top * 1000, ls="--", c="tab:green", lw=1, label=f"box top {box_top*1000:.1f}")
ax[1].axhline(bi["rim_z_m"] * 1000, ls="--", c="tab:red", lw=1, label=f"bin rim {bi['rim_z_m']*1000:.1f}")
ax[1].axhline(z_travel * 1000, ls=":", c="0.3", lw=1, label=f"travel {z_travel*1000:.1f}")
ax[1].axhline(z_surf_pre * 1000, ls="-.", c="tab:brown", lw=1, label=f"pellet surface {z_surf_pre*1000:.1f}")
for nm, T in tools.items():
    ax[1].scatter(T[:, 0] * 1000, T[:, 2] * 1000, s=1, alpha=0.5, label=f"tool @ {nm}")
ax[1].set_title("side view (world XZ, mm)"); ax[1].set_aspect("equal"); ax[1].legend(fontsize=6); ax[1].grid(alpha=.3)
T = tools["place_target"]
ax[2].scatter(T[:, 1] * 1000, T[:, 2] * 1000, s=3, c="tab:blue", label="tool shell @ release")
bv = np.asarray(binm.vertices) * 1000
m = np.abs(bv[:, 0] - bc[0] * 1000) < 6
ax[2].scatter(bv[m, 1], bv[m, 2], s=3, c="tab:red", label="bin section")
ax[2].axhline(z_rel * 1000, ls="--", c="k", lw=1, label=f"release lip z {z_rel*1000:.1f}")
ax[2].set_title("release detail (YZ, mm) — box(DEME) frame"); ax[2].set_aspect("equal")
ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
fig.suptitle("rev34 preflight geometry snapshot (box/DEME frame) — declared scene fixture, no physics run")
fig.tight_layout(); fig.savefig(out / "geometry_decision_snapshot.png", dpi=120); plt.close(fig)
print("saved", out / "geometry_decision_snapshot.png")
