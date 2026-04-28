"""STL geometry audit for RoArm-M3 gripper.

Reads gripper_link.stl, gripper_left_link.stl, link5.stl from Waveshare
official URDF. Reports AABB dimensions, finger tip position, and computes
jaw inner width as a function of rotation angle (0~1.5 rad = 0~85.94 deg).

URDF transforms (from roarm_m3.xacro):
  link5 frame -> gripper_link: xyz=(0, 0.018821, 0.052035), rpy=(-pi/2,-pi/2,0)
  link5 frame -> hand_tcp: xyz=(0, 0, 0.115428), rpy=(pi/2, -pi/2, 0)  (FIXED)

The gripper_left_link.stl has no joint in URDF — assumed fixed counter-jaw
welded to link5 housing (we treat it as in link5 frame, NOT moved by joint).
"""
import numpy as np
import trimesh

# ======================================================================
# URDF transforms (meters -> mm)
# ======================================================================

def rpy_to_R(roll, pitch, yaw):
    """URDF RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

PI = np.pi
T_LINK5_TO_GRIPPER = {
    "xyz_m": np.array([0.0, 0.018821, 0.052035]),
    "rpy_rad": np.array([-PI/2, -PI/2, 0.0]),
}
T_LINK5_TO_TCP = {
    "xyz_m": np.array([0.0, 0.0, 0.115428]),
    "rpy_rad": np.array([PI/2, -PI/2, 0.0]),
}

def transform_mesh(mesh, R, t_m):
    """Apply rotation R (3x3) and translation t (meters) to mesh vertices."""
    out = mesh.copy()
    out.vertices = (R @ out.vertices.T).T + t_m
    return out

# ======================================================================
# Load meshes
# ======================================================================

print("=" * 78)
print("RoArm-M3 GRIPPER STL GEOMETRY AUDIT")
print("Source: github.com/waveshareteam/roarm_ws (ros2-humble)")
print("=" * 78)

m_grip = trimesh.load("mesh_audit/gripper_link.stl", force="mesh")
m_left = trimesh.load("mesh_audit/gripper_left_link.stl", force="mesh")
m_l5 = trimesh.load("mesh_audit/link5.stl", force="mesh")

# URDF mesh scale=0.001 means STL is in mm, URDF reads as meters scale-down.
# trimesh just reports raw vertex units. Per Waveshare URDF convention,
# STL vertices are likely in mm. Confirm via AABB scale.
def stl_units_check(mesh, name):
    bb = mesh.bounding_box.extents
    print(f"  {name:30s} AABB: {bb[0]:7.2f} x {bb[1]:7.2f} x {bb[2]:7.2f} (raw STL units)")
    return bb

print("\n--- STL files (raw vertex units, expected mm per URDF scale=0.001) ---")
bb_grip = stl_units_check(m_grip, "gripper_link.stl")
bb_left = stl_units_check(m_left, "gripper_left_link.stl")
bb_l5 = stl_units_check(m_l5, "link5.stl")

# Sanity: link5 STL Z extent should match link5_to_hand_tcp ~115mm region
print("\n  Sanity: link5 max Z extent (raw) vs URDF link5_to_hand_tcp 115.428mm")
print(f"  link5 Z range: [{m_l5.vertices[:, 2].min():.2f}, {m_l5.vertices[:, 2].max():.2f}]")

# If STL is in mm, AABB ~10-150 mm makes sense. If in meters, ~0.01-0.15.
unit_mm = bb_l5.max() > 1.5  # heuristic: >1.5 means mm
print(f"\n  Inferred STL unit: {'mm' if unit_mm else 'meters'}")
SCALE = 1.0 if unit_mm else 1000.0  # scale to mm

# ======================================================================
# Place gripper_link in link5 frame at angle theta (0=closed, 1.5rad=open)
# ======================================================================

print("\n--- Placing gripper_link in link5 frame ---")
print(f"  Joint origin xyz (mm): {T_LINK5_TO_GRIPPER['xyz_m']*1000}")
print(f"  Joint origin rpy (rad): {T_LINK5_TO_GRIPPER['rpy_rad']}")
print(f"  Axis: [0, 0, 1] in joint frame (z-axis = rotation axis)")

# At theta=0 (closed): apply only the joint origin transform (no rotation about axis).
# The STL vertices are in gripper_link's own frame; the axis direction in that
# frame is +Z. To put in link5 frame: R_origin @ Rz(theta) @ vertex + t_origin.

R_origin_gl = rpy_to_R(*T_LINK5_TO_GRIPPER["rpy_rad"])
t_origin_gl_mm = T_LINK5_TO_GRIPPER["xyz_m"] * 1000

R_origin_tcp = rpy_to_R(*T_LINK5_TO_TCP["rpy_rad"])
t_origin_tcp_mm = T_LINK5_TO_TCP["xyz_m"] * 1000

def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def gripper_in_link5(mesh, theta_rad):
    """Place gripper_link mesh in link5 frame, rotated by theta about joint axis."""
    v = mesh.vertices * SCALE  # to mm
    v_rotated = (Rz(theta_rad) @ v.T).T  # rotate about local Z
    v_link5 = (R_origin_gl @ v_rotated.T).T + t_origin_gl_mm
    return v_link5

def left_in_link5(mesh):
    """gripper_left_link: assume same origin as right gripper_link (mirror counter-jaw)."""
    # Hypothesis: gripper_left_link.stl is the counter-jaw, fixed in link5 frame.
    # We don't know its mount point — try same origin as gripper_link first.
    v = mesh.vertices * SCALE
    v_link5 = (R_origin_gl @ v.T).T + t_origin_gl_mm
    return v_link5

# hand_tcp position in link5 frame
tcp_link5_mm = t_origin_tcp_mm
print(f"\n  hand_tcp in link5 frame (mm): {tcp_link5_mm}")

# ======================================================================
# Geometry analysis
# ======================================================================

print("\n--- Gripper jaw geometry @ closed (theta=0) ---")
v_grip_closed = gripper_in_link5(m_grip, 0.0)
v_left_fixed = left_in_link5(m_left)
v_l5 = m_l5.vertices * SCALE  # already in link5 frame

# AABB in link5 frame
def aabb(vertices, label):
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    print(f"  {label}:")
    print(f"    X range [{mn[0]:7.2f}, {mx[0]:7.2f}]  width  {mx[0]-mn[0]:6.2f}")
    print(f"    Y range [{mn[1]:7.2f}, {mx[1]:7.2f}]  depth  {mx[1]-mn[1]:6.2f}")
    print(f"    Z range [{mn[2]:7.2f}, {mx[2]:7.2f}]  height {mx[2]-mn[2]:6.2f}")
    return mn, mx

aabb(v_l5, "link5 (link5 frame)")
aabb(v_grip_closed, "gripper_link @ theta=0 (link5 frame)")
aabb(v_left_fixed, "gripper_left_link (link5 frame, hypothesis: same origin)")
aabb(np.vstack([v_l5, v_grip_closed, v_left_fixed]),
     "FULL gripper assembly (link5+grip+left)")

print(f"\n  hand_tcp in link5 frame (mm): {tcp_link5_mm}")
print(f"  hand_tcp Z = {tcp_link5_mm[2]:.2f}")

# Finger tip: most distal point along the gripper opening direction.
# After URDF transform, the gripper rotation axis in link5 frame is some direction.
# The "distal" direction (finger tip) should be the extremum away from the joint origin.
# Heuristic: find vertex farthest from joint origin (t_origin_gl_mm).
print("\n--- Finger tip estimate ---")
def farthest_from(vertices, origin_mm, label):
    d = np.linalg.norm(vertices - origin_mm, axis=1)
    idx = d.argmax()
    p = vertices[idx]
    dist = d[idx]
    print(f"  {label}: farthest vertex {p} (dist={dist:.2f} mm from joint origin)")
    return p, dist

p_tip_grip, d_tip_grip = farthest_from(v_grip_closed, t_origin_gl_mm, "gripper_link tip")
p_tip_left, d_tip_left = farthest_from(v_left_fixed, t_origin_gl_mm, "gripper_left_link tip")
p_tip_grip_from_tcp, _ = farthest_from(v_grip_closed, tcp_link5_mm, "gripper_link tip (from hand_tcp)")
p_tip_left_from_tcp, _ = farthest_from(v_left_fixed, tcp_link5_mm, "gripper_left_link tip (from hand_tcp)")

# Finger length = farthest-from-joint-origin distance
print(f"\n  CANDIDATE finger length (gripper_link, max from joint origin) = {d_tip_grip:.2f} mm")
print(f"  CANDIDATE finger length (gripper_left_link)                    = {d_tip_left:.2f} mm")

# ======================================================================
# Jaw inner width vs angle (sweep theta)
# ======================================================================

print("\n--- Jaw inner width vs theta (rotating gripper_link, fixed gripper_left_link) ---")
print(f"  {'theta_rad':>10} {'theta_deg':>10} {'min_dist_mm':>13}")

THETAS = np.linspace(0, 1.5, 16)  # 0 to 85.94 deg
for theta in THETAS:
    v_g = gripper_in_link5(m_grip, theta)
    # Min pairwise distance between the two jaws (sample every 50th vertex for speed)
    sample_g = v_g[::max(1, len(v_g)//500)]
    sample_l = v_left_fixed[::max(1, len(v_left_fixed)//500)]
    diff = sample_g[:, None, :] - sample_l[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    print(f"  {theta:10.4f} {np.degrees(theta):10.2f} {d.min():13.3f}")

print("\n  NOTE: 'min_dist' is closest distance between two jaw surfaces.")
print("  If gripper_left_link is mounted at SAME origin as gripper_link, this")
print("  represents inner-jaw clearance at angle theta. If mount differs, this")
print("  is wrong and we need the other jaw's actual joint/fixed-mount point.")

print("=" * 78)
print("DONE")
