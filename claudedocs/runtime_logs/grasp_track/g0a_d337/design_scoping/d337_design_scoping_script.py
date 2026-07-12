#!/usr/bin/env python3
"""D337 design-time offline scoping (NOT decision evidence).

Measures the natural tool-axis elevation of the position-only family and the
raw-tool clearance landscape vs an explicit tool-axis elevation variable, using
the URDF collision STLs (D334-verified bit-near parity with the audited stage
meshes) and the same hppfcl query path as D336.
"""
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path('/home/cgxr/Documents/Robotics/RoArm_Project')
sys.path.insert(0, str(REPO))

import trimesh
import hppfcl

from sim_scripts import cube10cm_top_view_d323_grasp_g0a_frame_repair_probe as d323
from sim_scripts import cyl34_top_view_d330_grasp_g0a_alignment_probe as d330
from sim_scripts import cyl34_top_view_d332_grasp_g0a_static_collision_discriminator as d332

CENTER = d332.OBJECT_CENTER_LOCAL_M
RADIUS = d332.CYLINDER_RADIUS_M
HEIGHT = d332.CYLINDER_HEIGHT_M
TANGENT = np.array([0.0, -1.0, 0.0])  # ADOPTED_TANGENT_SIGN=-1, radial=[1,0,0]
RADIAL = np.array([1.0, 0.0, 0.0])


def rpy_to_rot(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


# gripper_link pose in link5 frame (URDF joint origin, q5=0)
G_IN_L5_T = np.array([0.0, 0.018821, 0.052035])
G_IN_L5_R = rpy_to_rot([-1.5708, -1.5708, 0.0])

link5_mesh = trimesh.load(REPO / 'local_assets/roarm_m3/urdf/meshes/link5.stl')
grip_mesh = trimesh.load(REPO / 'local_assets/roarm_m3/urdf/meshes/gripper_link_collision_g2a.stl')
link5_v = np.asarray(link5_mesh.vertices, dtype=np.float64) * 0.001
link5_f = np.asarray(link5_mesh.faces, dtype=np.int64)
grip_v = np.asarray(grip_mesh.vertices, dtype=np.float64) * 0.001
grip_f = np.asarray(grip_mesh.faces, dtype=np.int64)
link5_bvh = d332._build_raw_bvh(hppfcl, link5_v, link5_f)
grip_bvh = d332._build_raw_bvh(hppfcl, grip_v, grip_f)
cylinder = hppfcl.Cylinder(RADIUS, HEIGHT)
cyl_tf = hppfcl.Transform3f(np.eye(3), np.asarray(CENTER, dtype=np.float64))


def exact_sep_mm(bvh, rot, pos):
    tf = hppfcl.Transform3f(rot, pos)
    q = d332._fcl_query(hppfcl, bvh, tf, cylinder, cyl_tf)
    if not q['is_collision']:
        return float(q['signed_distance_mm']), 'clear' if q['signed_distance_mm'] >= 0.1 else 'near'
    req = hppfcl.CollisionRequest(); req.enable_contact = True; req.num_max_contacts = 64
    res = hppfcl.CollisionResult()
    hppfcl.collide(bvh, tf, cylinder, cyl_tf, req, res)
    depths = [abs(float(res.getContact(i).penetration_depth)) for i in range(res.numContacts())]
    return (-max(depths) * 1000.0 if depths else float('nan')), 'overlap'


def target_tcp_for(r_mm, t_mm):
    tcp = np.asarray(CENTER, dtype=np.float64).copy()
    tcp -= RADIAL * (r_mm * 1e-3)
    tcp -= TANGENT * (t_mm * 1e-3)
    tcp[2] = CENTER[2]
    return tcp


def elevation_of(z_axis):
    return math.degrees(math.atan2(-z_axis[2], math.hypot(z_axis[0], z_axis[1])))


def eval_pose(q_deg, target_tcp):
    tcp, l5_pos, l5_rot = d323._fk_runtime_tcp(np.asarray(q_deg))
    g_pos = l5_pos + l5_rot @ G_IN_L5_T
    g_rot = l5_rot @ G_IN_L5_R
    l5_sep, l5_state = exact_sep_mm(link5_bvh, l5_rot, l5_pos)
    g_sep, g_state = exact_sep_mm(grip_bvh, g_rot, g_pos)
    align, _ = d330._evaluate_alignment(
        trial=1, obj_center=CENTER, obj_radius_m=RADIUS, obj_height_m=HEIGHT,
        target_tcp=target_tcp, tangent=TANGENT, actual_tcp=tcp, link5_rot=l5_rot,
        obj_start_w=CENTER, obj_final_w=CENTER,
        target_arm=np.radians(np.asarray(q_deg)[:5]), actual_arm=np.radians(np.asarray(q_deg)[:5]),
        ik_failure_steps=0)
    return {
        'tcp_err_mm': float(np.linalg.norm(tcp - target_tcp) * 1000),
        'elev_deg': elevation_of(l5_rot[:, 2]),
        'l5_sep': l5_sep, 'g_sep': g_sep, 'g_state': g_state,
        'jaw_tan_deg': align['jaw_tangent_error_deg'],
        'jaw_gap_mm': align['fixed_jaw_face_gap_mm'],
        'jaw_pen_mm': align['fixed_jaw_penetration_mm'],
        'below_top_mm': align['contact_point_below_top_mm'],
        'gates': bool(align['pass_jaw_tangent'] and align['pass_fixed_jaw_gap']
                      and align['pass_no_penetration'] and align['pass_contact_height']),
    }


print('=== natural elevation of position-only family (HOME seed) ===')
for (r, t) in [(7.0, 11.0), (14.6, 13.9), (15.25, 9.0)]:
    tt = target_tcp_for(r, t)
    ik = d323._solve_runtime_ik(tt, d332.HOME_DEG, max_iter=120, pos_tol_mm=1.0)
    q = np.asarray(ik['q_deg']); q[5] = 0
    m = eval_pose(q, tt)
    print(f"(r,t)=({r},{t}) conv={ik['converged']} pos_err={ik['pos_err_mm']:.3f}mm "
          f"natural_elev={m['elev_deg']:+.2f}deg l5={m['l5_sep']:+.3f} grip={m['g_sep']:+.3f}mm ({m['g_state']})")

print()
print('=== elevation sweep (axis-target IK), anchors x phi ===')
print('r,t,phi | conv iter pos_err z_err | elev | l5_sep grip_sep state | jaw_tan gap pen below_top gates')
for (r, t) in [(7.0, 11.0), (14.6, 13.9), (15.25, 9.0), (11.0, 11.5)]:
    tt = target_tcp_for(r, t)
    psi = math.atan2(tt[1], tt[0])
    u = np.array([math.cos(psi), math.sin(psi), 0.0])
    for phi in np.arange(-10.0, 90.1, 5.0):
        z_t = math.cos(math.radians(phi)) * u + np.array([0, 0, -math.sin(math.radians(phi))])
        x_t = np.array([math.sin(psi), -math.cos(psi), 0.0])
        ik = d323._solve_runtime_ik(tt, d332.HOME_DEG, target_x_axis=x_t, target_z_axis=z_t,
                                    max_iter=600, pos_tol_mm=1.0, axis_tol_deg=3.0)
        q = np.asarray(ik['q_deg']); q[5] = 0
        m = eval_pose(q, tt)
        flag = ' <<<' if (m['g_sep'] >= 0.1 and m['l5_sep'] >= 0.1 and m['gates']
                          and ik['converged'] and m['tcp_err_mm'] <= 5.0) else ''
        print(f"({r:5.2f},{t:5.2f},{phi:+6.1f}) | {str(ik['converged'])[0]} {ik['iterations']:3d} "
              f"{ik['pos_err_mm']:7.3f} {ik['z_axis_err_deg']:6.2f} | {m['elev_deg']:+7.2f} | "
              f"{m['l5_sep']:+8.3f} {m['g_sep']:+8.3f} {m['g_state']:7s} | "
              f"{m['jaw_tan_deg']:6.2f} {m['jaw_gap_mm']:+7.3f} {m['jaw_pen_mm']:6.3f} "
              f"{m['below_top_mm']:7.2f} {str(m['gates'])[0]}{flag}")
    print()

# ---- PART 2: full-mesh validation + q5 (gripper opening) sweep ----
print()
print('=== PART 2: stage-mesh identity + q5 sweep (full gripper_link.stl, raw soup) ===')
grip_full = trimesh.load(REPO / 'local_assets/roarm_m3/urdf/meshes/gripper_link.stl', process=False)
gf_v = np.asarray(grip_full.vertices, dtype=np.float64) * 0.001
gf_f = np.asarray(grip_full.faces, dtype=np.int64)
print('gripper_link.stl raw: vtx', len(gf_v), 'faces', len(gf_f), '(stage recorded 41094/13698)')
print('bounds', gf_v.min(axis=0).round(4), gf_v.max(axis=0).round(4))
g2a_raw = trimesh.load(REPO / 'local_assets/roarm_m3/urdf/meshes/gripper_link_collision_g2a.stl', process=False)
print('g2a raw: vtx', len(g2a_raw.vertices), 'faces', len(g2a_raw.faces),
      'bounds', (np.asarray(g2a_raw.vertices)*0.001).min(axis=0).round(4),
      (np.asarray(g2a_raw.vertices)*0.001).max(axis=0).round(4))
gf_bvh = d332._build_raw_bvh(hppfcl, gf_v, gf_f)

def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def grip_sep_at(q_deg, q5_rad, bvh):
    tcp, l5_pos, l5_rot = d323._fk_runtime_tcp(np.asarray(q_deg))
    g_pos = l5_pos + l5_rot @ G_IN_L5_T
    g_rot = l5_rot @ G_IN_L5_R @ Rz(q5_rad)
    return exact_sep_mm(bvh, g_rot, g_pos)

print()
print('--- validation vs runtime audit: old target (7,11), q5=0, full mesh ---')
tt = target_tcp_for(7.0, 11.0)
ik = d323._solve_runtime_ik(tt, d332.HOME_DEG, max_iter=120, pos_tol_mm=1.0)
q = np.asarray(ik['q_deg']); q[5] = 0
sep, state = grip_sep_at(q, 0.0, gf_bvh)
print(f'offline full-mesh gripper @ q5=0: {sep:+.6f}mm ({state})  [runtime D334: -5.956677mm overlap]')

print()
print('--- q5 sweep at old target (7,11), position-only pose ---')
for q5 in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.571]:
    sep, state = grip_sep_at(q, q5, gf_bvh)
    print(f'q5={q5:5.3f}rad ({math.degrees(q5):5.1f}deg): gripper {sep:+9.3f}mm ({state})')

print()
print('--- q5=1.571 (fully open) across position-only family anchors ---')
for (r, t) in [(7.0, 11.0), (14.6, 13.9), (15.25, 9.0), (11.0, 11.5), (0.0, 11.0), (3.0, 12.0)]:
    tt2 = target_tcp_for(r, t)
    ik2 = d323._solve_runtime_ik(tt2, d332.HOME_DEG, max_iter=120, pos_tol_mm=1.0)
    q2 = np.asarray(ik2['q_deg']); q2[5] = 0
    m2 = eval_pose(q2, tt2)
    sep_o, state_o = grip_sep_at(q2, 1.571, gf_bvh)
    sep_c, state_c = grip_sep_at(q2, 0.0, gf_bvh)
    print(f'(r,t)=({r:5.2f},{t:5.2f}) conv={str(ik2["converged"])[0]} pos_err={ik2["pos_err_mm"]:.3f} | '
          f'l5={m2["l5_sep"]:+7.3f} | grip q5=0: {sep_c:+8.3f} ({state_c}) -> q5=1.571: {sep_o:+8.3f} ({state_o}) | '
          f'jaw_tan={m2["jaw_tan_deg"]:.2f} gap={m2["jaw_gap_mm"]:+.3f} below_top={m2["below_top_mm"]:.1f} gates={str(m2["gates"])[0]}')

print()
print('=== PART 3: table clearance + q5=1.5413 (real max open 88.3deg) check ===')
TABLE_TOP_Z = -0.012117
for (r, t) in [(7.0, 11.0), (14.6, 13.9)]:
    tt3 = target_tcp_for(r, t)
    ik3 = d323._solve_runtime_ik(tt3, d332.HOME_DEG, max_iter=120, pos_tol_mm=1.0)
    q3 = np.asarray(ik3['q_deg']); q3[5] = 0
    tcp, l5_pos, l5_rot = d323._fk_runtime_tcp(q3)
    for q5 in [0.0, 1.5413, 1.571]:
        g_pos = l5_pos + l5_rot @ G_IN_L5_T
        g_rot = l5_rot @ G_IN_L5_R @ Rz(q5)
        gw = (g_rot @ gf_v.T).T + g_pos
        l5w = (l5_rot @ link5_v.T).T + l5_pos
        sep, state = grip_sep_at(q3, q5, gf_bvh)
        print(f'(r,t)=({r},{t}) q5={q5:.4f}: grip_sep={sep:+8.3f} ({state}) | '
              f'grip minz={gw[:,2].min():+.4f} l5 minz={l5w[:,2].min():+.4f} (table_top {TABLE_TOP_Z}) | '
              f'grip clear of table: {gw[:,2].min() - TABLE_TOP_Z:+.4f}m')
