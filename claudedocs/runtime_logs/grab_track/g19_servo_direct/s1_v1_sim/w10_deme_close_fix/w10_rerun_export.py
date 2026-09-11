"""W10 D341(W8 exporter 사본, recording_id·app_id 만 w10) Rerun 관측 아티팩트 — 셀 결과(npz·timeline·JSON) 를 isaaclab python(rerun 0.34.1) 으로 RRD 로 굽고 완결 계약을 검증한다.

usage: ~/miniconda3/envs/isaaclab/bin/python w10_rerun_export.py <cell_dir> [tag=w10]
산출: <cell_dir>/scoop_s1_seed460_<tag>.rrd / .rbl / _inspection.png / _rerun_validation.json  (기존 파일 있으면 거부)
정본은 npz/JSON. Rerun 의 Float32 사본은 검수용이며 게이트에 되먹이지 않는다(D341).
"""
import hashlib, json, math, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[6]; sys.path.insert(0, str(REPO))
cell_dir = Path(sys.argv[1]).resolve(); tag = sys.argv[2] if len(sys.argv) > 2 else "w10"
stem = cell_dir / f"scoop_s1_seed460_{tag}"
res = json.load(open(cell_dir / "scoop_s1_seed460.json")); z = np.load(cell_dir / "scoop_s1_seed460.npz")
tl = json.load(open(cell_dir / "timeline_seed460.json"))["rows"]
pile = np.load(res["params"].get("_pile_path", "") or next(k for k in res["inputs_sha16"] if k.endswith(".npz")), allow_pickle=True)

import rerun as rr
from roarm_rl.rerun_contract import RERUN_CONTRACT_VERSION, validate_rerun_artifact
import roarm_rl.viz_debug as viz_debug
assert str(rr.__version__) == "0.34.1" == RERUN_CONTRACT_VERSION, (rr.__version__, RERUN_CONTRACT_VERSION)
interp_bin = str(Path(sys.executable).resolve().parent); os.environ["PATH"] = interp_bin + os.pathsep + os.environ.get("PATH", "")
rrd, rbl, png, valp = stem.with_suffix(".rrd"), stem.with_suffix(".rbl"), stem.with_name(stem.name + "_inspection.png"), stem.with_name(stem.name + "_rerun_validation.json")
for p in (rrd, rbl, png, valp):
    if p.exists():
        raise FileExistsError(p)

def open_box(b):
    x0, x1 = b[0]; y0, y1 = b[1]; z0, z1 = b[2]
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float32)
    T = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.uint32)
    return V, T

box = np.asarray(z["box_bounds_m"], float); Vb, Tb = open_box(box)
t = np.asarray(z["frame_t_s"], float); ph = np.asarray(z["frame_phase"], int); nF, nD = z["nodes_F_m"], z["nodes_D_m"]
assert len(t) == len(tl) == len(nF) == len(nD) > 0
assert np.allclose(t, [r["sim_t"] for r in tl], rtol=0, atol=1e-10)
cp, cf, cfr = z["contact_point_m"], z["contact_force_N"], z["contact_frame"]
sp, sr = np.asarray(z["sphere_positions_m"], float), np.asarray(z["sphere_radii_m"], np.float32)
k = int(res["heightmap"]["spheres_per_particle"]); in_cav = np.repeat(np.asarray(z["in_cavity"], bool), k); carried = np.repeat(np.asarray(z["carried"], bool), k)
col_post = np.where(in_cav[:, None], [[235, 140, 40, 230]], np.where(carried[:, None], [[120, 200, 240, 200]], [[190, 180, 150, 150]])).astype(np.uint8)
pre_pos, pre_rad = np.asarray(pile["positions_m"], float), np.asarray(pile["radii_m"], np.float32)
hm_pre, hm_post = np.asarray(z["heightmap_pre_m"], float), np.asarray(z["heightmap_m"], float)
cell = res["heightmap"]["cell_m"]; rows, cols = hm_pre.shape
xs = box[0, 0] + (np.arange(cols) + 0.5) * cell; ys = box[1, 0] + (np.arange(rows) + 0.5) * cell; X, Y = np.meshgrid(xs, ys)
def hm_points(h):
    v = np.clip(h / max(float(hm_pre.max()), 1e-6), 0, 1)
    c = np.stack([(255 * v), (200 * (1 - v) + 40 * v), 255 * (1 - v), np.full_like(v, 255)], -1).astype(np.uint8)
    return np.stack([X, Y, h], -1).reshape(-1, 3), c.reshape(-1, 4)
cr = res["crater"]; cx, cy = np.array(cr["center_xy_mm"]) / 1000; r_max = cr["params"]["r_max_mm"] / 1000
z_c = float(hm_post[min(rows - 1, max(0, int((cy - box[1, 0]) / cell))), min(cols - 1, max(0, int((cx - box[0, 0]) / cell)))])
az_dir = {"+x": (1, 0), "+y": (0, 1), "-x": (-1, 0), "-y": (0, -1)}
ray_o = np.array([[cx, cy, z_c + 0.002]] * 4); ray_v = np.array([[r_max * d[0], r_max * d[1], 0.0] for d in az_dir.values()])
fit_pts, fit_lab = [], []
for name, azd in cr["azimuths"].items():
    if azd.get("angle_deg") is None:
        continue
    d = az_dir[name]
    for i in azd["fit_bins"]:
        r_ = azd["r_mm"][i] / 1000; fit_pts.append([cx + d[0] * r_, cy + d[1] * r_, azd["h_post_ring_mm"][i] / 1000]); fit_lab.append(f"{name} {azd['angle_deg']:.1f}deg")
pre_pos_pts, pre_col = hm_points(hm_pre); post_pos_pts, post_col = hm_points(hm_post)

points = [
    {"entity_path": "geometry/pile/spheres_pre", "positions_m": pre_pos, "radii": pre_rad, "colors": np.tile(np.array([[150, 150, 160, 90]], np.uint8), (len(pre_pos), 1)),
     "coordinate_frame": "world_m", "sequence": {"frame": 0, "phase": 0}, "duration": {"sim_time_s": 0.0}},
    {"entity_path": "geometry/pile/spheres_post", "positions_m": sp, "radii": sr, "colors": col_post,
     "coordinate_frame": "world_m", "sequence": {"frame": len(t) - 1, "phase": int(ph[-1])}, "duration": {"sim_time_s": float(t[-1])}},
    # 정적 사본: 헤드리스 스크린샷의 기본 시간 커서(+0.9 s)에서도 최종 더미·포획·툴 최종 자세가 보이도록(1차 w8 검수 한계)
    {"entity_path": "geometry/pile/spheres_final_static", "positions_m": sp, "radii": sr, "colors": col_post, "coordinate_frame": "world_m", "static": True},
    {"entity_path": "geometry/tool/final_static", "positions_m": np.concatenate([nF[-1], nD[-1]]), "radii": 0.0008, "colors": [30, 200, 120], "coordinate_frame": "world_m", "static": True},
    {"entity_path": "geometry/heightmap/pre", "positions_m": pre_pos_pts, "radii": np.full(len(pre_pos_pts), 0.0015, np.float32), "colors": pre_col, "coordinate_frame": "world_m", "static": True},
    {"entity_path": "geometry/heightmap/post", "positions_m": post_pos_pts, "radii": np.full(len(post_pos_pts), 0.0015, np.float32), "colors": post_col, "coordinate_frame": "world_m", "static": True},
]
if fit_pts:
    points.append({"entity_path": "geometry/crater/fit_rings", "positions_m": np.asarray(fit_pts), "radii": np.full(len(fit_pts), 0.003, np.float32),
                   "colors": np.tile(np.array([[255, 40, 40, 255]], np.uint8), (len(fit_pts), 1)), "labels": fit_lab, "coordinate_frame": "world_m", "static": True})
arrows = [{"entity_path": "geometry/crater/azimuth_rays", "origins_m": ray_o, "vectors_m": ray_v, "radii": np.full(4, 0.0012, np.float32),
           "colors": np.tile(np.array([[255, 255, 255, 255]], np.uint8), (4, 1)), "labels": [f"{n} {(cr['azimuths'][n].get('angle_deg') or float('nan')):.1f}deg" for n in az_dir], "coordinate_frame": "world_m", "static": True}]
scalars, events = [], []
keys = ("z_lip_mm", "q_deg", "lipF_N", "M_hinge_res_Nm", "v_particle_max", "n_door", "n_fixed", "Fz_fixed_up_N", "max_single_contact_N")
step = 1  # D341: 툴·접촉·스칼라는 실행한 모든 sync. 입자 위치는 별도 원자료의 0.05 s 간격.
for i in range(len(t)):
    row = tl[i] if i < len(tl) else tl[-1]
    tm = {"sequence": {"frame": i, "phase": int(ph[i])}, "duration": {"sim_time_s": float(t[i])}}
    for kk in keys:
        scalars.append({"entity_path": f"metrics/{kk}", "value": float(row.get(kk, 0.0)), **tm})
    if i % step == 0 or i == len(t) - 1:
        points.append({"entity_path": "geometry/tool/fixed_nodes", "positions_m": nF[i], "radii": 0.0008, "colors": [60, 170, 90], "coordinate_frame": "world_m", **tm})
        points.append({"entity_path": "geometry/tool/door_nodes", "positions_m": nD[i], "radii": 0.0008, "colors": [60, 110, 220], "coordinate_frame": "world_m", **tm})
        m = cfr == i
        # 빈 접촉도 기록해 이전 프레임 힘이 계속 남아 보이는 것을 막는다.
        points.append({"entity_path": "contacts/points", "positions_m": cp[m], "radii": 0.0010, "colors": [220, 60, 60], "coordinate_frame": "world_m", **tm})
        arrows.append({"entity_path": "contacts/forces", "origins_m": cp[m], "vectors_m": cf[m] * 0.005, "colors": [220, 60, 60], "coordinate_frame": "world_m", **tm})   # 1 N = 5 mm 표시 배율
for st in res["door"]["stops"]:
    i = int(np.searchsorted(t, st.get("sim_t", t[-1])))
    events.append({"entity_path": "events/decision", "text": f"DOOR_STOP phase={st['phase']} q={st['q_deg']} reason={st['reason']} M={st.get('M_hinge_res_Nm')}", "level": "INFO",
                   "sequence": {"frame": min(i, len(t) - 1), "phase": int(ph[min(i, len(t) - 1)])}, "duration": {"sim_time_s": float(st.get("sim_t", t[-1]))}})
cap = res["capture"]
events.append({"entity_path": "events/decision", "text": f"CAPTURE n_in_cavity={cap['n_in_cavity']} mass_g={cap['mass_g']} fill_vs_bulk={cap['fill_vs_bulk']} carried_z={cap['n_carried_z']} diverged={res['diverged']}",
               "level": "INFO", "sequence": {"frame": len(t) - 1, "phase": int(ph[-1])}, "duration": {"sim_time_s": float(t[-1])}})
events.append({"entity_path": "events/decision", "text": "CRATER_ANGLE_deg " + " ".join(f"{n}={(v.get('angle_deg') if v.get('angle_deg') is None else round(v['angle_deg'], 1))}" for n, v in cr["azimuths"].items())
               + f" center_mm={[round(v, 1) for v in cr['center_xy_mm']]} removed_cm3={cr['removed_volume_cm3']:.2f} (report only, not a verdict)",
               "level": "INFO", "sequence": {"frame": len(t) - 1, "phase": int(ph[-1])}, "duration": {"sim_time_s": float(t[-1])}})
meshes = [{"entity_path": "geometry/container", "vertices_m": Vb, "triangles": Tb, "color_rgba": [80, 145, 210, 38], "coordinate_frame": "world_m", "static": True}]

# 기존 render timeline의 실제 입자를 전개한다. 보간·새 물리 실행 없음.
from scipy.spatial.transform import Rotation
rt_path = Path(res["render_timeline"]["path"])
rt = np.load(rt_path)
tpl = json.loads(str(pile["clump_template_json"].item()))
offsets = np.asarray(tpl["offsets_m"], float)
particle_radii = np.tile(np.asarray(tpl["sphere_radii_m"], np.float32), len(rt["clump_pos_m"][0]))
for ri, ti in enumerate(rt["t_s"]):
    rots = Rotation.from_quat(rt["clump_quat_xyzw"][ri]).as_matrix()
    expanded = (rt["clump_pos_m"][ri, :, None, :] + np.einsum("nij,kj->nki", rots, offsets)).reshape(-1, 3)
    points.append({"entity_path": "geometry/pile/animated", "positions_m": expanded, "radii": particle_radii,
                   "colors": col_post, "coordinate_frame": "world_m", "sequence": {"frame": int(rt["timeline_frame"][ri]), "phase": int(rt["phase"][ri])},
                   "duration": {"sim_time_s": float(ti)}})

# 발산/힘 이벤트가 있으면 실제 링버퍼와 구–구 접촉 덤프도 보존한다.
extra_entities = set()
extra_components = {}
event_ring_counts = {}
for event_path in sorted(cell_dir.glob("diverge_event*_seed460.json")):
    event_npz = event_path.with_suffix(".npz")
    if not event_npz.exists():
        continue
    ev = json.loads(event_path.read_text()); ez = np.load(event_npz)
    prefix = "diagnostic/" + event_path.stem
    culprit = int(ez["culprit"])
    for ei, et in enumerate(ez["t_s"]):
        near = ez["near_frame"] == ei
        pos = ez["near_pos_m"][near]; quat = ez["near_quat_xyzw"][near]
        expanded = (pos[:,None,:] + np.einsum("nij,kj->nki", Rotation.from_quat(quat).as_matrix(), offsets)).reshape(-1,3)
        colors = np.repeat(np.where((ez["near_ids"][near] == culprit)[:,None], [[255,50,30]], [[170,170,190]]), len(offsets), axis=0)
        tm = {"sequence": {"frame": int(np.argmin(np.abs(t-et)))}, "duration": {"sim_time_s": float(et)}}
        points.append({"entity_path": prefix+"/near_spheres", "positions_m": expanded,
                       "radii": np.tile(tpl["sphere_radii_m"],len(pos)), "colors": colors, "coordinate_frame": "world_m", **tm})
        is_culprit = ez["near_ids"][near] == culprit
        arrows.append({"entity_path": prefix+"/culprit_velocity", "origins_m": pos[is_culprit], "vectors_m": ez["near_vel_m_s"][near][is_culprit]*0.002,
                       "colors": [255,50,30], "coordinate_frame": "world_m", **tm})
    pairs = ev.get("contact_detail",{}).get("pairs",[])
    if pairs:
        tm = {"sequence": {"frame": int(np.argmin(np.abs(t-ev["trigger"]["sim_t"])))}, "duration": {"sim_time_s": float(ev["trigger"]["sim_t"])}}
        arrows.append({"entity_path": prefix+"/contact_forces", "origins_m": np.asarray([p["point_mm"] for p in pairs])*0.001,
                       "vectors_m": np.asarray([p["force"] for p in pairs])*0.005, "colors": [240,80,30], "coordinate_frame": "world_m", **tm})
        extra_entities.add("/"+prefix+"/contact_forces")
        extra_components["/"+prefix+"/contact_forces"] = ["Arrows3D:origins", "Arrows3D:vectors"]
    extra_entities |= {"/"+prefix+"/near_spheres", "/"+prefix+"/culprit_velocity"}
    extra_components["/"+prefix+"/near_spheres"] = ["Points3D:positions", "Points3D:radii"]
    event_ring_counts[event_path.name] = {"frames": len(ez["t_s"]), "contact_pairs_at_trigger": len(pairs), "culprit": culprit}

# 첫 폐합 정지(발산이면 최대속도 sync)의 실제 메시와 명목 q 프레임을 함께 보여준다.
decision_i = int(np.argmax([r["v_particle_max"] for r in tl])) if res["diverged"] else min(len(t)-1, int(np.searchsorted(t, res["door"]["stops"][0]["sim_t"])))
import sim_deme_scoop_s1 as sim_s1
q_open = res["params"]["door_open_servo_deg"] - res["params"]["servo_zero_offset_deg"]
fm, dm, _, _, hrel, _, _ = sim_s1.load_tool(res["params"], q_open)
fixed_origin = np.asarray(nF[decision_i], float).mean(0) - np.asarray(fm.vertices).mean(0)
target_origin = fixed_origin + hrel
dv = np.asarray(dm.vertices, float)
actual_nodes = np.asarray(nD[decision_i], float)
u, _, vt = np.linalg.svd((dv-dv.mean(0)).T @ (actual_nodes-actual_nodes.mean(0)))
correction = np.eye(3); correction[-1,-1] = np.linalg.det(vt.T @ u.T)
actual_rot_delta = vt.T @ correction @ u.T
actual_origin = actual_nodes.mean(0) - actual_rot_delta @ dv.mean(0)
actual_rot = actual_rot_delta @ sim_s1.R_W @ sim_s1.roty(q_open)
target_rot = sim_s1.R_W @ sim_s1.roty(tl[decision_i]["q_deg"])
frames = [viz_debug.frame_from_axes(name, origin, x_axis=rot[:,0], z_axis=rot[:,2], role=role, label=label)
          for name, origin, rot, role, label in [
              ("nominal_door", target_origin, target_rot, "target", "nominal q / hinge"),
              ("actual_door", actual_origin, actual_rot, "actual", "actual mesh / hinge")]]
frame_png = stem.with_name(stem.name + "_decision_frames.png")
viz_debug.snapshot_frame_plot(frame_png, frames, title=f"W10 {cell_dir.name}: decision frame {decision_i}",
                             annotations=["Nominal q is the Python state; actual pose is fitted from saved mesh nodes.", "Markers are observability only; no change to the physics verdict."])
for part, nodes, color in (("fixed",nF[decision_i],[60,170,90]),("door",nD[decision_i],[60,110,220])):
    points.append({"entity_path": f"decision/{part}", "positions_m": nodes, "radii": 0.0008, "colors": color, "coordinate_frame": "world_m", "static": True})
mask = cfr == decision_i
points.append({"entity_path": "decision/contacts", "positions_m": cp[mask], "radii": 0.001, "colors": [230,50,50], "coordinate_frame": "world_m", "static": True})
arrows.append({"entity_path": "decision/forces", "origins_m": cp[mask], "vectors_m": cf[mask]*0.005, "colors": [230,50,50], "coordinate_frame": "world_m", "static": True})
for name, origin, rot, color in (("target_axes",target_origin,target_rot,[240,50,30]),("actual_axes",actual_origin,actual_rot,[30,170,240])):
    arrows.append({"entity_path": f"decision/{name}", "origins_m": np.repeat(origin[None],3,axis=0), "vectors_m": rot.T*0.02,
                   "colors": color, "coordinate_frame": "world_m", "static": True})

def bp(mode):
    import rerun.blueprint as rrb
    assert mode == "w10_scoop"
    return rrb.Blueprint(rrb.Vertical(
        rrb.Horizontal(rrb.Spatial3DView(origin="/", contents=["/geometry/pile/animated", "/geometry/tool/fixed_nodes", "/geometry/tool/door_nodes", "/contacts/**", "/diagnostic/**"], name="timeline: particles + tool + contacts"),
                       rrb.Spatial3DView(origin="/", contents=["/geometry/pile/spheres_final_static", "/geometry/tool/final_static"], name=f"final capture: {cap['mass_g']:.4f} g"),
                       rrb.Spatial3DView(origin="/", contents=["/decision/**", "/frames/**"], name=f"close decision t={t[decision_i]:.5f}s: nominal / actual"), column_shares=[0.4, 0.35, 0.25]),
        rrb.Horizontal(rrb.TimeSeriesView(origin="/metrics", contents=["/metrics/lipF_N", "/metrics/q_deg", "/metrics/z_lip_mm"], name="Float64 metrics (lipF_N, q_deg, z_lip_mm)"),
                       rrb.Spatial3DView(origin="/", contents=["/geometry/heightmap/post", "/geometry/crater/**"], name="post heightmap + crater fit"),
                       rrb.TextLogView(origin="/events", contents="/events/**", name="decision events"), column_shares=[0.4, 0.3, 0.3]), row_shares=[0.65, 0.35]),
        rrb.TimePanel(timeline="sim_time_s", play_state="paused"),
        auto_layout=False, auto_views=False, collapse_panels=True)
orig = viz_debug.build_rerun_blueprint
try:
    viz_debug.build_rerun_blueprint = bp
    status = viz_debug.log_rerun(rrd, coordinate_frames=[{"frame": "world_m", "parent_frame": "tf#/", "entity_path": "coordinate_frames/world_m"}],
                                 frames=frames, meshes=meshes, points=points, arrows=arrows, scalar_trace=scalars, events=events,
                                 recording_metadata={"artifact": "DEME_SCOOP_S1_W10_RERUN_V2", "cell_dir": str(cell_dir), "result_json_sha256": hashlib.sha256((cell_dir / "scoop_s1_seed460.json").read_bytes()).hexdigest(),
                                                     "npz_sha256": hashlib.sha256((cell_dir / "scoop_s1_seed460.npz").read_bytes()).hexdigest(), "n_frames": int(len(t)), "tool_node_frame_stride": step,
                                                     "decision_frame": decision_i, "particle_frames": len(rt["t_s"]), "contact_frames_logged": len(t), "particle_color_semantics": "final capture membership, constant through replay",
                                                     "event_rings": event_ring_counts, "force_arrow_scale_m_per_N": 0.005, "culprit_velocity_arrow_scale_s": 0.002,
                                                     "scientific_authority": "scoop_s1_seed460.json / .npz (Float64); Rerun copies are Float32 observability only"},
                                 recording_id=f"w10_scoop_{cell_dir.name}_{tag}", blueprint_path=rbl, blueprint_mode="w10_scoop", live_viewer=False, app_id="roarm_w10_scoop")
finally:
    viz_debug.build_rerun_blueprint = orig
if not status.get("ok"):
    raise SystemExit(f"log_rerun failed: {status}")
ents = {"/metadata/run", "/coordinate_frames/world_m", "/geometry/container", "/metadata/meshes/geometry__container", "/geometry/pile/spheres_pre", "/geometry/pile/spheres_post", "/geometry/pile/spheres_final_static", "/geometry/tool/final_static",
        "/geometry/heightmap/pre", "/geometry/heightmap/post", "/geometry/crater/azimuth_rays", "/geometry/tool/fixed_nodes", "/geometry/tool/door_nodes", "/contacts/points", "/contacts/forces", "/events/decision"} | {f"/metrics/{kk}" for kk in keys}
if fit_pts:
    ents.add("/geometry/crater/fit_rings")
ents |= {"/geometry/pile/animated", "/decision/fixed", "/decision/door", "/decision/contacts", "/decision/forces", "/decision/target_axes", "/decision/actual_axes"}
ents |= {f"/frames/{name}{suffix}" for name in ("nominal_door", "actual_door") for suffix in ("", "/origin")}
ents |= extra_entities
comp = {"/geometry/pile/spheres_post": ["Points3D:colors", "Points3D:positions", "Points3D:radii"], "/geometry/tool/door_nodes": ["Points3D:positions"], "/contacts/forces": ["Arrows3D:origins", "Arrows3D:vectors"],
        "/metrics/lipF_N": ["Scalars:scalars"], "/metrics/q_deg": ["Scalars:scalars"], "/events/decision": ["TextLog:level", "TextLog:text"], "/geometry/container": ["Mesh3D:triangle_indices", "Mesh3D:vertex_positions"]}
comp.update({"/geometry/pile/animated": ["Points3D:colors", "Points3D:positions", "Points3D:radii"],
             "/decision/target_axes": ["Arrows3D:origins", "Arrows3D:vectors"], "/decision/actual_axes": ["Arrows3D:origins", "Arrows3D:vectors"]})
comp.update(extra_components)
val = validate_rerun_artifact(rrd, expected_entity_paths=sorted(ents), expected_timeline_names=["blueprint", "log_time", "frame", "phase", "sim_time_s"], exact_entity_paths=sorted(ents),
                              exact_timeline_names=["blueprint", "log_time", "frame", "phase", "sim_time_s"], expected_entity_components=comp, blueprint_path=rbl, screenshot_path=png,
                              screenshot_window_size="3200x1800", cli_path=Path(interp_bin) / "rerun", expected_version="0.34.1", timeout_s=300.0)
val["log_status_summary"] = {kk: status.get(kk) for kk in ("ok", "bytes", "rerun_sdk_version", "sink_attached_before_logging", "sink_finalized", "flush_ok", "blueprint_status")}
val["source"] = {"cell_dir": str(cell_dir), "result_json_sha256": hashlib.sha256((cell_dir / "scoop_s1_seed460.json").read_bytes()).hexdigest()}
val["coverage"] = {"source_syncs": len(t), "tool_syncs_logged": len(t), "contact_syncs_logged": len(t), "particle_frames_logged": len(rt["t_s"]),
                   "decision_frame": decision_i, "decision_frames_png": str(frame_png), "raw_arrays_authority": True}
from verify_rrd_coverage import verify as verify_coverage
coverage = verify_coverage(cell_dir, rrd)
coverage_path = stem.with_name(stem.name + "_coverage.json")
coverage_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2))
val["coverage_readback"] = {"pass": coverage["pass"], "path": str(coverage_path)}
val["pass"] = bool(val.get("pass") and coverage["pass"])
json.dump(val, open(valp, "w"), ensure_ascii=False, indent=2, default=str)
print(("RERUN_EXPORT_OK" if val.get("pass") else "RERUN_CONTRACT_FAIL"), rrd, png, valp, f"{rrd.stat().st_size/1e6:.1f} MB")
