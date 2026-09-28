"""W13 Rerun 관측 아티팩트 (D341 완결 계약) — 원시 npz/JSON 을 RRD/RBL 로 굽고 계약을 검증한다.

usage: ~/miniconda3/envs/isaaclab/bin/python w13_rerun_export.py <attempt_dir>
산출: <attempt_dir>/rerun/{w13.rrd, w13.rbl, w13_inspection.png, w13_rerun_validation.json}
기존 파일이 있으면 거부한다(덮어쓰기 금지).

정본은 npz/JSON 이다. Rerun 의 Float32 사본은 검수층이며 과학 게이트에 되먹이지 않는다(D341).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raw_row_identity as RRI                                       # noqa: E402

MAIN = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(MAIN))
att = Path(sys.argv[1]).resolve()
res = json.load(open(att / "w13_cycle_seed460.json"))
z = np.load(att / "w13_cycle_seed460.npz", allow_pickle=True)

import rerun as rr                                                    # noqa: E402
from roarm_rl.rerun_contract import RERUN_CONTRACT_VERSION, validate_rerun_artifact   # noqa: E402
import roarm_rl.viz_debug as viz_debug                                # noqa: E402

assert str(rr.__version__) == "0.34.1" == RERUN_CONTRACT_VERSION, (rr.__version__, RERUN_CONTRACT_VERSION)
interp = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = interp + os.pathsep + os.environ.get("PATH", "")
outd = att / "rerun"
outd.mkdir(exist_ok=True)
rrd, rbl = outd / "w13.rrd", outd / "w13.rbl"
png, valp = outd / "w13_inspection.png", outd / "w13_rerun_validation.json"
for p in (rrd, rbl, png, valp):
    if p.exists():
        raise FileExistsError(p)

PHASES = json.loads(str(z["metadata_json"]))["phase_order"]
INV = [str(v) for v in np.asarray(z["inventory_labels"])]
t = np.asarray(z["sync_t_s"], float)
ph = np.asarray(z["sync_phase_code"], int)
pf_t = np.asarray(z["particle_frame_t_s"], float)
pf_s = np.asarray(z["particle_frame_sync_index"], int)
pos = z["particle_pos_m"]
quat = z["particle_quat_xyzw"]
inv = z["inventory_code"]
nF, nD = z["nodes_F_m"], z["nodes_D_m"]
cp, cf, ci = z["contact_point_m"], z["contact_force_N"], np.asarray(z["contact_sync_index"], int)
cmesh = np.asarray(z["contact_mesh"], int)          # 0 fixed / 1 door / 2 tray(source wall)
pile = np.load([k for k in res["inputs_sha256"] if k.endswith(".npz")][0], allow_pickle=True)
tpl = json.loads(str(pile["clump_template_json"]))
offs = np.asarray(tpl["offsets_m"], float)
srad = np.asarray(tpl["sphere_radii_m"], np.float32)
from scipy.spatial.transform import Rotation                          # noqa: E402

COLOR = np.array([[160, 150, 130, 150], [235, 140, 40, 235], [60, 170, 90, 235],
                  [200, 60, 60, 220], [120, 200, 240, 220], [190, 190, 190, 140]], np.uint8)


def tri_mesh(key_v, key_f, color):
    return {"entity_path": f"geometry/{key_v.split('_')[0]}", "vertices_m": np.asarray(z[key_v], np.float32),
            "triangles": np.asarray(z[key_f], np.uint32), "color_rgba": color,
            "coordinate_frame": "world_m", "static": True}


# 접촉 role 표는 **원시 metadata 의 mesh_role_to_id 에서 읽는다**(이름 추측 금지, ERRATUM_02 ②).
_META = json.loads(str(z["metadata_json"]))
_ROLE2ID = ((_META.get("contact_semantics") or {}).get("mesh_role_to_id")
            or {"fixed": 0, "door": 1, "tray": 2})
_ROLE_COLOR = {"fixed": [60, 200, 120], "door": [60, 130, 240],
               "tray": [230, 160, 40], "bin": [230, 90, 160]}
_ROLE_ENTITY = {"fixed": "tool_fixed", "door": "door", "tray": "tray_wall", "bin": "bin"}
# **원시 role 이름을 튜플에 보존**한다 — coverage 의 role 표를 원시 선언과 같은 키로 비교하려면
# entity 이름(tool_fixed …)만으로는 안 되고 원시 키(fixed …)가 필요하다(`RERUN_SOURCE_REVIEW_01` ②).
CONTACT_ROLES = [(int(v), _ROLE_ENTITY.get(k, k), _ROLE_COLOR.get(k, [200, 200, 200]), str(k))
                 for k, v in sorted(_ROLE2ID.items(), key=lambda kv: (kv[1] is None, kv[1]))
                 if v is not None]
n_logged_contact_rows = {nm_: 0 for _mid, nm_, _c, _rk in CONTACT_ROLES}
_unmapped = sorted(set(int(v) for v in np.unique(cmesh)) - {mid for mid, _n, _c, _rk in CONTACT_ROLES})

points, arrows, scalars, events, meshes = [], [], [], [], [
    tri_mesh("tray_vertices_m", "tray_faces", [80, 200, 120, 60]),
    tri_mesh("bin_vertices_m", "bin_faces", [220, 90, 90, 70])]

# ── 저장 행 정체성 (`RAW_SCHEMA_REQUIRED_ERRATUM_03` + `RERUN_SOURCE_REVIEW_01`) ─────
# `particle_frame_sync_index` 는 **비감소**이지 유일하지 않다 — 결정 스냅샷이 **같은 완료 sync ·
# 같은 source time** 에 의도적으로 행을 하나 더 붙일 수 있다. 그러면 dense sync 타임라인
# `frame=pf_s[i]` 과 `sim_time_s=pf_t[i]` 는 두 행에 **같은 시간 좌표**를 주므로, 두 행이
# datastore 에서 살아남았는지 아니면 같은 시각 마지막-쓰기만 남았는지 구분할 수 없다.
# → 입자 기록에 **명시적으로 단조 증가하는 `particle_frame_row=i` sequence 타임라인**을 준다.
#   원시 sync/time 은 지우지 않고 **데이터로** 보존한다. dense sync 타임라인은 dense 스칼라·
#   접촉용으로 그대로 둔다. 같은 sync 행을 조용히 합치지 않고, 스테핑도 바꾸지 않는다.
pf_rows_logged = []
for i in range(len(pf_t)):
    si = int(pf_s[i]) if pf_s[i] >= 0 else 0
    tm = {"sequence": {"particle_frame_row": int(i), "frame": si, "phase": int(ph[si])},
          "duration": {"sim_time_s": float(pf_t[i])}}
    R = Rotation.from_quat(np.asarray(quat[i], float)).as_matrix()
    sp = (np.asarray(pos[i], float)[:, None, :] + np.einsum("nij,kj->nki", R, offs)).reshape(-1, 3)
    points.append({"entity_path": "geometry/particles", "positions_m": sp,
                   "radii": np.tile(srad, len(pos[i])), "colors": np.repeat(COLOR[np.asarray(inv[i], int)], len(offs), 0),
                   "coordinate_frame": "world_m", **tm})
    # 원시 sync/time/phase 를 행마다 **데이터로** 남긴다 → 같은 sync 행도 각자 복원 가능하다.
    scalars.append({"entity_path": "metrics/particle_frame_source_sync_index",
                    "value": float(pf_s[i]), **tm})
    scalars.append({"entity_path": "metrics/particle_frame_source_time_s",
                    "value": float(pf_t[i]), **tm})
    scalars.append({"entity_path": "metrics/particle_frame_source_phase_code",
                    "value": float(ph[si]), **tm})
    pf_rows_logged.append(int(i))

KEYS = [k[len("scalar_"):] for k in z.files if k.startswith("scalar_")]
for i in range(len(t)):
    tm = {"sequence": {"frame": i, "phase": int(ph[i])}, "duration": {"sim_time_s": float(t[i])}}
    for kk in KEYS:
        scalars.append({"entity_path": f"metrics/{kk}", "value": float(z[f"scalar_{kk}"][i]), **tm})
    scalars.append({"entity_path": "metrics/door_actual_deg", "value": float(z["door_actual_deg"][i]), **tm})
    scalars.append({"entity_path": "metrics/door_target_deg", "value": float(z["door_target_deg"][i]), **tm})
    points.append({"entity_path": "geometry/tool/fixed_nodes", "positions_m": nF[i], "radii": 0.0008,
                   "colors": [60, 170, 90], "coordinate_frame": "world_m", **tm})
    points.append({"entity_path": "geometry/tool/door_nodes", "positions_m": nD[i], "radii": 0.0008,
                   "colors": [60, 110, 220], "coordinate_frame": "world_m", **tm})
    m = ci == i
    for mid, nm_, col_, _rk in CONTACT_ROLES:
        mm = m & (cmesh == mid)
        n_logged_contact_rows[nm_] += int(mm.sum())      # **실제로 기록한 행 수**를 센다
        points.append({"entity_path": f"contacts/{nm_}/points", "positions_m": cp[mm], "radii": 0.0010,
                       "colors": col_, "coordinate_frame": "world_m", **tm})
        arrows.append({"entity_path": f"contacts/{nm_}/forces", "origins_m": cp[mm], "vectors_m": cf[mm] * 0.005,
                       "colors": col_, "coordinate_frame": "world_m", **tm})
    arrows.append({"entity_path": "pose/tool_target_vs_actual",
                   "origins_m": np.asarray(z["tool_target_pos_m"][i], float)[None],
                   "vectors_m": (np.asarray(z["tool_pos_m"][i], float) -
                                 np.asarray(z["tool_target_pos_m"][i], float))[None],
                   "colors": [255, 200, 0], "coordinate_frame": "world_m", **tm})

for k, tag in enumerate(np.asarray(z["decision_tags"])):
    si = int(z["decision_sync_index"][k])
    fi = int(z["decision_particle_frame_index"][k])
    c = np.bincount(np.asarray(inv[fi], int), minlength=len(INV))
    events.append({"entity_path": "events/decision",
                   "text": f"DECISION {tag} sync={si} frame={fi} " +
                           " ".join(f"{INV[j]}={int(c[j])}" for j in range(len(INV))),
                   # ERRATUM_03: 결정 참조는 **보존된 행**을 가리켜야 한다 → 정본 행 좌표도 함께 준다.
                   "level": "INFO",
                   "sequence": {"particle_frame_row": fi, "frame": si, "phase": int(ph[si])},
                   "duration": {"sim_time_s": float(t[si])}})
for st in res["door"]["stops"]:
    si = int(st.get("sync_index", 0))
    events.append({"entity_path": "events/decision",
                   "text": f"DOOR_STOP {st['subphase']} q={st['q_actual_deg']} reason={st['reason']} "
                           f"M={st.get('M_hinge_rel_Nm')}",
                   # ERRATUM_03: 결정 참조는 **보존된 행**을 가리켜야 한다 → 정본 행 좌표도 함께 준다.
                   "level": "INFO",
                   "sequence": {"particle_frame_row": fi, "frame": si, "phase": int(ph[si])},
                   "duration": {"sim_time_s": float(t[si])}})
events.append({"entity_path": "events/decision",
               "text": f"DELIVERY definite={res['delivery']['definite_delivered_n']} "
                       f"possible={res['delivery']['possible_delivered_n']} "
                       f"diverged={res['diverged']} stopped_early={res.get('stopped_early_after_phase')} "
                       "(raw arrays are authority; Rerun Float32 is inspection only)",
               "level": "INFO", "sequence": {"frame": len(t) - 1, "phase": int(ph[-1])},
               "duration": {"sim_time_s": float(t[-1])}})
# ── rev11: bridge 사전 인증 **결정 시점 스냅샷** (SPEC 요구) ────────────────────
BRIDGE = res.get("bridge_clearance") or []
if BRIDGE:
    bc = BRIDGE[0]
    pcert = bc.get("planned_certificate") or {}
    gw = pcert.get("global_worst") or {}
    bsync = int(bc.get("sync_index", 0))
    btm = {"sequence": {"frame": bsync, "phase": int(ph[bsync])}, "duration": {"sim_time_s": float(t[bsync])}}
    # 스키마가 fixed/door 로 분리됐다 — 옛 단일 키를 읽지 않는다.
    bridge_entities_logged = set()
    bridge_omitted = {}
    ptg = np.asarray(z["bridge_planned_fixed_pos_m"], float)
    ptg_door = np.asarray(z["bridge_planned_door_pos_m"], float)
    pseg = np.asarray(z["bridge_planned_target_segment"])
    pcur = np.asarray(z["tool_pos_m"][bsync], float)
    # ① 실제 현재 포즈
    points.append({"entity_path": "bridge/current_pose", "positions_m": pcur[None],
                   "radii": np.array([0.004], np.float32), "colors": np.array([[255, 255, 255, 255]], np.uint8),
                   "labels": [f"actual lip at decision (sync {bsync}, t={t[bsync]:.6f}s)"],
                   "coordinate_frame": "world_m", **btm})
    # ② 제안 bridge 목표(정렬 종점 = owner_pose(q_res))
    if len(ptg):
        n_al = int(bc.get("n_align_sync_targets") or 0)
        tgt = ptg[max(0, n_al - 1)]
        bridge_entities_logged.add("bridge/proposed_target")
        points.append({"entity_path": "bridge/proposed_target", "positions_m": np.asarray(tgt, float)[None],
                       "radii": np.array([0.005], np.float32),
                       "colors": np.array([[0, 255, 255, 255]], np.uint8),
                       "labels": [f"proposed align target owner_pose(q_res) {np.round(tgt*1000,3).tolist()} mm"],
                       "coordinate_frame": "world_m", **btm})
        # ③ 계획 경로 두 구간을 따로 그린다(정렬 / 관절)
        for seg, col in (("align_sync_targets", [255, 140, 0, 230]),
                         ("joint_sync_targets", [120, 200, 255, 230])):
            m_ = pseg == seg
            ep = "bridge/align_path" if seg == "align_sync_targets" else "bridge/joint_path"
            if m_.any():
                bridge_entities_logged.add(ep)
            else:
                bridge_omitted[ep] = f"no planned targets for segment {seg}"
            points.append({"entity_path": ep, "positions_m": ptg[m_] if m_.any() else np.zeros((0, 3)),
                           "radii": 0.0012, "colors": col, "coordinate_frame": "world_m", **btm})
    # ④ 최악 분리면을 낸 장애물 셀(AABB 박스 메시)
    cells_ = {c["name"]: c for c in (bc.get("static_inputs", {}).get("obstacle_cells") or [])}
    wc = cells_.get(gw.get("cell"))
    if wc is None:
        bridge_omitted["bridge/worst_cell"] = (
            f"no worst cell in certificate (cell={gw.get('cell')!r}) — aborted record preserved, "
            "no entity invented")
        bridge_omitted["bridge/worst_separating_plane"] = bridge_omitted["bridge/worst_cell"]
    if wc is not None:
        lo_ = np.asarray(wc["min_m"], float)
        hi_ = np.asarray(wc["max_m"], float)
        V = np.array([[lo_[0], lo_[1], lo_[2]], [hi_[0], lo_[1], lo_[2]], [hi_[0], hi_[1], lo_[2]],
                      [lo_[0], hi_[1], lo_[2]], [lo_[0], lo_[1], hi_[2]], [hi_[0], lo_[1], hi_[2]],
                      [hi_[0], hi_[1], hi_[2]], [lo_[0], hi_[1], hi_[2]]], np.float32)
        F = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                      [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], np.uint32)
        bridge_entities_logged.update({"bridge/worst_cell", "bridge/worst_separating_plane"})
        meshes.append({"entity_path": "bridge/worst_cell", "vertices_m": V, "triangles": F,
                       "color_rgba": [255, 60, 60, 90], "coordinate_frame": "world_m", "static": True})
        # ⑤ 최악 분리면: 그 축 방향으로 gap 만큼 그은 화살표(실제 분리 증거)
        ax_ = "xyz".index(str(gw.get("axis", "x")))
        o_ = pcur.copy()
        v_ = np.zeros(3)
        v_[ax_] = float(gw.get("gap_m", 0.0)) * (1.0 if (lo_[ax_] > pcur[ax_]) else -1.0)
        arrows.append({"entity_path": "bridge/worst_separating_plane", "origins_m": o_[None],
                       "vectors_m": v_[None], "colors": [255, 0, 255],
                       "coordinate_frame": "world_m", **btm})
    bridge_entities_logged.update({"bridge/current_pose", "events/bridge_clearance"})
    events.append({"entity_path": "events/bridge_clearance",
                   "text": (f"BRIDGE {bc.get('verdict')} pass={bc.get('pass')} "
                            f"z_reached_m={bc.get('z_reached_m')} q_res={bc.get('q_res_deg')} "
                            f"align_targets={bc.get('n_align_sync_targets')} "
                            f"joint_targets={bc.get('n_joint_sync_targets')} "
                            f"worst slack={gw.get('slack_m')} m gap={gw.get('gap_m')} "
                            f"bound={gw.get('motion_bound_m')} cell={gw.get('cell')} axis={gw.get('axis')} "
                            f"| elapsed bound {(pcert.get('elapsed_bound') or {}).get('rule')} "
                            f"| precheck {bc.get('precheck_summary')}"),
                   "level": "INFO", "sequence": {"frame": bsync, "phase": int(ph[bsync])},
                   "duration": {"sim_time_s": float(t[bsync])}})
    # 인증서가 중단(input_error 등)이면 worst 가 없다. **NaN 지표를 만들지 않고** 그 사실을 기록한다.
    if gw.get("slack_m") is not None and np.isfinite(float(gw["slack_m"])):
        scalars.append({"entity_path": "metrics/bridge_worst_slack_mm",
                        "value": float(gw["slack_m"]) * 1000.0, **btm})
        bridge_entities_logged.add("metrics/bridge_worst_slack_mm")
    else:
        bridge_omitted["metrics/bridge_worst_slack_mm"] = (
            "certificate has no finite global_worst (aborted/input_error) — "
            "no NaN metric invented")
        events.append({"entity_path": "events/bridge_clearance",
                       "text": ("BRIDGE_RECORD_INCOMPLETE: no finite global_worst; "
                                f"verdict={bc.get('verdict')} "
                                f"abort={str(bc.get('abort_detail'))[:300]}"),
                       "level": "WARN", "sequence": {"frame": bsync, "phase": int(ph[bsync])},
                       "duration": {"sim_time_s": float(t[bsync])}})

points.append({"entity_path": "geometry/markers/source_target",
               "positions_m": np.array([[0.0, 0.0, float(z["heightmap_pre_m"].max())],
                                        list(np.asarray(res["fixtures"]["bin"]["pos_m"], float))]),
               "radii": np.array([0.006, 0.006], np.float32),
               "colors": np.array([[255, 255, 0, 255], [255, 0, 255, 255]], np.uint8),
               "labels": ["scoop site (source)", "receiving bin (target)"],
               "coordinate_frame": "world_m", "static": True})


def bp(mode):
    import rerun.blueprint as rrb
    return rrb.Blueprint(rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", contents=["/geometry/**", "/contacts/**", "/pose/**",
                                                    "/bridge/**"],
                              name="W13 full cycle: particles + tool + fixtures + bridge decision"),
            rrb.TextLogView(origin="/events", contents="/events/**", name="decisions / door stops"),
            column_shares=[0.7, 0.3]),
        rrb.Horizontal(
            rrb.TimeSeriesView(origin="/metrics", contents=["/metrics/door_actual_deg", "/metrics/door_target_deg",
                                                            "/metrics/M_hinge_rel_Nm"], name="door"),
            rrb.TimeSeriesView(origin="/metrics", contents=["/metrics/v_particle_max", "/metrics/lip_track_err_mm",
                                                            "/metrics/n_tray_contacts",
                                                            "/metrics/bridge_worst_slack_mm"],
                               name="stability + bridge slack"),
            column_shares=[0.5, 0.5]),
        row_shares=[0.65, 0.35]),
        rrb.TimePanel(timeline="sim_time_s", play_state="paused"), auto_layout=False, auto_views=False,
        collapse_panels=True)


orig = viz_debug.build_rerun_blueprint
try:
    viz_debug.build_rerun_blueprint = bp
    status = viz_debug.log_rerun(
        rrd, coordinate_frames=[{"frame": "world_m", "parent_frame": "tf#/", "entity_path": "coordinate_frames/world_m"}],
        frames=[], meshes=meshes, points=points, arrows=arrows, scalar_trace=scalars, events=events,
        recording_metadata={
            "artifact": "DEME_W13_FULL_CYCLE_RERUN_V1", "attempt_dir": str(att),
            "result_json_sha256": hashlib.sha256((att / "w13_cycle_seed460.json").read_bytes()).hexdigest(),
            "npz_sha256": hashlib.sha256((att / "w13_cycle_seed460.npz").read_bytes()).hexdigest(),
            "n_sync": int(len(t)), "n_particle_frames": int(len(pf_t)), "phase_order": PHASES,
            "inventory_labels": INV, "particle_color_semantics": "per-frame inventory code",
            "force_arrow_scale_m_per_N": 0.005,
            # `RERUN_SOURCE_REVIEW_01` ②: 예전엔 여기에 id 0/1/2 만 **하드코딩**해서 RRD 의 run
            # metadata 가 자기 bin 접촉 entity 와 원시 role 선언을 부정했다. 이제 **원시
            # mesh_role_to_id 를 그대로** 직렬화하고, 아래 coverage 매핑과 동일성을 요구한다.
            "contact_mesh_role_to_id": {str(k): int(v) for k, v in _ROLE2ID.items() if v is not None},
            "contact_mesh_id": {str(int(v)): str(k) for k, v in _ROLE2ID.items() if v is not None},
            "contact_role_id_is_semantic_not_owner_id": bool(
                (_META.get("contact_semantics") or {}).get("role_id_is_semantic_not_owner_id", False)),
            "particle_frame_row_is_authoritative_identity": True,
            "particle_frame_row_timeline": "particle_frame_row",
            "particle_frame_sync_index_is_nondecreasing_not_unique": True,
            "scientific_authority": "Original JSON/NPZ are authority; Rerun Float32 copies are inspection only"},
        recording_id=f"w13_{att.name}", blueprint_path=rbl, blueprint_mode="w13", live_viewer=False,
        app_id="roarm_w13_full_cycle")
finally:
    viz_debug.build_rerun_blueprint = orig
if not status.get("ok"):
    raise SystemExit(f"log_rerun failed: {status}")

ents = {"/metadata/run", "/coordinate_frames/world_m", "/geometry/tray", "/geometry/bin",
        "/metadata/meshes/geometry__tray", "/metadata/meshes/geometry__bin",
        "/geometry/particles", "/geometry/tool/fixed_nodes", "/geometry/tool/door_nodes",
        "/geometry/markers/source_target",
        *[f"/contacts/{nm_}/{k_}" for _mid, nm_, _c, _rk in CONTACT_ROLES for k_ in ("points", "forces")],
        "/pose/tool_target_vs_actual", "/events/decision"} | \
    {f"/metrics/{k}" for k in KEYS} | {"/metrics/door_actual_deg", "/metrics/door_target_deg"} | \
    {"/metrics/particle_frame_source_sync_index", "/metrics/particle_frame_source_time_s",
     "/metrics/particle_frame_source_phase_code"}
if BRIDGE:
    # **실제로 로그한 것만** 정확 계약에 넣는다. 중단 인증서에서 없는 entity 를 요구해
    # 거짓 PASS 도, 지어낸 PASS 도 만들지 않는다(코디네이터 msg_b2c0bdf6cf77).
    ents |= {"/" + e for e in bridge_entities_logged}
    if "bridge/worst_cell" in bridge_entities_logged:
        ents |= {"/metadata/meshes/bridge__worst_cell"}
_TIMELINES = ["blueprint", "log_time", "frame", "particle_frame_row", "phase", "sim_time_s"]
comp = {"/geometry/particles": ["Points3D:colors", "Points3D:positions", "Points3D:radii"],
        "/metrics/particle_frame_source_sync_index": ["Scalars:scalars"],
        "/metrics/particle_frame_source_time_s": ["Scalars:scalars"],
        "/metrics/particle_frame_source_phase_code": ["Scalars:scalars"],
        **{f"/contacts/{nm_}/forces": ["Arrows3D:origins", "Arrows3D:vectors"]
           for _mid, nm_, _c, _rk in CONTACT_ROLES},
        "/pose/tool_target_vs_actual": ["Arrows3D:origins", "Arrows3D:vectors"],
        "/metrics/door_actual_deg": ["Scalars:scalars"], "/events/decision": ["TextLog:level", "TextLog:text"],
        "/geometry/tray": ["Mesh3D:triangle_indices", "Mesh3D:vertex_positions"]}
if BRIDGE:
    _bcomp = {"bridge/current_pose": ["Points3D:positions", "Points3D:radii"],
              "bridge/proposed_target": ["Points3D:positions", "Points3D:radii"],
              "bridge/align_path": ["Points3D:positions"],
              "bridge/joint_path": ["Points3D:positions"],
              "bridge/worst_cell": ["Mesh3D:triangle_indices", "Mesh3D:vertex_positions"],
              "bridge/worst_separating_plane": ["Arrows3D:origins", "Arrows3D:vectors"],
              "events/bridge_clearance": ["TextLog:level", "TextLog:text"],
              "metrics/bridge_worst_slack_mm": ["Scalars:scalars"]}
    comp.update({"/" + k_: v_ for k_, v_ in _bcomp.items() if k_ in bridge_entities_logged})
val = validate_rerun_artifact(rrd, expected_entity_paths=sorted(ents), exact_entity_paths=sorted(ents),
                              # ERRATUM_03: 입자 저장 행의 **정본 시간 좌표**를 정확 계약에 넣는다.
                              expected_timeline_names=_TIMELINES, exact_timeline_names=_TIMELINES,
                              expected_entity_components=comp, blueprint_path=rbl, screenshot_path=png,
                              screenshot_window_size="3200x1800", cli_path=Path(interp) / "rerun",
                              expected_version="0.34.1", timeout_s=600.0)
val["log_status_summary"] = {k: status.get(k) for k in ("ok", "bytes", "rerun_sdk_version",
                                                        "sink_attached_before_logging", "sink_finalized",
                                                        "flush_ok", "blueprint_status")}
_logged_total = int(sum(n_logged_contact_rows.values()))
# ERRATUM_03 행 정체성은 공유 정본 모듈이 판정한다(exporter/renderer 사본 금지).
_RI = RRI.row_identity_report(
    particle_frame_row=pf_rows_logged,
    particle_frame_sync_index=pf_s, particle_frame_t_s=pf_t, sync_phase_code=ph,
    source_sync_index=[int(pf_s[i]) for i in pf_rows_logged],
    source_time_s=[float(pf_t[i]) for i in pf_rows_logged],
    source_phase_code=[int(ph[int(pf_s[i])]) for i in pf_rows_logged],
    decision_particle_frame_index=np.asarray(z["decision_particle_frame_index"], int),
    n_raw=int(len(pf_t)), require_full_coverage=True)
_meta_role2id = {str(k): int(v) for k, v in _ROLE2ID.items() if v is not None}
_cov_role2id = {rk_: mid for mid, _nm, _c, rk_ in CONTACT_ROLES}   # **원시 키**로 비교
val["coverage"] = {"source_syncs": int(len(t)), "syncs_logged": int(len(t)),
                   "particle_frames_logged": int(len(pf_t)),
                   "particle_frame_row_identity": _RI,
                   "particle_frame_row_timeline": "particle_frame_row",
                   # `RERUN_SOURCE_REVIEW_01` ②: metadata role 표 == coverage role 표 **요구**.
                   "contact_role_map_from_raw_metadata": _meta_role2id,
                   "contact_role_map_equals_recording_metadata": bool(_meta_role2id == _cov_role2id),
                   # ⚠️ len(ci) 는 원시 행 수(복사 총계)다. 실제로 RRD 에 넣은 행만 센다.
                   "contacts_in_raw": int(len(ci)),
                   "contacts_logged": _logged_total,
                   "contacts_logged_by_role": dict(n_logged_contact_rows),
                   "contact_role_to_id": {nm_: mid for mid, nm_, _c, _rk in CONTACT_ROLES},
                   "contact_entity_for_raw_role": {rk_: nm_ for _m, nm_, _c, rk_ in CONTACT_ROLES},
                   "bridge_entities_logged": sorted(bridge_entities_logged) if BRIDGE else [],
                   "bridge_entities_omitted": (bridge_omitted if BRIDGE else {}),
                   "bridge_record_complete": bool(BRIDGE and not bridge_omitted),
                   "contact_mesh_ids_in_raw_without_role": _unmapped,
                   "all_raw_contact_rows_logged": bool(_logged_total == int(len(ci))
                                                       and not _unmapped),
                   "raw_arrays_authority": True}
# ERRATUM_03 / `RERUN_SOURCE_REVIEW_01`: 행 정체성과 role 표 정합은 **게이트다**.
# 지어낸 PASS 를 만들지 않고, 어긋나면 그 자리에서 실패시킨다.
_cov = val["coverage"]
_row_gate = {
    "row_identity_all_ok": bool(_RI["all_ok"]),
    "row_identity_failures": list(_RI["failures"]),
    "contact_role_map_equals_recording_metadata": bool(_cov["contact_role_map_equals_recording_metadata"]),
    "particle_frame_row_timeline_in_exact_contract": bool("particle_frame_row" in _TIMELINES),
}
_row_ok = (_RI["all_ok"] and _row_gate["contact_role_map_equals_recording_metadata"]
           and _row_gate["particle_frame_row_timeline_in_exact_contract"])
val["raw_row_identity_gate"] = dict(_row_gate, all_ok=bool(_row_ok))
val["pass"] = bool(val.get("pass")) and bool(_row_ok)
json.dump(val, open(valp, "w"), ensure_ascii=False, indent=2, default=str)
print(("RERUN_EXPORT_OK" if val.get("pass") else "RERUN_CONTRACT_FAIL"), rrd, png, valp,
      f"{rrd.stat().st_size/1e6:.1f} MB")
if not val.get("pass"):
    print("raw_row_identity_gate:", json.dumps(val["raw_row_identity_gate"], ensure_ascii=False))
    raise SystemExit(1)
