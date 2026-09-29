"""post06 좌표 변환 CPU 자체검사 — numpy 만 쓴다(Isaac·GPU·설치 0, 원자료는 읽기만).

무엇을 검사하나
  (a) rev34 스텁 원자료(규약 A)의 `metadata_json.w25_frame` 을 읽어 **첫 입자 프레임**을 로봇 좌표로
      옮기고 xy 경계상자·중심·긴 변 방향을 **계산해서** 적는다(기대값을 단언하지 않는다).
  (b) W19 A 원자료(`w25_frame` 없음)에서 post05 변환과 post06 변환의 **최대 절대 차**를 잰다.
  (c) sync 0 의 공구 owner 포즈(고정부·문)가 같은 R·t 를 타는지 — 위치·자세 변환 전/후 값과
      CAD 배치식 등가성(`deme_to_disp(place(v,p,q)) == place(v, R·p+o, R·R_owner)`)을 잰다.

어떻게 검사하나
  재구현이 아니라 **패치된 파일의 함수 본문을 AST 로 꺼내 그대로 실행**한다(post05 도 같은 방식).
  그래서 여기서 나온 수치는 렌더가 실제로 쓰는 식의 수치다. 실행은 CPU·numpy 뿐이다.

주장하지 않는 것
  이것은 렌더 결과 육안 검수가 아니다. 화면에 무엇이 보이는지는 GPU 렌더 뒤에만 말할 수 있다.
"""
import ast
import hashlib
import json
import math
import re
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as npfmt

HERE = Path(__file__).resolve().parent
POST05 = Path("/home/cgxr/orca/workspaces/RoArm_Project/w25-render-cad/claudedocs/research/"
              "w25_render_cad_20260928/isaac_replay_w13_post05.py")
POST06 = HERE / "isaac_replay_w13_post06.py"
FK_SRC = Path("/home/cgxr/orca/workspaces/RoArm_Project/w19-replay/claudedocs/runtime_logs/grasp_track/"
              "w19_runpod_d487/A_full_cycle/replay_20260918/rev/src/w13_fk.py")
RUN_A = Path("/home/cgxr/orca/workspaces/RoArm_Project/w25-rev34-fullcycle/claudedocs/runtime_logs/"
             "grasp_track/w25_realign_fullcycle_d498/rev34/dryrun/paperbox_final_n67737")
RUN_W19 = Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/"
               "w19_runpod_d487/A_full_cycle/run_01")
PLATE_Z = 0.38                                   # post06 상수와 같은 값(같은 파일에서 읽어 대조한다)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def grab_funcs(path, names):
    """파일에서 함수 정의 원문을 꺼내 dict 로. 같은 이름이 여럿이면 마지막 것을 쓴다."""
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            seg = ast.get_source_segment(src, node)
            out[node.name] = textwrap.dedent(" " * node.col_offset + seg)
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(f"{path} 에서 함수를 찾지 못했다: {missing}")
    return out


def make_ns(path, names, **binds):
    ns = {"np": np, "math": math}
    for name, src in grab_funcs(path, names).items():
        exec(compile(src, str(path), "exec"), ns)            # noqa: S102 — 패치본 원문을 그대로 실행
    ns.update(binds)
    return ns


def shoulder_above_plate():
    """`w13_fk.py` 의 상수를 하드코딩하지 않고 원문에서 읽는다."""
    m = re.search(r"^SHOULDER_ABOVE_PLATE\s*=\s*([0-9.+\-eE ]+)$", FK_SRC.read_text(), re.M)
    if not m:
        raise SystemExit("SHOULDER_ABOVE_PLATE 를 찾지 못했다")
    expr = m.group(1).strip()
    if not re.fullmatch(r"[0-9.+\-eE ]+", expr):
        raise SystemExit(f"상수 표현이 숫자식이 아니다: {expr!r}")
    return float(eval(expr, {"__builtins__": {}}, {})), expr    # noqa: S307 — 숫자식만 통과시킨 뒤


def read_frame0(npz_path, member):
    """(F,N,3) 같은 배열의 **첫 프레임만** 순차 읽기로 꺼낸다(전체 배열을 메모리에 올리지 않는다)."""
    with zipfile.ZipFile(npz_path) as zf, zf.open(member + ".npy") as f:
        ver = npfmt.read_magic(f)
        if ver == (1, 0):
            shape, fortran, dtype = npfmt.read_array_header_1_0(f)
        elif ver == (2, 0):
            shape, fortran, dtype = npfmt.read_array_header_2_0(f)
        else:
            raise SystemExit(f"알 수 없는 npy 버전 {ver}")
        if fortran or len(shape) != 3:
            raise SystemExit(f"{member}: 예상과 다른 저장 형태 shape={shape} fortran={fortran}")
        n = int(shape[1] * shape[2])
        return np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype).reshape(shape[1:]), tuple(shape)


def quat_xyzw_to_mat(q):
    """`w13_kinematics.quat_xyzw_to_mat` 과 같은 식(독립 재구현, 검사자 코드를 import 하지 않는다)."""
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)


def load_case(run_dir):
    z = np.load(run_dir / "w13_cycle_seed460.npz", allow_pickle=True)
    res = json.load(open(run_dir / "w13_cycle_seed460.json"))
    meta = json.loads(str(z["metadata_json"]))
    t_robot = np.asarray(res["frames"]["adapter"]["t_robot_m"], float)
    return z, res, meta, t_robot


def build_namespaces(meta, t_robot, sh):
    """post05·post06 변환 함수를 각각 같은 입력으로 묶는다."""
    W25F = meta.get("w25_frame")
    ROT = None if W25F is None else np.asarray(W25F["R_robot_box"], float)
    if ROT is not None and float(np.abs(ROT - np.eye(3)).max()) == 0.0:
        ROT = None
    origin = np.array([t_robot[0], t_robot[1], t_robot[2] + PLATE_Z + sh], float)
    ns6 = make_ns(POST06, ["_rot_quat_xyzw", "deme_to_disp", "disp_box", "disp_quat",
                           "box_to_robot", "robot_to_box", "robot_to_box_rot"],
                  ROT=ROT, origin_disp=origin, t_robot=t_robot, Q_ROT=None)
    ns6["Q_ROT"] = None if ROT is None else ns6["_rot_quat_xyzw"](ROT)
    ns5 = make_ns(POST05, ["deme_to_disp"], origin_disp=origin)
    return W25F, ROT, origin, ns5, ns6


def main():
    sh, sh_expr = shoulder_above_plate()
    rep = {"artifact": "W25_POST06_COORD_SELFTEST_V1",
           "what": "post06 표시 좌표 변환(회전 포함)의 CPU 자체검사. 물리 0 · GPU 0 · 원자료 읽기 전용.",
           "scripts": {str(POST05): sha256(POST05), str(POST06): sha256(POST06)},
           "constants": {"PLATE_Z_m": PLATE_Z, "SHOULDER_ABOVE_PLATE_expr": sh_expr,
                         "SHOULDER_ABOVE_PLATE_m": sh, "source": str(FK_SRC)}}

    # ── (a) rev34 스텁(규약 A) ────────────────────────────────────────────
    zA, resA, metaA, tA = load_case(RUN_A)
    W25F, ROT, originA, ns5A, ns6A = build_namespaces(metaA, tA, sh)
    if W25F is None:
        raise SystemExit("(a) 대상에 w25_frame 이 없다 — 규약 A 원자료가 아니다")
    posA, shpA = read_frame0(RUN_A / "w13_cycle_seed460.npz", "particle_pos_m")
    posA = np.asarray(posA, float)
    p_rob = ns6A["box_to_robot"](posA)                  # 상자 → 로봇(표시 z 오프셋 없음)
    p_dis = ns6A["deme_to_disp"](posA)                  # 상자 → 표시
    box = np.asarray(zA["box_bounds_m"], float)
    box_lo_d, box_hi_d = ns6A["disp_box"](box[:, 0], box[:, 1])
    rob_lo, rob_hi = p_rob.min(0), p_rob.max(0)
    ext = rob_hi[:2] - rob_lo[:2]
    rep["a_rev34_stub_convention_A"] = {
        "run": str(RUN_A), "npz_sha256": sha256(RUN_A / "w13_cycle_seed460.npz"),
        "w25_frame": W25F,
        "R_is_rotation_matrix": bool(float(np.abs(ROT @ ROT.T - np.eye(3)).max()) <= 1e-12
                                     and abs(float(np.linalg.det(ROT)) - 1.0) <= 1e-12),
        "R_is_pure_z_rotation": bool(abs(float(ROT[2, 2]) - 1.0) <= 1e-12
                                     and float(np.abs(ROT[2, :2]).max()) <= 1e-12
                                     and float(np.abs(ROT[:2, 2]).max()) <= 1e-12),
        "rotation_about_z_deg": round(math.degrees(math.atan2(float(ROT[1, 0]), float(ROT[0, 0]))), 9),
        "t_robot_json_vs_metadata_max_abs_diff_m": float(
            np.abs(np.asarray(W25F["t_robot_m"], float) - tA).max()),
        "particle_pos_m_shape": list(shpA), "n_particles_frame0": int(posA.shape[0]),
        "box_bounds_m_box_frame": box.tolist(),
        "box_inner_size_mm": [round(float((box[i, 1] - box[i, 0]) * 1000.0), 4) for i in range(3)],
        "frame0_box_frame": {"xy_min_m": [float(v) for v in posA.min(0)[:2]],
                             "xy_max_m": [float(v) for v in posA.max(0)[:2]],
                             "z_min_max_m": [float(posA.min(0)[2]), float(posA.max(0)[2])]},
        "frame0_robot_frame": {
            "xy_min_m": [float(v) for v in rob_lo[:2]], "xy_max_m": [float(v) for v in rob_hi[:2]],
            "xy_center_m": [round(float((rob_lo[0] + rob_hi[0]) / 2), 9),
                            round(float((rob_lo[1] + rob_hi[1]) / 2), 9)],
            "xy_extent_mm": [round(float(ext[0] * 1000.0), 4), round(float(ext[1] * 1000.0), 4)],
            "long_axis": ("x" if ext[0] > ext[1] else ("y" if ext[1] > ext[0] else "tie")),
            "z_min_max_m": [float(p_rob.min(0)[2]), float(p_rob.max(0)[2])]},
        "frame0_display_frame": {"xy_center_m": [round(float((p_dis.min(0)[0] + p_dis.max(0)[0]) / 2), 9),
                                                 round(float((p_dis.min(0)[1] + p_dis.max(0)[1]) / 2), 9)],
                                 "z_min_max_m": [float(p_dis.min(0)[2]), float(p_dis.max(0)[2])]},
        "tray_box_display_lo_hi_m": [box_lo_d.tolist(), box_hi_d.tolist()],
        "tray_box_display_size_ordered": bool(np.all(box_hi_d - box_lo_d >= 0.0)),
        "z_double_count_check": {
            "origin_disp_z_m": float(originA[2]),
            "t_robot_z_m": float(tA[2]),
            "expected_origin_z_m": float(tA[2] + PLATE_Z + sh),
            "rotation_changes_z": bool(float(np.abs((posA @ ROT.T)[:, 2] - posA[:, 2]).max()) > 0.0),
            "disp_z_minus_box_z_max_abs_dev_from_origin_z_m": float(
                np.abs((p_dis[:, 2] - posA[:, 2]) - originA[2]).max())},
        "not_a_claim": "이 수치는 좌표 계산 결과다. 렌더 화면을 본 것이 아니다."}

    # (a-2) 패치하지 않았다면 얼마나 틀리는가 — post05 식(평행이동만)을 같은 원자료에 적용해 차를 잰다.
    tool_pA = np.asarray(zA["tool_pos_m"], float)
    lip_unpatched = tool_pA + tA                       # post05 `joints_for` 의 식
    lip_patched = ns6A["box_to_robot"](tool_pA)        # post06
    d_lip = np.linalg.norm(lip_patched - lip_unpatched, axis=1)
    d_par = np.linalg.norm(ns6A["deme_to_disp"](posA) - ns5A["deme_to_disp"](posA), axis=1)
    rep["a2_error_if_unpatched"] = {
        "why": "post05(평행이동만)를 규약 A 원자료에 그대로 쓰면 생기는 어긋남. 패치의 크기를 수치로 적는다.",
        "sync0_lip_robot_post05_m": [float(v) for v in lip_unpatched[0]],
        "sync0_lip_robot_post06_m": [float(v) for v in lip_patched[0]],
        "lip_robot_offset_mm": {"sync0": round(float(d_lip[0]) * 1000.0, 4),
                                "max": round(float(d_lip.max()) * 1000.0, 4),
                                "mean": round(float(d_lip.mean()) * 1000.0, 4),
                                "n_sync": int(len(d_lip))},
        "particle_display_offset_frame0_mm": {"max": round(float(d_par.max()) * 1000.0, 4),
                                              "mean": round(float(d_par.mean()) * 1000.0, 4)},
        "owner_orientation_offset_deg": round(abs(math.degrees(math.acos(
            max(-1.0, min(1.0, (float(np.trace(ROT)) - 1.0) / 2.0))))), 9)}

    # ── (b) W19 A (w25_frame 없음) → post05 와 같은 값이어야 한다 ─────────
    zW, resW, metaW, tW = load_case(RUN_W19)
    W25F_W, ROT_W, originW, ns5W, ns6W = build_namespaces(metaW, tW, sh)
    posW, shpW = read_frame0(RUN_W19 / "w13_cycle_seed460.npz", "particle_pos_m")
    quatW, shpWq = read_frame0(RUN_W19 / "w13_cycle_seed460.npz", "particle_quat_xyzw")
    posW = np.asarray(posW, float)
    quatW = np.asarray(quatW, float)
    tool_pW = np.asarray(zW["tool_pos_m"], float)
    door_pW = np.asarray(zW["door_pos_m"], float)
    diffs = {
        "particle_pos_frame0": float(np.abs(ns6W["deme_to_disp"](posW) - ns5W["deme_to_disp"](posW)).max()),
        "particle_quat_frame0": float(np.abs(ns6W["disp_quat"](quatW) - quatW).max()),
        "tool_pos_all_sync": float(np.abs(ns6W["deme_to_disp"](tool_pW)
                                          - ns5W["deme_to_disp"](tool_pW)).max()),
        "door_pos_all_sync": float(np.abs(ns6W["deme_to_disp"](door_pW)
                                          - ns5W["deme_to_disp"](door_pW)).max()),
        "box_to_robot_vs_post05_plus_t": float(np.abs(ns6W["box_to_robot"](tool_pW)
                                                      - (tool_pW + tW)).max()),
        "robot_to_box_vs_post05_minus_t": float(np.abs(ns6W["robot_to_box"](tool_pW)
                                                       - (tool_pW - tW)).max()),
        "disp_box_lo_vs_post05": float(np.abs(
            ns6W["disp_box"](np.asarray(zW["box_bounds_m"], float)[:, 0],
                             np.asarray(zW["box_bounds_m"], float)[:, 1])[0]
            - ns5W["deme_to_disp"](np.asarray(zW["box_bounds_m"], float)[:, 0])).max()),
        "disp_box_hi_vs_post05": float(np.abs(
            ns6W["disp_box"](np.asarray(zW["box_bounds_m"], float)[:, 0],
                             np.asarray(zW["box_bounds_m"], float)[:, 1])[1]
            - ns5W["deme_to_disp"](np.asarray(zW["box_bounds_m"], float)[:, 1])).max()),
    }
    rep["b_w19A_regression_no_w25_frame"] = {
        "run": str(RUN_W19), "npz_sha256": sha256(RUN_W19 / "w13_cycle_seed460.npz"),
        "w25_frame_present": bool(W25F_W is not None), "ROT_branch": "None (post05 와 같은 식)",
        "particle_pos_m_shape": list(shpW), "particle_quat_xyzw_shape": list(shpWq),
        "n_tool_sync_rows": int(tool_pW.shape[0]),
        "max_abs_diff_post05_vs_post06_m": diffs,
        "all_zero": bool(max(diffs.values()) == 0.0),
        "rotation_applied": False}

    # ── (c) owner 포즈(고정부·문) 변환 + CAD 배치식 등가성 ─────────────────
    cases = {}
    rng = np.random.default_rng(460)
    v_local = rng.normal(0.0, 0.05, size=(64, 3))        # CAD owner-local 대역의 표본 점(형상 무관 항등식)
    for nm, (zz, ns6, ROTx, originx) in {"a_rev34_stub": (zA, ns6A, ROT, originA),
                                         "b_w19A": (zW, ns6W, ROT_W, originW)}.items():
        row = {}
        for owner, kp, kq in (("fixed", "tool_pos_m", "tool_quat_xyzw"),
                              ("door", "door_pos_m", "door_quat_xyzw")):
            p0 = np.asarray(zz[kp][0], float)
            q0 = np.asarray(zz[kq][0], float)
            R0 = quat_xyzw_to_mat(q0)
            Rx = np.eye(3) if ROTx is None else ROTx
            p_disp = ns6["deme_to_disp"](p0)
            R_disp = Rx @ R0
            # CAD 배치식 등가성: deme_to_disp(place(v,p0,q0)) == v @ (R·R0)ᵀ + (R·p0 + origin)
            lhs = ns6["deme_to_disp"](v_local @ R0.T + p0)
            rhs = v_local @ R_disp.T + p_disp
            row[owner] = {
                "sync0_pos_box_m": [float(v) for v in p0],
                "sync0_quat_xyzw_raw": [float(v) for v in q0],
                "sync0_pos_robot_m": [float(v) for v in ns6["box_to_robot"](p0)],
                "sync0_pos_display_m": [float(v) for v in p_disp],
                "R_owner_box": R0.tolist(),
                "R_owner_display": R_disp.tolist(),
                "R_owner_changed": bool(float(np.abs(R_disp - R0).max()) > 0.0),
                "R_owner_max_abs_change": float(np.abs(R_disp - R0).max()),
                "cad_place_identity_max_abs_diff_m": float(np.abs(lhs - rhs).max()),
                "cad_place_identity_holds": bool(float(np.abs(lhs - rhs).max()) <= 1e-15)}
        # 문 ↔ 고정부 상대 자세는 회전 뒤에도 보존되어야 한다(조립이 통째로 돈다)
        Rf = quat_xyzw_to_mat(np.asarray(zz["tool_quat_xyzw"][0], float))
        Rd = quat_xyzw_to_mat(np.asarray(zz["door_quat_xyzw"][0], float))
        Rx = np.eye(3) if ROTx is None else ROTx
        rel_before = Rf.T @ Rd
        rel_after = (Rx @ Rf).T @ (Rx @ Rd)
        d_before = np.asarray(zz["door_pos_m"][0], float) - np.asarray(zz["tool_pos_m"][0], float)
        d_after = ns6["deme_to_disp"](np.asarray(zz["door_pos_m"][0], float)) - \
            ns6["deme_to_disp"](np.asarray(zz["tool_pos_m"][0], float))
        row["assembly_rigidity"] = {
            "relative_rotation_max_abs_change": float(np.abs(rel_after - rel_before).max()),
            "door_minus_fixed_distance_box_m": float(np.linalg.norm(d_before)),
            "door_minus_fixed_distance_display_m": float(np.linalg.norm(d_after)),
            "distance_max_abs_change_m": float(abs(np.linalg.norm(d_after) - np.linalg.norm(d_before)))}
        cases[nm] = row
    rep["c_owner_pose_and_cad_placement"] = cases

    rep["checks"] = {
        "a_R_is_pure_z_rotation": rep["a_rev34_stub_convention_A"]["R_is_pure_z_rotation"],
        "a_t_robot_json_eq_metadata": rep["a_rev34_stub_convention_A"][
            "t_robot_json_vs_metadata_max_abs_diff_m"] == 0.0,
        "a_z_not_double_counted": rep["a_rev34_stub_convention_A"]["z_double_count_check"][
            "disp_z_minus_box_z_max_abs_dev_from_origin_z_m"] == 0.0,
        "a_display_box_corners_ordered": rep["a_rev34_stub_convention_A"]["tray_box_display_size_ordered"],
        "b_w19A_post05_identical": rep["b_w19A_regression_no_w25_frame"]["all_zero"],
        "c_cad_place_identity": all(
            cases[nm][o]["cad_place_identity_holds"] for nm in cases for o in ("fixed", "door")),
        "c_assembly_rigid": all(
            cases[nm]["assembly_rigidity"]["relative_rotation_max_abs_change"] <= 1e-15
            and cases[nm]["assembly_rigidity"]["distance_max_abs_change_m"] <= 1e-15 for nm in cases),
        "c_rev34_owner_rotation_applied": cases["a_rev34_stub"]["fixed"]["R_owner_changed"],
        "c_w19A_owner_rotation_absent": not cases["b_w19A"]["fixed"]["R_owner_changed"]}
    rep["n_checks"] = len(rep["checks"])
    rep["n_passed"] = int(sum(1 for v in rep["checks"].values() if v))
    rep["all_pass"] = bool(rep["n_passed"] == rep["n_checks"])

    out = HERE / "selftest_post06.json"
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=2)
    a = rep["a_rev34_stub_convention_A"]["frame0_robot_frame"]
    print(f"[a] rev34 스텁 첫 입자 프레임(로봇 좌표): xy 중심 {a['xy_center_m']} m · "
          f"xy 크기 {a['xy_extent_mm']} mm · 긴 변 = {a['long_axis']}")
    print(f"[b] W19 A post05 대비 최대 절대 차 = {max(diffs.values()):.17g} m (항목 {len(diffs)}개)")
    for nm in cases:
        f_ = cases[nm]["fixed"]
        print(f"[c] {nm} 고정부 sync0: box {f_['sync0_pos_box_m']} → display {f_['sync0_pos_display_m']} · "
              f"R 변화 {f_['R_owner_max_abs_change']:.3g} · CAD 항등식 {f_['cad_place_identity_max_abs_diff_m']:.3g} m")
    print(f"SELFTEST {rep['n_passed']}/{rep['n_checks']} "
          f"{'PASS' if rep['all_pass'] else 'FAIL: ' + str([k for k, v in rep['checks'].items() if not v])}"
          f" -> {out}")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
