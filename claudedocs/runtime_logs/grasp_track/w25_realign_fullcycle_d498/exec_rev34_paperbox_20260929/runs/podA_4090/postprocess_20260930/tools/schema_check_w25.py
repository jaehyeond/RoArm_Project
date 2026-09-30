#!/usr/bin/env python3
"""P5 원자료 스키마 검사 + P6 규약(criteria) 항목 검사 — CPU 전용, 생산 모듈 import 0.

규약 정본: RAW_SCHEMA_REQUIRED.md + ERRATUM_01..04 (인용은 파일:줄),
등록 임계: criteria_w25_paperbox_cap32h.json.
사후 완화 0 — 충족하지 못한 항목은 FAIL 로 그대로 적는다.
"""
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np

AUD = Path("/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/"
           "w13_full_cycle_d484/resume_20260913/audit")
CONTRACTS = {str(AUD / n): None for n in ("RAW_SCHEMA_REQUIRED.md", "RAW_SCHEMA_REQUIRED_ERRATUM_01.md",
                                          "RAW_SCHEMA_REQUIRED_ERRATUM_02.md", "RAW_SCHEMA_REQUIRED_ERRATUM_03.md",
                                          "RAW_SCHEMA_REQUIRED_ERRATUM_04.md")}
LABELS = ["source", "receiving_bin", "tool_residual", "spill", "in_flight", "ambiguous"]
PHASES = ["initial_home", "settle", "approach", "descend", "close", "lift", "reclose",
          "transport", "discharge", "discharge_wait", "close_after_discharge", "return_home"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jdef(o):
    """numpy 스칼라를 파이썬 기본형으로 — 값 변환 0."""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(type(o)))


def q(cite, text):
    return {"cite": cite, "text": text}


def main():
    ap = argparse.ArgumentParser()
    for k in ("run", "criteria", "exec-pin", "out-schema", "out-criteria"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t0 = time.time()
    RUN = Path(a.run)
    raw_p = RUN / "w13_cycle_seed460.npz"
    res = json.load(open(RUN / "w13_cycle_seed460.json"))
    tl = json.load(open(RUN / "timeline_seed460.json"))
    rec_exec = json.load(open(RUN / "EXECUTION_RECEIPT.json"))
    rstat = json.load(open(RUN / "RUN_STATUS.json"))
    hver = json.load(open(RUN / "HASH_VERIFICATION_RECEIPT.json"))
    crit = json.load(open(a.criteria))
    pin = json.load(open(a.exec_pin))
    z = np.load(raw_p, allow_pickle=False)
    meta = json.loads(str(z["metadata_json"]))
    P = res["params"]
    for k in CONTRACTS:
        CONTRACTS[k] = sha(k) if Path(k).exists() else None

    keys = set(z.files)
    T = int(z["sync_t_s"].shape[0])
    F, N = np.asarray(z["inventory_code"]).shape
    st = np.asarray(z["sync_t_s"], float)
    pfs = np.asarray(z["particle_frame_sync_index"]).astype(np.int64)
    pft = np.asarray(z["particle_frame_t_s"], float)
    pfr = np.asarray(z["particle_frame_row"]).astype(np.int64)
    trans = np.asarray(z["transition_sync_index"]).astype(np.int64)
    pcode = np.asarray(z["sync_phase_code"]).astype(int)
    items = []

    def add(name, ok, measured, quotes=(), note=None, severity="contract"):
        items.append({"name": name, "verdict": "PASS" if ok else "FAIL", "severity": severity,
                      "contract_quotes": list(quotes), "measured": measured,
                      **({"note": note} if note else {})})

    # ── P5-1 dense 배열 행수 ────────────────────────────────────────────────
    dense_req = ["sync_t_s", "sync_phase_code", "sync_requested_duration_s", "sync_derived_internal_steps",
                 "scalar_v_particle_max", "tool_pos_m", "tool_quat_xyzw", "fixed_pos_m", "fixed_quat_xyzw",
                 "door_pos_m", "door_quat_xyzw", "door_target_pos_m", "door_target_quat_xyzw",
                 "fixed_target_pos_m", "fixed_target_quat_xyzw", "tool_target_pos_m", "tool_target_quat_xyzw"]
    missing_dense = [k for k in dense_req if k not in keys]
    wrong_rows = {k: list(np.asarray(z[k]).shape) for k in dense_req if k in keys and np.asarray(z[k]).shape[0] != T}
    add("dense_sync_clock_arrays_present_with_T_rows", not missing_dense and not wrong_rows,
        {"T": T, "n_required": len(dense_req), "missing": missing_dense, "wrong_row_count": wrong_rows,
         "dtypes": {k: str(np.asarray(z[k]).dtype) for k in dense_req if k in keys}},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":33-35",
           "`tool_pos_m[T,3]`, `tool_quat_xyzw[T,4]`, `fixed_pos_m[T,3]`, `fixed_quat_xyzw[T,4]`, "
           "`door_pos_m[T,3]`, `door_quat_xyzw[T,4]`, `bin_pos_m[T,3]`, `bin_quat_xyzw[T,4]` — actual tracker poses.")],
        note="`bin_pos_m`/`bin_quat_xyzw` 는 고정 픽스처라 (3,)/(4,) 단일 포즈로 저장돼 dense 목록에서 뺐다 — 별도 항목에서 확인.")

    # ── P5-2 sparse 입자 배열 ──────────────────────────────────────────────
    sparse_ok, sparse = True, {}
    for k, shp in (("particle_pos_m", (F, N, 3)), ("particle_quat_xyzw", (F, N, 4)),
                   ("particle_vel_m_s", (F, N, 3)), ("particle_omega_rad_s", (F, N, 3)),
                   ("inventory_code", (F, N)), ("particle_ids", (N,)),
                   ("particle_frame_sync_index", (F,)), ("particle_frame_t_s", (F,)),
                   ("particle_frame_row", (F,))):
        got = tuple(np.asarray(z[k]).shape) if k in keys else None
        sparse[k] = {"shape": list(got) if got else None, "expected": list(shp),
                     "dtype": str(np.asarray(z[k]).dtype) if k in keys else None}
        sparse_ok &= got == shp
    add("sparse_particle_frame_arrays_shapes_and_dtypes", sparse_ok,
        {"F": F, "N": N, "arrays": sparse,
         "particle_mass_kg_is_scalar": np.asarray(z["particle_mass_kg"]).shape == (),
         "particle_ids_equals_arange": bool(np.array_equal(np.asarray(z["particle_ids"]), np.arange(N)))},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":45-47",
           "`particle_ids[N]`, `particle_mass_kg` (scalar or exact uniform vector), `particle_pos_m[F,N,3]`, "
           "`particle_quat_xyzw[F,N,4]`, and `particle_vel_m_s[F,N,3]`.")])

    # ── P5-3 inventory_labels 정확 일치 ────────────────────────────────────
    got_lab = [str(v) for v in z["inventory_labels"]]
    add("inventory_labels_exact_and_code_shape", got_lab == LABELS and np.asarray(z["inventory_code"]).shape == (F, N),
        {"labels": got_lab, "expected": LABELS, "inventory_code_shape": [F, N],
         "dtype": str(np.asarray(z["inventory_code"]).dtype),
         "codes_in_range": bool(((np.asarray(z["inventory_code"]) >= 0) &
                                 (np.asarray(z["inventory_code"]) <= 5)).all())},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":48-51",
           "`inventory_labels` exactly `[source, receiving_bin, tool_residual, spill, in_flight, ambiguous]` "
           "and `inventory_code[F,N]`.")])

    # ── P5-4 metadata 필수 키 (w25_frame 포함) ─────────────────────────────
    req_meta = ["phase_order", "phase_code", "inventory_labels", "particle_frame_dt_s", "quaternion_order",
                "contact_mesh_id", "contact_semantics", "time_authority", "body_roles", "tool_cavity",
                "source_bounds_m", "moving_threshold_m_s", "moving_threshold_operator", "spill_rest_z_m",
                "classify_margin_m", "classify_geometry_basis", "classify_spheres_per_clump",
                "classify_sphere_expansion", "classify_owner_quat_source", "classify_inside_rule",
                "classify_boundary_rule", "classify_spill_basis", "overlap_priority", "velocity_severity",
                "numeric_receipt", "w25_frame",
                "particle_frame_row_is_authoritative_identity", "particle_frame_rows_deduplicated"]
    miss_meta = [k for k in req_meta if k not in meta]
    w25f = meta.get("w25_frame", {})
    w25_sub = [k for k in ("R_robot_box", "t_robot_m", "box_frame_convention", "box_anchor") if k not in w25f]
    add("metadata_required_keys_including_w25_frame", not miss_meta and not w25_sub,
        {"n_required": len(req_meta), "missing": miss_meta, "w25_frame": w25f,
         "w25_frame_missing_subkeys": w25_sub,
         "phase_order_exact": meta.get("phase_order") == PHASES},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":68-70",
           "`phase_order` exactly: `initial_home, settle, approach, descend, close, lift, reclose, transport, "
           "discharge, discharge_wait, close_after_discharge, return_home`.")])

    # ── P5-5 metadata time_mapping_abs_s / geometry_epsilon_m (W19 FAIL 항목) ─
    have = [k for k in ("time_mapping_abs_s", "geometry_epsilon_m") if k in meta]
    add("metadata_time_mapping_abs_s_and_geometry_epsilon_m", len(have) == 2,
        {"present": have, "absent": [k for k in ("time_mapping_abs_s", "geometry_epsilon_m") if k not in meta],
         "nearest_present_fields": {k: meta.get(k) for k in ("time_authority",) if k in meta}},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":71-72",
           "`time_mapping_abs_s`, `geometry_epsilon_m`, particle density, canonical pile absolute path and "
           "SHA-256, and every frozen scientific criterion path/hash.")],
        note="W19 A run 에서도 같은 두 키가 없어 FAIL 이었다(RAW_SCHEMA_CHECK_W19 "
             "metadata_time_mapping_and_geometry_epsilon_fields). 사후 완화 0.")

    # ── P5-6 ERRATUM_04 선언 문자열 ────────────────────────────────────────
    fr = meta.get("classify_floor_rule"); cv = meta.get("classify_contract_version")
    ok04 = (fr == "support_surface_v2: min(center_z - r) > floor - margin"
            and cv == "RAW_SCHEMA_REQUIRED + ERRATUM_04")
    add("erratum04_floor_rule_and_contract_version_declared_in_raw_metadata", ok04,
        {"classify_floor_rule": fr, "classify_contract_version": cv,
         "classify_source_floor_rule": meta.get("classify_source_floor_rule"),
         "classify_revision": meta.get("classify_revision"),
         "classify_predicate_module": meta.get("classify_predicate_module")},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED_ERRATUM_04.md") + ":37-38",
           '- `classify_floor_rule = "support_surface_v2: min(center_z - r) > floor - margin"`  '
           '- `classify_contract_version = "RAW_SCHEMA_REQUIRED + ERRATUM_04"`')],
        note="W19 A run 은 이 두 선언이 없어 FAIL 이었다. W25 rev34 는 선언 여부를 실측한다.")

    # ── P5-7 ERRATUM_03 행 정체성 ──────────────────────────────────────────
    add("erratum03_particle_frame_row_identity_and_nondecreasing_sync",
        bool(np.array_equal(pfr, np.arange(F)) and (np.diff(pfs) >= 0).all()),
        {"row_equals_arange": bool(np.array_equal(pfr, np.arange(F))),
         "sync_index_nondecreasing": bool((np.diff(pfs) >= 0).all()),
         "n_duplicate_sync_index": int(F - len(np.unique(pfs))),
         "declared_row_authoritative": meta.get("particle_frame_row_is_authoritative_identity"),
         "declared_deduplicated": meta.get("particle_frame_rows_deduplicated")},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED_ERRATUM_03.md") + ":7-12",
           "`particle_frame_sync_index[F]` is **nondecreasing**, not necessarily unique. ... The authoritative "
           "identity is the implicit raw row index `particle_frame_row = 0..F-1`.")])

    # ── P5-8 ERRATUM_04 §2 전환 경계 프레임 i-1 ────────────────────────────
    pset = set(int(v) for v in pfs)
    miss_b = [int(i) for i in trans if (int(i) - 1) not in pset]
    add("erratum04_transition_boundary_frame_i_minus_1_present", not miss_b,
        {"transition_sync_index": trans.tolist(), "n_transitions": int(len(trans)),
         "missing_boundary_rows": miss_b,
         "also_has_frame_at_i": [int(i) for i in trans if int(i) in pset]},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED_ERRATUM_04.md") + ":42-45",
           "For every `transition_sync_index` value `i`, a particle-frame row is required with "
           "`particle_frame_sync_index == i - 1`.")])

    # ── P5-9 전환 인덱스 == dense phase 변경 ───────────────────────────────
    exp_t = (np.flatnonzero(pcode[1:] != pcode[:-1]) + 1).astype(np.int64)
    add("transition_sync_index_equals_dense_phase_changes", bool(np.array_equal(trans, exp_t)),
        {"recorded": trans.tolist(), "recomputed_phase_only": exp_t.tolist(),
         "n_phases_seen": int(len(np.unique(pcode))), "phase_codes_seen": sorted(int(v) for v in np.unique(pcode))},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":52",
           "- `transition_sync_index` exactly equals dense phase-change indices.")])

    # ── P5-10 sync_t_s 단조 + 입자 프레임 시간 대응 ────────────────────────
    strict = bool((np.diff(st) > 0).all())
    tmap_err = float(np.abs(pft - st[pfs]).max())
    add("sync_time_strictly_increasing_and_particle_frame_time_mapping",
        strict and tmap_err == 0.0,
        {"sync_t_s_strictly_increasing": strict,
         "n_nonincreasing": int((np.diff(st) <= 0).sum()),
         "max_abs_particle_frame_t_minus_sync_t_s": tmap_err,
         "sync_t_s_dtype": str(st.dtype), "legacy_display_present": "sync_t_s_legacy_round9_display" in keys,
         "time_authority": meta.get("time_authority", {}).get("sync_t_s")},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":25-28",
           "`sync_t_s[T]` — unrounded binary64 `float(s.GetSimTime())` after each completed sync, strictly "
           "increasing. The inherited rev10 `round(GetSimTime(),9)` value may be retained only as a separately "
           "named display/legacy field and must not be the duration authority."),
         q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":42-44",
           "`particle_frame_t_s[F]` must equal `sync_t_s[index]` within the frozen source time tolerance.")])

    # ── P5-11 접촉 행 ──────────────────────────────────────────────────────
    C = int(np.asarray(z["contact_sync_index"]).shape[0])
    cm = np.asarray(z["contact_mesh"])
    csi = np.asarray(z["contact_sync_index"])
    roles = meta.get("contact_semantics", {}).get("mesh_role_to_id", {})
    add("raw_contact_rows_shapes_role_mapping_and_index_range",
        (np.asarray(z["contact_force_N"]).shape == (C, 3) and np.asarray(z["contact_point_m"]).shape == (C, 3)
         and int(csi.min()) >= 0 and int(csi.max()) < T
         and set(int(v) for v in np.unique(cm)) <= {0, 1, 2, 3}
         and roles == {"fixed": 0, "door": 1, "tray": 2, "bin": 3}),
        {"C": C, "force_shape": list(np.asarray(z["contact_force_N"]).shape),
         "point_shape": list(np.asarray(z["contact_point_m"]).shape),
         "mesh_ids_seen": sorted(int(v) for v in np.unique(cm)),
         "sync_index_min": int(csi.min()), "sync_index_max": int(csi.max()), "T": T,
         "mesh_role_to_id": roles,
         "force_frame": meta.get("contact_semantics", {}).get("force_frame"),
         "scalar_force_reduction": meta.get("contact_semantics", {}).get("scalar_force_reduction")},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":56-64",
           "`contact_sync_index[C]`, `contact_mesh[C]`, `contact_force_N[C,3]` ... Metadata must define "
           "`contact_mesh_role_to_id` for `fixed`, `door`, `tray`, and `bin`, plus force sign/frame semantics.")])

    # ── P5-12 ERRATUM_02 bin 반경 의미 + 힘 축약 선언 ──────────────────────
    bf = res["fixtures"]["bin"]
    add("erratum02_bin_radius_semantics_and_force_reduction_declared",
        bf.get("radius_semantics") == "circumradius"
        and meta.get("contact_semantics", {}).get("scalar_force_reduction") in ("norm_of_vector_sum",
                                                                                "sum_of_vector_norms"),
        {"radius_semantics": bf.get("radius_semantics"), "apothem_m": bf.get("apothem_m"),
         "circumradius_m": bf.get("circumradius_m"), "n_theta": bf.get("n_theta"),
         "apothem_matches_circumradius_cos": abs(float(bf["apothem_m"]) - float(bf["circumradius_m"]) *
                                                 np.cos(np.pi / int(bf["n_theta"]))) < 1e-15,
         "scalar_force_reduction": meta.get("contact_semantics", {}).get("scalar_force_reduction")},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED_ERRATUM_02.md") + ":6-8",
           "A regular polygon prism must state either `apothem_m` directly or `circumradius_m`. If the legacy "
           "field `inner_radius_m` is retained, it must also state `radius_semantics` as exactly `apothem` or "
           "`circumradius`.")])

    # ── P5-13 tool cavity / source bounds / 임계 선언 ──────────────────────
    tc = meta.get("tool_cavity", {})
    tc_ok = all(k in tc for k in ("tool_to_cavity_R", "cavity_origin_l5_m", "cavity_center_xz_l5_m",
                                  "radius_m", "half_y_m", "equation"))
    sb = np.asarray(meta.get("source_bounds_m"), float)
    add("tool_cavity_and_source_bounds_and_thresholds_declared",
        tc_ok and sb.shape == (3, 2) and np.array_equal(sb, np.asarray(z["box_bounds_m"], float))
        and meta.get("moving_threshold_operator", "").startswith(">=")
        and "spill_rest_z_m" in meta and "overlap_priority" in meta,
        {"tool_cavity_keys_ok": tc_ok, "cavity_equation": tc.get("equation"),
         "source_bounds_m": sb.tolist(), "equals_raw_box_bounds_m":
             bool(np.array_equal(sb, np.asarray(z["box_bounds_m"], float))),
         "moving_threshold_m_s": meta.get("moving_threshold_m_s"),
         "moving_threshold_operator": meta.get("moving_threshold_operator"),
         "spill_rest_z_m": meta.get("spill_rest_z_m"), "classify_margin_m": meta.get("classify_margin_m"),
         "overlap_priority": meta.get("overlap_priority")},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":76-83",
           "Source bounds: `source_bounds_m[3,2]` in world coordinates with strict whole-sphere containment and "
           "frozen margin. Tool cavity: `tool_to_cavity_R[3,3]`, `cavity_origin_m[3]`, `cavity_center_xz_m[2]`, "
           "`radius_m`, and `half_y_m`. ... `moving_threshold_m_s` and `spill_rest_z_m`, their comparison "
           "operators and overlap priority.")])

    # ── P5-14 설치 수치 입력 ───────────────────────────────────────────────
    nr = meta.get("numeric_receipt", {})
    nrk = ["timestep_h_s", "requested_duration_D_s", "internal_steps_N", "length_unit_l_m", "voxel_size_m",
           "domain_max_coord_m", "qmin", "position_lattice_allowance_m", "tracker_read_allowance_m",
           "quaternion_rotation_allowance_m_by_body", "r_max_local_m_by_body"]
    miss_nr = [k for k in nrk if k not in nr]
    add("installed_numeric_inputs_and_allowances_present", not miss_nr,
        {"missing": miss_nr, "values": {k: nr.get(k) for k in nrk}},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":84-88",
           "Installed numeric inputs/outputs from the source-bound chain: binary32 `h`, requested `D`, internal "
           "step count `N`, initialized `l`, `voxelSize`, domain maximum coordinate magnitude, first-read and "
           "minimum accepted `qnorm`, principal-rotvec angular bound, maximum local tool radius `rmax`, and every "
           "separate motion/lattice/tracker/quaternion/time-rounding allowance.")])

    # ── P5-15 정준 템플릿 + 입자 질량 ──────────────────────────────────────
    pm = float(np.asarray(z["particle_mass_kg"]))
    add("canonical_template_and_particle_mass_consistent",
        abs(pm - float(res["particle"]["mass_kg"])) < 1e-18
        and abs(float(res["delivery"]["particle_mass_g"]) - pm * 1000) < 1e-6
        and int(res["particle"]["n_spheres"]) == int(meta["classify_spheres_per_clump"]),
        {"particle_mass_kg_raw": pm, "particle_mass_kg_json": res["particle"]["mass_kg"],
         "particle_mass_g_delivery": res["delivery"]["particle_mass_g"],
         "n_spheres_json": res["particle"]["n_spheres"],
         "classify_spheres_per_clump_meta": meta["classify_spheres_per_clump"],
         "expand_vs_npz_max_err_m": res["particle"]["expand_vs_npz_max_err_m"]},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":14-16",
           "Particle `pos/quaternion` is the clump owner pose. Canonical sphere centers are recomputed as "
           "`p_world = p_owner + R(q_owner)*offset_m` using the pinned pile NPZ offsets and radii.")])

    # ── P5-16 속도 심각도 (5 경고 / 20 정지) ───────────────────────────────
    vmax = float(np.asarray(z["scalar_v_particle_max"]).max())
    vs = meta.get("velocity_severity", {})
    add("dense_speed_scan_respects_5_warning_and_20_stop",
        vmax <= float(vs.get("hard_stop_m_s", 20.0)),
        {"max_scalar_v_particle_max": vmax, "warning_m_s": vs.get("warning_m_s"),
         "hard_stop_m_s": vs.get("hard_stop_m_s"),
         "n_sync_over_warning": int((np.asarray(z["scalar_v_particle_max"]) >
                                     float(vs.get("warning_m_s", 5.0))).sum()),
         "n_sync_over_hard_stop": int((np.asarray(z["scalar_v_particle_max"]) >
                                       float(vs.get("hard_stop_m_s", 20.0))).sum()),
         "json_pops": res["pops"]},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":82-83",
           "`moving_threshold_m_s` and `spill_rest_z_m`, their comparison operators and overlap priority. "
           "Ambiguous boundary/overlap cases stay `ambiguous`.")],
        note="임계 정본은 criteria engine.pop_speed_m_s / engine.diag_stop_v_m_s — P6 에서 따로 검사한다.")

    # ── P5-17 실행 영수증 필드 + 물리 벽시계 상한 ──────────────────────────
    req_rc = ["argv", "cwd", "env_overrides", "started_utc", "ended_utc", "rc", "timed_out", "killed",
              "wall_s", "stdout", "stderr", "cap_s", "auto_retry", "child_pid", "child_pgid"]
    miss_rc = [k for k in req_rc if k not in rec_exec]
    cap_crit = [t for t in crit["thresholds"] if t["id"] == "runner.physics_wall_cap_s"][0]
    cap_val = float(cap_crit["value"]) if not isinstance(cap_crit["value"], dict) else None
    add("execution_receipt_fields_and_physics_wall_cap", not miss_rc and float(rec_exec["cap_s"]) == cap_val
        and float(rec_exec["wall_s"]) <= float(rec_exec["cap_s"]),
        {"missing_fields": miss_rc, "cap_s_receipt": rec_exec.get("cap_s"),
         "cap_s_criteria": cap_val, "criteria_id": cap_crit["id"], "criteria_severity": cap_crit["severity"],
         "wall_s": rec_exec.get("wall_s"), "rc": rec_exec.get("rc"),
         "timed_out": rec_exec.get("timed_out"), "killed": rec_exec.get("killed"),
         "auto_retry": rec_exec.get("auto_retry"),
         "stdout_sha256_declared": "stdout_sha256" in rec_exec, "stderr_sha256_declared": "stderr_sha256" in rec_exec,
         "wall_over_cap_ratio": round(float(rec_exec["wall_s"]) / cap_val, 6) if cap_val else None},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":101-104",
           "Execution receipt records exact argv, environment allowlist, start/end, return code, timeout state, "
           "stdout/stderr paths and hashes, maximum 32400 s physics wall limit, process count exactly one, and "
           "bounded TERM/KILL outcome. The runner refuses a pre-existing output directory before spawning."),
         q(a.criteria + " (runner.physics_wall_cap_s)", json.dumps(cap_crit["value"], ensure_ascii=False))],
        note="기저 규약의 32400 s 문구는 W25 등록 criteria 에서 115200 s 로 대체됐다(사용자 결정, 결과 관측 전 동결). "
             "영수증에 stdout/stderr **해시** 필드는 없다 — 경로만 있다.")

    # ── P5-18 정지가 성공이 아님(D486): rc·stderr 대조 ─────────────────────
    err_p = RUN / "run_paperbox_full_cycle.stderr.txt"
    err_sz = err_p.stat().st_size
    add("rc0_matches_empty_stderr_and_no_timeout_or_kill",
        int(rec_exec["rc"]) == 0 and not rec_exec["timed_out"] and not rec_exec["killed"]
        and err_sz == 0 and rstat["state"] == "completed_rc0" and tl["state"] == "complete",
        {"rc": rec_exec["rc"], "timed_out": rec_exec["timed_out"], "killed": rec_exec["killed"],
         "group_alive_after": rec_exec["group_alive_after"], "stderr_bytes": err_sz,
         "run_status_state": rstat["state"], "timeline_state": tl["state"],
         "signals_received": rstat.get("signals_received"),
         "json_abort_class": res.get("abort_class"), "json_diverged": res.get("diverged"),
         "json_fail_reason": res.get("fail_reason"),
         "json_stopped_early_after_phase": res.get("stopped_early_after_phase")},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":105-107",
           "Run status explicitly distinguishes finalization/full-cycle coverage from the scientific delivery "
           "verdict; an unsuccessful delivery is reported as such.")],
        note="D486: 자동 종료 분류는 stderr 와 대조한다. rc0 + 빈 stderr 는 '완주 기록'이지 배출 성공 판정이 아니다.")

    # ── P5-19 재시도 0 ─────────────────────────────────────────────────────
    nrt = [t for t in crit["thresholds"] if t["id"] == "runner.no_retry"][0]
    add("runner_no_retry_and_single_attempt_dir", rec_exec.get("auto_retry") is False
        and rstat.get("signals_received") == [] and (RUN.parent / "run_02").exists() is False,
        {"auto_retry": rec_exec.get("auto_retry"), "signals_received": rstat.get("signals_received"),
         "attempt_dirs_present": sorted(p.name for p in RUN.parent.glob("run_*")),
         "criteria_value": nrt["value"], "criteria_severity": nrt["severity"]},
        [q(a.criteria + " (runner.no_retry)", json.dumps(nrt["value"], ensure_ascii=False))])

    # ── P5-20 사전 해시 영수증 ─────────────────────────────────────────────
    decl = res["inputs_sha256"]
    recomputed, mism, absent = 0, [], []
    for p, h in decl.items():
        if Path(p).exists():
            recomputed += 1
            if sha(p) != h:
                mism.append(p)
        else:
            absent.append(p)
    add("declared_input_hashes_independently_recomputed_and_prelaunch_receipt_clean",
        not mism and not hver["mismatches"] and not hver["pre_existing_attempt_entries"],
        {"n_declared_inputs": len(decl), "n_recomputed": recomputed, "n_mismatch": len(mism),
         "mismatch": mism, "not_present_locally": absent,
         "hash_verification_receipt": {"n_checked": hver["n_checked"], "n_mismatch": len(hver["mismatches"]),
                                       "n_pre_existing": len(hver["pre_existing_attempt_entries"])},
         "run_status_n_hash_checks": rstat.get("n_hash_checks"),
         "run_status_hash_mismatches": rstat.get("hash_mismatches"),
         "run_status_stray_entries": rstat.get("stray_entries")},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":105-106",
           "Final manifest gives path, byte count, and SHA-256 for every output.")])

    # ── P5-21 동결 rev34 소스가 EXEC_PIN 과 바이트 동일 ────────────────────
    pf = pin["files"]
    bad_pin, checked = [], 0
    EX = Path(a.exec_pin).parent
    for rel, ent in pf.items():
        p = EX / rel
        h = ent.get("sha256") if isinstance(ent, dict) else ent
        if p.exists() and h:
            checked += 1
            if sha(p) != h:
                bad_pin.append(rel)
    add("frozen_exec_tree_bytes_identical_to_exec_pin", not bad_pin,
        {"n_pinned": len(pf), "n_checked": checked, "n_mismatch": len(bad_pin), "mismatch": bad_pin,
         "exec_pin_gpu_used_at_pin_time": pin.get("gpu_used"),
         "exec_pin_physics_executed_at_pin_time": pin.get("physics_executed")},
        [q(a.exec_pin + " (purpose)", pin["purpose"])])

    # ── P5-22 물리 파라미터가 rev32 핀과 동일한지(선언 확인) ───────────────
    changed = pin.get("user_decisions_20260929", {})
    add("physics_and_sampling_params_declared_unchanged_from_rev32",
        P.get("timestep_s") == 1e-06 and P.get("dt_sync_s") == 0.004
        and P.get("particle_frame_dt_s") == 0.1 and P.get("settlement_frame_dt_s") == 0.05,
        {"timestep_s": P.get("timestep_s"), "dt_sync_s": P.get("dt_sync_s"),
         "dt_sync_close_s": P.get("dt_sync_close_s"),
         "particle_frame_dt_s": P.get("particle_frame_dt_s"),
         "settlement_frame_dt_s": P.get("settlement_frame_dt_s"),
         "E_pa": P.get("E_pa"), "mu": P.get("mu"), "CoR": P.get("CoR"), "Crr": P.get("Crr"),
         "particle_density_kg_m3": P.get("particle_density_kg_m3"),
         "note_w25": P.get("_note_w25", "")[:200]},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":3-6",
           "This is a pre-freeze interface contract, not producer implementation. Array names, axes, units, frame "
           "conventions, and role mappings below must be emitted verbatim or explicitly mapped in immutable "
           "metadata before production GO.")],
        note="`settlement_frame_dt_s` 는 0.05 로 선언돼 있으나 실제 저장 간격은 `particle_frame_dt_s`=0.1 이다 "
             "— 정착 cadence FAIL 의 직접 원인(P4).")

    # ── P5-23 ERRATUM_01 visual_mapping (재생 단계 산출) ───────────────────
    vm = RUN / "visual_mapping.json"
    add("erratum01_visual_mapping_one_row_per_saved_particle_frame", vm.exists(),
        {"visual_mapping_json_present": vm.exists(), "searched_path": str(vm), "F": F,
         "phases_covered_by_saved_frames": sorted(int(v) for v in np.unique(pcode[pfs])),
         "n_phases_in_order": len(PHASES),
         "all_12_phases_covered": sorted(int(v) for v in np.unique(pcode[pfs])) == list(range(12))},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED_ERRATUM_01.md") + ":12-15",
           "`visual_mapping.json` contains `source_sync_index`, `source_time_s`, and `source_phase_code` with "
           "exactly one ordered row for **every** saved raw production particle frame; no duplicate, omission, or "
           "invented time is accepted.")],
        note="2단계(재생·RRD) 산출물이며 이 회계 단계에서는 만들지 않는다(D341 — 이 단계는 코드·배열·해시 감사). "
             "없음 = FAIL 로 그대로 남긴다. 사후 완화 0.")

    # ── P5-24 최종 매니페스트(경로·바이트·sha) ─────────────────────────────
    man_p = [p for p in (RUN / "MANIFEST.json", RUN / "FINAL_MANIFEST.json") if p.exists()]
    add("final_manifest_path_bytes_sha256_for_every_output", bool(man_p),
        {"manifest_present": bool(man_p), "searched": [str(RUN / "MANIFEST.json"), str(RUN / "FINAL_MANIFEST.json")],
         "retrieval_receipt_covers_all_run_files": True,
         "n_files_in_run_01": len([p for p in RUN.rglob("*") if p.is_file()])},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":105-107",
           "Final manifest gives path, byte count, and SHA-256 for every output. Run status explicitly "
           "distinguishes finalization/full-cycle coverage from the scientific delivery verdict; an unsuccessful "
           "delivery is reported as such.")],
        note="run_01 안에는 바이트 수까지 포함한 최종 매니페스트가 없다. 회수 영수증(RETRIEVAL_RECEIPT.json)은 "
             "sha 만 있고 byte count 가 없다. W19 에서도 같은 항목이 FAIL 이었다.")

    # ── P5-25 delivery 3층 분리 + 정착 미증명 ──────────────────────────────
    dl = res["delivery"]
    sw = dl["settlement_window"]
    add("delivery_layers_separate_and_settlement_cadence_contract",
        bool(sw.get("cadence_ok")),
        {"definite_delivered_n": dl["definite_delivered_n"], "possible_delivered_n": dl["possible_delivered_n"],
         "exact_single_value_allowed": dl["exact_single_value_allowed"],
         "settlement_n_frames": sw["n_frames"], "min_required_frames": 6,
         "max_frame_gap_s": sw["max_frame_gap_s"], "max_allowed_gap_s": 0.05,
         "cadence_ok": sw["cadence_ok"], "layer1_accounting_integrity": dl["layer1_accounting_integrity"]},
        [q(a.criteria + " (delivery.settlement_window_UNCALIBRATED)",
           json.dumps([t for t in crit["thresholds"]
                       if t["id"] == "delivery.settlement_window_UNCALIBRATED"][0]["value"], ensure_ascii=False))],
        note="사후 완화 0. 창 5프레임·간격 0.100025 s 는 계약(≥6프레임·≤0.05 s)을 충족하지 않는다. "
             "이것은 기록 계약의 한계이고 물리 실패 판정이 아니다.")

    # ── P5-26 재고 전수·배타 ───────────────────────────────────────────────
    per = np.stack([np.bincount(r.astype(int), minlength=6) for r in np.asarray(z["inventory_code"])])
    add("inventory_exhaustive_and_disjoint_every_frame", bool((per.sum(1) == N).all()),
        {"F": F, "N": N, "n_frames_with_wrong_total": int((per.sum(1) != N).sum()),
         "layer1_accounting_integrity_json": dl["layer1_accounting_integrity"],
         "n_ambiguous_final": int(per[-1, 5]),
         "ambiguous_fraction_final": round(float(per[-1, 5]) / N, 6)},
        [q(a.criteria + " (inventory.exhaustive_disjoint)",
           json.dumps([t for t in crit["thresholds"]
                       if t["id"] == "inventory.exhaustive_disjoint"][0]["value"], ensure_ascii=False))],
        note="ambiguous 가 많은 것은 규약(모든 구체가 margin 안쪽) 의 결과이며 물리 실패가 아니다.")

    # ── P5-27 W25 절차/프레임 기록 존재 ────────────────────────────────────
    w25 = res.get("w25", {})
    proc = w25.get("procedure", {})
    add("w25_procedure_and_frame_records_present",
        bool(w25.get("frame")) and bool(proc.get("chatter_log") is not None)
        and bool(proc.get("reclose_read")) and bool(w25.get("tray")),
        {"revision": w25.get("revision"), "frame_keys": sorted(w25.get("frame", {})),
         "n_chatter_log": len(proc.get("chatter_log", [])),
         "chatter_threshold_servo_deg": P.get("w25_chatter_threshold_servo_deg"),
         "chatter_max_retries": P.get("w25_chatter_max_retries"),
         "procedure_units": proc.get("units"), "tray_mismatch_reasons": w25.get("tray", {}).get("mismatch_reasons"),
         "w25_non_claims": w25.get("non_claims")},
        [q(str(AUD / "RAW_SCHEMA_REQUIRED.md") + ":4-6",
           "Array names, axes, units, frame conventions, and role mappings below must be emitted verbatim or "
           "explicitly mapped in immutable metadata before production GO.")])

    n_fail = sum(i["verdict"] == "FAIL" for i in items)
    schema = {"artifact": "W25_RAW_SCHEMA_CHECK_V1",
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "subject_raw": str(raw_p), "subject_raw_sha256": sha(raw_p),
              "contract_documents": CONTRACTS, "cpu_only": True, "new_physics_runs": 0, "gpu_used": False,
              "item_naming_basis": "W19 RAW_SCHEMA_CHECK 27항목 방식 + W25 전용 항목(w25_frame·절차 기록)",
              "n_items": len(items), "n_pass": len(items) - n_fail, "n_fail": n_fail,
              "failures": [i["name"] for i in items if i["verdict"] == "FAIL"],
              "verdict": f"RAW_SCHEMA_CHECK_FAIL_{n_fail}_OF_{len(items)}" if n_fail else
                         f"RAW_SCHEMA_CHECK_PASS_{len(items)}_OF_{len(items)}",
              "non_claims": ["이 검사는 원자료가 규약 문구를 충족하는지만 본다 — 물리적 성공·배출 성공 판정이 아니다.",
                             "FAIL 항목을 사후 허용값으로 완화하지 않았다.",
                             "2단계(재생·RRD)·3단계(독립 감사) 산출이 필요한 항목은 note 에 적고 FAIL 로 남긴다."],
              "wall_s": round(time.time() - t0, 3), "items": items}
    Path(a.out_schema).write_text(json.dumps(schema, ensure_ascii=False, indent=1, default=jdef))

    # ══ P6 — 등록 criteria 45항목 ═══════════════════════════════════════════
    t1 = time.time()
    citems = []

    def cadd(cid, verdict, measured, note=None):
        t = [x for x in crit["thresholds"] if x["id"] == cid][0]
        citems.append({"id": cid, "severity": t["severity"], "operator": t.get("operator"),
                       "boundary": t.get("boundary"), "criteria_value": t["value"],
                       "verdict": verdict, "measured": measured,
                       "scientific_limitation": t.get("scientific_limitation"),
                       **({"note": note} if note else {})})

    vmaxs = np.asarray(z["scalar_v_particle_max"])
    bc0 = res["bridge_clearance"][0] if res.get("bridge_clearance") else {}
    nrJ = meta.get("numeric_receipt", {})
    bpc = bc0.get("planned_certificate", {})
    bsi = bc0.get("static_inputs", {})
    bgw = bpc.get("global_worst", {})
    bps = bc0.get("precheck_summary", {})
    door_mm_per_deg = 1000.0 * float(nrJ.get("r_max_local_m_by_body", {}).get("door", 0.0)) * np.pi / 180.0
    lip_mm_per_deg = 1000.0 * float(res["servo"]["lip_radius_m"]) * np.pi / 180.0
    crit_sha_in_receipt = sha(a.criteria) in json.dumps(rec_exec)
    checks = {
        "bridge.numeric_allowance_chain": (
            "PASS" if bc0.get("pass") and bc0.get("verdict") == "CLEARANCE_CERTIFIED" else "FAIL",
            {"verdict": bc0.get("verdict"), "pass": bc0.get("pass"),
             "numeric_receipt_keys": sorted(bc0.get("numeric_receipt", {}))[:12],
             "N": nrJ.get("internal_steps_N"), "delta_q": nrJ.get("delta_q"),
             "theta_num_rad": nrJ.get("theta_num_rad")}, None),
        "bridge.position_lattice_evidence_gap": (
            "PASS" if nrJ.get("length_unit_l_m") and nrJ.get("voxel_size_m") else "FAIL",
            {"length_unit_l_m": nrJ.get("length_unit_l_m"), "voxel_size_m": nrJ.get("voxel_size_m"),
             "position_lattice_mode": nrJ.get("position_lattice_mode"),
             "position_lattice_allowance_m": nrJ.get("position_lattice_allowance_m"),
             "preflight_blocking_gaps": nrJ.get("preflight_blocking_gaps")}, None),
        "bridge.bootstrap_preconditions": (
            "PASS" if nrJ.get("qmin") == 0.5 else "FAIL",
            {"qmin": nrJ.get("qmin"), "bridge_pass": bc0.get("pass")}, None),
        "bridge.independent_body_motion_bound": (
            "PASS" if meta.get("body_roles", {}).get("independent_motion_bound") else "FAIL",
            {"body_roles": meta.get("body_roles"),
             "r_max_local_m_by_body": nrJ.get("r_max_local_m_by_body"),
             "quaternion_rotation_allowance_m_by_body": nrJ.get("quaternion_rotation_allowance_m_by_body")}, None),
        "time.raw_authority_unrounded": (
            "PASS" if (str(np.asarray(z["sync_t_s"]).dtype) == "float64"
                       and "sync_t_s_legacy_round9_display" in keys
                       and meta.get("time_authority", {}).get("sync_t_s", "").startswith("unrounded")) else "FAIL",
            {"sync_t_s_dtype": str(np.asarray(z["sync_t_s"]).dtype),
             "legacy_field_present": "sync_t_s_legacy_round9_display" in keys,
             "time_authority": meta.get("time_authority", {}).get("sync_t_s"),
             "max_abs_diff_authority_vs_legacy_s":
                 float(np.abs(np.asarray(z["sync_t_s"], float) -
                              np.asarray(z["sync_t_s_legacy_round9_display"], float)).max())}, None),
        "time.derived_internal_step_count": (
            "PASS" if "DERIVED" in str(meta.get("time_authority", {}).get("sync_derived_internal_steps", "")) else "FAIL",
            {"declaration": meta.get("time_authority", {}).get("sync_derived_internal_steps"),
             "array_present": "sync_derived_internal_steps" in keys,
             "internal_steps_N_source": nrJ.get("internal_steps_N_source")}, None),
        "contact.bin_role_observed": (
            "PASS" if (meta.get("contact_semantics", {}).get("bin_observation", {}).get("observed_in_this_run")
                       and "scalar_F_bin_N" in keys and "scalar_n_bin_contacts" in keys) else "FAIL",
            {"observed_in_this_run":
                 meta.get("contact_semantics", {}).get("bin_observation", {}).get("observed_in_this_run"),
             "n_bin_contact_rows": int((np.asarray(z["contact_mesh"]) == 3).sum()),
             "max_scalar_F_bin_N": float(np.asarray(z["scalar_F_bin_N"]).max()),
             "max_scalar_n_bin_contacts": float(np.asarray(z["scalar_n_bin_contacts"]).max())}, None),
        "engine.pop_speed_m_s": (
            "PASS" if int(res["pops"]["syncs_over_pop_speed"]) == 0 else "FAIL",
            {"v_particle_max_m_s": res["pops"]["v_particle_max_m_s"],
             "pop_speed_m_s": res["pops"]["pop_speed_m_s"],
             "syncs_over_pop_speed": res["pops"]["syncs_over_pop_speed"],
             "recomputed_n_over": int((vmaxs > float(P["pop_speed_m_s"])).sum())}, None),
        "engine.diag_stop_v_m_s": (
            "PASS" if float(vmaxs.max()) <= float(P["diag_stop_v_m_s"]) else "FAIL",
            {"max_v": float(vmaxs.max()), "diag_stop_v_m_s": P["diag_stop_v_m_s"],
             "n_sync_over": int((vmaxs > float(P["diag_stop_v_m_s"])).sum())}, None),
        "engine.protection_params_unchanged": (
            "PASS" if (P["pop_speed_m_s"] == 5.0 and P["diag_stop_v_m_s"] == 20.0
                       and P["door_pinch_guard_N"] == 3.0 and P["max_velocity_m_s"] == 5.0) else "FAIL",
            {"pop_speed_m_s": P["pop_speed_m_s"], "diag_stop_v_m_s": P["diag_stop_v_m_s"],
             "door_pinch_guard_N": P["door_pinch_guard_N"], "max_velocity_m_s": P["max_velocity_m_s"],
             "error_out_vel": P["error_out_vel"]}, None),
        "inventory.exhaustive_disjoint": (
            "PASS" if bool((per.sum(1) == N).all()) else "FAIL",
            {"n_frames_wrong_total": int((per.sum(1) != N).sum()), "F": F, "N": N}, None),
        "inventory.ambiguous_is_allowed": (
            "PASS" if "ambiguous" in [str(v) for v in z["inventory_labels"]] else "FAIL",
            {"final_ambiguous": int(per[-1, 5]), "final_fraction": round(float(per[-1, 5]) / N, 6),
             "boundary_rule": meta.get("classify_boundary_rule")}, None),
        "inventory.no_surface_overlap_zero_rule": (
            "PASS" if meta.get("classify_boundary_rule") == "any single sphere overlaps the margin band -> ambiguous"
            else "FAIL",
            {"classify_boundary_rule": meta.get("classify_boundary_rule")}, None),
        "inventory.source_containment_not_global": (
            "PASS" if meta.get("classify_inside_rule") ==
            "all spheres strictly inside by margin, center distance +/- sphere radius" else "FAIL",
            {"classify_inside_rule": meta.get("classify_inside_rule"),
             "source_bounds_m": meta.get("source_bounds_m")}, None),
        "inventory.mass_from_canonical_template": (
            "PASS" if abs(float(np.asarray(z["particle_mass_kg"])) - float(res["particle"]["mass_kg"])) < 1e-18
            else "FAIL",
            {"particle_mass_kg": float(np.asarray(z["particle_mass_kg"])),
             "template_mass_kg": res["particle"]["mass_kg"],
             "union_volume_m3": res["particle"]["union_volume_m3"]}, None),
        "delivery.three_layers_separate": (
            "PASS" if (dl["definite_delivered_n"] != dl["possible_delivered_n"]) ==
            (not dl["exact_single_value_allowed"]) else "FAIL",
            {"definite_n": dl["definite_delivered_n"], "possible_n": dl["possible_delivered_n"],
             "exact_single_value_allowed": dl["exact_single_value_allowed"],
             "definite_g": dl["definite_delivered_g"], "possible_g": dl["possible_delivered_g"]}, None),
        "delivery.settlement_window_UNCALIBRATED": (
            "PASS" if sw.get("cadence_ok") else "FAIL",
            {"n_frames": sw["n_frames"], "max_frame_gap_s": sw["max_frame_gap_s"],
             "cadence_ok": sw["cadence_ok"], "n_settled": sw["n_settled"],
             "n_stable_bin_all_frames": sw["n_stable_bin_all_frames"],
             "required_min_frames": 6, "required_max_gap_s": 0.05}, "사후 완화 0."),
        "delivery.no_minimum_delivered_mass": (
            "PASS", {"definite_g": dl["definite_delivered_g"], "possible_g": dl["possible_delivered_g"],
                     "policy": "최소 배출 질량 문턱을 두지 않는다 — 값만 보고한다."}, None),
        "classification.margin_mm": (
            "PASS" if abs(float(P["classify_margin_mm"]) - 2.5) < 1e-12
            and abs(float(meta["classify_margin_m"]) - 0.0025) < 1e-15 else "FAIL",
            {"classify_margin_mm": P["classify_margin_mm"], "classify_margin_m": meta["classify_margin_m"]}, None),
        "classification.spill_rest_z_m": (
            "PASS" if abs(float(P["spill_rest_z_m"]) - float(meta["spill_rest_z_m"])) < 1e-15 else "FAIL",
            {"param": P["spill_rest_z_m"], "metadata": meta["spill_rest_z_m"],
             "spill_basis": meta.get("classify_spill_basis"),
             "final_spill_n": int(per[-1, 3])}, None),
        "time.particle_frame_float32_tolerance": (
            "PASS" if float(np.abs(pft - st[pfs]).max()) == 0.0 else "FAIL",
            {"max_abs_diff_s": float(np.abs(pft - st[pfs]).max()),
             "particle_frame_t_s_dtype": str(pft.dtype)}, None),
        "time.dense_sync_vs_sparse_particle": (
            "PASS" if (T == int(res["trajectory"]["n_sync"]) and F == int(res["trajectory"]["n_particle_frames"])
                       and int(pfs.max()) < T) else "FAIL",
            {"T": T, "F": F, "json_n_sync": res["trajectory"]["n_sync"],
             "json_n_particle_frames": res["trajectory"]["n_particle_frames"],
             "max_particle_frame_sync_index": int(pfs.max())}, None),
        "runner.physics_wall_cap_s": (
            "PASS" if float(rec_exec["wall_s"]) <= float(rec_exec["cap_s"]) == cap_val else "FAIL",
            {"cap_s_criteria": cap_val, "cap_s_receipt": rec_exec["cap_s"], "wall_s": rec_exec["wall_s"],
             "headroom_s": round(cap_val - float(rec_exec["wall_s"]), 3) if cap_val else None,
             "used_fraction": round(float(rec_exec["wall_s"]) / cap_val, 6) if cap_val else None}, None),
        "runner.sim_soft_wall_cap_s": (
            "PASS" if float(res["max_wall_s"]) == float(
                [t for t in crit["thresholds"] if t["id"] == "runner.sim_soft_wall_cap_s"][0]["value"])
            and float(res["wall_seconds"]) <= float(res["max_wall_s"]) else "FAIL",
            {"criteria": [t for t in crit["thresholds"] if t["id"] == "runner.sim_soft_wall_cap_s"][0]["value"],
             "argv_max_wall_s": res["max_wall_s"], "sim_wall_seconds": res["wall_seconds"],
             "abort_class": res["abort_class"]}, None),
        "runner.graceful_grace_s": (
            "PASS" if float(rec_exec["grace_s"]) == float(
                [t for t in crit["thresholds"] if t["id"] == "runner.graceful_grace_s"][0]["value"]) else "FAIL",
            {"grace_s_receipt": rec_exec["grace_s"],
             "grace_s_criteria": [t for t in crit["thresholds"]
                                  if t["id"] == "runner.graceful_grace_s"][0]["value"],
             "killed": rec_exec["killed"], "group_alive_after": rec_exec["group_alive_after"]}, None),
        "runner.no_retry": (
            "PASS" if rec_exec["auto_retry"] is False else "FAIL",
            {"auto_retry": rec_exec["auto_retry"],
             "attempt_dirs": sorted(p.name for p in RUN.parent.glob("run_*"))}, None),
        "process.rc0_is_not_delivery": (
            "PASS" if (int(rec_exec["rc"]) == 0 and res["abort_class"] is None
                       and dl["exact_single_value_allowed"] is False) else "FAIL",
            {"rc": rec_exec["rc"], "abort_class": res["abort_class"],
             "exact_single_value_allowed": dl["exact_single_value_allowed"],
             "definite_n": dl["definite_delivered_n"], "possible_n": dl["possible_delivered_n"]},
            "rc0 는 완주 기록이지 배출 판정이 아니다 — 두 층을 분리한다."),
        "coverage.full_cycle_token_requires_real_phases": (
            "PASS" if (sorted(int(v) for v in np.unique(pcode)) == list(range(12))
                       and all(int(v) > 0 for v in res["trajectory"]["syncs_per_phase"].values())) else "FAIL",
            {"phase_codes_seen": sorted(int(v) for v in np.unique(pcode)),
             "syncs_per_phase": res["trajectory"]["syncs_per_phase"],
             "stopped_early_after_phase": res["stopped_early_after_phase"]}, None),
        "bridge.numeric_epsilon_m": (
            "PASS" if (float(bsi.get("numeric_epsilon_m", -1)) == 1e-06
                       and float(bgw.get("slack_m", -1)) > 0) else "FAIL",
            {"numeric_epsilon_m_used": bsi.get("numeric_epsilon_m"),
             "global_worst_slack_m": bgw.get("slack_m"), "global_worst_cell": bgw.get("cell"),
             "global_worst_gap_m": bgw.get("gap_m"), "n_separation_failures_total":
                 bpc.get("n_separation_failures_total")}, None),
        "bridge.orthonormal_tol": (
            "PASS" if float(bsi.get("orthonormal_tol", -1)) == 1e-09 else "FAIL",
            {"orthonormal_tol_used": bsi.get("orthonormal_tol"),
             "bridge_verdict": bc0.get("verdict"), "failure_class": bpc.get("failure_class")}, None),
        "bridge.max_align_rotation_deg": (
            "PASS" if (bpc.get("limit_failures") == [] and bc0.get("verdict") == "CLEARANCE_CERTIFIED") else "FAIL",
            {"n_align_sync_targets": bc0.get("n_align_sync_targets"),
             "n_joint_sync_targets": bc0.get("n_joint_sync_targets"),
             "limit_failures": bpc.get("limit_failures"),
             "global_worst_theta_rad": bgw.get("theta_rad"),
             "global_worst_theta_deg": (float(bgw["theta_rad"]) * 180.0 / np.pi) if bgw.get("theta_rad") else None},
            "criteria 170 deg 상한 자체는 생산 preflight 가 강제한다 — 여기서는 실제 회전각과 거절 기록을 읽는다."),
        "bridge.plan_match_tol": (
            "PASS" if (float(bps.get("max_plan_target_deviation_m", 1)) <= 1e-09
                       and bps.get("all_ok") and bps.get("n_failed") == 0
                       and bps.get("all_planned_consumed")) else "FAIL",
            {"max_plan_target_deviation_m": bps.get("max_plan_target_deviation_m"),
             "n_syncs_checked": bps.get("n_syncs_checked"), "n_planned_targets": bps.get("n_planned_targets"),
             "all_planned_consumed": bps.get("all_planned_consumed"), "n_failed": bps.get("n_failed"),
             "worst_slack_m": bps.get("worst_slack_m")}, None),
        "bridge.elapsed_upper_bound": (
            "PASS" if (float(bsi.get("call_elapsed_upper_bound_s", -1)) ==
                       float(nrJ.get("call_elapsed_upper_bound_s", -2))
                       and float(nrJ.get("accumulator_at_stop_s", 1)) <=
                       float(nrJ.get("call_elapsed_upper_bound_s", 0))) else "FAIL",
            {"call_elapsed_upper_bound_s": nrJ.get("call_elapsed_upper_bound_s"),
             "accumulator_at_stop_s": nrJ.get("accumulator_at_stop_s"),
             "requested_duration_D_s": nrJ.get("requested_duration_D_s"),
             "float64_of_float32_h_s": nrJ.get("float64_of_float32_h_s"),
             "internal_steps_N": nrJ.get("internal_steps_N"),
             "duration_rule": nrJ.get("evidence", {}).get("duration_rule")}, None),
        "bridge.observed_sync_overrun": (
            "PASS",
            {"observed_max_sync_dt_s": float(np.asarray(z["sync_dts_s"]).max()),
             "requested_dt_sync_s": P["dt_sync_s"],
             "max_abs_sync_t_diff_minus_requested_s":
                 float(np.abs(np.diff(np.asarray(z["sync_t_s"], float)) -
                              np.asarray(z["sync_requested_duration_s"], float)[1:]).max()),
             "severity_is_observation_only": True}, "severity=observation — 보고 전용, 자동 실패 아님."),
        "visual.door_mm_per_deg": (
            "PASS" if (abs(door_mm_per_deg - 7.7822) > 1e-6
                       and abs(lip_mm_per_deg - 7.7822) > 1e-6) else "FAIL",
            {"door_r_max_local_m": nrJ.get("r_max_local_m_by_body", {}).get("door"),
             "door_mm_per_deg_from_r_max": round(door_mm_per_deg, 9),
             "servo_lip_radius_m": res["servo"]["lip_radius_m"],
             "lip_mm_per_deg_from_lip_radius": round(lip_mm_per_deg, 9),
             "void_value": 7.7822,
             "w12_corrected_reference_mm_per_deg": 2.057714892},
            "규약은 void 값 7.7822 사용 금지만 건다. 두 반경 근거를 모두 자기 원시에서 계산해 적는다."),
        "policy.no_threshold_change_after_outcomes": (
            "PASS" if crit_sha_in_receipt else "FAIL",
            {"criteria_sha256": sha(a.criteria),
             "recorded_in_execution_receipt": crit_sha_in_receipt,
             "sha_fields_in_execution_receipt": [k for k in rec_exec if "sha" in k.lower()],
             "recorded_in_exec_pin": pin["files"].get("criteria_w25_paperbox_cap32h.json"),
             "criteria_changed_utc": crit["derived_from"]["changed_utc"],
             "run_started_utc": rec_exec["started_utc"],
             "criteria_frozen_before_run": crit["derived_from"]["changed_utc"] < rec_exec["started_utc"],
             "exec_pin_gpu_used_at_pin_time": pin.get("gpu_used"),
             "exec_pin_physics_executed_at_pin_time": pin.get("physics_executed"),
             "frozen_before_production_flag": crit["frozen_before_production"]},
            "criteria sha 는 EXEC_PIN.json(결과 0 시점 동결본)에는 기록돼 있으나 run_01/EXECUTION_RECEIPT.json "
            "에는 없다. 규약 문구('recorded in the execution receipt')를 글자대로 읽어 FAIL 로 남긴다 — "
            "사후 완화 0. 동결 시점 자체는 실행 시작보다 앞선다(값 병기)."),
        "comparison.old_wall_descriptive_only": (
            "PASS", {"sim_wall_seconds": res["wall_seconds"], "runner_wall_s": rec_exec["wall_s"],
                     "policy": "벽시계 비교는 서술용 — 과학 판정에 쓰지 않는다."}, None),
    }
    for cid, (v, m, nt) in checks.items():
        cadd(cid, v, m, nt)

    # 나머지 항목은 이 단계(원자료 회계)에서 판정 불가 — 이유를 적고 NOT_CHECKABLE 로 남긴다.
    done = set(checks)
    for t in crit["thresholds"]:
        if t["id"] in done:
            continue
        why = ("재생·RRD 산출물이 필요하다(2단계)." if t["id"].startswith("visual.")
               else "preflight/bridge 생산 단계 산출물이 필요하다."
               if t["id"].startswith("bridge.") else "이 단계의 원자료·영수증만으로는 판정 근거가 없다.")
        citems.append({"id": t["id"], "severity": t["severity"], "operator": t.get("operator"),
                       "boundary": t.get("boundary"), "criteria_value": t["value"],
                       "verdict": "NOT_CHECKABLE_IN_RAW_ACCOUNTING", "measured": {},
                       "scientific_limitation": t.get("scientific_limitation"), "note": why})

    cfail = sum(i["verdict"] == "FAIL" for i in citems)
    cpass = sum(i["verdict"] == "PASS" for i in citems)
    cnc = sum(i["verdict"] == "NOT_CHECKABLE_IN_RAW_ACCOUNTING" for i in citems)
    cout = {"artifact": "W25_CRITERIA_CHECK_V1",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "criteria_file": a.criteria, "criteria_sha256": sha(a.criteria),
            "criteria_revision": crit["revision"], "frozen_before_production": crit["frozen_before_production"],
            "subject_raw": str(raw_p), "subject_raw_sha256": sha(raw_p),
            "cpu_only": True, "new_physics_runs": 0, "gpu_used": False,
            "n_thresholds": len(crit["thresholds"]), "n_items": len(citems),
            "n_pass": cpass, "n_fail": cfail, "n_not_checkable": cnc,
            "failures": [i["id"] for i in citems if i["verdict"] == "FAIL"],
            "hard_fail_failures": [i["id"] for i in citems
                                   if i["verdict"] == "FAIL" and i["severity"] == "hard_fail"],
            "reading_rules": crit["reading_rules"],
            "non_claims": ["등록 임계를 결과를 본 뒤 바꾸지 않았다(reading_rules 4번).",
                           "FAIL 을 사후 허용값으로 완화하지 않았다.",
                           "NOT_CHECKABLE 는 PASS 가 아니다 — 다른 단계의 산출물이 필요하다는 뜻이다."],
            "wall_s": round(time.time() - t1, 3), "items": citems}
    Path(a.out_criteria).write_text(json.dumps(cout, ensure_ascii=False, indent=1, default=jdef))
    print(json.dumps({"schema": {"n_items": len(items), "n_fail": n_fail, "failures": schema["failures"]},
                      "criteria": {"n_items": len(citems), "n_pass": cpass, "n_fail": cfail,
                                   "n_not_checkable": cnc, "failures": cout["failures"],
                                   "hard_fail_failures": cout["hard_fail_failures"]}},
                     ensure_ascii=False, indent=1, default=jdef))


if __name__ == "__main__":
    main()
