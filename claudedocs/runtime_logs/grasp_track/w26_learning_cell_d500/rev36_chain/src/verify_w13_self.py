"""W13 자기검증기 — 읽기 전용. 산출이 없거나 미완이면 반드시 거부한다.

usage:
    python verify_w13_self.py --check        [--run <run_dir>]   -> W13_SELF_CHECK_OK
    python verify_w13_self.py --check-visual  [--run <run_dir>]   -> W13_VISUAL_CHECK_OK

--check        연속 사이클이 동결 phase 열을 실제로 돌았는가 · NPZ restart 가 아닌가 · 전 입자 회계가 맞는가
--check-visual 완결된 RRD/RBL 과 전 사이클 Isaac 산출이 원자료 매핑 계약을 지키는가

**생산자 자기검사다. 감사 oracle 이 아니다**(계약 §8). 어떤 파일도 쓰지 않는다.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w13_kinematics as K                                             # noqa: E402

MAIN_REPO = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(MAIN_REPO))
import sim_deme_scoop_s1 as W11SRC                                     # noqa: E402

PILE_SHA256 = "659d6b0bc771678a0c7209d91f550edc933d03e41922245ea0adb64eeb818812"
SRC_SHA256 = "2e40f7ed279dad42794d156c3e5e823d6ce0d8cb770ec9aad1b9280a33a7e933"
PHASES = ["initial_home", "settle", "approach", "descend", "close", "lift", "reclose",
          "transport", "discharge", "discharge_wait", "close_after_discharge", "return_home"]
DECISIONS = ["initial_home_end", "settle_end", "approach_end", "descend_end", "close_stop", "lift_end",
             "reclose_end", "bridge_clearance_decision", "transport_end", "release_before", "release_after",
             "wait_end", "close_after_discharge_end", "return_home_end"]
BRIDGE_SEGMENTS = ("actual_to_resume_align", "resume_to_post_lift_joint")
BRIDGE_LEGS = ("align_from_actual_pose", "align_from_commanded_pose")
INV_NAMES = ["source", "receiving_bin", "tool_residual", "spill", "in_flight", "ambiguous"]


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail):
        self.rows.append({"check": name, "pass": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def ok(self):
        return bool(self.rows) and all(r["pass"] for r in self.rows)

    def dump(self):
        for r in self.rows:
            print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['check']}: {r['detail']}")


def _need(c, path, name):
    p = Path(path)
    return c.add(name, p.exists() and p.stat().st_size > 0, str(path))


def load_run(run, c):
    j, n = run / "w13_cycle_seed460.json", run / "w13_cycle_seed460.npz"
    for p, nm in ((j, "artifact_result_json"), (n, "artifact_raw_npz")):
        if not _need(c, p, nm):
            return None, None
    return json.load(open(j)), np.load(n, allow_pickle=True)


def check_science(run, pile_sha256=PILE_SHA256):
    """rev34: pile_sha256 인자 추가(기본 = rev32 동결 더미). W25-C 새 더미를 쓰면 그 sha 를 명시해 넘긴다."""
    c = Checks()
    res, z = load_run(run, c)
    if res is None:
        return c, None

    c.add("run_completed_full_cycle",
          (not res["diverged"]) and res.get("stopped_early_after_phase") is None
          and res.get("abort_class") is None
          and not res["smoke"] and res.get("smoke_max_particles") is None,
          f"diverged={res['diverged']} stopped_early={res.get('stopped_early_after_phase')} "
          f"abort_class={res.get('abort_class')} "
          f"smoke={res['smoke']} max_particles={res.get('smoke_max_particles')} fail={res.get('fail_reason')}")

    h = res.get("inputs_sha256", {})
    pile_k = [k for k in h if k.endswith(".npz")]
    c.add("input_pile_frozen", len(pile_k) == 1 and h[pile_k[0]] == pile_sha256,
          f"{[h.get(k) for k in pile_k]} vs {pile_sha256}")
    src_k = [k for k in h if k.endswith("sim_deme_scoop_s1.py")]
    c.add("frozen_source_unmodified", len(src_k) == 1 and h[src_k[0]] == SRC_SHA256,
          f"{h.get(src_k[0]) if src_k else None} vs {SRC_SHA256}")
    bad = [k for k in h if ("w10_" in k or "w11_" in k or "w12_" in k)]
    c.add("no_previous_output_as_input", not bad, f"{bad or 'none'}")
    c.add("no_restart_field", "restart_from" not in res and "restart_from" not in list(z.keys()),
          "결과 JSON/NPZ 에 restart 입력 필드 없음")

    # ── 동결 phase 열이 압축했을 때 정확히 같은가 ─────────────────────────
    pc = np.asarray(z["sync_phase_code"], int)
    comp = [PHASES[int(pc[0])]]
    for v in pc[1:]:
        if PHASES[int(v)] != comp[-1]:
            comp.append(PHASES[int(v)])
    c.add("phase_sequence_exact", comp == PHASES, f"{comp}")
    per = {p: int((pc == i).sum()) for i, p in enumerate(PHASES)}
    c.add("all_phases_have_syncs", all(v > 0 for v in per.values()), json.dumps(per, ensure_ascii=False))

    # ── 시간 연속성 ───────────────────────────────────────────────────────
    t = np.asarray(z["sync_t_s"], float)
    dts = np.asarray(z["sync_dts_s"], float)
    w = np.asarray(z["sync_wall_elapsed_s"], float)
    gaps = np.diff(t)
    max_sync = float(max(res["params"]["dt_sync_s"], res["params"]["dt_sync_descend_s"],
                         res["params"].get("dt_sync_close_s") or 0.0))
    c.add("solver_time_nondecreasing", bool((gaps >= -1e-12).all()), f"min gap {gaps.min():.3e} s")
    c.add("no_time_gap", bool(gaps.max() <= max_sync + 1e-9), f"max gap {gaps.max():.6f} ≤ {max_sync}")
    c.add("wall_time_nondecreasing", bool((np.diff(w) >= -1e-9).all()), f"wall {w[0]:.3f}→{w[-1]:.3f} s")
    c.add("t0_is_zero_dts_home", bool(abs(t[0]) < 1e-12 and abs(dts[0]) < 1e-12 and PHASES[int(pc[0])] == "initial_home"),
          f"t0={t[0]} dts0={dts[0]} phase0={PHASES[int(pc[0])]}")
    c.add("sum_of_syncs_matches_sim_time", bool(abs(dts.sum() - t[-1]) < 1e-3),
          f"Σdts={dts.sum():.6f} vs t_end={t[-1]:.6f}")

    # ── 두 시간축 연결 ────────────────────────────────────────────────────
    pf_t = np.asarray(z["particle_frame_t_s"], float)
    pf_s = np.asarray(z["particle_frame_sync_index"], int)
    ok_map = bool((pf_s >= -1).all() and (pf_s < len(t)).all()
                  and np.allclose(pf_t[pf_s >= 0], t[pf_s[pf_s >= 0]], atol=1e-9))
    c.add("particle_frames_map_to_syncs", ok_map, f"{len(pf_t)} frames, index range [{pf_s.min()}, {pf_s.max()}]")
    tr = np.asarray(z["transition_sync_index"], int)
    # rev30 (ERRATUM_04 §2): 전환 i 의 경계 상태 프레임은 직전 행 i-1 에 있어야 한다(정확히 i 는 선택).
    c.add("transitions_have_boundary_frames", bool(set(int(v) - 1 for v in tr) <= set(int(v) for v in pf_s)),
          f"{len(tr)} transitions, 전부 i-1 경계 입자 프레임을 갖는가")
    # rev29: 규약 RAW_SCHEMA_REQUIRED.md:52 — dense phase 코드가 바뀐 행과 정확히 같아야 한다(행 0·subphase 제외).
    _pc = np.asarray(z["sync_phase_code"]).astype(int)
    _exp = (np.flatnonzero(_pc[1:] != _pc[:-1]) + 1).tolist()
    c.add("transitions_phase_only_exact", tr.tolist() == _exp,
          f"recorded {tr.tolist()} vs phase-change rows {_exp}")
    fp = {int(pc[min(int(s_), len(pc) - 1)]) for s_ in pf_s if s_ >= 0}
    c.add("particle_frames_cover_all_phases", set(range(len(PHASES))) <= fp, f"{sorted(fp)}")

    # ── 회계: 배타·전수·독립 재계산 ────────────────────────────────────────
    n_p = int(res["particle"]["n"])
    inv = np.asarray(z["inventory_code"])
    ids = np.asarray(z["particle_ids"], int)
    c.add("particle_ids_unique_and_constant",
          bool(len(np.unique(ids)) == n_p and inv.shape[1] == n_p
               and np.asarray(z["particle_pos_m"]).shape[1] == n_p),
          f"n={n_p}, inv{inv.shape}, pos{np.asarray(z['particle_pos_m']).shape}")
    sums = inv.shape[1] - np.array([np.bincount(r.astype(int), minlength=6).sum() for r in inv])
    c.add("inventory_exhaustive_every_frame", bool((sums == 0).all()), f"프레임별 합계 이탈 {int(np.abs(sums).max())}")
    dt_tags = [str(v) for v in np.asarray(z["decision_tags"])]
    c.add("decision_tags_exact", dt_tags == DECISIONS, f"{dt_tags}")

    # 독립 재분류(면분할 용기 + 회전 공동)
    P = res["params"]
    bi = res["fixtures"]["bin"]
    box = np.asarray(z["box_bounds_m"], float)
    RW = W11SRC.R_W
    L5 = np.array(P["lip_l5_mm"], float) / 1000.0
    C5 = np.array(P["bowl_center_l5_mm"], float) / 1000.0
    n_th = int(P["bin_n_theta"])
    th = np.linspace(0, 2 * math.pi, n_th, endpoint=False) + math.pi / n_th
    nrm = np.stack([np.cos(th), np.sin(th)], 1)
    apo = bi["inner_r_m"] * math.cos(math.pi / n_th)
    bc = np.asarray(bi["center_xy_m"], float)
    marg = float(P["classify_margin_mm"]) / 1000.0
    v_mov = 9.81 * float(P["dt_sync_s"])
    box_top = float(box[2, 1])

    # 독립 oracle 도 **회전된 알 전체 형상**으로 판정한다(감사 `RAW_CALLSITE_REVIEW_01.md`).
    # 템플릿은 생산자가 쓴 `res["particle"]`(= shape_info) 에서 되살린다. 구 더미면 단일 구로 떨어진다.
    pt = res["particle"]
    tpl_v = ({"sphere_radii_m": pt["sphere_radii_m"], "offsets_m": pt["offsets_m"]}
             if ("sphere_radii_m" in pt and "offsets_m" in pt) else None)
    k_sph_v = 1 if tpl_v is None else len(tpl_v["sphere_radii_m"])
    rad_v = (float(pt["radius_m"]) if "radius_m" in pt
             else float(pt["bounding_diameter_m"]) / 2.0)

    def reclass(pp, vv, oq, p_f, q_f):
        R_f = K.quat_xyzw_to_mat(q_f)
        n = len(pp)
        if tpl_v is None:
            S = np.asarray(pp, float).reshape(n, 1, 3)
            Rr = np.full((n, 1), rad_v)
        else:
            sp, sr = W11SRC.expand_spheres(pp, oq, tpl_v)       # clump-major, 생산자와 같은 전개식
            S = np.asarray(sp, float).reshape(n, k_sph_v, 3)
            Rr = np.asarray(sr, float).reshape(n, k_sph_v)
        p5 = (RW.T @ (R_f.T @ (S.reshape(-1, 3) - p_f).T)).T + L5
        p5 = p5.reshape(n, k_sph_v, 3)
        r_c = np.hypot(p5[:, :, 0] - C5[0], p5[:, :, 2] - C5[2])
        y_c = np.abs(p5[:, :, 1])
        r_in, hy = P["bowl_r_in_mm"] / 1000.0, P["cheek_half_y_mm"] / 1000.0
        code = np.full(n, 5, np.int8)
        mov = np.linalg.norm(vv, axis=1) >= v_mov
        rest = ~mov
        it = (((r_c + Rr) < r_in - marg) & ((y_c + Rr) < hy - marg)).all(1)
        nt = (((r_c - Rr) < r_in + marg) & ((y_c - Rr) < hy + marg)).any(1)
        dm = ((S[:, :, :2] - bc) @ nrm.T).max(2)
        ib = (((dm + Rr) < apo - marg) & ((S[:, :, 2] - Rr) > bi["floor_inner_z_m"] - marg)   # rev30 ERRATUM_04
              & ((S[:, :, 2] + Rr) < bi["rim_z_m"] - marg)).all(1)
        nb = (((dm - Rr) < apo + marg) & ((S[:, :, 2] + Rr) > bi["floor_inner_z_m"] - marg)
              & ((S[:, :, 2] - Rr) < bi["rim_z_m"] + marg)).any(1)
        isr = (((S[:, :, 0] - Rr) > box[0, 0] + marg) & ((S[:, :, 0] + Rr) < box[0, 1] - marg)
               & ((S[:, :, 1] - Rr) > box[1, 0] + marg) & ((S[:, :, 1] + Rr) < box[1, 1] - marg)
               & ((S[:, :, 2] - Rr) > box[2, 0] - marg) & ((S[:, :, 2] + Rr) < box_top - marg)).all(1)   # rev30 ERRATUM_04 support floor
        nsr = (((S[:, :, 0] + Rr) > box[0, 0] - marg) & ((S[:, :, 0] - Rr) < box[0, 1] + marg)
               & ((S[:, :, 1] + Rr) > box[1, 0] - marg) & ((S[:, :, 1] - Rr) < box[1, 1] + marg)
               & ((S[:, :, 2] + Rr) > box[2, 0] - marg) & ((S[:, :, 2] - Rr) < box_top + marg)).any(1)
        left = np.ones(n, bool)
        code[it] = 2; left &= ~it
        m = left & nt; code[m] = 5; left &= ~nt
        m = left & ib; code[m & rest] = 1; code[m & mov] = 4; left &= ~m
        m = left & nb; code[m] = 5; left &= ~m
        m = left & isr; code[m & rest] = 0; code[m & mov] = 4; left &= ~m
        m = left & nsr; code[m] = 5; left &= ~m
        m = left & mov; code[m] = 4; left &= ~m
        below = (S[:, :, 2] + Rr).max(1) <= P["spill_rest_z_m"]
        code[left & below] = 3
        code[left & ~below] = 5
        return code

    d_sync = np.asarray(z["decision_sync_index"], int)
    d_fi = np.asarray(z["decision_particle_frame_index"], int)
    mism = []
    for k in range(len(d_tags := dt_tags)):
        fi, si = int(d_fi[k]), int(d_sync[k])
        rc = reclass(np.asarray(z["particle_pos_m"][fi], float), np.asarray(z["particle_vel_m_s"][fi], float),
                     np.asarray(z["particle_quat_xyzw"][fi], float),
                     np.asarray(z["tool_pos_m"][si], float), np.asarray(z["tool_quat_xyzw"][si], float))
        mism.append(int((rc != np.asarray(inv[fi])).sum()))
    c.add("inventory_independently_reproduced", max(mism) == 0, f"{dict(zip(d_tags, mism))}")

    # ── 전이 행렬 재계산 ──────────────────────────────────────────────────
    ok_tr = True
    det = []
    for k, trr in enumerate(res["transitions"]):
        M = np.asarray(trr["matrix"], int)
        rec = np.zeros((6, 6), int)
        np.add.at(rec, (np.asarray(inv[int(d_fi[k])], int), np.asarray(inv[int(d_fi[k + 1])], int)), 1)
        good = bool(M.sum() == n_p and np.array_equal(M, rec))
        ok_tr &= good
        det.append({"from": trr["from"], "to": trr["to"], "moved": trr["moved"], "match": good})
    c.add("transitions_consistent", ok_tr, json.dumps(det, ensure_ascii=False))

    # ── 규정 운동 / 강체 힌지 ─────────────────────────────────────────────
    le = float(res["trajectory"]["max_lip_track_err_mm"])
    he = float(res["trajectory"]["max_hinge_track_err_mm"])
    dv = float(res["trajectory"]["max_door_rigid_vel_resid_mm_s"])
    c.add("prescribed_pose_tracking", le < 1e-2 and he < 1e-2, f"lip {le} mm · hinge {he} mm (<0.01)")
    c.add("door_rigidly_coupled", dv < 1.0, f"max |v_door - (v_tool + w x r)| = {dv} mm/s")
    src = (HERE / "sim_w13_full_cycle.py").read_text()
    c.add("no_teleport_in_source", "trk_f.SetPos(" not in src and "trk_d.SetPos(" not in src
          and "trk_f.SetOriQ(" not in src and "trk_d.SetOriQ(" not in src,
          "구현 소스에 트래커 SetPos/SetOriQ 순간이동 호출 없음")

    # ── HOME 왕복 ─────────────────────────────────────────────────────────
    hr = float(res["trajectory"]["home_pose_return_err_mm"])
    c.add("home_start_equals_end", hr < 1.0, f"HOME 복귀 오차 {hr} mm")
    c.add("door_closed_at_both_homes",
          abs(float(z["door_actual_deg"][0])) < 0.5 and abs(float(z["door_actual_deg"][-1])) < 0.5,
          f"door {float(z['door_actual_deg'][0]):.4f}° → {float(z['door_actual_deg'][-1]):.4f}°")

    # ── 문 정지 기록 ──────────────────────────────────────────────────────
    stops = {s["subphase"]: s["reason"] for s in res["door"]["stops"]}
    c.add("door_stops_recorded",
          all(k2 in stops for k2 in ("close", "door_open_at_approach", "close_after_discharge")),
          json.dumps(res["door"]["stops"], ensure_ascii=False))

    # ── 배출 판정(하한/상한 + 정착 관측창) ────────────────────────────────
    d = res["delivery"]
    sw = d.get("settlement_window", {})
    c.add("settlement_window_cadence", bool(sw.get("cadence_ok")),
          f"frames={sw.get('n_frames')} max_gap={sw.get('max_frame_gap_s')} s")
    c.add("delivery_reported_as_interval",
          ("definite_delivered_n" in d and "possible_delivered_n" in d
           and d["possible_delivered_n"] >= d["definite_delivered_n"]),
          f"definite {d.get('definite_delivered_n')} ≤ possible {d.get('possible_delivered_n')} "
          f"({d.get('definite_delivered_g')} g / {d.get('possible_delivered_g')} g)")
    c.add("accounting_integrity_layer1", bool(d.get("layer1_accounting_integrity")),
          "모든 결정 시점에서 6분류 합 = 전체 입자 수")

    # ── rev11 bridge 사전 인증 (V2: 정확한 per-sync 명령열 + 출처 근거 잔차) ─────
    bcs = res.get("bridge_clearance") or []
    c.add("bridge_certificate_present", len(bcs) == 1, f"{len(bcs)} certificate(s)")
    if len(bcs) == 1:
        bc = bcs[0]
        pcert = bc.get("planned_certificate") or {}
        pre = bc.get("precheck_summary") or {}
        si = bc.get("static_inputs") or {}
        c.add("bridge_certified", bc.get("verdict") == "CLEARANCE_CERTIFIED" and bool(bc.get("pass")),
              f"{bc.get('verdict')} pass={bc.get('pass')} abort={bc.get('abort_detail')}")
        c.add("bridge_zero_physics_before_decision", bool(bc.get("certify_consumed_zero_physics")),
              f"steps before/after certify = {bc.get('physics_steps_before_certify')}/"
              f"{bc.get('physics_steps_after_certify')}")
        c.add("bridge_planned_certificate_clean",
              int(pcert.get("n_separation_failures_total", 1)) == 0
              and not pcert.get("limit_failures")
              and not ((pcert.get("joint_radius_cross_check") or {}).get("violations")),
              f"sep={pcert.get('n_separation_failures_total')} "
              f"limits={len(pcert.get('limit_failures') or [])} "
              f"rho_violations={len((pcert.get('joint_radius_cross_check') or {}).get('violations') or [])}")
        gw = pcert.get("global_worst") or {}
        c.add("bridge_worst_slack_positive", float(gw.get("slack_m", -1)) > 0.0,
              f"worst slack {gw.get('slack_m')} m @ {gw.get('segment')}[{gw.get('index')}] "
              f"cell={gw.get('cell')} axis={gw.get('axis')} gap={gw.get('gap_m')} "
              f"bound={gw.get('motion_bound_m')} eps={gw.get('numeric_epsilon_m')}")
        c.add("bridge_exact_command_sequence_certified",
              int(pcert.get("n_intervals", -1)) == int(bc.get("n_align_sync_targets", 0))
              + int(bc.get("n_joint_sync_targets", 0)),
              f"n_intervals {pcert.get('n_intervals')} == align {bc.get('n_align_sync_targets')} "
              f"+ joint {bc.get('n_joint_sync_targets')} sync targets")
        c.add("bridge_precheck_all_ok", bool(pre.get("all_ok")) and bool(pre.get("all_planned_consumed"))
              and int(pre.get("n_failed", 1)) == 0,
              json.dumps(pre, ensure_ascii=False, default=float))
        c.add("bridge_precheck_covered_every_sync",
              int(pre.get("n_syncs_checked", -1)) == int(pre.get("n_planned_targets", -2))
              == int(bc.get("n_align_sync_targets", 0)) + int(bc.get("n_joint_sync_targets", 0)),
              f"checked {pre.get('n_syncs_checked')} / planned {pre.get('n_planned_targets')}")
        c.add("bridge_precheck_worst_slack_positive", float(pre.get("worst_slack_m") or -1) > 0.0,
              f"per-sync worst slack {pre.get('worst_slack_m')} m")
        c.add("bridge_decision_recorded_z_reached", bc.get("z_reached_m") is not None,
              f"z_reached_m={bc.get('z_reached_m')} sim_t={bc.get('sim_t_s')} sync={bc.get('sync_index')}")
        # 원시 배열 독립 재계산 — 계획 구간
        rows = np.asarray(z["bridge_interval_rows"], float)
        cols = [str(x) for x in np.asarray(z["bridge_interval_columns"])]
        nm_raw = [str(x) for x in np.asarray(z["bridge_interval_names"])]
        ix = {k2: cols.index(k2) for k2 in ("gap_m", "motion_bound_m", "numeric_epsilon_m", "slack_m")}
        c.add("bridge_raw_intervals_saved",
              len(nm_raw) == int(pcert.get("n_intervals", -1)) * 2 and len(nm_raw) > 0,
              f"{len(nm_raw)} raw rows vs {int(pcert.get('n_intervals', -1)) * 2} expected (2 parts/interval)")
        c.add("bridge_raw_slack_matches_formula",
              bool(rows.size) and float(np.nanmax(np.abs(
                  (rows[:, ix["gap_m"]] - rows[:, ix["motion_bound_m"]] - rows[:, ix["numeric_epsilon_m"]])
                  - rows[:, ix["slack_m"]]))) < 1e-12,
              "slack == gap - bound - epsilon (원시 배열 독립 재계산)")
        c.add("bridge_raw_all_slack_positive", bool(rows.size) and float(rows[:, ix["slack_m"]].min()) > 0.0,
              f"원시 최소 slack {float(rows[:, ix['slack_m']].min()) if rows.size else None} m")
        c.add("bridge_raw_segments_exact",
              set(n.split("|")[0] for n in nm_raw) == {"align_sync_targets", "joint_sync_targets"},
              f"{sorted(set(n.split('|')[0] for n in nm_raw))}")
        # 원시 배열 독립 재계산 — per-sync precheck
        prows = np.asarray(z["bridge_precheck_rows"], float)
        pcols = [str(x) for x in np.asarray(z["bridge_precheck_columns"])]
        pj = {k2: pcols.index(k2) for k2 in ("ok", "worst_slack_m", "observed_residual_pos_m",
                                             "modeled_residual_pos_m", "observed_residual_rot_rad",
                                             "modeled_residual_rot_rad", "plan_pos_dev_m")}
        c.add("bridge_precheck_rows_saved",
              len(prows) == int(pre.get("n_syncs_checked", -1)) and len(prows) > 0,
              f"{len(prows)} precheck rows vs {pre.get('n_syncs_checked')}")
        c.add("bridge_precheck_rows_all_ok", bool(prows.size) and float(prows[:, pj["ok"]].min()) == 1.0,
              f"ok 최소 {float(prows[:, pj['ok']].min()) if prows.size else None}")
        eps_c = float(si.get("numeric_epsilon_m", 0.0))
        c.add("bridge_observed_residual_within_source_model",
              bool(prows.size) and bool((prows[:, pj["observed_residual_pos_m"]]
                                         <= prows[:, pj["modeled_residual_pos_m"]] + eps_c).all())
              and bool((prows[:, pj["observed_residual_rot_rad"]]
                        <= prows[:, pj["modeled_residual_rot_rad"]] + eps_c).all()),
              f"관측 잔차 최대 {float(prows[:, pj['observed_residual_pos_m']].max())*1000:.9f} mm "
              f"vs 모형 최대 {float(prows[:, pj['modeled_residual_pos_m']].max())*1000:.9f} mm" if prows.size
              else "precheck 행 없음")
        c.add("bridge_plan_matched_live_targets",
              bool(prows.size) and float(prows[:, pj["plan_pos_dev_m"]].max()) <= 1e-9,
              f"재현 명령열 ↔ 실제 명령 목표 최대 차 "
              f"{float(prows[:, pj['plan_pos_dev_m']].max()):.3e} m" if prows.size else "행 없음")
        # 설치본 유도 경과시간 상한이 실제 run **전 구간**에서 지켜졌는가.
        # 상한식: A_final(D) ≤ nextafter(D + float64(float32(h)), +inf) — 감사 DEME_SOURCE_BOUND.md.
        # 이 검사는 사후 확인(진단)이며, pre-step 안전 증명은 precheck 의 분리 판정이 맡는다.
        tt = np.asarray(z["sync_t_s"], float)
        dd = np.asarray(z["sync_dts_s"], float)
        over = np.diff(tt) - dd[1:]
        h_ts = float(si.get("timestep_s", 0.0))
        h32 = float(np.float32(h_ts)) if h_ts > 0 else 0.0
        ub = np.array([math.nextafter(float(d) + h32, math.inf) - float(d) for d in dd[1:]]) \
            if (over.size and h32 > 0) else np.zeros(0)
        c.add("sync_elapsed_within_installed_source_upper_bound",
              bool(over.size) and bool((over <= ub + 1e-15).all()) and float(over.min()) >= -1e-15,
              f"sync 경과시간 초과 min {float(over.min()):.9e} max {float(over.max()):.9e} s; "
              f"설치본 유도 상한 최대 {float(ub.max()) if ub.size else None:.9e} s "
              f"(float32(h)={h32:.9e}); 위반 {int((over > ub + 1e-15).sum()) if ub.size else None}건")

    summary = {"sim_time_s": res["trajectory"]["sim_time_s"], "wall_seconds": res["wall_seconds"],
               "n_sync": int(len(t)), "n_particle_frames": int(len(pf_t)),
               "syncs_per_phase": per, "inventory_final": d.get("inventory_final"),
               "definite_delivered_g": d.get("definite_delivered_g"),
               "possible_delivered_g": d.get("possible_delivered_g"),
               "exact_single_value_allowed": d.get("exact_single_value_allowed"),
               "door_stops": res["door"]["stops"], "max_lip_track_err_mm": le,
               "align_moves": res["trajectory"].get("align_moves"),
               "abort_class": res.get("abort_class"),
               "bridge_verdicts": [b.get("verdict") for b in bcs],
               "bridge_planned_worst_slack_m": ((bcs[0].get("planned_certificate") or {}).get("global_worst")
                                                or {}).get("slack_m") if bcs else None,
               "bridge_precheck_summary": bcs[0].get("precheck_summary") if bcs else None,
               "ik_warning_waypoints": res["frames"].get("ik_warning_waypoints")}
    return c, summary


def check_visual(run):
    c = Checks()
    res, z = load_run(run, c)
    if res is None:
        return c, None
    rr_dir = run / "rerun"
    rrd, rbl = rr_dir / "w13.rrd", rr_dir / "w13.rbl"
    val, insp = rr_dir / "w13_rerun_validation.json", rr_dir / "inspection.json"
    png = rr_dir / "w13_inspection.png"
    for p, nm in ((rrd, "rrd_exists"), (rbl, "rbl_exists"), (val, "rerun_validation_exists"),
                  (png, "headless_screenshot_exists"), (insp, "inspection_record_exists")):
        if not _need(c, p, nm):
            return c, None
    v = json.load(open(val))
    c.add("rerun_contract_pass", bool(v.get("pass")), f"validate_rerun_artifact pass={v.get('pass')}")
    c.add("rerun_version_0_34_1",
          str(v.get("log_status_summary", {}).get("rerun_sdk_version") or v.get("rerun_version")) == "0.34.1",
          json.dumps(v.get("log_status_summary", {}), ensure_ascii=False))
    c.add("sink_finalized", bool(v.get("log_status_summary", {}).get("sink_finalized")),
          "파일 싱크가 첫 로그 전에 붙고 종료로 finalize 됐는가")
    c.add("coverage_full_timeline",
          int(v.get("coverage", {}).get("syncs_logged", -1)) == int(np.asarray(z["sync_t_s"]).shape[0]),
          json.dumps(v.get("coverage", {}), ensure_ascii=False))
    ents = set(v.get("entity_paths") or v.get("entities") or [])
    bridge_ents = {"/bridge/current_pose", "/bridge/proposed_target", "/bridge/align_path",
                   "/bridge/joint_path", "/bridge/worst_cell", "/bridge/worst_separating_plane",
                   "/events/bridge_clearance"}
    c.add("rerun_bridge_decision_entities", bridge_ents <= ents if ents else False,
          f"누락 {sorted(bridge_ents - ents)}" if ents else "검증 JSON 에 entity 목록 없음")
    ins = json.load(open(insp))
    paths = ins.get("inspected_paths", [])
    c.add("actual_visual_inspection_recorded",
          bool(paths) and all(Path(p).exists() for p in paths) and len(ins.get("observations", [])) >= len(paths),
          f"검수 파일 {len(paths)}개 · 관찰 {len(ins.get('observations', []))}건")

    iso = run / "isaac"
    man, vid = iso / "render_manifest.json", iso / "w13_full_cycle.mp4"
    for p, nm in ((man, "isaac_manifest_exists"), (vid, "isaac_video_exists")):
        if not _need(c, p, nm):
            return c, None
    m = json.load(open(man))
    c.add("isaac_overlays_declared",
          all(m.get("overlays", {}).get(k) for k in ("source_time_s", "phase", "fixed_bin", "tool",
                                                     "source_marker", "target_marker")),
          json.dumps(m.get("overlays", {}), ensure_ascii=False))
    nrt = int(np.asarray(z["particle_frame_t_s"]).shape[0])
    c.add("isaac_frames_match_raw", int(m.get("n_frames", -1)) == nrt,
          f"video {m.get('n_frames')} vs raw particle frames {nrt}")
    c.add("isaac_source_times_from_raw",
          bool(np.allclose(np.asarray(m.get("source_time_s", []), float),
                           np.asarray(z["particle_frame_t_s"], float), atol=1e-9)),
          f"표시 원자료 시간 {len(m.get('source_time_s', []))}개 일치")
    c.add("no_basic_writer", "BasicWriter" not in json.dumps(m, ensure_ascii=False), "BasicWriter 덤프 없음")
    return c, {"rrd_MB": round(rrd.stat().st_size / 1e6, 2), "video_MB": round(vid.stat().st_size / 1e6, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-visual", action="store_true")
    ap.add_argument("--run", default=str(HERE / "run_01"))
    ap.add_argument("--pile-sha256", default=PILE_SHA256, help="rev34: 기대 더미 sha256(기본 = rev32 동결 더미)")
    a = ap.parse_args()
    if not (a.check or a.check_visual):
        raise SystemExit("--check 또는 --check-visual 중 하나가 필요하다")
    run = Path(a.run)
    if not run.is_dir():
        print(f"  [FAIL] run_dir_exists: {run}")
        print("W13_SELF_CHECK_FAIL" if a.check else "W13_VISUAL_CHECK_FAIL")
        return 1
    c, summary = (check_science(run, a.pile_sha256) if a.check else check_visual(run))
    c.dump()
    if summary:
        print("  요약: " + json.dumps(summary, ensure_ascii=False, default=float))
    print(("W13_SELF_CHECK_OK" if c.ok else "W13_SELF_CHECK_FAIL") if a.check
          else ("W13_VISUAL_CHECK_OK" if c.ok else "W13_VISUAL_CHECK_FAIL"))
    return 0 if c.ok else 1


if __name__ == "__main__":
    sys.exit(main())
