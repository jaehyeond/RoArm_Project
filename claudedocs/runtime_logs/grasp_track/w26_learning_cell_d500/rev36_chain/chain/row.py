"""rev36-chain 데이터 행 기록기(CPU, 물리 0). 셀 run 하나 → 학습 데이터 한 행(row_*.json + row_*.npz).

행 스키마 W26_ROW_V1 (학습 방향 정본: 형상 예측기 먼저, 질량은 기록만)
    입력(관측)  hm_pre_{truth,cam} : 퍼내기 직전(settle_end) 높이지도, 상자 좌표 5 mm 격자(44×62), m
    행동       cmd_box_xy_m        : 로봇에 준 goto_xyz 명령점(상자 좌표) + roll_deg(손목 롤, 09-30 허용). lip_box_xy_m = FK 립. base_deg·tool_yaw_box_deg
    결과(형상)  hm_post_{truth,cam}: 들린 알·트레이 밖 알을 뺀 재닫기 끝(reclose_end) 알로 만든 높이지도
    결과(양)   lifted_count·lifted_mass_g : 기하 라벨(settle 윗면 + 15 mm 위 알 중심, lift_end) — 라벨 규칙 정본
    보조       crater(구덩이 부피·최대 깊이·관측 중심), 실패 표시, 물성 값, 출처 해시, 카메라·연산자 출처
    truth = 알 구로 만든 정확한 높이(`heightmap_from_particles`), cam = 카메라 시점 렌더 + 실물 연산자(`cam_render`).
    cam 은 가림 칸이 valid=False, 벽 한 칸 띠는 벽 안쪽 면이 섞일 수 있어 `wall_band` 로 따로 표시한다.
    저장 좌표(2026-09-30 결정) = **규약 B(실물 로봇·상자 방향, 로봇이 상자 +y 쪽)**. 시뮬 내부는 검증된 규약 A 라
    저장 시 변환한다: 높이지도 180° 회전(격자가 상자 중심 대칭), 명령·립·구덩이 중심 xy 부호 반전, 툴 yaw +180°, 카메라 T = W24 원본(B).
    베이스·롤 관절각과 깊이 영상(카메라 프레임)은 그대로. provenance.frame_transform 에 기록.
    hm_post 는 알이 아직 움직이는 순간일 수 있다(다음 셀 settle 전). 가라앉은 뒤 형상 = 다음 셀 행의 hm_pre_truth 이며,
    행 파일을 다시 쓰지 않고 연쇄 목록(chain.json 의 `post_settled_row`)으로 잇는다.

사용: python row.py <cell_run_dir> <parent_pile.npz> <params.json> <out_prefix> [--chain-json '{...}'] [--no-cam]
"""
import argparse, glob, hashlib, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import site_map as SM          # noqa: E402
import next_pile as NP         # noqa: E402
import cam_render as CR        # noqa: E402
from roarm_rl.heightmap import GridSpec, heightmap_from_particles   # noqa: E402

SCHEMA = "W26_ROW_V2"
STORE_FRAME = "B"   # 2026-09-30 사용자 결정: 저장 좌표 = 실물 로봇·상자 방향(실물 파이프라인 D496·W24 카메라 정의 = 규약 B, 로봇이 상자 +y 쪽)
CRATER_MIN_DEPTH_M = 0.003          # cell_post.py 의 구덩이 마스크와 같은 3 mm
CRATER_R_MAX_M = 0.060
PHYS_KEYS = ("E_pa", "E_mesh_pa", "nu", "CoR", "mu", "Crr", "particle_density_kg_m3", "timestep_s", "cd_update_freq",
             "plunge_mm", "lift_mm", "door_open_servo_deg", "close_end_joint_deg", "w25_proc_chatter",
             "w25_proc_open_at_surface", "diag_fine_sync_s", "box_frame_convention")


def grid(box):
    c = 0.005
    return GridSpec(origin_xy_m=(float(box[0, 0]), float(box[1, 0])), cell_m=c,
                    shape=(int(np.ceil((box[1, 1] - box[1, 0]) / c)), int(np.ceil((box[0, 1] - box[0, 0]) / c))),
                    frame="deme_box_floor_center", z_datum_m=0.0)


def wall_band(spec):
    m = np.zeros(spec.shape, bool); m[:1, :] = m[-1:, :] = True; m[:, :1] = m[:, -1:] = True
    return m


def crater(pre, post, spec, site):
    dh = pre.astype(np.float64) - post.astype(np.float64)
    rows, cols = spec.shape
    yc = spec.origin_xy_m[1] + (np.arange(rows) + 0.5) * spec.cell_m
    xc = spec.origin_xy_m[0] + (np.arange(cols) + 0.5) * spec.cell_m
    X, Y = np.meshgrid(xc, yc)
    near = np.hypot(X - site[0], Y - site[1]) <= CRATER_R_MAX_M
    m = near & (dh >= CRATER_MIN_DEPTH_M)
    w = np.where(m, dh, 0.0)
    cen = [float((X * w).sum() / w.sum()), float((Y * w).sum() / w.sum())] if w.sum() > 0 else None
    return {"removed_volume_ml": float(np.clip(dh, 0, None).sum() * spec.cell_m ** 2 * 1e6),
            "added_volume_ml": float(np.clip(-dh, 0, None).sum() * spec.cell_m ** 2 * 1e6),
            "max_depth_mm": float(dh.max() * 1000), "n_cells_ge_3mm": int(m.sum()),
            "observed_center_box_xy_m": cen,
            "center_minus_lip_mm": None if cen is None else [(cen[0] - site[0]) * 1000, (cen[1] - site[1]) * 1000],
            "rule": f"dh = pre − post(truth). 중심 = 립에서 {CRATER_R_MAX_M*1000:.0f} mm 안 dh ≥ {CRATER_MIN_DEPTH_M*1000:.0f} mm 칸의 dh 가중 무게중심"}


def build(run, parent_pile, params, out_prefix, chain=None, cam=True):
    t0 = time.time()
    P = SM.load_params(params)
    zp = np.load(parent_pile, allow_pickle=True); tpl = json.loads(str(zp["clump_template_json"]))
    npz = NP.cell_npz(run); z = np.load(npz, allow_pickle=True); fidx = NP.frames(z)
    rj_path = npz[:-4] + ".json"; rj = json.load(open(rj_path))
    box, _ = SM.FK.w25_tray_bounds(np.asarray(zp["box_bounds_m"], float), P, fail_closed=False)
    spec = grid(box)
    top, thr = NP.lift_threshold(z, fidx)
    fs, fl, fr = fidx["settle_end"], fidx["lift_end"], fidx["reclose_end"]

    def spheres(f, mask=None):
        p = np.asarray(z["particle_pos_m"][f], np.float64); q = np.asarray(z["particle_quat_xyzw"][f], np.float64)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        if mask is not None:
            p, q = p[mask], q[mask]
        return SM.W11SRC.expand_spheres(p, q, tpl)

    lift_end_z = np.asarray(z["particle_pos_m"][fl], np.float64)[:, 2]
    lifted_count = int((lift_end_z > thr).sum())
    rc = np.asarray(z["particle_pos_m"][fr], np.float64)[:, 2] > thr
    pre_c, pre_r = spheres(fs)
    post_c, post_r = spheres(fr, ~rc)
    hm_pre = heightmap_from_particles(pre_c, pre_r, spec); hm_post = heightmap_from_particles(post_c, post_r, spec)
    w26 = (rj.get("w26_cell") or {}); site_info = w26.get("site") or {}
    lip = site_info.get("lip_box_xy_m") or [rj["scoop_site"]["x_mm"] / 1000.0, rj["scoop_site"]["y_mm"] / 1000.0]
    arrays = {"hm_pre_truth": hm_pre.height, "hm_post_truth": hm_post.height,
              "hm_pre_truth_valid": hm_pre.valid, "hm_post_truth_valid": hm_post.valid, "wall_band": wall_band(spec)}
    cam_meta = None
    if cam:
        conv = P.get("box_frame_convention")
        T = CR.camera_T(conv)
        tray = SM.K.tray_mesh(box, P["tray_wall_t_mm"] / 1000.0); tris = np.asarray(tray.vertices)[np.asarray(tray.faces)]
        for name, (c, r) in (("pre", (pre_c, pre_r)), ("post", (post_c, post_r))):
            D = CR.render_depth(c, r, tris, T)
            hm, filt = CR.heightmap_from_render(D, T, spec)
            arrays[f"hm_{name}_cam"] = hm.height; arrays[f"hm_{name}_cam_valid"] = hm.valid
            arrays[f"depth_{name}_mm_u16"] = np.where(np.isfinite(D), np.round(D * 1000), 0).astype(np.uint16)
            t_ok = arrays[f"hm_{name}_truth_valid"] & hm.valid & ~arrays["wall_band"]
            err = (hm.height.astype(np.float64) - arrays[f"hm_{name}_truth"].astype(np.float64))[t_ok]
            arrays[f"_stat_{name}"] = {"cam_valid_frac": float(hm.valid.mean()),
                                       "cam_minus_truth_mm_interior": {"mean": float(err.mean() * 1000),
                                                                       "rms": float(np.sqrt((err ** 2).mean()) * 1000),
                                                                       "p99_abs": float(np.percentile(np.abs(err), 99) * 1000)}}
        cam_meta = {"T_box_depthcam": T.tolist(), "box_frame_convention": conv, "source": CR.W24_SOURCE,
                    "intrinsics": {k: v for k, v in CR.INTR_NOMINAL.items()}, "operator": CR.REAL_OP,
                    "pre": arrays.pop("_stat_pre"), "post": arrays.pop("_stat_post"),
                    "note": "공칭 내부 파라미터·팔 가림 없음 — 실기 카메라 도착 전 대리값"}
    cr = crater(hm_pre.height, hm_post.height, spec, lip)
    # ── 저장 좌표 변환(A → B): 회전 180° ──
    conv = P.get("box_frame_convention"); transform = None
    if STORE_FRAME == "B" and conv == "A":
        for k in list(arrays):
            if k.startswith("hm_") or k == "wall_band":
                arrays[k] = np.ascontiguousarray(arrays[k][::-1, ::-1])
        neg = lambda v: None if v is None else [-float(v[0]), -float(v[1])]
        site_info = dict(site_info, cmd_box_xy_m=neg(site_info.get("cmd_box_xy_m")), lip_box_xy_m=neg(site_info.get("lip_box_xy_m")),
                         tool_yaw_box_deg=None if site_info.get("tool_yaw_box_deg") is None else ((float(site_info["tool_yaw_box_deg"]) + 180.0 + 180.0) % 360.0 - 180.0))
        lip = neg(lip)
        cr = dict(cr, observed_center_box_xy_m=neg(cr.get("observed_center_box_xy_m")),
                  center_minus_lip_mm=None if cr.get("center_minus_lip_mm") is None else [-cr["center_minus_lip_mm"][0], -cr["center_minus_lip_mm"][1]])
        if cam_meta is not None:
            cam_meta = dict(cam_meta, T_box_depthcam=CR.camera_T("B").tolist(), box_frame_convention="B")
        transform = "p_B = diag(-1,-1,1)·p_A (z 축 180° 회전); 높이지도 [::-1, ::-1]; yaw+180°; 관절각·깊이 영상 불변"
    elif STORE_FRAME == "B" and conv != "B":
        raise SystemExit(f"저장 규약 B 로 바꿀 수 없는 시뮬 규약 {conv!r}")
    run_ok = (rj.get("stopped_early_after_phase") == "reclose") and (rj.get("diverged") is False) and (rj.get("abort_class") is None)
    row = {"schema": SCHEMA, "row_id": Path(out_prefix).name, "chain": chain or {},
           "frame": {"grid": "deme_box_floor_center 5 mm (rows=y, cols=x)", "shape": list(spec.shape),
                     "origin_xy_m": list(spec.origin_xy_m), "stored_convention": STORE_FRAME,
                     "robot_side": "B: 로봇 = 상자 +y 쪽(실물 D496·W24 와 동일)" if STORE_FRAME == "B" else "A: 로봇 = 상자 −y 쪽",
                     "sim_internal_convention": conv, "frame_transform": transform},
           "action": {"cmd_box_xy_m": site_info.get("cmd_box_xy_m"), "lip_box_xy_m": lip,
                      "base_deg": site_info.get("base_deg"), "roll_deg": site_info.get("roll_deg"),
                      "tool_yaw_box_deg": site_info.get("tool_yaw_box_deg"), "q5_deg": site_info.get("q5_deg"),
                      "wall_clearance_min_mm": None if site_info.get("wall_clearance_min_m") is None
                      else site_info["wall_clearance_min_m"] * 1000},
           "label": {"lifted_count": lifted_count, "lifted_mass_g": lifted_count * float(tpl["mass_kg"]) * 1000.0,
                     "rule": f"settle_end source 알 중심 z 최대({top:.6f} m) + 15 mm 위 알 중심 개수, lift_end",
                     "removed_count_reclose_end": int(rc.sum()), "pellet_mass_mg": float(tpl["mass_kg"]) * 1e6},
           "crater": cr,
           "flags": {"run_ok": bool(run_ok), "diverged": rj.get("diverged"), "abort_class": rj.get("abort_class"),
                     "stopped_after": rj.get("stopped_early_after_phase"),
                     "door_stops": [s.get("reason") for s in (rj.get("door") or {}).get("stops", [])],
                     "empty_scoop": lifted_count == 0},
           "physics": {k: P.get(k) for k in PHYS_KEYS},
           "camera": cam_meta,
           "provenance": {"cell_run": str(run), "cell_npz_sha256": NP.sha256(npz), "cell_json_sha256": NP.sha256(rj_path),
                          "parent_pile": str(parent_pile), "parent_pile_sha256": NP.sha256(parent_pile),
                          "params": str(params), "params_sha256": NP.sha256(params),
                          "frames": {"pre": "settle_end", "label": "lift_end", "post": "reclose_end"},
                          "frame_t_s": {k: float(z["particle_frame_t_s"][fidx[k]]) for k in ("settle_end", "lift_end", "reclose_end")},
                          "build_wall_s": None}}
    np.savez_compressed(str(out_prefix) + ".npz", **{k: np.asarray(v) for k, v in arrays.items()})
    row["provenance"]["build_wall_s"] = round(time.time() - t0, 1)
    row["provenance"]["row_npz_sha256"] = NP.sha256(str(out_prefix) + ".npz")
    Path(str(out_prefix) + ".json").write_text(json.dumps(row, ensure_ascii=False, indent=1) + "\n")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_run"); ap.add_argument("parent_pile"); ap.add_argument("params"); ap.add_argument("out_prefix")
    ap.add_argument("--chain-json", default=None); ap.add_argument("--no-cam", action="store_true"); a = ap.parse_args()
    r = build(a.cell_run, a.parent_pile, a.params, a.out_prefix, json.loads(a.chain_json) if a.chain_json else None, not a.no_cam)
    print(json.dumps({"label": r["label"], "crater": {k: r["crater"][k] for k in ("removed_volume_ml", "max_depth_mm", "observed_center_box_xy_m", "center_minus_lip_mm")},
                      "flags": r["flags"], "camera": None if r["camera"] is None else {k: r["camera"][k] for k in ("pre", "post")},
                      "build_wall_s": r["provenance"]["build_wall_s"]}, ensure_ascii=False))
