"""rev36-chain 다음 더미 저장기(CPU, 물리 0). 셀 run 이 멈춘 순간(기본 reclose_end)의 알 자세에서
**들린 알**과 **트레이 밖 알**을 빼고 나머지를 더미 NPZ(생성기와 같은 키)로 저장한다. 다음 셀은 이 NPZ 로
시작하고, 셀 첫 단계(settle 0.1 s)가 남은 알을 다시 가라앉힌다.

규칙 (사전 고정 — 라벨과 같은 기하 기준)
    들린 알 = 이 셀 settle_end 에서 source 로 분류된 알 중심 z 최대(= 더미 윗면) + 15 mm 보다 높은 알 중심.
      라벨(lift_end)과 같은 문턱을 **저장 시점 프레임**에 적용한다. 두 시점 개수 차이는 기록한다.
    트레이 밖 알 = 구 하나라도 선언 트레이 안쪽 경계를 tol(0.5 mm) 넘게 벗어난 알(`w13_fk.w25_tray_bounds` 와 같은 식).
    남은 알 = 나머지. 속도는 저장하지 않는다(0). 자세 = 저장 프레임의 float32 값을 float64 로, 쿼터니언은 정규화.
    질량 보존 검사: 남은 + 들린 + 트레이 밖 = 부모 알 수 (아니면 실패).

사용: python next_pile.py <cell_run_dir> <parent_pile.npz> <params.json> <out.npz> [--frame reclose_end]
"""
import argparse, glob, hashlib, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import site_map as SM                 # noqa: E402  (같은 경로 설정·파라미터 병합)
W11SRC, FK = SM.W11SRC, SM.FK

LIFT_THRESHOLD_M = 0.015


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cell_npz(run):
    c = sorted(glob.glob(str(Path(run) / "w13_cycle_seed*.npz")))
    if len(c) != 1:
        raise SystemExit(f"셀 NPZ 가 하나가 아니다: {c}")
    return c[0]


def frames(z):
    return dict(zip([str(t) for t in z["decision_tags"]], z["decision_particle_frame_index"].tolist()))


def lift_threshold(z, fidx):
    fs = fidx["settle_end"]
    top = float(z["particle_pos_m"][fs][z["inventory_code"][fs] == 0][:, 2].max())
    return top, top + LIFT_THRESHOLD_M


def build(run, parent_pile, params, out, frame="reclose_end"):
    P = SM.load_params(params)
    zp = np.load(parent_pile, allow_pickle=True)
    npz = cell_npz(run); z = np.load(npz, allow_pickle=True); fidx = frames(z)
    if frame not in fidx:
        raise SystemExit(f"저장 프레임 {frame} 이 셀 결정 목록에 없다: {list(fidx)}")
    n_parent = len(zp["clump_positions_m"])
    if z["particle_pos_m"].shape[1] != n_parent:
        raise SystemExit(f"셀 알 수 {z['particle_pos_m'].shape[1]} ≠ 부모 더미 {n_parent} — 다른 더미의 run")
    tpl = json.loads(str(zp["clump_template_json"]))
    top, thr = lift_threshold(z, fidx)
    fr = fidx[frame]
    pos = np.asarray(z["particle_pos_m"][fr], np.float64)
    q = np.asarray(z["particle_quat_xyzw"][fr], np.float64); q /= np.linalg.norm(q, axis=1, keepdims=True)
    lifted = pos[:, 2] > thr
    n_lift_label = int((np.asarray(z["particle_pos_m"][fidx["lift_end"]], np.float64)[:, 2] > thr).sum()) \
        if "lift_end" in fidx else None
    sp, sr = W11SRC.expand_spheres(pos, q, tpl)
    k = len(tpl["sphere_radii_m"])
    box_decl, _ = FK.w25_tray_bounds(np.asarray(zp["box_bounds_m"], float), P, fail_closed=False)
    tol = float(P.get("tray_match_tol_mm", 0.5)) / 1000.0
    b = box_decl
    bad_s = (((sp[:, 0] - sr) < b[0, 0] - tol) | ((sp[:, 0] + sr) > b[0, 1] + tol) |
             ((sp[:, 1] - sr) < b[1, 0] - tol) | ((sp[:, 1] + sr) > b[1, 1] + tol) | ((sp[:, 2] + sr) > b[2, 1] + tol))
    out_tray = bad_s.reshape(-1, k).any(axis=1) & ~lifted
    keep = ~lifted & ~out_tray
    n_keep, n_lift, n_out = int(keep.sum()), int(lifted.sum()), int(out_tray.sum())
    if n_keep + n_lift + n_out != n_parent:
        raise SystemExit("질량 보존 실패")
    kp, kq = pos[keep], q[keep]
    ksp, ksr = W11SRC.expand_spheres(kp, kq, tpl)
    FK.w25_tray_bounds(np.asarray(zp["box_bounds_m"], float), P, ksp, ksr, fail_closed=True)   # 다음 셀이 거부하지 않는지
    parent_idx = np.nonzero(keep)[0].astype(np.int64)
    lineage_parent = np.asarray(zp["w26_lineage_root_index"], np.int64)[parent_idx] \
        if "w26_lineage_root_index" in zp.files else parent_idx
    rec = {"artifact": "W26_NEXT_PILE", "cell_run": str(run), "cell_npz": npz, "cell_npz_sha256": sha256(npz),
           "parent_pile": str(parent_pile), "parent_pile_sha256": sha256(parent_pile), "params": str(params),
           "frame": frame, "frame_index": int(fr), "frame_t_s": float(z["particle_frame_t_s"][fr]),
           "settle_top_z_m": top, "lift_threshold_z_m": thr, "rule": __doc__.split("규칙")[1].split("사용:")[0].strip(),
           "n_parent": n_parent, "n_kept": n_keep, "n_removed_lifted": n_lift, "n_removed_out_of_tray": n_out,
           "n_lifted_at_lift_end_label": n_lift_label,
           "removed_mass_g": (n_lift + n_out) * float(tpl["mass_kg"]) * 1000.0,
           "kept_max_speed_m_s_at_frame": float(np.linalg.norm(np.asarray(z["particle_vel_m_s"][fr], np.float64)[keep], axis=1).max()),
           "kept_top_z_m": float((ksp[:, 2] + ksr).max())}
    meta = {"generator": "w26 rev36_chain next_pile.py", "w26_next_pile": rec,
            "array_contract": {"clump_positions_m": [n_keep, 3], "clump_quaternions_xyzw": [n_keep, 4],
                               "positions_m": [n_keep * k, 3], "radii_m": [n_keep * k]}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, box_bounds_m=np.asarray(zp["box_bounds_m"], np.float64),
                        clump_template_json=zp["clump_template_json"],
                        clump_positions_m=kp, clump_quaternions_xyzw=kq,
                        positions_m=ksp, radii_m=ksr, initial_positions_m=ksp.copy(),
                        velocities_m_s=np.zeros_like(ksp), particle_ids=np.arange(len(ksp), dtype=np.int64),
                        clump_ids=np.repeat(np.arange(n_keep, dtype=np.int64), k),
                        settle_history=np.zeros((0, 6)), w26_parent_index=parent_idx,
                        w26_lineage_root_index=lineage_parent,
                        metadata_json=np.asarray(json.dumps(meta, ensure_ascii=False)))
    rec["out"] = str(out); rec["out_sha256"] = sha256(out)
    Path(str(out) + ".json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_run"); ap.add_argument("parent_pile"); ap.add_argument("params"); ap.add_argument("out")
    ap.add_argument("--frame", default="reclose_end"); a = ap.parse_args()
    r = build(a.cell_run, a.parent_pile, a.params, a.out, a.frame)
    print(json.dumps({k: r[k] for k in ("n_parent", "n_kept", "n_removed_lifted", "n_removed_out_of_tray",
                                        "n_lifted_at_lift_end_label", "removed_mass_g", "kept_top_z_m")}, ensure_ascii=False))
