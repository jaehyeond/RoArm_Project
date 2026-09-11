"""Compare RRD timeline coverage against the saved W10 callback arrays (read-only).

This is an observability check, never a replacement for the scientific JSON/NPZ.
Run with isaaclab Python (rerun 0.34.1). No GPU or new simulation is needed.
"""
from pathlib import Path
import json
import sys
import numpy as np
from rerun.experimental import RrdReader


def verify(cell, rrd):
    cell, rrd = Path(cell), Path(rrd)
    z = np.load(cell / "scoop_s1_seed460.npz")
    res = json.loads((cell / "scoop_s1_seed460.json").read_text())
    rows = json.loads((cell / "timeline_seed460.json").read_text())["rows"]
    rt = np.load(res["render_timeline"]["path"])
    n = len(z["frame_t_s"])
    reader = RrdReader(rrd)
    checks = {}
    contact_counts = np.bincount(z["contact_frame"], minlength=n)
    contracts = {
        "/geometry/tool/fixed_nodes": ("Points3D:positions", np.arange(n), np.full(n, z["nodes_F_m"].shape[1])),
        "/geometry/tool/door_nodes": ("Points3D:positions", np.arange(n), np.full(n, z["nodes_D_m"].shape[1])),
        "/contacts/points": ("Points3D:positions", np.arange(n), contact_counts),
        "/contacts/forces": ("Arrows3D:vectors", np.arange(n), contact_counts),
        "/geometry/pile/animated": ("Points3D:positions", rt["timeline_frame"], np.full(len(rt["timeline_frame"]), len(z["sphere_positions_m"]))),
    }
    for key in ("z_lip_mm", "q_deg", "lipF_N", "M_hinge_res_Nm", "v_particle_max", "n_door", "n_fixed", "Fz_fixed_up_N", "max_single_contact_N"):
        contracts[f"/metrics/{key}"] = ("Scalars:scalars", np.arange(n), np.ones(n, dtype=int))
    for entity, (component, expected_frames, expected_counts) in contracts.items():
        frames, counts, times, scalar_values = [], [], [], []
        for chunk in reader.stream().filter(content=entity, has_timeline="frame", components=component):
            rb = chunk.to_record_batch()
            fields = [f.name for f in rb.schema if (f.metadata or {}).get(b"rerun:component", b"").decode() == component]
            if len(fields) != 1:
                raise ValueError((entity, fields))
            batches = rb.column(fields[0]).to_pylist()
            frames.extend(rb.column("frame").to_pylist())
            times.extend(rb.column("sim_time_s").cast("int64").to_pylist())
            counts.extend(-1 if b is None else len(b) for b in batches)
            if component == "Scalars:scalars":
                scalar_values.extend(float("nan") if b is None or len(b) != 1 else b[0] for b in batches)
        order = np.argsort(frames)
        expected_order = np.argsort(expected_frames)
        observed = np.asarray(frames, dtype=np.int64)[order]
        wanted = np.asarray(expected_frames, dtype=np.int64)[expected_order]
        frame_ok = np.array_equal(observed, wanted)
        count_ok = np.array_equal(np.asarray(counts)[order], np.asarray(expected_counts)[expected_order])
        source_times = rt["t_s"] if entity.endswith("/animated") else z["frame_t_s"]
        time_ok = np.array_equal(np.asarray(times, dtype=np.int64)[order], np.rint(source_times * 1e9).astype(np.int64)[expected_order])
        scalar_ok = True
        if component == "Scalars:scalars":
            key = entity.rsplit("/", 1)[1]
            scalar_ok = np.array_equal(np.asarray(scalar_values)[order], [float(r.get(key, 0.0)) for r in rows])
        checks[entity] = {"pass": bool(frame_ok and count_ok and time_ok and scalar_ok), "observed_rows": len(observed),
                          "expected_rows": len(wanted), "frames_exact": frame_ok, "batch_lengths_exact": count_ok,
                          "times_exact_nanoseconds": time_ok, "scalar_values_exact": scalar_ok}
    return {"artifact": "W10_RRD_COVERAGE", "rrd": str(rrd), "source_cell": str(cell), "source_syncs": n,
            "source_particle_frames": len(rt["t_s"]), "checks": checks, "pass": all(c["pass"] for c in checks.values())}


if __name__ == "__main__":
    result = verify(sys.argv[1], sys.argv[2])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
