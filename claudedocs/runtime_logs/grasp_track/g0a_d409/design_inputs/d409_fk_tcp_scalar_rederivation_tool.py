"""D409 design-input: re-derive the FK TCP reproduction scalar from stored D349 literals.

Replaces the handover-only figure "FK 0.0013mm" (no repo artifact; see
session_20260803_grasp_g0a_real_first_funnel_decisions_state_update.md:138-140)
with a filed, deterministic re-derivation from the frozen D349 measurement JSON.

Computation: Euclidean distance between
  frozen_candidate_alignment.commanded_tcp_{x,y,z}_m  (offline-FK TCP of the
  commanded joints, stored by D349) and
  frozen_candidate_alignment.actual_tcp_{x,y,z}_m     (Isaac live TCP, stored
  by D349),
using only stored float literals — no FK re-execution, no Isaac, no physics.

Determinism contract: the canonical payload (sorted keys, repr floats) must be
byte-identical across two independent process runs; the wrapper script compares
sha256 of the payloads.
"""
import hashlib
import json
import math
import sys

D349_PATH = (
    "/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/"
    "grasp_track/g0a_d349/d349_frozen_target_distance_measurement.json"
)


def main() -> None:
    raw = open(D349_PATH, "rb").read()
    src_sha = hashlib.sha256(raw).hexdigest()
    doc = json.loads(raw)

    align = doc["frozen_candidate_alignment"]
    guard = doc["target_state_guard"]

    commanded = [align["commanded_tcp_x_m"], align["commanded_tcp_y_m"], align["commanded_tcp_z_m"]]
    actual = [align["actual_tcp_x_m"], align["actual_tcp_y_m"], align["actual_tcp_z_m"]]
    deltas_m = [c - a for c, a in zip(commanded, actual)]
    dist_m = math.sqrt(sum(d * d for d in deltas_m))

    payload = {
        "artifact": "d409_fk_tcp_scalar_rederivation",
        "method": (
            "euclidean distance between stored commanded_tcp_* and actual_tcp_* literals of "
            "frozen_candidate_alignment; pure stored-scalar recomputation, no FK re-execution"
        ),
        "source_file": D349_PATH,
        "source_sha256": src_sha,
        "inputs": {
            "commanded_tcp_m": [repr(v) for v in commanded],
            "actual_tcp_m": [repr(v) for v in actual],
            "commanded_joint_rad_float32": [repr(v) for v in guard["commanded_joint_rad_float32"]],
            "actual_joint_rad_float32": [repr(v) for v in guard["actual_joint_rad_float32"]],
        },
        "deltas_m": [repr(d) for d in deltas_m],
        "distance_m": repr(dist_m),
        "distance_mm": repr(dist_m * 1000.0),
        "distance_mm_rounded_7": round(dist_m * 1000.0, 7),
        "handover_claim_mm": 0.0013,
        "handover_claim_repo_artifact": "absent (handover-only figure; do not cite)",
        "notes": (
            "float64 joint vector is NOT stored in the D349 JSON (float32 only); this scalar is a "
            "stored-literal recomputation and does not claim FK re-execution reproduction. Any FK "
            "re-execution figure must be produced separately by the D409 harness with its own "
            "determinism check."
        ),
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write(canonical)


if __name__ == "__main__":
    main()
