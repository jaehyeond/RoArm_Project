"""실행 **전에** 소스·입력·설정·메시 해시와 출력 경로를 예약 고정한다(출력 해시는 실행 뒤에 채운다)."""
import hashlib, json, subprocess, sys, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
attempt = Path(sys.argv[1]); label = sys.argv[2]
SRC = ["sim_w13_full_cycle.py", "w13_kinematics.py", "w13_fk.py", "verify_w13_self.py",
       "compare_wall_regression.py", "w13_rerun_export.py", "freeze_manifest.py",
       "preflight_geometry.py", "preflight_rigid_hinge.py", "preflight_snapshot.py"]
EXT = ["/home/cgxr/Documents/Robotics/RoArm_Project/sim_deme_scoop_s1.py",
       "/home/cgxr/Documents/Robotics/RoArm_Project/sim_scripts/roarm_kinematics.py",
       "/home/cgxr/Documents/Robotics/RoArm_Project/hw_s1_scoop_probe.py",
       "/home/cgxr/Documents/Robotics/RoArm_Project/hw_s1_manual.py",
       "/home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/urdf/roarm_m3_s1_v1.urdf",
       "/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1/fixed_ALL.stl",
       "/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1/door_ALL_jawframe.stl",
       "/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1/design.json",
       "/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz",
       "/home/cgxr/Documents/Robotics/RoArm_Project/roarm_rl/viz_debug.py",
       "/home/cgxr/Documents/Robotics/RoArm_Project/roarm_rl/rerun_contract.py",
       "/home/cgxr/Documents/Robotics/RoArm_Project/roarm_rl/heightmap.py",
       str(HERE.parent / "params_w13.json")]
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
import deme
man = {"artifact": "W13_PROSPECTIVE_MANIFEST", "label": label,
       "frozen_before_execution_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "note": "출력 해시는 실행 전에 알 수 없다. 소스/입력/설정/메시 해시와 출력 **경로**만 사전 고정하고, "
               "실행 뒤 finalize_manifest 가 산출 해시를 append 한다.",
       "implementation_sources_sha256": {f: sha(HERE / f) for f in SRC},
       "external_frozen_inputs_sha256": {p: sha(p) for p in EXT},
       "engine": {"DEME": deme.__version__, "python": sys.version.split()[0], "gpu": gpu},
       "executed_commands_frozen_paths": json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else None,
       "reserved_output_paths": {
           "attempt_dir": str(attempt),
           "result_json": str(attempt / "w13_cycle_seed460.json"),
           "raw_npz": str(attempt / "w13_cycle_seed460.npz"),
           "timeline_json": str(attempt / "timeline_seed460.json"),
           "stdout": str(attempt / "stdout.txt"), "stderr": str(attempt / "stderr.txt"),
           "obj_dir": str(attempt / "_obj"), "rerun_dir": str(attempt / "rerun")}}
attempt.mkdir(parents=True, exist_ok=True)
out = attempt / "MANIFEST_prospective.json"
if out.exists():
    raise SystemExit(f"이미 존재한다(덮어쓰지 않는다): {out}")
json.dump(man, open(out, "w"), ensure_ascii=False, indent=2)
print(json.dumps(man, ensure_ascii=False, indent=1))
