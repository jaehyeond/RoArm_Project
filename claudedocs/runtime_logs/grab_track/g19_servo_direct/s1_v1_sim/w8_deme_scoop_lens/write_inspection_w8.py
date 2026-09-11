"""D341 육안 검수 기록 (W8). usage: python write_inspection_w8.py <cell_dir> <tag> <observations.json>
observations.json = {"inspected_by": str, "observations": [str,...], "limitations": [str,...]}
→ <cell_dir>/scoop_s1_seed460_<tag>_inspection.json (validation JSON pass 와 스크린샷 sha 를 함께 적는다)."""
import hashlib, json, sys
from pathlib import Path
cell = Path(sys.argv[1]); tag = sys.argv[2]; obs = json.load(open(sys.argv[3]))
stem = cell / f"scoop_s1_seed460_{tag}"; val = json.load(open(stem.with_name(stem.name + "_rerun_validation.json")))
png = Path(val["headless_render"]["path"]); res = json.load(open(cell / "scoop_s1_seed460.json"))
rec = {"artifact": "DEME_SCOOP_S1_W8_VISUAL_INSPECTION_V1", "cell_dir": str(cell), "result_json_sha256": hashlib.sha256((cell / "scoop_s1_seed460.json").read_bytes()).hexdigest(),
       "rrd": str(stem.with_suffix(".rrd")), "rbl": str(stem.with_suffix(".rbl")), "screenshot_path": str(png), "screenshot_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
       "rerun_version": val.get("version"), "completion_contract_pass": bool(val.get("pass") is True),
       "run_under_inspection": {"site_xy_mm": [res["scoop_site"]["x_mm"], res["scoop_site"]["y_mm"]], "n_in_cavity": res["capture"]["n_in_cavity"], "mass_g": res["capture"]["mass_g"],
                                "door_stops": res["door"]["stops"], "crater_angle_deg": {k: v.get("angle_deg") for k, v in res["crater"]["azimuths"].items()}, "diverged": res["diverged"]},
       "inspected_by": obs["inspected_by"],
       "inspection_method": "headless-rendered blueprint screenshot opened with the image reader and read panel by panel; every observation is something visible in the image, not an inference from the JSON",
       "visual_inspection_complete": True, "observations": obs["observations"], "limitations_seen_in_the_image": obs.get("limitations", []), "non_claims": res["non_claims"]}
out = stem.with_name(stem.name + "_inspection.json"); out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n"); print("INSPECTION_WRITTEN", out)
