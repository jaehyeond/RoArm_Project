"""W26 알 템플릿 재맞춤(CPU). 실측(질량 26.5 mg·사진 치수 4.6×3.8×3.2 mm·물 부양 → 905 kg/m³)으로 렌즈 7구 클럼프를
같은 생성기(pellet-model `sim_pellet_model.build_template`, lens 규칙 = volume_neutral_ring_radius)로 다시 만들고,
기존 템플릿(4.5×3.8×2.5, W25 더미 NPZ 안 clump_template_json)을 같은 코드로 재현해 대조한다."""
import json, math, sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/cgxr/orca/workspaces/RoArm_Project/pellet-model")
import sim_pellet_model as PM

OUT = Path(__file__).resolve().parent
PILE = "/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/pile_flat40_20260929/pile_lens6_a4p5_b3p8_c2p5_slab40_outer_310x220_n67737_rho0p503_seed460.npz"
RHO = 905.0

def make(axes):
    pel = PM.lens_equivalent_pellet(axes, RHO)
    t = PM.build_template(pel, "lens", lens_axes_mm=axes, lens_n_ring=6)
    d = {"axes_mm": list(axes), "mass_mg": t.mass_kg * 1e6, "union_volume_mm3": t.union_volume_m3 * 1e9,
         "ellipsoid_volume_mm3": math.pi / 6 * axes[0] * axes[1] * axes[2],
         "moi_kg_m2": list(t.moi_kg_m2), "sphere_radius_m": t.sphere_radius_m, "bounding_diameter_mm": t.bounding_diameter_m * 1e3,
         "n_spheres": t.n_spheres, "offsets_m": [list(o) for o in t.offsets_m],
         "sphere_radii_m": list(getattr(t, "sphere_radii_m", [])) or None,
         "lens": getattr(t, "lens", None)}
    return t, d

old_t, old = make((4.5, 3.8, 2.5))
new_t, new = make((4.6, 3.8, 3.2))
z = np.load(PILE, allow_pickle=True); ref = json.loads(str(z["clump_template_json"]))
rep = {"mass_mg_code_vs_npz": [old["mass_mg"], ref["mass_kg"] * 1e6],
       "union_volume_code_vs_npz_mm3": [old["union_volume_mm3"], ref["union_volume_m3"] * 1e9],
       "ring_fraction_code_vs_npz": [old["lens"]["ring_radius_fraction"] if old["lens"] else None, ref["lens"]["ring_radius_fraction"]],
       "radii_equal": np.allclose(np.asarray(old["sphere_radii_m"] or [], float), np.asarray(ref["sphere_radii_m"], float), rtol=0, atol=1e-12) if old["sphere_radii_m"] else None,
       "offsets_max_abs_diff_m": float(np.abs(np.asarray(old["offsets_m"]) - np.asarray(ref["offsets_m"])).max()),
       "moi_rel_diff": [abs(a / b - 1) for a, b in zip(old["moi_kg_m2"], ref["moi_kg_m2"])]}
lens_new = new["lens"] or {}
summary = {"artifact": "W26_PELLET_TEMPLATE_REFIT", "generator": "/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/sim_pellet_model.py",
           "generator_sha256": hashlib.sha256(open("/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/sim_pellet_model.py", "rb").read()).hexdigest(),
           "inputs": {"mass_measured_mg": 26.48, "axes_from_photos_mm": [4.62, 3.79, "3.1~3.7(옆모습)"], "axes_used_mm": [4.6, 3.8, 3.2],
                      "thickness_rule": "타원체 부피 × 905 = 26.5 mg 가 되는 두께 3.19 → 3.2 (사진 범위 안)", "density_kg_m3": RHO, "float_test": "뜸(충전제 없음)"},
           "reproduce_old_template": rep,
           "old": {k: old[k] for k in ("axes_mm", "mass_mg", "union_volume_mm3", "ellipsoid_volume_mm3", "bounding_diameter_mm", "moi_kg_m2")},
           "new": {k: new[k] for k in ("axes_mm", "mass_mg", "union_volume_mm3", "ellipsoid_volume_mm3", "bounding_diameter_mm", "moi_kg_m2")},
           "new_lens": {k: lens_new.get(k) for k in ("ring_radius_fraction", "ring_radius", "volume_error_vs_ellipsoid", "roughness_over_c", "fallback", "n_overlapping_pairs", "min_centre_distance_over_radius_sum", "mc_convergence")},
           "new_vs_old": {"mass_ratio": new["mass_mg"] / old["mass_mg"], "volume_ratio": new["union_volume_mm3"] / old["union_volume_mm3"],
                          "bounding_diameter_ratio": new["bounding_diameter_mm"] / old["bounding_diameter_mm"],
                          "n_pellets_40mm_layer_est": int(round(67737 * old["union_volume_mm3"] / new["union_volume_mm3"])),
                          "note": "40 mm 층 알 수는 부피비 역수로 추정(같은 충전율 가정) — 실제는 생성기가 정한다"},
           "new_template_json": {"mass_kg": new_t.mass_kg, "moi_kg_m2": list(new_t.moi_kg_m2), "sphere_radii_m": new["sphere_radii_m"], "offsets_m": new["offsets_m"],
                                 "union_volume_m3": new_t.union_volume_m3, "bounding_diameter_m": new_t.bounding_diameter_m, "lens": lens_new}}
(OUT / "template_refit_20260930.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float) + "\n")
print(json.dumps({k: summary[k] for k in ("reproduce_old_template", "old", "new", "new_lens", "new_vs_old")}, ensure_ascii=False, indent=1, default=float))
