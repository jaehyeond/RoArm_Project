"""W8 후처리: 최종 프레임에서 '쉬고 있는' 입자만으로 퍼낸 뒤 heightmap 을 다시 만들어 구덩이 옆면 각을 다시 잰다.

왜: 최종 프레임에 문 틈으로 흘러내리는 중인(공중) 알과 툴 겉면에 얹힌 알이 carried 문턱(펠릿면+40 mm) 아래에 있어
   heightmap_m 에 80 mm 급 스파이크로 들어갔다(cell_F_c/heightmap_seed460.png 가운데 노란 셀). 정의를 바꾸지 않고
   "쉬는 입자 = 그 xy 의 퍼내기 전 표면(hm_pre) + rest_margin 아래에 중심이 있는 입자" 로 한 번 더 걸러 같은 crater_angles() 를 돌린다.
usage: python crater_rest_writer.py <cell_dir> [rest_margin_mm=10]
→ <cell_dir>/crater_rest_seed460.json, heightmap_rest_seed460.png, crater_profiles_rest_seed460.png
"""
import json, math, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[6]; sys.path.insert(0, str(REPO))
import importlib.util
spec = importlib.util.spec_from_file_location("s1", REPO / "sim_deme_scoop_s1.py"); s1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s1)
from roarm_rl.heightmap import GridSpec, heightmap_from_particles

cell = Path(sys.argv[1]); margin = float(sys.argv[2]) / 1000 if len(sys.argv) > 2 else 0.010
res = json.load(open(cell / "scoop_s1_seed460.json")); z = np.load(cell / "scoop_s1_seed460.npz")
P = res["params"]; box = np.asarray(z["box_bounds_m"], float); cellm = res["heightmap"]["cell_m"]
spec_ = GridSpec(origin_xy_m=(float(box[0, 0]), float(box[1, 0])), cell_m=cellm, shape=tuple(res["heightmap"]["shape"]), frame="deme_box_floor_center", z_datum_m=0.0)
sp, sr = np.asarray(z["sphere_positions_m"], float), np.asarray(z["sphere_radii_m"], float)
k = int(res["heightmap"]["spheres_per_particle"]); carried = np.repeat(np.asarray(z["carried"], bool), k); in_cav = np.repeat(np.asarray(z["in_cavity"], bool), k)
hm_pre = np.asarray(z["heightmap_pre_m"], float)
col = np.clip(((sp[:, 0] - box[0, 0]) / cellm).astype(int), 0, hm_pre.shape[1] - 1); row = np.clip(((sp[:, 1] - box[1, 0]) / cellm).astype(int), 0, hm_pre.shape[0] - 1)
above = sp[:, 2] - hm_pre[row, col]
rest = (~carried) & (~in_cav) & (above <= margin)
excluded = (~carried) & ~rest
hm_rest = heightmap_from_particles(sp[rest], sr[rest], spec_).height
crater = s1.crater_angles(hm_pre, hm_rest, spec_, (res["scoop_site"]["x_mm"] / 1000, res["scoop_site"]["y_mm"] / 1000), P)
# 보조 각(정의 밖, 보고용): 5 mm 격자에서 밴드 안 고리가 2개 미만인 가파른 벽을 위해 r_peak 고리와 그 바깥에서 처음 dh < 0.2·d_max 가 되는 고리
# 사이의 시컨트 기울기 atan(Δdh/Δr). 사전 등록 정의(직선 적합) 가 None 인 방위에서만 의미가 있다.
for name, az in crater["azimuths"].items():
    az["secant_angle_deg"] = None
    if az.get("r_peak_mm") is None:
        continue
    r_ = az["r_mm"]; d_ = az["dh_ring_mm"]; ipk = int(round(az["r_peak_mm"] / (cellm * 1000) - 0.5)); dmax = az["d_max_mm"]
    for i in range(ipk + 1, len(r_)):
        if d_[i] is None:
            break
        if d_[i] < 0.2 * dmax:
            az["secant_angle_deg"] = float(math.degrees(math.atan((d_[ipk] - d_[i]) / (r_[i] - r_[ipk])))); az["secant_r_mm"] = [r_[ipk], r_[i]]
            break
out = {"artifact": "W8_CRATER_REST_V1", "rest_rule": f"resting sphere = not carried, not in cavity, centre <= hm_pre(xy) + {margin*1000:.0f} mm (excludes in-flight/perched pellets)",
       "n_spheres_total": int(len(sp)), "n_spheres_rest": int(rest.sum()), "n_spheres_excluded_in_flight_or_perched": int(excluded.sum()), "n_spheres_carried": int(carried.sum()),
       "excluded_clumps_approx": int(round(excluded.sum() / k)), "excluded_z_range_mm": [float(sp[excluded, 2].min() * 1000), float(sp[excluded, 2].max() * 1000)] if excluded.any() else None,
       "heightmap_rest_max_mm": float(hm_rest.max() * 1000), "heightmap_original_max_mm": float(np.asarray(z["heightmap_m"]).max() * 1000), "crater": crater,
       "crater_original_from_result_json": {kk: v.get("angle_deg") for kk, v in res["crater"]["azimuths"].items()}}
json.dump(out, open(cell / "crater_rest_seed460.json", "w"), ensure_ascii=False, indent=2)
s1.plot_heightmaps(cell, "rest_seed460", spec_, hm_pre, hm_rest, crater, (res["scoop_site"]["x_mm"] / 1000, res["scoop_site"]["y_mm"] / 1000))
np.savez_compressed(cell / "heightmap_rest_seed460.npz", heightmap_rest_m=hm_rest, rest_mask_spheres=rest)
print("SECANT", {kk: (None if v.get("secant_angle_deg") is None else round(v["secant_angle_deg"], 1)) for kk, v in crater["azimuths"].items()})
print("REST", cell.name, "excluded spheres", int(excluded.sum()), "rest max mm", round(float(hm_rest.max() * 1000), 1),
      {kk: (None if v.get("angle_deg") is None else round(v["angle_deg"], 1)) for kk, v in crater["azimuths"].items()}, "removed cm3", round(crater["removed_volume_cm3"], 2))
