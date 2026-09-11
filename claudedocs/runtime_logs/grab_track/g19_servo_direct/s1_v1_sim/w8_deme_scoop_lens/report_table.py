"""보고서 표 생성(수치는 전부 결과 JSON 에서). usage: python report_table.py  → markdown 표 stdout"""
import json, re
from pathlib import Path
OUT = Path(__file__).resolve().parent; V = "F"; CELLS = ["c", "xp50", "xm50"]
runlog = (OUT / "run.log").read_text(); stage = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(r"stage (\S+) rc=(\d+) wall=(\d+)s", runlog)}
def f(x, d=1): return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x))
rows = []
for c in CELLS:
    p = OUT / f"cell_{V}_{c}/scoop_s1_seed460.json"
    if not p.exists():
        rows.append(f"| {c} | (없음) |"); continue
    r = json.load(open(p)); rest_p = OUT / f"cell_{V}_{c}/crater_rest_seed460.json"; rest = json.load(open(rest_p)) if rest_p.exists() else None
    st = r["door"]["stops"]; az = (rest["crater"] if rest else r["crater"])["azimuths"]
    stops = " → ".join("%.2f°(%s)" % (s_["q_deg"], s_["reason"]) for s_ in st)
    ang = " / ".join(f"{k} {f(v.get('angle_deg'))}" + (f"(시컨트 {f(v.get('secant_angle_deg'))})" if v.get("angle_deg") is None and v.get("secant_angle_deg") is not None else "") for k, v in az.items())
    rows.append(f"| {c} | ({r['scoop_site']['x_mm']:.0f}, {r['scoop_site']['y_mm']:.0f}) | {r['scoop_site']['surface_z_mm']:.1f} | {f(r['trajectory']['plunge_reached_mm'])} / {r['trajectory']['descend_hold_steps']} | "
                f"{stops} | {r['door']['servo_deg_final']:.2f} | {r['door']['lip_gap_final_mm']:.1f} | {r['door']['n_pinched_at_lip']} | "
                f"{r['forces']['close_peak_lipF_N']:.2f} / {r['forces']['descend_peak_F_fixed_N']:.2f} | **{r['capture']['n_in_cavity']} / {r['capture']['mass_g']:.2f}** | {r['capture']['fill_vs_bulk']:.3f} | {r['capture']['n_carried_z']} | "
                f"{r['pops']['steps_over_pop_speed']} ({r['pops']['v_particle_max_m_s']:.2f}) | {(rest or r)['crater']['removed_volume_cm3']:.1f} / {f((rest or r)['crater']['dh_max_mm'])} | {ang} | {r['wall_seconds']:.0f} |")
print("| 셀 | 위치 mm | 펠릿면 mm | 잠김 도달 mm / 팔 정지 | 문 정지 (close → reclose) | 서보 환산° | 립 틈 mm | 물림 | 립 등가 피크 / 하강 팔 힘 피크 N | 포획 개 / g | 충전율 | z 딸림 | pop 스텝(v_max m/s) | 제거 부피 cm³ / 최대 깊이 mm | 옆면 각 ° (+x / +y / −x / −y, rest) | 벽시계 s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
print()
mass = [json.load(open(OUT / f"cell_{V}_{c}/scoop_s1_seed460.json"))["capture"]["mass_g"] for c in CELLS if (OUT / f"cell_{V}_{c}/scoop_s1_seed460.json").exists()]
import numpy as np
if len(mass) > 1: print(f"평균 {np.mean(mass):.2f} g · COV {np.std(mass, ddof=1)/np.mean(mass)*100:.1f} % (n={len(mass)})")
print("\n발산 시도:"); 
for d, lab in (("cell_c", "사전 등록 c"), ("cell_xp50", "사전 등록 xp50"), ("cell_A_c", "옵션 A c"), ("cell_E_c", "옵션 E c")):
    rws = json.load(open(OUT / d / "timeline_seed460.json"))["rows"]; last = rws[-1]; err = (OUT / d / "stderr.txt").read_text(); m = re.search(r"max velocity is ([0-9.]+)", err)
    print(f"| {lab} | {last['phase']} i={last['i']} q={last['q_deg']}° M={last['M_hinge_res_Nm']:.3f} 단일최대={last['max_single_contact_N']:.2f} N v_max={last['v_particle_max']:.2f} | {m.group(1) if m else '?'} m/s |")
