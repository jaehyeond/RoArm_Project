#!/usr/bin/env python3
"""로봇 URDF + S1 그랩 합성 (D480). 벤더 `roarm_m3.urdf` 텍스트에서
  ① `gripper_link`(순정 가동 조) 블록만 S1 가동부(문)로 교체 — 같은 관절 link5_to_gripper_link 가 그대로 문을 돌린다
  ② `grab_fixed` 링크(고정부: 판+스파인+반쪽 보울, link5 프레임) + `link5_to_grab_fixed` 고정 조인트 주입
메시: visual = *_ALL(정확 윤곽) · collision = 볼록 조각별(D446). 문 조각은 gripper_link 프레임으로 변환 (x=Z−52.035, y=X, z=Y−18.821).
출력: local_assets/roarm_m3/urdf/roarm_m3_s1.urdf + meshes/s1_*.stl + meshes/collision_s1/*.stl + s1_meta.json
  --tag v1 이면 s1 → s1_v1 로 이름이 바뀐다(roarm_m3_s1_v1.urdf, meshes/s1_v1_*.stl, meshes/collision_s1_v1/, s1_v1_meta.json). v0 산출은 안 건드린다(09-09).
사용: python compose_roarm_s1_urdf.py [s1 형상 폴더] [--tag v1]
"""
import sys, re, json, hashlib, shutil, argparse
from pathlib import Path
import numpy as np, trimesh

REPO = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser(); _ap.add_argument("src", nargs="?", default=str(REPO / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v0")); _ap.add_argument("--tag", default="")
_a = _ap.parse_args(); SRC = Path(_a.src).resolve(); TAG = "s1" + (f"_{_a.tag}" if _a.tag else "")
RDIR = REPO / "local_assets/roarm_m3/urdf"; MESH = RDIR / "meshes"; COL = MESH / f"collision_{TAG}"
OUT = RDIR / f"roarm_m3_{TAG}.urdf"
P = json.load(open(SRC / "design.json"))["params"]; RHO = P["density_g_cm3"] * 1e-3   # g/mm³
JAW = np.array([52.035, 0.0, 18.821])                                                   # link5 → jaw: (Z−52.035, X, Y−18.821)

def to_jaw(m):
    v = m.vertices; m2 = m.copy(); m2.vertices = np.stack([v[:, 2] - JAW[0], v[:, 0], v[:, 1] - JAW[2]], 1); m2.fix_normals(); return m2

def inertial(pieces):
    m_tot = 0.0; com = np.zeros(3)
    for m in pieces: m.density = RHO; m_tot += m.mass; com += m.mass * np.asarray(m.center_mass)
    com /= m_tot; I = np.zeros((3, 3))
    for m in pieces:
        d = np.asarray(m.center_mass) - com; I += np.asarray(m.moment_inertia) + m.mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return m_tot * 1e-3, com / 1000.0, I * 1e-9          # kg, m, kg·m²

def sha16(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]

def link_xml(name, vis_fn, col_fns, mass, com, I, rgba, extra=""):
    col = "\n".join(f'  <collision>\n    <origin xyz="0 0 0" rpy="0 0 0"/>\n    <geometry><mesh filename="meshes/{COL.name}/{c}" scale="0.001 0.001 0.001"/></geometry>\n  </collision>' for c in col_fns)
    return (f'<link name="{name}">\n  <inertial>\n    <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>\n    <mass value="{mass:.6f}"/>\n'
            f'    <inertia ixx="{I[0,0]:.6E}" ixy="{I[0,1]:.6E}" ixz="{I[0,2]:.6E}" iyy="{I[1,1]:.6E}" iyz="{I[1,2]:.6E}" izz="{I[2,2]:.6E}"/>\n  </inertial>\n'
            f'  <visual>\n    <origin xyz="0 0 0" rpy="0 0 0"/>\n    <geometry><mesh filename="meshes/{vis_fn}" scale="0.001 0.001 0.001"/></geometry>\n'
            f'    <material name="{rgba}"/>\n  </visual>\n{col}\n</link>')

def main():
    COL.mkdir(parents=True, exist_ok=True)
    for old in COL.glob("*.stl"): old.unlink()
    # 문(가동부): 조각 → jaw 프레임
    door_pieces = sorted(p for p in SRC.glob("door_*.stl") if not p.name.startswith("door_ALL"))
    dp = [to_jaw(trimesh.load(p, force="mesh")) for p in door_pieces]
    for p, m in zip(door_pieces, dp): m.export(COL / p.name)
    door_vis = to_jaw(trimesh.load(SRC / "door_ALL.stl", force="mesh")); door_vis.export(MESH / f"{TAG}_door.stl")
    # 관성 = 생성기가 낳은 시각 조각(doorvis_*.stl, 각각 닫힌 부피). split() 은 퇴화면 제거 뒤 부피를 망친다(09-03).
    parts = [to_jaw(trimesh.load(p, force="mesh")) for p in sorted(SRC.glob("doorvis_*.stl"))]
    m_d, c_d, I_d = inertial(parts if parts else dp)
    # 고정부: link5 프레임 그대로
    fixed_pieces = sorted(p for p in SRC.glob("fixed_*.stl") if not p.name.startswith("fixed_ALL"))
    for p in fixed_pieces: shutil.copy(p, COL / p.name)
    fixed_vis = trimesh.load(SRC / "fixed_ALL.stl", force="mesh"); fixed_vis.export(MESH / f"{TAG}_fixed.stl")
    fparts = [trimesh.load(p, force="mesh") for p in sorted(SRC.glob("fixedvis_*.stl"))]
    m_f, c_f, I_f = inertial(fparts if fparts else [trimesh.load(p, force="mesh") for p in fixed_pieces])
    # URDF 텍스트
    txt = (RDIR / "roarm_m3.urdf").read_text()
    pat = re.compile(r'<link name="gripper_link">.*?</link>', re.S)
    assert len(pat.findall(txt)) == 1
    new_door = link_xml("gripper_link", f"{TAG}_door.stl", [p.name for p in door_pieces], m_d, c_d, I_d, "s1_blue")
    txt2 = pat.sub(lambda _: new_door, txt)
    # --tag 경로에서만(09-09 W1b): inertial 없는 링크(hand_tcp)는 임포터 link_density 0 → PhysX 기본 1 kg 을 받아 어깨 모멘트가 10배로 부풀었다.
    #    미소 관성(1e-4 kg, 대각 1e-8 kg·m²)을 주입. 루트 world 는 fix_root_link 로 고정돼 관절 하중과 무관하므로 제외.
    tiny_links = [n for n in re.findall(r'<link name="([^"]+)"\s*/>', txt2) if n != "world"] if _a.tag else []
    for n in tiny_links:
        txt2 = re.sub(rf'<link name="{n}"\s*/>', f'<link name="{n}">\n  <inertial>\n    <origin xyz="0 0 0" rpy="0 0 0"/>\n    <mass value="0.0001"/>\n    <inertia ixx="1E-08" ixy="0" ixz="0" iyy="1E-08" iyz="0" izz="1E-08"/>\n  </inertial>\n</link>', txt2)
    fixed_block = link_xml("grab_fixed", f"{TAG}_fixed.stl", [p.name for p in fixed_pieces], m_f, c_f, I_f, "s1_green")
    joint = ('<joint name="link5_to_grab_fixed" type="fixed">\n  <origin xyz="0 0 0" rpy="0 0 0"/>\n  <parent link="link5"/>\n  <child link="grab_fixed"/>\n</joint>')
    mats = '<material name="s1_blue"><color rgba="0.20 0.45 0.85 1.0"/></material>\n<material name="s1_green"><color rgba="0.25 0.70 0.35 1.0"/></material>\n'
    # 재질은 벤더 silver 정의 바로 뒤(최상위, 사용처보다 앞)에 둔다
    i = txt2.index("</material>") + len("</material>"); txt2 = txt2[:i] + "\n" + mats + txt2[i:]
    inject = ("\n<!-- ===== S1 그랩 합성 (compose_roarm_s1_urdf.py, D480): gripper_link = 인쇄 문(포크+암+반쪽 보울), grab_fixed = 고정부. 순정 가동 조 메시 제거 ===== -->\n" + fixed_block + "\n" + joint + "\n")
    assert txt2.count("</robot>") == 1
    OUT.write_text(txt2.replace("</robot>", inject + "</robot>"))
    meta = {"source_dir": str(SRC), "base_urdf_sha16": sha16(RDIR / "roarm_m3.urdf"), "design_json_sha16": sha16(SRC / "design.json"), "door_ALL_sha16": sha16(SRC / "door_ALL.stl"), "fixed_ALL_sha16": sha16(SRC / "fixed_ALL.stl"),
            "door": {"mass_kg": round(m_d, 5), "com_m_jawframe": [round(v, 5) for v in c_d], "n_collision": len(dp)},
            "fixed": {"mass_kg": round(m_f, 5), "com_m_link5": [round(v, 5) for v in c_f], "n_collision": len(fixed_pieces)},
            "joint": "link5_to_gripper_link (벤더, 0~1.571 rad) 가 문을 직접 구동. 열림 29.3° = 0.511 rad 에서 입 58 mm",
            "frames": {"gripper_link": "x=link5 Z−52.035, y=link5 X, z=link5 Y−18.821 (벤더 rpy −90,−90,0)", "grab_fixed": "link5 프레임 동일"},
            "wrist_pitch": {"joint": "link3_to_link4", "urdf_limit_rad": 1.92, "firmware_clamp_deg": 90, "note": "URDF ±110° 는 벤더값 유지. 실물 펌웨어는 ±90° 로 클램프(SDK 는 110 통과) — D481 ①"},
            "tiny_inertial_links": {"links": tiny_links, "mass_kg": 1e-4, "inertia_diag_kgm2": 1e-8, "why": "URDF inertial 없음 → Isaac 임포터 기본 1 kg 방지 (W1b 09-09)"}}
    json.dump(meta, open(RDIR / f"{TAG}_meta.json", "w"), ensure_ascii=False, indent=1)
    import xml.etree.ElementTree as ET
    r = ET.parse(OUT).getroot(); print(f"합성 → {OUT}  링크 {len(r.findall('link'))} · 조인트 {len(r.findall('joint'))}")
    print(f"  문 {m_d*1000:.1f} g CoM(jaw) {np.round(c_d*1000,1)} mm · 고정 {m_f*1000:.1f} g CoM(link5) {np.round(c_f*1000,1)} mm · collision {len(dp)}+{len(fixed_pieces)}")
    missing = [m.get('filename') for m in r.iter('mesh') if not (RDIR / m.get('filename')).exists()]
    print("  누락 메시:", missing if missing else "없음")

if __name__ == "__main__":
    main()
