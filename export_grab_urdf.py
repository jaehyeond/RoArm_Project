#!/usr/bin/env python3
"""Phase 3 — 그랩 v1 을 **구동 1축 URDF** 로 내보낸다 (D473 결정).

URDF 트리 (폐루프 = 기어쌍·4절 링크는 생략, D472 ④: 닫힌 루프는 URDF 트리로 표현 불가):
    grab_base (브래킷+요크, 고정)
      → [grab_shell_L_joint: revolute, 축 = 힌지 -Z, 원점 = 피벗 L] → grab_shell_L
      → [grab_shell_R_joint: revolute mimic(shell_L, ×1), 축 = 힌지 +Z, 원점 = 피벗 R] → grab_shell_R

- 관절 변수 = **셸 물리 회전각**(0~44.5°). 서보각(0~89°)→셸각→개구(mm) **비선형**(D463)은
  URDF 로 표현 불가(revolute 는 선형)이므로 companion JSON 의 표로 남긴다(스크립트 후처리).
- 셸 L 은 -phi, 셸 R 은 +phi(거울)로 벌어진다 → shell_L 축 -Z, shell_R 축 +Z, mimic ×1.
- 관성: _ALL.stl 은 볼록 조각들의 합이라 watertight 가 아닐 수 있으므로, **설계 스크립트의
  볼록 조각을 다시 만들어** 조각별 질량특성을 합산(신뢰성). 밀도 = PLA 1.24 g/cm³(하드웨어 볼트 제외).
- 🔴 손목 롤 제약(D473 ⑥): 개구>44 mm → |롤| 제한. URDF 관절 한계로는 못 거는 **구성 의존** 제약이라
  companion JSON 에 표로 남긴다(제어/시뮬이 읽어 강제).

출력: claudedocs/runtime_logs/grab_track/g17_yoke_alu/urdf/{grab_v1.urdf, meshes/*.stl, grab_v1_meta.json}
"""
import sys, json, math, shutil, hashlib
from pathlib import Path
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent
sys.argv = ["x"]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "sim_scripts"))
import scoop_grab_v1_design as G

SRC = REPO / "claudedocs/runtime_logs/grab_track/g17_yoke_alu"
OUT = SRC / "urdf"
MESH_OUT = OUT / "meshes"
P, K = G.P, G.kin(G.P)
RHO = P["density_g_cm3"] * 1e-3          # g/mm³
travel = math.radians(P["shell_travel_deg"])
pivL = np.array([K["pivot_L"][0], K["pivot_L"][1], 0.0])   # mm
pivR = np.array([K["pivot_R"][0], K["pivot_R"][1], 0.0])


def body_inertial(pieces):
    """볼록 조각 집합의 (질량 kg, CoM mm(grab-local), 관성텐서 kg·m² about CoM)."""
    m_tot = 0.0
    com = np.zeros(3)
    for m in pieces:
        m.density = RHO
        m_tot += m.mass
        com += m.mass * np.asarray(m.center_mass)
    com /= m_tot
    I = np.zeros((3, 3))
    for m in pieces:
        Ic = np.asarray(m.moment_inertia)        # g·mm² about piece CoM, grab-local 축
        d = np.asarray(m.center_mass) - com      # mm
        I += Ic + m.mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
    I_si = I * 1e-9                               # g·mm² → kg·m²  (×1e-3 ×1e-6)
    return m_tot * 1e-3, com, I_si                # kg, mm, kg·m²


def rpy_from_R(Rm):
    """ZYX(roll,pitch,yaw) URDF rpy 를 R 에서 역산."""
    sy = -Rm[2, 0]
    if abs(sy) < 0.999999:
        pitch = math.asin(sy)
        roll = math.atan2(Rm[2, 1], Rm[2, 2])
        yaw = math.atan2(Rm[1, 0], Rm[0, 0])
    else:                                          # gimbal
        pitch = math.copysign(math.pi / 2, sy)
        roll = math.atan2(-Rm[1, 2], Rm[1, 1]); yaw = 0.0
    return roll, pitch, yaw


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    MESH_OUT.mkdir(parents=True, exist_ok=True)

    # 1) 몸체별 조각 → 관성. (설계 스크립트에서 다시 생성 = 볼록·watertight 보장)
    base_p, base_n = G.build_bracket(P)
    sL_p, sL_n = G.build_shell(P, -1)
    sR_p, sR_n = G.build_shell(P, +1)
    bodies = {}
    for name, pieces, joint_org in (("grab_base", base_p, np.zeros(3)),
                                    ("grab_shell_L", sL_p, pivL),
                                    ("grab_shell_R", sR_p, pivR)):
        mass, com_mm, I = body_inertial([p.copy() for p in pieces])
        com_link = (com_mm - joint_org) / 1000.0            # link 프레임 CoM (m)
        bodies[name] = dict(mass=mass, com=com_link, I=I, joint_org=joint_org)

    # 2) 메시 복사 (source sha 기록 — D470)
    mesh_src = {"grab_base": "bracket_ALL.stl",
                "grab_shell_L": "shell_L_ALL.stl", "grab_shell_R": "shell_R_ALL.stl"}
    shas = {}
    for link, fn in mesh_src.items():
        shutil.copy(SRC / fn, MESH_OUT / fn)
        shas[fn] = sha16(SRC / fn)

    # 3) placement (grab-local -> link5) = 마운트 고정 조인트 origin
    src37 = open(REPO / "sim_scripts/p37_g2_grab_v1_attach_probe.py").read().split("def main()")[0]
    ns = {"__file__": str(REPO / "sim_scripts/p37_g2_grab_v1_attach_probe.py")}
    exec(compile(src37, "p37", "exec"), ns)
    T = ns["placement"]()
    roll, pitch, yaw = rpy_from_R(T[:3, :3])
    mount_xyz = (T[:3, 3] / 1000.0).tolist()

    # 4) URDF 작성
    def inertial_xml(b, indent):
        c = b["com"]; I = b["I"]
        return (f'{indent}<inertial>\n'
                f'{indent}  <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>\n'
                f'{indent}  <mass value="{b["mass"]:.7f}"/>\n'
                f'{indent}  <inertia ixx="{I[0,0]:.7E}" ixy="{I[0,1]:.7E}" ixz="{I[0,2]:.7E}" '
                f'iyy="{I[1,1]:.7E}" iyz="{I[1,2]:.7E}" izz="{I[2,2]:.7E}"/>\n'
                f'{indent}</inertial>\n')

    def link_xml(name, mesh_fn, vis_org):
        b = bodies[name]
        o = f'{vis_org[0]:.6f} {vis_org[1]:.6f} {vis_org[2]:.6f}'
        return (f'  <link name="{name}">\n'
                + inertial_xml(b, "    ")
                + f'    <visual>\n      <origin xyz="{o}" rpy="0 0 0"/>\n'
                f'      <geometry><mesh filename="meshes/{mesh_fn}" scale="0.001 0.001 0.001"/></geometry>\n'
                f'      <material name="grab_orange"/>\n    </visual>\n'
                f'    <collision>\n      <origin xyz="{o}" rpy="0 0 0"/>\n'
                f'      <geometry><mesh filename="meshes/{mesh_fn}" scale="0.001 0.001 0.001"/></geometry>\n'
                f'    </collision>\n  </link>\n')

    vis_L = (-pivL) / 1000.0
    vis_R = (-pivR) / 1000.0
    pL = pivL / 1000.0; pR = pivR / 1000.0
    urdf = f'''<?xml version="1.0"?>
<!-- 그랩 v1 Phase 3 구동 1축 URDF (D473). 자동 생성: export_grab_urdf.py
     source = grab_track/g17_yoke_alu/ · 폐루프(기어쌍·4절) 생략 · 관절 = 셸 물리각(선형)
     🔴 서보→셸→개구 비선형 + 손목 롤 제약은 grab_v1_meta.json 참조.
     link5 부착(고정 조인트) origin: xyz(m) {mount_xyz} rpy {[round(roll,5),round(pitch,5),round(yaw,5)]}
       parent=link5(순정)  child=grab_base  — 로봇과 합성 시 이 조인트를 추가한다. -->
<robot name="grab_v1">
  <material name="grab_orange"><color rgba="0.90 0.55 0.10 1"/></material>

{link_xml("grab_base", "bracket_ALL.stl", (0.0, 0.0, 0.0))}
{link_xml("grab_shell_L", "shell_L_ALL.stl", vis_L)}
{link_xml("grab_shell_R", "shell_R_ALL.stl", vis_R)}
  <!-- 구동 1축: 셸 L. 셸은 -phi 로 벌어지므로 축 = 힌지 -Z. 한계 = 셸 물리각 0~44.5°. -->
  <joint name="grab_shell_L_joint" type="revolute">
    <origin xyz="{pL[0]:.6f} {pL[1]:.6f} {pL[2]:.6f}" rpy="0 0 0"/>
    <parent link="grab_base"/>
    <child link="grab_shell_L"/>
    <axis xyz="0 0 -1"/>
    <limit lower="0.0" upper="{travel:.5f}" effort="2.94" velocity="3.14"/>
  </joint>
  <!-- 셸 R = 셸 L 의 거울. 축 = 힌지 +Z, mimic ×1(같은 크기 반대 방향). -->
  <joint name="grab_shell_R_joint" type="revolute">
    <origin xyz="{pR[0]:.6f} {pR[1]:.6f} {pR[2]:.6f}" rpy="0 0 0"/>
    <parent link="grab_base"/>
    <child link="grab_shell_R"/>
    <axis xyz="0 0 1"/>
    <limit lower="0.0" upper="{travel:.5f}" effort="2.94" velocity="3.14"/>
    <mimic joint="grab_shell_L_joint" multiplier="1" offset="0"/>
  </joint>
</robot>
'''
    (OUT / "grab_v1.urdf").write_text(urdf)

    # 5) companion meta
    lk = G.linkage_solve(P); rows = lk["rows"]
    table = [[round(r["servo_deg"], 2), round(r["shell_deg"], 3), round(r["mouth_mm"], 2)]
             for r in rows]
    meta = {
        "artifact": "grab_v1_urdf/1", "generator": "export_grab_urdf.py",
        "source_dir": str(SRC.relative_to(REPO)), "source_sha256_16": shas,
        "design_json_sha256_16": sha16(SRC / "design.json"),
        "tree": "grab_base -> shell_L(revolute,-Z,pivotL) + shell_R(mimic x1,+Z,pivotR)",
        "joint_var": "셸 물리 회전각(rad). 0=닫힘, 0.77667=완전개방(44.5°)",
        "omitted": "기어쌍·4절 링크(폐루프, URDF 트리 표현 불가 — D472 ④). 링크는 시뮬 개폐를 흉내낼 뿐.",
        "link5_attach_fixed_joint": {"parent": "link5", "child": "grab_base",
                                     "origin_xyz_m": [round(x, 6) for x in mount_xyz],
                                     "origin_rpy_rad": [round(roll, 6), round(pitch, 6), round(yaw, 6)],
                                     "note": "p37 placement() 와 동일. 로봇 URDF 와 합성 시 추가."},
        "servo_shell_mouth_nonlinear": {"columns": ["servo_deg", "shell_deg", "mouth_mm"],
                                        "rows": table,
                                        "why": "서보각→셸각→개구는 비선형(D463). URDF revolute 는 선형이라 관절=셸각, 서보 매핑은 이 표로."},
        "wrist_roll_constraint": {"rule": "개구>44 mm → 손목 롤 제한(구성 의존, D473 ⑥)",
                                  "safe_roll_abs_deg_by_mouth_mm": {"0": 180, "29": 180, "44": 106, "52": 8, "58": 14},
                                  "collider": "servocrank(링크 선재), 요크 아님. URDF 관절 한계로는 못 건다 — 제어/시뮬이 이 표를 읽어 강제."},
        "non_claims": [
            "관절=셸 물리각(정확). **서보 토크·기어 전동·백래시·유격은 표현 안 함**(Phase 4 측정 대상).",
            "collision = visual _ALL 메시(볼록 분해 안 됨). 시뮬 임포트가 분해해야 정확한 접촉.",
            "관성 = **3 몸체(브래킷+셸2)의 출력물(PLA)만**. 🔴 4절 링크(약 8 g)는 폐루프라 생략 → "
            "URDF 총질량 45.6 g < 설계 출력물 53.6 g(차이 = 링크). 알루 볼트 등 하드웨어 7.8 g 도 미포함.",
            "mimic 태그는 파서마다 처리 다름(Isaac/MuJoCo 확인 필요). 폐루프 강성은 없음.",
            "link5 부착은 **문서화**일 뿐 — 이 URDF 자체는 grab_base 루트 standalone.",
        ],
    }
    (OUT / "grab_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"URDF  → {OUT / 'grab_v1.urdf'}")
    for n, b in bodies.items():
        print(f"  {n:14s} mass {b['mass']*1000:6.2f} g  CoM(link,mm) "
              f"[{b['com'][0]*1000:6.1f} {b['com'][1]*1000:6.1f} {b['com'][2]*1000:6.1f}]")
    print(f"셸 travel = {P['shell_travel_deg']}° = {travel:.5f} rad · mimic ×1 · 축 -Z/+Z")
    print(f"link5 부착 origin xyz(m) {[round(x,4) for x in mount_xyz]} rpy {[round(roll,3),round(pitch,3),round(yaw,3)]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
