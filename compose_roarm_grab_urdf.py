#!/usr/bin/env python3
"""로봇 URDF + 그랩 v1 합성 (link5 부착 고정 조인트). D473 후속.

벤더 `roarm_m3.urdf` 는 **무수정** — 그 텍스트를 그대로 두고 `</robot>` 앞에
그랩 3링크 + 셸 2조인트 + `link5_to_grab_base` 고정 조인트를 주입해 새 파일로 쓴다.
그랩 링크/조인트는 **`grab_v1.urdf` 에서 읽어** 메시명만 로봇 meshes 폴더 규약으로 재작성한다
(단일 소스 — export_grab_urdf.py 가 낳은 자산이 정본). 부착 변환은 `grab_v1_meta.json`.

출력: local_assets/roarm_m3/urdf/roarm_m3_with_grab.urdf
      + meshes/{grab_base,grab_shell_L,grab_shell_R}.stl (그랩 메시 복사)
"""
import sys, json, shutil, re
from pathlib import Path
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parent
ROBOT_DIR = REPO / "local_assets/roarm_m3/urdf"
ROBOT_URDF = ROBOT_DIR / "roarm_m3.urdf"
ROBOT_MESH = ROBOT_DIR / "meshes"
GRAB_DIR = REPO / "claudedocs/runtime_logs/grab_track/g17_yoke_alu/urdf"
OUT = ROBOT_DIR / "roarm_m3_with_grab.urdf"

# 그랩 메시 → 로봇 meshes 폴더로 (grab_ 접두사 = 우리 자산 표시, 벤더와 구분)
MESH_MAP = {"meshes/bracket_ALL.stl": "meshes/grab_base.stl",
            "meshes/shell_L_ALL.stl": "meshes/grab_shell_L.stl",
            "meshes/shell_R_ALL.stl": "meshes/grab_shell_R.stl"}
SRC_MESH = {"meshes/grab_base.stl": GRAB_DIR / "meshes/bracket_ALL.stl",
            "meshes/grab_shell_L.stl": GRAB_DIR / "meshes/shell_L_ALL.stl",
            "meshes/grab_shell_R.stl": GRAB_DIR / "meshes/shell_R_ALL.stl"}


def main():
    meta = json.loads((GRAB_DIR / "grab_v1_meta.json").read_text())
    att = meta["link5_attach_fixed_joint"]
    xyz = att["origin_xyz_m"]; rpy = att["origin_rpy_rad"]

    # 1) 그랩 메시 복사
    for dst, src in SRC_MESH.items():
        shutil.copy(src, ROBOT_DIR / dst)

    # 2) grab_v1.urdf 에서 링크/조인트 추출 + 메시명 재작성
    groot = ET.parse(GRAB_DIR / "grab_v1.urdf").getroot()
    ET.register_namespace("", "")
    blocks = []
    for el in list(groot):
        if el.tag == "material":                      # grab_orange 재정의(로봇엔 없음)
            blocks.append(ET.tostring(el, encoding="unicode").strip())
        elif el.tag == "link":
            for mesh in el.iter("mesh"):
                fn = mesh.get("filename")
                if fn in MESH_MAP:
                    mesh.set("filename", MESH_MAP[fn])
            blocks.append(ET.tostring(el, encoding="unicode").strip())
        elif el.tag == "joint":                       # 셸 2조인트
            blocks.append(ET.tostring(el, encoding="unicode").strip())

    # 3) link5 -> grab_base 고정 조인트
    fixed = (f'<joint name="link5_to_grab_base" type="fixed">\n'
             f'  <origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" '
             f'rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>\n'
             f'  <parent link="link5"/>\n  <child link="grab_base"/>\n</joint>')

    # 4) 벤더 URDF 텍스트에 </robot> 앞으로 주입 (벤더 텍스트 무수정)
    robot_txt = ROBOT_URDF.read_text()
    inject = ("\n<!-- ===== 그랩 v1 합성 (compose_roarm_grab_urdf.py, D473) =====\n"
              "     link5_to_grab_base 고정 조인트 + 셸 2조인트(구동 1축 + mimic).\n"
              "     폐루프·비선형·손목롤 제약은 grab_v1_meta.json 참조. 순정 gripper_link 는 유지(D462 드라이브아웃). -->\n"
              + "\n".join(blocks) + "\n" + fixed + "\n")
    assert robot_txt.count("</robot>") == 1
    composed = robot_txt.replace("</robot>", inject + "</robot>")
    OUT.write_text(composed)

    # 5) 무결성: 벤더 텍스트가 통째로 보존됐는가 (주입 전 원문이 부분열로 존재)
    vendor_body = robot_txt[:robot_txt.rindex("</robot>")]
    assert vendor_body in composed, "벤더 URDF 텍스트가 변형됐다"
    print(f"합성 URDF → {OUT}")
    print(f"  주입: 그랩 링크 3 + 셸 조인트 2 + link5_to_grab_base 고정")
    print(f"  부착 origin xyz(m) {xyz} rpy {rpy}")
    print(f"  그랩 메시 복사: {list(SRC_MESH)}")
    # 링크/조인트 수 카운트
    ctree = ET.parse(OUT).getroot()
    print(f"  합성 후 링크 {len(ctree.findall('link'))} · 조인트 {len(ctree.findall('joint'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
