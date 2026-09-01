#!/usr/bin/env python3
"""그랩 v1 조립 상태를 웹 뷰어용 데이터로 내보낸다.

부품 STL 은 각자 자기 로컬 좌표에 있어서 그대로 겹쳐 보면 조립이 안 보인다.
`p37_g2_grab_v1_attach_probe.placement()` 가 쓰는 것과 **같은 변환**으로 link5
프레임에 올려서, 팔 끝(link5, 순정 고정 조)에 어떻게 붙는지 그대로 보여준다.

출력: assembly.json  — 부품별 삼각형 배열(mm, link5 프레임) + 조립 메타.
      좌표계는 link5 기준이고 뷰어가 보기 좋게 중심만 옮긴다.
"""
import json, sys
from pathlib import Path
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent
SRC = REPO / "claudedocs/runtime_logs/grab_track/g3_linkage"   # 🔴 설계 좌표계 STL.
# g4_flat / g5_oriented 의 STL 은 **출력용으로 눕힌 좌표**라 조립 변환을 걸면 틀린다
# (셸 L/R bbox 가 동일하게 나오는 것이 그 증거였다). 형상은 같고 자세만 다르다.
LINK5 = REPO / "local_assets/roarm_m3/urdf/meshes/link5.stl"
DESIGN = REPO / "claudedocs/runtime_logs/grab_track/g3_linkage/design.json"
OUT = REPO / "claudedocs/runtime_logs/grab_track/g6_viewer"

# p37 placement() 와 동일. 여기서 다시 유도하지 않고 **그대로 옮겨 적는다** —
# 두 곳이 갈라지면 뷰어가 조용히 거짓 조립을 보여준다.
BLADE_HOLES_YZ = [(-13.34, 83.46), (11.85, 83.46), (-13.34, 102.90), (11.85, 102.90)]
BLADE_X = (-11.54, -10.03)


def placement(bracket_thk, standoff):
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, -1.0, 0.0]])
    cy = float(np.mean([h[0] for h in BLADE_HOLES_YZ]))
    cz = float(np.mean([h[1] for h in BLADE_HOLES_YZ]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [BLADE_X[0] - bracket_thk / 2.0, cy, cz + standoff]
    return T


PARTS = [
    ("link5",   LINK5,                   "#6b7280", "팔 끝 링크 + 순정 고정 조 (출력 대상 아님)"),
    ("bracket", SRC / "bracket_ALL.stl", "#f59e0b", "브래킷 — 고정 조 4볼트 사각형에 물려 피벗 2개를 세운다"),
    ("shell_L", SRC / "shell_L_ALL.stl", "#3b82f6", "왼쪽 셸 (조개 반쪽)"),
    ("shell_R", SRC / "shell_R_ALL.stl", "#22c55e", "오른쪽 셸 (조개 반쪽)"),
    ("linkage", SRC / "linkage_ALL.stl", "#ef4444", "4절 링크 — 서보 회전을 두 셸 대칭 폐합으로 바꾼다"),
]


def main():
    P = json.loads(DESIGN.read_text())["params"]
    D = json.loads(DESIGN.read_text())["derived"]
    T = placement(P["bracket_thk_mm"], P["bracket_standoff_mm"])
    out = {"frame": "link5 (mm)", "parts": [],
           "design": {"pivot_gap_mm": P["pivot_gap_mm"],
                      "mouth_open_mm": P["mouth_open_mm"],
                      "shell_travel_deg": P["shell_travel_deg"],
                      "bracket_bolt_dy_mm": P["bracket_bolt_dy_mm"],
                      "bracket_bolt_dz_mm": P["bracket_bolt_dz_mm"],
                      "bolt_clear_d_mm": P["bolt_clear_d_mm"],
                      "pivot_shaft_d_mm": P["pivot_shaft_d_mm"],
                      "jaw_bolt_yz_mm": P["jaw_bolt_yz_mm"],
                      "tool_mass_g": D["tool_mass_g"],
                      "load_per_scoop_g": D["load_per_scoop_g"],
                      "self_load_ratio": D["self_load_ratio"]},
           "bolt_holes_link5_yz": BLADE_HOLES_YZ}

    for name, path, color, note in PARTS:
        if not path.exists():
            print(f"  [건너뜀] {name}: {path} 없음")
            continue
        m = trimesh.load(path)
        if name != "link5":                      # 순정 부품은 이미 link5 프레임
            m.apply_transform(T)
        v = m.vertices[m.faces].reshape(-1, 3).astype(np.float32)
        # 법선은 내보내지 않는다 — 뷰어가 삼각형에서 직접 구한다(평면 셰이딩).
        # 그냥 넣으면 JSON 이 2배가 되고, 어차피 같은 정보다.
        out["parts"].append({
            "name": name, "color": color, "note": note,
            "printed": name != "link5",
            "tris": len(m.faces),
            "verts": [round(float(x), 2) for x in v.ravel()],
        })
        print(f"  {name:8s} 삼각형 {len(m.faces):6d}  bbox "
              f"{np.round(m.bounds[0],1)} ~ {np.round(m.bounds[1],1)}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "assembly.json"
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"-> {p}  ({p.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
