#!/usr/bin/env python3
"""STL 을 베드 접지가 가장 큰 면 위에 눕힌다.

왜 필요한가 (2026-09-01, 출력 2연속 실패):
  4부품을 설계 배향 그대로 슬라이스했더니 1층 접지 총합이 **1.4 cm²** 였다.
  shell 은 9 mm², linkage 는 14 mm² — 세워 놓은 것과 같다. 첫 몇 층에서 떨어져
  나갔고 노즐은 50층까지 허공에 뽑았다 (P1S 는 탈락을 감지하지 못한다).

  `--orient 1`(슬라이서 자동 배향)은 답이 아니다. 09-01 실측에서 오히려 높이를
  59.5 -> 60.7 mm 로 키웠다. 접지가 아니라 다른 것을 최적화한다.

선택 규칙 (재현 가능 — 눈대중 아님):
  볼록껍질의 각 면 법선을 바닥으로 돌려 보고 (접지면적, 높이)를 잰다.
  접지가 최대인 것을 고르되, **최대의 80% 이상이면서 20% 이상 낮은** 배향이 있으면
  그쪽을 택한다. 높은 부품일수록 냉각 수축으로 모서리가 들리기 때문이다.

접지면적 정의: 최저점에서 0.3 mm 안에 있고 법선이 아래를 향하는(n_z < -0.7)
삼각형 면적의 합. 브림은 포함하지 않는다 — 브림은 슬라이서가 따로 붙인다.
"""
import sys, json, math
from pathlib import Path
import numpy as np
import trimesh

TOL_MM = 0.3          # 바닥으로 칠 높이 여유
NZ_DOWN = -0.7        # 아래를 향한다고 볼 법선 z
AREA_KEEP = 0.80      # 최대 접지의 이 비율 이상이면 후보
HEIGHT_WIN = 0.80     # 그중 높이가 이 배 이하로 낮으면 갈아탄다

# 🔴 2026-09-01 2차 실패에서 추가. 접지만 보고 고르면 안 된다.
# 1차 판(접지만 최대화)은 접지 78 -> 467 mm² 로 고쳤지만 그 대가로
#   · 기어축이 90° 누워 이빨이 층으로 쌓였고 (형상·강도 둘 다 망가짐)
#   · 오버행이 843 -> 4,892 mm² 로 5.8배 늘었다 (서포트 금지라 전부 늘어짐)
# 힌지축(설계 Z)에는 기어·피벗보스·핀·로드아이·셸크랭크허브가 **전부 동축**이다.
# 이 축을 수직으로 세우면 이빨이 면내로 찍히고 모든 보어가 진원으로 나온다.
FUNC_AXIS = np.array([0.0, 0.0, 1.0])   # 설계 좌표계의 힌지축
FUNC_TILT_MAX_DEG = 10.0                # 이 이상 기울면 후보에서 제외
OVERHANG_DEG = 45.0                     # 이보다 가파르면 서포트 필요 구간


def contact_area(m):
    z = m.vertices[:, 2].min()
    c = m.triangles.mean(axis=1)
    sel = (c[:, 2] < z + TOL_MM) & (m.face_normals[:, 2] < NZ_DOWN)
    return float(m.area_faces[sel].sum())


def overhang_area(m):
    """서포트 없이는 늘어지는 면적 (mm²). 법선이 아래를 향하고 수평에서 많이 기운 면."""
    lim = -math.cos(math.radians(90 - OVERHANG_DEG))
    return float(m.area_faces[m.face_normals[:, 2] < lim].sum())


def best_orientation(m):
    """(회전행렬, 접지, 높이, 후보표) 를 돌려준다.

    기능축이 수직인 배향만 후보로 본다 — 접지는 그 안에서 고른다.
    접지가 아무리 커도 기어 이빨이 층으로 쌓이면 쓸 수 없는 부품이 나온다.
    """
    cands, seen = [], set()
    for nrm in m.convex_hull.face_normals:
        key = tuple(np.round(nrm, 2))
        if key in seen:
            continue
        seen.add(key)
        R = trimesh.geometry.align_vectors(nrm, [0, 0, -1])
        mm = m.copy()
        mm.apply_transform(R)
        ax = R[:3, :3] @ FUNC_AXIS
        tilt = math.degrees(math.acos(min(1.0, abs(float(ax[2])))))
        cands.append({"R": R, "area": contact_area(mm), "height": float(mm.extents[2]),
                      "overhang": overhang_area(mm), "func_tilt_deg": round(tilt, 2),
                      "normal": [round(float(v), 3) for v in nrm]})
    ok = [c for c in cands if c["func_tilt_deg"] <= FUNC_TILT_MAX_DEG]
    if not ok:
        raise SystemExit(f"기능축을 {FUNC_TILT_MAX_DEG}° 안으로 세우는 배향이 없다 — "
                         f"형상이나 FUNC_AXIS 를 다시 볼 것")
    cands = ok
    top = max(cands, key=lambda c: c["area"])
    # 순서 의존이면 안 된다. 거울상 부품이 다른 답을 내면 규칙이 틀린 것이다
    # (첫 판에서 shell_L 449/33.4 · shell_R 467/28.1 로 갈렸다 — greedy 누적 탓).
    # 접지가 최대의 AREA_KEEP 이상인 후보 중 **가장 낮은 것**, 동률이면 접지가 큰 것.
    keep = [c for c in cands if c["area"] >= AREA_KEEP * top["area"]]
    pick = min(keep, key=lambda c: (round(c["height"], 3), -c["area"]))
    return pick, top, cands


def main():
    if len(sys.argv) < 3:
        sys.exit("사용: orient_for_print.py <출력디렉터리> <STL...>")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    report = {"rule": {"tol_mm": TOL_MM, "nz_down": NZ_DOWN,
                       "area_keep": AREA_KEEP, "height_win": HEIGHT_WIN},
              "parts": []}
    for src in sys.argv[2:]:
        src = Path(src)
        m = trimesh.load(src)
        before = {"area": contact_area(m), "height": float(m.extents[2]),
                  "overhang": overhang_area(m)}
        pick, top, _ = best_orientation(m)
        m.apply_transform(pick["R"])
        m.apply_translation([0, 0, -m.vertices[:, 2].min()])   # 베드에 앉힌다
        dst = out / src.name
        m.export(dst)
        row = {"stl": dst.name,
               "before": {k: round(v, 1) for k, v in before.items()},
               "after": {"area": round(pick["area"], 1),
                         "height": round(pick["height"], 1),
                         "overhang": round(pick["overhang"], 1),
                         "func_tilt_deg": pick["func_tilt_deg"],
                         "normal_to_bed": pick["normal"]},
               "max_area_option": {"area": round(top["area"], 1),
                                   "height": round(top["height"], 1)},
               "gain": round(pick["area"] / before["area"], 1) if before["area"] > 0 else None}
        report["parts"].append(row)
        print(f"{src.name:20s} 접지 {before['area']:6.0f} -> {pick['area']:6.0f} mm² · "
              f"높이 {before['height']:5.1f} -> {pick['height']:5.1f} · "
              f"오버행 {before['overhang']:6.0f} -> {pick['overhang']:6.0f} mm² · "
              f"기능축 {pick['func_tilt_deg']:4.1f}°")
    (out / "orientation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out/'orientation.json'}")


if __name__ == "__main__":
    main()
