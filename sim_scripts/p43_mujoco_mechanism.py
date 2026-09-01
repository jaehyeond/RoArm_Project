#!/usr/bin/env python3
"""그랩 v1 4절 링크 — MuJoCo 동역학. **서보 토크로 실제 닫히는가?**

왜 필요한가 (2026-09-01, 사용자 지적):
  "모습만 그럴싸하게 출력해서 다 달았는데 막상 전원 넣고 돌리니 안 움직일 수도 있잖아."
  지금까지 검증된 것은 **정지 자세 91개가 서로 안 부딪힌다**(p37 G1~G9)와
  **정적 토크 계산**뿐이다. 마찰도 관성도 어디에도 없다.
  설계 JSON 에 friction 항 0건, inertia 항 0건 — 즉 운동을 푼 적이 없다.

기구 (D463·D464):
  non-Grashof triple rocker. 어느 링크도 완전 회전하지 않는다.
    고정 링크 77.811 mm  (서보축 ↔ 셸 피벗축 수직거리)
    입력 크랭크 21.0     (순정 가동 조에 볼트 체결 — 서보가 여기를 돌린다)
    로드     50.178
    출력 크랭크 39.958   (셸 피벗)
  s+l = 98.811 > p+q = 90.136  -> non-Grashof 확인.

🔴 닫힌 루프다. URDF 는 트리만 되므로 표현 불가.
   MuJoCo `equality/connect` 로 로드 끝과 출력 크랭크 핀을 묶는다.
   **이걸 빠뜨리면 링크가 끊어진 채 돌아가 "잘 움직인다"는 거짓 결과가 나온다.**
   그래서 아래 --verify 는 루프를 뺀 음성 대조부터 돌린다.

검증 순서 (이 순서를 지킬 것):
  1) 루프 없이  -> 로드 끝이 크랭크 핀에서 떨어져야 정상 (음성 대조)
  2) 루프 있음  -> 서보각→셸각이 설계 표 91행과 일치해야 정상 (양성 대조)
  3) 일치한 뒤에야 마찰·토크 동역학을 신뢰할 수 있다
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import importlib.util
import sys as _sys

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DESIGN = REPO / "claudedocs/runtime_logs/grab_track/g9_sidefix/design.json"

MM = 1e-3  # MuJoCo 는 SI. 설계는 mm.


def _load_design_module():
    """설계 스크립트를 그대로 import 한다.

    🔴 링크 치수를 손으로 옮겨 적으면 안 된다. 첫 판에서 두 축을 x 축에 나란히
       놓았다가 셸각이 설계와 최대 35.95도 어긋났다 — 방향 정보를 버린 탓이다.
       설계가 푸는 실좌표(A, B)와 linkage_solve 의 해를 **그대로** 받아쓴다.
    """
    spec = importlib.util.spec_from_file_location("gv1", REPO / "scoop_grab_v1_design.py")
    mod = importlib.util.module_from_spec(spec)
    _argv, _sys.argv = _sys.argv, ["x"]
    try:
        spec.loader.exec_module(mod)
    finally:
        _sys.argv = _argv
    return mod


def build_mjcf(G, *, close_loop=True, friction=0.0, damping=1e-4):
    """4절 + 기어 1:1 대칭을 MJCF 로. **설계 좌표를 그대로 쓴다.**"""
    P = G.P
    lk = G.linkage_solve(P)
    A = np.array(lk["servo_axis_xy"], float) * MM      # 서보축 (13.54, 73.145) mm
    B = np.array(lk["pivot_L_xy"], float) * MM         # 셸 피벗 (-13, 0) mm
    r_in = lk["crank_servo_r_mm"] * MM
    r_out = lk["crank_shell_r_mm"] * MM
    rod = lk["rod_len_mm"] * MM
    b0 = math.radians(lk["crank_servo_a0_deg"])        # -106
    g0 = math.radians(lk["crank_shell_a0_deg"])        # +128
    trav = math.radians(P["shell_travel_deg"])
    servo_max = math.radians(P["servo_travel_deg"])
    gap = P["pivot_gap_mm"] * MM

    m_tool = 49.91e-3
    m_shell, m_rod, m_crank = m_tool * 0.34, m_tool * 0.03, m_tool * 0.04

    loop = ('<connect name="loop" site1="rod_tip" site2="out_pin" '
            'solref="0.002 1" solimp="0.99 0.999 0.001"/>') if close_loop else ""

    # 크랭크 핀 위치 (시작 자세). 조인트 각 0 = 설계 시작각.
    pin_in = (r_in * math.cos(b0), r_in * math.sin(b0))
    pin_out = (r_out * math.cos(g0), r_out * math.sin(g0))

    return f"""
<mujoco model="grab_v1_linkage">
  <option gravity="0 0 -9.81" timestep="0.0002" integrator="implicitfast"/>
  <compiler angle="radian"/>
  <default>
    <joint damping="{damping}" frictionloss="{friction}"/>
    <geom rgba="0.6 0.6 0.7 1" contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="crank_in" pos="{A[0]:.6f} {A[1]:.6f} 0">
      <joint name="servo" type="hinge" axis="0 0 1" range="0 {servo_max:.6f}" limited="true"/>
      <geom type="capsule" fromto="0 0 0 {pin_in[0]:.6f} {pin_in[1]:.6f} 0"
            size="0.003" mass="{m_crank}"/>
      <body name="rod" pos="{pin_in[0]:.6f} {pin_in[1]:.6f} 0">
        <joint name="j_rod" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 {rod:.6f} 0 0" size="0.0025" mass="{m_rod}"/>
        <site name="rod_tip" pos="{rod:.6f} 0 0" size="0.001"/>
      </body>
    </body>
    <body name="crank_out" pos="{B[0]:.6f} {B[1]:.6f} 0">
      <joint name="shellL" type="hinge" axis="0 0 1" range="{-trav:.6f} 0.001" limited="true"/>
      <geom type="capsule" fromto="0 0 0 {pin_out[0]:.6f} {pin_out[1]:.6f} 0"
            size="0.003" mass="{m_crank}"/>
      <geom type="box" pos="-0.010 0.018 0" size="0.012 0.018 0.025"
            mass="{m_shell}" rgba="0.2 0.45 0.85 1"/>
      <site name="out_pin" pos="{pin_out[0]:.6f} {pin_out[1]:.6f} 0" size="0.001"/>
    </body>
    <body name="shell_r" pos="{B[0] + gap:.6f} {B[1]:.6f} 0">
      <joint name="shellR" type="hinge" axis="0 0 1"/>
      <geom type="box" pos="0.010 0.018 0" size="0.012 0.018 0.025"
            mass="{m_shell}" rgba="0.2 0.7 0.35 1"/>
    </body>
  </worldbody>
  <equality>
    {loop}
    <joint name="gear" joint1="shellR" joint2="shellL" polycoef="0 -1 0 0 0"/>
  </equality>
  <actuator>
    <motor name="servo" joint="servo" gear="1" ctrlrange="-2 2"/>
    <motor name="load" joint="shellL" gear="1" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""


def _angles(model, data):
    return (data.qpos[model.joint("servo").qposadr[0]],
            data.qpos[model.joint("shellL").qposadr[0]])


def verify_kinematics(G, table):
    """루프 없이(음성) / 있게(양성) 돌려 설계 표와 대조한다."""
    out = {}
    for tag, close in (("no_loop_negative_control", False), ("with_loop", True)):
        m = mujoco.MjModel.from_xml_string(build_mjcf(G, close_loop=close))
        d = mujoco.MjData(m)
        sid_t = m.site("rod_tip").id
        sid_p = m.site("out_pin").id
        gaps, rows = [], []
        for servo_deg in np.linspace(0, G.P["servo_travel_deg"], 19):
            d.qpos[:] = 0
            d.qpos[m.joint("servo").qposadr[0]] = math.radians(servo_deg)
            mujoco.mj_forward(m, d)
            if close:
                # 루프 구속을 만족시키도록 이완
                for _ in range(400):
                    mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            gap = float(np.linalg.norm(d.site_xpos[sid_t] - d.site_xpos[sid_p]))
            gaps.append(gap * 1000.0)
            rows.append({"servo_deg": float(servo_deg),
                         "shell_deg": math.degrees(d.qpos[m.joint("shellL").qposadr[0]]),
                         "loop_gap_mm": gap * 1000.0})
        out[tag] = {"max_loop_gap_mm": max(gaps), "mean_loop_gap_mm": float(np.mean(gaps)),
                    "rows": rows}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true", help="기구학 대조만 (동역학 없음)")
    a = ap.parse_args()

    d = json.loads(DESIGN.read_text())
    P, D = d["params"], d["derived"]
    table = (d.get("linkage") or D.get("linkage") or {}).get("table", [])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    res = {"artifact": "P43_MUJOCO_MECHANISM_V1",
           "design_source": str(DESIGN.relative_to(REPO)),
           "mujoco_version": mujoco.__version__,
           "linkage_mm": {"ground": D["servo_to_pivotL_perp_mm"], "in": P["crank_servo_r_mm"],
                          "rod": 50.178, "out": 39.9581},
           "grashof": "non-Grashof triple rocker (s+l=98.811 > p+q=90.136)"}

    G = _load_design_module()
    ver = verify_kinematics(G, table)
    res["kinematics_verification"] = {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                                      for k, v in ver.items()}
    res["kinematics_rows_with_loop"] = ver["with_loop"]["rows"]
    print("=== 1단계 기구학 대조 ===")
    print(f"  루프 없음(음성 대조) 최대 갭 {ver['no_loop_negative_control']['max_loop_gap_mm']:8.2f} mm"
          "   <- 커야 정상 (링크가 끊어져 있음)")
    print(f"  루프 있음            최대 갭 {ver['with_loop']['max_loop_gap_mm']:8.3f} mm"
          "   <- 0 에 가까워야 정상")
    if table:
        des = {round(r["servo_deg"]): r["shell_deg"] for r in table}
        errs = []
        for r in ver["with_loop"]["rows"]:
            k = min(des, key=lambda x: abs(x - r["servo_deg"]))
            if abs(k - r["servo_deg"]) < 1.5:
                errs.append(abs(abs(r["shell_deg"]) - des[k]))
        if errs:
            res["shell_angle_max_err_deg"] = max(errs)
            print(f"  설계 표 대비 셸각 최대 오차 {max(errs):.2f} deg")
    (out / "p43_mechanism.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out/'p43_mechanism.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
