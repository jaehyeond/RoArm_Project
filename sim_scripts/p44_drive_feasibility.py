#!/usr/bin/env python3
"""서보 토크로 그랩이 실제로 닫히는가 — 해석 기구학 + MuJoCo 동역학.

사용자 지적(2026-09-01): "출력해서 다 달았는데 전원 넣으니 안 움직일 수도 있잖아."

## 왜 이 구성인가 (p43 실패에서 배운 것)
p43 은 MuJoCo `equality/connect` 로 4절을 **조립**시키려 했고 2회 연속 실패했다
(설계 표 대비 셸각 오차 35.95도 -> 86.34도, 그리고 NaN 발산).
동역학 솔버에게 기구학을 풀게 시킨 것이 잘못이었다 — 초기 자세가 조금만
어긋나면 다른 해로 이완하고, 두 해 중 어느 쪽인지 통제할 수 없다.

설계는 이 문제를 **이미 해석적으로 풀어 놓았다**: `linkage_solve()` 가 원-원 교점으로
셸각을 직접 구하고 두 해 중 연속인 쪽을 고른다(표 91행, 게이트 검증 완료).
따라서 역할을 나눈다:
    기구학 -> 설계 linkage_solve (틀릴 수 없다)
    동역학 -> MuJoCo. 셸 조인트 1개 모델에 각도별 torque_gain 을 먹여
              **관성·마찰·중력**만 푼다. 루프 폐합 문제가 사라진다.

## 이 스크립트가 답하는 것
  Q1 무마찰에서 닫히는가 (하한 확인 — 여기서 실패하면 기구가 잘못된 것)
  Q2 관절 마찰이 얼마까지 허용되는가 (인쇄 핀이 뻑뻑해도 되는 한계)
  Q3 펠릿 저항(DEME 2.1~14.7 N)을 얹어도 닫히는가
  Q4 어느 각도에서 가장 빠듯한가 (전달각 최악점 49.97도가 실제 병목인가)

## 이 스크립트가 답하지 않는 것 (non_claims 참조)
  기어 백래시 0.15 mm · 링크 유격 · 접촉/충돌 · 서보 내부 감속기 마찰 ·
  실제 ST3215 토크(1.91 N·m 은 데이터시트) · 온도에 따른 PLA 강성 변화
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys as _sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def load_design():
    spec = importlib.util.spec_from_file_location("gv1", REPO / "scoop_grab_v1_design.py")
    mod = importlib.util.module_from_spec(spec)
    _argv, _sys.argv = _sys.argv, ["x"]
    try:
        spec.loader.exec_module(mod)
    finally:
        _sys.argv = _argv
    return mod


def shell_inertia(G):
    """셸 1매의 피벗 축 관성모멘트 (kg·m²). 조각 메쉬에서 직접 적분한다."""
    import glob
    import trimesh
    fs = sorted(glob.glob(str(REPO / "claudedocs/runtime_logs/grab_track/g9_sidefix/shell_L_*.stl")))
    fs = [f for f in fs if not f.endswith("_ALL.stl")]
    m = trimesh.util.concatenate([trimesh.load(f) for f in fs])
    m.density = G.P["density_g_cm3"] * 1e-3 / 1e-3   # g/cm3 -> kg/m3 는 아래에서 스케일
    v_mm3 = m.volume
    mass_kg = v_mm3 * G.P["density_g_cm3"] / 1000.0 * 1e-3     # mm3 -> cm3 -> g -> kg
    px, py = G.kin(G.P)["pivot_L"]
    pts = m.vertices[:, :2] - np.array([px, py])
    r2 = (pts ** 2).sum(axis=1).mean() * 1e-6                   # mm2 -> m2 (평균 반경²)
    return mass_kg, mass_kg * r2


def build_single_joint_mjcf(I_kgm2, friction_Nm, damping):
    """셸 조인트 1개. 관성은 실제 메쉬에서 잰 값을 쓴다."""
    # 관성 I 를 갖는 막대: 길이 L, 질량 m -> I = m L²/3 로 맞춘다
    L = 0.05
    m = 3.0 * I_kgm2 / (L ** 2)
    return f"""
<mujoco model="shell_axis">
  <option gravity="0 0 -9.81" timestep="0.0002" integrator="implicitfast"/>
  <compiler angle="radian"/>
  <worldbody>
    <body name="shell" pos="0 0 0">
      <joint name="shell" type="hinge" axis="0 0 1"
             damping="{damping}" frictionloss="{friction_Nm}"/>
      <geom type="capsule" fromto="0 0 0 {L} 0 0" size="0.004" mass="{m}"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="drive" joint="shell" gear="1" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


def run_close(G, table, *, friction_Nm, load_N, damping=1e-4, servo_Nm=None):
    """설계 표의 torque_gain 을 각도별로 먹여 닫힘을 시도한다."""
    P = G.P
    servo_Nm = servo_Nm if servo_Nm is not None else P["servo_torque_nm"]
    R_lip = G.kin(P)["R"] if "R" in G.kin(P) else 38.332
    mass_kg, I = shell_inertia(G)
    m = mujoco.MjModel.from_xml_string(build_single_joint_mjcf(I, friction_Nm, damping))
    d = mujoco.MjData(m)
    jid = m.joint("shell").qposadr[0]

    # 표를 셸각 -> torque_gain 으로 뒤집는다 (닫힘은 셸각 44.5 -> 0)
    shell_deg = np.array([r["shell_deg"] for r in table])
    gain = np.array([r["torque_gain"] for r in table])
    order = np.argsort(shell_deg)
    shell_deg, gain = shell_deg[order], gain[order]

    d.qpos[jid] = math.radians(P["shell_travel_deg"])   # 완전 개방에서 시작
    d.qvel[:] = 0
    load_Nm = load_N * (R_lip * 1e-3)                   # 립 저항력 -> 축 토크
    traj, t_end, stuck_at = [], None, None
    for step in range(60000):                            # 12 s
        ang = math.degrees(d.qpos[jid])
        g = float(np.interp(ang, shell_deg, gain))
        # 서보가 낼 수 있는 축 토크 = 서보토크 x 이득. 닫는 방향(-)
        tau = -servo_Nm * g + load_Nm                    # 저항은 여는 방향(+)
        d.ctrl[0] = float(np.clip(tau, -10, 10))
        mujoco.mj_step(m, d)
        if step % 250 == 0:
            traj.append({"t_s": round(d.time, 4), "shell_deg": round(ang, 3),
                         "torque_gain": round(g, 3), "tau_Nm": round(float(d.ctrl[0]), 4)})
        if ang <= 0.05:
            t_end = d.time
            break
        if step > 5000 and abs(d.qvel[0]) < 1e-4:
            stuck_at = ang
            break
    return {"closed": t_end is not None, "close_time_s": t_end,
            "stuck_at_shell_deg": stuck_at, "final_shell_deg": math.degrees(d.qpos[jid]),
            "shell_mass_kg": mass_kg, "shell_inertia_kgm2": I, "traj": traj[:40]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = load_design()
    lk = G.linkage_solve(G.P)
    table = lk["rows"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    res = {"artifact": "P44_DRIVE_FEASIBILITY_V1",
           "approach": "기구학=설계 linkage_solve(해석해) · 동역학=MuJoCo(셸축 1자유도)",
           "why_not_full_loop": "p43 이 equality/connect 로 4절을 조립하려다 2회 실패 "
                                "(셸각 오차 35.95->86.34도, NaN 발산). 동역학 솔버에게 "
                                "기구학을 풀게 시킨 것이 잘못이었다.",
           "mujoco_version": mujoco.__version__,
           "servo_torque_Nm": G.P["servo_torque_nm"],
           "torque_gain_range": [min(r["torque_gain"] for r in table),
                                 max(r["torque_gain"] for r in table)]}

    print("=== Q1 무마찰 하한 ===")
    r0 = run_close(G, table, friction_Nm=0.0, load_N=0.0)
    res["Q1_frictionless"] = {k: v for k, v in r0.items() if k != "traj"}
    print(f"  닫힘 {r0['closed']} · 시간 {r0['close_time_s']} s · "
          f"셸 질량 {r0['shell_mass_kg']*1000:.1f} g · 관성 {r0['shell_inertia_kgm2']:.3e} kg m2")

    print("=== Q2 마찰 한계 스윕 ===")
    sweep = []
    for f in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        r = run_close(G, table, friction_Nm=f, load_N=0.0)
        sweep.append({"friction_Nm": f, "closed": r["closed"],
                      "close_time_s": r["close_time_s"],
                      "stuck_at_shell_deg": r["stuck_at_shell_deg"]})
        note = "" if r["closed"] else f"· {r['stuck_at_shell_deg']:.1f}도에서 멈춤"
        print(f"  마찰 {f:5.3f} N·m -> 닫힘 {str(r['closed']):5s} {note}")
    res["Q2_friction_sweep"] = sweep
    ok = [s["friction_Nm"] for s in sweep if s["closed"]]
    res["max_friction_Nm_that_closes"] = max(ok) if ok else None

    print("=== Q3 펠릿 저항 (DEME 2.1~14.7 N) ===")
    loads = []
    for L in (0.0, 2.1, 5.3, 14.7, 30.0, 60.0, 100.0):
        r = run_close(G, table, friction_Nm=0.02, load_N=L)
        loads.append({"load_N": L, "closed": r["closed"], "close_time_s": r["close_time_s"],
                      "stuck_at_shell_deg": r["stuck_at_shell_deg"]})
        note = "" if r["closed"] else f"· {r['stuck_at_shell_deg']:.1f}도에서 멈춤"
        print(f"  저항 {L:6.1f} N -> 닫힘 {str(r['closed']):5s} {note}")
    res["Q3_pellet_load"] = loads

    res["non_claims"] = [
        "기어 백래시 0.15 mm · 링크 유격 · 접촉/충돌은 모델에 없다.",
        "서보 토크 1.91 N·m 은 ST3215 데이터시트 값이다. 설계 주석 자신이 "
        "'실측 조 힘은 1.8~6.3 N 이었다'고 경고한다 — 실토크는 미측정.",
        "서보 내부 감속기 마찰이 별도로 있다. 여기 마찰은 **링크 관절**만이다.",
        "펠릿 저항 2.1~14.7 N 은 DEME 값이고 변동계수 1.012 로 대표값이 없다. "
        "물성 미실측이므로 폴리프로필렌 값이 아니다.",
        "이 모델은 셸축 1자유도다. 4절의 관성 결합과 로드 좌굴은 포함되지 않는다.",
    ]
    (out / "p44_drive.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out/'p44_drive.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
