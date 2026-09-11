#!/usr/bin/env python3
"""W1b (09-09) 질량·정역학 프로브 — PhysX 가 실제 쓰는 링크별 질량/관성/COM 표 + USD 에 authored 된 physics:mass/density 유무
+ HOME·스쿱 자세에서 어깨(link1_to_link2)·팔꿈치(link2_to_link3) 중력 모멘트를 (a) Isaac applied_torque(PD 계산값) 와
(b) PhysX 질량·COM 월드좌표 정역학 재계산 Σ[(r_com−r_joint)×m·g]·axis 으로 낸다. 관절축 = 자식 링크 프레임 z(URDF axis 0 0 1).
팔 effort_limit_sim 은 포화를 피하려 8.0(비물리, 판정 미사용) — D478 규약대로 명시. D477: JSON 선기록·os._exit 워치독.
사용: OMNI_KIT_ACCEPT_EULA=YES python mass_static_probe_isaaclab.py --headless --usd <usd> --out <dir> --tag pre|post
"""
import argparse, os, json, math, time, hashlib
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True); parser.add_argument("--out", required=True); parser.add_argument("--tag", default="pre")
AppLauncher.add_app_launcher_args(parser); args = parser.parse_args()
app_launcher = AppLauncher(args); simulation_app = app_launcher.app
import torch, numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation
import omni.usd
from pxr import Usd

USD = os.path.abspath(args.usd); OUT = os.path.abspath(args.out); os.makedirs(OUT, exist_ok=True)
RES = os.path.join(OUT, f"mass_static_{args.tag}.json")
res = {"ok": False, "usd": USD, "usd_sha16": hashlib.sha256(open(USD, "rb").read()).hexdigest()[:16], "tag": args.tag, "g": 9.81, "arm_effort_limit_sim": 8.0, "note_effort": "비물리 데모값(D478) — 포화 회피용, 판정에 미사용"}
json.dump(res, open(RES, "w"), indent=1)
ARMJ = ["base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5"]; J = "link5_to_gripper_link"
POSES = {"home": {}, "scoop": {"link1_to_link2": 0.39, "link2_to_link3": 1.39, "link3_to_link4": 1.34}}
ROBOT = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=USD, articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True, enabled_self_collisions=False), rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}),
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=ARMJ, stiffness=400.0, damping=40.0, effort_limit_sim=8.0),
               "door": ImplicitActuatorCfg(joint_names_expr=[J], stiffness=300.0, damping=30.0, effort_limit_sim=1.96)})

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome = AssetBaseCfg(prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
    robot: ArticulationCfg = ROBOT

def authored(stage, bn):
    out = {}
    for b in bn:
        p = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{b}"); d = {}
        for a in ("physics:mass", "physics:density", "physics:diagonalInertia", "physics:centerOfMass"):
            at = p.GetAttribute(a); d[a] = (None if not at or not at.HasAuthoredValue() else (list(at.Get()) if hasattr(at.Get(), "__len__") else float(at.Get())))
        d["apiSchemas"] = [s for s in p.GetAppliedSchemas()] if p else None; out[b] = d
    return out

def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device)); scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=1.0)); sim.reset()
    robot: Articulation = scene["robot"]; jn = list(robot.joint_names); bn = list(robot.body_names); dt = sim.get_physics_dt()
    res["joint_names"] = jn; res["body_names"] = bn
    m = robot.root_physx_view.get_masses()[0].cpu().numpy(); I = robot.root_physx_view.get_inertias()[0].cpu().numpy().reshape(-1, 3, 3); c = robot.root_physx_view.get_coms()[0].cpu().numpy()
    res["physx_bodies"] = {b: {"mass_kg": float(m[i]), "inertia_diag_kgm2": [float(I[i][k, k]) for k in range(3)], "com_body_m": [float(v) for v in c[i][:3]]} for i, b in enumerate(bn)}
    res["physx_mass_total_kg"] = float(m.sum()); res["usd_authored"] = authored(omni.usd.get_context().get_stage(), bn)
    g = np.array([0, 0, -9.81]); tgt = torch.zeros(1, robot.num_joints, device=sim.device); res["poses"] = {}
    for pname, pose in POSES.items():
        tgt.zero_()
        for k, v in pose.items(): tgt[0, jn.index(k)] = v
        for _ in range(int(3.0 / dt)):
            robot.set_joint_position_target(tgt); scene.write_data_to_sim(); sim.step(); scene.update(dt)
        q = robot.data.joint_pos[0].cpu().numpy(); tau = robot.data.applied_torque[0].cpu().numpy()
        pw = robot.data.body_link_pos_w[0].cpu().numpy(); qw = robot.data.body_link_quat_w[0].cpu().numpy(); cw = robot.data.body_com_pos_w[0].cpu().numpy()
        rec = {"target": {k: float(tgt[0, i]) for i, k in enumerate(jn)}, "measured": {k: round(float(q[i]), 5) for i, k in enumerate(jn)}, "deflection_deg": {k: round(math.degrees(float(q[i] - tgt[0, i])), 4) for i, k in enumerate(jn)},
               "applied_torque_Nm": {k: round(float(tau[i]), 5) for i, k in enumerate(jn)}, "static": {}}
        for jname, child, distal in (("link1_to_link2", "link2", ["link2", "link3", "link4", "link5", "gripper_link", "hand_tcp", "grab_fixed"]), ("link2_to_link3", "link3", ["link3", "link4", "link5", "gripper_link", "hand_tcp", "grab_fixed"])):
            ic = bn.index(child); rj = pw[ic]; Rw = Rotation.from_quat([qw[ic][1], qw[ic][2], qw[ic][3], qw[ic][0]]).as_matrix(); ax = Rw @ np.array([0, 0, 1.0])
            M = 0.0; per = {}
            for b in distal:
                i = bn.index(b); mi = float(np.dot(np.cross(cw[i] - rj, m[i] * g), ax)); per[b] = round(mi, 5); M += mi
            rec["static"][jname] = {"gravity_moment_Nm": round(M, 5), "per_body_Nm": per, "axis_w": [round(float(v), 4) for v in ax], "joint_pos_w": [round(float(v), 4) for v in rj], "applied_torque_Nm": round(float(tau[jn.index(jname)]), 5)}
        res["poses"][pname] = rec; print("[static]", pname, json.dumps({k: v["gravity_moment_Nm"] for k, v in rec["static"].items()}), "tau", {k: rec["applied_torque_Nm"][k] for k in ("link1_to_link2", "link2_to_link3")}, flush=True)
    res["ok"] = True
    json.dump(res, open(RES, "w"), indent=1, ensure_ascii=False)
    print("[mass]", json.dumps({b: round(v["mass_kg"], 5) for b, v in res["physx_bodies"].items()}), "total", round(res["physx_mass_total_kg"], 5), flush=True)

try:
    main()
except Exception:
    import traceback; res["error"] = traceback.format_exc(); json.dump(res, open(RES, "w"), indent=1, ensure_ascii=False); print("[mass] ERROR", res["error"][-1200:], flush=True)
import threading
threading.Thread(target=lambda: (time.sleep(20), os._exit(0)), daemon=True).start()
simulation_app.close()
