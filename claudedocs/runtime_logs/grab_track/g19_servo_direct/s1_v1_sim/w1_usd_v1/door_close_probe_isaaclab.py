#!/usr/bin/env python3
"""G4 (W1, 09-09): S1 v1 합성 USD 에서 문(link5_to_gripper_link)을 start-deg 열림에서 목표 0 rad 로 닫아
문이 고정 립과 접촉해 멈추는 각을 잰다. self-collision ON(문↔고정부 충돌 필요), 팔 HOME, 중력 ON.
접촉 증거 = ContactSensor(gripper_link, 필터 grab_fixed) 힘. --no-self-collision 이면 대조군(관절 하한 0 만 작용).
D477/D478 규약: JSON 선기록, os._exit 워치독, BasicWriter 없음, effort_limit_sim 명시.
사용: OMNI_KIT_ACCEPT_EULA=YES python door_close_probe_isaaclab.py --headless --usd <usd> --out <dir> [--start-deg 20] [--no-self-collision]
"""
import argparse, os, json, math, time, hashlib
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True); parser.add_argument("--out", required=True)
parser.add_argument("--start-deg", type=float, default=20.0); parser.add_argument("--settle-s", type=float, default=4.0)
parser.add_argument("--no-self-collision", action="store_true")
AppLauncher.add_app_launcher_args(parser); args = parser.parse_args()
app_launcher = AppLauncher(args); simulation_app = app_launcher.app
import torch, numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
from pxr import Usd
import omni.usd

USD = os.path.abspath(args.usd); OUT = os.path.abspath(args.out); os.makedirs(OUT, exist_ok=True)
tag = "selfcol_off" if args.no_self_collision else "selfcol_on"
RES = os.path.join(OUT, f"door_close_{tag}.json")
res = {"ok": False, "usd": USD, "usd_sha16": hashlib.sha256(open(USD, "rb").read()).hexdigest()[:16], "self_collision": not args.no_self_collision,
       "start_deg": args.start_deg, "settle_s": args.settle_s, "door_effort_limit_Nm": 1.96, "door_pd": [300.0, 30.0], "convention": "URDF rad: 0 닫힘, + 열림 (= servo_deg 규약)"}
json.dump(res, open(RES, "w"), indent=1)
J = "link5_to_gripper_link"; ARMJ = ["base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5"]
ROBOT = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=USD, activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True, enabled_self_collisions=not args.no_self_collision),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}),
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=ARMJ, stiffness=400.0, damping=40.0, effort_limit_sim=8.0),   # 비물리 데모값 (D478)
               "door": ImplicitActuatorCfg(joint_names_expr=[J], stiffness=300.0, damping=30.0, effort_limit_sim=1.96)})  # ST3215-HS 실물 상한
CONTACT = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_link", filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/grab_fixed"], update_period=0.0, history_length=1)

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome = AssetBaseCfg(prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
    robot: ArticulationCfg = ROBOT
    contact: ContactSensorCfg = CONTACT

def collision_offsets(stage):
    out = {}
    for link in ("gripper_link", "grab_fixed"):
        vals = set()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{link}")):
            a = prim.GetAttribute("physxCollision:contactOffset"); b = prim.GetAttribute("physxCollision:restOffset")
            if a and a.HasAuthoredValue() or b and b.HasAuthoredValue():
                vals.add((float(a.Get()) if a and a.Get() is not None else None, float(b.Get()) if b and b.Get() is not None else None))
        out[link] = sorted(vals, key=str)
    return out

def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device)); scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=1.0)); sim.reset()
    robot: Articulation = scene["robot"]; cs: ContactSensor = scene["contact"]
    jn = list(robot.joint_names); iJ = jn.index(J); dt = sim.get_physics_dt()
    res["joint_names"] = jn; res["body_names"] = list(robot.body_names); res["physics_dt"] = dt
    res["collision_offsets_m"] = collision_offsets(omni.usd.get_context().get_stage())
    lim = robot.data.joint_pos_limits[0, iJ].cpu().numpy().tolist(); res["door_joint_limits_rad"] = [round(v, 4) for v in lim]
    pos = robot.data.default_joint_pos.clone(); vel = robot.data.default_joint_vel.clone(); pos[0, iJ] = math.radians(args.start_deg)
    robot.write_joint_state_to_sim(pos, vel); robot.set_joint_position_target(pos); scene.write_data_to_sim()
    for _ in range(int(0.5 / dt)):                       # 열린 상태로 0.5 s 안정
        sim.step(); scene.update(dt)
    tgt = pos.clone(); tgt[0, iJ] = 0.0
    log, t = [], 0.0; n = int(round(args.settle_s / dt))
    for k in range(n):
        robot.set_joint_position_target(tgt); scene.write_data_to_sim(); sim.step(); scene.update(dt); t += dt
        q = float(robot.data.joint_pos[0, iJ]); tau = float(robot.data.applied_torque[0, iJ])
        fm = cs.data.force_matrix_w; fn = cs.data.net_forces_w
        f_fixed = float(torch.linalg.norm(fm[0, 0, 0])) if fm is not None else None
        f_net = float(torch.linalg.norm(fn[0, 0])) if fn is not None else None
        if k % 4 == 0 or k == n - 1:
            log.append({"t": round(t, 4), "door_deg": round(math.degrees(q), 4), "tau_Nm": round(tau, 4), "F_door_fixed_N": (round(f_fixed, 4) if f_fixed is not None else None), "F_net_N": (round(f_net, 4) if f_net is not None else None)})
    json.dump(log, open(os.path.join(OUT, f"door_close_log_{tag}.json"), "w"))
    last = [r for r in log if r["t"] > t - 0.5]
    degs = [r["door_deg"] for r in last]
    first = next((r for r in log if r["F_door_fixed_N"] is not None and r["F_door_fixed_N"] > 1e-3), None)
    res.update({"stop_deg_mean_last0p5s": round(float(np.mean(degs)), 4), "stop_deg_min": round(min(degs), 4), "stop_deg_max": round(max(degs), 4),
                "settled": bool(max(degs) - min(degs) < 0.05), "F_door_fixed_N_last0p5s_mean": round(float(np.mean([r["F_door_fixed_N"] or 0.0 for r in last])), 4),
                "F_door_fixed_N_max": round(max((r["F_door_fixed_N"] or 0.0) for r in log), 4), "tau_Nm_last0p5s_mean": round(float(np.mean([r["tau_Nm"] for r in last])), 4),
                "first_contact": ({"t": first["t"], "door_deg": first["door_deg"]} if first else None), "door_deg_min_overall": round(min(r["door_deg"] for r in log), 4),
                "finite": bool(all(np.isfinite([r["door_deg"], r["tau_Nm"]]).all() for r in log)), "n_log": len(log)})
    res["ok"] = bool(res["finite"] and res["settled"])
    json.dump(res, open(RES, "w"), indent=1, ensure_ascii=False)
    print("[door_close]", json.dumps({k: v for k, v in res.items() if k not in ("joint_names", "body_names", "usd")}, ensure_ascii=False), flush=True)

try:
    main()
except Exception:
    import traceback; res["error"] = traceback.format_exc(); json.dump(res, open(RES, "w"), indent=1, ensure_ascii=False); print("[door_close] ERROR", res["error"][-1200:], flush=True)
import threading
threading.Thread(target=lambda: (time.sleep(20), os._exit(0)), daemon=True).start()
simulation_app.close()
