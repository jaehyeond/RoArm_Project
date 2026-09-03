#!/usr/bin/env python3
"""Isaac Lab 병렬 스모크 — 합성 USD(로봇+그랩 g18)를 N 개 환경으로 복제해 스텝·구동·계측 (D476 후속).

무엇을 답하나
    1. N 환경 로드가 되는가 (관절 수·몸체 수·collision 353 조각 × N)
    2. 셸 관절을 0 ↔ 0.7767 rad 로 구동하면 전 환경이 추종하는가 (최대 오차)
    3. NaN·발산 없이 스텝되는가, 스텝 시간은 얼마인가 (수천 환경 외삽 근거)
    4. (--enable_cameras) 격자 스냅샷 PNG
⚠️ D447: 예외는 SimulationApp 이 삼킨다 → 결과 JSON 을 호출부가 검증. 물리 접촉(펠릿)은 여기 없다(DEME 별개).
사용: OMNI_KIT_ACCEPT_EULA=YES python sim_isaaclab_parallel_smoke.py --headless [--enable_cameras] --num_envs 64 --out <dir>
"""
import argparse, os, sys, time, json
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--usd", type=str, default="local_assets/roarm_m3/usd/roarm_m3_with_grab.usd")
parser.add_argument("--out", type=str, default="claudedocs/runtime_logs/grab_track/g18_nut_trap/isaaclab_smoke")
parser.add_argument("--steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass

USD = os.path.abspath(args.usd); OUT = os.path.abspath(args.out); os.makedirs(OUT, exist_ok=True)
res = {"num_envs": args.num_envs, "usd": USD, "ok": False}

ROBOT = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True, enabled_self_collisions=False),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=300.0, damping=30.0)},
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome = AssetBaseCfg(prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
    robot: ArticulationCfg = ROBOT


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device))
    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=0.9))
    sim.reset()
    robot: Articulation = scene["robot"]
    jn = list(robot.joint_names); bn = list(robot.body_names)
    res["joint_names"] = jn; res["body_names"] = bn
    iL, iR = jn.index("grab_shell_L_joint"), jn.index("grab_shell_R_joint")
    dt = sim.get_physics_dt()
    tgt = torch.zeros(args.num_envs, robot.num_joints, device=sim.device)
    errs, tstep = [], []
    open_val = 0.77667
    for k in range(args.steps):
        phase = 0.5 * (1 - torch.cos(torch.tensor(2 * 3.14159265 * k / args.steps)))   # 0→1→0
        tgt[:, iL] = open_val * phase; tgt[:, iR] = open_val * phase
        robot.set_joint_position_target(tgt)
        scene.write_data_to_sim()
        t0 = time.perf_counter(); sim.step(); tstep.append(time.perf_counter() - t0)
        scene.update(dt)
        q = robot.data.joint_pos
        errs.append(float((q[:, [iL, iR]] - tgt[:, [iL, iR]]).abs().max()))
    q = robot.data.joint_pos; pos = robot.data.body_pos_w
    res.update({
        "num_joints": robot.num_joints, "num_bodies": robot.num_bodies,
        "finite": bool(torch.isfinite(q).all() and torch.isfinite(pos).all()),
        "shell_track_err_max_rad": max(errs), "shell_track_err_last_rad": errs[-1],
        "shell_pos_final_rad": [float(q[:, iL].mean()), float(q[:, iR].mean())],
        "shell_pos_spread_across_envs_rad": float((q[:, iL].max() - q[:, iL].min())),
        "arm_joint_abs_max_rad": float(q[:, :5].abs().max()),
        "step_ms_mean": 1000 * sum(tstep[20:]) / max(1, len(tstep) - 20), "step_ms_max": 1000 * max(tstep[20:]),
        "steps": args.steps, "physics_dt": dt, "device": str(sim.device)})
    res["ok"] = bool(res["finite"] and max(errs[len(errs) // 2:]) < 0.15)
    # 🔴 수치 JSON 을 **스냅샷보다 먼저** 쓴다 (09-03: 스냅샷 단계 폭주로 결과를 잃었다).
    json.dump(res, open(os.path.join(OUT, f"smoke_{args.num_envs}.json"), "w"), indent=2)
    # 스냅샷 (RTX 카메라가 켜진 경우) — BasicWriter 는 Isaac Lab 앱에서 매 프레임 기록을 멈추지 않아 32 GB 를 쌓았다.
    #    annotator 로 픽셀 1장만 직접 받아 저장한다.
    if getattr(args, "enable_cameras", False):
        try:
            import omni.replicator.core as rep
            from PIL import Image
            n = args.num_envs; side = int(n ** 0.5) + 1
            c = (0.45 * side, 0.45 * side, 0.15); d = 0.9 * side
            cam = rep.create.camera(position=(c[0] + d * 0.7, c[1] - d * 0.9, d * 0.8), look_at=c)
            rp = rep.create.render_product(cam, (1600, 1000))
            ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
            img = ann.get_data()
            Image.fromarray(img[..., :3]).save(os.path.join(OUT, f"grid_{n}envs.png"))
            res["snapshot"] = f"grid_{n}envs.png"
            ann.detach(); rep.orchestrator.stop()
            json.dump(res, open(os.path.join(OUT, f"smoke_{args.num_envs}.json"), "w"), indent=2)
        except Exception as e:
            res["snapshot_error"] = repr(e)
            json.dump(res, open(os.path.join(OUT, f"smoke_{args.num_envs}.json"), "w"), indent=2)
    print("[smoke]", json.dumps({k: v for k, v in res.items() if k not in ("joint_names", "body_names")}))


try:
    main()
except Exception as e:                       # D447: 삼켜지기 전에 파일로 남긴다
    import traceback; res["error"] = traceback.format_exc()
    json.dump(res, open(os.path.join(OUT, f"smoke_{args.num_envs}.json"), "w"), indent=2); print("[smoke] ERROR", e)
# 🔴 close() 가 결과 기록 후 무한 대기한 사례(09-03, 1,100 초+) → 결과는 이미 파일에 있으니 20 초 뒤 강제 종료.
import threading
threading.Thread(target=lambda: (time.sleep(20), os._exit(0)), daemon=True).start()
simulation_app.close()
