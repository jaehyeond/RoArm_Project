#!/usr/bin/env python3
"""URDF → USD 변환 (Isaac Sim 5.1 / IsaacLab UrdfConverter), **convex_decomposition** 콜라이더.

🔴 D446 재발 방지: IsaacLab 기본 `convert_urdf.py` 는 `collider_type="convex_hull"` 이라
   **오목한 스쿱 보울 공동이 단일 볼록껍질로 채워진다**(펠릿이 안 들어감). D446 이 바로 이 함정
   (벤더 USD 1-hull 근사가 실기하를 가려 13/13↔0/13). 셸은 볼록 조각들로 만들어졌으므로
   `convex_decomposition` 으로 임포트해 공동을 살린다.

⚠️ D447: `SimulationApp.close()` 는 예외를 삼키고 exit 0 → **호출부가 산출 USD 실물을 따로 검증**할 것.
사용:  OMNI_KIT_ACCEPT_EULA=YES python sim_urdf_to_usd.py <in.urdf> <out.usd> --headless
"""
import argparse, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="URDF->USD, convex_decomposition collider (scoop 공동 보존)")
parser.add_argument("input", type=str)
parser.add_argument("output", type=str)
parser.add_argument("--merge-joints", action="store_true", default=False)
parser.add_argument("--fix-base", action="store_true", default=False)
parser.add_argument("--collider", default="convex_decomposition",
                    choices=["convex_hull", "convex_decomposition"],
                    help="convex_hull = collision 이 이미 볼록 조각별일 때(조각=자기 hull=정확). "
                         "convex_decomposition = 단일 _ALL 오목 메시일 때(VHACD 로 공동 근사).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import contextlib
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main():
    in_path = os.path.abspath(args_cli.input)
    out_path = os.path.abspath(args_cli.output)
    assert os.path.isfile(in_path), f"입력 URDF 없음: {in_path}"

    cfg = UrdfConverterCfg(
        asset_path=in_path,
        usd_dir=os.path.dirname(out_path),
        usd_file_name=os.path.basename(out_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        collider_type=args_cli.collider,           # 🔴 스쿱 공동 보존 (D446)
        # 🔴 D477 (09-03): 이 플래그는 이름과 **반대**로 동작한다 — IsaacLab urdf_converter.py:130 이
        #    값을 그대로 `import_config.set_parse_mimic()` 에 넘기고, 임포터 테스트(test_urdf.py:475~491)가
        #    parse_mimic=True ⇒ PhysxMimicJointAPI 생성임을 증명한다. True 로 두면 셸_R 이 드라이브 없는
        #    PhysX mimic(gearing −1, 25 Hz, ζ 0.005, 한계 −8.9°~53.4°)이 되어 병렬 스텝에서 상한(0.931 rad)에 걸렸다.
        #    독립 구동 관절(드라이브 + URDF 한계)을 원하면 **False**.
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
            target_type="position",
        ),
    )
    conv = UrdfConverter(cfg)
    print(f"[convert] usd_path = {conv.usd_path}")
    print(f"[convert] collider_type = {args_cli.collider}")
    return conv.usd_path


usd_path = None
with contextlib.suppress(Exception):
    usd_path = main()
simulation_app.close()
# D447: close() 뒤에는 exit code 를 믿지 말고 실물로 판단. 여기선 경로만 찍고 검증은 호출부.
print(f"[done] {usd_path}")
