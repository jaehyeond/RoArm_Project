#!/usr/bin/env python3
"""Prepare a diagnostic RoArm URDF variant for 2cm cube grasp tests.

This script does not launch Isaac and does not edit the repo's default URDF/USD.
It copies the local RoArm URDF asset directory to a requested output directory
and swaps only the gripper_link collision mesh reference. The resulting URDF is
intended for a separate, explicitly approved IsaacLab USD conversion and the
USD-selectable grasp/lift wrapper.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_URDF_DIR = REPO / "local_assets/roarm_m3/urdf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_urdf_dir", type=Path, default=DEFAULT_SOURCE_URDF_DIR)
    ap.add_argument("--output_urdf_dir", type=Path, required=True)
    ap.add_argument(
        "--collision_mesh",
        default="meshes/gripper_link.stl",
        help="Replacement mesh path written into the copied URDF collision element.",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = args.source_urdf_dir.expanduser().resolve()
    dst = args.output_urdf_dir.expanduser().resolve()
    src_urdf = src / "roarm_m3.urdf"
    dst_urdf = dst / "roarm_m3.urdf"
    if not src_urdf.exists():
        raise FileNotFoundError(f"source URDF not found: {src_urdf}")
    if dst.exists():
        if not args.force:
            raise FileExistsError(f"output exists; pass --force to replace: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    text = dst_urdf.read_text(encoding="utf-8")
    old = 'filename="meshes/gripper_link_collision_g2a.stl"'
    new = f'filename="{args.collision_mesh}"'
    if old not in text:
        raise RuntimeError(f"expected collision mesh reference not found in {dst_urdf}")
    text = text.replace(old, new, 1)
    dst_urdf.write_text(text, encoding="utf-8")

    print("[cube2cm_prepare_urdf] static_only=YES isaac_run=NO env_default_edits=NO chain_defaults_edits=NO")
    print(f"[cube2cm_prepare_urdf] source_urdf={src_urdf}")
    print(f"[cube2cm_prepare_urdf] output_urdf={dst_urdf}")
    print("[cube2cm_prepare_urdf] replaced gripper collision mesh:")
    print(f"[cube2cm_prepare_urdf]   old={old}")
    print(f"[cube2cm_prepare_urdf]   new={new}")
    print("[cube2cm_prepare_urdf] NEXT_CONVERSION_REQUIRED=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
