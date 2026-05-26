"""P0.1 hardware sanity — Track B cube calibration (2026-05-26).

Checks (tech_cube_grasp_anchors.md P0.1):
  1. Follower torque ON -> INIT_POS [0,0,90,0,0,5], max_diff <= 3 deg.
  2. Kinect 1-frame capture (720P) + save PNG for cube-visibility inspection.
  3. pose_ctrl IK smoke test (OPT-IN, default OFF) -> RELIABLE / UNRELIABLE.
     Default plan is joint-direct; pose_ctrl only assists P0.4 if this passes.

  (Camera remount sanity is a separate interactive GUI tool — run it by hand:
     python hw_camera_remount_verify.py --mode check
   ORB<10px / SSIM>0.95 against hw_camera_reference.png. Not invoked here to
   avoid a blocking cv2 window.)

Follower = /dev/ttyUSB1 only (HARD RULE #13). Leader USB0 forbidden.

Usage:
  python hw_p0_sanity.py --dry-run                 # logic check, no robot
  python hw_p0_sanity.py                           # checks 1 + 2
  python hw_p0_sanity.py --pose-ctrl-smoke         # also check 3 (guarded)
"""
import argparse
import os
import time

import safety_p0_guards as G

MAX_DIFF_DEG = 3.0          # A3-strengthened (was 5)
SMOKE_MAX_DIFF = 5.0        # pose_ctrl round-trip joint tolerance
KINECT_DIR = "logs/hw_sanity_p0"


def check_init_pos(arm, dry_run):
    print(">>> Check 1: torque ON -> INIT_POS, max_diff <= 3 deg")
    if not dry_run:
        arm.torque_set(cmd=1)
        time.sleep(0.3)
    G.move_joints(arm, list(G.INIT_POS), settle_idx=1, settle_target=0)
    time.sleep(1.0)
    state = G.safe_get(arm)
    diffs = [abs(s - t) for s, t in zip(state, G.INIT_POS)]
    max_diff = max(diffs)
    print(f"    state={[round(s,2) for s in state]}")
    print(f"    per-joint diff={[round(d,2) for d in diffs]}  max_diff={max_diff:.2f}")
    ok = max_diff <= MAX_DIFF_DEG
    print(f"    Check 1: {'PASS' if ok else 'FAIL'} (max_diff {max_diff:.2f} "
          f"{'<=' if ok else '>'} {MAX_DIFF_DEG})")
    return ok


def check_kinect(dry_run):
    print(">>> Check 2: Kinect 1-frame capture + cube-visibility inspection")
    if dry_run:
        print("    [dry-run] skip Kinect capture")
        return True
    try:
        import pyk4a
        from pyk4a import Config, PyK4A
        k4a = PyK4A(Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        ))
        k4a.start()
        cap = k4a.get_capture()
        rgb = cap.color[:, :, :3]
        k4a.stop()
        os.makedirs(KINECT_DIR, exist_ok=True)
        path = os.path.join(KINECT_DIR, "p0_kinect_frame.png")
        import cv2
        cv2.imwrite(path, rgb)
        h, w = rgb.shape[:2]
        # 30mm cube pixel estimate at 224 resize: cube_px_720 * 224/720
        print(f"    saved {path}  ({w}x{h})")
        print(f"    INSPECT: place a cube in the workspace; confirm it spans enough")
        print(f"    pixels after 224 resize (>~10px advised). 720p->224 factor "
              f"{224.0/h:.3f}.")
        return True
    except Exception as e:
        print(f"    Check 2: FAIL — Kinect error: {e}")
        return False


def check_pose_ctrl_smoke(arm, dry_run):
    print(">>> Check 3: pose_ctrl IK smoke test (OPT-IN, guarded)")
    print("    WARNING: get/ctrl index semantics may differ; commanding pose_get()")
    print("    output back may move the arm. Slow speed + FK guard + abort armed.")
    ok_fk, p0, reason = G.fk_guard(arm)
    if not ok_fk:
        print(f"    pre-FK guard FAIL ({reason}) -> safe home, skip smoke")
        G.go_safe_home(arm)
        return False
    print(f"    pose_get -> {[round(v,1) for v in p0]}")
    j_before = G.safe_get(arm)
    try:
        if dry_run:
            arm.pose_ctrl(list(p0))
        else:
            arm.pose_ctrl(list(p0))
            time.sleep(G.INTER_CMD_DELAY)
            time.sleep(1.0)
    except Exception as e:
        print(f"    pose_ctrl raised: {e}")
        print("    -> pose_ctrl UNRELIABLE. Use joint-direct only (P0 default).")
        G.go_safe_home(arm)
        return False
    ok_fk2, _, reason2 = G.fk_guard(arm)
    if not ok_fk2:
        print(f"    post-FK guard FAIL ({reason2}) -> safe home")
        G.go_safe_home(arm)
        return False
    j_after = G.safe_get(arm)
    max_diff = max(abs(a - b) for a, b in zip(j_after, j_before))
    reliable = max_diff <= SMOKE_MAX_DIFF
    print(f"    joints before={[round(j,1) for j in j_before]}")
    print(f"    joints after ={[round(j,1) for j in j_after]}  max_diff={max_diff:.2f}")
    print(f"    Check 3: pose_ctrl {'RELIABLE' if reliable else 'UNRELIABLE'} "
          f"(round-trip max_diff {max_diff:.2f} vs {SMOKE_MAX_DIFF})")
    if not reliable:
        print("    -> Use joint-direct only; do not let pose_ctrl assist P0.4.")
    return reliable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pose-ctrl-smoke", action="store_true",
                    help="also run the guarded pose_ctrl IK smoke test (default OFF)")
    args = ap.parse_args()

    arm = G.connect_follower(dry_run=args.dry_run)
    cleanup = G.install_safe_exit(arm)

    results = {}
    results["1_init_pos"] = check_init_pos(arm, args.dry_run)
    results["2_kinect"] = check_kinect(args.dry_run)
    if args.pose_ctrl_smoke:
        results["3_pose_ctrl"] = check_pose_ctrl_smoke(arm, args.dry_run)

    print("=" * 60)
    print("P0.1 SANITY SUMMARY:")
    for k, v in results.items():
        print(f"  {k:14s}: {'PASS' if v else 'FAIL/UNRELIABLE'}")
    print("  remount sanity (run separately): "
          "python hw_camera_remount_verify.py --mode check")
    print("=" * 60)

    cleanup()  # G7: safe home + torque OFF + disconnect (idempotent)


if __name__ == "__main__":
    main()
