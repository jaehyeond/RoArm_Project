"""Phase ST-C v3 진단 #1 — Sim demo direct replay test on real arm.

목적
----
Sim ep 50 (첫 stacking demo)의 frame 0~50 action sequence를 모델 없이
real arm Follower(/dev/ttyUSB1)에 직접 명령 → TCP z 3가지 측정 + 비교.

가설 검증
---------
- H1 (sim-real Z gap): 5/05 deploy에서 모델 출력으로 TCP z = -98mm 측정
  (sim 학습 의도 +33mm 대비 131mm 차이). 원인 후보:
    (a) URDF FK ≠ SDK firmware FK (좌표계 origin 차이)
    (b) Sim과 real의 joint convention 차이
    (c) 모델 output 자체가 sim 학습 action과 다름

이 진단으로 (a)/(b)와 (c)를 분리:
- 같은 joint angles 명령했을 때 real arm의 URDF-FK z와 SDK-FK z가
  비슷하면 → kinematics 문제 아님 → (c) 모델 의심.
- URDF-FK z ≈ +33 인데 SDK-FK z ≈ -98 → 두 FK system origin 130mm gap.
- URDF-FK z ≈ -98 → real arm이 sim 의도와 다른 자세로 도달.

Safety
------
- frame 0~50까지만 (grasp 직전까지). gripper close 명령 발생 시 dataset
  상에서 frame을 미리 점검. --gripper-mode 로 override 가능.
- speed=200, acc=80 (deploy_smolvla.py 기본보다 보수)
- dwell=0.15s/step → ~6.7Hz 명령. 50 frames ≈ 7.5s.
- 모든 action은 사전 joint-limit check 통과해야 진행.
- finally에서 HOME 복귀 + torque off + disconnect.

Output
------
- logs/sim_replay_ep<E>_f<F>_<TS>.csv (frame별 cmd/meas/3가지 FK z)
- logs/sim_replay_ep<E>_f<F>_<TS>.png (TCP z 시계열 3 line)
- 콘솔에 H1 verdict 요약

Run
---
1) 사전 dry-run (로봇 미연결, action/sim FK 분포만 출력):
   python replay_sim_demo_real.py --dry-run

2) Real replay (R3 다시 세워둔 후, USB1 Follower 연결 확인):
   python replay_sim_demo_real.py --gripper-mode open

   (--gripper-mode open: gripper를 60° open 고정 → grasp 시도 안 함, R4 안전)
   (--gripper-mode follow: sim action[5] 그대로 따라감 → grasp close도 발생,
    하지만 frame 50까지면 sponge에 닿기 직전이라 통상 안전)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "sim_scripts"))

# ---- Silence SDK print spam (CLAUDE.md 패턴) ----
logging.getLogger().setLevel(logging.CRITICAL)
from roarm_sdk.common import DataProcessor, JsonCmd, handle_m3_feedback  # noqa: E402


def _silent_process(self, data, genre):
    if not data:
        return None
    res, valid_data = [], []
    if genre == JsonCmd.FEEDBACK_GET:
        valid_data = [data['x'], data['y'], data['z']]
        if self.type == "roarm_m3":
            valid_data = handle_m3_feedback(valid_data, data)
    else:
        valid_data = data
    res.append(valid_data)
    return res


DataProcessor._process_received = _silent_process

from roarm_sdk.roarm import roarm  # noqa: E402
from roarm_kinematics import fk_tcp  # noqa: E402  (returns xyz in METERS)


# ---- Hardware joint limits (CLAUDE.md table — NOT v6 clipped limits) ----
JOINT_LIMITS_DEG = [
    (-190.0, 190.0),  # base
    (-110.0, 110.0),  # shoulder
    (-70.0, 190.0),   # elbow (asymmetric!)
    (-110.0, 110.0),  # wrist_p
    (-190.0, 190.0),  # wrist_r
    (-10.0, 100.0),   # gripper
]

DEFAULT_DATASET = (
    REPO / "lerobot_dataset_v6_stacking_v3" / "data" / "chunk-000" / "file-000.parquet"
)
HOME_DEG = [0.0, 0.0, 90.0, 0.0, 0.0, 5.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    p.add_argument("--episode", type=int, default=50,
                   help="ep 50 = first stacking demo (task_index=1)")
    p.add_argument("--max-frame", type=int, default=50,
                   help="Replay up to and including this frame index (inclusive)")
    p.add_argument("--speed", type=int, default=200,
                   help="SDK joints_angle_ctrl speed (deploy default ~250)")
    p.add_argument("--acc", type=int, default=80)
    p.add_argument("--dwell", type=float, default=0.15,
                   help="Sleep after each command (s) — 0.15 ≈ 6.7Hz")
    p.add_argument("--port", type=str, default="/dev/ttyUSB1",
                   help="Follower port (HARD RULE #16)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--gripper-mode", type=str, default="open",
                   choices=["follow", "open", "closed"],
                   help="follow=sim action[5] / open=force 60° (safest) / closed=force 5°")
    p.add_argument("--dry-run", action="store_true",
                   help="No robot connection; print sim trajectory FK distribution only")
    return p.parse_args()


def safe_joints_get(arm, max_retries=5):
    for _ in range(max_retries):
        try:
            r = arm.joints_angle_get()
            if r is not None and len(r) == 6:
                return list(r)
        except (KeyError, TypeError, IndexError, AttributeError):
            pass
        time.sleep(0.05)
    return None


def safe_pose_get(arm, max_retries=3):
    """SDK firmware FK pose; mm units expected (deploy_smolvla.py:817 compares against -130)."""
    for _ in range(max_retries):
        try:
            p = arm.pose_get()
            if p and len(p) >= 3:
                return list(p[:3])
        except Exception:
            time.sleep(0.05)
    return None


def in_limits(j):
    return all(lo <= a <= hi for a, (lo, hi) in zip(j, JOINT_LIMITS_DEG))


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO / "logs"
    log_dir.mkdir(exist_ok=True)
    tag = f"ep{args.episode}_f{args.max_frame}"
    csv_path = log_dir / f"sim_replay_{tag}_{ts}.csv"
    png_path = log_dir / f"sim_replay_{tag}_{ts}.png"

    # ============================================================
    # [1/5] Load + validate sim trajectory.
    # ============================================================
    print(f"[1/5] Loading {args.dataset}")
    if not Path(args.dataset).exists():
        sys.exit(f"ERROR: dataset not found: {args.dataset}")
    df = pd.read_parquet(args.dataset)
    ep = df[df["episode_index"] == args.episode].reset_index(drop=True)
    if len(ep) == 0:
        sys.exit(f"ERROR: episode {args.episode} not found in dataset")
    task_idx = int(ep["task_index"].iloc[0])
    print(f"  ep {args.episode}: len={len(ep)}, task_index={task_idx}")
    if args.max_frame >= len(ep):
        print(f"  WARN: max_frame {args.max_frame} >= ep len {len(ep)}, clamping to {len(ep)-1}")
        args.max_frame = len(ep) - 1

    actions = np.stack(ep["action"].values[: args.max_frame + 1]).astype(np.float64)
    print(f"  Replay window: frames 0..{args.max_frame} ({len(actions)} actions)")

    # Frame samples
    for f in (0, 5, 10, 20, 30, 40, args.max_frame):
        if f <= args.max_frame:
            print(f"    action[{f:3d}] = {[f'{v:+.2f}' for v in actions[f]]}")

    # Gripper close detection
    gripper_close_frames = [i for i, a in enumerate(actions) if a[5] < 30.0]
    print(f"  Gripper close frames (action[5]<30°) in window [{args.gripper_mode=}]: "
          f"{gripper_close_frames if len(gripper_close_frames) <= 20 else f'{len(gripper_close_frames)} frames'}")

    # Hard joint-limit check (ABORT before any robot motion)
    bad = [i for i, a in enumerate(actions) if not in_limits(a)]
    if bad:
        sys.exit(f"ERROR: action OUT OF JOINT LIMITS at frames {bad[:10]}{'...' if len(bad)>10 else ''}")
    print(f"  Joint limit check: PASS for {len(actions)} frames")

    # Sim "expected" FK z (URDF, m → mm)
    sim_xyz_m = np.array([fk_tcp(a[:6]) for a in actions])  # (N, 3) meters
    sim_z_mm = sim_xyz_m[:, 2] * 1000.0
    print(f"  Sim URDF-FK z (mm): min={sim_z_mm.min():+.1f}  max={sim_z_mm.max():+.1f}  "
          f"mean={sim_z_mm.mean():+.1f}")
    print(f"    z[0]={sim_z_mm[0]:+.1f}  z[10]={sim_z_mm[10]:+.1f}  z[20]={sim_z_mm[20]:+.1f}  "
          f"z[max-pos]={sim_z_mm.argmax()}@{sim_z_mm.max():+.1f}  z[end]={sim_z_mm[-1]:+.1f}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping robot. Above sim FK z is the target the real arm should track.")
        return

    # ============================================================
    # [2/5] Connect arm + go to HOME.
    # ============================================================
    print(f"\n[2/5] Connecting Follower at {args.port}")
    print(f"  ⚠️  R3 sponge가 다시 직립되어 있는지 확인! (5/05 retryD에서 R3 knock 확인됨)")
    print(f"  ⚠️  USB1 = Follower (HARD RULE #16). 잘못된 포트면 즉시 Ctrl+C.")
    arm = roarm(roarm_type="roarm_m3", port=args.port, baudrate=args.baud)
    rows = []
    try:
        arm.torque_set(cmd=1)
        time.sleep(0.3)
        print(f"  Moving to HOME = action[0] = {actions[0].tolist()}")
        arm.joints_angle_ctrl(angles=actions[0].tolist(), speed=120, acc=60)
        time.sleep(2.5)
        cur = safe_joints_get(arm)
        if cur is None:
            sys.exit("ERROR: joints_angle_get returned None at HOME")
        urdf_xyz_m = fk_tcp(np.asarray(cur[:6]))
        sdk_xyz_mm = safe_pose_get(arm)
        print(f"  HOME measured joints: {[f'{a:+.2f}' for a in cur]}")
        print(f"  HOME URDF-FK (mm): x={urdf_xyz_m[0]*1000:+.1f}  y={urdf_xyz_m[1]*1000:+.1f}  "
              f"z={urdf_xyz_m[2]*1000:+.1f}")
        if sdk_xyz_mm is not None:
            print(f"  HOME SDK-pose  (mm): x={sdk_xyz_mm[0]:+.1f}  y={sdk_xyz_mm[1]:+.1f}  "
                  f"z={sdk_xyz_mm[2]:+.1f}")
            print(f"  >>> HOME z gap (SDK − URDF) = {sdk_xyz_mm[2] - urdf_xyz_m[2]*1000:+.1f}mm")
        else:
            print("  WARN: arm.pose_get() returned None at HOME")

        # ============================================================
        # [3/5] Replay loop.
        # ============================================================
        print(f"\n[3/5] Replaying frames 0..{args.max_frame}  "
              f"speed={args.speed} acc={args.acc} dwell={args.dwell}s "
              f"gripper_mode={args.gripper_mode}")
        t0 = time.time()
        for i in range(args.max_frame + 1):
            cmd = actions[i].tolist()
            if args.gripper_mode == "open":
                cmd[5] = 60.0
            elif args.gripper_mode == "closed":
                cmd[5] = 5.0
            # in_limits already checked at load time; re-check after override
            if not in_limits(cmd):
                print(f"  ABORT at frame {i}: cmd out of limits {cmd}")
                break
            arm.joints_angle_ctrl(angles=cmd, speed=args.speed, acc=args.acc)
            time.sleep(args.dwell)

            measured = safe_joints_get(arm)
            sdk_pose = safe_pose_get(arm)
            if measured is None:
                print(f"  WARN: joints_angle_get fail at frame {i}, skipping log")
                continue
            urdf_xyz = fk_tcp(np.asarray(measured[:6]))
            real_urdf_z_mm = urdf_xyz[2] * 1000.0
            real_sdk_z_mm = sdk_pose[2] if sdk_pose else float("nan")
            sim_target_z_mm = sim_z_mm[i]

            row = {
                "frame": i,
                "t_s": round(time.time() - t0, 3),
                **{f"cmd_j{k}": round(cmd[k], 3) for k in range(6)},
                **{f"meas_j{k}": round(measured[k], 3) for k in range(6)},
                "sim_target_urdf_x_mm": round(sim_xyz_m[i, 0] * 1000, 2),
                "sim_target_urdf_y_mm": round(sim_xyz_m[i, 1] * 1000, 2),
                "sim_target_urdf_z_mm": round(sim_target_z_mm, 2),
                "real_meas_urdf_x_mm": round(urdf_xyz[0] * 1000, 2),
                "real_meas_urdf_y_mm": round(urdf_xyz[1] * 1000, 2),
                "real_meas_urdf_z_mm": round(real_urdf_z_mm, 2),
                "real_sdk_x_mm": round(sdk_pose[0], 2) if sdk_pose else float("nan"),
                "real_sdk_y_mm": round(sdk_pose[1], 2) if sdk_pose else float("nan"),
                "real_sdk_z_mm": round(real_sdk_z_mm, 2),
                "diff_realUrdf_minus_simTarget_mm": round(real_urdf_z_mm - sim_target_z_mm, 2),
                "diff_sdk_minus_realUrdf_mm": (
                    round(real_sdk_z_mm - real_urdf_z_mm, 2) if sdk_pose else float("nan")
                ),
                "diff_sdk_minus_simTarget_mm": (
                    round(real_sdk_z_mm - sim_target_z_mm, 2) if sdk_pose else float("nan")
                ),
            }
            rows.append(row)
            if i % 5 == 0 or i == args.max_frame:
                gap_sdk = (
                    f"sdk={real_sdk_z_mm:+7.1f}" if sdk_pose else "sdk=NaN  "
                )
                print(f"  f{i:3d}  sim={sim_target_z_mm:+7.1f}  realURDF={real_urdf_z_mm:+7.1f}  "
                      f"{gap_sdk}  Δ(realURDF−sim)={real_urdf_z_mm-sim_target_z_mm:+6.1f}mm")

        # ============================================================
        # [4/5] Return to HOME safely.
        # ============================================================
        print(f"\n[4/5] Returning to HOME {HOME_DEG}")
        arm.joints_angle_ctrl(angles=HOME_DEG, speed=120, acc=60)
        time.sleep(2.5)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] User Ctrl+C — returning to HOME and disconnecting safely")
        try:
            arm.joints_angle_ctrl(angles=HOME_DEG, speed=120, acc=60)
            time.sleep(2.0)
        except Exception:
            pass
    finally:
        try:
            arm.torque_set(cmd=0)
        except Exception:
            pass
        arm.disconnect()

    if not rows:
        print("ERROR: no rows logged.")
        return

    # ============================================================
    # [5/5] CSV + plot + verdict.
    # ============================================================
    print(f"\n[5/5] Saving CSV → {csv_path}")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    arr_sim = np.array([r["sim_target_urdf_z_mm"] for r in rows])
    arr_real_urdf = np.array([r["real_meas_urdf_z_mm"] for r in rows])
    arr_real_sdk = np.array([r["real_sdk_z_mm"] for r in rows], dtype=np.float64)

    print("\n========== SUMMARY (n=%d) ==========" % len(rows))
    print(f"sim_target_urdf_z (mm):    min={arr_sim.min():+.1f}  max={arr_sim.max():+.1f}  mean={arr_sim.mean():+.1f}")
    print(f"real_meas_urdf_z  (mm):    min={arr_real_urdf.min():+.1f}  max={arr_real_urdf.max():+.1f}  mean={arr_real_urdf.mean():+.1f}")
    if not np.all(np.isnan(arr_real_sdk)):
        valid = ~np.isnan(arr_real_sdk)
        print(f"real_sdk_z        (mm):    min={arr_real_sdk[valid].min():+.1f}  max={arr_real_sdk[valid].max():+.1f}  mean={arr_real_sdk[valid].mean():+.1f}")

    diff_real_vs_sim = arr_real_urdf - arr_sim
    print(f"\nΔ (realURDF − simTarget) (mm):  mean={diff_real_vs_sim.mean():+.1f}  "
          f"abs_max={np.abs(diff_real_vs_sim).max():.1f}")
    if not np.all(np.isnan(arr_real_sdk)):
        diff_sdk_vs_real = arr_real_sdk - arr_real_urdf
        valid = ~np.isnan(diff_sdk_vs_real)
        print(f"Δ (SDK − realURDF)       (mm):  mean={diff_sdk_vs_real[valid].mean():+.1f}  "
              f"abs_max={np.abs(diff_sdk_vs_real[valid]).max():.1f}")
        print(f"  (이 값이 ~130mm 근처면 SDK와 URDF FK origin 차이 = H1 메인 후보)")

    # Verdict
    abs_real_vs_sim = abs(diff_real_vs_sim.mean())
    print("\n========== VERDICT ==========")
    if abs_real_vs_sim < 10:
        print(">>> realURDF ≈ simTarget — kinematics 일치. H1(URDF-level) 기각.")
        print("    다음 진단: 모델 output 자체가 sim 학습 action과 다른지 확인.")
    elif abs_real_vs_sim < 30:
        print(">>> realURDF ≈ simTarget within tracking tolerance (~%.1fmm). "
              "Real arm이 sim 의도 따라감." % abs_real_vs_sim)
    else:
        print(">>> realURDF significantly differs from simTarget by mean %.1fmm" % diff_real_vs_sim.mean())
        print("    Real arm tracking 실패 또는 kinematic convention 차이.")

    if not np.all(np.isnan(arr_real_sdk)):
        sdk_minus_urdf = (arr_real_sdk - arr_real_urdf)[valid].mean()
        if abs(sdk_minus_urdf) > 50:
            print(f">>> SDK FK와 URDF FK가 평균 {sdk_minus_urdf:+.1f}mm 차이 — "
                  f"H1(SDK-URDF FK origin gap) STRONG 후보")
            print(f"    deploy_smolvla.py가 SDK pose_get 사용해서 -98mm 측정 → ")
            print(f"    같은 자세를 URDF FK로 보면 z={arr_real_urdf.mean():+.1f}mm일 수 있음.")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        frames = [r["frame"] for r in rows]
        plt.figure(figsize=(11, 5.5))
        plt.plot(frames, arr_sim, label="sim_target (URDF FK on action)", linewidth=2, color="#1f77b4")
        plt.plot(frames, arr_real_urdf, label="real_meas (URDF FK on encoder)", linewidth=2, color="#2ca02c")
        if not np.all(np.isnan(arr_real_sdk)):
            plt.plot(frames, arr_real_sdk, label="real_sdk (SDK firmware FK)", linewidth=2,
                     color="#d62728", linestyle="--")
        plt.axhline(33.0, color="gray", linestyle=":", label="sim grasp target +33mm")
        plt.axhline(-12.0, color="brown", linestyle=":", label="table top -12mm")
        plt.axhline(-98.0, color="purple", linestyle=":", label="5/05 deploy z dive")
        plt.xlabel("Frame index")
        plt.ylabel("TCP z (mm world)")
        plt.title(f"Sim ep{args.episode} replay (frames 0..{args.max_frame}): TCP z comparison")
        plt.legend(loc="best", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        print(f"\nPlot saved → {png_path}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
