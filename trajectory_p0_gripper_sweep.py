"""P0.2 Gauge-method gripper sweep — Track B cube calibration (2026-05-26).

Gauge method (user-confirmed): arm held at HOME, a human places the 30mm cube
between the jaws by hand, and ONLY the gripper command is swept. This yields an
OBJECT-AGNOSTIC cmd->settled-state(->jaw mm) curve: future objects are then
mapped by width alone, no re-sweep. Static hold only — dynamic grasp is
validated later in P0.4 (transport) + P0.7 (L-F gate).

Convention (gripper_calibrate_v4.py:13): cmd 0 = jaw CLOSED, cmd ~89 = jaw OPEN.
A 30mm cube blocks the jaw from reaching a low commanded width, so a held cube
shows gap = settled_state - cmd > 0. The human marks hold Y/N at each step
(does the cube stay when gently nudged?).

Guards used (P0.2 set): G1 G2 G3 G4 G7 G8 G10  (see safety_p0_guards.py).
G5/G6 (FK z/dist) are not swept here — arm posture is fixed at HOME.

Usage:
  python trajectory_p0_gripper_sweep.py --dry-run     # logic check, no robot
  python trajectory_p0_gripper_sweep.py               # real, Follower USB1
"""
import argparse
import time

import safety_p0_guards as G

# Sweep list (tech_cube_grasp_anchors.md P0.2). Open->closed so the cube is
# already seated when jaws begin to press. Auto-stops at the stall plateau, so
# the deep-close steps (low cmd) are usually never reached.
SWEEP = [40, 35, 30, 25, 20, 15, 10, 5, 0]
HOLD_S = 6.0          # window for human to nudge-test the cube
START_GRIPPER = 40    # jaws open enough to seat a 30mm cube before sweep
# Servo protection: once the jaw is blocked by the rigid cube, commanding lower
# only piles up stall torque with no extra grip (verified 2026-05-26: cmd30/25
# both -> state 37.88). Stop the sweep when that plateau or a large gap appears.
PLATEAU_TOL = 0.6     # deg: 2 consecutive blocked settled states this close = stall
MAX_GAP = 16.0        # deg: hard stop if pent-up close command exceeds this


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="run control flow with a mock arm (no serial)")
    ap.add_argument("--hold-s", type=float, default=HOLD_S)
    args = ap.parse_args()

    arm = G.connect_follower(dry_run=args.dry_run)
    cleanup = G.install_safe_exit(arm)            # G7

    # Park arm at HOME, jaws open to seat the cube. Hold base/shoulder/elbow.
    home = list(G.INIT_POS)
    home[5] = START_GRIPPER
    print(f">>> Parking at HOME {home} (jaws open). Place the 30mm cube between "
          f"the jaws by hand, then press Enter.")
    G.move_joints(arm, home, settle_idx=5, settle_target=START_GRIPPER)  # G8/G9/G10
    if not args.dry_run:
        input()

    base, shoulder, elbow, wp, wr, _ = G.safe_get(arm)

    print("=" * 76)
    print("P0.2 GAUGE SWEEP (cmd -> settled state -> gap). Mark hold Y/N each step.")
    print("  Nudge cube gently during HOLD window; if it stays = HOLD Y.")
    print("=" * 76)
    print(f"  {'cmd':>4} {'settled':>9} {'gap':>7} {'after':>9} {'drift':>7} {'jam?':>16}")

    records = []
    prev_settled = None
    stall_state = None
    for cmd in SWEEP:
        g = G.clamp_gripper(cmd)                  # G2
        angles = [base, shoulder, elbow, wp, wr, g]
        _, _, _, status = G.move_joints(          # G8/G9/G10
            arm, angles, settle_idx=5, settle_target=g)  # G3
        settled = G.safe_get(arm)[5]
        gap = settled - g                         # >0 => jaw blocked (cube held)
        print(f"  cmd {g:>3} settling={status} state={settled:6.2f} gap={gap:+.2f}"
              f"  --> NUDGE-TEST cube now ({args.hold_s:.0f}s), record HOLD Y/N")
        time.sleep(args.hold_s)
        after = G.safe_get(arm)[5]
        ok, drift = G.drift_check(after, settled)  # G4 (here drift = cube shift)
        jam = "ok" if ok else "cube shifted/jam"
        print(f"       after={after:6.2f} drift={drift:+.2f} [{jam}]")
        records.append((g, settled, gap, after, drift, jam))

        # servo protection: stop once the rigid cube blocks the jaw
        blocked = status == "TIMEOUT"
        plateau = prev_settled is not None and abs(settled - prev_settled) < PLATEAU_TOL
        if blocked and plateau:
            stall_state = settled
            print(f"  [auto-stop] stall plateau ~{settled:.2f} deg "
                  f"(2 blocked cmds within {PLATEAU_TOL} deg). Lower cmd only adds torque.")
            break
        if gap > MAX_GAP:
            stall_state = settled
            print(f"  [auto-stop] gap {gap:+.2f} > {MAX_GAP} (servo protection).")
            break
        prev_settled = settled

    # G7-aware release: open jaws BEFORE cleanup (cleanup sets gripper->5 which
    # would crush a cube still in the jaws).
    print("\nOpening jaws to release the cube. REMOVE the cube, then press Enter.")
    G.move_joints(arm, [base, shoulder, elbow, wp, wr, START_GRIPPER],
                  settle_idx=5, settle_target=START_GRIPPER)
    if not args.dry_run:
        input()

    print("=" * 76)
    print("OBJECT-AGNOSTIC CURVE (settled state per cmd; gap>0 = jaw blocked by cube):")
    print(f"  {'cmd':>4} {'settled':>9} {'gap':>7} {'after':>9} {'drift':>7}  jam")
    for cmd, s, gap, after, drift, jam in records:
        print(f"  {cmd:>4} {s:>9.2f} {gap:>+7.2f} {after:>9.2f} {drift:>+7.2f}  {jam}")
    if stall_state is not None:
        print(f"P0.2 STALL ANCHOR: 30mm cube blocks jaw at state ~{stall_state:.2f} deg.")
        print(f"  -> grip cmd target ~ {stall_state-10:.0f} (firm, bounded gap). "
              f"Do NOT command 0-5 (servo stall, no extra grip).")
    print("=" * 76)
    print("Offline: caliper jaw inner width at 2-3 settled states -> jaw_mm(state) fit.")

    cleanup()  # G7: safe home + torque OFF + disconnect (idempotent)


if __name__ == "__main__":
    main()
