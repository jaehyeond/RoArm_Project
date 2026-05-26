"""P0 safety guards G1-G10 — Track B cube calibration (2026-05-26).

Shared helper for P0.1 sanity (hw_p0_sanity.py) + P0.2 Gauge sweep
(trajectory_p0_gripper_sweep.py). 4-agent B3 cross-validated
(tech_cube_grasp_anchors.md "안전 가드" table).

Follower = /dev/ttyUSB1 ONLY (HARD RULE #13). Leader USB0 is forbidden in P0.

Guards:
  G1 connect dummy cmd + sleep(1)     G6  FK dist>DIST_MAX  -> safe home
  G2 gripper cmd clamp (-10,100)      G7  SIGINT+atexit -> torque off + safe home
  G3 poll_until_settled               G8  inter-command delay >=1.0s
  G4 drift check -> cube jam INVALID  G9  full JOINT_LIMITS clamp
  G5 FK z<Z_FLOOR -> safe home        G10 speed=200 acc=50 fixed (speed=1000 forbidden)

T:106 ESP32 reset: session start ONLY (never mid-measurement -> pose loss).
"""
import time
import atexit
import signal
import logging

logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("BaseController").setLevel(logging.CRITICAL)

# ---- verified constants (source files cited) ----------------------------
PORT_FOLLOWER = "/dev/ttyUSB1"                 # HARD RULE #13
SPEED = 200                                    # G10 (gripper_calibrate_v4)
ACC = 50                                       # G10
INTER_CMD_DELAY = 1.0                          # G8 (ESP32 buffer flush)
Z_FLOOR = -130                                 # G5 (deploy_smolvla.py:83)
DIST_MAX = 420                                 # G6 (deploy_smolvla.py:84)
GRIPPER_LIMIT = (-10, 100)                     # G2 (collect_data_manual.py:64)
# G9 full limits (collect_data_manual.py:64)
JOINT_LIMITS = [(-190, 190), (-110, 110), (-70, 190),
                (-110, 110), (-190, 190), (-10, 100)]
INIT_POS = [0, 0, 90, 0, 0, 5]                 # HOME (HARD RULE #1 + gripper 5)
SAFE_HOME = list(INIT_POS)


# ---- SDK silence monkeypatch (CLAUDE.md correct pattern) ----------------
def _install_silent_process():
    from roarm_sdk.common import DataProcessor, JsonCmd, handle_m3_feedback

    def _silent_process(self, data, genre):
        if not data:
            return None
        res, valid_data = [], []
        if genre == JsonCmd.FEEDBACK_GET:
            valid_data = [data["x"], data["y"], data["z"]]
            if self.type == "roarm_m3":
                valid_data = handle_m3_feedback(valid_data, data)
        else:
            valid_data = data
        res.append(valid_data)
        return res

    DataProcessor._process_received = _silent_process


# ---- G2 / G9 clamps ------------------------------------------------------
def clamp_gripper(cmd):
    """G2: clamp a single gripper command to hardware range."""
    lo, hi = GRIPPER_LIMIT
    return max(lo, min(hi, cmd))


def clamp_joints(angles):
    """G9: clamp all 6 joints to hardware JOINT_LIMITS."""
    return [max(lo, min(hi, a)) for a, (lo, hi) in zip(angles, JOINT_LIMITS)]


# ---- safe state read -----------------------------------------------------
def safe_get(arm, retries=8):
    """5+ retry joints read (SDK intermittent None/KeyError workaround)."""
    for _ in range(retries):
        try:
            v = arm.joints_angle_get()
            if v is not None and len(v) == 6 and all(x is not None for x in v):
                return list(v)
        except Exception:
            pass
        time.sleep(0.15)
    raise RuntimeError("safe_get failed after retries")


def safe_pose(arm, retries=5):
    """FK pose read with retry; returns [x,y,z,t,r,g] or None."""
    for _ in range(retries):
        try:
            p = arm.pose_get()
            if p and len(p) >= 3:
                return list(p)
        except Exception:
            pass
        time.sleep(0.1)
    return None


# ---- G3 poll-until-settled (gripper_calibrate_v4.py:63-76) --------------
def poll_until_settled(arm, target, idx=5, max_s=12.0, poll=1.0,
                       settle_tol=0.3, reach_tol=1.5):
    """Read joint `idx` every `poll`s. Settled when stable AND near target."""
    states = []
    t0 = time.time()
    while time.time() - t0 < max_s:
        time.sleep(poll)
        s = safe_get(arm)[idx]
        states.append(s)
        if len(states) >= 2:
            stable = abs(states[-1] - states[-2]) < settle_tol
            reached = abs(states[-1] - target) < reach_tol
            if stable and reached:
                return states[-1], len(states), time.time() - t0, "SETTLED"
    return (states[-1] if states else float("nan"),
            len(states), time.time() - t0, "TIMEOUT")


# ---- G4 drift / cube-jam check ------------------------------------------
def drift_check(state_after, settled, thr=1.0):
    """G4: |after-settled|>thr => external force (cube jam) => INVALID."""
    drift = state_after - settled
    return abs(drift) <= thr, drift


# ---- G5 / G6 FK guard ----------------------------------------------------
def fk_guard(arm):
    """G5+G6: returns (ok, pose, reason). ok=False => caller must safe-home."""
    p = safe_pose(arm)
    if p is None:
        return False, None, "pose_get_failed"
    x, y, z = p[0], p[1], p[2]
    dist = (x * x + y * y) ** 0.5
    if z < Z_FLOOR:
        return False, p, f"z {z:.1f} < Z_FLOOR {Z_FLOOR}"
    if dist > DIST_MAX:
        return False, p, f"dist {dist:.0f} > DIST_MAX {DIST_MAX}"
    return True, p, "ok"


# ---- safe motion helpers (G8 + G9 + G10 applied) ------------------------
def move_joints(arm, angles, speed=SPEED, acc=ACC, settle_idx=None,
                settle_target=None):
    """G9 clamp + G10 fixed speed + G8 delay. Optional settle wait."""
    if speed > SPEED:
        raise ValueError(f"G10 violation: speed {speed} > {SPEED} (1000 forbidden)")
    cmd = clamp_joints(angles)
    arm.joints_angle_ctrl(angles=cmd, speed=speed, acc=acc)
    time.sleep(INTER_CMD_DELAY)  # G8
    if settle_idx is not None and settle_target is not None:
        return poll_until_settled(arm, settle_target, idx=settle_idx)
    return None


def go_safe_home(arm):
    """Return arm to HOME at safe speed (best-effort, never raises)."""
    try:
        arm.joints_angle_ctrl(angles=list(SAFE_HOME), speed=SPEED, acc=ACC)
        time.sleep(INTER_CMD_DELAY)
    except Exception as e:
        print(f"  [go_safe_home] warning: {e}")


# ---- G1 connect ----------------------------------------------------------
def connect_follower(dry_run=False):
    """G1: connect Follower, dummy read + sleep(1) so first cmd is not dropped.
    Returns an arm object (real or DryRunArm)."""
    if dry_run:
        print("[dry-run] connect_follower -> DryRunArm (no serial)")
        return DryRunArm()
    _install_silent_process()
    from roarm_sdk.roarm import roarm
    arm = roarm(roarm_type="roarm_m3", port=PORT_FOLLOWER, baudrate=115200)
    time.sleep(0.5)
    try:
        arm.joints_angle_get()  # G1 dummy read to flush first-command drop
    except Exception:
        pass
    time.sleep(1.0)
    arm.torque_set(cmd=1)
    time.sleep(0.3)
    return arm


# ---- G7 SIGINT + atexit -> torque off + safe home -----------------------
def install_safe_exit(arm):
    """G7: on Ctrl-C or interpreter exit, safe-home then torque OFF.
    Idempotent — runs once even if the script also cleans up explicitly."""
    state = {"done": False}

    def _cleanup(*_a):
        if state["done"]:
            return
        state["done"] = True
        print("\n[G7] safe-exit: safe home + torque OFF")
        try:
            go_safe_home(arm)
            arm.torque_set(cmd=0)
            arm.disconnect()
        except Exception as e:
            print(f"  [G7] cleanup warning: {e}")

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda *_a: (_cleanup(), exit(130)))
    return _cleanup


# ---- DryRunArm: mock for --dry-run validation (no hardware) --------------
class DryRunArm:
    """Logs commands, returns plausible state so control flow runs offline."""

    def __init__(self):
        self._state = list(INIT_POS)

    def joints_angle_get(self):
        return list(self._state)

    def joints_angle_ctrl(self, angles, speed, acc):
        print(f"  [dry-run] joints_angle_ctrl angles={[round(a,1) for a in angles]} "
              f"speed={speed} acc={acc}")
        self._state = list(angles)

    def gripper_angle_ctrl(self, angle, speed, acc):
        print(f"  [dry-run] gripper_angle_ctrl angle={angle} speed={speed} acc={acc}")
        self._state[5] = angle

    def pose_ctrl(self, pose):
        print(f"  [dry-run] pose_ctrl pose={[round(p,1) for p in pose]}")

    def pose_get(self):
        # plausible HOME-ish pose [x,y,z,t,r,g] (mm + deg)
        return [300.0, 0.0, 200.0, 0.0, 0.0, 5.0]

    def torque_set(self, cmd):
        print(f"  [dry-run] torque_set cmd={cmd}")

    def move_init(self):
        print("  [dry-run] move_init")
        self._state = list(INIT_POS)

    def disconnect(self):
        print("  [dry-run] disconnect")
