# Leader-Follower Pipeline Critical Review
Data Agent — 2026-04-01

---

## SUMMARY VERDICT

The L-F pipeline has **3 confirmed bugs** and **4 design concerns** that could silently degrade training data quality. None are catastrophic, but B3 and the raw-vs-clamped issue can produce incorrect action labels that corrupt the dataset.

---

## CONFIRMED BUGS

### B1 — Zone classification uses Follower base angle, not Leader base angle (Lines 576-578, 935)

**Location**: `collect_data_manual.py`, lines 571-578 (save_episode) and line 935 (OSD)

**Code**:
```python
# Lines 576-578 (save_episode):
gf_base = gf["data"].get("angles", [0])[0]   # "angles" = Follower angles
ep_zone = classify_zone(gf_base)

# Line 935 (OSD):
base_angle = angles[0]   # angles = Follower angles
```

**Problem**: In L-F mode, `angles` is the Follower's base angle. Zone should represent where the sponge is, which maps to the Leader's commanded position — not where the Follower physically arrived. With servo lag and clamping, they can diverge at the grasp moment. Episodes near zone boundaries (e.g., base=10° boundary between CENTER and RIGHT) can be misclassified.

**Fix**: In L-F mode, use `leader_angles[0]` for zone classification:
```python
if self.lf_mode and "leader_angles" in gf["data"]:
    gf_base = gf["data"]["leader_angles"][0]
else:
    gf_base = gf["data"].get("angles", [0])[0]
ep_zone = classify_zone(gf_base)
```

---

### B2 — Raw leader_angles saved as action, but clamped values sent to Follower (Lines 879-894)

**Location**: `collect_data_manual.py`, lines 879-894

**Code**:
```python
clamped = [max(lo, min(hi, a))
           for a, (lo, hi) in zip(leader_angles, JOINT_LIMITS)]
self.robot.joints_angle_ctrl(angles=clamped, speed=0, acc=0)  # Follower commanded clamped
# ...
self.save_frame(rgb, depth, angles, pose, second_rgb, leader_angles)  # RAW saved as action
```

**Problem**: The Follower is commanded to `clamped` values but the dataset stores `leader_angles` (raw, potentially outside JOINT_LIMITS) as the action. If the Leader goes to, say, base=+195° (just outside ±190°), the action label will be +195° but the Follower was only commanded to +190°. The model learns to predict +195° for a state the robot can only reach at +190°. This is a label–execution mismatch.

In normal operation the operator drives Leader freely and rarely exceeds limits, but the inconsistency is structurally wrong. The ground truth for "what was actually commanded" is the clamped value.

**Fix**: Save clamped values as action:
```python
self.save_frame(rgb, depth, angles, pose, second_rgb, clamped)
```

---

### B3 — _safe_angle_read failure returns [0,0,0,0,0,0] silently (Lines 396-405)

**Location**: `collect_data_manual.py`, lines 396-405

**Code**:
```python
def _safe_angle_read(self, arm):
    for _ in range(5):
        try:
            angles = arm.joints_angle_get()
            if angles is not None and len(angles) >= 6:
                return list(angles)
        except Exception:
            time.sleep(0.05)
    return [0, 0, 0, 0, 0, 0]   # silent fallback
```

**Problem**: If the Leader's serial communication fails after 5 retries, `get_leader_angles()` returns `[0,0,0,0,0,0]`. This means:

1. `frame_data["leader_angles"]` is stored as `[0,0,0,0,0,0]`.
2. In `convert_to_lerobot_v3.py` line 188, the check `if "leader_angles" in frame_data` succeeds — key exists.
3. `action = [0,0,0,0,0,0]` gets written to the dataset for that frame.
4. The mirror command at line 881 sends `clamped = [0,0,0,0,0,0]` to Follower — sudden motion toward home mid-episode.

There is NO detection, warning, or episode invalidation for this failure mode. The corrupt frame is saved silently.

The Follower `_safe_angle_read` has the same problem: if Follower read fails, `state = [0,0,0,0,0,0]` is saved.

**Fix**: Return `None` on failure, then detect and handle:

```python
def _safe_angle_read(self, arm):
    for _ in range(5):
        try:
            angles = arm.joints_angle_get()
            if angles is not None and len(angles) >= 6:
                return list(angles)
        except Exception:
            time.sleep(0.05)
    return None   # caller must handle

# In main loop (after get_leader_angles()):
leader_angles = self.get_leader_angles()
if self.lf_mode:
    if leader_angles is None:
        # Read failure: skip mirror command, drop frame if recording
        if self.is_recording:
            print("[WARN] Leader read failed — frame dropped", flush=True)
        # do NOT call save_frame this iteration
        continue  # or set a flag to skip save_frame below
    else:
        clamped = [max(lo, min(hi, a))
                   for a, (lo, hi) in zip(leader_angles, JOINT_LIMITS)]
        self.robot.joints_angle_ctrl(angles=clamped, speed=0, acc=0)
```

Similarly for Follower (`get_robot_angles`): if None, use previous frame's angles rather than [0,0,0,0,0,0], or skip saving.

---

## DESIGN CONCERNS

### D1 — HOME validation reads Follower, not Leader (Line 728)

In L-F mode, Space-key HOME check reads `get_robot_angles()` (Follower). This is semantically correct — the model observes the Follower's state, so Follower should be at HOME before recording. However, if the operator wiggled the Leader before pressing Space, the Follower has tracked it away from HOME. The block message only says "현재 dist=X°" without specifying which arm is out of position. In practice the operator may be confused about why HOME fails.

**Recommendation**: Print which arm is out of position, or explicitly note "Follower must be at HOME."

---

### D2 — Gripper timing statistics track Follower gripper (Lines 884-917)

`grip_was_open`, `grip_open_frame`, `grip_close_frame`, `shoulder_at_grip_close` all use `gripper = angles[5]` (Follower). The Follower mirrors the Leader with ~1 servo cycle lag. For a ~40-frame grasp window, a 1–2 frame shift in `grip_close_frame` affects which base angle is selected for zone classification (line 574 uses `grip_close_frame` to index `current_episode`). Low severity but compounded with B1.

---

### D3 — No real-time hard block on over-quota zone during recording

The ZONE_OVERQUOTA_LIMIT soft block only triggers at save time (lines 580-592). The operator completes a full 6–7 second episode before learning the zone is over-quota. They can then force-save anyway. Zone distribution enforcement is advisory, not structural.

The OSD does show zone info in real time (lines 971-988), including an "!! ZONE FULL" banner, but there is no recording-start block even when over quota. A motivated or inattentive operator can continuously over-fill one zone.

---

### D4 — Last-frame action fallback in convert is not reachable for L-F data (Line 194)

`convert_to_lerobot_v3.py` line 194: `action = state.copy()` is only reached when there is no `leader_angles` key AND it is the last frame. For L-F episodes this branch is unreachable in normal operation. It only becomes reachable if B3 is not fixed and a frame somehow has no `leader_angles` stored (which cannot happen with the current code since `[0,0,0,0,0,0]` is always stored). No action needed independently, but fix B3 first.

---

## TIMING CORRECTNESS SUMMARY (Questions A, B)

The per-iteration execution order is:
```
1. get_camera_frame()     (Kinect capture ~15ms)
2. get_robot_angles()     (Follower read → saved as state)
3. get_robot_pose()       (Follower FK)
4. get_leader_angles()    (Leader read → saved as action)
5. joints_angle_ctrl()    (mirror command to Follower)
6. save_frame(state, action)
7. OSD + cv2.waitKey
8. time.sleep(0.01)
```

Follower read (2) happens BEFORE mirror command (5). This is the standard (s_t, a_t) IL convention: state is "where the Follower is now," action is "where it should go." The Follower will be near `action` by the next frame. This is correct.

Both readings (Follower at step 2, Leader at step 4) happen within the same loop iteration, separated by ~5ms. They are not hardware-synchronized but the ~5ms skew at 30fps is negligible for this task.

---

## CONVERT FILE CORRECTNESS (Questions D, G)

**D: Same-timestep, no offset.** `state = follower_angles[t]`, `action = leader_angles[t]`. Correct for L-F IL.

**G: Leader read failure.** Currently silent (B3). With B3 fix (return None, drop frame), this becomes safe.

**Video encoding**: `cv2.imwrite` saves JPG (lossy). This is standard for VLA datasets. No issue.

**State/action shape**: Both (6,) float32. Feature definition in convert matches `JOINT_LIMITS` dimension. Correct.

---

## CLEANUP CORRECTNESS (Question E, F)

**E: HOME enforcement in L-F mode.**
- `run()` lines 830-842: Both arms commanded HOME, double-sent, 3s settle. Correct.
- `on_key_press 'i'` lines 771-782: Both arms to HOME. Correct.
- Space check (line 728): Uses Follower. Semantically correct (Follower = model observation).

**F: Race conditions.**
- `is_recording`, `pending_confirmation`, `current_episode` shared between pynput listener thread and main loop.
- No locks used. CPython GIL makes list append + len() read safe in practice.
- The save_episode path (called from listener thread) modifies `current_episode` while main loop may be inside `save_frame()` appending to it. In CPython this is GIL-protected. Technically unsafe but no observed failures expected.

---

## PRIORITY TABLE

| ID | Severity | One-line description |
|----|----------|----------------------|
| B3 | HIGH     | Serial read failure → silent [0,0,0,0,0,0] action in dataset + sudden Follower home-snap |
| B2 | MEDIUM   | Raw leader angles saved, but clamped values sent to robot (label mismatch for out-of-limit poses) |
| B1 | LOW      | Zone uses Follower base angle; should use Leader base angle for sponge position accuracy |
| D1 | INFO     | HOME check doesn't say which arm is out of position |
| D2 | INFO     | Gripper timing uses Follower (1-frame lag from Leader) |
| D3 | INFO     | No recording-start hard block for over-quota zones |
| D4 | INFO     | Last-frame fallback unreachable for L-F data (benign) |
| TH | INFO     | Threadless access to shared state (GIL covers in practice) |
