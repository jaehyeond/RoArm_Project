---
name: collect_data_manual FAIL flow analysis
description: Detailed analysis of save_episode() FAIL validation flow, bugs found, and fixes applied (2026-03-26)
type: project
---

# FAIL Validation Flow in collect_data_manual.py (2026-03-26)

## Root Cause of "no FAIL reason in terminal"
pynput keyboard listener runs in a background thread. Python's stdout is fully buffered
(4096 bytes) when conda run does not allocate a PTY. The print() calls in save_episode()
go into the buffer but do not flush until the buffer fills or the process exits.

**Fix applied**: All print() calls in save_episode() and on_key_press() now use flush=True.
sys.stdout.flush() added as an explicit backup after the FAIL block.

**Why:** conda run / background thread = no PTY = fully buffered stdout = print() output sits
in buffer invisibly until flushed.

**How to apply:** Any future print() calls from pynput listener thread must use flush=True.

## All FAIL Conditions (hard issues → pending_confirmation)

| # | Condition | Threshold | Code line |
|---|-----------|-----------|-----------|
| F1 | Gripper never opened during episode | max_gripper < 40° | L437-438 |
| F2 | Gripper range too small (opened but barely) | (max - min) < 15° | L442-443 |
| F3 | Gripper closed while arm too high | shoulder_at_grip_close < 40° | L450-452 |
| F4 | Gripper closed too high in Z | z_at_grip_close > 130mm | L460-462 |
| F5 | Too few frames | num_frames < 90 | L471-472 |

## Soft WARNING conditions (auto-save still proceeds)
- W1: gripper peak < 50° (but ≥ 40°)
- W2: shoulder at close 40-50°
- W3: gripper opened but no close detected (grip_was_open=True, shoulder_at_grip_close=None)
- W4: 90 ≤ num_frames < 120
- W5: num_frames > 600
- W6: min_z > 160mm (shallow grasp)

## Most Likely Cause for Reported Episode
OSD: Frames=127, Z=-110mm, Sh=45deg, Grip=2deg [CLOSED], ZONE=NEAR

Most likely F1: "그리퍼 미개방 (max=X° < 40°)"
- Gripper was 2° at end → if it never exceeded 40° during the episode, grip_was_open=False
- F5 ruled out (127 ≥ 90), Z fine (-110mm << 160mm), Sh not the trigger (F3 only applies
  if grip_was_open=True first)
- If gripper peaked at e.g. 35°, F1 triggers (threshold is 40°, not 35°)

## pending_confirmation Blocking Behavior
When pending_confirmation=True, on_key_press intercepts ALL keys and returns early.
Space (start new recording) is blocked until Enter or Backspace resolves the FAIL.
This is correct design, but FAIL reason was invisible in terminal (see root cause above).

## No Home Position Required Between Episodes
save_episode() → _reset_episode_tracking() does NOT move robot home or require home.
User can start next episode from any arm position immediately after saving.
Data quality concern: SmolVLA episodes should start from consistent home position.
Script does NOT enforce this — user responsibility.

## _reset_episode_tracking() Completeness
Properly clears: current_episode, is_recording, pending_confirmation, pending_fail_reasons,
min/max elbow/gripper/z, max_shoulder, shoulder_at_grip_close, z_at_grip_close,
grip_was_open, grip_open_frame, grip_close_frame, prev_gripper.
episode_count is NOT reset (correct — incremented separately before reset call).

## OSD Enhancement
Added pending_fail_reasons list: stored at FAIL time, displayed on OSD under the FAIL banner.
User can now see WHY it failed on the camera window without needing the terminal.
Format: "[1] 그리퍼 미개방 (max=35° < 40°)" etc., one line per reason.

## Files Modified
- collect_data_manual.py: import sys added, all FAIL/save print() calls use flush=True,
  pending_fail_reasons field added, OSD shows fail reasons
