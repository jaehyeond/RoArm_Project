# T3U side meeting1 trace-video preregistration

Status: preregistered before renderer source creation or any output frame.

## Purpose and authority boundary

- Purpose: produce a lab-meeting-readable MP4 from the already completed P13
  native trace, without rerunning Isaac Sim/PhysX and without changing any
  scientific result.
- This is a **posthoc trace visualization — not RTX, not scientific authority**.
  Native P13 JSON/NPZ remains the numerical authority.
- Existing `t3u_side_preflight13_*` artifacts are read-only and will not be
  overwritten, renamed, or amended.
- No hardware, Isaac/Kit, CUDA, training, new simulation, or physics state write
  is permitted in this task.

## Frozen inputs

- Trace: `t3u_side_preflight13_trace.npz`
  - SHA-256: `ee67d3516a1c7871e5f48d455b420c3f5985ae889bceb097536904548e8134ee`
- Results: `t3u_side_preflight13_results.json`
  - SHA-256: `8324ed7a9682ccb297985dd733c9e91c480bed9ce65bb02672d5b40226eea6d5`
- Plan: `t3u_side_preflight13_plan.json`
  - SHA-256: `d7fcfb47c26c38f4817ce7630671d915e0d77a4b3bcc1f2d7df40fd816f94f66`
- Representative is frozen to the result/plan binding:
  `trial_id=c05_o00`, active slot `0`, candidate
  `side_sdg_005_raw_025092`, zero pinch offset.
- Frozen P13 outcome shown in the video:
  representative classification `premature_jaw_contact`; population-selected
  verdict `NO_BILATERAL_SIDE_CONTACT`; success count `0/5`.

## Rendering contract

- CPU-only Python/matplotlib renderer using the Agg backend.
- Source trace length must be exactly 2340 rows, with recorded physics steps
  exactly `1..2340`, and representative slot read from the scalar NPZ binding.
- Frame indices are exactly `range(0, 2340, 10)`: 234 frames.
- Video is exactly 20 fps, H.264/yuv420p MP4, target 1280x720, duration 11.7 s.
- Every frame must state:
  `posthoc trace visualization — not RTX, not scientific authority`.
- Main view: actual moving-body centers (`moving_body_pos_m`) joined in body
  order, fixed base, actual oriented D29xH50 cylinder, actual TCP, planned grasp
  TCP and antipodal midpoint, and available object-contact force arrows.
- Side panels: actual/target gripper joint, fixed/moving jaw force magnitudes,
  object/TCP height, current phase and physics step, and frozen failure verdict.
- Rendered values are display copies only; no rendered value is fed back into a
  gate or result.

## Forward-only outputs

- Renderer: `t3u_side_meeting1_trace_video_render.py`
- Frames: `t3u_side_meeting1_trace_video_frames/frame_0000.png` through
  `frame_0233.png`
- MP4: `t3u_side_meeting1_trace_video.mp4`
- Contact sheet: `t3u_side_meeting1_trace_video_contact_sheet.png`
- Manifest: `t3u_side_meeting1_trace_video_manifest.json`

## Completion gates

1. All three frozen input hashes match before rendering.
2. Exactly 234 nonempty PNG frames are generated at 1280x720.
3. `ffprobe` reports H.264, yuv420p, 1280x720, 20 fps, 234 decoded frames, and
   11.7 s duration.
4. A full decode to null exits zero.
5. Manifest records input/output hashes, every frame hash, phase coverage, key
   source/result metrics, and explicit non-authority wording.
6. Contact sheet is visually inspected before delivery; observations may report
   the failed grasp honestly but must not reinterpret the native verdict.

## Pre-output contract correction

The renderer's first input-only validation (before creating the frame directory
or any generated artifact) found that `physics_step` is the one-based recorded
counter `1..2340`, while array/source indices are `0..2339`. The contract above
was corrected before output generation. This changes no sampled row: frame
source indices remain exactly `0,10,...,2330` and carry recorded physics steps
`1,11,...,2331`.
