# Side-midpoint grasp — lab-meeting brief (P13)

Date: 2026-08-13 KST  
Scope: fixed-base RoArm-M3, analytic D29 x H50 mm / 24.83 g cylinder, five radial offsets  
Verdict: **GRASP FAILED — 0/5 (`NO_BILATERAL_SIDE_CONTACT`)**

## One-sentence result

The arm reached all five side-midpoint targets (1.5253–1.5258 mm arrival error), but
both jaws recorded 0 N during the close phase and the cylinder's corrected lift stayed
within -0.000237 to +0.000792 mm, far below the required >6 mm.

## Exact candidate results

| trial | radial offset (mm) | arrival (mm) | close fixed/moving (N) | corrected object lift (mm) | TCP rise (mm) | classification |
|---|---:|---:|---:|---:|---:|---|
| c05_o00 | 0.00 | 1.525674 | 0 / 0 | +0.000196 | 24.0516 | premature_jaw_contact |
| c05_o01 | 0.25 | 1.525837 | 0 / 0 | +0.000244 | 24.0486 | no_bilateral_close |
| c05_o02 | 0.50 | 1.525629 | 0 / 0 | +0.000190 | 24.1066 | no_bilateral_close |
| c05_o03 | 0.75 | 1.525481 | 0 / 0 | +0.000792 | 24.1515 | no_bilateral_close |
| c05_o04 | 1.00 | 1.525311 | 0 / 0 | -0.000237 | 24.1584 | no_bilateral_close |

Physics completed 2 diagnostic callbacks plus 2,340 task callbacks in 74.14 s.
Numeric integrity, measurement validity, contact instrumentation, and the excluded
support-contact witness all passed. The preflight itself is not a canonical scientific
run and remains marked non-authoritative.

## Meeting attachments

- `t3u_side_meeting1_trace_video.mp4` — 11.7 s, H.264, 1280x720, 20 fps,
  234/234 frames fully decoded. This is a CPU posthoc visualization of the exact trace,
  not an RTX render and not an independent scientific authority.
- `t3u_side_meeting1_trace_video_contact_sheet.png` — eight representative frames.
- `t3u_side_preflight13_timeline.rrd` — replayable Rerun record.
- `t3u_side_preflight13_results.json` — metric authority.
- `t3u_side_preflight13_trace.npz` — full 2,340-step numeric trace.
- `t3u_side_meeting1_lab_bundle_v3.zip` — current 9-file presentation/evidence bundle,
  4,172,534 bytes, ZIP integrity 9/9 PASS, SHA-256
  `2bcdc926e60b1848026cf4c6bcd62610e04f1047bedabe355f92cb953ce67ac1`.

MP4 SHA-256:
`14a9b6d9ef6dee9fae0210c7f7eda524692548d3d62e3a3608972f10b51f8414`

## What the video shows

The robot executes settle, approach, stage, descend, close, hold, and lift. The TCP
rises by about 24 mm during lift while the cylinder remains on the support. The force
panels show no bilateral jaw contact during close, so later lift-phase jaw loads cannot
be interpreted as a valid grasp.

## Do not claim

- Do not call this a successful grasp; success is 0/5.
- Do not use the large separation in `decision_snapshot.png` as a target error. That
  diagnostic compares a target TCP origin with an actual link5 body origin.
- Do not call the meeting MP4 an Isaac RTX camera render. The original stopped-timeline
  Replicator renderer stalled before its first frame; the physics trace itself completed.

## Next bounded action

Keep the completed P13 physics immutable. Two local RTX-only paths have now failed at the
first Replicator warmup without producing a frame: the original P13 renderer was terminated
after its 7,200 s bound, and the independently audited forward-only `render1` repair was
terminated after its 90 s post-capture no-progress bound. Both process groups were reaped
and the GPU returned to its pre-run PID set. Do not retry either consumed prefix.

A RunPod RTX PRO 6000 Blackwell Server Edition (96 GB) A/B test also produced zero RTX
frames. CUDA compute was healthy (`torch.cuda=True`, `cuInit=0`), and the exact P13 input
gate passed after reproducing the original absolute repository path and nanosecond mtimes.
Kit then reported that no suitable graphics GPU/foundation interface existed. `vulkaninfo`
confirmed that the Pod exposed the CUDA device but not `/dev/nvidia-modeset`; the generic
PyTorch container therefore did not expose a Vulkan/RTX graphics device. The Pod was stopped
and deleted, and the account returned to zero Pods. Raw evidence is in
`t3u_side_rendercloud1_runpod_evidence.tar.gz` (SHA-256
`5469c2fcf6eaed522c6a670ebc1731e8b0da360b15972237cf288d80e91e0610`).

Use the verified CPU trace video and contact sheet for the lab meeting, explicitly labeled
as a failed-grasp diagnostic and not an RTX or scientific-authority artifact. A future cloud
RTX replay requires a graphics-capable Isaac Sim 5.1 container/runtime that passes
`vulkaninfo` and exposes `/dev/nvidia-modeset` before installing or transferring the project.
Any grasp improvement is a separate future case because it would change the scientific
trajectory or target variables.
