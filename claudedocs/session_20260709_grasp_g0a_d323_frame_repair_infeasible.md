# D323 G0a Frame Repair

이번 case의 신규 변수: `[]` — D323은 G0a repair이며 사다리 전진이 아니다.

## Scope

- Active Case: `G0a`
- Goal: D322 실패 원인이 TCP/jaw frame 표현 오류인지 감사하고, frame 기반 목표 표현으로 같은 G0a 판정을 재시도한다.
- Non-goals: G0b, cylinder, gripper close, grasp, lift, PPO/RL, render scale-up, VLA, RoArm, B200, friction/material change, position randomization.

## Runtime Artifacts

- Script: `sim_scripts/cube10cm_top_view_d323_grasp_g0a_frame_repair_probe.py`
- Frame audit: `claudedocs/runtime_logs/grasp_track/g0a_d323/frame_audit.json`
- Summary: `claudedocs/runtime_logs/grasp_track/g0a_d323/g0a_d323_alignment_summary.json`
- Markdown summary: `claudedocs/runtime_logs/grasp_track/g0a_d323/g0a_d323_alignment_summary.md`

## Frame Audit

Live Isaac frame audit confirmed the runtime TCP contract:

| pose | TCP in link5 frame | TCP offset error | link5 +z world |
|---|---:|---:|---|
| home | `[0, 0, 0.115428]` m | `0.000044mm` | `[0.999999, -0.000004, -0.001012]` |
| audit_pose_a | `[0, 0, 0.115428]` m | `0.000048mm` | `[0.337739, 0.000427, -0.941240]` |
| audit_pose_b | `[0, 0, 0.115428]` m | `0.000063mm` | `[0.382497, 0.178359, -0.906578]` |

`hand_tcp` is not a separate body in the runtime articulation; the env computes TCP as:

`TCP = link5 position + link5 rotation * [0, 0, 0.115428]`.

The static gripper-link origin is not the fixed-jaw face. At the audited open poses, `gripper_link` sits at approximately `[0, 0.018821, 0.052035]` in link5 frame. G0a should not use that body origin as a grasp-face proxy.

## Feasibility Result

The requested strict frame family was:

- link5 `+z` tool axis horizontal and radial from base to cube.
- link5 `+x` jaw-separation axis horizontal and tangential.
- cube center is TCP `+x` side by `42mm`.
- TCP tip passes the cube near face by `10mm` along radial.

Offline IK did not find a joint-limit-valid solution within the `5mm / 3deg` gate.

| candidate | TCP error | link5 +x error | link5 +z error | interpretation |
|---|---:|---:|---:|---|
| best strict attempt | `35.729mm` | `5.942deg` | `43.015deg` | strict target not feasible |
| position-only, tangent `-1` | `0.261mm` | `9.148deg` | `69.124deg` | position reachable, strict tool-axis orientation not reachable |
| position-only, tangent `+1` | `0.261mm` | `170.852deg` | `69.124deg` | position reachable with opposite jaw orientation |

## Verdict

`D323_G0A_STRICT_POSE_INFEASIBLE_STOP`

The Step 3 retrial was not run because the prompt explicitly required stopping if the strict pose family is impossible. Running the 10-trial alignment with this target would be a forced invalid experiment, not a G0a repair.

## Interpretation

D322 was not only an offset-direction bug. D323 verifies the TCP offset contract, then shows the stricter horizontal side-grasp orientation is not reachable at the 10cm cube center height with this 5-DOF RoArm/link5 runtime frame. The reachable family can place the TCP at the desired side location, but link5 `+z` remains down/backward by about `69deg`.

Next valid G0a work is to define an attainable alignment target representation from the audited frame contract, such as fixed-jaw/TCP position plus the reachable wrist-axis family. Do not tune the `42mm` or `10mm` offsets repeatedly, and do not advance to G0b/cylinder/gripper close until the reachable G0a alignment criterion is explicitly redefined and passes.
