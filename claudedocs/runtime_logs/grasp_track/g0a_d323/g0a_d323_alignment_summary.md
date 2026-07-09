# D323 G0a Frame Repair Probe

이번 case는 G0a repair이며 신규 변수나 사다리 전진은 없다.

- verdict: `D323_G0A_STRICT_POSE_INFEASIBLE_STOP`
- trials: `0`
- pass_all: `0/0`
- hard_failure: `True`
- frame audit: `claudedocs/runtime_logs/grasp_track/g0a_d323/frame_audit.json`
- output CSV: `claudedocs/runtime_logs/grasp_track/g0a_d323/g0a_d323_alignment_trials.csv`

## Stop Reason

Requested link5 +z radial and link5 +x tangent pose family was not feasible within 5mm/3deg thresholds; Step 3 retrial not run by prompt stop rule.

## Best Strict Attempt

- tcp pose error: `35.729 mm`
- link5 +x error: `5.942 deg`
- link5 +z error: `43.015 deg`

## Criteria

- TCP pose error <= 5mm and link5 axis orientation error <= 3deg.
- Fixed-jaw face gap to cube face <= 3mm and no penetration.
- Cube XY displacement < 5mm.
- Strict pass requires all 10 trials to satisfy all criteria.

## Trial Table

| trial | pose err mm | orient err deg | face gap mm | penetration mm | cube disp mm | pass |
|---:|---:|---:|---:|---:|---:|:---:|

## Failure Counts

- tcp_pose: `0`
- orientation: `0`
- fixed_jaw_gap: `0`
- fixed_jaw_penetration: `0`
- cube_displacement: `0`
- arm_joint_tracking: `0`
