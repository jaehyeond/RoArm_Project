# D286 D256 Reset Bin Actor Probe Comparison

- verdict: `D286_NO_RESET_BIN_OR_ACTION_SCALE_FIX_READY_FOR_PPO`

## Runs

- `default_action_scale0040_steps580_corrected`: action_scale `0.04`, cap threshold `0.25`, cap max `0.8229166865348816`, useful max `0.0`, overshoot max `0.0`
  - ep 1-208: cap `0.6302083730697632`, useful `0.0`, overshoot `0.0`, policy_abs_max `2.7579410076141357`, cube_y `-0.06911564684238564`
  - ep 209-370: cap `0.7604166865348816`, useful `0.0`, overshoot `0.0`, policy_abs_max `3.0143487453460693`, cube_y `-0.0543918915948755`
  - ep 371-537: cap `0.8229166865348816`, useful `0.0`, overshoot `0.0`, policy_abs_max `2.346695899963379`, cube_y `-0.02238095218480444`
  - ep 538-715: cap `0.703125`, useful `0.0`, overshoot `0.0`, policy_abs_max `2.1663737297058105`, cube_y `0.012263513459647829`
  - ep 716-999: cap `0.78125`, useful `0.0`, overshoot `0.0`, policy_abs_max `2.0915462970733643`, cube_y `0.0936602445788124`
- `action_scale0010_steps580_corrected`: action_scale `0.01`, cap threshold `1.0`, cap max `0.0833333358168602`, useful max `0.0`, overshoot max `0.0`
  - ep 1-208: cap `0.010416666977107525`, useful `0.0`, overshoot `0.0`, policy_abs_max `1.2095534801483154`, cube_y `-0.06911564684238564`
  - ep 209-370: cap `0.015625`, useful `0.0`, overshoot `0.0`, policy_abs_max `1.224318265914917`, cube_y `-0.0543918915948755`
  - ep 371-537: cap `0.0052083334885537624`, useful `0.0`, overshoot `0.0`, policy_abs_max `1.3920042514801025`, cube_y `-0.02238095218480444`
  - ep 538-715: cap `0.0781250074505806`, useful `0.0`, overshoot `0.0`, policy_abs_max `1.7992123365402222`, cube_y `0.012263513459647829`
  - ep 716-999: cap `0.0833333358168602`, useful `0.0`, overshoot `0.0`, policy_abs_max `2.226398468017578`, cube_y `0.0936602445788124`

## Interpretation

- Default action_scale=0.04 keeps no useful signal and produces severe cap pressure across all bins.
- action_scale=0.01 removes most cap pressure but still produces no useful/contact signal across all bins.
- Therefore the next fix should not be long PPO and should not be reset-bin filtering alone; it should repair the actor/teacher bridge or add action projection/constraint before another PPO smoke.
