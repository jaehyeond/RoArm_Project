# D288 D256 Label/Env Contract Audit

- verdict: `D288_LABEL_CLEAN_TEACHER_ONLINE_CONTRACT_MISMATCH_CONFIRMED`
- no PPO learning: `True`
- Isaac Lab launched: `False`
- overshoot threshold: `0.020 m`
- manifest: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/label_package_d248/episode_split_manifest.csv`

## D256 label split

- train_clean_positive episodes: `737`
- train clean overshoot episodes: `0`
- train clean max_xy >= threshold: `0`
- train clean contact/reaction: `737` / `737`
- train clean max_tap_disp_xy_m min/p50/p90/p95/p99/max: `0.000671 / 0.005821 / 0.013904 / 0.016036 / 0.018031 / 0.019745`
- train clean max_tap_disp_along_m min/p50/p90/p95/p99/max: `0.000000 / 0.005283 / 0.010592 / 0.014561 / 0.017648 / 0.019458`
- eval overshoot episodes: `167`
- eval overshoot seen episodes: `167`
- eval overshoot max_tap_disp_xy_m min/p50/p90/p95/p99/max: `0.020069 / 0.070663 / 0.162419 / 0.203046 / 0.258294 / 0.264307`

## D287 online probes

- teacher probe verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_FAIL_NO_SAFE_BIN`
- teacher safe bins: `[]`
- teacher overshoot rate range: `0.125000..0.875000`
- teacher useful rate range: `0.125000..0.875000`
- actor probe verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_FAIL_NO_SAFE_BIN`
- actor safe bins: `[]`
- actor overshoot rate range: `0.125000..0.593750`
- actor useful rate range: `0.218750..0.562500`

## Interpretation

D256 train_clean_positive labels are clean under the same 0.020 m XY overshoot threshold, while D287 online teacher/actor probes still overshoot. This points to an online teacher/action execution contract problem, not a permissive label problem.

## Next order

1. Do not run long PPO.
2. Run/review D256 recorded-action replay in the live env.
3. If replay is clean, rebuild or constrain the teacher/action bridge before actor distillation.
4. If replay also overshoots, fix env physics/action application or label-env semantics first.
5. Only after teacher-off/bin diagnostics pass, run tiny PPO smoke plus TensorBoard gate.
