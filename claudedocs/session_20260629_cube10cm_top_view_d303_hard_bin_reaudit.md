# Session 2026-06-29 - Cube10cm top-view D303 hard-bin re-audit

## Scope

- Re-audit D302 before using it as a repair target.
- No PPO training, long PPO, PPO ladder, partial actor preservation, real actor
  update, render, cleanup, RunPod/B200/SSH, Track A, VLA fine-tuning, or RoArm
  deployment was performed.
- Goal: determine whether D301 hard episodes fail in fresh one-bin processes or
  only after sequential multi-bin reuse inside one Isaac process.

## Why this was necessary

D302 initially made the hard-bin issue look like actor/teacher action failure on
episodes `13/322/935`. During follow-up, ep13 passed in fresh single-bin
processes, which contradicted the sequential multi-bin D302 result. That meant
the D302 interpretation had to be re-audited before any actor repair or
teacher-KL decision.

## Runtime checks

### D256 recorded-action replay

- Episodes: `221,198,13,322,935`
- `num_envs=5`, `collect_steps=580`, `hold_steps=3`
- Result:
  - contact/reaction/useful: `1.0/1.0/1.0`
  - overshoot: `0.0`
  - max XY: `0.0018194274744018912m`
  - mean XY: `0.0004215548397041857m`
- Artifact:
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/d303_hard_episode_d256_replay_dataset.pt`

### Offline actor-vs-D256 action comparison

- Actor: D300 seed `29604` `model_0.pt`
- Dataset: D303 hard-episode D256 replay dataset
- Result:
  - verdict: `D290_OFFLINE_ACTOR_BATCH_DIAGNOSTIC_PASS`
  - samples: `2900`
  - MSE/MAE/cosine: `0.007595527917146683` /
    `0.03969154506921768` / `0.9679368734359741`
  - actor pred abs mean/max: `0.22799161076545715` /
    `1.4814906120300293`
  - target abs mean/max: `0.2265177220106125` / `1.0`
- Artifact:
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/d303_offline_actor_vs_d256_hard_episodes/offline_actor_batch_diagnostic_summary_d290.json`

### Manual closed-loop recovery

- Episodes: `221,198,13,322,935`
- Reset path: manual D256 pose reset
- Result:
  - verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
  - useful/overshoot: `1.0/0.0`
  - max XY mean/max: `0.00007656301750103012m` /
    `0.00030884623993188143m`
  - actor-vs-recorded MSE/MAE/cosine: `0.1194179430603981` /
    `0.20330607891082764` / `0.7233831882476807`
  - actor-vs-recovery MSE/MAE/cosine: `0.6908975839614868` /
    `0.6131635904312134` / `0.0008526476449333131`
- Interpretation:
  - Manual closed-loop execution does not reproduce D302 hard-bin overshoot.
  - Recovery labels are very different from actor actions, but the actor still
    remains safe under this manual reset run.
- Artifact:
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/d303_closed_loop_recovery_hard_episodes/closed_loop_recovery_summary_d303_closed_loop_recovery_hard_episodes.json`

### Fresh one-bin env-hook probes

- Actor: D300 seed `29604` `model_0.pt`
- `num_envs=5`
- Fresh process per episode.
- Result:
  - ep13: useful `1.0`, overshoot `0.0`
  - ep322: useful `1.0`, overshoot `0.0`
  - ep935: useful `1.0`, overshoot `0.0`
- Artifacts:
  - `.../d303_envhook_ep13_n5_actor/d256_reset_bin_actor_probe_summary_d286.json`
  - `.../d303_envhook_ep322_n5_actor/d256_reset_bin_actor_probe_summary_d286.json`
  - `.../d303_envhook_ep935_n5_actor/d256_reset_bin_actor_probe_summary_d286.json`

### Sequential multi-bin replay in one process

- Same actor and same five hard episode ranges, but run sequentially in one
  Isaac process.
- Result:
  - `221/198` passed
  - ep13 overshot
  - ep322 no-useful
  - ep935 overshot
- Interpretation:
  - This reproduces the D302 pattern, but only in sequential multi-bin reuse.
  - This matches the existing D289 warning that multi-batch collection inside
    one Isaac process is unsafe for this probe family.

## Decision

- D302 multi-bin actor/teacher hard-bin failure claims are superseded.
- Do not use sequential multi-bin Isaac probes as blocker or promotion evidence.
- Fresh-process one-bin probes are required for hard-bin diagnostics.
- D257 teacher-KL, hard-bin supervised repair, or action projection should not
  be applied based on D302 alone.
- The unresolved blocker returns to the true PPO collection path. Future tiny
  PPO gates must use the new `collection_final_env_trace_iter_<N>.jsonl` export
  from `roarm_rl/train_cube_push_ppo.py`.

## Verdict

`D303_HARD_BIN_MULTI_PROCESS_REAUDIT_SUPERSEDES_D302_NO_REPAIR_YET`
