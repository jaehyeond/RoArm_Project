# START_HERE.md

Last updated: 2026-06-05 KST (B200 disconnected; Track A remains blocked and separate. Active branch is the professor 10cm/0.72kg cube push/tap DiffIK diagnosis. Earlier final-displacement gates failed, and settled-start fixed16 still showed DiffIK clipping/lag, but local DLS feasibility means do not conclude the object is too heavy. The clarified objective is reaction/tap first and final displacement secondary. A local non-GPU reaction gate audit requires reaction evidence plus contact evidence, no posewrite, and no overshoot; it separates reaction PASS from teacher/RL readiness. seed938 is the negative control FAIL (`contact_evidence_rate=0.0`), seed939/seed940 PASS reaction gate but remain `teacher_quality_ready=false`, seed941 randomized 16-env FAILed on contact evidence `0.625`, cap-only seed942 worsened contact evidence to `0.5`, and fixed-y+ seed943 confirmed the y+ bucket is weak (`contact_evidence_rate=0.375`). A local y+ geometry/reach audit shows contact rows average max displacement `0.010986278m`, no-contact rows only `0.000069159m`, and workspace bins are asymmetric (`cube_y0_m<=0` contact `0.625`, `cube_y0_m>0` contact `0.125`, `cube_x0_m<0.25` contact `0.111111`, `cube_x0_m>=0.25` contact `0.714286`). A follow-up y+ trace path/actuator audit shows the target itself moves about `0.020000m` in world y and keeps start-cube side-center z, but final TCP error is mostly vertical (`0.844` contact / `0.859` no-contact z-error fraction) and clip_any remains `1.0` in both traced groups. The approved seed944 fixed-y+ height050 screen is a height-only FAIL: final TCP error improved to `0.022889409m`, but contact evidence fell to `0.0`, reaction was only `0.6875`, and clip remained `0.949198734`; do not use +5cm target height as a data/RL fix. The approved seed945 fixed-y+ good-workspace screen (`x=0.295,y=-0.044`) PASSed reaction/contact (`1.0/1.0`) with no overshoot/posewrite, proving workspace is a real contact discriminator, but `teacher_quality_ready=false` because final TCP error is `0.065514732m` and clip is `1.0`. Added default-preserving `--base_lateral_offset_m`; seed946 good-workspace lateral `-0.020m` PASSed reaction/contact and final 1cm relocation (`final_gate_rate=1.0`, `max_disp_mean_m=0.011251196m`, no overshoot/posewrite), but still `teacher_quality_ready=false` due final TCP error `0.062820967m` and DiffIK clip `1.0`. Added `sim_scripts/cube10cm_push_diffik_probe.py` as the standard 10cm entrypoint; it injects 10cm/0.72kg professor defaults into the shared legacy 3cm-named DiffIK engine while preserving explicit user overrides. Static checks passed; no new GPU runtime, dataset generation, PPO/RL scale-up, VLA, Track A runtime, 1024/10k, or larger candidate audit is approved from this.)

Latest correction: for the professor 10cm/0.72kg push/tap branch, final 1cm
displacement is no longer the primary objective if the real task only needs a
tap/reaction. Treat measured contact, transient displacement, cube lift/z delta,
and cube speed as the primary reaction-event evidence; keep final displacement as
a secondary relocation metric. The reaction audit script confirms seed938 FAILs
because contact evidence is `0.0`, while seed939 and seed940 PASS reaction gate.
seed940 has `max_disp_mean_m=0.010990217`, transient gate `1.0`, final gate
`0.0`, no overshoot, and `teacher_quality_ready=false`. The approved seed941
randomized 16-env screen did not pass the reaction gate: reaction evidence was
`1.0`, but contact evidence was only `0.625`; no-posewrite and no-overshoot were
OK, and teacher quality still failed due TCP error/clipping. The approved cap050
seed942 diagnostic also failed: contact evidence dropped to `0.5`, transient gate
to `0.1875`, final gate to `0.0`, and teacher quality stayed false. The fixed-y+
seed943 diagnostic confirmed y+ is a weak direction bucket: reaction `0.9375`,
contact evidence `0.375`, no overshoot, and teacher quality false. The local
posthoc y+ geometry/reach audit now shows this is not merely reaction-rate
noise: contact rows average max displacement `0.010986278m`, no-contact rows
only `0.000069159m`, and the workspace is asymmetric (`cube_y0_m<=0` contact
`0.625` versus `cube_y0_m>0` contact `0.125`; `cube_x0_m<0.25` contact
`0.111111` versus `cube_x0_m>=0.25` contact `0.714286`). Traced envs still hold
large final vertical TCP-target error (`0.043-0.070m`) at side-center targets,
so next work remains y+ target path/reach/lateral-height/actuator tracking, not
RL/data scale-up. A follow-up trace-path/actuator audit shows target world-y
motion is about `0.020000m` and final target z stays at the start-cube
side-center height, while final TCP error is mostly z error and both contact and
no-contact traced groups keep `clip_any=1.0`. The seed944 height050 y+ screen
then rejected the height-only shortcut: contact evidence `0.0`, reaction
`0.6875`, final TCP error `0.022889409m`, clip `0.949198734`, no posewrite, and
no overshoot. The seed945 good-workspace y+ screen (`fixed_cube_x_m=0.295`,
`fixed_cube_y_m=-0.044`) restored reaction/contact to `1.0/1.0` with no
overshoot/posewrite, but teacher quality remains false because final TCP error
is `0.065514732m` and DiffIK clip is `1.0`. The default-preserving
`--base_lateral_offset_m` patch let seed946 test the lateral error directly; at
`-0.020m`, good-workspace y+ reached reaction/contact `1.0/1.0`, final 1cm gate
`1.0`, max displacement `0.011251196m`, no overshoot/posewrite, but teacher
quality still false from final TCP error `0.062820967m` and clip `1.0`.
To remove the confusing legacy filename issue, future professor 10cm DiffIK
commands should prefer `sim_scripts/cube10cm_push_diffik_probe.py`. That wrapper
does not duplicate physics logic; it injects 10cm/0.72kg tap/reaction defaults
into the shared DiffIK probe and lets explicit command-line overrides win. Its
default displacement/stop/gate values are `0.001m`, not a fixed final 1cm
relocation objective. After explicit approval, one tiny local stiff600 screen
changed only `--arm_stiffness_override 600` inside the seed946 geometry. It kept
tap/reaction PASS (`reaction/contact=1.0/1.0`, no posewrite, no overshoot), but
it is not an improvement over seed946: final relocation secondary is false, max
displacement fell to `0.004667487m`, final displacement to `0.003845483m`, clip
stayed `1.0`, and teacher quality stayed false despite TCP error improving to
`0.046062952m`. Do not treat lower TCP error alone as progress. A follow-up
approved direction-generalization screen seed947 kept seed946 goodxy
(`x=0.295,y=-0.044`) and lateral `-0.020m` but released `fixed_push_dir`; it
FAILed reaction gate because contact evidence was only `0.5625`, with no
posewrite and no overshoot. Direction split shows y+ is the only controlled
direction (`controlled=1.0`, but low-motion `0.75`), x- had contact `0/7`, and
x+/y- had contact but controlled `0`. This is strong evidence seed946 is a
y+-specific/contact-geometry pocket, not a direction-general teacher.

This is the rolling current-state dashboard. It is not full history. Durable
rules live in `claudedocs/DECISIONS.md`; experiment history lives in
`claudedocs/EXPERIMENT_LEDGER.md`; detailed logs live in `claudedocs/session_*.md`.

Do **not** use `HANDOFF.md` or `TASKS.md` as current state.

## B200 Retired / Backup Truth

- NHN/Sogang B200 access expired on 2026-05-22 at 23:59 KST, and the user
  reported B200 now shows disconnect on 2026-05-23 KST. Future research must
  not depend on entering B200 through SSH or on B200-only paths.
- Do not copy, request, or depend on `.ssh` private material. We preserved
  research artifacts, logs, code snapshots, checkpoints, env specs, and wandb
  cache; not login secrets.
- Track A B200 evidence is locally preserved and verified: B200 `/tmp/p7_branch_b_*`
  ↔ `b200_backup_20260522_final/tmp_p7` has 494 files, path+size hash
  `c308d1a682560cf51136cdd1a018c50ce2e7b488f1a0d4620e31abf7de80cfd4`,
  and file-content aggregate hash
  `cca0586b77c36ee79532d0640f9a35b2f1056654ab2758f256ea2bc1f149a4ae`.
- Track A B200 `sim_scripts` snapshot is locally preserved and verified:
  53 non-pycache files, path+size hash
  `98563bbc3d27426351abd13272a88537009372b2c709b46d2a5021560c5ea23a`,
  file-content aggregate hash
  `fefe4c873c1e45ec4cb95226a2c1a0d53860e4eca926c93d3da1b9887c9ca83f`.
- Track B B200 outputs are locally preserved, but split across
  `b200_backup_20260522_final/outputs`, `b200_backup_20260521`, and
  `openvla_oft_b200_pulls`. Do not assume
  `b200_backup_20260522_final/outputs/openvla_oft_v6_b200` is complete; the
  complete OpenVLA full checkpoints live in `openvla_oft_b200_pulls`.
- Full verification details:
  `claudedocs/session_20260522_b200_retirement_track_a_b_backup_verified.md`
  and `b200_backup_20260522_final/README_BACKUP.md`.

## Current Truth

**Track A active line**: P7/Branch B normalized 3cm cube grasp primitive.

- **Urgent professor push/tap branch (2026-05-26)** is separate from Track A
  grasp. Added `sim_scripts/cube3cm_push_rollout_probe.py` md5
  `8d329b79106e7ca2c03fa91b7ac87170` and ran local IsaacLab 3cm cube push/tap
  scripted rollouts at 16, 1024, 5120, and 20480 trials. The 20480 run lives in
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/`.
  Runtime stdout md5 `2aad344f08f95c880e43bc0d7f655998`; summary md5
  `5c9278450b5531afb7b0ca2a1fed46ee`; per-env CSV md5
  `4c2864301bea8e2ae798a8f77adf23ab`; audit md5
  `3e0096ba54e7cc0ec0e55b1b26a50b8e`. Runtime line 20 confirms no grasp,
  no attach/object posewrite, no training, no dataset; line 21 confirms 6D
  normalized joint-delta action semantics. Line 42 reports
  `total_trials=20480`, `ik_ok_rate=1.0000`, `disp_xy_mean_m=0.031809`,
  `disp_xy_p95_m=0.089702`, `moved_5mm_rate=0.8774`,
  `push_positive_1mm_rate=0.9086`, zero action saturation, zero grasp/attach/
  posewrite. Audit lines 5-21 show important caveats: low-motion rate
  `0.073340`, direction asymmetry, and high-speed/outlier impacts up to
  `0.521036748m` displacement and `4.549609073m/s`.
- **Professor cube-push learned-policy follow-up (2026-05-26)**: added a separate
  no-attach cube-push RL env/entry/eval path, not Track A grasp. Current
  `roarm_rl/roarm_cube_push_env.py` md5 `34254ac2fc3ede7a7844bd434fc9781d`;
  `train_cube_push_ppo.py` md5 `6dd733710ae3ec1e69cf8ad9e944948b`;
  `eval_cube_push_policy.py` md5 `bec3214d391862e4196d221775f4477d`.
  Static audit line 37 PASS and eval override check line 22 PASS. V4
  action-smoothed/velocity-limited policy improved frozen 1k impact from the
  speed-guard baseline but still had impact `0.245576787`
  (`ppo_smooth_limit_model49_eval1024_audit.out:3-17`). V5 scripted-teacher
  warm-start improved training logs, but teacher-off frozen 1k eval regressed to
  impact `0.257686676` and clean success `0.095168375`
  (`ppo_teacher_warmstart_model49_eval1024_audit.out:3-17`). V6 policy-only
  contact-speed curriculum reduced frozen 1k impact to `0.153782895`, but clean
  success was only `0.110197368`, so verdict remains
  `CONTACT_SPEED_MODEL49_EVAL_IMPROVED_BUT_NO_10K`
  (`ppo_contact_speed_model49_eval1024_audit.out:3-18`). Teacher-on diagnostic
  was also weak/unsafe: impact `0.162448980`, clean success `0.067755102`,
  verdict `TEACHER_ON_DIAGNOSTIC_UNSAFE_OR_WEAK_NOT_LEARNED_NO_10K`
  (`ppo_contact_speed_teacher_on_eval1024_audit.out:3-18`). Therefore do **not**
  run learned-policy 10k/100k scaling yet. The professor's "known endpoint
  means use IK near the cube first" request should now be pursued more
  literally with an **IsaacLab built-in DifferentialIKController** cube-push
  probe: provide end-effector/TCP targets near the cube, let IsaacLab compute
  joint targets from the live Jacobian, then audit physics push/tap results.
  The existing RoArm `ik_dls` path is valid IK evidence but is not the same as
  IsaacLab built-in Differential IK. Do this before any new learned-policy
  10k/100k scaling or Track A runtime.
- **Professor cube-push IsaacLab Differential IK follow-up (2026-05-27)**:
  added `sim_scripts/cube3cm_push_diffik_probe.py` md5
  `cbb2176a80ed2a2c55552d0d98bc9ab9`,
  `cube3cm_push_diffik_audit.py` md5
  `5ed85775e31f805f4d43885a1de80246`, and
  `cube3cm_push_diffik_posthoc.py` md5
  `6bfc8ea3eac942d0af4c8fc852738f0e`. The short 16-env smoke was a useful
  negative: mechanism PASS but low-motion `1.000000000` and final TCP error
  `0.161282191m` (`diffik_probe_smoke16_seed777_audit.out:1-6`). With longer
  reach/horizon, 16-env reached controlled `0.937500000`, low-motion
  `0.062500000`, impact `0` (`diffik_probe_reach16_seed778_audit.out:1-6`).
  Frozen 1024 eval then ran headless with IsaacLab built-in
  `DifferentialIKController`, no RoArm-local IK control loop, no grasp, no
  attach/object posewrite, no training, and no dataset. Audit lines 1-6 report
  mechanism PASS, controlled `0.892578125`, impact `0.023437500`,
  low-motion `0.136718750`, `disp_xy_mean_m=0.034856980`, max speed
  `1.931515932m/s`, and final TCP error mean `0.028779610m`.
  Posthoc line 7 identifies weak direction `(1, 0)` with controlled
  `0.633333333`; line 8 identifies worst initial-position grid `(1, 1)` by
  low+impact. This is an IsaacLab scripted Differential IK physics result, not
  PPO/VLA learning and not Track A grasp success.
- **Professor cube-push Differential IK trajectory v2 (2026-05-27)**: updated
  `sim_scripts/cube3cm_push_diffik_probe.py` with default-preserving
  `--trajectory_variant v2` for the weak `(1, 0)` direction: closer precontact,
  lower TCP target height, shorter push-through, longer approach/push horizon,
  and smaller per-step DiffIK joint cap. Smoke16 seed780 exited 0 and audit
  PASS, but had `v2_posx_env_count=0`, so it was only a mechanism smoke.
  Reach16 seed779 included 6 `(1,0)` envs and audit lines 1-6 PASS with
  controlled `1.000000000`, impact `0`, low-motion `0.062500000`. Frozen 1024
  seed779 then exited 0 and audit lines 1-6 PASS with controlled
  `0.932617188`, impact `0.038085938`, low-motion `0.051757812`, success marker
  `0.580078125`, final TCP error `0.024324538`, and clip rate `0.666682201`.
  Compare-to-v1 lines 2-4 show the mixed result: overall controlled/low/final
  TCP improved, `(1,0)` controlled improved `0.633333333 -> 0.785185185` and
  low-motion improved `0.274074074 -> 0.085185185`, but `(1,0)` impact worsened
  `0.088888889 -> 0.144444444` and success marker fell. Grid `(1,1)` improved
  low-motion `0.304687500 -> 0.023437500`, but gained nonzero impact
  `0 -> 0.031250000`. Verdict: useful scripted Differential IK physics
  evidence, not learned policy, not Track A grasp, not dataset/teacher-ready.
- **Professor cube-push Differential IK trajectory v3 (2026-05-27)**: current
  `sim_scripts/cube3cm_push_diffik_probe.py` md5
  `dc6ca5a222f0bd9437d5f83bf5449729` adds default-preserving
  `--trajectory_variant v3` for `(1,0)`: lower contact height, shorter
  push-through, longer/slower pos-x horizon, and lower pos-x joint-step cap.
  Smoke16 seed780 audit lines 1-6 PASS but had `v3_posx_env_count=0`, so it was
  mechanism-only. Reach16 seed779 included 6 `(1,0)` envs and audit lines 1-6
  PASS with controlled `1.000000000`, impact `0`, low-motion `0.062500000`.
  Frozen 1024 seed779 audit lines 1-6 PASS with controlled `0.969726562`,
  impact `0.004882812`, low-motion `0.035156250`, success marker
  `0.604492188`, final TCP error `0.023551417`, and zero posewrite. Posthoc
  line 6 reports `(1,0)` controlled `0.929629630`, impact `0.014814815`,
  low-motion `0.088888889`. Compare lines 2-3 show v3 improves over v2 on
  overall impact `0.038085938 -> 0.004882812` and `(1,0)` impact
  `0.144444444 -> 0.014814815`; line 9 shows remaining `(1,0)` impacts are
  still tip-angle outliers only. Critical caveat: `(1,0)` success marker dropped
  to `0.314814815` and clip is `1.000000000`, so v3 is a strong scripted
  physics/statistics candidate but not automatically a dataset teacher.
- **Professor cube-push Differential IK v3 10,240-trial audit (2026-05-27)**:
  ran local IsaacLab `num_envs=1024`, `episodes=10`, seed779, total 10,240
  scripted DiffIK trials. Stdout lines 20-21 confirm built-in
  `DifferentialIKController`, no RoArm-local IK loop, no training/dataset/grasp/
  posewrite, `trajectory_variant=v3`. Audit lines 1-6 PASS with
  `csv_rows=10240`, controlled `0.943164062`, impact `0.007519531`,
  low-motion `0.042480469`, success marker `0.594824219`, final TCP error
  `0.023529604`, and zero posewrite. Posthoc line 6: `(1,0)` n=2566,
  controlled `0.874512860`, impact `0.012860483`, low-motion `0.122759158`,
  success marker `0.296570538`. Compare-to-1024 lines 2-3: overall impact stays
  below 1%, `(1,0)` impact stays below 2%, but `(1,0)` low-motion worsens and
  success remains weak. Verdict: professor-style 10k scripted push/tap robustness
  evidence PASS; still not learned policy, not Track A grasp, not dataset-ready.
- **Professor cube-push Differential IK v3 visualization replay (2026-05-27)**:
  generated a professor-facing MP4 from a captured v3 trace; this is a replay
  artifact for visualization, not a new training run, dataset generation, or
  fresh physics recomputation. Current `cube3cm_push_diffik_probe.py` md5 is
  `1e39836eb02a22c12e084a4279e6b4e7`; replay renderer
  `sim_scripts/cube3cm_push_diffik_render_trace.py` md5 is
  `2adb116ae2c441420873a8384d3a7b17`. Selected single-env case is
  `diffik_probe_v3_reach16_seed779.csv` line 5: env_id `3`, direction `(1,0)`,
  start `(x,y)=(0.353590250,-0.073313951)m`, displacement
  `0.036002159m`, controlled `1`, impact `0`, low-motion `0`, success marker
  `1`. Trace summary lines 46-49 show trace CSV, env_id `3`, 145 trace frames,
  and `training=false`. Render stdout line 447 confirms `frames=145`, MP4 path,
  `training=NO`, `dataset_generation=NO`, `physics_recomputed=NO`. Render
  summary lines 19-27 confirm `30fps`, `1280x720`, 145 written frames, output
  path, `physics_recomputed=false`, `training=false`. A corrected four-direction
  parallel replay now uses env_id `[0,3,4,7]` for directions `(0,-1)`, `(1,0)`,
  `(0,1)`, `(-1,0)` respectively; render stdout line 447 confirms
  `frames=145 env_count=4`, and render summary lines 12-19/91-112 confirm white
  background, black actual RoArm URDF STL mesh, gray table, pink cube, `30fps`,
  145 frames, 2x2 layout, `physics_recomputed=false`, and `training=false`.
  Earlier black FK-proxy render artifacts were rejected/superseded because they
  were not actual RoArm geometry.
- **Professor cube-push DiffIK dataset v2 + BC learned rollout (2026-05-28)**:
  added all-env trace capture and auditable dataset/BC tooling. Current md5s:
  `cube3cm_push_diffik_probe.py` `2342f31701e91af57d0f311db4eeec87`,
  dataset builder `a0d18ef1b34415c96d036ba42952e37e`, dataset audit
  `f677ea14809a0a2091bc13c5254d4fae`, BC train
  `df03abb00188cfd9b644b0ef410a0e14`, BC rollout
  `a56e9a96feaad196fef6e1081c0116ec`, rollout audit
  `121721561bcde8df141f56207dcab14d`. Source v3.1 1024 trace seed779 audit
  lines 1-6: controlled `0.964843750`, impact `0.004882812`, low-motion
  `0.034179688`, success `0.611328125`; posthoc line 6 still marks `(1,0)`
  weak. Dataset build lines 1-6 selected 320 final-success teacher trajectories
  into 46,400 rows, balanced 80 per direction, split `224/48/48`, candidate YES.
  Dataset audit lines 1-7 PASS full state-action dataset with schema/finite/split/
  direction/mechanism OK and final controlled/success `1.0`, impact/low `0.0`.
  BC train lines 1-4 PASS checkpoint with test MSE `0.007494668` and mean test
  MAE `0.000745819rad`. Learned BC rollout 1024 seed883 audit lines 1-6 PASS
	  without DiffIK: controlled `0.945312500`, impact `0.012695312`, low-motion
	  `0.026367188`, success `0.648437500`, posewrite 0. Caveat remains: `(1,0)`
	  line 5 success `0.453488372`, impact `0.038759690`. This is professor-branch
	  teacher-filtered state-action BC, not Track A grasp, not PPO/RL/VLA, not image
	  dataset, and not 10k/100k learned-policy robustness.
- **Professor cube-push v3.2 teacher sweep + bucket-balanced BC v2 (2026-05-28)**:
  added `cube3cm_push_diffik_bucket_audit.py` and extended the dataset builder/audit
  for `direction_posx_bucket` selection. v3.2 teacher parameter sweeps were useful
  negatives: t270/p036 raised `(1,0)` success but also raised `(1,0)` impact, while
  t270/p030 and t257/p034 were not robust across seed790/seed791. Dataset v3 was
  built without new physics from the v3.1 all-env trace: build lines 1-6 selected
  180 final-success teacher trajectories, 26,100 rows, 45 per direction, and
  `(1,0)` low/mid/high-x `15/15/15`. Dataset audit lines 1-8 PASS, including
  `balance_mode=direction_posx_bucket`, bucket OK, split bucket OK. BC v2_bucket
  train lines 1-4 PASS with test MSE `0.022629632`, mean test MAE
  `0.001227073rad`. 1024 learned rollout seed883 with default clip PASS overall:
  controlled `0.961914062`, impact `0.015625000`, low-motion `0.016601562`,
  success `0.679687500`; `(1,0)` success improved to `0.527131783`, but bucket
  audit fails because low_x impact is `0.068493151` and high_x impact is
  `0.061855670`. Clip `0.035` improved overall success to `0.689453125` but still
  failed low_x impact. Verdict: learned BC quality improved, but per-bucket safety
  gate blocks PPO/RL scale-up.
- **Professor cube-push safety-aware BC v3 gate (2026-05-28)**: extended
  `cube3cm_push_diffik_train_bc.py` with optional safety-weighted BC loss and
  `cube3cm_push_bc_policy_rollout.py` with auditable per-bucket action scaling /
  smoothing; fixed bucket-audit learned-policy reporting. Current md5s:
  train `cbe7c2e8d44fe7a92cb2ba69f29b518a`, rollout
  `aa0b5ef06db903058724a71f61225f0b`, bucket audit
  `c69ff72a4a31228868169016ab2f2d08`. Safety BC train lines 1-5 PASS with test
  MSE `0.018879525`, mean MAE `0.001225158rad`; checkpoint md5
  `03b159809ddca64aad6d6449b7f44876`. 1024 frozen learned rollout seed883
  lines 1-6 PASS overall: controlled `0.953125000`, impact `0.004882812`,
  low-motion `0.030273438`, success `0.662109375`; bucket lines 7-10 PASS with
  low_x impact `0.041095890`, high_x impact `0`. Cross-seed seed884 lines 1-6
  PASS overall: controlled `0.943359375`, impact `0.010742188`, low-motion
  `0.024414062`, success `0.662109375`; bucket lines 7-10 PASS with low_x impact
  `0.035714286`, high_x impact `0`. Critical caveat: low_x low-motion is still
  high (`0.315068493` seed883, `0.261904762` seed884), so this unblocks only the
  next small safety-aware learned-policy gate, not 10k/100k robustness or PPO
  scale-up without explicit approval.
- **Professor cube-push safety-aware PPO warm-start smoke (2026-05-28)**: after
  explicit approval, added default-off BC teacher/imitation warm-start support to
  `roarm_rl/roarm_cube_push_env.py` md5 `9806c1fcfb4666355f825418da5b7d75`,
  `train_cube_push_ppo.py` md5 `5466a9c9d40a7f09d397fbffa7cdb878`,
  `eval_cube_push_policy.py` md5 `fa68ee654c969aff7938867894acf125`, optional
  PPO bucket audit support md5 `62f74ce38c9a44f0f0790e00559f634a`, and new
  `cube3cm_push_ppo_rollout_audit.py` md5 `b92260c8f0986c1b6bfe233fcf417d01`.
  Prior docs say system Python failed with missing `gymnasium`, but current local
  stderr citation is a mismatch; verified valid runtime was `conda run -n isaaclab`
  on `cuda:0`. Smoke12 training seed885 completed 73,728 timesteps and wrote
  `model_11.pt` md5 `c9f945a4d1eacd817d4733e7d9b7e48e`; training stdout lines
  47-78 confirm BC teacher blend `0.35`, imitation reward `0.30`, no attach/no
  dataset, and `cuda:0`. Teacher-off frozen 1024 eval seed886 is a mechanism PASS
  but performance FAIL: audit lines 1-5 show controlled `0.470703125`, impact
  `0.087890625`, low-motion `0.344726562`, success `0.078125000`; bucket lines
  1-10 fail with `(-1,0)` impact `0.230483271` and `(1,0)` success
  `0.070833333`. Teacher-on short/safety-limited diagnostic is also weak
  (success `0.050781250`, impact `0.097656250`). Direct-like teacher-on diagnostic
  with 6s horizon/home reset/relaxed action loop partially recovers controlled
  `0.792968750`, impact `0.054687500`, success `0.417968750` and passes the loose
	  posx bucket screen, proving the checkpoint is not dead but the PPO action-loop
	  warm-start bridge/curriculum is mismatched. Do not call PPO `model_11.pt` a
	  successful learned policy; do not run PPO 10k/100k scale-up from it.
- **Professor cube-push BC teacher bridge redesign diagnostics (2026-05-29)**:
  added default-preserving bridge controls: `joint_delta_reference` and
  `bc_teacher_phase_timing`. Current md5s: env
  `a0483108ef0fc8ab2f27a58b6edd8c13`, train
  `7032616ded5617b546149227f4c0d110`, eval
  `b10fad43cfd3b0ca543390ad6011135f`. Static checks passed. Teacher-on
  direct-step/joint-pos 128-env seed889 recovered overall but failed low_x bucket:
  controlled `0.984375000`, impact `0.007812500`, success `0.601562500`, low_x
  success `0.133333333`. Low_x scale `1.0` seed890 passed the small teacher-on
  bridge screen: controlled `0.992187500`, impact `0.007812500`, low-motion
  `0.007812500`, success `0.765625000`, and bucket PASS with low_x success
  `0.538461538`. But this is teacher-on only. Existing `model_11.pt` under the
  new action loop is teacher-off zero-motion at 128 (controlled `0`, low-motion
  `1`, success `0`). A tiny 128-env smoke8 PPO distillation run wrote `model_7.pt`
  md5 `5ed5ac34dc624ac8c660d9176378b357`, but imitation MSE stayed around
  `0.56-0.59` and teacher-off 128 still failed with controlled `0`, low-motion
  `1`, success `0`. Verdict: bridge mismatch is partly fixed, but PPO actor
  learning is still not solved. Do **not** run teacher-off 1024, PPO scale-up,
  10k/100k, dataset generation, or Track A runtime from these PPO checkpoints.
  Next valid step is a true supervised actor/normalized-action distillation or
  stronger actor initialization before PPO.
- **Professor cube-push rsl_rl actor distillation gate (2026-05-29)**: added
  `roarm_rl/distill_cube_push_actor.py`, which collects the BC teacher's
  normalized joint-delta actions through the same direct-step/joint-pos loop and
  writes a normal rsl_rl checkpoint. Distillation seed894 used 128 envs x 600
  steps = 76,800 samples from `model_7.pt`; checkpoint
  `model_actor_distill.pt` md5 `57811cfb054ca7ac39b134d1d97cd543`.
  Supervised fit improved val MSE `0.169735238 -> 0.000794161`, but the only
  allowed teacher-off 128 audit seed895 still failed: controlled `0.101562500`,
  impact `0`, low-motion `0.929687500`, success `0.031250000`; bucket audit
  failed with low_x/mid_x success `0` and high_x success `0.076923077`. Verdict:
  actor no longer outputs pure zero, but closed-loop push is still effectively
  low-motion. Do **not** run teacher-off 1024, PPO scale-up, 10k/100k learned
  robustness, dataset generation, or Track A runtime from this checkpoint.
  Next valid step is a closed-loop/action-target analysis of why low-MSE one-step
  normalized action imitation collapses to low-motion, or a stronger rollout-
  aware actor initialization; then repeat only teacher-off 128.
- **Professor cube-push waypoint actor 128 gate (2026-05-29)**: trace showed
  the first actor-distilled checkpoint failed because actor-visited states diverged
  from the teacher and because the actor observation did not expose the teacher's
  phase/moving TCP waypoint. Added default-off `policy_obs_target_mode` with
  `bc_teacher_tcp_target`, plus trace/analysis tools. `model_actor_waypoint.pt`
  improved teacher-off 128 seed901 to controlled `0.679687500`, impact `0`,
  low-motion `0.242187500`, success `0.273437500`, but bucket failed low_x.
  Waypoint + on-policy DAgger1 improved seed904 to controlled `0.968750000`,
  impact `0`, success `0.617187500`, but bucket still failed low_x success
  `0.117647059`. The low_x-scale `1.3` on-policy checkpoint
  `model_actor_waypoint_lowx130.pt` md5 `606d19fff713e7468d395af4a027d08a`
  passed the first teacher-off 128 first-episode gate seed906: audit controlled
  `0.937500000`, impact `0`, low-motion `0.093750000`, success `0.546875000`;
  bucket audit PASS with `(1,0)` low_x success `0.571428571`, mid_x success
  `0`, high_x success `0.428571429`, and zero impact. After explicit approval,
  the same checkpoint passed three teacher-off 1024 first-episode overall/
  per-bucket gates with no teacher action blend: seed907 controlled
  `0.924804688`, impact `0`, low-motion `0.109375000`, success `0.511718750`;
  seed908 controlled `0.925781250`, impact `0`, low-motion `0.114257812`,
  success `0.523437500`; seed909 controlled `0.920898438`, impact `0`,
  low-motion `0.125976562`, success `0.506835938`. This is now a 3x1024
  teacher-off learned-policy gate PASS using waypoint observations, but still
  not 10k/100k robustness, not dataset readiness, and not Track A/PPO/VLA
  final success. Follow-up candidate screen showed high_x actor scale `1.0`
  fails 128 bucket, gain `0.045` is a 3x1024 PASS non-canonical deployment
  candidate with only modest/mixed improvement, and gain `0.050` is mixed
  after a seed907 pilot; canonical remains `model_actor_waypoint_lowx130.pt`
  with gain `0.040`. After explicit approval, a single-stage 10240-env audit
  failed during IsaacLab env creation before policy rollout, so it is not a
  policy failure. The fallback 10x1024 sharded first-episode teacher-off audit
  on seeds 912-921 produced 10,240 rows and PASSed mechanism and posx bucket
  gates: controlled `0.927148437`, impact `0.000097656`, low-motion
	  `0.106054687`, success `0.524902344`; low_x success `0.406947891`, mid_x
	  success `0.183497537`, high_x success `0.213625866`. This supports a
	  sharded 10k teacher-off robust learned-policy gate PASS, but still not
	  dataset readiness, not Track A evidence, and not PPO/RL/VLA final success.
- **Professor cube-push metric reframe and target-extension probe (2026-06-02)**:
  code/log review shows the old `success_marker` is a strict task marker, not the
  only professor-relevant push/tap metric. In code, the cube is fixed at
  `CUBE_SIZE_M=0.030` and mass `0.020kg`; env success requires controlled push,
  no impact, `disp_along >= 0.030m`, target-distance tolerance, and speed cap.
  The sharded 10k threshold analysis shows the canonical actor is much stronger
  at stable smaller pushes than the 3cm marker alone suggests: for direction
  `(1,0)`, `disp_ge_5mm=0.906199678`, `disp_ge_10mm=0.842592593`,
  `disp_ge_20mm=0.770531401`, but `disp_ge_30mm=0.266505636`. Posx buckets show
  mid/high are near-perfect at 10mm and strong at 20mm, then fall sharply at
  30mm. Therefore report professor push/tap evidence as `1/5/10/20/30mm`,
  `disp/object_size`, controlled, no-impact, and low-motion, not as 3cm success
  alone. Added default-preserving code knobs for weighted actor distillation and
  mid/high BC teacher push-through overrides. The weighted mid/high actor
  candidate passed one-step fit but failed the 128 posx bucket screen
  (`low_x=0.083333333`, `mid_x=0.090909091`, `high_x=0.833333333`) and is
  rejected. The target-extension probe on the canonical checkpoint improved
  same-seed mid_x/high_x (`mid_x=0.272727273`, `high_x=1.000000000`) but still
  failed due low_x (`0.166666667`), so do not scale it. Before changing cube
  size, define whether `10*10*10` means mm or cm and whether mass is measured,
  density-preserving, or deliberately fixed as a diagnostic. A 2026-06-04
  local-only update to `sim_scripts/cube3cm_push_diffik_bucket_audit.py` now emits
  the hierarchical report directly. The generated sharded-10k report logs
  `cube_size_m=0.030000`, `cube_mass_kg=0.020000`, density `740.741kg/m^3`,
  no-impact, `disp/object_size`, and displacement-only 1/5/10/20/30mm columns
  in
  `model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_hierarchical_bucket.out`.
  Overall `disp/object_size_mean=0.775020338`; forward `(1,0)` reports
  `disp/object_size_mean=0.743007598`, 5mm `0.906199678`, 10mm `0.842592593`,
  20mm `0.770531401`, and 30mm `0.266505636`.
- **Professor cube10cm DiffIK gate prep (2026-06-04)**: the next professor
  request is interpreted as a separate 10cm/0.72kg object push/tap diagnostic,
  not as Track A and not as a requirement to push a full 10cm object length.
  The 1cm gate is now explicit. The 128 v1 gate, fixed16 side-center gate, and
  settled-start fixed16 rerun all failed with `diffik_clip_rate_mean=1.0` and
  low/no motion. The settled-start patch fixed the reset-buffer z mismatch but
  not the rollout.
  A follow-up 4-env trace diagnostic added per-step link5/TCP targets, raw and
  clipped deltas, robot targets, and actuator follow errors. cap `0.120` and
  drive-boost controls prove the 10cm/0.72kg object can move, but they overshoot
  by 5-9cm and are not teacher-ready. Code review also found the old through
  target was far-face/cross-object (`cube + half + push_through`), so
  `--through_target_mode near_face` was added. Near-face default/long-approach
  runs still failed to contact (`disp_ge_gate_rate=0.0`, final TCP error about
  `0.061m`, min TCP-cube about `0.083m`), while near-face drive boost passed the
  gate but moved about `0.050m`. Next valid work is one tiny controlled near-face
  contact controller with actuator/step scheduling and displacement/contact stop;
  not randomization, 128/1024/10k, dataset generation, PPO/RL, VLA, or Track A.
- Track A goal: first make the sim/Isaac Lab contact primitive reliable, then
  move toward broad sim/lab dataset collection and learning.
- Dataset generation and training are blocked until close_26 proxy audit PASS,
  then hold-lift PASS, then small pilot dataset/replay PASS.
- v4/v5/v6/v7 rigid/offset variants, soft-contact material-only, virtual
  compression+damping, target-guarded v1 through v7, and the first pre-fix v8
  runtime have all failed close_26 posthoc audit. None is grasp success.
- This is an Isaac proxy/contact primitive failure, not proof that the real robot
  cannot grasp the foam cube.
- The RL-to-expert-to-rollout-to-demo pipeline is valid only after a Track A
  Stage 0 no-attach contact gate exists. Existing default Pick/Stack PPO envs use
  kinematic attach / posewrite and must not be used as Track A no-attach expert
  evidence.
- Latest Track A B200 v6 `close_26` runtime is FAIL, not grasp success.
- 2026-05-26 step-plan update: the professor-style
  RL→expert→rollout→dataset→learning pipeline is still the right high-level
  path, but only after Stage 0 no-attach contact primitive PASS. Added local
  static design artifact
  `sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py`
  md5 `14a462526945f3c5bca1c5e8c3e13525`; it reports that v6 pre-freeze recovery
  rows increased target error by `0.000626m` and support gap by `0.000319m`.
- 2026-05-26 v7 active-recovery code/readiness update: added default-off v7
  finite-difference TCP recovery, matching audit support, and readiness negative
  controls. Local static checks pass, including synthetic v7 PASS, v7 no-active
  recovery rejection, archived v6 log rejection as v7, and readiness
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`. No runtime was run.
- 2026-05-26 approved local v7 runtime attempt was blocked by infrastructure, not
  physics: IsaacLab metadata emitted, but local CUDA/NVIDIA access failed before
  environment creation and no close step/aggregate lines were produced. The
  immediate audit correctly failed. Preserved logs live in
  `claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/`.
- 2026-05-26 RunPod/Codex continuation setup: Claude had RunPod MCP configured,
  but Codex did not. Added `[mcp_servers.runpod]` to
  `/home/cgxr/.codex/config.toml` from Claude's RunPod MCP config, with the
  `RUNPOD_API_KEY` value not printed. Backup:
  `/home/cgxr/.codex/config.toml.bak_runpod_20260526` md5
  `1ef4acf6f1c92a64b9bbd79a2e35b7e7`. Same-session `tool_search` still did not
  expose `mcp__runpod__...`, so each new Codex session must verify loaded tools
  before using RunPod MCP. A later Codex session did expose `mcp__runpod__...`
  and `list_pods` returned no GPU pods.
- 2026-05-26 post-reboot local CUDA update: user rebooted the local Ubuntu PC.
  Boot time now `2026-05-26 14:08`; host NVIDIA kernel/userspace now match at
  `580.159.03`. Host `nvidia-smi` and `conda run -n isaaclab` CUDA checks pass
  only when run outside the default Codex sandbox. The default Codex sandbox
  hides `/dev/nvidia*`, so sandboxed `nvidia-smi` still fails; this is not a host
  CUDA failure. v7 readiness still reports
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`. The old `/tmp` RunPod overlay is
  gone after reboot; recreate it if RunPod is needed. Top local backup USD md5
  remains `4497024d25abab11de5c50e144124553`.
- 2026-05-26 post-reboot v7 runtime/audit result: exactly one local
  close_26-only v7 active-recovery runtime ran with escalated Codex execution and
  immediate audit. This is a real physics/audit FAIL, not a CUDA infrastructure
  block. Logs:
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/`.
  Runtime stdout md5 `621d00b9d157b4e70178c28f94ca4c7f`; audit stdout md5
  `406b96557d94418f16273e517ec4d69b`. Runtime lines 389-391 show v7 active
  recovery did trigger (3 writes, 0 IK failures, negative counter-gap deltas),
  but runtime line 392 is first support hard-freeze (`counter_gap=0.002048m >
  0.002m`, target `0.002962m` still inside gate), line 393 is first target+
  support breach (`target=0.003059m`, gap `0.002104m`), and line 424 aggregate
  has close_reached NO, 31 hard freezes, attach/posewrite 0, telemetry-only YES,
  success_claim NO. Audit line 19 fails close_reached; line 32 fails
  hard-freezes-zero; lines 54-56 fail hard-freeze/fixed-target/fixed-support;
  line 66 `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- 2026-05-26 v7 failure static analysis result: analyzer
  `sim_scripts/p7_branch_b_cube2cm_v7_failure_analyzer.py` md5
  `e13605f058cd1908ff3d863e8239fbc4`; output
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v7_failure_static_analysis.out`
  md5 `0fbf57f32473fa253ee1082b888bdcb1`. It parsed 45 close rows,
  found 3 v7 active rows, and classified audit mismatch NO, late trigger YES,
  candidate prediction mismatch YES, weak TCP follow YES, contact geometry
  suspect YES, and hard-safety lockout after active YES. Active followups
  predicted target/gap improvement but observed target and support gap worsening
  with negative TCP follow ratios.
- 2026-05-26 v8 observed-recovery static design result: static-only script
  `sim_scripts/p7_branch_b_cube2cm_target_guarded_v8_observed_recovery_static_design.py`
  md5 `56a382377b7fb0f0c6391bf59163af0d`; output
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_observed_recovery_static_design.out`
  md5 `c14e80ec5fc69c6e6e17925d61f81d0b`. Output lines 1-2 verify v7 runtime
  and v7 analysis md5s. Line 3 finds first projected reserve trigger at runtime
  line 386 / close step 9, line 4 shows v7 first active at runtime line 389 /
  step 12, and lines 5-7 show the projected reserve trigger was 3 steps before
  v7 active and 6 steps before first support breach. Lines 8-11 reject unchanged
  v7 by observed response/TCP follow; lines 21-28 define the v8 design contract
  and explicitly report `RUNTIME_READY=NO`, `STATIC_V8_DESIGN_DONE=YES`.
- 2026-05-26 pre-runtime v8 runtime-candidate static readiness result:
  default-off v8 candidate and matching audit/readiness support were implemented;
  at that point no v8 physics runtime had run. Runtime probe md5
  `7e6dfc35bbfeacb5d1689f2f175e5120`;
  audit md5 `8dbf621c983ec03f46e5d52843781fda`; readiness md5
  `a31ced20b754a4a42058349525d1a435`. Readiness output
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_runtime_candidate_readiness.out`
  md5 `6a2a62808451175b65e5d522b695b8b6`: lines 1-2 local/static/no forbidden
  mechanisms, lines 3-4 wiring/metadata guard PASS, lines 5-13 negative controls
  PASS, line 14 synthetic v8 PASS accepted, line 16 future command uses preserved
  local backup USD path, and line 19 `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
  A separate v8 audit of the preserved post-reboot v7 runtime
  `v8_rejects_post_reboot_v7_audit.out` md5
  `cb082918d92a0f95b585ade432c34730` correctly fails: lines 5/14/15 reject
  metadata, lines 20/30 reject close/hard-freeze success, and lines 53/55/57/58
  reject missing v8 reserve trigger, observed worsening, non-positive TCP follow,
  and missing counter-contact modeling.
- 2026-05-26 approved local v8 runtime/audit result: exactly one local
  close_26-only v8 observed-recovery runtime ran with escalated Codex
  GPU/Isaac execution and immediate audit. It FAILED; this is not grasp success.
  Logs:
  `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/`.
  Runtime stdout md5 `74095570c2d6a60abdf522c2413735db`; audit stdout md5
  `7cd38eddb1dc9c925b01948cbc5cb416`. Audit line 20 fails
  `close_reached`; lines 26/35-36 fail because virtual damping was inactive and
  wrote zero velocity updates; lines 45-49/53 show no recovery/trigger; line 60
  reports `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`. Static failure analysis
  output md5 `7e81773b91d39658a3ec5c6eaf878f0c`: first hard freeze is runtime
  line 384 / step 7, target/support breach is runtime line 392 / step 15, and
  v8 trigger/recovery were never seen.
- 2026-05-26 post-fail v8 code fix: pre-fix v8 inherited target-guarded close
  activation but missed virtual damping activation. Added v8 to the
  `virtual_damping_active` mechanism and added a readiness regression check.
  Runtime probe md5 is now `acae0ca2e85a522dd4ac8fb583cb8fb8`; readiness md5
  is now `dc2bdaa8d882f12b5cc901a677caccc0`. Post-fix readiness output
  `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/readiness_after_v8_damping_fix.out`
  md5 `b652520a81792bf12373ff742cdba6b5`, lines 5 and 20 report the new
  damping-inheritance check PASS and `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
  No second/post-fix runtime has run.

**Track B/OpenVLA** is separate. Do not use Track B training or eval status as
Track A contact success evidence. Latest Track B P3 result remains: best deploy
ckpt = step 7500; steps 10000+ are collapsed and must not be deployed.
Track B data/continuation assets are backed up locally; P5 real robot deploy is
still pending local reboot/CUDA verification and user approval for robot motion.

**Track B Cube Task Pivot (2026-05-26, user-confirmed)**: sponge → **cube 3×3×3cm
× 5개 → 3+2 pyramid stacking** (L1=3, L2=2) 신규 task. Camera = Azure Kinect 고정
v6 동일 viewpoint. Sponge HARD RULES #19/#20/#24 자동 SUPERSEDED (HARD RULE #18
사용자 명시 정정 우선). Track A 직접 비교 → **sim demo 증강으로 재포지셔닝**
(Track A close_26 PASS 후 cube stacking sim demos co-training).

Hyperparam 갱신 (P3 7500→10000 collapse 회피): per_gpu_batch=8 + grad_accum=4 →
**effective batch=32** (vanilla OpenVLA-OFT LoRA 최소 권장치, 우리 P2 effective
8은 1/4였음). LR `5e-4` → **`2.5e-4`** (½, linear scaling 보수적). grad_clip_norm
=1.0, warmup 1K step, cosine. RunPod **A100 80GB**, 30K step ~8h ~$13.

데이터 신규 수집: **250ep (200 cube stacking + 50 cube pick), 일 50ep × 5일**,
ep당 ~400fr → 80K frames (v6 6942fr 대비 11.5×, task horizon 10× 근거).

7-phase plan: P0 cube+gripper calib (0.5일) → P1 데이터 수집 (4일, mid γ-gate)
→ P2 LeRobot 변환 (0.5일) → P3 RunPod 학습 (1일) → P4 12-ckpt offline eval rank
(0.5일) → P5 real multi-position deploy (0.5일) → P6 Track A close_26 PASS 후 sim
demo co-train (별개 trace) → P7 비교 paper (1일). 상세:
`claudedocs/session_20260526_track_b_cube_task_pivot_plan.md`, ledger row 123.
v6 sponge ckpt 7500 deploy (P5 pending CUDA reboot)는 별개 보존, cube pivot과 무관.

**Track B Cube P0 EXECUTION (2026-05-26, P0.1+P0.2 done, blocked at HW disconnect)**:
스크립트 3개 신규 — `safety_p0_guards.py`(G1-G10+DryRunArm, move_joints speed>200
ValueError), `trajectory_p0_gripper_sweep.py`(Gauge+stall auto-stop+cube release),
`hw_p0_sanity.py`(P0.1+pose_ctrl smoke default-OFF). 전부 py_compile+dry-run OK.
**포트 serial 검증**: Leader=USB0/`7842…`, Follower=USB1/`ee7a…` (5/22 기록 일치, 재연결 시 by-id 재검증 필수).
**P0.1 PASS**: max_diff 1.93°≤3° / Kinect cube 224-resize ~6-11px (P1 전 가시성 flag).
**P0.2 anchor lock**: 30mm cube→moving-jaw state **~37.88°** 정지, hold Y, **grip cmd
target ~28** (cmd 0-5 금지=서보 stall, P1 분포 "0-5°" 폐기). pad 없음 시사.
**fixed-jaw URDF 확증**: gripper movable joint `link5_to_gripper_link` 1개뿐 → cube
center≠TCP 중심선 → P0.3/0.4/L-F는 lateral offset 반영 필수.
**세션말 두 팔 USB 동시 단선** (`/dev/ttyUSB*`+by-id 소실, lsusb CP210x 둘 다 없음,
Kinect 정상) → pose_ctrl smoke/P0.3부터 재연결 대기. 사용자: 팔 전원/USB허브 확인.
다음: 재연결→by-id 재검증→`hw_p0_sanity.py --pose-ctrl-smoke`(P0.3 IK 게이팅)→
P0.3 fixed-jaw 반영→P0.4 grasp z→P0.7 L-F 5/5→P1. 상세:
`claudedocs/session_20260526_track_b_cube_p0_execution.md`,
`~/.claude/.../memory/tech_cube_grasp_anchors.md`(P0.1/0.2 실측 반영).

**P0 plan 4-agent 교차검증 + 사용자 결정 5건 확정 (2026-05-26)**: 코드/데이터
변경 없음, 메모리 docs 4개만. P0 순서 = P0.0 전제(pad/mass/matte) → P0.1 HW
sanity(max_diff≤3° + cube 가시성 + v6 viewpoint remount sanity + pose_ctrl smoke)
→ P0.2 jaw sweep(**Gauge 방식**: arm HOME 고정+수동 cube+gripper cmd만, object-
agnostic curve) → P0.3 approach angle(scripted joint, FK z guard) → P0.4 grasp z
(**FK primary**, 후보 +8/+12/+15mm tipping 비교) → P0.5 stacking z(L-F 관찰만)
→ P0.6 pyramid jaw 간섭 → P0.7 gate(L-F single pick 5/5) → P0.8 lock-in. 결정
5건: ① cube 5개 보유 ② P0.2 Gauge(future task 재사용) ③ camera v6 유지(환경 동일
→재calib 불필요) ④ IK는 joint 직접+pose_ctrl 보조 ⑤ grasp z tipping 비교 + P0.5
경량화. 정정: cube top z=**+18mm world**(table -12.12+30), grasp z FK 실측
+9~15mm, wrist_pitch +14° shift. 다음: P0.0 hands-on → deploy-agent P0.1/P0.2
스크립트(safety_p0_guards.py G1-G10 + Gauge sweep). 상세:
`claudedocs/session_20260526_track_b_cube_p0_plan_crossvalidated.md`, ledger row
126, `~/.claude/.../memory/tech_cube_grasp_anchors.md`. sponge anchor SUPERSEDED.

**Track B P4 result — 2026-05-22 ~17:00 KST (deploy prep + offline + hw sanity
all PASS, real deploy pending CUDA reboot + openvla-7b 14GB download)**:

- Built `deploy_openvla_oft.py` 561 lines mirroring `deploy_smolvla.py` 4/9 Plan 3
  SUCCESS setup (INIT_POS [0,0,90,0,0,5] HOME, JOINT_SPEED_CAPS
  [500,500,500,300,300,300], gripper-only unlock pattern `arm.gripper_angle_ctrl(
  angle, speed=1000, acc=0)` directly after `joints_angle_ctrl`, Z_FLOOR=-130mm,
  DIST_MAX=420mm, Follower-only `--port /dev/ttyUSB0` blocked). Inference path
  replaces SmolVLA with OpenVLA-OFT (224×224 PIL RGB + language prompt, no state
  input, chunk (8,6) BOUNDS_Q99-denorm via `vla.predict_action`).
- Inline `L1RegressionActionHead` (deploy_openvla_oft.py:78-138) bypasses
  `prismatic.models.__init__` → `vlas` → `vla.materialize` → `dlimp` chain.
  See `claudedocs/DECISIONS.md` D086 for full rationale.
- Offline sanity 1+2+3 PASS (CPU only):
  1. Inline L1 head strict-load from B200 ckpt 7500 `action_head--7500_checkpoint.pt`
     after `module.` prefix strip: missing=0, unexpected=0, 134,328,326 params,
     forward (1,48,4096)→(1,8,6) OK.
  2. `dataset_statistics.json` key `roarm_v6_pick` q01/q99 for all 6 joints inside
     JOINT_LIMITS even at ±1.0 saturation.
  3. Script `ast.parse` PASS + 3 critical sub-imports OK.
- Hardware sanity 4+5 PASS:
  4. Kinect 720P NFOV_UNBINNED 1-frame capture (1280×720×3 BGR) →
     `logs/hw_sanity_20260522/kinect_sanity_frame.png`.
  5. Follower `/dev/ttyUSB1` (serial `ee7a06468e98ef1194edca63a8793231`, Leader
     USB0 serial `7842202ff8d9ef11b33f513dc8728757` per
     `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/tech_leader_follower_setup.md`)
     → torque ON → INIT_POS reached in 0.5s, max_diff=1.93°,
     FK pose x=353 y=2 z=204 mm (Z_FLOOR/DIST_MAX safe).
- Blockers for Step 6 real deploy:
  - CUDA driver mismatch `Failed to initialize NVML / NVML lib 580.159 / Error 804
    forward compatibility` → `torch.cuda.is_available()=False`. Fix = `sudo reboot`
    (no PC power-cycle needed).
  - `openvla/openvla-7b` HF cache 14 GB download in background at pinned revision
    `47a0ec7fc4ec123775a391911046cf33cf9ed83f`, ~2 GB / 14 GB at session end.
- `roarm` conda env additions: `peft 0.18.0` (`--no-deps`), `rich 15.0.0`,
  `timm 0.9.16` (HARD RULE #15 pin), prismatic editable from
  `/home/cgxr/Documents/Robotics/openvla-oft/`.
- Full detail: `claudedocs/session_20260522_track_b_p4_deploy_prep_offline_hw_sanity.md`.
- DECISIONS: D086 OpenVLA-OFT local inference deps + inline action head pattern.
- Ledger row: 2026-05-22 (Track B P4 deploy prep ...) at line 118.
- Next session: `sudo reboot` → verify CUDA → resume snapshot_download → GPU
  dry-run sanity (1-chunk inference) → Kinect dry-run → real deploy
  (multi-position, head-to-head vs SmolVLA v6 4/9 Plan 3 SUCCESS baseline).
  Verbatim continuation prompt in P4 session doc.

**Track B P4.5 — 2026-05-22 ~19:00 KST (post-P4 verification, real deploy
aborted at Step 0, reboot still pending)**:

- Session entered under premise "reboot done after P4, proceed to GPU dry-run
  + real deploy". Premise verified **FALSE**: `uptime`=1d 20:02,
  `who -b`=2026-05-20 22:53. No reboot between P4 prep (today 2026-05-22) and
  this session.
- `nvidia-smi` still returns `Failed to initialize NVML: Driver/library
  version mismatch / NVML library version: 580.159`. Kernel module
  `580.126.09`, userspace `libnvidia-ml.so.580.159.03`. Same P4 Blocker (a).
- `conda run -n roarm python -c "import torch; print(torch.cuda.is_available())"`
  → `False` (Error 804 forward compatibility).
- `openvla/openvla-7b` HF cache 14 GB on disk (17 blobs) but
  `snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f/` only shows
  `model-00003-of-00003.safetensors` symlink; 00001/00002 symlinks not
  finalized. Likely byte-complete in blobs; next session must re-run
  `snapshot_download` for idempotent fixup.
- No `deploy_openvla_oft.py` change, no env change, no Isaac, no RL, no robot
  command, no Track A file touched. User explicitly chose "Reboot 후 새 세션
  (권장)" via AskUserQuestion.
- Full detail: `claudedocs/session_20260522_track_b_p4_5_reboot_blocked.md`
  (includes verbatim continuation prompt for the post-reboot Track B P5 real
  deploy session).
- Ledger row: 2026-05-22 (Track B P4.5 post-P4 verification ...) at line 119.
- No new DECISIONS entry (no durable lesson — operational reboot omission,
  not a new rule).
- Next: user runs `sudo reboot` from terminal (assistant cannot run sudo
  autonomously). After ~1 min, new Claude Code session, paste P4.5 doc's
  continuation prompt to start Track B P5.

## Latest Verified Track A B200 Evidence

User approved exactly one close_26-only v6 projected-guard runtime on B200 GPU0,
followed immediately by v6 posthoc audit. It failed.

- Runtime stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out`
  md5 `9a4f8825a88ee3c9d93d83e5b9a28b41`, 430 lines.
- Runtime stderr:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.err`
  md5 `947cab475a1eff6ad2f3ccea6505d8c4`, 3 lines / 377 bytes.
- Audit stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out`
  md5 `480a3355864937763eb665e086aadbb0`, 58 lines.
- Audit stderr md5 `d41d8cd98f00b204e9800998ecf8427e` (empty).

Key v6 runtime lines:

- line 43: strict diagnostic-only, close_26-only, v6 flag YES, no training,
  constraints, SurfaceGripper, transport/release, gate tuning, posewrite, or
  success claim.
- line 45: `mode=target_guarded_micro_close_v6_projected_guard_diagnostic`
  and separate-approval marker YES.
- lines 393-397: v6 correctly blocked advance when projected support/target
  margins went unsafe; recovery writes continued with IK OK.
- line 398: first hard freeze/support-gate breach. `target_error=0.002914m`
  was still inside the fixed 0.003m target gate, but counter support gap was
  `0.002075m > 0.002m`; hard freeze YES.
- line 399: both fixed gates were breached: target error `0.003052m > 0.003m`
  and support gap `0.002146m > 0.002m`.
- lines 427-428: posthoc FAIL, 4 advances, 41 holds, zero zero-backlog holds,
  zero safety rollbacks, 12 recovery writes, 0 IK failures, 29 hard freezes,
  `close_reached=NO`, attach/posewrite zero, telemetry-only YES,
  success_claim NO.

Key v6 audit lines:

- line 18: `close_reached pass=NO`.
- lines 27-30: zero zero-backlog holds, zero safety rollbacks, positive recovery
  writes, and zero IK failures all PASS.
- line 31: hard safety freezes zero FAIL (`value=29`).
- lines 51-53: hard freeze / fixed target / fixed support criteria FAIL from
  runtime lines 398-426.
- lines 54-56: recovery present, preemptive trigger seen, and IK OK all PASS.
- line 58: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

Interpretation:

- v6 fixed the specific v5 mistake at old line 394: it did not advance once the
  projection went unsafe.
- v6 still failed because the recovery/hold behavior did not reduce target and
  support error fast enough after advance was blocked. The first failure is a
  support-gate hard freeze at runtime line 398, followed by target+support breach
  at line 399.
- Runtime exit 0 is not success. Success requires audit line 58 to be YES.

## Previous Track A Evidence To Keep In Mind

- v5 runtime/audit remain FAIL: stdout md5
  `f93ddaa75920a560777f8f9c8fae26f0`, audit md5
  `7709c2bc37424bc7c3874e978b34d104`. v5 line 394 advanced while support margin
  was too small; v6 corrected that specific advance decision but not the overall
  close_26 outcome.
- D083: RL-to-expert-to-rollout-to-demo is valid only after a no-attach Stage 0
  contact gate. Existing attach-based Pick/Stack PPO envs are not Track A
  evidence.
- D084: v5 recovery writes alone were insufficient; next advance needed projected
  fixed target/support margin checks.
- D085: v6 projection alone is also insufficient; once projection blocks advance,
  the mechanism must actively recover target/support before hard freeze.

## Current Direction

1. Do not rerun v2, v3, v4, v5, v6, or v7 unchanged.
2. For the professor's immediate cube3cm push/tap branch, keep it separate from
   Track A grasp/dataset/training. The canonical waypoint actor
   `model_actor_waypoint_lowx130.pt` at gain `0.040` has a sharded 10k
   teacher-off first-episode robustness gate PASS, but it is not dataset-ready,
   not Track A evidence, and not PPO/RL/VLA final success.
3. The professor-branch metric cleanup is implemented in the bucket audit/report:
   use the hierarchical sharded-10k output to report `1/5/10/20/30mm`,
   `disp/object_size`, controlled, no-impact, and low-motion. Treat the old 3cm
   `success_marker` as a strict task marker, not the sole objective.
4. Future professor-branch code edits must start with a short code review of
   reset, target generation, action/clipping, logging, and metrics before patching.
5. The professor-branch 10cm/0.72kg push/tap objective is now reaction-first,
   not final-displacement-first. The old 128 v1, fixed16 side-center, and
   settled-start fixed16 gates still failed as final 1cm displacement gates.
   Later 4-env diagnostics narrowed the controller issue: far-face/cross-object
   TCP targets are too aggressive; near-face geometry reduces TCP error; default
   actuator/step control fails to contact; drive boost can move the object. The
   measured-stop freeze seed940 run is a fixed-geometry reaction-event PASS under
   the clarified push/tap criterion: measured contact `1.0`, transient max
   displacement mean `0.010990217m`, transient 1cm rate `1.0`, no overshoot, but
   final displacement gate `0.0` and DiffIK clipping/lag remain. The local
   reaction audit is implemented. The approved seed941 randomized 16-env screen
   FAILed because contact evidence was only `0.625` despite reaction evidence
   `1.0`; no-posewrite and overshoot checks were OK, and teacher quality stayed
   false. Local direction buckets showed `x+` and `y-` contacted in seed941,
   while `y+` and `x-` were weak; cap-only seed942 worsened contact evidence to
   `0.5` and did not remove clipping/lag. The fixed-y+ seed943 screen confirmed
   y+ is weak: reaction `0.9375`, contact evidence `0.375`, no overshoot, final
   TCP error about `0.0706m`, and clip `1.0`. The local y+ geometry/reach audit
   shows contact/no-contact is also workspace-position dependent and vertically
   under-reached at the traced side-center target. The y+ trace-path audit shows
   target world-y advance is `0.019999981m`, final target z is essentially start
   cube z, final z error dominates TCP error, and clip_any is `1.0` for both
   contact and no-contact traced groups. The approved height050 y+ discriminator
   seed944 improved final TCP error to `0.022889409m` but killed contact evidence
   (`0.0`) and failed reaction (`0.6875`), so target-height-only is not the
   recovery path. The approved seed945 good-workspace screen fixed cube
   `x=0.295,y=-0.044` and PASSed the reaction gate (`reaction=1.0`,
   `contact=1.0`, no overshoot/posewrite), but teacher quality still failed
   (`final_tcp_err=0.065514732m`, clip `1.0`). The seed946 lateral `-0.020m`
   screen in that good workspace is the strongest 10cm y+ teacher-candidate
   evidence so far: reaction/contact `1.0/1.0`, final and transient 1cm gates
   `1.0`, no overshoot/posewrite, but teacher quality still false
   (`final_tcp_err=0.062820967m`, clip `1.0`). Next work is actuator/IK tracking
   cleanup or a tiny robustness check around this exact candidate, not cap-only
   escalation, dataset generation, or RL scale-up yet.
6. Do not run Track A dataset generation, PPO/training, rollout, hold-lift,
   transport/release, constraints, SurfaceGripper, or gate tuning from this
   result.
7. v7 active recovery is implemented and diagnostic telemetry works, but the
   post-reboot close_26 audit FAILED. It is not grasp success.
8. The first approved v8 runtime FAILED before recovery could trigger. Do not
   rerun that pre-fix v8 state. A post-fail static fix now makes v8 inherit
   virtual damping and reports `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`, but
   this is not physics validation. The next valid Track A action is exactly one
   post-fix close_26-only v8 runtime only after explicit user approval, followed
   immediately by v8 audit. In Codex, any future GPU/Isaac command still needs
   `sandbox_permissions=require_escalated` because the default sandbox hides
   `/dev/nvidia*` even though host CUDA is healthy.
9. Runtime PASS is not enough for Track A data; next gate is hold-lift.
10. Track A dataset/training remain blocked until close_26 PASS + hold-lift PASS + small
   pilot dataset/replay PASS. Then proceed: no-attach RL env → random sanity →
   PPO smoke → expert rollout → pilot dataset → replay/audit → large dataset →
   BC/VLA/IL training.
11. Do not plan future work around B200 SSH. Use local backups plus local/RunPod
   GPUs. Any remote compute should start by rebuilding/verifying env and smoke
   tests from backed-up artifacts.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` D083-D124
4. `claudedocs/EXPERIMENT_LEDGER.md` latest rows
5. `claudedocs/session_20260522_track_a_v6_projected_guard_runtime_fail.md`
6. `claudedocs/session_20260522_track_a_contact_rl_stage0_preflight.md`
7. `claudedocs/session_20260526_track_a_stage0_to_dataset_step_plan.md`
8. `claudedocs/session_20260526_track_a_v7_active_recovery_static_readiness.md`
9. `claudedocs/session_20260526_track_a_v7_local_runtime_cuda_blocked.md`
10. `claudedocs/session_20260522_b200_retirement_track_a_b_backup_verified.md`
11. `claudedocs/session_20260523_b200_disconnected_next_steps.md`
12. `b200_backup_20260522_final/README_BACKUP.md`
13. `sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py`
14. `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
15. `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
16. `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
17. `claudedocs/session_20260526_runpod_mcp_codex_registration_and_next_prompt.md`
18. `claudedocs/session_20260526_track_a_cuda_reboot_codex_sandbox_ready.md`
19. `claudedocs/session_20260526_track_a_v7_active_recovery_runtime_fail.md`
20. `claudedocs/session_20260526_track_a_v7_failure_static_analysis.md`
21. `claudedocs/session_20260526_track_a_v8_observed_recovery_static_design.md`
22. `sim_scripts/p7_branch_b_cube2cm_target_guarded_v8_observed_recovery_static_design.py`
23. `claudedocs/session_20260526_track_a_v8_runtime_candidate_static_readiness.md`
24. `claudedocs/session_20260526_track_a_v8_runtime_fail_and_damping_wiring_fix.md`
25. `claudedocs/session_20260604_cube10cm_diffik_teacher_gate_prep.md`
26. `claudedocs/session_20260604_cube3cm_hierarchical_bucket_audit.md`
27. `claudedocs/session_20260602_cube3cm_push_metric_reframe_targetext.md`
28. `claudedocs/session_20260529_cube3cm_waypoint_actor_gate.md`
29. `claudedocs/session_20260529_cube3cm_actor_distillation_gate.md`
30. `claudedocs/session_20260529_cube3cm_bc_teacher_bridge_redesign.md`
31. `claudedocs/session_20260528_cube3cm_safety_rl_warmstart.md`
32. `claudedocs/session_20260526_cube3cm_push_rollout_probe_professor_request.md`
32. `sim_scripts/cube3cm_push_rollout_probe.py`
33. `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runtime.out`
34. `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/rollout_stats_audit.out`
35. `claudedocs/session_20260526_cube3cm_push_rl_reward_curriculum.md`
36. `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_speed_guard_model49_eval1024_audit.out`

## Do Not Trust As Current

- `HANDOFF.md`
- `TASKS.md`
- Any claim that runtime exit code 0 means grasp success
- Any claim that target-guarded v1 through v6 passed close_26
- Any claim that target-guarded v7 passed close_26
- Any claim that v5, v6, or v7 should be rerun unchanged
- Any claim that v7 candidate-level selected margins or negative counter-gap
  deltas prove observed runtime recovery
- Any claim that v8 observed-recovery static design is a physics result or means
  v8 is runtime-ready
- Any claim that v8 readiness YES means close_26 physics PASS
- Any claim that the first approved v8 runtime passed close_26
- Any claim that pre-fix v8 should be rerun unchanged, or that v8 readiness before
  the damping-inheritance check proved virtual damping was active
- Any claim that the professor cube3cm push/tap rollout is Track A grasp success,
  PPO/VLA training output, or dataset readiness
- Any claim that the professor cube-push PPO checkpoint is a successful learned
  policy before teacher-off 128 and then 1024 overall/per-bucket audits pass
- Any claim that `model_actor_distill.pt` is a successful learned policy; its
  teacher-off 128 audit failed low-motion/per-bucket gates
- Any claim that `model_actor_waypoint_lowx130.pt` is PPO/RL/VLA final success,
  dataset-ready, or Track A evidence; it has a sharded 10k teacher-off learned
  policy gate PASS for the separate professor cube3cm branch, but not those
  broader claims
- Any claim that the old 3cm `success_marker` is the professor push/tap objective
  by itself. Report hierarchical displacement thresholds and `disp/object_size`
  together with controlled/no-impact.
- Any claim that final 1cm displacement is still the only professor 10cm/0.72kg
  push/tap success criterion. Under the clarified objective, reaction/contact,
  transient displacement, z lift, and speed matter first; final displacement is
  secondary unless the task is relocation.
- Any future 10cm wrapper/default command that encodes final 1cm relocation as the
  primary objective. The wrapper defaults are tap/reaction `0.001m`; 1cm remains a
  secondary relocation/transient diagnostic only when explicitly used.
- Any claim that one lucky success from massive IsaacLab randomization is a
  learned policy, teacher-data readiness, or sim-to-real evidence by itself.
- Any claim that speed-only motion without contact evidence is a push/tap PASS.
  The reaction gate requires contact evidence too.
- Any claim that seed941 randomized 16-env screen passed the reaction gate. It
  failed on contact evidence (`0.625 < 1.0`) and remains teacher-quality false.
- Any claim that simply increasing DiffIK joint-step cap solves the randomized
  10cm reaction gate. seed942 cap050 failed worse on contact evidence (`0.5`).
- Any claim that y+ randomized failures are noise only. Fixed-y+ seed943 also
  failed contact evidence (`0.375`) with no overshoot and high clip/error.
- Any claim that y+ can be fixed by relabeling speed/z reaction as success. The
  local geometry audit shows no-contact rows had only `0.000069159m` mean max
  push displacement despite reaction-like speed/z signals, and traced envs kept
  large vertical TCP-target error at the side-center target.
- Any claim that y+ failure is caused by a missing y target advance. The local
  trace-path audit shows target world-y moves about `0.020000m`; the unresolved
  issue is vertical reach/actuator clipping and workspace-conditioned contact.
- Any claim that +5cm target height fixes fixed-y+ 10cm contact. seed944
  height050 reduced final TCP error but produced contact evidence `0.0`,
  reaction `0.6875`, and reaction gate FAIL.
- Any claim that seed945 good-workspace reaction PASS means 10cm teacher/data/RL
  readiness. It passes contact/reaction, but teacher quality is still false due
  final TCP error `0.065514732m` and DiffIK clip `1.0`.
- Any claim that seed946 lateral `-0.020m` makes 10cm data/RL ready. It is the
  best y+ candidate so far and passes final 1cm reaction/relocation gates, but
  teacher quality remains false because final TCP error is `0.062820967m` and
  DiffIK clip is `1.0`.
- Any claim that the professor 10cm branch must directly call the legacy
  `cube3cm_push_diffik_probe.py` filename. Prefer the new 10cm wrapper entrypoint
  for future 10cm commands, while keeping the shared engine and old logs intact.
- Any claim that the weighted mid/high actor candidate or the target-extension
  probe is ready for 1024/10k scale-up; both failed the 128 posx bucket screen
- Any Track B/OpenVLA training status as evidence for Track A contact success
- Any claim that existing default Pick/Stack PPO envs produce Track A-valid
  no-attach contact experts
- Any plan that requires new B200 SSH access after 2026-05-22 23:59 KST
- Any assumption that `.ssh` secrets were or should be copied as research data
- Any claim that Codex RunPod MCP is available or unavailable without checking
  both `/home/cgxr/.codex/config.toml` and the currently loaded tool namespace
- Any assumption that all complete Track B outputs live under
  `b200_backup_20260522_final/outputs` alone
- Any use of stale RunPod pod `az53n8t8alp8pz` from 2026-05-06 unless the user
  explicitly confirms it is current and active
- Any claim that default Codex sandbox `nvidia-smi` failure means host CUDA is
  still broken. Post-reboot host CUDA is healthy; default sandbox hides
  `/dev/nvidia*`.
- Any assumption that `/tmp/track_a_v7_active_recovery_runpod_overlay_20260526.tar.gz`
  still exists after reboot. `/tmp` is volatile; recreate the overlay if RunPod
  is needed.

## Current Dirty/Untracked Note

Dirty/untracked state is expected. Do not revert it unless explicitly requested.
Track B/OpenVLA files may be present; they are separate from Track A verdicts.

## Continuation Prompt For Next Session

```
Read CLAUDE.md first, then follow Current-State Protocol exactly.

한국어로 브리핑하고, 비판적/분석적으로 진행.
기억만으로 말하지 말고 반드시 파일/라인과 로컬 백업 로그 라인을 확인.
HANDOFF.md / TASKS.md 사용 금지.
기존 dirty/untracked 상태를 임의로 되돌리지 말 것.
B200은 만료/disconnect 상태다. 절대 ssh JHPark / B200 재접속 / 추가 pull / .ssh 복사 시도 금지.
local + RunPod + 로컬 백업만 사용.

Start by running:
git status --short --untracked-files=all

Must read:
1. CLAUDE.md
2. START_HERE.md
3. claudedocs/DECISIONS.md D083-D106
4. claudedocs/EXPERIMENT_LEDGER.md latest rows
5. claudedocs/session_20260522_track_a_v6_projected_guard_runtime_fail.md
6. claudedocs/session_20260522_track_a_contact_rl_stage0_preflight.md
7. claudedocs/session_20260526_track_a_stage0_to_dataset_step_plan.md
8. claudedocs/session_20260526_track_a_v7_active_recovery_static_readiness.md
9. claudedocs/session_20260526_track_a_v7_local_runtime_cuda_blocked.md
10. b200_backup_20260522_final/README_BACKUP.md
11. sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py
12. sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py
13. sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py
14. sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py
15. claudedocs/session_20260526_runpod_mcp_codex_registration_and_next_prompt.md
16. claudedocs/session_20260526_track_a_cuda_reboot_codex_sandbox_ready.md
17. claudedocs/session_20260526_track_a_v7_active_recovery_runtime_fail.md
18. claudedocs/session_20260526_track_a_v7_failure_static_analysis.md
19. claudedocs/session_20260526_track_a_v8_observed_recovery_static_design.md
20. sim_scripts/p7_branch_b_cube2cm_target_guarded_v8_observed_recovery_static_design.py
21. claudedocs/session_20260526_track_a_v8_runtime_candidate_static_readiness.md
22. claudedocs/session_20260526_track_a_v8_runtime_fail_and_damping_wiring_fix.md
23. claudedocs/session_20260526_cube3cm_push_rollout_probe_professor_request.md
24. sim_scripts/cube3cm_push_rollout_probe.py
25. claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runtime.out
26. claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/rollout_stats_audit.out
27. claudedocs/session_20260526_cube3cm_push_rl_reward_curriculum.md
28. claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_speed_guard_model49_eval1024_audit.out
29. claudedocs/session_20260527_cube3cm_diffik_v3_10k_audit.md
30. claudedocs/session_20260527_cube3cm_diffik_v3_visualization.md
31. claudedocs/session_20260528_cube3cm_diffik_dataset_bc_policy.md
32. claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_state_action_dataset_v2_1024_seed779_audit.out
33. claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_state_action_dataset_v2_1024_seed779_bc_train.out
34. claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/bc_mlp_joint_delta_v1_rollout1024_seed883_audit.out

Current Track A state:
- v6 close_26 audit FAIL, not grasp success.
- Runtime stdout:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out
  md5 9a4f8825a88ee3c9d93d83e5b9a28b41
- Audit stdout:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out
  md5 480a3355864937763eb665e086aadbb0
- Reverify runtime lines 43,45,393-399,427-428 and audit lines 18,27-31,51-58 before citing.
- Interpretation: v6 blocked unsafe projected advance but recovery writes did not reduce target/support margins. First support failure line 398; first target+support breach line 399.

Professor pipeline decision:
- RL→expert→rollout→dataset→learning is valid only after Stage 0 no-attach contact primitive.
- Do NOT use existing RoArm-Pick/Stack PPO envs as clean Track A expert sources because they use kinematic attach / write_root_pose_to_sim.
- Do NOT start PPO/training/dataset/rollout first.

Current v7 status:
- Default-off v7 active target/support recovery candidate is implemented and has
  now been physics-tested locally after reboot.
- It uses finite-difference TCP candidate sweep with current object pose and robot joint target writes only.
- Objective: maximize minimum fixed target/support margin; reduce counter gap before next close advance while keeping target error inside fixed 3mm gate.
- Matching audit/readiness support and negative controls exist.
- Local static checks already passed: py_compile, git diff --check, synthetic v7 pass, v7 no-active-recovery reject, archived v6 log reject as v7, readiness.
- Approved local runtime attempt on 2026-05-26 did not produce a physics result: local IsaacLab emitted v7 metadata, then CUDA failed before env creation.
- Preserved local-block logs:
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/runtime.out
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/runtime.err
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/audit.out
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/cuda_check.txt
- Reverify runtime.out lines 28,31,33,35; runtime.err lines 9-16,32-52; audit.out lines 3,16,18,29-30,35; cuda_check.txt lines 1-3,12-13 before citing.
- User rebooted the local Ubuntu PC after that blocked attempt. Host CUDA is now healthy: boot time 2026-05-26 14:08, NVIDIA kernel/userspace 580.159.03, host nvidia-smi OK, isaaclab torch CUDA True/device_count 1 when run outside the default Codex sandbox.
- Default Codex sandbox still hides /dev/nvidia*, so sandboxed nvidia-smi fails. This is a sandbox device visibility issue, not host CUDA failure. Run GPU/Isaac commands in Codex with sandbox_permissions=require_escalated.
- Post-reboot readiness still reports READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES.
- Post-reboot close_26-only v7 runtime/audit FAILED:
  claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/runtime.out
  md5 621d00b9d157b4e70178c28f94ca4c7f
  claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/audit.out
  md5 406b96557d94418f16273e517ec4d69b
- Reverify runtime.out lines 389-393,423-424 and audit.out lines 19,32,54-56,60-66 before citing.
- Interpretation: v7 active recovery did trigger and passed its v7-specific audit checks, but fixed support failed first at runtime line 392 and fixed target failed at line 393. This is not close_26 success and v7 must not be rerun unchanged.

Current v8 status:
- Static-only v8 observed-recovery design is done:
  sim_scripts/p7_branch_b_cube2cm_target_guarded_v8_observed_recovery_static_design.py
  md5 56a382377b7fb0f0c6391bf59163af0d
  claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_observed_recovery_static_design.out
  md5 c14e80ec5fc69c6e6e17925d61f81d0b
- Reverify v8 output lines 1-28 before citing.
- Interpretation: v8 static design finds projected reserve depletion at runtime line 386 / step 9, before v7 first active at line 389 / step 12 and first support breach at line 392 / step 15. It rejects unchanged v7 by observed-response/TCP-follow checks. This is not a physics result: output line 27 says RUNTIME_READY=NO.
- Default-off v8 runtime candidate + matching v8 audit/readiness were implemented:
  sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py pre-fix md5 7e6dfc35bbfeacb5d1689f2f175e5120
  sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py md5 8dbf621c983ec03f46e5d52843781fda
  sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py pre-fix md5 a31ced20b754a4a42058349525d1a435
  claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_runtime_candidate_readiness.out md5 6a2a62808451175b65e5d522b695b8b6
- Reverify readiness lines 1-19 and v8_rejects_post_reboot_v7_audit.out lines 5,14,15,20,30,53,55,57,58,60 before citing.
- The first approved v8 close_26 runtime has now been run and FAILED:
  claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/runtime.out
  md5 74095570c2d6a60abdf522c2413735db
  claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/audit.out
  md5 7cd38eddb1dc9c925b01948cbc5cb416
  claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/v8_runtime_failure_static_analysis.out
  md5 7e81773b91d39658a3ec5c6eaf878f0c
- Reverify runtime.out lines 4,6,8,423-424; audit.out lines 20,26,35-36,45-49,53,60; and v8_runtime_failure_static_analysis.out lines 4,8-12 before citing.
- Post-fail static fix: v8 now inherits virtual damping. Current runtime probe md5 acae0ca2e85a522dd4ac8fb583cb8fb8; readiness md5 dc2bdaa8d882f12b5cc901a677caccc0; readiness_after_v8_damping_fix.out md5 b652520a81792bf12373ff742cdba6b5 lines 5 and 20 PASS. No post-fix v8 runtime has run.

Current RunPod/Codex state:
- Claude has RunPod MCP configured, but Codex initially did not.
- Codex config was updated at /home/cgxr/.codex/config.toml with [mcp_servers.runpod], command npx, args ["-y", "@runpod/mcp-server@latest"], and RUNPOD_API_KEY copied from Claude config without printing the value.
- Backup exists: /home/cgxr/.codex/config.toml.bak_runpod_20260526 md5 1ef4acf6f1c92a64b9bbd79a2e35b7e7.
- Same-session tool_search after config edit initially did not expose mcp__runpod__..., but later Codex sessions did expose mcp__runpod__. In this session, tool_search exposed mcp__runpod__ and `list_pods(computeType=GPU)` returned `[]`. Still verify loaded tools before claiming RunPod MCP can be used.
- Do not use stale RunPod pod az53n8t8alp8pz from 2026-05-06 unless the user explicitly confirms it is current and active.
- The old minimal RunPod overlay at /tmp/track_a_v7_active_recovery_runpod_overlay_20260526.tar.gz was lost after reboot because /tmp is volatile. Recreate it if RunPod is needed.
- Local backup top USD path:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd
  md5 4497024d25abab11de5c50e144124553.

Current professor cube10cm push/tap branch:
- Separate from Track A close_26 grasp and separate from the older 3cm dataset/BC
  line. The active objective is tap/reaction first, not final 1cm relocation.
- Future 10cm DiffIK commands should use:
  `sim_scripts/cube10cm_push_diffik_probe.py`
  while keeping `sim_scripts/cube3cm_push_diffik_probe.py` as the shared legacy
  engine for old tools/logs.
- Guard before any new 10cm runtime:
  `python sim_scripts/cube10cm_tap_objective_contract_audit.py`
  The latest guard JSON says contract `professor_cube10cm_tap_reaction`,
  final 1cm relocation default `false`, tap defaults `0.001m`, explicit 1cm
  override allowed, and verdict PASS.
- Latest object-level candidate remains seed946:
  reaction/contact `1.0/1.0`, no posewrite, no overshoot, final relocation
  secondary pass, but `teacher_quality_ready=false`.
- Latest actuator-tracking screen stiff600:
  reaction/contact `1.0/1.0`, no posewrite, no overshoot, but only tap-scale
  displacement (`max_disp_mean_m=0.004667487`) and `final_relocation_pass=false`;
  `teacher_quality_ready=false`, clip `1.0`, final TCP error `0.046062952m`.
  This does not supersede seed946 as the strongest object-level candidate.
- Latest direction-generalization screen seed947:
  same goodxy/lateral recipe as seed946 but random directions. It FAILed reaction
  gate: contact evidence `0.5625`, reaction `1.0`, no posewrite, no overshoot.
  Direction split: y+ controlled `1.0` but low-motion `0.75`; x- contact `0.0`;
  x+/y- contact `1.0` but controlled `0.0`. This blocks 1024/10240/data.
- Latest x- contact-geometry screen seed948:
  fixed goodxy/lateral with `fixed_push_dir=[-1,0]` and only
  `push_through_m=0.020`. It FAILed harder: contact evidence `0.0`, reaction
  `1.0`, no posewrite, no overshoot, max displacement `0.000009052m`, final
  displacement `-0.000160992m`, clip `1.0`, final TCP error `0.062163922m`.
  This falsifies the simple "deeper near-face target fixes x-" hypothesis.
- Latest x- reach/IK feasibility screen seed949:
  fixed goodxy/lateral and fixed x-, but with `tcp_center_height_offset_m=0.050`.
  It restored tap contact/reaction: reaction/contact `1.0/1.0`, no posewrite,
  no overshoot, contact stop `1.0`, final TCP error `0.012976003m`, clip
  `0.460576925`, and next-step audit `teacher_quality_ready=true`. However it is
  still tap-scale only: final displacement `0.001272157m`, max displacement
  `0.001294456m`, low-motion `1.0`, final relocation secondary false. This is a
  reach/height clue, not 1024/data readiness.
- Latest next-step audit says:
  `RUN_TINY_HELDOUT_ROBUSTNESS_CHECK_BEFORE_DATASET_OR_RL`
  for seed949, but do not start 1024/data: low-motion remains `1.0` and balanced
  directions are not established.

Next concrete step:
1. Do not start dataset generation, PPO/RL, VLA, Track A, 1024/10k scale-up, or
   broad random search from seed946.
2. Before any new 10cm GPU runtime, run and cite:
   `python sim_scripts/cube10cm_tap_objective_contract_audit.py`
   and
   `python sim_scripts/cube10cm_next_research_step_audit.py`
3. Do not repeat stiffness-only escalation as a success claim. The stiff600 screen
   preserved tap contact but reduced useful displacement and left clipping at `1.0`.
4. Do not run 1024/10240 trace/data from seed946/seed947/seed948/seed949. First
   prove direction-specific height/contact geometry in tiny heldout screens and
   remove low-motion/balance blockers.
5. Any next GPU experiment still requires explicit approval and must be one tiny
   local IsaacLab screen inside the working tap geometry, changing only one
   geometry/contact parameter. It must use
   `sandbox_permissions=require_escalated` and must not be reported as RL/data.
6. Judge the result by reaction/contact/no-posewrite/no-overshoot first; report
   final 1cm only as secondary relocation evidence.
```
