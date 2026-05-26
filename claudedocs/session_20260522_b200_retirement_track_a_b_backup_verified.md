# Session 2026-05-22 — B200 retirement: Track A/B backup verification

## TL;DR

As of 2026-05-22, the NHN/Sogang B200 lease ends at 23:59 KST. Future work must
not rely on re-entering B200 through SSH or on B200-only paths. Research can
continue from local backups plus local/RunPod GPU rental.

Strict scope:

- Do not copy or depend on `.ssh` private material. The backed-up artifacts are
  research outputs, logs, code snapshots, checkpoints, env specs, and wandb
  cache, not login secrets.
- Track A evidence is complete for the current B200 evidence scope:
  `/tmp/p7_branch_b_*` plus B200 `code/sim_scripts`.
- Track B continuation assets are complete for the current B200 output scope:
  all 9 B200 `roarm_b200/outputs/*` directories are preserved locally, although
  not all are under one local directory.
- Track B real robot P5 deploy is still not complete; it remains a local
  post-reboot deploy task. The backup being complete does not mean deployment
  was run.

## Track A Backup Verification

Source scope:

- B200 `/tmp/p7_branch_b_*`
- B200 `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts`

Local backup:

- `b200_backup_20260522_final/tmp_p7`
- `b200_backup_20260522_final/code/sim_scripts`

Reverified on 2026-05-22:

| Item | Local | Remote | Verdict |
|---|---:|---:|---|
| Track A `/tmp` files | 494 | 494 | MATCH |
| Track A `/tmp` path+size hash | `c308d1a682560cf51136cdd1a018c50ce2e7b488f1a0d4620e31abf7de80cfd4` | same | MATCH |
| Track A `/tmp` file-content aggregate hash | `cca0586b77c36ee79532d0640f9a35b2f1056654ab2758f256ea2bc1f149a4ae` | same | MATCH |
| `sim_scripts` non-pycache files | 53 | 53 | MATCH |
| `sim_scripts` path+size hash | `98563bbc3d27426351abd13272a88537009372b2c709b46d2a5021560c5ea23a` | same | MATCH |
| `sim_scripts` file-content aggregate hash | `fefe4c873c1e45ec4cb95226a2c1a0d53860e4eca926c93d3da1b9887c9ca83f` | same | MATCH |

Key Track A logs preserved:

| Log | md5 | Lines | Verdict |
|---|---:|---:|---|
| v5 runtime stdout | `f93ddaa75920a560777f8f9c8fae26f0` | 430 | FAIL evidence preserved |
| v5 audit stdout | `7709c2bc37424bc7c3874e978b34d104` | 59 | FAIL evidence preserved |
| v6 runtime stdout | `9a4f8825a88ee3c9d93d83e5b9a28b41` | 430 | FAIL evidence preserved |
| v6 audit stdout | `480a3355864937763eb665e086aadbb0` | 58 | FAIL evidence preserved |

Interpretation remains unchanged: v6 is not grasp success. Runtime line 398 is
the first support-gate / hard-freeze failure; audit line 58 is
`SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

## Track B Backup Verification

B200 output directories found under `roarm_b200/outputs`:

1. `dryrun_stacking`
2. `dryrun_stacking_v2`
3. `openvla_oft_v6_b200`
4. `openvla_oft_v6_eval`
5. `openvla_oft_v6_smoke`
6. `smolvla_v6_b200`
7. `smolvla_v6_stacking_b200`
8. `smolvla_v6_stacking_v2_b200`
9. `smolvla_v6_stacking_v3_b200`

Verified under `b200_backup_20260522_final/outputs`:

| Directory | Files | Size | Manifest sha256 |
|---|---:|---:|---|
| `dryrun_stacking` | 12 | 1.5G | `a3d4db27a108b217d077a71ba7e5f621b3da6df274ccc74151cce1ff55f1f656` |
| `dryrun_stacking_v2` | 12 | 1.5G | `4bef003d7c80ec22fea707a55918bee9791998bfb0569a277c119847dba965f6` |
| `openvla_oft_v6_eval` | 5 | 48K | `2be09e86be1b4a35bfe5fe71df3306c710ebf5d10dd65f543f8c988848dd0a9d` |
| `openvla_oft_v6_smoke` | 27 | 1.4G | `cc2672e8ea9239c71bb8d35804345f9de4f6835b25c93809c407959bb5922219` |
| `smolvla_v6_stacking_b200` | 48 | 6.0G | `008e815516683cb1e7c983319595099659978ffed14f73203c76072f4bc1b0c6` |
| `smolvla_v6_stacking_v2_b200` | 48 | 6.0G | `82445b91d532b8f8fdd3af5aa472c1c3b2e315252f8739e8f5c8127fc56ac3a5` |

Verified in older local backup `b200_backup_20260521`:

| B200 directory | Local directory | Files | Size | Manifest sha256 |
|---|---|---:|---:|---|
| `smolvla_v6_b200` | `outputs_smolvla_v6_b200` | 48 | 6.0G | `299629fb06434c72d0207b97fe067a5f366406c1f4c5824423c92cf69eac5764` |
| `smolvla_v6_stacking_v3_b200` | `outputs_smolvla_v6_stacking_v3_b200` | 96 | 12G | `cc6b5e5553255c2149b94ef2587d73ef89269d65f611cca0e3b558c87422975d` |

Verified in `openvla_oft_b200_pulls`:

- Covers B200 `outputs/openvla_oft_v6_b200`.
- Remote manifest: 157 files,
  `8de4ac7dea4107d576c0355379a7d995166cbf627a8ed076ab78ddd35c88ffb2`.
- Local manifest: 160 files,
  `aacfdea194d9a8b849d83f1a92a08375abfeee6bd51ade934aa40de34009a04c`.
- `comm -23 remote local` = 0 missing remote files.
- Local has three extras: `_pull.log`,
  `openvla_oft_v6_eval_20260522_121028.json`,
  `openvla_oft_v6_eval_20260522_121028_partial.json`.
- Best deploy checkpoint remains ckpt 7500 from P3 offline eval. The eval JSON
  sha256 is `3707a0ee1efd189868eb0421a1f56b2a71ec16dfb3b87632772a0e5a87332bf0`.

Additional continuation assets:

| Asset | Verification |
|---|---|
| `b200_backup_20260522_final/env_specs` | local/remote files match: `env_roarm_b200_full.yaml` 44 bytes, `pip_freeze_roarm_b200.txt` 3487 bytes, manifest sha256 `5e357fb4ebd4efc1a9b2918af30ecbec39128c8a54d93029557dd1f1fdb01151` |
| `b200_backup_20260522_final/wandb_cache` | local/remote match: 35 files, 5.7M, manifest sha256 `d68c65cb1f08ed76a02634952e62b1d4c24b3300f39ec3c7dee13649db8ce871` |

## Critical Caveats

- `b200_backup_20260522_final/outputs/openvla_oft_v6_b200`,
  `outputs/smolvla_v6_b200`, and `outputs/smolvla_v6_stacking_v3_b200` are not
  the complete copies. Use `openvla_oft_b200_pulls` and `b200_backup_20260521`
  for those.
- B200 access after 2026-05-22 23:59 KST should be treated as unavailable.
  Future sessions must not plan SSH/B200 reruns as a required step.
- RunPod/local continuation is feasible, but not zero-effort: rebuild the env
  from `env_specs/pip_freeze_roarm_b200.txt` or project scripts, verify CUDA,
  verify checkpoint paths, then run small smoke tests before full training.
- Track A still has no close_26 success. Track A dataset generation/training is
  blocked until close_26 PASS, hold-lift PASS, and pilot replay PASS.
- Track B P5 real deploy remains pending local reboot/CUDA verification and
  user approval for robot motion. Backup completion is not a deploy result.

## Next Work Without B200

Track A:

- Local/static/code-first active target/support recovery after v6 projected
  block.
- Runtime, PPO/training, dataset generation, hold-lift, constraints,
  SurfaceGripper, transport/release, or gate tuning still require separate
  approval and should target local/RunPod infrastructure, not B200.

Track B:

- Rebuild or use local `roarm`/OpenVLA env.
- Use ckpt 7500 from `openvla_oft_b200_pulls`.
- Finish local reboot/CUDA check, then Track B P5 deploy protocol from
  `session_20260522_track_b_p4_5_reboot_blocked.md`.
