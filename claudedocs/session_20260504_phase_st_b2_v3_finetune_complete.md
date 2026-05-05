# Phase ST-B2 v3 — B200 Finetune Complete

**Date**: 2026-05-04 night
**Duration**: 19:09:37 → 20:34:15 KST (84min 38s wall)
**Output**: outputs/smolvla_v6_stacking_v3_b200/checkpoints/{005000,010000,015000,020000}/pretrained_model

## Summary

V6 base (lerobot_dataset_v6, 50ep pick) + Stacking v3 (50ep edge-stand 47mm tall # tower) finetune on B200, 20K steps. Loss saturate at 15K~20K (loss 0.005). 4 ckpts byte-exact rsynced to Lenovo for ST-C deploy comparison.

## Configuration

| Param | Value | Source |
|---|---|---|
| Pretrained | `outputs/smolvla_v6_b200/checkpoints/last` (v6 base, 020000) | HARD RULE #16 train_config source-of-truth |
| Dataset | `lerobot_dataset_v6_stacking_v3` (100ep / 14242fr / 2 tasks / AV1) | 5/04 ST-A v3 build |
| Camera | `observation.images.top` (single Azure Kinect) | HARD RULE #16, empty_cameras=0 |
| Batch / Steps | 64 / 20000 | User spec |
| LR schedule | warmup=500 / peak=5e-5 / decay=20000 / decay_lr=1e-6 (cosine) | v2 pattern × steps×2 |
| Save freq | 2500 (8 ckpts) | User spec |
| Seed | 1000 | Reproducibility |
| Video backend | torchcodec | HARD RULE #15 (nightly cu128) |
| GPU | 0 (UUID c553ca20...) | HARD RULE #13 Lenovo lock-in |

## Loss Curve

| Step | Loss | grdn | LR | Note |
|---|---|---|---|---|
| 100 | 0.398 | 1.716 | 5.1e-6 | warmup |
| 200 | 0.142 | 0.438 | 1.5e-5 | warmup |
| 1K | 0.018 | 0.219 | 5.0e-5 | peak LR (v2 dito) |
| 5K | 0.008 | 0.170 | 4.3e-5 | early plateau |
| 10K | 0.006 | 0.139 | 2.7e-5 | mid plateau |
| 15K | 0.005 | 0.114 | 9.4e-6 | saturate |
| 20K | 0.005 | 0.091 | 1.0e-6 | final |

**Saturate**: 15K~20K plateau (loss diff <0.001). updt_s 0.225s/step stable.

## Weight Diff (vs v6_b200/last base)

| Step | Bit-exact | Changed | max\|diff\| | rel L2 |
|---|---|---|---|---|
| 5K | 378/500 | 122 | 0.052 | **0.76%** |
| 10K | 378/500 | 122 | 0.066 | 0.87% |
| 15K | 378/500 | 122 | 0.069 | 0.90% |
| 20K | 378/500 | 122 | 0.070 | **0.90%** (saturate) |

**Pattern**: Vision encoder + frozen layers identical (matches v2 / v6 reproducibility 4/28 evening). 122 trainable changed = lm_expert self_attn.k_proj cluster.

**v3 vs v2 comparison**: v3 rel L2 0.76~0.90% **10× smaller than v2 (7.20~7.46%)**. Why: edge-stand sponges closer to v6 distribution → less adaptation needed → better preservation of v6 grasp capabilities + adds stacking on top.

## Rsync Verification (B200 → Lenovo)

```
5K  md5: dfb6ff6d2ff86f0544e64eed347611d5  ✓ byte-exact
10K md5: a7f55f0f4163c79e5f08da6d51642f3b  ✓ byte-exact
15K md5: 039eace3061cf8d589902e6a98129700  ✓ byte-exact
20K md5: 19d9cfcaafc314d033f355f1fe760b14  ✓ byte-exact
```

Total transferred: 4 × 1.2GB = 4.8GB. Local: `outputs/smolvla_v6_stacking_v3_b200/checkpoints/{005000,010000,015000,020000}/pretrained_model/`. last → 020000 symlink.

## HARD RULE Compliance

- #11 `/half-clone` 거부 0회 (autonomous background)
- #13 dual-PC env (cgxr@Lenovo Lenovo + sogang_jhki@JHPark-container B200), GPU 0 only
- #14 fail-fast guard (`set -e` + `whoami==sogang_jhki` + `[[ -z "$ROARM_B200_ROOT" ]] && exit 1`)
- #15 nightly cu128 torch + torchcodec maintained
- #16 train_config source-of-truth: `observation.images.top` 1개 자동 매핑 (4090 reproducibility)
- #17 sim render 4090, B200 학습 only

## Latent Concerns (ST-C 진입 전 모니터)

1. **90 epoch 학습** (SmolVLA 권장 10-15 epoch 초과). loss 0.005 매우 낮음 → 5K (~22 epoch) ckpt가 generalization에 더 안전 가능. 그러나 v3는 saturate 일찍 일어남 (rel L2 +0.14% from 5K to 20K) → ckpt 선택 차이 작음.
2. **ST-C 1차 = 5K 우선**, 그 후 10K/15K/20K 비교 deploy.
3. **INIT_POS 검토**: `[0,0,90,0,0,0]` (deploy default) vs stacking ep[0] `[0,0,90,0,0,5.0]` (gripper closed) 차이. 작은 OOD 가능 → 미세 조정 검토.

## Next Session Entry — Phase ST-C v3 Deploy

```bash
# USB 연결 (Leader=USB0, Follower=USB1)
# 5K ckpt 우선
python deploy_smolvla.py \
  --checkpoint outputs/smolvla_v6_stacking_v3_b200/checkpoints/005000/pretrained_model \
  --task "Stack four pink sponges into a # pattern" \
  --start-pos init \
  --max-steps 300
# CSV logger fix (5/03 evening) 적용 완료
```

Stage 1: 4 source sponge spawn → grip & place sponge #1 at L1.sp1. 4 ckpts 비교.

## Files Modified / Created

- `outputs/smolvla_v6_stacking_v3_b200/` (new, 4.8GB local)
- `lerobot_dataset_v6_stacking_v3/` (already exists, ST-A v3 build)
- B200: `outputs/smolvla_v6_stacking_v3_b200/` (full 8 ckpts + training_state, ~12GB on Lustre)
- B200: `scratch/launch_train_v6_stacking_v3_b200.sh` (launch script)
- B200: `scratch/train_v6_stacking_v3_b200.{stdout,stderr}.log` (training logs)
- B200: `scratch/weight_diff_v3.py` (weight diff verification)
