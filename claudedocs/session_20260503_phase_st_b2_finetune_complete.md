# 2026-05-03 — Phase ST-B2 완료 (B200 finetune v6+stacking_v2, 5K saturate)

## 요약
- B200 finetune 10K steps PASS (42min 20s, EXIT 0)
- v6_b200/last → v6_stacking_v2_b200/{5K, 10K, last}
- 5K~10K weight diff saturate (rel L2 7.20% → 7.46%, +0.26%만 추가)
- **5K ckpt가 generalization에 더 나을 가능성** — deploy 시 5K + 10K 둘 다 비교 권장

## Hyperparam (4090 train_config.json source-of-truth + finetune 조정)
| 항목 | 값 | 근거 |
|---|---|---|
| pretrained_path | v6_b200/last/pretrained_model | 시작점 = 4/28 evening B200 deploy-equivalent ckpt |
| peak_lr | **5e-5** | 4090 v6 종료 LR(2.79e-5)의 1.8×, 4090 1e-4의 절반 (finetune 표준) |
| warmup | 500 | finetune 짧게 |
| decay_steps | 10000 | = steps (cosine 1주기 정확 완성) |
| decay_lr | 1e-6 | |
| batch_size | 64 | 4090 동일 |
| steps | 10000 | |
| seed | 1000 | 4090 동일 |
| save_freq | 2500 | 4 ckpt + last |
| video_backend | torchcodec | B200 nightly torch (HARD RULE #15) |
| input_features | observation.images.top + state | HARD RULE #16 |

## Cross-check 결과 (5단계 의심 검증)

### 1. train_config.json 무결성 — PASS
모든 hyperparam 의도 그대로 저장. dataset.repo_id=`roarm_m3_stacking_v2`, video_backend=torchcodec, empty_cameras=0.

### 2. ckpt 구조 + size — PASS
- 5K/10K/last(symlink to 010000) 정상
- pretrained_model: model.safetensors 1.2GB + train_config.json + preprocessor/postprocessor + step_5_normalizer + step_0_unnormalizer

### 3. Weight diff (vs v6_b200/last) — saturate confirmed
- 378/500 tensors **bit-exact** (vision encoder + frozen layers, 4/28 evening v6 reproducibility 패턴 정확 일치)
- 122 layers 변형: lm_expert(action expert) self_attn.k_proj 중심 (13~14% rel L2)
- 5K vs v6_b200: rel L2 = **7.20%**
- 10K vs v6_b200: rel L2 = **7.46%** (5K→10K +0.26%만 추가 = saturate)
- max |diff| 0.05 (양호)
- 결론: **5K 이후 학습은 거의 의미 없음**. 5K가 충분.

### 4. Loss curve — saturate from step 5K
| Step | Loss | grdn | LR |
|---|---|---|---|
| 100 | 0.416 | 1.797 | 5.1e-6 |
| 1K | 0.018 | 0.218 | 4.9e-5 (peak) |
| 2K | 0.012 | 0.200 | 4.5e-5 |
| 3K | 0.010 | 0.180 | 4.0e-5 |
| 4K | 0.009 | 0.163 | 3.3e-5 |
| 5K | 0.007 | 0.140 | 2.5e-5 |
| 6K | 0.007 | 0.134 | 1.8e-5 |
| 7K | 0.006 | 0.114 | 1.1e-5 |
| 8K | 0.006 | 0.108 | 5.5e-6 |
| 9K | 0.006 | 0.099 | 2.1e-6 |
| 10K | 0.006 | 0.100 | 1.0e-6 |

- step 1K 이후 빠른 수렴 (v6 ckpt 시작이라 자연)
- 4090 v6 step 1K loss 0.040 vs 본 finetune 0.018 (절반)
- 5K~10K loss 변화 미미 (0.007 → 0.006)
- grdn 감소 monotonic (1.797 → 0.099) = 학습 안정

### 5. Normalizer refit — 정상
v6 stats (6942 frames) → new stats (14242 frames):
| Joint | v6 mean | new mean | Δ | 해석 |
|---|---|---|---|---|
| base | 14.11 | 0.42 | -13.7° | stacking ±Y 대칭 |
| shoulder | 28.4 | 36.5 | +8° | stacking 어깨 들어올림 |
| elbow | 64.6 | 58.7 | -6° | OK |
| wrist_p | 53.3 | 65.2 | +11.9° | stacking top-down +90° |
| wrist_r | 8.9 | 22.9 | +14° | stacking L2 +90° |
| gripper | 23.9 | 25.6 | +1.7° | OK |

step 100 loss=0.416 (높음) → step 1K 0.018 (빠른 적응) = 모델이 새 stats에 적응한 시그널.

## 4090 deploy 준비 (HARD RULE #17)
- 5K ckpt: `outputs/smolvla_v6_stacking_v2_b200/checkpoints/005000/pretrained_model/` (md5 byte-exact PASS)
- 10K ckpt: `outputs/smolvla_v6_stacking_v2_b200/checkpoints/010000/pretrained_model/`
- last → 010000 (symlink)

## Phase ST-C (USB 연결 후 다음 세션)
1. SSH JHPark Port 47110 (이미 chmod 600 4/29)
2. **5K ckpt deploy 우선**: `python deploy_smolvla.py --checkpoint outputs/smolvla_v6_stacking_v2_b200/checkpoints/005000/pretrained_model --task "Stack four pink sponges into a # pattern" --start-pos init --max-steps 300`
3. 10K ckpt deploy 비교
4. Stage 1: 우물정자 build (4 source → #1)
5. INIT_POS=[0,0,90,0,0,0] vs stacking ep[0]=[0,0,90,0,0,5.0] gripper 미세 조정 검토

## 잠재 이슈 / 후속 검토
- **Overfit risk**: 45 epoch 학습 (SmolVLA 권장 10-15 epoch). train loss 0.006 매우 낮음 → 5K (22 epoch)이 더 안전 가능
- **TCP z 최댓값 +343mm** (HOME bridge 11 frames > +180mm 안전 임계): JOINT_SPEED_CAPS deploy 보호
- **wrist_p +90° transit** (+3° v6 OOD): finetune 후 in-dist
- **lying flat sponge 53% 압축** (47→22mm): 실배포 grip 강도 검증

## HARD RULES 준수
- #11 NO /half-clone (Stop hook 94% 거부)
- #13 Lenovo cgxr-Legion-Pro-7 / GPU UUID 05b1a3f8 (sim/local) + sogang_jhki@JHPark-container / GPU 0 c553ca20 (B200) 분리 ✓
- #14 fail-fast guard ([[ -z "$ROARM_B200_ROOT" ]] && exit 1) ✓
- #15 nightly cu128 torch 2.12 + torchcodec 0.12 활용 ✓
- #16 4090 train_config source-of-truth (input_features `observation.images.top` 1개) ✓
- #17 sim render = 4090, B200 학습 전용 ✓

---

**기록자**: Claude Opus 4.7 (1M context). Stop hook 94% 시점 마무리.
