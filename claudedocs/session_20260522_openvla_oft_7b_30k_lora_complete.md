# Session 2026-05-22 — OpenVLA-OFT 7B + RoArm M3 v6 LeRobot 30K LoRA Finetune COMPLETE

## TL;DR

5/21 evening 세션의 P2 plan (OpenVLA-OFT 7B 30K LoRA on v6 real)을 5/22 새벽
B200에서 **2h 23min**에 끝까지 학습 완료. 12개 checkpoint (2.5K/5K/.../30K) 각
679MB 저장됨. Track B (CoRL 2026) paper의 첫 번째 hard evidence 확보.

## Verified Facts

### Training Run Summary
- **Start**: 5/22 00:58:39 KST (B200, GPU 0, `sogang_jhki`)
- **End**: 5/22 03:23:39 KST
- **Wall time**: 2h 25min total (training compute = 2h 23min, save overhead ~2min)
- **Throughput**: **3.84 it/s @ batch_size=8** (~0.26s/step on B200 sm_100)
- **Estimated 4090 equivalent**: ~12-15h (B200 ~4-5× faster)

### Model + Data
- **Base**: `openvla/openvla-7b` (Llama2-7B backbone + DinoSigLIP vision tower)
- **Trainable params**: 109,563,904 LoRA + 134,328,326 action_head = **243,892,230** (~244M)
  - LoRA: rank=32, alpha=16, target_modules="all-linear", dropout=0
  - Action head: L1RegressionActionHead, input_dim=4096, action_dim=6
- **All params**: 7,650,801,088 (~7.65B)
- **Dataset**: v6 LeRobot (50ep, 6942 frames, 30 fps, AV1 video, "Pick up the sponge")
- **Action chunking**: NUM_ACTIONS_CHUNK=8, BOUNDS_Q99 normalization
- **Input**: 1 image (224×224), no proprio
- **No image aug, no validation set**

### Hyperparams
- Learning rate: 5e-4 (cosine to 1e-5 over `num_steps_before_decay`=100K)
- Batch size: 8 (single GPU, no grad accum, no DDP)
- Total tokens seen: 30000 × 8 = **240,000 samples** = ~35 epochs over 6942
- AdamW, no warmup
- save_freq=2500 → 12 checkpoints
- `merge_lora_during_training=False` (required — see PEFT 0.18 issue below)

### Checkpoints (B200 `outputs/openvla_oft_v6_b200/`)
| Step | Path | Size |
|---|---|---|
| 2500 | `.../openvla-7b+roarm_v6_pick+b8+lr-0.0005+lora-r32+dropout-0.0--v6_30k--2500_chkpt/` | 679M |
| 5000 | ... `--5000_chkpt/` | 679M |
| 7500 | ... `--7500_chkpt/` | 679M |
| 10000 | ... `--10000_chkpt/` | 679M |
| 12500 | ... `--12500_chkpt/` | 679M |
| 15000 | ... `--15000_chkpt/` | 679M |
| 17500 | ... `--17500_chkpt/` | 679M |
| 20000 | ... `--20000_chkpt/` | 679M |
| 22500 | ... `--22500_chkpt/` | 679M |
| 25000 | ... `--25000_chkpt/` | 679M |
| 27500 | ... `--27500_chkpt/` | 679M |
| 30000 | ... `--30000_chkpt/` | 679M |
- Total = **8.15 GB**
- Each contains: `action_head--N_checkpoint.pt`, `lora_adapter/`, `processor_*.json`,
  `tokenizer*`, `dataset_statistics.json`, `preprocessor_config.json`,
  `processing_prismatic.py`

### Pull Status (07:48 KST start)
- Background rsync from B200 → Lenovo `openvla_oft_b200_pulls/` at ~10 MB/s
- ETA ~14 min (8.15 GB / 10 MB/s)
- PID 1135710, log `openvla_oft_b200_pulls/_pull.log`

## Decisions / Tech Debt (HARD RULE candidates)

### D-OFT-1 — OpenVLA-OFT B200 환경 셋업 (실패 경로 + 성공 경로)

**Failed paths (do not repeat)**:
1. `pip install -e .` with `requires=torch==2.2.0` — HARD RULE #15 위반 시 B200 sm_100
   fail. **요구되는 install: `pip install -e . --no-deps`**.
2. `--attn_implementation="sdpa"` alone — stock transformers 4.57의 `_supports_sdpa`
   property가 PrismaticPreTrainedModel super.__init__ 도중 `self.language_model` 미설정
   상태에서 호출됨. **필수 patch**: cached `modeling_prismatic.py` line 207의
   `@property def _supports_sdpa` 4줄을 `_supports_sdpa: bool = True` 1줄 class
   attr로 교체.
3. HF Hub 원본 `modeling_prismatic.py` 사용 — `set_num_images_in_input`,
   `get_num_patches` 등 openvla-oft fork-specific 메서드 부재. **필수**:
   `code/openvla-oft/prismatic/extern/hf/{modeling_prismatic,configuration_prismatic,processing_prismatic}.py`를 hub snapshot에 덮어쓰기.
4. `merge_lora_during_training=True` (default) — PEFT 0.18 호환성 문제로 step 5
   체크포인트 저장 시 22분+ hang (CPU 99%, GPU 25GB idle, NFS write 정체). **필수**:
   `--merge_lora_during_training False` 명시. Inference 시 별도 merge 가능.
5. `flash-attn` build with stock PyPI source — `nvcc -gencode arch=compute_80,sm_80
   compute_90,sm_90`만 빌드 → B200 sm_100 미지원. **대체**: torch nightly의
   `attn_implementation="sdpa"` 사용으로 충분.

**Working install order on B200 (sm_100 / nightly cu128 / Python 3.11)**:
```
pip install --no-deps peft==0.18.0 sentencepiece json-numpy matplotlib rich timm==0.9.16 diffusers==0.30.3
pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"
pip install tensorflow==2.15.0 tensorflow_datasets==4.9.3
pip install --no-deps tensorflow_graphics==2021.12.3 OpenEXR
pip install --no-deps plyfile uvicorn fastapi protobuf
cd code/openvla-oft && pip install -e . --no-deps
```

**timm 0.9.16 필수** (prismatic은 `< 1.0.0` 강제). transformers 4.57.6은 warning
only (not fatal). tokenizers 0.22.2도 warning only.

### D-OFT-2 — ROARM_M3_CONSTANTS argv 트릭

`prismatic/vla/constants.py:55`의 `detect_robot_platform()`은 `sys.argv` 검색해서
"roarm" 키워드 발견 시 ROARM_M3 constants (ACTION_DIM=6, PROPRIO_DIM=6,
NUM_ACTIONS_CHUNK=8) 로드. **학습 명령에 `--dataset_name roarm_*` 포함 필수**
(우리는 `roarm_v6_pick` 사용). 기본값은 LIBERO (action_dim=7) → RoArm dim mismatch.

### D-OFT-3 — LeRobot v3 → RLDS bypass

openvla-oft `finetune.py`는 `RLDSDataset` (TFDS) 강제. **RLDS 변환 없이 LeRobot v3
직접 사용 가능**: `LeRobotV3RLDSCompatDataset` 작성 (`openvla_oft_roarm/lerobot_rlds_compat.py`).
- LeRobotDataset 0.4.4: `ds.meta.episodes["dataset_from_index"/"dataset_to_index"]`로
  episode 경계 얻기 (`ds.episode_data_index`는 deprecated).
- Image: `ds[i]["observation.images.top"]` = (3, 720, 1280) float [0,1] →
  PIL resize → (H, W, 3) uint8.
- Action chunking: 마지막 액션 padding으로 episode 끝에서도 NUM_ACTIONS_CHUNK 보장.
- Q99 정규화: `stats.json`의 `q01`/`q99` 사용, `clip(2*(x-q01)/(q99-q01+1e-8) - 1, -1, 1)`.

`finetune.py`의 `for batch_idx, batch in enumerate(dataloader)`는 RLDS infinite-loop
가정 — 우리는 finite map dataset이라 `_infinite_dataloader(dl)` generator로
래핑 필요.

### D-OFT-4 — B200 단일 GPU 학습 (torchrun standalone)

DDP 멀티 GPU 불필요 (B200 단일로 충분). `torchrun --standalone --nproc_per_node=1`로
실행. `accelerate.PartialState()` 자동 처리.

## Files Created / Modified (this session)

Local:
- `openvla_oft_roarm/lerobot_rlds_compat.py` — LeRobot v3 → RLDS-compat Dataset (160 lines)
- `openvla_oft_roarm/train_roarm_v6.py` — modified finetune.py (data path swap, sdpa, infinite loop)
- `openvla_oft_roarm/launch_openvla_oft_v6_smoke.sh` — 10-step smoke (b2, save_freq=5)
- `openvla_oft_roarm/launch_openvla_oft_v6_full.sh` — 30K finetune (b8, save_freq=2500)
- `openvla_oft_roarm/pull_checkpoints.sh` — rsync pull helper

B200 (`code/openvla_oft_roarm/`):
- 위 4개 + `finetune_orig.py` (참조)
- HF cache patches:
  - `models--openvla--openvla-7b/snapshots/.../modeling_prismatic.py` (overwritten + sdpa patch)
  - `models--openvla--openvla-7b/snapshots/.../configuration_prismatic.py` (overwritten)
  - `models--openvla--openvla-7b/snapshots/.../processing_prismatic.py` (overwritten)
  - `models--openvla--openvla-7b/snapshots/.../*.hub_orig.bak` (백업본 보존)
- 위 patch는 `update_auto_map()` 호출 시 매번 transformers_modules cache에 복사됨.

## Track B (CoRL 2026 paper) Implications

이 session은 paper의 **"3-VLA real-to-sim" 첫 번째 baseline** 학습 완료에 해당:

- **VLA #1: SmolVLA 450M** — v6 4/05 학습 완료, 4/9 SUCCESS (sponge pick)
- **VLA #2: OpenVLA-OFT 7B** — **이 세션 완료** (5/22 03:23), 30K LoRA r=32
- **VLA #3: π₀ 3.3B** — 미수행, RunPod RTX A6000 후속 작업

평가 미수행 (deploy/offline eval은 별도 세션). 본 세션은 **학습 자체의 reproducibility 증명**:
- $200 RoArm M3 (low-cost arm) + single Kinect + 50ep + 7B VLA LoRA = B200 2.5h에 가능
- 12개 checkpoint sweep으로 best 선택 가능 (offline L2/z-score 평가 후)

## Open Questions / Next Steps

1. **Offline eval**: 12개 checkpoint 중 best 선정 (L2 error + z-score + diversity vs v6 holdout).
   - 4090 또는 B200 release 전 가능
2. **Real deploy**: 4090에서 deploy_smolvla.py 패턴으로 RoArm M3 실시간 추론.
   - inference 시 LoRA를 base에 merge 필요 (offline merge_and_unload)
3. **Comparison vs SmolVLA**: 동일 v6 50ep로 학습한 SmolVLA 450M vs OpenVLA-OFT 7B
   동일 시작 위치/sponge 위치 grid에서 success rate 비교
4. **Track A**: 5/21 P7/Branch B compliance 작업과 무관 (Track A는 sim/lab 작업)

## Risks / Known Issues

1. **R-OFT-1**: Inference 시 stock transformers 4.57의 `_attn_implementation` 자동
   탐색이 fail할 가능성 → inference 코드도 `attn_implementation="sdpa"` 명시 필요.
2. **R-OFT-2**: 30K = 35 epochs over 6942 frames. **Overfit 위험 HIGH** (특히 가장
   늦은 checkpoint 30K). Offline eval에서 early stopping point (5K/10K/15K) 비교
   필수.
3. **R-OFT-3**: HF cache의 patch는 새 `from_pretrained` 시 `update_auto_map()`에
   의해 transformers_modules cache가 hub snapshot에서 재복사됨. **Patch가 hub
   snapshot에도 적용되어 있어야** 영구. 우리는 둘 다 patch했지만, `huggingface-cli
   download --force-redownload` 등으로 cache 무효화되면 patch loss.
4. **R-OFT-4**: BOUNDS_Q99 정규화는 train stats 기반. Deploy 시 state 분포가 train
   범위 outside → clipping으로 [-1, 1] 한도. Eval/deploy 코드는 동일 q01/q99
   사용해야 함 (각 checkpoint의 `dataset_statistics.json` 참조).

## Resource Usage

- **B200 GPU 0**: 25 GB peak memory (out of 182 GB available)
- **B200 disk** (`/NHNHOME/.../JHPark`): 8.15 GB output + ~30 GB model cache + ~600 MB pip
- **Lenovo disk** (`/`): 85 GB free pre-pull → ~77 GB after pull
- **Network**: B200 ↔ Lenovo ~10 MB/s (today, vs 2.67 MB/s yesterday)

## Cross-check (HARD RULE #4 compliance)

- ✅ 12 checkpoints exist on B200 (ls verified)
- ✅ `Max step 30000 reached!` printed in stdout
- ✅ `Saving Model Checkpoint for Step 30000` printed
- ✅ B200 process count = 0 after completion
- ✅ Pull background started at ~10 MB/s, log confirms transfer
- ✅ No `Traceback`, `Error`, `OOM`, `Killed` in last 50 lines of training log
- ⚠️ No `loss=` value recorded in stdout (wandb offline only). Need to inspect wandb
  log at `.cache/wandb/wandb/offline-run-20260522_005832-hxx0jmr6/` for loss curves.
