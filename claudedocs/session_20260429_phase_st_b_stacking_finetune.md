# Phase ST-A → ST-B 진행 기록 (2026-04-28 ~ 2026-04-29)

> **목적**: v6 single-pick 학습 결과를 N=2 sponge stacking task로 finetune. 4/24 결정한 sim co-training 전략 실행.
> **범위**: Phase ST-A (sim demo 생성 코드) + Phase ST-B 1/3, 2/3, 3/3 (sim demo 50개 → merge dataset → B200 finetune). ST-C (real deploy)는 별도 세션.
> **방법론**: 매 단계 검증·검토·확인·의심하면서 sequential 진행. HARD RULE 1-17 준수.

---

## 0. 한눈에 — 산출물 인벤토리 (검증 완료)

### 코드 (`sim_scripts/`, 5개 신규)
| 파일 | 크기 | 작성 시각 (KST) | 역할 |
|---|---|---|---|
| `roarm_kinematics.py` | 6247 B | 4/28 22:18 | URDF FK + numerical Jacobian + DLS IK + V6WarmStart |
| `generate_stacking_demos.py` | 12879 B | 4/28 22:13 | 50 demos × 24 anchor pose IK + linear interp 95 frame |
| `render_stacking_demos.py` | 16817 B | 4/28 22:29 | Isaac Sim 1회 부팅 + 50 demo per-frame 2-sponge 위치 update |
| `sim_to_lerobot_stacking.py` | 5954 B | 4/29 00:47 | sim demos → LeRobot v3 변환 (4750 frame add_frame) |
| `merge_v6_stacking.py` | 4014 B | 4/29 01:00 | LeRobot native `aggregate_datasets()` (mp4 stream copy, stats bit-perfect) |

### 데이터 (로컬)
| 폴더 | 크기 | 내용 |
|---|---|---|
| `sim_demos_v1/` | 1012 K | 50 × `demo_NNNN_trajectory.csv` (95 frame) + `demo_NNNN_anchors.csv` (24 anchor) + `summary.json` |
| `sim_renders_v3/` | 1.4 GB | 50 ep × 95 frame = 4750 PNG (1280×720, 235 ms/frame B200 @ 4090 render) |
| `lerobot_dataset_stacking_v1/` | 24 MB | sim 단독 LeRobot v3 (4750 frame, libsvtav1, intermediate) |
| **`lerobot_dataset_v6_stacking_v1/`** | **98 MB** | **합본 (v6 50ep + stacking 50ep = 100 ep / 11692 frame / 2 task), av1 stream copy** |

### 데이터 (B200 server)
- `/NHNHOME/.../roarm_b200/data/lerobot_dataset_v6_stacking_v1/` 98 MB (rsync 4/29 02:04 KST, 6.3s)

### 학습 결과 (B200)
- `outputs/smolvla_v6_stacking_b200/checkpoints/{002500, 005000, 007500, 010000, last}/` 총 6.0 GB
- `last/pretrained_model/model.safetensors` = **1,197,789,224 bytes (1.20 GB)** = SmolVLA 450M params bf16

### Git
- 본 세션 시작 직전 commit: `afeb452 B200 sim환경 셋팅` (현재 HEAD)
- 본 세션 결과는 미커밋 (사용자 결정 대기)

---

## 1. Phase ST-A — Sim Demo 생성 코드 작성 (4/28 late-night-2)

### 1.1 v6 분포 분석 → IK 설계 제약 도출
- v6 50ep parquet 직접 분석: elbow [+9°, +126°], shoulder-elbow corr=-0.689, TCP z [+19, +421] mm.
- Stacking target z = +239 mm → **deeply in-distribution** ✓
- v6 grasp = lateral (finger forward + slight down). Sponge upright 47mm 면 lateral grip = v6 분포 match.

### 1.2 `roarm_kinematics.py` (6247 B)
- RoArm M3 6-DOF chain: base → shoulder → elbow → wrist_p → wrist_r → gripper
- SDK degrees == URDF degrees (4/24 sim_v1 RMSE 0.43° 검증된 매핑 그대로)
- Numerical Jacobian (finite diff, eps=1e-3) + DLS IK (damping=0.05) + V6WarmStart class
- **검증 PASS 4건**: HOME [0,0,90,0,0,0] TCP=(+344, 0, +344)mm / round-trip / nearest-warmstart / 8 stacking waypoints (모두 1mm tol 수렴).

### 1.3 `generate_stacking_demos.py` (12879 B)
- 24 anchor (3-step × 8 phase: A_top→Temp / A_bot→B / Temp→B_top)
- Linear joint interpolation 95 frame/demo, 30 fps
- seed 0~49 layout xy ±10mm randomization (`np.random.default_rng(seed)` 6× uniform 순서 보존)
- **결과**: 50 demos × 4750 frame, IK 0 fail, max IK error 1.00 mm, **모든 joint v6 in-distribution** (out_low=0, out_high=0)

### 1.4 미해결 의심 (4/28 기록)
- (a) HOME→첫 anchor 5 frame transition jerk (training non-blocking, v6 동일 패턴)
- (b) Step 3 lift elbow branch jump 47°/4 frame (학습 noise 흡수 가능)
- → **본 세션에서 학습 진행해서 사후 평가하기로 결정** ✓

---

## 2. Phase ST-B 1/3 — Isaac Sim 렌더링 (4/29 early-morning)

### 2.1 작업 (`render_stacking_demos.py`)
- 1회 Isaac Sim 부팅 + 50 demo 반복 + 2-sponge prim (TopSponge dark pink + BotSponge light pink) per-frame 위치 update
- **Held interval 로직**: `HOLD_INTERVALS = {"top": [(12,28),(70,86)], "bot": [(41,57)]}` (close anchor frame ↔ open anchor frame)
- Held sponge = TCP+(0,0,-0.0325m) (sponge center 32.5mm below TCP, lateral grasp mid-side)
- Layout per-seed = generate_stacking_demos.py rng 순서 재현

### 2.2 검증 (의심 → 확인)
- ✅ ANCHOR_FRAMES (26 entries) vs `demo_0000_anchors.csv` max_err=**0.000000 deg** (완전 동일)
- ✅ seed=0 layout 재현 + FK at anchor 3 (S1.close): TCP=(+281.8, -4.6, +210.2)mm vs A=(+282.7, -4.6) → diff x=-0.9mm y=0.0mm z=+0.2mm (1mm IK tol 내)
- ✅ Seed=0 dry run: 17.1s, 95 PNG, 180ms/frame, 29MB. 시각 검증 4 frames (f0000 2-stack at A 2색 구분 / f0021 transit / f0028 drop at Temp / f0094 2-stack at B)

### 2.3 50-demo 본 렌더 (background `b5oxdmjgo`)
- Started 4/28 22:33 KST → finished 22:51 KST. **18.6분**, 235 ms/frame, 4750 PNG, **1.4 GB**
- 시각 검증: ep0 + ep49 frame_0094 모두 2-stack at B 정확 (seed별 layout perturbation 반영)

---

## 3. Phase ST-B 2/3 — LeRobot v3 변환 + Merge (4/29 00:47 ~ 01:00)

### 3.1 sim_to_lerobot_stacking.py (5954 B)
- `LeRobotDataset.create()` API + `add_frame()` 4750회 (lerobot 0.4.4 source 확인 후 수정)
- Single task: `"Stack the pink sponge at A onto B via Temp buffer"`
- **버그 수정**: `task` keyword arg X → `frame` dict 내부 key (lerobot 0.4.4 contract 준수)
- 250.3s (4분, background `b1ookyi4k`) → `lerobot_dataset_stacking_v1/` 24 MB / 50 ep × 95 frame = 4750 frame build PASS
- **검증**: ds.meta.total_episodes=50, total_frames=4750, ds[0] image (3,720,1280) float32 [0, 0.96], state first=[0,0,90,0,0,30]=HOME ✓

### 3.2 Approach 결정 (Approach C 채택)
v6 + stacking 합본 dataset 필요 이유: lerobot 0.4.4 `--dataset.repo_id` single argument 제약.

| Approach | 방법 | 비용 | 채택? |
|---|---|---|---|
| A | LeRobotDataset.create + add_frame v6+stacking 둘 다 re-add | ~10-15분, av1 재인코딩 | ❌ |
| B | 직접 parquet/MP4 concat, stats aggregate 수동 | 빠르지만 stats 정합성 risk | ❌ |
| **C** | **LeRobot native `lerobot/datasets/aggregate.py:aggregate_datasets()`** | **mp4 stream copy 재인코딩 0, stats parallel-variance bit-perfect** | ✅ |

> Approach C는 다른 병렬 세션 검토 후 사용자 권장 결정.

### 3.3 merge_v6_stacking.py (4014 B) 실행
- **0.4초** PASS → `lerobot_dataset_v6_stacking_v1/` 98 MB (data 472 KB + meta 204 KB + video 98 MB)

### 3.4 합본 검증 (의심 → 확인)
- ✅ total_eps=100, total_frames=11692 (= v6 6942 + stacking 4750)
- ✅ total_tasks=2 (0=`Pick up the sponge\n` / 1=`Stack the pink sponge at A onto B via Temp buffer`)
- ✅ av1 codec 보존, 1280×720@30fps, duration=389.733s 정확
- ✅ Single chunk-000/file-000.mp4 (200MB threshold 미만)
- ✅ **Stats aggregation bit-perfect**: mean/min/max max_err=**0.00e+00** vs parallel-variance formula (수동 계산 비교)
- ✅ ds[6942] (first stacking) state=[0,0,90,0,0,30]=HOME, episode_index=50, task_index=1 — boundary 정확

### 3.5 .gitignore 확장
- `lerobot_dataset_v*/` `lerobot_dataset_stacking_*/` `sim_demos_v*/` `sim_renders_v*/` `sim_v*/` `.inference_compare_patched/` `compare_*.{stdout,stderr}` `.claude/scheduled_tasks.lock`
- GitHub Desktop의 "files too large" 경고는 `.inference_compare_patched/{4090,b200}/model.safetensors` symlink dereferenced size 표시 — 실제 git blob ~120 byte이지만 절대경로 + outputs/ 포인터로 환경 의존 → ignore 정답.

---

## 4. Phase ST-B 3/3 — B200 Finetune (4/29 02:11 ~ 02:54)

### Step 1/7 — SSH/env/torch sm_100/lerobot 검증 ✅
- SSH alias `JHPark` (~/.ssh/config 등록, 4/27 late-night-2부터 사용 중)
- env.sh source PASS: `whoami=sogang_jhki`, `hostname=JHPark-container`, `CUDA_VISIBLE_DEVICES=GPU-c553ca20-...` (HARD RULE #13 lock-in)
- `torch=2.12.0.dev20260407+cu128 sm_100=True torchcodec=0.12.0.dev20260407+cu128 lerobot=0.4.4` (HARD RULE #15)
- v6 pretrained `outputs/smolvla_v6_b200/checkpoints/last/pretrained_model/` 7 파일 모두 존재
- 디스크 800G 여유

### Step 2/7 — Dataset rsync ✅
- 98 MB / 6.3 s / **15.8 MB/s** (single mp4 97 MB + meta 5 파일)

### Step 3/7 — Server-side 무결성 검증 ✅
- info.json: 100 ep / 11692 frame / 2 task / codebase v3.0
- LeRobotDataset load PASS: ds[0] state=[0.35, 4.48, 91.32] (v6 ep0 첫 frame), ds[6942] state=[0,0,90] (stacking ep50 HOME), ds[11691] task_index=1 (stacking ep99 마지막)
- Image (3, 720, 1280) float32 [0, 1] ✓
- tasks.parquet: 0=pick / 1=stacking ✓

### Step 4/7 — Launch script 작성 + collision check ✅
- `scratch/launch_train_v6_stacking_b200.sh` (4/28 v6 launch 패턴 그대로 + 5개 변경)
- 변경 항목:
  | 항목 | v6 (4/28) | stacking finetune (4/29) | 이유 |
  |---|---|---|---|
  | pretrained_path | `lerobot/smolvla_base` (HF) | local v6 last/pretrained_model | finetune 시작점 |
  | repo_id | local/smolvla_v6_b200 | local/smolvla_v6_stacking_b200 | 충돌 회피 |
  | dataset.root | lerobot_dataset_v6 | lerobot_dataset_v6_stacking_v1 | 합본 dataset |
  | steps | 20000 | 10000 | sim 4750 frame이 v6 68% → overfit 회피 |
  | save_freq | 5000 | 2500 | 10K에 4 ckpt + last |
  | scheduler_decay_steps | 30000 | 10000 | cosine full 사용 |
- 보존 항목 (4/28 baseline 유지): batch=64 / num_workers=4 / lr=1e-4 / warmup=1000 / decay_lr=2.5e-6 / wandb=false
- syntax check OK, output_dir collision-free

### Step 5/7 — 50-step Dry Run ✅ (의심 1건 해소)
- exit 0, "Loading weights from local directory" 확인 (local pretrained_path 작동)
- num_learnable_params=99,880,992 (100M, Action Expert만, train_expert_only=True 자동 보존)
- num_total_params=450,046,176 (450M, 전체 SmolVLA)
- Loss 0.173 (step 10) → 0.063 (step 50)
- updt_s 0.21s, data_s 0.02s — 4/28 v6 baseline (0.22 / 0.034) 근접

⚠️ **의심 발견 → 소스 검증으로 해소**: step 10 lr=8.0e-5 → step 50 lr=5.2e-6, warmup_steps=1000인데 50 step만에 floor 도달.

해소 과정:
1. v6 last/training_state/scheduler_state.json `last_epoch=20000` 확인 → resume일까 의심
2. dryrun training_state/scheduler_state.json `last_epoch=50` 확인 → **fresh start** 확정 (v6 step 이어받지 않음)
3. lerobot 소스 `lerobot/optim/schedulers.py:99-115` 직접 읽음:
   ```python
   if num_training_steps < self.num_decay_steps:
       scale_factor = num_training_steps / self.num_decay_steps
       actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
       actual_decay_steps = num_training_steps
   ```
4. **결론**: dryrun 50 < decay 10000 → scale=0.005 → actual_warmup=5, actual_decay=50 → step 5 peak → cosine 50 frame floor. 정상 동작.
5. **본 학습 (10000 == 10000)**: `<` False → **NOT triggered** → actual_warmup=1000 + actual_decay=10000. 의도대로 작동 ✓

### Step 6/7 — Production Launch ✅
- tmux session `roarm_train_stacking_b200` started **02:11:38 KST** (Wed Apr 29 2026)
- Cold start 14s (02:11:44 → 02:11:58)
- Config dump 검증: steps=10000 / batch=64 / num_warmup=1000 / num_decay=10000 / peak_lr=1e-4 / decay_lr=2.5e-6 / seed=1000 / **resume=False** / rename_map={}
- Effective batch 64 × 1, num_learnable=100M, num_total=450M ✓

### Step 7/7 — Cold Start + 첫 step 검증 ✅
- "Start offline training" reached
- GPU 76% util, 12 GB / 192 GB VRAM (6%), 596 W / 1000 W max, 35°C

### Mid-run check (02:38 KST, step ~6200) ✅
- tmux alive, 25 min uptime
- Checkpoints `002500`, `005000` 저장됨
- Loss 0.007 saturated (step 5K부터 수평 — 작은 dataset 11692 frame 빠른 수렴)
- Gradient norm 0.13 안정, divergence 0
- LR cosine 3.4e-5 (peak 1e-4의 34%, step 6200/10000 진행률에 매치)
- updt_s 0.224 / data_s 0.024 (v6 baseline 0.22/0.034 근접)
- GPU 95% util / 644 W / 36°C

### 종료 검증 (02:55 KST) ✅ — 전 항목 PASS
| # | 항목 | 결과 |
|---|---|---|
| 1 | tmux session | ✅ alive (script `bash` hold) |
| 2 | "End of training" | ✅ 02:53:58, "Training finished" 02:54:01 |
| 3 | 5 checkpoints | ✅ 002500/005000/007500/010000/last 모두 |
| 4 | last/pretrained_model + training_state | ✅ 7+5 파일 정상 |
| 5 | **Final LR = 2.5e-06 = decay_lr 정확 도달** | ✅ cosine floor |
| 6 | GPU idle | ✅ 0% util / 0 MiB / 190 W standby |
| 7 | Output total | 6.0 GB (4 ckpt × 1.5 G + last symlink) |

### 학습 흐름 정량 정리
| Step | Loss | LR | 비고 |
|---|---|---|---|
| 50 (dryrun) | 0.063 | (5e-6 dryrun floor) | warmup peak 후 |
| ~5K | 0.008 | 5.0e-5 | saturation 진입 |
| ~6K (mid-run) | 0.007 | 4.4e-5 | 안정 |
| ~9K | 0.005 | 3.8e-6 | decay phase 추가 미세 개선 |
| **10K (final)** | **0.005** | **2.5e-6** | floor 정확 도달 |

### 시간/비용
- **Wall clock**: 02:11:38 → 02:54:01 = **42 min 23 s**
- 학습 부분만: 02:11:58 → 02:53:58 = **42 min 0 s**
- 4/28 v6 (20K steps) 1h 25min 대비 정확히 절반 — steps 비례 확인 ✓
- updt_s 동일 (0.224 vs 0.22), data_s 더 빠름 (0.024 vs 0.034) — 합본 dataset 작아서 cache 효과

---

## 5. 의심·교차 검증 기록 (이번 세션 학습 가치)

| # | 의심 | 검증 방법 | 결론 |
|---|---|---|---|
| 1 | 로컬 pretrained_path가 training_state도 자동 resume? | dryrun training_state/scheduler_state.json `last_epoch` 확인 | **No, fresh start** (`last_epoch=50` not 20050) |
| 2 | Dryrun step 10 lr=8e-5 비정상? | lerobot 소스 직접 읽기 (`lerobot/optim/schedulers.py:99-115`) | **정상** — `num_training_steps < num_decay_steps`일 때 auto-scale 작동 |
| 3 | 본 학습 (10K=10K)에서 schedule 의도대로? | scale_factor 조건 `<` (strict less-than) 확인 | **OK** — 10000 == 10000은 트리거 안 됨, actual_warmup=1000/decay=10000 |
| 4 | Dataset boundary 정확 (v6 6942 + stacking 4750)? | ds[0]/ds[6942]/ds[11691] state + task_index 직접 sample | **PASS** — boundary frame이 HOME state(0,0,90)부터 시작 |
| 5 | Stats aggregation bit-perfect? | parallel-variance formula 수동 계산 vs aggregate 결과 | **max_err=0.00e+00** |
| 6 | torch sm_100 + lerobot install 순서 깨졌나? | `torch.cuda.get_arch_list()` 안에 sm_100 확인 | **HARD RULE #15 유지**, sm_100 OK |
| 7 | Final LR cosine floor 정확? | `scheduler_state.json:_last_lr` 확인 | **2.5e-06 정확 도달** |
| 8 | Loss 0.005 saturation = 학습 됐나 vs 진짜 OOD에 대해 모르는 채? | (사후) test_inference_official.py로 hold-out 평가 필요 | **❌ ST-C 진입 전 처리 필요** |

---

## 6. Phase ST-C 진입 전 — 사용자 결정 필요 사항

### Phase ST-C 진입 즉시 가능 작업 (제약 없음)
1. **last/pretrained_model rsync B200 → 4090** (~1.20 GB, ~70초 ETA, HARD RULE #17)
2. **로컬 4090에서 offline inference** (`test_inference_official.py`):
   - v6 baseline (`outputs/smolvla_v6_b200/checkpoints/last/`) vs stacking finetune 결과 비교
   - L2 error / z-score / diversity per task (pick vs stacking)
   - **핵심 질문**: stacking finetune이 v6 grasp behavior 손상시켰나? (catastrophic forgetting check)

### 결정 필요 사항
| # | 결정 | 옵션 |
|---|---|---|
| D1 | rsync 시점 | 지금 / 나중 (B200 1개월 대여 중) |
| D2 | ckpt cleanup | 전체 보존 (6.0 GB) / 일부 정리 (last + 010000 + 002500만 → 4.5 GB) |
| D3 | offline test 위치 | 4090 (HARD RULE #17 — 권장) / B200 (compute 빠름이지만 deploy 환경 아님) |
| D4 | Real deploy 시점 | 오늘 / 내일 |
| D5 | Real deploy 시작 task | (a) 단순 v6 pick (regression check) → (b) Layout A 단독 → (c) 3-step A→Temp→B |
| D6 | Dryrun output 1.5 GB cleanup? | 보존 (audit) / 삭제 (감사 후) |
| D7 | git commit policy | 지금 (산출물 파일은 .gitignore, 코드 5개 + 본 md 만 commit 가능) / ST-C 후 한꺼번에 |

### 잠정 미해결 (ST-C 진행하면서 평가)
- 4/28 late-night-2 미해결 2건 (HOME→첫 anchor jerk / Step 3 lift elbow branch jump) — 학습 noise 흡수했을 가능성, real deploy에서 확인
- Sim2real gap (4/24 SigLIP 0.7222 GO baseline) — stacking 새 visual context (2-stack tower)에서 SigLIP 재측정 필요할 수 있음

---

## 7. 다음 세션 재개 지점 (continuation)

```text
세션 재개 지점: Phase ST-C 진입 직전.

직전 완료:
- B200 finetune 42m 23s, 5 ckpts saved at outputs/smolvla_v6_stacking_b200/checkpoints/{002500,005000,007500,010000,last}/
- last/pretrained_model/model.safetensors = 1.20 GB
- Final loss 0.005 / final LR 2.5e-6 (decay floor 정확)
- 합본 dataset (lerobot_dataset_v6_stacking_v1/ 98MB) 로컬 + B200 양쪽 보유

D1-D7 결정 필요. 최소 D1+D3+D5 결정 후 Phase ST-C 진입.

권장 진행 순서:
1. D1=지금 → rsync (~70초)
2. D3=4090 → offline test (test_inference_official.py 수정 필요할 수 있음 — task_index 0 vs 1 분리 평가)
3. D5=(a) v6 pick regression → (b) → (c) 점진적
4. D7=ST-C 완료 후 묶어서 commit (코드 5개 + session md 2개)

HARD RULES: #11 NO /half-clone, #13 GPU UUID c553ca20, #15 nightly cu128, #16 train_config 4090 source-of-truth, #17 deploy=4090.
```

---

**기록자**: Claude Opus 4.7 (1M context). 본 문서는 user 요청에 따라 Phase ST-C 진입 전까지 산출물 + 검증 + 의심 해소 과정을 step-by-step으로 정리. Phase ST-C 진행은 별도 세션에서 이 문서를 referenece로 시작.
