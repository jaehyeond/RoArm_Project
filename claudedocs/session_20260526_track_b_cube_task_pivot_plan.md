# Session 2026-05-26 — Track B Cube Task Pivot + 7-Phase Plan Lock-in

## TL;DR

Track B sponge task → **cube 3×3×3cm task**로 명시 전환 (사용자 결정). 5 cube
**3+2 pyramid stacking** (1층 3개 + 2층 2개) 새 task. Sponge 관련 HARD RULES
#19 (edge-stand 47mm) / #20 (# tower geometry) 자동 무효 (HARD RULE #18 사용자
명시 정정 우선). Sponge 관련 grasp anchor, layout, geometry 모두 무효.

OpenVLA-OFT 7B 학습 hyperparam 갱신: effective batch 8 → **32** (gradient
accumulation × 4), LR `5e-4` → **`2.5e-4`** (P3 collapse 회피),
grad_clip_norm=1.0, warmup 1K step + cosine.

데이터 신규 수집: **250ep (200 stacking + 50 cube pick) ~4일** 권장.

학습 자원: **RunPod A100 80GB** (B200 retirement 이후), B200 env_specs
`pip_freeze_roarm_b200.txt` (3.5K) 로 재구축.

비교 프레이밍: Track A 독립 평가 아니라 **sim demo 증강으로 재포지셔닝**.
Track A close_26 PASS 후 cube stacking sim demos co-training.

## Locked-in Decisions (2026-05-26 user-confirmed)

1. **Task**: cube 3×3×3cm × 5개 → **3+2 pyramid stacking** (L1=3 cube, L2=2 cube).
2. **Camera**: Azure Kinect 고정 위치 유지 (v6와 동일 viewpoint).
3. **Comparison framing**: Track A는 sim demo 증강으로 재포지셔닝 (독립 vs 비교 아님).
4. **GPU**: RunPod A100 80GB, per_gpu_batch=8, gradient_accumulation_steps=4,
   effective batch=32, LR=2.5e-4, grad_clip_norm=1.0, warmup=1K, cosine schedule.
5. **Data scale**: 200ep cube stacking + 50ep cube pick = 250ep.

## Supersedes (sponge → cube pivot)

| 이전 rule | 상태 | 사유 |
|---|---|---|
| HARD RULE #19 (sponge edge-stand 47mm 잡기) | **SUPERSEDED 2026-05-26** | task가 cube로 변경. Sponge orientation 개념 자체 사라짐. |
| HARD RULE #20 (# tower geometry, c2c 87/67mm) | **SUPERSEDED 2026-05-26** | # tower → 3+2 pyramid로 형태 자체 변경. |
| HARD RULE #21 (3-way 비교 lock-in) | 부분 유효 | Track B 부분은 cube task에 적용, Track A 직접 비교는 sim demo 증강으로 재포지셔닝. |
| HARD RULE #24 (v7 stacking 신규 수집 규약) | **DEFERRED 5/07 + SUPERSEDED 5/26** | v7 sponge stacking 규약 무효, cube task용 신규 수집 규약은 본 session에서 정의. |
| Pre-existing v6 ckpt 7500 deploy | 유효 (별도) | sponge pick deploy는 P5에서 결과 보존. cube pivot과 무관. |

## Batch Size 개념 정리 (cube 학습 hyperparam 근거)

| 항목 | 의미 | 우리 영향 |
|---|---|---|
| 1 sample | (224×224 RGB, language prompt, 8-step action chunk = 8×6 floats) | OpenVLA-OFT 입력 형식 |
| 1 iteration | batch만큼 forward → loss mean → backward → optimizer step | gradient update 1회 |
| Effective batch | per-GPU batch × GPU 개수 × gradient_accumulation_steps | gradient noise floor 결정 |
| Vanilla OpenVLA-OFT 논문 | per-GPU 8 × 8 GPUs = 64-128 effective | 우리 8과 1/8-1/16 차이 |
| LoRA fine-tune 최소 권장 | effective 32 | 우리 P2 effective 8은 1/4 |
| P3 collapse | LR 5e-4 + eff_batch 8 + 7B params → 7.5K→10K step 발산 | hyperparam 보수화 필요 |
| Linear scaling rule | effective batch N× → LR N× | 4× batch (32/8) → LR 2× recommended |
| 보수적 적용 | effective 8→32 (+grad clip + warmup), LR 5e-4 → 2.5e-4 (½) | P3 collapse risk 크게 감소, but 성공 미보증 |

## 7-Phase Plan

| Phase | 기간 | 액션 | Gate |
|---|---|---|---|
| **P0** Cube + gripper calib | 0.5일 | Cube 30mm 단일 pick. Gripper jaw 명령각 측정 (sponge anchor `tech_gripper_grasp_anchors.md` 무효, cube 30mm 신규 측정). Grasp z 측정 (sponge +33mm world → cube는 cube top z = +30mm 위 ~+15mm grasp 추정, 실측 필요). | 1 cube 안정 grasp 5/5 |
| **P1** 데이터 수집 | 4일 | 50ep cube pick (L-F teleop, USB0 leader / USB1 follower, Azure Kinect 고정) + 200ep 3+2 pyramid stacking. 일 50ep × 5일. Mid-collection γ-gate (50ep, 100ep checkpoint mini-SmolVLA 학습 vision-blindness 검증) | 일 50ep, mid γ-gate PASS |
| **P2** LeRobot 변환 + sanity | 0.5일 | `convert_to_lerobot_v3.py` (또는 cube-adapted) + `data_episode_quality.py` + `data_distribution_simple.py` | parquet/video shape OK, action distribution 정상 |
| **P3** RunPod 학습 | 1일 | RunPod A100 80GB pod create, env from `b200_backup_20260522_final/env_specs/pip_freeze_roarm_b200.txt`. OpenVLA-OFT 7B LoRA r=32, per_gpu_batch=8, grad_accum=4 (effective=32), LR=2.5e-4, grad_clip=1.0, warmup=1K, cosine, 30K step. ~8h, ~$13. Wandb online (B200은 offline cache만). | offline eval L2 < 15° + 7.5K-10K window collapse 없음 |
| **P4** Deploy ckpt 선택 | 0.5일 | 12 ckpts offline eval (`eval_offline_v6.py` cube-adapted, holdout 5ep + train_sanity 2ep). Rank by holdout l2_step0_mean. | best deploy ckpt 선정 |
| **P5** Real deploy | 0.5일 | Multi-position 5-cube pyramid stacking test (5-10 trials × 다양한 cube 시작 위치). Plan 3 gripper unlock 패턴 재사용 (v6 4/9 SUCCESS). | 5-cube full success rate, partial success (3-cube), drift, total time |
| **P6** Track A sim demo 증강 | 별개 trace | Track A close_26 PASS 시점에 cube stacking sim demos (Isaac Sim/Lab) 생성 → real + sim co-train | Track A blocker 해결 후 별도 timing |
| **P7** 비교 paper | 1일 | OpenVLA-OFT (real-only) vs OpenVLA-OFT (real + sim co-train) ablation. CoRL 2026 paper section. | Table + figure ready |

## Cube Layout Grid (P1 데이터 수집용)

- L1 base 위치 workspace: x ∈ [+250, +330] mm, y ∈ [-100, +100] mm.
- 5×5 grid = 25 cells × 8ep/cell = 200ep stacking.
- Cube spawn (각 ep 5개 cube): 다양한 시작 위치, table 위 분산 배치.
- Yaw: cube symmetric이라 강제 randomize 안 함 (operator 자연스럽게).
- Pick variability (50ep): single cube 한 개를 workspace 다양한 곳에 두고 pick만.

## Pre-Work Checklist (다음 세션 진입 시 read)

| 작업 유형 | 필수 read |
|---|---|
| Cube + gripper calib (P0) | `~/.claude/projects/.../memory/tech_gripper_grasp_anchors.md` (sponge anchor — 무효 표시 후 cube 신규 측정), `~/.claude/projects/.../memory/tech_leader_follower_setup.md` |
| 데이터 수집 (P1) | `collect_data_manual.py`, `~/.claude/projects/.../memory/feedback_v5_data_collection_failure.md` (HARD RULE #1 HOME 시작), HARD RULE #6 카메라 고정 |
| LeRobot 변환 (P2) | `convert_to_lerobot_v3.py`, `data_episode_quality.py` |
| RunPod 학습 (P3) | `b200_backup_20260522_final/env_specs/pip_freeze_roarm_b200.txt`, `b200_backup_20260522_final/code/openvla_oft_roarm/train_roarm_v6.py` (cube-adapted), HARD RULE #15 (단 nightly cu128 → A100은 stable cu126도 OK 검증 필요) |
| Deploy (P5) | `deploy_openvla_oft.py` (561 lines, v6 ckpt 7500 deploy-ready 검증 완료), HARD RULE #13 Follower=USB1 |

## Open Questions / Next Session Decisions

1. **P0 vs Track A 정리 우선순위**: P0 cube calib 즉시 시작 vs Track A close_26 진행 정리 먼저 (사용자 결정 필요).
2. **RunPod env 검증**: A100 80GB가 B200 sm_100 ≠ A100 sm_80이라 nightly cu128 → stable cu126 + same transformers 4.57.6 호환성 검증 필요.
3. **Cube 30mm 물리적 확보**: 3×3×3cm cube ×5개 보유 여부 (인벤토리 확인) — 없으면 발주/3D 프린트 시간 추가.
4. **Track A timing**: P6 (sim demo 증강) 시점에 Track A가 close_26 PASS 했는지가 paper 완성 시점 좌우. 현재 close_26 v2-v7 모두 FAIL, structural recovery 설계 단계.

## Files NOT Modified This Session

- `deploy_openvla_oft.py` (sponge v6 deploy 로직은 cube task에 그대로 reuse 가능 — image+language prompt만 cube task로 바꾸면 됨)
- Track A files (별도 세션 영역)
- v6 LeRobot dataset (`lerobot_dataset_v6/` 등) — sponge dataset, cube task는 신규 수집

## HARD RULE Compliance

- ✅ #1 HOME 시작 (P1 데이터 수집 시 명시)
- ✅ #4 No "확실하게 된다" 주장. "확실히 보장은 X, P3 collapse risk 감소" 명시.
- ✅ #5 JOINT_LIMITS 보존
- ✅ #11 `/half-clone` 거부 유지 (115% context emergency 직전에서도 거부)
- ✅ #13 Follower=USB1, Leader=USB0
- ✅ #18 사용자 명시 정정 절대 우선 (sponge → cube task pivot은 사용자 명시 → #19/#20 자동 supersede)
- ⚠️ #19 SUPERSEDED 2026-05-26 (cube task pivot)
- ⚠️ #20 SUPERSEDED 2026-05-26 (cube task pivot)
- ⚠️ #21 부분 유효 (Track B cube에 적용, Track A 직접 비교는 sim 증강으로)
- ⚠️ #24 SUPERSEDED 2026-05-26 (sponge v7 → cube task)
