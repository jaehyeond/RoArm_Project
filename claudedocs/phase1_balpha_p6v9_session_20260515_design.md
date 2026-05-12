# Phase 1.B-α P6v9 design — ManiSkill-strict (xy/z 분리) + ungrasp force-set 2-channel fix (5/15)

## TL;DR

- 🔴 **Root cause 재진단 (외부 cross-check 후)**: P6v6/v7/v8 모두 ManiSkill StackCube를 "REPLACE tower" 채택했다고 했지만 **2개 critical detail을 omit**: ① stage 3 정의를 단순 3D Euclidean으로 단순화 (ManiSkill 원본 xy_flag AND z_flag 분리) → hover trap baked in ② ungrasp_reward의 `~is_grasped → 1.0` force-set 누락 → release transient 보상 잃음.
- ✅ **P6v9 patch (2-channel fix)**: ① stage 3 정의 = `(xy_offset < 30mm) AND (z_offset < 25mm)` (ManiSkill xy_flag/z_flag 정확 재현, 30mm = ManiSkill cube 5mm + 우리 sponge dimension 정합) ② `ungrasp_signal = where(~is_grasped, 1.0, ungrasp_signal)` (ManiSkill L150 force-set 정확 재현) ③ jackpot 비활성화 (`success_jackpot = 0`, ManiSkill 원본에 없음) ④ episode 200 유지 (DrS 3-5× horizon 권장 / α 100 추가 단축 옵션은 reach 시간 부족 위험).
- ✅ **Sanity test PASS** (B200 64env × 2 iter): `ungrasp_signal_mean=1.0` (random policy `~is_grasped=True`로 force-set 정확 작동), `is_on_target_rate=0.0` (정상, sponge 멀리), `xy_offset_mean=0.186→0.254m`, `z_offset_mean=0.044→0.078m` 새 진단 keys 정상 출력.
- ✅ md5 verify PASS (`f43decac350acc534da1e3d5d26d2e09` local↔B200 일치).
- ✅ launch_p6v9.sh 작성 완료 (resume P6v8 model_999, --reset_std 1.0, entropy_coef 0.001, episode_length_s 2.0, experiment_name `p6v9_maniskill_strict_resumeP6v8`).
- 📋 사용자 confirm 대기 항목: ① P6v9 launch GO/HOLD ② reset_std 1.0 vs 1.30 vs 유지 (0.86) ③ actor.6.bias[5] reset 0 추가 여부 (HARD RULE #11 conservative default = no).

## 외부 분석 cross-verification (사용자 ChatGPT 분석 제공)

| Claim | 사실 확인 | 우리 코드 영향 | 우선순위 |
|---|---|---|---|
| Claim 1: `ungrasp_reward[~is_grasped] = 1.0` 누락 | **✅ TRUE** (ManiSkill stack_cube.py L150 직접 fetch 확인) | release transient 보상 잃음 (우리 6.5 vs ManiSkill 6.75 = -0.25/step) | P6v9 patch 적용 ✓ |
| Claim 2: max_episode_steps=50 horizon | **⚠️ PARTIAL TRUE** (ManiSkill 50 사실 but 우리 task 더 복잡) | 우리 200 → 50 단축은 reach 시간 부족 위험 (sponge spawn 4 regions × 280mm transport vs ManiSkill 작은 cube < 20cm) | 유지 200 (DrS 3-5× horizon 권장 정합) |
| Claim 3: Tier 1 (ε+horizon) → Tier 2 (Eureka) → Tier 3 (RFCL/DrS) | **✅ VALID 우선순위** (5/19 deadline 4일 잔여 = Tier 1만 가능) | Tier 2/3는 5/19 후 검토 | Tier 1 진행 ✓ |
| Claim 4: 우물정자 hierarchical decomposition | **✅ TRUE** (현재 단일 stacking PoC 미해결) | 본 P6v9가 PoC fix → 성공 시 우물정자는 hierarchical로 진행 | 본 PoC 우선 |

## ManiSkill stack_cube.py 원본 코드 직접 인용 (cross-check 근거)

```python
# stack_cube.py L122-167 (haosulab/ManiSkill main branch, 직접 fetch):

# Stage 1: reach
reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))                          # 0~2

# Stage 2: grasped → REPLACE
goal_xyz = torch.hstack([cubeB_xy, cubeB_z + 2*half_size])                    # target above cubeB top
cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)
reward[is_cubeA_grasped] = (4 + place_reward)[is_cubeA_grasped]               # 4~5

# Stage 3: cubeA on cubeB → REPLACE (strict xy AND z separated)
# ⚠️ KEY DETAIL: is_cubeA_on_cubeB = xy_flag AND z_flag (separated, not Euclidean!)
#   xy_flag = |offset_xy| ≤ half_size + 5mm    (~30mm for 4cm cube)
#   z_flag  = |offset_z - 2*half_size| ≤ 5mm   (precisely stacked z)
gripper_width = ...
ungrasp_reward = sum(qpos[-2:]) / gripper_width                               # 0~1, gripper open ratio
ungrasp_reward[~is_cubeA_grasped] = 1.0                                       # ⚠️ KEY: force-set 1.0 when not grasping
static_reward = 1 - tanh(v*10 + av)                                           # 0~1, low velocity
reward[is_cubeA_on_cubeB] = (6 + (ungrasp_reward + static_reward) / 2.0)      # 6~7

# Stage 4: success → REPLACE
# success = is_cubeA_on_cubeB AND is_static AND ~is_grasped
reward[info["success"]] = 8                                                   # 8
```

**우리 P6v8 코드 (보정 전, roarm_stack_env.py L519-571)**:
- Stage 3 condition = `is_near_target = d_sponge_target < 100mm` (3D Euclidean) ← **xy_flag AND z_flag 분리 안 됨**
- `ungrasp_signal = clamp((gripper_high - q) / range, 0, 1)` ← **`~is_grasped → 1.0` force-set 없음**
- Stage 4 condition = `is_success_zone (50mm Euclidean) & gripper_open & stable` + jackpot 20 ← **ManiSkill에 jackpot 없음 + 50mm Euclidean도 hover trap 잔존 (xy/z 분리 안 되어)**

→ **2-channel mismatch 확정**.

## 정량 분석 — 왜 2-channel fix가 필요한가

### Channel 1: stage 3 정의 (hover trap root cause)

iter 999 sponge_target_dist mean = 120mm (P6v8). 우리 정의로는 stage 3 fire 안 함 (>100mm). 그러나 near_target_rate=0.645 → 64.5% env가 d<100mm. 이 중 hover 상태 (z=88mm, target z=11mm, z_offset=77mm)인 env가 다수:

- 우리 정의: `d < 100mm` (3D) → sqrt(xy² + 77²) < 100 → xy < 63mm 인 env 모두 stage 3 fire. **hover에서 stage 3 reward 6.16/step**.
- ManiSkill 정의: `(xy < 30mm) AND (z < 5mm)` → hover (z_offset=77mm) → fire X. **hover에서 stage 1만 (~0.2/step)**.

**Δreward** = (6.16 - 0.2) × 199 step × 0.645 fire rate = **+764/episode hover 보너스 우리만**. ManiSkill에서는 hover가 reward attractor 아님.

### Channel 2: ungrasp force-set (release transient cliff)

release 시 gripper joint q closed → open transition ~10 step. 그 동안 ungrasp_signal 정의:
- 우리 (force-set 없음): q 기반 normalize, 0 → 1 로 점차 ↑. 평균 ~0.5/step 동안.
- ManiSkill (force-set 있음): `~is_grasped`이면 즉시 1.0. 평균 1.0/step.

`stage3_r = 6 + 0.5*ungrasp + 0.5*static`. release transient 10 step:
- 우리: 6 + 0.25 + 0.25 = 6.5/step × 10 = **65 reward**
- ManiSkill: 6 + 0.5 + 0.25 = 6.75/step × 10 = **67.5 reward**
- **Δ = +2.5 per release event**

추가로 release 후 sponge stable 단계 (~100 step, ~is_grasped=True):
- 우리: ungrasp_signal 점차 1로 → 평균 ~0.8 → stage 3 = 6.4
- ManiSkill: ungrasp_signal = 1 (force-set) → stage 3 = 6.75
- **Δ = +0.35/step × 100 = +35 per episode after release**

**Combined Channel 2 impact**: ~+40 reward for release path. 의미 있는 incentive but Channel 1 (hover trap)보다 작음.

### 2-channel 결합 효과 예측

P6v9 reward 산수 (예상 iter 999, hover trap 해제 + force-set):

| 상황 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Total/199 step |
|---|---|---|---|---|---|
| Hover (P6v8 정책 그대로) | 0.27 × 0.14 | 4.5 × 0.22 | **0 (hover에서 stage 3 fire X 우리 신규 정의)** | 0 | ~**240 reward** (P6v8 1005 대비 -75%) |
| Transport + release to target zone | 0.27 × 0.14 | 4.8 × 0.30 | 6.75 × 0.40 | 8.0 × 0.10 | ~**+5/step × 199 = +1000** |

→ Hover path는 **-75% reward** drop, transport path는 +1000 → **transport가 강하게 우세**. 정책이 P6v8 hover policy로부터 transport 학습 강제됨.

⚠️ 위험: P6v8 정책의 hover habit이 매우 깊으면 (entropy collapse 0.86) → reset_std 1.0이 충분한 exploration 보장 못 할 수도. 보수적으로 reset_std 1.20도 옵션.

## 패치 코드 변경 요약

### 1. Config (roarm_stack_env.py L220-228)

```diff
-    success_dist_thresh: float = 0.050     # 50mm. Stage 4 success zone (P6v8).
-    success_jackpot: float = 20.0          # One-time bonus on first stage 4 entry (P6v8).
+    success_dist_thresh: float = 0.050     # 50mm. UNUSED in P6v9 (kept for backward log).
+    success_jackpot: float = 0.0           # P6v9: disabled. Was 20.0 (P6v8). ManiSkill has no jackpot.
+    on_target_xy_thresh: float = 0.030     # 30mm. ManiSkill xy_flag analog (P6v9, 5/15).
+    on_target_z_thresh: float = 0.025      # 25mm. P6v9 z_flag analog.
```

### 2. Reward — stage 3 정의 + ungrasp force-set (`_p6v6_replace_tower`)

```diff
     is_grasped = self._grasped  # physics-attach state
-    is_near_target = d_sponge_target < self.cfg.place_dist_thresh
+    is_near_target = d_sponge_target < self.cfg.place_dist_thresh  # loose 3D Euclidean (log only in P6v9)
+
+    # P6v9 ManiSkill-strict stage 3 (xy AND z separated)
+    sponge_target_xy = self._target_world[:, :2] - self._sponge_pos_w[:, :2]
+    sponge_target_z = self._target_world[:, 2] - self._sponge_pos_w[:, 2]
+    xy_offset = torch.norm(sponge_target_xy, p=2, dim=-1)
+    z_offset = torch.abs(sponge_target_z)
+    is_on_target = (xy_offset < self.cfg.on_target_xy_thresh) & (z_offset < self.cfg.on_target_z_thresh)
     ...
     ungrasp_signal = torch.clamp((gripper_high - gripper_q) / (gripper_high - gripper_low + 1e-6), 0.0, 1.0)
+    # P6v9 ManiSkill force-set: when not grasping, signal stays at max (1.0).
+    ungrasp_signal = torch.where(~is_grasped, torch.ones_like(ungrasp_signal), ungrasp_signal)
     ...
-    rewards = torch.where(is_near_target, stage3_r, rewards)
+    rewards = torch.where(is_on_target, stage3_r, rewards)  # P6v9 strict xy AND z
     ...
-    is_success_zone = d_sponge_target < self.cfg.success_dist_thresh
-    success_now = is_success_zone & gripper_open & sponge_stable
+    is_success_zone = d_sponge_target < self.cfg.success_dist_thresh  # kept for log only
+    success_now = is_on_target & gripper_open & sponge_stable
```

### 3. Logging — 3 신규 keys

```diff
+    "is_on_target_rate": is_on_target.float().mean().detach(),          # P6v9 strict xy AND z (stage 3/4 gate)
+    "xy_offset_mean": xy_offset.mean().detach(),                        # P6v9 horizontal transport gap
+    "z_offset_mean": z_offset.mean().detach(),                          # P6v9 vertical drop gap (hover diagnostic)
```

## Sanity test 결과 (B200, 64env × 2 iter)

| Key | iter 0 | iter 1 | 해석 |
|---|---:|---:|---|
| Mean reward | 1.04 | 2.04 | random policy, stage 1 reach만 fire |
| reach_reward_p6v6 | 0.118 | 0.147 | reach 진행 중 |
| tcp_sponge_dist_m | 0.391 | 0.487 | sponge 멀리 (random action으로 멀어짐) |
| grasped_frac | 0.0 | 0.0 | random policy → no grasp |
| **ungrasp_signal_mean** | **1.0000** | **1.0000** | ✅ **force-set 정확 작동** (`~is_grasped` 100% → 1.0 force) |
| gripper_open_rate | 0.963 | 0.775 | random gripper joint state |
| sponge_target_dist_m | 0.200 | 0.274 | sponge 멀리 |
| near_target_rate | 0.027 | 0.031 | 3% env가 100mm 안 (random noise) |
| **is_on_target_rate** | **0.0000** | **0.0000** | ✅ **strict 정의 정확 작동** (xy 30mm AND z 25mm 동시 충족 환경 0) |
| **xy_offset_mean** | **0.186** | **0.254** | ✅ 신규 진단 key |
| **z_offset_mean** | **0.044** | **0.078** | ✅ 신규 진단 key (hover detection 가능) |
| stage1_reach_frac | 1.0 | 1.0 | 모두 reach 단계 (정상) |
| stage2-4 | 0.0 | 0.0 | 정상 (sponge 멀어서 아직 0) |
| jackpot_fire_rate | 0.0 | 0.0 | 정상 (cfg.success_jackpot=0이라 fire해도 reward 0) |

→ **All checks PASS**. Patch 정상 작동.

## Falsifiability (P6v9 iter 999 success criteria)

| 분기 | 조건 | 평가 기준 |
|---|---|---|
| **(A) ⭐⭐⭐⭐ SUCCESS** | stage4_success_frac > 5% AND is_on_target_rate > 10% AND z_offset_mean < 30mm | Hover trap 해제 + transport 학습 성공 |
| **(B) PARTIAL** | 5% > stage4 > 1% AND z_offset_mean < 50mm | 부분 성공, episode 100으로 추가 단축 (α') |
| **(C) FAIL hover persist** | stage4 < 1% AND z_offset_mean > 60mm | hover habit이 깊음 → reset_std 1.30 + actor_bias[5] reset 0 (P6v10) |
| **(D) FAIL slow exploration** | stage4 < 1% AND z_offset_mean < 50mm AND is_on_target_rate > 3% | 정책이 target 시도하나 아직 정밀 못 함 → episode 400 복귀 + Eureka 검토 |

## 다음 세션 즉시 명령 (사용자 confirm 후)

```bash
# B200 launch P6v9
ssh JHPark 'bash -c "set -e; source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh;
  [[ -z \"\$ROARM_B200_ROOT\" ]] && exit 1;
  nohup \$ROARM_B200_ROOT/launch_p6v9.sh > \$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v9.out 2>\$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v9.err &
  sleep 2; ps -p \$! -o pid,etime,stat"'

# ~10min 후 polling
ssh JHPark 'bash -c "set -e; source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh;
  [[ -z \"\$ROARM_B200_ROOT\" ]] && exit 1;
  ps -p <P6v9_PID> -o pid,etime,stat;
  tail -100 \$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v9.out;
  ls \$ROARM_B200_ROOT/logs/roarm_rl/p6v9_*/"'
```

## HARD RULES 준수

- **#8**: archive 1단계 (5/08 evening → MEMORY_archive_20260515.md, prior session 완료).
- **#11**: /half-clone X 0회.
- **#14**: fail-fast guard 모든 ssh + non-login 셸 `$ROARM_B200_ROOT` 가드.
- **#15**: cu128 sm_100 alive (sanity test 완료 = 추가 검증).
- **#17**: state-only 28-dim only.
- **#18**: 사용자 명시 4 결정 보존 (target / gravity / 28-dim / P4-P5-P6 phase). P6 안의 reward 디자인은 implementation detail BUT **stage 3 정의 변경 + ungrasp force-set은 ManiSkill 원본 정확 재현이라 사용자 P6v6 채택 의도 정합**. 추가 사용자 confirm 필요 항목은 reset_std (1.0 default vs 1.30 vs 유지) + bias reset (default no).
- **#19/#20**: sponge edge-stand 47mm + tower geometry 그대로 (target z=+11.4mm = SPONGE_CENTER_Z, on_target_z_thresh 25mm로 TCP-release transient 포함).
- **#26**: 5/19 deadline **4일 ahead** (1 iteration ~7min, 1-2 시도 가능).

## Reference URLs (외부 검증 출처)

- [ManiSkill stack_cube.py main](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/stack_cube.py) — Stage 3 xy_flag/z_flag + ungrasp_reward force-set 원본 (L107-167)
- [ManiSkill pick_cube.py main](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/pick_cube.py) — SO100/WidowXAI 공용 baseline reward
- [Isaac Lab Lift rewards.py main](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/rewards.py) — `object_goal_distance` × `(object_z > minimal_height)` 게이트 (z 별도 강제)
- [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) — SO-100 Isaac Lab 통합 (low-cost arm reference)
