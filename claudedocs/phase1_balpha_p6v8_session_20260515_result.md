# Phase 1.B-α P6v8 결과 polling — α+Fix A 결합은 release intent 2.2× 유도 BUT 성공률 0으로 회귀, 분기 (C) FAIL with transport bottleneck (5/15 session)

## TL;DR

- 🔴 **분기 (C) FAIL 확정 + jackpot 작동 0**: stage4_success_frac **0.0153 → 0.0000** (P6v7 → P6v8, 절대 0으로 회귀), jackpot_fire_rate **0.0000 1000 iter 내내** (fire 0회).
- ✅ **부분 진전 (Fix A jackpot signal 작동 X but α 효과 ON)**: gripper_open_rate **0.0272 → 0.0608** (+2.2×) — release intent 정확히 emerging / stage3_neartgt_frac **0.7634 → 0.6450** (-12%p) — hover dominance 부분 해소.
- 🔴 **Root cause: TRANSPORT BOTTLENECK 신규 진단**: agent가 sponge를 target까지 50mm 안까지 운반 못함 (is_success_zone_rate=0.88%, sponge_target_dist mean=**120mm > 50mm × 2.4**). jackpot은 zone 진입 시점에 fire하므로 transport 실패 = jackpot fire 0 = stage 4 success 0. **P6v6/v7 hold-path 문제는 release path cliff였지만, P6v8은 transport reach cliff로 이동**.
- 🔴 **회귀 원인 (정의 artifact + 실질 회귀 절반씩)**: ① stage 4 정의 100mm→50mm 변경으로 정의가 5× 더 strict (artifact) / ② 그러나 sponge_target_dist 110mm→120mm 실제로 +10mm 멀어짐 (실질, +9%) + grasped 93→86%로 -7%p drop (실질).
- ✅ **사용자 confirm 대기**: 분기 (C) 매핑 fallback 4개 후보 — Fix A jackpot 20→50 강화 / α 200→100 추가 / β stage 4 continuous 8→50 / γ transport shaping (stage 3 안 d-shaped term).
- ✅ B200 학습 7:03 wall @ 247-260K steps/s, 22 ckpts.

## B200 학습 종료 (PID 2159872)

| 항목 | 값 |
|---|---:|
| Wall time | **7:03** |
| Steps/s | 247-260K |
| Total timesteps | 98.3M |
| Iters | 1000 |
| Envs | 4096 |
| Ckpts | 22 (model_0~999, 50 iter step) |
| experiment_name | `p6v8_alpha_fix_a_resumeP6v7` |
| Resume from | P6v7 model_999 (no reset_std use ckpt 0.96, no reset_actor_bias, entropy_coef 0.001) |
| episode_length_s | **2.0 (=200 step)** ✓ α applied |
| Final std | **0.86** (0.96 → 0.86, P6v7 더 빠른 entropy collapse 진행) |
| 종료 시각 | 2026-05-12 09:58:37Z (KST 18:58) |

## P6v7 vs P6v8 iter 999 비교 (결정적 evidence)

| 지표 | P6v7 (iter 999) | P6v8 (iter 999) | Δ | 해석 |
|---|---:|---:|---:|---|
| Mean reward | 2221.06 | **1005.54** | -55% | episode 400→200 절반이라 reward 절반 정합 |
| action_std | 0.96 | **0.86** | -0.10 | entropy collapse 가속 |
| value_function loss | (이전 미기록) | **70.81** | — | 매우 높음 → reward 분포 변화로 value function 재학습 중 |
| episode length | (400) | **199.00** | -50% | α 정확 적용 ✓ |
| tcp_sponge_dist (m) | 0.0135 | 0.0246 | +11mm | 약간 회귀 (reach 단계 14% 차지 늘어남) |
| **sponge_target_dist (m)** | **0.1103** | **0.1203** | **+10mm (+9%)** | **🔴 transport 회귀 (예측: 더 가까이 가야하는데 멀어짐)** |
| sponge_height (m) | 0.0916 | 0.0882 | -3.4mm | hover 비슷 (~9cm) |
| grasped_frac | 0.9333 | **0.8627** | **-7.1%p** | **🔴 grasp 안정성 ↓ (α 단축 영향)** |
| was_grasped_rate | 0.9336 | 0.8627 | -7.1%p | latch 정상 작동 (= grasped) |
| **gripper_open_rate** | **0.0272** | **0.0608** | **+2.2×** | **✅ release intent emerging (Fix A 의도대로 작동)** |
| sponge_grounded_rate | 0.0038 | 0.0071 | +0.3%p | noise 수준 |
| sponge_stable_rate | 0.2614 | 0.2696 | +0.8%p | 비슷 |
| near_target_rate | 0.7771 | 0.6450 | -13%p | α 영향 (짧은 episode에서 near 도달 시간 부족) |
| **is_success_zone_rate** (50mm) | (P6v7 X, 신규) | **0.0088** | — | **🔴 0.88%만 50mm zone 진입 — transport 핵심 bottleneck** |
| **jackpot_fire_rate** | (P6v7 X, 신규) | **0.0000** | — | **🔴 1000 iter 내내 1회도 fire 안 함** |
| stage1_reach_frac | 0.0654 | 0.1355 | +2× | reach 단계에 더 머무름 (episode 짧아 진행 늦음) |
| stage2_grasp_frac | 0.1559 | 0.2195 | +6.4%p | grasp-but-not-near 시간 ↑ |
| **stage3_neartgt_frac** | **0.7634** | **0.6450** | **-12%p** | **✅ hover dominance 부분 완화 (α 효과)** |
| **stage4_success_frac** (50mm) | (P6v7=1.53% @ 100mm) | **0.0000** | — | **🔴 절대 0 (정의 stricter + 실질 transport 실패)** |
| place_success_rate | 0.0153 | 0.0000 | — | stage 4와 동일 |
| ungrasp_signal_mean | 0.0616 | 0.1388 | +0.077 | release intent와 정합 |
| static_signal_mean | 0.2471 | 0.2562 | +0.9%p | 비슷 |

## α + Fix A 작동 검증 — 코드 레벨 정확 적용

- **α (episode 400→200)**: ✓ `Mean episode length: 199.00` 정확 (200 step truncation, off-by-one 1 step 정상).
- **Fix A (is_success_zone=50mm 분리)**: ✓ `is_success_zone_rate=0.0088` 신규 키 출력. zone 진입은 0.88%로 매우 낮음.
- **Fix A (jackpot rising edge)**: ✓ `jackpot_fire_rate=0.0000` 신규 키 출력. 그러나 1000 iter 내내 fire 0회 — zone 진입 자체가 너무 드물거나 success_now 추가 조건 (stable AND ~grasped) 충족 안 됨.
- **Action_std reset 없음**: ✓ ckpt 0.96 사용 (0.96 → 0.86 단조 감소).
- ✅ 패치 코드 레벨 정확 적용 검증.

## 분기 매핑 (사용자 명시 P6v8 ablation criteria)

| 분기 | 조건 | 실제 P6v8 | 매핑 |
|---|---|---|---|
| **(A) ⭐⭐⭐⭐ SUCCESS** | stage4 > 5% AND open > 5% AND stage3 < 50% | stage4=**0%**, open=6.08%, stage3=64.5% | **FAIL** (stage4 0% << 5%, stage3 64.5% > 50%) |
| **(B) PARTIAL** | stage4 2-5% AND open 2-5% | stage4=**0%** (< 2%), open=6.08% (> 5%) | **FAIL** on stage4 |
| **🔴 (C) FAIL** | stage4 ≤ 2% OR P6v7과 차이 미미 | stage4=**0%** ≤ 2%, but open +2.2× (변화 있음) | **MATCH on stage4, partial signal on open** |
| **(D) 의외 회귀** | stage4 < 1% OR 학습 발산 | stage4=**0%** < 1% (P6v7 1.53% → 0% 회귀) | **MATCH but largely definition artifact (50mm stricter)** |

→ **결론: 분기 (C) FAIL with PARTIAL signal on release**.
- Primary criteria 미달 (stage4=0%)
- Secondary positive: gripper_open_rate +2.2×, stage3 dominance -12%p (release intent emerging, hover 완화)
- jackpot_fire_rate=0 → jackpot 메커니즘 자체가 engage 안 함 (transport 실패)

## Root cause 재진단 — P6v8 신규 bottleneck

**P6v6/v7 진단 (hold-path globally optimal)**:
- 가설: stage 3 reward (6+/step) > stage 4 transition gain (-2.7/step net loss for release path)
- P6v8 fix: α + Fix A로 release path 압도 시도

**P6v8 결과 → 진단 갱신 (TRANSPORT BOTTLENECK)**:
- α (episode 200) + jackpot (20) + stage 4 continuous (+8/step in 50mm zone)이 reward landscape를 정상으로 reshape
- 정책 입장: **HOLD path vs TRANSPORT path 비교**
  - HOLD path (110mm hover, grasped): stage 3 ≈ 6.20/step × fire 64.5% ≈ **4.00/step avg** → 199 step × 4 ≈ **+796 reward**
  - TRANSPORT path (target까지 50mm 운반 + release + stable):
    - 운반 ~30 step (stage 3 continuous 6.2/step) ≈ +186
    - zone 진입 + release + stable: jackpot 20 + 8/step × 100 step ≈ **+820** (전체 path **+1006**)
  - **TRANSPORT path 보상이 HOLD보다 +210 우세** → 정책이 학습해야 하지만 0회 fire = **physical feasibility issue**
- **신규 가설**: 47mm edge-stand sponge를 50mm 반경 zone에 안정 release하려면 매우 정밀한 drop 필요. 정책이 학습 못 한 이유:
  - ① 50mm zone 진입 자체가 0.88%만 fire → 운반 어려움 (transport reach 단계 cliff)
  - ② zone 진입 시 release 후 sponge가 zone 밖으로 굴러 나갈 가능성 (47mm 높이 unstable)
  - ③ jackpot rising edge는 first fire라 single experience만 PPO에 신호 → credit assignment 어려움

**즉**: P6v6/v7의 release-path cliff는 해소됐지만, **transport-path cliff (sponge_target_dist 120→50mm 단계 reach 능력)가 신규 bottleneck**으로 등장.

## 보상 산수 재검증 (iter 999)

| Component | 산수 | 추정 reward / 199 step |
|---|---|---:|
| Stage 1 reach (13.55%): `2*(1-tanh(5*0.025))` × fire 13.55% | 1.81 × 0.1355 = 0.245/step | +49 |
| Stage 2 grasp (21.95%): 4 + (1-tanh(5*0.12)) × fire 21.95% | (4 + 0.46) × 0.2195 = 0.979/step | +195 |
| Stage 3 near (64.50%): 6 + 0.5×0.14 + 0.5×0.26 × fire 64.5% | 6.20 × 0.645 = 4.000/step | +796 |
| Stage 4 success (0%): 8 + jackpot 20 × fire 0% | 0/step | +0 |
| Action penalty | -0.0284/step | -6 |
| **합계 estimate** | | **+1034** |
| **실제 Mean reward** | | **+1005** |

→ 산수 정합 (오차 -29 / 0.8%, value_function loss 70.8 = 학습 중 fluctuation 범위).

## 다음 fix 후보 (분기 (C) FAIL + transport bottleneck 진단 종합)

| 옵션 | 변경 | 효과 | 위험 | 권장도 |
|---|---|---|---|---|
| **(α' 추가 단축) episode 200→100** | `--episode_length_s 1.0` | stage 3 누적 50% 추가 ↓, 정책이 decisive transport 강제 | reach/grasp 시간 부족 가능 | ⭐⭐ |
| **(Fix A 강화) jackpot 20→50** | `success_jackpot = 50.0` | first-success bonus 2.5×, but jackpot fire 0이라 효과 미지 | jackpot 자체가 fire 안 하면 무효 | ⭐ |
| **(β) stage 4 continuous 8→50** | stage 4 latched reward 8→50 (사용자 명시 fallback) | zone 진입 시 매 step +50 vs hover +6.2 → 강력한 gradient | zone 진입 자체가 0.88%라 실효 ↓ until transport 학습 | ⭐⭐ |
| **🟢 (γ transport shaping) stage 3 안 d-shaped term** | stage 3 reward를 `6 + 2*(1-tanh(5*d_sponge_target))` 으로 — target 가까울수록 stage 3 reward 자체 ↑ | **무 cliff smooth gradient** within stage 3 → 정책이 자연스럽게 target 가까이 sponge 운반 / cliff 회피 | reward 구조 변경 (HARD RULE #18 사용자 확인 필요) | **⭐⭐⭐⭐ 권장** |
| (δ actuator/init γ fix) | stiffness/damping/effort/init q 정합 (5/13 evening Track C) | sim2real transfer 정확도 ↑ but RL 학습 가능성에 직접 영향 X | scope creep | 5/19 후 |

**권장 우선순위 (사용자 confirm 필요)**:

1. **🟢 (γ transport shaping) + (β stage 4 8→50) 결합** — P6v9. stage 3 안에 transport gradient 도입 + stage 4 강화. **resume P6v8 model_999, 1000 iter ~7min**.
2. (β 단독, stage 4 8→50) — γ 위험 회피용 fallback. 효과 약할 가능성.
3. (α' 추가 단축, episode 100) — α 효과 검증 추가 시도. reach 시간 부족 위험.
4. (Fix A 강화, jackpot 50) — jackpot 자체가 fire 0이라 효과 미지.

**Falsifiability** (P6v9 success criteria iter 999, γ+β 결합 시):
- `is_success_zone_rate > 0.05` (현재 0.0088 5× ↑ minimum, transport 학습 검증)
- `jackpot_fire_rate > 0.02` (현재 0 → 비-zero, success path 학습)
- `stage4_success_frac > 0.05` (현재 0 → 5%+, place success 정량)
- `gripper_open_rate > 0.10` (현재 0.06 → 10%+, release 추가 강화)

## 부수 발견 — P6v7→P6v8 회귀 정량 분석

P6v7 stage4_success_frac=1.53% → P6v8 0.00% 회귀의 절반 분해:
- **정의 artifact**: P6v7 stage 4 정의는 `is_near_target` (100mm) → 100mm 안 + grasped + ungrasp + stable. P6v8: `is_success_zone` (50mm) + ~grasped + stable. **5× stricter + 추가 ~grasped 조건**. 만약 P6v8를 P6v7 정의로 측정하면 stage4가 ~1-2% 정도 나올 가능성.
- **실질 회귀**: sponge_target_dist 110→120mm (+10mm worse), grasped 93→86% (-7%p worse). α 단축으로 grasp 안정성 ↓.

따라서 P6v8 fail은 **순수 회귀 아님** (정의 강화 + 실질 일부 회귀 혼합). 그러나 transport bottleneck 진단은 valid (50mm zone 진입 0.88% = 절대 수치).

## HARD RULES 준수

- **#8 archive**: 5/08 evening (A2 #1 patch + chicken-and-egg #2 진단, P6v3 plateau) → `MEMORY_archive_20260515.md` 본문 그대로 이동 + MEMORY.md 한 줄 pointer. ⚠️ 5/14 evening 노트 "8 full bodies, limit 5 violation 3개 잔존" 후 5/15 prepend (+1) - archive (-1) = 9→8 (4 jansson). **다음 세션 추가 archive 후보**: 5/09 (P6v4 cliff effect) / 5/11 (P6v5 launch).
- **#11**: /half-clone X 0회 (Stop hook 거부 시 continuation prompt + MEMORY 업데이트).
- **#14**: 모든 ssh `set -e + source env.sh + [[ -z $ROARM_B200_ROOT ]] && exit 1` 패턴 준수.
- **#15**: cu128 sm_100 alive (학습 7:03 완료 = 추가 검증).
- **#17**: state-only 28-dim only (visual RL X).
- **#18**: 사용자 명시 4 결정 보존 (target / gravity / 28-dim / P4-P5-P6 phase 구조). P6 안의 reward 디자인 + episode length + success zone + jackpot은 implementation detail. **γ transport shaping은 stage 3 reward 구조 변경이라 사용자 confirm 필요**.
- **#19/#20**: sponge edge-stand 47mm + tower geometry 그대로.
- **#26**: 5/19 deadline **4일 ahead** (B200 RL training PoC 진행, 5/19까지 의미 있는 결과물 필요).

## B200 inventory

```
$ROARM_B200_ROOT/logs/roarm_rl/p6v8_alpha_fix_a_resumeP6v7/
  ├── 22 ckpts (model_0 ~ model_999, 50 iter step) — model_999.pt = next P6v9 resume 후보
  ├── git/  (학습 시 코드 snapshot)
  └── events.out.tfevents.1778579478.JHPark-container.2159872.0
$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v8.{out,err}
$ROARM_B200_ROOT/launch_p6v8.sh
$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py  md5=87fb22a7... (5/14 evening α+Fix A 적용)
$ROARM_B200_ROOT/code/roarm_rl/train_ppo.py  md5=4fb9ff1f... (5/14 evening --episode_length_s flag 추가)
```

## 다음 세션 즉시 명령 (사용자 confirm 후 P6v9 launch 시)

```bash
# 1) γ transport shaping patch (사용자 confirm 필요)
# roarm_rl/roarm_stack_env.py 후보 변경:
#   stage 3 reward: 6 → 6 + 2*(1-tanh(5*d_sponge_target))
#   (stage 4 zone 도달 시 stage 3 reward 자체가 8 = stage 4 base와 맞춤)
# 또는 β: stage 4 continuous reward 8 → 50 (단독 변경, 더 적은 위험)

# 2) md5 verify + sanity test PASS (B200 측 mismatch 시 launch 금지)
md5sum roarm_rl/roarm_stack_env.py
ssh JHPark "md5sum \$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py"

# 3) B200 launch
ssh JHPark 'bash -c "set -e; source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh;
  [[ -z \"\$ROARM_B200_ROOT\" ]] && exit 1;
  nohup \$ROARM_B200_ROOT/launch_p6v9.sh > \$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v9.out 2>\$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v9.err &
  sleep 2; ps -p \$! -o pid,etime,stat"'

# 4) 결과 polling (~10min 후)
ssh JHPark 'bash -c "set -e; source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh;
  [[ -z \"\$ROARM_B200_ROOT\" ]] && exit 1;
  ps -p <P6v9_PID> -o pid,etime,stat;
  tail -100 \$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v9.out;
  ls \$ROARM_B200_ROOT/logs/roarm_rl/p6v9_*/"'
```
