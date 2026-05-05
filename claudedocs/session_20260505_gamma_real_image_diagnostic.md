# Phase ST-C v3 진단 γ — Real Kinect Vision-Blind 분리 측정

**날짜**: 2026-05-05 evening
**목적**: 5/05 evening 진단 #2 (sim image, vision-blind verdict)이 sim OOD 때문인지, 실제 finetune 손상인지 분리. v6 base는 4/9 5/5 deploy success → real image에서는 vision-active해야 함.
**입력**: 15 real Kinect layouts (HOME pose 고정, edge-stand sponges, S1~S4 + asymmetric)
**모델**: v6 base / v3 5K / v3 10K / v3 20K (4 ckpts)

## TL;DR

1. **이전 가설 정정**: "finetune이 v6 vision capability 전반 손상" → **틀림**
2. **새 정설**: **Base joint selective vision-blindness** by first-grasp = S1 fixed sim distribution
3. **다른 joints**: shoulder/elbow/wrist/gripper는 finetune 후 **σ_vision 증가** (1.3-7×)
4. **ckpt-step 효과 미미**: 5K→20K base σ_vision ~0.96° 고정. 더 학습해도 base blindness 회복 안 됨
5. **β 권장**: Real stacking teleop 30-50ep (varied layouts → 자연스러운 base 다양성) → v3 base blindness 해결 가능

## Real layouts capture

- **N=15** edge-stand sponge layouts, HARD RULE #19 confirmed
- HOME pose [0,2.3,90.8,0.3,0.1,5.4] 고정 (max err 2.3°)
- 4-quadrant + asymmetric + dense/sparse 다양성
- File: `data/real_layouts_20260505_170501/episode_{000-014}/frame_0000.png`
- Pairwise diff mean ~7-10 (random noise ~2-3 baseline 대비 의미있는 variation)
- ⚠️ 일부 frame에 사용자 hand visible (relative comparison엔 영향 없음)

## Diagnostic 결과 (real layouts × 4 ckpts)

### σ_vision FIRST action (deg)

| Joint | v6_base | v3_5K | v3_10K | v3_20K | Δ% (5K vs v6) |
|---|---|---|---|---|---|
| **base** | **2.195** | **0.857** | 0.963 | 0.968 | **39%** ↓↓ |
| shoulder | 0.384 | 0.816 | 0.732 | 0.701 | 213% ↑ |
| elbow | 0.350 | 0.675 | 0.570 | 0.543 | 193% ↑ |
| wrist_p | 0.596 | 0.849 | 0.749 | 0.755 | 142% ↑ |
| wrist_r | 0.376 | 0.764 | 0.698 | 0.674 | 203% ↑ |
| **gripper** | 0.175 | 1.251 | 1.126 | 1.140 | **715%** ↑↑ |

### σ_vision/σ_noise FIRST ratio

| Joint | v6_base | v3_5K | v3_10K | v3_20K |
|---|---|---|---|---|
| base | 2.05 [weak] | **1.05 [BLIND]** | 2.24 [weak] | 2.26 [weak] |
| gripper | 1.25 [BLIND] | 2.39 | 3.31 | 4.67 |

⚠️ Ratio 회복은 σ_noise 감소 때문 (σ_vision은 5K=10K=20K ~0.96° 동일)

### First-action spread (range max-min, deg)

| Joint | v6_base | v3_5K | v3_10K | v3_20K |
|---|---|---|---|---|
| base | **8.17°** | 3.00° | 3.83° | 3.93° |
| gripper | 0.60° | 4.65° | 4.09° | 4.18° |

→ v6 base joint은 8° 범위로 image에 응답, v3는 ~3.5° 범위로 축소
→ v6 gripper은 0.6° 거의 무응답, v3는 4.5° 응답 (stacking은 grip 정밀 필요)

## Root cause 정정

**5/05 evening 결론 (sim image)**: "전체 vision-blind" — **부정확**
**5/05 night 결론 (real image)**: **Joint-selective vision-blindness, base only**

### Why base only?

Sim demo 50/50 first-grasp = S1 fixed (-Y area, base ≈ -10~-20°)
→ 모델이 "image와 무관하게 base를 -방향으로 회전" 학습
→ shoulder/elbow/wrist는 grasp 정확도 위해 image 응답 강화 (+150~700%)
→ base만 image-invariant default가 됨

### 5/05 deploy 행동 정확히 설명

5/05 deploy 3회 systematic 패턴:
- TCP Y -79 ~ -144mm (-Y direction, S1/R3 area)
- TCP z -98 ~ -99mm (deep)
- 50/50 first-visit = R3 area
- gripper toggle 11회

이는 정확히 "base joint = sim default (-방향) + 다른 joints = image-respond grasp" 행동.
즉 **모델이 vision encoder는 사용하나, base 방향만 학습된 default로 고정**.

## β plan implication

### Cause #1 (최우선): first-grasp 분포 단조성
- Sim demos 50/50 = S1 → base joint default 학습
- **해결**: Real teleop 30-50ep with varied first-grasps (자연스럽게 4 quadrant 다 포함)

### Cause #2 (부차): sim-real visual gap
- v6 base가 real image에 응답 (base spread 8.17° = 의미 있음)
- 그러나 base ratio 2.05 = "weak" (sim 1.31보다 좋지만 strong 아님)
- v6 자체도 real에서 vision-active 정도 약함

### 권장 β (B200 활용 극대화)

```
Phase 1: Real teleop stacking 30-50ep (USB1 Follower + Kinect)
    - L-F mode (USB0 Leader + USB1 Follower)
    - Edge-stand sponges, 4 sources → # tower
    - 다양한 source 위치/order (S1/S2/S3/S4 first-grasp 균일 분포)
    - 예상 시간 4-5h
Phase 2: B200 finetune from v6 base
    - Dataset = v6 (50ep) + real_stacking (30-50ep) [merged]
    - Steps 10K (충분, 20K saturate 확인됨)
    - Save_freq 2K (5 ckpts)
    - 예상 GPU 시간 ~1-1.5h
Phase 3: γ re-test
    - 기존 15 real layouts + 5 ckpts
    - base σ_vision 회복 여부 확인 (목표 8°+ → v6 수준)
Phase 4 (옵션): 만약 base 회복 부족 시
    - sim_v4 diverse 100ep 추가 (random first-grasp shuffle)
    - 3-way merged finetune
```

### 권장 거부할 옵션

- **옵션 α (sim diversity only)**: 4090 only 의미 없음 (사용자 지시), v3 finetune 본질 문제 = real visual gap
- **vision encoder freeze**: 복잡도 ↑, 효과 불확실 (현재 v3는 vision encoder 사용 중, 단지 base만 default 학습됨)

## Files

- 진단 결과 4 JSONs: `logs/vision_diag_20260505_172158.json` (v6 base) + `_172241` (v3 5K) + `_172332` (10K) + `_172401` (20K)
- 4 ckpt PNGs: 같은 timestamp `.png`
- 15 real layouts: `data/real_layouts_20260505_170501/episode_*/frame_0000.png`
- Collage: `data/real_layouts_20260505_170501/_collage.png`
- Sample 4: `claudedocs/gamma_layouts_sample_4.png`

## HARD RULE 준수

- **#11** /half-clone X (continuation prompt + MEMORY)
- **#16** train_config source-of-truth (Follower=USB1 검증, joints_angle_get 3회 동일)
- **#17** 4090 (Lenovo 본 PC, B200 0회 — γ는 inference만)
- **#18** 사용자 정정 우선 (edge-stand 확인)
- **#19** Edge-stand 47mm tall 확인
- **#6** Kinect 위치 고정 (15 layouts 동일 viewpoint)
