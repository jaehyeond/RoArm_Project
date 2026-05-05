# Session 2026-05-03 evening — Phase ST-C 1차 PIVOT (Well # pattern v3)

## 0. Trigger

ST-C 1차 5K deploy 진행 중 사용자 cross-check:
> "왜 또 스펀지가 이렇게 누워 있어? 우리 분명 이거 스펀지 다 세워둔 형태로 하기로 했잖아."

User photo proof (7장):
- Yellow sponge 4 angles: edge-stand 3장 + lying flat 1장 (비교 예시)
- Pink # tower 2장: side view + top view (4-sponge edge-stand 2-layer cross)

## 1. Confirmed orientation: EDGE-STAND (47mm tall)

**Sponge dim**: 125mm × 47mm × 22mm (확정 — project_hardware_inventory.md 매치)

**Edge-stand definition**:
- 47mm vertical (table → top)
- 22mm wide on table (footprint width)
- 125mm long horizontal (footprint length)
- Footprint: 125 × 22 mm, top-view aspect 5.7:1

**Excluded interpretations**:
- ❌ Lying flat (22mm tall, 125 × 47 footprint, aspect 2.7:1) — sim_demos_v2 잘못
- ❌ Vertical pillar (125mm tall, 22 × 47 footprint, aspect 0.5:1 TALL) — 4/30 late-evening MEMORY 잘못

## 2. Cross-verification with v6 numerical evidence

| Metric | v6 measured (MEMORY 4/30) | Edge-stand expected | Lying flat expected | Vertical pillar expected |
|---|---|---|---|---|
| TCP grasp z (mid) | +33.6mm (mean) | +23.5mm (mid 47mm sponge) → +33mm은 상부 70% grip | +11mm (mid 22mm sponge) ❌ +22mm off | +62.5mm (mid 125mm pillar) ❌ -29mm off |
| Gripper close cmd | ~25.8° (jaw ~20mm) | 22mm width close ✓ | 22mm thickness close ✓ | 22mm width close ✓ |
| wrist_pitch (grasp) | +74.6° mean (range +59~+87°) | TOP-DOWN OK ✓ | TOP-DOWN +90° required ❌ | LATERAL +0° required ❌ |
| Top-view aspect | 5-6:1 thin bar (사진) | 5.7:1 ✓ | 2.7:1 ❌ | 0.5:1 (TALL) ❌ |

**판정**: Edge-stand이 v6 데이터 + 사용자 사진과 가장 잘 매치. Vertical pillar(125mm tall)는 v6 grasp z +33mm와 정합 안 됨. Lying flat은 사진 aspect와 안 맞음.

## 3. # tower geometry (corrected, photo 5+6 기반)

```
SIDE VIEW (Layer 1 X-axis, Layer 2 Y-axis)        TOP VIEW (#)
                                                      
                                                 Y      
   ━━━━━━━━━━━━━━━━━━━━━━━━━ ← L2 sp4 (z=47-94)  ↑       
   ━━━━━━━━━━━━━━━━━━━━━━━━━ ← L2 sp3            │ ┌─┐ ┌─┐
       ║                ║                         │ │ │ │ │
       ║ ← L1 sp1, sp2  ║                         │═┼═┼═┼═┤  ← L2 (Y-axis)
       ║   (z=0-47)     ║                         │═┼═┼═┼═┤  ← L2
       ║                ║                         │ │ │ │ │  ← L1 (X-axis)
                                                  │ └─┘ └─┘
                                                  └────────→ X
```

### Layer 1 (X-axis, edge-stand, z=0-47mm):
- 2 sponges parallel along X (125mm extent in X)
- Inner gap (사용자 측정): **60-70mm** (Y direction empty space)
- Center-to-center Y: 22 + 65 = **87mm** (NOT 112mm — 4/30 MEMORY 잘못)
- L1.sp1 (X=+280, Y=-43.5), L1.sp2 (X=+280, Y=+43.5) (예시 — base coord, mid-symmetric)

### Layer 2 (Y-axis, edge-stand, z=47-94mm):
- 2 sponges parallel along Y (125mm extent in Y), perpendicular to L1
- Inner gap (사용자 측정): **40-50mm** (X direction empty space)
- Center-to-center X: 22 + 45 = **67mm**
- L2.sp3 (X=+280-33.5=+246.5, Y=0), L2.sp4 (X=+280+33.5=+313.5, Y=0)

### Total tower:
- Height: 94mm (z=0 to 94)
- Footprint: ~125 × ~125mm (extents from layer overlap)

## 4. Decision history (errors traced)

| Date | 결정 | 옳음/잘못 | 손실 |
|---|---|---|---|
| 4/01 | v6 수집 = edge-stand sponge | ✓ | - |
| 4/29 evening | sim_demos_v1 lying-flat 가정 | ✗ | 50 demos × 128fr 폐기 |
| 4/30 morning | "v6 lying-flat 잘못" → upright pivot | ✓ (방향) | - |
| 4/30 afternoon | upright = vertical pillar 125mm tall 가정 | ✗ | 부분 폐기 |
| 4/30 evening | 사용자 정정: "이미지 2처럼 세웠잖아" | ✓ (사용자) | - |
| 4/30 late-evening | Claude "v6 PNG 재분석 → vertical pillar 확정 + 우물정자=lying-flat" 결론 | ✗✗ (사용자 정정 무효화 + 잘못된 재해석) | sim_demos_v2 50ep × 146fr |
| 5/01 | sim_demos_v2 lying-flat # 50 demos | ✗ | 50 demos × 146fr + lerobot_dataset_v6_stacking_v2 (116MB) |
| 5/03 morning | B200 finetune 10K on lying-flat data | ✗ | 5K + 10K ckpt 폐기 + B200 GPU 42min |
| 5/03 ST-C 1차 | 5K deploy → base stationary OOD | ✗ (예측 가능) | logs/v2_5K.csv (분석 가치만 있음) |
| 5/03 evening | 사용자 추가 정정 (7장 사진) → edge-stand 확정 | ✓ | (현재) |

**Root cause** (반복): Claude가 v6 PNG/parquet 재분석으로 사용자 명시 정정을 무효화. Geometric inference > user mandate 우선순위 위반.

## 5. Cleanup actions (사용자 confirm 필요)

```bash
# 폐기 (lying-flat 자세 기반 모든 산출물)
rm -rf sim_demos_v2/                                    # 50 demos lying-flat
rm -rf sim_renders_v4/                                   # 50 ep PNG (2.2GB)
rm -rf lerobot_dataset_stacking_v2/                      # 42MB
rm -rf lerobot_dataset_v6_stacking_v2/                   # 116MB (v6 50ep 보존 위해 별도 추출 후 삭제 필요)
# B200: outputs/smolvla_v6_stacking_v2_b200/checkpoints/{005000,010000} (사용자 confirm 후 삭제, baseline 비교용 보존도 가능)
```

**보존**:
- `lerobot_dataset_v6/` (v6 50ep real, edge-stand sponge — UNCHANGED, 학습 ground truth)
- `outputs/smolvla_v6_b200/checkpoints/last/` (v6-only finetune base, edge-stand 학습됨)
- `roarm_kinematics.py` (FK + IK)
- `sim_to_lerobot_stacking.py` (변환기)
- `merge_v6_stacking.py`

## 6. Phase ST-A v3 design (edge-stand)

### Sub-A3 v3 generate_stacking_demos_v3.py
- SPONGE_LONG=125mm, SPONGE_MED=47mm, SPONGE_SHORT=22mm
- Edge-stand spawn: long_axis on table, med_axis vertical, short_axis on table
  - Layer 1 source: long axis = X direction OR Y direction (random)
  - Layer 1 dest: long axis = X (fixed)
  - Layer 2 dest: long axis = Y (fixed, perpendicular to L1)
- TCP grasp z: target sponge top edge (+47mm) − 14mm clearance = **+33mm world** (v6 mean 매치)
  - Open approach: TCP +60mm (clearance 13mm above sponge top)
  - Close grasp: TCP +33mm (mid-upper, finger pinches 22mm width)
- Gripper convention (4/30 late-evening 확인): state 작은=closed, state 큰=open
  - G_OPEN cmd = 60° (jaw ~45mm, 22mm clearance)
  - G_CLOSE cmd = 5° (jaw ~3mm, mech close grip on 22mm)
- wrist_pitch: ~+75° (TOP-DOWN, v6 mean)
- wrist_roll:
  - Source pickup: random from {0°, 90°} per source orient
  - L1 place: 0° (long axis = X)
  - L2 place: 90° (long axis = Y)

### Layout v3 (단일 base center 예시):
```python
center_x = +280  # mm, base coord
center_y = 0
L1_sp1_y = center_y - 43.5  # = -43.5  (87mm c2c, inner 65mm)
L1_sp2_y = center_y + 43.5  # = +43.5
L2_sp3_x = center_x - 33.5  # = +246.5  (67mm c2c, inner 45mm)
L2_sp4_x = center_x + 33.5  # = +313.5
# All sponges: edge-stand, mid-z = +23.5mm world (47mm tall)
# L2 mid-z = +23.5+47 = +70.5mm world (on top of L1)
```

### Source positions (4-region random per seed, 기존 sim_demos_v2 partition 재사용 가능):
- Slot 0 (좌하): X∈[+150,+250], Y∈[-220,-130]
- Slot 1 (좌상): X∈[+150,+250], Y∈[+70,+200]
- Slot 2 (우하): X∈[+330,+430], Y∈[-220,-100]
- Slot 3 (우상): X∈[+330,+430], Y∈[+50,+200]
- 단, # destination region (X∈[+220,+340], Y∈[-100,+100]) 은 source 제외
- Random orient ∈ {0°, 90°} per source (wrist_roll at pickup)

### Phase 순서 (4-step, 32 anchor, 146 frames):
S1 source0 → L1.sp1 (place at L1, +X end first)
S2 source1 → L1.sp2 (other side L1)
S3 source2 → L2.sp3 (place at L2, perpendicular, -X side)
S4 source3 → L2.sp4 (other side L2)

각 step 8 anchor: above_src → at_src → close → lift → transit → at_dst → open → lift_off
+ HOME_start, HOME_end = 34 anchor

### Sub-A4 stacking_scene_v3.py (Isaac Sim)
- 4 source spawn at 4-region random positions, edge-stand orientation
- 4 ghost markers at # destination positions (cyan, semi-transparent)
- 1층 sponges = X-axis edge-stand
- 2층 sponges = Y-axis edge-stand, +47mm z

### Sub-A5 render_stacking_demos_v3.py
- Held offset TCP+(0,0,-23.5mm) (sponge mid-height under TCP)
- HOLD_INTERVALS_V3: per layout.json anchor frame_map
- 50-demo render → sim_renders_v5/ (~22min, 4090)

### Sub-A6 sim_to_lerobot_stacking_v3.py
- 50 ep × 146 fr → lerobot_dataset_stacking_v3/ (~42MB)
- Task = "Stack four pink sponges into a # pattern"

### Sub-A7 merge_v6_stacking_v3.py
- v6 (50 ep, edge-stand pick) + stacking_v3 (50 ep, edge-stand stack)
- 100 ep × ~14242 frames → lerobot_dataset_v6_stacking_v3/ (~116MB)

## 7. Phase ST-B2 v3 (B200 finetune)

- Base: outputs/smolvla_v6_b200/checkpoints/last/pretrained_model
- Hyper: peak_lr=5e-5, warmup=500, decay_steps=10K, decay_lr=1e-6, batch=64, steps=10K
- save_freq=2500, seed=1000, video_backend=torchcodec
- 5K + 10K ckpt 비교 (5K가 saturate일 가능성 5/03 morning 분석 동일)
- ETA: 42min B200

## 8. Phase ST-C v3 (Deploy)

- 사용자 sponge 4개 edge-stand 배치 (4 region random, 또는 fixed 위치)
- INIT_POS=[0,0,90,0,0,5] (HOME closed)
- `python deploy_smolvla.py --checkpoint outputs/smolvla_v6_stacking_v3_b200/checkpoints/005000/pretrained_model --task "Stack four pink sponges into a # pattern" --start-pos init --max-steps 300 --log-csv logs/v3_5K.csv --save-frames-dir logs/frames_v3_5K`
- CSV state/fk logger fix 적용 완료 (deploy_smolvla.py:849-862)

## 9. Open questions

1. **L1/L2 어느 layer 먼저?** Photo 5에서 bottom = L1 (X-axis). Photo 6에서 위 = L2 (Y-axis). 이대로 진행.
2. **Source random per seed 충분한가?** v6 random sponge 위치 학습 완료. stacking 50 demo 같은 random.
3. **Layer 2 place 시 TCP z?** sponge bottom +47mm + mid 23.5mm = TCP +70.5mm. v6 max grasp z +50mm 초과 → release region z [+19,+421] 안. 학습 가능 추정.
4. **Real deploy 시 L2 sponge가 L1 위로 안전 이동?** sim에서 collision check 필수. 또는 deploy 시 lifted z=+150mm transit 강제.
5. **Sub-A3 IK 재시도 max=200 충분?** v3 dst가 v2와 다름 (gap 작아짐) → IK 가능성 재검증 필요.

## 10. New HARD RULES proposed

- **#18 사용자 명시 정정 > Claude 추론** (절대 우선)
- **#19 Sponge orientation = Edge-stand 47mm tall** (확정)
- **#20 # tower = 2-layer cross stacking** (L1 X-axis + L2 Y-axis edge-stand)

## 11. Continuation prompt (next session)

```
Phase ST-A v3 진입. 4/30 late-evening 잘못 완전 폐기 + edge-stand 재구축.

## HARD RULES 추가 적용 (이번 세션 추가)
- #18 사용자 명시 정정 절대 우선 (Claude 추론 기각 금지)
- #19 Sponge = edge-stand (47mm tall, 22mm wide, 125mm long)
- #20 # tower = L1 X-axis + L2 Y-axis edge-stand cross stacking

## 폐기 대상 (cleanup 후 진행)
- sim_demos_v2/ + sim_renders_v4/
- lerobot_dataset_stacking_v2/ + lerobot_dataset_v6_stacking_v2/
- outputs/smolvla_v6_stacking_v2_b200/ (B200, 사용자 confirm 후)

## Sub-tasks (HARD RULE #17 — sim 작업은 4090 로컬)
1. Cleanup (사용자 confirm)
2. Sub-A3 v3: sim_scripts/generate_stacking_demos_v3.py 신규 작성
   - Edge-stand 47mm tall, TCP grasp z=+33mm world (v6 mean)
   - L1 X-axis (Y c2c=87mm), L2 Y-axis (X c2c=67mm)
   - G_OPEN=60°, G_CLOSE=5°
3. Sub-A4 v3: stacking_scene_v3.py (4 source edge-stand spawn + cyan ghost L1/L2)
4. Sub-A5 v3: render_stacking_demos_v3.py (held offset TCP-23.5mm)
5. Sub-A6 v3: 50-demo render → sim_renders_v5/ (4090, ~22min)
6. Sub-A7 v3: sim_to_lerobot_stacking_v3.py
7. Sub-A8 v3: merge_v6_stacking_v3.py (100ep, edge-stand 일관)

## Phase ST-B2 v3 (B200)
- Base: outputs/smolvla_v6_b200/checkpoints/last/pretrained_model
- 동일 hyper: peak_lr=5e-5, warmup=500, decay=10K, batch=64
- ETA 42min

## Phase ST-C v3 (4090 deploy)
- 사용자 sponge 4개 edge-stand 배치 + Kinect 캡처 검증
- 5K → 10K 비교 deploy
- CSV logger fix 적용 완료

## 사용자 답변 요청 (진행 전)
1. Cleanup 진행 OK?
2. v3 design (Step 6 of session_20260503_st_c1_pivot_well_pattern_v3.md) 확인 OK?
3. # tower 정확 inner gap 재확인 (1층 60-70mm, 2층 40-50mm 맞나요?)

## 중요 reference
- claudedocs/session_20260503_st_c1_pivot_well_pattern_v3.md (full design)
- 사용자 사진 7장 (yellow edge-stand 4 + pink # tower 2)
- claudedocs/v6_sponge_check/ (v6 grasp PNG, edge-stand 확인)
```
