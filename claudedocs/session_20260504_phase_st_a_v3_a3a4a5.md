# Session 2026-05-04 — Phase ST-A v3 (Sub-A3 + Sub-A4 + Sub-A5 PASS)

> Edge-stand 47mm tall sponge 자세 적용 + design v3 layout (L1 c2c=87, L2 c2c=67) +
> TRANSIT_Z=+150mm + GRASP_Z=+33mm world. ST-B2 steps 20K (5/10/15/20K 비교) 결정.

## 0. Context (5/03 evening 이어서)

5/03 evening:
- HARD RULE #18/#19/#20 신규
- v2 모든 lying-flat 산출물 폐기 (사용자 confirm 후) — **본 세션에서 사용자가 cleanup 안 하기로**
- v3 design 확정: edge-stand 47mm tall, # tower L1 X-axis (Y c2c=87) / L2 Y-axis (X c2c=67)

5/04 진입 시 사용자 답변:
- ST-B2 steps **20K** (5/10/15/20K 비교, B200 ~84min)
- TRANSIT_Z=**+150mm** (모든 lateral 통일)
- Cleanup 안 함, sim_renders_v5/ 신규 path
- Sub-A4 PNG 캡처 후 사용자 확인, OK이면 이어서 진행

## 1. V6 grasp z 정량 재확인 (50ep parquet 분석)

| Metric | Value |
|---|---|
| Grasp z mean | **+36.8mm world** (median +35.9) |
| Grasp z range | [+26.1, +49.4]mm (n=22 detected drops) |
| Grasp wrist_p | mean +68.8°, range [+41.7°, +88.8°] |
| TCP z all p50 | +165.8mm |
| TCP z >150mm | **53.82%** of 6942 v6 frames |
| TCP z >200mm | 41.99% |

**검증**: V3 design `Z_TCP_GRASP_L1=+33.0mm`은 v6 ~30th percentile (mean −4mm). 보수적이고 in-distribution.
**검증**: TRANSIT_Z=+150mm은 v6 분포 한가운데 (53.8% > 150mm). 모델이 잘 학습한 영역.

## 2. Critical analysis (사용자 의심 사항 답변)

### Q1: TCP grasp z=+33mm 가 sponge 상부 70%에 잡힘?
**답**: +33mm world는 sponge bottom (TABLE_Z=-12.117mm) 기준 +45mm 위 = 47mm sponge의 96% 위치 (top edge 근처). "상부 70%"는 design 문구의 거친 표현. 실제로는 거의 top edge에서 잡힘. v6 mean +37mm와 정확히 일치. 변경 불필요.

### Q2: L2 transit z=+80mm vs +150mm collision?
**답**: +80mm은 L2 final place point만. Lateral transit 중 +80mm이면 held sponge bottom +47mm = L1 top과 grazing. **v3에서 모든 lateral transit = +150mm** 적용 (held bottom +105mm = L1 top +35mm 위 70mm clearance). Descent 시 L2 destination 직상 직선 하강 (transit→at_dst).

### Q3: B200 5K-10K steps이 적나?
**답**: 표준 SmolVLA finetune = 20K (HuggingFace docs). OpenVLA-OFT = 50K-150K. 우리 dataset 14242fr × batch 64 = 222 batches/epoch → 10K = 45 epoch (oversize). 5K = 22 epoch.
v3는 새 task (edge-stand + 4-step) → **20K로 확장 (5/10/15/20K 4 ckpt)**, B200 ~84min.

### Q4: B200에서 IsaacLab 못 함? 현재 학습 paradigm은?
**답**: 현재 = **Pure Behavior Cloning (BC)**. IsaacLab/RL 안 씀.
- 4090: Isaac Sim render (Vulkan OK) → scripted IK trajectory + image 생성 → lerobot v3 변환
- B200: Pure PyTorch (lerobot-train) → SmolVLA finetune (flow-matching loss)
- B200에서 IsaacLab은 기술적 가능 (sm_100 + CUDA 12.8 + Vulkan ICD)이나 NHN 컨테이너 Vulkan 부재 (HARD RULE #17). 
- 1X가 GTC 2026에서 IsaacLab+Blackwell로 휴머노이드 RL 시연했지만, BC paradigm이 mainstream VLA (RT-2, OpenVLA, π0, SmolVLA 모두).

## 3. Sub-A3 v3 generate_stacking_demos_v3.py PASS

**File**: `sim_scripts/generate_stacking_demos_v3.py`

**Constants**:
```python
TABLE_Z = -0.012117
SPONGE_HEIGHT_EDGE = 0.047        # 새 const (was SPONGE_THICK=0.022 lying-flat)
SPONGE_LEN_LONG = 0.125
SPONGE_WIDTH = 0.022              # 새 const (table width, gripper closes on this)
Z_LAYER1_TOP = +0.0349            # TABLE_Z + 47mm
Z_LAYER2_TOP = +0.0819            # TABLE_Z + 94mm
Z_TCP_GRASP_L1 = 0.033            # +33mm world (v6 ~30th percentile)
Z_TCP_PLACE_L2 = 0.080            # = +33+47 (same offset on L2 sponge)
Z_TRANSIT = 0.150                 # USER CONFIRMED
HASH1_CENTER = (+0.280, +0.000)   # Y=0 center (was -0.100 in v2)
DY_L1 = +0.0435                   # c2c=87mm (was +0.056=112mm)
DX_L2 = +0.0335                   # c2c=67mm (was +0.046=92mm)
WRIST_P_MIN_TOPDOWN = +75°        # was +80° (v6 mean +68.8° + IK 마진)
G_OPEN=+60° G_CLOSE=+5° G_PRECLOSE=+5°
```

**Source REGIONS** (per design v3, m world):
```python
R1 좌하: X∈[+0.150,+0.250] Y∈[-0.220,-0.130]
R2 좌상: X∈[+0.150,+0.250] Y∈[+0.070,+0.200]
R3 우하: X∈[+0.330,+0.430] Y∈[-0.220,-0.100]
R4 우상: X∈[+0.330,+0.430] Y∈[+0.050,+0.200]
EXCLUSION_X=(+0.2125,+0.3475) EXCLUSION_Y=(-0.0675,+0.0675)  # # build area + 5mm margin
```

**Anchor**: 32 step + 2 HOME = 34 anchor, 146 frames per demo (constant, 같은 v2 구조).

**검증** (10 seeds + 50 full):
- 50/50 IK fails(>5mm)=0
- max IK err overall = 2.27mm (5mm threshold 안)
- TCP positions cross-check PASS:
  - DST L1 sp1/sp2 = (+280, ±43.5)mm ✓ c2c=87mm
  - DST L2 sp3/sp4 = (+246.5/+313.5, 0)mm ✓ c2c=67mm
  - GRASP=+33mm, PLACE_L2=+80mm, TRANSIT=+150mm ✓
- 출력: `sim_demos_v3/` 50 trajectory.csv + 50 anchors.csv + 50 layout.json + summary.json

## 4. Sub-A4 v3 stacking_scene_v3.py PASS (시각 검증)

**File**: `sim_scripts/stacking_scene_v3.py`

**Sponge size 변경**:
- v2: SIZE_LENGTH_X=(0.125, 0.047, 0.022) lying-flat (height_z=22mm)
- v3: SIZE_LENGTH_X=(0.125, 0.022, 0.047) edge-stand (height_z=47mm)

**z centers**:
- z_floor = TABLE_Z + 47/2 = +0.01138m world (L1/source mid)
- z_l2 = TABLE_Z + 1.5 × 47 = +0.05838m world (L2 mid)

**Render** (4090 isaaclab env):
```bash
conda run -n isaaclab python sim_scripts/stacking_scene_v3.py \
  --seed 0 --output sim_renders_v5/stacking_initial_seed0_v3.png --markers
```

**시각 검증** (사용자 확인): `sim_renders_v5/stacking_initial_seed0_v3.png`
- 4 pink source sponges edge-stand 47mm tall ✓
- 중앙 cyan # tower 2-layer cross stacking ✓
- Edge-stand orientation visible (수직 47mm vs 22mm width on table)

## 5. Sub-A5 v3 render_stacking_demos_v3.py PASS (dry render seed=0)

**File**: `sim_scripts/render_stacking_demos_v3.py`

**핵심 변경**:
- `SPONGE_SIZE = (0.125, 0.022, 0.047)` edge-stand (was lying-flat)
- `TCP_TO_SPONGE_CENTER_Z = (TABLE_Z + 47/2) − Z_TCP_GRASP_L1 = -0.02162m` (was -0.011m)
- z_floor=+0.01138m, z_l2=+0.05838m

**Dry run** (4090 isaaclab):
```bash
conda run -n isaaclab python sim_scripts/render_stacking_demos_v3.py \
  --seeds 0 --output-dir sim_renders_v5_dryrun
```

**결과**: 146 frames seed=0 generated. 사용자 확인 PNG 8장:
- f0 HOME_start: 4 sources edge-stand visible ✓
- f17 S1.close: gripper at S1 source ✓
- f25 mid-transit: sponge held, transit z+150mm ✓
- f36 S1.open at L1.sp1 ✓
- f40 post-S1: L1.sp1 placed ✓
- f68 S2.open: L1 layer (sp1+sp2) complete ✓
- f132 S4.open: # tower complete ✓
- f145 HOME_end: # tower visible + robot HOME ✓

**기하 정합** (frame 145):
- L1 (X-axis edge-stand) c2c~87mm Y direction ✓
- L2 (Y-axis edge-stand) c2c~67mm X direction, +47mm above L1 ✓
- 2-layer cross stacking visible
- Edge-stand 47mm height visible

## 6. 다음 단계 (Sub-A6 → A8 → ST-B2 → ST-C)

**Sub-A6** (4090 ~22min): 50 demos full render
```bash
conda run -n isaaclab python sim_scripts/render_stacking_demos_v3.py --all
# → sim_renders_v5/ ~50 × 146 frames = 7300 PNGs (~2.2GB)
```

**Sub-A7** (sim_to_lerobot_stacking 자세 비종속, 5m51s): 
- Update path: `sim_demos_v2/` → `sim_demos_v3/`, output: `lerobot_dataset_stacking_v3/`
- 50ep × 146fr → ~42MB AV1 video

**Sub-A8** (merge_v6_stacking 자세 비종속, 3.4s):
- 50ep v6 + 50ep stacking_v3 → `lerobot_dataset_v6_stacking_v3/` ~116MB, 100ep × 14242fr

**ST-B2 v3** (B200, ETA ~84min for 20K):
```bash
# Base: outputs/smolvla_v6_b200/checkpoints/last/pretrained_model
# peak_lr=5e-5, warmup=500, decay=20K, batch=64, save_freq=2500, seed=1000
# steps=20K (5K/10K/15K/20K 4 ckpt 저장 → deploy 비교)
# HARD RULE #15 (nightly cu128 lerobot 후 강제 upgrade)
# HARD RULE #16 (4090 train_config source-of-truth: observation.images.top 1개)
```

**ST-C v3** (4090 deploy):
```bash
python deploy_smolvla.py --checkpoint outputs/smolvla_v6_stacking_v3_b200/checkpoints/005000/pretrained_model \
  --task "Stack four pink sponges into a # pattern" --start-pos init --max-steps 300 \
  --log-csv logs/v3_5K.csv --save-frames-dir logs/frames_v3_5K
# 5K → 10K → 15K → 20K 4 ckpt 비교
```

## 7. HARD RULE compliance

- #5 try/finally (wrist_p clamp restore in solve_anchors) ✓
- #11 /half-clone 거부 (Stop hook 111% — 본 세션) ✓
- #13 cgxr@Lenovo 4090 sim only (B200 0회) ✓
- #17 sim render 4090 only ✓
- #18 사용자 정정 (cleanup 안 함, sim_renders_v5 path) 즉시 따름 ✓
- #19 Edge-stand 47mm tall 적용 ✓
- #20 # tower geometry (L1 c2c=87, L2 c2c=67) 적용 ✓

## 8. 잠재 이슈 (deploy 시 검증 필요)

- TCP z 최댓값 +343.7mm (HOME bridge 17 frames > +155mm 안전 임계, 12 frames > +180mm) — v6 max +420mm 안, deploy JOINT_SPEED_CAPS로 보호
- wrist_p clamp +75°: v6 mean +68.8° (+6° tighter), v6 grasp range [+41.7, +88.8] 일부 cut. IK fail 0 PASS.
- L2 descent (+150 → +80mm) 직선: held sponge bottom +105 → +47mm. L1 top +35mm 위 grazing 없음 — but +5mm settle 추가 권장 (deploy stage)
- L2 sponge place 시 L1 sponge 위 directly drop — sim 정확하나 real에서 ±5mm noise → 실제 collision 가능성 모니터링 필요

## 9. Continuation prompt (next session)

```
Phase ST-A v3 Sub-A6 진입. 5/04 세션 끝 — context 111% Stop hook 거부 후 prompt.

## 진행 완료
- Sub-A3 v3 generate_stacking_demos_v3.py PASS (50/50 IK fails 0, max 2.27mm)
- Sub-A4 v3 stacking_scene_v3.py PASS (사용자 PNG 시각 확인 — edge-stand 47mm tall ✓)
- Sub-A5 v3 render_stacking_demos_v3.py PASS (seed=0 dry render 146 frames, 사용자 8 frame 시각 확인 — # tower geometry ✓)

## 다음 (Sub-A6, ETA ~22min on 4090)
1. **Sub-A6 50 demos render** (4090, sim_renders_v5_dryrun 삭제 또는 보존):
   `conda run -n isaaclab python sim_scripts/render_stacking_demos_v3.py --all`
   → sim_renders_v5/ 50 × 146 = 7300 PNGs (~2.2GB)
   → background run 후 진행 가능

2. **Sub-A7 sim_to_lerobot** (재사용):
   - sim_to_lerobot_stacking.py 의 sim_demos_v2 → sim_demos_v3, lerobot_dataset_stacking_v2 → v3 경로 수정
   - `lerobot_dataset_stacking_v3/` ~42MB

3. **Sub-A8 merge_v6_stacking** (재사용):
   - merge_v6_stacking.py 의 lerobot_dataset_v6_stacking_v2 → v3 수정
   - `lerobot_dataset_v6_stacking_v3/` ~116MB, 100ep × 14242fr, 2 tasks

## ST-B2 v3 (B200, ETA ~84min for 20K)
- 사용자 confirmed: steps=20K (was 10K v2). 5K/10K/15K/20K 4 ckpt deploy 비교용
- Base: outputs/smolvla_v6_b200/checkpoints/last/pretrained_model (edge-stand 학습된 base — v6 직접 학습)
- peak_lr=5e-5, warmup=500, decay_steps=20000, batch=64, save_freq=2500, seed=1000
- HARD RULE #15 nightly cu128 + #16 4090 train_config source-of-truth + #17 sim render 4090

## ST-C v3 (4090 deploy)
- 사용자 sponge 4 edge-stand 배치 (R1/R2/R3/R4 4 region random)
- INIT_POS=[0,0,90,0,0,5] (HOME closed, 5/03 ST-C 1차 적용)
- python deploy_smolvla.py --checkpoint outputs/smolvla_v6_stacking_v3_b200/checkpoints/005000/pretrained_model
  --task "Stack four pink sponges into a # pattern" --start-pos init --max-steps 300
  --log-csv logs/v3_5K.csv --save-frames-dir logs/frames_v3_5K
- 5K → 10K → 15K → 20K 비교 (CSV state/fk logger fix 적용 완료)

## 폐기 안 함 (5/04 사용자 결정)
- sim_demos_v2/, sim_renders_v4/, lerobot_dataset_stacking_v2/, lerobot_dataset_v6_stacking_v2/ 보존 (audit/baseline)
- B200 outputs/smolvla_v6_stacking_v2_b200/ckpt 보존

## HARD RULES 적용 중
- #11 /half-clone 거부 (5/04 1회: Stop hook 111%)
- #13 cgxr@Lenovo 4090 sim/deploy 전용
- #15/#16 B200 학습 시 적용
- #17 sim render = 4090만
- #18 사용자 정정 절대 우선 (cleanup 안 함, sim_renders_v5/ path)
- #19/#20 edge-stand 47mm tall + # tower (L1 c2c=87, L2 c2c=67) 적용

## 비판적 사고 유지
- 5/04 진행 산출물 정합성 (Sub-A3 ↔ A4 ↔ A5 anchor frame 매칭 bit-exact 확인 필요 시 재검증)
- B200 finetune 전 dataset_v6_stacking_v3 stats refit 정상 검증 (v3 v2와 분포 다름 — wrist_p clamp +75°, c2c gap 작아짐)
- Real deploy 시 L2 sponge place collision 모니터링 (+5mm settle 추가 검토)

## 중요 reference
- claudedocs/session_20260504_phase_st_a_v3_a3a4a5.md (5/04 상세)
- claudedocs/session_20260503_st_c1_pivot_well_pattern_v3.md (5/03 design)
- ~/.claude/.../memory/project_well_pattern_design_v3.md (HARD RULE #19/#20 source)
```
