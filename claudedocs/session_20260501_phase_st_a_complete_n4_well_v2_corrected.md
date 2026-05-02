# 2026-05-01 — Phase ST-A 완료 (N=4 # 우물정자 lying flat stacking, V3 corrections 적용)

> **결과**: Sub-A3 ~ Sub-A8 모두 PASS. lerobot_dataset_v6_stacking_v2 116MB, 100 eps × 14242 frames.

## 1. 4/30 late-evening 정정 사항 적용

| 항목 | 4/30 afternoon (잘못) | 본 세션 적용 |
|---|---|---|
| Gripper convention | G_OPEN=+5 / G_CLOSE=+60 (거꾸로) | **G_OPEN=+60 / G_CLOSE=+5** (state 작은=closed) |
| Sponge thickness | 20mm | **22mm** |
| TCP grasp z | +32.5mm world (release region) | **+10mm world** (sponge top edge) |
| wrist_p clamp | +60° | **+80°** (top-down enforce, IK 여유) |
| Source layout | 고정 + jitter | **4 region 랜덤 + sponge body overlap exclusion + IK feasibility 재시도** |

## 2. 핵심 코드 변경 (Sub-A3)

**[sim_scripts/generate_stacking_demos_v2.py](../sim_scripts/generate_stacking_demos_v2.py)** — 완전 재작성:
- 32 step anchor (4 step × 8 anchor) + 2 HOME = 34 anchor total
- Constant 146 frames per demo (HOME bridge=10, transit=8, others 3-4)
- `sample_layout(seed, ws)` → 4 region 랜덤 + IK feasibility 재시도 (max 200)
- `_sponge_body_overlaps_exclusion(x, y, orient)` → sponge body 사각형 vs # footprint 박스 교차 검사
- EXCLUSION_X=(+0.206, +0.354), EXCLUSION_Y=(-0.168, -0.015) (full # footprint + 5mm margin)
- Source orientation 랜덤 ∈ {X, Y}, source wrist_r에 따라 결정
- L1 dst wrist_r=0° / L2 dst wrist_r=+90° (고정)
- Output: trajectory.csv + anchors.csv + **layout.json** (sources, orients, anchor_frame_map 등)

## 3. 50 demos 생성 결과

50 demos / 146 frames each / 7300 frames total
- **IK fails 0** (max err ≤1mm 모두)
- attempt range 0-83 (mean ~25)
- 49.0s 생성 시간

Orientation diversity (seed 0-9):
- 동일 (X→X 또는 Y→Y, no rotation): 일부 seed
- 다른 (X→Y 또는 Y→X, transit 중 90° rotation): seed 1, 2, 3, 5, 6, 7, 8, 9 등

## 4. Sub-A4 시각 검증

**[sim_renders_v4/stacking_initial_seed0_v2.png](../sim_renders_v4/stacking_initial_seed0_v2.png)**:
- 4 pink source sponges 모두 보임 (S1/S2 length-X, S3/S4 length-Y, seed=0)
- 4 cyan ghost markers # destination 패턴
- **Source ↔ # destination 분리 완벽** (sponge body overlap 0)

## 5. Sub-A5 시각 검증 (seed=0, 146 frames, 32.8s)

| Frame | 상태 | 시각 결과 |
|---|---|---|
| f0 | HOME, all 4 at sources | 4 pink sponge ✓ |
| f14 (S1.at_src) | gripper 하강 | 정상 ✓ |
| f17 (S1.close) | gripper close on S1 | 정상 ✓ |
| f30 (S1 transit) | held during transit | sponge held, 다른 3개 source ✓ |
| f36 (S1.open) | placed at L1.sp1 | 1 placed + 3 source ✓ |
| f145 (HOME_end) | complete # 패턴 | 4 sponges in # 형태 ✓ |

## 6. Sub-A6 50-demo full render (background)

- **22 min total** (1325s, 181ms/frame avg)
- 50 episodes × 146 frames = 7300 PNGs
- sim_renders_v4/ 2.2GB

## 7. Sub-A7 LeRobot 변환 (lerobot_dataset_stacking_v2)

- 5m 51s
- 42MB
- 50 eps × 7300 frames
- AV1 video encoding
- **observation.state[0] = [0, 0, 90, 0, 0, 5.0]** = HOME closed ✓
- Task instruction: "Stack four pink sponges into a # pattern"

## 8. Sub-A8 v6 + stacking_v2 merge

- **3.4s** (mp4 stream copy)
- 116MB (75MB v6 + ~40MB stacking_v2)
- **100 eps** (v6 ep 0-49 → out 0-49, stacking ep 0-49 → out 50-99)
- **14242 frames** (6942 v6 + 7300 stacking_v2)
- **2 tasks**: 0=Pick (v6) / 1=Stack four pink sponges # pattern (stacking_v2)
- Stats aggregation bit-perfect
- ds[6942] (first stacking) state=[0, 0, 90, 0, 0, 5.0] ✓

## 9. HARD RULES 준수

- **#5 JOINT_LIMITS try/finally only** ✓ (wrist_p mutation in solve_anchors)
- **#11 NO /half-clone** ✓ (Stop hook 129% 거부)
- **#13 Lenovo cgxr-Legion-Pro-7 / GPU UUID 05b1a3f8** ✓ (모든 sim 작업)
- **#17 Sim render 4090 only** ✓ (Sub-A6 50-demo render = 4090)

## 10. 다음 단계 (Phase ST-B2)

1. **B200 rsync**: lerobot_dataset_v6_stacking_v2/ (116MB, ~1분)
2. **B200 finetune**: 시작점 = `outputs/smolvla_v6_b200/checkpoints/last/pretrained_model`
   - batch=64, steps=10K, fps=30
   - LR: cosine decay
   - HARD RULE #15: nightly cu128 lerobot install 후 강제 upgrade
   - HARD RULE #16: 4090 train_config.json source-of-truth (input_features `observation.images.top` 1개)
   - 예상 시간: ~42분 (B200, 5K도 가능 검토)
3. **Validation**: B200 outputs/smolvla_v6_stacking_v2_b200 → 4090 deploy-equivalent 검증 (5K~10K saturate)
4. **Phase ST-C**: real deploy (USB 연결 후)
   - Step 1-2: SSH JHPark Port 47110 + chmod 600 (이미 완료)
   - Step 4-6: USB 연결 후
   - Stage 1: 우물정자 build (4 source → #1)
   - Stage 2 (선택): #1 → #2 relocate

## 11. 잠재 이슈 / 후속 검토

- **TCP z 최댓값 +343mm** (HOME bridge 10 frames 동안): 11 frames > +180mm 안전 임계 — 실배포 시 JOINT_SPEED_CAPS로 완화 (deploy_smolvla.py 기본 보호)
- **wrist_p +90°** (transit, +3° v6 OOD high): finetune 학습 가능 acceptable
- **Lying flat sponge 53% 압축 (47→22mm)**: foam 가능, 실배포에서 grip 강도 검증 필요

---

**기록자**: Claude Opus 4.7 (1M context). Stop hook 129% context 시점 종료.
