# Phase ST-A 재설계 — N=4 # (우물정자) Lying Flat Stacking (2026-04-29 ~ 04-30)

> **목적**: ST-C 진입 직전 사용자가 sim 자세 잘못 (upright vertical) 발견 → 재설계 시작.
> **결정**: N=4 lying flat # 우물정자 + 2-stage (build + relocate) curriculum.
> **상태**: Sub-A2 완료 (layout 수치 + 결정 항목 confirm). Sub-A2.5 완료 (코드 재사용성). Sub-A3 (anchor 코드 작성) 직전.

---

## 0. 세션 시작 시점 — ST-C 진입 시도

직전 세션 (4/28~29) 산출물:
- `outputs/smolvla_v6_stacking_b200/checkpoints/last/pretrained_model/` — B200 finetune 1.20GB (4/29 02:53)
- `lerobot_dataset_v6_stacking_v1/` — 합본 100ep 11692 frame 98MB
- ST-A/B 산출물 5개 sim_scripts (`generate_stacking_demos.py` 등)

본 세션 시작 plan: **Phase ST-C real deploy** (Step 1-7 검증 → 4090 ckpt rsync → offline test → real deploy).

---

## 1. ST-C Step 1-2 진행 (성공)

### Step 1 entry verification
- ✅ pwd / git (HEAD=afeb452 untracked=session md만)
- ✅ SSH JHPark Port 47110 + chmod 600 (4/29 18:57 SSH 변경 수용)
- ✅ Azure Kinect / CUDA RTX 4090
- ❌ /dev/ttyUSB* 없음 → 사용자 USB 연결 필요 (Step 4-6 차단)

### Step 2 B200 → 4090 ckpt rsync
- `outputs/smolvla_v6_stacking_b200/checkpoints/last/pretrained_model/` 7 파일 1.20GB
- 45초, 25.24 MB/s
- model.safetensors byte-exact match (B200=1,197,789,224 / 4090=1,197,789,224) ✅
- train_config.json identical (diff PASS, 6340 byte) ✅
- → **legacy upright stacking ckpt, audit-only로 보관**

---

## 2. 사용자 발견 — Sim 자세 잘못 (Pivot 시점)

### 사용자 지적
sim_renders_v3/episode_002/frame_0003.png 보고:
- 현재 sim = sponge upright (length 125mm vertical) 2-stack
- 사용자 의도 = **lying flat + # 우물정자** (sponge가 가로로 누워서 #모양)

물리적 근거:
- Sponge ~5g, lateral force에 무게중심 높은 upright stack 무너짐
- Lying flat # 패턴이 안정적 stacking

### 검증 (cross-verification)

**stacking_scene.py / render_stacking_demos.py 코드**:
- `SPONGE_SIZE_M = (0.022, 0.047, 0.125)` ← Z=125mm (vertical) **upright 확정**

**v6 데이터 ep0 frame 0/30/65/80/102 직접 확인**:
- sponge length 125mm = vertical 방향 (upright)
- gripper top-down 접근, 양 jaw로 thickness 20mm 양옆 핀치 (lateral pinch)
- state[4] (wrist_r) ep0 frame 모두 ~0° (이전에는 "v6 wrist_r=0°만"이라 단정)

**roarm_kinematics.py JOINT_LIMITS_DEG (재검증)**:
- `wrist_r: (-90.0, +90.0)` ← v6 [-60, +84] 명시
- → **v6 실 분포 [-60°, +84°]** = 2층 grip +90° 거의 in-distribution
- **R2 위험 정정**: HIGH → LOW

---

## 3. 사용자 의도 정확 파악 (사진 10장 분석)

사진 10장 모두 # (우물정자) 패턴 보여줌 (top-down 사진 4가 가장 명확):
- 4 sponge lying flat (두께 20mm vertical, 125×47mm 면이 horizontal)
- 1층 = sponge 2개 length 같은 방향, 평행
- 2층 = sponge 2개 length 90° 직교 (1층과 perpendicular)
- 위에서 보면 # 모양, 외곽 footprint ≈ 125×125mm 정사각형

**2-stage 요구**:
| Stage | 내용 | Pick-place |
|---|---|---|
| Stage 1 (build) | 흩어진 4 sponge → #1 우물정 | 4 |
| Stage 2 (relocate) | #1 → #2 옆 새 # stack, robot 자율 순서 | 4 |
| Total | 2-stage long-horizon | 8 |

"Robot 판단" = VLA가 demo에서 본 순서 학습 (자율은 imitation 한계). 물리: 2층 먼저 빼야 1층 안 무너짐.

---

## 4. 결정 사항 (사용자 confirm)

### Q-A2.1~5 권장 default 모두 confirm

| # | 결정 |
|---|---|
| Q-A2.1 # 위치 | #1 (+280, -100) / #2 (+280, +100), 거리 200mm |
| Q-A2.2 Source orient | OPT-A2 — source1/2 length X (1층용) + source3/4 length Y (2층용) |
| Q-A2.3 순서 | **Curriculum** — Stage 1 먼저, ST-C1 검증 후 Stage 2 |
| Q-A2.4 # 간격 | tentative 94mm (center 거리), 빈 공간 47mm = sponge width |
| Q-A2.5 Source 4 위치 | (+220,-180) (+220,-20) (+340,-180) (+340,+20) |

### Curriculum Phase 진행 plan

| Phase | 내용 | Demo | 시간 |
|---|---|---|---|
| ST-A1 | Stage 1 only (4-step build #1) | 50 demos | 1-2일 |
| ST-B1 | B200 finetune Stage 1 only | - | ~1.5h |
| ST-C1 | Real deploy Stage 1 → 70%+ 검증 | - | 1일 |
| ST-A2 | Stage 2 추가 (relocate 4-step) | +50 demos | 1-2일 |
| ST-B2 | Finetune Stage 1+2 합본 | - | ~2h |
| ST-C2 | Real deploy 8-step 완성 | - | 2-3일 |

---

## 5. Sub-A2.5 — 이전 코드 재사용성 (중복 작업 회피)

| 파일 | 재사용 % | 결정 |
|---|---|---|
| **roarm_kinematics.py** | **100%** | 그대로 사용. wrist_r 분포 [-60, +84]° v6-derived 발견 |
| **generate_stacking_demos.py** | 40% | step()/solve_anchors()/interpolate_trajectory()/summarize_demo() 그대로. 24 anchor / upright / 3-step 폐기 → **새 v2 작성** |
| **render_stacking_demos.py** | 60% | Isaac Sim boot/URDF/camera/material 그대로. sponge_state_for_frame + 2-sponge prim 변경 → **새 v2 작성** |
| **sim_to_lerobot_stacking.py** | 100% | 그대로 |
| **merge_v6_stacking.py** | 100% | 그대로 |
| stacking_scene.py | 30% | 새 v2 작성 (4-sponge # spawn) |

**새 작성 필요**: 3개 (`generate_stacking_demos_v2.py` + `render_stacking_demos_v2.py` + `stacking_scene_v2.py`)

---

## 6. v1 → v2 핵심 변경 항목

| 항목 | v1 (upright N=2) | v2 (# lying flat N=4) |
|---|---|---|
| Sponge cube scale | (22, 47, 125) mm Z=length | 1층: (125, 47, 20) X=length / 2층: (47, 125, 20) Y=length |
| Anchor 수 (Stage 1) | 24 (3-step × 8) | 32 (4-step × 8) |
| Pick-place steps | 3 | 4 (source1→layer1.s1, source2→layer1.s2, source3→layer2.s3, source4→layer2.s4) |
| Frames / Demo | 95 | ~130 |
| Anchor structure | (tag, xyz, g_cmd) | **(tag, xyz, g_cmd, wrist_r_deg)** ← 4-tuple |
| Sponge orient | upright Z-up | lying flat Z=thickness 20mm |
| TCP-to-sponge offset (held) | (0,0,-32.5) lateral | (0,0,-10) top-down |
| wrist_r at pick | 0° default | 1층=0° / **2층=+90°** |

---

## 7. 미해결 (Sub-A3 진입 직전)

### 결정 필요
- # 정확 간격 measurement
  - **(A)** 사용자 실측 부탁 (1층 sponge 안쪽 거리 mm)
  - **(B)** Default 47mm 빈 공간 + sim render 시각 검증 → iteration

### Sub-A3+ 진행 시점에 필요
- generate_stacking_demos_v2.py 새 작성 (4-step Stage 1 only)
  - anchor structure 확장 (wrist_r 4번째 인자)
  - solve_anchors에 wrist_r override (2층 anchor에서 +90° 강제)
  - SAFETY_Z_MAX_TRAIN 재계산 (lying flat 2층 top z=+28mm + transit z=+150mm 충분)
- stacking_scene_v2.py (4-sponge spawn 시각 검증)
- render_stacking_demos_v2.py (sponge_state_for_frame lying flat orientation)

---

## 8. 산출물 (이번 세션)

### Code 변경 — 없음 (재설계 plan만)

### Data
- `outputs/smolvla_v6_stacking_b200/checkpoints/last/pretrained_model/` — B200 → 4090 rsync 1.20GB (legacy upright, audit-only)
- `/tmp/v6_ep0_frame{0,30,65,80,102}.png` — v6 frame 추출 (upright 검증용)

### Documentation (this file)
- `claudedocs/session_20260429_phase_st_a_redesign_n4_well_pattern.md`

---

## 9. 다음 세션 진입 (continuation)

**진입 즉시 작업**:
1. 본 md 읽기 (이 파일)
2. 사용자 # 간격 측정 답변 받기 (또는 default 47mm로 시작 = 옵션 B)
3. Sub-A3: `generate_stacking_demos_v2.py` 새 작성 시작
   - 4-step Stage 1 only (build #1)
   - anchor 4-tuple (wrist_r 추가)
   - 2층 anchor wrist_r=+90° clamp
4. Sub-A4: FK/IK + wrist_r=90° in v6 분포 정확 검증
5. Sub-A5: stacking_scene_v2.py 4-sponge spawn + 1-frame Isaac Sim render → 사용자 시각 비교

**HARD RULES 적용**:
- #11 NO /half-clone (이번에도 1회 거부)
- #16 train_config 4090 source-of-truth
- #17 Sim render = 4090 (B200 사용 안 함)

---

**기록자**: Claude Opus 4.7 (1M context). 본 세션은 ST-C 진입 시도 중 사용자가 sim 자세 잘못 발견 → ST-A 재설계로 회귀. Layout/Curriculum/코드 재사용성 결정 완료. Sub-A3 (코드 작성)부터 다음 세션.
