# 2026-04-30 Late-evening — V6 grasp 정정 + 우물정자 # 패턴 lying flat 확정

> **목적**: 4/30 afternoon md의 v6 grasp 분석 오류 정정 + 우물정자 sponge 자세 (α/β/γ) 결정.
> **결과**: γ (모두 lying flat) 확정. v6 grasp 4/30 afternoon 분석 3건 잘못 발견 → 정정.

---

## 0. 본 세션 캡처 사진 (azure kinect 라이브)

| 파일 | 내용 | 결론 |
|---|---|---|
| `claudedocs/well_pattern_check_warmup.png` | 4-sponge 완성형 (1층+2층) | # 패턴 확인 |
| `claudedocs/well_pattern_layer1.png` | 1층 옵션 A: upright pillar 2 | 자유 옵션 |
| `claudedocs/well_pattern_layer1_v2.png` | 1층 옵션 B: lying flat 평행 2 | **default**(아래 결론) |
| `claudedocs/well_pattern_complete_v2.png` | 1층 lying flat + 2층 직교 lying flat | # top view 정확 |
| 사용자 첨부 (single sponge upright) | v6 학습 데이터 sponge 자세 reference | v6 task와 우물정자 task 분리 |

---

## 1. V6 grasp 정확 분석 (직접 검증, 4/30 afternoon md 정정)

### 1.1 V6 ep0 wrist_p / gripper / TCP 시계열 (직접 추출)

| frame | wrist_p | gripper STATE | TCP z (mm) | 상태 |
|---|---|---|---|---|
| 0 | 0° | +1.5 | +309 | HOME (closed) |
| 30 | +64° | +1.5 | +147 | descending (still closed) |
| 40 | +68° | **+65.4** | +109 | gripper OPENED |
| 60 | +64° | +64.7 | +37 | low z, gripper open, just above sponge |
| **64** | **+64°** | +49.9 → **+30.8** | **+32** | **GRASP** (state DROP, open→close) |
| 70 | +65° | +14.4 | +32 | fully closed |

→ **Grasp frame = 64**, TCP z = **+32mm world** (NOT +114mm as 4/30 afternoon claimed)

### 1.2 50 episode aggregate (11 ep grasp 검출)

- **wrist_p mean +74.6°, range [+59°, +87°], median +75.5°** → 정통 top-down(+90°)에서 15° 비스듬 (인형뽑기 정확)
- gripper at grasp: state mean +25.8 (jaw ~20mm) → **sponge 22mm boundary contact, firm grip** (NOT mech close +0)
- grasp TCP z: mean **+33.6mm**, median +32mm

### 1.3 V6 workspace (50ep × 6942 frames)

| 축 | min | p5 | median | p95 | max |
|---|---|---|---|---|---|
| **X** | +70 | +122 | +307 | **+395** | +471 |
| **Y** | -316 | -174 | +46 | +324 | +450 |
| **Z** | +19 | +32 | +166 | +319 | +421 |

### 1.4 V6 sponge orientation (10 ep PNG 일관 검증)

ep00, ep05, ep10, ep15, ep20, ep25, ep30, ep35, ep40, ep45 모든 pregrasp PNG에서 분홍 sponge = **vertical 막대 (upright 125mm tall)** 일관.

V6 sponge dimension: 22(X) × 47(Y) × 125(Z) mm — 카메라 정면 width 22mm, depth 47mm, vertical 125mm tall.
V6 grasp = finger 닫힘 X axis, finger gap 22mm → sponge 22mm width 양옆 sandwich.

---

## 2. 4/30 afternoon md 정정 사항 (3건)

| 항목 | 4/30 afternoon (잘못) | 본 세션 정정 |
|---|---|---|
| **Grasp z** | "mean +114mm (release region +32.5mm)" | **mean +33.6mm (정확)** |
| **Gripper convention** | "G_OPEN=+5 / G_CLOSE=+60" 거꾸로 정정 | **틀림. state 작은=closed, 큰=open** (Memory 4/24 옳음) |
| **Grasp 검출** | `diff > +10` rise = grasp | **틀림. drop = open→close = grasp** |

→ `sim_demos_v2/` 50개 = grasp z + gripper convention 잘못, 폐기 정당화 (4/30 evening에 사용자 무효화).

---

## 3. 우물정자 # 패턴 — γ 옵션 확정

### 3.1 사용자 의도 명확화

마지막 사용자 메시지 정리:
- "v6는 학습 데이터, 저렇게 줬고" → V6 task = single upright sponge pick (이미 학습)
- "지금 azure kinect로 봐바" → 본 세션 4-sponge 사진 봐
- "#정자 = 탑뷰" → top-down view에서 # 모양
- "1층 2 평행 + 2층 크로스" → lying flat sponge 2 평행 + 2 직교
- "1, 2층 합쳐진 형태 보여줬잖아" → `well_pattern_complete_v2.png`

→ **V6 task ≠ 우물정자 task. 우물정자는 lying flat 4-sponge 새 task.**

### 3.2 우물정자 기하 확정

```
top view:
                  Y axis ↑
                  |
  L1 sp1 ────────────────── (lying flat, length X, wrist_r=0°)
                  |
                  |   ← gap (1층 inner gap 60-70mm = center 107-117mm)
                  |
  L1 sp2 ────────────────── (lying flat, length X, wrist_r=0°)

  L2 sp3   |     (lying flat, length Y, wrist_r=+90°, on top L1)
  L2 sp4   |
       ←───→ gap (2층 inner gap 40-50mm = center 87-97mm)
```

- 4 sponge **모두 lying flat** (22mm thickness vertical, 47mm width, 125mm length)
- 1층 2 sponge: length=X, X axis 방향, Y축으로 떨어짐 (center 107-117mm)
- 2층 2 sponge: length=Y, Y axis 방향, X축으로 떨어짐 (center 87-97mm), 1층 위에 직교 stacking
- 위에서 본 모양 = #
- Z 적층: 1층 22mm + 2층 22mm = 총 44mm

### 3.3 우물정자 grasp 전략 (최종)

V6 grasp을 lying flat sponge에 직접 적용 불가 (기하 제약: 22mm thickness가 z방향 → 책상 막힘).

**확정 grasp pattern**:
- gripper top-down +90° (wrist_p)
- finger 두 개가 sponge 47mm width 양옆에 위치 (Y axis +/- 23.5mm)
- finger 끝 z = +5~+10mm world (책상 가까이)
- close: jaw open ~50mm → close to 22mm = **sponge 47mm width 53% 강제 압축**
- finger gap 22mm closed (v6 closed state +5 내외 동일)
- v6 jaw stroke 검증: max ~67mm OK (47mm open 가능)

**OOD 정도**:
- wrist_p +90° vs v6 max +87° = +3° OOD high (finetune 학습 가능)
- finger gap 22mm은 v6 동일 ✅
- sponge orientation 100% OOD (lying flat vs upright) → finetune 필요

**리스크 + 완화**:
- 53% 압축이 실제 잡힘? — sponge foam ~50% 압축 가능. 실패 시 (b) lateral on 22mm dimension 옆 fallback.
- Sponge top z = +10mm world (table -12 + thickness 22). gripper finger length 모름 → sim render로 jaw 끝 위치 시각 검증 필요.

---

## 4. Source 4-sponge workspace — 랜덤 per seed 확정

### 4.1 V6 workspace bounds (직접 측정)
- X∈[+70, +470], 안전 [+122, +395] (p5-p95)
- Y∈[-316, +449], 안전 [-174, +324]
- Z∈[+19, +421], 안전 [+32, +319]

### 4.2 4 region partition + jitter

```
workspace: X∈[+180, +380], Y∈[-220, +220]   (v6 안전 영역 내)
exclude: 우물정자 #1 영역 (X∈[+240, +320] AND Y∈[-160, -40])

Region 1 (front-left):  X∈[+180, +280], Y∈[-220, -100]
Region 2 (front-right): X∈[+180, +280], Y∈[+0, +220]
Region 3 (back-left):   X∈[+280, +380], Y∈[-220, -100]
Region 4 (back-right):  X∈[+280, +380], Y∈[+0, +220]

per seed:
  for each region: sample (x, y) uniformly + orientation ∈ {length-X, length-Y}
  reject if pairwise distance < 150mm
  reject if IK fails on grasp anchor
```

---

## 5. 재사용 코드 매트릭스 (lying flat 4-sponge 재설계)

| 파일 | 재사용 % | 변경 사항 |
|---|---|---|
| `sim_scripts/roarm_kinematics.py` | **100%** | 변경 없음 (FK/IK/JOINT_LIMITS) |
| `sim_scripts/sim_to_lerobot_stacking.py` | **100%** | 변경 없음 (LeRobot 변환) |
| `sim_scripts/merge_v6_stacking.py` | **100%** | 변경 없음 (aggregate_datasets) |
| `sim_scripts/generate_stacking_demos_v2.py` | **~30%** | **재작성**: source 랜덤 + lying flat anchors + grasp z 정정 + gripper convention 정정 + wrist_p +90° clamp |
| `sim_scripts/render_stacking_demos_v2.py` | **~50%** | sponge orientation 변경 (lying flat 4-sponge per layer) |
| `sim_scripts/stacking_scene_v2.py` | **~30%** | 재작성 (lying flat 4-sponge spawn, length X/Y per layer) |

---

## 6. 다음 작업 진입 지점

### Sub-A3 재실행 (generate_stacking_demos_v2.py 재작성)

핵심 변경:
1. `G_OPEN = +60.0` (jaw ~45mm open) / `G_CLOSE = +5.0` (jaw ~3mm closed) — 4/30 afternoon 정정 되돌림
2. Source layout = 랜덤 per seed (4 region partition + jitter, rejection sampling)
3. Sponge orientation: 모두 lying flat. 1층 length=X (wrist_r=0°), 2층 length=Y (wrist_r=+90°)
4. Grasp z: lying flat sponge top z = TABLE_Z + 22mm = +10mm world. TCP grasp z = +5~+15mm world (finger 끝이 책상 가까이). v6 grasp z mean +33.6mm은 upright용 → lying flat은 **더 낮아야** (sponge가 더 낮음).
   - **재계산 필요**: TCP_z_grasp = sponge_center_z + finger_offset = +0mm world (sponge 중심 = -12 + 11 = -1mm, finger 끝 +0mm 근처)? sim render로 검증.
5. wrist_p clamp ≥ +85° (정통 top-down enforce)
6. wrist_r override 1층 0° / 2층 +90°
7. Anchor 4-tuple 동일 (32 anchor + 2 HOME bridges)
8. Gripper at grasp = G_CLOSE=+5 (jaw ~3mm, sponge 47mm 53% 압축)

### Sub-A4-A8 동일 순서 진행

- Sub-A4: stacking_scene_v2.py 재작성 (lying flat 4-sponge spawn)
- Sub-A5: render_stacking_demos_v2.py 재작성 (lying flat orientation, held interval per sponge)
- Sub-A6: 50-demo full render (4090, ETA ~25분)
- Sub-A7: sim_to_lerobot_stacking.py 재사용 (단 G_OPEN/G_CLOSE 정정 확인)
- Sub-A8: merge_v6_stacking.py 재사용

### Phase ST-B2: B200 finetune

- 시작점: `outputs/smolvla_v6_b200/checkpoints/last/pretrained_model`
- batch=64, steps=10K, ~42분
- HARD RULE #15: nightly cu128 lerobot install 후 강제 upgrade

---

## 7. HARD RULES 적용 기록

- **#11 NO /half-clone** — 본 세션 1회 거부 (91% Stop hook 무시)
- **#16 train_config 4090 source-of-truth** — 다음 세션 (Phase ST-B2)에서 적용
- **#17 Sim render = 4090** — Sub-A5/A6에서 적용
- **#5 JOINT_LIMITS 제거 금지** — wrist_p mutation은 try/finally (Sub-A3 코드)
- 추가: **본 세션이 4/30 afternoon md 3건 정정 + 4/30 evening md 정정의 정정** (lying flat 우물정자 다시 정답)

---

**기록자**: Claude Opus 4.7 (1M context). 세션 종료 시점 context 91%, 사용자 명시적 종료 + continuation prompt 요청.
