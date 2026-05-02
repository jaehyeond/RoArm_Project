# Phase ST-A Sub-A3 — generate_stacking_demos_v2.py 작성 + 50 demos PASS (2026-04-30)

> **목적**: N=4 # (우물정자) lying flat stacking 4-step procedural demo 생성기 신규 작성.
> **완료 범위**: Sub-A3 코드 작성 + 50 demo 검증 (6400 frames, IK fails 0).
> **남은 작업**: Sub-A4 stacking_scene_v2.py + Sub-A5 render_stacking_demos_v2.py.

---

## 0. 산출물 (검증 완료)

| 파일 | 크기 | 역할 |
|---|---|---|
| `sim_scripts/generate_stacking_demos_v2.py` | 9.3KB / ~280 lines | 32-anchor pick-place IK + linear interp |
| `sim_demos_v2/demo_NNNN_trajectory.csv` × 50 | 6400 frames total | 128 frame/demo @ 30fps |
| `sim_demos_v2/demo_NNNN_anchors.csv` × 50 | 1700 anchor states | 34 anchor/demo (32 + 2 HOME) |
| `sim_demos_v2/summary.json` | 50 entries | IK stats per seed |

---

## 1. 직전 가정 검증 (모순 발견 → 정정)

### 1.1 V6 sponge orientation 모호성
- 직전 4/29 md "v6 = upright (3중 일치)" vs 사용자 "v6할때 스펀지 눕혀서" 모순
- 본 세션 검증: v6 ep0 frame 50 PNG 직접 확인 + 50ep grasp pose 정량 분석
- **결과: V6 = upright sponge** (사용자 기억 오류 가능)
  - PNG: 분홍 sponge가 vertical stick 형태
  - V6 50ep grasp at CLOSE transition (no offset): TCP z mean **+114mm** range [+74, +194]
  - V6 wrist_p mean +73° (top-down에 가까움), wrist_r mean +17°

### 1.2 직전 측정 오류 정정
- 직전 세션에서 `np.diff(grip_action) < -10`을 close transition으로 잘못 추정
- 실제로는 grip 값 감소 = release transition. close = `diff > +10`
- 정정된 v6 grasp z = mean +114mm (직전 +32.5mm은 release region)

### 1.3 V6 → V2 매핑 결정
- V2 = lying flat # (사용자 의도, 사진 매치, 물리 안정성)
- V6 (upright lateral grasp +114mm) → V2 (lying flat top-down +32.5mm) **OOD -82mm**
- V6 trajectory에 z=+19~+50mm 구간 (release region, frame 65 z=+31mm) 존재 →
  finetune로 "low z grasp+lift" 학습 가능 (v6 release 패턴 reverse)
- **Risk acknowledged + 진행 결정** (사용자 의도 보존)

### 1.4 Gripper 방향 정정
- V1 코드: G_OPEN=60 / G_CLOSE=10 → **거꾸로**
- V6 실제: + 큰 값 = 닫힘 (frame 0=1.5 open, frame 50=64.5 closed)
- V2: G_OPEN=+5 / G_CLOSE=+60 / G_PRECLOSE=+5 (v6 frame 0 매치)

---

## 2. Sub-A3 코드 설계 핵심

### 2.1 Layout (sub-A2 confirm)
```
HASH1_CENTER = (+0.280, -0.100)  # m world
SOURCES_M    = [(+0.220,-0.180), (+0.220,-0.020), (+0.340,-0.180), (+0.340,+0.020)]
DST_L1_SP1   = (+0.280, -0.147)  # length X, wrist_r=0°
DST_L1_SP2   = (+0.280, -0.053)  # length X, wrist_r=0°
DST_L2_SP3   = (+0.233, -0.100)  # length Y, wrist_r=+90°
DST_L2_SP4   = (+0.327, -0.100)  # length Y, wrist_r=+90°
GAP=47mm, footprint X=125 × Y=141 (직사각형)
```

### 2.2 Heights
```
TABLE_Z=-12.117mm, SPONGE_THICK=20mm
Z_TCP_GRASP_L1 = +32.5mm  (1층 grasp/place; v6 release region)
Z_TCP_PLACE_L2 = +52.5mm  (2층 place; +20mm above L1, v6 +1.7mm extension)
Z_APPROACH = +40mm hover, Z_TRANSIT = +150mm
```

### 2.3 Anchor 4-tuple
- `(tag, target_xyz_world, gripper_cmd, wrist_r_deg)`
- 32 anchor (4 step × 8) + 2 HOME bridges = 34 total
- 8 anchor/step: above_src → at_src → close → lift → transit → at_dst → open → lift_off

### 2.4 wrist_p clamp ≥+60° (top-down enforce)
- `solve_anchors_v2()` 내 `roarm_kinematics.JOINT_LIMITS_DEG["wrist_p"] = (+60, +90)` 임시 mutate
- `try/finally`로 원복 보장 (mutation side-effect 방지)
- 효과: IK가 wrist_p +60° 미만 lateral solution 회피
- v6 grasp wrist_p mean +73° 매치

### 2.5 wrist_r override
- IK 후 강제 `q[4] = wrist_r_deg` (gripper 방향 정확 정렬)
- TCP position에 거의 영향 없음 (wrist_r은 TCP frame z축 회전, position 동일)

---

## 3. 비판적 검증 — 발견 + 수정 사항

### 3.1 🔴 Critical bug: HOME_start state mismatch (FIXED)
**증상**: IK가 HOME_start의 target TCP=(+343,0,+343)을 풀 때 wrist_p clamp +60° 강제로 인해
state=[0, 11.8, 43.7, **60.0**, 0, 5] 반환. 실제 HOME=[0,0,90,0,0,5]과 다름.
**Deploy 영향**: 로봇이 HOME에서 시작 → traj frame 0 = [0, 11.8, 43.7, 60, 0, 5] → 즉각 큰 jump motion.
**Fix**: `solve_anchors_v2`에서 `tag in ("HOME_start", "HOME_end")` 분기 → IK 우회 + HOME state 직접 사용.
**검증**: frame 0 = HOME 정확 일치, frame 5 = S1.above_src smooth transition.

### 3.2 🟡 Wrist_r jerk: S2.lift_off (0°) → S3.above_src (+90°) (FIXED)
**증상**: 1층 → 2층 transition에서 wrist_r 0→+90° in 3 inter frames = **30°/frame** jerk.
**Fix**: `get_seg_frames()`에 wrist_r delta 검사 → `delta > base * 15°/frame` 시 frames 늘림.
**결과**: S2→S3 inter frames 3 → **6**, 30°/frame → **15°/frame** (절반). HOME_end bridge wr 90→0 in 5 frames = 18°/frame (acceptable).

### 3.3 🟢 SAFETY z print misleading (Cosmetic, no fix)
- `summarize_demo` "TCP z >+155mm: 8 frame"은 HOME bridges만 (HOME TCP z=+343mm)
- Stacking 영역만 [+32.5, +150.3]mm (전부 SAFETY 안)
- 학습/deploy 영향 없음 — print만 misleading

---

## 4. 50 demo aggregate 분석

| Metric | 값 | v6 분포 비교 |
|---|---|---|
| Total frames | 6400 | (v6 6942) 0.92× |
| Frames per demo | 128 (모두 동일) | - |
| IK max err | mean 0.96mm / max 1.00mm | tol 1mm ✓ |
| IK fails total | **0** | ✅ |
| base | [-41.7, +5.1] mean -17.6 | v6 [-49, +76] in ✓ |
| shoulder | [0, +56.9] mean +39.2 | v6 [-17, +68] in ✓ |
| elbow | [+42.9, +101.4] mean +63.7 | v6 [+9, +126] in ✓ |
| wrist_p | [0, +90] mean +70.8 | v6 [-25, +90] in ✓ (mean grasp +73° 매치) |
| **wrist_r** | [0, +90] mean +42.9 | **+6° OOD high=2800 frames** (acceptable) |
| gripper | [+5, +60] mean +32.5 | direction corrected ✓ |
| TCP z | [+32.5, +343.7]mm | v6 [+18.9, +420.9] in ✓ (z<+19mm = 0) |

---

## 5. 남은 의심 (Sub-A4-A6에서 시각 검증)

| # | 의심 | 검증 방법 |
|---|---|---|
| 1 | TCP-to-sponge offset 정확성 (jaw가 lying flat sponge 두께 20mm 양옆 핀치 가능?) | Sub-A5: 1 frame Isaac render → jaw position 시각 |
| 2 | # 패턴 footprint 정확성 (직사각형 141×125mm) | Sub-A5: top-down view 캡처 → 사용자 사진 비교 |
| 3 | 1층 sponge orientation (length X 정확) vs 2층 (length Y 정확) | Sub-A4: stacking_scene_v2.py 4-sponge spawn 시각 |
| 4 | 2층 place 시 1층 sp1+sp2 양쪽에 동시에 안착 (지지점 4) | Sub-A5: f102 (S4.lift_off) 시각 |

---

## 6. 다음 세션 진입 (Sub-A4-A7)

### Sub-A4: stacking_scene_v2.py 신규 작성
- 4-sponge lying flat # spawn (init layout: source 4 위치)
- 1층용 source 1/2 = length X, 2층용 source 3/4 = length Y orientation
- 1-frame Isaac Sim 부팅 시각 검증

### Sub-A5: render_stacking_demos_v2.py 신규 작성 (40% v1 재사용 — Isaac boot/URDF/camera)
- 1회 Isaac Sim 부팅 + 50 demo 반복
- 4-sponge prim (lying flat orientation, layer-specific length 방향)
- Held interval: TCP+(0,0,-34.6mm) approx (jaw가 sponge 양옆 핀치) — Sub-A5 시각 검증으로 확정
  - HOLD_INTERVALS_V2 = {sponge_idx: [(close_frame, open_frame)] for each of 4 sponges}
  - 각 sponge close/open frame은 anchor index 기반 (S{N}.close → S{N}.open frame 사이 held)

### Sub-A6: 50 demo full render → sim_renders_v4/ (~25분 ETA, 1.5GB 예상)

### Sub-A7: sim_to_lerobot_stacking.py 재사용 100% — sim_demos_v2 + sim_renders_v4 → lerobot_dataset_stacking_v2 변환

### Sub-A8: merge_v6_stacking.py 재사용 100% — v6 + stacking_v2 합본 → lerobot_dataset_v6_stacking_v2 (~98MB 예상)

### Phase ST-B2: B200 finetune (이전 outputs/smolvla_v6_b200/checkpoints/last/pretrained_model 시작점, batch=64 steps=10K, ~42분 ETA)

---

## 7. HARD RULES 적용 기록

- **#11 NO /half-clone** — 본 세션 1회 거부 (88% Stop hook 무시)
- **#16 train_config 4090 source-of-truth** — Sub-A3 단계 적용 안 됨 (학습 X)
- **#17 Sim render = 4090** — Sub-A5/A6에서 적용 (4090 Isaac Sim)
- **#5 JOINT_LIMITS 제거 금지** — wrist_p mutation은 try/finally로 보호 (Sub-A3 코드 안에서만)

---

**기록자**: Claude Opus 4.7 (1M context).
본 세션은 Sub-A3 단일 단계 (코드 작성 + 50 demos 검증) 완료. Sub-A4-7은 다음 세션.
