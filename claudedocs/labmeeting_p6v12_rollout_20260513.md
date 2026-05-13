# 랩미팅 보충 — P6v12/v13/v14b 정책 rollout 시각화 (2026-05-13)

> **위치**: [labmeeting_5slides_20260512.md](labmeeting_5slides_20260512.md) Slide 4 (실패 작업) 보강 — P6v12/P6v13/P6v14 3행에 대응하는 **정량+정성 영상 evidence**.
> **목적**: "place_success ≈ 0" 같은 수치만으로 안 보이는 **실제 정책 동작 = 3가지 다른 farming pattern**을 정성적으로 보이기. "Reward shaping = farming spot 이동만"의 직접 증거.

---

## 영상 (3개)

| 정책 | 파일 | 크기 | 패턴 |
|---|---|---:|---|
| **P6v12** | `claudedocs/figures/p6v12_rollout/p6v12_rollout.mp4` | 190 KB | close-hover farm (잡고 target까지, release 거부) |
| **P6v13** | `claudedocs/figures/p6v13_rollout/p6v13_rollout.mp4` | 233 KB | high-altitude wander (잡고 15cm+ 들어서 xy 방황) |
| **P6v14b** | `claudedocs/figures/p6v14b_rollout/p6v14b_rollout.mp4` | 170 KB | stage-2 freeze (잡고 100 step 동결, catastrophic forgetting from P6v14a) |

각 영상 공통:
- 200 frame, 6.6s @ 30fps, 1280×720 USD render (Isaac Sim 5.1 headless + replicator RGB annotator)
- 색상 (교수님 지시): 로봇=검정 / 책상=흰 / 배경=회색 / sponge=분홍 (1-sponge)
- 시점: **Kinect calib intrinsics + extrinsics** (`sim_scripts/kinect_calib.yaml`, HFOV 92.91°, eye=(0.72, −0.001, 0.62) m) — `sim_renders_v2/stacking_initial.png` 동일 시점
- 모두 **같은 spawn (+186, −171, +23) mm + 같은 target**에서 출발 → 정책 별로 farming spot이 명확히 다름

원본 trajectory CSV: `claudedocs/figures/{p6v12,p6v13,p6v14b}_rollout/{p6vXX}_trajectory.csv` (각 ckpt → state-only env 200-step rollout).

---

## 정책별 final pose 비교 (frame 197 = reset 직전)

> frame 198-199는 episode truncation 직후 home reset (3 영상 동일) — 의미있는 정책 행동은 frame 197까지.

| 정책 | sponge xyz (mm) | TCP xyz (mm) | target까지 xy (mm) | z (target +11mm 위) | gripper(j5°) | wrist_roll j4° |
|---|---|---|---:|---:|---:|---:|
| **P6v12** | (+272, −54, **+24**) | (+275, −56, +23) | xy=55mm | **+13 mm** (zone 안) | 90 (close) | 124 |
| **P6v13** | (+365, −54, **+111**) | (+366, −57, +113) | xy=89mm | **+100 mm** (zone 밖 부양) | 90 (close) | 146 |
| **P6v14b** | (+327, −111, **+84**) | (+328, −114, +86) | xy=120mm | **+73 mm** (y OOB freeze) | 90 (close) | 137 |

→ **수치는 stage 통과 0이지만, 영상에서 각자 다른 spot에서 잡고 멈춤이 직접 보임**.

---

## 정량 정합 (영상 ↔ training metric cross-verify)

| Metric | P6v12 iter 999 | P6v13 iter 999 | P6v14b iter 999 | 영상에서 확인 |
|---|---:|---:|---:|---|
| `is_on_target_rate` | 0.406 | (B200 미확인) | (B200 미확인) | P6v12 f175 TCP가 target xy 안 ✓ |
| `gripper_open_rate` | 0.064 | — | 0.066 | 3개 모두 close 유지, release 0회 ✓ |
| `stage4_success_frac` | 0.0002 | — | 0.0000 | 3개 모두 200 step 동안 0 fire ✓ |
| `stage2_grasp_frac` | — | — | 0.871 | P6v14b 잡고 t=75~175 정지 ✓ |
| `z_offset_mean` | 0.048 m | — | — | P6v12 f175 sponge z=+25 vs target +11 → +14mm hover ✓ |

→ **수치 + 영상 = 같은 failure mode를 두 view에서 cross-verify**. P6v12 close-hover, P6v13 high-altitude wander, P6v14b stage-2 freeze.

---

## 3-frame 캡션 (각 영상 t=0 / t=100 / t=175)

### P6v12 (close-hover)

| Frame | 상태 | 캡션 |
|---|---|---|
| `frame_0000.png` | Home init | `[0, 0, +90, 0, 0, 0]°`, sponge spawn (+186, −171, +23) mm |
| `frame_0100.png` | Grasped + transport | base→+15°, sponge x: 175→290 mm, j4=39° 회전 |
| `frame_0175.png` | **Close-hover 1.4cm above target** | TCP (+275, −58, +27), gripper closed — **정책 release 거부** |

### P6v13 (high-altitude wander)

| Frame | 상태 | 캡션 |
|---|---|---|
| `frame_0000.png` | Home init | 동일 spawn |
| `frame_0100.png` | **Lift to z=+128mm** | sponge (+313, +109, **+128**) — target 위 12cm 부양 |
| `frame_0175.png` | XY wander @ high-z | sponge (+343, −88, +173) — **target 위 16cm 부양 상태로 xy 방황** |

### P6v14b (stage-2 freeze, catastrophic forgetting)

| Frame | 상태 | 캡션 |
|---|---|---|
| `frame_0000.png` | Home init | 동일 spawn |
| `frame_0100.png` | Grasped, stationary | sponge (+328, −111, +82) — **고정** |
| `frame_0175.png` | **여전히 stationary 100 step째** | sponge (+326, −110, +83) — Δ<3mm in 75 step. grasp_frac=87% 박혀있음 |

---

## 🚨 Render artifact disclosure (교수님 발표용)

영상에서 gripper가 sponge를 약간 비스듬히 잡은 듯한 시각 artifact가 보일 수 있음. **학습 측은 정상**:

| Layer | 상태 | 증거 |
|---|---|---|
| **학습 (policy)** | ✅ 정상 | CSV로 TCP↔Sponge 정합 t=24/75/150 모두 Δxy=2.7~3.4 mm, Δz≤1.0 mm. grasp 위치 깔끔 |
| **환경 (sim env Bug #1)** | ⚠️ 버그 | [roarm_rl/roarm_stack_env.py:961](roarm_rl/roarm_stack_env.py#L961) — `pose7[:, 3:7] = self._sponge.data.root_quat_w[env_ids]` ← grasp 중 sponge quat을 wrist_roll에 propagate 안 함. 학습은 진행되나 sim 내부 sponge가 spawn 자세에서 잘 안 돌아감 |
| **렌더 (cosmetic fix 적용)** | ✅ patched | `render_p6v12_trajectory_replay.py:204,267` — sponge prim AddRotateZOp + 매 프레임 `grasped일 때 j4_deg, else 0` 적용. wrist_roll 따라 sponge가 visually 회전 |

**핵심 disclosure (교수님 발표 시)**:
> "정량적으로 TCP-Sponge 정합 평균 3 mm. 영상에서 어색해 보일 수 있는 부분은 sim env가 wrist_roll을 sponge quat에 propagate하지 않는 버그 (line 961). **렌더 측에서 j4_deg를 sponge Z축 회전으로 visualize 적용해 시각 정합 보강** — env 상태와 다르게 렌더되도록 의도적으로 보정함. 정책 학습 측은 정상이며 env Bug #1은 Phase 4 (multi-sponge L2 90° 회전) 진입 시 패치 예정."

---

## 시사점 — 6번째 reward farming pattern 정성 evidence

랩미팅 5-slide Slide 4 표의 **P6v12 → P6v13 → P6v14** 3행은 모두 "matrix 안 자리만 다른" 6번째 farming pattern. **3 영상 동일 spawn에서 3가지 다른 spot으로 수렴**:

- **P6v12**: stage 3 zone *안* close-hover farm (z=+13mm, xy 55mm)
- **P6v13**: stage 3 zone *밖* outside-zone hover (z=+100mm, xy 89mm) — zone 진입 자체 회피
- **P6v14b**: stage 2 grasp freeze (z=+73mm, xy 120mm OOB) — P6v14a로부터 1-iter catastrophic forgetting

**Pure reward shaping 6회 누적 실증** (P6v9~v14b) → **7번째 P6v14a에서 Phase 0a pre-grasp init (사용자 명시 Option α)으로 cold-start exploration bottleneck 우회 → stage 4 0 → 77.8% 돌파** (영상화 후속 candidate). P6v14c (5/13 새벽 launch)는 release-aware bridge 검증 → iter 0 stage4=36.5% 증명되었으나 PPO 1-iter forget → 5/14 새벽 **Option D BC pivot** 결정.

---

## 재생 방법

```bash
# 영상 직접 재생
xdg-open claudedocs/figures/p6v12_rollout/p6v12_rollout.mp4
xdg-open claudedocs/figures/p6v13_rollout/p6v13_rollout.mp4
xdg-open claudedocs/figures/p6v14b_rollout/p6v14b_rollout.mp4

# frame 직접 확인
ls claudedocs/figures/p6v12_rollout/replay/frame_0{000,100,175}.png

# 재생성 (CSV 보존, frame+MP4만 재인코딩) — 일반화된 인자
conda run -n isaaclab --no-capture-output python -u \
  -m scripts.render_p6v12_trajectory_replay \
  --csv claudedocs/figures/p6vXX_rollout/p6vXX_trajectory.csv \
  --out_dir claudedocs/figures/p6vXX_rollout
```

CSV → frame 재추출 (state-only env policy rollout):

```bash
conda run -n isaaclab --no-capture-output python -u \
  -m scripts.extract_p6v12_trajectory \
  --checkpoint local_ckpts/p6vXX_model_999.pt \
  --out claudedocs/figures/p6vXX_rollout/p6vXX_trajectory.csv
```

---

## 산출물 인벤토리

```
claudedocs/figures/
├── p6v12_rollout/
│   ├── p6v12_trajectory.csv         201줄 (header + 200 step × 14 col)
│   ├── p6v12_rollout.mp4            190 KB, 6.6s @ 30fps, 1280×720
│   ├── replay/                      200 PNG (frame_0000~0199, AddRotateZOp 적용)
│   └── _concat.txt                  ffmpeg concat list
├── p6v13_rollout/
│   ├── p6v13_trajectory.csv
│   └── p6v13_rollout.mp4            233 KB
└── p6v14b_rollout/
    ├── p6v14b_trajectory.csv
    └── p6v14b_rollout.mp4           170 KB

local_ckpts/
├── p6v12_model_999.pt    (기존, close-hover farm)
├── p6v13_model_999.pt    (5/13 scp from B200, high-altitude wander)
└── p6v14b_model_999.pt   (5/13 scp from B200, stage-2 freeze)

scripts/
├── extract_p6v12_trajectory.py        --checkpoint --out 인자만 바꾸면 다른 ckpt에 재사용
├── render_p6v12_trajectory_replay.py  --csv --out_dir 인자만 바꾸면 다른 csv 영상화 (mp4 이름 = out_dir basename)
└── debug_roarm_prim_tree.py           URDF prim tree 진단 (검정 fix 근거)
```

**핵심 fix 기록** (2026-05-13 새벽~점심):
1. **로봇 검정 fix**: Isaac Sim URDF importer가 STL mesh를 fabric backend에 둠 → USD stage에는 Mesh prim 0개. 모든 link visual은 single MDL material `/roarm_m3/Looks/material_silver` (input `diffuse_color_constant`)에 binding. **Fix**: existing silver shader의 `diffuse_color_constant` (0.7, 0.7, 0.7) → (0.03, 0.03, 0.03) 직접 override = 1줄 fix.
2. **카메라 1:1 fix**: 기존 angled top-down → **Kinect calib intrinsics + extrinsics** 사용 (`sim_scripts/kinect_calib.yaml`). `sim_renders_v2/stacking_initial.png` 동일 시점.
3. **Sponge orientation cosmetic fix**: `AddRotateZOp` 추가 + 매 프레임 `grasped일 때 j4_deg, else 0` 적용. env Bug #1 cosmetic 보강.
4. **Script generalization**: `extract_p6v12_trajectory.py --checkpoint --out` 인자화 / `render_p6v12_trajectory_replay.py --csv --out_dir` 인자화. mp4 이름 = `out_dir.name + ".mp4"` 자동 생성.

---

*Last updated: 2026-05-13 오후 (3-video 확장)*
