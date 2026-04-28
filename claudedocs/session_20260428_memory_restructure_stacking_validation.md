# Session 2026-04-28 (저녁) — Memory Restructure + v6 Stacking Feasibility + Sim Env Validation

## Context
4/28 evening B200 trajectory diff PASS 직후 세션 계승. 사용자 강조: "B200 학습 / 4090 시각화" 분리, Vulkan 신경 안 씀. v6 4/9 Plan 3 SUCCESS 이미 검증됨 (사용자 지적으로 정정).

## 검증 완료 사항

### v6 Stacking Feasibility (parquet 직접 분석, 6942 frames)
| 지표 | 값 | 의미 |
|---|---|---|
| elbow > 90° (위쪽 작업) | 22.3% (1546/6942) | top pick z=+200 in-distribution |
| elbow < 50° (아래 작업) | 33.5% (2329/6942) | bottom pick z=+30 in-distribution |
| z_at_grip_close ep0 | -93mm ESP32 = +29mm URDF world | 테이블 표면(+42mm) 위 sponge 잡음 |
| gripper range | 0.97~88° (mean 24°) | open/close 모두 학습됨 |

**4/24 분석 정정**: "v6 ~50% 재사용"은 z 자체 아닌 **action sequence 종류** 기준이 정확. **Pick은 ✅, Place는 ❌ (v6에 없음), tower context image는 ❌** (single sponge만 학습). z 분포 자체는 stacking 양 영역 모두 커버.

### Sim2Real Gap 정량화 (이미 측정)
- **SigLIP 0.7222 ± 0.016** (48/50 ep ≥ 0.70 GO)
- **Joint replay RMSE 0.43°** (max 1.55° wrist_p)
- **Kinect calibration RMSE 10.13mm**, table plane RMSE 1.24mm tilt 2.5°
- Sim 70% brighter than real (dome light gap)

### Sim Env 진입 가능
- conda env `isaaclab` (Isaac Sim 5.1) ✅
- URDF + Kinect calib + table_plane + sponge_poses 모두 존재
- stacking_scene.py 실행 PASS (PNG 238KB), Layout A/B/Temp 정확

## 변경된 파일

| 파일 | 변경 |
|---|---|
| `CLAUDE.md` | 5섹션 정정 (Camera nuance / Data 100ep→sim co-training / L-F 명시 / Current Status 2/11→4/28 / Reference 최신). 511줄 |
| `~/.claude/.../memory/MEMORY.md` | 76KB→24KB, 211줄→146줄. HARD RULE #17 추가 (B200=학습 / 4090=sim+deploy). Recent 5개 압축. Topic Files 정렬 |
| `~/.claude/.../memory/MEMORY_archive_20260428.md` | 신규. 21 archive sessions 본문 그대로 보존 (HARD RULE #8). 174줄 40KB |
| `sim_scripts/stacking_scene.py` | 2-sponge stack debug (top color 어두운 핑크 + 2mm gap + update 5→30). 첫 렌더는 사용자가 4/27에 만든 것 |
| `sim_renders_v2/stacking_initial_v2.png` | 신규 렌더. 단, single 핑크처럼 보임 — top sponge 시각 미확정 의심 1개 |

## 의심 + 미해결 사항

1. **stacking_scene.py top sponge 시각 미확정** — Top spawn 코드는 정확하지만 PNG는 single 핑크. 가설: robot HOME pose elbow 90° 접힘 → gripper assembly가 sponge stack 가림 OR 카메라 foreshortening. **Non-blocking** (demo 생성 시 별도 script 사용).
2. **Procedural IK 정확도** — RoArm M3 6-DOF는 다중 IK solution. v6 데이터 elbow up branch 확인 필요.
3. **Sim → Real visual transfer** — sim_v1 SigLIP 0.7222는 pick task. Stacking은 새 visual context (tower image), gap 더 클 가능성.
4. **v6 + sim co-training ratio** — 50:50 vs 80:20 미결정.
5. **Place 정확도** — Top sponge B 위에 ±5mm 안에 놓아야 stack 안정.

## 다음 단계 — Phase ST-A → ST-B → ST-C (2-3주)

```
ST-A (1-2일): generate_stacking_demos.py 작성
├── RoArm M3 6-DOF analytical IK (numpy)
├── 24 waypoint × 50 demos × 30fps ≈ 36K frames
├── Randomization: A/B/Temp xy ±10mm, sponge yaw ±5°
└── sim_to_lerobot.py 변환 → lerobot_dataset_stacking_v1/

ST-B (1.5h B200): finetune
├── v6 ckpt에서 시작
├── Co-training: lerobot_dataset_v6 + stacking_v1 (50:50)
└── batch=64 steps=20000 (v6와 동일)

ST-C (3-7일): Real deploy
├── 4090 + Plan 3 패치 (gripper speed unlock 4/9)
├── A→Temp → A→B → Temp→B 3-step
└── 5-10 시도 정량화
```

### 사용자 결정 5가지 (다음 세션 시작 시 답변 필요)
1. Demo 개수: **50** (권장, v6와 1:1) / 100
2. Curriculum: **바로 C 2-stack** (권장) / A→B→C 단계별
3. Co-training ratio: **50:50** (권장) / 80:20 / 100% sim
4. IK: **numpy analytical** (권장) / isaaclab solver
5. Safety limit z>+148mm: **soft (training)+hard (deploy)** (권장) / hard / soft / 없음
