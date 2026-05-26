# Session 2026-05-26 — Track B Cube P0 Plan 4-Agent Cross-Validation + Lock-in

## TL;DR

Track B cube task P0 (cube + gripper calibration) **plan을 4-agent 교차검증 +
2회 self-critique + 사용자 결정 5건으로 확정**. 코드/데이터 변경 없음 — 메모리
docs 4개만 수정/생성. 실제 P0 hands-on 측정 + scripted 스크립트 작성은 미수행
(다음 세션). Track A 영역 미접촉.

## 이번 세션 산출물 (메모리 docs only)

1. `~/.claude/.../memory/tech_gripper_grasp_anchors.md` — **SUPERSEDED 2026-05-26**
   표기 (sponge → cube pivot, HARD RULE #18). sponge linear-fit base만 historical
   reference로 보존.
2. `~/.claude/.../memory/tech_cube_grasp_anchors.md` — **신규**. P0.0~P0.8 확정안
   + world-frame z 정정 + 안전가드 G1-G10 + stacking z 계층 + P1 분포 설계 +
   V6 실패교훈 대책. P0 측정 placeholder (hands-on pending).
3. `~/.claude/.../memory/project_hardware_inventory.md` — cube 30mm×5개 추가
   (2026-05-26 사용자 확인).
4. `~/.claude/.../memory/MEMORY.md` — Pre-Work Checklist에 🟢 Cube+gripper calib
   (P0) line + Topic index cube anchor 추가 + sponge anchor SUPERSEDED 표기.

## 사용자 결정 5건 (확정)

1. **Cube 5개 보유** 확인 → 인벤토리 차단 해소.
2. **P0.2 = Gauge 방식** (arm HOME 고정 + 수동 cube 배치 + gripper cmd만 sweep).
   사유: object-agnostic curve → future 다양한/long/복잡 task에 재사용 (물체는
   width만 측정→cmd 도출). static hold만 측정 → dynamic은 P0.4/P0.7 보완.
3. **Camera = v6 viewpoint 유지** (재calib 불필요). 사유: 카메라·로봇·작업환경
   동일, 물체만 sponge→cube. hand-eye extrinsic + table plane(-12.12mm) 그대로
   유효. P0.1 remount는 "안 움직였나" 1회 sanity (재calib 아님). sponge ckpt
   transfer 이점. HARD RULE #6 "수집 세션 내 고정"에 부합.
4. **IK = joint-angle 직접 우선** + pose_ctrl(SDK 존재하나 미검증)은 P0.1 smoke
   test 후 P0.4 위치 다양화에만 보조.
5. **grasp z tipping 후보 비교** (P0.4 +8/+12/+15mm world) + **P0.5 경량화**
   (z 계층 관찰 기록만, 정밀값은 P1 첫 ep).

## 정정 사항 (검증)

- **cube top z = +18mm world** (table -12.12 + 30). placeholder 초안의 "+30mm
  world"는 table-relative/world 혼동 오류. cube center = +3mm world.
- grasp z = **FK `pose_get()` primary** (robot base frame 직접). Kinect depth는
  hand-eye RMSE 10.13mm vs cube 30mm = 불확도 1/3, 95% CI ±20mm → cube top/side
  z 구분 불가 → secondary sanity only.
- v6 sponge grasp z +33mm → cube +9~15mm (Δ-18~24mm) → wrist_pitch 분포 +14°
  right-shift 예상 (53.8°→~68° mean, data-agent v6 parquet 실측 기반).

## P0 확정 순서 (P0.0~P0.8)

| substep | 제어 | 핵심 | Gate |
|---|---|---|---|
| P0.0 전제 | 수동 | gripper pad 유무(pad시 linear fit 무효) / cube mass / matte | pad 기록 |
| P0.1 HW sanity | scripted | torque→INIT_POS max_diff≤3° + Kinect 1-frame + **cube 가시성**(30mm@224 resize) + remount sanity + **pose_ctrl smoke** | viewpoint+가시+IK 판정 |
| P0.2 jaw sweep | **Gauge** | arm HOME + 수동 cube + cmd 0~40° sweep, object-agnostic curve, cube hold 최소 cmd | hold cmd 확정 |
| P0.3 approach angle | scripted(joint) | wrist_p 75/60/45° + FK z guard(G5)/dist(G6) | ≥1 angle 3/3 |
| P0.4 grasp z | **FK primary** | 후보 +8/+12/+15mm tipping 비교, pose_get() 5회, 2-3 위치 | tipping최소 + 4/5 transport |
| P0.5 stacking z | L-F 시연 | z 4계층 관찰 기록만 | 4계층 기록 |
| P0.6 pyramid jaw 간섭 | scripted | L1 3-cube jaw open clearance | clearance 확인 |
| P0.7 gate | **L-F** | single cube pick 5/5 (pyramid는 P0.5/P1) | **5/5** |
| P0.8 lock-in | 기록 | 본 결과 채움 | — |

- M4 (dataset_mean FK check) → P5 deploy 전 gate로 deferred (추정 mean
  [10,35,55,68,5,15]).
- M5 (slippage 정량화) → P0.4 transport test에 통합.

## 4-Agent 교차검증 핵심 (read-only 분석)

- **A1 Manipulation**: 원안 4결함 — gate가 탐색 전(CRITICAL), L-F는 jaw 정밀
  sweep 불가, grasp z는 angle 후, cube rigid cmd~40° (linear fit 30°+ 미검증).
- **A3 Hardware**: GPU 무관(serial+Kinect). Kinect depth grasp z 측정 신뢰
  불가 → FK primary. remount verify 도구 존재. max_diff 5°→3° 강화.
- **B3 Safety**: 안전가드 G1-G10 (gripper clamp, poll_until_settled, drift 기반
  cube jam 감지, FK z<-130mm guard, SIGINT+atexit torque OFF, cmd간 1.0s delay,
  speed=200 고정/speed=1000 금지). 재사용: gripper_calibrate_v4.py,
  test_phase0_auto_calibrate.py, scan_servos.py(T:106 1회만).
- **data-agent**: cube top +18mm world 정정, wrist_pitch +14° shift, 누락 5측정
  (L2 place z/pyramid jaw/release height/dataset_mean FK/slippage). P1 분포:
  Group A 50 pick (5×5 grid×2, base 5-zone×10, wrist_pitch 78-84° gate), Group B
  200 stacking (5×5×8, full completion), gripper close 0-5° gate. V5 zone bias
  회피.

## 코드 검증 사실

- SDK 메서드: `pose_ctrl`(IK) 존재, `pose_get`(FK), `gripper_angle_ctrl`,
  `joints_angle_ctrl`. deploy_smolvla.py:135 + phase0 모두 pose_get(FK)만 사용,
  pose_ctrl(IK)은 미사용/미검증.
- collect_data_manual.py:64 JOINT_LIMITS gripper (-10,100), :847-848 clamp 로직.
- deploy_smolvla.py:681-689 Plan3 gripper-unlock (gripper_angle_ctrl 별도 호출).
- v6 reference 존재: `collected_data_v6`, `lerobot_dataset_v6` (remount verify용).

## 다음 단계 추천 (step-by-step)

```
[다음 세션 — 신선한 context]
Step 3. P0.0 결과 확인 (pad 유무 → Gauge curve 해석 방향)
Step 4. deploy-agent로 P0.1/P0.2 스크립트 작성 + dry-run
        (safety_p0_guards.py G1-G10 + Gauge sweep + pose_ctrl smoke)
Step 5. hands-on 측정 P0.1→P0.7 순차 (gate마다 PASS)
Step 6. P0.8 lock-in → P1 진입 (250ep = 50 pick + 200 stacking)
```

(Step 1 세션 보존 + Step 2 P0.0 체크리스트 제공은 본 세션에서 수행)

## Track A 경계

본 세션 Track A 미접촉. Track A 최신 truth = v6 close_26 audit FAIL, v7 active
recovery static-ready only, local v7 runtime CUDA blocked (2026-05-26). 별도
세션 영역 (START_HERE Track A regions).

## HARD RULE Compliance

- ✅ #6 camera 고정 (v6 viewpoint 유지 결정).
- ✅ #11 `/half-clone` 거부 (121% context에서도 거부, project-state + continuation
  으로 세션 넘김).
- ✅ #13 Follower=USB1, Leader=USB0 (P0.2/0.3 follower만 명시).
- ✅ #18 사용자 명시 정정 우선 (cube pivot, 결정 5건 honored).
- ⚠️ #19/#20/#24 SUPERSEDED 2026-05-26 (cube pivot).

## P0.0 hands-on 체크리스트 (다음 세션 또는 지금)

- [ ] Gripper pad/고무 유무 육안 확인 (있으면 sponge linear-fit base 무효 → curve 재해석)
- [ ] Cube 5개 30mm 실측 (caliper/자) — 정확히 30mm인지
- [ ] Cube mass 측정 (저울, g)
- [ ] Cube 표면 matte vs glossy (glossy면 Kinect NFOV depth 포화 위험)
- [ ] Cube 색상 기록 (Kinect 가시성 + 학습 입력)
