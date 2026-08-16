# 58th — `g0b_d444` fg1 실행: flying-gripper 13 pose 물리 판별 = **0/13 전패**, 병목 층위가 "접촉 불가"에서 "유지 불가"로 이동

- 날짜: 2026-08-13 KST (57th 직후 동일 날짜 Claude 세션)
- Active case: `g0b_d444` (D444 개시, 사용자 승인 완료 상태에서 boot) — **fg1 물리 실행 완료**
- 이번 case의 신규 변수: `[팔 제거 = 그리퍼 단독 fixed-root articulation]` (1개, 57th prereg 그대로)
- Session progress rule: 실패 가능 물리 실험 **1회 실행** (fg1, 13 pose × 420 steps, SUCCESS 게이트 실패 가능) — 충족.
- 성격: 부트 재검증 → 자산 추출 → 하네스 스모크 2회 → 러너 저작 → **본 실행** → 분석/상태 갱신.

## 1. 부트 재검증 (불일치 0)

HEAD == origin/master == `9cbd959`, working tree clean. START_HERE 57th판 · fg1_prereg.md ·
D444 · 57th doc 정독. attempt3 5-layer SHA **5/5 핀 일치**, `numpy==1.26.0` /
`psutil==5.9.8`, `isaacsim.replicator.grasping` extension.toml SHA `5e599aaf…` 일치
(version 1.0.9).

## 2. `fg1_gripper_only.usd` 추출 (prereg §3)

**방식 = 참조 + 열거 가능한 최소 오버라이드** (복사 0 — 핀된 attempt3 파일에서 조성되므로
verbatim이 구조적으로 보장):

1. `/fg1_gripper` (defaultPrim) → attempt3 `roarm_m3.usd` `</roarm_m3>` 상대경로 참조
2. `active=False`: world·link1~link4·Looks + joints의 q5 외 5개 조인트
3. `root_joint` `physics:body1` → `/fg1_gripper/link5` 재타깃. **조인트 프레임은 미저작
   유지** (attempt3 원본 규약 — 아래 §3-S1의 teleport 앵커 메커니즘의 핵심)
4. link5 xform → identity (root 프레임 == link5 프레임, prereg 요구)
5. gripper_link xform → T_rel = inv(T_l5)@T_gl (float64, 저작값에서 유도;
   t=[-2.94e-9, 0.018821007, 0.052035025], q≈(0.5,-0.5,-0.5,-0.5) — q5 조인트
   localPos0/localRot0와 정합)

- SHA-256: `0e9fc601df9379fabc118eb2495ac0100350ef9931662413c5a2c0f00690dd76`
- 검증 게이트 (러너가 실행 시작 시 재검증, 전부 PASS):
  - (a) hull census: link5 **64 enabled + 1 disabled(legacy)** / gripper_link 동일 ✓
  - (b) q5 조인트 15개 속성 attempt3 대비 **bit-일치 0 diff** (axis Z, limit
    [0, 90.01166534423828]°, stiffness 1.7453292608261108, damping
    0.01745329238474369, maxForce 2.5, maxJointVelocity 179.9087…) ✓
  - (c) 메쉬 참조: attempt3는 외부 파일 참조 없음(internal reference + inline 기하) →
    게이트를 inline points/faceVertexCounts/faceVertexIndices SHA-256으로 적용,
    **조당 66메쉬(시각 1 + 충돌 65) × 2조 전부 일치, diff 0** ✓
- ⚠️ Looks 비활성화로 시각 재질 바인딩이 dangling — 물리 무영향, 렌더 회색 (기록).

## 3. 하네스 스모크 (스크래치패드, 후보 pose 미사용 — fg1 태그 소비 전 리스크 제거)

Smoke 1 (`fg1_smoke_harness.py`, park pose 원격 배치):

- **S1 PASS**: teleport→reparse 후 fixed-root 앵커 — root drift **0.0 m**, link5 3.0e-8 m
  (60 steps). 미저작 프레임 root_joint가 현재 자세에 재앵커됨을 실증.
- **S2/S6: 격리 temp scene 모드의 결함 2건 실증** →
  1차 실행(isolate_simulation=True)에서 (a) 정적 지지면 collider가 temp scene 멤버가
  아니어서 원통이 60 step 만에 **4.96 m 자유낙하**, (b) scene 간 이동(마이그레이션)이
  body 초기상태 스냅샷을 오염시켜 reset이 spawn이 아닌 **낙하 중 자세로 "복원"**.
  → default scene 모드(`physics_scene_path=None`, 공식 지원)로 전환 후 **전부 PASS**
  (S2 obj drift 3.7e-7 m, S6 spawn 복원 3.7e-7 m). **이것이 DEV-2의 증거.**
- **S4 PASS**: q5 pregrasp 상태(JointStateAPI 88.30998496351378°) + drive 유지.
- **S5 PASS**: mid-run 지지면 collider 비활성 → default scene 240 step에서 원통 낙하
  (35 m, vel cap 10 m/s), gripper root drift 0.0.
- **S8 PASS (계측기 캘리브레이션)**: 정지 접촉력 median **0.24358586 N** vs 이론
  m·g **0.24358230 N** — 4자리 일치. (52nd D439 양성 대조와 동일 결론의 재현.)

Smoke 2 (`fg1_smoke2_stepsub_jawcontact.py`):

- **T1 PASS**: physics step-event 구독이 30/30 step 정확 카운트.
- **T2 PASS**: articulation link 접촉이 **body 경로**(`/World/fg1_gripper/link5`)로 귀속
  (조야한 중첩 진단 pose 1회, 후보 pose 아님, close 없음 — 귀속 확인 목적만).

## 4. 러너 `sim_scripts/p17_g0b_fg1_cyld29h50_flying_gripper_grasp_probe.py`

- 사전 preflight (Isaac 밖): 핀 10파일 + extension manifest + env 핀 → **13 pose 구성**:
  - side 8: sdg2 `geometry_mapped_roarm_targets.link5_origin_target_base_m` +
    `R_base_link5_proposal` verbatim (quat 교차검증 max dev 2.2e-16), close = **14.0°**
    (DEV-3: q5_control null → D431 ⑥ 대역 14~22° 최심값; 닫힘 = q5 감소)
  - rim 5: **n8b** `theta_ladder_full_q5` θ∈{6,15,24,29,35} 행 (φ,q5,δ) — D431 ② 인용값과
    1e-9 일치 검증 후 소비. world 매핑: chat=n8.axis_dir(θ,φ) verbatim,
    top_center_link5=[0,0,0.115428+δ], R@chat=-ẑ 최소 회전, 구성 게이트 max err
    **6.7e-16**. close = 행 q5 − 2°.
- **DEV-1 (prereg §4(b) erratum)**: prereg가 인용한 `t3r_n8_tilt_results.json`의 θ별
  argmax-bite 행은 θ=6/15에서 **음수 bite**(−0.1258/−0.0055 mm, 3-anchor 격자 한계)로
  D431 ②가 기술한 행과 모순 — 의도된 행(전체 34-q5 스윕 argmax)은
  `t3r_n8b_tiltmin_results.json`에 존재. n8b 소비 + 양 파일 SHA 핀 + 반대행을
  results에 병기.
- Phases: PREGRASP(open 88.30998496351378°, 60 step) → CLOSE(120 step), dt 1/60,
  `GraspingManager.evaluate_grasp_poses` (close 목표별 4그룹 호출, 전역 순서 보존) →
  **HANG** (manager 밖: 지지면 collider off, 240 step = 30×8 청크, DEV-4).
- Gates: `close_bilateral` = CLOSE 중 같은 step에서 min(F_fixed, F_moving) > 0.01 N /
  `HOLD` = hang 낙하 < 6 mm / SUCCESS = 둘 다.
- 계측: PhysxContactReportAPI threshold 0 명시 저작 (50th-b N-4 교훈) + contact-report
  구독(임펄스/dt) + step-event 구독으로 (pose, phase, step) 태깅.
- D442 lifecycle: 전 산출물 fsync → pre-close sentinel(`fg1_exit_status.txt`) →
  `SimulationApp.close()` 최종.

## 5. 본 실행 결과 — **0/13, `FG1_ALL_13_FAIL_GRIPPER_GEOMETRY_BOTTLENECK_SUPPORTED`**

wall **25.5 s** (물리 13×420 step, CPU PhysX, RTX 렌더 0). 측정 유효 **13/13**
(root 앵커 오차 <1e-6 m, spawn drift 전 pose 0.000 mm, harness selfcheck PASS).

### 5-1. side 8 pose: 전부 `BILATERAL_NO_HOLD` — ★ 병목 층위 이동의 직접 증거

| pose | bilateral peak [N] | bilateral>0.01N steps | post-close 물체 xy 이탈 [mm] | close 종단 q5 [°] |
|---|---|---|---|---|
| side_000 | 5.508 | 11 | 16.0 | 14.00 |
| side_001 | 6.613 | 16 | 18.6 | 14.00 |
| side_002 | 5.576 | 11 | 23.3 | 14.00 |
| side_003 | 5.517 | 12 | 17.9 | 14.00 |
| side_004 | 6.120 | 10 | 77.3 | 14.00 |
| side_005 | 6.147 | 12 | 15.8 | 14.00 |
| side_006 | 5.869 | 12 | 16.7 | 14.00 |
| side_007 | 5.762 | 11 | 16.4 | 14.00 |

- **양측 접촉은 실재한다**: close 중 10~16 step 동안 양 조 동시 5.5~6.6 N.
  55th P13(팔 포함, 동일 sdg2 후보 계열)은 **0/0/0 N** (`NO_BILATERAL_SIDE_CONTACT`).
- **그러나 유지가 안 된다**: 조가 물체 폭 대응각(~17-22°)을 **지나 14.0°까지 완전히
  닫히면서** 원통이 측방으로 밀려남(post-close에도 지지면 위, z≈0.025 m, xy 15.8~77.3 mm
  이탈). 접촉력 프로파일 = 스파이크 후 0 (배출 순간). hang 시작 시 물체는 이미 조 밖
  → 지지면 제거 후 자유낙하 (~35 m, vel cap 10 m/s).
- 해석 경계: 이 배출은 (조 수렴 형상 × 관통 닫힘 목표 14°)의 합작. **"폭에서 정지"
  닫힘 정책(예: 17~20° 목표 + 힘 평형 유지)은 이번 실행이 시험하지 않았다** —
  후속 case 후보 변수.

### 5-2. rim 5 pose: 정적 admission이 실제 닫힘 스윕에서 소멸

| pose | taxonomy | preclose peak [N] | close 중 조 접촉 steps | post-close xy 이탈 [mm] |
|---|---|---|---|---|
| rim_theta06 | PRECLOSE_COLLISION | 0.014 | 0 | 0.0 |
| rim_theta15 | BILATERAL_NO_HOLD | 0.052 | 16 (peak 0.250 N) | 0.2 |
| rim_theta24 | PRECLOSE_COLLISION | 0.052 | 0 | 0.0 |
| rim_theta29 | PRECLOSE_COLLISION | 0.044 | 0 | 0.0 |
| rim_theta35 | PRECLOSE_COLLISION | 0.044 | 0 | 0.0 |

- n8b 행은 **행의 q5에서** 계산된 정적 admission인데, 물리 시행은 PREGRASP가 88.31°로
  열고 CLOSE가 스윕한다(prereg §5 규약). 개방 정착 중 스침(0.014~0.052 N, 게이트
  0.01 초과 → PRECLOSE 분류)이 있었고, **닫힘 스윕 중에는 θ15의 0.25 N 순간 접촉
  외에 조-물체 접촉 자체가 0** — 물체는 스폰에서 사실상 안 움직였다(≤0.2 mm).
- 즉 rim-tilt 정적 양수 bite는 이 phase 규약의 동적 실행에서 파지 기회로 전환되지
  않았다. (기움 각도의 문제인지 phase 규약의 문제인지는 이 실행으로 분리 불가.)

### 5-3. 분기 판정 (prereg §1)

**전 pose 실패 → 분기 (i): 병목 = 그리퍼 기하 — SUPPORTED.**
팔·IK·접근 궤적을 전부 제거하고 기하적으로 이상적인 배치를 부여해도, 동결 attempt3
조 형상은 D29×H50 강체 원통을 (a) side에서는 물었다가 배출하고 (b) rim에서는 물지
못한다. 55th까지의 "접촉이 안 생긴다"(팔 포함)와 결합하면: **팔 층위는 접촉 형성을
막았고, 그리퍼 층위는 접촉 후 유지를 막는다 — 두 층 모두 결함이며 상위 병목은
그리퍼 기하다.** Non-claims (prereg): 실물 파지, IK 도달성, 마찰 현실성, D419 재판정,
닫힘 정책 일반화 전부 주장 안 함.

## 6. D341 완주

- rerun 0.34.1 핀 + save-only 파일 싱크 + footer `rrd verify` PASS + exact entity 15종 +
  timeline 4종(blueprint/log_time/pose_index/global_step) + component 계약 + 고정
  blueprint + `.rbl` + headless 2400×1400 PNG → `validate_rerun_artifact`
  **pass=True errors=[]**.
- **실제 육안 검수** (`fg1_inspection.png`, 5.28 MB): 패널 1(verdict 문서)·패널 3(pose
  00~12 phase/verdict 로그 전부 판독, WARN 라벨이 results와 일치)·패널 4(접촉력 —
  side 8개의 5.5~7 N 사토스 파형 후 0, rim 구간 평탄)·패널 5(hang_drop ≈ 35,000 mm
  수평선) 전부 판독 가능, 수치 데이터와 정합. ⚠️ 한계 2건: 패널 2(3D)는 기본 카메라
  프레이밍이 mm 스케일 기하를 벗어나 실질 판독 불가(39th 전례와 동일 계열 — 정밀
  기하 판독은 RRD 인터랙티브 뷰 필요), 우상단 토스트 3개가 패널 3 일부를 가림.
  결정 판독은 시계열 패널 + results/trace가 담당하므로 판정 영향 없음.

## 7. 산출물 (전부 `g0b_d444/`, forward-only)

| 파일 | SHA-256 (16) | bytes |
|---|---|---|
| fg1_results.json | `f3c2f2e24263a817` | 45,416 |
| fg1_trace.npz | `ad61b8995d6616c9` | 169,244 |
| fg1_timeline.rrd | `93564fba7909c50f` | 604,247 |
| fg1_timeline.rbl | (results manifest) | 55,535 |
| fg1_rerun_validation.json | (동) | 29,562 |
| fg1_inspection.png | (동) | 5,279,243 |
| fg1_gripper_only.usd | `0e9fc601df9379fa` | 2,321 |
| fg1_script.py.txt / fg1_argv.txt / fg1_stdout.log / fg1_exit_status.txt | (동) | — |

- `fg1_failure.json` 부재 = 정상 종료 경로. exit sentinel: `PRE_CLOSE_SENTINEL rc=0`.
- 종료 시점 attempt3 5-layer + fg1 usd SHA 재검증 **6/6 불변** (prereg §3 종료 조건).

## 8. 불변 확인 / 순응

- `g0b_d420` prefix 편집 0 (읽기 전용 소비만: sdg2 candidates, n8b, n8 script, n8 results).
- D427·D429·D431·D441·D443 재판정 0. `g0a_pass=false` 불변. 로봇 0, RunPod 0,
  lerobot-train 0, git commit/push 0 (명시 지시 없음).
- prereg 이탈은 DEV-1~DEV-4로 results.json + 본 doc에 전량 기록 (전부 reactive,
  스모크/파일 증거 기반).

## 9. 다음 결정 경계 (사용자)

fg1은 case의 분기 질문에 답했다. 후속은 전부 사용자 결정:

1. **교수님 보고**: "그리퍼 기하 병목" 증거 패키지 (fg1 + 55th P13 + D430/D441 계보).
2. **fg2 후보 (신규 변수 1)**: 닫힘 정책 변경 — 관통 목표(14°) 대신 폭-정지 목표
   (17~20°) + 유지 검사. side 배출이 정책 산물인지 형상 산물인지 분리.
3. **물체 직경 축소** (D≤20, 40th 경로 ①) — HARD RULE #18, 사용자 명시 결정 필요.
4. rim-pinch 기움 case — 여전히 교수님 컨펌 대기 (BACKLOG, D419/#18).

## 10. 세션 말미 stop-hook /half-clone 거부 (사후 추가)

최종 브리핑 완료 직후 stop hook이 context 207%를 이유로 `/half-clone`을 요구 →
**거부** (HARD RULE #11 + AGENTS.md Context 95% emergency protocol #4). 상태 문서
5종(START_HERE 58th판·LEDGER·D445·본 doc·MEMORY.md 회전+prepend)은 이미 갱신 완료
상태였으므로 추가 마감 작업은 본 §10 기록 + continuation prompt 출력뿐.
누적 거부 카운터: 57th 45회 이후 이번이 **46회 [가정 표기 유지]**.
