# `g0b_d444` / `fg1` preregistration — flying-gripper isolation: 그리퍼 단독 물리에서 D29×H50 원통 양측 파지 판별

- Date: 2026-08-13 KST (57th session)
- User authority: 57th 채팅에서 사용자가 "승인할게"로 명시 승인한 "flying-gripper 판별
  실험" (56th 브리핑 옵션 B의 구체안 1). D419 top-down 규약의 변경이 아니다 — 팔 자세
  규약이 아예 적용되지 않는 병목-격리 진단이며, 파지 규약 재판정을 하지 않는다.
- 이번 case의 신규 변수: `[팔 제거 = 그리퍼 단독 fixed-root articulation]` (1개)
- Scope: 물리 실행 O (실패 가능 실험), Isaac Sim O (로컬 RTX 4090, RTX 렌더 불필요 —
  `render=False` 직접 물리), 로봇 하드웨어 0, RunPod 0, lerobot-train 0.

## 1. Decision question / non-claims

- 질문: 팔 기구학·IK·테이블 도달·접근 궤적 제약을 전부 제거했을 때, 동결 attempt3
  그리퍼 기하가 D29×H50 / 24.83 g 원통을 **양측 접촉으로 쥐고 유지**할 수 있는가?
- 분기: (i) 전 pose 실패 → 병목 = **그리퍼 기하** 확정 (하드웨어 개조/물체 변경 논거,
  교수님 보고 자료). (ii) 1개 이상 성공 → 병목 = **팔/포즈/궤적** (기움 case 논거 강화).
- Non-claims: RoArm IK/도달성, 실물 파지 성공, 마찰 현실성 (D441 ⑥ 마찰 결론 금지 유지),
  학습 라벨, D419 규약 판정, force-closure 일반론. 성공해도 "팔로 가능"을 주장하지 않는다.

## 2. Method authority (installed source, version-matched — 57th 감사)

- `isaacsim.replicator.grasping` **1.0.9** (Isaac Sim 5.1.0.0 / Kit 107.3). NVIDIA 공식:
  <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/synthetic_data_generation/tutorial_replicator_grasping_sdg.html>
- 설치 소스 file:line 근거 (57th 직접 열람):
  - 외부 pose 주입 공식 지원: `GraspingManager.evaluate_grasp_poses(grasp_poses=[(loc,quat),...])`
    (`grasping_manager.py:978-1033`); YAML `poses` component 로드도 지원 (`:321-330`).
  - gripper 배치 = root prim xform 텔레포트: `set_gripper_pose` → `set_transform_attributes`
    (`:1150-1160`) ⇒ **그리퍼는 독립 root의 단독 articulation이어야 한다** (팔 내부 서브트리 불가).
  - 직접 물리 + 렌더 off: `simulate_all_grasp_phases(render=False, physics_scene_path=…,
    isolate_simulation=True)` → `simulate_physics_async` (`:666-806`), 임시 복제 물리 씬 격리.
  - phase는 joint drive target만 설정 (`:775-789`) — root 이동/lift 기능 없음.
  - `object_simulation_phases`는 1.0.9에서 선언만 존재 (`:78,:109`) — **안정성 검사 미구현**
    ⇒ 유지(hold) 판정은 §5의 자체 hang-test 게이트로 한다.
  - manager의 `write_grasp_results` YAML (`:1093-1147`)은 최종 joint state만 기록 — 보조
    기록이며, 성공/실패 권위는 자체 게이트 (§5).

## 3. Frozen inputs / pins

- Asset 권위: attempt3 5-layer USD (root/base/physics/robot/sensor,
  `claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/roarm_m3_fullmesh_fixed_point_parts/`).
  full SHA-256은 `g0b_d420/t3s_side_sdg2_config.json`의 핀 그대로 (`a4be58e8…`,
  `ea0ee8f2…`, `043a5d35…`, `2227536f…`, `3f44081f…`) — 실행 시작/종료에 재검증, drift = fatal.
- 파생 자산 (신규, forward-only, 이 case 폴더 안): `fg1_gripper_only.usd` —
  attempt3 robot layer에서 link5 + gripper_link + q5 joint + 기존 drive 파라미터 서브트리를
  **verbatim 추출**하고 root를 link5 프레임 fixed-base articulation으로 저작.
  기하·조인트 수치 변경 0. 추출 검증 게이트: (a) 조당 active convexHull **64+64**,
  legacy collider 각 1 disabled (p14 계약 동일), (b) q5 drive
  stiffness/damping/maxForce/joint limit이 원본 layer 값과 정확 일치, (c) 시각/충돌 메쉬
  참조 파일 SHA 일치. 하나라도 불일치 = fatal, 실행 금지.
- Object: 해석적 원통 D `0.029 m` / H `0.050 m` / mass `0.02483 kg`, 정립(yaw 0),
  `seed0_S4` base-frame 중심 `[0.4235072423787768, 0.17237803311822986, 0.025] m`,
  지지면 z=0 (t3s_side_sdg2 §3 계약 그대로 → side 후보 pose를 좌표 재유도 없이 verbatim
  소비하기 위함). cylinder 물성/마찰은 g0b_d420 물리 계약 값을 그대로 상속 (재저작 금지).
- Env pins: `numpy==1.26.0`, `psutil==5.9.8` 실행 전후 확인 (D326/D325). 설치 확장
  manifest SHA `5e599aafec0d1c66776c70318535faeffc539e66070d64bf5ca15f6c5e21393a` 재확인.

## 4. Pose set (probe 입력 — 변수 아님, 총 13 pose)

- (a) **side 8**: `g0b_d420/t3s_side_sdg2_candidates.json` canonical 8행 —
  `R_base_link5_proposal` + `geometry_mapped_roarm_targets`의 link5-origin 타깃을
  gripper-root(=link5 프레임) world pose로 verbatim 사용. open aperture는 sdg2 §4
  `gripper_maximum_aperture 0.035` 근거의 기존 open 상태.
- (b) **rim-tilt 5**: `g0b_d420/t3r_n8_tilt_results.json`에서 θ∈{6,15,24,29,35}° 각 θ의
  argmax-bite 행 (φ, q5, depth). pose 재구성 닫힌 형식은 동결
  `g0b_d420/t3r_n8_tilt_script.py.txt`에서 **verbatim import** (n8b/D431 ② 방식, 재유도 0).
- pose별 실행 순서 셔플 금지 — 위 열거 순서 고정. 13 pose 전부 실행이 완주 조건
  (조기 중단 시 실패 marker에 사유 기록).

## 5. Phases + gates (성공/실패 권위)

- Phase `PREGRASP`: q5 = open target (자산 원본 open 값), 60 steps @ dt 1/60 — settle.
- Phase `CLOSE`: q5 target = (a) side 후보: sdg2 후보의 `q5_control`이 null이므로
  D29 폭 대응 닫힘각 + D431 ⑥ 대역(14~22°) 내 최심값 / (b) rim 후보: 해당 n8 행 q5 − 2°
  여유. 120 steps @ dt 1/60.
- **HANG TEST** (manager 밖 자체 단계): 지지면 collider 비활성 → 240 steps → 물체
  z-낙하와 그리퍼-물체 상대 변위 측정.
- Gates: `close_bilateral` = 같은 physics step에서 min(F_fixed, F_moving) > **0.01 N**
  (PhysX contact report; t3y/D441과 동일 기준) / `HOLD` = hang 240 steps 후 물체 낙하
  < **6 mm** (기존 lift gate 준용) / **SUCCESS = close_bilateral AND HOLD**.
- 실패 분류 taxonomy: `NO_JAW_CONTACT` / `ONE_JAW_ONLY_FIXED` / `ONE_JAW_ONLY_MOVING` /
  `BILATERAL_NO_HOLD` / `PRECLOSE_COLLISION`. lift/hang 단계의 순간 양측력은 파지 증거로
  해석 금지 (54th·55th 규칙 유지).

## 6. Outputs (forward-only, `g0b_d444/`만 쓰기 가능)

`fg1_prereg.md`(본 문서) / `fg1_gripper_only.usd` / `fg1_script.py.txt` / `fg1_argv.txt` /
`fg1_results.json` / `fg1_trace.npz` / `fg1_timeline.rrd` / `fg1_timeline.rbl` /
`fg1_rerun_validation.json` / `fg1_inspection.png` / `fg1_stdout.log` /
`fg1_exit_status.txt` / (실패 시) `fg1_failure.json`.

Lifecycle (D442 준수): 모든 결과/RRD를 fsync → pre-close sentinel 기록 →
`SimulationApp.close()`를 마지막 terminal call로. close 이후 내부 marker에 의존 금지.
기존 태그/prefix 편집·삭제·이동 0. 실패 시 같은 태그 재실행 금지, 신규 태그(fg2)로만.

## 7. D341 observability

RRD save-only 기록에 실제 결정 대상 로깅: 그리퍼 두 조·원통·지지면 기하, pose별
접촉력 시계열(fixed/moving/bilateral-min), hang 낙하 스칼라, phase 경계 타임라인.
완료 조건: rerun 0.34.1 핀 + footer `rrd verify` PASS + exact entity/timeline/component
계약 + 고정 blueprint + `.rbl` + headless PNG + **실제 육안 검수 관찰 기록** (스크린샷
생성만으로 "inspected" 보고 금지).
