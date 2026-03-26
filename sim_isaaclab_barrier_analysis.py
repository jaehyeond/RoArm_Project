"""
sim_isaaclab_barrier_analysis.py
[A2 SIM2REAL] Isaac Lab → RoArm-M3 전이 장벽 완전 분석

이 파일은 실행 코드가 아닌 분석 문서입니다.
"sim-to-real gap만 해결하면 된다"는 질문에 대한 정직한 답변.
"""

# =============================================================================
# SECTION 1: Isaac Lab 내장 태스크 라이브러리 (실제 확인)
# =============================================================================

ISAAC_LAB_BUILT_IN_TASKS = {
    # manager_based/manipulation/ 하위 태스크들 (실제 소스 트리 확인)
    "reach": {
        "description": "End-effector를 목표 pose로 이동",
        "supported_robots": ["Franka", "custom URDF (우리가 구현함)"],
        "observation_inputs": [
            "joint_pos_rel",        # relative joint positions
            "joint_vel_rel",        # relative joint velocities
            "pose_command",         # target EE pose (7-dim: pos+quat) — GROUND TRUTH
            "last_action",          # previous action
        ],
        "trainable_in_sim": True,
        "transferable_to_roarm_m3": "CONDITIONAL — see barriers",
    },
    "lift": {
        "description": "테이블 위 오브젝트를 들어올리기",
        "supported_robots": ["Franka (primary)", "OpenArm"],
        "observation_inputs": [
            "joint_pos_rel",
            "joint_vel_rel",
            "object_position_in_robot_root_frame",  # GROUND TRUTH object pose
            "ee_frame_pose",                         # GROUND TRUTH EE pose
        ],
        "trainable_in_sim": True,
        "transferable_to_roarm_m3": "HIGH BARRIER — object pose is ground truth",
    },
    "stack": {
        "description": "큐브를 다른 큐브 위에 쌓기",
        "supported_robots": ["UR10 + Robotiq gripper"],
        "observation_inputs": [
            "SAME AS LIFT + second object pose",
        ],
        "trainable_in_sim": True,
        "transferable_to_roarm_m3": "VERY HIGH BARRIER — 2 object poses ground truth",
    },
    "cabinet": {
        "description": "서랍/문 열고 닫기",
        "supported_robots": ["Franka"],
        "observation_inputs": [
            "joint states",
            "cabinet joint state (GROUND TRUTH)",
        ],
        "trainable_in_sim": True,
        "transferable_to_roarm_m3": "HIGH BARRIER — cabinet articulation state",
    },
    "inhand": {
        "description": "손 안에서 물체 재배치 (dexterous manipulation)",
        "supported_robots": ["Kuka+Allegro hand", "Shadow Hand"],
        "trainable_in_sim": True,
        "transferable_to_roarm_m3": "NOT APPLICABLE — RoArm M3은 dexterous hand 없음",
    },
    "pick_place": {
        "description": "물체를 집어서 목표 위치에 놓기",
        "supported_robots": ["GR1T2 humanoid", "Unitree G1"],
        "trainable_in_sim": True,
        "transferable_to_roarm_m3": "HIGH BARRIER — humanoid용, arm-only 버전 없음",
    },
    "place": {
        "description": "이미 잡은 물체를 목표 위치에 놓기",
        "supported_robots": ["Franka (primary)"],
        "trainable_in_sim": True,
        "transferable_to_roarm_m3": "HIGH BARRIER — see lift barriers",
    },
}

# =============================================================================
# SECTION 2: 전이 장벽 완전 목록 (sim-to-real gap보다 훨씬 많다)
# =============================================================================

TRANSFER_BARRIERS = {

    # -------------------------------------------------------------------------
    # BARRIER GROUP A: 아키텍처/파이프라인 문제 (sim-to-real gap과 무관)
    # -------------------------------------------------------------------------

    "A1_no_conversion_pipeline": {
        "category": "ARCHITECTURE",
        "severity": "CRITICAL",
        "description": "Isaac Lab → LeRobot v3 변환 파이프라인이 존재하지 않음",
        "detail": """
        Isaac Lab RL policy 출력 형식:
          - torch.Tensor (batch x action_dim), dtype=float32
          - normalized, scale=0.5, use_default_offset=True
          - radians (relative delta)

        RoArm-M3 SDK 입력 형식:
          - arm.joints_angle_ctrl(angles=[...], speed=500, acc=200)
          - degrees (absolute)
          - 6개 관절 (gripper 포함)

        필요한 변환:
          relative_rad_delta → absolute_deg = (current_deg) + (action * 0.5 * 180/π)

        이 코드가 현재 존재하지 않음. 1-2일 작업.
        """,
        "is_sim_to_real_gap": False,
        "estimated_fix_time": "1-2 days",
    },

    "A2_action_space_mismatch": {
        "category": "ARCHITECTURE",
        "severity": "CRITICAL",
        "description": "Isaac Lab은 5관절 제어, RoArm은 6관절 (gripper 포함)",
        "detail": """
        Isaac Lab reach 구현:
          joint_names = ["base_link_to_link1", "link1_to_link2",
                         "link2_to_link3", "link3_to_link4", "link4_to_link5"]
          → 5개. gripper_joint 없음.

        실제 RoArm-M3:
          6개 관절 (0=base, 1=shoulder, 2=elbow, 3=wrist_pitch, 4=wrist_roll, 5=gripper)

        Pick-and-place에는 gripper를 따로 제어하는 RL policy 필요.
        현재 reach task는 gripper 없음 → 집기 불가.
        """,
        "is_sim_to_real_gap": False,
        "estimated_fix_time": "2-4 hours (add gripper joint to task)",
    },

    "A3_control_frequency_unsynchronized": {
        "category": "ARCHITECTURE",
        "severity": "HIGH",
        "description": "Sim 30Hz vs 실제 루프 빈도 미정의",
        "detail": """
        Isaac Lab 설정:
          sim.dt = 1/60 sec
          decimation = 2
          → policy_frequency = 30Hz

        실제 RoArm 배포:
          USB serial 지연: ~20-50ms (가변)
          현재 deploy loop: rate control 없음

        30Hz 유지 실패 시 policy가 OOD 상태로 진입.
        (policy는 자신이 30Hz로 실행된다고 가정하고 학습됨)
        """,
        "is_sim_to_real_gap": False,
        "estimated_fix_time": "2-4 hours (add rate limiter in deploy loop)",
    },

    "A4_observation_requires_ground_truth_object_state": {
        "category": "ARCHITECTURE — THE BIGGEST ONE",
        "severity": "CRITICAL",
        "description": "Lift/stack/place task의 observation에 ground truth 물체 위치가 포함됨",
        "detail": """
        lift/mdp/observations.py에서 확인:
          object_position_in_robot_root_frame()
          → env.scene["object"].data.root_pos_w   ← 시뮬레이터가 알고 있는 정확한 위치

        실제 배포 시:
          "object"의 root_pos_w를 어디서 얻나?
          → 정밀 카메라 + pose estimation 파이프라인 필요
          → ArUco 마커: ~5-10mm 오차
          → FoundationPose/BundleSDF: GPU 추론 20-100ms, 실시간 어려움
          → 없으면 policy가 완전히 실패

        이것은 sim-to-real gap이 아니다.
        이것은 완전히 다른 파이프라인 요구사항이다.

        reach task도 마찬가지:
          pose_command = mdp.generated_commands("ee_pose")
          → sim이 내부적으로 생성하는 target EE pose
          → 실제 배포에서 이 target을 어떻게 제공할 것인가?
          → 외부에서 target_xyz를 지정해야 함 (텔레옵? 좌표 하드코딩?)
        """,
        "is_sim_to_real_gap": False,
        "estimated_fix_time": "2-4 weeks (object pose estimation pipeline)",
    },

    # -------------------------------------------------------------------------
    # BARRIER GROUP B: 진짜 sim-to-real gap
    # -------------------------------------------------------------------------

    "B1_physics_gap_actuator_model": {
        "category": "SIM-TO-REAL GAP",
        "severity": "HIGH",
        "description": "Isaac Lab ImplicitActuator vs 실제 서보 모터 동역학",
        "detail": """
        Isaac Lab (우리 roarm_m3.py 확인 필요):
          ImplicitActuatorCfg — PD controller, implicit integration
          stiffness, damping 파라미터 → 기본값 사용 (sysid 없음)

        실제 RoArm-M3:
          TTL 서보 모터 (Feetech STS3215)
          내부 PID + 감속기 → 복잡한 비선형 응답
          USB 직렬 통신 지연: ~10-50ms

        Gap 정량화:
          stiffness error → 추정 불가 (sysid 미수행)
          typical range in literature: 30-50% 응답 오차
        """,
        "is_sim_to_real_gap": True,
        "fix_approach": "System identification (step response test)",
        "estimated_work": "3-5 days",
    },

    "B2_physics_gap_contact_and_friction": {
        "category": "SIM-TO-REAL GAP",
        "severity": "HIGH for grasp tasks, LOW for reach",
        "description": "접촉 물리: Coulomb 마찰 모델 vs 실제 고무/플라스틱 접촉",
        "detail": """
        Isaac Lab 기본값:
          static_friction = 0.5, dynamic_friction = 0.5 (단순 Coulomb)
          contact_offset = 0.005m, rest_offset = 0.0m

        실제 물체:
          골판지 상자: μ_s ≈ 0.4-0.6 (방향 의존적)
          플라스틱 표면: μ_s ≈ 0.2-0.4
          접촉 면적: 그리퍼 형상에 따라 복잡한 분포

        Reach task에서는 이 gap이 거의 영향 없음.
        Pick-and-place에서는 grasp success rate에 직접 영향.

        Domain randomization으로 부분 완화 가능:
          friction: Uniform(0.2, 0.8) — 표준적
          object mass: Uniform(50%, 150%) — 표준적
          But: 실제 그리퍼 deformability는 DR로 불가능
        """,
        "is_sim_to_real_gap": True,
        "fix_approach": "DR + rubber/deformable gripper model",
        "estimated_work": "1-2 weeks",
    },

    "B3_physics_gap_no_domain_randomization": {
        "category": "SIM-TO-REAL GAP",
        "severity": "HIGH",
        "description": "현재 EventCfg에 DR이 거의 없음",
        "detail": """
        현재 구현 (reach_env_cfg.py):
          EventCfg:
            reset_robot_joints: position_range=(0.5, 1.5) — 초기 위치만 randomize

        빠진 DR:
          - mass randomization (MISSING)
          - friction randomization (MISSING)
          - actuator delay randomization (MISSING)
          - observation noise beyond small uniform (partial)
          - camera noise / lighting (MISSING — vision tasks용)

        DR 없이 trained policy는 test-time에서 sim 환경에만 최적화됨.
        """,
        "is_sim_to_real_gap": True,
        "fix_approach": "Add EventTermCfg for mass, friction, delay",
        "estimated_work": "2-3 days",
    },

    "B4_convergence_problem": {
        "category": "TRAINING QUALITY",
        "severity": "CRITICAL",
        "description": "100 iteration = 미수렴. 실제 배포 불가능한 policy",
        "detail": """
        실제 확인된 데이터:
          100 iter 후 position_error = 0.097m (9.7cm)
          RoArm-M3 도달 범위: ~0.5m → 오차가 workspace의 19.4%

        충분한 학습:
          minimum: 1000 iter (~4.3분 at 49K steps/sec)
          권장: 2000 iter (9분)
          Franka lift paper 기준: 5000-10000 iter

        이것도 sim-to-real gap이 아니다. 그냥 미완성 학습.
        """,
        "is_sim_to_real_gap": False,
        "estimated_fix_time": "4-10 minutes of training",
    },

    # -------------------------------------------------------------------------
    # BARRIER GROUP C: 인식 파이프라인 (VLA와 근본적으로 다른 문제)
    # -------------------------------------------------------------------------

    "C1_perception_pipeline_missing": {
        "category": "PERCEPTION",
        "severity": "CRITICAL for task-driven deployment",
        "description": "RL policy에는 카메라 입력이 없음 — target 좌표를 외부에서 주입해야 함",
        "detail": """
        현재 reach task observation:
          joint_pos_rel, joint_vel_rel, pose_command, last_action

        pose_command = 시뮬레이터가 내부적으로 생성한 목표 위치

        실제 배포 시나리오:
          Option 1: 하드코딩된 좌표 → 고정된 물체에만 작동
          Option 2: 텔레옵으로 목표 지정 → 자율성 없음
          Option 3: 카메라 + object detection → 별도 ML 파이프라인

        VLA (SmolVLA)와의 근본적 차이:
          VLA: image → action (e2e)
          RL: pose_command + joint_state → action

        RL이 물체를 "보고" 자율적으로 집으려면 카메라 기반
        object pose estimation을 별도로 구축해야 함.
        이것은 sim-to-real gap이 아니라 아키텍처 결정이다.
        """,
        "is_sim_to_real_gap": False,
        "estimated_fix_time": "2-4 weeks (full pipeline)",
    },

    "C2_robot_specific_urdf_vs_real": {
        "category": "MODEL ACCURACY",
        "severity": "MEDIUM",
        "description": "URDF 물리 파라미터가 실제 로봇과 다를 가능성",
        "detail": """
        roarm_m3.urdf:
          질량, 관성 텐서: 제조사 제공 or 자동 추정 (CAD 기반)
          관절 마찰: URDF 기본값 (보통 0)

        실제 로봇:
          서보 모터 감속기의 마찰 → 비선형
          케이블 routing이 관절 토크에 영향

        정량화: system identification 없이 불가능.
        """,
        "is_sim_to_real_gap": True,
        "fix_approach": "Step response test + sysid script",
        "estimated_work": "3-5 days",
    },
}

# =============================================================================
# SECTION 3: "sim-to-real gap만의 문제인가?" — 정직한 비율 분석
# =============================================================================

HONEST_BARRIER_ANALYSIS = {
    "question": "Is 'just sim-to-real gap' an oversimplification?",

    "answer": "YES. 극도로 단순화된 말이다.",

    "barrier_breakdown": {
        "true_sim_to_real_gap": {
            "percentage": "~30%",
            "items": [
                "actuator dynamics mismatch (B1)",
                "contact/friction model (B2)",
                "domain randomization missing (B3)",
                "URDF accuracy (C2)",
            ],
        },
        "architecture_and_pipeline_issues": {
            "percentage": "~70%",
            "items": [
                "action space conversion 미구현 (A1) — gap 아님, 코딩 문제",
                "control frequency 미동기화 (A3) — gap 아님, 엔지니어링 문제",
                "object pose ground truth 의존 (A4) — gap 아님, 완전 다른 파이프라인",
                "perception pipeline 없음 (C1) — gap 아님, 아키텍처 선택",
                "학습 미수렴 (B4) — gap 아님, 그냥 덜 훈련됨",
            ],
        },
    },

    "the_core_insight": """
    Isaac Lab RL policy의 진짜 문제는 이것이다:

    RL policy는 "목표 좌표 X를 주면 거기 도달한다"는 기계를 만든다.
    VLA는 "이미지를 보고 무엇을 해야 하는지 결정한다"는 기계를 만든다.

    Pick-and-place를 자율적으로 수행하려면:

    RL 방식:
      [카메라] → [object detection] → [pose estimation] → [goal coordinate]
              → [RL policy] → [action conversion] → [30Hz rate-controlled]
              → [RoArm M3]

      이 파이프라인의 각 단계가 별도 ML 모델 + 엔지니어링.

    VLA 방식:
      [카메라] → [SmolVLA] → [action] → [RoArm M3]

    RoArm M3 수준의 consumer arm + limited compute에서는
    VLA가 단순하고 실용적이다.
    RL은 더 높은 성능을 낼 수 있지만 훨씬 더 많은 엔지니어링이 필요하다.
    """,
}

# =============================================================================
# SECTION 4: 작업별 실제 전이 가능성 평가
# =============================================================================

TASK_TRANSFER_FEASIBILITY = {

    "reach (current implementation)": {
        "sim_trainable": "YES — 이미 구현됨, 100iter 미수렴 but 구조는 맞음",
        "real_transfer_possible": "YES, with work",
        "work_required": [
            "학습 완료 (2000+ iter, ~9분)",
            "action conversion 구현 (relative_rad → absolute_deg)",
            "target coordinate 제공 방법 결정 (하드코딩 or 텔레옵)",
            "30Hz rate control 루프 구현",
            "DR 추가 (mass, friction)",
        ],
        "total_work": "3-5 days",
        "success_probability": "MEDIUM (40-60%)",
        "note": "Reach는 가장 단순. 물체 인식 불필요. 목표 좌표만 있으면 됨",
    },

    "lift/pick-and-place": {
        "sim_trainable": "YES — Franka 버전 존재, RoArm M3로 port 가능",
        "real_transfer_possible": "YES, but significant work",
        "work_required": [
            "Isaac Lab lift env → RoArm M3 port (1-2주)",
            "RoArm M3 gripper 모델링 + 접촉 파라미터 sysid",
            "Object pose estimation pipeline 구축 (2-4주)",
            "DR 추가 (friction, mass, object appearance)",
            "충분한 학습 (5000+ iter)",
            "Isaac Lab → LeRobot v3 converter (1-2주)",
        ],
        "total_work": "6-10 weeks",
        "success_probability": "LOW-MEDIUM (20-40%)",
        "note": "물체 pose estimation이 bottleneck. 이게 없으면 policy에 입력 자체가 없음",
    },

    "cabinet_open": {
        "sim_trainable": "YES — Franka 버전 존재",
        "real_transfer_possible": "CONDITIONAL",
        "work_required": [
            "물리적으로 동일한 캐비닛 필요 (sim에서 쓴 것과 같은 USD 모델)",
            "캐비닛 관절 상태 측정 센서 or 카메라 기반 추정",
            "RoArm M3 IK range가 캐비닛 handle에 닿는지 확인",
        ],
        "total_work": "4-8 weeks",
        "success_probability": "LOW (15-30%)",
        "note": "캐비닛 관절 상태 ground truth 의존. 실제에서 측정 어려움",
    },

    "stack": {
        "sim_trainable": "YES — UR10 버전 존재",
        "real_transfer_possible": "VERY DIFFICULT",
        "work_required": [
            "2개 물체의 pose estimation 동시 필요",
            "매우 precise한 grasp 필요 (sim grasp 성공 ≠ real grasp 성공)",
            "큐브 사이 안정적 접촉 물리 = sim-to-real gap 최대",
        ],
        "total_work": "3-6 months",
        "success_probability": "VERY LOW (5-15%)",
        "note": "접촉 rich task는 sim-to-real gap이 가장 크게 나타나는 영역",
    },
}

# =============================================================================
# SECTION 5: 권장사항
# =============================================================================

RECOMMENDATIONS = {
    "for_current_stage": """
    Stage 1에서 Isaac Lab RL을 지금 추구하지 말 것.
    이유:
    1. VLA(SmolVLA)가 이미 e2e 파이프라인을 제공함 — 추가 perception 파이프라인 불필요
    2. Reach task 전이에도 3-5일 엔지니어링 필요 (Stage 1 목표인 150ep 수집보다 부가가치 낮음)
    3. Pick-and-place는 6-10주 — Stage 2 이후

    Isaac Lab의 올바른 역할 (Stage 2+):
    - Sim에서 대량 trajectory 생성 → VLA fine-tune data로 활용
    - RL trajectory를 behavioral cloning 데이터로 변환
    - 위험한 동작(충돌 등) sim에서 먼저 테스트
    """,

    "if_you_must_try_reach_now": """
    Reach task만 시도한다면 최소 작업:
    1. train.py로 2000+ iter 학습 (~9분)
    2. deploy script에 action conversion 추가:
       real_deg = current_deg + (sim_action * 0.5 * 180 / π)
    3. 30Hz rate control loop 추가
    4. target coordinate를 하드코딩 (특정 물체 위치)
    총 소요: 1-2일
    성공 확률: 낮음 (DR 없음, sysid 없음)
    """,

    "sim_to_real_gap_is_30_percent_of_the_problem": """
    핵심 메시지:
    "sim-to-real gap만 해결하면 된다"는 거짓말이다.

    실제 장벽의 70%는 sim-to-real gap과 무관한 파이프라인/아키텍처 문제다:
    - action space 변환 코드가 없음
    - object pose estimation이 없음
    - 학습이 미완성
    - control frequency 동기화가 없음

    이런 기계적인 문제들을 다 해결한 후에야
    "남은 gap이 얼마인가"를 논할 수 있다.
    지금은 그 단계가 아니다.
    """,
}


if __name__ == "__main__":
    print("[A2 SIM2REAL] Isaac Lab barrier analysis loaded.")
    print(f"Tasks analyzed: {list(TASK_TRANSFER_FEASIBILITY.keys())}")
    print(f"Barriers identified: {len(TRANSFER_BARRIERS)}")
    print()
    print("KEY FINDING:")
    print(RECOMMENDATIONS["sim_to_real_gap_is_30_percent_of_the_problem"])
