# RoArm-M3-Pro USD Model Analysis Report

**분석 일자**: 2024-12-31
**USD 파일**: `E:\RoArm_Project\RoARM PRO_M3.usd`
**공식 URDF 소스**: [waveshareteam/roarm_ws_em0](https://github.com/waveshareteam/roarm_ws_em0)

---

## 1. 모델 개요

### 로봇 사양
| 항목 | 값 |
|------|-----|
| 모델명 | RoArm-M3-Pro |
| 제조사 | Waveshare |
| DOF | 4 (USD/URDF 기준) |
| 실제 서보 | 5+1 DOF (ST3235) |
| 총 질량 | ~0.636 kg |

### 링크 구조
```
base_link (0.326 kg)
    └── link1 (0.093 kg)
        └── link2 (0.090 kg)
            └── link3 (0.123 kg)
                └── gripper_link (0.004 kg)
                └── hand_tcp (TCP point, fixed)
```

---

## 2. 조인트 제한 (Joint Limits)

### USD vs URDF 비교

| Joint | USD (degrees) | URDF (radians) | 변환 확인 | 상태 |
|-------|---------------|----------------|-----------|------|
| base_link_to_link1 | ±180.0° | ±3.1416 rad | 180° = π rad | ✅ 일치 |
| link1_to_link2 | ±90.0° | ±1.5708 rad | 90° = π/2 rad | ✅ 일치 |
| link2_to_link3 | -57.3° ~ +180.0° | -1.0 ~ +3.1416 rad | 정확히 일치 | ✅ 일치 |
| link3_to_gripper_link | 0° ~ +85.9° | 0 ~ +1.5 rad | 85.9° ≈ 1.5 rad | ✅ 일치 |

**참고**: USD는 Degrees 단위, URDF는 Radians 단위를 사용합니다. 실제 값은 동일합니다.

### 조인트 기능 설명
| Joint | 기능 | 범위 |
|-------|------|------|
| base_link_to_link1 | Base 회전 (Yaw) | 360° 전방향 |
| link1_to_link2 | Shoulder (어깨) | ±90° |
| link2_to_link3 | Elbow (팔꿈치) | -57° ~ +180° |
| link3_to_gripper_link | Gripper (그리퍼) | 0° ~ 86° |

---

## 3. Drive API 설정 (USD 내장값)

### 현재 USD 설정값
| Joint | Stiffness | Damping | MaxForce |
|-------|-----------|---------|----------|
| base_link_to_link1 | 2.69 | 0.001 | 3.4e+38 (무제한) |
| link1_to_link2 | 3.63 | 0.001 | 3.4e+38 (무제한) |
| link2_to_link3 | 5.74 | 0.002 | 3.4e+38 (무제한) |
| link3_to_gripper_link | 0.19 | 0.00008 | 3.4e+38 (무제한) |

### 문제점
- **Stiffness가 매우 낮음**: 로봇이 흐물흐물하게 움직임
- **Damping이 매우 낮음**: 진동 발생 가능
- **MaxForce가 무제한**: 실제 서보 토크와 불일치

### 권장 설정 (Isaac Lab ActuatorCfg에서 오버라이드)
```python
# roarm_cfg.py에서 설정
actuators={
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["base_link_to_link1", "link1_to_link2", "link2_to_link3"],
        effort_limit_sim=100.0,    # 또는 실제 서보 기준 2.94 N·m
        velocity_limit_sim=100.0,
        stiffness=1000.0,          # USD 값 오버라이드
        damping=100.0,             # USD 값 오버라이드
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["link3_to_gripper_link"],
        effort_limit_sim=50.0,
        velocity_limit_sim=100.0,
        stiffness=500.0,
        damping=50.0,
    ),
}
```

---

## 4. 물리 속성 (Mass/Inertia)

### 질량 비교
| Link | USD Mass (kg) | URDF Mass (kg) | 상태 |
|------|---------------|----------------|------|
| base_link | 0.3264 | 0.326 | ✅ 일치 |
| link1 | 0.0929 | 0.093 | ✅ 일치 |
| link2 | 0.0896 | 0.090 | ✅ 일치 |
| link3 | 0.1230 | 0.123 | ✅ 일치 |
| gripper_link | 0.0037 | 0.004 | ✅ 일치 |
| **총합** | **0.636** | **0.636** | ✅ |

### 관성 텐서 (Inertia)
URDF에 정의된 관성 텐서가 USD로 변환됨. 각 링크의 관성 중심(CoM)도 올바르게 설정됨.

---

## 5. 실제 서보 사양 (ST3235)

### Waveshare ST3235 Serial Bus Servo
| 항목 | 값 |
|------|-----|
| 토크 | 30 kg·cm @ 12V (≈ 2.94 N·m) |
| 해상도 | 12-bit (4096 steps) |
| 정밀도 | 0.088° |
| 회전 범위 | 360° (무제한 모드 가능) |
| 감속비 | 1:345 |
| 전압 | 6-12.6V |
| 케이스 | 알루미늄 합금 (T6061) |

### Sim2Real 파라미터 매핑
| 시뮬레이션 파라미터 | 실제 서보 값 | 권장 설정 |
|---------------------|-------------|-----------|
| effort_limit_sim | 30 kg·cm = 2.94 N·m | 2.94 ~ 100.0 |
| velocity_limit_sim | ~0.73 RPM max | 0.73 ~ 100.0 |
| stiffness | 서보 PID 기반 | 1000 ~ 10000 |
| damping | 서보 PID 기반 | 100 ~ 1000 |

---

## 6. USD 모델 vs 실제 로봇 차이점

### DOF 차이
| 구분 | DOF | 설명 |
|------|-----|------|
| 실제 로봇 | 5+1 (6개 서보) | Base, Shoulder, Elbow, Wrist Pitch, Wrist Roll, Gripper |
| USD/URDF 모델 | 4 | Base, Shoulder, Elbow, Gripper (Wrist 단순화) |

### 단순화된 부분
1. **Wrist Pitch (J4)**: 모델에 없음 - link3에 통합
2. **Wrist Roll (J5)**: 모델에 없음 - link3에 통합
3. **Gripper 내부 구조**: 2개 핑거 → 단일 gripper_link로 단순화

### 영향
- Reach 태스크: 영향 없음 (End-effector 위치 도달만 필요)
- Pick-and-Place: 그리퍼 열림/닫힘으로 단순화 가능
- 정밀 매니퓰레이션: Wrist DOF 추가 필요할 수 있음

---

## 7. Isaac Lab 설정 가이드

### 현재 설정 파일
- **로봇 설정**: `isaaclab_roarm/roarm_cfg.py`
- **환경 설정**: `isaaclab_roarm/roarm_reach_env_cfg.py`
- **학습 설정**: `isaaclab_roarm/agents/rsl_rl_ppo_cfg.py`

### ArticulationCfg 우선순위
```
1. USD 파일 (기본값)
   └── Joint Limits: ✅ 올바르게 설정됨
   └── Mass/Inertia: ✅ 올바르게 설정됨
   └── Stiffness/Damping: ⚠️ 낮음 (오버라이드 필요)

2. ArticulationCfg (Python)
   └── ActuatorCfg에서 Stiffness/Damping 오버라이드
   └── effort_limit_sim, velocity_limit_sim 설정

3. 최종 적용값
   └── USD 값 + Python 오버라이드 = 시뮬레이션 값
```

### 조인트 제한 자동 적용
Isaac Lab은 USD에서 조인트 제한을 자동으로 읽어옵니다:
- `articulation.data.joint_pos_limits` - 위치 제한
- `articulation.data.joint_vel_limits` - 속도 제한

별도 설정 없이 USD의 조인트 제한이 시뮬레이션에 적용됩니다.

---

## 8. 검증 스크립트

### USD 분석 스크립트
```bash
C:\isaac-sim\python.bat E:\RoArm_Project\check_usd_joint_limits.py
```

### 출력 예시
```
[1] Revolute Joint 분석
Joint Name                     Lower (rad)     Upper (rad)
base_link_to_link1             -180.0004       180.0004
link1_to_link2                 -90.0002        90.0002
link2_to_link3                 -57.2958        180.0004
link3_to_gripper_link          0.0000          85.9437

[4] 물리 속성 (Mass, Inertia)
base_link: mass = 0.3264 kg
link1: mass = 0.0929 kg
link2: mass = 0.0896 kg
link3: mass = 0.1230 kg
gripper_link: mass = 0.0037 kg
```

---

## 9. 결론 및 권장사항

### 현재 상태
| 항목 | 상태 | 조치 필요 |
|------|------|----------|
| 조인트 구조 | ✅ 정상 | 없음 |
| 조인트 제한 | ✅ 정상 | 없음 |
| 질량/관성 | ✅ 정상 | 없음 |
| Stiffness/Damping | ⚠️ USD 값 낮음 | roarm_cfg.py에서 오버라이드 (완료) |
| Effort/Velocity Limits | ⚠️ USD 무제한 | roarm_cfg.py에서 설정 (완료) |

### 권장 다음 단계
1. ✅ **현재 설정으로 학습 테스트 진행** - USD 재변환 불필요
2. 학습 결과 확인 후 Stiffness/Damping 튜닝
3. 실제 로봇 Sim2Real 테스트 시 파라미터 미세 조정

### 참고 자료
- [Waveshare RoArm-M3 Wiki](https://www.waveshare.com/wiki/RoArm-M3)
- [ST3235 Servo Specifications](https://www.waveshare.com/st3235-servo.htm)
- [Isaac Lab - Writing an Asset Configuration](https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html)
- [Isaac Lab - Importing a New Asset](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html)

---

*이 문서는 check_usd_joint_limits.py 스크립트 실행 결과와 공식 URDF 분석을 기반으로 작성되었습니다.*
