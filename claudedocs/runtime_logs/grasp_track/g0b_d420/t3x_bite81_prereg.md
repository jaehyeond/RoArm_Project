# t3x_bite81 preregistration — IK-conditioned bite window, D29×H50

## 1. 질문과 범위

이번 case의 신규 변수: `[tool-axis tilt theta extension, workspace radius/pose]`.

53rd의 `t3w_reach1`은 같은 top-centre grasp point에서 위치에 따라 plan-feasibility
boundary가 달라진다는 것을 보였다. 그러나 기존 `t3r_n10_ctq5`의 물림 창은
theta ≤ 35 deg까지만 측정됐고, `max(fixed jaw, moving jaw) > 0`인 **한쪽 조
admission**이었다. 이 실행은 다음 반증 가능한 질문을 측정한다.

> D29×H50 원통, frozen attempt3 collision asset, top-centre grasp, radial approach,
> phi=0 조건에서 theta > 35 deg까지 확장했을 때, 실제 p10 IK/FK 자세 중
> 지지면과 유한 원통을 침범하지 않으면서 양쪽 조가 모두 원통 옆면 아래로
> 들어가는 q5 구간이 하나라도 존재하는가?

`q5`는 기존 n10의 측정축을 재사용하며 신규 변수로 세지 않는다. 물체 크기,
질량, 파지점, collision asset은 변경하지 않는다. Isaac/PhysX는 이 사전측정
다음 단계이며, 이 스크립트 자체는 CPU 기하·기구학만 실행한다.

## 2. 입력과 동결 조건

- `p10_g0b_t3t_cyld29h50_tilted_close_sweep_grasp_probe.py`: IK/FK와 pose planner
- `p12_g0b_t3w_reach_boundary_radius_azimuth_sweep.py` 및
  `t3w_reach1_{results.json,grid.npz}`: 위치별 plan-feasibility 근거
- `t3r_n10_ctq5_results.json`: theta=29/35 회귀 기준
- attempt3 root/physics USD: PhysX가 소비하는 frozen collision asset
- D29×H50, support plane `z=0`, top-centre grasp, `dpsi=0`, `phi_target=0`
- `numpy==1.26.0`, `psutil==5.9.8`, `rerun-sdk==0.34.1`

모든 입력은 SHA-256으로 gate한다. 기존 파일은 읽기 전용이고 재분해·수정하지
않는다. 신규 산출물 prefix는 오직 `t3x_bite81_*`이며 하나라도 존재하면 exit 3이다.
실행 시작 시 p13 소스의 바이트와 SHA-256을 잡고, 종료 직전 같은 파일을 다시
읽어 동일성을 확인한다. 동결본은 종료 시점 파일이 아니라 시작 바이트를 쓰며,
실행 중 소스가 달라졌으면 과학 결과와 무관하게 `SOURCE_DRIFT_INVALID`, exit 3이다.

## 3. 측정 절차

1. attempt3 USD에서 enabled convexHull을 읽고 `link5=64`, `gripper_link=64`,
   legacy collider 전부 disabled, non-convexHull 0을 확인한다.
2. collision hull 표면을 기존 0.5 mm pitch로 읽고 theta controls
   `{6,15,24,29,35}`와 extension `{36,...,81}`에서 q5를 coarse 0.5 deg로
   훑은 뒤 창 경계/최댓값 주변을 0.1 deg로 재측정한다.
3. 기존 n10의 theta=29/35 collision 결과(최댓값, q5*, 창 경계,
   fixed/moving bite)를 재현하지 못하면 새 주장을 내지 않는다.
4. 각 q5에서 다음을 별도로 기록한다.
   - unilateral bite = `max(bite_fixed, bite_moving)`
   - bilateral bite = `min(bite_fixed, bite_moving)`; 둘 중 하나라도 없으면 없음
   - positive window는 위 두 정의로 각각 따로 계산한다.
5. 실제 pose는 `seed0_S1..S4`, `r=0.45`, `r=0.525` 대조군에서 p10 planner로
   계산한다. 모든 theta를 순서대로 평가하고 마지막 feasible 자세를 선택한다.
   approach/descend/lift 각각 position ≤3 mm, axis residual ≤5 deg가 필요하며,
   wrist-roll은 approach/descend/lift 세 phase 모두 별도로 v6
   `[-90,+90] deg`를 통과해야 한다. 각 phase 값과 실패 phase를 기록한다.
6. 선택 자세의 실제 FK rotation으로 world-down을 link5 frame에 투영해
   `theta_actual`, `phi_actual`을 재도출한다. nominal theta가 아니라 이 방향으로
   collision-hull 물림 창을 다시 측정한다.
7. 창 내부 q5에서 측정된 descent delta로 plan을 다시 만들고, actual FK로 두
   collision body를 world에 놓는다. 정확한 유한 D29×H50 내부 샘플 수와
   support-plane 최소 z를 기록한다. `r=0.525`는 외부 위치·지면 침범을 잡는
   negative control이며 물리 후보가 될 수 없다.

## 4. 차단 gate와 판정

- `X0`: 신규 산출물 0개
- `X1`: 입력 SHA 및 환경 pin 전부 일치
- `X2`: enabled convexHull 64+64, legacy disabled, non-convexHull 0
- `X3`: n10 theta=29/35 수치 회귀 PASS
- `X4`: 각 pose row의 phase IK와 wrist-roll 결과를 명시적으로 기록
- `X5`: 실제 FK 자세의 finite-cylinder penetration count와 table minimum z 기록
- `X6`: 시작/종료 p13 소스 바이트 동일; 다르면 `SOURCE_DRIFT_INVALID`, exit 3

과학 판정:

- `BILATERAL_WINDOW_EXISTS_IN_SPAWN_ENVELOPE`: source envelope 안에서 phase IK,
  wrist-roll, finite-object, table gate를 모두 통과한 bilateral window가 ≥1개
- `NO_BILATERAL_WINDOW_IN_SPAWN_ENVELOPE`: 위 후보 0개

두 판정 모두 다음 PhysX 실행을 막지 않는다. bilateral 후보는 positive stratum,
unilateral-only 창은 명시적 negative-control stratum으로 넘긴다. PhysX가 최종
contact/lift 권위다.

## 5. 주장하지 않는 것

- force closure, lift success, 실제 로봇 성공
- side-face midpoint grasp (`D419` top-centre는 변경하지 않음)
- `g0a_pass=true`, D439 재판정, Arm-F 저작 정당화
- 0.5 mm hull sampling보다 정밀한 continuous collision 보증
- `r=0.525`, theta≈81 deg 가지의 사용 가능성

## 6. 산출물과 관측성

`t3x_bite81_{results.json,grid.npz,curves.csv,timeline.rrd,timeline.rbl,
rerun_validation.json,inspection.png,script.py.txt,argv.txt}`.

Float64 JSON/NPZ가 수치 권위다. RRD는 actual candidate pose의 두 convex-jaw
cloud, finite cylinder, support plane, TCP/tool axis와 판정 scalars를 기록한다.
고정 blueprint, `rrd verify`, exact entity/timeline/component contract, RBL export,
headless screenshot과 실제 육안 검수가 모두 필요하다.
